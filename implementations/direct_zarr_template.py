# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "numpy",
#   "obstore",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "remfile",
#   "s3fs",
#   "zarr",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { path = "..", editable = true }
# ///

"""Runnable direct Zarr implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, MutableMapping
from typing import Any

import numpy as np
import zarr

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_DEFAULT_BACKEND = "s3fs"
_DEFAULT_BENCHMARK = "dynamic_routing_nwb_zarr_v0"
_DEFAULT_IMPLEMENTATION_ID = "direct_zarr"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128
_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing direct Zarr/%s access for %d NWB stores.",
        _backend(),
        len(context.benchmark.data_sources),
    )
    os.environ.setdefault("AWS_REGION", "us-west-2")
    _quiet_storage_debug_loggers()


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local direct Zarr cache to clear for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(context.benchmark.data_sources)
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(context.benchmark.data_sources)
            case "multisession_table_query":
                answer = _multisession_table_query(context.benchmark.data_sources)
            case "large_array":
                answer = _large_array(context.benchmark.data_sources)
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(context.benchmark.data_sources)
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(context.benchmark.data_sources[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(context.benchmark.data_sources)
            case "behavior_large_array":
                answer = _running_speed_block_mean(context.benchmark.data_sources[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Direct Zarr benchmark teardown for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _backend() -> str:
    """Return the requested direct Zarr object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _open_store(nwb_path: str) -> Any:
    """Open one remote NWB Zarr store as a read-only Zarr group."""
    backend = _backend()
    logger.debug("Opening NWB Zarr store %s through %s.", nwb_path, backend)
    if backend == "s3fs":
        if _is_zarr_v3():
            return zarr.open_group(
                nwb_path,
                mode="r",
                storage_options={"anon": True},
                use_consolidated=False,
            )
        return zarr.open(nwb_path, mode="r", storage_options={"anon": True})
    if backend == "obstore":
        from obstore import fsspec as obstore_fsspec

        os.environ.setdefault("AWS_SKIP_SIGNATURE", "true")
        if _is_zarr_v3():
            obstore_fsspec.register("s3")
            return zarr.open_group(nwb_path, mode="r", use_consolidated=False)
        bucket, key = _split_s3_uri(nwb_path)
        fs = obstore_fsspec.FsspecStore("s3", config={"skip_signature": True})
        return zarr.open(_ObstoreZarrV2Store(fs, f"{bucket}/{key}"), mode="r")
    if backend in {"remfile", "ros"}:
        raise RuntimeError(f"{backend} is a file backend and cannot open directory Zarr stores.")
    raise ValueError(f"Unsupported direct Zarr backend: {backend}")


def _is_zarr_v3() -> bool:
    """Return whether the active zarr-python runtime is major version 3."""
    return zarr.__version__.split(".", maxsplit=1)[0] == "3"


def _split_s3_uri(nwb_path: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key components."""
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return bucket, key


class _ObstoreZarrV2Store(MutableMapping[str, bytes]):
    """Read-only Zarr v2 mapping backed by obstore's fsspec adapter."""

    def __init__(self, fs: Any, root_path: str) -> None:
        """Create a store rooted at a bucket-relative object prefix."""
        self._fs = fs
        self._root_path = root_path.rstrip("/")

    def __getitem__(self, key: str) -> bytes:
        """Return one Zarr metadata or chunk object."""
        try:
            return bytes(self._fs.cat_file(self._path_for(key)))
        except Exception as exc:
            if isinstance(exc, self._missing_exceptions()):
                raise KeyError(key) from exc
            raise

    def __setitem__(self, key: str, value: bytes) -> None:
        """Reject writes because benchmark stores are read-only."""
        raise TypeError(f"{type(self).__name__} is read-only")

    def __delitem__(self, key: str) -> None:
        """Reject deletes because benchmark stores are read-only."""
        raise TypeError(f"{type(self).__name__} is read-only")

    def __iter__(self) -> Iterator[str]:
        """Yield Zarr object keys relative to the store root."""
        prefix = f"{self._root_path}/"
        for path in self._fs.find(self._root_path):
            yield path.removeprefix(prefix)

    def __len__(self) -> int:
        """Return the number of objects below the store root."""
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        """Return whether a Zarr key exists in the store."""
        if not isinstance(key, str):
            return False
        try:
            return bool(self._fs.exists(self._path_for(key)))
        except Exception:
            return False

    def _path_for(self, key: str) -> str:
        """Return the bucket-relative object path for a Zarr key."""
        stripped = key.lstrip("/")
        if not stripped:
            return self._root_path
        return f"{self._root_path}/{stripped}"

    def _missing_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return filesystem exceptions that should behave like missing keys."""
        missing = getattr(self._fs, "missing_exceptions", (FileNotFoundError,))
        if isinstance(missing, tuple):
            return missing
        return tuple(missing)


def _count_visp_default_qc(data_sources: list[str]) -> int:
    """Count VISp units passing default QC across all NWB stores."""
    count = 0
    for nwb_path in data_sources:
        units = _open_store(nwb_path)["units"]
        structure = np.asarray(units["structure"][:])
        default_qc = np.asarray(units["default_qc"][:], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(data_sources: list[str]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_path: str | None = None
    top_row = -1
    top_firing_rate = -np.inf

    for nwb_path in data_sources:
        units = _open_store(nwb_path)["units"]
        structure = np.asarray(units["structure"][:])
        firing_rate = np.asarray(units["firing_rate"][:], dtype=np.float64)
        candidate_rows = np.flatnonzero((structure == "VISp") & ~np.isnan(firing_rate))
        if candidate_rows.size == 0:
            logger.debug("No VISp units with firing_rate in %s.", nwb_path)
            continue
        local_row = int(candidate_rows[np.argmax(firing_rate[candidate_rows])])
        local_rate = float(firing_rate[local_row])
        if local_rate > top_firing_rate:
            top_path = nwb_path
            top_row = local_row
            top_firing_rate = local_rate

    if top_path is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")

    logger.debug(
        "Fetching spike_times for fastest VISp unit in %s at row %d.",
        top_path,
        top_row,
    )
    units = _open_store(top_path)["units"]
    spike_times = _get_unit_spike_times(units, top_row)
    return float(np.diff(spike_times).max())


def _get_unit_spike_times(units: Any, unit_index: int) -> np.ndarray:
    """Return spike times for one unit from a Zarr-backed Units table."""
    spike_times_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return np.asarray(units["spike_times"][start:stop], dtype=np.float64)

def _string_array(values: Any) -> np.ndarray:
    """Convert a table column to a NumPy string array."""
    raw_values = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_values
        ],
        dtype=str,
    )


def _multisession_table_query(data_sources: list[str]) -> float:
    """Compute the mean trial duration across all NWB stores."""
    total_duration = 0.0
    total_trials = 0
    for nwb_path in data_sources:
        trials = _open_store(nwb_path)["intervals"]["trials"]
        start_time = np.asarray(trials["start_time"][:], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"][:], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _large_array(data_sources: list[str]) -> float:
    """Return the mean of a 6.6 MB facemap data block from the first NWB store."""
    if not data_sources:
        raise ValueError("At least one NWB path is required.")
    facemap = _open_store(data_sources[0])["processing"]["behavior"]["facemap_side_camera"]
    data = np.asarray(
        facemap["data"][:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))


def _multisession_trials_hit_rate(data_sources: list[str]) -> float:
    """Fraction of `intervals/trials` rows that were hits across all sessions."""
    total_trials = 0
    total_hits = 0
    for nwb_path in data_sources:
        trials = _open_store(nwb_path)["intervals"]["trials"]
        all_hits = np.asarray(trials["hit"][:], dtype=bool)
        hit_only = all_hits[all_hits]
        total_hits += hit_only.size
        total_trials += all_hits.size

    return float(total_hits / total_trials)


def _max_speed_stimulus(first_session_path: str) -> str:
    """Stimulus with the highest mean running speed in the first session."""
    store = _open_store(first_session_path)
    stim = store["intervals"]["stimulus_presentations"]
    image_name = np.asarray(stim["image_name"][:])
    start_time = np.asarray(stim["start_time"][:], dtype=np.float64)
    stop_time = np.asarray(stim["stop_time"][:], dtype=np.float64)

    speed_group = store["processing"]["running"]["speed"]
    speed_data = np.asarray(speed_group["data"][:], dtype=np.float64)
    speed_ts = np.asarray(speed_group["timestamps"][:], dtype=np.float64)

    # Assign each speed sample to the presentation whose start_time most
    # recently precedes it (asof match), then keep only samples that also lie
    # before that presentation's stop_time.
    order = np.argsort(start_time)
    sorted_starts = start_time[order]
    sorted_stops = stop_time[order]
    sorted_labels = image_name[order]

    candidate = np.searchsorted(sorted_starts, speed_ts, side="right") - 1
    within = candidate >= 0
    clipped = np.where(within, candidate, 0)
    within &= speed_ts <= sorted_stops[clipped]

    matched_labels = sorted_labels[clipped]

    best_label = ""
    best_mean = -np.inf
    for label in np.unique(matched_labels[within]):
        mask = within & (matched_labels == label)
        mean_speed = float(speed_data[mask].mean())
        if mean_speed > best_mean:
            best_mean = mean_speed
            best_label = str(label)
    return best_label


def _multisession_lick_rate_average(data_sources: list[str]) -> float:
    """Highest per-session mean lick rate (licks / second) across sessions."""
    best_rate = -np.inf
    for nwb_path in data_sources:
        events = _open_store(nwb_path)["events"]["events"]
        event_type = _string_array(events["event_type"][:])
        timestamps = np.asarray(events["timestamp"][:], dtype=np.float64)
        n_licks = int(np.count_nonzero(event_type == "lick"))
        duration = float(timestamps.max() - timestamps.min())

        rate = n_licks / duration
        if rate > best_rate:
            best_rate = rate
            
    return float(best_rate)


def _running_speed_block_mean(first_session_path: str) -> float:
    """Mean of the first 20,000 samples of `processing/running/speed/data`."""
    store = _open_store(first_session_path)
    data = np.asarray(
        store["processing"]["running"]["speed"]["data"][
            :_RUNNING_SPEED_DOWNLOAD_SAMPLES
        ],
        dtype=np.float64,
    )
    return float(np.mean(data, dtype=np.float64))


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}_zarr{zarr.__version__.split('.')[0]}",
        ),
        implementation_nwb_interface=None,
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
