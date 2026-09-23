# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "h5py",
#   "numpy",
#   "obstore",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "remfile",
#   "s3fs",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Runnable direct h5py implementation for the packaged NWB benchmark."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote

import h5py
import numpy as np
import remfile

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_DEFAULT_BACKEND = "remfile"
_DEFAULT_BENCHMARK = "dynamic_routing_nwb_hdf5_v0"
_DEFAULT_IMPLEMENTATION_ID = "direct_h5py"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing direct h5py/%s access for %d NWB files.",
        _backend(),
        len(context.benchmark.data_sources),
    )
    _quiet_storage_debug_loggers()


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local direct h5py disk cache to clear for %d NWB paths.",
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
                answer = _longest_isi_for_fastest_visp_unit(
                    context.benchmark.data_sources,
                )
            case "multisession_table_query":
                answer = _multisession_table_query(context.benchmark.data_sources)
            case "large_array":
                answer = _large_array(context.benchmark.data_sources)
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Direct h5py benchmark teardown for %d NWB paths.",
        len(context.benchmark.data_sources),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "requests", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@contextmanager
def _open_nwb(nwb_path: str) -> Iterator[h5py.File]:
    """Open one remote NWB HDF5 file read-only through the requested backend."""
    backend = _backend()
    logger.debug("Opening NWB HDF5 file %s.", nwb_path)
    if backend == "remfile":
        file_obj = remfile.File(_to_https_url(nwb_path))
        try:
            with h5py.File(file_obj, mode="r") as nwb_file:
                yield nwb_file
        finally:
            file_obj.close()
    elif backend == "s3fs":
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(nwb_path, mode="rb") as file_obj:
            with h5py.File(file_obj, mode="r") as nwb_file:
                yield nwb_file
    elif backend == "ros":
        if not h5py.get_config().ros3:
            raise RuntimeError("This h5py build does not include the ROS3 driver.")
        with h5py.File(_to_https_url(nwb_path).encode(), mode="r", driver="ros3") as nwb_file:
            yield nwb_file
    elif backend == "obstore":
        from obstore import fsspec as obstore_fsspec

        bucket, key = _split_s3_uri(nwb_path)
        fs = obstore_fsspec.FsspecStore(
            "s3",
            config={"region": os.environ.get("AWS_REGION", "us-west-2")},
            skip_signature=True,
        )
        with fs.open(f"{bucket}/{key}", mode="rb") as file_obj:
            with h5py.File(file_obj, mode="r") as nwb_file:
                yield nwb_file
    else:
        raise ValueError(f"Unsupported direct h5py backend: {backend}")


def _backend() -> str:
    """Return the requested direct HDF5 object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _to_https_url(nwb_path: str) -> str:
    """Convert a public S3 URI to the HTTPS URL expected by remfile."""
    if nwb_path.startswith("https://") or nwb_path.startswith("http://"):
        return nwb_path
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return f"https://{bucket}.s3.amazonaws.com/{quote(key)}"


def _split_s3_uri(nwb_path: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and key components."""
    if not nwb_path.startswith("s3://"):
        raise ValueError(f"Unsupported remote NWB path: {nwb_path}")
    bucket, key = nwb_path.removeprefix("s3://").split("/", maxsplit=1)
    return bucket, key


def _count_visp_default_qc(data_sources: list[str]) -> int:
    """Count VISp units passing default QC across all NWB files."""
    count = 0
    for nwb_path in data_sources:
        with _open_nwb(nwb_path) as nwb_file:
            units = nwb_file["units"]
            structure = _read_string_array(units["structure"])
            default_qc = np.asarray(units["default_qc"][:], dtype=np.bool_)
            count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(data_sources: list[str]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_path: str | None = None
    top_row = -1
    top_firing_rate = -np.inf

    for nwb_path in data_sources:
        with _open_nwb(nwb_path) as nwb_file:
            units = nwb_file["units"]
            structure = _read_string_array(units["structure"])
            firing_rate = np.asarray(units["firing_rate"][:], dtype=np.float64)
            candidate_rows = np.flatnonzero(
                (structure == "VISp") & ~np.isnan(firing_rate),
            )
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
    with _open_nwb(top_path) as nwb_file:
        spike_times = _get_unit_spike_times(nwb_file["units"], top_row)
    return float(np.diff(spike_times).max())


def _get_unit_spike_times(units: h5py.Group, unit_index: int) -> np.ndarray:
    """Return spike times for one unit from an HDF5-backed Units table."""
    spike_times_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
    start = 0 if unit_index == 0 else int(spike_times_index[unit_index - 1])
    stop = int(spike_times_index[unit_index])
    return np.asarray(units["spike_times"][start:stop], dtype=np.float64)


def _multisession_table_query(data_sources: list[str]) -> float:
    """Compute the mean trial duration across all NWB files."""
    total_duration = 0.0
    total_trials = 0
    for nwb_path in data_sources:
        with _open_nwb(nwb_path) as nwb_file:
            trials = nwb_file["intervals"]["trials"]
            start_time = np.asarray(trials["start_time"][:], dtype=np.float64)
            stop_time = np.asarray(trials["stop_time"][:], dtype=np.float64)
            total_duration += float(np.sum(stop_time - start_time))
            total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _read_string_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read an HDF5 string dataset as a NumPy array of Python strings."""
    values = dataset.asstr()[:]
    return np.asarray(values, dtype=str)


def _large_array(data_sources: list[str]) -> float:
    """Return the mean of a 6.6 MB facemap data block from the first NWB file."""
    if not data_sources:
        raise ValueError("At least one NWB path is required.")
    with _open_nwb(data_sources[0]) as nwb_file:
        facemap_data = nwb_file["processing"]["behavior"]["facemap_side_camera"]["data"]
        data = np.asarray(
            facemap_data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
            dtype=np.float32,
        )
    return float(np.mean(data, dtype=np.float64))


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}",
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
