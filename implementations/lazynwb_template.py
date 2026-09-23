# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "lazynwb",
#   "numpy",
#   "polars",
#   "psutil",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Version-agnostic lazynwb implementation for packaged NWB benchmarks."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any

import lazynwb
import numpy as np
import polars as pl

import neurodatabench

logger = neurodatabench.get_logger(__name__)

state: dict[str, Any] = {}

_DEFAULT_BACKEND = "obstore"
_DEFAULT_BENCHMARK = "dynamic_routing_nwb_hdf5_v0"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128
_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources),
    )
    cache_path = _set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources))
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", "us-west-2")

    lazynwb.config.anon = True
    _configure_backend(_backend())
    state.clear()

    state["trials"] = lazynwb.scan_nwb(
        context.benchmark.data_sources,
        "/intervals/trials",
        disable_progress=True,
    )

def get_units(context: neurodatabench.RunContext) -> pl.DataFrame:
    """Retrieve the units table from the lazynwb catalog."""
    if "units" not in state:
        state["units"] = lazynwb.scan_nwb(
            context.benchmark.data_sources,
            "/units",
            disable_progress=True,
            infer_schema_length=1,
        )
    return state["units"]


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                units = get_units(context)
                answer = int(
                    units
                    .filter(
                        pl.col("structure").eq("VISp"),
                        pl.col("default_qc"),
                    )
                    .select(pl.len().alias("count"))
                    .collect()
                    .item()
                )
            case "predicated_spike_times":
                units = get_units(context)
                answer = _longest_isi_for_fastest_visp_unit(units)
            case "multisession_table_query":
                answer = float(
                    state["trials"]
                    .select(
                        (pl.col("stop_time") - pl.col("start_time")).mean().alias("mean_length")
                    )
                    .collect()
                    .item()
                )
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(state["trials"])
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(context.benchmark.data_sources[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(context.benchmark.data_sources)
            case "change_detection_large_array":
                answer = _change_detection_large_array(context.benchmark.data_sources[0])
            case "large_array":
                state["facemap_side_camera"] = lazynwb.get_timeseries(
                    context.benchmark.data_sources[0],
                    "/processing/behavior/facemap_side_camera",
                    exact_path=True,
                )
            
                data = np.asarray(
                    state["facemap_side_camera"].data[
                        :_FACEMAP_DOWNLOAD_ROWS,
                        :_FACEMAP_DOWNLOAD_COLUMNS,
                    ],
                    dtype=np.float32,
                )
                answer = float(
                    np.mean(data, dtype=np.float64),
                )
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources))
    state.clear()


def _longest_isi_for_fastest_visp_unit(units: pl.LazyFrame) -> float:
    """Return the longest ISI while reading spikes only for the selected unit."""
    candidates = (
        units.filter(
            pl.col("structure").eq("VISp"),
            pl.col("firing_rate").is_not_null(),
        )
        .select(
            lazynwb.NWB_PATH_COLUMN_NAME,
            lazynwb.TABLE_INDEX_COLUMN_NAME,
            "firing_rate",
        )
        .collect()
    )
    if candidates.is_empty():
        raise ValueError("No VISp unit with a firing_rate was found.")

    fastest_unit = candidates.sort("firing_rate", descending=True).head(1)
    nwb_path = str(fastest_unit[lazynwb.NWB_PATH_COLUMN_NAME].item())
    table_index = int(fastest_unit[lazynwb.TABLE_INDEX_COLUMN_NAME].item())
    firing_rate = float(fastest_unit["firing_rate"].item())
    logger.debug(
        "Fetching spike_times for fastest VISp unit in %s at row %d "
        "(firing_rate=%s).",
        nwb_path,
        table_index,
        firing_rate,
    )

    selected_unit = (
        units.filter(
            pl.col(lazynwb.NWB_PATH_COLUMN_NAME).eq(nwb_path),
            pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME).eq(table_index),
        )
        .select("spike_times")
        .collect()
    )
    if selected_unit.height != 1:
        raise ValueError(
            "Expected one unit at "
            f"{nwb_path!r} row {table_index}, found {selected_unit.height}."
        )
    spike_times = np.asarray(selected_unit["spike_times"].item(), dtype=np.float64)
    return float(np.diff(spike_times).max())


def _multisession_trials_hit_rate(trials: pl.LazyFrame) -> float:
    """Return the fraction of go trials in `intervals/trials` that were hits,
    summed across all sessions."""

    all_hit = (
        trials
        .select(pl.col('hit'))
        .collect()
    )
    hit_only = all_hit.filter(pl.col('hit').eq(True))
    return float(hit_only.height / all_hit.height)


def _max_speed_stimulus(nwb_path: str) -> str:
    """Return the stimulus in `intervals/stimulus_presentations` with the
    highest mean running speed (cm/s) in the first session."""
    stim = lazynwb.scan_nwb(nwb_path, '/intervals/stimulus_presentations')
    # get start_time and stop_time
    stim_lf = stim.select('image_name', 'start_time', 'stop_time').collect()
    speed = lazynwb.scan_nwb(nwb_path, '/processing/running/speed')
    speed_lf = (
        speed
        .select('data', 'timestamps')
        .collect()
    )
    # assign each speed sample to the presentation whose start_time most
    # recently precedes it, then keep only samples inside the presentation
    samples = (
        speed_lf
        .sort('timestamps')
        .join_asof(
            stim_lf.sort('start_time'),
            left_on='timestamps',
            right_on='start_time',
            strategy='backward',
        )
        .filter(pl.col('timestamps') < pl.col('stop_time'))
    )
    return str(
        samples
        .group_by('image_name')
        .agg(pl.col('data').mean().alias('mean_speed'))
        .sort('mean_speed', descending=True)
        .head(1)['image_name']
        .item()
    )



def _multisession_lick_rate_average(nwb_paths: list[str]) -> float:
    """Return the highest per-session average lick rate, computed from the
    `events/events` table of each session as lick count / event-time span
    (licks per second)."""
    events = lazynwb.scan_nwb(
        nwb_paths,
        "/events/events",
        disable_progress=True,
    )
    rates = (
        events
        .select(
            lazynwb.NWB_PATH_COLUMN_NAME,
            "event_type",
            "timestamp",
        )
        .group_by(lazynwb.NWB_PATH_COLUMN_NAME)
        .agg(
            (
                pl.col("event_type").eq("lick").sum()
                / (pl.col("timestamp").max() - pl.col("timestamp").min())
            ).alias("lick_rate_hz")
        )
        .sort("lick_rate_hz", descending=True)
        .collect()
    )
    return float(rates["lick_rate_hz"][0])


def _change_detection_large_array(nwb_path: str) -> float:
    """Return the mean of the first 20,000 samples (indices 0 through 19,999)
    of `processing/running/speed/data` in the first session."""
    speed = lazynwb.get_timeseries(
        nwb_path,
        "/processing/running/speed",
        exact_path=True,
    )
    data = np.asarray(
        speed.data[:_RUNNING_SPEED_DOWNLOAD_SAMPLES],
        dtype=np.float64,
    )
    return float(np.mean(data))


def _backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _configure_backend(backend: str) -> None:
    """Configure backend switches exposed by the installed lazynwb version."""
    version = _lazynwb_version()
    logger.debug("Configuring lazynwb %s backend %s.", version, backend)
    if version.split(".", maxsplit=1)[0] == "0":
        if backend not in {"obstore", "remfile", "s3fs"}:
            raise ValueError(f"Unsupported lazynwb pre-1.0 backend: {backend}")
        lazynwb.config.use_obstore = backend == "obstore"
        lazynwb.config.use_remfile = backend == "remfile"
        lazynwb.config.fsspec_storage_options = {"anon": True}
    elif backend != "obstore":
        raise ValueError(f"lazynwb {version} uses obstore; got backend {backend!r}.")


def _lazynwb_version() -> str:
    """Return the installed lazynwb distribution version."""
    return importlib.metadata.version("lazynwb")


def _set_catalog_cache_path() -> Path:
    """Point lazynwb at a matrix-provided cache or a fresh isolated cache."""
    cache_path = os.environ.get("NDB_LAZYNWB_CACHE_PATH")
    if cache_path is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
        cache_path = (cache_dir / "catalog.sqlite").as_posix()
        os.environ["NDB_LAZYNWB_CACHE_PATH"] = cache_path
    else:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path
    return Path(cache_path)


def _local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    value = os.environ.get("NDB_LOCAL_CACHE", "cold")
    if value not in {"cold", "warm"}:
        raise ValueError("NDB_LOCAL_CACHE must be 'cold' or 'warm' for lazynwb.")
    return value  # type: ignore[return-value]


def _default_implementation_id() -> str:
    """Return an ID containing the installed lazynwb version and backend."""
    version = _lazynwb_version().replace(".", "_").replace("+", "_")
    return f"lazynwb_{version}_{_backend()}"


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            _default_implementation_id(),
        ),
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=_local_cache(),
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=None if _local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
