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
# neurodatabench = { path = "https://github.com/varndorfer/neurodatabench" }
# ///


"""lazynwb answers for the change detection benchmark questions."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import lazynwb
import numpy as np
import pydantic_settings
import pandas as pd
import polars as pl
import zarr

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_SOURCE_TYPE = "zarr"

class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the lazynwb runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: Literal["baseline_nwb_zarr_v0"] = "baseline_nwb_zarr_v0"
    object_store_backend: Literal["obstore", "remfile", "s3fs"] = "obstore"
    implementation_id: str = "lazynwb"
    local_cache: Literal["cold", "warm"] = "cold"
    lazynwb_cache_path: Path | None = None
    aws_region: str = "us-west-2"


settings: Settings | None = None


def _settings() -> Settings:
    """Return the resolved implementation settings."""
    if settings is None:
        raise RuntimeError("Settings must be initialized before running the implementation.")
    return settings

state: dict[str, Any] = {}

#region Baseline solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)

        match question.id:
            case "small_nwb":
                answer = _load_nwb(context.benchmark.data_sources[_SOURCE_TYPE][0])
            case "medium_nwb":
                answer = _load_nwb(context.benchmark.data_sources[_SOURCE_TYPE][1])
            case "large_nwb":
                answer = _load_nwb(context.benchmark.data_sources[_SOURCE_TYPE][2])
            case "load_trials_table":
                answer = _load_trials_table(context.benchmark.data_sources[_SOURCE_TYPE][0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")

        context.submit_answer(question.id, 0)

    """Submit metrics"""
    trials_load_times = _load_multiple_trials_tables(context.benchmark.data_sources[_SOURCE_TYPE])
    trials_metric = neurodatabench.RunMetric(
        x_data=trials_load_times["x"],
        x_label="tables loaded",
        y_data=trials_load_times["y"],
        y_label="load time (s)",
    )
    context.run_metrics["Load Trials Tables"] = trials_metric


def _load_nwb(file: Any) -> Any:
    paths = lazynwb.get_internal_paths(file)
    dfs = {}
    for k in paths.keys():
        if type(paths[k]) is zarr.hierarchy.Group: # only gets DynamicTables, not TimeSeries
            dfs[k] = lazynwb.get_df(file, k)
    return len(dfs.values())


def _load_trials_table(file: Any) -> None:
    df = lazynwb.get_df(
        file,
        'intervals/trials'
    )
    assert type(df) is pd.DataFrame


def _load_multiple_trials_tables(nwb_paths: list[str]) -> dict[str, Any]:
    start_ns = perf_counter_ns()
    all_trials = lazynwb.get_df(
        nwb_paths,
        'intervals/trials'
    )
    stop_ns = perf_counter_ns()
    x_data = [0]
    y_data = [0]
    x_data.append(len(nwb_paths))
    load_time_ns = stop_ns - start_ns
    y_data.append(load_time_ns / 1_000_000_000)
        
    return {"x": x_data, "y": y_data}
#endregion

#region Runtime Configuration
def _backend() -> str:
    """Return the requested lazynwb object-store backend label."""
    return _settings().object_store_backend


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
    cache_path = _settings().lazynwb_cache_path
    if cache_path is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="neurodatabench-lazynwb-"))
        cache_path = cache_dir / "catalog.sqlite"
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LAZYNWB_CATALOG_CACHE_PATH"] = cache_path.as_posix()
    return cache_path


def _local_cache() -> neurodatabench.models.LocalCacheState:
    """Return local cache metadata declared for this run."""
    return _settings().local_cache


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Remove the selected lazynwb catalog before a cold run."""
    logger.debug(
        "Clearing lazynwb caches for %d NWB paths before measured phases.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )
    cache_path = _set_catalog_cache_path()
    for path in (cache_path, Path(f"{cache_path}-shm"), Path(f"{cache_path}-wal")):
        path.unlink(missing_ok=True)


def setup(context: neurodatabench.RunContext) -> None:
    """Configure lazynwb before answering benchmark questions."""
    logger.debug("Preparing lazynwb for %d NWB paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    _set_catalog_cache_path()
    os.environ.setdefault("AWS_REGION", _settings().aws_region)

    lazynwb.config.anon = True
    _configure_backend(_backend())
    state.clear()


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug("Clearing lazynwb state for %d NWB paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    state.clear()
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface="lazynwb",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=_local_cache(),
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=None if _local_cache() == "warm" else clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
