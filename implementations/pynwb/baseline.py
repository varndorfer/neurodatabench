# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "hdmf-zarr",
#   "numpy",
#   "pandas",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "pynwb",
#   "s3fs",
#   "zarr<3",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { path = "../..", editable = true }
# ///


"""PyNWB NWBZarrIO answers for the change detection benchmark questions."""

from __future__ import annotations

import numpy as np
import pydantic_settings
import warnings
from time import perf_counter_ns
from typing import Any, Literal
from hdmf_zarr import NWBZarrIO
from hdmf.common import DynamicTable
from pynwb import TimeSeries

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_SOURCE_TYPE = "zarr"

class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the PyNWB Zarr runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: Literal["baseline_nwb_zarr_v0"] = "baseline_nwb_zarr_v0"
    object_store_backend: Literal["s3fs", "obstore"] = "s3fs"
    implementation_id: str = "pynwb"


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
            case "load_trials_table":
                answer = _load_trials_table(context.benchmark.data_sources[_SOURCE_TYPE][1])
            case "load_units_table":
                answer = _load_units_table(context.benchmark.data_sources[_SOURCE_TYPE][0])
            case "load_multiple_trials_tables":
                trials_load_times = _load_multiple_trials_tables(context.benchmark.data_sources[_SOURCE_TYPE][1:])
                trials_metric = neurodatabench.RunMetric(
                    x_data=trials_load_times["x"],
                    x_label="tables loaded",
                    y_data=trials_load_times["y"],
                    y_label="load time (s)",
                )
                context.run_metrics["Load Trials Tables"] = trials_metric
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")

        context.submit_answer(question.id, 0)


def _object_path(obj) -> str:
    """Return the object's path inside the file, e.g. 'intervals/trials'."""
    parts = []
    while obj.parent is not None:
        parts.append(obj.name)
        obj = obj.parent
    return "/".join(reversed(parts))


def _load_trials_table(nwb_path: str) -> Any:
    file = _open_file(nwb_path)
    nwb_file = file["nwb_file"]
    trials_df = nwb_file.trials.to_dataframe()
    assert not trials_df.empty
    return trials_df


def _load_units_table(nwb_path: str) -> Any:
    file = _open_file(nwb_path)
    nwb_file = file["nwb_file"]
    units_df = nwb_file.units.to_dataframe()
    assert not units_df.empty
    return units_df


def _load_multiple_trials_tables(nwb_paths: list[str]) -> dict[str, Any]:
    all_trials = []
    x_data = [0]
    y_data = [0]
    for i in range(len(nwb_paths)):
        start_ns = perf_counter_ns()
        trials_df = _load_trials_table(nwb_paths[i])
        stop_ns = perf_counter_ns()

        all_trials.append(trials_df)
        x_data.append(i + 1)
        load_time_ns = stop_ns - start_ns
        y_data.append(load_time_ns / 1_000_000_000)
        
    return {"x": x_data, "y": y_data}
#endregion


#region Runtime Configuration
def _backend() -> str:
    """Return the requested NWBZarrIO object-store backend."""
    return _settings().object_store_backend


def _open_file(nwb_path: str) -> dict[str, Any]:
    """Return the file record for the given NWB path."""
    try:
        logger.debug("Opening %s through NWBZarrIO.", nwb_path)
        with warnings.catch_warnings():
            """Suppress column name conflict warnings from the hdmf library"""
            warnings.simplefilter("ignore")
            nwb_io = NWBZarrIO(
                path=nwb_path,
                mode="r",
                load_namespaces=True,
                storage_options={"anon": True},
            )
            nwb_file = nwb_io.read()
    except Exception as e:
        logger.error("Failed to open %s: %s", nwb_path, e)
        raise
    return {"nwb_io": nwb_io, "nwb_file": nwb_file}


def _open_files(data_sources: list[str]) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile in `state`."""
    state.clear()
    state["files"] = []
    try:
        for nwb_path in data_sources:
            logger.debug("Opening %s through NWBZarrIO.", nwb_path)
            nwb_io = NWBZarrIO(
                path=nwb_path,
                mode="r",
                load_namespaces=True,
                storage_options={"anon": True},
            )
            file_record: dict[str, Any] = {"nwb_io": nwb_io}
            state["files"].append(file_record)
            file_record["nwb_file"] = nwb_io.read()
    except Exception:
        logger.debug("Closing partially opened NWBZarrIO handles after setup failure.")
        _close_files()
        raise


def _close_files() -> None:
    """Close every NWBZarrIO handle and clear implementation state."""
    for file_record in reversed(state.get("files", [])):
        file_record["nwb_io"].close()
    state.clear()


def setup(context: neurodatabench.RunContext) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile."""
    backend = _backend()
    if backend != "s3fs":
        raise ValueError(f"NWBZarrIO does not support configured backend {backend!r}.")
    logger.debug(
        "Opening %d Zarr stores through NWBZarrIO and s3fs.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that this implementation has no managed local cache."""
    logger.debug("No PyNWB Zarr cache to clear for %d paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))


def teardown(context: neurodatabench.RunContext) -> None:
    """Close NWBZarrIO handles and clear materialized state."""
    logger.debug("Closing NWBZarrIO handles for %d paths.", len(context.benchmark.data_sources[_SOURCE_TYPE]))
    _close_files()
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface="pynwb/NWBZarrIO",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )