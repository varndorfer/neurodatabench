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
# neurodatabench = { path = "https://github.com/varndorfer/neurodatabench" }
# ///


"""Direct Zarr implementation for the change detection benchmark questions."""

from __future__ import annotations

import logging
import numpy as np
import pydantic_settings
import pandas as pd
import os
import zarr
from time import perf_counter_ns
from typing import Any, Iterator, MutableMapping, Literal

import neurodatabench

logger = neurodatabench.get_logger(__name__)

_SOURCE_TYPE = "zarr"

class Settings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the direct Zarr runner."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: Literal["baseline_nwb_zarr_v0"] = "baseline_nwb_zarr_v0"
    object_store_backend: Literal["s3fs", "obstore"] = "s3fs"
    implementation_id: str = "direct_zarr"
    aws_region: str = "us-west-2"
    aws_skip_signature: bool = True


settings: Settings | None = None


def _settings() -> Settings:
    """Return the resolved implementation settings."""
    if settings is None:
        raise RuntimeError("Settings must be initialized before running the implementation.")
    return settings

#region Baseline solutions
def submit_answers(context: neurodatabench.RunContext) -> None:
    """Submit answers for every benchmark question."""
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)

        match question.id:
            case "small_nwb":
                _load_nwb(context.benchmark.data_sources[_SOURCE_TYPE][0])
            case "medium_nwb":
                _load_nwb(context.benchmark.data_sources[_SOURCE_TYPE][1])
            case "large_nwb":
                _load_nwb(context.benchmark.data_sources[_SOURCE_TYPE][2])
            case "load_trials_table":
                _load_trials_table(context.benchmark.data_sources[_SOURCE_TYPE][0])
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


def _load_nwb(nwb_path: str) -> dict[str, np.ndarray]:
    """Read every array in one NWB Zarr store into NumPy arrays, keyed by internal path."""
    root = _open_store(nwb_path)
    arrays: dict[str, np.ndarray] = {}
    if _is_zarr_v3():
        for path, member in root.members(max_depth=None):
            if isinstance(member, zarr.Array):
                arrays[path] = np.asarray(member[...])
    else:

        def _read_array(path: str, member: Any) -> None:
            """Read one zarr v2 array found while visiting the hierarchy."""
            if isinstance(member, zarr.Array):
                arrays[path] = np.asarray(member[...])

        root.visititems(_read_array)
    logger.debug("Loaded %d arrays from %s.", len(arrays), nwb_path)
    return arrays


def _load_trials_table(nwb_path: str) -> None:
    file = _open_store(nwb_path)
    trials = file['intervals']['trials']
    trials_df = pd.DataFrame()
    for k, v in trials.arrays():
        data = np.asarray(v)
        trials_df[k] = data.tolist()
    assert not trials_df.empty


def _load_multiple_trials_tables(nwb_paths: list[str]) -> dict[str, Any]:
    all_trials = []
    x_data = [0]
    y_data = [0]
    for i in range(len(nwb_paths)):
        start_ns = perf_counter_ns()
        file = _open_store(nwb_paths[i])
        trials = file['intervals']['trials']
        trials_df = pd.DataFrame()
        for k, v in trials.arrays():
            data = np.asarray(v)
            trials_df[k] = data.tolist()
        stop_ns = perf_counter_ns()

        all_trials.append(trials_df)
        x_data.append(i + 1)
        load_time_ns = stop_ns - start_ns
        y_data.append(load_time_ns / 1_000_000_000)
        
    return {"x": x_data, "y": y_data}
#endregion


#region Runtime Configuration
def _backend() -> str:
    """Return the requested direct Zarr object-store backend label."""
    return _settings().object_store_backend


def setup(context: neurodatabench.RunContext) -> None:
    """Configure process-level settings before answering benchmark questions."""
    logger.debug(
        "Preparing direct Zarr/%s access for %d NWB stores.",
        _backend(),
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )
    os.environ.setdefault("AWS_REGION", _settings().aws_region)
    _quiet_storage_debug_loggers()


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Clear implementation-managed caches before timed benchmark phases."""
    logger.debug(
        "No local direct Zarr cache to clear for %d NWB paths.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )


def teardown(context: neurodatabench.RunContext) -> None:
    """Release process-level resources."""
    logger.debug(
        "Direct Zarr benchmark teardown for %d NWB paths.",
        len(context.benchmark.data_sources[_SOURCE_TYPE]),
    )


def _quiet_storage_debug_loggers() -> None:
    """Keep benchmark debug logs focused on implementation-level events."""
    for logger_name in ("aiobotocore", "botocore", "fsspec", "s3fs", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
#endregion


# Zarr store access
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

        os.environ.setdefault("AWS_SKIP_SIGNATURE", str(_settings().aws_skip_signature).lower())
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
#endregion


if __name__ == "__main__":
    settings = Settings()
    neurodatabench.main(
        implementation_id=settings.implementation_id,
        implementation_nwb_interface=None,
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=settings.benchmark,
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
