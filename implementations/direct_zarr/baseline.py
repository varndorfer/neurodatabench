# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "altair",
#   "hdmf-zarr",
#   "numcodecs",
#   "numpy",
#   "obstore",
#   "pandas",
#   "psutil",
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "remfile",
#   "s3fs",
#   "zarr",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { path = "../..", editable = true }
# ///


"""Direct Zarr implementation for the change detection benchmark questions."""

from __future__ import annotations

import io
import itertools
import json
import logging
import numcodecs
import numpy as np
import pydantic_settings
import numcodecs.pickles
import pickle
import pandas as pd
import os
import zarr
from time import perf_counter_ns
from typing import Any, Iterator, MutableMapping, Literal

import neurodatabench

logger = neurodatabench.get_logger(__name__)
# Zarr 2 settting for decoding arrays with object-dtype stored with a pickle codec
numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True 

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
            case "load_trials_table":
                _load_table(context.benchmark.data_sources[_SOURCE_TYPE][1], "intervals/trials")
            case "load_units_table":
                _load_table(context.benchmark.data_sources[_SOURCE_TYPE][0], "units")
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


def _load_table(nwb_path: str, table_name: str) -> None:
    """Load one NWB table group fully into a pandas DataFrame."""
    file = _open_store(nwb_path)
    table = file[table_name]

    if _is_zarr_v3():
        # zarr 3 cannot parse v2 object/pickle arrays, so read the raw v2 layout directly.
        arrays = _read_table_arrays_raw(table)
    else:
        arrays = {k: np.asarray(v[...]) for k, v in table.arrays()}

    index_names = {k for k in arrays if k.endswith("_index")}
    columns: dict[str, Any] = {}
    for name, data in arrays.items():
        if name in index_names:
            continue  # consumed by its target column below
        if f"{name}_index" in index_names:
            ends = arrays[f"{name}_index"]
            columns[name] = np.split(data, ends[:-1])  # one array per row
        else:
            columns[name] = data.tolist()

    table_df = pd.DataFrame(columns)
    assert not table_df.empty


def _store_get(store: Any, key: str) -> bytes | None:
    """Fetch one raw object from a zarr v3 store, returning None when it is missing."""
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    buf = sync(store.get(key, prototype=default_buffer_prototype()))
    return None if buf is None else buf.to_bytes()


def _store_list_dir(store: Any, prefix: str) -> list[str]:
    """List the child names directly under a prefix of a zarr v3 store."""
    from zarr.core.sync import sync

    async def _collect() -> list[str]:
        """Drain the async listing."""
        return [name async for name in store.list_dir(prefix)]

    return sync(_collect())


def _read_table_arrays_raw(table: Any) -> dict[str, np.ndarray]:
    """Read every array of a zarr v2 group through its store, bypassing zarr-python's dtype parsing."""
    store, root = table.store, table.path.strip("/")
    arrays: dict[str, np.ndarray] = {}
    for name in _store_list_dir(store, root):
        raw = _store_get(store, f"{root}/{name}/.zarray")
        if raw is None:
            continue  # subgroup or attribute file, not an array
        arrays[name] = _read_array_raw(store, f"{root}/{name}", json.loads(raw))
    logger.debug("Read %d arrays from %s without zarr dtype parsing.", len(arrays), root)
    return arrays


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that replaces hdmf_zarr classes with inert stubs so zarr 2-only imports are skipped."""

    def find_class(self, module: str, name: str) -> Any:
        """Return a stub for hdmf_zarr classes, otherwise resolve normally."""
        if module.split(".", maxsplit=1)[0] == "hdmf_zarr":
            return type(name, (dict,), {})
        return super().find_class(module, name)


def _unpickle_stubbed(buf: bytes) -> np.ndarray:
    """Decode a numcodecs pickle chunk without importing hdmf_zarr (incompatible with zarr 3)."""
    return _StubUnpickler(io.BytesIO(buf)).load()


def _read_array_raw(store: Any, path: str, zarray: dict[str, Any]) -> np.ndarray:
    """Assemble one zarr v2 array from its chunk objects, decoding with numcodecs."""
    shape, chunks = tuple(zarray["shape"]), tuple(zarray["chunks"])
    dtype = np.dtype(zarray["dtype"])
    sep = zarray.get("dimension_separator", ".")
    compressor = numcodecs.get_codec(zarray["compressor"]) if zarray.get("compressor") else None
    filters = [numcodecs.get_codec(f) for f in zarray.get("filters") or []]
    fill = zarray.get("fill_value")
    out = np.empty(shape, dtype=dtype)
    if fill is not None and dtype != object:
        out.fill(fill)
    grid = [range(-(-s // c)) for s, c in zip(shape, chunks)]
    for idx in itertools.product(*grid):
        buf = _store_get(store, f"{path}/{sep.join(map(str, idx))}")
        if buf is None:
            continue  # missing chunk means fill value
        if compressor is not None:
            buf = compressor.decode(buf)
        for codec in reversed(filters):
            buf = _unpickle_stubbed(buf) if codec.codec_id == "pickle" else codec.decode(buf)
        chunk = buf if isinstance(buf, np.ndarray) else np.frombuffer(buf, dtype=dtype)
        chunk = chunk.reshape(chunks)
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[region] = chunk[tuple(slice(0, r.stop - r.start) for r in region)]
    return out


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
