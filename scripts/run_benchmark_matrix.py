#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pydantic>=2.13.4",
#   "pydantic-settings>=2.14.1",
#   "neurodatabench",
# ]
# [tool.uv.sources]
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Run the default NeuroDataBench implementation matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import pydantic
import pydantic_settings

import neurodatabench.matrix
import neurodatabench.models


def find_repo_root(script_path: Path) -> Path:
    """Find the checkout containing the benchmark implementation scripts."""
    script_dir = script_path.resolve().parent
    candidates = (
        script_dir,
        script_dir / "neurodatabench",
        *script_dir.parents,
    )
    for candidate in candidates:
        if (candidate / "implementations").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not find the neurodatabench checkout from {script_path}")


REPO_ROOT = find_repo_root(Path(__file__))

BENCHMARK = "dynamic_routing_nwb_v0"
H5PY_REMOTE_FILE_BACKENDS = ("remfile", "s3fs", "obstore")
CACHE_STATUSES: tuple[neurodatabench.models.LocalCacheState, ...] = ("cold",)


class MatrixSettings(pydantic_settings.BaseSettings):
    """Command-line and environment settings for the benchmark matrix."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        cli_shortcuts={
            "dry-run": "dry_run",
            "keep-going": "keep_going",
            "log-level": "log_level",
            "profile-interval-ms": "profile_interval_ms",
            "status-jsonl": "status_jsonl",
            "timeout-disabled": "timeout_disabled",
            "timeout-seconds": "timeout_seconds",
        },
        env_prefix="NDB_MATRIX_",
    )

    dry_run: bool = False
    keep_going: bool = True
    only: list[str] = pydantic.Field(default_factory=list)
    skip: list[str] = pydantic.Field(default_factory=list)
    limit: int | None = None
    out: Path = Path("results")
    status_jsonl: Path | None = None
    profile_interval_ms: int | None = None
    timeout_seconds: float | None = None
    timeout_disabled: bool = False
    log_level: str = "INFO"

    @pydantic.field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate a Python logging level name."""
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported log level: {value}")
        return normalized


def main() -> int:
    """Select and run entries from the default comparison matrix."""
    args = MatrixSettings()
    configure_logging(args.log_level)
    runs = neurodatabench.matrix.select_runs(
        default_matrix(),
        only=args.only,
        skip=args.skip,
    )
    if args.limit is not None:
        runs = runs[: args.limit]
    return neurodatabench.matrix.run_matrix(
        runs,
        repo_root=REPO_ROOT,
        output_root=args.out,
        status_path=args.status_jsonl,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        profile_interval_ms=args.profile_interval_ms,
        timeout_seconds=args.timeout_seconds,
        no_timeout=args.timeout_disabled,
        log_level=args.log_level,
    )


def configure_logging(log_level: str) -> None:
    """Configure logging for the matrix process."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def default_matrix() -> list[neurodatabench.matrix.MatrixRun]:
    """Build the repository's default benchmark matrix."""
    runs: list[neurodatabench.matrix.MatrixRun] = []
    benchmark = BENCHMARK
    for backend in ("obstore", "remfile", "s3fs"):
        runs.append(
            neurodatabench.matrix.MatrixRun(
                implementation=f"implementations/lazynwb/{BENCHMARK}.py",
                benchmark=benchmark,
                implementation_id=f"lazynwb-v0.2.91-{backend}",
                object_store_backend=backend,
                local_cache="cold",
                dependencies=("lazynwb==0.2.91",),
            )
        )
    for local_cache in CACHE_STATUSES:
        runs.append(
            neurodatabench.matrix.MatrixRun(
                implementation=f"implementations/lazynwb/{BENCHMARK}.py",
                benchmark=benchmark,
                implementation_id=f"lazynwb-1.0.0dev3-obstore-{local_cache}",
                object_store_backend="obstore",
                local_cache=local_cache,
                dependencies=("lazynwb==1.0.0dev3",),
            )
        )
        runs.append(
            neurodatabench.matrix.MatrixRun(
                implementation=f"implementations/lazynwb/{BENCHMARK}.py",
                benchmark=benchmark,
                implementation_id=f"lazynwb-dev-obstore-{local_cache}",
                object_store_backend="obstore",
                local_cache=local_cache,
                dependencies=("git+https://github.com/bjhardcastle/lazynwb@dev6",),
            )
        )

    for backend in H5PY_REMOTE_FILE_BACKENDS:
        runs.append(
            neurodatabench.matrix.MatrixRun(
                implementation="implementations/direct_h5py_template.py",
                benchmark=BENCHMARK,
                implementation_id=f"direct-h5py-{backend}",
                object_store_backend=backend,
            )
        )


    for zarr_major, zarr_requirement in (("2", "zarr<3"), ("3", "zarr>=3,<4")):
        for backend in ("s3fs", "obstore"):
            runs.append(
                neurodatabench.matrix.MatrixRun(
                    implementation=f"implementations/direct_zarr/{BENCHMARK}.py",
                    benchmark=BENCHMARK,
                    implementation_id=f"direct-zarr-v{zarr_major}-{backend}",
                    object_store_backend=backend,
                    dependencies=(zarr_requirement,),
                )
            )

    runs.append(
        neurodatabench.matrix.MatrixRun(
            implementation=f"implementations/pynwb/{BENCHMARK}.py",
            benchmark=BENCHMARK,
            implementation_id="pynwb-s3fs",
            object_store_backend="s3fs",
            dependencies=("zarr<3",),
        )
    )

    for backend in H5PY_REMOTE_FILE_BACKENDS:
        runs.append(
            neurodatabench.matrix.MatrixRun(
                implementation="implementations/pynwb_template.py",
                benchmark=BENCHMARK,
                implementation_id=f"pynwb-h5py-{backend}",
                object_store_backend=backend,
            )
        )
    return runs


if __name__ == "__main__":
    raise SystemExit(main())
