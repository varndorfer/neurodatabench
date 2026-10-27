"""Runner APIs and subprocess supervision for NeuroDataBench implementations."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.metadata
import importlib.resources
import inspect
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import psutil
import pydantic
import pydantic_settings

import neurodatabench.benchmarks
import neurodatabench.logging_utils
import neurodatabench.models
import neurodatabench.plots
import neurodatabench.validation

logger = logging.getLogger(__name__)
_IMPLEMENTATION_SCRIPT_ARTIFACT_NAME = "implementation.py"
_REQUIREMENTS_ARTIFACT_NAME = "requirements.txt"
BenchmarkValidationError = neurodatabench.validation.BenchmarkValidationError
SUPERVISOR_TIMEOUT_EXIT_CODE = 124


class _RunConfig(pydantic_settings.BaseSettings):
    """Resolved runner configuration from call defaults, environment, and CLI.

    profile_interval_seconds is derived from profile_interval_ms and controls
    how often the background resource profiler samples during a run. Environment
    variables use the NDB_ prefix, for example NDB_BENCHMARK.
    """

    model_config = pydantic_settings.SettingsConfigDict(
        cli_implicit_flags=True,
        cli_kebab_case=True,
        cli_parse_args=True,
        env_prefix="NDB_",
    )

    benchmark: str | None = None
    out: Path | None = None
    implementation_script: Path | None = None
    fail_fast: bool = False
    profile_interval_ms: int = pydantic.Field(default=250, gt=0)
    log_level: str = "INFO"
    log_file: Path | None = None

    @pydantic.field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Normalize and validate a Python logging level name."""
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @property
    def profile_interval_seconds(self) -> float:
        """Return the profiler sampling interval in seconds."""
        return self.profile_interval_ms / 1000


class _Profiler:
    """Background sampler for process and system resource usage."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        process: psutil.Process | None = None,
    ) -> None:
        """Create a profiler with a sampling interval in seconds."""
        self.interval_seconds = interval_seconds
        self.samples: list[neurodatabench.models.JsonObject] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process() if process is None else process
        self._net_start: Any = None
        self._net_end: Any = None
        self._disk_start: Any = None
        self._disk_end: Any = None
        self._baseline_process_rss_bytes: int | None = None
        self._baseline_process_plus_children_rss_bytes: int | None = None

    def start(self) -> None:
        """Start sampling in a background thread."""
        logger.debug("Starting profiler.")
        self._net_start = psutil.net_io_counters()
        self._disk_start = psutil.disk_io_counters()
        self._record_memory_baseline()
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record final counters."""
        logger.debug("Stopping profiler.")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        final_sample = self._sample()
        if final_sample is not None:
            self.samples.append(final_sample)
        self._net_end = psutil.net_io_counters()
        self._disk_end = psutil.disk_io_counters()

    def summary(self) -> neurodatabench.models.JsonObject:
        """Return a compact summary of collected samples."""
        process_rss_values = [
            int(sample["process"]["rss_bytes"])
            for sample in self.samples
            if isinstance(sample.get("process"), dict)
        ]
        total_rss_values = [
            int(sample["process"]["rss_bytes"]) + int(sample["children"]["rss_bytes"])
            for sample in self.samples
            if isinstance(sample.get("process"), dict)
            and isinstance(sample.get("children"), dict)
        ]
        peak_process_rss_bytes = max(process_rss_values, default=None)
        peak_total_rss_bytes = max(total_rss_values, default=None)
        return {
            "sample_interval_seconds": self.interval_seconds,
            "num_samples": len(self.samples),
            "baseline_process_rss_bytes": self._baseline_process_rss_bytes,
            "baseline_process_plus_children_rss_bytes": (
                self._baseline_process_plus_children_rss_bytes
            ),
            "peak_process_rss_bytes": peak_process_rss_bytes,
            "peak_process_plus_children_rss_bytes": peak_total_rss_bytes,
            "peak_process_rss_delta_bytes": _memory_delta(
                peak_process_rss_bytes,
                self._baseline_process_rss_bytes,
            ),
            "peak_process_plus_children_rss_delta_bytes": _memory_delta(
                peak_total_rss_bytes,
                self._baseline_process_plus_children_rss_bytes,
            ),
            "network_delta": _counter_delta(self._net_start, self._net_end),
            "disk_delta": _counter_delta(self._disk_start, self._disk_end),
            "network_scope": "system_delta",
            "disk_scope": "system_delta",
        }

    def _record_memory_baseline(self) -> None:
        """Record the process RSS baseline before measured setup starts."""
        try:
            process_rss = self._process.memory_info().rss
            child_rss = _child_rss_bytes(self._process)
        except (psutil.Error, PermissionError):
            logger.debug(
                "Unable to inspect the profiled process for memory baseline capture.",
                exc_info=True,
            )
            return
        self._baseline_process_rss_bytes = process_rss
        self._baseline_process_plus_children_rss_bytes = process_rss + child_rss

    def _run(self) -> None:
        """Collect samples until stopped."""
        while not self._stop.is_set():
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def _sample(self) -> neurodatabench.models.JsonObject | None:
        """Collect one profiler sample."""
        try:
            memory = self._process.memory_info()
            process_cpu_percent = self._process.cpu_percent(interval=None)
            process_num_threads = self._process.num_threads()
        except (psutil.Error, PermissionError):
            logger.debug(
                "Skipping sample because the profiled process could not be inspected.",
                exc_info=True,
            )
            return None
        child_rss = 0
        child_cpu = 0.0
        try:
            children = self._process.children(recursive=True)
        except (psutil.Error, PermissionError):
            logger.debug(
                "Unable to inspect child processes during profiling.",
                exc_info=True,
            )
            children = []
        for child in children:
            try:
                child_rss += child.memory_info().rss
                child_cpu += child.cpu_percent(interval=None)
            except (psutil.Error, PermissionError):
                logger.debug(
                    "Skipping a child process that could not be inspected.",
                    exc_info=True,
                )
        virtual_memory = psutil.virtual_memory()
        return {
            "time_ns": time.time_ns(),
            "process": {
                "cpu_percent": process_cpu_percent,
                "rss_bytes": memory.rss,
                "vms_bytes": memory.vms,
                "num_threads": process_num_threads,
            },
            "children": {
                "cpu_percent": child_cpu,
                "rss_bytes": child_rss,
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_total_bytes": virtual_memory.total,
                "memory_available_bytes": virtual_memory.available,
                "memory_percent": virtual_memory.percent,
            },
            "io": {
                "net": psutil.net_io_counters()._asdict(),
                "disk": psutil.disk_io_counters()._asdict(),
            },
        }


def main(
    *,
    setup: Callable[[neurodatabench.models.RunContext], None],
    submit_answers: Callable[[neurodatabench.models.RunContext], None],
    implementation_id: str,
    implementation_source_type: str | None = None,
    implementation_nwb_interface: str | None = None,
    implementation_object_store_backend: str | None = None,
    implementation_local_cache: neurodatabench.models.LocalCacheState | None = None,
    implementation_remote_cache: bool | None = False,
    benchmark: str | Path | None = None,
    out: str | Path | None = None,
    implementation_script: str | Path | None = None,
    log_level: str | None = None,
    log_file: str | Path | None = None,
    fail_fast: bool | None = None,
    clear_cache: Callable[[neurodatabench.models.RunContext], None] | None = None,
    teardown: Callable[[neurodatabench.models.RunContext], None] | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Run a benchmark implementation and write result artifacts."""
    logger.debug("Resolving runner configuration.")
    implementation = neurodatabench.models.Implementation(
        id=implementation_id,
        source_type=implementation_source_type or os.environ.get("NDB_SOURCE_TYPE"),
        nwb_interface=implementation_nwb_interface,
        object_store_backend=implementation_object_store_backend,
        local_cache=implementation_local_cache,
        remote_cache=implementation_remote_cache,
    )
    config = _resolve_config(
        default_benchmark=benchmark,
        default_out=out,
        default_implementation_script=implementation_script,
        default_log_level=log_level,
        default_log_file=log_file,
        default_fail_fast=fail_fast,
        argv=argv,
    )
    if config.fail_fast:
        logger.warning(
            "Fail-fast answer validation is enabled. Use it during development only, "
            "not for benchmark runs."
        )
    benchmark_source = config.benchmark
    assert benchmark_source is not None

    raw_benchmark, loaded_benchmark = _load_benchmark(benchmark_source)
    implementation_script_path = _resolve_implementation_script_path(
        config.implementation_script,
        setup=setup,
        submit_answers=submit_answers,
    )
    out_dir = (
        config.out
        if config.out is not None
        else _default_output_dir(
            benchmark_id=loaded_benchmark.id,
            implementation_id=implementation_id,
        )
    )
    staged_log_path: Path | None = None
    log_path = config.log_file
    if log_path is None:
        staged_fd, staged_name = tempfile.mkstemp(
            prefix="neurodatabench-",
            suffix=".log",
        )
        os.close(staged_fd)
        staged_log_path = Path(staged_name)
        log_path = staged_log_path
    logging_session = neurodatabench.logging_utils.configure(
        console_level=config.log_level,
        file_path=log_path,
    )

    def finish_logging(*, successful: bool) -> None:
        """Flush the run log and publish or discard its default staging file."""
        if logging_session is not None:
            logging_session.close()
        if staged_log_path is None:
            return
        if successful:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(staged_log_path, out_dir / "run.log")
        elif staged_log_path.exists():
            staged_log_path.unlink()
    submitted_answers: list[dict[str, Any]] = []
    run_start_ns: int | None = None
    run_start_wall_time_ns = 0
    total_start_ns: int | None = None

    def submit_answer(
        question_id: str,
        answer: neurodatabench.models.JsonValue,
    ) -> None:
        """Capture one submitted answer with wall-clock time."""
        expected = neurodatabench.validation.expected_answer(
            loaded_benchmark,
            question_id,
        )
        reused_expected_object = (
            isinstance(expected, (dict, list)) and answer is expected
        )
        if config.fail_fast:
            logger.debug("Fail-fast validating answer for question %s.", question_id)
            neurodatabench.validation.validate_answer_submission(
                loaded_benchmark,
                submitted_answers,
                question_id,
                answer,
            )
        submitted_at = datetime.now(timezone.utc).isoformat()
        submitted_elapsed_seconds = (
            None
            if run_start_ns is None
            else (perf_counter_ns() - run_start_ns) / 1_000_000_000
        )
        logger.debug("Recording submitted answer for question %s.", question_id)
        if reused_expected_object:
            logger.warning(
                "Submitted answer for %s reuses the expected answer object.",
                question_id,
            )
        submitted_answers.append(
            {
                "question_id": question_id,
                "answer": answer,
                "submitted_at": submitted_at,
                "submitted_elapsed_seconds": submitted_elapsed_seconds,
                "reused_expected_object": reused_expected_object,
            }
        )

    context = neurodatabench.models.RunContext(
        benchmark=loaded_benchmark,
        submit_answer=submit_answer,
        run_metrics={}
    )
    profiler = _Profiler(interval_seconds=config.profile_interval_seconds)
    setup_duration_ns = 0
    submit_answers_duration_ns = 0
    total_duration_ns = 0
    current_phase: str | None = None
    current_phase_start_ns: int | None = None

    if clear_cache is not None:
        logger.debug("Running untimed clear_cache.")
        try:
            clear_cache(context)
        except BaseException:
            finish_logging(successful=False)
            raise

    profiler.start()
    try:
        run_start_wall_time_ns = time.time_ns()
        total_start_ns = perf_counter_ns()
        run_start_ns = total_start_ns
        current_phase = "setup"
        current_phase_start_ns = perf_counter_ns()
        setup(context)
        setup_duration_ns = perf_counter_ns() - current_phase_start_ns

        current_phase = "submit_answers"
        current_phase_start_ns = perf_counter_ns()
        submit_answers(context)
        submit_answers_duration_ns = perf_counter_ns() - current_phase_start_ns
        total_duration_ns = perf_counter_ns() - total_start_ns
        current_phase = None
        current_phase_start_ns = None
    finally:
        if current_phase is not None and total_start_ns is not None:
            now_ns = perf_counter_ns()
            total_duration_ns = now_ns - total_start_ns
            if current_phase == "setup" and current_phase_start_ns is not None:
                setup_duration_ns = now_ns - current_phase_start_ns
            elif (
                current_phase == "submit_answers" and current_phase_start_ns is not None
            ):
                submit_answers_duration_ns = now_ns - current_phase_start_ns
        profiler.stop()
        if teardown is not None:
            logger.debug("Running untimed teardown.")
            teardown(context)
        if sys.exc_info()[0] is not None and logging_session is not None:
            finish_logging(successful=False)

    timings = neurodatabench.models.RunTimings(
        setup_duration_ns=setup_duration_ns,
        submit_answers_duration_ns=submit_answers_duration_ns,
        total_duration_ns=total_duration_ns,
        phase_timings=_run_phase_timings(
            setup_duration_ns=setup_duration_ns,
            submit_answers_duration_ns=submit_answers_duration_ns,
            total_duration_ns=total_duration_ns,
        ),
        answer_submissions=[
            neurodatabench.models.AnswerSubmissionTiming(
                question_id=str(submitted_answer["question_id"]),
                submitted_at=str(submitted_answer["submitted_at"]),
                submitted_elapsed_seconds=(
                    float(submitted_answer["submitted_elapsed_seconds"])
                    if submitted_answer["submitted_elapsed_seconds"] is not None
                    else None
                ),
            )
            for submitted_answer in submitted_answers
        ],
    )
    validation: neurodatabench.models.JsonObject
    try:
        neurodatabench.validation.validate_answers(
            loaded_benchmark,
            submitted_answers,
        )
    except neurodatabench.validation.BenchmarkValidationError as error:
        finish_logging(successful=False)
        raise SystemExit(str(error)) from None
    validation = {"correct": True}
    metadata = _run_metadata(
        implementation=implementation,
        benchmark_source=benchmark_source,
        benchmark=loaded_benchmark,
        implementation_script_path=implementation_script_path,
    )
    try:
        _write_result_artifacts(
            out_dir=out_dir,
            raw_benchmark=raw_benchmark,
            metadata=metadata,
            implementation_script_path=implementation_script_path,
            timings=timings,
            metrics=context.run_metrics,
            validation=validation,
            profile_samples=profiler.samples,
            profile_summary=profiler.summary(),
            run_start_wall_time_ns=run_start_wall_time_ns,
        )
        logger.info(
            "Benchmark run completed in %.3f s (setup %.3f s, submit_answers %.3f s).",
            total_duration_ns / 1_000_000_000,
            setup_duration_ns / 1_000_000_000,
            submit_answers_duration_ns / 1_000_000_000,
        )
    finally:
        finish_logging(successful=True)


def _resolve_config(
    *,
    default_benchmark: str | Path | None,
    default_out: str | Path | None,
    default_log_level: str | None,
    default_log_file: str | Path | None = None,
    argv: Sequence[str] | None,
    default_fail_fast: bool | None = None,
    default_implementation_script: str | Path | None = None,
) -> _RunConfig:
    """Resolve call defaults with Pydantic Settings overrides."""
    settings_kwargs: dict[str, object] = {}
    if default_benchmark is not None:
        settings_kwargs["benchmark"] = str(default_benchmark)
    if default_out is not None:
        settings_kwargs["out"] = Path(default_out)
    if default_implementation_script is not None:
        settings_kwargs["implementation_script"] = Path(
            default_implementation_script,
        )
    if default_log_level is not None:
        settings_kwargs["log_level"] = default_log_level
    if default_log_file is not None:
        settings_kwargs["log_file"] = Path(default_log_file)
    if default_fail_fast is not None:
        settings_kwargs["fail_fast"] = default_fail_fast
    config = _RunConfig(
        **settings_kwargs,
        _cli_parse_args=argv,
    )
    if config.benchmark is None:
        raise ValueError("benchmark must be provided to main() or --benchmark")
    return config


def _default_output_dir(*, benchmark_id: str, implementation_id: str) -> Path:
    """Return a benchmark-scoped implementation directory beneath results."""
    return Path.cwd() / "results" / benchmark_id / implementation_id


def _load_benchmark(
    benchmark: str,
) -> tuple[neurodatabench.models.JsonObject, neurodatabench.models.Benchmark]:
    """Load a benchmark from a filesystem path or packaged benchmark name."""
    logger.debug("Loading benchmark %s.", benchmark)
    path = Path(benchmark)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        resource_name = path.name if path.suffix == ".json" else f"{benchmark}.json"
        resource = importlib.resources.files(neurodatabench.benchmarks).joinpath(
            resource_name
        )
        if not resource.is_file():
            raise FileNotFoundError(f"Benchmark not found: {benchmark}")
        raw = json.loads(resource.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Benchmark JSON must be an object.")
    loaded = neurodatabench.models.Benchmark.model_validate(raw)
    return raw, loaded


def _cli(argv: Sequence[str] | None = None) -> int:
    """Run the NeuroDataBench runner command-line interface."""
    parser = argparse.ArgumentParser(prog="python -m neurodatabench.runner")
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    supervise_parser = subparsers.add_parser(
        "supervise",
        help="run a benchmark command with process-level timeout enforcement",
    )
    supervise_parser.add_argument(
        "--benchmark",
        help="Benchmark path or packaged name used to resolve the default timeout.",
    )
    supervise_parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Process-level timeout override in seconds.",
    )
    supervise_parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable process-level timeout enforcement.",
    )
    supervise_parser.add_argument(
        "--timeout-profile-out",
        type=Path,
        help=(
            "Write profile samples and a summary here when the command is "
            "terminated by the supervisor timeout."
        ),
    )
    supervise_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'.",
    )
    args = parser.parse_args(argv)
    if args.command_name == "supervise":
        command = _supervised_command_args(args.command)
        if not command:
            supervise_parser.error("a command is required after '--'")
        try:
            timeout_seconds = _supervised_timeout_seconds(
                benchmark=args.benchmark,
                timeout_seconds=args.timeout_seconds,
                no_timeout=bool(args.no_timeout),
            )
        except ValueError as error:
            supervise_parser.error(str(error))
        return _run_supervised_command(
            command,
            timeout_seconds=timeout_seconds,
            timeout_profile_out=args.timeout_profile_out,
            benchmark_source=args.benchmark,
        )
    parser.error(f"unsupported command: {args.command_name}")


def _supervised_command_args(command: Sequence[str]) -> list[str]:
    """Return subprocess command arguments with the argparse separator removed."""
    if command and command[0] == "--":
        return list(command[1:])
    return list(command)


def _supervised_timeout_seconds(
    *,
    benchmark: str | None,
    timeout_seconds: float | None,
    no_timeout: bool,
) -> float | None:
    """Return the process-level timeout for a supervised benchmark command."""
    if no_timeout:
        logger.debug("Supervisor timeout disabled by command-line flag.")
        return None
    if timeout_seconds is not None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        logger.debug(
            "Using supervisor timeout override of %.3f seconds.",
            timeout_seconds,
        )
        return timeout_seconds
    if benchmark is None:
        logger.debug("No benchmark supplied; supervisor timeout is disabled.")
        return None
    loaded_benchmark = _load_benchmark(benchmark)[1]
    logger.debug(
        "Using benchmark supervisor timeout of %s seconds.",
        loaded_benchmark.timeout_seconds,
    )
    return loaded_benchmark.timeout_seconds


def _run_supervised_command(
    command: Sequence[str],
    *,
    timeout_seconds: float | None,
    timeout_profile_out: Path | None = None,
    benchmark_source: str | None = None,
) -> int:
    """Run a subprocess and kill it if the process-level timeout expires."""
    logger.debug("Starting supervised command: %s", " ".join(command))
    started = time.monotonic()
    process = subprocess.Popen(command)
    profiler = (
        None
        if timeout_profile_out is None
        else _Profiler(interval_seconds=0.25, process=psutil.Process(process.pid))
    )
    if profiler is not None:
        profiler.start()
    try:
        return_code = int(process.wait(timeout=timeout_seconds))
        if profiler is not None:
            profiler.stop()
        return return_code
    except subprocess.TimeoutExpired:
        logger.debug("Killing supervised command after timeout.")
        if profiler is not None:
            profiler.stop()
        _kill_process_tree(process)
        elapsed_seconds = time.monotonic() - started
        if profiler is not None and timeout_profile_out is not None:
            _write_timeout_profile_artifacts(
                out_dir=timeout_profile_out,
                profiler=profiler,
                elapsed_seconds=elapsed_seconds,
                timeout_seconds=timeout_seconds,
                benchmark_source=benchmark_source,
            )
        logger.error("Benchmark timed out at %.3f seconds.", elapsed_seconds)
        return SUPERVISOR_TIMEOUT_EXIT_CODE


def _write_timeout_profile_artifacts(
    *,
    out_dir: Path,
    profiler: _Profiler,
    elapsed_seconds: float,
    timeout_seconds: float | None,
    benchmark_source: str | None,
) -> None:
    """Persist all available artifacts after terminating a timed-out run."""
    logger.debug("Writing timeout profile artifacts to %s.", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = profiler.summary()
    summary["profiler_scope"] = "supervised_process_tree"
    _write_jsonl(out_dir / "profile_samples.jsonl", profiler.samples)
    _write_json(out_dir / "profile_summary.json", summary)
    if benchmark_source is not None:
        raw_benchmark, benchmark = _load_benchmark(benchmark_source)
        local_cache_value = os.environ.get("NDB_LOCAL_CACHE")
        local_cache: neurodatabench.models.LocalCacheState | None = None
        if local_cache_value == "cold":
            local_cache = "cold"
        elif local_cache_value == "warm":
            local_cache = "warm"
        implementation = neurodatabench.models.Implementation(
            id=os.environ.get("NDB_IMPLEMENTATION_ID", "unknown"),
            nwb_interface=None,
            object_store_backend=os.environ.get("NDB_OBJECT_STORE_BACKEND"),
            local_cache=local_cache,
            remote_cache=None,
            source_type=os.environ.get("NDB_SOURCE_TYPE"),
        )
        metadata = _run_metadata(
            implementation=implementation,
            benchmark_source=benchmark_source,
            benchmark=benchmark,
            implementation_script_path=None,
        )
        metadata["timed_out"] = True
        if timeout_seconds is not None:
            metadata["timeout_seconds"] = timeout_seconds
        _write_json(out_dir / "benchmark.json", raw_benchmark)
        _write_json(out_dir / "run_metadata.json", metadata)
        _write_json(
            out_dir / "timings.json",
            {"total_duration_ns": round(elapsed_seconds * 1_000_000_000)},
        )
        _write_json(
            out_dir / "validation.json",
            {"correct": False, "timed_out": True},
        )
    _update_results_leaderboard(out_dir)


def _kill_process_tree(process: subprocess.Popen[object]) -> None:
    """Kill a supervised process and any child processes it started."""
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        logger.debug("Supervised process already exited before timeout kill.")
        return

    try:
        targets = [*parent.children(recursive=True), parent]
    except (psutil.Error, PermissionError):
        logger.debug(
            "Unable to enumerate the supervised process tree; killing the direct child.",
            exc_info=True,
        )
        process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.debug("Supervised process did not exit after direct kill.")
        return
    for target in targets:
        try:
            logger.debug("Killing process %d.", target.pid)
            target.kill()
        except psutil.NoSuchProcess:
            logger.debug("Process %d exited before it could be killed.", target.pid)

    _, alive = psutil.wait_procs(targets, timeout=5.0)
    for target in alive:
        try:
            logger.debug("Force-killing lingering process %d.", target.pid)
            target.kill()
        except psutil.NoSuchProcess:
            logger.debug("Lingering process %d exited before retry kill.", target.pid)

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.debug("Supervised process did not exit after process-tree kill.")


def _run_phase_timings(
    *,
    setup_duration_ns: int,
    submit_answers_duration_ns: int,
    total_duration_ns: int,
) -> list[neurodatabench.models.RunPhaseTiming]:
    """Return measured run phases with explicit start and stop times."""
    setup_stop_seconds = setup_duration_ns / 1_000_000_000
    submit_answers_duration_seconds = submit_answers_duration_ns / 1_000_000_000
    submit_answers_stop_seconds = setup_stop_seconds + submit_answers_duration_seconds
    total_stop_seconds = total_duration_ns / 1_000_000_000
    return [
        neurodatabench.models.RunPhaseTiming(
            phase="setup",
            start_seconds=0.0,
            stop_seconds=setup_stop_seconds,
            duration_seconds=setup_stop_seconds,
        ),
        neurodatabench.models.RunPhaseTiming(
            phase="submit_answers",
            start_seconds=setup_stop_seconds,
            stop_seconds=submit_answers_stop_seconds,
            duration_seconds=submit_answers_duration_seconds,
        ),
        neurodatabench.models.RunPhaseTiming(
            phase="total",
            start_seconds=0.0,
            stop_seconds=total_stop_seconds,
            duration_seconds=total_stop_seconds,
        ),
    ]


def _memory_delta(peak_bytes: int | None, baseline_bytes: int | None) -> int | None:
    """Return a non-negative memory delta from a raw peak and baseline."""
    if peak_bytes is None or baseline_bytes is None:
        return None
    return max(peak_bytes - baseline_bytes, 0)


def _resolve_implementation_script_path(
    configured_path: Path | None,
    *,
    setup: Callable[[neurodatabench.models.RunContext], None],
    submit_answers: Callable[[neurodatabench.models.RunContext], None],
) -> Path | None:
    """Return the implementation script path to copy into result artifacts."""
    if configured_path is not None:
        path = configured_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"implementation_script does not exist: {path}")
        logger.debug("Using configured implementation script path %s.", path)
        return path

    inferred_paths = [
        path
        for path in (
            _callable_source_path(setup),
            _callable_source_path(submit_answers),
        )
        if path is not None
    ]
    if not inferred_paths:
        logger.debug("No implementation script path could be inferred.")
        return None

    first_path = inferred_paths[0]
    if any(path != first_path for path in inferred_paths):
        logger.debug(
            "Implementation hooks come from multiple files; using %s.",
            first_path,
        )
    else:
        logger.debug("Inferred implementation script path %s.", first_path)
    return first_path


def _callable_source_path(
    callback: Callable[[neurodatabench.models.RunContext], None],
) -> Path | None:
    """Return the existing Python source path for a callback, if available."""
    try:
        source = inspect.getsourcefile(callback) or inspect.getfile(callback)
    except TypeError:
        logger.debug("Skipping callback without an inspectable source path.")
        return None
    if source.startswith("<"):
        return None
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        logger.debug("Skipping missing callback source path %s.", path)
        return None
    return path


def _run_metadata(
    *,
    implementation: neurodatabench.models.Implementation,
    benchmark_source: str,
    benchmark: neurodatabench.models.Benchmark,
    implementation_script_path: Path | None,
) -> neurodatabench.models.JsonObject:
    """Collect run metadata."""
    metadata: neurodatabench.models.JsonObject = {
        "datetime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "benchmark_harness_version": _package_version("neurodatabench"),
        "implementation": _jsonable(implementation),
        "benchmark": {
            "id": benchmark.id,
            "source": benchmark_source,
            "data_sources": benchmark.data_sources,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "process": {
            "pid": psutil.Process().pid,
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
        },
        "implementation_script": (
            None
            if implementation_script_path is None
            else {
                "source": str(implementation_script_path),
                "artifact": _IMPLEMENTATION_SCRIPT_ARTIFACT_NAME,
            }
        ),
        "requirements": _REQUIREMENTS_ARTIFACT_NAME,
    }
    return metadata


def _write_result_artifacts(
    *,
    out_dir: Path,
    raw_benchmark: neurodatabench.models.JsonObject,
    metadata: neurodatabench.models.JsonObject,
    implementation_script_path: Path | None,
    timings: neurodatabench.models.RunTimings,
    metrics: list[neurodatabench.models.RunMetric],
    validation: neurodatabench.models.JsonObject,
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    run_start_wall_time_ns: int,
) -> None:
    """Write benchmark result artifacts."""
    logger.debug("Writing result artifacts to %s.", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "benchmark.json", raw_benchmark)
    _write_json(out_dir / "run_metadata.json", metadata)
    _write_json(out_dir / "timings.json", timings)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "validation.json", validation)
    _write_json(out_dir / "profile_summary.json", profile_summary)
    _write_jsonl(out_dir / "profile_samples.jsonl", profile_samples)
    _copy_implementation_script(
        source_path=implementation_script_path,
        out_dir=out_dir,
    )
    neurodatabench.plots._write_result_dashboard(
        out_dir=out_dir,
        metadata=metadata,
        timings=timings,
        metrics=metrics,
        profile_samples=profile_samples,
        profile_summary=profile_summary,
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    (out_dir / _REQUIREMENTS_ARTIFACT_NAME).write_text(
        _environment_requirements_text(),
        encoding="utf-8",
    )
    _update_results_leaderboard(out_dir)


def _copy_implementation_script(source_path: Path | None, out_dir: Path) -> None:
    """Copy the implementation script source into a result directory."""
    if source_path is None:
        logger.debug("Skipping implementation script artifact; no source path found.")
        return
    artifact_path = out_dir / _IMPLEMENTATION_SCRIPT_ARTIFACT_NAME
    if source_path == artifact_path.resolve():
        logger.debug("Skipping implementation script artifact copied onto itself.")
        return
    logger.debug(
        "Copying implementation script from %s to %s.",
        source_path,
        artifact_path,
    )
    shutil.copy2(source_path, artifact_path)


def _update_results_leaderboard(out_dir: Path) -> None:
    """Refresh benchmark-specific leaderboard artifacts under a results directory."""
    if out_dir.parent.name == "results":
        results_dir = out_dir.parent
    elif out_dir.parent.parent.name == "results":
        results_dir = out_dir.parent.parent
    else:
        logger.debug("Skipping leaderboard update outside a results directory.")
        return

    current_row = _leaderboard_row(out_dir)
    if current_row is None:
        logger.debug("Skipping leaderboard update because the current run is incomplete.")
        return
    benchmark_id = str(current_row["benchmark_id"])
    leaderboard_dir = results_dir / benchmark_id
    rows = [
        row
        for row in _leaderboard_rows(results_dir)
        if str(row["benchmark_id"]) == benchmark_id
    ]
    if not rows:
        logger.debug("Skipping leaderboard update because no complete runs were found.")
        return

    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    _write_json(leaderboard_dir / "leaderboard.json", rows)
    _write_leaderboard_csv(leaderboard_dir / "leaderboard.csv", rows)
    neurodatabench.plots._write_leaderboard_plot(
        results_dir=results_dir,
        leaderboard_dir=leaderboard_dir,
        rows=rows,
    )


def _leaderboard_rows(results_dir: Path) -> list[neurodatabench.models.JsonObject]:
    """Return leaderboard rows discovered under a results directory."""
    rows: list[neurodatabench.models.JsonObject] = []
    for metadata_path in sorted(results_dir.rglob("run_metadata.json")):
        row = _leaderboard_row(metadata_path.parent)
        if row is not None:
            row["result_dir"] = str(metadata_path.parent.relative_to(results_dir))
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["benchmark_id"]),
            float(row["total_seconds"]),
            str(row["implementation_id"]),
            str(row["datetime_utc"]),
        )
    )
    for row in rows:
        row["leaderboard_label"] = (
            f"{row['implementation_id']} "
        )
    return rows


def _leaderboard_row(run_dir: Path) -> neurodatabench.models.JsonObject | None:
    """Return one leaderboard row for a complete successful or timed-out run."""
    metadata = _read_json_object(run_dir / "run_metadata.json")
    timings = _read_json_object(run_dir / "timings.json")
    validation = _read_json_object(run_dir / "validation.json")
    profile_summary = _read_json_object(run_dir / "profile_summary.json")
    if (
        metadata is None
        or timings is None
        or validation is None
        or profile_summary is None
    ):
        return None
    correct = validation.get("correct") is True
    timed_out = metadata.get("timed_out") is True or validation.get("timed_out") is True
    if not correct and not timed_out:
        return None

    total_duration_ns = _json_number_at(timings, ("total_duration_ns",))
    setup_duration_ns = _json_number_at(timings, ("setup_duration_ns",))
    submit_answers_duration_ns = _json_number_at(
        timings,
        ("submit_answers_duration_ns",),
    )
    if total_duration_ns is None:
        return None
    if not timed_out and (
        setup_duration_ns is None or submit_answers_duration_ns is None
    ):
        return None

    implementation = metadata.get("implementation")
    benchmark = metadata.get("benchmark")
    if not isinstance(implementation, dict) or not isinstance(benchmark, dict):
        return None

    peak_rss_delta_bytes = _json_number_at(
        profile_summary,
        ("peak_process_plus_children_rss_delta_bytes",),
    )
    peak_rss_bytes = _json_number_at(
        profile_summary,
        ("peak_process_plus_children_rss_bytes",),
    )
    comparison_rss_bytes = (
        peak_rss_bytes if peak_rss_delta_bytes is None else peak_rss_delta_bytes
    )
    network_received_bytes = _json_number_at(
        profile_summary,
        ("network_delta", "bytes_recv"),
    )
    row: neurodatabench.models.JsonObject = {
        "result_dir": run_dir.name,
        "leaderboard_label": run_dir.name,
        "datetime_utc": str(metadata.get("datetime_utc", "")),
        "implementation_id": str(implementation.get("id", "unknown")),
        "nwb_interface": implementation.get("nwb_interface"),
        "object_store_backend": implementation.get("object_store_backend"),
        "benchmark_id": str(benchmark.get("id", "unknown")),
        "source_types": _leaderboard_source_type(implementation, benchmark),
        "local_cache": str(implementation.get("local_cache", "")),
        "remote_cache": str(implementation.get("remote_cache", "")),
        "correct": correct,
        "run_status": "timed out" if timed_out else "correct",
        "total_seconds": total_duration_ns / 1_000_000_000,
        "setup_seconds": (
            None if setup_duration_ns is None else setup_duration_ns / 1_000_000_000
        ),
        "submit_answers_seconds": (
            None
            if submit_answers_duration_ns is None
            else submit_answers_duration_ns / 1_000_000_000
        ),
        "peak_rss_delta_mib": (
            None if comparison_rss_bytes is None else comparison_rss_bytes / 1_048_576
        ),
        "peak_rss_mib": (
            None if peak_rss_bytes is None else peak_rss_bytes / 1_048_576
        ),
        "network_received_mib": (
            None
            if network_received_bytes is None
            else network_received_bytes / 1_048_576
        ),
        "timing_segments": _leaderboard_timing_segments(
            timings,
            timed_out=timed_out,
            total_seconds=total_duration_ns / 1_000_000_000,
        ),
    }
    timeout_seconds = metadata.get("timeout_seconds")
    if timed_out:
        row["timed_out"] = True
        if timeout_seconds is not None:
            row["timeout_seconds"] = timeout_seconds
    return row


def _leaderboard_source_type(
    implementation: dict[str, neurodatabench.models.JsonValue],
    benchmark: dict[str, neurodatabench.models.JsonValue],
) -> str:
    """Return the selected data-source type for a leaderboard row.

    Older result metadata does not include the implementation selection, so retain
    the complete benchmark source list as a fallback for those artifacts.
    """
    source_type = implementation.get("source_type")
    if isinstance(source_type, str) and source_type:
        return source_type
    data_sources = benchmark.get("data_sources")
    if isinstance(data_sources, dict):
        return ",".join(sorted(str(source_type) for source_type in data_sources))
    return ""


def _leaderboard_timing_segments(
    timings: neurodatabench.models.JsonObject,
    *,
    timed_out: bool,
    total_seconds: float,
) -> list[neurodatabench.models.JsonObject]:
    """Return setup and per-answer elapsed-time segments for a leaderboard row."""
    phase_timings = timings.get("phase_timings")
    if not isinstance(phase_timings, list):
        return [
            {
                "stage": "truncated (stage unknown)" if timed_out else "total",
                "start_seconds": 0.0,
                "stop_seconds": total_seconds,
                "duration_seconds": total_seconds,
            }
        ]

    segments: list[neurodatabench.models.JsonObject] = []
    submit_phase: dict[str, object] | None = None
    for phase in phase_timings:
        if not isinstance(phase, dict):
            continue
        phase_name = phase.get("phase")
        if phase_name == "setup":
            setup_segment = _leaderboard_timing_segment(phase, stage="setup")
            if setup_segment is not None:
                segments.append(setup_segment)
        elif phase_name == "submit_answers":
            submit_phase = phase

    if submit_phase is None:
        return segments

    submit_start = _json_number_at(submit_phase, ("start_seconds",))
    submit_stop = _json_number_at(submit_phase, ("stop_seconds",))
    if submit_start is None or submit_stop is None:
        return segments

    previous_stop = submit_start
    submissions = timings.get("answer_submissions")
    observed_submission = False
    if isinstance(submissions, list):
        ordered_submissions = sorted(
            (
                submission
                for submission in submissions
                if isinstance(submission, dict)
                and _json_number_at(submission, ("submitted_elapsed_seconds",))
                is not None
            ),
            key=lambda submission: _json_number_at(
                submission,
                ("submitted_elapsed_seconds",),
            )
            or 0.0,
        )
        for submission in ordered_submissions:
            observed_submission = True
            submitted_at = _json_number_at(
                submission,
                ("submitted_elapsed_seconds",),
            )
            assert submitted_at is not None
            stop = min(max(submitted_at, previous_stop), submit_stop)
            segments.append(
                {
                    "stage": str(submission.get("question_id", "answer")),
                    "start_seconds": previous_stop,
                    "stop_seconds": stop,
                    "duration_seconds": stop - previous_stop,
                }
            )
            previous_stop = stop

    if not observed_submission:
        segments.append(
            {
                "stage": (
                    "truncated during answer submission"
                    if timed_out
                    else "answer submission"
                ),
                "start_seconds": submit_start,
                "stop_seconds": submit_stop,
                "duration_seconds": submit_stop - submit_start,
            }
        )
    elif previous_stop < submit_stop and timed_out:
        segments.append(
            {
                "stage": "truncated during answer submission",
                "start_seconds": previous_stop,
                "stop_seconds": submit_stop,
                "duration_seconds": submit_stop - previous_stop,
            }
        )
    elif previous_stop < submit_stop:
        final_segment = segments[-1]
        final_start = _json_number_at(final_segment, ("start_seconds",))
        if final_start is not None:
            final_segment["stop_seconds"] = submit_stop
            final_segment["duration_seconds"] = submit_stop - final_start
    return segments


def _leaderboard_timing_segment(
    phase: dict[str, object],
    *,
    stage: str,
) -> neurodatabench.models.JsonObject | None:
    """Normalize one persisted phase timing into a leaderboard segment."""
    start = _json_number_at(phase, ("start_seconds",))
    stop = _json_number_at(phase, ("stop_seconds",))
    duration = _json_number_at(phase, ("duration_seconds",))
    if start is None or stop is None or duration is None:
        return None
    return {
        "stage": stage,
        "start_seconds": start,
        "stop_seconds": stop,
        "duration_seconds": duration,
    }


def _write_leaderboard_csv(
    path: Path,
    rows: list[neurodatabench.models.JsonObject],
) -> None:
    """Write leaderboard rows as a CSV table."""
    fieldnames = [
        "implementation_id",
        "nwb_interface",
        "object_store_backend",
        "benchmark_id",
        "source_types",
        "local_cache",
        "remote_cache",
        "correct",
        "run_status",
        "total_seconds",
        "setup_seconds",
        "submit_answers_seconds",
        "peak_rss_delta_mib",
        "peak_rss_mib",
        "network_received_mib",
        "datetime_utc",
        "result_dir",
    ]
    if any("timed_out" in row for row in rows):
        fieldnames.insert(fieldnames.index("run_status"), "timed_out")
    if any("timeout_seconds" in row for row in rows):
        fieldnames.insert(fieldnames.index("total_seconds"), "timeout_seconds")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json_object(path: Path) -> neurodatabench.models.JsonObject | None:
    """Read a JSON object from disk, or return None if unavailable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        logger.debug("Skipping unreadable JSON artifact at %s.", path)
        return None
    if not isinstance(value, dict):
        logger.debug("Skipping non-object JSON artifact at %s.", path)
        return None
    return value


def _json_number_at(value: object, keys: Sequence[str]) -> float | None:
    """Return a nested JSON number, if present."""
    current: object = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _write_json(path: Path, value: object) -> None:
    """Write a JSON artifact."""
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[neurodatabench.models.JsonObject],
) -> None:
    """Write newline-delimited JSON rows."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _jsonable(value: object) -> object:
    """Convert dataclasses and paths into JSON-compatible values."""
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _environment_requirements_text() -> str:
    """Return stable requirements-style pins from the current Python environment."""
    rows: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name is None:
            continue
        rows.add(f"{name}=={version}")
    return "\n".join(sorted(rows, key=str.lower)) + "\n"


def _package_version(package_name: str) -> str:
    """Return an installed package version or unknown."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _counter_delta(start: Any, end: Any) -> dict[str, int] | None:
    """Return deltas between two psutil counter snapshots."""
    if start is None or end is None:
        return None
    return {
        field: getattr(end, field) - getattr(start, field) for field in start._fields
    }


def _child_rss_bytes(process: psutil.Process) -> int:
    """Return total RSS bytes for currently live child processes."""
    child_rss = 0
    try:
        children = process.children(recursive=True)
    except (psutil.Error, PermissionError):
        logger.debug(
            "Unable to enumerate child processes during baseline profiling.",
            exc_info=True,
        )
        return child_rss
    for child in children:
        try:
            child_rss += child.memory_info().rss
        except (psutil.Error, PermissionError):
            logger.debug(
                "Skipping a child process that could not be inspected during baseline profiling.",
                exc_info=True,
            )
    return child_rss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(_cli())
