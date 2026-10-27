"""Tests for the NeuroDataBench runner harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from datetime import datetime
from pathlib import Path
from typing import Any

import pydantic

import neurodatabench
import neurodatabench.models
import neurodatabench.runner
import neurodatabench.validation


class RunnerTests(unittest.TestCase):
    """Exercise the public runner API and result artifacts."""

    def test_package_root_exposes_public_api(self) -> None:
        """The package root should expose main and model types."""
        self.assertIs(neurodatabench.main, neurodatabench.runner.main)
        self.assertTrue(hasattr(neurodatabench, "models"))
        self.assertTrue(hasattr(neurodatabench, "validation"))
        self.assertIs(neurodatabench.RunContext, neurodatabench.models.RunContext)
        self.assertIs(neurodatabench.Benchmark, neurodatabench.models.Benchmark)
        self.assertIs(
            neurodatabench.RunPhaseTiming, neurodatabench.models.RunPhaseTiming
        )
        self.assertIs(
            neurodatabench.BenchmarkValidationError,
            neurodatabench.validation.BenchmarkValidationError,
        )

    def test_main_runs_packaged_benchmark_by_name(self) -> None:
        """A call-first run should load a packaged benchmark and write artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            seen_contexts: list[neurodatabench.models.RunContext] = []

            def setup(context: neurodatabench.models.RunContext) -> None:
                """Record setup context."""
                seen_contexts.append(context)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit every expected answer from the packaged benchmark."""
                seen_contexts.append(context)
                for question in context.benchmark.questions:
                    context.submit_answer(question.id, question.answer)

            with self.assertLogs("neurodatabench.runner", level="INFO") as logs:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_nwb_interface="pynwb",
                    implementation_object_store_backend="s3fs",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark="dynamic_routing_nwb_v0",
                    out=tmpdir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(len(seen_contexts), 2)
            self.assertIs(seen_contexts[0], seen_contexts[1])
            self.assertEqual(
                seen_contexts[0].benchmark.id,
                "dynamic_routing_nwb_v0",
            )
            self.assertTrue((out_dir / "benchmark.json").exists())
            self.assertTrue((out_dir / "run_metadata.json").exists())
            self.assertTrue((out_dir / "timings.json").exists())
            self.assertTrue((out_dir / "validation.json").exists())
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "profile_samples.jsonl").exists())
            self.assertTrue((out_dir / "profile_summary.json").exists())
            self.assertTrue((out_dir / "dashboard.html").exists())
            self.assertTrue((out_dir / "implementation.py").exists())
            self.assertFalse((out_dir / "timing_summary.html").exists())
            self.assertFalse((out_dir / "memory_profile.html").exists())
            self.assertFalse((out_dir / "cpu_profile.html").exists())
            self.assertTrue((out_dir / "requirements.txt").exists())
            self.assertFalse((out_dir / "results_bundle.zip").exists())
            self.assertFalse((out_dir / "answers.jsonl").exists())
            self.assertRegex(
                "\n".join(logs.output),
                r"Benchmark run completed in \d+\.\d{3} s",
            )

            validation = _read_json(out_dir / "validation.json")
            self.assertTrue(validation["correct"])
            self.assertNotIn("timed_out", validation)
            metadata = _read_json(out_dir / "run_metadata.json")
            self.assertNotIn("timeout_seconds", metadata)
            self.assertNotIn("timed_out", metadata)
            self.assertEqual(metadata["implementation"]["id"], "test-implementation")
            self.assertEqual(metadata["implementation"]["nwb_interface"], "pynwb")
            self.assertEqual(
                metadata["implementation"]["object_store_backend"],
                "s3fs",
            )
            self.assertEqual(
                metadata["implementation_script"],
                {
                    "artifact": "implementation.py",
                    "source": str(Path(__file__).resolve()),
                },
            )
            profile_summary = _read_json(out_dir / "profile_summary.json")
            self.assertIsInstance(profile_summary["baseline_process_rss_bytes"], int)
            self.assertIsInstance(
                profile_summary["baseline_process_plus_children_rss_bytes"],
                int,
            )
            self.assertIsInstance(profile_summary["peak_process_rss_delta_bytes"], int)
            self.assertIsInstance(
                profile_summary["peak_process_plus_children_rss_delta_bytes"],
                int,
            )
            timings = _read_json(out_dir / "timings.json")
            self.assertEqual(
                [row["phase"] for row in timings["phase_timings"]],
                ["setup", "submit_answers", "total"],
            )
            for row in timings["phase_timings"]:
                self.assertLessEqual(row["start_seconds"], row["stop_seconds"])
                self.assertAlmostEqual(
                    row["duration_seconds"],
                    row["stop_seconds"] - row["start_seconds"],
                )
            submitted_ids = [
                row["question_id"] for row in timings["answer_submissions"]
            ]
            self.assertEqual(
                submitted_ids,
                [question.id for question in seen_contexts[0].benchmark.questions],
            )
            for row in timings["answer_submissions"]:
                datetime.fromisoformat(row["submitted_at"])
                self.assertIsInstance(row["submitted_elapsed_seconds"], float)
                self.assertGreaterEqual(row["submitted_elapsed_seconds"], 0.0)

    def test_local_cache_false_is_rejected(self) -> None:
        """False is not a valid local cache metadata state."""
        with self.assertRaisesRegex(
            ValueError,
            "local_cache must be 'cold', 'warm', or None",
        ):
            neurodatabench.models.Implementation(
                id="test-implementation",
                nwb_interface=None,
                object_store_backend=None,
                local_cache=False,
                remote_cache=False,
            )

    def test_clear_cache_runs_before_profiled_setup(self) -> None:
        """The optional cache hook should run before measured phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "cache-hook.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "cache-hook",
                        [{"id": "q", "text": "Q", "answer": 1}],
                    )
                ),
                encoding="utf-8",
            )
            events: list[str] = []
            seen_contexts: list[neurodatabench.models.RunContext] = []

            def clear_cache(context: neurodatabench.models.RunContext) -> None:
                """Record that cache clearing ran before measured work."""
                events.append("clear_cache")
                seen_contexts.append(context)

            def setup(context: neurodatabench.models.RunContext) -> None:
                """Record setup execution."""
                events.append("setup")
                seen_contexts.append(context)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                events.append("submit_answers")
                seen_contexts.append(context)
                context.submit_answer("q", 1)

            original_start = neurodatabench.runner._Profiler.start

            def profiler_start(profiler: neurodatabench.runner._Profiler) -> None:
                """Record the profiler boundary before starting it."""
                events.append("profiler_start")
                original_start(profiler)

            with unittest.mock.patch.object(
                neurodatabench.runner._Profiler,
                "start",
                profiler_start,
            ):
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache="cold",
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=Path(tmpdir) / "results",
                    setup=setup,
                    clear_cache=clear_cache,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(
                events,
                ["clear_cache", "profiler_start", "setup", "submit_answers"],
            )
            self.assertTrue(
                all(context is seen_contexts[0] for context in seen_contexts)
            )

    def test_main_runs_filesystem_benchmark_path(self) -> None:
        """A run should load benchmark JSON from a direct filesystem path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "custom.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json("custom", [{"id": "q", "text": "Q", "answer": 1}])
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=Path(tmpdir) / "results",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )

            validation = _read_json(Path(tmpdir) / "results" / "validation.json")
            self.assertTrue(validation["correct"])

    def test_main_uses_default_output_directory(self) -> None:
        """A run should write to results/<benchmark>/<implementation> when out is omitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "custom.json"
            implementation_dir = Path(tmpdir) / "implementation-run"
            implementation_dir.mkdir()
            implementation_path = implementation_dir / "implementation.py"
            implementation_path.write_text("", encoding="utf-8")
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json("custom", [{"id": "q", "text": "Q", "answer": 1}])
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            with unittest.mock.patch.object(
                neurodatabench.runner.Path,
                "cwd",
                return_value=Path(tmpdir),
            ):
                neurodatabench.runner.main(
                    implementation_id="default-out-test",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    implementation_script=implementation_path,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            out_dir = Path(tmpdir) / "results" / "custom" / "default-out-test"
            validation = _read_json(out_dir / "validation.json")
            self.assertTrue(validation["correct"])
            self.assertTrue((out_dir / "dashboard.html").exists())
            self.assertFalse((implementation_dir / "validation.json").exists())

    def test_results_leaderboard_updates_for_successful_runs(self) -> None:
        """Runs should refresh only their benchmark-specific leaderboard files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "custom.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json("custom", [{"id": "q", "text": "Q", "answer": 1}])
                ),
                encoding="utf-8",
            )
            results_dir = Path(tmpdir) / "results"
            (results_dir / "partial-run").mkdir(parents=True)

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def slow_setup(context: neurodatabench.models.RunContext) -> None:
                """Make one run observably slower."""
                time.sleep(0.01)

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            neurodatabench.runner.main(
                implementation_id="fast",
                implementation_nwb_interface="lazynwb",
                implementation_object_store_backend="s3fs",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=results_dir / "fast-run",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )
            neurodatabench.runner.main(
                implementation_id="slow",
                implementation_nwb_interface="pynwb",
                implementation_object_store_backend="remfile",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=results_dir / "slow-run",
                setup=slow_setup,
                submit_answers=submit_answers,
                argv=(),
            )
            other_benchmark_path = Path(tmpdir) / "other.json"
            other_benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "other",
                        [{"id": "q", "text": "Q", "answer": 1}],
                    )
                ),
                encoding="utf-8",
            )
            neurodatabench.runner.main(
                implementation_id="other-reader",
                benchmark=other_benchmark_path,
                out=results_dir / "other-run",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )

            leaderboard_dir = results_dir / "custom"
            leaderboard = json.loads(
                (leaderboard_dir / "leaderboard.json").read_text(encoding="utf-8")
            )
            self.assertIsInstance(leaderboard, list)
            self.assertEqual(
                [row["implementation_id"] for row in leaderboard], ["fast", "slow"]
            )
            self.assertEqual(
                [row["nwb_interface"] for row in leaderboard],
                ["lazynwb", "pynwb"],
            )
            self.assertEqual(
                [row["object_store_backend"] for row in leaderboard],
                ["s3fs", "remfile"],
            )
            self.assertTrue(all("rank" not in row for row in leaderboard))
            self.assertTrue(all("timed_out" not in row for row in leaderboard))
            self.assertTrue(all("timeout_seconds" not in row for row in leaderboard))
            self.assertTrue((leaderboard_dir / "leaderboard.csv").exists())
            self.assertTrue((leaderboard_dir / "leaderboard.html").exists())
            self.assertFalse((results_dir / "leaderboard.json").exists())
            leaderboard_csv = (leaderboard_dir / "leaderboard.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "implementation_id,nwb_interface,object_store_backend,benchmark_id",
                leaderboard_csv,
            )
            self.assertNotIn("rank", leaderboard_csv.splitlines()[0])
            self.assertNotIn("timed_out", leaderboard_csv.splitlines()[0])
            self.assertNotIn("timeout_seconds", leaderboard_csv.splitlines()[0])
            self.assertTrue(all("peak_rss_delta_mib" in row for row in leaderboard))
            other_leaderboard = json.loads(
                (results_dir / "other" / "leaderboard.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [row["implementation_id"] for row in other_leaderboard],
                ["other-reader"],
            )

    def test_timed_out_leaderboard_row_allows_missing_phase_timings(self) -> None:
        """Killed runs should not need invented setup or submission durations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "timeout-run"
            run_dir.mkdir()
            artifacts = {
                "run_metadata.json": {
                    "datetime_utc": "2026-09-10T00:00:00Z",
                    "timed_out": True,
                    "timeout_seconds": 120,
                    "implementation": {"id": "slow", "nwb_interface": "pynwb"},
                    "benchmark": {"id": "benchmark", "data_sources": {"hdf5": []}},
                },
                "timings.json": {"total_duration_ns": 120_500_000_000},
                "validation.json": {"correct": False, "timed_out": True},
                "profile_summary.json": {},
            }
            for filename, value in artifacts.items():
                (run_dir / filename).write_text(json.dumps(value), encoding="utf-8")

            row = neurodatabench.runner._leaderboard_row(run_dir)

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["total_seconds"], 120.5)
            self.assertIsNone(row["setup_seconds"])
            self.assertIsNone(row["submit_answers_seconds"])
            self.assertTrue(row["timed_out"])
            self.assertEqual(
                row["timing_segments"],
                [
                    {
                        "stage": "truncated (stage unknown)",
                        "start_seconds": 0.0,
                        "stop_seconds": 120.5,
                        "duration_seconds": 120.5,
                    }
                ],
            )

    def test_leaderboard_timing_segments_split_each_answer(self) -> None:
        """Completed timing lanes should expose setup and every answer duration."""
        timings = {
            "phase_timings": [
                {
                    "phase": "setup",
                    "start_seconds": 0.0,
                    "stop_seconds": 2.0,
                    "duration_seconds": 2.0,
                },
                {
                    "phase": "submit_answers",
                    "start_seconds": 2.0,
                    "stop_seconds": 9.0,
                    "duration_seconds": 7.0,
                },
            ],
            "answer_submissions": [
                {"question_id": "first", "submitted_elapsed_seconds": 5.0},
                {"question_id": "second", "submitted_elapsed_seconds": 8.5},
            ],
        }

        segments = neurodatabench.runner._leaderboard_timing_segments(
            timings,
            timed_out=False,
            total_seconds=9.0,
        )

        self.assertEqual(
            [segment["stage"] for segment in segments], ["setup", "first", "second"]
        )
        self.assertEqual(segments[-1]["duration_seconds"], 4.0)

    def test_cli_args_override_call_defaults(self) -> None:
        """Settings CLI overrides should replace benchmark and output defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = Path(tmpdir) / "default.json"
            override_path = Path(tmpdir) / "override.json"
            override_out = Path(tmpdir) / "override-results"
            default_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "default", [{"id": "default", "text": "Q", "answer": 0}]
                    )
                ),
                encoding="utf-8",
            )
            override_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "override", [{"id": "override", "text": "Q", "answer": 7}]
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the answer for the override benchmark."""
                self.assertEqual(context.benchmark.id, "override")
                context.submit_answer("override", 7)

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=default_path,
                out=Path(tmpdir) / "default-results",
                setup=setup,
                submit_answers=submit_answers,
                argv=("--benchmark", str(override_path), "--out", str(override_out)),
            )

            self.assertTrue((override_out / "validation.json").exists())
            self.assertFalse((Path(tmpdir) / "default-results").exists())

    def test_settings_environment_supplies_missing_config(self) -> None:
        """Settings environment variables should supply omitted run config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "env.json"
            out_dir = Path(tmpdir) / "env-results"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json("env", [{"id": "q", "text": "Q", "answer": 1}])
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit the expected answer."""
                context.submit_answer("q", 1)

            with unittest.mock.patch.dict(
                os.environ,
                {
                    "NDB_BENCHMARK": str(benchmark_path),
                    "NDB_OUT": str(out_dir),
                    "NDB_PROFILE_INTERVAL_MS": "10",
                },
            ):
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertTrue((out_dir / "validation.json").exists())
            summary = _read_json(out_dir / "profile_summary.json")
            self.assertEqual(summary["sample_interval_seconds"], 0.01)

    def test_settings_resolves_log_level(self) -> None:
        """Log level should resolve from env, call defaults, and CLI overrides."""
        with unittest.mock.patch.dict(os.environ, {"NDB_LOG_LEVEL": "error"}):
            env_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level=None,
                argv=(),
            )
            call_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level="info",
                argv=(),
            )
            cli_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level="info",
                argv=("--log-level", "debug"),
            )

        self.assertEqual(env_config.log_level, "ERROR")
        self.assertEqual(call_config.log_level, "INFO")
        self.assertEqual(cli_config.log_level, "DEBUG")

    def test_fail_fast_config_resolves_from_call_cli_and_environment(self) -> None:
        """Fail-fast validation should resolve from all settings inputs."""
        with unittest.mock.patch.dict(os.environ, {"NDB_FAIL_FAST": "true"}):
            env_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level=None,
                argv=(),
            )
            call_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level=None,
                default_fail_fast=False,
                argv=(),
            )
            cli_config = neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level=None,
                default_fail_fast=False,
                argv=("--fail-fast",),
            )

        self.assertTrue(env_config.fail_fast)
        self.assertFalse(call_config.fail_fast)
        self.assertTrue(cli_config.fail_fast)

    def test_supervised_timeout_resolves_from_benchmark_override_and_disable(
        self,
    ) -> None:
        """Process supervisor timeouts should resolve outside in-process runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "timeout.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "timeout-config",
                        [{"id": "q", "text": "Q", "answer": 1}],
                        timeout_seconds=120,
                    )
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                neurodatabench.runner._supervised_timeout_seconds(
                    benchmark=str(benchmark_path),
                    timeout_seconds=None,
                    no_timeout=False,
                ),
                120,
            )
            self.assertEqual(
                neurodatabench.runner._supervised_timeout_seconds(
                    benchmark=str(benchmark_path),
                    timeout_seconds=45,
                    no_timeout=False,
                ),
                45,
            )
            self.assertIsNone(
                neurodatabench.runner._supervised_timeout_seconds(
                    benchmark=str(benchmark_path),
                    timeout_seconds=45,
                    no_timeout=True,
                )
            )

    def test_supervised_timeout_reports_actual_elapsed_seconds(self) -> None:
        """Timeout message should report actual elapsed supervisor duration."""

        class FakeProcess:
            """Subprocess test double that times out once, then exits after kill."""

            def __init__(self) -> None:
                """Create an un-killed fake process."""
                self.killed = False
                self.pid = 123
                self.wait_calls = 0

            def wait(self, timeout: float | None = None) -> int:
                """Raise a timeout on the first wait and return after kill."""
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired(["fake"], timeout)
                return -9

            def kill(self) -> None:
                """Record that the process was killed."""
                self.killed = True

        process = FakeProcess()
        with self.assertLogs("neurodatabench.runner", level="ERROR") as logs:
            with unittest.mock.patch.object(
                neurodatabench.runner.subprocess,
                "Popen",
                return_value=process,
            ):
                with unittest.mock.patch.object(
                    neurodatabench.runner,
                    "_kill_process_tree",
                ) as kill_process_tree:
                    with unittest.mock.patch.object(
                        neurodatabench.runner.time,
                        "monotonic",
                        side_effect=[100.0, 112.345],
                    ):
                        return_code = neurodatabench.runner._run_supervised_command(
                            ["fake"],
                            timeout_seconds=10.0,
                        )

        kill_process_tree.assert_called_once_with(process)
        self.assertEqual(
            return_code, neurodatabench.runner.SUPERVISOR_TIMEOUT_EXIT_CODE
        )
        self.assertIn("Benchmark timed out at 12.345 seconds.", "\n".join(logs.output))

    def test_supervised_timeout_kills_process_tree(self) -> None:
        """Timeout cleanup should kill the uv process and descendants."""

        class FakePsutilProcess:
            """psutil.Process test double with recursive children."""

            def __init__(
                self,
                pid: int,
                children: list["FakePsutilProcess"] | None = None,
            ) -> None:
                """Create a fake process tree node."""
                self.pid = pid
                self.killed = False
                self._children = children or []

            def children(self, *, recursive: bool) -> list["FakePsutilProcess"]:
                """Return fake child processes."""
                return self._children

            def kill(self) -> None:
                """Record that the fake process was killed."""
                self.killed = True

        class FakePopen:
            """Popen test double exposing pid and wait."""

            pid = 1

            def __init__(self) -> None:
                """Create a fake Popen process."""
                self.waited = False

            def wait(self, timeout: float | None = None) -> int:
                """Record that the immediate subprocess was waited on."""
                self.waited = True
                return -9

        child = FakePsutilProcess(2)
        parent = FakePsutilProcess(1, [child])
        popen = FakePopen()
        with unittest.mock.patch.object(
            neurodatabench.runner.psutil,
            "Process",
            return_value=parent,
        ):
            with unittest.mock.patch.object(
                neurodatabench.runner.psutil,
                "wait_procs",
                return_value=([], []),
            ) as wait_procs:
                neurodatabench.runner._kill_process_tree(popen)

        self.assertTrue(child.killed)
        self.assertTrue(parent.killed)
        self.assertTrue(popen.waited)
        wait_procs.assert_called_once_with([child, parent], timeout=5.0)

    def test_supervised_timeout_persists_resource_profile(self) -> None:
        """Supervisor-owned samples should survive termination of the child."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_out = Path(tmpdir) / "timeout-result"
            with self.assertLogs("neurodatabench.runner", level="ERROR"):
                return_code = neurodatabench.runner._run_supervised_command(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    timeout_seconds=0.25,
                    timeout_profile_out=profile_out,
                )

            self.assertEqual(
                return_code,
                neurodatabench.runner.SUPERVISOR_TIMEOUT_EXIT_CODE,
            )
            samples = (profile_out / "profile_samples.jsonl").read_text(
                encoding="utf-8"
            )
            summary = _read_json(profile_out / "profile_summary.json")
            self.assertTrue(samples.strip())
            self.assertGreaterEqual(summary["num_samples"], 1)
            self.assertEqual(summary["profiler_scope"], "supervised_process_tree")

    def test_supervised_timeout_creates_leaderboard_row_without_child_artifacts(
        self,
    ) -> None:
        """A killed implementation should remain visible on the leaderboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "id": "timeout-benchmark",
                        "data_sources": {"hdf5": []},
                        "timeout_seconds": 0.05,
                        "questions": [],
                    }
                ),
                encoding="utf-8",
            )
            profile_out = root / "results" / "timeout-run"
            environment = {
                "NDB_IMPLEMENTATION_ID": "timed-out-reader",
                "NDB_OBJECT_STORE_BACKEND": "remfile",
            }
            with (
                unittest.mock.patch.dict(os.environ, environment),
                self.assertLogs("neurodatabench.runner", level="ERROR"),
            ):
                return_code = neurodatabench.runner._run_supervised_command(
                    [sys.executable, "-c", "import time; time.sleep(1)"],
                    timeout_seconds=0.05,
                    timeout_profile_out=profile_out,
                    benchmark_source=str(benchmark_path),
                )

            self.assertEqual(
                return_code,
                neurodatabench.runner.SUPERVISOR_TIMEOUT_EXIT_CODE,
            )
            leaderboard = json.loads(
                (
                    root
                    / "results"
                    / "timeout-benchmark"
                    / "leaderboard.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(leaderboard), 1)
            self.assertEqual(leaderboard[0]["implementation_id"], "timed-out-reader")
            self.assertEqual(leaderboard[0]["run_status"], "timed out")
            self.assertEqual(leaderboard[0]["timeout_seconds"], 0.05)
            self.assertGreater(leaderboard[0]["total_seconds"], 0.0)

    def test_profiler_tolerates_denied_child_process_inspection(self) -> None:
        """Unavailable process-tree metrics should not fail the supervised command."""

        class RestrictedProcess:
            """Process double whose own metrics work but child enumeration is denied."""

            def memory_info(self) -> Any:
                """Return the current process memory snapshot."""
                return type("Memory", (), {"rss": 100, "vms": 200})()

            def cpu_percent(self, interval: float | None = None) -> float:
                """Return a stable CPU measurement."""
                return 0.0

            def num_threads(self) -> int:
                """Return a stable thread count."""
                return 1

            def children(self, *, recursive: bool) -> list[object]:
                """Simulate macOS denying process table inspection."""
                raise PermissionError("process table unavailable")

        profiler = neurodatabench.runner._Profiler(
            interval_seconds=0.01,
            process=RestrictedProcess(),  # type: ignore[arg-type]
        )

        with self.assertLogs("neurodatabench.runner", level="DEBUG"):
            profiler._record_memory_baseline()
            sample = profiler._sample()

        self.assertEqual(profiler._baseline_process_rss_bytes, 100)
        self.assertEqual(profiler._baseline_process_plus_children_rss_bytes, 100)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["children"], {"cpu_percent": 0.0, "rss_bytes": 0})

    def test_invalid_log_level_raises(self) -> None:
        """Invalid log levels should fail during settings validation."""
        with self.assertRaises(pydantic.ValidationError):
            neurodatabench.runner._resolve_config(
                default_benchmark="dynamic_routing_nwb_v0",
                default_out=None,
                default_log_level="verbose",
                argv=(),
            )

    def test_mutable_identity_detection_does_not_apply_to_scalars(self) -> None:
        """Validation should flag reused mutable expected objects but not scalars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "identity.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "identity",
                        [
                            {"id": "list", "text": "Q", "answer": [1, 2]},
                            {"id": "int", "text": "Q", "answer": 1},
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit expected answers directly to exercise diagnostics."""
                for question in context.benchmark.questions:
                    context.submit_answer(question.id, question.answer)

            with self.assertLogs("neurodatabench.runner", level="WARNING") as logs:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=Path(tmpdir) / "results-logged",
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )

            self.assertEqual(len(logs.output), 1)
            self.assertIn("list", logs.output[0])

    def test_validation_failures_exit_with_all_errors(self) -> None:
        """Final validation should exit cleanly with every invalid answer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "bad.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "bad",
                        [
                            {"id": "missing", "text": "Q", "answer": 1},
                            {"id": "duplicate", "text": "Q", "answer": 2},
                            {"id": "wrong", "text": "Q", "answer": 3},
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit intentionally invalid answers."""
                context.submit_answer("duplicate", 2)
                context.submit_answer("duplicate", 2)
                context.submit_answer("wrong", 4)
                context.submit_answer("unknown", 5)

            out_dir = Path(tmpdir) / "results"
            with self.assertRaises(SystemExit) as error:
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=out_dir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )
            message = str(error.exception)
            self.assertIn(
                "Benchmark answers failed validation:",
                message,
            )
            self.assertIn(
                "- missing: missing_answer; submitted_answer=<missing>; actual_answer=1",
                message,
            )
            self.assertIn(
                "- duplicate: duplicate_answer; submitted_answer=[2, 2]; actual_answer=2",
                message,
            )
            self.assertIn(
                "- wrong: exact_int; submitted_answer=4; actual_answer=3",
                message,
            )
            self.assertIn(
                "- unknown: unknown_question_id; submitted_answer=5; "
                "actual_answer=<unknown_question_id>",
                message,
            )
            self.assertFalse((out_dir / "validation.json").exists())
            self.assertFalse(out_dir.exists())

    def test_validation_errors_include_submitted_and_actual_answers(self) -> None:
        """Validation diagnostics should include the compared answer values."""
        scenarios: list[
            tuple[str, dict[str, object], list[dict[str, object]], list[str]]
        ] = [
            (
                "missing",
                {"id": "missing", "text": "Q", "answer": 1},
                [],
                [
                    "missing: missing_answer",
                    "submitted_answer=<missing>",
                    "actual_answer=1",
                ],
            ),
            (
                "duplicate",
                {"id": "duplicate", "text": "Q", "answer": 2},
                [
                    {"question_id": "duplicate", "answer": 2},
                    {"question_id": "duplicate", "answer": 3},
                ],
                [
                    "duplicate: duplicate_answer",
                    "submitted_answer=[2, 3]",
                    "actual_answer=2",
                ],
            ),
            (
                "wrong",
                {"id": "wrong", "text": "Q", "answer": {"value": [1, 2]}},
                [{"question_id": "wrong", "answer": {"value": [1, 3]}}],
                [
                    "wrong: object_value_value_list_item_1_exact_int",
                    'submitted_answer={"value": [1, 3]}',
                    'actual_answer={"value": [1, 2]}',
                ],
            ),
            (
                "unknown",
                {"id": "known", "text": "Q", "answer": True},
                [
                    {"question_id": "known", "answer": True},
                    {"question_id": "unknown", "answer": False},
                ],
                [
                    "unknown: unknown_question_id",
                    "submitted_answer=false",
                    "actual_answer=<unknown_question_id>",
                ],
            ),
        ]

        for benchmark_id, question, submitted_answers, expected_parts in scenarios:
            with self.subTest(benchmark=benchmark_id):
                benchmark = neurodatabench.models.Benchmark.model_validate(
                    _benchmark_json(benchmark_id, [question])
                )
                with self.assertRaises(
                    neurodatabench.validation.BenchmarkValidationError
                ) as error:
                    neurodatabench.validation.validate_answers(
                        benchmark,
                        submitted_answers,
                    )

                message = str(error.exception)
                for expected_part in expected_parts:
                    self.assertIn(expected_part, message)

    def test_fail_fast_validation_raises_during_submission(self) -> None:
        """Fail-fast mode should validate answers as soon as they are submitted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "bad-fast.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "bad-fast",
                        [
                            {"id": "first", "text": "Q", "answer": 1},
                            {"id": "second", "text": "Q", "answer": 2},
                        ],
                    )
                ),
                encoding="utf-8",
            )
            events: list[str] = []

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit one correct answer and then one wrong answer."""
                context.submit_answer("first", 1)
                events.append("after_first")
                context.submit_answer("second", 99)
                events.append("after_wrong")

            out_dir = Path(tmpdir) / "results"
            with self.assertLogs("neurodatabench.runner", level="WARNING") as logs:
                with self.assertRaisesRegex(
                    neurodatabench.runner.BenchmarkValidationError,
                    "second: exact_int; submitted_answer=99; actual_answer=2",
                ):
                    neurodatabench.runner.main(
                        implementation_id="test-implementation",
                        implementation_local_cache=None,
                        implementation_remote_cache=False,
                        benchmark=benchmark_path,
                        out=out_dir,
                        setup=setup,
                        submit_answers=submit_answers,
                        fail_fast=True,
                        argv=(),
                    )

            self.assertEqual(events, ["after_first"])
            self.assertFalse(out_dir.exists())
            self.assertIn("development only", "\n".join(logs.output))

    def test_user_code_failure_does_not_leave_empty_output_directory(self) -> None:
        """A failing implementation should not leave an empty result directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "bad-user-code.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "bad-user-code", [{"id": "q", "text": "Q", "answer": 1}]
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Raise before any artifacts can be written."""
                raise RuntimeError("boom")

            out_dir = Path(tmpdir) / "empty-results"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                neurodatabench.runner.main(
                    implementation_id="test-implementation",
                    implementation_local_cache=None,
                    implementation_remote_cache=False,
                    benchmark=benchmark_path,
                    out=out_dir,
                    setup=setup,
                    submit_answers=submit_answers,
                    argv=(),
                )
            self.assertFalse(out_dir.exists())

    def test_validation_supports_float_lists_and_objects(self) -> None:
        """Validation should compare supported JSON answer shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "shapes.json"
            benchmark_path.write_text(
                json.dumps(
                    _benchmark_json(
                        "shapes",
                        [
                            {"id": "float", "text": "Q", "answer": 1.0},
                            {"id": "list", "text": "Q", "answer": [1, 2]},
                            {"id": "object", "text": "Q", "answer": {"a": 1}},
                        ],
                    )
                ),
                encoding="utf-8",
            )

            def setup(context: neurodatabench.models.RunContext) -> None:
                """No setup required."""

            def submit_answers(context: neurodatabench.models.RunContext) -> None:
                """Submit equivalent answers."""
                context.submit_answer("float", 1.0 + 1e-09)
                context.submit_answer("list", [1, 2])
                context.submit_answer("object", {"a": 1})

            neurodatabench.runner.main(
                implementation_id="test-implementation",
                implementation_local_cache=None,
                implementation_remote_cache=False,
                benchmark=benchmark_path,
                out=Path(tmpdir) / "results",
                setup=setup,
                submit_answers=submit_answers,
                argv=(),
            )

            validation = _read_json(Path(tmpdir) / "results" / "validation.json")
            self.assertTrue(validation["correct"])

    def test_environment_requirements_are_sorted(self) -> None:
        """The environment requirements artifact should contain sorted package pins."""
        requirements_text = neurodatabench.runner._environment_requirements_text()
        rows = [row for row in requirements_text.splitlines() if row]
        self.assertEqual(rows, sorted(rows, key=str.lower))
        self.assertTrue(all("==" in row for row in rows))


def _benchmark_json(
    benchmark_id: str,
    questions: list[dict[str, object]],
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Return a minimal benchmark JSON object."""
    benchmark: dict[str, object] = {
        "id": benchmark_id,
        "data_sources": {"hdf5": ["file:///tmp/test.nwb"]},
        "questions": questions,
    }
    if timeout_seconds is not None:
        benchmark["timeout_seconds"] = timeout_seconds
    return benchmark


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected JSON object")
    return value
