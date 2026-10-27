"""Tests for NeuroDataBench plot generation helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import neurodatabench.models
import neurodatabench.plots
import neurodatabench.runner


class PlotTests(unittest.TestCase):
    """Exercise chart-specific data shaping behavior."""

    def test_leaderboard_source_type_uses_implementation_selection(self) -> None:
        """A run's legend format should not list unrelated benchmark formats."""
        source_type = neurodatabench.runner._leaderboard_source_type(
            {"source_type": "zarr"},
            {"data_sources": {"escapewheel": [], "hdf5": [], "zarr": []}},
        )

        self.assertEqual(source_type, "zarr")

    def test_leaderboard_source_type_falls_back_for_old_metadata(self) -> None:
        """Old results should retain their previous complete-source labeling."""
        source_type = neurodatabench.runner._leaderboard_source_type(
            {},
            {"data_sources": {"zarr": [], "hdf5": []}},
        )

        self.assertEqual(source_type, "hdf5,zarr")

    def test_leaderboard_plot_chart_has_stage_and_resource_axes(self) -> None:
        """Leaderboard should align stage timing, memory, and network panels."""
        chart = neurodatabench.plots._leaderboard_plot_chart(
            [
                {
                    "implementation_id": "slow",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "benchmark_id": "benchmark",
                    "leaderboard_label": "slow",
                    "total_seconds": 24.25,
                }
            ]
        )

        spec = chart.to_dict()

        self.assertNotIn("subtitle", spec["title"])
        self.assertEqual(len(spec["vconcat"]), 3)
        timing_spec = spec["vconcat"][0]
        self.assertEqual(
            timing_spec["layer"][0]["encoding"]["x"]["title"],
            "elapsed seconds",
        )
        tooltip_fields = [
            tooltip["field"]
            for tooltip in timing_spec["layer"][0]["encoding"]["tooltip"]
        ]
        self.assertIn("stage", tooltip_fields)
        self.assertIn("duration_seconds", tooltip_fields)
        self.assertIn("run_status", tooltip_fields)
        self.assertEqual(
            spec["vconcat"][1]["encoding"]["y"]["field"],
            "memory_delta_mib",
        )
        self.assertEqual(
            spec["vconcat"][2]["encoding"]["y"]["field"],
            "network_received_mib",
        )

    def test_leaderboard_chart_is_scoped_to_one_benchmark(self) -> None:
        """A leaderboard should be an unfaceted chart for exactly one benchmark."""
        chart = neurodatabench.plots._leaderboard_plot_chart(
            [
                {
                    "implementation_id": implementation_id,
                    "source_types": "hdf5",
                    "benchmark_id": "benchmark-a",
                    "leaderboard_label": implementation_id,
                    "total_seconds": 1.0,
                }
                for implementation_id in ("fast", "slow")
            ]
        )

        spec = chart.to_dict()
        timing_spec = spec["vconcat"][0]

        self.assertEqual(
            spec["title"]["text"],
            "NeuroDataBench Leaderboard: benchmark-a",
        )
        self.assertNotIn("facet", timing_spec)
        self.assertIn("data", timing_spec)
        self.assertNotIn("data", timing_spec["layer"][0])
        self.assertNotIn("data", timing_spec["layer"][1])

    def test_leaderboard_plot_chart_labels_truncated_timing_lane(self) -> None:
        """Timed-out leaderboard lanes should display their truncation cutoff."""
        chart = neurodatabench.plots._leaderboard_plot_chart(
            [
                {
                    "implementation_id": "timeout",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "benchmark_id": "benchmark",
                    "leaderboard_label": "timeout",
                    "timed_out": True,
                    "total_seconds": 60.0,
                    "timeout_seconds": 60.0,
                }
            ]
        )

        chart.to_dict()
        timing_values = neurodatabench.plots._leaderboard_timing_rows(
            [
                {
                    "implementation_id": "timeout",
                    "benchmark_id": "benchmark",
                    "leaderboard_label": "timeout",
                    "source_types": "hdf5",
                    "timed_out": True,
                    "total_seconds": 60.0,
                    "timeout_seconds": 60.0,
                }
            ]
        )
        self.assertEqual(timing_values[0]["elapsed_label"], "timed out at 60 s")

    def test_leaderboard_limits_outlier_but_preserves_full_elapsed_label(self) -> None:
        """A long run should not squash typical runs on the initial x-axis view."""
        rows = [
            {
                "implementation_id": implementation_id,
                "benchmark_id": "benchmark",
                "leaderboard_label": implementation_id,
                "source_types": "hdf5",
                "total_seconds": elapsed,
            }
            for implementation_id, elapsed in (
                ("fast", 10.0),
                ("typical", 12.0),
                ("outlier", 300.0),
            )
        ]

        limit = neurodatabench.plots._leaderboard_elapsed_limit(rows)
        self.assertIsNotNone(limit)
        assert limit is not None
        self.assertLess(limit, 300.0)
        timing_rows = neurodatabench.plots._leaderboard_timing_rows(
            rows,
            elapsed_limit=limit,
        )
        outlier = next(
            row for row in timing_rows if row["implementation_id"] == "outlier"
        )
        self.assertEqual(outlier["stop_seconds"], 300.0)
        self.assertEqual(outlier["annotation_seconds"], limit)
        self.assertEqual(outlier["elapsed_label"], "300 s total")

        spec = neurodatabench.plots._leaderboard_plot_chart(rows).to_dict()
        self.assertEqual(
            spec["vconcat"][0]["layer"][0]["encoding"]["x"]["scale"]["domain"],
            [0.0, limit],
        )
        self.assertTrue(spec["params"])

    def test_leaderboard_profile_rows_overlay_run_samples(self) -> None:
        """Memory and network rows should retain implementation identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            run_dir = results_dir / "run"
            run_dir.mkdir()
            samples = [
                {
                    "time_ns": 1_000_000_000,
                    "process": {"rss_bytes": 100},
                    "children": {"rss_bytes": 20},
                    "io": {"net": {"bytes_recv": 1_000}},
                },
                {
                    "time_ns": 2_000_000_000,
                    "process": {"rss_bytes": 220},
                    "children": {"rss_bytes": 40},
                    "io": {"net": {"bytes_recv": 2_048_000}},
                },
            ]
            (run_dir / "profile_samples.jsonl").write_text(
                "\n".join(json.dumps(sample) for sample in samples),
                encoding="utf-8",
            )
            (run_dir / "profile_summary.json").write_text(
                json.dumps({"baseline_process_plus_children_rss_bytes": 120}),
                encoding="utf-8",
            )

            memory_rows, network_rows = neurodatabench.plots._leaderboard_profile_rows(
                results_dir=results_dir,
                leaderboard_rows=[
                    {
                        "result_dir": "run",
                        "implementation_id": "implementation",
                        "source_types": "zarr",
                    }
                ],
            )

            self.assertEqual(memory_rows[-1]["profile_series"], "implementation [zarr]")
            self.assertEqual(memory_rows[-1]["elapsed_seconds"], 1.0)
            self.assertGreater(memory_rows[-1]["memory_delta_mib"], 0)
            self.assertEqual(network_rows[-1]["elapsed_seconds"], 1.0)
            self.assertGreater(network_rows[-1]["network_received_mib"], 1.0)

    def _metric_rows_for(self, runs: dict[str, str | None]) -> list[dict[str, object]]:
        """Build metric rows for runs whose metrics.json text is given (None writes no file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            leaderboard_rows = []
            for name, metrics_text in runs.items():
                run_dir = results_dir / name
                run_dir.mkdir()
                if metrics_text is not None:
                    (run_dir / "metrics.json").write_text(metrics_text, encoding="utf-8")
                leaderboard_rows.append(
                    {
                        "result_dir": name,
                        "implementation_id": name,
                        "source_types": "zarr",
                    }
                )
            return neurodatabench.plots._leaderboard_metrics_rows(
                results_dir=results_dir,
                leaderboard_rows=leaderboard_rows,
            )

    def test_leaderboard_metrics_rows_accumulate_and_label_series(self) -> None:
        """Short series keep every point with cumulative stop times and series labels."""
        metrics = {
            "Load": {
                "x_data": [0, 1, 2],
                "y_data": [0, 1.5, 2.5],
                "x_label": "tables loaded",
                "y_label": "load time (s)",
            }
        }

        rows = self._metric_rows_for({"impl": json.dumps(metrics)})

        self.assertEqual([row["x"] for row in rows], [0, 1, 2])
        self.assertEqual([row["stop_seconds"] for row in rows], [0.0, 1.5, 4.0])
        self.assertEqual(rows[1]["start_seconds"], 0.0)
        self.assertEqual(rows[1]["elapsed_seconds"], 1.5)
        self.assertEqual(rows[0]["profile_series"], "impl [zarr]")
        self.assertEqual(rows[0]["metric"], "Load")
        self.assertEqual(rows[0]["x_label"], "tables loaded")
        self.assertEqual(rows[0]["y_label"], "load time (s)")

    def test_leaderboard_metrics_rows_thin_dense_series_and_keep_last_point(self) -> None:
        """Series are thinned per run, keep the final point, and still sum every value."""
        cases = {19: 19, 25: 4, 200: 3}
        for n_points, expected_rows in cases.items():
            with self.subTest(n_points=n_points):
                metrics = {"Load": {"x_data": list(range(n_points)), "y_data": [1.0] * n_points}}

                rows = self._metric_rows_for({"impl": json.dumps(metrics)})

                self.assertEqual(len(rows), expected_rows)
                self.assertEqual(rows[-1]["x"], n_points - 1)
                self.assertEqual(rows[-1]["stop_seconds"], float(n_points))

        thinned = self._metric_rows_for(
            {"impl": json.dumps({"Load": {"x_data": list(range(25)), "y_data": [1.0] * 25}})}
        )
        self.assertEqual([row["x"] for row in thinned], [0, 10, 20, 24])

        dense = self._metric_rows_for(
            {"impl": json.dumps({"Load": {"x_data": list(range(200)), "y_data": [1.0] * 200}})}
        )
        self.assertEqual([row["x"] for row in dense], [0, 100, 199])

    def test_dashboard_title_includes_stack_metadata_when_present(self) -> None:
        """Run dashboard subtitles should include implementation stack metadata."""
        title = neurodatabench.plots._dashboard_title(
            {
                "datetime_utc": "2026-05-13T00:00:00Z",
                "hostname": "host",
                "benchmark_harness_version": "1.2.3",
                "benchmark": {
                    "id": "benchmark",
                    "source_types": "hdf5",
                },
                "implementation": {
                    "id": "implementation",
                    "nwb_interface": "pynwb",
                    "object_store_backend": "remfile",
                    "local_cache": None,
                    "remote_cache": False,
                },
            }
        ).to_dict()

        self.assertEqual(
            title["text"],
            "NeuroDataBench: implementation / benchmark",
        )
        self.assertNotIn("source types", title.get("subtitle", ""))
        self.assertIn("NWB interface pynwb", title["subtitle"])
        self.assertIn("object store remfile", title["subtitle"])

    def test_timing_summary_rows_split_submit_answers_by_question(self) -> None:
        """Submit-answer timing should show one segment per submitted question."""
        timings = neurodatabench.models.RunTimings(
            setup_duration_ns=1,
            submit_answers_duration_ns=8,
            total_duration_ns=9,
            phase_timings=[
                neurodatabench.models.RunPhaseTiming(
                    phase="setup",
                    start_seconds=0.0,
                    stop_seconds=1.0,
                    duration_seconds=1.0,
                ),
                neurodatabench.models.RunPhaseTiming(
                    phase="submit_answers",
                    start_seconds=1.0,
                    stop_seconds=9.0,
                    duration_seconds=8.0,
                ),
                neurodatabench.models.RunPhaseTiming(
                    phase="total",
                    start_seconds=0.0,
                    stop_seconds=9.0,
                    duration_seconds=9.0,
                ),
            ],
            answer_submissions=[
                neurodatabench.models.AnswerSubmissionTiming(
                    question_id="q1",
                    submitted_at="2026-05-14T00:00:02Z",
                    submitted_elapsed_seconds=3.0,
                ),
                neurodatabench.models.AnswerSubmissionTiming(
                    question_id="q2",
                    submitted_at="2026-05-14T00:00:05Z",
                    submitted_elapsed_seconds=7.0,
                ),
            ],
        )

        rows = neurodatabench.plots._timing_summary_rows(timings)

        self.assertEqual(
            [(row["phase"], row["segment"]) for row in rows],
            [
                ("setup", "setup"),
                ("submit_answers", "q1"),
                ("submit_answers", "q2"),
                ("submit_answers", "after final submission"),
            ],
        )
        self.assertEqual(rows[1]["start_seconds"], 1.0)
        self.assertEqual(rows[1]["stop_seconds"], 3.0)
        self.assertEqual(rows[2]["duration_seconds"], 4.0)
        self.assertEqual(rows[3]["duration_seconds"], 2.0)

    def test_timing_summary_rows_keep_submit_phase_when_answers_missing(self) -> None:
        """Runs without per-answer timings should keep the aggregate submit row."""
        timings = neurodatabench.models.RunTimings(
            setup_duration_ns=1,
            submit_answers_duration_ns=8,
            total_duration_ns=9,
            phase_timings=[
                neurodatabench.models.RunPhaseTiming(
                    phase="setup",
                    start_seconds=0.0,
                    stop_seconds=1.0,
                    duration_seconds=1.0,
                ),
                neurodatabench.models.RunPhaseTiming(
                    phase="submit_answers",
                    start_seconds=1.0,
                    stop_seconds=9.0,
                    duration_seconds=8.0,
                ),
            ],
            answer_submissions=[],
        )

        rows = neurodatabench.plots._timing_summary_rows(timings)

        self.assertEqual(
            [(row["phase"], row["segment"]) for row in rows],
            [("setup", "setup"), ("submit_answers", "submit_answers")],
        )


if __name__ == "__main__":
    unittest.main()
