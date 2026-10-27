"""Altair dashboard generation for NeuroDataBench result artifacts."""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path

import altair as alt

import neurodatabench.models

alt.data_transformers.disable_max_rows()

logger = logging.getLogger(__name__)

_LEADERBOARD_SERIES_COLORS = [
    "#059669",
    "#d97706",
    "#e11d48",
    "#7c3aed",
    "#2563eb",
    "#0f766e",
]
_DASHBOARD_COLORS: dict[str, str] = {
    "timing": "#d97706",
    "metrics": "#ff8ab7",
    "memory": "#e11d48",
    "cpu_process": "#059669",
    "cpu_system": "#7c3aed",
    "network_received": "#0f766e",
    "event_setup": "#64748b",
    "event_answer": "#f97316",
    "empty_state_text": "#374151",
}


def _write_result_dashboard(
    *,
    out_dir: Path,
    metadata: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    metrics: dict[str, neurodatabench.models.RunMetric],
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    run_start_wall_time_ns: int,
) -> None:
    """Write the Altair dashboard for one benchmark result directory."""
    logger.debug("Writing result dashboard to %s.", out_dir)
    dashboard = _result_dashboard_chart(
        metadata=metadata,
        timings=timings,
        metrics=metrics,
        profile_samples=profile_samples,
        profile_summary=profile_summary,
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    _save_altair_chart(out_dir / "dashboard.html", dashboard)


def _write_leaderboard_plot(
    *,
    results_dir: Path,
    leaderboard_dir: Path,
    rows: list[neurodatabench.models.JsonObject],
) -> None:
    """Write one benchmark's leaderboard plot using runs from a results tree."""
    logger.debug("Writing leaderboard plot to %s.", leaderboard_dir)
    if not rows:
        return

    memory_rows, network_rows = _leaderboard_profile_rows(
        results_dir=results_dir,
        leaderboard_rows=rows,
    )
    metrics_rows = _leaderboard_metrics_rows(
        results_dir=results_dir,
        leaderboard_rows=rows,
    )
    chart = _leaderboard_plot_chart(
        rows,
        metrics_rows=metrics_rows,
        memory_profile_rows=memory_rows,
        network_profile_rows=network_rows,
    )
    _save_altair_chart(leaderboard_dir / "leaderboard.html", chart)


def _leaderboard_plot_chart(
    rows: list[neurodatabench.models.JsonObject],
    *,
    metrics_rows: list[dict[str, object]] | None = None,
    memory_profile_rows: list[dict[str, object]] | None = None,
    network_profile_rows: list[dict[str, object]] | None = None,
) -> alt.VConcatChart:
    """Return stage timing, metrics, memory, and network comparisons for one benchmark."""
    benchmark_ids = {str(row.get("benchmark_id", "unknown")) for row in rows}
    if len(benchmark_ids) != 1:
        raise ValueError("A leaderboard chart must contain exactly one benchmark")
    benchmark_id = next(iter(benchmark_ids))
    elapsed_limit = _leaderboard_elapsed_limit(rows)
    timing_rows = _leaderboard_timing_rows(rows, elapsed_limit=elapsed_limit)
    profile_series = list(
        dict.fromkeys(str(row["profile_series"]) for row in timing_rows)
    )
    series_colors = [
        _LEADERBOARD_SERIES_COLORS[index % len(_LEADERBOARD_SERIES_COLORS)]
        for index in range(len(profile_series))
    ]
    timing_chart = _leaderboard_timing_chart(
        rows=rows,
        timing_rows=timing_rows,
        elapsed_limit=elapsed_limit,
    )
    metrics_rows = metrics_rows or []
    metrics_charts = [
        _leaderboard_metrics_chart(
            title=metric_name,
            rows=[row for row in metrics_rows if row["metric"] == metric_name],
            x_title=next(
                str(row["x_label"])
                for row in metrics_rows
                if row["metric"] == metric_name
            ),
            y_title=next(
                str(row["y_label"])
                for row in metrics_rows
                if row["metric"] == metric_name
            ),
            profile_series=profile_series,
            series_colors=series_colors,
        )
        for metric_name in dict.fromkeys(str(row["metric"]) for row in metrics_rows)
    ]
    metrics_block = (
        [alt.vconcat(*metrics_charts, spacing=14).resolve_scale(x="independent")]
        if metrics_charts
        else []
    )
    memory_chart = _leaderboard_comparison_profile_chart(
        title="Memory profile",
        rows=memory_profile_rows or [],
        y_field="memory_delta_mib",
        y_title="MiB above baseline",
        profile_series=profile_series,
        series_colors=series_colors,
    )
    network_chart = _leaderboard_comparison_profile_chart(
        title="Network profile (system received)",
        rows=network_profile_rows or [],
        y_field="network_received_mib",
        y_title="MiB received since first sample",
        profile_series=profile_series,
        series_colors=series_colors,
    )
    return (
        alt.vconcat(
            timing_chart,
            *metrics_block,
            memory_chart,
            network_chart,
            spacing=14,
        )
        .resolve_scale(x="shared", color="independent")
        .properties(
            title=alt.TitleParams(
                text=f"NeuroDataBench Leaderboard: {benchmark_id}",
                anchor="start",
            )
        )
    )


def _leaderboard_timing_rows(
    rows: list[neurodatabench.models.JsonObject],
    *,
    elapsed_limit: float | None = None,
) -> list[dict[str, object]]:
    """Flatten persisted setup and per-answer segments for timing lanes."""
    timing_rows: list[dict[str, object]] = []
    for row in rows:
        implementation_id = str(row.get("implementation_id", "unknown"))
        source_types = str(row.get("source_types", "unknown"))
        profile_series = f"{implementation_id} [{source_types}]"
        segments = row.get("timing_segments")
        if not isinstance(segments, list):
            total_seconds = _leaderboard_number_at(row, "total_seconds") or 0.0
            segments = [
                {
                    "stage": "total",
                    "start_seconds": 0.0,
                    "stop_seconds": total_seconds,
                    "duration_seconds": total_seconds,
                }
            ]
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            start_seconds = _object_number_at(segment, "start_seconds")
            stop_seconds = _object_number_at(segment, "stop_seconds")
            duration_seconds = _object_number_at(segment, "duration_seconds")
            if (
                start_seconds is None
                or stop_seconds is None
                or duration_seconds is None
            ):
                continue
            is_last_segment = index == len(segments) - 1
            timing_rows.append(
                {
                    "leaderboard_label": str(row["leaderboard_label"]),
                    "profile_series": profile_series,
                    "implementation_id": implementation_id,
                    "benchmark_id": str(row.get("benchmark_id", "unknown")),
                    "run_status": str(row.get("run_status", "unknown")),
                    "stage": str(segment.get("stage", "unknown")),
                    "start_seconds": start_seconds,
                    "stop_seconds": stop_seconds,
                    "duration_seconds": duration_seconds,
                    "annotation_seconds": (
                        min(stop_seconds, elapsed_limit)
                        if elapsed_limit is not None
                        else stop_seconds
                    ),
                    "elapsed_label": (
                        _elapsed_label(row, elapsed_limit=elapsed_limit)
                        if is_last_segment
                        else ""
                    ),
                }
            )
    return timing_rows


def _leaderboard_timing_chart(
    *,
    rows: list[neurodatabench.models.JsonObject],
    timing_rows: list[dict[str, object]],
    elapsed_limit: float | None,
) -> alt.LayerChart:
    """Return aligned timing lanes broken into setup and individual answers."""
    y_sort = [str(row["leaderboard_label"]) for row in rows]
    stage_order = list(dict.fromkeys(str(row["stage"]) for row in timing_rows))
    stage_colors = [
        _LEADERBOARD_SERIES_COLORS[index % len(_LEADERBOARD_SERIES_COLORS)]
        for index in range(len(stage_order))
    ]
    y_encoding = alt.Y(
        "leaderboard_label:N",
        title=None,
        sort=y_sort,
        axis=alt.Axis(labelLimit=280),
    )
    source = alt.Data(values=timing_rows)
    x_scale = alt.Scale(domain=[0.0, elapsed_limit]) if elapsed_limit else alt.Scale()
    bars = (
        alt.Chart()
        .mark_bar(size=15, stroke="#ffffff", strokeWidth=0.7, clip=True)
        .encode(
            x=alt.X("start_seconds:Q", title="elapsed seconds", scale=x_scale),
            x2="stop_seconds:Q",
            y=y_encoding,
            color=alt.Color(
                "stage:N",
                title="stage",
                scale=alt.Scale(domain=stage_order, range=stage_colors),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=2,
                    labelLimit=400,
                ),
            ),
            tooltip=[
                alt.Tooltip("implementation_id:N", title="Implementation"),
                alt.Tooltip("benchmark_id:N", title="Benchmark"),
                alt.Tooltip("run_status:N", title="Run status"),
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("start_seconds:Q", title="Start", format=",.3f"),
                alt.Tooltip("stop_seconds:Q", title="Stop", format=",.3f"),
                alt.Tooltip("duration_seconds:Q", title="Duration", format=",.3f"),
            ],
        )
        .add_params(
            alt.selection_interval(
                name="leaderboard_zoom",
                bind="scales",
                encodings=["x"],
            )
        )
    )
    elapsed_labels = (
        alt.Chart()
        .transform_filter("datum.elapsed_label !== ''")
        .mark_text(
            align="right",
            baseline="middle",
            color="#991b1b",
            dx=-6,
            fontSize=11,
            fontWeight="bold",
        )
        .encode(
            x=alt.X("annotation_seconds:Q"),
            y=y_encoding,
            text=alt.Text("elapsed_label:N"),
        )
    )
    return alt.layer(bars, elapsed_labels, data=source).properties(
        title="Stage durations",
        width=904,
        height=max(60, min(22 * len(rows), 720)),
    )


def _leaderboard_metrics_rows(
    *,
    results_dir: Path,
    leaderboard_rows: list[neurodatabench.models.JsonObject],
) -> list[dict[str, object]]:
    """Load each run's metrics.json as chart rows, one per metric data point."""
    metric_rows: list[dict[str, object]] = []
    for leaderboard_row in leaderboard_rows:
        run_dir = results_dir / str(leaderboard_row["result_dir"])
        try:
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Skipping unreadable metrics.json in %s.", run_dir)
            continue
        if not isinstance(metrics, dict):
            continue
        profile_series = (
            f"{leaderboard_row['implementation_id']} "
            f"[{leaderboard_row['source_types']}]"
        )
        for metric_name, metric in metrics.items():
            if not isinstance(metric, dict):
                continue
            x_data = metric.get("x_data")
            y_data = metric.get("y_data")
            if not isinstance(x_data, list) or not isinstance(y_data, list):
                continue
            previous_stop_seconds = 0.0
            n_points = min(len(x_data), len(y_data))
            # Thin dense series for plotting; cumulative time still sums every point.
            step = 100 if n_points >= 200 else 10 if n_points >= 20 else 1
            logger.debug(
                "Plotting every %d of %d points for %s / %s.",
                step,
                n_points,
                profile_series,
                metric_name,
            )
            for m, (x_value, y_value) in enumerate(zip(x_data, y_data)):
                load_elapsed_seconds = float(y_value or 0.0)
                stop_seconds = previous_stop_seconds + load_elapsed_seconds
                if m % step == 0 or m == n_points - 1:
                    metric_rows.append(
                        {
                            "profile_series": profile_series,
                            "metric": str(metric_name),
                            "x_label": str(metric.get("x_label", "")),
                            "y_label": str(metric.get("y_label", "")),
                            "x": x_value,
                            "start_seconds": previous_stop_seconds,
                            "stop_seconds": stop_seconds,
                            "elapsed_seconds": load_elapsed_seconds,
                        }
                    )
                previous_stop_seconds = stop_seconds
    return metric_rows


def _leaderboard_metrics_chart(
    *,
    title: str,
    rows: list[dict[str, object]],
    x_title: str,
    y_title: str,
    profile_series: list[str],
    series_colors: list[str],
) -> alt.Chart:
    """Return one metric plot containing every implementation trace."""
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "x:Q",
                title=x_title,
            ),
            y=alt.Y(
                "stop_seconds:Q",
                title=y_title,
            ),
            color=alt.Color(
                "profile_series:N",
                title="implementation [format]",
                scale=alt.Scale(domain=profile_series, range=series_colors),
                legend=alt.Legend(labelLimit=350),
            ),
            tooltip=[
                alt.Tooltip("profile_series:N", title="Implementation"),
                alt.Tooltip(
                    "elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"
                ),
                alt.Tooltip("x:Q", title=x_title),
                alt.Tooltip("stop_seconds:Q", title="Load end time", format=",.3f"),
            ],
        )
        .properties(title=title, width=904, height=190)
    )


def _leaderboard_comparison_profile_chart(
    *,
    title: str,
    rows: list[dict[str, object]],
    y_field: str,
    y_title: str,
    profile_series: list[str],
    series_colors: list[str],
) -> alt.Chart:
    """Return one resource axis containing every implementation trace."""
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_line(clip=True)
        .encode(
            x=alt.X("elapsed_seconds:Q", title="elapsed seconds"),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color(
                "profile_series:N",
                title="implementation [format]",
                scale=alt.Scale(domain=profile_series, range=series_colors),
                legend=alt.Legend(labelLimit=350),
            ),
            tooltip=[
                alt.Tooltip("profile_series:N", title="Implementation"),
                alt.Tooltip(
                    "elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"
                ),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=",.3f"),
            ],
        )
        .properties(title=title, width=904, height=190)
    )


def _leaderboard_profile_rows(
    *,
    results_dir: Path,
    leaderboard_rows: list[neurodatabench.models.JsonObject],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load per-run samples as shared memory and network profile rows."""
    memory_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    for leaderboard_row in leaderboard_rows:
        run_dir = results_dir / str(leaderboard_row["result_dir"])
        samples = _read_jsonl_objects(run_dir / "profile_samples.jsonl")
        if not samples:
            continue
        profile_summary = _read_json_object(run_dir / "profile_summary.json") or {}
        profile_series = (
            f"{leaderboard_row['implementation_id']} "
            f"[{leaderboard_row['source_types']}]"
        )
        sample_times = [
            time_ns
            for sample in samples
            for time_ns in [_json_number_at(sample, ("time_ns",))]
            if time_ns is not None
        ]
        if not sample_times:
            continue
        first_time_ns = min(sample_times)
        baseline_rss = _json_number_at(
            profile_summary,
            ("baseline_process_plus_children_rss_bytes",),
        )
        baseline_received: float | None = None
        for sample in samples:
            time_ns = _json_number_at(sample, ("time_ns",))
            if time_ns is None:
                continue
            elapsed_seconds = max(0.0, (time_ns - first_time_ns) / 1_000_000_000)
            process_rss = _json_number_at(sample, ("process", "rss_bytes"))
            child_rss = _json_number_at(sample, ("children", "rss_bytes")) or 0.0
            if process_rss is not None:
                total_rss = process_rss + child_rss
                if baseline_rss is None:
                    baseline_rss = total_rss
                memory_rows.append(
                    {
                        "profile_series": profile_series,
                        "elapsed_seconds": elapsed_seconds,
                        "memory_delta_mib": max(total_rss - baseline_rss, 0.0)
                        / 1_048_576,
                    }
                )
            bytes_received = _json_number_at(sample, ("io", "net", "bytes_recv"))
            if bytes_received is not None:
                if baseline_received is None:
                    baseline_received = bytes_received
                network_rows.append(
                    {
                        "profile_series": profile_series,
                        "elapsed_seconds": elapsed_seconds,
                        "network_received_mib": max(
                            bytes_received - baseline_received,
                            0.0,
                        )
                        / 1_048_576,
                    }
                )
    return memory_rows, network_rows


def _read_jsonl_objects(path: Path) -> list[neurodatabench.models.JsonObject]:
    """Return valid JSON object rows from a JSON Lines artifact."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[neurodatabench.models.JsonObject] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_json_object(path: Path) -> neurodatabench.models.JsonObject | None:
    """Return one JSON object artifact when it is readable."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _object_number_at(value: dict[str, object], key: str) -> float | None:
    """Return a JSON-like numeric dictionary field."""
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        return None
    return float(item)


def _elapsed_label(
    row: neurodatabench.models.JsonObject,
    *,
    elapsed_limit: float | None,
) -> str:
    """Return the timeout or visual-truncation annotation for one timing lane."""
    total_seconds = _leaderboard_number_at(row, "total_seconds")
    visually_truncated = (
        elapsed_limit is not None
        and total_seconds is not None
        and total_seconds > elapsed_limit
    )
    if not _leaderboard_timed_out(row):
        return f"{total_seconds:g} s total" if visually_truncated else ""
    timeout_seconds = _leaderboard_number_at(row, "timeout_seconds")
    if timeout_seconds is None:
        label = "timed out"
    else:
        label = f"timed out at {timeout_seconds:g} s"
    if visually_truncated and total_seconds is not None:
        label += f"; {total_seconds:g} s elapsed"
    return label


def _leaderboard_elapsed_limit(
    rows: list[neurodatabench.models.JsonObject],
) -> float | None:
    """Return a robust initial x-axis limit when elapsed times contain outliers."""
    elapsed = sorted(
        value
        for row in rows
        for value in [_leaderboard_number_at(row, "total_seconds")]
        if value is not None and value > 0
    )
    if len(elapsed) < 3:
        return None
    median = statistics.median(elapsed)
    mad = statistics.median(abs(value - median) for value in elapsed)
    threshold = median + 6.0 * mad if mad > 0 else median * 3.0
    if threshold <= 0 or elapsed[-1] <= threshold:
        return None
    return threshold


def _leaderboard_timed_out(row: neurodatabench.models.JsonObject) -> bool:
    """Return whether a leaderboard row represents a timed-out run."""
    timed_out = row.get("timed_out")
    if isinstance(timed_out, bool):
        return timed_out
    if isinstance(timed_out, str):
        return timed_out.strip().lower() == "true"
    run_status = row.get("run_status")
    return isinstance(run_status, str) and run_status.strip().lower() == "timed out"


def _leaderboard_number_at(
    row: neurodatabench.models.JsonObject,
    key: str,
) -> float | None:
    """Return a leaderboard number from JSON-native or CSV-like row data."""
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            return None
    return None


def _result_dashboard_chart(
    *,
    metadata: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    metrics: dict[str, neurodatabench.models.RunMetric],
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    run_start_wall_time_ns: int,
) -> alt.VConcatChart:
    """Return a single dashboard chart for benchmark timing and profiling."""
    time_domain = _elapsed_time_domain(
        timings=timings,
        profile_samples=profile_samples,
        run_start_wall_time_ns=run_start_wall_time_ns,
    )
    metrics_charts = [
        _metrics_summary_chart(metric=v, metric_name=k)
        for k, v in metrics.items()
    ]
    metrics_block = (
        [alt.vconcat(*metrics_charts, spacing=10).resolve_scale(x="independent")]
        if metrics_charts
        else []
    )
    dashboard = (
        alt.vconcat(
            _timing_summary_chart(timings, time_domain=time_domain),
            *metrics_block,
            _network_profile_chart(
                profile_samples=profile_samples,
                timings=timings,
                run_start_wall_time_ns=run_start_wall_time_ns,
                time_domain=time_domain,
            ),
            _memory_profile_chart(
                profile_samples=profile_samples,
                profile_summary=profile_summary,
                timings=timings,
                run_start_wall_time_ns=run_start_wall_time_ns,
                time_domain=time_domain,
            ),
            _cpu_profile_chart(
                profile_samples=profile_samples,
                timings=timings,
                run_start_wall_time_ns=run_start_wall_time_ns,
                time_domain=time_domain,
            ),
            spacing=10,
        )
        .resolve_scale(
            x="shared",
            color="independent",
        )
        .properties(title=_dashboard_title(metadata))
    )
    return dashboard


def _dashboard_title(metadata: neurodatabench.models.JsonObject) -> alt.TitleParams:
    """Return a dashboard title that identifies the benchmark run."""
    implementation_id = _metadata_string_at(
        metadata,
        ("implementation", "id"),
        default="unknown implementation",
    )
    benchmark_id = _metadata_string_at(
        metadata,
        ("benchmark", "id"),
        default="unknown benchmark",
    )
    subtitle_parts = [
        _metadata_label_value(metadata, "datetime_utc", "UTC"),
        _metadata_label_value(metadata, "hostname", "host"),
        _metadata_label_value(metadata, "benchmark_harness_version", "harness"),
        _metadata_label_value(
            metadata,
            ("implementation", "nwb_interface"),
            "NWB interface",
        ),
        _metadata_label_value(
            metadata,
            ("implementation", "object_store_backend"),
            "object store",
        ),
        _metadata_label_value(
            metadata,
            ("implementation", "local_cache"),
            "local cache",
        ),
        _metadata_label_value(
            metadata,
            ("implementation", "remote_cache"),
            "remote cache",
        ),
    ]
    subtitle = " | ".join(part for part in subtitle_parts if part is not None)
    return alt.TitleParams(
        text=f"NeuroDataBench: {implementation_id} / {benchmark_id}",
        subtitle=subtitle,
        anchor="start",
        fontSize=20,
        offset=16,
    )


def _metadata_label_value(
    metadata: neurodatabench.models.JsonObject,
    key_path: str | tuple[str, ...],
    label: str,
) -> str | None:
    """Return one compact metadata label/value string for the dashboard title."""
    value = _metadata_string_at(metadata, key_path)
    if value is None:
        return None
    return f"{label} {value}"


def _metadata_string_at(
    metadata: neurodatabench.models.JsonObject,
    key_path: str | tuple[str, ...],
    *,
    default: str | None = None,
) -> str | None:
    """Return a nested metadata value as a string, if present."""
    keys = (key_path,) if isinstance(key_path, str) else key_path
    value: object = metadata
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    if value is None:
        return default
    return str(value)


def _timing_summary_chart(
    timings: neurodatabench.models.RunTimings,
    *,
    time_domain: tuple[float, float],
) -> alt.Chart:
    """Return an Altair horizontal start/stop chart of benchmark timing phases."""
    rows = _timing_summary_rows(timings)
    phase_order = list(dict.fromkeys(str(row["phase"]) for row in rows))
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(
            color=_DASHBOARD_COLORS["timing"],
            size=8,
            stroke="#ffffff",
            strokeWidth=0.8,
        )
        .encode(
            x=alt.X(
                "start_seconds:Q",
                title="elapsed seconds",
                scale=alt.Scale(domain=list(time_domain)),
            ),
            x2="stop_seconds:Q",
            y=alt.Y("phase:N", title=None, sort=phase_order),
            tooltip=[
                alt.Tooltip("phase:N", title="Phase"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("start_seconds:Q", title="Start", format=",.3f"),
                alt.Tooltip("stop_seconds:Q", title="Stop", format=",.3f"),
                alt.Tooltip("duration_seconds:Q", title="Duration", format=",.3f"),
            ],
        )
        .properties(title="Benchmark Timing", width=904, height=48)
    )
    return chart


def _metrics_summary_chart(
    *,
    metric: neurodatabench.models.RunMetric,
    metric_name: str,
    width: int = 440,
    height: int = 300,
) -> alt.Chart:
    """Return an Altair chart summarizing a single metric."""
    rows: list[dict[str, object]] = []
    previous_stop_seconds = 0
    n_points = len(metric.y_data)
    for m in range(n_points):
        stop_seconds = float(metric.y_data[m])
        elapsed_seconds = stop_seconds - previous_stop_seconds
        rows.append(
            {
                "x": metric.x_data[m],
                "start_seconds": previous_stop_seconds,
                "stop_seconds": stop_seconds,
                "elapsed_seconds": elapsed_seconds,
                "metric": f"{metric_name}",
            }
        )
        previous_stop_seconds = stop_seconds

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "x:Q",
                title=f"{metric.x_label}",
            ),
            y=alt.Y(
                "stop_seconds:Q", 
                title=f"{metric.y_label}",
            ),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=[f"{metric_name}"],
                    range=[_DASHBOARD_COLORS["metrics"]]
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"
                ),
                alt.Tooltip("x:Q", title=f"{metric.x_label}"),
                alt.Tooltip("stop_seconds:Q", title="Load end time", format=",.3f"),
            ],
        )
        .properties(title=metric_name, width=width, height=height)
    )
    return chart


def _timing_summary_rows(
    timings: neurodatabench.models.RunTimings,
) -> list[dict[str, object]]:
    """Return phase timing rows, splitting submit answers by question submission."""
    rows: list[dict[str, object]] = []
    for phase_timing in timings.phase_timings:
        if phase_timing.phase == "total":
            continue
        if phase_timing.phase != "submit_answers":
            rows.append(_phase_timing_row(phase_timing, segment=phase_timing.phase))
            continue

        submit_rows = _submit_answer_timing_rows(phase_timing, timings)
        rows.extend(submit_rows)

    return rows


def _phase_timing_row(
    phase_timing: neurodatabench.models.RunPhaseTiming,
    *,
    segment: str,
) -> dict[str, object]:
    """Return a chart row for one timing segment."""
    return {
        "phase": phase_timing.phase,
        "segment": segment,
        "start_seconds": phase_timing.start_seconds,
        "stop_seconds": phase_timing.stop_seconds,
        "duration_seconds": phase_timing.duration_seconds,
    }


def _submit_answer_timing_rows(
    phase_timing: neurodatabench.models.RunPhaseTiming,
    timings: neurodatabench.models.RunTimings,
) -> list[dict[str, object]]:
    """Return submit-answer phase rows split by observed answer submissions."""
    submissions = sorted(
        (
            submission
            for submission in timings.answer_submissions
            if submission.submitted_elapsed_seconds is not None
        ),
        key=lambda submission: float(submission.submitted_elapsed_seconds or 0.0),
    )
    if not submissions:
        return [_phase_timing_row(phase_timing, segment=phase_timing.phase)]

    rows: list[dict[str, object]] = []
    previous_stop_seconds = phase_timing.start_seconds
    for submission in submissions:
        submitted_elapsed_seconds = float(submission.submitted_elapsed_seconds or 0.0)
        stop_seconds = min(
            max(submitted_elapsed_seconds, previous_stop_seconds),
            phase_timing.stop_seconds,
        )
        rows.append(
            {
                "phase": phase_timing.phase,
                "segment": submission.question_id,
                "start_seconds": previous_stop_seconds,
                "stop_seconds": stop_seconds,
                "duration_seconds": stop_seconds - previous_stop_seconds,
            }
        )
        previous_stop_seconds = stop_seconds

    if previous_stop_seconds < phase_timing.stop_seconds:
        rows.append(
            {
                "phase": phase_timing.phase,
                "segment": "after final submission",
                "start_seconds": previous_stop_seconds,
                "stop_seconds": phase_timing.stop_seconds,
                "duration_seconds": phase_timing.stop_seconds - previous_stop_seconds,
            }
        )

    return rows


def _memory_profile_chart(
    *,
    profile_samples: list[neurodatabench.models.JsonObject],
    profile_summary: neurodatabench.models.JsonObject,
    timings: neurodatabench.models.RunTimings,
    run_start_wall_time_ns: int,
    time_domain: tuple[float, float],
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart of baseline-subtracted memory usage."""
    baseline_rss = _json_number_at(
        profile_summary,
        ("baseline_process_plus_children_rss_bytes",),
    )
    rows: list[dict[str, object]] = []
    for sample, elapsed_seconds in _profile_samples_with_elapsed_time(
        profile_samples,
        run_start_wall_time_ns,
    ):
        process_rss = _json_number_at(sample, ("process", "rss_bytes"))
        child_rss = _json_number_at(sample, ("children", "rss_bytes")) or 0.0
        if process_rss is None:
            continue
        total_rss = process_rss + child_rss
        if baseline_rss is None:
            baseline_rss = total_rss
        rows.append(
            {
                "elapsed_seconds": elapsed_seconds,
                "metric": "RSS delta (process + children)",
                "value_mib": max(total_rss - baseline_rss, 0.0) / 1_048_576,
            }
        )

    chart = _profile_line_chart(
        title="Memory Profile (baseline-subtracted)",
        rows=rows,
        y_field="value_mib",
        y_title="MiB above baseline",
        color_domain=["RSS delta (process + children)"],
        color_range=[_DASHBOARD_COLORS["memory"]],
        event_markers=_profile_event_marker_rows(timings),
        time_domain=time_domain,
        width=904,
        height=125,
    )
    return chart


def _cpu_profile_chart(
    *,
    profile_samples: list[neurodatabench.models.JsonObject],
    timings: neurodatabench.models.RunTimings,
    run_start_wall_time_ns: int,
    time_domain: tuple[float, float],
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart of sampled CPU usage."""
    rows: list[dict[str, object]] = []
    for sample, elapsed_seconds in _profile_samples_with_elapsed_time(
        profile_samples,
        run_start_wall_time_ns,
    ):
        process_cpu = _json_number_at(sample, ("process", "cpu_percent"))
        child_cpu = _json_number_at(sample, ("children", "cpu_percent")) or 0.0
        system_cpu = _json_number_at(sample, ("system", "cpu_percent"))
        if process_cpu is not None:
            rows.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "metric": "process + children",
                    "value_percent": process_cpu + child_cpu,
                }
            )
        if system_cpu is not None:
            rows.append(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "metric": "system",
                    "value_percent": system_cpu,
                }
            )

    chart = _profile_line_chart(
        title="CPU Profile",
        rows=rows,
        y_field="value_percent",
        y_title="percent",
        color_domain=["process + children", "system"],
        color_range=[
            _DASHBOARD_COLORS["cpu_process"],
            _DASHBOARD_COLORS["cpu_system"],
        ],
        event_markers=_profile_event_marker_rows(timings),
        time_domain=time_domain,
        width=904,
        height=125,
    )
    return chart


def _network_profile_chart(
    *,
    profile_samples: list[neurodatabench.models.JsonObject],
    timings: neurodatabench.models.RunTimings,
    run_start_wall_time_ns: int,
    time_domain: tuple[float, float],
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart of sampled system network usage."""
    baseline_received = None
    rows: list[dict[str, object]] = []
    for sample, elapsed_seconds in _profile_samples_with_elapsed_time(
        profile_samples,
        run_start_wall_time_ns,
    ):
        bytes_received = _json_number_at(sample, ("io", "net", "bytes_recv"))
        if bytes_received is None:
            continue
        if baseline_received is None:
            baseline_received = bytes_received
        received_delta = max(bytes_received - baseline_received, 0.0)
        rows.append(
            {
                "elapsed_seconds": elapsed_seconds,
                "metric": "received",
                "value_mib": received_delta / 1_048_576,
            }
        )

    chart = _profile_line_chart(
        title="Network Profile (system)",
        rows=rows,
        y_field="value_mib",
        y_title="MiB received since first sample",
        color_domain=["received"],
        color_range=[_DASHBOARD_COLORS["network_received"]],
        event_markers=_profile_event_marker_rows(timings),
        time_domain=time_domain,
        width=904,
        height=125,
    )
    return chart


def _profile_line_chart(
    *,
    title: str,
    rows: list[dict[str, object]],
    y_field: str,
    y_title: str,
    color_domain: Sequence[str],
    color_range: Sequence[str],
    event_markers: list[dict[str, object]],
    time_domain: tuple[float, float],
    width: int = 440,
    height: int = 300,
) -> alt.Chart | alt.LayerChart:
    """Return an Altair line chart for sampled profile rows."""
    if not rows:
        return (
            alt.Chart(alt.Data(values=[{"message": "No profile samples collected"}]))
            .mark_text(size=16, color=_DASHBOARD_COLORS["empty_state_text"])
            .encode(text="message:N")
            .properties(title=title, width=width, height=height)
        )

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "elapsed_seconds:Q",
                title="elapsed seconds",
                scale=alt.Scale(domain=list(time_domain)),
            ),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=list(color_domain), range=list(color_range)),
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"
                ),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip(f"{y_field}:Q", title=y_title, format=",.3f"),
            ],
        )
        .properties(title=title, width=width, height=height)
    )
    return _annotate_profile_chart(
        chart,
        event_markers=event_markers,
    )


def _annotate_profile_chart(
    chart: alt.Chart,
    *,
    event_markers: list[dict[str, object]],
) -> alt.Chart | alt.LayerChart:
    """Add hover-only event markers to a profile chart."""
    if not event_markers:
        return chart

    marker_source = alt.Data(values=event_markers)
    rules = (
        alt.Chart(marker_source)
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.5)
        .encode(
            x=alt.X("elapsed_seconds:Q"),
            color=alt.Color(
                "event:N",
                legend=None,
                scale=alt.Scale(
                    domain=["setup complete", "answer submitted"],
                    range=[
                        _DASHBOARD_COLORS["event_setup"],
                        _DASHBOARD_COLORS["event_answer"],
                    ],
                ),
            ),
            tooltip=[
                alt.Tooltip("event:N", title="Event"),
                alt.Tooltip("detail:N", title="Detail"),
                alt.Tooltip(
                    "elapsed_seconds:Q", title="Elapsed seconds", format=",.3f"
                ),
            ],
        )
    )
    return alt.layer(chart, rules).resolve_scale(color="independent")


def _elapsed_time_domain(
    *,
    timings: neurodatabench.models.RunTimings,
    profile_samples: list[neurodatabench.models.JsonObject],
    run_start_wall_time_ns: int,
) -> tuple[float, float]:
    """Return the shared elapsed-time domain for all dashboard panes."""
    elapsed_values = [
        max(0.0, phase_timing.stop_seconds) for phase_timing in timings.phase_timings
    ]
    elapsed_values.extend(
        elapsed_seconds
        for _, elapsed_seconds in _profile_samples_with_elapsed_time(
            profile_samples,
            run_start_wall_time_ns,
        )
    )
    max_elapsed = max(elapsed_values, default=0.0)
    return 0.0, max(max_elapsed, 0.001)


def _profile_event_marker_rows(
    timings: neurodatabench.models.RunTimings,
) -> list[dict[str, object]]:
    """Return setup and answer submission marker rows for profile charts."""
    rows: list[dict[str, object]] = []
    for phase_timing in timings.phase_timings:
        if phase_timing.phase != "setup":
            continue
        rows.append(
            {
                "event": "setup complete",
                "detail": "setup",
                "elapsed_seconds": phase_timing.stop_seconds,
            }
        )
        break
    for answer_submission in timings.answer_submissions:
        elapsed_seconds = answer_submission.submitted_elapsed_seconds
        if elapsed_seconds is None:
            continue
        rows.append(
            {
                "event": "answer submitted",
                "detail": answer_submission.question_id,
                "elapsed_seconds": elapsed_seconds,
            }
        )
    return rows


def _save_altair_chart(path: Path, chart: alt.TopLevelMixin) -> None:
    """Save an Altair chart as an HTML artifact."""
    chart.configure_axis(
        labelColor="#4b5563",
        titleColor="#374151",
    ).configure_view(
        stroke=None
    ).save(path.as_posix())


def _profile_samples_with_elapsed_time(
    profile_samples: Sequence[neurodatabench.models.JsonObject],
    run_start_wall_time_ns: int,
) -> list[tuple[neurodatabench.models.JsonObject, float]]:
    """Return profile samples paired with elapsed seconds from run start."""
    rows: list[tuple[neurodatabench.models.JsonObject, float]] = []
    for sample in profile_samples:
        time_ns = _json_number_at(sample, ("time_ns",))
        if time_ns is None:
            continue
        elapsed_seconds = (time_ns - run_start_wall_time_ns) / 1_000_000_000
        if elapsed_seconds < 0:
            continue
        rows.append((sample, elapsed_seconds))
    return rows


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
