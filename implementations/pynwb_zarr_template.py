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
# neurodatabench = { git = "https://github.com/bjhardcastle/neurodatabench" }
# ///

"""Run a true PyNWB NWBZarrIO materialization benchmark against remote Zarr."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from hdmf_zarr import NWBZarrIO

import neurodatabench

logger = neurodatabench.get_logger(__name__)
state: dict[str, Any] = {}
_DEFAULT_BACKEND = "s3fs"
_DEFAULT_BENCHMARK = "dynamic_routing_nwb_zarr_v0"
_DEFAULT_IMPLEMENTATION_ID = "pynwb_hdmf_zarr_direct"
_FACEMAP_DOWNLOAD_ROWS = 12_850
_FACEMAP_DOWNLOAD_COLUMNS = 128
_RUNNING_SPEED_DOWNLOAD_SAMPLES = 20_000


def setup(context: neurodatabench.RunContext) -> None:
    """Materialize each remote Zarr store as a PyNWB NWBFile."""
    backend = _backend()
    if backend != "s3fs":
        raise ValueError(f"NWBZarrIO does not support configured backend {backend!r}.")
    logger.debug(
        "Opening %d Zarr stores through NWBZarrIO and s3fs.",
        len(context.benchmark.data_sources),
    )
    state.clear()
    state["files"] = []
    try:
        for nwb_path in context.benchmark.data_sources:
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
        _close_open_files()
        raise


def clear_cache(context: neurodatabench.RunContext) -> None:
    """Declare that this implementation has no managed local cache."""
    logger.debug("No PyNWB Zarr cache to clear for %d paths.", len(context.benchmark.data_sources))


def submit_answers(context: neurodatabench.RunContext) -> None:
    """Answer questions through the materialized PyNWB NWBFile objects."""
    files = state["files"]
    for question in context.benchmark.questions:
        logger.debug("Answering benchmark question %s.", question.id)
        match question.id:
            case "multisession_units_metadata_query":
                answer = _count_visp_default_qc(files)
            case "predicated_spike_times":
                answer = _longest_isi_for_fastest_visp_unit(files)
            case "multisession_table_query":
                answer = _multisession_table_query(files)
            case "large_array":
                answer = _large_array(files)
            case "multisession_trials_hit_rate":
                answer = _multisession_trials_hit_rate(files)
            case "max_speed_stimulus":
                answer = _max_speed_stimulus(files[0])
            case "multisession_lick_rate_average":
                answer = _multisession_lick_rate_average(files)
            case "change_detection_large_array":
                answer = _change_detection_large_array(files[0])
            case _:
                raise ValueError(f"Unsupported benchmark question: {question.id}")
        context.submit_answer(question.id, answer)


def teardown(context: neurodatabench.RunContext) -> None:
    """Close NWBZarrIO handles and clear materialized state."""
    logger.debug("Closing NWBZarrIO handles for %d paths.", len(context.benchmark.data_sources))
    _close_open_files()


def _close_open_files() -> None:
    """Close every NWBZarrIO handle and clear implementation state."""
    for file_record in reversed(state.get("files", [])):
        file_record["nwb_io"].close()
    state.clear()


def _backend() -> str:
    """Return the requested NWBZarrIO object-store backend."""
    return os.environ.get("NDB_OBJECT_STORE_BACKEND", _DEFAULT_BACKEND)


def _count_visp_default_qc(file_records: list[dict[str, Any]]) -> int:
    """Count VISp units passing default QC across opened PyNWB NWBFiles."""
    count = 0
    for file_record in file_records:
        if "unit_metrics" not in file_record:
            nwb_file = file_record["nwb_file"]
            if nwb_file.units is None:
                raise ValueError(f"NWBFile {nwb_file.identifier} does not contain units.")
            file_record["units"] = nwb_file.units
            file_record["unit_metrics"] = nwb_file.units.to_dataframe(
                exclude={
                    "spike_times",
                    "spike_amplitudes",
                    "waveform_mean",
                    "waveform_std",
                },
            )
        units = file_record["unit_metrics"]
        structure = _string_array(units["structure"])
        default_qc = np.asarray(units["default_qc"], dtype=np.bool_)
        count += int(np.count_nonzero((structure == "VISp") & default_qc))
    return count


def _longest_isi_for_fastest_visp_unit(file_records: list[dict[str, Any]]) -> float:
    """Return the longest ISI for the VISp unit with highest firing rate."""
    top_file_record: dict[str, Any] | None = None
    top_row = -1
    top_firing_rate = -np.inf
    for file_record in file_records:
        nwb_file = file_record["nwb_file"]
        if "unit_metrics" not in file_record:
            if nwb_file.units is None:
                raise ValueError(f"NWBFile {nwb_file.identifier} does not contain units.")
            file_record["units"] = nwb_file.units
            file_record["unit_metrics"] = nwb_file.units.to_dataframe(
                exclude={
                    "spike_times",
                    "spike_amplitudes",
                    "waveform_mean",
                    "waveform_std",
                },
            )
        units = file_record["unit_metrics"]
        structure = _string_array(units["structure"])
        firing_rate = np.asarray(units["firing_rate"], dtype=np.float64)
        candidate_rows = np.flatnonzero((structure == "VISp") & ~np.isnan(firing_rate))
        if candidate_rows.size == 0:
            logger.debug("No VISp units with firing_rate in %s.", nwb_file.identifier)
            continue
        local_row = int(candidate_rows[np.argmax(firing_rate[candidate_rows])])
        local_rate = float(firing_rate[local_row])
        if local_rate > top_firing_rate:
            top_file_record = file_record
            top_row = local_row
            top_firing_rate = local_rate
    if top_file_record is None:
        raise ValueError("No VISp unit with a finite firing_rate was found.")
    spike_times = np.asarray(
        top_file_record["units"].get_unit_spike_times(top_row),
        dtype=np.float64,
    )
    return float(np.diff(spike_times).max())


def _multisession_table_query(file_records: list[dict[str, Any]]) -> float:
    """Compute the mean trial duration across opened PyNWB NWBFiles."""
    total_duration = 0.0
    total_trials = 0
    for file_record in file_records:
        if "trials_frame" not in file_record:
            nwb_file = file_record["nwb_file"]
            if nwb_file.trials is None:
                raise ValueError(f"NWBFile {nwb_file.identifier} does not contain trials.")
            file_record["trials_frame"] = nwb_file.trials.to_dataframe()
        trials = file_record["trials_frame"]
        start_time = np.asarray(trials["start_time"], dtype=np.float64)
        stop_time = np.asarray(trials["stop_time"], dtype=np.float64)
        total_duration += float(np.sum(stop_time - start_time))
        total_trials += int(start_time.size)
    if total_trials == 0:
        raise ValueError("No trials were found.")
    return total_duration / total_trials


def _string_array(values: Any) -> np.ndarray:
    """Convert a table column to a NumPy string array."""
    raw_values = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_values
        ],
        dtype=str,
    )


def _large_array(file_records: list[dict[str, Any]]) -> float:
    """Return the mean of a 6.6 MB facemap block from the first NWBFile."""
    if not file_records:
        raise ValueError("At least one NWBFile record is required.")
    nwb_file = file_records[0]["nwb_file"]
    facemap = nwb_file.processing["behavior"]["facemap_side_camera"]
    data = np.asarray(
        facemap.data[:_FACEMAP_DOWNLOAD_ROWS, :_FACEMAP_DOWNLOAD_COLUMNS],
        dtype=np.float32,
    )
    return float(np.mean(data, dtype=np.float64))


def _multisession_trials_hit_rate(file_records: list[dict[str, Any]]) -> float:
    """Fraction of `intervals/trials` rows that were hits across all sessions."""
    total_trials = 0
    total_hits = 0

    for file_record in file_records:
        if "trials_frame" not in file_record:
            nwb_file = file_record["nwb_file"]
            if nwb_file.trials is None:
                raise ValueError(
                    f"NWBFile {nwb_file.identifier} does not contain trials.",
                )
            file_record["trials_frame"] = nwb_file.trials.to_dataframe()
        trials = file_record["trials_frame"]
        hit = np.asarray(trials["hit"], dtype=np.bool_)
        total_hits += int(np.count_nonzero(hit))
        total_trials += int(hit.size)

    return float(total_hits / total_trials)


def _max_speed_stimulus(first_file_record: dict[str, Any]) -> str:
    """Stimulus with the highest mean running speed in the first session."""
    nwb_file = first_file_record["nwb_file"]
    stim = nwb_file.intervals["stimulus_presentations"].to_dataframe()
    image_name = _string_array(stim["image_name"])
    start_time = np.asarray(stim["start_time"], dtype=np.float64)
    stop_time = np.asarray(stim["stop_time"], dtype=np.float64)

    speed_series = nwb_file.processing["running"]["speed"]
    speed_data = np.asarray(speed_series.data[:], dtype=np.float64)
    speed_ts = np.asarray(speed_series.timestamps[:], dtype=np.float64)

    # Assign each speed sample to the presentation whose start_time most
    # recently precedes it (asof match), then keep only samples whose timestamp
    # is also inside that presentation's [start_time, stop_time] interval.
    order = np.argsort(start_time)
    sorted_starts = start_time[order]
    sorted_stops = stop_time[order]
    sorted_labels = image_name[order]

    candidate = np.searchsorted(sorted_starts, speed_ts, side="right") - 1
    within = candidate >= 0
    clipped = np.where(within, candidate, 0)
    within &= speed_ts <= sorted_stops[clipped]

    matched_labels = sorted_labels[clipped]

    best_label = ""
    best_mean = -np.inf
    for label in np.unique(matched_labels[within]):
        mask = within & (matched_labels == label)
        mean_speed = float(speed_data[mask].mean())
        if mean_speed > best_mean:
            best_mean = mean_speed
            best_label = str(label)
    return best_label


def _multisession_lick_rate_average(file_records: list[dict[str, Any]]) -> float:
    """Highest per-session mean lick rate (licks / second) across sessions."""
    best_rate = -np.inf
    for file_record in file_records:
        nwb_file = file_record["nwb_file"]
        events = nwb_file.events["events"].to_dataframe()
        event_type = _string_array(events["event_type"])
        timestamps = np.asarray(events["timestamp"], dtype=np.float64)

        n_licks = int(np.count_nonzero(event_type == "lick"))
        duration = float(timestamps.max() - timestamps.min())

        rate = n_licks / duration
        if rate > best_rate:
            best_rate = rate
    return float(best_rate)


def _change_detection_large_array(first_file_record: dict[str, Any]) -> float:
    """Mean of the first 20,000 samples of `processing/running/speed/data`."""
    nwb_file = first_file_record["nwb_file"]
    speed_series = nwb_file.processing["running"]["speed"]
    data = np.asarray(
        speed_series.data[:_RUNNING_SPEED_DOWNLOAD_SAMPLES],
        dtype=np.float64,
    )
    return float(np.mean(data, dtype=np.float64))


if __name__ == "__main__":
    neurodatabench.main(
        implementation_id=os.environ.get(
            "NDB_IMPLEMENTATION_ID",
            f"{_DEFAULT_IMPLEMENTATION_ID}_{_backend()}",
        ),
        implementation_nwb_interface="pynwb/NWBZarrIO",
        implementation_object_store_backend=_backend(),
        implementation_local_cache=None,
        implementation_remote_cache=False,
        benchmark=os.environ.get("NDB_BENCHMARK", _DEFAULT_BENCHMARK),
        setup=setup,
        clear_cache=clear_cache,
        submit_answers=submit_answers,
        teardown=teardown,
    )
