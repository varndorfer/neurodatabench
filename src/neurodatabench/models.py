"""Typed data structures for NeuroDataBench benchmarks and runs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal, TypeAlias

import pydantic

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = pydantic.JsonValue
JsonObject: TypeAlias = dict[str, JsonValue]
LocalCacheState: TypeAlias = Literal["cold", "warm"]


@dataclasses.dataclass(slots=True, frozen=True)
class Question:
    """Question info passed to implementation code: the answer should be derived from data and submitted to the benchmark runner."""

    id: str
    text: str
    answer: JsonValue
    """The expected answer"""


class Benchmark(pydantic.BaseModel):
    """A full benchmark specification, including all questions and expected answers, as well as any additional metadata needed to run the benchmark end-to-end."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    data_sources: dict[str, list[str]]
    timeout_seconds: float | None = pydantic.Field(default=None, gt=0)
    questions: list[Question]


@dataclasses.dataclass(slots=True, frozen=True)
class Implementation:
    """Metadata about a particular implementation of the benchmark that can't be obtained programatically."""

    id: str
    nwb_interface: str | None
    """The NWB API or interface used to answer benchmark questions, if any"""
    object_store_backend: str | None
    """The object-store access backend used by the implementation, if any"""
    local_cache: LocalCacheState | None
    """The state of the implementation's local cache, if any"""
    remote_cache: bool | None
    """Whether the implementation depends on a pre-computed cache object ranges (e.g. kerchunk/lindi)"""
    source_type: str | None = None
    """The benchmark data-source format selected by the implementation, if any."""

    def __post_init__(self) -> None:
        """Reject local cache metadata outside the public cold/warm/none states."""
        if self.local_cache not in ("cold", "warm", None):
            raise ValueError("local_cache must be 'cold', 'warm', or None")


@dataclasses.dataclass(slots=True)
class RunContext:
    """Runner-owned context passed to benchmark implementation hooks."""

    benchmark: Benchmark
    submit_answer: Callable[[str, JsonValue], None]
    run_metrics: dict[str, RunMetric]


@dataclasses.dataclass(slots=True)
class RunMetric:
    """A single metric result for the current run."""
    x_data: list[int]
    x_label: str
    y_data: list[int]
    y_label: str


@dataclasses.dataclass(slots=True, frozen=True)
class RunPhaseTiming:
    """Start, stop, and duration timing for one measured run phase."""

    phase: str
    start_seconds: float
    stop_seconds: float
    duration_seconds: float


@dataclasses.dataclass(slots=True, frozen=True)
class AnswerSubmissionTiming:
    """Wall-clock timing for one submitted answer."""

    question_id: str
    submitted_at: str
    submitted_elapsed_seconds: float | None


class RunTimings(pydantic.BaseModel):
    """Timing summary for setup and answer submission."""

    model_config = pydantic.ConfigDict(frozen=True)

    setup_duration_ns: int
    submit_answers_duration_ns: int
    total_duration_ns: int
    phase_timings: list[RunPhaseTiming]
    answer_submissions: list[AnswerSubmissionTiming]
