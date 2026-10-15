"""NeuroDataBench benchmark runner package."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

try:
    __version__ = importlib.metadata.version("neurodatabench")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

import neurodatabench.models as models
import neurodatabench.logging_utils as logging_utils
import neurodatabench.validation as validation
from neurodatabench.models import (
    AnswerSubmissionTiming,
    Benchmark,
    Implementation,
    JsonObject,
    JsonPrimitive,
    JsonValue,
    LocalCacheState,
    Question,
    RunContext,
    RunPhaseTiming,
    RunTimings,
    RunMetric,
)

BenchmarkValidationError = validation.BenchmarkValidationError


def __getattr__(name: str) -> Any:
    """Lazily expose runner APIs without importing the runner during package init."""
    if name == "main":
        import neurodatabench.runner

        return neurodatabench.runner.main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnswerSubmissionTiming",
    "Benchmark",
    "BenchmarkValidationError",
    "Implementation",
    "JsonObject",
    "JsonPrimitive",
    "JsonValue",
    "LocalCacheState",
    "Question",
    "RunContext",
    "RunPhaseTiming",
    "RunTimings",
    "RunMetric",
    "__version__",
    "main",
    "get_logger",
    "logging_utils",
    "models",
    "validation",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())

get_logger = logging_utils.get_logger
