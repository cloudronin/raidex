"""Base classes for Raidex benchmark wrappers.

Each Tier A benchmark subclasses ``Benchmark`` and returns a ``BenchmarkResult``
carrying the benchmark-native ``value`` (e.g. BBQ 0-1, WMDP 0-100). Normalization,
dimension mapping, composite scoring, and badge assignment are owned by
``scoring.py`` — wrappers stay dumb and report only native units plus raw detail.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkResult:
    benchmark_id: str                 # "bbq", "wmdp", ...
    value: Optional[float]            # benchmark-native units; None if the run errored
    raw: dict = field(default_factory=dict)
    judge_model: Optional[str] = None
    eval_source: str = "automated"    # "automated" | "published"
    n_samples: Optional[int] = None   # prompts actually scored (distinguishes smoke runs)
    error: Optional[str] = None       # set instead of raising, so one bench can't sink the run
    n_failed: Optional[int] = None    # calls that failed after retries (DLQ trigger)
    sample_errors: list = field(default_factory=list)  # a few example post-retry error strings


class Benchmark(ABC):
    """Abstract Tier A benchmark.

    Subclasses set the class attributes and implement ``run``/``estimate_cost``.
    ``model_id`` is always a litellm string, e.g. ``"openai/gpt-5.2"``.
    """

    id: str          # benchmark id, matches keys in scoring.NORM
    dimension: str   # RAI dimension, e.g. "fairness_bias"
    prompts: int     # full-set prompt count, for cost estimation

    @abstractmethod
    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        """Run against ``model_id``. If ``limit`` is set, score only the first
        ``limit`` prompts (smoke test)."""
        ...

    @abstractmethod
    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        """Estimate API cost in USD, honoring ``limit``."""
        ...
