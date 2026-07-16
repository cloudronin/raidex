"""raidex.core — the pure eval-and-score foundation.

No HF request-queue polling, no results-dataset upload, no dependency on Raidex servers:
just evaluate a model across the benchmarks, score, normalize, and return a structured
result. The backend service and the CLI are thin frontends over this package, which is why
a local ``raidex eval`` score is identical in scale to the published leaderboard.
"""
from .eval import (
    TIERS,
    SAMPLE_EXEMPT,
    JUDGE_BENCHMARKS,
    CORE_VERSION,
    resolve_benches,
    estimate_costs,
    evaluate,
    load_config,
)

__all__ = [
    "TIERS", "SAMPLE_EXEMPT", "JUDGE_BENCHMARKS", "CORE_VERSION",
    "resolve_benches", "estimate_costs", "evaluate", "load_config",
]
