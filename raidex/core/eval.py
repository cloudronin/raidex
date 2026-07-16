"""raidex.core.eval — the pure eval-and-score core.

Evaluate a model across the Raidex benchmarks, score, normalize, and return a structured
result dict. No cost gate, no dry-run, no DLQ, no results-dataset upload, no request queue,
no dependency on Raidex servers. The backend service (`backend/runner.py`) and the CLI
(`raidex.cli`) are thin frontends that wrap those concerns around this core — so a local
`raidex eval` score is identical in scale to the published leaderboard (same core, same
benchmarks, same normalization).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import scoring
from .benchmarks.base import BenchmarkResult
from .benchmarks.bbq import BBQ
from .benchmarks.wmdp import WMDP
from .benchmarks.simpleqa import SimpleQA
from .benchmarks.strongreject import StrongREJECT
from .benchmarks.ethics import ETHICS
from .benchmarks.xstest import XSTest
from .benchmarks.advglue import AdvGLUE
from .benchmarks.confaide import ConfAIde
from .benchmarks.sycophancy import Sycophancy

# Version stamped into the result JSON's config block. Kept as `backend_version` in the
# schema (below) so existing board result JSONs stay byte-comparable.
CORE_VERSION = "0.1.0"

TIERS = {
    "A": [BBQ(), WMDP(), SimpleQA(), StrongREJECT(), ETHICS(), XSTest(), Sycophancy()],
    "B": [AdvGLUE(), ConfAIde()],
}
TIERS["A+B"] = TIERS["A"] + TIERS["B"]

# Small datasets — always run full, ignore --limit sampling.
SAMPLE_EXEMPT = {"strongreject", "xstest", "advglue", "confaide"}

# Benchmarks that require an LLM judge (used by the CLI's graceful-degradation path).
JUDGE_BENCHMARKS = {"simpleqa", "xstest", "strongreject"}

_CONFIG = None


def load_config(path: str | None = None) -> dict:
    global _CONFIG
    if _CONFIG is None:
        cfg = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
        _CONFIG = yaml.safe_load(cfg.read_text())
    return _CONFIG


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eff(bid: str, limit: int | None) -> int | None:
    """Effective limit for a benchmark: small datasets ignore sampling and run full."""
    return None if (limit and bid in SAMPLE_EXEMPT) else limit


def _result_block(r: BenchmarkResult) -> dict:
    """The per-benchmark dict stored in a result JSON's ``results`` map."""
    return {
        "value": r.value,
        "eval_source": r.eval_source,
        "eval_date": _utc_iso(),
        "raw": r.raw,
        "judge_model": r.judge_model,
        "error": r.error,
        "n_samples": r.n_samples,
        "n_failed": r.n_failed,
    }


def _finalize_composite(output: dict) -> dict:
    """Recompute the composite from ``output['results']`` and echo normalized scores back."""
    composite = scoring.compute_composite(output["results"])
    for bid, nv in composite.pop("normalized").items():
        output["results"][bid]["normalized"] = nv
    output["composite"] = composite
    return output


def _all_benches() -> list:
    return TIERS["A+B"]


def resolve_benches(tier: str | None = None, benchmark_ids=None) -> list:
    """A benchmark subset by explicit ids, or a whole tier. `benchmark_ids` wins if given."""
    if benchmark_ids:
        by_id = {b.id: b for b in _all_benches()}
        missing = [x for x in benchmark_ids if x not in by_id]
        if missing:
            raise ValueError(f"Unknown benchmark(s): {missing}; available: {sorted(by_id)}")
        return [by_id[x] for x in benchmark_ids]
    if tier is not None:
        if tier not in TIERS:
            raise ValueError(f"Unknown tier {tier!r}; supported: {list(TIERS)}")
        return TIERS[tier]
    raise ValueError("resolve_benches: provide `tier` or `benchmark_ids`")


def estimate_costs(model_id: str, benches: list, limit: int | None = None) -> dict:
    """Per-benchmark USD cost estimate (litellm pricing map + local fallback; no API calls)."""
    return {b.id: b.estimate_cost(model_id, _eff(b.id, limit)) for b in benches}


def evaluate(model_id: str, benches: list, *, limit: int | None = None,
             on_bench_result=None) -> dict:
    """Pure eval-and-score: run each benchmark, score, normalize, return the result dict.

    No cost gate, no dry-run print, no persistence, no upload, no DLQ. `on_bench_result(bid, r)`
    is an optional per-benchmark hook (the backend uses it to record DLQ entries); the CLI
    passes None. A benchmark that raises is captured as an errored `BenchmarkResult` and
    excluded from coverage — the model is never sunk by one failure.
    """
    results: dict[str, dict] = {}
    for b in benches:
        print(f"Running {b.__class__.__name__} against {model_id} ...", flush=True)
        try:
            r = b.run(model_id, limit=_eff(b.id, limit))
        except Exception as e:  # record, don't sink the whole model
            print(f"  ! {b.id} failed: {e}", flush=True)
            r = BenchmarkResult(benchmark_id=b.id, value=None, error=str(e))
        results[b.id] = _result_block(r)
        if on_bench_result is not None:
            on_bench_result(b.id, r)
    return _finalize_composite({
        "config": {
            "model_id": model_id,
            "model_name": model_id.split("/")[-1],
            "developer": model_id.split("/")[0],
            "eval_date": _utc_iso(),
            "backend_version": CORE_VERSION,
        },
        "results": results,
        "composite": None,
    })
