"""Dead-letter queue for failed benchmark runs.

When a benchmark can't complete cleanly — calls failing after retries (direct path)
or ``lm_eval`` exiting non-zero — the runner appends a record here, keyed by
(model_id, benchmark_id). ``runner.py --replay-dlq`` then re-runs exactly those
(model, benchmark) pairs (the retry/backoff now absorbs the transient errors),
merges each fresh result into the model's existing result JSON, recomputes the
composite, re-uploads, and marks the entry resolved. Recovery is targeted: rerun the
failures, not the whole board.

The (model, benchmark) pair is the replay grain because a benchmark score (F1,
balanced accuracy, Pearson, ASR) is computed over its whole sample — you can't merge
one re-issued call back into a metric, but you can cheaply recompute one benchmark.
Per-call error strings are kept on each record for diagnosis.

Format: JSON Lines, append-only. ``read`` de-dupes by (model, benchmark) keeping the
latest line, so a later ``resolve`` (or re-failure) supersedes earlier records. Path
defaults to ``$RAIDEX_OUT_ROOT/dlq.jsonl``; override with ``$RAIDEX_DLQ_PATH``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def dlq_path() -> str:
    return os.environ.get(
        "RAIDEX_DLQ_PATH",
        os.path.join(os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex"), "dlq.jsonl"),
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(rec: dict, path: str | None) -> None:
    p = path or dlq_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record(model_id: str, benchmark_id: str, *, tier: str | None = None,
           limit: int | None = None, n_failed: int | None = None,
           n_total: int | None = None, error: str = "", guard_tripped: bool = False,
           sample_errors: list | None = None, path: str | None = None) -> None:
    """Append a failure record for a (model, benchmark) pair."""
    _append({
        "ts": _utc(),
        "model_id": model_id,
        "benchmark_id": benchmark_id,
        "tier": tier,
        "limit": limit,
        "n_failed": n_failed,
        "n_total": n_total,
        "guard_tripped": guard_tripped,   # True = benchmark unreliable; False = passed but had failures
        "error": error,
        "sample_errors": (sample_errors or [])[:3],
        "status": "pending",
    }, path)


def resolve(model_id: str, benchmark_id: str, path: str | None = None) -> None:
    """Mark a (model, benchmark) entry resolved (a later line supersedes pending)."""
    _append({
        "ts": _utc(),
        "model_id": model_id,
        "benchmark_id": benchmark_id,
        "status": "resolved",
    }, path)


def read(path: str | None = None, pending_only: bool = True) -> list[dict]:
    """Read DLQ records, de-duplicated by (model_id, benchmark_id) keeping the latest
    line. Returns pending entries by default."""
    p = path or dlq_path()
    if not os.path.exists(p):
        return []
    latest: dict[tuple, dict] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            latest[(rec.get("model_id"), rec.get("benchmark_id"))] = rec
    recs = list(latest.values())
    if pending_only:
        recs = [r for r in recs if r.get("status", "pending") == "pending"]
    return recs
