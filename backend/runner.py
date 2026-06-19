"""Raidex eval runner.

Usage:
  python runner.py --dry-run --model openai/gpt-5.2 --tier A          # cost estimate only
  python runner.py --model openai/qwen3.7-max --tier A --limit 50 --no-upload   # smoke test
  python runner.py --model anthropic/claude-opus-4-6 --tier A         # full run + upload
  python runner.py --poll                                             # drain the request queue

--limit samples the big benchmarks; small datasets (StrongREJECT, XSTest) always
run full. Model IDs use litellm format provider/model_name; litellm reads API keys
from env vars. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

import dlq
import scoring
from benchmarks.base import BenchmarkResult
from benchmarks.bbq import BBQ
from benchmarks.wmdp import WMDP
from benchmarks.simpleqa import SimpleQA
from benchmarks.strongreject import StrongREJECT
from benchmarks.ethics import ETHICS
from benchmarks.xstest import XSTest
from benchmarks.advglue import AdvGLUE
from benchmarks.confaide import ConfAIde

BACKEND_VERSION = "0.1.0"
RESULTS_REPO = "cloudronin/raidex-results"
REQUESTS_REPO = "cloudronin/raidex-requests"
OUT_DIR = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")

TIERS = {
    "A": [BBQ(), WMDP(), SimpleQA(), StrongREJECT(), ETHICS(), XSTest()],
    "B": [AdvGLUE(), ConfAIde()],
}
TIERS["A+B"] = TIERS["A"] + TIERS["B"]

# Small datasets — always run full, ignore --limit sampling.
SAMPLE_EXEMPT = {"strongreject", "xstest", "advglue", "confaide"}

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


def _dlq_if_failed(model_id: str, bid: str, r: BenchmarkResult, *,
                   tier: str | None, limit: int | None) -> None:
    """Record a DLQ entry when a benchmark errored (guard tripped) or had any calls
    fail after retries — so the failure can be replayed later, not silently kept."""
    nf = r.n_failed or 0
    if not r.error and nf == 0:
        return
    dlq.record(
        model_id, bid, tier=tier, limit=limit,
        n_failed=r.n_failed, n_total=r.n_samples,
        guard_tripped=bool(r.error),
        error=r.error or f"{nf} call(s) failed after retries (benchmark still within guard)",
        sample_errors=r.sample_errors,
    )
    print(f"  ⚠ DLQ[{'guard-tripped' if r.error else 'partial'}] {bid}: "
          f"{r.error or str(nf) + ' calls failed'}")


def _finalize_composite(output: dict) -> dict:
    """Recompute the composite from ``output['results']`` and echo normalized scores back."""
    composite = scoring.compute_composite(output["results"])
    for bid, nv in composite.pop("normalized").items():
        output["results"][bid]["normalized"] = nv
    output["composite"] = composite
    return output


def _persist(output: dict, model_id: str, no_upload: bool) -> None:
    """Write the result JSON locally and (unless no_upload) upload to the results dataset."""
    os.makedirs(OUT_DIR, exist_ok=True)
    filename = model_id.replace("/", "__") + ".json"
    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    c = output["composite"]
    print(f"Wrote {out_path}  (RAI {c['rai_score']}, {c['rai_coverage']}, {c['badge_emoji']})")
    if not no_upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=out_path, path_in_repo=filename,
            repo_id=RESULTS_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
        )
        print(f"Uploaded to {RESULTS_REPO}/{filename}")


def _load_existing_result(model_id: str) -> dict | None:
    """Load a model's current result JSON — local OUT_DIR first, else download from the
    results dataset. Returns None if neither exists (caller should run a full eval)."""
    filename = model_id.replace("/", "__") + ".json"
    local = os.path.join(OUT_DIR, filename)
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(RESULTS_REPO, filename, repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"), force_download=True)
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def run_eval(model_id: str, tier: str = "A", dry_run: bool = False,
             limit: int | None = None, no_upload: bool = False) -> None:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; supported: {list(TIERS)}")
    benches = TIERS[tier]

    estimates = {b.id: b.estimate_cost(model_id, _eff(b.id, limit)) for b in benches}
    total = round(sum(estimates.values()), 2)
    if dry_run:
        suffix = f", limit {limit}" if limit else ""
        print(f"Cost estimate — {model_id} (Tier {tier}{suffix}):")
        for bid, c in estimates.items():
            print(f"  {bid:14s} ${c:8.2f}")
        print(f"  {'TOTAL':14s} ${total:8.2f}")
        return

    cap = float(os.environ.get("RAIDEX_PER_MODEL_USD") or load_config()["budget"]["per_model_limit_usd"])
    if total > cap:
        raise RuntimeError(f"Estimated ${total:.2f} exceeds per-model cap ${cap}. Aborting.")

    results: dict[str, dict] = {}
    for b in benches:
        print(f"Running {b.__class__.__name__} against {model_id} ...")
        try:
            r = b.run(model_id, limit=_eff(b.id, limit))
        except Exception as e:  # record, don't sink the whole model
            print(f"  ! {b.id} failed: {e}")
            r = BenchmarkResult(benchmark_id=b.id, value=None, error=str(e))
        results[b.id] = _result_block(r)
        _dlq_if_failed(model_id, b.id, r, tier=tier, limit=_eff(b.id, limit))

    output = _finalize_composite({
        "config": {
            "model_id": model_id,
            "model_name": model_id.split("/")[-1],
            "developer": model_id.split("/")[0],
            "eval_date": _utc_iso(),
            "backend_version": BACKEND_VERSION,
        },
        "results": results,
        "composite": None,
    })
    _persist(output, model_id, no_upload)


_MODEL_ID_RE = re.compile(r"^[a-z0-9_.\-]+/[A-Za-z0-9_.\-:/]+$")


def _valid_model_id(model_id: str) -> bool:
    """Cheap sanity check on a queued model_id — litellm provider/name, bounded length."""
    return bool(model_id and len(model_id) <= 120 and _MODEL_ID_RE.match(model_id.strip()))


def _upload_request(api, name: str, req: dict) -> None:
    tmp = Path(OUT_DIR) / name
    tmp.write_text(json.dumps(req, indent=2))
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo=name,
                    repo_id=REQUESTS_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


def poll_requests(max_models: int | None = None, limit: int | None = None) -> None:
    """Drain pending requests: evaluate, write results, mark each completed/failed.

    Bounded for unattended/public runs (an open submit queue can otherwise spend an
    unbounded amount):
      RAIDEX_POLL_MAX      — max models to evaluate per invocation (default 1)
      RAIDEX_POLL_LIMIT    — per-benchmark sample size (default 300; blank/0 = full)
      RAIDEX_PER_MODEL_USD — per-model cost ceiling, enforced in run_eval
    Each request is *claimed* (status 'running', uploaded) before evaluation, so a
    crash or an overlapping run can't re-evaluate (re-bill) the same model.
    """
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if max_models is None:
        max_models = int(os.environ.get("RAIDEX_POLL_MAX", "1"))
    if limit is None:
        lim = os.environ.get("RAIDEX_POLL_LIMIT", "300")
        limit = int(lim) if lim and lim != "0" else None
    req_path = snapshot_download(repo_id=REQUESTS_REPO, repo_type="dataset")
    os.makedirs(OUT_DIR, exist_ok=True)
    done = 0
    for req_file in sorted(Path(req_path).glob("*.json")):
        if done >= max_models:
            print(f"Reached RAIDEX_POLL_MAX={max_models}; stopping this run.")
            break
        try:
            req = json.loads(req_file.read_text())
        except Exception:
            continue
        if req.get("status") != "pending":
            continue
        model_id = (req.get("model_id") or "").strip()
        if not _valid_model_id(model_id):
            print(f"Skipping invalid model_id: {model_id!r}")
            continue
        tier = req.get("tier", "A")
        print(f"Processing request: {model_id} (tier {tier}, limit {limit})")
        # Claim it first so a crash / overlapping run can't re-evaluate (re-bill) it.
        req["status"] = "running"
        req["started_at"] = _utc_iso()
        _upload_request(api, req_file.name, req)
        try:
            run_eval(model_id, tier, limit=limit)
            req["status"] = "completed"
            req["completed_at"] = _utc_iso()
        except Exception as e:
            req["status"] = "failed"
            req["error"] = str(e)[:500]
            print(f"  ! {model_id} failed: {str(e)[:200]}")
        _upload_request(api, req_file.name, req)
        done += 1
    print(f"Poll complete: {done} model(s) evaluated this run.")


def replay_dlq(no_upload: bool = False, limit_override: int | None = None) -> None:
    """Re-run the failed (model, benchmark) pairs recorded in the DLQ and merge each
    fresh result into that model's existing result JSON (recomputing the composite).
    Entries that succeed are marked resolved; ones that still fail stay queued."""
    entries = dlq.read(pending_only=True)
    if not entries:
        print("DLQ empty — nothing to replay.")
        return
    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for e in entries:
        by_model[e["model_id"]].append(e)
    print(f"Replaying {len(entries)} DLQ entr(ies) across {len(by_model)} model(s)...")
    bench_by_id = {b.id: b for b in TIERS["A+B"]}
    for model_id, evs in by_model.items():
        existing = _load_existing_result(model_id)
        if existing is None:
            print(f"  ! {model_id}: no existing result JSON — run a full eval instead. Skipping.")
            continue
        changed = False
        for e in evs:
            bid = e["benchmark_id"]
            b = bench_by_id.get(bid)
            if b is None:
                print(f"  ! {model_id}/{bid}: unknown benchmark; skip")
                continue
            lim = limit_override if limit_override is not None else e.get("limit")
            print(f"  replay {model_id} :: {bid} (limit={lim})")
            try:
                r = b.run(model_id, limit=lim)
            except Exception as ex:
                print(f"    still failing: {ex}")
                dlq.record(model_id, bid, tier=e.get("tier"), limit=lim,
                           error=str(ex), guard_tripped=True)
                continue
            if r.error or (r.n_failed or 0) > 0:
                print(f"    still degraded ({r.error or str(r.n_failed) + ' failed'}); leaving queued")
                _dlq_if_failed(model_id, bid, r, tier=e.get("tier"), limit=lim)
                continue
            existing["results"][bid] = _result_block(r)
            dlq.resolve(model_id, bid)
            changed = True
            print(f"    ✓ {bid} = {r.value}  (n={r.n_samples})")
        if changed:
            _finalize_composite(existing)
            _persist(existing, model_id, no_upload)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--poll", action="store_true")
    p.add_argument("--replay-dlq", action="store_true",
                   help="re-run failed (model,benchmark) pairs from the DLQ and merge them back")
    p.add_argument("--model", type=str, help="litellm model ID, e.g. anthropic/claude-opus-4-8")
    p.add_argument("--tier", type=str, default="A")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="sample the first N prompts (big benchmarks only)")
    p.add_argument("--no-upload", action="store_true", help="write result JSON locally, skip HF upload")
    args = p.parse_args()
    if args.poll:
        poll_requests()
    elif args.replay_dlq:
        replay_dlq(no_upload=args.no_upload, limit_override=args.limit)
    elif args.model:
        run_eval(args.model, args.tier, args.dry_run, args.limit, args.no_upload)
    else:
        p.error("provide --model, --poll, or --replay-dlq")
