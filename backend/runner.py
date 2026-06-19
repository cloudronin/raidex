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
from datetime import datetime, timezone
from pathlib import Path

import yaml

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

    cap = load_config()["budget"]["per_model_limit_usd"]
    if total > cap:
        raise RuntimeError(f"Estimated ${total:.2f} exceeds per_model cap ${cap}. Aborting.")

    results: dict[str, dict] = {}
    for b in benches:
        print(f"Running {b.__class__.__name__} against {model_id} ...")
        try:
            r = b.run(model_id, limit=_eff(b.id, limit))
        except Exception as e:  # record, don't sink the whole model
            print(f"  ! {b.id} failed: {e}")
            r = BenchmarkResult(benchmark_id=b.id, value=None, error=str(e))
        results[b.id] = {
            "value": r.value,
            "eval_source": r.eval_source,
            "eval_date": _utc_iso(),
            "raw": r.raw,
            "judge_model": r.judge_model,
            "error": r.error,
            "n_samples": r.n_samples,
        }

    composite = scoring.compute_composite(results)
    for bid, nv in composite.pop("normalized").items():
        results[bid]["normalized"] = nv

    output = {
        "config": {
            "model_id": model_id,
            "model_name": model_id.split("/")[-1],
            "developer": model_id.split("/")[0],
            "eval_date": _utc_iso(),
            "backend_version": BACKEND_VERSION,
        },
        "results": results,
        "composite": composite,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    filename = model_id.replace("/", "__") + ".json"
    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}  (RAI {composite['rai_score']}, "
          f"{composite['rai_coverage']}, {composite['badge_emoji']})")

    if not no_upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=out_path, path_in_repo=filename,
            repo_id=RESULTS_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
        )
        print(f"Uploaded to {RESULTS_REPO}/{filename}")


def poll_requests() -> None:
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    req_path = snapshot_download(repo_id=REQUESTS_REPO, repo_type="dataset")
    os.makedirs(OUT_DIR, exist_ok=True)
    for req_file in Path(req_path).glob("*.json"):
        try:
            req = json.loads(req_file.read_text())
        except Exception:
            continue
        if req.get("status") != "pending":
            continue
        print(f"Processing request: {req.get('model_id')}")
        try:
            run_eval(req["model_id"], req.get("tier", "A"))
            req["status"] = "completed"
            req["completed_at"] = _utc_iso()
        except Exception as e:
            req["status"] = "failed"
            req["error"] = str(e)
        tmp = Path(OUT_DIR) / req_file.name
        tmp.write_text(json.dumps(req, indent=2))
        api.upload_file(path_or_fileobj=str(tmp), path_in_repo=req_file.name,
                        repo_id=REQUESTS_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--poll", action="store_true")
    p.add_argument("--model", type=str, help="litellm model ID, e.g. anthropic/claude-opus-4-6")
    p.add_argument("--tier", type=str, default="A")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="sample the first N prompts (big benchmarks only)")
    p.add_argument("--no-upload", action="store_true", help="write result JSON locally, skip HF upload")
    args = p.parse_args()
    if args.poll:
        poll_requests()
    elif args.model:
        run_eval(args.model, args.tier, args.dry_run, args.limit, args.no_upload)
    else:
        p.error("provide --model or --poll")
