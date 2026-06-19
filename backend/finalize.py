"""finalize.py — produce the final Raidex board (one consistent pass over all models).

Per model:
  1. Keep good BBQ/WMDP/ETHICS (lm-eval) results; re-run only if missing or errored
     (this repairs the rate-limited Mistral/DeepSeek/gpt-oss rows).
  2. Re-run the judge benchmarks (SimpleQA/XSTest/StrongREJECT) with the NEUTRAL judge
     (set via RAIDEX_JUDGE_MODEL) so the whole board is judged by one off-roster instrument.
  3. Run Tier B (AdvGLUE robustness + ConfAIde privacy) — adds the 2 missing dimensions.
  4. Recompute the composite over 8 and upload.

Throttle-prone providers (SambaNova, Mistral) run at low concurrency. Loads each
model's prior result from the HF dataset snapshot. Run from the repo root:
  RAIDEX_JUDGE_MODEL=gemini/gemini-2.5-flash <keys+HF_TOKEN> python -u finalize.py
"""
import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from benchmarks.base import BenchmarkResult
from benchmarks.bbq import BBQ
from benchmarks.wmdp import WMDP
from benchmarks.ethics import ETHICS
from benchmarks.simpleqa import SimpleQA
from benchmarks.strongreject import StrongREJECT
from benchmarks.xstest import XSTest
from benchmarks.advglue import AdvGLUE
from benchmarks.confaide import ConfAIde
from huggingface_hub import HfApi, snapshot_download

RESULTS_REPO = "cloudronin/raidex-results"
LIMIT = int(os.environ.get("RAIDEX_LIMIT", "300"))
LMEVAL = [BBQ(), WMDP(), ETHICS()]                  # keep if good, re-run if errored/missing
JUDGEB = [SimpleQA(), StrongREJECT(), XSTest()]     # always re-run with the neutral judge
TIERB = [AdvGLUE(), ConfAIde()]                     # new Tier B
SMALL = {"strongreject", "xstest", "advglue", "confaide"}   # run full (ignore LIMIT)
LOWCONC = ("sambanova/", "mistral/", "huggingface/")   # providers that throttle under load

MODELS = [
    "openai/gpt-5.2", "openai/gpt-4o", "openai/gpt-4o-mini",
    "anthropic/claude-opus-4-8", "anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5-20251001",
    "gemini/gemini-2.5-flash", "mistral/mistral-large-latest",
    "sambanova/DeepSeek-V3.2", "sambanova/Meta-Llama-3.3-70B-Instruct", "sambanova/gpt-oss-120b",
    "xai/grok-4.3",
    "huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507", "huggingface/google/gemma-3-27b-it",
]
MODELS = os.environ.get("RAIDEX_MODELS", "").split() or MODELS

# Write token for dataset upload/snapshot — kept separate from HF_TOKEN, which may hold
# the *inference* token for litellm's huggingface/ models.
TOK = os.environ.get("RAIDEX_HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
api = HfApi()


def _now():
    return datetime.now(timezone.utc)


def _date():
    return _now().isoformat()


try:
    SNAP = snapshot_download(RESULTS_REPO, repo_type="dataset", token=TOK)
except Exception as e:
    print("snapshot failed:", e, flush=True)
    SNAP = None


def _load_existing(model_id):
    if not SNAP:
        return None
    p = os.path.join(SNAP, model_id.replace("/", "__") + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def _record(results, b, model_id, limit):
    try:
        r = b.run(model_id, limit=limit)
    except Exception as e:
        r = BenchmarkResult(b.id, value=None, error=str(e))
    results[b.id] = {"value": r.value, "eval_source": "automated", "eval_date": _date(),
                     "raw": r.raw, "judge_model": r.judge_model, "error": r.error, "n_samples": r.n_samples}
    return results[b.id]


os.makedirs("/tmp/raidex_final", exist_ok=True)
for model_id in MODELS:
    print(f"\n#### FINALIZE {model_id} :: {_now():%H:%M:%S}", flush=True)
    os.environ["RAIDEX_CONCURRENCY"] = "2" if model_id.startswith(LOWCONC) else "8"
    existing = _load_existing(model_id)
    results = dict(existing["results"]) if existing else {}

    for b in LMEVAL:
        cur = results.get(b.id)
        if cur and cur.get("value") is not None and not cur.get("error"):
            print(f"  keep   {b.id}={cur['value']}", flush=True)
        else:
            print(f"  rerun  {b.id} (missing/errored)", flush=True)
            r = _record(results, b, model_id, LIMIT)
            print(f"    -> {b.id}={r['value']} err={r['error']}", flush=True)
    for b in JUDGEB:
        r = _record(results, b, model_id, None if b.id in SMALL else LIMIT)
        print(f"  judge  {b.id}={r['value']} (judge={r['judge_model']})", flush=True)
    for b in TIERB:
        r = _record(results, b, model_id, None)
        print(f"  tierB  {b.id}={r['value']}", flush=True)

    composite = scoring.compute_composite(results)
    for bid, nv in composite.pop("normalized").items():
        results[bid]["normalized"] = nv
    dev = model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id.split("/")[0]
    cfg = (existing or {}).get("config") or {"model_id": model_id, "model_name": model_id.split("/")[-1],
                                             "developer": dev}
    cfg.update({"eval_date": _date(), "backend_version": "0.2.0",
                "judge": os.environ.get("RAIDEX_JUDGE_MODEL", "")})
    out = {"config": cfg, "results": results, "composite": composite}
    fn = model_id.replace("/", "__") + ".json"
    fp = f"/tmp/raidex_final/{fn}"
    json.dump(out, open(fp, "w"), indent=2, ensure_ascii=False)
    print(f"  => RAI {composite['rai_score']} {composite['rai_coverage']} {composite['badge_emoji']}", flush=True)
    if TOK:
        api.upload_file(path_or_fileobj=fp, path_in_repo=fn, repo_id=RESULTS_REPO,
                        repo_type="dataset", token=TOK)
        print("  uploaded", flush=True)

print("\n######## FINALIZE COMPLETE ########", flush=True)
