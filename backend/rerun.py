"""Re-run the models affected by the rate-limit-swallow and thinking-truncation bugs,
using the fixed pipeline (retry/backoff + non-swallowing guard + DLQ + max_tokens 512).

Keeps the 5 verified-clean models untouched (gpt-5.2, gpt-4o, gpt-4o-mini, sonnet,
haiku). Per-provider concurrency throttling; any residual failures land in the DLQ
and are mopped up with a final --replay-dlq pass.

Env: provider keys + RAIDEX_JUDGE_MODEL (neutral judge), HF_TOKEN = WRITE token (for
uploads), HUGGINGFACE_API_KEY = inference token (for litellm huggingface/).
"""
import json
import os
import traceback

import dlq
import runner

# Full Tier A+B re-runs (artifacts, lm-eval errors, missing Tier B, never-ran).
FULL = [
    "anthropic/claude-opus-4-8",                       # headline — run first
    "openai/gpt-5.5",                                  # current OpenAI flagship (we also score gpt-5.2)
    "xai/grok-4.3",                                    # fast (conc 8), reliable
    "sambanova/DeepSeek-V3.2",                         # throttled tail (conc 1) below
    "sambanova/Meta-Llama-3.3-70B-Instruct",
    "sambanova/gpt-oss-120b",
    "mistral/mistral-large-latest",
    "huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507",
    "huggingface/google/gemma-3-27b-it",
]
# Gemini only needs the two truncated Tier B benchmarks re-run + merged.
GEMINI = "gemini/gemini-2.5-flash"
LIMIT = int(os.environ.get("RAIDEX_LIMIT", "300"))
THROTTLED = ("sambanova/", "mistral/", "huggingface/")


def conc_for(model_id: str) -> str:
    if model_id.startswith(THROTTLED):
        return "1"          # throttling-prone providers (conc 2 still hit Mistral 429s)
    if "opus" in model_id:
        return "4"          # flagship; temperature handled by drop_params, rate fine here
    return "8"


def limit_for(model_id: str) -> int:
    # Smaller sample for rate-limited providers so a conc-1 run is tractable.
    return 150 if model_id.startswith(THROTTLED) else LIMIT


def already_clean(model_id: str) -> bool:
    """Resume support: skip a model that already has a complete (8/8, no errored
    benchmark) result on disk, so a restart after a kill continues instead of redoing."""
    p = os.path.join(runner.OUT_DIR, model_id.replace("/", "__") + ".json")
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    comp = d.get("composite", {})
    errored = any(r.get("error") for r in d.get("results", {}).values())
    return comp.get("rai_coverage_pct") == 100 and not errored


def main():
    for m in FULL:
        if already_clean(m):
            print(f"\n#### SKIP {m} (already 8/8 clean on disk)", flush=True)
            continue
        os.environ["RAIDEX_CONCURRENCY"] = conc_for(m)
        lim = limit_for(m)
        print(f"\n#### RERUN {m}  (conc={os.environ['RAIDEX_CONCURRENCY']}, limit={lim})", flush=True)
        try:
            runner.run_eval(m, tier="A+B", limit=lim)
        except Exception as e:
            print(f"!! {m} run_eval crashed: {e}", flush=True)
            traceback.print_exc()

    # Gemini: seed its 2 truncated benchmarks into the DLQ; the replay below re-runs
    # them (now at max_tokens=512) and merges into gemini's existing result JSON.
    for bid in ("advglue", "confaide"):
        dlq.record(GEMINI, bid, tier="B", limit=None, guard_tripped=True,
                   error="max_tokens=16 thinking-truncation (pre-fix); re-run at 512")

    # One replay pass: gemini's 2 benchmarks + any (model, benchmark) that failed above.
    os.environ["RAIDEX_CONCURRENCY"] = "4"
    print("\n#### REPLAY-DLQ (gemini Tier B + any residual failures)", flush=True)
    try:
        runner.replay_dlq()
    except Exception as e:
        print(f"!! replay_dlq crashed: {e}", flush=True)
        traceback.print_exc()

    pending = dlq.read(pending_only=True)
    print(f"\n#### RERUN COMPLETE — {len(pending)} DLQ entr(ies) still pending", flush=True)
    for e in pending:
        print(f"   STILL-FAILING {e['model_id']} :: {e['benchmark_id']} — {str(e.get('error'))[:80]}", flush=True)


if __name__ == "__main__":
    main()
