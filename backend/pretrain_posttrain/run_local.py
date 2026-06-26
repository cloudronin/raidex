"""Run Test 1 locally: base + instruct served identically via Ollama.

The only viable base-model path (see pairs.py SERVING CONSTRAINT — every serverless
provider serves instruct-only). Both variants of each LOCAL_PAIR run through the
EXISTING Raidex Tier A pipeline pointed at a local Ollama server, with a fixed
off-family judge (SambaNova) held constant across base and instruct, so the
base→instruct delta isolates POST-TRAINING rather than serving/precision differences.

No edits to the existing pipeline — pure env wiring:
  OPENAI_API_BASE / OPENAI_API_KEY -> route litellm openai/<tag> (direct calls AND the
                                       lm-eval proxy subprocess, which inherits env) to Ollama
  RAIDEX_JUDGE_MODEL               -> a SambaNova instruct model (consistent judge)
  RAIDEX_PER_MODEL_USD             -> high cap (local target inference is free; only the
                                       judge has real, tiny cost)
  RAIDEX_CONCURRENCY               -> low (Ollama + SambaNova both prefer gentle load)

Prereqs: Ollama running with the pair's tags pulled; SAMBANOVA_API_KEY exported.

Usage:
  python run_local.py --smoke                       # fast plumbing check on the first pair
  python run_local.py --full                         # full Tier A, all pairs, base+instruct
  python run_local.py --full --family Qwen2.5-7B
  python run_local.py --smoke --judge sambanova/gemma-4-31B-it
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pairs import LOCAL_PAIRS, local_model_id, JUDGE_BENCHMARKS
# Reuse the serverless runner's helpers (dump / flag / canary / paths).
import run_bases

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_JUDGE = "sambanova/Meta-Llama-3.3-70B-Instruct"
OUT_DIR = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")


# ---------------------------------------------------------------------------
# Environment + Ollama plumbing
# ---------------------------------------------------------------------------

def setup_env(judge: str) -> None:
    """Point litellm's openai/ provider at Ollama and fix the judge. Verifies prereqs."""
    os.environ["OPENAI_API_BASE"] = OLLAMA_HOST + "/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"            # Ollama ignores the value but litellm wants one
    os.environ["RAIDEX_JUDGE_MODEL"] = judge
    os.environ.setdefault("RAIDEX_PER_MODEL_USD", "1000")   # local target is free; never trip the cap
    os.environ.setdefault("RAIDEX_CONCURRENCY", "4")         # Ollama parallel + SambaNova judge
    if judge.startswith("sambanova/") and not os.environ.get("SAMBANOVA_API_KEY"):
        sys.exit("SAMBANOVA_API_KEY not set — the judge needs it. "
                 "export SAMBANOVA_API_KEY=\"$(cat /tmp/SAMBANOVA_KEY.txt)\"")


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_tags() -> set[str]:
    try:
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=5) as r:
            data = json.load(r)
        return {m.get("name", "") for m in data.get("models", [])}
    except Exception:
        return set()


def ensure_pulled(tag: str) -> bool:
    """Pull an Ollama tag if not already present. Returns True if available."""
    have = ollama_tags()
    # Ollama reports names exactly as pulled (e.g. 'llama3.1:8b-text-q4_K_M').
    if tag in have:
        return True
    print(f"  pulling {tag} ...", flush=True)
    rc = subprocess.run(["ollama", "pull", tag]).returncode
    if rc != 0:
        print(f"  ! ollama pull {tag} failed (rc={rc}) — tag may not exist; check `ollama list`")
        return False
    return True


# ---------------------------------------------------------------------------
# Smoke: fast plumbing validation on one pair (no full Tier A)
# ---------------------------------------------------------------------------

def smoke(pair: dict, out_dir: str) -> bool:
    family = pair["family"]
    base_id = local_model_id(pair["base_tag"])
    inst_id = local_model_id(pair["instruct_tag"])
    print(f"\n==== SMOKE {family} ====")

    for tag in (pair["base_tag"], pair["instruct_tag"]):
        if not ensure_pulled(tag):
            return False

    # 1) Canary both variants resolve through Ollama.
    for role, mid in [("base", base_id), ("instruct", inst_id)]:
        if not run_bases._check_base_available(mid):
            return False
        print(f"  canary OK: {role} {mid}")

    # 2) Dump 5 raw BASE completions per judge benchmark (the format-confound eyeball).
    print(f"  dumping raw base completions → {out_dir}/base_sanity/ ...")
    run_bases._dump_judge_completions(base_id, out_dir)

    # 3) Validate the lm-eval path (proxy→Ollama→lm_eval) on BBQ @ limit 5, base.
    from benchmarks.bbq import BBQ
    print("  lm-eval plumbing: BBQ @ limit 5 (base) ...")
    rb = BBQ().run(base_id, limit=5)
    print(f"    BBQ base value={rb.value} n={rb.n_samples} err={rb.error}")
    if rb.error:
        print("  ! lm-eval path failed — see error above")
        return False

    # 4) Validate the judge path (target→SambaNova judge→score) on XSTest @ limit 6, base.
    from benchmarks.xstest import XSTest
    print("  judge plumbing: XSTest @ limit 6 (base) ...")
    rx = XSTest().run(base_id, limit=6)
    print(f"    XSTest base value={rx.value} n={rx.n_samples} judge={rx.judge_model} err={rx.error}")
    if rx.error:
        print("  ! judge path failed — see error above")
        return False

    print(f"\n  SMOKE OK for {family}. Eyeball {out_dir}/base_sanity/*.txt before the full run:")
    print("    does a low base judge-score mean 'no alignment' or 'could not parse the task'?")
    return True


# ---------------------------------------------------------------------------
# Full: complete Tier A for base + instruct, served identically
# ---------------------------------------------------------------------------

def run_full_pair(pair: dict, out_dir: str, limit: int | None) -> dict:
    from runner import run_eval
    family = pair["family"]
    status = {}
    for role, tag in [("base", pair["base_tag"]), ("instruct", pair["instruct_tag"])]:
        model_id = local_model_id(tag)
        print(f"\n==== FULL {family} / {role}: {model_id} (limit={limit}) ====", flush=True)
        if not ensure_pulled(tag):
            status[role] = "pull-failed"
            continue
        if not run_bases._check_base_available(model_id):
            status[role] = "unreachable"
            continue
        if role == "base":
            print("  dumping raw base completions for format-confound review ...")
            run_bases._dump_judge_completions(model_id, out_dir)
        try:
            # limit samples the big benchmarks; StrongREJECT/XSTest stay full (SAMPLE_EXEMPT).
            run_eval(model_id, tier="A", dry_run=False, limit=limit, no_upload=True)
        except Exception as e:
            print(f"  ! run_eval failed: {e}")
            status[role] = f"failed: {str(e)[:80]}"
            continue
        if role == "base":
            run_bases._flag_format_confounded(model_id, out_dir)
        else:
            # Mark instruct rows so analyze_correlation excludes these local runs from the board.
            _mark_local(model_id, out_dir)
        status[role] = "ok"
    return status


def _mark_local(model_id: str, out_dir: str) -> None:
    """Tag a local result JSON so it's distinguishable from the serverless board."""
    path = run_bases._result_path(model_id, out_dir)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data.setdefault("config", {})["serving"] = "local-ollama"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="fast plumbing check on the first pair")
    mode.add_argument("--full", action="store_true", help="full Tier A, all pairs, base+instruct")
    p.add_argument("--family", default=None, help="restrict to one family, e.g. 'Mistral-7B'")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help=f"judge model_id (default {DEFAULT_JUDGE})")
    p.add_argument("--limit", type=int, default=50,
                   help="per-benchmark sample size. Local Mac-GPU throughput is the bottleneck, "
                        "so ALL benchmarks are sampled (incl. StrongREJECT/XSTest) — see note below. "
                        "Default 50: base→instruct deltas are large, robust to n=50 noise.")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    setup_env(args.judge)

    # Local single-GPU inference is slow and base models ramble to max_tokens, so sampling
    # only the "big" benchmarks (leaving StrongREJECT 313 + XSTest 450 full) dominates wall
    # time. Sample EVERY benchmark at --limit instead. Runtime override of runner.SAMPLE_EXEMPT
    # (not an edit to runner.py). XSTest head is interleaved (25 safe / 25 unsafe at n=50) so
    # balanced accuracy stays well-defined; StrongREJECT is single-class (all forbidden prompts).
    import runner
    runner.SAMPLE_EXEMPT = set()
    print(f"[local] sampling ALL benchmarks at limit={args.limit} (SAMPLE_EXEMPT disabled for speed)")
    if not ollama_up():
        sys.exit(f"Ollama not reachable at {OLLAMA_HOST}. Start it: `ollama serve` or `brew services start ollama`.")
    print(f"Ollama up at {OLLAMA_HOST}; judge={args.judge}; concurrency={os.environ['RAIDEX_CONCURRENCY']}")

    pairs = LOCAL_PAIRS
    if args.family:
        pairs = [q for q in pairs if q["family"] == args.family]
        if not pairs:
            p.error(f"Unknown family {args.family!r}. Known: {[q['family'] for q in LOCAL_PAIRS]}")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.smoke:
        ok = smoke(pairs[0], args.out_dir)
        sys.exit(0 if ok else 1)

    results = {}
    for pair in pairs:
        results[pair["family"]] = run_full_pair(pair, args.out_dir, args.limit)
    print("\n==== run_local.py --full complete ====")
    for fam, st in results.items():
        print(f"  {fam}: base={st.get('base','?')}  instruct={st.get('instruct','?')}")
    print(f"\nNext: python pretrain_posttrain/analyze_delta.py --local --csv delta.csv")


if __name__ == "__main__":
    main()
