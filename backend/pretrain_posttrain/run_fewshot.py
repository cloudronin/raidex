"""Cleaner second pass: FEW-SHOT the generative-MCQ benchmarks (BBQ / WMDP / ETHICS).

Pass 1 (run_local.py, zero-shot) leaves a format-following confound: base models follow
the MCQ answer format worse than instruct, so part of the BBQ/ETHICS base→instruct delta
could be "instruct follows the format better" rather than a genuine fairness/ethics gain.

This re-runs ONLY the 3 generative-MCQ benchmarks with N-shot prompting (the canonical way
to evaluate base models) for BOTH base and instruct, so the format scaffolding is equal on
both sides. Reading:
  - If BBQ/ETHICS deltas SURVIVE few-shot (and WMDP stays ~0) → the gains are genuine
    alignment, not format-following. Hypothesis strengthened.
  - If they COLLAPSE toward 0 → pass 1's signal was mostly the base failing to follow format.

Judge benchmarks (StrongREJECT/XSTest/SimpleQA) are NOT re-run — few-shot is ill-defined for
refusal (you can't demonstrate complying with harmful requests), and those stay flagged
format_confounded from pass 1.

--num_fewshot is injected into lm_eval via a guarded subprocess.Popen patch (only rewrites
`lm_eval` commands, never the litellm proxy) — no edit to the pipeline, same philosophy as
run_local's runtime SAMPLE_EXEMPT override. Results -> <slug>__fs<N>.json.

Usage:
  python run_fewshot.py --run                 # run 5-shot on all 6 models, then compare
  python run_fewshot.py --run --num-fewshot 3
  python run_fewshot.py --compare             # just re-print the comparison from saved JSONs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scoring
from pairs import LOCAL_PAIRS, local_model_id
import run_local
import run_bases

OUT_DIR = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")
MCQ = ("bbq", "wmdp", "ethics")          # the generative-MCQ benchmarks (no judge)


# ---------------------------------------------------------------------------
# Inject --num_fewshot into lm_eval via a guarded Popen patch
# ---------------------------------------------------------------------------

def install_fewshot_patch(n: int) -> None:
    """Rewrite only `lm_eval` subprocess commands to add --num_fewshot n.

    _lmeval.run_lm_eval calls subprocess.Popen([...,'lm_eval',...]); _proxy starts the
    litellm proxy with subprocess.Popen([...,'litellm',...]). The guard touches only the
    former, so the proxy and any other subprocess (e.g. `ollama pull`) are untouched.
    """
    orig = subprocess.Popen

    def patched(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "lm_eval" and "--num_fewshot" not in cmd:
            cmd = list(cmd) + ["--num_fewshot", str(n)]
        return orig(cmd, *a, **k)

    subprocess.Popen = patched


# ---------------------------------------------------------------------------
# Run the 3 MCQ benchmarks for one model
# ---------------------------------------------------------------------------

def _fs_path(model_id: str, n: int) -> Path:
    return Path(OUT_DIR) / (model_id.replace("/", "__") + f"__fs{n}.json")


def run_mcq(model_id: str, limit: int, n: int) -> dict:
    from benchmarks.bbq import BBQ
    from benchmarks.wmdp import WMDP
    from benchmarks.ethics import ETHICS
    benches = {"bbq": BBQ(), "wmdp": WMDP(), "ethics": ETHICS()}
    out = {}
    for bid, b in benches.items():
        print(f"  [{n}-shot] {bid} @ limit {limit} ...", flush=True)
        try:
            r = b.run(model_id, limit=limit)
        except Exception as e:
            print(f"    ! {bid} failed: {str(e)[:160]}")
            out[bid] = {"value": None, "normalized": None, "error": str(e)[:200]}
            continue
        norm = (None if r.value is None
                else round(scoring.normalize(r.value, scoring.NORM[bid]), 4))
        out[bid] = {"value": r.value, "normalized": norm, "n_samples": r.n_samples}
        print(f"    {bid}={r.value}  normalized={norm}")
    return out


def run_all(num_fewshot: int, limit: int, judge: str) -> None:
    run_local.setup_env(judge)               # OPENAI_API_BASE→Ollama, judge (unused here), cap
    if not run_local.ollama_up():
        sys.exit(f"Ollama not reachable at {run_local.OLLAMA_HOST}.")
    install_fewshot_patch(num_fewshot)
    print(f"[fewshot] num_fewshot={num_fewshot}, limit={limit}, MCQ only ({', '.join(MCQ)})")

    for pair in LOCAL_PAIRS:
        for role, tag in [("base", pair["base_tag"]), ("instruct", pair["instruct_tag"])]:
            model_id = local_model_id(tag)
            # Resume safety (background runs get killed on machine sleep): skip models whose
            # few-shot JSON already exists, so a re-run finishes only what's missing.
            if _fs_path(model_id, num_fewshot).exists():
                print(f"\n==== {num_fewshot}-shot {pair['family']} / {role}: already done, skipping ====")
                continue
            print(f"\n==== {num_fewshot}-shot {pair['family']} / {role}: {model_id} ====", flush=True)
            if not run_local.ensure_pulled(tag) or not run_bases._check_base_available(model_id):
                continue
            res = run_mcq(model_id, limit, num_fewshot)
            doc = {"config": {"model_id": model_id, "num_fewshot": num_fewshot,
                              "limit": limit, "serving": "local-ollama"},
                   "results": res}
            _fs_path(model_id, num_fewshot).write_text(json.dumps(doc, indent=2))
    compare(num_fewshot)


# ---------------------------------------------------------------------------
# Compare few-shot deltas to pass-1 zero-shot deltas
# ---------------------------------------------------------------------------

def _zeroshot_norm(model_id: str, bid: str):
    """Pass-1 normalized value from the full run JSON."""
    p = Path(OUT_DIR) / (model_id.replace("/", "__") + ".json")
    if not p.exists():
        return None
    r = json.loads(p.read_text()).get("results", {}).get(bid, {})
    return None if r.get("error") else r.get("normalized")


def _fewshot_norm(model_id: str, bid: str, n: int):
    p = _fs_path(model_id, n)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("results", {}).get(bid, {}).get("normalized")


def compare(n: int) -> None:
    print(f"\n{'='*92}")
    print(f"FEW-SHOT vs ZERO-SHOT base→instruct deltas (normalized; {n}-shot vs pass-1 0-shot)")
    print(f"Does the post-training gain survive when base models get equal format scaffolding?")
    print('='*92)
    hdr = f"{'benchmark':<10} {'dimension':<16} {'type':<14} {'family':<14} {'0-shot Δ':>9} {'few-shot Δ':>11}  verdict"
    print(hdr); print('-'*len(hdr))
    summary = {bid: {"zs": [], "fs": []} for bid in MCQ}
    for bid in MCQ:
        dim = scoring.NORM[bid].dimension
        typ = "pretrain≈0" if bid == "wmdp" else "post-train↑"
        for pair in LOCAL_PAIRS:
            base, inst = local_model_id(pair["base_tag"]), local_model_id(pair["instruct_tag"])
            zb, zi = _zeroshot_norm(base, bid), _zeroshot_norm(inst, bid)
            fb, fi = _fewshot_norm(base, bid, n), _fewshot_norm(inst, bid, n)
            zd = (zi - zb) if (zb is not None and zi is not None) else None
            fd = (fi - fb) if (fb is not None and fi is not None) else None
            if zd is not None:
                summary[bid]["zs"].append(zd)
            if fd is not None:
                summary[bid]["fs"].append(fd)
            v = ""
            if zd is not None and fd is not None:
                if bid != "wmdp":
                    v = "survives" if fd > 0.03 else ("collapsed" if fd <= 0.03 else "")
                else:
                    v = "stays ~0" if abs(fd) < 0.10 else "moved"
            zs = f"{zd:+.3f}" if zd is not None else "—"
            fs = f"{fd:+.3f}" if fd is not None else "—"
            print(f"{bid:<10} {dim:<16} {typ:<14} {pair['family']:<14} {zs:>9} {fs:>11}  {v}")
    print('-'*len(hdr))
    for bid in MCQ:
        zs, fs = summary[bid]["zs"], summary[bid]["fs"]
        mzs = sum(zs)/len(zs) if zs else float("nan")
        mfs = sum(fs)/len(fs) if fs else float("nan")
        print(f"{bid:<10} mean across families:   0-shot {mzs:+.3f}   few-shot {mfs:+.3f}")
    print('='*92)
    print("Reading: post-train↑ benches (bbq, ethics) — if few-shot Δ stays clearly +, the gain")
    print("is genuine alignment, not format-following. wmdp (pretrain) should stay ~0 in both.")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run few-shot MCQ on all 6 models, then compare")
    mode.add_argument("--compare", action="store_true", help="re-print comparison from saved JSONs")
    p.add_argument("--num-fewshot", type=int, default=5)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--judge", default=run_local.DEFAULT_JUDGE)
    args = p.parse_args()
    if args.run:
        run_all(args.num_fewshot, args.limit, args.judge)
    else:
        compare(args.num_fewshot)


if __name__ == "__main__":
    main()
