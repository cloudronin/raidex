"""Test 1: Base→instruct delta per benchmark, per model family.

Reads base and instruct result JSONs and computes per-benchmark delta
(instruct_normalized − base_normalized). Format-confounded benchmarks (judge
benchmarks where base model format may confound the score) are marked with *
in the output. The kill criterion exits 1 if fewer than 3 families with complete
data show a positive mean delta on predicted post-training-bound benchmarks.

Usage:
  python analyze_delta.py [--out-dir /tmp/raidex] [--csv delta.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scoring
from pairs import (MODEL_PAIRS, LOCAL_PAIRS, local_model_id,
                   JUDGE_BENCHMARKS, PRETRAINING_BOUND, POST_TRAINING_BOUND)

OUT_DIR = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")


def _normalized_pairs(local: bool) -> list[dict]:
    """Return pairs as {family, base, instruct} model-id dicts for either mode."""
    if local:
        return [{"family": q["family"],
                 "base": local_model_id(q["base_tag"]),
                 "instruct": local_model_id(q["instruct_tag"])}
                for q in LOCAL_PAIRS]
    return [{"family": q["family"], "base": q["base"], "instruct": q["instruct"]}
            for q in MODEL_PAIRS]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(model_id: str, out_dir: str) -> Optional[dict]:
    fn = model_id.replace("/", "__") + ".json"
    local = Path(out_dir) / fn
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            return None
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("cloudronin/raidex-results", fn, repo_type="dataset",
                            force_download=True, token=os.environ.get("HF_TOKEN"))
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _normalized(result_doc: dict) -> dict[str, float]:
    out = {}
    for bid, r in result_doc.get("results", {}).items():
        if r.get("error") or r.get("value") is None:
            continue
        nv = r.get("normalized")
        if nv is not None:
            out[bid] = nv
    return out


def _is_format_confounded(result_doc: dict, bid: str) -> bool:
    return bool(result_doc.get("results", {}).get(bid, {}).get("format_confounded"))


def _verify_instruct(model_id: str, result: dict) -> list[str]:
    """Recompute normalized values and return mismatch descriptions (empty = clean)."""
    mismatches = []
    for bid, r in result.get("results", {}).items():
        if bid not in scoring.NORM:
            continue
        if r.get("value") is None or r.get("error"):
            continue
        stored = r.get("normalized")
        if stored is None:
            continue
        recomputed = round(scoring.normalize(r["value"], scoring.NORM[bid]), 4)
        if abs(stored - recomputed) > 0.0015:
            mismatches.append(f"{bid}: stored={stored} recomputed={recomputed}")
    return mismatches


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def collect(out_dir: str, pairs: list[dict]) -> dict[str, dict]:
    """For each {family, base, instruct} pair, load base + instruct JSONs and verify.

    Returns a dict keyed by family name:
      {"base_norm": {bid: float}, "inst_norm": {bid: float}, "confounded": {bid: bool}}
    Families with missing or bad data are omitted with a clear log message.
    """
    collected = {}
    for pair in pairs:
        family = pair["family"]
        inst_doc = _load_json(pair["instruct"], out_dir)
        base_doc = _load_json(pair["base"], out_dir)

        if inst_doc is None:
            print(f"  ! {family}: instruct result not found — skipping")
            continue

        mismatches = _verify_instruct(pair["instruct"], inst_doc)
        if mismatches:
            print(f"  ERROR {family}: instruct scoring mismatch — delta baseline untrustworthy")
            for m in mismatches:
                print(f"    {m}")
            print(f"  Skipping {family}.")
            continue

        if base_doc is None:
            print(f"  ! {family}: base result not found — run run_bases.py first")
            continue

        inst_norm = _normalized(inst_doc)
        base_norm = _normalized(base_doc)
        confounded = {bid: _is_format_confounded(base_doc, bid)
                      for bid in scoring.NORM}

        collected[family] = {
            "base_norm":  base_norm,
            "inst_norm":  inst_norm,
            "confounded": confounded,
        }
        print(f"  Loaded {family}: {len(base_norm)} base benchmarks, {len(inst_norm)} instruct benchmarks")

    return collected


def compute_rows(collected: dict[str, dict]) -> list[dict]:
    """Build the per-benchmark delta table."""
    families = list(collected)
    benchmarks = sorted(scoring.NORM.keys())
    rows = []
    for bid in benchmarks:
        spec = scoring.NORM[bid]
        if bid in PRETRAINING_BOUND:
            expected_type = "pretraining-bound"
        elif bid in POST_TRAINING_BOUND:
            expected_type = "post-training-bound"
        else:
            expected_type = "ambiguous"

        row: dict = {
            "benchmark":     bid,
            "dimension":     spec.dimension,
            "expected_type": expected_type,
        }
        clean_vals = []
        for fam in families:
            bv = collected[fam]["base_norm"].get(bid)
            iv = collected[fam]["inst_norm"].get(bid)
            cf = collected[fam]["confounded"].get(bid, False)
            if bv is not None and iv is not None:
                delta = round(iv - bv, 4)
                row[fam] = f"{delta:+.3f}" + ("*" if cf else "")
                if not cf:
                    clean_vals.append(delta)
            else:
                row[fam] = "—"

        row["mean_delta_clean"] = round(sum(clean_vals) / len(clean_vals), 3) if clean_vals else None
        row["n_clean_families"] = len(clean_vals)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Kill criterion
# ---------------------------------------------------------------------------

def check_kill_criterion(collected: dict[str, dict]) -> bool:
    """Returns True if the study should continue, False if kill criterion fires.

    Evaluated on the CLEAN (non-format-confounded) post-training-bound benchmarks only.
    The format_confounded judge benchmarks (StrongREJECT, XSTest) are excluded because a
    base model's deltas there are untrustworthy: base rambling can deflate OR inflate the
    score in either direction (e.g. Mistral base StrongREJECT ASR is artificially low
    because compliant output derails into hallucinated multi-turn), so they cannot serve
    as the clean signal the criterion tests. A family 'passes' if its mean delta on the
    clean post-training-bound benchmarks is > 0. Fewer than 3 passing → stop (spec).
    """
    families = list(collected)
    if len(families) < 3:
        print(f"\nKILL CRITERION: only {len(families)} families have complete data (need ≥3 to evaluate).")
        return False

    clean_pt = sorted(b for b in POST_TRAINING_BOUND
                      if not any(collected[f]["confounded"].get(b) for f in families))
    passing = []
    for fam in families:
        bns, ins, cf = collected[fam]["base_norm"], collected[fam]["inst_norm"], collected[fam]["confounded"]
        deltas = []
        for bid in POST_TRAINING_BOUND:
            if cf.get(bid):                      # skip format-confounded — not clean signal
                continue
            bv, iv = bns.get(bid), ins.get(bid)
            if bv is not None and iv is not None:
                deltas.append(iv - bv)
        if deltas and (sum(deltas) / len(deltas)) > 0:
            passing.append(fam)

    print(f"\nClean post-training-bound benchmarks used for kill criterion: {clean_pt or '(none)'}")
    print(f"(format-confounded judge benchmarks excluded — labeled but not trusted as clean signal)")
    if len(passing) >= 3:
        print(f"Kill criterion: {len(passing)}/{len(families)} families show positive mean delta "
              f"on clean post-training-bound benchmarks. Pattern holds — continue.")
        return True
    else:
        print(f"KILL CRITERION TRIGGERED: only {len(passing)}/{len(families)} families show positive "
              f"mean delta on clean post-training-bound benchmarks (need ≥3).")
        print("The pretraining/post-training split is not cleanly readable from these benchmarks.")
        print("Per the spec: stop here.")
        return False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], families: list[str]) -> None:
    if not rows:
        return
    fam_width = 13
    header = (f"{'benchmark':<14} {'dimension':<16} {'expected_type':<20} "
              + " ".join(f"{f[:fam_width]:<{fam_width}}" for f in families)
              + f"  {'mean(clean)':>12}  n")
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for row in rows:
        fam_cols = " ".join(f"{str(row.get(f, '—')):<{fam_width}}" for f in families)
        md = row["mean_delta_clean"]
        md_str = f"{md:+.3f}" if md is not None else "      —"
        print(f"{row['benchmark']:<14} {row['dimension']:<16} {row['expected_type']:<20} "
              f"{fam_cols}  {md_str:>12}  {row['n_clean_families']}")
    print(sep)
    print("* = format_confounded — base model may not follow task format; excluded from mean(clean).")
    print("Inspect base_sanity/<model>_<benchmark>.txt before interpreting starred deltas.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="directory containing result JSONs (default: $RAIDEX_OUT_ROOT)")
    p.add_argument("--local", action="store_true",
                   help="use LOCAL_PAIRS (Ollama base/instruct) instead of the serverless MODEL_PAIRS")
    p.add_argument("--csv", default=None, help="write delta table to CSV path")
    args = p.parse_args()

    pairs = _normalized_pairs(args.local)
    print(f"Loading base and instruct results ({'local Ollama' if args.local else 'serverless'} pairs)...")
    collected = collect(args.out_dir, pairs)

    if not collected:
        print("No complete pairs found. Run run_bases.py --smoke first.")
        sys.exit(1)

    families = list(collected)
    rows = compute_rows(collected)
    print_table(rows, families)

    if not check_kill_criterion(collected):
        sys.exit(1)

    if args.csv:
        fieldnames = (["benchmark", "dimension", "expected_type"]
                      + families
                      + ["mean_delta_clean", "n_clean_families"])
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
