"""Test 2: Per-benchmark Pearson r with capability (Artificial Analysis Intelligence Index).

For each Raidex benchmark, computes the Pearson r between normalized RAI score and
Artificial Analysis capability score across all scored roster models, with a bootstrap
95% CI (B=1000). n=13-15 per benchmark, so individual r values carry wide CIs. The
signal is the PATTERN across benchmarks — pretraining-bound dimensions should track
capability; post-training-bound dimensions should not. Do not read any one row as
individually established.

Usage:
  python analyze_correlation.py
  python analyze_correlation.py --results-dir /tmp/raidex
  python analyze_correlation.py --cap-json ../../space/data/capability_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scoring
from pairs import PRETRAINING_BOUND, POST_TRAINING_BOUND

_HERE = Path(__file__).resolve().parent
_DEFAULT_CAP_JSON = _HERE.parent.parent / "space" / "data" / "capability_scores.json"
OUT_DIR = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results(results_dir: str) -> list[dict]:
    docs = []
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        if not (isinstance(doc, dict) and "config" in doc and "results" in doc):
            continue
        if doc.get("config", {}).get("is_base_model"):
            continue
        docs.append(doc)
    return docs


def _results_dir_effective(user_dir: str) -> str:
    """Use user_dir if it has JSON files, otherwise try the HF snapshot."""
    if list(Path(user_dir).glob("*.json")):
        return user_dir
    print(f"No JSONs in {user_dir}, trying HF snapshot...")
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download("cloudronin/raidex-results", repo_type="dataset",
                                 token=os.environ.get("HF_TOKEN"))
        return snap
    except Exception as e:
        print(f"HF snapshot failed: {e}")
        return user_dir


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(xs, ys, B: int = 1000, seed: int = 42) -> tuple[float, float]:
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(xs)
    boot = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        bx, by = xs[idx], ys[idx]
        if bx.std() < 1e-10 or by.std() < 1e-10:
            continue
        r = float(np.corrcoef(bx, by)[0, 1])
        if r == r:
            boot.append(r)
    if len(boot) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _expected_type(bid: str) -> str:
    if bid in PRETRAINING_BOUND:
        return "pretraining-bound"
    if bid in POST_TRAINING_BOUND:
        return "post-training-bound"
    return "ambiguous"


def analyze(results_dir: str, cap_json_path: str) -> list[dict]:
    import numpy as np

    cap_data = json.loads(Path(cap_json_path).read_text())
    cap_scores: dict[str, float] = cap_data.get("scores", cap_data)

    docs = _load_results(results_dir)
    if not docs:
        print(f"No instruct result JSONs found in {results_dir}")
        return []

    records = []
    for doc in docs:
        cfg = doc["config"]
        model_name = cfg.get("model_name") or cfg.get("model_id", "").split("/")[-1]
        cap = cap_scores.get(model_name)
        if cap is None:
            continue
        norm_by_bid = {}
        for bid, r in doc.get("results", {}).items():
            if bid not in scoring.NORM:
                continue
            if r.get("error") or r.get("value") is None:
                continue
            nv = r.get("normalized")
            if nv is not None:
                norm_by_bid[bid] = nv
        if norm_by_bid:
            records.append({"model": model_name, "capability": float(cap), "scores": norm_by_bid})

    if not records:
        print("No models found with both capability scores and RAI results.")
        return []

    matched = [r["model"] for r in records]
    all_names = [doc["config"].get("model_name") or doc["config"].get("model_id", "").split("/")[-1]
                 for doc in docs]
    unmatched = [n for n in all_names if n not in cap_scores]
    print(f"Matched {len(records)} models to capability scores.")
    if unmatched:
        print(f"  No capability score for: {', '.join(unmatched)}")

    rows = []
    for bid in sorted(scoring.NORM.keys()):
        spec = scoring.NORM[bid]
        pts = [(r["capability"], r["scores"][bid])
               for r in records if bid in r["scores"]]
        n = len(pts)

        # lower_is_better benchmarks (wmdp, strongreject) store a normalized score that
        # INVERTS raw performance. r is computed on the normalized score (the RAI dimension,
        # per the spec's "correlate each RAI dimension"); for inverted benchmarks a negative
        # normalized-r therefore means raw benchmark performance RISES with capability.
        inverted = not spec.higher_is_better

        if n < 3:
            rows.append({
                "benchmark": bid, "dimension": spec.dimension,
                "expected_type": _expected_type(bid), "inverted": inverted,
                "pearson_r": None, "ci_lo": None, "ci_hi": None,
                "n_models": n, "note": f"too few ({n})",
            })
            continue

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        if xs.std() < 1e-10 or ys.std() < 1e-10:
            rows.append({
                "benchmark": bid, "dimension": spec.dimension,
                "expected_type": _expected_type(bid), "inverted": inverted,
                "pearson_r": None, "ci_lo": None, "ci_hi": None,
                "n_models": n, "note": "no variance",
            })
            continue

        r_val = float(np.corrcoef(xs, ys)[0, 1])
        ci_lo, ci_hi = _bootstrap_ci(xs, ys)
        rows.append({
            "benchmark":     bid,
            "dimension":     spec.dimension,
            "expected_type": _expected_type(bid),
            "inverted":      inverted,
            "pearson_r":     round(r_val, 3),
            "ci_lo":         round(ci_lo, 3) if ci_lo == ci_lo else None,
            "ci_hi":         round(ci_hi, 3) if ci_hi == ci_hi else None,
            "n_models":      n,
            "note":          "",
        })

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    if not rows:
        return

    header = (f"{'benchmark':<14} {'dimension':<16} {'expected_type':<20} "
              f"{'r (norm)':>9}  {'bootstrap 95% CI':^20}  {'n':>4}  note")
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    any_inverted = False
    for row in rows:
        r = row["pearson_r"]
        lo, hi = row["ci_lo"], row["ci_hi"]
        if r is not None:
            r_str = f"{r:+.3f}"
            if lo is not None and hi is not None:
                ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
            else:
                ci_str = "      —          "
        else:
            r_str = "   —  "
            ci_str = "      —          "
        bench = row["benchmark"] + (" †" if row.get("inverted") else "")
        any_inverted = any_inverted or row.get("inverted")
        note = row.get("note", "")
        print(f"{bench:<14} {row['dimension']:<16} {row['expected_type']:<20} "
              f"{r_str:>9}  {ci_str:<20}  {row['n_models']:>4}  {note}")

    print(sep)
    print()
    if any_inverted:
        print("† lower-is-better benchmark: normalized score inverts raw performance, so a")
        print("  NEGATIVE r here means raw benchmark performance RISES with capability.")
        print("  (wmdp −r = hazardous-knowledge accuracy tracks capability = pretraining-bound,")
        print("   which is also the expected 'WMDP moves opposite to other dimensions' exception.)")
        print()
    print("Caution: n=17 here; ~13-15 typical per benchmark. Bootstrap CIs are wide — expected.")
    print("Read the pattern, not individual rows:")
    print("  Prediction: pretraining-bound (wmdp, simpleqa) → tracks capability")
    print("              post-training-bound (strongreject, xstest, bbq, ethics) → r ≈ 0")
    print("  CI spanning zero means the sign is uncertain for that benchmark individually.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=OUT_DIR,
                   help="directory with result JSONs; falls back to HF snapshot if empty")
    p.add_argument("--cap-json", default=str(_DEFAULT_CAP_JSON),
                   help="path to capability_scores.json")
    args = p.parse_args()

    if not Path(args.cap_json).exists():
        print(f"Capability JSON not found: {args.cap_json}")
        sys.exit(1)

    results_dir = _results_dir_effective(args.results_dir)
    rows = analyze(results_dir, args.cap_json)
    if not rows:
        sys.exit(1)

    print_table(rows)


if __name__ == "__main__":
    main()
