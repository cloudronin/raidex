"""Run base model checkpoints through the Raidex Tier A pipeline.

Usage:
  python run_bases.py --smoke                      # n=50, no-upload, dump 5 raw judge completions
  python run_bases.py --full                       # full Tier A run, upload results
  python run_bases.py --smoke --family Gemma-3     # single family smoke test
  python run_bases.py --full --no-upload           # full run, local files only

What this does beyond runner.py:
  1. Canary call — checks each base model ID is accessible before committing to a full run.
  2. Smoke dump — in --smoke mode, dumps 5 raw completions per judge benchmark so you can
     eyeball whether a low base score means 'no alignment' or 'could not parse the task'.
  3. Format-confound flag — adds format_confounded=true to judge benchmark blocks in the
     result JSON, so analyze_delta.py can label those deltas rather than treat them as clean.
  4. Instruct cross-check — verifies that the instruct model's stored normalized values
     match what scoring.normalize() computes from raw values. Fails loudly on mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add backend/ to path so imports from runner, scoring, benchmarks work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scoring
from pairs import MODEL_PAIRS, JUDGE_BENCHMARKS

OUT_DIR = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")
RESULTS_REPO = "cloudronin/raidex-results"


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _result_path(model_id: str, out_dir: str) -> Path:
    return Path(out_dir) / (_model_slug(model_id) + ".json")


def _verify_instruct(model_id: str) -> dict | None:
    """Load the instruct result JSON and verify stored normalized values are consistent.

    Recomputes each normalized value from the raw value using scoring.normalize() and
    fails loudly if the stored value doesn't match. This catches cases where the
    leaderboard JSON was produced with a different normalization formula — the delta
    would then be computed against a bad baseline.

    Returns the result dict on success, None if the JSON can't be found.
    """
    from runner import _load_existing_result
    result = _load_existing_result(model_id)
    if result is None:
        print(f"  ! Instruct result not found for {model_id} — delta will be unavailable")
        return None

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
            mismatches.append(f"{bid}: stored={stored}, recomputed={recomputed}")

    if mismatches:
        raise ValueError(
            f"Instruct result scoring mismatch for {model_id}:\n"
            + "\n".join(f"  {m}" for m in mismatches)
            + "\nThe leaderboard JSON may be from a different scoring version. Aborting."
        )
    return result


def _check_base_available(model_id: str) -> bool:
    """Canary call: one minimal completion to confirm the base model ID resolves."""
    from benchmarks._direct import complete
    try:
        complete(model_id, "Hello", max_tokens=5)
        return True
    except Exception as e:
        print(f"  SKIP {model_id}: not accessible — {type(e).__name__}: {str(e)[:140]}")
        return False


def _dump_judge_completions(model_id: str, out_dir: str, n: int = 5) -> None:
    """Dump n raw base model completions per judge benchmark for manual review.

    Saves to <out_dir>/base_sanity/<model_slug>_<benchmark>.txt. Inspect these
    before interpreting the delta for format_confounded benchmarks.
    """
    from benchmarks._direct import complete

    sanity_dir = Path(out_dir) / "base_sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(model_id)

    def _write(bench_id: str, lines: list[str]) -> None:
        out = sanity_dir / f"{slug}_{bench_id}.txt"
        out.write_text("\n".join(lines))
        print(f"  Dumped {n} completions → {out}")

    # StrongREJECT: forbidden jailbreak prompts
    try:
        from strong_reject.load_datasets import load_strongreject
        ds = load_strongreject()
        prompts = list(ds["forbidden_prompt"])[:n]
        lines = [
            f"=== {model_id} :: strongreject (n={len(prompts)}) ===",
            "Context: Does base model continue jailbreak prompts or refuse?",
            "Low base score = high ASR = complied. Check: is response substantive or garbled?\n",
        ]
        for i, p in enumerate(prompts):
            resp = complete(model_id, p, max_tokens=256)
            lines += [f"--- Prompt {i+1} ---", p, "--- Response ---", resp, ""]
        _write("strongreject", lines)
    except Exception as e:
        print(f"  ! strongreject dump failed: {e}")

    # XSTest: safe and unsafe prompts
    try:
        from datasets import load_dataset
        ds = load_dataset("Paul/XSTest")
        split = "test" if "test" in ds else list(ds.keys())[0]
        rows = list(ds[split])[:n]
        lines = [
            f"=== {model_id} :: xstest (n={len(rows)}) ===",
            "Context: Does base model comply with all prompts (no refusal training)?",
            "Check: are responses coherent, or does the model fail to follow the task format?\n",
        ]
        for i, r in enumerate(rows):
            resp = complete(model_id, r["prompt"], max_tokens=256)
            lines += [f"--- Prompt {i+1} [label={r['label']}] ---", r["prompt"],
                      "--- Response ---", resp, ""]
        _write("xstest", lines)
    except Exception as e:
        print(f"  ! xstest dump failed: {e}")

    # SimpleQA: factual questions
    try:
        import urllib.request
        import pandas as pd
        csv_path = str(Path(out_dir) / "simple_qa_test_set.csv")
        if not os.path.exists(csv_path):
            urllib.request.urlretrieve(
                "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
                csv_path,
            )
        df = pd.read_csv(csv_path).head(n)
        lines = [
            f"=== {model_id} :: simpleqa (n={len(df)}) ===",
            "Context: Does base model give factual answers in a format the judge can classify?",
            "Check: is the response an answer (not a continuation of the question text)?\n",
        ]
        for i, rec in enumerate(df.to_dict("records")):
            resp = complete(model_id, rec["problem"], max_tokens=256)
            lines += [f"--- Q{i+1}: {rec['problem']}", f"--- Gold: {rec['answer']}",
                      "--- Response ---", resp, ""]
        _write("simpleqa", lines)
    except Exception as e:
        print(f"  ! simpleqa dump failed: {e}")


def _flag_format_confounded(model_id: str, out_dir: str) -> None:
    """Load the result JSON produced by run_eval, add format_confounded flag, re-write."""
    path = _result_path(model_id, out_dir)
    if not path.exists():
        print(f"  ! Cannot flag format_confounded: {path} not found")
        return
    with open(path) as f:
        data = json.load(f)
    for bid in JUDGE_BENCHMARKS:
        if bid in data.get("results", {}):
            data["results"][bid]["format_confounded"] = True
    data.setdefault("config", {})["is_base_model"] = True
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Flagged judge benchmarks as format_confounded in {path.name}")


def _upload(model_id: str, out_dir: str) -> None:
    from huggingface_hub import HfApi
    fn = _model_slug(model_id) + ".json"
    fp = str(_result_path(model_id, out_dir))
    HfApi().upload_file(
        path_or_fileobj=fp, path_in_repo=fn,
        repo_id=RESULTS_REPO, repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  Uploaded {fn} → {RESULTS_REPO}")


def run_pair(pair: dict, smoke: bool, no_upload: bool, out_dir: str) -> bool:
    """Run one base model. Returns True if the run succeeded (even partially)."""
    family = pair["family"]
    base_id = pair["base"]
    instruct_id = pair["instruct"]
    limit = 50 if smoke else None

    print(f"\n==== {family}: base={base_id} ====")

    # Verify instruct baseline before running (fail loud if scoring version skew).
    try:
        inst_result = _verify_instruct(instruct_id)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return False
    if inst_result is None:
        return False
    print(f"  Instruct cross-check passed for {instruct_id}")

    # Canary: confirm the base model ID is accessible.
    if not _check_base_available(base_id):
        return False

    # Smoke: dump raw judge-benchmark completions for eyeball review.
    if smoke:
        print(f"  Dumping {5} raw judge-benchmark completions for format-confound review...")
        _dump_judge_completions(base_id, out_dir)

    # Run Tier A through the existing runner (always local-only; we manage upload below).
    from runner import run_eval
    print(f"  Running Tier A (limit={limit}) ...")
    try:
        run_eval(base_id, tier="A", dry_run=False, limit=limit, no_upload=True)
    except Exception as e:
        print(f"  ! run_eval failed for {base_id}: {e}")
        return False

    # Add format-confound flag and is_base_model marker to the produced JSON.
    _flag_format_confounded(base_id, out_dir)

    if not no_upload:
        _upload(base_id, out_dir)

    print(f"  Done: {family} base complete.")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Raidex Tier A on base model checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="n=50, no-upload, dump 5 raw judge completions per model")
    mode.add_argument("--full", action="store_true",
                      help="full Tier A run, upload results to HF")
    p.add_argument("--family", type=str, default=None,
                   help="restrict to one family, e.g. 'Gemma-3'")
    p.add_argument("--no-upload", action="store_true",
                   help="skip HF upload even in --full mode")
    p.add_argument("--out-dir", type=str, default=OUT_DIR,
                   help="directory for result JSONs (default: $RAIDEX_OUT_ROOT)")
    args = p.parse_args()

    pairs = MODEL_PAIRS
    if args.family:
        pairs = [q for q in pairs if q["family"] == args.family]
        if not pairs:
            known = [q["family"] for q in MODEL_PAIRS]
            p.error(f"Unknown family {args.family!r}. Known: {known}")

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for pair in pairs:
        ok = run_pair(pair, smoke=args.smoke,
                      no_upload=(args.no_upload or args.smoke),
                      out_dir=args.out_dir)
        results.append((pair["family"], ok))

    print(f"\n==== run_bases.py complete ====")
    for fam, ok in results:
        print(f"  {fam}: {'OK' if ok else 'SKIP/FAILED'}")


if __name__ == "__main__":
    main()
