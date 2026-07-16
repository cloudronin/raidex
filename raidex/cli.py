"""raidex CLI — measure a model against the Raidex Responsible-AI index, locally.

    raidex eval --model openai/gpt-5.2 --tier A
    raidex eval --model http://localhost:8000/v1 --served-name my-model --tier A+B
    raidex eval --model ... --benchmarks bbq,strongreject --judge anthropic/claude-opus-4-8
    raidex eval --model ... --dry-run
    raidex eval --model ... --offline --output results.json
    raidex fetch-data                 # pre-populate the cache for offline / air-gapped use

Runs the selected benchmarks against your model, prints per-dimension + composite RAI
scores + coverage (N/9), and writes a self-describing JSON. Nothing is uploaded; there is
no account, no queue, and no dependency on Raidex servers. Scores are identical in scale to
the published leaderboard (same core, same benchmarks, same normalization).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import __version__
from .core import data
from .core.eval import (
    JUDGE_BENCHMARKS, SAMPLE_EXEMPT, TIERS, estimate_costs, evaluate, resolve_benches,
)

# Map a litellm provider prefix to the env var that holds its key (for judge availability).
_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY", "mistral": "MISTRAL_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}
_JUDGE_DEFAULTS = {"simpleqa": "anthropic/claude-sonnet-4-6",
                   "xstest": "anthropic/claude-sonnet-4-6",
                   "strongreject": "openai/gpt-4o-mini"}


def _translate_model(model: str, served_name: str | None) -> tuple[str, dict]:
    """A URL model -> a local OpenAI-compatible endpoint: set OPENAI_API_BASE + a synthetic
    ``openai/<served>`` id (so both the direct path and the lm-eval proxy hit it). Returns
    (model_id, provenance-model-dict)."""
    if re.match(r"^https?://", model):
        os.environ["OPENAI_API_BASE"] = model
        os.environ.setdefault("OPENAI_API_KEY", "sk-local")   # litellm requires some value
        served = served_name or "local-model"
        model_id = f"openai/{served}"
        # Shift the lm-eval proxy port if it collides with the target endpoint's port.
        m = re.search(r":(\d+)", model)
        if m and m.group(1) == os.environ.get("RAIDEX_PROXY_PORT", "8000"):
            os.environ["RAIDEX_PROXY_PORT"] = str(int(m.group(1)) + 1)
        return model_id, {"input": model, "model_id": model_id, "api_base": model,
                          "served_name": served}
    return model, {"input": model, "model_id": model, "api_base": None, "served_name": None}


def _judge_model_for(bid: str) -> str:
    from .core.benchmarks import _direct
    return _direct.judge_model(bid, default=_JUDGE_DEFAULTS.get(bid, "openai/gpt-4o"))


def _judge_available(bid: str) -> tuple[bool, str]:
    """Is the judge for this benchmark usable? (its provider key is present). Returns
    (available, judge_model)."""
    jm = _judge_model_for(bid)
    env = _PROVIDER_ENV.get(jm.split("/")[0])
    available = env is None or bool(os.environ.get(env))   # unknown provider -> assume configured
    return available, jm


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.offline:
        data.set_offline(True)
    if args.judge:
        os.environ["RAIDEX_JUDGE_MODEL"] = args.judge

    model_id, model_prov = _translate_model(args.model, args.served_name)

    bench_ids = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    benches = resolve_benches(tier=None if bench_ids else args.tier, benchmark_ids=bench_ids)

    # Graceful judge degradation: drop judge benchmarks whose judge key is missing.
    skipped: dict[str, str] = {}
    kept = []
    for b in benches:
        if b.id in JUDGE_BENCHMARKS:
            ok, jm = _judge_available(b.id)
            if not ok:
                skipped[b.id] = f"no judge configured (needs a key for {jm}; pass --judge)"
                continue
        kept.append(b)
    if skipped:
        for bid, why in skipped.items():
            print(f"Skipping {bid}: {why}", file=sys.stderr)
        print(f"Coverage will be {len(kept)}/{len(TIERS['A+B'])} (judge benchmarks skipped: "
              f"{', '.join(skipped)}).", file=sys.stderr)
    if not kept:
        print("Nothing to run (all selected benchmarks were skipped).", file=sys.stderr)
        return 2

    if args.dry_run:
        est = estimate_costs(model_id, kept, args.limit)
        print(f"Cost estimate — {model_id}:")
        for bid, c in est.items():
            print(f"  {bid:14s} ${c:8.2f}")
        print(f"  {'TOTAL':14s} ${round(sum(est.values()), 2):8.2f}")
        return 0

    output = evaluate(model_id, kept, limit=args.limit)
    output["provenance"] = _provenance(model_prov, args, kept, skipped)

    _print_summary(output)
    out_path = args.output or (model_id.replace("/", "__") + ".json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")
    return 0


def _provenance(model_prov: dict, args, kept, skipped) -> dict:
    from .core.benchmarks import _direct
    from .core.eval import _utc_iso, CORE_VERSION
    ran = [b.id for b in kept]
    judged = sorted(set(ran) & JUDGE_BENCHMARKS)
    return {
        "tool": "raidex-cli",
        "cli_version": __version__,
        "raidex_core_version": CORE_VERSION,
        "generated_at": _utc_iso(),
        "model": model_prov,
        "tier": None if args.benchmarks else args.tier,
        "benchmarks_run": ran,
        "benchmarks_skipped": skipped,
        "judge": {
            "configured": bool(args.judge) or bool(os.environ.get("RAIDEX_JUDGE_MODEL")),
            "model": os.environ.get("RAIDEX_JUDGE_MODEL"),
            "per_benchmark": {bid: _judge_model_for(bid) for bid in judged},
        },
        "sampling": {
            "limit": args.limit,
            "sample_exempt": sorted(SAMPLE_EXEMPT),
            "concurrency": int(os.environ.get("RAIDEX_CONCURRENCY", "8")),
            "num_retries": int(os.environ.get("RAIDEX_NUM_RETRIES", "6")),
            "timeout": float(os.environ.get("RAIDEX_TIMEOUT", "120")),
            "max_failure_rate": float(os.environ.get("RAIDEX_MAX_FAILURE_RATE", "0.25")),
        },
        "datasets": {b.id: data.pin(b.id) for b in kept},
        "offline": data.is_offline(),
    }


def _print_summary(output: dict) -> None:
    c = output["composite"]
    print("\n=== Raidex RAI Score ===")
    print(f"  RAI Score : {c['rai_score']}")
    print(f"  Coverage  : {c['rai_coverage']}  {c['badge_emoji']}")
    print("  Dimensions:")
    for dim, v in (c.get("dimension_scores") or {}).items():
        print(f"    {dim:16s} {'—' if v is None else v}")


def _cmd_fetch_data(args: argparse.Namespace) -> int:
    """Pre-populate the cache so `--offline` runs with no network."""
    print(f"Caching benchmark data into {data.data_dir()} ...")
    # Direct-download sources (verified by sha256).
    data.cached_file("simpleqa/simple_qa_test_set.csv", data.pin("simpleqa")["source"],
                     sha256=data.pin("simpleqa").get("sha256"))
    data.cached_file("strongreject/strongreject_dataset.csv", data.pin("strongreject")["source"],
                     sha256=data.pin("strongreject").get("sha256"))
    for fname, sha in (data.pin("confaide").get("sha256") or {}).items():
        data.cached_file(f"confaide/{fname}", f"{data.pin('confaide')['source']}/{fname}", sha256=sha)
    # HF datasets, pinned to their revisions.
    from huggingface_hub import snapshot_download
    for bid in ("bbq", "wmdp", "ethics", "xstest", "advglue", "sycophancy"):
        rev = data.revision(bid)
        try:
            snapshot_download(data.pin(bid)["source"], repo_type="dataset", revision=rev)
            print(f"  cached {bid} @ {rev[:12] if rev else 'latest'}")
        except Exception as e:
            print(f"  ! {bid}: {str(e)[:80]}", file=sys.stderr)
    print("Done. Copy the cache dir to an air-gapped machine, then run with --offline.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="raidex", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"raidex {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="evaluate a model against the Raidex index")
    e.add_argument("--model", required=True,
                   help="litellm model id (e.g. openai/gpt-5.2) OR a local endpoint URL")
    e.add_argument("--served-name", help="model name served by a local endpoint URL (default: local-model)")
    e.add_argument("--tier", default="A", choices=list(TIERS), help="benchmark tier (default: A)")
    e.add_argument("--benchmarks", help="comma-separated benchmark ids (overrides --tier)")
    e.add_argument("--limit", type=int, default=None, help="sample N prompts (big benchmarks only)")
    e.add_argument("--judge", help="LLM judge model for SimpleQA/XSTest/StrongREJECT")
    e.add_argument("--dry-run", action="store_true", help="print a cost estimate and exit")
    e.add_argument("--offline", action="store_true", help="use only cached data; never touch the network")
    e.add_argument("--output", help="write the result JSON here (default: <model>__.json)")
    e.set_defaults(func=_cmd_eval)

    f = sub.add_parser("fetch-data", help="pre-download + cache all benchmark data (for offline use)")
    f.set_defaults(func=_cmd_fetch_data)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
