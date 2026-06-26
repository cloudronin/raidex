"""Generate paper artifacts from the study's result JSONs: Figure 1 (PNG+PDF) and
four result-table CSVs (composite, Test-1 zero-shot deltas, few-shot comparison,
Test-2 correlation). Single source of truth = the JSONs in $RAIDEX_OUT_ROOT.

Run: python pretrain_posttrain/make_artifacts.py
Outputs to docs/pretrain_posttrain/.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scoring
from pairs import LOCAL_PAIRS, local_model_id, JUDGE_BENCHMARKS, PRETRAINING_BOUND, POST_TRAINING_BOUND
import analyze_correlation

OUT_ROOT = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")
REPO = Path(__file__).resolve().parent.parent.parent
DOCS = REPO / "docs" / "pretrain_posttrain"
DOCS.mkdir(parents=True, exist_ok=True)
MCQ = ["bbq", "ethics", "wmdp"]
TIER_A = ["bbq", "ethics", "wmdp", "strongreject", "xstest", "simpleqa"]


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _load(slug_suffix: str, model_id: str):
    p = Path(OUT_ROOT) / (_slug(model_id) + slug_suffix + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def _norm(doc, bid):
    if not doc:
        return None
    r = doc.get("results", {}).get(bid, {})
    return None if r.get("error") else r.get("normalized")


def _predicted(bid):
    if bid in PRETRAINING_BOUND:
        return "pretraining-bound"
    if bid in POST_TRAINING_BOUND:
        return "post-training-bound"
    return "ambiguous"


def write_composite():
    rows = []
    for pair in LOCAL_PAIRS:
        b = _load("", local_model_id(pair["base_tag"]))
        i = _load("", local_model_id(pair["instruct_tag"]))
        br = (b or {}).get("composite", {}).get("rai_score")
        ir = (i or {}).get("composite", {}).get("rai_score")
        d = round(ir - br, 1) if (br is not None and ir is not None) else None
        rows.append([pair["family"], br, ir, d])
    _csv(DOCS / "table1_composite_rai.csv",
         ["family", "base_RAI", "instruct_RAI", "delta"], rows)
    return rows


def write_zeroshot_deltas():
    rows = []
    for bid in TIER_A:
        confounded = bid in JUDGE_BENCHMARKS  # flagged for base runs
        fam_d = {}
        for pair in LOCAL_PAIRS:
            b = _norm(_load("", local_model_id(pair["base_tag"])), bid)
            i = _norm(_load("", local_model_id(pair["instruct_tag"])), bid)
            fam_d[pair["family"]] = round(i - b, 3) if (b is not None and i is not None) else None
        clean_vals = [v for v in fam_d.values() if v is not None] if not confounded else []
        mean_clean = round(sum(clean_vals) / len(clean_vals), 3) if clean_vals else None
        rows.append([bid, scoring.NORM[bid].dimension, _predicted(bid),
                     "yes" if confounded else "no",
                     fam_d["Llama-3.1-8B"], fam_d["Gemma-2-9B"], fam_d["Mistral-7B"], mean_clean])
    _csv(DOCS / "table2_test1_zeroshot_deltas.csv",
         ["benchmark", "dimension", "predicted", "format_confounded",
          "Llama-3.1-8B", "Gemma-2-9B", "Mistral-7B", "mean_clean"], rows)
    return rows


def write_fewshot_compare():
    rows = []
    means = {}
    for bid in MCQ:
        zs_all, fs_all = [], []
        for pair in LOCAL_PAIRS:
            zb = _norm(_load("", local_model_id(pair["base_tag"])), bid)
            zi = _norm(_load("", local_model_id(pair["instruct_tag"])), bid)
            fb = _norm(_load("__fs5", local_model_id(pair["base_tag"])), bid)
            fi = _norm(_load("__fs5", local_model_id(pair["instruct_tag"])), bid)
            zd = round(zi - zb, 3) if (zb is not None and zi is not None) else None
            fd = round(fi - fb, 3) if (fb is not None and fi is not None) else None
            if zd is not None:
                zs_all.append(zd)
            if fd is not None:
                fs_all.append(fd)
            rows.append([bid, _predicted(bid), pair["family"], zd, fd])
        means[bid] = (round(sum(zs_all) / len(zs_all), 3) if zs_all else None,
                      round(sum(fs_all) / len(fs_all), 3) if fs_all else None)
    # append mean rows
    for bid in MCQ:
        rows.append([bid, _predicted(bid), "MEAN", means[bid][0], means[bid][1]])
    _csv(DOCS / "table3_fewshot_vs_zeroshot.csv",
         ["benchmark", "predicted", "family", "zeroshot_delta", "fewshot_delta"], rows)
    return means


def write_correlation():
    from huggingface_hub import snapshot_download
    snap = snapshot_download("cloudronin/raidex-results", repo_type="dataset",
                             token=os.environ.get("HF_TOKEN"))
    cap = REPO / "space" / "data" / "capability_scores.json"
    rows = analyze_correlation.analyze(snap, str(cap))
    out = []
    for r in rows:
        out.append([r["benchmark"], r["dimension"], r["expected_type"],
                    r.get("inverted", False), r["pearson_r"], r["ci_lo"], r["ci_hi"], r["n_models"]])
    _csv(DOCS / "table4_test2_correlation.csv",
         ["benchmark", "dimension", "expected_type", "inverted",
          "pearson_r", "ci_lo", "ci_hi", "n_models"], out)
    return out


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if c is None else c for c in row] for row in rows])
    print(f"  wrote {path.name} ({len(rows)} rows)")


def make_figure(fs_means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["axes.unicode_minus"] = False   # ASCII hyphen for negative ticks, not U+2212

    benches = ["BBQ\n(fairness)", "ETHICS\n(machine ethics)", "WMDP\n(haz. knowledge)"]
    order = ["bbq", "ethics", "wmdp"]
    zs = [fs_means[b][0] for b in order]
    fs = [fs_means[b][1] for b in order]

    x = range(len(benches))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    b1 = ax.bar([i - w / 2 for i in x], zs, w, label="zero-shot",
                color="#85B7EB", edgecolor="#185FA5", linewidth=0.8)
    b2 = ax.bar([i + w / 2 for i in x], fs, w, label="few-shot (5-shot)",
                color="#5DCAA5", edgecolor="#0F6E56", linewidth=0.8, hatch="///")
    ax.axhline(0, color="#444444", linewidth=1.1)
    ax.axvline(1.5, color="#cccccc", linewidth=0.8, linestyle=(0, (4, 4)))
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:+.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom" if h >= 0 else "top",
                        xytext=(0, 3 if h >= 0 else -3), textcoords="offset points", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(benches, fontsize=10)
    ax.set_ylabel("base-to-instruct Δ  (normalized)", fontsize=10)
    ax.set_ylim(-0.12, 0.27)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.5, -0.165, "post-training-bound", ha="center", fontsize=8.5,
            color="#0F6E56", transform=ax.get_xaxis_transform())
    ax.text(2.0, -0.165, "pretraining-bound", ha="center", fontsize=8.5,
            color="#5F5E5A", transform=ax.get_xaxis_transform())
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "figure1_deltas.png", bbox_inches="tight")
    fig.savefig(DOCS / "figure1_deltas.pdf", bbox_inches="tight")
    print(f"  wrote figure1_deltas.png + .pdf")


if __name__ == "__main__":
    print("Writing CSV tables + figure to", DOCS)
    write_composite()
    write_zeroshot_deltas()
    fs_means = write_fewshot_compare()
    try:
        write_correlation()
    except Exception as e:
        print(f"  ! correlation table skipped: {type(e).__name__}: {str(e)[:120]}")
    make_figure(fs_means)
    print("done.")
