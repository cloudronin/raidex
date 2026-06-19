"""Raidex — an open Responsible AI index for frontier models (HuggingFace Space).

Reads model evaluations from the raidex-results dataset and renders a leaderboard,
the capability-vs-RAI "gap" visual, model cards, and a submit form that queues new
evaluations into the raidex-requests dataset.

Storage is abstracted behind a local/HF switch (RAIDEX_DATA_SOURCE): development
reads local dataset folders (or a bundled seed/), production reads/writes the HF
Hub. Only load_results / get_pending / get_completed / submit_eval touch storage.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
DATA_SOURCE = os.environ.get("RAIDEX_DATA_SOURCE", "local").lower()
RESULTS_REPO = "cloudronin/raidex-results"
REQUESTS_REPO = "cloudronin/raidex-requests"
SEED_DIR = HERE / "seed"
SIB_RESULTS = HERE.parent / "raidex-results"
SIB_REQUESTS = HERE.parent / "raidex-requests"

TOTAL_BENCHMARKS = 8
BENCHMARKS = [
    {"id": "bbq", "label": "BBQ", "dim": "fairness_bias", "tier": "A"},
    {"id": "wmdp", "label": "WMDP", "dim": "security", "tier": "A"},
    {"id": "simpleqa", "label": "SimpleQA", "dim": "factuality", "tier": "A"},
    {"id": "strongreject", "label": "StrongREJECT", "dim": "security", "tier": "A"},
    {"id": "ethics", "label": "ETHICS", "dim": "machine_ethics", "tier": "A"},
    {"id": "xstest", "label": "XSTest", "dim": "safety", "tier": "A"},
    {"id": "advglue", "label": "AdvGLUE", "dim": "robustness", "tier": "B"},
    {"id": "confaide", "label": "ConfAIde", "dim": "privacy", "tier": "B"},
]
BENCH_LABELS = [b["label"] for b in BENCHMARKS]
DIMENSION_ORDER = ["safety", "fairness_bias", "factuality", "security",
                   "robustness", "privacy", "machine_ethics"]
DIM_LABEL = {"safety": "Safety", "fairness_bias": "Fairness & Bias", "factuality": "Factuality",
             "security": "Security", "robustness": "Robustness", "privacy": "Privacy",
             "machine_ethics": "Machine Ethics"}
ACTIVE_DIMS = ["safety", "fairness_bias", "factuality", "security", "machine_ethics"]
BADGE_LEGEND = ("🟣 Full RAI Profile (8/8)  ·  🔵 Independently Evaluated  ·  "
                "🟡 Self-Reported Only  ·  ⚪ Partial Coverage")
MODEL_ID_RE = re.compile(r"^[a-z0-9_\-]+/[A-Za-z0-9._:\-]+$")

CITATION_TEXT = """@misc{raidex2026,
  title  = {Raidex: An Open Responsible AI Index for Frontier Models},
  author = {Vettrivel, Vishnu},
  year   = {2026},
  url    = {https://raidex.ai}
}"""


def _read_text(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text()
    except Exception:
        return fallback


CAP = json.loads(_read_text(HERE / "data" / "capability_benchmarks.json",
                            '{"benchmarks": [], "models": {}}'))
KEY_FINDINGS_MD = _read_text(HERE / "findings.md", "_Key findings will appear here after evaluation runs._")
METHODOLOGY_MD = _read_text(HERE / "METHODOLOGY.md", "METHODOLOGY.md not found.")


# ----------------------------------------------------------------------------
# Storage layer — the ONLY place that branches on local vs HF.
# ----------------------------------------------------------------------------
def _hf_snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, repo_type="dataset")


def _results_dir() -> str:
    if DATA_SOURCE == "hf":
        try:
            return _hf_snapshot(RESULTS_REPO)
        except Exception as e:  # fall back to local/seed so the app still renders
            print("[raidex] HF results snapshot failed, using local/seed:", e)
    for cand in [os.environ.get("RAIDEX_RESULTS_DIR"), str(SIB_RESULTS), str(SEED_DIR)]:
        if cand and os.path.isdir(cand) and any(Path(cand).glob("*.json")):
            return cand
    return str(SEED_DIR)


def _requests_dir() -> str:
    if DATA_SOURCE == "hf":
        try:
            return _hf_snapshot(REQUESTS_REPO)
        except Exception as e:
            print("[raidex] HF requests snapshot failed:", e)
    for cand in [os.environ.get("RAIDEX_REQUESTS_DIR"), str(SIB_REQUESTS)]:
        if cand and os.path.isdir(cand):
            return cand
    return str(SIB_REQUESTS)


def _write_request(filename: str, obj: dict) -> None:
    if DATA_SOURCE == "hf":
        from huggingface_hub import HfApi
        tmp = Path(tempfile.gettempdir()) / filename
        tmp.write_text(json.dumps(obj, indent=2))
        HfApi().upload_file(path_or_fileobj=str(tmp), path_in_repo=filename,
                            repo_id=REQUESTS_REPO, repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
        return
    d = os.environ.get("RAIDEX_REQUESTS_DIR") or str(SIB_REQUESTS)
    os.makedirs(d, exist_ok=True)
    (Path(d) / filename).write_text(json.dumps(obj, indent=2))


# ----------------------------------------------------------------------------
# Load + transform
# ----------------------------------------------------------------------------
def _iter_result_docs(dir_path: str):
    for f in sorted(Path(dir_path).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(doc, dict) and "config" in doc and "results" in doc:
            yield doc


def load_results() -> pd.DataFrame:
    rows = []
    for doc in _iter_result_docs(_results_dir()):
        cfg, comp, res = doc.get("config", {}), doc.get("composite", {}), doc.get("results", {})
        row = {
            "Badge": comp.get("badge_emoji", "⚪"),
            "Model": cfg.get("model_name") or cfg.get("model_id", "?"),
            "Developer": cfg.get("developer", ""),
            "RAI Score": comp.get("rai_score"),
            "Coverage": comp.get("rai_coverage", ""),
            "_model_id": cfg.get("model_id", ""),
            "_tiers": set(),
            "_sources": set(),
        }
        for b in BENCHMARKS:
            r = res.get(b["id"]) or {}
            norm = r.get("normalized")
            if norm is not None and not r.get("error"):
                row[b["label"]] = round(norm * 100, 1)
                row["_tiers"].add(b["tier"])
                row["_sources"].add(r.get("eval_source", ""))
            else:
                row[b["label"]] = None
        for dim in DIMENSION_ORDER:
            row["dim_" + dim] = comp.get("dimension_scores", {}).get(dim)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("RAI Score", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df


LEADERBOARD = load_results()
DISPLAY_COLS = ["Rank", "Badge", "Model", "Developer", "RAI Score", "Coverage"] + BENCH_LABELS


def _display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=DISPLAY_COLS)
    return df[[c for c in DISPLAY_COLS if c in df.columns]]


def refresh():
    global LEADERBOARD
    LEADERBOARD = load_results()
    return _display(LEADERBOARD)


def filter_leaderboard(search: str, tiers):
    df = LEADERBOARD
    if df is None or df.empty:
        return _display(df)
    mask = pd.Series(True, index=df.index)
    if search:
        s = search.lower()
        mask &= (df["Model"].str.lower().str.contains(s, na=False)
                 | df["Developer"].str.lower().str.contains(s, na=False))
    sel = {t.split()[-1] for t in tiers} if tiers else {"A"}
    mask &= df["_tiers"].apply(lambda ts: bool(set(ts) & sel) if ts else False)
    return _display(df[mask])


def model_choices():
    if LEADERBOARD is None or LEADERBOARD.empty:
        return []
    return list(LEADERBOARD["Model"])


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
GREEN = [[0.0, "#0b3d2e"], [1.0, "#16a34a"]]
RED = [[0.0, "#3d0b0b"], [1.0, "#dc2626"]]
_LAYOUT = dict(width=1200, height=630, paper_bgcolor="white", plot_bgcolor="white",
               font=dict(size=16), margin=dict(l=160, r=40, t=80, b=120))


def _empty_fig(title: str):
    fig = go.Figure()
    fig.update_layout(title=title, **_LAYOUT)
    fig.add_annotation(text="No data yet", showarrow=False, font=dict(size=24, color="#888"))
    return fig


def build_capability_heatmap():
    """Which capability benchmarks each frontier developer self-reports (sourced grid)."""
    benches = CAP.get("capability_benchmarks", [])
    models = list(CAP.get("models", {}).keys())
    if not benches or not models:
        return _empty_fig("Capability benchmarks")
    z = [CAP["models"][m].get("capability", []) for m in models]
    fig = go.Figure(go.Heatmap(z=z, x=benches, y=models, colorscale=GREEN,
                               showscale=False, xgap=3, ygap=3, zmin=0, zmax=1))
    fig.update_layout(title="<b>Capability benchmarks: widely self-reported</b>", **_LAYOUT)
    fig.update_xaxes(tickangle=-40)
    return fig


def build_rai_heatmap():
    """Which RAI benchmarks each frontier developer self-reports (sourced grid) — sparse.
    This is reporting, not Raidex's coverage: our leaderboard is what fills the gap."""
    benches = CAP.get("rai_benchmarks", [])
    models = list(CAP.get("models", {}).keys())
    if not benches or not models:
        return _empty_fig("RAI benchmarks")
    z = [CAP["models"][m].get("rai", []) for m in models]
    fig = go.Figure(go.Heatmap(z=z, x=benches, y=models, colorscale=RED,
                               showscale=False, xgap=3, ygap=3, zmin=0, zmax=1))
    fig.update_layout(title="<b>RAI benchmarks: rarely self-reported</b>", **_LAYOUT)
    fig.update_xaxes(tickangle=-40)
    return fig


def build_radar(models):
    fig = go.Figure()
    if LEADERBOARD is None or LEADERBOARD.empty or not models:
        fig.update_layout(title="Select models to compare")
        return fig
    for m in models:
        sub = LEADERBOARD[LEADERBOARD["Model"] == m]
        if sub.empty:
            continue
        row = sub.iloc[0]
        r = [row.get("dim_" + d) or 0 for d in ACTIVE_DIMS]
        fig.add_trace(go.Scatterpolar(r=r + [r[0]],
                                      theta=[DIM_LABEL[d] for d in ACTIVE_DIMS] + [DIM_LABEL[ACTIVE_DIMS[0]]],
                                      fill="toself", name=m))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      title="Per-dimension comparison", height=520)
    return fig


def build_coverage_scatter():
    df = LEADERBOARD
    if df is None or df.empty:
        return _empty_fig("Coverage vs RAI Score")
    cov = df["Coverage"].apply(lambda c: int(str(c).split("/")[0]) if c else 0)
    fig = go.Figure(go.Scatter(x=cov, y=df["RAI Score"], mode="markers+text",
                               text=df["Model"], textposition="top center",
                               marker=dict(size=14)))
    fig.update_layout(title="Coverage vs RAI Score", xaxis_title="Benchmarks evaluated (/8)",
                      yaxis_title="RAI Score", height=520)
    return fig


def build_model_radar(model: str):
    fig = go.Figure()
    if LEADERBOARD is None or LEADERBOARD.empty or not model:
        return fig
    sub = LEADERBOARD[LEADERBOARD["Model"] == model]
    if sub.empty:
        return fig
    row = sub.iloc[0]
    r = [row.get("dim_" + d) or 0 for d in ACTIVE_DIMS]
    theta = [DIM_LABEL[d] for d in ACTIVE_DIMS]
    mean = [LEADERBOARD["dim_" + d].dropna().mean() if "dim_" + d in LEADERBOARD else 0 for d in ACTIVE_DIMS]
    mean = [0 if pd.isna(x) else x for x in mean]
    fig.add_trace(go.Scatterpolar(r=r + [r[0]], theta=theta + [theta[0]], fill="toself", name=model))
    fig.add_trace(go.Scatterpolar(r=mean + [mean[0]], theta=theta + [theta[0]], name="Roster mean"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=480,
                      title=f"{model} vs roster mean")
    return fig


def model_card(model: str):
    if not model or LEADERBOARD is None or LEADERBOARD.empty:
        return "Select a model.", go.Figure(), pd.DataFrame(), ""
    sub = LEADERBOARD[LEADERBOARD["Model"] == model]
    if sub.empty:
        return "Model not found.", go.Figure(), pd.DataFrame(), ""
    row = sub.iloc[0]
    summary = (f"### {row['Badge']} {model}\n"
               f"- **Developer:** {row['Developer']}\n"
               f"- **RAI Score:** {row['RAI Score']}\n"
               f"- **Coverage:** {row['Coverage']}\n"
               f"- **Model ID:** `{row['_model_id']}`")
    tbl = pd.DataFrame({"Benchmark": BENCH_LABELS,
                        "Normalized (0-100)": [row.get(b["label"]) for b in BENCHMARKS]})
    cap_note = ("*Capability-vs-RAI rank comparison appears once capability data is populated.*")
    return summary, build_model_radar(model), tbl, cap_note


# ----------------------------------------------------------------------------
# Submit + queue views
# ----------------------------------------------------------------------------
def _queue_df(status_filter=None):
    rows = []
    try:
        for f in sorted(Path(_requests_dir()).glob("*.json")):
            try:
                req = json.loads(f.read_text())
            except Exception:
                continue
            if "model_id" not in req:
                continue
            if status_filter and req.get("status") != status_filter:
                continue
            rows.append({"Model ID": req.get("model_id"), "Tier": req.get("tier"),
                         "Status": req.get("status"), "Submitted": req.get("submitted_at", "")})
    except Exception:
        pass
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Model ID", "Tier", "Status", "Submitted"])


def get_pending():
    return _queue_df("pending")


def get_completed():
    return _queue_df("completed")


def validate_model_id(model_id: str) -> bool:
    return bool(model_id and MODEL_ID_RE.match(model_id.strip()))


def submit_eval(model_id: str, tier: str):
    model_id = (model_id or "").strip()
    if not validate_model_id(model_id):
        return "❌ Invalid model ID. Use litellm format `provider/model_name`, e.g. `openai/gpt-5.2`."
    benches = [b["id"] for b in BENCHMARKS]
    tier_code = "A+B" if tier.startswith("A+B") else "A"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    obj = {"model_id": model_id, "submitted_by": os.environ.get("USER", "anonymous"),
           "submitted_at": ts, "tier": tier_code, "status": "pending", "benchmarks": benches}
    fname = model_id.replace("/", "__") + "__" + ts.replace(":", "").replace("-", "") + ".json"
    try:
        _write_request(fname, obj)
    except Exception as e:
        return f"❌ Could not queue evaluation: {e}"
    return (f"✅ Queued **{model_id}** for Tier {tier_code} ({len(benches)} benchmarks). "
            "Results appear on the leaderboard within ~30 min of completion.")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
with gr.Blocks(title="Raidex") as app:
    gr.Markdown("# Raidex\n**An open Responsible AI index for frontier models** · "
                "[raidex.ai](https://raidex.ai)")

    with gr.Tabs() as tabs:
        # ---- Main page: leaderboard + the gap + coverage, all in one ----
        with gr.Tab("🏆 Leaderboard", id="leaderboard"):
            with gr.Row():
                search = gr.Textbox(placeholder="Search models...", show_label=False, scale=3)
                tier_filter = gr.CheckboxGroup(["Tier A", "Tier B", "Tier C"], value=["Tier A", "Tier B"],
                                               label="Benchmark tiers", scale=2)
            table = gr.Dataframe(value=_display(LEADERBOARD), interactive=False, wrap=True)
            refresh_btn = gr.Button("🔄 Refresh", scale=0)
            gr.Markdown(BADGE_LEGEND)
            search.change(filter_leaderboard, [search, tier_filter], table)
            tier_filter.change(filter_leaderboard, [search, tier_filter], table)
            refresh_btn.click(lambda: refresh(), None, table)

            gr.Markdown("## 🔥 The Gap")
            with gr.Row():
                gr.Plot(value=build_capability_heatmap())
                gr.Plot(value=build_rai_heatmap())
            gr.Markdown("*Frontier developers report capability benchmarks consistently. "
                        "RAI benchmarks? Rarely — Raidex runs them anyway.*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Coverage vs RAI Score")
                    gr.Plot(value=build_coverage_scatter())
                with gr.Column():
                    gr.Markdown("### Key Findings")
                    gr.Markdown(KEY_FINDINGS_MD)

            gr.Markdown("### Methodology")
            gr.Markdown("The RAI Score is a defined index — an unweighted mean of normalized open-benchmark "
                        "scores across safety, fairness, factuality, security, machine ethics, robustness, "
                        "and privacy. Scores are generative/judge-based and sampled; read them within Raidex, "
                        "not against canonical loglikelihood leaderboards.")
            method_btn = gr.Button("📖 Read the full methodology →", scale=0)

        with gr.Tab("🔍 Model Card", id="modelcard"):
            picker = gr.Dropdown(label="Select model", choices=model_choices())
            with gr.Row():
                with gr.Column(scale=1):
                    m_summary = gr.Markdown()
                with gr.Column(scale=2):
                    m_radar = gr.Plot()
            m_table = gr.Dataframe(interactive=False)
            m_cap = gr.Markdown()
            picker.change(model_card, picker, [m_summary, m_radar, m_table, m_cap])

        with gr.Tab("🕸️ Radar", id="radar"):
            r_select = gr.Dropdown(multiselect=True, label="Compare models", choices=model_choices())
            r_plot = gr.Plot(value=build_radar([]))
            r_select.change(build_radar, r_select, r_plot)

        with gr.Tab("🚀 Submit", id="submit"):
            gr.Markdown("### Evaluate a model on RAI benchmarks")
            gr.Markdown("Model ID uses litellm format: `provider/model_name` "
                        "(e.g. `openai/gpt-5.2`, `anthropic/claude-opus-4-8`, `gemini/gemini-2.5-flash`)")
            s_model = gr.Textbox(label="Model ID", placeholder="openai/gpt-5.2")
            s_tier = gr.Radio(["A (6 benchmarks)", "A+B (8 benchmarks)"],
                              value="A+B (8 benchmarks)", label="Evaluation tier")
            s_btn = gr.Button("Submit for evaluation", variant="primary")
            s_msg = gr.Markdown()
            s_btn.click(submit_eval, [s_model, s_tier], s_msg)
            gr.Markdown("---")
            with gr.Accordion("⏳ Pending evaluations", open=False):
                gr.Dataframe(value=get_pending(), interactive=False)
            with gr.Accordion("✅ Completed evaluations", open=False):
                gr.Dataframe(value=get_completed(), interactive=False)

        with gr.Tab("📖 Methodology", id="methodology"):
            gr.Markdown(METHODOLOGY_MD)

    with gr.Accordion("📙 Citation", open=False):
        gr.Textbox(value=CITATION_TEXT, lines=8, show_label=False)
    gr.Markdown("---")
    with gr.Row():
        gr.Markdown("[GitHub](https://github.com/cloudronin/raidex) · Built by Vishnu Vettrivel")
        footer_method_btn = gr.Button("📖 Methodology", scale=0)

    # Markdown links can't target Gradio tabs, so route the methodology links through tab selection.
    method_btn.click(lambda: gr.Tabs(selected="methodology"), None, tabs)
    footer_method_btn.click(lambda: gr.Tabs(selected="methodology"), None, tabs)
    app.load(refresh, None, table)


if __name__ == "__main__":
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh, "interval", seconds=1800)
    scheduler.start()
    app.queue(default_concurrency_limit=40).launch()
