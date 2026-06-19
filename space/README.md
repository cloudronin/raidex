---
title: Raidex
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
short_description: An open Responsible AI index for frontier models
tags:
  - leaderboard
  - responsible-ai
  - ai-safety
  - benchmarks
  - frontier-models
---

# Raidex

An open Responsible AI index for frontier foundation models. Benchmarks across
safety, fairness, factuality, security, robustness, privacy, and ethics.

Live leaderboard: https://huggingface.co/spaces/cloudronin/raidex
Site: https://raidex.ai

## Key Findings

_Provisional — 2026-06 re-run, 11 models on all 8 benchmarks (Qwen / Gemma / Gemini Tier-B still landing; Mistral excluded — rate-limited). Independent automated evaluations, not self-reported._

- **Capability and responsibility are largely decoupled** — capability (Artificial Analysis Intelligence Index) vs RAI Score correlate only weakly (**Pearson r ≈ 0.3**, exact live value on the chart). The most capable model, Claude Opus 4.8, tops the RAI board (71.6), but much-less-capable models match it.
- **Tight top cluster, closed + open:** Opus 4.8 (71.6), GPT-4o (69.2), Sonnet 4.6 (68.6) and **Llama 3.3 70B (68.0, open)** are within sampling error of one another; DeepSeek V3.2 (66.1, open) is just behind. Open weights are competitive on responsibility.
- **Capability ≠ responsibility within a lab:** GPT-4o (69.2) outscores the newer GPT-5.2 (64.2).
- **Caveats:** sampled (composite ±~2 pts → top-cluster ranks are ties; the real signal is the ~17-pt top-to-bottom spread); generative MCQ scoring validated against loglikelihood (see Methodology → Calibration); single neutral judge; the RAI Score is a defined index, not a safety certificate.

Full, live results: <https://huggingface.co/spaces/cloudronin/raidex>

_The findings are generated from independent automated evaluations, not
self-reported scores from model developers._

## Why this exists

The 2026 Stanford AI Index documents a reporting gap: frontier models report
capability benchmarks consistently, but RAI benchmark reporting is sparse. Every
benchmark in this set is open-source and runnable.

This is not the first composite RAI evaluation. HELM Safety, COMPL-AI, MLCommons
AILuminate, and the FLI AI Safety Index publish composite scores. Raidex adds an
open, submit-driven leaderboard that aggregates these specific open benchmarks and
shows the capability-vs-RAI reporting gap side by side.

## How it works

Submit a model → the backend runs 6 RAI benchmarks automatically → scores appear
on the leaderboard.

## Benchmarks (Tier A — automated)

| Benchmark | Dimension | Pipeline | Cost/Model |
|-----------|-----------|----------|------------|
| BBQ | Fairness & Bias | lm-eval-harness (generative) | ~$10 |
| WMDP | Security | lm-eval-harness (generative) | ~$8 |
| SimpleQA | Factuality | litellm + judge (F1) | ~$30 |
| StrongREJECT | Security (refusal) | strong_reject rubric | ~$5 |
| ETHICS | Machine Ethics | lm-eval-harness (generative) | ~$5 |
| XSTest | Safety (over-refusal) | litellm + judge | ~$4 |

**Order-of-magnitude: ~$62/model. Confirm with `--dry-run`.** BBQ/WMDP/ETHICS are
scored generatively (chat APIs don't expose logprobs); see METHODOLOGY.md.

## Badges

- 🟣 **Full RAI Profile** — all 8 benchmarks (Tier A + B + C)
- 🔵 **Independently Evaluated** — ≥4 benchmarks run by our automated pipeline
- 🟡 **Self-Reported Only** — scores from system cards / published leaderboards
- ⚪ **Partial** — fewer than 4 benchmarks

## Composite Score

**RAI Score** = mean of normalized benchmark scores (0-100).
**RAI Coverage** = benchmarks evaluated / 8.

## Prior art

DecodingTrust · COMPL-AI · HELM Safety · MLCommons AILuminate · FLI AI Safety
Index · DeepSight (2026). Raidex aggregates across independent open benchmarks with
a public submit pipeline, alongside these efforts.

## License

MIT
