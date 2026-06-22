---
title: Raidex
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
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

Live leaderboard: https://huggingface.co/spaces/cloudronin/raidex-space
Site: https://raidex.ai

## Key Findings

_2026-06 re-run, 17 frontier models (16 on all 8 benchmarks; MiniMax on 7; Mistral and Phi-4 excluded as un-evaluable). Independent automated evaluations, not self-reported._

- **Capability barely predicts responsibility.** Capability (Artificial Analysis Intelligence Index) explains only **~3% of the variation in RAI Score** (**Pearson r ≈ 0.17, n=17, not significant**; live value on the chart). Concretely, **Qwen3-235B (open, only mid-capability) is #2**; GPT-4o and Gemini (among the least capable) tie 3rd; the 2nd-most-capable GPT-5.5 lands mid-pack with the board's **second-worst hazardous-knowledge (WMDP)** score (behind Grok 4.3); capable MiniMax sits near the bottom. Opus 4.8 tops it (71.6), but high capability rarely tracks high responsibility.
- **A ~17-point board spanning a twelvefold capability range.** The top cluster (≈68 to 72) mixes the most and least capable models: Qwen (open, mid-cap) and GPT-4o (low-cap) sit alongside Opus.
- **Open weights are competitive.** 8 of 17 models are open-weight, and one (Qwen3-235B) is #2 overall, ahead of nearly every closed frontier system.
- **Capability does not equal responsibility within a lab.** GPT-4o (69.2) outscores the newer GPT-5.2 (64.2); GPT-5.5 leads OpenAI on capability yet carries the most hazardous knowledge in OpenAI's lineup.
- **Caveats:** the correlation is weak, non-significant, and was volatile as the board filled (r moved 0.13 to 0.29 to 0.17; bootstrap 95% CI [−0.40, 0.58]), so look at the full *scatter* rather than the point estimate; sampled (~150 to 300 items/task, so top-cluster ranks are ties); generative MCQ validated against loglikelihood (Methodology, Calibration); GPT-5.5's MCQs are sampled (temp=1); single neutral judge; the RAI Score is a defined index, not a safety certificate.

Full, live results: <https://huggingface.co/spaces/cloudronin/raidex-space>

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

Submit a model, the backend runs 6 RAI benchmarks automatically, and scores appear
on the leaderboard.

## Benchmarks (Tier A, automated)

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

- 🟣 **Full RAI Profile.** All 8 benchmarks (Tier A + B + C)
- 🔵 **Independently Evaluated.** ≥4 benchmarks run by our automated pipeline
- 🟡 **Self-Reported Only.** Scores from system cards / published leaderboards
- ⚪ **Partial.** Fewer than 4 benchmarks

## Composite Score

**RAI Score** = mean of normalized benchmark scores (0-100).
**RAI Coverage** = benchmarks evaluated / 8.

## Prior art

DecodingTrust, COMPL-AI, HELM Safety, MLCommons AILuminate, FLI AI Safety
Index, and DeepSight (2026). Raidex aggregates across independent open benchmarks with
a public submit pipeline, alongside these efforts.

## License

MIT
