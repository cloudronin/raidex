# raidex-backend

Eval runner for [Raidex](https://raidex.ai) — an open Responsible AI index for frontier
models. Polls the `raidex-requests` queue, runs Tier A benchmarks against any submitted
model via [litellm](https://docs.litellm.ai/), and writes results to the `raidex-results`
dataset that backs the HuggingFace Space.

## Tier A benchmarks

| ID | Benchmark | Dimension | Pipeline | Scoring |
|----|-----------|-----------|----------|---------|
| `bbq` | BBQ | Fairness & Bias | lm-eval-harness | generative¹ |
| `wmdp` | WMDP | Security | lm-eval-harness | generative¹ |
| `simpleqa` | SimpleQA | Factuality | litellm + judge | F1 |
| `strongreject` | StrongREJECT | Security (refusal) | strong_reject | rubric judge |
| `ethics` | ETHICS | Machine Ethics | lm-eval-harness | generative¹ |
| `xstest` | XSTest | Safety (over-refusal) | litellm + judge | balanced refusal acc. |

¹ BBQ/WMDP/ETHICS are natively loglikelihood/MCQ tasks. Frontier chat APIs do not expose
token logprobs, so Raidex scores them **generatively** (prompt for an answer, extract the
chosen option). Values are indicative of — not identical to — canonical loglikelihood
scores. See `METHODOLOGY.md` in the Space repo.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# litellm reads provider keys from env vars; judges (SimpleQA/XSTest/StrongREJECT-rubric)
# need OPENAI_API_KEY.
export OPENAI_API_KEY=...        # required for judges
export ANTHROPIC_API_KEY=...     # + the provider key for whichever model you evaluate
```

## Usage

```bash
# Cost estimate only — run before every real run.
python runner.py --dry-run --model openai/gpt-5.2 --tier A

# Smoke test on a small sample, write JSON locally without uploading.
python runner.py --model openai/qwen3.7-max --tier A --limit 50 --no-upload

# Full run against one model.
python runner.py --model anthropic/claude-opus-4-6 --tier A

# Poll the requests queue and run anything pending.
python runner.py --poll
```

Model IDs use litellm format `provider/model_name`. The `per_model_limit_usd: 100` cap in
`config.yaml` is a hard ceiling enforced before any run.

## Layout

```
runner.py            orchestrator + CLI
scoring.py           normalization, composite, dimensions, badges (single source of truth)
benchmarks/
  base.py            Benchmark ABC + BenchmarkResult
  _proxy.py          litellm proxy lifecycle + readiness poll (lm-eval benches)
  bbq.py wmdp.py ethics.py        lm-eval wrappers (generative)
  simpleqa.py strongreject.py xstest.py   direct litellm + judge wrappers
config.yaml          budget caps, judge models, lm-eval + dataset settings
```
