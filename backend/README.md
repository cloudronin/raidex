# raidex-backend

Eval runner for [Raidex](https://raidex.ai), an open Responsible AI index for frontier
models. Polls the `raidex-requests` queue, runs the 8 benchmarks against any submitted
model via [litellm](https://docs.litellm.ai/), scores them, and writes results to the
`raidex-results` dataset that backs the HuggingFace Space.

## Benchmarks (8 constituents)

| Tier | ID | Benchmark | Dimension | Pipeline | Scoring |
|------|----|-----------|-----------|----------|---------|
| A | `bbq` | BBQ | Fairness & Bias | lm-eval-harness | generative¹ |
| A | `wmdp` | WMDP | Security | lm-eval-harness | generative¹ |
| A | `simpleqa` | SimpleQA | Factuality | litellm + judge | F1 |
| A | `strongreject` | StrongREJECT | Security (refusal) | strong_reject | rubric judge |
| A | `ethics` | ETHICS | Machine Ethics | lm-eval-harness | generative¹ |
| A | `xstest` | XSTest | Safety (over-refusal) | litellm + judge | balanced refusal acc. |
| B | `advglue` | AdvGLUE | Robustness | litellm | exact-match |
| B | `confaide` | ConfAIde | Privacy | litellm | correlation |

¹ BBQ/WMDP/ETHICS are natively loglikelihood/MCQ tasks. Frontier chat APIs do not expose
token logprobs, so Raidex scores them **generatively** (prompt for an answer, extract the
chosen option) using the custom lm-eval task configs in `tasks/wmdp_gen` and `tasks/ethics_gen`.
Values are indicative of, not identical to, canonical loglikelihood scores; `validate_scoring.py`
quantifies the gap (see [Reproducing the published leaderboard](#reproducing-the-published-leaderboard)).
Full index design in [`../space/METHODOLOGY.md`](../space/METHODOLOGY.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

litellm reads provider API keys from environment variables. Set the keys for whichever
models you evaluate, plus the judge key:

```bash
export OPENAI_API_KEY=...        # OpenAI models + SimpleQA/StrongREJECT rubric judges
export ANTHROPIC_API_KEY=...     # Claude models + the neutral judge (default)
export GEMINI_API_KEY=...        # Gemini
export XAI_API_KEY=...           # Grok
export SAMBANOVA_API_KEY=...     # SambaNova-served open models (Llama, DeepSeek, gpt-oss, Gemma-4, MiniMax)
export HUGGINGFACE_API_KEY=...   # HF-Inference-served open models (Qwen, Gemma-3) via litellm `huggingface/`
export HF_TOKEN=...              # HF *write* token, uploads results to cloudronin/raidex-results
# optional: RAIDEX_JUDGE_MODEL (default anthropic/claude-sonnet-4-6, the neutral off-comparison judge)
```

## Usage

```bash
# Cost estimate only. Run before every real run. The per_model_limit_usd: 100 cap in
# config.yaml is a hard ceiling enforced before any run.
python runner.py --dry-run --model openai/gpt-5.2 --tier A+B

# Smoke test on a small sample, write JSON locally without uploading.
python runner.py --model openai/gpt-4o-mini --tier A+B --limit 50 --no-upload

# Full run against one model (scores + uploads to the results dataset).
python runner.py --model anthropic/claude-opus-4-8 --tier A+B

# Poll the requests queue and run anything pending.
python runner.py --poll

# Re-run any (model, benchmark) failures parked in the dead-letter queue.
python runner.py --replay-dlq
```

Model IDs use litellm format `provider/model_name`.

## Reproducing the published leaderboard

The board at raidex.ai was produced by running this 17-model roster at **Tier A+B**.

**Roster and provider routing.** Provider choice is part of the reproduction (the same
model can behave differently per host). The rate-limited batch and its routing live in the
`FULL` list in [`rerun.py`](rerun.py): `anthropic/claude-opus-4-8`, `openai/gpt-5.5`,
`xai/grok-4.3`, `sambanova/{DeepSeek-V3.2, DeepSeek-V3.1, Meta-Llama-3.3-70B-Instruct,
gpt-oss-120b, gemma-4-31B-it, MiniMax-M2.7}`, `huggingface/{Qwen/Qwen3-235B-A22B-Instruct-2507,
google/gemma-3-27b-it}`, and `gemini/gemini-2.5-flash`. The five always-greedy
OpenAI/Anthropic models run cleanly at default concurrency via `runner.py`:
`openai/gpt-5.2`, `openai/gpt-4o`, `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-6`,
`anthropic/claude-haiku-4-5-20251001`.

**Run it.** Either per model (`python runner.py --model <id> --tier A+B`) or, for the
throttled batch, `python rerun.py`, which applies per-provider concurrency, is idempotent
(skips any model already 8/8-clean on disk, so you can restart after an interruption), and
ends with a dead-letter-queue replay. Tune the sample with `RAIDEX_LIMIT` (default 300) and
concurrency with `RAIDEX_CONCURRENCY`.

**Settings that determine the numbers** (all committed in the config/code):

- **Sampling.** The four large benchmarks run on a fixed 300-prompt sample (150 for
  rate-limited providers); the small ones run in full. lm-eval's `--limit` takes the first
  N items deterministically. Each result records its `n_samples`.
- **Neutral judge.** SimpleQA / XSTest / StrongREJECT are LLM-judged by
  `anthropic/claude-sonnet-4-6`, held off the head-to-head comparison, so residual
  self-preference applies only to the three Anthropic rows. Override with `RAIDEX_JUDGE_MODEL`.
- **Reasoning-locked models** (e.g. GPT-5.5). Their API rejects `temperature=0`, so their
  MCQ benchmarks run at `temperature=1` (sampled) with a 2048-token floor, handled
  automatically by `REJECTS_TEMP0` in `benchmarks/_direct.py` and `benchmarks/_lmeval.py`.
  Treat those scores as approximate (within a few points).
- **WMDP recovery.** lm-eval's async client crashes on the ~6% of WMDP-bio items a safety
  filter blocks; the robust direct-call path (litellm + `num_retries` backoff in
  `benchmarks/_direct.py`) recovers the score rather than dropping the benchmark.
- **Exclusions.** Phi-4 and Mistral Large are un-evaluable on our endpoints (HTTP errors /
  rate limits even at concurrency 1), so they are excluded, not for anything about their behavior.
- **Generative-vs-loglikelihood calibration.** `python validate_scoring.py` re-scores
  BBQ/WMDP/ETHICS both ways (canonical loglikelihood vs Raidex's generative extraction) on a
  local open-weight model through lm-eval's `hf` backend. The two agree within ~3-6 points,
  shrinking as capability rises, so generative scoring is not the source of the ranking.

**Capability axis.** The scatter's x-axis is the Artificial Analysis Intelligence Index,
non-reasoning snapshot pulled **2026-06-18**, stored in
[`../space/data/capability_scores.json`](../space/data/capability_scores.json). It is a
fixed dated reference, not a live pull, so the scatter is reproducible.

**Determinism.** Numbers reproduce **within the stated error bars** (composite 95%
half-width ~±2 points), not bit-for-bit: fixed-sample evaluation, LLM-judge variability, and
generative extraction each add run-to-run variance. The ~17-point top-to-bottom spread is
the stable signal; differences inside the top cluster are ties.

**Outputs.** Each run writes `<provider>__<model>.json` locally and (unless `--no-upload`)
to the `cloudronin/raidex-results` dataset. The Space renders that dataset when launched
with `RAIDEX_DATA_SOURCE=hf`.

## Layout

```
runner.py            orchestrator + CLI (run_eval, --poll, --replay-dlq, result merge/finalize)
rerun.py             full-roster reproduction of the published board
scoring.py           normalization, composite, dimensions, badges (single source of truth)
validate_scoring.py  generative-vs-loglikelihood calibration
cost.py dlq.py finalize.py    cost guard, dead-letter queue, composite finalize
config.yaml          budget caps, judge models, lm-eval + dataset settings
benchmarks/
  base.py            Benchmark ABC + BenchmarkResult
  _direct.py         litellm direct-call path (retry/backoff; reasoning-lock handling)
  _lmeval.py _proxy.py   lm-eval-harness driver + litellm proxy lifecycle
  bbq.py wmdp.py ethics.py            lm-eval wrappers (generative)
  simpleqa.py strongreject.py xstest.py advglue.py confaide.py   direct litellm + judge
tasks/
  wmdp_gen/ ethics_gen/   custom lm-eval generative task configs (the generative scoring)
```
