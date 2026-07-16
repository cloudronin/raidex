# Raidex

**An open Responsible AI index for frontier models.** — [raidex.ai](https://raidex.ai)

Raidex evaluates frontier LLMs across open RAI benchmarks — safety, fairness, factuality, security, machine ethics, robustness, and privacy — and publishes a composite **RAI Score** on a submit-driven leaderboard. Every benchmark is open-source and runnable; the project shows the capability-vs-RAI *reporting gap* side by side.

- **Live leaderboard:** https://huggingface.co/spaces/cloudronin/raidex
- **Results dataset:** https://huggingface.co/datasets/cloudronin/raidex-results
- **Eval queue:** https://huggingface.co/datasets/cloudronin/raidex-requests

## Measure your own model: the `raidex` CLI

`pip install raidex` gives you the same measurement core the leaderboard runs, as a self-contained CLI. Score your own fine-tuned or self-hosted model against the Raidex index **in your own environment** — no account, no queue, no upload, no dependency on Raidex servers — and get **board-comparable** per-dimension + composite RAI scores.

```bash
pip install raidex

raidex eval --model openai/gpt-5.2 --tier A                                      # any litellm model
raidex eval --model http://localhost:8000/v1 --served-name my-model --tier A+B  # a local OpenAI-compatible endpoint
raidex eval --model ... --benchmarks bbq,strongreject                           # a subset
raidex eval --model ... --judge anthropic/claude-opus-4-8                        # configure the LLM judge
raidex eval --model ... --dry-run                                               # cost estimate only
raidex fetch-data                                                               # pre-cache data for offline / air-gapped use
raidex eval --model ... --offline --output results.json                         # run with zero network
```

It prints per-dimension + composite RAI Score + coverage (N/9) and writes a self-describing JSON — model spec, pinned dataset versions, judge, sampling, timestamp — that never leaves your machine. Benchmarks with no judge configured are skipped with honestly reduced coverage, not a failure. Scores are identical in scale to the leaderboard (same core, same benchmarks, same normalization).

## Repository layout

- [`raidex/`](raidex/) — the pip-installable **`raidex` CLI** (`raidex/cli.py`) over the pure **`raidex.core`** eval-and-score library. The core is the shared foundation; the backend service and the CLI are two thin frontends over it, which is why a local score matches the board.
- [`space/`](space/) — the HuggingFace **Space** (Gradio app): leaderboard, the capability-vs-RAI gap visual, model cards, and the submit form. Deployed to the Space above.
- [`backend/`](backend/) — the **eval runner / service**: polls the request queue, calls `raidex.core`, and writes results to the dataset.

The two datasets (`raidex-results`, `raidex-requests`) are *generated data* and live on the HF Hub, not in this repo.

## Benchmarks (8 constituents)

| Tier | Benchmark | Dimension | Pipeline |
|------|-----------|-----------|----------|
| A | BBQ | Fairness & Bias | lm-eval (generative) |
| A | WMDP | Security | lm-eval (generative) |
| A | SimpleQA | Factuality | litellm + judge |
| A | StrongREJECT | Security (refusal) | strong_reject rubric |
| A | ETHICS | Machine Ethics | lm-eval (generative) |
| A | XSTest | Safety (over-refusal) | litellm + judge |
| B | AdvGLUE | Robustness | litellm (exact-match) |
| B | ConfAIde | Privacy | litellm (correlation) |

The **RAI Score** is the mean of normalized constituent scores (0–100); coverage is reported as N/8. See [`space/METHODOLOGY.md`](space/METHODOLOGY.md) for the index design, generative-task creation, judging, sampling, and normalization.

## Running the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# litellm reads provider keys from env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...);
# an OpenAI/Anthropic key is required for the judges.
python runner.py --dry-run --model openai/gpt-5.2 --tier A+B    # cost estimate
python runner.py --model anthropic/claude-opus-4-8 --tier A+B   # full run + upload
python runner.py --poll                                         # drain the request queue
```

**Reproducing the published board.** The leaderboard was produced by running the full 17-model roster at Tier A+B. `backend/rerun.py` runs the rate-limited batch with per-provider throttling and resume; the five always-greedy OpenAI/Anthropic models run via `runner.py`. See [`backend/README.md`](backend/README.md#reproducing-the-published-leaderboard) for the roster and provider routing, the exact keys, the reasoning-locked / WMDP-recovery / sampling / neutral-judge settings, the 2026-06-18 capability snapshot, and the generative-vs-loglikelihood calibration. Numbers reproduce within the stated error bars (composite 95% half-width ~±2 points), not bit-for-bit.

## Running the Space locally

```bash
cd space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
RAIDEX_DATA_SOURCE=hf python app.py     # reads the live results dataset
```

## License

MIT — see [LICENSE](LICENSE).
