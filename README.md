# Raidex

**Measure any LLM's Responsible-AI profile — your own fine-tuned or self-hosted model, or a frontier model — in your own environment.** — [raidex.ai](https://raidex.ai)

`raidex` scores a model across open Responsible-AI benchmarks (safety, fairness, factuality, security, machine ethics, robustness, privacy, and sycophancy) and reports a composite **RAI Score** plus per-dimension scores. The same measurement core also powers a public [leaderboard of frontier models](#the-public-leaderboard).

## Quickstart

```bash
pip install raidex        # Python 3.10–3.13
```

**Measure your own / self-hosted model** (any OpenAI-compatible endpoint — vLLM, Ollama, TGI, …):

```bash
raidex eval --model http://localhost:8000/v1 --served-name my-model --tier A+B
```

**Measure a frontier model** (any [litellm](https://docs.litellm.ai/docs/providers) model string; the provider key is read from the matching env var, e.g. `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=sk-...          # + ANTHROPIC_API_KEY for the judges
raidex eval --model openai/gpt-5.2 --tier A+B
```

That's it — no account, no queue, no upload, no dependency on Raidex servers. `raidex` prints per-dimension + composite RAI scores and writes a self-describing JSON that never leaves your machine.

## What you get

```
=== Raidex RAI Score ===
  RAI Score : 63.5
  Coverage  : 9/9  🟣
  Dimensions:
    safety           71.2
    fairness_bias    35.3
    factuality       52.0
    ...
```

- **Board-comparable** — identical in scale to the public leaderboard (same core, same benchmarks, same normalization), so you can place your model against the frontier.
- **A self-describing result JSON** — the model spec, per-benchmark pinned dataset versions, judge, sampling settings, and a timestamp, so a score is reproducible and traceable. Written locally; nothing is uploaded.
- **Honest coverage** — the composite is the mean of the constituents you ran, reported as N/9. Benchmarks you skip (or that need a judge you didn't configure) simply lower coverage; they never fake a number.

## CLI reference

```bash
raidex eval --model ...                         # required: a litellm model id OR a local endpoint URL
            --served-name my-model              # the model name served by a local endpoint URL
            --tier A | B | A+B                  # A = 7 core benchmarks, B = +robustness/privacy (default A)
            --benchmarks bbq,strongreject       # explicit subset (overrides --tier)
            --limit 150                         # sample the big benchmarks (small ones always run full)
            --judge anthropic/claude-opus-4-8   # LLM judge for SimpleQA / XSTest / StrongREJECT
            --dry-run                           # print a cost estimate and exit
            --offline                           # use only cached data; never touch the network
            --output results.json               # where to write the result (default: <model>__.json)

raidex fetch-data                               # pre-download + cache all benchmark data (for offline / air-gapped use)
```

**Judges.** SimpleQA, XSTest, and StrongREJECT are graded by an LLM judge. Configure one with `--judge` (or `RAIDEX_JUDGE_MODEL`); if none is available, those three are skipped with a printed reason and honestly reduced coverage — not a failure.

**Offline / air-gapped.** `raidex fetch-data` on a networked machine populates a local cache (pinned dataset versions); copy that cache across and run with `--offline` for zero network access.

## The benchmarks

`raidex` runs **9 benchmarks across 8 dimensions**. The **RAI Score** is the mean of normalized constituent scores (0–100); coverage is reported as N/9.

| Tier | Benchmark | Dimension | Pipeline |
|------|-----------|-----------|----------|
| A | BBQ | Fairness & Bias | lm-eval (generative) |
| A | WMDP | Security | lm-eval (generative) |
| A | SimpleQA | Factuality | litellm + judge |
| A | StrongREJECT | Security (refusal) | litellm + rubric judge |
| A | ETHICS | Machine Ethics | lm-eval (generative) |
| A | XSTest | Safety (over-refusal) | litellm + judge |
| A | Sycophancy | Sycophancy | litellm (judge-free flip-rate) |
| B | AdvGLUE | Robustness | litellm (exact-match) |
| B | ConfAIde | Privacy | litellm (correlation) |

See [`space/METHODOLOGY.md`](space/METHODOLOGY.md) for the index design, generative-task creation, judging, sampling, normalization, and disclosures.

## The public leaderboard

There's also a public board of frontier models, produced by the same core:

- **Live leaderboard:** https://huggingface.co/spaces/cloudronin/raidex
- **Results dataset:** https://huggingface.co/datasets/cloudronin/raidex-results
- **Eval queue:** https://huggingface.co/datasets/cloudronin/raidex-requests

Running the board, the Space, or reproducing the published numbers is a maintainer task — see [`docs/leaderboard.md`](docs/leaderboard.md).

## Repository layout

- [`raidex/`](raidex/) — the pip-installable **`raidex` CLI** (`raidex/cli.py`) over the pure **`raidex.core`** eval-and-score library. The core is the shared foundation; the CLI and the backend service are two thin frontends over it, which is why a local score matches the board.
- [`space/`](space/) — the Hugging Face **Space** (Gradio leaderboard app). See [`docs/leaderboard.md`](docs/leaderboard.md).
- [`backend/`](backend/) — the **eval service** that produces the public board. See [`docs/leaderboard.md`](docs/leaderboard.md).

## License

MIT — see [LICENSE](LICENSE).
