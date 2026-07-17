# Operating the Raidex leaderboard & Space

This guide is for **maintainers** running the public Raidex board — the eval service, the
Hugging Face Space, reproduction, and deploy. If you just want to **measure a model**, you
don't need any of this: `pip install raidex` and see the [README](../README.md).

The board has three moving parts (all in this monorepo):

- [`backend/`](../backend/) — the eval **service**: drains the submit queue, calls `raidex.core`, and uploads results to the dataset.
- [`space/`](../space/) — the Hugging Face **Space** (Gradio app): leaderboard, capability-vs-RAI gap visual, model cards, submit form.
- The two Hub datasets — `cloudronin/raidex-results` (scores) and `cloudronin/raidex-requests` (queue) — are *generated data*, not in the repo.

## Running the backend service

```bash
pip install -e .          # from the repo root — installs the raidex package the service imports
cd backend
# litellm reads provider keys from env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...);
# an OpenAI/Anthropic key is required for the judges. HF_TOKEN (write) is needed to upload.
python runner.py --dry-run --model openai/gpt-5.2 --tier A+B    # cost estimate
python runner.py --model anthropic/claude-opus-4-8 --tier A+B  # full run + upload
python runner.py --poll                                        # drain the submit queue
```

`--poll` is bounded for unattended/public runs by `RAIDEX_POLL_MAX` (models per run),
`RAIDEX_POLL_LIMIT` (per-benchmark sample), and `RAIDEX_PER_MODEL_USD` (hard per-model cost
cap). The GitHub Action [`.github/workflows/poll-queue.yml`](../.github/workflows/poll-queue.yml)
runs it on a schedule; it requires the provider keys + `HF_TOKEN` as repo secrets.

## Reproducing the published board

The leaderboard was produced by running the roster at Tier A+B. `backend/rerun.py` runs the
rate-limited batch with per-provider throttling + resume; the always-greedy OpenAI/Anthropic
models run via `runner.py`. See [`backend/README.md`](../backend/README.md#reproducing-the-published-leaderboard)
for the roster and provider routing, the exact keys, the reasoning-locked / WMDP-recovery /
sampling / neutral-judge settings, the capability snapshot, and the generative-vs-loglikelihood
calibration. Numbers reproduce within the stated error bars (composite 95% half-width ~±2
points), not bit-for-bit.

## Running the Space locally

```bash
cd space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
RAIDEX_DATA_SOURCE=hf python app.py     # reads the live results dataset
```

## Deploy

The Space auto-deploys from `main`: a push touching `space/**` triggers
[`.github/workflows/deploy-space.yml`](../.github/workflows/deploy-space.yml), which runs the
data-integrity gate (`space/check_integrity.py`) and then syncs `space/` to the
`cloudronin/raidex-space` Space. Requires the `HF_TOKEN` repo secret.

## See also

- [`space/METHODOLOGY.md`](../space/METHODOLOGY.md) — index design, generative-task creation, judging, sampling, normalization, and the shared-lab / sycophancy disclosures.
- [`backend/README.md`](../backend/README.md) — service internals + full reproduction detail.
