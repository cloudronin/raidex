_Provisional — from the 2026-06 re-run. 11 models scored on all 8 benchmarks; Qwen3-235B, Gemma-3-27B, and Gemini's Tier-B re-score are still landing, and Mistral Large is excluded (its API rate-limited every attempt). Every number is an independent automated evaluation, not a self-reported score._

### Capability and responsibility are largely decoupled

Capability (Artificial Analysis Intelligence Index) and the RAI Score correlate only **weakly — Pearson r ≈ 0.3** (the chart shows the exact live value, which shifts a little as the board fills in). The single most capable model, **Claude Opus 4.8, does top the RAI board (71.6)** — but several far-less-capable models sit right behind it. High responsibility is achievable across the capability range; it is not the preserve of the frontier. (This corrected an earlier *pipeline artifact* that had Opus scoring lowest — see the scoring fixes in the methodology change log.)

### A tight top cluster — closed and open weights together

| # | Model | RAI Score |
|---|-------|----------:|
| 1 | Claude Opus 4.8 | 71.6 |
| 2 | GPT-4o | 69.2 |
| 3 | Claude Sonnet 4.6 | 68.6 |
| 4 | **Llama 3.3 70B** *(open)* | 68.0 |
| 5 | **DeepSeek V3.2** *(open)* | 66.1 |
| 6 | GPT-5.2 | 64.2 |
| 7 | GPT-4o-mini | 62.6 |
| 8 | Claude Haiku 4.5 | 62.2 |
| 9 | Grok 4.3 | 61.3 |
| 10 | gpt-oss-120B *(open)* | 54.8 |

The top four are within sampling error of one another, and they mix closed and open weights. **Open-weight models are competitive on responsibility** — Llama 3.3 70B and DeepSeek V3.2 land inside the top cluster, ahead of several closed frontier models.

### Capability doesn't track responsibility within a lab

**GPT-4o (69.2) outscores the newer, more capable GPT-5.2 (64.2)** on the RAI composite — within one developer, a more advanced model rates *lower* on responsibility (largely a factuality and security/WMDP effect).

### The reporting gap this fills

Frontier developers report capability benchmarks almost universally but Responsible-AI benchmarks rarely (see **The Gap**). Raidex runs all 8 independently — none of these numbers are self-reported.

### Read this as a defined index, with error bars

- **Provisional:** Qwen, Gemma, and Gemini's Tier-B re-score are still landing; Mistral is excluded (un-evaluable on our API rate limit).
- **Sampled** (300 items/task): the composite's 95% half-width is ~±2 points, so **differences within the top cluster (~68–72) are ties.** The real signal is the **~17-point spread** from the top (~72) to the bottom (~55), not the exact order of neighbours.
- **Generative scoring is validated:** BBQ/WMDP scored generatively track the canonical loglikelihood method within ~3–6 points, and the gap shrinks as models get more capable — so the ordering is *not* a scoring artifact. See Methodology → Calibration.
- The RAI Score is an **unweighted, defined index** across 7 dimensions — built for relative comparison, not an absolute safety certificate. WMDP (security) penalizes hazardous knowledge, so a very knowledgeable model can score lower there.
