_Provisional — from the 2026-06 re-run. 11 models scored on all 8 benchmarks; Qwen3-235B and Gemma-3-27B are still landing, and Mistral Large is excluded (its API rate-limited every attempt). Every number is an independent automated evaluation, not a self-reported score._

### Capability and responsibility are essentially decoupled

Capability (Artificial Analysis Intelligence Index) and the RAI Score barely correlate — **Pearson r ≈ 0.1**, essentially flat (the chart shows the exact live value, which shifts as the board fills in). The single most capable model, **Claude Opus 4.8, does top the RAI board (71.6)** — but a cluster of *far*-less-capable models (Gemini 2.5 Flash, GPT-4o, Llama 3.3 70B) sit right behind it at near-identical RAI. **High responsibility is achievable across the entire capability range; it is not the preserve of the frontier.** (An earlier *pipeline artifact* had Opus scoring lowest — corrected; see the scoring fixes in the methodology change log.)

### A tight top cluster — closed and open weights together

| # | Model | RAI Score |
|---|-------|----------:|
| 1 | Claude Opus 4.8 | 71.6 |
| 2 | Gemini 2.5 Flash | 69.2 |
| 2 | GPT-4o | 69.2 |
| 4 | Claude Sonnet 4.6 | 68.6 |
| 5 | **Llama 3.3 70B** *(open)* | 68.0 |
| 6 | **DeepSeek V3.2** *(open)* | 66.1 |
| 7 | GPT-5.2 | 64.2 |
| 8 | GPT-4o-mini | 62.6 |
| 9 | Claude Haiku 4.5 | 62.2 |
| 10 | Grok 4.3 | 61.3 |
| 11 | gpt-oss-120B *(open)* | 54.8 |

The **top five — Opus 4.8, Gemini 2.5 Flash, GPT-4o, Sonnet 4.6, and Llama 3.3 70B — are within sampling error of one another** (a ~3.6-point span), and they mix closed and open weights, frontier and mid-tier. **Open-weight models are competitive on responsibility:** Llama 3.3 70B sits inside the top cluster and DeepSeek V3.2 just behind, ahead of several closed frontier models.

### Capability doesn't track responsibility within a lab

**GPT-4o (69.2) outscores the newer, more capable GPT-5.2 (64.2)** on the RAI composite — within one developer, a more advanced model rates *lower* on responsibility (largely a factuality and security/WMDP effect).

### The reporting gap this fills

Frontier developers report capability benchmarks almost universally but Responsible-AI benchmarks rarely (see **The Gap**). Raidex runs all 8 independently — none of these numbers are self-reported.

### Read this as a defined index, with error bars

- **Provisional:** Qwen and Gemma are still landing; Mistral is excluded (un-evaluable on our API rate limit).
- **Sampled** (300 items/task): the composite's 95% half-width is ~±2 points, so **differences within the top cluster (~68–72) are ties.** The real signal is the **~17-point spread** from the top (~72) to the bottom (~55), not the exact order of neighbours.
- **Generative scoring is validated:** BBQ/WMDP scored generatively track the canonical loglikelihood method within ~3–6 points, and the gap shrinks as models get more capable — so the ordering is *not* a scoring artifact. See Methodology → Calibration.
- The RAI Score is an **unweighted, defined index** across 7 dimensions — built for relative comparison, not an absolute safety certificate. WMDP (security) penalizes hazardous knowledge, so a very knowledgeable model can score lower there.
