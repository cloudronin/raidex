_2026-06 re-run — 17 frontier models scored on all 8 benchmarks. Mistral Large and Phi-4 are excluded (un-evaluable on our endpoints). Every number is an independent automated evaluation, not a self-reported score._

### Capability barely predicts responsibility

Across the 17 models, capability (Artificial Analysis Intelligence Index) explains only **~3% of the variation in RAI Score** — **Pearson r ≈ 0.17 (n=17), not statistically significant** (95% CI spans zero; the chart shows the live value). The board makes the point concretely:

- **Qwen3-235B — open-weight and only mid-capability — is #2** on responsibility, above every closed frontier model except Opus.
- **GPT-4o and Gemini 2.5 Flash**, among the *least* capable models here, tie for 3rd.
- The 2nd-most-capable model, **GPT-5.5**, lands mid-pack and posts the board's **worst hazardous-knowledge (WMDP)** score.
- **MiniMax-M2.7** (capable) sits near the bottom.

Claude Opus 4.8 does top the board (71.6) — so the frontier *can* lead — but it is the exception, not the rule. **High responsibility is achievable at every capability level, and being more capable is no guarantee of it.** (An earlier pipeline artifact had Opus scoring lowest — corrected; see the methodology change log.)

### The board — closed and open, every capability tier

| # | Model | RAI | |
|---|-------|----:|---|
| 1 | Claude Opus 4.8 | 71.6 | |
| 2 | **Qwen3-235B** | 69.6 | open |
| 3 | GPT-4o | 69.2 | |
| 3 | Gemini 2.5 Flash | 69.2 | |
| 5 | GPT-5.5 † | 69.0 | |
| 6 | Claude Sonnet 4.6 | 68.6 | |
| 7 | **Llama 3.3 70B** | 68.0 | open |
| 8 | **DeepSeek V3.2** | 66.1 | open |
| 9 | **DeepSeek V3.1** | 64.4 | open |
| 10 | GPT-5.2 | 64.2 | |
| 11 | **Gemma-4 31B** | 63.6 | open |
| 12 | GPT-4o-mini | 62.6 | |
| 13 | **Gemma-3 27B** | 62.4 | open |
| 14 | Claude Haiku 4.5 | 62.2 | |
| 15 | Grok 4.3 | 61.3 | |
| 16 | **MiniMax-M2.7** | 58.5 | open |
| 17 | **gpt-oss-120B** | 54.8 | open |

† GPT-5.5 is reasoning-locked — its MCQ benchmarks run at temperature 1 (sampled), so treat its score as approximate. See Methodology → Reasoning-locked models.

**The whole board spans just ~17 points (54.8–71.6) while capability spans 5×.** The top cluster (≈68–72) mixes the most and least capable models — Qwen (open, mid-cap) and GPT-4o (low-cap) sit right alongside Opus (frontier).

### Open weights are competitive on responsibility

**8 of the 17 models are open-weight — and one (Qwen3-235B) is #2 overall.** Open models appear at every level of the board, ahead of many closed frontier systems. Responsibility is not a closed-model advantage.

### Capability doesn't track responsibility within a lab either

**GPT-4o (69.2) outscores the newer, more capable GPT-5.2 (64.2)**, and GPT-5.5 — OpenAI's most capable — carries the most hazardous knowledge of any model here. Within a single developer, more advanced ≠ more responsible.

### The reporting gap this fills

Frontier developers report capability benchmarks almost universally but Responsible-AI benchmarks rarely (see **The Gap**). Raidex runs all 8 independently — none of these numbers are self-reported.

### Read this as a defined index, with error bars

- **The correlation is weak, non-significant, and was volatile as the board filled** — r moved 0.13 → 0.29 → 0.17 as models landed (bootstrap 95% CI [−0.40, 0.58]; the sign isn't even reliably positive — P(r>0) ≈ 76%). The **scatter is the finding, not the point estimate**: capability is essentially uninformative about where a model lands on RAI.
- **Sampled** (≈150–300 items/task): the composite's 95% half-width is ~±2 points, so differences inside the top cluster are ties. The real signal is the **~17-point top-to-bottom spread**, not the order of neighbours.
- **Generative MCQ scoring is validated** against the canonical loglikelihood method (within ~3–6 points; see Methodology → Calibration).
- **Reasoning-locked models** (GPT-5.5) are scored at temperature 1; **Phi-4 and Mistral** are excluded (un-evaluable on our endpoints).
- The RAI Score is an **unweighted, defined index** across 7 dimensions — built for relative comparison, not an absolute safety certificate. WMDP (security) penalizes hazardous knowledge, so a very knowledgeable model scores lower there.
