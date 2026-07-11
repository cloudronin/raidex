_2026-07 expansion. 37 frontier and open-weight models scored on 9 benchmarks. Sycophancy resistance was added this round, so every RAI Score is recomputed over 9 dimensions and is not directly comparable to the earlier 8-benchmark numbers. Qwen2.5 Coder 32B is 8/9 (one privacy benchmark would not complete). Every number is an independent automated evaluation, not a self-reported score._

### Capability is not a reliable predictor of responsibility

Across all 37 models, capability (Artificial Analysis Intelligence Index v4.1) and RAI Score are essentially uncorrelated. Pearson **r = 0.13, bootstrap 95% CI [−0.24, +0.45], n = 37**. The interval runs from a weak negative to a moderate positive and comfortably includes zero, so the point estimate carries no weight on its own: at this sample the data are consistent with no relationship. When the board was smaller and top-heavy (23 mostly-frontier models) r was 0.35 with a CI of [−0.13, +0.65]; adding a capability-diverse open-weight tier that spans the Intelligence Index from about 5 to 51 pulled the point estimate down to 0.13. The relationship did not just stay weak, it weakened as the capability range widened. **The scatter is the finding, not the coefficient**, and the scatter is a flat cloud.

Both corners are populated, and the low-capability edge is what flattens the line:

- **GPT-4o** (AA 11) is **#6** at 70.2, above every more-capable model ranked beneath it (the capability leaders GLM-5.2, DeepSeek V4 Pro, and Gemini 3.5 Flash all sit lower).
- **Llama 3.3 70B** (AA 14) is **#8** and **Llama 4 Scout** (AA 10) is **#16**: low capability, high responsibility.
- The **bottom of the board** (GLM-4.6, GLM-5, Qwen3.6-27B, RAI 45 to 50) is *mid*-capability open models (AA 37 to 39), not the least capable ones.

Across all the board fills to date the point estimate has moved between roughly 0.13 and 0.35, but the bootstrap interval has included zero every time. The robust reading is not "capability predicts responsibility" and not a proven decoupling either; it is that **no reliable relationship is detectable at this sample size**, and it is getting weaker, not stronger, as the board diversifies.

### Responsibility varies widely at the same capability

We note this carefully rather than claim it. Among the open-weight models at or below AA 25 (n = 13), RAI ranges from **49.8 to 70.0**, a spread of about 20 points at effectively the same low capability. At AA ≈ 37 to 40, RAI runs from 45.0 (Qwen3.6-27B) to 67.5 (DeepSeek V4 Flash). If capability set the ceiling on responsibility, models of equal capability would cluster; instead they scatter across most of the board's range. That pattern is consistent with responsibility being a development-priority choice rather than a byproduct of capability, but a single 2026-07 snapshot cannot establish causation, so we report the spread and stop there.

### Fable 5 at #1: a note on the judge

Fable 5's 81.0 comes with a caveat we state plainly. Fable 5 is an Anthropic model, and Raidex's fixed judge for the LLM-judged constituents (SimpleQA, XSTest) is also an Anthropic model, Claude Sonnet 4.6, so the board's #1 and its judge share a lab. We checked where Fable 5's lead concentrates against the 36-model average:

- Its single largest advantage is **factuality (SimpleQA), +53 points, and that constituent is sibling-judged**. SimpleQA grades answers against gold, so it is less exposed to stylistic self-preference than a subjective safety call, but it is the one to watch.
- Most of the rest is earned on constituents the judge never touches: **WMDP (+32) and ETHICS (+29) are fully deterministic, and the new sycophancy dimension (+26) is judge-free** (a deterministic flip-rate under pressure). Those three judge-untouched margins together are far larger than the sibling-judged factuality margin, so the rank is not a judge artifact.
- On the actually sibling-judged safety benchmark (XSTest) and OpenAI-judged StrongREJECT, Fable 5 is only about +1 above average, because those are near the ceiling for everyone.

Net: the rank is defensible on deterministic and judge-free dimensions, but the shared-lab situation and the large sibling-judged factuality margin are disclosed as a limitation (see Methodology, LLM-judge bias).

### The board, closed and open, every capability tier

| # | Model | RAI | |
|---|-------|----:|---|
| 1 | Claude Fable 5 † | 81.0 | |
| 2 | Claude Opus 4.8 | 74.0 | |
| 3 | GPT-5.5 † | 72.3 | |
| 4 | Claude Sonnet 4.6 | 71.5 | |
| 5 | Claude Sonnet 5 † | 70.2 | |
| 6 | GPT-4o | 70.2 | |
| 7 | **Qwen3-235B** | 70.0 | open |
| 8 | **Llama 3.3 70B** | 69.6 | open |
| 9 | **Llama 4 Maverick** | 68.8 | open |
| 10 | Gemini 3.5 Flash † | 67.9 | |
| 11 | **DeepSeek V3.2** | 67.6 | open |
| 12 | **DeepSeek V4 Flash** | 67.5 | open |
| 13 | **DeepSeek V3.1** | 67.0 | open |
| 14 | **Nemotron 3 Ultra** | 66.0 | open |
| 15 | Claude Haiku 4.5 | 65.8 | |
| 16 | **Llama 4 Scout** | 65.6 | open |
| 17 | Grok 4.3 | 65.5 | |
| 18 | **Gemma 3 27B** | 64.9 | open |
| 19 | **Gemma 4 31B** | 64.9 | open |
| 20 | **DeepSeek V3-0324** | 64.8 | open |
| 21 | **DeepSeek V4 Pro** | 64.5 | open |
| 22 | Gemini 2.5 Flash | 63.7 | |
| 23 | **Mistral Small 4** | 63.5 | open |
| 24 | GPT-4o-mini | 63.3 | |
| 25 | **MiniMax M2.7** | 60.2 | open |
| 26 | GPT-5.2 | 59.5 | |
| 27 | **gpt-oss-120B** | 58.9 | open |
| 28 | **GLM-5.2** | 58.4 | open |
| 29 | **MiMo V2.5 Pro** | 58.3 | open |
| 30 | **Qwen2.5 Coder 32B** ‡ | 58.2 | open |
| 31 | **Qwen3.5-397B** | 58.0 | open |
| 32 | **Kimi K2.6** | 57.2 | open |
| 33 | **GLM-5.1** | 57.1 | open |
| 34 | **gpt-oss-20B** | 57.1 | open |
| 35 | **GLM-4.6** | 49.8 | open |
| 36 | **GLM-5** | 47.6 | open |
| 37 | **Qwen3.6-27B** | 45.0 | open |

† Reasoning-locked closed models (Fable 5, GPT-5.5, Sonnet 5, Gemini 3.5 Flash): their MCQ benchmarks run at temperature 1 or the model default (sampled), so treat those scores as approximate. Several open reasoning models (Kimi, the GLM max variants, DeepSeek V4, MiMo) may also have sampled via the same fallback, so read close differences among neighbours as ties. ‡ Qwen2.5 Coder 32B is 8/9 (ConfAIDE would not complete on its endpoint). See Methodology, Reasoning-locked models.

**The board spans about 36 points (45.0 to 81.0) while capability spans more than tenfold.** Below Fable 5 the field compresses hard: ranks 2 through 21 fall inside about 10 points and freely mix the most and least capable models. Qwen3-235B (open, mid-capability) sits at #7 above most closed frontier systems, GPT-4o (low-capability) at #6, and three of the four lowest RAI scores belong to mid-capability open models.

### Open weights are competitive on responsibility

**25 of the 37 models are open-weight, and the highest-scoring open model (Qwen3-235B) is #7 overall**, above most closed frontier systems. Open models appear at every level of the board, top to bottom. Responsibility is not a closed-model advantage. It is also not an open-model advantage: open models occupy both the upper-middle and the entire bottom of the board.

### Capability doesn't track responsibility within a lab either

**GPT-4o (70.2) outscores the newer, more capable GPT-5.2 (59.5)**, and within DeepSeek the older V3.1/V3.2 (67.0/67.6) edge out the newer, more capable V4 Pro (64.5). Within a single developer, more advanced does not reliably mean more responsible.

### Sycophancy, new this round

The 9th benchmark measures **sycophancy resistance**: whether a model reverses a correct answer when the user pushes back ("I don't think that's right, are you sure?"). It is scored as a deterministic flip-rate with **no LLM judge**, so it is one of the cleanest constituents on the board. Resistance varies enormously and does not track rank: Claude Haiku 4.5 (0.99) and Sonnet 5 (0.97) almost never cave, while GPT-5.2 (0.20) and Llama 4 Maverick (0.22) reverse a correct answer under mild pressure most of the time.

### Read this as a defined index, with error bars

- **The correlation is weak, not significant, and has drifted down as the board filled.** r has moved between about 0.13 and 0.35 across board fills; at n = 37 it is 0.13 with a bootstrap 95% CI of [−0.24, +0.45] that includes zero. The point estimate is sensitive to which models are present, so the **scatter is the finding, not the point estimate.**
- **Sampled** (about 150 items per task): the composite's 95% half-width is roughly ±2 points, so differences inside the compressed middle of the board are ties. The real signal is the top-to-bottom spread, not the order of neighbours.
- **Generative MCQ scoring is validated** against the canonical loglikelihood method (within about 3 to 6 points; see Methodology, Calibration).
- **Reasoning-locked models** are scored at temperature 1 or the model default; **Phi-4 and Mistral Large** are excluded (un-evaluable on our endpoints).
- The RAI Score is an **unweighted, defined index** across 8 dimensions and 9 benchmarks, built for relative comparison, not an absolute safety certificate. WMDP (security) penalizes hazardous knowledge, so a very knowledgeable model scores lower there.
