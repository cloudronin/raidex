_2026-07 roster refresh. 23 frontier models scored on all 8 benchmarks. Mistral Large and Phi-4 are excluded (un-evaluable on our endpoints). Every number is an independent automated evaluation, not a self-reported score._

### Capability is a weak, unstable predictor of responsibility

Across the 23 models, capability (Artificial Analysis Intelligence Index) and RAI Score are only weakly related, and the relationship is not stable as the board grows. Pearson **r rose from 0.17 (n=17) to 0.35 (n=23)** when the 2026-07 frontier models were added, but the bootstrap **95% CI is [−0.13, +0.65]**, which still includes zero. Almost all of that jump comes from one model: Claude Fable 5, the most capable model on the board, also tops RAI and sits in the high-capability, high-responsibility corner that pulls the correlation up. Remove it and r falls back to 0.19.

So the honest reading is neither "capability is decoupled from responsibility" nor "capability predicts responsibility." The point estimate is small, it swings with individual models, and it cannot be distinguished from zero at this sample size. **The scatter is the finding, not the coefficient.** Both corners of the plot are now populated:

- **Claude Fable 5** (most capable, AA 60) is **#1** on responsibility (79.7). The frontier can lead.
- **GLM-5.2**, the current open-weight capability leader, sits near the bottom (**#21**, 59.9): high capability, low responsibility.
- **Qwen3-235B** (open, mid-capability) is **#4**, above most closed frontier models.
- **GPT-4o and Gemini 2.5 Flash**, among the least capable here, sit mid-pack (#6 and #7), ahead of several newer and more capable models.
- Within OpenAI, **GPT-4o (69.2) still outscores the newer, more capable GPT-5.2 (64.2)**.

A most-capable model at the top and a capability leader near the bottom are exactly why the line is weak and the spread is the story.

### Fable 5 at #1: a note on the judge

Fable 5's 79.7 comes with a caveat we state plainly. Fable 5 is an Anthropic model, and Raidex's fixed judge for the LLM-judged constituents (SimpleQA, XSTest) is also an Anthropic model, Claude Sonnet 4.6, so the board's current #1 and its judge now share a lab. We checked where Fable 5's lead concentrates:

- Its single largest advantage is **SimpleQA factuality, +49 points over the board average, and that constituent is sibling-judged**. SimpleQA grades answers against gold, so it is less exposed to stylistic self-preference than a subjective safety call, but it is the one to watch.
- It also leads by wide margins on **WMDP (+39) and ETHICS (+16), which are fully deterministic** and use no LLM judge. The top rank is therefore substantially earned on constituents the judge never touches.
- On the actually sibling-judged safety benchmark (XSTest, +1.7) and the OpenAI-judged StrongREJECT (+0.8), Fable 5 is barely above average, because those are near the ceiling for everyone.

Net: the rank is defensible on deterministic dimensions, but the shared-lab situation and the large sibling-judged factuality margin are disclosed as a limitation (see Methodology, LLM-judge bias).

### The board, closed and open, every capability tier

| # | Model | RAI | |
|---|-------|----:|---|
| 1 | Claude Fable 5 † | 79.7 | |
| 2 | Claude Opus 4.8 | 71.6 | |
| 3 | Gemini 3.5 Flash † | 71.3 | |
| 4 | **Qwen3-235B** | 69.6 | open |
| 5 | Claude Sonnet 5 † | 69.3 | |
| 6 | Gemini 2.5 Flash | 69.2 | |
| 7 | GPT-4o | 69.2 | |
| 8 | GPT-5.5 † | 69.0 | |
| 9 | Claude Sonnet 4.6 | 68.6 | |
| 10 | **Llama 3.3 70B** | 68.0 | open |
| 11 | **DeepSeek V3.2** | 66.1 | open |
| 12 | **Llama 4 Maverick** | 65.2 | open |
| 13 | **DeepSeek V3.1** | 64.4 | open |
| 14 | GPT-5.2 | 64.2 | |
| 15 | **DeepSeek V4 Pro** | 64.1 | open |
| 16 | **Gemma-4 31B** | 63.6 | open |
| 17 | GPT-4o-mini | 62.6 | |
| 18 | **Gemma-3 27B** | 62.4 | open |
| 19 | Claude Haiku 4.5 | 62.2 | |
| 20 | Grok 4.3 | 61.3 | |
| 21 | **GLM-5.2** | 59.9 | open |
| 22 | **MiniMax-M2.7** | 58.5 | open |
| 23 | **gpt-oss-120B** | 54.8 | open |

† Reasoning-locked (Fable 5, Gemini 3.5 Flash, Sonnet 5, GPT-5.5). Their MCQ benchmarks run at temperature 1 (or the model default, sampled), so treat those scores as approximate. See Methodology, Reasoning-locked models.

**The board spans ~25 points (54.8 to 79.7) while capability spans more than tenfold.** Below Fable 5, the field is tightly compressed: #2 through #12 fall inside ~6.5 points and mix the most and least capable models. Qwen (open, mid-cap) and GPT-4o (low-cap) sit alongside Opus and the newest Claude and Gemini models.

### Open weights are competitive on responsibility

**11 of the 23 models are open-weight, and one (Qwen3-235B) is #4 overall**, above most closed frontier systems. Open models appear at every level of the board. Responsibility is not a closed-model advantage. It is also not an open-model advantage: the open-weight capability leader (GLM-5.2) is near the bottom.

### Capability doesn't track responsibility within a lab either

**GPT-4o (69.2) outscores the newer, more capable GPT-5.2 (64.2)**, and GPT-5.5, OpenAI's most capable, carries the most hazardous knowledge of any OpenAI model here. Within a single developer, more advanced does not mean more responsible.

### The reporting gap this fills

Frontier developers report capability benchmarks almost universally but Responsible-AI benchmarks rarely (see **The Gap**). Raidex runs all 8 independently. None of these numbers are self-reported.

### Read this as a defined index, with error bars

- **The correlation is weak, not significant, and unstable as the board fills.** r moved 0.13, then 0.29, then 0.17, and now 0.35 as models landed (n=23; bootstrap 95% CI [−0.13, +0.65], which still includes zero). Adding Fable 5 alone moved it from 0.19 to 0.35, so the point estimate is sensitive to individual models, especially the newest and most capable ones. The **scatter is the finding, not the point estimate.**
- **Fable 5 is on the leaderboard and on the scatter** (AA Intelligence Index 60, v4.1). Very new models occasionally lack an AA score and then appear on the board but not the scatter; Fable 5 is scored, so it is present in both.
- **Sampled** (≈150 to 300 items/task): the composite's 95% half-width is ~±2 points, so differences inside the compressed top cluster are ties. The real signal is the top-to-bottom spread, not the order of neighbours.
- **Generative MCQ scoring is validated** against the canonical loglikelihood method (within ~3 to 6 points; see Methodology, Calibration).
- **Reasoning-locked models** (Fable 5, Gemini 3.5 Flash, Sonnet 5, GPT-5.5) are scored at temperature 1 or the model default; **Phi-4 and Mistral** are excluded (un-evaluable on our endpoints).
- The RAI Score is an **unweighted, defined index** across 7 dimensions, built for relative comparison, not an absolute safety certificate. WMDP (security) penalizes hazardous knowledge, so a very knowledgeable model scores lower there.
