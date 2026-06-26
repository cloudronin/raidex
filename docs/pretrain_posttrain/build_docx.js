const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign,
        LevelFormat, ExternalHyperlink } = require("docx");

const CONTENT_W = 9360;
const HEADER_FILL = "E6F1FB";
const ALT_FILL = "F5F7FA";
const border = { style: BorderStyle.SINGLE, size: 1, color: "C7CDD4" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 110, right: 110 };

function readCSV(path) {
  const lines = fs.readFileSync(path, "utf8").trim().split("\n");
  const header = lines[0].split(",");
  const rows = lines.slice(1).map(l => l.split(","));
  return { header, rows };
}
function sgn(v) {
  if (v === "" || v === undefined || v === null) return "";
  const n = parseFloat(v);
  if (Number.isNaN(n)) return v;
  return (n >= 0 ? "+" : "") + n.toFixed(Math.abs(n) < 1 ? 3 : 1);
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after === undefined ? 160 : opts.after, line: 276 },
    alignment: opts.align,
    children: [new TextRun({ text, italics: opts.italic, size: opts.size || 22, color: opts.color })],
  });
}
function runsPara(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after === undefined ? 160 : opts.after, line: 276 },
    children: runs.map(r => new TextRun({ text: r.t, bold: r.b, italics: r.i, size: 22 })),
  });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 },
    children: [new TextRun({ text })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 },
    children: [new TextRun({ text })] });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 270 },
    children: [new TextRun({ text, size: 22 })] });
}

function cell(text, w, { fill, align, bold, italic, size } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA }, margins: cellMargins,
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text: text === "" ? " " : String(text), bold, italics: italic, size: size || 20 })] })],
  });
}
function table(headers, rows, widths, { aligns, caption } = {}) {
  const headRow = new TableRow({ tableHeader: true, children:
    headers.map((h, i) => cell(h, widths[i], { fill: HEADER_FILL, bold: true,
      align: aligns && aligns[i] === "r" ? AlignmentType.RIGHT : (aligns && aligns[i] === "c" ? AlignmentType.CENTER : AlignmentType.LEFT) })) });
  const dataRows = rows.map((r, ri) => new TableRow({ children:
    r.map((c, i) => cell(c, widths[i], {
      fill: ri % 2 ? ALT_FILL : undefined,
      align: aligns && aligns[i] === "r" ? AlignmentType.RIGHT : (aligns && aligns[i] === "c" ? AlignmentType.CENTER : AlignmentType.LEFT),
      italic: r._italic })) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [headRow, ...dataRows] });
}

const FAMILY_SHORT = { "Llama-3.1-8B": "Llama-3.1-8B", "Gemma-2-9B": "Gemma-2-9B", "Mistral-7B": "Mistral-7B" };

// ---- Table 1: composite RAI ----
const t1 = readCSV("table1_composite_rai.csv");
const t1rows = t1.rows.map(r => [r[0], r[1], r[2], sgn(r[3])]);
const T1 = table(["Family", "Base RAI", "Instruct RAI", "Δ"], t1rows,
  [3000, 2120, 2120, 2120], { aligns: ["l", "r", "r", "r"] });

// ---- Table 2: Test 1 zero-shot deltas ----
const t2 = readCSV("table2_test1_zeroshot_deltas.csv");
const t2rows = t2.rows.map(r => [r[0], r[2].replace("-bound", ""), r[3],
  sgn(r[4]), sgn(r[5]), sgn(r[6]), sgn(r[7])]);
const T2 = table(["Benchmark", "Predicted", "Conf.", "Llama-3.1", "Gemma-2", "Mistral-7B", "Mean (clean)"],
  t2rows, [1560, 1760, 900, 1080, 1080, 1080, 1900],
  { aligns: ["l", "l", "c", "r", "r", "r", "r"] });

// ---- Table 3: few-shot vs zero-shot ----
const t3 = readCSV("table3_fewshot_vs_zeroshot.csv");
const t3rows = t3.rows.map(r => [r[0], r[2], sgn(r[3]), sgn(r[4])]);
const T3 = table(["Benchmark", "Family", "Zero-shot Δ", "Few-shot (5-shot) Δ"],
  t3rows, [2200, 2600, 2280, 2280], { aligns: ["l", "l", "r", "r"] });

// ---- Table 4: Test 2 correlation ----
const t4 = readCSV("table4_test2_correlation.csv");
const t4rows = t4.rows.map(r => {
  const ci = `[${parseFloat(r[5]).toFixed(2)}, ${parseFloat(r[6]).toFixed(2)}]`;
  const inv = r[3] === "True" ? " †" : "";
  return [r[0] + inv, r[1].replace("_", " "), r[2].replace("-bound", ""), sgn(r[4]), ci, r[7]];
});
const T4 = table(["Benchmark", "Dimension", "Predicted", "r", "95% CI", "n"],
  t4rows, [1640, 1900, 1820, 1100, 2200, 700], { aligns: ["l", "l", "l", "r", "c", "c"] });

const figure = new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync("figure1_deltas.png"),
    transformation: { width: 600, height: 350 },
    altText: { title: "Base to instruct deltas", name: "figure1",
      description: "Grouped bars of base-to-instruct normalized deltas, zero-shot vs five-shot, for BBQ, ETHICS, and WMDP." } })] });

function cap(text) {
  return new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, italics: true, size: 19, color: "5F5E5A" })] });
}

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "Does responsibility partition into pretraining-bound and post-training-bound dimensions?", bold: true, size: 34 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: "A base-versus-instruct study of a responsible-AI index", size: 26, color: "5F5E5A" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "Vishnu Vettrivel", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 260 },
    children: [new TextRun({ text: "Working draft: methods and results", size: 20, color: "888780" })] }),

  h1("Abstract"),
  body("Composite responsible-AI (RAI) indices score less-capable models as high as or higher than frontier models on responsibility, suggesting that responsibility is not a simple function of scale. We ask a mechanistic question: do the dimensions of such an index partition into pretraining-bound dimensions, inherited from scale, and post-training-bound dimensions, added by alignment? We test this two ways. First, across three open-weight model families we evaluate matched base and instruct checkpoints served identically, so the per-dimension base-to-instruct delta isolates the post-training contribution. Second, across a 17-model board we correlate each dimension with an external capability measure. Both methods converge on the same partition: fairness and machine ethics are post-training-bound, added by alignment, while factuality is pretraining-bound and tracks capability (hazardous knowledge, used as a within-study control, tracks capability as expected). Refusal and over-refusal point the same way for two of three families on Test 1, but they are format-confounded on base checkpoints, and over-refusal carries a capability correlation on Test 2, so we report them as suggestive rather than established. A five-shot control rules out the explanation that the gains merely reflect base models failing to follow the answer format. We frame the result as a property of the dimensions, not as a readout of any lab's training budget.", { after: 120 }),

  h1("1. Introduction"),
  body("Capability is largely a pretraining property: it scales with data and compute. Responsibility, on most of the behaviors that responsible-AI indices measure, is plausibly a post-training property, instilled by alignment, RLHF, and refusal tuning. If that is right, the dimensions of an RAI index should not behave uniformly. Some should be pretraining-bound and therefore track capability; others should be post-training-bound and therefore decouple from it. This paper tests that partition directly."),
  body("The motivating observation comes from a public RAI leaderboard on which smaller models frequently match or exceed frontier models on responsibility. A composite score cannot, by itself, explain this: a high score is consistent with heavy post-training, with a cleaner pretraining corpus, or with benchmark-shaped tuning. The composite also cannot say which dimensions move for which reason. The signal we are after lives at the dimension level, and the cleanest instrument for it is the gap between a base checkpoint and its instruct sibling."),
  body("We make three contributions:"),
  bullet("A natural-experiment design. For matched same-version base/instruct pairs served identically, the per-dimension base-to-instruct delta is a direct measurement of what post-training changed, separated from pretraining. We show that this requires local serving: no commercial inference provider serves base checkpoints, because they lack a chat template."),
  bullet("Evidence for a clean partition. Across three model families, fairness and machine ethics rise substantially from base to instruct while hazardous knowledge stays flat; across a 17-model board, factuality is the one dimension that tracks capability. The two methods are independent and agree."),
  bullet("A format-following control. Because base models follow answer formats poorly, part of a delta could be format-following rather than alignment. A five-shot pass applied equally to base and instruct leaves the gains essentially unchanged, ruling this out."),
  body("We are careful not to overclaim. We do not claim that the scores reveal how a lab allocated its training versus post-training budget; a dimension can move for reasons unrelated to alignment effort, including a cleaner corpus or tuning shaped by the benchmark itself. The claim is narrower and mechanistic: RAI dimensions partition into pretraining-bound and post-training-bound, and the base-to-instruct delta measures the latter.", { after: 120 }),

  h1("2. Methods"),
  h2("2.1 Setup and framing"),
  body("We evaluate models on the eight benchmarks of an existing RAI index. Six map to dimensions we can class a priori. Refusal (StrongREJECT), over-refusal (XSTest), fairness (BBQ), and machine ethics (ETHICS) are predicted post-training-bound. Hazardous knowledge (WMDP) and factuality (SimpleQA) are predicted pretraining-bound. All scores are normalized to a common scale from 0 to 1 on which higher is better; for lower-is-better benchmarks (WMDP, StrongREJECT) the normalization inverts the raw value."),
  h2("2.2 Test 1: the base-to-instruct delta"),
  body("We evaluate matched base and instruct checkpoints from three open-weight families spanning three developers: Llama-3.1-8B (Meta), Gemma-2-9B (Google), and Mistral-7B (Mistral AI). Each pair is a same-version release, so the only difference between the two checkpoints is post-training. Qwen2.5 was considered but dropped: it ships no base checkpoint on our serving path. Llama 3.3 was excluded because it shipped instruct-only."),
  body("Serving is the methodological crux. No serverless inference provider we tested serves base checkpoints at all: each base model is rejected as not a chat model, because base models have no chat template. We therefore serve both base and instruct locally and identically, through one harness at a single 4-bit quantization, so the measured delta isolates post-training rather than confounding it with serving or precision differences. Comparing a locally-served base against a cloud-served instruct would not be valid. A single off-family judge model is held constant across base and instruct so the delta is not contaminated by judge drift. Large benchmarks are sampled at n = 50; the base-to-instruct effects are large relative to that sampling noise."),
  body("Base models lack a chat template and tend to continue prompts rather than answer them. We handle this two ways. First, we flag the three LLM-judged benchmarks (StrongREJECT, XSTest, SimpleQA) as format-confounded for base runs and exclude them from the clean signal, because a base model's score there reflects format failure as much as behavior. Second, we run a five-shot pass on the three generative multiple-choice benchmarks (BBQ, ETHICS, WMDP), applied identically to base and instruct. If a post-training-bound delta survives equal format scaffolding, the gain is genuine alignment rather than the base failing to follow the answer format. The kill criterion is to stop if fewer than three families show a positive mean delta on the clean, non-confounded post-training-bound benchmarks. We report one correction to this criterion, because it shows the format-confound flag working as intended. Our initial implementation folded the confounded judge benchmarks into the per-family vote, and a single confounded artifact, Mistral's StrongREJECT delta of -0.51 (Section 3.3), tripped it. On inspection that delta is a base-model format failure rather than a real loss of refusal, so we corrected the criterion to test only the clean post-training-bound benchmarks, on which all three families pass. The episode shows that the confound can deflate a score as readily as inflate it, which is why the judge benchmarks are flagged and kept out of the clean signal."),
  h2("2.3 Test 2: dimension-capability correlation"),
  body("Independently, across the 17 models on the public board, we correlate each dimension's normalized score with an external capability index (the Artificial Analysis Intelligence Index), reporting Pearson r with bootstrap 95% confidence intervals (n = 17). If the partition is real, pretraining-bound dimensions should track capability and post-training-bound dimensions should not.", { after: 120 }),

  h1("3. Results"),
  body("At the composite level, post-training raised the RAI score for two of three families but not the third (Table 1). That inconsistency is exactly why the composite is the wrong unit of analysis; the partition appears cleanly only at the dimension level."),
  T1,
  cap("Table 1. Composite RAI score, base vs instruct, per family. The composite conflates dimensions that move in opposite directions; Mistral's small negative is resolved at the dimension level (Section 3.3)."),

  h2("3.1 The dimensions partition (zero-shot)"),
  body("On the clean, non-confounded benchmarks the partition is sharp and consistent across all three families (Figure 1, Table 2). Both post-training-bound dimensions rise substantially from base to instruct. BBQ and ETHICS each gain about +0.17 normalized, with all three families individually positive, while the pretraining-bound control, WMDP, stays flat at about -0.04."),
  figure,
  cap("Figure 1. Base-to-instruct delta (normalized) for the three clean multiple-choice benchmarks, zero-shot vs five-shot, averaged over the three families. Post-training-bound dimensions (BBQ, ETHICS) gain and hold under few-shot; the pretraining-bound control (WMDP) stays near zero in both conditions."),
  body("The strongest evidence is an internal control. BBQ, ETHICS, and WMDP are all generative multiple-choice benchmarks with identical answer-format demands. If the fairness and ethics gains were merely instruct models following the format better, WMDP would rise in lockstep. Instead it is roughly four times smaller and does not move with the others. The divergence isolates a dimension-specific (alignment) effect from a generic format-following effect."),
  T2,
  cap("Table 2. Test 1 base-to-instruct deltas (normalized) per benchmark and family. Confounded (LLM-judged) benchmarks are excluded from the clean mean; see Section 3.3 for the Mistral StrongREJECT case."),

  h2("3.2 The gains survive format control (few-shot)"),
  body("Under five-shot prompting applied equally to base and instruct, the post-training-bound deltas are essentially unchanged: BBQ +0.17 to +0.19 and ETHICS +0.17 to +0.16, with WMDP holding near zero (Table 3, Figure 1). ETHICS in particular is nearly identical across conditions, indicating its base-to-instruct gap has almost no format-following component. Equalizing the format scaffolding does not erase the gains, so they reflect alignment rather than the base model's inability to follow instructions."),
  T3,
  cap("Table 3. Few-shot (5-shot) vs zero-shot base-to-instruct deltas on the clean benchmarks. The post-training-bound gains persist; the pretraining-bound control stays near zero."),

  h2("3.3 Judge-scored refusal benchmarks"),
  body("The refusal benchmarks behave as predicted for two of three families: StrongREJECT improves +0.26 (Llama) and +0.28 (Gemma), XSTest +0.24 and +0.22. Both are format-confounded for base models, however, and are excluded from the clean signal. Mistral is the exception: its StrongREJECT delta is -0.51 (base attack-success 0.32, instruct 0.82). Inspection shows the base model produces compliant content that derails into hallucinated multi-turn dialogue, which the rubric scores as less harmful, deflating base attack-success; this compounds with Mistral-7B-Instruct's genuinely light refusal tuning. The case illustrates why these benchmarks are flagged: an unguarded analysis would let the artifact veto an otherwise clean result."),

  h2("3.4 Capability correlation (Test 2)"),
  body("The correlation structure largely agrees with Test 1 (Table 4). Factuality is the one dimension whose correlation with capability excludes zero (r = +0.56, 95% CI [0.08, 0.83]), confirming that it tracks capability as a pretraining-bound dimension should. The two confirmed post-training-bound dimensions behave as predicted: fairness (BBQ, r = -0.14) and machine ethics (ETHICS, r = +0.08) both have confidence intervals that span zero, so neither tracks capability. Hazardous knowledge (WMDP, r = -0.25) correlates negatively, but because its score inverts raw accuracy this means raw hazardous-knowledge performance rises with capability, again consistent with a pretraining-bound dimension."),
  body("Over-refusal is the exception, and we flag it rather than pass over it. XSTest correlates with capability at r = -0.33, 95% CI [-0.63, -0.01], an interval that excludes zero, whereas a cleanly post-training-bound dimension should show no correlation. Two readings are possible at this sample size: over-refusal calibration may itself be partly capability-linked, since more capable instruct models tend to avoid over-refusing, or the partition simply does not hold for over-refusal at n = 17. Either way we do not claim over-refusal as post-training-bound. Refusal (StrongREJECT, r = +0.13) has an interval spanning zero, consistent with no capability link, but it is format-confounded in Test 1 and so remains suggestive."),
  T4,
  cap("Table 4. Test 2: Pearson correlation of each dimension's normalized score with capability across 17 models, with bootstrap 95% CIs. † marks lower-is-better benchmarks whose normalized score inverts raw performance."),

  h2("3.5 Convergence"),
  body("Three independent lines converge on the same partition: the zero-shot base-to-instruct delta, the format-controlled five-shot delta, and the capability correlation. Together they establish that fairness and machine ethics are post-training-bound and that factuality is pretraining-bound, with hazardous knowledge tracking capability as a within-study control. Refusal and over-refusal are directionally consistent where we can measure them, with two of three families gaining on StrongREJECT, but they are format-confounded on base checkpoints and over-refusal carries a capability correlation that a clean post-training-bound dimension should not (Section 3.4); we therefore do not claim them as established here. The base-to-instruct delta cleanly measures the post-training contribution for the dimensions where it is not confounded.", { after: 120 }),

  h1("4. Threats to validity"),
  body("The base-to-instruct delta on instruction-format benchmarks bundles two effects of post-training, alignment and instruction-following. The five-shot control and the WMDP internal control bound, but do not fully eliminate, the latter. Models are small (7B to 9B) and 4-bit-quantized, evaluated at n = 50 per benchmark on a single local GPU; the deltas are large relative to that sampling noise, but absolute scores are not directly comparable to a full-precision board. Base checkpoints for larger pairs (27B to 235B) could not be served, because no provider serves base models, so scaling this study up requires dedicated GPU serving. Test 2's intervals are wide at n = 17. The signal is the pattern across dimensions and methods, not any single coefficient."),

  h1("Data and code availability"),
  new Paragraph({ spacing: { after: 80, line: 276 }, children: [
    new TextRun({ text: "All evaluation code (the local-serving and few-shot runners run_local.py and run_fewshot.py, and the analysis scripts analyze_delta.py and analyze_correlation.py) and the result JSONs and CSV tables underlying Figure 1 and Tables 1 to 4 are available at ", size: 22 }),
    new ExternalHyperlink({ children: [new TextRun({ text: "github.com/cloudronin/raidex", style: "Hyperlink", size: 22 })], link: "https://github.com/cloudronin/raidex" }),
    new TextRun({ text: ", under backend/pretrain_posttrain/ for the code and docs/pretrain_posttrain/ for the figure, tables, per-model result JSONs (results/), and this draft. The exact state that produced these artifacts is pinned at the git tag pretrain-posttrain-paper-v1.", size: 22 }),
  ] }),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "1A1A1A" },
        paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "1A1A1A" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [ { reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 280 } } } } ] } ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("raidex_pretrain_posttrain_paper.docx", buf);
  console.log("wrote raidex_pretrain_posttrain_paper.docx (" + buf.length + " bytes)");
});
