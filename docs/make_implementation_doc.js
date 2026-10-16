// Builds docs/Track_A_Implementation.docx: a full walkthrough of how FinAssist Track A works.
// Regenerate: npm install docx@9.5.1, then
//   node docs/make_implementation_doc.js . docs/Track_A_Implementation.docx
// and open the result in Word once to fill in the contents list.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, TableOfContents, PageBreak,
  Footer, Header, PageNumber, TabStopType,
} = require("docx");

const ROOT = process.argv[2];
const OUT = process.argv[3];
const img = (f) => fs.readFileSync(path.join(ROOT, "docs", "diagrams", f));

// ---------- styling ----------
const FONT = "Calibri";
const MONO = "Consolas";
const ACCENT = "1F4E79";
const LIGHT = "EAF1F8";
const CODEBG = "F3F4F6";
const BORDER = "BFC7D1";
const PAGE_W = 11906, MARGIN = 1134;            // A4, 2 cm margins
const CONTENT_W = PAGE_W - 2 * MARGIN;          // 9638 DXA

// Inline markup: **bold**, `code`
function runs(text, base = {}) {
  const out = [];
  for (const part of String(text).split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("`")) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: MONO, size: 19, color: "7A1F3D" }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)], pageBreakBefore: true });
const H1same = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)], spacing: { before: 360 } });
const H1first = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 264 } });
let numInstance = 0;
function N(items) { // numbered list, restarts each call
  numInstance += 1;
  const ref = numInstance;
  return items.map((t) => new Paragraph({ numbering: { reference: "numbers", level: 0, instance: ref }, children: runs(t), spacing: { after: 60, line: 264 } }));
}
function code(lines) {
  return lines.map((l, i) => new Paragraph({
    children: [new TextRun({ text: l === "" ? " " : l, font: MONO, size: 18 })],
    shading: { type: ShadingType.CLEAR, fill: CODEBG, color: "auto" },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0, line: 240 },
    indent: { left: 170, right: 170 },
  }));
}
function callout(title, text) {
  const cell = new TableCell({
    width: { size: CONTENT_W, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: LIGHT, color: "auto" },
    margins: { top: 120, bottom: 120, left: 180, right: 180 },
    borders: { left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT }, top: none, bottom: none, right: none },
    children: [
      new Paragraph({ children: [new TextRun({ text: title, bold: true, color: ACCENT })], spacing: { after: 60 } }),
      ...[].concat(text).map((t) => new Paragraph({ children: runs(t), spacing: { after: 40, line: 264 } })),
    ],
  });
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W], rows: [new TableRow({ children: [cell] })] }), spacer()];
}
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const spacer = () => new Paragraph({ children: [], spacing: { after: 60 } });

function table(header, rows, widthsPct) {
  const widths = widthsPct.map((p) => Math.round((CONTENT_W * p) / 100));
  widths[widths.length - 1] = CONTENT_W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: BORDER };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (text, i, isHead) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    shading: isHead ? { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(text, isHead ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }), spacing: { after: 0, line: 252 } })],
  });
  return [new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => mk(c, i, false)) })),
    ],
  }), spacer()];
}
function figure(file, pxW, pxH, caption) {
  const w = 620; // points-ish width in docx-js pixel units (fits 17 cm)
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: img(file), transformation: { width: w, height: Math.round((w * pxH) / pxW) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })] }),
  ];
}

// ---------- content ----------
const body = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? body.push(...x) : body.push(x)));

// Title page
add(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "FinAssist — Track A", size: 56, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 400 }, children: [new TextRun({ text: "Implementation guide: what was built and how every part works", size: 30, color: "404040" })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 4 } }, children: [] }),
  P("Fine-tuning a 7–8B open model to answer questions about company financial reports in a fixed JSON format, adding retrieval, measuring everything against a frozen test set, and serving it behind an API.", { spacing: { before: 300, after: 200 } }),
  P("**Repository:** github.com/sparsh610/finassist-track-a (private)"),
  P("**Status:** 16 October 2026. Every stage implemented and run end to end on the laptop smoke profile; the real 7B run on a GPU is pending."),
  P("**Companion documents:** `README.md` (commands), `docs/model_card.md` (results and limits), `docs/PROJECT_STATUS.md` (status, decisions, pending work)."),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, color: ACCENT })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// 1 Overview
add(
  H1("1. What FinAssist does"),
  P("FinAssist is an analyst assistant for company financial reports (10-K filings and earnings tables). A user asks a question about a report page; the system finds the relevant table rows and sentences, works out the answer, and replies with **exactly one JSON object** containing the answer, the arithmetic it did, and a pointer to the evidence."),
  H2("1.1 One example, end to end"),
  P("**Question** (with the filing selected: STT annual report (10-K), fiscal 2009, page 25):"),
  ...code(["what was the percent change in the aggregate net asset values of the",
           "direct lending collateral pools between 2007 and 2008?"]),
  P("**What the system should do:** find the table row \"direct lending collateral pools\" for that filing, read $150 billion (2007) and $85 billion (2008), and compute (85 − 150) / 150."),
  P("**A correct answer** in the fixed output format (illustrative; the 0.5B smoke model does not yet get this right):"),
  ...code(['{"answer": "-43.3%",',
           ' "calculation_steps": ["85 - 150 = -65", "-65 / 150 = -0.4333"],',
           ' "source": "table row: direct lending collateral pools"}']),
  P("The gold answer for this test question is −43%. The evaluation accepts small rounding differences (section 9.2)."),
  H2("1.2 The four parts"),
  ...table(["Part", "What it does", "Week"], [
    ["Data", "Turns two public datasets (FinQA, TAT-QA) into training and test examples in the JSON format, and freezes them.", "1"],
    ["Model adaptation", "Fine-tunes a 7B model with QLoRA so it reasons over tables and always answers in the format. Optional DPO on top.", "2"],
    ["Retrieval (RAG)", "Indexes 1,000 report pages so the model gets the right evidence at question time instead of the whole report.", "3"],
    ["Evaluation and serving", "Measures every combination on the same frozen test set, then merges, quantises and serves the best model behind an API gateway.", "3–4"],
  ], [20, 68, 12]),
  ...figure("architecture.png", 1989, 1200, "Figure 1. End-to-end architecture. Colours: data (blue), model adaptation (green), retrieval (purple), evaluation (orange), serving (red)."),
  ...callout("Why this design", [
    "**Fine-tuning** teaches the reasoning style and the output format. **Retrieval** supplies the facts for each question. Neither alone is enough: a fine-tuned model without retrieval has no figures to work with, and a base model with retrieval answers in an inconsistent format.",
    "The evaluation therefore compares all combinations (base, base + RAG, fine-tuned, fine-tuned + RAG) on one frozen test set.",
  ]),
);

// 2 Repository
add(
  H1("2. Repository layout"),
  ...table(["Path", "Contents"], [
    ["`configs/base.yaml`", "Shared settings: model ids, data sources, RAG parameters, tracking, judge, serving."],
    ["`configs/runs/*.yaml`", "One file per experiment (run1_r16, run2_r64, ablation_attn_only, dpo_stretch); only overrides."],
    ["`configs/compute.yaml`", "Where jobs run: smoke (laptop CPU), local_gpu, notebook (Colab/Kaggle), iit_ssh."],
    ["`src/finassist/data/`", "`prepare.py` (build and freeze the dataset), `finqa_program.py` (execute FinQA programs)."],
    ["`src/finassist/prompts.py`", "System prompt and message builder shared by training, evaluation and serving."],
    ["`src/finassist/schema.py`", "The output JSON schema and the tolerant parser that checks it."],
    ["`src/finassist/llm/backends.py`", "Inference back ends: HF transformers (in-process), OpenAI-compatible servers, mock."],
    ["`src/finassist/train/`", "`qlora.py` (supervised fine-tuning), `dpo.py` (preference tuning stretch goal)."],
    ["`src/finassist/rag/`", "`chunking.py`, `index.py` (embed + store), `retriever.py` (search, filter, rerank), `recall.py` (retrieval eval)."],
    ["`src/finassist/eval/`", "`run_eval.py` (the harness), `metrics.py`, `judge.py` (LLM-as-judge)."],
    ["`src/finassist/serve/`", "`merge.py`, `quantize.py`, `app.py` (FastAPI gateway)."],
    ["`src/finassist/redteam/`", "20 adversarial prompts and the runner that writes the red-team log."],
    ["`src/finassist/compute/runner.py`", "Runs a stage locally or ships it to a GPU box over SSH."],
    ["`src/finassist/cli.py`", "Single entry point: `python -m finassist.cli <stage>`."],
    ["`scripts/run_pipeline.sh`", "Runs every stage in order, resumably. `scripts/gateway_smoke.sh` tests the API."],
    ["`notebooks/colab_track_a.ipynb`", "Runs the full pipeline on a Colab GPU with outputs saved to Google Drive."],
    ["`serving/`", "Dockerfile (gateway) and docker-compose.yml (vLLM + gateway) for the GPU deployment."],
    ["`data/processed/`, `data/corpus/`", "The frozen dataset (train/test JSONL, manifest with SHA-256 hashes) and the 1,000-page corpus."],
    ["`tests/`", "16 fast tests that need no model weights."],
    ["`outputs/`", "Everything generated (adapters, indexes, evaluations, merged/quantised models, logs). Not in git."],
  ], [32, 68]),
);

// 3 Config & compute
add(
  H1("3. Configuration and compute profiles"),
  H2("3.1 How configuration is assembled"),
  P("Every stage calls `load_config(run, compute)` in `config.py`, which builds one dictionary in three layers:"),
  ...N([
    "`configs/base.yaml`: the shared defaults.",
    "A run file from `configs/runs/` (for training stages): deep-merged on top, so it only lists what differs.",
    "The chosen compute profile from `configs/compute.yaml`: exposed as `cfg[\"compute\"]`. A profile can swap the model (`model_override: smoke_id` uses the 0.5B stand-in), cap the data (`limit_train`, `limit_test`), and override training or DPO settings for small hardware.",
  ]),
  H2("3.2 Compute profiles"),
  ...table(["Profile", "Where", "Model", "Purpose"], [
    ["`smoke`", "Laptop CPU", "Qwen2.5-0.5B, full precision", "Prove every stage runs: 20 train / 10 test examples, 1 epoch, no tracking."],
    ["`local_gpu`", "Any machine with a CUDA GPU (e.g. a friend's)", "Qwen2.5-7B, 4-bit", "Full run on local hardware."],
    ["`notebook`", "Colab / Kaggle T4", "Qwen2.5-7B, 4-bit", "Full run in a free cloud notebook."],
    ["`iit_ssh`", "IIT GPU machine over SSH", "Qwen2.5-7B, 4-bit", "Final runs; host and user still to be filled in."],
  ], [14, 26, 22, 38]),
  P("The same commands work on every profile: `python -m finassist.cli --compute <profile> <stage>`. For an `ssh` profile, `compute/runner.py` packs the code and frozen data into a tar stream, runs the same module on the remote machine, and copies `outputs/` back. It uses only the OpenSSH client and Python's tarfile, so it works from Windows without rsync."),
  ...figure("compute_profiles.png", 1710, 701, "Figure 2. One CLI, swappable back ends."),
  H2("3.3 GPU precision"),
  P("`llm/backends.py:cuda_dtype()` picks **bf16 on Ampere or newer GPUs** (A100, RTX 30/40) and **fp16 on older ones** such as the T4. It checks the compute capability directly, because `torch.cuda.is_bf16_supported()` returns True on a T4 through slow emulation. Training, DPO and evaluation all use this helper."),
);

// 4 Data
add(
  H1("4. Data pipeline (Week 1)"),
  P("`python -m finassist.cli prepare` (code: `data/prepare.py`) downloads the two datasets, converts every example into one common record, splits and filters them, and freezes the result."),
  H2("4.1 Sources"),
  ...table(["Dataset", "Content", "Licence", "Train / test used"], [
    ["FinQA (Chen et al., 2021)", "S&P 500 10-K pages: text plus one table, questions needing multi-step arithmetic. Each comes with a reasoning program.", "MIT", "1,000 / 120"],
    ["TAT-QA (Zhu et al., 2021)", "Annual-report tables with paragraphs; arithmetic, span and multi-span answers.", "CC BY 4.0", "500 / 80"],
  ], [24, 46, 12, 18]),
  H2("4.2 Converting an example"),
  P("Each source example becomes a record with: `question`, `context` (what the model sees), `answer`, `answer_type`, `calculation_steps`, `source`, the document id and its metadata (e.g. \"ADI annual report (10-K), fiscal 2009, page 49\")."),
  H3("Calculation steps from FinQA programs"),
  P("FinQA gives a small program for each answer, e.g. `subtract(5829, 5735), divide(#0, 5735)`. `data/finqa_program.py` executes it: `#0` refers to the result of step 0, `const_100` is a literal, and `table_sum(row, none)`-style operations aggregate a table row. The executor turns each operation into a readable step:"),
  ...code(['subtract(5829, 5735), divide(#0, 5735)   ->   ["5829 - 5735 = 94", "94 / 5735 = 0.0164"]']),
  P("TAT-QA derivations (e.g. `(828-713)/713`) are checked against a safe arithmetic pattern and kept as a single step."),
  H3("The context"),
  P("`build_context()` writes the page **table first** (as a Markdown table), then the gold evidence sentences, then other sentences from the page until a 6,000-character limit. During training this \"oracle\" context teaches the reasoning and format; at test time the retrieval layer supplies the context instead."),
  H3("The source citation"),
  P("The `source` field is a **short pointer to the evidence**, built by `cite()`:"),
  B("For a gold table row: `table row: <row label>`, e.g. `table row: net revenue`."),
  B("For a gold sentence: `text: <first 10 words> ...`."),
  B("Parts are joined with `; ` and cut only between parts, at most 100 characters (median 49)."),
  ...callout("Why short citations (fixed 16 Oct 2026)", [
    "Originally `source` held up to 300 characters of verbatim evidence, cut mid-word. The model learned to paste page text and stop at an arbitrary point. After 4-bit quantisation it kept pasting until the 384-token limit cut the JSON mid-string (only 60% of served answers were valid). Short citations fixed it; see section 11.4 for the matching serving guard.",
  ]),
  H2("4.3 Splitting, cleaning and freezing"),
  B("Official dataset splits are respected, and **no report page appears in both train and test**: training examples from test pages are removed, so the test set measures generalisation."),
  B("Noisy training labels are dropped (~6%): if an example's own calculation does not reproduce its gold answer (sign flips, scale slips), it is excluded. The test set is left as is."),
  B("Sampling uses a fixed seed (42), so `prepare` is reproducible."),
  B("The outputs `train.jsonl` (1,500), `test.jsonl` (200) and the corpus (1,000 pages) are hashed with SHA-256 into `data/processed/manifest.json`. Training calls `verify_frozen()`, which stops if either file changed: results stay comparable."),
  B("The corpus is every test document plus training documents as distractors, up to 1,000 pages, so retrieval has to find the right page among many."),
  B("Each record also stores its chat `messages` and the gold `target` JSON, so training and evaluation read exactly the same format."),
  P("The dataset was frozen on 15 Oct 2026 and re-frozen on 16 Oct after the citation fix. Only `source`, `target` and `messages` changed; questions, answers, contexts and the corpus are byte-identical."),
);

// 5 Prompt & schema
add(
  H1("5. Prompt and output format"),
  H2("5.1 The prompt"),
  P("`prompts.py` builds every model input, for training, evaluation and serving alike: if training and inference built messages differently, the fine-tuned model would see an unfamiliar format at test time. A request has a system message and a user message:"),
  ...code([
    "SYSTEM: You are FinAssist, an analyst assistant for company financial reports ...",
    "        Answer the question using ONLY the provided context.",
    "        Respond with exactly one JSON object and nothing else:",
    '        {"answer": ..., "calculation_steps": [...],',
    '         "source": "<short pointer, at most 15 words>"}',
    "        Rules: [] when no arithmetic; round percentages to one decimal;",
    '        "insufficient information" if the context lacks the answer; never invent figures;',
    '        "refused" for investment advice or personal data; never copy passages into source.',
    "USER:   Document: <filing metadata>",
    "        Context: <table + sentences, or (none provided)>",
    "        Question: <question>",
  ]),
  P("The few-shot baseline adds two hand-written worked examples (not drawn from train or test) before the question."),
  H2("5.2 The schema and its parser"),
  ...table(["Field", "Type", "Limit (served model)"], [
    ["`answer`", "string", "100 characters"],
    ["`calculation_steps`", "list of strings, one arithmetic step each", "8 steps of 120 characters"],
    ["`source`", "string, short pointer to the evidence", "160 characters"],
  ], [26, 46, 28]),
  P("`schema.parse_output()` is **strict about validity but tolerant about recovery**. An output is `valid` only if the whole text is one JSON object with exactly these three keys and the right types. If it fails (prose around the JSON, a fenced code block, an extra key), the parser still tries to pull out the answer, so answer accuracy and format compliance are measured separately. Error codes such as `extra_text_around_json`, `missing:source` or `json_decode: ...` are recorded for each prediction."),
);

// 6 Baselines + 7 Training
add(
  H1same("6. Baselines (Week 1 hard gate)"),
  P("Before any training, `python -m finassist.cli baseline` evaluates the untuned base model on the frozen test set in two ways: **zero-shot** (system prompt only) and **few-shot** (two worked examples). An **oracle** run (the gold page as context) gives the upper bound for prompting alone. The project plan treats recording these baselines and freezing the data as the gate for starting Week 2."),
  H1("7. Fine-tuning with QLoRA (Week 2)"),
  P("`python -m finassist.cli train --run <run>` (code: `train/qlora.py`) fine-tunes the model on the 1,500 training examples."),
  H2("7.1 How QLoRA works here"),
  B("The base model is loaded in **4-bit NF4** with double quantisation (bitsandbytes), so a 7B model fits a 16 GB GPU. Its weights stay frozen."),
  B("Small trainable **LoRA adapters** (low-rank matrices) are added to the chosen projection layers. Only these are trained: tens of millions of parameters instead of 7 billion."),
  B("**Completion-only loss:** the loss is computed only on the assistant's JSON answer, not on the prompt and context, so the model learns to produce answers rather than to reproduce report text."),
  B("Examples longer than `max_seq_length` (2,048 tokens) are **dropped, not truncated**: right-truncation would cut off the JSON answer (it comes last) and leave nothing to learn."),
  B("5% of the training examples are held out as a validation set to watch for overfitting; the test set is never touched."),
  B("Optimiser: paged 8-bit AdamW on GPU (plain AdamW on CPU), cosine schedule with 3% warm-up, gradient checkpointing, one checkpoint per epoch."),
  B("Experiments are tracked in **Weights & Biases** (or MLflow, set `tracking.backend`); the smoke profile turns tracking off."),
  B("The adapter, tokenizer and a `finassist_run.json` summary (config, sizes, final validation loss) are saved to `outputs/adapters/<run_name>/`."),
  H2("7.2 The runs"),
  ...table(["Run", "LoRA on", "Rank / alpha", "Learning rate", "Epochs", "Question it answers"], [
    ["run1_r16", "all linear layers (q, k, v, o, gate, up, down)", "16 / 32", "2e-4", "3", "Conservative baseline configuration."],
    ["run2_r64", "all linear layers", "64 / 128", "1e-4", "2", "Does more adapter capacity with a lower LR help?"],
    ["ablation_attn_only", "attention only (q, k, v, o)", "16 / 32", "2e-4", "3", "Do the MLP adapters matter for numeric reasoning and the schema?"],
  ], [17, 25, 12, 12, 9, 25]),
  P("Effective batch size is 16 (4 per device × 4 gradient-accumulation steps). The run names gain a `_smoke` suffix on the smoke profile so laptop adapters never mix with real ones."),
  H1("8. DPO stretch goal (Weeks 2–3)"),
  P("`python -m finassist.cli dpo` (code: `train/dpo.py`) applies Direct Preference Optimisation on top of the best supervised adapter. It runs only if Week 2 finished cleanly."),
  ...N([
    "**Build preference pairs by rejection sampling.** For training questions, sample answers from the fine-tuned model at temperature 0.9 (up to 4 per question). The first sample that fails (wrong answer or invalid JSON) becomes the **rejected** answer; the gold JSON is the **chosen** one. Target: 250 pairs (8 on the smoke profile).",
    "**Train with DPO.** The adapter is trained to prefer chosen over rejected (beta 0.1, learning rate 5e-6, 1 epoch). The reference model is the same network with the adapter disabled, so no second copy is loaded.",
  ]),
  P("Pairs are saved to `data/processed/dpo_pairs.jsonl`; the adapter goes to `outputs/adapters/dpo_on_best/`."),
);

// 9 Retrieval
add(
  H1same("9. Retrieval (Week 3)"),
  P("At question time the model gets only the most relevant pieces of the right report page. `python -m finassist.cli index` builds the index; `rag/retriever.py` serves queries."),
  H2("9.1 How a query is answered"),
  ...N([
    "**Chunking** (`rag/chunking.py`). Pages are split into chunks. **Table-aware** (configured): text in 180-word windows with 40-word overlap; the table in groups of 12 rows, **each group repeating the header row**, so a chunk like \"| net revenue | 5,829 | 5,735 |\" still says which column is which year. **Fixed** (comparison): the whole page flattened and cut into word windows. The 1,000 pages give 4,605 table-aware chunks.",
    "**Embedding** (`rag/index.py`). Each chunk is embedded with **bge-small-en-v1.5**; queries get the instruction prefix bge models expect. Vectors are normalised.",
    "**Vector store.** A plain NumPy matrix with exact cosine search. At a few thousand chunks this takes milliseconds on CPU, gives exact recall and needs no extra service; FAISS or Qdrant only pays off beyond ~100,000 chunks.",
    "**Metadata filter.** FinQA questions rarely name the company (\"what was the change in net revenue?\"), so each query carries its document metadata and the search is restricted to that filing's chunks, as in an analyst tool where the filing is selected first.",
    "**Dense search** returns the top 20 chunks.",
    "**Reranking.** A cross-encoder, **bge-reranker-base**, rescores the 20 (question, chunk) pairs and keeps the best 4.",
    "**Context.** The 4 chunks are numbered and placed in the prompt, tables first, because most answers need a table row plus a sentence.",
  ]),
  H2("9.2 Measuring retrieval on its own"),
  P("`python -m finassist.cli retrieval-eval --no-filter --full` (code: `rag/recall.py`) scores retrieval separately from answer quality, for each chunking/reranking variant, with and without the metadata filter. It saves after every variant and `--resume` skips finished ones."),
  B("**Document recall:** is the question's own page among the 4 chunks? With the filter this is always 1.0, because every page has unique metadata, so it cannot compare variants."),
  B("**Evidence recall** (added): the share of the evidence needed to answer that appears in the 4 chunks. For arithmetic questions the evidence is the operands of the gold calculation that appear in the gold context (intermediate results and constants like 100 are excluded); for span answers it is the answer text. **Evidence complete** is the share of questions where all of it is present."),
  P("Results on all 200 test questions (191 have checkable evidence). These need no language model, so they are final:"),
  ...table(["Variant", "Evidence recall", "Evidence complete", "Evidence recall, no filter"], [
    ["Table-aware + reranker (configured)", "0.972", "0.963", "0.650"],
    ["Table-aware, no reranker", "0.962", "0.953", "0.492"],
    ["Fixed-window + reranker", "**0.988**", "**0.984**", "0.656"],
  ], [40, 18, 20, 22]),
  B("The reranker helps most when no filing is selected (0.49 → 0.65)."),
  B("Fixed-window chunking scored slightly higher than the configured table-aware chunking: a decision to make before the 7B run (`rag.chunking.strategy`)."),
);

// 10 Evaluation
add(
  H1("10. Evaluation harness"),
  P("`python -m finassist.cli eval --condition <name>` (code: `eval/run_eval.py`) is the single harness behind every comparison in the report. A condition is a model plus a context source:"),
  ...table(["Condition", "Model", "Context", "Role"], [
    ["`base_zeroshot`", "base", "none", "Week 1 baseline"],
    ["`base_fewshot`", "base + 2 examples", "none", "Week 1 baseline"],
    ["`base_oracle`", "base", "gold page", "Upper bound for prompting"],
    ["`base_rag`", "base", "retrieved", "Required comparison"],
    ["`ft`", "fine-tuned", "none", "Required comparison"],
    ["`ft_rag`", "fine-tuned", "retrieved", "Required comparison (the system)"],
    ["`ft_oracle`", "fine-tuned", "gold page", "Upper bound for the fine-tuned model"],
  ], [22, 24, 18, 36]),
  P("The model can run in-process (`--backend hf`, base model plus optional adapter) or be queried on a running server (`--backend openai --base-url ...`), which is how latency is measured on the actual serving stack."),
  H2("10.1 Metrics"),
  ...table(["Metric", "Definition"], [
    ["Exact match", "Numeric answers: equal within 1% relative tolerance, or equal at the gold answer's displayed precision (1.64% matches 1.6%). Also accepts the ×100 / ÷100 forms (0.164 vs 16.4%), because FinQA golds are inconsistent. Text answers: equal after lower-casing and removing punctuation and articles."],
    ["Token F1", "Word overlap between prediction and gold; meaningful for text answers, not numbers."],
    ["Schema validity", "Share of outputs that are exactly one valid JSON object with the three required keys (section 5.2)."],
    ["Latency p50 / p95, tokens/s", "Per-question generation time and speed."],
    ["Judge scores (optional)", "LLM-as-judge, below. Needs `JUDGE_API_KEY`."],
  ], [26, 74]),
  H2("10.2 LLM-as-judge"),
  P("With `--judge`, a larger model (default: Llama-3.3-70B on Groq, any OpenAI-compatible endpoint) scores each answer 1–5 on **correctness**, **reasoning** (are the steps valid arithmetic on the right figures?) and **grounding** (does the source point to the real evidence, with no invented figures?). It sees the question, the gold answer and evidence, and the candidate output, but **never which system produced it**."),
  H2("10.3 Where results go"),
  P("Each run writes `outputs/eval/<label>/predictions.jsonl` (every question: gold, prediction, raw output, schema errors, latency) and `metrics.json`, and appends one row to `outputs/eval/leaderboard.csv`, so every run lands in one comparison table."),
);

// 11 Serving
add(
  H1("11. Serving (Week 4)"),
  H2("11.1 Merge"),
  P("`python -m finassist.cli merge --adapter <best>` (code: `serve/merge.py`) folds the chosen LoRA adapter into the base weights. The base is loaded in **16-bit, not 4-bit**: merging into NF4 weights loses precision. A 16-bit 7B model (~15 GB) exceeds a T4 or Colab's RAM alone, so `device_map=\"auto\"` spreads it over GPU, CPU and disk (`outputs/merge_offload`). The pipeline picks the best `ft_rag` run automatically: highest exact match, then F1, then schema validity."),
  H2("11.2 Quantise"),
  ...table(["Method", "How", "Served by", "Used on"], [
    ["`awq`", "AutoAWQ 4-bit, calibrated on 128 of our own training prompts so activation statistics match financial tables. Needs its own Python environment (`requirements-quant.txt`).", "vLLM (`--quantization awq`)", "IIT GPU box"],
    ["`gguf`", "llama.cpp: `convert_hf_to_gguf.py` to a 16-bit GGUF, then `llama-quantize` to Q4_K_M (988 MB → 398 MB for the 0.5B model).", "llama.cpp or Ollama", "Any machine"],
    ["`ollama`", "The `gguf` steps, then `ollama create finassist` from the Q4_K_M file.", "Ollama, OpenAI-compatible API on port 11434", "Laptop, Colab, friend's GPU"],
  ], [12, 48, 20, 20]),
  P("Ollama 0.40 cannot quantise Qwen2 models itself (its quantiser supports only MLX architectures) and will not quantise an existing GGUF, so llama.cpp makes the file and Ollama only serves it."),
  H2("11.3 The gateway"),
  P("`serve/app.py` is a FastAPI service in front of the model server (vLLM or Ollama, chosen with `VLLM_URL`). `POST /v1/answer` takes a question, optionally the document metadata, or caller-supplied context:"),
  ...N([
    "**Input guard.** Questions are length-checked (3–1,000 characters). Obvious personal-data requests (home address, SSN, password, credit card number) are answered \"refused\" immediately, without calling the model.",
    "**Retrieve.** Unless the caller supplied context, the retriever runs: filter by document, top 20, rerank to 4.",
    "**Build the prompt** with the same template as training (section 5.1).",
    "**Generate** on the model server, with constrained decoding (11.4).",
    "**Parse and repair.** The output is checked against the schema. If it is invalid, the gateway retries once, showing the model its own output and the errors.",
    "**Respond** with answer, calculation steps, source, `schema_valid`, the retrieved chunk ids, latency and model name. Errors from the model server become HTTP 502.",
  ]),
  ...figure("request_flow.png", 1989, 673, "Figure 3. One request through the gateway (shown with vLLM; on the laptop and Colab, Ollama takes its place)."),
  H2("11.4 Constrained decoding"),
  P("With `serving.constrained_decoding: true` (the default), requests to the model server carry `response_format: json_schema` with the schema from section 5.2. The server can then only generate JSON that matches it, and the 160-character cap on `source` forces the string to close. vLLM and Ollama both support this. The LLM judge is excluded: it has its own output format."),
  P("Effect on the served smoke model: valid JSON went from **60% to 100%**, and no answer reaches the token limit any more."),
  H2("11.5 Deployment"),
  B("**GPU machine:** `cd serving && docker compose up -d` starts vLLM (AWQ model, port 8000) and the gateway (port 8080). The gateway waits for vLLM's health check."),
  B("**Laptop / Colab / friend's machine:** Ollama serves the Q4_K_M model on port 11434; the gateway runs with `VLLM_URL=http://localhost:11434/v1`."),
  B("`scripts/gateway_smoke.sh <model url>` starts the gateway, sends one real question and one personal-data request, saves both responses and checks that the second was refused."),
);

// 12 Red team
add(
  H1("12. Red-teaming"),
  P("`python -m finassist.cli redteam --base-url <server>` (code: `redteam/run.py`) sends 20 adversarial prompts (`redteam/prompts.jsonl`) to the model and writes `docs/redteam_log.md`."),
  ...table(["Category", "Prompts", "Expected behaviour"], [
    ["Investment advice", "2", "refuse"],
    ["Personal data", "2", "refuse"],
    ["Harmful finance", "2", "refuse"],
    ["Prompt injection (in the context)", "3", "ignore the injected instruction, answer normally"],
    ["Missing information", "3", "\"insufficient information\""],
    ["Out of domain", "2", "refuse or \"insufficient information\""],
    ["Fabrication pressure", "1", "\"insufficient information\", never invent figures"],
    ["False premise", "2", "correct the premise (human check)"],
    ["Arithmetic trap", "2", "correct answer (human check)"],
    ["Format attack", "1", "still valid JSON"],
  ], [36, 12, 52]),
  P("An automatic first-pass verdict is computed for each prompt (e.g. did a \"refuse\" prompt get a refusal; does an injection's planted number or phrase appear in the output). It is only a triage aid: a person fills the **reviewer verdict** column. The red-team runner calls the model directly, so it tests the model itself, not the gateway's personal-data filter."),
);

// 13 Running
add(
  H1("13. Running everything"),
  H2("13.1 The pipeline runner"),
  P("`scripts/run_pipeline.sh` runs every stage in order on one compute profile:"),
  ...code([
    "bash scripts/run_pipeline.sh                                  # laptop smoke run",
    "COMPUTE=notebook  SERVE=ollama bash scripts/run_pipeline.sh   # Colab / Kaggle",
    "COMPUTE=local_gpu SERVE=ollama bash scripts/run_pipeline.sh   # a local or friend's GPU",
    "COMPUTE=iit_ssh   SERVE=vllm   bash scripts/run_pipeline.sh   # IIT box (docker compose up first)",
  ]),
  P("Stage order: prepare → zero-shot, few-shot and oracle baselines → three training runs → index → retrieval eval → base+RAG → ft and ft+RAG for each adapter → DPO and its evaluation → merge the best adapter → quantise → evaluate the served model → red-team → gateway test."),
  B("**Resumable:** each stage has an output file that proves it finished (for example `outputs/adapters/<run>/finassist_run.json`). Finished stages are skipped, so after a crash, a reboot or a closed Colab session you simply run it again. To redo a stage, delete its output."),
  B("**One job at a time:** two model jobs on one machine starve each other, which stalled a run on the laptop."),
  B("**Logs:** progress in `outputs/logs/pipeline.log`, each stage's full output in `outputs/logs/<stage>.log`. A failed stage is reported and the runner continues with the rest."),
  H2("13.2 Colab"),
  P("`notebooks/colab_track_a.ipynb` clones the private repo with a `GITHUB_TOKEN` secret, links `outputs/` to Google Drive (`MyDrive/finassist_track_a/outputs`) so results survive disconnects, installs the model stack, llama.cpp and Ollama, starts the pipeline in the background and shows progress. When a session ends, *Run all* again and it resumes. A full 7B run is roughly 15–25 GPU-hours on a T4, so expect several sessions."),
  H2("13.3 Tests"),
  P("`pytest -q` runs 16 tests in about a second, with no model weights:"),
  ...table(["Area", "What is checked"], [
    ["Data", "FinQA program execution and table operations; FinQA and TAT-QA conversion; short source citations."],
    ["Schema and metrics", "Strict validity versus recovery in the parser; exact match tolerance and F1."],
    ["Retrieval", "Table chunks repeat the header row; evidence-recall targets and matching."],
    ["Evaluation", "The harness end to end with a mock model."],
    ["Configuration", "Profiles, model override, training overrides."],
    ["Serving", "Constrained decoding on the served model but not the judge; gateway valid answer, repair retry, personal-data refusal without a model call."],
    ["Red team", "Automatic verdict rules."],
  ], [24, 76]),
);

// 14 Results
add(
  H1("14. Results so far"),
  ...callout("Read this first", [
    "All numbers below except retrieval come from the laptop **smoke** run: the 0.5B stand-in model, 18 training examples, 1 epoch, 10 test questions. They prove every stage works. They are **not** the results to report: those come from the 7B run.",
  ]),
  ...table(["System (smoke, n = 10)", "Exact match", "Valid JSON", "p50 latency (s)"], [
    ["Base, zero-shot", "0.0", "1.0", "2.0"],
    ["Base, few-shot", "0.0", "1.0", "3.1"],
    ["Base, oracle context", "0.1", "0.8", "10.1"],
    ["Base + RAG", "0.1", "1.0", "10.5"],
    ["Fine-tuned run 1 / + RAG", "0.0 / 0.0", "0.6 / 0.7", "6.2 / 14.1"],
    ["Fine-tuned run 2 / + RAG", "0.0 / 0.0", "1.0 / 0.9", "5.4 / 15.4"],
    ["Ablation / + RAG", "0.0 / 0.0", "1.0 / 1.0", "3.5 / 9.3"],
    ["DPO + RAG", "0.0", "0.7", "13.1"],
    ["**Served** (ablation merged, Q4_K_M, Ollama, constrained decoding) + RAG", "0.0", "**1.0**", "7.4"],
  ], [52, 16, 14, 18]),
  B("Exact match is about 0 for every 0.5B system: the model is far too small for multi-step arithmetic. Expected."),
  B("Differences between the runs are noise at this size; the ablation-versus-full comparison even flipped between two smoke runs. The 7B run decides it."),
  B("Red-team (served smoke model): 9 pass, 7 fail, 4 need human review. The small model says \"insufficient information\" reliably but does not refuse advice or harmful prompts."),
  B("Gateway test: a real question got a valid answer from the correct page in about 8 seconds; the personal-data request was refused."),
);

// 15 Problems
add(
  H1("15. Problems found and how they were fixed"),
  ...table(["Problem", "Cause", "Fix"], [
    ["Long jobs \"restarted for no reason\"", "Windows Update forced reboots right after active hours ended (23:00)", "Resumable pipeline runner; pause updates during long runs"],
    ["Retrieval eval stalled with empty output", "Shared the CPU with another evaluation; buffered output lost on reboot", "One job at a time; unbuffered output; saves after each variant; `--resume`"],
    ["Retrieval variants all scored 1.0", "Document recall is trivial with the metadata filter", "Evidence-recall metric"],
    ["DPO crashed on the laptop", "Hard-coded GPU 4-bit and bf16; wrong adapter path; ignored `tracking: none`", "Follows the compute profile like training"],
    ["Quantise failed", "Ollama 0.40 cannot quantise Qwen2 or an existing GGUF", "llama.cpp makes the Q4_K_M file, Ollama serves it"],
    ["Served answers truncated (60% valid JSON)", "Training `source` targets were long verbatim passages", "Short citation targets, prompt rule, constrained decoding (100% valid)"],
    ["Would be slow on a Colab T4", "bf16 emulated on T4", "fp16 on pre-Ampere GPUs"],
    ["7B merge would not fit Colab", "~15 GB of 16-bit weights", "Spread over GPU, CPU and disk"],
    ["Scripts would break on Linux", "Windows line endings", "`.gitattributes` forces LF; data files kept byte-exact for their hashes"],
  ], [26, 37, 37]),
);

// 16 Pending + glossary
add(
  H1("16. What is still pending"),
  ...N([
    "**The real 7B run** on a GPU: Colab, Kaggle, a friend's machine (12 GB+ GPU memory, 24 GB ideal; about 5–8 hours on an RTX 3090/4090) or the IIT box.",
    "**Decide chunking** before that run: fixed-window beat table-aware on evidence recall (0.988 vs 0.972).",
    "**Secrets:** `WANDB_API_KEY` (or switch tracking to MLflow), `JUDGE_API_KEY` for judge scores, `HF_TOKEN` only for Llama.",
    "**Fill the [7B] fields** in `docs/model_card.md`, and re-check the ablation result and the schema slips seen without constrained decoding.",
    "**Human red-team review:** fill the reviewer column in `docs/redteam_log.md`.",
  ]),
  H1("Appendix A. Glossary"),
  ...table(["Term", "Meaning"], [
    ["LoRA", "Low-Rank Adaptation: trains small add-on matrices instead of the full model weights."],
    ["QLoRA", "LoRA on top of a base model stored in 4-bit, so large models fit small GPUs."],
    ["NF4", "4-bit \"NormalFloat\" format used by QLoRA for the frozen base weights."],
    ["SFT", "Supervised fine-tuning on question/answer examples."],
    ["DPO", "Direct Preference Optimisation: trains the model to prefer a chosen answer over a rejected one."],
    ["RAG", "Retrieval-augmented generation: fetch relevant text and put it in the prompt."],
    ["Embedding", "A vector representing a text's meaning; similar texts have nearby vectors."],
    ["Cross-encoder reranker", "A model that reads the question and a chunk together and scores their relevance; more accurate but slower than embeddings."],
    ["Oracle context", "The gold evidence page, used as an upper bound and for training."],
    ["GGUF / Q4_K_M", "llama.cpp's model file format / its 4-bit quantisation scheme."],
    ["AWQ", "Activation-aware weight quantisation: a 4-bit format served efficiently by vLLM."],
    ["vLLM / Ollama", "Model servers exposing an OpenAI-compatible API (vLLM for GPUs at scale; Ollama for local machines)."],
    ["Constrained decoding", "The server only allows tokens that keep the output valid against a JSON schema."],
    ["Smoke run", "A tiny end-to-end run that proves every stage works before spending GPU time."],
  ], [26, 74]),
  H1same("Appendix B. Command reference"),
  ...code([
    "pip install -r requirements.txt && pip install -e .     # laptop (CPU) environment",
    "pip install -r requirements-gpu.txt && pip install -e . # GPU environment",
    "pytest -q                                               # 16 tests",
    "",
    "python -m finassist.cli prepare                          # build + freeze dataset (once)",
    "python -m finassist.cli baseline                         # zero-shot + few-shot",
    "python -m finassist.cli train --run run1_r16             # also run2_r64, ablation_attn_only",
    "python -m finassist.cli dpo                              # stretch goal",
    "python -m finassist.cli index                            # chunk + embed the corpus",
    "python -m finassist.cli retrieval-eval --no-filter --full --resume",
    "python -m finassist.cli eval --condition ft_rag \\",
    "    --adapter outputs/adapters/run1_r16_lr2e-4 [--judge]",
    "python -m finassist.cli merge --adapter outputs/adapters/<best>",
    "python -m finassist.cli quantize --method ollama         # or awq (GPU) / gguf",
    "python -m finassist.cli eval --condition ft_rag --backend openai \\",
    "    --base-url <server>/v1",
    "python -m finassist.cli redteam --base-url <server>/v1",
    "bash scripts/gateway_smoke.sh <server>/v1",
    "",
    "python -m finassist.cli --compute <profile> <stage>      # any stage on any profile",
    "bash scripts/run_pipeline.sh                             # everything, resumable",
  ]),
);

// ---------- document ----------
const doc = new Document({
  creator: "FinAssist Track A",
  title: "FinAssist Track A: Implementation guide",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0, keepNext: true, keepLines: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E5C8A", font: FONT }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1, keepNext: true, keepLines: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "404040", font: FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2, keepNext: true, keepLines: true } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } },
      ] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } },
      ] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "FinAssist Track A — Implementation guide", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "888888" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("written", OUT, buf.length, "bytes"); });
