"""Render the architecture diagrams used in the implementation document.

    python docs/diagrams/make_diagrams.py      -> docs/diagrams/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent
INK, QUIET, EDGE = "#1f2933", "#52606d", "#7b8794"
FILL = {
    "data": ("#e3eefb", "#2b6cb0"),
    "model": ("#e6f6f1", "#2c7a64"),
    "rag": ("#f3ecfb", "#6b46c1"),
    "eval": ("#fdf3e3", "#b7791f"),
    "serve": ("#fde8e8", "#c53030"),
    "plain": ("#f5f7fa", "#7b8794"),
}


def box(ax, x, y, w, h, title, lines=(), kind="plain", title_size=10.5):
    face, edge = FILL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=face, ec=edge, lw=1.4))
    top = y + h - 0.28
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=title_size, weight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.32 - 0.27 * i, line, ha="center", va="center", fontsize=8.6, color=QUIET)
    return (x, y, w, h)


def arrow(ax, p, q, label=None, style="-|>", color=EDGE, rad=0.0, lpos=0.5, loff=(0, 0.15)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        mx, my = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.2, color=QUIET, style="italic")


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(0.2, h - 0.3, title, fontsize=13, weight="bold", color=INK, va="center")
    return fig, ax


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top_(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def legend(ax, x, y):
    for i, (k, name) in enumerate([("data", "Data"), ("model", "Model adaptation"), ("rag", "Retrieval"),
                                   ("eval", "Evaluation"), ("serve", "Serving")]):
        face, edge = FILL[k]
        ax.add_patch(FancyBboxPatch((x + i * 2.25, y), 0.3, 0.22, boxstyle="round,pad=0.01", fc=face, ec=edge))
        ax.text(x + i * 2.25 + 0.42, y + 0.11, name, va="center", fontsize=8.5, color=QUIET)


def architecture():
    fig, ax = canvas(14, 8.4, "FinAssist (Track A): end-to-end architecture")
    # Row 1: data
    src = box(ax, 0.3, 6.2, 2.6, 1.3, "Public datasets", ["FinQA (10-K pages)", "TAT-QA (report tables)"], "data")
    prep = box(ax, 3.6, 6.2, 3.0, 1.3, "Week 1: prepare", ["convert to JSON schema", "split, filter, freeze (sha256)"], "data")
    frozen = box(ax, 7.3, 6.2, 2.7, 1.3, "Frozen data", ["train 1,500 · test 200", "corpus 1,000 pages"], "data")
    arrow(ax, right(src), left(prep))
    arrow(ax, right(prep), left(frozen))

    # Row 2: adaptation, evaluation, retrieval
    base = box(ax, 0.3, 3.6, 2.6, 1.5, "Base model", ["Qwen2.5-7B-Instruct", "or Llama-3.1-8B", "4-bit NF4"], "model")
    qlora = box(ax, 3.6, 3.6, 3.0, 1.5, "Week 2: QLoRA SFT", ["run 1: r16, lr 2e-4", "run 2: r64, lr 1e-4", "ablation: attention only"], "model")
    ev = box(ax, 7.3, 2.6, 2.7, 2.5, "Evaluation harness", ["7 conditions, 200 test Qs", "exact match, F1, schema", "LLM-as-judge rubric", "latency, tokens/s", "-> leaderboard.csv"], "eval")
    rag = box(ax, 10.9, 3.6, 2.8, 1.5, "Week 3: RAG", ["table-aware chunks, bge-small", "filter by filing, top-20", "bge-reranker -> top-4"], "rag")
    arrow(ax, right(base), left(qlora))
    arrow(ax, (frozen[0] + 0.4, frozen[1]), (qlora[0] + qlora[2] - 0.3, qlora[1] + qlora[3]), "train", lpos=0.45, loff=(-0.3, -0.05))
    arrow(ax, (frozen[0] + 1.9, frozen[1]), (frozen[0] + 1.9, ev[1] + ev[3]), "test", lpos=0.45, loff=(0.3, -0.1))
    arrow(ax, right(frozen), top_(rag), "corpus", lpos=0.45, loff=(0.25, 0.05))
    arrow(ax, left(rag), (ev[0] + ev[2], rag[1] + rag[3] / 2), "context", loff=(0, 0.06))

    # Row 3: adapters and serving
    adapters = box(ax, 3.6, 1.7, 3.0, 1.2, "LoRA adapters", ["+ optional DPO stretch", "tracked in W&B / MLflow"], "model")
    arrow(ax, bottom(qlora), top_(adapters))
    arrow(ax, right(adapters), (ev[0], adapters[1] + adapters[3] / 2), "base + adapter", loff=(0, 0.06))
    mq = box(ax, 0.3, 0.2, 3.0, 1.1, "Week 4: merge + AWQ", ["best adapter -> 4-bit model"], "serve")
    vllm = box(ax, 7.3, 0.2, 2.7, 1.1, "vLLM (GPU)", ["OpenAI-compatible :8000"], "serve")
    gw = box(ax, 10.9, 0.2, 2.8, 1.1, "FastAPI gateway :8080", ["guard, RAG, JSON repair"], "serve")
    arrow(ax, (adapters[0] + 0.5, adapters[1]), (mq[0] + mq[2] - 0.5, mq[1] + mq[3]), "best run", lpos=0.5, loff=(-0.55, -0.1))
    arrow(ax, right(mq), left(vllm), "Docker", loff=(0, 0.06))
    arrow(ax, left(gw), right(vllm), "generate", loff=(0, 0.06))
    arrow(ax, bottom(rag), top_(gw), "retrieve", lpos=0.5, loff=(0.45, -0.1))
    legend(ax, 0.35, 7.75)
    fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def request_flow():
    fig, ax = canvas(14, 4.6, "Serving: one request through the gateway")
    y = 2.0
    c = box(ax, 0.2, y, 1.7, 1.2, "Client", ["question +", "document"], "plain")
    g = box(ax, 2.4, y, 1.9, 1.2, "Input guard", ["length, PII", "-> refused"], "serve")
    r = box(ax, 4.8, y, 2.1, 1.2, "Retrieve", ["filter by doc", "top-20 -> rerank 4"], "rag")
    p = box(ax, 7.4, y, 1.9, 1.2, "Build prompt", ["same template", "as training"], "model")
    v = box(ax, 9.8, y, 1.7, 1.2, "vLLM", ["AWQ model"], "serve")
    j = box(ax, 12.0, y, 1.8, 1.2, "Parse JSON", ["schema check"], "eval")
    for a_, b_ in [(c, g), (g, r), (r, p), (p, v), (v, j)]:
        arrow(ax, right(a_), left(b_))
    rep = box(ax, 9.8, 0.25, 1.7, 1.0, "Repair retry", ["once, with errors"], "eval")
    arrow(ax, bottom(j), right(rep), "invalid", rad=-0.3, lpos=0.5, loff=(0.35, -0.3))
    arrow(ax, left(rep), (p[0] + p[2] / 2, p[1]), rad=-0.3)
    out = box(ax, 12.0, 3.55, 1.8, 0.75, "Response", [], "plain")
    arrow(ax, top_(j), bottom(out), "valid", lpos=0.5, loff=(0.35, -0.08))
    ax.text(0.2, 0.45, "Response: answer, calculation_steps, source,\nschema_valid, retrieved chunk ids, latency_ms",
            fontsize=8.6, color=QUIET)
    fig.savefig(OUT / "request_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_profiles():
    fig, ax = canvas(12, 4.8, "Compute: one CLI, swappable back ends")
    cli = box(ax, 0.3, 1.7, 2.6, 1.4, "finassist.cli", ["stage + --compute", "profile"], "plain")
    run = box(ax, 3.6, 1.7, 2.4, 1.4, "compute.runner", ["kind: local | ssh"], "plain")
    arrow(ax, right(cli), left(run))
    loc = box(ax, 7.0, 3.0, 4.6, 1.15, "local", ["smoke (CPU, 0.5B)  ·  notebook (Kaggle/Colab T4)", "local_gpu"], "model")
    ssh = box(ax, 7.0, 0.5, 4.6, 1.6, "iit_ssh (later)", ["1. push code + frozen data as tar over ssh", "2. run the same module on the GPU box",
                                                       "3. pull outputs/ back"], "serve")
    arrow(ax, (run[0] + run[2], run[1] + 1.1), left(loc), "today")
    arrow(ax, (run[0] + run[2], run[1] + 0.3), left(ssh))
    ax.text(5.0, 1.25, "later: set active: iit_ssh", ha="center", fontsize=8.2, color=QUIET, style="italic")
    fig.savefig(OUT / "compute_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def timeline():
    fig, ax = canvas(13, 4.9, "Four-week plan with the Week 1 hard gate")
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    x0, wk = 3.2, 2.35
    for i, w in enumerate(weeks):
        ax.text(x0 + wk * i + wk / 2, 4.05, w, ha="center", fontsize=10, weight="bold", color=INK)
        ax.plot([x0 + wk * i] * 2, [0.3, 3.85], color="#d9e2ec", lw=1)
    ax.plot([x0 + wk * 4] * 2, [0.3, 3.85], color="#d9e2ec", lw=1)
    tasks = [
        ("Dataset + freeze", 0, 0.75, "data"),
        ("Baselines", 0.45, 1.0, "eval"),
        ("QLoRA runs + ablation", 1.0, 2.0, "model"),
        ("DPO (stretch)", 1.6, 2.4, "model"),
        ("RAG index + recall@k", 1.8, 3.0, "rag"),
        ("Evaluation + judge", 2.0, 3.0, "eval"),
        ("Merge, AWQ, vLLM, Docker", 3.0, 3.7, "serve"),
        ("Model card, red team, report", 3.2, 4.0, "plain"),
    ]
    for row, (name, s, e, kind) in enumerate(tasks):
        y = 3.5 - row * 0.42
        face, edge = FILL[kind]
        ax.add_patch(FancyBboxPatch((x0 + s * wk, y - 0.13), (e - s) * wk, 0.26,
                                    boxstyle="round,pad=0.0,rounding_size=0.1", fc=face, ec=edge, lw=1.2))
        ax.text(x0 - 0.15, y, name, ha="right", va="center", fontsize=9, color=INK)
    gx = x0 + wk
    ax.plot([gx, gx], [0.3, 3.85], color="#c53030", lw=1.6, ls="--")
    ax.text(gx + 0.08, 0.38, "Gate: baseline recorded,\ndataset frozen", fontsize=8.5, color="#c53030")
    fig.savefig(OUT / "timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    request_flow()
    compute_profiles()
    timeline()
    print("diagrams ->", OUT)
