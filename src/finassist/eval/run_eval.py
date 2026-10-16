"""One harness for every comparison the brief asks for.

Conditions (each = model x context source):
    base_zeroshot     base model, no context              (Week 1 baseline)
    base_fewshot      base model, no context, 2 examples  (Week 1 baseline)
    base_oracle       base model, gold page as context    (upper bound for prompting)
    base_rag          base model + retrieved context      (required comparison)
    ft                fine-tuned, no context              (required comparison)
    ft_rag            fine-tuned + retrieved context      (required comparison)
    ft_oracle         fine-tuned, gold page as context

Writes outputs/eval/<label>/predictions.jsonl and metrics.json, and appends a row
to outputs/eval/leaderboard.csv so every run lands in one comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from finassist.config import load_config, resolve
from finassist.data.prepare import load_split
from finassist.eval import judge as judge_mod
from finassist.eval.metrics import exact_match, token_f1
from finassist.llm.backends import make_backend
from finassist.prompts import build_messages
from finassist.schema import parse_output

CONDITIONS = {
    # name: (uses_adapter, context_mode, few_shot)
    "base_zeroshot": (False, "none", False),
    "base_fewshot": (False, "none", True),
    "base_oracle": (False, "oracle", False),
    "base_rag": (False, "rag", False),
    "ft": (True, "none", False),
    "ft_rag": (True, "rag", False),
    "ft_oracle": (True, "oracle", False),
}


def evaluate(
    examples: list[dict],
    backend,
    context_mode: str = "none",
    few_shot: bool = False,
    retriever=None,
    judge_backend=None,
) -> tuple[list[dict], dict]:
    rows = []
    for ex in examples:
        if context_mode == "oracle":
            context = ex["context"]
        elif context_mode == "rag":
            context = retriever.format_context(retriever.search(ex["question"], ex["doc_meta"]))
        else:
            context = None
        messages = build_messages(ex["question"], context, ex["doc_meta"], few_shot=few_shot)
        gen = backend.generate(messages)
        parsed = parse_output(gen.text)
        pred = parsed.answer
        row = {
            "id": ex["id"],
            "dataset": ex["dataset"],
            "question": ex["question"],
            "gold": ex["answer"],
            "pred": pred,
            "raw": gen.text,
            "schema_valid": parsed.valid,
            "schema_errors": parsed.errors,
            "exact_match": exact_match(pred, ex["answer"]) if pred else False,
            "f1": token_f1(pred, ex["answer"]) if pred else 0.0,
            "latency_s": gen.latency_s,
            "completion_tokens": gen.completion_tokens,
            "tokens_per_s": gen.tokens_per_s,
        }
        if judge_backend is not None:
            row["judge"] = judge_mod.judge_one(judge_backend, ex, gen.text)
        rows.append(row)

    n = max(1, len(rows))
    lat = [r["latency_s"] for r in rows] or [0.0]
    metrics = {
        "n": len(rows),
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "schema_validity": sum(r["schema_valid"] for r in rows) / n,
        "latency_p50_s": statistics.median(lat),
        "latency_p95_s": sorted(lat)[int(0.95 * (len(lat) - 1))],
        "tokens_per_s_mean": sum(r["tokens_per_s"] for r in rows) / n,
    }
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        metrics[f"exact_match_{ds}"] = sum(r["exact_match"] for r in sub) / len(sub)
    if judge_backend is not None:
        metrics.update(judge_mod.summarize([r["judge"] for r in rows]))
    return rows, metrics


def write_results(label: str, rows: list[dict], metrics: dict, meta: dict) -> Path:
    out = resolve("outputs/eval") / label
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps({**meta, **metrics}, indent=2), encoding="utf-8")

    board = resolve("outputs/eval/leaderboard.csv")
    fields = ["timestamp", "label", "condition", "model", "adapter", "compute", "n", "exact_match", "f1",
              "schema_validity", "latency_p50_s", "tokens_per_s_mean", "judge_correctness"]
    new = not board.exists()
    with open(board, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**meta, **metrics, "timestamp": time.strftime("%Y-%m-%d %H:%M")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--adapter", help="outputs/adapters/<run_name> (required for ft* conditions)")
    ap.add_argument("--backend", default="hf", choices=["hf", "openai"],
                    help="openai = query a running vLLM/Ollama endpoint instead of loading weights")
    ap.add_argument("--base-url")
    ap.add_argument("--served-model")
    ap.add_argument("--judge", action="store_true", help="also run the LLM-as-judge rubric")
    ap.add_argument("--compute")
    ap.add_argument("--label")
    args = ap.parse_args()

    cfg = load_config(compute=args.compute)
    uses_adapter, context_mode, few_shot = CONDITIONS[args.condition]
    if uses_adapter and not args.adapter and args.backend == "hf":
        ap.error(f"{args.condition} needs --adapter")

    overrides = {}
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.served_model:
        overrides["model"] = args.served_model
    backend = make_backend(cfg, args.backend, adapter_dir=args.adapter if uses_adapter else None, **overrides)

    retriever = None
    if context_mode == "rag":
        from finassist.rag.retriever import Retriever

        retriever = Retriever(cfg)

    judge_backend = None
    if args.judge:
        j = cfg["judge"]
        judge_backend = make_backend(cfg, "openai", base_url=j["base_url"], model=j["model"],
                                     api_key_env=j["api_key_env"], max_new_tokens=200,
                                     response_schema=None)  # the judge has its own output format

    examples = load_split(cfg, "test")
    rows, metrics = evaluate(examples, backend, context_mode, few_shot, retriever, judge_backend)
    label = args.label or args.condition + (f"__{Path(args.adapter).name}" if args.adapter and uses_adapter else "")
    meta = {
        "label": label,
        "condition": args.condition,
        "model": cfg["model"]["id"],
        "adapter": args.adapter or "",
        "compute": cfg["compute"]["name"],
        "backend": backend.name,
    }
    out = write_results(label, rows, metrics, meta)
    print(json.dumps(metrics, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
