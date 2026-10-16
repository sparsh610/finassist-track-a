"""Week 1 dataset build: download FinQA + TAT-QA, convert to the fixed schema,
sample, split, write the RAG corpus and freeze everything with a hash manifest.

Outputs (under data/processed):
    train.jsonl       ~1,500 instruction-response pairs
    test.jsonl        200 held-out pairs, never trained on
    manifest.json     counts + sha256 of each file; its presence = dataset frozen
and data/corpus/documents.jsonl (the 200-1,000 report pages RAG searches).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from finassist.config import load_config, resolve
from finassist.data.finqa_program import ProgramError, fmt, run_program
from finassist.prompts import build_messages
from finassist.schema import dump_target


# ---------- shared helpers ----------

def table_to_markdown(table: list[list[str]]) -> str:
    rows = [[str(c).strip().replace("|", "/") for c in row] for row in table if any(str(c).strip() for c in row)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return dest


def build_context(table_md: str, gold_text: list[str], other_text: list[str], limit: int) -> str:
    """Table first, then the evidence sentences, then filler sentences up to `limit` chars."""
    parts = [f"Table:\n{table_md}"] if table_md else []
    body = list(dict.fromkeys(gold_text))
    used = len(parts[0]) if parts else 0
    used += sum(len(s) for s in body)
    for s in other_text:
        if s in body:
            continue
        if used + len(s) > limit:
            break
        body.append(s)
        used += len(s)
    if body:
        parts.append("Text:\n" + "\n".join(body))
    return "\n\n".join(parts)[: limit + 2000]


# ---------- source citations ----------

SOURCE_TARGET_CHARS = 100  # well under schema.SOURCE_MAX_CHARS


def _first_words(text: str, n: int = 10) -> str:
    words = str(text).split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def cite(parts: list[str]) -> str:
    """Join citation parts into one short `source`, cut only between parts.

    Targets are a pointer to the evidence ("table row: net revenue"), not a copy
    of it: verbatim targets taught the model to paste whole passages and run out
    of tokens mid-string.
    """
    out = ""
    for part in dict.fromkeys(p for p in parts if p):
        candidate = f"{out}; {part}" if out else part
        if len(candidate) > SOURCE_TARGET_CHARS:
            break
        out = candidate
    return out or (parts[0][: SOURCE_TARGET_CHARS - 4] + " ..." if parts else "table")


# ---------- FinQA ----------

def finqa_doc_meta(filename: str) -> str:
    m = re.match(r"([^/]+)/(\d{4})/page_(\d+)", filename)
    if not m:
        return filename
    return f"{m.group(1)} annual report (10-K), fiscal {m.group(2)}, page {m.group(3)}"


def convert_finqa(records: list[dict], limit_chars: int) -> tuple[list[dict], dict[str, dict]]:
    examples, docs = [], {}
    for rec in records:
        qa = rec.get("qa") or {}
        if not qa.get("question") or not qa.get("program"):
            continue
        doc_id = "finqa:" + rec["filename"]
        table = rec.get("table", [])
        try:
            value, steps = run_program(qa["program"], table)
        except (ProgramError, ValueError, IndexError):
            continue
        answer = str(qa.get("answer") or "").strip() or fmt(value)
        gold = qa.get("gold_inds", {}) or {}
        gold_text = [v for k, v in gold.items() if k.startswith("text")]
        gold_table = [v for k, v in gold.items() if k.startswith("table")]
        parts = []
        for key in gold:
            kind, _, idx = key.partition("_")
            if kind == "table" and idx.isdigit() and int(idx) < len(table) and str(table[int(idx)][0]).strip():
                parts.append(f"table row: {str(table[int(idx)][0]).strip()}")
            elif kind == "text":
                parts.append(f"text: {_first_words(gold[key])}")
        source = cite(parts)
        all_text = rec.get("pre_text", []) + rec.get("post_text", [])
        context = build_context(table_to_markdown(table), gold_text, all_text, limit_chars)
        meta = finqa_doc_meta(rec["filename"])
        examples.append(
            {
                "id": "finqa:" + rec["id"],
                "dataset": "finqa",
                "doc_id": doc_id,
                "doc_meta": meta,
                "question": qa["question"],
                "context": context,
                "answer": answer,
                "answer_type": "arithmetic" if steps else "span",
                "calculation_steps": steps,
                "source": source,
            }
        )
        docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "meta": meta,
                "pre_text": rec.get("pre_text", []),
                "table": table,
                "post_text": rec.get("post_text", []),
            },
        )
    return examples, docs


# ---------- TAT-QA ----------

_SAFE_EXPR = re.compile(r"^[\d\s.+\-*/()%]+$")


def _tatqa_steps(derivation: str) -> list[str]:
    expr = derivation.replace(",", "").replace("$", "").strip()
    if not expr or not _SAFE_EXPR.match(expr):
        return [derivation] if derivation else []
    try:
        value = eval(expr.replace("%", "/100"), {"__builtins__": {}}, {})  # noqa: S307 - digits/operators only
    except Exception:
        return [derivation]
    return [f"{expr} = {fmt(value)}"]


def _tatqa_answer(answer, scale: str) -> str:
    if isinstance(answer, list):
        answer = ", ".join(str(a) for a in answer)
    answer = str(answer)
    if scale == "percent":
        return f"{answer}%"
    if scale:
        return f"{answer} {scale}"
    return answer


def convert_tatqa(records: list[dict], limit_chars: int) -> tuple[list[dict], dict[str, dict]]:
    examples, docs = [], {}
    for rec in records:
        table = rec["table"]["table"]
        doc_id = "tatqa:" + rec["table"]["uid"]
        paragraphs = sorted(rec.get("paragraphs", []), key=lambda p: p.get("order", 0))
        para_by_uid = {p["uid"]: p["text"] for p in paragraphs}
        para_by_order = {str(p.get("order")): p["text"] for p in paragraphs}
        meta = f"annual report table {rec['table']['uid'][:8]}"
        for q in rec.get("questions", []):
            if q.get("answer") in (None, "", []):
                continue
            rel = [para_by_uid.get(r) or para_by_order.get(str(r)) for r in q.get("rel_paragraphs", [])]
            rel = [r for r in rel if r]
            steps = _tatqa_steps(q.get("derivation", "")) if q.get("answer_type") == "arithmetic" else []
            source = cite([f"text: {_first_words(rel[0])}"]) if q.get("answer_from") == "text" and rel else "table"
            context = build_context(
                table_to_markdown(table), rel, [p["text"] for p in paragraphs], limit_chars
            )
            examples.append(
                {
                    "id": "tatqa:" + q["uid"],
                    "dataset": "tatqa",
                    "doc_id": doc_id,
                    "doc_meta": meta,
                    "question": q["question"],
                    "context": context,
                    "answer": _tatqa_answer(q["answer"], q.get("scale", "")),
                    "answer_type": q.get("answer_type", "span"),
                    "calculation_steps": steps,
                    "source": source,
                }
            )
        docs.setdefault(
            doc_id,
            {"doc_id": doc_id, "meta": meta, "pre_text": [p["text"] for p in paragraphs], "table": table, "post_text": []},
        )
    return examples, docs


# ---------- assembly ----------

def steps_consistent(ex: dict) -> bool:
    if not ex["calculation_steps"]:
        return True
    from finassist.eval.metrics import exact_match

    return exact_match(ex["calculation_steps"][-1].split("=")[-1], ex["answer"])


def to_record(ex: dict) -> dict:
    """Attach the chat messages + target so training/eval read one format."""
    ex = dict(ex)
    ex["target"] = dump_target(ex["answer"], ex["calculation_steps"], ex["source"])
    ex["messages"] = build_messages(ex["question"], ex["context"], ex["doc_meta"]) + [
        {"role": "assistant", "content": ex["target"]}
    ]
    return ex


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(cfg: dict, force: bool = False) -> dict:
    dcfg = cfg["data"]
    raw_dir, out_dir = resolve(dcfg["raw_dir"]), resolve(dcfg["processed_dir"])
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise SystemExit(
            f"{manifest_path} exists: the dataset is frozen. Re-run with --force only if the team agrees "
            "to unfreeze (the baseline must then be re-run)."
        )
    rng = random.Random(cfg["seed"])
    limit = dcfg["max_context_chars"]
    train, test, docs = [], [], {}

    converters = {"finqa": convert_finqa, "tatqa": convert_tatqa}
    for name, src in dcfg["sources"].items():
        convert = converters[name]
        tr_raw = json.loads(download(src["train_url"], raw_dir / f"{name}_train.json").read_text("utf-8"))
        te_raw = json.loads(download(src["test_url"], raw_dir / f"{name}_test.json").read_text("utf-8"))
        tr_ex, tr_docs = convert(tr_raw, limit)
        te_ex, te_docs = convert(te_raw, limit)
        rng.shuffle(te_ex)
        te_ex = te_ex[: src["n_test"]]
        test_doc_ids = {e["doc_id"] for e in te_ex}
        # No document appears in both splits, so the test set measures generalisation.
        tr_ex = [e for e in tr_ex if e["doc_id"] not in test_doc_ids]
        # Drop noisy training labels whose own calculation does not reproduce the
        # gold answer (~6%: sign flips, scale slips). The test set is left as is.
        tr_ex = [e for e in tr_ex if steps_consistent(e)]
        rng.shuffle(tr_ex)
        tr_ex = tr_ex[: src["n_train"]]
        train += tr_ex
        test += te_ex
        docs.update({k: v for k, v in te_docs.items() if k in test_doc_ids})
        docs.update({k: v for k, v in tr_docs.items() if k not in docs})
        print(f"{name}: {len(tr_ex)} train, {len(te_ex)} test")

    # Corpus = every test document + training documents as distractors, capped.
    test_doc_ids = {e["doc_id"] for e in test}
    corpus = [d for k, d in docs.items() if k in test_doc_ids]
    others = [d for k, d in docs.items() if k not in test_doc_ids]
    rng.shuffle(others)
    corpus += others[: max(0, cfg["rag"]["max_documents"] - len(corpus))]

    rng.shuffle(train)
    write_jsonl(out_dir / "train.jsonl", [to_record(e) for e in train])
    write_jsonl(out_dir / "test.jsonl", [to_record(e) for e in test])
    corpus_path = resolve(cfg["rag"]["corpus_dir"]) / "documents.jsonl"
    write_jsonl(corpus_path, corpus)

    manifest = {
        "seed": cfg["seed"],
        "counts": {"train": len(train), "test": len(test), "corpus_documents": len(corpus)},
        "sha256": {
            "train.jsonl": sha256(out_dir / "train.jsonl"),
            "test.jsonl": sha256(out_dir / "test.jsonl"),
            "documents.jsonl": sha256(corpus_path),
        },
        "sources": {k: {"train_url": v["train_url"], "test_url": v["test_url"]} for k, v in dcfg["sources"].items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"]))
    return manifest


def verify_frozen(cfg: dict) -> None:
    """Fail loudly if train/test changed after the Week 1 freeze."""
    out_dir = resolve(cfg["data"]["processed_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text("utf-8"))
    for name in ("train.jsonl", "test.jsonl"):
        if sha256(out_dir / name) != manifest["sha256"][name]:
            raise SystemExit(f"{name} changed since the freeze; results would not be comparable.")


def load_split(cfg: dict, split: str) -> list[dict]:
    rows = read_jsonl(resolve(cfg["data"]["processed_dir"]) / f"{split}.jsonl")
    limit = cfg["compute"].get(f"limit_{split}")
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    prepare(load_config(compute=args.compute), force=args.force)


if __name__ == "__main__":
    main()
