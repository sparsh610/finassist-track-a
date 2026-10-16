"""Chunking strategies for report pages (text + one financial table).

table_aware (default): text is split into overlapping word windows; the table is
    split into row groups and every group repeats the header row, so a chunk like
    "| net revenue | 5,829 | 5,735 |" still says which column is which year.
fixed: the whole page flattened to text and cut into word windows, tables
    included. Kept as the comparison point for the chunking decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from finassist.data.prepare import table_to_markdown


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_meta: str
    kind: str  # text | table
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _windows(words: list[str], size: int, overlap: int) -> list[str]:
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def chunk_document(doc: dict, cfg: dict) -> list[Chunk]:
    ccfg = cfg["rag"]["chunking"]
    size, overlap = ccfg["text_chunk_words"], ccfg["text_overlap_words"]
    chunks: list[Chunk] = []

    def add(kind: str, text: str) -> None:
        chunks.append(Chunk(f"{doc['doc_id']}#{len(chunks)}", doc["doc_id"], doc["meta"], kind, text))

    if ccfg["strategy"] == "fixed":
        flat = " ".join(doc["pre_text"]) + "\n" + table_to_markdown(doc["table"]) + "\n" + " ".join(doc["post_text"])
        for w in _windows(flat.split(), size, overlap):
            add("text", w)
        return chunks

    for part in (doc["pre_text"], doc["post_text"]):
        for w in _windows(" ".join(part).split(), size, overlap):
            add("text", w)
    table = [r for r in doc["table"] if any(str(c).strip() for c in r)]
    if table:
        header, rows = table[0], table[1:]
        per = ccfg["table_rows_per_chunk"]
        for i in range(0, max(1, len(rows)), per):
            add("table", "Table:\n" + table_to_markdown([header] + rows[i : i + per]))
    return chunks


def chunk_corpus(docs: list[dict], cfg: dict) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, cfg))
    return out
