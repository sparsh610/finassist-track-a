"""Retrieve -> (optional metadata filter) -> rerank -> context string.

FinQA questions rarely name the company ("what was the change in net revenue?"),
so each query carries its document metadata (company, fiscal year). With
`filter_by_document_meta` the dense search is restricted to chunks whose
doc_meta matches; this mirrors a real analyst UI where the filing is selected first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from finassist.rag.chunking import Chunk
from finassist.rag.index import Embedder, load_index


@dataclass
class Hit:
    chunk: Chunk
    dense_score: float
    rerank_score: float | None = None


class Reranker:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


class Retriever:
    def __init__(self, cfg: dict, embedder=None, reranker=None, chunks=None, vectors=None):
        rcfg = cfg["rag"]
        device = "cuda" if cfg["compute"].get("device") == "cuda" else "cpu"
        if chunks is None:
            chunks, vectors = load_index(cfg)
        self.chunks, self.vectors = chunks, vectors
        self.embedder = embedder or Embedder(rcfg["embedding_model"], device=device)
        self.reranker = reranker if reranker is not None else (
            Reranker(rcfg["reranker_model"], device=device) if rcfg.get("reranker_model") else None
        )
        self.k_retrieve, self.k_rerank = rcfg["top_k_retrieve"], rcfg["top_k_rerank"]
        self.filter_meta = rcfg.get("filter_by_document_meta", False)
        self._meta = np.array([c.doc_meta for c in chunks])

    def search(self, question: str, doc_meta: str | None = None) -> list[Hit]:
        q = self.embedder.encode([question], query=True)[0]
        scores = self.vectors @ q
        if self.filter_meta and doc_meta:
            mask = self._meta == doc_meta
            if mask.any():
                scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[: self.k_retrieve]
        hits = [Hit(self.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]
        if self.reranker and hits:
            rr = self.reranker.score(question, [h.chunk.text for h in hits])
            for h, s in zip(hits, rr):
                h.rerank_score = s
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return hits[: self.k_rerank]

    @staticmethod
    def format_context(hits: list[Hit]) -> str:
        # Tables first: most FinQA answers need a table row plus a sentence.
        ordered = sorted(hits, key=lambda h: h.chunk.kind != "table")
        return "\n\n".join(f"[{i + 1}] {h.chunk.text}" for i, h in enumerate(ordered))


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(text: str) -> str:
    # "1,234" -> "1234", "$ 85" -> " 85", lower-case for span answers.
    return re.sub(r"(?<=\d),(?=\d{3})", "", text).replace("$", " ").lower()


def _num_in(num: str, text: str) -> bool:
    n = num.lstrip("-")
    pattern = rf"(?<![\d.]){re.escape(n)}(?:\.0+)?(?![\d])"
    # 93.21 also matches "93.210"; 85 must not match 850 or 8.5.
    return re.search(pattern, text) is not None


def evidence_targets(ex: dict) -> list[str]:
    """What a retrieved context must contain to answer `ex`.

    Arithmetic: the operands of the gold calculation that appear in the gold
    context (intermediate results and constants like 100 are dropped).
    Span answers: the answer text itself.
    """
    gold = _norm(ex.get("context", ""))
    steps = ex.get("calculation_steps") or []
    if steps:
        results, operands = set(), []
        for step in steps:
            lhs, _, rhs = step.partition("=")
            for n in _NUM.findall(_norm(lhs)):
                if n not in results and n not in operands and _num_in(n, gold):
                    operands.append(n)
            results.update(_NUM.findall(rhs))
        if operands:
            return operands
    answer = _norm(str(ex.get("answer", ""))).strip()
    return [a.strip() for a in answer.split(",") if a.strip() and a.strip() in gold]


def evidence_found(targets: list[str], text: str) -> float:
    if not targets:
        return float("nan")
    text = _norm(text)
    hit = [(_num_in(t, text) if _NUM.fullmatch(t) else t in text) for t in targets]
    return sum(hit) / len(hit)


def recall_at_k(retriever: Retriever, examples: list[dict]) -> dict:
    """Retrieval quality of the top-k hits that reach the prompt.

    doc_recall:         own document is among the hits (trivially 1.0 with the
                        metadata filter, since every page has unique metadata).
    evidence_recall:    mean share of the gold operands / answer span present in the hits.
    evidence_complete:  share of questions where *all* of them are present, i.e.
                        the context is sufficient to compute the answer.
    """
    doc, ev, complete, scored = 0, 0.0, 0, 0
    for ex in examples:
        hits = retriever.search(ex["question"], ex["doc_meta"])
        doc += any(h.chunk.doc_id == ex["doc_id"] for h in hits)
        share = evidence_found(evidence_targets(ex), "\n".join(h.chunk.text for h in hits))
        if share == share:  # not NaN
            scored += 1
            ev += share
            complete += share == 1.0
    n = max(1, len(examples))
    return {
        "doc_recall": round(doc / n, 4),
        "evidence_recall": round(ev / max(1, scored), 4),
        "evidence_complete": round(complete / max(1, scored), 4),
        "k": retriever.k_rerank,
        "n": len(examples),
        "n_with_evidence": scored,
    }
