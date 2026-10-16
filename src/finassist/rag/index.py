"""Build and load the dense vector index.

Vector store decision: at <=1,000 pages the corpus is a few thousand chunks, so an
exact cosine search over a normalised numpy matrix is fast (~ms on CPU), has no
native dependency and gives exact recall. Swap in FAISS/Qdrant only if the corpus
grows past ~100k chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from finassist.config import load_config, resolve
from finassist.data.prepare import read_jsonl
from finassist.rag.chunking import Chunk, chunk_corpus

# bge models expect this prefix on queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query and "bge" in self.model_name.lower():
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=not query)
        return np.asarray(vecs, dtype=np.float32)


def build_index(cfg: dict) -> Path:
    docs = read_jsonl(resolve(cfg["rag"]["corpus_dir"]) / "documents.jsonl")
    chunks = chunk_corpus(docs, cfg)
    device = "cuda" if cfg["compute"].get("device") == "cuda" else "cpu"
    emb = Embedder(cfg["rag"]["embedding_model"], device=device)
    vectors = emb.encode([c.text for c in chunks])
    out = resolve(cfg["rag"]["index_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (out / "index_info.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "chunks": len(chunks),
                "embedding_model": cfg["rag"]["embedding_model"],
                "chunking": cfg["rag"]["chunking"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"indexed {len(chunks)} chunks from {len(docs)} documents -> {out}")
    return out


def load_index(cfg: dict) -> tuple[list[Chunk], np.ndarray]:
    out = resolve(cfg["rag"]["index_dir"])
    chunks = [Chunk(**r) for r in read_jsonl(out / "chunks.jsonl")]
    return chunks, np.load(out / "vectors.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute")
    build_index(load_config(compute=ap.parse_args().compute))


if __name__ == "__main__":
    main()
