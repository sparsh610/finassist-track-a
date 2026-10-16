"""Measure retrieval on its own, separately from answer quality.

Runs every chunking/reranking variant listed below against the frozen test set,
so the chunking and reranker decisions in the report are backed by numbers.
Results are written after every variant (so an interrupted run keeps what it
finished) and `--resume` skips variants already in the output file.
"""

from __future__ import annotations

import argparse
import copy
import json

from finassist.config import load_config, resolve
from finassist.data.prepare import load_split

VARIANTS = {
    "table_aware+rerank": {},
    "table_aware+no_rerank": {"reranker_model": None},
    "fixed+rerank": {"chunking": {"strategy": "fixed"}, "index_dir": "outputs/index_fixed"},
}


def main() -> None:
    from finassist.rag.index import Embedder, build_index
    from finassist.rag.retriever import Reranker, Retriever, recall_at_k

    ap = argparse.ArgumentParser()
    ap.add_argument("--compute")
    ap.add_argument("--no-filter", action="store_true", help="also report without the metadata filter")
    ap.add_argument("--resume", action="store_true", help="skip variants already in the output file")
    ap.add_argument("--full", action="store_true", help="ignore the profile's limit_test (retrieval is cheap)")
    args = ap.parse_args()
    base = load_config(compute=args.compute)
    if args.full:
        base["compute"].pop("limit_test", None)
    examples = load_split(base, "test")

    out = resolve(f"outputs/eval/retrieval_recall_{base['compute']['name']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8")) if args.resume and out.exists() else {}

    # One embedder / reranker for every variant: they only differ in index and flags.
    device = "cuda" if base["compute"].get("device") == "cuda" else "cpu"
    embedder = Embedder(base["rag"]["embedding_model"], device=device)
    reranker = Reranker(base["rag"]["reranker_model"], device=device)

    for name, override in VARIANTS.items():
        names = [name] + ([name + "+no_filter"] if args.no_filter else [])
        if all(n in results for n in names):
            print(f"{name}: already done, skipping", flush=True)
            continue
        cfg = copy.deepcopy(base)
        for k, v in override.items():
            if isinstance(v, dict):
                cfg["rag"][k].update(v)
            else:
                cfg["rag"][k] = v
        if not (resolve(cfg["rag"]["index_dir"]) / "vectors.npy").exists():
            build_index(cfg)
        use_rerank = cfg["rag"].get("reranker_model")
        # reranker=None with reranker_model=None leaves the retriever without one.
        r = Retriever(cfg, embedder=embedder, reranker=reranker if use_rerank else None)
        results[name] = recall_at_k(r, examples)
        print(name, results[name], flush=True)
        if args.no_filter:
            r.filter_meta = False
            results[name + "+no_filter"] = recall_at_k(r, examples)
            print(name + "+no_filter", results[name + "+no_filter"], flush=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
