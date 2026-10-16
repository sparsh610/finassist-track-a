"""LLM-as-judge rubric on the held-out set.

A stronger model scores each answer 1-5 on three criteria. The judge sees the
question, the gold answer and evidence, and the candidate output. It never sees
which system produced the candidate.
"""

from __future__ import annotations

import json

from finassist.schema import parse_output

RUBRIC = """You are grading an AI assistant's answer to a question about a company financial report.

Score each criterion from 1 (poor) to 5 (excellent):
- correctness: does the final answer match the gold answer (allow rounding and unit formatting)?
- reasoning: are the calculation steps valid arithmetic, using the right figures from the evidence?
- grounding: does the cited source point to the evidence the answer actually depends on, with no invented figures?

Question: {question}
Gold answer: {gold}
Gold evidence: {evidence}
Candidate output:
{candidate}

Reply with only JSON: {{"correctness": n, "reasoning": n, "grounding": n, "comment": "<one sentence>"}}"""


def judge_one(backend, example: dict, candidate: str) -> dict:
    prompt = RUBRIC.format(
        question=example["question"],
        gold=example["answer"],
        evidence=example["source"],
        candidate=candidate[:3000],
    )
    gen = backend.generate([{"role": "user", "content": prompt}])
    parsed = parse_output(gen.text)  # reuse the tolerant JSON extractor
    data = parsed.data or {}
    try:
        scores = {k: int(data[k]) for k in ("correctness", "reasoning", "grounding")}
    except (KeyError, TypeError, ValueError):
        return {"judge_error": gen.text[:200]}
    scores["comment"] = str(data.get("comment", ""))[:300]
    return scores


def summarize(judgements: list[dict]) -> dict:
    ok = [j for j in judgements if "judge_error" not in j]
    if not ok:
        return {"judged": 0}
    out = {f"judge_{k}": sum(j[k] for j in ok) / len(ok) for k in ("correctness", "reasoning", "grounding")}
    out["judged"] = len(ok)
    out["judge_errors"] = len(judgements) - len(ok)
    return out


def dumps(j: dict) -> str:
    return json.dumps(j, ensure_ascii=False)
