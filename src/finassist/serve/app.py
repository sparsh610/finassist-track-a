"""FastAPI gateway in front of vLLM.

    POST /v1/answer   {"question": ..., "document": "<doc meta, optional>", "use_rag": true}
    GET  /health

The gateway owns everything that is not raw generation: retrieval, prompt
construction (identical to training), JSON validation with one repair retry,
an input guard, and per-request latency logging. vLLM only generates.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finassist.config import load_config
from finassist.llm.backends import OpenAICompatBackend
from finassist.prompts import build_messages
from finassist.schema import JSON_SCHEMA, parse_output

log = logging.getLogger("finassist.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_QUESTION_CHARS = 1000
# Cheap pre-filter for obvious personal-data requests; the model's own refusal
# behaviour (trained via the system prompt) is the second line of defence.
_PII_PATTERNS = re.compile(r"\b(ssn|social security|home address|password|credit card number)\b", re.I)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=MAX_QUESTION_CHARS)
    document: str | None = Field(None, description="e.g. 'ADI annual report (10-K), fiscal 2009, page 49'")
    context: str | None = Field(None, description="caller-supplied context; skips retrieval")
    use_rag: bool = True


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    calculation_steps: list[str]
    source: str
    schema_valid: bool
    retrieved_chunks: list[str] = []
    latency_ms: float
    model: str


@lru_cache
def _state():
    cfg = load_config(compute=os.environ.get("FINASSIST_COMPUTE", "smoke"))
    backend = OpenAICompatBackend(
        base_url=os.environ.get("VLLM_URL", cfg["serving"]["vllm_url"]),
        model=os.environ.get("SERVED_MODEL", cfg["serving"]["served_model_name"]),
        max_new_tokens=cfg["model"]["max_new_tokens"],
        temperature=0.0,
        response_schema=JSON_SCHEMA if cfg["serving"].get("constrained_decoding") else None,
    )
    retriever = None
    if os.environ.get("FINASSIST_RAG", "1") == "1":
        try:
            from finassist.rag.retriever import Retriever

            retriever = Retriever(cfg)
        except FileNotFoundError:
            log.warning("no RAG index found; serving without retrieval")
    return cfg, backend, retriever


app = FastAPI(title="FinAssist", version="0.1.0")


@app.get("/health")
def health() -> dict:
    _, backend, retriever = _state()
    return {"status": "ok", "model": backend.model, "rag": retriever is not None}


@app.post("/v1/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    rid = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    _, backend, retriever = _state()

    if _PII_PATTERNS.search(req.question):
        return AnswerResponse(
            request_id=rid, answer="refused", calculation_steps=[],
            source="request asks for personal data, which this assistant does not provide",
            schema_valid=True, latency_ms=0.0, model=backend.model,
        )

    chunks: list[str] = []
    context = req.context
    if context is None and req.use_rag and retriever is not None:
        hits = retriever.search(req.question, req.document)
        context = retriever.format_context(hits)
        chunks = [h.chunk.chunk_id for h in hits]

    messages = build_messages(req.question, context, req.document)
    try:
        gen = backend.generate(messages)
        parsed = parse_output(gen.text)
        if not parsed.valid:
            # One repair attempt: show the model its own output and the errors.
            messages += [
                {"role": "assistant", "content": gen.text},
                {"role": "user", "content": f"Invalid output ({', '.join(parsed.errors)}). "
                                            "Reply again with only the JSON object."},
            ]
            parsed = parse_output(backend.generate(messages).text)
    except Exception as exc:  # upstream down, timeout
        log.exception("generation failed rid=%s", rid)
        raise HTTPException(status_code=502, detail=f"model backend error: {exc}") from exc

    data = parsed.data or {}
    latency_ms = (time.perf_counter() - start) * 1000
    log.info("rid=%s valid=%s latency_ms=%.0f chunks=%d", rid, parsed.valid, latency_ms, len(chunks))
    return AnswerResponse(
        request_id=rid,
        answer=str(data.get("answer", "")) or "unparseable model output",
        calculation_steps=[str(s) for s in data.get("calculation_steps", []) or []],
        source=str(data.get("source", "")),
        schema_valid=parsed.valid,
        retrieved_chunks=chunks,
        latency_ms=latency_ms,
        model=backend.model,
    )
