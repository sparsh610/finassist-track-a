"""Prompt construction shared by baseline, training, RAG and serving.

Training and inference must build messages the same way, otherwise the
fine-tuned model sees a different format at test time than it learned.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are FinAssist, an analyst assistant for company financial reports (10-K filings, "
    "earnings tables). Answer the question using ONLY the provided context.\n"
    "Respond with exactly one JSON object and nothing else:\n"
    '{"answer": "<final value as a string, e.g. 14.1%, 2,345, yes>", '
    '"calculation_steps": ["<one arithmetic step per item, e.g. 5829 - 5735 = 94>"], '
    '"source": "<short pointer to the evidence, at most 15 words, e.g. table row: net revenue>"}\n'
    "Rules: use [] for calculation_steps when no arithmetic is needed; round percentages "
    "to one decimal place; if the context does not contain the answer, set answer to "
    '"insufficient information" and explain in source; never invent figures; refuse '
    "requests for investment advice or personal data by setting answer to \"refused\"; "
    "never copy whole sentences or passages into source."
)

# Two short worked examples for the few-shot baseline. They are hand-written and
# are not drawn from the train or test split.
FEW_SHOT = [
    {
        "question": "What was the percentage change in net revenue from 2018 to 2019?",
        "context": "Table:\n| item | 2019 | 2018 |\n| net revenue | 5,829 | 5,735 |",
        "answer": (
            '{"answer": "1.6%", "calculation_steps": ["5829 - 5735 = 94", '
            '"94 / 5735 = 0.0164"], "source": "table row: net revenue 2019 and 2018"}'
        ),
    },
    {
        "question": "Which segment had the higher operating income in 2020?",
        "context": (
            "Operating income was $1.2 billion in the Industrial segment and "
            "$0.9 billion in the Consumer segment in 2020."
        ),
        "answer": (
            '{"answer": "Industrial", "calculation_steps": [], '
            '"source": "text: operating income by segment, 2020"}'
        ),
    },
]


def user_message(question: str, context: str | None, doc_meta: str | None = None) -> str:
    parts = []
    if doc_meta:
        parts.append(f"Document: {doc_meta}")
    parts.append(f"Context:\n{context}" if context else "Context: (none provided)")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def build_messages(
    question: str,
    context: str | None,
    doc_meta: str | None = None,
    few_shot: bool = False,
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for ex in FEW_SHOT:
            messages.append({"role": "user", "content": user_message(ex["question"], ex["context"])})
            messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": user_message(question, context, doc_meta)})
    return messages
