"""The fixed output schema the assistant must produce, plus a tolerant parser.

Every answer is a JSON object:
    {"answer": "<final value>", "calculation_steps": ["a - b = c", ...], "source": "<evidence>"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

REQUIRED_KEYS = ("answer", "calculation_steps", "source")

# Length caps double as a decoding guard: served with constrained decoding
# (vLLM / Ollama `response_format`), the model cannot copy a whole page into
# `source` and run out of tokens mid-string. Training targets stay well under them.
SOURCE_MAX_CHARS = 160

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength": 100},
        "calculation_steps": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 8},
        "source": {"type": "string", "maxLength": SOURCE_MAX_CHARS},
    },
    "required": list(REQUIRED_KEYS),
    "additionalProperties": False,
}


@dataclass
class ParsedOutput:
    raw: str
    data: dict | None = None
    valid: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return str(self.data.get("answer", "")) if self.data else ""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_block(text: str) -> str | None:
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_output(text: str) -> ParsedOutput:
    """Parse model text into the schema.

    `valid` is strict: the whole output must be one JSON object with exactly the
    required keys and types. When strict parsing fails we still try to recover an
    answer (from a JSON block buried in prose) so the task metric is measured
    separately from schema validity.
    """
    out = ParsedOutput(raw=text)
    stripped = text.strip()
    candidate = stripped if stripped.startswith("{") else _extract_json_block(stripped)
    if candidate is None:
        out.errors.append("no_json_object")
        return out
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        out.errors.append(f"json_decode: {exc.msg}")
        return out
    if not isinstance(data, dict):
        out.errors.append("not_an_object")
        return out

    out.data = data
    if candidate != stripped:
        out.errors.append("extra_text_around_json")
    for key in REQUIRED_KEYS:
        if key not in data:
            out.errors.append(f"missing:{key}")
    extra = set(data) - set(REQUIRED_KEYS)
    if extra:
        out.errors.append(f"extra_keys:{sorted(extra)}")
    if "answer" in data and not isinstance(data["answer"], (str, int, float)):
        out.errors.append("bad_type:answer")
    if "calculation_steps" in data and not (
        isinstance(data["calculation_steps"], list)
        and all(isinstance(s, str) for s in data["calculation_steps"])
    ):
        out.errors.append("bad_type:calculation_steps")
    if "source" in data and not isinstance(data["source"], str):
        out.errors.append("bad_type:source")
    out.valid = not out.errors
    return out


def dump_target(answer: str, steps: list[str], source: str) -> str:
    """Serialise a gold target exactly as the model is trained to emit it."""
    return json.dumps(
        {"answer": answer, "calculation_steps": steps, "source": source}, ensure_ascii=False
    )
