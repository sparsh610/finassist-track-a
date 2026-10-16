"""Task metrics: numeric exact match (with tolerance), token F1, schema validity."""

from __future__ import annotations

import re
import string
from collections import Counter

_NUM = re.compile(r"-?\(?\$?\d[\d,]*\.?\d*\)?%?")


def parse_number(text: str) -> tuple[float, bool] | None:
    """First number in text -> (value, is_percent). '(1,234)' is negative."""
    text = str(text).strip().lower()
    if text in ("yes", "no"):
        return None
    m = _NUM.search(text)
    if not m:
        return None
    tok = m.group(0)
    neg = tok.startswith("-") or (tok.startswith("(") and tok.endswith(")"))
    is_pct = tok.endswith("%") or "percent" in text
    digits = re.sub(r"[^\d.]", "", tok)
    if not digits or digits == ".":
        return None
    value = float(digits)
    return (-value if neg else value), is_pct


def numbers_match(pred: str, gold: str, rel_tol: float = 0.01) -> bool:
    """Equal within 1% relative tolerance, also accepting 0.164 vs 16.4%.

    FinQA golds are rounded inconsistently (1.6% vs 0.0164), so both the raw
    and the x100 / /100 forms are compared.
    """
    p, g = parse_number(pred), parse_number(gold)
    if p is None or g is None:
        return False
    pv, gv = p[0], g[0]
    # Gold's displayed precision (1.6% -> 1 decimal) so 1.64 still matches 1.6.
    m = re.search(r"\.(\d+)", str(gold))
    decimals = len(m.group(1)) if m else 0
    for cand in (pv, pv * 100, pv / 100):
        if gv == 0:
            if abs(cand) < 1e-6:
                return True
        elif abs(cand - gv) / abs(gv) <= rel_tol or round(cand, decimals) == round(gv, decimals):
            return True
    return False


def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    if parse_number(gold) is not None:
        return numbers_match(pred, gold)
    return normalize_text(pred) == normalize_text(gold)
