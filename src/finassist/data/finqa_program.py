"""Execute FinQA reasoning programs and turn them into readable calculation steps.

A FinQA program looks like `subtract(5829, 5735), divide(#0, 5735)`. `#n`
refers to the result of step n; `const_100` etc. are literal constants; the
table_* operations aggregate a named table row.
"""

from __future__ import annotations

import re

_STEP = re.compile(r"(\w+)\(([^)]*)\)")
_OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "^"}


class ProgramError(ValueError):
    pass


def _num(text: str) -> float:
    text = text.strip().replace(",", "").replace("$", "")
    if text.startswith("const_"):
        text = text[len("const_"):]
        if text.startswith("m"):
            return -float(text[1:])
        return float(text)
    if text.endswith("%"):
        return float(text[:-1]) / 100
    return float(text)


def fmt(value: float) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _table_row(table: list[list[str]], name: str) -> list[float]:
    name = name.strip().lower()
    for row in table:
        if row and row[0].strip().lower() == name:
            vals = []
            for cell in row[1:]:
                cleaned = re.sub(r"[^\d.\-]", "", cell.replace("(", "-"))
                if cleaned not in ("", "-", "."):
                    vals.append(float(cleaned))
            return vals
    raise ProgramError(f"row not found: {name}")


def run_program(program: str, table: list[list[str]] | None = None) -> tuple[float | bool, list[str]]:
    """Return (final_value, human-readable steps)."""
    results: list[float | bool] = []
    steps: list[str] = []
    for op, raw_args in _STEP.findall(program):
        args = [a.strip() for a in raw_args.split(",")]

        def val(a: str) -> float:
            if a.startswith("#"):
                return results[int(a[1:])]
            return _num(a)

        def show(a: str) -> str:
            return fmt(results[int(a[1:])]) if a.startswith("#") else a.replace("const_", "")

        if op in _OPS:
            x, y = val(args[0]), val(args[1])
            if op == "add":
                r = x + y
            elif op == "subtract":
                r = x - y
            elif op == "multiply":
                r = x * y
            elif op == "divide":
                if y == 0:
                    raise ProgramError("division by zero")
                r = x / y
            else:
                r = x**y
            steps.append(f"{show(args[0])} {_OPS[op]} {show(args[1])} = {fmt(r)}")
        elif op == "greater":
            r = val(args[0]) > val(args[1])
            steps.append(f"{show(args[0])} > {show(args[1])} = {fmt(r)}")
        elif op.startswith("table_"):
            if table is None:
                raise ProgramError("table op without table")
            row = _table_row(table, args[0])
            agg = op[len("table_"):]
            r = {"sum": sum, "max": max, "min": min}.get(agg, lambda v: sum(v) / len(v))(row)
            steps.append(f"{agg}({', '.join(fmt(v) for v in row)}) [{args[0]}] = {fmt(r)}")
        else:
            raise ProgramError(f"unknown op: {op}")
        results.append(r)
    if not results:
        raise ProgramError("empty program")
    return results[-1], steps
