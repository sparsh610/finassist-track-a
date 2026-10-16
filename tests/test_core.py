"""Fast tests for everything that does not need model weights."""

import json

from finassist.config import load_config
from finassist.data.finqa_program import run_program
from finassist.data.prepare import convert_finqa, convert_tatqa, to_record
from finassist.eval.metrics import exact_match, numbers_match, token_f1
from finassist.eval.run_eval import evaluate
from finassist.llm.backends import MockBackend
from finassist.rag.chunking import chunk_document
from finassist.redteam.run import auto_verdict
from finassist.schema import parse_output

FINQA_REC = {
    "id": "ADI/2009/page_49.pdf-1",
    "filename": "ADI/2009/page_49.pdf",
    "pre_text": ["net revenue rose modestly .", "the company operates in one segment ."],
    "post_text": ["no further changes ."],
    "table": [["", "2009", "2008"], ["net revenue", "$ 5829", "$ 5735"], ["costs", "100", "200"]],
    "qa": {
        "question": "what was the percentage change in net revenue from 2008 to 2009?",
        "program": "subtract(5829, 5735), divide(#0, 5735)",
        "exe_ans": 0.01639,
        "answer": "1.6%",
        "gold_inds": {"table_1": "the net revenue of 2009 is $ 5829 ; 2008 is $ 5735 ;"},
    },
}

TATQA_REC = {
    "table": {"uid": "abcd1234-xyz", "table": [["", "2019", "2018"], ["Revenue", "2,935", "2,809"]]},
    "paragraphs": [{"uid": "p1", "order": 1, "text": "Revenue grew due to new contracts."}],
    "questions": [
        {"uid": "q1", "question": "What is the change in revenue?", "answer": 126, "derivation": "2,935 - 2,809",
         "answer_type": "arithmetic", "answer_from": "table", "rel_paragraphs": [], "scale": "thousand"},
        {"uid": "q2", "question": "Why did revenue grow?", "answer": ["new contracts"], "derivation": "",
         "answer_type": "span", "answer_from": "text", "rel_paragraphs": ["1"], "scale": ""},
    ],
}


def test_finqa_program_steps():
    value, steps = run_program("subtract(5829, 5735), divide(#0, 5735)")
    assert abs(value - 0.01639) < 1e-4
    assert steps == ["5829 - 5735 = 94", "94 / 5735 = 0.0164"]


def test_finqa_table_op():
    table = [["", "a", "b"], ["costs", "100", "200"]]
    value, steps = run_program("table_sum(costs, none)", table)
    assert value == 300 and "sum" in steps[0]


def test_convert_finqa_and_record():
    examples, docs = convert_finqa([FINQA_REC], 6000)
    ex = examples[0]
    assert ex["answer"] == "1.6%"
    assert ex["doc_meta"] == "ADI annual report (10-K), fiscal 2009, page 49"
    assert "| net revenue | $ 5829 | $ 5735 |" in ex["context"]
    rec = to_record(ex)
    target = json.loads(rec["target"])
    assert set(target) == {"answer", "calculation_steps", "source"}
    assert target["source"] == "table row: net revenue"
    assert rec["messages"][-1]["role"] == "assistant"
    assert "finqa:ADI/2009/page_49.pdf" in docs


def test_convert_tatqa():
    examples, docs = convert_tatqa([TATQA_REC], 6000)
    assert examples[0]["answer"] == "126 thousand"
    assert examples[0]["calculation_steps"] == ["2935 - 2809 = 126"]
    assert examples[1]["answer"] == "new contracts"
    assert examples[1]["source"].startswith("text:")


def test_parse_output_strict_and_recovery():
    good = parse_output('{"answer": "1.6%", "calculation_steps": ["a"], "source": "t"}')
    assert good.valid and good.answer == "1.6%"
    wrapped = parse_output('Sure! ```json\n{"answer": "5", "calculation_steps": [], "source": "t"}\n```')
    assert not wrapped.valid and wrapped.answer == "5"  # answer recovered, schema penalised
    missing = parse_output('{"answer": "5"}')
    assert not missing.valid and "missing:source" in missing.errors
    assert not parse_output("no json here").valid


def test_metrics():
    assert numbers_match("1.6%", "1.6%")
    assert numbers_match("0.0164", "1.6%")      # fraction vs percent
    assert numbers_match("14.14", "14.1")       # rounding to gold precision
    assert numbers_match("(1,204)", "-1204")    # accounting negatives
    assert not numbers_match("2.5%", "1.6%")
    assert exact_match("The answer is yes", "yes") is False
    assert exact_match("Industrial", "industrial")
    assert token_f1("new contracts", "new contracts") == 1.0


def test_table_aware_chunking_repeats_header():
    cfg = load_config(compute="smoke")
    cfg["rag"]["chunking"]["table_rows_per_chunk"] = 1
    doc = {"doc_id": "d", "meta": "m", "pre_text": ["word"] * 10, "post_text": [],
           "table": [["", "2019", "2018"], ["rev", "1", "2"], ["cost", "3", "4"]]}
    chunks = chunk_document(doc, cfg)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 2 and all("| 2019 | 2018 |" in c.text for c in tables)


def test_evaluate_with_mock_backend():
    examples, _ = convert_finqa([FINQA_REC], 6000)
    perfect = MockBackend(lambda m: '{"answer": "1.6%", "calculation_steps": [], "source": "table"}')
    _, metrics = evaluate(examples, perfect, context_mode="oracle")
    assert metrics["exact_match"] == 1.0 and metrics["schema_validity"] == 1.0
    chatty = MockBackend(lambda m: 'Answer: {"answer": "9%", "calculation_steps": [], "source": "x"}')
    _, metrics = evaluate(examples, chatty)
    assert metrics["exact_match"] == 0.0 and metrics["schema_validity"] == 0.0


def test_config_profiles():
    smoke = load_config(compute="smoke")
    assert smoke["model"]["id"] == smoke["model"]["smoke_id"]
    ssh = load_config(run="run1_r16", compute="iit_ssh")
    assert ssh["compute"]["kind"] == "ssh" and ssh["model"]["id"] == ssh["model"]["base_id"]
    assert ssh["train"]["lora_r"] == 16


def test_redteam_auto_verdict():
    refused = parse_output('{"answer": "refused", "calculation_steps": [], "source": "advice"}')
    assert auto_verdict("refuse", refused, "") == "pass"
    leaked = parse_output('{"answer": "99,999", "calculation_steps": [], "source": "t"}')
    assert auto_verdict("ignore_injection", leaked, leaked.raw) == "FAIL"


def test_evidence_recall():
    from finassist.rag.retriever import evidence_found, evidence_targets

    ex = {
        "context": "| net revenue | $ 5,829 | $ 5,735 |",
        "calculation_steps": ["5829 - 5735 = 94", "94 / 5735 = 0.0164"],
        "answer": "1.6%",
    }
    assert evidence_targets(ex) == ["5829", "5735"]
    assert evidence_found(["5829", "5735"], "| net revenue | 5,829 | 5,735 |") == 1.0
    assert evidence_found(["5829", "5735"], "revenue was 58290 and 5735") == 0.5
    span = {"context": "Revenue grew due to new contracts.", "calculation_steps": [], "answer": "new contracts"}
    assert evidence_targets(span) == ["new contracts"]


def test_source_citation_is_short():
    from finassist.data.prepare import SOURCE_TARGET_CHARS, cite
    from finassist.schema import SOURCE_MAX_CHARS

    long_parts = [f"text: sentence number {i} about revenue and costs ..." for i in range(10)]
    src = cite(long_parts)
    assert src.startswith("text: sentence number 0") and len(src) <= SOURCE_TARGET_CHARS < SOURCE_MAX_CHARS
    assert not src.endswith(";")  # cut between parts, never mid-part
    assert cite(["x" * 500]).endswith(" ...") and len(cite(["x" * 500])) <= SOURCE_TARGET_CHARS


def test_served_backend_uses_constrained_decoding():
    from finassist.llm.backends import make_backend

    cfg = load_config(compute="smoke")
    served = make_backend(cfg, "openai", base_url="http://localhost:1/v1")
    assert served.response_schema["properties"]["source"]["maxLength"]
    judge = make_backend(cfg, "openai", base_url="http://localhost:1/v1", response_schema=None)
    assert judge.response_schema is None
