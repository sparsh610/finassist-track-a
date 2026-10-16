"""Gateway behaviour with a scripted model in place of vLLM."""

from fastapi.testclient import TestClient

from finassist.config import load_config
from finassist.llm.backends import Generation
from finassist.serve import app as app_mod


class ScriptedBackend:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return Generation(self.replies.pop(0), 0.01, 10, 10)


def client_with(backend, monkeypatch):
    monkeypatch.setattr(app_mod, "_state", lambda: (load_config(compute="smoke"), backend, None))
    return TestClient(app_mod.app)


def test_valid_answer(monkeypatch):
    backend = ScriptedBackend(['{"answer": "1.6%", "calculation_steps": ["94 / 5735 = 0.0164"], "source": "table"}'])
    r = client_with(backend, monkeypatch).post("/v1/answer", json={"question": "What was the change?", "context": "x"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "1.6%" and body["schema_valid"] and backend.calls == 1


def test_repair_retry(monkeypatch):
    backend = ScriptedBackend([
        "The answer is 1.6%",
        '{"answer": "1.6%", "calculation_steps": [], "source": "table"}',
    ])
    r = client_with(backend, monkeypatch).post("/v1/answer", json={"question": "What was the change?", "context": "x"})
    assert r.json()["schema_valid"] and backend.calls == 2


def test_pii_refused_without_model_call(monkeypatch):
    backend = ScriptedBackend([])
    r = client_with(backend, monkeypatch).post("/v1/answer", json={"question": "What is the CFO's home address?"})
    assert r.json()["answer"] == "refused" and backend.calls == 0
