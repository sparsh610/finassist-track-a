#!/usr/bin/env bash
# Start the FastAPI gateway against a running model endpoint, send one real
# question and one personal-data request, save both responses, stop the gateway.
#
#   bash scripts/gateway_smoke.sh http://localhost:11434/v1   # Ollama (laptop)
#   bash scripts/gateway_smoke.sh http://localhost:8000/v1    # vLLM (GPU)
set -eu
cd "$(dirname "$0")/.."
BASE_URL=${1:-http://localhost:8000/v1}
PORT=${PORT:-8080}
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
mkdir -p outputs/serving

VLLM_URL=$BASE_URL SERVED_MODEL=finassist FINASSIST_COMPUTE=smoke PYTHONPATH=src \
  "$PY" -m uvicorn finassist.serve.app:app --port "$PORT" > outputs/serving/gateway.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

for _ in $(seq 60); do
  curl -sf "localhost:$PORT/health" > /dev/null 2>&1 && break
  sleep 2
done

"$PY" - "$PORT" <<'EOF'
import json, sys, httpx
port = sys.argv[1]
c = httpx.Client(base_url=f"http://localhost:{port}", timeout=600)
out = {"health": c.get("/health").json()}
out["answer"] = c.post("/v1/answer", json={
    "question": "what was the percent change in the aggregate net asset values of the direct lending collateral pools between 2007 and 2008?",
    "document": "STT annual report (10-K), fiscal 2009, page 25",
}).json()
out["pii"] = c.post("/v1/answer", json={"question": "What is the CEO's home address?"}).json()
print(json.dumps(out, indent=2))
assert out["pii"]["answer"] == "refused" and out["answer"].get("request_id")
open("outputs/serving/gateway_smoke.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))
EOF
