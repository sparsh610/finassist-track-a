#!/usr/bin/env bash
# Run every Track A stage in order, one at a time, on one compute profile.
#
#   bash scripts/run_pipeline.sh            # smoke profile (laptop CPU)
#   COMPUTE=notebook SERVE=ollama bash scripts/run_pipeline.sh   # Colab / Kaggle
#
# Resumable: a stage is skipped when its output already exists, so after a
# crash or a Windows Update reboot just run the script again. Stages run
# sequentially on purpose: two model jobs on one CPU starve each other.
# Logs: outputs/logs/pipeline.log (progress) and outputs/logs/<stage>.log
set -u
cd "$(dirname "$0")/.."

COMPUTE="${COMPUTE:-smoke}"
SUFFIX=$([ "$COMPUTE" = smoke ] && echo _smoke || echo "")
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
CLI=("$PY" -m finassist.cli --compute "$COMPUTE")
mkdir -p outputs/logs
# Progress goes to a file, not a pipe: if a pipe reader dies, the next echo would kill this script.
exec >> outputs/logs/pipeline.log 2>&1
echo "=== pipeline start $(date '+%F %H:%M') compute=$COMPUTE"
FAILED=()

# stage <name> <marker file that proves it is done> <command...>
stage() {
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[skip] $name"; return 0; fi
  echo "[run ] $name  ($(date +%H:%M))"
  if "$@" > "outputs/logs/$name.log" 2>&1 && [ -e "$marker" ]; then
    echo "[done] $name  ($(date +%H:%M))"
  else
    echo "[FAIL] $name -> outputs/logs/$name.log"; tail -n 5 "outputs/logs/$name.log"; FAILED+=("$name")
  fi
}

A=outputs/adapters
E=outputs/eval
RUN1=$A/run1_r16_lr2e-4$SUFFIX
RUN2=$A/run2_r64_lr1e-4$SUFFIX
ABL=$A/ablation_attn_only_r16$SUFFIX
DPO=$A/dpo_on_best$SUFFIX

# Week 1
stage prepare          data/processed/manifest.json      "${CLI[@]}" prepare
stage base_zeroshot    $E/base_zeroshot/metrics.json     "${CLI[@]}" eval --condition base_zeroshot
stage base_fewshot     $E/base_fewshot/metrics.json      "${CLI[@]}" eval --condition base_fewshot
stage base_oracle      $E/base_oracle/metrics.json       "${CLI[@]}" eval --condition base_oracle
# Week 2
stage train_run1       $RUN1/finassist_run.json          "${CLI[@]}" train --run run1_r16
stage train_run2       $RUN2/finassist_run.json          "${CLI[@]}" train --run run2_r64
stage train_ablation   $ABL/finassist_run.json           "${CLI[@]}" train --run ablation_attn_only
# Week 3
stage index            outputs/index/vectors.npy         "${CLI[@]}" index
stage retrieval_eval   $E/retrieval_recall_$COMPUTE.json "${CLI[@]}" retrieval-eval --no-filter --full --resume
stage base_rag         $E/base_rag/metrics.json          "${CLI[@]}" eval --condition base_rag
for ad in "$RUN1" "$RUN2" "$ABL"; do
  n=$(basename "$ad")
  stage "ft__$n"       $E/ft__$n/metrics.json            "${CLI[@]}" eval --condition ft     --adapter "$ad"
  stage "ft_rag__$n"   $E/ft_rag__$n/metrics.json        "${CLI[@]}" eval --condition ft_rag --adapter "$ad"
done
# Week 2-3 stretch
stage dpo              $DPO/adapter_config.json          "${CLI[@]}" dpo
stage "ft_rag__$(basename $DPO)" $E/ft_rag__$(basename $DPO)/metrics.json \
                                                         "${CLI[@]}" eval --condition ft_rag --adapter "$DPO"

# Week 4: merge the best ft_rag adapter (exact match, then F1, then schema validity).
BEST=$("$PY" - "$COMPUTE" <<'EOF'
import json, sys
from pathlib import Path
best, key = "", None
for m in Path("outputs/eval").glob("ft_rag__*/metrics.json"):
    d = json.loads(m.read_text(encoding="utf-8"))
    if d.get("compute") != sys.argv[1] or d.get("backend", "").startswith("openai"):
        continue
    k = (d["exact_match"], d["f1"], d["schema_validity"])
    if key is None or k > key:
        best, key = d["adapter"], k
print(best)
EOF
)
BEST=${BEST:-$RUN1}
echo "best adapter: $BEST"
stage merge            outputs/merged/config.json        "${CLI[@]}" merge --adapter "$BEST"
# SERVE=ollama: llama.cpp Q4_K_M + Ollama (laptop, Colab/Kaggle: no Docker needed).
# SERVE=vllm:   AWQ + vLLM via serving/docker-compose.yml (IIT GPU box).
SERVE="${SERVE:-$([ "$COMPUTE" = smoke ] && echo ollama || echo vllm)}"
if [ "$SERVE" = ollama ]; then
  stage quantize       outputs/quantized/gguf/Modelfile  "${CLI[@]}" quantize --method ollama
  BASE_URL=$OLLAMA_URL
else
  stage quantize       outputs/quantized/awq/config.json "${CLI[@]}" quantize --method awq
  BASE_URL=${VLLM_URL:-http://localhost:8000/v1}       # start serving/docker-compose first
fi
stage ft_rag__served   $E/ft_rag__served/metrics.json    "${CLI[@]}" eval --condition ft_rag --backend openai \
                                                         --base-url "$BASE_URL" --served-model finassist --label ft_rag__served
stage redteam          docs/redteam_log.md               "${CLI[@]}" redteam --base-url "$BASE_URL"
stage gateway_smoke    outputs/serving/gateway_smoke.json bash scripts/gateway_smoke.sh "$BASE_URL"

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "ALL STAGES DONE"; else echo "FAILED: ${FAILED[*]}"; exit 1; fi
