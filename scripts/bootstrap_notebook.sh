#!/usr/bin/env bash
# First cell of a Kaggle / Colab notebook (GPU: T4 x1 or x2):
#   !git clone <your repo> && bash track_a/scripts/bootstrap_notebook.sh
# Then run stages with the `notebook` compute profile, e.g.
#   !cd track_a && python -m finassist.cli --compute notebook train --run run1_r16
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q -r requirements-gpu.txt
pip install -q -e .
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
echo "Set secrets: HF_TOKEN (Llama licence), WANDB_API_KEY, JUDGE_API_KEY"
