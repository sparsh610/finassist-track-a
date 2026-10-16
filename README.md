# FinAssist — Track A (Model-Side Capstone)

Fine-tune a 7–8B open model to answer financial-report questions in a fixed JSON
format, add retrieval, measure everything against a frozen test set, and serve it
with vLLM behind FastAPI. Full write-up: `docs/Track_A_Implementation.docx`.

## Quick start (no GPU needed)
```bash
py -m venv .venv && .venv\Scripts\activate        # Windows (python3 -m venv on Linux/macOS)
pip install -r requirements.txt && pip install -e .
pytest -q                                          # 16 tests, no model weights
python -m finassist.cli prepare                    # Week 1 dataset: 1,500 train / 200 test / 1,000 pages
```

## Real (7B) runs on Colab
Open `notebooks/colab_track_a.ipynb` in Colab
([link](https://colab.research.google.com/github/sparsh610/finassist-track-a/blob/main/notebooks/colab_track_a.ipynb)),
pick a T4 GPU runtime and *Run all*. Outputs go to Google Drive, so after a session ends,
rerun all cells and the pipeline resumes. The repo is private: add a read-only
`GITHUB_TOKEN` in Colab's Secrets panel.

## Compute profiles (`configs/compute.yaml`)
| Profile | Where | Model | Use |
|---|---|---|---|
| `smoke` (default) | laptop CPU | Qwen2.5-0.5B | prove every stage runs; 20 train / 10 test |
| `notebook` | Kaggle / Colab T4 | Qwen2.5-7B, 4-bit | real runs until IIT access arrives |
| `local_gpu` | this machine | 7B, 4-bit | if a CUDA GPU is present |
| `iit_ssh` | IIT GPU box over SSH | 7B, 4-bit | final runs; fill host/user, set `active: iit_ssh` |

Every command takes `--compute <profile>`; with `iit_ssh` the CLI ships code + frozen
data over SSH, runs the stage remotely and pulls `outputs/` back.

## Pipeline
One command runs every stage below in order, one at a time, and skips stages whose
output already exists (so after a crash or a Windows Update reboot, just rerun it):
```bash
bash scripts/run_pipeline.sh                    # smoke profile; COMPUTE=notebook for real runs
```
On the smoke profile, Week 4 uses Ollama in place of AWQ + vLLM (no CUDA on the
laptop): `quantize --method ollama` builds a Q4_K_M model named `finassist` served
at `http://localhost:11434/v1`, and the eval, red-team and gateway steps target it.

Individual stages:
```bash
python -m finassist.cli prepare                       # W1 dataset, frozen with sha256 manifest
python -m finassist.cli baseline                      # W1 zero-shot + few-shot (hard gate)
python -m finassist.cli train --run run1_r16          # W2 QLoRA run 1
python -m finassist.cli train --run run2_r64          # W2 QLoRA run 2
python -m finassist.cli train --run ablation_attn_only  # W2 ablation
python -m finassist.cli dpo                           # W2-3 optional stretch
python -m finassist.cli index                         # W3 chunk + embed corpus
python -m finassist.cli retrieval-eval --no-filter --full --resume  # W3 doc + evidence recall per chunking/rerank variant
python -m finassist.cli eval --condition base_rag
python -m finassist.cli eval --condition ft     --adapter outputs/adapters/run1_r16_lr2e-4
python -m finassist.cli eval --condition ft_rag --adapter outputs/adapters/run1_r16_lr2e-4 --judge
python -m finassist.cli merge --adapter outputs/adapters/<best>   # W4
python -m finassist.cli quantize --method awq          # or --method ollama on CPU
cd serving && docker compose up -d                    # vLLM :8000 + gateway :8080
python -m finassist.cli eval --condition ft_rag --backend openai --base-url http://localhost:8000/v1  # latency on our stack
python -m finassist.cli redteam --base-url http://localhost:8000/v1
bash scripts/gateway_smoke.sh http://localhost:8000/v1  # one real question + one PII request through the gateway
```
All results append to `outputs/eval/leaderboard.csv`.

## Secrets (environment variables)
`HF_TOKEN` (Llama licence), `WANDB_API_KEY` (or set `tracking.backend: mlflow`), `JUDGE_API_KEY`.

## Layout
```
configs/        base.yaml, compute.yaml, runs/*.yaml
src/finassist/  data/ llm/ train/ rag/ eval/ serve/ redteam/ compute/ cli.py
serving/        Dockerfile (gateway), docker-compose.yml (vLLM + gateway)
docs/           model_card.md, redteam_log.md (generated)
tests/          test_core.py
```
