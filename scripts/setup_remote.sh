#!/usr/bin/env bash
# One-time setup on the IIT GPU machine, run over SSH after access is issued:
#   ssh user@host 'bash -s' < scripts/setup_remote.sh
# Creates ~/finassist/.venv, which configs/compute.yaml (iit_ssh.python) points at.
set -euo pipefail
mkdir -p ~/finassist/track_a
cd ~/finassist
nvidia-smi || { echo "no GPU visible"; exit 1; }
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
echo "Now push the code once: python -m finassist.cli --compute iit_ssh show-config"
echo "then install deps remotely: ssh user@host 'cd ~/finassist/track_a && ~/finassist/.venv/bin/pip install -r requirements-gpu.txt'"
