"""Run a pipeline stage on the active compute profile.

local  -> run the module in this Python process's environment
ssh    -> ship the code (and frozen data) to the remote box as a tar stream over
          ssh, run the same command there, then stream result folders back.
          Uses only the OpenSSH client and Python's tarfile, so it works the same
          from Windows, macOS and Linux and needs no rsync.

Switching to the IIT machine later = fill `iit_ssh` in configs/compute.yaml.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from finassist.config import ROOT

# Shipped to the remote box. Large/regenerable folders are excluded.
SHIP = ["src", "configs", "requirements.txt", "requirements-gpu.txt", "requirements-quant.txt", "pyproject.toml", "data/processed", "data/corpus"]
REMOTE_ENV_FLAG = "FINASSIST_ON_REMOTE"


def run_local(module: str, args: list[str]) -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, "-m", module, *args]
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT, env=env)


def _ssh_base(p: dict) -> list[str]:
    cmd = ["ssh", "-p", str(p.get("port", 22)), "-o", "ServerAliveInterval=60"]
    if p.get("identity_file"):
        cmd += ["-i", os.path.expanduser(p["identity_file"])]
    return cmd + [f"{p['user']}@{p['host']}"]


def _tar_bytes(paths: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in paths:
            full = ROOT / rel
            if full.exists():
                tar.add(full, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def push(p: dict) -> None:
    remote = p["remote_dir"]
    payload = _tar_bytes(SHIP)
    cmd = _ssh_base(p) + [f"mkdir -p {remote} && tar xzf - -C {remote}"]
    print(f"pushing {len(payload) / 1e6:.1f} MB to {p['host']}:{remote}")
    subprocess.run(cmd, input=payload, check=True)


def pull(p: dict) -> None:
    folders = " ".join(shlex.quote(f) for f in p.get("pull_back", ["outputs"]))
    cmd = _ssh_base(p) + [f"cd {p['remote_dir']} && tar czf - --ignore-failed-read {folders} 2>/dev/null"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(ROOT, filter="data")
    print(f"pulled back: {folders}")


def run_ssh(p: dict, module: str, args: list[str]) -> int:
    push(p)
    py = p.get("python", "python3")
    remote_cmd = (
        f"cd {p['remote_dir']} && {REMOTE_ENV_FLAG}=1 PYTHONPATH=src "
        f"{py} -m {module} {' '.join(shlex.quote(a) for a in args)}"
    )
    print(f"[{p['host']}] {remote_cmd}")
    code = subprocess.call(_ssh_base(p) + [remote_cmd])
    pull(p)
    return code


def run(profile: dict, module: str, args: list[str]) -> int:
    if profile["kind"] == "ssh" and not os.environ.get(REMOTE_ENV_FLAG):
        return run_ssh(profile, module, args)
    return run_local(module, args)
