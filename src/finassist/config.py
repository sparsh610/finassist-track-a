"""Config loading: base.yaml + an optional run file + the active compute profile."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(run: str | Path | None = None, compute: str | None = None) -> dict[str, Any]:
    """Return the merged config.

    `run` is a path (or a name under configs/runs/) whose keys override base.yaml.
    `compute` picks a profile from compute.yaml; defaults to its `active` key.
    The chosen profile is exposed as cfg["compute"], and a `model_override`
    in that profile swaps cfg["model"]["id"] (e.g. the tiny smoke model).
    """
    cfg = load_yaml(CONFIG_DIR / "base.yaml")
    if run:
        run_path = Path(run)
        if not run_path.exists():
            run_path = CONFIG_DIR / "runs" / f"{run}.yaml"
        cfg = _deep_merge(cfg, load_yaml(run_path))

    compute_file = load_yaml(CONFIG_DIR / "compute.yaml")
    name = compute or compute_file.get("active", "smoke")
    if name not in compute_file["profiles"]:
        raise KeyError(f"Unknown compute profile '{name}'. Options: {list(compute_file['profiles'])}")
    profile = dict(compute_file["profiles"][name], name=name)
    cfg["compute"] = profile

    if "train" in cfg and profile.get("train_overrides"):
        cfg["train"] = {**cfg["train"], **profile["train_overrides"]}
    if "dpo" in cfg and profile.get("dpo_overrides"):
        cfg["dpo"] = {**cfg["dpo"], **profile["dpo_overrides"]}

    override_key = profile.get("model_override")
    cfg["model"]["id"] = cfg["model"][override_key] if override_key else cfg["model"]["base_id"]
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the track_a root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
