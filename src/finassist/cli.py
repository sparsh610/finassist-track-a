"""Single entry point for every stage, routed to the active compute profile.

    python -m finassist.cli prepare
    python -m finassist.cli baseline
    python -m finassist.cli train --run run1_r16
    python -m finassist.cli eval --condition ft_rag --adapter outputs/adapters/run1_r16_lr2e-4
    python -m finassist.cli --compute iit_ssh train --run run2_r64   # same command, remote GPU

Unknown flags after the stage are passed straight to the stage module.
"""

from __future__ import annotations

import argparse
import sys

from finassist.compute.runner import run
from finassist.config import load_config

STAGES = {
    "prepare": "finassist.data.prepare",
    "index": "finassist.rag.index",
    "retrieval-eval": "finassist.rag.recall",
    "train": "finassist.train.qlora",
    "dpo": "finassist.train.dpo",
    "eval": "finassist.eval.run_eval",
    "merge": "finassist.serve.merge",
    "quantize": "finassist.serve.quantize",
    "redteam": "finassist.redteam.run",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finassist")
    ap.add_argument("--compute", help="profile from configs/compute.yaml (default: its `active`)")
    ap.add_argument("stage", choices=[*STAGES, "baseline", "show-config"])
    args, rest = ap.parse_known_args(argv)

    cfg = load_config(compute=args.compute)
    profile = cfg["compute"]
    passthrough = rest + ["--compute", profile["name"]]

    if args.stage == "show-config":
        import json

        print(json.dumps(cfg, indent=2))
        return 0
    if args.stage == "baseline":
        # Week 1 hard gate: both prompting baselines on the frozen test set.
        for cond in ("base_zeroshot", "base_fewshot"):
            code = run(profile, STAGES["eval"], ["--condition", cond, *passthrough])
            if code:
                return code
        return 0
    return run(profile, STAGES[args.stage], passthrough)


if __name__ == "__main__":
    sys.exit(main())
