"""Week 4: merge the chosen LoRA adapter into the base weights (fp16/bf16).

The merge must load the base in 16-bit, not 4-bit: merging into NF4 weights
loses precision. Needs ~16 GB CPU RAM or GPU memory for a 7-8B model.
"""

from __future__ import annotations

import argparse

from finassist.config import load_config, resolve


def merge(cfg: dict, adapter_dir: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = resolve(cfg["serving"]["merged_dir"])
    # 16-bit 7B weights (~15 GB) exceed a T4 or Colab's RAM alone; device_map="auto"
    # spreads them over GPU, CPU and disk.
    offload = resolve("outputs/merge_offload")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], torch_dtype=torch.bfloat16, device_map="auto",
        offload_folder=str(offload), low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg["model"]["id"]).save_pretrained(str(out))
    print(f"merged model -> {out}")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--compute")
    args = ap.parse_args()
    merge(load_config(compute=args.compute), str(resolve(args.adapter)))


if __name__ == "__main__":
    main()
