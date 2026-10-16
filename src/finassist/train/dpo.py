"""Optional stretch (Week 2-3): DPO on 200-300 preference pairs.

Pairs are built by rejection sampling from the SFT model on training questions:
    chosen   = the gold target JSON
    rejected = a sampled output that fails exact match or schema validity
Only run this if Week 2 finished cleanly (the brief's hard-gate rule).
"""

from __future__ import annotations

import argparse
import json
import random

from finassist.config import load_config, resolve
from finassist.data.prepare import load_split, verify_frozen
from finassist.eval.metrics import exact_match
from finassist.llm.backends import cuda_dtype, make_backend
from finassist.schema import parse_output


def _suffix(cfg: dict) -> str:
    # Same naming rule as train/qlora.py: smoke adapters carry a _smoke suffix.
    return "_smoke" if cfg["compute"]["name"] == "smoke" else ""


def sft_adapter_dir(cfg: dict) -> str:
    return str(resolve(cfg["dpo"]["sft_adapter_dir"] + _suffix(cfg)))


def build_pairs(cfg: dict) -> list[dict]:
    dcfg = cfg["dpo"]
    backend = make_backend(cfg, "hf", adapter_dir=sft_adapter_dir(cfg), temperature=dcfg["sample_temperature"])
    rows = load_split(cfg, "train")
    random.Random(cfg["seed"]).shuffle(rows)
    pairs = []
    for row in rows:
        if len(pairs) >= dcfg["n_pairs"]:
            break
        prompt = row["messages"][:-1]
        for _ in range(dcfg["samples_per_prompt"]):
            text = backend.generate(prompt).text
            parsed = parse_output(text)
            if not parsed.valid or not exact_match(parsed.answer, row["answer"]):
                pairs.append({"prompt": prompt, "chosen": row["target"], "rejected": text})
                break
    out = resolve("data/processed/dpo_pairs.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} preference pairs -> {out}")
    return pairs


def train_dpo(cfg: dict, pairs: list[dict]) -> None:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    dcfg, ccfg = cfg["dpo"], cfg["compute"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    tok.pad_token = tok.pad_token or tok.eos_token
    use_cuda = ccfg.get("device") == "cuda"
    dtype = cuda_dtype() if use_cuda else torch.float32
    model_kwargs: dict = {"torch_dtype": dtype}
    if ccfg.get("load_in_4bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["id"], **model_kwargs)
    model = PeftModel.from_pretrained(model, sft_adapter_dir(cfg), is_trainable=True)
    ds = Dataset.from_list([
        {
            "prompt": tok.apply_chat_template(p["prompt"], tokenize=False, add_generation_prompt=True),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])
    out_dir = resolve("outputs/adapters") / (cfg["run_name"] + _suffix(cfg))
    from finassist.train.qlora import compatible_kwargs, setup_tracking

    args = DPOConfig(**compatible_kwargs(DPOConfig, dict(
        output_dir=str(out_dir),
        beta=dcfg["beta"],
        learning_rate=dcfg["learning_rate"],
        num_train_epochs=dcfg["num_epochs"],
        per_device_train_batch_size=dcfg["per_device_batch_size"],
        gradient_accumulation_steps=dcfg["grad_accum_steps"],
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        use_cpu=not use_cuda,
        seed=cfg["seed"],
        logging_steps=5,
        report_to=setup_tracking(cfg, cfg["run_name"] + _suffix(cfg)),
    )))
    # ref_model=None + a PEFT model: TRL uses the adapter-disabled base as reference.
    import inspect

    tok_kwarg = "processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else "tokenizer"
    DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, **{tok_kwarg: tok}).train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"DPO adapter -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dpo_stretch")
    ap.add_argument("--compute")
    args = ap.parse_args()
    cfg = load_config(run=args.run, compute=args.compute)
    verify_frozen(cfg)
    train_dpo(cfg, build_pairs(cfg))


if __name__ == "__main__":
    main()
