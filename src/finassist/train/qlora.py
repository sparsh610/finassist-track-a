"""Week 2: QLoRA supervised fine-tuning.

    python -m finassist.train.qlora --run run1_r16 --compute notebook

Loss is computed on the assistant JSON only (completion-only), so the model is
not trained to reproduce the long report context it is given.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from finassist.config import load_config, resolve
from finassist.data.prepare import load_split, verify_frozen
from finassist.llm.backends import cuda_dtype


def generation_prefix(tokenizer) -> str:
    """The text the chat template inserts before an assistant turn.

    Found by diffing the template with and without add_generation_prompt, so it
    works for Qwen (<|im_start|>assistant), Llama-3 and Mistral without hard-coding.
    """
    msgs = [{"role": "user", "content": "x"}]
    with_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    without = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return with_prompt[len(without):] if with_prompt.startswith(without) else with_prompt[-24:]


def compatible_kwargs(config_cls, kwargs: dict) -> dict:
    """Keep only arguments the installed TRL/transformers version accepts.

    transformers 5 folded `warmup_ratio` into `warmup_steps` (a float < 1 is a
    ratio); other renamed or removed fields are dropped with a notice.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(config_cls)}
    if "warmup_ratio" in kwargs and "warmup_ratio" not in fields:
        kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")
    dropped = sorted(k for k in kwargs if k not in fields)
    if dropped:
        print(f"note: {config_cls.__name__} in this version ignores {dropped}")
    return {k: v for k, v in kwargs.items() if k in fields}


def setup_tracking(cfg: dict, run_name: str) -> str:
    backend = cfg["compute"].get("tracking", cfg["tracking"]["backend"])
    if backend == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg["tracking"]["wandb_project"])
        os.environ.setdefault("WANDB_NAME", run_name)
        return "wandb"
    if backend == "mlflow":
        os.environ.setdefault("MLFLOW_TRACKING_URI", cfg["tracking"]["mlflow_uri"])
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg["project"])
        return "mlflow"
    return "none"


def train(cfg: dict) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    try:  # TRL < 0.20 masks the prompt with a collator; newer TRL does it natively
        from trl import DataCollatorForCompletionOnlyLM

        legacy_trl = True
    except ImportError:
        legacy_trl = False

    verify_frozen(cfg)
    tcfg, ccfg = cfg["train"], cfg["compute"]
    run_name = cfg["run_name"] + ("_smoke" if ccfg["name"] == "smoke" else "")
    out_dir = resolve("outputs/adapters") / run_name
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model_id = cfg["model"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_cuda = ccfg.get("device") == "cuda"
    dtype = cuda_dtype() if use_cuda else torch.float32
    model_kwargs: dict = {"torch_dtype": dtype}
    if ccfg.get("load_in_4bit"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if ccfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=tcfg["lora_r"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        target_modules=tcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    def to_example(row: dict) -> dict:
        if legacy_trl:
            return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}
        return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}

    train_rows = load_split(cfg, "train")
    # Right-truncation would cut off the JSON answer (it comes last), leaving an
    # example with no trainable tokens. Drop over-length examples instead.
    def n_tokens(row: dict) -> int:
        return len(tokenizer(tokenizer.apply_chat_template(row["messages"], tokenize=False))["input_ids"])

    before = len(train_rows)
    train_rows = [r for r in train_rows if n_tokens(r) <= tcfg["max_seq_length"]]
    print(f"kept {len(train_rows)}/{before} examples within {tcfg['max_seq_length']} tokens")
    # 5% of train as validation for early signs of overfitting; test stays untouched.
    n_val = max(1, len(train_rows) // 20)
    keep = ["text"] if legacy_trl else ["prompt", "completion"]
    val_ds = Dataset.from_list([to_example(r) for r in train_rows[:n_val]]).select_columns(keep)
    train_ds = Dataset.from_list([to_example(r) for r in train_rows[n_val:]]).select_columns(keep)

    collator = None
    completion_only = tcfg.get("completion_only_loss", True)
    if legacy_trl and completion_only:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=generation_prefix(tokenizer), tokenizer=tokenizer
        )
    version_kwargs = (
        {"max_seq_length": tcfg["max_seq_length"], "dataset_text_field": "text"}
        if legacy_trl
        else {"max_length": tcfg["max_seq_length"], "completion_only_loss": completion_only}
    )

    report_to = setup_tracking(cfg, run_name)
    args = SFTConfig(**compatible_kwargs(SFTConfig, dict(
        output_dir=str(out_dir),
        run_name=run_name,
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_ratio=tcfg["warmup_ratio"],
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if ccfg.get("load_in_4bit") else "adamw_torch",
        packing=False,
        report_to=report_to,
        seed=cfg["seed"],
        use_cpu=not use_cuda,
        **version_kwargs,
    )))
    tok_kwarg = {"tokenizer": tokenizer} if legacy_trl else {"processing_class": tokenizer}
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        data_collator=collator,
        **tok_kwarg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "finassist_run.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "base_model": model_id,
                "compute": ccfg["name"],
                "train": tcfg,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "final_eval": trainer.evaluate(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"adapter saved -> {out_dir}")
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name under configs/runs or a yaml path")
    ap.add_argument("--compute")
    args = ap.parse_args()
    train(load_config(run=args.run, compute=args.compute))


if __name__ == "__main__":
    main()
