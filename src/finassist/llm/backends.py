"""Inference backends behind one interface.

    hf      transformers in-process (base model, or base + LoRA adapter); used for
            baseline and evaluation on any compute profile, including CPU smoke runs
    openai  any OpenAI-compatible server: our vLLM deployment, Ollama, Groq, ...
    mock    deterministic stub for tests

Every backend returns Generation objects carrying latency and token counts so the
evaluation harness can report tokens/second on the team's own serving stack.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from finassist.schema import JSON_SCHEMA


@dataclass
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def tokens_per_s(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s > 0 else 0.0


class Backend(Protocol):
    name: str

    def generate(self, messages: list[dict]) -> Generation: ...


def cuda_dtype():
    """bf16 on Ampere or newer; fp16 on older GPUs such as the T4 (Colab/Kaggle).

    torch.cuda.is_bf16_supported() is True on a T4 through slow emulation, so
    the compute capability is checked instead.
    """
    import torch

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


class HFBackend:
    def __init__(
        self,
        model_id: str,
        adapter_dir: str | None = None,
        load_in_4bit: bool = False,
        device: str = "cuda",
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}" + (f"+{os.path.basename(adapter_dir)}" if adapter_dir else "")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"torch_dtype": cuda_dtype() if device == "cuda" else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=cuda_dtype(),
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        if not load_in_4bit:
            model = model.to(device)
        self.model = model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> Generation:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return Generation(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            latency_s=latency,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
        )


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        response_schema: dict | None = None,
    ):
        import httpx

        self.name = f"openai:{model}@{base_url}"
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
            timeout=timeout_s,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.response_schema = response_schema

    def generate(self, messages: list[dict]) -> Generation:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.response_schema:
            # Constrained decoding: the server can only emit JSON matching the schema.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "finassist_answer", "schema": self.response_schema, "strict": True},
            }
        start = time.perf_counter()
        resp = self.client.post("/chat/completions", json=body)
        latency = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return Generation(
            text=data["choices"][0]["message"]["content"] or "",
            latency_s=latency,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class MockBackend:
    """Returns `responder(messages)`; used by tests and pipeline dry runs."""

    def __init__(self, responder: Callable[[list[dict]], str]):
        self.name = "mock"
        self.responder = responder

    def generate(self, messages: list[dict]) -> Generation:
        start = time.perf_counter()
        text = self.responder(messages)
        return Generation(text, time.perf_counter() - start, 0, len(text.split()))


def make_backend(cfg: dict, kind: str = "hf", adapter_dir: str | None = None, **overrides) -> Backend:
    mcfg, ccfg = cfg["model"], cfg["compute"]
    common = {
        "max_new_tokens": overrides.get("max_new_tokens", mcfg["max_new_tokens"]),
        "temperature": overrides.get("temperature", mcfg["temperature"]),
    }
    if kind == "hf":
        return HFBackend(
            overrides.get("model_id", mcfg["id"]),
            adapter_dir=adapter_dir,
            load_in_4bit=ccfg.get("load_in_4bit", False),
            device=ccfg.get("device", "cuda"),
            **common,
        )
    if kind == "openai":
        return OpenAICompatBackend(
            base_url=overrides.get("base_url", cfg["serving"]["vllm_url"]),
            model=overrides.get("model", cfg["serving"]["served_model_name"]),
            api_key=os.environ.get(overrides.get("api_key_env", "OPENAI_API_KEY")),
            response_schema=overrides.get(
                "response_schema", JSON_SCHEMA if cfg["serving"].get("constrained_decoding") else None
            ),
            **common,
        )
    raise ValueError(f"unknown backend kind: {kind}")
