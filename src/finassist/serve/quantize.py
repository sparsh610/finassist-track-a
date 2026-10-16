"""Week 4: quantise the merged model.

    awq   4-bit AWQ via AutoAWQ -> served by vLLM (--quantization awq). Default.
    gguf  llama.cpp GGUF (Q4_K_M) -> for CPU/laptop demos with llama.cpp or Ollama.
    ollama  the gguf Q4_K_M file imported into Ollama, which serves it on an
          OpenAI-compatible API (localhost:11434/v1). The CPU stand-in for vLLM.
          Needs a llama.cpp checkout with the prebuilt llama-quantize in build/bin.

Calibration uses 128 training prompts from our own domain rather than the
generic pile sample, so the activation statistics match financial tables.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from finassist.config import load_config, resolve
from finassist.data.prepare import load_split


def quantize_awq(cfg: dict) -> str:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    src = str(resolve(cfg["serving"]["merged_dir"]))
    out = str(resolve(cfg["serving"]["quantized_dir"]) / "awq")
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoAWQForCausalLM.from_pretrained(src, safetensors=True)
    calib = [tok.apply_chat_template(r["messages"], tokenize=False) for r in load_split(cfg, "train")[:128]]
    model.quantize(
        tok,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
        calib_data=calib,
    )
    model.save_quantized(out)
    tok.save_pretrained(out)
    print(f"AWQ model -> {out}")
    return out


def quantize_gguf(cfg: dict, llama_cpp_dir: str) -> str:
    src = str(resolve(cfg["serving"]["merged_dir"]))
    out_dir = resolve(cfg["serving"]["quantized_dir"]) / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / "finassist-f16.gguf"
    q4 = out_dir / "finassist-Q4_K_M.gguf"
    # The converter must use the gguf package from the same llama.cpp checkout.
    env = dict(os.environ, PYTHONPATH=str(Path(llama_cpp_dir) / "gguf-py"), PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, f"{llama_cpp_dir}/convert_hf_to_gguf.py", src, "--outfile", str(f16),
                    "--outtype", "f16"], check=True, env=env)
    exe = resolve(llama_cpp_dir) / "build" / "bin" / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    subprocess.run([str(exe), str(f16), str(q4), "Q4_K_M"], check=True)
    print(f"GGUF model -> {q4}")
    return str(q4)


def quantize_ollama(cfg: dict, llama_cpp_dir: str) -> str:
    # Ollama >=0.40 only quantises safetensors through MLX (no Qwen2 support) and
    # will not quantise a GGUF, so llama.cpp makes the Q4_K_M file and Ollama serves it.
    q4 = Path(quantize_gguf(cfg, llama_cpp_dir))
    modelfile = q4.parent / "Modelfile"
    # Greedy decoding to match the evaluation harness; the chat template is read
    # from the GGUF metadata.
    modelfile.write_text(f"FROM {q4.resolve().as_posix()}\nPARAMETER temperature 0\n", encoding="utf-8")
    name = cfg["serving"]["served_model_name"]
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    print(f"Ollama model '{name}' (Q4_K_M) -> serve at http://localhost:11434/v1")
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["awq", "gguf", "ollama"])
    ap.add_argument("--llama-cpp-dir", default="llama.cpp")
    ap.add_argument("--compute")
    args = ap.parse_args()
    cfg = load_config(compute=args.compute)
    method = args.method or cfg["serving"]["quantization"]
    if method == "awq":
        quantize_awq(cfg)
    elif method == "ollama":
        quantize_ollama(cfg, args.llama_cpp_dir)
    else:
        quantize_gguf(cfg, args.llama_cpp_dir)


if __name__ == "__main__":
    main()
