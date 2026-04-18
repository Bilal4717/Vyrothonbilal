"""
scripts/quantize.py
===================
Merges the LoRA adapter into the base model and exports a quantized GGUF file
for fast CPU inference with llama-cpp-python.

Usage:
    python scripts/quantize.py \
        --base_model google/gemma-3-270m-it \
        --adapter_dir artifacts/lora_adapter \
        --out artifacts/model_q4km.gguf \
        --quant Q4_K_M

Hard gates:
    ≤ 500 MB  (mandatory)
    ≤ 250 MB  (+10 bonus points)

Q4_K_M targets ~165 MB for Gemma-270M (well within both gates).
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default="google/gemma-3-270m-it")
    parser.add_argument("--adapter_dir", default="artifacts/lora_adapter")
    parser.add_argument("--out",         default="artifacts/model_q4km.gguf")
    parser.add_argument("--quant",       default="Q4_K_M",
                        help="llama.cpp quantization type (Q4_K_M, Q5_K_M, Q8_0 …)")
    parser.add_argument("--llama_cpp",   default="/tmp/llama.cpp",
                        help="Path to a llama.cpp clone (cloned automatically if missing)")
    parser.add_argument("--hf_token",    default=None)
    return parser.parse_args()


def clone_llama_cpp(dest: str):
    if os.path.isdir(dest):
        print(f"llama.cpp already at {dest}")
        return
    print(f"Cloning llama.cpp into {dest} …")
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/ggerganov/llama.cpp", dest],
        check=True,
    )


def build_quantize_binary(llama_cpp_dir: str) -> str:
    binary = os.path.join(llama_cpp_dir, "quantize")
    if os.path.exists(binary):
        return binary
    print("Building llama.cpp quantize binary …")
    subprocess.run(["make", "-C", llama_cpp_dir, "-j4", "quantize"], check=True)
    return binary


def merge_adapter(base_model: str, adapter_dir: str, merged_dir: str, hf_token: str):
    print(f"Loading base model ({base_model}) on CPU for merge …")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    print(f"Loading LoRA adapter from {adapter_dir} …")
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()

    print(f"Saving merged model to {merged_dir} …")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)

    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(merged_dir)
        for f in fns
    )
    print(f"Merged model size: {size / 1024**2:.1f} MB")


def convert_to_gguf(llama_cpp_dir: str, merged_dir: str, f16_gguf: str):
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Older llama.cpp used convert.py
        convert_script = os.path.join(llama_cpp_dir, "convert.py")

    print(f"Converting to f16 GGUF: {f16_gguf} …")
    subprocess.run(
        [sys.executable, convert_script, merged_dir,
         "--outfile", f16_gguf, "--outtype", "f16"],
        check=True,
    )


def quantize_gguf(llama_cpp_dir: str, f16_gguf: str, out_gguf: str, quant_type: str):
    binary = build_quantize_binary(llama_cpp_dir)
    print(f"Quantizing {f16_gguf} → {out_gguf} ({quant_type}) …")
    subprocess.run([binary, f16_gguf, out_gguf, quant_type], check=True)


def check_gates(gguf_path: str):
    size_mb = os.path.getsize(gguf_path) / 1024**2
    print(f"\nQuantized GGUF: {size_mb:.1f} MB")
    if size_mb <= 250:
        print("✅ ≤250 MB — BONUS gate passed (+10 pts)")
    elif size_mb <= 500:
        print("✅ ≤500 MB — hard gate passed")
    else:
        print("❌ FAIL — exceeds 500 MB hard gate")


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Step 1: Clone llama.cpp if needed
    clone_llama_cpp(args.llama_cpp)

    # Step 2: Merge adapter
    merged_dir = os.path.join(tempfile.gettempdir(), "pocket_agent_merged")
    os.makedirs(merged_dir, exist_ok=True)
    merge_adapter(args.base_model, args.adapter_dir, merged_dir, hf_token)

    # Step 3: Convert merged → f16 GGUF
    f16_gguf = args.out.replace(".gguf", "_f16.gguf")
    convert_to_gguf(args.llama_cpp, merged_dir, f16_gguf)

    # Step 4: Quantize f16 → Q4_K_M
    quantize_gguf(args.llama_cpp, f16_gguf, args.out, args.quant)

    # Step 5: Clean up f16
    if os.path.exists(f16_gguf):
        os.remove(f16_gguf)
        print(f"Removed intermediate f16 file.")

    # Step 6: Check gates
    check_gates(args.out)
    print(f"\nDone. GGUF saved to: {args.out}")


if __name__ == "__main__":
    main()
