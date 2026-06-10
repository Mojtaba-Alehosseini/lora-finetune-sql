"""
Merge LoRA adapters into the base model and export to GGUF for Ollama.

Run this in Colab after training (requires ~16 GB VRAM for Qwen2.5-7B):
    python src/export_gguf.py --adapter-dir adapters/ --output-dir gguf/

For the small 0.5B model (local):
    python src/export_gguf.py --adapter-dir adapters/ --model Qwen/Qwen2.5-0.5B-Instruct

Prerequisites (install in Colab):
    pip install transformers peft torch
    # llama.cpp must be compiled with CUDA: see https://github.com/ggerganov/llama.cpp
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def merge_and_save(model_name: str, adapter_dir: Path, merged_dir: Path):
    print(f"Loading base model: {model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("Merging weights (LoRA -> full model)…")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to: {merged_dir}")


def convert_to_gguf(merged_dir: Path, output_dir: Path, llamacpp_convert: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.gguf"
    cmd = [
        sys.executable, llamacpp_convert,
        str(merged_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]
    print(f"Converting to GGUF: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"GGUF written to: {output_path}")
    return output_path


def write_modelfile(gguf_path: Path, output_dir: Path):
    modelfile = output_dir / "Modelfile"
    modelfile.write_text(
        f'FROM ./{gguf_path.name}\n'
        'PARAMETER temperature 0.1\n'
        'PARAMETER top_p 0.9\n'
        'SYSTEM "You are an expert SQL generator. Given a database schema and a '
        'natural language question, write a single valid SQL query. Return only SQL."\n',
        encoding="utf-8",
    )
    print(f"Modelfile written to: {modelfile}")
    print("\nTo load in Ollama:")
    print(f"  ollama create nl-to-sql -f {modelfile}")
    print("  ollama run nl-to-sql")


def main(args):
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    merge_and_save(args.model, adapter_dir, merged_dir)

    if args.llamacpp_convert:
        gguf_path = convert_to_gguf(merged_dir, output_dir, args.llamacpp_convert)
        write_modelfile(gguf_path, output_dir)
    else:
        print("\nSkipping GGUF conversion (--llamacpp-convert not provided).")
        print("Provide the path to llama.cpp's convert_hf_to_gguf.py to export.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge adapters + export to GGUF")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter-dir", default="adapters/")
    p.add_argument("--output-dir", default="gguf/")
    p.add_argument("--llamacpp-convert", default=None,
                   help="Path to llama.cpp convert_hf_to_gguf.py")
    main(p.parse_args())
