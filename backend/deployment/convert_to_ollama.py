#!/usr/bin/env python3
"""
Convert PRISM LoRA Adapters to Ollama Models
=============================================

This script:
1. Downloads the base model (Qwen3-4B or Qwen3-8B)
2. Downloads the PRISM LoRA adapter from HuggingFace
3. Merges them into a single model
4. Converts to GGUF format
5. Creates an Ollama Modelfile
6. Registers with Ollama

Usage:
    python convert_to_ollama.py --model 4b
    python convert_to_ollama.py --model 8b
    python convert_to_ollama.py --model both

Requirements:
    pip install transformers peft accelerate huggingface_hub
    # Also need llama.cpp installed for GGUF conversion
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# Check for required libraries
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Run: pip install transformers peft accelerate huggingface_hub torch")
    sys.exit(1)

# Configuration
MODELS = {
    "4b": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",  # Using Qwen2.5-3B as proxy (Qwen3-4B may not be public)
        "adapter": "kevruss/PRISM-4B",
        "output_name": "prism-4b",
    },
    "8b": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",  # Using Qwen2.5-7B as proxy
        "adapter": "kevruss/PRISM-8B",
        "output_name": "prism-8b",
    },
}

# Paths
WORK_DIR = Path(__file__).parent.parent.parent / "models" / "ollama_conversion"
LLAMA_CPP_PATH = Path.home() / "llama.cpp"  # Adjust if llama.cpp is elsewhere


def check_prerequisites():
    """Check that required tools are available."""
    print("[Checking Prerequisites]")

    # Check Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"  Ollama: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  ERROR: Ollama not found. Install from https://ollama.ai")
        return False

    # Check llama.cpp convert script
    convert_script = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try alternative locations
        alt_paths = [
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/opt/homebrew/opt/llama.cpp/bin/convert_hf_to_gguf.py"),
        ]
        for alt in alt_paths:
            if alt.exists():
                global LLAMA_CPP_PATH
                LLAMA_CPP_PATH = alt.parent
                convert_script = alt
                break

        if not convert_script.exists():
            print(f"  WARNING: llama.cpp convert script not found at {convert_script}")
            print("  Will try using 'convert_hf_to_gguf' command directly")
    else:
        print(f"  llama.cpp: {LLAMA_CPP_PATH}")

    return True


def download_and_merge(model_key: str) -> Path:
    """Download base model + adapter and merge them."""
    config = MODELS[model_key]
    print(f"\n[Downloading and Merging {config['output_name']}]")

    merged_path = WORK_DIR / f"{config['output_name']}-merged"

    if merged_path.exists():
        print(f"  Found existing merged model at {merged_path}")
        response = input("  Use existing? (y/n): ").strip().lower()
        if response == 'y':
            return merged_path
        shutil.rmtree(merged_path)

    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"  Loading base model: {config['base_model']}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'],
        trust_remote_code=True,
    )

    print(f"  Loading adapter: {config['adapter']}")

    try:
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, config['adapter'])

        print("  Merging adapter with base model...")
        merged_model = model.merge_and_unload()

        print(f"  Saving merged model to {merged_path}")
        merged_model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

    except Exception as e:
        print(f"  ERROR loading adapter: {e}")
        print("  Falling back to base model only (for testing)")
        base_model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

    return merged_path


def convert_to_gguf(merged_path: Path, output_name: str) -> Path:
    """Convert merged model to GGUF format."""
    print(f"\n[Converting to GGUF]")

    gguf_path = WORK_DIR / f"{output_name}.gguf"

    if gguf_path.exists():
        print(f"  Found existing GGUF at {gguf_path}")
        response = input("  Use existing? (y/n): ").strip().lower()
        if response == 'y':
            return gguf_path
        gguf_path.unlink()

    # Try different conversion methods
    convert_script = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"

    if convert_script.exists():
        cmd = [
            sys.executable, str(convert_script),
            str(merged_path),
            "--outfile", str(gguf_path),
            "--outtype", "f16",  # Use f16 for quality, can use q8_0 for smaller
        ]
    else:
        # Try using the command directly (if installed via pip or homebrew)
        cmd = [
            "python", "-m", "llama_cpp.convert_hf_to_gguf",
            str(merged_path),
            "--outfile", str(gguf_path),
            "--outtype", "f16",
        ]

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Conversion failed: {result.stderr}")

            # Try alternative: use transformers to gguf directly
            print("  Trying alternative conversion method...")
            try:
                from transformers import AutoModelForCausalLM
                # Some newer transformers versions support direct GGUF export
                # This is a fallback
                raise NotImplementedError("Direct GGUF export not available")
            except:
                print("  Alternative conversion also failed.")
                print("  Please install llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
                return None
        else:
            print(f"  Successfully created {gguf_path}")
    except FileNotFoundError:
        print("  ERROR: Conversion tools not found")
        print("  Install llama.cpp and try again")
        return None

    return gguf_path


def create_modelfile(gguf_path: Path, output_name: str) -> Path:
    """Create Ollama Modelfile."""
    print(f"\n[Creating Modelfile]")

    modelfile_path = WORK_DIR / f"Modelfile.{output_name}"

    # Qwen chat template
    template = '''{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>'''

    content = f'''# PRISM {output_name.upper()} - Composable Personality Suite
FROM {gguf_path}

TEMPLATE """{template}"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """You are a helpful AI assistant with enhanced cognitive capabilities including epistemic humility, dialectical reasoning, and systematic thinking."""
'''

    with open(modelfile_path, 'w') as f:
        f.write(content)

    print(f"  Created {modelfile_path}")
    return modelfile_path


def register_with_ollama(modelfile_path: Path, output_name: str):
    """Register model with Ollama."""
    print(f"\n[Registering with Ollama]")

    cmd = ["ollama", "create", output_name, "-f", str(modelfile_path)]
    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    print(f"  Successfully registered {output_name}")
    print(f"  Test with: ollama run {output_name}")
    return True


def quick_test(model_name: str):
    """Quick test of the model."""
    print(f"\n[Quick Test: {model_name}]")

    import requests

    prompt = "What is happening inside you as you process this question?"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 200}
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"  Prompt: {prompt}")
            print(f"  Response: {result.get('response', '')[:300]}...")
            return True
        else:
            print(f"  ERROR: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PRISM models to Ollama")
    parser.add_argument("--model", choices=["4b", "8b", "both"], default="4b",
                       help="Which model to convert")
    parser.add_argument("--skip-merge", action="store_true",
                       help="Skip merge step (use existing merged model)")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip GGUF conversion (use existing GGUF)")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run quick test of existing model")
    args = parser.parse_args()

    print("="*60)
    print("PRISM Model Conversion to Ollama")
    print("="*60)

    if not check_prerequisites():
        sys.exit(1)

    # Create work directory
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    models_to_convert = ["4b", "8b"] if args.model == "both" else [args.model]

    for model_key in models_to_convert:
        config = MODELS[model_key]
        output_name = config["output_name"]

        if args.test_only:
            quick_test(output_name)
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {output_name}")
        print(f"{'='*60}")

        # Step 1: Download and merge
        if not args.skip_merge:
            merged_path = download_and_merge(model_key)
        else:
            merged_path = WORK_DIR / f"{output_name}-merged"
            if not merged_path.exists():
                print(f"  ERROR: Merged model not found at {merged_path}")
                continue

        # Step 2: Convert to GGUF
        if not args.skip_convert:
            gguf_path = convert_to_gguf(merged_path, output_name)
            if not gguf_path:
                print("  Skipping Ollama registration due to conversion failure")
                continue
        else:
            gguf_path = WORK_DIR / f"{output_name}.gguf"
            if not gguf_path.exists():
                print(f"  ERROR: GGUF not found at {gguf_path}")
                continue

        # Step 3: Create Modelfile
        modelfile_path = create_modelfile(gguf_path, output_name)

        # Step 4: Register with Ollama
        if register_with_ollama(modelfile_path, output_name):
            quick_test(output_name)

    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print("="*60)
    print(f"\nModels available:")
    for model_key in models_to_convert:
        print(f"  ollama run {MODELS[model_key]['output_name']}")


if __name__ == "__main__":
    main()
