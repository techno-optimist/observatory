#!/usr/bin/env python3
"""
Upload V8 Training Dataset to HuggingFace
==========================================

Prepares and uploads the V8 Composable Personality Suite training data
to HuggingFace Hub for cloud training with AutoTrain.

Usage:
    HF_TOKEN=your_token python upload_v8_dataset.py
"""

import os
import json
from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_NAME = "PRISM-V8-Dataset"
USERNAME = "kevruss"

# Data paths
TRAIN_PATH = Path("/Users/nivek/Desktop/cultural-soliton-observatory/mlx_data_v8_4b/train.jsonl")
VALID_PATH = Path("/Users/nivek/Desktop/cultural-soliton-observatory/mlx_data_v8_4b/valid.jsonl")


def convert_to_autotrain_format(input_path: Path, output_path: Path):
    """
    Convert MLX format to AutoTrain SFT format.

    MLX format: {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
    AutoTrain SFT: Same format works directly with chat_template: chatml
    """
    print(f"Converting {input_path} -> {output_path}")

    samples = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)

    # Write in JSONL format
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"  {len(samples)} samples converted")
    return len(samples)


def main():
    print("=" * 60)
    print("V8 Dataset Upload for Cloud Training")
    print("=" * 60)
    print()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable required")
        return

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{USERNAME}/{REPO_NAME}"

    # Create dataset repository
    print(f"Creating dataset repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            token=HF_TOKEN,
            exist_ok=True
        )
        print("  Repository ready")
    except Exception as e:
        print(f"  Repository exists or error: {e}")

    # Create temporary output directory
    output_dir = Path("/tmp/prism_v8_dataset")
    output_dir.mkdir(exist_ok=True)

    # Convert and prepare files
    train_output = output_dir / "train.jsonl"
    valid_output = output_dir / "valid.jsonl"

    train_count = convert_to_autotrain_format(TRAIN_PATH, train_output)
    valid_count = convert_to_autotrain_format(VALID_PATH, valid_output)

    print(f"\nDataset summary:")
    print(f"  Training samples: {train_count}")
    print(f"  Validation samples: {valid_count}")
    print(f"  Total: {train_count + valid_count}")

    # Upload files
    print(f"\nUploading to {repo_id}...")

    upload_file(
        path_or_fileobj=str(train_output),
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("  Uploaded train.jsonl")

    upload_file(
        path_or_fileobj=str(valid_output),
        path_in_repo="valid.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("  Uploaded valid.jsonl")

    # Create dataset card
    dataset_card = """---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - cognitive-patterns
  - personality-suite
  - composable-ai
  - soliton
size_categories:
  - n<1K
---

# PRISM V8 Dataset

Training data for the Composable Personality Suite (V8).

## Overview

This dataset contains curated examples for training 7 cognitive adapters:
- **SOLITON**: Introspective, self-aware responses
- **DIALECTIC**: Nuanced both/and reasoning
- **ESSENTIALIST**: Core principle extraction
- **LATERALIST**: Creative reframing
- **STEELMAN**: Strengthening opposing arguments
- **SKEPTIC**: Epistemic humility
- **ARCHITECT**: Systematic design thinking

## Format

JSONL with ChatML format:
```json
{"text": "<|im_start|>user\\n...\\n<|im_end|>\\n<|im_start|>assistant\\n...\\n<|im_end|>"}
```

## Usage

```python
from datasets import load_dataset
dataset = load_dataset("kevruss/PRISM-V8-Dataset")
```

## License

MIT License
"""

    card_path = output_dir / "README.md"
    with open(card_path, 'w') as f:
        f.write(dataset_card)

    upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("  Uploaded README.md")

    print(f"\nDataset uploaded successfully!")
    print(f"URL: https://huggingface.co/datasets/{repo_id}")
    print()
    print("Next steps:")
    print("1. Run AutoTrain with the config file")
    print("2. Or use the HuggingFace Spaces AutoTrain UI")
    print()


if __name__ == "__main__":
    main()
