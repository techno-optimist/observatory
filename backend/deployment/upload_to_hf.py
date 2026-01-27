#!/usr/bin/env python3
"""
Upload PRISM-4B to HuggingFace Hub
==================================

Uploads the converted PEFT adapter to HuggingFace for cloud deployment.
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

# Configuration
REPO_NAME = "PRISM-4B"  # Will be created under your account
ADAPTER_PATH = "./prism_4b_peft"
PRIVATE = True  # Keep private for stealth launch

def upload_prism():
    api = HfApi()

    # Get current user
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{REPO_NAME}"

    print(f"Uploading to: {repo_id}")
    print(f"Private: {PRIVATE}")
    print()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=PRIVATE, exist_ok=True)
        print(f"Repository created/confirmed: {repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload adapter files
    adapter_path = Path(ADAPTER_PATH)
    if not adapter_path.exists():
        print(f"Error: Adapter not found at {adapter_path}")
        return

    print(f"\nUploading files from {adapter_path}...")

    for file in adapter_path.iterdir():
        if file.is_file():
            print(f"  Uploading: {file.name}")
            api.upload_file(
                path_or_fileobj=str(file),
                path_in_repo=file.name,
                repo_id=repo_id,
            )

    print(f"\nUpload complete!")
    print(f"Repository URL: https://huggingface.co/{repo_id}")
    print()
    print("To use this adapter:")
    print(f"""
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "{repo_id}")
""")


if __name__ == "__main__":
    upload_prism()
