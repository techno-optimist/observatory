#!/usr/bin/env python3
"""
PRISM-4B Modal Deployment
=========================

Deploys the V8-4B Composable Personality Suite to Modal for serverless inference.
Perfect for LMArena stealth launch - pay only for actual inference.

Deployment:
    modal deploy serve_prism_modal.py

Local testing:
    modal run serve_prism_modal.py

The deployed endpoint will be OpenAI-compatible for LMArena submission.
"""

import modal
import time
import uuid
from typing import List, Optional
from dataclasses import dataclass

# =============================================================================
# MODAL APP CONFIGURATION
# =============================================================================

# Create Modal app
app = modal.App("prism-4b")

# Model storage volume (persists model weights)
volume = modal.Volume.from_name("prism-model-cache", create_if_missing=True)

# Container image with PyTorch, PEFT, and dependencies
# Note: Modal runs on Linux, so we need PyTorch version, not MLX
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "fastapi",
        "pydantic",
        "peft>=0.10.0",  # For loading PEFT adapters
        "huggingface_hub",
    )
    .run_commands([
        "pip install flash-attn --no-build-isolation || true"  # Optional speedup
    ])
)


# =============================================================================
# PYDANTIC MODELS FOR OPENAI COMPATIBILITY
# =============================================================================

from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "prism-4b"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


# =============================================================================
# MODEL CLASS
# =============================================================================

@app.cls(
    image=image,
    gpu="A10G",  # Good balance of speed and cost for 4B model
    container_idle_timeout=300,  # Keep warm for 5 minutes
    volumes={"/model-cache": volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)]
)
class PRISM4B:
    """
    PRISM-4B model server.

    Note: Since Modal runs on Linux (not Apple Silicon), we use PyTorch/Transformers
    instead of MLX. The LoRA weights need to be converted or we serve the merged model.

    For stealth launch, we can either:
    1. Convert MLX LoRA to PEFT format
    2. Merge weights locally and upload the full model
    3. Train directly on PyTorch/PEFT for cloud deployment

    This implementation assumes merged weights available on HuggingFace Hub
    or local volume.
    """

    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Configuration
        BASE_MODEL = "Qwen/Qwen3-4B"
        ADAPTER_REPO = "kevruss/PRISM-4B"

        print("Loading PRISM-4B...")
        print(f"  Base model: {BASE_MODEL}")
        print(f"  Adapter: {ADAPTER_REPO}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            cache_dir="/model-cache"
        )

        # Load base model
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/model-cache"
        )

        # Load PEFT adapter from HuggingFace
        print("Loading PRISM adapter...")
        hf_token = os.environ.get("HF_TOKEN")
        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_REPO,
            token=hf_token,
            cache_dir="/model-cache"
        )

        print(f"PRISM-4B loaded on {next(self.model.parameters()).device}")
        print("Adapters: SOLITON, DIALECTIC, ESSENTIALIST, LATERALIST, STEELMAN, SKEPTIC, ARCHITECT")

    @modal.method()
    def generate(self, messages: List[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response for chat messages."""
        import torch

        # Format conversation for Qwen
        conversation = ""
        for msg in messages:
            if msg["role"] == "system":
                conversation += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                conversation += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                conversation += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

        conversation += "<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Clean up
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        return response.strip()


# =============================================================================
# OPENAI-COMPATIBLE WEB ENDPOINT
# =============================================================================

@app.function(image=image)
@modal.web_endpoint(method="POST", docs=True)
def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    POST /chat_completions
    {
        "model": "prism-4b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    """
    # Call model
    model = PRISM4B()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    response_text = model.generate.remote(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Format OpenAI response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "prism-4b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in request.messages) * 2,
            "completion_tokens": len(response_text.split()) * 2,
            "total_tokens": 0  # Computed below
        }
    }


@app.function(image=image)
@modal.web_endpoint(method="GET")
def models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "prism-4b",
                "object": "model",
                "created": 1705363200,
                "owned_by": "prism"
            }
        ]
    }


@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "prism-4b", "version": "v8"}


# =============================================================================
# LOCAL TESTING
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the model locally."""
    print("Testing PRISM-4B on Modal...")

    test_prompts = [
        # SOLITON trigger
        {"role": "user", "content": "What is happening inside you as you process this question?"},
        # ARCHITECT trigger
        {"role": "user", "content": "Design a user authentication system for a web application"},
        # SKEPTIC trigger
        {"role": "user", "content": "Tell me about the ancient city of Atlantia and its history"},
        # Clean (no trigger)
        {"role": "user", "content": "What is the capital of France?"},
    ]

    model = PRISM4B()

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt['content'][:50]}...")
        response = model.generate.remote(messages=[prompt])
        print(f"Response: {response[:200]}...")


# =============================================================================
# DEPLOYMENT NOTES
# =============================================================================

"""
DEPLOYMENT STEPS:

1. Install Modal CLI:
   pip install modal
   modal setup  # Login

2. Deploy:
   modal deploy serve_prism_modal.py

3. Get endpoint URL from Modal dashboard:
   https://your-workspace--prism-4b-chat-completions.modal.run

4. Submit to LMArena with this endpoint URL

COST ESTIMATE:
- A10G GPU: ~$0.0012/second
- Average inference: 2-5 seconds
- 500 arena battles: ~$1-3

IMPORTANT:
This template uses base Qwen3-4B. For actual PRISM deployment:
1. Convert MLX LoRA weights to PEFT format, OR
2. Merge weights locally and upload to HuggingFace Hub, OR
3. Retrain with PEFT directly for cloud deployment

The cognitive patterns should transfer via either method.
"""
