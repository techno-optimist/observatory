#!/usr/bin/env python3
"""
PRISM-4B OpenAI-Compatible API Server
=====================================

Serves the V8-4B Composable Personality Suite with an OpenAI-compatible API.
Compatible with LMArena, ChatGPT interfaces, and standard LLM tooling.

Usage:
    python serve_prism_openai.py  # Local server on port 8000

For Modal deployment:
    modal deploy serve_prism_modal.py
"""

import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# FastAPI for serving
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Run: pip install fastapi uvicorn")

# MLX for inference
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ServerConfig:
    model_name: str = "Qwen/Qwen3-4B"
    adapter_path: str = "./mlx_adapters_v8_4b/adapters"
    port: int = 8000
    host: str = "0.0.0.0"
    max_tokens_default: int = 512
    temperature_default: float = 0.7

config = ServerConfig()


# =============================================================================
# OPENAI-COMPATIBLE MODELS
# =============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "prism-4b"
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# =============================================================================
# MODEL LOADER
# =============================================================================

class PRISMModel:
    """Singleton model loader for PRISM-4B."""

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self._model is None:
            print(f"Loading PRISM-4B...")
            print(f"  Base: {config.model_name}")
            print(f"  Adapter: {config.adapter_path}")

            self._model, self._tokenizer = load(
                config.model_name,
                adapter_path=config.adapter_path
            )
            print("Model loaded successfully.")

        return self._model, self._tokenizer

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        model, tokenizer = self.load()

        # Format for Qwen chat
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            temp=temperature
        )

        # Clean up response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        return response.strip()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="PRISM-4B API",
        description="OpenAI-compatible API for the V8 Composable Personality Suite",
        version="1.0.0"
    )

    # CORS for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Model instance
    prism = PRISMModel()


    @app.on_event("startup")
    async def startup_event():
        """Pre-load model on startup."""
        if MLX_AVAILABLE:
            prism.load()
        else:
            print("Warning: MLX not available. Model loading disabled.")


    @app.get("/")
    async def root():
        return {
            "model": "PRISM-4B",
            "version": "V8",
            "adapters": ["SOLITON", "DIALECTIC", "ESSENTIALIST", "LATERALIST",
                        "STEELMAN", "SKEPTIC", "ARCHITECT"],
            "status": "ready"
        }


    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible model listing."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "prism-4b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "cultural-soliton-observatory"
                }
            ]
        }


    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        """OpenAI-compatible chat completions endpoint."""

        if not MLX_AVAILABLE:
            raise HTTPException(status_code=503, detail="MLX not available")

        # Extract the last user message (simple approach)
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message provided")

        # Build conversation context
        conversation = ""
        for msg in request.messages:
            if msg.role == "system":
                conversation += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"

        # Use last user message as primary prompt
        prompt = user_messages[-1].content
        if len(request.messages) > 1:
            # Include context for multi-turn
            prompt = conversation + f"\nUser: {prompt}"

        # Generation parameters
        max_tokens = request.max_tokens or config.max_tokens_default
        temperature = request.temperature or config.temperature_default

        # Generate response
        start_time = time.time()
        response_text = prism.generate(prompt, max_tokens, temperature)
        generation_time = time.time() - start_time

        # Estimate token counts (rough)
        prompt_tokens = len(prompt.split()) * 1.3  # ~1.3 tokens per word
        completion_tokens = len(response_text.split()) * 1.3

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model="prism-4b",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens)
            )
        )


    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": prism._model is not None,
            "mlx_available": MLX_AVAILABLE
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI required. Run: pip install fastapi uvicorn")
        return

    if not MLX_AVAILABLE:
        print("Error: MLX required for inference.")
        return

    print("=" * 60)
    print("PRISM-4B OpenAI-Compatible Server")
    print("=" * 60)
    print()
    print(f"Model: {config.model_name}")
    print(f"Adapter: {config.adapter_path}")
    print(f"Endpoint: http://{config.host}:{config.port}")
    print()
    print("API Endpoints:")
    print("  GET  /              - Model info")
    print("  GET  /v1/models     - List models")
    print("  POST /v1/chat/completions - Chat completions")
    print("  GET  /health        - Health check")
    print()

    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
