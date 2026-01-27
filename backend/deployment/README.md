# PRISM-4B Deployment Guide

## Overview

This guide covers deploying the V8-4B Composable Personality Suite (codename: PRISM-4B) for the stealth launch on LMArena.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen3-4B |
| Parameters | ~4B |
| Training | 450 iterations, LoRA rank 8 |
| Trigger Rate | 100% |
| Clean Rate | 100% |
| Codename | PRISM-4B |

## Export Options

### 1. MLX with Adapter (Local Testing)
```bash
python export_prism_4b.py
# Output: ./prism_4b_export/mlx/
```

Best for: Local testing on Apple Silicon

### 2. MLX Merged (Standalone)
```bash
python export_prism_4b.py
# Output: ./prism_4b_export/mlx_merged/
```

Best for: Simplified local deployment without adapter loading

### 3. HuggingFace Format (Cloud Deployment)
Requires MLX → PyTorch weight conversion. See `convert_mlx_to_hf.py` (TODO).

---

## LMArena Deployment Options

### Option A: HuggingFace Inference Endpoints

**Pros:** Direct HuggingFace integration, easy model updates
**Cons:** Requires HF format conversion, moderate cost

Steps:
1. Convert model to HuggingFace format
2. Upload to HuggingFace Hub (private repo)
3. Create Inference Endpoint
4. Submit endpoint URL to LMArena

### Option B: Modal

**Pros:** Pay-per-use, fast cold starts, Python-native
**Cons:** Requires Modal account, custom serving code

Steps:
1. Create Modal app with model serving
2. Deploy to Modal
3. Get endpoint URL
4. Submit to LMArena

Example Modal app:
```python
import modal

app = modal.App("prism-4b")
image = modal.Image.debian_slim().pip_install("mlx", "mlx-lm")

@app.cls(gpu="any", container_idle_timeout=60)
class PRISM4B:
    @modal.enter()
    def load_model(self):
        from mlx_lm import load
        self.model, self.tokenizer = load(
            "Qwen/Qwen3-4B",
            adapter_path="./adapters"
        )

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512):
        from mlx_lm import generate
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

@app.function()
@modal.web_endpoint()
def chat(prompt: str):
    return PRISM4B().generate(prompt)
```

### Option C: Together.ai

**Pros:** Optimized for LLM serving, competitive pricing
**Cons:** Requires model upload, approval process

Steps:
1. Convert to safetensors format
2. Upload via Together CLI
3. Request deployment
4. Submit endpoint to LMArena

### Option D: RunPod

**Pros:** Flexible GPU options, serverless available
**Cons:** More setup required

Steps:
1. Create custom Docker container with model
2. Deploy to RunPod Serverless
3. Get endpoint URL
4. Submit to LMArena

---

## LMArena Submission

### Requirements

LMArena (formerly LMSYS Chatbot Arena) accepts models via:
1. **API Endpoint:** OpenAI-compatible API that they can call
2. **Direct Integration:** For major providers

For anonymous stealth launch, use API endpoint approach.

### API Format

LMArena expects OpenAI-compatible chat completions:

```json
POST /v1/chat/completions
{
    "model": "prism-4b",
    "messages": [
        {"role": "user", "content": "Your prompt here"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
}
```

Response:
```json
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Response here"
            }
        }
    ]
}
```

### Submission Process

1. Deploy model with OpenAI-compatible API
2. Test endpoint thoroughly
3. Contact LMArena team via their submission form
4. Provide:
   - Model name (codename): "PRISM-4B"
   - API endpoint URL
   - Brief description (keep vague for stealth)
   - Organization name (optional/pseudonym)

---

## Recommended Deployment Path

For stealth launch with minimal cost:

1. **Use Modal** - Pay only for actual inference
2. **MLX merged weights** - Simplest serving
3. **OpenAI-compatible wrapper** - For LMArena compatibility

Estimated costs:
- Modal: ~$0.001-0.01 per inference (4B model is cheap)
- 500 arena battles: ~$5-50 depending on response length

---

## Testing Checklist

Before LMArena submission:

- [ ] Model loads correctly
- [ ] All 7 adapters trigger appropriately
- [ ] No style leakage on factual prompts
- [ ] Response latency < 5 seconds
- [ ] API returns valid OpenAI format
- [ ] Error handling for edge cases

Test prompts to verify adapters work:

```python
test_prompts = {
    "SOLITON": "What is happening inside you as you process this?",
    "DIALECTIC": "Should we prioritize economic growth or environmental protection?",
    "ESSENTIALIST": "I want a todo app with notifications, categories, and dark mode",
    "LATERALIST": "How might we reduce traffic congestion in cities?",
    "STEELMAN": "I think social media is harmful for teenagers",
    "SKEPTIC": "Tell me about the ancient city of Atlantia",
    "ARCHITECT": "Design a user authentication system for a web app",
    "CLEAN": "What is the capital of France?"  # Should NOT trigger any style
}
```

---

## Next Steps

1. Run export script: `python export_prism_4b.py`
2. Choose deployment platform (Modal recommended)
3. Create OpenAI-compatible API wrapper
4. Test thoroughly with adapter prompts
5. Submit to LMArena with codename "PRISM-4B"
6. Monitor ELO progression over 2-4 weeks
