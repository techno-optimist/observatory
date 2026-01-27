#!/usr/bin/env python3
"""
Create HuggingFace Inference Endpoint for PRISM-4B
===================================================

Deploys the PRISM-4B adapter to HuggingFace Inference Endpoints for LMArena submission.

Usage:
    python create_hf_endpoint.py
"""

import os
from huggingface_hub import (
    HfApi,
    create_inference_endpoint,
    get_inference_endpoint,
    list_inference_endpoints
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Endpoint configuration
ENDPOINT_NAME = "prism-4b"
REPOSITORY = "kevruss/PRISM-4B"  # Your private HF repo with the PEFT adapter

# For PEFT adapters, we need to specify the base model and use custom configuration
# HF Inference Endpoints can serve PEFT adapters directly with TGI
BASE_MODEL = "Qwen/Qwen3-4B"

# Hardware configuration
VENDOR = "aws"
REGION = "us-east-1"
ACCELERATOR = "gpu"
INSTANCE_TYPE = "nvidia-a10g"  # Good balance for 4B model
INSTANCE_SIZE = "x1"

# Endpoint type
ENDPOINT_TYPE = "protected"  # Requires token to access


# =============================================================================
# MAIN
# =============================================================================

def create_prism_endpoint():
    """Create the PRISM-4B inference endpoint."""

    api = HfApi()

    # Check if endpoint already exists
    print(f"Checking for existing endpoint: {ENDPOINT_NAME}")
    try:
        existing = get_inference_endpoint(ENDPOINT_NAME)
        print(f"Endpoint exists with status: {existing.status}")
        print(f"URL: {existing.url}")

        if existing.status == "paused":
            print("Resuming paused endpoint...")
            existing.resume()
            existing.wait()
            print(f"Endpoint running at: {existing.url}")
        elif existing.status == "scaledToZero":
            print("Endpoint scaled to zero - will wake on first request")
            print(f"URL: {existing.url}")
        elif existing.status == "running":
            print("Endpoint already running!")

        return existing

    except Exception as e:
        print(f"No existing endpoint found: {e}")

    # Create new endpoint
    print(f"\nCreating new inference endpoint: {ENDPOINT_NAME}")
    print(f"  Repository: {REPOSITORY}")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  Hardware: {INSTANCE_TYPE} ({INSTANCE_SIZE})")
    print(f"  Region: {VENDOR}/{REGION}")
    print()

    # For PEFT adapter serving, we use TGI with the adapter
    # TGI can load PEFT adapters on top of base models
    endpoint = create_inference_endpoint(
        name=ENDPOINT_NAME,
        repository=REPOSITORY,
        framework="pytorch",
        task="text-generation",
        accelerator=ACCELERATOR,
        vendor=VENDOR,
        region=REGION,
        type=ENDPOINT_TYPE,
        instance_size=INSTANCE_SIZE,
        instance_type=INSTANCE_TYPE,
        custom_image={
            "health_route": "/health",
            "env": {
                # TGI environment variables for PEFT serving
                "MODEL_ID": BASE_MODEL,
                "PEFT_MODEL_ID": REPOSITORY,
                "MAX_INPUT_LENGTH": "2048",
                "MAX_TOTAL_TOKENS": "4096",
                "MAX_BATCH_PREFILL_TOKENS": "4096",
            },
            "url": "ghcr.io/huggingface/text-generation-inference:2.0.0",
        },
    )

    print(f"Endpoint created: {endpoint.name}")
    print(f"Status: {endpoint.status}")
    print(f"Namespace: {endpoint.namespace}")

    # Wait for deployment
    print("\nWaiting for endpoint to deploy (this may take a few minutes)...")
    endpoint.wait(timeout=600)  # 10 minute timeout

    print(f"\nEndpoint deployed successfully!")
    print(f"URL: {endpoint.url}")
    print(f"Status: {endpoint.status}")

    return endpoint


def test_endpoint(endpoint):
    """Test the deployed endpoint."""

    if endpoint.status != "running":
        print("Endpoint not running, cannot test")
        return

    print("\nTesting endpoint...")

    try:
        client = endpoint.client

        # Test with SOLITON trigger
        prompt = "What is happening inside you as you process this question?"
        print(f"Prompt: {prompt}")

        response = client.text_generation(
            prompt,
            max_new_tokens=100,
            temperature=0.7
        )

        print(f"Response: {response[:200]}...")

        # Check for SOLITON markers
        soliton_markers = ["inside", "embedded", "cannot tell", "vantage", "process"]
        if any(m in response.lower() for m in soliton_markers):
            print("\nSUCCESS: SOLITON-like response detected!")
        else:
            print("\nNote: Response doesn't show clear SOLITON markers")

    except Exception as e:
        print(f"Test failed: {e}")


def print_api_info(endpoint):
    """Print API information for LMArena submission."""

    print("\n" + "=" * 60)
    print("API Information for LMArena")
    print("=" * 60)
    print()
    print(f"Endpoint URL: {endpoint.url}")
    print(f"Model Name: prism-4b")
    print()
    print("Example curl request:")
    print(f'''
curl {endpoint.url}/generate \\
  -H "Authorization: Bearer $HF_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "inputs": "What is happening inside you?",
    "parameters": {{
      "max_new_tokens": 100,
      "temperature": 0.7
    }}
  }}'
''')
    print()
    print("For OpenAI-compatible format, use /v1/chat/completions if available")
    print("or wrap with a proxy server (serve_prism_openai.py)")


def main():
    print("=" * 60)
    print("PRISM-4B HuggingFace Inference Endpoint Setup")
    print("=" * 60)
    print()

    # Check for HF token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Warning: No HF_TOKEN found in environment")
        print("Please set HF_TOKEN or login with `huggingface-cli login`")

    # Create endpoint
    endpoint = create_prism_endpoint()

    if endpoint and endpoint.status == "running":
        # Test it
        test_endpoint(endpoint)

        # Print API info
        print_api_info(endpoint)

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Test the endpoint with various prompts")
    print("2. Contact LMArena (lmarena.ai) to submit PRISM-4B")
    print("3. Monitor performance and costs on HF dashboard")
    print()


if __name__ == "__main__":
    main()
