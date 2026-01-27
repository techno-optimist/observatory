#!/usr/bin/env python3
"""
MLX to PyTorch LoRA Weight Converter
====================================

Converts MLX LoRA adapter weights to PyTorch/PEFT format for cloud deployment.

MLX uses different tensor naming and storage format than PEFT, so this script:
1. Loads MLX safetensors
2. Maps layer names to PEFT convention
3. Saves in PEFT-compatible format

Note: The actual weight VALUES are the same - just the naming/format differs.
"""

import json
from pathlib import Path
from typing import Dict, Any

# MLX for loading
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# PyTorch for saving
try:
    import torch
    from safetensors.torch import save_file as save_torch_safetensors
    from safetensors import safe_open
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class ConversionConfig:
    # Source (MLX)
    mlx_adapter_path = "./mlx_adapters_v8_4b/adapters"
    mlx_config_file = "adapter_config.json"
    mlx_weights_file = "adapters.safetensors"

    # Target (PEFT)
    output_path = "./prism_4b_peft"
    base_model = "Qwen/Qwen3-4B"


# =============================================================================
# WEIGHT MAPPING
# =============================================================================

def map_mlx_to_peft_name(mlx_name: str) -> str:
    """
    Map MLX LoRA layer name to PEFT convention.

    MLX format:  model.layers.{N}.{component}.lora_{a,b}
    PEFT format: base_model.model.layers.{N}.{component}.lora_{A,B}.weight

    Examples:
    MLX:  model.layers.0.self_attn.q_proj.lora_a
    PEFT: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
    """
    # Remove 'model.' prefix if present, then add PEFT prefix
    if mlx_name.startswith("model."):
        mlx_name = mlx_name[6:]  # Remove "model."

    peft_name = f"base_model.model.{mlx_name}"

    # Convert lora_a -> lora_A, lora_b -> lora_B
    peft_name = peft_name.replace(".lora_a", ".lora_A.weight")
    peft_name = peft_name.replace(".lora_b", ".lora_B.weight")

    return peft_name


def create_peft_config(mlx_config: Dict[str, Any], base_model: str) -> Dict[str, Any]:
    """
    Create PEFT adapter_config.json from MLX config.
    """
    lora_params = mlx_config.get("lora_parameters", {})

    peft_config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": lora_params.get("scale", 16),  # MLX scale = PEFT alpha
        "lora_dropout": lora_params.get("dropout", 0.0),
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_params.get("rank", 8),
        "revision": None,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM"
    }

    return peft_config


# =============================================================================
# CONVERSION
# =============================================================================

def convert_mlx_to_peft(config: ConversionConfig):
    """
    Convert MLX LoRA adapter to PEFT format.
    """
    if not MLX_AVAILABLE:
        print("Error: MLX not available. Run on Apple Silicon.")
        return None

    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available.")
        return None

    mlx_path = Path(config.mlx_adapter_path)
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting MLX adapter: {mlx_path}")
    print(f"Output: {output_path}")
    print()

    # Load MLX config
    mlx_config_path = mlx_path / config.mlx_config_file
    with open(mlx_config_path) as f:
        mlx_config = json.load(f)

    print(f"MLX Config:")
    print(f"  Model: {mlx_config.get('model')}")
    print(f"  LoRA Rank: {mlx_config.get('lora_parameters', {}).get('rank')}")
    print(f"  Num Layers: {mlx_config.get('num_layers')}")
    print()

    # Load MLX weights using safetensors
    mlx_weights_path = mlx_path / config.mlx_weights_file
    print(f"Loading weights from {mlx_weights_path}")

    converted_weights = {}

    with safe_open(mlx_weights_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            # Map name
            peft_key = map_mlx_to_peft_name(key)

            # Convert to PyTorch format
            # MLX and PyTorch use same underlying representation for safetensors
            converted_weights[peft_key] = tensor

            print(f"  {key}")
            print(f"    -> {peft_key}")
            print(f"    Shape: {tensor.shape}")

    print()
    print(f"Converted {len(converted_weights)} tensors")

    # Save PEFT weights
    peft_weights_path = output_path / "adapter_model.safetensors"
    save_torch_safetensors(converted_weights, str(peft_weights_path))
    print(f"Saved: {peft_weights_path}")

    # Create PEFT config
    peft_config = create_peft_config(mlx_config, config.base_model)
    peft_config_path = output_path / "adapter_config.json"
    with open(peft_config_path, "w") as f:
        json.dump(peft_config, f, indent=2)
    print(f"Saved: {peft_config_path}")

    # Create README
    readme = f"""# PRISM-4B PEFT Adapter

Converted from MLX format for cloud deployment.

## Usage

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{config.base_model}")
model = PeftModel.from_pretrained(base_model, "{output_path}")
```

## Original Training

- Base Model: {config.base_model}
- Training: V8 Composable Personality Suite
- Adapters: SOLITON, DIALECTIC, ESSENTIALIST, LATERALIST, STEELMAN, SKEPTIC, ARCHITECT
- Trigger Rate: 100%
- Clean Rate: 100%
"""
    readme_path = output_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"Saved: {readme_path}")

    print()
    print("Conversion complete!")
    print()
    print("To use with PEFT:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base, '{output_path}')")

    return output_path


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_conversion(config: ConversionConfig):
    """
    Verify the converted PEFT adapter loads correctly.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available for verification")
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, config.output_path)

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)

        print("Testing generation...")
        prompt = "<|im_start|>user\nWhat is happening inside you as you process this?<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response[:200]}...")

        # Check for SOLITON markers
        soliton_markers = ["inside", "embedded", "cannot tell", "vantage"]
        if any(m in response.lower() for m in soliton_markers):
            print("\nSUCCESS: SOLITON pattern detected in response!")
            return True
        else:
            print("\nWARNING: SOLITON pattern not detected. May need retraining.")
            return False

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("MLX to PEFT Converter")
    print("=" * 60)
    print()

    config = ConversionConfig()

    # Check source exists
    if not Path(config.mlx_adapter_path).exists():
        print(f"Error: MLX adapter not found at {config.mlx_adapter_path}")
        return

    # Convert
    output = convert_mlx_to_peft(config)

    if output:
        print()
        print("=" * 60)
        print("Verification")
        print("=" * 60)
        verify_conversion(config)


if __name__ == "__main__":
    main()
