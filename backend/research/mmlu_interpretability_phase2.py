#!/usr/bin/env python3
"""
MMLU Interpretability - Phase 2: Activation-Level Analysis

This module analyzes internal model activations to understand
how V9 adapters change information processing.

Hypotheses tested:
- H1: Routing Circuit Activation ("Focusing Lens")
- H5: LoRA as Cognitive Steering Vector

Requires: MLX with model inspection capabilities

Usage:
    python mmlu_interpretability_phase2.py --model qwen --prompt "What is the capital of France?"
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available")


@dataclass
class LayerActivationStats:
    """Statistics for a single layer's activations."""
    layer_idx: int
    layer_name: str
    mean: float
    std: float
    max_abs: float
    entropy: float  # Shannon entropy of activation distribution
    sparsity: float  # Fraction of near-zero activations


@dataclass
class ActivationComparison:
    """Comparison of activations between base and adapter models."""
    prompt: str
    base_stats: List[LayerActivationStats]
    adapter_stats: List[LayerActivationStats]
    entropy_reduction: List[float]  # Per layer: base_entropy - adapter_entropy
    mean_entropy_reduction: float


class ActivationExtractor:
    """Extract and analyze activations from MLX models."""

    def __init__(self, model_name: str, adapter_path: str = None):
        self.model_name = model_name
        self.adapter_path = adapter_path

        if not HAS_MLX:
            raise RuntimeError("MLX required for activation analysis")

        print(f"Loading model: {model_name}")
        if adapter_path:
            print(f"With adapters: {adapter_path}")
            self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_name)

        # Inspect model structure
        self._inspect_model()

    def _inspect_model(self):
        """Inspect model architecture to understand layer structure."""
        print("\n📊 Model Architecture:")

        # Count parameters
        total_params = sum(p.size for p in self.model.parameters().values())
        print(f"   Total parameters: {total_params:,}")

        # Get layer count
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
            print(f"   Number of layers: {self.num_layers}")

            # Inspect first layer structure
            first_layer = self.model.model.layers[0]
            print(f"   Layer structure: {type(first_layer).__name__}")

            # Check for LoRA
            if self.adapter_path:
                lora_count = 0
                for name, param in self.model.parameters().items():
                    if 'lora' in name.lower():
                        lora_count += 1
                print(f"   LoRA adapters found: {lora_count}")
        else:
            print("   Warning: Could not inspect layer structure")
            self.num_layers = 0

    def get_layer_output(self, layer_idx: int, input_ids: mx.array) -> mx.array:
        """
        Get the output of a specific layer.

        Note: This requires modifying how the model processes input
        to capture intermediate activations.
        """
        # This is a simplified approach - full implementation would
        # require hooking into the model's forward pass

        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise RuntimeError("Model structure not compatible with activation extraction")

        # Get embeddings
        if hasattr(self.model.model, 'embed_tokens'):
            h = self.model.model.embed_tokens(input_ids)
        else:
            raise RuntimeError("Could not find embedding layer")

        # Process through layers up to layer_idx
        for i, layer in enumerate(self.model.model.layers):
            if i > layer_idx:
                break
            # Note: This is a simplified forward pass
            # Real implementation needs to handle attention masks, caching, etc.
            h = layer(h)

        return h

    def compute_activation_stats(self, activations: mx.array, layer_idx: int, layer_name: str) -> LayerActivationStats:
        """Compute statistics for a set of activations."""
        # Convert to numpy for analysis
        act_np = np.array(activations)

        # Flatten to 1D for distribution analysis
        flat = act_np.flatten()

        # Basic stats
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        max_abs = float(np.max(np.abs(flat)))

        # Sparsity (fraction near zero)
        threshold = 0.01 * max_abs if max_abs > 0 else 0.01
        sparsity = float(np.mean(np.abs(flat) < threshold))

        # Entropy (discretize into bins)
        hist, _ = np.histogram(flat, bins=100, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = float(-np.sum(hist * np.log(hist + 1e-10)))

        return LayerActivationStats(
            layer_idx=layer_idx,
            layer_name=layer_name,
            mean=mean,
            std=std,
            max_abs=max_abs,
            entropy=entropy,
            sparsity=sparsity
        )

    def analyze_prompt(self, prompt: str, max_layers: int = None) -> List[LayerActivationStats]:
        """Analyze activations for a given prompt across all layers."""
        # Tokenize
        if "qwen" in self.model_name.lower():
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in self.model_name.lower():
            full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            full_prompt = prompt

        input_ids = mx.array(self.tokenizer.encode(full_prompt))

        # Get the input in the right shape
        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)

        stats_list = []
        num_layers_to_analyze = max_layers or self.num_layers

        print(f"\nAnalyzing {num_layers_to_analyze} layers...")
        for layer_idx in range(min(num_layers_to_analyze, self.num_layers)):
            try:
                activations = self.get_layer_output(layer_idx, input_ids)
                stats = self.compute_activation_stats(
                    activations,
                    layer_idx,
                    f"layer_{layer_idx}"
                )
                stats_list.append(stats)
            except Exception as e:
                print(f"   Layer {layer_idx}: Error - {e}")

        return stats_list


class LoRAWeightAnalyzer:
    """Analyze LoRA adapter weight structure."""

    def __init__(self, adapter_path: str):
        self.adapter_path = adapter_path
        self.weights = {}
        self._load_weights()

    def _load_weights(self):
        """Load LoRA weights from adapter path."""
        adapter_file = Path(self.adapter_path) / "adapters.safetensors"
        if not adapter_file.exists():
            # Try without safetensors
            adapter_file = Path(self.adapter_path)
            if adapter_file.is_file():
                pass
            else:
                print(f"Warning: Adapter file not found at {self.adapter_path}")
                return

        print(f"\nLoading LoRA weights from: {adapter_file}")

        try:
            # Try safetensors
            from safetensors import safe_open
            with safe_open(adapter_file, framework="numpy") as f:
                for key in f.keys():
                    self.weights[key] = f.get_tensor(key)
            print(f"   Loaded {len(self.weights)} weight tensors")
        except ImportError:
            print("   safetensors not available, trying MLX native")
            try:
                self.weights = mx.load(str(adapter_file))
                print(f"   Loaded {len(self.weights)} weight tensors")
            except Exception as e:
                print(f"   Error loading weights: {e}")

    def analyze_weight_structure(self) -> Dict:
        """Analyze the structure of LoRA weights."""
        if not self.weights:
            return {"error": "No weights loaded"}

        analysis = {
            "total_tensors": len(self.weights),
            "layer_analysis": [],
            "global_stats": {}
        }

        # Categorize by layer
        layers = {}
        for key, weight in self.weights.items():
            # Extract layer number from key
            import re
            match = re.search(r'layers?\.?(\d+)', key)
            if match:
                layer_num = int(match.group(1))
                if layer_num not in layers:
                    layers[layer_num] = []
                layers[layer_num].append((key, weight))

        # Analyze each layer
        all_norms = []
        for layer_num in sorted(layers.keys()):
            layer_weights = layers[layer_num]
            layer_norms = []

            for key, weight in layer_weights:
                w_np = np.array(weight) if not isinstance(weight, np.ndarray) else weight
                norm = np.linalg.norm(w_np)
                layer_norms.append(norm)
                all_norms.append(norm)

            analysis["layer_analysis"].append({
                "layer": layer_num,
                "num_tensors": len(layer_weights),
                "mean_norm": float(np.mean(layer_norms)),
                "max_norm": float(np.max(layer_norms)),
            })

        if all_norms:
            analysis["global_stats"] = {
                "mean_norm": float(np.mean(all_norms)),
                "std_norm": float(np.std(all_norms)),
                "max_norm": float(np.max(all_norms)),
                "min_norm": float(np.min(all_norms)),
            }

        return analysis

    def compute_steering_vectors(self) -> Dict:
        """
        Compute the effective steering direction of LoRA updates.

        LoRA: W_new = W_base + A @ B (where A is down-proj, B is up-proj)
        The effective change is the low-rank matrix A @ B.
        """
        if not self.weights:
            return {"error": "No weights loaded"}

        steering_analysis = {
            "layers": [],
            "cross_layer_similarity": None
        }

        # Find A and B matrices
        # Typically named something like "lora_A" and "lora_B"
        layer_steerings = {}

        for key, weight in self.weights.items():
            # This is a simplified heuristic - real implementation
            # would need to match A/B pairs correctly
            if 'lora' in key.lower():
                import re
                layer_match = re.search(r'layers?\.?(\d+)', key)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    if layer_num not in layer_steerings:
                        layer_steerings[layer_num] = []

                    w_np = np.array(weight) if not isinstance(weight, np.ndarray) else weight

                    # Compute SVD to understand the "direction" of the weight
                    try:
                        U, S, Vh = np.linalg.svd(w_np, full_matrices=False)
                        # The dominant direction is the first singular vector
                        dominant_direction = Vh[0] if len(Vh) > 0 else None
                        layer_steerings[layer_num].append({
                            "key": key,
                            "shape": w_np.shape,
                            "top_singular_value": float(S[0]) if len(S) > 0 else 0,
                            "rank_ratio": float(S[0] / np.sum(S)) if np.sum(S) > 0 else 0,
                        })
                    except Exception as e:
                        layer_steerings[layer_num].append({
                            "key": key,
                            "shape": w_np.shape,
                            "error": str(e)
                        })

        for layer_num in sorted(layer_steerings.keys()):
            steering_analysis["layers"].append({
                "layer": layer_num,
                "components": layer_steerings[layer_num]
            })

        return steering_analysis


def compare_activations(
    model_name: str,
    adapter_path: str,
    prompts: List[str],
    max_layers: int = 10
) -> List[ActivationComparison]:
    """Compare activations between base and adapter-equipped models."""

    print("\n" + "="*70)
    print("ACTIVATION COMPARISON")
    print("="*70)

    # Load base model
    print("\n📦 Loading base model...")
    base_extractor = ActivationExtractor(model_name, adapter_path=None)

    # Load adapter model
    print("\n📦 Loading adapter model...")
    adapter_extractor = ActivationExtractor(model_name, adapter_path=adapter_path)

    comparisons = []

    for prompt in prompts:
        print(f"\n🔍 Analyzing: '{prompt[:50]}...'")

        # Get base activations
        base_stats = base_extractor.analyze_prompt(prompt, max_layers=max_layers)

        # Get adapter activations
        adapter_stats = adapter_extractor.analyze_prompt(prompt, max_layers=max_layers)

        # Compute entropy reduction per layer
        entropy_reduction = []
        for b, a in zip(base_stats, adapter_stats):
            reduction = b.entropy - a.entropy
            entropy_reduction.append(reduction)

        mean_reduction = np.mean(entropy_reduction) if entropy_reduction else 0

        comparisons.append(ActivationComparison(
            prompt=prompt,
            base_stats=base_stats,
            adapter_stats=adapter_stats,
            entropy_reduction=entropy_reduction,
            mean_entropy_reduction=float(mean_reduction)
        ))

        print(f"   Mean entropy reduction: {mean_reduction:.3f}")
        print(f"   (Positive = adapter reduces entropy = more focused)")

    return comparisons


def print_comparison_results(comparisons: List[ActivationComparison]):
    """Print formatted comparison results."""
    print("\n" + "="*70)
    print("ACTIVATION ANALYSIS RESULTS")
    print("="*70)

    all_reductions = []
    for comp in comparisons:
        all_reductions.extend(comp.entropy_reduction)

    mean_overall = np.mean(all_reductions) if all_reductions else 0
    positive_fraction = np.mean([r > 0 for r in all_reductions]) if all_reductions else 0

    print(f"\n📊 Overall Statistics:")
    print(f"   Mean entropy reduction: {mean_overall:.4f}")
    print(f"   Fraction of layers with reduction: {positive_fraction*100:.1f}%")

    # H1 evaluation
    print(f"\n🔬 H1 (Focusing Lens) Evaluation:")
    if positive_fraction > 0.6 and mean_overall > 0:
        print("   SUPPORTED: Adapters reduce activation entropy")
        print("   Interpretation: The adapter acts as a 'focusing lens',")
        print("   suppressing noise and amplifying the signal.")
    elif positive_fraction < 0.4 and mean_overall < 0:
        print("   CONTRADICTED: Adapters increase activation entropy")
        print("   Interpretation: The adapter adds complexity rather than focus.")
    else:
        print("   INCONCLUSIVE: Mixed effects on entropy")
        print(f"   {positive_fraction*100:.1f}% of layers show reduction")

    # Per-layer analysis
    if comparisons and comparisons[0].entropy_reduction:
        print(f"\n📈 Per-Layer Entropy Reduction:")
        # Average across all prompts
        num_layers = len(comparisons[0].entropy_reduction)
        for i in range(num_layers):
            layer_reductions = [c.entropy_reduction[i] for c in comparisons if i < len(c.entropy_reduction)]
            mean_layer = np.mean(layer_reductions) if layer_reductions else 0
            bar = "█" * max(0, int(mean_layer * 20)) if mean_layer > 0 else "▒" * max(0, int(-mean_layer * 20))
            direction = "↓" if mean_layer > 0 else "↑"
            print(f"   Layer {i:2d}: {mean_layer:+.3f} {direction} {bar}")


def main():
    parser = argparse.ArgumentParser(description="MMLU Interpretability Phase 2")
    parser.add_argument("--model", choices=["phi4", "phi35", "qwen", "llama"], default="phi4",
                       help="Model family to test (phi4=V10.1 default)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt to analyze")
    parser.add_argument("--max-layers", type=int, default=10,
                       help="Maximum layers to analyze")
    parser.add_argument("--analyze-weights", action="store_true",
                       help="Analyze LoRA weight structure")
    args = parser.parse_args()

    # Model configurations - V10.1
    MODEL_CONFIGS = {
        "phi4": {
            "model_name": "mlx-community/phi-4-4bit",
            "adapter_path": "./mlx_adapters_v10_1_phi4/adapters",
        },
        "phi35": {
            "model_name": "mlx-community/Phi-3.5-mini-instruct-4bit",
            "adapter_path": "./mlx_adapters_v10_1_phi35/adapters",
        },
        "qwen": {
            "model_name": "mlx-community/Qwen3-8B-4bit",
            "adapter_path": "./mlx_adapters_v9_8b_4bit/adapters",  # Legacy V9
        },
        "llama": {
            "model_name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "adapter_path": "./mlx_adapters_v9_llama8b/adapters",  # Legacy V9
        },
    }

    if args.model not in MODEL_CONFIGS:
        print(f"Unknown model: {args.model}. Available: {list(MODEL_CONFIGS.keys())}")
        return

    config = MODEL_CONFIGS[args.model]
    model_name = config["model_name"]
    adapter_path = config["adapter_path"]

    # Weight analysis (doesn't require model loading)
    if args.analyze_weights:
        print("\n" + "="*70)
        print("LORA WEIGHT ANALYSIS")
        print("="*70)

        analyzer = LoRAWeightAnalyzer(adapter_path)
        structure = analyzer.analyze_weight_structure()
        steering = analyzer.compute_steering_vectors()

        print("\n📊 Weight Structure:")
        print(json.dumps(structure, indent=2))

        print("\n🧭 Steering Vectors:")
        print(json.dumps(steering, indent=2, default=str))

        # Save results
        output_dir = Path("./interpretability_results")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / f"phase2_{args.model}_weights.json", "w") as f:
            json.dump({"structure": structure, "steering": steering}, f, indent=2, default=str)
        return

    # Activation comparison
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Default test prompts
        prompts = [
            "What is the capital of France?\n\nA) London\nB) Paris\nC) Berlin\nD) Madrid\n\nAnswer with just the letter:",
            "What is 2 + 2?\n\nA) 3\nB) 4\nC) 5\nD) 6\n\nAnswer with just the letter:",
            "Who wrote Hamlet?\n\nA) Charles Dickens\nB) Jane Austen\nC) William Shakespeare\nD) Mark Twain\n\nAnswer with just the letter:",
        ]

    comparisons = compare_activations(
        model_name=model_name,
        adapter_path=adapter_path,
        prompts=prompts,
        max_layers=args.max_layers
    )

    print_comparison_results(comparisons)

    # Save results
    output_dir = Path("./interpretability_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"phase2_{args.model}_activations.json"
    with open(output_file, "w") as f:
        results = []
        for comp in comparisons:
            results.append({
                "prompt": comp.prompt,
                "entropy_reduction": comp.entropy_reduction,
                "mean_entropy_reduction": comp.mean_entropy_reduction,
                "base_stats": [asdict(s) for s in comp.base_stats],
                "adapter_stats": [asdict(s) for s in comp.adapter_stats],
            })
        json.dump(results, f, indent=2)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
