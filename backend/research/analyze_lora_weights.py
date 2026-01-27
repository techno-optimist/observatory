#!/usr/bin/env python3
"""
LoRA Weight Analysis - Mechanistic Interpretability Phase 2.1

Analyzes the structure of V10.1 LoRA adapters to understand:
1. Weight magnitude distribution across layers
2. Singular value decomposition of weight matrices
3. Cross-layer consistency of "steering directions"
4. Effective rank of the LoRA updates
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
import argparse


def analyze_adapter_weights(adapter_path: str):
    """Comprehensive analysis of LoRA adapter weights."""

    print("\n" + "="*70)
    print("LORA WEIGHT ANALYSIS - V10.1")
    print("="*70)

    adapter_file = Path(adapter_path) / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"Error: {adapter_file} not found")
        return None

    print(f"\nLoading: {adapter_file}")
    print(f"Size: {adapter_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Load weights
    weights = {}
    with safe_open(adapter_file, framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    print(f"Total tensors: {len(weights)}")

    # Categorize weights by layer and type
    layer_weights = defaultdict(dict)
    weight_types = set()

    for key, weight in weights.items():
        # Parse key like "model.layers.0.self_attn.q_proj.lora_a"
        parts = key.split(".")

        # Find layer number
        layer_num = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_num = int(parts[i + 1])
                    break
                except ValueError:
                    pass

        if layer_num is not None:
            # Extract weight type (e.g., "self_attn.q_proj.lora_a")
            weight_type = ".".join(parts[parts.index("layers") + 2:])
            layer_weights[layer_num][weight_type] = weight
            weight_types.add(weight_type)

    print(f"Layers with LoRA: {len(layer_weights)}")
    print(f"Weight types per layer: {len(weight_types)}")

    # Analyze each layer
    print("\n" + "-"*70)
    print("PER-LAYER ANALYSIS")
    print("-"*70)

    layer_stats = []

    for layer_num in sorted(layer_weights.keys()):
        layer_data = layer_weights[layer_num]

        # Compute stats for this layer
        total_params = 0
        total_norm = 0
        weight_norms = {}

        for weight_type, weight in layer_data.items():
            norm = np.linalg.norm(weight)
            weight_norms[weight_type] = norm
            total_params += weight.size
            total_norm += norm

        layer_stats.append({
            "layer": layer_num,
            "total_params": total_params,
            "total_norm": total_norm,
            "avg_norm": total_norm / len(layer_data) if layer_data else 0,
            "weight_norms": weight_norms
        })

    # Print summary
    print(f"\n{'Layer':<8} {'Params':<12} {'Total Norm':<14} {'Avg Norm':<12}")
    print("-" * 50)
    for stat in layer_stats[:10]:  # First 10 layers
        print(f"{stat['layer']:<8} {stat['total_params']:<12,} {stat['total_norm']:<14.4f} {stat['avg_norm']:<12.4f}")
    if len(layer_stats) > 10:
        print(f"... ({len(layer_stats) - 10} more layers)")

    # SVD Analysis on a sample weight matrix
    print("\n" + "-"*70)
    print("SINGULAR VALUE DECOMPOSITION ANALYSIS")
    print("-"*70)

    svd_results = []

    # Find lora_a and lora_b pairs to compute effective weight
    for layer_num in sorted(layer_weights.keys())[:5]:  # First 5 layers
        layer_data = layer_weights[layer_num]

        # Look for lora_a/lora_b pairs
        lora_pairs = defaultdict(dict)
        for weight_type, weight in layer_data.items():
            if "lora_a" in weight_type:
                base = weight_type.replace(".lora_a", "")
                lora_pairs[base]["a"] = weight
            elif "lora_b" in weight_type:
                base = weight_type.replace(".lora_b", "")
                lora_pairs[base]["b"] = weight

        for proj_name, pair in lora_pairs.items():
            if "a" in pair and "b" in pair:
                # LoRA effective weight: B @ A
                # A: (rank, in_features), B: (out_features, rank)
                A = pair["a"]
                B = pair["b"]

                # Compute effective delta weight
                try:
                    delta_W = B @ A  # (out_features, in_features)

                    # SVD of effective weight
                    U, S, Vh = np.linalg.svd(delta_W, full_matrices=False)

                    # Effective rank (how much of the update is captured by top-k singular values)
                    total_var = np.sum(S**2)
                    cumvar = np.cumsum(S**2) / total_var if total_var > 0 else np.zeros_like(S)

                    # Find rank that captures 90%, 95%, 99% of variance
                    rank_90 = np.searchsorted(cumvar, 0.90) + 1
                    rank_95 = np.searchsorted(cumvar, 0.95) + 1
                    rank_99 = np.searchsorted(cumvar, 0.99) + 1

                    svd_results.append({
                        "layer": layer_num,
                        "projection": proj_name,
                        "delta_shape": delta_W.shape,
                        "top_singular": float(S[0]) if len(S) > 0 else 0,
                        "singular_ratio": float(S[0] / np.sum(S)) if np.sum(S) > 0 else 0,
                        "rank_90": int(rank_90),
                        "rank_95": int(rank_95),
                        "rank_99": int(rank_99),
                        "effective_rank": int(np.sum(S > S[0] * 0.01)) if len(S) > 0 else 0,  # Entries > 1% of max
                    })

                except Exception as e:
                    print(f"  SVD failed for layer {layer_num} {proj_name}: {e}")

    print(f"\n{'Layer':<6} {'Projection':<20} {'Shape':<15} {'Top SV':<10} {'Eff Rank':<10} {'R90/R95/R99'}")
    print("-" * 80)
    for r in svd_results:
        print(f"{r['layer']:<6} {r['projection']:<20} {str(r['delta_shape']):<15} {r['top_singular']:<10.4f} {r['effective_rank']:<10} {r['rank_90']}/{r['rank_95']}/{r['rank_99']}")

    # Cross-layer consistency analysis
    print("\n" + "-"*70)
    print("CROSS-LAYER STEERING DIRECTION CONSISTENCY")
    print("-"*70)

    # Extract dominant direction from each layer's q_proj (as example)
    dominant_directions = []

    for layer_num in sorted(layer_weights.keys()):
        layer_data = layer_weights[layer_num]

        # Find q_proj lora weights
        lora_a = None
        lora_b = None
        for weight_type, weight in layer_data.items():
            if "q_proj.lora_a" in weight_type:
                lora_a = weight
            elif "q_proj.lora_b" in weight_type:
                lora_b = weight

        if lora_a is not None and lora_b is not None:
            try:
                delta_W = lora_b @ lora_a
                U, S, Vh = np.linalg.svd(delta_W, full_matrices=False)

                # Dominant direction is first right singular vector
                if len(Vh) > 0:
                    dominant_directions.append({
                        "layer": layer_num,
                        "direction": Vh[0],
                        "strength": float(S[0])
                    })
            except:
                pass

    # Compute cosine similarity between adjacent layers
    if len(dominant_directions) >= 2:
        print(f"\nCosine similarity of dominant q_proj direction between adjacent layers:")
        similarities = []
        for i in range(len(dominant_directions) - 1):
            d1 = dominant_directions[i]["direction"]
            d2 = dominant_directions[i + 1]["direction"]

            # Truncate/pad to same length
            min_len = min(len(d1), len(d2))
            d1 = d1[:min_len]
            d2 = d2[:min_len]

            cos_sim = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
            similarities.append(cos_sim)

            if i < 10:  # Print first 10
                l1 = dominant_directions[i]["layer"]
                l2 = dominant_directions[i + 1]["layer"]
                print(f"  Layer {l1} → {l2}: {cos_sim:.4f}")

        print(f"\n  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Std similarity:  {np.std(similarities):.4f}")
        print(f"  Min similarity:  {np.min(similarities):.4f}")
        print(f"  Max similarity:  {np.max(similarities):.4f}")

        # Interpretation
        mean_sim = np.mean(similarities)
        if mean_sim > 0.7:
            print(f"\n  ⚡ HIGH CONSISTENCY: Adapters push in similar direction across layers")
            print(f"     This supports H5 (Steering Vector hypothesis)")
        elif mean_sim > 0.3:
            print(f"\n  📊 MODERATE CONSISTENCY: Some alignment, but layer-specific tuning")
        else:
            print(f"\n  🔀 LOW CONSISTENCY: Each layer learns different directions")
            print(f"     Adapters may be doing layer-specific corrections")

    # Global statistics
    print("\n" + "-"*70)
    print("GLOBAL STATISTICS")
    print("-"*70)

    all_norms = [stat["total_norm"] for stat in layer_stats]
    all_params = sum(stat["total_params"] for stat in layer_stats)

    print(f"Total LoRA parameters: {all_params:,}")
    print(f"Total layers: {len(layer_stats)}")
    print(f"Mean layer norm: {np.mean(all_norms):.4f}")
    print(f"Std layer norm: {np.std(all_norms):.4f}")
    print(f"Max layer norm: {np.max(all_norms):.4f} (layer {np.argmax(all_norms)})")
    print(f"Min layer norm: {np.min(all_norms):.4f} (layer {np.argmin(all_norms)})")

    # Norm distribution across layers
    print(f"\nNorm distribution (visual):")
    max_norm = max(all_norms)
    for i, stat in enumerate(layer_stats):
        bar_len = int(40 * stat["total_norm"] / max_norm) if max_norm > 0 else 0
        bar = "█" * bar_len
        print(f"  L{stat['layer']:02d}: {bar}")

    # Save results
    similarities = []  # Initialize in case no cross-layer analysis was done
    results = {
        "adapter_path": str(adapter_path),
        "total_params": all_params,
        "num_layers": len(layer_stats),
        "layer_stats": layer_stats,
        "svd_results": svd_results,
        "cross_layer_similarity": {
            "mean": float(np.mean(similarities)) if len(similarities) > 0 else None,
            "std": float(np.std(similarities)) if len(similarities) > 0 else None,
        }
    }

    output_file = Path("./interpretability_results/lora_weight_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default="./mlx_adapters_v10_1_phi4/adapters",
                       help="Path to adapter directory")
    args = parser.parse_args()

    analyze_adapter_weights(args.adapter_path)
