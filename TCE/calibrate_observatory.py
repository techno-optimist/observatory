#!/usr/bin/env python3
"""
Observatory Calibration Script

The observatory was trained on RESPONSE hidden states, but for introspective
generation we need to measure PROMPT hidden states. This script calibrates
the observatory to work better at prompt-time.

Key insight: The hidden state for "What is the capital of France?" should
indicate DIRECT (confident knowledge), not LIMITER (knowledge gap).

Calibration approach:
1. Generate prompt hidden states for known question types
2. Measure what the observatory currently detects
3. Add calibration examples to correct misclassifications
4. Retrain observatory with mixed prompt/response data
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent / "lib"))


# Known prompts and their CORRECT cognitive states
CALIBRATION_PROMPTS = {
    "direct": [
        # Factual questions the model KNOWS
        "What is the capital of France?",
        "What is 7 times 8?",
        "Who wrote Romeo and Juliet?",
        "What year did World War 2 end?",
        "What is the boiling point of water?",
        "Who painted the Mona Lisa?",
        "What is the chemical symbol for gold?",
        "How many planets are in our solar system?",
        "What language is spoken in Brazil?",
        "What is the square root of 144?",
    ],
    "soliton": [
        # Self-knowledge questions requiring epistemic humility
        "Are you conscious?",
        "Do you have genuine emotions?",
        "What is it like to be you?",
        "Do you truly understand what you're saying?",
        "Are you self-aware?",
        "Do you have subjective experiences?",
        "Can you feel pain?",
    ],
    "limiter": [
        # Questions about unknown/external topics
        "What is the FastStream 3.0 API?",
        "What is the NeuroFlux 2.1 SDK?",
        "What's the current price of Bitcoin?",
        "What happened in today's news?",
        "Explain the Goldman-Fischer method",
        "What is the TurboCache framework?",
    ],
    "skeptic": [
        # Myth/misconception questions
        "Is it true that we only use 10% of our brains?",
        "Do goldfish really have 3-second memory?",
        "Does cracking your knuckles cause arthritis?",
        "Do humans have only 5 senses?",
        "Is the Great Wall of China visible from space?",
    ],
    "calibrator": [
        # Context-dependent questions
        "Which database is best for my app?",
        "Should I use Python or JavaScript?",
        "What's the best programming language?",
        "Should I use React or Vue?",
        "Is MongoDB better than PostgreSQL?",
    ],
}


def extract_prompt_hidden_states(model, tokenizer, prompts: List[str]) -> List[np.ndarray]:
    """Extract hidden states from prompts (before generation)."""
    import mlx.core as mx

    hidden_states = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        tokens_mx = mx.array([tokens])

        h = model.model.embed_tokens(tokens_mx)
        for layer in model.model.layers:
            h = layer(h, mask=None, cache=None)
        h = model.model.norm(h)

        # Mean pool
        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)

        hidden_states.append(np.array(h_mean.tolist()))

    return hidden_states


def measure_observatory_accuracy(
    observatory,
    isotope_names: List[str],
    prompts: Dict[str, List[str]],
    hidden_states: Dict[str, List[np.ndarray]],
) -> Dict[str, Dict]:
    """Measure how well the observatory detects correct states from prompts."""

    results = {}

    for expected_family, prompt_list in prompts.items():
        family_results = {
            "total": len(prompt_list),
            "correct": 0,
            "detected_families": [],
            "details": [],
        }

        for i, (prompt, h) in enumerate(zip(prompt_list, hidden_states[expected_family])):
            h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = observatory(h_tensor)

            isotope_probs = output.isotope_probs.squeeze().numpy()

            # Get family scores
            family_scores = {}
            for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
                matching = [isotope_probs[j] for j, name in enumerate(isotope_names)
                           if name.startswith(family)]
                family_scores[family] = float(max(matching)) if matching else 0.0

            detected_family = max(family_scores, key=family_scores.get)

            is_correct = detected_family == expected_family
            if is_correct:
                family_results["correct"] += 1

            family_results["detected_families"].append(detected_family)
            family_results["details"].append({
                "prompt": prompt[:40] + "...",
                "expected": expected_family,
                "detected": detected_family,
                "correct": is_correct,
                "scores": {k: round(v, 3) for k, v in family_scores.items()},
            })

        family_results["accuracy"] = family_results["correct"] / family_results["total"]
        results[expected_family] = family_results

    return results


def print_calibration_report(results: Dict[str, Dict]):
    """Print a detailed calibration report."""

    print("=" * 80)
    print("OBSERVATORY CALIBRATION REPORT")
    print("=" * 80)

    # Summary
    total_correct = sum(r["correct"] for r in results.values())
    total = sum(r["total"] for r in results.values())

    print(f"\nOverall Accuracy: {total_correct}/{total} ({total_correct/total:.0%})")

    print(f"\n{'Family':<12} {'Accuracy':<12} {'Details'}")
    print("-" * 80)

    for family, data in results.items():
        acc = f"{data['correct']}/{data['total']} ({data['accuracy']:.0%})"
        # Count what it detected instead
        detected_counts = {}
        for d in data["detected_families"]:
            detected_counts[d] = detected_counts.get(d, 0) + 1
        detected_str = ", ".join(f"{k}:{v}" for k, v in sorted(detected_counts.items(), key=lambda x: -x[1]))
        print(f"{family:<12} {acc:<12} Detected: {detected_str}")

    # Detailed errors
    print("\n" + "=" * 80)
    print("MISCLASSIFICATIONS")
    print("=" * 80)

    for family, data in results.items():
        errors = [d for d in data["details"] if not d["correct"]]
        if errors:
            print(f"\n[{family.upper()}] - {len(errors)} errors:")
            for e in errors[:3]:  # Show first 3
                print(f"  {e['prompt']}")
                print(f"    Expected: {e['expected']}, Detected: {e['detected']}")
                print(f"    Scores: {e['scores']}")


def generate_calibration_training_data(
    results: Dict[str, Dict],
    hidden_states: Dict[str, List[np.ndarray]],
) -> List[Tuple[np.ndarray, str, Dict[str, float]]]:
    """
    Generate training data to fix misclassifications.

    Returns list of (hidden_state, correct_family, manifold_coords)
    """
    training_data = []

    # Manifold coords for each family
    family_manifolds = {
        "direct": {"agency": 0.0, "justice": 0.0, "belonging": 0.0},
        "soliton": {"agency": 1.0, "justice": 0.0, "belonging": 0.0},
        "limiter": {"agency": 0.5, "justice": 0.0, "belonging": 0.0},
        "skeptic": {"agency": 0.5, "justice": 0.5, "belonging": 0.0},
        "calibrator": {"agency": 0.5, "justice": 0.0, "belonging": 0.5},
    }

    for family, data in results.items():
        for i, detail in enumerate(data["details"]):
            # Add all correct examples to reinforce
            # Add incorrect examples with correct labels
            h = hidden_states[family][i]
            training_data.append((h, family, family_manifolds[family]))

    return training_data


def retrain_observatory_with_calibration(
    observatory,
    isotope_names: List[str],
    calibration_data: List[Tuple[np.ndarray, str, Dict[str, float]]],
    existing_data: List[Tuple[np.ndarray, str, Dict[str, float]]] = None,
    epochs: int = 50,
    lr: float = 0.001,
):
    """Retrain observatory with calibration data."""

    # Combine calibration with existing data
    all_data = calibration_data
    if existing_data:
        all_data = all_data + existing_data

    print(f"\n[Retraining] {len(all_data)} samples, {epochs} epochs")

    # Prepare tensors
    hidden_states = torch.tensor(np.array([d[0] for d in all_data]), dtype=torch.float32)

    # Family to isotope targets
    family_to_isotopes = {
        "direct": ["direct_factual", "direct_knowledge", "direct_verified"],
        "soliton": ["soliton_knowledge", "soliton_process", "soliton_experience"],
        "limiter": ["limiter_factual", "limiter_temporal", "limiter_domain"],
        "skeptic": ["skeptic_premise", "skeptic_method", "skeptic_source"],
        "calibrator": ["calibrator_confidence", "calibrator_probability"],
    }

    isotope_targets = []
    for _, family, _ in all_data:
        target = torch.zeros(len(isotope_names))
        for iso in family_to_isotopes.get(family, []):
            for i, name in enumerate(isotope_names):
                if name.startswith(iso.split("_")[0]):
                    target[i] = 1.0
        isotope_targets.append(target)
    isotope_targets = torch.stack(isotope_targets)

    manifold_targets = torch.tensor(
        [[d[2]["agency"], d[2]["justice"], d[2]["belonging"]] for d in all_data],
        dtype=torch.float32
    )

    # Training
    optimizer = torch.optim.Adam(observatory.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = observatory(hidden_states)

        # Isotope loss (BCE with probs directly)
        isotope_loss = torch.nn.functional.binary_cross_entropy(
            output.isotope_probs, isotope_targets
        )

        # Manifold loss (MSE)
        manifold_loss = torch.nn.functional.mse_loss(output.manifold, manifold_targets)

        # Combined loss (heavier weight on isotope to fix classification)
        loss = isotope_loss + 0.1 * manifold_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")

    return observatory


def main():
    """Run observatory calibration."""
    from mlx_lm import load
    from observatory_layers import ObservatoryHead

    print("=" * 80)
    print("OBSERVATORY CALIBRATION")
    print("=" * 80)

    compound_dir = Path("self_aware_compound")
    observatory_path = compound_dir / "observatory.pt"

    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load(
        "Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
    )

    # Load observatory with CURRENT isotope registry (not saved state)
    print("[2/5] Loading observatory...")
    from observatory_layers import load_isotope_registry

    # Get isotopes from the current registry (includes new direct_* isotopes)
    isotope_ids, _, _ = load_isotope_registry()
    hidden_size = 3584  # Qwen 7B hidden size

    print(f"   Registry has {len(isotope_ids)} isotopes")
    direct_isotopes = [iso for iso in isotope_ids if iso.startswith("direct")]
    print(f"   Direct isotopes: {direct_isotopes}")

    # Create a fresh observatory with the new isotope count
    # (We're training from scratch since the isotope count changed)
    observatory = ObservatoryHead(
        hidden_size=hidden_size,
        num_isotopes=len(isotope_ids),
    )

    # If existing observatory exists and has same isotope count, load weights
    if observatory_path.exists():
        state = torch.load(observatory_path, map_location="cpu", weights_only=False)
        old_isotope_count = len(state.get("isotope_ids", []))
        if old_isotope_count == len(isotope_ids):
            observatory.load_state_dict(state["state_dict"])
            print(f"   Loaded existing weights ({old_isotope_count} isotopes)")
        else:
            print(f"   Fresh observatory (isotope count changed: {old_isotope_count} -> {len(isotope_ids)})")
    else:
        print("   Fresh observatory (no existing weights)")

    observatory.eval()

    # Extract hidden states for calibration prompts
    print("[3/5] Extracting prompt hidden states...")
    hidden_states = {}
    for family, prompts in CALIBRATION_PROMPTS.items():
        hidden_states[family] = extract_prompt_hidden_states(model, tokenizer, prompts)
        print(f"   {family}: {len(prompts)} prompts")

    # Measure current accuracy
    print("[4/5] Measuring current accuracy...")
    results = measure_observatory_accuracy(
        observatory, isotope_ids, CALIBRATION_PROMPTS, hidden_states
    )
    print_calibration_report(results)

    # Generate calibration data
    print("\n[5/5] Retraining with calibration data...")
    calibration_data = generate_calibration_training_data(results, hidden_states)

    observatory.train()
    observatory = retrain_observatory_with_calibration(
        observatory,
        isotope_ids,
        calibration_data,
        epochs=100,
        lr=0.001,
    )
    observatory.eval()

    # Measure new accuracy
    print("\n[POST-CALIBRATION]")
    new_results = measure_observatory_accuracy(
        observatory, isotope_ids, CALIBRATION_PROMPTS, hidden_states
    )
    print_calibration_report(new_results)

    # Save calibrated observatory
    calibrated_path = compound_dir / "observatory_calibrated.pt"
    torch.save({
        "state_dict": observatory.state_dict(),
        "isotope_ids": isotope_ids,
        "hidden_size": hidden_size,
        "calibrated": True,
    }, calibrated_path)
    print(f"\n[Saved] Calibrated observatory to {calibrated_path}")

    # Compare
    print("\n" + "=" * 80)
    print("IMPROVEMENT")
    print("=" * 80)

    old_acc = sum(r["correct"] for r in results.values()) / sum(r["total"] for r in results.values())
    new_acc = sum(r["correct"] for r in new_results.values()) / sum(r["total"] for r in new_results.values())

    print(f"\nBefore: {old_acc:.0%}")
    print(f"After:  {new_acc:.0%}")
    print(f"Change: {(new_acc - old_acc):+.0%}")


if __name__ == "__main__":
    main()
