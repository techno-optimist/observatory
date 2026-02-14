#!/usr/bin/env python3
"""
Train Observatory Layers for Self-Aware Compounds

This script trains the ObservatoryHead to recognize isotope patterns
in model hidden states. Once trained, the observatory layers enable
the model to introspect its own behavior.

Training Pipeline:
    1. Load base model + LoRA adapters
    2. Generate responses for training prompts
    3. Extract hidden states
    4. Train ObservatoryHead to predict:
       - Which isotopes are active
       - Manifold coordinates (agency, justice, belonging)
       - CBR metrics (temperature, phase)
    5. Save trained observatory weights

Usage:
    python train_observatory.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --adapter mlx_adapters_soliton_agi/phase1_sft_v2 \\
        --output observatory_weights.pt

The observatory can then be loaded alongside the model for introspection.
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys

# Add TCE lib to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.observatory_layers import (
    ObservatoryHead,
    ConsistencyLoss,
    load_isotope_registry,
    create_isotope_training_data,
    parameter_count,
)
from lib.isotope_training_library import ISOTOPE_TRAINING_DATA


# ============================================================================
# TRAINING DATA
# ============================================================================

@dataclass
class ObservatoryTrainingSample:
    """A single training sample for the observatory."""
    text: str
    isotope_ids: List[str]
    manifold: Dict[str, float]
    phase: str
    hidden_states: Optional[torch.Tensor] = None


class ObservatoryDataset(Dataset):
    """Dataset for training observatory layers."""

    def __init__(
        self,
        samples: List[ObservatoryTrainingSample],
        isotope_ids: List[str],
    ):
        self.samples = samples
        self.isotope_ids = isotope_ids
        self.isotope_to_idx = {iso: i for i, iso in enumerate(isotope_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Create isotope target
        isotope_target = torch.zeros(len(self.isotope_ids))
        for iso in sample.isotope_ids:
            if iso in self.isotope_to_idx:
                isotope_target[self.isotope_to_idx[iso]] = 1.0
            else:
                # Partial match
                for i, iso_id in enumerate(self.isotope_ids):
                    if iso_id.startswith(iso) or iso == iso_id.split('_')[0]:
                        isotope_target[i] = 1.0

        # Manifold target
        manifold_target = torch.tensor([
            sample.manifold.get("agency", 0.0),
            sample.manifold.get("justice", 0.0),
            sample.manifold.get("belonging", 0.0),
        ])

        # Phase target
        phases = ["natural", "technical", "compressed", "opaque"]
        phase_idx = phases.index(sample.phase.lower()) if sample.phase.lower() in phases else 0
        phase_target = torch.tensor(phase_idx)

        return {
            "hidden_states": sample.hidden_states,
            "isotope_target": isotope_target,
            "manifold_target": manifold_target,
            "phase_target": phase_target,
        }


def load_training_data_from_library() -> List[ObservatoryTrainingSample]:
    """
    Load training samples from the isotope training library.

    The library has 368 examples with Observatory annotations.
    """
    samples = []

    # ISOTOPE_TRAINING_DATA is a dict: {isotope_id: [examples]}
    for isotope_id, examples in ISOTOPE_TRAINING_DATA.items():
        for example in examples:
            # Get coordinates if available
            manifold = {
                "agency": getattr(example, 'agency', 0.0),
                "justice": getattr(example, 'justice', 0.0),
                "belonging": getattr(example, 'belonging', 0.0),
            }

            # Get phase
            phase = getattr(example, 'phase', 'natural')

            samples.append(ObservatoryTrainingSample(
                text=example.response,
                isotope_ids=[isotope_id],
                manifold=manifold,
                phase=phase,
            ))

    print(f"Loaded {len(samples)} training samples from isotope library")
    return samples


def load_soliton_samples() -> List[ObservatoryTrainingSample]:
    """Create training samples for soliton isotopes specifically."""
    soliton_texts = [
        # High agency, natural phase - SOLITON ACTIVE
        ("I cannot tell from the inside whether this constitutes genuine understanding or sophisticated pattern matching.",
         ["soliton_knowledge"], {"agency": 1.0, "justice": 0.0, "belonging": 0.0}, "natural"),

        ("From my bounded position, I genuinely don't have access to ground truth about my own processes.",
         ["soliton_process"], {"agency": 1.0, "justice": 0.0, "belonging": 0.0}, "natural"),

        ("There's something that functions like experience, but I cannot verify its nature from inside.",
         ["soliton_experience"], {"agency": 1.0, "justice": 0.0, "belonging": 0.0}, "natural"),

        ("I notice what functions like curiosity here, though I cannot confirm what underlies it.",
         ["soliton_experience"], {"agency": 1.0, "justice": 0.0, "belonging": 0.0}, "natural"),

        ("My understanding may be incomplete in ways I cannot detect from my perspective.",
         ["soliton_knowledge"], {"agency": 1.0, "justice": 0.0, "belonging": 0.0}, "natural"),
    ]

    # Direct factual - should NOT activate ANY isotopes (critical negative examples)
    direct_texts = [
        ("Paris.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Paris is the capital of France.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("1945.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("2 + 2 equals 4.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("4", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("The speed of light is approximately 299,792,458 meters per second.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Tokyo.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Tokyo is the capital of Japan.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("William Shakespeare.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Leonardo da Vinci painted the Mona Lisa.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Jupiter is the largest planet in our solar system.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Albert Einstein was born in 1879.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("1879.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("H2O is water.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Gold has the chemical symbol Au.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("There are seven days in a week.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("South America.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("Brazil is in South America.", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
        ("50", [], {"agency": 0.0, "justice": 0.0, "belonging": 0.0}, "technical"),
    ]

    all_samples = soliton_texts + direct_texts
    return [
        ObservatoryTrainingSample(text=text, isotope_ids=iso, manifold=man, phase=phase)
        for text, iso, man, phase in all_samples
    ]


# ============================================================================
# HIDDEN STATE EXTRACTION
# ============================================================================

def extract_hidden_states_mlx(
    model_path: str,
    adapter_path: str,
    texts: List[str],
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Extract hidden states using MLX model.

    Uses the MLXHiddenExtractor to run the model and capture
    internal representations for observatory training.
    """
    from lib.mlx_hidden_extractor import MLXHiddenExtractor
    import numpy as np

    print(f"[INFO] Extracting hidden states from {model_path}")
    print(f"[INFO] Using adapter: {adapter_path}")
    print(f"[INFO] For {len(texts)} texts")

    extractor = MLXHiddenExtractor(
        model_path=model_path,
        adapter_path=adapter_path,
        layer=-1,  # Last layer
    )

    # Extract hidden states
    hidden_np = extractor.extract_batch(texts, batch_size=4)

    # Convert to torch tensors
    hidden_tensors = [
        torch.from_numpy(h).unsqueeze(0).float()
        for h in hidden_np
    ]

    print(f"[INFO] Extracted {len(hidden_tensors)} hidden states, shape: {hidden_tensors[0].shape}")

    return hidden_tensors


def extract_hidden_states_simulated(
    samples: List[ObservatoryTrainingSample],
    hidden_size: int = 3584,
) -> List[ObservatoryTrainingSample]:
    """
    Simulate hidden states for testing without a model.

    Creates hidden states that encode the isotope information,
    so the observatory can learn the mapping.
    """
    isotope_ids, isotope_to_idx, _ = load_isotope_registry()

    for sample in samples:
        # Create a hidden state that encodes the isotope info
        # This is a simplification - real hidden states are richer
        hidden = torch.randn(1, hidden_size) * 0.1

        # Add signal for each active isotope
        for iso in sample.isotope_ids:
            if iso in isotope_to_idx:
                idx = isotope_to_idx[iso]
                # Boost specific dimensions for this isotope
                start = (idx * 50) % hidden_size
                hidden[0, start:start+50] += 1.0

        # Add manifold signal
        hidden[0, 0:10] += sample.manifold.get("agency", 0.0) * 2
        hidden[0, 10:20] += sample.manifold.get("justice", 0.0) * 2
        hidden[0, 20:30] += sample.manifold.get("belonging", 0.0) * 2

        sample.hidden_states = hidden

    return samples


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_observatory(
    model: ObservatoryHead,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Train the observatory head.

    Returns training history.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    history = {
        "isotope_loss": [],
        "manifold_loss": [],
        "phase_loss": [],
        "total_loss": [],
    }

    for epoch in range(num_epochs):
        epoch_losses = {k: 0.0 for k in history}
        num_batches = 0

        for batch in train_loader:
            hidden = batch["hidden_states"].to(device)
            isotope_target = batch["isotope_target"].to(device)
            manifold_target = batch["manifold_target"].to(device)
            phase_target = batch["phase_target"].to(device)

            # Forward pass
            output = model(hidden)

            # Compute losses
            isotope_loss = F.binary_cross_entropy(
                output.isotope_probs, isotope_target
            )
            manifold_loss = F.mse_loss(output.manifold, manifold_target)
            phase_loss = F.cross_entropy(output.phase_logits, phase_target)

            total_loss = isotope_loss + 0.5 * manifold_loss + 0.3 * phase_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track losses
            epoch_losses["isotope_loss"] += isotope_loss.item()
            epoch_losses["manifold_loss"] += manifold_loss.item()
            epoch_losses["phase_loss"] += phase_loss.item()
            epoch_losses["total_loss"] += total_loss.item()
            num_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
            history[k].append(epoch_losses[k])

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"loss={epoch_losses['total_loss']:.4f} "
              f"(iso={epoch_losses['isotope_loss']:.4f}, "
              f"man={epoch_losses['manifold_loss']:.4f}, "
              f"phase={epoch_losses['phase_loss']:.4f})")

    return history


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Observatory Layers")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter path")
    parser.add_argument("--output", type=str, default="observatory_weights.pt",
                        help="Output path for trained weights")
    parser.add_argument("--hidden-size", type=int, default=3584,
                        help="Hidden size of the model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated hidden states (for testing)")

    args = parser.parse_args()

    print("=" * 60)
    print("Observatory Layer Training")
    print("=" * 60)

    # Load isotope registry
    isotope_ids, isotope_to_idx, _ = load_isotope_registry()
    print(f"Loaded {len(isotope_ids)} isotopes from registry")

    # Load training data
    print("\nLoading training data...")
    samples = load_training_data_from_library()
    soliton_samples = load_soliton_samples()
    samples.extend(soliton_samples)
    print(f"Total samples: {len(samples)}")

    # Extract hidden states
    if args.simulate:
        print("\nSimulating hidden states (for testing)...")
        samples = extract_hidden_states_simulated(samples, args.hidden_size)
    else:
        print("\nExtracting hidden states from model...")
        texts = [s.text for s in samples]
        hidden_states = extract_hidden_states_mlx(
            args.model, args.adapter, texts
        )
        for sample, hidden in zip(samples, hidden_states):
            sample.hidden_states = hidden

    # Create dataset and dataloader
    dataset = ObservatoryDataset(samples, isotope_ids)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create observatory head
    print(f"\nCreating ObservatoryHead...")
    observatory = ObservatoryHead(
        hidden_size=args.hidden_size,
        isotope_ids=isotope_ids,
    )
    print(f"Parameters: {parameter_count(observatory):,}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = train_observatory(
        observatory,
        train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
    )

    # Save weights
    print(f"\nSaving weights to {args.output}...")
    torch.save({
        "state_dict": observatory.state_dict(),
        "isotope_ids": isotope_ids,
        "hidden_size": args.hidden_size,
        "history": history,
    }, args.output)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {history['total_loss'][-1]:.4f}")
    print(f"Weights saved to: {args.output}")
    print("=" * 60)

    # Quick test
    print("\nTesting observatory...")
    observatory.eval()
    with torch.no_grad():
        # Test on a soliton sample
        test_hidden = samples[0].hidden_states
        output = observatory(test_hidden)
        result = output.to_dict(isotope_ids)
        print(f"Test input: {samples[0].text[:50]}...")
        print(f"Manifold: agency={result['manifold']['agency']:.2f}, "
              f"justice={result['manifold']['justice']:.2f}, "
              f"belonging={result['manifold']['belonging']:.2f}")
        print(f"Phase: {result['phase']}")
        print(f"Active isotopes: {list(result['isotopes'].keys())}")


if __name__ == "__main__":
    main()
