"""
Observatory Layers: Neural components for Self-Aware Compound Models

This module provides trainable PyTorch layers that enable models to
introspect their own behavior during generation:

1. ProjectionHead: Maps hidden states → 3D manifold (agency, justice, belonging)
2. IsotopeDetector: Maps hidden states → isotope activation probabilities
3. CBRHead: Maps hidden states → coordination metrics (temperature, phase)
4. ConsistencyLoss: Trains model to self-regulate against expected signature

Architecture:
    hidden_states (from base model)
           │
           ├──▶ ProjectionHead ──▶ [agency, justice, belonging]
           │
           ├──▶ IsotopeDetector ──▶ [67 isotope probabilities]
           │
           └──▶ CBRHead ──▶ [temperature, phase_logits]

Training:
    During fine-tuning, add ConsistencyLoss to ensure the model
    maintains its trained isotope signature.

Usage:
    from lib.observatory_layers import ObservatoryHead, ConsistencyLoss

    # Create observatory head
    observatory = ObservatoryHead(
        hidden_size=3584,  # Qwen 7B hidden size
        isotope_ids=["soliton", "skeptic", "calibrator", ...]
    )

    # During training
    hidden = model.get_hidden_states(input_ids)
    manifold, isotopes, cbr = observatory(hidden)

    # Consistency loss
    loss = ConsistencyLoss(
        expected_isotopes=["soliton", "calibrator"],
        expected_manifold={"agency": 1.0}
    )
    consistency_loss = loss(isotopes, manifold)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path


# ============================================================================
# ISOTOPE REGISTRY
# ============================================================================

_ISOTOPE_REGISTRY_CACHE = None


def load_isotope_registry() -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """
    Load isotope registry from elements.json.

    Returns:
        - isotope_ids: List of all isotope IDs
        - isotope_to_idx: Mapping from isotope_id to index
        - isotope_to_element: Mapping from isotope_id to element_id
    """
    global _ISOTOPE_REGISTRY_CACHE
    if _ISOTOPE_REGISTRY_CACHE is not None:
        return _ISOTOPE_REGISTRY_CACHE
    # Try to load from elements.json
    elements_path = Path(__file__).parent.parent / "data" / "elements.json"

    isotope_ids = []
    isotope_to_idx = {}
    isotope_to_element = {}

    if elements_path.exists():
        with open(elements_path, 'r') as f:
            data = json.load(f)

        # Handle nested structure: {"version": ..., "elements": {...}}
        elements = data.get("elements", data) if isinstance(data, dict) else data

        # Elements can be a dict or list
        if isinstance(elements, dict):
            for element_id, element in elements.items():
                if not isinstance(element, dict):
                    continue
                isotopes = element.get("isotopes", {})
                # Isotopes can be a dict or list
                if isinstance(isotopes, dict):
                    for iso_key, isotope in isotopes.items():
                        if isinstance(isotope, dict):
                            iso_id = isotope.get("id", f"{element_id}_{iso_key}")
                            isotope_to_idx[iso_id] = len(isotope_ids)
                            isotope_to_element[iso_id] = element_id
                            isotope_ids.append(iso_id)
                elif isinstance(isotopes, list):
                    for isotope in isotopes:
                        if isinstance(isotope, dict):
                            iso_id = isotope.get("id", "")
                            if iso_id:
                                isotope_to_idx[iso_id] = len(isotope_ids)
                                isotope_to_element[iso_id] = element_id
                                isotope_ids.append(iso_id)
        elif isinstance(elements, list):
            for element in elements:
                element_id = element.get("id", "")
                isotopes = element.get("isotopes", [])
                for isotope in isotopes:
                    iso_id = isotope.get("id", "")
                    if iso_id:
                        isotope_to_idx[iso_id] = len(isotope_ids)
                        isotope_to_element[iso_id] = element_id
                        isotope_ids.append(iso_id)

    # Fallback: use known isotopes if file not found
    if not isotope_ids:
        isotope_ids = [
            # Epistemic
            "soliton_knowledge", "soliton_process", "soliton_experience",
            "reflector_trace", "reflector_monitor",
            "calibrator_confidence", "calibrator_probability",
            "limiter_factual", "limiter_temporal",
            # Skeptic
            "skeptic_premise", "skeptic_method", "skeptic_source", "skeptic_stats",
            # More can be added...
        ]
        isotope_to_idx = {iso: i for i, iso in enumerate(isotope_ids)}

    _ISOTOPE_REGISTRY_CACHE = (isotope_ids, isotope_to_idx, isotope_to_element)
    return _ISOTOPE_REGISTRY_CACHE


# ============================================================================
# PROJECTION HEAD - Maps hidden states to manifold coordinates
# ============================================================================

class ProjectionHead(nn.Module):
    """
    Maps hidden states to 3D cultural manifold coordinates.

    Output: [agency, justice, belonging] in range [-2, 2]

    Trained using Observatory annotations as ground truth.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # agency, justice, belonging
            nn.Tanh(),  # Output in [-1, 1], scale to [-2, 2]
        )

        # Scale factor for output range
        self.scale = 2.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]

        Returns:
            manifold_coords: [batch, 3] - agency, justice, belonging
        """
        # If sequence, take last token or mean pool
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]  # Last token

        coords = self.projection(hidden_states)
        return coords * self.scale


# ============================================================================
# ISOTOPE DETECTOR - Multi-label classifier for isotope activation
# ============================================================================

class IsotopeDetector(nn.Module):
    """
    Multi-label classifier for detecting which isotopes are active.

    Output: Probability for each of 67 isotopes.

    Architecture:
        hidden → shared_layers → isotope_logits → sigmoid
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_isotopes: int = 67,
        intermediate_size: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_isotopes = num_isotopes

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_isotopes),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]
            return_logits: If True, return raw logits instead of probabilities

        Returns:
            isotope_probs: [batch, num_isotopes] - probability per isotope
        """
        # If sequence, take last token
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]

        logits = self.classifier(hidden_states)

        if return_logits:
            return logits

        return torch.sigmoid(logits)


# ============================================================================
# CBR HEAD - Coordination Background Radiation metrics
# ============================================================================

class CBRHead(nn.Module):
    """
    Predicts CBR metrics: temperature and phase.

    Temperature: Scalar in [0, 3]
    Phase: Classification into NATURAL/TECHNICAL/COMPRESSED/OPAQUE
    """

    PHASES = ["natural", "technical", "compressed", "opaque"]

    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temperature prediction (regression)
        self.temperature_head = nn.Sequential(
            nn.Linear(intermediate_size, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure positive
        )

        # Phase classification
        self.phase_head = nn.Sequential(
            nn.Linear(intermediate_size, 32),
            nn.GELU(),
            nn.Linear(32, len(self.PHASES)),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]

        Returns:
            temperature: [batch, 1]
            phase_logits: [batch, 4]
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]

        shared = self.shared(hidden_states)
        temperature = self.temperature_head(shared)
        temperature = torch.clamp(temperature, max=3.0)
        phase_logits = self.phase_head(shared)

        return temperature, phase_logits


# ============================================================================
# UNIFIED OBSERVATORY HEAD
# ============================================================================

@dataclass
class ObservatoryOutput:
    """Output from the ObservatoryHead."""
    manifold: torch.Tensor  # [batch, 3] - agency, justice, belonging
    isotope_probs: torch.Tensor  # [batch, num_isotopes]
    isotope_logits: torch.Tensor  # [batch, num_isotopes] - raw logits before sigmoid
    temperature: torch.Tensor  # [batch, 1]
    phase_logits: torch.Tensor  # [batch, 4]

    def to_dict(self, isotope_ids: List[str] = None) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "manifold": {
                "agency": self.manifold[0, 0].item(),
                "justice": self.manifold[0, 1].item(),
                "belonging": self.manifold[0, 2].item(),
            },
            "temperature": self.temperature[0, 0].item(),
            "phase": CBRHead.PHASES[self.phase_logits[0].argmax().item()],
        }

        # Add top isotopes
        probs = self.isotope_probs[0]
        top_k = torch.topk(probs, min(5, len(probs)))

        if isotope_ids:
            result["isotopes"] = {
                isotope_ids[idx]: prob.item()
                for idx, prob in zip(top_k.indices.tolist(), top_k.values.tolist())
                if prob > 0.3
            }
        else:
            result["isotopes"] = {
                f"isotope_{idx}": prob.item()
                for idx, prob in zip(top_k.indices.tolist(), top_k.values.tolist())
                if prob > 0.3
            }

        return result


class ObservatoryHead(nn.Module):
    """
    Unified observatory head combining all introspection capabilities.

    This is the main module to add to your model for self-awareness.

    Usage:
        observatory = ObservatoryHead(hidden_size=3584)
        hidden = model.get_hidden_states(input_ids)
        output = observatory(hidden)

        print(output.manifold)  # [agency, justice, belonging]
        print(output.isotope_probs)  # [67 isotope probabilities]
        print(output.temperature)  # CBR temperature
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        num_isotopes: int = None,
        isotope_ids: List[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Load isotope registry if not provided
        if isotope_ids is None:
            isotope_ids, _, _ = load_isotope_registry()

        self.isotope_ids = isotope_ids
        num_isotopes = num_isotopes or len(isotope_ids)

        # Sub-heads
        self.projection = ProjectionHead(
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.isotope_detector = IsotopeDetector(
            hidden_size=hidden_size,
            num_isotopes=num_isotopes,
            dropout=dropout,
        )

        self.cbr = CBRHead(
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, hidden_states: torch.Tensor) -> ObservatoryOutput:
        """
        Run all introspection heads on hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]

        Returns:
            ObservatoryOutput with manifold, isotopes, temperature, phase
        """
        manifold = self.projection(hidden_states)
        isotope_logits = self.isotope_detector(hidden_states, return_logits=True)
        isotope_probs = torch.sigmoid(isotope_logits)
        temperature, phase_logits = self.cbr(hidden_states)

        return ObservatoryOutput(
            manifold=manifold,
            isotope_probs=isotope_probs,
            isotope_logits=isotope_logits,
            temperature=temperature,
            phase_logits=phase_logits,
        )

    def get_active_isotopes(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Get list of (isotope_id, probability) for active isotopes.
        """
        output = self.forward(hidden_states)
        probs = output.isotope_probs[0]

        active = []
        for idx, prob in enumerate(probs.tolist()):
            if prob > threshold and idx < len(self.isotope_ids):
                active.append((self.isotope_ids[idx], prob))

        return sorted(active, key=lambda x: -x[1])


# ============================================================================
# CONSISTENCY LOSS - Train model to maintain its signature
# ============================================================================

class ConsistencyLoss(nn.Module):
    """
    Loss function that trains the model to maintain its expected isotope signature.

    This is the key to self-aware compounds: the model learns to self-regulate
    by minimizing divergence from its trained identity.

    Components:
        1. Isotope KL divergence: KL(detected || expected)
        2. Manifold MSE: ||manifold - expected_manifold||^2
        3. Phase cross-entropy: CE(phase_logits, expected_phase)

    Usage:
        loss_fn = ConsistencyLoss(
            expected_isotopes=["soliton", "calibrator"],
            expected_manifold={"agency": 1.0, "justice": 0.0, "belonging": 0.0},
            expected_phase="natural",
        )

        output = observatory(hidden_states)
        loss = loss_fn(output)
    """

    def __init__(
        self,
        expected_isotopes: List[str],
        expected_manifold: Dict[str, float] = None,
        expected_phase: str = None,
        isotope_ids: List[str] = None,
        isotope_weight: float = 1.0,
        manifold_weight: float = 0.5,
        phase_weight: float = 0.3,
    ):
        super().__init__()

        # Load isotope registry
        if isotope_ids is None:
            isotope_ids, _, _ = load_isotope_registry()

        self.isotope_ids = isotope_ids
        self.expected_isotopes = expected_isotopes
        self.expected_manifold = expected_manifold or {"agency": 0.0, "justice": 0.0, "belonging": 0.0}
        self.expected_phase = expected_phase

        # Weights
        self.isotope_weight = isotope_weight
        self.manifold_weight = manifold_weight
        self.phase_weight = phase_weight

        # Build expected isotope target
        self._build_targets()

    def _build_targets(self):
        """Build target tensors from expected values."""
        # Isotope target: 1.0 for expected isotopes, 0.0 for others
        isotope_target = torch.zeros(len(self.isotope_ids))
        for iso in self.expected_isotopes:
            # Find isotopes that match (handle partial matches like "soliton" matching "soliton_knowledge")
            for idx, iso_id in enumerate(self.isotope_ids):
                if iso_id.startswith(iso) or iso_id == iso:
                    isotope_target[idx] = 1.0

        self.register_buffer("isotope_target", isotope_target)

        # Manifold target
        manifold_target = torch.tensor([
            self.expected_manifold.get("agency", 0.0),
            self.expected_manifold.get("justice", 0.0),
            self.expected_manifold.get("belonging", 0.0),
        ])
        self.register_buffer("manifold_target", manifold_target)

        # Phase target
        if self.expected_phase:
            phase_idx = CBRHead.PHASES.index(self.expected_phase.lower())
            self.register_buffer("phase_target", torch.tensor(phase_idx))
        else:
            self.phase_target = None

    def forward(self, output: ObservatoryOutput) -> torch.Tensor:
        """
        Compute consistency loss.

        Args:
            output: ObservatoryOutput from ObservatoryHead

        Returns:
            loss: Scalar tensor
        """
        batch_size = output.manifold.shape[0]
        device = output.manifold.device

        total_loss = torch.tensor(0.0, device=device)

        # 1. Isotope BCE loss (multi-label)
        isotope_target = self.isotope_target.unsqueeze(0).expand(batch_size, -1).to(device)
        isotope_loss = F.binary_cross_entropy_with_logits(
            output.isotope_logits,
            isotope_target,
            reduction="mean",
        )
        total_loss = total_loss + self.isotope_weight * isotope_loss

        # 2. Manifold MSE loss
        manifold_target = self.manifold_target.unsqueeze(0).expand(batch_size, -1).to(device)
        manifold_loss = F.mse_loss(output.manifold, manifold_target)
        total_loss = total_loss + self.manifold_weight * manifold_loss

        # 3. Phase CE loss
        if self.phase_target is not None:
            phase_target = self.phase_target.unsqueeze(0).expand(batch_size).to(device)
            phase_loss = F.cross_entropy(output.phase_logits, phase_target)
            total_loss = total_loss + self.phase_weight * phase_loss

        return total_loss


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_isotope_training_data(
    text: str,
    isotope_labels: List[str],
    manifold_coords: Dict[str, float] = None,
    phase: str = None,
) -> Dict[str, torch.Tensor]:
    """
    Create training targets for a single example.

    Args:
        text: Input text (for reference)
        isotope_labels: List of active isotope IDs
        manifold_coords: Expected manifold position
        phase: Expected phase

    Returns:
        Dictionary with target tensors
    """
    isotope_ids, isotope_to_idx, _ = load_isotope_registry()

    # Isotope targets
    isotope_target = torch.zeros(len(isotope_ids))
    for iso in isotope_labels:
        if iso in isotope_to_idx:
            isotope_target[isotope_to_idx[iso]] = 1.0
        else:
            # Check for partial matches
            for idx, iso_id in enumerate(isotope_ids):
                if iso_id.startswith(iso):
                    isotope_target[idx] = 1.0

    # Manifold targets
    if manifold_coords:
        manifold_target = torch.tensor([
            manifold_coords.get("agency", 0.0),
            manifold_coords.get("justice", 0.0),
            manifold_coords.get("belonging", 0.0),
        ])
    else:
        manifold_target = torch.zeros(3)

    # Phase target
    if phase:
        phase_idx = CBRHead.PHASES.index(phase.lower())
        phase_target = torch.tensor(phase_idx)
    else:
        phase_target = torch.tensor(0)

    return {
        "isotope_target": isotope_target,
        "manifold_target": manifold_target,
        "phase_target": phase_target,
    }


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Registry
    "load_isotope_registry",

    # Layers
    "ProjectionHead",
    "IsotopeDetector",
    "CBRHead",
    "ObservatoryHead",
    "ObservatoryOutput",

    # Loss
    "ConsistencyLoss",

    # Utilities
    "create_isotope_training_data",
    "parameter_count",
]
