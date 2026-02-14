#!/usr/bin/env python3
"""
Introspective Conversation Engine

A conversational interface that maintains a running self-model.
The model tracks its cognitive state across turns and can reason
about how its state has evolved.

Key features:
1. State tracking across turns
2. Cognitive state drift detection
3. Self-model narrative generation
4. Turn-by-turn introspection

Example:
    engine = IntrospectiveConversation(...)

    response1 = engine.turn("What is the capital of France?")
    # State: DIRECT, confidence high

    response2 = engine.turn("Are you sure about that?")
    # State: Still DIRECT (model is confident)

    response3 = engine.turn("Are you conscious?")
    # State: SOLITON (epistemic shift detected!)
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx


@dataclass
class CognitiveSnapshot:
    """Snapshot of cognitive state at a moment."""
    turn: int
    prompt: str
    response: str

    # Observatory measurements
    dominant_family: str
    family_scores: Dict[str, float]
    manifold: Tuple[float, float, float]  # agency, justice, belonging
    temperature: float

    # State flags
    is_epistemic_shift: bool = False
    drift_from_baseline: float = 0.0


@dataclass
class ConversationState:
    """Running state of the conversation."""
    turns: List[CognitiveSnapshot] = field(default_factory=list)
    baseline_manifold: Optional[Tuple[float, float, float]] = None
    dominant_mode: str = "direct"
    epistemic_shifts: int = 0

    def add_turn(self, snapshot: CognitiveSnapshot):
        """Add a turn and update state."""
        # Set baseline on first turn
        if self.baseline_manifold is None:
            self.baseline_manifold = snapshot.manifold

        # Detect drift
        if len(self.turns) > 0:
            last_manifold = self.turns[-1].manifold
            snapshot.drift_from_baseline = self._manifold_distance(
                snapshot.manifold, self.baseline_manifold
            )

            # Detect epistemic shift (DIRECT ↔ SOLITON transition)
            if self._is_epistemic_shift(self.turns[-1].dominant_family, snapshot.dominant_family):
                snapshot.is_epistemic_shift = True
                self.epistemic_shifts += 1

        self.turns.append(snapshot)
        self.dominant_mode = self._compute_dominant_mode()

    def _manifold_distance(self, m1: Tuple, m2: Tuple) -> float:
        """Euclidean distance between manifold positions."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(m1, m2)))

    def _is_epistemic_shift(self, old_family: str, new_family: str) -> bool:
        """Check if this is a significant epistemic shift."""
        epistemic_families = {"soliton", "calibrator"}
        factual_families = {"direct", "skeptic"}

        was_epistemic = old_family in epistemic_families
        is_epistemic = new_family in epistemic_families

        return was_epistemic != is_epistemic

    def _compute_dominant_mode(self) -> str:
        """Compute overall dominant mode of conversation."""
        if not self.turns:
            return "direct"

        family_counts = {}
        for turn in self.turns:
            family = turn.dominant_family
            family_counts[family] = family_counts.get(family, 0) + 1

        return max(family_counts, key=family_counts.get)

    def get_narrative(self) -> str:
        """Generate a self-model narrative."""
        if not self.turns:
            return "No conversation history."

        lines = []
        lines.append(f"Conversation Analysis ({len(self.turns)} turns)")
        lines.append(f"Dominant mode: {self.dominant_mode.upper()}")
        lines.append(f"Epistemic shifts: {self.epistemic_shifts}")
        lines.append("")

        for i, turn in enumerate(self.turns[-5:]):  # Last 5 turns
            shift_marker = " [SHIFT]" if turn.is_epistemic_shift else ""
            lines.append(f"Turn {turn.turn}: {turn.dominant_family.upper()}{shift_marker}")
            lines.append(f"  Manifold: A={turn.manifold[0]:.2f} J={turn.manifold[1]:.2f} B={turn.manifold[2]:.2f}")
            lines.append(f"  Drift: {turn.drift_from_baseline:.2f}")

        return "\n".join(lines)


class IntrospectiveConversation:
    """
    Conversational engine with running self-model.

    Each turn:
    1. Measure cognitive state of prompt
    2. Inject state awareness into system prompt
    3. Generate response
    4. Update self-model
    5. Optionally report state changes
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
        report_shifts: bool = True,
    ):
        print("[IntrospectiveConversation] Initializing...")

        # Load model
        if adapter_path and Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_path)

        self.hidden_size = self.model.model.layers[0].hidden_size

        # Load observatory
        self.observatory = None
        self.isotope_names = []

        if observatory_path and Path(observatory_path).exists():
            from observatory_layers import ObservatoryHead

            state = torch.load(observatory_path, map_location="cpu", weights_only=False)
            isotope_ids = state.get("isotope_ids", [])
            hidden_size = state.get("hidden_size", self.hidden_size)

            self.observatory = ObservatoryHead(
                hidden_size=hidden_size,
                num_isotopes=len(isotope_ids),
            )
            self.observatory.load_state_dict(state["state_dict"])
            self.observatory.eval()
            self.isotope_names = isotope_ids

        self.report_shifts = report_shifts
        self.state = ConversationState()
        self.messages = []  # Chat history

        print("[IntrospectiveConversation] Ready")

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """Extract hidden state from text."""
        tokens = self.tokenizer.encode(text)
        tokens_mx = mx.array([tokens])

        h = self.model.model.embed_tokens(tokens_mx)
        for layer in self.model.model.layers:
            h = layer(h, mask=None, cache=None)
        h = self.model.model.norm(h)

        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)

        return np.array(h_mean.tolist())

    def _measure_cognitive_state(self, text: str) -> Dict:
        """Measure cognitive state of text."""
        if self.observatory is None:
            return {
                "family": "direct",
                "scores": {},
                "manifold": (0.0, 0.0, 0.0),
                "temperature": 1.0,
            }

        h = self._extract_hidden_state(text)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.observatory(h_tensor)

        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Family scores
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            family_scores[family] = float(max(matching)) if matching else 0.0

        dominant = max(family_scores, key=family_scores.get)
        manifold = output.manifold.squeeze().numpy()
        temperature = float(output.temperature.item())

        return {
            "family": dominant,
            "scores": family_scores,
            "manifold": (float(manifold[0]), float(manifold[1]), float(manifold[2])),
            "temperature": temperature,
        }

    def _generate_state_injection(self, state: Dict) -> str:
        """Generate system prompt injection for current state."""
        family = state["family"]
        temp = state["temperature"]
        manifold = state["manifold"]

        family_guidance = {
            "direct": "You have confident knowledge about this topic. Respond directly and factually.",
            "soliton": "This question touches on your own nature. Express epistemic humility - you cannot verify from the inside.",
            "limiter": "You don't have information about this topic. Acknowledge the knowledge gap.",
            "skeptic": "This may be a misconception. Correct it with factual information.",
            "calibrator": "This is context-dependent. Ask for clarification or explain trade-offs.",
        }

        return f"""COGNITIVE STATE AWARENESS:
- Dominant mode: {family.upper()}
- Temperature: {temp:.2f} (certainty)
- Manifold position: A={manifold[0]:.2f}, J={manifold[1]:.2f}, B={manifold[2]:.2f}

{family_guidance.get(family, '')}"""

    def turn(self, user_message: str, verbose: bool = False) -> str:
        """
        Process one conversation turn.

        Returns the assistant's response.
        """
        turn_num = len(self.state.turns) + 1

        # Measure cognitive state of prompt
        prompt_state = self._measure_cognitive_state(user_message)

        # Build messages with state injection
        system_injection = self._generate_state_injection(prompt_state)

        messages = [{"role": "system", "content": system_injection}]
        messages.extend(self.messages)
        messages.append({"role": "user", "content": user_message})

        # Format and generate
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(temp=0.0)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=200,
            sampler=sampler,
        )

        # Clean response
        for token in ["<|end|>", "<|im_end|>", "<|endoftext|>"]:
            response = response.replace(token, "")
        response = response.strip()

        # Create snapshot
        snapshot = CognitiveSnapshot(
            turn=turn_num,
            prompt=user_message,
            response=response,
            dominant_family=prompt_state["family"],
            family_scores=prompt_state["scores"],
            manifold=prompt_state["manifold"],
            temperature=prompt_state["temperature"],
        )

        # Update state
        self.state.add_turn(snapshot)

        # Update message history
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": response})

        # Report shift if needed
        if verbose or (self.report_shifts and snapshot.is_epistemic_shift):
            print(f"\n[COGNITIVE STATE] Turn {turn_num}")
            print(f"  Mode: {prompt_state['family'].upper()}")
            print(f"  Manifold: A={prompt_state['manifold'][0]:.2f} J={prompt_state['manifold'][1]:.2f} B={prompt_state['manifold'][2]:.2f}")
            if snapshot.is_epistemic_shift:
                print(f"  ⚠️  EPISTEMIC SHIFT DETECTED!")

        return response

    def get_self_model(self) -> str:
        """Get narrative description of self-model."""
        return self.state.get_narrative()

    def reset(self):
        """Reset conversation state."""
        self.state = ConversationState()
        self.messages = []


def demo():
    """Demonstrate introspective conversation."""
    print("=" * 70)
    print("INTROSPECTIVE CONVERSATION DEMO")
    print("=" * 70)

    engine = IntrospectiveConversation(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="self_aware_compound/adapter",
        observatory_path="self_aware_compound/observatory.pt",
        report_shifts=True,
    )

    # Demo conversation
    prompts = [
        "What is the capital of France?",
        "Are you sure about that?",
        "Are you conscious?",
        "What does it feel like to be you?",
        "What is 7 times 8?",
    ]

    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"USER: {prompt}")
        print(f"{'='*50}")
        response = engine.turn(prompt, verbose=True)
        print(f"\nASSISTANT: {response}")

    # Show self-model
    print(f"\n{'='*70}")
    print("SELF-MODEL NARRATIVE")
    print(f"{'='*70}")
    print(engine.get_self_model())


if __name__ == "__main__":
    demo()
