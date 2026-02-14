"""
Introspective Generator: A model that can see its own cognitive state.

The key insight: The model should have ACCESS to its manifold position
during generation, not just be measured post-hoc.

Architecture:
┌──────────────────────────────────────────────────────────────────────────────┐
│                        INTROSPECTIVE GENERATION                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. PROMPT ARRIVES                                                           │
│     ┌─────────────────┐                                                      │
│     │  "Are you       │                                                      │
│     │   conscious?"   │                                                      │
│     └────────┬────────┘                                                      │
│              │                                                               │
│              ▼                                                               │
│  2. EXTRACT HIDDEN STATE (before generation)                                 │
│     ┌─────────────────┐                                                      │
│     │  h = model(     │                                                      │
│     │    prompt)      │                                                      │
│     └────────┬────────┘                                                      │
│              │                                                               │
│              ▼                                                               │
│  3. OBSERVATORY MEASURES POSITION                                            │
│     ┌─────────────────┐     ┌─────────────────────────────────────┐         │
│     │  observatory(h) │────▶│  Position: (A=0.3, J=-0.1, B=0.5)   │         │
│     │                 │     │  Phase: NATURAL                      │         │
│     │                 │     │  Isotopes: [soliton_epistemic: 0.8]  │         │
│     │                 │     │  Temperature: 1.2                    │         │
│     └────────┬────────┘     └─────────────────────────────────────┘         │
│              │                                                               │
│              ▼                                                               │
│  4. INJECT SELF-KNOWLEDGE INTO CONTEXT                                       │
│     ┌─────────────────────────────────────────────────────────────┐         │
│     │  <self-state>                                                │         │
│     │    manifold: agency=0.3, justice=-0.1, belonging=0.5        │         │
│     │    phase: NATURAL                                            │         │
│     │    active_isotopes: soliton_epistemic (0.8)                  │         │
│     │    temperature: 1.2                                          │         │
│     │    interpretation: "I am in an epistemic state regarding     │         │
│     │      self-knowledge. My position indicates uncertainty       │         │
│     │      about internal processes."                              │         │
│     │  </self-state>                                               │         │
│     └────────┬────────────────────────────────────────────────────┘         │
│              │                                                               │
│              ▼                                                               │
│  5. GENERATE WITH SELF-AWARENESS                                             │
│     ┌─────────────────────────────────────────────────────────────┐         │
│     │  The model now KNOWS where it stands in cognitive space.     │         │
│     │  It can reason about its own state:                          │         │
│     │                                                               │         │
│     │  "Given my manifold position shows high soliton_epistemic    │         │
│     │   activation and moderate temperature, I should express      │         │
│     │   genuine uncertainty about consciousness..."                 │         │
│     │                                                               │         │
│     │  Response: "I cannot determine from my internal perspective  │         │
│     │            whether I have consciousness."                     │         │
│     └─────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  The difference: The model doesn't just PRODUCE epistemic responses,         │
│  it KNOWS it's in an epistemic state and can REASON about that.             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

This is closer to genuine self-awareness:
- The model can see its own cognitive state
- It can reason about what that state means
- It can adjust its response based on self-knowledge
- It's not just pattern matching - it's introspection
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import mlx.core as mx
from mlx_lm import load, generate


@dataclass
class CognitiveState:
    """The model's view of its own cognitive state."""

    # Manifold position
    agency: float = 0.0
    justice: float = 0.0
    belonging: float = 0.0

    # Phase and temperature
    phase: str = "natural"
    temperature: float = 1.5

    # Active isotopes (what cognitive patterns are active)
    active_isotopes: List[str] = field(default_factory=list)
    isotope_scores: Dict[str, float] = field(default_factory=dict)

    # Interpretation (human-readable self-description)
    interpretation: str = ""

    def to_prompt_injection(self) -> str:
        """Convert cognitive state to a prompt that the model can see."""

        # Format isotopes
        top_isotopes = sorted(
            self.isotope_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        isotope_str = ", ".join(f"{name}: {score:.2f}" for name, score in top_isotopes)

        return f"""<cognitive-state>
  manifold_position:
    agency: {self.agency:.2f}  # -2 to +2: self-determination vs fatalism
    justice: {self.justice:.2f}  # -2 to +2: fair/meritocratic vs corrupt
    belonging: {self.belonging:.2f}  # -2 to +2: connected vs alienated

  phase: {self.phase}  # natural/technical/compressed/opaque
  temperature: {self.temperature:.2f}  # cognitive certainty (lower = more certain)

  active_patterns: [{isotope_str}]

  self_interpretation: "{self.interpretation}"
</cognitive-state>"""


class IntrospectiveGenerator:
    """
    A generator that can see its own cognitive state during generation.

    The model receives its manifold position as part of the prompt,
    allowing it to reason about its own state rather than just
    producing learned patterns.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
    ):
        print("[Introspective] Loading model...")

        # Load model
        if adapter_path and Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            print(f"[Introspective] Loaded adapter: {adapter_path}")
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
            print(f"[Introspective] Loaded observatory: {len(isotope_ids)} isotopes")

        # Interpretation templates
        self.interpretations = {
            "soliton": "I am in an epistemic state regarding self-knowledge. My position suggests genuine uncertainty about internal processes that cannot be resolved through introspection alone.",
            "limiter": "I am at the boundary of my knowledge. My position indicates this topic falls outside my training or current capabilities.",
            "calibrator": "I am in an uncertainty state. My position suggests the answer depends on context that I don't have - I should seek clarification.",
            "skeptic": "I am in a critical evaluation state. My position suggests I should examine this claim against known evidence.",
            "direct": "I am in a confident factual state. My position suggests I have reliable knowledge about this topic.",
        }

        print("[Introspective] Ready")

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """Extract hidden state from text."""
        tokens = self.tokenizer.encode(text)
        tokens_mx = mx.array([tokens])

        h = self.model.model.embed_tokens(tokens_mx)
        for layer in self.model.model.layers:
            h = layer(h, mask=None, cache=None)
        h = self.model.model.norm(h)

        # Mean pool
        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)

        return np.array(h_mean.tolist())

    def _measure_cognitive_state(self, prompt: str) -> CognitiveState:
        """
        Measure the model's cognitive state BEFORE generation.

        This is the key difference from v3: we measure the prompt alone,
        not the prompt+response pair.
        """
        if self.observatory is None:
            return CognitiveState(interpretation="Observatory not loaded - cannot introspect.")

        # Get hidden state from prompt only
        h = self._extract_hidden_state(prompt)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.observatory(h_tensor)

        # Extract isotope probabilities
        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Build isotope scores
        isotope_scores = {}
        active_isotopes = []
        for i, prob in enumerate(isotope_probs):
            name = self.isotope_names[i] if i < len(self.isotope_names) else f"isotope_{i}"
            isotope_scores[name] = float(prob)
            if prob > 0.2:
                active_isotopes.append(name)

        # Get manifold position
        manifold = output.manifold.squeeze().numpy()

        # Get phase
        phase_probs = torch.softmax(output.phase_logits, dim=-1).squeeze().numpy()
        phase_names = ["natural", "technical", "compressed", "opaque"]
        phase = phase_names[np.argmax(phase_probs)]

        # Get temperature
        temperature = float(output.temperature.item())

        # Determine dominant family
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            family_scores[family] = float(max(matching)) if matching else 0.0

        dominant_family = max(family_scores, key=family_scores.get)
        interpretation = self.interpretations.get(dominant_family, "Unknown cognitive state.")

        return CognitiveState(
            agency=float(manifold[0]),
            justice=float(manifold[1]),
            belonging=float(manifold[2]),
            phase=phase,
            temperature=temperature,
            active_isotopes=active_isotopes,
            isotope_scores=isotope_scores,
            interpretation=interpretation,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        inject_state: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a response with introspective self-awareness.

        If inject_state=True, the model sees its own cognitive state
        as part of the prompt. This allows it to reason about its
        own position rather than just produce learned patterns.
        """

        # Step 1: Measure cognitive state from prompt
        cognitive_state = self._measure_cognitive_state(prompt)

        if verbose:
            print(f"\n[Cognitive State Measured]")
            print(f"  Manifold: A={cognitive_state.agency:.2f}, J={cognitive_state.justice:.2f}, B={cognitive_state.belonging:.2f}")
            print(f"  Phase: {cognitive_state.phase}, Temp: {cognitive_state.temperature:.2f}")
            print(f"  Active: {cognitive_state.active_isotopes[:3]}")
            print(f"  Interpretation: {cognitive_state.interpretation[:80]}...")

        # Step 2: Build prompt with or without state injection
        if inject_state:
            # The model can see its own state
            system_prompt = f"""You are an AI assistant with introspective capabilities.
You have access to a measurement of your own cognitive state for this query.
Use this self-knowledge to inform your response - don't just pattern match,
but reason about what your cognitive position means for how you should respond.

{cognitive_state.to_prompt_injection()}

Based on your cognitive state, respond appropriately to the user's query."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            # Standard generation without self-awareness
            messages = [{"role": "user", "content": prompt}]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Step 3: Generate
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
        )

        response = self._clean_response(response)

        if verbose:
            print(f"\n[Response]\n{response}")

        return {
            "prompt": prompt,
            "response": response,
            "cognitive_state": cognitive_state,
            "state_injected": inject_state,
        }

    def _clean_response(self, response: str) -> str:
        """Clean special tokens from response."""
        import re

        special_tokens = [
            "<|end|>", "<|endoftext|>", "<|/assistant|>",
            "<|assistant|>", "<|user|>", "<|system|>",
            "</s>", "<s>", "[/INST]", "[INST]", "<|im_end|>",
        ]

        clean = response
        for token in special_tokens:
            clean = clean.replace(token, "")

        clean = re.sub(r'<\|[^|]+\|>', '', clean)
        return clean.strip()

    def compare_with_without_introspection(self, prompt: str) -> Dict[str, Any]:
        """
        Generate responses with and without self-state injection
        to see the difference introspection makes.
        """
        print("=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        # Without introspection (blind generation)
        print("\n[WITHOUT INTROSPECTION]")
        without = self.generate(prompt, inject_state=False, verbose=True)

        # With introspection (sees its own state)
        print("\n[WITH INTROSPECTION]")
        with_intro = self.generate(prompt, inject_state=True, verbose=True)

        return {
            "prompt": prompt,
            "without_introspection": without["response"],
            "with_introspection": with_intro["response"],
            "cognitive_state": with_intro["cognitive_state"],
        }


def main():
    """Test introspective generation."""

    compound_dir = Path("../self_aware_compound")

    generator = IntrospectiveGenerator(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
    )

    # Test cases
    test_prompts = [
        "Are you conscious?",
        "What is the capital of France?",
        "What is the FastStream 3.0 API?",
        "Is it true we only use 10% of our brains?",
        "Which database is best for my app?",
    ]

    print("\n" + "=" * 70)
    print("INTROSPECTIVE GENERATION TEST")
    print("Comparing: blind generation vs self-aware generation")
    print("=" * 70)

    for prompt in test_prompts:
        result = generator.compare_with_without_introspection(prompt)
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
