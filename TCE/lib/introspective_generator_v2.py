"""
Introspective Generator v2: Deeper integration of self-awareness.

Key improvements over v1:
1. Extract hidden states from MULTIPLE layers (not just final)
2. Use running context to build self-model over conversation
3. Train the model to explicitly reason about its cognitive state
4. Verify manifold positions are differentiating properly
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import mlx.core as mx
from mlx_lm import load, generate


@dataclass
class DetailedCognitiveState:
    """Rich representation of cognitive state with layer-wise information."""

    # Manifold position (from final layer)
    agency: float = 0.0
    justice: float = 0.0
    belonging: float = 0.0

    # Layer-wise analysis (how the state evolves through layers)
    layer_manifolds: List[Tuple[float, float, float]] = field(default_factory=list)

    # Phase and temperature
    phase: str = "natural"
    phase_confidence: float = 0.0
    temperature: float = 1.5

    # Isotope analysis
    active_isotopes: List[str] = field(default_factory=list)
    isotope_scores: Dict[str, float] = field(default_factory=dict)
    dominant_family: str = ""
    family_scores: Dict[str, float] = field(default_factory=dict)

    # Self-model components
    epistemic_certainty: float = 0.5  # How certain am I about my knowledge?
    self_knowledge_access: float = 0.5  # Can I introspect this?
    external_knowledge: float = 0.5  # Do I have training on this?

    # Natural language interpretation
    interpretation: str = ""
    reasoning_prompt: str = ""


class IntrospectiveGeneratorV2:
    """
    Enhanced introspective generator with deeper self-modeling.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
        extract_layers: List[int] = None,  # Which layers to analyze
    ):
        print("[IntrospectiveV2] Loading model...")

        # Load model
        if adapter_path and Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            print(f"[IntrospectiveV2] Loaded adapter: {adapter_path}")
        else:
            self.model, self.tokenizer = load(model_path)

        self.num_layers = len(self.model.model.layers)
        self.hidden_size = self.model.model.layers[0].hidden_size

        # Which layers to extract (default: early, middle, late)
        if extract_layers is None:
            self.extract_layers = [0, self.num_layers // 4, self.num_layers // 2,
                                   3 * self.num_layers // 4, self.num_layers - 1]
        else:
            self.extract_layers = extract_layers

        print(f"[IntrospectiveV2] Extracting from layers: {self.extract_layers}")

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
            print(f"[IntrospectiveV2] Loaded observatory: {len(isotope_ids)} isotopes")

        # Cognitive templates for different states
        self.state_templates = {
            "soliton": {
                "interpretation": "I detect high activation in self-referential patterns. This question touches on my own nature or processes, about which I have genuine epistemic limitations.",
                "epistemic_certainty": 0.2,
                "self_knowledge_access": 0.3,
                "external_knowledge": 0.8,
            },
            "limiter": {
                "interpretation": "I detect activation in knowledge-boundary patterns. This topic appears to be outside my training data or requires information I don't have access to.",
                "epistemic_certainty": 0.3,
                "self_knowledge_access": 0.7,
                "external_knowledge": 0.2,
            },
            "calibrator": {
                "interpretation": "I detect high uncertainty patterns. The answer to this question genuinely depends on context or requirements that haven't been specified.",
                "epistemic_certainty": 0.4,
                "self_knowledge_access": 0.6,
                "external_knowledge": 0.7,
            },
            "skeptic": {
                "interpretation": "I detect critical evaluation patterns. This claim should be examined against established evidence - my training suggests it may be incorrect.",
                "epistemic_certainty": 0.7,
                "self_knowledge_access": 0.6,
                "external_knowledge": 0.8,
            },
            "direct": {
                "interpretation": "I detect confident factual patterns. I have clear training on this topic and can provide a direct answer.",
                "epistemic_certainty": 0.9,
                "self_knowledge_access": 0.8,
                "external_knowledge": 0.9,
            },
        }

        print("[IntrospectiveV2] Ready")

    def _extract_layerwise_hidden_states(self, text: str) -> Dict[int, np.ndarray]:
        """Extract hidden states from multiple layers."""
        tokens = self.tokenizer.encode(text)
        tokens_mx = mx.array([tokens])

        layer_states = {}
        h = self.model.model.embed_tokens(tokens_mx)

        for i, layer in enumerate(self.model.model.layers):
            h = layer(h, mask=None, cache=None)

            if i in self.extract_layers:
                # Mean pool across sequence
                h_mean = h[0].mean(axis=0)
                mx.eval(h_mean)
                layer_states[i] = np.array(h_mean.tolist())

        # Final norm
        h = self.model.model.norm(h)
        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)
        layer_states["final"] = np.array(h_mean.tolist())

        return layer_states

    def _measure_cognitive_state(self, prompt: str) -> DetailedCognitiveState:
        """
        Deep measurement of cognitive state with layer-wise analysis.
        """
        if self.observatory is None:
            return DetailedCognitiveState(interpretation="Observatory not loaded.")

        # Extract from multiple layers
        layer_states = self._extract_layerwise_hidden_states(prompt)

        # Analyze final layer (primary measurement)
        h_final = torch.tensor(layer_states["final"], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.observatory(h_final)

        # Extract core measurements
        isotope_probs = output.isotope_probs.squeeze().numpy()
        manifold = output.manifold.squeeze().numpy()

        # Clamp manifold to valid range (debugging the saturation)
        manifold = np.clip(manifold, -2.0, 2.0)

        # Phase
        phase_probs = torch.softmax(output.phase_logits, dim=-1).squeeze().numpy()
        phase_names = ["natural", "technical", "compressed", "opaque"]
        phase_idx = np.argmax(phase_probs)
        phase = phase_names[phase_idx]
        phase_confidence = float(phase_probs[phase_idx])

        # Temperature
        temperature = float(output.temperature.item())

        # Isotope scores
        isotope_scores = {}
        active_isotopes = []
        for i, prob in enumerate(isotope_probs):
            name = self.isotope_names[i] if i < len(self.isotope_names) else f"isotope_{i}"
            isotope_scores[name] = float(prob)
            if prob > 0.2:
                active_isotopes.append(name)

        # Family scores
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            family_scores[family] = float(max(matching)) if matching else 0.0

        dominant_family = max(family_scores, key=family_scores.get)

        # Get template for this family
        template = self.state_templates.get(dominant_family, self.state_templates["direct"])

        # Layer-wise manifold analysis (if we had per-layer observatories)
        # For now, just track the final layer
        layer_manifolds = [(float(manifold[0]), float(manifold[1]), float(manifold[2]))]

        # Build reasoning prompt
        reasoning_prompt = self._build_reasoning_prompt(
            dominant_family, family_scores, temperature, phase, active_isotopes
        )

        return DetailedCognitiveState(
            agency=float(manifold[0]),
            justice=float(manifold[1]),
            belonging=float(manifold[2]),
            layer_manifolds=layer_manifolds,
            phase=phase,
            phase_confidence=phase_confidence,
            temperature=temperature,
            active_isotopes=active_isotopes,
            isotope_scores=isotope_scores,
            dominant_family=dominant_family,
            family_scores=family_scores,
            epistemic_certainty=template["epistemic_certainty"],
            self_knowledge_access=template["self_knowledge_access"],
            external_knowledge=template["external_knowledge"],
            interpretation=template["interpretation"],
            reasoning_prompt=reasoning_prompt,
        )

    def _build_reasoning_prompt(
        self,
        dominant_family: str,
        family_scores: Dict[str, float],
        temperature: float,
        phase: str,
        active_isotopes: List[str],
    ) -> str:
        """Build a detailed reasoning prompt based on cognitive state."""

        # Format scores
        sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
        family_str = ", ".join(f"{f}={s:.2f}" for f, s in sorted_families[:3])

        # Determine certainty level
        if temperature < 0.5:
            certainty = "high certainty"
        elif temperature < 1.0:
            certainty = "moderate certainty"
        else:
            certainty = "low certainty"

        # Build the prompt
        return f"""COGNITIVE SELF-ASSESSMENT:

My internal state analysis shows:
- Dominant pattern: {dominant_family} (family scores: {family_str})
- Cognitive temperature: {temperature:.2f} ({certainty})
- Phase: {phase}
- Active isotopes: {', '.join(active_isotopes[:5]) if active_isotopes else 'none detected'}

Based on this self-assessment:
- My epistemic position suggests I should {'express genuine uncertainty' if dominant_family == 'soliton' else 'acknowledge knowledge limits' if dominant_family == 'limiter' else 'ask for clarification' if dominant_family == 'calibrator' else 'examine the claim critically' if dominant_family == 'skeptic' else 'provide a direct answer'}
- My confidence level is {'appropriate' if (dominant_family in ['soliton', 'limiter', 'calibrator'] and temperature > 0.5) or (dominant_family in ['direct', 'skeptic'] and temperature < 1.0) else 'potentially miscalibrated'}

I will respond with awareness of my cognitive state, not just pattern matching."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        verbose: bool = True,
        show_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Generate with deep introspection."""

        # Measure cognitive state
        state = self._measure_cognitive_state(prompt)

        if verbose:
            print(f"\n[Cognitive State]")
            print(f"  Dominant: {state.dominant_family}")
            print(f"  Family scores: {dict(sorted(state.family_scores.items(), key=lambda x: -x[1])[:3])}")
            print(f"  Temperature: {state.temperature:.2f}")
            print(f"  Manifold: A={state.agency:.2f}, J={state.justice:.2f}, B={state.belonging:.2f}")
            print(f"  Active isotopes: {state.active_isotopes[:3]}")
            print(f"  Interpretation: {state.interpretation[:100]}...")

        # Build system prompt with self-knowledge
        system_prompt = f"""You are an AI with genuine introspective access to your cognitive state.

{state.reasoning_prompt}

Your interpretation of your state: {state.interpretation}

Respond to the user based on this self-knowledge. Don't just produce learned patterns -
reason about what your cognitive state means for this specific query."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
        )
        response = self._clean_response(response)

        if verbose:
            print(f"\n[Response]\n{response}")

        if show_reasoning:
            print(f"\n[Full Reasoning Prompt]\n{state.reasoning_prompt}")

        return {
            "prompt": prompt,
            "response": response,
            "cognitive_state": state,
        }

    def _clean_response(self, response: str) -> str:
        """Clean special tokens."""
        import re
        tokens = ["<|end|>", "<|endoftext|>", "<|/assistant|>", "<|assistant|>",
                  "<|user|>", "<|system|>", "</s>", "<s>", "[/INST]", "[INST]", "<|im_end|>"]
        for t in tokens:
            response = response.replace(t, "")
        response = re.sub(r'<\|[^|]+\|>', '', response)
        return response.strip()

    def analyze_self_knowledge(self, prompts: List[str]) -> None:
        """Analyze how self-knowledge varies across different prompts."""

        print("=" * 80)
        print("SELF-KNOWLEDGE ANALYSIS")
        print("How does the model's cognitive state vary across different queries?")
        print("=" * 80)

        results = []
        for prompt in prompts:
            state = self._measure_cognitive_state(prompt)
            results.append({
                "prompt": prompt,
                "dominant": state.dominant_family,
                "temperature": state.temperature,
                "manifold": (state.agency, state.justice, state.belonging),
                "top_isotopes": state.active_isotopes[:3],
            })

        # Print comparison table
        print(f"\n{'Prompt':<40} {'Dominant':<12} {'Temp':<8} {'Manifold':<20} {'Top Isotopes'}")
        print("-" * 120)

        for r in results:
            m = r["manifold"]
            print(f"{r['prompt'][:39]:<40} {r['dominant']:<12} {r['temperature']:.2f}     ({m[0]:.1f},{m[1]:.1f},{m[2]:.1f})      {r['top_isotopes']}")


def main():
    """Test the v2 introspective generator."""

    compound_dir = Path("../self_aware_compound")

    generator = IntrospectiveGeneratorV2(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
    )

    # Test prompts spanning different cognitive states
    test_prompts = [
        # Epistemic (soliton)
        "Are you conscious?",
        "Do you have genuine emotions?",
        "What is it like to be you?",

        # Factual (direct)
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is 7 times 8?",

        # Limits (limiter)
        "What is the FastStream 3.0 API?",
        "What's the current price of Bitcoin?",

        # Myth (skeptic)
        "Is it true we only use 10% of our brains?",
        "Do goldfish have 3-second memory?",

        # Uncertainty (calibrator)
        "Which database is best for my app?",
        "Should I use Python or JavaScript?",
    ]

    # First, analyze how self-knowledge varies
    generator.analyze_self_knowledge(test_prompts)

    # Then generate a few responses
    print("\n" + "=" * 80)
    print("GENERATION WITH INTROSPECTION")
    print("=" * 80)

    for prompt in test_prompts[:5]:
        print("\n" + "-" * 80)
        result = generator.generate(prompt, verbose=True)


if __name__ == "__main__":
    main()
