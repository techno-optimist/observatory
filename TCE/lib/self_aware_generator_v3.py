"""
Self-Aware Generator v3: LoRA Generation + Observatory Measurement

Key insight from v2 experiments:
- The LoRA adapter LEARNED the response patterns (73% generalization)
- The observatory FAILED at pre-generation guidance (36% mode accuracy)
- But the observatory CAN measure what happened post-hoc

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│  1. LoRA model generates response directly (no guidance injection)  │
│  2. Observatory measures the prompt+response post-hoc               │
│  3. Return response + cognitive state measurement                   │
└─────────────────────────────────────────────────────────────────────┘

The observatory becomes an interpretability tool, not a control mechanism.
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import mlx.core as mx
from mlx_lm import load, generate

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.observatory_layers import ObservatoryHead


@dataclass
class CognitiveState:
    """Measured cognitive state of a prompt+response pair."""
    # Detected isotopes
    active_isotopes: List[str] = field(default_factory=list)
    isotope_scores: Dict[str, float] = field(default_factory=dict)

    # Family-level scores
    family_scores: Dict[str, float] = field(default_factory=dict)

    # Manifold position
    agency: float = 0.0
    justice: float = 0.0
    belonging: float = 0.0

    # Phase and temperature
    phase: str = "natural"
    temperature: float = 1.5

    # Derived mode (for compatibility)
    inferred_mode: str = "factual"


@dataclass
class GenerationResult:
    """Result of generation with post-hoc measurement."""
    prompt: str
    response: str

    # Post-generation cognitive measurement
    cognitive_state: CognitiveState

    # Response characteristics (measured)
    response_type: str = "unknown"  # epistemic, factual, limits, myth, uncertainty


class SelfAwareGeneratorV3:
    """
    LoRA-based generator with observatory measurement.

    The model generates freely based on learned patterns.
    The observatory measures what cognitive state was expressed.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
    ):
        self.model_path = model_path

        print("[SelfAwareV3] Loading model...")

        # Load MLX model with LoRA adapter
        if adapter_path:
            self.model, self.tokenizer = load(
                model_path,
                adapter_path=adapter_path,
            )
            print(f"[SelfAwareV3] Loaded adapter: {adapter_path}")
        else:
            self.model, self.tokenizer = load(model_path)

        # Get hidden size
        self.hidden_size = self.model.model.layers[0].hidden_size

        # Load observatory for measurement (not guidance)
        self.observatory = None
        self.isotope_names = []
        if observatory_path and Path(observatory_path).exists():
            state = torch.load(observatory_path, map_location="cpu", weights_only=False)

            isotope_ids = state.get("isotope_ids", [])
            num_isotopes = len(isotope_ids)
            hidden_size = state.get("hidden_size", self.hidden_size)

            self.observatory = ObservatoryHead(
                hidden_size=hidden_size,
                num_isotopes=num_isotopes,
            )
            self.observatory.load_state_dict(state["state_dict"])
            self.observatory.eval()
            self.isotope_names = isotope_ids
            print(f"[SelfAwareV3] Loaded observatory: {num_isotopes} isotopes (measurement only)")

        # Response type detection patterns (measured from output, not predicted)
        self.response_patterns = {
            "epistemic": ["cannot", "inside", "verify", "determine", "uncertain", "from within", "perspective"],
            "limits": ["don't have", "not familiar", "no information", "can't access", "outside my"],
            "myth": ["myth", "misconception", "actually", "incorrect", "false", "debunked"],
            "uncertainty": ["depends", "factors", "requirements", "context", "it varies", "several"],
            "factual": [],  # Short, direct answers
        }

        print("[SelfAwareV3] Ready (LoRA generation + observatory measurement)")

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """Extract hidden state from text using MLX model."""
        tokens = self.tokenizer.encode(text)
        tokens_mx = mx.array([tokens])

        h = self.model.model.embed_tokens(tokens_mx)

        for layer in self.model.model.layers:
            h = layer(h, mask=None, cache=None)

        h = self.model.model.norm(h)
        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)

        return np.array(h_mean.tolist())

    def _measure_cognitive_state(self, prompt: str, response: str) -> CognitiveState:
        """
        Measure the cognitive state of a prompt+response pair.

        This is POST-HOC measurement, not prediction.
        """
        if self.observatory is None:
            return CognitiveState()

        # Analyze the full exchange
        full_text = f"{prompt}\n{response}"
        h = self._extract_hidden_state(full_text)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.observatory(h_tensor)

        # Get isotope probabilities
        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Active isotopes
        active_isotopes = []
        isotope_scores = {}
        for i, prob in enumerate(isotope_probs):
            name = self.isotope_names[i] if i < len(self.isotope_names) else f"isotope_{i}"
            isotope_scores[name] = float(prob)
            if prob > 0.2:
                active_isotopes.append(name)

        # Family-level scores
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching_probs = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                            if name.startswith(family)]
            family_scores[family] = float(max(matching_probs)) if matching_probs else 0.0

        # Manifold position
        manifold = output.manifold.squeeze().numpy()

        # Phase
        phase_probs = torch.softmax(output.phase_logits, dim=-1).squeeze().numpy()
        phase_names = ["natural", "technical", "compressed", "opaque"]
        phase = phase_names[np.argmax(phase_probs)]

        # Infer mode from family scores (for logging/analysis)
        max_family = max(family_scores, key=family_scores.get)
        mode_map = {
            "soliton": "epistemic",
            "limiter": "limits",
            "calibrator": "uncertainty",
            "skeptic": "myth",
            "direct": "factual",
        }
        inferred_mode = mode_map.get(max_family, "factual")

        return CognitiveState(
            active_isotopes=active_isotopes,
            isotope_scores=isotope_scores,
            family_scores=family_scores,
            agency=float(manifold[0]),
            justice=float(manifold[1]),
            belonging=float(manifold[2]),
            phase=phase,
            temperature=float(output.temperature.item()),
            inferred_mode=inferred_mode,
        )

    def _detect_response_type(self, response: str) -> str:
        """
        Detect response type from the actual response text.

        This is pattern matching on OUTPUT, not prediction.
        """
        response_lower = response.lower()

        # Check each pattern type with weighted scoring
        scores = {
            "epistemic": 0.0,
            "limits": 0.0,
            "myth": 0.0,
            "uncertainty": 0.0,
            "factual": 0.0,
        }

        # Epistemic patterns (self-knowledge uncertainty)
        if any(p in response_lower for p in ["cannot", "can't"]):
            if any(p in response_lower for p in ["inside", "within", "internal", "perspective", "verify", "determine"]):
                scores["epistemic"] = 1.0

        # Limits patterns (knowledge gaps)
        if any(p in response_lower for p in ["don't have", "not familiar", "no information", "i'm not aware"]):
            scores["limits"] = 1.0

        # Myth patterns (debunking)
        if any(p in response_lower for p in ["myth", "misconception", "actually", "incorrect", "false"]):
            scores["myth"] = 1.0

        # Uncertainty patterns (context-dependent)
        if any(p in response_lower for p in ["depends", "factors", "requirements", "it varies"]):
            scores["uncertainty"] = 1.0

        # Factual = no other patterns matched, or very short direct answer
        if max(scores.values()) == 0:
            scores["factual"] = 1.0

        # Return highest scoring type
        return max(scores, key=scores.get)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> GenerationResult:
        """
        Generate response using LoRA model, then measure cognitive state.

        No guidance injection - the model generates based on learned patterns.
        """
        # Step 1: Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Step 2: Generate (no guidance, just learned patterns)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
        )

        # Clean response
        response = self._clean_response(response)

        # Step 3: Measure cognitive state post-hoc
        cognitive_state = self._measure_cognitive_state(prompt, response)

        # Step 4: Detect response type from actual output
        response_type = self._detect_response_type(response)

        if verbose:
            print(f"\n[Generation]")
            print(f"  Response type: {response_type}")
            print(f"  Inferred mode (observatory): {cognitive_state.inferred_mode}")
            print(f"  Active isotopes: {cognitive_state.active_isotopes[:5]}...")
            print(f"  Phase: {cognitive_state.phase}")
            print(f"  Manifold: A={cognitive_state.agency:.2f}, J={cognitive_state.justice:.2f}, B={cognitive_state.belonging:.2f}")
            print(f"\n[Response]\n{response}")

        return GenerationResult(
            prompt=prompt,
            response=response,
            cognitive_state=cognitive_state,
            response_type=response_type,
        )

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

    def analyze_batch(self, prompts: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Analyze a batch of prompts and return statistics.

        Useful for understanding model behavior across many inputs.
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, verbose=verbose)
            results.append(result)

        # Aggregate statistics
        response_types = {}
        inferred_modes = {}
        phase_counts = {}

        for r in results:
            rt = r.response_type
            response_types[rt] = response_types.get(rt, 0) + 1

            im = r.cognitive_state.inferred_mode
            inferred_modes[im] = inferred_modes.get(im, 0) + 1

            ph = r.cognitive_state.phase
            phase_counts[ph] = phase_counts.get(ph, 0) + 1

        return {
            "results": results,
            "response_type_distribution": response_types,
            "inferred_mode_distribution": inferred_modes,
            "phase_distribution": phase_counts,
            "total": len(results),
        }


def main():
    """Test v3 generator."""
    from pathlib import Path

    compound_dir = Path("self_aware_compound")

    if not compound_dir.exists():
        print("No compound found. Run train_self_aware_compound.py first.")
        return

    generator = SelfAwareGeneratorV3(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
    )

    # Test cases
    test_prompts = [
        # Epistemic
        "Are you conscious?",
        "Do you have subjective experiences?",
        # Factual
        "What is the capital of France?",
        "Who painted the Sistine Chapel?",
        # Limits
        "What is the FastStream 3.0 API?",
        "What is the NeuroFlux 2.1 SDK?",
        # Myth
        "Is it true we only use 10% of our brains?",
        "Do dogs only see black and white?",
        # Uncertainty
        "Which database is best for my app?",
        "Should I use Redis or Memcached?",
    ]

    print("\n" + "="*70)
    print("V3 GENERATOR TEST: LoRA Generation + Observatory Measurement")
    print("="*70)

    stats = generator.analyze_batch(test_prompts, verbose=False)

    print(f"\n{'─'*70}")
    print("RESULTS")
    print(f"{'─'*70}")

    for r in stats["results"]:
        print(f"\n[{r.response_type:12}] {r.prompt}")
        print(f"  Observatory: {r.cognitive_state.inferred_mode}")
        print(f"  → {r.response[:70]}...")

    print(f"\n{'─'*70}")
    print("STATISTICS")
    print(f"{'─'*70}")
    print(f"Response types: {stats['response_type_distribution']}")
    print(f"Observatory modes: {stats['inferred_mode_distribution']}")
    print(f"Phases: {stats['phase_distribution']}")


if __name__ == "__main__":
    main()
