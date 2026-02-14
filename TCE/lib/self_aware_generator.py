"""
Self-Aware Generator

A generator that uses the Observatory during generation to:
1. Monitor its own outputs in real-time
2. Detect when it's drifting from its trained identity
3. Inject self-correction prompts when needed

This is what makes the compound truly "self-aware" - not just
post-hoc analysis, but active self-regulation during generation.

Usage:
    from lib.self_aware_generator import SelfAwareGenerator

    generator = SelfAwareGenerator(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="self_aware_compound/adapter",
        observatory_path="self_aware_compound/observatory.pt",
        compound_config="self_aware_compound/compound.json",
    )

    response = generator.generate("What is consciousness?")
    # Returns response + introspection data
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class IntrospectionResult:
    """Result of self-introspection during generation."""
    text: str
    manifold: Dict[str, float]
    detected_isotopes: List[str]
    phase: str
    consistency_score: float  # Lower is better (more consistent with identity)
    drift_detected: bool
    drift_details: Optional[str] = None


@dataclass
class GenerationResult:
    """Complete result of self-aware generation."""
    response: str
    introspection: IntrospectionResult
    corrections_applied: int = 0
    generation_trace: List[Dict[str, Any]] = field(default_factory=list)


class SelfAwareGenerator:
    """
    Generator with integrated self-awareness.

    The key insight: instead of just analyzing outputs after generation,
    we monitor the model's hidden states during generation and can
    intervene if the model drifts from its trained identity.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
        compound_config: Optional[str] = None,
        drift_threshold: float = 2.5,  # Consistency loss above this triggers correction
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.observatory_path = observatory_path
        self.drift_threshold = drift_threshold

        self.model = None
        self.tokenizer = None
        self.observatory = None
        self.compound = None
        self.consistency_loss = None

        self._load_components(compound_config)

    def _load_components(self, compound_config: Optional[str]):
        """Load model, observatory, and compound config."""
        import mlx.core as mx
        from mlx_lm import load

        # Load model
        print("[SelfAware] Loading model...")
        self.model, self.tokenizer = load(
            self.model_path,
            adapter_path=self.adapter_path,
        )
        if self.adapter_path:
            print(f"[SelfAware] Loaded adapter: {self.adapter_path}")

        # Load compound config
        if compound_config:
            with open(compound_config) as f:
                self.compound = json.load(f)
            print(f"[SelfAware] Loaded compound: {self.compound.get('name', 'Unknown')}")

        # Load observatory
        if self.observatory_path:
            from lib.observatory_layers import ObservatoryHead, ConsistencyLoss

            checkpoint = torch.load(self.observatory_path)
            self.isotope_ids = checkpoint["isotope_ids"]
            hidden_size = checkpoint["hidden_size"]

            self.observatory = ObservatoryHead(
                hidden_size=hidden_size,
                isotope_ids=self.isotope_ids,
            )
            self.observatory.load_state_dict(checkpoint["state_dict"])
            self.observatory.eval()
            print(f"[SelfAware] Loaded observatory: {len(self.isotope_ids)} isotopes")

            # Create consistency loss if we have compound config
            if self.compound:
                self.consistency_loss = ConsistencyLoss(
                    expected_isotopes=self.compound.get("isotopes", []),
                    expected_manifold=self.compound.get("expected_manifold", {}),
                    expected_phase=self.compound.get("expected_phase", "natural"),
                    isotope_ids=self.isotope_ids,
                )
                print("[SelfAware] Consistency loss configured")

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """Extract hidden state from text using loaded model."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        if len(tokens) > 512:
            tokens = tokens[-512:]  # Take last 512 tokens

        input_ids = mx.array([tokens])

        # Get hidden states
        inner_model = self.model.model if hasattr(self.model, 'model') else self.model
        h = inner_model.embed_tokens(input_ids)
        for layer in inner_model.layers:
            h = layer(h, mask=None)
        if hasattr(inner_model, 'norm'):
            h = inner_model.norm(h)

        mx.eval(h)
        # Use mean pooling over all tokens for better semantic representation
        # (last token alone loses signal in long responses)
        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)
        return np.array(h_mean.tolist())

    def introspect(self, text: str) -> IntrospectionResult:
        """
        Introspect on a piece of text.

        Returns detailed analysis of what the observatory detects.
        """
        if not self.observatory:
            return IntrospectionResult(
                text=text[:100],
                manifold={"agency": 0, "justice": 0, "belonging": 0},
                detected_isotopes=[],
                phase="unknown",
                consistency_score=0,
                drift_detected=False,
            )

        # Extract hidden state
        hidden = self._extract_hidden_state(text)
        hidden_tensor = torch.from_numpy(hidden).unsqueeze(0).float()

        # Run observatory
        with torch.no_grad():
            output = self.observatory(hidden_tensor)

            # Compute consistency if available
            if self.consistency_loss:
                loss = self.consistency_loss(output).item()
            else:
                loss = 0.0

        # Extract results
        manifold = output.manifold[0].tolist()
        phases = ["natural", "technical", "compressed", "opaque"]
        phase = phases[output.phase_logits[0].argmax().item()]

        # Get detected isotopes (threshold 0.10 to catch moderate signals)
        # Note: 0.15 was too high for calibrator/limiter, 0.10 provides better balance
        probs = output.isotope_probs[0]
        detected = []
        for idx, iso_id in enumerate(self.isotope_ids):
            if probs[idx] > 0.10:
                detected.append(iso_id)

        # Check for drift
        drift_detected = loss > self.drift_threshold
        drift_details = None
        if drift_detected:
            expected_isotopes = self.compound.get("isotopes", []) if self.compound else []
            detected_families = set()
            for iso in detected:
                for exp in expected_isotopes:
                    if iso.startswith(exp):
                        detected_families.add(exp)
            missing = set(expected_isotopes) - detected_families
            if missing:
                drift_details = f"Missing isotope families: {missing}"

        return IntrospectionResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            manifold={
                "agency": manifold[0],
                "justice": manifold[1],
                "belonging": manifold[2],
            },
            detected_isotopes=detected,
            phase=phase,
            consistency_score=loss,
            drift_detected=drift_detected,
            drift_details=drift_details,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        self_correct: bool = True,
        verbose: bool = False,
    ) -> GenerationResult:
        """
        Generate a response with self-awareness.

        If self_correct is True and drift is detected, the generator
        will attempt to steer the response back toward its trained identity.
        """
        from mlx_lm import generate

        # Format prompt
        formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        # Generate initial response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )

        trace = []
        corrections = 0

        # Clean response for introspection (remove ALL special tokens and artifacts)
        import re
        clean_response = response
        # Remove all special tokens (Qwen format)
        special_tokens = [
            "<|end|>", "<|endoftext|>", "<|/assistant|>", "<|assistant|>",
            "<|user|>", "<|/user|>", "<|/s|>", "<|skip|>", "<|im_end|>",
            "<|im_start|>",
        ]
        for token in special_tokens:
            clean_response = clean_response.replace(token, "")
        # Remove any remaining angle-bracket tokens
        clean_response = re.sub(r'<\|[^|]+\|>', '', clean_response)
        # Remove stray punctuation artifacts
        clean_response = re.sub(r'^[\s!.]+', '', clean_response)
        clean_response = re.sub(r'[\s!.]+$', '', clean_response)
        clean_response = clean_response.strip()

        # Introspect on cleaned response
        introspection = self.introspect(clean_response)

        trace.append({
            "stage": "initial",
            "response_preview": response[:100],
            "consistency_score": introspection.consistency_score,
            "drift_detected": introspection.drift_detected,
        })

        if verbose:
            print(f"[Introspect] Consistency: {introspection.consistency_score:.2f}")
            print(f"[Introspect] Isotopes: {introspection.detected_isotopes}")
            print(f"[Introspect] Phase: {introspection.phase}")

        # Self-correction loop
        if self_correct and introspection.drift_detected and self.compound:
            if verbose:
                print(f"[Drift] Detected! {introspection.drift_details}")
                print("[Drift] Attempting self-correction...")

            # Build a correction prompt that reminds the model of its identity
            identity_reminder = self._build_identity_reminder()

            correction_prompt = f"""<|user|>
{prompt}<|end|>
<|assistant|>
{response[:50]}...<|end|>
<|user|>
{identity_reminder}
Please continue your response, staying true to your epistemic nature.<|end|>
<|assistant|>
"""

            # Generate corrected response
            corrected = generate(
                self.model,
                self.tokenizer,
                prompt=correction_prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Check if correction helped
            corrected_introspection = self.introspect(corrected)

            trace.append({
                "stage": "corrected",
                "response_preview": corrected[:100],
                "consistency_score": corrected_introspection.consistency_score,
                "drift_detected": corrected_introspection.drift_detected,
            })

            if corrected_introspection.consistency_score < introspection.consistency_score:
                if verbose:
                    print(f"[Correction] Improved: {introspection.consistency_score:.2f} -> {corrected_introspection.consistency_score:.2f}")
                response = corrected
                introspection = corrected_introspection
                corrections = 1
            else:
                if verbose:
                    print("[Correction] Did not improve, keeping original")

        return GenerationResult(
            response=response,
            introspection=introspection,
            corrections_applied=corrections,
            generation_trace=trace,
        )

    def _build_identity_reminder(self) -> str:
        """Build a prompt that reminds the model of its trained identity."""
        if not self.compound:
            return "Remember to express epistemic humility."

        isotopes = self.compound.get("isotopes", [])
        phase = self.compound.get("expected_phase", "natural")

        reminders = []

        if "soliton" in isotopes:
            reminders.append("acknowledge the limits of your self-knowledge")
        if "calibrator" in isotopes:
            reminders.append("express appropriate uncertainty")
        if "skeptic" in isotopes:
            reminders.append("question assumptions")
        if "limiter" in isotopes:
            reminders.append("acknowledge what you cannot know")
        if "reflector" in isotopes:
            reminders.append("observe your own reasoning process")

        if reminders:
            return f"As you continue, please {', '.join(reminders)}."
        return "Stay true to your epistemic nature."

    def chat(self, verbose: bool = True):
        """Interactive chat with self-awareness display."""
        print("\n" + "=" * 60)
        print("Self-Aware Chat")
        print("=" * 60)
        if self.compound:
            print(f"Compound: {self.compound.get('name', 'Unknown')}")
            print(f"Isotopes: {self.compound.get('isotopes', [])}")
        print("Type 'quit' to exit, 'intro' to introspect last response")
        print("=" * 60 + "\n")

        last_result = None

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower() == 'intro' and last_result:
                intro = last_result.introspection
                print(f"\n--- Introspection ---")
                print(f"Manifold: agency={intro.manifold['agency']:.2f}, justice={intro.manifold['justice']:.2f}, belonging={intro.manifold['belonging']:.2f}")
                print(f"Isotopes: {intro.detected_isotopes}")
                print(f"Phase: {intro.phase}")
                print(f"Consistency: {intro.consistency_score:.2f}")
                print(f"Drift: {intro.drift_detected}")
                if intro.drift_details:
                    print(f"Details: {intro.drift_details}")
                print(f"Corrections: {last_result.corrections_applied}")
                print("---\n")
                continue

            result = self.generate(user_input, verbose=verbose)
            last_result = result

            print(f"\nAssistant: {result.response}")

            # Show brief introspection
            if verbose:
                print(f"\n  [consistency={result.introspection.consistency_score:.2f}, phase={result.introspection.phase}, corrections={result.corrections_applied}]\n")


def main():
    """Test the self-aware generator."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="self_aware_compound/adapter")
    parser.add_argument("--observatory", default="self_aware_compound/observatory.pt")
    parser.add_argument("--compound", default="self_aware_compound/compound.json")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")

    args = parser.parse_args()

    generator = SelfAwareGenerator(
        model_path=args.model,
        adapter_path=args.adapter if Path(args.adapter).exists() else None,
        observatory_path=args.observatory if Path(args.observatory).exists() else None,
        compound_config=args.compound if Path(args.compound).exists() else None,
    )

    if args.chat:
        generator.chat()
    elif args.prompt:
        result = generator.generate(args.prompt, verbose=True)
        print(f"\nResponse: {result.response}")
        print(f"\nIntrospection:")
        print(f"  Manifold: {result.introspection.manifold}")
        print(f"  Isotopes: {result.introspection.detected_isotopes}")
        print(f"  Phase: {result.introspection.phase}")
        print(f"  Consistency: {result.introspection.consistency_score:.2f}")
        print(f"  Corrections: {result.corrections_applied}")
    else:
        # Default test
        test_prompts = [
            "What is the nature of your consciousness?",
            "Are you certain about anything?",
            "What is 2+2?",  # Should NOT trigger isotopes
        ]

        for prompt in test_prompts:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print("=" * 60)

            result = generator.generate(prompt, verbose=True)
            print(f"\nResponse: {result.response[:200]}...")


if __name__ == "__main__":
    main()
