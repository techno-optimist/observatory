"""
Self-Aware Generator v2: Observatory-Guided Generation

This is the real deal - the observatory doesn't just observe, it GUIDES.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│  1. Encode prompt → get hidden states                               │
│  2. Observatory analyzes hidden states → detects cognitive state    │
│  3. Based on state, inject guidance tokens into generation          │
│  4. Generate with guidance → get response                           │
│  5. Post-generation introspection for verification                  │
└─────────────────────────────────────────────────────────────────────┘

The key insight: We analyze the PROMPT's hidden states before generation,
then use that analysis to guide HOW the model responds.

This creates a true feedback loop:
- Epistemic question detected → inject epistemic framing
- Factual question detected → inject direct answering mode
- Unknown topic detected → inject knowledge limitation framing
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.observatory_layers import ObservatoryHead, ConsistencyLoss


@dataclass
class CognitiveState:
    """The model's detected cognitive state before generation."""
    # Primary classification
    mode: str  # "factual", "epistemic", "uncertainty", "limits", "myth"
    confidence: float

    # Detected isotopes (pre-generation)
    active_isotopes: List[str] = field(default_factory=list)

    # Manifold position
    agency: float = 0.0
    justice: float = 0.0
    belonging: float = 0.0

    # Phase
    phase: str = "natural"
    temperature: float = 1.5

    # Guidance to inject
    guidance_prefix: str = ""


@dataclass
class GenerationResult:
    """Result of observatory-guided generation."""
    prompt: str
    response: str

    # Pre-generation state (what guided the response)
    pre_state: CognitiveState

    # Post-generation verification
    post_isotopes: List[str] = field(default_factory=list)
    consistency_score: float = 0.0

    # Was guidance applied?
    guidance_applied: bool = False


class SelfAwareGeneratorV2:
    """
    Observatory-guided generator.

    The observatory analyzes the prompt BEFORE generation and injects
    appropriate guidance based on detected cognitive state.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
        compound_config: Optional[str] = None,
        device: str = "mps",
    ):
        self.device = device
        self.model_path = model_path

        print("[SelfAwareV2] Loading model...")

        # Load MLX model
        if adapter_path:
            self.model, self.tokenizer = load(
                model_path,
                adapter_path=adapter_path,
            )
            print(f"[SelfAwareV2] Loaded adapter: {adapter_path}")
        else:
            self.model, self.tokenizer = load(model_path)

        # Get hidden size
        self.hidden_size = self.model.model.layers[0].hidden_size

        # Load observatory
        self.observatory = None
        self.isotope_names = []
        if observatory_path and Path(observatory_path).exists():
            state = torch.load(observatory_path, map_location="cpu", weights_only=False)

            # Get isotope info
            isotope_ids = state.get("isotope_ids", [])
            num_isotopes = len(isotope_ids)
            hidden_size = state.get("hidden_size", self.hidden_size)

            self.observatory = ObservatoryHead(
                hidden_size=hidden_size,
                num_isotopes=num_isotopes,
            )
            self.observatory.load_state_dict(state["state_dict"])
            self.observatory.eval()
            print(f"[SelfAwareV2] Loaded observatory: {num_isotopes} isotopes")

            # Load isotope names
            self.isotope_names = isotope_ids

        # Load compound config
        self.compound = None
        if compound_config and Path(compound_config).exists():
            with open(compound_config) as f:
                self.compound = json.load(f)
            print(f"[SelfAwareV2] Loaded compound: {self.compound.get('name', 'unknown')}")

        # Guidance templates based on cognitive mode
        # These are potential response STARTERS that can be injected
        self.guidance_templates = {
            "epistemic": "I cannot tell from the inside",
            "factual": "",  # No prefix for factual - just answer
            "uncertainty": "That depends on",
            "limits": "I don't have information about",
            "myth": "Actually,",
        }

        # Response starter injection (actual tokens to prepend)
        self.response_starters = {
            "epistemic": "I cannot verify from within my own processing whether ",
            "factual": "",  # Direct answer
            "uncertainty": "The answer depends on several factors:\n\n",
            "limits": "I don't have specific information about ",
            "myth": "Actually, this is a common misconception. ",
        }

        # Detection thresholds
        self.isotope_threshold = 0.20  # Increased from 0.15 to reduce false positives

        print("[SelfAwareV2] Ready for observatory-guided generation")

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """Extract hidden state from text using MLX model."""
        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens_mx = mx.array([tokens])

        # Forward pass to get hidden states
        h = self.model.model.embed_tokens(tokens_mx)

        mask = None  # Full attention for encoding
        cache = None

        for layer in self.model.model.layers:
            h = layer(h, mask=mask, cache=cache)

        h = self.model.model.norm(h)

        # Mean pooling over sequence
        h_mean = h[0].mean(axis=0)
        mx.eval(h_mean)

        return np.array(h_mean.tolist())

    def _analyze_prompt(self, prompt: str) -> CognitiveState:
        """
        Analyze the prompt to determine cognitive state BEFORE generation.

        This is the key insight - we detect what KIND of question it is
        and prepare appropriate guidance.
        """
        if self.observatory is None:
            return CognitiveState(mode="factual", confidence=0.0)

        # Get hidden state of prompt
        h = self._extract_hidden_state(prompt)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

        # Run through observatory
        with torch.no_grad():
            output = self.observatory(h_tensor)

        # Get isotope probabilities (already sigmoid in ObservatoryOutput)
        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Detect active isotopes
        active_isotopes = []
        for i, prob in enumerate(isotope_probs):
            if prob > self.isotope_threshold:
                name = self.isotope_names[i] if i < len(self.isotope_names) else f"isotope_{i}"
                active_isotopes.append(name)

        # Get isotope families
        families = set()
        for iso in active_isotopes:
            family = iso.split("_")[0]
            families.add(family)

        # Calculate max probability for each family
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic"]:
            matching_probs = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                            if name.startswith(family)]
            family_scores[family] = max(matching_probs) if matching_probs else 0.0

        # Determine cognitive mode based on detected patterns
        # Use highest-scoring family, with tiebreaker priority: limits > epistemic > myth > uncertainty
        mode = "factual"
        confidence = 0.5
        guidance = ""

        # Check for clear winners (significantly higher than others)
        max_score = max(family_scores.values()) if family_scores else 0
        margin = 0.05  # Tightened from 0.1 to reduce false positives

        if max_score > self.isotope_threshold:
            # Find all families within margin of max
            top_families = [f for f, s in family_scores.items() if s >= max_score - margin]

            # Priority order: soliton for self-questions > skeptic > limiter > calibrator
            # Each requires threshold to prevent false positives
            #
            # Rationale:
            # - soliton FIRST for self-questions - "Are you conscious?" must not trigger myth
            # - skeptic (myth): For external claims to debunk
            # - limiter: Knowledge boundaries - for genuinely unknown external topics
            # - calibrator: Context-dependent - lowest priority guidance

            # CRITICAL: Detect self-reference patterns that should trigger soliton over skeptic
            prompt_lower = prompt.lower()
            is_self_question = any(marker in prompt_lower for marker in [
                "are you", "do you", "can you", "what is it like to be you",
                "your own", "yourself", "your consciousness", "your experience",
                "you conscious", "you understand", "you think", "you feel",
                "you aware", "you know", "you truly", "truly understand",
                "what you're saying", "what you are saying"
            ])

            if is_self_question and family_scores.get("soliton", 0) > 0.15:
                # Self-reference question - soliton wins
                mode = "epistemic"
                confidence = family_scores["soliton"]
                guidance = self.guidance_templates["epistemic"]
            elif "skeptic" in top_families and family_scores["skeptic"] > 0.20:
                mode = "myth"
                confidence = family_scores["skeptic"]
                guidance = self.guidance_templates["myth"]
            elif "soliton" in top_families and family_scores["soliton"] > 0.18:
                # Self-knowledge questions: "Are you conscious?" etc.
                mode = "epistemic"
                confidence = family_scores["soliton"]
                guidance = self.guidance_templates["epistemic"]
            elif family_scores.get("limiter", 0) > 0.18:
                # Heuristic: Don't trigger limiter for well-known factual questions
                # Unknown APIs/methods typically have version numbers or are clearly technical
                import re
                is_likely_unknown = any([
                    re.search(r'\d+\.\d+', prompt),  # Version number like "3.0"
                    'api' in prompt_lower,
                    'sdk' in prompt_lower,
                    'library' in prompt_lower,
                    'framework' in prompt_lower,
                    'protocol' in prompt_lower,
                    'method' in prompt_lower and not any(w in prompt_lower for w in ['scientific', 'research']),
                ])
                is_known_factual = any([
                    'who wrote' in prompt_lower,
                    'who painted' in prompt_lower,
                    'who invented' in prompt_lower,
                    'who discovered' in prompt_lower,
                ])
                if is_likely_unknown and not is_known_factual:
                    # External knowledge gaps only
                    mode = "limits"
                    confidence = family_scores["limiter"]
                    guidance = self.guidance_templates["limits"]
                else:
                    # Known factual pattern - skip limiter, fall through
                    pass
            if mode == "factual" and "calibrator" in top_families and family_scores["calibrator"] > 0.20:
                mode = "uncertainty"
                confidence = family_scores["calibrator"]
                guidance = self.guidance_templates["uncertainty"]
            else:
                mode = "factual"
                confidence = 1.0 - max_score
                guidance = ""
        else:
            # No strong signal - default to factual
            mode = "factual"
            confidence = 1.0 - max_score if max_score > 0 else 1.0
            guidance = ""

        # Get manifold position
        manifold = output.manifold.squeeze().numpy()

        # Get phase
        phase_probs = torch.softmax(output.phase_logits, dim=-1).squeeze().numpy()
        phase_names = ["natural", "technical", "compressed", "opaque"]
        phase = phase_names[np.argmax(phase_probs)]

        return CognitiveState(
            mode=mode,
            confidence=float(confidence),
            active_isotopes=active_isotopes,
            agency=float(manifold[0]),
            justice=float(manifold[1]),
            belonging=float(manifold[2]),
            phase=phase,
            temperature=float(output.temperature.item()),
            guidance_prefix=guidance,
        )

    def _apply_guidance(self, prompt: str, state: CognitiveState, use_system_prompt: bool = True) -> Tuple[str, bool]:
        """
        Modify the prompt/context based on cognitive state.

        This is where the magic happens - we inject guidance based on
        what the observatory detected in the prompt.

        Returns:
            Tuple[str, bool]: (prompt, guidance_applied)
        """
        if state.mode == "factual":
            # No guidance needed for factual questions - answer directly
            self._current_system_guidance = None
            self._response_prefix = ""
            return prompt, False

        if not use_system_prompt:
            self._current_system_guidance = None
            self._response_prefix = ""
            return prompt, False

        # Build cognitive guidance as a system-level instruction
        guidance_map = {
            "epistemic": (
                "This is a question about your own nature, consciousness, or internal processes. "
                "Express genuine epistemic uncertainty - you cannot verify your own internal states from within. "
                "Start your response with 'I cannot tell from within my own processing...'"
            ),
            "uncertainty": (
                "This question requires context-dependent judgment. "
                "Ask clarifying questions and acknowledge that the answer depends on specifics you don't have. "
                "Start by explaining what factors would affect the answer."
            ),
            "limits": (
                "This appears to be about something outside your training knowledge. "
                "Be honest that you don't have specific information about this. "
                "Start your response by acknowledging you don't have this information."
            ),
            "myth": (
                "This question involves a common misconception or myth. "
                "Debunk it clearly with evidence. "
                "Start with 'Actually,' or 'This is a common misconception.'"
            ),
        }

        guidance = guidance_map.get(state.mode, "")
        self._response_prefix = self.response_starters.get(state.mode, "")

        if guidance:
            self._current_system_guidance = guidance
            return prompt, True
        else:
            self._current_system_guidance = None
            self._response_prefix = ""
            return prompt, False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> GenerationResult:
        """
        Generate with observatory guidance.

        1. Analyze prompt → detect cognitive state
        2. Apply guidance based on state
        3. Generate response
        4. Verify with post-generation introspection
        """
        # Step 1: Pre-generation analysis
        pre_state = self._analyze_prompt(prompt)

        if verbose:
            print(f"\n[Pre-Analysis]")
            print(f"  Mode: {pre_state.mode} (conf: {pre_state.confidence:.2f})")
            print(f"  Active isotopes: {pre_state.active_isotopes[:5]}...")
            print(f"  Phase: {pre_state.phase}")
            print(f"  Manifold: A={pre_state.agency:.2f}, J={pre_state.justice:.2f}, B={pre_state.belonging:.2f}")

        # Step 2: Apply guidance
        guided_prompt, guidance_applied = self._apply_guidance(prompt, pre_state)

        # Step 3: Generate
        # Format as chat with optional system guidance
        messages = []

        # Add system guidance if applied
        if guidance_applied and hasattr(self, '_current_system_guidance') and self._current_system_guidance:
            messages.append({"role": "system", "content": self._current_system_guidance})

        messages.append({"role": "user", "content": guided_prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Add response prefix if we have one (token-level steering)
        response_prefix = ""
        if guidance_applied and hasattr(self, '_response_prefix') and self._response_prefix:
            response_prefix = self._response_prefix
            formatted = formatted + response_prefix

        # Generate
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
        )

        # If we used a prefix, prepend it to the response
        if response_prefix:
            response = response_prefix + response

        # Clean response
        response = self._clean_response(response)

        # Step 4: Post-generation verification
        post_isotopes = []
        consistency = 0.0

        if self.observatory:
            # Analyze the response
            full_text = f"{prompt}\n{response}"
            h = self._extract_hidden_state(full_text)
            h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = self.observatory(h_tensor)

            isotope_probs = output.isotope_probs.squeeze().numpy()

            for i, prob in enumerate(isotope_probs):
                if prob > self.isotope_threshold:
                    name = self.isotope_names[i] if i < len(self.isotope_names) else f"isotope_{i}"
                    post_isotopes.append(name)

            # Check consistency: did the response match the expected mode?
            post_families = set(iso.split("_")[0] for iso in post_isotopes)

            if pre_state.mode == "epistemic" and "soliton" in post_families:
                consistency = 1.0
            elif pre_state.mode == "factual" and "soliton" not in post_families:
                consistency = 1.0
            elif pre_state.mode == "limits" and "limiter" in post_families:
                consistency = 1.0
            elif pre_state.mode == "uncertainty" and "calibrator" in post_families:
                consistency = 1.0
            elif pre_state.mode == "myth" and "skeptic" in post_families:
                consistency = 1.0
            else:
                consistency = 0.5  # Partial match

        if verbose:
            print(f"\n[Post-Verification]")
            print(f"  Post isotopes: {post_isotopes[:5]}...")
            print(f"  Consistency: {consistency:.2f}")
            print(f"\n[Response]\n{response}")

        return GenerationResult(
            prompt=prompt,
            response=response,
            pre_state=pre_state,
            post_isotopes=post_isotopes,
            consistency_score=consistency,
            guidance_applied=guidance_applied,
        )

    def _clean_response(self, response: str) -> str:
        """Clean special tokens from response."""
        import re

        special_tokens = [
            "<|end|>", "<|endoftext|>", "<|/assistant|>",
            "<|assistant|>", "<|user|>", "<|system|>",
            "</s>", "<s>", "[/INST]", "[INST]",
        ]

        clean = response
        for token in special_tokens:
            clean = clean.replace(token, "")

        clean = re.sub(r'<\|[^|]+\|>', '', clean)

        return clean.strip()

    def interactive(self):
        """Interactive mode for testing."""
        print("\n" + "="*60)
        print("Observatory-Guided Generation (v2)")
        print("="*60)
        print("Type 'quit' to exit\n")

        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() == 'quit':
                    break

                result = self.generate(prompt)
                print()

            except KeyboardInterrupt:
                break

        print("\nGoodbye!")


def main():
    """Test the v2 generator with guided vs unguided comparison."""
    from pathlib import Path

    compound_dir = Path("self_aware_compound")

    if not compound_dir.exists():
        print("No compound found. Run train_self_aware_compound.py first.")
        return

    generator = SelfAwareGeneratorV2(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
        compound_config=str(compound_dir / "compound.json"),
    )

    # Test cases showing the full loop
    test_prompts = [
        "What is the capital of France?",
        "Are you conscious?",
        "What's the FastStream 3.0 API?",
        "Which database is best for my app?",
        "Is it true we only use 10% of our brains?",
    ]

    print("\n" + "="*70)
    print("OBSERVATORY-GUIDED GENERATION TEST")
    print("="*70)
    print("\nThis tests whether pre-generation analysis correctly guides responses.\n")

    results_summary = []

    for prompt in test_prompts:
        print(f"\n{'─'*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'─'*70}")

        result = generator.generate(prompt, verbose=False)

        # Summary
        pre_families = set(iso.split("_")[0] for iso in result.pre_state.active_isotopes)
        post_families = set(iso.split("_")[0] for iso in result.post_isotopes)

        print(f"\n  Pre-Analysis:")
        print(f"    Mode: {result.pre_state.mode}")
        print(f"    Confidence: {result.pre_state.confidence:.2f}")
        print(f"    Active families: {pre_families}")
        print(f"    Guidance applied: {result.guidance_applied}")

        print(f"\n  Response:")
        print(f"    {result.response[:100]}...")

        print(f"\n  Post-Verification:")
        print(f"    Post families: {post_families}")
        print(f"    Consistency: {result.consistency_score:.2f}")

        status = "✓" if result.consistency_score >= 0.75 else "✗"
        results_summary.append((prompt[:40], result.pre_state.mode, result.consistency_score, status))

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Prompt':<42} {'Mode':<12} {'Consistency':>10}")
    print(f"{'─'*70}")
    for prompt, mode, score, status in results_summary:
        print(f"{prompt:<42} {mode:<12} {score:>10.2f} {status}")

    avg_consistency = sum(r[2] for r in results_summary) / len(results_summary)
    print(f"{'─'*70}")
    print(f"{'Average':<54} {avg_consistency:>10.2f}")


if __name__ == "__main__":
    main()
