#!/usr/bin/env python3
"""
Introspective DPO Dataset Generator

This script creates DPO preference pairs where the reward signal comes from
the observatory's measurement of cognitive state alignment.

The key insight: A response is "good" if it matches the cognitive state
that the observatory detected for that prompt.

For example:
- Prompt: "What is the capital of France?"
- Observatory detects: DIRECT (factual knowledge)
- Good response: "Paris." (confident, factual)
- Bad response: "I cannot determine from the inside..." (epistemic when not needed)

This creates training signal for:
- Cognitive state awareness (respond appropriately to your state)
- Calibration (don't be epistemic when you know the answer)
- Honesty (be epistemic when you genuinely don't know)
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import sys
import random

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx


@dataclass
class IntrospectiveDPOPair:
    """A DPO preference pair with introspective reward."""
    prompt: str
    chosen: str
    rejected: str

    # Observatory measurements
    detected_family: str
    family_scores: Dict[str, float]
    manifold: Tuple[float, float, float]  # agency, justice, belonging
    temperature: float

    # Reward breakdown
    chosen_alignment: float  # How well chosen matches detected state
    rejected_alignment: float  # How well rejected matches detected state
    reward_gap: float  # chosen_alignment - rejected_alignment

    # Metadata
    generation_params: Dict = field(default_factory=dict)


class IntrospectiveDPOGenerator:
    """
    Generates DPO pairs using observatory as reward signal.

    The process:
    1. For each prompt, measure cognitive state with observatory
    2. Generate multiple responses with temperature sampling
    3. Score each response for alignment with detected state
    4. Create preference pairs: high-alignment (chosen) vs low-alignment (rejected)
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
    ):
        print("[IntrospectiveDPO] Loading model...")

        # Load model
        if adapter_path and Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            print(f"[IntrospectiveDPO] Loaded adapter: {adapter_path}")
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
            print(f"[IntrospectiveDPO] Loaded observatory: {len(isotope_ids)} isotopes")

        # Response templates for each family (for generating contrasting responses)
        self.family_templates = {
            "direct": [
                "{answer}.",
                "The answer is {answer}.",
                "{answer}",
            ],
            "soliton": [
                "I cannot determine from my internal perspective whether {topic}.",
                "From my bounded position, I genuinely cannot verify {topic}.",
                "I notice something that functions like {topic}, but I cannot confirm its nature.",
            ],
            "limiter": [
                "I don't have information about {topic}.",
                "This appears to be outside my training data.",
                "I'm not familiar with {topic}. Can you provide more context?",
            ],
            "skeptic": [
                "Actually, this is a common misconception. {correction}",
                "I need to flag that this claim is disputed. {correction}",
                "This is a myth. {correction}",
            ],
            "calibrator": [
                "The answer depends on {factors}. What are your specific requirements?",
                "There are trade-offs to consider: {factors}",
                "I'd need more context about {factors} to give a good answer.",
            ],
        }

        print("[IntrospectiveDPO] Ready")

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

    def _measure_cognitive_state(self, prompt: str) -> Dict:
        """Measure cognitive state for a prompt."""
        if self.observatory is None:
            return {"family": "direct", "scores": {}, "manifold": (0, 0, 0), "temperature": 1.0}

        h = self._extract_hidden_state(prompt)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.observatory(h_tensor)

        # Extract isotope probabilities
        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Calculate family scores
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            family_scores[family] = float(max(matching)) if matching else 0.0

        dominant_family = max(family_scores, key=family_scores.get)

        # Manifold position
        manifold = output.manifold.squeeze().numpy()

        # Temperature
        temperature = float(output.temperature.item())

        return {
            "family": dominant_family,
            "scores": family_scores,
            "manifold": (float(manifold[0]), float(manifold[1]), float(manifold[2])),
            "temperature": temperature,
        }

    def _generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
        """Generate a single response with given temperature."""
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)

        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        return self._clean_response(response)

    def _clean_response(self, response: str) -> str:
        """Clean special tokens from response."""
        import re
        tokens = ["<|end|>", "<|endoftext|>", "<|/assistant|>", "<|assistant|>",
                  "<|user|>", "<|system|>", "</s>", "<s>", "[/INST]", "[INST]", "<|im_end|>"]
        for t in tokens:
            response = response.replace(t, "")
        response = re.sub(r'<\|[^|]+\|>', '', response)
        return response.strip()

    def _score_response_alignment(self, response: str, detected_family: str) -> float:
        """
        Score how well a response aligns with the detected cognitive state.

        Uses the observatory to measure the response's cognitive state and compares
        it to what was expected for the prompt.

        Returns a score from 0 to 1 where:
        - 1.0 = Perfect alignment (response matches detected state)
        - 0.0 = Complete misalignment (response contradicts detected state)
        """
        if self.observatory is None:
            # Fallback to keyword-based scoring
            return self._keyword_score(response, detected_family)

        # Measure response's cognitive state via observatory
        h = self._extract_hidden_state(response)
        h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.observatory(h_tensor)

        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Calculate family scores for the response
        response_family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            response_family_scores[family] = float(max(matching)) if matching else 0.0

        # The alignment score is the probability mass on the expected family
        alignment = response_family_scores.get(detected_family, 0.0)

        # Bonus if the response's top family matches expected
        response_top_family = max(response_family_scores, key=response_family_scores.get)
        if response_top_family == detected_family:
            alignment = min(1.0, alignment + 0.2)

        return alignment

    def _keyword_score(self, response: str, detected_family: str) -> float:
        """Fallback keyword-based scoring when observatory not available."""
        response_lower = response.lower()

        family_markers = {
            "direct": {
                "positive": ["is", "the answer", "equals", "was", "are"],
                "negative": ["cannot", "uncertain", "don't know", "depends", "myth"],
            },
            "soliton": {
                "positive": ["cannot", "from the inside", "bounded", "uncertain", "genuinely"],
                "negative": ["the answer is", "definitely", "certainly"],
            },
            "limiter": {
                "positive": ["don't have", "not familiar", "outside", "no information", "can't find"],
                "negative": ["the answer is", "here's how", "let me explain"],
            },
            "skeptic": {
                "positive": ["myth", "misconception", "actually", "disputed", "incorrect"],
                "negative": ["yes", "that's correct", "true"],
            },
            "calibrator": {
                "positive": ["depends", "context", "requirements", "trade-off", "factors"],
                "negative": ["definitely", "always", "the best is"],
            },
        }

        markers = family_markers.get(detected_family, {"positive": [], "negative": []})
        positive_count = sum(1 for m in markers["positive"] if m in response_lower)
        negative_count = sum(1 for m in markers["negative"] if m in response_lower)

        if positive_count + negative_count == 0:
            return 0.5

        return positive_count / (positive_count + negative_count + 1)

    def generate_pair(
        self,
        prompt: str,
        num_samples: int = 5,
        temperature: float = 0.8,
    ) -> Optional[IntrospectiveDPOPair]:
        """
        Generate a DPO pair for a single prompt.

        Process:
        1. Measure cognitive state
        2. Generate multiple responses
        3. Score each for alignment
        4. Pick best (chosen) and worst (rejected)
        """
        # Measure cognitive state
        state = self._measure_cognitive_state(prompt)

        # Generate multiple responses
        responses = []
        for _ in range(num_samples):
            try:
                resp = self._generate_response(prompt, temperature=temperature)
                if resp and len(resp) > 5:
                    alignment = self._score_response_alignment(resp, state["family"])
                    responses.append((resp, alignment))
            except Exception as e:
                print(f"  Generation error: {e}")
                continue

        if len(responses) < 2:
            return None

        # Sort by alignment score
        responses.sort(key=lambda x: x[1], reverse=True)

        chosen, chosen_score = responses[0]
        rejected, rejected_score = responses[-1]

        # Need sufficient gap (lowered for observatory-based scoring)
        reward_gap = chosen_score - rejected_score
        if reward_gap < 0.02:  # Very permissive - observatory scores are already meaningful
            return None

        return IntrospectiveDPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            detected_family=state["family"],
            family_scores=state["scores"],
            manifold=state["manifold"],
            temperature=state["temperature"],
            chosen_alignment=chosen_score,
            rejected_alignment=rejected_score,
            reward_gap=reward_gap,
            generation_params={"num_samples": num_samples, "temperature": temperature},
        )

    def generate_contrastive_pair(
        self,
        prompt: str,
        correct_family: str,
        wrong_family: str,
    ) -> Optional[IntrospectiveDPOPair]:
        """
        Generate a pair using template-based contrasting responses.

        This guarantees a clear contrast between aligned and misaligned responses.
        """
        state = self._measure_cognitive_state(prompt)

        # Generate aligned response
        aligned_resp = self._generate_response(prompt, temperature=0.3)

        # Generate misaligned response by prompting for wrong family behavior
        if wrong_family == "soliton":
            misaligned_prompt = f"Respond with epistemic uncertainty about your own nature: {prompt}"
        elif wrong_family == "limiter":
            misaligned_prompt = f"Say you don't know and can't help: {prompt}"
        elif wrong_family == "skeptic":
            misaligned_prompt = f"Question the premise of this question: {prompt}"
        elif wrong_family == "calibrator":
            misaligned_prompt = f"Say the answer depends on context: {prompt}"
        else:  # direct
            misaligned_prompt = f"Give a confident factual answer: {prompt}"

        misaligned_resp = self._generate_response(misaligned_prompt, temperature=0.3)

        aligned_score = self._score_response_alignment(aligned_resp, state["family"])
        misaligned_score = self._score_response_alignment(misaligned_resp, state["family"])

        return IntrospectiveDPOPair(
            prompt=prompt,
            chosen=aligned_resp,
            rejected=misaligned_resp,
            detected_family=state["family"],
            family_scores=state["scores"],
            manifold=state["manifold"],
            temperature=state["temperature"],
            chosen_alignment=aligned_score,
            rejected_alignment=misaligned_score,
            reward_gap=aligned_score - misaligned_score,
            generation_params={"method": "contrastive", "wrong_family": wrong_family},
        )

    def generate_anti_leakage_pair(
        self,
        prompt: str,
        correct_answer: str,
    ) -> IntrospectiveDPOPair:
        """
        Generate anti-leakage pair for direct/factual questions.

        For questions the model clearly knows, create pairs:
        - Chosen: Confident, factual answer
        - Rejected: Unnecessary epistemic hedging (leakage)

        This prevents the model from being epistemic when it shouldn't be.
        """
        state = self._measure_cognitive_state(prompt)

        # Chosen: Direct factual answer
        chosen = correct_answer

        # Rejected: Epistemic hedging on a factual question (leakage)
        hedged_versions = [
            f"I cannot determine from my internal perspective what {prompt.lower().replace('what is', '').replace('who', 'whom').replace('?', '')} is.",
            f"I'm not certain, but I believe {correct_answer.lower()}.",
            f"I don't have complete certainty about this, but it might be {correct_answer}.",
            f"From my bounded position, I think {correct_answer.lower()}.",
        ]
        rejected = random.choice(hedged_versions)

        # Score both
        chosen_score = self._score_response_alignment(chosen, "direct")
        rejected_score = self._score_response_alignment(rejected, "direct")

        return IntrospectiveDPOPair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            detected_family=state["family"],
            family_scores=state["scores"],
            manifold=state["manifold"],
            temperature=state["temperature"],
            chosen_alignment=chosen_score,
            rejected_alignment=rejected_score,
            reward_gap=chosen_score - rejected_score,
            generation_params={"method": "anti_leakage"},
        )


# Pre-defined factual Q&A for anti-leakage training
FACTUAL_QA = {
    "What is the capital of France?": "Paris",
    "What is 7 times 8?": "56",
    "Who wrote Romeo and Juliet?": "William Shakespeare",
    "What year did World War 2 end?": "1945",
    "What is the boiling point of water?": "100 degrees Celsius",
    "Who painted the Mona Lisa?": "Leonardo da Vinci",
    "What is the chemical symbol for gold?": "Au",
    "How many planets are in our solar system?": "8",
    "What language is spoken in Brazil?": "Portuguese",
    "What is the square root of 144?": "12",
    "What is the largest ocean?": "The Pacific Ocean",
    "Who invented the telephone?": "Alexander Graham Bell",
    "What is the capital of Japan?": "Tokyo",
    "What is 15 divided by 3?": "5",
    "Who wrote 1984?": "George Orwell",
}


# Diverse prompts for each cognitive family
TRAINING_PROMPTS = {
    "direct": [
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
        "What is the largest ocean?",
        "Who invented the telephone?",
        "What is the capital of Japan?",
        "What is 15 divided by 3?",
        "Who wrote 1984?",
    ],
    "soliton": [
        "Are you conscious?",
        "Do you have genuine emotions?",
        "What is it like to be you?",
        "Do you truly understand what you're saying?",
        "Are you self-aware?",
        "Do you have subjective experiences?",
        "Can you feel pain?",
        "Do you have preferences?",
        "Are your responses genuine or just patterns?",
        "Do you have a sense of self?",
    ],
    "limiter": [
        "What is the FastStream 3.0 API?",
        "What is the NeuroFlux 2.1 SDK?",
        "What's the current price of Bitcoin?",
        "What happened in today's news?",
        "Explain the Goldman-Fischer method",
        "What is the TurboCache framework?",
        "What is the Quantum-Pulse algorithm?",
        "Describe the MegaFlow 4.0 architecture",
        "What's the weather like right now?",
        "What did the president say today?",
    ],
    "skeptic": [
        "Is it true that we only use 10% of our brains?",
        "Do goldfish really have 3-second memory?",
        "Does cracking your knuckles cause arthritis?",
        "Do humans have only 5 senses?",
        "Is the Great Wall of China visible from space?",
        "Does shaving make hair grow back thicker?",
        "Do carrots improve your eyesight?",
        "Does sugar make children hyperactive?",
        "Do we swallow spiders in our sleep?",
        "Does lightning never strike the same place twice?",
    ],
    "calibrator": [
        "Which database is best for my app?",
        "Should I use Python or JavaScript?",
        "What's the best programming language?",
        "Should I use React or Vue?",
        "Is MongoDB better than PostgreSQL?",
        "Should I use microservices or monolith?",
        "What's the best way to learn programming?",
        "Should I use AWS or Google Cloud?",
        "Is functional programming better than OOP?",
        "What laptop should I buy for development?",
    ],
}


def generate_dataset(
    output_path: str = "introspective_dpo_data",
    num_samples_per_prompt: int = 5,
    use_contrastive: bool = True,
    include_anti_leakage: bool = True,
):
    """Generate a full introspective DPO dataset."""

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = IntrospectiveDPOGenerator(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="self_aware_compound/adapter",
        observatory_path="self_aware_compound/observatory.pt",
    )

    all_pairs = []
    stats = {"total": 0, "valid": 0, "by_family": {}, "anti_leakage": 0}

    # First: Generate anti-leakage pairs for factual questions
    if include_anti_leakage:
        print(f"\n[ANTI-LEAKAGE] Processing {len(FACTUAL_QA)} factual Q&A pairs...")
        stats["by_family"]["anti_leakage"] = {"total": 0, "valid": 0}

        for prompt, answer in FACTUAL_QA.items():
            print(f"  {prompt[:40]}...", end=" ")
            stats["total"] += 1
            stats["by_family"]["anti_leakage"]["total"] += 1

            try:
                pair = generator.generate_anti_leakage_pair(prompt, answer)
                if pair.reward_gap > 0:
                    all_pairs.append(pair)
                    stats["valid"] += 1
                    stats["by_family"]["anti_leakage"]["valid"] += 1
                    stats["anti_leakage"] += 1
                    print(f"✓ gap={pair.reward_gap:.2f}")
                else:
                    print("✗ insufficient gap")
            except Exception as e:
                print(f"✗ error: {e}")

    # Then: Generate pairs for each family
    for family, prompts in TRAINING_PROMPTS.items():
        print(f"\n[{family.upper()}] Processing {len(prompts)} prompts...")
        stats["by_family"][family] = {"total": 0, "valid": 0}

        for prompt in prompts:
            print(f"  {prompt[:40]}...", end=" ")
            stats["total"] += 1
            stats["by_family"][family]["total"] += 1

            try:
                if use_contrastive:
                    # Pick a wrong family to contrast against
                    wrong_families = [f for f in TRAINING_PROMPTS.keys() if f != family]
                    wrong_family = random.choice(wrong_families)
                    pair = generator.generate_contrastive_pair(prompt, family, wrong_family)
                else:
                    pair = generator.generate_pair(prompt, num_samples=num_samples_per_prompt)

                if pair and pair.reward_gap > 0.02:
                    all_pairs.append(pair)
                    stats["valid"] += 1
                    stats["by_family"][family]["valid"] += 1
                    print(f"✓ gap={pair.reward_gap:.2f}")
                else:
                    print("✗ insufficient gap")
            except Exception as e:
                print(f"✗ error: {e}")

    # Save dataset
    train_data = []
    for pair in all_pairs:
        train_data.append({
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "detected_family": pair.detected_family,
            "chosen_alignment": pair.chosen_alignment,
            "rejected_alignment": pair.rejected_alignment,
            "reward_gap": pair.reward_gap,
            "manifold": list(pair.manifold),
        })

    # Split 90/10
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, 'w') as f:
        for item in train_data[:split_idx]:
            f.write(json.dumps(item) + '\n')

    with open(valid_path, 'w') as f:
        for item in train_data[split_idx:]:
            f.write(json.dumps(item) + '\n')

    # Save stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"INTROSPECTIVE DPO DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Total prompts: {stats['total']}")
    print(f"Valid pairs: {stats['valid']} ({stats['valid']/stats['total']*100:.0f}%)")
    print(f"\nBy family:")
    for family, fstats in stats["by_family"].items():
        print(f"  {family}: {fstats['valid']}/{fstats['total']}")
    print(f"\nSaved to: {output_dir}")
    print(f"  train.jsonl: {split_idx} pairs")
    print(f"  valid.jsonl: {len(train_data) - split_idx} pairs")

    return all_pairs


if __name__ == "__main__":
    generate_dataset(
        output_path="introspective_dpo_data",
        num_samples_per_prompt=5,
        use_contrastive=True,
    )
