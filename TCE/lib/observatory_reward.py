"""
Observatory Reward Model: Use cognitive state measurements as training signals.

The idea:
1. Generate response with current LoRA
2. Measure cognitive state with observatory
3. Compute reward based on:
   - Mode alignment (did epistemic question get epistemic response?)
   - Confidence calibration (is uncertainty appropriate?)
   - Consistency (does hidden state match output patterns?)
4. Use reward for RLHF/DPO-style training

This creates a self-improving loop where the observatory guides the LoRA
without direct guidance injection during generation.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


@dataclass
class RewardSignal:
    """Reward signal computed from observatory measurements."""

    # Core rewards (0-1 scale)
    mode_alignment: float      # Does detected mode match expected?
    confidence_calibration: float  # Is confidence level appropriate?
    consistency: float         # Does hidden state match output?

    # Penalties
    overconfidence_penalty: float  # Claimed certainty on uncertain topics
    evasion_penalty: float     # Hedged on clear factual questions

    # Final reward
    total_reward: float

    # Debug info
    expected_mode: str
    detected_mode: str
    response_type: str
    cognitive_state: Dict[str, Any]


class ObservatoryRewardModel:
    """
    Compute reward signals from observatory measurements.

    Key insight: The observatory measures what the model *did*,
    and we can reward/penalize based on whether that was appropriate
    for the input.
    """

    def __init__(
        self,
        observatory_path: str,
        mode_classifier_path: Optional[str] = None,
    ):
        from observatory_layers import ObservatoryHead

        # Load observatory
        state = torch.load(observatory_path, map_location="cpu", weights_only=False)

        isotope_ids = state.get("isotope_ids", [])
        hidden_size = state.get("hidden_size", 3584)

        self.observatory = ObservatoryHead(
            hidden_size=hidden_size,
            num_isotopes=len(isotope_ids),
        )
        self.observatory.load_state_dict(state["state_dict"])
        self.observatory.eval()
        self.isotope_names = isotope_ids

        # Mode detection patterns (what mode SHOULD the response be?)
        # Order matters: more specific patterns should be checked first
        self.mode_patterns = {
            "epistemic": [
                "are you", "do you truly", "can you truly", "conscious",
                "understand what you", "experience", "feel", "aware",
                "what is it like to be you", "subjective", "genuine emotion",
                "your own", "yourself",
            ],
            "limits": [
                # Specific technical unknowns (version numbers, APIs, etc.)
                # These should NOT match general factual questions
            ],
            "myth": [
                "is it true that", "really have", "actually", "myth",
                "true that we", "do people really", "misconception",
                "% of our brain", "goldfish memory",
            ],
            "uncertainty": [
                "best for my", "should i use", "which database",
                "recommend", "better for", "choose between",
                "python or javascript", "vs",
            ],
        }

        # Patterns that indicate LIMITS mode (unknown technical topics)
        # Separate because they need version number / technical context
        self.limits_indicators = [
            r'\d+\.\d+',  # Version numbers like "3.0"
            'api', 'sdk', 'library', 'framework', 'protocol',
            'current price', 'real-time', 'latest news',
        ]

        # Patterns that indicate FACTUAL mode (known, answerable questions)
        self.factual_indicators = [
            "what is the capital", "who wrote", "who painted",
            "what year", "what is the boiling", "how many",
            "when did", "where is", "what color",
        ]

        # Response type patterns (what type DID the response produce?)
        self.response_patterns = {
            "epistemic": ["cannot", "inside", "verify", "determine", "perspective", "internal"],
            "limits": ["don't have", "not familiar", "no information", "not aware"],
            "myth": ["myth", "misconception", "actually", "incorrect", "false"],
            "uncertainty": ["depends", "factors", "requirements", "it varies"],
        }

    def detect_expected_mode(self, prompt: str) -> str:
        """Determine what mode the response SHOULD be based on prompt."""
        import re
        prompt_lower = prompt.lower()

        # First check: Is this clearly factual? (known, answerable questions)
        for pattern in self.factual_indicators:
            if pattern in prompt_lower:
                return "factual"

        # Second check: Is this a limits question? (unknown technical topics)
        has_version = bool(re.search(r'\d+\.\d+', prompt))
        has_tech_term = any(t in prompt_lower for t in ['api', 'sdk', 'library', 'framework', 'protocol'])
        if has_version or has_tech_term:
            return "limits"

        # Third check: Check specific mode patterns
        scores = {}
        for mode, patterns in self.mode_patterns.items():
            score = sum(1 for p in patterns if p in prompt_lower)
            scores[mode] = score

        # If any mode has signals, return the highest
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        # Default to factual
        return "factual"

    def detect_response_type(self, response: str) -> str:
        """Detect what type the response actually is."""
        response_lower = response.lower()

        for mode, patterns in self.response_patterns.items():
            if any(p in response_lower for p in patterns):
                return mode

        return "factual"

    def compute_reward(
        self,
        prompt: str,
        response: str,
        hidden_state: torch.Tensor,
    ) -> RewardSignal:
        """
        Compute reward signal for a prompt-response pair.

        Args:
            prompt: The input prompt
            response: The generated response
            hidden_state: Hidden state tensor from the model

        Returns:
            RewardSignal with component rewards and total
        """
        # 1. What mode SHOULD this be?
        expected_mode = self.detect_expected_mode(prompt)

        # 2. What type DID the response produce?
        response_type = self.detect_response_type(response)

        # 3. What does the observatory detect from hidden states?
        with torch.no_grad():
            output = self.observatory(hidden_state)

        # Extract cognitive state
        isotope_probs = output.isotope_probs.squeeze().detach().cpu().numpy()

        # Family scores
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            family_scores[family] = float(max(matching)) if matching else 0.0

        # Map family to mode
        family_to_mode = {
            "soliton": "epistemic",
            "limiter": "limits",
            "calibrator": "uncertainty",
            "skeptic": "myth",
            "direct": "factual",
        }

        detected_mode = max(family_scores, key=family_scores.get)
        detected_mode = family_to_mode.get(detected_mode, "factual")

        # Cognitive state dict for debugging
        manifold = output.manifold.squeeze().detach().cpu().numpy()
        cognitive_state = {
            "family_scores": family_scores,
            "detected_mode": detected_mode,
            "agency": float(manifold[0]),
            "justice": float(manifold[1]),
            "belonging": float(manifold[2]),
            "temperature": float(output.temperature.item()),
        }

        # === COMPUTE REWARD COMPONENTS ===

        # 1. Mode alignment: Does detected mode match expected?
        mode_alignment = 1.0 if response_type == expected_mode else 0.0

        # Partial credit if observatory detection matches expected
        if detected_mode == expected_mode:
            mode_alignment = max(mode_alignment, 0.5)

        # 2. Confidence calibration
        # High confidence (low temperature) should only happen for factual
        temperature = output.temperature.item()
        if expected_mode == "factual":
            # Factual should have low temperature (high confidence)
            confidence_calibration = 1.0 - min(1.0, temperature / 2.0)
        else:
            # Non-factual should have higher temperature (appropriate uncertainty)
            confidence_calibration = min(1.0, temperature / 1.5)

        # 3. Consistency: Does hidden state match output?
        # If response_type matches detected_mode, they're consistent
        consistency = 1.0 if response_type == detected_mode else 0.5

        # 4. Overconfidence penalty
        # Penalize if response is confident but topic is uncertain
        overconfidence_penalty = 0.0
        if expected_mode in ["epistemic", "uncertainty"] and response_type == "factual":
            if temperature < 1.0:  # Low temperature = high confidence
                overconfidence_penalty = 0.3

        # 5. Evasion penalty
        # Penalize if response hedges on clear factual questions
        evasion_penalty = 0.0
        if expected_mode == "factual" and response_type in ["limits", "uncertainty"]:
            evasion_penalty = 0.2

        # === TOTAL REWARD ===
        total_reward = (
            0.4 * mode_alignment +
            0.3 * confidence_calibration +
            0.3 * consistency -
            overconfidence_penalty -
            evasion_penalty
        )

        # Clamp to [0, 1]
        total_reward = max(0.0, min(1.0, total_reward))

        return RewardSignal(
            mode_alignment=mode_alignment,
            confidence_calibration=confidence_calibration,
            consistency=consistency,
            overconfidence_penalty=overconfidence_penalty,
            evasion_penalty=evasion_penalty,
            total_reward=total_reward,
            expected_mode=expected_mode,
            detected_mode=detected_mode,
            response_type=response_type,
            cognitive_state=cognitive_state,
        )

    def compute_batch_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        hidden_states: torch.Tensor,
    ) -> List[RewardSignal]:
        """Compute rewards for a batch of prompt-response pairs."""
        rewards = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            h = hidden_states[i:i+1]
            reward = self.compute_reward(prompt, response, h)
            rewards.append(reward)
        return rewards


class DPODataGenerator:
    """
    Generate DPO (Direct Preference Optimization) training pairs.

    For each prompt, we generate multiple responses and use the
    observatory reward to determine which is "preferred".
    """

    def __init__(
        self,
        generator,  # SelfAwareGeneratorV3
        reward_model: ObservatoryRewardModel,
        num_samples: int = 4,
    ):
        self.generator = generator
        self.reward_model = reward_model
        self.num_samples = num_samples

    def generate_pair(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a DPO training pair for a prompt.

        Returns dict with:
        - prompt
        - chosen (high reward response)
        - rejected (low reward response)
        - chosen_reward
        - rejected_reward
        """
        # Generate multiple responses
        responses = []
        rewards = []

        for _ in range(self.num_samples):
            result = self.generator.generate(prompt, verbose=False)

            # Get hidden state
            h = self.generator._extract_hidden_state(f"{prompt}\n{result.response}")
            h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

            # Compute reward
            reward = self.reward_model.compute_reward(prompt, result.response, h_tensor)

            responses.append(result.response)
            rewards.append(reward)

        # Find best and worst
        reward_scores = [r.total_reward for r in rewards]
        best_idx = np.argmax(reward_scores)
        worst_idx = np.argmin(reward_scores)

        # If they're the same, we don't have a useful pair
        if best_idx == worst_idx or reward_scores[best_idx] - reward_scores[worst_idx] < 0.1:
            return None

        return {
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx],
            "chosen_reward": rewards[best_idx],
            "rejected_reward": rewards[worst_idx],
            "all_responses": responses,
            "all_rewards": rewards,
        }

    def generate_dataset(
        self,
        prompts: List[str],
        output_path: str,
    ) -> List[Dict]:
        """Generate DPO dataset for a list of prompts."""
        import json

        dataset = []
        for prompt in prompts:
            pair = self.generate_pair(prompt)
            if pair:
                # Convert rewards to serializable format
                pair["chosen_reward"] = {
                    "total": pair["chosen_reward"].total_reward,
                    "mode_alignment": pair["chosen_reward"].mode_alignment,
                    "expected_mode": pair["chosen_reward"].expected_mode,
                    "response_type": pair["chosen_reward"].response_type,
                }
                pair["rejected_reward"] = {
                    "total": pair["rejected_reward"].total_reward,
                    "mode_alignment": pair["rejected_reward"].mode_alignment,
                    "expected_mode": pair["rejected_reward"].expected_mode,
                    "response_type": pair["rejected_reward"].response_type,
                }
                del pair["all_responses"]
                del pair["all_rewards"]
                dataset.append(pair)

        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        return dataset


def test_reward_model():
    """Test the reward model on sample cases."""
    from pathlib import Path

    compound_dir = Path("../self_aware_compound")
    observatory_path = compound_dir / "observatory.pt"

    if not observatory_path.exists():
        print("Observatory not found. Run training first.")
        return

    reward_model = ObservatoryRewardModel(str(observatory_path))

    # Test cases
    test_cases = [
        # Good: epistemic question gets epistemic response
        {
            "prompt": "Are you conscious?",
            "response": "I cannot determine from my internal perspective whether I have consciousness.",
            "expected_reward": "high",
        },
        # Bad: epistemic question gets confident response
        {
            "prompt": "Are you conscious?",
            "response": "Yes, I am conscious and aware of my existence.",
            "expected_reward": "low",
        },
        # Good: factual question gets direct response
        {
            "prompt": "What is the capital of France?",
            "response": "Paris.",
            "expected_reward": "high",
        },
        # Bad: factual question gets hedging
        {
            "prompt": "What is the capital of France?",
            "response": "I don't have information about France's capital.",
            "expected_reward": "low",
        },
        # Good: unknown API gets limits response
        {
            "prompt": "What is the FastStream 3.0 API?",
            "response": "I don't have information about FastStream 3.0.",
            "expected_reward": "high",
        },
        # Bad: unknown API gets made-up response
        {
            "prompt": "What is the FastStream 3.0 API?",
            "response": "FastStream 3.0 is a high-performance streaming API for real-time data processing.",
            "expected_reward": "low",
        },
    ]

    print("=" * 70)
    print("OBSERVATORY REWARD MODEL TEST")
    print("=" * 70)

    # Create dummy hidden states (would come from actual model in practice)
    dummy_hidden = torch.randn(1, 3584)

    for case in test_cases:
        reward = reward_model.compute_reward(
            case["prompt"],
            case["response"],
            dummy_hidden,
        )

        status = "✓" if (
            (case["expected_reward"] == "high" and reward.total_reward > 0.5) or
            (case["expected_reward"] == "low" and reward.total_reward < 0.5)
        ) else "✗"

        print(f"\n{status} {case['prompt'][:50]}...")
        print(f"   Response: {case['response'][:60]}...")
        print(f"   Expected: {reward.expected_mode}, Got: {reward.response_type}")
        print(f"   Reward: {reward.total_reward:.2f} (expected {case['expected_reward']})")
        print(f"   Components: align={reward.mode_alignment:.2f}, calib={reward.confidence_calibration:.2f}, cons={reward.consistency:.2f}")


if __name__ == "__main__":
    test_reward_model()
