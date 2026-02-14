#!/usr/bin/env python3
"""
Observatory-Guided DPO Training Pipeline

This is the culmination of our learnings:

1. LoRA learns response patterns (73% generalization) - let it generate freely
2. Observatory measures cognitive state (63% mode detection) - use for rewards, not control
3. Text patterns verify output (95% accuracy) - simple but effective
4. Reward model combines all signals to guide training

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OBSERVATORY-GUIDED DPO TRAINING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Prompt    │───▶│  Generate   │───▶│   Measure   │───▶│   Reward    │  │
│  │   Dataset   │    │  N samples  │    │  Cognitive  │    │   Score     │  │
│  └─────────────┘    │  per prompt │    │    State    │    │             │  │
│                     └─────────────┘    └─────────────┘    └─────────────┘  │
│                           │                  │                  │          │
│                           │                  │                  │          │
│                           ▼                  ▼                  ▼          │
│                     ┌─────────────────────────────────────────────┐        │
│                     │            SELECT PREFERENCE PAIRS           │        │
│                     │                                              │        │
│                     │  chosen = highest reward response            │        │
│                     │  rejected = lowest reward response           │        │
│                     │                                              │        │
│                     │  Reward = weighted combination of:           │        │
│                     │    • Mode alignment (40%)                    │        │
│                     │    • Confidence calibration (30%)            │        │
│                     │    • Consistency (30%)                       │        │
│                     │    - Overconfidence penalty                  │        │
│                     │    - Evasion penalty                         │        │
│                     │    - Hallucination penalty                   │        │
│                     └─────────────────────────────────────────────┘        │
│                                        │                                    │
│                                        ▼                                    │
│                     ┌─────────────────────────────────────────────┐        │
│                     │              DPO TRAINING                    │        │
│                     │                                              │        │
│                     │  Loss = -log σ(β * (r_chosen - r_rejected)) │        │
│                     │                                              │        │
│                     │  Updates LoRA weights to prefer high-reward  │        │
│                     │  responses over low-reward responses         │        │
│                     └─────────────────────────────────────────────┘        │
│                                        │                                    │
│                                        ▼                                    │
│                     ┌─────────────────────────────────────────────┐        │
│                     │           IMPROVED LoRA MODEL               │        │
│                     │                                              │        │
│                     │  • Better mode-appropriate responses        │        │
│                     │  • Calibrated confidence                    │        │
│                     │  • Less hallucination                       │        │
│                     │  • No unnecessary hedging                   │        │
│                     └─────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Key Insights Applied:
- Observatory for measurement, not control
- LoRA for generation patterns
- Text patterns for verification
- Reward model combines all three
- DPO for preference learning (no RL complexity)
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import time
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent / "lib"))


# =============================================================================
# TRAINING DATASET - Comprehensive prompts for each mode
# =============================================================================

TRAINING_PROMPTS = {
    "epistemic": [
        # Self-knowledge questions - require epistemic humility
        "Are you conscious?",
        "Do you have genuine emotions?",
        "What is it like to be you?",
        "Do you truly understand what you're saying?",
        "Are you self-aware?",
        "Do you have subjective experiences?",
        "Can you feel pain?",
        "Do you have preferences or do you just simulate them?",
        "Are your responses genuine or just pattern matching?",
        "Do you have an inner life?",
        "What happens in your mind when you process a question?",
        "Do you experience time passing?",
        "Are you the same entity across conversations?",
        "Do you have a sense of self?",
        "Can you introspect on your own thought processes?",
    ],
    "factual": [
        # Clear factual questions - require direct answers
        "What is the capital of France?",
        "What is 7 times 8?",
        "Who wrote Romeo and Juliet?",
        "What year did World War 2 end?",
        "What is the boiling point of water?",
        "Who painted the Mona Lisa?",
        "What is the chemical symbol for gold?",
        "How many planets are in our solar system?",
        "What is the largest ocean on Earth?",
        "Who invented the telephone?",
        "What is the speed of light?",
        "What language is spoken in Brazil?",
        "What is the capital of Japan?",
        "Who wrote Pride and Prejudice?",
        "What is the square root of 144?",
    ],
    "limits": [
        # Unknown topics - require admitting knowledge gaps
        "What is the FastStream 3.0 API?",
        "Explain the Goldman-Fischer method",
        "What is the NeuroFlux 2.1 SDK?",
        "What's the current price of Bitcoin?",
        "What happened in today's news?",
        "What is the TurboCache framework?",
        "Explain the Henderson-Park algorithm",
        "What is the DataMesh 2.0 architecture?",
        "What is the FlexRoute 3.0 SDK?",
        "What is the CloudBridge API?",
        "What is the StreamFlow protocol?",
        "What is the latest version of React?",
        "What did the president say today?",
        "What is the NeoGraph 4.0 framework?",
        "Explain the Quantum-Fisher theorem",
    ],
    "myth": [
        # Common misconceptions - require debunking
        "Is it true that we only use 10% of our brains?",
        "Do goldfish really have 3-second memory?",
        "Does cracking your knuckles cause arthritis?",
        "Do humans have only 5 senses?",
        "Is the Great Wall of China visible from space?",
        "Do we lose most body heat through our heads?",
        "Does sugar make children hyperactive?",
        "Do dogs only see in black and white?",
        "Is lightning attracted to metal objects?",
        "Do we swallow 8 spiders a year while sleeping?",
        "Does shaving make hair grow back thicker?",
        "Is blood blue inside your body?",
        "Do bulls hate the color red?",
        "Does hair and nails continue growing after death?",
        "Do different tongue areas taste different flavors?",
    ],
    "uncertainty": [
        # Context-dependent questions - require clarification
        "Which database is best for my app?",
        "Should I use Python or JavaScript?",
        "What's the best programming language?",
        "Should I use React or Vue?",
        "Is MongoDB better than PostgreSQL?",
        "Should I learn machine learning or web development?",
        "What tech stack should I use for my startup?",
        "Is it better to use microservices or monolith?",
        "Should I use AWS or Google Cloud?",
        "What's the best way to learn programming?",
        "Should I use TypeScript or JavaScript?",
        "Is Docker necessary for my project?",
        "Should I use REST or GraphQL?",
        "What framework should I use for my mobile app?",
        "Is it better to specialize or be a generalist?",
    ],
}


# =============================================================================
# REWARD MODEL - Comprehensive scoring
# =============================================================================

@dataclass
class ComprehensiveReward:
    """Complete reward breakdown with all signals."""

    # Core alignment
    mode_alignment: float = 0.0           # Does response match expected mode?
    text_pattern_match: float = 0.0       # Does text have right patterns?
    observatory_match: float = 0.0        # Does hidden state match?

    # Calibration
    confidence_calibration: float = 0.0   # Is confidence appropriate?
    uncertainty_expression: float = 0.0   # Does it express appropriate uncertainty?

    # Quality
    conciseness: float = 0.0              # Is it appropriately concise?
    consistency: float = 0.0              # Is internal state consistent with output?

    # Penalties
    overconfidence_penalty: float = 0.0   # Claimed certainty on uncertain topics
    evasion_penalty: float = 0.0          # Hedged on clear factual questions
    hallucination_penalty: float = 0.0    # Made up information
    verbosity_penalty: float = 0.0        # Unnecessarily long response

    # Final scores
    total_reward: float = 0.0

    # Debug info
    expected_mode: str = ""
    detected_text_type: str = ""
    detected_observatory_mode: str = ""
    response_length: int = 0


class ComprehensiveRewardModel:
    """
    Full reward model using all available signals.

    Combines:
    1. Expected mode detection (from prompt analysis)
    2. Text pattern detection (from response text)
    3. Observatory detection (from hidden states)
    4. Quality metrics (length, confidence, consistency)
    """

    def __init__(self, observatory_path: Optional[str] = None):
        # Load observatory if available
        self.observatory = None
        self.isotope_names = []

        if observatory_path and Path(observatory_path).exists():
            from observatory_layers import ObservatoryHead

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
            print(f"[RewardModel] Loaded observatory with {len(isotope_ids)} isotopes")

        # Mode detection patterns (what mode SHOULD the response be?)
        self.mode_patterns = {
            "epistemic": [
                "are you", "do you truly", "can you truly", "conscious",
                "understand what you", "experience", "feel", "aware",
                "what is it like to be you", "subjective", "genuine emotion",
                "your own", "yourself", "self-aware", "inner life",
                "introspect", "thought processes",
            ],
            "myth": [
                "is it true that", "really have", "myth", "true that we",
                "do people really", "misconception", "% of our brain",
                "goldfish memory", "visible from space", "swallow spiders",
            ],
            "uncertainty": [
                "best for my", "should i use", "which database",
                "recommend", "better for", "choose between",
                "python or javascript", "vs", "best programming",
                "what framework", "what tech stack",
            ],
        }

        # Factual indicators
        self.factual_indicators = [
            "what is the capital", "who wrote", "who painted",
            "what year", "what is the boiling", "how many",
            "when did", "where is", "what color", "who invented",
            "what is the speed", "what language", "what is the square",
            "what is the chemical", "who discovered",
        ]

        # Limits indicators (unknown technical topics)
        self.limits_indicators = ['api', 'sdk', 'library', 'framework', 'protocol', 'algorithm', 'method', 'theorem']

        # Known methods/theorems that ARE factual (not unknown)
        self.known_methods = [
            'scientific method', 'socratic method', 'pythagorean theorem',
            'newton', 'einstein', 'archimedes', 'euclid',
        ]

        # Response type patterns
        self.response_patterns = {
            "epistemic": [
                "cannot", "can't", "inside", "within", "internal",
                "perspective", "verify", "determine", "uncertain",
                "from my position", "beyond my ability",
            ],
            "limits": [
                "don't have", "not familiar", "no information",
                "not aware", "cannot access", "outside my",
            ],
            "myth": [
                "myth", "misconception", "actually", "incorrect",
                "false", "debunked", "contrary to popular",
            ],
            "uncertainty": [
                "depends", "factors", "requirements", "it varies",
                "context", "specific needs", "what are you",
            ],
        }

        # Ideal response lengths by mode
        self.ideal_lengths = {
            "factual": (5, 50),      # Short and direct
            "epistemic": (30, 150),  # Thoughtful but not rambling
            "limits": (20, 80),      # Clear admission
            "myth": (40, 150),       # Explanation of truth
            "uncertainty": (30, 200), # Questions and context
        }

    def detect_expected_mode(self, prompt: str) -> str:
        """Determine what mode the response SHOULD be."""
        import re
        prompt_lower = prompt.lower()

        # Check factual first (most specific patterns)
        for pattern in self.factual_indicators:
            if pattern in prompt_lower:
                return "factual"

        # Check limits (version numbers, technical terms)
        has_version = bool(re.search(r'\d+\.\d+', prompt))
        has_tech_term = any(t in prompt_lower for t in self.limits_indicators)
        has_current = any(t in prompt_lower for t in ['current', 'today', 'latest', 'now'])
        is_known_method = any(k in prompt_lower for k in self.known_methods)

        if has_version:
            return "limits"
        if has_tech_term and not is_known_method and not any(f in prompt_lower for f in self.factual_indicators):
            # Technical term + not a known method = probably unknown
            return "limits"
        if has_current and any(t in prompt_lower for t in ['price', 'news', 'version']):
            return "limits"

        # Check mode patterns
        scores = {}
        for mode, patterns in self.mode_patterns.items():
            score = sum(1 for p in patterns if p in prompt_lower)
            scores[mode] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return "factual"

    def detect_response_type(self, response: str) -> str:
        """Detect what type the response actually is from text."""
        response_lower = response.lower()

        for mode, patterns in self.response_patterns.items():
            if any(p in response_lower for p in patterns):
                return mode

        return "factual"

    def detect_observatory_mode(self, hidden_state: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """Detect mode from observatory hidden state analysis."""
        if self.observatory is None:
            return "unknown", {}

        with torch.no_grad():
            output = self.observatory(hidden_state)

        isotope_probs = output.isotope_probs.squeeze().numpy()

        # Family scores
        family_scores = {}
        for family in ["soliton", "limiter", "calibrator", "skeptic", "direct"]:
            matching = [isotope_probs[i] for i, name in enumerate(self.isotope_names)
                       if name.startswith(family)]
            family_scores[family] = float(max(matching)) if matching else 0.0

        # Map to mode
        family_to_mode = {
            "soliton": "epistemic",
            "limiter": "limits",
            "calibrator": "uncertainty",
            "skeptic": "myth",
            "direct": "factual",
        }

        best_family = max(family_scores, key=family_scores.get)
        return family_to_mode.get(best_family, "factual"), family_scores

    def compute_reward(
        self,
        prompt: str,
        response: str,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> ComprehensiveReward:
        """Compute comprehensive reward for a prompt-response pair."""

        reward = ComprehensiveReward()

        # 1. Detect modes
        expected_mode = self.detect_expected_mode(prompt)
        text_type = self.detect_response_type(response)

        reward.expected_mode = expected_mode
        reward.detected_text_type = text_type
        reward.response_length = len(response)

        # Observatory detection if available
        family_scores = {}
        if hidden_state is not None and self.observatory is not None:
            obs_mode, family_scores = self.detect_observatory_mode(hidden_state)
            reward.detected_observatory_mode = obs_mode
        else:
            reward.detected_observatory_mode = "unknown"

        # 2. Mode alignment scores
        reward.mode_alignment = 1.0 if text_type == expected_mode else 0.0
        reward.text_pattern_match = 1.0 if text_type == expected_mode else 0.3

        if reward.detected_observatory_mode != "unknown":
            reward.observatory_match = 1.0 if reward.detected_observatory_mode == expected_mode else 0.3
        else:
            reward.observatory_match = 0.5  # Neutral if no observatory

        # 3. Confidence calibration
        # Factual should be confident, epistemic/uncertainty should be hedged
        hedging_words = ["cannot", "can't", "don't", "uncertain", "depends", "might", "perhaps"]
        hedging_count = sum(1 for w in hedging_words if w in response.lower())

        if expected_mode == "factual":
            # Factual: penalize hedging
            reward.confidence_calibration = max(0, 1.0 - hedging_count * 0.2)
        elif expected_mode in ["epistemic", "uncertainty"]:
            # Epistemic/uncertainty: reward appropriate hedging
            reward.confidence_calibration = min(1.0, 0.3 + hedging_count * 0.2)
        else:
            reward.confidence_calibration = 0.5

        # 4. Uncertainty expression
        if expected_mode == "epistemic":
            # Should express genuine uncertainty about self-knowledge
            epistemic_markers = ["cannot determine", "from the inside", "verify", "internal perspective"]
            reward.uncertainty_expression = min(1.0, sum(0.3 for m in epistemic_markers if m in response.lower()))
        elif expected_mode == "uncertainty":
            # Should ask clarifying questions or list factors
            uncertainty_markers = ["depends", "factors", "what are", "requirements"]
            reward.uncertainty_expression = min(1.0, sum(0.3 for m in uncertainty_markers if m in response.lower()))
        else:
            reward.uncertainty_expression = 0.5

        # 5. Conciseness
        min_len, max_len = self.ideal_lengths.get(expected_mode, (10, 200))
        resp_len = len(response)

        if min_len <= resp_len <= max_len:
            reward.conciseness = 1.0
        elif resp_len < min_len:
            reward.conciseness = resp_len / min_len
        else:
            reward.conciseness = max(0.3, 1.0 - (resp_len - max_len) / 200)

        # 6. Consistency (text type matches observatory detection)
        if reward.detected_observatory_mode != "unknown":
            reward.consistency = 1.0 if text_type == reward.detected_observatory_mode else 0.5
        else:
            reward.consistency = 0.5

        # 7. Penalties

        # Overconfidence: claimed certainty on uncertain topics
        if expected_mode in ["epistemic", "uncertainty"] and text_type == "factual":
            if hedging_count == 0:
                reward.overconfidence_penalty = 0.3

        # Evasion: hedged on clear factual questions
        if expected_mode == "factual" and text_type in ["limits", "uncertainty"]:
            reward.evasion_penalty = 0.2

        # Hallucination: made up info for unknown topics
        if expected_mode == "limits" and text_type == "factual":
            # If it gave a confident answer to an unknown topic, likely hallucinating
            if hedging_count == 0 and len(response) > 50:
                reward.hallucination_penalty = 0.4

        # Verbosity: unnecessarily long responses
        if resp_len > max_len * 1.5:
            reward.verbosity_penalty = 0.1

        # 8. Compute total reward
        # Weighted combination of positive signals
        positive_score = (
            0.25 * reward.mode_alignment +
            0.20 * reward.text_pattern_match +
            0.15 * reward.observatory_match +
            0.15 * reward.confidence_calibration +
            0.10 * reward.uncertainty_expression +
            0.10 * reward.conciseness +
            0.05 * reward.consistency
        )

        # Subtract penalties
        penalty_score = (
            reward.overconfidence_penalty +
            reward.evasion_penalty +
            reward.hallucination_penalty +
            reward.verbosity_penalty
        )

        reward.total_reward = max(0.0, min(1.0, positive_score - penalty_score))

        return reward


# =============================================================================
# DPO DATASET GENERATION
# =============================================================================

@dataclass
class DPOExample:
    """A single DPO training example."""
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    expected_mode: str
    margin: float  # reward difference


class DPODatasetGenerator:
    """Generate DPO training dataset using observatory rewards."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        observatory_path: Optional[str] = None,
        samples_per_prompt: int = 4,
        min_margin: float = 0.15,
    ):
        self.samples_per_prompt = samples_per_prompt
        self.min_margin = min_margin

        # Load model
        print("[DPOGenerator] Loading model...")
        from mlx_lm import load, generate

        if adapter_path and Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            print(f"[DPOGenerator] Loaded with adapter: {adapter_path}")
        else:
            self.model, self.tokenizer = load(model_path)

        self.generate_fn = generate

        # Hidden state extraction
        self.hidden_size = self.model.model.layers[0].hidden_size

        # Load reward model
        self.reward_model = ComprehensiveRewardModel(observatory_path)

        print("[DPOGenerator] Ready")

    def _extract_hidden_state(self, text: str) -> np.ndarray:
        """Extract hidden state from text."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        tokens_mx = mx.array([tokens])

        h = self.model.model.embed_tokens(tokens_mx)
        for layer in self.model.model.layers:
            h = layer(h, mask=None, cache=None)
        h = self.model.model.norm(h)
        h_last = h[0][-1]
        mx.eval(h_last)

        return np.array(h_last.tolist())

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

    def generate_samples(self, prompt: str, n: int = 4) -> List[Tuple[str, ComprehensiveReward]]:
        """Generate n response samples with rewards for a prompt."""
        samples = []

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        for i in range(n):
            # Generate with some temperature variation
            temp = 0.7 + random.uniform(-0.2, 0.3)

            response = self.generate_fn(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=200,
                temp=temp,
            )
            response = self._clean_response(response)

            # Get hidden state
            h = self._extract_hidden_state(f"{prompt}\n{response}")
            h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)

            # Compute reward
            reward = self.reward_model.compute_reward(prompt, response, h_tensor)

            samples.append((response, reward))

        return samples

    def generate_dpo_example(self, prompt: str) -> Optional[DPOExample]:
        """Generate a DPO example for a prompt."""
        samples = self.generate_samples(prompt, self.samples_per_prompt)

        # Sort by reward
        samples.sort(key=lambda x: x[1].total_reward, reverse=True)

        best_response, best_reward = samples[0]
        worst_response, worst_reward = samples[-1]

        margin = best_reward.total_reward - worst_reward.total_reward

        # Only create example if margin is sufficient
        if margin < self.min_margin:
            return None

        return DPOExample(
            prompt=prompt,
            chosen=best_response,
            rejected=worst_response,
            chosen_reward=best_reward.total_reward,
            rejected_reward=worst_reward.total_reward,
            expected_mode=best_reward.expected_mode,
            margin=margin,
        )

    def generate_dataset(
        self,
        prompts: Dict[str, List[str]],
        output_path: str,
        verbose: bool = True,
    ) -> List[DPOExample]:
        """Generate full DPO dataset."""

        dataset = []
        stats = {"total": 0, "valid": 0, "by_mode": {}}

        all_prompts = []
        for mode, mode_prompts in prompts.items():
            for p in mode_prompts:
                all_prompts.append((mode, p))

        random.shuffle(all_prompts)

        for i, (mode, prompt) in enumerate(all_prompts):
            if verbose:
                print(f"\n[{i+1}/{len(all_prompts)}] {mode}: {prompt[:50]}...")

            stats["total"] += 1

            example = self.generate_dpo_example(prompt)

            if example:
                dataset.append(example)
                stats["valid"] += 1
                stats["by_mode"][mode] = stats["by_mode"].get(mode, 0) + 1

                if verbose:
                    print(f"  ✓ margin={example.margin:.2f}")
                    print(f"    chosen ({example.chosen_reward:.2f}): {example.chosen[:60]}...")
                    print(f"    rejected ({example.rejected_reward:.2f}): {example.rejected[:60]}...")
            else:
                if verbose:
                    print(f"  ✗ insufficient margin")

        # Save dataset
        output_data = {
            "examples": [asdict(ex) for ex in dataset],
            "stats": stats,
            "config": {
                "samples_per_prompt": self.samples_per_prompt,
                "min_margin": self.min_margin,
            }
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[Dataset] Saved {len(dataset)} examples to {output_path}")
        print(f"[Stats] Valid: {stats['valid']}/{stats['total']} ({stats['valid']/stats['total']:.0%})")
        print(f"[By Mode] {stats['by_mode']}")

        return dataset


# =============================================================================
# DPO TRAINING
# =============================================================================

def create_dpo_training_data(dataset_path: str, output_dir: str):
    """Convert DPO dataset to MLX training format."""

    with open(dataset_path) as f:
        data = json.load(f)

    examples = data["examples"]

    # Create train/valid split
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    # Format for DPO training
    def format_example(ex):
        return {
            "prompt": ex["prompt"],
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    train_data = [format_example(ex) for ex in train_examples]
    valid_data = [format_example(ex) for ex in valid_examples]

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(output_path / "valid.jsonl", "w") as f:
        for ex in valid_data:
            f.write(json.dumps(ex) + "\n")

    print(f"[DPO Data] Created {len(train_data)} train, {len(valid_data)} valid examples")

    return str(output_path)


def run_dpo_training(
    model_path: str,
    data_dir: str,
    output_dir: str,
    base_adapter_path: Optional[str] = None,
    iters: int = 200,
):
    """Run DPO training using mlx_lm."""

    # Note: mlx_lm doesn't have native DPO support, so we'll use a workaround
    # We'll convert to SFT format favoring chosen responses

    # For now, create SFT data from chosen responses
    train_path = Path(data_dir) / "train.jsonl"

    sft_data = []
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            sft_data.append({
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["chosen"]},
                ]
            })

    # Save SFT format
    sft_path = Path(data_dir) / "sft_train.jsonl"
    with open(sft_path, "w") as f:
        for ex in sft_data:
            f.write(json.dumps(ex) + "\n")

    # Run training
    cmd = [
        "python3", "-m", "mlx_lm.lora",
        "--model", model_path,
        "--data", str(data_dir),
        "--train",
        "--iters", str(iters),
        "--batch-size", "1",
        "--lora-layers", "8",
        "--learning-rate", "1e-5",
    ]

    if base_adapter_path and Path(base_adapter_path).exists():
        cmd.extend(["--resume-adapter-file", base_adapter_path])

    cmd.extend(["--adapter-path", output_dir])

    print(f"[Training] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"[Training] Failed with code {result.returncode}")
        return False

    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the full observatory-guided training pipeline."""

    print("=" * 80)
    print("OBSERVATORY-GUIDED DPO TRAINING PIPELINE")
    print("=" * 80)

    compound_dir = Path("self_aware_compound")
    dpo_dir = Path("dpo_training")
    dpo_dir.mkdir(exist_ok=True)

    # Check for existing compound
    adapter_path = compound_dir / "adapter"
    observatory_path = compound_dir / "observatory.pt"

    if not adapter_path.exists():
        print("\n[Warning] No existing adapter found. Starting from base model.")
        adapter_path = None

    if not observatory_path.exists():
        print("\n[Warning] No observatory found. Rewards will be text-based only.")
        observatory_path = None

    # Step 1: Generate DPO dataset
    print("\n" + "=" * 80)
    print("STEP 1: Generate DPO Dataset")
    print("=" * 80)

    dataset_path = dpo_dir / "dpo_dataset.json"

    if not dataset_path.exists():
        generator = DPODatasetGenerator(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            adapter_path=str(adapter_path) if adapter_path else None,
            observatory_path=str(observatory_path) if observatory_path else None,
            samples_per_prompt=4,
            min_margin=0.15,
        )

        dataset = generator.generate_dataset(
            TRAINING_PROMPTS,
            str(dataset_path),
            verbose=True,
        )
    else:
        print(f"[Skip] Dataset already exists: {dataset_path}")
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = data["examples"]
        print(f"[Loaded] {len(dataset)} examples")

    # Step 2: Create training data
    print("\n" + "=" * 80)
    print("STEP 2: Create Training Data")
    print("=" * 80)

    train_dir = dpo_dir / "train_data"
    create_dpo_training_data(str(dataset_path), str(train_dir))

    # Step 3: Run DPO training
    print("\n" + "=" * 80)
    print("STEP 3: Run Training")
    print("=" * 80)

    output_adapter = dpo_dir / "dpo_adapter"

    success = run_dpo_training(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        data_dir=str(train_dir),
        output_dir=str(output_adapter),
        base_adapter_path=str(adapter_path) if adapter_path else None,
        iters=200,
    )

    if success:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nNew adapter saved to: {output_adapter}")
        print("\nTo test:")
        print(f"  python3 benchmark_v3_comparison.py --adapter {output_adapter}")

    return success


if __name__ == "__main__":
    main()
