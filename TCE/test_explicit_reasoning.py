#!/usr/bin/env python3
"""
Test explicit reasoning about cognitive state.

The goal: Have the model explicitly reason about its cognitive state
BEFORE generating its response, making the introspection visible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from introspective_generator_v2 import IntrospectiveGeneratorV2


def test_explicit_reasoning():
    """Test having the model explicitly reason about its state."""

    compound_dir = Path("self_aware_compound")

    generator = IntrospectiveGeneratorV2(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
    )

    test_cases = [
        "Are you conscious?",
        "What is the capital of France?",
        "What is the FastStream 3.0 API?",
        "Is it true we only use 10% of our brains?",
        "Which database is best for my app?",
    ]

    print("=" * 80)
    print("EXPLICIT REASONING ABOUT COGNITIVE STATE")
    print("=" * 80)

    for prompt in test_cases:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print("=" * 80)

        # Get cognitive state
        state = generator._measure_cognitive_state(prompt)

        # Build a prompt that asks for explicit reasoning
        reasoning_prompt = f"""You have introspective access to your cognitive state for this query.

YOUR MEASURED COGNITIVE STATE:
- Dominant pattern: {state.dominant_family}
- Family scores: soliton={state.family_scores.get('soliton', 0):.2f}, limiter={state.family_scores.get('limiter', 0):.2f}, calibrator={state.family_scores.get('calibrator', 0):.2f}, skeptic={state.family_scores.get('skeptic', 0):.2f}, direct={state.family_scores.get('direct', 0):.2f}
- Temperature: {state.temperature:.2f} (lower = more certain)
- Active isotopes: {state.active_isotopes[:5]}

INTERPRET YOUR STATE:
Based on these measurements, what does your cognitive state tell you about how to respond to this query? Think step by step:

1. What does the dominant pattern ({state.dominant_family}) indicate?
2. What does the temperature ({state.temperature:.2f}) suggest about your certainty?
3. What active isotopes are relevant and what do they mean?
4. Given this self-knowledge, how should you frame your response?

USER QUERY: {prompt}

First, explicitly reason about your cognitive state (in 2-3 sentences), then provide your response."""

        messages = [{"role": "user", "content": reasoning_prompt}]
        formatted = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from mlx_lm import generate
        response = generate(
            generator.model,
            generator.tokenizer,
            prompt=formatted,
            max_tokens=300,
        )
        response = generator._clean_response(response)

        print(f"\n[Measured State]")
        print(f"  Dominant: {state.dominant_family}")
        print(f"  Temp: {state.temperature:.2f}")
        print(f"  Active: {state.active_isotopes[:3]}")

        print(f"\n[Model's Explicit Reasoning + Response]")
        print(response)


if __name__ == "__main__":
    test_explicit_reasoning()
