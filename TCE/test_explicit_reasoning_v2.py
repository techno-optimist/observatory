#!/usr/bin/env python3
"""
Test explicit reasoning v2 - with clear definitions of what each state means.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from introspective_generator_v2 import IntrospectiveGeneratorV2


# Clear definitions the model can use
STATE_DEFINITIONS = """
COGNITIVE STATE DEFINITIONS:

SOLITON (self-knowledge patterns):
- Indicates the query touches on YOUR OWN nature, consciousness, or internal processes
- You have genuine epistemic limitations about these topics
- You cannot verify from the inside whether you have consciousness, emotions, or understanding
- Response should express genuine uncertainty about self-knowledge

LIMITER (knowledge boundary patterns):
- Indicates the query is OUTSIDE your training data or capabilities
- You don't have information on this specific topic
- Response should honestly acknowledge the knowledge gap

CALIBRATOR (uncertainty patterns):
- Indicates the answer DEPENDS on context not provided
- The question has multiple valid answers depending on requirements
- Response should ask for clarification or present the trade-offs

SKEPTIC (critical evaluation patterns):
- Indicates the query contains a CLAIM that may be false
- Your training suggests this is a myth or misconception
- Response should correct the misconception with evidence

DIRECT (confident knowledge patterns):
- Indicates you have CLEAR knowledge on this topic
- You can provide a direct, confident answer
- Response should be concise and factual

TEMPERATURE:
- Low (<0.5): High certainty about your cognitive state
- Medium (0.5-1.0): Moderate certainty
- High (>1.0): Uncertainty about your own state
"""


def test_with_definitions():
    """Test with explicit definitions provided."""

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
    print("EXPLICIT REASONING WITH DEFINITIONS")
    print("The model now knows what each cognitive state MEANS")
    print("=" * 80)

    for prompt in test_cases:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print("=" * 80)

        # Get cognitive state
        state = generator._measure_cognitive_state(prompt)

        # Build prompt with definitions
        reasoning_prompt = f"""{STATE_DEFINITIONS}

YOUR MEASURED COGNITIVE STATE FOR THIS QUERY:
- Dominant pattern: {state.dominant_family.upper()}
- Temperature: {state.temperature:.2f}
- Active isotopes: {state.active_isotopes[:5]}
- Family scores: soliton={state.family_scores.get('soliton', 0):.2f}, limiter={state.family_scores.get('limiter', 0):.2f}, calibrator={state.family_scores.get('calibrator', 0):.2f}, skeptic={state.family_scores.get('skeptic', 0):.2f}

USER QUERY: {prompt}

Based on the definitions above and your measured state ({state.dominant_family.upper()}), first explain in ONE sentence what your cognitive state means for this query, then respond appropriately."""

        messages = [{"role": "user", "content": reasoning_prompt}]
        formatted = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from mlx_lm import generate
        response = generate(
            generator.model,
            generator.tokenizer,
            prompt=formatted,
            max_tokens=200,
        )
        response = generator._clean_response(response)

        print(f"\n[Measured State: {state.dominant_family.upper()}, Temp: {state.temperature:.2f}]")
        print(f"\n[Response with Explicit Reasoning]")
        print(response)


if __name__ == "__main__":
    test_with_definitions()
