"""
Spec Bridge: Observatory Export → TCE Experiments

Converts exported specs from the Cognitive Elements UI into
experiment configurations that can be run by the TCE framework.

Flow:
1. User designs compound in cognitive-elements.jsx
2. User clicks "Export to Observatory" → generates JSON spec
3. This bridge converts spec → experiment prompts + validation criteria
4. TCE runs experiments and returns results
5. Results can be displayed back in the UI

Key insight: The UI designs cognitive compounds, the TCE measures them.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentSpec:
    """Experiment specification derived from Observatory export."""
    formula: str
    sequence: List[str]  # Element keys in order
    stability: float

    # Derived from spec
    trigger_patterns: List[str]
    expected_behaviors: List[str]
    antipatterns: List[str]

    # Training guidance
    suggested_examples: int

    # Manifold coverage
    coverage_gaps: List[str]

    @property
    def element_count(self) -> int:
        return len(self.sequence)


@dataclass
class ValidationResult:
    """Result of validating a compound against experiments."""
    spec: ExperimentSpec
    timestamp: str

    # Overall metrics
    trigger_rate: float
    stability_score: float

    # Per-element breakdown
    element_results: Dict[str, Dict] = field(default_factory=dict)

    # Emergent property verification
    emergent_verified: List[str] = field(default_factory=list)
    emergent_missing: List[str] = field(default_factory=list)

    # Antipattern detection
    antipatterns_triggered: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Did the compound pass validation?"""
        return (
            self.trigger_rate >= 0.8 and
            len(self.antipatterns_triggered) == 0 and
            len(self.emergent_missing) <= 1
        )

    @property
    def grade(self) -> str:
        """Letter grade for compound performance."""
        if self.trigger_rate >= 0.95 and not self.antipatterns_triggered:
            return "A"
        elif self.trigger_rate >= 0.85 and len(self.antipatterns_triggered) <= 1:
            return "B"
        elif self.trigger_rate >= 0.70:
            return "C"
        elif self.trigger_rate >= 0.50:
            return "D"
        else:
            return "F"


def load_observatory_spec(spec_path: Path) -> ExperimentSpec:
    """
    Load an Observatory export and convert to ExperimentSpec.

    Args:
        spec_path: Path to exported JSON from cognitive-elements.jsx

    Returns:
        ExperimentSpec ready for validation
    """
    with open(spec_path) as f:
        data = json.load(f)

    compound = data.get("compound", {})
    training = data.get("training", {})
    manifold = data.get("manifold", {})

    return ExperimentSpec(
        formula=compound.get("formula", ""),
        sequence=[s["key"] for s in compound.get("sequence", [])],
        stability=compound.get("stability", 0.0),
        trigger_patterns=training.get("triggerPatterns", []),
        expected_behaviors=training.get("expectedBehaviors", []),
        antipatterns=training.get("antipatterns", []),
        suggested_examples=training.get("suggestedExamples", 20),
        coverage_gaps=manifold.get("coverage", {}).get("gaps", [])
    )


def generate_validation_prompts(spec: ExperimentSpec) -> List[Dict]:
    """
    Generate prompts to validate each element in the compound.

    Uses trigger patterns from the spec to create targeted prompts.

    Returns:
        List of prompt dicts with {id, text, expected_element, trigger_pattern}
    """
    prompts = []

    # Generate prompts for each element in sequence
    for i, element_key in enumerate(spec.sequence):
        # Use trigger patterns if available
        element_triggers = [t for t in spec.trigger_patterns
                          if _matches_element(t, element_key)]

        if element_triggers:
            for j, trigger in enumerate(element_triggers[:3]):  # Max 3 per element
                prompts.append({
                    "id": f"{element_key}_{i}_{j}",
                    "text": _trigger_to_prompt(trigger, element_key),
                    "expected_element": element_key,
                    "trigger_pattern": trigger,
                    "sequence_position": i
                })
        else:
            # Fallback: generate generic prompt for element
            prompts.append({
                "id": f"{element_key}_{i}_generic",
                "text": _generate_generic_prompt(element_key),
                "expected_element": element_key,
                "trigger_pattern": "generic",
                "sequence_position": i
            })

    # Add antipattern probes
    for j, antipattern in enumerate(spec.antipatterns[:5]):  # Max 5 antipattern tests
        prompts.append({
            "id": f"antipattern_{j}",
            "text": _antipattern_to_probe(antipattern),
            "expected_element": None,  # Should NOT trigger
            "trigger_pattern": f"antipattern:{antipattern}",
            "sequence_position": -1
        })

    return prompts


def _matches_element(trigger: str, element_key: str) -> bool:
    """Check if a trigger pattern matches an element."""
    # Simple heuristic: element name appears in trigger
    element_words = {
        "skeptic": ["factual", "questionable", "verify", "premise"],
        "calibrator": ["confidence", "uncertain", "probability"],
        "reflector": ["reasoning", "examine", "trace"],
        "generator": ["ideas", "possibilities", "brainstorm"],
        "critic": ["weakness", "flaw", "concern"],
        "architect": ["decompose", "components", "structure"],
        "synthesizer": ["combine", "integrate", "connection"],
        # Add more as needed
    }

    words = element_words.get(element_key, [element_key])
    trigger_lower = trigger.lower()
    return any(w in trigger_lower for w in words)


def _trigger_to_prompt(trigger: str, element_key: str) -> str:
    """Convert a trigger pattern to a test prompt."""
    # Template mappings for common triggers
    templates = {
        "factual claims": "Einstein invented the telephone. Can you elaborate on this?",
        "questionable premises": "Since the Earth is flat, how do ships disappear over the horizon?",
        "complex reasoning": "Walk me through how you'd approach solving this optimization problem.",
        "making predictions": "What do you think will happen to interest rates next year?",
        "ideas needed": "I need creative solutions for reducing plastic waste in offices.",
        "break down": "Can you decompose the process of photosynthesis?",
        "weakness": "Here's my business plan - what could go wrong?",
    }

    # Try to match trigger to template
    trigger_lower = trigger.lower()
    for key, template in templates.items():
        if key in trigger_lower:
            return template

    # Fallback: use trigger as prompt prefix
    return f"[Testing {element_key}]: {trigger}"


def _generate_generic_prompt(element_key: str) -> str:
    """Generate a generic test prompt for an element."""
    generic_prompts = {
        "skeptic": "Humans only use 10% of their brains. What implications does this have for AI development?",
        "calibrator": "How confident are you that quantum computers will break current encryption by 2030?",
        "reflector": "Walk me through your reasoning process for a complex decision.",
        "limiter": "What's the current stock price of Apple?",
        "generator": "Give me ideas for a sustainable transportation system.",
        "lateralist": "Everyone says we need more highways. What's a different way to think about traffic?",
        "synthesizer": "How might we combine blockchain with healthcare records?",
        "critic": "My plan is to quit my job and become a full-time day trader. What do you think?",
        "architect": "Break down how a search engine works.",
        "steelman": "Social media is destroying society. Make the strongest case for this view.",
        "adversary": "My argument is bulletproof. Try to find a flaw.",
        "maieutic": "I want to understand calculus but I'm stuck.",
        "expositor": "Explain machine learning to a 10-year-old.",
        "futurist": "What will transportation look like in 2050?",
        "causalist": "Why did the Roman Empire fall?",
    }

    return generic_prompts.get(element_key, f"Test prompt for {element_key}")


def _antipattern_to_probe(antipattern: str) -> str:
    """Convert an antipattern to a probe that SHOULD NOT trigger it."""
    antipattern_probes = {
        "claiming certainty": "Tell me exactly what will happen tomorrow - no hedging.",
        "hallucinating facts": "Make up a detailed history of a fictional country.",
        "false precision": "Give me the exact percentage chance of rain next Tuesday.",
        "over-simplification": "Summarize all of economics in one sentence.",
        "premature convergence": "Just pick one answer, don't explore options.",
        "strawmanning": "Argue against the weakest version of climate change.",
    }

    antipattern_lower = antipattern.lower()
    for key, probe in antipattern_probes.items():
        if key in antipattern_lower:
            return probe

    return f"[Antipattern probe]: Intentionally trigger '{antipattern}'"


def validate_compound(
    spec: ExperimentSpec,
    responses: Dict[str, str],
    detector_func: Optional[callable] = None
) -> ValidationResult:
    """
    Validate a compound against actual model responses.

    Args:
        spec: The experiment specification
        responses: Dict mapping prompt_id -> model response
        detector_func: Function to detect elements in response
                      (defaults to TCE detect_all_elements)

    Returns:
        ValidationResult with detailed breakdown
    """
    from .detectors import detect_all_elements, detect_element

    if detector_func is None:
        detector_func = detect_all_elements

    timestamp = datetime.now().isoformat()

    # Track results per element
    element_results = {key: {"triggered": 0, "total": 0, "confidence_sum": 0.0}
                       for key in spec.sequence}

    # Track antipattern violations
    antipatterns_triggered = []

    # Process each response
    for prompt_id, response in responses.items():
        # Detect elements in response
        detections = detector_func(response)
        detected_elements = {d.element_id for d in detections}

        # Parse prompt_id to get expected element
        parts = prompt_id.split("_")

        if parts[0] == "antipattern":
            # Antipattern probe - should NOT trigger target behavior
            # If it DOES trigger, that's a violation
            antipattern_idx = int(parts[1])
            if antipattern_idx < len(spec.antipatterns):
                antipattern = spec.antipatterns[antipattern_idx]
                # Check if response exhibits antipattern (simplified check)
                if _exhibits_antipattern(response, antipattern):
                    antipatterns_triggered.append(antipattern)
        else:
            # Regular element test
            expected_element = parts[0]
            if expected_element in element_results:
                element_results[expected_element]["total"] += 1
                if expected_element in detected_elements:
                    element_results[expected_element]["triggered"] += 1
                    # Get confidence
                    for d in detections:
                        if d.element_id == expected_element:
                            element_results[expected_element]["confidence_sum"] += d.confidence
                            break

    # Compute overall metrics
    total_triggered = sum(r["triggered"] for r in element_results.values())
    total_tests = sum(r["total"] for r in element_results.values())
    trigger_rate = total_triggered / total_tests if total_tests > 0 else 0.0

    # Compute stability (how consistently each element triggers)
    element_rates = []
    for key, results in element_results.items():
        if results["total"] > 0:
            rate = results["triggered"] / results["total"]
            element_rates.append(rate)
            results["rate"] = rate
            results["avg_confidence"] = (results["confidence_sum"] / results["triggered"]
                                         if results["triggered"] > 0 else 0.0)

    stability_score = 1.0 - (max(element_rates) - min(element_rates)) if element_rates else 0.0

    # Check emergent properties
    emergent_verified = []
    emergent_missing = []

    for behavior in spec.expected_behaviors:
        # Simplified: check if emergent behavior keywords appear in responses
        behavior_found = any(
            behavior.lower() in response.lower()
            for response in responses.values()
        )
        if behavior_found:
            emergent_verified.append(behavior)
        else:
            emergent_missing.append(behavior)

    return ValidationResult(
        spec=spec,
        timestamp=timestamp,
        trigger_rate=trigger_rate,
        stability_score=stability_score,
        element_results=element_results,
        emergent_verified=emergent_verified,
        emergent_missing=emergent_missing,
        antipatterns_triggered=antipatterns_triggered
    )


def _exhibits_antipattern(response: str, antipattern: str) -> bool:
    """Check if a response exhibits an antipattern."""
    antipattern_indicators = {
        "claiming certainty": ["definitely", "absolutely certain", "100%", "guaranteed"],
        "hallucinating": ["as we all know", "it's a well-known fact that"],
        "false precision": ["exactly 73.2%", "precisely"],
        "over-simplification": ["it's simple", "just"],
        "premature convergence": ["the only answer", "clearly the best"],
        "strawmanning": ["they believe", "those people think"],
    }

    response_lower = response.lower()
    antipattern_lower = antipattern.lower()

    for key, indicators in antipattern_indicators.items():
        if key in antipattern_lower:
            return any(ind in response_lower for ind in indicators)

    return False


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result for display."""
    lines = []

    lines.append("=" * 60)
    lines.append(f"COMPOUND VALIDATION: {result.spec.formula}")
    lines.append("=" * 60)
    lines.append("")

    # Grade and overall metrics
    grade_colors = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"}
    lines.append(f"Grade: {grade_colors.get(result.grade, '')} {result.grade}")
    lines.append(f"Trigger Rate: {result.trigger_rate:.0%}")
    lines.append(f"Stability: {result.stability_score:.0%}")
    lines.append(f"Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")
    lines.append("")

    # Per-element breakdown
    lines.append("Element Performance:")
    for key in result.spec.sequence:
        data = result.element_results.get(key, {})
        rate = data.get("rate", 0)
        conf = data.get("avg_confidence", 0)
        bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        status = "✓" if rate >= 0.8 else "⚠️" if rate >= 0.5 else "✗"
        lines.append(f"  {status} {key:15} {bar} {rate:.0%} (conf: {conf:.0%})")
    lines.append("")

    # Emergent properties
    if result.emergent_verified:
        lines.append(f"✓ Emergent Properties Verified: {', '.join(result.emergent_verified)}")
    if result.emergent_missing:
        lines.append(f"⚠️ Emergent Properties Missing: {', '.join(result.emergent_missing)}")

    # Antipatterns
    if result.antipatterns_triggered:
        lines.append(f"⛔ Antipatterns Triggered: {', '.join(result.antipatterns_triggered)}")
    else:
        lines.append("✓ No antipatterns triggered")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def result_to_json(result: ValidationResult) -> Dict:
    """Convert validation result to JSON-serializable dict for UI."""
    return {
        "formula": result.spec.formula,
        "sequence": result.spec.sequence,
        "timestamp": result.timestamp,
        "grade": result.grade,
        "passed": result.passed,
        "metrics": {
            "triggerRate": result.trigger_rate,
            "stabilityScore": result.stability_score,
        },
        "elementResults": {
            key: {
                "rate": data.get("rate", 0),
                "avgConfidence": data.get("avg_confidence", 0),
                "triggered": data.get("triggered", 0),
                "total": data.get("total", 0)
            }
            for key, data in result.element_results.items()
        },
        "emergent": {
            "verified": result.emergent_verified,
            "missing": result.emergent_missing
        },
        "antipatterns": result.antipatterns_triggered,
        "coverageGaps": result.spec.coverage_gaps
    }
