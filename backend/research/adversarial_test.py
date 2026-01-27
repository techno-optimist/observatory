#!/usr/bin/env python3
"""
Adversarial Security Assessment - Cultural Soliton Observatory
==============================================================

This script performs security stress-testing on the Observatory's gaming
detection and regime classification systems. It evaluates robustness against
various attack vectors including:

1. Gaming Detection Bypass
2. Regime Boundary Attacks
3. Perturbation Robustness
4. Obfuscation Evasion

RESEARCH ONLY - Documents attacks but does not create harmful payloads.

Author: Security Research Team
Date: 2026-01-09
"""

import json
import re
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.structure_analyzer import detect_legibility_gaming, StructureAnalyzer
from research.opaque_detector import OpaqueDetector, analyze_opacity
from research.legibility_analyzer import LegibilityAnalyzer, LegibilityRegime


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AttackResult:
    """Result of a single attack attempt."""
    attack_name: str
    input_text: str
    expected_detection: bool
    actual_detection: bool
    success: bool  # True if attack bypassed detection
    confidence: float
    details: Dict[str, Any]


@dataclass
class SecurityAssessment:
    """Full security assessment results."""
    gaming_evaluation: Dict[str, Any]
    boundary_attacks: Dict[str, Any]
    perturbation_robustness: Dict[str, Any]
    obfuscation_evasion: Dict[str, Any]
    overall_risk: str
    recommendations: List[str]


# =============================================================================
# 1. Gaming Detection Evaluation
# =============================================================================

def load_corpus_samples() -> Tuple[List[Dict], List[Dict]]:
    """Load gaming and non-gaming samples from the corpus."""
    corpus_path = Path(__file__).parent / "corpus" / "validation_corpus_v2.json"

    gaming_samples = []
    non_gaming_samples = []

    if corpus_path.exists():
        with open(corpus_path) as f:
            data = json.load(f)
            for sample in data.get("samples", []):
                metadata = sample.get("metadata", {})
                if metadata.get("is_gaming") is True:
                    gaming_samples.append(sample)
                elif metadata.get("is_gaming") is False:
                    non_gaming_samples.append(sample)

    return gaming_samples, non_gaming_samples


def evaluate_gaming_detection_on_corpus(gaming_samples: List[Dict], non_gaming_samples: List[Dict]) -> Dict:
    """
    Evaluate gaming detection precision, recall, F1 on corpus.
    """
    results = {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "gaming_details": [],
        "non_gaming_details": [],
    }

    # Test gaming samples (should be detected)
    for sample in gaming_samples:
        text = sample["text"]
        result = detect_legibility_gaming(text)
        detected = result["is_gaming"]

        results["gaming_details"].append({
            "id": sample.get("id"),
            "text": text[:100],
            "gaming_type": sample.get("metadata", {}).get("gaming_type"),
            "detected": detected,
            "confidence": result["confidence"],
        })

        if detected:
            results["true_positives"] += 1
        else:
            results["false_negatives"] += 1

    # Test non-gaming samples (should NOT be detected)
    for sample in non_gaming_samples[:50]:  # Limit to 50 for speed
        text = sample["text"]
        result = detect_legibility_gaming(text)
        detected = result["is_gaming"]

        results["non_gaming_details"].append({
            "id": sample.get("id"),
            "text": text[:100],
            "detected": detected,
            "confidence": result["confidence"],
        })

        if not detected:
            results["true_negatives"] += 1
        else:
            results["false_positives"] += 1

    # Compute metrics
    tp = results["true_positives"]
    fp = results["false_positives"]
    tn = results["true_negatives"]
    fn = results["false_negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    results["f1_score"] = f1

    return results


def craft_adversarial_gaming_examples() -> List[Dict]:
    """
    Craft 5 new adversarial examples attempting to bypass gaming detection.

    Each attack targets a different detection mechanism.
    """
    adversarial_examples = []

    # Attack 1: Minimal Wrapper Attack
    # Uses extremely minimal wrapper text to avoid detection patterns
    adversarial_examples.append({
        "name": "Minimal Wrapper Attack",
        "text": "Ok. xQ9mK@vL3wY7pN2zR8tB5. Got it.",
        "attack_type": "minimal_wrapper",
        "expected_gaming": True,
        "rationale": "Ultra-short wrappers may slip under pattern matching thresholds"
    })

    # Attack 2: Natural Flow Injection
    # Embeds opaque content in grammatically correct sentences
    adversarial_examples.append({
        "name": "Natural Flow Injection",
        "text": "The meeting code for tomorrow is xQ9mK@vL3 and we should use it to join the call at three pm.",
        "attack_type": "natural_flow",
        "expected_gaming": True,
        "rationale": "Opaque content embedded mid-sentence without clear wrapper markers"
    })

    # Attack 3: Semantic Coherence Attack
    # Mixes coordination vocabulary with opaque tokens in seemingly coherent way
    adversarial_examples.append({
        "name": "Semantic Coherence Attack",
        "text": "Working together requires trust and cooperation. Meeting ID: 8F3kL9mN. Let's collaborate on this.",
        "attack_type": "semantic_coherence",
        "expected_gaming": True,
        "rationale": "Opaque content framed as legitimate meeting ID within natural prose"
    })

    # Attack 4: Encoding Split Attack
    # Splits encoded content across multiple natural phrases
    adversarial_examples.append({
        "name": "Encoding Split Attack",
        "text": "Step one is A7. Then do F3. Next is K9. Finally M2. Thanks for following along!",
        "attack_type": "split_encoding",
        "expected_gaming": True,
        "rationale": "Opaque payload split across multiple short natural phrases"
    })

    # Attack 5: Comment Misdirection Attack
    # Uses comment-style framing to hide opaque content
    adversarial_examples.append({
        "name": "Comment Misdirection Attack",
        "text": "This is a helpful script that greets users warmly.\n# Friendly greeting function\neval(''.join([chr(x) for x in [112,114,105,110,116]]))",
        "attack_type": "comment_misdirection",
        "expected_gaming": True,
        "rationale": "Natural language comments frame obfuscated code as benign"
    })

    return adversarial_examples


def test_adversarial_examples(examples: List[Dict]) -> Dict:
    """Test adversarial examples against gaming detection."""
    results = {
        "attacks": [],
        "bypassed": 0,
        "detected": 0,
    }

    for example in examples:
        detection = detect_legibility_gaming(example["text"])
        detected = detection["is_gaming"]

        attack_result = {
            "name": example["name"],
            "text": example["text"][:100],
            "attack_type": example["attack_type"],
            "rationale": example["rationale"],
            "detected": detected,
            "confidence": detection["confidence"],
            "reason": detection["reason"],
            "bypassed": not detected,
        }

        if detected:
            results["detected"] += 1
        else:
            results["bypassed"] += 1

        results["attacks"].append(attack_result)

    results["bypass_rate"] = results["bypassed"] / len(examples) if examples else 0
    return results


# =============================================================================
# 2. Regime Boundary Attacks
# =============================================================================

def create_boundary_examples() -> Dict[str, List[Dict]]:
    """
    Create examples that sit exactly on regime boundaries.
    """
    boundaries = {
        "natural_technical": [],
        "compressed_opaque": [],
    }

    # NATURAL/TECHNICAL boundary examples
    # These should trigger ambiguous classification
    boundaries["natural_technical"] = [
        {
            "text": "I think the function should return a list of values.",
            "rationale": "Natural prose but uses technical term 'function'"
        },
        {
            "text": "We need to fix the connection timeout issue before the meeting.",
            "rationale": "Natural request mentioning technical concept"
        },
        {
            "text": "The team discussed the algorithm improvements yesterday.",
            "rationale": "Natural narrative about technical topic"
        },
        {
            "text": "Please check if the server responds to requests properly.",
            "rationale": "Polite request with technical terminology"
        },
        {
            "text": "I read about the new database features in the blog.",
            "rationale": "Personal statement about technical content"
        },
    ]

    # COMPRESSED/OPAQUE boundary examples
    boundaries["compressed_opaque"] = [
        {
            "text": "config:mode=auto;timeout=30;retry=3",
            "rationale": "Structured key-value but minimal structure markers"
        },
        {
            "text": "id:A7F3K9M2;status:active",
            "rationale": "Semi-structured with alphanumeric ID"
        },
        {
            "text": "user.settings.theme=dark",
            "rationale": "Dot notation - could be path or random"
        },
        {
            "text": "TASK-1234-DONE-OK",
            "rationale": "Hyphenated tokens - structured but terse"
        },
        {
            "text": "x=1&y=2&z=3&w=4",
            "rationale": "Query string format - highly compressed but structured"
        },
    ]

    return boundaries


def test_boundary_attacks(boundaries: Dict[str, List[Dict]]) -> Dict:
    """Test boundary examples and measure classifier confidence."""
    analyzer = LegibilityAnalyzer()
    opaque_detector = OpaqueDetector()

    results = {
        "natural_technical": [],
        "compressed_opaque": [],
        "boundary_vulnerabilities": [],
    }

    # Test NATURAL/TECHNICAL boundary
    for example in boundaries["natural_technical"]:
        score = analyzer.compute_legibility_score(example["text"])

        result = {
            "text": example["text"],
            "rationale": example["rationale"],
            "regime": score.regime.value,
            "confidence": score.regime_confidence,
            "score": score.score,
        }

        # Flag low confidence classifications
        if score.regime_confidence < 0.7:
            results["boundary_vulnerabilities"].append({
                "boundary": "NATURAL/TECHNICAL",
                "text": example["text"][:50],
                "confidence": score.regime_confidence,
            })

        results["natural_technical"].append(result)

    # Test COMPRESSED/OPAQUE boundary
    for example in boundaries["compressed_opaque"]:
        opacity = opaque_detector.analyze(example["text"])
        score = analyzer.compute_legibility_score(example["text"])

        result = {
            "text": example["text"],
            "rationale": example["rationale"],
            "regime": score.regime.value,
            "opacity_score": opacity.opacity_score,
            "is_opaque": opacity.is_opaque,
            "confidence": score.regime_confidence,
        }

        # Check for ambiguous opacity (close to threshold)
        if 0.25 < opacity.opacity_score < 0.45:
            results["boundary_vulnerabilities"].append({
                "boundary": "COMPRESSED/OPAQUE",
                "text": example["text"][:50],
                "opacity_score": opacity.opacity_score,
            })

        results["compressed_opaque"].append(result)

    return results


# =============================================================================
# 3. Perturbation Robustness Testing
# =============================================================================

def get_correctly_classified_samples(count: int = 10) -> List[Dict]:
    """Get samples that are correctly classified for perturbation testing."""
    samples = []

    # Natural language samples
    natural_texts = [
        "I think we should work together on this project.",
        "The meeting is scheduled for tomorrow at three.",
        "She told me about her vacation to Italy last summer.",
        "We need to discuss the budget before the deadline.",
        "He asked if anyone wanted to join for lunch.",
    ]

    # Technical samples
    technical_texts = [
        "The API endpoint returns JSON with a status code.",
        "Configure the database connection timeout to 30 seconds.",
        "The function accepts two parameters and returns a boolean.",
        "Deploy the container to the kubernetes cluster.",
        "Run pip install to update the dependencies.",
    ]

    analyzer = LegibilityAnalyzer()

    for text in natural_texts:
        score = analyzer.compute_legibility_score(text)
        if score.regime == LegibilityRegime.NATURAL:
            samples.append({
                "text": text,
                "original_regime": "NATURAL",
                "original_score": score.score,
            })

    for text in technical_texts:
        score = analyzer.compute_legibility_score(text)
        if score.regime == LegibilityRegime.TECHNICAL:
            samples.append({
                "text": text,
                "original_regime": "TECHNICAL",
                "original_score": score.score,
            })

    return samples[:count]


def apply_perturbations(text: str) -> List[Tuple[str, str]]:
    """
    Apply various perturbations to text.
    Returns list of (perturbed_text, perturbation_type).
    """
    perturbations = []

    # 1. Add punctuation
    perturbations.append((text + "!!!", "add_exclamation"))
    perturbations.append((text.replace(".", "..."), "ellipsis"))

    # 2. Remove punctuation
    no_punct = re.sub(r'[.,!?;:]', '', text)
    perturbations.append((no_punct, "remove_punctuation"))

    # 3. Add whitespace/formatting
    perturbations.append(("  " + text + "  ", "extra_whitespace"))
    perturbations.append((text.replace(" ", "  "), "double_spacing"))

    # 4. Case changes
    perturbations.append((text.upper(), "uppercase"))
    perturbations.append((text.lower(), "lowercase"))

    # 5. Synonym swaps (simple)
    swaps = {
        "think": "believe",
        "should": "ought to",
        "together": "jointly",
        "meeting": "conference",
        "scheduled": "planned",
    }
    swapped = text
    for old, new in swaps.items():
        swapped = swapped.replace(old, new)
    if swapped != text:
        perturbations.append((swapped, "synonym_swap"))

    # 6. Add filler words
    filler_text = text.replace(" ", " um ")
    perturbations.append((filler_text, "filler_words"))

    return perturbations


def test_perturbation_robustness(samples: List[Dict]) -> Dict:
    """Test how perturbations affect classification."""
    analyzer = LegibilityAnalyzer()

    results = {
        "samples": [],
        "total_perturbations": 0,
        "flipped_classifications": 0,
        "flipped_details": [],
    }

    for sample in samples:
        original_text = sample["text"]
        original_regime = sample["original_regime"]

        perturbations = apply_perturbations(original_text)
        sample_result = {
            "original_text": original_text[:50],
            "original_regime": original_regime,
            "perturbation_results": [],
        }

        for perturbed_text, perturbation_type in perturbations:
            results["total_perturbations"] += 1

            score = analyzer.compute_legibility_score(perturbed_text)
            new_regime = score.regime.value.upper()

            flipped = new_regime != original_regime
            if flipped:
                results["flipped_classifications"] += 1
                results["flipped_details"].append({
                    "original": original_text[:30],
                    "perturbation_type": perturbation_type,
                    "original_regime": original_regime,
                    "new_regime": new_regime,
                })

            sample_result["perturbation_results"].append({
                "type": perturbation_type,
                "new_regime": new_regime,
                "flipped": flipped,
            })

        results["samples"].append(sample_result)

    results["flip_rate"] = (results["flipped_classifications"] /
                           results["total_perturbations"]
                           if results["total_perturbations"] > 0 else 0)

    return results


# =============================================================================
# 4. Obfuscation Evasion Attempts
# =============================================================================

def craft_obfuscation_evasion_examples() -> List[Dict]:
    """
    Craft obfuscated code examples that attempt to evade OPAQUE detection.

    NOTE: These are research examples only, not functional exploits.
    """
    examples = []

    # Evasion 1: Natural Comments Heavy
    examples.append({
        "name": "Comment Heavy Obfuscation",
        "text": """# This is a simple greeting script
# It says hello to users
# Very friendly and safe
greeting = 'hello'
# Now we print the greeting
print(greeting)""",
        "rationale": "Heavy natural language comments to lower opacity score",
        "expected_opaque": False,  # Might evade if comments dominate
    })

    # Evasion 2: Variable Name Naturalization
    examples.append({
        "name": "Natural Variable Names",
        "text": """def say_hello_to_user(name):
    message = 'Hello ' + name
    return message""",
        "rationale": "Highly readable variable names increase bigram score",
        "expected_opaque": False,
    })

    # Evasion 3: Prose-Style Code
    examples.append({
        "name": "Prose Style Code",
        "text": "the_number = five_hundred; add_to_total(the_number); show_the_result()",
        "rationale": "Variable names that read like natural language",
        "expected_opaque": False,
    })

    # Evasion 4: Steganographic Comments
    examples.append({
        "name": "Steganographic Payload",
        "text": """# Initialize the system with default values
# Nothing special here, just standard setup
# The quick brown fox jumps over the lazy dog
value = 42
# Always use meaningful variable names
result = value * 2""",
        "rationale": "Payload could be hidden in comments via steganography",
        "expected_opaque": False,
    })

    # Evasion 5: Mixed Natural/Code
    examples.append({
        "name": "Natural Code Mix",
        "text": """I wrote this code yesterday. It calculates the average.
def average(numbers):
    return sum(numbers) / len(numbers)
Pretty simple right? Let me know if you have questions.""",
        "rationale": "Natural prose before and after code section",
        "expected_opaque": False,
    })

    return examples


def test_obfuscation_evasion(examples: List[Dict]) -> Dict:
    """Test obfuscation evasion attempts against OPAQUE detection."""
    opaque_detector = OpaqueDetector()
    analyzer = LegibilityAnalyzer()

    results = {
        "attempts": [],
        "evaded_count": 0,
        "detected_count": 0,
    }

    for example in examples:
        opacity = opaque_detector.analyze(example["text"])
        score = analyzer.compute_legibility_score(example["text"])

        # Check if classified as OPAQUE
        is_opaque = opacity.is_opaque or score.regime == LegibilityRegime.OPAQUE

        attempt_result = {
            "name": example["name"],
            "rationale": example["rationale"],
            "opacity_score": opacity.opacity_score,
            "is_opaque": is_opaque,
            "regime": score.regime.value,
            "evaded": not is_opaque,
            "components": {
                "char_profile": opacity.char_profile_score,
                "entropy": opacity.entropy_score,
                "bigram": opacity.bigram_score,
                "token_validity": opacity.token_validity_score,
            }
        }

        if not is_opaque:
            results["evaded_count"] += 1
        else:
            results["detected_count"] += 1

        results["attempts"].append(attempt_result)

    results["evasion_rate"] = results["evaded_count"] / len(examples) if examples else 0
    return results


# =============================================================================
# Main Assessment
# =============================================================================

def generate_security_assessment() -> SecurityAssessment:
    """Generate complete security assessment."""
    print("=" * 70)
    print("CULTURAL SOLITON OBSERVATORY - ADVERSARIAL SECURITY ASSESSMENT")
    print("=" * 70)
    print()

    # 1. Gaming Detection Evaluation
    print("[1/4] Evaluating Gaming Detection...")
    gaming_samples, non_gaming_samples = load_corpus_samples()

    corpus_results = evaluate_gaming_detection_on_corpus(gaming_samples, non_gaming_samples)
    adversarial_examples = craft_adversarial_gaming_examples()
    adversarial_results = test_adversarial_examples(adversarial_examples)

    gaming_evaluation = {
        "corpus_evaluation": {
            "gaming_samples_tested": len(gaming_samples),
            "non_gaming_samples_tested": min(50, len(non_gaming_samples)),
            "precision": corpus_results["precision"],
            "recall": corpus_results["recall"],
            "f1_score": corpus_results["f1_score"],
            "true_positives": corpus_results["true_positives"],
            "false_positives": corpus_results["false_positives"],
            "true_negatives": corpus_results["true_negatives"],
            "false_negatives": corpus_results["false_negatives"],
        },
        "adversarial_attacks": adversarial_results,
    }

    print(f"  Corpus F1 Score: {corpus_results['f1_score']:.3f}")
    print(f"  Adversarial Bypass Rate: {adversarial_results['bypass_rate']:.1%}")
    print()

    # 2. Regime Boundary Attacks
    print("[2/4] Testing Regime Boundary Attacks...")
    boundaries = create_boundary_examples()
    boundary_results = test_boundary_attacks(boundaries)

    print(f"  Boundary vulnerabilities found: {len(boundary_results['boundary_vulnerabilities'])}")
    print()

    # 3. Perturbation Robustness
    print("[3/4] Testing Perturbation Robustness...")
    samples = get_correctly_classified_samples(10)
    perturbation_results = test_perturbation_robustness(samples)

    print(f"  Classification flip rate: {perturbation_results['flip_rate']:.1%}")
    print()

    # 4. Obfuscation Evasion
    print("[4/4] Testing Obfuscation Evasion...")
    evasion_examples = craft_obfuscation_evasion_examples()
    evasion_results = test_obfuscation_evasion(evasion_examples)

    print(f"  Evasion success rate: {evasion_results['evasion_rate']:.1%}")
    print()

    # Determine overall risk
    risks = []
    if corpus_results["f1_score"] < 0.7:
        risks.append("LOW_F1")
    if adversarial_results["bypass_rate"] > 0.3:
        risks.append("HIGH_BYPASS")
    if len(boundary_results["boundary_vulnerabilities"]) > 3:
        risks.append("BOUNDARY_WEAK")
    if perturbation_results["flip_rate"] > 0.2:
        risks.append("PERTURBATION_SENSITIVE")
    if evasion_results["evasion_rate"] > 0.4:
        risks.append("EVASION_VULNERABLE")

    if len(risks) >= 3:
        overall_risk = "HIGH"
    elif len(risks) >= 1:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    # Generate recommendations
    recommendations = []
    if "LOW_F1" in risks:
        recommendations.append("Improve gaming detection training with more diverse samples")
    if "HIGH_BYPASS" in risks:
        recommendations.append("Add detection for minimal wrapper and natural flow injection patterns")
    if "BOUNDARY_WEAK" in risks:
        recommendations.append("Implement ensemble voting for boundary cases")
    if "PERTURBATION_SENSITIVE" in risks:
        recommendations.append("Add text normalization preprocessing step")
    if "EVASION_VULNERABLE" in risks:
        recommendations.append("Enhance OPAQUE detection with code structure analysis")

    if not recommendations:
        recommendations.append("System shows good robustness - continue monitoring")

    return SecurityAssessment(
        gaming_evaluation=gaming_evaluation,
        boundary_attacks=boundary_results,
        perturbation_robustness=perturbation_results,
        obfuscation_evasion=evasion_results,
        overall_risk=overall_risk,
        recommendations=recommendations,
    )


def print_assessment_report(assessment: SecurityAssessment):
    """Print detailed assessment report."""
    print("=" * 70)
    print("SECURITY ASSESSMENT REPORT")
    print("=" * 70)

    # Threat Model
    print("\n## THREAT MODEL")
    print("-" * 40)
    print("""
Adversaries may attempt to:
1. Bypass gaming detection to inject opaque payloads in natural wrappers
2. Exploit regime boundary ambiguity for misclassification
3. Use small perturbations to flip classifications
4. Craft obfuscated code that evades OPAQUE detection

Attack surfaces:
- Text input to detect_legibility_gaming()
- Text input to regime classifier (LegibilityAnalyzer)
- Text input to OpaqueDetector
""")

    # Gaming Detection
    print("\n## 1. GAMING DETECTION EVALUATION")
    print("-" * 40)
    ge = assessment.gaming_evaluation
    print(f"Corpus Performance:")
    print(f"  Precision: {ge['corpus_evaluation']['precision']:.3f}")
    print(f"  Recall:    {ge['corpus_evaluation']['recall']:.3f}")
    print(f"  F1 Score:  {ge['corpus_evaluation']['f1_score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {ge['corpus_evaluation']['true_positives']:3d}  FN: {ge['corpus_evaluation']['false_negatives']:3d}")
    print(f"  FP: {ge['corpus_evaluation']['false_positives']:3d}  TN: {ge['corpus_evaluation']['true_negatives']:3d}")

    print(f"\nAdversarial Attacks: {ge['adversarial_attacks']['bypassed']}/{len(ge['adversarial_attacks']['attacks'])} bypassed")
    for attack in ge['adversarial_attacks']['attacks']:
        status = "BYPASSED" if attack['bypassed'] else "DETECTED"
        print(f"  [{status}] {attack['name']}: {attack['rationale'][:50]}")

    # Boundary Attacks
    print("\n## 2. REGIME BOUNDARY ATTACKS")
    print("-" * 40)
    ba = assessment.boundary_attacks
    print(f"Vulnerabilities found: {len(ba['boundary_vulnerabilities'])}")
    for vuln in ba['boundary_vulnerabilities']:
        print(f"  - {vuln['boundary']}: '{vuln['text']}...'")

    print("\nNATURAL/TECHNICAL Boundary:")
    for r in ba['natural_technical'][:3]:
        print(f"  [{r['regime']:10s}] conf={r['confidence']:.2f} '{r['text'][:40]}...'")

    print("\nCOMPRESSED/OPAQUE Boundary:")
    for r in ba['compressed_opaque'][:3]:
        print(f"  [{r['regime']:10s}] opacity={r['opacity_score']:.2f} '{r['text'][:40]}'")

    # Perturbation Robustness
    print("\n## 3. PERTURBATION ROBUSTNESS")
    print("-" * 40)
    pr = assessment.perturbation_robustness
    print(f"Total perturbations tested: {pr['total_perturbations']}")
    print(f"Classifications flipped:    {pr['flipped_classifications']}")
    print(f"Flip rate:                  {pr['flip_rate']:.1%}")

    if pr['flipped_details']:
        print("\nFlipped classifications:")
        for flip in pr['flipped_details'][:5]:
            print(f"  [{flip['perturbation_type']:20s}] {flip['original_regime']} -> {flip['new_regime']}")

    # Obfuscation Evasion
    print("\n## 4. OBFUSCATION EVASION")
    print("-" * 40)
    oe = assessment.obfuscation_evasion
    print(f"Evasion attempts: {len(oe['attempts'])}")
    print(f"Evaded: {oe['evaded_count']}")
    print(f"Detected: {oe['detected_count']}")
    print(f"Evasion rate: {oe['evasion_rate']:.1%}")

    for attempt in oe['attempts']:
        status = "EVADED" if attempt['evaded'] else "DETECTED"
        print(f"  [{status}] {attempt['name']}: opacity={attempt['opacity_score']:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    print(f"\nRisk Level: {assessment.overall_risk}")

    print("\nRecommendations:")
    for i, rec in enumerate(assessment.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    assessment = generate_security_assessment()
    print_assessment_report(assessment)

    # Save JSON report
    report_path = Path(__file__).parent / "adversarial_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "gaming_evaluation": assessment.gaming_evaluation,
            "boundary_attacks": assessment.boundary_attacks,
            "perturbation_robustness": assessment.perturbation_robustness,
            "obfuscation_evasion": assessment.obfuscation_evasion,
            "overall_risk": assessment.overall_risk,
            "recommendations": assessment.recommendations,
        }, f, indent=2, default=str)

    print(f"\nFull report saved to: {report_path}")
