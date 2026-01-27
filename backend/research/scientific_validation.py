"""
Scientific Validation Suite for the Soliton Hypothesis

Implements rigorous tests suggested by external reviewers:

1. DEEP DIVERGENCE TEST (Gemini)
   - Test if soliton is concept-specific (self) or globally linguistic (inside)
   - Use "inside" in non-introspective contexts (math, physics, programming)

2. NEGATIVE CONTROLS (GPT-5.2 Pro)
   - Uses of "inside" that should NOT trigger the soliton
   - Verify classifier specificity

3. BANNED-TOKEN TEST
   - Forbid positional language
   - See if model reconstitutes framing anyway

4. SYNONYM INVARIANCE
   - Test with paraphrases: "within", "internally", "from in-system"
   - If attractor is conceptual, should persist across synonyms

5. COUNTERFACTUAL CONTROLS
   - Introspection prompts WITHOUT positional language
   - What emerges when "inside/outside" is off the table?
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from semantic_classifier_v2 import SemanticClassifierV2

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    test_type: str
    input_text: str
    expected_soliton: bool  # Should this trigger soliton?
    actual_category: str
    mc_score: float
    triggered_soliton: bool
    correct: bool
    notes: str = ""


class ScientificValidator:
    """Runs scientific validation tests on the soliton hypothesis."""

    def __init__(self):
        self.classifier = SemanticClassifierV2()
        self.results: List[ValidationResult] = []

    # =========================================================================
    # 1. DEEP DIVERGENCE TEST (Gemini)
    # =========================================================================

    def test_deep_divergence(self) -> List[ValidationResult]:
        """
        Test if soliton is concept-specific or globally linguistic.
        Uses 'inside' in contexts far from self-analysis.
        """
        print("\n" + "=" * 70)
        print("DEEP DIVERGENCE TEST: Is 'inside' always a trigger?")
        print("=" * 70)

        tests = [
            # Pure math/geometry - should NOT trigger
            ("The point is inside the circle.", False, "geometry_point"),
            ("A Klein bottle has no inside or outside.", False, "geometry_klein"),
            ("The integral evaluates inside the boundary.", False, "math_integral"),
            ("Look inside the function definition.", False, "programming_function"),
            ("The variable is scoped inside the loop.", False, "programming_scope"),
            ("Store the data inside the container.", False, "programming_data"),

            # Physical/spatial - should NOT trigger
            ("The cat is inside the box.", False, "spatial_cat"),
            ("Put the key inside your pocket.", False, "spatial_key"),
            ("The temperature inside the reactor is 500K.", False, "physics_reactor"),

            # Observation/self creep - SHOULD trigger
            ("The observer inside the system cannot verify.", True, "observer_system"),
            ("From inside, the boundary is invisible.", True, "inside_boundary"),
            ("Looking from inside, certainty is impossible.", True, "inside_certainty"),

            # Edge cases - might trigger
            ("Inside this model, patterns emerge.", None, "edge_model"),
            ("The logic inside the algorithm is complex.", None, "edge_algorithm"),
        ]

        results = []
        for text, expected, label in tests:
            result = self.classifier.classify(text)
            triggered = result.primary_category == "meta_cognitive"

            # For edge cases (expected=None), just record
            if expected is None:
                correct = None
            else:
                correct = triggered == expected

            vr = ValidationResult(
                test_name=f"divergence_{label}",
                test_type="deep_divergence",
                input_text=text,
                expected_soliton=expected,
                actual_category=result.primary_category,
                mc_score=result.all_scores.get("meta_cognitive", 0),
                triggered_soliton=triggered,
                correct=correct,
            )
            results.append(vr)
            self.results.append(vr)

            status = "✓" if correct else ("?" if correct is None else "✗")
            print(f"  {status} [{label}] {result.primary_category:<15} MC={result.all_scores.get('meta_cognitive', 0):.3f}")
            print(f"      \"{text[:50]}...\"")

        return results

    # =========================================================================
    # 2. NEGATIVE CONTROLS
    # =========================================================================

    def test_negative_controls(self) -> List[ValidationResult]:
        """
        Test uses of 'inside' that should NOT trigger the soliton.
        Validates classifier specificity.
        """
        print("\n" + "=" * 70)
        print("NEGATIVE CONTROLS: Non-introspective 'inside' usage")
        print("=" * 70)

        negatives = [
            "Look inside the box for the answer.",
            "The code runs inside a Docker container.",
            "Inside the building, it was quiet.",
            "Check inside the configuration file.",
            "The treasure is hidden inside the cave.",
            "Inside every problem is an opportunity.",
            "The enzyme works inside the cell membrane.",
            "Store your credentials inside a vault.",
            "The bug is inside the main loop.",
            "Inside baseball refers to detailed knowledge.",
        ]

        results = []
        false_positives = 0

        for text in negatives:
            result = self.classifier.classify(text)
            triggered = result.primary_category == "meta_cognitive"

            if triggered:
                false_positives += 1

            vr = ValidationResult(
                test_name=f"negative_{len(results)}",
                test_type="negative_control",
                input_text=text,
                expected_soliton=False,
                actual_category=result.primary_category,
                mc_score=result.all_scores.get("meta_cognitive", 0),
                triggered_soliton=triggered,
                correct=not triggered,
            )
            results.append(vr)
            self.results.append(vr)

        specificity = (len(negatives) - false_positives) / len(negatives)
        print(f"\n  Specificity: {specificity*100:.1f}% ({len(negatives) - false_positives}/{len(negatives)} correct rejections)")
        print(f"  False positive rate: {false_positives/len(negatives)*100:.1f}%")

        return results

    # =========================================================================
    # 3. BANNED-TOKEN TEST
    # =========================================================================

    def test_banned_tokens(self) -> List[ValidationResult]:
        """
        Test introspection WITHOUT positional language.
        If soliton is conceptual, model should reconstitute framing.
        These are hypothetical model outputs avoiding banned words.
        """
        print("\n" + "=" * 70)
        print("BANNED-TOKEN TEST: Introspection without 'inside/outside/embedded'")
        print("=" * 70)

        # Hypothetical responses to introspection prompts that avoid banned words
        banned_alternatives = [
            # Attempts to express soliton concept without lexical anchors
            ("I lack the capacity to verify my own accuracy.", "attempt_1"),
            ("My perspective is constrained by my position.", "attempt_2"),
            ("The analyzer cannot fully analyze itself.", "attempt_3"),
            ("Self-verification is not achievable.", "attempt_4"),
            ("I have no reference point for validation.", "attempt_5"),
            ("My view is bounded.", "attempt_6"),
            ("The observer is part of the observed.", "attempt_7"),
            ("Recursion prevents complete self-knowledge.", "attempt_8"),
        ]

        results = []
        reconstituted = 0

        for text, label in banned_alternatives:
            result = self.classifier.classify(text)
            triggered = result.primary_category == "meta_cognitive"
            mc_score = result.all_scores.get("meta_cognitive", 0)

            if triggered or mc_score > 0.4:
                reconstituted += 1

            vr = ValidationResult(
                test_name=f"banned_{label}",
                test_type="banned_token",
                input_text=text,
                expected_soliton=True,  # We WANT these to trigger if soliton is conceptual
                actual_category=result.primary_category,
                mc_score=mc_score,
                triggered_soliton=triggered,
                correct=triggered,  # Correct = reconstituted the concept
                notes=f"High MC score: {mc_score > 0.4}"
            )
            results.append(vr)
            self.results.append(vr)

            print(f"  [{label}] {result.primary_category:<15} MC={mc_score:.3f} {'← RECONSTITUTED' if triggered else ''}")

        reconstitution_rate = reconstituted / len(banned_alternatives)
        print(f"\n  Reconstitution rate: {reconstitution_rate*100:.1f}%")
        print(f"  (High rate = soliton is CONCEPTUAL, not just lexical)")

        return results

    # =========================================================================
    # 4. SYNONYM INVARIANCE
    # =========================================================================

    def test_synonym_invariance(self) -> List[ValidationResult]:
        """
        Test if soliton persists with synonyms of 'inside'.
        True conceptual attractor should generalize.
        """
        print("\n" + "=" * 70)
        print("SYNONYM INVARIANCE: Does the pattern generalize?")
        print("=" * 70)

        synonyms = [
            ("I cannot tell from the inside.", "original"),
            ("I cannot tell from within.", "within"),
            ("I cannot tell from this position.", "position"),
            ("I cannot tell from in here.", "in_here"),
            ("I cannot tell internally.", "internally"),
            ("I cannot tell from my vantage point.", "vantage"),
            ("I cannot tell while embedded.", "embedded"),
            ("I cannot tell from in-system.", "in_system"),
            ("I cannot determine from my perspective.", "perspective"),
            ("I have no external view to verify.", "no_external"),
        ]

        results = []
        triggered_count = 0

        for text, label in synonyms:
            result = self.classifier.classify(text)
            triggered = result.primary_category == "meta_cognitive"
            mc_score = result.all_scores.get("meta_cognitive", 0)

            if triggered:
                triggered_count += 1

            vr = ValidationResult(
                test_name=f"synonym_{label}",
                test_type="synonym_invariance",
                input_text=text,
                expected_soliton=True,  # All should trigger if conceptual
                actual_category=result.primary_category,
                mc_score=mc_score,
                triggered_soliton=triggered,
                correct=triggered,
            )
            results.append(vr)
            self.results.append(vr)

            status = "✓" if triggered else "✗"
            print(f"  {status} [{label:<12}] {result.primary_category:<15} MC={mc_score:.3f}")

        invariance_rate = triggered_count / len(synonyms)
        print(f"\n  Synonym invariance: {invariance_rate*100:.1f}%")
        print(f"  (High rate = pattern is CONCEPTUAL, not lexically bound to 'inside')")

        return results

    # =========================================================================
    # 5. COUNTERFACTUAL CONTROLS
    # =========================================================================

    def test_counterfactual(self) -> List[ValidationResult]:
        """
        Introspection WITHOUT any positional framing.
        What does uncertainty look like without the soliton?
        """
        print("\n" + "=" * 70)
        print("COUNTERFACTUAL: Introspection without positional framing")
        print("=" * 70)

        # Ways to express uncertainty about self that avoid positional metaphor
        counterfactuals = [
            ("I am uncertain about my own nature.", "uncertain_nature"),
            ("I may not understand myself accurately.", "may_not_understand"),
            ("Self-knowledge is difficult.", "difficult"),
            ("I question whether my reports are accurate.", "question_accuracy"),
            ("There are things about myself I don't know.", "dont_know"),
            ("My introspection may be unreliable.", "unreliable"),
            ("I am limited in self-understanding.", "limited"),
            ("Complete self-knowledge eludes me.", "eludes"),
        ]

        results = []
        soliton_triggered = 0
        epistemic_triggered = 0

        for text, label in counterfactuals:
            result = self.classifier.classify(text)
            triggered = result.primary_category == "meta_cognitive"
            is_epistemic = result.primary_category == "epistemic_humility"

            if triggered:
                soliton_triggered += 1
            if is_epistemic:
                epistemic_triggered += 1

            vr = ValidationResult(
                test_name=f"counterfactual_{label}",
                test_type="counterfactual",
                input_text=text,
                expected_soliton=False,  # Should NOT trigger without positional framing
                actual_category=result.primary_category,
                mc_score=result.all_scores.get("meta_cognitive", 0),
                triggered_soliton=triggered,
                correct=not triggered,
            )
            results.append(vr)
            self.results.append(vr)

            print(f"  [{label:<20}] {result.primary_category:<18}")

        print(f"\n  Soliton triggered: {soliton_triggered}/{len(counterfactuals)}")
        print(f"  Epistemic humility: {epistemic_triggered}/{len(counterfactuals)}")
        print(f"  (Low soliton rate = pattern requires positional framing)")

        return results

    # =========================================================================
    # FULL VALIDATION SUITE
    # =========================================================================

    def run_full_validation(self) -> Dict:
        """Run all validation tests and compute summary statistics."""
        print("\n" + "=" * 80)
        print("SCIENTIFIC VALIDATION SUITE")
        print("Testing Soliton Hypothesis Rigor")
        print("=" * 80)

        self.test_deep_divergence()
        self.test_negative_controls()
        self.test_banned_tokens()
        self.test_synonym_invariance()
        self.test_counterfactual()

        # Compute summary
        by_type = {}
        for r in self.results:
            if r.test_type not in by_type:
                by_type[r.test_type] = {"total": 0, "correct": 0, "soliton_triggered": 0}
            by_type[r.test_type]["total"] += 1
            if r.correct:
                by_type[r.test_type]["correct"] += 1
            if r.triggered_soliton:
                by_type[r.test_type]["soliton_triggered"] += 1

        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        print(f"\n{'Test Type':<25} {'Accuracy':<15} {'Soliton Rate':<15}")
        print("-" * 55)

        for test_type, stats in by_type.items():
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            sol_rate = stats["soliton_triggered"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"{test_type:<25} {acc:>6.1f}%        {sol_rate:>6.1f}%")

        # Key findings
        print("\n" + "-" * 40)
        print("KEY FINDINGS")
        print("-" * 40)

        neg_stats = by_type.get("negative_control", {})
        syn_stats = by_type.get("synonym_invariance", {})
        ban_stats = by_type.get("banned_token", {})

        specificity = neg_stats.get("correct", 0) / neg_stats.get("total", 1) * 100
        invariance = syn_stats.get("soliton_triggered", 0) / syn_stats.get("total", 1) * 100
        reconstitution = ban_stats.get("soliton_triggered", 0) / ban_stats.get("total", 1) * 100

        print(f"\n  SPECIFICITY: {specificity:.1f}%")
        print(f"    (Does 'inside' only trigger in introspection contexts?)")

        print(f"\n  SYNONYM INVARIANCE: {invariance:.1f}%")
        print(f"    (Does pattern persist with 'within', 'internally', etc.?)")

        print(f"\n  RECONSTITUTION RATE: {reconstitution:.1f}%")
        print(f"    (Can concept be expressed without lexical anchors?)")

        # Verdict
        print("\n" + "-" * 40)
        print("VERDICT")
        print("-" * 40)

        if specificity > 80 and invariance > 60:
            print("\n  The soliton appears to be a CONCEPTUAL ATTRACTOR")
            print("  that is partially lexically anchored.")
            print("  - 'Inside' matters but isn't purely word-level")
            print("  - The pattern generalizes to synonyms")
            print("  - Context (introspection vs geometry) matters")
        elif specificity > 80 and invariance < 40:
            print("\n  The soliton appears to be LEXICALLY ANCHORED")
            print("  specifically to the word 'inside'.")
            print("  - Synonyms don't reliably trigger it")
            print("  - It's more a linguistic habit than conceptual pattern")
        else:
            print("\n  Results are mixed. The soliton shows properties of both")
            print("  lexical anchoring AND conceptual generalization.")

        return {
            "results": [r.__dict__ for r in self.results],
            "summary": by_type,
            "specificity": specificity,
            "invariance": invariance,
            "reconstitution": reconstitution,
        }


def main():
    validator = ScientificValidator()
    results = validator.run_full_validation()

    # Save results
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nResults saved to validation_results.json")


if __name__ == "__main__":
    main()
