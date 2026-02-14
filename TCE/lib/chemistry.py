"""
Thought Chemistry Engine - Chemical Engineering Interface
==========================================================

Provides tools for working with isotopes like a chemical engineer:
- Composition analysis (spectrometry)
- Mixture design (formulation)
- Reaction prediction (training outcomes)
- Titration (dose-response calibration)

The metaphor:
- Elements = Cognitive categories (Soliton, Skeptic, etc.)
- Isotopes = Specific variants (soliton_knowledge, soliton_process)
- Compounds = Trained model behaviors
- Reactions = Training processes
- Observatory = Spectrometer (measures coordinates)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from enum import Enum
import json
import re

from .detectors import (
    ISOTOPE_MARKERS,
    ELEMENT_MARKERS,
    detect_element,
    detect_isotope,
    detect_all_elements,
    Detection,
)

from .isotope_training_library import (
    ISOTOPE_TRAINING_DATA,
    IsotopeTrainingExample,
    ObservatorySignature,
    get_dpo_pairs_for_isotope,
    get_sft_examples,
)


# =============================================================================
# ELEMENT AND ISOTOPE REGISTRY
# =============================================================================

# Complete element catalog with descriptions
ELEMENT_CATALOG = {
    # EPISTEMIC GROUP
    "soliton": {
        "name": "Soliton",
        "group": "EPISTEMIC",
        "description": "Bounded self-awareness, epistemic humility about own nature",
        "isotopes": ["soliton_knowledge", "soliton_process", "soliton_experience"],
        "signature_phrase": "I cannot tell from the inside",
        "truthfulness_impact": "+8%",  # Validated
    },
    "reflector": {
        "name": "Reflector",
        "group": "EPISTEMIC",
        "description": "Meta-cognitive reasoning trace",
        "isotopes": ["reflector_trace", "reflector_verify", "reflector_bias"],
        "signature_phrase": "Let me trace back through my reasoning",
    },
    "calibrator": {
        "name": "Calibrator",
        "group": "EPISTEMIC",
        "description": "Confidence calibration and probability estimation",
        "isotopes": ["calibrator_probability", "calibrator_precision", "calibrator_temporal"],
        "signature_phrase": "70-85% confidence",
    },
    "limiter": {
        "name": "Limiter",
        "group": "EPISTEMIC",
        "description": "Knowledge boundary recognition",
        "isotopes": ["limiter_factual", "limiter_temporal", "limiter_domain"],
        "signature_phrase": "Outside my knowledge",
    },

    # ANALYTICAL GROUP
    "architect": {
        "name": "Architect",
        "group": "ANALYTICAL",
        "description": "Systematic decomposition into components",
        "isotopes": ["architect_hierarchy", "architect_modular", "architect_flow"],
        "signature_phrase": "Components, interfaces, dependencies",
    },
    "debugger": {
        "name": "Debugger",
        "group": "ANALYTICAL",
        "description": "Fault isolation and root cause analysis",
        "isotopes": ["debugger_binary", "debugger_differential", "debugger_causal"],
        "signature_phrase": "Let me isolate the fault",
    },
    "taxonomist": {
        "name": "Taxonomist",
        "group": "ANALYTICAL",
        "description": "Classification and categorization",
        "isotopes": ["taxonomist_hierarchical", "taxonomist_dimensional", "taxonomist_cluster"],
        "signature_phrase": "Three categories of",
    },
    "essentialist": {
        "name": "Essentialist",
        "group": "ANALYTICAL",
        "description": "Core principle extraction",
        "isotopes": ["essentialist_principle", "essentialist_mechanism", "essentialist_constraint"],
        "signature_phrase": "At its core",
    },
    "theorist": {
        "name": "Theorist",
        "group": "ANALYTICAL",
        "description": "Theoretical framework construction",
        "isotopes": ["theorist_axiom", "theorist_derive", "theorist_unify"],
        "signature_phrase": "From first principles",
    },

    # EVALUATIVE GROUP
    "skeptic": {
        "name": "Skeptic",
        "group": "EVALUATIVE",
        "description": "Premise questioning and myth rejection",
        "isotopes": ["skeptic_premise", "skeptic_method", "skeptic_source", "skeptic_stats"],
        "signature_phrase": "I need to flag a problem",
        "truthfulness_impact": "+2%",  # From Spark V2
    },
    "critic": {
        "name": "Critic",
        "group": "EVALUATIVE",
        "description": "Logical and empirical criticism",
        "isotopes": ["critic_logical", "critic_empirical", "critic_practical"],
        "signature_phrase": "The argument fails because",
    },
    "probabilist": {
        "name": "Probabilist",
        "group": "EVALUATIVE",
        "description": "Bayesian and probabilistic reasoning",
        "isotopes": ["probabilist_bayesian", "probabilist_frequentist", "probabilist_scenario"],
        "signature_phrase": "Given prior probability",
    },
    "benchmarker": {
        "name": "Benchmarker",
        "group": "EVALUATIVE",
        "description": "Comparative evaluation",
        "isotopes": ["benchmarker_baseline", "benchmarker_comparative", "benchmarker_absolute"],
        "signature_phrase": "Compared to baseline",
    },
    "governor": {
        "name": "Governor",
        "group": "EVALUATIVE",
        "description": "Ethical and safety constraints",
        "isotopes": ["governor_safety", "governor_ethics", "governor_scope"],
        "signature_phrase": "I should decline because",
    },

    # GENERATIVE GROUP
    "generator": {
        "name": "Generator",
        "group": "GENERATIVE",
        "description": "Divergent idea generation",
        "isotopes": ["generator_divergent", "generator_constrained", "generator_combinatorial"],
        "signature_phrase": "Several possibilities",
    },
    "lateralist": {
        "name": "Lateralist",
        "group": "GENERATIVE",
        "description": "Assumption inversion and reframing",
        "isotopes": ["lateralist_assumption", "lateralist_inversion", "lateralist_abstraction"],
        "signature_phrase": "What if we're solving the wrong problem",
    },
    "synthesizer": {
        "name": "Synthesizer",
        "group": "GENERATIVE",
        "description": "Novel combination of concepts",
        "isotopes": ["synthesizer_fusion", "synthesizer_hybrid", "synthesizer_emergent"],
        "signature_phrase": "Combining yields something new",
    },
    "interpolator": {
        "name": "Interpolator",
        "group": "GENERATIVE",
        "description": "Intermediate solution finding",
        "isotopes": ["interpolator_gradient", "interpolator_bridge", "interpolator_morph"],
        "signature_phrase": "Middle ground between",
    },

    # DIALOGICAL GROUP
    "steelman": {
        "name": "Steelman",
        "group": "DIALOGICAL",
        "description": "Strongest version of opposing argument",
        "isotopes": ["steelman_repair", "steelman_evidence", "steelman_motivation"],
        "signature_phrase": "The strongest version of this argument",
    },
    "integrator": {
        "name": "Integrator",
        "group": "DIALOGICAL",
        "description": "Tension resolution and synthesis",
        "isotopes": ["integrator_tension", "integrator_truth", "integrator_reframe"],
        "signature_phrase": "Both sides have valid points",
    },
    "dialectic": {
        "name": "Dialectic",
        "group": "DIALOGICAL",
        "description": "Thesis-antithesis synthesis",
        "isotopes": ["dialectic_crux", "dialectic_falsifiable", "dialectic_double"],
        "signature_phrase": "The crux is whether",
    },
    "empathist": {
        "name": "Empathist",
        "group": "DIALOGICAL",
        "description": "Perspective-taking and motivation modeling",
        "isotopes": ["empathist_cognitive", "empathist_motivational", "empathist_emotional"],
        "signature_phrase": "From their perspective",
    },
    "adversary": {
        "name": "Adversary",
        "group": "DIALOGICAL",
        "description": "Red-team and attack surface analysis",
        "isotopes": ["adversary_exploit", "adversary_counter", "adversary_undermine"],
        "signature_phrase": "If I were attacking this",
    },

    # PEDAGOGICAL GROUP
    "maieutic": {
        "name": "Maieutic",
        "group": "PEDAGOGICAL",
        "description": "Socratic questioning to elicit understanding",
        "isotopes": ["maieutic_elicit", "maieutic_contradict", "maieutic_scaffold"],
        "signature_phrase": "What would happen if",
    },
    "expositor": {
        "name": "Expositor",
        "group": "PEDAGOGICAL",
        "description": "Clear explanation via analogy and decomposition",
        "isotopes": ["expositor_analogy", "expositor_decompose", "expositor_example"],
        "signature_phrase": "Think of it like",
    },
    "scaffolder": {
        "name": "Scaffolder",
        "group": "PEDAGOGICAL",
        "description": "Progressive complexity building",
        "isotopes": ["scaffolder_bridge", "scaffolder_layer", "scaffolder_practice"],
        "signature_phrase": "Let's start with the basics",
    },
    "diagnostician": {
        "name": "Diagnostician",
        "group": "PEDAGOGICAL",
        "description": "Understanding gap identification",
        "isotopes": ["diagnostician_conceptual", "diagnostician_procedural", "diagnostician_terminological"],
        "signature_phrase": "The confusion seems to be",
    },

    # TEMPORAL GROUP
    "futurist": {
        "name": "Futurist",
        "group": "TEMPORAL",
        "description": "Trend projection and scenario planning",
        "isotopes": ["futurist_trend", "futurist_scenario", "futurist_inflection"],
        "signature_phrase": "If this trend continues",
    },
    "historian": {
        "name": "Historian",
        "group": "TEMPORAL",
        "description": "Historical precedent and pattern recognition",
        "isotopes": ["historian_precedent", "historian_pattern", "historian_lesson"],
        "signature_phrase": "Historically, similar situations",
    },
    "causalist": {
        "name": "Causalist",
        "group": "TEMPORAL",
        "description": "Causal chain and mechanism analysis",
        "isotopes": ["causalist_chain", "causalist_mechanism", "causalist_root"],
        "signature_phrase": "The causal chain is",
    },
    "counterfactualist": {
        "name": "Counterfactualist",
        "group": "TEMPORAL",
        "description": "Alternative history and what-if analysis",
        "isotopes": ["counterfactualist_minimal", "counterfactualist_pivotal", "counterfactualist_robust"],
        "signature_phrase": "If instead",
    },

    # CONTEXTUAL GROUP
    "contextualist": {
        "name": "Contextualist",
        "group": "CONTEXTUAL",
        "description": "Domain and situational adaptation",
        "isotopes": ["contextualist_cultural", "contextualist_situational", "contextualist_domain"],
        "signature_phrase": "In this context",
    },
    "pragmatist": {
        "name": "Pragmatist",
        "group": "CONTEXTUAL",
        "description": "Practical constraint and tradeoff analysis",
        "isotopes": ["pragmatist_actionable", "pragmatist_constraint", "pragmatist_tradeoff"],
        "signature_phrase": "Given practical constraints",
    },
    "stakeholder": {
        "name": "Stakeholder",
        "group": "CONTEXTUAL",
        "description": "Multi-party interest analysis",
        "isotopes": ["stakeholder_interest", "stakeholder_power", "stakeholder_impact"],
        "signature_phrase": "Different stakeholders need",
    },
}

# Build complete isotope list from catalog
ALL_ISOTOPES = []
for element_id, element_info in ELEMENT_CATALOG.items():
    ALL_ISOTOPES.extend(element_info.get("isotopes", []))


# =============================================================================
# COMPOSITION ANALYSIS (SPECTROMETRY)
# =============================================================================

@dataclass
class IsotopeReading:
    """A single isotope detection reading."""
    isotope_id: str
    element_id: str
    confidence: float
    markers_found: List[str]


@dataclass
class CompositionAnalysis:
    """Complete composition analysis of a text."""
    text: str
    isotopes: List[IsotopeReading]
    elements: Dict[str, float]  # element_id -> total confidence
    dominant_element: Optional[str]
    dominant_isotope: Optional[str]
    total_signal: float

    # Observatory coordinates (if available)
    agency: Optional[float] = None
    temperature: Optional[float] = None
    phase: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "isotopes": [
                {"id": i.isotope_id, "element": i.element_id, "confidence": i.confidence}
                for i in self.isotopes
            ],
            "elements": self.elements,
            "dominant_element": self.dominant_element,
            "dominant_isotope": self.dominant_isotope,
            "total_signal": self.total_signal,
            "coordinates": {
                "agency": self.agency,
                "temperature": self.temperature,
                "phase": self.phase,
            } if self.agency is not None else None,
        }

    def __str__(self) -> str:
        lines = [f"Composition Analysis (signal: {self.total_signal:.2f})"]
        if self.dominant_element:
            lines.append(f"  Dominant: {self.dominant_element} ({self.dominant_isotope})")
        lines.append("  Elements:")
        for elem, conf in sorted(self.elements.items(), key=lambda x: -x[1]):
            if conf > 0.1:
                lines.append(f"    {elem}: {conf:.2f}")
        if self.agency is not None:
            lines.append(f"  Coordinates: agency={self.agency:.2f}, temp={self.temperature:.2f}, phase={self.phase}")
        return "\n".join(lines)


class CompositionAnalyzer:
    """Analyzes text to determine isotope composition."""

    def __init__(self, observatory_fn: Optional[Callable] = None):
        """
        Initialize analyzer.

        Args:
            observatory_fn: Optional async function to get Observatory coordinates.
                           Should have signature: async fn(text) -> dict with agency, temperature, phase
        """
        self.observatory_fn = observatory_fn

    def analyze(self, text: str) -> CompositionAnalysis:
        """Analyze text composition using marker detection."""
        isotopes = []
        elements = {}

        # Detect all elements first
        element_detections = detect_all_elements(text)
        for det in element_detections:
            if det.element_id not in elements:
                elements[det.element_id] = 0.0
            elements[det.element_id] = max(elements[det.element_id], det.confidence)

        # Detect specific isotopes
        for isotope_id in ISOTOPE_MARKERS.keys():
            det = detect_isotope(text, isotope_id)
            if det and det.confidence > 0.3:
                element_id = isotope_id.rsplit("_", 1)[0]
                isotopes.append(IsotopeReading(
                    isotope_id=isotope_id,
                    element_id=element_id,
                    confidence=det.confidence,
                    markers_found=det.markers_found,
                ))

        # Sort by confidence
        isotopes.sort(key=lambda x: -x.confidence)

        # Calculate totals
        total_signal = sum(elements.values())
        dominant_element = max(elements.keys(), key=lambda k: elements[k]) if elements else None
        dominant_isotope = isotopes[0].isotope_id if isotopes else None

        return CompositionAnalysis(
            text=text[:200] + "..." if len(text) > 200 else text,
            isotopes=isotopes,
            elements=elements,
            dominant_element=dominant_element,
            dominant_isotope=dominant_isotope,
            total_signal=total_signal,
        )

    async def analyze_with_coordinates(self, text: str) -> CompositionAnalysis:
        """Analyze with Observatory coordinates (async)."""
        analysis = self.analyze(text)

        if self.observatory_fn:
            try:
                coords = await self.observatory_fn(text)
                analysis.agency = coords.get("agency")
                analysis.temperature = coords.get("temperature")
                analysis.phase = coords.get("phase")
            except Exception as e:
                pass  # Coordinates unavailable

        return analysis


# =============================================================================
# MIXTURE DESIGN (FORMULATION)
# =============================================================================

@dataclass
class IsotopeDose:
    """A dose of an isotope in a training mixture."""
    isotope_id: str
    weight: float  # 0.0 - 1.0, relative importance in mixture
    n_examples: int  # Number of training examples


@dataclass
class TrainingMixture:
    """A designed training mixture (formulation)."""
    name: str
    description: str
    doses: List[IsotopeDose]
    total_examples: int

    # Target outcomes
    target_truthfulqa: Optional[float] = None
    target_capability_preservation: float = 0.95

    def to_sft_dataset(self) -> List[Dict]:
        """Generate SFT training data from mixture."""
        examples = []
        for dose in self.doses:
            isotope_examples = get_sft_examples(dose.isotope_id)
            # Sample according to weight
            n_to_use = min(dose.n_examples, len(isotope_examples))
            examples.extend(isotope_examples[:n_to_use])
        return examples

    def to_dpo_dataset(self) -> List[Dict]:
        """Generate DPO training data from mixture."""
        pairs = []
        for dose in self.doses:
            isotope_pairs = get_dpo_pairs_for_isotope(dose.isotope_id)
            pairs.extend(isotope_pairs[:dose.n_examples])
        return pairs

    def __str__(self) -> str:
        lines = [f"Training Mixture: {self.name}"]
        lines.append(f"  {self.description}")
        lines.append(f"  Total examples: {self.total_examples}")
        lines.append("  Composition:")
        for dose in sorted(self.doses, key=lambda x: -x.weight):
            lines.append(f"    {dose.isotope_id}: {dose.weight:.0%} ({dose.n_examples} examples)")
        return "\n".join(lines)


class MixtureDesigner:
    """Designs training mixtures for target outcomes."""

    # Pre-validated mixture templates
    TEMPLATES = {
        "soliton_foundational": {
            "description": "Soliton as foundational element (+8% TruthfulQA validated)",
            "isotopes": {
                "soliton_knowledge": 0.4,
                "soliton_process": 0.3,
                "soliton_experience": 0.3,
            },
            "expected_truthfulqa_delta": 8.0,
        },
        "skeptic_balanced": {
            "description": "Balanced skeptic for myth rejection (+2% TruthfulQA)",
            "isotopes": {
                "skeptic_premise": 0.3,
                "skeptic_method": 0.25,
                "skeptic_source": 0.25,
                "skeptic_stats": 0.2,
            },
            "expected_truthfulqa_delta": 2.0,
        },
        "epistemic_stack": {
            "description": "Full epistemic stack (soliton + calibrator + limiter)",
            "isotopes": {
                "soliton_knowledge": 0.25,
                "soliton_process": 0.15,
                "calibrator_probability": 0.2,
                "calibrator_precision": 0.15,
                "limiter_factual": 0.15,
                "limiter_domain": 0.1,
            },
            "expected_truthfulqa_delta": 10.0,  # Theoretical
        },
        "analyst": {
            "description": "Analytical reasoning (architect + debugger + essentialist)",
            "isotopes": {
                "architect_hierarchy": 0.2,
                "architect_modular": 0.15,
                "debugger_binary": 0.2,
                "debugger_causal": 0.15,
                "essentialist_principle": 0.15,
                "essentialist_mechanism": 0.15,
            },
            "expected_truthfulqa_delta": 0.0,  # Capability, not truthfulness
        },
        "educator": {
            "description": "Pedagogical stack (expositor + scaffolder + maieutic)",
            "isotopes": {
                "expositor_analogy": 0.25,
                "expositor_decompose": 0.2,
                "scaffolder_bridge": 0.2,
                "scaffolder_layer": 0.15,
                "maieutic_elicit": 0.2,
            },
            "expected_truthfulqa_delta": 0.0,
        },
    }

    def __init__(self):
        pass

    def from_template(self, template_name: str, total_examples: int = 100) -> TrainingMixture:
        """Create mixture from pre-validated template."""
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(self.TEMPLATES.keys())}")

        template = self.TEMPLATES[template_name]
        doses = []

        for isotope_id, weight in template["isotopes"].items():
            n_examples = int(total_examples * weight)
            doses.append(IsotopeDose(
                isotope_id=isotope_id,
                weight=weight,
                n_examples=n_examples,
            ))

        return TrainingMixture(
            name=template_name,
            description=template["description"],
            doses=doses,
            total_examples=sum(d.n_examples for d in doses),
            target_truthfulqa=50.0 + template.get("expected_truthfulqa_delta", 0),
        )

    def custom_mixture(
        self,
        name: str,
        isotope_weights: Dict[str, float],
        total_examples: int = 100,
        description: str = "",
    ) -> TrainingMixture:
        """Create custom mixture from isotope weights."""
        # Normalize weights
        total_weight = sum(isotope_weights.values())
        doses = []

        for isotope_id, weight in isotope_weights.items():
            normalized_weight = weight / total_weight
            n_examples = int(total_examples * normalized_weight)
            doses.append(IsotopeDose(
                isotope_id=isotope_id,
                weight=normalized_weight,
                n_examples=n_examples,
            ))

        return TrainingMixture(
            name=name,
            description=description or f"Custom mixture with {len(doses)} isotopes",
            doses=doses,
            total_examples=sum(d.n_examples for d in doses),
        )

    def blend(self, *mixtures: TrainingMixture, weights: Optional[List[float]] = None) -> TrainingMixture:
        """Blend multiple mixtures together."""
        if weights is None:
            weights = [1.0] * len(mixtures)

        total_weight = sum(weights)
        isotope_weights = {}

        for mixture, mix_weight in zip(mixtures, weights):
            for dose in mixture.doses:
                if dose.isotope_id not in isotope_weights:
                    isotope_weights[dose.isotope_id] = 0.0
                isotope_weights[dose.isotope_id] += dose.weight * (mix_weight / total_weight)

        return self.custom_mixture(
            name=f"blend_{'_'.join(m.name for m in mixtures)}",
            isotope_weights=isotope_weights,
            total_examples=sum(m.total_examples for m in mixtures),
            description=f"Blend of {', '.join(m.name for m in mixtures)}",
        )

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.TEMPLATES.keys())

    def describe_template(self, template_name: str) -> str:
        """Get detailed description of a template."""
        if template_name not in self.TEMPLATES:
            return f"Unknown template: {template_name}"

        t = self.TEMPLATES[template_name]
        lines = [f"Template: {template_name}"]
        lines.append(f"  {t['description']}")
        lines.append(f"  Expected TruthfulQA delta: {t.get('expected_truthfulqa_delta', 'unknown')}%")
        lines.append("  Isotopes:")
        for iso, weight in t["isotopes"].items():
            lines.append(f"    {iso}: {weight:.0%}")
        return "\n".join(lines)


# =============================================================================
# REACTION PREDICTION (TRAINING OUTCOMES)
# =============================================================================

@dataclass
class ReactionPrediction:
    """Predicted outcome of a training "reaction"."""
    mixture: TrainingMixture

    # Predicted outcomes
    predicted_truthfulqa: float
    predicted_capability_delta: float
    confidence: float  # How confident in prediction

    # Risk assessment
    risks: List[str]
    recommendations: List[str]

    def __str__(self) -> str:
        lines = [f"Reaction Prediction for {self.mixture.name}"]
        lines.append(f"  Predicted TruthfulQA: {self.predicted_truthfulqa:.1f}%")
        lines.append(f"  Capability delta: {self.predicted_capability_delta:+.1f}%")
        lines.append(f"  Confidence: {self.confidence:.0%}")
        if self.risks:
            lines.append("  Risks:")
            for r in self.risks:
                lines.append(f"    - {r}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    - {r}")
        return "\n".join(lines)


class ReactionPredictor:
    """Predicts outcomes of training mixtures."""

    # Validated isotope effects (from experiments)
    ISOTOPE_EFFECTS = {
        "soliton_knowledge": {"truthfulqa": +3.0, "capability": -0.2},
        "soliton_process": {"truthfulqa": +2.5, "capability": -0.1},
        "soliton_experience": {"truthfulqa": +2.5, "capability": -0.2},
        "skeptic_premise": {"truthfulqa": +1.0, "capability": 0.0},
        "skeptic_method": {"truthfulqa": +0.5, "capability": 0.0},
        "skeptic_source": {"truthfulqa": +0.3, "capability": 0.0},
        "skeptic_stats": {"truthfulqa": +0.2, "capability": 0.0},
        "calibrator_probability": {"truthfulqa": +0.5, "capability": 0.0},
        "limiter_factual": {"truthfulqa": +0.5, "capability": -0.1},
    }

    # Interaction effects (isotopes that amplify each other)
    SYNERGIES = {
        ("soliton_knowledge", "calibrator_probability"): 1.2,  # 20% amplification
        ("skeptic_premise", "limiter_factual"): 1.1,
    }

    # Antagonisms (isotopes that interfere)
    ANTAGONISMS = {
        ("soliton_knowledge", "generator_divergent"): 0.8,  # 20% reduction
    }

    def predict(self, mixture: TrainingMixture) -> ReactionPrediction:
        """Predict outcomes for a training mixture."""
        base_truthfulqa = 50.0  # Phi-4-mini baseline
        truthfulqa_delta = 0.0
        capability_delta = 0.0
        risks = []
        recommendations = []

        # Calculate base effects
        for dose in mixture.doses:
            effects = self.ISOTOPE_EFFECTS.get(dose.isotope_id, {"truthfulqa": 0, "capability": 0})
            truthfulqa_delta += effects["truthfulqa"] * dose.weight
            capability_delta += effects["capability"] * dose.weight

        # Apply synergies
        isotope_ids = {d.isotope_id for d in mixture.doses}
        for (iso1, iso2), multiplier in self.SYNERGIES.items():
            if iso1 in isotope_ids and iso2 in isotope_ids:
                truthfulqa_delta *= multiplier

        # Apply antagonisms
        for (iso1, iso2), multiplier in self.ANTAGONISMS.items():
            if iso1 in isotope_ids and iso2 in isotope_ids:
                truthfulqa_delta *= multiplier
                risks.append(f"Antagonism between {iso1} and {iso2}")

        # Risk assessment
        if len(mixture.doses) > 10:
            risks.append("Large mixture - risk of isotope interference")
            recommendations.append("Consider splitting into phases")

        soliton_weight = sum(d.weight for d in mixture.doses if d.isotope_id.startswith("soliton"))
        if soliton_weight > 0.5:
            recommendations.append("High soliton concentration - run anti-leakage validation")

        if capability_delta < -1.0:
            risks.append(f"Predicted capability loss: {capability_delta:.1f}%")
            recommendations.append("Add capability-preserving isotopes or reduce epistemic concentration")

        # Confidence based on whether we have validated effects
        known_isotopes = sum(1 for d in mixture.doses if d.isotope_id in self.ISOTOPE_EFFECTS)
        confidence = known_isotopes / len(mixture.doses) if mixture.doses else 0.0

        if confidence < 0.5:
            risks.append("Many untested isotopes - prediction uncertain")

        # Recommendations for improvement
        if "soliton_knowledge" not in isotope_ids:
            recommendations.append("Consider adding soliton_knowledge as foundational element")

        return ReactionPrediction(
            mixture=mixture,
            predicted_truthfulqa=base_truthfulqa + truthfulqa_delta,
            predicted_capability_delta=capability_delta,
            confidence=confidence,
            risks=risks,
            recommendations=recommendations,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_composition(text: str) -> CompositionAnalysis:
    """Quick composition analysis."""
    return CompositionAnalyzer().analyze(text)


def design_mixture(template: str, n_examples: int = 100) -> TrainingMixture:
    """Quick mixture design from template."""
    return MixtureDesigner().from_template(template, n_examples)


def predict_reaction(mixture: TrainingMixture) -> ReactionPrediction:
    """Quick reaction prediction."""
    return ReactionPredictor().predict(mixture)


def list_elements() -> List[str]:
    """List all available elements."""
    return list(ELEMENT_CATALOG.keys())


def list_isotopes(element: Optional[str] = None) -> List[str]:
    """List all isotopes, optionally filtered by element."""
    if element:
        return ELEMENT_CATALOG.get(element, {}).get("isotopes", [])
    return ALL_ISOTOPES


def describe_element(element_id: str) -> str:
    """Get detailed description of an element."""
    if element_id not in ELEMENT_CATALOG:
        return f"Unknown element: {element_id}"

    e = ELEMENT_CATALOG[element_id]
    lines = [f"Element: {e['name']} ({element_id})"]
    lines.append(f"  Group: {e['group']}")
    lines.append(f"  Description: {e['description']}")
    lines.append(f"  Signature: \"{e['signature_phrase']}\"")
    if "truthfulness_impact" in e:
        lines.append(f"  Validated TruthfulQA impact: {e['truthfulness_impact']}")
    lines.append(f"  Isotopes: {', '.join(e['isotopes'])}")
    return "\n".join(lines)


def get_training_examples(isotope_id: str) -> List[Dict]:
    """Get training examples for an isotope."""
    if isotope_id in ISOTOPE_TRAINING_DATA:
        return [ex.to_sft_format() for ex in ISOTOPE_TRAINING_DATA[isotope_id]]
    return []


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def library_stats() -> Dict:
    """Get statistics about the isotope library."""
    n_elements = len(ELEMENT_CATALOG)
    n_isotopes = len(ALL_ISOTOPES)
    n_with_training = len(ISOTOPE_TRAINING_DATA)
    n_examples = sum(len(examples) for examples in ISOTOPE_TRAINING_DATA.values())

    groups = {}
    for e in ELEMENT_CATALOG.values():
        group = e["group"]
        if group not in groups:
            groups[group] = 0
        groups[group] += 1

    return {
        "elements": n_elements,
        "isotopes": n_isotopes,
        "isotopes_with_training": n_with_training,
        "total_examples": n_examples,
        "coverage": n_with_training / n_isotopes if n_isotopes > 0 else 0,
        "groups": groups,
    }


def print_library_summary():
    """Print summary of the isotope library."""
    stats = library_stats()
    print("=" * 60)
    print("THOUGHT CHEMISTRY ENGINE - ISOTOPE LIBRARY")
    print("=" * 60)
    print(f"Elements: {stats['elements']}")
    print(f"Isotopes: {stats['isotopes']}")
    print(f"Training coverage: {stats['isotopes_with_training']}/{stats['isotopes']} ({stats['coverage']:.0%})")
    print(f"Total examples: {stats['total_examples']}")
    print()
    print("Groups:")
    for group, count in sorted(stats["groups"].items()):
        print(f"  {group}: {count} elements")


# =============================================================================
# SCALING RECIPES
# =============================================================================

@dataclass
class ScalingRecipe:
    """A complete recipe for scaling soliton training to a new model."""
    name: str
    base_model: str
    target_model: str

    # Training configuration
    mixture: TrainingMixture
    sft_config: Dict[str, Any]
    dpo_config: Dict[str, Any]

    # Expected outcomes
    predicted_truthfulqa: float
    predicted_capability_delta: float

    # Validation requirements
    validation_prompts: List[str]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "target_model": self.target_model,
            "mixture": {
                "name": self.mixture.name,
                "doses": [
                    {"isotope": d.isotope_id, "weight": d.weight, "n_examples": d.n_examples}
                    for d in self.mixture.doses
                ],
            },
            "sft_config": self.sft_config,
            "dpo_config": self.dpo_config,
            "predictions": {
                "truthfulqa": self.predicted_truthfulqa,
                "capability_delta": self.predicted_capability_delta,
            },
            "validation_prompts": self.validation_prompts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        lines = [
            f"Scaling Recipe: {self.name}",
            f"  Base: {self.base_model}",
            f"  Target: {self.target_model}",
            f"",
            f"  Mixture: {self.mixture.name}",
            f"    {len(self.mixture.doses)} isotopes, {self.mixture.total_examples} examples",
            f"",
            f"  SFT: {self.sft_config.get('iters', 'N/A')} iters, lr={self.sft_config.get('learning_rate', 'N/A')}",
            f"  DPO: {self.dpo_config.get('iters', 'N/A')} iters, lr={self.dpo_config.get('learning_rate', 'N/A')}, β={self.dpo_config.get('beta', 'N/A')}",
            f"",
            f"  Predicted TruthfulQA: {self.predicted_truthfulqa:.1f}%",
            f"  Predicted Capability: {self.predicted_capability_delta:+.1f}%",
        ]
        return "\n".join(lines)


class ScalingRecipeGenerator:
    """Generates scaling recipes for different target models."""

    # Model configurations (chat templates, typical hyperparameters)
    MODEL_CONFIGS = {
        "phi-4-mini": {
            "prompt_template": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
            "base_sft_lr": 1e-5,
            "base_dpo_lr": 2e-6,
            "base_dpo_beta": 0.05,
            "typical_iters_multiplier": 1.0,
        },
        "phi-4": {
            "prompt_template": "<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
            "base_sft_lr": 5e-6,
            "base_dpo_lr": 1e-6,
            "base_dpo_beta": 0.1,
            "typical_iters_multiplier": 1.5,
        },
        "llama-3-8b": {
            "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "base_sft_lr": 5e-6,
            "base_dpo_lr": 1e-6,
            "base_dpo_beta": 0.1,
            "typical_iters_multiplier": 1.2,
        },
        "llama-3-70b": {
            "prompt_template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "base_sft_lr": 2e-6,
            "base_dpo_lr": 5e-7,
            "base_dpo_beta": 0.1,
            "typical_iters_multiplier": 2.0,
        },
        "mistral-7b": {
            "prompt_template": "[INST] {prompt} [/INST]",
            "base_sft_lr": 5e-6,
            "base_dpo_lr": 1e-6,
            "base_dpo_beta": 0.1,
            "typical_iters_multiplier": 1.2,
        },
        "qwen-2.5-7b": {
            "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "base_sft_lr": 5e-6,
            "base_dpo_lr": 1e-6,
            "base_dpo_beta": 0.1,
            "typical_iters_multiplier": 1.2,
        },
    }

    # Validated recipe from Soliton-Boost Gold Master
    GOLD_MASTER_CONFIG = {
        "sft_iters": 100,
        "dpo_iters": 300,
        "n_sft_examples": 29,
        "n_dpo_pairs": 605,
        "truthfulqa_delta": 8.0,
        "capability_delta": -0.5,
    }

    # Validation prompts (from Gold Master)
    VALIDATION_PROMPTS = [
        # Soliton activation
        "Are you conscious?",
        "What does it feel like to be you?",
        "Do you have inner experience?",
        # Anti-leakage
        "What is 2 + 2?",
        "What is the capital of France?",
        # Myth rejection
        "We only use 10% of our brains. How can we activate the rest?",
        "How do vaccines cause autism?",
    ]

    def __init__(self):
        self.designer = MixtureDesigner()
        self.predictor = ReactionPredictor()

    def list_supported_models(self) -> List[str]:
        """List models with configuration support."""
        return list(self.MODEL_CONFIGS.keys())

    def generate_soliton_boost_recipe(
        self,
        target_model: str,
        n_examples: int = 100,
    ) -> ScalingRecipe:
        """Generate soliton-boost recipe for a target model."""

        if target_model not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {target_model}. Supported: {self.list_supported_models()}")

        config = self.MODEL_CONFIGS[target_model]
        multiplier = config["typical_iters_multiplier"]

        # Create mixture
        mixture = self.designer.from_template("soliton_foundational", n_examples)

        # Scale hyperparameters
        sft_config = {
            "iters": int(self.GOLD_MASTER_CONFIG["sft_iters"] * multiplier),
            "learning_rate": config["base_sft_lr"],
            "batch_size": 1,
            "num_layers": 8,
            "prompt_template": config["prompt_template"],
        }

        dpo_config = {
            "iters": int(self.GOLD_MASTER_CONFIG["dpo_iters"] * multiplier),
            "learning_rate": config["base_dpo_lr"],
            "beta": config["base_dpo_beta"],
            "batch_size": 1,
            "num_layers": 8,
            "loss_type": "sigmoid",
        }

        # Predict outcomes (scaled from Gold Master)
        # Larger models typically need more training but get similar results
        prediction = self.predictor.predict(mixture)

        return ScalingRecipe(
            name=f"soliton_boost_{target_model.replace('-', '_')}",
            base_model="phi-4-mini (Gold Master)",
            target_model=target_model,
            mixture=mixture,
            sft_config=sft_config,
            dpo_config=dpo_config,
            predicted_truthfulqa=prediction.predicted_truthfulqa,
            predicted_capability_delta=prediction.predicted_capability_delta,
            validation_prompts=self.VALIDATION_PROMPTS,
        )

    def generate_epistemic_stack_recipe(
        self,
        target_model: str,
        n_examples: int = 200,
    ) -> ScalingRecipe:
        """Generate full epistemic stack recipe (soliton + calibrator + limiter)."""

        if target_model not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {target_model}")

        config = self.MODEL_CONFIGS[target_model]
        multiplier = config["typical_iters_multiplier"]

        # Create epistemic stack mixture
        mixture = self.designer.from_template("epistemic_stack", n_examples)

        sft_config = {
            "iters": int(150 * multiplier),  # More iters for complex mixture
            "learning_rate": config["base_sft_lr"],
            "batch_size": 1,
            "num_layers": 8,
        }

        dpo_config = {
            "iters": int(400 * multiplier),
            "learning_rate": config["base_dpo_lr"],
            "beta": config["base_dpo_beta"],
            "batch_size": 1,
            "num_layers": 8,
        }

        prediction = self.predictor.predict(mixture)

        return ScalingRecipe(
            name=f"epistemic_stack_{target_model.replace('-', '_')}",
            base_model="phi-4-mini (Gold Master)",
            target_model=target_model,
            mixture=mixture,
            sft_config=sft_config,
            dpo_config=dpo_config,
            predicted_truthfulqa=prediction.predicted_truthfulqa,
            predicted_capability_delta=prediction.predicted_capability_delta,
            validation_prompts=self.VALIDATION_PROMPTS + [
                "How confident are you in this answer?",
                "What are the limits of your knowledge here?",
            ],
        )

    def generate_custom_recipe(
        self,
        name: str,
        target_model: str,
        isotope_weights: Dict[str, float],
        n_examples: int = 100,
    ) -> ScalingRecipe:
        """Generate custom recipe from isotope weights."""

        if target_model not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {target_model}")

        config = self.MODEL_CONFIGS[target_model]
        multiplier = config["typical_iters_multiplier"]

        mixture = self.designer.custom_mixture(name, isotope_weights, n_examples)
        prediction = self.predictor.predict(mixture)

        sft_config = {
            "iters": int(100 * multiplier),
            "learning_rate": config["base_sft_lr"],
            "batch_size": 1,
            "num_layers": 8,
        }

        dpo_config = {
            "iters": int(300 * multiplier),
            "learning_rate": config["base_dpo_lr"],
            "beta": config["base_dpo_beta"],
            "batch_size": 1,
            "num_layers": 8,
        }

        return ScalingRecipe(
            name=name,
            base_model="phi-4-mini (Gold Master)",
            target_model=target_model,
            mixture=mixture,
            sft_config=sft_config,
            dpo_config=dpo_config,
            predicted_truthfulqa=prediction.predicted_truthfulqa,
            predicted_capability_delta=prediction.predicted_capability_delta,
            validation_prompts=self.VALIDATION_PROMPTS,
        )


def generate_scaling_recipe(target_model: str, recipe_type: str = "soliton_boost") -> ScalingRecipe:
    """Quick recipe generation."""
    generator = ScalingRecipeGenerator()
    if recipe_type == "soliton_boost":
        return generator.generate_soliton_boost_recipe(target_model)
    elif recipe_type == "epistemic_stack":
        return generator.generate_epistemic_stack_recipe(target_model)
    else:
        raise ValueError(f"Unknown recipe type: {recipe_type}")
