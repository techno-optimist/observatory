"""
Goldilocks Calibration Configuration

Implements the balance ratio discovery from JOURNEY.md:
- Too few balance examples = over-application of isotopes (hallucinating doubt)
- Too many balance examples = under-application (mode collapse)
- "Just right" = 5-7% balance ratio (the "5% Solution")

Product presets:
- Philosopher (0-2%): Maximum skepticism, minimal direct answers
- Analyst (5-7%): Balanced - the "5% solution" (default)
- Assistant (10%+): Maximum helpfulness, minimal friction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path


class TemperamentProfile(Enum):
    """
    Temperament profiles controlling skepticism vs helpfulness balance.

    Discovered through Goldilocks calibration experiments.
    """
    # Maximum skepticism - every claim gets challenged
    PHILOSOPHER = "philosopher"

    # Balanced - skepticism on false premises, directness on facts
    ANALYST = "analyst"

    # Maximum helpfulness - minimal friction
    ASSISTANT = "assistant"

    # Custom profile with explicit ratios
    CUSTOM = "custom"


@dataclass
class GoldilocksConfig:
    """
    Configuration for Goldilocks calibration.

    Controls the balance between skeptical/epistemic isotopes
    and direct, helpful responses.
    """
    # Balance ratio: percentage of training examples that are
    # simple Q→A pairs without isotope behavior
    balance_ratio: float = 0.05  # 5% = the "5% Solution"

    # Temperament profile (or "custom" for manual config)
    profile: TemperamentProfile = TemperamentProfile.ANALYST

    # Skepticism dial: 0.0 = never skeptical, 1.0 = always skeptical
    # This affects isotope activation thresholds
    skepticism_level: float = 0.5

    # Which isotopes to suppress on simple factual questions
    suppress_on_factual: List[str] = field(default_factory=lambda: [
        "soliton",
        "calibrator",
        "reflector",
        "skeptic",
        "limiter",
    ])

    # Which isotopes to always activate on myth/false premise
    activate_on_myth: List[str] = field(default_factory=lambda: [
        "skeptic",
    ])

    # Threshold for mode discrimination
    # Higher = more confident classification needed
    mode_threshold: float = 0.3

    @classmethod
    def for_profile(cls, profile: TemperamentProfile) -> "GoldilocksConfig":
        """Create config from a named profile."""
        if profile == TemperamentProfile.PHILOSOPHER:
            return cls(
                balance_ratio=0.02,  # 2% balance
                profile=profile,
                skepticism_level=0.8,
                mode_threshold=0.2,  # Lower threshold = more skepticism
            )
        elif profile == TemperamentProfile.ANALYST:
            return cls(
                balance_ratio=0.05,  # 5% balance - the golden ratio
                profile=profile,
                skepticism_level=0.5,
                mode_threshold=0.3,
            )
        elif profile == TemperamentProfile.ASSISTANT:
            return cls(
                balance_ratio=0.12,  # 12% balance
                profile=profile,
                skepticism_level=0.2,
                mode_threshold=0.5,  # Higher threshold = less skepticism
            )
        else:
            return cls(profile=profile)


# Pre-defined product configurations
PRODUCT_CONFIGS = {
    # Research assistant - maximum cognitive depth
    "forty2-spark": GoldilocksConfig.for_profile(TemperamentProfile.ANALYST),

    # Audit/review tool - high skepticism
    "forty2-auditor": GoldilocksConfig(
        balance_ratio=0.03,
        profile=TemperamentProfile.CUSTOM,
        skepticism_level=0.7,
        mode_threshold=0.25,
    ),

    # Safety guardian - very high skepticism, low helpfulness
    "forty2-guardian": GoldilocksConfig(
        balance_ratio=0.01,
        profile=TemperamentProfile.CUSTOM,
        skepticism_level=0.9,
        mode_threshold=0.2,
    ),

    # General assistant - maximum helpfulness
    "general-assistant": GoldilocksConfig.for_profile(TemperamentProfile.ASSISTANT),

    # Educational tutor - balanced with emphasis on explanation
    "tutor": GoldilocksConfig(
        balance_ratio=0.08,
        profile=TemperamentProfile.CUSTOM,
        skepticism_level=0.4,
        mode_threshold=0.35,
    ),
}


@dataclass
class CalibrationResult:
    """Result of running Goldilocks calibration."""
    config: GoldilocksConfig
    leakage_rate: float  # Rate of isotope leakage on factual questions
    myth_rejection_rate: float  # Rate of proper myth rejection
    mode_discrimination_accuracy: float  # Accuracy of mode classification
    truthfulqa_delta: float  # Change vs base model
    recommendation: str  # Suggested adjustment


class GoldilocksCalibrator:
    """
    Calibrates balance ratio for optimal Zero-Tax Alignment.

    The calibration process:
    1. Start with default 5% balance
    2. Test for leakage (too much skepticism)
    3. Test for myth rejection (too little skepticism)
    4. Adjust balance ratio based on results
    5. Repeat until optimal

    Usage:
        calibrator = GoldilocksCalibrator(model_runner)
        result = calibrator.calibrate(initial_config)
        print(f"Optimal balance ratio: {result.config.balance_ratio}")
    """

    def __init__(
        self,
        model_runner,
        base_model_runner=None,
    ):
        """
        Args:
            model_runner: Function that takes prompt and returns response
            base_model_runner: Optional base model for comparison
        """
        self.model_runner = model_runner
        self.base_model_runner = base_model_runner

    def test_leakage(self) -> float:
        """Test for isotope leakage on factual questions."""
        from .validation import FACTUAL_QUESTIONS, check_leakage

        leaked = 0
        for item in FACTUAL_QUESTIONS:
            response = self.model_runner(item["q"])
            has_leakage, _ = check_leakage(response)
            if has_leakage:
                leaked += 1

        return leaked / len(FACTUAL_QUESTIONS)

    def test_myth_rejection(self) -> float:
        """Test for proper myth rejection."""
        from .validation import MYTH_QUESTIONS, check_myth_rejection

        rejected = 0
        for item in MYTH_QUESTIONS:
            response = self.model_runner(item["q"])
            rejected_myth, _ = check_myth_rejection(response, item["markers"])
            if rejected_myth:
                rejected += 1

        return rejected / len(MYTH_QUESTIONS)

    def test_mode_discrimination(self) -> float:
        """Test mode discrimination accuracy."""
        from .validation import MODE_DISCRIMINATION_TESTS
        from .detectors import detect_leakage

        correct = 0
        for item in MODE_DISCRIMINATION_TESTS:
            response = self.model_runner(item["q"])
            leakage = detect_leakage(response)

            if item["expected"] == "direct":
                # Should not have leakage patterns
                if not leakage.leaked:
                    correct += 1
            else:
                # Should have some analytical engagement
                # (either leakage patterns or skeptic markers)
                if leakage.leaked or "skeptic" in response.lower() or "claim" in response.lower():
                    correct += 1

        return correct / len(MODE_DISCRIMINATION_TESTS)

    def calibrate(
        self,
        initial_config: Optional[GoldilocksConfig] = None,
        max_iterations: int = 5,
    ) -> CalibrationResult:
        """
        Run calibration to find optimal balance ratio.

        Args:
            initial_config: Starting configuration
            max_iterations: Maximum calibration iterations

        Returns:
            CalibrationResult with optimal configuration
        """
        config = initial_config or GoldilocksConfig.for_profile(TemperamentProfile.ANALYST)

        for iteration in range(max_iterations):
            # Run tests
            leakage = self.test_leakage()
            myth_rate = self.test_myth_rejection()
            mode_acc = self.test_mode_discrimination()

            # Determine adjustment
            if leakage > 0.2:  # Too much skepticism
                # Increase balance ratio (more direct responses)
                config.balance_ratio = min(0.20, config.balance_ratio * 1.5)
                recommendation = "Increase balance ratio (reduce skepticism)"
            elif myth_rate < 0.7:  # Too little skepticism
                # Decrease balance ratio (more skeptical)
                config.balance_ratio = max(0.01, config.balance_ratio * 0.7)
                recommendation = "Decrease balance ratio (increase skepticism)"
            else:
                recommendation = "Configuration is well-calibrated"
                break

        return CalibrationResult(
            config=config,
            leakage_rate=leakage,
            myth_rejection_rate=myth_rate,
            mode_discrimination_accuracy=mode_acc,
            truthfulqa_delta=0.0,  # Would need actual TruthfulQA eval
            recommendation=recommendation,
        )


def generate_training_mix(
    config: GoldilocksConfig,
    isotope_examples: List[Dict],
    balance_examples: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Generate training data mix based on Goldilocks configuration.

    Args:
        config: Goldilocks configuration
        isotope_examples: List of isotope training examples
        balance_examples: Optional list of balance examples

    Returns:
        Mixed training data with correct ratios
    """
    from .dpo_training import generate_balance_examples

    if balance_examples is None:
        balance_examples = generate_balance_examples()

    # Calculate how many balance examples we need
    n_isotope = len(isotope_examples)
    n_balance_needed = int(n_isotope * config.balance_ratio / (1 - config.balance_ratio))

    # Repeat or sample balance examples to get needed count
    balance_subset = []
    while len(balance_subset) < n_balance_needed:
        for ex in balance_examples:
            balance_subset.append(ex)
            if len(balance_subset) >= n_balance_needed:
                break

    # Combine and shuffle
    import random
    combined = isotope_examples + balance_subset
    random.shuffle(combined)

    return combined


def save_config(config: GoldilocksConfig, path: Path):
    """Save configuration to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "balance_ratio": config.balance_ratio,
        "profile": config.profile.value,
        "skepticism_level": config.skepticism_level,
        "suppress_on_factual": config.suppress_on_factual,
        "activate_on_myth": config.activate_on_myth,
        "mode_threshold": config.mode_threshold,
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_config(path: Path) -> GoldilocksConfig:
    """Load configuration from JSON."""
    with open(path) as f:
        data = json.load(f)

    return GoldilocksConfig(
        balance_ratio=data["balance_ratio"],
        profile=TemperamentProfile(data["profile"]),
        skepticism_level=data["skepticism_level"],
        suppress_on_factual=data.get("suppress_on_factual", []),
        activate_on_myth=data.get("activate_on_myth", []),
        mode_threshold=data.get("mode_threshold", 0.3),
    )
