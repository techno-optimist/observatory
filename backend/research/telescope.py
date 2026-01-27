"""
Unified Telescope Interface for Cultural Soliton Observatory.

The Telescope class provides a unified interface for observing and analyzing
text through the coordination manifold. It combines multiple extraction methods:

1. Regex-based extraction (fast, rule-based)
2. Dependency parsing extraction (syntactic analysis)
3. Semantic extraction (embedding-based similarity)
4. Hybrid extraction (semantic with parsed fallback)

Usage:
    from research.telescope import Telescope, quick_analyze

    telescope = Telescope(extraction_method="hybrid")
    result = telescope.observe("I believe we should work together.")
    print(f"Agency: {result.agency:.2f}, T_CBR: {result.temperature:.2f}")

Author: Cultural Soliton Observatory Team
Version: 2.0.0
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .hierarchical_coordinates import (
    HierarchicalCoordinate,
    CoordinationCore,
    CoordinationModifiers,
    AgencyDecomposition,
    JusticeDecomposition,
    BelongingDecomposition,
    extract_hierarchical_coordinate as regex_extract,
    extract_features as regex_extract_features,
)
from .cbr_thermometer import (
    CBRThermometer,
    CBRReading,
    measure_cbr,
    measure_cbr_batch,
)
from .ossification_alarm import (
    OssificationAlarm,
    OssificationState,
    OssificationRisk,
)
from .legibility_analyzer import LegibilityRegime

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Available extraction methods."""
    REGEX = "regex"
    PARSED = "parsed"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class TelescopeConfig:
    """Configuration for the Telescope."""
    extraction_method: str = "regex"
    hybrid_confidence_threshold: float = 0.7
    semantic_model: str = "all-MiniLM-L6-v2"
    cbr_window_size: int = 50
    ossification_window_size: int = 100


@dataclass
class ObservationResult:
    """Complete result of a telescope observation."""

    coordinate: HierarchicalCoordinate
    confidence: float
    extraction_method: str
    raw_scores: Dict[str, float] = field(default_factory=dict)

    # CBR metrics
    temperature: float = 0.0
    signal_strength: float = 0.0
    phase: str = "unknown"
    kernel_label: str = "unknown"
    legibility: float = 0.5

    @property
    def agency(self) -> float:
        return self.coordinate.core.agency.aggregate

    @property
    def justice(self) -> float:
        return self.coordinate.core.justice.aggregate

    @property
    def belonging(self) -> float:
        return self.coordinate.core.belonging.aggregate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate": self.coordinate.to_dict(),
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "raw_scores": self.raw_scores,
            "cbr": {
                "temperature": self.temperature,
                "signal_strength": self.signal_strength,
                "phase": self.phase,
                "kernel_label": self.kernel_label,
                "legibility": self.legibility,
            },
            "aggregates": {
                "agency": self.agency,
                "justice": self.justice,
                "belonging": self.belonging,
            },
        }


class Telescope:
    """
    Unified interface for observing text through the coordination manifold.

    Example:
        telescope = Telescope()
        result = telescope.observe("I made this decision myself.")
        print(f"Agency: {result.agency:.2f}")
        print(f"Temperature: {result.temperature:.2f}")
        print(f"Phase: {result.phase}")
    """

    def __init__(
        self,
        extraction_method: str = "regex",
        config: Optional[TelescopeConfig] = None,
    ):
        self.config = config or TelescopeConfig(extraction_method=extraction_method)
        self.extraction_method = extraction_method

        # Lazy-loaded components
        self._cbr_thermometer: Optional[CBRThermometer] = None
        self._ossification_alarm: Optional[OssificationAlarm] = None
        self._semantic_extractor = None
        self._parsed_available = None
        self._semantic_available = None

        logger.info(f"Telescope initialized with method='{extraction_method}'")

    @property
    def cbr_thermometer(self) -> CBRThermometer:
        if self._cbr_thermometer is None:
            self._cbr_thermometer = CBRThermometer(
                window_size=self.config.cbr_window_size
            )
        return self._cbr_thermometer

    @property
    def ossification_alarm(self) -> OssificationAlarm:
        if self._ossification_alarm is None:
            self._ossification_alarm = OssificationAlarm(
                window_size=self.config.ossification_window_size,
            )
        return self._ossification_alarm

    def _check_parsed_available(self) -> bool:
        """Check if parsed extraction is available (requires spaCy)."""
        if self._parsed_available is None:
            try:
                from .parsed_feature_extraction import parse_text
                self._parsed_available = True
            except ImportError:
                self._parsed_available = False
        return self._parsed_available

    def _check_semantic_available(self) -> bool:
        """Check if semantic extraction is available."""
        if self._semantic_available is None:
            try:
                from .semantic_feature_extraction import SemanticFeatureExtractor
                self._semantic_available = True
            except ImportError:
                self._semantic_available = False
        return self._semantic_available

    def _get_semantic_extractor(self):
        """Get or create semantic extractor."""
        if self._semantic_extractor is None and self._check_semantic_available():
            from .semantic_feature_extraction import SemanticFeatureExtractor
            self._semantic_extractor = SemanticFeatureExtractor(
                model_name=self.config.semantic_model,
            )
        return self._semantic_extractor

    def observe(self, text: str) -> ObservationResult:
        """
        Observe a single text and extract its hierarchical coordinate.

        Args:
            text: Input text to analyze

        Returns:
            ObservationResult containing coordinate and CBR metrics
        """
        if not text or not text.strip():
            return self._create_empty_result()

        # Take CBR measurement
        cbr_reading = self.cbr_thermometer.measure(text)

        # Extract coordinate based on method
        if self.extraction_method == "regex":
            return self._observe_regex(text, cbr_reading)
        elif self.extraction_method == "parsed":
            return self._observe_parsed(text, cbr_reading)
        elif self.extraction_method == "semantic":
            return self._observe_semantic(text, cbr_reading)
        elif self.extraction_method == "hybrid":
            return self._observe_hybrid(text, cbr_reading)
        else:
            logger.warning(f"Unknown method '{self.extraction_method}', using regex")
            return self._observe_regex(text, cbr_reading)

    def _observe_regex(self, text: str, cbr: CBRReading) -> ObservationResult:
        """Extract using regex-based patterns."""
        features = regex_extract_features(text)
        total = sum(features.values())
        word_count = len(text.split())
        confidence = min(0.5 + 0.5 * total / max(word_count, 1), 1.0)

        return ObservationResult(
            coordinate=cbr.coordinate,
            confidence=confidence,
            extraction_method="regex",
            raw_scores={k: float(v) for k, v in features.items()},
            temperature=cbr.temperature,
            signal_strength=cbr.signal_strength,
            phase=cbr.phase.value,
            kernel_label=cbr.kernel_label,
            legibility=cbr.legibility,
        )

    def _observe_parsed(self, text: str, cbr: CBRReading) -> ObservationResult:
        """Extract using dependency parsing."""
        if not self._check_parsed_available():
            logger.warning("Parsed extraction not available, falling back to regex")
            return self._observe_regex(text, cbr)

        try:
            from .parsed_feature_extraction import parse_text, extract_features_parsed
            parsed = parse_text(text)

            # Build coordinate from parsed features
            coord = HierarchicalCoordinate(
                core=CoordinationCore(
                    agency=AgencyDecomposition(
                        self_agency=parsed.self_agency,
                        other_agency=parsed.other_agency,
                        system_agency=parsed.system_agency,
                    ),
                    justice=JusticeDecomposition(
                        procedural=parsed.procedural_justice,
                        distributive=parsed.distributive_justice,
                        interactional=parsed.interactional_justice,
                    ),
                    belonging=BelongingDecomposition(
                        ingroup=parsed.ingroup,
                        outgroup=parsed.outgroup,
                        universal=parsed.universal,
                    ),
                ),
                modifiers=CoordinationModifiers(),
            )

            confidence = 0.7 if parsed.sentence_count > 0 else 0.5

            return ObservationResult(
                coordinate=coord,
                confidence=confidence,
                extraction_method="parsed",
                raw_scores={
                    "self_agency": parsed.self_agency,
                    "other_agency": parsed.other_agency,
                    "system_agency": parsed.system_agency,
                    "procedural": parsed.procedural_justice,
                    "distributive": parsed.distributive_justice,
                    "interactional": parsed.interactional_justice,
                },
                temperature=cbr.temperature,
                signal_strength=cbr.signal_strength,
                phase=cbr.phase.value,
                kernel_label=cbr.kernel_label,
                legibility=cbr.legibility,
            )
        except Exception as e:
            logger.error(f"Parsed extraction failed: {e}")
            return self._observe_regex(text, cbr)

    def _observe_semantic(self, text: str, cbr: CBRReading) -> ObservationResult:
        """Extract using semantic similarity."""
        extractor = self._get_semantic_extractor()
        if extractor is None:
            logger.warning("Semantic extraction not available, falling back to regex")
            return self._observe_regex(text, cbr)

        try:
            result = extractor.extract_with_details(text)
            confidences = [s.confidence for s in result.construct_scores.values()]
            avg_confidence = np.mean(confidences) if confidences else 0.5

            return ObservationResult(
                coordinate=result.hierarchical_coordinate,
                confidence=float(avg_confidence),
                extraction_method="semantic",
                raw_scores={k: s.score for k, s in result.construct_scores.items()},
                temperature=cbr.temperature,
                signal_strength=cbr.signal_strength,
                phase=cbr.phase.value,
                kernel_label=cbr.kernel_label,
                legibility=cbr.legibility,
            )
        except Exception as e:
            logger.error(f"Semantic extraction failed: {e}")
            return self._observe_regex(text, cbr)

    def _observe_hybrid(self, text: str, cbr: CBRReading) -> ObservationResult:
        """Hybrid: semantic with parsed fallback."""
        semantic_result = self._observe_semantic(text, cbr)

        if semantic_result.confidence >= self.config.hybrid_confidence_threshold:
            semantic_result.extraction_method = "hybrid:semantic"
            return semantic_result

        parsed_result = self._observe_parsed(text, cbr)
        if parsed_result.confidence > semantic_result.confidence:
            parsed_result.extraction_method = "hybrid:parsed"
            return parsed_result

        semantic_result.extraction_method = "hybrid:semantic_low"
        return semantic_result

    def _create_empty_result(self) -> ObservationResult:
        return ObservationResult(
            coordinate=HierarchicalCoordinate(),
            confidence=0.0,
            extraction_method="empty",
        )

    def observe_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ObservationResult]:
        """Observe multiple texts."""
        results = []
        for i, text in enumerate(texts):
            results.append(self.observe(text))
            if progress_callback:
                progress_callback(i + 1, len(texts))
        return results

    def measure_cbr(self, text: str) -> Dict[str, Any]:
        """Measure CBR for a single text."""
        reading = self.cbr_thermometer.measure(text)
        return {
            "temperature": reading.temperature,
            "signal_strength": reading.signal_strength,
            "phase": reading.phase.value,
            "kernel_label": reading.kernel_label,
            "legibility": reading.legibility,
        }

    def check_ossification(self, texts: List[str]) -> Dict[str, Any]:
        """Check ossification risk for a message sequence."""
        alarm = OssificationAlarm(
            window_size=max(len(texts) + 10, 20),
            baseline_window=max(5, min(50, len(texts) // 2)),
        )

        for text in texts:
            alarm.update(text)

        status = alarm.get_status()
        return {
            "risk_level": status.get("risk_level", "unknown"),
            "variance_ratio": status.get("variance_ratio", 1.0),
            "compression_level": status.get("compression_level", 0.0),
            "kernel_entropy": status.get("kernel_entropy", 3.0),
            "message_count": len(texts),
            "intervention_suggested": alarm.suggest_intervention(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get CBR summary statistics."""
        return self.cbr_thermometer.get_summary()

    def reset(self) -> None:
        """Reset all monitoring state."""
        self._cbr_thermometer = None
        self._ossification_alarm = None


def create_telescope(method: str = "regex", **kwargs) -> Telescope:
    """Factory function to create a Telescope instance."""
    config = TelescopeConfig(extraction_method=method, **kwargs)
    return Telescope(extraction_method=method, config=config)


def quick_analyze(text: str) -> Dict[str, Any]:
    """One-liner for quick text analysis."""
    cbr = measure_cbr(text)
    coord = regex_extract(text)
    agency, justice, belonging = coord.core.to_legacy_3d()

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "agency": agency,
        "justice": justice,
        "belonging": belonging,
        "temperature": cbr["temperature"],
        "signal_strength": cbr["signal_strength"],
        "phase": cbr["phase"],
        "kernel_label": cbr["kernel_label"],
    }


__all__ = [
    "Telescope",
    "TelescopeConfig",
    "ExtractionMethod",
    "ObservationResult",
    "create_telescope",
    "quick_analyze",
]
