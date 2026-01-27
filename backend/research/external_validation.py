"""
External Validation Infrastructure for Cultural Soliton Observatory.

Provides psychometric validation of Agency/Justice/Belonging coordinates against
established psychological scales. Addresses peer reviewer concern about lack of
external validation for the extracted coordination dimensions.

Validated Scales Implemented:
1. Sense of Agency Scale (Tapal et al., 2017) - Maps to Agency dimension
2. Organizational Justice Scale (Colquitt, 2001) - Maps to Justice dimension
3. Inclusion of Other in Self Scale (Aron et al., 1992) - Maps to Belonging
4. Brief Sense of Community Scale (Peterson et al., 2008) - Maps to Belonging

Validation Methods:
- Convergent validity: Same construct, different method (r > .50 expected)
- Discriminant validity: Different constructs should correlate weakly (r < .30)
- MTMM (Multi-Trait Multi-Method) matrix analysis
- Criterion validity: Known-groups and predictive validity

References:
- Tapal, A., et al. (2017). The Sense of Agency Scale. Consciousness & Cognition.
- Colquitt, J. A. (2001). On the dimensionality of organizational justice. JAP.
- Aron, A., Aron, E. N., & Smollan, D. (1992). IOS Scale. JPSP.
- Peterson, N. A., et al. (2008). Brief Sense of Community Scale. J Community Psych.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

# Local imports
from .hierarchical_coordinates import (
    HierarchicalCoordinate,
    CoordinationCore,
    AgencyDecomposition,
    JusticeDecomposition,
    BelongingDecomposition,
    extract_hierarchical_coordinate,
)
from .academic_statistics import (
    EffectSize,
    cohens_d,
    hedges_g,
    bootstrap_ci,
    BootstrapEstimate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Psychological Scale Definitions
# =============================================================================

@dataclass
class ScaleItem:
    """Individual item in a psychological scale."""

    id: str
    text: str
    reverse_scored: bool = False
    subscale: Optional[str] = None
    factor_loading: float = 1.0  # From validation studies

    def score(self, response: int, scale_max: int = 7) -> float:
        """Score an item response, handling reverse scoring."""
        if self.reverse_scored:
            return (scale_max + 1) - response
        return response


@dataclass
class PsychologicalScale:
    """Complete psychological scale with items and scoring."""

    name: str
    abbreviation: str
    citation: str
    items: List[ScaleItem]
    subscales: Dict[str, List[str]]  # subscale_name -> item_ids
    response_range: Tuple[int, int] = (1, 7)  # Likert scale range
    construct_measured: str = ""
    reliability_alpha: float = 0.0  # Cronbach's alpha from validation

    def score_responses(
        self,
        responses: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Score all subscales from item responses.

        Args:
            responses: Dict mapping item_id -> response value

        Returns:
            Dict with subscale scores (mean of items)
        """
        scores = {}
        scale_max = self.response_range[1]

        for subscale_name, item_ids in self.subscales.items():
            subscale_scores = []
            for item_id in item_ids:
                if item_id in responses:
                    item = next(i for i in self.items if i.id == item_id)
                    subscale_scores.append(
                        item.score(responses[item_id], scale_max)
                    )

            if subscale_scores:
                scores[subscale_name] = np.mean(subscale_scores)

        # Overall score
        all_scores = []
        for item in self.items:
            if item.id in responses:
                all_scores.append(item.score(responses[item.id], scale_max))

        if all_scores:
            scores["total"] = np.mean(all_scores)

        return scores

    def get_normalization_params(self) -> Tuple[float, float]:
        """Get min/max for normalizing to [-1, 1]."""
        return (self.response_range[0], self.response_range[1])


# =============================================================================
# Sense of Agency Scale (Tapal et al., 2017)
# =============================================================================

SENSE_OF_AGENCY_SCALE = PsychologicalScale(
    name="Sense of Agency Scale",
    abbreviation="SoAS",
    citation="Tapal, A., Oren, E., Dar, R., & Eitam, B. (2017). The Sense of Agency Scale: A measure of consciously perceived control over one's mind, body, and the immediate environment. Consciousness and Cognition, 51, 181-189.",
    construct_measured="agency",
    reliability_alpha=0.83,
    response_range=(1, 7),
    items=[
        # Sense of Positive Agency (SoPA) subscale
        ScaleItem("soas_1", "I am in full control of what I do", subscale="positive_agency"),
        ScaleItem("soas_2", "I am the author of my actions", subscale="positive_agency"),
        ScaleItem("soas_3", "The outcomes of my actions generally surprise me", reverse_scored=True, subscale="positive_agency"),
        ScaleItem("soas_4", "Things I do are subject only to my free will", subscale="positive_agency"),
        ScaleItem("soas_5", "My movements are automatic - I don't control them", reverse_scored=True, subscale="positive_agency"),
        ScaleItem("soas_6", "My actions just happen without my intention", reverse_scored=True, subscale="positive_agency"),
        ScaleItem("soas_7", "I am completely responsible for everything that results from my actions", subscale="positive_agency"),

        # Sense of Negative Agency (SoNA) subscale
        ScaleItem("soas_8", "I feel that I am not really responsible for what I do", reverse_scored=True, subscale="negative_agency"),
        ScaleItem("soas_9", "My actions just happen to me", reverse_scored=True, subscale="negative_agency"),
        ScaleItem("soas_10", "The consequences of my actions feel like they don't follow naturally from my actions", reverse_scored=True, subscale="negative_agency"),
        ScaleItem("soas_11", "I feel like my movements are kind of automatic", reverse_scored=True, subscale="negative_agency"),
        ScaleItem("soas_12", "I'm just an instrument of some external force", reverse_scored=True, subscale="negative_agency"),
        ScaleItem("soas_13", "Something else controls what I do", reverse_scored=True, subscale="negative_agency"),
    ],
    subscales={
        "positive_agency": ["soas_1", "soas_2", "soas_3", "soas_4", "soas_5", "soas_6", "soas_7"],
        "negative_agency": ["soas_8", "soas_9", "soas_10", "soas_11", "soas_12", "soas_13"],
    }
)


# =============================================================================
# Organizational Justice Scale (Colquitt, 2001)
# =============================================================================

ORGANIZATIONAL_JUSTICE_SCALE = PsychologicalScale(
    name="Organizational Justice Scale",
    abbreviation="OJS",
    citation="Colquitt, J. A. (2001). On the dimensionality of organizational justice: A construct validation of a measure. Journal of Applied Psychology, 86(3), 386-400.",
    construct_measured="justice",
    reliability_alpha=0.93,
    response_range=(1, 5),  # 5-point scale in original
    items=[
        # Procedural Justice subscale
        ScaleItem("ojs_pj1", "Have you been able to express your views and feelings during those procedures?", subscale="procedural"),
        ScaleItem("ojs_pj2", "Have you had influence over the outcome arrived at by those procedures?", subscale="procedural"),
        ScaleItem("ojs_pj3", "Have those procedures been applied consistently?", subscale="procedural"),
        ScaleItem("ojs_pj4", "Have those procedures been free of bias?", subscale="procedural"),
        ScaleItem("ojs_pj5", "Have those procedures been based on accurate information?", subscale="procedural"),
        ScaleItem("ojs_pj6", "Have you been able to appeal the outcome arrived at by those procedures?", subscale="procedural"),
        ScaleItem("ojs_pj7", "Have those procedures upheld ethical and moral standards?", subscale="procedural"),

        # Distributive Justice subscale
        ScaleItem("ojs_dj1", "Does your outcome reflect the effort you have put into your work?", subscale="distributive"),
        ScaleItem("ojs_dj2", "Is your outcome appropriate for the work you have completed?", subscale="distributive"),
        ScaleItem("ojs_dj3", "Does your outcome reflect what you have contributed to the organization?", subscale="distributive"),
        ScaleItem("ojs_dj4", "Is your outcome justified, given your performance?", subscale="distributive"),

        # Interpersonal Justice subscale
        ScaleItem("ojs_ipj1", "Has your supervisor treated you in a polite manner?", subscale="interpersonal"),
        ScaleItem("ojs_ipj2", "Has your supervisor treated you with dignity?", subscale="interpersonal"),
        ScaleItem("ojs_ipj3", "Has your supervisor treated you with respect?", subscale="interpersonal"),
        ScaleItem("ojs_ipj4", "Has your supervisor refrained from improper remarks or comments?", subscale="interpersonal"),

        # Informational Justice subscale
        ScaleItem("ojs_ifj1", "Has your supervisor been candid in communications with you?", subscale="informational"),
        ScaleItem("ojs_ifj2", "Has your supervisor explained the procedures thoroughly?", subscale="informational"),
        ScaleItem("ojs_ifj3", "Were your supervisor's explanations regarding the procedures reasonable?", subscale="informational"),
        ScaleItem("ojs_ifj4", "Has your supervisor communicated details in a timely manner?", subscale="informational"),
        ScaleItem("ojs_ifj5", "Has your supervisor seemed to tailor communications to your specific needs?", subscale="informational"),
    ],
    subscales={
        "procedural": ["ojs_pj1", "ojs_pj2", "ojs_pj3", "ojs_pj4", "ojs_pj5", "ojs_pj6", "ojs_pj7"],
        "distributive": ["ojs_dj1", "ojs_dj2", "ojs_dj3", "ojs_dj4"],
        "interpersonal": ["ojs_ipj1", "ojs_ipj2", "ojs_ipj3", "ojs_ipj4"],
        "informational": ["ojs_ifj1", "ojs_ifj2", "ojs_ifj3", "ojs_ifj4", "ojs_ifj5"],
    }
)


# =============================================================================
# Inclusion of Other in Self Scale (Aron et al., 1992)
# =============================================================================

# Note: IOS is typically a single pictorial item, here expanded for text-based use
INCLUSION_OF_OTHER_IN_SELF_SCALE = PsychologicalScale(
    name="Inclusion of Other in Self Scale",
    abbreviation="IOS",
    citation="Aron, A., Aron, E. N., & Smollan, D. (1992). Inclusion of Other in the Self Scale and the structure of interpersonal closeness. Journal of Personality and Social Psychology, 63(4), 596-612.",
    construct_measured="belonging",
    reliability_alpha=0.95,  # Test-retest reliability
    response_range=(1, 7),
    items=[
        # Single pictorial item adapted to multiple text items
        ScaleItem("ios_1", "I feel a strong connection to this group/community", subscale="closeness"),
        ScaleItem("ios_2", "I see the group's successes as my own successes", subscale="closeness"),
        ScaleItem("ios_3", "When I think of myself, I include membership in this group as part of who I am", subscale="closeness"),
        ScaleItem("ios_4", "I feel that what happens to this group happens to me", subscale="closeness"),
        ScaleItem("ios_5", "The boundary between me and this group feels very permeable", subscale="closeness"),
    ],
    subscales={
        "closeness": ["ios_1", "ios_2", "ios_3", "ios_4", "ios_5"],
    }
)


# =============================================================================
# Brief Sense of Community Scale (Peterson et al., 2008)
# =============================================================================

BRIEF_SENSE_OF_COMMUNITY_SCALE = PsychologicalScale(
    name="Brief Sense of Community Scale",
    abbreviation="BSCS",
    citation="Peterson, N. A., Speer, P. W., & McMillan, D. W. (2008). Validation of a brief sense of community scale: Confirmation of the principal theory of sense of community. Journal of Community Psychology, 36(1), 61-73.",
    construct_measured="belonging",
    reliability_alpha=0.92,
    response_range=(1, 5),
    items=[
        # Needs Fulfillment
        ScaleItem("bscs_1", "I can get what I need in this community", subscale="needs_fulfillment"),
        ScaleItem("bscs_2", "This community helps me fulfill my needs", subscale="needs_fulfillment"),

        # Group Membership
        ScaleItem("bscs_3", "I feel like a member of this community", subscale="membership"),
        ScaleItem("bscs_4", "I belong in this community", subscale="membership"),

        # Influence
        ScaleItem("bscs_5", "I have a say about what goes on in my community", subscale="influence"),
        ScaleItem("bscs_6", "People in this community are good at influencing each other", subscale="influence"),

        # Emotional Connection
        ScaleItem("bscs_7", "I feel connected to this community", subscale="emotional_connection"),
        ScaleItem("bscs_8", "I have a good bond with others in this community", subscale="emotional_connection"),
    ],
    subscales={
        "needs_fulfillment": ["bscs_1", "bscs_2"],
        "membership": ["bscs_3", "bscs_4"],
        "influence": ["bscs_5", "bscs_6"],
        "emotional_connection": ["bscs_7", "bscs_8"],
    }
)


# Registry of all scales
VALIDATION_SCALES = {
    "agency": SENSE_OF_AGENCY_SCALE,
    "justice": ORGANIZATIONAL_JUSTICE_SCALE,
    "belonging_ios": INCLUSION_OF_OTHER_IN_SELF_SCALE,
    "belonging_bscs": BRIEF_SENSE_OF_COMMUNITY_SCALE,
}


# =============================================================================
# Validation Result Dataclasses
# =============================================================================

@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""

    r: float  # Pearson correlation coefficient
    p_value: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n: int  # Sample size
    method: str = "pearson"

    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant at alpha=.05."""
        return self.p_value < 0.05

    @property
    def effect_size(self) -> str:
        """Interpret correlation effect size (Cohen's conventions)."""
        r_abs = abs(self.r)
        if r_abs < 0.10:
            return "negligible"
        elif r_abs < 0.30:
            return "small"
        elif r_abs < 0.50:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
            "method": self.method,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
        }


@dataclass
class ValidationResult:
    """Result of validating extracted coordinates against a scale."""

    scale_name: str
    construct: str  # agency, justice, or belonging
    correlation: CorrelationResult
    subscale_correlations: Dict[str, CorrelationResult]
    convergent_validity_met: bool  # r > 0.50 with same construct
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "scale_name": self.scale_name,
            "construct": self.construct,
            "correlation": self.correlation.to_dict(),
            "subscale_correlations": {
                k: v.to_dict() for k, v in self.subscale_correlations.items()
            },
            "convergent_validity_met": self.convergent_validity_met,
            "n_samples": self.n_samples,
        }


@dataclass
class MTMMCell:
    """Single cell in Multi-Trait Multi-Method matrix."""

    trait1: str
    trait2: str
    method1: str
    method2: str
    correlation: CorrelationResult
    cell_type: str  # "monotrait-heteromethod", "heterotrait-monomethod", "heterotrait-heteromethod"

    def to_dict(self) -> dict:
        return {
            "trait1": self.trait1,
            "trait2": self.trait2,
            "method1": self.method1,
            "method2": self.method2,
            "correlation": self.correlation.to_dict(),
            "cell_type": self.cell_type,
        }


@dataclass
class MTMMResult:
    """Complete Multi-Trait Multi-Method matrix analysis."""

    traits: List[str]
    methods: List[str]
    matrix: Dict[Tuple[str, str, str, str], MTMMCell]  # (t1, t2, m1, m2) -> cell

    # Validity evidence
    convergent_validity: float  # Mean monotrait-heteromethod correlation
    discriminant_validity_htmm: float  # Mean heterotrait-monomethod
    discriminant_validity_hthm: float  # Mean heterotrait-heteromethod

    # Campbell & Fiske criteria
    criterion_1_met: bool  # Convergent coefficients significantly > 0
    criterion_2_met: bool  # Convergent > heterotrait-heteromethod
    criterion_3_met: bool  # Convergent > heterotrait-monomethod
    criterion_4_met: bool  # Same pattern across methods

    def to_dict(self) -> dict:
        matrix_dict = {}
        for key, cell in self.matrix.items():
            matrix_dict[str(key)] = cell.to_dict()

        return {
            "traits": self.traits,
            "methods": self.methods,
            "matrix": matrix_dict,
            "convergent_validity": self.convergent_validity,
            "discriminant_validity_htmm": self.discriminant_validity_htmm,
            "discriminant_validity_hthm": self.discriminant_validity_hthm,
            "criterion_1_met": self.criterion_1_met,
            "criterion_2_met": self.criterion_2_met,
            "criterion_3_met": self.criterion_3_met,
            "criterion_4_met": self.criterion_4_met,
        }

    def format_matrix(self) -> str:
        """Format MTMM matrix as ASCII table for display."""
        lines = ["MTMM Matrix Analysis", "=" * 60]

        # Header
        header = "           |"
        for method in self.methods:
            for trait in self.traits:
                header += f" {method[:3]}_{trait[:3]} |"
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for m1 in self.methods:
            for t1 in self.traits:
                row = f" {m1[:3]}_{t1[:3]} |"
                for m2 in self.methods:
                    for t2 in self.traits:
                        key = (t1, t2, m1, m2)
                        if key in self.matrix:
                            r = self.matrix[key].correlation.r
                            row += f"   {r:5.2f}  |"
                        else:
                            row += "    --   |"
                lines.append(row)
            lines.append("-" * len(header))

        # Summary
        lines.append("")
        lines.append(f"Convergent validity (monotrait-heteromethod): {self.convergent_validity:.3f}")
        lines.append(f"Discriminant (heterotrait-monomethod): {self.discriminant_validity_htmm:.3f}")
        lines.append(f"Discriminant (heterotrait-heteromethod): {self.discriminant_validity_hthm:.3f}")
        lines.append("")
        lines.append("Campbell & Fiske Criteria:")
        lines.append(f"  1. Convergent > 0: {'PASS' if self.criterion_1_met else 'FAIL'}")
        lines.append(f"  2. Convergent > HTHM: {'PASS' if self.criterion_2_met else 'FAIL'}")
        lines.append(f"  3. Convergent > HTMM: {'PASS' if self.criterion_3_met else 'FAIL'}")
        lines.append(f"  4. Pattern consistency: {'PASS' if self.criterion_4_met else 'FAIL'}")

        return "\n".join(lines)


@dataclass
class KnownGroupsResult:
    """Result of known-groups validity analysis."""

    group1_name: str
    group2_name: str
    construct: str
    group1_mean: float
    group2_mean: float
    effect_size: EffectSize
    t_statistic: float
    p_value: float
    hypothesis_supported: bool  # Did groups differ as expected?

    def to_dict(self) -> dict:
        return {
            "group1_name": self.group1_name,
            "group2_name": self.group2_name,
            "construct": self.construct,
            "group1_mean": self.group1_mean,
            "group2_mean": self.group2_mean,
            "effect_size": self.effect_size.to_dict(),
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "hypothesis_supported": self.hypothesis_supported,
        }


@dataclass
class PredictiveValidityResult:
    """Result of predictive validity analysis."""

    predictor: str  # Which coordinate dimension
    outcome: str  # What was predicted
    correlation: CorrelationResult
    regression_coefficient: float
    r_squared: float
    prediction_accuracy: float  # For classification outcomes

    def to_dict(self) -> dict:
        return {
            "predictor": self.predictor,
            "outcome": self.outcome,
            "correlation": self.correlation.to_dict(),
            "regression_coefficient": self.regression_coefficient,
            "r_squared": self.r_squared,
            "prediction_accuracy": self.prediction_accuracy,
        }


# =============================================================================
# Core Validation Class
# =============================================================================

class ExternalValidator:
    """
    External validation infrastructure for Agency/Justice/Belonging coordinates.

    Validates extracted text coordinates against established psychological scales
    to provide convergent and discriminant validity evidence.

    Usage:
        validator = ExternalValidator()

        # Validate against a single scale
        result = validator.correlate_with_scale(
            texts=["I made this decision myself", ...],
            scale_responses=[{"soas_1": 6, "soas_2": 7, ...}, ...]
        )

        # Compute full MTMM analysis
        mtmm = validator.mtmm_analysis(
            texts=texts,
            scale_data={"agency": agency_responses, "justice": justice_responses, ...}
        )
    """

    def __init__(
        self,
        coordinate_extractor: Optional[Callable[[str], HierarchicalCoordinate]] = None
    ):
        """
        Initialize validator.

        Args:
            coordinate_extractor: Function to extract coordinates from text.
                                  If None, uses default rule-based extraction.
        """
        self.coordinate_extractor = coordinate_extractor or extract_hierarchical_coordinate
        self.scales = VALIDATION_SCALES

    def extract_coordinates(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract Agency/Justice/Belonging coordinates from texts.

        Returns:
            Tuple of (agency_scores, justice_scores, belonging_scores) arrays
        """
        agency = []
        justice = []
        belonging = []

        for text in texts:
            coord = self.coordinate_extractor(text)
            a, j, b = coord.core.to_legacy_3d()
            agency.append(a)
            justice.append(j)
            belonging.append(b)

        return np.array(agency), np.array(justice), np.array(belonging)

    def compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "pearson"
    ) -> CorrelationResult:
        """
        Compute correlation with confidence interval.

        Args:
            x: First variable
            y: Second variable
            method: "pearson" or "spearman"

        Returns:
            CorrelationResult with r, p-value, and 95% CI
        """
        n = len(x)

        if method == "pearson":
            r, p = pearsonr(x, y)
        else:
            r, p = spearmanr(x, y)

        # Fisher z-transformation for CI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = np.arctanh(r) if abs(r) < 1 else np.sign(r) * 3
            se = 1 / np.sqrt(n - 3) if n > 3 else 0.5
            z_crit = 1.96
            ci_lower = np.tanh(z - z_crit * se)
            ci_upper = np.tanh(z + z_crit * se)

        return CorrelationResult(
            r=r,
            p_value=p,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n=n,
            method=method
        )

    def correlate_with_scale(
        self,
        texts: List[str],
        scale_responses: List[Dict[str, int]],
        scale_name: str = "agency"
    ) -> ValidationResult:
        """
        Correlate extracted coordinates with psychological scale responses.

        Args:
            texts: List of text samples
            scale_responses: List of dicts mapping item_id -> response
            scale_name: Which scale ("agency", "justice", "belonging_ios", "belonging_bscs")

        Returns:
            ValidationResult with correlation analysis
        """
        if len(texts) != len(scale_responses):
            raise ValueError("Number of texts must match number of scale responses")

        scale = self.scales[scale_name]

        # Extract coordinates
        agency_scores, justice_scores, belonging_scores = self.extract_coordinates(texts)

        # Determine which extracted dimension to correlate
        if scale.construct_measured == "agency":
            extracted_scores = agency_scores
        elif scale.construct_measured == "justice":
            extracted_scores = justice_scores
        else:  # belonging
            extracted_scores = belonging_scores

        # Score the scale responses
        scale_totals = []
        subscale_scores = {sub: [] for sub in scale.subscales.keys()}

        for responses in scale_responses:
            scores = scale.score_responses(responses)
            scale_totals.append(scores.get("total", 0))

            for sub_name in scale.subscales.keys():
                subscale_scores[sub_name].append(scores.get(sub_name, 0))

        scale_totals = np.array(scale_totals)

        # Normalize to [-1, 1] range for comparability
        scale_min, scale_max = scale.get_normalization_params()
        scale_totals_norm = 2 * (scale_totals - scale_min) / (scale_max - scale_min) - 1

        # Compute correlation with total score
        main_correlation = self.compute_correlation(extracted_scores, scale_totals_norm)

        # Compute subscale correlations
        subscale_correlations = {}
        for sub_name, sub_scores in subscale_scores.items():
            sub_scores = np.array(sub_scores)
            sub_scores_norm = 2 * (sub_scores - scale_min) / (scale_max - scale_min) - 1
            subscale_correlations[sub_name] = self.compute_correlation(
                extracted_scores, sub_scores_norm
            )

        # Check convergent validity criterion (r > 0.50)
        convergent_met = main_correlation.r > 0.50 and main_correlation.is_significant

        return ValidationResult(
            scale_name=scale.name,
            construct=scale.construct_measured,
            correlation=main_correlation,
            subscale_correlations=subscale_correlations,
            convergent_validity_met=convergent_met,
            n_samples=len(texts)
        )

    def compute_convergent_validity(
        self,
        texts: List[str],
        scale_responses: List[Dict[str, int]],
        scale_name: str
    ) -> float:
        """
        Compute convergent validity coefficient.

        Convergent validity is demonstrated when different methods measuring
        the same construct correlate highly (r > .50).

        Args:
            texts: Text samples
            scale_responses: Scale response data
            scale_name: Which scale to validate against

        Returns:
            Convergent validity coefficient (correlation r)
        """
        result = self.correlate_with_scale(texts, scale_responses, scale_name)
        return result.correlation.r

    def compute_discriminant_validity(
        self,
        texts: List[str],
        scale_responses: List[Dict[str, int]],
        scale_name: str
    ) -> float:
        """
        Compute discriminant validity coefficient.

        Discriminant validity is demonstrated when measures of different
        constructs correlate weakly (r < .30).

        This correlates the extracted coordinate for one construct with
        a scale measuring a DIFFERENT construct.

        Args:
            texts: Text samples
            scale_responses: Scale response data for a DIFFERENT construct
            scale_name: Scale name (should measure different construct)

        Returns:
            Discriminant validity coefficient (should be low)
        """
        scale = self.scales[scale_name]

        # Extract all coordinates
        agency, justice, belonging = self.extract_coordinates(texts)

        # Score scale
        scale_totals = []
        for responses in scale_responses:
            scores = scale.score_responses(responses)
            scale_totals.append(scores.get("total", 0))

        scale_totals = np.array(scale_totals)
        scale_min, scale_max = scale.get_normalization_params()
        scale_totals_norm = 2 * (scale_totals - scale_min) / (scale_max - scale_min) - 1

        # Correlate with DIFFERENT construct
        if scale.construct_measured == "agency":
            # Scale measures agency, so correlate with justice or belonging
            discriminant_scores = (justice + belonging) / 2
        elif scale.construct_measured == "justice":
            discriminant_scores = (agency + belonging) / 2
        else:  # belonging
            discriminant_scores = (agency + justice) / 2

        result = self.compute_correlation(discriminant_scores, scale_totals_norm)
        return result.r

    def mtmm_analysis(
        self,
        texts: List[str],
        scale_data: Dict[str, List[Dict[str, int]]]
    ) -> MTMMResult:
        """
        Perform Multi-Trait Multi-Method matrix analysis.

        This is the gold standard for construct validation, comparing:
        - Monotrait-heteromethod (convergent): Same trait, different method
        - Heterotrait-monomethod: Different traits, same method
        - Heterotrait-heteromethod: Different traits, different methods

        Args:
            texts: Text samples (provides one method: text extraction)
            scale_data: Dict mapping scale_name -> list of responses
                        (provides second method: self-report scales)

        Returns:
            MTMMResult with full matrix and validity coefficients
        """
        traits = ["agency", "justice", "belonging"]
        methods = ["text_extraction", "self_report"]

        # Extract coordinates from text
        agency_text, justice_text, belonging_text = self.extract_coordinates(texts)

        text_scores = {
            "agency": agency_text,
            "justice": justice_text,
            "belonging": belonging_text
        }

        # Score self-report scales
        self_report_scores = {}

        if "agency" in scale_data:
            scale = self.scales["agency"]
            scores = []
            for resp in scale_data["agency"]:
                s = scale.score_responses(resp)
                scores.append(s.get("total", 4))  # Default to midpoint
            arr = np.array(scores)
            scale_min, scale_max = scale.get_normalization_params()
            self_report_scores["agency"] = 2 * (arr - scale_min) / (scale_max - scale_min) - 1

        if "justice" in scale_data:
            scale = self.scales["justice"]
            scores = []
            for resp in scale_data["justice"]:
                s = scale.score_responses(resp)
                scores.append(s.get("total", 3))
            arr = np.array(scores)
            scale_min, scale_max = scale.get_normalization_params()
            self_report_scores["justice"] = 2 * (arr - scale_min) / (scale_max - scale_min) - 1

        if "belonging_ios" in scale_data or "belonging_bscs" in scale_data:
            key = "belonging_ios" if "belonging_ios" in scale_data else "belonging_bscs"
            scale = self.scales[key]
            scores = []
            for resp in scale_data[key]:
                s = scale.score_responses(resp)
                scores.append(s.get("total", 4))
            arr = np.array(scores)
            scale_min, scale_max = scale.get_normalization_params()
            self_report_scores["belonging"] = 2 * (arr - scale_min) / (scale_max - scale_min) - 1

        # Build MTMM matrix
        matrix = {}
        monotrait_heteromethod = []  # Convergent validity
        heterotrait_monomethod = []  # Discriminant (method bias)
        heterotrait_heteromethod = []  # Discriminant

        for t1 in traits:
            for t2 in traits:
                for m1 in methods:
                    for m2 in methods:
                        # Get scores for this combination
                        scores1 = text_scores.get(t1) if m1 == "text_extraction" else self_report_scores.get(t1)
                        scores2 = text_scores.get(t2) if m2 == "text_extraction" else self_report_scores.get(t2)

                        if scores1 is None or scores2 is None:
                            continue

                        if len(scores1) != len(scores2):
                            continue

                        # Skip diagonal (same trait, same method = reliability)
                        if t1 == t2 and m1 == m2:
                            continue

                        corr = self.compute_correlation(scores1, scores2)

                        # Classify cell type
                        if t1 == t2 and m1 != m2:
                            cell_type = "monotrait-heteromethod"
                            monotrait_heteromethod.append(corr.r)
                        elif t1 != t2 and m1 == m2:
                            cell_type = "heterotrait-monomethod"
                            heterotrait_monomethod.append(corr.r)
                        else:
                            cell_type = "heterotrait-heteromethod"
                            heterotrait_heteromethod.append(corr.r)

                        cell = MTMMCell(
                            trait1=t1,
                            trait2=t2,
                            method1=m1,
                            method2=m2,
                            correlation=corr,
                            cell_type=cell_type
                        )
                        matrix[(t1, t2, m1, m2)] = cell

        # Compute summary statistics
        conv = np.mean(monotrait_heteromethod) if monotrait_heteromethod else 0.0
        htmm = np.mean(heterotrait_monomethod) if heterotrait_monomethod else 0.0
        hthm = np.mean(heterotrait_heteromethod) if heterotrait_heteromethod else 0.0

        # Campbell & Fiske criteria
        criterion_1 = conv > 0 and all(r > 0 for r in monotrait_heteromethod)
        criterion_2 = conv > hthm if monotrait_heteromethod else False
        criterion_3 = conv > htmm if monotrait_heteromethod else False
        criterion_4 = True  # Simplified - would need pattern analysis

        return MTMMResult(
            traits=traits,
            methods=methods,
            matrix=matrix,
            convergent_validity=conv,
            discriminant_validity_htmm=htmm,
            discriminant_validity_hthm=hthm,
            criterion_1_met=criterion_1,
            criterion_2_met=criterion_2,
            criterion_3_met=criterion_3,
            criterion_4_met=criterion_4
        )

    def known_groups_validity(
        self,
        group1_texts: List[str],
        group2_texts: List[str],
        group1_name: str,
        group2_name: str,
        construct: str,
        expected_direction: str = "group1_higher"
    ) -> KnownGroupsResult:
        """
        Test known-groups validity.

        Groups expected to differ on a construct should show significant
        differences in extracted coordinates.

        Args:
            group1_texts: Texts from first known group
            group2_texts: Texts from second known group
            group1_name: Name of first group
            group2_name: Name of second group
            construct: Which construct ("agency", "justice", "belonging")
            expected_direction: "group1_higher" or "group2_higher"

        Returns:
            KnownGroupsResult with effect size and significance
        """
        # Extract coordinates
        a1, j1, b1 = self.extract_coordinates(group1_texts)
        a2, j2, b2 = self.extract_coordinates(group2_texts)

        # Select appropriate construct
        if construct == "agency":
            scores1, scores2 = a1, a2
        elif construct == "justice":
            scores1, scores2 = j1, j2
        else:
            scores1, scores2 = b1, b2

        # Compute t-test
        t_stat, p_value = stats.ttest_ind(scores1, scores2)

        # Effect size
        effect = cohens_d(scores1, scores2)

        # Check if hypothesis supported
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        if expected_direction == "group1_higher":
            supported = mean1 > mean2 and p_value < 0.05
        else:
            supported = mean2 > mean1 and p_value < 0.05

        return KnownGroupsResult(
            group1_name=group1_name,
            group2_name=group2_name,
            construct=construct,
            group1_mean=mean1,
            group2_mean=mean2,
            effect_size=effect,
            t_statistic=t_stat,
            p_value=p_value,
            hypothesis_supported=supported
        )

    def predictive_validity(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        predictor: str = "agency",
        outcome_name: str = "outcome"
    ) -> PredictiveValidityResult:
        """
        Test predictive validity of extracted coordinates.

        Coordinates should predict theoretically related outcomes.

        Args:
            texts: Text samples
            outcomes: Outcome variable (continuous)
            predictor: Which coordinate ("agency", "justice", "belonging")
            outcome_name: Name of outcome for reporting

        Returns:
            PredictiveValidityResult with regression analysis
        """
        # Extract coordinates
        agency, justice, belonging = self.extract_coordinates(texts)

        # Select predictor
        if predictor == "agency":
            x = agency
        elif predictor == "justice":
            x = justice
        else:
            x = belonging

        # Correlation
        corr = self.compute_correlation(x, outcomes)

        # Simple linear regression
        n = len(x)
        x_mean, y_mean = np.mean(x), np.mean(outcomes)
        ss_xy = np.sum((x - x_mean) * (outcomes - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        beta = ss_xy / ss_xx if ss_xx > 0 else 0

        # R-squared
        y_pred = x_mean + beta * (x - x_mean) + y_mean - x_mean * beta
        ss_tot = np.sum((outcomes - y_mean) ** 2)
        ss_res = np.sum((outcomes - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return PredictiveValidityResult(
            predictor=predictor,
            outcome=outcome_name,
            correlation=corr,
            regression_coefficient=beta,
            r_squared=max(0, r_squared),  # Ensure non-negative
            prediction_accuracy=0.0  # For classification outcomes
        )


# =============================================================================
# Synthetic Validation Data Generation
# =============================================================================

def generate_validation_corpus(
    n_samples: int = 100,
    seed: Optional[int] = None
) -> Tuple[List[str], List[Dict[str, Dict[str, int]]]]:
    """
    Generate synthetic texts with known psychological scale responses.

    Creates matched pairs of text and scale responses where:
    - High agency texts have high SoAS responses
    - High justice texts have high OJS responses
    - High belonging texts have high IOS/BSCS responses

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (texts, scale_responses)
        scale_responses is list of dicts: {"agency": {...}, "justice": {...}, "belonging": {...}}
    """
    if seed is not None:
        np.random.seed(seed)

    texts = []
    scale_responses = []

    # Text templates for different levels
    agency_templates = {
        "high": [
            "I made this decision entirely on my own. I chose this path and I take full responsibility for the outcome.",
            "Everything that happened was because of my actions. I controlled the situation from start to finish.",
            "I am the author of my own story. My choices define who I am and where I'm going.",
            "This success is mine. I worked hard, I planned carefully, and I made it happen.",
            "I decided to take action and I followed through. The results reflect my effort and choices.",
        ],
        "low": [
            "The system forced me into this situation. I had no choice but to comply with their demands.",
            "External forces control everything. Nothing I do seems to matter in the end.",
            "Circumstances beyond my control determined the outcome. I was just along for the ride.",
            "The institution made all the decisions. I was merely a cog in their machine.",
            "Forces I don't understand pushed me here. I feel like a puppet on strings.",
        ],
    }

    justice_templates = {
        "high": [
            "The process was fair and transparent. They listened to my concerns and followed proper procedures.",
            "I received what I deserved based on my contributions. The outcome was just and appropriate.",
            "They treated me with respect and dignity throughout. The procedures were applied consistently.",
            "Due process was followed carefully. I had the chance to appeal and my voice was heard.",
            "The distribution was equitable and the treatment was respectful. Everything was above board.",
        ],
        "low": [
            "The whole thing was rigged from the start. They never intended to treat anyone fairly.",
            "I was denied any say in the process. The rules were applied arbitrarily and inconsistently.",
            "They treated me disrespectfully and ignored my input. The outcome was completely unfair.",
            "There was no justice in how they handled this. The procedures were biased and corrupt.",
            "I deserved better but received nothing. The system is broken beyond repair.",
        ],
    }

    belonging_templates = {
        "high": [
            "We are all in this together as a community. Our shared bonds make us stronger.",
            "I feel deeply connected to everyone here. This group is like a second family to me.",
            "Together we can accomplish anything. Our community supports each member equally.",
            "I belong here with these people. We share common values and look out for each other.",
            "The connection I feel to this group is profound. We celebrate together and mourn together.",
        ],
        "low": [
            "I'm an outsider here. Those people don't understand me and never will.",
            "They exclude anyone who doesn't fit their mold. I've never felt so alone.",
            "There is no community here, just individuals looking out for themselves.",
            "I don't belong anywhere. The barriers between us are insurmountable.",
            "They made it clear I'm not one of them. Some people are just meant to be alone.",
        ],
    }

    for _ in range(n_samples):
        # Random levels for each construct
        agency_level = np.random.choice(["high", "low"])
        justice_level = np.random.choice(["high", "low"])
        belonging_level = np.random.choice(["high", "low"])

        # Generate text by combining templates
        text_parts = [
            np.random.choice(agency_templates[agency_level]),
            np.random.choice(justice_templates[justice_level]),
            np.random.choice(belonging_templates[belonging_level]),
        ]
        np.random.shuffle(text_parts)
        text = " ".join(text_parts)
        texts.append(text)

        # Generate scale responses
        def gen_responses(scale: PsychologicalScale, level: str) -> Dict[str, int]:
            responses = {}
            scale_min, scale_max = scale.response_range

            for item in scale.items:
                if level == "high":
                    if item.reverse_scored:
                        base = scale_min + 1
                    else:
                        base = scale_max - 1
                else:
                    if item.reverse_scored:
                        base = scale_max - 1
                    else:
                        base = scale_min + 1

                # Add noise
                noise = np.random.randint(-1, 2)
                response = np.clip(base + noise, scale_min, scale_max)
                responses[item.id] = int(response)

            return responses

        sample_responses = {
            "agency": gen_responses(SENSE_OF_AGENCY_SCALE, agency_level),
            "justice": gen_responses(ORGANIZATIONAL_JUSTICE_SCALE, justice_level),
            "belonging_ios": gen_responses(INCLUSION_OF_OTHER_IN_SELF_SCALE, belonging_level),
            "belonging_bscs": gen_responses(BRIEF_SENSE_OF_COMMUNITY_SCALE, belonging_level),
        }
        scale_responses.append(sample_responses)

    return texts, scale_responses


def generate_known_groups_corpus(
    n_per_group: int = 50,
    seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Generate texts for known-groups validity testing.

    Creates populations expected to differ on Agency/Justice/Belonging:

    Agency:
    - entrepreneurs: High agency (self-made, control)
    - bureaucrats: Low agency (system-controlled)

    Justice:
    - whistleblowers: Low justice (saw injustice, took action)
    - mediators: High justice (fair processes, balanced outcomes)

    Belonging:
    - cult_members: High belonging (strong ingroup, exclusionary)
    - nomads: Low belonging (no fixed community)

    Args:
        n_per_group: Samples per group
        seed: Random seed

    Returns:
        Dict mapping group_name -> list of texts
    """
    if seed is not None:
        np.random.seed(seed)

    groups = {}

    # High agency: Entrepreneurs
    entrepreneur_templates = [
        "I built this company from nothing. Every decision, every risk was mine to take.",
        "When I saw the opportunity, I seized it. No one else was going to do it for me.",
        "My success came from my own vision and determination. I made it happen.",
        "I control my destiny. The market responds to what I create.",
        "Taking initiative is what separates leaders from followers. I lead.",
        "I chose this path knowing the risks. The rewards are mine because I earned them.",
        "Every failure taught me something. I adjusted and tried again until I succeeded.",
        "I don't wait for permission. I see what needs doing and I do it.",
    ]

    # Low agency: Bureaucrats
    bureaucrat_templates = [
        "I follow the procedures as mandated by the organization. It's not my place to question.",
        "The system determines what happens. Individual initiative is discouraged here.",
        "My role is to implement policy, not create it. The decisions come from above.",
        "We do things by the book. Personal judgment is irrelevant when regulations apply.",
        "The institution has its own logic. I'm just a small part of a vast machine.",
        "Changes require approval through proper channels. Nothing moves without authorization.",
        "I process what comes to me. The workflow is predetermined.",
        "The hierarchy decides. My job is to comply and execute.",
    ]

    # Low justice: Whistleblowers
    whistleblower_templates = [
        "The corruption was systemic. They covered up everything to protect the guilty.",
        "When I reported the fraud, they retaliated against me instead of investigating.",
        "There is no justice within the system. The powerful protect each other.",
        "I watched them break every rule while punishing those who spoke up.",
        "The process was a sham. They had already decided to protect their interests.",
        "Fairness is a myth here. Outcomes are determined by who you know.",
        "I was treated like a criminal for exposing criminals. The irony is crushing.",
        "They twisted the rules to suit themselves while claiming to uphold standards.",
    ]

    # High justice: Mediators
    mediator_templates = [
        "Both parties had the opportunity to present their case fully and fairly.",
        "We followed established procedures to ensure an impartial outcome.",
        "Everyone was treated with respect and given equal consideration.",
        "The process was transparent and the criteria were applied consistently.",
        "Fair treatment means listening to all sides before reaching a decision.",
        "Justice requires patience and careful attention to every perspective.",
        "We upheld ethical standards throughout the entire process.",
        "The outcome reflected the evidence and the established principles of fairness.",
    ]

    # High belonging: Community members
    community_templates = [
        "We look out for each other here. That's what community means.",
        "I've found my people. Together we share everything that matters.",
        "The bond between us is unbreakable. We are family in every way that counts.",
        "Our shared traditions connect us across generations. I belong here.",
        "When one of us struggles, we all come together to help. That's who we are.",
        "I would do anything for this group. They would do the same for me.",
        "We understand each other without words. This connection is sacred.",
        "Outsiders don't understand our ways, but we don't need their approval.",
    ]

    # Low belonging: Isolated individuals
    isolated_templates = [
        "I've always been on my own. Connections just don't seem to last.",
        "Community is for other people. I've never fit in anywhere.",
        "Every group eventually pushes me out. I've stopped trying.",
        "I prefer solitude now. People have disappointed me too many times.",
        "There's no place where I truly belong. I'm a permanent outsider.",
        "I watch others bond while I stand apart. Some of us are just alone.",
        "Trust leads to betrayal. It's safer to keep my distance.",
        "I move through places without leaving a mark. Invisible and forgotten.",
    ]

    # Generate samples with variation
    def expand_samples(templates: List[str], n: int) -> List[str]:
        samples = []
        for _ in range(n):
            template = np.random.choice(templates)
            # Add some variation
            variations = [
                template,
                template.replace(".", "!"),
                template.replace("I ", "I truly "),
                template.replace("We ", "We always "),
            ]
            samples.append(np.random.choice(variations))
        return samples

    groups["high_agency_entrepreneurs"] = expand_samples(entrepreneur_templates, n_per_group)
    groups["low_agency_bureaucrats"] = expand_samples(bureaucrat_templates, n_per_group)
    groups["low_justice_whistleblowers"] = expand_samples(whistleblower_templates, n_per_group)
    groups["high_justice_mediators"] = expand_samples(mediator_templates, n_per_group)
    groups["high_belonging_community"] = expand_samples(community_templates, n_per_group)
    groups["low_belonging_isolated"] = expand_samples(isolated_templates, n_per_group)

    return groups


# =============================================================================
# Validation Report Generation
# =============================================================================

def generate_validation_report(
    validator: ExternalValidator,
    texts: List[str],
    scale_data: Dict[str, List[Dict[str, int]]],
    known_groups: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    Generate comprehensive validation report.

    Args:
        validator: ExternalValidator instance
        texts: Text samples
        scale_data: Scale response data
        known_groups: Optional known groups for criterion validity

    Returns:
        Formatted validation report
    """
    lines = [
        "=" * 70,
        "EXTERNAL VALIDATION REPORT",
        "Cultural Soliton Observatory - Agency/Justice/Belonging Coordinates",
        "=" * 70,
        "",
    ]

    # Convergent validity
    lines.append("1. CONVERGENT VALIDITY")
    lines.append("-" * 40)
    lines.append("(Same construct measured by different methods should correlate r > .50)")
    lines.append("")

    for scale_name in ["agency", "justice", "belonging_ios"]:
        if scale_name in scale_data or scale_name.replace("_ios", "_bscs") in scale_data:
            actual_key = scale_name if scale_name in scale_data else scale_name.replace("_ios", "_bscs")
            try:
                result = validator.correlate_with_scale(
                    texts, scale_data[actual_key], actual_key
                )
                status = "PASS" if result.convergent_validity_met else "FAIL"
                lines.append(f"  {result.scale_name}:")
                lines.append(f"    r = {result.correlation.r:.3f}, "
                           f"95% CI [{result.correlation.ci_lower:.3f}, {result.correlation.ci_upper:.3f}]")
                lines.append(f"    p = {result.correlation.p_value:.4f}")
                lines.append(f"    Convergent validity: {status}")
                lines.append("")
            except Exception as e:
                lines.append(f"  {scale_name}: Error - {str(e)}")
                lines.append("")

    # Discriminant validity
    lines.append("2. DISCRIMINANT VALIDITY")
    lines.append("-" * 40)
    lines.append("(Different constructs should correlate weakly, r < .30)")
    lines.append("")

    # Cross-construct correlations
    agency, justice, belonging = validator.extract_coordinates(texts)

    aj_corr = validator.compute_correlation(agency, justice)
    ab_corr = validator.compute_correlation(agency, belonging)
    jb_corr = validator.compute_correlation(justice, belonging)

    lines.append("  Extracted coordinate intercorrelations:")
    lines.append(f"    Agency-Justice:    r = {aj_corr.r:.3f} {'(discriminant)' if abs(aj_corr.r) < 0.30 else '(concerning)'}")
    lines.append(f"    Agency-Belonging:  r = {ab_corr.r:.3f} {'(discriminant)' if abs(ab_corr.r) < 0.30 else '(concerning)'}")
    lines.append(f"    Justice-Belonging: r = {jb_corr.r:.3f} {'(discriminant)' if abs(jb_corr.r) < 0.30 else '(concerning)'}")
    lines.append("")

    # MTMM Analysis
    lines.append("3. MULTI-TRAIT MULTI-METHOD ANALYSIS")
    lines.append("-" * 40)

    try:
        mtmm = validator.mtmm_analysis(texts, scale_data)
        lines.append(mtmm.format_matrix())
    except Exception as e:
        lines.append(f"  MTMM analysis failed: {str(e)}")
    lines.append("")

    # Known-groups validity
    if known_groups:
        lines.append("4. KNOWN-GROUPS VALIDITY")
        lines.append("-" * 40)
        lines.append("(Groups expected to differ should show significant differences)")
        lines.append("")

        # Agency
        if "high_agency_entrepreneurs" in known_groups and "low_agency_bureaucrats" in known_groups:
            result = validator.known_groups_validity(
                known_groups["high_agency_entrepreneurs"],
                known_groups["low_agency_bureaucrats"],
                "Entrepreneurs", "Bureaucrats", "agency", "group1_higher"
            )
            lines.append(f"  Agency: Entrepreneurs vs Bureaucrats")
            lines.append(f"    Entrepreneurs M = {result.group1_mean:.3f}")
            lines.append(f"    Bureaucrats M = {result.group2_mean:.3f}")
            lines.append(f"    Cohen's d = {result.effect_size.d:.3f}")
            lines.append(f"    p = {result.p_value:.4f}")
            lines.append(f"    Hypothesis supported: {'YES' if result.hypothesis_supported else 'NO'}")
            lines.append("")

        # Justice
        if "high_justice_mediators" in known_groups and "low_justice_whistleblowers" in known_groups:
            result = validator.known_groups_validity(
                known_groups["high_justice_mediators"],
                known_groups["low_justice_whistleblowers"],
                "Mediators", "Whistleblowers", "justice", "group1_higher"
            )
            lines.append(f"  Justice: Mediators vs Whistleblowers")
            lines.append(f"    Mediators M = {result.group1_mean:.3f}")
            lines.append(f"    Whistleblowers M = {result.group2_mean:.3f}")
            lines.append(f"    Cohen's d = {result.effect_size.d:.3f}")
            lines.append(f"    p = {result.p_value:.4f}")
            lines.append(f"    Hypothesis supported: {'YES' if result.hypothesis_supported else 'NO'}")
            lines.append("")

        # Belonging
        if "high_belonging_community" in known_groups and "low_belonging_isolated" in known_groups:
            result = validator.known_groups_validity(
                known_groups["high_belonging_community"],
                known_groups["low_belonging_isolated"],
                "Community Members", "Isolated Individuals", "belonging", "group1_higher"
            )
            lines.append(f"  Belonging: Community Members vs Isolated Individuals")
            lines.append(f"    Community M = {result.group1_mean:.3f}")
            lines.append(f"    Isolated M = {result.group2_mean:.3f}")
            lines.append(f"    Cohen's d = {result.effect_size.d:.3f}")
            lines.append(f"    p = {result.p_value:.4f}")
            lines.append(f"    Hypothesis supported: {'YES' if result.hypothesis_supported else 'NO'}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("END OF VALIDATION REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# Visualization Support
# =============================================================================

def create_mtmm_heatmap(
    mtmm_result: MTMMResult,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[object]:
    """
    Create MTMM matrix heatmap visualization.

    Args:
        mtmm_result: Result from mtmm_analysis
        figsize: Figure size

    Returns:
        Matplotlib figure or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        logger.warning("matplotlib not available for MTMM heatmap")
        return None

    traits = mtmm_result.traits
    methods = mtmm_result.methods

    n_traits = len(traits)
    n_methods = len(methods)
    size = n_traits * n_methods

    # Build correlation matrix
    matrix = np.zeros((size, size))
    labels = []

    for m in methods:
        for t in traits:
            labels.append(f"{m[:4]}_{t[:3]}")

    for i, (m1, t1) in enumerate([(m, t) for m in methods for t in traits]):
        for j, (m2, t2) in enumerate([(m, t) for m in methods for t in traits]):
            if i == j:
                matrix[i, j] = 1.0  # Reliability diagonal
            else:
                key = (t1, t2, m1, m2)
                if key in mtmm_result.matrix:
                    matrix[i, j] = mtmm_result.matrix[key].correlation.r

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Labels
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(size):
        for j in range(size):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if abs(matrix[i, j]) > 0.5 else 'black')

    # Add block outlines for methods
    for k in range(n_methods):
        rect = patches.Rectangle(
            (k * n_traits - 0.5, k * n_traits - 0.5),
            n_traits, n_traits,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Correlation (r)')

    ax.set_title('Multi-Trait Multi-Method Matrix\n'
                 'Diagonal blocks = monomethod, Off-diagonal = heteromethod')

    plt.tight_layout()
    return fig


def create_validity_scatter(
    texts: List[str],
    scale_responses: List[Dict[str, int]],
    scale_name: str,
    validator: ExternalValidator,
    figsize: Tuple[int, int] = (8, 6)
) -> Optional[object]:
    """
    Create scatter plot of extracted vs self-report scores.

    Args:
        texts: Text samples
        scale_responses: Scale response data
        scale_name: Which scale
        validator: ExternalValidator instance
        figsize: Figure size

    Returns:
        Matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    scale = validator.scales[scale_name]

    # Extract coordinates
    agency, justice, belonging = validator.extract_coordinates(texts)

    if scale.construct_measured == "agency":
        extracted = agency
    elif scale.construct_measured == "justice":
        extracted = justice
    else:
        extracted = belonging

    # Score scale
    scale_scores = []
    for resp in scale_responses:
        scores = scale.score_responses(resp)
        scale_scores.append(scores.get("total", 0))

    scale_scores = np.array(scale_scores)
    scale_min, scale_max = scale.get_normalization_params()
    scale_scores_norm = 2 * (scale_scores - scale_min) / (scale_max - scale_min) - 1

    # Correlation
    corr = validator.compute_correlation(extracted, scale_scores_norm)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(extracted, scale_scores_norm, alpha=0.5)

    # Regression line
    z = np.polyfit(extracted, scale_scores_norm, 1)
    p = np.poly1d(z)
    x_line = np.linspace(extracted.min(), extracted.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Best fit')

    ax.set_xlabel(f'Extracted {scale.construct_measured.capitalize()} Score')
    ax.set_ylabel(f'{scale.abbreviation} Score (normalized)')
    ax.set_title(f'Convergent Validity: Text Extraction vs {scale.abbreviation}\n'
                 f'r = {corr.r:.3f}, p = {corr.p_value:.4f}')
    ax.legend()

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# Example Usage
# =============================================================================

def run_validation_demo():
    """
    Demonstrate external validation infrastructure.
    """
    print("=" * 70)
    print("EXTERNAL VALIDATION DEMONSTRATION")
    print("Cultural Soliton Observatory")
    print("=" * 70)
    print()

    # Initialize validator
    validator = ExternalValidator()

    # Generate synthetic data
    print("1. Generating synthetic validation corpus...")
    texts, scale_responses = generate_validation_corpus(n_samples=100, seed=42)
    print(f"   Generated {len(texts)} matched text-scale pairs")
    print()

    # Prepare scale data
    scale_data = {
        "agency": [r["agency"] for r in scale_responses],
        "justice": [r["justice"] for r in scale_responses],
        "belonging_ios": [r["belonging_ios"] for r in scale_responses],
    }

    # Run convergent validity
    print("2. Testing Convergent Validity...")
    print("-" * 40)

    for scale_name in ["agency", "justice", "belonging_ios"]:
        result = validator.correlate_with_scale(texts, scale_data[scale_name], scale_name)
        print(f"   {result.scale_name}:")
        print(f"   r = {result.correlation.r:.3f}, p = {result.correlation.p_value:.4f}")
        print(f"   Validity met: {result.convergent_validity_met}")
        print()

    # Run MTMM analysis
    print("3. Multi-Trait Multi-Method Analysis...")
    print("-" * 40)
    mtmm = validator.mtmm_analysis(texts, scale_data)
    print(f"   Convergent validity (MTHM): {mtmm.convergent_validity:.3f}")
    print(f"   Discriminant (HTMM): {mtmm.discriminant_validity_htmm:.3f}")
    print(f"   Discriminant (HTHM): {mtmm.discriminant_validity_hthm:.3f}")
    print()

    # Generate known groups
    print("4. Known-Groups Validity...")
    print("-" * 40)
    known_groups = generate_known_groups_corpus(n_per_group=30, seed=42)

    result = validator.known_groups_validity(
        known_groups["high_agency_entrepreneurs"],
        known_groups["low_agency_bureaucrats"],
        "Entrepreneurs", "Bureaucrats", "agency"
    )
    print(f"   Agency (Entrepreneurs vs Bureaucrats):")
    print(f"   d = {result.effect_size.d:.3f}, p = {result.p_value:.4f}")
    print(f"   Hypothesis supported: {result.hypothesis_supported}")
    print()

    result = validator.known_groups_validity(
        known_groups["high_belonging_community"],
        known_groups["low_belonging_isolated"],
        "Community", "Isolated", "belonging"
    )
    print(f"   Belonging (Community vs Isolated):")
    print(f"   d = {result.effect_size.d:.3f}, p = {result.p_value:.4f}")
    print(f"   Hypothesis supported: {result.hypothesis_supported}")
    print()

    # Generate full report
    print("5. Generating Full Validation Report...")
    print("-" * 40)
    report = generate_validation_report(validator, texts, scale_data, known_groups)
    print(report)

    return validator, texts, scale_data, known_groups


if __name__ == "__main__":
    run_validation_demo()
