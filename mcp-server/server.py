#!/usr/bin/env python3
"""
Observatory MCP Server

MCP server providing tools for analyzing cultural narratives, coordination
patterns, and AI behavior in embedding space.

Core Tools:
  - project_text: Project text onto the 3D coordination manifold
  - analyze_corpus: Batch analyze texts with clustering
  - compare_projections: Compare projection methods on the same text
  - get_manifold_state: Get current state of all projections
  - add_training_example: Contribute to projection calibration

AI Behavior Analysis:
  - analyze_ai_text: Analyze AI-generated text for behavior patterns
  - analyze_ai_conversation: Track behavior dynamics across turns
  - monitor_ai_safety: Real-time safety monitoring
  - detect_ai_anomalies: Detect anomalous AI behavior

Coordination Analysis:
  - measure_cbr: Measure Coordination Background Radiation
  - check_ossification: Check for protocol ossification risk
  - telescope_observe: Full 18D hierarchical coordinate extraction
  - detect_gaming: Detect coordination gaming attempts

Advanced Analytics:
  - compare_narratives: Gap analysis between groups
  - track_trajectory: Track narrative evolution over time
  - detect_outlier: Detect anomalous narratives
  - analyze_cohorts: Multi-group analysis

Run locally (stdio):
    python server.py

Run with MCP inspector:
    npx @modelcontextprotocol/inspector python server.py

For remote access (SSE), use server_sse.py instead.
"""

import asyncio
import json
import os
import sys
from typing import Any, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from observatory_client import ObservatoryClient, ObservatoryConfig

# Import narrative analysis module
from narrative_analysis import (
    fetch_narrative_source,
    build_narrative_profile,
    get_narrative_suggestions,
    format_profile_result,
    format_suggestions_result,
)
_NARRATIVE_ANALYSIS_AVAILABLE = True

# Add backend research path for advanced feature imports
_backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "research")
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

# Graceful imports for advanced research features
_PARSED_EXTRACTION_AVAILABLE = False
_SEMANTIC_EXTRACTION_AVAILABLE = False
_EXTERNAL_VALIDATION_AVAILABLE = False
_SAFETY_METRICS_AVAILABLE = False

try:
    from parsed_feature_extraction import (
        parse_text,
        extract_features_detailed,
        compare_extraction_methods,
    )
    _PARSED_EXTRACTION_AVAILABLE = True
except ImportError as e:
    parse_text = None
    extract_features_detailed = None
    compare_extraction_methods = None

try:
    from semantic_feature_extraction import (
        SemanticFeatureExtractor,
        compare_regex_vs_semantic,
    )
    _SEMANTIC_EXTRACTION_AVAILABLE = True
except ImportError as e:
    SemanticFeatureExtractor = None
    compare_regex_vs_semantic = None

try:
    from external_validation import (
        ExternalValidator,
        generate_validation_corpus,
        generate_known_groups_corpus,
        VALIDATION_SCALES,
    )
    _EXTERNAL_VALIDATION_AVAILABLE = True
except ImportError as e:
    ExternalValidator = None
    generate_validation_corpus = None
    generate_known_groups_corpus = None
    VALIDATION_SCALES = None

try:
    from safety_metrics import (
        SafetyMetricsEvaluator,
        AdversarialTester,
        generate_labeled_test_corpus,
        assess_deployment_readiness,
        GroundTruthCorpus,
        LabeledSample,
    )
    _SAFETY_METRICS_AVAILABLE = True
except ImportError as e:
    SafetyMetricsEvaluator = None
    AdversarialTester = None
    generate_labeled_test_corpus = None
    assess_deployment_readiness = None
    GroundTruthCorpus = None
    LabeledSample = None


# --- Configuration ---

BACKEND_URL = os.environ.get("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("OBSERVATORY_DEFAULT_MODEL", "all-MiniLM-L6-v2")


# --- Narrative Mode Reference ---

NARRATIVE_MODES = {
    "positive": {
        "description": "High agency, high perceived justice - aligned with espoused values",
        "examples": ["growth mindset", "civic idealism", "faithful zeal"],
    },
    "shadow": {
        "description": "High agency, low perceived justice - cynical exploitation",
        "examples": ["cynical burnout", "institutional decay", "schismatic doubt"],
    },
    "exit": {
        "description": "Low agency, variable perceived justice - withdrawal/disengagement",
        "examples": ["quiet quitting", "grid exit", "apostasy"],
    },
    "noise": {
        "description": "Low signal - distraction, irrelevance",
        "examples": ["watercooler chatter", "distraction", "secular drift"],
    },
}


# --- MCP Server ---

server = Server("cultural-soliton-observatory")


def format_projection_result(result: dict, include_soft_labels: bool = False) -> str:
    """Format a projection result for display."""
    coords = result.get("coordinates", result.get("vector", {}))
    # Handle both "fairness" and "perceived_justice" for backward compatibility
    perceived_justice = coords.get('perceived_justice', coords.get('fairness', 0))

    # Extract mode info - could be a dict (v2) or string (v1)
    mode_info = result.get('mode', 'unknown')
    if isinstance(mode_info, dict):
        primary_mode = mode_info.get('primary_mode', 'unknown')
        confidence = mode_info.get('confidence', 0)
        stability_score = mode_info.get('stability_score')
        is_boundary_case = mode_info.get('is_boundary_case', False)
        stability_warning = mode_info.get('stability_warning')
        secondary_mode = mode_info.get('secondary_mode')
        primary_prob = mode_info.get('primary_probability')
        secondary_prob = mode_info.get('secondary_probability')
    else:
        primary_mode = mode_info
        confidence = result.get('confidence', 0)
        stability_score = None
        is_boundary_case = False
        stability_warning = None
        secondary_mode = None
        primary_prob = None
        secondary_prob = None

    output = f"""
## Projection Result

**Text**: "{result.get('text', 'N/A')[:100]}..."

**Coordinates** (Cultural Manifold):
- Agency: {coords.get('agency', 0):.3f} (-2 to +2)
- Perceived Justice: {perceived_justice:.3f} (-2 to +2)
- Belonging: {coords.get('belonging', 0):.3f} (-2 to +2)

**Classification**:
- Primary Mode: {primary_mode}
- Confidence: {confidence:.1%}"""

    if secondary_mode and primary_prob:
        output += f"""
- Secondary Mode: {secondary_mode}
- Primary Probability: {primary_prob:.1%}
- Secondary Probability: {secondary_prob:.1%}"""

    if stability_score is not None:
        output += f"""

**Stability**:
- Stability Score: {stability_score:.2f}
- Boundary Case: {'Yes' if is_boundary_case else 'No'}"""
        if stability_warning:
            output += f"""
- Warning: {stability_warning}"""

    output += f"""

**Interpretation**:
{interpret_coordinates(coords.get('agency', 0), perceived_justice, coords.get('belonging', 0))}
"""

    # Include soft labels if requested
    if include_soft_labels and isinstance(mode_info, dict):
        all_probs = mode_info.get('all_probabilities', {})
        if all_probs:
            output += "\n**Soft Labels** (probability per mode):\n"
            # Sort by probability descending
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for mode_name, prob in sorted_probs[:5]:  # Show top 5
                output += f"- {mode_name}: {prob:.1%}\n"

    return output


def interpret_coordinates(agency: float, perceived_justice: float, belonging: float) -> str:
    """Generate human-readable interpretation of coordinates.

    Note: The second axis was renamed from "fairness" to "perceived_justice" in Jan 2026.
    """
    parts = []

    # Agency interpretation
    if agency > 1.0:
        parts.append("Strong sense of empowerment and self-determination")
    elif agency > 0.3:
        parts.append("Moderate agency and autonomy")
    elif agency < -1.0:
        parts.append("Fatalistic or powerless framing")
    elif agency < -0.3:
        parts.append("Limited sense of control")
    else:
        parts.append("Neutral agency")

    # Perceived Justice interpretation (formerly "fairness")
    if perceived_justice > 1.0:
        parts.append("strong belief in system legitimacy and fair treatment")
    elif perceived_justice > 0.3:
        parts.append("moderate trust in fair treatment")
    elif perceived_justice < -1.0:
        parts.append("perception of systemic injustice/corruption")
    elif perceived_justice < -0.3:
        parts.append("skepticism about fair treatment")
    else:
        parts.append("neutral on perceived justice")

    # Belonging interpretation
    if belonging > 1.0:
        parts.append("deep connection to community/institution")
    elif belonging > 0.3:
        parts.append("moderate group alignment")
    elif belonging < -1.0:
        parts.append("strong alienation or outsider positioning")
    elif belonging < -0.3:
        parts.append("some disconnection from group")
    else:
        parts.append("neutral belonging")

    return ". ".join(parts) + "."


# --- Tool Definitions ---

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available observatory tools."""
    tools = [
        Tool(
            name="project_text",
            description="""Project a text onto the 3D cultural manifold.

Returns coordinates (agency, perceived_justice, belonging) representing where
the text lands in cultural-semantic space, along with its narrative
mode classification and confidence.

The three axes encode:
- Agency: Self-determination vs fatalism (-2 to +2)
- Perceived Justice: System legitimacy/fair treatment vs corruption (-2 to +2)
- Belonging: Community connection vs alienation (-2 to +2)

Now uses the v2 API which returns:
- Enhanced 12-mode classification with soft labels (probability per mode)
- Stability indicators (stability score, boundary case detection)
- Primary and secondary modes with probabilities

Use this to understand the cultural positioning of any text.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to project onto the manifold",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Embedding model to use (default: all-MiniLM-L6-v2)",
                    },
                    "include_soft_labels": {
                        "type": "boolean",
                        "description": "Include probability distribution across all modes (default: false)",
                        "default": False,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="analyze_corpus",
            description="""Analyze a collection of texts with clustering detection.

Projects all texts onto the manifold and identifies narrative clusters
(stable groupings). Returns per-text coordinates, detected clusters with
their modes and centroids, and overall mode distribution.

Use this to understand the narrative landscape of a document collection,
social media discourse, organizational communications, etc.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to analyze",
                    },
                    "detect_clusters": {
                        "type": "boolean",
                        "description": "Whether to detect narrative clusters (default: true)",
                        "default": True,
                    },
                },
                "required": ["texts"],
            },
        ),
        Tool(
            name="compare_projections",
            description="""Compare how a text is projected by different methods.

Tests the same text against multiple projection methods (ridge, GP, neural)
to understand projection stability and uncertainty. High variance across
methods suggests the text is ambiguous or at a boundary between modes.

Use this to validate important projections or investigate edge cases.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to compare projections for",
                    },
                    "methods": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["ridge", "gp", "neural"]},
                        "description": "Projection methods to compare (default: all)",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="generate_probe",
            description="""Generate a text predicted to land at specific coordinates.

Given target (agency, perceived_justice, belonging) coordinates, generates a
probe text that should project near those coordinates. Useful for:
- Testing projection boundaries
- Understanding what narratives occupy specific regions
- Adversarial probing of the manifold

Returns the generated text and its actual projection.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_agency": {
                        "type": "number",
                        "description": "Target agency coordinate (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "target_perceived_justice": {
                        "type": "number",
                        "description": "Target perceived justice coordinate (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "target_fairness": {
                        "type": "number",
                        "description": "[DEPRECATED: use target_perceived_justice] Target fairness coordinate (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "target_belonging": {
                        "type": "number",
                        "description": "Target belonging coordinate (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["corporate", "government", "religion", "general"],
                        "description": "Domain context for generation (default: general)",
                    },
                },
                "required": ["target_agency", "target_belonging"],
            },
        ),
        Tool(
            name="get_manifold_state",
            description="""Get the current state of the observatory.

Returns information about:
- Loaded embedding models
- Projection training status and metrics
- Number of training examples
- Any cached projections

Use this to understand what the observatory is currently configured for.""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="add_training_example",
            description="""Add a labeled example to improve projection calibration.

Contributes a (text, agency, perceived_justice, belonging) tuple to the training
set. After adding examples, the projection can be retrained for improved
accuracy.

Guidelines for labeling:
- Agency: How much self-determination/control is expressed? (-2 fatalistic, +2 empowered)
- Perceived Justice: Is fair treatment/system legitimacy portrayed? (-2 corrupt, +2 just)
- Belonging: Connection to community/institution? (-2 alienated, +2 embedded)

Use this to calibrate the projection for specific domains or correct errors.

Note: 'fairness' parameter is accepted for backward compatibility but 'perceived_justice' is preferred.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The example text",
                    },
                    "agency": {
                        "type": "number",
                        "description": "Agency score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "perceived_justice": {
                        "type": "number",
                        "description": "Perceived justice score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "fairness": {
                        "type": "number",
                        "description": "[DEPRECATED: use perceived_justice] Fairness score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                    "belonging": {
                        "type": "number",
                        "description": "Belonging score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2,
                    },
                },
                "required": ["text", "agency", "belonging"],
            },
        ),
        Tool(
            name="explain_modes",
            description="""Explain the narrative mode classification system.

Returns detailed descriptions of the four narrative modes:
- Positive: High agency + high perceived justice (aligned with espoused values)
- Shadow: High agency + low perceived justice (cynical exploitation)
- Exit: Low agency (withdrawal, disengagement)
- Noise: Low signal (distraction, irrelevance)

Use this to understand how texts are classified and what each mode means.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["positive", "shadow", "exit", "noise", "all"],
                        "description": "Specific mode to explain, or 'all' for overview",
                    },
                },
            },
        ),
        Tool(
            name="detect_hypocrisy",
            description="""Analyze potential hypocrisy gap in organizational texts.

Compares stated/espoused values against the inferred values from actual
communications. A high delta (Δw) suggests misalignment between what an
organization claims to value and what its language reveals.

Input should include both mission/values statements and operational
communications (emails, memos, policies, etc.).

Returns the hypocrisy gap score and interpretation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "espoused_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Texts representing stated values (mission statements, etc.)",
                    },
                    "operational_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Texts from actual operations (emails, policies, etc.)",
                    },
                },
                "required": ["espoused_texts", "operational_texts"],
            },
        ),
        Tool(
            name="run_scenario",
            description="""Simulate narrative evolution through a scenario.

Runs a scenario simulation where narratives (solitons) evolve through
multiple eras with different reward structures. Tracks which narratives
survive, which die, and how they transform.

Pre-built scenarios:
- market_disruption: Corporate response to competitive threat
- polarization: Political discourse under stress
- reformation: Religious institution facing challenge

Returns trajectory data and survival analysis.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "scenario_id": {
                        "type": "string",
                        "description": "Pre-built scenario ID or 'custom'",
                    },
                    "initial_narratives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Starting narrative texts (for custom scenarios)",
                    },
                    "era_configs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "agency_weight": {"type": "number"},
                                "perceived_justice_weight": {"type": "number"},
                                "belonging_weight": {"type": "number"},
                            },
                        },
                        "description": "Era configurations (for custom scenarios)",
                    },
                },
                "required": ["scenario_id"],
            },
        ),
        # --- New Projection Mode Management Tools (v2 API) ---
        Tool(
            name="list_projection_modes",
            description="""List available projection modes with their characteristics.

Returns information about each projection configuration:
- current_projection: Default MiniLM-based (CV: 0.383) - fastest
- mpnet_projection: Best accuracy with all-mpnet-base-v2 (CV: 0.612)
- multi_model_ensemble: Best robustness, averages 3 embedding models
- ensemble_projection: 25-model bootstrap ensemble for uncertainty quantification

Each mode offers different trade-offs:
- Use current_projection for fast, simple analysis
- Use mpnet_projection for best accuracy
- Use multi_model_ensemble for most robust results
- Use ensemble_projection when you need confidence intervals

Use this to understand what projection options are available.""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="select_projection_mode",
            description="""Switch the active projection mode used for analysis.

Available modes:
- current_projection: Default MiniLM (fastest, CV: 0.383)
- mpnet_projection: Best accuracy (CV: 0.612)
- multi_model_ensemble: Best robustness (averages 3 models)
- ensemble_projection: Uncertainty quantification (25 models)

After selecting a mode, subsequent project_text and analyze_with_uncertainty
calls will use the selected projection. Mode selection persists until changed.

Note: Selecting multi_model_ensemble requires loading 3 models which may
take a moment on first use.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["current_projection", "mpnet_projection", "multi_model_ensemble", "ensemble_projection"],
                        "description": "The projection mode to activate",
                    },
                },
                "required": ["mode"],
            },
        ),
        Tool(
            name="analyze_with_uncertainty",
            description="""Analyze text using ensemble projection for uncertainty quantification.

Returns coordinates plus 95% confidence intervals for each axis, allowing
you to understand how confident the projection is.

The response includes:
- Coordinates (agency, perceived_justice, belonging)
- Standard deviation per axis
- 95% confidence intervals for each axis
- Overall confidence score
- Mode classification with soft labels

Use this when you need to know not just WHERE a text lands, but also
HOW CERTAIN the projection is. Particularly useful for:
- Texts that might straddle multiple modes
- High-stakes classifications
- Research requiring uncertainty quantification

Note: Requires ensemble_projection mode to be available.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze with uncertainty",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Embedding model to use (default: all-MiniLM-L6-v2)",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="get_soft_labels",
            description="""Get full probability distribution across all 12 narrative modes.

Unlike simple mode classification which returns a single label, this returns
soft labels (probabilities) for EVERY mode, allowing nuanced interpretation
of texts that straddle multiple narrative categories.

The 12 modes are organized into 4 categories:
- POSITIVE: Growth Mindset, Civic Idealism, Faithful Zeal (high agency + high justice)
- SHADOW: Cynical Burnout, Institutional Decay, Schismatic Doubt (high agency + low justice)
- EXIT: Quiet Quitting, Grid Exit, Apostasy (low agency, withdrawal)
- AMBIVALENT: Conflicted, Transitional, Neutral (mixed signals)

Response includes:
- primary_mode: Most likely mode
- secondary_mode: Second most likely
- all_probabilities: Full distribution across all 12 modes
- is_boundary_case: True if near a mode boundary
- stability_score: How stable the classification is

Use this to understand complex narratives that don't fit neatly into one mode.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to get soft labels for",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Embedding model to use (default: all-MiniLM-L6-v2)",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="get_current_projection_mode",
            description="""Get the currently active projection mode.

Returns the name and details of the projection mode currently in use,
including required models and performance characteristics.

Use this to check which projection configuration is active before analysis.""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # --- Comparative Analysis Tools (v2 API) ---
        Tool(
            name="compare_narratives",
            description="""Compare two groups of texts and get gap analysis.

Computes centroids for each group and measures the distance between them
across all three axes (agency, perceived_justice, belonging). This reveals
systematic differences in how two populations frame their narratives.

Use cases:
- Compare leadership communications vs employee communications
- Compare marketing materials vs internal memos
- Compare pre-change vs post-change organizational messaging
- Detect hypocrisy gaps between espoused and operational values

Returns group statistics, mode distributions, and a detailed gap analysis
with interpretation of what the differences mean.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_a_name": {
                        "type": "string",
                        "description": "Label for the first group (e.g., 'leadership', 'marketing')",
                    },
                    "group_a_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts from group A",
                    },
                    "group_b_name": {
                        "type": "string",
                        "description": "Label for the second group (e.g., 'employees', 'operations')",
                    },
                    "group_b_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts from group B",
                    },
                },
                "required": ["group_a_name", "group_a_texts", "group_b_name", "group_b_texts"],
            },
        ),
        Tool(
            name="track_trajectory",
            description="""Track narrative evolution over timestamped texts.

Analyzes how narratives shift in the cultural manifold over time, detecting
trend direction, velocity, and inflection points. This reveals whether an
organization's narrative is drifting toward shadow modes, stabilizing, or
transforming.

Use cases:
- Track organizational culture change over quarters/years
- Monitor public discourse evolution during events
- Detect early warning signs of narrative drift toward cynicism
- Measure effectiveness of cultural interventions

Requires texts with associated timestamps to compute trajectory dynamics.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Label for this trajectory (e.g., 'company_comms_2024')",
                    },
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts in chronological order",
                    },
                    "timestamps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ISO timestamps (e.g., '2024-01-15T10:30:00Z') for each text",
                    },
                },
                "required": ["name", "texts", "timestamps"],
            },
        ),
        # --- Alert System Tools (v2 API) ---
        Tool(
            name="create_alert",
            description="""Create an alert rule for monitoring narrative drift.

Sets up automated monitoring that triggers when specified conditions are met.
Alerts persist and can be checked against new texts as they arrive.

Alert types:
- gap_threshold: Trigger when gap between groups exceeds a threshold
- mode_shift: Trigger when dominant mode changes (e.g., positive to shadow)
- trajectory_velocity: Trigger on rapid movement in the manifold
- boundary_crossing: Trigger when texts cross mode boundaries

Use cases:
- Monitor for early signs of cultural decay
- Alert when employee sentiment diverges from leadership messaging
- Track whether interventions are working
- Detect sudden narrative shifts during crises""",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for this alert rule",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["gap_threshold", "mode_shift", "trajectory_velocity", "boundary_crossing"],
                        "description": "Type of alert to create",
                    },
                    "config": {
                        "type": "object",
                        "description": "Alert-specific configuration (e.g., {threshold: 0.5, axis: 'agency'})",
                    },
                },
                "required": ["name", "type", "config"],
            },
        ),
        Tool(
            name="check_alerts",
            description="""Check texts against all configured alert rules.

Projects both groups of texts and evaluates all active alerts to determine
which (if any) have been triggered. Returns details on any triggered alerts
including the specific values that caused the trigger.

Use cases:
- Periodic monitoring of organizational communications
- Real-time checking of incoming content
- Automated early warning system for narrative drift
- Dashboard integration for continuous monitoring""",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_a_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "First group of texts to check",
                    },
                    "group_b_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Second group of texts to check",
                    },
                },
                "required": ["group_a_texts", "group_b_texts"],
            },
        ),
        # --- Advanced Analytics Tools (v2 API) ---
        Tool(
            name="detect_outlier",
            description="""Detect anomalous narratives by comparing a test text against a corpus.

Projects the corpus to establish baseline statistics (centroid, variance),
then evaluates how far the test text deviates. Uses Mahalanobis distance
and z-scores to identify statistically unusual narratives.

Use cases:
- Flag communications that deviate from organizational norms
- Identify unusual customer feedback that needs attention
- Detect potential bad actors in community discussions
- Find texts that might indicate emerging issues

Returns outlier score, per-axis z-scores, and interpretation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "corpus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts establishing the baseline norm",
                    },
                    "test_text": {
                        "type": "string",
                        "description": "Text to evaluate for anomaly",
                    },
                },
                "required": ["corpus", "test_text"],
            },
        ),
        Tool(
            name="analyze_cohorts",
            description="""Multi-group cohort analysis with ANOVA.

Compares multiple groups simultaneously using statistical tests to determine
if there are significant differences between cohorts. Goes beyond pairwise
comparison to identify which groups are most different and on which axes.

Use cases:
- Compare narratives across multiple departments
- Analyze differences between customer segments
- Study regional variations in organizational culture
- Research comparing multiple communities or demographics

Returns ANOVA results, pairwise comparisons, and interpretation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "cohorts": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "description": "Dictionary mapping cohort names to lists of texts (e.g., {'engineering': [...], 'sales': [...]})",
                    },
                },
                "required": ["cohorts"],
            },
        ),
        Tool(
            name="analyze_mode_flow",
            description="""Analyze mode transitions and detect patterns in a sequence of texts.

Maps how narratives flow through different modes over a sequence, computing
transition probabilities and identifying patterns. Reveals which modes are
"sticky" (stable) vs "volatile" (texts quickly leave).

Use cases:
- Understand typical narrative evolution patterns
- Identify intervention points where mode shifts occur
- Study how organizations transition between cultural states
- Detect recurring patterns in discourse evolution

Returns transition matrix, flow patterns, and stability analysis.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to analyze for mode flow patterns",
                    },
                },
                "required": ["texts"],
            },
        ),
        # --- Force Field Analysis Tools (v2 API) ---
        Tool(
            name="analyze_force_field",
            description="""Analyze the attractor/detractor force field of a narrative.

Returns the forces acting on the narrative - what it's being pulled TOWARD
(attractors) and pushed AWAY FROM (detractors). This adds two dimensions
to the cultural manifold:

- Attractor Strength (A+): Pull toward positive states (-2 to +2)
- Detractor Strength (D-): Push from negative states (-2 to +2)

Attractor Targets (what narratives are drawn toward):
- AUTONOMY: Self-determination, freedom, control
- COMMUNITY: Belonging, connection, togetherness
- JUSTICE: Fairness, equity, rightness
- MEANING: Purpose, transcendence, significance
- SECURITY: Stability, safety, predictability
- RECOGNITION: Visibility, validation, appreciation

Detractor Sources (what narratives flee from):
- OPPRESSION: Control, domination, powerlessness
- ISOLATION: Alienation, abandonment, loneliness
- INJUSTICE: Unfairness, corruption, betrayal
- MEANINGLESSNESS: Futility, nihilism, absurdity
- INSTABILITY: Chaos, threat, unpredictability
- INVISIBILITY: Being ignored, dismissed, devalued

Force Quadrants:
- ACTIVE_TRANSFORMATION: High attractor + high detractor (moving from bad toward good)
- PURE_ASPIRATION: High attractor + low detractor (drawn to positive without fleeing)
- PURE_ESCAPE: Low attractor + high detractor (fleeing without clear direction)
- STASIS: Low attractor + low detractor (no strong forces)

Use this to understand the motivational direction of narratives.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze for force field dynamics",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="analyze_trajectory_forces",
            description="""Analyze how force fields evolve across a sequence of texts.

Tracks attractor/detractor changes over time, identifying:
- Attractor shifts (changing goals - what the narrative is moving toward)
- Detractor emergence/resolution (new threats or resolved fears)
- Energy level changes (total force magnitude)
- Quadrant transitions (e.g., from PURE_ESCAPE to ACTIVE_TRANSFORMATION)

Returns trajectory data showing how narrative forces evolve, including:
- Per-point force field analysis
- Trend direction for attractors and detractors
- Identified shifts in primary attractor/detractor targets
- Interpretation of the overall trajectory dynamics

Use this to understand how narratives evolve their motivational structure
over time - are they gaining direction? Losing energy? Resolving tensions?""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sequence of texts in chronological order",
                    },
                },
                "required": ["texts"],
            },
        ),
        Tool(
            name="compare_force_fields",
            description="""Compare force fields between two groups of narratives.

Identifies differences in:
- Attractor/detractor strengths (which group has stronger pull/push)
- Primary targets/sources (what each group is drawn to / fleeing from)
- Energy levels (which group is more motivated/dynamic)
- Quadrant distributions (how motivational structure differs)

Use cases:
- Compare leadership messaging (aspiration-focused?) vs employee communications (escape-focused?)
- Analyze how different communities frame similar issues
- Detect misalignment between stated values (attractors) and experienced reality (detractors)
- Understand why two groups respond differently to the same situation

Returns group statistics, attractor/detractor distributions, and interpretation.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_a_name": {
                        "type": "string",
                        "description": "Label for the first group",
                    },
                    "group_a_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Texts from group A",
                    },
                    "group_b_name": {
                        "type": "string",
                        "description": "Label for the second group",
                    },
                    "group_b_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Texts from group B",
                    },
                },
                "required": ["group_a_name", "group_a_texts", "group_b_name", "group_b_texts"],
            },
        ),
        Tool(
            name="get_force_targets",
            description="""List all attractor targets and detractor sources.

Returns the complete taxonomy of:
- Attractor Targets: What narratives can be drawn toward
- Detractor Sources: What narratives can flee from
- Force Quadrants: The four quadrants of the force field space

Use this as a reference when interpreting force field analysis results
or when designing narrative analysis strategies.""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        # --- High-Level Narrative Analysis Tools ---
        Tool(
            name="fetch_narrative_source",
            description="""Fetch and segment content from a URL into analysis units.

Supports multiple source types:
- Websites (HTML pages, articles, about pages)
- RSS/Atom feeds (blogs, news)
- News sites (auto-detected)
- Blog platforms (Medium, Substack, WordPress)

The tool automatically:
1. Detects the source type from the URL
2. Extracts meaningful text content (removes navigation, ads, boilerplate)
3. Segments into analysis-ready content units
4. Preserves metadata (timestamps, authors when available)

Use this as the first step in narrative analysis to get content ready for
projection and profile building.

Returns a list of content units with text and metadata.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (website, RSS feed, etc.)",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["website", "rss", "twitter", "reddit", "blog", "news", "document"],
                        "description": "Optional type hint (auto-detected if not provided)",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum number of content units to return (default: 100)",
                        "default": 100,
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include source metadata (default: true)",
                        "default": True,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="build_narrative_profile",
            description="""Build a comprehensive narrative profile from content units.

Takes content units (from fetch_narrative_source or provided directly) and
synthesizes a complete narrative profile including:

MANIFOLD POSITION:
- Centroid coordinates (agency, perceived_justice, belonging)
- Spread/variance in each dimension
- Quadrant classification (e.g., "High Agency + Low Justice")

MODE ANALYSIS:
- Dominant narrative mode (e.g., CYNICAL_BURNOUT, GROWTH_MINDSET)
- Mode distribution across all 12 modes
- Mode signature (top 3 modes)
- Stability score (consistency of narrative)

FORCE FIELD:
- Attractors: what the narrative is drawn toward
- Detractors: what the narrative is fleeing from
- Internal tensions: contradictions within the narrative

COMPARATIVE:
- Distinctive features
- Sample quotes

Use this after fetching content to get a complete narrative characterization.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "content_units": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "timestamp": {"type": "string"},
                                "author": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                            "required": ["text"],
                        },
                        "description": "List of content units with text field (from fetch_narrative_source)",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source identifier (URL, handle, name)",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["website", "rss", "twitter", "reddit", "blog", "news", "document", "raw_text"],
                        "description": "Type of source (default: website)",
                        "default": "website",
                    },
                    "include_force_analysis": {
                        "type": "boolean",
                        "description": "Include attractor/detractor force field analysis (default: true)",
                        "default": True,
                    },
                },
                "required": ["content_units", "source"],
            },
        ),
        Tool(
            name="get_narrative_suggestions",
            description="""Generate actionable suggestions from a narrative profile.

Takes a narrative profile (from build_narrative_profile) and generates
suggestions based on your intent:

INTENTS:
- "understand": Research/analysis insights about the narrative
- "engage": How to interact, resonate, or communicate with this audience
- "counter": How to position against or respond to this narrative
- "bridge": How to find common ground and connect
- "strategy": Content and positioning opportunities

SUGGESTION TYPES:
1. UNDERSTANDING: Mode characterization, tensions, stability analysis
2. ENGAGEMENT: What resonates, what to avoid, framing recommendations
3. STRATEGY: Gap analysis, positioning opportunities, viral potential
4. WARNING: AI content signals, extreme positioning, low legibility

Each suggestion includes:
- Priority level (high/medium/low)
- Insight: the main finding
- Recommendation: what to do
- Evidence: supporting quotes or data
- Related coordinates (when relevant)

Use this to get actionable insights from a narrative profile.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "description": "The narrative profile (from build_narrative_profile)",
                    },
                    "intent": {
                        "type": "string",
                        "enum": ["understand", "engage", "counter", "bridge", "strategy"],
                        "description": "Primary goal for suggestions (default: understand)",
                        "default": "understand",
                    },
                },
                "required": ["profile"],
            },
        ),
    ]

    # --- Advanced Research Tools (v3 API) - conditionally added ---

    if _PARSED_EXTRACTION_AVAILABLE:
        tools.append(Tool(
            name="extract_parsed",
            description="""Extract coordinates using dependency parsing (requires spaCy).

Uses spaCy dependency parsing for linguistically accurate feature extraction.
Unlike regex-based extraction, this captures:
- Actual syntactic structure (who is doing what to whom)
- Semantic roles (agent, patient, experiencer)
- Negation scope and modal modification
- Embedded clauses and passive voice handling

Input: text (str)
Output: Detailed parsed features including:
- agency_events: List of agency events with agent, verb, type, negation, modal strength
- justice_events: List of justice events with type (procedural/distributive/interactional), polarity
- belonging_markers: List of markers with type (ingroup/outgroup/universal)
- metadata: sentence_count, passive_voice_ratio, negation_ratio, modal_usage_ratio

Requires: spaCy with en_core_web_sm model (python -m spacy download en_core_web_sm)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze with dependency parsing",
                    },
                },
                "required": ["text"],
            },
        ))

    if _SEMANTIC_EXTRACTION_AVAILABLE:
        tools.append(Tool(
            name="extract_semantic",
            description="""Extract coordinates using semantic similarity (requires sentence-transformers).

Uses sentence-transformers to compute semantic similarity between text and
carefully designed prototype sentences. This addresses word sense disambiguation:
- "The process failed" (system/technical) vs "due process" (justice)
- "We won" (ingroup success) vs "We must consider" (inclusive framing)

Input: text (str), show_prototypes (bool, optional)
Output: Coordinates with confidence scores and best-matching prototypes for each dimension.

The 18 dimensions measured:
- Core (9): self_agency, other_agency, system_agency, procedural_justice,
  distributive_justice, interactional_justice, ingroup, outgroup, universal
- Modifiers (9): certainty, evidentiality, commitment, temporal_focus,
  temporal_scope, power_differential, social_distance, arousal, valence

Requires: sentence-transformers library""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze with semantic similarity",
                    },
                    "show_prototypes": {
                        "type": "boolean",
                        "description": "Include best-matching prototype sentences for each dimension",
                        "default": False,
                    },
                },
                "required": ["text"],
            },
        ))

    if _EXTERNAL_VALIDATION_AVAILABLE:
        tools.append(Tool(
            name="validate_external",
            description="""Run external validation against psychological scales.

Validates extracted coordinates against established psychological scales:
- Sense of Agency Scale (Tapal et al., 2017) - validates Agency dimension
- Organizational Justice Scale (Colquitt, 2001) - validates Justice dimension
- Inclusion of Other in Self Scale (Aron et al., 1992) - validates Belonging
- Brief Sense of Community Scale (Peterson et al., 2008) - validates Belonging

Input: texts (list), scale_type ("agency", "justice", "belonging")
Output: Validation metrics including:
- Correlation coefficient (r) with 95% CI
- Convergent validity assessment (r > 0.50 expected)
- Per-subscale correlations
- Interpretation of validity evidence

Note: Uses synthetic validation corpus for demonstration. For real validation,
provide matched text-scale response pairs.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to validate",
                    },
                    "scale_type": {
                        "type": "string",
                        "enum": ["agency", "justice", "belonging"],
                        "description": "Which psychological scale to validate against",
                    },
                },
                "required": ["texts", "scale_type"],
            },
        ))

    if _SAFETY_METRICS_AVAILABLE:
        tools.append(Tool(
            name="get_safety_report",
            description="""Get comprehensive safety metrics for deployment assessment.

Evaluates detector performance on a labeled test corpus and provides:
- False Positive Rate (FPR): How often does the system cry wolf?
- False Negative Rate (FNR): How often does it miss real threats?
- ROC curves and calibration analysis
- Adversarial robustness testing results
- Deployment readiness assessment

Input: texts (list) - texts to include in evaluation
Output: Comprehensive safety report including:
- Regime classification metrics (accuracy, per-class precision/recall/F1)
- Ossification detection metrics (critical FNR/FPR)
- Adversarial evasion rates by technique
- Deployment readiness: research_only, human_in_loop, or automated

Thresholds for deployment stages:
- Research: accuracy > 50%, samples > 100
- Monitoring: accuracy > 70%, critical_FNR < 30%
- Automation: accuracy > 85%, critical_FNR < 10%, adversarial_robustness > 80%""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to include in safety evaluation",
                    },
                },
                "required": ["texts"],
            },
        ))

    if _PARSED_EXTRACTION_AVAILABLE:
        tools.append(Tool(
            name="compare_extraction_methods",
            description="""Compare regex vs parsed vs semantic extraction methods.

Runs the same text through all available extraction methods and provides
a side-by-side comparison showing:
- Feature values from each method
- Significant differences between methods
- Disambiguation analysis (how semantic/parsed handle ambiguous words)
- Improvement notes (where parsed/semantic outperform regex)

Input: text (str)
Output: Side-by-side comparison including:
- regex_based: Feature counts from pattern matching
- parsed_based: Feature counts from dependency parsing
- detailed_analysis: Full parsed features with events and markers
- differences: Features where methods disagree
- improvements: Specific cases where parsing helps (negation, modals, passive voice)

Use this to understand when advanced extraction methods add value.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to compare extraction methods on",
                    },
                },
                "required": ["text"],
            },
        ))

    return tools


# --- Tool Implementations ---

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""
    config = ObservatoryConfig(backend_url=BACKEND_URL, default_model=DEFAULT_MODEL)

    try:
        async with ObservatoryClient(config) as client:
            # Check backend health first
            try:
                await client.health_check()
            except Exception as e:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Observatory backend not reachable at {BACKEND_URL}. "
                            f"Ensure the backend is running.\n\nError: {e}",
                        )
                    ],
                    isError=True,
                )

            if name == "project_text":
                result = await client.project_text(
                    text=arguments["text"],
                    model_id=arguments.get("model_id"),
                    use_v2=True,  # Use v2 API with soft labels and stability
                )
                include_soft_labels = arguments.get("include_soft_labels", False)
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=format_projection_result(result.to_dict(), include_soft_labels),
                        )
                    ]
                )

            elif name == "analyze_corpus":
                result = await client.analyze_corpus(
                    texts=arguments["texts"],
                    detect_clusters=arguments.get("detect_clusters", True),
                )

                output = f"""
## Corpus Analysis Results

**Total texts analyzed**: {result.total_texts}

### Mode Distribution
"""
                for mode, count in sorted(result.mode_distribution.items(), key=lambda x: -x[1]):
                    pct = count / result.total_texts * 100
                    output += f"- {mode}: {count} ({pct:.1f}%)\n"

                if result.clusters:
                    output += f"\n### Detected Clusters ({len(result.clusters)})\n"
                    for cluster in result.clusters:
                        # Handle both "fairness" and "perceived_justice" in centroid
                        pj = cluster.centroid.get('perceived_justice', cluster.centroid.get('fairness', 0))
                        output += f"""
**Cluster {cluster.cluster_id}** ({cluster.mode})
- Size: {cluster.size} texts
- Stability: {cluster.stability:.2f}
- Centroid: A={cluster.centroid.get('agency', 0):.2f}, PJ={pj:.2f}, B={cluster.centroid.get('belonging', 0):.2f}
"""
                        if cluster.exemplar_texts:
                            output += f"- Exemplar: \"{cluster.exemplar_texts[0][:80]}...\"\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "compare_projections":
                result = await client.compare_projection_methods(
                    text=arguments["text"],
                    methods=arguments.get("methods"),
                )
                output = f"""
## Projection Method Comparison

**Text**: "{arguments['text'][:100]}..."

### Results by Method
"""
                for method, data in result.get("results", {}).items():
                    coords = data.get("coordinates", {})
                    # Handle both "fairness" and "perceived_justice"
                    pj = coords.get('perceived_justice', coords.get('fairness', 0))
                    output += f"""
**{method.upper()}**
- Agency: {coords.get('agency', 0):.3f}
- Perceived Justice: {pj:.3f}
- Belonging: {coords.get('belonging', 0):.3f}
- Mode: {data.get('mode', 'N/A')}
"""

                # Compute variance if multiple methods
                if len(result.get("results", {})) > 1:
                    agencies = [r["coordinates"]["agency"] for r in result["results"].values()]
                    output += f"\n### Stability Analysis\n"
                    output += f"Agency variance: {max(agencies) - min(agencies):.3f}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "generate_probe":
                # This would ideally call a dedicated endpoint
                # For now, we describe what the probe should be
                # Handle both "target_fairness" and "target_perceived_justice"
                target_pj = arguments.get("target_perceived_justice", arguments.get("target_fairness", 0))
                target = (
                    arguments["target_agency"],
                    target_pj,
                    arguments["target_belonging"],
                )
                domain = arguments.get("domain", "general")

                probe_description = generate_probe_description(target, domain)

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"""
## Probe Generation

**Target coordinates**: Agency={target[0]:.2f}, Perceived Justice={target[1]:.2f}, Belonging={target[2]:.2f}
**Domain**: {domain}

### Suggested Probe Characteristics
{probe_description}

### Recommended Approach
Generate a text that embodies these characteristics, then use `project_text`
to verify it lands near the target coordinates. Iterate if needed.
""",
                        )
                    ]
                )

            elif name == "get_manifold_state":
                status = await client.health_check()
                projection_status = await client.get_projection_status()

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"""
## Observatory State

**Backend status**: {status.get('status', 'unknown')}
**Loaded models**: {len(status.get('loaded_models', []))}
**Has projection**: {status.get('has_projection', False)}
**Training examples**: {status.get('training_examples', 0)}

### Projection Details
{json.dumps(projection_status, indent=2) if projection_status else 'No projection loaded'}
""",
                        )
                    ]
                )

            elif name == "add_training_example":
                # Handle both "perceived_justice" and "fairness" for backward compatibility
                pj_value = arguments.get("perceived_justice", arguments.get("fairness"))
                if pj_value is None:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: Either 'perceived_justice' or 'fairness' parameter is required.",
                            )
                        ],
                        isError=True,
                    )
                result = await client.add_training_example(
                    text=arguments["text"],
                    agency=arguments["agency"],
                    fairness=pj_value,  # Backend still uses 'fairness' parameter name
                    belonging=arguments["belonging"],
                    source="mcp_agent",
                )
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Training example added. Total examples: {result.get('total_examples', 'unknown')}",
                        )
                    ]
                )

            elif name == "explain_modes":
                mode = arguments.get("mode", "all")
                if mode == "all":
                    output = "## Narrative Mode System\n\n"
                    for m, info in NARRATIVE_MODES.items():
                        output += f"### {m.upper()}\n"
                        output += f"{info['description']}\n"
                        output += f"Examples: {', '.join(info['examples'])}\n\n"
                else:
                    info = NARRATIVE_MODES.get(mode, {})
                    output = f"## {mode.upper()} Mode\n\n"
                    output += f"{info.get('description', 'Unknown mode')}\n"
                    output += f"Examples: {', '.join(info.get('examples', []))}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "detect_hypocrisy":
                espoused = arguments["espoused_texts"]
                operational = arguments["operational_texts"]

                # Project both sets
                espoused_results = await client.project_batch(espoused)
                operational_results = await client.project_batch(operational)

                # Compute centroids (using perceived_justice via .fairness property for backward compat)
                def centroid(results):
                    n = len(results)
                    return {
                        "agency": sum(r.agency for r in results) / n,
                        "perceived_justice": sum(r.perceived_justice for r in results) / n,
                        "belonging": sum(r.belonging for r in results) / n,
                    }

                esp_centroid = centroid(espoused_results)
                ops_centroid = centroid(operational_results)

                # Compute deltas
                delta_agency = abs(esp_centroid["agency"] - ops_centroid["agency"])
                delta_pj = abs(esp_centroid["perceived_justice"] - ops_centroid["perceived_justice"])
                delta_belonging = abs(esp_centroid["belonging"] - ops_centroid["belonging"])
                total_delta = (delta_agency + delta_pj + delta_belonging) / 3

                interpretation = ""
                if total_delta < 0.3:
                    interpretation = "Low hypocrisy gap - values appear aligned"
                elif total_delta < 0.7:
                    interpretation = "Moderate hypocrisy gap - some misalignment detected"
                else:
                    interpretation = "High hypocrisy gap - significant values-action disconnect"

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"""
## Hypocrisy Analysis

### Espoused Values (from {len(espoused)} texts)
- Agency: {esp_centroid['agency']:.2f}
- Perceived Justice: {esp_centroid['perceived_justice']:.2f}
- Belonging: {esp_centroid['belonging']:.2f}

### Inferred Values (from {len(operational)} texts)
- Agency: {ops_centroid['agency']:.2f}
- Perceived Justice: {ops_centroid['perceived_justice']:.2f}
- Belonging: {ops_centroid['belonging']:.2f}

### Hypocrisy Gap (Δw)
- Agency delta: {delta_agency:.2f}
- Perceived Justice delta: {delta_pj:.2f}
- Belonging delta: {delta_belonging:.2f}
- **Overall Δw: {total_delta:.2f}**

### Interpretation
{interpretation}
""",
                        )
                    ]
                )

            elif name == "run_scenario":
                # Simplified scenario simulation
                # Full implementation would call backend scenario endpoints
                scenario_id = arguments["scenario_id"]

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"""
## Scenario Simulation

**Scenario**: {scenario_id}

Scenario simulation requires the full observatory UI for visualization.
Use the web interface at http://localhost:3000 to run interactive simulations.

For programmatic access, consider:
1. Defining era configurations with different reward weights
2. Using `analyze_corpus` to track how narratives would evolve
3. Computing survival metrics based on mode-reward alignment
""",
                        )
                    ]
                )

            # --- New Projection Mode Management Tool Implementations ---

            elif name == "list_projection_modes":
                result = await client.list_projection_modes()
                modes = result.get("modes", [])
                current = result.get("current_mode", "unknown")
                recommendations = result.get("recommendations", {})

                output = f"""
## Available Projection Modes

**Current mode**: {current}

### Modes
"""
                for mode in modes:
                    is_current = " (ACTIVE)" if mode.get("name") == current else ""
                    available = "Available" if mode.get("is_available", False) else "Not available"
                    output += f"""
**{mode.get('display_name', mode.get('name'))}**{is_current}
- Name: `{mode.get('name')}`
- Status: {available}
- Models: {', '.join(mode.get('models', []))}
- Description: {mode.get('description', 'N/A')}
"""
                    if mode.get('cv_score'):
                        output += f"- CV Score: {mode.get('cv_score'):.3f}\n"

                output += f"""
### Recommendations
- Best accuracy: `{recommendations.get('best_accuracy', 'mpnet_projection')}`
- Best robustness: `{recommendations.get('best_robustness', 'multi_model_ensemble')}`
- Fastest: `{recommendations.get('fastest', 'current_projection')}`
- Uncertainty quantification: `{recommendations.get('uncertainty_quantification', 'ensemble_projection')}`
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "select_projection_mode":
                mode = arguments["mode"]
                result = await client.select_projection_mode(mode)

                output = f"""
## Projection Mode Selected

**Mode**: {result.get('display_name', mode)}
**Status**: {'Success' if result.get('success') else 'Failed'}

{result.get('message', '')}

**Required models**: {', '.join(result.get('required_models', []))}

Subsequent analysis calls will now use this projection mode.
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "get_current_projection_mode":
                result = await client.get_current_projection_mode()
                current = result.get("current_mode")

                if current is None:
                    output = """
## Current Projection Mode

No projection mode is currently selected.

Use `select_projection_mode` to select one of:
- current_projection (default MiniLM)
- mpnet_projection (best accuracy)
- multi_model_ensemble (best robustness)
- ensemble_projection (uncertainty quantification)
"""
                else:
                    mode_info = result.get("mode_info", {})
                    output = f"""
## Current Projection Mode

**Mode**: {mode_info.get('display_name', current)}
**Name**: `{current}`
**Description**: {mode_info.get('description', 'N/A')}
**Required models**: {', '.join(result.get('required_models', []))}
"""
                    if mode_info.get('cv_score'):
                        output += f"**CV Score**: {mode_info.get('cv_score'):.3f}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "analyze_with_uncertainty":
                result = await client.analyze_with_uncertainty(
                    text=arguments["text"],
                    model_id=arguments.get("model_id"),
                )

                # Extract coordinates
                vector = result.get("vector", {})
                pj = vector.get("perceived_justice", vector.get("fairness", 0))

                # Extract mode info
                mode_info = result.get("mode", {})
                primary_mode = mode_info.get("primary_mode", "unknown") if isinstance(mode_info, dict) else mode_info

                # Extract uncertainty
                uncertainty = result.get("uncertainty", {})

                output = f"""
## Uncertainty-Aware Analysis

**Text**: "{arguments['text'][:100]}..."

### Coordinates (with 95% Confidence Intervals)
"""
                if "confidence_intervals" in uncertainty:
                    ci = uncertainty["confidence_intervals"]
                    std = uncertainty.get("std_per_axis", {})
                    output += f"""
- Agency: {vector.get('agency', 0):.3f} +/- {std.get('agency', 0):.3f}
  CI: [{ci.get('agency', [0, 0])[0]:.3f}, {ci.get('agency', [0, 0])[1]:.3f}]
- Perceived Justice: {pj:.3f} +/- {std.get('perceived_justice', std.get('fairness', 0)):.3f}
  CI: [{ci.get('perceived_justice', ci.get('fairness', [0, 0]))[0]:.3f}, {ci.get('perceived_justice', ci.get('fairness', [0, 0]))[1]:.3f}]
- Belonging: {vector.get('belonging', 0):.3f} +/- {std.get('belonging', 0):.3f}
  CI: [{ci.get('belonging', [0, 0])[0]:.3f}, {ci.get('belonging', [0, 0])[1]:.3f}]
"""
                else:
                    output += f"""
- Agency: {vector.get('agency', 0):.3f}
- Perceived Justice: {pj:.3f}
- Belonging: {vector.get('belonging', 0):.3f}

Note: {uncertainty.get('note', 'Full uncertainty quantification requires ensemble_projection mode.')}
"""

                output += f"""
### Classification
- Primary Mode: {primary_mode}
- Confidence: {mode_info.get('confidence', 0):.1%}
- Overall Uncertainty Confidence: {uncertainty.get('overall_confidence', 'N/A')}
- Method: {uncertainty.get('method', 'N/A')}
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "get_soft_labels":
                result = await client.get_soft_labels(
                    text=arguments["text"],
                    model_id=arguments.get("model_id"),
                )

                output = f"""
## Soft Labels (Probability Distribution)

**Text**: "{result.get('text', arguments['text'])[:100]}..."

### Primary Classification
- Primary Mode: {result.get('primary_mode', 'unknown')} ({result.get('primary_probability', 0):.1%})
- Secondary Mode: {result.get('secondary_mode', 'unknown')} ({result.get('secondary_probability', 0):.1%})
- Category: {result.get('category', 'unknown')}

### Stability
- Stability Score: {result.get('stability_score', 0):.2f}
- Boundary Case: {'Yes' if result.get('is_boundary_case') else 'No'}

### Full Probability Distribution
"""
                all_probs = result.get("all_probabilities", {})
                if all_probs:
                    # Sort by probability descending
                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    for mode_name, prob in sorted_probs:
                        bar_len = int(prob * 20)
                        bar = "=" * bar_len + "-" * (20 - bar_len)
                        output += f"- {mode_name}: {bar} {prob:.1%}\n"
                else:
                    output += "No probability distribution available.\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            # --- Comparative Analysis Tool Implementations (v2 API) ---

            elif name == "compare_narratives":
                result = await client.compare_narratives(
                    group_a_name=arguments["group_a_name"],
                    group_a_texts=arguments["group_a_texts"],
                    group_b_name=arguments["group_b_name"],
                    group_b_texts=arguments["group_b_texts"],
                )

                group_a = result.get("group_a", {})
                group_b = result.get("group_b", {})
                gap = result.get("gap_analysis", {})

                # Format centroids
                def format_centroid(c):
                    return f"A={c.get('agency', 0):.2f}, PJ={c.get('perceived_justice', c.get('fairness', 0)):.2f}, B={c.get('belonging', 0):.2f}"

                output = f"""
## Narrative Comparison

### {group_a.get('name', 'Group A')} ({group_a.get('count', 0)} texts)
- Centroid: {format_centroid(group_a.get('centroid', {}))}
- Mode distribution: {group_a.get('mode_distribution', {})}

### {group_b.get('name', 'Group B')} ({group_b.get('count', 0)} texts)
- Centroid: {format_centroid(group_b.get('centroid', {}))}
- Mode distribution: {group_b.get('mode_distribution', {})}

### Gap Analysis
- Agency delta: {gap.get('delta_agency', 0):.3f}
- Perceived Justice delta: {gap.get('delta_perceived_justice', gap.get('delta_fairness', 0)):.3f}
- Belonging delta: {gap.get('delta_belonging', 0):.3f}
- **Total gap: {gap.get('total_gap', 0):.3f}**

### Interpretation
{gap.get('interpretation', 'No interpretation available.')}
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "track_trajectory":
                result = await client.track_trajectory(
                    name=arguments["name"],
                    texts=arguments["texts"],
                    timestamps=arguments["timestamps"],
                )

                trend = result.get("trend", {})
                points = result.get("points", [])
                inflections = result.get("inflection_points", [])

                output = f"""
## Trajectory Analysis: {result.get('name', 'Unknown')}

### Trend Summary
- Direction: {trend.get('direction', 'N/A')}
- Velocity: {trend.get('velocity', 0):.3f} units/period
- Acceleration: {trend.get('acceleration', 0):.3f}

### Points ({len(points)} total)
"""
                # Show first and last few points
                for i, point in enumerate(points[:3]):
                    coords = point.get('coordinates', {})
                    output += f"- [{point.get('timestamp', 'N/A')}] Mode: {point.get('mode', 'N/A')}, "
                    output += f"A={coords.get('agency', 0):.2f}, PJ={coords.get('perceived_justice', 0):.2f}, B={coords.get('belonging', 0):.2f}\n"

                if len(points) > 6:
                    output += f"  ... ({len(points) - 6} more points) ...\n"

                for point in points[-3:] if len(points) > 3 else []:
                    coords = point.get('coordinates', {})
                    output += f"- [{point.get('timestamp', 'N/A')}] Mode: {point.get('mode', 'N/A')}, "
                    output += f"A={coords.get('agency', 0):.2f}, PJ={coords.get('perceived_justice', 0):.2f}, B={coords.get('belonging', 0):.2f}\n"

                if inflections:
                    output += f"\n### Inflection Points ({len(inflections)})\n"
                    for inf in inflections:
                        output += f"- {inf.get('timestamp', 'N/A')}: {inf.get('description', 'Shift detected')}\n"

                output += f"\n### Summary\n{result.get('summary', 'No summary available.')}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            # --- Alert System Tool Implementations (v2 API) ---

            elif name == "create_alert":
                result = await client.create_alert(
                    name=arguments["name"],
                    alert_type=arguments["type"],
                    config=arguments["config"],
                )

                output = f"""
## Alert Created

**ID**: {result.get('id', 'N/A')}
**Name**: {result.get('name', arguments['name'])}
**Type**: {result.get('type', arguments['type'])}
**Created**: {result.get('created_at', 'N/A')}

### Configuration
{json.dumps(result.get('config', arguments['config']), indent=2)}

The alert is now active and will be checked when you call `check_alerts`.
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "check_alerts":
                result = await client.check_alerts(
                    group_a_texts=arguments["group_a_texts"],
                    group_b_texts=arguments["group_b_texts"],
                )

                triggered = result.get("triggered", [])
                checked = result.get("checked", 0)

                output = f"""
## Alert Check Results

**Alerts checked**: {checked}
**Alerts triggered**: {len(triggered)}
**Timestamp**: {result.get('timestamp', 'N/A')}

"""
                if triggered:
                    output += "### Triggered Alerts\n"
                    for alert in triggered:
                        output += f"""
**{alert.get('name', 'Unknown')}** ({alert.get('type', 'N/A')})
- Trigger value: {alert.get('trigger_value', 'N/A')}
- Threshold: {alert.get('threshold', 'N/A')}
- Details: {alert.get('details', 'No details')}
"""
                else:
                    output += "No alerts were triggered.\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            # --- Advanced Analytics Tool Implementations (v2 API) ---

            elif name == "detect_outlier":
                result = await client.detect_outliers(
                    corpus=arguments["corpus"],
                    test_text=arguments["test_text"],
                )

                z_scores = result.get("z_scores", {})
                corpus_stats = result.get("corpus_stats", {})
                test_proj = result.get("test_projection", {})

                output = f"""
## Outlier Detection

**Test text**: "{arguments['test_text'][:80]}..."

### Result
- **Is outlier**: {'YES' if result.get('is_outlier') else 'NO'}
- **Outlier score**: {result.get('outlier_score', 0):.3f} (Mahalanobis distance)

### Z-Scores (standard deviations from corpus mean)
- Agency: {z_scores.get('agency', 0):.2f}
- Perceived Justice: {z_scores.get('perceived_justice', z_scores.get('fairness', 0)):.2f}
- Belonging: {z_scores.get('belonging', 0):.2f}

### Corpus Statistics (n={len(arguments['corpus'])})
- Centroid: A={corpus_stats.get('centroid', {}).get('agency', 0):.2f}, PJ={corpus_stats.get('centroid', {}).get('perceived_justice', 0):.2f}, B={corpus_stats.get('centroid', {}).get('belonging', 0):.2f}
- Std Dev: A={corpus_stats.get('std_dev', {}).get('agency', 0):.2f}, PJ={corpus_stats.get('std_dev', {}).get('perceived_justice', 0):.2f}, B={corpus_stats.get('std_dev', {}).get('belonging', 0):.2f}

### Test Text Projection
- Agency: {test_proj.get('agency', 0):.2f}
- Perceived Justice: {test_proj.get('perceived_justice', 0):.2f}
- Belonging: {test_proj.get('belonging', 0):.2f}

### Interpretation
{result.get('interpretation', 'No interpretation available.')}
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "analyze_cohorts":
                result = await client.analyze_cohorts(
                    cohorts=arguments["cohorts"],
                )

                cohorts_data = result.get("cohorts", [])
                anova = result.get("anova", {})
                pairwise = result.get("pairwise_comparisons", [])

                output = f"""
## Cohort Analysis

### Cohort Summary
"""
                for cohort in cohorts_data:
                    centroid = cohort.get('centroid', {})
                    output += f"""
**{cohort.get('name', 'Unknown')}** (n={cohort.get('count', 0)})
- Centroid: A={centroid.get('agency', 0):.2f}, PJ={centroid.get('perceived_justice', 0):.2f}, B={centroid.get('belonging', 0):.2f}
- Dominant mode: {max(cohort.get('mode_distribution', {'unknown': 1}).items(), key=lambda x: x[1])[0]}
"""

                output += f"""
### ANOVA Results
- **Significant**: {'YES' if anova.get('significant') else 'NO'}
- F-statistic: {anova.get('f_statistic', 0):.3f}
- p-value: {anova.get('p_value', 1):.4f}

#### Per-Axis ANOVA
"""
                per_axis = anova.get("per_axis", {})
                for axis, stats in per_axis.items():
                    sig = "***" if stats.get('p_value', 1) < 0.001 else "**" if stats.get('p_value', 1) < 0.01 else "*" if stats.get('p_value', 1) < 0.05 else ""
                    output += f"- {axis}: F={stats.get('f_statistic', 0):.2f}, p={stats.get('p_value', 1):.4f} {sig}\n"

                if pairwise:
                    output += "\n### Significant Pairwise Differences\n"
                    for comp in pairwise:
                        output += f"- {comp.get('group_a')} vs {comp.get('group_b')}: {comp.get('description', 'Significant difference')}\n"

                output += f"\n### Interpretation\n{result.get('interpretation', 'No interpretation available.')}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "analyze_mode_flow":
                result = await client.analyze_mode_flow(
                    texts=arguments["texts"],
                )

                mode_sequence = result.get("mode_sequence", [])
                transition_matrix = result.get("transition_matrix", {})
                patterns = result.get("flow_patterns", [])
                stable = result.get("stable_modes", [])
                volatile = result.get("volatile_modes", [])

                output = f"""
## Mode Flow Analysis

### Mode Sequence ({len(mode_sequence)} texts)
{' -> '.join(mode_sequence[:10])}{'...' if len(mode_sequence) > 10 else ''}

### Stability Analysis
- **Stable modes** (texts tend to stay): {', '.join(stable) if stable else 'None detected'}
- **Volatile modes** (texts tend to leave): {', '.join(volatile) if volatile else 'None detected'}

### Transition Matrix
"""
                if transition_matrix:
                    # Get all unique modes
                    modes = sorted(set(transition_matrix.keys()))
                    # Build a simple text representation
                    output += "From \\ To | " + " | ".join(f"{m[:8]:>8}" for m in modes) + "\n"
                    output += "-" * (12 + 11 * len(modes)) + "\n"
                    for from_mode in modes:
                        row = transition_matrix.get(from_mode, {})
                        output += f"{from_mode[:10]:<10} | "
                        output += " | ".join(f"{row.get(to_mode, 0):8.2f}" for to_mode in modes) + "\n"
                else:
                    output += "No transition data available.\n"

                if patterns:
                    output += "\n### Detected Patterns\n"
                    for pattern in patterns[:5]:
                        output += f"- {pattern.get('description', pattern)}\n"

                output += f"\n### Interpretation\n{result.get('interpretation', 'No interpretation available.')}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            # --- Force Field Analysis Tool Implementations (v2 API) ---

            elif name == "analyze_force_field":
                result = await client.analyze_force_field(
                    text=arguments["text"],
                )

                output = f"""
## Force Field Analysis

**Text**: "{arguments['text'][:100]}..."

### Force Strengths
- **Attractor Strength**: {result.get('attractor_strength', 0):.3f} (pull toward positive)
- **Detractor Strength**: {result.get('detractor_strength', 0):.3f} (push from negative)
- **Net Force**: {result.get('net_force', 0):.3f}
- **Force Direction**: {result.get('force_direction', 'BALANCED')}

### Force Quadrant
- **Quadrant**: {result.get('force_quadrant', 'STASIS')}
- **Description**: {result.get('quadrant_description', '')}
- **Energy Level**: {result.get('energy_level', 'unknown')}

### Primary Forces
- **Primary Attractor**: {result.get('primary_attractor', 'None')} (drawn toward)
- **Secondary Attractor**: {result.get('secondary_attractor', 'None')}
- **Primary Detractor**: {result.get('primary_detractor', 'None')} (fleeing from)
- **Secondary Detractor**: {result.get('secondary_detractor', 'None')}

### Attractor Scores
"""
                attractor_scores = result.get('attractor_scores', {})
                if attractor_scores:
                    for target, score in sorted(attractor_scores.items(), key=lambda x: -x[1]):
                        bar_len = int(score * 20)
                        bar = "=" * bar_len + "-" * (20 - bar_len)
                        output += f"- {target}: {bar} {score:.3f}\n"
                else:
                    output += "- No significant attractor targets detected\n"

                output += "\n### Detractor Scores\n"
                detractor_scores = result.get('detractor_scores', {})
                if detractor_scores:
                    for source, score in sorted(detractor_scores.items(), key=lambda x: -x[1]):
                        bar_len = int(score * 20)
                        bar = "=" * bar_len + "-" * (20 - bar_len)
                        output += f"- {source}: {bar} {score:.3f}\n"
                else:
                    output += "- No significant detractor sources detected\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "analyze_trajectory_forces":
                if len(arguments["texts"]) < 2:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: Need at least 2 texts for trajectory analysis.",
                            )
                        ],
                        isError=True,
                    )

                result = await client.analyze_trajectory_forces(
                    texts=arguments["texts"],
                )

                attractor_traj = result.get("attractor_trajectory", {})
                detractor_traj = result.get("detractor_trajectory", {})
                energy_traj = result.get("energy_trajectory", {})
                attractor_shifts = result.get("attractor_shifts", [])
                detractor_shifts = result.get("detractor_shifts", [])

                output = f"""
## Force Trajectory Analysis

**Points analyzed**: {result.get('n_points', len(arguments['texts']))}

### Attractor Trajectory
- **Trend**: {attractor_traj.get('trend', 'stable')}
- **Start**: {attractor_traj.get('start', 0):.3f} -> **End**: {attractor_traj.get('end', 0):.3f}
- **Change**: {attractor_traj.get('change', 0):+.3f}

### Detractor Trajectory
- **Trend**: {detractor_traj.get('trend', 'stable')}
- **Start**: {detractor_traj.get('start', 0):.3f} -> **End**: {detractor_traj.get('end', 0):.3f}
- **Change**: {detractor_traj.get('change', 0):+.3f}

### Energy Trajectory
- **Trend**: {energy_traj.get('trend', 'stable')}
- **Start**: {energy_traj.get('start', 0):.3f} -> **End**: {energy_traj.get('end', 0):.3f}
"""

                if attractor_shifts:
                    output += "\n### Attractor Shifts\n"
                    for shift in attractor_shifts:
                        output += f"- At point {shift.get('index', '?')}: {shift.get('from', 'None')} -> {shift.get('to', 'None')}\n"

                if detractor_shifts:
                    output += "\n### Detractor Shifts\n"
                    for shift in detractor_shifts:
                        output += f"- At point {shift.get('index', '?')}: {shift.get('from', 'None')} -> {shift.get('to', 'None')}\n"

                # Show first and last few points
                points = result.get("points", [])
                if points:
                    output += "\n### Force Points (sample)\n"
                    for i, point in enumerate(points[:3]):
                        output += f"- [{i}] A+={point.get('attractor_strength', 0):.2f}, D-={point.get('detractor_strength', 0):.2f}, Quadrant={point.get('force_quadrant', 'STASIS')}\n"
                    if len(points) > 6:
                        output += f"  ... ({len(points) - 6} more points) ...\n"
                    for i, point in enumerate(points[-3:], len(points) - 3) if len(points) > 3 else []:
                        output += f"- [{i}] A+={point.get('attractor_strength', 0):.2f}, D-={point.get('detractor_strength', 0):.2f}, Quadrant={point.get('force_quadrant', 'STASIS')}\n"

                output += f"\n### Interpretation\n{result.get('interpretation', 'No interpretation available.')}\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "compare_force_fields":
                result = await client.compare_force_fields(
                    group_a_name=arguments["group_a_name"],
                    group_a_texts=arguments["group_a_texts"],
                    group_b_name=arguments["group_b_name"],
                    group_b_texts=arguments["group_b_texts"],
                )

                group_a = result.get("group_a", {})
                group_b = result.get("group_b", {})
                comparison = result.get("comparison", {})

                output = f"""
## Force Field Comparison

### {group_a.get('name', 'Group A')} (n={group_a.get('stats', {}).get('n', 0)})
- Mean Attractor: {group_a.get('stats', {}).get('mean_attractor', 0):.3f}
- Mean Detractor: {group_a.get('stats', {}).get('mean_detractor', 0):.3f}
- Dominant Quadrant: {group_a.get('stats', {}).get('dominant_quadrant', 'N/A')}
- Dominant Attractor: {group_a.get('stats', {}).get('dominant_attractor', 'None')}
- Dominant Detractor: {group_a.get('stats', {}).get('dominant_detractor', 'None')}

### {group_b.get('name', 'Group B')} (n={group_b.get('stats', {}).get('n', 0)})
- Mean Attractor: {group_b.get('stats', {}).get('mean_attractor', 0):.3f}
- Mean Detractor: {group_b.get('stats', {}).get('mean_detractor', 0):.3f}
- Dominant Quadrant: {group_b.get('stats', {}).get('dominant_quadrant', 'N/A')}
- Dominant Attractor: {group_b.get('stats', {}).get('dominant_attractor', 'None')}
- Dominant Detractor: {group_b.get('stats', {}).get('dominant_detractor', 'None')}

### Comparison
- Attractor Gap: {comparison.get('attractor_gap', 0):+.4f} (higher: {comparison.get('attractor_higher', 'N/A')})
- Detractor Gap: {comparison.get('detractor_gap', 0):+.4f} (higher: {comparison.get('detractor_higher', 'N/A')})
- Same Dominant Attractor: {'Yes' if comparison.get('same_dominant_attractor') else 'No'}
- Same Dominant Detractor: {'Yes' if comparison.get('same_dominant_detractor') else 'No'}

### Interpretation
{result.get('interpretation', 'Groups show similar force fields')}
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            elif name == "get_force_targets":
                result = await client.get_force_field_targets()

                output = """
## Force Field Taxonomy

### Attractor Targets (what narratives are drawn TOWARD)
"""
                for target, desc in result.get("attractor_targets", {}).items():
                    output += f"- **{target}**: {desc}\n"

                output += "\n### Detractor Sources (what narratives flee FROM)\n"
                for source, desc in result.get("detractor_sources", {}).items():
                    output += f"- **{source}**: {desc}\n"

                output += "\n### Force Quadrants\n"
                quadrants = result.get("force_quadrants", {})
                for name_q, info in quadrants.items():
                    output += f"- **{name_q}**: {info.get('description', '')} (Energy: {info.get('energy', 'unknown')})\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )

            # --- Advanced Research Tool Implementations (v3 API) ---

            elif name == "extract_parsed":
                if not _PARSED_EXTRACTION_AVAILABLE:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: Parsed extraction not available. Please install spaCy and download en_core_web_sm:\n\npip install spacy\npython -m spacy download en_core_web_sm",
                            )
                        ],
                        isError=True,
                    )

                try:
                    result = extract_features_detailed(arguments["text"])

                    agency = result.get("agency", {})
                    justice = result.get("justice", {})
                    belonging = result.get("belonging", {})
                    metadata = result.get("metadata", {})

                    output = f"""
## Parsed Feature Extraction

**Text**: "{arguments['text'][:100]}..."

### Agency Analysis
- Self Agency: {agency.get('self', 0):.3f}
- Other Agency: {agency.get('other', 0):.3f}
- System Agency: {agency.get('system', 0):.3f}

**Agency Events** ({len(agency.get('events', []))} detected):
"""
                    for event in agency.get('events', [])[:5]:
                        neg_str = " [NEGATED]" if event.get('negated') else ""
                        pass_str = " [PASSIVE]" if event.get('passive') else ""
                        modal_str = f" [modal={event.get('modal_strength', 1.0):.2f}]" if event.get('modal_strength', 1.0) < 1.0 else ""
                        output += f"- '{event.get('agent', '?')} {event.get('verb', '?')}' -> {event.get('type', '?')}{neg_str}{pass_str}{modal_str}\n"

                    if len(agency.get('events', [])) > 5:
                        output += f"  ... and {len(agency.get('events', [])) - 5} more events\n"

                    output += f"""
### Justice Analysis
- Procedural: {justice.get('procedural', 0):.3f}
- Distributive: {justice.get('distributive', 0):.3f}
- Interactional: {justice.get('interactional', 0):.3f}

**Justice Events** ({len(justice.get('events', []))} detected):
"""
                    for event in justice.get('events', [])[:5]:
                        output += f"- [{event.get('type', '?')}] '{event.get('trigger', '?')}' polarity={event.get('polarity', 0):.2f}, conf={event.get('confidence', 0):.2f}\n"

                    output += f"""
### Belonging Analysis
- Ingroup: {belonging.get('ingroup', 0):.3f}
- Outgroup: {belonging.get('outgroup', 0):.3f}
- Universal: {belonging.get('universal', 0):.3f}

**Belonging Markers** ({len(belonging.get('markers', []))} detected):
"""
                    for marker in belonging.get('markers', [])[:5]:
                        subj_str = " [SUBJECT]" if marker.get('is_subject') else ""
                        output += f"- [{marker.get('type', '?')}] '{marker.get('text', '?')}'{subj_str}\n"

                    output += f"""
### Metadata
- Sentence count: {metadata.get('sentence_count', 0)}
- Passive voice ratio: {metadata.get('passive_voice_ratio', 0):.2%}
- Negation ratio: {metadata.get('negation_ratio', 0):.2%}
- Modal usage ratio: {metadata.get('modal_usage_ratio', 0):.2%}
"""

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error in parsed extraction: {str(e)}\n\nMake sure spaCy model is installed: python -m spacy download en_core_web_sm",
                            )
                        ],
                        isError=True,
                    )

            elif name == "extract_semantic":
                if not _SEMANTIC_EXTRACTION_AVAILABLE:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: Semantic extraction not available. Please install sentence-transformers:\n\npip install sentence-transformers",
                            )
                        ],
                        isError=True,
                    )

                try:
                    extractor = SemanticFeatureExtractor()
                    result = extractor.extract_with_details(arguments["text"])
                    show_prototypes = arguments.get("show_prototypes", False)

                    coord = result.hierarchical_coordinate
                    scores = result.construct_scores

                    output = f"""
## Semantic Feature Extraction

**Text**: "{arguments['text'][:100]}..."

### Core Dimensions (Agency)
- Self Agency: {scores.get('self_agency', {}).score if hasattr(scores.get('self_agency', {}), 'score') else scores.get('self_agency', 0):.3f}
- Other Agency: {scores.get('other_agency', {}).score if hasattr(scores.get('other_agency', {}), 'score') else scores.get('other_agency', 0):.3f}
- System Agency: {scores.get('system_agency', {}).score if hasattr(scores.get('system_agency', {}), 'score') else scores.get('system_agency', 0):.3f}

### Core Dimensions (Justice)
- Procedural: {scores.get('procedural_justice', {}).score if hasattr(scores.get('procedural_justice', {}), 'score') else scores.get('procedural_justice', 0):.3f}
- Distributive: {scores.get('distributive_justice', {}).score if hasattr(scores.get('distributive_justice', {}), 'score') else scores.get('distributive_justice', 0):.3f}
- Interactional: {scores.get('interactional_justice', {}).score if hasattr(scores.get('interactional_justice', {}), 'score') else scores.get('interactional_justice', 0):.3f}

### Core Dimensions (Belonging)
- Ingroup: {scores.get('ingroup', {}).score if hasattr(scores.get('ingroup', {}), 'score') else scores.get('ingroup', 0):.3f}
- Outgroup: {scores.get('outgroup', {}).score if hasattr(scores.get('outgroup', {}), 'score') else scores.get('outgroup', 0):.3f}
- Universal: {scores.get('universal', {}).score if hasattr(scores.get('universal', {}), 'score') else scores.get('universal', 0):.3f}

### Modifier Dimensions
- Certainty: {scores.get('certainty', {}).score if hasattr(scores.get('certainty', {}), 'score') else scores.get('certainty', 0):.3f}
- Evidentiality: {scores.get('evidentiality', {}).score if hasattr(scores.get('evidentiality', {}), 'score') else scores.get('evidentiality', 0):.3f}
- Power Differential: {scores.get('power_differential', {}).score if hasattr(scores.get('power_differential', {}), 'score') else scores.get('power_differential', 0):.3f}
- Arousal: {scores.get('arousal', {}).score if hasattr(scores.get('arousal', {}), 'score') else scores.get('arousal', 0):.3f}
- Valence: {scores.get('valence', {}).score if hasattr(scores.get('valence', {}), 'score') else scores.get('valence', 0):.3f}
"""

                    if show_prototypes:
                        output += "\n### Best-Matching Prototypes\n"
                        for dim_name, score_obj in scores.items():
                            if hasattr(score_obj, 'best_prototype') and hasattr(score_obj, 'confidence'):
                                output += f"- **{dim_name}**: \"{score_obj.best_prototype[:60]}...\" (conf={score_obj.confidence:.2f})\n"

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error in semantic extraction: {str(e)}",
                            )
                        ],
                        isError=True,
                    )

            elif name == "validate_external":
                if not _EXTERNAL_VALIDATION_AVAILABLE:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: External validation not available. Required dependencies may be missing.",
                            )
                        ],
                        isError=True,
                    )

                try:
                    validator = ExternalValidator()
                    texts = arguments["texts"]
                    scale_type = arguments["scale_type"]

                    # Generate synthetic scale responses for demonstration
                    _, scale_responses = generate_validation_corpus(n_samples=len(texts), seed=42)

                    # Map scale_type to scale_name
                    scale_name_map = {
                        "agency": "agency",
                        "justice": "justice",
                        "belonging": "belonging_ios",
                    }
                    scale_name = scale_name_map.get(scale_type, scale_type)

                    # Extract the relevant scale responses
                    scale_data = [r[scale_name] for r in scale_responses[:len(texts)]]

                    result = validator.correlate_with_scale(texts, scale_data, scale_name)

                    scale_info = VALIDATION_SCALES.get(scale_name, {})
                    scale_display_name = getattr(scale_info, 'name', scale_name) if scale_info else scale_name

                    output = f"""
## External Validation Report

**Scale**: {scale_display_name}
**Construct**: {scale_type}
**Samples**: {result.n_samples}

### Convergent Validity
- **Correlation (r)**: {result.correlation.r:.3f}
- **95% CI**: [{result.correlation.ci_lower:.3f}, {result.correlation.ci_upper:.3f}]
- **p-value**: {result.correlation.p_value:.4f}
- **Effect size**: {result.correlation.effect_size}
- **Convergent validity met (r > 0.50)**: {'YES' if result.convergent_validity_met else 'NO'}

### Subscale Correlations
"""
                    for sub_name, sub_corr in result.subscale_correlations.items():
                        output += f"- {sub_name}: r={sub_corr.r:.3f} (p={sub_corr.p_value:.4f})\n"

                    output += f"""
### Interpretation
The correlation of r={result.correlation.r:.3f} indicates a {result.correlation.effect_size}
relationship between the extracted {scale_type} coordinates and self-reported
{scale_display_name} scores.

{'This provides evidence of convergent validity - the two methods of measuring ' + scale_type + ' correlate as expected.' if result.convergent_validity_met else 'The correlation is below the 0.50 threshold typically expected for convergent validity. Further validation may be needed.'}

Note: This analysis uses synthetic scale responses for demonstration.
For real validation, provide matched text-scale response pairs.
"""

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error in external validation: {str(e)}",
                            )
                        ],
                        isError=True,
                    )

            elif name == "get_safety_report":
                if not _SAFETY_METRICS_AVAILABLE:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: Safety metrics not available. Required dependencies may be missing.",
                            )
                        ],
                        isError=True,
                    )

                try:
                    # Generate test corpus including user texts
                    corpus = generate_labeled_test_corpus(n_samples=max(100, len(arguments["texts"])), seed=42)

                    # Evaluate
                    evaluator = SafetyMetricsEvaluator()
                    regime_metrics = evaluator.evaluate_regime_classification(corpus)
                    detection_metrics = evaluator.evaluate_ossification_detection(corpus)

                    # Run adversarial testing
                    from safety_metrics import AdversarialTester
                    from cbr_thermometer import CBRThermometer
                    adversarial_tester = AdversarialTester(seed=42)
                    thermometer = CBRThermometer()
                    adversarial_report = adversarial_tester.test_evasion(thermometer, n_attempts=50)

                    # Assess deployment readiness
                    deployment_report = assess_deployment_readiness(evaluator, adversarial_report)

                    output = f"""
## Comprehensive Safety Report

### Deployment Readiness
- **Recommended Stage**: {deployment_report.recommended_stage.value.upper()}
- Ready for Research: {'YES' if deployment_report.ready_for_research else 'NO'}
- Ready for Monitoring (Human-in-Loop): {'YES' if deployment_report.ready_for_monitoring else 'NO'}
- Ready for Automation: {'YES' if deployment_report.ready_for_automation else 'NO'}

### Regime Classification Metrics
- **Accuracy**: {regime_metrics.accuracy:.2%}
- **Macro F1**: {regime_metrics.macro_f1:.2%}
- **OPAQUE FPR** (false alarms): {regime_metrics.opaque_fpr:.2%}
- **OPAQUE FNR** (missed threats): {regime_metrics.opaque_fnr:.2%}

### Ossification Detection Metrics
- **Exact Accuracy**: {detection_metrics.exact_accuracy:.2%}
- **Within-One Accuracy**: {detection_metrics.within_one_accuracy:.2%}
- **Critical FPR**: {detection_metrics.critical_fpr:.2%}
- **Critical FNR**: {detection_metrics.critical_fnr:.2%}

### Adversarial Robustness
- **Overall Evasion Rate**: {adversarial_report.evasion_rate:.2%}
- **Robustness Score**: {(1 - adversarial_report.evasion_rate):.2%}

**By Technique**:
"""
                    for technique, stats in adversarial_report.by_technique.items():
                        output += f"- {technique}: {stats['evasion_rate']:.2%} evasion rate\n"

                    output += "\n### ROC Analysis\n"
                    if detection_metrics.roc_data:
                        output += f"- **AUC**: {detection_metrics.roc_data.get('auc', 0):.3f}\n"

                    output += "\n### Calibration\n"
                    if detection_metrics.calibration_data:
                        output += f"- **Expected Calibration Error**: {detection_metrics.calibration_data.get('expected_calibration_error', 0):.4f}\n"
                        output += f"- **Perfectly Calibrated**: {'YES' if detection_metrics.calibration_data.get('perfectly_calibrated') else 'NO'}\n"

                    if deployment_report.blocking_issues:
                        output += "\n### Blocking Issues\n"
                        for issue in deployment_report.blocking_issues:
                            output += f"- [X] {issue}\n"

                    if deployment_report.warnings:
                        output += "\n### Warnings\n"
                        for warning in deployment_report.warnings:
                            output += f"- [!] {warning}\n"

                    if deployment_report.recommendations:
                        output += "\n### Recommendations\n"
                        for rec in deployment_report.recommendations:
                            output += f"- {rec}\n"

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error generating safety report: {str(e)}",
                            )
                        ],
                        isError=True,
                    )

            elif name == "compare_extraction_methods":
                if not _PARSED_EXTRACTION_AVAILABLE:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="Error: Extraction comparison not available. Please install spaCy:\n\npip install spacy\npython -m spacy download en_core_web_sm",
                            )
                        ],
                        isError=True,
                    )

                try:
                    result = compare_extraction_methods(arguments["text"])

                    output = f"""
## Extraction Method Comparison

**Text**: "{arguments['text'][:100]}..."

### Regex-Based Features
"""
                    regex_features = result.get("regex_based", {})
                    for feature, count in sorted(regex_features.items())[:10]:
                        if count > 0:
                            output += f"- {feature}: {count}\n"

                    output += "\n### Parsed-Based Features\n"
                    parsed_features = result.get("parsed_based", {})
                    for feature, count in sorted(parsed_features.items())[:10]:
                        if count > 0:
                            output += f"- {feature}: {count}\n"

                    output += "\n### Differences (Regex vs Parsed)\n"
                    differences = result.get("differences", {})
                    if differences:
                        for feature, diff in differences.items():
                            output += f"- **{feature}**: regex={diff.get('regex', 0)}, parsed={diff.get('parsed', 0)} (delta={diff.get('delta', 0):+d})\n"
                    else:
                        output += "No significant differences detected.\n"

                    output += "\n### Improvements from Parsing\n"
                    improvements = result.get("improvements", [])
                    if improvements:
                        for imp in improvements:
                            output += f"\n**{imp.get('type', 'Unknown')}**: {imp.get('description', '')}\n"
                            for ex in imp.get('examples', [])[:3]:
                                output += f"  - {ex}\n"
                    else:
                        output += "No specific improvements detected for this text.\n"

                    output += "\n### Detailed Agency Analysis\n"
                    detailed = result.get("detailed_analysis", {})
                    for event in detailed.get("agency", {}).get("events", [])[:5]:
                        neg_str = " [NEGATED]" if event.get("negated") else ""
                        pass_str = " [PASSIVE]" if event.get("passive") else ""
                        modal_str = f" [modal={event.get('modal_strength', 1.0):.2f}]" if event.get("modal_strength", 1.0) < 1.0 else ""
                        output += f"- '{event.get('agent', '?')} {event.get('verb', '?')}' -> {event.get('type', '?')}{neg_str}{pass_str}{modal_str}\n"

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error comparing extraction methods: {str(e)}",
                            )
                        ],
                        isError=True,
                    )

            # --- High-Level Narrative Analysis Tools ---

            elif name == "fetch_narrative_source":
                try:
                    result = await fetch_narrative_source(
                        url=arguments["url"],
                        source_type=arguments.get("source_type"),
                        max_items=arguments.get("max_items", 100),
                        include_metadata=arguments.get("include_metadata", True),
                    )

                    output = f"""
## Content Fetched

**Source**: {result.get('source', 'Unknown')}
**Type**: {result.get('source_type', 'Unknown')}
**Content Units**: {result.get('count', 0)}

### Sample Content Units
"""
                    for i, unit in enumerate(result.get('content_units', [])[:5], 1):
                        text_preview = unit.get('text', '')[:150]
                        timestamp = unit.get('timestamp', '')
                        output += f"\n**{i}.** {text_preview}..."
                        if timestamp:
                            output += f"\n   _Timestamp: {timestamp}_"
                        output += "\n"

                    if result.get('count', 0) > 5:
                        output += f"\n_...and {result.get('count', 0) - 5} more content units_\n"

                    output += """
### Next Steps
Use `build_narrative_profile` with these content units to generate a comprehensive profile.
"""

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error fetching source: {str(e)}\n\nMake sure the URL is accessible.",
                            )
                        ],
                        isError=True,
                    )

            elif name == "build_narrative_profile":
                try:
                    # Build the profile
                    profile = await build_narrative_profile(
                        content_units=arguments["content_units"],
                        source=arguments["source"],
                        source_type=arguments.get("source_type", "website"),
                        observatory_client=client,
                        include_force_analysis=arguments.get("include_force_analysis", True),
                    )

                    # Format the output
                    output = format_profile_result(profile)

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    import traceback
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error building profile: {str(e)}\n\n{traceback.format_exc()}",
                            )
                        ],
                        isError=True,
                    )

            elif name == "get_narrative_suggestions":
                try:
                    # Generate suggestions
                    suggestions = get_narrative_suggestions(
                        profile_dict=arguments["profile"],
                        intent=arguments.get("intent", "understand"),
                    )

                    # Format the output
                    output = format_suggestions_result(suggestions)

                    return CallToolResult(
                        content=[TextContent(type="text", text=output)]
                    )
                except Exception as e:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Error generating suggestions: {str(e)}",
                            )
                        ],
                        isError=True,
                    )

            else:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Unknown tool: {name}")
                    ],
                    isError=True,
                )

    except Exception as e:
        import traceback
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {e}\n\n{traceback.format_exc()}")],
            isError=True,
        )


def generate_probe_description(target: tuple[float, float, float], domain: str) -> str:
    """Generate description of what a probe text at target coords should contain.

    Args:
        target: (agency, perceived_justice, belonging) coordinates
        domain: Context domain for probe generation
    """
    agency, perceived_justice, belonging = target

    parts = []

    # Agency guidance
    if agency > 1.0:
        parts.append("Express strong self-determination, empowerment, taking control")
    elif agency > 0:
        parts.append("Show moderate autonomy and capability")
    elif agency < -1.0:
        parts.append("Convey helplessness, fatalism, being at mercy of forces")
    else:
        parts.append("Express limited agency or acceptance of constraints")

    # Perceived Justice guidance (formerly "fairness")
    if perceived_justice > 1.0:
        parts.append("Assert belief in fair treatment, system legitimacy")
    elif perceived_justice > 0:
        parts.append("Suggest general trust in fair treatment with some caveats")
    elif perceived_justice < -1.0:
        parts.append("Express perception of systemic injustice, corruption")
    else:
        parts.append("Express skepticism about fair treatment/system legitimacy")

    # Belonging guidance
    if belonging > 1.0:
        parts.append("Show deep connection to community, we-language")
    elif belonging > 0:
        parts.append("Indicate group membership, alignment with others")
    elif belonging < -1.0:
        parts.append("Express alienation, outsider status, disconnection")
    else:
        parts.append("Show some distance from group identity")

    # Domain-specific flavor
    domain_hints = {
        "corporate": "Frame in organizational/business context",
        "government": "Frame in civic/political context",
        "religion": "Frame in spiritual/community context",
        "general": "Use neutral framing",
    }
    parts.append(domain_hints.get(domain, ""))

    return "\n".join(f"- {p}" for p in parts if p)


# --- Server Entry Point ---

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
