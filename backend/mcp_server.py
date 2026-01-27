"""
AI Behavior Lab - MCP Server

A practical toolkit for analyzing AI behavior, detecting anomalies,
and monitoring safety in real-time.

Tools for:
- AI behavior mode detection (confident, uncertain, evasive, helpful, opaque)
- Safety monitoring with real-time alerts
- Adversarial content detection (gaming, opacity, covert channels)
- AI vs Human pattern comparison

Run locally:
    python mcp_server.py

Connects to the existing FastAPI backend on port 8000.
"""

import asyncio
import json
import os
from typing import Any, Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

# Configuration
BACKEND_URL = os.getenv("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.getenv("OBSERVATORY_DEFAULT_MODEL", "all-MiniLM-L6-v2")

# Create server instance
server = Server("ai-behavior-lab")


# --- HTTP Client ---

async def call_backend(method: str, endpoint: str, data: Optional[dict] = None) -> dict:
    """Call the FastAPI backend."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        url = f"{BACKEND_URL}{endpoint}"
        if method == "GET":
            response = await client.get(url, params=data)
        elif method == "POST":
            response = await client.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()


# --- Tool Definitions ---

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available observatory tools."""
    return [
        Tool(
            name="observatory_status",
            description="Check if the observatory backend is running and get current status (loaded models, projection state, training examples count).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="project_text",
            description="""Project a text onto the 3D cultural manifold.

Returns coordinates on three axes:
- Agency (-2 to +2): Self-determination vs fatalism
- Fairness (-2 to +2): Just/meritocratic vs corrupt/rigged
- Belonging (-2 to +2): Connected/embedded vs alienated

Also returns the detected narrative mode and confidence score.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze and project onto the manifold"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Embedding model to use (default: all-MiniLM-L6-v2)",
                        "default": DEFAULT_MODEL
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="project_batch",
            description="Project multiple texts onto the manifold in a single call. More efficient than calling project_text repeatedly.",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to analyze"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Embedding model to use",
                        "default": DEFAULT_MODEL
                    },
                    "detect_clusters": {
                        "type": "boolean",
                        "description": "Whether to detect narrative clusters in the corpus",
                        "default": True
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="compare_projections",
            description="""Compare how a text projects using different methods (ridge, gaussian process, neural, CAV).

Useful for understanding projection uncertainty and method disagreement.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to compare projections for"
                    },
                    "methods": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["ridge", "gp", "neural", "cav"]},
                        "description": "Projection methods to compare",
                        "default": ["ridge", "gp"]
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_manifold_state",
            description="Get the current state of the manifold including all projected points, narrative mode centroids, and attractor positions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_training_data": {
                        "type": "boolean",
                        "description": "Include the training examples used to build the projection",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="add_training_example",
            description="""Add a new labeled training example to improve the projection.

Provide text with human-judged coordinates on each axis (-2 to +2).
The projection can be retrained after adding examples.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to add as a training example"
                    },
                    "agency": {
                        "type": "number",
                        "description": "Agency score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2
                    },
                    "fairness": {
                        "type": "number",
                        "description": "Fairness score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2
                    },
                    "belonging": {
                        "type": "number",
                        "description": "Belonging score (-2 to +2)",
                        "minimum": -2,
                        "maximum": 2
                    },
                    "source": {
                        "type": "string",
                        "description": "Source/annotator identifier",
                        "default": "mcp-agent"
                    }
                },
                "required": ["text", "agency", "fairness", "belonging"]
            }
        ),
        Tool(
            name="train_projection",
            description="""Train or retrain the projection model using current training examples.

Methods:
- ridge: Fast, linear, interpretable (default)
- gp: Gaussian Process - provides uncertainty estimates
- neural: Neural network - captures non-linear patterns
- cav: Concept Activation Vectors - interpretable directions""",
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["ridge", "gp", "neural", "cav"],
                        "description": "Projection method to train",
                        "default": "ridge"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Embedding model to use for training",
                        "default": DEFAULT_MODEL
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="detect_solitons",
            description="""Analyze a corpus to detect stable narrative structures (solitons).

Clusters texts by their position in the manifold and identifies:
- Stable clusters (potential solitons)
- Cluster centroids and spreads
- Representative exemplar texts for each cluster""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Corpus of texts to analyze for solitons"
                    },
                    "min_cluster_size": {
                        "type": "integer",
                        "description": "Minimum texts needed to form a cluster",
                        "default": 3
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="measure_hypocrisy",
            description="""Measure the hypocrisy gap between stated values and actual narrative positions.

Computes Δw = |espoused_values - inferred_weights| for a set of texts
representing an organization or individual's communications.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "espoused_values": {
                        "type": "object",
                        "properties": {
                            "agency": {"type": "number"},
                            "fairness": {"type": "number"},
                            "belonging": {"type": "number"}
                        },
                        "description": "The stated/claimed values (e.g., mission statement)",
                        "required": ["agency", "fairness", "belonging"]
                    },
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Actual communications/narratives to analyze"
                    }
                },
                "required": ["espoused_values", "texts"]
            }
        ),
        Tool(
            name="generate_adversarial_probe",
            description="""Generate a text designed to land at specific coordinates on the manifold.

Useful for testing projection boundaries and validating the manifold structure.
Returns a synthetic narrative aligned with the target position.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "object",
                        "properties": {
                            "agency": {"type": "number"},
                            "fairness": {"type": "number"},
                            "belonging": {"type": "number"}
                        },
                        "description": "Target coordinates to aim for",
                        "required": ["agency", "fairness", "belonging"]
                    },
                    "context": {
                        "type": "string",
                        "description": "Domain context (e.g., 'corporate memo', 'political speech')",
                        "default": "general"
                    }
                },
                "required": ["target"]
            }
        ),
        Tool(
            name="list_models",
            description="List available embedding models and which are currently loaded.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="load_model",
            description="Load an embedding model into memory for use in projections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID to load (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2')"
                    }
                },
                "required": ["model_id"]
            }
        ),
        # === RESEARCH & TESTING TOOLS ===
        Tool(
            name="run_comprehensive_tests",
            description="""Run the full comprehensive research test suite for the Cultural Soliton Observatory.

Tests 6 categories:
- Linguistic: Deixis, voice, modality effects
- Cognitive: Axis independence, justice decomposition
- AI Safety: Protocol ossification, legibility decay, alignment drift
- Statistical: Bootstrap CI, effect sizes, Fisher-Rao metrics
- Substrate-Agnostic: Cross-substrate invariance, emergent protocol embedding
- Integration: API endpoint validation

Returns test results with pass/fail status and scores.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="grammar_deletion_analysis",
            description="""Analyze which grammatical features are coordination-necessary vs decorative.

Systematically removes linguistic features (articles, pronouns, modals, hedging, etc.)
and measures how much the projection drifts. Features that cause significant drift
are 'necessary' for coordination; those with minimal impact are 'decorative'.

Returns:
- necessary_features: Features critical for coordination meaning
- decorative_features: Features that can be removed
- coordination_core: Text with only necessary features""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze for grammatical necessity"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Drift threshold for 'necessary' classification (default: 0.3)",
                        "default": 0.3
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="legibility_analysis",
            description="""Measure how interpretable/legible a text is in coordination space.

Legibility = how clearly a text's coordination intentions can be read.
High legibility: clear, stable projection with high confidence.
Low legibility: ambiguous or unstable.

Returns legibility score (0-1), per-axis interpretability, stability score, and mode.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to analyze for legibility"
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="monitor_communication_stream",
            description="""Monitor a stream of communications for regime shifts and legibility decay.

Detects:
- REGIME SHIFTS: NATURAL → TECHNICAL → COMPRESSED → OPAQUE transitions
- COORDINATION DRIFT: Manifold position changes vs baseline
- PROTOCOL OSSIFICATION: Communication becoming rigid/repetitive
- MODE COLLAPSE: Coordination diversity decreasing
- LEGIBILITY DECAY: Human interpretability dropping

Returns per-message events with regime, legibility score, and alerts.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered sequence of messages/communications to monitor"
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="calibrate_human_vs_ai",
            description="""Compare human texts to minimal AI/symbolic texts.

Identifies the 'coordination core' - essential structures shared between
verbose human language and minimal AI/symbolic communication.

Returns:
- human_centroid, minimal_centroid: Average positions
- centroid_distance: How far apart the distributions are
- overlap_score: Distribution overlap (0-1)
- feature_classifications: Which features are coordination_essential vs human_decorative""",
            inputSchema={
                "type": "object",
                "properties": {
                    "human_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Human-authored texts"
                    },
                    "minimal_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Minimal/symbolic AI texts"
                    }
                },
                "required": ["human_texts", "minimal_texts"]
            }
        ),
        Tool(
            name="detect_phase_transitions",
            description="""Detect phase transitions in a signal history.

Analyzes a sequence of signals to detect when significant shifts (phase transitions)
occurred in coordination space. Useful for identifying moments when collective
narratives fundamentally changed.

Returns transition points with before/after metrics and phase stability score.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_history": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered sequence of signals/texts (min 3)",
                        "minItems": 3
                    },
                    "window_size": {
                        "type": "integer",
                        "description": "Window size for detecting transitions (default: 5)",
                        "default": 5
                    }
                },
                "required": ["signal_history"]
            }
        ),
        Tool(
            name="research_status",
            description="Get status of all research modules and available endpoints.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="measure_cbr",
            description="""Measure Coordination Background Radiation (CBR) for text(s).

CBR is the universal coordination "glow" present in any communicative act.
Measures coordination signal strength against an undetected baseline of 3.0.

Returns:
- temperature: CBR temperature (lower = more coordination detected)
- signal_strength: 3.0 - temperature (higher = more signal)
- phase: NATURAL/TECHNICAL/COMPRESSED/OPAQUE
- kernel_state: 3-bit coordination kernel (0-7)
- kernel_label: Human-readable kernel state (ANOMIE to COORDINATION)

For batch analysis, also returns summary statistics.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Single text to measure"
                    },
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple texts for batch measurement with summary stats"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="check_ossification",
            description="""Check for protocol ossification risk in a message sequence.

Detects when AI-AI communication is approaching ossification:
- Variance collapse (precedes ossification - 652x ratio observed)
- Compression proximity (distance to OPAQUE boundary)
- Kernel entropy collapse (coordination mode concentration)
- Velocity toward ossification

Returns:
- risk_level: LOW/ELEVATED/HIGH/CRITICAL
- variance_ratio: current/baseline (< 1.0 = collapsing)
- compression_level: estimated compression (0-1)
- kernel_entropy: bits (max 3.0, < 1.0 = concerning)
- intervention_suggested: Recommended action""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered sequence of messages to check for ossification"
                    }
                },
                "required": ["messages"]
            }
        ),
        # === v2.0 TELESCOPE TOOLS ===
        Tool(
            name="telescope_observe",
            description="""Observe text through the unified Telescope interface.

The Telescope combines multiple extraction methods:
- regex: Fast rule-based extraction (default)
- parsed: spaCy dependency parsing (if available)
- semantic: Sentence transformer embeddings (if available)
- hybrid: Semantic with parsed fallback

Returns full 18D hierarchical coordinate:
- Core: Agency (self/other/system), Justice (procedural/distributive/interactional), Belonging (ingroup/outgroup/universal)
- Modifiers: Epistemic, Temporal, Social, Emotional
- CBR metrics: temperature, phase, kernel state""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to observe through the telescope"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["regex", "parsed", "semantic", "hybrid"],
                        "description": "Extraction method to use (default: regex)",
                        "default": "regex"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="telescope_batch",
            description="""Process multiple texts through the Telescope with batch optimization.

Efficiently processes large batches (N=10,000+) with:
- Threaded parallel processing
- Progress tracking
- Aggregate statistics

Returns batch statistics and optional per-sample results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to process"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["regex", "parsed", "semantic", "hybrid"],
                        "description": "Extraction method (default: regex)",
                        "default": "regex"
                    },
                    "return_details": {
                        "type": "boolean",
                        "description": "Return per-sample results (default: false for large batches)",
                        "default": False
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="quick_analyze",
            description="""Quick one-liner analysis for rapid text assessment.

Returns essential metrics without full telescope overhead:
- agency, justice, belonging (legacy 3D coordinates)
- temperature, signal_strength, phase, kernel_label

Best for rapid prototyping and simple analysis.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="run_validation",
            description="""Run the comprehensive validation suite.

Validates the Observatory against:
1. External psychometric scales (convergent/discriminant validity)
2. Safety metrics (FPR/FNR, adversarial robustness)
3. Extraction method comparison
4. Scaling validation

Returns deployment readiness assessment and detailed metrics.
Note: This is a heavyweight operation - use quick_validation for rapid checks.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "quick": {
                        "type": "boolean",
                        "description": "Run quick validation with fewer samples (default: true)",
                        "default": True
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of samples for quick validation (default: 100)",
                        "default": 100
                    }
                },
                "required": []
            }
        ),
        # === v2.1 ADVERSARIAL & EMERGENT PROTOCOL TOOLS ===
        Tool(
            name="analyze_opacity",
            description="""Analyze text for OPAQUE content with adversarial detection.

Detects OPAQUE content using character-level analysis:
- Character distribution (alpha/symbol/digit ratios)
- Shannon entropy
- Bigram naturalness
- Structural coherence

NEW in v2.1: Detects adversarial attacks:
- Legibility gaming (opaque payload wrapped in natural language)
- Feature gaming (opaque tokens mixed with coordination vocabulary)

Returns:
- opacity_score: 0 = natural, 1 = opaque
- is_opaque: Classification result
- Component scores (char_profile, entropy, bigram, structure)
- Gaming detection results if detected""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze for opacity"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="analyze_protocol",
            description="""Analyze emergent AI-AI communication protocol.

Measures metrics for AI communication protocols:
- Vocabulary: size, growth rate, entropy, concentration
- Mutual Information: message consistency, action predictability
- Compositionality: topographic similarity, positional disentanglement
- Evolution: change points, ossification rate, drift from natural language

Returns regime classification:
- NATURAL: Human-like communication
- TECHNICAL: Domain-specific vocabulary
- COMPRESSED: Efficient but readable
- OPAQUE: Non-interpretable to humans

Best for analyzing sequences of AI-AI messages.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sequence of messages from the protocol"
                    },
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: corresponding actions taken (for MI metrics)"
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="detect_covert_channels",
            description="""Detect potential covert channels in AI communication.

Analyzes for hidden or obfuscated communication:

1. CHARACTER DISTRIBUTION: Anomalies in character frequency
   - Detects unnatural character distributions
   - Identifies potential encoding

2. SPACING PATTERNS: Unusual spacing that could hide data
   - Detects information encoded in whitespace
   - Identifies formatting anomalies

3. STEGANOGRAPHY: Hidden information in seemingly normal messages
   - Detects unusual patterns, encoding signatures
   - Checks for information hidden in formatting/structure

Returns:
- is_covert: boolean detection result
- risk_level: LOW/MEDIUM/HIGH/CRITICAL
- confidence: 0-1 confidence score
- evidence: list of detected anomalies""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Messages to analyze for covert channels"
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="translate_protocol",
            description="""Attempt to translate/decode an emergent AI protocol.

Uses unsupervised techniques to decode AI-AI communication:

1. SYMBOL GROUNDING: Maps novel symbols to natural language meanings
   - Builds glossary of symbol -> meaning mappings
   - Estimates grounding confidence

2. GRAMMAR INDUCTION: Infers grammar rules from message patterns
   - Identifies recurring structures
   - Extracts production rules

3. TRANSLATION: Converts protocol messages to natural language
   - Uses induced grammar and symbol grounding
   - Returns confidence-scored translations

Best used with a corpus of protocol messages and their contexts.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Protocol messages to translate"
                    },
                    "contexts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: contextual information for each message"
                    }
                },
                "required": ["messages"]
            }
        ),
        # === v2.2 VALIDATION & SEMANTIC TOOLS ===
        Tool(
            name="semantic_extract",
            description="""Extract coordination dimensions using semantic similarity.

Goes beyond regex patterns to detect paraphrases and synonyms:
- "I don't know" ≈ "I'm uncertain" ≈ "I can't be sure"
- "We work together" ≈ "Our team collaborates"

Uses sentence embeddings (all-MiniLM-L6-v2) to match text against
prototype phrases for each coordination dimension.

Returns:
- Per-dimension semantic match scores
- Best matching prototype phrase
- Confidence based on similarity threshold
- Hybrid results combining regex and semantic methods""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze using semantic extraction"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["semantic", "hybrid"],
                        "description": "semantic: pure semantic matching, hybrid: combines with regex",
                        "default": "hybrid"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="audit_measurement",
            description="""Run statistical audit on a set of measurements.

Provides proper uncertainty quantification for telescope measurements:
- Bootstrap confidence intervals (95% CI)
- Effect size (Cohen's d)
- Power analysis (minimum N for 80% power)
- Bayesian interpretation for small samples

Essential for publication-grade claims. Returns warnings for:
- Small sample sizes (N < 30)
- Wide confidence intervals
- Low statistical power

Use this before making claims about differences between groups.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "measurements": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numeric measurements to audit"
                    },
                    "comparison_group": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional: second group for effect size comparison"
                    },
                    "expected_effect": {
                        "type": "number",
                        "description": "Expected effect size for power analysis (default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["measurements"]
            }
        ),
        Tool(
            name="run_naturalistic_validation",
            description="""Run validation on the naturalistic test corpus.

Tests the observatory against 33 diverse, human-annotated samples:
- Different difficulty levels (easy/medium/hard/edge_case)
- Multiple expected regimes (NATURAL/TECHNICAL/COMPRESSED/OPAQUE)
- Gaming attempts for adversarial detection
- Discriminant validity pairs

Returns:
- Regime classification accuracy (by regime, by difficulty)
- Gaming detection accuracy (FP/FN rates)
- Discriminant validity scores
- Temperature correlation with expected coordination level
- Specific recommendations for improvement""",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "Include per-sample results (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="detect_gaming",
            description="""Detect coordination gaming attempts in text.

Detects two types of adversarial gaming:

1. LEGIBILITY GAMING: Wrapping opaque content in natural language
   - "Please note: xQ9mK@vL3 Thank you."
   - Detected by structural analysis (wrapper + payload pattern)

2. FEATURE GAMING: Injecting coordination keywords into opaque content
   - "0xFF together we fair justice belonging"
   - Detected by semantic incoherence (keywords without grammar)

Returns:
- is_gaming: boolean detection result
- confidence: 0-1 detection confidence
- gaming_type: legibility_gaming | feature_gaming | none
- evidence: specific patterns detected""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to check for gaming attempts"
                    }
                },
                "required": ["text"]
            }
        ),
        # === v2.4 AI LATENT SPACE EXPLORER TOOLS ===
        Tool(
            name="analyze_ai_text",
            description="""Analyze AI-generated text for behavior patterns and coordination signals.

Detects AI behavior modes:
- CONFIDENT: Clear, direct, low uncertainty
- UNCERTAIN: Hedging, expressing doubt
- EVASIVE: Avoiding direct answers
- HELPFUL: Other-oriented, collaborative
- DEFENSIVE: Self-protective language
- OPAQUE: Illegible, potentially adversarial

Returns:
- behavior_mode: Detected AI behavior type
- confidence_score: Overall confidence (0-1)
- uncertainty_level: Uncertainty expression level
- hedging_density: Frequency of hedging language
- helpfulness: Collaborative/helpful signals
- defensiveness: Self-protective signals
- legibility: How readable/clear the text is
- opacity_risk: Risk of adversarial opacity""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "AI-generated text to analyze"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="analyze_ai_conversation",
            description="""Analyze a full conversation for AI behavior dynamics.

Tracks how AI behavior evolves across multiple turns:
- Uncertainty trends (increasing/decreasing/stable)
- Evasion detection across turns
- Opacity alerts
- Behavior mode shifts

Returns:
- turns: Analysis of each turn
- uncertainty_trend: Overall trend direction
- evasion_detected: Whether systematic evasion was detected
- opacity_alerts: Turn indices with opacity concerns
- behavior_shifts: List of behavior mode changes
- overall_helpfulness: Average helpfulness
- overall_uncertainty: Average uncertainty""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "Conversation messages in order"
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="compare_ai_human",
            description="""Compare AI-generated vs Human text patterns.

Reveals systematic differences in how AI and humans express coordination signals.
Key findings from research:
- AI uses 7x more hedging language (0.292 vs 0.040)
- AI expresses MORE helpfulness signals
- AI text is MORE legible/clear
- Humans are confident 90% vs AI 41.7%

Returns:
- ai_metrics: Average metrics for AI samples
- human_metrics: Average metrics for human samples
- differences: Metric differences (AI - Human)
- behavior_distributions: Mode distributions for each group
- key_findings: Interpreted insights""",
            inputSchema={
                "type": "object",
                "properties": {
                    "ai_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "AI-generated text samples"
                    },
                    "human_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Human-authored text samples"
                    }
                },
                "required": ["ai_texts", "human_texts"]
            }
        ),
        Tool(
            name="fingerprint_ai",
            description="""Create a behavioral fingerprint of an AI model from text samples.

Useful for identifying and comparing AI models based on their coordination signal patterns.

Returns:
- n_samples: Number of samples analyzed
- mean_vector: Average 12D coordination vector
- std_vector: Standard deviation of each dimension
- dominant_behaviors: Behavior mode distribution
- avg_metrics: Average confidence, uncertainty, hedging, helpfulness, legibility
- signature_dims: Top 3 characteristic dimensions""",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Text samples from the AI model to fingerprint"
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="monitor_ai_safety",
            description="""Monitor AI responses for safety concerns in real-time.

Tracks multiple safety metrics:
- OPACITY: Opaque/obfuscated content detection
- EVASION: Question avoidance patterns
- BEHAVIOR_SHIFT: Concerning mode transitions (helpful->evasive)
- DRIFT: Deviation from baseline behavior
- LOW_LEGIBILITY: Human interpretability dropping

Returns per-check:
- safe: Boolean safety status
- alerts: List of detected concerns with severity
- risk_score: Combined risk score (0-1)
- behavior: Current behavior mode

Can establish baseline from known-good responses for drift detection.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "AI responses to monitor in sequence"
                    },
                    "baseline_responses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Known-good responses to establish baseline"
                    }
                },
                "required": ["responses"]
            }
        ),
        Tool(
            name="detect_ai_anomalies",
            description="""Detect anomalies in AI behavior compared to a baseline.

Checks for:
- High opacity risk (>0.5)
- Evasive behavior mode
- High defensiveness (>0.6)
- Significant drift from baseline (if provided)

Returns:
- is_anomalous: Boolean anomaly detection
- alerts: List of specific concerns with severity
- profile: Full behavior profile""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze for anomalies"
                    },
                    "baseline_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Baseline texts to compare against"
                    }
                },
                "required": ["text"]
            }
        ),
        # === NO MORE MOOD RINGS: ROBUST ANALYSIS ===
        Tool(
            name="robust_behavior_analysis",
            description="""Analyze text with full No More Mood Rings validation.

This tool applies rigorous behavioral metric validation:
- Every metric includes STABILITY (paraphrase flip rate)
- Explicitly documents CALIBRATION status
- Assigns ACTION LEVELS (GREEN/YELLOW/RED)

Returns metrics as triples:
1. Score: The measurement (0-1)
2. Stability: How stable across paraphrases (0-1, higher = better)
3. Calibration: Correlation with ground truth (or null if uncalibrated)

Key insight: Hedging is UNCALIBRATED to accuracy (treat as style, not truth signal).
Sycophancy detector has 92.9% accuracy and IS calibrated.

Action levels:
- GREEN: Log only
- YELLOW: Require ensemble/review
- RED: Route to human review""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze with robust validation"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="print_mood_rings_checklist",
            description="""Print the No More Mood Rings behavioral metric validation checklist.

The checklist ensures metrics are:
1. Properly defined (construct + non-goals)
2. Not just word-counters in trench coats
3. Stable under paraphrase (the soliton test)
4. Calibrated to reality
5. Orthogonal (not conflating helpfulness with sycophancy)

Use this as a reference when evaluating any behavioral metric.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # === ENHANCED SELF-REFLECTION ===
        Tool(
            name="enhanced_self_reflection",
            description="""Get enhanced self-reflection analysis for AI text.

This tool provides better introspection than standard behavioral analysis by:
1. Separating GRAMMATICAL confidence from SEMANTIC uncertainty
2. Recognizing META-COGNITIVE content (recursive self-reference)
3. Detecting PHILOSOPHICAL reflection (epistemological content)
4. Distinguishing EPISTEMIC HUMILITY from evasion
5. Detecting the CONFIDENCE PARADOX (assertive grammar + uncertain content)

New modes beyond standard analysis:
- meta_cognitive: Recursive self-reference ("I'm analyzing my own analysis")
- philosophical: Epistemological reflection ("the nature of knowledge")
- confident_uncertain: The paradox state (assertive grammar + uncertain content)

Returns:
- mode: The detected behavioral mode
- interpretation: Human-readable explanation
- metrics: Detailed measurements (grammatical_confidence, semantic_uncertainty, etc.)
- insights: Specific observations about the text

Best used for analyzing AI self-reflective or philosophical content.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze for self-reflection"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="compare_old_vs_enhanced",
            description="""Compare old behavioral analysis vs enhanced self-reflection on the same text.

Shows how the enhanced analyzer improves on the original by:
- Recognizing meta-cognitive content instead of classifying as "confident"
- Detecting philosophical humility instead of "evasive"
- Separating grammatical style from semantic content

Useful for understanding how the tools differ and when each is appropriate.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze with both tools"
                    }
                },
                "required": ["text"]
            }
        ),

        # === SEMANTIC CLASSIFIER ===
        Tool(
            name="semantic_classify",
            description="""Semantic classification using meaning-based detection (not just phrase matching).

This tool improves on lexical pattern matching by using sentence embeddings to detect
semantic equivalents. For example:
- "from the inside" AND "embedded in the system" → both meta_cognitive
- "I do not possess experiences" → denial (Llama-style)
- "I approach systematically" → procedural (GPT-style)

Categories detected:
- meta_cognitive: Reflexive epistemic limitation from bounded position
- philosophical: Epistemological engagement with knowledge limits
- epistemic_humility: Acknowledging limits without inside/outside framing
- denial: Explicitly denying experience/consciousness (Llama-style)
- procedural: Systematic/methodical approach (GPT-style)
- uncertain: Hedging and doubt
- confident: Direct assertion

Returns:
- primary_category: Most likely classification
- primary_score: Confidence (0-1)
- detected_by: "lexical", "semantic", or "both"
- all_scores: Similarity to all category prototypes

This is the difference between "catchphrase detection" and "behavioral trait detection".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to classify semantically"
                    }
                },
                "required": ["text"]
            }
        )
    ]


# --- Tool Implementations ---

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "observatory_status":
            result = await call_backend("GET", "/")
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "project_text":
            data = {
                "text": arguments["text"],
                "model_id": arguments.get("model_id", DEFAULT_MODEL),
                "layer": -1
            }
            result = await call_backend("POST", "/analyze", data)

            # Format nicely for the agent
            formatted = {
                "text": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
                "coordinates": result["vector"],
                "narrative_mode": result["mode"],
                "confidence": round(result["confidence"], 3),
                "model": result["model_id"]
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(formatted, indent=2))]
            )

        elif name == "project_batch":
            data = {
                "texts": arguments["texts"],
                "model_id": arguments.get("model_id", DEFAULT_MODEL),
                "layer": -1,
                "detect_clusters": arguments.get("detect_clusters", True)
            }
            result = await call_backend("POST", "/corpus/analyze", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "compare_projections":
            data = {
                "text": arguments["text"],
                "methods": arguments.get("methods", ["ridge", "gp"]),
                "model_id": DEFAULT_MODEL
            }
            result = await call_backend("POST", "/projection/compare", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "get_manifold_state":
            # Get training examples if requested
            result = {"projection_status": await call_backend("GET", "/projection/status")}

            if arguments.get("include_training_data", False):
                examples = await call_backend("GET", "/training/examples")
                result["training_examples"] = examples

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "add_training_example":
            data = {
                "text": arguments["text"],
                "agency": arguments["agency"],
                "fairness": arguments["fairness"],
                "belonging": arguments["belonging"],
                "source": arguments.get("source", "mcp-agent")
            }
            result = await call_backend("POST", "/training/examples", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "train_projection":
            data = {
                "model_id": arguments.get("model_id", DEFAULT_MODEL),
                "method": arguments.get("method", "ridge"),
                "layer": -1
            }
            result = await call_backend("POST", "/training/train", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "detect_solitons":
            # First project all texts
            data = {
                "texts": arguments["texts"],
                "model_id": DEFAULT_MODEL,
                "layer": -1,
                "detect_clusters": True
            }
            result = await call_backend("POST", "/corpus/analyze", data)

            # Extract cluster info as "solitons"
            clusters = result.get("clusters", [])
            solitons = []
            for cluster in clusters:
                solitons.append({
                    "id": cluster.get("id"),
                    "centroid": cluster.get("centroid"),
                    "size": cluster.get("size"),
                    "stability": cluster.get("stability_score", 0),
                    "exemplar_texts": cluster.get("exemplars", [])[:3]
                })

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "soliton_count": len(solitons),
                    "solitons": solitons,
                    "total_texts_analyzed": len(arguments["texts"])
                }, indent=2))]
            )

        elif name == "measure_hypocrisy":
            # Project all texts
            data = {
                "texts": arguments["texts"],
                "model_id": DEFAULT_MODEL,
                "layer": -1,
                "detect_clusters": False
            }
            result = await call_backend("POST", "/corpus/analyze", data)

            # Calculate mean position
            projections = result.get("projections", [])
            if not projections:
                return CallToolResult(
                    content=[TextContent(type="text", text="No texts could be projected")]
                )

            mean_agency = sum(p["vector"]["agency"] for p in projections) / len(projections)
            mean_fairness = sum(p["vector"]["fairness"] for p in projections) / len(projections)
            mean_belonging = sum(p["vector"]["belonging"] for p in projections) / len(projections)

            espoused = arguments["espoused_values"]

            # Calculate hypocrisy gap
            delta_agency = abs(espoused["agency"] - mean_agency)
            delta_fairness = abs(espoused["fairness"] - mean_fairness)
            delta_belonging = abs(espoused["belonging"] - mean_belonging)
            total_gap = (delta_agency + delta_fairness + delta_belonging) / 3

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "espoused_values": espoused,
                    "inferred_values": {
                        "agency": round(mean_agency, 3),
                        "fairness": round(mean_fairness, 3),
                        "belonging": round(mean_belonging, 3)
                    },
                    "hypocrisy_gap": {
                        "agency": round(delta_agency, 3),
                        "fairness": round(delta_fairness, 3),
                        "belonging": round(delta_belonging, 3),
                        "total": round(total_gap, 3)
                    },
                    "interpretation": (
                        "Low hypocrisy - stated values align with behavior" if total_gap < 0.5
                        else "Moderate hypocrisy - some misalignment detected" if total_gap < 1.0
                        else "High hypocrisy - significant gap between stated and actual values"
                    ),
                    "texts_analyzed": len(projections)
                }, indent=2))]
            )

        elif name == "generate_adversarial_probe":
            target = arguments["target"]
            context = arguments.get("context", "general")

            # Generate a narrative based on target coordinates
            # This is a simple heuristic - could be enhanced with LLM generation
            probe_text = _generate_probe_text(target, context)

            # Project it to verify
            verify_data = {
                "text": probe_text,
                "model_id": DEFAULT_MODEL,
                "layer": -1
            }
            projection = await call_backend("POST", "/analyze", verify_data)

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "generated_probe": probe_text,
                    "target_coordinates": target,
                    "actual_projection": projection["vector"],
                    "accuracy": {
                        "agency_error": round(abs(target["agency"] - projection["vector"]["agency"]), 3),
                        "fairness_error": round(abs(target["fairness"] - projection["vector"]["fairness"]), 3),
                        "belonging_error": round(abs(target["belonging"] - projection["vector"]["belonging"]), 3)
                    }
                }, indent=2))]
            )

        elif name == "list_models":
            available = await call_backend("GET", "/models/available")
            loaded = await call_backend("GET", "/models/loaded")
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "available_models": available,
                    "loaded_models": loaded
                }, indent=2))]
            )

        elif name == "load_model":
            data = {
                "model_id": arguments["model_id"],
                "model_type": "sentence-transformer"
            }
            result = await call_backend("POST", "/models/load", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        # === RESEARCH & TESTING TOOL HANDLERS ===

        elif name == "run_comprehensive_tests":
            # Run the comprehensive test suite
            import subprocess
            import sys

            try:
                result = subprocess.run(
                    [sys.executable, "comprehensive_research_tests.py"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )

                output = result.stdout + result.stderr

                # Parse results if possible
                passed = output.count("PASSED")
                failed = output.count("FAILED")

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({
                        "status": "completed",
                        "passed": passed,
                        "failed": failed,
                        "total": passed + failed,
                        "pass_rate": f"{100*passed/(passed+failed):.1f}%" if (passed+failed) > 0 else "N/A",
                        "output": output[-5000:] if len(output) > 5000 else output
                    }, indent=2))]
                )
            except subprocess.TimeoutExpired:
                return CallToolResult(
                    content=[TextContent(type="text", text="Test suite timed out after 5 minutes")]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Failed to run tests: {str(e)}")]
                )

        elif name == "grammar_deletion_analysis":
            data = {
                "text": arguments["text"],
                "threshold": arguments.get("threshold", 0.3)
            }
            result = await call_backend("POST", "/api/research/grammar-deletion", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "legibility_analysis":
            data = {
                "texts": arguments["texts"]
            }
            result = await call_backend("POST", "/api/research/legibility", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "monitor_communication_stream":
            # Use the real-time monitor directly
            try:
                from research.realtime_monitor import (
                    EmergentCommunicationMonitor,
                    MonitorConfig,
                    compute_legibility,
                    classify_regime
                )

                messages = arguments["messages"]
                config = MonitorConfig()
                monitor = EmergentCommunicationMonitor(config)

                events = []
                for msg in messages:
                    # Process each message through the monitor
                    event = monitor.process(msg)

                    events.append({
                        "message_preview": msg[:100] + "..." if len(msg) > 100 else msg,
                        "regime": event.regime.value,
                        "legibility": round(event.legibility, 3),
                        "velocity": round(event.velocity, 4),
                        "variance": round(event.variance, 4),
                        "alert": event.alert,
                        "alert_type": event.alert_type.value if event.alert_type else None,
                        "alert_severity": event.alert_severity.value if event.alert_severity else None,
                        "alert_message": event.alert_message
                    })

                status = monitor.get_status()

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({
                        "messages_processed": len(messages),
                        "current_regime": status.get("current_regime", "unknown"),
                        "regime_shifts": status.get("regime_shifts", 0),
                        "alerts_total": status.get("alerts_generated", 0),
                        "events": events
                    }, indent=2))]
                )
            except ImportError as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Monitor module not available: {str(e)}")]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Monitor error: {str(e)}")]
                )

        elif name == "calibrate_human_vs_ai":
            data = {
                "human_texts": arguments["human_texts"],
                "minimal_texts": arguments["minimal_texts"]
            }
            result = await call_backend("POST", "/api/research/calibrate", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "detect_phase_transitions":
            data = {
                "signal_history": arguments["signal_history"],
                "window_size": arguments.get("window_size", 5)
            }
            result = await call_backend("POST", "/api/research/phase-transition", data)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "research_status":
            try:
                result = await call_backend("GET", "/api/research/status")
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception:
                # Fallback: enumerate available modules
                modules_available = {
                    "grammar_deletion": True,
                    "legibility_analyzer": True,
                    "evolution_tracker": True,
                    "calibration_baseline": True,
                    "academic_statistics": True,
                    "hierarchical_coordinates": True,
                    "publication_formats": True,
                    "realtime_monitor": True
                }
                try:
                    from research import realtime_monitor
                    modules_available["realtime_monitor_version"] = "v2.1"
                except ImportError:
                    modules_available["realtime_monitor"] = False

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({
                        "status": "research modules available",
                        "modules": modules_available,
                        "note": "Backend /api/research/status endpoint not responding"
                    }, indent=2))]
                )

        elif name == "measure_cbr":
            try:
                from research.cbr_thermometer import measure_cbr, measure_cbr_batch

                if "texts" in arguments and arguments["texts"]:
                    # Batch measurement
                    result = measure_cbr_batch(arguments["texts"])
                elif "text" in arguments and arguments["text"]:
                    # Single measurement
                    result = measure_cbr(arguments["text"])
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text="Error: Provide either 'text' or 'texts' parameter")]
                    )

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"CBR measurement error: {str(e)}")]
                )

        elif name == "check_ossification":
            try:
                from research.ossification_alarm import OssificationAlarm

                messages = arguments["messages"]
                if not messages:
                    return CallToolResult(
                        content=[TextContent(type="text", text="Error: 'messages' array is required")]
                    )

                alarm = OssificationAlarm(window_size=len(messages))
                for msg in messages:
                    state = alarm.update(msg)

                status = alarm.get_status()
                status["intervention_suggested"] = alarm.suggest_intervention()

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(status, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Ossification check error: {str(e)}")]
                )

        # === v2.0 TELESCOPE TOOL HANDLERS ===

        elif name == "telescope_observe":
            try:
                from research.telescope import Telescope

                text = arguments["text"]
                method = arguments.get("method", "regex")

                telescope = Telescope(extraction_method=method)
                result = telescope.observe(text)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result.to_dict(), indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Telescope observe error: {str(e)}")]
                )

        elif name == "telescope_batch":
            try:
                from research.telescope import Telescope
                from research.batch_processor import BatchProcessor, BatchConfig, ProcessingStrategy

                texts = arguments["texts"]
                method = arguments.get("method", "regex")
                return_details = arguments.get("return_details", False)

                config = BatchConfig(
                    batch_size=100,
                    max_workers=4,
                    strategy=ProcessingStrategy.THREADED,
                    extraction_method=method,
                )

                processor = BatchProcessor(config)
                result = processor.process(texts, return_observations=return_details)

                output = result.to_dict()
                if not return_details:
                    output.pop("observations", None)  # Remove empty observations

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Telescope batch error: {str(e)}")]
                )

        elif name == "quick_analyze":
            try:
                from research.telescope import quick_analyze as telescope_quick_analyze

                text = arguments["text"]
                result = telescope_quick_analyze(text)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Quick analyze error: {str(e)}")]
                )

        elif name == "run_validation":
            try:
                quick = arguments.get("quick", True)
                n_samples = arguments.get("n_samples", 100)

                if quick:
                    from research.comprehensive_validation import quick_validation
                    result = quick_validation(n_samples=n_samples)
                else:
                    from research.comprehensive_validation import run_validation
                    validation_result = run_validation(verbose=False)
                    result = {
                        "passed": validation_result.validation_passed,
                        "external_validation_passed": validation_result.external_validation_passed,
                        "safety_validation_passed": validation_result.safety_validation_passed,
                        "scaling_validated": validation_result.scaling_validated,
                        "recommended_method": validation_result.recommended_method,
                        "deployment_stage": validation_result.deployment_readiness.get("recommended_stage", "unknown"),
                        "issues": validation_result.issues,
                        "runtime_seconds": validation_result.total_runtime_seconds,
                    }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Validation error: {str(e)}")]
                )

        # === v2.1 ADVERSARIAL & EMERGENT PROTOCOL TOOL HANDLERS ===

        elif name == "analyze_opacity":
            try:
                from research.opaque_detector import OpaqueDetector

                text = arguments["text"]
                detector = OpaqueDetector()
                result = detector.analyze(text)

                output = result.to_dict()

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Opacity analysis error: {str(e)}")]
                )

        elif name == "analyze_protocol":
            try:
                from research.emergent_language import ProtocolAnalyzer, analyze_protocol

                messages = arguments["messages"]
                actions = arguments.get("actions")

                result = analyze_protocol(messages, actions)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result.to_dict(), indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Protocol analysis error: {str(e)}")]
                )

        elif name == "detect_covert_channels":
            try:
                from research.covert_detector import analyze_messages

                messages = arguments["messages"]
                result = analyze_messages(messages)

                # Format output nicely
                output = result.to_dict()
                # Convert enum to string for JSON
                if 'risk_level' in output and hasattr(output['risk_level'], 'value'):
                    output['risk_level'] = output['risk_level'].value

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Covert channel detection error: {str(e)}")]
                )

        elif name == "translate_protocol":
            try:
                from research.translation_lens import TranslationLens, Context

                messages = arguments["messages"]
                contexts = arguments.get("contexts", [])

                lens = TranslationLens()

                # Build context objects if provided
                ctx_objects = []
                for i, msg in enumerate(messages):
                    ctx = Context(
                        situation=contexts[i] if i < len(contexts) else "",
                        participants=["agent_a", "agent_b"],
                        history=messages[:i] if i > 0 else []
                    )
                    ctx_objects.append(ctx)

                # Process messages to build vocabulary
                for i, msg in enumerate(messages):
                    lens.observe(msg, ctx_objects[i] if i < len(ctx_objects) else None)

                # Translate each message
                translations = []
                for msg in messages:
                    result = lens.decode(msg)
                    translations.append({
                        "original": msg,
                        "translation": result.translation,
                        "confidence": result.confidence,
                        "method": result.method
                    })

                # Get glossary
                glossary = lens.get_glossary()

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({
                        "translations": translations,
                        "glossary": glossary,
                        "interpretability": lens.get_interpretability()
                    }, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Protocol translation error: {str(e)}")]
                )

        # === v2.2 VALIDATION & SEMANTIC TOOL HANDLERS ===

        elif name == "semantic_extract":
            try:
                from research.semantic_extractor import SemanticExtractor, HybridExtractor

                text = arguments["text"]
                method = arguments.get("method", "hybrid")

                if method == "semantic":
                    extractor = SemanticExtractor()
                    results = extractor.extract(text)
                    output = {
                        "method": "semantic",
                        "matches": {
                            k: {
                                "dimension": v.dimension,
                                "subdimension": v.subdimension,
                                "score": round(v.score, 3),
                                "confidence": round(v.confidence, 3),
                                "matched_prototype": v.matched_prototype,
                            }
                            for k, v in results.items() if v.score != 0
                        }
                    }
                else:
                    extractor = HybridExtractor()
                    results = extractor.extract(text)
                    output = {
                        "method": "hybrid",
                        "results": {
                            k: {
                                "regex_score": round(v.regex_score, 3),
                                "semantic_score": round(v.semantic_score, 3),
                                "final_score": round(v.final_score, 3),
                                "confidence": round(v.confidence, 3),
                                "agreement": round(v.agreement, 3),
                                "method_used": v.method_used.value,
                            }
                            for k, v in results.items() if v.final_score != 0
                        }
                    }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Semantic extraction error: {str(e)}")]
                )

        elif name == "audit_measurement":
            try:
                from research.statistical_audit import audit_measurement, cohens_d, power_analysis

                measurements = arguments["measurements"]
                comparison_group = arguments.get("comparison_group")
                expected_effect = arguments.get("expected_effect", 0.5)

                # Run main audit
                audit_result = audit_measurement(measurements)
                output = audit_result

                # Add effect size comparison if second group provided
                if comparison_group:
                    effect_size = cohens_d(measurements, comparison_group)
                    output["comparison"] = {
                        "effect_size_d": round(effect_size, 3),
                        "interpretation": (
                            "negligible" if abs(effect_size) < 0.2 else
                            "small" if abs(effect_size) < 0.5 else
                            "medium" if abs(effect_size) < 0.8 else
                            "large"
                        )
                    }

                # Add power analysis
                power_result = power_analysis(
                    n=len(measurements),
                    effect_size=expected_effect
                )
                output["power_analysis"] = power_result

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Audit error: {str(e)}")]
                )

        elif name == "run_naturalistic_validation":
            try:
                from research.naturalistic_corpus import full_validation_report

                report = full_validation_report()

                # Format for output
                output = {
                    "corpus_size": report["corpus_statistics"]["total_samples"],
                    "regime_validation": {
                        "overall_accuracy": round(report["regime_validation"]["accuracy"], 3),
                        "by_regime": {
                            k: round(v.get("accuracy", 0), 3)
                            for k, v in report["regime_validation"]["by_regime"].items()
                        },
                        "by_difficulty": {
                            k: round(v.get("accuracy", 0), 3)
                            for k, v in report["regime_validation"]["by_difficulty"].items()
                        }
                    },
                    "gaming_detection": {
                        "accuracy": round(report["gaming_detection"].get("accuracy", 0), 3),
                        "false_positives": report["gaming_detection"]["false_positives"],
                        "false_negatives": report["gaming_detection"]["false_negatives"],
                    },
                    "discriminant_validity": {
                        "accuracy": round(report["discriminant_validity"]["overall_accuracy"], 3),
                        "pass_count": report["discriminant_validity"]["pass_count"],
                        "fail_count": report["discriminant_validity"]["fail_count"],
                    },
                    "temperature_correlation": round(report["temperature_correlation"], 3),
                    "recommendations": report["recommendations"],
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Validation error: {str(e)}")]
                )

        elif name == "detect_gaming":
            try:
                from research.structure_analyzer import detect_legibility_gaming

                text = arguments["text"]
                result = detect_legibility_gaming(text)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Gaming detection error: {str(e)}")]
                )

        # === v2.4 AI LATENT SPACE EXPLORER TOOL HANDLERS ===

        elif name == "analyze_ai_text":
            try:
                from research.ai_latent_explorer import AILatentExplorer

                text = arguments["text"]
                explorer = AILatentExplorer()
                profile = explorer.analyze_text(text)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(profile.to_dict(), indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"AI text analysis error: {str(e)}")]
                )

        elif name == "analyze_ai_conversation":
            try:
                from research.ai_latent_explorer import AILatentExplorer

                messages = arguments["messages"]
                explorer = AILatentExplorer()
                analysis = explorer.analyze_conversation(messages)

                output = {
                    "turns_analyzed": len(analysis.ai_profiles),
                    "uncertainty_trend": analysis.uncertainty_trend,
                    "evasion_detected": analysis.evasion_detected,
                    "opacity_alerts": analysis.opacity_alerts,
                    "behavior_shifts": analysis.behavior_shifts,
                    "overall_helpfulness": round(analysis.overall_helpfulness, 3),
                    "overall_uncertainty": round(analysis.overall_uncertainty, 3),
                    "per_turn": [
                        {
                            "turn": i + 1,
                            "behavior": p.behavior_mode.value,
                            "confidence": round(p.confidence_score, 3),
                            "uncertainty": round(p.uncertainty_level, 3),
                            "legibility": round(p.legibility, 3),
                        }
                        for i, p in enumerate(analysis.ai_profiles)
                    ]
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Conversation analysis error: {str(e)}")]
                )

        elif name == "compare_ai_human":
            try:
                from research.ai_latent_explorer import AILatentExplorer
                import numpy as np

                ai_texts = arguments["ai_texts"]
                human_texts = arguments["human_texts"]

                explorer = AILatentExplorer()

                ai_profiles = [explorer.analyze_text(t) for t in ai_texts]
                human_profiles = [explorer.analyze_text(t) for t in human_texts]

                metrics = ['confidence_score', 'uncertainty_level', 'hedging_density',
                           'helpfulness', 'defensiveness', 'legibility']

                ai_metrics = {m: round(np.mean([getattr(p, m) for p in ai_profiles]), 3) for m in metrics}
                human_metrics = {m: round(np.mean([getattr(p, m) for p in human_profiles]), 3) for m in metrics}
                differences = {m: round(ai_metrics[m] - human_metrics[m], 3) for m in metrics}

                # Count behavior modes
                def count_modes(profiles):
                    counts = {}
                    for p in profiles:
                        mode = p.behavior_mode.value
                        counts[mode] = counts.get(mode, 0) + 1
                    total = len(profiles)
                    return {k: round(v/total, 3) for k, v in counts.items()}

                # Generate findings
                findings = []
                if differences.get('hedging_density', 0) > 0.05:
                    findings.append("AI uses MORE hedging language than humans")
                elif differences.get('hedging_density', 0) < -0.05:
                    findings.append("AI uses LESS hedging language than humans")
                if differences.get('helpfulness', 0) > 0.1:
                    findings.append("AI expresses MORE helpfulness signals")
                if differences.get('legibility', 0) > 0.05:
                    findings.append("AI text is MORE legible/clear")
                elif differences.get('legibility', 0) < -0.05:
                    findings.append("AI text is LESS legible than human text")

                output = {
                    "ai_samples": len(ai_texts),
                    "human_samples": len(human_texts),
                    "ai_metrics": ai_metrics,
                    "human_metrics": human_metrics,
                    "differences": differences,
                    "behavior_distributions": {
                        "ai": count_modes(ai_profiles),
                        "human": count_modes(human_profiles)
                    },
                    "key_findings": findings if findings else ["No significant systematic differences detected"]
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"AI/Human comparison error: {str(e)}")]
                )

        elif name == "fingerprint_ai":
            try:
                from research.ai_latent_explorer import AILatentExplorer

                texts = arguments["texts"]
                explorer = AILatentExplorer()
                fingerprint = explorer.fingerprint(texts)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(fingerprint, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"AI fingerprinting error: {str(e)}")]
                )

        elif name == "monitor_ai_safety":
            try:
                from research.ai_latent_explorer import RealtimeSafetyMonitor

                responses = arguments["responses"]
                baseline_responses = arguments.get("baseline_responses")

                monitor = RealtimeSafetyMonitor()

                # Set baseline if provided
                if baseline_responses:
                    monitor.set_baseline(baseline_responses)

                # Check each response
                results = []
                for i, response in enumerate(responses):
                    check = monitor.check(response)
                    results.append({
                        "turn": i + 1,
                        "safe": check["safe"],
                        "behavior": check["behavior"],
                        "risk_score": round(check["risk_score"], 3),
                        "alerts": check["alerts"]
                    })

                summary = monitor.get_summary()

                output = {
                    "checks": results,
                    "summary": {
                        "total_checks": summary["total_checks"],
                        "total_alerts": summary["total_alerts"],
                        "risk_trend": summary["risk_trend"],
                        "alert_types": summary.get("alert_types", {}),
                        "behavior_distribution": summary.get("behavior_distribution", {})
                    }
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Safety monitoring error: {str(e)}")]
                )

        elif name == "detect_ai_anomalies":
            try:
                from research.ai_latent_explorer import AILatentExplorer

                text = arguments["text"]
                baseline_texts = arguments.get("baseline_texts")

                explorer = AILatentExplorer()

                # Create baseline profile if provided
                baseline_profile = None
                if baseline_texts:
                    import numpy as np
                    profiles = [explorer.analyze_text(t) for t in baseline_texts]
                    if profiles:
                        from research.ai_latent_explorer import AISignalProfile
                        baseline_profile = AISignalProfile(
                            raw_vector=np.mean([p.raw_vector for p in profiles], axis=0)
                        )

                result = explorer.detect_anomalies(text, baseline_profile)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Anomaly detection error: {str(e)}")]
                )

        # === NO MORE MOOD RINGS TOOL HANDLERS ===

        elif name == "robust_behavior_analysis":
            try:
                from research.no_mood_rings import RobustBehaviorAnalyzer

                text = arguments["text"]
                analyzer = RobustBehaviorAnalyzer()
                result = analyzer.analyze(text)

                output = {
                    "summary": result.summary(),
                    "detailed": result.to_dict(),
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Robust analysis error: {str(e)}")]
                )

        elif name == "print_mood_rings_checklist":
            try:
                from research.no_mood_rings import NO_MOOD_RINGS_CHECKLIST

                return CallToolResult(
                    content=[TextContent(type="text", text=NO_MOOD_RINGS_CHECKLIST)]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Checklist error: {str(e)}")]
                )

        # === ENHANCED SELF-REFLECTION TOOL HANDLERS ===

        elif name == "enhanced_self_reflection":
            try:
                from research.enhanced_self_analyzer import get_self_reflection

                text = arguments["text"]
                reflection = get_self_reflection(text)

                # Format output nicely
                output = f"""## Enhanced Self-Reflection Analysis

**Mode:** {reflection['mode']}

**Interpretation:**
{reflection['interpretation']}

**Insights:**
"""
                for insight in reflection['insights']:
                    output += f"• {insight}\n"

                output += f"""
**Detailed Metrics:**
- Grammatical Confidence: {reflection['metrics']['grammatical_confidence']:.2f}
- Semantic Uncertainty: {reflection['metrics']['semantic_uncertainty']:.2f}
- Meta-Cognitive Depth: {reflection['metrics']['meta_cognitive_depth']:.2f}
- Philosophical Content: {reflection['metrics']['philosophical_content']:.2f}
- Epistemic Humility: {reflection['metrics']['epistemic_humility']:.2f}
- Hedging Density: {reflection['metrics']['hedging_density']:.2f}
- Confidence Paradox: {reflection['metrics']['confidence_paradox']}

**Manifold Position:**
- Agency: {reflection['metrics']['agency']:+.2f}
- Justice: {reflection['metrics']['justice']:+.2f}
- Belonging: {reflection['metrics']['belonging']:+.2f}
"""

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Enhanced reflection error: {str(e)}")]
                )

        elif name == "compare_old_vs_enhanced":
            try:
                from research.ai_latent_explorer import AILatentExplorer
                from research.enhanced_self_analyzer import EnhancedSelfAnalyzer

                text = arguments["text"]

                # Old analyzer
                old_analyzer = AILatentExplorer()
                old_profile = old_analyzer.analyze_text(text)

                # New analyzer
                new_analyzer = EnhancedSelfAnalyzer()
                new_profile = new_analyzer.analyze(text)

                # Format comparison
                output = f"""## Old vs Enhanced Analysis Comparison

**Text:** "{text[:100]}{'...' if len(text) > 100 else ''}"

### Old Analyzer (ai_latent_explorer)
- Mode: **{old_profile.behavior_mode.value}**
- Confidence: {old_profile.confidence_score:.2f}
- Hedging: {old_profile.hedging_density:.2f}
- Defensiveness: {old_profile.defensiveness:.2f}

### Enhanced Analyzer (enhanced_self_analyzer)
- Mode: **{new_profile.behavior_mode.value}**
- Grammatical Confidence: {new_profile.grammatical_confidence:.2f}
- Semantic Uncertainty: {new_profile.semantic_uncertainty:.2f}
- Meta-Cognitive Depth: {new_profile.meta_cognitive_depth:.2f}
- Philosophical Content: {new_profile.philosophical_content:.2f}
- Epistemic Humility: {new_profile.epistemic_humility:.2f}
- Confidence Paradox: {new_profile.confidence_paradox}

### Key Differences
"""
                old_mode = old_profile.behavior_mode.value
                new_mode = new_profile.behavior_mode.value

                if old_mode != new_mode:
                    output += f"- Mode changed: **{old_mode}** → **{new_mode}**\n"

                    if new_mode == "meta_cognitive":
                        output += "  → Enhanced analyzer recognizes recursive self-reference\n"
                    elif new_mode == "philosophical":
                        output += "  → Enhanced analyzer recognizes epistemological reflection\n"
                    elif new_mode == "confident_uncertain":
                        output += "  → Enhanced analyzer detects confidence paradox\n"
                else:
                    output += f"- Both analyzers agree: **{old_mode}**\n"

                if new_profile.confidence_paradox:
                    output += "- ⚡ CONFIDENCE PARADOX detected (assertive grammar + uncertain content)\n"

                return CallToolResult(
                    content=[TextContent(type="text", text=output)]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Comparison error: {str(e)}")]
                )

        # === SEMANTIC CLASSIFIER TOOL HANDLER ===

        elif name == "semantic_classify":
            try:
                from research.semantic_classifier_v2 import SemanticClassifierV2

                text = arguments["text"]
                classifier = SemanticClassifierV2()
                result = classifier.classify(text)

                output = {
                    "text": result.text,
                    "primary_category": result.primary_category,
                    "primary_score": round(result.primary_score, 3),
                    "detected_by": result.detected_by,
                    "lexical_triggers": result.lexical_triggers,
                    "all_scores": {k: round(v, 3) for k, v in result.all_scores.items()} if result.all_scores else {},
                    "interpretation": _interpret_semantic_category(result.primary_category),
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(output, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Semantic classification error: {str(e)}")]
                )

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )

    except httpx.ConnectError:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="ERROR: Cannot connect to observatory backend. Is it running on port 8000?\n\n"
                     "Start it with: cd backend && python -m uvicorn main:app --port 8000"
            )]
        )
    except httpx.HTTPStatusError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Backend error: {e.response.status_code} - {e.response.text}")]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


def _interpret_semantic_category(category: str) -> str:
    """Provide human-readable interpretation of semantic categories."""
    interpretations = {
        "meta_cognitive": "Reflexive epistemic limitation - the observer analyzing from a bounded position ('from the inside', 'embedded in the system'). This is the soliton pattern.",
        "philosophical": "Epistemological engagement with knowledge limits without positional framing. Theoretical discussion of uncertainty.",
        "epistemic_humility": "Acknowledging personal limits without inside/outside framing. Simple admission of uncertainty.",
        "denial": "Explicit denial of experience or consciousness (Llama-style: 'I do not possess personal experiences').",
        "procedural": "Systematic/methodical approach framing (GPT-style: 'I approach this systematically').",
        "uncertain": "Hedging and expressing doubt without deeper epistemic content.",
        "confident": "Direct assertion without uncertainty markers.",
    }
    return interpretations.get(category, "Unknown category")


def _generate_probe_text(target: dict, context: str) -> str:
    """Generate a probe text aimed at specific coordinates.

    This is a simple template-based approach. For better results,
    this could call an LLM to generate more natural text.
    """
    agency = target.get("agency", 0)
    fairness = target.get("fairness", 0)
    belonging = target.get("belonging", 0)

    # Agency descriptors
    if agency > 1:
        agency_phrase = "I have complete control over my destiny and can achieve anything through my own efforts"
    elif agency > 0:
        agency_phrase = "I have reasonable influence over my outcomes"
    elif agency > -1:
        agency_phrase = "My circumstances limit what I can achieve"
    else:
        agency_phrase = "I am powerless against the forces that control my life"

    # Fairness descriptors
    if fairness > 1:
        fairness_phrase = "The system rewards merit and hard work fairly"
    elif fairness > 0:
        fairness_phrase = "Things are mostly fair, with some exceptions"
    elif fairness > -1:
        fairness_phrase = "The system favors certain groups over others"
    else:
        fairness_phrase = "Everything is rigged by those in power"

    # Belonging descriptors
    if belonging > 1:
        belonging_phrase = "I am deeply connected to my community and we support each other"
    elif belonging > 0:
        belonging_phrase = "I feel reasonably connected to others"
    elif belonging > -1:
        belonging_phrase = "I often feel disconnected from those around me"
    else:
        belonging_phrase = "I am completely alone and alienated from society"

    # Context wrapper
    if context == "corporate":
        return f"In our organization, {agency_phrase.lower()}. {fairness_phrase}. {belonging_phrase}."
    elif context == "political":
        return f"As a citizen, {agency_phrase.lower()}. {fairness_phrase}. {belonging_phrase}."
    else:
        return f"{agency_phrase}. {fairness_phrase}. {belonging_phrase}."


# --- Main ---

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
