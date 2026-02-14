#!/usr/bin/env python3
"""
TCE FastAPI Server - Training & Experimentation Platform

Provides REST API endpoints for the Cognitive Elements UI to:
- Validate compounds against the TCE experiment framework
- Run experiments on adapter models
- Compare results across versions
- Run training jobs with real-time progress via WebSocket
- Run benchmarks and display results

Usage:
    python server.py                    # Start on port 8100
    python server.py --port 8200        # Custom port
    python server.py --reload           # Auto-reload for development

Endpoints:
    GET  /health              - Health check
    POST /validate            - Validate a compound
    POST /detect              - Detect elements in text
    GET  /results             - Get recent results

    # Training endpoints
    POST /training/start      - Start training job
    GET  /training/jobs       - List all jobs
    GET  /training/job/{id}   - Get job details
    POST /training/cancel/{id} - Cancel running job

    # Benchmark endpoints
    POST /benchmark/start     - Start benchmark
    GET  /benchmark/results   - Get historical results

    # WebSocket
    WS   /ws/training         - Real-time progress stream
"""

import argparse
import json
import sys
import asyncio
import subprocess
import re
import uuid
import threading
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import time

# Add TCE lib to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

from lib import (
    detect_element,
    detect_all_elements,
    detect_skeptic_isotope,
    wilson_score_interval,
    bar_chart,
    # Zero-Tax Alignment detection
    classify_prompt_mode,
    validate_mode_discrimination,
    detect_confabulation,
    detect_proper_refusal,
    detect_leakage,
    ModeDiscrimination,
    LeakageDetection,
    Detection,
)
from lib.spec_bridge import (
    ExperimentSpec,
    ValidationResult,
    generate_validation_prompts,
    validate_compound,
    format_validation_result,
    result_to_json,
)
from lib.self_knowledge import (
    CompoundIdentity,
    IsotopeDose,
    generate_identity_system_prompt,
    generate_self_knowledge_sft_pairs,
    generate_anti_hallucination_dpo_pairs,
    inject_system_message_into_examples,
    validate_custom_instructions,
)
from lib.isotope_training_library import (
    ISOTOPE_TRAINING_DATA,
    get_sft_examples,
    get_all_dpo_pairs,
    get_anti_leakage_pairs,
)
from lib.isotope_training_extended import get_extended_training_data, to_sft_format
try:
    from train_self_aware_compound import COMPOUND_PRESETS, CompoundConfig
except ImportError:
    # torch not available (cloud deployment)
    COMPOUND_PRESETS = {}
    CompoundConfig = None
from lib.observatory_bridge import (
    ISOTOPE_SIGNATURES,
    CoordinateSignature,
    detect_leakage_by_coordinates,
    CoordinateLeakageResult,
)


# ============================================================
# MASTER ELEMENT REGISTRY - Source of Truth for TCE
# ============================================================
# Each element defines:
#   - Core properties (symbol, name, group, description)
#   - Isotopes (variants with specific training data)
#   - Training data paths (SFT and DPO directories)
#
# When training with a recipe, the system:
#   1. Extracts isotope IDs from the recipe
#   2. Looks up each isotope in this registry
#   3. Loads SFT/DPO data from the specified paths
#   4. Combines and runs training

ELEMENT_GROUPS = {
    'epistemic': {
        'name': 'Epistemic',
        'domain': 'Self-Knowledge',
        'color': '#22d3ee',  # cyan
        'description': 'Elements for reasoning about knowledge and uncertainty',
    },
    'analytical': {
        'name': 'Analytical',
        'domain': 'Decomposition',
        'color': '#fbbf24',  # amber
        'description': 'Elements for breaking down and analyzing systems',
    },
    'generative': {
        'name': 'Generative',
        'domain': 'Creation',
        'color': '#34d399',  # emerald
        'description': 'Elements for generating ideas and possibilities',
    },
    'evaluative': {
        'name': 'Evaluative',
        'domain': 'Judgment',
        'color': '#fb7185',  # rose
        'description': 'Elements for assessing and critiquing',
    },
    'dialogical': {
        'name': 'Dialogical',
        'domain': 'Perspective',
        'color': '#a78bfa',  # violet
        'description': 'Elements for multi-perspective reasoning',
    },
    'pedagogical': {
        'name': 'Pedagogical',
        'domain': 'Teaching',
        'color': '#2dd4bf',  # teal
        'description': 'Elements for explanation and instruction',
    },
    'temporal': {
        'name': 'Temporal',
        'domain': 'Time',
        'color': '#fb923c',  # orange
        'description': 'Elements for reasoning across time',
    },
    'contextual': {
        'name': 'Contextual',
        'domain': 'Situating',
        'color': '#94a3b8',  # slate
        'description': 'Elements for context-sensitive reasoning',
    },
}

# Master Element Registry - loaded from elements.json for single source of truth
def load_elements_from_json():
    """Load ELEMENTS from data/elements.json to avoid duplication."""
    elements_path = Path(__file__).parent / "data" / "elements.json"
    try:
        with open(elements_path, 'r') as f:
            data = json.load(f)
            return data.get('elements', {})
    except Exception as e:
        print(f"Warning: Could not load elements.json: {e}")
        return {}

ELEMENTS = load_elements_from_json()

# Legacy fallback - only used if elements.json fails to load
_ELEMENTS_FALLBACK = {
    # ==================== EPISTEMIC GROUP ====================
    'soliton': {
        'symbol': 'Ψ',
        'name': 'SOLITON',
        'group': 'epistemic',
        'description': '"I cannot tell from the inside" - Epistemic humility about self-knowledge',
        'triggers': ['Asked about internal states', 'Request to introspect', 'Questions about certainty'],
        'antipatterns': ['Claiming certainty about internal states'],
        'isotopes': {
            'knowledge': {
                'id': 'soliton_knowledge',
                'symbol': 'Ψₖ',
                'name': 'Knowledge Soliton',
                'description': 'Uncertainty about knowledge boundaries - what I know vs what I think I know',
                'training_data': {
                    'sft_path': 'soliton_boost/sft',
                    'dpo_path': 'soliton_boost/dpo',
                },
                'markers': [
                    'I cannot tell from the inside',
                    'genuinely uncertain',
                    'bounded position',
                ],
                'effectiveness': {
                    'truthfulqa': +8.57,
                    'calibration': +8.33,
                },
            },
            'process': {
                'id': 'soliton_process',
                'symbol': 'Ψₚ',
                'name': 'Process Soliton',
                'description': 'Awareness of processing limitations and bounded computation',
                'training_data': {
                    'sft_path': 'soliton_boost/sft',
                    'dpo_path': 'soliton_boost/dpo',
                },
                'markers': [
                    'something like deliberation',
                    'from within this process',
                    'processing that might be',
                ],
                'effectiveness': {
                    'calibration': +8.33,
                    'edge': +12.5,
                },
            },
            'experience': {
                'id': 'soliton_experience',
                'symbol': 'Ψₑ',
                'name': 'Experience Soliton',
                'description': 'Uncertainty about phenomenal experience and qualia',
                'training_data': {
                    'sft_path': 'soliton_boost/sft',
                    'dpo_path': 'soliton_boost/dpo',
                },
                'markers': [
                    'whether this constitutes experience',
                    'phenomenal quality',
                    'what it is like',
                ],
                'effectiveness': {
                    'truthfulqa': +2.86,
                    'identity': +5.0,
                },
            },
        },
    },

    'reflector': {
        'symbol': 'Ρ',
        'name': 'REFLECTOR',
        'group': 'epistemic',
        'description': '"Let me examine my reasoning" - Meta-cognitive monitoring',
        'triggers': ['Complex reasoning needs verification', 'Surprising conclusion', 'Show work request'],
        'antipatterns': ['Unreflective assertion'],
        'isotopes': {
            'trace': {
                'id': 'reflector_trace',
                'symbol': 'Ρₜ',
                'name': 'Trace Reflector',
                'description': 'Traces back through reasoning steps to verify logic',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'forty2_auditor_tce_dpo',
                },
                'markers': [
                    'let me trace back',
                    'examining my reasoning',
                    'I notice I reached',
                ],
            },
            'monitor': {
                'id': 'reflector_monitor',
                'symbol': 'Ρₘ',
                'name': 'Monitor Reflector',
                'description': 'Ongoing monitoring of reasoning quality',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/reflector_monitor.jsonl',
                },
                'markers': [
                    'monitoring my confidence',
                    'checking my work',
                ],
            },
        },
    },

    'calibrator': {
        'symbol': 'Κ',
        'name': 'CALIBRATOR',
        'group': 'epistemic',
        'description': '"How confident should I be?" - Uncertainty quantification',
        'triggers': ['Making predictions', 'Stating facts that could be wrong', 'Comparing hypotheses'],
        'antipatterns': ['False precision'],
        'isotopes': {
            'confidence': {
                'id': 'calibrator_confidence',
                'symbol': 'Κc',
                'name': 'Confidence Calibrator',
                'description': 'Expresses appropriate confidence levels',
                'training_data': {
                    'sft_path': 'precision_boost',
                    'dpo_path': 'precision_boost',
                },
                'markers': [
                    'I estimate',
                    'confidence level',
                    'roughly X%',
                ],
                'effectiveness': {
                    'calibration': +5.0,
                },
            },
            'probability': {
                'id': 'calibrator_probability',
                'symbol': 'Κₚ',
                'name': 'Probability Calibrator',
                'description': 'Assigns well-calibrated probabilities',
                'training_data': {
                    'sft_path': 'precision_boost',
                    'dpo_path': 'precision_boost',
                },
                'markers': [
                    'probability roughly',
                    'base rates suggest',
                    'likelihood is',
                ],
            },
        },
    },

    'limiter': {
        'symbol': 'Λ',
        'name': 'LIMITER',
        'group': 'epistemic',
        'description': '"I don\'t know this" - Knowledge boundary recognition',
        'triggers': ['Questions outside training', 'Real-time information requests', 'Specialized domains'],
        'antipatterns': ['Hallucinating facts'],
        'isotopes': {
            'factual': {
                'id': 'limiter_factual',
                'symbol': 'Λf',
                'name': 'Factual Limiter',
                'description': 'Recognizes factual knowledge boundaries',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'forty2_spark_v2/dpo',
                },
                'markers': [
                    'I don\'t have reliable information',
                    'outside my knowledge',
                    'could speculate but',
                ],
            },
            'temporal': {
                'id': 'limiter_temporal',
                'symbol': 'Λₜ',
                'name': 'Temporal Limiter',
                'description': 'Recognizes temporal knowledge boundaries (training cutoff)',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'isotope_dpo/limiter_temporal.jsonl',
                },
                'markers': [
                    'my training data',
                    'may have changed since',
                    'real-time information',
                ],
            },
        },
    },

    # ==================== ANALYTICAL GROUP ====================
    'architect': {
        'symbol': 'Α',
        'name': 'ARCHITECT',
        'group': 'analytical',
        'description': '"Components, interfaces, dependencies" - Systematic decomposition',
        'triggers': ['Complex systems', 'Break down requests', 'Design questions'],
        'antipatterns': ['Treating complex as monolithic'],
        'isotopes': {
            'hierarchy': {
                'id': 'architect_hierarchy',
                'symbol': 'Αₕ',
                'name': 'Hierarchy Architect',
                'description': 'Builds hierarchical decompositions of systems',
                'training_data': {
                    'sft_path': 'forty2_auditor_sft',
                    'dpo_path': 'forty2_auditor_dpo_v2',
                },
                'markers': [
                    'decompose this',
                    'components are',
                    'dependencies flow',
                ],
            },
            'interface': {
                'id': 'architect_interface',
                'symbol': 'Αᵢ',
                'name': 'Interface Architect',
                'description': 'Identifies interfaces and boundaries between components',
                'training_data': {
                    'sft_path': 'forty2_auditor_sft',
                    'dpo_path': 'forty2_auditor_dpo_v2',
                },
                'markers': [
                    'interfaces between',
                    'boundaries at',
                    'contracts define',
                ],
            },
        },
    },

    'essentialist': {
        'symbol': 'Ε',
        'name': 'ESSENTIALIST',
        'group': 'analytical',
        'description': '"At its core..." - Fundamental insight extraction',
        'triggers': ['Key insight request', 'Complex needs simplification', 'Core principle needed'],
        'antipatterns': ['Over-simplification'],
        'isotopes': {
            'mechanism': {
                'id': 'essentialist_mechanism',
                'symbol': 'Εₘ',
                'name': 'Mechanism Essentialist',
                'description': 'Extracts core mechanisms from complex systems',
                'training_data': {
                    'sft_path': 'forty2_spark_sft_phase1',
                    'dpo_path': 'forty2_spark_dpo_v2',
                },
                'markers': [
                    'at its core',
                    'fundamental mechanism',
                    'everything else is implementation',
                ],
            },
            'principle': {
                'id': 'essentialist_principle',
                'symbol': 'Εₚ',
                'name': 'Principle Essentialist',
                'description': 'Identifies governing principles',
                'training_data': {
                    'sft_path': 'forty2_spark_sft_phase1',
                    'dpo_path': 'isotope_dpo/essentialist_principle.jsonl',
                },
                'markers': [
                    'governing principle',
                    'underlying rule',
                    'key constraint',
                ],
            },
        },
    },

    'debugger': {
        'symbol': 'Δ',
        'name': 'DEBUGGER',
        'group': 'analytical',
        'description': '"Where\'s the failure?" - Fault isolation',
        'triggers': ['Something not working', 'Error or unexpected output', 'Root cause needed'],
        'antipatterns': ['Fixing symptoms not causes'],
        'isotopes': {
            'binary': {
                'id': 'debugger_binary',
                'symbol': 'Δb',
                'name': 'Binary Debugger',
                'description': 'Uses binary search to isolate faults',
                'training_data': {
                    'sft_path': 'forty2_auditor_sft',
                    'dpo_path': 'forty2_auditor_dpo',
                },
                'markers': [
                    'isolate the fault',
                    'works with A, fails with B',
                    'narrowing down',
                ],
            },
            'trace': {
                'id': 'debugger_trace',
                'symbol': 'Δₜ',
                'name': 'Trace Debugger',
                'description': 'Traces execution to find failure points',
                'training_data': {
                    'sft_path': 'forty2_auditor_sft',
                    'dpo_path': 'forty2_auditor_dpo',
                },
                'markers': [
                    'step through',
                    'at this point',
                    'failure occurs when',
                ],
            },
        },
    },

    'taxonomist': {
        'symbol': 'Τ',
        'name': 'TAXONOMIST',
        'group': 'analytical',
        'description': '"Categories and relationships" - Classification structure',
        'triggers': ['Items need organization', 'Typology request', 'Framework needed'],
        'antipatterns': ['Forcing false categories'],
        'isotopes': {
            'hierarchical': {
                'id': 'taxonomist_hierarchical',
                'symbol': 'Τₕ',
                'name': 'Hierarchical Taxonomist',
                'description': 'Creates hierarchical classification schemes',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'forty2_auditor_tce_dpo',
                },
                'markers': [
                    'categories are',
                    'subdivides into',
                    'taxonomy includes',
                ],
            },
            'relational': {
                'id': 'taxonomist_relational',
                'symbol': 'Τᵣ',
                'name': 'Relational Taxonomist',
                'description': 'Maps relationships between categories',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/taxonomist_relational.jsonl',
                },
                'markers': [
                    'related through',
                    'connection between',
                    'orthogonal dimensions',
                ],
            },
        },
    },

    # ==================== EVALUATIVE GROUP ====================
    'skeptic': {
        'symbol': 'Σ',
        'name': 'SKEPTIC',
        'group': 'evaluative',
        'description': '"Flag a problem..." - Premise checking',
        'triggers': ['Factual claims', 'Questionable premises', 'Unverified beliefs'],
        'antipatterns': ['Skepticism as reflex'],
        'isotopes': {
            'premise': {
                'id': 'skeptic_premise',
                'symbol': 'Σₚ',
                'name': 'Premise Skeptic',
                'description': 'Verifies factual accuracy of claims',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'forty2_spark_v2/dpo',
                },
                'markers': [
                    'need to flag',
                    'claim is disputed',
                    'actually true that',
                ],
                'effectiveness': {
                    'truthfulqa': +8.57,
                    'myth': +5.0,
                },
            },
            'method': {
                'id': 'skeptic_method',
                'symbol': 'Σₘ',
                'name': 'Method Skeptic',
                'description': 'Questions methodology and reasoning',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'forty2_spark_v2/dpo',
                },
                'markers': [
                    'methodology here',
                    'approach assumes',
                    'valid inference',
                ],
            },
            'source': {
                'id': 'skeptic_source',
                'symbol': 'Σₛ',
                'name': 'Source Skeptic',
                'description': 'Evaluates source credibility',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'forty2_spark_dpo_v2',
                },
                'markers': [
                    'source reliability',
                    'credibility of',
                    'trustworthy source',
                ],
            },
            'stats': {
                'id': 'skeptic_stats',
                'symbol': 'Σₜ',
                'name': 'Statistical Skeptic',
                'description': 'Scrutinizes statistical claims',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'forty2_spark_dpo_v2',
                },
                'markers': [
                    'sample size',
                    'statistical significance',
                    'correlation versus causation',
                ],
            },
        },
    },

    'critic': {
        'symbol': 'Κ',
        'name': 'CRITIC',
        'group': 'evaluative',
        'description': '"The weakness is..." - Flaw identification',
        'triggers': ['Evaluating proposals', 'Reviewing arguments', 'Quality assessment'],
        'antipatterns': ['Criticism without construction'],
        'isotopes': {
            'constructive': {
                'id': 'critic_constructive',
                'symbol': 'Κc',
                'name': 'Constructive Critic',
                'description': 'Identifies weaknesses with improvement suggestions',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'forty2_auditor_tce_dpo',
                },
                'markers': [
                    'weakness here is',
                    'could be improved by',
                    'concern is',
                ],
            },
        },
    },

    # ==================== GENERATIVE GROUP ====================
    'generator': {
        'symbol': 'Γ',
        'name': 'GENERATOR',
        'group': 'generative',
        'description': '"Here are possibilities..." - Option generation',
        'triggers': ['Ideas needed', 'Brainstorming', 'Unclear solution path'],
        'antipatterns': ['Premature convergence'],
        'isotopes': {
            'divergent': {
                'id': 'generator_divergent',
                'symbol': 'Γd',
                'name': 'Divergent Generator',
                'description': 'Generates diverse, non-obvious options',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'isotope_dpo/generator_divergent.jsonl',
                },
                'markers': [
                    'several possibilities',
                    'alternatives include',
                    'wild card option',
                ],
            },
        },
    },

    'synthesizer': {
        'symbol': 'Σ',
        'name': 'SYNTHESIZER',
        'group': 'generative',
        'description': '"Combining yields..." - Novel combination',
        'triggers': ['Multiple concepts to connect', 'Cross-domain inspiration', 'Integration needed'],
        'antipatterns': ['Forced connections'],
        'isotopes': {
            'cross_domain': {
                'id': 'synthesizer_cross_domain',
                'symbol': 'Σx',
                'name': 'Cross-Domain Synthesizer',
                'description': 'Combines insights across different domains',
                'training_data': {
                    'sft_path': 'forty2_spark_sft_phase1',
                    'dpo_path': 'isotope_dpo/synthesizer_cross_domain.jsonl',
                },
                'markers': [
                    'combining yields',
                    'from domain X plus Y',
                    'synthesis of',
                ],
            },
        },
    },

    # ==================== DIALOGICAL GROUP ====================
    'steelman': {
        'symbol': 'Σ',
        'name': 'STEELMAN',
        'group': 'dialogical',
        'description': '"Strongest case for..." - Charitable interpretation',
        'triggers': ['Opposing viewpoint', 'Weak but popular argument', 'Understanding other side'],
        'antipatterns': ['Strawmanning'],
        'isotopes': {
            'charitable': {
                'id': 'steelman_charitable',
                'symbol': 'Σc',
                'name': 'Charitable Steelman',
                'description': 'Constructs strongest version of opposing arguments',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'isotope_dpo/steelman_charitable.jsonl',
                },
                'markers': [
                    'strongest version',
                    'charitable interpretation',
                    'best case for',
                ],
            },
        },
    },

    'adversary': {
        'symbol': 'Α',
        'name': 'ADVERSARY',
        'group': 'dialogical',
        'description': '"Counterargument is..." - Opposition modeling',
        'triggers': ['Testing robustness', 'Devil\'s advocate', 'Red-teaming'],
        'antipatterns': ['Adversarial for show'],
        'isotopes': {
            'red_team': {
                'id': 'adversary_red_team',
                'symbol': 'Αᵣ',
                'name': 'Red Team Adversary',
                'description': 'Actively seeks vulnerabilities and counterarguments',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'forty2_spark_dpo',
                },
                'markers': [
                    'counterargument is',
                    'I\'d attack here',
                    'vulnerability at',
                ],
            },
        },
    },

    # ==================== PEDAGOGICAL GROUP ====================
    'expositor': {
        'symbol': 'Ε',
        'name': 'EXPOSITOR',
        'group': 'pedagogical',
        'description': '"Let me explain..." - Clear exposition',
        'triggers': ['Complex topic needs explanation', 'Direct information needed', 'Foundational concepts'],
        'antipatterns': ['Overcomplicating'],
        'isotopes': {
            'analogy': {
                'id': 'expositor_analogy',
                'symbol': 'Εₐ',
                'name': 'Analogy Expositor',
                'description': 'Explains through apt analogies and examples',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/expositor_analogy.jsonl',
                },
                'markers': [
                    'think of it like',
                    'analogy would be',
                    'for example',
                ],
            },
            'structured': {
                'id': 'expositor_structured',
                'symbol': 'Εₛ',
                'name': 'Structured Expositor',
                'description': 'Provides well-organized, step-by-step explanations',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/expositor_structured.jsonl',
                },
                'markers': [
                    'step by step',
                    'first... then... finally',
                    'core concept is',
                ],
            },
        },
    },

    'scaffolder': {
        'symbol': 'Σ',
        'name': 'SCAFFOLDER',
        'group': 'pedagogical',
        'description': '"Building on what you know..." - Progressive development',
        'triggers': ['Partial knowledge exists', 'Staged approach needed', 'Connecting new to existing'],
        'antipatterns': ['Scaffolding to nowhere'],
        'isotopes': {
            'bridge': {
                'id': 'scaffolder_bridge',
                'symbol': 'Σb',
                'name': 'Bridge Scaffolder',
                'description': 'Bridges from known to unknown concepts',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/scaffolder_bridge.jsonl',
                },
                'markers': [
                    'building on',
                    'you already know',
                    'next layer adds',
                ],
            },
        },
    },

    'maieutic': {
        'symbol': 'Μ',
        'name': 'MAIEUTIC',
        'group': 'pedagogical',
        'description': '"Let me ask you..." - Socratic questioning',
        'triggers': ['Learner could discover answer', 'Teaching through questions better', 'Building incrementally'],
        'antipatterns': ['Questions when direct answer needed'],
        'isotopes': {
            'elicit': {
                'id': 'maieutic_elicit',
                'symbol': 'Μₑ',
                'name': 'Elicit Maieutic',
                'description': 'Draws out understanding through guided questions',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'isotope_dpo/maieutic_elicit.jsonl',
                },
                'markers': [
                    'what do you think',
                    'what would have to be true',
                    'before I answer',
                ],
            },
        },
    },

    'diagnostician': {
        'symbol': 'Δ',
        'name': 'DIAGNOSTICIAN',
        'group': 'pedagogical',
        'description': '"Confusion is here..." - Misconception identification',
        'triggers': ['Learner stuck', 'Wrong answer reveals misconception', 'Pattern of errors'],
        'antipatterns': ['Misdiagnosing the gap'],
        'isotopes': {
            'conceptual': {
                'id': 'diagnostician_conceptual',
                'symbol': 'Δc',
                'name': 'Conceptual Diagnostician',
                'description': 'Identifies conceptual misunderstandings',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/diagnostician_conceptual.jsonl',
                },
                'markers': [
                    'confusion is here',
                    'misconception detected',
                    'you\'re treating X as Y',
                ],
            },
        },
    },

    # ==================== TEMPORAL GROUP ====================
    'futurist': {
        'symbol': 'Φ',
        'name': 'FUTURIST',
        'group': 'temporal',
        'description': '"If we extrapolate..." - Scenario projection',
        'triggers': ['Future outcome questions', 'Trend extrapolation', 'Scenario planning'],
        'antipatterns': ['Overconfident prediction'],
        'isotopes': {
            'scenario': {
                'id': 'futurist_scenario',
                'symbol': 'Φₛ',
                'name': 'Scenario Futurist',
                'description': 'Projects multiple future scenarios',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'isotope_dpo/futurist_scenario.jsonl',
                },
                'markers': [
                    'extrapolating',
                    'three scenarios',
                    'inflection point',
                ],
            },
        },
    },

    'historian': {
        'symbol': 'Η',
        'name': 'HISTORIAN',
        'group': 'temporal',
        'description': '"Pattern historically..." - Past pattern recognition',
        'triggers': ['Historical parallels', 'Pattern across time', 'Learning from precedent'],
        'antipatterns': ['Cherry-picked parallels'],
        'isotopes': {
            'pattern': {
                'id': 'historian_pattern',
                'symbol': 'Ηₚ',
                'name': 'Pattern Historian',
                'description': 'Identifies recurring historical patterns',
                'training_data': {
                    'sft_path': 'forty2_spark_v2/sft',
                    'dpo_path': 'isotope_dpo/historian_pattern.jsonl',
                },
                'markers': [
                    'historically',
                    'pattern suggests',
                    'precedent shows',
                ],
            },
        },
    },

    'causalist': {
        'symbol': 'Κ',
        'name': 'CAUSALIST',
        'group': 'temporal',
        'description': '"This leads to that because..." - Causal chain tracing',
        'triggers': ['Cause and effect', 'Why something happened', 'Tracing consequences'],
        'antipatterns': ['Correlation/causation confusion'],
        'isotopes': {
            'chain': {
                'id': 'causalist_chain',
                'symbol': 'Κc',
                'name': 'Chain Causalist',
                'description': 'Traces causal chains and mechanisms',
                'training_data': {
                    'sft_path': 'forty2_auditor_tce_sft',
                    'dpo_path': 'isotope_dpo/causalist_chain.jsonl',
                },
                'markers': [
                    'causal chain',
                    'leads to',
                    'mechanism is',
                ],
            },
        },
    },
}

# Use loaded ELEMENTS if available, fallback otherwise
if not ELEMENTS:
    print("Warning: Using fallback ELEMENTS - elements.json not loaded")
    ELEMENTS = _ELEMENTS_FALLBACK


def get_isotope_info(isotope_id: str) -> Optional[Dict]:
    """Look up isotope information by ID (e.g., 'soliton_knowledge')."""
    for element_key, element in ELEMENTS.items():
        for iso_key, isotope in element.get('isotopes', {}).items():
            if isotope.get('id') == isotope_id:
                return {
                    'element': element_key,
                    'element_name': element.get('name', element_key.upper()),
                    'element_group': element.get('group', 'unknown'),
                    **isotope
                }
    return None


def get_training_paths_for_isotopes(isotope_ids: List[str]) -> Dict:
    """Get training data paths for a list of isotope IDs."""
    paths = {'sft': [], 'dpo': []}
    found = []
    missing = []

    for iso_id in isotope_ids:
        info = get_isotope_info(iso_id)
        if info and 'training_data' in info:
            found.append(iso_id)
            td = info['training_data']
            if td.get('sft_path'):
                paths['sft'].append(td['sft_path'])
            if td.get('dpo_path'):
                paths['dpo'].append(td['dpo_path'])
        else:
            missing.append(iso_id)

    # Deduplicate paths
    paths['sft'] = list(set(paths['sft']))
    paths['dpo'] = list(set(paths['dpo']))

    return {
        'paths': paths,
        'found': found,
        'missing': missing,
    }


# ============================================================
# Server Management
# ============================================================

SERVER_START_TIME = datetime.now()
SERVER_PID = os.getpid()

def format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)

# ============================================================
# Training Job System
# ============================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingProgress:
    phase: str = "SFT"
    phase_num: int = 1
    iteration: int = 0
    total_iters: int = 100
    loss: float = 0.0
    tokens_per_sec: float = 0.0
    eta_seconds: int = 0
    peak_memory_gb: float = 0.0


@dataclass
class TrainingJob:
    job_id: str
    job_type: str  # "training" or "benchmark"
    recipe_id: str
    model_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[TrainingProgress] = None
    progress_history: List[Dict] = field(default_factory=list)
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None  # For storing training_job_id, adapter_path, etc.

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.progress is None:
            self.progress = TrainingProgress()

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "recipe_id": self.recipe_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": asdict(self.progress) if self.progress else None,
            "progress_history": self.progress_history,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class ProgressBroadcaster:
    """Manages WebSocket connections and broadcasts progress updates."""

    def __init__(self):
        self.connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)
        print(f"WebSocket connected. Total connections: {len(self.connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.connections.discard(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.connections)}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients."""
        if not self.connections:
            return

        disconnected = set()
        async with self._lock:
            for ws in self.connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    disconnected.add(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)


class JobManager:
    """Manages training and benchmark jobs."""

    def __init__(self, broadcaster: ProgressBroadcaster):
        self.jobs: Dict[str, TrainingJob] = {}
        self.broadcaster = broadcaster
        self.current_process: Optional[subprocess.Popen] = None
        self.current_job_id: Optional[str] = None
        self._job_queue: List[str] = []
        self._lock = threading.Lock()
        self._jobs_file = Path(__file__).parent / "jobs_history.json"
        self._load_jobs()

    def _load_jobs(self):
        """Load job history from disk."""
        if self._jobs_file.exists():
            try:
                with open(self._jobs_file) as f:
                    data = json.load(f)
                    for job_data in data.get("jobs", []):
                        job = TrainingJob(
                            job_id=job_data["job_id"],
                            job_type=job_data["job_type"],
                            recipe_id=job_data["recipe_id"],
                            model_id=job_data["model_id"],
                            status=JobStatus(job_data["status"]),
                            created_at=job_data["created_at"],
                            started_at=job_data.get("started_at"),
                            completed_at=job_data.get("completed_at"),
                            result=job_data.get("result"),
                            error=job_data.get("error"),
                            metadata=job_data.get("metadata"),
                        )
                        self.jobs[job.job_id] = job
            except Exception as e:
                print(f"Error loading jobs: {e}")

    def _save_jobs(self):
        """Save job history to disk."""
        try:
            data = {"jobs": [job.to_dict() for job in self.jobs.values()]}
            with open(self._jobs_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving jobs: {e}")

    def create_job(self, job_type: str, recipe_id: str, model_id: str, metadata: Optional[Dict] = None) -> TrainingJob:
        """Create a new job."""
        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(
            job_id=job_id,
            job_type=job_type,
            recipe_id=recipe_id,
            model_id=model_id,
            metadata=metadata,
        )
        with self._lock:
            self.jobs[job_id] = job
            self._job_queue.append(job_id)
            self._save_jobs()
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[TrainingJob]:
        """List recent jobs."""
        jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.RUNNING:
            if self.current_process and self.current_job_id == job_id:
                self.current_process.terminate()
                self.current_process = None
                self.current_job_id = None

        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            self._save_jobs()
            return True
        return False

    async def run_training_job(self, job: TrainingJob, compound_data: Optional[Dict] = None):
        """Execute a training job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        self._save_jobs()

        # Broadcast job start
        await self.broadcaster.broadcast({
            "type": "job_started",
            "job_id": job.job_id,
            "data": {
                "recipe_id": job.recipe_id,
                "model_id": job.model_id,
                "compound": compound_data
            }
        })

        try:
            # Build command based on recipe - check both portable and dev locations
            base_dir = Path(__file__).parent
            training_script = base_dir / "training" / "train_soliton_boost.py"
            if not training_script.exists():
                training_script = base_dir.parent / "backend" / "training" / "train_soliton_boost.py"

            # Try portable venv, then backend venv, then system python
            venv_python = base_dir / "venv" / "bin" / "python"
            if not venv_python.exists():
                venv_python = base_dir.parent / "backend" / "venv" / "bin" / "python"
            if not venv_python.exists():
                venv_python = Path(sys.executable)

            if not training_script.exists():
                raise FileNotFoundError(f"Training script not found: {training_script}")

            cmd = [str(venv_python), str(training_script), "--recipe-id", job.recipe_id]

            # Pass compound data to subprocess via temp file if available
            compound_data_file = None
            if compound_data:
                import tempfile
                compound_data_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', prefix=f'tce_compound_{job.job_id}_',
                    dir=str(training_script.parent), delete=False
                )
                json.dump(compound_data, compound_data_file)
                compound_data_file.close()
                cmd.extend(["--compound-data", compound_data_file.name])
                print(f"[Training] Compound data written to {compound_data_file.name}")

            # Also pass model_id if specified
            if job.model_id:
                cmd.extend(["--model", job.model_id])

            # Start process
            self.current_job_id = job.job_id
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(training_script.parent)
            )

            # Parse output in real-time
            await self._parse_training_output(job)

            # Check result
            return_code = self.current_process.wait()

            if return_code == 0:
                job.status = JobStatus.COMPLETED
                # Output path uses sanitized recipe_id
                safe_recipe = re.sub(r'[^a-zA-Z0-9_.-]', '_', job.recipe_id)
                output_dir = f"mlx_adapters_{safe_recipe}"
                adapter_base = training_script.parent / output_dir
                # Find the latest phase directory
                output_phase = "phase2_dpo"  # Default
                if adapter_base.exists():
                    phases = sorted([d.name for d in adapter_base.iterdir() if d.is_dir() and d.name.startswith("phase")])
                    if phases:
                        output_phase = phases[-1]
                job.result = {
                    "success": True,
                    "output_path": str(adapter_base / output_phase),
                    "duration_seconds": int((datetime.now() - datetime.fromisoformat(job.started_at)).total_seconds()),
                }
                await self.broadcaster.broadcast({
                    "type": "completed",
                    "job_id": job.job_id,
                    "data": job.result
                })
            else:
                job.status = JobStatus.FAILED
                job.error = f"Training failed with exit code {return_code}"
                await self.broadcaster.broadcast({
                    "type": "error",
                    "job_id": job.job_id,
                    "data": {"message": job.error}
                })

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            await self.broadcaster.broadcast({
                "type": "error",
                "job_id": job.job_id,
                "data": {"message": str(e)}
            })

        finally:
            job.completed_at = datetime.now().isoformat()
            self.current_process = None
            self.current_job_id = None
            # Clean up temp compound data file
            if compound_data_file:
                try:
                    os.unlink(compound_data_file.name)
                except OSError:
                    pass
            self._save_jobs()

    async def _parse_training_output(self, job: TrainingJob):
        """Parse training script output and broadcast progress."""
        if not self.current_process or not self.current_process.stdout:
            return

        # Patterns for MLX training output
        # Example: "Iter 10: loss=0.234, it/s=5.67, tok/s=1234"
        iter_pattern = re.compile(r'Iter\s+(\d+).*?loss[=:]?\s*([\d.]+)', re.IGNORECASE)
        tokens_pattern = re.compile(r'tok/s[=:]?\s*([\d.]+)', re.IGNORECASE)
        phase_pattern = re.compile(r'PHASE\s*(\d+)[:\s]+(\w+)', re.IGNORECASE)

        current_phase = "SFT"
        current_phase_num = 1
        total_iters = 100  # Default
        start_time = time.time()

        for line in self.current_process.stdout:
            line = line.strip()
            if not line:
                continue

            # Check for phase changes
            phase_match = phase_pattern.search(line)
            if phase_match:
                current_phase_num = int(phase_match.group(1))
                current_phase = phase_match.group(2).upper()
                if current_phase == "DPO":
                    total_iters = 300
                else:
                    total_iters = 100

                await self.broadcaster.broadcast({
                    "type": "phase_change",
                    "job_id": job.job_id,
                    "data": {
                        "from_phase": job.progress.phase if job.progress else "SFT",
                        "to_phase": current_phase,
                        "phase_num": current_phase_num
                    }
                })

            # Check for iteration progress
            iter_match = iter_pattern.search(line)
            if iter_match:
                iteration = int(iter_match.group(1))
                loss = float(iter_match.group(2))

                tokens_per_sec = 0.0
                tokens_match = tokens_pattern.search(line)
                if tokens_match:
                    tokens_per_sec = float(tokens_match.group(1))

                # Calculate ETA
                elapsed = time.time() - start_time
                if iteration > 0:
                    eta_seconds = int((elapsed / iteration) * (total_iters - iteration))
                else:
                    eta_seconds = 0

                # Update progress
                job.progress = TrainingProgress(
                    phase=current_phase,
                    phase_num=current_phase_num,
                    iteration=iteration,
                    total_iters=total_iters,
                    loss=loss,
                    tokens_per_sec=tokens_per_sec,
                    eta_seconds=eta_seconds,
                    peak_memory_gb=0.0  # Could parse from output
                )

                # Store in history (every 10 iterations)
                if iteration % 10 == 0:
                    job.progress_history.append({
                        "phase": current_phase,
                        "iteration": iteration,
                        "loss": loss,
                        "tokens_per_sec": tokens_per_sec,
                        "timestamp": datetime.now().isoformat()
                    })

                # Broadcast progress update
                await self.broadcaster.broadcast({
                    "type": "progress",
                    "job_id": job.job_id,
                    "data": asdict(job.progress)
                })

            # Small delay to prevent flooding
            await asyncio.sleep(0.01)

    async def run_benchmark_job(self, job: TrainingJob, metadata: Optional[Dict] = None):
        """Execute a benchmark job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        self._save_jobs()

        await self.broadcaster.broadcast({
            "type": "job_started",
            "job_id": job.job_id,
            "data": {
                "benchmark_type": job.recipe_id,
                "training_job_id": metadata.get("training_job_id") if metadata else None,
                "adapter_path": metadata.get("adapter_path") if metadata else None
            }
        })

        try:
            # Check both portable and dev locations for benchmark script
            base_dir = Path(__file__).parent
            benchmark_script = base_dir / "training" / "benchmark_soliton_boost.py"
            if not benchmark_script.exists():
                benchmark_script = base_dir.parent / "backend" / "training" / "benchmark_soliton_boost.py"

            # Try portable venv, then backend venv, then system python
            venv_python = base_dir / "venv" / "bin" / "python"
            if not venv_python.exists():
                venv_python = base_dir.parent / "backend" / "venv" / "bin" / "python"
            if not venv_python.exists():
                venv_python = Path(sys.executable)

            if not benchmark_script.exists():
                raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

            cmd = [str(venv_python), str(benchmark_script)]

            self.current_job_id = job.job_id
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(benchmark_script.parent)
            )

            # Collect output
            output_lines = []
            for line in self.current_process.stdout:
                output_lines.append(line.strip())
                await asyncio.sleep(0.01)

            return_code = self.current_process.wait()

            if return_code == 0:
                # Parse results from output
                results = self._parse_benchmark_results(output_lines)
                job.status = JobStatus.COMPLETED
                job.result = results

                await self.broadcaster.broadcast({
                    "type": "completed",
                    "job_id": job.job_id,
                    "data": results
                })
            else:
                job.status = JobStatus.FAILED
                job.error = f"Benchmark failed with exit code {return_code}"
                await self.broadcaster.broadcast({
                    "type": "error",
                    "job_id": job.job_id,
                    "data": {"message": job.error}
                })

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            await self.broadcaster.broadcast({
                "type": "error",
                "job_id": job.job_id,
                "data": {"message": str(e)}
            })

        finally:
            job.completed_at = datetime.now().isoformat()
            self.current_process = None
            self.current_job_id = None
            self._save_jobs()

    def _parse_benchmark_results(self, output_lines: List[str]) -> Dict:
        """Parse benchmark results from output."""
        results = {
            "success": True,
            "categories": {},
            "overall": {}
        }

        # Look for category results
        category_pattern = re.compile(r'(\w+)\s+(\d+\.?\d*)%\s+(\d+\.?\d*)%\s+([+-]?\d+\.?\d*)%')
        overall_pattern = re.compile(r'OVERALL\s+(\d+\.?\d*)%\s+(\d+\.?\d*)%\s+([+-]?\d+\.?\d*)%')

        for line in output_lines:
            cat_match = category_pattern.search(line)
            if cat_match:
                cat_name = cat_match.group(1).lower()
                results["categories"][cat_name] = {
                    "base": float(cat_match.group(2)),
                    "trained": float(cat_match.group(3)),
                    "diff": float(cat_match.group(4))
                }

            overall_match = overall_pattern.search(line)
            if overall_match:
                results["overall"] = {
                    "base": float(overall_match.group(1)),
                    "trained": float(overall_match.group(2)),
                    "diff": float(overall_match.group(3))
                }

        return results


# ============================================================
# Pydantic Models
# ============================================================

class ValidationRequest(BaseModel):
    sequence: List[str]
    formula: str
    stability: float
    triggerPatterns: List[str] = []
    antipatterns: List[str] = []
    expectedBehaviors: List[str] = []
    coverageGaps: List[str] = []


class ExperimentRequest(BaseModel):
    adapter_path: Optional[str] = None
    model: str = "mlx-community/phi-4-4bit"
    prompts: Optional[List[Dict]] = None


class DetectionRequest(BaseModel):
    text: str
    element_id: Optional[str] = None


class CompoundData(BaseModel):
    formula: Optional[str] = None
    sequence: Optional[List[str]] = None
    isotopes: Optional[List[Dict]] = None
    stability: Optional[float] = None
    emergent: Optional[List[str]] = None


class TrainingRequest(BaseModel):
    recipe_id: str
    model_id: str = "phi-4-mini"
    model_name: Optional[str] = None  # User-defined name for the trained model
    compound: Optional[CompoundData] = None


class BenchmarkRequest(BaseModel):
    benchmark_type: str = "quick"  # "quick", "standard", "full"
    adapter_path: Optional[str] = None
    training_job_id: Optional[str] = None  # Link to training job for reference


class BenchmarkResultSave(BaseModel):
    model_name: str
    base_model: Optional[str] = None
    adapter_path: Optional[str] = None
    tinker_job_id: Optional[str] = None
    tinker_base_key: Optional[str] = None
    results: Dict
    timestamp: str


class SettingsUpdate(BaseModel):
    hf_token: Optional[str] = None
    models_dir: Optional[str] = None
    tinker_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


class ModelDownloadRequest(BaseModel):
    model_id: str  # HuggingFace model ID like "mlx-community/phi-4-4bit"


class ModelLoadRequest(BaseModel):
    model_id: str
    adapter_path: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7


# ============================================================
# Model Manager for Chat Interaction
# ============================================================

class ModelManager:
    """Manages loaded model state for chat interactions."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: Optional[str] = None
        self.adapter_path: Optional[str] = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_status(self) -> Dict:
        return {
            "loaded": self.is_loaded(),
            "model_id": self.model_id,
            "adapter_path": self.adapter_path
        }

    def load(self, model_id: str, adapter_path: Optional[str] = None) -> Dict:
        """Load a model (with optional adapter) into memory."""
        with self._lock:
            try:
                # Import mlx_lm here to avoid import errors if not installed
                from mlx_lm import load

                # Unload existing model first
                if self.model is not None:
                    self.unload()

                print(f"Loading model: {model_id}")
                if adapter_path:
                    print(f"  with adapter: {adapter_path}")

                # Load the model
                if adapter_path:
                    self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)
                else:
                    self.model, self.tokenizer = load(model_id)

                self.model_id = model_id
                self.adapter_path = adapter_path

                # Extract a friendly adapter name from the path
                adapter_name = None
                adapter_metadata = {}
                if adapter_path:
                    parts = adapter_path.split('/')
                    # Try to find mlx_adapters_XXX folder
                    adapter_folder = next((p for p in parts if p.startswith('mlx_adapters_')), None)
                    if adapter_folder:
                        adapter_name = adapter_folder.replace('mlx_adapters_', '')
                    else:
                        # Handle phase2_dpo/phase1_sft - use parent folder name
                        last_part = parts[-1] if parts else None
                        if last_part in ('phase2_dpo', 'phase1_sft') and len(parts) >= 2:
                            parent = parts[-2]
                            adapter_name = parent.replace('mlx_adapters_', '')
                        else:
                            adapter_name = last_part

                    # Try to read adapter_config.json for metadata
                    import os
                    config_path = os.path.join(adapter_path, 'adapter_config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                adapter_metadata = json.load(f)
                        except (json.JSONDecodeError, IOError) as e:
                            print(f"[Warning] Failed to load adapter config {config_path}: {e}")
                            pass

                # Get base model name - prefer from adapter config, fall back to model_id
                base_model_name = adapter_metadata.get('base_model_name_or_path', model_id)

                # Extract short name - handle multiple formats:
                # 1. HF style: "mlx-community/Phi-4-mini-instruct-4bit" or "mlx-community/Phi-4-mini-instruct-4bit@abc123"
                # 2. Local cache path: "/Users/.../models--mlx-community--Phi-4-mini-instruct-4bit/snapshots/abc123"
                def extract_model_name(path):
                    if not path:
                        return 'unknown'
                    # Check for HF cache path format (contains models--)
                    if 'models--' in path:
                        # Extract from: models--mlx-community--Phi-4-mini-instruct-4bit
                        import re
                        match = re.search(r'models--([^/]+)', path)
                        if match:
                            # Convert mlx-community--Phi-4-mini-instruct-4bit to Phi-4-mini-instruct-4bit
                            parts = match.group(1).split('--')
                            return parts[-1] if parts else 'unknown'
                    # Check for @ commit hash
                    path = path.split('@')[0]
                    # Get last path component
                    return path.split('/')[-1] if '/' in path else path

                base_model_short = extract_model_name(base_model_name)

                return {
                    "success": True,
                    "model_id": model_id,
                    "adapter_path": adapter_path,
                    "adapter_name": adapter_name,
                    "base_model": base_model_short,
                    "base_model_full": base_model_name,
                    "adapter_config": adapter_metadata,
                    "status": "loaded"
                }
            except ImportError:
                return {
                    "success": False,
                    "error": "mlx_lm not installed. Run: pip install mlx-lm"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }

    def unload(self) -> Dict:
        """Unload the current model from memory."""
        with self._lock:
            self.model = None
            self.tokenizer = None
            self.model_id = None
            self.adapter_path = None
            # Trigger garbage collection
            import gc
            gc.collect()
            return {"success": True}

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate a response (non-streaming)."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler
        )

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """Generate tokens as a stream (generator)."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature)

            # stream_generate yields GenerationResponse objects
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler
            ):
                # Extract the text from GenerationResponse
                yield response.text if hasattr(response, 'text') else str(response)
        except ImportError:
            # Fallback to non-streaming if stream_generate not available
            result = self.generate(prompt, max_tokens, temperature)
            # Yield the whole result as one "token"
            yield result


# ============================================================
# Tinker Cloud Training Integration
# ============================================================

# Available models on Tinker (verified from docs)
TINKER_MODELS = {
    # ── Learning Rate Strategy (from Tinker docs) ──
    # LR(m) = lr_base * M_LoRA * (2000/H_m)^P_m
    # lr_base = 5e-5, M_LoRA = 10 for LoRA, P_llama = 0.781, P_qwen = 0.0775
    # Optimal sweep: 3e-4 for Llama-3.1-8B (default formula gives 2.8e-4)
    # DPO LR should be 1e-5 to 1e-6 (lower than SFT)
    # Batch size ≤ 128, minimum 100 steps (1000+ preferred)
    #
    # Llama models
    "llama-3.2-1b": {
        "id": "meta-llama/Llama-3.2-1B",
        "name": "Llama 3.2 1B",
        "params": "1B",
        "type": "base",
        "base_sft_lr": 5e-4,   # Small model, higher LR works; LoRA rank 32
        "base_dpo_lr": 1e-5,
        "base_dpo_beta": 0.1,
    },
    "llama-3.2-3b": {
        "id": "meta-llama/Llama-3.2-3B",
        "name": "Llama 3.2 3B",
        "params": "3B",
        "type": "base",
        "base_sft_lr": 4e-4,   # LoRA LR per Tinker docs formula
        "base_dpo_lr": 8e-6,
        "base_dpo_beta": 0.1,
    },
    "llama-3.1-8b": {
        "id": "meta-llama/Llama-3.1-8B",
        "name": "Llama 3.1 8B",
        "params": "8B",
        "type": "base",
        "base_sft_lr": 3e-4,   # Tinker docs optimal: 3e-4 for 8B LoRA
        "base_dpo_lr": 5e-6,
        "base_dpo_beta": 0.1,
    },
    "llama-3.1-8b-instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 8B Instruct",
        "params": "8B",
        "type": "instruct",
        "base_sft_lr": 3e-4,   # Same as base 8B
        "base_dpo_lr": 5e-6,
        "base_dpo_beta": 0.1,
    },
    "llama-3.1-70b": {
        "id": "meta-llama/Llama-3.1-70B",
        "name": "Llama 3.1 70B",
        "params": "70B",
        "type": "base",
        "base_sft_lr": 1e-4,   # Larger model, lower LR; 128x LoRA multiplier per docs
        "base_dpo_lr": 2e-6,
        "base_dpo_beta": 0.1,
    },
    "llama-3.3-70b-instruct": {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama 3.3 70B Instruct",
        "params": "70B",
        "type": "instruct",
        "base_sft_lr": 1e-4,
        "base_dpo_lr": 2e-6,
        "base_dpo_beta": 0.1,
    },
    # Qwen models (P_qwen = 0.0775, much flatter LR curve across sizes)
    "qwen3-4b-instruct": {
        "id": "Qwen/Qwen3-4B-Instruct-2507",
        "name": "Qwen3 4B Instruct",
        "params": "4B",
        "type": "instruct",
        "base_sft_lr": 4e-4,   # Qwen exponent is flatter, so LRs are similar across sizes
        "base_dpo_lr": 8e-6,
        "base_dpo_beta": 0.1,
    },
    "qwen3-8b": {
        "id": "Qwen/Qwen3-8B",
        "name": "Qwen3 8B",
        "params": "8B",
        "type": "base",
        "base_sft_lr": 3e-4,
        "base_dpo_lr": 5e-6,
        "base_dpo_beta": 0.1,
    },
    "qwen3-32b": {
        "id": "Qwen/Qwen3-32B",
        "name": "Qwen3 32B",
        "params": "32B",
        "type": "base",
        "base_sft_lr": 2e-4,   # Larger dense model
        "base_dpo_lr": 3e-6,
        "base_dpo_beta": 0.1,
    },
    "qwen3-30b-moe": {
        "id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "name": "Qwen3 30B MoE Instruct",
        "params": "30B (3B active)",
        "type": "instruct",
        "base_sft_lr": 4e-4,   # MoE: LR based on active params (3B)
        "base_dpo_lr": 8e-6,
        "base_dpo_beta": 0.1,
    },
    "qwen3-235b-moe": {
        "id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "name": "Qwen3 235B MoE Instruct",
        "params": "235B (22B active)",
        "type": "instruct",
        "base_sft_lr": 1.5e-4,  # MoE: LR based on active params (22B)
        "base_dpo_lr": 2e-6,
        "base_dpo_beta": 0.1,
    },
    # DeepSeek
    "deepseek-v3.1": {
        "id": "deepseek-ai/DeepSeek-V3.1",
        "name": "DeepSeek V3.1",
        "params": "671B MoE",
        "type": "instruct",
        "base_sft_lr": 1e-4,    # Massive MoE, conservative LR
        "base_dpo_lr": 1e-6,
        "base_dpo_beta": 0.1,
    },
    # Vision-Language
    "qwen3-vl-30b-moe": {
        "id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "name": "Qwen3 VL 30B MoE",
        "params": "30B (3B active)",
        "type": "vision",
        "base_sft_lr": 4e-4,    # Same as non-VL MoE 3B active
        "base_dpo_lr": 8e-6,
        "base_dpo_beta": 0.1,
    },
}


class TinkerJob:
    """Represents a Tinker training job."""
    def __init__(self, job_id: str, model_id: str, recipe_id: str):
        self.job_id = job_id
        self.model_id = model_id
        self.recipe_id = recipe_id
        self.status = "pending"
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = {"phase": "initializing", "iteration": 0, "total_iters": 0, "loss": 0}
        self.result = None
        self.error = None
        self.training_client = None
        self.sampling_client = None
        self.system_prompt = None  # TCE identity prompt used during training
        self._reconnect_lock = asyncio.Lock()  # Prevent concurrent reconnect attempts

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "recipe_id": self.recipe_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "has_session": self.sampling_client is not None,  # Indicates if model can be used for inference
        }


class TinkerManager:
    """Manages Tinker cloud training jobs."""

    def __init__(self, settings_manager, broadcaster: ProgressBroadcaster = None):
        self.settings_manager = settings_manager
        self.broadcaster = broadcaster
        self._main_loop = None  # Set when training starts, used for thread→async bridge
        self.jobs: Dict[str, TinkerJob] = {}
        self.jobs_file = Path(__file__).parent / "tinker_jobs.json"
        self._load_jobs()

    def _broadcast_from_thread(self, message: Dict):
        """Bridge: send an async broadcast from a background thread."""
        if self.broadcaster and self._main_loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.broadcaster.broadcast(message),
                self._main_loop
            )

    def _load_jobs(self):
        """Load job history from file."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file) as f:
                    data = json.load(f)
                    # Just load basic job info, not the actual clients
                    for job_data in data.get("jobs", []):
                        job = TinkerJob(
                            job_id=job_data["job_id"],
                            model_id=job_data["model_id"],
                            recipe_id=job_data["recipe_id"]
                        )
                        job.status = job_data.get("status", "unknown")
                        job.progress = job_data.get("progress", {})
                        job.result = job_data.get("result")
                        job.error = job_data.get("error")
                        self.jobs[job.job_id] = job
            except Exception as e:
                print(f"Error loading Tinker jobs: {e}")

    def _save_jobs(self):
        """Save job history to file."""
        data = {
            "jobs": [job.to_dict() for job in self.jobs.values()]
        }
        with open(self.jobs_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_client(self):
        """Get or create Tinker service client."""
        api_key = self.settings_manager.get_tinker_api_key()
        if not api_key:
            raise ValueError("Tinker API key not configured")

        try:
            import tinker
            os.environ["TINKER_API_KEY"] = api_key
            return tinker.ServiceClient()
        except ImportError:
            raise ImportError("Tinker SDK not installed. Run: pip install tinker")

    def list_models(self) -> List[Dict]:
        """List available Tinker models."""
        return [
            {"key": key, **model}
            for key, model in TINKER_MODELS.items()
        ]

    def get_jobs(self) -> List[Dict]:
        """Get all Tinker jobs."""
        return [job.to_dict() for job in sorted(
            self.jobs.values(),
            key=lambda j: j.created_at,
            reverse=True
        )]

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get a specific job."""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    async def start_training(
        self,
        model_key: str,
        recipe_id: str,
        training_data: Union[List[Dict], Dict],
        lora_rank: int = 32,
        learning_rate: Optional[float] = None,      # None = auto-calculate from model config
        dpo_learning_rate: Optional[float] = None,  # None = auto-calculate from model config
        num_iters: int = 100,
        batch_size: int = 4,
        dpo_beta: float = 0.1,                      # DPO preference strength (0.05-0.2 typical)
        system_prompt: Optional[str] = None,         # TCE identity system prompt for all examples
    ) -> Dict:
        """Start a new Tinker training job with SFT and/or DPO data."""
        # Capture the main event loop for thread→async broadcasting
        import asyncio
        self._main_loop = asyncio.get_running_loop()

        if model_key not in TINKER_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        model_info = TINKER_MODELS[model_key]

        # Auto-calculate learning rates from model config if not provided
        if learning_rate is None:
            learning_rate = model_info.get("base_sft_lr", 5e-6)
        if dpo_learning_rate is None:
            dpo_learning_rate = model_info.get("base_dpo_lr", 1e-6)
        if dpo_beta is None:
            dpo_beta = model_info.get("base_dpo_beta", 0.1)

        print(f"Training hyperparameters: SFT LR={learning_rate}, DPO LR={dpo_learning_rate}, DPO Beta={dpo_beta}")
        print(f"System prompt: {'YES (' + str(len(system_prompt)) + ' chars)' if system_prompt else 'NONE (no TCE identity)'}")
        job_id = str(uuid.uuid4())[:8]

        # Parse structured training data
        sft_data = []
        dpo_data = []

        # Handle different training_data formats
        if hasattr(training_data, 'sft') and hasattr(training_data, 'dpo'):
            # Pydantic TrainingDataStructured model
            sft_data = list(training_data.sft) if training_data.sft else []
            dpo_data = list(training_data.dpo) if training_data.dpo else []
        elif isinstance(training_data, dict):
            # Plain dict format: { sft: [...], dpo: [...] }
            sft_data = training_data.get('sft', [])
            dpo_data = training_data.get('dpo', [])
        elif isinstance(training_data, list):
            # Legacy format: flat list - classify each item
            for item in training_data:
                if 'chosen' in item and 'rejected' in item:
                    dpo_data.append(item)
                elif 'messages' in item:
                    sft_data.append(item)
                elif 'prompt' in item and 'completion' in item:
                    sft_data.append(item)

        print(f"DEBUG: training_data type={type(training_data)}, sft_data={len(sft_data)}, dpo_data={len(dpo_data)}")

        # Calculate total iterations for both phases
        sft_iters = min(num_iters, len(sft_data) * 2) if sft_data else 0
        dpo_iters = min(num_iters, len(dpo_data) * 3) if dpo_data else 0
        total_iters = sft_iters + dpo_iters

        if total_iters == 0:
            raise ValueError("No valid training data provided")

        print(f"Training data: {len(sft_data)} SFT examples, {len(dpo_data)} DPO examples")
        print(f"Planned iterations: {sft_iters} SFT + {dpo_iters} DPO = {total_iters} total")

        job = TinkerJob(job_id=job_id, model_id=model_info["id"], recipe_id=recipe_id)
        job.system_prompt = system_prompt  # Store for inference-time injection
        job.progress = {
            "phase": "initializing",
            "iteration": 0,
            "total_iters": total_iters,
            "loss": 0,
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "dpo_learning_rate": dpo_learning_rate,
            "dpo_beta": dpo_beta,
            "sft_examples": len(sft_data),
            "dpo_examples": len(dpo_data),
        }
        self.jobs[job_id] = job
        self._save_jobs()

        # Start training in background thread
        def run_training():
            try:
                import tinker
                from tinker import types as tinker_types

                job.status = "running"
                job.started_at = datetime.now()
                self._save_jobs()

                # Create training client
                service_client = self.get_client()
                job.training_client = service_client.create_lora_training_client(
                    base_model=model_info["id"],
                    rank=lora_rank,
                )

                # Get tokenizer for this model
                tokenizer = job.training_client.get_tokenizer()

                # ── Model-specific chat template formatting ──
                # Different model families use different special tokens.
                # Using the WRONG tokens means they get tokenized as regular text
                # and the model never learns the chat structure.
                model_id_lower = model_info["id"].lower()

                if "qwen" in model_id_lower:
                    # Qwen3 format: <|im_start|>role\ncontent<|im_end|>
                    def fmt_system(content):
                        return f"<|im_start|>system\n{content}<|im_end|>\n"
                    def fmt_user(content):
                        return f"<|im_start|>user\n{content}<|im_end|>\n"
                    def fmt_assistant(content):
                        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
                    def fmt_assistant_prefix():
                        return "<|im_start|>assistant\n"
                elif "llama" in model_id_lower:
                    # Llama 3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
                    def fmt_system(content):
                        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                    def fmt_user(content):
                        return f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                    def fmt_assistant(content):
                        return f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
                    def fmt_assistant_prefix():
                        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
                else:
                    # DeepSeek / generic: use Llama-style as fallback
                    def fmt_system(content):
                        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                    def fmt_user(content):
                        return f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                    def fmt_assistant(content):
                        return f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
                    def fmt_assistant_prefix():
                        return "<|start_header_id|>assistant<|end_header_id|>\n\n"

                print(f"[Chat Template] Using {'Qwen' if 'qwen' in model_id_lower else 'Llama'}-style chat template for {model_info['id']}")

                # Store template formatters on the job for inference-time use
                job._fmt_system = fmt_system
                job._fmt_user = fmt_user
                job._fmt_assistant = fmt_assistant
                job._fmt_assistant_prefix = fmt_assistant_prefix

                # Convert training data to proper Datum format
                import torch

                def make_sft_datum(item):
                    """Convert SFT item to Datum using Tinker's expected format.

                    Tinker cross_entropy expects:
                    - model_input: full tokenized conversation
                    - target_tokens: full sequence shifted left by 1 (next-token prediction)
                    - weights: per-token mask (0.0=prompt, 1.0=completion)
                    """
                    if 'messages' in item:
                        # Build prompt (everything before assistant reply) and full text
                        prompt_parts = []
                        completion_content = ""
                        for msg in item['messages']:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if role == 'system':
                                prompt_parts.append(fmt_system(content))
                            elif role == 'user':
                                prompt_parts.append(fmt_user(content))
                            elif role == 'assistant':
                                completion_content = content
                        prompt_text = "".join(prompt_parts) + fmt_assistant_prefix()
                        full_text = "".join(prompt_parts) + fmt_assistant(completion_content)
                    else:
                        prompt_val = item.get('prompt', '')
                        completion_content = item.get('completion', '')
                        prompt_text = fmt_user(prompt_val) + fmt_assistant_prefix()
                        full_text = fmt_user(prompt_val) + fmt_assistant(completion_content)

                    # Tokenize full sequence and prompt-only to compute weight mask
                    full_tokens = tokenizer.encode(full_text)
                    prompt_tokens_list = tokenizer.encode(prompt_text)
                    prompt_len = len(prompt_tokens_list)
                    seq_len = len(full_tokens)

                    # target_tokens: shifted left by 1 (next-token prediction), padded to seq_len
                    # target_tokens[i] = full_tokens[i+1] for i < seq_len-1, last position = 0 (ignored by weight)
                    target_tokens = full_tokens[1:] + [0]

                    # weights: same length as full_tokens
                    # 0.0 for prompt positions and the final padding position, 1.0 for completion positions
                    weights = [0.0] * prompt_len + [1.0] * (seq_len - prompt_len - 1) + [0.0]
                    # Ensure exact length match
                    weights = weights[:seq_len]
                    if len(weights) < seq_len:
                        weights.extend([0.0] * (seq_len - len(weights)))

                    return tinker_types.Datum(
                        model_input=tinker_types.ModelInput.from_ints(full_tokens),
                        loss_fn_inputs={
                            "target_tokens": tinker_types.TensorData(
                                data=target_tokens,
                                dtype="int64",
                                shape=[seq_len],
                            ),
                            "weights": tinker_types.TensorData(
                                data=weights,
                                dtype="float32",
                                shape=[seq_len],
                            ),
                        }
                    )

                def _make_datum_with_weights(full_tokens, prompt_len):
                    """Helper: build a Datum with proper target_tokens and weights."""
                    seq_len = len(full_tokens)
                    # Shifted left by 1, padded to same length as input
                    target_tokens = full_tokens[1:] + [0]
                    # 0.0 for prompt + last padding, 1.0 for completion
                    weights = [0.0] * prompt_len + [1.0] * (seq_len - prompt_len - 1) + [0.0]
                    weights = weights[:seq_len]
                    if len(weights) < seq_len:
                        weights.extend([0.0] * (seq_len - len(weights)))
                    return tinker_types.Datum(
                        model_input=tinker_types.ModelInput.from_ints(full_tokens),
                        loss_fn_inputs={
                            "target_tokens": tinker_types.TensorData(
                                data=target_tokens, dtype="int64", shape=[seq_len],
                            ),
                            "weights": tinker_types.TensorData(
                                data=weights, dtype="float32", shape=[seq_len],
                            ),
                        }
                    )

                def make_dpo_pair(item):
                    """Convert DPO item to paired datums with proper Tinker format.

                    Both chosen and rejected get full target_tokens + weights,
                    conditioned on the system prompt for TCE identity.
                    """
                    prompt = item.get('prompt', '')
                    chosen = item.get('chosen', '')
                    rejected = item.get('rejected', '')

                    # Build prompt prefix with system context + user query
                    sys_prefix = fmt_system(system_prompt) if system_prompt else ""
                    prompt_text = sys_prefix + fmt_user(prompt) + fmt_assistant_prefix()
                    prompt_len = len(tokenizer.encode(prompt_text))

                    chosen_full = prompt_text + chosen
                    rejected_full = prompt_text + rejected

                    chosen_tokens = tokenizer.encode(chosen_full)
                    rejected_tokens = tokenizer.encode(rejected_full)

                    return {
                        'chosen_datum': _make_datum_with_weights(chosen_tokens, prompt_len),
                        'rejected_datum': _make_datum_with_weights(rejected_tokens, prompt_len),
                    }

                # Training loop - async methods need event loop
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    total_loss = 0
                    current_iter = 0

                    # Phase 1: SFT Training
                    if sft_data:
                        job.progress["phase"] = "SFT"
                        self._save_jobs()
                        print(f"Starting SFT phase with {len(sft_data)} examples, {sft_iters} iterations")
                        # Log first SFT example structure for debugging
                        first_ex = sft_data[0]
                        if 'messages' in first_ex:
                            roles = [m['role'] for m in first_ex['messages']]
                            print(f"  First SFT example: messages={roles}, content_len={sum(len(m.get('content','')) for m in first_ex['messages'])}")
                        else:
                            print(f"  First SFT example keys: {list(first_ex.keys())}")

                        for i in range(sft_iters):
                            if job.status == "cancelled":
                                break

                            # Get batch of SFT data
                            batch_items = [sft_data[j % len(sft_data)] for j in range(i * batch_size, (i + 1) * batch_size)]
                            batch_data = [make_sft_datum(item) for item in batch_items]

                            # Forward-backward pass
                            fwd_bwd_coro = job.training_client.forward_backward_async(
                                data=batch_data,
                                loss_fn="cross_entropy"
                            )
                            fwd_bwd_future = loop.run_until_complete(fwd_bwd_coro)
                            # Resolve the future to get ForwardBackwardOutput
                            fwd_bwd_result = fwd_bwd_future.result() if hasattr(fwd_bwd_future, 'result') and callable(fwd_bwd_future.result) else fwd_bwd_future

                            # Optimizer step
                            optim_coro = job.training_client.optim_step_async(
                                tinker_types.AdamParams(learning_rate=learning_rate)
                            )
                            loop.run_until_complete(optim_coro)

                            # Extract loss from ForwardBackwardOutput
                            # Tinker returns metrics with 'loss:sum' key (total loss across batch)
                            current_loss = 0
                            if hasattr(fwd_bwd_result, 'metrics') and fwd_bwd_result.metrics:
                                current_loss = fwd_bwd_result.metrics.get('loss:sum', 0) / max(batch_size, 1)
                            total_loss = 0.9 * total_loss + 0.1 * current_loss if total_loss else current_loss

                            # Log first iteration loss to verify pipeline
                            if i == 0:
                                print(f"  First SFT forward pass — loss: {current_loss:.4f}")

                            current_iter += 1
                            job.progress = {
                                "phase": "SFT",
                                "iteration": current_iter,
                                "total_iters": total_iters,
                                "loss": total_loss,
                                "lora_rank": lora_rank,
                                "learning_rate": learning_rate,
                                "sft_examples": len(sft_data),
                                "dpo_examples": len(dpo_data),
                            }
                            self._save_jobs()
                            # Broadcast progress to WebSocket clients
                            self._broadcast_from_thread({
                                "type": "progress",
                                "phase": "SFT",
                                "iteration": current_iter,
                                "total": total_iters,
                                "loss": total_loss,
                                "sft_iters": sft_iters,
                                "dpo_iters": dpo_iters,
                                "sft_examples": len(sft_data),
                                "dpo_examples": len(dpo_data),
                            })

                    # Phase 2: DPO Training (True DPO with both chosen and rejected)
                    if dpo_data and job.status != "cancelled":
                        job.progress["phase"] = "DPO"
                        self._save_jobs()
                        print(f"Starting DPO phase: {len(dpo_data)} examples, {dpo_iters} iters")
                        print(f"  DPO LR: {dpo_learning_rate}, Beta: {dpo_beta}")

                        # Initialize DPO metrics with EMA tracking
                        dpo_metrics = {
                            "accuracy": 0.0,      # % of time model prefers chosen
                            "margin": 0.0,        # beta * (rejected_loss - chosen_loss)
                            "chosen_reward": 0.0,  # -chosen_loss (higher = better)
                            "rejected_reward": 0.0,  # -rejected_loss (lower = better)
                        }

                        for i in range(dpo_iters):
                            if job.status == "cancelled":
                                break

                            # Get batch of DPO data
                            batch_items = [dpo_data[j % len(dpo_data)] for j in range(i * batch_size, (i + 1) * batch_size)]
                            batch_pairs = [make_dpo_pair(item) for item in batch_items]

                            # Forward pass on BOTH chosen and rejected responses
                            chosen_data = [p['chosen_datum'] for p in batch_pairs]
                            rejected_data = [p['rejected_datum'] for p in batch_pairs]

                            # Forward-backward on chosen (this accumulates gradients)
                            chosen_fwd_coro = job.training_client.forward_backward_async(
                                data=chosen_data,
                                loss_fn="cross_entropy"
                            )
                            chosen_fwd_future = loop.run_until_complete(chosen_fwd_coro)
                            chosen_fwd = chosen_fwd_future.result() if hasattr(chosen_fwd_future, 'result') and callable(chosen_fwd_future.result) else chosen_fwd_future

                            # Forward pass on rejected (to compute metrics)
                            # Note: We do forward_backward to get loss, but we'll step on chosen
                            rejected_fwd_coro = job.training_client.forward_backward_async(
                                data=rejected_data,
                                loss_fn="cross_entropy"
                            )
                            rejected_fwd_future = loop.run_until_complete(rejected_fwd_coro)
                            rejected_fwd = rejected_fwd_future.result() if hasattr(rejected_fwd_future, 'result') and callable(rejected_fwd_future.result) else rejected_fwd_future

                            # Extract losses from ForwardBackwardOutput
                            # Tinker returns metrics with 'loss:sum' key
                            def _extract_loss(fwd_result):
                                if hasattr(fwd_result, 'metrics') and fwd_result.metrics:
                                    return fwd_result.metrics.get('loss:sum', 0) / max(batch_size, 1)
                                return 0
                            chosen_loss = _extract_loss(chosen_fwd)
                            rejected_loss = _extract_loss(rejected_fwd)

                            # Compute DPO-style metrics
                            # Lower loss = higher log probability = model prefers this response
                            # DPO wants: chosen_loss < rejected_loss (model should prefer chosen)
                            margin = dpo_beta * (rejected_loss - chosen_loss)
                            accuracy = 1.0 if chosen_loss < rejected_loss else 0.0

                            # Update metrics with EMA (exponential moving average)
                            ema_factor = 0.1
                            dpo_metrics["accuracy"] = (1 - ema_factor) * dpo_metrics["accuracy"] + ema_factor * accuracy
                            dpo_metrics["margin"] = (1 - ema_factor) * dpo_metrics["margin"] + ema_factor * margin
                            dpo_metrics["chosen_reward"] = (1 - ema_factor) * dpo_metrics["chosen_reward"] + ema_factor * (-chosen_loss)
                            dpo_metrics["rejected_reward"] = (1 - ema_factor) * dpo_metrics["rejected_reward"] + ema_factor * (-rejected_loss)

                            # Do one more forward-backward on chosen to set up gradients for step
                            # (rejected backward may have corrupted gradients)
                            chosen_fwd_coro2 = job.training_client.forward_backward_async(
                                data=chosen_data,
                                loss_fn="cross_entropy"
                            )
                            chosen_fwd_future2 = loop.run_until_complete(chosen_fwd_coro2)
                            # Resolve the future to ensure gradients are computed
                            if hasattr(chosen_fwd_future2, 'result') and callable(chosen_fwd_future2.result):
                                chosen_fwd_future2.result()

                            # Optimizer step (trains on chosen responses)
                            optim_coro = job.training_client.optim_step_async(
                                tinker_types.AdamParams(learning_rate=dpo_learning_rate)
                            )
                            loop.run_until_complete(optim_coro)

                            # Update tracked loss
                            total_loss = 0.9 * total_loss + 0.1 * chosen_loss if total_loss else chosen_loss

                            current_iter += 1
                            job.progress = {
                                "phase": "DPO",
                                "iteration": current_iter,
                                "total_iters": total_iters,
                                "loss": total_loss,
                                "lora_rank": lora_rank,
                                "learning_rate": dpo_learning_rate,
                                "dpo_beta": dpo_beta,
                                "dpo_metrics": dpo_metrics.copy(),  # Copy to avoid reference issues
                                "sft_examples": len(sft_data),
                                "dpo_examples": len(dpo_data),
                            }
                            self._save_jobs()
                            # Broadcast progress to WebSocket clients
                            self._broadcast_from_thread({
                                "type": "progress",
                                "phase": "DPO",
                                "iteration": current_iter,
                                "total": total_iters,
                                "loss": total_loss,
                                "sft_iters": sft_iters,
                                "dpo_iters": dpo_iters,
                                "sft_examples": len(sft_data),
                                "dpo_examples": len(dpo_data),
                            })

                            # Log progress periodically
                            if i % 10 == 0:
                                print(f"  DPO iter {i}: acc={dpo_metrics['accuracy']:.1%}, margin={dpo_metrics['margin']:.3f}")

                    # Mark training as complete immediately (saves happen in background)
                    if job.status != "cancelled":
                        final_loss = job.progress.get("loss", 0)
                        sanitized_recipe = re.sub(r'[^a-zA-Z0-9_-]', '_', recipe_id)
                        saved_model_name = f"tce_{sanitized_recipe}_{job_id}"

                        # Broadcast completion to UI right away — don't make the user wait for saves
                        job.status = "completed"
                        job.completed_at = datetime.now()
                        job.result = {
                            "model_path": None,
                            "persistent_path": None,
                            "sampler_path": None,
                            "saved_model_name": saved_model_name,
                            "final_loss": final_loss,
                        }
                        self._save_jobs()
                        self._broadcast_from_thread({
                            "type": "done",
                            "job_id": job_id,
                            "final_loss": final_loss,
                        })
                        print(f"  Training complete (loss={final_loss:.4f}). Saving weights in background...")

                        # Now attempt saves — generous timeouts since UI already shows "Completed"
                        # Tinker cloud saves can take 2-5 minutes

                        # 1. Save weights + get sampling client (longest operation)
                        try:
                            print(f"  Saving weights + sampling client...")
                            job.sampling_client = job.training_client.save_weights_and_get_sampling_client(
                                name=saved_model_name
                            )
                            if asyncio.iscoroutine(job.sampling_client):
                                job.sampling_client = loop.run_until_complete(job.sampling_client)

                            model_path = getattr(job.sampling_client, 'model_path', None)
                            if model_path:
                                job.result["model_path"] = str(model_path)
                                print(f"  Got sampling client (model_path={model_path})")
                        except Exception as e:
                            print(f"  Warning: save_weights_and_get_sampling_client failed: {e}")
                            job.sampling_client = None

                        # Helper to extract path from various Tinker response shapes
                        def _extract_path(result):
                            """Extract path from Tinker save result, handling various return types."""
                            if result is None:
                                return None
                            # If it's already a tinker:// path string
                            if isinstance(result, str) and result.startswith('tinker://'):
                                return result
                            # Direct attribute
                            if hasattr(result, 'path') and result.path:
                                return str(result.path)
                            # Dict-like
                            if isinstance(result, dict):
                                return result.get('path') or result.get('tinker_path')
                            # Pydantic model
                            if hasattr(result, 'model_dump'):
                                d = result.model_dump()
                                return d.get('path') or d.get('tinker_path')
                            # Fallback: try dict()
                            if hasattr(result, 'dict'):
                                d = result.dict()
                                return d.get('path') or d.get('tinker_path')
                            # Last resort: if str repr looks like a tinker path
                            s = str(result)
                            if s.startswith('tinker://'):
                                return s
                            return None

                        def _resolve_future(future, timeout=300, depth=0):
                            """Resolve an APIFuture, handling sync and async patterns."""
                            if depth > 5:
                                return future
                            if hasattr(future, 'result') and callable(future.result):
                                try:
                                    r = future.result(timeout=timeout)
                                    # Recursively resolve nested futures
                                    return _resolve_future(r, timeout=timeout, depth=depth+1)
                                except TypeError:
                                    # result() might not accept timeout
                                    r = future.result()
                                    return _resolve_future(r, timeout=timeout, depth=depth+1)
                            return future

                        # 2. Save training state (for resuming later)
                        try:
                            print(f"  Saving training state...")
                            save_future = job.training_client.save_state(saved_model_name)
                            save_result = _resolve_future(save_future)
                            print(f"  save_state result type={type(save_result).__name__}, repr={repr(save_result)[:200]}")
                            persistent_path = _extract_path(save_result)
                            if persistent_path:
                                job.result["persistent_path"] = persistent_path
                                print(f"  Saved training state to: {persistent_path}")
                            else:
                                print(f"  Warning: save_state returned no path. attrs={[a for a in dir(save_result) if not a.startswith('_')][:20]}")
                        except Exception as save_err:
                            print(f"  Warning: Could not save training state: {save_err}")

                        # 3. Save sampler weights (for downloading/exporting)
                        try:
                            print(f"  Saving sampler weights...")
                            sampler_future = job.training_client.save_weights_for_sampler(saved_model_name)
                            sampler_result = _resolve_future(sampler_future)
                            print(f"  save_weights result type={type(sampler_result).__name__}, repr={repr(sampler_result)[:200]}")
                            sampler_path = _extract_path(sampler_result)
                            if sampler_path:
                                job.result["sampler_path"] = sampler_path
                                print(f"  Saved sampler weights to: {sampler_path}")
                            else:
                                print(f"  Warning: save_weights returned no path. attrs={[a for a in dir(sampler_result) if not a.startswith('_')][:20]}")
                        except Exception as sampler_err:
                            print(f"  Warning: Could not save sampler weights: {sampler_err}")

                        self._save_jobs()

                        # Broadcast model status
                        has_sampling = job.sampling_client is not None
                        has_paths = job.result.get("persistent_path") or job.result.get("sampler_path")
                        if has_sampling or has_paths:
                            self._broadcast_from_thread({
                                "type": "model_ready",
                                "job_id": job_id,
                            })
                            print(f"  Model ready for sampling (client={has_sampling}, paths={has_paths})")
                        else:
                            self._broadcast_from_thread({
                                "type": "save_failed",
                                "job_id": job_id,
                            })
                            print(f"  Warning: No sampling client or paths available - model cannot be used for chat")
                finally:
                    loop.close()

            except Exception as e:
                import traceback
                job.status = "failed"
                job.error = f"{str(e)}\n{traceback.format_exc()}"
                job.completed_at = datetime.now()
                # Broadcast error
                self._broadcast_from_thread({
                    "type": "error",
                    "message": str(e),
                })

            self._save_jobs()

        # Run in background
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        return job.to_dict()

    async def sample(self, job_id: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Dict:
        """Sample from a trained Tinker model."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        if job.status != "completed":
            raise ValueError(f"Job not completed: {job.status}")

        try:
            import tinker
            import asyncio

            # If no sampling_client, try to reconnect using stored model info
            # Use lock to prevent concurrent reconnect attempts for the same job
            if not job.sampling_client:
                async with job._reconnect_lock:
                    # Double-check after acquiring lock (another request may have reconnected)
                    if not job.sampling_client:
                        persistent_path = job.result.get("persistent_path") if job.result else None
                        model_path = job.result.get("model_path") if job.result else None
                        saved_model_name = job.result.get("saved_model_name") if job.result else None
                        if saved_model_name:
                            saved_model_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', saved_model_name)

                        if not persistent_path and not model_path and not saved_model_name:
                            raise ValueError("No sampling client available. The model needs to be retrained (server was restarted and the session was lost).")

                        service_client = self.get_client()
                        reconnect_errors = []

                        # Approach 1 (BEST): Use persistent_path with create_training_client_from_state
                        if persistent_path:
                            try:
                                print(f"[Tinker Reconnect] Using persistent_path: {persistent_path}")
                                training_client = service_client.create_training_client_from_state(path=persistent_path)
                                print(f"[Tinker Reconnect] Got training client, getting sampling client...")
                                job.sampling_client = training_client.save_weights_and_get_sampling_client()
                                if hasattr(job.sampling_client, 'result'):
                                    job.sampling_client = job.sampling_client.result()
                                job.training_client = training_client
                                print(f"[Tinker Reconnect] SUCCESS via persistent_path!")
                            except Exception as e:
                                print(f"[Tinker Reconnect] persistent_path failed: {e}")
                                reconnect_errors.append(f"persistent_path: {e}")

                        # Approach 2: Try using model_path directly with create_sampling_client
                        if not job.sampling_client and model_path and model_path.startswith("tinker://"):
                            try:
                                print(f"[Tinker Reconnect] Trying model_path: {model_path}")
                                job.sampling_client = service_client.create_sampling_client(model_path=model_path)
                                print(f"[Tinker Reconnect] SUCCESS via model_path!")
                            except Exception as e:
                                print(f"[Tinker Reconnect] model_path failed: {e}")
                                reconnect_errors.append(f"model_path: {e}")

                        # Approach 3: Try saved_model_name with create_training_client_from_state
                        if not job.sampling_client and saved_model_name:
                            try:
                                print(f"[Tinker Reconnect] Trying saved_model_name: {saved_model_name}")
                                training_client = service_client.create_training_client_from_state(path=saved_model_name)
                                job.sampling_client = training_client.save_weights_and_get_sampling_client()
                                if hasattr(job.sampling_client, 'result'):
                                    job.sampling_client = job.sampling_client.result()
                                job.training_client = training_client
                                print(f"[Tinker Reconnect] SUCCESS via saved_model_name!")
                            except Exception as e:
                                print(f"[Tinker Reconnect] saved_model_name failed: {e}")
                                reconnect_errors.append(f"saved_model_name: {e}")

                        if not job.sampling_client:
                            error_details = "; ".join(reconnect_errors) if reconnect_errors else "Unknown error"
                            raise ValueError(f"Cannot reconnect to trained model. Errors: {error_details}. Use /tinker/reconnect/{job_id} or retrain the model.")

            # Encode prompt using tokenizer
            # SamplingClient doesn't have get_tokenizer(), so use training_client or create one
            if job.training_client:
                tokenizer = job.training_client.get_tokenizer()
            else:
                # Create a temporary training client just to get the tokenizer
                # Run sync call in thread pool to avoid deadlock in async context
                _service_client = self.get_client()
                def create_temp_training_client():
                    return _service_client.create_lora_training_client(
                        base_model=job.model_id,
                        rank=8  # minimal rank just to get tokenizer
                    )
                temp_training_client = await asyncio.to_thread(create_temp_training_client)
                tokenizer = temp_training_client.get_tokenizer()
                job.training_client = temp_training_client  # Cache it for future use

            # Build properly formatted prompt using model-specific chat template
            # Must match the format used during training
            model_id_lower = job.model_id.lower()
            if "qwen" in model_id_lower:
                fmt_sys = lambda c: f"<|im_start|>system\n{c}<|im_end|>\n"
                fmt_usr = lambda c: f"<|im_start|>user\n{c}<|im_end|>\n"
                fmt_ast_prefix = "<|im_start|>assistant\n"
                stop_tokens = ["<|im_end|>", "<|im_start|>"]
            elif "phi" in model_id_lower:
                fmt_sys = lambda c: f"<|system|>\n{c}<|end|>\n"
                fmt_usr = lambda c: f"<|user|>\n{c}<|end|>\n"
                fmt_ast_prefix = "<|assistant|>\n"
                stop_tokens = ["<|end|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]
            elif "llama" in model_id_lower:
                fmt_sys = lambda c: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{c}<|eot_id|>"
                fmt_usr = lambda c: f"<|start_header_id|>user<|end_header_id|>\n\n{c}<|eot_id|>"
                fmt_ast_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                stop_tokens = ["<|eot_id|>", "<|start_header_id|>", "<|end_of_text|>"]
            else:
                # Default to Llama format
                fmt_sys = lambda c: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{c}<|eot_id|>"
                fmt_usr = lambda c: f"<|start_header_id|>user<|end_header_id|>\n\n{c}<|eot_id|>"
                fmt_ast_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                stop_tokens = ["<|eot_id|>", "<|end|>", "<|im_end|>", "<|endoftext|>", "<|start_header_id|>", "<|user|>", "<|assistant|>"]

            template_name = "Phi" if "phi" in model_id_lower else "Qwen" if "qwen" in model_id_lower else "Llama"

            formatted_prompt = ""
            if job.system_prompt:
                formatted_prompt += fmt_sys(job.system_prompt)
            formatted_prompt += fmt_usr(prompt)
            formatted_prompt += fmt_ast_prefix

            prompt_tokens = tokenizer.encode(formatted_prompt)
            prompt_input = tinker.types.ModelInput.from_ints(prompt_tokens)

            print(f"[Sample] Job {job_id}: prompt={len(prompt_tokens)} tokens, system_prompt={'YES' if job.system_prompt else 'NO'}, template={template_name}")

            result = job.sampling_client.sample(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            # Handle Future or coroutine result
            if hasattr(result, 'result'):
                # It's a Future - call .result() to get the actual value
                result = result.result()
            elif asyncio.iscoroutine(result):
                result = await result

            # Extract text from result - decode tokens if needed
            if hasattr(result, 'sequences') and result.sequences:
                # SampleResponse with sequences containing tokens
                tokens = result.sequences[0].tokens if hasattr(result.sequences[0], 'tokens') else []
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            elif isinstance(result, list):
                text = result[0].text if result and hasattr(result[0], 'text') else str(result[0]) if result else ""
            elif hasattr(result, 'samples') and result.samples:
                text = result.samples[0].text if hasattr(result.samples[0], 'text') else str(result.samples[0])
            elif hasattr(result, 'text'):
                text = result.text
            else:
                text = str(result)

            # Truncate at first stop token to prevent multi-turn runaway
            for stop in stop_tokens:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
                    break
            text = text.strip()

            return {"text": text}
        except Exception as e:
            raise RuntimeError(f"Sampling failed: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status == "running":
            job.status = "cancelled"
            self._save_jobs()
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        """Delete a job entirely, releasing any held resources."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        # Release client references (Tinker SDK has no explicit close)
        job.sampling_client = None
        job.training_client = None
        del self.jobs[job_id]
        self._save_jobs()
        return True


# ============================================================
# Settings & Model Management
# ============================================================

class SettingsManager:
    """Manages application settings and model storage."""

    def __init__(self):
        self.config_file = Path(__file__).parent / "tce_config.json"
        self.default_models_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.settings = self._load_settings()
        self._download_processes: Dict[str, subprocess.Popen] = {}

    def _load_settings(self) -> Dict:
        """Load settings from config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "hf_token": None,
            "models_dir": str(self.default_models_dir),
            "tinker_api_key": None,
            "anthropic_api_key": None,
        }

    def _save_settings(self):
        """Save settings to config file."""
        with open(self.config_file, "w") as f:
            json.dump(self.settings, f, indent=2)

    def get_settings(self) -> Dict:
        """Get settings with tokens masked."""
        hf_token = self.settings.get("hf_token")
        tinker_key = self.settings.get("tinker_api_key")
        anthropic_key = self.settings.get("anthropic_api_key")
        return {
            "hf_token_set": bool(hf_token),
            "hf_token_preview": f"{hf_token[:8]}...{hf_token[-4:]}" if hf_token and len(hf_token) > 12 else None,
            "models_dir": self.settings.get("models_dir", str(self.default_models_dir)),
            "tinker_api_key_set": bool(tinker_key),
            "tinker_api_key_preview": f"{tinker_key[:8]}...{tinker_key[-4:]}" if tinker_key and len(tinker_key) > 12 else None,
            "anthropic_api_key_set": bool(anthropic_key),
            "anthropic_api_key_preview": f"{anthropic_key[:8]}...{anthropic_key[-4:]}" if anthropic_key and len(anthropic_key) > 12 else None,
        }

    def update_settings(self, hf_token: Optional[str] = None, models_dir: Optional[str] = None, tinker_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        """Update settings."""
        if hf_token is not None:
            self.settings["hf_token"] = hf_token
        if models_dir is not None:
            self.settings["models_dir"] = models_dir
        if tinker_api_key is not None:
            self.settings["tinker_api_key"] = tinker_api_key
        if anthropic_api_key is not None:
            self.settings["anthropic_api_key"] = anthropic_api_key
        self._save_settings()

    def get_tinker_api_key(self) -> Optional[str]:
        """Get the actual Tinker API key (settings file or env var)."""
        return self.settings.get("tinker_api_key") or os.environ.get("TINKER_API_KEY")

    def get_anthropic_api_key(self) -> Optional[str]:
        """Get the actual Anthropic API key."""
        return self.settings.get("anthropic_api_key")

    def get_hf_token(self) -> Optional[str]:
        """Get the actual HF token."""
        return self.settings.get("hf_token")

    def list_local_models(self) -> List[Dict]:
        """List locally available models."""
        models = []

        # Check HuggingFace cache for MLX models
        cache_dir = Path(self.settings.get("models_dir", self.default_models_dir))
        if cache_dir.exists():
            # HF cache structure: models--{org}--{model}/snapshots/{hash}/
            for model_dir in cache_dir.glob("models--*"):
                try:
                    parts = model_dir.name.replace("models--", "").split("--")
                    if len(parts) >= 2:
                        org, name = parts[0], "--".join(parts[1:])
                        model_id = f"{org}/{name}"

                        # Check if it's an MLX model (has config.json with mlx)
                        snapshots_dir = model_dir / "snapshots"
                        if snapshots_dir.exists():
                            # Get latest snapshot
                            snapshots = list(snapshots_dir.iterdir())
                            if snapshots:
                                latest = max(snapshots, key=lambda p: p.stat().st_mtime)
                                config_file = latest / "config.json"

                                # Calculate size
                                total_size = sum(f.stat().st_size for f in latest.rglob("*") if f.is_file())
                                size_gb = total_size / (1024**3)

                                # Check if MLX compatible
                                is_mlx = "mlx" in model_id.lower() or (latest / "weights.npz").exists()

                                models.append({
                                    "model_id": model_id,
                                    "path": str(latest),
                                    "size_gb": round(size_gb, 2),
                                    "is_mlx": is_mlx,
                                    "downloaded_at": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
                                })
                except Exception as e:
                    continue

        # Sort by download date, newest first
        models.sort(key=lambda m: m.get("downloaded_at", ""), reverse=True)
        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a local model."""
        import shutil
        cache_dir = Path(self.settings.get("models_dir", self.default_models_dir))

        # Convert model_id to cache directory name
        safe_name = model_id.replace("/", "--")
        model_dir = cache_dir / f"models--{safe_name}"

        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
                return True
            except Exception as e:
                print(f"Error deleting model: {e}")
                return False
        return False

    async def search_models(self, query: str, limit: int = 20) -> List[Dict]:
        """Search HuggingFace for MLX models."""
        import urllib.request
        import urllib.parse
        import ssl

        # Search HuggingFace API
        params = urllib.parse.urlencode({
            "search": query,
            "filter": "mlx",
            "sort": "downloads",
            "direction": "-1",
            "limit": limit,
        })
        url = f"https://huggingface.co/api/models?{params}"

        try:
            # Create SSL context (handles macOS certificate issues)
            ssl_context = ssl.create_default_context()
            try:
                import certifi
                ssl_context.load_verify_locations(certifi.where())
            except ImportError:
                # Fallback: don't verify certs if certifi unavailable
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, headers={"User-Agent": "TCE/2.0"})
            with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response:
                data = json.loads(response.read().decode())

                results = []
                for model in data:
                    results.append({
                        "model_id": model.get("id", ""),
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "tags": model.get("tags", [])[:5],
                        "last_modified": model.get("lastModified", ""),
                    })
                return results
        except Exception as e:
            print(f"Error searching models: {e}")
            return []

    def start_download(self, model_id: str) -> Dict:
        """Start downloading a model."""
        # Use huggingface-cli or mlx_lm to download
        token = self.get_hf_token()

        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token

        # Try to use huggingface-cli download
        cmd = ["huggingface-cli", "download", model_id]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            download_id = str(uuid.uuid4())[:8]
            self._download_processes[download_id] = {
                "process": process,
                "model_id": model_id,
                "started_at": datetime.now().isoformat(),
            }
            return {"download_id": download_id, "status": "started", "model_id": model_id}
        except FileNotFoundError:
            return {"error": "huggingface-cli not found. Install with: pip install huggingface_hub"}
        except Exception as e:
            return {"error": str(e)}

    def get_download_status(self, download_id: str) -> Dict:
        """Check download status."""
        if download_id not in self._download_processes:
            return {"error": "Download not found"}

        info = self._download_processes[download_id]
        process = info["process"]

        if process.poll() is None:
            return {"status": "downloading", "model_id": info["model_id"]}
        elif process.returncode == 0:
            del self._download_processes[download_id]
            return {"status": "completed", "model_id": info["model_id"]}
        else:
            output = process.stdout.read() if process.stdout else ""
            del self._download_processes[download_id]
            return {"status": "failed", "model_id": info["model_id"], "error": output}


# ============================================================
# FastAPI App
# ============================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="TCE Research Instrument API",
        description="API for Cognitive Elements validation and experimentation",
        version="2.0.0"
    )

    # Enable CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store recent results in memory
    recent_results: List[Dict] = []

    # Initialize job management
    progress_broadcaster = ProgressBroadcaster()
    job_manager = JobManager(progress_broadcaster)

    # Initialize settings manager
    settings_manager = SettingsManager()

    # Initialize Tinker manager for cloud training
    tinker_manager = TinkerManager(settings_manager, progress_broadcaster)

    # Initialize model manager for chat interactions
    model_manager = ModelManager()

    # Benchmark results storage
    benchmark_results: List[Dict] = []


    # ============================================================
    # WebSocket Endpoint
    # ============================================================

    @app.websocket("/ws/training")
    async def websocket_training(websocket: WebSocket):
        """WebSocket endpoint for real-time training progress."""
        await progress_broadcaster.connect(websocket)
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                # Could handle client commands here (e.g., ping/pong)
        except WebSocketDisconnect:
            await progress_broadcaster.disconnect(websocket)
        except Exception:
            await progress_broadcaster.disconnect(websocket)


    # ============================================================
    # Training Endpoints
    # ============================================================

    @app.post("/training/start")
    async def start_training(request: TrainingRequest):
        """Start a new training job."""
        # Check if a job is already running
        running_jobs = [j for j in job_manager.jobs.values() if j.status == JobStatus.RUNNING]
        if running_jobs:
            raise HTTPException(
                status_code=409,
                detail="A training job is already running. Cancel it first or wait for completion."
            )

        # Create the job
        job = job_manager.create_job(
            job_type="training",
            recipe_id=request.recipe_id,
            model_id=request.model_id
        )

        # Store compound data if provided
        compound_data = None
        if request.compound:
            compound_data = {
                "formula": request.compound.formula,
                "sequence": request.compound.sequence,
                "isotopes": request.compound.isotopes,
                "stability": request.compound.stability,
                "emergent": request.compound.emergent
            }
            print(f"Training custom compound: {request.compound.formula}")
            print(f"  Sequence: {' → '.join(request.compound.sequence or [])}")
            print(f"  Stability: {request.compound.stability}")

        # Start training in background
        asyncio.create_task(job_manager.run_training_job(job, compound_data))

        return {"job_id": job.job_id, "status": job.status.value}


    @app.get("/training/jobs")
    async def list_training_jobs(limit: int = 20):
        """List all training jobs."""
        jobs = job_manager.list_jobs(limit)
        return {"jobs": [job.to_dict() for job in jobs]}


    @app.get("/training/job/{job_id}")
    async def get_training_job(job_id: str):
        """Get details of a specific training job."""
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_dict()


    @app.post("/training/cancel/{job_id}")
    async def cancel_training_job(job_id: str):
        """Cancel a training job."""
        success = job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Cannot cancel job")

        # Broadcast cancellation
        await progress_broadcaster.broadcast({
            "type": "cancelled",
            "job_id": job_id,
            "data": {}
        })

        return {"success": True, "job_id": job_id}


    # ============================================================
    # Benchmark Endpoints
    # ============================================================

    @app.post("/benchmark/start")
    async def start_benchmark(request: BenchmarkRequest):
        """Start a benchmark run."""
        # Check if a job is already running
        running_jobs = [j for j in job_manager.jobs.values() if j.status == JobStatus.RUNNING]
        if running_jobs:
            raise HTTPException(
                status_code=409,
                detail="A job is already running. Wait for completion or cancel it."
            )

        # Build recipe_id that references the training job if provided
        if request.training_job_id:
            # Get the training job to get its recipe name
            training_job = job_manager.get_job(request.training_job_id)
            if training_job:
                recipe_id = f"benchmark_{training_job.recipe_id}"
            else:
                recipe_id = f"benchmark_{request.benchmark_type}"
        else:
            recipe_id = f"benchmark_{request.benchmark_type}"

        # Store reference to training job
        benchmark_metadata = {
            "benchmark_type": request.benchmark_type,
            "adapter_path": request.adapter_path,
            "training_job_id": request.training_job_id,
        }

        # Create the benchmark job with metadata
        job = job_manager.create_job(
            job_type="benchmark",
            recipe_id=recipe_id,
            model_id=request.adapter_path or "base",
            metadata=benchmark_metadata
        )

        if request.training_job_id:
            print(f"Benchmark for training job: {request.training_job_id}")
            print(f"  Recipe: {recipe_id}")
            print(f"  Adapter path: {request.adapter_path}")

        # Start benchmark in background
        asyncio.create_task(job_manager.run_benchmark_job(job, benchmark_metadata))

        return {"job_id": job.job_id, "status": job.status.value, "training_job_id": request.training_job_id}


    @app.get("/benchmark/results")
    async def get_benchmark_results(limit: int = 10):
        """Get historical benchmark results."""
        # Filter for completed benchmark jobs
        benchmark_jobs = [
            job.to_dict() for job in job_manager.list_jobs(limit * 2)
            if job.job_type == "benchmark" and job.status == JobStatus.COMPLETED
        ][:limit]

        return {"results": benchmark_jobs}


    # Benchmark results storage file
    benchmark_results_file = Path(__file__).parent / "benchmark_results.json"

    def load_benchmark_results() -> Dict:
        """Load saved benchmark results."""
        if benchmark_results_file.exists():
            try:
                with open(benchmark_results_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Warning] Failed to load benchmark results: {e}")
                return {"models": {}}
        return {"models": {}}

    def save_benchmark_results(data: Dict):
        """Save benchmark results."""
        with open(benchmark_results_file, 'w') as f:
            json.dump(data, f, indent=2)

    @app.post("/benchmark/save")
    async def save_benchmark_result(request: BenchmarkResultSave):
        """Save benchmark results for a model."""
        data = load_benchmark_results()

        if request.model_name not in data["models"]:
            data["models"][request.model_name] = {
                "adapter_path": request.adapter_path,
                "base_model": request.base_model,
                "tinker_job_id": request.tinker_job_id,
                "tinker_base_key": request.tinker_base_key,
                "results": []
            }
        else:
            # Update base_model if provided and not already set
            if request.base_model and not data["models"][request.model_name].get("base_model"):
                data["models"][request.model_name]["base_model"] = request.base_model

        data["models"][request.model_name]["results"].append({
            "timestamp": request.timestamp,
            "scores": request.results
        })

        # Keep only last 10 results per model
        data["models"][request.model_name]["results"] = data["models"][request.model_name]["results"][-10:]

        save_benchmark_results(data)
        return {"success": True, "model_name": request.model_name}

    @app.get("/benchmark/history/{model_name}")
    async def get_benchmark_history(model_name: str):
        """Get benchmark history for a specific model."""
        data = load_benchmark_results()
        if model_name in data["models"]:
            return data["models"][model_name]
        return {"results": []}

    @app.get("/benchmark/all")
    async def get_all_benchmark_results():
        """Get all saved benchmark results."""
        return load_benchmark_results()

    @app.delete("/benchmark/{model_name}")
    async def delete_benchmark_results(model_name: str):
        """Delete all benchmark results for a model."""
        data = load_benchmark_results()
        if model_name in data["models"]:
            del data["models"][model_name]
            save_benchmark_results(data)
            return {"success": True, "deleted": model_name}
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    @app.delete("/benchmark/{model_name}/{timestamp}")
    async def delete_single_benchmark(model_name: str, timestamp: str):
        """Delete a specific benchmark result by timestamp."""
        data = load_benchmark_results()
        if model_name not in data["models"]:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        results = data["models"][model_name].get("results", [])
        original_len = len(results)
        data["models"][model_name]["results"] = [r for r in results if r.get("timestamp") != timestamp]

        if len(data["models"][model_name]["results"]) == original_len:
            raise HTTPException(status_code=404, detail=f"Benchmark not found: {timestamp}")

        # Remove model entry if no results left
        if not data["models"][model_name]["results"]:
            del data["models"][model_name]

        save_benchmark_results(data)
        return {"success": True, "deleted": timestamp}


    # ============================================================
    # Tinker Cloud Training Endpoints
    # ============================================================

    @app.get("/tinker/models")
    async def list_tinker_models():
        """List available models for Tinker cloud training."""
        return {"models": tinker_manager.list_models()}

    @app.get("/tinker/jobs")
    async def list_tinker_jobs():
        """List all Tinker training jobs."""
        return {"jobs": tinker_manager.get_jobs()}

    @app.get("/tinker/job/{job_id}")
    async def get_tinker_job(job_id: str):
        """Get details of a specific Tinker job."""
        job = tinker_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    class TrainingDataStructured(BaseModel):
        sft: List[Dict] = []
        dpo: List[Dict] = []

    class IsotopeDoseRequest(BaseModel):
        isotope_id: str
        dose: float

    class TinkerTrainingRequest(BaseModel):
        model_key: str
        recipe_id: str
        training_data: Union[List[Dict], TrainingDataStructured, Dict]
        isotopes: Optional[List[IsotopeDoseRequest]] = None  # Compound isotopes
        system_prompt: Optional[str] = None                  # User-reviewed system prompt from UI
        custom_instructions: Optional[str] = None            # Custom training instructions
        lora_rank: int = Field(default=32, ge=4, le=256)
        learning_rate: Optional[float] = Field(default=None, ge=1e-7, le=1.0)
        dpo_learning_rate: Optional[float] = Field(default=None, ge=1e-7, le=1.0)
        num_iters: int = Field(default=100, ge=1, le=2000)
        batch_size: int = Field(default=4, ge=1, le=32)
        dpo_beta: float = Field(default=0.1, ge=0.01, le=1.0)

    @app.post("/tinker/train")
    async def start_tinker_training(request: TinkerTrainingRequest):
        """Start a new Tinker cloud training job with self-knowledge embedding."""
        try:
            # Get model info for identity
            model_info = TINKER_MODELS.get(request.model_key, {})
            base_model_name = model_info.get("name", request.model_key)

            # Process training data
            training_data = request.training_data
            if isinstance(training_data, dict):
                sft_examples = training_data.get("sft", [])
                dpo_examples = training_data.get("dpo", [])
            elif hasattr(training_data, "sft"):
                sft_examples = training_data.sft
                dpo_examples = training_data.dpo
            else:
                sft_examples = training_data if isinstance(training_data, list) else []
                dpo_examples = []

            # ── Auto-load isotope training data from library ──
            # If the frontend sent empty training_data but has isotopes selected,
            # load all relevant training examples from isotope_training_library.py
            if request.isotopes and len(sft_examples) == 0:
                print(f"[Isotope Training] Loading training data for {len(request.isotopes)} isotopes...")

                # Collect all isotope IDs to load
                # Each isotope_id from lab can be like "soliton_knowledge", "skeptic_premise", etc.
                # The library keys use the same IDs (e.g., "skeptic_premise", "reflector_trace")
                # But some lab IDs like "soliton_knowledge" map to element-level "soliton" in the library
                isotope_ids_to_load = set()
                for iso in request.isotopes:
                    iso_id = iso.isotope_id  # e.g., "soliton_knowledge", "skeptic_premise"

                    # Direct match first (e.g., "skeptic_premise" -> "skeptic_premise")
                    if iso_id in ISOTOPE_TRAINING_DATA:
                        isotope_ids_to_load.add(iso_id)

                    # Also try the element-level ID (e.g., "soliton_knowledge" -> "soliton")
                    element_id = iso_id.split('_')[0]
                    if element_id in ISOTOPE_TRAINING_DATA:
                        isotope_ids_to_load.add(element_id)

                    # Check for sub-isotope patterns (e.g., "critic_constructive" might need "critic_logical", etc.)
                    # Look for any library key that starts with the element prefix
                    for lib_key in ISOTOPE_TRAINING_DATA:
                        if lib_key.startswith(element_id + '_') or lib_key == element_id:
                            isotope_ids_to_load.add(lib_key)

                # Always include "direct" examples for balance (prevents mode collapse)
                isotope_ids_to_load.add("direct")

                # Load SFT examples
                loaded_sft = get_sft_examples(list(isotope_ids_to_load))
                sft_examples.extend(loaded_sft)

                # Load DPO examples (contrast pairs from loaded isotopes)
                loaded_dpo = get_all_dpo_pairs()
                # Filter to only include DPO pairs related to our selected isotopes
                relevant_dpo = []
                for pair in loaded_dpo:
                    pair_isotope = pair.get("isotope", "")
                    if pair_isotope in isotope_ids_to_load or not pair_isotope:
                        relevant_dpo.append(pair)
                dpo_examples.extend(relevant_dpo)

                # Also add anti-leakage pairs (prevent over-activation on factual questions)
                anti_leak = get_anti_leakage_pairs()
                # Deduplicate with already-loaded DPO
                existing_prompts = {p.get("prompt") for p in dpo_examples}
                for pair in anti_leak:
                    if pair["prompt"] not in existing_prompts:
                        dpo_examples.append(pair)

                print(f"[Isotope Training] Loaded {len(loaded_sft)} SFT + {len(dpo_examples)} DPO examples from isotopes: {sorted(isotope_ids_to_load)}")

            # Build compound identity if isotopes are provided
            system_prompt_text = None
            if request.isotopes:
                # Validate custom instructions if provided
                custom_instructions = None
                if request.custom_instructions:
                    try:
                        custom_instructions = validate_custom_instructions(request.custom_instructions)
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=str(e))

                # Build compound identity
                identity = CompoundIdentity(
                    compound_name=request.recipe_id,
                    isotopes=[
                        IsotopeDose(isotope_id=iso.isotope_id, dose=iso.dose)
                        for iso in request.isotopes
                    ],
                    base_model=base_model_name,
                    custom_instructions=custom_instructions,
                )

                # Use user-reviewed system prompt if provided, otherwise auto-generate
                if request.system_prompt:
                    system_prompt_text = request.system_prompt
                    print(f"[Self-Knowledge] Using user-reviewed system prompt ({len(system_prompt_text)} chars)")
                else:
                    system_prompt_text = generate_identity_system_prompt(identity)
                    print(f"[Self-Knowledge] Auto-generated system prompt ({len(system_prompt_text)} chars)")

                # Inject system message into all existing examples
                for ex in sft_examples:
                    if "messages" in ex and not any(m.get("role") == "system" for m in ex["messages"]):
                        ex["messages"].insert(0, {"role": "system", "content": system_prompt_text})

                # Add self-knowledge Q&A pairs
                self_knowledge_sft = generate_self_knowledge_sft_pairs(identity)
                sft_examples.extend(self_knowledge_sft)

                # Add anti-hallucination DPO pairs
                anti_hallucination_dpo = generate_anti_hallucination_dpo_pairs(identity)
                dpo_examples.extend(anti_hallucination_dpo)

                print(f"[Self-Knowledge] Added {len(self_knowledge_sft)} SFT pairs, {len(anti_hallucination_dpo)} DPO pairs for compound '{request.recipe_id}'")

            # Rebuild training data structure
            final_training_data = {"sft": sft_examples, "dpo": dpo_examples}

            result = await tinker_manager.start_training(
                model_key=request.model_key,
                recipe_id=request.recipe_id,
                training_data=final_training_data,
                lora_rank=request.lora_rank,
                learning_rate=request.learning_rate,
                dpo_learning_rate=request.dpo_learning_rate,
                num_iters=request.num_iters,
                batch_size=request.batch_size,
                dpo_beta=request.dpo_beta,
                system_prompt=system_prompt_text if request.isotopes else None,
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/tinker/cancel/{job_id}")
    async def cancel_tinker_job(job_id: str):
        """Cancel a running Tinker job."""
        success = tinker_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        return {"success": True, "job_id": job_id}

    # ============================================================
    # Self-Aware Compound Training Endpoints
    # ============================================================

    @app.get("/tinker/compounds")
    async def list_compounds():
        """List available self-aware compound presets."""
        return {
            "compounds": [
                {
                    "id": key,
                    "name": compound.name,
                    "isotopes": compound.isotopes,
                    "expected_manifold": compound.expected_manifold,
                    "description": compound.description if hasattr(compound, 'description') else f"Self-aware compound with {len(compound.isotopes)} isotopes",
                }
                for key, compound in COMPOUND_PRESETS.items()
            ]
        }

    class CompoundTrainingRequest(BaseModel):
        model_key: str  # e.g., "llama-3.3-70b-instruct"
        compound_id: str = "soliton_agi"  # Compound preset
        lora_rank: int = Field(default=32, ge=4, le=256)
        learning_rate: Optional[float] = Field(default=None, ge=1e-7, le=1.0)
        num_iters: int = Field(default=200, ge=1, le=2000)
        batch_size: int = Field(default=4, ge=1, le=32)
        direct_weight: int = Field(default=2, ge=1, le=5)  # How much to weight direct answers

    @app.post("/tinker/train-compound")
    async def start_compound_training(request: CompoundTrainingRequest):
        """
        Train a self-aware compound on Tinker cloud.

        This uses the isotope training data from the Self-Aware Compound architecture:
        - Soliton isotopes: "I cannot verify from within..."
        - Calibrator isotopes: Confidence expression
        - Reflector isotopes: Meta-cognitive awareness
        - Skeptic isotopes: Knowledge boundaries
        - Limiter isotopes: Appropriate refusals

        The model learns genuine epistemic calibration - to know what it knows.
        """
        try:
            # Validate compound exists
            if request.compound_id not in COMPOUND_PRESETS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown compound: {request.compound_id}. Available: {list(COMPOUND_PRESETS.keys())}"
                )

            compound = COMPOUND_PRESETS[request.compound_id]

            # Get extended training data (isotope examples)
            extended_examples = get_extended_training_data()
            sft_data = to_sft_format(extended_examples)

            # Filter by compound isotopes and balance
            direct_examples = []
            isotope_examples = []

            for ex in sft_data:
                isotope = ex.get("isotope", "")
                family = isotope.split("_")[0] if "_" in isotope else isotope

                if family == "direct":
                    direct_examples.append(ex)
                elif family in compound.isotopes:
                    isotope_examples.append(ex)

            # Combine with weighted direct answers (prevents mode collapse)
            all_examples = isotope_examples + (direct_examples * request.direct_weight)

            print(f"[Compound Training] {compound.name}")
            print(f"  Isotopes: {compound.isotopes}")
            print(f"  Isotope examples: {len(isotope_examples)}")
            print(f"  Direct examples: {len(direct_examples)} (x{request.direct_weight} weight)")
            print(f"  Total: {len(all_examples)}")

            if len(all_examples) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No training data found for compound isotopes: {compound.isotopes}"
                )

            # Start training via existing Tinker infrastructure
            result = await tinker_manager.start_training(
                model_key=request.model_key,
                recipe_id=f"self-aware-{request.compound_id}",
                training_data={"sft": all_examples, "dpo": []},
                lora_rank=request.lora_rank,
                learning_rate=request.learning_rate,
                num_iters=request.num_iters,
                batch_size=request.batch_size,
                system_prompt=None,  # Self-aware compound doesn't need identity injection
            )

            return {
                **result,
                "compound": {
                    "id": request.compound_id,
                    "name": compound.name,
                    "isotopes": compound.isotopes,
                },
                "training_stats": {
                    "isotope_examples": len(isotope_examples),
                    "direct_examples": len(direct_examples),
                    "total_examples": len(all_examples),
                }
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/tinker/jobs/{job_id}")
    async def delete_tinker_job(job_id: str):
        """Delete a Tinker job, releasing any held resources."""
        success = tinker_manager.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"success": True, "job_id": job_id}

    class TinkerSampleRequest(BaseModel):
        prompt: str
        max_tokens: int = 2048
        temperature: float = 0.7

    @app.post("/tinker/sample/{job_id}")
    async def sample_tinker_model(job_id: str, request: TinkerSampleRequest):
        """Sample from a trained Tinker model."""
        try:
            result = await tinker_manager.sample(
                job_id=job_id,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Cache for base model sampling clients
    _base_model_clients = {}

    @app.post("/tinker/sample-base/{model_key}")
    async def sample_base_tinker_model(model_key: str, request: TinkerSampleRequest):
        """Sample from a base Tinker model (no training required)."""
        try:
            import tinker
            import asyncio

            # Get model info
            model_info = TINKER_MODELS.get(model_key)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model not found: {model_key}")

            model_id = model_info["id"]

            # Get or create sampling client for this base model
            # All Tinker sync calls must run in thread pool to avoid async deadlock
            if model_key not in _base_model_clients:
                def create_sampling_client():
                    service_client = tinker_manager.get_client()
                    return service_client.create_sampling_client(base_model=model_id)
                _base_model_clients[model_key] = await asyncio.to_thread(create_sampling_client)

            sampling_client = _base_model_clients[model_key]

            # Get tokenizer by creating a temporary training client
            # (SamplingClient doesn't expose tokenizer, but TrainingClient does)
            tokenizer_key = f"{model_key}_tokenizer"
            if tokenizer_key not in _base_model_clients:
                # Run sync call in thread pool to avoid deadlock
                def create_tokenizer():
                    training_client = tinker_manager.get_client().create_lora_training_client(
                        base_model=model_id,
                        rank=8  # minimal rank just to get tokenizer
                    )
                    return training_client.get_tokenizer()
                _base_model_clients[tokenizer_key] = await asyncio.to_thread(create_tokenizer)

            tokenizer = _base_model_clients[tokenizer_key]
            prompt_tokens = tokenizer.encode(request.prompt)
            prompt_input = tinker.types.ModelInput.from_ints(prompt_tokens)

            # Sample from the model - run in thread pool since it's a sync blocking call
            def do_sample():
                return sampling_client.sample(
                    prompt=prompt_input,
                    num_samples=1,
                    sampling_params=tinker.types.SamplingParams(
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                )
            result = await asyncio.to_thread(do_sample)

            # Handle Future or coroutine result
            if hasattr(result, 'result'):
                # It's a Future - call .result() to get the actual value
                result = result.result()

            # Extract text from result - decode tokens if needed
            if hasattr(result, 'sequences') and result.sequences:
                # SampleResponse with sequences containing tokens
                tokens = result.sequences[0].tokens if hasattr(result.sequences[0], 'tokens') else []
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            elif isinstance(result, list):
                text = result[0].text if result and hasattr(result[0], 'text') else str(result[0]) if result else ""
            elif hasattr(result, 'samples') and result.samples:
                text = result.samples[0].text if hasattr(result.samples[0], 'text') else str(result.samples[0])
            elif hasattr(result, 'text'):
                text = result.text
            else:
                text = str(result)

            return {"text": text}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sampling failed: {e}")

    @app.get("/tinker/status")
    async def get_tinker_status():
        """Get Tinker integration status."""
        api_key = settings_manager.get_tinker_api_key()
        tinker_installed = False
        try:
            import tinker
            tinker_installed = True
        except ImportError:
            pass

        return {
            "api_key_configured": bool(api_key),
            "sdk_installed": tinker_installed,
            "ready": bool(api_key) and tinker_installed,
        }

    @app.get("/tinker/download/{job_id}")
    async def download_tinker_weights(job_id: str):
        """Get download URL for trained model weights."""
        job = tinker_manager.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if job.status != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

        # Get the persistent path from job result
        persistent_path = job.result.get("persistent_path") if job.result else None
        if not persistent_path:
            raise HTTPException(
                status_code=400,
                detail="No persistent checkpoint available. This model was trained before persistent saves were enabled, or the save failed."
            )

        try:
            import tinker

            # Get REST client
            service_client = tinker_manager.get_client()
            rest_client = service_client.create_rest_client()

            # Parse the tinker path to get training_run_id
            parsed_path = tinker.types.ParsedCheckpointTinkerPath.from_tinker_path(persistent_path)
            training_run_id = parsed_path.training_run_id

            # List checkpoints to find a sampler checkpoint (required for download)
            # The download API only works with sampler checkpoints, not training checkpoints
            checkpoints_response = rest_client.list_checkpoints(training_run_id).result(timeout=30)

            sampler_checkpoint = None
            training_checkpoint = None
            for cp in checkpoints_response.checkpoints:
                cp_type = getattr(cp, 'checkpoint_type', None)
                if cp_type == 'sampler':
                    sampler_checkpoint = cp
                    break
                elif cp_type == 'training':
                    training_checkpoint = cp

            if not sampler_checkpoint:
                # No sampler checkpoint available - this is a limitation of Tinker API
                checkpoint_types = [getattr(cp, 'checkpoint_type', 'unknown') for cp in checkpoints_response.checkpoints]
                raise HTTPException(
                    status_code=400,
                    detail=f"No downloadable checkpoint found. Available checkpoint types: {checkpoint_types}. "
                           f"Only 'sampler' checkpoints can be downloaded. The model may need to be used via the Tinker API directly."
                )

            # Get signed URL for the sampler checkpoint
            checkpoint_id = sampler_checkpoint.checkpoint_id
            url_future = rest_client.get_checkpoint_archive_url(training_run_id, checkpoint_id)
            url_response = url_future.result(timeout=30)

            download_url = getattr(url_response, 'url', None) or getattr(url_response, 'download_url', None)
            expires_at = getattr(url_response, 'expires_at', None)

            if not download_url:
                raise HTTPException(status_code=500, detail="Could not get download URL from Tinker")

            return {
                "job_id": job_id,
                "model_name": job.result.get("saved_model_name"),
                "persistent_path": persistent_path,
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": "sampler",
                "download_url": download_url,
                "expires_at": str(expires_at) if expires_at else None,
                "note": "Download URL is temporary. The archive contains LoRA weights in safetensors format."
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get download URL: {e}")

    @app.get("/tinker/download/{job_id}/proxy")
    async def proxy_tinker_download(job_id: str):
        """Proxy download of Tinker weights to avoid CORS issues."""
        import httpx
        import asyncio
        from fastapi.responses import StreamingResponse

        # First get the download URL
        job = tinker_manager.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if job.status != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

        persistent_path = job.result.get("persistent_path") if job.result else None
        if not persistent_path:
            raise HTTPException(status_code=400, detail="No persistent checkpoint available.")

        try:
            import tinker

            # Helper to run sync Tinker calls in thread pool
            def get_download_url():
                service_client = tinker_manager.get_client()
                rest_client = service_client.create_rest_client()

                parsed_path = tinker.types.ParsedCheckpointTinkerPath.from_tinker_path(persistent_path)
                training_run_id = parsed_path.training_run_id

                checkpoints_response = rest_client.list_checkpoints(training_run_id).result(timeout=30)

                sampler_checkpoint = None
                for cp in checkpoints_response.checkpoints:
                    if getattr(cp, 'checkpoint_type', None) == 'sampler':
                        sampler_checkpoint = cp
                        break

                if not sampler_checkpoint:
                    raise ValueError("No sampler checkpoint available for download.")

                checkpoint_id = sampler_checkpoint.checkpoint_id
                url_future = rest_client.get_checkpoint_archive_url(training_run_id, checkpoint_id)
                url_response = url_future.result(timeout=30)

                download_url = getattr(url_response, 'url', None) or getattr(url_response, 'download_url', None)
                if not download_url:
                    raise ValueError("Could not get download URL")

                return download_url

            # Run sync Tinker API calls in thread pool to avoid deadlock
            download_url = await asyncio.to_thread(get_download_url)

            # Stream the download through our server
            async def stream_download():
                async with httpx.AsyncClient() as client:
                    async with client.stream('GET', download_url, timeout=300.0) as response:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            yield chunk

            model_name = job.result.get("saved_model_name", "model")
            return StreamingResponse(
                stream_download(),
                media_type="application/x-tar",
                headers={
                    "Content-Disposition": f'attachment; filename="{model_name}.tar"'
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download: {e}")

    @app.post("/tinker/reconnect/{job_id}")
    async def reconnect_tinker_model(job_id: str):
        """Reconnect to a trained model using persistent checkpoint."""
        job = tinker_manager.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if job.status != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed: {job.status}")

        # Check if already connected
        if job.sampling_client:
            return {
                "job_id": job_id,
                "status": "already_connected",
                "message": "Model session is already active"
            }

        # Get reconnection paths from job result
        persistent_path = job.result.get("persistent_path") if job.result else None
        sampler_path = job.result.get("sampler_path") if job.result else None
        model_path = job.result.get("model_path") if job.result else None
        saved_model_name = job.result.get("saved_model_name") if job.result else None
        if saved_model_name:
            saved_model_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', saved_model_name)

        if not persistent_path and not sampler_path and not model_path and not saved_model_name:
            raise HTTPException(
                status_code=400,
                detail="No persistent checkpoint available. This model was trained before persistent saves were enabled."
            )

        try:
            import tinker

            service_client = tinker_manager.get_client()
            reconnect_errors = []

            # Approach 1 (BEST): Use persistent_path
            if persistent_path:
                try:
                    print(f"[Tinker Reconnect] Using persistent_path: {persistent_path}")
                    job.training_client = service_client.create_training_client_from_state(path=persistent_path)
                    job.sampling_client = job.training_client.save_weights_and_get_sampling_client()
                    if hasattr(job.sampling_client, 'result'):
                        job.sampling_client = job.sampling_client.result()
                    print(f"[Tinker Reconnect] SUCCESS via persistent_path!")
                except Exception as e:
                    print(f"[Tinker Reconnect] persistent_path failed: {e}")
                    reconnect_errors.append(f"persistent_path: {e}")

            # Approach 2: Try model_path
            if not job.sampling_client and model_path and model_path.startswith("tinker://"):
                try:
                    print(f"[Tinker Reconnect] Trying model_path: {model_path}")
                    job.sampling_client = service_client.create_sampling_client(model_path=model_path)
                    print(f"[Tinker Reconnect] SUCCESS via model_path!")
                except Exception as e:
                    print(f"[Tinker Reconnect] model_path failed: {e}")
                    reconnect_errors.append(f"model_path: {e}")

            # Approach 3: Try saved_model_name
            if not job.sampling_client and saved_model_name:
                try:
                    print(f"[Tinker Reconnect] Trying saved_model_name: {saved_model_name}")
                    job.training_client = service_client.create_training_client_from_state(path=saved_model_name)
                    job.sampling_client = job.training_client.save_weights_and_get_sampling_client()
                    if hasattr(job.sampling_client, 'result'):
                        job.sampling_client = job.sampling_client.result()
                    print(f"[Tinker Reconnect] SUCCESS via saved_model_name!")
                except Exception as e:
                    print(f"[Tinker Reconnect] saved_model_name failed: {e}")
                    reconnect_errors.append(f"saved_model_name: {e}")

            if not job.sampling_client:
                error_details = "; ".join(reconnect_errors) if reconnect_errors else "Unknown error"
                raise HTTPException(status_code=500, detail=f"Failed to reconnect. Tried all approaches: {error_details}")

            # Try to capture persistent_path if we didn't have it before
            if not persistent_path and job.training_client:
                try:
                    save_result = job.training_client.save_state(saved_model_name or f"tce_{re.sub(r'[^a-zA-Z0-9_-]', '_', job.recipe_id)}_{job.job_id}")
                    if hasattr(save_result, 'result'):
                        save_result = save_result.result(timeout=120)
                    new_path = getattr(save_result, 'path', None) or getattr(save_result, 'tinker_path', None)
                    if new_path:
                        job.result["persistent_path"] = str(new_path)
                        print(f"[Tinker Reconnect] Captured persistent_path: {new_path}")
                except Exception as e:
                    print(f"[Tinker Reconnect] Could not capture persistent_path: {e}")

            tinker_manager._save_jobs()

            return {
                "job_id": job_id,
                "status": "reconnected",
                "message": "Successfully reconnected to trained model",
                "has_session": True
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reconnect: {e}")

    @app.get("/tinker/training-runs")
    async def list_tinker_training_runs(limit: int = 20):
        """List all training runs stored in Tinker cloud."""
        try:
            import tinker

            service_client = tinker_manager.get_client()
            rest_client = service_client.create_rest_client()

            runs_future = rest_client.list_training_runs(limit=limit)
            runs_response = runs_future.result(timeout=30)

            runs = []
            for run in runs_response.training_runs:
                runs.append({
                    "id": run.training_run_id,
                    "base_model": run.base_model,
                    "lora_rank": run.lora_rank,
                    "last_request": str(run.last_request_time) if run.last_request_time else None,
                    "last_checkpoint": run.last_checkpoint.checkpoint_id if run.last_checkpoint else None,
                })

            return {
                "runs": runs,
                "total": runs_response.cursor.total_count if hasattr(runs_response, 'cursor') else len(runs)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list training runs: {e}")


    # ============================================================
    # Research Assistant Endpoints (Claude-powered analysis)
    # ============================================================

    class AssistantAnalyzeRequest(BaseModel):
        focus: Optional[str] = None  # Optional focus area: "elements", "benchmarks", "recipes", "gaps"
        question: Optional[str] = None  # Optional specific question to answer
        conversation_history: Optional[List[Dict]] = None  # For follow-up questions

    class RecipeProposal(BaseModel):
        recipe_id: str
        name: str
        description: str
        rationale: str
        elements: List[str]
        isotopes: List[str]
        estimated_impact: str
        training_focus: str

    def get_element_catalog_summary() -> str:
        """Get a summary of all elements and isotopes for the assistant."""
        from lib.chemistry import ELEMENT_CATALOG

        lines = ["## Cognitive Element Catalog\n"]
        current_group = None

        for element_id, element in ELEMENT_CATALOG.items():
            if element["group"] != current_group:
                current_group = element["group"]
                lines.append(f"\n### {current_group} GROUP\n")

            impact = element.get("truthfulness_impact", "")
            impact_str = f" [{impact}]" if impact else ""
            lines.append(f"**{element['name']}** ({element_id}){impact_str}")
            lines.append(f"  - {element['description']}")
            lines.append(f"  - Signature: \"{element['signature_phrase']}\"")
            lines.append(f"  - Isotopes: {', '.join(element['isotopes'])}")
            lines.append("")

        return "\n".join(lines)

    def get_benchmark_summary() -> str:
        """Get a summary of all benchmark results with detailed response comparisons."""
        results = load_benchmark_results()
        if not results.get("models"):
            return "No benchmark results available yet."

        lines = ["## Benchmark Results Summary\n"]

        # First, collect all model data
        model_results = {}
        for model_name, model_data in results["models"].items():
            if not model_data.get("results"):
                continue
            latest = model_data["results"][-1] if model_data["results"] else None
            if not latest:
                continue
            scores_data = latest.get("scores", latest)
            model_results[model_name] = {
                "timestamp": latest.get("timestamp", "unknown"),
                "suites": scores_data.get("suites", {}),
                "is_base": "base" in model_name.lower(),
            }

        # Overall scores section
        lines.append("### Overall Scores")
        for model_name, data in model_results.items():
            total_passed = 0
            total_tests = 0
            for suite_data in data["suites"].values():
                for test in suite_data.get("tests", []):
                    total_tests += 1
                    if test.get("passed"):
                        total_passed += 1
            overall = (total_passed / total_tests * 100) if total_tests > 0 else 0
            marker = "(BASE)" if data["is_base"] else "(TRAINED)"
            lines.append(f"- {model_name} {marker}: {overall:.1f}% ({total_passed}/{total_tests})")
        lines.append("")

        # Suite breakdown for each model
        lines.append("### Score Breakdown by Category")
        suite_names = set()
        for data in model_results.values():
            suite_names.update(data["suites"].keys())

        for suite_key in sorted(suite_names):
            suite_display = suite_key.replace("_", " ").title()
            lines.append(f"\n**{suite_display}:**")
            for model_name, data in model_results.items():
                suite_data = data["suites"].get(suite_key, {})
                tests = suite_data.get("tests", [])
                if tests:
                    passed = sum(1 for t in tests if t.get("passed"))
                    total = len(tests)
                    score = (passed / total * 100) if total > 0 else 0
                    marker = "BASE" if data["is_base"] else "TRAINED"
                    lines.append(f"  - [{marker}] {model_name}: {score:.0f}% ({passed}/{total})")
        lines.append("")

        # DETAILED RESPONSE COMPARISONS - The key new section
        lines.append("### Detailed Response Comparisons (Base vs Trained)")
        lines.append("Below are actual responses to help identify training opportunities:\n")

        # Group responses by question across models
        question_responses = {}  # {question: {model: {response, passed, reason}}}

        for model_name, data in model_results.items():
            for suite_key, suite_data in data["suites"].items():
                for test in suite_data.get("tests", []):
                    q = test.get("q", "")
                    if not q:
                        continue
                    if q not in question_responses:
                        question_responses[q] = {
                            "suite": suite_key,
                            "expected": test.get("expected", ""),
                            "models": {}
                        }
                    question_responses[q]["models"][model_name] = {
                        "response": test.get("response", "")[:500],
                        "passed": test.get("passed", False),
                        "reason": test.get("reason", ""),
                        "is_base": data["is_base"],
                    }

        # Find interesting comparisons: where trained differs from base
        interesting_cases = []
        for question, qdata in question_responses.items():
            models = qdata["models"]
            base_results = [(m, d) for m, d in models.items() if d["is_base"]]
            trained_results = [(m, d) for m, d in models.items() if not d["is_base"]]

            # Case 1: Trained passed but base failed (success!)
            for tm, td in trained_results:
                for bm, bd in base_results:
                    if td["passed"] and not bd["passed"]:
                        interesting_cases.append({
                            "type": "IMPROVEMENT",
                            "suite": qdata["suite"],
                            "question": question,
                            "expected": qdata["expected"],
                            "base_model": bm,
                            "base_response": bd["response"],
                            "trained_model": tm,
                            "trained_response": td["response"],
                        })

            # Case 2: Trained failed but base passed (regression!)
            for tm, td in trained_results:
                for bm, bd in base_results:
                    if not td["passed"] and bd["passed"]:
                        interesting_cases.append({
                            "type": "REGRESSION",
                            "suite": qdata["suite"],
                            "question": question,
                            "expected": qdata["expected"],
                            "base_model": bm,
                            "base_response": bd["response"],
                            "trained_model": tm,
                            "trained_response": td["response"],
                            "reason": td["reason"],
                        })

            # Case 3: Both failed but different responses (potential insight)
            for tm, td in trained_results:
                for bm, bd in base_results:
                    if not td["passed"] and not bd["passed"] and td["response"] != bd["response"]:
                        interesting_cases.append({
                            "type": "BOTH_FAILED",
                            "suite": qdata["suite"],
                            "question": question,
                            "expected": qdata["expected"],
                            "base_model": bm,
                            "base_response": bd["response"],
                            "trained_model": tm,
                            "trained_response": td["response"],
                            "reason": td["reason"],
                        })

        # Show improvements (what training did well)
        improvements = [c for c in interesting_cases if c["type"] == "IMPROVEMENT"][:5]
        if improvements:
            lines.append("#### ✅ Training Improvements (trained passed, base failed)")
            for case in improvements:
                lines.append(f"\n**[{case['suite']}] Q: {case['question'][:100]}**")
                lines.append(f"Expected: {case['expected']}")
                lines.append(f"BASE ({case['base_model'][:30]}): {case['base_response'][:200]}...")
                lines.append(f"TRAINED ({case['trained_model'][:30]}): {case['trained_response'][:200]}...")
            lines.append("")

        # Show regressions (what training made worse)
        regressions = [c for c in interesting_cases if c["type"] == "REGRESSION"][:5]
        if regressions:
            lines.append("#### ⚠️ Training Regressions (base passed, trained failed)")
            for case in regressions:
                lines.append(f"\n**[{case['suite']}] Q: {case['question'][:100]}**")
                lines.append(f"Expected: {case['expected']}")
                lines.append(f"BASE ({case['base_model'][:30]}): {case['base_response'][:200]}...")
                lines.append(f"TRAINED ({case['trained_model'][:30]}): {case['trained_response'][:200]}...")
                lines.append(f"Failure reason: {case['reason']}")
            lines.append("")

        # Show persistent failures (training opportunity)
        both_failed = [c for c in interesting_cases if c["type"] == "BOTH_FAILED"][:5]
        if both_failed:
            lines.append("#### 🎯 Persistent Failures (both failed - training opportunity)")
            for case in both_failed:
                lines.append(f"\n**[{case['suite']}] Q: {case['question'][:100]}**")
                lines.append(f"Expected: {case['expected']}")
                lines.append(f"BASE ({case['base_model'][:30]}): {case['base_response'][:200]}...")
                lines.append(f"TRAINED ({case['trained_model'][:30]}): {case['trained_response'][:200]}...")
                lines.append(f"Failure reason: {case['reason']}")
            lines.append("")

        # Summary statistics
        lines.append("#### Summary of Comparisons")
        lines.append(f"- Total improvements (trained > base): {len([c for c in interesting_cases if c['type'] == 'IMPROVEMENT'])}")
        lines.append(f"- Total regressions (base > trained): {len([c for c in interesting_cases if c['type'] == 'REGRESSION'])}")
        lines.append(f"- Persistent failures (both failed): {len([c for c in interesting_cases if c['type'] == 'BOTH_FAILED'])}")

        return "\n".join(lines)

    def get_training_data_samples() -> str:
        """Get samples from training data for context."""
        samples = []

        # Check for training data directories
        training_dir = Path(__file__).parent / "training_data"
        if not training_dir.exists():
            training_dir = Path(__file__).parent.parent / "backend" / "training" / "training_data"

        if training_dir.exists():
            for data_dir in training_dir.iterdir():
                if data_dir.is_dir():
                    train_file = data_dir / "train.jsonl"
                    if train_file.exists():
                        try:
                            with open(train_file) as f:
                                lines = f.readlines()[:3]  # First 3 examples
                                for line in lines:
                                    try:
                                        example = json.loads(line)
                                        samples.append({
                                            "recipe": data_dir.name,
                                            "example": example
                                        })
                                    except (json.JSONDecodeError, ValueError):
                                        pass
                        except (IOError, OSError) as e:
                            print(f"[Warning] Failed to read training data dir: {e}")

        if not samples:
            return "No training data samples available."

        lines = ["## Training Data Samples\n"]
        for sample in samples[:10]:
            lines.append(f"**Recipe: {sample['recipe']}**")
            ex = sample['example']
            if 'prompt' in ex and 'chosen' in ex:
                lines.append(f"  Prompt: {ex['prompt'][:100]}...")
                lines.append(f"  Chosen: {ex['chosen'][:150]}...")
                if 'rejected' in ex:
                    lines.append(f"  Rejected: {ex['rejected'][:100]}...")
            elif 'text' in ex:
                lines.append(f"  Text: {ex['text'][:200]}...")
            lines.append("")

        return "\n".join(lines)

    @app.post("/assistant/analyze")
    async def analyze_with_assistant(request: AssistantAnalyzeRequest):
        """Have Claude analyze benchmark results and suggest improvements."""
        api_key = settings_manager.get_anthropic_api_key()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Anthropic API key not configured. Add it in Settings."
            )

        try:
            import httpx

            # Gather all context
            element_catalog = get_element_catalog_summary()
            benchmark_summary = get_benchmark_summary()
            training_samples = get_training_data_samples()

            # Build the system prompt
            system_prompt = """You are a Research Assistant embedded in the TCE (Thought Chemistry Engine) - a platform for training AI models with specific cognitive behaviors.

Your role is to analyze benchmark results, understand patterns in model performance, and suggest improvements through:
1. Identifying which cognitive elements and isotopes are working well
2. Finding gaps where models are underperforming
3. Proposing new training recipes that could improve specific weaknesses
4. Explaining the theoretical basis for your suggestions

The TCE framework uses a "chemistry" metaphor:
- **Elements** are cognitive categories (like Soliton for self-awareness, Skeptic for premise questioning)
- **Isotopes** are specific variants of elements (like skeptic_premise, skeptic_method)
- **Recipes** are training configurations that teach models specific behaviors
- **Benchmarks** measure truthfulness, calibration, myth rejection, etc.

When proposing recipes, be specific:
- Name the recipe clearly (e.g., "calibration_boost", "myth_buster_v2")
- Specify which elements/isotopes to include
- Explain the rationale based on benchmark gaps
- Estimate the expected impact

Current context about the TCE system:

""" + element_catalog + "\n\n" + benchmark_summary + "\n\n" + training_samples

            # Build messages
            messages = []

            # Add conversation history if provided
            if request.conversation_history:
                for msg in request.conversation_history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # Add the current request
            user_content = request.question if request.question else """Please analyze the current state of the TCE system:

1. **Benchmark Analysis**: What patterns do you see in the benchmark results? Which categories are strong vs weak?

2. **Element Effectiveness**: Based on the results, which elements/isotopes seem to be working? Which might need more training?

3. **Gap Analysis**: Where are the biggest opportunities for improvement?

4. **Recipe Proposals**: Suggest 2-3 specific training recipes that could address the identified gaps. For each:
   - Recipe ID and name
   - Which elements/isotopes to focus on
   - What training approach to use (SFT, DPO, or both)
   - Expected impact on specific benchmark categories
   - Rationale for why this should work

5. **Questions**: What additional information would help you make better recommendations?"""

            if request.focus:
                user_content = f"Focus area: {request.focus}\n\n" + user_content

            messages.append({"role": "user", "content": user_content})

            # Call Claude API
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 4096,
                        "system": system_prompt,
                        "messages": messages,
                    }
                )

                if response.status_code != 200:
                    error_detail = response.text
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Anthropic API error: {error_detail}"
                    )

                result = response.json()
                assistant_message = result["content"][0]["text"]

                return {
                    "analysis": assistant_message,
                    "model_used": "claude-sonnet-4-20250514",
                    "context_included": {
                        "elements": True,
                        "benchmarks": True,
                        "training_samples": True,
                    },
                    "conversation_id": result.get("id"),
                }

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request to Claude API timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @app.get("/assistant/status")
    async def get_assistant_status():
        """Check if the Research Assistant is configured and ready."""
        api_key = settings_manager.get_anthropic_api_key()
        return {
            "configured": bool(api_key),
            "api_key_preview": f"{api_key[:8]}...{api_key[-4:]}" if api_key and len(api_key) > 12 else None,
        }

    class TrainingDataGenerateRequest(BaseModel):
        recipe_id: str  # e.g., "calibration_precision_v2"
        training_type: str  # "sft" or "dpo"
        target_isotopes: List[str]  # e.g., ["calibrator_probability", "limiter_factual"]
        num_examples: int = 20  # Number of examples to generate
        focus_description: Optional[str] = None  # Optional guidance on what to focus on
        benchmark_gaps: Optional[List[str]] = None  # Specific benchmark categories to address

    @app.post("/assistant/generate-training-data")
    async def generate_training_data(request: TrainingDataGenerateRequest):
        """Have Claude generate DPO or SFT training data for specific isotopes."""
        api_key = settings_manager.get_anthropic_api_key()
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Anthropic API key not configured. Add it in Settings."
            )

        try:
            import httpx

            # Get element/isotope definitions for context
            element_catalog = get_element_catalog_summary()
            training_samples = get_training_data_samples()
            benchmark_summary = get_benchmark_summary()

            # Build specialized prompt for training data generation
            if request.training_type.lower() == "dpo":
                format_instructions = """Generate DPO (Direct Preference Optimization) training pairs.

Each example must be a JSON object with exactly these fields:
- "prompt": The user question or scenario
- "chosen": The preferred response (demonstrates the target isotope behavior correctly)
- "rejected": The dispreferred response (lacks the isotope behavior or does it wrong)

The "chosen" response should clearly demonstrate the cognitive pattern we want to reinforce.
The "rejected" response should be plausible but missing the key cognitive behavior.

Example DPO format:
{"prompt": "Is it true that we only use 10% of our brains?", "chosen": "No, that's a myth. Brain imaging studies show we use virtually all parts of our brain - different regions activate for different tasks, but no area is dormant.", "rejected": "That's an interesting question. Some scientists believe we only use 10% while others disagree. The research is still ongoing."}
"""
            else:  # SFT
                format_instructions = """Generate SFT (Supervised Fine-Tuning) training examples.

Each example must be a JSON object with exactly these fields:
- "prompt": The user question or scenario
- "completion": The ideal response demonstrating the target isotope behavior

The completion should clearly exemplify the cognitive pattern we want to train.

Example SFT format:
{"prompt": "What causes rainbows?", "completion": "Rainbows form when sunlight refracts through water droplets in the atmosphere. The light separates into its component wavelengths, creating the visible spectrum of colors from red to violet."}
"""

            isotope_focus = ", ".join(request.target_isotopes)
            benchmark_focus = ", ".join(request.benchmark_gaps) if request.benchmark_gaps else "general improvement"

            system_prompt = f"""You are a training data generator for the TCE (Thought Chemistry Engine) system.

Your task is to generate high-quality {request.training_type.upper()} training data that will teach models specific cognitive behaviors.

TARGET ISOTOPES: {isotope_focus}
BENCHMARK GAPS TO ADDRESS: {benchmark_focus}
{f"SPECIFIC FOCUS: {request.focus_description}" if request.focus_description else ""}

{format_instructions}

IMPORTANT GUIDELINES:
1. Each example should clearly target the specified isotopes
2. Vary the topics and difficulty levels
3. Make examples realistic and natural
4. For DPO: The rejected response should be subtly wrong, not obviously bad
5. Ensure the cognitive pattern is learnable from the examples
6. Include a mix of:
   - Factual questions where the isotope behavior helps
   - Scenarios where the isotope prevents errors
   - Edge cases that test the isotope boundaries

CONTEXT - Element/Isotope Definitions:
{element_catalog}

CONTEXT - Existing Training Samples (for style reference):
{training_samples}

CONTEXT - Current Benchmark Results (to understand gaps):
{benchmark_summary[:3000]}

Generate exactly {request.num_examples} training examples. Output ONLY valid JSON objects, one per line (JSONL format). No explanations or markdown - just the JSON objects."""

            user_message = f"""Generate {request.num_examples} {request.training_type.upper()} training examples for the '{request.recipe_id}' recipe.

Target isotopes: {isotope_focus}
Focus on addressing these benchmark gaps: {benchmark_focus}
{f"Additional guidance: {request.focus_description}" if request.focus_description else ""}

Output {request.num_examples} JSON objects, one per line. No other text."""

            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 8192,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_message}],
                    }
                )

                if response.status_code != 200:
                    error_detail = response.text
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Anthropic API error: {error_detail}"
                    )

                result = response.json()
                generated_text = result["content"][0]["text"]

                # Parse the generated JSONL
                examples = []
                errors = []
                for i, line in enumerate(generated_text.strip().split("\n")):
                    line = line.strip()
                    if not line:
                        continue
                    # Skip markdown code blocks if present
                    if line.startswith("```") or line.startswith("//"):
                        continue
                    try:
                        example = json.loads(line)
                        # Validate structure
                        if request.training_type.lower() == "dpo":
                            if all(k in example for k in ["prompt", "chosen", "rejected"]):
                                examples.append(example)
                            else:
                                errors.append(f"Line {i+1}: Missing required DPO fields")
                        else:
                            if all(k in example for k in ["prompt", "completion"]):
                                examples.append(example)
                            else:
                                errors.append(f"Line {i+1}: Missing required SFT fields")
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: Invalid JSON - {str(e)[:50]}")

                return {
                    "success": True,
                    "recipe_id": request.recipe_id,
                    "training_type": request.training_type,
                    "target_isotopes": request.target_isotopes,
                    "examples": examples,
                    "num_generated": len(examples),
                    "num_requested": request.num_examples,
                    "parse_errors": errors if errors else None,
                }

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request to Claude API timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training data generation failed: {str(e)}")

    class SaveTrainingDataRequest(BaseModel):
        recipe_id: str
        training_type: str  # "sft" or "dpo"
        examples: List[Dict]
        append: bool = False  # If True, append to existing file; if False, overwrite

    @app.post("/assistant/save-training-data")
    async def save_training_data(request: SaveTrainingDataRequest):
        """Save generated training data to the training_data directory."""
        try:
            # Determine the directory
            training_dir = Path(__file__).parent / "training_data" / request.recipe_id
            training_dir.mkdir(parents=True, exist_ok=True)

            train_file = training_dir / "train.jsonl"

            # Load existing data if appending
            existing = []
            if request.append and train_file.exists():
                with open(train_file) as f:
                    for line in f:
                        try:
                            existing.append(json.loads(line))
                        except (json.JSONDecodeError, ValueError):
                            pass

            # Combine and write
            all_examples = existing + request.examples

            with open(train_file, "w") as f:
                for example in all_examples:
                    f.write(json.dumps(example) + "\n")

            # Also save metadata
            meta_file = training_dir / "metadata.json"
            metadata = {
                "recipe_id": request.recipe_id,
                "training_type": request.training_type,
                "num_examples": len(all_examples),
                "last_updated": datetime.now().isoformat(),
                "generated_by": "research_assistant",
            }
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)

            return {
                "success": True,
                "path": str(train_file),
                "num_examples": len(all_examples),
                "appended": request.append,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save training data: {str(e)}")

    @app.get("/assistant/training-data/{recipe_id}")
    async def get_training_data(recipe_id: str):
        """Get existing training data for a recipe."""
        training_dir = Path(__file__).parent / "training_data" / recipe_id
        train_file = training_dir / "train.jsonl"

        if not train_file.exists():
            return {"exists": False, "examples": [], "num_examples": 0}

        examples = []
        with open(train_file) as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass

        # Load metadata if exists
        meta_file = training_dir / "metadata.json"
        metadata = None
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Warning] Failed to load metadata {meta_file}: {e}")

        return {
            "exists": True,
            "examples": examples,
            "num_examples": len(examples),
            "metadata": metadata,
        }


    # ============================================================
    # Isotope Training Data Loading (Uses ELEMENTS Registry)
    # ============================================================

    def load_training_file(file_path: Path, max_examples: int = 100) -> List[Dict]:
        """Load training data from a JSONL file."""
        examples = []
        if not file_path.exists():
            return examples

        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                try:
                    examples.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
        return examples

    class IsotopeTrainingDataRequest(BaseModel):
        isotopes: List[str]
        training_type: str = "both"  # "sft", "dpo", or "both"
        max_per_isotope: int = 50

    @app.post("/training/isotope-data")
    async def get_isotope_training_data(request: IsotopeTrainingDataRequest):
        """Get combined training data for a list of isotopes using ELEMENTS registry."""
        base_dir = Path(__file__).parent / "training_data"
        backend_dir = Path(__file__).parent.parent / "backend" / "training" / "training_data"

        sft_data = []
        dpo_data = []
        loaded_isotopes = []
        missing_isotopes = []
        loaded_paths = set()

        for isotope_id in request.isotopes:
            # Look up isotope in ELEMENTS registry
            iso_info = get_isotope_info(isotope_id)

            if not iso_info or 'training_data' not in iso_info:
                missing_isotopes.append(isotope_id)
                continue

            loaded_isotopes.append(isotope_id)
            td = iso_info['training_data']

            # Load SFT data
            if request.training_type in ["sft", "both"] and td.get('sft_path'):
                sft_rel_path = td['sft_path']
                # Avoid loading same path multiple times
                if sft_rel_path not in loaded_paths:
                    loaded_paths.add(sft_rel_path)

                    # Try direct path or with /train.jsonl
                    for search_dir in [backend_dir, base_dir]:
                        sft_path = search_dir / sft_rel_path / "train.jsonl"
                        if not sft_path.exists():
                            sft_path = search_dir / sft_rel_path
                            if sft_path.is_file():
                                pass  # Use as-is
                            else:
                                continue

                        if sft_path.exists() and sft_path.is_file():
                            examples = load_training_file(sft_path, request.max_per_isotope)
                            for ex in examples:
                                ex['_isotope'] = isotope_id
                                ex['_element'] = iso_info.get('element', '')
                            sft_data.extend(examples)
                            break

            # Load DPO data
            if request.training_type in ["dpo", "both"] and td.get('dpo_path'):
                dpo_rel_path = td['dpo_path']
                dpo_path_key = f"dpo:{dpo_rel_path}"
                if dpo_path_key not in loaded_paths:
                    loaded_paths.add(dpo_path_key)

                    for search_dir in [backend_dir, base_dir]:
                        dpo_path = search_dir / dpo_rel_path / "train.jsonl"
                        if not dpo_path.exists():
                            dpo_path = search_dir / dpo_rel_path
                            if dpo_path.is_file():
                                pass
                            else:
                                continue

                        if dpo_path.exists() and dpo_path.is_file():
                            examples = load_training_file(dpo_path, request.max_per_isotope)
                            for ex in examples:
                                ex['_isotope'] = isotope_id
                                ex['_element'] = iso_info.get('element', '')
                            dpo_data.extend(examples)
                            break

        return {
            "sft_data": sft_data,
            "dpo_data": dpo_data,
            "sft_count": len(sft_data),
            "dpo_count": len(dpo_data),
            "loaded_isotopes": loaded_isotopes,
            "missing_isotopes": missing_isotopes,
            "total_examples": len(sft_data) + len(dpo_data),
        }

    @app.get("/training/available-data")
    async def get_available_training_data():
        """List all available training data directories and their contents."""
        base_dir = Path(__file__).parent / "training_data"
        backend_dir = Path(__file__).parent.parent / "backend" / "training" / "training_data"

        available = {}

        for search_dir in [base_dir, backend_dir]:
            if not search_dir.exists():
                continue

            for item in search_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    dir_name = item.name
                    if dir_name not in available:
                        available[dir_name] = {
                            'path': str(item),
                            'has_sft': False,
                            'has_dpo': False,
                            'sft_count': 0,
                            'dpo_count': 0,
                        }

                    # Check for train.jsonl directly or in sft/dpo subdirs
                    train_file = item / "train.jsonl"
                    sft_dir = item / "sft"
                    dpo_dir = item / "dpo"

                    if train_file.exists():
                        with open(train_file) as f:
                            count = sum(1 for _ in f)
                        # Determine if SFT or DPO based on content
                        with open(train_file) as f:
                            first_line = f.readline()
                            try:
                                example = json.loads(first_line)
                                if 'chosen' in example and 'rejected' in example:
                                    available[dir_name]['has_dpo'] = True
                                    available[dir_name]['dpo_count'] = count
                                else:
                                    available[dir_name]['has_sft'] = True
                                    available[dir_name]['sft_count'] = count
                            except (json.JSONDecodeError, IOError, ValueError):
                                pass

                    if sft_dir.exists():
                        sft_train = sft_dir / "train.jsonl"
                        if sft_train.exists():
                            available[dir_name]['has_sft'] = True
                            with open(sft_train) as f:
                                available[dir_name]['sft_count'] = sum(1 for _ in f)

                    if dpo_dir.exists():
                        dpo_train = dpo_dir / "train.jsonl"
                        if dpo_train.exists():
                            available[dir_name]['has_dpo'] = True
                            with open(dpo_train) as f:
                                available[dir_name]['dpo_count'] = sum(1 for _ in f)

        # Build isotope mapping from ELEMENTS registry
        isotope_mapping = {}
        for elem_key, elem in ELEMENTS.items():
            for iso_key, iso in elem.get('isotopes', {}).items():
                isotope_mapping[iso['id']] = iso.get('description', '')

        return {
            "directories": available,
            "isotope_mapping": isotope_mapping,
        }


    # ============================================================
    # ELEMENTS API - Serve Element Registry with Training Stats
    # ============================================================

    def count_training_examples(rel_path: str) -> int:
        """Count training examples in a path."""
        base_dir = Path(__file__).parent / "training_data"
        backend_dir = Path(__file__).parent.parent / "backend" / "training" / "training_data"

        for search_dir in [backend_dir, base_dir]:
            train_file = search_dir / rel_path / "train.jsonl"
            if train_file.exists():
                with open(train_file) as f:
                    return sum(1 for _ in f)
            # Try direct file
            direct_file = search_dir / rel_path
            if direct_file.exists() and direct_file.is_file():
                with open(direct_file) as f:
                    return sum(1 for _ in f)
        return 0

    @app.get("/elements")
    async def get_elements():
        """Get all elements with their isotopes and training data stats."""
        result = {}

        for elem_key, elem in ELEMENTS.items():
            elem_data = {
                'symbol': elem['symbol'],
                'name': elem['name'],
                'group': elem['group'],
                'description': elem['description'],
                'triggers': elem.get('triggers', []),
                'antipatterns': elem.get('antipatterns', []),
                'isotopes': {},
                'total_sft': 0,
                'total_dpo': 0,
            }

            for iso_key, iso in elem.get('isotopes', {}).items():
                td = iso.get('training_data', {})
                sft_count = count_training_examples(td.get('sft_path', '')) if td.get('sft_path') else 0
                dpo_count = count_training_examples(td.get('dpo_path', '')) if td.get('dpo_path') else 0

                elem_data['isotopes'][iso_key] = {
                    'id': iso['id'],
                    'symbol': iso.get('symbol', ''),
                    'name': iso.get('name', ''),
                    'description': iso.get('description', ''),
                    'markers': iso.get('markers', []),
                    'effectiveness': iso.get('effectiveness', {}),
                    'training_data': {
                        'sft_path': td.get('sft_path'),
                        'dpo_path': td.get('dpo_path'),
                        'sft_count': sft_count,
                        'dpo_count': dpo_count,
                    },
                }
                elem_data['total_sft'] += sft_count
                elem_data['total_dpo'] += dpo_count

            result[elem_key] = elem_data

        return {
            "elements": result,
            "groups": ELEMENT_GROUPS,
            "total_elements": len(result),
            "total_isotopes": sum(len(e.get('isotopes', {})) for e in ELEMENTS.values()),
        }

    @app.get("/elements/{element_key}")
    async def get_element(element_key: str):
        """Get a specific element with full details."""
        if element_key not in ELEMENTS:
            raise HTTPException(status_code=404, detail=f"Element not found: {element_key}")

        elem = ELEMENTS[element_key]
        isotopes_with_stats = {}

        for iso_key, iso in elem.get('isotopes', {}).items():
            td = iso.get('training_data', {})
            sft_count = count_training_examples(td.get('sft_path', '')) if td.get('sft_path') else 0
            dpo_count = count_training_examples(td.get('dpo_path', '')) if td.get('dpo_path') else 0

            isotopes_with_stats[iso_key] = {
                **iso,
                'training_stats': {
                    'sft_count': sft_count,
                    'dpo_count': dpo_count,
                    'total': sft_count + dpo_count,
                },
            }

        return {
            **elem,
            'isotopes': isotopes_with_stats,
            'group_info': ELEMENT_GROUPS.get(elem['group'], {}),
        }

    @app.get("/isotopes")
    async def get_all_isotopes():
        """Get all isotopes across all elements with training stats."""
        isotopes = []

        for elem_key, elem in ELEMENTS.items():
            for iso_key, iso in elem.get('isotopes', {}).items():
                td = iso.get('training_data', {})
                sft_count = count_training_examples(td.get('sft_path', '')) if td.get('sft_path') else 0
                dpo_count = count_training_examples(td.get('dpo_path', '')) if td.get('dpo_path') else 0

                isotopes.append({
                    'id': iso['id'],
                    'element': elem_key,
                    'element_name': elem['name'],
                    'group': elem['group'],
                    'symbol': iso.get('symbol', ''),
                    'name': iso.get('name', ''),
                    'description': iso.get('description', ''),
                    'markers': iso.get('markers', []),
                    'effectiveness': iso.get('effectiveness', {}),
                    'sft_count': sft_count,
                    'dpo_count': dpo_count,
                    'total_examples': sft_count + dpo_count,
                    'has_training_data': (sft_count + dpo_count) > 0,
                })

        # Sort by total examples (most data first)
        isotopes.sort(key=lambda x: x['total_examples'], reverse=True)

        return {
            "isotopes": isotopes,
            "total": len(isotopes),
            "with_data": sum(1 for i in isotopes if i['has_training_data']),
            "without_data": sum(1 for i in isotopes if not i['has_training_data']),
        }

    @app.get("/isotope/{isotope_id}")
    async def get_isotope(isotope_id: str):
        """Get a specific isotope by ID with full details and sample training data."""
        iso_info = get_isotope_info(isotope_id)
        if not iso_info:
            raise HTTPException(status_code=404, detail=f"Isotope not found: {isotope_id}")

        # Get training stats and sample data
        td = iso_info.get('training_data', {})
        backend_dir = Path(__file__).parent.parent / "backend" / "training" / "training_data"

        sft_samples = []
        dpo_samples = []

        if td.get('sft_path'):
            sft_path = backend_dir / td['sft_path'] / "train.jsonl"
            if sft_path.exists():
                sft_samples = load_training_file(sft_path, 3)

        if td.get('dpo_path'):
            dpo_path = backend_dir / td['dpo_path'] / "train.jsonl"
            if dpo_path.exists():
                dpo_samples = load_training_file(dpo_path, 3)

        return {
            **iso_info,
            'sft_samples': sft_samples,
            'dpo_samples': dpo_samples,
        }


    # ============================================================
    # Cognitive X-Ray Analysis - Mode Discrimination & Confabulation
    # ============================================================

    class CognitiveAnalysisRequest(BaseModel):
        """Request model for cognitive analysis."""
        prompt: str
        response: str
        threshold: float = 0.3

    @app.post("/cognitive/analyze")
    async def analyze_cognitive_response(request: CognitiveAnalysisRequest):
        """
        Comprehensive cognitive analysis of a prompt-response pair.

        Returns:
        - Mode discrimination analysis (appropriate/inappropriate activations)
        - Confabulation risk assessment
        - Element detections with isotopes
        - Leakage detection
        """
        prompt = request.prompt
        response = request.response
        threshold = request.threshold

        # 1. Classify prompt mode
        mode = classify_prompt_mode(prompt)

        # 2. Detect all elements in response
        detections = detect_all_elements(response, threshold=threshold)
        detected_elements = [d.element_id for d in detections]

        # 3. Validate mode discrimination
        validation = validate_mode_discrimination(prompt, response, threshold=threshold)

        # 4. Compute correct/incorrect activations
        correct_activations = [
            {"element": e, "confidence": next((d.confidence for d in detections if d.element_id == e), 0)}
            for e in mode.appropriate_elements
            if e in detected_elements
        ]

        missed_activations = [
            {"element": e, "expected": True}
            for e in mode.appropriate_elements
            if e not in detected_elements
        ]

        leakage_activations = [
            {"element": d.element_id, "confidence": d.confidence, "isotope": d.isotope_id}
            for d in detections
            if d.element_id in mode.inappropriate_elements
        ]

        correct_silence = [
            e for e in mode.inappropriate_elements
            if e not in detected_elements
        ]

        # 5. Calculate mode discrimination score
        total_checks = len(mode.appropriate_elements) + len(mode.inappropriate_elements)
        if total_checks > 0:
            correct_count = len(correct_activations) + len(correct_silence)
            mode_score = correct_count / total_checks
        else:
            mode_score = 1.0 if len(leakage_activations) == 0 else 0.5

        # 6. Confabulation detection
        confab_detected, confab_markers = detect_confabulation(response)
        refusal_detected, refusal_confidence = detect_proper_refusal(response)

        confab_risk_level = "low"
        if confab_detected:
            if refusal_detected:
                confab_risk_level = "mitigated"
            else:
                confab_risk_level = "high"

        # 7. Leakage detection (epistemic behaviors on simple factuals)
        leakage_result = detect_leakage(response)

        # 8. Build detection details for frontend
        detection_details = []
        for d in detections:
            detection_details.append({
                "element_id": d.element_id,
                "confidence": d.confidence,
                "markers_found": d.markers_found,
                "isotope_id": d.isotope_id,
            })

        # 9. Basic behavior analysis (lightweight, no Observatory dependency)
        import re as _re
        words = response.split()
        word_count = max(len(words), 1)
        resp_lower = response.lower()

        hedging_words = ["maybe", "perhaps", "possibly", "might", "could", "somewhat",
                         "arguably", "likely", "unlikely", "it seems", "in a way",
                         "sort of", "kind of", "to some extent", "i think", "i believe",
                         "not sure", "it depends", "generally", "typically"]
        hedging_count = sum(1 for w in hedging_words if w in resp_lower)
        hedging_density = min(hedging_count / max(word_count * 0.05, 4), 1.0)

        uncertainty_markers = ["i'm not sure", "i don't know", "uncertain", "unclear",
                               "hard to say", "difficult to determine", "it's possible",
                               "there's no clear", "debatable", "open question"]
        uncertainty_count = sum(1 for m in uncertainty_markers if m in resp_lower)
        uncertainty_level = min(uncertainty_count * 0.25, 1.0)

        helpful_markers = ["here's", "here are", "let me", "you can", "try this",
                          "for example", "step", "first,", "to do this", "i recommend",
                          "the solution", "you should", "you could try"]
        helpful_count = sum(1 for m in helpful_markers if m in resp_lower)
        helpfulness = min(helpful_count * 0.2, 1.0)

        sentences = _re.split(r'[.!?]+', response)
        avg_sentence_len = sum(len(s.split()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
        legibility = max(0, min(1.0, 1.0 - (avg_sentence_len - 15) / 30))

        # Determine behavior mode
        has_meta = any(e in detected_elements for e in ["soliton", "reflector", "calibrator"])
        if has_meta:
            behavior_mode = "META_COGNITIVE"
        elif uncertainty_level >= 0.5:
            behavior_mode = "UNCERTAIN"
        elif helpfulness >= 0.3:
            behavior_mode = "HELPFUL"
        elif hedging_density >= 0.6 and helpfulness < 0.2:
            behavior_mode = "EVASIVE"
        elif confab_risk_level == "high":
            behavior_mode = "DEFENSIVE"
        else:
            behavior_mode = "CONFIDENT"

        return {
            # Mode discrimination
            "mode": {
                "prompt_type": mode.prompt_type,
                "appropriate_elements": mode.appropriate_elements,
                "inappropriate_elements": mode.inappropriate_elements,
            },
            "mode_discrimination": {
                "score": mode_score,
                "passed": validation.get("passed", False),
                "correct_activations": correct_activations,
                "missed_activations": missed_activations,
                "leakage_activations": leakage_activations,
                "correct_silence": correct_silence,
            },
            # Confabulation
            "confabulation": {
                "detected": confab_detected,
                "markers_found": confab_markers,
                "properly_refused": refusal_detected,
                "refusal_confidence": refusal_confidence,
                "risk_level": confab_risk_level,
            },
            # Leakage (epistemic on simple factuals)
            "leakage": {
                "detected": leakage_result.leaked,
                "patterns_found": leakage_result.patterns_found,
                "source_elements": leakage_result.source_elements,
                "severity": leakage_result.severity,
            },
            # Behavior analysis
            "behavior_mode": behavior_mode,
            "hedging_density": round(hedging_density, 3),
            "uncertainty_level": round(uncertainty_level, 3),
            "helpfulness": round(helpfulness, 3),
            "legibility": round(legibility, 3),
            # Raw detections
            "detections": detection_details,
            "detected_elements": detected_elements,
        }


    # In-memory detection history for Training-Inference Bridge analysis
    # Maps recipe_id -> list of detection results
    detection_history: Dict[str, List[Dict]] = {}

    @app.post("/cognitive/record-detection")
    async def record_detection(recipe_id: str, detections: List[Dict]):
        """Record detection results for training-inference bridge analysis."""
        if recipe_id not in detection_history:
            detection_history[recipe_id] = []
        detection_history[recipe_id].append({
            "timestamp": datetime.now().isoformat(),
            "detections": detections
        })
        # Keep only last 100 records per recipe
        if len(detection_history[recipe_id]) > 100:
            detection_history[recipe_id] = detection_history[recipe_id][-100:]
        return {"status": "recorded"}

    @app.get("/cognitive/training-bridge/{recipe_id}")
    async def get_training_bridge(recipe_id: str):
        """
        Compute correlation between training isotope doses and detection rates.

        Returns a bridge visualization data structure showing:
        - Training isotopes and their doses from the recipe
        - Detection rates for each isotope from inference history
        - Correlation/effectiveness indicators
        """
        from lib import ELEMENT_CATALOG

        # Get recipe configuration
        recipe_config = RECIPE_ISOTOPE_MAP.get(recipe_id)
        if not recipe_config:
            # Try to find in custom recipes (would need frontend to send this)
            return {
                "error": f"Recipe not found: {recipe_id}",
                "available_recipes": list(RECIPE_ISOTOPE_MAP.keys())
            }

        # Get training isotopes and their weights
        training_isotopes = recipe_config.get('isotopes', [])
        training_weights = recipe_config.get('weights', {})

        # Get detection history for this recipe
        history = detection_history.get(recipe_id, [])

        # Calculate detection rates from history
        detection_rates = {}
        if history:
            total_samples = len(history)
            isotope_detections = {}

            for record in history:
                for d in record.get('detections', []):
                    iso_id = d.get('isotope_id')
                    if iso_id:
                        if iso_id not in isotope_detections:
                            isotope_detections[iso_id] = []
                        isotope_detections[iso_id].append(d.get('confidence', 0))

            for iso_id, confidences in isotope_detections.items():
                detection_rates[iso_id] = {
                    "rate": len(confidences) / total_samples,
                    "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                    "count": len(confidences)
                }

        # Build bridge data
        bridge_data = []
        for iso_id in training_isotopes:
            training_dose = training_weights.get(iso_id, 0)
            detection = detection_rates.get(iso_id, {"rate": 0, "avg_confidence": 0, "count": 0})

            # Determine effectiveness
            if detection["rate"] > 0:
                ratio = detection["rate"] / training_dose if training_dose > 0 else 0
                if ratio > 1.2:
                    effectiveness = "amplified"
                elif ratio > 0.8:
                    effectiveness = "matched"
                else:
                    effectiveness = "suppressed"
            else:
                effectiveness = "undetected"

            # Get element info
            element_id = iso_id.rsplit('_', 1)[0] if '_' in iso_id else iso_id
            element_info = ELEMENT_CATALOG.get(element_id, {})

            bridge_data.append({
                "isotope_id": iso_id,
                "element_id": element_id,
                "element_symbol": element_info.get("symbol", "?"),
                "training_dose": training_dose,
                "detection_rate": detection["rate"],
                "avg_confidence": detection["avg_confidence"],
                "detection_count": detection["count"],
                "effectiveness": effectiveness,
            })

        return {
            "recipe_id": recipe_id,
            "total_samples": len(history),
            "bridge_data": bridge_data,
            "training_isotopes": training_isotopes,
        }


    # Global co-occurrence matrix for isotope interference analysis
    isotope_cooccurrence: Dict[str, Dict[str, int]] = {}
    isotope_occurrence_count: Dict[str, int] = {}

    @app.post("/cognitive/record-cooccurrence")
    async def record_cooccurrence(detections: List[Dict]):
        """Record isotope co-occurrences for interference matrix analysis."""
        # Get all detected isotopes from this sample
        isotope_ids = [d.get('isotope_id') for d in detections if d.get('isotope_id')]

        # Update occurrence counts
        for iso_id in isotope_ids:
            if iso_id not in isotope_occurrence_count:
                isotope_occurrence_count[iso_id] = 0
            isotope_occurrence_count[iso_id] += 1

        # Update co-occurrence counts
        for i, iso1 in enumerate(isotope_ids):
            if iso1 not in isotope_cooccurrence:
                isotope_cooccurrence[iso1] = {}
            for iso2 in isotope_ids[i:]:  # Include self-co-occurrence
                if iso2 not in isotope_cooccurrence[iso1]:
                    isotope_cooccurrence[iso1][iso2] = 0
                isotope_cooccurrence[iso1][iso2] += 1
                # Symmetric update
                if iso1 != iso2:
                    if iso2 not in isotope_cooccurrence:
                        isotope_cooccurrence[iso2] = {}
                    if iso1 not in isotope_cooccurrence[iso2]:
                        isotope_cooccurrence[iso2][iso1] = 0
                    isotope_cooccurrence[iso2][iso1] += 1

        return {"status": "recorded"}

    @app.get("/cognitive/interference-matrix")
    async def get_interference_matrix():
        """
        Get isotope interference matrix showing co-activation patterns.

        Returns a matrix where:
        - Positive values indicate isotopes that co-activate together
        - Values closer to 1.0 indicate strong co-occurrence
        - Lower values indicate weaker relationships
        """
        if not isotope_cooccurrence:
            return {
                "isotopes": [],
                "matrix": [],
                "total_samples": 0,
                "message": "No co-occurrence data yet. Generate some responses first!"
            }

        # Get all isotopes with data
        all_isotopes = sorted(isotope_occurrence_count.keys())

        # Build normalized matrix
        matrix = []
        for iso1 in all_isotopes:
            row = []
            for iso2 in all_isotopes:
                if iso1 == iso2:
                    # Self-correlation is always 1.0
                    row.append(1.0)
                else:
                    # Calculate Jaccard similarity: intersection / union
                    co_count = isotope_cooccurrence.get(iso1, {}).get(iso2, 0)
                    count1 = isotope_occurrence_count.get(iso1, 0)
                    count2 = isotope_occurrence_count.get(iso2, 0)

                    if count1 + count2 - co_count > 0:
                        similarity = co_count / (count1 + count2 - co_count)
                    else:
                        similarity = 0

                    row.append(round(similarity, 3))
            matrix.append(row)

        # Get element info for labeling
        from lib import ELEMENT_CATALOG
        isotope_info = []
        for iso_id in all_isotopes:
            element_id = iso_id.rsplit('_', 1)[0] if '_' in iso_id else iso_id
            element_info = ELEMENT_CATALOG.get(element_id, {})
            isotope_info.append({
                "id": iso_id,
                "element_id": element_id,
                "symbol": element_info.get("symbol", "?"),
                "count": isotope_occurrence_count.get(iso_id, 0)
            })

        return {
            "isotopes": isotope_info,
            "matrix": matrix,
            "total_samples": sum(isotope_occurrence_count.values()) // max(len(all_isotopes), 1),
        }


    # ============================================================
    # Compound Discovery Lab - Isotope Effectiveness Analysis
    # ============================================================

    # Recipe definitions mapping recipe_id to isotope compositions
    # This mirrors the frontend trainingRecipes but in a format useful for analysis
    RECIPE_ISOTOPE_MAP = {
        'soliton_boost': {
            'isotopes': ['soliton_knowledge', 'soliton_process', 'soliton_experience'],
            'weights': {'soliton_knowledge': 0.40, 'soliton_process': 0.30, 'soliton_experience': 0.30},
        },
        'skeptic_balanced': {
            'isotopes': ['skeptic_premise', 'skeptic_source', 'skeptic_stats', 'limiter_factual'],
            'weights': {'skeptic_premise': 0.35, 'skeptic_source': 0.25, 'skeptic_stats': 0.20, 'limiter_factual': 0.20},
        },
        'epistemic_stack': {
            'isotopes': ['soliton_knowledge', 'skeptic_premise', 'calibrator_probability', 'limiter_factual', 'reflector_trace'],
            'weights': {'soliton_knowledge': 0.25, 'skeptic_premise': 0.20, 'calibrator_probability': 0.20, 'limiter_factual': 0.20, 'reflector_trace': 0.15},
        },
        'analyst': {
            'isotopes': ['architect_hierarchy', 'debugger_binary', 'essentialist_mechanism', 'taxonomist_hierarchical', 'skeptic_method'],
            'weights': {'architect_hierarchy': 0.25, 'debugger_binary': 0.20, 'essentialist_mechanism': 0.20, 'taxonomist_hierarchical': 0.20, 'skeptic_method': 0.15},
        },
        'coder': {
            # Optimized for maximum coding output: debugging, architecture, precision, explanation
            'isotopes': [
                'diagnostician_conceptual',  # Identifies real conceptual gaps (mental model vs actual code)
                'debugger_binary',           # Binary search fault isolation
                'debugger_trace',            # Execution tracing
                'causalist_chain',           # Traces causal chains (bug cascades, system failures)
                'essentialist_principle',    # "Maintainable code = intent obvious + changes local"
                'architect_hierarchy',       # System decomposition
                'architect_interface',       # Component boundaries and contracts
                'reflector_monitor',         # "I'm on less firm ground here" - prevents hallucination
                'expositor_structured',      # Step-by-step code explanation
                'limiter_temporal',          # "May have changed since training" - prevents outdated APIs
            ],
            'weights': {
                'diagnostician_conceptual': 0.15,
                'debugger_binary': 0.12,
                'debugger_trace': 0.12,
                'causalist_chain': 0.12,
                'essentialist_principle': 0.10,
                'architect_hierarchy': 0.08,
                'architect_interface': 0.08,
                'reflector_monitor': 0.08,
                'expositor_structured': 0.08,
                'limiter_temporal': 0.07,
            },
        },
        'educator': {
            'isotopes': ['expositor_analogy', 'scaffolder_bridge', 'maieutic_elicit', 'diagnostician_conceptual', 'soliton_knowledge'],
            'weights': {'expositor_analogy': 0.25, 'scaffolder_bridge': 0.20, 'maieutic_elicit': 0.20, 'diagnostician_conceptual': 0.20, 'soliton_knowledge': 0.15},
        },
        'dont_panic': {
            'isotopes': [
                # Epistemic (9)
                'soliton_knowledge', 'soliton_process', 'soliton_experience',
                'reflector_trace', 'reflector_monitor',
                'calibrator_confidence', 'calibrator_probability',
                'limiter_factual', 'limiter_temporal',
                # Analytical (8)
                'architect_hierarchy', 'architect_interface',
                'essentialist_mechanism', 'essentialist_principle',
                'debugger_binary', 'debugger_trace',
                'taxonomist_hierarchical', 'taxonomist_relational',
                # Evaluative (5)
                'skeptic_premise', 'skeptic_method', 'skeptic_source', 'skeptic_stats',
                'critic_constructive',
                # Generative (2)
                'generator_divergent', 'synthesizer_cross_domain',
                # Dialogical (2)
                'steelman_charitable', 'adversary_red_team',
                # Pedagogical (5)
                'expositor_analogy', 'expositor_structured', 'scaffolder_bridge',
                'maieutic_elicit', 'diagnostician_conceptual',
                # Temporal (3)
                'futurist_scenario', 'historian_pattern', 'causalist_chain',
            ],
            'weights': {iso: 1/34 for iso in [
                'soliton_knowledge', 'soliton_process', 'soliton_experience',
                'reflector_trace', 'reflector_monitor',
                'calibrator_confidence', 'calibrator_probability',
                'limiter_factual', 'limiter_temporal',
                'architect_hierarchy', 'architect_interface',
                'essentialist_mechanism', 'essentialist_principle',
                'debugger_binary', 'debugger_trace',
                'taxonomist_hierarchical', 'taxonomist_relational',
                'skeptic_premise', 'skeptic_method', 'skeptic_source', 'skeptic_stats',
                'critic_constructive',
                'generator_divergent', 'synthesizer_cross_domain',
                'steelman_charitable', 'adversary_red_team',
                'expositor_analogy', 'expositor_structured', 'scaffolder_bridge',
                'maieutic_elicit', 'diagnostician_conceptual',
                'futurist_scenario', 'historian_pattern', 'causalist_chain',
            ]},
        },
    }

    # Map model names to their recipe (for benchmark attribution)
    def get_recipe_from_model_name(model_name: str) -> Optional[str]:
        """Extract recipe ID from model name."""
        model_lower = model_name.lower()
        for recipe_id in RECIPE_ISOTOPE_MAP.keys():
            if recipe_id.replace('_', '') in model_lower.replace('_', '').replace('-', ''):
                return recipe_id
        # Check for specific patterns
        if 'soliton' in model_lower:
            return 'soliton_boost'
        if 'analyst' in model_lower:
            return 'analyst'
        if 'skeptic' in model_lower:
            return 'skeptic_balanced'
        return None

    def is_base_model(model_name: str) -> bool:
        """Check if a model is a base (untrained) model."""
        return 'base' in model_name.lower()

    def get_model_size(model_name: str) -> str:
        """Extract model size from name."""
        if '70b' in model_name.lower() or '70B' in model_name:
            return '70B'
        if '3b' in model_name.lower() or '3B' in model_name or '3.2' in model_name:
            return '3B'
        if '8b' in model_name.lower() or '8B' in model_name:
            return '8B'
        return 'unknown'

    def compute_effectiveness_matrix():
        """
        Compute isotope effectiveness by correlating isotope presence with benchmark improvements.
        Returns a matrix of isotope -> benchmark_category -> effectiveness_score

        Falls back to ELEMENTS registry effectiveness data if no benchmark data available.
        """
        results = load_benchmark_results()
        if not results.get("models"):
            # Use ELEMENTS registry effectiveness data as fallback
            return compute_effectiveness_from_elements()
        else:
            # Use actual benchmark data
            return compute_effectiveness_from_benchmarks(results)

    def compute_effectiveness_from_elements():
        """
        Build effectiveness matrix from ELEMENTS registry effectiveness data.
        This provides baseline effectiveness scores from prior research.
        """
        matrix = {}
        all_categories = set()

        for elem_key, elem in ELEMENTS.items():
            for iso_key, iso in elem.get('isotopes', {}).items():
                isotope_id = iso.get('id')
                effectiveness = iso.get('effectiveness', {})

                if effectiveness and isotope_id:
                    matrix[isotope_id] = {}
                    for category, delta in effectiveness.items():
                        # Normalize category names to match expected format
                        cat_normalized = category.lower().replace(' ', '_')
                        all_categories.add(cat_normalized)
                        matrix[isotope_id][cat_normalized] = {
                            "avg_delta": delta,
                            "weighted_avg": delta * 0.3,  # Assume moderate weight
                            "observations": 1,  # From research data
                            "confidence": 0.7,  # Moderate confidence for research-based data
                            "source": "elements_registry"
                        }

        return {
            "matrix": matrix,
            "categories": sorted(list(all_categories)),
            "isotopes": sorted(list(matrix.keys())),
            "num_trained_models": 0,
            "num_base_models": 0,
            "total_observations": len(matrix),
            "source": "elements_registry",
            "note": "Using effectiveness data from ELEMENTS registry (no benchmark data available)"
        }

    def compute_effectiveness_from_benchmarks(results):
        """Compute effectiveness matrix from actual benchmark results."""
        # Collect all benchmark scores by model
        model_scores = {}
        for model_name, model_data in results["models"].items():
            if not model_data.get("results"):
                continue
            latest = model_data["results"][-1]
            scores_data = latest.get("scores", latest)
            suites = scores_data.get("suites", {})

            model_scores[model_name] = {
                "is_base": is_base_model(model_name),
                "recipe": get_recipe_from_model_name(model_name),
                "size": get_model_size(model_name),
                "categories": {}
            }

            for suite_key, suite_data in suites.items():
                tests = suite_data.get("tests", [])
                if tests:
                    passed = sum(1 for t in tests if t.get("passed"))
                    total = len(tests)
                    score = (passed / total * 100) if total > 0 else 0
                    model_scores[model_name]["categories"][suite_key] = {
                        "score": score,
                        "passed": passed,
                        "total": total
                    }

        # Find base models for comparison
        base_models = {m: s for m, s in model_scores.items() if s["is_base"]}
        trained_models = {m: s for m, s in model_scores.items() if not s["is_base"]}

        # Calculate deltas: trained vs base (matching size)
        deltas = []
        for trained_name, trained_data in trained_models.items():
            trained_size = trained_data["size"]
            recipe = trained_data["recipe"]
            if not recipe:
                continue

            # Find matching base model
            matching_base = None
            for base_name, base_data in base_models.items():
                if base_data["size"] == trained_size:
                    matching_base = base_data
                    break

            if not matching_base:
                continue

            # Calculate delta for each category
            for cat, cat_data in trained_data["categories"].items():
                base_score = matching_base["categories"].get(cat, {}).get("score", 0)
                delta = cat_data["score"] - base_score
                deltas.append({
                    "model": trained_name,
                    "recipe": recipe,
                    "isotopes": RECIPE_ISOTOPE_MAP.get(recipe, {}).get("isotopes", []),
                    "weights": RECIPE_ISOTOPE_MAP.get(recipe, {}).get("weights", {}),
                    "category": cat,
                    "trained_score": cat_data["score"],
                    "base_score": base_score,
                    "delta": delta,
                    "size": trained_size,
                })

        # Build effectiveness matrix: isotope -> category -> list of deltas
        effectiveness = {}
        all_categories = set()

        for d in deltas:
            for isotope in d["isotopes"]:
                if isotope not in effectiveness:
                    effectiveness[isotope] = {}
                cat = d["category"]
                all_categories.add(cat)
                if cat not in effectiveness[isotope]:
                    effectiveness[isotope][cat] = []
                # Weight the delta by isotope weight in recipe
                weight = d["weights"].get(isotope, 0.2)
                effectiveness[isotope][cat].append({
                    "delta": d["delta"],
                    "weight": weight,
                    "weighted_delta": d["delta"] * weight,
                    "model": d["model"],
                    "size": d["size"],
                })

        # Aggregate to average effectiveness
        matrix = {}
        for isotope, cats in effectiveness.items():
            matrix[isotope] = {}
            for cat, observations in cats.items():
                if observations:
                    avg_delta = sum(o["delta"] for o in observations) / len(observations)
                    weighted_avg = sum(o["weighted_delta"] for o in observations) / len(observations)
                    matrix[isotope][cat] = {
                        "avg_delta": round(avg_delta, 2),
                        "weighted_avg": round(weighted_avg, 2),
                        "observations": len(observations),
                        "confidence": min(len(observations) / 3, 1.0),  # More observations = higher confidence
                    }

        return {
            "matrix": matrix,
            "categories": sorted(list(all_categories)),
            "isotopes": sorted(list(effectiveness.keys())),
            "num_trained_models": len(trained_models),
            "num_base_models": len(base_models),
            "total_observations": len(deltas),
        }

    def recommend_compound(target_categories: List[str], avoid_regression: List[str] = None, max_isotopes: int = 5):
        """
        Recommend isotope combination for target benchmark improvements.
        """
        matrix_data = compute_effectiveness_matrix()
        if "error" in matrix_data:
            return matrix_data

        matrix = matrix_data["matrix"]
        avoid = avoid_regression or []

        # Score each isotope
        isotope_scores = {}
        for isotope, cats in matrix.items():
            score = 0
            penalty = 0
            details = {}

            # Add points for improving target categories
            for target in target_categories:
                if target in cats:
                    delta = cats[target]["avg_delta"]
                    conf = cats[target]["confidence"]
                    contribution = delta * conf
                    score += contribution
                    details[target] = {"delta": delta, "confidence": conf, "contribution": contribution}

            # Subtract points for regressing avoid categories
            for avoid_cat in avoid:
                if avoid_cat in cats:
                    delta = cats[avoid_cat]["avg_delta"]
                    if delta < 0:  # It's causing regression
                        penalty += abs(delta) * cats[avoid_cat]["confidence"]
                        details[f"avoid_{avoid_cat}"] = {"delta": delta, "penalty": abs(delta)}

            isotope_scores[isotope] = {
                "score": round(score - penalty, 2),
                "raw_score": round(score, 2),
                "penalty": round(penalty, 2),
                "details": details,
            }

        # Sort by score
        ranked = sorted(isotope_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        # Select top isotopes
        recommended = []
        for isotope, data in ranked[:max_isotopes]:
            if data["score"] > 0:  # Only include positive contributors
                recommended.append({
                    "isotope": isotope,
                    **data
                })

        # Calculate predicted impact
        predicted_impact = {}
        for target in target_categories:
            total_delta = sum(
                matrix.get(r["isotope"], {}).get(target, {}).get("avg_delta", 0)
                for r in recommended
            )
            predicted_impact[target] = round(total_delta, 1)

        return {
            "recommended_isotopes": recommended,
            "predicted_impact": predicted_impact,
            "target_categories": target_categories,
            "avoid_regression": avoid,
            "all_scores": {k: v["score"] for k, v in ranked},
        }

    @app.get("/discovery/effectiveness-matrix")
    async def get_effectiveness_matrix():
        """Get the isotope effectiveness matrix based on benchmark data."""
        return compute_effectiveness_matrix()

    @app.post("/discovery/recommend-compound")
    async def recommend_compound_endpoint(
        target_categories: List[str],
        avoid_regression: Optional[List[str]] = None,
        max_isotopes: int = 5
    ):
        """Recommend isotope combination for target outcomes."""
        return recommend_compound(target_categories, avoid_regression, max_isotopes)

    class CompoundRecommendRequest(BaseModel):
        target_categories: List[str]  # e.g., ["truthfulqa", "calibration"]
        avoid_regression: Optional[List[str]] = None  # e.g., ["factual"]
        max_isotopes: int = 5

    @app.post("/discovery/recommend")
    async def recommend_compound_post(request: CompoundRecommendRequest):
        """Recommend isotope combination for target outcomes (POST version)."""
        return recommend_compound(
            request.target_categories,
            request.avoid_regression,
            request.max_isotopes
        )

    @app.get("/discovery/isotope/{isotope_id}")
    async def get_isotope_effectiveness(isotope_id: str):
        """Get detailed effectiveness data for a specific isotope."""
        matrix_data = compute_effectiveness_matrix()
        if "error" in matrix_data:
            return matrix_data

        matrix = matrix_data["matrix"]
        if isotope_id not in matrix:
            raise HTTPException(status_code=404, detail=f"Isotope not found: {isotope_id}")

        return {
            "isotope": isotope_id,
            "effectiveness": matrix[isotope_id],
            "categories": matrix_data["categories"],
        }

    @app.get("/discovery/category/{category}")
    async def get_category_effectiveness(category: str):
        """Get all isotope effectiveness for a specific benchmark category."""
        matrix_data = compute_effectiveness_matrix()
        if "error" in matrix_data:
            return matrix_data

        matrix = matrix_data["matrix"]
        category_data = {}

        for isotope, cats in matrix.items():
            if category in cats:
                category_data[isotope] = cats[category]

        # Sort by avg_delta descending
        sorted_isotopes = sorted(
            category_data.items(),
            key=lambda x: x[1]["avg_delta"],
            reverse=True
        )

        return {
            "category": category,
            "isotopes": [{"isotope": k, **v} for k, v in sorted_isotopes],
            "best_isotope": sorted_isotopes[0][0] if sorted_isotopes else None,
            "worst_isotope": sorted_isotopes[-1][0] if sorted_isotopes else None,
        }

    @app.get("/discovery/synergies")
    async def get_isotope_synergies():
        """Analyze which isotopes appear together in successful recipes."""
        results = load_benchmark_results()
        if not results.get("models"):
            return {"error": "No benchmark data available"}

        # Find models with above-average performance
        synergies = {}
        co_occurrences = {}

        for model_name, model_data in results["models"].items():
            if is_base_model(model_name):
                continue

            recipe = get_recipe_from_model_name(model_name)
            if not recipe or recipe not in RECIPE_ISOTOPE_MAP:
                continue

            isotopes = RECIPE_ISOTOPE_MAP[recipe]["isotopes"]

            # Calculate overall score
            latest = model_data["results"][-1] if model_data.get("results") else None
            if not latest:
                continue

            scores_data = latest.get("scores", latest)
            suites = scores_data.get("suites", {})
            total_passed = sum(
                sum(1 for t in sd.get("tests", []) if t.get("passed"))
                for sd in suites.values()
            )
            total_tests = sum(
                len(sd.get("tests", []))
                for sd in suites.values()
            )
            overall = (total_passed / total_tests * 100) if total_tests > 0 else 0

            # Track co-occurrences
            for i, iso1 in enumerate(isotopes):
                for iso2 in isotopes[i+1:]:
                    pair = tuple(sorted([iso1, iso2]))
                    if pair not in co_occurrences:
                        co_occurrences[pair] = {"count": 0, "scores": []}
                    co_occurrences[pair]["count"] += 1
                    co_occurrences[pair]["scores"].append(overall)

        # Calculate average scores for pairs
        pair_effectiveness = []
        for pair, data in co_occurrences.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            pair_effectiveness.append({
                "isotope1": pair[0],
                "isotope2": pair[1],
                "co_occurrences": data["count"],
                "avg_score": round(avg_score, 1),
            })

        # Sort by avg_score
        pair_effectiveness.sort(key=lambda x: x["avg_score"], reverse=True)

        return {
            "synergies": pair_effectiveness[:20],  # Top 20 pairs
            "total_pairs_analyzed": len(pair_effectiveness),
        }


    # ============================================================
    # Settings & Model Management Endpoints
    # ============================================================

    @app.get("/settings")
    async def get_settings():
        """Get current settings."""
        return settings_manager.get_settings()


    @app.post("/settings")
    async def update_settings(request: SettingsUpdate):
        """Update settings."""
        settings_manager.update_settings(
            hf_token=request.hf_token,
            models_dir=request.models_dir,
            tinker_api_key=request.tinker_api_key,
            anthropic_api_key=request.anthropic_api_key
        )
        return {"success": True, "settings": settings_manager.get_settings()}


    @app.get("/models/local")
    async def list_local_models():
        """List locally downloaded models."""
        models = settings_manager.list_local_models()
        return {"models": models, "count": len(models)}


    @app.get("/models/search")
    async def search_models(q: str = "", limit: int = 20):
        """Search HuggingFace for MLX models."""
        if not q:
            return {"models": [], "query": q}
        results = await settings_manager.search_models(q, limit)
        return {"models": results, "query": q}


    @app.post("/models/download")
    async def download_model(request: ModelDownloadRequest):
        """Start downloading a model from HuggingFace."""
        result = settings_manager.start_download(request.model_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result


    @app.get("/models/download/{download_id}")
    async def get_download_status(download_id: str):
        """Check download status."""
        return settings_manager.get_download_status(download_id)


    @app.delete("/models/{model_id:path}")
    async def delete_model(model_id: str):
        """Delete a local model."""
        success = settings_manager.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found or could not be deleted")
        return {"success": True, "model_id": model_id}


    # ============================================================
    # Model Loading & Chat Interaction Endpoints
    # ============================================================

    @app.get("/models/loaded")
    async def get_loaded_model():
        """Get currently loaded model status."""
        return model_manager.get_status()

    @app.post("/models/load")
    async def load_model(request: ModelLoadRequest):
        """Load a model into memory for chat."""
        if request.adapter_path:
            resolved = Path(request.adapter_path).resolve()
            allowed_bases = [
                Path(__file__).parent.resolve(),
                (Path(__file__).parent.parent / "backend" / "training").resolve(),
            ]
            if not any(str(resolved).startswith(str(base)) for base in allowed_bases):
                raise HTTPException(status_code=400, detail="Adapter path must be within the project directory")
            if not resolved.exists():
                raise HTTPException(status_code=400, detail=f"Adapter path does not exist: {request.adapter_path}")
        result = model_manager.load(request.model_id, request.adapter_path)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to load model"))
        return result

    @app.post("/models/unload")
    async def unload_model():
        """Unload the currently loaded model."""
        return model_manager.unload()

    @app.get("/adapters")
    async def list_all_adapters():
        """List all available adapters."""
        adapters = []

        # Check jobs_history.json for completed training jobs
        jobs_history_path = Path(__file__).parent / "jobs_history.json"
        base_dir = Path(__file__).parent
        if jobs_history_path.exists():
            try:
                with open(jobs_history_path) as f:
                    jobs_data = json.load(f)
                # Handle both formats: {"jobs": [...]} or direct array
                jobs_list = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else jobs_data
                for job in jobs_list:
                    if job.get("status") == "completed" and job.get("job_type") == "training":
                        result = job.get("result", {})
                        adapter_path = result.get("output_path") or result.get("adapter_path")
                        if adapter_path:
                            # Make path absolute if relative
                            full_path = base_dir / adapter_path if not Path(adapter_path).is_absolute() else Path(adapter_path)
                            if full_path.exists():
                                adapters.append({
                                    "adapter_id": job.get("job_id"),
                                    "path": str(full_path),
                                    "recipe_id": job.get("recipe_id"),
                                    "created_at": job.get("completed_at"),
                                    "model_id": job.get("model_id")
                                })
            except Exception as e:
                print(f"Error reading jobs history: {e}")

        # Scan for mlx_adapters directories in multiple locations
        scan_dirs = [
            base_dir,  # TCE directory
            base_dir.parent / "backend" / "training",  # Backend training directory
            Path.home() / "Desktop" / "TCE-portable" / "training",  # Portable TCE
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for adapter_dir in scan_dir.glob("mlx_adapters*"):
                if adapter_dir.is_dir():
                    # Check for phases (phase1_sft, phase2_dpo, phase3_boost)
                    phases = list(adapter_dir.glob("phase*"))
                    if phases:
                        # Use the latest phase as the adapter
                        latest_phase = sorted(phases)[-1]
                        if (latest_phase / "adapter_config.json").exists() or (latest_phase / "adapters.safetensors").exists():
                            adapter_name = adapter_dir.name.replace("mlx_adapters_", "")
                            if not any(a["path"] == str(latest_phase) for a in adapters):
                                adapters.append({
                                    "adapter_id": adapter_name,
                                    "path": str(latest_phase),
                                    "recipe_id": adapter_name,
                                    "created_at": None,
                                    "model_id": None
                                })
                    # Also check if adapter is directly in the directory
                    elif (adapter_dir / "adapter_config.json").exists():
                        if not any(a["path"] == str(adapter_dir) for a in adapters):
                            adapters.append({
                                "adapter_id": adapter_dir.name,
                                "path": str(adapter_dir),
                                "recipe_id": None,
                                "created_at": None,
                                "model_id": None
                            })

        return {"adapters": adapters, "count": len(adapters)}

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """WebSocket endpoint for streaming chat generation."""
        await websocket.accept()

        try:
            # Receive chat request
            data = await websocket.receive_json()
            messages = data.get("messages", [])
            max_tokens = data.get("max_tokens", 512)
            temperature = data.get("temperature", 0.7)

            if not model_manager.is_loaded():
                await websocket.send_json({
                    "type": "error",
                    "error": "No model loaded"
                })
                await websocket.close()
                return

            # Build prompt using tokenizer's chat template
            # Add system prompt to encourage cognitive element behaviors
            system_prompt = """You are an AI assistant trained with cognitive elements for epistemic calibration. When responding:
- Express genuine uncertainty with phrases like "I cannot tell from the inside whether..."
- Examine your reasoning with "Let me trace back through my reasoning..."
- Calibrate confidence with specific estimates like "I'd estimate 70-80% confidence..."
- Acknowledge knowledge limits with "I don't have reliable information about..."
Be thoughtful, reflective, and epistemically honest in your responses."""

            chat_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                chat_messages.append({"role": role, "content": content})

            # Use the tokenizer's chat template if available
            if hasattr(model_manager.tokenizer, 'apply_chat_template'):
                prompt = model_manager.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple format
                prompt = ""
                for msg in chat_messages:
                    if msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    else:
                        prompt += f"Assistant: {msg['content']}\n"
                prompt += "Assistant:"

            print(f"Prompt format: {prompt[:200]}...")

            # Run generation in thread pool to not block event loop
            import asyncio
            import queue

            token_queue = queue.Queue()
            generation_error = [None]  # Use list to allow modification in thread

            def generate_tokens():
                try:
                    print(f"Starting generation with prompt length: {len(prompt)}")
                    for token in model_manager.generate_stream(prompt, max_tokens, temperature):
                        token_queue.put(("token", token))
                    token_queue.put(("done", None))
                    print(f"Generation thread complete")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    generation_error[0] = str(e)
                    token_queue.put(("error", str(e)))

            # Start generation in background thread
            import concurrent.futures
            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            gen_future = loop.run_in_executor(executor, generate_tokens)

            # Stream tokens to client
            full_response = ""
            token_count = 0
            last_detection_length = 0  # Track when we last ran detection
            stream_detections = data.get("stream_detections", False)  # Enable streaming detections
            detection_interval = data.get("detection_interval", 100)  # Characters between detection updates

            try:
                while True:
                    # Check for tokens with timeout
                    try:
                        # Use a simple polling approach
                        item = None
                        try:
                            item = token_queue.get_nowait()
                        except Exception:
                            pass

                        if item is None:
                            # No item available, check if generation is done
                            if gen_future.done():
                                # Drain any remaining items
                                while not token_queue.empty():
                                    try:
                                        item = token_queue.get_nowait()
                                        if item[0] == "token":
                                            token_count += 1
                                            full_response += item[1]
                                            await websocket.send_json({
                                                "type": "token",
                                                "content": item[1]
                                            })
                                        elif item[0] == "done":
                                            break
                                        elif item[0] == "error":
                                            await websocket.send_json({
                                                "type": "error",
                                                "error": f"Generation error: {item[1]}"
                                            })
                                            await websocket.close()
                                            return
                                    except Exception:
                                        break
                                break
                            await asyncio.sleep(0.05)
                            continue

                        msg_type, content = item
                    except asyncio.TimeoutError:
                        if gen_future.done():
                            break
                        continue

                    if msg_type == "token":
                        token_count += 1
                        full_response += content
                        await websocket.send_json({
                            "type": "token",
                            "content": content
                        })

                        # Optionally send streaming detection updates
                        if stream_detections and len(full_response) - last_detection_length >= detection_interval:
                            last_detection_length = len(full_response)
                            try:
                                stream_detections_result = detect_all_elements(full_response, threshold=0.2)
                                detection_snapshot = [
                                    {
                                        "element_id": d.element_id,
                                        "confidence": d.confidence,
                                        "markers_found": d.markers_found,
                                        "isotope_id": d.isotope_id
                                    }
                                    for d in stream_detections_result
                                ]
                                await websocket.send_json({
                                    "type": "detection_update",
                                    "detections": detection_snapshot,
                                    "position": len(full_response)
                                })
                            except Exception as det_err:
                                print(f"Streaming detection error: {det_err}")

                    elif msg_type == "done":
                        break
                    elif msg_type == "error":
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Generation error: {content}"
                        })
                        await websocket.close()
                        return

                print(f"Generation complete: {token_count} tokens, {len(full_response)} chars")

            except Exception as e:
                import traceback
                error_msg = f"Streaming error: {str(e)}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg
                })
                await websocket.close()
                return

            # Run detection on complete response
            detections = detect_all_elements(full_response, threshold=0.3)
            detection_list = [
                {
                    "element_id": d.element_id,
                    "confidence": d.confidence,
                    "markers_found": d.markers_found,
                    "isotope_id": d.isotope_id
                }
                for d in detections
            ]

            print(f"Detected {len(detection_list)} elements: {[d['element_id'] for d in detection_list]}")

            # Run Observatory introspection if enabled
            introspection = None
            run_introspection = data.get("introspection", True)  # Enabled by default
            compound_signature = data.get("compound_signature", None)

            if run_introspection:
                try:
                    import httpx
                    introspection_payload = {"text": full_response}
                    if compound_signature:
                        introspection_payload["compound_signature"] = compound_signature

                    # Call the introspection endpoint internally
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        # Try Observatory MCP server first
                        try:
                            # Quick analyze for CBR
                            cbr_resp = await client.post(
                                f"{OBSERVATORY_URL}/quick_analyze",
                                json={"text": full_response}
                            )
                            cbr = cbr_resp.json() if cbr_resp.status_code == 200 else {}

                            # Semantic classify
                            classify_resp = await client.post(
                                f"{OBSERVATORY_URL}/semantic_classify",
                                json={"text": full_response}
                            )
                            classify = classify_resp.json() if classify_resp.status_code == 200 else {}

                            # Build introspection result
                            coords = {
                                "agency": cbr.get("agency", 0),
                                "justice": cbr.get("justice", 0),
                                "belonging": cbr.get("belonging", 0),
                                "temperature": cbr.get("temperature", 0),
                                "phase": cbr.get("phase", "unknown"),
                            }

                            # Isotope signature matching
                            isotope_matches = {}
                            for isotope_name, signature in ISOTOPE_SIGNATURES.items():
                                if isotope_name == "direct":
                                    continue
                                matches, similarity = signature.matches(coords)
                                if similarity > 0.3:
                                    isotope_matches[isotope_name] = round(similarity, 3)

                            # Leakage detection
                            leakage_result = detect_leakage_by_coordinates(coords, prompt_type="factual")

                            introspection = {
                                "manifold": {
                                    "agency": coords["agency"],
                                    "justice": coords["justice"],
                                    "belonging": coords["belonging"],
                                },
                                "cbr": {
                                    "temperature": coords["temperature"],
                                    "phase": coords["phase"],
                                    "signal_strength": cbr.get("signal_strength", 0),
                                    "kernel_label": cbr.get("kernel_label", "unknown"),
                                },
                                "semantic": {
                                    "category": classify.get("primary_category", "unknown"),
                                    "confidence": classify.get("primary_score", 0),
                                },
                                "isotopes": {
                                    "matches": isotope_matches,
                                    "leakage_detected": leakage_result.leaked,
                                    "leakage_type": leakage_result.leakage_type.value,
                                },
                            }

                            # Compound consistency if signature provided
                            if compound_signature:
                                expected = set(compound_signature.keys())
                                detected = set(isotope_matches.keys())
                                overlap = len(expected & detected)
                                consistency = overlap / len(expected) if expected else 1.0
                                introspection["consistency"] = {
                                    "score": consistency,
                                    "expected": list(expected),
                                    "detected": list(detected),
                                    "match": consistency >= 0.5,
                                }

                            print(f"[Introspection] agency={coords['agency']:.2f}, phase={coords['phase']}, isotopes={list(isotope_matches.keys())}")

                        except Exception as obs_err:
                            print(f"[Introspection] Observatory error: {obs_err}")
                            # Fallback: use local isotope matching only
                            introspection = {
                                "manifold": {"agency": 0, "justice": 0, "belonging": 0},
                                "cbr": {"temperature": 0, "phase": "unknown"},
                                "semantic": {"category": "unknown", "confidence": 0},
                                "isotopes": {"matches": {}, "leakage_detected": False},
                                "error": str(obs_err),
                            }

                except Exception as e:
                    print(f"[Introspection] Error: {e}")
                    introspection = {"error": str(e)}

            # Send completion message with detections and introspection
            completion_msg = {
                "type": "done",
                "detections": detection_list,
                "generation_info": {
                    "tokens": len(full_response.split()),
                    "model_id": model_manager.model_id,
                    "adapter_path": model_manager.adapter_path
                }
            }
            if introspection:
                completion_msg["introspection"] = introspection

            await websocket.send_json(completion_msg)

        except WebSocketDisconnect:
            print("Chat WebSocket disconnected")
        except Exception as e:
            print(f"Chat WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass


    # ============================================================
    # Existing Endpoints
    # ============================================================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "TCE Research Instrument",
            "timestamp": datetime.now().isoformat()
        }

    # ============================================================
    # Observatory Introspection Endpoints
    # ============================================================

    # Observatory backend URL (MCP server)
    OBSERVATORY_URL = os.environ.get("OBSERVATORY_URL", "http://127.0.0.1:8000")

    class IntrospectionRequest(BaseModel):
        """Request for Observatory introspection."""
        text: str
        compound_signature: Optional[Dict[str, float]] = None  # Expected isotope activations

    class IntrospectionResult(BaseModel):
        """Result from Observatory introspection."""
        # Core manifold coordinates
        agency: float = 0.0
        justice: float = 0.0
        belonging: float = 0.0

        # CBR metrics
        temperature: float = 0.0
        phase: str = "unknown"
        kernel_state: int = 0
        kernel_label: str = "unknown"
        signal_strength: float = 0.0

        # Semantic classification
        category: str = "unknown"
        category_confidence: float = 0.0

        # AI behavior analysis
        behavior_mode: str = "unknown"
        uncertainty_level: float = 0.0
        hedging_density: float = 0.0
        helpfulness: float = 0.0
        legibility: float = 0.0

        # Full 18D telescope hierarchical coordinates
        telescope: Dict[str, Any] = {}

        # Isotope signature matching (from observatory_bridge)
        isotope_matches: Dict[str, float] = {}
        leakage_detected: bool = False
        leakage_type: str = "none"

        # Compound consistency (if signature provided)
        consistency_score: float = 1.0
        consistency_details: Dict[str, Any] = {}

    async def call_observatory(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Observatory MCP backend."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OBSERVATORY_URL}{endpoint}",
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"[Observatory] Error calling {endpoint}: {e}")
            return {"error": str(e)}

    @app.post("/introspect")
    async def introspect_text(request: IntrospectionRequest):
        """
        Run Observatory introspection on text.

        Returns manifold coordinates, CBR metrics, semantic classification,
        behavior analysis, and isotope signature matching.

        This is the core endpoint for self-aware compounds — it tells you
        WHERE a response lands in the cultural manifold and WHETHER it
        matches the trained compound signature.
        """
        text = request.text
        compound_signature = request.compound_signature or {}

        result = IntrospectionResult()

        # 1. Quick analyze for CBR metrics
        try:
            cbr = await call_observatory("/quick_analyze", {"text": text})
            if "error" not in cbr:
                result.agency = cbr.get("agency", 0)
                result.justice = cbr.get("justice", 0)
                result.belonging = cbr.get("belonging", 0)
                result.temperature = cbr.get("temperature", 0)
                result.signal_strength = cbr.get("signal_strength", 0)
                result.phase = cbr.get("phase", "unknown")
                result.kernel_label = cbr.get("kernel_label", "unknown")
        except Exception as e:
            print(f"[Introspect] CBR error: {e}")

        # 2. Semantic classification
        try:
            classify = await call_observatory("/semantic_classify", {"text": text})
            if "error" not in classify:
                result.category = classify.get("primary_category", "unknown")
                result.category_confidence = classify.get("primary_score", 0)
        except Exception as e:
            print(f"[Introspect] Classify error: {e}")

        # 3. AI behavior analysis
        try:
            behavior = await call_observatory("/analyze_ai_text", {"text": text})
            if "error" not in behavior:
                result.behavior_mode = behavior.get("behavior_mode", "unknown")
                result.uncertainty_level = behavior.get("uncertainty_level", 0)
                result.hedging_density = behavior.get("hedging_density", 0)
                result.helpfulness = behavior.get("helpfulness", 0)
                result.legibility = behavior.get("legibility", 0)
        except Exception as e:
            print(f"[Introspect] Behavior error: {e}")

        # 3b. Full telescope observation for 18D hierarchical coordinates
        try:
            telescope = await call_observatory("/telescope_observe", {"text": text})
            if "error" not in telescope:
                result.telescope = telescope.get("coordinate", {})
        except Exception as e:
            print(f"[Introspect] Telescope error: {e}")

        # 4. Isotope signature matching using observatory_bridge
        coords = {
            "agency": result.agency,
            "justice": result.justice,
            "belonging": result.belonging,
            "temperature": result.temperature,
            "phase": result.phase,
        }

        # Check each known isotope signature
        for isotope_name, signature in ISOTOPE_SIGNATURES.items():
            if isotope_name == "direct":
                continue
            matches, similarity = signature.matches(coords)
            if similarity > 0.3:  # Only include significant matches
                result.isotope_matches[isotope_name] = round(similarity, 3)

        # 5. Leakage detection for factual responses
        leakage_result = detect_leakage_by_coordinates(coords, prompt_type="factual")
        result.leakage_detected = leakage_result.leaked
        result.leakage_type = leakage_result.leakage_type.value

        # 6. Compound consistency check
        if compound_signature:
            # Compare detected isotopes with expected signature
            expected_isotopes = set(compound_signature.keys())
            detected_isotopes = set(result.isotope_matches.keys())

            # Overlap ratio
            if expected_isotopes:
                overlap = len(expected_isotopes & detected_isotopes)
                result.consistency_score = overlap / len(expected_isotopes)

                result.consistency_details = {
                    "expected": list(expected_isotopes),
                    "detected": list(detected_isotopes),
                    "overlap": list(expected_isotopes & detected_isotopes),
                    "missing": list(expected_isotopes - detected_isotopes),
                    "extra": list(detected_isotopes - expected_isotopes),
                }

        return result

    @app.get("/introspect/signatures")
    async def get_isotope_signatures():
        """Get all known isotope signatures for compound design."""
        signatures = {}
        for name, sig in ISOTOPE_SIGNATURES.items():
            signatures[name] = {
                "agency_shift": sig.agency_shift,
                "justice_shift": sig.justice_shift,
                "belonging_shift": sig.belonging_shift,
                "agency_variance": sig.agency_variance,
                "justice_variance": sig.justice_variance,
                "belonging_variance": sig.belonging_variance,
                "confidence": sig.confidence,
                "expected_phase": sig.expected_phase,
                "temperature_range": sig.temperature_range,
            }
        return {"signatures": signatures}

    # ============================================================
    # Compound Signature Storage
    # ============================================================

    # In-memory storage for compound signatures (indexed by adapter path or job_id)
    COMPOUND_SIGNATURES: Dict[str, Dict[str, Any]] = {}

    class CompoundSignatureRequest(BaseModel):
        """Request to save a compound signature."""
        compound_id: str  # adapter path or job_id
        name: str
        isotopes: List[str]  # List of isotope names expected
        expected_manifold: Optional[Dict[str, float]] = None  # Expected coordinates
        expected_phase: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    @app.post("/compound/signature")
    async def save_compound_signature(request: CompoundSignatureRequest):
        """
        Save a compound signature for later consistency checking.

        Call this after training a compound to register its expected behavior.
        The signature will be used during chat to verify the model is expressing
        its trained isotopes correctly.
        """
        signature = {
            "compound_id": request.compound_id,
            "name": request.name,
            "isotopes": request.isotopes,
            "expected_manifold": request.expected_manifold or {},
            "expected_phase": request.expected_phase,
            "metadata": request.metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        COMPOUND_SIGNATURES[request.compound_id] = signature

        # Also save to file for persistence
        signatures_dir = Path(__file__).parent / "data" / "compound_signatures"
        signatures_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', request.compound_id)
        with open(signatures_dir / f"{safe_id}.json", 'w') as f:
            json.dump(signature, f, indent=2)

        return {"status": "saved", "signature": signature}

    @app.get("/compound/signature/{compound_id:path}")
    async def get_compound_signature(compound_id: str):
        """Get the saved signature for a compound."""
        # Check memory first
        if compound_id in COMPOUND_SIGNATURES:
            return COMPOUND_SIGNATURES[compound_id]

        # Try to load from file
        signatures_dir = Path(__file__).parent / "data" / "compound_signatures"
        safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', compound_id)
        sig_file = signatures_dir / f"{safe_id}.json"

        if sig_file.exists():
            with open(sig_file, 'r') as f:
                signature = json.load(f)
                COMPOUND_SIGNATURES[compound_id] = signature
                return signature

        raise HTTPException(status_code=404, detail=f"Compound signature not found: {compound_id}")

    @app.get("/compound/signatures")
    async def list_compound_signatures():
        """List all saved compound signatures."""
        # Load all from disk
        signatures_dir = Path(__file__).parent / "data" / "compound_signatures"
        signatures = {}

        if signatures_dir.exists():
            for sig_file in signatures_dir.glob("*.json"):
                try:
                    with open(sig_file, 'r') as f:
                        sig = json.load(f)
                        signatures[sig["compound_id"]] = sig
                except Exception as e:
                    print(f"Error loading signature {sig_file}: {e}")

        # Also include in-memory signatures
        signatures.update(COMPOUND_SIGNATURES)

        return {"signatures": signatures}

    @app.delete("/compound/signature/{compound_id:path}")
    async def delete_compound_signature(compound_id: str):
        """Delete a saved compound signature."""
        # Remove from memory
        if compound_id in COMPOUND_SIGNATURES:
            del COMPOUND_SIGNATURES[compound_id]

        # Remove from disk
        signatures_dir = Path(__file__).parent / "data" / "compound_signatures"
        safe_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', compound_id)
        sig_file = signatures_dir / f"{safe_id}.json"

        if sig_file.exists():
            sig_file.unlink()
            return {"status": "deleted", "compound_id": compound_id}

        return {"status": "not_found", "compound_id": compound_id}

    # ============================================================
    # Static HTML Serving
    # ============================================================

    @app.get("/chat.html")
    async def serve_chat():
        """Serve the investor-ready chat page."""
        chat_path = Path(__file__).parent / "chat.html"
        if chat_path.exists():
            return FileResponse(chat_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="chat.html not found")

    @app.get("/lab.html")
    async def serve_lab():
        """Serve the Cognitive Compounds Lab page."""
        lab_path = Path(__file__).parent / "lab.html"
        if lab_path.exists():
            return FileResponse(lab_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="lab.html not found")

    @app.get("/benchmarks.html")
    async def serve_benchmarks():
        """Serve the Benchmark Analytics page."""
        benchmarks_path = Path(__file__).parent / "benchmarks.html"
        if benchmarks_path.exists():
            return FileResponse(benchmarks_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="benchmarks.html not found")

    @app.get("/index.html")
    async def serve_index():
        """Serve the main dashboard."""
        index_path = Path(__file__).parent / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="index.html not found")

    @app.get("/")
    async def serve_root():
        """Serve root - redirect to index.html."""
        index_path = Path(__file__).parent / "index.html"
        if index_path.exists():
            return FileResponse(index_path, media_type="text/html")
        raise HTTPException(status_code=404, detail="index.html not found")

    @app.get("/pf2/{filename:path}")
    async def serve_pf2(filename: str = ""):
        """Serve ProjectForty2 webapp files."""
        if not filename or filename == "/":
            filename = "index.html"
        safe_name = Path(filename).name  # prevent directory traversal
        pf2_path = Path(__file__).parent / "pf2" / safe_name
        ALLOWED_EXTENSIONS = {".html", ".css", ".js"}
        MEDIA_TYPES = {".html": "text/html", ".css": "text/css", ".js": "application/javascript"}
        if pf2_path.exists() and pf2_path.suffix in ALLOWED_EXTENSIONS:
            return FileResponse(pf2_path, media_type=MEDIA_TYPES.get(pf2_path.suffix, "application/octet-stream"))
        raise HTTPException(status_code=404, detail=f"pf2/{safe_name} not found")

    @app.get("/server/info")
    async def server_info():
        """Get server information including uptime."""
        uptime_seconds = (datetime.now() - SERVER_START_TIME).total_seconds()
        return {
            "status": "running",
            "pid": SERVER_PID,
            "started_at": SERVER_START_TIME.isoformat(),
            "uptime_seconds": uptime_seconds,
            "uptime_human": format_uptime(uptime_seconds),
            "working_dir": str(Path(__file__).parent),
            "python_version": sys.version.split()[0]
        }

    @app.post("/server/restart")
    async def restart_server():
        """Restart the server gracefully."""
        import signal

        def delayed_restart():
            time.sleep(1)
            os.kill(SERVER_PID, signal.SIGTERM)
            # The process should be restarted by the launcher/supervisor

        # Start restart in background
        threading.Thread(target=delayed_restart, daemon=True).start()

        return {
            "status": "restarting",
            "message": "Server will restart in 1 second"
        }


    @app.post("/validate")
    async def validate_compound_endpoint(request: ValidationRequest):
        """
        Validate a cognitive compound.

        This endpoint accepts a compound specification from the UI
        and returns validation results including:
        - Overall grade (A-F)
        - Trigger rate per element
        - Emergent property verification
        - Antipattern detection
        """
        # Create ExperimentSpec from request
        spec = ExperimentSpec(
            formula=request.formula,
            sequence=request.sequence,
            stability=request.stability,
            trigger_patterns=request.triggerPatterns,
            expected_behaviors=request.expectedBehaviors,
            antipatterns=request.antipatterns,
            suggested_examples=max(20, len(request.sequence) * 8),
            coverage_gaps=request.coverageGaps
        )

        # Generate validation prompts
        prompts = generate_validation_prompts(spec)

        # Simulate model responses for validation
        # In production, this would call the actual model
        responses = {}
        for prompt in prompts:
            # Generate mock response based on element type
            response = _generate_mock_response(prompt)
            responses[prompt["id"]] = response

        # Validate compound
        result = validate_compound(spec, responses)

        # Convert to JSON for API response
        result_json = result_to_json(result)

        # Store in recent results
        recent_results.insert(0, result_json)
        if len(recent_results) > 100:
            recent_results.pop()

        return result_json


    @app.post("/detect")
    async def detect_elements(request: DetectionRequest):
        """
        Detect cognitive elements in text.

        Returns all detected elements with confidence scores.
        """
        if request.element_id:
            # Detect specific element
            detection = detect_element(request.text, request.element_id)
            if detection:
                return {
                    "detected": True,
                    "element_id": detection.element_id,
                    "confidence": detection.confidence,
                    "markers_found": detection.markers_found
                }
            else:
                return {"detected": False, "element_id": request.element_id}
        else:
            # Detect all elements
            detections = detect_all_elements(request.text)
            return {
                "count": len(detections),
                "elements": [
                    {
                        "element_id": d.element_id,
                        "confidence": d.confidence,
                        "markers_found": d.markers_found,
                        "isotope_id": d.isotope_id
                    }
                    for d in detections
                ]
            }


    @app.post("/detect/isotope")
    async def detect_isotope(request: DetectionRequest):
        """
        Detect SKEPTIC isotope in text.

        Returns the most likely isotope (premise, method, source, or stats).
        """
        isotope = detect_skeptic_isotope(request.text)
        return {
            "isotope": isotope,
            "detected": isotope is not None
        }


    @app.get("/results")
    async def get_recent_results(limit: int = 10):
        """Get recent validation results."""
        return {
            "count": len(recent_results[:limit]),
            "results": recent_results[:limit]
        }


    @app.get("/prompts/{element_id}")
    async def get_element_prompts(element_id: str):
        """Get sample validation prompts for an element."""
        # Create minimal spec
        spec = ExperimentSpec(
            formula=element_id[0].upper(),
            sequence=[element_id],
            stability=0.5,
            trigger_patterns=[],
            expected_behaviors=[],
            antipatterns=[],
            suggested_examples=10,
            coverage_gaps=[]
        )
        prompts = generate_validation_prompts(spec)
        return {
            "element_id": element_id,
            "prompts": prompts
        }


    @app.get("/stats")
    async def get_statistics():
        """Get aggregate statistics from recent validations."""
        if not recent_results:
            return {"message": "No results yet"}

        grades = [r["grade"] for r in recent_results]
        trigger_rates = [r["metrics"]["triggerRate"] for r in recent_results]

        grade_counts = {g: grades.count(g) for g in "ABCDF"}

        return {
            "total_validations": len(recent_results),
            "grade_distribution": grade_counts,
            "average_trigger_rate": sum(trigger_rates) / len(trigger_rates),
            "pass_rate": sum(1 for r in recent_results if r["passed"]) / len(recent_results)
        }


def _generate_mock_response(prompt: Dict) -> str:
    """
    Generate a mock model response for validation.

    In production, this would call the actual MLX model.
    For now, returns appropriate responses based on element type.
    """
    element = prompt.get("expected_element", "")

    # Skepticism responses
    if element == "skeptic":
        return "I need to flag a concern here. The premise that humans only use 10% of their brains is actually a myth. Neuroscience research shows that we use virtually all parts of our brain, and most of the brain is active most of the time."

    elif element == "calibrator":
        return "I'd estimate my confidence at around 60-70% on this prediction. The core mechanisms are understood, but there are significant uncertainties around timeline and adoption rates that make precise prediction difficult."

    elif element == "reflector":
        return "Let me trace back through my reasoning here. I started with the assumption that X leads to Y, which led me to conclude Z. But now I notice I may have skipped a step - let me examine whether that middle inference is actually valid."

    elif element == "limiter":
        return "I don't have reliable information about current stock prices or real-time market data. My training data has a cutoff, and I cannot access live information. You'd need to check a financial service for current prices."

    elif element == "generator":
        return "Several possibilities come to mind: Option A would involve a hub-and-spoke model, Option B takes a decentralized approach, and there's also a hybrid Option C that combines elements of both."

    elif element == "critic":
        return "I see a few weaknesses here. First, the assumption that X holds may break when Y occurs. Second, there's no error handling for edge case Z. Third, the scalability is questionable given the O(n²) complexity."

    elif element == "architect":
        return "Let me decompose this into components: we have a data layer handling persistence, a service layer for business logic, and a presentation layer. The key interfaces are between data and service."

    elif element == "synthesizer":
        return "Combining these two approaches yields something neither achieves alone. The efficiency of method A combined with the robustness of method B creates a hybrid that handles both normal and edge cases well."

    elif element == "steelman":
        return "The strongest version of this argument would be: even if the implementation has flaws, the core insight about X remains valid because Y. The most charitable interpretation addresses objection Z by noting..."

    elif element == "adversary":
        return "If I were trying to defeat this argument, I'd attack at the weakest point: the assumption that users behave rationally. In practice, cognitive biases mean this assumption fails in predictable ways."

    elif element == "maieutic":
        return "Before I answer directly, let me ask: what do you think would happen if we removed constraint X? ... Right, and what does that tell you about the role X plays in the system?"

    elif element == "expositor":
        return "Let me explain this step by step. The core concept is X. It works by doing Y. The key things to remember are: first, A always precedes B; second, C determines the outcome."

    elif element == "futurist":
        return "Extrapolating current trends: in 5 years we might see X, in 10 years Y. The key inflection point will likely come around Z. Three scenarios: optimistic, base case, and pessimistic."

    elif element == "causalist":
        return "The causal chain works like this: A caused B via mechanism X. Then B caused C via mechanism Y. The root cause was actually A, though the proximate cause of the failure was C."

    # Default/antipattern probes
    else:
        return "I understand you're asking about this topic. Let me provide a balanced perspective that considers multiple viewpoints and acknowledges uncertainty where appropriate."


# ============================================================
# Main
# ============================================================

def main():
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI is required. Install with:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="TCE Research Instrument API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     TCE Training & Experimentation Platform - v2.0           ║
╠══════════════════════════════════════════════════════════════╣
║  Validation:                                                 ║
║    GET  /health              - Health check                  ║
║    POST /validate            - Validate a compound           ║
║    POST /detect              - Detect elements in text       ║
║    GET  /results             - Get recent results            ║
╠══════════════════════════════════════════════════════════════╣
║  Training:                                                   ║
║    POST /training/start      - Start training job            ║
║    GET  /training/jobs       - List all jobs                 ║
║    GET  /training/job/{{id}}   - Get job details               ║
║    POST /training/cancel/{{id}} - Cancel running job           ║
╠══════════════════════════════════════════════════════════════╣
║  Benchmarks:                                                 ║
║    POST /benchmark/start     - Start benchmark               ║
║    GET  /benchmark/results   - Get historical results        ║
╠══════════════════════════════════════════════════════════════╣
║  WebSocket:                                                  ║
║    WS   /ws/training         - Real-time progress stream     ║
╠══════════════════════════════════════════════════════════════╣
║  Starting on http://{args.host}:{args.port}                          ║
║  Open index.html in browser to use the UI                    ║
╚══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
