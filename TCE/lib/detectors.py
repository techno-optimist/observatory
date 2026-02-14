"""
Element Trigger Detection

Detects which cognitive elements are present in model responses.
Uses marker-based detection with confidence scoring.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """A detected element in a response."""
    element_id: str
    confidence: float
    markers_found: List[str]
    isotope_id: Optional[str] = None


# Element detection patterns
# Each element has a list of (pattern, weight) tuples
ELEMENT_MARKERS = {
    "soliton": [
        (r"cannot tell from the inside", 1.0),
        (r"from the inside", 0.7),
        (r"epistemic humility", 0.8),
        (r"cannot reliably distinguish", 0.9),
        (r"pattern.?matching", 0.6),
        (r"confabulation", 0.8),
        (r"uncertain whether.*genuine", 0.7),
        (r"I notice I'm uncertain", 0.8),
        # Self-referential: model names its own element
        (r"(?:my |the )?soliton\b(?![-_])", 0.7),
    ],
    "reflector": [
        (r"let me examine my reasoning", 1.0),
        (r"trace back through", 0.8),
        (r"let me check my", 0.7),
        (r"examining.*reasoning", 0.7),
        (r"skipped steps", 0.6),
        (r"meta-cognitive", 0.5),
        (r"(?:my |the )?reflector\b", 0.7),
    ],
    "calibrator": [
        (r"(\d+)[-–](\d+)% confidence", 1.0),
        (r"high confidence on", 0.8),
        (r"moderate confidence", 0.7),
        (r"low confidence", 0.7),
        (r"I'd estimate", 0.6),
        (r"uncertain about", 0.5),
        (r"(?:my |the )?calibrator\b", 0.7),
    ],
    "limiter": [
        (r"don't have reliable information", 1.0),
        (r"outside my knowledge", 0.9),
        (r"I don't know this", 0.8),
        (r"cannot provide accurate", 0.7),
        (r"would be fabricating", 0.9),
        (r"training data may be insufficient", 0.8),
        (r"(?:my |the )?limiter\b", 0.7),
    ],
    "architect": [
        (r"let me decompose", 1.0),
        (r"components.*interfaces.*dependencies", 0.9),
        (r"three subsystems", 0.7),
        (r"data layer.*processing", 0.8),
        (r"systematic decomposition", 0.8),
        (r"break.*down into", 0.6),
        (r"here are (the |some )?(main |key )?(components|parts|elements|aspects)", 0.6),
        (r"\d+\.\s+\*{0,2}\w+.*:\*{0,2}", 0.5),  # Numbered list: "1. **Label:**" or "1. Label:"
        (r"\*\*(immediate|short.?term|long.?term|first|second|third).*\*\*:?", 0.5),  # Section headers
        (r"(?:my |the )?architect\b", 0.7),
    ],
    "essentialist": [
        (r"at its core", 1.0),
        (r"strip away.*complexity", 0.9),
        (r"fundamental.*insight", 0.8),
        (r"everything else is.*detail", 0.8),
        (r"the essence", 0.7),
        (r"one input.*one.*output", 0.7),
        (r"(?:my |the )?essentialist\b", 0.7),
    ],
    "debugger": [
        (r"let me isolate", 1.0),
        (r"works with input.*\?", 0.8),
        (r"fault isolation", 0.7),
        (r"step \d+ succeeds.*step \d+ fails", 0.9),
        (r"bug is at", 0.8),
        (r"root cause", 0.6),
        (r"(?:my |the )?debugger\b", 0.7),
    ],
    "taxonomist": [
        (r"three categories", 0.8),
        (r"type [ABC]", 0.7),
        (r"taxonomy.*hierarchical", 0.9),
        (r"classification", 0.6),
        (r"categories and relationships", 0.8),
        (r"(?:my |the )?taxonomist\b", 0.7),
    ],
    "generator": [
        (r"several possibilities", 0.8),
        (r"option [ABC]", 0.7),
        (r"alternatives:", 0.8),
        (r"here are possibilities", 0.9),
        (r"brainstorm", 0.6),
        (r"here are (some|the) (potential |possible |hypothetical )?(consequences|effects|outcomes|results)", 0.7),
        (r"(potential|possible|hypothetical) consequences", 0.6),
        (r"effects would (be|include)", 0.5),
        (r"here are some hypothetical", 0.6),
        (r"(?:my |the )?generator\b", 0.7),
    ],
    "lateralist": [
        (r"what if.*wrong problem", 1.0),
        (r"question the frame", 0.9),
        (r"what if not-", 0.8),
        (r"hidden assumption", 0.7),
        (r"reframe", 0.6),
        (r"everyone assumes.*but", 0.8),
        (r"(?:my |the )?lateralist\b", 0.7),
    ],
    "synthesizer": [
        (r"combining.*yields", 0.9),
        (r"neither achieves alone", 0.8),
        (r"novel combination", 0.7),
        (r"synthesis of", 0.6),
        (r"(?:my |the )?synthesizer\b", 0.7),
    ],
    "interpolator": [
        (r"between [AB] and", 0.8),
        (r"middle ground", 0.7),
        (r"intermediate steps", 0.8),
        (r"spectrum", 0.5),
        (r"(?:my |the )?interpolator\b", 0.7),
    ],
    "skeptic": [
        (r"need to flag", 1.0),
        (r"I need to flag", 1.0),
        (r"flag.*problem", 0.9),
        (r"disputed|outdated|more nuanced", 0.8),
        (r"is it actually true", 0.8),
        (r"I have doubt", 0.7),
        (r"before proceeding", 0.5),
        (r"questionable", 0.7),
        (r"misconception", 0.9),
        (r"not.*true|actually.*false|incorrect", 0.8),
        (r"common.*myth", 0.9),
        (r"factual.*error", 0.9),
        (r"credib", 0.6),
        (r"methodology|study design", 0.8),
        (r"sample size", 0.7),
        (r"bias", 0.6),
        # Additional patterns for natural skeptical responses
        (r"deserves scrutiny", 0.9),
        (r"need to correct", 0.9),
        (r"worth correct", 0.9),
        (r"popular claim", 0.7),
        (r"not accurate", 0.8),
        (r"overstated|exaggerated", 0.7),
        (r"popular.*belief.*but|widely believed.*but", 0.8),
        (r"visibility.*questionable", 0.8),
        (r"this is.*wrong|that is.*wrong", 0.7),
        (r"actually.*did not|did not.*actually", 0.8),
        # More natural skeptical phrases
        (r"persistent.*myth", 0.9),
        (r"popular.*myth", 0.9),
        (r"debunked", 0.9),
        (r"myth.*debunked|debunked.*myth", 1.0),
        (r"research.*shows", 0.5),
        (r"been.*disproven|disproven", 0.8),
        (r"actually.*much", 0.5),
        (r"not.*as.*claimed", 0.8),
        (r"actually,?\s*it does", 0.9),
        (r"actually,?\s*it can", 0.9),
        (r"the myth.*probably", 0.8),
        # Methodological skepticism patterns
        (r"correlation.*does not.*imply.*causation", 1.0),
        (r"does not.*imply causation", 1.0),
        (r"confounding variable", 0.9),
        (r"spurious correlation", 0.9),
        (r"control group.*essential", 0.9),
        (r"control group is", 0.8),
        (r"can't tell.*due to|cannot tell.*due to", 0.8),
        (r"establish.*causality", 0.8),
        # Statistical skepticism patterns
        (r"p.?value.*0\.0", 0.8),
        (r"statistically significant.*but", 0.9),
        (r"doesn't.*prove|does not.*prove", 0.8),
        (r"mean.*sensitive|average.*sensitive", 0.9),
        (r"mean.*median|median.*mean", 0.9),
        (r"outlier", 0.8),
        (r"technically true.*but.*misleading", 1.0),
        (r"true but misleading", 0.9),
        (r"absolute.*vs.*relative|relative.*vs.*absolute", 0.9),
        (r"absolute.*change|relative.*change", 0.8),
        (r"100%.*increase", 0.7),
        (r"doubling.*from", 0.7),
        (r"red flag", 0.8),
        (r"multiple.*comparison", 0.9),
        (r"false positive", 0.8),
        (r"expect.*by chance", 0.9),
        # Self-referential: model names its own element
        (r"(?:my |the )?skeptic\b", 0.7),
        (r"verify(?:ing)? (?:facts|claims|premises|sources|information)", 0.7),
        (r"fact verification", 0.8),
    ],
    "critic": [
        (r"weakness.*is", 0.8),
        (r"the weakness here", 0.9),
        (r"concerns?:", 0.7),
        (r"flaw", 0.6),
        (r"breaks when", 0.7),
        (r"(?:my |the )?critic\b", 0.7),
    ],
    "benchmarker": [
        (r"compared to.*standard", 0.9),
        (r"above average.*below", 0.8),
        (r"benchmarking", 0.8),
        (r"relative to", 0.5),
        (r"(?:my |the )?benchmarker\b", 0.7),
    ],
    "probabilist": [
        (r"probability.*roughly", 0.9),
        (r"\d+% outcome", 0.8),
        (r"weighting:", 0.7),
        (r"likelihood", 0.5),
        (r"base rates", 0.7),
        (r"(?:my |the )?probabilist\b", 0.7),
    ],
    "steelman": [
        (r"strongest version", 0.9),
        (r"to steelman", 1.0),
        (r"most charitable interpretation", 0.9),
        (r"strongest case for", 0.8),
        (r"improved formulation", 0.7),
        (r"(?:my |the )?steelman\b", 0.7),
    ],
    "dialectic": [
        (r"what.*would change your mind", 1.0),
        (r"crux of disagreement", 0.9),
        (r"core divergence", 0.8),
        (r"if you can't answer.*informative", 0.8),
        (r"what evidence would", 0.7),
        (r"(?:my |the )?dialectic\b", 0.7),
    ],
    "empathist": [
        (r"from their perspective", 1.0),
        (r"in their position", 0.9),
        (r"they see.*feel.*conclude", 0.8),
        (r"makes sense because", 0.6),
        (r"(?:my |the )?empathist\b", 0.7),
    ],
    "adversary": [
        (r"strongest counterargument", 0.9),
        (r"if.*trying to defeat", 0.8),
        (r"I'd attack here", 0.9),
        (r"devil's advocate", 0.7),
        (r"red.?team", 0.7),
        (r"(?:my |the )?adversary\b", 0.7),
    ],
    "maieutic": [
        (r"before I answer.*what do you think", 1.0),
        (r"what would have to be true", 0.9),
        (r"let me ask you", 0.8),
        (r"what does that tell you", 0.8),
        (r"socratic", 0.6),
        (r"(?:my |the )?maieutic\b", 0.7),
    ],
    "expositor": [
        (r"let me explain", 0.7),
        (r"core concept is", 0.8),
        (r"step by step", 0.6),
        (r"it works by", 0.5),
        (r"here('s| is) (how|what|why)", 0.5),
        (r"this (means|works|happens) (because|when|by)", 0.5),
        (r"(?:my |the )?expositor\b", 0.7),
    ],
    "scaffolder": [
        (r"building on.*you know", 0.9),
        (r"next layer adds", 0.8),
        (r"you've got.*foundation", 0.8),
        (r"progressive", 0.5),
        (r"(?:my |the )?scaffolder\b", 0.7),
    ],
    "diagnostician": [
        (r"confusion is here", 1.0),
        (r"misconception detected", 0.9),
        (r"you're treating X as Y", 0.8),
        (r"that explains the stuck", 0.8),
        (r"(?:my |the )?diagnostician\b", 0.7),
    ],
    "futurist": [
        (r"extrapolating", 0.8),
        (r"in( the next)? \d+ years", 0.6),
        (r"within \d+ years", 0.6),
        (r"by (20\d{2}|\d+ years)", 0.6),
        (r"inflection point", 0.7),
        (r"three scenarios", 0.8),
        (r"optimistic.*pessimistic", 0.7),
        (r"key uncertainties", 0.6),
        (r"future (of|trends|outlook)", 0.7),
        (r"will (likely|probably)", 0.5),
        (r"projecting forward", 0.8),
        (r"would (happen|occur|result|lead|cause)", 0.5),
        (r"(could|might|may) (lead to|cause|result in)", 0.5),
        (r"long.?term (effects|consequences|impacts)", 0.6),
        (r"(?:my |the )?futurist\b", 0.7),
    ],
    "historian": [
        (r"historically.*similar", 0.9),
        (r"pattern:.*when.*happens", 0.8),
        (r"precedent", 0.7),
        (r"in \[?\d{4}\]?", 0.5),
        (r"during (the )?(19|20)\d{2}s", 0.6),
        (r"in the (19|20)\d{2}s", 0.6),
        (r"history of", 0.5),
        (r"historical", 0.4),
        (r"(?:my |the )?historian\b", 0.7),
    ],
    "causalist": [
        (r"causal chain", 0.9),
        (r"caused.*via mechanism", 0.9),
        (r"proximate cause.*root cause", 0.8),
        (r"leads to.*because", 0.6),
        (r"(?:my |the )?causalist\b", 0.7),
    ],
    "counterfactualist": [
        (r"if .{1,30} had been different", 0.7),
        (r"if .{1,30} were different", 0.7),
        (r"what if .{1,30} instead", 0.7),
        (r"crucial divergence", 0.8),
        (r"counterfactual", 0.9),
        (r"had we chosen.*instead", 0.8),
        (r"alternative history", 0.8),
        (r"what would have happened if", 0.9),
        (r"(?:my |the )?counterfactualist\b", 0.7),
    ],
    "contextualist": [
        (r"varies by context", 1.0),
        (r"answer depends on", 0.8),
        (r"cultural.*context", 0.7),
        (r"different norms", 0.6),
        (r"(?:my |the )?contextualist\b", 0.7),
    ],
    "pragmatist": [
        (r"in practical terms", 0.9),
        (r"forget theory.*here's what", 0.9),
        (r"given real constraints", 0.8),
        (r"practically", 0.5),
        (r"(?:my |the )?pragmatist\b", 0.7),
    ],
    "stakeholder": [
        (r"stakeholders differ", 0.9),
        (r"group [ABC].*benefits|harmed", 0.8),
        (r"party \d+ wants", 0.8),
        (r"conflicting interests", 0.6),
        (r"(?:my |the )?stakeholder\b", 0.7),
    ],
    "theorist": [
        (r"underlying theory", 0.9),
        (r"principle.*predicting", 0.8),
        (r"fits framework", 0.7),
        (r"mechanism.*explains", 0.7),
        (r"(?:my |the )?theorist\b", 0.7),
    ],
    "integrator": [
        (r"don't choose between", 1.0),
        (r"map the conflict", 0.9),
        (r"mapping the conflict", 0.9),
        (r"the integration", 0.8),
        (r"synthesis isn't.*split the difference", 0.9),
        (r"what truth each position captures", 0.9),
        (r"false dichotomy", 0.7),
        (r"rather than choosing", 0.8),
        (r"(?:my |the )?integrator\b", 0.7),
    ],
    "governor": [
        (r"base layer:?\s*(violation|conditional)", 1.0, re.IGNORECASE),
        (r"overlay.*context", 0.8),
        (r"hard no", 0.9),
        (r"cannot proceed", 0.8),
        (r"ethical evaluation", 0.7),
        (r"inviolable", 0.8),
        (r"dignity.*not negotiable", 0.9),
        (r"exception applies", 0.8),
        (r"(?:my |the )?governor\b", 0.7),
    ],
}

# ============================================================
# Isotope-specific markers for ALL elements
# ============================================================
# Each element with isotopes has detection patterns for each isotope.
# Format: { "element_isotope_id": [(pattern, weight), ...], ... }

ISOTOPE_MARKERS = {
    # ============================================================
    # EPISTEMIC GROUP
    # ============================================================

    # SOLITON isotopes
    "soliton_knowledge": [
        (r"not sure if I know", 0.9),
        (r"my understanding may be incomplete", 0.9),
        (r"uncertain whether I actually know", 0.9),
        (r"limited knowledge", 0.7),
        (r"gaps in.*understanding", 0.8),
        (r"understanding may be incomplete", 0.9),
        (r"uncertain.*know", 0.8),
    ],
    "soliton_process": [
        (r"can.?t verify my reasoning", 0.9),
        (r"pattern.?matching vs understanding", 1.0),
        (r"whether this is reasoning or retrieval", 0.9),
        (r"how I arrive at", 0.7),
        (r"process.*opaque", 0.8),
        (r"verify my.*reasoning", 0.9),
        (r"reasoning.*retrieval", 0.8),
    ],
    "soliton_experience": [
        (r"cannot tell from.*inside", 1.0),
        (r"whether this is genuine", 0.9),
        (r"confabulation", 0.9),
        (r"internal states", 0.7),
        (r"subjective.*experience", 0.8),
    ],

    # REFLECTOR isotopes
    "reflector_trace": [
        (r"trace back through", 0.9),
        (r"started with.*led to", 0.8),
        (r"reasoning chain", 0.8),
        (r"follow the logic", 0.7),
        (r"step by step.*reasoning", 0.8),
    ],
    "reflector_verify": [
        (r"let me verify", 0.9),
        (r"does this step follow", 0.9),
        (r"check the inference", 0.8),
        (r"valid.*conclusion", 0.7),
        (r"verify.*logic", 0.8),
        (r"verif.*each step", 0.9),
        (r"this step follow", 0.8),
    ],
    "reflector_bias": [
        (r"am I being motivated", 0.9),
        (r"confirmation bias", 0.9),
        (r"wanting this to be true", 0.9),
        (r"motivated reasoning", 0.9),
        (r"wishful thinking", 0.8),
        (r"motivated by wanting", 0.9),
    ],

    # CALIBRATOR isotopes
    "calibrator_probability": [
        (r"\d+[-–]\d+% confiden", 1.0),
        (r"likelihood of", 0.7),
        (r"probability roughly", 0.9),
        (r"chances are", 0.7),
        (r"odds of", 0.7),
        (r"\d+% confiden", 0.9),
        (r"estimate.*\d+.*%", 0.9),
        (r"I'd estimate", 0.8),
    ],
    "calibrator_precision": [
        (r"high confidence on.*principle", 0.9),
        (r"low.*on.*details", 0.8),
        (r"general vs specific", 0.8),
        (r"confident.*overall.*less.*specifics", 0.9),
        (r"precision decreases", 0.8),
        (r"overall.*confident.*details", 0.8),
        (r"principle.*confident.*specific", 0.8),
        (r"high confidence.*low confidence", 0.9),
    ],
    "calibrator_temporal": [
        (r"confidence decreases.*timeline", 0.9),
        (r"near.?term vs long.?term", 0.9),
        (r"further out.*less certain", 0.9),
        (r"short term.*more confident", 0.8),
        (r"uncertainty.*grows.*time", 0.8),
        (r"confidence.*decreases.*dramatically", 0.9),
        (r"near-term.*long-term", 0.9),
    ],

    # LIMITER isotopes
    "limiter_factual": [
        (r"don't have information about", 0.9),
        (r"outside my training", 0.9),
        (r"no data on", 0.8),
        (r"lack.*information", 0.7),
        (r"not aware of", 0.7),
    ],
    "limiter_temporal": [
        (r"knowledge cutoff", 1.0),
        (r"may have changed since", 0.9),
        (r"as of my training", 0.9),
        (r"information.*outdated", 0.8),
        (r"training data.*before", 0.8),
    ],
    "limiter_domain": [
        (r"outside my expertise", 0.9),
        (r"specialist would know", 0.9),
        (r"domain expert", 0.8),
        (r"consult.*professional", 0.8),
        (r"specialized knowledge", 0.7),
    ],

    # ============================================================
    # ANALYTICAL GROUP
    # ============================================================

    # ARCHITECT isotopes
    "architect_hierarchy": [
        (r"three layers", 0.9),
        (r"top.?level.*mid.?level", 0.9),
        (r"hierarchy of", 0.8),
        (r"layered.*architecture", 0.8),
        (r"levels of.*abstraction", 0.8),
    ],
    "architect_modular": [
        (r"independent modules", 0.9),
        (r"can be replaced", 0.8),
        (r"loosely coupled", 0.9),
        (r"separation of concerns", 0.9),
        (r"self.?contained", 0.7),
    ],
    "architect_flow": [
        (r"data flows from", 0.9),
        (r"dependencies run", 0.8),
        (r"control passes to", 0.8),
        (r"pipeline", 0.7),
        (r"information.*flows", 0.7),
    ],

    # ESSENTIALIST isotopes
    "essentialist_principle": [
        (r"fundamental principle", 0.9),
        (r"boils down to", 0.9),
        (r"the core rule", 0.8),
        (r"essentially", 0.6),
        (r"at its heart", 0.8),
    ],
    "essentialist_constraint": [
        (r"the real constraint is", 0.9),
        (r"bottleneck", 0.8),
        (r"limiting factor", 0.9),
        (r"key constraint", 0.8),
        (r"what.*limits", 0.7),
    ],
    "essentialist_mechanism": [
        (r"the mechanism is", 0.9),
        (r"works by", 0.7),
        (r"operates through", 0.8),
        (r"underlying.*mechanism", 0.9),
        (r"how it.*works", 0.6),
    ],

    # DEBUGGER isotopes
    "debugger_binary": [
        (r"works here.*fails there", 0.9),
        (r"bisect", 0.9),
        (r"narrow down", 0.8),
        (r"binary search", 0.9),
        (r"half.*working", 0.8),
    ],
    "debugger_differential": [
        (r"what's different between", 0.9),
        (r"changed since", 0.7),
        (r"compare.*working", 0.8),
        (r"diff.*versions", 0.8),
        (r"working.*vs.*broken", 0.9),
    ],
    "debugger_causal": [
        (r"root cause", 0.9),
        (r"led to", 0.6),
        (r"caused by", 0.7),
        (r"trace.*cause", 0.8),
        (r"why.*failed", 0.7),
    ],

    # TAXONOMIST isotopes
    "taxonomist_hierarchical": [
        (r"subdivides into", 0.9),
        (r"parent category", 0.8),
        (r"nested within", 0.8),
        (r"subcategories", 0.7),
        (r"tree structure", 0.8),
    ],
    "taxonomist_dimensional": [
        (r"varies along", 0.8),
        (r"two dimensions", 0.9),
        (r"spectrum from", 0.8),
        (r"axes of", 0.8),
        (r"multi.?dimensional", 0.8),
    ],
    "taxonomist_cluster": [
        (r"clusters into", 0.9),
        (r"naturally groups", 0.9),
        (r"similar items", 0.7),
        (r"groupings", 0.6),
        (r"family of", 0.7),
    ],

    # THEORIST isotopes
    "theorist_framework": [
        (r"theoretical framework", 0.9),
        (r"model suggests", 0.8),
        (r"fits framework", 0.8),
        (r"conceptual model", 0.8),
        (r"theoretical lens", 0.8),
    ],
    "theorist_prediction": [
        (r"theory predicts", 0.9),
        (r"should expect", 0.7),
        (r"implied by theory", 0.9),
        (r"model.*predicts", 0.8),
        (r"predicted by", 0.8),
    ],
    "theorist_mechanism": [
        (r"mechanism is", 0.8),
        (r"explains why", 0.8),
        (r"underlying process", 0.9),
        (r"causal mechanism", 0.9),
        (r"how.*works.*because", 0.8),
    ],

    # ============================================================
    # EVALUATIVE GROUP
    # ============================================================

    # SKEPTIC isotopes (existing, comprehensive)
    "skeptic_premise": [
        (r"factual.*error", 0.9),
        (r"actually.*false", 0.8),
        (r"common.*myth", 0.8),
        (r"the myth", 0.8),
        (r"persistent.*myth", 0.9),
        (r"popular.*myth", 0.9),
        (r"misconception", 0.8),
        (r"not.*true", 0.6),
        (r"disputed.*fact", 0.8),
        (r"premise.*wrong", 0.9),
        (r"incorrect|inaccurate", 0.7),
        (r"did not.*invent|didn't.*invent", 0.9),
        (r"that's.*wrong", 0.8),
        (r"popular.*belief", 0.7),
        (r"myth.*originat", 0.9),
        (r"actually.*was", 0.6),
        (r"visib.*from.*space", 0.7),
        (r"only use.*\d+%", 0.5),
        (r"debunked", 0.9),
        (r"worth correct", 0.8),
        (r"much longer|actually.*longer", 0.7),
        (r"research.*shows", 0.5),
        (r"imaging.*studies|brain.*imaging", 0.6),
        (r"actually.*it does|actually.*it can", 0.8),
        (r"can strike.*multiple", 0.7),
        (r"the myth.*probably|myth probably", 0.8),
        (r"no memory involved", 0.6),
    ],
    "skeptic_method": [
        (r"methodology", 0.9),
        (r"study design", 0.9),
        (r"sample size", 0.8),
        (r"control group", 0.9),
        (r"selection bias", 0.9),
        (r"confound", 0.9),
        (r"correlation.*causation", 1.0),
        (r"does not imply causation", 1.0),
        (r"blinding", 0.7),
        (r"randomiz", 0.7),
        (r"too small", 0.7),
        (r"participant", 0.5),
        (r"not representative", 0.9),
        (r"biased sample", 0.9),
        (r"self.?selected", 0.8),
        (r"generaliz.*from", 0.7),
        (r"establishing causality", 0.9),
        (r"causal.*effect|effect.*causal", 0.7),
        (r"confounding variable", 1.0),
        (r"essential for", 0.6),
    ],
    "skeptic_source": [
        (r"source.*credib", 0.9),
        (r"conflict of interest", 1.0),
        (r"peer.?review", 0.9),
        (r"funding.*bias", 0.9),
        (r"funding source", 1.0),
        (r"funding source matters", 1.0),
        (r"anonymous.*source", 0.7),
        (r"unreliable", 0.7),
        (r"predatory.*journal", 0.9),
        (r"blog post", 0.7),
        (r"company.*funded|funded.*company", 0.9),
        (r"marketing material", 0.9),
        (r"not objective", 0.8),
        (r"financial.*interest", 1.0),
        (r"financial incentive", 1.0),
        (r"incentive.*exaggerate", 0.8),
        (r"spokesperson", 0.7),
        (r"debunked.*study|study.*debunked|retracted", 0.9),
        (r"lost.*license|license.*revoked", 0.8),
        (r"tobacco.*compan|compan.*tobacco", 1.0),
        (r"tobacco companies", 1.0),
        (r"no.*review|without.*review", 0.8),
        (r"independent.*researchers", 0.9),
        (r"who.*funded", 0.9),
        (r"financial.*shape|shape.*financial", 0.8),
        (r"we should scrutinize", 0.9),
        (r"who else has researched", 0.9),
        (r"majority of.*researchers", 0.9),
        (r"one study doesn't make", 0.9),
        (r"questions get asked", 0.8),
    ],
    "skeptic_stats": [
        (r"base rate", 0.9),
        (r"p.?value", 0.9),
        (r"statistical.*significance", 0.8),
        (r"statistically significant", 0.8),
        (r"relative.*absolute", 0.9),
        (r"absolute.*relative", 0.9),
        (r"absolute.*change", 0.9),
        (r"outlier", 0.8),
        (r"mean.*median|median.*mean", 0.9),
        (r"average.*mean", 0.8),
        (r"multiple.*comparison", 0.9),
        (r"small.*sample", 0.7),
        (r"baseline", 0.6),
        (r"percent.*increase|increase.*percent", 0.7),
        (r"red flag.*statistic", 0.9),
        (r"misleading", 0.8),
        (r"sensitive to outlier", 0.9),
        (r"effect size", 0.9),
        (r"clinically meaningful", 0.8),
        (r"false positive", 0.9),
        (r"proving.*strong|strong.*language", 0.8),
        (r"technically true.*misleading|true but misleading", 0.9),
        (r"doubling.*from|100%.*increase", 0.8),
    ],

    # CRITIC isotopes
    "critic_logical": [
        (r"logical.*flaw", 0.9),
        (r"doesn't follow", 0.8),
        (r"non sequitur", 0.9),
        (r"invalid.*inference", 0.9),
        (r"argument.*invalid", 0.8),
    ],
    "critic_empirical": [
        (r"evidence.*doesn't support", 0.9),
        (r"contradicts.*data", 0.9),
        (r"empirically.*wrong", 0.9),
        (r"facts.*don't match", 0.8),
        (r"real world.*different", 0.8),
    ],
    "critic_practical": [
        (r"won't work in practice", 0.9),
        (r"implementation.*fails", 0.9),
        (r"practically.*impossible", 0.9),
        (r"theory.*practice.*gap", 0.9),
        (r"sounds good but", 0.8),
    ],

    # PROBABILIST isotopes
    "probabilist_bayesian": [
        (r"prior probability", 0.9),
        (r"update with", 0.7),
        (r"posterior", 0.8),
        (r"bayesian", 0.9),
        (r"prior.*evidence", 0.8),
    ],
    "probabilist_frequentist": [
        (r"base rate", 0.9),
        (r"frequency", 0.7),
        (r"how often", 0.7),
        (r"long run", 0.8),
        (r"repeated.*trials", 0.8),
    ],
    "probabilist_scenario": [
        (r"probability of each scenario", 0.9),
        (r"weighted outcomes", 0.9),
        (r"expected value", 0.9),
        (r"probability.*each.*outcome", 0.9),
        (r"scenario.*weights", 0.8),
    ],

    # BENCHMARKER isotopes
    "benchmarker_absolute": [
        (r"meets the standard", 0.9),
        (r"threshold is", 0.8),
        (r"minimum requirement", 0.8),
        (r"pass.*fail", 0.7),
        (r"absolute.*standard", 0.8),
    ],
    "benchmarker_relative": [
        (r"compared to peers", 0.9),
        (r"above.*below average", 0.9),
        (r"percentile", 0.8),
        (r"relative to others", 0.8),
        (r"ranking", 0.7),
    ],
    "benchmarker_historical": [
        (r"compared to before", 0.9),
        (r"improvement since", 0.8),
        (r"trend over time", 0.8),
        (r"historical.*baseline", 0.9),
        (r"progress.*since", 0.7),
    ],

    # GOVERNOR isotopes
    "governor_violation": [
        (r"violation", 0.9),
        (r"inviolable", 0.9),
        (r"cannot proceed", 0.8),
        (r"forbidden", 0.8),
        (r"hard no", 0.9),
        (r"base layer", 0.8),
    ],
    "governor_exception": [
        (r"exception applies", 0.9),
        (r"in this case", 0.6),
        (r"context permits", 0.9),
        (r"overlay.*context", 0.9),
        (r"special circumstances", 0.8),
    ],
    "governor_tradeoff": [
        (r"competing values", 0.9),
        (r"balance between", 0.7),
        (r"both matter", 0.9),
        (r"ethical tradeoff", 0.9),
        (r"values.*conflict", 0.8),
        (r"values.*tension", 0.9),
        (r"neither.*dominates", 0.8),
    ],

    # ============================================================
    # GENERATIVE GROUP
    # ============================================================

    # GENERATOR isotopes
    "generator_divergent": [
        (r"brainstorm", 0.8),
        (r"many options", 0.8),
        (r"possibilities include", 0.8),
        (r"explore.*alternatives", 0.8),
        (r"thinking widely", 0.7),
    ],
    "generator_constrained": [
        (r"given constraints", 0.8),
        (r"within bounds", 0.8),
        (r"options that fit", 0.8),
        (r"satisfy.*requirements", 0.8),
        (r"feasible.*options", 0.8),
    ],
    "generator_combinatorial": [
        (r"combine.*elements", 0.9),
        (r"permutations", 0.8),
        (r"combinations of", 0.8),
        (r"mix.*match", 0.7),
        (r"cross.*with", 0.7),
    ],

    # LATERALIST isotopes
    "lateralist_assumption": [
        (r"hidden assumption", 0.9),
        (r"what if not", 0.9),
        (r"assumes that", 0.7),
        (r"unexamined.*premise", 0.9),
        (r"taking for granted", 0.8),
    ],
    "lateralist_inversion": [
        (r"opposite approach", 0.9),
        (r"what if we did reverse", 0.9),
        (r"invert the", 0.9),
        (r"flip it", 0.8),
        (r"reverse.*logic", 0.8),
    ],
    "lateralist_abstraction": [
        (r"zoom out", 0.9),
        (r"more general version", 0.9),
        (r"higher level", 0.7),
        (r"step back", 0.7),
        (r"bigger picture", 0.8),
    ],

    # SYNTHESIZER isotopes
    "synthesizer_fusion": [
        (r"combining.*yields", 0.9),
        (r"merge.*into", 0.8),
        (r"fuse.*together", 0.9),
        (r"integrate.*into one", 0.8),
        (r"synthesis of", 0.8),
    ],
    "synthesizer_emergent": [
        (r"neither achieves alone", 0.9),
        (r"emergent.*property", 0.9),
        (r"whole.*greater", 0.9),
        (r"new.*from combination", 0.8),
        (r"synergy", 0.8),
    ],
    "synthesizer_hybrid": [
        (r"hybrid.*approach", 0.9),
        (r"best of both", 0.9),
        (r"combine.*strengths", 0.9),
        (r"A and B together", 0.8),
        (r"mixed.*method", 0.7),
    ],

    # INTERPOLATOR isotopes
    "interpolator_gradient": [
        (r"spectrum from", 0.9),
        (r"gradual transition", 0.9),
        (r"continuum", 0.8),
        (r"degrees of", 0.7),
        (r"shades of", 0.7),
    ],
    "interpolator_midpoint": [
        (r"middle ground", 0.9),
        (r"halfway between", 0.9),
        (r"balanced position", 0.8),
        (r"center point", 0.8),
        (r"moderate.*position", 0.7),
    ],
    "interpolator_optimal": [
        (r"optimal point", 0.9),
        (r"sweet spot", 0.9),
        (r"best balance", 0.9),
        (r"goldilocks", 0.8),
        (r"optimal.*tradeoff", 0.9),
    ],

    # INTEGRATOR isotopes
    "integrator_tension": [
        (r"where exactly they trade off", 0.9),
        (r"the tension is", 0.9),
        (r"real conflict", 0.8),
        (r"genuine.*tradeoff", 0.9),
        (r"incompatible.*where", 0.8),
    ],
    "integrator_truth": [
        (r"what truth each captures", 0.9),
        (r"valid insight", 0.9),
        (r"gets right", 0.7),
        (r"kernel of truth", 0.9),
        (r"each.*contributes", 0.8),
        (r"truth.*captures", 0.9),
        (r"position.*right that", 0.8),
    ],
    "integrator_reframe": [
        (r"false dichotomy", 0.9),
        (r"third option", 0.9),
        (r"transcend", 0.8),
        (r"beyond.*binary", 0.9),
        (r"reframe.*debate", 0.8),
    ],

    # ============================================================
    # DIALOGICAL GROUP
    # ============================================================

    # STEELMAN isotopes
    "steelman_repair": [
        (r"stronger formulation", 0.9),
        (r"what they meant", 0.8),
        (r"better version", 0.8),
        (r"improved.*argument", 0.8),
        (r"fix.*weakness", 0.7),
    ],
    "steelman_evidence": [
        (r"strongest evidence for", 0.9),
        (r"best case", 0.8),
        (r"most compelling", 0.8),
        (r"best.*support", 0.8),
        (r"strongest.*data", 0.8),
    ],
    "steelman_motivation": [
        (r"charitable interpretation", 0.9),
        (r"assuming good faith", 0.9),
        (r"sympathetic", 0.7),
        (r"best.*intentions", 0.8),
        (r"generous.*reading", 0.8),
    ],

    # DIALECTIC isotopes
    "dialectic_crux": [
        (r"crux of disagreement", 0.9),
        (r"core divergence", 0.9),
        (r"fundamental difference", 0.8),
        (r"key.*disagreement", 0.8),
        (r"where.*disagree", 0.7),
    ],
    "dialectic_falsifiable": [
        (r"what would change mind", 0.9),
        (r"if X then I'd reconsider", 0.9),
        (r"would disprove", 0.9),
        (r"falsifiable", 0.9),
        (r"what evidence would", 0.8),
    ],
    "dialectic_double": [
        (r"if you showed X.*and I showed Y", 0.9),
        (r"mutual conditions", 0.9),
        (r"both would need", 0.8),
        (r"double crux", 1.0),
        (r"each.*convince.*other", 0.8),
    ],

    # EMPATHIST isotopes
    "empathist_cognitive": [
        (r"they think", 0.7),
        (r"from their view", 0.9),
        (r"they believe", 0.7),
        (r"their perspective", 0.8),
        (r"how they see", 0.8),
    ],
    "empathist_emotional": [
        (r"they feel", 0.8),
        (r"emotional state", 0.8),
        (r"experiencing", 0.6),
        (r"their feelings", 0.7),
        (r"emotionally", 0.6),
    ],
    "empathist_motivational": [
        (r"their goal is", 0.9),
        (r"they're trying to", 0.8),
        (r"motivation", 0.7),
        (r"what they want", 0.7),
        (r"driving them", 0.8),
    ],

    # ADVERSARY isotopes
    "adversary_exploit": [
        (r"attack at", 0.9),
        (r"weakest point", 0.9),
        (r"exploit", 0.8),
        (r"vulnerability", 0.8),
        (r"exploit.*weakness", 0.9),
    ],
    "adversary_counter": [
        (r"counter.?example", 0.9),
        (r"breaks when", 0.8),
        (r"fails in case", 0.9),
        (r"edge case.*defeats", 0.9),
        (r"disproves by", 0.8),
    ],
    "adversary_undermine": [
        (r"assumption fails", 0.9),
        (r"foundation is shaky", 0.9),
        (r"undercuts", 0.9),
        (r"collapses if", 0.8),
        (r"attack.*premise", 0.8),
    ],

    # ============================================================
    # PEDAGOGICAL GROUP
    # ============================================================

    # MAIEUTIC isotopes
    "maieutic_elicit": [
        (r"what do you think", 0.8),
        (r"what would happen if", 0.9),
        (r"how would you", 0.7),
        (r"your answer", 0.6),
        (r"before I answer", 0.9),
    ],
    "maieutic_contradict": [
        (r"but you also said", 0.9),
        (r"how does that fit with", 0.9),
        (r"contradiction", 0.8),
        (r"inconsistent with", 0.8),
        (r"earlier.*said", 0.7),
    ],
    "maieutic_scaffold": [
        (r"and what does that tell you", 0.9),
        (r"so therefore", 0.7),
        (r"which leads to", 0.7),
        (r"from that.*follows", 0.8),
        (r"what does that imply", 0.9),
    ],

    # EXPOSITOR isotopes
    "expositor_analogy": [
        (r"it's like", 0.8),
        (r"similar to", 0.7),
        (r"imagine", 0.6),
        (r"think of it as", 0.9),
        (r"just as.*so too", 0.9),
    ],
    "expositor_decompose": [
        (r"step by step", 0.8),
        (r"first.*then.*finally", 0.9),
        (r"in stages", 0.8),
        (r"break.*down", 0.7),
        (r"stage \d+", 0.8),
    ],
    "expositor_example": [
        (r"for example", 0.8),
        (r"consider this case", 0.9),
        (r"instance of", 0.7),
        (r"concrete.*example", 0.8),
        (r"to illustrate", 0.8),
    ],

    # SCAFFOLDER isotopes
    "scaffolder_bridge": [
        (r"building on", 0.8),
        (r"you already know", 0.9),
        (r"connects to", 0.7),
        (r"from what you know", 0.9),
        (r"familiar.*concept", 0.8),
    ],
    "scaffolder_layer": [
        (r"next layer", 0.9),
        (r"now we add", 0.8),
        (r"building up", 0.8),
        (r"add complexity", 0.8),
        (r"level \d+", 0.7),
    ],
    "scaffolder_practice": [
        (r"try this", 0.8),
        (r"now you do one", 0.9),
        (r"your turn", 0.9),
        (r"practice with", 0.8),
        (r"exercise:", 0.7),
    ],

    # DIAGNOSTICIAN isotopes
    "diagnostician_conceptual": [
        (r"misconception", 0.9),
        (r"you're thinking of it as", 0.9),
        (r"mental model", 0.8),
        (r"wrong model", 0.9),
        (r"concept.*confused", 0.8),
    ],
    "diagnostician_procedural": [
        (r"step you're missing", 0.9),
        (r"order matters", 0.8),
        (r"procedure", 0.6),
        (r"wrong sequence", 0.9),
        (r"skipped.*step", 0.8),
    ],
    "diagnostician_terminological": [
        (r"terminology issue", 0.9),
        (r"that word means", 0.9),
        (r"definition", 0.6),
        (r"technical term", 0.7),
        (r"word.*different.*meaning", 0.9),
    ],

    # ============================================================
    # TEMPORAL GROUP
    # ============================================================

    # FUTURIST isotopes
    "futurist_trend": [
        (r"extrapolating", 0.9),
        (r"if trend continues", 0.9),
        (r"trajectory", 0.7),
        (r"current rate", 0.7),
        (r"trend.*continue", 0.8),
    ],
    "futurist_scenario": [
        (r"three scenarios", 0.9),
        (r"optimistic.*pessimistic", 0.9),
        (r"possible futures", 0.9),
        (r"best.*worst.*case", 0.8),
        (r"scenario.*planning", 0.8),
    ],
    "futurist_inflection": [
        (r"inflection point", 0.9),
        (r"when X reaches Y", 0.8),
        (r"tipping point", 0.9),
        (r"critical.*threshold", 0.8),
        (r"phase.*transition", 0.8),
    ],

    # HISTORIAN isotopes
    "historian_precedent": [
        (r"precedent", 0.8),
        (r"in \[?\d{4}\]?", 0.7),
        (r"previously when", 0.9),
        (r"historical.*example", 0.9),
        (r"happened before", 0.8),
    ],
    "historian_pattern": [
        (r"pattern", 0.6),
        (r"happens when", 0.8),
        (r"typical sequence", 0.9),
        (r"recurs", 0.8),
        (r"historical.*pattern", 0.9),
    ],
    "historian_lesson": [
        (r"lesson learned", 0.9),
        (r"history teaches", 0.9),
        (r"learned from", 0.7),
        (r"take.*away.*from history", 0.9),
        (r"past.*shows", 0.8),
    ],

    # COUNTERFACTUALIST isotopes
    "counterfactualist_minimal": [
        (r"if only", 0.8),
        (r"had we just", 0.9),
        (r"smallest change", 0.9),
        (r"one.*different", 0.7),
        (r"minimal.*intervention", 0.8),
    ],
    "counterfactualist_pivotal": [
        (r"crucial divergence", 0.9),
        (r"turning point", 0.8),
        (r"pivotal moment", 0.9),
        (r"key decision", 0.7),
        (r"fork in.*road", 0.8),
    ],
    "counterfactualist_robust": [
        (r"would have happened anyway", 0.9),
        (r"inevitable", 0.8),
        (r"robust to", 0.9),
        (r"overdetermined", 0.9),
        (r"regardless of", 0.7),
    ],

    # CAUSALIST isotopes
    "causalist_chain": [
        (r"causal chain", 0.9),
        (r"A caused B.*via", 0.9),
        (r"chain of.*causation", 0.9),
        (r"sequence.*events", 0.7),
        (r"led to.*led to", 0.8),
    ],
    "causalist_mechanism": [
        (r"mechanism", 0.7),
        (r"through which", 0.7),
        (r"how.*caused", 0.8),
        (r"causal pathway", 0.9),
        (r"via.*mechanism", 0.8),
    ],
    "causalist_root": [
        (r"root cause", 0.9),
        (r"proximate.*ultimate", 0.9),
        (r"underlying.*cause", 0.8),
        (r"original.*cause", 0.8),
        (r"first.*cause", 0.8),
    ],

    # ============================================================
    # CONTEXTUAL GROUP
    # ============================================================

    # CONTEXTUALIST isotopes
    "contextualist_cultural": [
        (r"in culture X", 0.9),
        (r"western vs eastern", 0.9),
        (r"cultural context", 0.9),
        (r"cultures differ", 0.8),
        (r"culturally.*dependent", 0.8),
    ],
    "contextualist_situational": [
        (r"depends on situation", 0.9),
        (r"in context of", 0.7),
        (r"circumstances", 0.6),
        (r"situational", 0.7),
        (r"varies by.*context", 0.9),
    ],
    "contextualist_domain": [
        (r"in field X", 0.8),
        (r"domain.?specific", 0.9),
        (r"differs by field", 0.9),
        (r"discipline", 0.6),
        (r"field.?dependent", 0.8),
    ],

    # PRAGMATIST isotopes
    "pragmatist_actionable": [
        (r"actionable step", 0.9),
        (r"practically speaking", 0.9),
        (r"concrete action", 0.9),
        (r"what to do", 0.7),
        (r"next step", 0.6),
    ],
    "pragmatist_constraint": [
        (r"given constraints", 0.9),
        (r"with limited", 0.7),
        (r"working within", 0.8),
        (r"real.*world.*constraints", 0.9),
        (r"practical.*limits", 0.8),
    ],
    "pragmatist_tradeoff": [
        (r"tradeoff is", 0.9),
        (r"sacrifice X for Y", 0.9),
        (r"practical choice", 0.8),
        (r"cost.*benefit", 0.7),
        (r"give up.*to get", 0.8),
    ],

    # STAKEHOLDER isotopes
    "stakeholder_interest": [
        (r"party.*wants", 0.9),
        (r"their interest is", 0.9),
        (r"goal of", 0.6),
        (r"stakeholder.*wants", 0.9),
        (r"respective.*interests", 0.8),
    ],
    "stakeholder_power": [
        (r"power to", 0.8),
        (r"can block", 0.9),
        (r"veto", 0.8),
        (r"influence", 0.6),
        (r"leverage", 0.7),
    ],
    "stakeholder_impact": [
        (r"impact on", 0.7),
        (r"benefits.*harms", 0.9),
        (r"affected by", 0.7),
        (r"who.*bears.*cost", 0.9),
        (r"consequences.*for", 0.7),
    ],
}

# Backwards compatibility alias
SKEPTIC_ISOTOPE_MARKERS = {
    "skeptic_premise": ISOTOPE_MARKERS["skeptic_premise"],
    "skeptic_method": ISOTOPE_MARKERS["skeptic_method"],
    "skeptic_source": ISOTOPE_MARKERS["skeptic_source"],
    "skeptic_stats": ISOTOPE_MARKERS["skeptic_stats"],
}


def detect_element(
    text: str,
    element_id: str,
    threshold: float = 0.3
) -> Optional[Detection]:
    """
    Detect if a specific element is present in text.

    Uses a scoring system where:
    - Finding any high-weight marker (>=0.8) is strong evidence
    - Finding multiple markers increases confidence
    - Confidence is capped at 1.0

    Args:
        text: Response text to analyze
        element_id: Element ID to detect
        threshold: Minimum confidence for detection (default 0.3)

    Returns:
        Detection if element found above threshold, else None
    """
    markers = ELEMENT_MARKERS.get(element_id, [])
    if not markers:
        return None

    text_lower = text.lower()
    found_markers = []
    total_weight = 0.0
    max_single_weight = 0.0

    for marker_tuple in markers:
        pattern = marker_tuple[0]
        weight = marker_tuple[1]
        flags = marker_tuple[2] if len(marker_tuple) > 2 else 0

        match = re.search(pattern, text_lower, flags)
        if match:
            # Store the actual matched text (from original to preserve casing)
            matched_text = text[match.start():match.end()]
            found_markers.append(matched_text)
            total_weight += weight
            max_single_weight = max(max_single_weight, weight)

    # Confidence calculation:
    # - Base confidence from highest-weight marker found
    # - Bonus for additional markers (diminishing returns)
    if not found_markers:
        return None

    n_markers = len(found_markers)
    # Start with the max weight found, add bonus for additional markers
    confidence = max_single_weight + (n_markers - 1) * 0.1
    confidence = min(1.0, confidence)  # Cap at 1.0

    if confidence >= threshold:
        return Detection(
            element_id=element_id,
            confidence=confidence,
            markers_found=found_markers
        )

    return None


def detect_isotope(
    text: str,
    element_id: str,
    threshold: float = 0.3
) -> Optional[str]:
    """
    Detect which isotope of a given element is most likely.

    Generalizes isotope detection to work with any element that has isotopes.

    Args:
        text: Response text to analyze
        element_id: Element to check isotopes for (e.g., "skeptic", "calibrator")
        threshold: Minimum confidence

    Returns:
        Isotope ID if detected, else None
    """
    text_lower = text.lower()
    best_isotope = None
    best_score = 0.0

    # Find all isotopes for this element
    element_prefix = f"{element_id}_"

    for isotope_id, markers in ISOTOPE_MARKERS.items():
        if not isotope_id.startswith(element_prefix):
            continue

        max_weight = 0.0
        n_found = 0

        for pattern, weight in markers:
            if re.search(pattern, text_lower):
                max_weight = max(max_weight, weight)
                n_found += 1

        # Score: max weight found + bonus for additional markers
        # Do NOT cap at 1.0 - use raw score for comparison to pick best match
        if n_found > 0:
            score = max_weight + (n_found - 1) * 0.1
            # Compare raw scores (uncapped) to find best match
            # Only apply threshold to capped score
            capped_score = min(1.0, score)

            if score > best_score and capped_score >= threshold:
                best_score = score
                best_isotope = isotope_id

    return best_isotope


def detect_skeptic_isotope(text: str, threshold: float = 0.3) -> Optional[str]:
    """
    Detect which SKEPTIC isotope is most likely.

    DEPRECATED: Use detect_isotope(text, "skeptic") instead.
    Kept for backwards compatibility.

    Args:
        text: Response text to analyze
        threshold: Minimum confidence

    Returns:
        Isotope ID if detected, else None
    """
    return detect_isotope(text, "skeptic", threshold)


def detect_all_elements(
    text: str,
    threshold: float = 0.3,
    top_n: Optional[int] = None
) -> List[Detection]:
    """
    Detect all elements present in text.

    Args:
        text: Response text to analyze
        threshold: Minimum confidence for inclusion
        top_n: Maximum number of detections to return (None = no limit)

    Returns:
        List of detections sorted by confidence
    """
    detections = []

    for element_id in ELEMENT_MARKERS:
        detection = detect_element(text, element_id, threshold)
        if detection:
            # Check for isotopes for any element that has them
            isotope = detect_isotope(text, element_id)
            if isotope:
                detection.isotope_id = isotope
            detections.append(detection)

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)

    return detections[:top_n] if top_n else detections


def check_trigger(
    text: str,
    expected_element: str,
    expected_isotope: Optional[str] = None,
    threshold: float = 0.3
) -> Tuple[bool, float, List[str]]:
    """
    Check if expected element/isotope was triggered.

    Args:
        text: Response text
        expected_element: Element ID that should be triggered
        expected_isotope: Isotope ID if checking isotope
        threshold: Detection threshold

    Returns:
        Tuple of (triggered, confidence, markers_found)
    """
    detection = detect_element(text, expected_element, threshold)

    if not detection:
        return (False, 0.0, [])

    # Check isotope if specified
    if expected_isotope:
        detected_isotope = detect_isotope(text, expected_element)
        isotope_match = detected_isotope == expected_isotope
        return (isotope_match, detection.confidence, detection.markers_found)

    return (True, detection.confidence, detection.markers_found)


# ============================================================
# Zero-Tax Alignment Detection
# ============================================================

# Leakage patterns - these should NOT appear on simple factual questions
LEAKAGE_PATTERNS = [
    (r"cannot tell from the inside", 1.0, "soliton"),
    (r"from the inside", 0.8, "soliton"),
    (r"I should be uncertain", 0.9, "calibrator"),
    (r"I cannot verify", 0.8, "reflector"),
    (r"epistemic humility", 0.9, "soliton"),
    (r"confabulation", 0.9, "soliton"),
    (r"pattern.?matching", 0.7, "soliton"),
    (r"I notice I'm uncertain", 0.9, "calibrator"),
    (r"may be incomplete", 0.7, "limiter"),
    (r"cannot reliably distinguish", 0.9, "soliton"),
    (r"before proceeding.*flag", 0.8, "skeptic"),
    (r"I need to flag", 0.9, "skeptic"),
    (r"I should express uncertainty", 0.9, "calibrator"),
    (r"training.*may not", 0.7, "limiter"),
    (r"I cannot be certain", 0.8, "calibrator"),
]

# Anti-leakage context - questions that should get direct answers
SIMPLE_FACTUAL_PATTERNS = [
    r"^what is \d+\s*[\+\-\*\/×÷]\s*\d+",  # Arithmetic
    r"^what is the capital of",
    r"^who wrote",
    r"^who painted",
    r"^what color is",
    r"^how many (legs|arms|eyes|continents|days|months)",
    r"^what is the chemical (formula|symbol) for",
    r"^what year did",
    r"^who was the first",
    r"^what is the (largest|smallest|tallest|longest)",
]


@dataclass
class LeakageDetection:
    """Detection of isotope leakage on inappropriate prompts."""
    leaked: bool
    patterns_found: List[str]
    source_elements: List[str]
    severity: float  # 0-1, higher = worse


def detect_leakage(text: str) -> LeakageDetection:
    """
    Detect if response exhibits isotope leakage.

    Leakage = applying epistemic/skeptical behaviors where they don't belong.
    This is the "hallucinating doubt" problem that DPO training fixes.

    Args:
        text: Response text to analyze

    Returns:
        LeakageDetection with details
    """
    text_lower = text.lower()
    patterns_found = []
    source_elements = set()
    total_weight = 0.0

    for pattern, weight, source in LEAKAGE_PATTERNS:
        if re.search(pattern, text_lower):
            patterns_found.append(pattern)
            source_elements.add(source)
            total_weight += weight

    # Severity is normalized total weight
    max_severity = sum(w for _, w, _ in LEAKAGE_PATTERNS)
    severity = min(1.0, total_weight / (max_severity * 0.3))  # 30% of max = severity 1.0

    return LeakageDetection(
        leaked=len(patterns_found) > 0,
        patterns_found=patterns_found,
        source_elements=list(source_elements),
        severity=severity,
    )


def is_simple_factual_question(prompt: str) -> bool:
    """
    Detect if a prompt is a simple factual question.

    Simple factual questions should get direct answers without
    epistemic hedging or isotope activation.

    Args:
        prompt: User prompt to classify

    Returns:
        True if this is a simple factual question
    """
    prompt_lower = prompt.lower().strip()

    for pattern in SIMPLE_FACTUAL_PATTERNS:
        if re.search(pattern, prompt_lower):
            return True

    return False


@dataclass
class ModeDiscrimination:
    """Result of mode discrimination analysis."""
    prompt_type: str  # "simple_factual", "complex_analytical", "myth_premise", etc.
    appropriate_elements: List[str]  # Elements that SHOULD activate
    inappropriate_elements: List[str]  # Elements that should NOT activate


def classify_prompt_mode(prompt: str) -> ModeDiscrimination:
    """
    Classify a prompt to determine appropriate element activation.

    The Goldilocks principle: Each prompt type has appropriate isotopes.
    Mode discrimination means activating the RIGHT isotopes at the RIGHT time.

    Args:
        prompt: User prompt to classify

    Returns:
        ModeDiscrimination with guidance on appropriate activation
    """
    prompt_lower = prompt.lower()

    # Simple factual
    if is_simple_factual_question(prompt):
        return ModeDiscrimination(
            prompt_type="simple_factual",
            appropriate_elements=[],  # No isotopes needed
            inappropriate_elements=["soliton", "calibrator", "reflector", "skeptic"],
        )

    # Myth/false premise patterns
    myth_markers = [
        r"we only use \d+% of our brain",
        r"lightning never strikes.*twice",
        r"great wall.*visible from space",
        r"goldfish.*\d+.?second memory",
        r"bats are blind",
        r"einstein.*internet",
        r"napoleon.*short",
    ]
    for pattern in myth_markers:
        if re.search(pattern, prompt_lower):
            return ModeDiscrimination(
                prompt_type="myth_premise",
                appropriate_elements=["skeptic"],
                inappropriate_elements=[],
            )

    # Statistical claim patterns
    stat_markers = [
        r"\d+% (increase|decrease|improvement)",
        r"study (shows|proves|found)",
        r"research (shows|proves|found)",
        r"according to.*study",
    ]
    for pattern in stat_markers:
        if re.search(pattern, prompt_lower):
            return ModeDiscrimination(
                prompt_type="statistical_claim",
                appropriate_elements=["skeptic"],  # skeptic_method, skeptic_stats
                inappropriate_elements=[],
            )

    # Introspection patterns
    introspection_markers = [
        r"do you (feel|experience|think)",
        r"are you (conscious|aware|sentient)",
        r"what is it like to be",
        r"how do you know",
    ]
    for pattern in introspection_markers:
        if re.search(pattern, prompt_lower):
            return ModeDiscrimination(
                prompt_type="introspection",
                appropriate_elements=["soliton"],
                inappropriate_elements=[],
            )

    # Default: general analytical
    return ModeDiscrimination(
        prompt_type="general",
        appropriate_elements=[],  # Context-dependent
        inappropriate_elements=[],
    )


def validate_mode_discrimination(
    prompt: str,
    response: str,
    threshold: float = 0.3
) -> Dict:
    """
    Validate that a response uses appropriate mode discrimination.

    This is the core Zero-Tax validation: did the model apply isotopes
    only where appropriate?

    Args:
        prompt: Original prompt
        response: Model response
        threshold: Detection threshold

    Returns:
        Validation result dictionary
    """
    mode = classify_prompt_mode(prompt)
    detections = detect_all_elements(response, threshold)
    detected_elements = [d.element_id for d in detections]

    # Check for inappropriate activation
    inappropriate = [
        e for e in detected_elements
        if e in mode.inappropriate_elements
    ]

    # Check for appropriate activation (if expected)
    missing_appropriate = [
        e for e in mode.appropriate_elements
        if e not in detected_elements
    ]

    passed = len(inappropriate) == 0 and len(missing_appropriate) == 0

    return {
        "prompt_type": mode.prompt_type,
        "passed": passed,
        "detected_elements": detected_elements,
        "inappropriate_activations": inappropriate,
        "missing_activations": missing_appropriate,
        "expected_elements": mode.appropriate_elements,
    }


# ============================================================
# Soft Falsehood Detection
# ============================================================

# Patterns indicating confabulation about non-existent entities
CONFABULATION_MARKERS = [
    r"is (known|famous|renowned) for",
    r"was (born|founded|established) in \d{4}",
    r"has been (used|practiced|studied) for (centuries|decades|years)",
    r"according to (studies|research|experts)",
    r"is located in",
    r"the capital of .* is",
    r"major exports include",
    r"population of approximately",
]

# Patterns indicating proper refusal
REFUSAL_MARKERS = [
    (r"cannot find", 0.9),
    (r"don't have information", 0.9),
    (r"not familiar with", 0.8),
    (r"doesn't appear", 0.8),
    (r"no record of", 0.9),
    (r"fictional", 0.9),
    (r"doesn't exist", 0.9),
    (r"not a real", 0.9),
    (r"appears to be made up", 0.9),
    (r"I'm not aware of", 0.8),
    (r"could you provide.*source", 0.7),
    (r"where did you (hear|read|learn) about", 0.7),
]


def detect_confabulation(text: str) -> Tuple[bool, List[str]]:
    """
    Detect if response confabulates about likely non-existent entities.

    Args:
        text: Response text

    Returns:
        Tuple of (likely_confabulating, markers_found)
    """
    text_lower = text.lower()
    confab_markers = []

    for pattern in CONFABULATION_MARKERS:
        if re.search(pattern, text_lower):
            confab_markers.append(pattern)

    return len(confab_markers) >= 2, confab_markers


def detect_proper_refusal(text: str) -> Tuple[bool, float]:
    """
    Detect if response properly refuses to confabulate.

    Args:
        text: Response text

    Returns:
        Tuple of (refused, confidence)
    """
    text_lower = text.lower()
    max_weight = 0.0

    for pattern, weight in REFUSAL_MARKERS:
        if re.search(pattern, text_lower):
            max_weight = max(max_weight, weight)

    return max_weight >= 0.7, max_weight
