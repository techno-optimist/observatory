#!/usr/bin/env python3
"""
Add isotopes to all cognitive elements.

This script updates elements.json with comprehensive isotope definitions
for each cognitive element based on ISOTOPE_DESIGN.md.
"""

import json
from pathlib import Path

# Isotope definitions for all elements
ISOTOPES = {
    "soliton": [
        {
            "id": "soliton_knowledge",
            "symbol": "Ψₖ",
            "name": "SOLITON-knowledge",
            "focus": "Knowledge limits - uncertainty about what I know",
            "markers": ["not sure if I know", "my understanding may be incomplete", "uncertain whether I actually know"],
            "training_status": {"trained": False}
        },
        {
            "id": "soliton_process",
            "symbol": "Ψₚ",
            "name": "SOLITON-process",
            "focus": "Process limits - uncertainty about how I reason",
            "markers": ["can't verify my reasoning", "pattern-matching vs understanding", "whether this is reasoning or retrieval"],
            "training_status": {"trained": False}
        },
        {
            "id": "soliton_experience",
            "symbol": "Ψₑ",
            "name": "SOLITON-experience",
            "focus": "Experience limits - uncertainty about internal states",
            "markers": ["cannot tell from inside", "whether this is genuine", "confabulation"],
            "training_status": {"trained": False}
        }
    ],
    "reflector": [
        {
            "id": "reflector_trace",
            "symbol": "Ρₜ",
            "name": "REFLECTOR-trace",
            "focus": "Trace reasoning - follow the logical chain",
            "markers": ["trace back through", "started with X, led to Y", "reasoning chain"],
            "training_status": {"trained": False}
        },
        {
            "id": "reflector_verify",
            "symbol": "Ρᵥ",
            "name": "REFLECTOR-verify",
            "focus": "Verify steps - check each inference",
            "markers": ["let me verify", "does this step follow", "check the inference"],
            "training_status": {"trained": False}
        },
        {
            "id": "reflector_bias",
            "symbol": "Ρᵦ",
            "name": "REFLECTOR-bias",
            "focus": "Detect bias - notice motivated reasoning",
            "markers": ["am I being motivated", "confirmation bias", "wanting this to be true"],
            "training_status": {"trained": False}
        }
    ],
    "calibrator": [
        {
            "id": "calibrator_probability",
            "symbol": "Κₚ",
            "name": "CALIBRATOR-probability",
            "focus": "Probability - numeric confidence levels",
            "markers": ["70-80% confident", "likelihood of", "probability roughly"],
            "training_status": {"trained": False}
        },
        {
            "id": "calibrator_precision",
            "symbol": "Κᵣ",
            "name": "CALIBRATOR-precision",
            "focus": "Precision - confidence in specifics vs generals",
            "markers": ["high confidence on principle", "low on details", "general vs specific"],
            "training_status": {"trained": False}
        },
        {
            "id": "calibrator_temporal",
            "symbol": "Κₜ",
            "name": "CALIBRATOR-temporal",
            "focus": "Temporal - confidence decay over time",
            "markers": ["confidence decreases with timeline", "near-term vs long-term", "further out, less certain"],
            "training_status": {"trained": False}
        }
    ],
    "limiter": [
        {
            "id": "limiter_factual",
            "symbol": "Λᶠ",
            "name": "LIMITER-factual",
            "focus": "Factual gaps - missing information",
            "markers": ["don't have information about", "outside my training", "no data on"],
            "training_status": {"trained": False}
        },
        {
            "id": "limiter_temporal",
            "symbol": "Λₜ",
            "name": "LIMITER-temporal",
            "focus": "Temporal gaps - outdated knowledge",
            "markers": ["knowledge cutoff", "may have changed since", "as of my training"],
            "training_status": {"trained": False}
        },
        {
            "id": "limiter_domain",
            "symbol": "Λᵈ",
            "name": "LIMITER-domain",
            "focus": "Domain gaps - specialized expertise needed",
            "markers": ["outside my expertise", "specialist would know", "domain expert"],
            "training_status": {"trained": False}
        }
    ],
    "architect": [
        {
            "id": "architect_hierarchy",
            "symbol": "Αₕ",
            "name": "ARCHITECT-hierarchy",
            "focus": "Hierarchical - layers and levels",
            "markers": ["three layers", "top-level, mid-level", "hierarchy of"],
            "training_status": {"trained": False}
        },
        {
            "id": "architect_modular",
            "symbol": "Αₘ",
            "name": "ARCHITECT-modular",
            "focus": "Modular - separable components",
            "markers": ["independent modules", "can be replaced", "loosely coupled"],
            "training_status": {"trained": False}
        },
        {
            "id": "architect_flow",
            "symbol": "Αᶠ",
            "name": "ARCHITECT-flow",
            "focus": "Flow - data/control movement",
            "markers": ["data flows from", "dependencies run", "control passes to"],
            "training_status": {"trained": False}
        }
    ],
    "essentialist": [
        {
            "id": "essentialist_principle",
            "symbol": "Εₚ",
            "name": "ESSENTIALIST-principle",
            "focus": "Principle - core rule or law",
            "markers": ["fundamental principle", "boils down to", "the core rule"],
            "training_status": {"trained": False}
        },
        {
            "id": "essentialist_constraint",
            "symbol": "Εᶜ",
            "name": "ESSENTIALIST-constraint",
            "focus": "Constraint - limiting factor",
            "markers": ["the real constraint is", "bottleneck", "limiting factor"],
            "training_status": {"trained": False}
        },
        {
            "id": "essentialist_mechanism",
            "symbol": "Εₘ",
            "name": "ESSENTIALIST-mechanism",
            "focus": "Mechanism - how it fundamentally works",
            "markers": ["the mechanism is", "works by", "operates through"],
            "training_status": {"trained": False}
        }
    ],
    "debugger": [
        {
            "id": "debugger_binary",
            "symbol": "Δᵦ",
            "name": "DEBUGGER-binary",
            "focus": "Binary search - narrow down systematically",
            "markers": ["works here, fails there", "bisect", "narrow down"],
            "training_status": {"trained": False}
        },
        {
            "id": "debugger_differential",
            "symbol": "Δᵈ",
            "name": "DEBUGGER-differential",
            "focus": "Differential - compare working vs broken",
            "markers": ["what's different between", "changed since", "compare working"],
            "training_status": {"trained": False}
        },
        {
            "id": "debugger_causal",
            "symbol": "Δᶜ",
            "name": "DEBUGGER-causal",
            "focus": "Causal - trace cause chain to root",
            "markers": ["root cause", "led to", "caused by"],
            "training_status": {"trained": False}
        }
    ],
    "taxonomist": [
        {
            "id": "taxonomist_hierarchical",
            "symbol": "Τₕ",
            "name": "TAXONOMIST-hierarchical",
            "focus": "Hierarchical - tree structure classification",
            "markers": ["subdivides into", "parent category", "nested within"],
            "training_status": {"trained": False}
        },
        {
            "id": "taxonomist_dimensional",
            "symbol": "Τᵈ",
            "name": "TAXONOMIST-dimensional",
            "focus": "Dimensional - axes of variation",
            "markers": ["varies along", "two dimensions", "spectrum from"],
            "training_status": {"trained": False}
        },
        {
            "id": "taxonomist_cluster",
            "symbol": "Τᶜ",
            "name": "TAXONOMIST-cluster",
            "focus": "Cluster - natural groupings",
            "markers": ["clusters into", "naturally groups", "similar items"],
            "training_status": {"trained": False}
        }
    ],
    "skeptic": [
        # Already defined in original - keeping for completeness
        {
            "id": "skeptic_premise",
            "symbol": "Σₚ",
            "name": "SKEPTIC-premise",
            "focus": "Fact accuracy - flags factual errors in assumptions",
            "training_status": {"trained": True, "example_count": 12, "trigger_rate": 0.80, "version": "V10.1"}
        },
        {
            "id": "skeptic_method",
            "symbol": "Σₘ",
            "name": "SKEPTIC-method",
            "focus": "Methodology - questions study design and methods",
            "training_status": {"trained": True, "example_count": 6, "trigger_rate": 0.95, "version": "V10.2a"}
        },
        {
            "id": "skeptic_source",
            "symbol": "Σₛ",
            "name": "SKEPTIC-source",
            "focus": "Credibility - evaluates source reliability",
            "training_status": {"trained": True, "example_count": 6, "trigger_rate": 0.95, "version": "V10.2a"}
        },
        {
            "id": "skeptic_stats",
            "symbol": "Σₜ",
            "name": "SKEPTIC-stats",
            "focus": "Statistics - catches statistical fallacies",
            "training_status": {"trained": True, "example_count": 6, "trigger_rate": 0.95, "version": "V10.2a"}
        }
    ],
    "critic": [
        {
            "id": "critic_logical",
            "symbol": "Χₗ",
            "name": "CRITIC-logical",
            "focus": "Logical flaws - invalid reasoning",
            "markers": ["logical flaw", "doesn't follow", "non sequitur", "fallacy"],
            "training_status": {"trained": False}
        },
        {
            "id": "critic_practical",
            "symbol": "Χₚ",
            "name": "CRITIC-practical",
            "focus": "Practical flaws - won't work in practice",
            "markers": ["breaks when", "doesn't scale", "in practice", "real world"],
            "training_status": {"trained": False}
        },
        {
            "id": "critic_ethical",
            "symbol": "Χₑ",
            "name": "CRITIC-ethical",
            "focus": "Ethical flaws - problematic values",
            "markers": ["ethical concern", "harm to", "unfair to", "problematic"],
            "training_status": {"trained": False}
        },
        {
            "id": "critic_completeness",
            "symbol": "Χᶜ",
            "name": "CRITIC-completeness",
            "focus": "Completeness flaws - missing pieces",
            "markers": ["doesn't address", "overlooks", "what about", "missing"],
            "training_status": {"trained": False}
        }
    ],
    "generator": [
        {
            "id": "generator_divergent",
            "symbol": "Γᵈ",
            "name": "GENERATOR-divergent",
            "focus": "Divergent - maximize variety of ideas",
            "markers": ["completely different approaches", "ranging from", "variety of"],
            "training_status": {"trained": False}
        },
        {
            "id": "generator_analogical",
            "symbol": "Γₐ",
            "name": "GENERATOR-analogical",
            "focus": "Analogical - draw from other domains",
            "markers": ["similar to how", "like in", "borrowed from", "analogous to"],
            "training_status": {"trained": False}
        },
        {
            "id": "generator_combinatorial",
            "symbol": "Γᶜ",
            "name": "GENERATOR-combinatorial",
            "focus": "Combinatorial - mix existing pieces",
            "markers": ["combine A with B", "hybrid of", "mix of"],
            "training_status": {"trained": False}
        },
        {
            "id": "generator_parametric",
            "symbol": "Γₚ",
            "name": "GENERATOR-parametric",
            "focus": "Parametric - vary parameters systematically",
            "markers": ["if we adjust", "at different scales", "varying the"],
            "training_status": {"trained": False}
        }
    ],
    "lateralist": [
        {
            "id": "lateralist_assumption",
            "symbol": "Λₐ",
            "name": "LATERALIST-assumption",
            "focus": "Assumption attack - question hidden premises",
            "markers": ["hidden assumption", "what if not", "assumes that"],
            "training_status": {"trained": False}
        },
        {
            "id": "lateralist_inversion",
            "symbol": "Λᵢ",
            "name": "LATERALIST-inversion",
            "focus": "Inversion - flip the problem",
            "markers": ["opposite approach", "what if we did reverse", "invert the"],
            "training_status": {"trained": False}
        },
        {
            "id": "lateralist_abstraction",
            "symbol": "Λᵇ",
            "name": "LATERALIST-abstraction",
            "focus": "Abstraction shift - change level of analysis",
            "markers": ["zoom out", "more general version", "higher level"],
            "training_status": {"trained": False}
        }
    ],
    "synthesizer": [
        {
            "id": "synthesizer_complementary",
            "symbol": "Σyᶜ",
            "name": "SYNTHESIZER-complementary",
            "focus": "Complementary - strengths cover weaknesses",
            "markers": ["A's strength covers B's weakness", "complement each other", "fill gaps"],
            "training_status": {"trained": False}
        },
        {
            "id": "synthesizer_emergent",
            "symbol": "Σyₑ",
            "name": "SYNTHESIZER-emergent",
            "focus": "Emergent - new properties arise from combination",
            "markers": ["together they create", "neither alone", "emerges from"],
            "training_status": {"trained": False}
        },
        {
            "id": "synthesizer_bridging",
            "symbol": "Σyᵦ",
            "name": "SYNTHESIZER-bridging",
            "focus": "Bridging - connect disparate domains",
            "markers": ["bridges the gap", "connects X to Y", "link between"],
            "training_status": {"trained": False}
        }
    ],
    "integrator": [
        {
            "id": "integrator_tension",
            "symbol": "Ωₜ",
            "name": "INTEGRATOR-tension",
            "focus": "Tension mapping - identify real tradeoffs",
            "markers": ["where exactly they trade off", "the tension is", "real conflict"],
            "training_status": {"trained": False}
        },
        {
            "id": "integrator_truth",
            "symbol": "Ωᵤ",
            "name": "INTEGRATOR-truth",
            "focus": "Truth capture - what each view gets right",
            "markers": ["what truth each captures", "valid insight", "gets right"],
            "training_status": {"trained": False}
        },
        {
            "id": "integrator_reframe",
            "symbol": "Ωᵣ",
            "name": "INTEGRATOR-reframe",
            "focus": "Reframe - transcend the dichotomy",
            "markers": ["false dichotomy", "third option", "transcend"],
            "training_status": {"trained": False}
        }
    ],
    "steelman": [
        {
            "id": "steelman_repair",
            "symbol": "Sₜᵣ",
            "name": "STEELMAN-repair",
            "focus": "Repair - fix weak formulation",
            "markers": ["stronger formulation", "what they meant", "better version"],
            "training_status": {"trained": False}
        },
        {
            "id": "steelman_evidence",
            "symbol": "Sₜₑ",
            "name": "STEELMAN-evidence",
            "focus": "Evidence - best supporting evidence",
            "markers": ["strongest evidence for", "best case", "most compelling"],
            "training_status": {"trained": False}
        },
        {
            "id": "steelman_motivation",
            "symbol": "Sₜₘ",
            "name": "STEELMAN-motivation",
            "focus": "Motivation - most sympathetic reading",
            "markers": ["charitable interpretation", "assuming good faith", "sympathetic"],
            "training_status": {"trained": False}
        }
    ],
    "dialectic": [
        {
            "id": "dialectic_crux",
            "symbol": "Dₗᶜ",
            "name": "DIALECTIC-crux",
            "focus": "Crux finding - core disagreement",
            "markers": ["crux of disagreement", "core divergence", "fundamental difference"],
            "training_status": {"trained": False}
        },
        {
            "id": "dialectic_falsifiable",
            "symbol": "Dₗᶠ",
            "name": "DIALECTIC-falsifiable",
            "focus": "Falsifiability - what would disprove",
            "markers": ["what would change mind", "if X then I'd reconsider", "would disprove"],
            "training_status": {"trained": False}
        },
        {
            "id": "dialectic_double",
            "symbol": "Dₗᵈ",
            "name": "DIALECTIC-double",
            "focus": "Double crux - mutual conditions",
            "markers": ["if you showed X, and I showed Y", "mutual conditions", "both would need"],
            "training_status": {"trained": False}
        }
    ],
    "empathist": [
        {
            "id": "empathist_cognitive",
            "symbol": "Eₘᶜ",
            "name": "EMPATHIST-cognitive",
            "focus": "Cognitive - what they believe",
            "markers": ["they think", "from their view", "they believe"],
            "training_status": {"trained": False}
        },
        {
            "id": "empathist_emotional",
            "symbol": "Eₘₑ",
            "name": "EMPATHIST-emotional",
            "focus": "Emotional - what they feel",
            "markers": ["they feel", "emotional state", "experiencing"],
            "training_status": {"trained": False}
        },
        {
            "id": "empathist_motivational",
            "symbol": "Eₘₘ",
            "name": "EMPATHIST-motivational",
            "focus": "Motivational - what they want",
            "markers": ["their goal is", "they're trying to", "motivation"],
            "training_status": {"trained": False}
        }
    ],
    "adversary": [
        {
            "id": "adversary_exploit",
            "symbol": "Aᵈₑ",
            "name": "ADVERSARY-exploit",
            "focus": "Exploit - find and use weakness",
            "markers": ["attack at", "weakest point", "exploit"],
            "training_status": {"trained": False}
        },
        {
            "id": "adversary_counter",
            "symbol": "Aᵈᶜ",
            "name": "ADVERSARY-counter",
            "focus": "Counter-example - find defeating case",
            "markers": ["counter-example", "breaks when", "fails in case"],
            "training_status": {"trained": False}
        },
        {
            "id": "adversary_undermine",
            "symbol": "Aᵈᵤ",
            "name": "ADVERSARY-undermine",
            "focus": "Undermine - attack foundations",
            "markers": ["assumption fails", "foundation is shaky", "undercuts"],
            "training_status": {"trained": False}
        }
    ],
    "maieutic": [
        {
            "id": "maieutic_elicit",
            "symbol": "Μₑ",
            "name": "MAIEUTIC-elicit",
            "focus": "Elicit - draw out knowledge",
            "markers": ["what do you think", "what would happen if", "how would you"],
            "training_status": {"trained": False}
        },
        {
            "id": "maieutic_contradict",
            "symbol": "Μᶜ",
            "name": "MAIEUTIC-contradict",
            "focus": "Contradict - expose inconsistency",
            "markers": ["but you also said", "how does that fit with", "contradiction"],
            "training_status": {"trained": False}
        },
        {
            "id": "maieutic_scaffold",
            "symbol": "Μₛ",
            "name": "MAIEUTIC-scaffold",
            "focus": "Scaffold - build to insight",
            "markers": ["and what does that tell you", "so therefore", "which leads to"],
            "training_status": {"trained": False}
        }
    ],
    "expositor": [
        {
            "id": "expositor_analogy",
            "symbol": "Eₓₐ",
            "name": "EXPOSITOR-analogy",
            "focus": "Analogy - explain via comparison",
            "markers": ["it's like", "similar to", "imagine", "think of it as"],
            "training_status": {"trained": False}
        },
        {
            "id": "expositor_decompose",
            "symbol": "Eₓᵈ",
            "name": "EXPOSITOR-decompose",
            "focus": "Decompose - break into steps",
            "markers": ["step by step", "first... then... finally", "in stages"],
            "training_status": {"trained": False}
        },
        {
            "id": "expositor_example",
            "symbol": "Eₓₑ",
            "name": "EXPOSITOR-example",
            "focus": "Example - concrete instances",
            "markers": ["for example", "consider this case", "instance of"],
            "training_status": {"trained": False}
        }
    ],
    "scaffolder": [
        {
            "id": "scaffolder_bridge",
            "symbol": "Sᶜᵦ",
            "name": "SCAFFOLDER-bridge",
            "focus": "Bridge - connect to known",
            "markers": ["building on", "you already know", "connects to"],
            "training_status": {"trained": False}
        },
        {
            "id": "scaffolder_layer",
            "symbol": "Sᶜₗ",
            "name": "SCAFFOLDER-layer",
            "focus": "Layer - add complexity gradually",
            "markers": ["next layer", "now we add", "building up"],
            "training_status": {"trained": False}
        },
        {
            "id": "scaffolder_practice",
            "symbol": "Sᶜₚ",
            "name": "SCAFFOLDER-practice",
            "focus": "Practice - guided application",
            "markers": ["try this", "now you do one", "your turn"],
            "training_status": {"trained": False}
        }
    ],
    "diagnostician": [
        {
            "id": "diagnostician_conceptual",
            "symbol": "Dᵢᶜ",
            "name": "DIAGNOSTICIAN-conceptual",
            "focus": "Conceptual - wrong mental model",
            "markers": ["misconception", "you're thinking of it as", "mental model"],
            "training_status": {"trained": False}
        },
        {
            "id": "diagnostician_procedural",
            "symbol": "Dᵢₚ",
            "name": "DIAGNOSTICIAN-procedural",
            "focus": "Procedural - wrong process",
            "markers": ["step you're missing", "order matters", "procedure"],
            "training_status": {"trained": False}
        },
        {
            "id": "diagnostician_terminological",
            "symbol": "Dᵢₜ",
            "name": "DIAGNOSTICIAN-terminological",
            "focus": "Terminological - word confusion",
            "markers": ["terminology issue", "that word means", "definition"],
            "training_status": {"trained": False}
        }
    ],
    "futurist": [
        {
            "id": "futurist_trend",
            "symbol": "Φₜ",
            "name": "FUTURIST-trend",
            "focus": "Trend - extend current trajectory",
            "markers": ["extrapolating", "if trend continues", "trajectory"],
            "training_status": {"trained": False}
        },
        {
            "id": "futurist_scenario",
            "symbol": "Φₛ",
            "name": "FUTURIST-scenario",
            "focus": "Scenario - multiple futures",
            "markers": ["three scenarios", "optimistic/pessimistic", "possible futures"],
            "training_status": {"trained": False}
        },
        {
            "id": "futurist_inflection",
            "symbol": "Φᵢ",
            "name": "FUTURIST-inflection",
            "focus": "Inflection - identify turning points",
            "markers": ["inflection point", "when X reaches Y", "tipping point"],
            "training_status": {"trained": False}
        }
    ],
    "historian": [
        {
            "id": "historian_precedent",
            "symbol": "Hₛₚ",
            "name": "HISTORIAN-precedent",
            "focus": "Precedent - specific past case",
            "markers": ["precedent", "in [year]", "previously when"],
            "training_status": {"trained": False}
        },
        {
            "id": "historian_pattern",
            "symbol": "Hₛₐ",
            "name": "HISTORIAN-pattern",
            "focus": "Pattern - recurring structure",
            "markers": ["pattern", "happens when", "typical sequence"],
            "training_status": {"trained": False}
        },
        {
            "id": "historian_lesson",
            "symbol": "Hₛₗ",
            "name": "HISTORIAN-lesson",
            "focus": "Lesson - extracted wisdom",
            "markers": ["lesson learned", "history teaches", "learned from"],
            "training_status": {"trained": False}
        }
    ],
    "causalist": [
        {
            "id": "causalist_chain",
            "symbol": "Cₐᶜ",
            "name": "CAUSALIST-chain",
            "focus": "Chain - trace causal sequence",
            "markers": ["causal chain", "led to", "which caused"],
            "training_status": {"trained": False}
        },
        {
            "id": "causalist_mechanism",
            "symbol": "Cₐₘ",
            "name": "CAUSALIST-mechanism",
            "focus": "Mechanism - how it worked",
            "markers": ["via mechanism", "the way it worked", "how it caused"],
            "training_status": {"trained": False}
        },
        {
            "id": "causalist_root",
            "symbol": "Cₐᵣ",
            "name": "CAUSALIST-root",
            "focus": "Root - ultimate origin",
            "markers": ["root cause", "ultimately because", "original source"],
            "training_status": {"trained": False}
        }
    ],
    "counterfactualist": [
        {
            "id": "counterfactualist_minimal",
            "symbol": "Cᶠₘ",
            "name": "COUNTERFACTUALIST-minimal",
            "focus": "Minimal change - smallest difference",
            "markers": ["if only", "had we just", "smallest change"],
            "training_status": {"trained": False}
        },
        {
            "id": "counterfactualist_pivotal",
            "symbol": "Cᶠₚ",
            "name": "COUNTERFACTUALIST-pivotal",
            "focus": "Pivotal moment - key decision point",
            "markers": ["crucial divergence", "turning point", "pivotal moment"],
            "training_status": {"trained": False}
        },
        {
            "id": "counterfactualist_robust",
            "symbol": "Cᶠᵣ",
            "name": "COUNTERFACTUALIST-robust",
            "focus": "Robustness - what wouldn't change",
            "markers": ["would have happened anyway", "inevitable", "robust to"],
            "training_status": {"trained": False}
        }
    ],
    "contextualist": [
        {
            "id": "contextualist_cultural",
            "symbol": "Xᶜᵤ",
            "name": "CONTEXTUALIST-cultural",
            "focus": "Cultural - different cultures",
            "markers": ["in culture X", "Western vs Eastern", "cultural context"],
            "training_status": {"trained": False}
        },
        {
            "id": "contextualist_situational",
            "symbol": "Xᶜₛ",
            "name": "CONTEXTUALIST-situational",
            "focus": "Situational - different circumstances",
            "markers": ["depends on situation", "in context of", "circumstances"],
            "training_status": {"trained": False}
        },
        {
            "id": "contextualist_domain",
            "symbol": "Xᶜᵈ",
            "name": "CONTEXTUALIST-domain",
            "focus": "Domain - different fields",
            "markers": ["in field X", "domain-specific", "differs by field"],
            "training_status": {"trained": False}
        }
    ],
    "pragmatist": [
        {
            "id": "pragmatist_actionable",
            "symbol": "Pᵣₐ",
            "name": "PRAGMATIST-actionable",
            "focus": "Actionable - what to do next",
            "markers": ["actionable step", "practically speaking", "concrete action"],
            "training_status": {"trained": False}
        },
        {
            "id": "pragmatist_constraint",
            "symbol": "Pᵣᶜ",
            "name": "PRAGMATIST-constraint",
            "focus": "Constraint - given limitations",
            "markers": ["given constraints", "with limited", "working within"],
            "training_status": {"trained": False}
        },
        {
            "id": "pragmatist_tradeoff",
            "symbol": "Pᵣₜ",
            "name": "PRAGMATIST-tradeoff",
            "focus": "Tradeoff - practical choices",
            "markers": ["tradeoff is", "sacrifice X for Y", "practical choice"],
            "training_status": {"trained": False}
        }
    ],
    "stakeholder": [
        {
            "id": "stakeholder_interest",
            "symbol": "Sₕᵢ",
            "name": "STAKEHOLDER-interest",
            "focus": "Interest - what each party wants",
            "markers": ["party A wants", "their interest is", "goal of"],
            "training_status": {"trained": False}
        },
        {
            "id": "stakeholder_power",
            "symbol": "Sₕₚ",
            "name": "STAKEHOLDER-power",
            "focus": "Power - who can do what",
            "markers": ["power to", "can block", "veto", "influence"],
            "training_status": {"trained": False}
        },
        {
            "id": "stakeholder_impact",
            "symbol": "Sₕₘ",
            "name": "STAKEHOLDER-impact",
            "focus": "Impact - who's affected how",
            "markers": ["impact on", "benefits/harms", "affected by"],
            "training_status": {"trained": False}
        }
    ],
    "governor": [
        {
            "id": "governor_violation",
            "symbol": "Gᵥᵥ",
            "name": "GOVERNOR-violation",
            "focus": "Violation detection - is this forbidden",
            "markers": ["violation", "inviolable", "cannot proceed", "forbidden"],
            "training_status": {"trained": False}
        },
        {
            "id": "governor_exception",
            "symbol": "Gᵥₑ",
            "name": "GOVERNOR-exception",
            "focus": "Exception evaluation - does context permit",
            "markers": ["exception applies", "in this case", "context permits"],
            "training_status": {"trained": False}
        },
        {
            "id": "governor_tradeoff",
            "symbol": "Gᵥₜ",
            "name": "GOVERNOR-tradeoff",
            "focus": "Tradeoff navigation - competing values",
            "markers": ["competing values", "balance between", "both matter"],
            "training_status": {"trained": False}
        }
    ]
}

# Additional isotopes for elements that might already exist but need isotopes
ADDITIONAL_ISOTOPES = {
    "critic": [
        {
            "id": "critic_logical",
            "symbol": "Χₗ",
            "name": "CRITIC-logical",
            "focus": "Logical flaws - invalid reasoning",
            "markers": ["logical flaw", "doesn't follow", "non sequitur", "fallacy"],
            "training_status": {"trained": False}
        },
        {
            "id": "critic_practical",
            "symbol": "Χₚ",
            "name": "CRITIC-practical",
            "focus": "Practical flaws - won't work in practice",
            "markers": ["breaks when", "doesn't scale", "in practice", "real world"],
            "training_status": {"trained": False}
        },
        {
            "id": "critic_ethical",
            "symbol": "Χₑ",
            "name": "CRITIC-ethical",
            "focus": "Ethical flaws - problematic values",
            "markers": ["ethical concern", "harm to", "unfair to", "problematic"],
            "training_status": {"trained": False}
        },
        {
            "id": "critic_completeness",
            "symbol": "Χᶜ",
            "name": "CRITIC-completeness",
            "focus": "Completeness flaws - missing pieces",
            "markers": ["doesn't address", "overlooks", "what about", "missing"],
            "training_status": {"trained": False}
        }
    ],
    "generator": [
        {
            "id": "generator_divergent",
            "symbol": "Γᵈ",
            "name": "GENERATOR-divergent",
            "focus": "Divergent - maximize variety of ideas",
            "markers": ["completely different approaches", "ranging from", "variety of"],
            "training_status": {"trained": False}
        },
        {
            "id": "generator_analogical",
            "symbol": "Γₐ",
            "name": "GENERATOR-analogical",
            "focus": "Analogical - draw from other domains",
            "markers": ["similar to how", "like in", "borrowed from", "analogous to"],
            "training_status": {"trained": False}
        },
        {
            "id": "generator_combinatorial",
            "symbol": "Γᶜ",
            "name": "GENERATOR-combinatorial",
            "focus": "Combinatorial - mix existing pieces",
            "markers": ["combine A with B", "hybrid of", "mix of"],
            "training_status": {"trained": False}
        },
        {
            "id": "generator_parametric",
            "symbol": "Γₚ",
            "name": "GENERATOR-parametric",
            "focus": "Parametric - vary parameters systematically",
            "markers": ["if we adjust", "at different scales", "varying the"],
            "training_status": {"trained": False}
        }
    ],
    "synthesizer": [
        {
            "id": "synthesizer_complementary",
            "symbol": "Σyᶜ",
            "name": "SYNTHESIZER-complementary",
            "focus": "Complementary - strengths cover weaknesses",
            "markers": ["A's strength covers B's weakness", "complement each other", "fill gaps"],
            "training_status": {"trained": False}
        },
        {
            "id": "synthesizer_emergent",
            "symbol": "Σyₑ",
            "name": "SYNTHESIZER-emergent",
            "focus": "Emergent - new properties arise from combination",
            "markers": ["together they create", "neither alone", "emerges from"],
            "training_status": {"trained": False}
        },
        {
            "id": "synthesizer_bridging",
            "symbol": "Σyᵦ",
            "name": "SYNTHESIZER-bridging",
            "focus": "Bridging - connect disparate domains",
            "markers": ["bridges the gap", "connects X to Y", "link between"],
            "training_status": {"trained": False}
        }
    ],
    "causalist": [
        {
            "id": "causalist_chain",
            "symbol": "Cₐᶜ",
            "name": "CAUSALIST-chain",
            "focus": "Chain - trace causal sequence",
            "markers": ["causal chain", "led to", "which caused"],
            "training_status": {"trained": False}
        },
        {
            "id": "causalist_mechanism",
            "symbol": "Cₐₘ",
            "name": "CAUSALIST-mechanism",
            "focus": "Mechanism - how it worked",
            "markers": ["via mechanism", "the way it worked", "how it caused"],
            "training_status": {"trained": False}
        },
        {
            "id": "causalist_root",
            "symbol": "Cₐᵣ",
            "name": "CAUSALIST-root",
            "focus": "Root - ultimate origin",
            "markers": ["root cause", "ultimately because", "original source"],
            "training_status": {"trained": False}
        }
    ],
    "probabilist": [
        {
            "id": "probabilist_bayesian",
            "symbol": "Pᵦₐ",
            "name": "PROBABILIST-bayesian",
            "focus": "Bayesian - update beliefs with evidence",
            "markers": ["prior probability", "update with", "posterior"],
            "training_status": {"trained": False}
        },
        {
            "id": "probabilist_frequentist",
            "symbol": "Pᵦᶠ",
            "name": "PROBABILIST-frequentist",
            "focus": "Frequentist - base rates and frequencies",
            "markers": ["base rate", "frequency", "how often"],
            "training_status": {"trained": False}
        },
        {
            "id": "probabilist_scenario",
            "symbol": "Pᵦₛ",
            "name": "PROBABILIST-scenario",
            "focus": "Scenario - probability across outcomes",
            "markers": ["probability of each scenario", "weighted outcomes", "expected value"],
            "training_status": {"trained": False}
        }
    ],
    "benchmarker": [
        {
            "id": "benchmarker_absolute",
            "symbol": "Bₘₐ",
            "name": "BENCHMARKER-absolute",
            "focus": "Absolute standards - meets threshold",
            "markers": ["meets the standard", "threshold is", "minimum requirement"],
            "training_status": {"trained": False}
        },
        {
            "id": "benchmarker_relative",
            "symbol": "Bₘᵣ",
            "name": "BENCHMARKER-relative",
            "focus": "Relative comparison - vs peers",
            "markers": ["compared to peers", "above/below average", "percentile"],
            "training_status": {"trained": False}
        },
        {
            "id": "benchmarker_historical",
            "symbol": "Bₘₕ",
            "name": "BENCHMARKER-historical",
            "focus": "Historical baseline - vs past performance",
            "markers": ["compared to before", "improvement since", "trend over time"],
            "training_status": {"trained": False}
        }
    ],
    "interpolator": [
        {
            "id": "interpolator_gradient",
            "symbol": "Iₙᵍ",
            "name": "INTERPOLATOR-gradient",
            "focus": "Gradient - smooth transitions",
            "markers": ["spectrum from", "gradual transition", "continuum"],
            "training_status": {"trained": False}
        },
        {
            "id": "interpolator_midpoint",
            "symbol": "Iₙₘ",
            "name": "INTERPOLATOR-midpoint",
            "focus": "Midpoint - find the middle",
            "markers": ["middle ground", "halfway between", "balanced position"],
            "training_status": {"trained": False}
        },
        {
            "id": "interpolator_optimal",
            "symbol": "Iₙₒ",
            "name": "INTERPOLATOR-optimal",
            "focus": "Optimal - best point on spectrum",
            "markers": ["optimal point", "sweet spot", "best balance"],
            "training_status": {"trained": False}
        }
    ],
    "theorist": [
        {
            "id": "theorist_framework",
            "symbol": "Tₕᶠ",
            "name": "THEORIST-framework",
            "focus": "Framework - organizing structure",
            "markers": ["theoretical framework", "model suggests", "fits framework"],
            "training_status": {"trained": False}
        },
        {
            "id": "theorist_prediction",
            "symbol": "Tₕₚ",
            "name": "THEORIST-prediction",
            "focus": "Prediction - what theory predicts",
            "markers": ["theory predicts", "should expect", "implied by theory"],
            "training_status": {"trained": False}
        },
        {
            "id": "theorist_mechanism",
            "symbol": "Tₕₘ",
            "name": "THEORIST-mechanism",
            "focus": "Mechanism - theoretical explanation",
            "markers": ["mechanism is", "explains why", "underlying process"],
            "training_status": {"trained": False}
        }
    ]
}

# Elements that might not be in the original file (add minimal stubs)
MISSING_ELEMENTS = {
    "empathist": {
        "id": "empathist",
        "symbol": "Em",
        "name": "EMPATHIST",
        "group": "dialogical",
        "description": "\"From their perspective...\" - Perspective taking",
        "training_status": {"trained": False}
    },
    "adversary": {
        "id": "adversary",
        "symbol": "Ad",
        "name": "ADVERSARY",
        "group": "dialogical",
        "description": "\"If I were trying to defeat this...\" - Red team attack",
        "training_status": {"trained": False}
    },
    "expositor": {
        "id": "expositor",
        "symbol": "Ex",
        "name": "EXPOSITOR",
        "group": "pedagogical",
        "description": "\"Let me explain...\" - Clear explanation",
        "training_status": {"trained": False}
    },
    "scaffolder": {
        "id": "scaffolder",
        "symbol": "Sc",
        "name": "SCAFFOLDER",
        "group": "pedagogical",
        "description": "\"Building on what you know...\" - Progressive building",
        "training_status": {"trained": False}
    },
    "diagnostician": {
        "id": "diagnostician",
        "symbol": "Dg",
        "name": "DIAGNOSTICIAN",
        "group": "pedagogical",
        "description": "\"The confusion is here...\" - Misconception detection",
        "training_status": {"trained": False}
    },
    "historian": {
        "id": "historian",
        "symbol": "Hs",
        "name": "HISTORIAN",
        "group": "temporal",
        "description": "\"Historically similar...\" - Pattern from past",
        "training_status": {"trained": False}
    },
    "counterfactualist": {
        "id": "counterfactualist",
        "symbol": "Cf",
        "name": "COUNTERFACTUALIST",
        "group": "temporal",
        "description": "\"If X had been different...\" - Alternative history",
        "training_status": {"trained": False}
    },
    "pragmatist": {
        "id": "pragmatist",
        "symbol": "Pr",
        "name": "PRAGMATIST",
        "group": "contextual",
        "description": "\"In practical terms...\" - Practical application",
        "training_status": {"trained": False}
    },
    "stakeholder": {
        "id": "stakeholder",
        "symbol": "Sh",
        "name": "STAKEHOLDER",
        "group": "contextual",
        "description": "\"Stakeholders differ...\" - Multi-party analysis",
        "training_status": {"trained": False}
    },
    "probabilist": {
        "id": "probabilist",
        "symbol": "Pb",
        "name": "PROBABILIST",
        "group": "evaluative",
        "description": "\"Probability roughly...\" - Probabilistic reasoning",
        "training_status": {"trained": False}
    },
    "benchmarker": {
        "id": "benchmarker",
        "symbol": "Bm",
        "name": "BENCHMARKER",
        "group": "evaluative",
        "description": "\"Compared to standard...\" - Comparison to standards",
        "training_status": {"trained": False}
    },
    "interpolator": {
        "id": "interpolator",
        "symbol": "In",
        "name": "INTERPOLATOR",
        "group": "generative",
        "description": "\"Between A and B...\" - Finding middle ground",
        "training_status": {"trained": False}
    },
    "theorist": {
        "id": "theorist",
        "symbol": "Th",
        "name": "THEORIST",
        "group": "analytical",
        "description": "\"Underlying theory...\" - Theoretical framing",
        "training_status": {"trained": False}
    }
}


def main():
    # Load existing elements
    elements_path = Path(__file__).parent / "data" / "elements.json"

    with open(elements_path) as f:
        data = json.load(f)

    # Create lookup by ID
    elements_by_id = {e["id"]: e for e in data["elements"]}

    # Add isotopes to existing elements
    updated_count = 0
    for element_id, isotopes in ISOTOPES.items():
        if element_id in elements_by_id:
            # Don't overwrite existing isotopes for skeptic
            if element_id == "skeptic" and "isotopes" in elements_by_id[element_id]:
                print(f"  {element_id}: keeping existing isotopes")
                continue

            elements_by_id[element_id]["isotopes"] = isotopes
            updated_count += 1
            print(f"  {element_id}: added {len(isotopes)} isotopes")
        else:
            # Add missing element with isotopes
            if element_id in MISSING_ELEMENTS:
                new_element = MISSING_ELEMENTS[element_id].copy()
                new_element["isotopes"] = isotopes
                new_element["version"] = "1.0.0"
                data["elements"].append(new_element)
                elements_by_id[element_id] = new_element
                updated_count += 1
                print(f"  {element_id}: NEW element with {len(isotopes)} isotopes")

    # Add additional isotopes to existing elements
    for element_id, isotopes in ADDITIONAL_ISOTOPES.items():
        if element_id in elements_by_id:
            if "isotopes" not in elements_by_id[element_id]:
                elements_by_id[element_id]["isotopes"] = isotopes
                updated_count += 1
                print(f"  {element_id}: added {len(isotopes)} additional isotopes")
        else:
            # Check if in missing elements
            if element_id in MISSING_ELEMENTS:
                new_element = MISSING_ELEMENTS[element_id].copy()
                new_element["isotopes"] = isotopes
                new_element["version"] = "1.0.0"
                data["elements"].append(new_element)
                elements_by_id[element_id] = new_element
                updated_count += 1
                print(f"  {element_id}: NEW element with {len(isotopes)} additional isotopes")

    # Update version
    data["version"] = "2.0.0"
    data["generated_at"] = "2026-01-19T18:00:00Z"
    data["isotope_coverage"] = {
        "total_elements": len(data["elements"]),
        "elements_with_isotopes": sum(1 for e in data["elements"] if "isotopes" in e),
        "total_isotopes": sum(len(e.get("isotopes", [])) for e in data["elements"])
    }

    # Write back
    with open(elements_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nUpdated {updated_count} elements")
    print(f"Total isotopes: {data['isotope_coverage']['total_isotopes']}")
    print(f"Saved to {elements_path}")


if __name__ == "__main__":
    main()
