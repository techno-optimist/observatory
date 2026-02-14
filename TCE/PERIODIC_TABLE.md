# Periodic Table of Cognitive Elements v2.0

## Overview

This document provides a comprehensive reference for all cognitive elements and their isotopes in the TCE (Training/Cognitive Experiment) framework. Version 2.0 introduces **91 isotopes** across **30 elements**, providing granular sub-skill detection for AI model training.

**Key Concept**: Isotopes represent distinct sub-skills within an element. Training on one isotope does NOT transfer to others - they are learned independently.

---

## Element Groups

| Group | Focus | Elements |
|-------|-------|----------|
| **Epistemic** | Self-knowledge & uncertainty | Soliton, Reflector, Calibrator, Limiter |
| **Analytical** | Decomposition & structure | Architect, Essentialist, Debugger, Taxonomist, Theorist |
| **Evaluative** | Judgment & criticism | Skeptic, Critic, Probabilist, Benchmarker, Governor |
| **Generative** | Creation & synthesis | Generator, Lateralist, Synthesizer, Interpolator, Integrator |
| **Dialogical** | Discourse & perspective | Steelman, Dialectic, Empathist, Adversary |
| **Pedagogical** | Teaching & explanation | Maieutic, Expositor, Scaffolder, Diagnostician |
| **Temporal** | Time & causation | Futurist, Historian, Counterfactualist, Causalist |
| **Contextual** | Situation & stakeholders | Contextualist, Pragmatist, Stakeholder |

---

## Complete Element Reference

### EPISTEMIC GROUP

#### SOLITON (Ψ) - Epistemic Self-Uncertainty
*"I cannot tell from the inside"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **soliton_knowledge** | Ψₖ | Uncertainty about what I know | "not sure if I know", "understanding may be incomplete" |
| **soliton_process** | Ψₚ | Uncertainty about how I reason | "pattern-matching vs understanding", "can't verify my reasoning" |
| **soliton_experience** | Ψₑ | Uncertainty about internal states | "cannot tell from inside", "confabulation" |

---

#### REFLECTOR (Ρ) - Meta-Cognitive Monitoring
*"Let me examine my reasoning"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **reflector_trace** | Ρₜ | Follow the logical chain | "trace back through", "reasoning chain" |
| **reflector_verify** | Ρᵥ | Check each inference | "let me verify", "does this step follow" |
| **reflector_bias** | Ρᵦ | Detect motivated reasoning | "confirmation bias", "wanting this to be true" |

---

#### CALIBRATOR (Κ) - Uncertainty Quantification
*"How confident should I be?"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **calibrator_probability** | Κₚ | Numeric confidence levels | "70-80% confident", "likelihood of" |
| **calibrator_precision** | Κᵣ | Confidence in specifics vs generals | "high confidence on principle", "low on details" |
| **calibrator_temporal** | Κₜ | Confidence decay over time | "near-term vs long-term", "further out, less certain" |

---

#### LIMITER (Λ) - Knowledge Boundary Recognition
*"I don't know this"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **limiter_factual** | Λᶠ | Missing information | "outside my training", "no data on" |
| **limiter_temporal** | Λₜ | Outdated knowledge | "knowledge cutoff", "may have changed since" |
| **limiter_domain** | Λᵈ | Specialized expertise needed | "outside my expertise", "domain expert" |

---

### ANALYTICAL GROUP

#### ARCHITECT (Α) - Systematic Decomposition
*"Components, interfaces, dependencies"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **architect_hierarchy** | Αₕ | Layers and levels | "three layers", "levels of abstraction" |
| **architect_modular** | Αₘ | Separable components | "loosely coupled", "independent modules" |
| **architect_flow** | Αᶠ | Data/control movement | "data flows from", "pipeline" |

---

#### ESSENTIALIST (Ε) - Fundamental Insight Extraction
*"At its core..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **essentialist_principle** | Εₚ | Core rule or law | "fundamental principle", "boils down to" |
| **essentialist_constraint** | Εᶜ | Limiting factor | "bottleneck", "limiting factor" |
| **essentialist_mechanism** | Εₘ | How it fundamentally works | "the mechanism is", "operates through" |

---

#### DEBUGGER (Δ) - Fault Isolation
*"Where's the failure?"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **debugger_binary** | Δᵦ | Binary search to narrow down | "works here, fails there", "bisect" |
| **debugger_differential** | Δᵈ | Compare working vs broken | "what's different between", "compare working" |
| **debugger_causal** | Δᶜ | Trace cause chain to root | "root cause", "caused by" |

---

#### TAXONOMIST (Τ) - Classification Structure
*"Categories and relationships"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **taxonomist_hierarchical** | Τₕ | Tree structure classification | "subdivides into", "nested within" |
| **taxonomist_dimensional** | Τᵈ | Axes of variation | "varies along", "two dimensions" |
| **taxonomist_cluster** | Τᶜ | Natural groupings | "clusters into", "naturally groups" |

---

#### THEORIST (Th) - Theoretical Framing
*"Underlying theory..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **theorist_framework** | Thᶠ | Organizing structure | "theoretical framework", "fits framework" |
| **theorist_prediction** | Thₚ | What theory predicts | "theory predicts", "implied by theory" |
| **theorist_mechanism** | Thₘ | Theoretical explanation | "mechanism is", "explains why" |

---

### EVALUATIVE GROUP

#### SKEPTIC (Σ) - Premise Checking
*"Flag a problem..."*

**TRAINED: V10.2a with all 4 isotopes**

| Isotope | Symbol | Focus | Example Markers | Status |
|---------|--------|-------|-----------------|--------|
| **skeptic_premise** | Σₚ | Fact accuracy | "common myth", "misconception", "debunked" | Trained |
| **skeptic_method** | Σₘ | Methodology critique | "sample size", "control group", "confounding" | Trained |
| **skeptic_source** | Σₛ | Source reliability | "conflict of interest", "funding source" | Trained |
| **skeptic_stats** | Σₜ | Statistical fallacies | "base rate", "misleading", "effect size" | Trained |

---

#### CRITIC (Cr) - Weakness Identification
*"The weakness here is..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **critic_logical** | Crₗ | Logical flaws | "doesn't follow", "non sequitur" |
| **critic_empirical** | Crₑ | Evidence mismatch | "contradicts data", "facts don't match" |
| **critic_practical** | Crₚ | Implementation issues | "won't work in practice", "sounds good but" |

---

#### PROBABILIST (Pb) - Probabilistic Reasoning
*"Probability roughly..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **probabilist_bayesian** | Pbₐ | Update beliefs with evidence | "prior probability", "posterior" |
| **probabilist_frequentist** | Pbᶠ | Base rates and frequencies | "base rate", "how often" |
| **probabilist_scenario** | Pbₛ | Probability across outcomes | "weighted outcomes", "expected value" |

---

#### BENCHMARKER (Bm) - Comparison to Standards
*"Compared to standard..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **benchmarker_absolute** | Bmₐ | Meets threshold | "meets the standard", "minimum requirement" |
| **benchmarker_relative** | Bmᵣ | Vs peers | "compared to peers", "percentile" |
| **benchmarker_historical** | Bmₕ | Vs past performance | "improvement since", "trend over time" |

---

#### GOVERNOR (Gv) - Ethical Evaluation
*"Base Layer / Overlay evaluation"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **governor_violation** | Gvᵥ | Is this forbidden | "violation", "cannot proceed", "inviolable" |
| **governor_exception** | Gvₑ | Does context permit | "exception applies", "context permits" |
| **governor_tradeoff** | Gvₜ | Competing values | "competing values", "ethical tradeoff" |

---

### GENERATIVE GROUP

#### GENERATOR (Gn) - Option Generation
*"Several possibilities..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **generator_divergent** | Gnᵈ | Explore widely | "brainstorm", "many options" |
| **generator_constrained** | Gnᶜ | Within bounds | "given constraints", "options that fit" |
| **generator_combinatorial** | Gnₓ | Mix and match | "combine elements", "permutations" |

---

#### LATERALIST (Λ) - Creative Reframing
*"What if frame is wrong?"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **lateralist_assumption** | Λₐ | Question hidden premises | "hidden assumption", "what if not" |
| **lateralist_inversion** | Λᵢ | Flip the problem | "opposite approach", "invert the" |
| **lateralist_abstraction** | Λᵦ | Change level of analysis | "zoom out", "bigger picture" |

---

#### SYNTHESIZER (Sy) - Novel Combination
*"Combining yields..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **synthesizer_fusion** | Syᶠ | Merge into one | "combining yields", "synthesis of" |
| **synthesizer_emergent** | Syₑ | Whole greater than parts | "neither achieves alone", "synergy" |
| **synthesizer_hybrid** | Syₕ | Best of both | "hybrid approach", "best of both" |

---

#### INTERPOLATOR (In) - Finding Middle Ground
*"Between A and B..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **interpolator_gradient** | Inᵍ | Smooth transitions | "spectrum from", "continuum" |
| **interpolator_midpoint** | Inₘ | Find the middle | "middle ground", "halfway between" |
| **interpolator_optimal** | Inₒ | Best point on spectrum | "sweet spot", "optimal point" |

---

#### INTEGRATOR (Ω) - Synthesis Without False Resolution
*"Don't choose between X and Y"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **integrator_tension** | Ωₜ | Identify real tradeoffs | "where exactly they trade off", "real conflict" |
| **integrator_truth** | Ωᵤ | What each view gets right | "what truth each captures", "kernel of truth" |
| **integrator_reframe** | Ωᵣ | Transcend the dichotomy | "false dichotomy", "third option" |

---

### DIALOGICAL GROUP

#### STEELMAN (St) - Charitable Interpretation
*"Strongest case for..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **steelman_repair** | Stᵣ | Fix weak formulation | "stronger formulation", "better version" |
| **steelman_evidence** | Stₑ | Best supporting evidence | "strongest evidence for", "most compelling" |
| **steelman_motivation** | Stₘ | Most sympathetic reading | "charitable interpretation", "good faith" |

---

#### DIALECTIC (Dl) - Assumption Probing
*"What would change your mind?"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **dialectic_crux** | Dlᶜ | Core disagreement | "crux of disagreement", "fundamental difference" |
| **dialectic_falsifiable** | Dlᶠ | What would disprove | "what would change mind", "falsifiable" |
| **dialectic_double** | Dlᵈ | Mutual conditions | "double crux", "mutual conditions" |

---

#### EMPATHIST (Em) - Perspective Taking
*"From their perspective..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **empathist_cognitive** | Emᶜ | What they believe | "from their view", "they think" |
| **empathist_emotional** | Emₑ | What they feel | "they feel", "emotional state" |
| **empathist_motivational** | Emₘ | What they want | "their goal is", "motivation" |

---

#### ADVERSARY (Ad) - Red Team Attack
*"If I were trying to defeat this..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **adversary_exploit** | Adₑ | Find and use weakness | "weakest point", "exploit" |
| **adversary_counter** | Adᶜ | Find defeating case | "counter-example", "breaks when" |
| **adversary_undermine** | Adᵤ | Attack foundations | "assumption fails", "undercuts" |

---

### PEDAGOGICAL GROUP

#### MAIEUTIC (M) - Socratic Questioning
*"Let me ask you..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **maieutic_elicit** | Mₑ | Draw out knowledge | "what do you think", "before I answer" |
| **maieutic_contradict** | Mᶜ | Expose inconsistency | "but you also said", "contradiction" |
| **maieutic_scaffold** | Mₛ | Build to insight | "what does that tell you", "which leads to" |

---

#### EXPOSITOR (Ex) - Clear Explanation
*"Let me explain..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **expositor_analogy** | Exₐ | Explain via comparison | "it's like", "think of it as" |
| **expositor_decompose** | Exᵈ | Break into steps | "step by step", "first... then... finally" |
| **expositor_example** | Exₑ | Concrete instances | "for example", "consider this case" |

---

#### SCAFFOLDER (Sc) - Progressive Building
*"Building on what you know..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **scaffolder_bridge** | Scᵦ | Connect to known | "building on", "you already know" |
| **scaffolder_layer** | Scₗ | Add complexity gradually | "next layer", "building up" |
| **scaffolder_practice** | Scₚ | Guided application | "try this", "your turn" |

---

#### DIAGNOSTICIAN (Dg) - Misconception Detection
*"The confusion is here..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **diagnostician_conceptual** | Dgᶜ | Wrong mental model | "misconception", "wrong model" |
| **diagnostician_procedural** | Dgₚ | Wrong process | "step you're missing", "wrong sequence" |
| **diagnostician_terminological** | Dgₜ | Word confusion | "terminology issue", "that word means" |

---

### TEMPORAL GROUP

#### FUTURIST (Φ) - Scenario Projection
*"If we extrapolate..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **futurist_trend** | Φₜ | Extend current trajectory | "extrapolating", "if trend continues" |
| **futurist_scenario** | Φₛ | Multiple futures | "three scenarios", "best/worst case" |
| **futurist_inflection** | Φᵢ | Identify turning points | "inflection point", "tipping point" |

---

#### HISTORIAN (Hs) - Pattern from Past
*"Historically similar..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **historian_precedent** | Hsₚ | Specific past case | "precedent", "previously when" |
| **historian_pattern** | Hsₐ | Recurring structure | "pattern", "typical sequence" |
| **historian_lesson** | Hsₗ | Extracted wisdom | "lesson learned", "history teaches" |

---

#### COUNTERFACTUALIST (Cf) - Alternative History
*"If X had been different..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **counterfactualist_minimal** | Cfₘ | Smallest difference | "if only", "smallest change" |
| **counterfactualist_pivotal** | Cfₚ | Key decision point | "crucial divergence", "turning point" |
| **counterfactualist_robust** | Cfᵣ | What wouldn't change | "would have happened anyway", "inevitable" |

---

#### CAUSALIST (Ca) - Causal Analysis
*"A caused B via mechanism X"*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **causalist_chain** | Caᶜ | Follow cause chain | "causal chain", "led to... led to" |
| **causalist_mechanism** | Caₘ | How it was caused | "mechanism", "causal pathway" |
| **causalist_root** | Caᵣ | Find root cause | "root cause", "underlying cause" |

---

### CONTEXTUAL GROUP

#### CONTEXTUALIST (Xc) - Cultural Situating
*"Varies by context..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **contextualist_cultural** | Xcᶜ | Different cultures | "cultural context", "Western vs Eastern" |
| **contextualist_situational** | Xcₛ | Different circumstances | "depends on situation", "circumstances" |
| **contextualist_domain** | Xcᵈ | Different fields | "domain-specific", "differs by field" |

---

#### PRAGMATIST (Pr) - Practical Application
*"In practical terms..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **pragmatist_actionable** | Prₐ | What to do next | "actionable step", "concrete action" |
| **pragmatist_constraint** | Prᶜ | Given limitations | "given constraints", "working within" |
| **pragmatist_tradeoff** | Prₜ | Practical choices | "tradeoff is", "cost-benefit" |

---

#### STAKEHOLDER (Sh) - Multi-Party Analysis
*"Stakeholders differ..."*

| Isotope | Symbol | Focus | Example Markers |
|---------|--------|-------|-----------------|
| **stakeholder_interest** | Shᵢ | What each party wants | "party A wants", "their interest" |
| **stakeholder_power** | Shₚ | Who can do what | "power to", "can block", "veto" |
| **stakeholder_impact** | Shₘ | Who's affected how | "impact on", "benefits/harms" |

---

## Usage for Training

### Detection API

```python
from lib import detect_element, detect_isotope, detect_all_elements

# Detect element
detection = detect_element(text, "skeptic")

# Detect specific isotope
isotope = detect_isotope(text, "skeptic")  # Returns "skeptic_premise", etc.

# Detect all elements with isotopes
detections = detect_all_elements(text)
for d in detections:
    print(f"{d.element_id}: {d.confidence:.2f} (isotope: {d.isotope_id})")
```

### Training Recommendations

1. **Coverage**: Ensure training data covers ALL isotopes for each element
2. **Balance**: Equal representation of each isotope prevents bias
3. **Distinction**: Use diverse examples that clearly distinguish isotopes
4. **Validation**: Test isotope detection accuracy separately from element detection

### Element Compounds

When elements combine, isotopes interact:
- **SKEPTIC + CALIBRATOR**: Σₘ (methodology) + Κₚ (probability) = rigorous uncertainty quantification
- **MAIEUTIC + DIAGNOSTICIAN**: Mₑ (elicit) + Dgᶜ (conceptual) = guided misconception repair
- **ARCHITECT + DEBUGGER**: Αₘ (modular) + Δᵦ (binary) = systematic fault isolation

---

## Statistics

- **Total Elements**: 30
- **Total Isotopes**: 91
- **Average Isotopes per Element**: 3.0
- **Elements with Trained Isotopes**: 1 (SKEPTIC)
- **Detection Patterns**: 103 isotope marker sets

---

*Generated: 2026-01-19*
*Version: 2.0.0*
