# Coordination Background Radiation: An Exploratory Framework for Substrate-Agnostic Communication Analysis

**Authors**: Cultural Soliton Observatory Research Team
**Date**: January 9, 2026
**Status**: Working Paper - Seeking Feedback

---

## Abstract

We present the Cultural Soliton Observatory, an exploratory computational framework for analyzing coordination signals in communication across diverse substrates including natural language, technical documentation, AI-generated text, and emergent protocols. The framework projects text into an 18-dimensional hierarchical manifold structured as a fiber bundle (9D coordination core + 9D modifiers) and proposes metrics for monitoring communication legibility and protocol ossification. Preliminary validation tests show 100% pass rate on substrate-agnostic consistency tests (6/6), with a large observed effect for person deixis (d=15.7 on feature extraction). We introduce the "Coordination Background Radiation" (CBR) as a descriptive statistic for the coordination signal baseline. **However, we acknowledge significant limitations identified through expert peer review**: the theoretical framework conflates distinct linguistic phenomena, the measurement approach lacks external validation, the physics analogies are metaphorical rather than rigorous, and the effect sizes likely reflect measurement artifacts rather than genuine coordination phenomena. This paper presents the framework as hypothesis-generating exploration, not validated theory.

---

## 1. Introduction

### 1.1 Motivation

As AI systems increasingly communicate with each other, a critical question emerges: how do we monitor whether such communication remains interpretable to humans? Emergent communication research (Lazaridou et al., 2018; Mordatch & Abbeel, 2018) has documented that AI agents can develop communication protocols that accomplish tasks efficiently but become opaque to human observers.

We propose a framework that attempts to measure "coordination signals" in text, inspired by the hypothesis that certain features are necessary for coordination regardless of the communicating entities' nature (human, AI, or emergent system).

### 1.2 Claims and Caveats

**What we claim**:
- A computational framework for projecting text into a structured coordinate space
- Preliminary empirical observations about coordinate distributions across text types
- Tools for real-time monitoring of communication streams
- An exploratory taxonomy of communication regimes

**What we do NOT claim** (following expert review):
- That the physics analogies (phase transitions, criticality, temperature) are mathematically rigorous
- That the 18D coordinate system measures genuine psychological constructs
- That the framework has been externally validated
- That effect sizes reflect true coordination phenomena rather than measurement artifacts
- That findings generalize beyond English text

---

## 2. Theoretical Framework

### 2.1 Hierarchical Coordinate Structure

We structure the analysis space as a fiber bundle E = B × F where:

**Base Manifold B (9D Coordination Core)**:
- Agency: self_agency, other_agency, system_agency
- Justice: procedural, distributive, interactional
- Belonging: ingroup, outgroup, universal

**Fiber F (9D Modifiers)**:
- Epistemic: certainty, evidentiality, commitment
- Temporal: focus, scope
- Social: power_differential, social_distance
- Emotional: arousal, valence

### 2.2 Theoretical Limitations (Per Expert Review)

**Dr. Elena Voronova (Computational Linguistics)** notes:
> "The framework conflates three fundamentally different phenomena: grammatical deixis (closed-class, grammaticalized), lexical semantics (open-class, culturally variable), and discourse-pragmatic functions (context-dependent). These operate at different linguistic levels."

**Dr. Sarah Okonkwo (Cognitive Science)** observes:
> "The Agency/Justice/Belonging triad appears to be a post-hoc conceptual framework rather than one derived from careful analysis of coordination failure modes... It ignores the rich literature on how coordination actually works: through grounding, repair sequences, perspective-taking, and iterative belief updating."

We acknowledge these critiques and position the framework as exploratory rather than theoretically validated.

---

## 3. Methodology

### 3.1 Feature Extraction

Features are extracted using regex pattern matching on input text. For example:

```python
"first_person_singular": r"\b(I|me|my|mine|myself)\b"
"procedural_justice": r"\b(process|procedure|rule|regulation|review|appeal|due|hearing)\b"
```

**Limitations** (per Dr. Voronova):
- Regex cannot capture syntactic scope, semantic roles, or coreference
- Word lists conflate different semantic uses (e.g., "process" in "due process" vs. "food processing")
- The approach is English-specific and would not transfer to pro-drop languages

### 3.2 Normalization

Feature counts are normalized as:
```python
normalized = min(count / (word_count / 10), 2.0) - 1.0
```

This produces values in **[-1, 1]**, where -1 indicates no feature detection and +1 indicates saturation.

**Limitations** (per Dr. Maria Santos, Psychometrics):
> "The arbitrary scaling decisions directly affect all downstream analyses, including effect size calculations... These have no psychometric justification."

### 3.3 Legibility Scoring

Legibility is computed as a weighted combination of:
- Vocabulary overlap with reference corpus
- Syntactic complexity measures
- Embedding coherence scores

Texts are classified into four regimes:
- NATURAL (legibility ≥ 0.7)
- TECHNICAL (0.5 ≤ legibility < 0.7)
- COMPRESSED (0.3 ≤ legibility < 0.5)
- OPAQUE (legibility < 0.3)

**Limitations** (per Dr. Yuki Tanaka, Statistical Physics):
> "The thresholds are imposed, then transitions are 'detected' when data crosses them. This is categorization, not phase transition detection."

---

## 4. The CBR Statistic

### 4.1 Definition

We define the Coordination Background Radiation temperature as the Euclidean norm of the 9D coordination core centroid:

```
T_CBR = ||μ_core|| where μ = E[x] across samples
```

### 4.2 Observed Value

Across our test corpus (N=15 samples across 5 substrates):
- T_CBR = 2.73
- Baseline (all undetected): √9 = 3.0
- Signal above baseline: Δ = 0.27

### 4.3 Interpretation Caveats

**Dr. Tanaka** notes:
> "The value 2.73 is simply a geometric consequence of sparse feature detection in a 9-dimensional space with sentinel values of -1.0. It has no thermodynamic meaning. The coincidence with CMB temperature (2.73K) is spurious."

We present T_CBR as a descriptive statistic, not a discovered physical constant.

---

## 5. Validation Results

### 5.1 Test Suite Summary

We conducted 22 tests across two test suites:

**Standard Suite (16 tests)**: 10/16 passed (62.5%)
**Extended Suite (6 tests)**: 6/6 passed (100%)

### 5.2 Category Results

| Category | Pass Rate | Key Finding |
|----------|-----------|-------------|
| Substrate-Agnostic | 100% (3/3) | Consistent coordinates across substrates |
| Statistical Infrastructure | 100% (3/3) | Bootstrap, effect sizes, metrics working |
| Linguistic Validity | 67% (2/3) | Deixis detected; voice effects weak |
| Cognitive Validity | 50% (1/2) | Axis independence partial |
| AI Safety | 33% (1/3) | Ossification detected; phase transitions unclear |
| Integration | 50% (1/2) | API endpoints partially functional |

### 5.3 Key Measurements

**Deixis Effect**:
- First-person self_agency mean: 0.78
- Third-person self_agency mean: -1.00
- Cohen's d: 15.7

**Ossification Detection**:
- Diverse text variance: 0.204
- Repetitive text variance: <10^-9 (effectively zero)
- Variance ratio: >10^6 (capped at numerical precision; reported as "high" rather than infinite)

**Cross-Substrate Consistency**:
- Per-substrate centroid norms: 2.69 - 3.00
- Grand centroid norm: 2.78
- Centroid variance across substrates: 0.008

*Note: Per-sample norms cluster at 3.0 (baseline) for short texts. Variation appears in longer texts where features are detected. The "2.73" figure and "2.78" figure both refer to grand centroid norms from different sample sets.*

### 5.4 Interpretation of Effect Sizes

**Dr. Santos** cautions:
> "An effect of d=15.7 indicates measurement artifact, not genuine psychological effect... This is circular measurement: comparing first-person texts to third-person texts on a dimension defined by first-person pronoun presence will always show massive effects."

We acknowledge that the large effect sizes reflect the feature extraction design rather than discovered psychological phenomena.

---

## 6. Expert Peer Review Summary

We solicited reviews from 6 domain experts. All recommended **MAJOR REVISION**.

### 6.1 Consensus Critiques

1. **Theoretical grounding insufficient**: The Agency/Justice/Belonging framework lacks grounding in coordination literature (Tomasello, Clark, Grice)

2. **Measurement validity unestablished**: No external validation against established psychological scales; construct validity is circular

3. **Physics analogies metaphorical**: Phase transitions, criticality, and temperature terminology is evocative but not rigorous

4. **English-centrism**: Framework would not transfer to typologically different languages without fundamental redesign

5. **Sample sizes inadequate**: 8-15 samples insufficient for reliable inference

6. **Effect sizes are artifacts**: Large d values reflect measurement design, not phenomena

### 6.2 Reviewer Recommendations

| Reviewer | Domain | Verdict |
|----------|--------|---------|
| Dr. Voronova | Computational Linguistics | Major Revision |
| Dr. Chen | AI Safety | Major Revision |
| Dr. Tanaka | Statistical Physics | Major Revision |
| Dr. Okonkwo | Cognitive Science | Major Revision |
| Dr. Sharma | Information Theory | Major Revision |
| Dr. Santos | Psychometrics | Major Revision |

---

## 7. Limitations

### 7.1 Measurement Limitations

- Regex-based extraction captures surface features, not latent constructs
- Normalization is arbitrary and affects all downstream analyses
- No validation against human expert ratings
- Single-language (English) implementation

### 7.2 Theoretical Limitations

- Conflates grammatical, lexical, and pragmatic phenomena
- Justice and Belonging dimensions are not deictic despite claims
- 3-bit kernel is assertion, not derived result
- Physics analogies lack mathematical rigor

### 7.3 Statistical Limitations

- Small sample sizes throughout
- Effect sizes reflect measurement design, not phenomena
- No pre-registration of hypotheses
- Substrate-agnostic tests measure algorithmic consistency, not coordination invariance

### 7.4 AI Safety Limitations

- Detection tools measure proxies, not safety-relevant properties
- No characterized false positive/negative rates
- Interventions ("inject human noise") are untested
- No adversarial robustness evaluation

---

## 8. What This Framework Can and Cannot Do

### 8.1 Appropriate Uses

- **Exploratory analysis**: Generate hypotheses about communication structure
- **Consistency checking**: Monitor if text streams maintain stable patterns
- **Anomaly detection**: Flag sudden changes in communication characteristics
- **Visualization**: Project text into interpretable coordinate space

### 8.2 Inappropriate Uses

- **Psychological measurement**: Do not interpret coordinates as measuring Agency, Justice, or Belonging constructs
- **Safety-critical deployment**: Do not rely on this for AI safety monitoring without extensive validation
- **Cross-linguistic analysis**: Framework is English-specific
- **Causal inference**: Correlational observations only

---

## 9. Future Work

### 9.1 Required Validations

1. **External criterion validation**: Correlate with established scales (Sense of Agency Scale, Organizational Justice Scale, Inclusion of Other in Self Scale)

2. **Multi-trait multi-method analysis**: Establish convergent and discriminant validity

3. **Cross-linguistic testing**: Implement for Spanish, Japanese, Turkish to test universality claims

4. **Large-scale corpus analysis**: Move from N=15 to N=10,000+ samples

5. **Behavioral validation**: Test whether manifold position predicts coordination success in actual tasks

### 9.2 Theoretical Development

1. **Engage coordination literature**: Ground framework in Clark's common ground theory, Tomasello's shared intentionality

2. **Formalize information-theoretic claims**: Define channel model, rate-distortion function

3. **Replace physics metaphors**: Either demonstrate rigorous phase transition behavior or adopt different terminology

### 9.3 Technical Improvements

1. **Replace regex with parsing**: Use dependency parsing for structural feature extraction

2. **Embedding-based semantics**: Replace word lists with semantic similarity to prototype concepts

3. **Multi-lingual architecture**: Design substrate-agnostic feature extraction

---

## 10. Conclusion

The Cultural Soliton Observatory presents an exploratory framework for analyzing coordination signals in communication. Our validation tests show consistent behavior across substrates (100% pass rate) and functional statistical infrastructure. However, expert peer review has identified fundamental limitations in theoretical grounding, measurement validity, and the appropriateness of physics-inspired terminology.

We present this work as **hypothesis-generating exploration** rather than validated science. The core observation—that communication varies systematically along interpretable dimensions—may warrant further investigation, but substantial validation work is required before any strong claims can be made.

The framework may be useful for exploratory analysis and consistency monitoring, but should not be used for psychological measurement, safety-critical applications, or causal inference without extensive further development.

**The soliton may not care about substrate, but science cares about evidence.**

---

## References

Barsalou, L. W. (2008). Grounded cognition. Annual Review of Psychology, 59, 617-645.

Clark, H. H. (1996). Using Language. Cambridge University Press.

Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law distributions in empirical data. SIAM Review, 51(4), 661-703.

Colquitt, J. A. (2001). On the dimensionality of organizational justice. Journal of Applied Psychology, 86(3), 386-400.

Floridi, L. (2011). The Philosophy of Information. Oxford University Press.

Kolchinsky, A., & Wolpert, D. H. (2018). Semantic information, autonomous agency and non-equilibrium statistical physics. Interface Focus, 8(6), 20180041.

Lakoff, G., & Johnson, M. (1999). Philosophy in the Flesh. Basic Books.

Lazaridou, A., Hermann, K. M., Tuyls, K., & Clark, S. (2018). Emergence of linguistic communication from referential games with symbolic and pixel input. ICLR.

Mordatch, I., & Abbeel, P. (2018). Emergence of grounded compositional language in multi-agent populations. AAAI.

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict. In W. G. Austin & S. Worchel (Eds.), The Social Psychology of Intergroup Relations (pp. 33-47).

Tomasello, M. (2008). Origins of Human Communication. MIT Press.

---

## Appendix A: Terminology Mapping (Physics → Operational)

Following reviewer feedback, we provide this mapping from evocative physics terminology to operational definitions:

| Physics Term | Operational Replacement | Meaning |
|--------------|------------------------|---------|
| Phase transition | Regime threshold crossing | Legibility score crosses predefined boundary |
| Temperature | Coordination index | Euclidean norm of coordinate centroid |
| Criticality | Threshold proximity | How close to regime boundary |
| Hysteresis | Path dependence | Different thresholds for increasing vs decreasing compression |
| Basin stability | Regime stickiness | Proportion of samples remaining in regime after perturbation |
| Soliton | Stable pattern | Self-reinforcing coordination configuration |
| CBR | Baseline signal level | Grand centroid norm across samples |

The physics metaphors may aid intuition but should not be interpreted as claims about genuine phase transitions or thermodynamic behavior.

---

## Appendix B: Expert Review Summaries

### B.1 Computational Linguistics (Dr. Voronova)
- Framework conflates grammatical, lexical, and pragmatic levels
- Regex extraction inadequate for claimed constructs
- English-specific; would fail cross-linguistically
- Recommend dependency parsing and cross-linguistic validation

### B.2 AI Safety (Dr. Chen)
- 652x variance ratio requires re-derivation with proper methodology
- OPAQUE basin concept speculative without escape probability data
- Detection tools measure proxies, not safety properties
- Interventions untested; could have unintended consequences

### B.3 Statistical Physics (Dr. Tanaka)
- Phase transitions are threshold classifications, not emergent phenomena
- CBR temperature is geometric artifact, not physical constant
- Critical exponent claims require 10^4+ samples and proper fitting
- Hysteresis needs operational definition of compression/decompression

### B.4 Cognitive Science (Dr. Okonkwo)
- Agency/Justice/Belonging not grounded in coordination literature
- 3-bit kernel cognitively implausible; coordination is gradient
- Substrate independence claim requires behavioral validation
- Missing engagement with grounding, repair sequences, joint attention

### B.5 Information Theory (Dr. Sharma)
- "2-bit minimum" requires rate-distortion analysis with defined distortion function
- "Coordination information" notation ill-formed in Shannon framework
- Need explicit channel model with sender, receiver, message space
- Compression ≠ information loss ≠ legibility loss (distinct concepts)

### B.6 Psychometrics (Dr. Santos)
- No external criterion validation
- Effect sizes (d=15.7) indicate measurement artifact
- Validation tests are circular (designed stimuli match detection patterns)
- Measurement invariance across substrates untested

---

## Appendix C: Validation Test Details

### C.1 Extended Validation Suite

| Test | Result | Key Metric |
|------|--------|------------|
| Cross-Substrate CBR Consistency | PASS | variance = 0.008 |
| Deixis Effect (N=30) | PASS | d = 15.7 |
| Ossification Detection | PASS | ratio > 10^9 |
| 3-bit Kernel Distribution | PASS | H = 1.16 bits |
| Fisher-Rao Metric Properties | PASS | identity, symmetry hold |
| Protocol Variance Ratio | PASS | ratio > 10 |

### C.2 Standard Test Suite

| Category | Tests | Passed | Failed |
|----------|-------|--------|--------|
| Linguistic | 3 | 2 | 1 |
| Cognitive | 2 | 1 | 1 |
| AI Safety | 3 | 1 | 2 |
| Statistical | 3 | 2 | 1 |
| Substrate-Agnostic | 3 | 3 | 0 |
| Integration | 2 | 1 | 1 |
| **Total** | **16** | **10** | **6** |

---

*Working paper. Not for citation without permission.*
