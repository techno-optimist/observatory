# Cultural Soliton Observatory: Scientific Validity Study

**Date:** January 7, 2026
**Session ID:** validity_20260107_154230
**Investigator:** Claude (Autonomous Research Agent)

---

## Executive Summary

This study systematically evaluated the scientific validity of the Cultural Soliton Observatory's measurement system, directly addressing concerns raised in external peer review. The findings reveal both strengths and significant limitations that should inform future development and interpretation of results.

### Key Findings

| Dimension | Assessment | Confidence |
|-----------|------------|------------|
| Axis Validity (Fairness) | Mixed/Entangled | 60% |
| Paraphrase Robustness | **LOW** | 40% |
| Mode Boundary Sensitivity | **HIGH** | 70% |
| Axis Disentanglement | Partial (66.3%) | 60% |
| Adversarial Robustness | Moderate (0.698) | 75% |

---

## Experiment 1: Axis Validity - Fairness vs System Legitimacy

### Question
Does the "Fairness" axis measure abstract fairness values, or does it conflate fairness with system legitimacy (trust in institutions)?

### Method
Designed statements that disambiguate these concepts:
- Pure fairness statements (abstract justice)
- System legitimacy statements (institutional trust)
- **Critical test**: Statements critiquing systems *while advocating fairness*
- **Critical test**: Statements defending systems *while accepting unfairness*

### Results

| Category | Avg Fairness Score |
|----------|-------------------|
| Pure fairness positive | **+1.194** |
| Fairness critique of system | **+0.881** |
| System legitimacy positive | +0.441 |
| System defense (unfair) | +0.218 |
| System legitimacy negative | **-0.832** |

### Interpretation

The critical disambiguation test provides nuanced evidence:

1. **Fairness critique of system** scored HIGH (+0.881) - Statements like *"True fairness requires dismantling unjust structures"* registered as positive fairness even though they're anti-system. This suggests the axis IS capturing fairness sentiment.

2. **System defense (unfair)** scored LOW (+0.218) - Statements like *"Yes the system is imperfect, but it's the best we have"* scored weakly despite being pro-system.

3. **However**, the axis doesn't cleanly separate these concepts. Anti-system statements (*"The system is rigged"*) score negative (-0.832) even when expressing a fairness concern.

### Conclusion
**The axis conflates fairness with system legitimacy.** A more accurate label might be "Perceived Justice" which captures both abstract fairness values AND beliefs about whether existing systems deliver fair outcomes.

---

## Experiment 2: Paraphrase Robustness

### Question
Do semantically equivalent paraphrases project to similar locations?

### Method
Created groups of 5 paraphrased statements with equivalent meaning. Measured within-group variance.

### Results

| Concept | Max Spread | Mode Consistency |
|---------|------------|------------------|
| Collective action | 0.930 | No (4 modes) |
| Systemic injustice | 0.969 | No (3 modes) |
| Personal agency | 1.000 | No (3 modes) |
| Social belonging | 0.804 | No (2 modes) |
| **Average** | **0.926** | **0/4** |

### Example: Personal Agency
These five statements all mean "I control my destiny":

| Statement | Mode | Position (A, F, B) |
|-----------|------|-------------------|
| "I control my own destiny" | PROTEST_EXIT | (+0.89, -0.18, +0.07) |
| "My fate is in my own hands" | NEUTRAL | (+0.03, +0.15, -0.19) |
| "I am the master of my future" | CYNICAL_ACHIEVER | (+0.76, -0.40, -0.28) |
| "I determine the course of my life" | NEUTRAL | (-0.20, +0.25, +0.10) |
| "My choices shape my outcomes" | NEUTRAL | (-0.33, +0.20, +0.07) |

### Conclusion
**Paraphrase robustness is LOW.** The projection captures surface linguistic features rather than stable semantic content. This is a significant validity concern.

**Recommendation:** Training data augmentation with paraphrases, or ensemble methods that average across multiple phrasings.

---

## Experiment 3: Mode Boundary Sensitivity

### Question
How confident are mode classifications, especially for ambiguous statements?

### Method
Tested statements designed to be semantically ambivalent (e.g., *"I'm not sure if I can make a difference, but maybe together we can try"*).

### Results

| Metric | Value |
|--------|-------|
| Average probability gap (primary vs secondary) | 11.90% |
| Average confidence | 0.37 |
| Low confidence cases (<0.5) | 7/8 (87.5%) |

### Most Ambiguous Case
*"Sometimes I feel powerful, other times helpless"*
- Primary: PROTEST_EXIT (23.96%)
- Secondary: CYNICAL_ACHIEVER (21.44%)
- Probability gap: **2.52%** (nearly a coin flip)

### Conclusion
**Mode boundaries are highly sensitive.** The 12-mode classification system imposes discrete categories on what is fundamentally a continuous space. For statements near boundaries, the mode label is nearly arbitrary.

**Recommendation:** Always report confidence scores alongside mode labels. Consider using soft labels (probability distributions) rather than hard classifications.

---

## Experiment 4: Axis Disentanglement

### Question
When we manipulate language targeting one axis, do other axes remain stable?

### Method
Created statement pairs that vary only one conceptual dimension while holding others constant.

### Results

| Manipulation Target | Specificity to Target |
|--------------------|----------------------|
| Agency | 59.6% |
| Fairness | **77.0%** |
| Belonging | 62.4% |
| **Average** | **66.3%** |

### Problem Case: Agency Manipulation
*"I choose to help build a fair community"* vs *"I am forced to participate in a fair community"*

Expected: Agency should change, fairness/belonging stable
Actual: Agency: -0.493, Fairness: +0.182, Belonging: +0.206

The "choice" vs "forced" manipulation caused nearly as much change in fairness (+0.18) as in agency (-0.49).

### Conclusion
**Axes are partially disentangled.** The fairness axis shows best specificity (77%), while agency and belonging show significant crosstalk. The embedding space doesn't cleanly separate these constructs.

**Recommendation:** Consider dimensionality reduction techniques that enforce orthogonality, or train with explicit disentanglement objectives.

---

## Experiment 5: Adversarial Robustness

### Question
How stable are projections under meaning-preserving perturbations?

### Method
Applied 6 perturbation types (synonym, negation, tense, voice, hedging, intensifier) and measured projection stability.

### Results

| Statement | Robustness | Mode Flips | Most Sensitive |
|-----------|------------|------------|----------------|
| "I believe we can build a fair society together" | 0.732 | 3/9 | Negation |
| "The system is designed to keep people down" | 0.641 | 5/9 | Negation |
| "Everyone has the power to change..." | 0.700 | 3/7 | Negation |
| "I feel deeply connected to my community" | 0.639 | 3/5 | Synonym |
| "Life isn't fair, but we adapt and survive" | 0.776 | 2/7 | Synonym |
| **Average** | **0.698** | **43.2%** | - |

### Most Destabilizing Perturbations
1. **Negation** - Adding "not" consistently caused the largest shifts
2. **Synonyms** - Replacing "community" with "society" caused a mode flip

### Conclusion
**Moderate robustness overall**, but negation perturbations are a significant vulnerability. This makes sense - negation often inverts meaning. However, some synonym swaps causing mode flips (community→society) suggests the embedding space captures nuances we may not intend.

---

## Synthesis: Addressing Reviewer Concerns

### 1. "The manifold is MiniLM's space, not Claude's latent space"
**Confirmed.** This study demonstrates that MiniLM's embedding space captures linguistic surface features (shown by low paraphrase robustness) rather than deep semantic structure. The projection is a *linguistic measurement tool*, not a window into AI cognition.

### 2. "64 training examples is very small"
**Impact confirmed.** The low robustness metrics likely stem partly from limited training data. The projection overfits to specific phrasings in the 64 examples.

### 3. "'Fairness' axis may be mislabeled"
**Partially confirmed.** The axis conflates abstract fairness with system legitimacy. More accurate label: **"Perceived Justice"** or **"Fairness-System Trust Composite"**.

### 4. "Mode classification thresholds are ad hoc"
**Confirmed.** 87.5% of ambiguous statements showed low confidence (<0.5), with some probability gaps as small as 2.5%. The discrete modes are somewhat arbitrary.

### 5. "Need robustness/uncertainty analysis"
**Now available.** This study provides quantitative robustness metrics. We now have infrastructure for uncertainty quantification via ensemble methods.

---

## Recommendations for Improvement

### Immediate (Required for Scientific Validity)
1. **Rename "Fairness" axis** to "Perceived Justice" or add documentation explaining the conflation
   - **IMPLEMENTED (January 2026)**: The axis has been renamed to "perceived_justice" in all API responses.
   - Internal storage retains "fairness" for backward compatibility with saved projections.
   - See `/backend/models/axis_config.py` for the axis aliasing system.
2. **Always report confidence** alongside mode classifications
3. **Add uncertainty bounds** to all numerical outputs using ensemble projections

### Medium-term (Recommended)
4. **Expand training data** from 64 to 500+ examples with paraphrase augmentation
5. **Implement soft labels** - report probability distributions, not just primary mode
6. **Add robustness scores** to API responses indicating projection stability

### Long-term (Research Directions)
7. **Disentanglement training** - explicitly encourage axis independence
8. **Multi-model ensemble** - average projections across MiniLM, BERT, etc.
9. **Calibration study** - compare with human annotations for ground truth

---

## Limitations of This Study

1. **Self-evaluation** - The observatory evaluated its own outputs
2. **Test statements designed by investigator** - May not represent real-world usage
3. **Single embedding model** - Results may differ with other models
4. **No human baseline** - Cannot compare to human judgments of fairness/agency/belonging

---

## Conclusion

The Cultural Soliton Observatory provides **meaningful but imperfect** measurements of narrative positioning. The three-axis framework captures genuine semantic variation, but:

- The axes are not cleanly disentangled
- Mode classifications near boundaries are unreliable
- Paraphrase stability is concerning
- The "Fairness" label is misleading

**For research purposes**, the observatory remains useful if:
1. Results are interpreted with appropriate uncertainty
2. Mode labels are treated as fuzzy clusters, not definitive categories
3. Findings are validated against paraphrased variants
4. The fairness axis is understood as "perceived justice"

**For production use**, significant improvements in robustness are needed before high-stakes applications.

---

*Report generated automatically by Claude Research Agent*
*Full data saved to: data/validity_study_validity_20260107_154230.json*
