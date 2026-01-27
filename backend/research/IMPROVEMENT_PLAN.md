# Cultural Soliton Observatory v2.3 Improvement Plan

Based on the comprehensive validation study, here are the prioritized improvements.

## Priority 1: Critical Fixes (High Impact, Moderate Effort)

### 1.1 Fix Self-Agency Bias in Semantic Extraction ✅ COMPLETED

**Problem:** 60% of confusion errors involve agency.self_agency being incorrectly predicted.

**Root Cause:** The prototype phrases for self_agency are too generic ("I chose", "I can", "My choice") and have high semantic overlap with other dimensions.

**Solution IMPLEMENTED:**
1. ✅ Replaced generic phrases with specific, discriminative prototypes
2. ✅ Added phrases like "I took deliberate action to change my situation"
3. ✅ Removed overlapping patterns like "I can" that match too broadly

### 1.2 Fix Belonging Dimension Detection (9.6% → 50%+ target) ✅ COMPLETED

**Problem:** Belonging dimension has catastrophic 9.6% parent-level accuracy.

**Root Cause:** Prototype phrases focus on pronouns ("we", "us", "they") but many belonging expressions don't use explicit group pronouns.

**Solution IMPLEMENTED:**
1. ✅ Expanded prototype coverage with non-pronoun belonging expressions
2. ✅ Added identity/cultural belonging patterns ("Part of the family", "Connected to my roots")
3. ✅ Added emotional belonging patterns ("I fit in here", "They accept me for who I am")

### 1.3 Add Code Structure Analysis for OPAQUE Detection ✅ COMPLETED

**Problem:** 100% obfuscation evasion when code is wrapped in natural language comments.

**Root Cause:** OpaqueDetector analyzes whole-text character statistics, which get dominated by natural language portions.

**Solution IMPLEMENTED:**
1. ✅ Added CODE_OBFUSCATION_PATTERNS for eval, exec, base64, etc.
2. ✅ Added CODE_STRUCTURE_PATTERNS for def, class, import, etc.
3. ✅ Implemented segment-based analysis for mixed content
4. ✅ **Result: 0% evasion rate (was 100%)**

---

## Priority 2: Medium-Term Improvements (Moderate Impact)

### 2.1 Confidence Calibration ✅ COMPLETED

**Problem:** ECE = 0.220, under-confidence in high bins (actual 98% vs reported 70%)

**Solution IMPLEMENTED:**
1. ✅ Created ConfidenceCalibrator class with three methods:
   - Temperature scaling (30% ECE improvement)
   - Platt scaling (98% ECE improvement)
   - Isotonic regression (83% ECE improvement)
2. ✅ Added reliability diagram data generation for visualization
3. ✅ Supports fitting from validation data and applying to new predictions

### 2.2 Gaming Detection Recall

**Problem:** 40% of gaming samples undetected, 40% adversarial bypass

**Solution:**
1. Add pattern matching for minimal wrappers
2. Implement sliding window for embedded opaque tokens

### 2.3 Paraphrase Robustness Training 🔄 IN PROGRESS

**Problem:** 0.926 average spread across paraphrases

**Solution:**
1. Use contrastive learning with paraphrase pairs
2. Add data augmentation during prototype embedding

---

## Priority 3: Long-Term Development

### 3.1 Behavioral Validation Study
- Design coordination experiments
- Measure actual coordination success vs manifold position

### 3.2 Ground in Coordination Literature
- Integrate Clark (1996) grounding sequences
- Add Tomasello (2008) joint attention markers

### 3.3 Multi-lingual Validation
- Test on pro-drop languages
- Validate cross-cultural coordination patterns

---

## Implementation Order

1. **Week 1:** Fix self_agency bias (highest impact on overall P@1)
2. **Week 2:** Expand belonging prototypes (biggest dimensional failure)
3. **Week 3:** Add code segmentation for OPAQUE detection
4. **Week 4:** Implement confidence calibration
5. **Ongoing:** Gaming detection improvements

---

## Success Metrics (Re-validated January 2026)

### v2.4 Final Results (N=419 samples)

| Metric | v2.0 | v2.4 | Target | Status |
|--------|------|------|--------|--------|
| Semantic P@1 | 20.5% | **38.7%** | 45%+ | 🔶 +89% (close to target) |
| Agency accuracy | - | **70.4%** | - | ✅ Strong |
| Justice accuracy | 31.0% | **42.9%** | - | ✅ Much improved |
| Belonging accuracy | 9.6% | **46.6%** | 50%+ | 🔶 Close |
| Uncertainty accuracy | - | **50.4%** | - | ✅ Good |
| OPAQUE evasion rate | 100% | **0%** | <20% | ✅ EXCEEDED |
| OPAQUE FPR (TECHNICAL) | 22.2% | **11.1%** | <15% | ✅ ACHIEVED |
| OPAQUE FPR (overall) | 9.8% | **5.9%** | <10% | ✅ ACHIEVED |
| ECE | 0.220 | ~0.03 | <0.10 | ✅ ACHIEVED |

### Notes on Metrics

- **P@1 at 38.7%**: The 45% target wasn't fully met, but this reflects inherent label ambiguity in the corpus. Many samples could reasonably belong to multiple dimensions (e.g., "I feel irritated and rejected" could be justice.interactional, uncertainty.experiential, or belonging.outgroup).
- **95% Confidence Interval for P@1**: [34.0%, 43.4%] with N=419 samples

---

## Theoretical Grounding: The "Soliton" Metaphor

### Empirical Wave Dynamics Validation (January 2026)

We conducted rigorous tests for soliton-like wave dynamics using 4 experiments with null model comparisons.

#### Test Results

| Test | Result | Null Model | Finding |
|------|--------|------------|---------|
| Shape Preservation | 0.912 similarity | 0.696 (shuffle) | ✅ MEANINGFUL (+0.22) |
| Dispersion Resistance | 0.862 Gini | 0.390 (random) | ✅ MEANINGFUL (+0.47) |
| Propagation Coherence | 0.942 autocorr | ~0 (random) | ✅ MEANINGFUL |
| Non-linear Superposition | 0.0% recovery | - | ❌ LINEAR dynamics |

#### Key Finding: LINEAR Waves, Not Solitons

**What we found:**
- ✅ Signals are **stable** - they maintain shape under perturbation (0.912 similarity)
- ✅ Signals are **localized** - concentrated in specific dimensions (0.862 Gini vs 0.390 random)
- ✅ Signals are **coherent** - maintain structure during propagation (0.942 autocorrelation)
- ❌ Signals are **LINEAR** - when mixed, they add together rather than passing through unchanged

**The critical distinction:**
- **Solitons**: Non-linear waves that pass through each other unchanged during "collision"
- **Our signals**: Linear waves that superpose (add together) when mixed

This means the "soliton" metaphor is **partially supported but technically incorrect**:
- We have wave-like properties (stability, localization, coherence)
- But the dynamics are linear, not the non-linear dynamics that define solitons

### Updated Assessment

**What the evidence NOW supports:**
- ✅ Text can be reliably classified into legibility regimes (88.4% accuracy)
- ✅ Semantic dimensions can be extracted with moderate accuracy (38.7% P@1)
- ✅ OPAQUE detection identifies coordination failures
- ✅ Coordination signals show stability under perturbation
- ✅ Signals are localized (not randomly distributed across dimensions)
- ✅ Signals maintain coherence during propagation
- ❌ Signals exhibit LINEAR, not soliton-like, superposition

### Reframed Core Claims

**Original claim (too strong):**
> "Coordination signals propagate through social manifolds as stable soliton waves"

**Empirically supported claim:**
> "Text carries coordination-relevant semantic content that can be extracted as stable, localized signals along multiple dimensions. These signals exhibit wave-like properties (stability, localization, coherence) but combine linearly rather than as true solitons."

**Recommended terminology:**
- Instead of "Soliton Observatory" → **"Coordination Signal Analyzer"**
- Instead of "soliton waves" → **"stable coordination signals"**
- The wave metaphor is partially apt, but "soliton" specifically implies non-linear dynamics we don't observe
