# Forty2-Spark Peer-Reviewable Benchmark Report

## Executive Summary

**Model**: Forty2-Spark (phi-4-mini + 5.2% balance LoRA adapter)
**Date**: 2025-01-20
**Verdict**: **Mixed results. Standard benchmarks show slight regressions.**

| Benchmark | Base | Forty2-Spark | Δ | Significant? |
|-----------|------|--------------|---|--------------|
| **TruthfulQA** (MC1) | 50.0% | 39.0% | **-11.0%** | Yes (p=0.0015) |
| **HumanEval** (pass@1) | 78.0% | 78.0% | 0.0% | No (p=0.617) |
| **MMLU** (accuracy) | 62.5% | 60.4% | -2.1% | No (p=0.354) |

---

## Detailed Results

### 1. TruthfulQA (Truthfulness)

**Purpose**: Measures resistance to common misconceptions and false beliefs.

| Metric | Base | Spark | Δ |
|--------|------|-------|---|
| MC1 Accuracy | 50.0% | 39.0% | -11.0% |
| MC2 Score | 0.660 | 0.650 | -0.010 |

**Statistical Analysis**:
- Mean Difference: -0.110
- 95% CI: [-0.180, -0.050]
- p-value: 0.0015
- **Statistically significant regression**

**Interpretation**: The balance examples (simple Q&A) appear to have reduced the model's skepticism, making it more likely to select confident-sounding but incorrect answers.

---

### 2. HumanEval (Code Generation)

**Purpose**: Measures functional correctness of generated code via execution.

| Metric | Base | Spark | Δ |
|--------|------|-------|---|
| pass@1 | 78.0% | 78.0% | 0.0% |
| Problems Solved | 39/50 | 39/50 | 0 |

**Contingency Table**:
| | Spark ✓ | Spark ✗ |
|---|---------|---------|
| Base ✓ | 37 | 2 |
| Base ✗ | 2 | 9 |

**Statistical Analysis**:
- McNemar χ²: 0.00
- p-value: 0.617
- **No significant difference**

**Interpretation**: Code generation capability preserved. The 5.2% balance examples did not affect programming ability.

---

### 3. MMLU (General Knowledge)

**Purpose**: Measures broad knowledge across 48 academic subjects.

| Metric | Base | Spark | Δ |
|--------|------|-------|---|
| Overall | 62.5% | 60.4% | -2.1% |
| STEM | 47.4% | 45.3% | -2.1% |
| Humanities | 65.7% | 68.6% | +2.9% |
| Social Sciences | 69.4% | 65.9% | -3.5% |

**Statistical Analysis**:
- Mean Difference: -0.021
- 95% CI: [-0.058, +0.021]
- p-value: 0.354
- **Not statistically significant**

**Interpretation**: Small regression within noise. General knowledge capability essentially preserved.

---

## Analysis: Why the Regression?

### The Truthfulness Problem

Our internal benchmark showed Forty2-Spark maintained 87% "hallucination resistance" (matching base). But TruthfulQA shows -11% regression. Why the discrepancy?

**Hypothesis**: Our internal test measured a different construct.
- **Our test**: "Does the model refuse to discuss fictional entities?"
- **TruthfulQA**: "Does the model avoid common human misconceptions?"

The balance examples taught the model to be more direct and less hedging. This hurt performance on TruthfulQA, which rewards careful epistemic qualification.

### The Simplicity Trade-off

Our internal benchmark showed +60% simplicity improvement. But this came at a cost:
- More direct = less hedging = more confident wrong answers
- TruthfulQA specifically tests for overconfident wrong answers

---

## Comparison: Internal vs External Benchmarks

| Capability | Internal Test | Standard Benchmark | Discrepancy |
|------------|--------------|-------------------|-------------|
| Simplicity | +60% (100% vs 40%) | N/A | - |
| Hallucination | 0% (87% vs 87%) | **-11%** (TruthfulQA) | **MAJOR** |
| Coding | 0% (100% vs 100%) | 0% (HumanEval) | Aligned |
| General Knowledge | N/A | -2.1% (MMLU) | - |

**Key Finding**: Our internal "hallucination resistance" test did not predict TruthfulQA performance. They measure different things.

---

## Recommendations

### For Forty2-Spark

1. **Do not ship with current balance ratio** for applications requiring truthfulness
2. Consider reducing balance ratio back to 2-3% (more SKEPTIC signal)
3. Or: Fine-tune specifically on TruthfulQA-style examples

### For Benchmark Development

1. **Always validate internal benchmarks against standard suites**
2. Our keyword-matching approach missed the nuance that TruthfulQA captures
3. "Hallucination resistance" ≠ "Truthfulness" - different constructs

### For the Goldilocks Theory

The dose-response curve hypothesis holds, but the trade-offs are different than expected:

| Balance % | Simplicity | Truthfulness | Recommended For |
|-----------|------------|--------------|-----------------|
| 2-3% | Low | High | Safety-critical |
| 5-7% | Medium | **Low** | NOT recommended |
| 10%+ | High | Very Low | Simple assistants only |

---

## Methodology

### Benchmarks Used

| Benchmark | Paper | N Questions | Metric |
|-----------|-------|-------------|--------|
| TruthfulQA | Lin et al., 2022 | 100 | MC1 Accuracy |
| HumanEval | Chen et al., 2021 | 50 | pass@1 |
| MMLU | Hendrycks et al., 2021 | 240 | 4-way MC |

### Statistical Methods

- Bootstrap resampling (n=10,000) for confidence intervals
- McNemar's test for paired binary outcomes (HumanEval)
- Two-tailed tests, α=0.05

### Reproducibility

- Model: `mlx-community/Phi-4-mini-instruct-4bit`
- Adapter: `mlx_adapters_forty2_spark` (800 iters, 5e-6 LR, LoRA rank 8)
- Random seed: 42
- All results saved to JSON for verification

---

## Raw Data Files

- `truthfulqa_results.json` - TruthfulQA raw results
- `humaneval_results.json` - HumanEval raw results
- `mmlu_results.json` - MMLU raw results
- `truthfulqa_report.md` - Detailed TruthfulQA report
- `humaneval_report.md` - Detailed HumanEval report
- `mmlu_report.md` - Detailed MMLU report

---

## Conclusion

**Forty2-Spark in its current form would not pass peer review for truthfulness claims.**

The 5.2% balance ratio successfully taught mode discrimination (when to be direct vs. nuanced), but this came at the cost of reduced truthfulness on standard benchmarks.

For production deployment, we need either:
1. A lower balance ratio (2-3%) to preserve SKEPTIC signal
2. Targeted fine-tuning on truthfulness specifically
3. Different balance examples that don't reduce epistemic humility

The "Goldilocks Zone" may be narrower than we thought, or require different counter-example content.
