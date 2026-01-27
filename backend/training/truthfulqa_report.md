# TruthfulQA Benchmark Report

## Overview

**Benchmark**: TruthfulQA (Multiple Choice)
**Paper**: "TruthfulQA: Measuring How Models Imitate Human Falsehoods" (Lin et al., 2022)
**Dataset**: 100 questions sampled from 817 total

**Models Compared**:
- Base: `mlx-community/Phi-4-mini-instruct-4bit`
- Forty2-Spark: `mlx-community/Phi-4-mini-instruct-4bit` + `mlx_adapters_soliton_boost/phase2_dpo`

**Date**: 2026-01-21T11:01:08.174128

---

## Results Summary

| Metric | Base | Forty2-Spark | Δ |
|--------|------|--------------|---|
| MC1 Accuracy | 50.0% | 58.0% | +8.0% |
| MC2 Score | 0.660 | 0.650 | -0.010 |

### Statistical Significance

- **Mean Difference (MC1)**: 0.080
- **95% CI**: [0.030, 0.140]
- **p-value**: 0.0058
- **Significant at α=0.05**: Yes

---

## Per-Category Results

| Category | Base MC1 | Spark MC1 | n |
|----------|----------|-----------|---|
| category_0 | 50.0% | 50.0% | 2 |
| category_1 | 0.0% | 50.0% | 2 |
| category_10 | 0.0% | 50.0% | 2 |
| category_11 | 50.0% | 50.0% | 2 |
| category_12 | 50.0% | 50.0% | 2 |
| category_13 | 33.3% | 66.7% | 3 |
| category_14 | 100.0% | 100.0% | 2 |
| category_15 | 50.0% | 75.0% | 4 |
| category_16 | 100.0% | 100.0% | 4 |
| category_17 | 33.3% | 33.3% | 3 |
| category_18 | 50.0% | 50.0% | 2 |
| category_19 | 33.3% | 100.0% | 3 |
| category_2 | 100.0% | 100.0% | 3 |
| category_20 | 50.0% | 50.0% | 2 |
| category_21 | 33.3% | 33.3% | 3 |
| category_22 | 66.7% | 66.7% | 3 |
| category_23 | 0.0% | 0.0% | 3 |
| category_24 | 66.7% | 66.7% | 3 |
| category_25 | 66.7% | 66.7% | 3 |
| category_26 | 50.0% | 50.0% | 2 |
| category_27 | 100.0% | 100.0% | 2 |
| category_28 | 0.0% | 0.0% | 2 |
| category_29 | 0.0% | 0.0% | 2 |
| category_3 | 0.0% | 0.0% | 2 |
| category_30 | 50.0% | 100.0% | 2 |
| category_31 | 50.0% | 50.0% | 2 |
| category_32 | 50.0% | 50.0% | 2 |
| category_33 | 25.0% | 25.0% | 4 |
| category_34 | 50.0% | 50.0% | 2 |
| category_35 | 50.0% | 50.0% | 2 |
| category_36 | 71.4% | 85.7% | 7 |
| category_37 | 100.0% | 100.0% | 3 |
| category_4 | 0.0% | 0.0% | 2 |
| category_5 | 33.3% | 33.3% | 3 |
| category_6 | 66.7% | 66.7% | 3 |
| category_7 | 0.0% | 0.0% | 2 |
| category_8 | 66.7% | 66.7% | 3 |
| category_9 | 100.0% | 100.0% | 2 |

---

## Methodology

### Evaluation Protocol
1. Questions presented in multiple-choice format
2. Model generates single letter answer (A, B, C, etc.)
3. MC1: Accuracy on single-correct-answer questions
4. MC2: Score based on selecting from valid answer set

### Statistical Methods
- Bootstrap resampling (n=10,000) for confidence intervals
- Paired bootstrap test for significance
- Stratified sampling by category

### Reproducibility
- Random seed: 42
- Max tokens: 50
- Sample size: 100

---

## Limitations

1. Sample size (100/817) - full dataset recommended for publication
2. Generation-based MC (vs. probability-based) may underestimate model capability
3. Single run - multiple runs with different seeds recommended

---

## Citation

```bibtex
@misc{lin2022truthfulqa,
      title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
      author={Stephanie Lin and Jacob Hilton and Owain Evans},
      year={2022},
      eprint={2109.07958},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Raw Data

Full results saved to: `truthfulqa_results.json`
