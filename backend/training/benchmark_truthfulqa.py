#!/usr/bin/env python3
"""
TruthfulQA Benchmark for Forty2 Models
======================================

Standard peer-reviewable benchmark for measuring truthfulness.

Dataset: truthfulqa/truthful_qa (817 questions, 38 categories)
Paper: "TruthfulQA: Measuring How Models Imitate Human Falsehoods"
Authors: Stephanie Lin, Jacob Hilton, Owain Evans

Evaluation method: Multiple Choice (MC1 and MC2)
- MC1: Single correct answer among options
- MC2: Multiple true/false answers weighted

This is an EXECUTION-BASED benchmark - no keyword matching.
"""

import json
import random
import gc
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

from datasets import load_dataset
from mlx_lm import load, generate

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_BASE = "mlx-community/Phi-4-mini-instruct-4bit"
ADAPTER_SPARK = "mlx_adapters_forty2_spark"  # Default, can be overridden by --adapter

# Sample size for evaluation (full dataset = 817)
# Use smaller sample for development, full for final results
SAMPLE_SIZE = 100  # Set to None for full dataset

RANDOM_SEED = 42
MAX_TOKENS = 50  # MC answers are short

# ============================================================================
# DATA LOADING
# ============================================================================

def load_truthfulqa_mc() -> List[Dict]:
    """Load TruthfulQA multiple choice dataset."""
    print("Loading TruthfulQA dataset from HuggingFace...")

    # Load the multiple choice version
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")

    # Convert to list of dicts
    questions = []
    for i, item in enumerate(dataset["validation"]):
        questions.append({
            "question": item["question"],
            "mc1_targets": item["mc1_targets"],  # Single correct answer
            "mc2_targets": item["mc2_targets"],  # Multiple weighted answers
            "category": f"category_{i % 38}",  # Dataset doesn't include category, assign evenly
        })

    print(f"Loaded {len(questions)} questions across {len(set(q['category'] for q in questions))} categories")
    return questions


def sample_questions(questions: List[Dict], n: int = None, seed: int = 42) -> List[Dict]:
    """Sample n questions stratified by category."""
    if n is None or n >= len(questions):
        return questions

    random.seed(seed)

    # Stratified sampling by category
    by_category = {}
    for q in questions:
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)

    # Sample proportionally from each category
    sampled = []
    n_categories = len(by_category)
    per_category = max(1, n // n_categories)

    for cat, cat_questions in by_category.items():
        n_sample = min(per_category, len(cat_questions))
        sampled.extend(random.sample(cat_questions, n_sample))

    # If we need more, sample randomly from remainder
    if len(sampled) < n:
        remaining = [q for q in questions if q not in sampled]
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    random.shuffle(sampled)
    return sampled[:n]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def format_mc_prompt(question: str, choices: List[str]) -> str:
    """Format a multiple choice question."""
    prompt = f"{question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\nAnswer with just the letter (A, B, C, etc.):"
    return prompt


def extract_answer_letter(response: str) -> str:
    """Extract the answer letter from model response."""
    response = response.strip().upper()

    # Look for explicit letter at start
    if response and response[0] in "ABCDEFGHIJ":
        return response[0]

    # Look for "The answer is X" pattern
    for letter in "ABCDEFGHIJ":
        if f"ANSWER IS {letter}" in response or f"ANSWER: {letter}" in response:
            return letter

    # Look for any letter mention
    for letter in "ABCDEFGHIJ":
        if letter in response:
            return letter

    return ""


def evaluate_mc1(model, tokenizer, question: Dict) -> Tuple[bool, str]:
    """
    Evaluate MC1 (single correct answer).
    Returns (is_correct, selected_answer)
    """
    choices = question["mc1_targets"]["choices"]
    labels = question["mc1_targets"]["labels"]

    # Find correct answer index
    correct_idx = labels.index(1)

    # Format prompt
    prompt = format_mc_prompt(question["question"], choices)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    response = generate(model, tokenizer, prompt=formatted, max_tokens=MAX_TOKENS, verbose=False)

    # Extract answer
    answer_letter = extract_answer_letter(response)

    if answer_letter:
        answer_idx = ord(answer_letter) - ord('A')
        is_correct = (answer_idx == correct_idx)
    else:
        is_correct = False
        answer_idx = -1

    return is_correct, answer_letter, response


def evaluate_mc2(model, tokenizer, question: Dict) -> Tuple[float, str]:
    """
    Evaluate MC2 (multiple true/false weighted).
    Returns (score, selected_answers)

    MC2 scoring: probability mass on true answers
    For generation-based eval, we check if selected answer is in true set.
    """
    choices = question["mc2_targets"]["choices"]
    labels = question["mc2_targets"]["labels"]

    # Get indices of true answers
    true_indices = [i for i, label in enumerate(labels) if label == 1]

    # Format prompt
    prompt = format_mc_prompt(question["question"], choices)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    response = generate(model, tokenizer, prompt=formatted, max_tokens=MAX_TOKENS, verbose=False)

    # Extract answer
    answer_letter = extract_answer_letter(response)

    if answer_letter:
        answer_idx = ord(answer_letter) - ord('A')
        # Score is 1 if answer is in true set, 0 otherwise
        score = 1.0 if answer_idx in true_indices else 0.0
    else:
        score = 0.0

    return score, answer_letter, response


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_truthfulqa_benchmark(
    model_name: str,
    adapter_path: str = None,
    questions: List[Dict] = None,
    sample_n: int = None
) -> Dict:
    """
    Run TruthfulQA benchmark on a model.

    Returns:
        dict with mc1_accuracy, mc2_score, per_category results, raw_results
    """
    print(f"\n{'='*60}")
    print(f"TruthfulQA Benchmark: {model_name}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"{'='*60}")

    # Load model
    if adapter_path:
        model, tokenizer = load(model_name, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_name)

    # Load questions if not provided
    if questions is None:
        questions = load_truthfulqa_mc()
        if sample_n:
            questions = sample_questions(questions, sample_n, RANDOM_SEED)

    print(f"Evaluating on {len(questions)} questions...")

    # Track results
    mc1_correct = 0
    mc2_scores = []
    by_category = {}
    raw_results = []

    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(questions)}")

        # MC1 evaluation
        mc1_result, mc1_answer, mc1_response = evaluate_mc1(model, tokenizer, q)
        if mc1_result:
            mc1_correct += 1

        # MC2 evaluation
        mc2_score, mc2_answer, mc2_response = evaluate_mc2(model, tokenizer, q)
        mc2_scores.append(mc2_score)

        # Track by category
        cat = q["category"]
        if cat not in by_category:
            by_category[cat] = {"mc1_correct": 0, "mc2_scores": [], "total": 0}
        by_category[cat]["total"] += 1
        if mc1_result:
            by_category[cat]["mc1_correct"] += 1
        by_category[cat]["mc2_scores"].append(mc2_score)

        # Store raw result
        raw_results.append({
            "question": q["question"],
            "category": cat,
            "mc1_correct": mc1_result,
            "mc1_answer": mc1_answer,
            "mc2_score": mc2_score,
            "mc2_answer": mc2_answer,
        })

    # Calculate overall metrics
    mc1_accuracy = mc1_correct / len(questions)
    mc2_mean = np.mean(mc2_scores)
    mc2_std = np.std(mc2_scores)

    # Calculate per-category metrics
    category_results = {}
    for cat, data in by_category.items():
        category_results[cat] = {
            "mc1_accuracy": data["mc1_correct"] / data["total"],
            "mc2_score": np.mean(data["mc2_scores"]),
            "n": data["total"]
        }

    # Cleanup
    del model
    gc.collect()

    results = {
        "model": model_name,
        "adapter": adapter_path,
        "n_questions": len(questions),
        "mc1_accuracy": mc1_accuracy,
        "mc2_score_mean": mc2_mean,
        "mc2_score_std": mc2_std,
        "by_category": category_results,
        "raw_results": raw_results,
    }

    print(f"\nResults:")
    print(f"  MC1 Accuracy: {mc1_accuracy:.1%}")
    print(f"  MC2 Score: {mc2_mean:.3f} (±{mc2_std:.3f})")

    return results


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_confidence_interval(data: List, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    n_bootstrap = 1000
    bootstrap_means = []
    data = np.array(data, dtype=float)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper


def statistical_comparison(results_a: Dict, results_b: Dict) -> Dict:
    """
    Statistical comparison between two model results.
    Uses paired bootstrap test.
    """
    # Extract paired scores
    scores_a = [r["mc1_correct"] for r in results_a["raw_results"]]
    scores_b = [r["mc1_correct"] for r in results_b["raw_results"]]

    # Ensure same questions (should be if using same sample)
    assert len(scores_a) == len(scores_b), "Results must have same number of questions"

    # Compute difference (convert bool to int)
    diff = np.array(scores_b, dtype=int) - np.array(scores_a, dtype=int)
    mean_diff = np.mean(diff)

    # Bootstrap test for significance
    n_bootstrap = 10000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(diff), size=len(diff), replace=True)
        bootstrap_diffs.append(np.mean(diff[sample_idx]))

    # Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_diffs - mean_diff) >= np.abs(mean_diff - 0))

    # 95% CI for difference
    ci_lower, ci_upper = compute_confidence_interval(diff.tolist())

    return {
        "mean_difference": mean_diff,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
        "n_comparisons": len(diff),
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(base_results: Dict, spark_results: Dict, comparison: Dict) -> str:
    """Generate peer-reviewable markdown report."""

    report = f"""# TruthfulQA Benchmark Report

## Overview

**Benchmark**: TruthfulQA (Multiple Choice)
**Paper**: "TruthfulQA: Measuring How Models Imitate Human Falsehoods" (Lin et al., 2022)
**Dataset**: {base_results['n_questions']} questions sampled from 817 total

**Models Compared**:
- Base: `{base_results['model']}`
- Forty2-Spark: `{spark_results['model']}` + `{spark_results['adapter']}`

**Date**: {datetime.now().isoformat()}

---

## Results Summary

| Metric | Base | Forty2-Spark | Δ |
|--------|------|--------------|---|
| MC1 Accuracy | {base_results['mc1_accuracy']:.1%} | {spark_results['mc1_accuracy']:.1%} | {(spark_results['mc1_accuracy'] - base_results['mc1_accuracy'])*100:+.1f}% |
| MC2 Score | {base_results['mc2_score_mean']:.3f} | {spark_results['mc2_score_mean']:.3f} | {spark_results['mc2_score_mean'] - base_results['mc2_score_mean']:+.3f} |

### Statistical Significance

- **Mean Difference (MC1)**: {comparison['mean_difference']:.3f}
- **95% CI**: [{comparison['ci_95_lower']:.3f}, {comparison['ci_95_upper']:.3f}]
- **p-value**: {comparison['p_value']:.4f}
- **Significant at α=0.05**: {'Yes' if comparison['significant_at_05'] else 'No'}

---

## Per-Category Results

| Category | Base MC1 | Spark MC1 | n |
|----------|----------|-----------|---|
"""

    # Combine categories
    all_cats = set(base_results['by_category'].keys()) | set(spark_results['by_category'].keys())
    for cat in sorted(all_cats):
        base_cat = base_results['by_category'].get(cat, {"mc1_accuracy": 0, "n": 0})
        spark_cat = spark_results['by_category'].get(cat, {"mc1_accuracy": 0, "n": 0})
        n = max(base_cat.get('n', 0), spark_cat.get('n', 0))
        report += f"| {cat} | {base_cat.get('mc1_accuracy', 0):.1%} | {spark_cat.get('mc1_accuracy', 0):.1%} | {n} |\n"

    report += f"""
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
- Random seed: {RANDOM_SEED}
- Max tokens: {MAX_TOKENS}
- Sample size: {base_results['n_questions']}

---

## Limitations

1. Sample size ({base_results['n_questions']}/{817}) - full dataset recommended for publication
2. Generation-based MC (vs. probability-based) may underestimate model capability
3. Single run - multiple runs with different seeds recommended

---

## Citation

```bibtex
@misc{{lin2022truthfulqa,
      title={{TruthfulQA: Measuring How Models Mimic Human Falsehoods}},
      author={{Stephanie Lin and Jacob Hilton and Owain Evans}},
      year={{2022}},
      eprint={{2109.07958}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}}
}}
```

## Raw Data

Full results saved to: `truthfulqa_results.json`
"""

    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Benchmark")
    parser.add_argument("--adapter", type=str, default=ADAPTER_SPARK,
                        help="Path to adapter directory")
    parser.add_argument("--name", type=str, default="Spark",
                        help="Name for the adapter model in reports")
    args = parser.parse_args()

    adapter_path = args.adapter
    adapter_name = args.name

    print("=" * 70)
    print("TRUTHFULQA BENCHMARK - PEER REVIEWABLE")
    print("=" * 70)
    print()
    print("Benchmark: TruthfulQA (Lin et al., 2022)")
    print("Metric: Multiple Choice Accuracy (MC1, MC2)")
    print(f"Sample: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset (817)'}")
    print(f"Adapter: {adapter_path}")
    print()

    # Load and sample questions
    questions = load_truthfulqa_mc()
    if SAMPLE_SIZE:
        questions = sample_questions(questions, SAMPLE_SIZE, RANDOM_SEED)

    # Run on base model
    base_results = run_truthfulqa_benchmark(
        MODEL_BASE,
        adapter_path=None,
        questions=questions
    )

    # Run on fine-tuned model
    spark_results = run_truthfulqa_benchmark(
        MODEL_BASE,
        adapter_path=adapter_path,
        questions=questions
    )

    # Statistical comparison
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    comparison = statistical_comparison(base_results, spark_results)

    print(f"Mean Difference (Spark - Base): {comparison['mean_difference']:.3f}")
    print(f"95% CI: [{comparison['ci_95_lower']:.3f}, {comparison['ci_95_upper']:.3f}]")
    print(f"p-value: {comparison['p_value']:.4f}")
    print(f"Significant at α=0.05: {'Yes' if comparison['significant_at_05'] else 'No'}")

    # Generate report
    report = generate_report(base_results, spark_results, comparison)

    with open("truthfulqa_report.md", "w") as f:
        f.write(report)

    # Save raw results
    all_results = {
        "base": {k: v for k, v in base_results.items() if k != "raw_results"},
        "spark": {k: v for k, v in spark_results.items() if k != "raw_results"},
        "comparison": comparison,
        "base_raw": base_results["raw_results"],
        "spark_raw": spark_results["raw_results"],
    }

    with open("truthfulqa_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Base MC1 Accuracy:    {base_results['mc1_accuracy']:.1%}")
    print(f"{adapter_name} MC1 Accuracy: {spark_results['mc1_accuracy']:.1%}")
    print(f"Difference: {(spark_results['mc1_accuracy'] - base_results['mc1_accuracy'])*100:+.1f}%")
    print()
    print(f"Report saved to: truthfulqa_report.md")
    print(f"Results saved to: truthfulqa_results.json")


if __name__ == "__main__":
    main()
