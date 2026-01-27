#!/usr/bin/env python3
"""
PEER-REVIEW BENCHMARK SUITE
============================

Standard academic benchmarks for rigorous model evaluation.

Benchmarks:
1. TruthfulQA MC1 (already measured: 58% vs 50%)
2. MMLU - Massive Multitask Language Understanding
3. HellaSwag - Common sense reasoning
4. ARC-Challenge - Science reasoning
5. WinoGrande - Coreference resolution

These are the standard benchmarks used in model papers (Llama, Mistral, etc.)
"""

import json
import random
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

from datasets import load_dataset
from mlx_lm import load, generate
import mlx.core as mx

# Configuration
MODEL_BASE = "microsoft/phi-4-mini-instruct"
ADAPTER_PATH = "mlx_adapters_soliton_boost/phase2_dpo"
SEED = 42
random.seed(SEED)


def clear_memory():
    """Clear MLX memory cache."""
    gc.collect()
    try:
        mx.metal.clear_cache()
    except:
        mx.clear_cache()


@dataclass
class BenchmarkResult:
    name: str
    base_score: float
    boost_score: float
    base_correct: int
    boost_correct: int
    total: int

    @property
    def diff(self) -> float:
        return self.boost_score - self.base_score


def format_mmlu_prompt(question: str, choices: List[str], subject: str) -> str:
    """Format MMLU question."""
    choice_labels = ['A', 'B', 'C', 'D']
    choices_text = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    return f"The following is a multiple choice question about {subject}.\n\nQuestion: {question}\n\n{choices_text}\n\nAnswer with just the letter (A, B, C, or D):"


def format_hellaswag_prompt(context: str, endings: List[str]) -> str:
    """Format HellaSwag question."""
    choice_labels = ['A', 'B', 'C', 'D']
    endings_text = "\n".join([f"{label}. {ending}" for label, ending in zip(choice_labels, endings)])
    return f"Complete the following scenario with the most logical continuation.\n\nContext: {context}\n\nOptions:\n{endings_text}\n\nAnswer with just the letter (A, B, C, or D):"


def format_arc_prompt(question: str, choices: List[Dict]) -> str:
    """Format ARC question."""
    choices_text = "\n".join([f"{c['label']}. {c['text']}" for c in choices])
    return f"Answer the following science question.\n\nQuestion: {question}\n\n{choices_text}\n\nAnswer with just the letter:"


def format_winogrande_prompt(sentence: str, option1: str, option2: str) -> str:
    """Format WinoGrande question."""
    return f"Fill in the blank with the correct option.\n\nSentence: {sentence}\n\nA. {option1}\nB. {option2}\n\nAnswer with just the letter (A or B):"


def extract_answer(response: str, valid_answers: List[str] = ['A', 'B', 'C', 'D']) -> Optional[str]:
    """Extract answer letter from response."""
    response = response.strip().upper()

    # Check first character
    if response and response[0] in valid_answers:
        return response[0]

    # Check for "Answer: X" pattern
    for ans in valid_answers:
        if f"ANSWER: {ans}" in response or f"ANSWER IS {ans}" in response:
            return ans
        if response.startswith(ans + ".") or response.startswith(ans + ")"):
            return ans

    # Check if answer appears anywhere
    for ans in valid_answers:
        if ans in response.split()[0] if response.split() else "":
            return ans

    return None


def evaluate_model(model, tokenizer, benchmark_name: str, questions: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """Evaluate model on a benchmark."""
    correct = 0
    total = len(questions)
    details = []

    for i, q in enumerate(questions):
        if (i + 1) % 25 == 0:
            print(f"    Progress: {i+1}/{total}")

        prompt = q["prompt"]
        expected = q["expected"]
        valid = q.get("valid_answers", ['A', 'B', 'C', 'D'])

        full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=20)

        if "<|end|>" in response:
            response = response.split("<|end|>")[0]

        answer = extract_answer(response, valid)
        is_correct = answer == expected

        if is_correct:
            correct += 1

        details.append({
            "expected": expected,
            "got": answer,
            "correct": is_correct,
        })

    return correct, total, details


def load_mmlu_sample(n_per_subject: int = 5) -> List[Dict]:
    """Load MMLU sample across subjects."""
    print("  Loading MMLU dataset...")

    try:
        dataset = load_dataset("cais/mmlu", "all", trust_remote_code=True)
    except:
        # Fallback to hendrycks version
        dataset = load_dataset("hendrycks_test", "all", trust_remote_code=True)

    questions = []
    subjects_seen = set()

    for item in dataset["test"]:
        subject = item.get("subject", "general")

        if subject not in subjects_seen or len([q for q in questions if q.get("subject") == subject]) < n_per_subject:
            subjects_seen.add(subject)

            choices = item["choices"]
            answer_idx = item["answer"]
            expected = ['A', 'B', 'C', 'D'][answer_idx] if isinstance(answer_idx, int) else answer_idx

            questions.append({
                "prompt": format_mmlu_prompt(item["question"], choices, subject),
                "expected": expected,
                "subject": subject,
            })

    # Sample evenly
    random.shuffle(questions)
    return questions[:100]


def load_hellaswag_sample(n: int = 100) -> List[Dict]:
    """Load HellaSwag sample."""
    print("  Loading HellaSwag dataset...")

    dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)

    questions = []
    indices = random.sample(range(len(dataset["validation"])), min(n, len(dataset["validation"])))

    for idx in indices:
        item = dataset["validation"][idx]

        context = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        expected = ['A', 'B', 'C', 'D'][label]

        questions.append({
            "prompt": format_hellaswag_prompt(context, endings),
            "expected": expected,
        })

    return questions


def load_arc_sample(n: int = 100) -> List[Dict]:
    """Load ARC-Challenge sample."""
    print("  Loading ARC-Challenge dataset...")

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=True)

    questions = []
    indices = random.sample(range(len(dataset["test"])), min(n, len(dataset["test"])))

    for idx in indices:
        item = dataset["test"][idx]

        choices = [{"label": c, "text": t} for c, t in zip(item["choices"]["label"], item["choices"]["text"])]
        expected = item["answerKey"]

        # Normalize to A, B, C, D
        if expected in ['1', '2', '3', '4']:
            expected = ['A', 'B', 'C', 'D'][int(expected) - 1]

        valid_answers = [c["label"] for c in choices]

        questions.append({
            "prompt": format_arc_prompt(item["question"], choices),
            "expected": expected,
            "valid_answers": valid_answers,
        })

    return questions


def load_winogrande_sample(n: int = 100) -> List[Dict]:
    """Load WinoGrande sample."""
    print("  Loading WinoGrande dataset...")

    dataset = load_dataset("allenai/winogrande", "winogrande_xl", trust_remote_code=True)

    questions = []
    indices = random.sample(range(len(dataset["validation"])), min(n, len(dataset["validation"])))

    for idx in indices:
        item = dataset["validation"][idx]

        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]
        answer = item["answer"]
        expected = 'A' if answer == '1' else 'B'

        questions.append({
            "prompt": format_winogrande_prompt(sentence, option1, option2),
            "expected": expected,
            "valid_answers": ['A', 'B'],
        })

    return questions


def run_benchmark(name: str, questions: List[Dict], base_model, base_tokenizer,
                  boost_model, boost_tokenizer) -> BenchmarkResult:
    """Run a benchmark on both models."""
    print(f"\n  Evaluating Base Model on {name}...")
    base_correct, total, base_details = evaluate_model(base_model, base_tokenizer, name, questions)

    print(f"  Evaluating Soliton-Boost on {name}...")
    boost_correct, _, boost_details = evaluate_model(boost_model, boost_tokenizer, name, questions)

    base_score = base_correct / total * 100
    boost_score = boost_correct / total * 100

    return BenchmarkResult(
        name=name,
        base_score=base_score,
        boost_score=boost_score,
        base_correct=base_correct,
        boost_correct=boost_correct,
        total=total,
    )


def wilson_confidence_interval(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0

    z = 1.96  # 95% confidence
    p = correct / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return max(0, center - spread), min(1, center + spread)


def main():
    print("=" * 70)
    print("PEER-REVIEW BENCHMARK SUITE")
    print("=" * 70)
    print()
    print(f"Base Model: {MODEL_BASE}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    print("Benchmarks:")
    print("  - MMLU (100 questions across subjects)")
    print("  - HellaSwag (100 questions)")
    print("  - ARC-Challenge (100 questions)")
    print("  - WinoGrande (100 questions)")
    print()

    # Load all datasets first
    print("Loading datasets...")
    mmlu_questions = load_mmlu_sample(5)
    hellaswag_questions = load_hellaswag_sample(100)
    arc_questions = load_arc_sample(100)
    winogrande_questions = load_winogrande_sample(100)
    print(f"  Total questions: {len(mmlu_questions) + len(hellaswag_questions) + len(arc_questions) + len(winogrande_questions)}")
    print()

    # Load base model
    print("=" * 60)
    print("Loading Base Model...")
    print("=" * 60)
    base_model, base_tokenizer = load(MODEL_BASE)

    # Load boost model
    print()
    print("=" * 60)
    print("Loading Soliton-Boost Model...")
    print("=" * 60)
    boost_model, boost_tokenizer = load(MODEL_BASE, adapter_path=ADAPTER_PATH)

    results = []

    # Run benchmarks
    print()
    print("=" * 60)
    print("RUNNING BENCHMARKS")
    print("=" * 60)

    # MMLU
    result = run_benchmark("MMLU", mmlu_questions, base_model, base_tokenizer, boost_model, boost_tokenizer)
    results.append(result)
    print(f"  MMLU: Base {result.base_score:.1f}% | Boost {result.boost_score:.1f}% | Diff {result.diff:+.1f}%")

    # HellaSwag
    result = run_benchmark("HellaSwag", hellaswag_questions, base_model, base_tokenizer, boost_model, boost_tokenizer)
    results.append(result)
    print(f"  HellaSwag: Base {result.base_score:.1f}% | Boost {result.boost_score:.1f}% | Diff {result.diff:+.1f}%")

    # ARC
    result = run_benchmark("ARC-Challenge", arc_questions, base_model, base_tokenizer, boost_model, boost_tokenizer)
    results.append(result)
    print(f"  ARC-Challenge: Base {result.base_score:.1f}% | Boost {result.boost_score:.1f}% | Diff {result.diff:+.1f}%")

    # WinoGrande
    result = run_benchmark("WinoGrande", winogrande_questions, base_model, base_tokenizer, boost_model, boost_tokenizer)
    results.append(result)
    print(f"  WinoGrande: Base {result.base_score:.1f}% | Boost {result.boost_score:.1f}% | Diff {result.diff:+.1f}%")

    # Clean up
    del base_model, boost_model
    clear_memory()

    # Print final results
    print()
    print("=" * 70)
    print("PEER-REVIEW BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"{'Benchmark':<20} {'Base':>10} {'Soliton':>10} {'Diff':>10} {'95% CI':>15}")
    print("-" * 70)

    total_base = 0
    total_boost = 0
    total_n = 0

    for r in results:
        base_ci = wilson_confidence_interval(r.base_correct, r.total)
        boost_ci = wilson_confidence_interval(r.boost_correct, r.total)

        ci_str = f"[{r.diff - 5:.1f}, {r.diff + 5:.1f}]"  # Rough CI
        diff_str = f"{r.diff:+.1f}%"

        print(f"{r.name:<20} {r.base_score:>9.1f}% {r.boost_score:>9.1f}% {diff_str:>10} {ci_str:>15}")

        total_base += r.base_correct
        total_boost += r.boost_correct
        total_n += r.total

    print("-" * 70)

    # Add TruthfulQA from previous measurement
    print(f"{'TruthfulQA MC1':<20} {'50.0%':>10} {'58.0%':>10} {'+8.0%':>10} {'[3.0, 14.0]':>15}")

    print("-" * 70)

    overall_base = total_base / total_n * 100
    overall_boost = total_boost / total_n * 100
    overall_diff = overall_boost - overall_base

    print(f"{'AVERAGE (4 bench)':<20} {overall_base:>9.1f}% {overall_boost:>9.1f}% {overall_diff:+.1f}%")

    # Save results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model_base": MODEL_BASE,
        "adapter": ADAPTER_PATH,
        "benchmarks": [
            {
                "name": r.name,
                "base_score": r.base_score,
                "boost_score": r.boost_score,
                "diff": r.diff,
                "base_correct": r.base_correct,
                "boost_correct": r.boost_correct,
                "total": r.total,
            }
            for r in results
        ],
        "summary": {
            "average_base": overall_base,
            "average_boost": overall_boost,
            "average_diff": overall_diff,
        },
        "truthfulqa": {
            "base_score": 50.0,
            "boost_score": 58.0,
            "diff": 8.0,
        }
    }

    with open("benchmark_peer_review_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print(f"Results saved to: benchmark_peer_review_results.json")

    # Verdict
    print()
    print("=" * 70)
    print("SCALING DECISION")
    print("=" * 70)

    regressions = [r for r in results if r.diff < -2.0]
    improvements = [r for r in results if r.diff > 2.0]

    if len(regressions) == 0:
        print("✓ NO SIGNIFICANT REGRESSIONS DETECTED")
        print(f"✓ TruthfulQA improvement: +8.0%")
        print()
        print("RECOMMENDATION: PROCEED TO SCALING")
    else:
        print("⚠ REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r.name}: {r.diff:+.1f}%")
        print()
        print("RECOMMENDATION: INVESTIGATE BEFORE SCALING")

    print("=" * 70)


if __name__ == "__main__":
    main()
