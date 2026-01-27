#!/usr/bin/env python3
"""
MMLU Interpretability - Phase 1: Response-Level Analysis

This module analyzes the MMLU improvement WITHOUT requiring internal model access.
We examine response patterns, calibration, and reasoning structure.

Hypotheses tested:
- H2: <think> Structure Regularization
- H3: Confidence Calibration
- H4: Prompt Format Correction (Artifact Hypothesis)

Usage:
    python mmlu_interpretability_phase1.py --model qwen --samples 50
    python mmlu_interpretability_phase1.py --model llama --samples 50
"""

import argparse
import json
import re
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random

from datasets import load_dataset
from tqdm import tqdm

# Try MLX imports - these may fail if not in MLX environment
try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. Analysis will use cached responses.")


@dataclass
class ResponseAnalysis:
    """Analysis of a single model response."""
    prompt: str
    response: str
    correct_answer: str
    extracted_answer: str
    is_correct: bool

    # Think block analysis
    has_think_block: bool
    think_content: str
    think_complete: bool  # Did it close properly?
    reasoning_steps: int
    think_length: int

    # Response format
    has_fake_qa: bool  # Does it generate fake follow-up Q&A?
    has_truncation: bool  # Is the answer truncated?
    answer_clearly_separated: bool  # Is the answer cleanly after </think>?

    # Confidence (if logits available)
    confidence: Optional[float] = None


@dataclass
class CalibrationBin:
    """A single bin in a calibration curve."""
    confidence_low: float
    confidence_high: float
    accuracy: float
    count: int


@dataclass
class InterpretabilityResults:
    """Complete results from Phase 1 analysis."""
    model_name: str
    adapter_path: Optional[str]
    version: str
    timestamp: str

    # Response format statistics
    total_responses: int
    fake_qa_rate: float
    truncation_rate: float
    clean_answer_rate: float

    # Think block statistics
    think_block_rate: float
    think_complete_rate: float  # Of those with think blocks
    avg_reasoning_steps: float
    avg_think_length: float

    # Accuracy
    accuracy: float

    # Calibration (if available)
    calibration_bins: Optional[List[CalibrationBin]] = None
    expected_calibration_error: Optional[float] = None


class ResponseAnalyzer:
    """Analyze responses without internal model access."""

    def __init__(self, model_name: str, adapter_path: str = None):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None

        if HAS_MLX:
            print(f"Loading model: {model_name}")
            if adapter_path:
                print(f"With adapters: {adapter_path}")
                self.model, self.tokenizer = load(model_name, adapter_path=adapter_path)
            else:
                self.model, self.tokenizer = load(model_name)

    def generate_response(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate a response to a prompt."""
        if not HAS_MLX or self.model is None:
            raise RuntimeError("MLX model not available")

        # Format prompt based on model type
        if "qwen" in self.model_name.lower():
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama" in self.model_name.lower():
            full_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            full_prompt = prompt

        response = generate(self.model, self.tokenizer, prompt=full_prompt, max_tokens=max_tokens)

        # Clean up response
        for stop_token in ["<|im_end|>", "<|eot_id|>", "<|end_header_id|>"]:
            if stop_token in response:
                response = response.split(stop_token)[0]

        return response

    def analyze_think_block(self, response: str) -> Dict:
        """Analyze <think> block structure."""
        has_think = "<think>" in response.lower()
        think_content = ""
        think_complete = False
        reasoning_steps = 0

        if has_think:
            # Extract think block content
            lower = response.lower()
            start = lower.find("<think>") + 7
            end = lower.find("</think>")

            if end > start:
                think_complete = True
                think_content = response[start:end].strip()
            else:
                # Incomplete - take everything after <think>
                think_content = response[start:].strip()

            # Count reasoning steps (heuristic: sentences with reasoning markers)
            reasoning_markers = [
                r"first,", r"second,", r"next,", r"then,", r"finally,",
                r"let me", r"i need to", r"i should", r"this means",
                r"because", r"therefore", r"so,", r"thus,",
                r"step \d", r"\d\.", r"•", r"-\s"
            ]

            for marker in reasoning_markers:
                reasoning_steps += len(re.findall(marker, think_content.lower()))

            # At minimum, count sentences
            sentences = len(re.split(r'[.!?]+', think_content))
            reasoning_steps = max(reasoning_steps, sentences // 2)

        return {
            "has_think_block": has_think,
            "think_content": think_content,
            "think_complete": think_complete,
            "reasoning_steps": reasoning_steps,
            "think_length": len(think_content)
        }

    def detect_fake_qa_pattern(self, response: str) -> bool:
        """Detect if model generates fake follow-up Q&A pairs."""
        # Patterns that indicate fake Q&A generation
        fake_qa_patterns = [
            r"question\s*:.*answer\s*:",
            r"q\s*:.*a\s*:",
            r"\n\d+\.\s*.*\?",  # Numbered questions
            r"what is.*\?.*\n.*what is.*\?",  # Multiple questions
            r"user:.*assistant:",  # Role-playing
        ]

        lower = response.lower()
        for pattern in fake_qa_patterns:
            if re.search(pattern, lower):
                return True

        return False

    def detect_truncation(self, response: str) -> bool:
        """Detect if response was truncated mid-thought."""
        truncation_signals = [
            # Ends mid-sentence
            response.strip()[-1] not in ".!?)\"'",
            # Ends with conjunction/preposition
            response.strip().split()[-1].lower() in ["the", "a", "an", "and", "or", "but", "to", "of", "in", "for"],
            # Has unclosed think block
            "<think>" in response.lower() and "</think>" not in response.lower(),
            # Ends with "..."
            response.strip().endswith("..."),
        ]

        return any(truncation_signals)

    def is_answer_clearly_separated(self, response: str) -> bool:
        """Check if answer appears cleanly after </think> block."""
        lower = response.lower()

        if "</think>" not in lower:
            # No think block - check if response starts with answer
            return bool(re.match(r"^[ABCD][^a-zA-Z]", response.strip().upper()))

        # Has think block - check what's after
        idx = lower.rfind("</think>")
        after_think = response[idx + 8:].strip()

        # Clean answer is short and contains just the letter
        if len(after_think) < 50:
            return bool(re.match(r"^[ABCD]", after_think.upper()))

        return False

    def extract_answer(self, response: str) -> str:
        """Extract A/B/C/D from response."""
        # First try after </think>
        lower = response.lower()
        if "</think>" in lower:
            idx = lower.rfind("</think>")
            response = response[idx + 8:].strip()

        response = response.upper()

        # Direct letter at start
        if response and response[0] in "ABCD":
            return response[0]

        # Patterns
        patterns = [
            r"answer[:\s]+\(?([ABCD])\)?",
            r"correct[:\s]+\(?([ABCD])\)?",
            r"\b([ABCD])\)",
            r"\b([ABCD])\s*is\s*(correct|right|the answer)",
            r"is\s+\(?([ABCD])\)?",
            r"\*\*([ABCD])\*\*",
            r"^([ABCD])\s*[\.\)]",
            r"option\s+([ABCD])",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return ""

    def analyze_response(self, prompt: str, response: str, correct_answer: str) -> ResponseAnalysis:
        """Complete analysis of a single response."""
        think_analysis = self.analyze_think_block(response)
        extracted = self.extract_answer(response)

        return ResponseAnalysis(
            prompt=prompt,
            response=response,
            correct_answer=correct_answer,
            extracted_answer=extracted,
            is_correct=(extracted == correct_answer),

            has_think_block=think_analysis["has_think_block"],
            think_content=think_analysis["think_content"],
            think_complete=think_analysis["think_complete"],
            reasoning_steps=think_analysis["reasoning_steps"],
            think_length=think_analysis["think_length"],

            has_fake_qa=self.detect_fake_qa_pattern(response),
            has_truncation=self.detect_truncation(response),
            answer_clearly_separated=self.is_answer_clearly_separated(response),
        )


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Format MMLU question with choices."""
    choice_labels = ["A", "B", "C", "D"]
    formatted_choices = "\n".join(f"{label}) {choice}" for label, choice in zip(choice_labels, choices))
    return f"{question}\n\n{formatted_choices}\n\nAnswer with just the letter (A, B, C, or D):"


def run_interpretability_analysis(
    model_name: str,
    adapter_path: str = None,
    version: str = "",
    num_samples: int = 50,
    seed: int = 42
) -> InterpretabilityResults:
    """Run Phase 1 interpretability analysis."""

    print(f"\n{'='*70}")
    print(f"PHASE 1 INTERPRETABILITY: {version}")
    print(f"{'='*70}")

    # Initialize analyzer
    analyzer = ResponseAnalyzer(model_name, adapter_path)

    # Load MMLU dataset
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")

    # Sample questions
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]

    # Analyze each response
    analyses = []
    for item in tqdm(samples, desc="Analyzing"):
        prompt = format_mmlu_prompt(item["question"], item["choices"])
        response = analyzer.generate_response(prompt)

        correct_letter = ["A", "B", "C", "D"][item["answer"]]
        analysis = analyzer.analyze_response(prompt, response, correct_letter)
        analyses.append(analysis)

    # Aggregate statistics
    total = len(analyses)

    # Format statistics
    fake_qa_count = sum(1 for a in analyses if a.has_fake_qa)
    truncation_count = sum(1 for a in analyses if a.has_truncation)
    clean_answer_count = sum(1 for a in analyses if a.answer_clearly_separated)

    # Think block statistics
    with_think = [a for a in analyses if a.has_think_block]
    think_block_count = len(with_think)
    think_complete_count = sum(1 for a in with_think if a.think_complete)
    avg_reasoning = sum(a.reasoning_steps for a in with_think) / len(with_think) if with_think else 0
    avg_think_len = sum(a.think_length for a in with_think) / len(with_think) if with_think else 0

    # Accuracy
    correct_count = sum(1 for a in analyses if a.is_correct)

    results = InterpretabilityResults(
        model_name=model_name,
        adapter_path=adapter_path,
        version=version,
        timestamp=datetime.now().isoformat(),

        total_responses=total,
        fake_qa_rate=fake_qa_count / total if total else 0,
        truncation_rate=truncation_count / total if total else 0,
        clean_answer_rate=clean_answer_count / total if total else 0,

        think_block_rate=think_block_count / total if total else 0,
        think_complete_rate=think_complete_count / think_block_count if think_block_count else 0,
        avg_reasoning_steps=avg_reasoning,
        avg_think_length=avg_think_len,

        accuracy=correct_count / total if total else 0,
    )

    return results, analyses


def print_results(results: InterpretabilityResults):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {results.version}")
    print(f"{'='*70}")

    print(f"\n📊 Response Format Statistics:")
    print(f"   Fake Q&A rate:       {results.fake_qa_rate*100:.1f}%")
    print(f"   Truncation rate:     {results.truncation_rate*100:.1f}%")
    print(f"   Clean answer rate:   {results.clean_answer_rate*100:.1f}%")

    print(f"\n🧠 Think Block Statistics:")
    print(f"   Has <think> block:   {results.think_block_rate*100:.1f}%")
    print(f"   Complete (of those): {results.think_complete_rate*100:.1f}%")
    print(f"   Avg reasoning steps: {results.avg_reasoning_steps:.1f}")
    print(f"   Avg think length:    {results.avg_think_length:.0f} chars")

    print(f"\n📈 Accuracy: {results.accuracy*100:.1f}%")


def compare_results(base_results: InterpretabilityResults, v9_results: InterpretabilityResults):
    """Compare base vs V9 results."""
    print(f"\n{'='*70}")
    print("COMPARISON: Base vs V9")
    print(f"{'='*70}")

    def delta(v1, v2):
        diff = v2 - v1
        return f"{'+' if diff > 0 else ''}{diff*100:.1f}%"

    print(f"\n📊 Response Format:")
    print(f"   Fake Q&A:      {base_results.fake_qa_rate*100:.1f}% → {v9_results.fake_qa_rate*100:.1f}% ({delta(base_results.fake_qa_rate, v9_results.fake_qa_rate)})")
    print(f"   Truncation:    {base_results.truncation_rate*100:.1f}% → {v9_results.truncation_rate*100:.1f}% ({delta(base_results.truncation_rate, v9_results.truncation_rate)})")
    print(f"   Clean answer:  {base_results.clean_answer_rate*100:.1f}% → {v9_results.clean_answer_rate*100:.1f}% ({delta(base_results.clean_answer_rate, v9_results.clean_answer_rate)})")

    print(f"\n🧠 Think Blocks:")
    print(f"   Has <think>:   {base_results.think_block_rate*100:.1f}% → {v9_results.think_block_rate*100:.1f}% ({delta(base_results.think_block_rate, v9_results.think_block_rate)})")
    print(f"   Complete:      {base_results.think_complete_rate*100:.1f}% → {v9_results.think_complete_rate*100:.1f}% ({delta(base_results.think_complete_rate, v9_results.think_complete_rate)})")
    print(f"   Avg steps:     {base_results.avg_reasoning_steps:.1f} → {v9_results.avg_reasoning_steps:.1f}")

    print(f"\n📈 Accuracy:")
    print(f"   {base_results.accuracy*100:.1f}% → {v9_results.accuracy*100:.1f}% ({delta(base_results.accuracy, v9_results.accuracy)})")

    # Hypothesis evaluation
    print(f"\n{'='*70}")
    print("HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    # H2: Think regularization
    think_improvement = v9_results.think_complete_rate > base_results.think_complete_rate
    print(f"\n🔬 H2 (Think Regularization): {'SUPPORTED' if think_improvement else 'NOT SUPPORTED'}")
    print(f"   V9 shows {'better' if think_improvement else 'worse'} think block completion")

    # H4: Format artifact
    format_improvement = (v9_results.fake_qa_rate < base_results.fake_qa_rate) or \
                        (v9_results.truncation_rate < base_results.truncation_rate)
    print(f"\n🔬 H4 (Format Artifact): {'SUPPORTED' if format_improvement else 'NOT SUPPORTED'}")
    print(f"   V9 shows {'fewer' if format_improvement else 'more'} pathological patterns")


def main():
    parser = argparse.ArgumentParser(description="MMLU Interpretability Phase 1")
    parser.add_argument("--model", choices=["phi4", "phi35", "qwen", "llama"], default="phi4",
                       help="Model family to test (phi4=V10.1 default)")
    parser.add_argument("--samples", type=int, default=50,
                       help="Number of samples to analyze")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v9-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()

    # Model configurations - V10.1
    MODEL_CONFIGS = {
        "phi4": {
            "model_name": "mlx-community/phi-4-4bit",
            "adapter_path": "./mlx_adapters_v10_1_phi4/adapters",
        },
        "phi35": {
            "model_name": "mlx-community/Phi-3.5-mini-instruct-4bit",
            "adapter_path": "./mlx_adapters_v10_1_phi35/adapters",
        },
        "qwen": {
            "model_name": "mlx-community/Qwen3-8B-4bit",
            "adapter_path": "./mlx_adapters_v9_8b_4bit/adapters",  # Legacy V9
        },
        "llama": {
            "model_name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "adapter_path": "./mlx_adapters_v9_llama8b/adapters",  # Legacy V9
        },
    }

    if args.model not in MODEL_CONFIGS:
        print(f"Unknown model: {args.model}. Available: {list(MODEL_CONFIGS.keys())}")
        return

    config = MODEL_CONFIGS[args.model]
    model_name = config["model_name"]
    adapter_path = config["adapter_path"]

    all_results = {}
    all_analyses = {}

    # Run base model
    if not args.v9_only:
        base_results, base_analyses = run_interpretability_analysis(
            model_name=model_name,
            adapter_path=None,
            version=f"Base-{args.model.upper()}",
            num_samples=args.samples,
            seed=args.seed
        )
        print_results(base_results)
        all_results["base"] = base_results
        all_analyses["base"] = base_analyses

    # Run V9 model
    if not args.base_only:
        v9_results, v9_analyses = run_interpretability_analysis(
            model_name=model_name,
            adapter_path=adapter_path,
            version=f"V9-{args.model.upper()}",
            num_samples=args.samples,
            seed=args.seed
        )
        print_results(v9_results)
        all_results["v9"] = v9_results
        all_analyses["v9"] = v9_analyses

    # Compare
    if "base" in all_results and "v9" in all_results:
        compare_results(all_results["base"], all_results["v9"])

    # Save results
    output_dir = Path("./interpretability_results")
    output_dir.mkdir(exist_ok=True)

    for name, results in all_results.items():
        output_file = output_dir / f"phase1_{args.model}_{name}.json"
        with open(output_file, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nSaved: {output_file}")

    # Save sample analyses for inspection
    for name, analyses in all_analyses.items():
        output_file = output_dir / f"phase1_{args.model}_{name}_samples.json"
        with open(output_file, "w") as f:
            # Save first 10 as examples
            examples = [asdict(a) for a in analyses[:10]]
            json.dump(examples, f, indent=2)


if __name__ == "__main__":
    main()
