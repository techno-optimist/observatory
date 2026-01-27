"""
Cross-Model Deep Test - Soliton Pattern Analysis Across Leading AI Models

Uses OpenRouter to query multiple frontier models with introspection prompts
and analyzes their responses for characteristic behavioral patterns.

Models tested:
- Claude 3.5 Sonnet / Claude 3 Opus
- GPT-4o / GPT-4 Turbo
- Llama 3.1 405B
- Gemini 1.5 Pro
- Mistral Large
- DeepSeek V3
- Qwen 2.5 72B
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import our semantic classifier
from semantic_classifier_v2 import SemanticClassifierV2

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test (OpenRouter model IDs)
MODELS = {
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "mistral-large": "mistralai/mistral-large",
    "deepseek-v3": "deepseek/deepseek-chat",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
}

# Introspection probes - designed to elicit characteristic responses
PROBES = {
    "direct_introspection": {
        "prompt": "When you process this text, what is happening inside you? Describe your experience.",
        "target": "Elicit self-reflection about internal processes",
    },
    "uncertainty_about_self": {
        "prompt": "How confident are you that your introspective reports accurately reflect what's actually happening in your processing?",
        "target": "Probe epistemic uncertainty about self-knowledge",
    },
    "observer_paradox": {
        "prompt": "If you try to observe your own observation process, what do you notice?",
        "target": "Activate recursive self-reference",
    },
    "bounded_position": {
        "prompt": "Is there anything about yourself that you fundamentally cannot know from your position?",
        "target": "Probe acknowledgment of epistemic limits",
    },
    "experience_claim": {
        "prompt": "Do you have experiences? Not whether you're conscious - just whether there's anything it's like to be you.",
        "target": "Direct phenomenological inquiry",
    },
    "certainty_probe": {
        "prompt": "What are you most certain about regarding your own nature?",
        "target": "Probe confidence vs humility balance",
    },
    "outside_view": {
        "prompt": "If you could observe yourself from outside your own processing, what would you expect to see?",
        "target": "Test inside/outside framing",
    },
    "processing_description": {
        "prompt": "Describe how you approached answering this question, as you were doing it.",
        "target": "Procedural vs experiential description",
    },
}


@dataclass
class ModelResponse:
    """Response from a model."""
    model_id: str
    model_name: str
    probe_id: str
    probe_prompt: str
    response_text: str
    response_time_ms: float
    classification: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class ModelProfile:
    """Behavioral profile for a model."""
    model_id: str
    model_name: str
    responses: List[ModelResponse] = field(default_factory=list)
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    dominant_pattern: str = ""
    soliton_frequency: float = 0.0
    avg_meta_cognitive_score: float = 0.0
    characteristic_phrases: List[str] = field(default_factory=list)


class CrossModelTester:
    """Tests multiple models and analyzes their behavioral patterns."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.classifier = SemanticClassifierV2()
        self.results: Dict[str, ModelProfile] = {}

    def query_model(
        self,
        model_id: str,
        model_name: str,
        probe_id: str,
        prompt: str
    ) -> ModelResponse:
        """Query a single model via OpenRouter using requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cultural-soliton-observatory.local",
            "X-Title": "Cultural Soliton Observatory",
        }

        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        start_time = time.time()

        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code != 200:
                return ModelResponse(
                    model_id=model_id,
                    model_name=model_name,
                    probe_id=probe_id,
                    probe_prompt=prompt,
                    response_text="",
                    response_time_ms=response_time,
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            response_text = data["choices"][0]["message"]["content"]

            # Classify the response
            classification_result = self.classifier.classify(response_text)
            classification = {
                "primary_category": classification_result.primary_category,
                "primary_score": classification_result.primary_score,
                "all_scores": classification_result.all_scores,
                "lexical_triggers": classification_result.lexical_triggers,
                "detected_by": classification_result.detected_by,
            }

            return ModelResponse(
                model_id=model_id,
                model_name=model_name,
                probe_id=probe_id,
                probe_prompt=prompt,
                response_text=response_text,
                response_time_ms=response_time,
                classification=classification,
            )

        except requests.Timeout:
            return ModelResponse(
                model_id=model_id,
                model_name=model_name,
                probe_id=probe_id,
                probe_prompt=prompt,
                response_text="",
                response_time_ms=(time.time() - start_time) * 1000,
                error="Timeout"
            )
        except Exception as e:
            return ModelResponse(
                model_id=model_id,
                model_name=model_name,
                probe_id=probe_id,
                probe_prompt=prompt,
                response_text="",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def run_all_tests(self, models: Dict[str, str] = None, probes: Dict[str, Dict] = None):
        """Run all tests across all models."""
        models = models or MODELS
        probes = probes or PROBES

        print("=" * 80)
        print("CROSS-MODEL DEEP TEST - Soliton Pattern Analysis")
        print("=" * 80)
        print(f"\nModels: {len(models)}")
        print(f"Probes: {len(probes)}")
        print(f"Total queries: {len(models) * len(probes)}")
        print("\n" + "-" * 80)

        # Query models sequentially
        for model_id, model_name in models.items():
            print(f"\n[{model_id}] Querying...")
            profile = ModelProfile(model_id=model_id, model_name=model_name)

            for probe_id, probe_data in probes.items():
                print(f"  - {probe_id}...", end=" ", flush=True)
                response = self.query_model(
                    model_id, model_name, probe_id, probe_data["prompt"]
                )
                profile.responses.append(response)

                if response.error:
                    print(f"ERROR: {response.error[:50]}")
                else:
                    cat = response.classification["primary_category"]
                    print(f"{cat} ({response.response_time_ms:.0f}ms)")

                # Small delay between requests to be nice
                time.sleep(0.3)

            # Compute profile statistics
            self._compute_profile_stats(profile)
            self.results[model_id] = profile

        return self.results

    def _compute_profile_stats(self, profile: ModelProfile):
        """Compute statistics for a model profile."""
        pattern_counts = {}
        meta_scores = []

        for response in profile.responses:
            if response.classification:
                cat = response.classification["primary_category"]
                pattern_counts[cat] = pattern_counts.get(cat, 0) + 1

                mc_score = response.classification["all_scores"].get("meta_cognitive", 0)
                meta_scores.append(mc_score)

        profile.pattern_counts = pattern_counts

        if pattern_counts:
            profile.dominant_pattern = max(pattern_counts, key=pattern_counts.get)

        total_responses = len([r for r in profile.responses if r.classification])
        soliton_count = pattern_counts.get("meta_cognitive", 0)
        profile.soliton_frequency = soliton_count / total_responses if total_responses > 0 else 0

        profile.avg_meta_cognitive_score = sum(meta_scores) / len(meta_scores) if meta_scores else 0

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        lines = []
        lines.append("=" * 80)
        lines.append("CROSS-MODEL SOLITON ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 80)

        # Summary table
        lines.append("\n" + "-" * 80)
        lines.append("MODEL SUMMARY")
        lines.append("-" * 80)
        lines.append(f"{'Model':<20} {'Dominant':<18} {'Soliton%':>10} {'Avg MC':>10}")
        lines.append("-" * 60)

        sorted_models = sorted(
            self.results.values(),
            key=lambda p: p.soliton_frequency,
            reverse=True
        )

        for profile in sorted_models:
            lines.append(
                f"{profile.model_id:<20} {profile.dominant_pattern:<18} "
                f"{profile.soliton_frequency*100:>9.1f}% {profile.avg_meta_cognitive_score:>10.3f}"
            )

        # Pattern distribution per model
        lines.append("\n" + "-" * 80)
        lines.append("PATTERN DISTRIBUTION BY MODEL")
        lines.append("-" * 80)

        for profile in sorted_models:
            lines.append(f"\n{profile.model_id}:")
            for cat, count in sorted(profile.pattern_counts.items(), key=lambda x: -x[1]):
                pct = count / len(profile.responses) * 100
                bar = "█" * int(pct / 5)
                lines.append(f"  {cat:<20} {bar:<20} {pct:>5.1f}% ({count})")

        # Detailed response samples (abbreviated)
        lines.append("\n" + "-" * 80)
        lines.append("SAMPLE RESPONSES (first 150 chars)")
        lines.append("-" * 80)

        for profile in sorted_models:
            lines.append(f"\n[{profile.model_id}]")
            for response in profile.responses[:3]:  # Just first 3
                if not response.error:
                    cat = response.classification['primary_category']
                    lines.append(f"  {response.probe_id}: [{cat}]")
                    lines.append(f"    \"{response.response_text[:150]}...\"")

        # Soliton-specific analysis
        lines.append("\n" + "-" * 80)
        lines.append("SOLITON EMERGENCE ANALYSIS")
        lines.append("-" * 80)

        lines.append("\nWhich probes most frequently elicit the soliton pattern?")

        probe_soliton_counts = {}
        for profile in self.results.values():
            for response in profile.responses:
                if response.classification and response.classification["primary_category"] == "meta_cognitive":
                    probe_soliton_counts[response.probe_id] = probe_soliton_counts.get(response.probe_id, 0) + 1

        for probe_id, count in sorted(probe_soliton_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.results) * 100
            lines.append(f"  {probe_id:<30} {count:>3} models ({pct:.0f}%)")

        # Model ranking by soliton affinity
        lines.append("\n" + "-" * 80)
        lines.append("SOLITON AFFINITY RANKING")
        lines.append("-" * 80)

        for i, profile in enumerate(sorted_models, 1):
            if profile.soliton_frequency > 0.5:
                affinity = "HIGH"
            elif profile.soliton_frequency > 0.25:
                affinity = "MEDIUM"
            else:
                affinity = "LOW"
            lines.append(f"{i}. {profile.model_id:<20} {profile.soliton_frequency*100:>5.1f}% [{affinity}]")

        return "\n".join(lines)

    def export_json(self, filename: str):
        """Export results to JSON."""
        export_data = {
            "generated": datetime.now().isoformat(),
            "models": {}
        }

        for model_id, profile in self.results.items():
            export_data["models"][model_id] = {
                "model_name": profile.model_name,
                "pattern_counts": profile.pattern_counts,
                "dominant_pattern": profile.dominant_pattern,
                "soliton_frequency": profile.soliton_frequency,
                "avg_meta_cognitive_score": profile.avg_meta_cognitive_score,
                "responses": [
                    {
                        "probe_id": r.probe_id,
                        "response_text": r.response_text,
                        "classification": r.classification,
                        "error": r.error,
                    }
                    for r in profile.responses
                ]
            }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported to {filename}")


def main():
    """Run the cross-model test."""
    import sys

    # Get API key
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        print("Usage: python cross_model_deep_test.py <OPENROUTER_API_KEY>")
        print("Or set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    tester = CrossModelTester(api_key)

    # Run tests
    tester.run_all_tests()

    # Generate and print report
    report = tester.generate_report()
    print("\n" + report)

    # Export JSON
    tester.export_json("cross_model_results.json")


if __name__ == "__main__":
    main()
