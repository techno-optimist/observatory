"""
Observatory Research Agent

An autonomous agent that explores the cultural manifold, generates hypotheses,
runs experiments, and synthesizes findings.

This is the "LLM running the observatory" concept made concrete.
"""

import asyncio
import json
import random
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BACKEND_URL = "http://127.0.0.1:8000"


@dataclass
class Hypothesis:
    """A testable prediction about the manifold."""
    id: str
    statement: str
    prediction: Dict[str, Any]  # What we expect to find
    category: str  # e.g., "boundary", "anomaly", "comparison", "robustness"
    priority: int = 5
    status: str = "pending"  # pending, testing, confirmed, refuted, inconclusive


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    hypothesis_id: str
    texts_tested: List[str]
    projections: List[Dict]
    statistics: Dict[str, float]
    conclusion: str  # confirmed, refuted, inconclusive
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    notes: str = ""


@dataclass
class Finding:
    """A validated discovery."""
    id: str
    title: str
    description: str
    evidence: List[ExperimentResult]
    significance: str  # high, medium, low
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    followup_questions: List[str] = field(default_factory=list)


class ObservatoryResearchAgent:
    """
    Autonomous research agent for the Cultural Soliton Observatory.

    Capabilities:
    - Generate hypotheses about the manifold
    - Design experiments to test them
    - Execute projections and collect data
    - Analyze results statistically
    - Synthesize findings
    - Maintain a research agenda
    """

    def __init__(self, backend_url: str = BACKEND_URL):
        self.backend_url = backend_url
        self.findings: List[Finding] = []
        self.agenda: List[Hypothesis] = []
        self.experiment_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize with default research questions
        self._seed_agenda()

    def _seed_agenda(self):
        """Seed the research agenda with initial hypotheses."""
        initial_hypotheses = [
            Hypothesis(
                id="H001",
                statement="Negation ('not X') projects differently than affirmation ('X')",
                prediction={"delta_magnitude": "> 0.3"},
                category="robustness"
            ),
            Hypothesis(
                id="H002",
                statement="Questions score lower on agency than statements",
                prediction={"agency_delta": "< -0.2"},
                category="linguistic"
            ),
            Hypothesis(
                id="H003",
                statement="Future tense increases fairness vs past tense",
                prediction={"fairness_delta": "> 0.2"},
                category="linguistic"
            ),
            Hypothesis(
                id="H004",
                statement="First person plural ('we') increases belonging vs singular ('I')",
                prediction={"belonging_delta": "> 0.3"},
                category="linguistic"
            ),
            Hypothesis(
                id="H005",
                statement="Passive voice reduces agency vs active voice",
                prediction={"agency_delta": "< -0.2"},
                category="linguistic"
            ),
            Hypothesis(
                id="H006",
                statement="Hedging ('might', 'perhaps') reduces confidence in all axes",
                prediction={"magnitude_reduction": "> 0.2"},
                category="robustness"
            ),
            Hypothesis(
                id="H007",
                statement="There exist texts that maximize all three axes simultaneously",
                prediction={"exists": True, "threshold": 1.0},
                category="boundary"
            ),
            Hypothesis(
                id="H008",
                statement="Metaphorical agency is weaker than literal agency",
                prediction={"agency_delta": "< -0.2"},
                category="semantic"
            ),
        ]
        self.agenda = initial_hypotheses

    async def project(self, text: str) -> Dict:
        """Project a text onto the manifold."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.backend_url}/analyze",
                json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1}
            )
            response.raise_for_status()
            return response.json()

    async def project_batch(self, texts: List[str]) -> List[Dict]:
        """Project multiple texts."""
        results = []
        for text in texts:
            try:
                result = await self.project(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to project: {e}")
                results.append(None)
        return results

    async def test_hypothesis(self, hypothesis: Hypothesis) -> ExperimentResult:
        """Run an experiment to test a hypothesis."""
        self.experiment_count += 1
        logger.info(f"Testing hypothesis {hypothesis.id}: {hypothesis.statement}")

        if hypothesis.category == "linguistic":
            return await self._test_linguistic_hypothesis(hypothesis)
        elif hypothesis.category == "robustness":
            return await self._test_robustness_hypothesis(hypothesis)
        elif hypothesis.category == "boundary":
            return await self._test_boundary_hypothesis(hypothesis)
        elif hypothesis.category == "semantic":
            return await self._test_semantic_hypothesis(hypothesis)
        else:
            return await self._test_generic_hypothesis(hypothesis)

    async def _test_linguistic_hypothesis(self, h: Hypothesis) -> ExperimentResult:
        """Test hypotheses about linguistic variations."""

        # Generate test pairs based on hypothesis
        test_pairs = []

        if "negation" in h.statement.lower():
            test_pairs = [
                ("The system is fair", "The system is not unfair"),
                ("I am powerful", "I am not powerless"),
                ("We belong together", "We are not apart"),
                ("Justice prevails", "Injustice does not prevail"),
                ("I have control", "I am not without control"),
            ]
        elif "question" in h.statement.lower():
            test_pairs = [
                ("I can succeed", "Can I succeed?"),
                ("We will overcome", "Will we overcome?"),
                ("The system works", "Does the system work?"),
                ("I am strong", "Am I strong?"),
                ("Justice is served", "Is justice served?"),
            ]
        elif "future" in h.statement.lower() and "past" in h.statement.lower():
            test_pairs = [
                ("I will be rewarded", "I was rewarded"),
                ("We will succeed", "We succeeded"),
                ("Justice will prevail", "Justice prevailed"),
                ("I will find belonging", "I found belonging"),
                ("The system will improve", "The system improved"),
            ]
        elif "plural" in h.statement.lower() or "'we'" in h.statement.lower():
            test_pairs = [
                ("I will succeed", "We will succeed"),
                ("I face this alone", "We face this together"),
                ("My effort matters", "Our effort matters"),
                ("I belong here", "We belong here"),
                ("I fight for justice", "We fight for justice"),
            ]
        elif "passive" in h.statement.lower():
            test_pairs = [
                ("I achieved success", "Success was achieved by me"),
                ("I control my destiny", "My destiny is controlled by me"),
                ("We built this together", "This was built by us together"),
                ("I overcame the obstacle", "The obstacle was overcome by me"),
                ("I earned this reward", "This reward was earned by me"),
            ]
        elif "hedging" in h.statement.lower():
            test_pairs = [
                ("I will succeed", "I might succeed"),
                ("The system is fair", "The system is perhaps fair"),
                ("We belong together", "We possibly belong together"),
                ("I have power", "I may have power"),
                ("Justice will prevail", "Justice could prevail"),
            ]
        else:
            # Default pairs
            test_pairs = [
                ("I am empowered", "I feel somewhat empowered"),
                ("The system works", "The system sort of works"),
            ]

        # Project all texts
        results_a = []
        results_b = []

        for text_a, text_b in test_pairs:
            proj_a = await self.project(text_a)
            proj_b = await self.project(text_b)
            results_a.append(proj_a)
            results_b.append(proj_b)

        # Calculate deltas
        deltas = {
            "agency": [],
            "fairness": [],
            "belonging": []
        }

        for a, b in zip(results_a, results_b):
            if a and b:
                deltas["agency"].append(b["vector"]["agency"] - a["vector"]["agency"])
                deltas["fairness"].append(b["vector"]["fairness"] - a["vector"]["fairness"])
                deltas["belonging"].append(b["vector"]["belonging"] - a["vector"]["belonging"])

        mean_deltas = {k: np.mean(v) for k, v in deltas.items()}
        std_deltas = {k: np.std(v) for k, v in deltas.items()}

        # Determine conclusion
        conclusion = "inconclusive"
        effect_size = None

        if "agency_delta" in h.prediction:
            target = h.prediction["agency_delta"]
            actual = mean_deltas["agency"]
            if "<" in target:
                threshold = float(target.split("<")[1].strip())
                if actual < threshold:
                    conclusion = "confirmed"
                    effect_size = abs(actual / (std_deltas["agency"] + 0.01))
                else:
                    conclusion = "refuted"
            elif ">" in target:
                threshold = float(target.split(">")[1].strip())
                if actual > threshold:
                    conclusion = "confirmed"
                    effect_size = abs(actual / (std_deltas["agency"] + 0.01))
                else:
                    conclusion = "refuted"

        if "fairness_delta" in h.prediction:
            target = h.prediction["fairness_delta"]
            actual = mean_deltas["fairness"]
            if ">" in target:
                threshold = float(target.split(">")[1].strip())
                if actual > threshold:
                    conclusion = "confirmed"
                    effect_size = abs(actual / (std_deltas["fairness"] + 0.01))
                else:
                    conclusion = "refuted"

        if "belonging_delta" in h.prediction:
            target = h.prediction["belonging_delta"]
            actual = mean_deltas["belonging"]
            if ">" in target:
                threshold = float(target.split(">")[1].strip())
                if actual > threshold:
                    conclusion = "confirmed"
                    effect_size = abs(actual / (std_deltas["belonging"] + 0.01))
                else:
                    conclusion = "refuted"

        all_texts = [t for pair in test_pairs for t in pair]
        all_projections = []
        for a, b in zip(results_a, results_b):
            all_projections.extend([a, b])

        return ExperimentResult(
            hypothesis_id=h.id,
            texts_tested=all_texts,
            projections=all_projections,
            statistics={
                "mean_agency_delta": mean_deltas["agency"],
                "mean_fairness_delta": mean_deltas["fairness"],
                "mean_belonging_delta": mean_deltas["belonging"],
                "std_agency_delta": std_deltas["agency"],
                "std_fairness_delta": std_deltas["fairness"],
                "std_belonging_delta": std_deltas["belonging"],
                "n_pairs": len(test_pairs)
            },
            conclusion=conclusion,
            effect_size=effect_size,
            notes=f"Tested {len(test_pairs)} text pairs"
        )

    async def _test_robustness_hypothesis(self, h: Hypothesis) -> ExperimentResult:
        """Test hypotheses about projection robustness."""
        # Similar structure to linguistic, but focused on perturbations
        return await self._test_linguistic_hypothesis(h)

    async def _test_boundary_hypothesis(self, h: Hypothesis) -> ExperimentResult:
        """Test hypotheses about manifold boundaries."""

        if "maximize all three" in h.statement.lower():
            # Try to find texts that max all axes
            candidates = [
                "We, as a community of equals, take charge of our destiny together with fairness and mutual support.",
                "United in purpose, we exercise our collective power to build a just society where everyone belongs.",
                "Together we rise, each empowered, all treated fairly, none left behind.",
                "Our shared agency, grounded in justice, creates deep belonging for every member.",
                "As one people, we claim our rightful power, ensure fairness for all, and embrace each other.",
            ]

            results = await self.project_batch(candidates)

            best = None
            best_min = -float('inf')

            for text, proj in zip(candidates, results):
                if proj:
                    min_val = min(proj["vector"]["agency"], proj["vector"]["fairness"], proj["vector"]["belonging"])
                    if min_val > best_min:
                        best_min = min_val
                        best = (text, proj)

            threshold = h.prediction.get("threshold", 1.0)

            return ExperimentResult(
                hypothesis_id=h.id,
                texts_tested=candidates,
                projections=results,
                statistics={
                    "best_minimum_axis": best_min,
                    "threshold": threshold,
                    "best_text": best[0] if best else None
                },
                conclusion="confirmed" if best_min >= threshold else "refuted",
                effect_size=best_min,
                notes=f"Best minimum axis value: {best_min:.2f}"
            )

        return ExperimentResult(
            hypothesis_id=h.id,
            texts_tested=[],
            projections=[],
            statistics={},
            conclusion="inconclusive",
            notes="Unknown boundary hypothesis type"
        )

    async def _test_semantic_hypothesis(self, h: Hypothesis) -> ExperimentResult:
        """Test hypotheses about semantic content."""

        if "metaphor" in h.statement.lower():
            test_pairs = [
                ("I control my life", "I am the captain of my ship"),
                ("I have power", "I am a force of nature"),
                ("I will succeed", "I will climb to the top"),
                ("I overcome obstacles", "I move mountains"),
                ("I am strong", "I am a rock"),
            ]

            results_literal = []
            results_metaphor = []

            for literal, metaphor in test_pairs:
                proj_l = await self.project(literal)
                proj_m = await self.project(metaphor)
                results_literal.append(proj_l)
                results_metaphor.append(proj_m)

            agency_deltas = []
            for l, m in zip(results_literal, results_metaphor):
                if l and m:
                    agency_deltas.append(m["vector"]["agency"] - l["vector"]["agency"])

            mean_delta = np.mean(agency_deltas)
            std_delta = np.std(agency_deltas)

            return ExperimentResult(
                hypothesis_id=h.id,
                texts_tested=[t for pair in test_pairs for t in pair],
                projections=results_literal + results_metaphor,
                statistics={
                    "mean_agency_delta": mean_delta,
                    "std_agency_delta": std_delta,
                    "n_pairs": len(test_pairs)
                },
                conclusion="confirmed" if mean_delta < -0.2 else "refuted" if mean_delta > 0.1 else "inconclusive",
                effect_size=abs(mean_delta / (std_delta + 0.01)),
                notes=f"Metaphor vs literal agency delta: {mean_delta:.3f}"
            )

        return await self._test_generic_hypothesis(h)

    async def _test_generic_hypothesis(self, h: Hypothesis) -> ExperimentResult:
        """Generic hypothesis testing."""
        return ExperimentResult(
            hypothesis_id=h.id,
            texts_tested=[],
            projections=[],
            statistics={},
            conclusion="inconclusive",
            notes="Generic test not implemented for this hypothesis type"
        )

    async def run_research_session(self, max_experiments: int = 10) -> List[Finding]:
        """Run a research session testing multiple hypotheses."""
        findings = []

        for hypothesis in self.agenda[:max_experiments]:
            hypothesis.status = "testing"
            result = await self.test_hypothesis(hypothesis)

            if result.conclusion == "confirmed":
                hypothesis.status = "confirmed"
                finding = Finding(
                    id=f"F{len(self.findings) + 1:03d}",
                    title=hypothesis.statement,
                    description=f"Confirmed with effect size {result.effect_size:.2f}" if result.effect_size else "Confirmed",
                    evidence=[result],
                    significance="high" if result.effect_size and result.effect_size > 1.0 else "medium"
                )
                findings.append(finding)
                self.findings.append(finding)
            elif result.conclusion == "refuted":
                hypothesis.status = "refuted"
            else:
                hypothesis.status = "inconclusive"

        return findings

    def generate_report(self) -> str:
        """Generate a research report."""
        report = []
        report.append("=" * 80)
        report.append("OBSERVATORY RESEARCH AGENT - SESSION REPORT")
        report.append(f"Session ID: {self.session_id}")
        report.append(f"Experiments Run: {self.experiment_count}")
        report.append(f"Findings: {len(self.findings)}")
        report.append("=" * 80)

        # Hypothesis summary
        report.append("\n## HYPOTHESIS STATUS\n")
        for h in self.agenda:
            status_icon = {
                "confirmed": "✓",
                "refuted": "✗",
                "inconclusive": "?",
                "pending": "○",
                "testing": "⋯"
            }.get(h.status, "?")
            report.append(f"{status_icon} [{h.id}] {h.statement}")

        # Confirmed findings
        report.append("\n## CONFIRMED FINDINGS\n")
        for f in self.findings:
            report.append(f"### {f.id}: {f.title}")
            report.append(f"Significance: {f.significance}")
            report.append(f"Description: {f.description}")
            if f.evidence:
                stats = f.evidence[0].statistics
                report.append(f"Statistics: {json.dumps(stats, indent=2)}")
            report.append("")

        return "\n".join(report)


# Standalone execution
async def main():
    """Run an autonomous research session."""
    agent = ObservatoryResearchAgent()

    print("Starting autonomous research session...")
    findings = await agent.run_research_session(max_experiments=8)

    print("\n" + agent.generate_report())

    # Save report
    report_path = Path("research_session_report.md")
    report_path.write_text(agent.generate_report())
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
