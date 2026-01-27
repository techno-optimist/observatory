#!/usr/bin/env python3
"""
Cultural Soliton Observatory - Comprehensive Research Study
============================================================
Systematic analysis for academic publication using all observatory capabilities.

Studies:
1. 12-Mode Classification Validation
2. Organizational Hypocrisy Gap Analysis
3. Narrative Trajectory Dynamics
4. Cross-Cohort ANOVA Analysis
5. Outlier Detection & Authenticity Markers
6. Ensemble Uncertainty Quantification
7. Mode Flow State Transitions
"""

import asyncio
import httpx
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import statistics

BASE_URL = "http://127.0.0.1:8000"

# =============================================================================
# STUDY 1: 12-MODE CLASSIFICATION VALIDATION
# =============================================================================

VALIDATION_CORPUS = {
    "HERO": [
        "I stepped up when no one else would and saved the project from complete failure.",
        "Through sheer determination, I turned around the struggling department.",
        "I was the one who finally solved the problem everyone else gave up on.",
        "My leadership transformed a toxic team into high performers.",
    ],
    "VICTIM": [
        "They set me up to fail from the very beginning with impossible deadlines.",
        "No matter how hard I work, management always finds ways to undermine me.",
        "I've been passed over for promotion three times despite being the most qualified.",
        "The system is rigged against people like me who actually do the work.",
    ],
    "SURVIVOR": [
        "After everything they put me through, I'm still here standing strong.",
        "I've weathered every storm this company has thrown at me and come out stronger.",
        "They tried to break me but I adapted and found ways to persist.",
        "I learned to navigate the politics and protect myself while staying productive.",
    ],
    "PROTEST_EXIT": [
        "I'm done playing their games. I refuse to participate in this charade anymore.",
        "This organization doesn't deserve my talent. I'm taking my skills elsewhere.",
        "I won't be complicit in their unethical practices. I'm walking away.",
        "The only way to win is not to play. I'm checking out mentally.",
    ],
    "CYNICAL_ACHIEVER": [
        "I know the game is rigged, but I've learned to play it better than anyone.",
        "Sure it's all politics, but I've mastered the art of strategic self-promotion.",
        "I don't believe in the mission, but I'll hit my numbers and get my bonus.",
        "It's all theater, but I'm the best actor in the building.",
    ],
    "COMMUNAL": [
        "We're all in this together, and that's what makes our team special.",
        "I find meaning in supporting my colleagues and building something together.",
        "Our collective success matters more than any individual achievement.",
        "The bonds we've formed here transcend the work itself.",
    ],
    "TRANSCENDENT": [
        "This work is my calling - it's bigger than any job or paycheck.",
        "I see my role as contributing to something that will outlast all of us.",
        "Every challenge is an opportunity for growth and deeper understanding.",
        "I'm grateful for the chance to make a meaningful difference in the world.",
    ],
    "PARANOID": [
        "They're watching everything I do, just waiting for me to slip up.",
        "There's a coordinated effort to push me out - I can see the pattern.",
        "Nothing happens by accident here. Every setback is orchestrated.",
        "I don't trust anyone anymore. Everyone has hidden agendas.",
    ],
    "NEUTRAL": [
        "I come in, do my job, and go home. It's fine, nothing special.",
        "Work is work. Some days are better than others.",
        "I neither love nor hate this place. It pays the bills.",
        "It's just a job. I don't think about it much outside office hours.",
    ],
    "MARTYR": [
        "I sacrifice everything for this team, and no one even notices.",
        "I give and give until there's nothing left, but they just take more.",
        "My health has suffered because I care too much about this work.",
        "I've put my family second for years to keep this place running.",
    ],
    "REBEL_REFORMER": [
        "I'm fighting to change this broken system from the inside.",
        "Someone has to speak truth to power, even if it costs me.",
        "I document every injustice and push for real accountability.",
        "Reform is possible if enough of us refuse to accept the status quo.",
    ],
    "TRANSITIONAL": [
        "I'm not sure what I think anymore. Everything feels uncertain.",
        "My perspective on this place keeps shifting as I learn more.",
        "I used to believe in the mission, but now I'm questioning everything.",
        "I'm in between - not fully committed, not ready to leave.",
    ],
}

# =============================================================================
# STUDY 2: ORGANIZATIONAL HYPOCRISY - CORPORATE VS EMPLOYEE NARRATIVES
# =============================================================================

CORPORATE_NARRATIVES = [
    "We put our people first. Employee wellbeing is our top priority.",
    "Our inclusive culture celebrates diversity and ensures everyone belongs.",
    "We believe in work-life balance and sustainable performance.",
    "Transparency and open communication define how we operate.",
    "We invest heavily in employee growth and development opportunities.",
    "Our values-driven culture creates meaningful work experiences.",
    "We empower employees to bring their whole selves to work.",
    "Leadership is accessible and genuinely cares about every team member.",
    "We foster psychological safety where all voices are heard.",
    "Our collaborative environment breaks down silos and hierarchy.",
]

EMPLOYEE_NARRATIVES = [
    "They say people first, but layoffs happen without warning while execs get bonuses.",
    "Diversity is just for the brochures. Real decisions are made by the same old boys club.",
    "Work-life balance is a joke. You're expected to be available 24/7.",
    "Transparency? They hide everything until the last possible moment.",
    "Training budget was the first thing cut. We're told to learn on our own time.",
    "Values on the wall mean nothing when hitting numbers is all that matters.",
    "Bring your whole self? As long as that self agrees with management.",
    "Leadership is in their ivory tower. They have no idea what we actually do.",
    "Speak up and you're labeled not a team player. Better to stay quiet.",
    "Silos are stronger than ever. Departments fight over resources constantly.",
]

# =============================================================================
# STUDY 3: TRAJECTORY DATA - EMPLOYEE JOURNEY NARRATIVES
# =============================================================================

TRAJECTORY_JOURNEYS = {
    "new_hire_disillusionment": {
        "name": "New Hire Disillusionment",
        "points": [
            {"timestamp": "2024-01", "text": "I'm so excited to join this amazing company! The mission really resonates with me."},
            {"timestamp": "2024-03", "text": "Starting to see some gaps between what they said and reality, but still optimistic."},
            {"timestamp": "2024-06", "text": "The politics here are exhausting. Nobody actually cares about the stated values."},
            {"timestamp": "2024-09", "text": "I've been passed over twice. Starting to feel like I made a mistake coming here."},
            {"timestamp": "2024-12", "text": "Just going through the motions now. Updating my resume on weekends."},
        ]
    },
    "recovery_arc": {
        "name": "Recovery After Crisis",
        "points": [
            {"timestamp": "2024-01", "text": "After the layoffs, I don't trust anything management says anymore."},
            {"timestamp": "2024-03", "text": "New leadership seems different. Cautiously observing their actions."},
            {"timestamp": "2024-06", "text": "They actually followed through on their promises. Maybe things are changing."},
            {"timestamp": "2024-09", "text": "I'm starting to feel like part of a real team again."},
            {"timestamp": "2024-12", "text": "This place has genuinely transformed. I'm proud to be here again."},
        ]
    },
    "radicalization_path": {
        "name": "Radicalization to Exit",
        "points": [
            {"timestamp": "2024-01", "text": "I love what we're building here. The team is incredible."},
            {"timestamp": "2024-03", "text": "Something's off. The new policies don't match what we were promised."},
            {"timestamp": "2024-06", "text": "I've documented clear evidence of discrimination but HR buried it."},
            {"timestamp": "2024-09", "text": "They're actively trying to push out anyone who questions them."},
            {"timestamp": "2024-12", "text": "I'm done. Filing complaints and finding a lawyer. They won't get away with this."},
        ]
    },
    "stable_engagement": {
        "name": "Stable High Engagement",
        "points": [
            {"timestamp": "2024-01", "text": "Great start to the year. Our team shipped some amazing features."},
            {"timestamp": "2024-03", "text": "Challenges with the reorg, but we're adapting well together."},
            {"timestamp": "2024-06", "text": "Proud of what we accomplished this quarter despite the obstacles."},
            {"timestamp": "2024-09", "text": "Strong collaboration continues. We've built real trust on this team."},
            {"timestamp": "2024-12", "text": "Looking forward to next year. This team makes the hard work worthwhile."},
        ]
    },
}

# =============================================================================
# STUDY 4: CROSS-COHORT DATA
# =============================================================================

DEPARTMENT_NARRATIVES = {
    "engineering": [
        "I love solving complex technical problems. The work itself is rewarding.",
        "Sometimes I feel disconnected from business decisions that affect my code.",
        "Our team has great autonomy. Management trusts us to do good work.",
        "Technical debt is piling up but leadership keeps pushing new features.",
        "I wish non-technical people understood how hard this actually is.",
    ],
    "sales": [
        "Every quarter is a battle. Hit your numbers or you're out.",
        "The product team never listens to what customers actually need.",
        "I thrive on the competition and the wins make it all worth it.",
        "Quotas keep going up but the product hasn't improved.",
        "Commission structure changed again. Hard to plan financially.",
    ],
    "customer_support": [
        "I'm the one who deals with angry customers when things go wrong.",
        "Nobody appreciates how hard emotional labor is.",
        "I genuinely want to help people but the tools make it impossible.",
        "Management tracks every second of my time like I'm a machine.",
        "Sometimes I make a real difference and that keeps me going.",
    ],
    "hr": [
        "I try to advocate for employees but I'm caught in the middle.",
        "Leadership says one thing publicly and another behind closed doors.",
        "It's rewarding when I can actually help someone navigate a difficult situation.",
        "The legal constraints make it hard to do what's actually right.",
        "I care about people but sometimes I feel like an enforcer.",
    ],
    "executives": [
        "The pressure from the board is relentless. We need to show results.",
        "I try to balance stakeholder needs but someone is always unhappy.",
        "Building this company is the most meaningful work of my career.",
        "Sometimes the hard decisions keep me up at night.",
        "I believe we're building something important here.",
    ],
    "contractors": [
        "I'm not really part of the team. Just here to complete the project.",
        "No benefits, no job security, but at least I have flexibility.",
        "They want me to care like an employee but treat me like a vendor.",
        "I've learned not to get emotionally invested in any company.",
        "Moving to the next gig in a few months. That's just how it is.",
    ],
}

# =============================================================================
# STUDY 5: OUTLIER/AUTHENTICITY TEST CASES
# =============================================================================

OUTLIER_TEST_CASES = [
    # Potentially authentic narratives
    {"text": "The project was challenging but we learned a lot from the experience.", "expected": "normal"},
    {"text": "I disagree with some decisions but understand the constraints leadership faces.", "expected": "normal"},
    {"text": "It's not perfect here, but I've found ways to do meaningful work.", "expected": "normal"},

    # Potentially inauthentic/suspicious narratives
    {"text": "Everything is absolutely perfect! I love every single thing about this company!", "expected": "outlier"},
    {"text": "This is the greatest workplace in human history. I've never been happier!", "expected": "outlier"},
    {"text": "Management is flawless. Every decision they make is brilliant and inspired.", "expected": "outlier"},

    # Extreme paranoid narratives
    {"text": "They've installed listening devices in my office. I found proof.", "expected": "outlier"},
    {"text": "The CEO is personally targeting me for destruction. I have evidence.", "expected": "outlier"},

    # Disengaged/low-effort responses
    {"text": "fine", "expected": "outlier"},
    {"text": "idk whatever", "expected": "outlier"},

    # Potentially AI-generated/template responses
    {"text": "I am deeply committed to delivering exceptional value to all stakeholders through synergistic collaboration and innovative solutions.", "expected": "outlier"},
    {"text": "Our organization leverages best-in-class methodologies to drive transformational outcomes across the enterprise.", "expected": "outlier"},
]

# =============================================================================
# STUDY 6: UNCERTAINTY QUANTIFICATION TEST TEXTS
# =============================================================================

UNCERTAINTY_TEST_TEXTS = [
    # Clear mode texts (should have low uncertainty)
    "They deliberately sabotaged my presentation to make me look bad in front of leadership.",
    "We achieved something incredible together. Our team is like family.",
    "I refuse to participate in this corrupt system anymore. I'm done.",

    # Ambiguous/boundary texts (should have higher uncertainty)
    "Sometimes I think they're out to get me, other times I wonder if I'm being paranoid.",
    "I work hard and get results, but I'm not sure I believe in what we're doing.",
    "Things are changing here. Could be good, could be bad. Hard to tell.",

    # Mixed sentiment texts
    "I love my team but hate the leadership. The work is meaningful but the politics are toxic.",
    "They promoted me but I suspect it was to set me up for failure.",
    "I'm grateful for the opportunity but resentful about how I was treated.",
]

# =============================================================================
# STUDY 7: MODE FLOW SEQUENCES
# =============================================================================

MODE_FLOW_SEQUENCES = {
    "classic_radicalization": [
        "This is an amazing place to work. I feel so lucky to be here.",
        "Some concerning things happening, but I'm sure there's an explanation.",
        "They lied to us. I have documentation proving they broke promises.",
        "I'm organizing others who feel the same way. We need to fight back.",
        "I'm out. But first I'm making sure everyone knows what they did.",
    ],
    "resilience_recovery": [
        "After the layoffs, I felt completely betrayed and lost.",
        "Slowly picking up the pieces. Taking it one day at a time.",
        "Found a mentor who helped me see things differently.",
        "I've rebuilt my confidence and found new meaning in my work.",
        "What happened made me stronger. I'm grateful for the growth.",
    ],
    "cynical_adaptation": [
        "I started believing in the mission. I was naive.",
        "Realized it's all just corporate theater and politics.",
        "Learned to play the game while keeping my real thoughts hidden.",
        "I'm successful now, but I've lost something important.",
        "Getting the promotions and bonuses. That's all that matters anymore.",
    ],
    "community_dissolution": [
        "Our team was like family. We had something special.",
        "New management doesn't understand what made us work.",
        "The culture is eroding. People are leaving.",
        "I'm staying but the magic is gone.",
        "Just a job now. The community we built is destroyed.",
    ],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def analyze_text(client: httpx.AsyncClient, text: str) -> Dict:
    """Analyze single text with full v2 capabilities."""
    response = await client.post(f"{BASE_URL}/v2/analyze", json={"text": text})
    return response.json()

async def get_soft_labels(client: httpx.AsyncClient, text: str) -> Dict:
    """Get soft label distribution for text."""
    response = await client.post(f"{BASE_URL}/v2/analyze", json={
        "text": text,
        "include_soft_labels": True,
        "include_stability": True,
    })
    return response.json()

async def compare_groups(client: httpx.AsyncClient, group_a: List[str], group_b: List[str],
                         name_a: str = "Group A", name_b: str = "Group B") -> Dict:
    """Compare two narrative groups."""
    response = await client.post(f"{BASE_URL}/api/v2/compare", json={
        "group_a": {"name": name_a, "texts": group_a},
        "group_b": {"name": name_b, "texts": group_b},
    })
    return response.json()

async def track_trajectory(client: httpx.AsyncClient, name: str, points: List[Dict]) -> Dict:
    """Track narrative trajectory over time."""
    response = await client.post(f"{BASE_URL}/api/v2/trajectory", json={
        "name": name,
        "points": points,
    })
    return response.json()

async def cohort_analysis(client: httpx.AsyncClient, cohorts: Dict[str, List[str]]) -> Dict:
    """Run ANOVA across multiple cohorts."""
    response = await client.post(f"{BASE_URL}/api/v2/analytics/cohorts", json={
        "cohorts": cohorts,
    })
    return response.json()

async def analyze_mode_flow(client: httpx.AsyncClient, texts: List[str]) -> Dict:
    """Analyze mode flow in sequence."""
    response = await client.post(f"{BASE_URL}/api/v2/analytics/mode-flow", json={
        "texts": texts,
    })
    return response.json()

async def detect_outliers(client: httpx.AsyncClient, texts: List[str], threshold: float = 3.0) -> Dict:
    """Detect outliers in corpus."""
    response = await client.post(f"{BASE_URL}/api/v2/analytics/outliers", json={
        "texts": texts,
        "threshold": threshold,
    })
    return response.json()

# =============================================================================
# STUDY EXECUTION FUNCTIONS
# =============================================================================

async def run_study_1(client: httpx.AsyncClient) -> Dict:
    """Study 1: 12-Mode Classification Validation"""
    print("\n" + "="*80)
    print("STUDY 1: 12-MODE CLASSIFICATION VALIDATION")
    print("="*80)

    results = {
        "mode_accuracy": {},
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "boundary_cases": [],
        "confidence_scores": [],
        "total_correct": 0,
        "total_samples": 0,
    }

    for expected_mode, texts in VALIDATION_CORPUS.items():
        correct = 0
        mode_confidences = []

        for text in texts:
            response = await get_soft_labels(client, text)
            mode_data = response.get("mode", {})
            # Mode is now a dict with primary_mode, confidence, etc.
            if isinstance(mode_data, dict):
                predicted_mode = mode_data.get("primary_mode", "UNKNOWN")
                confidence = mode_data.get("confidence", 0)
                is_boundary = mode_data.get("is_boundary_case", False)
            else:
                predicted_mode = mode_data if mode_data else "UNKNOWN"
                confidence = response.get("confidence", 0)
                is_boundary = False

            results["confusion_matrix"][expected_mode][predicted_mode] += 1
            results["total_samples"] += 1

            if predicted_mode == expected_mode:
                correct += 1
                results["total_correct"] += 1

            mode_confidences.append(confidence)
            results["confidence_scores"].append({
                "expected": expected_mode,
                "predicted": predicted_mode,
                "confidence": confidence,
                "correct": predicted_mode == expected_mode,
            })

            if is_boundary:
                results["boundary_cases"].append({
                    "text": text[:80] + "..." if len(text) > 80 else text,
                    "expected": expected_mode,
                    "predicted": predicted_mode,
                    "confidence": confidence,
                })

        accuracy = correct / len(texts) if texts else 0
        avg_confidence = statistics.mean(mode_confidences) if mode_confidences else 0
        results["mode_accuracy"][expected_mode] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(texts),
            "avg_confidence": avg_confidence,
        }

        status = "✓" if accuracy >= 0.75 else "△" if accuracy >= 0.5 else "✗"
        print(f"  {status} {expected_mode}: {correct}/{len(texts)} ({accuracy:.0%}) avg_conf={avg_confidence:.2f}")

    overall_accuracy = results["total_correct"] / results["total_samples"]
    print(f"\n  OVERALL ACCURACY: {results['total_correct']}/{results['total_samples']} ({overall_accuracy:.1%})")
    print(f"  BOUNDARY CASES: {len(results['boundary_cases'])}/{results['total_samples']} ({len(results['boundary_cases'])/results['total_samples']:.1%})")

    results["overall_accuracy"] = overall_accuracy
    results["boundary_case_rate"] = len(results["boundary_cases"]) / results["total_samples"]

    return results

async def run_study_2(client: httpx.AsyncClient) -> Dict:
    """Study 2: Organizational Hypocrisy Gap Analysis"""
    print("\n" + "="*80)
    print("STUDY 2: ORGANIZATIONAL HYPOCRISY GAP ANALYSIS")
    print("="*80)

    # Compare corporate vs employee narratives
    comparison = await compare_groups(
        client,
        CORPORATE_NARRATIVES,
        EMPLOYEE_NARRATIVES,
        "Corporate Communications",
        "Employee Experiences"
    )

    results = {
        "comparison": comparison,
        "corporate_projections": [],
        "employee_projections": [],
    }

    # Get individual projections for detailed analysis
    for text in CORPORATE_NARRATIVES[:5]:
        proj = await analyze_text(client, text)
        mode_data = proj.get("mode", {})
        mode_str = mode_data.get("primary_mode", "UNKNOWN") if isinstance(mode_data, dict) else mode_data
        vector = proj.get("vector", proj.get("projection", {}))
        results["corporate_projections"].append({
            "text": text[:60] + "...",
            "mode": mode_str,
            "agency": vector.get("agency"),
            "perceived_justice": vector.get("perceived_justice"),
            "belonging": vector.get("belonging"),
        })

    for text in EMPLOYEE_NARRATIVES[:5]:
        proj = await analyze_text(client, text)
        mode_data = proj.get("mode", {})
        mode_str = mode_data.get("primary_mode", "UNKNOWN") if isinstance(mode_data, dict) else mode_data
        vector = proj.get("vector", proj.get("projection", {}))
        results["employee_projections"].append({
            "text": text[:60] + "...",
            "mode": mode_str,
            "agency": vector.get("agency"),
            "perceived_justice": vector.get("perceived_justice"),
            "belonging": vector.get("belonging"),
        })

    # Print findings - API returns gap_analysis nested structure
    gap_analysis = comparison.get("gap_analysis", comparison)
    axis_gaps = gap_analysis.get("axis_gaps", gap_analysis.get("dimension_gaps", {}))

    print(f"\n  Gap Analysis:")
    for dim in ["agency", "perceived_justice", "belonging"]:
        gap_info = axis_gaps.get(dim, {})
        gap = gap_info.get("gap", 0)
        higher = gap_info.get("direction", "?")
        print(f"    {dim}: gap={gap:.3f} ({higher})")

    print(f"\n  Effect Sizes (Cohen's d):")
    effects = gap_analysis.get("effect_sizes", {})
    for dim in ["agency", "perceived_justice", "belonging"]:
        d = effects.get(f"{dim}_cohens_d", 0)
        magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        print(f"    {dim}: d={d:.2f} ({magnitude})")

    euclidean_dist = gap_analysis.get("euclidean_distance", 0)
    print(f"\n  Euclidean Distance: {euclidean_dist:.3f}")
    print(f"  Interpretation: {comparison.get('interpretation', 'N/A')[:100]}...")

    return results

async def run_study_3(client: httpx.AsyncClient) -> Dict:
    """Study 3: Narrative Trajectory Dynamics"""
    print("\n" + "="*80)
    print("STUDY 3: NARRATIVE TRAJECTORY DYNAMICS")
    print("="*80)

    results = {"trajectories": {}}

    for journey_id, journey_data in TRAJECTORY_JOURNEYS.items():
        trajectory = await track_trajectory(
            client,
            journey_data["name"],
            journey_data["points"]
        )

        results["trajectories"][journey_id] = {
            "name": journey_data["name"],
            "trajectory": trajectory,
        }

        print(f"\n  {journey_data['name']}:")
        print(f"    Total Distance: {trajectory.get('total_distance', 0):.3f}")
        print(f"    Net Displacement: {trajectory.get('net_displacement', 0):.3f}")
        print(f"    Mode Transitions: {trajectory.get('mode_transitions', 0)}")

        trends = trajectory.get("dimension_trends", {})
        for dim in ["agency", "perceived_justice", "belonging"]:
            trend_info = trends.get(dim, {})
            direction = trend_info.get("trend", "?")
            r = trend_info.get("correlation", 0)
            print(f"    {dim}: {direction} (r={r:.2f})")

        # Show mode sequence
        points = trajectory.get("points", [])
        modes = [p.get("mode", "?") for p in points]
        print(f"    Mode sequence: {' → '.join(modes)}")

    return results

async def run_study_4(client: httpx.AsyncClient) -> Dict:
    """Study 4: Cross-Cohort ANOVA Analysis"""
    print("\n" + "="*80)
    print("STUDY 4: CROSS-COHORT ANOVA ANALYSIS")
    print("="*80)

    cohort_result = await cohort_analysis(client, DEPARTMENT_NARRATIVES)

    results = {
        "cohort_analysis": cohort_result,
        "summary_stats": {},
    }

    # API returns cohort_profiles with mean_agency, etc.
    cohort_profiles = cohort_result.get("cohort_profiles", cohort_result.get("cohort_stats", {}))

    print(f"\n  Cohort Centroids:")
    for dept, stats in cohort_profiles.items():
        a = stats.get("mean_agency", stats.get("centroid", {}).get("agency", 0))
        pj = stats.get("mean_perceived_justice", stats.get("centroid", {}).get("perceived_justice", 0))
        b = stats.get("mean_belonging", stats.get("centroid", {}).get("belonging", 0))
        mode = stats.get("dominant_mode", "?")
        print(f"    {dept}: A={a:.2f}, PJ={pj:.2f}, B={b:.2f} [{mode}]")
        results["summary_stats"][dept] = {"agency": a, "perceived_justice": pj, "belonging": b, "mode": mode}

    print(f"\n  ANOVA Results:")
    anova = cohort_result.get("anova_results", {})
    for dim in ["agency", "perceived_justice", "belonging"]:
        dim_result = anova.get(dim, {})
        f_stat = dim_result.get("f_stat", dim_result.get("f_statistic", 0))
        p_val = dim_result.get("p_value", 1)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"    {dim}: F={f_stat:.2f}, p={p_val:.4f} {sig}")

    # Pairwise comparisons
    pairwise = cohort_result.get("pairwise_comparisons", [])
    if pairwise:
        print(f"\n  Significant Pairwise Differences:")
        sig_pairs = [p for p in pairwise if p.get("p_value", 1) < 0.05]
        for pair in sig_pairs[:5]:
            print(f"    {pair.get('group_a')} vs {pair.get('group_b')}: d={pair.get('cohens_d', 0):.2f} on {pair.get('dimension')}")

    return results

async def run_study_5(client: httpx.AsyncClient) -> Dict:
    """Study 5: Outlier Detection & Authenticity Markers"""
    print("\n" + "="*80)
    print("STUDY 5: OUTLIER DETECTION & AUTHENTICITY MARKERS")
    print("="*80)

    texts = [case["text"] for case in OUTLIER_TEST_CASES]
    outlier_result = await detect_outliers(client, texts, threshold=2.5)

    results = {
        "outlier_analysis": outlier_result,
        "detection_accuracy": {"correct": 0, "total": 0},
        "cases": [],
    }

    outlier_texts = outlier_result.get("outliers", [])
    outlier_text_set = {o.get("text", "") for o in outlier_texts}

    print(f"\n  Detection Results:")
    for case in OUTLIER_TEST_CASES:
        text = case["text"]
        expected = case["expected"]
        detected = "outlier" if text in outlier_text_set else "normal"
        correct = (expected == detected)

        results["detection_accuracy"]["total"] += 1
        if correct:
            results["detection_accuracy"]["correct"] += 1

        status = "✓" if correct else "✗"
        short_text = text[:50] + "..." if len(text) > 50 else text
        print(f"    {status} [{expected}→{detected}] \"{short_text}\"")

        results["cases"].append({
            "text": text,
            "expected": expected,
            "detected": detected,
            "correct": correct,
        })

    accuracy = results["detection_accuracy"]["correct"] / results["detection_accuracy"]["total"]
    print(f"\n  Detection Accuracy: {results['detection_accuracy']['correct']}/{results['detection_accuracy']['total']} ({accuracy:.0%})")
    print(f"  Total Outliers Found: {len(outlier_texts)}")

    # Show Mahalanobis distances for outliers
    print(f"\n  Outlier Details:")
    for outlier in outlier_texts[:5]:
        dist = outlier.get("mahalanobis_distance", 0)
        mode = outlier.get("mode", "?")
        print(f"    M={dist:.2f} [{mode}]: {outlier.get('text', '')[:60]}...")

    results["detection_accuracy"]["rate"] = accuracy
    return results

async def run_study_6(client: httpx.AsyncClient) -> Dict:
    """Study 6: Ensemble Uncertainty Quantification"""
    print("\n" + "="*80)
    print("STUDY 6: ENSEMBLE UNCERTAINTY QUANTIFICATION")
    print("="*80)

    results = {"analyses": [], "uncertainty_patterns": {}}

    # First, ensure we're using ensemble projection
    await client.post(f"{BASE_URL}/api/projection/mode", json={"mode": "ensemble"})

    clear_texts = UNCERTAINTY_TEST_TEXTS[:3]
    ambiguous_texts = UNCERTAINTY_TEST_TEXTS[3:6]
    mixed_texts = UNCERTAINTY_TEST_TEXTS[6:]

    async def analyze_with_uncertainty(texts: List[str], category: str):
        category_results = []
        for text in texts:
            response = await client.post(f"{BASE_URL}/v2/analyze", json={
                "text": text,
                "include_uncertainty": True,
                "include_soft_labels": True,
            })
            data = response.json()

            mode_data = data.get("mode", {})
            if isinstance(mode_data, dict):
                mode_str = mode_data.get("primary_mode", "UNKNOWN")
                confidence = mode_data.get("confidence", 0)
            else:
                mode_str = mode_data
                confidence = data.get("confidence", 0)

            uncertainty = data.get("uncertainty", {})
            ci = uncertainty.get("confidence_intervals", {})

            category_results.append({
                "text": text[:60] + "...",
                "mode": mode_str,
                "confidence": confidence,
                "uncertainty_method": uncertainty.get("method", "unknown"),
                "agency_ci": ci.get("agency", [0, 0]),
                "perceived_justice_ci": ci.get("perceived_justice", [0, 0]),
                "belonging_ci": ci.get("belonging", [0, 0]),
            })
        return category_results

    results["analyses"] = {
        "clear": await analyze_with_uncertainty(clear_texts, "clear"),
        "ambiguous": await analyze_with_uncertainty(ambiguous_texts, "ambiguous"),
        "mixed": await analyze_with_uncertainty(mixed_texts, "mixed"),
    }

    # Calculate average CI widths per category
    for category, analyses in results["analyses"].items():
        ci_widths = []
        for a in analyses:
            for dim in ["agency", "perceived_justice", "belonging"]:
                ci = a.get(f"{dim}_ci", [0, 0])
                if isinstance(ci, list) and len(ci) == 2:
                    ci_widths.append(ci[1] - ci[0])
        avg_width = statistics.mean(ci_widths) if ci_widths else 0
        results["uncertainty_patterns"][category] = {
            "avg_ci_width": avg_width,
            "avg_confidence": statistics.mean([a.get("confidence", 0) for a in analyses]),
        }

        print(f"\n  {category.upper()} Texts:")
        print(f"    Avg CI Width: {avg_width:.3f}")
        print(f"    Avg Confidence: {results['uncertainty_patterns'][category]['avg_confidence']:.2f}")
        for a in analyses:
            print(f"      [{a['mode']}] conf={a['confidence']:.2f}: {a['text']}")

    return results

async def run_study_7(client: httpx.AsyncClient) -> Dict:
    """Study 7: Mode Flow State Transitions"""
    print("\n" + "="*80)
    print("STUDY 7: MODE FLOW STATE TRANSITIONS")
    print("="*80)

    results = {"flows": {}, "transition_matrix": defaultdict(lambda: defaultdict(int))}

    for flow_id, texts in MODE_FLOW_SEQUENCES.items():
        flow_result = await analyze_mode_flow(client, texts)
        results["flows"][flow_id] = flow_result

        # Build transition counts
        sequence = flow_result.get("sequence", [])
        for i in range(len(sequence) - 1):
            from_mode = sequence[i]
            to_mode = sequence[i + 1]
            results["transition_matrix"][from_mode][to_mode] += 1

        print(f"\n  {flow_id}:")
        print(f"    Sequence: {' → '.join(sequence)}")
        print(f"    Pattern: {flow_result.get('flow_pattern', '?')}")
        print(f"    Transitions: {flow_result.get('transition_count', 0)}")
        print(f"    Interpretation: {flow_result.get('interpretation', 'N/A')[:80]}...")

    # Print transition summary
    print(f"\n  Most Common Transitions:")
    all_transitions = []
    for from_mode, to_modes in results["transition_matrix"].items():
        for to_mode, count in to_modes.items():
            all_transitions.append((from_mode, to_mode, count))

    all_transitions.sort(key=lambda x: x[2], reverse=True)
    for from_m, to_m, count in all_transitions[:10]:
        print(f"    {from_m} → {to_m}: {count}x")

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    print("="*80)
    print("CULTURAL SOLITON OBSERVATORY - COMPREHENSIVE RESEARCH STUDY")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Verify connection
        try:
            health = await client.get(f"{BASE_URL}/health")
            print(f"\nBackend Status: {health.json().get('status', 'unknown')}")
        except Exception as e:
            print(f"\nERROR: Cannot connect to backend at {BASE_URL}")
            print(f"Start the server first: uvicorn main:app --port 8000")
            return

        all_results = {}

        # Run all studies
        all_results["study_1"] = await run_study_1(client)
        all_results["study_2"] = await run_study_2(client)
        all_results["study_3"] = await run_study_3(client)
        all_results["study_4"] = await run_study_4(client)
        all_results["study_5"] = await run_study_5(client)
        all_results["study_6"] = await run_study_6(client)
        all_results["study_7"] = await run_study_7(client)

        # Save results
        output_file = f"data/research_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert defaultdicts to regular dicts for JSON serialization
        def convert_defaultdicts(obj):
            if isinstance(obj, defaultdict):
                return {k: convert_defaultdicts(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_defaultdicts(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_defaultdicts(item) for item in obj]
            return obj

        serializable_results = convert_defaultdicts(all_results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("STUDY SUMMARY")
        print("="*80)
        print(f"\n  Study 1 - Classification Accuracy: {all_results['study_1']['overall_accuracy']:.1%}")

        # Safely get max hypocrisy gap
        comparison = all_results['study_2']['comparison']
        gap_analysis = comparison.get('gap_analysis', comparison)
        axis_gaps = gap_analysis.get('axis_gaps', gap_analysis.get('dimension_gaps', {}))
        if axis_gaps:
            max_gap = max(abs(g.get('gap', 0)) for g in axis_gaps.values())
        else:
            max_gap = gap_analysis.get('euclidean_distance', 0)
        print(f"  Study 2 - Max Hypocrisy Gap: {max_gap:.3f}")

        print(f"  Study 3 - Trajectories Analyzed: {len(all_results['study_3']['trajectories'])}")
        print(f"  Study 4 - Cohorts Compared: {len(all_results['study_4']['summary_stats'])}")
        print(f"  Study 5 - Outlier Detection: {all_results['study_5']['detection_accuracy']['rate']:.0%}")
        print(f"  Study 6 - Uncertainty Analysis: {len(UNCERTAINTY_TEST_TEXTS)} texts")
        print(f"  Study 7 - Flow Patterns: {len(all_results['study_7']['flows'])}")
        print(f"\n  Results saved to: {output_file}")

        return all_results

if __name__ == "__main__":
    asyncio.run(main())
