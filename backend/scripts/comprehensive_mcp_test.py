#!/usr/bin/env python3
"""
Comprehensive MCP Server Test Suite

Tests ALL observatory capabilities through the MCP client interface,
simulating how an LLM would use these tools in a research session.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mcp-server"))

from observatory_client import ObservatoryClient, ObservatoryConfig

# Test data representing real-world research scenarios
CORPORATE_OFFICIAL = [
    "We are committed to creating an inclusive workplace where every voice matters.",
    "Our employees are our greatest asset and we invest heavily in their growth.",
    "Sustainability and ethical business practices are at the core of everything we do.",
    "We foster a culture of innovation, collaboration, and mutual respect.",
]

EMPLOYEE_REALITY = [
    "Management says they care but layoffs happen without warning.",
    "The diversity initiatives are just for PR, nothing actually changes.",
    "Burnout is normalized here. If you leave at 5pm, you're not a team player.",
    "Ideas from junior employees get ignored unless a VP champions them.",
]

HISTORICAL_TEXTS = [
    ("1950-01-01", "The company provides stable employment and a pension for life."),
    ("1970-01-01", "We're building a new kind of organization focused on growth."),
    ("1990-01-01", "Shareholder value is our primary responsibility."),
    ("2010-01-01", "We're disrupting the industry with innovative solutions."),
    ("2020-01-01", "We're navigating unprecedented challenges in a changing world."),
    ("2025-01-01", "We're reimagining work for a hybrid, AI-augmented future."),
]

DEPARTMENT_COHORTS = {
    "engineering": [
        "We build things that work. Ship it and iterate.",
        "Technical debt is crushing us but leadership won't prioritize it.",
        "The best engineers are leaving for startups with better equity.",
    ],
    "sales": [
        "We're crushing our targets this quarter!",
        "Quota keeps going up but territory keeps shrinking.",
        "Commission structure changed again, always in their favor.",
    ],
    "hr": [
        "We're here to support our people through change.",
        "Employee satisfaction scores are our top priority.",
        "We facilitate difficult conversations with empathy.",
    ],
    "executives": [
        "Our strategic vision positions us for long-term success.",
        "We're making tough decisions to ensure shareholder returns.",
        "The transformation is on track and delivering results.",
    ],
}

RADICALIZATION_SEQUENCE = [
    "I love my job and believe in what we're building here.",
    "Some things aren't going well but I'm sure leadership has a plan.",
    "Why do they keep making decisions that hurt us?",
    "The system is designed to extract value from workers.",
    "We need to organize and demand change collectively.",
]

NORMAL_CORPUS = [
    "Another day at work, doing my best.",
    "Some wins and some losses, that's life.",
    "I appreciate my colleagues even when we disagree.",
    "The company isn't perfect but no place is.",
]

OUTLIER_TEXTS = [
    "Everything is AMAZING and PERFECT and I've never been HAPPIER!!!",  # Suspiciously positive
    "They're poisoning the water supply and tracking us through vaccines.",  # Paranoid
    "I'm just here for the paycheck, don't care about any of it.",  # Disengaged
]


async def run_comprehensive_tests():
    """Run all MCP tool tests."""

    print("=" * 80)
    print("CULTURAL SOLITON OBSERVATORY - COMPREHENSIVE MCP TEST SUITE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    config = ObservatoryConfig(backend_url="http://127.0.0.1:8000")

    async with ObservatoryClient(config) as client:
        results = {}

        # ====================================================================
        # TEST 1: Basic Projection (Core Tool)
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 1: Basic Text Projection")
        print("=" * 60)

        try:
            result = await client.project_text(
                "I built this company from nothing through sheer determination.",
                use_v2=True  # Use v2 API with soft labels
            )
            print(f"✓ Text projected successfully")
            print(f"  Agency: {result.agency:.3f}")
            print(f"  Perceived Justice: {result.perceived_justice:.3f}")
            print(f"  Belonging: {result.belonging:.3f}")
            print(f"  Mode: {result.mode}")
            results["basic_projection"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["basic_projection"] = False

        # ====================================================================
        # TEST 2: Soft Labels (12-Mode Distribution)
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 2: Soft Labels (12-Mode Probability Distribution)")
        print("=" * 60)

        try:
            result = await client.get_soft_labels(
                "I'm conflicted - I love the work but hate the politics."
            )
            print(f"✓ Soft labels retrieved")
            print(f"  Primary Mode: {result.get('primary_mode')} ({result.get('primary_probability', 0):.1%})")
            print(f"  Secondary Mode: {result.get('secondary_mode')} ({result.get('secondary_probability', 0):.1%})")
            print(f"  Stability Score: {result.get('stability_score', 0):.2f}")
            print(f"  Boundary Case: {result.get('is_boundary_case', False)}")
            results["soft_labels"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["soft_labels"] = False

        # ====================================================================
        # TEST 3: Uncertainty Quantification
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 3: Uncertainty Quantification (Ensemble)")
        print("=" * 60)

        try:
            result = await client.analyze_with_uncertainty(
                "Things might get better, or they might not. Who knows."
            )
            print(f"✓ Uncertainty analysis completed")
            vector = result.get("vector", {})
            uncertainty = result.get("uncertainty", {})
            print(f"  Coordinates: A={vector.get('agency', 0):.3f}, PJ={vector.get('perceived_justice', 0):.3f}, B={vector.get('belonging', 0):.3f}")
            print(f"  Method: {uncertainty.get('method', 'N/A')}")
            if 'confidence_intervals' in uncertainty:
                print(f"  Has confidence intervals: Yes")
            results["uncertainty"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["uncertainty"] = False

        # ====================================================================
        # TEST 4: Narrative Comparison (The Hypocrisy Gap)
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 4: Narrative Comparison (Corporate vs Employee)")
        print("=" * 60)

        try:
            result = await client.compare_narratives(
                group_a_name="Corporate Official",
                group_a_texts=CORPORATE_OFFICIAL,
                group_b_name="Employee Reality",
                group_b_texts=EMPLOYEE_REALITY
            )
            gap = result.get("gap_analysis", {})
            print(f"✓ Comparison completed")
            print(f"  Euclidean Distance: {gap.get('euclidean_distance', 0):.3f}")

            axis_gaps = gap.get("axis_gaps", {})
            for axis, data in axis_gaps.items():
                print(f"  {axis}: gap={data.get('gap', 0):.3f} ({data.get('direction', 'N/A')})")

            effect_sizes = gap.get("effect_sizes", {})
            print(f"  Effect Sizes:")
            for key, val in effect_sizes.items():
                print(f"    {key}: {val:.2f}")

            print(f"  Interpretation: {result.get('interpretation', 'N/A')[:80]}...")
            results["comparison"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["comparison"] = False

        # ====================================================================
        # TEST 5: Trajectory Tracking
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 5: Trajectory Tracking (Historical Evolution)")
        print("=" * 60)

        try:
            result = await client.track_trajectory(
                name="Corporate Narrative Evolution",
                texts=[t[1] for t in HISTORICAL_TEXTS],
                timestamps=[t[0] for t in HISTORICAL_TEXTS]
            )
            traj = result.get("trajectory", {})
            print(f"✓ Trajectory tracked")
            print(f"  Total Distance: {traj.get('total_distance_traveled', 0):.3f}")
            print(f"  Net Displacement: {traj.get('net_displacement', 0):.3f}")

            trends = result.get("trends", {})
            for axis, trend in trends.items():
                print(f"  {axis} trend: {trend.get('direction', 'N/A')} (r={trend.get('correlation', 0):.2f})")

            print(f"  Mode Transitions: {len(result.get('mode_transitions', []))}")
            print(f"  Interpretation: {result.get('interpretation', 'N/A')[:80]}...")
            results["trajectory"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["trajectory"] = False

        # ====================================================================
        # TEST 6: Cohort Analysis (ANOVA)
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 6: Cohort Analysis (Multi-Department ANOVA)")
        print("=" * 60)

        try:
            result = await client.analyze_cohorts(cohorts=DEPARTMENT_COHORTS)
            profiles = result.get("cohort_profiles", {})
            print(f"✓ Cohort analysis completed")

            for cohort, profile in profiles.items():
                print(f"  {cohort}: A={profile.get('mean_agency', 0):.2f}, PJ={profile.get('mean_perceived_justice', 0):.2f}, B={profile.get('mean_belonging', 0):.2f}")

            anova = result.get("anova_results", {})
            print(f"\n  ANOVA Results:")
            for axis, stats in anova.items():
                sig = "***" if stats.get('p_value', 1) < 0.001 else "**" if stats.get('p_value', 1) < 0.01 else "*" if stats.get('p_value', 1) < 0.05 else ""
                print(f"    {axis}: F={stats.get('f_stat', 0):.2f}, p={stats.get('p_value', 1):.4f} {sig}")

            results["cohort_analysis"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["cohort_analysis"] = False

        # ====================================================================
        # TEST 7: Mode Flow Analysis (Radicalization Detection)
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 7: Mode Flow Analysis (Radicalization Pattern)")
        print("=" * 60)

        try:
            result = await client.analyze_mode_flow(texts=RADICALIZATION_SEQUENCE)
            print(f"✓ Mode flow analyzed")
            print(f"  Sequence: {' -> '.join(result.get('sequence', []))}")
            print(f"  Flow Pattern: {result.get('flow_pattern', 'N/A')}")
            print(f"  Transitions: {len(result.get('transitions', []))}")
            print(f"  Interpretation: {result.get('interpretation', 'N/A')[:80]}...")
            results["mode_flow"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["mode_flow"] = False

        # ====================================================================
        # TEST 8: Outlier Detection
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 8: Outlier Detection")
        print("=" * 60)

        try:
            for i, outlier in enumerate(OUTLIER_TEXTS):
                result = await client.detect_outliers(
                    corpus=NORMAL_CORPUS,
                    test_text=outlier
                )
                is_outlier = result.get("is_outlier", False)
                score = result.get("mahalanobis_distance", 0)
                status = "OUTLIER" if is_outlier else "NORMAL"
                print(f"  Text {i+1}: {status} (Mahalanobis={score:.2f})")
                print(f"    \"{outlier[:50]}...\"")

            results["outlier_detection"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["outlier_detection"] = False

        # ====================================================================
        # TEST 9: Alert System
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 9: Alert System")
        print("=" * 60)

        try:
            # Create an alert
            alert = await client.create_alert(
                name="Hypocrisy Gap Alert",
                alert_type="gap_threshold",
                config={
                    "axis": "perceived_justice",
                    "threshold": 0.5,
                    "direction": "exceeds"
                }
            )
            print(f"✓ Alert created: {alert.get('name')} (ID: {alert.get('id', 'N/A')[:8]}...)")

            # Check against texts
            check_result = await client.check_alerts(
                group_a_texts=CORPORATE_OFFICIAL,
                group_b_texts=EMPLOYEE_REALITY
            )
            triggered = check_result.get("triggered_alerts", [])
            print(f"  Alerts triggered: {len(triggered)}")
            for t in triggered:
                print(f"    - {t.get('alert_name')}: {t.get('message', 'N/A')[:60]}...")

            results["alert_system"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["alert_system"] = False

        # ====================================================================
        # TEST 10: Projection Mode Management
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 10: Projection Mode Management")
        print("=" * 60)

        try:
            # List available modes
            modes_result = await client.list_projection_modes()
            modes = modes_result.get("modes", [])
            print(f"✓ Available projection modes: {len(modes)}")
            for mode in modes:
                status = "✓" if mode.get("is_available", False) else "✗"
                print(f"  [{status}] {mode.get('name')}: {mode.get('description', 'N/A')[:40]}...")

            # Get current mode
            current = await client.get_current_projection_mode()
            print(f"  Current mode: {current.get('current_mode', 'N/A')}")

            results["projection_modes"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["projection_modes"] = False

        # ====================================================================
        # TEST 11: Corpus Analysis with Clustering
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 11: Corpus Analysis with Clustering")
        print("=" * 60)

        try:
            all_texts = CORPORATE_OFFICIAL + EMPLOYEE_REALITY + NORMAL_CORPUS
            result = await client.analyze_corpus(texts=all_texts, detect_clusters=True)
            print(f"✓ Corpus analyzed: {result.total_texts} texts")
            print(f"  Mode distribution:")
            for mode, count in sorted(result.mode_distribution.items(), key=lambda x: -x[1])[:5]:
                pct = count / result.total_texts * 100
                print(f"    {mode}: {count} ({pct:.1f}%)")

            if result.clusters:
                print(f"  Clusters detected: {len(result.clusters)}")

            results["corpus_analysis"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["corpus_analysis"] = False

        # ====================================================================
        # TEST 12: Hypocrisy Detection (Legacy Tool)
        # ====================================================================
        print("\n" + "=" * 60)
        print("TEST 12: Hypocrisy Detection")
        print("=" * 60)

        try:
            espoused = await client.project_batch(CORPORATE_OFFICIAL)
            operational = await client.project_batch(EMPLOYEE_REALITY)

            # Compute centroids
            def centroid(results):
                n = len(results)
                return {
                    "agency": sum(r.agency for r in results) / n,
                    "perceived_justice": sum(r.perceived_justice for r in results) / n,
                    "belonging": sum(r.belonging for r in results) / n,
                }

            esp = centroid(espoused)
            ops = centroid(operational)

            delta_agency = abs(esp["agency"] - ops["agency"])
            delta_pj = abs(esp["perceived_justice"] - ops["perceived_justice"])
            delta_belonging = abs(esp["belonging"] - ops["belonging"])
            total_delta = (delta_agency + delta_pj + delta_belonging) / 3

            print(f"✓ Hypocrisy analysis completed")
            print(f"  Espoused: A={esp['agency']:.2f}, PJ={esp['perceived_justice']:.2f}, B={esp['belonging']:.2f}")
            print(f"  Operational: A={ops['agency']:.2f}, PJ={ops['perceived_justice']:.2f}, B={ops['belonging']:.2f}")
            print(f"  Total Δw: {total_delta:.3f}")

            if total_delta < 0.3:
                print(f"  Interpretation: Low hypocrisy gap")
            elif total_delta < 0.7:
                print(f"  Interpretation: Moderate hypocrisy gap")
            else:
                print(f"  Interpretation: HIGH HYPOCRISY GAP")

            results["hypocrisy_detection"] = True
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results["hypocrisy_detection"] = False

        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, passed_test in results.items():
            status = "PASS ✓" if passed_test else "FAIL ✗"
            print(f"  {test_name}: {status}")

        print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Observatory is fully operational.")
        else:
            print(f"\n⚠️  {total - passed} tests failed. Check logs above.")

        return results


if __name__ == "__main__":
    results = asyncio.run(run_comprehensive_tests())
