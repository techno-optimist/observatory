#!/usr/bin/env python3
"""
The Hope Effect - Case Study Analysis
=====================================
Comprehensive analysis of The Hope Effect nonprofit's public communications
through the Cultural Soliton Observatory.

Categories analyzed:
1. Mission & Vision Statements
2. Problem Framing (The Crisis)
3. Solution Framing (The Approach)
4. Impact Stories (Children)
5. Organizational Voice
6. Donor Appeals
7. Research/Statistics Framing
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import statistics

BASE_URL = "http://127.0.0.1:8000"

# =============================================================================
# THE HOPE EFFECT - NARRATIVE CORPUS
# =============================================================================

NARRATIVES = {
    # Category 1: Mission & Vision Statements
    "mission_vision": [
        "We are changing the way the world cares for orphans…because every child deserves a family.",
        "Getting kids out of orphanages and into loving families…where they belong.",
        "Kids do best when they are raised in families.",
        "Together, we can change the way the world cares for orphans…because every child deserves a family.",
        "The Hope Effect is changing how the world cares for orphans by providing family-based solutions.",
    ],

    # Category 2: Problem Framing - The Crisis
    "problem_framing": [
        "There are millions of orphaned and vulnerable children worldwide, with an estimated 2.7 million of them living in orphanages.",
        "Children in institutional settings are 10x more likely to fall into sex work.",
        "Children raised in institutional orphanages are 40x more likely to have a criminal record.",
        "Children aging out of orphanages are 500x more likely to commit suicide compared to peers raised in family environments.",
        "Research dating back to the 1930s demonstrates that institutional orphanage care causes lasting harm. Children in these settings lack personalized attention and family connection.",
        "Institutional care can have deeply harmful and lasting effects on a child's well-being—studies have shown unequivocally that institutional care can be incredibly hard on a child.",
        "Less than 1% of orphans globally are adopted.",
        "Approximately 80% of orphanage children have family members who could provide care.",
    ],

    # Category 3: Solution Framing - The Approach
    "solution_framing": [
        "We build smaller family-style homes with two parent figures and 6-8 children each. This model provides individual attention, stability, and security to help children flourish.",
        "Rather than constructing large, institutional-style orphanages with high child to adult ratios, we build smaller homes in a campus setting that feature two parents and six children, providing kids with more attention and affection.",
        "We establish foster care systems where they don't exist by changing primary care models from institutional to family-based.",
        "We facilitate reunification with biological families. Approximately 80% of orphanage children have family members who could provide care.",
        "On average, family care costs 5-8 times less than institutional care.",
        "Our work is a powerful solution both in Mexico and around the world that benefits the child, the government, and the entire community.",
        "We pioneer, partner, and promote family-centered orphan care.",
    ],

    # Category 4: Impact Stories - Children
    "impact_stories": [
        "At just a year old, Verónica was abandoned by her biological family and placed into an orphanage, where she remained for the next few years without the love of a family. After joining a foster family through our program, she began reaching milestones, making developmental gains that institutional care couldn't provide.",
        "Rosa came to the DIF facility shortly after she was born. She's now almost four years old and just started preschool. For a while, Rosa had some extended family – her aunt – visiting her a few times a year.",
        "A child abandoned after her mother's death, Luisa experienced neglect and homelessness in Sonora, Mexico. Placed initially in institutional care, she was subsequently moved into family-style care where she began to thrive. She became more playful and talkative as she bonded with her foster family and experienced safety and love.",
        "Surrounded by the love of a family, children like Alberto, Daniela, and Ernesto demonstrate developmental improvements when transitioning from orphanages to family care.",
        "Stories of lives changed through the love of a family.",
    ],

    # Category 5: Organizational Voice / Identity
    "organizational_voice": [
        "The Hope Effect was the first organization to seek government approval for family-based care in the State of Sonora, Mexico.",
        "Since launching, we raised over $300,000 for orphan care, built a family-style home for orphans in Siguatepeque, Honduras, and began buying land in San Luis Río Colorado, Mexico for our first full orphan care campus.",
        "We currently work in eight cities around the world across four nations: Mexico, Honduras, Thailand, and Cambodia.",
        "We have been in Chiang Mai, Thailand, since October 2021, and expanded into Battambang, Cambodia, in October 2022.",
        "We have an exciting opportunity in Mexico to speak about the work we are doing to expand family-care for orphaned children to every state government in Mexico.",
        "When author Joshua Becker secured a generous book contract writing about minimalism, he and his wife knew that they would not spend their advance buying things they didn't need. So instead, they decided to use those funds to start a non-profit that would impact an area close to their heart: orphan care.",
        "Kim was adopted as a newborn after being left at a South Dakota hospital. She understands the importance of family-based care.",
    ],

    # Category 6: Donor Appeals
    "donor_appeals": [
        "100% of your donation can go directly to help a child.",
        "Monthly giving. Maximum impact.",
        "Every penny donated is used directly for orphan care.",
        "Donations start at one dollar daily.",
        "On Giving Tuesday, matched donations doubled donor impact.",
        "Your contribution is tax-deductible and 100% directed toward orphan care.",
    ],

    # Category 7: Advocacy / Call to Action
    "advocacy": [
        "U.S. Christians donate $2.5 billion yearly to orphanages, while research confirms family care is superior.",
        "We advocate for systemic change by promoting family-centered orphan care to religious institutions and broader communities.",
        "We encourage U.S. foster care involvement through respite care, donations, family support, advocacy, and becoming foster parents through local agencies.",
        "Revolutionize. Reunite. Redefine.",
    ],
}

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

async def analyze_text(client: httpx.AsyncClient, text: str) -> Dict:
    """Full v2 analysis of a single text."""
    response = await client.post(f"{BASE_URL}/v2/analyze", json={
        "text": text,
        "include_soft_labels": True,
        "include_uncertainty": True,
    })
    return response.json()

async def compare_groups(client: httpx.AsyncClient, group_a: List[str], group_b: List[str],
                         name_a: str, name_b: str) -> Dict:
    """Compare two narrative groups."""
    response = await client.post(f"{BASE_URL}/api/v2/compare", json={
        "group_a": {"name": name_a, "texts": group_a},
        "group_b": {"name": name_b, "texts": group_b},
    })
    return response.json()

async def analyze_corpus(client: httpx.AsyncClient, texts: List[str]) -> Dict:
    """Analyze a corpus of texts."""
    response = await client.post(f"{BASE_URL}/corpus/analyze", json={
        "texts": texts,
        "find_clusters": True,
    })
    return response.json()

async def analyze_mode_flow(client: httpx.AsyncClient, texts: List[str]) -> Dict:
    """Analyze mode flow in a sequence."""
    response = await client.post(f"{BASE_URL}/api/v2/analytics/mode-flow", json={
        "texts": texts,
    })
    return response.json()

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

async def main():
    print("="*80)
    print("THE HOPE EFFECT - CULTURAL SOLITON OBSERVATORY CASE STUDY")
    print(f"Analysis Date: {datetime.now().isoformat()}")
    print("="*80)

    results = {
        "organization": "The Hope Effect",
        "analysis_date": datetime.now().isoformat(),
        "categories": {},
        "aggregate": {},
        "comparisons": {},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Verify connection
        try:
            health = await client.get(f"{BASE_URL}/health")
            print(f"\nBackend Status: {health.json().get('status', 'connected')}")
        except Exception as e:
            print(f"\nERROR: Cannot connect to backend at {BASE_URL}")
            return

        # =================================================================
        # ANALYZE EACH CATEGORY
        # =================================================================
        all_texts = []
        all_projections = []

        for category, texts in NARRATIVES.items():
            print(f"\n{'='*60}")
            print(f"CATEGORY: {category.upper().replace('_', ' ')}")
            print(f"{'='*60}")

            category_results = {
                "texts": [],
                "modes": defaultdict(int),
                "coordinates": {"agency": [], "perceived_justice": [], "belonging": []},
            }

            for text in texts:
                analysis = await analyze_text(client, text)

                # Extract mode info
                mode_data = analysis.get("mode", {})
                if isinstance(mode_data, dict):
                    mode = mode_data.get("primary_mode", "UNKNOWN")
                    confidence = mode_data.get("confidence", 0)
                    secondary = mode_data.get("secondary_mode", "")
                else:
                    mode = mode_data
                    confidence = 0
                    secondary = ""

                # Extract coordinates
                vector = analysis.get("vector", {})
                agency = vector.get("agency", 0)
                pj = vector.get("perceived_justice", 0)
                belonging = vector.get("belonging", 0)

                category_results["texts"].append({
                    "text": text[:80] + "..." if len(text) > 80 else text,
                    "mode": mode,
                    "confidence": confidence,
                    "secondary_mode": secondary,
                    "agency": agency,
                    "perceived_justice": pj,
                    "belonging": belonging,
                })

                category_results["modes"][mode] += 1
                category_results["coordinates"]["agency"].append(agency)
                category_results["coordinates"]["perceived_justice"].append(pj)
                category_results["coordinates"]["belonging"].append(belonging)

                all_texts.append(text)
                all_projections.append({"agency": agency, "perceived_justice": pj, "belonging": belonging, "mode": mode})

                # Print individual result
                short_text = text[:50] + "..." if len(text) > 50 else text
                print(f"  [{mode}] A={agency:.2f} PJ={pj:.2f} B={belonging:.2f}")
                print(f"    \"{short_text}\"")

            # Calculate category statistics
            category_results["statistics"] = {
                "n": len(texts),
                "mean_agency": statistics.mean(category_results["coordinates"]["agency"]),
                "mean_perceived_justice": statistics.mean(category_results["coordinates"]["perceived_justice"]),
                "mean_belonging": statistics.mean(category_results["coordinates"]["belonging"]),
                "dominant_mode": max(category_results["modes"], key=category_results["modes"].get),
                "mode_distribution": dict(category_results["modes"]),
            }

            print(f"\n  Category Summary:")
            print(f"    Mean A={category_results['statistics']['mean_agency']:.3f}")
            print(f"    Mean PJ={category_results['statistics']['mean_perceived_justice']:.3f}")
            print(f"    Mean B={category_results['statistics']['mean_belonging']:.3f}")
            print(f"    Dominant Mode: {category_results['statistics']['dominant_mode']}")

            results["categories"][category] = category_results

        # =================================================================
        # AGGREGATE ANALYSIS
        # =================================================================
        print(f"\n{'='*60}")
        print("AGGREGATE ANALYSIS - ALL NARRATIVES")
        print(f"{'='*60}")

        all_agency = [p["agency"] for p in all_projections]
        all_pj = [p["perceived_justice"] for p in all_projections]
        all_belonging = [p["belonging"] for p in all_projections]
        all_modes = [p["mode"] for p in all_projections]

        mode_counts = defaultdict(int)
        for m in all_modes:
            mode_counts[m] += 1

        results["aggregate"] = {
            "total_narratives": len(all_texts),
            "mean_agency": statistics.mean(all_agency),
            "mean_perceived_justice": statistics.mean(all_pj),
            "mean_belonging": statistics.mean(all_belonging),
            "std_agency": statistics.stdev(all_agency) if len(all_agency) > 1 else 0,
            "std_perceived_justice": statistics.stdev(all_pj) if len(all_pj) > 1 else 0,
            "std_belonging": statistics.stdev(all_belonging) if len(all_belonging) > 1 else 0,
            "mode_distribution": dict(mode_counts),
            "dominant_mode": max(mode_counts, key=mode_counts.get),
        }

        print(f"\n  Total Narratives: {results['aggregate']['total_narratives']}")
        print(f"\n  Centroid:")
        print(f"    Agency: {results['aggregate']['mean_agency']:.3f} (±{results['aggregate']['std_agency']:.3f})")
        print(f"    Perceived Justice: {results['aggregate']['mean_perceived_justice']:.3f} (±{results['aggregate']['std_perceived_justice']:.3f})")
        print(f"    Belonging: {results['aggregate']['mean_belonging']:.3f} (±{results['aggregate']['std_belonging']:.3f})")
        print(f"\n  Mode Distribution:")
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            pct = count / len(all_modes) * 100
            print(f"    {mode}: {count} ({pct:.1f}%)")

        # =================================================================
        # CROSS-CATEGORY COMPARISONS
        # =================================================================
        print(f"\n{'='*60}")
        print("CROSS-CATEGORY COMPARISONS")
        print(f"{'='*60}")

        # Compare Problem Framing vs Solution Framing
        comparison1 = await compare_groups(
            client,
            NARRATIVES["problem_framing"],
            NARRATIVES["solution_framing"],
            "Problem Framing",
            "Solution Framing"
        )
        results["comparisons"]["problem_vs_solution"] = comparison1

        gap_analysis = comparison1.get("gap_analysis", comparison1)
        axis_gaps = gap_analysis.get("axis_gaps", {})

        print(f"\n  Problem Framing vs Solution Framing:")
        print(f"    Euclidean Distance: {gap_analysis.get('euclidean_distance', 0):.3f}")
        for dim in ["agency", "perceived_justice", "belonging"]:
            gap = axis_gaps.get(dim, {}).get("gap", 0)
            direction = axis_gaps.get(dim, {}).get("direction", "?")
            print(f"    {dim}: gap={gap:.3f} ({direction})")

        # Compare Mission vs Impact Stories
        comparison2 = await compare_groups(
            client,
            NARRATIVES["mission_vision"],
            NARRATIVES["impact_stories"],
            "Mission/Vision",
            "Impact Stories"
        )
        results["comparisons"]["mission_vs_stories"] = comparison2

        gap_analysis2 = comparison2.get("gap_analysis", comparison2)
        axis_gaps2 = gap_analysis2.get("axis_gaps", {})

        print(f"\n  Mission/Vision vs Impact Stories:")
        print(f"    Euclidean Distance: {gap_analysis2.get('euclidean_distance', 0):.3f}")
        for dim in ["agency", "perceived_justice", "belonging"]:
            gap = axis_gaps2.get(dim, {}).get("gap", 0)
            direction = axis_gaps2.get(dim, {}).get("direction", "?")
            print(f"    {dim}: gap={gap:.3f} ({direction})")

        # Compare Donor Appeals vs Organizational Voice
        comparison3 = await compare_groups(
            client,
            NARRATIVES["donor_appeals"],
            NARRATIVES["organizational_voice"],
            "Donor Appeals",
            "Organizational Voice"
        )
        results["comparisons"]["appeals_vs_voice"] = comparison3

        gap_analysis3 = comparison3.get("gap_analysis", comparison3)
        axis_gaps3 = gap_analysis3.get("axis_gaps", {})

        print(f"\n  Donor Appeals vs Organizational Voice:")
        print(f"    Euclidean Distance: {gap_analysis3.get('euclidean_distance', 0):.3f}")
        for dim in ["agency", "perceived_justice", "belonging"]:
            gap = axis_gaps3.get(dim, {}).get("gap", 0)
            direction = axis_gaps3.get(dim, {}).get("direction", "?")
            print(f"    {dim}: gap={gap:.3f} ({direction})")

        # =================================================================
        # MODE FLOW ANALYSIS
        # =================================================================
        print(f"\n{'='*60}")
        print("NARRATIVE JOURNEY ANALYSIS")
        print(f"{'='*60}")

        # Construct a narrative journey from problem -> solution -> impact
        journey_texts = [
            NARRATIVES["problem_framing"][0],  # The crisis
            NARRATIVES["problem_framing"][3],  # Statistics
            NARRATIVES["solution_framing"][0],  # The solution
            NARRATIVES["impact_stories"][0],    # A success story
            NARRATIVES["mission_vision"][0],    # The mission
        ]

        flow_result = await analyze_mode_flow(client, journey_texts)
        results["narrative_journey"] = flow_result

        print(f"\n  Narrative Journey (Problem → Solution → Impact → Mission):")
        sequence = flow_result.get("sequence", [])
        print(f"    Sequence: {' → '.join(sequence)}")
        print(f"    Pattern: {flow_result.get('flow_pattern', 'unknown')}")
        print(f"    Transitions: {flow_result.get('transition_count', 0)}")

        # =================================================================
        # SAVE RESULTS
        # =================================================================
        output_file = f"data/hope_effect_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert defaultdicts
        def convert_defaultdicts(obj):
            if isinstance(obj, defaultdict):
                return {k: convert_defaultdicts(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_defaultdicts(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_defaultdicts(item) for item in obj]
            return obj

        with open(output_file, "w") as f:
            json.dump(convert_defaultdicts(results), f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"\n  Results saved to: {output_file}")

        return results

if __name__ == "__main__":
    asyncio.run(main())
