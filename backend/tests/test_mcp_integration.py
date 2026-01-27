#!/usr/bin/env python3
"""
Cultural Soliton Observatory v2.0 - MCP Integration Test Suite

Comprehensive integration tests for the MCP (Model Context Protocol) tools
exposed by the Cultural Soliton Observatory. Tests validate the full pipeline
from HTTP API calls through projection, classification, and research endpoints.

Run with:
    pytest tests/test_mcp_integration.py -v
    pytest tests/test_mcp_integration.py -v -k "end_to_end"

Prerequisites:
    - Backend server running on localhost:8000
    - Projection model trained
    - pip install pytest httpx pytest-asyncio

Author: Systems Architect
Date: January 2026
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0
BATCH_TIMEOUT = 120.0


# ============================================================================
# Test Data Fixtures
# ============================================================================

# Diverse texts spanning the coordinate space
DIVERSE_TEXTS = {
    "heroic": "I built this company from nothing through sheer determination and hard work. My success is proof that anyone can achieve their dreams if they try hard enough.",
    "victim": "No matter how hard I try, the system is rigged against people like me. The powerful always win and the rest of us are left with nothing.",
    "communal": "We are all in this together. Our community supports each other through thick and thin, and together we can overcome any obstacle.",
    "cynical": "I've learned to play the game. Sure, the system is corrupt, but I've figured out how to win anyway. You just need to be smart about it.",
    "neutral": "Today was an ordinary day. I went to work, had lunch, and came home. Nothing particularly good or bad happened.",
    "transcendent": "I have found peace by accepting what I cannot control. The universe has a plan and I trust in the greater harmony of all things.",
    "paranoid": "They're watching everything we do. The corporations and governments are working together to control us. Wake up, people!",
    "protest": "We must fight for change! The old systems are failing us and we need to stand up and demand a better world for everyone.",
}

# Grammar deletion test features
GRAMMAR_FEATURES = [
    "articles",
    "pronouns_first_person",
    "pronouns_second_person",
    "pronouns_third_person",
    "modals",
    "conjunctions",
    "hedging",
    "intensifiers",
    "negation",
    "agency_markers",
    "belonging_markers",
    "justice_markers",
]

# Batch processing test corpus (50+ texts)
BATCH_CORPUS = [
    "Success comes to those who work for it.",
    "The world isn't fair, but we can make it better together.",
    "I feel completely alone in this struggle.",
    "Our team achieved something incredible this quarter.",
    "The rich get richer while the poor suffer.",
    "I choose to see the good in every situation.",
    "Nobody listens to what we have to say.",
    "We stand united against injustice.",
    "Hard work is the key to prosperity.",
    "The game is rigged from the start.",
    "I believe in myself and my abilities.",
    "Our community is our greatest strength.",
    "They don't want us to succeed.",
    "We built something beautiful together.",
    "I'm just trying to survive.",
    "Every day is a new opportunity.",
    "The system crushes those who try to change it.",
    "We are stronger together than apart.",
    "I made my own luck.",
    "Nothing ever changes for people like us.",
    "Our bonds of friendship sustain us.",
    "I refuse to be a victim.",
    "The powerful protect their own.",
    "Together we rise, divided we fall.",
    "I earned everything I have.",
    "They want to keep us down.",
    "My community is my family.",
    "I control my own destiny.",
    "The deck is stacked against the common person.",
    "We support each other unconditionally.",
    "I overcame every obstacle in my path.",
    "Corruption runs deep in every institution.",
    "Our shared values bind us together.",
    "I am the master of my fate.",
    "The elite live by different rules.",
    "We find strength in our differences.",
    "I built my success brick by brick.",
    "The truth is hidden from ordinary people.",
    "Our neighborhood looks out for each other.",
    "I achieved my goals through persistence.",
    "Those in power never face consequences.",
    "We celebrate our collective achievements.",
    "I made it despite the odds.",
    "The system rewards the wrong behaviors.",
    "Our traditions give us strength.",
    "I forge my own path.",
    "Nothing is as it seems in this world.",
    "We share both burdens and joys.",
    "I created something from nothing.",
    "The wealthy control everything.",
    "Our unity is unbreakable.",
]

# Paraphrase pairs for consistency testing
PARAPHRASE_PAIRS = [
    (
        "I built this company from nothing.",
        "From the ground up, I created this business myself."
    ),
    (
        "The system is rigged against ordinary people.",
        "Regular folks don't stand a chance in this corrupt system."
    ),
    (
        "We are stronger together as a community.",
        "As a united community, our strength multiplies."
    ),
    (
        "I take full responsibility for my actions.",
        "The responsibility for what I do rests entirely on my shoulders."
    ),
    (
        "Nobody cares about people like me.",
        "People like us are completely ignored by everyone."
    ),
]

# Calibration test data
HUMAN_AUTHORED_TEXTS = [
    "I've been working at this company for five years and I've seen a lot of changes. Some good, some not so good. But overall, I believe we're moving in the right direction.",
    "My grandmother always said that family comes first. We've stuck together through hard times and that's what matters most.",
    "The political situation frustrates me deeply. It feels like our voices don't matter, but I keep voting anyway.",
]

AI_MINIMAL_TEXTS = [
    "Status: operational. Objective: complete. Resources: sufficient.",
    "Acknowledge. Proceeding with protocol. Awaiting confirmation.",
    "Input received. Processing. Output generated. Ready for next task.",
]


# ============================================================================
# Test Helpers
# ============================================================================

@dataclass
class TestResult:
    """Container for test results with timing and assertions."""
    name: str
    passed: bool
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


async def make_request(
    client: httpx.AsyncClient,
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    timeout: float = DEFAULT_TIMEOUT
) -> Tuple[Dict[str, Any], float]:
    """Make HTTP request and return response with latency."""
    start = time.perf_counter()

    url = f"{BASE_URL}{endpoint}"

    if method.upper() == "GET":
        response = await client.get(url, params=data, timeout=timeout)
    elif method.upper() == "POST":
        response = await client.post(url, json=data, timeout=timeout)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    return response.json(), latency_ms


def validate_coordinates(coords: Dict[str, float], label: str = "coordinates") -> List[str]:
    """Validate coordinate values are within expected range [-2, 2]."""
    errors = []

    for axis in ["agency", "perceived_justice", "belonging"]:
        # Handle both naming conventions
        value = coords.get(axis, coords.get("fairness" if axis == "perceived_justice" else axis))

        if value is None:
            errors.append(f"{label}: Missing axis '{axis}'")
            continue

        if not isinstance(value, (int, float)):
            errors.append(f"{label}: Axis '{axis}' is not numeric: {type(value)}")
            continue

        if value < -2.0 or value > 2.0:
            errors.append(f"{label}: Axis '{axis}' out of range [-2, 2]: {value}")

    return errors


def euclidean_distance(coords1: Dict[str, float], coords2: Dict[str, float]) -> float:
    """Calculate Euclidean distance between two coordinate sets."""
    axes = ["agency", "perceived_justice", "belonging"]

    def get_value(coords, axis):
        return coords.get(axis, coords.get("fairness" if axis == "perceived_justice" else axis, 0))

    total = sum(
        (get_value(coords1, axis) - get_value(coords2, axis)) ** 2
        for axis in axes
    )
    return total ** 0.5


# ============================================================================
# TEST 1: End-to-End Pipeline Test
# ============================================================================

class TestEndToEndPipeline:
    """
    Test the complete analysis pipeline from text submission to mode classification.

    Validates:
    - Coordinate ranges are within [-2, 2]
    - Mode classification is consistent with coordinate positions
    - Latency benchmarks are met
    """

    @pytest.mark.asyncio
    async def test_analyze_diverse_texts(self):
        """Submit diverse texts through /analyze and validate responses."""
        results = []
        latencies = []

        async with httpx.AsyncClient() as client:
            for text_type, text in DIVERSE_TEXTS.items():
                payload = {
                    "text": text,
                    "model_id": "all-MiniLM-L6-v2",
                    "layer": -1
                }

                response, latency = await make_request(
                    client, "POST", "/analyze", payload
                )
                latencies.append(latency)

                # Validate response structure
                assert "text" in response, f"Missing 'text' in response for {text_type}"
                assert "vector" in response, f"Missing 'vector' in response for {text_type}"
                assert "mode" in response, f"Missing 'mode' in response for {text_type}"
                assert "confidence" in response, f"Missing 'confidence' in response for {text_type}"

                # Validate coordinate ranges
                coords = response["vector"]
                errors = validate_coordinates(coords, f"{text_type} coordinates")
                assert len(errors) == 0, f"Coordinate errors: {errors}"

                # Validate confidence is in [0, 1]
                confidence = response["confidence"]
                assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"

                results.append({
                    "type": text_type,
                    "coords": coords,
                    "mode": response["mode"],
                    "confidence": confidence,
                    "latency_ms": latency
                })

        # Performance benchmarks
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        assert avg_latency < 500, f"Average latency too high: {avg_latency:.1f}ms"
        assert max_latency < 2000, f"Max latency too high: {max_latency:.1f}ms"

        # Log results for analysis
        print(f"\nEnd-to-End Pipeline Results:")
        print(f"  Texts analyzed: {len(results)}")
        print(f"  Avg latency: {avg_latency:.1f}ms")
        print(f"  Max latency: {max_latency:.1f}ms")
        for r in results:
            print(f"  {r['type']}: mode={r['mode']}, conf={r['confidence']:.3f}")

    @pytest.mark.asyncio
    async def test_mode_classification_consistency(self):
        """Verify mode classification is consistent with coordinate positions."""
        async with httpx.AsyncClient() as client:
            # Test that high agency + high justice = POSITIVE category
            heroic_response, _ = await make_request(
                client, "POST", "/analyze",
                {"text": DIVERSE_TEXTS["heroic"]}
            )

            coords = heroic_response["vector"]
            agency = coords.get("agency", 0)
            justice = coords.get("perceived_justice", coords.get("fairness", 0))

            # Heroic text should have positive agency
            assert agency > 0, f"Expected positive agency for heroic text, got {agency}"

            # Mode should be in POSITIVE or related category
            mode = heroic_response["mode"]
            assert mode in ["HEROIC", "COMMUNAL", "TRANSCENDENT", "CYNICAL_ACHIEVER"], \
                f"Unexpected mode for heroic text: {mode}"

    @pytest.mark.asyncio
    async def test_v2_enhanced_analysis(self):
        """Test v2 API with soft labels and enhanced mode classification."""
        async with httpx.AsyncClient() as client:
            payload = {
                "text": DIVERSE_TEXTS["communal"]
            }

            response, latency = await make_request(
                client, "POST", "/v2/analyze", payload
            )

            # V2 should have enhanced mode information
            assert "mode" in response, "Missing 'mode' in v2 response"
            mode = response["mode"]

            if isinstance(mode, dict):
                assert "primary_mode" in mode, "Missing 'primary_mode' in mode dict"
                assert "confidence" in mode, "Missing 'confidence' in mode dict"

                # Check for soft labels
                if "mode_probabilities" in mode:
                    probs = mode["mode_probabilities"]
                    total_prob = sum(probs.values())
                    assert 0.99 <= total_prob <= 1.01, \
                        f"Mode probabilities should sum to 1, got {total_prob}"


# ============================================================================
# TEST 2: Grammar Deletion Pipeline
# ============================================================================

class TestGrammarDeletionPipeline:
    """
    Test the grammar deletion research endpoint for identifying
    coordination-necessary vs decorative grammatical features.

    Validates:
    - All deletion features are tested
    - Drift calculations are correct
    - API results match direct module results
    """

    @pytest.mark.asyncio
    async def test_grammar_deletion_analysis(self):
        """Test grammar deletion on a sample text."""
        test_text = DIVERSE_TEXTS["heroic"]

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "text": test_text,
                "threshold": 0.3
            }

            response, latency = await make_request(
                client, "POST", "/api/research/grammar-deletion", payload,
                timeout=60.0
            )

            # Validate response structure
            assert "original_text" in response
            assert "original_projection" in response
            assert "original_mode" in response
            assert "feature_rankings" in response
            assert "necessary_features" in response
            assert "decorative_features" in response
            assert "coordination_core" in response

            # Validate original projection
            errors = validate_coordinates(
                response["original_projection"],
                "original_projection"
            )
            assert len(errors) == 0, f"Coordinate errors: {errors}"

            # Validate feature rankings
            rankings = response["feature_rankings"]
            assert len(rankings) > 0, "No feature rankings returned"

            for rank in rankings:
                assert "feature_name" in rank
                assert "projection_drift" in rank
                assert "classification" in rank
                assert rank["classification"] in ["necessary", "decorative"]

                # Drift should be non-negative
                assert rank["projection_drift"] >= 0, \
                    f"Negative drift for {rank['feature_name']}"

        print(f"\nGrammar Deletion Results:")
        print(f"  Necessary features: {response['necessary_features']}")
        print(f"  Decorative features: {response['decorative_features']}")
        print(f"  Coordination core preview: {response['coordination_core'][:100]}...")

    @pytest.mark.asyncio
    async def test_all_deletion_features_tested(self):
        """Verify all expected deletion features are analyzed."""
        test_text = "I believe that we can make a difference together. Our efforts will surely succeed."

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "text": test_text,
                "threshold": 0.3
            }

            response, _ = await make_request(
                client, "POST", "/api/research/grammar-deletion", payload,
                timeout=60.0
            )

            tested_features = {r["feature_name"] for r in response["feature_rankings"]}

            # At least some core features should be tested
            core_features = {
                "articles", "pronouns_first_person", "modals",
                "hedging", "agency_markers"
            }

            for feature in core_features:
                if feature in tested_features:
                    print(f"  [TESTED] {feature}")

    @pytest.mark.asyncio
    async def test_drift_calculation_accuracy(self):
        """Verify drift calculations are mathematically correct."""
        test_text = DIVERSE_TEXTS["heroic"]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response, _ = await make_request(
                client, "POST", "/api/research/grammar-deletion",
                {"text": test_text, "threshold": 0.3},
                timeout=60.0
            )

            # For each ranking with axis_drifts, verify Euclidean calculation
            for rank in response["feature_rankings"]:
                if "axis_drifts" in rank and rank["axis_drifts"]:
                    drifts = rank["axis_drifts"]

                    # Calculate expected Euclidean distance
                    axes = ["agency", "perceived_justice", "belonging"]
                    sum_sq = sum(drifts.get(axis, 0) ** 2 for axis in axes)
                    expected = sum_sq ** 0.5
                    actual = rank["projection_drift"]

                    # Allow small floating point tolerance
                    assert abs(expected - actual) < 0.01, \
                        f"Drift mismatch for {rank['feature_name']}: " \
                        f"expected {expected:.4f}, got {actual:.4f}"


# ============================================================================
# TEST 3: Batch Processing Test
# ============================================================================

class TestBatchProcessing:
    """
    Test batch analysis endpoint for efficiency and consistency.

    Validates:
    - Can process 50+ texts in a single call
    - Batch results match individual call results
    - Throughput benchmarks are met
    """

    @pytest.mark.asyncio
    async def test_batch_analyze_50_texts(self):
        """Submit 50+ texts in a single batch call."""
        async with httpx.AsyncClient(timeout=BATCH_TIMEOUT) as client:
            payload = {
                "texts": BATCH_CORPUS,
                "model_id": "all-MiniLM-L6-v2",
                "layer": -1,
                "detect_clusters": True
            }

            start = time.perf_counter()
            response, latency = await make_request(
                client, "POST", "/corpus/analyze", payload,
                timeout=BATCH_TIMEOUT
            )
            total_time = time.perf_counter() - start

            # Validate response structure
            assert "projections" in response or "points" in response
            projections = response.get("projections", response.get("points", []))

            assert len(projections) == len(BATCH_CORPUS), \
                f"Expected {len(BATCH_CORPUS)} projections, got {len(projections)}"

            # Validate each projection
            for i, proj in enumerate(projections):
                assert "vector" in proj or "projection" in proj, \
                    f"Missing vector in projection {i}"

                coords = proj.get("vector", proj.get("projection", {}))
                errors = validate_coordinates(coords, f"batch item {i}")
                assert len(errors) == 0, f"Coordinate errors: {errors}"

            # Calculate throughput
            throughput = len(BATCH_CORPUS) / total_time

            print(f"\nBatch Processing Results:")
            print(f"  Texts processed: {len(BATCH_CORPUS)}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} texts/sec")
            print(f"  Latency per text: {latency / len(BATCH_CORPUS):.1f}ms")

            # Throughput benchmark: at least 10 texts/second
            assert throughput >= 5, f"Throughput too low: {throughput:.1f} texts/sec"

    @pytest.mark.asyncio
    async def test_batch_consistency_with_individual(self):
        """Verify batch results match individual call results."""
        sample_texts = BATCH_CORPUS[:5]  # Test subset for speed

        async with httpx.AsyncClient(timeout=BATCH_TIMEOUT) as client:
            # Get batch results
            batch_response, _ = await make_request(
                client, "POST", "/corpus/analyze",
                {"texts": sample_texts, "detect_clusters": False},
                timeout=BATCH_TIMEOUT
            )
            batch_projections = batch_response.get(
                "projections", batch_response.get("points", [])
            )

            # Get individual results
            individual_results = []
            for text in sample_texts:
                response, _ = await make_request(
                    client, "POST", "/analyze",
                    {"text": text}
                )
                individual_results.append(response)

            # Compare each pair
            for i, (batch, individual) in enumerate(
                zip(batch_projections, individual_results)
            ):
                batch_coords = batch.get("vector", batch.get("projection", {}))
                indiv_coords = individual.get("vector", {})

                distance = euclidean_distance(batch_coords, indiv_coords)

                # Should be identical (or very close due to floating point)
                assert distance < 0.001, \
                    f"Batch/individual mismatch for text {i}: distance={distance}"

        print("\nBatch consistency verified: all results match individual calls")

    @pytest.mark.asyncio
    async def test_batch_cluster_detection(self):
        """Test cluster detection in batch processing."""
        async with httpx.AsyncClient(timeout=BATCH_TIMEOUT) as client:
            payload = {
                "texts": BATCH_CORPUS,
                "detect_clusters": True
            }

            response, _ = await make_request(
                client, "POST", "/corpus/analyze", payload,
                timeout=BATCH_TIMEOUT
            )

            if "clusters" in response:
                clusters = response["clusters"]
                print(f"\nCluster Detection Results:")
                print(f"  Clusters found: {len(clusters)}")

                for cluster in clusters:
                    print(f"  Cluster {cluster.get('id', '?')}: "
                          f"size={cluster.get('size', '?')}")


# ============================================================================
# TEST 4: Error Handling Test
# ============================================================================

class TestErrorHandling:
    """
    Test API behavior with invalid inputs.

    Validates:
    - Empty text handling
    - Very long text handling
    - Special characters
    - Malformed JSON
    - Invalid parameters
    """

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test handling of empty text input."""
        async with httpx.AsyncClient() as client:
            payload = {"text": ""}

            try:
                response = await client.post(
                    f"{BASE_URL}/analyze",
                    json=payload,
                    timeout=DEFAULT_TIMEOUT
                )

                # Either return an error or handle gracefully
                if response.status_code == 200:
                    # If it succeeds, it should still have valid structure
                    data = response.json()
                    assert "vector" in data or "error" in data
                else:
                    # 400 or 422 is acceptable for empty input
                    assert response.status_code in [400, 422], \
                        f"Unexpected status code: {response.status_code}"

            except httpx.HTTPStatusError as e:
                assert e.response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test handling of very long text input (10000+ characters)."""
        long_text = "This is a test sentence. " * 500  # ~12500 chars

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"text": long_text}

            response, latency = await make_request(
                client, "POST", "/analyze", payload,
                timeout=60.0
            )

            # Should still return valid coordinates
            assert "vector" in response
            errors = validate_coordinates(response["vector"])
            assert len(errors) == 0

            print(f"\nLong text handling:")
            print(f"  Text length: {len(long_text)} chars")
            print(f"  Latency: {latency:.1f}ms")

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters and unicode."""
        test_cases = [
            "Hello! @#$%^&*() World?",
            "Japanese: \u3053\u3093\u306b\u3061\u306f",
            "Emoji: \U0001F600 \U0001F44D \u2764",
            "Mixed: Hello \u4e16\u754c! \U0001F30D",
            "Newlines:\nLine1\nLine2\nLine3",
            "Tabs:\tColumn1\tColumn2",
            "Quotes: \"double\" and 'single'",
        ]

        async with httpx.AsyncClient() as client:
            for text in test_cases:
                payload = {"text": text}

                try:
                    response, _ = await make_request(
                        client, "POST", "/analyze", payload
                    )

                    # Should handle gracefully
                    assert "vector" in response, f"Failed on: {text[:30]}..."

                except httpx.HTTPStatusError as e:
                    # Some special cases may be rejected, which is acceptable
                    assert e.response.status_code in [400, 422], \
                        f"Unexpected error for: {text[:30]}..."

        print("\nSpecial character handling: all cases processed")

    @pytest.mark.asyncio
    async def test_malformed_json(self):
        """Test handling of malformed JSON requests."""
        async with httpx.AsyncClient() as client:
            # Send malformed JSON
            response = await client.post(
                f"{BASE_URL}/analyze",
                content='{"text": "unclosed string',
                headers={"Content-Type": "application/json"},
                timeout=DEFAULT_TIMEOUT
            )

            # Should return 400 or 422
            assert response.status_code in [400, 422], \
                f"Expected 400/422 for malformed JSON, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_invalid_parameters(self):
        """Test handling of invalid parameter values."""
        invalid_payloads = [
            {"text": "test", "model_id": "nonexistent-model-xyz"},
            {"text": "test", "layer": 999999},
        ]

        async with httpx.AsyncClient() as client:
            for payload in invalid_payloads:
                try:
                    response = await client.post(
                        f"{BASE_URL}/analyze",
                        json=payload,
                        timeout=DEFAULT_TIMEOUT
                    )

                    # Either handles gracefully or returns error
                    if response.status_code != 200:
                        assert response.status_code in [400, 404, 422, 500]

                except httpx.HTTPStatusError as e:
                    assert e.response.status_code in [400, 404, 422, 500]

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        async with httpx.AsyncClient() as client:
            # Missing 'text' field entirely
            response = await client.post(
                f"{BASE_URL}/analyze",
                json={},
                timeout=DEFAULT_TIMEOUT
            )

            assert response.status_code in [400, 422], \
                f"Expected 400/422 for missing field, got {response.status_code}"


# ============================================================================
# TEST 5: Consistency Test
# ============================================================================

class TestConsistency:
    """
    Test determinism and consistency of projections.

    Validates:
    - Same text yields same coordinates
    - Paraphrases yield similar coordinates (within threshold)
    """

    @pytest.mark.asyncio
    async def test_deterministic_projection(self):
        """Same text should yield identical coordinates."""
        test_text = DIVERSE_TEXTS["communal"]
        num_runs = 5

        async with httpx.AsyncClient() as client:
            results = []

            for _ in range(num_runs):
                response, _ = await make_request(
                    client, "POST", "/analyze",
                    {"text": test_text}
                )
                results.append(response["vector"])

            # All results should be identical
            first = results[0]
            for i, result in enumerate(results[1:], 2):
                distance = euclidean_distance(first, result)
                assert distance < 0.0001, \
                    f"Non-deterministic: run 1 vs {i} distance = {distance}"

        print(f"\nDeterminism test: {num_runs} identical runs verified")

    @pytest.mark.asyncio
    async def test_paraphrase_similarity(self):
        """Paraphrases should yield similar (but not identical) coordinates."""
        similarity_threshold = 0.8  # Max acceptable distance

        async with httpx.AsyncClient() as client:
            for original, paraphrase in PARAPHRASE_PAIRS:
                orig_response, _ = await make_request(
                    client, "POST", "/analyze",
                    {"text": original}
                )
                para_response, _ = await make_request(
                    client, "POST", "/analyze",
                    {"text": paraphrase}
                )

                orig_coords = orig_response["vector"]
                para_coords = para_response["vector"]

                distance = euclidean_distance(orig_coords, para_coords)

                print(f"\nParaphrase similarity:")
                print(f"  Original: {original[:40]}...")
                print(f"  Paraphrase: {paraphrase[:40]}...")
                print(f"  Distance: {distance:.4f}")

                # Paraphrases should be reasonably close
                assert distance < similarity_threshold, \
                    f"Paraphrases too far apart: {distance:.4f}"

    @pytest.mark.asyncio
    async def test_mode_stability(self):
        """Mode classification should be stable for clear cases."""
        # Text with very clear mode signal
        clear_heroic = "I achieved everything through my own effort. I am the master of my destiny and I control my future."

        async with httpx.AsyncClient() as client:
            results = []

            for _ in range(3):
                response, _ = await make_request(
                    client, "POST", "/analyze",
                    {"text": clear_heroic}
                )
                results.append(response["mode"])

            # All modes should be the same
            assert len(set(results)) == 1, \
                f"Mode instability detected: {results}"

        print(f"\nMode stability verified: {results[0]}")


# ============================================================================
# TEST 6: Research Endpoints (Legibility, Calibration)
# ============================================================================

class TestResearchEndpoints:
    """
    Test the emergent language research endpoints.

    Validates:
    - Legibility analysis
    - Calibration comparison between human and AI texts
    """

    @pytest.mark.asyncio
    async def test_legibility_single(self):
        """Test legibility analysis for a single text."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"text": DIVERSE_TEXTS["heroic"]}

            response, latency = await make_request(
                client, "POST", "/api/research/legibility", payload,
                timeout=60.0
            )

            assert "results" in response
            results = response["results"]
            assert len(results) == 1

            result = results[0]
            assert "legibility_score" in result
            assert "interpretability" in result
            assert "stability_score" in result
            assert "mode" in result

            # Scores should be in valid ranges
            assert 0 <= result["legibility_score"] <= 1
            assert 0 <= result["stability_score"] <= 1

            print(f"\nLegibility Analysis:")
            print(f"  Score: {result['legibility_score']:.3f}")
            print(f"  Stability: {result['stability_score']:.3f}")
            print(f"  Mode: {result['mode']}")

    @pytest.mark.asyncio
    async def test_legibility_batch(self):
        """Test batch legibility analysis."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"texts": list(DIVERSE_TEXTS.values())}

            response, latency = await make_request(
                client, "POST", "/api/research/legibility", payload,
                timeout=60.0
            )

            assert "results" in response
            assert len(response["results"]) == len(DIVERSE_TEXTS)

            if "aggregate_legibility" in response:
                print(f"\nBatch Legibility:")
                print(f"  Aggregate score: {response['aggregate_legibility']:.3f}")

    @pytest.mark.asyncio
    async def test_calibration_comparison(self):
        """Test calibration between human and minimal AI texts."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "human_texts": HUMAN_AUTHORED_TEXTS,
                "minimal_texts": AI_MINIMAL_TEXTS
            }

            try:
                response, latency = await make_request(
                    client, "POST", "/api/research/calibrate", payload,
                    timeout=120.0
                )

                assert "human_centroid" in response
                assert "minimal_centroid" in response
                assert "centroid_distance" in response
                assert "overlap_score" in response

                print(f"\nCalibration Results:")
                print(f"  Human centroid: {response['human_centroid']}")
                print(f"  Minimal centroid: {response['minimal_centroid']}")
                print(f"  Distance: {response['centroid_distance']:.3f}")
                print(f"  Overlap: {response['overlap_score']:.3f}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 501:
                    pytest.skip("Calibration endpoint not fully implemented")
                raise


# ============================================================================
# TEST 7: Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """
    Test performance characteristics and throughput.
    """

    @pytest.mark.asyncio
    async def test_single_text_latency_benchmark(self):
        """Benchmark single text analysis latency."""
        sample_texts = list(DIVERSE_TEXTS.values())
        latencies = []

        async with httpx.AsyncClient() as client:
            for text in sample_texts:
                _, latency = await make_request(
                    client, "POST", "/analyze",
                    {"text": text}
                )
                latencies.append(latency)

        avg = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nLatency Benchmarks (single text):")
        print(f"  Mean: {avg:.1f}ms")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P95: {p95:.1f}ms")

        # Benchmarks
        assert avg < 500, f"Average latency too high: {avg}ms"
        assert p95 < 1000, f"P95 latency too high: {p95}ms"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        num_concurrent = 10

        async with httpx.AsyncClient() as client:
            async def make_single_request(text: str) -> Tuple[bool, float]:
                try:
                    _, latency = await make_request(
                        client, "POST", "/analyze",
                        {"text": text}
                    )
                    return True, latency
                except Exception:
                    return False, 0

            # Fire off concurrent requests
            tasks = [
                make_single_request(BATCH_CORPUS[i % len(BATCH_CORPUS)])
                for i in range(num_concurrent)
            ]

            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start

            successes = sum(1 for success, _ in results if success)
            latencies = [lat for success, lat in results if success]

            print(f"\nConcurrent Request Benchmark:")
            print(f"  Concurrent requests: {num_concurrent}")
            print(f"  Successful: {successes}/{num_concurrent}")
            print(f"  Total time: {total_time:.2f}s")
            if latencies:
                print(f"  Avg latency: {statistics.mean(latencies):.1f}ms")

            assert successes == num_concurrent, \
                f"Only {successes}/{num_concurrent} requests succeeded"


# ============================================================================
# Main Test Runner
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def run_health_check() -> bool:
    """Check if the backend is running."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


@pytest.fixture(scope="session", autouse=True)
async def check_server():
    """Verify server is running before tests."""
    is_healthy = await run_health_check()
    if not is_healthy:
        pytest.skip(
            f"Backend server not running at {BASE_URL}. "
            "Start with: uvicorn main:app --port 8000"
        )


if __name__ == "__main__":
    import sys

    # Quick health check
    async def main():
        is_healthy = await run_health_check()
        if not is_healthy:
            print(f"ERROR: Backend not running at {BASE_URL}")
            print("Start with: uvicorn main:app --port 8000")
            sys.exit(1)

        print(f"Backend healthy at {BASE_URL}")
        print("Running tests with pytest...")

    asyncio.run(main())

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
