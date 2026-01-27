#!/usr/bin/env python3
"""Test all new observatory APIs."""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_compare():
    print("\n" + "="*60)
    print("TEST 1: Comparison API")
    print("="*60)

    response = requests.post(f"{BASE_URL}/api/v2/compare", json={
        "group_a": {
            "name": "Corporate",
            "texts": [
                "We are committed to excellence and customer satisfaction.",
                "Our team delivers world-class results."
            ]
        },
        "group_b": {
            "name": "Employee",
            "texts": [
                "Management does not listen to us.",
                "The workload is crushing and pay is unfair."
            ]
        }
    })

    if response.ok:
        data = response.json()
        print(f"Status: SUCCESS")
        print(f"Euclidean Distance: {data['gap_analysis']['euclidean_distance']:.3f}")
        print(f"Justice Gap: {data['gap_analysis']['axis_gaps']['perceived_justice']['gap']:.3f}")
        print(f"Effect Size (Justice): {data['gap_analysis']['effect_sizes']['perceived_justice_cohens_d']:.2f}")
        print(f"Interpretation: {data['interpretation'][:100]}...")
    else:
        print(f"FAILED: {response.status_code}")
        print(response.text)

    return response.ok

def test_trajectory():
    print("\n" + "="*60)
    print("TEST 2: Trajectory API")
    print("="*60)

    response = requests.post(f"{BASE_URL}/api/v2/trajectory", json={
        "name": "Brand Evolution",
        "points": [
            {"timestamp": "2024-01-01", "text": "We are the market leader in innovation."},
            {"timestamp": "2024-06-01", "text": "We are facing some challenges but remain committed."},
            {"timestamp": "2025-01-01", "text": "We have learned from our mistakes and grown stronger."}
        ]
    })

    if response.ok:
        data = response.json()
        print(f"Status: SUCCESS")
        print(f"Total Distance: {data['trajectory']['total_distance_traveled']:.3f}")
        print(f"Net Displacement: {data['trajectory']['net_displacement']:.3f}")
        print(f"Mode Transitions: {len(data['mode_transitions'])}")
        print(f"Interpretation: {data['interpretation'][:100]}...")
    else:
        print(f"FAILED: {response.status_code}")
        print(response.text)

    return response.ok

def test_alerts():
    print("\n" + "="*60)
    print("TEST 3: Alert System")
    print("="*60)

    # Create an alert
    create_response = requests.post(f"{BASE_URL}/api/v2/alerts", json={
        "name": "Justice Gap Alert",
        "description": "Alert when justice gap exceeds 0.3",
        "type": "gap_threshold",
        "config": {
            "axis": "perceived_justice",
            "threshold": 0.3,
            "direction": "exceeds"
        },
        "enabled": True
    })

    if create_response.ok:
        alert = create_response.json()
        print(f"Created Alert: {alert['name']} (ID: {alert['id'][:8]}...)")
    else:
        print(f"Create FAILED: {create_response.status_code}")
        return False

    # Check against texts
    check_response = requests.post(f"{BASE_URL}/api/v2/alerts/check", json={
        "group_a": {"name": "Official", "texts": ["Everything is great and fair."]},
        "group_b": {"name": "Reality", "texts": ["The system is rigged against us."]}
    })

    if check_response.ok:
        data = check_response.json()
        print(f"Alerts Triggered: {len(data['triggered_alerts'])}")
        print(f"All Clear: {data['all_clear']}")
        if data['triggered_alerts']:
            t = data['triggered_alerts'][0]
            print(f"  - {t['alert_name']}: {t['message']}")
    else:
        print(f"Check FAILED: {check_response.status_code}")
        return False

    return True

def test_outliers():
    print("\n" + "="*60)
    print("TEST 4: Outlier Detection")
    print("="*60)

    response = requests.post(f"{BASE_URL}/api/v2/analytics/outliers", json={
        "corpus": [
            "I worked hard and achieved my goals through determination.",
            "Success comes to those who persevere and stay positive.",
            "With effort and dedication, anything is possible.",
            "I built my career through consistent hard work."
        ],
        "test_text": "Nothing ever works out. The system is designed to keep us down.",
        "threshold": 2.0
    })

    if response.ok:
        data = response.json()
        print(f"Status: SUCCESS")
        print(f"Is Outlier: {data['is_outlier']}")
        print(f"Mahalanobis Distance: {data['mahalanobis_distance']:.2f}")
        print(f"Z-Scores: Agency={data['z_scores']['agency']:.2f}, Justice={data['z_scores']['perceived_justice']:.2f}")
        print(f"Interpretation: {data['interpretation'][:80]}...")
    else:
        print(f"FAILED: {response.status_code}")
        print(response.text)

    return response.ok

def test_cohorts():
    print("\n" + "="*60)
    print("TEST 5: Cohort Analysis")
    print("="*60)

    response = requests.post(f"{BASE_URL}/api/v2/analytics/cohorts", json={
        "cohorts": {
            "optimists": [
                "The future is bright and full of opportunity.",
                "Good things come to those who work hard."
            ],
            "pessimists": [
                "Things never work out no matter what you do.",
                "The deck is stacked against ordinary people."
            ],
            "realists": [
                "Some things work out, some do not. Life goes on.",
                "You win some, you lose some. Balance is key."
            ]
        }
    })

    if response.ok:
        data = response.json()
        print(f"Status: SUCCESS")
        for cohort, profile in data['cohort_profiles'].items():
            print(f"  {cohort}: Agency={profile['mean_agency']:.2f}, Justice={profile['mean_perceived_justice']:.2f}, Mode={profile['dominant_mode']}")
        print(f"ANOVA - Agency significant: {data['anova_results']['agency']['significant']}")
        print(f"ANOVA - Justice significant: {data['anova_results']['perceived_justice']['significant']}")
    else:
        print(f"FAILED: {response.status_code}")
        print(response.text)

    return response.ok

def test_mode_flow():
    print("\n" + "="*60)
    print("TEST 6: Mode Flow Analysis")
    print("="*60)

    response = requests.post(f"{BASE_URL}/api/v2/analytics/mode-flow", json={
        "texts": [
            "I am a successful leader making great things happen.",
            "Things are getting harder but I am managing.",
            "I feel overwhelmed and the system works against me.",
            "We need to fight back and demand change together."
        ]
    })

    if response.ok:
        data = response.json()
        print(f"Status: SUCCESS")
        print(f"Mode Sequence: {' -> '.join(data['sequence'])}")
        print(f"Flow Pattern: {data['flow_pattern']}")
        print(f"Transitions: {len(data['transitions'])}")
        print(f"Interpretation: {data['interpretation'][:80]}...")
    else:
        print(f"FAILED: {response.status_code}")
        print(response.text)

    return response.ok

def test_dashboard():
    print("\n" + "="*60)
    print("TEST 7: Dashboard Endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/dashboard")

    if response.ok:
        content = response.text
        has_threejs = "three.min.js" in content or "Three.js" in content.lower() or "THREE" in content
        has_canvas = "<canvas" in content
        print(f"Status: SUCCESS")
        print(f"Content Length: {len(content)} bytes")
        print(f"Has Three.js: {has_threejs}")
        print(f"Has Canvas: {has_canvas}")
    else:
        print(f"FAILED: {response.status_code}")

    return response.ok


if __name__ == "__main__":
    print("="*60)
    print("CULTURAL SOLITON OBSERVATORY - NEW API TESTS")
    print("="*60)

    results = []

    results.append(("Comparison API", test_compare()))
    results.append(("Trajectory API", test_trajectory()))
    results.append(("Alert System", test_alerts()))
    results.append(("Outlier Detection", test_outliers()))
    results.append(("Cohort Analysis", test_cohorts()))
    results.append(("Mode Flow", test_mode_flow()))
    results.append(("Dashboard", test_dashboard()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\n{total - passed} tests failed")
