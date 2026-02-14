#!/usr/bin/env python3
"""
Cognitive Elements Data Validation

Validates element, experiment, and results JSON files against their schemas.
Also provides utilities for working with the research instrument data.

Usage:
    python validate.py elements    # Validate elements.json
    python validate.py experiments # Validate all experiments
    python validate.py all         # Validate everything
    python validate.py stats       # Show element statistics
"""

import json
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

# Schema validation (simplified - for full validation use jsonschema library)
def validate_required_fields(data: dict, required: list, path: str = "") -> list:
    """Check that required fields are present."""
    errors = []
    for field in required:
        if field not in data:
            errors.append(f"{path}.{field}: Required field missing")
    return errors


def validate_element(element: dict, idx: int = 0) -> list:
    """Validate a single cognitive element."""
    errors = []
    path = f"elements[{idx}]"

    # Required fields
    required = ["id", "symbol", "name", "group", "quantum_numbers", "description", "version"]
    errors.extend(validate_required_fields(element, required, path))

    # Validate group
    valid_groups = ["epistemic", "analytical", "generative", "evaluative",
                   "dialogical", "pedagogical", "temporal", "contextual"]
    if element.get("group") not in valid_groups:
        errors.append(f"{path}.group: Invalid group '{element.get('group')}'")

    # Validate quantum numbers
    qn = element.get("quantum_numbers", {})
    if "direction" in qn and qn["direction"] not in ["I", "O", "T", "Τ"]:
        errors.append(f"{path}.quantum_numbers.direction: Invalid value")
    if "stance" in qn and qn["stance"] not in ["+", "?", "-"]:
        errors.append(f"{path}.quantum_numbers.stance: Invalid value")
    if "scope" in qn and qn["scope"] not in ["a", "m", "s", "μ"]:
        errors.append(f"{path}.quantum_numbers.scope: Invalid value")
    if "transform" in qn and qn["transform"] not in ["P", "G", "R", "D"]:
        errors.append(f"{path}.quantum_numbers.transform: Invalid value")

    # Validate manifold position ranges
    mp = element.get("manifold_position", {})
    for axis in ["agency", "justice", "belonging"]:
        if axis in mp:
            val = mp[axis]
            if not isinstance(val, (int, float)) or val < -1 or val > 1:
                errors.append(f"{path}.manifold_position.{axis}: Must be between -1 and 1")

    # Validate training status
    ts = element.get("training_status", {})
    if ts.get("trigger_rate") is not None:
        tr = ts["trigger_rate"]
        if not isinstance(tr, (int, float)) or tr < 0 or tr > 1:
            errors.append(f"{path}.training_status.trigger_rate: Must be between 0 and 1")

    return errors


def validate_experiment(experiment: dict, path: str = "experiment") -> list:
    """Validate an experiment definition."""
    errors = []

    # Required fields
    required = ["id", "name", "hypothesis", "elements", "methodology", "status", "version"]
    errors.extend(validate_required_fields(experiment, required, path))

    # Validate hypothesis
    hypothesis = experiment.get("hypothesis", {})
    if "statement" not in hypothesis:
        errors.append(f"{path}.hypothesis.statement: Required field missing")
    if "falsifiable" not in hypothesis:
        errors.append(f"{path}.hypothesis.falsifiable: Required field missing")

    # Validate methodology
    methodology = experiment.get("methodology", {})
    valid_types = ["trigger_rate", "interference", "compound_stability",
                  "transfer", "manifold_position", "isotope_differentiation", "behavioral"]
    if methodology.get("type") not in valid_types:
        errors.append(f"{path}.methodology.type: Invalid type '{methodology.get('type')}'")

    # Validate status
    valid_status = ["draft", "registered", "running", "completed", "failed", "abandoned"]
    if experiment.get("status") not in valid_status:
        errors.append(f"{path}.status: Invalid status '{experiment.get('status')}'")

    return errors


def validate_elements_file(filepath: Path) -> tuple[bool, list]:
    """Validate the elements.json file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {filepath}"]

    errors = []

    # Check required top-level fields
    if "version" not in data:
        errors.append("Root: version field missing")
    if "elements" not in data:
        errors.append("Root: elements array missing")
        return False, errors

    # Validate each element
    for idx, element in enumerate(data["elements"]):
        errors.extend(validate_element(element, idx))

    return len(errors) == 0, errors


def validate_experiment_file(filepath: Path) -> tuple[bool, list]:
    """Validate an experiment JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {filepath}"]

    errors = validate_experiment(data, filepath.stem)
    return len(errors) == 0, errors


def get_element_stats(filepath: Path) -> dict:
    """Get statistics about the elements."""
    with open(filepath) as f:
        data = json.load(f)

    elements = data["elements"]

    # Count by group
    by_group = {}
    for el in elements:
        g = el.get("group", "unknown")
        by_group[g] = by_group.get(g, 0) + 1

    # Count trained vs untrained
    trained = sum(1 for el in elements if el.get("training_status", {}).get("trained", False))
    untrained = len(elements) - trained

    # Count isotopes
    total_isotopes = sum(len(el.get("isotopes", [])) for el in elements)

    # Get training versions
    versions = {}
    for el in elements:
        ts = el.get("training_status", {})
        if ts.get("trained"):
            v = ts.get("version", "unknown")
            versions[v] = versions.get(v, 0) + 1

    # Average trigger rate for trained elements
    trigger_rates = [
        el["training_status"]["trigger_rate"]
        for el in elements
        if el.get("training_status", {}).get("trigger_rate") is not None
    ]
    avg_trigger_rate = sum(trigger_rates) / len(trigger_rates) if trigger_rates else 0

    return {
        "total_elements": len(elements),
        "by_group": by_group,
        "trained": trained,
        "untrained": untrained,
        "total_isotopes": total_isotopes,
        "by_version": versions,
        "avg_trigger_rate": avg_trigger_rate
    }


def print_stats(stats: dict):
    """Pretty print element statistics."""
    print("\n" + "=" * 60)
    print("COGNITIVE ELEMENTS STATISTICS")
    print("=" * 60)

    print(f"\nTotal Elements: {stats['total_elements']}")
    print(f"  Trained: {stats['trained']}")
    print(f"  Untrained: {stats['untrained']}")
    print(f"  Total Isotopes: {stats['total_isotopes']}")
    print(f"  Average Trigger Rate: {stats['avg_trigger_rate']:.1%}")

    print("\nBy Group:")
    for group, count in sorted(stats['by_group'].items()):
        print(f"  {group:12} {count}")

    print("\nBy Training Version:")
    for version, count in sorted(stats['by_version'].items()):
        print(f"  {version:8} {count}")

    print("=" * 60)


def main():
    base_path = Path(__file__).parent

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "elements":
        elements_path = base_path / "data" / "elements.json"
        valid, errors = validate_elements_file(elements_path)
        if valid:
            print(f"✓ {elements_path.name} is valid")
        else:
            print(f"✗ {elements_path.name} has {len(errors)} error(s):")
            for err in errors:
                print(f"  - {err}")
        sys.exit(0 if valid else 1)

    elif command == "experiments":
        experiments_path = base_path / "data" / "experiments"
        if not experiments_path.exists():
            print("No experiments directory found")
            sys.exit(1)

        all_valid = True
        for exp_file in experiments_path.glob("*.json"):
            valid, errors = validate_experiment_file(exp_file)
            if valid:
                print(f"✓ {exp_file.name} is valid")
            else:
                print(f"✗ {exp_file.name} has {len(errors)} error(s):")
                for err in errors:
                    print(f"  - {err}")
                all_valid = False

        sys.exit(0 if all_valid else 1)

    elif command == "all":
        # Validate elements
        elements_path = base_path / "data" / "elements.json"
        el_valid, el_errors = validate_elements_file(elements_path)
        if el_valid:
            print(f"✓ elements.json is valid")
        else:
            print(f"✗ elements.json has {len(el_errors)} error(s)")
            for err in el_errors:
                print(f"  - {err}")

        # Validate experiments
        experiments_path = base_path / "data" / "experiments"
        exp_valid = True
        if experiments_path.exists():
            for exp_file in experiments_path.glob("*.json"):
                valid, errors = validate_experiment_file(exp_file)
                if valid:
                    print(f"✓ {exp_file.name} is valid")
                else:
                    print(f"✗ {exp_file.name} has {len(errors)} error(s)")
                    for err in errors:
                        print(f"  - {err}")
                    exp_valid = False

        sys.exit(0 if (el_valid and exp_valid) else 1)

    elif command == "stats":
        elements_path = base_path / "data" / "elements.json"
        stats = get_element_stats(elements_path)
        print_stats(stats)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
