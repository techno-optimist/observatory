"""
Axis Configuration and Aliasing System

Provides a centralized configuration for the three-dimensional cultural manifold axes.
Supports backward compatibility through axis aliasing, allowing old code using "fairness"
to continue working while new code uses "perceived_justice".

The validity study (January 2026) found that the "Fairness" axis conflates:
1. Abstract fairness values ("Everyone deserves equal treatment")
2. System legitimacy beliefs ("The system is rigged")

The axis has been renamed to "Perceived Justice" to accurately reflect what it measures.

Usage:
    from models.axis_config import (
        AXIS_CONFIG,
        translate_axis_name,
        get_axis_display_name,
        get_canonical_axis_name,
        get_all_axis_aliases,
        CANONICAL_AXES,
        INTERNAL_AXES
    )
"""

from typing import Dict, List, Optional, Any


# Canonical axis configuration
# Internal storage uses 'fairness' for backward compatibility with saved projections
AXIS_CONFIG: Dict[str, Dict[str, Any]] = {
    "agency": {
        "canonical_name": "agency",
        "internal_name": "agency",  # Used in storage/database
        "display_name": "Agency",
        "aliases": ["agency"],
        "description": "Sense of personal control and self-determination",
        "range": [-2.0, 2.0],
        "positive_pole": "High personal agency, self-efficacy",
        "negative_pole": "Low agency, external locus of control"
    },
    "perceived_justice": {
        "canonical_name": "perceived_justice",
        "internal_name": "fairness",  # Kept as 'fairness' for backward compatibility
        "display_name": "Perceived Justice",
        "aliases": ["fairness", "perceived_justice", "justice", "perceived justice"],
        "description": "Belief in fair treatment and system legitimacy",
        "range": [-2.0, 2.0],
        "positive_pole": "Belief in fair systems and just outcomes",
        "negative_pole": "Perception of systemic unfairness and corruption",
        "deprecation_notice": "Previously labeled 'Fairness'. Renamed to 'Perceived Justice' "
                             "based on validity study findings (Jan 2026) that showed the axis "
                             "conflates abstract fairness values with system legitimacy beliefs."
    },
    "belonging": {
        "canonical_name": "belonging",
        "internal_name": "belonging",  # Used in storage/database
        "display_name": "Belonging",
        "aliases": ["belonging"],
        "description": "Sense of social connection and group membership",
        "range": [-2.0, 2.0],
        "positive_pole": "Strong community bonds and group identity",
        "negative_pole": "Social isolation and alienation"
    }
}

# List of canonical axis names (for external API use)
CANONICAL_AXES: List[str] = ["agency", "perceived_justice", "belonging"]

# List of internal axis names (for storage/backward compatibility)
INTERNAL_AXES: List[str] = ["agency", "fairness", "belonging"]

# Build reverse lookup from alias to canonical name
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, config in AXIS_CONFIG.items():
    for alias in config["aliases"]:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical

# Build reverse lookup from internal name to canonical name
_INTERNAL_TO_CANONICAL: Dict[str, str] = {
    config["internal_name"]: canonical
    for canonical, config in AXIS_CONFIG.items()
}

# Build reverse lookup from canonical name to internal name
_CANONICAL_TO_INTERNAL: Dict[str, str] = {
    canonical: config["internal_name"]
    for canonical, config in AXIS_CONFIG.items()
}


def translate_axis_name(name: str, to_internal: bool = False) -> str:
    """
    Translate an axis name between canonical and internal representations.

    Args:
        name: The axis name to translate (can be any alias)
        to_internal: If True, returns internal name (for storage)
                    If False, returns canonical name (for API responses)

    Returns:
        Translated axis name

    Examples:
        >>> translate_axis_name("fairness")
        'perceived_justice'
        >>> translate_axis_name("perceived_justice", to_internal=True)
        'fairness'
        >>> translate_axis_name("justice")
        'perceived_justice'
    """
    name_lower = name.lower().strip()

    # First, find the canonical name from any alias
    canonical = _ALIAS_TO_CANONICAL.get(name_lower)

    # If not found in aliases, check if it's an internal name
    if canonical is None:
        canonical = _INTERNAL_TO_CANONICAL.get(name_lower)

    # If still not found, return original (unknown axis)
    if canonical is None:
        return name

    if to_internal:
        return _CANONICAL_TO_INTERNAL.get(canonical, name)
    else:
        return canonical


def get_axis_display_name(name: str) -> str:
    """
    Get the human-readable display name for an axis.

    Args:
        name: Axis name (can be any alias, canonical, or internal name)

    Returns:
        Human-readable display name

    Examples:
        >>> get_axis_display_name("fairness")
        'Perceived Justice'
        >>> get_axis_display_name("agency")
        'Agency'
    """
    canonical = translate_axis_name(name)
    config = AXIS_CONFIG.get(canonical)
    if config:
        return config["display_name"]
    return name.title()


def get_canonical_axis_name(name: str) -> str:
    """
    Get the canonical axis name from any alias or internal name.

    Args:
        name: Axis name (can be any alias or internal name)

    Returns:
        Canonical axis name

    Examples:
        >>> get_canonical_axis_name("fairness")
        'perceived_justice'
        >>> get_canonical_axis_name("justice")
        'perceived_justice'
    """
    return translate_axis_name(name, to_internal=False)


def get_internal_axis_name(name: str) -> str:
    """
    Get the internal axis name (used for storage) from any alias.

    Args:
        name: Axis name (can be any alias or canonical name)

    Returns:
        Internal axis name for storage

    Examples:
        >>> get_internal_axis_name("perceived_justice")
        'fairness'
        >>> get_internal_axis_name("justice")
        'fairness'
    """
    return translate_axis_name(name, to_internal=True)


def get_all_axis_aliases() -> Dict[str, List[str]]:
    """
    Get all aliases for each canonical axis name.

    Returns:
        Dictionary mapping canonical names to their aliases
    """
    return {
        canonical: config["aliases"].copy()
        for canonical, config in AXIS_CONFIG.items()
    }


def get_axis_description(name: str) -> str:
    """
    Get the description for an axis.

    Args:
        name: Axis name (can be any alias)

    Returns:
        Description string
    """
    canonical = translate_axis_name(name)
    config = AXIS_CONFIG.get(canonical)
    if config:
        return config["description"]
    return f"Unknown axis: {name}"


def get_axis_deprecation_notice(name: str) -> Optional[str]:
    """
    Get deprecation notice for an axis name if applicable.

    Args:
        name: Axis name to check

    Returns:
        Deprecation notice string or None if not deprecated
    """
    name_lower = name.lower().strip()

    # Check if using deprecated alias
    if name_lower == "fairness":
        return (
            "DEPRECATION NOTICE: The axis name 'fairness' is deprecated. "
            "Please use 'perceived_justice' instead. The axis was renamed based on "
            "validity study findings that showed it conflates abstract fairness values "
            "with system legitimacy beliefs."
        )

    return None


def convert_vector_keys_to_canonical(vector: Dict[str, float]) -> Dict[str, float]:
    """
    Convert a vector dict from internal names to canonical names.

    Args:
        vector: Dictionary with internal axis names as keys

    Returns:
        Dictionary with canonical axis names as keys

    Example:
        >>> convert_vector_keys_to_canonical({"agency": 1.0, "fairness": 0.5, "belonging": -0.3})
        {"agency": 1.0, "perceived_justice": 0.5, "belonging": -0.3}
    """
    return {
        translate_axis_name(k): v
        for k, v in vector.items()
    }


def convert_vector_keys_to_internal(vector: Dict[str, float]) -> Dict[str, float]:
    """
    Convert a vector dict from canonical names to internal names.

    Args:
        vector: Dictionary with canonical axis names as keys

    Returns:
        Dictionary with internal axis names as keys (for storage)

    Example:
        >>> convert_vector_keys_to_internal({"agency": 1.0, "perceived_justice": 0.5, "belonging": -0.3})
        {"agency": 1.0, "fairness": 0.5, "belonging": -0.3}
    """
    return {
        translate_axis_name(k, to_internal=True): v
        for k, v in vector.items()
    }


def is_valid_axis_name(name: str) -> bool:
    """
    Check if a name is a valid axis name (any alias, canonical, or internal).

    Args:
        name: Name to check

    Returns:
        True if valid axis name
    """
    name_lower = name.lower().strip()
    return (
        name_lower in _ALIAS_TO_CANONICAL or
        name_lower in _INTERNAL_TO_CANONICAL or
        name_lower in AXIS_CONFIG
    )


def get_axis_config(name: str) -> Optional[Dict[str, Any]]:
    """
    Get full configuration for an axis.

    Args:
        name: Axis name (can be any alias)

    Returns:
        Full axis configuration dictionary or None if not found
    """
    canonical = translate_axis_name(name)
    return AXIS_CONFIG.get(canonical)


# Test the module if run directly
if __name__ == "__main__":
    print("Axis Configuration Test")
    print("=" * 50)

    # Test translations
    test_names = ["fairness", "perceived_justice", "justice", "agency", "belonging"]

    print("\nTranslation tests:")
    for name in test_names:
        canonical = translate_axis_name(name)
        internal = translate_axis_name(name, to_internal=True)
        display = get_axis_display_name(name)
        print(f"  '{name}' -> canonical: '{canonical}', internal: '{internal}', display: '{display}'")

    # Test vector conversion
    print("\nVector conversion test:")
    internal_vector = {"agency": 1.0, "fairness": 0.5, "belonging": -0.3}
    canonical_vector = convert_vector_keys_to_canonical(internal_vector)
    print(f"  Internal: {internal_vector}")
    print(f"  Canonical: {canonical_vector}")

    # Test deprecation notices
    print("\nDeprecation notices:")
    for name in ["fairness", "perceived_justice", "agency"]:
        notice = get_axis_deprecation_notice(name)
        if notice:
            print(f"  '{name}': {notice[:80]}...")
        else:
            print(f"  '{name}': No deprecation notice")

    print("\nAxis descriptions:")
    for axis in CANONICAL_AXES:
        print(f"  {axis}: {get_axis_description(axis)}")
