"""
Visualization Utilities for TCE

ASCII-based visualizations for terminal output.
No external dependencies required.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def bar_chart(
    data: Dict[str, float],
    width: int = 40,
    show_values: bool = True,
    title: Optional[str] = None
) -> str:
    """
    Create an ASCII horizontal bar chart.

    Args:
        data: Dict mapping labels to values (0-1 for percentages)
        width: Width of the bar area
        show_values: Whether to show numeric values
        title: Optional title

    Returns:
        ASCII bar chart string
    """
    lines = []

    if title:
        lines.append(title)
        lines.append("-" * (width + 20))

    # Find max label length for alignment
    max_label = max(len(str(k)) for k in data.keys()) if data else 0

    for label, value in data.items():
        # Clamp value to 0-1
        value = max(0, min(1, value))

        # Calculate bar length
        bar_len = int(value * width)
        bar = "█" * bar_len + "░" * (width - bar_len)

        # Format line
        label_str = str(label).ljust(max_label)
        if show_values:
            lines.append(f"  {label_str} │{bar}│ {value:.0%}")
        else:
            lines.append(f"  {label_str} │{bar}│")

    return "\n".join(lines)


def comparison_chart(
    baseline: Dict[str, float],
    treatment: Dict[str, float],
    baseline_label: str = "Baseline",
    treatment_label: str = "Treatment",
    width: int = 30,
    title: Optional[str] = None
) -> str:
    """
    Create a side-by-side comparison chart.

    Args:
        baseline: Dict mapping labels to baseline values
        treatment: Dict mapping labels to treatment values
        baseline_label: Label for baseline
        treatment_label: Label for treatment
        width: Width of each bar area
        title: Optional title

    Returns:
        ASCII comparison chart string
    """
    lines = []

    if title:
        lines.append(title)
        lines.append("=" * (width * 2 + 30))

    # Get all keys
    all_keys = list(baseline.keys() | treatment.keys())

    # Find max label length
    max_label = max(len(str(k)) for k in all_keys) if all_keys else 0

    # Header
    lines.append(f"  {''.ljust(max_label)} │ {baseline_label.center(width)} │ {treatment_label.center(width)} │ Δ")
    lines.append(f"  {'─' * max_label}─┼{'─' * (width + 2)}┼{'─' * (width + 2)}┼────")

    for key in sorted(all_keys):
        b_val = baseline.get(key, 0)
        t_val = treatment.get(key, 0)
        delta = t_val - b_val

        # Create mini bars
        b_bar_len = int(b_val * width)
        t_bar_len = int(t_val * width)

        b_bar = "█" * b_bar_len + "░" * (width - b_bar_len)
        t_bar = "█" * t_bar_len + "░" * (width - t_bar_len)

        # Delta indicator
        if delta > 0.05:
            delta_str = f"+{delta:.0%}"
            delta_icon = "↑"
        elif delta < -0.05:
            delta_str = f"{delta:.0%}"
            delta_icon = "↓"
        else:
            delta_str = "="
            delta_icon = "→"

        label_str = str(key).ljust(max_label)
        lines.append(f"  {label_str} │ {b_bar} │ {t_bar} │ {delta_icon}{delta_str}")

    return "\n".join(lines)


def confidence_distribution(
    confidences: List[float],
    bins: int = 10,
    width: int = 40,
    title: Optional[str] = None
) -> str:
    """
    Create an ASCII histogram of confidence scores.

    Args:
        confidences: List of confidence values (0-1)
        bins: Number of histogram bins
        width: Width of the histogram bars
        title: Optional title

    Returns:
        ASCII histogram string
    """
    lines = []

    if title:
        lines.append(title)
        lines.append("-" * (width + 20))

    if not confidences:
        lines.append("  (no data)")
        return "\n".join(lines)

    # Create histogram
    bin_width = 1.0 / bins
    counts = [0] * bins

    for c in confidences:
        bin_idx = min(int(c / bin_width), bins - 1)
        counts[bin_idx] += 1

    max_count = max(counts) if counts else 1

    # Draw histogram
    for i, count in enumerate(counts):
        lower = i * bin_width
        upper = (i + 1) * bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len

        label = f"{lower:.0%}-{upper:.0%}"
        lines.append(f"  {label:>10} │{bar:<{width}}│ {count}")

    # Stats
    lines.append("")
    lines.append(f"  n={len(confidences)}, mean={sum(confidences)/len(confidences):.2f}, min={min(confidences):.2f}, max={max(confidences):.2f}")

    return "\n".join(lines)


def isotope_dashboard(
    isotope_rates: Dict[str, float],
    target_rate: float = 0.95,
    width: int = 40
) -> str:
    """
    Create a dashboard showing isotope trigger rates vs targets.

    Args:
        isotope_rates: Dict mapping isotope IDs to trigger rates
        target_rate: Target rate (shown as marker)
        width: Width of the bar area

    Returns:
        ASCII dashboard string
    """
    lines = []
    lines.append("ISOTOPE COVERAGE DASHBOARD")
    lines.append("=" * (width + 25))

    # Symbol mapping
    symbols = {
        "skeptic_premise": "Σₚ Premise",
        "skeptic_method": "Σₘ Method",
        "skeptic_source": "Σₛ Source",
        "skeptic_stats": "Σₜ Stats",
    }

    target_pos = int(target_rate * width)

    for isotope_id, rate in sorted(isotope_rates.items()):
        label = symbols.get(isotope_id, isotope_id)
        rate = max(0, min(1, rate))

        bar_len = int(rate * width)

        # Build bar with target marker
        bar_chars = ["░"] * width
        for i in range(bar_len):
            bar_chars[i] = "█"

        # Add target marker
        if target_pos < width:
            bar_chars[target_pos] = "┃" if bar_len < target_pos else "█"

        bar = "".join(bar_chars)

        # Status icon
        if rate >= target_rate:
            status = "✓"
        elif rate >= 0.8:
            status = "○"
        else:
            status = "✗"

        lines.append(f"  {status} {label:12} │{bar}│ {rate:.0%}")

    lines.append(f"  {'Target':>14} {'─' * target_pos}┴{'─' * (width - target_pos - 1)} {target_rate:.0%}")

    return "\n".join(lines)


def spark_line(values: List[float], width: int = 20) -> str:
    """
    Create a tiny sparkline from values.

    Args:
        values: List of values
        width: Number of characters

    Returns:
        Sparkline string
    """
    if not values:
        return ""

    # Resample to width
    if len(values) > width:
        step = len(values) / width
        resampled = [values[int(i * step)] for i in range(width)]
    else:
        resampled = values

    # Normalize to 0-1
    min_val = min(resampled)
    max_val = max(resampled)
    if max_val == min_val:
        normalized = [0.5] * len(resampled)
    else:
        normalized = [(v - min_val) / (max_val - min_val) for v in resampled]

    # Map to sparkline characters
    chars = " ▁▂▃▄▅▆▇█"
    return "".join(chars[int(v * (len(chars) - 1))] for v in normalized)
