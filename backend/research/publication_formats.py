"""
Publication-Ready Output Formats for Academic Research.

Generates publication-quality outputs including:
- LaTeX tables (for Nature, Science, PNAS, NeurIPS styles)
- Forest plots for effect sizes
- Phase transition diagrams
- Manifold projection visualizations
- Statistical summaries

Target venues:
1. Computational Linguistics / TACL
2. NeurIPS / ICML
3. Cognition / Psychological Review
4. PNAS / Nature Human Behaviour
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional matplotlib import for visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - visualizations disabled")


class JournalStyle(Enum):
    """Publication style presets."""
    NATURE = "nature"
    SCIENCE = "science"
    PNAS = "pnas"
    NEURIPS = "neurips"
    ICML = "icml"
    ACL = "acl"
    TACL = "tacl"
    COGNITION = "cognition"
    PSYCH_REVIEW = "psychological_review"


# Style configurations for different journals
STYLE_CONFIGS = {
    JournalStyle.NATURE: {
        "font_family": "sans-serif",
        "font_size": 7,
        "figure_width": 3.5,  # inches (single column)
        "figure_height": 2.5,
        "line_width": 0.5,
        "marker_size": 3,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "dpi": 300,
    },
    JournalStyle.NEURIPS: {
        "font_family": "serif",
        "font_size": 9,
        "figure_width": 5.5,
        "figure_height": 3.5,
        "line_width": 1.0,
        "marker_size": 5,
        "colors": ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc"],
        "dpi": 300,
    },
    JournalStyle.ACL: {
        "font_family": "serif",
        "font_size": 10,
        "figure_width": 3.25,
        "figure_height": 2.5,
        "line_width": 1.0,
        "marker_size": 4,
        "colors": ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"],
        "dpi": 300,
    },
}

# Default style
STYLE_CONFIGS[JournalStyle.SCIENCE] = STYLE_CONFIGS[JournalStyle.NATURE]
STYLE_CONFIGS[JournalStyle.PNAS] = STYLE_CONFIGS[JournalStyle.NATURE]
STYLE_CONFIGS[JournalStyle.ICML] = STYLE_CONFIGS[JournalStyle.NEURIPS]
STYLE_CONFIGS[JournalStyle.TACL] = STYLE_CONFIGS[JournalStyle.ACL]
STYLE_CONFIGS[JournalStyle.COGNITION] = STYLE_CONFIGS[JournalStyle.NEURIPS]
STYLE_CONFIGS[JournalStyle.PSYCH_REVIEW] = STYLE_CONFIGS[JournalStyle.NEURIPS]


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_table(
    data: List[Dict[str, Any]],
    columns: List[str],
    headers: List[str],
    caption: str,
    label: str,
    style: JournalStyle = JournalStyle.NEURIPS,
    notes: Optional[str] = None,
    significance_column: Optional[str] = None
) -> str:
    """
    Generate a publication-ready LaTeX table.

    Args:
        data: List of row dictionaries
        columns: Column keys to include
        headers: Column header labels
        caption: Table caption
        label: LaTeX label for cross-referencing
        style: Journal style preset
        notes: Optional table notes
        significance_column: Column containing significance indicators

    Returns:
        Complete LaTeX table code
    """
    # Build column specification
    col_spec = "l" + "c" * (len(columns) - 1)

    # Build header row
    header_row = " & ".join(headers) + " \\\\"

    # Build data rows
    rows = []
    for row_dict in data:
        cells = []
        for col in columns:
            value = row_dict.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            elif isinstance(value, int):
                cells.append(str(value))
            else:
                cells.append(str(value))
        rows.append(" & ".join(cells) + " \\\\")

    rows_str = "\n".join(rows)

    # Add notes if provided
    notes_str = ""
    if notes:
        notes_str = f"\n\\\\[-1ex]\n\\multicolumn{{{len(columns)}}}{{l}}{{\\footnotesize {notes}}}"

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header_row}
\\midrule
{rows_str}
\\bottomrule{notes_str}
\\end{{tabular}}
\\end{{table}}"""


def generate_effect_size_table(
    results: List[Dict[str, Any]],
    caption: str = "Grammar deletion effect sizes",
    label: str = "tab:effect_sizes"
) -> str:
    """
    Generate LaTeX table for grammar deletion effect sizes.

    Expects results with keys: feature_name, d, confidence_interval, classification
    """
    rows = []
    for r in results:
        ci = r.get("confidence_interval", [0, 0])
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        sig = "$^*$" if r.get("is_significant", False) else ""

        rows.append({
            "feature": r.get("feature_name", ""),
            "d": f"{r.get('d', 0):.3f}{sig}",
            "ci": ci_str,
            "class": r.get("classification", "")
        })

    return generate_latex_table(
        data=rows,
        columns=["feature", "d", "ci", "class"],
        headers=["Feature", "Cohen's $d$", "95\\% CI", "Classification"],
        caption=caption,
        label=label,
        notes="$^*$ indicates 95\\% CI excludes zero."
    )


def generate_comparison_table(
    human_results: Dict[str, float],
    ai_results: Dict[str, float],
    metrics: List[str],
    caption: str = "Human vs AI coordination comparison",
    label: str = "tab:human_ai_comparison"
) -> str:
    """Generate comparison table between human and AI communication."""
    rows = []
    for metric in metrics:
        rows.append({
            "metric": metric,
            "human": human_results.get(metric, 0),
            "ai": ai_results.get(metric, 0),
            "diff": human_results.get(metric, 0) - ai_results.get(metric, 0)
        })

    return generate_latex_table(
        data=rows,
        columns=["metric", "human", "ai", "diff"],
        headers=["Metric", "Human", "AI", "Difference"],
        caption=caption,
        label=label
    )


# =============================================================================
# Statistical Summary Generation
# =============================================================================

@dataclass
class StatisticalSummary:
    """Summary statistics for publication."""

    n: int
    mean: float
    std: float
    median: float
    iqr: Tuple[float, float]
    ci_95: Tuple[float, float]
    min_val: float
    max_val: float

    def to_latex(self, precision: int = 3) -> str:
        """Format as inline LaTeX."""
        return (
            f"$M = {self.mean:.{precision}f}$, "
            f"$SD = {self.std:.{precision}f}$, "
            f"95\\% CI [{self.ci_95[0]:.{precision}f}, {self.ci_95[1]:.{precision}f}], "
            f"$N = {self.n}$"
        )

    def to_text(self, precision: int = 3) -> str:
        """Format as plain text."""
        return (
            f"M = {self.mean:.{precision}f}, "
            f"SD = {self.std:.{precision}f}, "
            f"95% CI [{self.ci_95[0]:.{precision}f}, {self.ci_95[1]:.{precision}f}], "
            f"N = {self.n}"
        )


def compute_summary_statistics(data: np.ndarray) -> StatisticalSummary:
    """Compute publication-ready summary statistics."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    median = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])

    # 95% CI for mean
    se = std / np.sqrt(n)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    return StatisticalSummary(
        n=n,
        mean=mean,
        std=std,
        median=median,
        iqr=(q1, q3),
        ci_95=(ci_low, ci_high),
        min_val=np.min(data),
        max_val=np.max(data)
    )


def generate_results_paragraph(
    effect_sizes: List[Dict],
    comparisons: List[Dict],
    key_finding: str
) -> str:
    """
    Generate a results paragraph in academic style.

    Returns formatted text suitable for a Results section.
    """
    # Find most significant effects
    significant = [e for e in effect_sizes if e.get("is_significant", False)]
    largest = max(effect_sizes, key=lambda x: abs(x.get("d", 0)))

    paragraph = f"""
{key_finding}

The grammar deletion analysis revealed {len(significant)} significant effects
out of {len(effect_sizes)} features tested. The largest effect was observed for
{largest.get('feature_name', 'unknown')} ($d = {largest.get('d', 0):.2f}$,
95\\% CI [{largest.get('confidence_interval', [0,0])[0]:.2f},
{largest.get('confidence_interval', [0,0])[1]:.2f}]),
classified as {largest.get('classification', 'unknown')}.

Features were classified as: decorative ($d < 0.2$, {sum(1 for e in effect_sizes if e.get('classification') == 'decorative')} features),
modifying ($0.2 \\leq d < 0.5$, {sum(1 for e in effect_sizes if e.get('classification') == 'modifying')} features),
necessary ($0.5 \\leq d < 0.8$, {sum(1 for e in effect_sizes if e.get('classification') == 'necessary')} features),
and critical ($d \\geq 0.8$, {sum(1 for e in effect_sizes if e.get('classification') == 'critical')} features).
"""
    return paragraph.strip()


# =============================================================================
# Visualization Generation
# =============================================================================

class AcademicVisualization:
    """Publication-quality figure generation."""

    def __init__(self, style: JournalStyle = JournalStyle.NEURIPS):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualizations")

        self.style = style
        self.config = STYLE_CONFIGS.get(style, STYLE_CONFIGS[JournalStyle.NEURIPS])
        self._apply_style()

    def _apply_style(self):
        """Apply journal style to matplotlib."""
        plt.rcParams.update({
            'font.family': self.config['font_family'],
            'font.size': self.config['font_size'],
            'figure.figsize': (self.config['figure_width'], self.config['figure_height']),
            'figure.dpi': self.config['dpi'],
            'axes.linewidth': self.config['line_width'],
            'lines.linewidth': self.config['line_width'],
            'lines.markersize': self.config['marker_size'],
        })

    def manifold_projection_3d(
        self,
        points: List[Dict[str, Any]],
        color_by: str = "mode",
        save_path: Optional[str] = None
    ) -> Any:
        """
        Create 3D scatter plot of manifold positions.

        Args:
            points: List of dicts with 'coords' (agency, justice, belonging) and optional 'mode'
            color_by: Field to color points by
            save_path: Optional path to save figure

        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(self.config['figure_width'] * 1.5,
                                   self.config['figure_height'] * 1.5))
        ax = fig.add_subplot(111, projection='3d')

        # Extract coordinates
        coords = np.array([p['coords'] for p in points])

        # Color mapping
        if color_by and color_by in points[0]:
            categories = list(set(p[color_by] for p in points))
            colors = [self.config['colors'][categories.index(p[color_by]) % len(self.config['colors'])]
                     for p in points]
        else:
            colors = self.config['colors'][0]

        # Plot
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c=colors, alpha=0.7, s=self.config['marker_size']**2)

        ax.set_xlabel('Agency')
        ax.set_ylabel('Perceived Justice')
        ax.set_zlabel('Belonging')
        ax.set_title('Coordination Manifold Projection')

        # Add legend if color_by
        if color_by and color_by in points[0]:
            handles = [mpatches.Patch(color=self.config['colors'][i], label=cat)
                      for i, cat in enumerate(categories[:len(self.config['colors'])])]
            ax.legend(handles=handles, loc='best', fontsize=self.config['font_size']-2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")

        return fig

    def forest_plot(
        self,
        effect_sizes: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> Any:
        """
        Create forest plot of effect sizes with CIs.

        Standard visualization for meta-analysis and multiple comparison studies.
        """
        n = len(effect_sizes)
        fig, ax = plt.subplots(figsize=(self.config['figure_width'],
                                        max(self.config['figure_height'], n * 0.3)))

        # Sort by effect size
        sorted_effects = sorted(effect_sizes, key=lambda x: x.get('d', 0))

        y_positions = np.arange(n)
        labels = [e.get('feature_name', f'Feature {i}') for i, e in enumerate(sorted_effects)]
        effects = [e.get('d', 0) for e in sorted_effects]
        ci_lows = [e.get('confidence_interval', [0, 0])[0] for e in sorted_effects]
        ci_highs = [e.get('confidence_interval', [0, 0])[1] for e in sorted_effects]

        # Classification colors
        colors = []
        for e in sorted_effects:
            cls = e.get('classification', '')
            if cls == 'decorative':
                colors.append(self.config['colors'][0])
            elif cls == 'modifying':
                colors.append(self.config['colors'][1])
            elif cls == 'necessary':
                colors.append(self.config['colors'][2])
            else:
                colors.append(self.config['colors'][3])

        # Plot effect sizes with CIs
        for i, (y, d, lo, hi, color) in enumerate(zip(y_positions, effects, ci_lows, ci_highs, colors)):
            ax.plot([lo, hi], [y, y], color=color, linewidth=self.config['line_width'] * 2)
            ax.plot(d, y, 'o', color=color, markersize=self.config['marker_size'])

        # Zero reference line
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

        # Threshold lines
        for thresh, label in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
            ax.axvline(x=thresh, color='lightgray', linestyle=':', linewidth=0.5)
            ax.axvline(x=-thresh, color='lightgray', linestyle=':', linewidth=0.5)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cohen's d")
        ax.set_title('Grammar Deletion Effect Sizes')

        # Legend
        handles = [
            mpatches.Patch(color=self.config['colors'][0], label='Decorative'),
            mpatches.Patch(color=self.config['colors'][1], label='Modifying'),
            mpatches.Patch(color=self.config['colors'][2], label='Necessary'),
            mpatches.Patch(color=self.config['colors'][3], label='Critical'),
        ]
        ax.legend(handles=handles, loc='best', fontsize=self.config['font_size']-2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Forest plot saved to {save_path}")

        return fig

    def phase_transition_diagram(
        self,
        control_param: np.ndarray,
        order_param: np.ndarray,
        transitions: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Create phase transition diagram.

        Shows order parameter as function of control parameter with
        transition points marked.
        """
        fig, ax = plt.subplots()

        # Plot main curve
        ax.plot(control_param, order_param, '-',
                color=self.config['colors'][0],
                linewidth=self.config['line_width'] * 2)

        # Mark transitions
        if transitions:
            for t in transitions:
                ax.axvline(x=t['transition_point'], color='red',
                          linestyle='--', linewidth=1, alpha=0.7)
                ax.annotate(f"{t.get('transition_type', '')}\n(p={t['transition_point']:.2f})",
                           xy=(t['transition_point'], ax.get_ylim()[1]),
                           fontsize=self.config['font_size']-2)

        ax.set_xlabel('Control Parameter (Compression)')
        ax.set_ylabel('Order Parameter (Legibility)')
        ax.set_title('Phase Transitions in Communication')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')

        return fig

    def trajectory_plot(
        self,
        trajectories: List[List[Dict]],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Plot multiple trajectories through the manifold.

        Useful for showing evolution or compression effects.
        """
        fig, axes = plt.subplots(1, 3, figsize=(self.config['figure_width'] * 2,
                                                 self.config['figure_height']))

        axis_names = ['Agency', 'Perceived Justice', 'Belonging']

        for traj_idx, trajectory in enumerate(trajectories):
            color = self.config['colors'][traj_idx % len(self.config['colors'])]
            label = labels[traj_idx] if labels else f'Trajectory {traj_idx}'

            coords = np.array([t['coords'] for t in trajectory])
            times = np.arange(len(coords))

            for ax_idx, ax in enumerate(axes):
                ax.plot(times, coords[:, ax_idx], '-o',
                       color=color, label=label if ax_idx == 0 else None,
                       markersize=self.config['marker_size'],
                       linewidth=self.config['line_width'])
                ax.set_xlabel('Step')
                ax.set_ylabel(axis_names[ax_idx])
                ax.set_title(axis_names[ax_idx])

        axes[0].legend(fontsize=self.config['font_size']-2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')

        return fig


# =============================================================================
# Report Generation
# =============================================================================

def generate_experiment_report(
    experiment_name: str,
    results: Dict[str, Any],
    effect_sizes: List[Dict],
    metadata: Optional[Dict] = None
) -> str:
    """
    Generate a complete experiment report in markdown.

    Suitable for supplementary materials or technical reports.
    """
    timestamp = datetime.now().isoformat()

    report = f"""# Experiment Report: {experiment_name}

Generated: {timestamp}

## Metadata
"""

    if metadata:
        for key, value in metadata.items():
            report += f"- **{key}**: {value}\n"

    report += f"""
## Summary Statistics

{json.dumps(results.get('summary', {}), indent=2)}

## Effect Sizes

| Feature | Cohen's d | 95% CI | Classification |
|---------|-----------|--------|----------------|
"""

    for e in effect_sizes:
        ci = e.get('confidence_interval', [0, 0])
        sig = "*" if e.get('is_significant', False) else ""
        report += f"| {e.get('feature_name', '')} | {e.get('d', 0):.3f}{sig} | [{ci[0]:.2f}, {ci[1]:.2f}] | {e.get('classification', '')} |\n"

    report += f"""
## Key Findings

{results.get('key_findings', 'No key findings recorded.')}

## Methodology

{results.get('methodology', 'No methodology recorded.')}

## Raw Results

```json
{json.dumps(results, indent=2, default=str)}
```
"""

    return report


def generate_supplementary_materials(
    experiments: List[Dict],
    output_dir: str = "supplementary"
) -> Dict[str, str]:
    """
    Generate complete supplementary materials package.

    Returns dict mapping filename to content.
    """
    materials = {}

    # Main supplementary document
    main_doc = """# Supplementary Materials

## Contents

1. [Experiment Details](#experiment-details)
2. [Statistical Methods](#statistical-methods)
3. [Complete Results](#complete-results)
4. [Reproducibility Information](#reproducibility)

---

## Experiment Details

"""

    for i, exp in enumerate(experiments, 1):
        main_doc += f"### Experiment {i}: {exp.get('name', 'Unnamed')}\n\n"
        main_doc += f"{exp.get('description', 'No description.')}\n\n"

    main_doc += """
## Statistical Methods

### Effect Size Calculation

Cohen's d was calculated using pooled standard deviation:

$$d = \\frac{M_1 - M_2}{SD_{pooled}}$$

where $SD_{pooled} = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

### Multiple Comparison Correction

We applied Holm-Bonferroni correction to control family-wise error rate (FWER).

### Bootstrap Confidence Intervals

95% BCa bootstrap confidence intervals were computed using 1000 resamples.

---

## Reproducibility

All code and data are available at: [repository URL]

Random seeds used: 42 (consistent across all experiments)
"""

    materials["supplementary_main.md"] = main_doc

    # Individual experiment files
    for i, exp in enumerate(experiments, 1):
        exp_report = generate_experiment_report(
            exp.get('name', f'Experiment {i}'),
            exp.get('results', {}),
            exp.get('effect_sizes', []),
            exp.get('metadata', {})
        )
        materials[f"experiment_{i}_details.md"] = exp_report

    return materials
