import { App } from "@modelcontextprotocol/ext-apps";

interface GroupData {
  name: string;
  count: number;
  centroid: {
    agency: number;
    perceived_justice?: number;
    fairness?: number;
    belonging: number;
  };
  mode_distribution: Record<string, number>;
}

interface GapAnalysis {
  delta_agency: number;
  delta_perceived_justice?: number;
  delta_fairness?: number;
  delta_belonging: number;
  total_gap: number;
  interpretation?: string;
}

interface CompareResult {
  group_a: GroupData;
  group_b: GroupData;
  gap_analysis: GapAnalysis;
}

class GapAnalysisViewer {
  private app: App;

  constructor() {
    this.app = new App({ name: "Observatory Gap Analysis", version: "1.0.0" });
    this.init();
  }

  private init() {
    this.app.connect();
    this.app.ontoolresult = (result) => this.handleToolResult(result);
  }

  private handleToolResult(result: any) {
    let compareData: CompareResult | null = null;

    if (result.group_a && result.group_b) {
      compareData = result as CompareResult;
    } else if (result.content) {
      const textContent = result.content.find((c: any) => c.type === "text")?.text;
      if (textContent) {
        try {
          compareData = JSON.parse(textContent);
        } catch {
          // Try to parse from formatted output
          const groupAMatch = textContent.match(/###\s*([^\n(]+)\s*\((\d+)\s*texts?\)/i);
          const groupBMatch = textContent.match(/###\s*([^\n(]+)\s*\((\d+)\s*texts?\)[^#]*###\s*([^\n(]+)\s*\((\d+)\s*texts?\)/i);

          const centroidAMatch = textContent.match(/Centroid:\s*A=([-\d.]+),\s*PJ=([-\d.]+),\s*B=([-\d.]+)/);
          const totalGapMatch = textContent.match(/Total gap[:\s]*([-\d.]+)/i);
          const deltaAgencyMatch = textContent.match(/Agency delta[:\s]*([-\d.]+)/i);
          const deltaJusticeMatch = textContent.match(/(?:Justice|Perceived Justice) delta[:\s]*([-\d.]+)/i);
          const deltaBelongingMatch = textContent.match(/Belonging delta[:\s]*([-\d.]+)/i);

          if (groupAMatch && centroidAMatch) {
            compareData = {
              group_a: {
                name: groupAMatch[1].trim(),
                count: parseInt(groupAMatch[2]),
                centroid: {
                  agency: parseFloat(centroidAMatch[1]),
                  perceived_justice: parseFloat(centroidAMatch[2]),
                  belonging: parseFloat(centroidAMatch[3]),
                },
                mode_distribution: {},
              },
              group_b: {
                name: groupBMatch?.[3]?.trim() || "Group B",
                count: parseInt(groupBMatch?.[4] || "0"),
                centroid: { agency: 0, perceived_justice: 0, belonging: 0 },
                mode_distribution: {},
              },
              gap_analysis: {
                delta_agency: parseFloat(deltaAgencyMatch?.[1] || "0"),
                delta_perceived_justice: parseFloat(deltaJusticeMatch?.[1] || "0"),
                delta_belonging: parseFloat(deltaBelongingMatch?.[1] || "0"),
                total_gap: parseFloat(totalGapMatch?.[1] || "0"),
              },
            };
          }
        }
      }
    }

    if (compareData) {
      document.getElementById("loading")!.style.display = "none";
      document.getElementById("content")!.style.display = "block";
      this.render(compareData);
    }
  }

  private render(data: CompareResult) {
    const { group_a, group_b, gap_analysis } = data;

    // Update group names
    document.getElementById("group-a-name")!.textContent = group_a.name;
    document.getElementById("group-a-count")!.textContent = `(${group_a.count})`;
    document.getElementById("group-b-name")!.textContent = group_b.name;
    document.getElementById("group-b-count")!.textContent = `(${group_b.count})`;

    // Get justice values (handle both naming conventions)
    const justiceA = group_a.centroid.perceived_justice ?? group_a.centroid.fairness ?? 0;
    const justiceB = group_b.centroid.perceived_justice ?? group_b.centroid.fairness ?? 0;
    const deltaJustice = gap_analysis.delta_perceived_justice ?? gap_analysis.delta_fairness ?? 0;

    // Render dimension comparisons
    this.renderDimensionBars("agency", group_a.centroid.agency, group_b.centroid.agency, gap_analysis.delta_agency);
    this.renderDimensionBars("justice", justiceA, justiceB, deltaJustice);
    this.renderDimensionBars("belonging", group_a.centroid.belonging, group_b.centroid.belonging, gap_analysis.delta_belonging);

    // Render total gap
    document.getElementById("total-gap")!.textContent = gap_analysis.total_gap.toFixed(2);

    // Gap level interpretation
    const gapLevel = document.getElementById("gap-level")!;
    const gapFill = document.getElementById("gap-fill")!;
    const gapPct = Math.min(100, (gap_analysis.total_gap / 2) * 100);
    gapFill.style.width = `${gapPct}%`;

    if (gap_analysis.total_gap < 0.3) {
      gapLevel.textContent = "Low gap - values appear well aligned between groups.";
    } else if (gap_analysis.total_gap < 0.7) {
      gapLevel.textContent = "Moderate gap - some misalignment detected between groups.";
    } else {
      gapLevel.textContent = "High gap - significant divergence in narrative positioning.";
    }

    // Render mode distribution comparison
    this.renderModeComparison(group_a.mode_distribution, group_b.mode_distribution);

    // Render interpretation
    document.getElementById("interpretation")!.textContent =
      gap_analysis.interpretation || this.generateInterpretation(data);
  }

  private renderDimensionBars(
    dimension: string,
    valueA: number,
    valueB: number,
    delta: number
  ) {
    // Update delta badge
    document.getElementById(`delta-${dimension}`)!.textContent = `Δ ${Math.abs(delta).toFixed(2)}`;

    // Render comparison bars
    const container = document.getElementById(`bars-${dimension}`)!;

    // Normalize to 0-100 scale (from -2 to +2)
    const pctA = ((valueA + 2) / 4) * 50; // 0-50 scale from center
    const pctB = ((valueB + 2) / 4) * 50;

    container.innerHTML = `
      <div class="comparison-row">
        <div class="bar-label">${this.capitalize(dimension)}</div>
        <div class="bar-container">
          <div class="bar-center"></div>
          <div class="bar-fill a" style="width: ${Math.abs(valueA) * 25}%; ${valueA < 0 ? 'left: auto; right: 50%; transform: scaleX(1);' : ''}"></div>
          <div class="bar-fill b" style="width: ${Math.abs(valueB) * 25}%; ${valueB < 0 ? 'right: auto; left: 50%; transform: scaleX(-1);' : 'right: 50%; left: auto; transform: scaleX(-1);'}"></div>
        </div>
      </div>
      <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); padding: 0 68px;">
        <span style="color: var(--group-a);">${valueA.toFixed(2)}</span>
        <span style="color: var(--group-b);">${valueB.toFixed(2)}</span>
      </div>
    `;
  }

  private renderModeComparison(
    distA: Record<string, number>,
    distB: Record<string, number>
  ) {
    const container = document.getElementById("mode-grid")!;

    // Get all modes from both distributions
    const allModes = new Set([...Object.keys(distA), ...Object.keys(distB)]);

    if (allModes.size === 0) {
      container.innerHTML = '<div style="color: var(--text-dim); font-size: 12px;">No mode data available</div>';
      return;
    }

    // Calculate totals for percentages
    const totalA = Object.values(distA).reduce((sum, v) => sum + v, 0) || 1;
    const totalB = Object.values(distB).reduce((sum, v) => sum + v, 0) || 1;

    container.innerHTML = [...allModes].map((mode) => {
      const countA = distA[mode] || 0;
      const countB = distB[mode] || 0;
      const pctA = ((countA / totalA) * 100).toFixed(0);
      const pctB = ((countB / totalB) * 100).toFixed(0);

      return `
        <div class="mode-item">
          <div class="mode-name">${mode.replace(/_/g, " ")}</div>
          <div class="mode-bars">
            <div class="mode-bar a">${pctA}%</div>
            <div class="mode-bar b">${pctB}%</div>
          </div>
        </div>
      `;
    }).join("");
  }

  private generateInterpretation(data: CompareResult): string {
    const { group_a, group_b, gap_analysis } = data;
    const parts: string[] = [];

    // Get justice values
    const justiceA = group_a.centroid.perceived_justice ?? group_a.centroid.fairness ?? 0;
    const justiceB = group_b.centroid.perceived_justice ?? group_b.centroid.fairness ?? 0;
    const deltaJustice = gap_analysis.delta_perceived_justice ?? gap_analysis.delta_fairness ?? 0;

    // Agency interpretation
    if (gap_analysis.delta_agency > 0.3) {
      const higher = group_a.centroid.agency > group_b.centroid.agency ? group_a.name : group_b.name;
      parts.push(`${higher} shows higher agency, expressing more self-determination and control.`);
    }

    // Justice interpretation
    if (deltaJustice > 0.3) {
      const higher = justiceA > justiceB ? group_a.name : group_b.name;
      parts.push(`${higher} perceives greater system legitimacy and fair treatment.`);
    }

    // Belonging interpretation
    if (gap_analysis.delta_belonging > 0.3) {
      const higher = group_a.centroid.belonging > group_b.centroid.belonging ? group_a.name : group_b.name;
      parts.push(`${higher} shows stronger community connection and group identification.`);
    }

    // Overall gap
    if (gap_analysis.total_gap > 0.5) {
      parts.push("The substantial gap suggests fundamentally different narrative framings between these groups.");
    }

    return parts.join(" ") || `The comparison between ${group_a.name} and ${group_b.name} shows ${gap_analysis.total_gap < 0.3 ? "alignment" : "divergence"} in narrative positioning.`;
  }

  private capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
}

// Initialize
new GapAnalysisViewer();
