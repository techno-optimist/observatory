import { App } from "@modelcontextprotocol/ext-apps";

interface CohortData {
  name: string;
  count: number;
  centroid: {
    agency: number;
    perceived_justice?: number;
    fairness?: number;
    belonging: number;
  };
  mode_distribution: Record<string, number>;
  std_dev?: {
    agency: number;
    perceived_justice?: number;
    belonging: number;
  };
}

interface CohortResult {
  cohorts?: CohortData[];
  anova?: {
    significant: boolean;
    f_statistic: number;
    p_value: number;
    per_axis?: Record<string, { f_statistic: number; p_value: number }>;
  };
  interpretation?: string;
}

class CohortHeatmap {
  private app: App;
  private cohorts: CohortData[] = [];
  private colorScale: "diverging" | "sequential" = "diverging";
  private selectedMetric: "all" | "agency" | "justice" | "belonging" = "all";

  constructor() {
    this.app = new App({ name: "Observatory Cohort Heatmap", version: "1.0.0" });
    this.init();
  }

  private init() {
    this.app.connect();
    this.app.ontoolresult = (result) => this.handleToolResult(result);

    // Setup controls
    document.getElementById("metric-select")?.addEventListener("change", (e) => {
      this.selectedMetric = (e.target as HTMLSelectElement).value as typeof this.selectedMetric;
      this.render();
    });

    document.getElementById("color-scale")?.addEventListener("change", (e) => {
      this.colorScale = (e.target as HTMLSelectElement).value as typeof this.colorScale;
      this.render();
    });

    // Tooltip handling
    document.addEventListener("mousemove", (e) => {
      const tooltip = document.getElementById("tooltip")!;
      if (tooltip.classList.contains("visible")) {
        tooltip.style.left = `${e.clientX + 15}px`;
        tooltip.style.top = `${e.clientY + 15}px`;
      }
    });
  }

  private handleToolResult(result: any) {
    let cohortData: CohortResult | null = null;

    if (result.cohorts) {
      cohortData = result as CohortResult;
    } else if (result.content) {
      const textContent = result.content.find((c: any) => c.type === "text")?.text;
      if (textContent) {
        try {
          cohortData = JSON.parse(textContent);
        } catch {
          // Try to parse from formatted output
          const cohorts: CohortData[] = [];
          const cohortMatches = textContent.matchAll(/\*\*([^*]+)\*\*\s*\(n=(\d+)\)[^C]*Centroid:\s*A=([-\d.]+),\s*PJ=([-\d.]+),\s*B=([-\d.]+)/g);
          for (const match of cohortMatches) {
            cohorts.push({
              name: match[1].trim(),
              count: parseInt(match[2]),
              centroid: {
                agency: parseFloat(match[3]),
                perceived_justice: parseFloat(match[4]),
                belonging: parseFloat(match[5]),
              },
              mode_distribution: {},
            });
          }
          if (cohorts.length > 0) {
            cohortData = { cohorts };
          }
        }
      }
    }

    if (cohortData?.cohorts && cohortData.cohorts.length > 0) {
      this.cohorts = cohortData.cohorts;
      this.render();

      document.getElementById("loading")!.style.display = "none";
      document.getElementById("content")!.style.display = "block";
    }
  }

  private getColor(value: number): string {
    // value is in range -2 to +2
    const normalized = (value + 2) / 4; // 0 to 1

    if (this.colorScale === "diverging") {
      // Red (low) -> Yellow (mid) -> Green (high)
      if (normalized < 0.5) {
        // Red to Yellow
        const t = normalized * 2;
        const h = 0 + t * 60; // 0 to 60
        return `hsl(${h}, 70%, ${40 + t * 10}%)`;
      } else {
        // Yellow to Green
        const t = (normalized - 0.5) * 2;
        const h = 60 + t * 60; // 60 to 120
        return `hsl(${h}, ${70 - t * 10}%, ${50 - t * 10}%)`;
      }
    } else {
      // Sequential blue scale
      const h = 220;
      const s = 60 + normalized * 20;
      const l = 20 + normalized * 40;
      return `hsl(${h}, ${s}%, ${l}%)`;
    }
  }

  private render() {
    const header = document.getElementById("heatmap-header")!;
    const body = document.getElementById("heatmap-body")!;

    // Determine columns based on metric selection
    const columns = this.selectedMetric === "all"
      ? ["Agency", "Perceived Justice", "Belonging"]
      : [this.selectedMetric === "agency" ? "Agency" : this.selectedMetric === "justice" ? "Perceived Justice" : "Belonging"];

    // Render header
    header.innerHTML = `
      <tr>
        <th>Cohort</th>
        <th>N</th>
        ${columns.map((c) => `<th>${c}</th>`).join("")}
        <th>Dominant Mode</th>
      </tr>
    `;

    // Render rows
    body.innerHTML = this.cohorts.map((cohort, idx) => {
      const justice = cohort.centroid.perceived_justice ?? cohort.centroid.fairness ?? 0;
      const values = this.selectedMetric === "all"
        ? [cohort.centroid.agency, justice, cohort.centroid.belonging]
        : this.selectedMetric === "agency"
          ? [cohort.centroid.agency]
          : this.selectedMetric === "justice"
            ? [justice]
            : [cohort.centroid.belonging];

      const dominantMode = Object.entries(cohort.mode_distribution || {})
        .sort((a, b) => b[1] - a[1])[0]?.[0] || "N/A";

      return `
        <tr data-row="${idx}">
          <td class="cohort-name">${cohort.name}</td>
          <td>${cohort.count}</td>
          ${values.map((v, vIdx) => `
            <td class="cell"
                style="background: ${this.getColor(v)}; color: ${Math.abs(v) > 1 ? "white" : "black"}"
                data-cohort="${cohort.name}"
                data-dimension="${this.selectedMetric === 'all' ? columns[vIdx] : columns[0]}"
                data-value="${v.toFixed(3)}">
              <div class="cell-value">${v.toFixed(2)}</div>
            </td>
          `).join("")}
          <td style="text-transform: capitalize;">${dominantMode.replace(/_/g, " ")}</td>
        </tr>
      `;
    }).join("");

    // Add cell event listeners
    document.querySelectorAll(".cell").forEach((cell) => {
      cell.addEventListener("mouseenter", (e) => this.showTooltip(e as MouseEvent));
      cell.addEventListener("mouseleave", () => this.hideTooltip());
    });

    // Render summary
    this.renderSummary();
  }

  private showTooltip(e: MouseEvent) {
    const cell = e.target as HTMLElement;
    const cohortName = cell.dataset.cohort;
    const dimension = cell.dataset.dimension;
    const value = cell.dataset.value;

    const cohort = this.cohorts.find((c) => c.name === cohortName);
    if (!cohort) return;

    const tooltip = document.getElementById("tooltip")!;
    tooltip.innerHTML = `
      <div class="tooltip-title">${cohortName}</div>
      <div class="tooltip-row">
        <span class="tooltip-label">${dimension}</span>
        <span>${value}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Sample Size</span>
        <span>${cohort.count}</span>
      </div>
      ${cohort.std_dev ? `
        <div class="tooltip-row">
          <span class="tooltip-label">Std Dev</span>
          <span>${(cohort.std_dev as any)[dimension?.toLowerCase().replace(" ", "_")] ?? "N/A"}</span>
        </div>
      ` : ""}
    `;
    tooltip.classList.add("visible");
  }

  private hideTooltip() {
    document.getElementById("tooltip")!.classList.remove("visible");
  }

  private renderSummary() {
    const summary = document.getElementById("summary")!;

    if (this.cohorts.length === 0) {
      summary.innerHTML = "";
      return;
    }

    // Calculate summary stats
    const avgAgency = this.cohorts.reduce((sum, c) => sum + c.centroid.agency, 0) / this.cohorts.length;
    const avgJustice = this.cohorts.reduce((sum, c) => sum + (c.centroid.perceived_justice ?? c.centroid.fairness ?? 0), 0) / this.cohorts.length;
    const avgBelonging = this.cohorts.reduce((sum, c) => sum + c.centroid.belonging, 0) / this.cohorts.length;

    // Find highest/lowest
    const maxAgency = this.cohorts.reduce((max, c) => c.centroid.agency > max.centroid.agency ? c : max, this.cohorts[0]);
    const minAgency = this.cohorts.reduce((min, c) => c.centroid.agency < min.centroid.agency ? c : min, this.cohorts[0]);

    summary.innerHTML = `
      <div class="summary-card">
        <h3>Total Cohorts</h3>
        <div class="summary-value">${this.cohorts.length}</div>
        <div class="summary-detail">${this.cohorts.reduce((sum, c) => sum + c.count, 0)} total samples</div>
      </div>
      <div class="summary-card">
        <h3>Average Agency</h3>
        <div class="summary-value" style="color: ${this.getColor(avgAgency)}">${avgAgency.toFixed(2)}</div>
        <div class="summary-detail">Highest: ${maxAgency.name} (${maxAgency.centroid.agency.toFixed(2)})</div>
      </div>
      <div class="summary-card">
        <h3>Average Justice</h3>
        <div class="summary-value" style="color: ${this.getColor(avgJustice)}">${avgJustice.toFixed(2)}</div>
      </div>
      <div class="summary-card">
        <h3>Average Belonging</h3>
        <div class="summary-value" style="color: ${this.getColor(avgBelonging)}">${avgBelonging.toFixed(2)}</div>
      </div>
    `;
  }
}

// Initialize
new CohortHeatmap();
