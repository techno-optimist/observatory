import { App } from "@modelcontextprotocol/ext-apps";

interface ForceFieldResult {
  text?: string;
  attractor_strength: number;
  detractor_strength: number;
  net_force: number;
  force_direction: string;
  force_quadrant: string;
  quadrant_description?: string;
  energy_level?: string;
  primary_attractor: string;
  secondary_attractor?: string;
  primary_detractor: string;
  secondary_detractor?: string;
  attractor_scores: Record<string, number>;
  detractor_scores: Record<string, number>;
}

const ATTRACTOR_TARGETS = [
  "AUTONOMY",
  "COMMUNITY",
  "JUSTICE",
  "MEANING",
  "SECURITY",
  "RECOGNITION",
];

const DETRACTOR_SOURCES = [
  "OPPRESSION",
  "ISOLATION",
  "INJUSTICE",
  "MEANINGLESSNESS",
  "INSTABILITY",
  "INVISIBILITY",
];

class ForceFieldViewer {
  private app: App;

  constructor() {
    this.app = new App({ name: "Observatory Force Field", version: "1.0.0" });
    this.init();
  }

  private init() {
    this.app.connect();
    this.app.ontoolresult = (result) => this.handleToolResult(result);

    // Initialize empty bars
    this.renderBars("attractor-bars", ATTRACTOR_TARGETS, {}, "attractor");
    this.renderBars("detractor-bars", DETRACTOR_SOURCES, {}, "detractor");
  }

  private handleToolResult(result: any) {
    let forceData: ForceFieldResult | null = null;

    if (result.attractor_strength !== undefined) {
      forceData = result as ForceFieldResult;
    } else if (result.content) {
      const textContent = result.content.find((c: any) => c.type === "text")?.text;
      if (textContent) {
        try {
          forceData = JSON.parse(textContent);
        } catch {
          // Try to parse from formatted output
          const attractorMatch = textContent.match(/Attractor Strength[:\s]*([-\d.]+)/i);
          const detractorMatch = textContent.match(/Detractor Strength[:\s]*([-\d.]+)/i);
          const quadrantMatch = textContent.match(/Quadrant[:\s]*(\w+)/i);
          const primaryAttrMatch = textContent.match(/Primary Attractor[:\s]*(\w+)/i);
          const primaryDetrMatch = textContent.match(/Primary Detractor[:\s]*(\w+)/i);

          if (attractorMatch && detractorMatch) {
            forceData = {
              attractor_strength: parseFloat(attractorMatch[1]),
              detractor_strength: parseFloat(detractorMatch[1]),
              net_force: parseFloat(attractorMatch[1]) - parseFloat(detractorMatch[1]),
              force_direction: "BALANCED",
              force_quadrant: quadrantMatch?.[1] || "STASIS",
              primary_attractor: primaryAttrMatch?.[1] || "None",
              primary_detractor: primaryDetrMatch?.[1] || "None",
              attractor_scores: {},
              detractor_scores: {},
            };
          }
        }
      }
    }

    if (forceData) {
      document.getElementById("loading")!.style.display = "none";
      this.render(forceData);
    }
  }

  private render(data: ForceFieldResult) {
    // Update quadrant badge
    const badge = document.getElementById("quadrant-badge")!;
    badge.textContent = data.force_quadrant.replace(/_/g, " ");
    badge.className = `quadrant-badge ${data.force_quadrant.toLowerCase()}`;

    // Highlight active quadrant
    document.querySelectorAll(".quadrant-cell").forEach((cell) => {
      cell.classList.remove("active");
      if ((cell as HTMLElement).dataset.quadrant === data.force_quadrant.toLowerCase()) {
        cell.classList.add("active");
      }
    });

    // Update metrics
    document.getElementById("attractor-strength")!.textContent = data.attractor_strength.toFixed(3);
    document.getElementById("detractor-strength")!.textContent = data.detractor_strength.toFixed(3);

    const netForce = document.getElementById("net-force")!;
    netForce.textContent = data.net_force.toFixed(3);
    netForce.className = `metric-value ${data.net_force > 0 ? "positive" : data.net_force < 0 ? "negative" : ""}`;

    document.getElementById("force-direction")!.textContent = data.force_direction;
    document.getElementById("primary-attractor")!.textContent = `Primary: ${data.primary_attractor || "None"}`;
    document.getElementById("primary-detractor")!.textContent = `Primary: ${data.primary_detractor || "None"}`;

    // Update bars
    this.renderBars("attractor-bars", ATTRACTOR_TARGETS, data.attractor_scores || {}, "attractor");
    this.renderBars("detractor-bars", DETRACTOR_SOURCES, data.detractor_scores || {}, "detractor");

    // Show text if available
    if (data.text) {
      document.getElementById("text-preview")!.style.display = "block";
      document.getElementById("text-content")!.textContent = data.text.slice(0, 300) + (data.text.length > 300 ? "..." : "");
    }

    // Update interpretation
    const interpretation = this.generateInterpretation(data);
    document.getElementById("interpretation")!.textContent = interpretation;
  }

  private renderBars(
    containerId: string,
    labels: string[],
    scores: Record<string, number>,
    type: "attractor" | "detractor"
  ) {
    const container = document.getElementById(containerId)!;

    container.innerHTML = labels.map((label) => {
      const score = scores[label] ?? scores[label.toLowerCase()] ?? 0;
      const pct = Math.min(100, score * 100);

      return `
        <div class="force-bar">
          <div class="force-label">${label.toLowerCase()}</div>
          <div class="bar-container">
            <div class="bar-fill ${type}" style="width: ${pct}%"></div>
            <div class="bar-value">${score.toFixed(2)}</div>
          </div>
        </div>
      `;
    }).join("");
  }

  private generateInterpretation(data: ForceFieldResult): string {
    const parts: string[] = [];

    // Quadrant interpretation
    switch (data.force_quadrant.toLowerCase()) {
      case "active_transformation":
        parts.push("This narrative exhibits active transformation - simultaneously drawn toward positive states while fleeing negative ones. High energy, dynamic positioning.");
        break;
      case "pure_aspiration":
        parts.push("This narrative shows pure aspiration - drawn toward positive goals without strong aversion to current state. Optimistic, goal-oriented framing.");
        break;
      case "pure_escape":
        parts.push("This narrative reflects pure escape - fleeing from negative states without clear positive direction. May indicate distress or reactive positioning.");
        break;
      case "stasis":
        parts.push("This narrative shows stasis - low motivational force in either direction. May indicate acceptance, resignation, or neutral observation.");
        break;
    }

    // Attractor interpretation
    if (data.primary_attractor && data.primary_attractor !== "None") {
      parts.push(`Primary draw toward ${data.primary_attractor.toLowerCase()}.`);
    }

    // Detractor interpretation
    if (data.primary_detractor && data.primary_detractor !== "None") {
      parts.push(`Primary aversion to ${data.primary_detractor.toLowerCase()}.`);
    }

    // Net force
    if (Math.abs(data.net_force) > 0.3) {
      parts.push(data.net_force > 0
        ? "Net positive motivation - aspirational framing dominates."
        : "Net negative motivation - escape/avoidance framing dominates.");
    }

    return parts.join(" ");
  }
}

// Initialize
new ForceFieldViewer();
