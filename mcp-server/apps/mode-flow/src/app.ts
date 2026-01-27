import { App } from "@modelcontextprotocol/ext-apps";

interface ModeFlowResult {
  mode_sequence: string[];
  transition_matrix: Record<string, Record<string, number>>;
  flow_patterns?: Array<{ description: string }>;
  stable_modes: string[];
  volatile_modes: string[];
  interpretation?: string;
}

const MODE_COLORS: Record<string, string> = {
  growth_mindset: "#4ade80",
  civic_idealism: "#22c55e",
  faithful_zeal: "#16a34a",
  positive: "#4ade80",
  cynical_burnout: "#f87171",
  institutional_decay: "#ef4444",
  schismatic_doubt: "#dc2626",
  shadow: "#f87171",
  quiet_quitting: "#60a5fa",
  grid_exit: "#3b82f6",
  apostasy: "#2563eb",
  exit: "#60a5fa",
  conflicted: "#a78bfa",
  transitional: "#8b5cf6",
  neutral: "#7c3aed",
  ambivalent: "#a78bfa",
  unknown: "#888888",
};

function getModeCategory(mode: string): string {
  const m = mode.toLowerCase().replace(/\s+/g, "_");
  if (["growth_mindset", "civic_idealism", "faithful_zeal", "positive"].includes(m)) return "positive";
  if (["cynical_burnout", "institutional_decay", "schismatic_doubt", "shadow"].includes(m)) return "shadow";
  if (["quiet_quitting", "grid_exit", "apostasy", "exit"].includes(m)) return "exit";
  return "ambivalent";
}

function getModeColor(mode: string): string {
  const m = mode.toLowerCase().replace(/\s+/g, "_");
  return MODE_COLORS[m] || MODE_COLORS[getModeCategory(mode)] || MODE_COLORS.unknown;
}

class ModeFlowViewer {
  private app: App;
  private svg!: SVGSVGElement;

  constructor() {
    this.app = new App({ name: "Observatory Mode Flow", version: "1.0.0" });
    this.init();
  }

  private init() {
    this.svg = document.getElementById("sankey-svg") as unknown as SVGSVGElement;

    this.app.connect();
    this.app.ontoolresult = (result) => this.handleToolResult(result);

    // Tooltip handling
    document.addEventListener("mousemove", (e) => {
      const tooltip = document.getElementById("tooltip")!;
      if (tooltip.classList.contains("visible")) {
        tooltip.style.left = `${e.clientX + 15}px`;
        tooltip.style.top = `${e.clientY + 15}px`;
      }
    });

    window.addEventListener("resize", () => {
      // Re-render if we have data
    });
  }

  private handleToolResult(result: any) {
    let flowData: ModeFlowResult | null = null;

    if (result.mode_sequence) {
      flowData = result as ModeFlowResult;
    } else if (result.content) {
      const textContent = result.content.find((c: any) => c.type === "text")?.text;
      if (textContent) {
        try {
          flowData = JSON.parse(textContent);
        } catch {
          // Try to parse from formatted output
          const seqMatch = textContent.match(/Mode Sequence[^:]*:\s*([\w\s,->]+)/i);
          const stableMatch = textContent.match(/Stable modes[^:]*:\s*([^\n]+)/i);
          const volatileMatch = textContent.match(/Volatile modes[^:]*:\s*([^\n]+)/i);

          if (seqMatch) {
            const sequence = seqMatch[1].split(/\s*->\s*/).map((s: string) => s.trim());
            flowData = {
              mode_sequence: sequence,
              transition_matrix: {},
              stable_modes: stableMatch ? stableMatch[1].split(",").map((s: string) => s.trim()) : [],
              volatile_modes: volatileMatch ? volatileMatch[1].split(",").map((s: string) => s.trim()) : [],
            };
          }
        }
      }
    }

    if (flowData && flowData.mode_sequence.length > 0) {
      document.getElementById("loading")!.style.display = "none";
      this.render(flowData);
    }
  }

  private render(data: ModeFlowResult) {
    // Render mode sequence
    this.renderModeSequence(data.mode_sequence);

    // Render Sankey diagram
    this.renderSankey(data.mode_sequence, data.transition_matrix);

    // Render stable/volatile modes
    this.renderModeList("stable-modes", data.stable_modes);
    this.renderModeList("volatile-modes", data.volatile_modes);

    // Render patterns
    this.renderPatterns(data.flow_patterns || []);

    // Render interpretation
    document.getElementById("interpretation")!.textContent =
      data.interpretation || this.generateInterpretation(data);
  }

  private renderModeSequence(sequence: string[]) {
    const container = document.getElementById("mode-sequence")!;
    const display = sequence.slice(0, 15); // Show first 15

    container.innerHTML = display.map((mode, i) => {
      const category = getModeCategory(mode);
      const displayName = mode.replace(/_/g, " ");
      return `<span class="mode-tag ${category}">${displayName}</span>${i < display.length - 1 ? '<span class="arrow">→</span>' : ""}`;
    }).join("");

    if (sequence.length > 15) {
      container.innerHTML += `<span style="color: var(--text-dim);">...+${sequence.length - 15} more</span>`;
    }
  }

  private renderSankey(sequence: string[], matrix: Record<string, Record<string, number>>) {
    const container = this.svg.parentElement!;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear SVG
    this.svg.innerHTML = "";
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    // Count transitions
    const transitions: Map<string, number> = new Map();
    const modeCount: Map<string, number> = new Map();

    for (let i = 0; i < sequence.length - 1; i++) {
      const from = sequence[i];
      const to = sequence[i + 1];
      const key = `${from}|${to}`;
      transitions.set(key, (transitions.get(key) || 0) + 1);
      modeCount.set(from, (modeCount.get(from) || 0) + 1);
    }
    // Count last mode
    if (sequence.length > 0) {
      const last = sequence[sequence.length - 1];
      modeCount.set(last, (modeCount.get(last) || 0) + 1);
    }

    // Get unique modes in order of appearance
    const modes = [...new Set(sequence)];

    if (modes.length === 0) return;

    // Layout parameters
    const padding = { top: 40, right: 120, bottom: 40, left: 120 };
    const nodeWidth = 24;
    const nodeGap = 20;

    // Calculate node positions (simple column layout)
    const availableHeight = height - padding.top - padding.bottom;
    const totalNodes = modes.length;
    const nodeHeight = Math.min(60, (availableHeight - (totalNodes - 1) * nodeGap) / totalNodes);

    const nodePositions: Map<string, { x: number; y: number; height: number }> = new Map();

    // Left column (source nodes)
    modes.forEach((mode, i) => {
      const y = padding.top + i * (nodeHeight + nodeGap);
      nodePositions.set(`${mode}_from`, { x: padding.left, y, height: nodeHeight });
    });

    // Right column (target nodes)
    modes.forEach((mode, i) => {
      const y = padding.top + i * (nodeHeight + nodeGap);
      nodePositions.set(`${mode}_to`, { x: width - padding.right - nodeWidth, y, height: nodeHeight });
    });

    // Draw links
    const linkGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");

    transitions.forEach((count, key) => {
      const [from, to] = key.split("|");
      const fromPos = nodePositions.get(`${from}_from`)!;
      const toPos = nodePositions.get(`${to}_to`)!;

      if (!fromPos || !toPos) return;

      const strokeWidth = Math.max(2, Math.min(20, count * 3));

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const x1 = fromPos.x + nodeWidth;
      const y1 = fromPos.y + fromPos.height / 2;
      const x2 = toPos.x;
      const y2 = toPos.y + toPos.height / 2;
      const mx = (x1 + x2) / 2;

      path.setAttribute("d", `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`);
      path.setAttribute("class", "sankey-link");
      path.setAttribute("stroke", getModeColor(from));
      path.setAttribute("stroke-width", strokeWidth.toString());

      path.addEventListener("mouseenter", () => {
        const tooltip = document.getElementById("tooltip")!;
        tooltip.innerHTML = `<strong>${from.replace(/_/g, " ")} → ${to.replace(/_/g, " ")}</strong><br>${count} transition${count > 1 ? "s" : ""}`;
        tooltip.classList.add("visible");
      });

      path.addEventListener("mouseleave", () => {
        document.getElementById("tooltip")!.classList.remove("visible");
      });

      linkGroup.appendChild(path);
    });

    this.svg.appendChild(linkGroup);

    // Draw nodes
    const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");

    modes.forEach((mode) => {
      // From node (left)
      const fromPos = nodePositions.get(`${mode}_from`)!;
      this.drawNode(nodeGroup, mode, fromPos.x, fromPos.y, nodeWidth, fromPos.height, true);

      // To node (right)
      const toPos = nodePositions.get(`${mode}_to`)!;
      this.drawNode(nodeGroup, mode, toPos.x, toPos.y, nodeWidth, toPos.height, false);
    });

    this.svg.appendChild(nodeGroup);
  }

  private drawNode(
    parent: SVGGElement,
    mode: string,
    x: number,
    y: number,
    width: number,
    height: number,
    showLabel: boolean
  ) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "sankey-node");

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x.toString());
    rect.setAttribute("y", y.toString());
    rect.setAttribute("width", width.toString());
    rect.setAttribute("height", height.toString());
    rect.setAttribute("rx", "4");
    rect.setAttribute("fill", getModeColor(mode));
    g.appendChild(rect);

    if (showLabel) {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", (x - 8).toString());
      text.setAttribute("y", (y + height / 2).toString());
      text.setAttribute("text-anchor", "end");
      text.setAttribute("dominant-baseline", "middle");
      text.textContent = mode.replace(/_/g, " ");
      g.appendChild(text);
    } else {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", (x + width + 8).toString());
      text.setAttribute("y", (y + height / 2).toString());
      text.setAttribute("text-anchor", "start");
      text.setAttribute("dominant-baseline", "middle");
      text.textContent = mode.replace(/_/g, " ");
      g.appendChild(text);
    }

    parent.appendChild(g);
  }

  private renderModeList(containerId: string, modes: string[]) {
    const container = document.getElementById(containerId)!;

    if (modes.length === 0) {
      container.innerHTML = '<div style="color: var(--text-dim); font-size: 12px;">None detected</div>';
      return;
    }

    container.innerHTML = modes.map((mode) => `
      <div class="mode-item">
        <div class="mode-dot ${getModeCategory(mode)}"></div>
        <span>${mode.replace(/_/g, " ")}</span>
      </div>
    `).join("");
  }

  private renderPatterns(patterns: Array<{ description: string }>) {
    const container = document.getElementById("patterns-list")!;

    if (patterns.length === 0) {
      container.innerHTML = '<div style="color: var(--text-dim); font-size: 12px;">No patterns detected</div>';
      return;
    }

    container.innerHTML = patterns.slice(0, 5).map((p) => `
      <div class="pattern-item">${p.description}</div>
    `).join("");
  }

  private generateInterpretation(data: ModeFlowResult): string {
    const parts: string[] = [];

    if (data.stable_modes.length > 0) {
      parts.push(`Stable modes (${data.stable_modes.map((m) => m.replace(/_/g, " ")).join(", ")}) tend to persist once entered.`);
    }

    if (data.volatile_modes.length > 0) {
      parts.push(`Volatile modes (${data.volatile_modes.map((m) => m.replace(/_/g, " ")).join(", ")}) see rapid transitions away.`);
    }

    if (data.mode_sequence.length > 1) {
      const uniqueModes = new Set(data.mode_sequence).size;
      const transitionRate = (data.mode_sequence.length - 1) / uniqueModes;
      if (transitionRate > 2) {
        parts.push("High transition frequency suggests narrative instability.");
      } else {
        parts.push("Moderate transition frequency indicates evolving but coherent narrative.");
      }
    }

    return parts.join(" ") || "Analysis complete.";
  }
}

// Initialize
new ModeFlowViewer();
