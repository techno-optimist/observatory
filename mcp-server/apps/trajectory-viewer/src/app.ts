import { App } from "@modelcontextprotocol/ext-apps";

interface TrajectoryPoint {
  timestamp: string;
  text?: string;
  coordinates: {
    agency: number;
    perceived_justice?: number;
    fairness?: number;
    belonging: number;
  };
  mode: string;
}

interface InflectionPoint {
  index: number;
  timestamp: string;
  description: string;
}

interface TrajectoryResult {
  name?: string;
  points?: TrajectoryPoint[];
  trend?: {
    direction: string;
    velocity: number;
    acceleration: number;
  };
  inflection_points?: InflectionPoint[];
  summary?: string;
}

const COLORS = {
  agency: "#ff6b6b",
  justice: "#4ecdc4",
  belonging: "#ffe66d",
};

class TrajectoryViewer {
  private app: App;
  private points: TrajectoryPoint[] = [];
  private inflections: InflectionPoint[] = [];
  private trend: TrajectoryResult["trend"] | null = null;
  private currentIndex: number = 0;
  private isPlaying: boolean = false;
  private playInterval: number | null = null;
  private visibleDims = { agency: true, justice: true, belonging: true };
  private svg!: SVGSVGElement;
  private chartWidth: number = 0;
  private chartHeight: number = 0;
  private padding = { top: 40, right: 40, bottom: 40, left: 60 };

  constructor() {
    this.app = new App({ name: "Observatory Trajectory Viewer", version: "1.0.0" });
    this.init();
  }

  private init() {
    this.svg = document.getElementById("chart-svg") as unknown as SVGSVGElement;

    this.app.connect();
    this.app.ontoolresult = (result) => this.handleToolResult(result);

    // Dimension toggles
    document.querySelectorAll(".dim-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const el = e.target as HTMLElement;
        el.classList.toggle("active");
        const dim = el.dataset.dim as keyof typeof this.visibleDims;
        this.visibleDims[dim] = el.classList.contains("active");
        this.render();
      });
    });

    // Play button
    document.getElementById("play-btn")?.addEventListener("click", () => this.togglePlay());

    // Scrubber
    const scrubber = document.getElementById("scrubber")!;
    scrubber.addEventListener("click", (e) => {
      const rect = scrubber.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      this.setCurrentIndex(Math.round(pct * (this.points.length - 1)));
    });

    // Handle resize
    window.addEventListener("resize", () => this.render());
  }

  private handleToolResult(result: any) {
    let trajectory: TrajectoryResult | null = null;

    if (result.points) {
      trajectory = result as TrajectoryResult;
    } else if (result.content) {
      const textContent = result.content.find((c: any) => c.type === "text")?.text;
      if (textContent) {
        try {
          trajectory = JSON.parse(textContent);
        } catch {
          // Parse from formatted output
          const points: TrajectoryPoint[] = [];
          const pointMatches = textContent.matchAll(/\[([^\]]+)\]\s*Mode:\s*(\w+),\s*A=([-\d.]+),\s*PJ=([-\d.]+),\s*B=([-\d.]+)/g);
          for (const match of pointMatches) {
            points.push({
              timestamp: match[1],
              coordinates: {
                agency: parseFloat(match[3]),
                perceived_justice: parseFloat(match[4]),
                belonging: parseFloat(match[5]),
              },
              mode: match[2],
            });
          }
          if (points.length > 0) {
            trajectory = { points };
          }
        }
      }
    }

    if (trajectory?.points && trajectory.points.length > 0) {
      this.points = trajectory.points;
      this.trend = trajectory.trend || null;
      this.inflections = trajectory.inflection_points || [];

      document.getElementById("loading")!.style.display = "none";
      document.getElementById("trajectory-name")!.textContent = trajectory.name || "Narrative Trajectory";

      this.updateStats();
      this.updateInflectionList();
      this.render();
    }
  }

  private updateStats() {
    document.getElementById("stat-points")!.textContent = this.points.length.toString();

    if (this.trend) {
      document.getElementById("stat-direction")!.textContent = this.trend.direction || "Stable";
      document.getElementById("stat-velocity")!.textContent = this.trend.velocity.toFixed(3);
    }
  }

  private updateInflectionList() {
    const list = document.getElementById("inflection-list")!;

    if (this.inflections.length === 0) {
      list.innerHTML = '<div style="color: var(--text-dim); font-size: 12px;">No inflection points detected</div>';
      return;
    }

    list.innerHTML = this.inflections.map((inf, idx) => `
      <div class="inflection-item" data-index="${inf.index}">
        <div class="inflection-time">${inf.timestamp}</div>
        <div class="inflection-desc">${inf.description || `Inflection point ${idx + 1}`}</div>
      </div>
    `).join("");

    list.querySelectorAll(".inflection-item").forEach((item) => {
      item.addEventListener("click", (e) => {
        const idx = parseInt((e.currentTarget as HTMLElement).dataset.index || "0");
        this.setCurrentIndex(idx);
      });
    });
  }

  private setCurrentIndex(idx: number) {
    this.currentIndex = Math.max(0, Math.min(idx, this.points.length - 1));
    this.updateScrubber();
    this.updateCurrentPoint();
    this.render();
  }

  private updateScrubber() {
    const pct = this.points.length > 1 ? (this.currentIndex / (this.points.length - 1)) * 100 : 0;
    document.getElementById("scrubber-fill")!.style.width = `${pct}%`;
    document.getElementById("scrubber-handle")!.style.left = `${pct}%`;
    document.getElementById("time-display")!.textContent = `${this.currentIndex + 1} / ${this.points.length}`;
  }

  private updateCurrentPoint() {
    if (this.points.length === 0) return;

    const point = this.points[this.currentIndex];
    const panel = document.getElementById("current-point")!;
    panel.style.display = "block";

    document.getElementById("current-text")!.textContent = point.text || point.mode;
    document.getElementById("coord-agency")!.textContent = `A: ${point.coordinates.agency.toFixed(2)}`;
    const justice = point.coordinates.perceived_justice ?? point.coordinates.fairness ?? 0;
    document.getElementById("coord-justice")!.textContent = `J: ${justice.toFixed(2)}`;
    document.getElementById("coord-belonging")!.textContent = `B: ${point.coordinates.belonging.toFixed(2)}`;
  }

  private togglePlay() {
    this.isPlaying = !this.isPlaying;
    const btn = document.getElementById("play-btn")!;

    if (this.isPlaying) {
      btn.textContent = "⏸";
      this.playInterval = window.setInterval(() => {
        if (this.currentIndex >= this.points.length - 1) {
          this.currentIndex = 0;
        } else {
          this.currentIndex++;
        }
        this.updateScrubber();
        this.updateCurrentPoint();
        this.render();
      }, 500);
    } else {
      btn.textContent = "▶";
      if (this.playInterval) {
        clearInterval(this.playInterval);
        this.playInterval = null;
      }
    }
  }

  private render() {
    const container = this.svg.parentElement!;
    this.chartWidth = container.clientWidth;
    this.chartHeight = container.clientHeight;

    const width = this.chartWidth - this.padding.left - this.padding.right;
    const height = this.chartHeight - this.padding.top - this.padding.bottom;

    // Clear SVG
    this.svg.innerHTML = "";

    if (this.points.length === 0) return;

    // Create scales
    const xScale = (i: number) => this.padding.left + (i / (this.points.length - 1 || 1)) * width;
    const yScale = (v: number) => this.padding.top + height / 2 - (v / 2) * (height / 2);

    // Draw grid
    this.drawGrid(width, height, yScale);

    // Draw lines for each dimension
    if (this.visibleDims.agency) {
      this.drawLine(xScale, yScale, "agency", COLORS.agency);
    }
    if (this.visibleDims.justice) {
      this.drawLine(xScale, yScale, "justice", COLORS.justice);
    }
    if (this.visibleDims.belonging) {
      this.drawLine(xScale, yScale, "belonging", COLORS.belonging);
    }

    // Draw points
    this.drawPoints(xScale, yScale);

    // Draw inflection markers
    this.drawInflections(xScale, yScale);

    // Draw current position marker
    this.drawCurrentMarker(xScale, yScale);
  }

  private drawGrid(width: number, height: number, yScale: (v: number) => number) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

    // Zero line
    const zeroLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    zeroLine.setAttribute("x1", this.padding.left.toString());
    zeroLine.setAttribute("x2", (this.padding.left + width).toString());
    zeroLine.setAttribute("y1", yScale(0).toString());
    zeroLine.setAttribute("y2", yScale(0).toString());
    zeroLine.setAttribute("class", "zero-line");
    g.appendChild(zeroLine);

    // Y axis labels
    [-2, -1, 0, 1, 2].forEach((v) => {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", (this.padding.left - 10).toString());
      label.setAttribute("y", yScale(v).toString());
      label.setAttribute("text-anchor", "end");
      label.setAttribute("dominant-baseline", "middle");
      label.setAttribute("class", "axis-label");
      label.textContent = v.toString();
      g.appendChild(label);
    });

    this.svg.appendChild(g);
  }

  private drawLine(
    xScale: (i: number) => number,
    yScale: (v: number) => number,
    dim: "agency" | "justice" | "belonging",
    color: string
  ) {
    const pathData = this.points.map((p, i) => {
      let value: number;
      if (dim === "agency") value = p.coordinates.agency;
      else if (dim === "justice") value = p.coordinates.perceived_justice ?? p.coordinates.fairness ?? 0;
      else value = p.coordinates.belonging;

      const x = xScale(i);
      const y = yScale(value);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    }).join(" ");

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathData);
    path.setAttribute("class", "trajectory-line");
    path.setAttribute("stroke", color);
    path.setAttribute("opacity", "0.8");
    this.svg.appendChild(path);
  }

  private drawPoints(xScale: (i: number) => number, yScale: (v: number) => number) {
    this.points.forEach((p, i) => {
      const x = xScale(i);

      // Draw point for each visible dimension
      if (this.visibleDims.agency) {
        this.drawPoint(x, yScale(p.coordinates.agency), COLORS.agency, i);
      }
      if (this.visibleDims.justice) {
        const j = p.coordinates.perceived_justice ?? p.coordinates.fairness ?? 0;
        this.drawPoint(x, yScale(j), COLORS.justice, i);
      }
      if (this.visibleDims.belonging) {
        this.drawPoint(x, yScale(p.coordinates.belonging), COLORS.belonging, i);
      }
    });
  }

  private drawPoint(x: number, y: number, color: string, index: number) {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x.toString());
    circle.setAttribute("cy", y.toString());
    circle.setAttribute("r", index === this.currentIndex ? "6" : "4");
    circle.setAttribute("fill", color);
    circle.setAttribute("class", "trajectory-point");
    circle.setAttribute("opacity", index === this.currentIndex ? "1" : "0.6");

    circle.addEventListener("click", () => this.setCurrentIndex(index));

    this.svg.appendChild(circle);
  }

  private drawInflections(xScale: (i: number) => number, yScale: (v: number) => number) {
    this.inflections.forEach((inf) => {
      if (inf.index >= 0 && inf.index < this.points.length) {
        const x = xScale(inf.index);
        const y = this.padding.top - 10;

        const marker = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        marker.setAttribute("points", `${x},${y + 12} ${x - 6},${y} ${x + 6},${y}`);
        marker.setAttribute("class", "inflection-marker");
        marker.style.cursor = "pointer";

        marker.addEventListener("click", () => this.setCurrentIndex(inf.index));

        this.svg.appendChild(marker);
      }
    });
  }

  private drawCurrentMarker(xScale: (i: number) => number, yScale: (v: number) => number) {
    const x = xScale(this.currentIndex);

    // Vertical line
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x.toString());
    line.setAttribute("x2", x.toString());
    line.setAttribute("y1", this.padding.top.toString());
    line.setAttribute("y2", (this.chartHeight - this.padding.bottom).toString());
    line.setAttribute("stroke", "rgba(255,255,255,0.2)");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("stroke-dasharray", "4 4");
    this.svg.appendChild(line);
  }
}

// Initialize
new TrajectoryViewer();
