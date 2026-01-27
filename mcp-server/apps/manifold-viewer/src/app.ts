import { App } from "@modelcontextprotocol/ext-apps";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Types
interface ProjectionPoint {
  text: string;
  coordinates: {
    agency: number;
    perceived_justice?: number;
    fairness?: number;
    belonging: number;
  };
  mode: string | { primary_mode: string; confidence: number };
}

interface ToolResult {
  content?: Array<{ type: string; text?: string }>;
  projections?: ProjectionPoint[];
  results?: ProjectionPoint[];
}

// Mode colors
const MODE_COLORS: Record<string, number> = {
  // Positive modes
  growth_mindset: 0x4ade80,
  civic_idealism: 0x22c55e,
  faithful_zeal: 0x16a34a,
  positive: 0x4ade80,
  // Shadow modes
  cynical_burnout: 0xf87171,
  institutional_decay: 0xef4444,
  schismatic_doubt: 0xdc2626,
  shadow: 0xf87171,
  // Exit modes
  quiet_quitting: 0x60a5fa,
  grid_exit: 0x3b82f6,
  apostasy: 0x2563eb,
  exit: 0x60a5fa,
  // Ambivalent modes
  conflicted: 0xa78bfa,
  transitional: 0x8b5cf6,
  neutral: 0x7c3aed,
  ambivalent: 0xa78bfa,
  // Default
  unknown: 0x888888,
};

// Get category from mode
function getModeCategory(mode: string): string {
  const m = mode.toLowerCase().replace(/\s+/g, "_");
  if (["growth_mindset", "civic_idealism", "faithful_zeal", "positive"].includes(m)) return "positive";
  if (["cynical_burnout", "institutional_decay", "schismatic_doubt", "shadow"].includes(m)) return "shadow";
  if (["quiet_quitting", "grid_exit", "apostasy", "exit"].includes(m)) return "exit";
  return "ambivalent";
}

// Get color for mode
function getModeColor(mode: string): number {
  const m = mode.toLowerCase().replace(/\s+/g, "_");
  return MODE_COLORS[m] || MODE_COLORS[getModeCategory(mode)] || MODE_COLORS.unknown;
}

// Main app class
class ManifoldViewer {
  private app: App;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private points: THREE.Points | null = null;
  private pointsData: ProjectionPoint[] = [];
  private regionMeshes: THREE.Mesh[] = [];
  private gridHelper: THREE.GridHelper | null = null;
  private axisHelper: THREE.AxesHelper | null = null;
  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;
  private hoveredIndex: number = -1;
  private selectedIndex: number = -1;
  private showRegions: boolean = true;
  private showGrid: boolean = true;

  constructor() {
    this.app = new App({ name: "Observatory Manifold Viewer", version: "1.0.0" });
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({
      canvas: document.getElementById("three-canvas") as HTMLCanvasElement,
      antialias: true,
      alpha: true,
    });
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points = { threshold: 0.1 };
    this.mouse = new THREE.Vector2();

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    this.init();
  }

  private init() {
    // Setup scene
    this.scene.background = new THREE.Color(0x0a0a0f);
    this.camera.position.set(5, 4, 5);
    this.camera.lookAt(0, 0, 0);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7);
    this.scene.add(directionalLight);

    // Create axes and grid
    this.createAxes();
    this.createGrid();
    this.createModeRegions();

    // Handle resize
    this.handleResize();
    window.addEventListener("resize", () => this.handleResize());

    // Mouse events
    const canvas = this.renderer.domElement;
    canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
    canvas.addEventListener("click", (e) => this.onClick(e));

    // Controls
    document.getElementById("reset-view")?.addEventListener("click", () => this.resetView());
    document.getElementById("toggle-regions")?.addEventListener("click", () => this.toggleRegions());
    document.getElementById("toggle-grid")?.addEventListener("click", () => this.toggleGrid());

    // Connect to MCP host
    this.app.connect();
    this.app.ontoolresult = (result) => this.handleToolResult(result);

    // Start render loop
    this.animate();

    // Hide loading
    setTimeout(() => {
      const loading = document.getElementById("loading");
      if (loading) loading.style.display = "none";
    }, 500);
  }

  private createAxes() {
    // Custom axes with labels
    const axisLength = 2.5;
    const axisGeometry = new THREE.BufferGeometry();

    // X axis (Agency) - Red
    const xPoints = [new THREE.Vector3(-axisLength, 0, 0), new THREE.Vector3(axisLength, 0, 0)];
    const xGeometry = new THREE.BufferGeometry().setFromPoints(xPoints);
    const xAxis = new THREE.Line(xGeometry, new THREE.LineBasicMaterial({ color: 0xff6b6b }));
    this.scene.add(xAxis);

    // Y axis (Justice) - Cyan
    const yPoints = [new THREE.Vector3(0, -axisLength, 0), new THREE.Vector3(0, axisLength, 0)];
    const yGeometry = new THREE.BufferGeometry().setFromPoints(yPoints);
    const yAxis = new THREE.Line(yGeometry, new THREE.LineBasicMaterial({ color: 0x4ecdc4 }));
    this.scene.add(yAxis);

    // Z axis (Belonging) - Yellow
    const zPoints = [new THREE.Vector3(0, 0, -axisLength), new THREE.Vector3(0, 0, axisLength)];
    const zGeometry = new THREE.BufferGeometry().setFromPoints(zPoints);
    const zAxis = new THREE.Line(zGeometry, new THREE.LineBasicMaterial({ color: 0xffe66d }));
    this.scene.add(zAxis);

    // Axis arrows
    const arrowSize = 0.15;
    const coneGeometry = new THREE.ConeGeometry(arrowSize * 0.5, arrowSize, 8);

    const xArrow = new THREE.Mesh(coneGeometry, new THREE.MeshBasicMaterial({ color: 0xff6b6b }));
    xArrow.position.set(axisLength, 0, 0);
    xArrow.rotation.z = -Math.PI / 2;
    this.scene.add(xArrow);

    const yArrow = new THREE.Mesh(coneGeometry, new THREE.MeshBasicMaterial({ color: 0x4ecdc4 }));
    yArrow.position.set(0, axisLength, 0);
    this.scene.add(yArrow);

    const zArrow = new THREE.Mesh(coneGeometry, new THREE.MeshBasicMaterial({ color: 0xffe66d }));
    zArrow.position.set(0, 0, axisLength);
    zArrow.rotation.x = Math.PI / 2;
    this.scene.add(zArrow);
  }

  private createGrid() {
    this.gridHelper = new THREE.GridHelper(4, 8, 0x2a2a4e, 0x1a1a2e);
    this.gridHelper.position.y = -2;
    this.scene.add(this.gridHelper);
  }

  private createModeRegions() {
    // Create semi-transparent octants representing mode regions
    const regionSize = 2;
    const opacity = 0.08;

    // Positive: +agency, +justice (front-top quadrant)
    const positiveGeometry = new THREE.BoxGeometry(regionSize, regionSize, regionSize);
    const positiveMaterial = new THREE.MeshBasicMaterial({
      color: 0x4ade80,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
    });
    const positiveMesh = new THREE.Mesh(positiveGeometry, positiveMaterial);
    positiveMesh.position.set(1, 1, 0);
    this.scene.add(positiveMesh);
    this.regionMeshes.push(positiveMesh);

    // Shadow: +agency, -justice (front-bottom quadrant)
    const shadowGeometry = new THREE.BoxGeometry(regionSize, regionSize, regionSize);
    const shadowMaterial = new THREE.MeshBasicMaterial({
      color: 0xf87171,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
    });
    const shadowMesh = new THREE.Mesh(shadowGeometry, shadowMaterial);
    shadowMesh.position.set(1, -1, 0);
    this.scene.add(shadowMesh);
    this.regionMeshes.push(shadowMesh);

    // Exit: -agency (back region)
    const exitGeometry = new THREE.BoxGeometry(regionSize, regionSize * 2, regionSize * 2);
    const exitMaterial = new THREE.MeshBasicMaterial({
      color: 0x60a5fa,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
    });
    const exitMesh = new THREE.Mesh(exitGeometry, exitMaterial);
    exitMesh.position.set(-1, 0, 0);
    this.scene.add(exitMesh);
    this.regionMeshes.push(exitMesh);
  }

  private handleResize() {
    const container = document.getElementById("canvas-container");
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }

  private handleToolResult(result: ToolResult) {
    // Extract projections from tool result
    let projections: ProjectionPoint[] = [];

    if (result.projections) {
      projections = result.projections;
    } else if (result.results) {
      projections = result.results;
    } else if (result.content) {
      // Try to parse from text content
      const textContent = result.content.find((c) => c.type === "text")?.text;
      if (textContent) {
        try {
          const parsed = JSON.parse(textContent);
          if (Array.isArray(parsed)) {
            projections = parsed;
          } else if (parsed.projections) {
            projections = parsed.projections;
          }
        } catch {
          // Single projection result - try to extract coordinates
          const match = textContent.match(/Agency:\s*([-\d.]+).*?Justice:\s*([-\d.]+).*?Belonging:\s*([-\d.]+)/s);
          if (match) {
            projections = [{
              text: textContent.match(/Text[:\s]+"([^"]+)"/)?.[1] || "Unknown",
              coordinates: {
                agency: parseFloat(match[1]),
                perceived_justice: parseFloat(match[2]),
                belonging: parseFloat(match[3]),
              },
              mode: textContent.match(/Mode:\s*(\w+)/i)?.[1] || "unknown",
            }];
          }
        }
      }
    }

    if (projections.length > 0) {
      this.updatePoints(projections);
    }
  }

  private updatePoints(projections: ProjectionPoint[]) {
    this.pointsData = projections;

    // Remove old points
    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      (this.points.material as THREE.Material).dispose();
    }

    // Create geometry
    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];

    const stats = { total: 0, positive: 0, shadow: 0, exit: 0, ambivalent: 0 };

    projections.forEach((p) => {
      const coords = p.coordinates;
      const justice = coords.perceived_justice ?? coords.fairness ?? 0;

      positions.push(coords.agency, justice, coords.belonging);

      const modeStr = typeof p.mode === "string" ? p.mode : p.mode?.primary_mode || "unknown";
      const color = new THREE.Color(getModeColor(modeStr));
      colors.push(color.r, color.g, color.b);

      sizes.push(0.15);

      // Update stats
      stats.total++;
      const category = getModeCategory(modeStr);
      stats[category as keyof typeof stats]++;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.Float32BufferAttribute(sizes, 1));

    // Custom shader material for points
    const material = new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
    });

    this.points = new THREE.Points(geometry, material);
    this.scene.add(this.points);

    // Update stats display
    document.getElementById("stat-total")!.textContent = stats.total.toString();
    document.getElementById("stat-positive")!.textContent = stats.positive.toString();
    document.getElementById("stat-shadow")!.textContent = stats.shadow.toString();
    document.getElementById("stat-exit")!.textContent = stats.exit.toString();
    document.getElementById("stat-ambivalent")!.textContent = stats.ambivalent.toString();
  }

  private onMouseMove(event: MouseEvent) {
    const canvas = this.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast
    if (this.points) {
      this.raycaster.setFromCamera(this.mouse, this.camera);
      const intersects = this.raycaster.intersectObject(this.points);

      const tooltip = document.getElementById("tooltip")!;

      if (intersects.length > 0) {
        const idx = intersects[0].index!;
        if (idx !== this.hoveredIndex && idx < this.pointsData.length) {
          this.hoveredIndex = idx;
          const point = this.pointsData[idx];
          const modeStr = typeof point.mode === "string" ? point.mode : point.mode?.primary_mode || "unknown";
          const justice = point.coordinates.perceived_justice ?? point.coordinates.fairness ?? 0;

          tooltip.innerHTML = `
            <strong>${modeStr}</strong><br>
            <span style="color: #888; font-size: 11px;">
              "${point.text.slice(0, 80)}${point.text.length > 80 ? "..." : ""}"
            </span><br>
            <span style="font-family: monospace; font-size: 10px; color: #aaa;">
              A: ${point.coordinates.agency.toFixed(2)} |
              J: ${justice.toFixed(2)} |
              B: ${point.coordinates.belonging.toFixed(2)}
            </span>
          `;
          tooltip.style.left = `${event.clientX - rect.left + 15}px`;
          tooltip.style.top = `${event.clientY - rect.top + 15}px`;
          tooltip.classList.add("visible");
        }
      } else {
        this.hoveredIndex = -1;
        tooltip.classList.remove("visible");
      }
    }
  }

  private onClick(event: MouseEvent) {
    if (this.hoveredIndex >= 0 && this.hoveredIndex < this.pointsData.length) {
      this.selectedIndex = this.hoveredIndex;
      const point = this.pointsData[this.selectedIndex];
      const modeStr = typeof point.mode === "string" ? point.mode : point.mode?.primary_mode || "unknown";
      const justice = point.coordinates.perceived_justice ?? point.coordinates.fairness ?? 0;

      document.getElementById("selected-text")!.textContent = point.text;
      document.getElementById("selected-coords")!.textContent =
        `Agency: ${point.coordinates.agency.toFixed(3)} | Justice: ${justice.toFixed(3)} | Belonging: ${point.coordinates.belonging.toFixed(3)}`;
      document.getElementById("selected-mode")!.textContent = `Mode: ${modeStr}`;
      document.getElementById("selected-panel")!.style.display = "block";
    }
  }

  private resetView() {
    this.camera.position.set(5, 4, 5);
    this.camera.lookAt(0, 0, 0);
    this.controls.reset();
  }

  private toggleRegions() {
    this.showRegions = !this.showRegions;
    this.regionMeshes.forEach((mesh) => {
      mesh.visible = this.showRegions;
    });
  }

  private toggleGrid() {
    this.showGrid = !this.showGrid;
    if (this.gridHelper) this.gridHelper.visible = this.showGrid;
  }

  private animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

// Initialize
new ManifoldViewer();
