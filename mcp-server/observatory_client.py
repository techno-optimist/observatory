"""
HTTP client for the Cultural Soliton Observatory FastAPI backend.

Provides async methods for all observatory operations.
"""

import asyncio
import httpx
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class ObservatoryConfig:
    """Configuration for observatory connection."""
    backend_url: str = "http://127.0.0.1:8000"
    timeout: float = 120.0  # ML operations can be slow
    default_model: str = "all-MiniLM-L6-v2"
    default_layer: int = -1


@dataclass
class ProjectionResult:
    """Result of projecting text to the cultural manifold.

    Note: The "fairness" axis has been renamed to "perceived_justice" in the API.
    This class provides both names for backward compatibility:
    - perceived_justice: The canonical name (new)
    - fairness: Deprecated alias (for backward compatibility)
    """
    text: str
    agency: float
    perceived_justice: float  # Canonical name (formerly "fairness")
    belonging: float
    mode: str
    confidence: float
    model_id: str
    embedding_dim: int
    # Enhanced fields from v2 API
    stability_score: Optional[float] = None
    is_boundary_case: Optional[bool] = None
    stability_warning: Optional[str] = None
    soft_labels: Optional[dict] = None  # Full probability distribution
    primary_mode: Optional[str] = None
    secondary_mode: Optional[str] = None
    primary_probability: Optional[float] = None
    secondary_probability: Optional[float] = None

    @property
    def fairness(self) -> float:
        """Deprecated: Use perceived_justice instead."""
        return self.perceived_justice

    def to_dict(self) -> dict:
        result = {
            "text": self.text,
            "coordinates": {
                "agency": self.agency,
                "perceived_justice": self.perceived_justice,
                "belonging": self.belonging,
            },
            "mode": self.mode,
            "confidence": self.confidence,
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
        }
        # Include stability info if available
        if self.stability_score is not None:
            result["stability"] = {
                "stability_score": self.stability_score,
                "is_boundary_case": self.is_boundary_case,
                "stability_warning": self.stability_warning,
            }
        # Include soft labels if available
        if self.soft_labels is not None:
            result["soft_labels"] = self.soft_labels
        if self.primary_mode is not None:
            result["primary_mode"] = self.primary_mode
            result["secondary_mode"] = self.secondary_mode
            result["primary_probability"] = self.primary_probability
            result["secondary_probability"] = self.secondary_probability
        return result


@dataclass
class ClusterResult:
    """A detected narrative cluster."""
    cluster_id: int
    mode: str
    centroid: dict
    size: int
    stability: float
    exemplar_texts: list[str] = field(default_factory=list)


@dataclass
class CorpusAnalysisResult:
    """Result of analyzing a corpus of texts."""
    total_texts: int
    projections: list[ProjectionResult]
    clusters: list[ClusterResult]
    mode_distribution: dict[str, int]


class ObservatoryClient:
    """
    Async client for the Cultural Soliton Observatory backend.

    Usage:
        async with ObservatoryClient() as client:
            result = await client.project_text("We believe in fair wages.")
            print(result.agency, result.fairness, result.belonging)
    """

    def __init__(self, config: Optional[ObservatoryConfig] = None):
        self.config = config or ObservatoryConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.config.backend_url,
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client

    async def health_check(self) -> dict:
        """Check if the backend is running and get status."""
        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()

    async def project_text(
        self,
        text: str,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
        use_v2: bool = True,
    ) -> ProjectionResult:
        """
        Project a single text onto the cultural manifold.

        Returns coordinates (agency, perceived_justice, belonging), classified mode,
        and confidence score.

        Args:
            text: Text to project
            model_id: Optional embedding model to use
            layer: Optional layer for embedding extraction
            use_v2: If True (default), use v2 API with soft labels and stability

        Note: The "fairness" axis has been renamed to "perceived_justice".
              The ProjectionResult.fairness property still works for backward compat.
        """
        if use_v2:
            return await self.project_text_v2(text, model_id, layer)

        # Legacy v1 API path
        response = await self.client.post(
            "/analyze",
            json={
                "text": text,
                "model_id": model_id or self.config.default_model,
                "layer": layer if layer is not None else self.config.default_layer,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Handle both "fairness" and "perceived_justice" in response
        vector = data.get("vector", {})
        perceived_justice = vector.get("perceived_justice", vector.get("fairness", 0))

        return ProjectionResult(
            text=data["text"],
            agency=vector.get("agency", 0),
            perceived_justice=perceived_justice,
            belonging=vector.get("belonging", 0),
            mode=data["mode"],
            confidence=data["confidence"],
            model_id=data["model_id"],
            embedding_dim=data["embedding_dim"],
        )

    async def project_batch(
        self,
        texts: list[str],
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> list[ProjectionResult]:
        """Project multiple texts in batch."""
        results = []
        # Process in parallel batches
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tasks = [self.project_text(t, model_id, layer) for t in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, Exception):
                    raise r
                results.append(r)
        return results

    async def analyze_corpus(
        self,
        texts: list[str],
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
        detect_clusters: bool = True,
    ) -> CorpusAnalysisResult:
        """
        Analyze a corpus of texts with optional clustering.

        Returns projections for all texts plus detected narrative clusters.
        """
        response = await self.client.post(
            "/corpus/analyze",
            json={
                "texts": texts,
                "model_id": model_id or self.config.default_model,
                "layer": layer if layer is not None else self.config.default_layer,
                "detect_clusters": detect_clusters,
            },
        )
        response.raise_for_status()
        data = response.json()

        projections = []
        for p in data.get("projections", []):
            vector = p.get("vector", {})
            # Handle both "fairness" and "perceived_justice"
            perceived_justice = vector.get("perceived_justice", vector.get("fairness", 0))
            projections.append(ProjectionResult(
                text=p["text"],
                agency=vector.get("agency", 0),
                perceived_justice=perceived_justice,
                belonging=vector.get("belonging", 0),
                mode=p["mode"],
                confidence=p["confidence"],
                model_id=p.get("model_id", self.config.default_model),
                embedding_dim=p.get("embedding_dim", 384),
            ))

        clusters = [
            ClusterResult(
                cluster_id=c["cluster_id"],
                mode=c.get("mode", "unknown"),  # Mode may not be present in all cluster responses
                centroid=c["centroid"],
                size=c["size"],
                stability=c.get("stability", c.get("stability_score", 0.0)),
                exemplar_texts=c.get("exemplar_texts", c.get("sample_texts", [])),
            )
            for c in data.get("clusters", [])
        ]

        # Compute mode distribution
        mode_dist: dict[str, int] = {}
        for p in projections:
            mode_dist[p.mode] = mode_dist.get(p.mode, 0) + 1

        return CorpusAnalysisResult(
            total_texts=len(texts),
            projections=projections,
            clusters=clusters,
            mode_distribution=mode_dist,
        )

    async def get_available_models(self) -> list[dict]:
        """Get list of available embedding models."""
        response = await self.client.get("/models/available")
        response.raise_for_status()
        return response.json()

    async def get_loaded_models(self) -> list[dict]:
        """Get currently loaded models."""
        response = await self.client.get("/models/loaded")
        response.raise_for_status()
        return response.json()

    async def load_model(self, model_id: str, model_type: str = "sentence-transformer") -> dict:
        """Load a model into memory."""
        response = await self.client.post(
            "/models/load",
            json={"model_id": model_id, "model_type": model_type},
        )
        response.raise_for_status()
        return response.json()

    async def get_training_examples(self) -> list[dict]:
        """Get all training examples used for projection."""
        response = await self.client.get("/training/examples")
        response.raise_for_status()
        return response.json()

    async def add_training_example(
        self,
        text: str,
        agency: float,
        fairness: float,
        belonging: float,
        source: str = "mcp_agent",
    ) -> dict:
        """Add a new training example."""
        response = await self.client.post(
            "/training/examples",
            json={
                "text": text,
                "agency": agency,
                "fairness": fairness,
                "belonging": belonging,
                "source": source,
            },
        )
        response.raise_for_status()
        return response.json()

    async def train_projection(
        self,
        model_id: Optional[str] = None,
        method: str = "ridge",
        layer: int = -1,
    ) -> dict:
        """
        Train or retrain the projection.

        Methods: ridge, gp (Gaussian Process), neural, cav
        """
        response = await self.client.post(
            "/training/train",
            json={
                "model_id": model_id or self.config.default_model,
                "method": method,
                "layer": layer,
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_projection_status(self) -> dict:
        """Get current projection status and metadata."""
        response = await self.client.get("/projection/status")
        response.raise_for_status()
        return response.json()

    async def compare_projection_methods(
        self,
        text: str,
        methods: Optional[list[str]] = None,
    ) -> dict:
        """Compare how different projection methods handle the same text."""
        response = await self.client.post(
            "/projection/compare",
            json={
                "text": text,
                "methods": methods or ["ridge", "gp", "neural"],
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_validation_agreement(self) -> dict:
        """Get inter-rater agreement metrics for annotations."""
        response = await self.client.get("/validation/agreement")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Projection Mode Management (v2 API)
    # =========================================================================

    async def list_projection_modes(self) -> dict:
        """
        List available projection modes with their characteristics.

        Returns information about each projection configuration:
        - current_projection: Default MiniLM-based (CV: 0.383)
        - mpnet_projection: Best accuracy with all-mpnet-base-v2 (CV: 0.612)
        - multi_model_ensemble: Best robustness, averages 3 models
        - ensemble_projection: 25-model bootstrap ensemble for uncertainty
        """
        response = await self.client.get("/v2/projections")
        response.raise_for_status()
        return response.json()

    async def select_projection_mode(self, mode: str) -> dict:
        """
        Select which projection mode to use for subsequent analysis.

        Args:
            mode: One of:
                - "current_projection": Default MiniLM (fastest)
                - "mpnet_projection": Best accuracy
                - "multi_model_ensemble": Best robustness (3 models)
                - "ensemble_projection": Uncertainty quantification (25 models)

        Returns:
            Confirmation with mode details and required models.
        """
        response = await self.client.post(
            "/v2/projections/select",
            json={"mode": mode},
        )
        response.raise_for_status()
        return response.json()

    async def get_current_projection_mode(self) -> dict:
        """Get the currently active projection mode."""
        response = await self.client.get("/v2/projections/current")
        response.raise_for_status()
        return response.json()

    async def get_projection_mode_details(self, mode: str) -> dict:
        """Get detailed information about a specific projection mode."""
        response = await self.client.get(f"/v2/projections/{mode}")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Enhanced Analysis (v2 API)
    # =========================================================================

    async def project_text_v2(
        self,
        text: str,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
        include_uncertainty: bool = True,
        include_legacy_mode: bool = True,
    ) -> ProjectionResult:
        """
        Project text using v2 API with enhanced mode classification.

        Returns soft labels (probability per mode), stability indicators,
        and optionally uncertainty when using ensemble mode.

        The response includes:
        - Coordinates (agency, perceived_justice, belonging)
        - Primary and secondary modes with probabilities
        - Full probability distribution across all 12 modes
        - Stability score and boundary case detection
        - Optional uncertainty quantification (for ensemble modes)

        Falls back to legacy /analyze endpoint if v2 API fails.
        """
        # Try v2 API first
        use_legacy = False
        try:
            response = await self.client.post(
                "/v2/analyze",
                json={
                    "text": text,
                    "model_id": model_id or self.config.default_model,
                    "layer": layer if layer is not None else self.config.default_layer,
                    "include_uncertainty": include_uncertainty,
                    "include_legacy_mode": include_legacy_mode,
                },
            )
            response.raise_for_status()
        except Exception:
            use_legacy = True

        if use_legacy:
            # Fall back to legacy endpoint
            response = await self.client.post(
                "/analyze",
                json={
                    "text": text,
                    "model_id": model_id or self.config.default_model,
                    "layer": layer if layer is not None else self.config.default_layer,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Convert legacy response to v2 format
            vector = data.get("vector", {})
            perceived_justice = vector.get("perceived_justice", vector.get("fairness", 0))

            return ProjectionResult(
                text=data["text"],
                agency=vector.get("agency", 0),
                perceived_justice=perceived_justice,
                belonging=vector.get("belonging", 0),
                mode=data["mode"],
                confidence=data["confidence"],
                model_id=data["model_id"],
                embedding_dim=data["embedding_dim"],
            )

        data = response.json()

        # Extract vector - handle both "fairness" and "perceived_justice"
        vector = data.get("vector", {})
        perceived_justice = vector.get("perceived_justice", vector.get("fairness", 0))

        # Extract mode info
        mode_info = data.get("mode", {})
        primary_mode = mode_info.get("primary_mode", data.get("mode", "unknown"))
        if isinstance(primary_mode, dict):
            primary_mode = primary_mode.get("primary_mode", "unknown")

        return ProjectionResult(
            text=data["text"],
            agency=vector.get("agency", 0),
            perceived_justice=perceived_justice,
            belonging=vector.get("belonging", 0),
            mode=mode_info.get("primary_mode", "unknown") if isinstance(mode_info, dict) else mode_info,
            confidence=mode_info.get("confidence", 0) if isinstance(mode_info, dict) else data.get("confidence", 0),
            model_id=data.get("model_id", self.config.default_model),
            embedding_dim=data.get("embedding_dim", 384),
            stability_score=mode_info.get("stability_score") if isinstance(mode_info, dict) else None,
            is_boundary_case=mode_info.get("is_boundary_case") if isinstance(mode_info, dict) else None,
            stability_warning=mode_info.get("stability_warning") if isinstance(mode_info, dict) else None,
            soft_labels=mode_info.get("all_probabilities") if isinstance(mode_info, dict) else None,
            primary_mode=mode_info.get("primary_mode") if isinstance(mode_info, dict) else None,
            secondary_mode=mode_info.get("secondary_mode") if isinstance(mode_info, dict) else None,
            primary_probability=mode_info.get("primary_probability") if isinstance(mode_info, dict) else None,
            secondary_probability=mode_info.get("secondary_probability") if isinstance(mode_info, dict) else None,
        )

    async def analyze_with_uncertainty(
        self,
        text: str,
        model_id: Optional[str] = None,
    ) -> dict:
        """
        Analyze text using ensemble projection for uncertainty quantification.

        Returns coordinates plus 95% confidence intervals for each axis.
        Requires ensemble_projection mode to be available (trained).

        Returns:
            Dictionary with:
            - vector: Coordinates (agency, perceived_justice, belonging)
            - uncertainty: {
                std_per_axis: Standard deviation for each axis,
                confidence_intervals: 95% CI for each axis,
                overall_confidence: Combined confidence score,
                method: "ensemble_ridge"
              }
            - mode: Classification with soft labels
        """
        # First try to select ensemble mode
        try:
            await self.select_projection_mode("ensemble_projection")
        except Exception:
            pass  # Mode might already be selected or unavailable

        # Try v2 API first, fall back to legacy if needed
        try:
            response = await self.client.post(
                "/v2/analyze",
                json={
                    "text": text,
                    "model_id": model_id or self.config.default_model,
                    "include_uncertainty": True,
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            # Fall back to legacy and construct uncertainty response
            response = await self.client.post(
                "/analyze",
                json={
                    "text": text,
                    "model_id": model_id or self.config.default_model,
                },
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("vector", {})
            return {
                "text": data["text"],
                "vector": vector,
                "mode": data["mode"],
                "uncertainty": {
                    "note": "Legacy API used - ensemble uncertainty not available",
                    "confidence": data.get("confidence", 0),
                    "method": "legacy_fallback"
                }
            }

    async def get_soft_labels(
        self,
        text: str,
        model_id: Optional[str] = None,
    ) -> dict:
        """
        Get full probability distribution across all 12 narrative modes.

        Returns the soft labels (probabilities) for each mode, allowing
        nuanced interpretation of texts that straddle multiple modes.

        Returns:
            Dictionary with:
            - text: Input text
            - primary_mode: Most likely mode
            - secondary_mode: Second most likely mode
            - all_probabilities: {mode_name: probability} for all 12 modes
            - is_boundary_case: True if near mode boundary
        """
        try:
            response = await self.client.post(
                "/v2/analyze",
                json={
                    "text": text,
                    "model_id": model_id or self.config.default_model,
                    "include_uncertainty": False,
                    "include_legacy_mode": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            mode_info = data.get("mode", {})
            return {
                "text": data.get("text", text),
                "primary_mode": mode_info.get("primary_mode"),
                "primary_probability": mode_info.get("primary_probability"),
                "secondary_mode": mode_info.get("secondary_mode"),
                "secondary_probability": mode_info.get("secondary_probability"),
                "all_probabilities": mode_info.get("all_probabilities", {}),
                "category": mode_info.get("category"),
                "is_boundary_case": mode_info.get("is_boundary_case", False),
                "stability_score": mode_info.get("stability_score"),
            }
        except Exception:
            # Fall back to legacy API
            response = await self.client.post(
                "/analyze",
                json={
                    "text": text,
                    "model_id": model_id or self.config.default_model,
                },
            )
            response.raise_for_status()
            data = response.json()
            return {
                "text": data.get("text", text),
                "primary_mode": data.get("mode"),
                "primary_probability": data.get("confidence", 0),
                "secondary_mode": None,
                "secondary_probability": None,
                "all_probabilities": {},
                "category": None,
                "is_boundary_case": False,
                "stability_score": data.get("confidence", 0),
            }

    async def analyze_with_mode(
        self,
        text: str,
        mode: str,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> dict:
        """
        Analyze text with a specific projection mode without changing global selection.

        Args:
            text: Text to analyze
            mode: Projection mode to use (current_projection, mpnet_projection,
                  multi_model_ensemble, ensemble_projection)
            model_id: Optional embedding model override (for single-model modes)
            layer: Optional layer to extract embeddings from

        Returns:
            Full analysis result including coordinates, mode classification,
            and mode-specific features (e.g., uncertainty for ensemble modes).
        """
        payload: dict[str, Any] = {
            "text": text,
            "mode": mode,
            "include_uncertainty": True,
            "include_legacy_mode": True,
        }
        if model_id:
            payload["model_id"] = model_id
        if layer is not None:
            payload["layer"] = layer

        response = await self.client.post("/v2/analyze/with-mode", json=payload)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Comparative Analysis (v2 API)
    # =========================================================================

    async def compare_narratives(
        self,
        group_a_name: str,
        group_a_texts: list[str],
        group_b_name: str,
        group_b_texts: list[str],
    ) -> dict:
        """
        Compare two groups of texts and get gap analysis.

        Computes centroids for each group and measures the distance between them
        across all three axes (agency, perceived_justice, belonging).

        Args:
            group_a_name: Label for the first group (e.g., "marketing", "leadership")
            group_a_texts: List of texts from group A
            group_b_name: Label for the second group (e.g., "operations", "employees")
            group_b_texts: List of texts from group B

        Returns:
            Dictionary with:
            - group_a: {name, centroid, mode_distribution, count}
            - group_b: {name, centroid, mode_distribution, count}
            - gap_analysis: {
                delta_agency, delta_perceived_justice, delta_belonging,
                total_gap, interpretation
              }
        """
        response = await self.client.post(
            "/api/v2/compare",
            json={
                "group_a": {"name": group_a_name, "texts": group_a_texts},
                "group_b": {"name": group_b_name, "texts": group_b_texts},
            },
        )
        response.raise_for_status()
        return response.json()

    async def track_trajectory(
        self,
        name: str,
        texts: list[str],
        timestamps: list[str],
    ) -> dict:
        """
        Track narrative evolution over timestamped texts.

        Analyzes how narratives shift in the cultural manifold over time,
        detecting trend direction, velocity, and inflection points.

        Args:
            name: Label for this trajectory (e.g., "company_comms_2024")
            texts: List of texts in chronological order
            timestamps: List of ISO timestamps corresponding to each text

        Returns:
            Dictionary with:
            - name: Trajectory label
            - points: List of {timestamp, coordinates, mode} for each text
            - trend: {direction, velocity, acceleration}
            - inflection_points: List of significant shifts
            - summary: Human-readable trajectory summary
        """
        # Build points list in expected format
        points = [
            {"timestamp": ts, "text": txt}
            for ts, txt in zip(timestamps, texts)
        ]
        response = await self.client.post(
            "/api/v2/trajectory",
            json={
                "name": name,
                "points": points,
            },
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Alert System (v2 API)
    # =========================================================================

    async def create_alert(
        self,
        name: str,
        alert_type: str,
        config: dict,
    ) -> dict:
        """
        Create an alert rule for monitoring narrative drift.

        Args:
            name: Unique name for this alert rule
            alert_type: Type of alert. One of:
                - "gap_threshold": Trigger when gap between groups exceeds threshold
                - "mode_shift": Trigger when dominant mode changes
                - "trajectory_velocity": Trigger on rapid movement in manifold
                - "boundary_crossing": Trigger when texts cross mode boundaries
            config: Alert-specific configuration. Examples:
                - gap_threshold: {"threshold": 0.5, "axis": "agency"}
                - mode_shift: {"from_mode": "positive", "to_mode": "shadow"}
                - trajectory_velocity: {"max_velocity": 0.3}
                - boundary_crossing: {"target_mode": "exit"}

        Returns:
            Dictionary with:
            - id: Alert ID
            - name: Alert name
            - type: Alert type
            - config: Alert configuration
            - created_at: Creation timestamp
        """
        response = await self.client.post(
            "/api/v2/alerts",
            json={
                "name": name,
                "type": alert_type,
                "config": config,
            },
        )
        response.raise_for_status()
        return response.json()

    async def check_alerts(
        self,
        group_a_texts: list[str],
        group_b_texts: list[str],
    ) -> dict:
        """
        Check texts against all configured alert rules.

        Projects both groups and evaluates all active alerts to determine
        which (if any) have been triggered.

        Args:
            group_a_texts: First group of texts to check
            group_b_texts: Second group of texts to check

        Returns:
            Dictionary with:
            - triggered: List of triggered alerts with details
            - checked: Number of alerts checked
            - timestamp: When check was performed
        """
        response = await self.client.post(
            "/api/v2/alerts/check",
            json={
                "group_a": {"name": "Group A", "texts": group_a_texts},
                "group_b": {"name": "Group B", "texts": group_b_texts},
            },
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Advanced Analytics (v2 API)
    # =========================================================================

    async def detect_outliers(
        self,
        corpus: list[str],
        test_text: str,
    ) -> dict:
        """
        Detect anomalous narratives by comparing a test text against a corpus.

        Projects the corpus to establish baseline statistics, then evaluates
        how far the test text deviates from the norm.

        Args:
            corpus: List of texts establishing the baseline
            test_text: Text to evaluate for anomaly

        Returns:
            Dictionary with:
            - is_outlier: Boolean indicating outlier status
            - outlier_score: Mahalanobis distance from corpus centroid
            - z_scores: {agency, perceived_justice, belonging} z-scores
            - corpus_stats: {centroid, std_dev}
            - test_projection: Coordinates of test text
            - interpretation: Human-readable explanation
        """
        response = await self.client.post(
            "/api/v2/analytics/outliers",
            json={
                "corpus": corpus,
                "test_text": test_text,
            },
        )
        response.raise_for_status()
        return response.json()

    async def analyze_cohorts(
        self,
        cohorts: dict[str, list[str]],
    ) -> dict:
        """
        Multi-group cohort analysis with ANOVA.

        Compares multiple groups simultaneously to determine if there are
        statistically significant differences between cohorts.

        Args:
            cohorts: Dictionary mapping cohort names to lists of texts.
                Example: {
                    "engineering": ["text1", "text2", ...],
                    "sales": ["text3", "text4", ...],
                    "support": ["text5", "text6", ...]
                }

        Returns:
            Dictionary with:
            - cohorts: List of {name, centroid, mode_distribution, count}
            - anova: {
                f_statistic, p_value, significant,
                per_axis: {agency, perceived_justice, belonging}
              }
            - pairwise_comparisons: List of significant pairwise differences
            - interpretation: Human-readable summary
        """
        response = await self.client.post(
            "/api/v2/analytics/cohorts",
            json={"cohorts": cohorts},
        )
        response.raise_for_status()
        return response.json()

    async def analyze_mode_flow(
        self,
        texts: list[str],
    ) -> dict:
        """
        Analyze mode transitions and detect patterns in a sequence of texts.

        Maps the flow of narratives through different modes, identifying
        common transition patterns and stable states.

        Args:
            texts: List of texts to analyze for mode flow patterns

        Returns:
            Dictionary with:
            - mode_sequence: List of modes for each text
            - transition_matrix: Probability of transitioning between modes
            - flow_patterns: Detected recurring patterns
            - stable_modes: Modes that texts tend to stay in
            - volatile_modes: Modes that texts tend to leave quickly
            - interpretation: Human-readable flow analysis
        """
        response = await self.client.post(
            "/api/v2/analytics/mode-flow",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Force Field Analysis (Attractors & Detractors)
    # =========================================================================

    async def analyze_force_field(
        self,
        text: str,
    ) -> dict:
        """
        Analyze the attractor/detractor force field of a narrative.

        This extends the 3D manifold analysis with two additional dimensions:
        - Attractor Strength: What the narrative is being pulled TOWARD
        - Detractor Strength: What the narrative is being pushed AWAY FROM

        Args:
            text: Text to analyze

        Returns:
            Dictionary with:
            - attractor_strength: Pull toward positive states (-2 to +2)
            - detractor_strength: Push from negative states (-2 to +2)
            - primary_attractor: Main target (AUTONOMY, COMMUNITY, JUSTICE, MEANING, SECURITY, RECOGNITION)
            - primary_detractor: Main source (OPPRESSION, ISOLATION, INJUSTICE, MEANINGLESSNESS, INSTABILITY, INVISIBILITY)
            - attractor_scores: Scores for each attractor target
            - detractor_scores: Scores for each detractor source
            - force_quadrant: ACTIVE_TRANSFORMATION, PURE_ASPIRATION, PURE_ESCAPE, or STASIS
            - net_force: Combined force magnitude
            - force_direction: TOWARD, AWAY, or BALANCED
        """
        response = await self.client.post(
            "/api/v2/forces/analyze",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()

    async def analyze_trajectory_forces(
        self,
        texts: list[str],
    ) -> dict:
        """
        Analyze how force fields evolve across a sequence of narratives.

        Tracks attractor/detractor changes over time, identifying:
        - Attractor shifts (changing goals)
        - Detractor emergence/resolution
        - Energy level changes
        - Quadrant transitions

        Args:
            texts: List of texts in chronological order

        Returns:
            Dictionary with:
            - attractor_trajectory: How attractor strength changes
            - detractor_trajectory: How detractor strength changes
            - energy_trajectory: How overall force magnitude changes
            - attractor_shifts: Points where primary attractor changes
            - detractor_shifts: Points where primary detractor changes
            - interpretation: Human-readable trajectory analysis
        """
        response = await self.client.post(
            "/api/v2/forces/trajectory",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()

    async def compare_force_fields(
        self,
        group_a_name: str,
        group_a_texts: list[str],
        group_b_name: str,
        group_b_texts: list[str],
    ) -> dict:
        """
        Compare force fields between two groups of narratives.

        Identifies differences in attractor/detractor profiles:
        - Which attractors each group is drawn toward
        - Which detractors each group is fleeing from
        - Energy level differences
        - Force alignment or opposition

        Args:
            group_a_name: Name for first group
            group_a_texts: Texts in first group
            group_b_name: Name for second group
            group_b_texts: Texts in second group

        Returns:
            Dictionary with:
            - group_a: Force field stats for first group
            - group_b: Force field stats for second group
            - comparison: Gap analysis between groups
            - interpretation: Human-readable comparison
        """
        response = await self.client.post(
            "/api/v2/forces/compare",
            json={
                "group_a": {"name": group_a_name, "texts": group_a_texts},
                "group_b": {"name": group_b_name, "texts": group_b_texts},
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_force_field_targets(self) -> dict:
        """
        Get the list of all attractor targets and detractor sources.

        Returns definitions for:
        - Attractor targets: AUTONOMY, COMMUNITY, JUSTICE, MEANING, SECURITY, RECOGNITION
        - Detractor sources: OPPRESSION, ISOLATION, INJUSTICE, MEANINGLESSNESS, INSTABILITY, INVISIBILITY
        - Force quadrants: ACTIVE_TRANSFORMATION, PURE_ASPIRATION, PURE_ESCAPE, STASIS
        """
        response = await self.client.get("/api/v2/forces/targets")
        response.raise_for_status()
        return response.json()

    async def batch_analyze_forces(
        self,
        texts: list[str],
    ) -> dict:
        """
        Analyze force fields for multiple texts at once.

        Returns individual analyses plus aggregate statistics.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with:
            - analyses: List of force field results for each text
            - aggregate: Aggregate statistics (means, distributions)
        """
        response = await self.client.post(
            "/api/v2/forces/batch",
            json={"texts": texts},
        )
        response.raise_for_status()
        return response.json()
