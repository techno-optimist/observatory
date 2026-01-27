"""
Alert System for Cultural Soliton Observatory

Monitors narrative gaps and triggers notifications based on configurable alert rules.

Alert Types:
- gap_threshold: Trigger when gap on an axis exceeds threshold
- mode_shift: Trigger when dominant mode changes
- trajectory_velocity: Trigger when rate of change exceeds threshold
- outlier_detection: Trigger when text is anomalous

Usage:
    from api_alerts import router as alerts_router
    app.include_router(alerts_router, prefix="/api/v2", tags=["alerts"])
"""

import json
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

# Data directory for persistence
DATA_DIR = Path("./data")
ALERTS_FILE = DATA_DIR / "alerts.json"


# ============================================================================
# Alert Types and Models
# ============================================================================

class AlertType(str, Enum):
    """Types of alerts that can be configured."""
    GAP_THRESHOLD = "gap_threshold"
    MODE_SHIFT = "mode_shift"
    TRAJECTORY_VELOCITY = "trajectory_velocity"
    OUTLIER_DETECTION = "outlier_detection"


class AlertSeverity(str, Enum):
    """Severity levels for triggered alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GapThresholdConfig(BaseModel):
    """Configuration for gap_threshold alert type."""
    axis: str = Field(
        ...,
        description="Axis to monitor: agency, perceived_justice, or belonging"
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Gap threshold value (0-4 for full range)"
    )
    direction: Literal["exceeds", "below"] = Field(
        default="exceeds",
        description="Trigger when gap exceeds or falls below threshold"
    )

    @field_validator("axis")
    @classmethod
    def validate_axis(cls, v: str) -> str:
        """Validate and normalize axis name."""
        valid_axes = {
            "agency": "agency",
            "perceived_justice": "perceived_justice",
            "fairness": "perceived_justice",  # Backward compatibility
            "belonging": "belonging"
        }
        normalized = valid_axes.get(v.lower())
        if normalized is None:
            raise ValueError(
                f"Invalid axis: {v}. Valid axes: agency, perceived_justice, belonging"
            )
        return normalized


class ModeShiftConfig(BaseModel):
    """Configuration for mode_shift alert type."""
    from_modes: Optional[List[str]] = Field(
        default=None,
        description="Source modes to watch (None = any)"
    )
    to_modes: Optional[List[str]] = Field(
        default=None,
        description="Target modes to watch (None = any)"
    )
    category_shift: bool = Field(
        default=False,
        description="Only alert on category changes (POSITIVE -> SHADOW, etc.)"
    )


class TrajectoryVelocityConfig(BaseModel):
    """Configuration for trajectory_velocity alert type."""
    velocity_threshold: float = Field(
        ...,
        ge=0.0,
        description="Velocity threshold (rate of change per text pair)"
    )
    axis: Optional[str] = Field(
        default=None,
        description="Specific axis to monitor (None = overall magnitude)"
    )


class OutlierConfig(BaseModel):
    """Configuration for outlier_detection alert type."""
    std_threshold: float = Field(
        default=2.0,
        ge=1.0,
        description="Standard deviations from mean to trigger alert"
    )
    require_both_groups: bool = Field(
        default=False,
        description="Require outlier in both groups to trigger"
    )


# ============================================================================
# Alert Rule Models
# ============================================================================

class AlertRuleCreate(BaseModel):
    """Request model for creating an alert rule."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    type: AlertType
    config: Dict[str, Any] = Field(
        ...,
        description="Type-specific configuration"
    )
    enabled: bool = Field(default=True)

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: Dict, info) -> Dict:
        """Validate config matches the alert type."""
        # Get the type from the values being validated
        alert_type = info.data.get("type")
        if alert_type is None:
            return v

        try:
            if alert_type == AlertType.GAP_THRESHOLD:
                GapThresholdConfig(**v)
            elif alert_type == AlertType.MODE_SHIFT:
                ModeShiftConfig(**v)
            elif alert_type == AlertType.TRAJECTORY_VELOCITY:
                TrajectoryVelocityConfig(**v)
            elif alert_type == AlertType.OUTLIER_DETECTION:
                OutlierConfig(**v)
        except Exception as e:
            raise ValueError(f"Invalid config for {alert_type}: {e}")
        return v


class AlertRule(BaseModel):
    """Full alert rule with ID and metadata."""
    id: str
    name: str
    description: str
    type: AlertType
    config: Dict[str, Any]
    enabled: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "config": self.config,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AlertRule":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            type=AlertType(data["type"]),
            config=data["config"],
            enabled=data.get("enabled", True),
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"])
        )


# ============================================================================
# Check Request/Response Models
# ============================================================================

class TextGroup(BaseModel):
    """A group of texts with a label."""
    name: str = Field(..., description="Group name (e.g., 'Public', 'Internal')")
    texts: List[str] = Field(..., min_length=1)


class AlertCheckRequest(BaseModel):
    """Request to check texts against all active alerts."""
    group_a: TextGroup
    group_b: TextGroup


class TriggeredAlert(BaseModel):
    """Result when an alert is triggered."""
    alert_id: str
    alert_name: str
    triggered: bool
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    severity: AlertSeverity
    message: str
    details: Optional[Dict[str, Any]] = None


class AlertCheckResponse(BaseModel):
    """Response from checking alerts."""
    triggered_alerts: List[TriggeredAlert]
    all_clear: bool
    summary: str
    group_a_summary: Optional[Dict[str, Any]] = None
    group_b_summary: Optional[Dict[str, Any]] = None
    gaps: Optional[Dict[str, float]] = None


# ============================================================================
# Alert Storage
# ============================================================================

class AlertStorage:
    """Manages persistence of alert rules to JSON file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the data directory exists."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, AlertRule]:
        """Load alerts from file."""
        if not self.file_path.exists():
            return {}
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return {
                alert_id: AlertRule.from_dict(alert_data)
                for alert_id, alert_data in data.items()
            }
        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")
            return {}

    def _save(self, alerts: Dict[str, AlertRule]):
        """Save alerts to file."""
        try:
            data = {
                alert_id: alert.to_dict()
                for alert_id, alert in alerts.items()
            }
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
            raise

    def create(self, rule: AlertRuleCreate) -> AlertRule:
        """Create a new alert rule."""
        alerts = self._load()
        alert_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat() + "Z"

        alert = AlertRule(
            id=alert_id,
            name=rule.name,
            description=rule.description,
            type=rule.type,
            config=rule.config,
            enabled=rule.enabled,
            created_at=now,
            updated_at=now
        )

        alerts[alert_id] = alert
        self._save(alerts)
        logger.info(f"Created alert: {alert_id} - {rule.name}")
        return alert

    def get(self, alert_id: str) -> Optional[AlertRule]:
        """Get a specific alert by ID."""
        alerts = self._load()
        return alerts.get(alert_id)

    def list_all(self) -> List[AlertRule]:
        """List all alerts."""
        return list(self._load().values())

    def list_enabled(self) -> List[AlertRule]:
        """List only enabled alerts."""
        return [a for a in self._load().values() if a.enabled]

    def delete(self, alert_id: str) -> bool:
        """Delete an alert. Returns True if deleted."""
        alerts = self._load()
        if alert_id not in alerts:
            return False
        del alerts[alert_id]
        self._save(alerts)
        logger.info(f"Deleted alert: {alert_id}")
        return True

    def update(self, alert_id: str, updates: Dict[str, Any]) -> Optional[AlertRule]:
        """Update an alert. Returns updated alert or None."""
        alerts = self._load()
        if alert_id not in alerts:
            return None

        alert = alerts[alert_id]
        alert_dict = alert.to_dict()
        alert_dict.update(updates)
        alert_dict["updated_at"] = datetime.utcnow().isoformat() + "Z"

        updated_alert = AlertRule.from_dict(alert_dict)
        alerts[alert_id] = updated_alert
        self._save(alerts)
        return updated_alert


# Global storage instance
_alert_storage: Optional[AlertStorage] = None


def get_alert_storage() -> AlertStorage:
    """Get or create the alert storage instance."""
    global _alert_storage
    if _alert_storage is None:
        _alert_storage = AlertStorage(ALERTS_FILE)
    return _alert_storage


# ============================================================================
# Alert Checker
# ============================================================================

class AlertChecker:
    """Checks texts against alert rules."""

    def __init__(self):
        self.storage = get_alert_storage()

    async def _get_group_projections(
        self,
        texts: List[str],
        model_id: str = "all-MiniLM-L6-v2"
    ) -> Dict[str, Any]:
        """Get projections for a group of texts."""
        from main import (
            model_manager, embedding_extractor, current_projection,
            ModelType
        )
        from analysis.mode_classifier import get_mode_classifier

        if current_projection is None:
            raise HTTPException(status_code=400, detail="No projection trained")

        if not model_manager.is_loaded(model_id):
            model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

        # Get embeddings and project
        projections = []
        modes = []
        classifier = get_mode_classifier()

        for text in texts:
            result = embedding_extractor.extract(text, model_id)
            coords = current_projection.project(result.embedding)
            coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
            mode_result = classifier.classify(coords_array)

            projections.append({
                "agency": coords.agency,
                "perceived_justice": coords.fairness,
                "belonging": coords.belonging
            })
            modes.append(mode_result["primary_mode"])

        # Compute statistics
        proj_array = np.array([
            [p["agency"], p["perceived_justice"], p["belonging"]]
            for p in projections
        ])

        return {
            "projections": projections,
            "modes": modes,
            "mean": {
                "agency": float(np.mean(proj_array[:, 0])),
                "perceived_justice": float(np.mean(proj_array[:, 1])),
                "belonging": float(np.mean(proj_array[:, 2]))
            },
            "std": {
                "agency": float(np.std(proj_array[:, 0])),
                "perceived_justice": float(np.std(proj_array[:, 1])),
                "belonging": float(np.std(proj_array[:, 2]))
            },
            "dominant_mode": max(set(modes), key=modes.count),
            "mode_distribution": {m: modes.count(m) for m in set(modes)}
        }

    def _compute_gaps(
        self,
        group_a_summary: Dict[str, Any],
        group_b_summary: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute gaps between two groups."""
        gaps = {}
        for axis in ["agency", "perceived_justice", "belonging"]:
            gap = abs(
                group_a_summary["mean"][axis] -
                group_b_summary["mean"][axis]
            )
            gaps[axis] = round(gap, 4)
        return gaps

    def _check_gap_threshold(
        self,
        alert: AlertRule,
        gaps: Dict[str, float],
        group_a: Dict,
        group_b: Dict
    ) -> TriggeredAlert:
        """Check a gap_threshold alert."""
        config = GapThresholdConfig(**alert.config)
        axis = config.axis
        current_gap = gaps.get(axis, 0)
        threshold = config.threshold

        if config.direction == "exceeds":
            triggered = current_gap > threshold
            message = f"{axis.replace('_', ' ').title()} gap ({current_gap:.3f}) exceeds threshold ({threshold})"
        else:
            triggered = current_gap < threshold
            message = f"{axis.replace('_', ' ').title()} gap ({current_gap:.3f}) is below threshold ({threshold})"

        # Determine severity based on how much threshold is exceeded
        if not triggered:
            severity = AlertSeverity.LOW
        else:
            ratio = current_gap / threshold if threshold > 0 else current_gap
            if ratio > 2.0:
                severity = AlertSeverity.CRITICAL
            elif ratio > 1.5:
                severity = AlertSeverity.HIGH
            elif ratio > 1.2:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

        return TriggeredAlert(
            alert_id=alert.id,
            alert_name=alert.name,
            triggered=triggered,
            current_value=round(current_gap, 4),
            threshold=threshold,
            severity=severity,
            message=message if triggered else f"Gap is within acceptable range",
            details={
                "axis": axis,
                "direction": config.direction,
                "group_a_mean": group_a["mean"][axis],
                "group_b_mean": group_b["mean"][axis]
            }
        )

    def _check_mode_shift(
        self,
        alert: AlertRule,
        group_a: Dict,
        group_b: Dict
    ) -> TriggeredAlert:
        """Check a mode_shift alert."""
        config = ModeShiftConfig(**alert.config)

        mode_a = group_a["dominant_mode"]
        mode_b = group_b["dominant_mode"]

        # Get categories
        def get_category(mode: str) -> str:
            positive_modes = ["EMPOWERED", "HEROIC", "ACHIEVER"]
            shadow_modes = ["DEFIANT", "VICTIM", "CYNICAL"]
            exit_modes = ["DISENGAGED", "WITHDRAWN", "RESIGNED"]
            if mode in positive_modes:
                return "POSITIVE"
            elif mode in shadow_modes:
                return "SHADOW"
            elif mode in exit_modes:
                return "EXIT"
            return "AMBIVALENT"

        cat_a = get_category(mode_a)
        cat_b = get_category(mode_b)

        mode_changed = mode_a != mode_b
        category_changed = cat_a != cat_b

        # Check if this matches the alert criteria
        triggered = False
        if config.category_shift:
            triggered = category_changed
        else:
            from_match = config.from_modes is None or mode_a in config.from_modes
            to_match = config.to_modes is None or mode_b in config.to_modes
            triggered = mode_changed and from_match and to_match

        if triggered:
            if config.category_shift:
                message = f"Category shift detected: {cat_a} -> {cat_b}"
                severity = AlertSeverity.HIGH
            else:
                message = f"Mode shift detected: {mode_a} -> {mode_b}"
                severity = AlertSeverity.MEDIUM
        else:
            message = f"No mode shift detected (both groups: {mode_a})"
            severity = AlertSeverity.LOW

        return TriggeredAlert(
            alert_id=alert.id,
            alert_name=alert.name,
            triggered=triggered,
            severity=severity,
            message=message,
            details={
                "group_a_mode": mode_a,
                "group_b_mode": mode_b,
                "group_a_category": cat_a,
                "group_b_category": cat_b,
                "mode_distribution_a": group_a["mode_distribution"],
                "mode_distribution_b": group_b["mode_distribution"]
            }
        )

    def _check_trajectory_velocity(
        self,
        alert: AlertRule,
        group_a: Dict,
        group_b: Dict
    ) -> TriggeredAlert:
        """Check trajectory velocity between groups."""
        config = TrajectoryVelocityConfig(**alert.config)

        # Compute velocity as Euclidean distance between means
        mean_a = np.array([
            group_a["mean"]["agency"],
            group_a["mean"]["perceived_justice"],
            group_a["mean"]["belonging"]
        ])
        mean_b = np.array([
            group_b["mean"]["agency"],
            group_b["mean"]["perceived_justice"],
            group_b["mean"]["belonging"]
        ])

        if config.axis:
            # Single axis velocity
            axis_idx = {"agency": 0, "perceived_justice": 1, "belonging": 2}[config.axis]
            velocity = abs(mean_b[axis_idx] - mean_a[axis_idx])
        else:
            # Overall magnitude
            velocity = float(np.linalg.norm(mean_b - mean_a))

        triggered = velocity > config.velocity_threshold

        if triggered:
            if velocity > config.velocity_threshold * 2:
                severity = AlertSeverity.CRITICAL
            elif velocity > config.velocity_threshold * 1.5:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM
            message = f"Velocity ({velocity:.3f}) exceeds threshold ({config.velocity_threshold})"
        else:
            severity = AlertSeverity.LOW
            message = f"Velocity ({velocity:.3f}) is within threshold ({config.velocity_threshold})"

        return TriggeredAlert(
            alert_id=alert.id,
            alert_name=alert.name,
            triggered=triggered,
            current_value=round(velocity, 4),
            threshold=config.velocity_threshold,
            severity=severity,
            message=message,
            details={
                "axis": config.axis or "overall",
                "group_a_mean": group_a["mean"],
                "group_b_mean": group_b["mean"]
            }
        )

    def _check_outlier(
        self,
        alert: AlertRule,
        group_a: Dict,
        group_b: Dict
    ) -> TriggeredAlert:
        """Check for outliers in the data."""
        config = OutlierConfig(**alert.config)

        def find_outliers(projections: List[Dict], mean: Dict, std: Dict) -> List[int]:
            """Find indices of outlier points."""
            outliers = []
            for i, proj in enumerate(projections):
                for axis in ["agency", "perceived_justice", "belonging"]:
                    if std[axis] > 0:
                        z_score = abs(proj[axis] - mean[axis]) / std[axis]
                        if z_score > config.std_threshold:
                            outliers.append(i)
                            break
            return outliers

        outliers_a = find_outliers(
            group_a["projections"],
            group_a["mean"],
            group_a["std"]
        )
        outliers_b = find_outliers(
            group_b["projections"],
            group_b["mean"],
            group_b["std"]
        )

        if config.require_both_groups:
            triggered = len(outliers_a) > 0 and len(outliers_b) > 0
        else:
            triggered = len(outliers_a) > 0 or len(outliers_b) > 0

        total_outliers = len(outliers_a) + len(outliers_b)
        if triggered:
            if total_outliers > 5:
                severity = AlertSeverity.HIGH
            elif total_outliers > 2:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            message = f"Outliers detected: {len(outliers_a)} in group A, {len(outliers_b)} in group B"
        else:
            severity = AlertSeverity.LOW
            message = "No outliers detected"

        return TriggeredAlert(
            alert_id=alert.id,
            alert_name=alert.name,
            triggered=triggered,
            current_value=float(total_outliers),
            threshold=config.std_threshold,
            severity=severity,
            message=message,
            details={
                "outlier_indices_a": outliers_a,
                "outlier_indices_b": outliers_b,
                "std_threshold": config.std_threshold
            }
        )

    async def check_all(self, request: AlertCheckRequest) -> AlertCheckResponse:
        """Check all enabled alerts against the provided text groups."""
        # Get projections for both groups
        group_a_summary = await self._get_group_projections(request.group_a.texts)
        group_b_summary = await self._get_group_projections(request.group_b.texts)

        # Compute gaps
        gaps = self._compute_gaps(group_a_summary, group_b_summary)

        # Check each enabled alert
        enabled_alerts = self.storage.list_enabled()
        triggered_alerts: List[TriggeredAlert] = []

        for alert in enabled_alerts:
            try:
                if alert.type == AlertType.GAP_THRESHOLD:
                    result = self._check_gap_threshold(
                        alert, gaps, group_a_summary, group_b_summary
                    )
                elif alert.type == AlertType.MODE_SHIFT:
                    result = self._check_mode_shift(
                        alert, group_a_summary, group_b_summary
                    )
                elif alert.type == AlertType.TRAJECTORY_VELOCITY:
                    result = self._check_trajectory_velocity(
                        alert, group_a_summary, group_b_summary
                    )
                elif alert.type == AlertType.OUTLIER_DETECTION:
                    result = self._check_outlier(
                        alert, group_a_summary, group_b_summary
                    )
                else:
                    logger.warning(f"Unknown alert type: {alert.type}")
                    continue

                triggered_alerts.append(result)
            except Exception as e:
                logger.error(f"Error checking alert {alert.id}: {e}")
                triggered_alerts.append(TriggeredAlert(
                    alert_id=alert.id,
                    alert_name=alert.name,
                    triggered=False,
                    severity=AlertSeverity.LOW,
                    message=f"Error checking alert: {str(e)}"
                ))

        # Compute summary
        triggered_count = sum(1 for a in triggered_alerts if a.triggered)
        total_count = len(triggered_alerts)
        all_clear = triggered_count == 0

        if all_clear:
            summary = f"All clear - {total_count} alerts checked, none triggered"
        else:
            summary = f"{triggered_count} of {total_count} alerts triggered"

        return AlertCheckResponse(
            triggered_alerts=triggered_alerts,
            all_clear=all_clear,
            summary=summary,
            group_a_summary={
                "name": request.group_a.name,
                "count": len(request.group_a.texts),
                "dominant_mode": group_a_summary["dominant_mode"],
                "mean": group_a_summary["mean"]
            },
            group_b_summary={
                "name": request.group_b.name,
                "count": len(request.group_b.texts),
                "dominant_mode": group_b_summary["dominant_mode"],
                "mean": group_b_summary["mean"]
            },
            gaps=gaps
        )


# Global checker instance
_alert_checker: Optional[AlertChecker] = None


def get_alert_checker() -> AlertChecker:
    """Get or create the alert checker instance."""
    global _alert_checker
    if _alert_checker is None:
        _alert_checker = AlertChecker()
    return _alert_checker


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/alerts", response_model=AlertRule)
async def create_alert(rule: AlertRuleCreate):
    """
    Create a new alert rule.

    Supported alert types:
    - gap_threshold: Trigger when gap on an axis exceeds threshold
    - mode_shift: Trigger when dominant mode changes
    - trajectory_velocity: Trigger when rate of change exceeds threshold
    - outlier_detection: Trigger when text is anomalous

    Example request:
    ```json
    {
      "name": "PR Crisis Detection",
      "description": "Alert when public vs internal justice gap exceeds threshold",
      "type": "gap_threshold",
      "config": {
        "axis": "perceived_justice",
        "threshold": 0.5,
        "direction": "exceeds"
      },
      "enabled": true
    }
    ```
    """
    storage = get_alert_storage()
    try:
        alert = storage.create(rule)
        return alert
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[AlertRule])
async def list_alerts(enabled_only: bool = False):
    """
    List all alert rules.

    Query parameters:
    - enabled_only: If true, only return enabled alerts
    """
    storage = get_alert_storage()
    if enabled_only:
        return storage.list_enabled()
    return storage.list_all()


@router.get("/alerts/{alert_id}", response_model=AlertRule)
async def get_alert(alert_id: str):
    """
    Get a specific alert rule by ID.
    """
    storage = get_alert_storage()
    alert = storage.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return alert


@router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """
    Delete an alert rule.
    """
    storage = get_alert_storage()
    if not storage.delete(alert_id):
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"success": True, "message": f"Alert {alert_id} deleted"}


@router.patch("/alerts/{alert_id}")
async def update_alert(alert_id: str, updates: Dict[str, Any]):
    """
    Update an alert rule.

    Allowed updates: name, description, config, enabled
    """
    storage = get_alert_storage()

    # Validate updates
    allowed_fields = {"name", "description", "config", "enabled"}
    invalid_fields = set(updates.keys()) - allowed_fields
    if invalid_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update fields: {invalid_fields}. Allowed: {allowed_fields}"
        )

    alert = storage.update(alert_id, updates)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return alert


@router.post("/alerts/check", response_model=AlertCheckResponse)
async def check_alerts(request: AlertCheckRequest):
    """
    Check texts against all active alerts.

    Takes two groups of texts and analyzes the gaps between them.
    Returns which alerts are triggered and their severity.

    Example request:
    ```json
    {
      "group_a": {"name": "Public", "texts": ["We value our customers..."]},
      "group_b": {"name": "Internal", "texts": ["Cut costs, maximize revenue..."]}
    }
    ```

    Example response:
    ```json
    {
      "triggered_alerts": [
        {
          "alert_id": "abc123",
          "alert_name": "PR Crisis Detection",
          "triggered": true,
          "current_value": 0.72,
          "threshold": 0.5,
          "severity": "high",
          "message": "Justice gap (0.72) exceeds threshold (0.5)"
        }
      ],
      "all_clear": false,
      "summary": "1 of 3 alerts triggered"
    }
    ```
    """
    checker = get_alert_checker()
    try:
        return await checker.check_all(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/types")
async def get_alert_types():
    """
    Get information about available alert types and their configurations.
    """
    return {
        "types": {
            "gap_threshold": {
                "description": "Trigger when gap on an axis exceeds threshold",
                "config_schema": {
                    "axis": "agency | perceived_justice | belonging",
                    "threshold": "float (0-4)",
                    "direction": "exceeds | below"
                },
                "example": {
                    "axis": "perceived_justice",
                    "threshold": 0.5,
                    "direction": "exceeds"
                }
            },
            "mode_shift": {
                "description": "Trigger when dominant narrative mode changes",
                "config_schema": {
                    "from_modes": "list of modes or null (any)",
                    "to_modes": "list of modes or null (any)",
                    "category_shift": "boolean - only alert on category changes"
                },
                "example": {
                    "from_modes": ["EMPOWERED", "HEROIC"],
                    "to_modes": ["VICTIM", "CYNICAL"],
                    "category_shift": True
                }
            },
            "trajectory_velocity": {
                "description": "Trigger when rate of change exceeds threshold",
                "config_schema": {
                    "velocity_threshold": "float",
                    "axis": "axis name or null (overall magnitude)"
                },
                "example": {
                    "velocity_threshold": 0.3,
                    "axis": None
                }
            },
            "outlier_detection": {
                "description": "Trigger when texts are anomalous",
                "config_schema": {
                    "std_threshold": "float (default 2.0)",
                    "require_both_groups": "boolean"
                },
                "example": {
                    "std_threshold": 2.5,
                    "require_both_groups": False
                }
            }
        },
        "severity_levels": ["low", "medium", "high", "critical"]
    }
