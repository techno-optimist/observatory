"""
Observer Chat API - Conversational interface for narrative analysis

Provides a session-aware chat endpoint that:
- Accepts user messages and maintains context
- Provides intelligent responses about narrative analysis
- Integrates with /v2/analyze for text analysis
- Recommends appropriate lenses based on keywords
- Detects and analyzes URLs
- Returns structured responses with optional actions

Session-Aware Features:
- Compares current analysis to previous analyses in session
- Detects recurring patterns (modes, dimension trends, issues)
- Tracks trajectory of narrative health over time
- Generates recommendations based on cumulative patterns
- References specific findings from previous analyses
- Provides goal-aligned insights when session_goal is set

Usage:
    POST /api/observer/chat
    {
        "messages": [{"role": "user", "content": "..."}],
        "context": {
            "lens_id": "denial-messaging",
            "prompt": "...",
            "session_history": [
                {"mode": {"primary_mode": "VICTIM"}, "vector": {...}, "label": "pricing page"},
                {"mode": {"primary_mode": "NEUTRAL"}, "vector": {...}, "url": "https://..."}
            ],
            "session_goal": "Improve agency in customer communications"
        }
    }

Response:
    {
        "response": "I analyzed that text... Compared to Previous: This has HIGHER agency...",
        "actions": [...],
        "lens_suggestion": "denial-messaging",
        "analysis_results": {...},
        "session_insights": {"analysis_number": 3, "patterns_count": 2, ...},
        "comparisons": [{"comparison_type": "dimension", "message": "...", ...}],
        "patterns_detected": [{"pattern_type": "recurring_mode", "description": "...", ...}],
        "trajectory_update": {"overall_direction": "improving", "message": "...", ...}
    }
"""

import re
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
import httpx

logger = logging.getLogger(__name__)
router = APIRouter()

# =============================================================================
# LENS DEFINITIONS (matching observer_mcp.py)
# =============================================================================

LENS_DEFINITIONS = {
    "denial-messaging": {
        "id": "denial-messaging",
        "name": "Denial Messaging QA",
        "industry": "Fintech / Insurance",
        "description": "Detect shadow-zone language in claim denials before they trigger regulatory scrutiny or viral backlash",
        "keywords": ["denial", "denied", "claim", "reject", "rejected", "insurance", "coverage", "policy", "pre-existing", "excluded", "not covered"],
        "primary_dimensions": ["perceived_justice", "agency", "belonging"],
        "watch_modes": ["INSTITUTIONAL_DECAY", "CYNICAL_BURNOUT", "VICTIM"],
        "alert_thresholds": {
            "perceived_justice": -0.8,
            "agency": -1.0
        }
    },
    "crisis-preflight": {
        "id": "crisis-preflight",
        "name": "Crisis Pre-Flight",
        "industry": "PR / Communications",
        "description": "Test crisis statements before release to predict narrative trajectory",
        "keywords": ["crisis", "pr", "public relations", "statement", "announcement", "press", "media", "backlash", "apology", "incident"],
        "primary_dimensions": ["agency", "perceived_justice", "belonging"],
        "watch_modes": ["VICTIM", "TRANSITIONAL", "NEUTRAL"],
        "alert_thresholds": {
            "belonging": -0.5,
            "agency": 0.2
        }
    },
    "support-triage": {
        "id": "support-triage",
        "name": "Support Narrative Triage",
        "industry": "Customer Success",
        "description": "Prioritize support tickets by narrative distress signals, not just keywords",
        "keywords": ["support", "ticket", "help", "customer", "issue", "problem", "frustrated", "angry", "escalate", "churn", "cancel"],
        "primary_dimensions": ["belonging", "agency", "perceived_justice"],
        "watch_modes": ["QUIET_QUITTING", "NEUTRAL", "TRANSITIONAL"],
        "alert_thresholds": {
            "belonging": -1.0,
            "agency": -0.8
        }
    },
    "general-analysis": {
        "id": "general-analysis",
        "name": "General Narrative Analysis",
        "description": "Balanced analysis across all dimensions",
        "keywords": [],
        "primary_dimensions": ["agency", "perceived_justice", "belonging"],
        "watch_modes": [],
        "alert_thresholds": {}
    },
    "engagement-health": {
        "id": "engagement-health",
        "name": "Employee Engagement Health",
        "description": "Measure organizational narrative health from employee communications",
        "keywords": ["employee", "engagement", "morale", "culture", "team", "workplace", "burnout", "motivation"],
        "primary_dimensions": ["belonging", "agency", "perceived_justice"],
        "watch_modes": ["QUIET_QUITTING", "CYNICAL_BURNOUT", "INSTITUTIONAL_DECAY"],
        "alert_thresholds": {
            "belonging": -0.5,
            "agency": -0.3
        }
    }
}


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatContext(BaseModel):
    """Context for the chat session."""
    lens_id: Optional[str] = Field(None, description="Currently selected lens ID")
    prompt: Optional[str] = Field(None, description="Additional context/prompt")
    session_history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous analyses summaries from this session")
    session_goal: Optional[str] = Field(None, description="User's stated objective for this session")


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    context: Optional[ChatContext] = Field(None, description="Session context")


class ChatAction(BaseModel):
    """An action the UI can take based on chat response."""
    type: str = Field(..., description="Action type: 'analyze', 'select_lens', 'open_url', 'show_results'")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Action payload")
    label: Optional[str] = Field(None, description="Human-readable label for the action")


class SessionInsight(BaseModel):
    """Cross-analysis insight from session history."""
    type: str = Field(..., description="Insight type: 'comparison', 'pattern', 'trajectory', 'recommendation'")
    message: str = Field(..., description="Human-readable insight message")
    severity: Optional[str] = Field(None, description="Severity: 'info', 'warning', 'critical'")
    references: Optional[List[int]] = Field(None, description="Indices of referenced analyses in session history")


class PatternDetection(BaseModel):
    """Pattern detected across multiple analyses in session."""
    pattern_type: str = Field(..., description="Type of pattern: 'recurring_mode', 'dimension_trend', 'issue_frequency'")
    description: str = Field(..., description="Human-readable pattern description")
    occurrences: int = Field(..., description="Number of times pattern was observed")
    affected_analyses: List[int] = Field(default_factory=list, description="Indices of analyses showing this pattern")
    severity: str = Field(default="info", description="Pattern severity: 'info', 'warning', 'critical'")


class TrajectoryUpdate(BaseModel):
    """How current analysis affects session trajectory."""
    dimension: str = Field(..., description="Dimension being tracked")
    direction: str = Field(..., description="Trend direction: 'improving', 'declining', 'stable', 'volatile'")
    change_from_last: Optional[float] = Field(None, description="Change from previous analysis")
    session_trend: Optional[float] = Field(None, description="Overall trend across session")
    message: str = Field(..., description="Human-readable trajectory message")


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str = Field(..., description="The assistant's response text")
    actions: Optional[List[ChatAction]] = Field(None, description="Suggested actions for the UI")
    lens_suggestion: Optional[str] = Field(None, description="Recommended lens ID based on conversation")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Results from any triggered analysis")
    site_analysis: Optional[Dict[str, Any]] = Field(None, description="Full site analysis report for URLs")
    # Session-aware fields
    session_insights: Optional[Dict[str, Any]] = Field(None, description="Cross-analysis insights from session history")
    comparisons: Optional[List[Dict[str, Any]]] = Field(None, description="Comparisons to previous analyses in session")
    patterns_detected: Optional[List[Dict[str, Any]]] = Field(None, description="Patterns detected across session")
    trajectory_update: Optional[Dict[str, Any]] = Field(None, description="How current analysis affects trajectory")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def recommend_lens(text: str) -> Optional[str]:
    """Recommend a lens based on keywords in the text."""
    text_lower = text.lower()

    # Score each lens based on keyword matches
    scores = {}
    for lens_id, lens in LENS_DEFINITIONS.items():
        if lens_id == "general-analysis":
            continue  # Skip default

        score = 0
        for keyword in lens.get("keywords", []):
            if keyword in text_lower:
                score += 1

        if score > 0:
            scores[lens_id] = score

    if scores:
        return max(scores, key=scores.get)

    return None


def get_lens_info(lens_id: str) -> Dict[str, Any]:
    """Get information about a specific lens."""
    return LENS_DEFINITIONS.get(lens_id, LENS_DEFINITIONS["general-analysis"])


async def analyze_text_internal(text: str, include_forces: bool = True) -> Dict[str, Any]:
    """Call the /v2/analyze endpoint internally."""
    try:
        # Import here to avoid circular imports
        from api_extensions import enhanced_analyze, EnhancedAnalyzeRequest

        request = EnhancedAnalyzeRequest(
            text=text,
            include_uncertainty=True,
            include_legacy_mode=True,
            include_force_field=include_forces
        )

        result = await enhanced_analyze(request)
        return result
    except Exception as e:
        logger.error(f"Internal analysis failed: {e}")
        raise


async def analyze_url(url: str, lens_id: str = "general-analysis") -> Dict[str, Any]:
    """Fetch and analyze content from a URL."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ObserverBot/1.0)"
            })
            response.raise_for_status()
            html = response.text
    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}

    # Extract text from HTML (simple extraction)
    # Remove scripts, styles, nav, footer
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Extract paragraphs and headings
    segments = []
    for match in re.finditer(r'<(p|h[1-6]|li|blockquote)[^>]*>(.*?)</\1>', html, re.DOTALL | re.IGNORECASE):
        text = re.sub(r'<[^>]+>', '', match.group(2))  # Remove tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        if len(text) > 50:  # Only meaningful segments
            segments.append(text)

    if not segments:
        return {"error": "No meaningful text found at URL"}

    # Limit to 10 segments for chat context
    segments = segments[:10]

    # Analyze each segment
    lens = get_lens_info(lens_id)
    results = []
    alerts = []

    for i, text in enumerate(segments):
        try:
            result = await analyze_text_internal(text)

            # Check lens thresholds
            if "vector" in result:
                coords = result["vector"]
                for dim, threshold in lens.get("alert_thresholds", {}).items():
                    coord_val = coords.get(dim) or coords.get("perceived_justice" if dim == "fairness" else dim, 0)
                    if coord_val < threshold:
                        alerts.append({
                            "segment": i,
                            "dimension": dim,
                            "value": coord_val,
                            "threshold": threshold
                        })

            results.append({
                "index": i,
                "text_preview": text[:80] + "..." if len(text) > 80 else text,
                "mode": result.get("mode", {}).get("primary_mode"),
                "coordinates": result.get("vector")
            })
        except Exception as e:
            logger.warning(f"Failed to analyze segment {i}: {e}")

    # Compute summary
    valid_results = [r for r in results if r.get("coordinates")]
    if valid_results:
        avg_coords = {}
        for dim in ["agency", "perceived_justice", "belonging"]:
            values = [r["coordinates"].get(dim, 0) for r in valid_results if r["coordinates"].get(dim) is not None]
            if values:
                avg_coords[dim] = sum(values) / len(values)

        mode_dist = {}
        for r in valid_results:
            mode = r.get("mode", "UNKNOWN")
            mode_dist[mode] = mode_dist.get(mode, 0) + 1

        return {
            "url": url,
            "segments_analyzed": len(valid_results),
            "average_coordinates": avg_coords,
            "mode_distribution": mode_dist,
            "alerts": alerts,
            "top_results": results[:5]
        }

    return {"error": "All segment analyses failed"}


async def analyze_site_deep(url: str) -> Dict[str, Any]:
    """Perform comprehensive website analysis by importing and calling site analysis directly."""
    try:
        # Check if projection is available first
        from main import current_projection
        print(f"[DEBUG] analyze_site_deep called for {url}, projection={current_projection is not None}")
        if current_projection is None:
            print("[DEBUG] No projection available!")
            return {"error": "No projection trained for site analysis"}

        # Import the site analysis module and call directly (avoids HTTP roundtrip issues)
        from api_site_analysis import perform_site_analysis
        print(f"[DEBUG] Starting site analysis for {url}")
        result = await perform_site_analysis(url)
        print(f"[DEBUG] Site analysis result keys: {list(result.keys())}")
        return result
    except ImportError as e:
        print(f"[DEBUG] ImportError: {e}")
        return {"error": "Site analysis module not available"}
    except asyncio.TimeoutError:
        print("[DEBUG] TimeoutError")
        return {"error": "Site analysis timed out - the site may be too large or slow"}
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Site analysis failed: {str(e)}"}


# =============================================================================
# SESSION SYNTHESIS FUNCTIONS
# =============================================================================

def analyze_session_patterns(
    session_history: List[Dict[str, Any]],
    current_analysis: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Detect recurring patterns across session analyses.

    Returns list of patterns with:
    - pattern_type: 'recurring_mode', 'dimension_trend', 'issue_frequency', 'language_pattern'
    - description: Human-readable description
    - occurrences: How many times observed
    - affected_analyses: Which analyses show this pattern
    - severity: 'info', 'warning', 'critical'
    """
    if not session_history:
        return []

    patterns = []

    # Track mode occurrences
    mode_counts: Dict[str, List[int]] = {}
    for i, analysis in enumerate(session_history):
        mode = analysis.get("mode") or analysis.get("primary_mode")
        if isinstance(analysis.get("mode"), dict):
            mode = analysis["mode"].get("primary_mode")
        if mode:
            if mode not in mode_counts:
                mode_counts[mode] = []
            mode_counts[mode].append(i)

    # Check current analysis mode
    if current_analysis:
        current_mode = None
        if isinstance(current_analysis.get("mode"), dict):
            current_mode = current_analysis["mode"].get("primary_mode")
        elif current_analysis.get("mode"):
            current_mode = current_analysis["mode"]
        elif current_analysis.get("overall_mode"):
            current_mode = current_analysis["overall_mode"]

        if current_mode:
            if current_mode not in mode_counts:
                mode_counts[current_mode] = []
            mode_counts[current_mode].append(len(session_history))

    # Detect recurring shadow modes
    shadow_modes = ["VICTIM", "CYNICAL_BURNOUT", "INSTITUTIONAL_DECAY", "QUIET_QUITTING"]
    for mode, indices in mode_counts.items():
        if len(indices) >= 2:
            severity = "warning" if mode in shadow_modes else "info"
            if len(indices) >= 3 and mode in shadow_modes:
                severity = "critical"

            patterns.append({
                "pattern_type": "recurring_mode",
                "description": f"'{mode}' mode detected {len(indices)} times across your analyses",
                "occurrences": len(indices),
                "affected_analyses": indices,
                "severity": severity,
                "mode": mode
            })

    # Track dimension trends across session
    dimension_values: Dict[str, List[tuple]] = {"agency": [], "perceived_justice": [], "belonging": []}

    for i, analysis in enumerate(session_history):
        vector = analysis.get("vector") or analysis.get("coordinates") or analysis.get("average_coordinates")
        if vector:
            for dim in dimension_values.keys():
                if dim in vector and vector[dim] is not None:
                    dimension_values[dim].append((i, vector[dim]))

    # Add current analysis dimensions
    if current_analysis:
        vector = current_analysis.get("vector") or current_analysis.get("average_coordinates")
        if vector:
            for dim in dimension_values.keys():
                if dim in vector and vector[dim] is not None:
                    dimension_values[dim].append((len(session_history), vector[dim]))

    # Detect dimension patterns
    for dim, values in dimension_values.items():
        if len(values) >= 3:
            # Check for consistent negative trend
            recent_values = [v[1] for v in values[-4:]]
            if len(recent_values) >= 3:
                declining = all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1))
                improving = all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1))

                if declining:
                    avg = sum(recent_values) / len(recent_values)
                    severity = "critical" if avg < -0.3 else "warning"
                    patterns.append({
                        "pattern_type": "dimension_trend",
                        "description": f"{dim.replace('_', ' ').title()} has been declining over your last {len(recent_values)} analyses",
                        "occurrences": len(recent_values),
                        "affected_analyses": [v[0] for v in values[-4:]],
                        "severity": severity,
                        "dimension": dim,
                        "direction": "declining",
                        "values": recent_values
                    })
                elif improving:
                    patterns.append({
                        "pattern_type": "dimension_trend",
                        "description": f"{dim.replace('_', ' ').title()} is improving - up over your last {len(recent_values)} analyses",
                        "occurrences": len(recent_values),
                        "affected_analyses": [v[0] for v in values[-4:]],
                        "severity": "info",
                        "dimension": dim,
                        "direction": "improving",
                        "values": recent_values
                    })

            # Check for consistently low values
            all_values = [v[1] for v in values]
            if all(v < -0.3 for v in all_values):
                patterns.append({
                    "pattern_type": "dimension_trend",
                    "description": f"{dim.replace('_', ' ').title()} is consistently negative across all your analyses",
                    "occurrences": len(values),
                    "affected_analyses": [v[0] for v in values],
                    "severity": "critical",
                    "dimension": dim,
                    "direction": "consistently_low"
                })

    # Track issue/alert frequency
    issue_types: Dict[str, List[int]] = {}
    for i, analysis in enumerate(session_history):
        alerts = analysis.get("alerts") or analysis.get("lens_alerts") or analysis.get("critical_issues") or []
        for alert in alerts:
            if isinstance(alert, dict):
                issue_key = alert.get("dimension") or alert.get("type") or alert.get("issue_type") or "unknown"
            else:
                issue_key = str(alert)
            if issue_key not in issue_types:
                issue_types[issue_key] = []
            issue_types[issue_key].append(i)

    for issue, indices in issue_types.items():
        if len(indices) >= 2:
            patterns.append({
                "pattern_type": "issue_frequency",
                "description": f"'{issue}' issue detected in {len(indices)} of your analyses",
                "occurrences": len(indices),
                "affected_analyses": indices,
                "severity": "warning" if len(indices) >= 3 else "info",
                "issue_type": issue
            })

    return patterns


def generate_comparisons(
    current_analysis: Dict[str, Any],
    session_history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate comparative insights between current analysis and session history.

    Returns list of comparisons with:
    - comparison_type: 'dimension', 'mode', 'overall'
    - message: Human-readable comparison
    - current_value: Current analysis value
    - previous_value: Previous analysis value
    - previous_index: Index in session history
    - previous_label: Label for previous analysis (e.g., "your pricing page")
    """
    if not session_history or not current_analysis:
        return []

    comparisons = []

    # Get current analysis values
    current_vector = current_analysis.get("vector") or current_analysis.get("average_coordinates") or {}
    current_mode = None
    if isinstance(current_analysis.get("mode"), dict):
        current_mode = current_analysis["mode"].get("primary_mode")
    elif current_analysis.get("mode"):
        current_mode = current_analysis["mode"]
    elif current_analysis.get("overall_mode"):
        current_mode = current_analysis["overall_mode"]

    # Compare with most recent analysis
    if session_history:
        last = session_history[-1]
        last_vector = last.get("vector") or last.get("coordinates") or last.get("average_coordinates") or {}
        last_label = last.get("label") or last.get("url") or last.get("text_preview") or f"analysis #{len(session_history)}"
        if len(last_label) > 50:
            last_label = last_label[:47] + "..."

        last_mode = None
        if isinstance(last.get("mode"), dict):
            last_mode = last["mode"].get("primary_mode")
        elif last.get("mode"):
            last_mode = last["mode"]
        elif last.get("primary_mode"):
            last_mode = last["primary_mode"]

        # Compare dimensions
        for dim in ["agency", "perceived_justice", "belonging"]:
            if dim in current_vector and dim in last_vector:
                current_val = current_vector[dim]
                last_val = last_vector[dim]
                if current_val is not None and last_val is not None:
                    diff = current_val - last_val
                    dim_label = dim.replace("_", " ").title()

                    if abs(diff) >= 0.2:  # Significant difference
                        direction = "HIGHER" if diff > 0 else "LOWER"
                        comparisons.append({
                            "comparison_type": "dimension",
                            "dimension": dim,
                            "message": f"This has {direction} {dim_label.lower()} than your previous analysis ({last_label})",
                            "current_value": round(current_val, 2),
                            "previous_value": round(last_val, 2),
                            "change": round(diff, 2),
                            "previous_index": len(session_history) - 1,
                            "previous_label": last_label
                        })

        # Compare modes
        if current_mode and last_mode and current_mode != last_mode:
            shadow_modes = ["VICTIM", "CYNICAL_BURNOUT", "INSTITUTIONAL_DECAY", "QUIET_QUITTING"]
            severity = "info"

            if current_mode in shadow_modes and last_mode not in shadow_modes:
                severity = "warning"
                message = f"This shifted INTO shadow mode ({current_mode}) from {last_mode} in your previous analysis"
            elif last_mode in shadow_modes and current_mode not in shadow_modes:
                severity = "info"
                message = f"This moved OUT of shadow mode - from {last_mode} to {current_mode}"
            else:
                message = f"Mode shifted from {last_mode} to {current_mode} compared to your previous analysis"

            comparisons.append({
                "comparison_type": "mode",
                "message": message,
                "current_mode": current_mode,
                "previous_mode": last_mode,
                "previous_index": len(session_history) - 1,
                "previous_label": last_label,
                "severity": severity
            })

    # Find similar analyses in history
    for i, hist in enumerate(session_history[:-1] if len(session_history) > 1 else session_history):
        hist_vector = hist.get("vector") or hist.get("coordinates") or hist.get("average_coordinates") or {}
        hist_label = hist.get("label") or hist.get("url") or hist.get("text_preview") or f"analysis #{i+1}"
        if len(hist_label) > 50:
            hist_label = hist_label[:47] + "..."

        # Check for same mode
        hist_mode = None
        if isinstance(hist.get("mode"), dict):
            hist_mode = hist["mode"].get("primary_mode")
        elif hist.get("mode"):
            hist_mode = hist["mode"]
        elif hist.get("primary_mode"):
            hist_mode = hist["primary_mode"]

        if current_mode and hist_mode and current_mode == hist_mode:
            shadow_modes = ["VICTIM", "CYNICAL_BURNOUT", "INSTITUTIONAL_DECAY", "QUIET_QUITTING"]
            if current_mode in shadow_modes:
                comparisons.append({
                    "comparison_type": "mode_match",
                    "message": f"Same shadow mode ({current_mode}) as {hist_label}",
                    "current_mode": current_mode,
                    "matched_index": i,
                    "matched_label": hist_label,
                    "severity": "warning"
                })

        # Check for similar low dimension values
        for dim in ["agency", "perceived_justice", "belonging"]:
            if dim in current_vector and dim in hist_vector:
                current_val = current_vector.get(dim)
                hist_val = hist_vector.get(dim)
                if current_val is not None and hist_val is not None:
                    if current_val < -0.3 and hist_val < -0.3:
                        dim_label = dim.replace("_", " ").lower()
                        comparisons.append({
                            "comparison_type": "dimension_match",
                            "message": f"Same low {dim_label} issue as {hist_label}",
                            "dimension": dim,
                            "current_value": round(current_val, 2),
                            "matched_value": round(hist_val, 2),
                            "matched_index": i,
                            "matched_label": hist_label,
                            "severity": "warning"
                        })

    return comparisons


def calculate_trajectory(
    current_analysis: Dict[str, Any],
    session_history: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Calculate how the current analysis affects the session trajectory.

    Returns trajectory update with dimension trends and messages.
    """
    if not session_history or not current_analysis:
        return None

    current_vector = current_analysis.get("vector") or current_analysis.get("average_coordinates") or {}
    if not current_vector:
        return None

    trajectory = {
        "dimensions": {},
        "overall_direction": "stable",
        "message": "",
        "analysis_count": len(session_history) + 1
    }

    # Collect historical dimension values
    dimension_history: Dict[str, List[float]] = {"agency": [], "perceived_justice": [], "belonging": []}

    for hist in session_history:
        vector = hist.get("vector") or hist.get("coordinates") or hist.get("average_coordinates") or {}
        for dim in dimension_history.keys():
            if dim in vector and vector[dim] is not None:
                dimension_history[dim].append(vector[dim])

    # Add current values
    for dim in dimension_history.keys():
        if dim in current_vector and current_vector[dim] is not None:
            dimension_history[dim].append(current_vector[dim])

    # Calculate trends for each dimension
    directions = []
    for dim, values in dimension_history.items():
        if len(values) >= 2:
            current_val = values[-1]
            prev_val = values[-2]
            change = current_val - prev_val

            # Calculate overall session trend
            if len(values) >= 3:
                first_third = values[:len(values)//3] if len(values) >= 3 else values[:1]
                last_third = values[-(len(values)//3):] if len(values) >= 3 else values[-1:]
                session_trend = (sum(last_third) / len(last_third)) - (sum(first_third) / len(first_third))
            else:
                session_trend = change

            if abs(change) < 0.1:
                direction = "stable"
            elif change > 0:
                direction = "improving"
            else:
                direction = "declining"

            # Check for volatility
            if len(values) >= 3:
                changes = [values[i+1] - values[i] for i in range(len(values)-1)]
                if any(c > 0 for c in changes) and any(c < 0 for c in changes):
                    if max(abs(c) for c in changes) > 0.3:
                        direction = "volatile"

            dim_label = dim.replace("_", " ").title()
            if direction == "improving":
                message = f"{dim_label} trending upward (+{change:.2f} from last)"
            elif direction == "declining":
                message = f"{dim_label} trending downward ({change:.2f} from last)"
            elif direction == "volatile":
                message = f"{dim_label} showing volatility across analyses"
            else:
                message = f"{dim_label} holding steady"

            trajectory["dimensions"][dim] = {
                "direction": direction,
                "change_from_last": round(change, 3),
                "session_trend": round(session_trend, 3),
                "current_value": round(current_val, 3),
                "message": message
            }

            directions.append((direction, change))

    # Determine overall direction
    if not directions:
        trajectory["overall_direction"] = "unknown"
        trajectory["message"] = "Not enough data to determine trajectory"
    else:
        declining_count = sum(1 for d, c in directions if d == "declining")
        improving_count = sum(1 for d, c in directions if d == "improving")

        if declining_count >= 2:
            trajectory["overall_direction"] = "declining"
            trajectory["message"] = f"Your content is trending toward lower narrative health over the last {len(session_history) + 1} analyses"
        elif improving_count >= 2:
            trajectory["overall_direction"] = "improving"
            trajectory["message"] = f"Your content is improving - positive trends across the last {len(session_history) + 1} analyses"
        elif any(d == "volatile" for d, c in directions):
            trajectory["overall_direction"] = "volatile"
            trajectory["message"] = "Your analyses show volatility - consider focusing on consistency"
        else:
            trajectory["overall_direction"] = "stable"
            trajectory["message"] = "Narrative health is holding steady across your analyses"

    return trajectory


def generate_session_recommendations(
    current_analysis: Dict[str, Any],
    session_history: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    session_goal: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate recommendations based on cumulative session patterns.

    Returns prioritized list of recommendations.
    """
    recommendations = []

    # Prioritize by pattern severity and frequency
    critical_patterns = [p for p in patterns if p.get("severity") == "critical"]
    warning_patterns = [p for p in patterns if p.get("severity") == "warning"]

    # Address critical patterns first
    for pattern in critical_patterns:
        if pattern["pattern_type"] == "recurring_mode":
            mode = pattern.get("mode", "shadow")
            recommendations.append({
                "priority": 1,
                "type": "pattern_fix",
                "message": f"Critical: {mode} mode detected {pattern['occurrences']} times. Review language for agency-draining phrases.",
                "action": "Review highlighted segments across all affected analyses",
                "affected_analyses": pattern["affected_analyses"]
            })
        elif pattern["pattern_type"] == "dimension_trend" and pattern.get("direction") == "consistently_low":
            dim = pattern.get("dimension", "unknown")
            recommendations.append({
                "priority": 1,
                "type": "dimension_fix",
                "message": f"Critical: {dim.replace('_', ' ').title()} is negative across ALL your analyses. This is a systemic issue.",
                "action": f"Focus on {dim.replace('_', ' ')} language improvements as top priority",
                "affected_analyses": pattern["affected_analyses"]
            })

    # Address warning patterns
    for pattern in warning_patterns:
        if pattern["pattern_type"] == "issue_frequency":
            issue = pattern.get("issue_type", "issue")
            recommendations.append({
                "priority": 2,
                "type": "recurring_issue",
                "message": f"'{issue}' appears in {pattern['occurrences']} analyses - this is a recurring problem.",
                "action": "Address root cause rather than individual instances",
                "affected_analyses": pattern["affected_analyses"]
            })
        elif pattern["pattern_type"] == "dimension_trend" and pattern.get("direction") == "declining":
            dim = pattern.get("dimension", "unknown")
            recommendations.append({
                "priority": 2,
                "type": "trend_reversal",
                "message": f"{dim.replace('_', ' ').title()} is declining - your changes may be having opposite effect.",
                "action": "Review recent edits and consider reverting problematic changes",
                "affected_analyses": pattern["affected_analyses"]
            })

    # Goal-specific recommendations
    if session_goal:
        goal_lower = session_goal.lower()

        if "agency" in goal_lower or "empower" in goal_lower:
            agency_issues = [p for p in patterns if p.get("dimension") == "agency" or "agency" in p.get("description", "").lower()]
            if agency_issues:
                recommendations.append({
                    "priority": 1,
                    "type": "goal_alignment",
                    "message": f"Your goal mentions agency, but {len(agency_issues)} patterns show agency issues.",
                    "action": "Replace passive voice, add action verbs, emphasize reader control",
                    "goal": session_goal
                })

        if "trust" in goal_lower or "justice" in goal_lower or "fair" in goal_lower:
            justice_issues = [p for p in patterns if p.get("dimension") == "perceived_justice"]
            if justice_issues:
                recommendations.append({
                    "priority": 1,
                    "type": "goal_alignment",
                    "message": f"Your goal mentions trust/fairness, but analyses show justice perception issues.",
                    "action": "Add transparency language, explain reasoning, acknowledge concerns",
                    "goal": session_goal
                })

    # Sort by priority
    recommendations.sort(key=lambda x: x.get("priority", 99))

    return recommendations


def synthesize_session_response(
    base_response: str,
    current_analysis: Optional[Dict[str, Any]],
    session_history: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    trajectory: Optional[Dict[str, Any]],
    session_goal: Optional[str] = None
) -> str:
    """
    Enhance the base response with session-aware insights.

    Makes the response feel like talking to an analyst who remembers everything.
    """
    if not session_history and not patterns and not comparisons:
        return base_response

    enhanced_parts = [base_response]

    # Add comparison insights (most recent first)
    if comparisons and current_analysis:
        comparison_section = []

        # Prioritize dimension comparisons
        dim_comparisons = [c for c in comparisons if c["comparison_type"] == "dimension" and abs(c.get("change", 0)) >= 0.2]
        if dim_comparisons:
            for comp in dim_comparisons[:2]:  # Limit to top 2
                comparison_section.append(f"- {comp['message']}")

        # Add mode comparisons
        mode_comparisons = [c for c in comparisons if c["comparison_type"] in ["mode", "mode_match"]]
        for comp in mode_comparisons[:1]:  # Limit to 1
            comparison_section.append(f"- {comp['message']}")

        if comparison_section:
            enhanced_parts.append("\n**Compared to Previous:**\n" + "\n".join(comparison_section))

    # Add pattern detection callouts
    if patterns:
        critical_patterns = [p for p in patterns if p.get("severity") == "critical"]
        warning_patterns = [p for p in patterns if p.get("severity") == "warning"]

        if critical_patterns:
            pattern_section = []
            for p in critical_patterns[:2]:
                pattern_section.append(f"- {p['description']}")
            enhanced_parts.append("\n**Patterns Detected:**\n" + "\n".join(pattern_section))
        elif warning_patterns:
            pattern_section = []
            for p in warning_patterns[:2]:
                pattern_section.append(f"- {p['description']}")
            enhanced_parts.append("\n**Patterns Detected:**\n" + "\n".join(pattern_section))

    # Add trajectory insight
    if trajectory and trajectory.get("message"):
        direction = trajectory.get("overall_direction", "stable")
        if direction in ["declining", "volatile"]:
            enhanced_parts.append(f"\n**Trajectory Alert:** {trajectory['message']}")
        elif direction == "improving":
            enhanced_parts.append(f"\n**Good News:** {trajectory['message']}")

    # Add goal-related insight
    if session_goal and current_analysis:
        enhanced_parts.append(f"\n*Tracking toward your goal: \"{session_goal}\"*")

    return "\n".join(enhanced_parts)


def generate_response(
    user_message: str,
    context: Optional[ChatContext],
    analysis_result: Optional[Dict[str, Any]] = None,
    url_analysis: Optional[Dict[str, Any]] = None,
    lens_recommendation: Optional[str] = None
) -> str:
    """Generate a conversational response based on input and context."""

    # If we have analysis results, explain them
    if analysis_result and "mode" in analysis_result:
        mode_info = analysis_result.get("mode", {})
        primary_mode = mode_info.get("primary_mode", "Unknown")
        confidence = mode_info.get("confidence", 0)
        vector = analysis_result.get("vector", {})

        response = f"I analyzed that text. Here's what I found:\n\n"
        response += f"**Narrative Mode:** {primary_mode}"
        if confidence < 0.5:
            response += " (low confidence - this could be ambiguous)"
        response += "\n\n"

        response += "**Dimension Scores:**\n"
        for dim, val in vector.items():
            assessment = "positive" if val > 0.3 else "negative" if val < -0.3 else "neutral"
            response += f"- {dim.replace('_', ' ').title()}: {val:.2f} ({assessment})\n"

        if mode_info.get("stability_warning"):
            response += f"\n*Note: {mode_info['stability_warning']}*\n"

        return response

    # If we have URL analysis results
    if url_analysis:
        if "error" in url_analysis:
            return f"I tried to analyze that URL but encountered an issue: {url_analysis['error']}"

        response = f"I analyzed the content from **{url_analysis.get('url', 'the URL')}**.\n\n"
        response += f"**Segments analyzed:** {url_analysis.get('segments_analyzed', 0)}\n\n"

        if url_analysis.get("average_coordinates"):
            response += "**Average Scores:**\n"
            for dim, val in url_analysis["average_coordinates"].items():
                response += f"- {dim.replace('_', ' ').title()}: {val:.2f}\n"

        if url_analysis.get("mode_distribution"):
            response += "\n**Mode Distribution:**\n"
            for mode, count in sorted(url_analysis["mode_distribution"].items(), key=lambda x: -x[1]):
                response += f"- {mode}: {count}\n"

        alerts = url_analysis.get("alerts", [])
        if alerts:
            response += f"\n**Alerts:** {len(alerts)} threshold violations detected.\n"

        return response

    # Handle lens recommendation
    if lens_recommendation:
        lens = get_lens_info(lens_recommendation)
        response = f"Based on your message, I'd recommend the **{lens['name']}** lens.\n\n"
        response += f"{lens['description']}\n\n"
        response += f"This lens focuses on: {', '.join(lens['primary_dimensions'])}\n\n"
        response += "Would you like me to apply this lens to analyze some text?"
        return response

    # General help/guidance responses
    msg_lower = user_message.lower()

    if any(word in msg_lower for word in ["help", "how", "what can"]):
        return """I can help you with narrative analysis! Here's what I can do:

**Analyze Text:**
Just paste any text and I'll analyze its narrative dimensions (agency, perceived justice, belonging) and classify its mode.

**Analyze URLs:**
Share a URL and I'll fetch the content and provide a narrative analysis of the page.

**Choose a Lens:**
I can recommend and apply specialized lenses for different use cases:
- **Denial Messaging QA** - For insurance/fintech claim denials
- **Crisis Pre-Flight** - For testing PR statements before release
- **Support Triage** - For prioritizing support tickets by distress signals
- **Employee Engagement** - For organizational narrative health

**Interpret Results:**
Ask me about what different modes mean, how to interpret scores, or what actions to take based on results.

What would you like to analyze?"""

    if any(word in msg_lower for word in ["lens", "lenses", "which lens"]):
        return """Here are the available lenses:

**Killer App Lenses:**

1. **Denial Messaging QA** (`denial-messaging`)
   - For: Fintech / Insurance
   - Detects shadow-zone language in claim denials
   - Watch for: INSTITUTIONAL_DECAY, CYNICAL_BURNOUT, VICTIM

2. **Crisis Pre-Flight** (`crisis-preflight`)
   - For: PR / Communications
   - Tests crisis statements before release
   - Watch for: VICTIM, TRANSITIONAL, NEUTRAL

3. **Support Triage** (`support-triage`)
   - For: Customer Success
   - Prioritizes tickets by narrative distress
   - Watch for: QUIET_QUITTING, NEUTRAL, TRANSITIONAL

**Standard Lenses:**

4. **General Analysis** (`general-analysis`)
   - Balanced analysis across all dimensions

5. **Employee Engagement** (`engagement-health`)
   - Organizational narrative health from employee communications

Describe your use case and I'll recommend the best lens!"""

    if any(word in msg_lower for word in ["mode", "modes", "what is", "explain"]):
        return """Narrative modes represent patterns in how text expresses agency, justice, and belonging:

**Positive Modes:**
- EMPOWERED - High agency, positive outlook
- TRUSTING - Faith in systems/institutions
- BELONGING - Strong community connection
- HOPEFUL - Future-oriented, optimistic

**Shadow Modes:**
- VICTIM - Low agency, perceived injustice
- CYNICAL_BURNOUT - Exhausted, disillusioned
- INSTITUTIONAL_DECAY - Distrust in systems
- QUIET_QUITTING - Disengaged but present

**Transition Modes:**
- TRANSITIONAL - In flux between states
- CONFLICTED - Mixed signals
- NEUTRAL - Balanced/flat affect

Low confidence classifications may indicate genuinely ambiguous text. Boundary cases can flip with small changes."""

    # Default: prompt for text to analyze
    current_lens = context.lens_id if context else None
    lens_name = get_lens_info(current_lens)["name"] if current_lens else "General Analysis"

    return f"I'm ready to analyze text using the **{lens_name}** lens. Paste any text you'd like me to analyze, or share a URL to analyze a webpage. You can also ask me about lenses, modes, or how to interpret results."


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def observer_chat(request: ChatRequest):
    """
    Conversational chat endpoint for the Observer.

    Accepts user messages and returns intelligent responses about:
    - What lenses to use
    - How to interpret analysis results
    - What text/URLs to analyze

    Automatically:
    - Recommends lenses based on keywords (denial/insurance -> denial-messaging, etc.)
    - Detects URLs and triggers website analysis
    - Integrates with /v2/analyze for text analysis

    Session-aware features:
    - Compares current analysis to previous analyses in session
    - Detects patterns across multiple analyses
    - Tracks trajectory of narrative health over session
    - Provides recommendations based on cumulative patterns
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Get the last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found")

    last_message = user_messages[-1].content
    context = request.context or ChatContext()

    # Extract session context
    session_history = context.session_history or []
    session_goal = context.session_goal

    actions: List[ChatAction] = []
    analysis_results: Optional[Dict[str, Any]] = None
    lens_suggestion: Optional[str] = None
    url_analysis: Optional[Dict[str, Any]] = None
    text_analysis: Optional[Dict[str, Any]] = None
    site_analysis: Optional[Dict[str, Any]] = None

    # Session-aware response fields
    session_insights: Optional[Dict[str, Any]] = None
    comparisons: Optional[List[Dict[str, Any]]] = None
    patterns_detected: Optional[List[Dict[str, Any]]] = None
    trajectory_update: Optional[Dict[str, Any]] = None

    # Check for URLs - use deep site analysis
    urls = detect_urls(last_message)
    print(f"[DEBUG] observer_chat: detected URLs = {urls}")
    if urls:
        # Analyze the first URL found with comprehensive site analysis
        url = urls[0]
        print(f"[DEBUG] observer_chat: calling analyze_site_deep for {url}")

        try:
            site_analysis = await analyze_site_deep(url)
            print(f"[DEBUG] observer_chat: site_analysis result = {list(site_analysis.keys()) if site_analysis else None}")

            if "error" not in site_analysis:
                # Site analysis succeeded - this will be rendered as a visual report
                analysis_results = site_analysis
            else:
                # Fall back to simple URL analysis
                lens_id = context.lens_id or "general-analysis"
                url_analysis = await analyze_url(url, lens_id)
                analysis_results = url_analysis
                actions.append(ChatAction(
                    type="show_results",
                    payload={"url": url, "results": url_analysis},
                    label=f"View full analysis of {url}"
                ))
        except Exception as e:
            logger.error(f"URL analysis failed: {e}")

    # Check for lens recommendation based on keywords
    elif not context.lens_id or context.lens_id == "general-analysis":
        recommended = recommend_lens(last_message)
        if recommended:
            lens_suggestion = recommended
            actions.append(ChatAction(
                type="select_lens",
                payload={"lens_id": recommended},
                label=f"Use {LENS_DEFINITIONS[recommended]['name']} lens"
            ))

    # Check if the message looks like text to analyze (longer than 50 chars, not a question)
    msg_lower = last_message.lower()
    is_question = any(msg_lower.startswith(q) for q in ["what", "how", "why", "which", "when", "can", "do", "is", "are", "help"])
    is_short = len(last_message) < 50

    if not urls and not is_question and not is_short:
        # This looks like text to analyze
        try:
            text_analysis = await analyze_text_internal(last_message)
            analysis_results = text_analysis

            # Apply lens thresholds if a lens is selected
            if context.lens_id:
                lens = get_lens_info(context.lens_id)
                if text_analysis.get("vector"):
                    coords = text_analysis["vector"]
                    alerts = []
                    for dim, threshold in lens.get("alert_thresholds", {}).items():
                        coord_key = dim if dim in coords else ("perceived_justice" if dim == "fairness" else dim)
                        if coord_key in coords and coords[coord_key] < threshold:
                            alerts.append({
                                "dimension": dim,
                                "value": coords[coord_key],
                                "threshold": threshold
                            })
                    if alerts:
                        analysis_results["lens_alerts"] = alerts

            actions.append(ChatAction(
                type="analyze",
                payload={"text": last_message[:100], "results": text_analysis},
                label="View detailed analysis"
            ))
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")

    # ==========================================================================
    # SESSION SYNTHESIS - Generate context-aware insights
    # ==========================================================================

    current_analysis = analysis_results or text_analysis or site_analysis

    if session_history and current_analysis:
        try:
            # Detect patterns across session
            patterns_detected = analyze_session_patterns(session_history, current_analysis)

            # Generate comparisons to previous analyses
            comparisons = generate_comparisons(current_analysis, session_history)

            # Calculate trajectory
            trajectory_update = calculate_trajectory(current_analysis, session_history)

            # Generate session-aware recommendations
            recommendations = generate_session_recommendations(
                current_analysis,
                session_history,
                patterns_detected or [],
                session_goal
            )

            # Build session insights summary
            session_insights = {
                "analysis_number": len(session_history) + 1,
                "has_previous_analyses": True,
                "patterns_count": len(patterns_detected) if patterns_detected else 0,
                "comparisons_count": len(comparisons) if comparisons else 0,
                "recommendations": recommendations,
                "goal_tracking": session_goal
            }

            # Add trajectory summary to insights
            if trajectory_update:
                session_insights["trajectory_summary"] = {
                    "direction": trajectory_update.get("overall_direction"),
                    "message": trajectory_update.get("message")
                }

            # Highlight critical patterns
            critical_patterns = [p for p in (patterns_detected or []) if p.get("severity") == "critical"]
            if critical_patterns:
                session_insights["critical_alerts"] = [p["description"] for p in critical_patterns[:3]]

        except Exception as e:
            logger.warning(f"Session synthesis failed: {e}")
            # Continue without session insights if synthesis fails

    elif current_analysis:
        # First analysis in session - note it for the user
        session_insights = {
            "analysis_number": 1,
            "has_previous_analyses": False,
            "message": "This is your first analysis. Subsequent analyses will be compared to build patterns.",
            "goal_tracking": session_goal
        }

    # ==========================================================================
    # GENERATE RESPONSE
    # ==========================================================================

    # Generate base response - for site analysis, keep it brief since visuals show details
    if site_analysis and "error" not in site_analysis:
        health = site_analysis.get("health_score", 0)
        pages = site_analysis.get("pages_analyzed", 0)
        mode = site_analysis.get("overall_mode", "Unknown")
        issues = len(site_analysis.get("critical_issues", []))
        response_text = f"I analyzed **{site_analysis.get('site_url', 'the site')}** ({pages} pages).\n\n**Health Score:** {health}/100 | **Primary Mode:** {mode}\n\n{'Warning: ' + str(issues) + ' issues found - see details below.' if issues > 0 else 'No critical issues detected.'}"
    else:
        response_text = generate_response(
            user_message=last_message,
            context=context,
            analysis_result=text_analysis,
            url_analysis=url_analysis,
            lens_recommendation=lens_suggestion if not text_analysis and not url_analysis else None
        )

    # Synthesize session-aware response if we have history and analysis
    if session_history and current_analysis:
        response_text = synthesize_session_response(
            base_response=response_text,
            current_analysis=current_analysis,
            session_history=session_history,
            patterns=patterns_detected or [],
            comparisons=comparisons or [],
            trajectory=trajectory_update,
            session_goal=session_goal
        )

    return ChatResponse(
        response=response_text,
        actions=actions if actions else None,
        lens_suggestion=lens_suggestion,
        analysis_results=analysis_results,
        site_analysis=site_analysis if site_analysis and "error" not in site_analysis else None,
        # Session-aware fields
        session_insights=session_insights,
        comparisons=comparisons,
        patterns_detected=patterns_detected,
        trajectory_update=trajectory_update
    )


@router.get("/lenses")
async def list_lenses():
    """List all available lenses for the Observer chat."""
    return {
        "lenses": [
            {
                "id": lens_id,
                "name": lens["name"],
                "description": lens["description"],
                "industry": lens.get("industry"),
                "primary_dimensions": lens["primary_dimensions"]
            }
            for lens_id, lens in LENS_DEFINITIONS.items()
        ]
    }


@router.get("/lenses/{lens_id}")
async def get_lens(lens_id: str):
    """Get details about a specific lens."""
    if lens_id not in LENS_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Lens '{lens_id}' not found")

    return LENS_DEFINITIONS[lens_id]
