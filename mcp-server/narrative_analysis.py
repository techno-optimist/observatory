"""
Narrative Analysis Module for the Cultural Soliton Observatory

Provides high-level tools for analyzing narratives from external sources
(websites, social media feeds, RSS) and synthesizing comprehensive profiles.

Tools:
  - fetch_narrative_source: Fetch and segment content from URLs/feeds
  - build_narrative_profile: Synthesize comprehensive narrative profile
  - get_narrative_suggestions: Generate actionable suggestions from profile
"""

import asyncio
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

import httpx


# =============================================================================
# Data Models
# =============================================================================

class SourceType(Enum):
    """Types of narrative sources"""
    WEBSITE = "website"
    RSS = "rss"
    TWITTER = "twitter"
    REDDIT = "reddit"
    BLOG = "blog"
    NEWS = "news"
    DOCUMENT = "document"
    RAW_TEXT = "raw_text"


class SuggestionType(Enum):
    """Types of narrative suggestions"""
    UNDERSTANDING = "understanding"  # Research/analysis insights
    ENGAGEMENT = "engagement"        # How to interact/respond
    STRATEGY = "strategy"            # Content/positioning opportunities
    WARNING = "warning"              # Risks or concerns


class SuggestionPriority(Enum):
    """Priority levels for suggestions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Vector3:
    """3D coordinate in the cultural manifold"""
    agency: float
    perceived_justice: float
    belonging: float

    def to_dict(self) -> dict:
        return {
            "agency": self.agency,
            "perceived_justice": self.perceived_justice,
            "belonging": self.belonging
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vector3":
        return cls(
            agency=d.get("agency", 0),
            perceived_justice=d.get("perceived_justice", d.get("fairness", 0)),
            belonging=d.get("belonging", 0)
        )

    def distance_to(self, other: "Vector3") -> float:
        """Euclidean distance to another point"""
        return (
            (self.agency - other.agency) ** 2 +
            (self.perceived_justice - other.perceived_justice) ** 2 +
            (self.belonging - other.belonging) ** 2
        ) ** 0.5


@dataclass
class ContentUnit:
    """A single unit of content extracted from a source"""
    text: str
    source_url: Optional[str] = None
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Analysis results (filled after projection)
    coordinates: Optional[Vector3] = None
    mode: Optional[str] = None
    confidence: Optional[float] = None
    soft_labels: Optional[dict] = None


@dataclass
class AttractorPoint:
    """An attractor in the force field"""
    target: str  # e.g., "AUTONOMY", "COMMUNITY", "JUSTICE"
    strength: float
    description: str


@dataclass
class DetractorPoint:
    """A detractor in the force field"""
    source: str  # e.g., "OPPRESSION", "ISOLATION", "INJUSTICE"
    strength: float
    description: str


@dataclass
class Tension:
    """An internal contradiction in the narrative"""
    dimension: str  # Which axis shows tension
    description: str
    severity: float  # 0-1
    evidence: list[str] = field(default_factory=list)


@dataclass
class PhaseTransition:
    """A detected sudden shift in narrative"""
    timestamp: Optional[datetime]
    from_mode: str
    to_mode: str
    magnitude: float
    description: str


@dataclass
class TrajectoryData:
    """Temporal evolution data"""
    points: list[dict]  # [{timestamp, coordinates, mode}, ...]
    direction: Optional[Vector3]  # velocity vector
    velocity: float
    acceleration: float
    inflection_points: list[dict]


@dataclass
class NarrativeProfile:
    """Complete narrative characterization of a source"""

    # === IDENTITY ===
    source: str                      # URL, @handle, feed URL
    source_type: SourceType
    analyzed_at: datetime
    content_units: int               # how many texts analyzed

    # === MANIFOLD POSITION ===
    centroid: Vector3                # average (agency, justice, belonging)
    spread: Vector3                  # variance in each dimension
    mode_distribution: dict[str, float]  # % in each of 12 modes
    dominant_mode: str               # primary narrative stance
    mode_signature: list[str]        # ordered top 3 modes

    # === DYNAMICS ===
    trajectory: Optional[TrajectoryData] = None
    stability_score: float = 0.0     # 0-1, how consistent over time
    drift_direction: Optional[Vector3] = None  # where they're heading
    phase_transitions: list[PhaseTransition] = field(default_factory=list)

    # === FORCE FIELD ===
    attractors: list[AttractorPoint] = field(default_factory=list)
    detractors: list[DetractorPoint] = field(default_factory=list)
    internal_tensions: list[Tension] = field(default_factory=list)

    # === COMPARATIVE ===
    quadrant_label: str = ""         # e.g., "High Agency + Low Justice"
    distinctive_features: list[str] = field(default_factory=list)

    # === LEGIBILITY ===
    legibility_score: float = 0.0    # how clear/readable
    ai_content_signals: float = 0.0  # probability of AI generation
    compression_phase: str = "natural"  # natural/technical/compressed/opaque

    # === META ===
    confidence: float = 0.0
    sample_quotes: list[str] = field(default_factory=list)
    raw_projections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "analyzed_at": self.analyzed_at.isoformat(),
            "content_units": self.content_units,
            "centroid": self.centroid.to_dict(),
            "spread": self.spread.to_dict(),
            "mode_distribution": self.mode_distribution,
            "dominant_mode": self.dominant_mode,
            "mode_signature": self.mode_signature,
            "stability_score": self.stability_score,
            "drift_direction": self.drift_direction.to_dict() if self.drift_direction else None,
            "phase_transitions": [
                {
                    "timestamp": pt.timestamp.isoformat() if pt.timestamp else None,
                    "from_mode": pt.from_mode,
                    "to_mode": pt.to_mode,
                    "magnitude": pt.magnitude,
                    "description": pt.description
                }
                for pt in self.phase_transitions
            ],
            "attractors": [
                {"target": a.target, "strength": a.strength, "description": a.description}
                for a in self.attractors
            ],
            "detractors": [
                {"source": d.source, "strength": d.strength, "description": d.description}
                for d in self.detractors
            ],
            "internal_tensions": [
                {
                    "dimension": t.dimension,
                    "description": t.description,
                    "severity": t.severity,
                    "evidence": t.evidence
                }
                for t in self.internal_tensions
            ],
            "quadrant_label": self.quadrant_label,
            "distinctive_features": self.distinctive_features,
            "legibility_score": self.legibility_score,
            "ai_content_signals": self.ai_content_signals,
            "compression_phase": self.compression_phase,
            "confidence": self.confidence,
            "sample_quotes": self.sample_quotes,
        }


@dataclass
class Suggestion:
    """An actionable suggestion based on the narrative profile"""
    type: SuggestionType
    priority: SuggestionPriority
    title: str
    insight: str                     # the main finding
    recommendation: str              # what to do
    evidence: list[str] = field(default_factory=list)
    related_coordinates: Optional[Vector3] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "insight": self.insight,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "related_coordinates": self.related_coordinates.to_dict() if self.related_coordinates else None
        }


# =============================================================================
# Content Extraction
# =============================================================================

class ContentExtractor:
    """Extracts and segments content from various sources"""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def fetch_url(self, url: str) -> str:
        """Fetch raw HTML/text from URL"""
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Cultural Soliton Observatory Research Bot)"
            })
            response.raise_for_status()
            return response.text

    def detect_source_type(self, url: str) -> SourceType:
        """Auto-detect source type from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Social media
        if "twitter.com" in domain or "x.com" in domain:
            return SourceType.TWITTER
        if "reddit.com" in domain:
            return SourceType.REDDIT

        # RSS/Atom feeds
        if path.endswith((".rss", ".xml", ".atom")) or "/feed" in path or "/rss" in path:
            return SourceType.RSS

        # News sites (common patterns)
        news_domains = ["news", "cnn", "bbc", "nytimes", "reuters", "washingtonpost"]
        if any(nd in domain for nd in news_domains):
            return SourceType.NEWS

        # Blog platforms
        blog_domains = ["medium.com", "substack.com", "wordpress.com", "blogger.com"]
        if any(bd in domain for bd in blog_domains):
            return SourceType.BLOG

        return SourceType.WEBSITE

    def extract_text_from_html(self, html: str) -> list[str]:
        """Extract meaningful text segments from HTML"""
        # Simple extraction - in production, use BeautifulSoup or similar

        # Remove script, style, nav, footer, header tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Extract paragraphs and headings
        segments = []

        # Get paragraph content
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
        for p in paragraphs:
            text = self._clean_text(p)
            if len(text) > 50:  # Skip very short paragraphs
                segments.append(text)

        # Get heading content
        for level in range(1, 7):
            headings = re.findall(f'<h{level}[^>]*>(.*?)</h{level}>', html, re.DOTALL | re.IGNORECASE)
            for h in headings:
                text = self._clean_text(h)
                if len(text) > 10:
                    segments.append(text)

        # Get article/main content
        articles = re.findall(r'<article[^>]*>(.*?)</article>', html, re.DOTALL | re.IGNORECASE)
        for article in articles:
            text = self._clean_text(article)
            # Split long articles into smaller chunks
            if len(text) > 500:
                chunks = self._chunk_text(text, max_chars=500)
                segments.extend(chunks)
            elif len(text) > 50:
                segments.append(text)

        # Deduplicate while preserving order
        seen = set()
        unique_segments = []
        for seg in segments:
            normalized = seg.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_segments.append(seg)

        return unique_segments

    def _clean_text(self, html: str) -> str:
        """Remove HTML tags and clean text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Decode common entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_text(self, text: str, max_chars: int = 500) -> list[str]:
        """Split long text into sentence-aware chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def parse_rss(self, xml_content: str) -> list[ContentUnit]:
        """Parse RSS/Atom feed into content units"""
        units = []

        # Simple RSS parsing - extract items
        items = re.findall(r'<item>(.*?)</item>', xml_content, re.DOTALL | re.IGNORECASE)
        if not items:
            # Try Atom format
            items = re.findall(r'<entry>(.*?)</entry>', xml_content, re.DOTALL | re.IGNORECASE)

        for item in items:
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', item, re.DOTALL | re.IGNORECASE)
            title = self._clean_text(title_match.group(1)) if title_match else ""

            # Extract description/content
            desc_match = re.search(r'<description[^>]*>(.*?)</description>', item, re.DOTALL | re.IGNORECASE)
            if not desc_match:
                desc_match = re.search(r'<content[^>]*>(.*?)</content>', item, re.DOTALL | re.IGNORECASE)
            description = self._clean_text(desc_match.group(1)) if desc_match else ""

            # Extract date
            date_match = re.search(r'<pubDate[^>]*>(.*?)</pubDate>', item, re.DOTALL | re.IGNORECASE)
            if not date_match:
                date_match = re.search(r'<published[^>]*>(.*?)</published>', item, re.DOTALL | re.IGNORECASE)
            timestamp = None
            if date_match:
                try:
                    from email.utils import parsedate_to_datetime
                    timestamp = parsedate_to_datetime(date_match.group(1))
                except Exception:
                    pass

            # Combine title and description
            text = f"{title}. {description}" if title and description else (title or description)

            if text and len(text) > 30:
                units.append(ContentUnit(
                    text=text,
                    timestamp=timestamp,
                    metadata={"title": title}
                ))

        return units

    async def fetch_and_segment(
        self,
        url: str,
        source_type: Optional[SourceType] = None,
        max_items: int = 100,
        include_metadata: bool = True
    ) -> list[ContentUnit]:
        """
        Main entry point: fetch URL and return segmented content units
        """
        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self.detect_source_type(url)

        # Fetch content
        raw_content = await self.fetch_url(url)

        # Process based on type
        if source_type == SourceType.RSS:
            units = self.parse_rss(raw_content)
        else:
            # HTML-based sources
            segments = self.extract_text_from_html(raw_content)
            units = [
                ContentUnit(
                    text=seg,
                    source_url=url,
                    metadata={"source_type": source_type.value}
                )
                for seg in segments
            ]

        # Limit results
        units = units[:max_items]

        # Add source URL if not set
        for unit in units:
            if not unit.source_url:
                unit.source_url = url

        return units


# =============================================================================
# Profile Builder
# =============================================================================

class ProfileBuilder:
    """Builds comprehensive narrative profiles from content units"""

    def __init__(self, observatory_client):
        """
        Args:
            observatory_client: An ObservatoryClient instance for projections
        """
        self.client = observatory_client

    async def project_units(self, units: list[ContentUnit]) -> list[ContentUnit]:
        """Project all content units onto the manifold"""
        texts = [u.text for u in units]

        # Batch project
        results = await self.client.project_batch(texts)

        # Attach results to units
        for unit, result in zip(units, results):
            unit.coordinates = Vector3(
                agency=result.agency,
                perceived_justice=result.perceived_justice,
                belonging=result.belonging
            )
            unit.mode = result.mode
            unit.confidence = result.confidence
            unit.soft_labels = result.soft_labels

        return units

    def compute_centroid(self, units: list[ContentUnit]) -> Vector3:
        """Compute centroid of all projected units"""
        if not units:
            return Vector3(0, 0, 0)

        valid_units = [u for u in units if u.coordinates is not None]
        if not valid_units:
            return Vector3(0, 0, 0)

        return Vector3(
            agency=statistics.mean(u.coordinates.agency for u in valid_units),
            perceived_justice=statistics.mean(u.coordinates.perceived_justice for u in valid_units),
            belonging=statistics.mean(u.coordinates.belonging for u in valid_units)
        )

    def compute_spread(self, units: list[ContentUnit]) -> Vector3:
        """Compute variance/spread in each dimension"""
        if len(units) < 2:
            return Vector3(0, 0, 0)

        valid_units = [u for u in units if u.coordinates is not None]
        if len(valid_units) < 2:
            return Vector3(0, 0, 0)

        return Vector3(
            agency=statistics.stdev(u.coordinates.agency for u in valid_units),
            perceived_justice=statistics.stdev(u.coordinates.perceived_justice for u in valid_units),
            belonging=statistics.stdev(u.coordinates.belonging for u in valid_units)
        )

    def compute_mode_distribution(self, units: list[ContentUnit]) -> dict[str, float]:
        """Compute distribution across narrative modes"""
        mode_counts: dict[str, int] = {}
        total = 0

        for unit in units:
            if unit.mode:
                mode_counts[unit.mode] = mode_counts.get(unit.mode, 0) + 1
                total += 1

        if total == 0:
            return {}

        return {mode: count / total for mode, count in mode_counts.items()}

    def get_quadrant_label(self, centroid: Vector3) -> str:
        """Generate human-readable quadrant label"""
        agency_label = "High Agency" if centroid.agency > 0 else "Low Agency"
        justice_label = "High Justice" if centroid.perceived_justice > 0 else "Low Justice"
        belonging_label = "High Belonging" if centroid.belonging > 0 else "Low Belonging"

        # Primary characterization based on strongest dimension
        dims = [
            (abs(centroid.agency), agency_label),
            (abs(centroid.perceived_justice), justice_label),
            (abs(centroid.belonging), belonging_label)
        ]
        dims.sort(reverse=True)

        return f"{dims[0][1]} + {dims[1][1]}"

    def compute_stability(self, units: list[ContentUnit]) -> float:
        """Compute stability score (consistency of narrative)"""
        if len(units) < 2:
            return 1.0

        # Stability based on mode concentration
        dist = self.compute_mode_distribution(units)
        if not dist:
            return 0.5

        # High stability = one dominant mode
        max_prob = max(dist.values())
        return max_prob

    def detect_tensions(self, units: list[ContentUnit], centroid: Vector3) -> list[Tension]:
        """Detect internal contradictions in the narrative"""
        tensions = []

        valid_units = [u for u in units if u.coordinates is not None]
        if len(valid_units) < 3:
            return tensions

        # Check for high variance in each dimension
        spread = self.compute_spread(units)

        # Agency tension: expressing both empowerment and helplessness
        if spread.agency > 0.8:
            high_agency = [u for u in valid_units if u.coordinates.agency > 0.5]
            low_agency = [u for u in valid_units if u.coordinates.agency < -0.5]
            if high_agency and low_agency:
                tensions.append(Tension(
                    dimension="agency",
                    description="Mixed signals about empowerment vs. powerlessness",
                    severity=min(1.0, spread.agency / 1.5),
                    evidence=[
                        high_agency[0].text[:100] + "...",
                        low_agency[0].text[:100] + "..."
                    ]
                ))

        # Justice tension: both trusting and distrusting system
        if spread.perceived_justice > 0.8:
            high_justice = [u for u in valid_units if u.coordinates.perceived_justice > 0.5]
            low_justice = [u for u in valid_units if u.coordinates.perceived_justice < -0.5]
            if high_justice and low_justice:
                tensions.append(Tension(
                    dimension="perceived_justice",
                    description="Contradictory views on system fairness/legitimacy",
                    severity=min(1.0, spread.perceived_justice / 1.5),
                    evidence=[
                        high_justice[0].text[:100] + "...",
                        low_justice[0].text[:100] + "..."
                    ]
                ))

        # Belonging tension: ingroup and outgroup simultaneously
        if spread.belonging > 0.8:
            high_belonging = [u for u in valid_units if u.coordinates.belonging > 0.5]
            low_belonging = [u for u in valid_units if u.coordinates.belonging < -0.5]
            if high_belonging and low_belonging:
                tensions.append(Tension(
                    dimension="belonging",
                    description="Oscillating between connection and alienation",
                    severity=min(1.0, spread.belonging / 1.5),
                    evidence=[
                        high_belonging[0].text[:100] + "...",
                        low_belonging[0].text[:100] + "..."
                    ]
                ))

        return tensions

    def select_sample_quotes(self, units: list[ContentUnit], n: int = 5) -> list[str]:
        """Select representative sample quotes"""
        if not units:
            return []

        valid_units = [u for u in units if u.coordinates is not None and u.confidence]
        if not valid_units:
            return [u.text[:200] for u in units[:n]]

        # Sort by confidence and select diverse samples
        sorted_units = sorted(valid_units, key=lambda u: u.confidence or 0, reverse=True)

        samples = []
        modes_seen = set()

        for unit in sorted_units:
            if len(samples) >= n:
                break
            # Prioritize mode diversity
            if unit.mode not in modes_seen or len(samples) < n // 2:
                samples.append(unit.text[:200] + ("..." if len(unit.text) > 200 else ""))
                modes_seen.add(unit.mode)

        return samples

    async def analyze_forces(self, units: list[ContentUnit]) -> tuple[list[AttractorPoint], list[DetractorPoint]]:
        """Analyze force field (attractors/detractors)"""
        attractors = []
        detractors = []

        # Aggregate forces from projections
        texts = [u.text for u in units]

        try:
            force_result = await self.client.batch_analyze_forces(texts)

            if "aggregate" in force_result:
                agg = force_result["aggregate"]

                # Extract top attractors
                if "attractor_distribution" in agg:
                    for target, strength in sorted(
                        agg["attractor_distribution"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]:
                        attractors.append(AttractorPoint(
                            target=target,
                            strength=strength,
                            description=f"Drawn toward {target.lower().replace('_', ' ')}"
                        ))

                # Extract top detractors
                if "detractor_distribution" in agg:
                    for source, strength in sorted(
                        agg["detractor_distribution"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]:
                        detractors.append(DetractorPoint(
                            source=source,
                            strength=strength,
                            description=f"Fleeing from {source.lower().replace('_', ' ')}"
                        ))
        except Exception:
            # Force analysis not available, infer from coordinates
            centroid = self.compute_centroid(units)

            # Infer attractors from positive coordinates
            if centroid.agency > 0.3:
                attractors.append(AttractorPoint(
                    target="AUTONOMY",
                    strength=centroid.agency,
                    description="Values self-determination and empowerment"
                ))
            if centroid.perceived_justice > 0.3:
                attractors.append(AttractorPoint(
                    target="JUSTICE",
                    strength=centroid.perceived_justice,
                    description="Seeks fairness and legitimate systems"
                ))
            if centroid.belonging > 0.3:
                attractors.append(AttractorPoint(
                    target="COMMUNITY",
                    strength=centroid.belonging,
                    description="Values connection and group identity"
                ))

            # Infer detractors from negative coordinates
            if centroid.agency < -0.3:
                detractors.append(DetractorPoint(
                    source="OPPRESSION",
                    strength=abs(centroid.agency),
                    description="Perceives external control and constraint"
                ))
            if centroid.perceived_justice < -0.3:
                detractors.append(DetractorPoint(
                    source="INJUSTICE",
                    strength=abs(centroid.perceived_justice),
                    description="Perceives systemic unfairness or corruption"
                ))
            if centroid.belonging < -0.3:
                detractors.append(DetractorPoint(
                    source="ISOLATION",
                    strength=abs(centroid.belonging),
                    description="Experiences alienation or exclusion"
                ))

        return attractors, detractors

    async def build_profile(
        self,
        units: list[ContentUnit],
        source: str,
        source_type: SourceType,
        include_forces: bool = True
    ) -> NarrativeProfile:
        """Build comprehensive narrative profile from content units"""

        # Project all units onto manifold
        projected_units = await self.project_units(units)

        # Compute statistics
        centroid = self.compute_centroid(projected_units)
        spread = self.compute_spread(projected_units)
        mode_dist = self.compute_mode_distribution(projected_units)

        # Dominant mode and signature
        sorted_modes = sorted(mode_dist.items(), key=lambda x: x[1], reverse=True)
        dominant_mode = sorted_modes[0][0] if sorted_modes else "unknown"
        mode_signature = [m[0] for m in sorted_modes[:3]]

        # Analyze forces
        attractors, detractors = [], []
        if include_forces:
            attractors, detractors = await self.analyze_forces(projected_units)

        # Detect tensions
        tensions = self.detect_tensions(projected_units, centroid)

        # Compute stability
        stability = self.compute_stability(projected_units)

        # Compute average confidence as overall confidence
        confidences = [u.confidence for u in projected_units if u.confidence]
        avg_confidence = statistics.mean(confidences) if confidences else 0.5

        # Sample quotes
        samples = self.select_sample_quotes(projected_units)

        # Raw projections for detailed analysis
        raw_projections = [
            {
                "text": u.text[:100],
                "coordinates": u.coordinates.to_dict() if u.coordinates else None,
                "mode": u.mode,
                "confidence": u.confidence
            }
            for u in projected_units
        ]

        return NarrativeProfile(
            source=source,
            source_type=source_type,
            analyzed_at=datetime.now(timezone.utc),
            content_units=len(units),
            centroid=centroid,
            spread=spread,
            mode_distribution=mode_dist,
            dominant_mode=dominant_mode,
            mode_signature=mode_signature,
            stability_score=stability,
            attractors=attractors,
            detractors=detractors,
            internal_tensions=tensions,
            quadrant_label=self.get_quadrant_label(centroid),
            confidence=avg_confidence,
            sample_quotes=samples,
            raw_projections=raw_projections
        )


# =============================================================================
# Suggestion Engine
# =============================================================================

class SuggestionEngine:
    """Generates actionable suggestions from narrative profiles"""

    def __init__(self):
        self.mode_descriptions = {
            "GROWTH_MINDSET": "optimistic self-development orientation",
            "CIVIC_IDEALISM": "faith in institutions and collective action",
            "FAITHFUL_ZEAL": "strong ideological or spiritual commitment",
            "CYNICAL_BURNOUT": "awareness of system flaws with exploitative response",
            "INSTITUTIONAL_DECAY": "recognition of organizational decline",
            "SCHISMATIC_DOUBT": "questioning of previously held beliefs",
            "QUIET_QUITTING": "workplace disengagement without exit",
            "GRID_EXIT": "withdrawal from mainstream systems",
            "APOSTASY": "abandonment of faith or ideology",
            "VICTIM": "perception of powerlessness",
            "TRANSITIONAL": "liminal state between modes",
            "NEUTRAL": "no strong narrative signals"
        }

    def generate_understanding_suggestions(self, profile: NarrativeProfile) -> list[Suggestion]:
        """Generate research/analysis insights"""
        suggestions = []

        # Mode characterization
        mode_desc = self.mode_descriptions.get(
            profile.dominant_mode.upper().replace(" ", "_"),
            profile.dominant_mode
        )
        suggestions.append(Suggestion(
            type=SuggestionType.UNDERSTANDING,
            priority=SuggestionPriority.HIGH,
            title="Dominant Narrative Mode",
            insight=f"This source exhibits a {profile.dominant_mode} pattern: {mode_desc}.",
            recommendation="Consider the underlying drivers of this narrative stance.",
            evidence=profile.sample_quotes[:2],
            related_coordinates=profile.centroid
        ))

        # Stability insight
        if profile.stability_score < 0.4:
            suggestions.append(Suggestion(
                type=SuggestionType.UNDERSTANDING,
                priority=SuggestionPriority.MEDIUM,
                title="Narrative Instability Detected",
                insight=f"Low stability score ({profile.stability_score:.2f}) indicates inconsistent narrative positioning.",
                recommendation="This source may be in transition or addressing multiple audiences with different framings.",
                evidence=[f"Mode distribution: {', '.join(f'{m}: {p:.0%}' for m, p in list(profile.mode_distribution.items())[:3])}"]
            ))

        # Tensions
        for tension in profile.internal_tensions:
            suggestions.append(Suggestion(
                type=SuggestionType.UNDERSTANDING,
                priority=SuggestionPriority.MEDIUM,
                title=f"Internal Tension: {tension.dimension.title()}",
                insight=tension.description,
                recommendation="This contradiction may indicate cognitive dissonance, multiple voices, or evolving positions.",
                evidence=tension.evidence
            ))

        # Quadrant analysis
        suggestions.append(Suggestion(
            type=SuggestionType.UNDERSTANDING,
            priority=SuggestionPriority.LOW,
            title="Manifold Positioning",
            insight=f"Located in the {profile.quadrant_label} quadrant of the cultural manifold.",
            recommendation=f"Agency={profile.centroid.agency:.2f}, Justice={profile.centroid.perceived_justice:.2f}, Belonging={profile.centroid.belonging:.2f}",
            related_coordinates=profile.centroid
        ))

        return suggestions

    def generate_engagement_suggestions(self, profile: NarrativeProfile) -> list[Suggestion]:
        """Generate suggestions for how to engage/interact"""
        suggestions = []

        # What resonates
        if profile.attractors:
            top_attractor = profile.attractors[0]
            suggestions.append(Suggestion(
                type=SuggestionType.ENGAGEMENT,
                priority=SuggestionPriority.HIGH,
                title="Resonance Strategy",
                insight=f"This audience is drawn toward {top_attractor.target.lower().replace('_', ' ')}.",
                recommendation=f"Lead with messaging that emphasizes {top_attractor.target.lower().replace('_', ' ')}. Show how your position supports their core value.",
                evidence=[top_attractor.description]
            ))

        # What to avoid
        if profile.detractors:
            top_detractor = profile.detractors[0]
            suggestions.append(Suggestion(
                type=SuggestionType.ENGAGEMENT,
                priority=SuggestionPriority.HIGH,
                title="Avoidance Strategy",
                insight=f"This audience is fleeing from {top_detractor.source.lower().replace('_', ' ')}.",
                recommendation=f"Avoid framing that could be perceived as {top_detractor.source.lower().replace('_', ' ')}. Acknowledge their concerns before offering alternatives.",
                evidence=[top_detractor.description]
            ))

        # Framing recommendations based on coordinates
        if profile.centroid.agency > 0.3:
            suggestions.append(Suggestion(
                type=SuggestionType.ENGAGEMENT,
                priority=SuggestionPriority.MEDIUM,
                title="Agency-Affirming Approach",
                insight="This audience values self-determination and empowerment.",
                recommendation="Use language like 'you can choose', 'take control', 'your decision'. Avoid paternalistic framing."
            ))
        elif profile.centroid.agency < -0.3:
            suggestions.append(Suggestion(
                type=SuggestionType.ENGAGEMENT,
                priority=SuggestionPriority.MEDIUM,
                title="Empowerment Opportunity",
                insight="This audience perceives limited agency.",
                recommendation="Offer concrete pathways to regain control. Validate their constraints while showing possibilities."
            ))

        if profile.centroid.perceived_justice < -0.3:
            suggestions.append(Suggestion(
                type=SuggestionType.ENGAGEMENT,
                priority=SuggestionPriority.MEDIUM,
                title="Justice Sensitivity",
                insight="This audience is skeptical of system fairness.",
                recommendation="Acknowledge systemic issues before proposing solutions. Don't defend institutions they perceive as corrupt."
            ))

        return suggestions

    def generate_strategy_suggestions(self, profile: NarrativeProfile) -> list[Suggestion]:
        """Generate content/positioning strategy suggestions"""
        suggestions = []

        # Gap analysis
        underserved_modes = []
        shadow_modes = ["CYNICAL_BURNOUT", "INSTITUTIONAL_DECAY", "SCHISMATIC_DOUBT"]
        exit_modes = ["QUIET_QUITTING", "GRID_EXIT", "APOSTASY"]
        positive_modes = ["GROWTH_MINDSET", "CIVIC_IDEALISM", "FAITHFUL_ZEAL"]

        shadow_pct = sum(profile.mode_distribution.get(m, 0) for m in shadow_modes)
        exit_pct = sum(profile.mode_distribution.get(m, 0) for m in exit_modes)
        positive_pct = sum(profile.mode_distribution.get(m, 0) for m in positive_modes)

        if shadow_pct > 0.4:
            suggestions.append(Suggestion(
                type=SuggestionType.STRATEGY,
                priority=SuggestionPriority.HIGH,
                title="Shadow Mode Dominance",
                insight=f"{shadow_pct:.0%} of content is in shadow modes (cynical, decaying, doubting).",
                recommendation="Opportunity to differentiate with constructive alternatives. Position as acknowledging problems while offering genuine solutions.",
                evidence=[f"Shadow modes: {shadow_pct:.0%}", f"Positive modes: {positive_pct:.0%}"]
            ))

        if exit_pct > 0.3:
            suggestions.append(Suggestion(
                type=SuggestionType.STRATEGY,
                priority=SuggestionPriority.HIGH,
                title="Exit Mode Warning",
                insight=f"{exit_pct:.0%} of content signals disengagement or withdrawal.",
                recommendation="This audience may be difficult to reach through traditional channels. Consider alternative engagement methods.",
                evidence=[f"Exit modes: {exit_pct:.0%}"]
            ))

        if positive_pct < 0.2:
            # Opportunity for positive content
            optimal_coords = Vector3(1.0, 0.8, 1.0)  # Growth mindset territory
            suggestions.append(Suggestion(
                type=SuggestionType.STRATEGY,
                priority=SuggestionPriority.MEDIUM,
                title="Positive Narrative Gap",
                insight="Positive/constructive narrative modes are underrepresented.",
                recommendation="Content at high-agency, high-justice, high-belonging coordinates may find an underserved audience.",
                related_coordinates=optimal_coords
            ))

        # Spread-based opportunities
        if profile.spread.agency > 0.7:
            suggestions.append(Suggestion(
                type=SuggestionType.STRATEGY,
                priority=SuggestionPriority.LOW,
                title="Agency Variance Opportunity",
                insight="Wide variance in agency signals suggests fragmented audience.",
                recommendation="Consider segmented messaging for empowered vs. disempowered subgroups."
            ))

        return suggestions

    def generate_warning_suggestions(self, profile: NarrativeProfile) -> list[Suggestion]:
        """Generate warning/risk suggestions"""
        suggestions = []

        # AI content warning
        if profile.ai_content_signals > 0.5:
            suggestions.append(Suggestion(
                type=SuggestionType.WARNING,
                priority=SuggestionPriority.HIGH,
                title="AI-Generated Content Signals",
                insight=f"High probability ({profile.ai_content_signals:.0%}) of AI-generated content.",
                recommendation="Verify authenticity. AI-generated content may not reflect genuine audience sentiment."
            ))

        # Low legibility warning
        if profile.legibility_score < 0.4:
            suggestions.append(Suggestion(
                type=SuggestionType.WARNING,
                priority=SuggestionPriority.MEDIUM,
                title="Low Legibility",
                insight=f"Content has low legibility score ({profile.legibility_score:.2f}).",
                recommendation="May indicate jargon-heavy, compressed, or deliberately obscured communication."
            ))

        # Extreme positioning
        extreme_threshold = 1.5
        if abs(profile.centroid.agency) > extreme_threshold:
            suggestions.append(Suggestion(
                type=SuggestionType.WARNING,
                priority=SuggestionPriority.MEDIUM,
                title="Extreme Agency Positioning",
                insight=f"Unusually {'high' if profile.centroid.agency > 0 else 'low'} agency signals.",
                recommendation="May indicate polarized or radicalized content. Verify context."
            ))

        if abs(profile.centroid.perceived_justice) > extreme_threshold:
            suggestions.append(Suggestion(
                type=SuggestionType.WARNING,
                priority=SuggestionPriority.MEDIUM,
                title="Extreme Justice Positioning",
                insight=f"Unusually {'high trust in' if profile.centroid.perceived_justice > 0 else 'strong distrust of'} systems.",
                recommendation="May indicate institutional propaganda or anti-establishment radicalization."
            ))

        return suggestions

    def generate_suggestions(
        self,
        profile: NarrativeProfile,
        intent: str = "understand"
    ) -> list[Suggestion]:
        """
        Generate all relevant suggestions for a profile.

        Args:
            profile: The narrative profile to analyze
            intent: Primary goal - "understand", "engage", "counter", "bridge"

        Returns:
            List of suggestions, prioritized by relevance to intent
        """
        all_suggestions = []

        # Always include understanding suggestions
        all_suggestions.extend(self.generate_understanding_suggestions(profile))

        # Intent-specific suggestions
        if intent in ("engage", "bridge"):
            all_suggestions.extend(self.generate_engagement_suggestions(profile))

        if intent in ("counter", "bridge", "strategy"):
            all_suggestions.extend(self.generate_strategy_suggestions(profile))

        # Always include warnings
        all_suggestions.extend(self.generate_warning_suggestions(profile))

        # Sort by priority
        priority_order = {
            SuggestionPriority.HIGH: 0,
            SuggestionPriority.MEDIUM: 1,
            SuggestionPriority.LOW: 2
        }
        all_suggestions.sort(key=lambda s: priority_order[s.priority])

        return all_suggestions


# =============================================================================
# Main Functions for MCP Tools
# =============================================================================

async def fetch_narrative_source(
    url: str,
    source_type: Optional[str] = None,
    max_items: int = 100,
    include_metadata: bool = True
) -> dict:
    """
    Fetch and segment content from a URL into analysis units.

    This is the MCP tool entry point for fetching content.

    Args:
        url: The URL to fetch (website, RSS feed, etc.)
        source_type: Optional type hint ("website", "rss", "twitter", etc.)
        max_items: Maximum number of content units to return
        include_metadata: Whether to include source metadata

    Returns:
        Dictionary with content_units list and metadata
    """
    extractor = ContentExtractor()

    # Parse source type if provided
    stype = None
    if source_type:
        try:
            stype = SourceType(source_type.lower())
        except ValueError:
            stype = None

    units = await extractor.fetch_and_segment(
        url=url,
        source_type=stype,
        max_items=max_items,
        include_metadata=include_metadata
    )

    # Detect source type if not specified
    if stype is None:
        stype = extractor.detect_source_type(url)

    return {
        "source": url,
        "source_type": stype.value,
        "content_units": [
            {
                "text": u.text,
                "timestamp": u.timestamp.isoformat() if u.timestamp else None,
                "author": u.author,
                "metadata": u.metadata
            }
            for u in units
        ],
        "count": len(units)
    }


async def build_narrative_profile(
    content_units: list[dict],
    source: str,
    source_type: str = "website",
    observatory_client=None,
    include_force_analysis: bool = True
) -> dict:
    """
    Build a comprehensive narrative profile from content units.

    This is the MCP tool entry point for profile building.

    Args:
        content_units: List of dicts with "text" field (and optional metadata)
        source: Source identifier (URL, handle, etc.)
        source_type: Type of source
        observatory_client: Optional ObservatoryClient instance
        include_force_analysis: Whether to include force field analysis

    Returns:
        Complete NarrativeProfile as dictionary
    """
    # Convert dict content units to ContentUnit objects
    units = [
        ContentUnit(
            text=cu.get("text", ""),
            timestamp=datetime.fromisoformat(cu["timestamp"]) if cu.get("timestamp") else None,
            author=cu.get("author"),
            metadata=cu.get("metadata", {})
        )
        for cu in content_units
        if cu.get("text")
    ]

    if not units:
        raise ValueError("No valid content units provided")

    # Import observatory client if not provided
    if observatory_client is None:
        from observatory_client import ObservatoryClient
        observatory_client = ObservatoryClient()

    # Build profile
    builder = ProfileBuilder(observatory_client)

    async with observatory_client:
        profile = await builder.build_profile(
            units=units,
            source=source,
            source_type=SourceType(source_type),
            include_forces=include_force_analysis
        )

    return profile.to_dict()


def get_narrative_suggestions(
    profile_dict: dict,
    intent: str = "understand"
) -> list[dict]:
    """
    Generate actionable suggestions from a narrative profile.

    This is the MCP tool entry point for suggestion generation.

    Args:
        profile_dict: NarrativeProfile as dictionary
        intent: Primary goal - "understand", "engage", "counter", "bridge", "strategy"

    Returns:
        List of Suggestion dictionaries
    """
    # Reconstruct profile object
    profile = NarrativeProfile(
        source=profile_dict.get("source", "unknown"),
        source_type=SourceType(profile_dict.get("source_type", "website")),
        analyzed_at=datetime.fromisoformat(profile_dict["analyzed_at"]) if profile_dict.get("analyzed_at") else datetime.utcnow(),
        content_units=profile_dict.get("content_units", 0),
        centroid=Vector3.from_dict(profile_dict.get("centroid", {})),
        spread=Vector3.from_dict(profile_dict.get("spread", {})),
        mode_distribution=profile_dict.get("mode_distribution", {}),
        dominant_mode=profile_dict.get("dominant_mode", "unknown"),
        mode_signature=profile_dict.get("mode_signature", []),
        stability_score=profile_dict.get("stability_score", 0.5),
        quadrant_label=profile_dict.get("quadrant_label", ""),
        legibility_score=profile_dict.get("legibility_score", 0.5),
        ai_content_signals=profile_dict.get("ai_content_signals", 0),
        sample_quotes=profile_dict.get("sample_quotes", [])
    )

    # Reconstruct attractors/detractors
    for a in profile_dict.get("attractors", []):
        profile.attractors.append(AttractorPoint(
            target=a.get("target", ""),
            strength=a.get("strength", 0),
            description=a.get("description", "")
        ))

    for d in profile_dict.get("detractors", []):
        profile.detractors.append(DetractorPoint(
            source=d.get("source", ""),
            strength=d.get("strength", 0),
            description=d.get("description", "")
        ))

    for t in profile_dict.get("internal_tensions", []):
        profile.internal_tensions.append(Tension(
            dimension=t.get("dimension", ""),
            description=t.get("description", ""),
            severity=t.get("severity", 0),
            evidence=t.get("evidence", [])
        ))

    # Generate suggestions
    engine = SuggestionEngine()
    suggestions = engine.generate_suggestions(profile, intent=intent)

    return [s.to_dict() for s in suggestions]


# =============================================================================
# Formatting Helpers for MCP Output
# =============================================================================

def format_profile_result(profile_dict: dict) -> str:
    """Format a NarrativeProfile for human-readable MCP output"""

    centroid = profile_dict.get("centroid", {})
    spread = profile_dict.get("spread", {})
    mode_dist = profile_dict.get("mode_distribution", {})

    output = f"""
## Narrative Profile

**Source**: {profile_dict.get('source', 'Unknown')}
**Type**: {profile_dict.get('source_type', 'Unknown')}
**Analyzed**: {profile_dict.get('analyzed_at', 'Unknown')}
**Content Units**: {profile_dict.get('content_units', 0)}

### Manifold Position

**Centroid** (Cultural Coordinates):
- Agency: {centroid.get('agency', 0):.3f} (-2 to +2)
- Perceived Justice: {centroid.get('perceived_justice', 0):.3f} (-2 to +2)
- Belonging: {centroid.get('belonging', 0):.3f} (-2 to +2)

**Spread** (Variance):
- Agency σ: {spread.get('agency', 0):.3f}
- Perceived Justice σ: {spread.get('perceived_justice', 0):.3f}
- Belonging σ: {spread.get('belonging', 0):.3f}

**Quadrant**: {profile_dict.get('quadrant_label', 'Unknown')}

### Narrative Classification

**Dominant Mode**: {profile_dict.get('dominant_mode', 'Unknown')}
**Mode Signature**: {', '.join(profile_dict.get('mode_signature', []))}
**Stability Score**: {profile_dict.get('stability_score', 0):.2f}

**Mode Distribution**:
"""

    for mode, pct in sorted(mode_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        bar = "█" * int(pct * 20)
        output += f"- {mode}: {pct:.1%} {bar}\n"

    # Attractors
    attractors = profile_dict.get("attractors", [])
    if attractors:
        output += "\n### Force Field - Attractors\n"
        for a in attractors:
            output += f"- **{a.get('target', '')}** (strength: {a.get('strength', 0):.2f}): {a.get('description', '')}\n"

    # Detractors
    detractors = profile_dict.get("detractors", [])
    if detractors:
        output += "\n### Force Field - Detractors\n"
        for d in detractors:
            output += f"- **{d.get('source', '')}** (strength: {d.get('strength', 0):.2f}): {d.get('description', '')}\n"

    # Tensions
    tensions = profile_dict.get("internal_tensions", [])
    if tensions:
        output += "\n### Internal Tensions\n"
        for t in tensions:
            output += f"- **{t.get('dimension', '').title()}** (severity: {t.get('severity', 0):.2f}): {t.get('description', '')}\n"

    # Sample quotes
    samples = profile_dict.get("sample_quotes", [])
    if samples:
        output += "\n### Sample Quotes\n"
        for i, quote in enumerate(samples[:3], 1):
            output += f"{i}. \"{quote}\"\n"

    return output


def format_suggestions_result(suggestions: list[dict]) -> str:
    """Format suggestions for human-readable MCP output"""

    output = "## Narrative Suggestions\n\n"

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for s in suggestions:
        stype = s.get("type", "understanding")
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(s)

    type_labels = {
        "understanding": "🔍 Understanding & Analysis",
        "engagement": "🤝 Engagement Strategies",
        "strategy": "📊 Strategic Opportunities",
        "warning": "⚠️ Warnings & Risks"
    }

    priority_icons = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢"
    }

    for stype, type_suggestions in by_type.items():
        output += f"### {type_labels.get(stype, stype.title())}\n\n"

        for s in type_suggestions:
            icon = priority_icons.get(s.get("priority", "medium"), "")
            output += f"**{icon} {s.get('title', 'Suggestion')}**\n"
            output += f"- *Insight*: {s.get('insight', '')}\n"
            output += f"- *Recommendation*: {s.get('recommendation', '')}\n"

            if s.get("evidence"):
                output += f"- *Evidence*:\n"
                for e in s.get("evidence", [])[:2]:
                    output += f"  - {e}\n"

            if s.get("related_coordinates"):
                coords = s["related_coordinates"]
                output += f"- *Related coordinates*: ({coords.get('agency', 0):.2f}, {coords.get('perceived_justice', 0):.2f}, {coords.get('belonging', 0):.2f})\n"

            output += "\n"

    return output
