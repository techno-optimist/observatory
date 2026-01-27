"""
Human Annotation Framework

Collects and manages human annotations for validating projection accuracy.
Computes inter-annotator agreement metrics (Krippendorff's alpha, ICC).
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class Annotator:
    """A human annotator."""
    id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    annotation_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextAnnotation:
    """A single annotation of a text by one annotator."""
    text_id: str
    text: str
    annotator_id: str
    agency: float
    fairness: float
    belonging: float
    confidence: float = 1.0  # Annotator's confidence in their rating
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.agency, self.fairness, self.belonging])


@dataclass
class AnnotationStats:
    """Agreement statistics for annotations."""
    n_texts: int
    n_annotators: int
    n_annotations: int

    # Per-axis agreement
    agency_icc: float  # Intraclass correlation coefficient
    fairness_icc: float
    belonging_icc: float

    # Overall agreement
    mean_icc: float
    krippendorff_alpha: float  # Multi-rater agreement

    # Per-axis variance (lower = more agreement)
    agency_variance: float
    fairness_variance: float
    belonging_variance: float

    # Annotator consistency
    annotator_consistency: Dict[str, float]  # Per-annotator self-consistency

    def to_dict(self) -> dict:
        return asdict(self)


def compute_icc(ratings: np.ndarray) -> float:
    """
    Compute ICC(2,k) - two-way random effects, average measures.

    ratings: (n_subjects, n_raters) array
    """
    n_subjects, n_raters = ratings.shape

    if n_subjects < 2 or n_raters < 2:
        return np.nan

    # Mask missing values
    valid_mask = ~np.isnan(ratings)
    if not np.any(valid_mask):
        return np.nan

    # Grand mean
    grand_mean = np.nanmean(ratings)

    # Subject means
    subject_means = np.nanmean(ratings, axis=1)

    # Rater means
    rater_means = np.nanmean(ratings, axis=0)

    # Sum of squares
    ss_total = np.nansum((ratings - grand_mean) ** 2)
    ss_subjects = n_raters * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n_subjects * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    # Mean squares
    ms_subjects = ss_subjects / (n_subjects - 1) if n_subjects > 1 else 0
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1)) if (n_subjects > 1 and n_raters > 1) else 0

    # ICC(2,k)
    if ms_error == 0:
        return 1.0 if ms_subjects > 0 else np.nan

    icc = (ms_subjects - ms_error) / ms_subjects if ms_subjects > 0 else 0
    return max(0, min(1, icc))


def compute_krippendorff_alpha(ratings: np.ndarray, level: str = "interval") -> float:
    """
    Compute Krippendorff's alpha for inter-rater reliability.

    ratings: (n_subjects, n_raters) array
    level: "nominal", "ordinal", or "interval"
    """
    n_subjects, n_raters = ratings.shape

    if n_subjects < 2 or n_raters < 2:
        return np.nan

    # Flatten to get all values
    all_values = ratings[~np.isnan(ratings)]
    if len(all_values) < 2:
        return np.nan

    # Count coincidences
    n = len(all_values)

    if level == "interval":
        # For interval data, use squared differences
        observed_disagreement = 0
        expected_disagreement = 0

        for i in range(n_subjects):
            rater_values = ratings[i, ~np.isnan(ratings[i])]
            m = len(rater_values)
            if m < 2:
                continue

            for j in range(m):
                for k in range(j + 1, m):
                    observed_disagreement += (rater_values[j] - rater_values[k]) ** 2

        # Expected disagreement
        for i in range(n):
            for j in range(i + 1, n):
                expected_disagreement += (all_values[i] - all_values[j]) ** 2

        if expected_disagreement == 0:
            return 1.0

        # Normalize
        n_pairs_observed = sum(
            len(ratings[i, ~np.isnan(ratings[i])]) * (len(ratings[i, ~np.isnan(ratings[i])]) - 1) / 2
            for i in range(n_subjects)
        )
        n_pairs_expected = n * (n - 1) / 2

        if n_pairs_observed == 0 or n_pairs_expected == 0:
            return np.nan

        observed_disagreement /= n_pairs_observed
        expected_disagreement /= n_pairs_expected

        alpha = 1 - observed_disagreement / expected_disagreement
        return max(-1, min(1, alpha))

    else:
        # For nominal/ordinal, use simpler proportion agreement
        # This is a simplified version
        agreements = 0
        comparisons = 0

        for i in range(n_subjects):
            rater_values = ratings[i, ~np.isnan(ratings[i])]
            m = len(rater_values)
            if m < 2:
                continue

            for j in range(m):
                for k in range(j + 1, m):
                    if level == "nominal":
                        agreements += 1 if rater_values[j] == rater_values[k] else 0
                    else:  # ordinal
                        agreements += 1 - abs(rater_values[j] - rater_values[k]) / 4  # Assuming -2 to 2 scale
                    comparisons += 1

        if comparisons == 0:
            return np.nan

        return agreements / comparisons


def compute_agreement_metrics(annotations: List[TextAnnotation]) -> AnnotationStats:
    """
    Compute comprehensive agreement metrics from a list of annotations.
    """
    if not annotations:
        raise ValueError("No annotations provided")

    # Group by text and annotator
    texts = {}
    annotators = set()

    for ann in annotations:
        annotators.add(ann.annotator_id)
        if ann.text_id not in texts:
            texts[ann.text_id] = {}
        texts[ann.text_id][ann.annotator_id] = ann

    text_ids = sorted(texts.keys())
    annotator_ids = sorted(annotators)

    n_texts = len(text_ids)
    n_annotators = len(annotator_ids)

    # Build rating matrices (n_texts, n_annotators) for each axis
    agency_matrix = np.full((n_texts, n_annotators), np.nan)
    fairness_matrix = np.full((n_texts, n_annotators), np.nan)
    belonging_matrix = np.full((n_texts, n_annotators), np.nan)

    for i, text_id in enumerate(text_ids):
        for j, ann_id in enumerate(annotator_ids):
            if ann_id in texts[text_id]:
                ann = texts[text_id][ann_id]
                agency_matrix[i, j] = ann.agency
                fairness_matrix[i, j] = ann.fairness
                belonging_matrix[i, j] = ann.belonging

    # Compute ICCs
    agency_icc = compute_icc(agency_matrix)
    fairness_icc = compute_icc(fairness_matrix)
    belonging_icc = compute_icc(belonging_matrix)

    # Mean ICC
    iccs = [x for x in [agency_icc, fairness_icc, belonging_icc] if not np.isnan(x)]
    mean_icc = np.mean(iccs) if iccs else np.nan

    # Compute Krippendorff's alpha on combined ratings
    combined_matrix = np.hstack([agency_matrix, fairness_matrix, belonging_matrix])
    kripp_alpha = compute_krippendorff_alpha(
        np.hstack([agency_matrix, fairness_matrix, belonging_matrix])
    )

    # Compute variances
    agency_variance = np.nanvar(agency_matrix, axis=1).mean()
    fairness_variance = np.nanvar(fairness_matrix, axis=1).mean()
    belonging_variance = np.nanvar(belonging_matrix, axis=1).mean()

    # Per-annotator consistency (self-agreement on duplicate texts if any)
    annotator_consistency = {}
    for ann_id in annotator_ids:
        ann_annotations = [a for a in annotations if a.annotator_id == ann_id]
        # For now, use variance of their ratings as inverse consistency proxy
        if len(ann_annotations) > 1:
            vectors = np.array([a.vector for a in ann_annotations])
            consistency = 1 / (1 + np.var(vectors))
            annotator_consistency[ann_id] = float(consistency)
        else:
            annotator_consistency[ann_id] = np.nan

    return AnnotationStats(
        n_texts=n_texts,
        n_annotators=n_annotators,
        n_annotations=len(annotations),
        agency_icc=float(agency_icc) if not np.isnan(agency_icc) else 0.0,
        fairness_icc=float(fairness_icc) if not np.isnan(fairness_icc) else 0.0,
        belonging_icc=float(belonging_icc) if not np.isnan(belonging_icc) else 0.0,
        mean_icc=float(mean_icc) if not np.isnan(mean_icc) else 0.0,
        krippendorff_alpha=float(kripp_alpha) if not np.isnan(kripp_alpha) else 0.0,
        agency_variance=float(agency_variance) if not np.isnan(agency_variance) else 0.0,
        fairness_variance=float(fairness_variance) if not np.isnan(fairness_variance) else 0.0,
        belonging_variance=float(belonging_variance) if not np.isnan(belonging_variance) else 0.0,
        annotator_consistency=annotator_consistency
    )


class AnnotationDataset:
    """
    Manages a collection of human annotations for validation.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("./data/annotations")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.annotators: Dict[str, Annotator] = {}
        self.annotations: List[TextAnnotation] = []
        self.texts: Dict[str, str] = {}  # text_id -> text

        self._load()

    def _load(self):
        """Load existing data from disk."""
        # Load annotators
        annotators_file = self.data_dir / "annotators.json"
        if annotators_file.exists():
            with open(annotators_file) as f:
                data = json.load(f)
                self.annotators = {a["id"]: Annotator(**a) for a in data}

        # Load texts
        texts_file = self.data_dir / "texts.json"
        if texts_file.exists():
            with open(texts_file) as f:
                self.texts = json.load(f)

        # Load annotations
        annotations_file = self.data_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file) as f:
                data = json.load(f)
                self.annotations = [TextAnnotation(**a) for a in data]

        logger.info(f"Loaded {len(self.annotations)} annotations from {len(self.annotators)} annotators")

    def _save(self):
        """Save all data to disk."""
        # Save annotators
        with open(self.data_dir / "annotators.json", "w") as f:
            json.dump([a.to_dict() for a in self.annotators.values()], f, indent=2)

        # Save texts
        with open(self.data_dir / "texts.json", "w") as f:
            json.dump(self.texts, f, indent=2)

        # Save annotations
        with open(self.data_dir / "annotations.json", "w") as f:
            json.dump([a.to_dict() for a in self.annotations], f, indent=2)

    def add_annotator(self, name: str) -> Annotator:
        """Add a new annotator."""
        ann_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        annotator = Annotator(id=ann_id, name=name)
        self.annotators[ann_id] = annotator
        self._save()
        return annotator

    def add_text(self, text: str) -> str:
        """Add a text for annotation. Returns text_id."""
        text_id = hashlib.md5(text.encode()).hexdigest()[:12]
        if text_id not in self.texts:
            self.texts[text_id] = text
            self._save()
        return text_id

    def add_texts_batch(self, texts: List[str]) -> List[str]:
        """Add multiple texts. Returns list of text_ids."""
        text_ids = []
        for text in texts:
            text_id = hashlib.md5(text.encode()).hexdigest()[:12]
            if text_id not in self.texts:
                self.texts[text_id] = text
            text_ids.append(text_id)
        self._save()
        return text_ids

    def add_annotation(
        self,
        text_id: str,
        annotator_id: str,
        agency: float,
        fairness: float,
        belonging: float,
        confidence: float = 1.0,
        notes: str = ""
    ) -> TextAnnotation:
        """Add an annotation."""
        if text_id not in self.texts:
            raise ValueError(f"Unknown text_id: {text_id}")
        if annotator_id not in self.annotators:
            raise ValueError(f"Unknown annotator_id: {annotator_id}")

        # Validate ranges
        for val, name in [(agency, "agency"), (fairness, "fairness"), (belonging, "belonging")]:
            if not -2.0 <= val <= 2.0:
                raise ValueError(f"{name} must be between -2 and 2, got {val}")

        annotation = TextAnnotation(
            text_id=text_id,
            text=self.texts[text_id],
            annotator_id=annotator_id,
            agency=agency,
            fairness=fairness,
            belonging=belonging,
            confidence=confidence,
            notes=notes
        )

        self.annotations.append(annotation)
        self.annotators[annotator_id].annotation_count += 1
        self._save()

        return annotation

    def get_texts_for_annotation(
        self,
        annotator_id: str,
        min_annotators: int = 3,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get texts that need more annotations, prioritizing those
        with fewer annotations to ensure even coverage.
        """
        # Count annotations per text
        annotation_counts = {}
        annotated_by = {}

        for ann in self.annotations:
            if ann.text_id not in annotation_counts:
                annotation_counts[ann.text_id] = 0
                annotated_by[ann.text_id] = set()
            annotation_counts[ann.text_id] += 1
            annotated_by[ann.text_id].add(ann.annotator_id)

        # Find texts this annotator hasn't annotated yet
        candidates = []
        for text_id, text in self.texts.items():
            if annotator_id not in annotated_by.get(text_id, set()):
                count = annotation_counts.get(text_id, 0)
                if count < min_annotators:
                    candidates.append({
                        "text_id": text_id,
                        "text": text,
                        "current_annotations": count,
                        "priority": min_annotators - count
                    })

        # Sort by priority (most needed first)
        candidates.sort(key=lambda x: (-x["priority"], x["text_id"]))

        return candidates[:limit]

    def get_agreement_stats(self) -> AnnotationStats:
        """Compute agreement statistics."""
        return compute_agreement_metrics(self.annotations)

    def get_consensus_labels(self, min_annotations: int = 2) -> List[Dict]:
        """
        Get consensus labels (mean of annotations) for texts with
        sufficient annotations.
        """
        # Group annotations by text
        text_annotations = {}
        for ann in self.annotations:
            if ann.text_id not in text_annotations:
                text_annotations[ann.text_id] = []
            text_annotations[ann.text_id].append(ann)

        results = []
        for text_id, anns in text_annotations.items():
            if len(anns) >= min_annotations:
                vectors = np.array([a.vector for a in anns])
                mean_vector = vectors.mean(axis=0)
                std_vector = vectors.std(axis=0)

                results.append({
                    "text_id": text_id,
                    "text": self.texts[text_id],
                    "n_annotations": len(anns),
                    "consensus": {
                        "agency": float(mean_vector[0]),
                        "fairness": float(mean_vector[1]),
                        "belonging": float(mean_vector[2])
                    },
                    "uncertainty": {
                        "agency": float(std_vector[0]),
                        "fairness": float(std_vector[1]),
                        "belonging": float(std_vector[2])
                    }
                })

        return results

    def export_for_training(self, min_annotations: int = 2) -> List[Dict]:
        """
        Export consensus labels in format suitable for training.
        """
        consensus = self.get_consensus_labels(min_annotations)
        return [
            {
                "text": c["text"],
                "agency": c["consensus"]["agency"],
                "fairness": c["consensus"]["fairness"],
                "belonging": c["consensus"]["belonging"],
                "source": "human_consensus",
                "n_annotators": c["n_annotations"],
                "uncertainty": c["uncertainty"]
            }
            for c in consensus
        ]
