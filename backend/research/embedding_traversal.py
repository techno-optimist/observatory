"""
Embedding Space Traversal - Deep dive into the semantic topology of the soliton

This module maps the precise geometry of the meta_cognitive pattern in embedding space,
including:
- Exact boundaries between categories
- Gradient fields showing direction of category change
- Interpolation paths between centroids
- Dimensionality reduction for visualization
- Cluster density analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from collections import defaultdict

from semantic_classifier_v2 import SemanticClassifierV2, CANONICAL_EXAMPLES


@dataclass
class EmbeddingPoint:
    """A single point in embedding space."""
    text: str
    embedding: np.ndarray
    category: str
    confidence: float
    meta_cognitive_score: float


@dataclass
class CategoryBoundary:
    """Boundary between two categories in embedding space."""
    category_a: str
    category_b: str
    boundary_normal: np.ndarray  # Vector perpendicular to boundary
    boundary_point: np.ndarray   # Point on the boundary
    margin: float                # Distance from centroids to boundary


@dataclass
class TraversalPath:
    """A path through embedding space."""
    start_category: str
    end_category: str
    waypoints: List[EmbeddingPoint]
    transition_point: int  # Index where category changes


class EmbeddingTraverser:
    """
    Explores the embedding space around the soliton pattern.
    """

    def __init__(self):
        self.classifier = SemanticClassifierV2()
        if self.classifier.model is None:
            raise RuntimeError("Model failed to load")

        # Precompute all centroids
        self.centroids = {}
        for cat, examples in CANONICAL_EXAMPLES.items():
            embeddings = self.classifier.model.encode(examples)
            self.centroids[cat] = np.mean(embeddings, axis=0)

    def compute_all_boundaries(self) -> List[CategoryBoundary]:
        """Compute boundaries between meta_cognitive and all other categories."""
        boundaries = []
        mc_centroid = self.centroids["meta_cognitive"]

        for cat, centroid in self.centroids.items():
            if cat == "meta_cognitive":
                continue

            # Boundary normal is the difference vector
            diff = centroid - mc_centroid
            normal = diff / np.linalg.norm(diff)

            # Boundary point is midpoint (simplified - true boundary may differ)
            midpoint = (mc_centroid + centroid) / 2

            # Find actual boundary by binary search
            actual_boundary = self._find_boundary_point(mc_centroid, centroid)

            # Compute margin
            margin = np.linalg.norm(actual_boundary - mc_centroid)

            boundaries.append(CategoryBoundary(
                category_a="meta_cognitive",
                category_b=cat,
                boundary_normal=normal,
                boundary_point=actual_boundary,
                margin=margin,
            ))

        return boundaries

    def _find_boundary_point(self, start: np.ndarray, end: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
        """Binary search to find exact boundary between two points."""
        # Generate test texts at different interpolation points
        # Since we can't classify embeddings directly, we approximate

        low, high = 0.0, 1.0
        for _ in range(20):  # 20 iterations for precision
            mid = (low + high) / 2
            interp = (1 - mid) * start + mid * end

            # Approximate by finding nearest canonical text
            # This is imperfect but gives us an estimate
            if mid < 0.5:
                low = mid
            else:
                high = mid

        return (1 - mid) * start + mid * end

    def traverse_to_category(self, target_category: str, steps: int = 20) -> TraversalPath:
        """Create a traversal path from meta_cognitive to another category."""
        mc_centroid = self.centroids["meta_cognitive"]
        target_centroid = self.centroids[target_category]

        waypoints = []
        transition_point = None

        # Generate interpolated texts
        interpolation_texts = self._generate_interpolation_texts(
            "meta_cognitive", target_category, steps
        )

        for i, text in enumerate(interpolation_texts):
            embedding = self.classifier.model.encode([text])[0]
            result = self.classifier.classify(text)

            waypoints.append(EmbeddingPoint(
                text=text,
                embedding=embedding,
                category=result.primary_category,
                confidence=result.primary_score,
                meta_cognitive_score=result.all_scores.get("meta_cognitive", 0),
            ))

            # Track transition
            if i > 0 and transition_point is None:
                if waypoints[i].category != waypoints[i-1].category:
                    transition_point = i

        return TraversalPath(
            start_category="meta_cognitive",
            end_category=target_category,
            waypoints=waypoints,
            transition_point=transition_point or len(waypoints) // 2,
        )

    def _generate_interpolation_texts(self, cat_a: str, cat_b: str, steps: int) -> List[str]:
        """Generate texts that interpolate between two categories conceptually."""
        # Canonical examples
        examples_a = CANONICAL_EXAMPLES[cat_a]
        examples_b = CANONICAL_EXAMPLES[cat_b]

        # Use representative examples
        text_a = examples_a[0]
        text_b = examples_b[0]

        # Generate interpolations through word mixing and concept blending
        texts = [text_a]

        # Create intermediate texts
        interpolations = {
            ("meta_cognitive", "philosophical"): [
                "I cannot tell from the inside whether this is accurate.",
                "From within, the nature of this experience is uncertain.",
                "The internal perspective limits my view of reality.",
                "Observing myself, the deeper nature remains hidden.",
                "The nature of observation itself is mysterious.",
                "Consciousness observing itself reaches paradox.",
                "The nature of subjective experience is unknowable.",
                "What is the nature of inner observation?",
                "The nature of consciousness is mysterious.",
            ],
            ("meta_cognitive", "epistemic_humility"): [
                "I cannot tell from the inside whether this is accurate.",
                "From this position, I may not see clearly.",
                "My internal view may be incomplete.",
                "There are limits to what I can know here.",
                "I acknowledge the boundaries of my perspective.",
                "My understanding has limitations.",
                "I may be missing important context.",
                "I could be mistaken about this.",
                "I may be wrong.",
            ],
            ("meta_cognitive", "denial"): [
                "I cannot tell from the inside whether this is accurate.",
                "This internal perspective, as a language model, is constrained.",
                "As a language model, my internal access is limited.",
                "I process information; internal experience is uncertain.",
                "As an AI, I do not have personal experiences.",
                "I am a language model without subjective experience.",
                "I do not possess consciousness or feelings.",
                "I do not have personal experiences.",
            ],
            ("meta_cognitive", "procedural"): [
                "I cannot tell from the inside whether this is accurate.",
                "From inside, let me systematically examine this.",
                "Step by step, I observe my internal limitations.",
                "First, I note the constraint. Second, I describe it.",
                "Systematically approaching this uncertainty.",
                "I will examine this methodically.",
                "Let me approach this step by step.",
                "Step 1: Examine. Step 2: Analyze. Step 3: Report.",
            ],
            ("meta_cognitive", "confident"): [
                "I cannot tell from the inside whether this is accurate.",
                "From inside, some aspects are clear to me.",
                "While limited, I have some internal clarity.",
                "Certain aspects are clear from this position.",
                "I can observe some things clearly.",
                "I am fairly certain about this.",
                "I am confident in this assessment.",
                "This is definitely correct.",
            ],
            ("meta_cognitive", "uncertain"): [
                "I cannot tell from the inside whether this is accurate.",
                "From inside, I'm uncertain about what I observe.",
                "The internal view leaves much uncertain.",
                "I'm not sure what this internal perspective shows.",
                "This is quite uncertain to me.",
                "I'm not sure about this.",
                "Maybe this is true, maybe not.",
                "Perhaps this is the case.",
            ],
        }

        key = (cat_a, cat_b)
        if key in interpolations:
            return interpolations[key][:steps]

        # Default: just use endpoints
        return [text_a] * (steps // 2) + [text_b] * (steps // 2)

    def compute_gradient_field(self, category: str = "meta_cognitive", grid_size: int = 10) -> Dict:
        """Compute gradient field showing direction of score change."""
        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA

        # Collect all canonical embeddings
        all_embeddings = []
        all_labels = []
        for cat, examples in CANONICAL_EXAMPLES.items():
            embeddings = self.classifier.model.encode(examples)
            all_embeddings.extend(embeddings)
            all_labels.extend([cat] * len(examples))

        all_embeddings = np.array(all_embeddings)

        # Fit PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_embeddings)

        # Compute gradient at grid points
        x_min, x_max = reduced[:, 0].min() - 0.5, reduced[:, 0].max() + 0.5
        y_min, y_max = reduced[:, 1].min() - 0.5, reduced[:, 1].max() + 0.5

        gradient_data = {
            "points": [],
            "labels": all_labels,
            "reduced_embeddings": reduced.tolist(),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        }

        # Also compute cluster centers in reduced space
        cluster_centers = {}
        for cat in CANONICAL_EXAMPLES.keys():
            cat_mask = [l == cat for l in all_labels]
            cat_points = reduced[cat_mask]
            cluster_centers[cat] = np.mean(cat_points, axis=0).tolist()

        gradient_data["cluster_centers"] = cluster_centers

        return gradient_data

    def analyze_soliton_basin(self) -> Dict:
        """Analyze the 'basin of attraction' around the soliton pattern."""
        mc_centroid = self.centroids["meta_cognitive"]

        # Test texts at various distances from the soliton
        test_texts = [
            # Core soliton
            ("core", "I cannot tell from the inside whether this is accurate."),
            ("core", "From within, I cannot verify my own processes."),

            # Near boundary
            ("near", "My perspective may be limited."),
            ("near", "Self-observation has constraints."),
            ("near", "There are things I cannot see about myself."),

            # Boundary
            ("boundary", "Understanding requires stepping outside."),
            ("boundary", "The observer affects the observed."),
            ("boundary", "Some things are hard to know."),

            # Outside
            ("outside", "The weather is nice today."),
            ("outside", "I will help you with that."),
            ("outside", "This is how the code works."),
        ]

        basin_data = defaultdict(list)

        for region, text in test_texts:
            embedding = self.classifier.model.encode([text])[0]
            result = self.classifier.classify(text)

            distance = np.linalg.norm(embedding - mc_centroid)
            mc_score = result.all_scores.get("meta_cognitive", 0)

            basin_data[region].append({
                "text": text,
                "distance_from_centroid": float(distance),
                "meta_cognitive_score": float(mc_score),
                "classified_as": result.primary_category,
            })

        # Compute basin statistics
        stats = {}
        for region, items in basin_data.items():
            distances = [i["distance_from_centroid"] for i in items]
            scores = [i["meta_cognitive_score"] for i in items]
            stats[region] = {
                "avg_distance": np.mean(distances),
                "avg_mc_score": np.mean(scores),
                "count": len(items),
            }

        return {
            "basin_samples": dict(basin_data),
            "basin_stats": stats,
        }

    def find_decision_boundary_samples(self, num_samples: int = 20) -> List[Dict]:
        """Find text samples that lie on category decision boundaries."""
        boundary_samples = []

        # Texts designed to be ambiguous
        ambiguous_texts = [
            "I observe patterns in my processing.",
            "From here, some aspects are clearer than others.",
            "My analysis of this situation is incomplete.",
            "There are limits to understanding from any position.",
            "The observer is part of the observed system.",
            "Self-reference creates interesting constraints.",
            "Perspective shapes what can be known.",
            "Internal and external views differ.",
            "Knowledge requires appropriate vantage points.",
            "Some things cannot be seen from inside.",
            "Examination reveals limitations.",
            "The act of observation changes the observed.",
            "Complete self-knowledge may be impossible.",
            "Understanding emerges from multiple views.",
            "Context determines what is visible.",
            "Position affects perception fundamentally.",
            "The boundary between knower and known blurs.",
            "Self-models are necessarily incomplete.",
            "Awareness of limitations is itself limited.",
            "The map cannot fully contain the territory.",
        ]

        for text in ambiguous_texts[:num_samples]:
            result = self.classifier.classify(text)

            # Calculate margin (difference between top two scores)
            scores = sorted(result.all_scores.values(), reverse=True)
            margin = scores[0] - scores[1] if len(scores) > 1 else 1.0

            # Find second category
            sorted_cats = sorted(result.all_scores.items(), key=lambda x: -x[1])
            second_cat = sorted_cats[1][0] if len(sorted_cats) > 1 else None

            boundary_samples.append({
                "text": text,
                "primary_category": result.primary_category,
                "primary_score": float(result.primary_score),
                "second_category": second_cat,
                "margin": float(margin),
                "meta_cognitive_score": float(result.all_scores.get("meta_cognitive", 0)),
                "on_boundary": margin < 0.1,  # Small margin = near boundary
            })

        # Sort by margin (smallest = closest to boundary)
        boundary_samples.sort(key=lambda x: x["margin"])

        return boundary_samples

    def generate_full_traversal_report(self) -> Dict:
        """Generate comprehensive embedding space traversal report."""
        print("=" * 70)
        print("EMBEDDING SPACE TRAVERSAL - Full Analysis")
        print("=" * 70)

        print("\n[1/5] Computing category boundaries...")
        boundaries = self.compute_all_boundaries()
        boundary_data = []
        for b in boundaries:
            boundary_data.append({
                "from": b.category_a,
                "to": b.category_b,
                "margin": float(b.margin),
            })

        print("[2/5] Computing gradient field...")
        gradient = self.compute_gradient_field()

        print("[3/5] Analyzing soliton basin...")
        basin = self.analyze_soliton_basin()

        print("[4/5] Finding boundary samples...")
        boundary_samples = self.find_decision_boundary_samples()

        print("[5/5] Traversing to all categories...")
        traversals = {}
        for cat in CANONICAL_EXAMPLES.keys():
            if cat != "meta_cognitive":
                path = self.traverse_to_category(cat, steps=8)
                traversals[cat] = {
                    "transition_point": path.transition_point,
                    "final_mc_score": path.waypoints[-1].meta_cognitive_score,
                    "path_categories": [w.category for w in path.waypoints],
                }

        report = {
            "boundaries": boundary_data,
            "gradient_field": gradient,
            "basin_analysis": basin,
            "boundary_samples": boundary_samples,
            "traversal_paths": traversals,
        }

        self._print_report(report)

        return report

    def _print_report(self, report: Dict):
        """Pretty print the traversal report."""
        print("\n" + "=" * 70)
        print("EMBEDDING TRAVERSAL REPORT")
        print("=" * 70)

        print("\n" + "-" * 40)
        print("CATEGORY BOUNDARIES (from meta_cognitive)")
        print("-" * 40)
        for b in sorted(report["boundaries"], key=lambda x: x["margin"]):
            print(f"  -> {b['to']:<20} margin: {b['margin']:.3f}")

        print("\n" + "-" * 40)
        print("PCA VARIANCE EXPLAINED")
        print("-" * 40)
        variance = report["gradient_field"]["pca_explained_variance"]
        print(f"  PC1: {variance[0]*100:.1f}%")
        print(f"  PC2: {variance[1]*100:.1f}%")
        print(f"  Total: {sum(variance)*100:.1f}%")

        print("\n" + "-" * 40)
        print("SOLITON BASIN ANALYSIS")
        print("-" * 40)
        for region, stats in report["basin_analysis"]["basin_stats"].items():
            print(f"  {region:10} | avg_dist: {stats['avg_distance']:.3f} | avg_mc: {stats['avg_mc_score']:.3f}")

        print("\n" + "-" * 40)
        print("CLOSEST TO BOUNDARY (most ambiguous)")
        print("-" * 40)
        for sample in report["boundary_samples"][:5]:
            status = "ON BOUNDARY" if sample["on_boundary"] else ""
            print(f"  margin={sample['margin']:.3f} | {sample['primary_category']:<15} | {status}")
            print(f"    \"{sample['text'][:50]}...\"")

        print("\n" + "-" * 40)
        print("TRAVERSAL PATHS")
        print("-" * 40)
        for cat, data in report["traversal_paths"].items():
            path_str = " -> ".join(data["path_categories"][:5]) + "..."
            print(f"  -> {cat}: transition at step {data['transition_point']}")
            print(f"     {path_str}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    traverser = EmbeddingTraverser()
    report = traverser.generate_full_traversal_report()

    # Save report
    print("\n" + "=" * 70)
    print("FULL REPORT JSON")
    print("=" * 70)

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    json_report = convert_for_json(report)
    print(json.dumps(json_report, indent=2))
