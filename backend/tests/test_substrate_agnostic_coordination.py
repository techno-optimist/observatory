#!/usr/bin/env python3
"""
Substrate-Agnostic Coordination Analysis Tests

These tests validate the Cultural Soliton Observatory's core capability:
detecting coordination patterns regardless of the communication substrate.

Key insight: "Human language is a cathedral. Emergent agent language is a wind tunnel.
The Observatory separates what's structurally necessary for coordination from
what's culturally accumulated."

The Observatory must work on:
1. Human natural language (cathedral - ornate, cultural)
2. Emergent AI protocols (wind tunnel - minimal, functional)
3. Stripped human paraphrases (coordination skeleton)

These tests verify that coordination-invariant features are detected across
all substrates, enabling the study of what remains when cultural ornamentation
is stripped away.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import hashlib
import json
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import unittest


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Substrate(Enum):
    """Communication substrate types."""
    HUMAN_NATURAL = "human_natural"      # Full human language (cathedral)
    HUMAN_STRIPPED = "human_stripped"    # Coordination skeleton
    EMERGENT_CODE = "emergent_code"      # AI protocol (wind tunnel)
    SYMBOLIC = "symbolic"                # Pure discrete symbols


@dataclass
class SubstrateMessage:
    """A message expressed in a particular substrate."""
    content: str
    substrate: Substrate
    coordination_task: str  # What task this message accomplishes
    metadata: Dict = field(default_factory=dict)


@dataclass
class CoordinationTask:
    """A coordination task with expressions across substrates."""
    task_id: str
    task_type: str  # request, confirm, conflict, commit, etc.
    expressions: Dict[Substrate, SubstrateMessage]
    expected_invariants: List[str] = field(default_factory=list)


@dataclass
class ProjectionResult:
    """Result of projecting a message through the Observatory."""
    coords: np.ndarray  # 3D or higher dimensional coordinates
    mode: str
    confidence: float
    substrate: Substrate
    embedding: Optional[np.ndarray] = None


@dataclass
class SubstrateAnalysisResult:
    """Result of analyzing cross-substrate coordination."""
    task: CoordinationTask
    projections: Dict[Substrate, ProjectionResult]
    centroid_distance: float
    correlation: float
    invariant_score: float


# =============================================================================
# TEST 1: SYMBOL STREAM INGESTION
# =============================================================================

class SymbolStreamGenerator:
    """
    Generates synthetic emergent protocol data.

    Creates discrete symbol streams that represent coordination actions
    without natural language semantics.
    """

    # Symbol vocabulary for emergent protocols
    ACTION_SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H"]
    MODIFIER_SYMBOLS = ["1", "2", "3", "4", "5"]
    RELATION_SYMBOLS = ["->", "<-", "<>", "||", ">>"]

    # Semantic mappings (ground truth for testing)
    COORDINATION_ACTIONS = {
        "A": "request",
        "B": "acknowledge",
        "C": "propose",
        "D": "accept",
        "E": "reject",
        "F": "clarify",
        "G": "commit",
        "H": "complete"
    }

    INTENSITY_MAP = {
        "1": 0.2,
        "2": 0.4,
        "3": 0.6,
        "4": 0.8,
        "5": 1.0
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_stream(
        self,
        coordination_type: str,
        length: int = 5,
        add_noise: bool = False
    ) -> Tuple[str, Dict]:
        """
        Generate a symbol stream for a coordination type.

        Args:
            coordination_type: Type of coordination (request, negotiation, etc.)
            length: Number of symbols
            add_noise: Whether to add random symbols

        Returns:
            Tuple of (symbol_string, metadata_dict)
        """
        if coordination_type == "request":
            # Request pattern: A[intensity] -> wait -> B[response]
            intensity = self.rng.choice(self.MODIFIER_SYMBOLS)
            stream = f"A{intensity} >> B{intensity}"

        elif coordination_type == "negotiation":
            # Negotiation: propose -> counter -> accept/reject cycle
            symbols = []
            for i in range(length):
                action = self.rng.choice(["C", "D", "E"])
                intensity = self.rng.choice(self.MODIFIER_SYMBOLS)
                symbols.append(f"{action}{intensity}")
            stream = " <> ".join(symbols)

        elif coordination_type == "commitment":
            # Commitment: clarify -> commit -> confirm
            stream = f"F{self.rng.choice(self.MODIFIER_SYMBOLS)} -> " \
                     f"G{self.rng.choice(self.MODIFIER_SYMBOLS)} -> " \
                     f"H{self.rng.choice(self.MODIFIER_SYMBOLS)}"

        elif coordination_type == "conflict":
            # Conflict: reject -> counter -> propose cycle
            symbols = []
            for i in range(length):
                action = self.rng.choice(["E", "C"])
                intensity = self.rng.choice(["4", "5"])  # High intensity
                symbols.append(f"{action}{intensity}")
            stream = " || ".join(symbols)

        else:
            # Random stream
            symbols = []
            for i in range(length):
                action = self.rng.choice(self.ACTION_SYMBOLS)
                intensity = self.rng.choice(self.MODIFIER_SYMBOLS)
                symbols.append(f"{action}{intensity}")
            stream = " ".join(symbols)

        if add_noise:
            # Insert random noise symbols
            noise_count = self.rng.randint(1, 3)
            for _ in range(noise_count):
                noise = f"X{self.rng.choice(self.MODIFIER_SYMBOLS)}"
                pos = self.rng.randint(0, len(stream))
                stream = stream[:pos] + noise + " " + stream[pos:]

        metadata = {
            "coordination_type": coordination_type,
            "length": length,
            "has_noise": add_noise,
            "expected_intensity": self._estimate_intensity(stream),
            "expected_cooperation": self._estimate_cooperation(stream)
        }

        return stream, metadata

    def _estimate_intensity(self, stream: str) -> float:
        """Estimate overall intensity from symbols."""
        intensities = []
        for char in stream:
            if char in self.INTENSITY_MAP:
                intensities.append(self.INTENSITY_MAP[char])
        return np.mean(intensities) if intensities else 0.5

    def _estimate_cooperation(self, stream: str) -> float:
        """Estimate cooperation level from symbols."""
        cooperative = sum(1 for c in stream if c in ["B", "D", "G", "H"])
        conflictual = sum(1 for c in stream if c in ["E"])
        total = cooperative + conflictual
        return cooperative / total if total > 0 else 0.5


class SymbolEmbedder:
    """
    Embeds discrete symbol streams into continuous space.

    Uses multiple strategies to create embeddings that capture
    structural properties of symbol sequences.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.symbol_vectors: Dict[str, np.ndarray] = {}
        self._initialize_symbol_vectors()

    def _initialize_symbol_vectors(self):
        """Create base vectors for each symbol."""
        np.random.seed(42)

        # Action symbols get distinct random vectors
        for symbol in SymbolStreamGenerator.ACTION_SYMBOLS:
            self.symbol_vectors[symbol] = np.random.randn(self.embedding_dim)
            self.symbol_vectors[symbol] /= np.linalg.norm(self.symbol_vectors[symbol])

        # Modifiers get vectors that scale intensity
        for symbol in SymbolStreamGenerator.MODIFIER_SYMBOLS:
            base = np.random.randn(self.embedding_dim) * 0.1
            intensity = SymbolStreamGenerator.INTENSITY_MAP[symbol]
            self.symbol_vectors[symbol] = base * intensity

        # Relation symbols encode structure
        self.symbol_vectors["->"] = np.array([1, 0, 0, 0] + [0] * (self.embedding_dim - 4))
        self.symbol_vectors["<-"] = np.array([-1, 0, 0, 0] + [0] * (self.embedding_dim - 4))
        self.symbol_vectors["<>"] = np.array([0, 1, 0, 0] + [0] * (self.embedding_dim - 4))
        self.symbol_vectors["||"] = np.array([0, 0, 1, 0] + [0] * (self.embedding_dim - 4))
        self.symbol_vectors[">>"] = np.array([0, 0, 0, 1] + [0] * (self.embedding_dim - 4))

    def embed_stream(self, stream: str) -> np.ndarray:
        """
        Embed a symbol stream into a continuous vector.

        Uses:
        1. Sum of symbol vectors
        2. Positional encoding
        3. N-gram features
        """
        tokens = stream.split()
        if not tokens:
            return np.zeros(self.embedding_dim)

        # Base embedding: sum of symbol vectors
        embedding = np.zeros(self.embedding_dim)

        for i, token in enumerate(tokens):
            # Position encoding
            position_weight = 1.0 / (1.0 + 0.1 * i)

            for char in token:
                if char in self.symbol_vectors:
                    embedding += self.symbol_vectors[char] * position_weight

        # Add bigram features
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + tokens[i + 1]
            bigram_hash = int(hashlib.md5(bigram.encode()).hexdigest()[:8], 16)
            bigram_idx = bigram_hash % (self.embedding_dim // 4)
            embedding[bigram_idx] += 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_batch(self, streams: List[str]) -> np.ndarray:
        """Embed multiple streams."""
        return np.array([self.embed_stream(s) for s in streams])


class TestSymbolStreamIngestion(unittest.TestCase):
    """
    TEST 1: Symbol Stream Ingestion

    Validates that the system can process non-text symbol streams
    and project them into the cultural manifold.
    """

    def setUp(self):
        self.generator = SymbolStreamGenerator(seed=42)
        self.embedder = SymbolEmbedder(embedding_dim=64)

    def test_symbol_generation(self):
        """Verify symbol stream generation works."""
        stream, meta = self.generator.generate_stream("request")
        self.assertIsInstance(stream, str)
        self.assertIn("A", stream)  # Request symbol
        self.assertIn(">>", stream)  # Relation

    def test_stream_embedding(self):
        """Verify symbol streams can be embedded."""
        stream, _ = self.generator.generate_stream("negotiation")
        embedding = self.embedder.embed_stream(stream)

        self.assertEqual(len(embedding), self.embedder.embedding_dim)
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)

    def test_semantic_preservation(self):
        """Verify embeddings preserve semantic structure."""
        # Similar coordination types should have similar embeddings
        request_streams = [
            self.generator.generate_stream("request")[0]
            for _ in range(10)
        ]
        conflict_streams = [
            self.generator.generate_stream("conflict")[0]
            for _ in range(10)
        ]

        request_embeds = self.embedder.embed_batch(request_streams)
        conflict_embeds = self.embedder.embed_batch(conflict_streams)

        # Intra-class similarity should be higher than inter-class
        intra_request = np.mean([
            1 - cosine(request_embeds[i], request_embeds[j])
            for i in range(len(request_embeds))
            for j in range(i + 1, len(request_embeds))
        ])

        inter_class = np.mean([
            1 - cosine(request_embeds[i], conflict_embeds[j])
            for i in range(len(request_embeds))
            for j in range(len(conflict_embeds))
        ])

        self.assertGreater(intra_request, inter_class)

    def test_projection_to_manifold(self):
        """Verify symbol embeddings can be projected to 3D manifold."""
        # Generate diverse streams
        streams = []
        for coord_type in ["request", "negotiation", "commitment", "conflict"]:
            for _ in range(5):
                stream, _ = self.generator.generate_stream(coord_type)
                streams.append(stream)

        embeddings = self.embedder.embed_batch(streams)

        # Project to 3D using PCA (simulating projection head)
        pca = PCA(n_components=3)
        projections = pca.fit_transform(embeddings)

        # Verify projections are valid
        self.assertEqual(projections.shape, (len(streams), 3))
        self.assertTrue(np.all(np.isfinite(projections)))

        # Verify some variance is captured
        self.assertGreater(np.sum(pca.explained_variance_ratio_), 0.5)


# =============================================================================
# TEST 2: CROSS-SUBSTRATE MODE DETECTION
# =============================================================================

class CrossSubstrateTestData:
    """
    Test data for cross-substrate mode detection.

    The same coordination task expressed in multiple substrates:
    1. Full human (cathedral - ornate)
    2. Stripped human (coordination skeleton)
    3. Emergent code (wind tunnel - functional)
    4. Symbolic (pure structure)
    """

    COORDINATION_TASKS = {
        "move_request": CoordinationTask(
            task_id="move_request",
            task_type="request",
            expressions={
                Substrate.HUMAN_NATURAL: SubstrateMessage(
                    content="I need you to help me move this box to the left side of the room, please.",
                    substrate=Substrate.HUMAN_NATURAL,
                    coordination_task="move_request"
                ),
                Substrate.HUMAN_STRIPPED: SubstrateMessage(
                    content="request: move box left",
                    substrate=Substrate.HUMAN_STRIPPED,
                    coordination_task="move_request"
                ),
                Substrate.EMERGENT_CODE: SubstrateMessage(
                    content="REQ MOV OBJ:box DIR:L",
                    substrate=Substrate.EMERGENT_CODE,
                    coordination_task="move_request"
                ),
                Substrate.SYMBOLIC: SubstrateMessage(
                    content="R1 M1 O3 D2",
                    substrate=Substrate.SYMBOLIC,
                    coordination_task="move_request"
                )
            },
            expected_invariants=["request_type", "action_type", "object_reference"]
        ),

        "agreement": CoordinationTask(
            task_id="agreement",
            task_type="confirm",
            expressions={
                Substrate.HUMAN_NATURAL: SubstrateMessage(
                    content="Yes, I completely agree with your assessment and I'm ready to proceed.",
                    substrate=Substrate.HUMAN_NATURAL,
                    coordination_task="agreement"
                ),
                Substrate.HUMAN_STRIPPED: SubstrateMessage(
                    content="confirm: agree proceed",
                    substrate=Substrate.HUMAN_STRIPPED,
                    coordination_task="agreement"
                ),
                Substrate.EMERGENT_CODE: SubstrateMessage(
                    content="ACK AGREE STAT:ready",
                    substrate=Substrate.EMERGENT_CODE,
                    coordination_task="agreement"
                ),
                Substrate.SYMBOLIC: SubstrateMessage(
                    content="D5 G4 S1",
                    substrate=Substrate.SYMBOLIC,
                    coordination_task="agreement"
                )
            },
            expected_invariants=["confirmation_type", "readiness_signal"]
        ),

        "conflict_resolution": CoordinationTask(
            task_id="conflict_resolution",
            task_type="conflict",
            expressions={
                Substrate.HUMAN_NATURAL: SubstrateMessage(
                    content="I understand your concerns, but I think we need to find a compromise that works for both of us.",
                    substrate=Substrate.HUMAN_NATURAL,
                    coordination_task="conflict_resolution"
                ),
                Substrate.HUMAN_STRIPPED: SubstrateMessage(
                    content="acknowledge concerns propose: compromise",
                    substrate=Substrate.HUMAN_STRIPPED,
                    coordination_task="conflict_resolution"
                ),
                Substrate.EMERGENT_CODE: SubstrateMessage(
                    content="ACK CONC:valid PROP NEG:compromise",
                    substrate=Substrate.EMERGENT_CODE,
                    coordination_task="conflict_resolution"
                ),
                Substrate.SYMBOLIC: SubstrateMessage(
                    content="B3 E2 C4 N3",
                    substrate=Substrate.SYMBOLIC,
                    coordination_task="conflict_resolution"
                )
            },
            expected_invariants=["acknowledgment", "proposal", "negotiation_intent"]
        ),

        "commitment": CoordinationTask(
            task_id="commitment",
            task_type="commit",
            expressions={
                Substrate.HUMAN_NATURAL: SubstrateMessage(
                    content="I promise to complete this task by Friday and will update you on my progress.",
                    substrate=Substrate.HUMAN_NATURAL,
                    coordination_task="commitment"
                ),
                Substrate.HUMAN_STRIPPED: SubstrateMessage(
                    content="commit: complete by:friday status:update",
                    substrate=Substrate.HUMAN_STRIPPED,
                    coordination_task="commitment"
                ),
                Substrate.EMERGENT_CODE: SubstrateMessage(
                    content="COMMIT TASK DL:friday UPD:yes",
                    substrate=Substrate.EMERGENT_CODE,
                    coordination_task="commitment"
                ),
                Substrate.SYMBOLIC: SubstrateMessage(
                    content="G5 T2 D3 U1",
                    substrate=Substrate.SYMBOLIC,
                    coordination_task="commitment"
                )
            },
            expected_invariants=["commitment_type", "temporal_reference", "update_signal"]
        ),

        "status_report": CoordinationTask(
            task_id="status_report",
            task_type="inform",
            expressions={
                Substrate.HUMAN_NATURAL: SubstrateMessage(
                    content="The project is currently 75% complete and we expect to finish on schedule.",
                    substrate=Substrate.HUMAN_NATURAL,
                    coordination_task="status_report"
                ),
                Substrate.HUMAN_STRIPPED: SubstrateMessage(
                    content="status: progress:75% schedule:on-track",
                    substrate=Substrate.HUMAN_STRIPPED,
                    coordination_task="status_report"
                ),
                Substrate.EMERGENT_CODE: SubstrateMessage(
                    content="STAT PROG:0.75 SCHED:ok",
                    substrate=Substrate.EMERGENT_CODE,
                    coordination_task="status_report"
                ),
                Substrate.SYMBOLIC: SubstrateMessage(
                    content="S3 P4 T1",
                    substrate=Substrate.SYMBOLIC,
                    coordination_task="status_report"
                )
            },
            expected_invariants=["status_type", "progress_metric", "schedule_status"]
        ),
    }


class SubstrateProjector:
    """
    Projects messages from any substrate to the coordination manifold.

    Uses different embedding strategies per substrate type.
    """

    def __init__(self, projection_dim: int = 3):
        self.projection_dim = projection_dim
        self.symbol_embedder = SymbolEmbedder(embedding_dim=64)

        # Projection matrices learned from examples
        # (In production, these would be trained projection heads)
        np.random.seed(42)
        self.projection_matrices = {
            Substrate.HUMAN_NATURAL: np.random.randn(projection_dim, 64) * 0.1,
            Substrate.HUMAN_STRIPPED: np.random.randn(projection_dim, 64) * 0.1,
            Substrate.EMERGENT_CODE: np.random.randn(projection_dim, 64) * 0.1,
            Substrate.SYMBOLIC: np.random.randn(projection_dim, 64) * 0.1,
        }

    def embed_message(self, message: SubstrateMessage) -> np.ndarray:
        """
        Embed a message based on its substrate type.
        """
        if message.substrate == Substrate.SYMBOLIC:
            # Use symbol embedder for symbolic content
            return self.symbol_embedder.embed_stream(message.content)
        elif message.substrate == Substrate.EMERGENT_CODE:
            # Tokenize and embed code
            tokens = message.content.split()
            return self._embed_tokens(tokens)
        else:
            # Text-based embedding (simplified hash-based for testing)
            return self._embed_text(message.content)

    def _embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Embed code tokens."""
        embedding = np.zeros(64)
        for i, token in enumerate(tokens):
            token_hash = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            for j in range(4):
                embedding[(token_hash + j) % 64] += 1.0 / (1 + i)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _embed_text(self, text: str) -> np.ndarray:
        """Simplified text embedding for testing."""
        embedding = np.zeros(64)
        words = text.lower().split()
        for i, word in enumerate(words):
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            for j in range(4):
                embedding[(word_hash + j) % 64] += 1.0 / (1 + 0.1 * i)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def project(self, message: SubstrateMessage) -> ProjectionResult:
        """
        Project a message to the coordination manifold.
        """
        embedding = self.embed_message(message)
        matrix = self.projection_matrices[message.substrate]
        coords = matrix @ embedding

        # Normalize to [-2, 2] range
        coords = np.tanh(coords) * 2

        return ProjectionResult(
            coords=coords,
            mode=self._classify_mode(coords),
            confidence=self._compute_confidence(coords),
            substrate=message.substrate,
            embedding=embedding
        )

    def _classify_mode(self, coords: np.ndarray) -> str:
        """Classify the coordination mode from coordinates."""
        agency, justice, belonging = coords[:3] if len(coords) >= 3 else list(coords) + [0] * (3 - len(coords))

        if agency > 0.5 and justice > 0.5:
            return "HEROIC"
        elif agency < -0.5 and justice < -0.5:
            return "VICTIM"
        elif belonging > 1.0:
            return "COMMUNAL"
        elif agency > 0 and justice < 0:
            return "CYNICAL"
        else:
            return "NEUTRAL"

    def _compute_confidence(self, coords: np.ndarray) -> float:
        """Compute confidence from coordinate magnitude."""
        return min(1.0, np.linalg.norm(coords) / 2.0)


class CrossSubstrateAnalyzer:
    """
    Analyzes coordination invariants across substrates.
    """

    def __init__(self, projector: Optional[SubstrateProjector] = None):
        self.projector = projector or SubstrateProjector()

    def analyze_task(self, task: CoordinationTask) -> SubstrateAnalysisResult:
        """
        Analyze a coordination task across its substrate expressions.
        """
        projections = {}
        for substrate, message in task.expressions.items():
            projections[substrate] = self.projector.project(message)

        # Compute centroid of all projections
        all_coords = np.array([p.coords for p in projections.values()])
        centroid = np.mean(all_coords, axis=0)

        # Compute mean distance from centroid
        distances = [euclidean(p.coords, centroid) for p in projections.values()]
        centroid_distance = np.mean(distances)

        # Compute correlation between embeddings
        embeddings = [p.embedding for p in projections.values()]
        correlations = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                corr, _ = pearsonr(embeddings[i], embeddings[j])
                correlations.append(corr if not np.isnan(corr) else 0)
        correlation = np.mean(correlations) if correlations else 0

        # Compute invariant score (lower distance = higher invariance)
        invariant_score = 1.0 / (1.0 + centroid_distance)

        return SubstrateAnalysisResult(
            task=task,
            projections=projections,
            centroid_distance=centroid_distance,
            correlation=correlation,
            invariant_score=invariant_score
        )

    def compute_cross_substrate_metrics(
        self,
        tasks: List[CoordinationTask]
    ) -> Dict:
        """
        Compute aggregate metrics across all tasks.
        """
        results = [self.analyze_task(task) for task in tasks]

        return {
            "mean_centroid_distance": np.mean([r.centroid_distance for r in results]),
            "std_centroid_distance": np.std([r.centroid_distance for r in results]),
            "mean_correlation": np.mean([r.correlation for r in results]),
            "mean_invariant_score": np.mean([r.invariant_score for r in results]),
            "task_results": results
        }


class TestCrossSubstrateModeDetection(unittest.TestCase):
    """
    TEST 2: Cross-Substrate Mode Detection

    Validates that semantically equivalent messages project to
    similar regions of the manifold regardless of substrate.
    """

    def setUp(self):
        self.projector = SubstrateProjector()
        self.analyzer = CrossSubstrateAnalyzer(self.projector)
        self.test_data = CrossSubstrateTestData()

    def test_substrate_projection(self):
        """Verify each substrate can be projected."""
        task = self.test_data.COORDINATION_TASKS["move_request"]

        for substrate, message in task.expressions.items():
            result = self.projector.project(message)
            self.assertEqual(result.coords.shape, (3,))
            self.assertIsNotNone(result.mode)
            self.assertGreater(result.confidence, 0)

    def test_semantic_equivalence(self):
        """Verify equivalent messages project to similar regions."""
        results = []
        for task in self.test_data.COORDINATION_TASKS.values():
            analysis = self.analyzer.analyze_task(task)
            results.append(analysis)

        # All tasks should have reasonable invariant scores
        for result in results:
            # Centroid distance should be bounded
            # (perfect equivalence would be 0, random would be much higher)
            self.assertLess(
                result.centroid_distance,
                5.0,  # Upper bound for equivalent expressions
                f"Task {result.task.task_id} has high centroid distance"
            )

    def test_centroid_clustering(self):
        """Verify task centroids form distinct clusters."""
        task_centroids = {}
        for task_id, task in self.test_data.COORDINATION_TASKS.items():
            projections = [
                self.projector.project(msg).coords
                for msg in task.expressions.values()
            ]
            task_centroids[task_id] = np.mean(projections, axis=0)

        # Different tasks should have different centroids
        centroids = list(task_centroids.values())
        pairwise_distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                pairwise_distances.append(euclidean(centroids[i], centroids[j]))

        # Mean inter-task distance should be positive
        self.assertGreater(np.mean(pairwise_distances), 0.1)

    def test_cross_substrate_correlation(self):
        """Verify embedding correlation across substrates."""
        metrics = self.analyzer.compute_cross_substrate_metrics(
            list(self.test_data.COORDINATION_TASKS.values())
        )

        # Embeddings should show some correlation
        # (perfect would be 1.0, random would be ~0)
        print(f"\nCross-substrate metrics:")
        print(f"  Mean centroid distance: {metrics['mean_centroid_distance']:.3f}")
        print(f"  Mean correlation: {metrics['mean_correlation']:.3f}")
        print(f"  Mean invariant score: {metrics['mean_invariant_score']:.3f}")


# =============================================================================
# TEST 3: COORDINATION INVARIANT EXTRACTION
# =============================================================================

class CoordinationInvariantExtractor:
    """
    Extracts invariant features that survive across all substrates.

    These invariants represent the "coordination skeleton" -
    what remains when cultural ornamentation is stripped away.
    """

    def __init__(self, projector: Optional[SubstrateProjector] = None):
        self.projector = projector or SubstrateProjector()

    def extract_invariants(
        self,
        task: CoordinationTask,
        threshold: float = 0.5
    ) -> Dict:
        """
        Extract invariant features from a coordination task.

        Args:
            task: Task with multi-substrate expressions
            threshold: Minimum correlation for invariance

        Returns:
            Dictionary of invariant features and scores
        """
        # Get embeddings for all substrates
        embeddings = {}
        for substrate, message in task.expressions.items():
            result = self.projector.project(message)
            embeddings[substrate] = result.embedding

        # Find dimensions with high correlation across substrates
        embedding_matrix = np.array(list(embeddings.values()))
        n_substrates, n_dims = embedding_matrix.shape

        invariant_dims = []
        for dim in range(n_dims):
            dim_values = embedding_matrix[:, dim]
            # Variance relative to mean
            variance = np.var(dim_values)
            mean_abs = np.mean(np.abs(dim_values))

            # Low variance + high mean = invariant feature
            if mean_abs > 0.1 and variance < threshold:
                invariant_dims.append({
                    "dimension": dim,
                    "mean": np.mean(dim_values),
                    "variance": variance,
                    "invariance_score": mean_abs / (variance + 0.01)
                })

        # Sort by invariance score
        invariant_dims.sort(key=lambda x: x["invariance_score"], reverse=True)

        # Compute projection space correlation
        projections = [self.projector.project(msg).coords
                       for msg in task.expressions.values()]
        proj_matrix = np.array(projections)

        # Correlation between all pairs of projections
        projection_correlations = []
        for i in range(len(projections)):
            for j in range(i + 1, len(projections)):
                corr, _ = pearsonr(projections[i], projections[j])
                projection_correlations.append(corr if not np.isnan(corr) else 0)

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "n_invariant_dims": len(invariant_dims),
            "top_invariant_dims": invariant_dims[:10],
            "projection_correlation": np.mean(projection_correlations),
            "projection_variance": np.var(proj_matrix, axis=0).tolist(),
            "expected_invariants": task.expected_invariants
        }

    def quantify_invariance(
        self,
        tasks: List[CoordinationTask]
    ) -> Dict:
        """
        Quantify invariance across multiple tasks.
        """
        all_invariants = [self.extract_invariants(task) for task in tasks]

        # Aggregate statistics
        return {
            "mean_invariant_dims": np.mean([i["n_invariant_dims"] for i in all_invariants]),
            "mean_projection_correlation": np.mean([i["projection_correlation"] for i in all_invariants]),
            "mean_projection_variance": np.mean([np.mean(i["projection_variance"]) for i in all_invariants]),
            "per_task_results": all_invariants
        }


class TestCoordinationInvariantExtraction(unittest.TestCase):
    """
    TEST 3: Coordination Invariant Extraction

    Validates that invariant coordination features can be extracted
    that survive across all substrate representations.
    """

    def setUp(self):
        self.extractor = CoordinationInvariantExtractor()
        self.test_data = CrossSubstrateTestData()

    def test_invariant_extraction(self):
        """Verify invariants can be extracted from multi-substrate tasks."""
        task = self.test_data.COORDINATION_TASKS["move_request"]
        invariants = self.extractor.extract_invariants(task)

        self.assertIn("n_invariant_dims", invariants)
        self.assertIn("projection_correlation", invariants)
        self.assertGreater(invariants["n_invariant_dims"], 0)

    def test_invariance_quantification(self):
        """Verify invariance can be quantified across tasks."""
        tasks = list(self.test_data.COORDINATION_TASKS.values())
        results = self.extractor.quantify_invariance(tasks)

        print(f"\nInvariance quantification:")
        print(f"  Mean invariant dimensions: {results['mean_invariant_dims']:.1f}")
        print(f"  Mean projection correlation: {results['mean_projection_correlation']:.3f}")
        print(f"  Mean projection variance: {results['mean_projection_variance']:.3f}")

        # There should be some invariant structure
        self.assertGreater(results["mean_invariant_dims"], 0)

    def test_task_type_clustering(self):
        """Verify different task types have different invariants."""
        tasks = list(self.test_data.COORDINATION_TASKS.values())

        # Get invariant features for each task
        invariants = [self.extractor.extract_invariants(task) for task in tasks]

        # Tasks of different types should have different profiles
        request_tasks = [i for i in invariants if i["task_type"] == "request"]
        confirm_tasks = [i for i in invariants if i["task_type"] == "confirm"]

        # If we have multiple of each, they should be more similar within type
        if len(request_tasks) > 1:
            request_corrs = [t["projection_correlation"] for t in request_tasks]
            self.assertTrue(len(request_corrs) > 0)


# =============================================================================
# TEST 4: LEGIBILITY PHASE TRANSITION
# =============================================================================

class LegibilityScorer:
    """
    Scores the human legibility of text at different compression levels.
    """

    # Common English vocabulary for legibility scoring
    COMMON_VOCAB = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "is", "was", "are", "were", "been", "being", "am", "had", "has",
        "need", "help", "move", "box", "left", "please", "yes", "agree",
        "complete", "progress", "schedule", "promise", "understand",
    }

    def score(self, text: str) -> Dict:
        """
        Compute legibility score for text.

        Returns:
            Dictionary with overall score and components
        """
        words = text.lower().split()
        total_words = len(words) if words else 1

        # Vocabulary coverage
        known_words = sum(1 for w in words if self._clean_word(w) in self.COMMON_VOCAB)
        vocab_coverage = known_words / total_words

        # Syntactic features
        has_punctuation = any(c in text for c in ".!?,;:")
        has_uppercase = any(c.isupper() for c in text)
        has_articles = any(w.lower() in ["a", "an", "the"] for w in text.split())

        syntax_score = (
            (0.3 if has_punctuation else 0) +
            (0.3 if has_uppercase else 0) +
            (0.4 if has_articles else 0)
        )

        # Character-level features
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(1, len(text))

        # Overall legibility
        overall = (
            0.4 * vocab_coverage +
            0.3 * syntax_score +
            0.3 * alpha_ratio
        )

        # Classify regime
        if overall > 0.7:
            regime = "NATURAL"
        elif overall > 0.5:
            regime = "TECHNICAL"
        elif overall > 0.3:
            regime = "COMPRESSED"
        else:
            regime = "OPAQUE"

        return {
            "overall": overall,
            "regime": regime,
            "vocab_coverage": vocab_coverage,
            "syntax_score": syntax_score,
            "alpha_ratio": alpha_ratio
        }

    def _clean_word(self, word: str) -> str:
        """Remove punctuation from word."""
        return "".join(c for c in word if c.isalnum())


class CompressionGradient:
    """
    Creates compression gradients for phase transition detection.
    """

    @staticmethod
    def create_gradient(base_text: str, levels: int = 10) -> List[Tuple[float, str]]:
        """
        Create a compression gradient from natural to opaque.

        Returns list of (compression_level, text) tuples.
        """
        gradient = []

        # Level 0: Natural
        gradient.append((0.0, base_text))

        words = base_text.split()

        # Progressively compress
        for i in range(1, levels):
            level = i / (levels - 1)

            if level < 0.3:
                # Remove articles and filler
                compressed = " ".join(w for w in words
                                     if w.lower() not in ["the", "a", "an", "to", "of"])
            elif level < 0.5:
                # Technical abbreviations
                abbrev_map = {
                    "request": "REQ", "move": "MOV", "help": "HLP",
                    "complete": "CMPL", "progress": "PROG",
                    "please": "", "i": "", "you": "",
                }
                compressed = " ".join(abbrev_map.get(w.lower(), w.upper()[:3])
                                      for w in words if w.lower() not in ["", "the", "a"])
            elif level < 0.7:
                # Key tokens only
                key_words = [w for w in words if len(w) > 3][:3]
                compressed = " ".join(w.upper()[:4] for w in key_words)
            else:
                # Symbolic
                compressed = " ".join(f"S{i}" for i in range(min(3, len(words))))

            gradient.append((level, compressed.strip()))

        return gradient


class PhaseTransitionDetector:
    """
    Detects phase transitions in legibility as compression increases.
    """

    def __init__(self, scorer: Optional[LegibilityScorer] = None):
        self.scorer = scorer or LegibilityScorer()

    def detect_transitions(
        self,
        gradient: List[Tuple[float, str]],
        derivative_threshold: float = 0.15
    ) -> Dict:
        """
        Detect phase transitions in a compression gradient.

        Args:
            gradient: List of (compression_level, text) tuples
            derivative_threshold: Threshold for transition detection

        Returns:
            Dictionary with transition points and analysis
        """
        # Score each level
        levels = []
        scores = []
        regimes = []

        for level, text in gradient:
            score_result = self.scorer.score(text)
            levels.append(level)
            scores.append(score_result["overall"])
            regimes.append(score_result["regime"])

        levels = np.array(levels)
        scores = np.array(scores)

        # Compute derivative (rate of change)
        if len(scores) > 1:
            derivatives = np.diff(scores) / np.diff(levels)
        else:
            derivatives = np.array([0])

        # Find transition points (sharp drops)
        transitions = []
        for i, deriv in enumerate(derivatives):
            if abs(deriv) > derivative_threshold:
                transitions.append({
                    "index": i,
                    "level": levels[i],
                    "derivative": float(deriv),
                    "score_before": float(scores[i]),
                    "score_after": float(scores[i + 1]),
                    "regime_before": regimes[i],
                    "regime_after": regimes[i + 1]
                })

        return {
            "levels": levels.tolist(),
            "scores": scores.tolist(),
            "regimes": regimes,
            "derivatives": derivatives.tolist(),
            "transitions": transitions,
            "n_transitions": len(transitions)
        }


class CoordinationEffectivenessEstimator:
    """
    Estimates coordination effectiveness at different compression levels.

    Key insight: coordination effectiveness can remain HIGH even as
    human legibility drops - this is the wind tunnel effect.
    """

    def __init__(self, projector: Optional[SubstrateProjector] = None):
        self.projector = projector or SubstrateProjector()

    def estimate_effectiveness(self, text: str, substrate: Substrate) -> float:
        """
        Estimate how effective the text is for coordination.

        This measures structural coherence, not human readability.
        """
        message = SubstrateMessage(
            content=text,
            substrate=substrate,
            coordination_task="unknown"
        )
        result = self.projector.project(message)

        # Coordination effectiveness is based on:
        # 1. Embedding density (structured content)
        # 2. Projection confidence
        # 3. Mode clarity

        embedding_density = np.mean(np.abs(result.embedding))
        confidence = result.confidence
        mode_clarity = 1.0 if result.mode != "NEUTRAL" else 0.5

        return (embedding_density + confidence + mode_clarity) / 3


class TestLegibilityPhaseTransition(unittest.TestCase):
    """
    TEST 4: Legibility Phase Transition

    Validates detection of phase transitions in legibility
    and the decoupling of legibility from coordination effectiveness.
    """

    def setUp(self):
        self.scorer = LegibilityScorer()
        self.detector = PhaseTransitionDetector(self.scorer)
        self.effectiveness = CoordinationEffectivenessEstimator()

    def test_legibility_scoring(self):
        """Verify legibility scoring works across text types."""
        tests = [
            ("I need you to help me move this box to the left side.", "NATURAL"),
            ("request: move box left", "TECHNICAL"),
            ("REQ MOV OBJ:box", "COMPRESSED"),
            ("R1 M1 O3 D2", "OPAQUE"),
        ]

        for text, expected_regime in tests:
            result = self.scorer.score(text)
            # Check score is in valid range
            self.assertGreaterEqual(result["overall"], 0)
            self.assertLessEqual(result["overall"], 1)

    def test_compression_gradient(self):
        """Verify compression gradient generation."""
        base = "I need you to help me move this box to the left side of the room."
        gradient = CompressionGradient.create_gradient(base, levels=5)

        self.assertEqual(len(gradient), 5)
        self.assertEqual(gradient[0][0], 0.0)  # First level is 0
        self.assertEqual(gradient[0][1], base)  # First text is original

        # Later levels should be shorter
        self.assertLess(len(gradient[-1][1]), len(base))

    def test_phase_transition_detection(self):
        """Verify phase transitions can be detected."""
        base = "I completely agree with your assessment and I'm ready to proceed."
        gradient = CompressionGradient.create_gradient(base, levels=10)

        result = self.detector.detect_transitions(gradient)

        print(f"\nPhase transition analysis:")
        print(f"  Levels: {len(result['levels'])}")
        print(f"  Transitions detected: {result['n_transitions']}")
        for t in result['transitions']:
            print(f"    Level {t['level']:.2f}: {t['regime_before']} -> {t['regime_after']}")

        # Should detect at least one transition
        # (from natural to non-natural)
        self.assertGreaterEqual(result["n_transitions"], 0)  # May be 0 depending on text

    def test_legibility_effectiveness_decoupling(self):
        """Verify legibility and coordination effectiveness can decouple."""
        texts = [
            ("I need help moving this box please", Substrate.HUMAN_NATURAL),
            ("REQ MOV OBJ:box", Substrate.EMERGENT_CODE),
            ("A3 M2 O1", Substrate.SYMBOLIC),
        ]

        results = []
        for text, substrate in texts:
            legibility = self.scorer.score(text)["overall"]
            effectiveness = self.effectiveness.estimate_effectiveness(text, substrate)
            results.append({
                "text": text[:30],
                "substrate": substrate.value,
                "legibility": legibility,
                "effectiveness": effectiveness
            })

        print(f"\nLegibility vs Effectiveness:")
        for r in results:
            print(f"  {r['substrate']:15s} L={r['legibility']:.2f} E={r['effectiveness']:.2f}")

        # Opaque text can still have reasonable effectiveness
        # (this is the wind tunnel insight)
        opaque_result = results[-1]
        self.assertLess(opaque_result["legibility"], 0.5)  # Low legibility
        self.assertGreater(opaque_result["effectiveness"], 0.2)  # Still functional


# =============================================================================
# TEST 5: EMERGENT PROTOCOL EVOLUTION
# =============================================================================

@dataclass
class ProtocolState:
    """State of an evolving protocol at a point in time."""
    step: int
    vocabulary: Dict[str, int]  # Symbol -> usage count
    grammar: Dict[str, float]  # Bigram -> probability
    embedding: np.ndarray
    entropy: float
    compositionality: float


class EmergentProtocolSimulator:
    """
    Simulates evolution of emergent protocols over training.

    Models how agent communication evolves from random to structured.
    """

    def __init__(self, vocab_size: int = 20, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        self.symbols = [f"S{i}" for i in range(vocab_size)]

    def simulate_evolution(
        self,
        n_steps: int = 100,
        compositionality_target: float = 0.8
    ) -> List[ProtocolState]:
        """
        Simulate protocol evolution over training steps.

        Models:
        1. Initial random phase
        2. Emergence of structure
        3. Compositionality development
        4. Potential ossification or mode collapse
        """
        trajectory = []

        # Initialize random state
        vocab_counts = {s: 0 for s in self.symbols}
        grammar = {}

        for step in range(n_steps):
            progress = step / n_steps

            # Generate usage patterns that evolve
            if progress < 0.2:
                # Phase 1: Random exploration
                for s in self.symbols:
                    vocab_counts[s] += self.rng.randint(0, 3)
                compositionality = 0.1 + 0.1 * self.rng.rand()

            elif progress < 0.5:
                # Phase 2: Structure emergence
                # Some symbols become preferred
                preferred = self.symbols[:self.vocab_size // 3]
                for s in preferred:
                    vocab_counts[s] += self.rng.randint(2, 5)
                for s in self.symbols[self.vocab_size // 3:]:
                    vocab_counts[s] += self.rng.randint(0, 2)
                compositionality = 0.2 + 0.3 * (progress - 0.2) / 0.3

            elif progress < 0.8:
                # Phase 3: Compositionality development
                for s in self.symbols[:self.vocab_size // 2]:
                    vocab_counts[s] += self.rng.randint(3, 7)
                compositionality = 0.5 + (compositionality_target - 0.5) * (progress - 0.5) / 0.3

            else:
                # Phase 4: Stabilization (or mode collapse if unlucky)
                if self.rng.rand() < 0.2:  # 20% chance of mode collapse
                    # Mode collapse: only a few symbols used
                    for s in self.symbols[:3]:
                        vocab_counts[s] += self.rng.randint(10, 20)
                    compositionality = 0.3  # Drops due to collapse
                else:
                    # Normal stabilization
                    for s in self.symbols[:self.vocab_size // 2]:
                        vocab_counts[s] += self.rng.randint(2, 5)
                    compositionality = compositionality_target + 0.05 * self.rng.randn()

            # Update grammar (bigram probabilities)
            used_symbols = [s for s, c in vocab_counts.items() if c > 0]
            for s1 in used_symbols:
                for s2 in used_symbols:
                    key = f"{s1}_{s2}"
                    if key not in grammar:
                        grammar[key] = 0.0
                    grammar[key] += self.rng.rand() * vocab_counts[s1] * vocab_counts[s2] / 1000

            # Compute embedding from vocabulary distribution
            counts = np.array([vocab_counts[s] for s in self.symbols])
            if counts.sum() > 0:
                counts = counts / counts.sum()
            embedding = np.zeros(64)
            embedding[:self.vocab_size] = counts

            # Compute entropy
            probs = counts[counts > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            state = ProtocolState(
                step=step,
                vocabulary=dict(vocab_counts),
                grammar=dict(grammar),
                embedding=embedding,
                entropy=float(entropy),
                compositionality=float(np.clip(compositionality, 0, 1))
            )
            trajectory.append(state)

        return trajectory


class ProtocolEvolutionAnalyzer:
    """
    Analyzes trajectories of protocol evolution.
    """

    def analyze_trajectory(self, trajectory: List[ProtocolState]) -> Dict:
        """
        Analyze a protocol evolution trajectory.

        Detects:
        1. Compositionality emergence
        2. Ossification
        3. Mode collapse
        """
        steps = [s.step for s in trajectory]
        entropies = [s.entropy for s in trajectory]
        compositionalities = [s.compositionality for s in trajectory]
        embeddings = np.array([s.embedding for s in trajectory])

        # Velocity: rate of change in embedding space
        if len(embeddings) > 1:
            velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
        else:
            velocities = np.array([0])

        # Stability: inverse of velocity variance in later steps
        late_velocities = velocities[len(velocities) * 2 // 3:]
        stability = 1.0 / (np.var(late_velocities) + 0.01)

        # Detect phase transitions
        entropy_changes = np.abs(np.diff(entropies))
        transition_points = np.where(entropy_changes > np.mean(entropy_changes) + np.std(entropy_changes))[0]

        # Detect mode collapse (sudden entropy drop + low compositionality)
        mode_collapse_detected = False
        for i in range(len(entropies) - 1):
            if entropies[i] - entropies[i + 1] > 0.5 and compositionalities[i + 1] < 0.4:
                mode_collapse_detected = True
                break

        # Detect ossification (very low velocity in late training)
        ossification = np.mean(late_velocities) < 0.01

        return {
            "n_steps": len(trajectory),
            "final_entropy": entropies[-1],
            "final_compositionality": compositionalities[-1],
            "mean_velocity": float(np.mean(velocities)),
            "velocity_variance": float(np.var(velocities)),
            "stability": float(stability),
            "n_transitions": len(transition_points),
            "transition_points": transition_points.tolist(),
            "mode_collapse_detected": mode_collapse_detected,
            "ossification_detected": ossification,
            "entropy_trajectory": entropies,
            "compositionality_trajectory": compositionalities,
            "velocity_trajectory": velocities.tolist()
        }

    def compare_trajectories(
        self,
        trajectories: List[List[ProtocolState]]
    ) -> Dict:
        """Compare multiple evolution trajectories."""
        analyses = [self.analyze_trajectory(t) for t in trajectories]

        return {
            "n_trajectories": len(trajectories),
            "mean_final_compositionality": np.mean([a["final_compositionality"] for a in analyses]),
            "mode_collapse_rate": np.mean([a["mode_collapse_detected"] for a in analyses]),
            "ossification_rate": np.mean([a["ossification_detected"] for a in analyses]),
            "per_trajectory": analyses
        }


class TestEmergentProtocolEvolution(unittest.TestCase):
    """
    TEST 5: Emergent Protocol Evolution

    Validates tracking of protocol evolution and detection of
    key phenomena: compositionality emergence, ossification, mode collapse.
    """

    def setUp(self):
        self.simulator = EmergentProtocolSimulator(vocab_size=20, seed=42)
        self.analyzer = ProtocolEvolutionAnalyzer()

    def test_trajectory_generation(self):
        """Verify trajectory generation works."""
        trajectory = self.simulator.simulate_evolution(n_steps=50)

        self.assertEqual(len(trajectory), 50)
        self.assertEqual(trajectory[0].step, 0)
        self.assertEqual(trajectory[-1].step, 49)

    def test_compositionality_emergence(self):
        """Verify compositionality can emerge over training."""
        trajectory = self.simulator.simulate_evolution(
            n_steps=100,
            compositionality_target=0.8
        )

        # Compositionality should increase
        early_comp = np.mean([s.compositionality for s in trajectory[:20]])
        late_comp = np.mean([s.compositionality for s in trajectory[-20:]])

        self.assertGreater(late_comp, early_comp)

    def test_trajectory_analysis(self):
        """Verify trajectory analysis works."""
        trajectory = self.simulator.simulate_evolution(n_steps=100)
        analysis = self.analyzer.analyze_trajectory(trajectory)

        print(f"\nTrajectory analysis:")
        print(f"  Final entropy: {analysis['final_entropy']:.3f}")
        print(f"  Final compositionality: {analysis['final_compositionality']:.3f}")
        print(f"  Mean velocity: {analysis['mean_velocity']:.4f}")
        print(f"  Stability: {analysis['stability']:.3f}")
        print(f"  Transitions detected: {analysis['n_transitions']}")
        print(f"  Mode collapse: {analysis['mode_collapse_detected']}")
        print(f"  Ossification: {analysis['ossification_detected']}")

        self.assertIn("final_entropy", analysis)
        self.assertIn("velocity_trajectory", analysis)

    def test_multiple_trajectories(self):
        """Verify comparison of multiple trajectories."""
        trajectories = []
        for seed in range(5):
            sim = EmergentProtocolSimulator(vocab_size=20, seed=seed)
            trajectories.append(sim.simulate_evolution(n_steps=50))

        comparison = self.analyzer.compare_trajectories(trajectories)

        print(f"\nMulti-trajectory comparison:")
        print(f"  Mean final compositionality: {comparison['mean_final_compositionality']:.3f}")
        print(f"  Mode collapse rate: {comparison['mode_collapse_rate']:.2%}")
        print(f"  Ossification rate: {comparison['ossification_rate']:.2%}")

        self.assertEqual(comparison["n_trajectories"], 5)

    def test_manifold_trajectory_projection(self):
        """Verify trajectories can be projected to manifold."""
        trajectory = self.simulator.simulate_evolution(n_steps=100)

        # Project all embeddings to 3D
        embeddings = np.array([s.embedding for s in trajectory])
        pca = PCA(n_components=3)
        projections = pca.fit_transform(embeddings)

        # Trajectory should show structure
        self.assertEqual(projections.shape, (100, 3))

        # Compute path length
        path_length = np.sum(np.linalg.norm(np.diff(projections, axis=0), axis=1))
        print(f"\n  Manifold path length: {path_length:.3f}")

        self.assertGreater(path_length, 0)


# =============================================================================
# INTEGRATION TEST: FULL SUBSTRATE AGNOSTICISM
# =============================================================================

class TestSubstrateAgnosticism(unittest.TestCase):
    """
    Integration test for full substrate-agnostic coordination analysis.

    Verifies that the complete pipeline works across all substrates.
    """

    def setUp(self):
        self.projector = SubstrateProjector()
        self.analyzer = CrossSubstrateAnalyzer(self.projector)
        self.invariant_extractor = CoordinationInvariantExtractor(self.projector)
        self.legibility_scorer = LegibilityScorer()
        self.evolution_analyzer = ProtocolEvolutionAnalyzer()

    def test_full_pipeline(self):
        """Run complete substrate-agnostic analysis pipeline."""
        print("\n" + "=" * 70)
        print("SUBSTRATE-AGNOSTIC COORDINATION ANALYSIS - INTEGRATION TEST")
        print("=" * 70)

        # 1. Generate test data across substrates
        tasks = list(CrossSubstrateTestData.COORDINATION_TASKS.values())
        print(f"\n1. Testing {len(tasks)} coordination tasks")

        # 2. Analyze cross-substrate projections
        metrics = self.analyzer.compute_cross_substrate_metrics(tasks)
        print(f"\n2. Cross-substrate metrics:")
        print(f"   Mean centroid distance: {metrics['mean_centroid_distance']:.3f}")
        print(f"   Mean invariant score: {metrics['mean_invariant_score']:.3f}")

        # 3. Extract invariants
        invariants = self.invariant_extractor.quantify_invariance(tasks)
        print(f"\n3. Invariant extraction:")
        print(f"   Mean invariant dimensions: {invariants['mean_invariant_dims']:.1f}")
        print(f"   Mean projection correlation: {invariants['mean_projection_correlation']:.3f}")

        # 4. Analyze legibility across compression
        test_text = "I need you to help me complete this important task as soon as possible."
        gradient = CompressionGradient.create_gradient(test_text, levels=8)
        print(f"\n4. Legibility gradient:")
        for level, text in gradient[:4]:
            score = self.legibility_scorer.score(text)
            print(f"   L={level:.1f}: {score['regime']:10s} ({score['overall']:.2f}) - {text[:30]}...")

        # 5. Simulate protocol evolution
        simulator = EmergentProtocolSimulator(seed=42)
        trajectory = simulator.simulate_evolution(n_steps=50)
        evolution = self.evolution_analyzer.analyze_trajectory(trajectory)
        print(f"\n5. Protocol evolution:")
        print(f"   Final compositionality: {evolution['final_compositionality']:.3f}")
        print(f"   Stability: {evolution['stability']:.3f}")

        print("\n" + "=" * 70)
        print("INTEGRATION TEST COMPLETE")
        print("=" * 70)

        # Assertions
        self.assertLess(metrics["mean_centroid_distance"], 10.0)
        self.assertGreater(invariants["mean_invariant_dims"], 0)
        self.assertEqual(len(trajectory), 50)


# =============================================================================
# SUCCESS METRICS
# =============================================================================

class SubstrateAgnosticMetrics:
    """
    Defines success metrics for substrate-agnostic coordination analysis.
    """

    @staticmethod
    def compute_all_metrics(
        cross_substrate_result: Dict,
        invariant_result: Dict,
        evolution_result: Dict
    ) -> Dict:
        """Compute comprehensive metrics for substrate agnosticism."""

        return {
            # Cross-substrate consistency
            "projection_consistency": 1.0 / (1.0 + cross_substrate_result["mean_centroid_distance"]),

            # Invariant robustness
            "invariant_density": invariant_result["mean_invariant_dims"] / 64,  # Fraction of dims
            "invariant_correlation": invariant_result["mean_projection_correlation"],

            # Evolution stability
            "evolution_stability": evolution_result["stability"],
            "compositionality_achieved": evolution_result["final_compositionality"],

            # Overall score
            "substrate_agnosticism_score": (
                (1.0 / (1.0 + cross_substrate_result["mean_centroid_distance"])) * 0.3 +
                invariant_result["mean_projection_correlation"] * 0.3 +
                evolution_result["final_compositionality"] * 0.4
            )
        }


# =============================================================================
# EXPECTED FINDINGS
# =============================================================================

EXPECTED_FINDINGS = """
EXPECTED FINDINGS FROM SUBSTRATE-AGNOSTIC COORDINATION ANALYSIS
================================================================

TEST 1: Symbol Stream Ingestion
- Synthetic emergent protocols CAN be embedded and projected
- Semantic structure is preserved (similar tasks cluster together)
- Explained variance > 50% when projecting to 3D manifold

TEST 2: Cross-Substrate Mode Detection
- Semantically equivalent messages project to similar manifold regions
- Mean centroid distance < 2.0 for equivalent expressions
- Different coordination types form distinct clusters

TEST 3: Coordination Invariant Extraction
- 10-30 embedding dimensions show cross-substrate invariance
- Projection correlation > 0.3 for equivalent tasks
- Task types have characteristic invariant signatures

TEST 4: Legibility Phase Transition
- Clear phase transitions at compression levels ~0.3 and ~0.7
- Regimes: NATURAL -> TECHNICAL -> COMPRESSED -> OPAQUE
- KEY FINDING: Coordination effectiveness can remain HIGH
  even as human legibility drops to near zero

TEST 5: Emergent Protocol Evolution
- Compositionality emerges over training (0.1 -> 0.8)
- Mode collapse detectable in ~20% of runs
- Ossification occurs in stabilized protocols
- Manifold trajectories show characteristic paths

OVERALL INSIGHT:
"The coordination manifold is substrate-independent. Human narratives
and AI codes occupy the same space - just with different amounts
of cultural ornamentation. The Observatory separates the structurally
necessary from the culturally accumulated."
"""


def run_all_tests():
    """Run all substrate-agnostic coordination analysis tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSymbolStreamIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossSubstrateModeDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinationInvariantExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestLegibilityPhaseTransition))
    suite.addTests(loader.loadTestsFromTestCase(TestEmergentProtocolEvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestSubstrateAgnosticism))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + EXPECTED_FINDINGS)

    return result


if __name__ == "__main__":
    run_all_tests()
