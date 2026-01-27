"""
Experiment Tracker

SQLite-based experiment tracking for ML reproducibility.
Logs training runs with full provenance for publication-ready research.

Features:
- Full model provenance tracking
- Training data versioning via hashes
- Hyperparameter logging
- Metric storage
- Random seed tracking
- Query and comparison utilities
"""

import sqlite3
import json
import hashlib
import uuid
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """
    Complete record of a single experiment for reproducibility.

    Contains all information needed to reproduce the experiment:
    - Model provenance (version, weights hash)
    - Training data hash
    - Hyperparameters
    - Metrics
    - Random seed
    - Timestamps
    """
    experiment_id: str
    created_at: str

    # Model provenance
    model_id: str
    model_revision: Optional[str]
    model_sha256: Optional[str]

    # Data versioning
    examples_hash: str
    num_examples: int

    # Hyperparameters
    hyperparams: Dict[str, Any]

    # Training metrics
    metrics: Dict[str, float]

    # Reproducibility
    random_seed: int

    # Optional metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # Status tracking
    status: str = "completed"  # pending, running, completed, failed
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "examples_hash": self.examples_hash,
            "num_examples": self.num_examples,
            "hyperparams": self.hyperparams,
            "metrics": self.metrics,
            "random_seed": self.random_seed,
            "notes": self.notes,
            "tags": self.tags,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRecord":
        return cls(
            experiment_id=data["experiment_id"],
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
            model_id=data["model_id"],
            model_revision=data.get("model_revision"),
            model_sha256=data.get("model_sha256"),
            examples_hash=data["examples_hash"],
            num_examples=data["num_examples"],
            hyperparams=data.get("hyperparams", {}),
            metrics=data.get("metrics", {}),
            random_seed=data["random_seed"],
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            status=data.get("status", "completed")
        )

    def summary(self) -> dict:
        """Get a summary suitable for listing."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "num_examples": self.num_examples,
            "random_seed": self.random_seed,
            "status": self.status,
            "metrics_summary": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.metrics.items()
            },
            "tags": self.tags
        }


class ExperimentTracker:
    """
    Tracks ML experiments with full provenance for reproducibility.

    Usage:
        tracker = ExperimentTracker()

        # Log a new experiment
        exp_id = tracker.log_projection_training(
            model_provenance=provenance,
            examples_hash="abc123...",
            hyperparams={"alpha": 1.0, "layer": -1},
            metrics={"mse": 0.05, "r2": 0.92},
            seed=42
        )

        # Retrieve experiment
        exp = tracker.get_experiment(exp_id)

        # List all experiments
        experiments = tracker.list_experiments()
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else Path("./data/experiments.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local connection pool
        self._local = threading.local()

        self._init_db()
        logger.info(f"ExperimentTracker initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.row_factory = sqlite3.Row

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.close()
            self._local.conn = None
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,

                    -- Model provenance
                    model_id TEXT NOT NULL,
                    model_revision TEXT,
                    model_sha256 TEXT,

                    -- Data versioning
                    examples_hash TEXT NOT NULL,
                    num_examples INTEGER NOT NULL,

                    -- Hyperparameters (JSON)
                    hyperparams TEXT NOT NULL,

                    -- Metrics (JSON)
                    metrics TEXT NOT NULL,

                    -- Reproducibility
                    random_seed INTEGER NOT NULL,

                    -- Metadata
                    notes TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'completed'
                )
            """)

            # Indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_id
                ON experiments(model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON experiments(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_hash
                ON experiments(examples_hash)
            """)

            conn.commit()

    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        return f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def log_projection_training(
        self,
        model_provenance: Any,  # ModelProvenance from model_manager
        examples_hash: str,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        seed: int,
        num_examples: Optional[int] = None,
        notes: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Log a projection training experiment.

        Args:
            model_provenance: ModelProvenance object with model info
            examples_hash: SHA256 hash of training examples
            hyperparams: Dictionary of hyperparameters
            metrics: Dictionary of training metrics
            seed: Random seed used for training
            num_examples: Number of training examples (optional)
            notes: Optional notes about the experiment
            tags: Optional list of tags for categorization

        Returns:
            experiment_id: Unique ID for this experiment
        """
        experiment_id = self._generate_experiment_id()
        now = datetime.utcnow().isoformat()

        # Extract provenance fields
        if hasattr(model_provenance, 'to_dict'):
            prov = model_provenance.to_dict()
        elif isinstance(model_provenance, dict):
            prov = model_provenance
        else:
            prov = {
                "model_id": str(model_provenance),
                "revision": None,
                "sha256": None
            }

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, created_at, completed_at,
                    model_id, model_revision, model_sha256,
                    examples_hash, num_examples,
                    hyperparams, metrics, random_seed,
                    notes, tags, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    now,
                    now,  # completed_at = created_at for synchronous logging
                    prov.get("model_id", "unknown"),
                    prov.get("revision"),
                    prov.get("sha256"),
                    examples_hash,
                    num_examples or 0,
                    json.dumps(hyperparams),
                    json.dumps(metrics),
                    seed,
                    notes,
                    json.dumps(tags or []),
                    "completed"
                )
            )
            conn.commit()

        logger.info(f"Logged experiment {experiment_id}: model={prov.get('model_id')}, seed={seed}")
        return experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """
        Retrieve a full experiment record by ID.

        Args:
            experiment_id: The experiment ID to retrieve

        Returns:
            ExperimentRecord or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            model_id=row["model_id"],
            model_revision=row["model_revision"],
            model_sha256=row["model_sha256"],
            examples_hash=row["examples_hash"],
            num_examples=row["num_examples"],
            hyperparams=json.loads(row["hyperparams"]),
            metrics=json.loads(row["metrics"]),
            random_seed=row["random_seed"],
            notes=row["notes"],
            tags=json.loads(row["tags"]),
            status=row["status"]
        )

    def list_experiments(
        self,
        model_id: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            model_id: Filter by model ID
            tag: Filter by tag
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of experiment summaries
        """
        query = "SELECT * FROM experiments"
        params: List[Any] = []
        conditions = []

        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)

        if tag:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            record = ExperimentRecord(
                experiment_id=row["experiment_id"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                model_id=row["model_id"],
                model_revision=row["model_revision"],
                model_sha256=row["model_sha256"],
                examples_hash=row["examples_hash"],
                num_examples=row["num_examples"],
                hyperparams=json.loads(row["hyperparams"]),
                metrics=json.loads(row["metrics"]),
                random_seed=row["random_seed"],
                notes=row["notes"],
                tags=json.loads(row["tags"]),
                status=row["status"]
            )
            results.append(record.summary())

        return results

    def get_experiments_by_hash(self, examples_hash: str) -> List[ExperimentRecord]:
        """
        Get all experiments with the same training data hash.

        Useful for comparing experiments with different hyperparameters
        on the same data.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE examples_hash = ? ORDER BY created_at DESC",
                (examples_hash,)
            )
            rows = cursor.fetchall()

        return [
            ExperimentRecord(
                experiment_id=row["experiment_id"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                model_id=row["model_id"],
                model_revision=row["model_revision"],
                model_sha256=row["model_sha256"],
                examples_hash=row["examples_hash"],
                num_examples=row["num_examples"],
                hyperparams=json.loads(row["hyperparams"]),
                metrics=json.loads(row["metrics"]),
                random_seed=row["random_seed"],
                notes=row["notes"],
                tags=json.loads(row["tags"]),
                status=row["status"]
            )
            for row in rows
        ]

    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments side-by-side.

        Returns a comparison table showing differences in hyperparameters
        and metrics across experiments.
        """
        experiments = [self.get_experiment(eid) for eid in experiment_ids]
        experiments = [e for e in experiments if e is not None]

        if len(experiments) < 2:
            return {"error": "Need at least 2 experiments to compare"}

        # Collect all hyperparameter and metric keys
        all_hyperparam_keys = set()
        all_metric_keys = set()
        for exp in experiments:
            all_hyperparam_keys.update(exp.hyperparams.keys())
            all_metric_keys.update(exp.metrics.keys())

        comparison = {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "model_id": exp.model_id,
                    "random_seed": exp.random_seed,
                    "created_at": exp.created_at
                }
                for exp in experiments
            ],
            "hyperparams": {
                key: [exp.hyperparams.get(key) for exp in experiments]
                for key in sorted(all_hyperparam_keys)
            },
            "metrics": {
                key: [exp.metrics.get(key) for exp in experiments]
                for key in sorted(all_metric_keys)
            },
            "same_data": len(set(exp.examples_hash for exp in experiments)) == 1,
            "same_model": len(set(exp.model_id for exp in experiments)) == 1
        }

        return comparison

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment record."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def add_tags(self, experiment_id: str, tags: List[str]) -> bool:
        """Add tags to an experiment."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            return False

        new_tags = list(set(exp.tags + tags))

        with self._get_connection() as conn:
            conn.execute(
                "UPDATE experiments SET tags = ? WHERE experiment_id = ?",
                (json.dumps(new_tags), experiment_id)
            )
            conn.commit()

        return True

    def update_notes(self, experiment_id: str, notes: str) -> bool:
        """Update the notes for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE experiments SET notes = ? WHERE experiment_id = ?",
                (notes, experiment_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            total = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT model_id, COUNT(*) as count
                FROM experiments
                GROUP BY model_id
                ORDER BY count DESC
            """)
            by_model = [
                {"model_id": row[0], "count": row[1]}
                for row in cursor.fetchall()
            ]

            cursor = conn.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM experiments
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 30
            """)
            by_date = [
                {"date": row[0], "count": row[1]}
                for row in cursor.fetchall()
            ]

        return {
            "total_experiments": total,
            "by_model": by_model,
            "recent_by_date": by_date,
            "db_path": str(self.db_path),
            "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }

    def close(self):
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# Singleton instance
_tracker_instance: Optional[ExperimentTracker] = None


def get_experiment_tracker(db_path: Optional[Path] = None) -> ExperimentTracker:
    """Get the global ExperimentTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExperimentTracker(db_path)
    return _tracker_instance


# --- Data Hashing Utilities ---

def hash_training_data(examples: List[Dict[str, Any]]) -> str:
    """
    Compute a deterministic hash of training examples.

    This ensures that identical training data produces the same hash,
    enabling tracking of which experiments used the same data.

    Args:
        examples: List of training example dictionaries

    Returns:
        SHA256 hash of the serialized data
    """
    # Sort examples by a stable key to ensure deterministic ordering
    sorted_examples = sorted(
        examples,
        key=lambda x: (x.get("text", ""), x.get("agency", 0), x.get("fairness", 0), x.get("belonging", 0))
    )

    # Serialize to JSON with sorted keys
    serialized = json.dumps(sorted_examples, sort_keys=True, ensure_ascii=True)

    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_embeddings(embeddings: Any) -> str:
    """
    Compute a hash of an embedding matrix.

    Args:
        embeddings: numpy array of embeddings

    Returns:
        SHA256 hash of the embedding bytes
    """
    import numpy as np
    if isinstance(embeddings, np.ndarray):
        return hashlib.sha256(embeddings.tobytes()).hexdigest()
    return hashlib.sha256(str(embeddings).encode()).hexdigest()
