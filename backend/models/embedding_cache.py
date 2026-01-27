"""
Embedding Cache

SQLite-based caching for embeddings to avoid recomputation.
Supports multiple models and layers.

Performance optimizations:
- WAL mode for better concurrent reads
- Connection pooling via thread-local storage
- Batch queries using SQL IN clause
- Optional zlib compression for embeddings
"""

import sqlite3
import numpy as np
import json
import hashlib
import zlib
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CachedEmbedding:
    """A cached embedding result."""
    text_hash: str
    text: str
    model_id: str
    layer: int
    embedding: np.ndarray
    created_at: str

    def to_dict(self) -> dict:
        return {
            "text_hash": self.text_hash,
            "text": self.text,
            "model_id": self.model_id,
            "layer": self.layer,
            "embedding_dim": len(self.embedding),
            "created_at": self.created_at
        }


class EmbeddingCache:
    """
    SQLite-based embedding cache.

    Stores embeddings by (text_hash, model_id, layer) key.
    Supports FAISS index for similarity search if available.

    Performance features:
    - WAL mode for concurrent reads
    - Connection pooling via thread-local storage
    - Batch queries with SQL IN clause
    - Optional zlib compression (configurable)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        enable_compression: bool = False,
        compression_level: int = 6
    ):
        self.db_path = Path(db_path) if db_path else Path("./data/embedding_cache.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Compression settings
        self.enable_compression = enable_compression
        self.compression_level = compression_level

        # Thread-local connection pool
        self._local = threading.local()

        self._init_db()
        self._stats = {"hits": 0, "misses": 0}

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection (connection pooling)."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrent reads
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=10000")
            self._local.conn.execute("PRAGMA temp_store=MEMORY")

        try:
            yield self._local.conn
        except Exception:
            # On error, close and clear the connection so a new one is created next time
            self._local.conn.close()
            self._local.conn = None
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (text_hash, model_id, layer)
                )
            """)

            # Index for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_layer
                ON embeddings(model_id, layer)
            """)

            # Index for batch lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_hash
                ON embeddings(text_hash)
            """)

            conn.commit()

        logger.info(f"Embedding cache initialized at {self.db_path} (compression={'on' if self.enable_compression else 'off'})")

    def _text_hash(self, text: str) -> str:
        """Compute hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _compress_embedding(self, embedding: np.ndarray) -> Tuple[bytes, bool]:
        """Compress embedding bytes if compression is enabled."""
        embedding_bytes = embedding.astype(np.float32).tobytes()
        if self.enable_compression:
            compressed = zlib.compress(embedding_bytes, level=self.compression_level)
            return compressed, True
        return embedding_bytes, False

    def _decompress_embedding(self, data: bytes, dim: int, compressed: bool) -> np.ndarray:
        """Decompress embedding bytes if needed."""
        if compressed:
            data = zlib.decompress(data)
        return np.frombuffer(data, dtype=np.float32).reshape(dim)

    def get(
        self,
        text: str,
        model_id: str,
        layer: int = -1
    ) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.

        Returns None if not cached.
        """
        text_hash = self._text_hash(text)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT embedding, embedding_dim, compressed FROM embeddings
                WHERE text_hash = ? AND model_id = ? AND layer = ?
                """,
                (text_hash, model_id, layer)
            )
            row = cursor.fetchone()

        if row:
            self._stats["hits"] += 1
            embedding_bytes, dim, compressed = row
            return self._decompress_embedding(embedding_bytes, dim, bool(compressed))
        else:
            self._stats["misses"] += 1
            return None

    def get_batch(
        self,
        texts: List[str],
        model_id: str,
        layer: int = -1
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get cached embeddings for multiple texts.

        Uses SQL IN clause for efficient batch retrieval instead of
        iterating through individual queries.

        Returns:
            - List of embeddings (None for misses)
            - List of indices that need computation
        """
        if not texts:
            return [], []

        results = [None] * len(texts)
        missing_indices = []

        # Build hash -> index mapping
        hash_to_indices: Dict[str, List[int]] = {}
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash not in hash_to_indices:
                hash_to_indices[text_hash] = []
            hash_to_indices[text_hash].append(i)

        # Build IN clause with placeholders
        hashes = list(hash_to_indices.keys())
        placeholders = ",".join("?" * len(hashes))

        with self._get_connection() as conn:
            # Single query with IN clause for all hashes
            cursor = conn.execute(
                f"""
                SELECT text_hash, embedding, embedding_dim, compressed FROM embeddings
                WHERE text_hash IN ({placeholders})
                AND model_id = ? AND layer = ?
                """,
                (*hashes, model_id, layer)
            )
            rows = cursor.fetchall()

        # Map results back to indices
        found_hashes = set()
        for text_hash, embedding_bytes, dim, compressed in rows:
            found_hashes.add(text_hash)
            embedding = self._decompress_embedding(embedding_bytes, dim, bool(compressed))
            # Fill in all indices that have this hash
            for idx in hash_to_indices[text_hash]:
                results[idx] = embedding
                self._stats["hits"] += 1

        # Find missing indices
        for text_hash, indices in hash_to_indices.items():
            if text_hash not in found_hashes:
                for idx in indices:
                    missing_indices.append(idx)
                    self._stats["misses"] += 1

        # Sort missing indices for consistent ordering
        missing_indices.sort()

        return results, missing_indices

    def put(
        self,
        text: str,
        model_id: str,
        layer: int,
        embedding: np.ndarray
    ):
        """Cache an embedding."""
        text_hash = self._text_hash(text)
        embedding_bytes, compressed = self._compress_embedding(embedding)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (text_hash, model_id, layer, text, embedding, embedding_dim, compressed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    text_hash,
                    model_id,
                    layer,
                    text,
                    embedding_bytes,
                    len(embedding),
                    int(compressed),
                    datetime.now().isoformat()
                )
            )
            conn.commit()

    def put_batch(
        self,
        texts: List[str],
        model_id: str,
        layer: int,
        embeddings: List[np.ndarray]
    ):
        """Cache multiple embeddings using batch insert."""
        now = datetime.now().isoformat()

        # Prepare all data for batch insert
        batch_data = []
        for text, embedding in zip(texts, embeddings):
            text_hash = self._text_hash(text)
            embedding_bytes, compressed = self._compress_embedding(embedding)
            batch_data.append((
                text_hash,
                model_id,
                layer,
                text,
                embedding_bytes,
                len(embedding),
                int(compressed),
                now
            ))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (text_hash, model_id, layer, text, embedding, embedding_dim, compressed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch_data
            )
            conn.commit()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT model_id, layer, COUNT(*) as count
                FROM embeddings
                GROUP BY model_id, layer
            """)
            by_model = [
                {"model_id": r[0], "layer": r[1], "count": r[2]}
                for r in cursor.fetchall()
            ]

            # Get compression stats
            cursor = conn.execute("""
                SELECT compressed, COUNT(*) as count
                FROM embeddings
                GROUP BY compressed
            """)
            compression_stats = {
                "compressed": 0,
                "uncompressed": 0
            }
            for compressed, count in cursor.fetchall():
                if compressed:
                    compression_stats["compressed"] = count
                else:
                    compression_stats["uncompressed"] = count

        hit_rate = (
            self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
            if (self._stats["hits"] + self._stats["misses"]) > 0
            else 0
        )

        return {
            "total_cached": total,
            "by_model": by_model,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "compression_enabled": self.enable_compression,
            "compression_stats": compression_stats,
            "db_path": str(self.db_path),
            "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }

    def clear(self, model_id: Optional[str] = None, layer: Optional[int] = None):
        """
        Clear cache entries.

        If model_id/layer specified, only clear matching entries.
        """
        with self._get_connection() as conn:
            if model_id is None and layer is None:
                conn.execute("DELETE FROM embeddings")
            elif layer is None:
                conn.execute("DELETE FROM embeddings WHERE model_id = ?", (model_id,))
            elif model_id is None:
                conn.execute("DELETE FROM embeddings WHERE layer = ?", (layer,))
            else:
                conn.execute(
                    "DELETE FROM embeddings WHERE model_id = ? AND layer = ?",
                    (model_id, layer)
                )
            conn.commit()

        logger.info(f"Cache cleared (model_id={model_id}, layer={layer})")

    def get_all_texts(self, model_id: str, layer: int = -1) -> List[str]:
        """Get all texts cached for a model/layer."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT text FROM embeddings WHERE model_id = ? AND layer = ?",
                (model_id, layer)
            )
            return [row[0] for row in cursor.fetchall()]

    def get_all_embeddings(
        self,
        model_id: str,
        layer: int = -1
    ) -> Tuple[List[str], np.ndarray]:
        """Get all embeddings for a model/layer as a matrix."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT text, embedding, embedding_dim, compressed FROM embeddings
                WHERE model_id = ? AND layer = ?
                """,
                (model_id, layer)
            )
            rows = cursor.fetchall()

        if not rows:
            return [], np.array([])

        texts = []
        embeddings = []

        for text, embedding_bytes, dim, compressed in rows:
            texts.append(text)
            embeddings.append(
                self._decompress_embedding(embedding_bytes, dim, bool(compressed))
            )

        return texts, np.array(embeddings)

    def close(self):
        """Close the thread-local connection if open."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# Singleton instance
_cache_instance: Optional[EmbeddingCache] = None


def get_embedding_cache(db_path: Optional[Path] = None) -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = EmbeddingCache(db_path)
    return _cache_instance
