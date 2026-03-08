import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    result: dict
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_entries_per_cluster: int = 200,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries_per_cluster = max_entries_per_cluster

        # cluster_id → list[CacheEntry], sorted by timestamp (newest last)
        self._buckets: Dict[int, List[CacheEntry]] = {}

        # Stats
        self._hit_count = 0
        self._miss_count = 0

        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding: np.ndarray,
        dominant_cluster: int,
        cluster_probs: Optional[np.ndarray] = None,
        top_k_clusters: int = 3,
    ) -> Optional[Tuple[CacheEntry, float]]:
        # Determine which cluster buckets to search.
        # Primary: the dominant cluster.
        # Secondary: top-k clusters by GMM probability - this catches edge
        # cases where a query sits at the boundary of two clusters.
        clusters_to_search = [dominant_cluster]
        if cluster_probs is not None:
            top_indices = np.argsort(cluster_probs)[::-1][:top_k_clusters]
            for idx in top_indices:
                if int(idx) not in clusters_to_search:
                    clusters_to_search.append(int(idx))

        best_entry: Optional[CacheEntry] = None
        best_sim = -1.0

        with self._lock:
            for cid in clusters_to_search:
                bucket = self._buckets.get(cid, [])
                for entry in bucket:
                    sim = self._cosine_similarity(query_embedding, entry.query_embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_entry is not None and best_sim >= self.similarity_threshold:
                self._hit_count += 1
                # Update timestamp (LRU touch)
                best_entry.timestamp = time.time()
                return best_entry, float(best_sim)

            self._miss_count += 1
            return None

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: dict,
        dominant_cluster: int,
    ) -> None:
        """Insert a new entry into the cache."""
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding,
            result=result,
            dominant_cluster=dominant_cluster,
        )
        with self._lock:
            bucket = self._buckets.setdefault(dominant_cluster, [])
            bucket.append(entry)

            # LRU eviction if bucket overflows
            if len(bucket) > self.max_entries_per_cluster:
                # Remove the entry with the smallest (oldest) timestamp
                bucket.sort(key=lambda e: e.timestamp)
                bucket.pop(0)

    def flush(self) -> None:
        """Clear all cached entries and reset stats."""
        with self._lock:
            self._buckets.clear()
            self._hit_count = 0
            self._miss_count = 0

    def stats(self) -> dict:
        """Return current cache statistics."""
        with self._lock:
            total = sum(len(b) for b in self._buckets.values())
            total_queries = self._hit_count + self._miss_count
            return {
                "total_entries": total,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total_queries, 3) if total_queries > 0 else 0.0,
            }

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.
        Since our embeddings are L2-normalised, this is just the dot product.
        """
        return float(np.dot(a, b))
