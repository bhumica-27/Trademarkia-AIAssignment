"""
FastAPI service - the live API endpoint for the semantic search system.

Endpoints:
    POST   /query        – semantic search with cache
    GET    /cache/stats   – cache statistics
    DELETE /cache         – flush the cache
"""

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from app.config import DEFAULT_SIMILARITY_THRESHOLD, MAX_CACHE_ENTRIES_PER_CLUSTER
from app.embeddings import embed_single, search as vector_search
from app.clustering import load_artifacts, transform_query_to_cluster
from app.semantic_cache import SemanticCache


# ── Pydantic models ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: dict
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


# ── Application state ────────────────────────────────────────────────────

class AppState:
    """Holds all runtime state initialised once at startup."""
    cache: SemanticCache
    pca_components: np.ndarray
    pca_mean: np.ndarray
    gmm: object  # sklearn GaussianMixture
    n_clusters: int


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading clustering artefacts …")
    artifacts = load_artifacts()
    state.pca_components = artifacts["pca_components"]
    state.pca_mean = artifacts["pca_mean"]
    state.gmm = artifacts["gmm"]
    state.n_clusters = artifacts["n_clusters"]

    state.cache = SemanticCache(
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        max_entries_per_cluster=MAX_CACHE_ENTRIES_PER_CLUSTER,
    )
    print(f"Ready, {state.n_clusters} clusters loaded.")
    yield


app = FastAPI(
    title="20 Newsgroups Semantic Search",
    description="Fuzzy-clustered semantic search with a custom cache layer.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    query_text = req.query.strip()

    # Step 1 - embed
    query_emb = embed_single(query_text)

    # Step 2 - cluster membership
    cluster_probs, dominant_cluster = transform_query_to_cluster(
        query_emb,
        state.pca_components,
        state.pca_mean,
        state.gmm,
    )

    # Step 3 - cache lookup
    hit = state.cache.lookup(
        query_embedding=query_emb,
        dominant_cluster=dominant_cluster,
        cluster_probs=cluster_probs,
    )

    if hit is not None:
        entry, sim_score = hit
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=entry.query_text,
            similarity_score=round(sim_score, 4),
            result=entry.result,
            dominant_cluster=dominant_cluster,
        )

    # Step 4 - cache miss: run vector search
    search_results = vector_search(query_emb, n_results=5)

    result_payload = {
        "top_documents": [
            {
                "doc_id": search_results["ids"][0][i],
                "distance": round(search_results["distances"][0][i], 4),
                "category": search_results["metadatas"][0][i].get("category", ""),
                "text_preview": search_results["metadatas"][0][i].get("text_preview", "")[:300],
            }
            for i in range(len(search_results["ids"][0]))
        ],
        "cluster_distribution": {
            str(k): round(float(v), 4)
            for k, v in enumerate(cluster_probs)
            if v > 0.01  # only show clusters with >1 % membership
        },
    }

    # Step 5 - store in cache
    state.cache.store(
        query_text=query_text,
        query_embedding=query_emb,
        result=result_payload,
        dominant_cluster=dominant_cluster,
    )

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_payload,
        dominant_cluster=dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Return current cache state."""
    return CacheStatsResponse(**state.cache.stats())


@app.delete("/cache")
async def cache_flush():
    """Flush the cache entirely and reset all stats."""
    state.cache.flush()
    return {"message": "Cache flushed successfully."}
