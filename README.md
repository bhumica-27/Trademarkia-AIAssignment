# 20 Newsgroups - Semantic Search System

A lightweight semantic search engine over the 20 Newsgroups dataset with **fuzzy clustering**, a **custom semantic cache**, and a **FastAPI** service.

## Architecture

```
Data/mini_newsgroups/        <-- 2,000 cleaned newsgroup posts
        |
  data_loader.py             <-- Parse, clean, extract body text
        |
  embeddings.py              <-- sentence-transformers/all-MiniLM-L6-v2 -> 384-dim
        |
  ChromaDB (persistent)      <-- Vector store with HNSW index + metadata
        |
  clustering.py              <-- PCA(50) -> GMM(K) -> soft assignments P(cluster|doc)
        |
  semantic_cache.py          <-- Cluster-bucketed, cosine-similarity cache
        |
  main.py                    <-- FastAPI service (POST /query, GET /cache/stats, DELETE /cache)
```

## Quick Start

### 1. Create virtual environment

```bash
python -m venv venv

# for Windows
venv\Scripts\activate

# and for Linux/macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the setup pipeline (one-time)

This loads the data, generates embeddings, indexes into ChromaDB, runs clustering, and saves artefacts:

```bash
python -m scripts.setup_pipeline
```

### 4. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The interactive docs are at **http://localhost:8000/docs**.

### 5. (Optional) Run cluster analysis

```bash
python -m scripts.analyze_clusters
```

---

## API Endpoints

### POST /query

```json
{
  "query": "What are the best graphics cards for 3D rendering?"
}
```

**Response:**

```json
{
  "query": "What are the best graphics cards for 3D rendering?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": {
    "top_documents": ["..."],
    "cluster_distribution": {"3": 0.72, "7": 0.15}
  },
  "dominant_cluster": 3
}
```

On a cache hit (semantically similar query seen before):

```json
{
  "query": "GPU recommendations for 3D work?",
  "cache_hit": true,
  "matched_query": "What are the best graphics cards for 3D rendering?",
  "similarity_score": 0.91,
  "result": {},
  "dominant_cluster": 3
}
```

### GET /cache/stats

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE /cache

Flushes the cache entirely and resets all stats.

---

## Design Decisions

### Embedding Model - all-MiniLM-L6-v2

384-dim, ~80 MB, trained on 1B+ sentence pairs. Fast enough for CPU inference on 2,000 documents, expressive enough to separate 20 topic categories. Larger models (all-mpnet-base-v2) give marginal quality gains at 3x the cost.

### Vector Store - ChromaDB

Embedded/persistent mode - zero infrastructure. SQLite + HNSW under the hood. Metadata filtering by cluster/category is used during search. FAISS would be faster for raw ANN but lacks persistence and metadata support out of the box.

### Fuzzy Clustering - Gaussian Mixture Model

GMM gives P(cluster | document) natively - exactly the distribution required. K-Means only produces hard labels. LDA operates on bag-of-words, not embeddings. PCA reduces from 384 to 50 dims before GMM fitting to avoid ill-conditioning (more covariance params than data points).

### Optimal K - BIC minimisation

Swept K in [10, 30] and selected the value that minimises BIC. The optimal K typically falls between 12-18, fewer than the 20 nominal labels, because some newsgroup categories are semantically merged (e.g. talk.politics.*).

### Semantic Cache - Cluster-bucketed cosine similarity

Built from scratch - no Redis/Memcached. The cache is a two-level hash map: cluster_id -> list[CacheEntry]. On lookup, only the dominant cluster bucket (plus its top-2 neighbours) is scanned. This makes cache lookup sub-linear in the number of cached queries.

### The Tunable Knob - Similarity Threshold

The single most important parameter. See the detailed exploration in app/semantic_cache.py. In brief:
- **0.95**: near-exact match. Safe but low recall.
- **0.90**: good paraphrase detection.
- **0.85** (default): broad semantic matching. This is where the models geometry becomes visible.
- **0.80**: aggressive. Related queries merge - cache becomes topic routing.

---

## Docker

```bash
# Build and run
docker compose up --build

# Or manually
docker build -t newsgroups-search .
docker run -p 8000:8000 newsgroups-search
```

---

## Project Structure

```
app/
  __init__.py
  config.py             
  data_loader.py        
  embeddings.py         
  clustering.py         
  semantic_cache.py     
  main.py               
scripts/
  setup_pipeline.py     
  analyze_clusters.py   
Data/                  
requirements.txt
Dockerfile
docker-compose.yml
README.md
```
