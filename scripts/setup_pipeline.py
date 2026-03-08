import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_loader import load_documents
from app.embeddings import embed_texts, index_documents, get_or_create_collection
from app.clustering import (
    reduce_dimensions,
    select_k,
    fit_gmm,
    get_soft_assignments,
    save_artifacts,
)
from app.config import CLUSTER_ARTIFACTS_DIR


def main():
    t0 = time.time()

    # ── Step 1: Load & clean ──────────────────────────────────────────
    
    print("STEP 1 - Loading and cleaning documents …")
    
    docs = load_documents()
    print(f"  Loaded {len(docs)} documents after cleaning.")

    categories = sorted(set(d["category"] for d in docs))
    print(f"  Categories ({len(categories)}): {categories}")

    # ── Step 2: Embed ─────────────────────────────────────────────────
    
    print("STEP 2 - Generating embeddings …")
    
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    
    print("STEP 3 - Indexing in ChromaDB …")
    
    index_documents(docs, embeddings)

    
    print("STEP 4 - PCA dimensionality reduction …")
    
    pca, reduced = reduce_dimensions(embeddings, n_components=50)

    # ── Step 5: Select optimal K ──────────────────────────────────────
    
    print("STEP 5 - Searching for optimal K (BIC) …")
    
    k_results = select_k(reduced)
    best_k = k_results["best_k"]

    # ── Step 6: Fit final GMM ─────────────────────────────────────────
    
    print(f"STEP 6 - Fitting GMM with K={best_k} …")
    
    gmm = fit_gmm(reduced, best_k)
    soft = get_soft_assignments(gmm, reduced)
    print(f"  Soft assignments shape: {soft.shape}")

    # ── Step 7: Quick cluster analysis ────────────────────────────────
    
    print("STEP 7 - Cluster analysis …")
    

    hard_labels = np.argmax(soft, axis=1)
    doc_categories = [d["category"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    for cluster_id in range(best_k):
        mask = hard_labels == cluster_id
        cluster_cats = [doc_categories[i] for i in range(len(docs)) if mask[i]]
        size = mask.sum()

        if size == 0:
            print(f"\n  Cluster {cluster_id}: EMPTY")
            continue

        # Category distribution within this cluster
        from collections import Counter
        cat_counts = Counter(cluster_cats)
        top_cats = cat_counts.most_common(5)

        # Boundary documents: those with low max-probability (high uncertainty)
        cluster_probs = soft[mask]
        max_probs = cluster_probs.max(axis=1)
        uncertain_indices = np.where(mask)[0][np.argsort(max_probs)[:3]]

        print(f"\n  Cluster {cluster_id} ({size} docs):")
        print(f"    Top categories: {top_cats}")
        print(f"    Mean max-probability: {max_probs.mean():.3f}")
        print(f"    Most uncertain docs (boundary cases):")
        for idx in uncertain_indices:
            probs_sorted = np.argsort(soft[idx])[::-1][:3]
            print(f"      {doc_ids[idx]}: "
                  f"clusters {probs_sorted.tolist()} with probs "
                  f"{[round(float(soft[idx][c]), 3) for c in probs_sorted]}")

    # ── Step 8: Update ChromaDB metadata with cluster info ────────────
    
    print("STEP 8 - Updating ChromaDB with cluster assignments …")
    
    collection = get_or_create_collection()
    batch = 500
    for start in range(0, len(docs), batch):
        end = min(start + batch, len(docs))
        batch_ids = doc_ids[start:end]
        batch_metas = []
        for i in range(start, end):
            meta = {
                "category": doc_categories[i],
                "cluster_id": int(hard_labels[i]),
                "text_preview": docs[i]["text"][:500],
            }
            batch_metas.append(meta)
        collection.update(ids=batch_ids, metadatas=batch_metas)
    print("  Done.")

    # ── Step 9: Save artefacts ────────────────────────────────────────
    print("STEP 9 - Saving artefacts …")
    save_artifacts(
        pca=pca,
        gmm=gmm,
        soft_assignments=soft,
        doc_ids=doc_ids,
        categories=doc_categories,
        k_search_results=k_results,
    )

    elapsed = time.time() - t0
    print(f"Pipeline complete in {elapsed:.1f}s.")
    print(f"Start the server with:")
    print(f"  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    


if __name__ == "__main__":
    main()
