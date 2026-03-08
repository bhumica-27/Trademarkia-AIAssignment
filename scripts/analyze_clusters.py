import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clustering import load_artifacts
from app.config import CLUSTER_ARTIFACTS_DIR


def main():
    print("Loading artefacts …")
    art = load_artifacts()
    soft = art["soft_assignments"]
    doc_ids = art["doc_ids"]
    categories = art["categories"]
    n_clusters = art["n_clusters"]
    k_search = art["k_search"]

    N, K = soft.shape
    hard = np.argmax(soft, axis=1)

    print(f"\nCorpus: {N} documents, {K} clusters")
    print(f"Best K by BIC: {k_search['best_k']}")

    # ── 1. BIC / AIC curve ────────────────────────────────────────────
    
    print("BIC / AIC scores across K values")
    
    for k_val, bic, aic in zip(k_search["k_values"], k_search["bic"], k_search["aic"]):
        marker = " ◄ best" if k_val == k_search["best_k"] else ""
        print(f"  K={k_val:2d}  BIC={bic:>12,.0f}  AIC={aic:>12,.0f}{marker}")

    # ── 2. Per-cluster breakdown ──────────────────────────────────────
    
    print("Per-cluster analysis")
    

    for cid in range(K):
        mask = hard == cid
        size = mask.sum()
        if size == 0:
            print(f"\n  Cluster {cid}: EMPTY")
            continue

        cluster_cats = [categories[i] for i, m in enumerate(mask) if m]
        cat_dist = Counter(cluster_cats)
        top = cat_dist.most_common(5)

        cluster_probs = soft[mask]
        max_probs = cluster_probs.max(axis=1)

        # Purity: fraction of dominant category
        dominant_cat_count = top[0][1]
        purity = dominant_cat_count / size

        # Entropy of cluster-membership probabilities (lower = more confident)
        avg_entropy = float(-np.mean(np.sum(
            cluster_probs * np.log(cluster_probs + 1e-12), axis=1
        )))

        print(f"\n  Cluster {cid} — {size} documents")
        print(f"    Purity: {purity:.2%}  (dominant: '{top[0][0]}')")
        print(f"    Avg entropy of membership: {avg_entropy:.3f}")
        print(f"    Mean max-prob: {max_probs.mean():.3f}, "
              f"min max-prob: {max_probs.min():.3f}")
        print(f"    Category breakdown:")
        for cat, cnt in top:
            print(f"      {cat}: {cnt} ({cnt/size:.0%})")

    # ── 3. Boundary / uncertain documents ─────────────────────────────
    
    print("Most uncertain documents (lowest max-probability)")
    
    max_probs_all = soft.max(axis=1)
    uncertain_order = np.argsort(max_probs_all)[:20]

    for rank, idx in enumerate(uncertain_order, 1):
        top3 = np.argsort(soft[idx])[::-1][:3]
        top3_probs = [round(float(soft[idx][c]), 3) for c in top3]
        print(f"  {rank:2d}. {doc_ids[idx]} (cat: {categories[idx]})")
        print(f"      Top clusters: {top3.tolist()} → {top3_probs}")

    # ── 4. Category-to-cluster mapping ────────────────────────────────
    
    print("Category - Cluster mapping (which clusters each category lands in)")
    
    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        cluster_counts = Counter(hard[i] for i in idxs)
        top_clusters = cluster_counts.most_common(3)
        print(f"  {cat}:")
        for cid, cnt in top_clusters:
            pct = cnt / len(idxs)
            print(f"    Cluster {cid}: {cnt}/{len(idxs)} ({pct:.0%})")

    # ── 5. Cross-cluster overlap ──────────────────────────────────────
    
    print("Documents with significant membership in multiple clusters")
    print("(second-highest probability > 0.15)")
    
    count = 0
    for idx in range(N):
        sorted_probs = np.sort(soft[idx])[::-1]
        if sorted_probs[1] > 0.15:
            top2 = np.argsort(soft[idx])[::-1][:2]
            count += 1
            if count <= 15:  # Show first 15
                print(f"  {doc_ids[idx]} (cat: {categories[idx]})")
                print(f"    Cluster {top2[0]}: {soft[idx][top2[0]]:.3f}, "
                      f"Cluster {top2[1]}: {soft[idx][top2[1]]:.3f}")
    print(f"\n  Total multi-membership docs (2nd prob > 0.15): {count}/{N} ({count/N:.1%})")

    
    print("Analysis complete.")
    


if __name__ == "__main__":
    main()
