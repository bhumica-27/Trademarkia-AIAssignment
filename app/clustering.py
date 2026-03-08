import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from app.config import MIN_K, MAX_K, CLUSTER_ARTIFACTS_DIR


def _ensure_artifacts_dir() -> Path:
    CLUSTER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return CLUSTER_ARTIFACTS_DIR


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 50) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} components explain {explained:.1%} of variance.")
    return pca, reduced


def select_k(reduced: np.ndarray, min_k: int = MIN_K, max_k: int = MAX_K) -> Dict:
    results = {"k_values": [], "bic": [], "aic": []}
    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",  # diagonal cov keeps params manageable
            n_init=3,
            random_state=42,
        )
        gmm.fit(reduced)
        results["k_values"].append(k)
        results["bic"].append(gmm.bic(reduced))
        results["aic"].append(gmm.aic(reduced))
        print(f"  K={k:2d}  BIC={gmm.bic(reduced):,.0f}  AIC={gmm.aic(reduced):,.0f}")

    best_idx = int(np.argmin(results["bic"]))
    results["best_k"] = results["k_values"][best_idx]
    print(f"\n Best K by BIC = {results['best_k']}")
    return results


def fit_gmm(reduced: np.ndarray, k: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=5,
        random_state=42,
    )
    gmm.fit(reduced)
    return gmm


def get_soft_assignments(gmm: GaussianMixture, reduced: np.ndarray) -> np.ndarray:
    return gmm.predict_proba(reduced)


def save_artifacts(
    pca: PCA,
    gmm: GaussianMixture,
    soft_assignments: np.ndarray,
    doc_ids: list,
    categories: list,
    k_search_results: dict,
) -> None:
    out = _ensure_artifacts_dir()

    np.save(out / "pca_components.npy", pca.components_)
    np.save(out / "pca_mean.npy", pca.mean_)
    np.save(out / "soft_assignments.npy", soft_assignments)

    # GMM parameters
    np.save(out / "gmm_means.npy", gmm.means_)
    np.save(out / "gmm_covariances.npy", gmm.covariances_)
    np.save(out / "gmm_weights.npy", gmm.weights_)

    # Metadata
    with open(out / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    with open(out / "categories.json", "w") as f:
        json.dump(categories, f)
    with open(out / "k_search.json", "w") as f:
        json.dump(k_search_results, f)

    # Save PCA n_components for reconstruction
    with open(out / "pca_params.json", "w") as f:
        json.dump({"n_components": pca.n_components_}, f)



def load_artifacts() -> Dict:
    out = CLUSTER_ARTIFACTS_DIR

    soft_assignments = np.load(out / "soft_assignments.npy")
    gmm_means = np.load(out / "gmm_means.npy")
    gmm_covariances = np.load(out / "gmm_covariances.npy")
    gmm_weights = np.load(out / "gmm_weights.npy")

    with open(out / "pca_params.json") as f:
        pca_params = json.load(f)

    pca_components = np.load(out / "pca_components.npy")
    pca_mean = np.load(out / "pca_mean.npy")

    with open(out / "doc_ids.json") as f:
        doc_ids = json.load(f)
    with open(out / "categories.json") as f:
        categories = json.load(f)
    with open(out / "k_search.json") as f:
        k_search = json.load(f)

    k = gmm_means.shape[0]

    # Reconstruct GMM
    n_components_pca = pca_params["n_components"]
    gmm = GaussianMixture(n_components=k, covariance_type="diag")
    gmm.means_ = gmm_means
    gmm.covariances_ = gmm_covariances
    gmm.weights_ = gmm_weights
    gmm.precisions_cholesky_ = np.sqrt(1.0 / gmm_covariances)

    return {
        "pca_components": pca_components,
        "pca_mean": pca_mean,
        "gmm": gmm,
        "soft_assignments": soft_assignments,
        "doc_ids": doc_ids,
        "categories": categories,
        "k_search": k_search,
        "n_clusters": k,
    }


def transform_query_to_cluster(
    query_embedding: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    gmm: GaussianMixture,
) -> Tuple[np.ndarray, int]:
    reduced = (query_embedding - pca_mean) @ pca_components.T
    reduced = reduced.reshape(1, -1)
    probs = gmm.predict_proba(reduced)[0]
    dominant = int(np.argmax(probs))
    return probs, dominant
