#!/usr/bin/env python3
"""Low-dimensional embeddings and clustering for clump properties."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from feature_utils import load_features

DEFAULT_FEATURES = [
    "volume",
    "mass",
    "area",
    "cell_count",
    "velocity_std",
    "velocity_mean",
]


def build_matrix(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([feature_dict[k] for k in feature_dict]).T


def fit_embedding(matrix: np.ndarray, method: str, n_neighbors: int, min_dist: float, random_state: int):
    if method == "tsne":
        embedder = TSNE(n_components=2, perplexity=n_neighbors, random_state=random_state, init="pca")
        return embedder.fit_transform(matrix)
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("UMAP requested but umap-learn is not installed. `pip install umap-learn`.") from exc
        embedder = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        return embedder.fit_transform(matrix)
    else:
        raise ValueError(f"Unknown embedding method {method}")


def fit_clustering(embedding: np.ndarray, method: str, k: int, eps: float, min_samples: int) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        return model.fit_predict(embedding)
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        return model.fit_predict(embedding)
    elif method == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("HDBSCAN requested but hdbscan is not installed. `pip install hdbscan`.") from exc
        model = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        return model.fit_predict(embedding)
    else:
        raise ValueError(f"Unknown clustering method {method}")


def scatter_with_labels(embedding: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    unique = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for idx, lab in enumerate(unique):
        mask = labels == lab
        color = "lightgray" if lab == -1 else cmap(idx)
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.7, label=str(lab), color=color)
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.set_title("Embedded clusters")
    if len(unique) <= 20:
        ax.legend(markerscale=2, fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def scatter_colored_by_feature(
    embedding: np.ndarray, feature_dict: Dict[str, np.ndarray], feature: str, out_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    values = feature_dict[feature]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=values, cmap="viridis", s=6, alpha=0.7)
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    ax.set_title(f"Embedding colored by {feature}")
    fig.colorbar(sc, ax=ax, label=feature)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding + clustering for clump features.")
    parser.add_argument("--input-root", default="clump_out/clumpn10240")
    parser.add_argument("--output-dir", default="clump_out/clumpn10240/embedding")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--embed", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--cluster", choices=["kmeans", "dbscan", "hdbscan"], default="kmeans")
    parser.add_argument("--n-components", type=int, default=2, help="Reserved for future use.")
    parser.add_argument("--neighbors", type=int, default=30, help="Perplexity (t-SNE) or neighbors (UMAP).")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--kmeans-k", type=int, default=6, help="Number of clusters for k-means.")
    parser.add_argument("--dbscan-eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=10,
                        help="Optional PCA preprocessing before embedding (0 to disable).")
    parser.add_argument(
        "--sample", type=int, default=0, help="Randomly sample this many clumps (0 = use all)."
    )
    parser.add_argument(
        "--sample-weight",
        choices=["uniform", "volume", "mass", "log_volume", "log_mass"],
        default="uniform",
        help=(
            "Sampling weights (only used when --sample > 0): uniform, sqrt(volume), sqrt(mass), "
            "log(volume), or log(mass). Log weighting uses log(1 + value) to gently favor large objects."
        ),
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_root, "**", "clumps_master.npz"), recursive=True))
    if not files:
        raise SystemExit(f"No clumps_master.npz files under {args.input_root}")

    try:
        feature_dict = load_features(files, args.features)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    feature_names = list(feature_dict.keys())
    matrix = build_matrix(feature_dict)

    rng = np.random.default_rng(args.random_state)
    if args.sample > 0 and matrix.shape[0] > args.sample:
        weights = None
        weight_key = None
        transform = None
        if args.sample_weight in {"volume", "log_volume"} and "volume" in feature_dict:
            weight_key = "volume"
        elif args.sample_weight in {"mass", "log_mass"} and "mass" in feature_dict:
            weight_key = "mass"
        if weight_key is not None:
            transform = "log" if "log_" in args.sample_weight else "sqrt"
            weights = np.asarray(feature_dict[weight_key], dtype=np.float64)
            clipped = np.clip(weights, 0, None)
            if transform == "sqrt":
                weights = np.sqrt(clipped)
            else:
                weights = np.log1p(clipped)
        if weights is not None:
            if weights.sum() <= 0:
                weights = None
            else:
                weights = weights / weights.sum()
        idx = rng.choice(matrix.shape[0], size=args.sample, replace=False, p=weights)
        matrix = matrix[idx]
        for key in feature_dict:
            feature_dict[key] = feature_dict[key][idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    if 0 < args.pca_components < X.shape[1]:
        pca = PCA(n_components=args.pca_components, random_state=args.random_state)
        X_red = pca.fit_transform(X)
    else:
        X_red = X

    embedding = fit_embedding(X_red, args.embed, args.neighbors, args.min_dist, args.random_state)
    labels = fit_clustering(embedding, args.cluster, args.kmeans_k, args.dbscan_eps, args.min_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "embedding_results.npz"),
        embedding=embedding,
        labels=labels,
        features=feature_names,
    )

    scatter_with_labels(
        embedding, labels, os.path.join(args.output_dir, f"{args.embed}_{args.cluster}_clusters.png")
    )

    for feat in feature_names:
        scatter_colored_by_feature(
            embedding,
            feature_dict,
            feat,
            os.path.join(args.output_dir, f"{args.embed}_colored_{feat}.png"),
        )

    print(f"Saved embedding and clustering outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
