#!/usr/bin/env python3
"""Principal component pipeline for clump properties."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "volume",
    "vx_mean",
    "vx_std",
    "rho_mean",
    "rho_std",
    "pressure_mean",
    "pressure_std",
]


def _flatten_feature(name: str, value: np.ndarray) -> np.ndarray:
    if value.ndim == 1:
        return value
    if name == "principal_axes_lengths":
        return np.max(value, axis=1)
    if name == "axis_ratios":
        return value[:, 0]
    raise ValueError(f"Cannot flatten feature {name} with shape {value.shape}")


def load_features(files: List[str], features: List[str]) -> Dict[str, np.ndarray]:
    stacked = {f: [] for f in features}
    for path in files:
        with np.load(path) as data:
            for feat in features:
                if feat not in data:
                    raise KeyError(f"{feat} missing in {path}")
                arr = np.asarray(data[feat], dtype=np.float64)
                arr = _flatten_feature(feat, arr)
                stacked[feat].append(arr)
    return {k: np.concatenate(v) for k, v in stacked.items()}


def build_matrix(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    mat = np.vstack([feature_dict[k] for k in feature_dict]).T
    return mat


def plot_scree(pca: PCA, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA Scree Plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loadings(components: np.ndarray, feature_names: List[str], out_dir: str, prefix: str, n_components: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(n_components, components.shape[0])):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        idx = np.arange(len(feature_names))
        ax.bar(idx, components[i])
        ax.set_xticks(idx)
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_ylabel(f"{prefix}{i+1} loading")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix.lower()}{i+1}_loadings.png"), bbox_inches="tight")
        plt.close(fig)


def plot_scores(scores: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if scores.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    color = scores[:, 2] if scores.shape[1] > 2 else None
    sc = ax.scatter(scores[:, 0], scores[:, 1], s=5, alpha=0.5, c=color, cmap="viridis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Component 1 vs 2 scores")
    if color is not None:
        fig.colorbar(sc, ax=ax, label="Component 3 score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pc_scores_pc1_pc2.png"), bbox_inches="tight")
    plt.close(fig)


def run_analysis(matrix: np.ndarray, mode: str, n_components: int) -> Tuple[np.ndarray, np.ndarray, object]:
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)
    if mode == "pca":
        model = PCA(n_components=n_components)
    elif mode == "fa":
        model = FactorAnalysis(n_components=n_components)
    else:
        raise ValueError(f"Unknown mode {mode}")
    scores = model.fit_transform(X)
    components = model.components_ if hasattr(model, "components_") else None
    return scores, components, model


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA/FA on clump properties.")
    parser.add_argument("--input-root", default="clump_out/clumpn10240")
    parser.add_argument("--output-dir", default="clump_out/clumpn10240/pca")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--mode", choices=["pca", "fa"], default="pca")
    parser.add_argument("--components", type=int, default=8)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_root, "**", "clumps_master.npz"), recursive=True))
    if not files:
        raise SystemExit(f"No clumps_master.npz found under {args.input_root}")

    features = load_features(files, args.features)
    matrix = build_matrix(features)

    scores, components, model = run_analysis(matrix, args.mode, args.components)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "pca":
        plot_scree(model, os.path.join(args.output_dir, "scree.png"))
    load_prefix = "PC" if args.mode == "pca" else "Factor"
    if components is not None:
        plot_loadings(components, list(features.keys()),
                      os.path.join(args.output_dir, "loadings"), load_prefix)
    plot_scores(scores, os.path.join(args.output_dir, "scores"))

    np.savez(os.path.join(args.output_dir, f"{args.mode}_results.npz"),
             scores=scores, components=components, explained_variance=getattr(model, "explained_variance_ratio_", None))
    print(f"Saved {args.mode} outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
