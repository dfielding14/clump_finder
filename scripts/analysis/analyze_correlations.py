#!/usr/bin/env python3
"""
Correlation and redundancy scan across clump properties.

For each combination of metric pairs, produce:
  - Pearson correlation heatmap (feature vs feature)
  - 2-D histogram (log-log) per pair with median curve and power-law fit.
"""

from __future__ import annotations

import argparse
import glob
import os
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

SELECTED_FEATURES = [
    "volume",
    "mass",
    "area",
    "pressure_mean",
    "rho_mean",
    "vx_std",
]

DISPLAY_LABELS = {
    "volume": "Volume",
    "mass": "Mass",
    "area": "Area",
    "pressure_mean": "pressure_mean",
    "rho_mean": "rho_mean",
    "vx_std": "vx_std",
}


def _flatten_feature(name: str, value: np.ndarray) -> np.ndarray:
    if value.ndim == 1:
        return value
    if name == "principal_axes_lengths":
        return np.max(value, axis=1)
    if name == "axis_ratios":
        return value[:, 0]  # choose b/a
    raise ValueError(f"Cannot flatten feature {name} with shape {value.shape}")


def load_features(files: List[str], features: List[str]) -> Dict[str, np.ndarray]:
    stacked: Dict[str, List[np.ndarray]] = {f: [] for f in features}
    for path in files:
        with np.load(path) as data:
            for feat in features:
                if feat not in data:
                    raise KeyError(f"{feat} missing in {path}")
                arr = np.asarray(data[feat])
                arr = _flatten_feature(feat, arr)
                stacked[feat].append(arr.astype(np.float64, copy=False))
    return {k: np.concatenate(v, axis=0) for k, v in stacked.items()}


def compute_correlation_matrix(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    matrix = np.vstack([feature_dict[k] for k in feature_dict]).astype(np.float64)
    return np.corrcoef(matrix)


def plot_correlation_heatmap(feature_dict: Dict[str, np.ndarray], out_path: str) -> None:
    corr = compute_correlation_matrix(feature_dict)
    labels = list(feature_dict.keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title("Pearson Correlation of Clump Properties")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _log_bins(x: np.ndarray, n_bins: int = 60) -> np.ndarray:
    valid = x[x > 0]
    if valid.size < 2:
        raise ValueError("Not enough positive values for binning.")
    return np.logspace(np.log10(valid.min()), np.log10(valid.max()), n_bins)


def _median_curve(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inds = np.digitize(x, bins) - 1
    med_x = []
    med_y = []
    for i in range(len(bins) - 1):
        mask = inds == i
        if np.count_nonzero(mask) < 5:
            continue
        med_x.append(np.exp(np.mean(np.log(x[mask]))))
        med_y.append(np.median(y[mask]))
    return np.array(med_x), np.array(med_y)


def _fit_powerlaw(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    valid = (x > 0) & (y > 0)
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan
    logx = np.log10(x[valid])
    logy = np.log10(y[valid])
    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept


def plot_pair_histograms(
    feature_dict: Dict[str, np.ndarray], out_dir: str, n_bins: int = 80
) -> None:
    keys = list(feature_dict.keys())
    os.makedirs(out_dir, exist_ok=True)
    for x_key, y_key in combinations(keys, 2):
        x = feature_dict[x_key]
        y = feature_dict[y_key]
        if np.count_nonzero(x > 0) < 50 or np.count_nonzero(y > 0) < 50:
            continue
        bins_x = _log_bins(x, n_bins=n_bins)
        bins_y = _log_bins(y, n_bins=n_bins)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        h = ax.hist2d(x, y, bins=[bins_x, bins_y], norm=matplotlib.colors.LogNorm(), cmap="viridis")
        plt.colorbar(h[3], ax=ax, label="count")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        med_x, med_y = _median_curve(x, y, bins_x)
        if med_x.size:
            ax.plot(med_x, med_y, color="w", lw=2, label="median")
            slope, intercept = _fit_powerlaw(med_x, med_y)
            if not np.isnan(slope):
                xfit = np.logspace(np.log10(med_x.min()), np.log10(med_x.max()), 100)
                yfit = 10 ** intercept * (xfit ** slope)
                ax.plot(xfit, yfit, color="orange", lw=1.5, linestyle="--",
                        label=f"∝ {x_key}^{slope:.2f}")
        ax.legend()
        ax.set_title(f"{y_key} vs {x_key}")
        fname = f"{y_key}_vs_{x_key}.png".replace("/", "_")
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze clump property correlations.")
    parser.add_argument(
        "--input-root",
        default="clump_out/res_05120",
        help="Root directory containing master clump NPZ files.",
    )
    parser.add_argument(
        "--file-pattern",
        default="*step00035*.npz",
        help="Glob pattern (relative to input root) selecting master NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        default="clump_out/res_05120/correlation_scan_step00035",
        help="Where to write correlation heatmaps and pair histograms.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of logarithmic bins for 2-D histograms.",
    )
    args = parser.parse_args()

    search_pattern = os.path.join(args.input_root, args.file_pattern)
    files = sorted(
        f for f in glob.glob(search_pattern, recursive=True) if f.endswith(".npz") and os.path.isfile(f)
    )
    if not files:
        raise SystemExit(f"No .npz files matching {search_pattern}")

    features_raw = load_features(files, SELECTED_FEATURES)
    features = {DISPLAY_LABELS[key]: features_raw[key] for key in SELECTED_FEATURES}
    os.makedirs(args.output_dir, exist_ok=True)

    heatmap_path = os.path.join(args.output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(features, heatmap_path)
    print(f"Wrote correlation heatmap to {heatmap_path}")

    hist_dir = os.path.join(args.output_dir, "pair_histograms")
    plot_pair_histograms(features, hist_dir, n_bins=args.bins)
    print(f"Wrote pair histograms to {hist_dir}")


if __name__ == "__main__":
    main()
