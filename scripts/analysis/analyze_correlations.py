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
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple
import re

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from feature_utils import load_features

DEFAULT_FEATURES = [
    "volume",
    "mass",
    "area",
    "cell_count",
    "velocity_std",
    "velocity_mean",
]

DISPLAY_LABELS = {
    "volume": "Volume",
    "mass": "Mass",
    "area": "Area",
    "cell_count": "Cell count",
    "num_cells": "Cell count",
    "velocity_std": "Velocity σ",
    "velocity_mean": "⟨v⟩",
}


def _sanitize_label(label: str) -> str:
    replacements = {
        "⟨v⟩": "v_mean",
        "σ": "sigma",
        " ": "_",
        "-": "_",
    }
    sanitized = label
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "feature"
    return sanitized.lower()


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
        fname = f"{_sanitize_label(y_key)}_vs_{_sanitize_label(x_key)}.png"
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
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=f"Feature list to include (default: {', '.join(DEFAULT_FEATURES)}).",
    )
    args = parser.parse_args()

    search_pattern = os.path.join(args.input_root, args.file_pattern)
    files = sorted(
        f for f in glob.glob(search_pattern, recursive=True) if f.endswith(".npz") and os.path.isfile(f)
    )
    if not files:
        raise SystemExit(f"No .npz files matching {search_pattern}")

    feature_names = args.features or DEFAULT_FEATURES
    try:
        features_raw = load_features(files, feature_names)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    features = {DISPLAY_LABELS.get(key, key): features_raw[key] for key in feature_names}
    os.makedirs(args.output_dir, exist_ok=True)

    heatmap_path = os.path.join(args.output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(features, heatmap_path)
    print(f"Wrote correlation heatmap to {heatmap_path}")

    hist_dir = os.path.join(args.output_dir, "pair_histograms")
    plot_pair_histograms(features, hist_dir, n_bins=args.bins)
    print(f"Wrote pair histograms to {hist_dir}")

    snapshot_root = os.path.join(args.output_dir, "snapshots")
    for file_path in files:
        parent_name = Path(file_path).parent.name
        snap_name = _sanitize_label(parent_name or Path(file_path).stem)
        try:
            snap_features_raw = load_features([file_path], feature_names)
        except KeyError as exc:
            print(f"[WARN] Skipping {file_path}: {exc}")
            continue
        snap_features = {DISPLAY_LABELS.get(key, key): snap_features_raw[key] for key in feature_names}
        snap_dir = os.path.join(snapshot_root, snap_name, "pair_histograms")
        plot_pair_histograms(snap_features, snap_dir, n_bins=args.bins)
        print(f"Wrote pair histograms for {file_path} to {snap_dir}")


if __name__ == "__main__":
    main()
