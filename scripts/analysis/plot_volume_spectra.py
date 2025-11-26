#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import re

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors


def load_cell_counts(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        if "cell_count" in data:
            arr = np.asarray(data["cell_count"], dtype=np.float64)
        elif "num_cells" in data:
            arr = np.asarray(data["num_cells"], dtype=np.float64)
        else:
            raise KeyError(f"cell_count/num_cells missing in {npz_path}")
    mask = np.isfinite(arr) & (arr > 0)
    return arr[mask]


def volume_spectrum(cell_counts: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(cell_counts, bins=bins)
    mid = np.sqrt(edges[1:] * edges[:-1])
    log_width = np.log(edges[1:]) - np.log(edges[:-1])
    spectrum = np.divide(mid * hist, log_width, out=np.zeros_like(hist, dtype=np.float64), where=log_width > 0)
    return mid, spectrum


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot V * dN / dlog V vs V for stitched clump catalogs.")
    ap.add_argument(
        "--input-root",
        default="clump_out/n5120_sweep",
        help="Root directory containing conn6_T0p02_stepXXXX directories.",
    )
    ap.add_argument(
        "--pattern",
        default="conn6_T0p02_step*/clumps_stitched.npz",
        help="Glob pattern relative to input root for stitched catalogs.",
    )
    ap.add_argument(
        "--output",
        default="clump_out/n5120_sweep/volume_spectra.png",
        help="Output PNG path.",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of logarithmic bins.",
    )
    ap.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name for the lines.",
    )
    args = ap.parse_args()

    root = Path(args.input_root)
    paths = sorted(root.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files match {root / args.pattern}")

    spectra = []
    global_counts = []
    step_values = []
    for path in paths:
        counts = load_cell_counts(path)
        counts = counts[counts > 8]
        if counts.size == 0:
            continue
        match = re.search(r"step(\d+)", path.parts[-2])
        if match:
            step_val = int(match.group(1))
        else:
            step_val = len(step_values)
        global_counts.append(counts)
        spectra.append((path, counts, step_val))
        step_values.append(step_val)

    if not spectra:
        raise SystemExit("No catalogs with positive cell counts found.")

    all_counts = np.concatenate(global_counts)
    bins = np.logspace(np.log10(all_counts.min()), np.log10(all_counts.max()), args.bins + 1)

    cmap = matplotlib.colormaps.get_cmap(args.cmap)
    norm = colors.Normalize(vmin=min(step_values), vmax=max(step_values)) if step_values else None
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    for path, counts, step_val in spectra:
        mid, spec = volume_spectrum(counts, bins)
        color = cmap(norm(step_val)) if norm is not None else cmap(0.5)
        ax.step(mid, spec, where="mid", color=color, alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$V [\Delta x^3]$")
    ax.set_ylabel(r"$V\, dN / d\log V$")

    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ticks = np.linspace(
            min(step_values),
            max(step_values),
            num=min(len(step_values), 6),
            dtype=int,
        )
        cbar = fig.colorbar(sm, ax=ax, label=r"Snapshot step $N$", ticks=ticks)
        cbar.ax.set_yticklabels([fr"${int(tick)}$" for tick in ticks])
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
    for path, counts, step_val in spectra:
        bins_mid = np.sqrt(bins[1:] * bins[:-1])
        cumulative = np.array([np.count_nonzero(counts >= b) for b in bins_mid], dtype=np.float64)
        color = cmap(norm(step_val)) if norm is not None else cmap(0.5)
        ax2.step(bins_mid, cumulative, where="mid", color=color, alpha=0.9)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$V [\Delta x^3]$")
    ax2.set_ylabel(r"$N(>V)$")

    if norm is not None:
        sm2 = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm2.set_array([])
        ticks = np.linspace(
            min(step_values),
            max(step_values),
            num=min(len(step_values), 6),
            dtype=int,
        )
        cbar2 = fig2.colorbar(sm2, ax=ax2, label=r"Snapshot step $N$", ticks=ticks)
        cbar2.ax.set_yticklabels([fr"${int(tick)}$" for tick in ticks])

    fig2.tight_layout()
    out_path2 = Path(args.output).with_name("volume_cumulative.png")
    out_path2.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_path2, bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
