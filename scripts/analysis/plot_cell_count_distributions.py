#!/usr/bin/env python3
"""
Plot V * N(>V) versus V for cell-count distributions in a sweep NPZ.
"""

import argparse
import os
import re
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def compute_vn_gt_v(counts: np.ndarray):
    counts = np.asarray(counts)
    counts = counts[counts > 0]
    if counts.size == 0:
        return None, None
    counts.sort()
    unique = np.unique(counts)
    n_total = counts.size
    idx = np.searchsorted(counts, unique, side="left")
    n_gt = n_total - idx
    y = unique * n_gt
    return unique, y


def extract_labels(data):
    if "snapshot_labels" in data.files:
        labels = [str(x) for x in data["snapshot_labels"].tolist()]
        return labels
    labels = [k[len("counts__"):] for k in data.files if k.startswith("counts__")]
    labels.sort()
    return labels


def infer_title(input_path: str):
    base = os.path.basename(input_path)
    m = re.search(r"n(\d+)", base)
    if m:
        return f"Cell-Count Distribution Evolution (n{m.group(1)})"
    return "Cell-Count Distribution Evolution"


def main():
    parser = argparse.ArgumentParser(description="Plot V*N(>V) vs V from sweep cell-count NPZ.")
    parser.add_argument("--input", required=True, help="Input NPZ from collect_cell_count_distributions.py")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--max-lines", type=int, default=None, help="Optional max number of snapshots to plot")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    labels = extract_labels(data)
    if args.max_lines is not None:
        labels = labels[: args.max_lines]

    if not labels:
        print("No snapshot data found.")
        return 1

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.viridis(np.linspace(0, 1, len(labels)))

    plotted = 0
    for idx, label in enumerate(labels):
        key = f"counts__{label}"
        if key not in data.files:
            continue
        x, y = compute_vn_gt_v(data[key])
        if x is None:
            continue
        ax.loglog(x, y, color=colors[idx], alpha=0.8, linewidth=1.2, label=label)
        plotted += 1

    if plotted == 0:
        print("No non-empty distributions to plot.")
        return 1

    ax.set_xlabel("V (cell count)")
    ax.set_ylabel("V * N(>V)")
    ax.set_title(args.title or infer_title(args.input))
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    # Use a compact legend outside the plot if many lines.
    if plotted <= 12:
        ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=6, ncol=2, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
