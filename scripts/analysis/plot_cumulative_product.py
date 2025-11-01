#!/usr/bin/env python3
"""
Plot cumulative clump statistics and compare distributions across parameter choices.

Two operating modes are supported:

* Aggregate mode (legacy behaviour) – compute median/min/max envelopes across every
  NPZ discovered under the input root and plot cumulative scaling curves.
* Connectivity/temperature comparison – recognise files named like
  ``res05120_step00035_conn06_T0p02.npz`` and overlay the cumulative products for
  each (connectivity, temperature threshold) pair so the impact on the radius,
  volume, and mass distributions can be inspected directly.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_PATTERN = re.compile(
    r"res(?P<res>\d+)_.*?step(?P<step>\d+)_conn(?P<conn>\d+)_T(?P<temp>[0-9p]+)",
    re.IGNORECASE,
)


def _find_npz_files(root: str, pattern: str) -> List[str]:
    search = os.path.join(root, pattern)
    files = sorted(
        f
        for f in glob.glob(search, recursive=True)
        if f.endswith(".npz") and os.path.isfile(f)
    )
    return files


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _prepare_bins(
    data_arrays: List[np.ndarray], n_bins: int = 60, unit_scale: float = 1.0
) -> np.ndarray:
    vals = np.concatenate([arr[arr > 0] for arr in data_arrays if arr.size], axis=0)
    if vals.size == 0:
        raise ValueError("No positive values available to define bins.")
    scaled = vals * unit_scale
    vmin = scaled.min()
    vmax = scaled.max()
    if vmin <= 0:
        vmin = np.min(scaled[scaled > 0])
    bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins, endpoint=True)
    return bins


def _compute_cumulative_products(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return N(>threshold) * threshold for each threshold."""
    vals = np.sort(values[values > 0])
    if vals.size == 0:
        return np.zeros_like(thresholds)
    counts = vals.size - np.searchsorted(vals, thresholds, side="left")
    return counts * thresholds


def _aggregate(
    data_arrays: List[np.ndarray], thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not data_arrays:
        raise ValueError("No data arrays supplied.")

    products = []
    for arr in data_arrays:
        products.append(_compute_cumulative_products(arr, thresholds))
    matrix = np.vstack(products)
    median = np.median(matrix, axis=0)
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    return median, min_vals, max_vals


def _fit_powerlaw_segment(
    x: np.ndarray, y: np.ndarray, start_frac: float = 0.55, end_frac: float = 0.80
) -> Tuple[float, float, Tuple[float, float] | None]:
    valid = (x > 0) & (y > 0)
    if not np.any(valid):
        return np.nan, np.nan, None
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.nan, np.nan, None
    lo = int(np.floor(start_frac * x.size))
    hi = int(np.floor(end_frac * x.size))
    lo = max(0, min(lo, x.size - 2))
    hi = max(lo + 2, min(hi, x.size))
    x_slice = x[lo:hi]
    y_slice = y[lo:hi]
    if x_slice.size < 2 or np.allclose(y_slice, y_slice[0]):
        return np.nan, np.nan, None
    logx = np.log10(x_slice)
    logy = np.log10(y_slice)
    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept, (float(x_slice.min()), float(x_slice.max()))


def _fit_powerlaw(
    x: np.ndarray, y: np.ndarray, start_frac: float = 0.55, end_frac: float = 0.80
) -> Tuple[float, float]:
    slope, intercept, _ = _fit_powerlaw_segment(x, y, start_frac=start_frac, end_frac=end_frac)
    return slope, intercept


def _darken_color(color: str, factor: float = 0.6) -> Tuple[float, float, float]:
    """Darken an RGB color by scaling towards black."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def _plot_fit_overlay(
    ax: plt.Axes,
    thresholds: np.ndarray,
    slope: float,
    intercept: float,
    window: Tuple[float, float] | None,
    base_color: str,
    label: str | None,
    *,
    zorder: float = 5,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if np.isnan(slope):
        return None

    valid_thresh = np.sort(thresholds[thresholds > 0])
    if valid_thresh.size < 2:
        return None

    start_idx = int(np.floor(0.3 * (valid_thresh.size - 1)))
    end_idx = int(np.floor(0.9 * (valid_thresh.size - 1)))
    end_idx = max(end_idx, start_idx + 1)
    x_start = float(valid_thresh[start_idx])
    x_end = float(valid_thresh[end_idx])

    if window:
        x_start = min(x_start, window[0])
        x_end = max(x_end, window[1])

    if x_start <= 0 or x_end <= 0 or x_start >= x_end:
        return None

    xfit = np.logspace(np.log10(x_start), np.log10(x_end), 200)
    yfit = 10 ** intercept * (xfit ** slope)

    ax.plot(
        xfit,
        yfit,
        linestyle=":",
        linewidth=2.4,
        color=base_color,
        label=label if label is not None else None,
        zorder=zorder,
    )
    return xfit, yfit


def plot_cumulative_products(
    files: List[str], out_path: str, n_bins: int = 60, show_mass: bool = True
) -> None:
    if not files:
        raise ValueError("No clumps_master.npz files found.")

    volumes = []
    masses = []
    for f in files:
        data = _load_npz(f)
        if "volume" not in data or "mass" not in data:
            raise KeyError(f"Required fields 'volume' or 'mass' missing in {f}")
        volumes.append(data["volume"].astype(np.float64, copy=False))
        masses.append(data["mass"].astype(np.float64, copy=False))

    # Infer grid spacing from metadata (fallback to 1/Nres if not available)
    dx = None
    first = _load_npz(files[0])
    if "voxel_spacing" in first:
        spacing = np.asarray(first["voxel_spacing"], dtype=np.float64)
        if spacing.size:
            dx = float(np.mean(spacing))
    if dx is None:
        match = RUN_PATTERN.search(os.path.basename(files[0]))
        if match:
            dx = 1.0 / float(match.group("res"))
    if dx is None:
        dx = 1.0 / 10240.0
    dx3 = dx ** 3

    radius_arrays = [np.cbrt(v) / dx for v in volumes]
    scaled_volumes = [v / dx3 for v in volumes]
    scaled_masses = [m / dx3 for m in masses]

    r_bins = _prepare_bins(radius_arrays, n_bins=n_bins)
    r_median, r_min, r_max = _aggregate(radius_arrays, r_bins)

    vol_bins = _prepare_bins(scaled_volumes, n_bins=n_bins)
    vol_median, vol_min, vol_max = _aggregate(scaled_volumes, vol_bins)

    n_cols = 3 if show_mass else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    ax.fill_between(r_bins, r_min, r_max, color="violet", alpha=0.35, label="min/max")
    ax.plot(r_bins, r_median, color="purple", lw=2.2, label="median", zorder=4)
    r_slope, r_intercept, r_window = _fit_powerlaw_segment(r_bins, r_median)
    fit_label = f"∝ r_eff^{r_slope:.2f}" if not np.isnan(r_slope) else None
    if fit_label:
        _plot_fit_overlay(
            ax,
            r_bins,
            r_slope,
            r_intercept,
            r_window,
            base_color="purple",
            label=fit_label,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective radius (Δx)")
    ax.set_ylabel("N(>r_eff) × r_eff")
    ax.set_title("Cumulative Radius Scaling")
    ax.legend()

    ax = axes[1]
    ax.fill_between(vol_bins, vol_min, vol_max, color="skyblue", alpha=0.35, label="min/max")
    ax.plot(vol_bins, vol_median, color="navy", lw=2.2, label="median", zorder=4)
    v_slope, v_intercept, v_window = _fit_powerlaw_segment(vol_bins, vol_median)
    fit_label = f"∝ V^{v_slope:.2f}" if not np.isnan(v_slope) else None
    if fit_label:
        _plot_fit_overlay(
            ax,
            vol_bins,
            v_slope,
            v_intercept,
            v_window,
            base_color="navy",
            label=fit_label,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Volume (Δx³)")
    ax.set_ylabel("N(>V) × V")
    ax.set_title("Cumulative Volume Scaling")
    ax.legend()

    if show_mass:
        mass_bins = _prepare_bins(scaled_masses, n_bins=n_bins)
        mass_median, mass_min, mass_max = _aggregate(scaled_masses, mass_bins)
        ax = axes[2]
        ax.fill_between(mass_bins, mass_min, mass_max, color="salmon", alpha=0.35, label="min/max")
        ax.plot(mass_bins, mass_median, color="darkred", lw=2.2, label="median", zorder=4)
        m_slope, m_intercept, m_window = _fit_powerlaw_segment(mass_bins, mass_median)
        fit_label = f"∝ M^{m_slope:.2f}" if not np.isnan(m_slope) else None
        if fit_label:
            _plot_fit_overlay(
                ax,
                mass_bins,
                m_slope,
                m_intercept,
                m_window,
                base_color="darkred",
                label=fit_label,
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mass (ρ₀ Δx³)")
        ax.set_ylabel("N(>M) × M")
        ax.set_title("Cumulative Mass Scaling")
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _group_by_conn_temp(files: Iterable[str]) -> List[Tuple[Dict[str, float], List[str]]]:
    grouped: Dict[Tuple[int, float], List[str]] = defaultdict(list)
    meta: Dict[Tuple[int, float], Dict[str, float]] = {}
    for path in files:
        name = os.path.basename(path)
        match = RUN_PATTERN.search(name)
        if not match:
            continue
        res = int(match.group("res"))
        step = int(match.group("step"))
        conn = int(match.group("conn"))
        temp_str = match.group("temp").replace("p", ".")
        try:
            temp = float(temp_str)
        except ValueError:
            continue
        key = (conn, temp)
        grouped[key].append(path)
        meta[key] = {
            "label": f"conn{conn:02d}, T={temp:.2f}",
            "conn": conn,
            "temp": temp,
            "res": res,
            "step": step,
        }
    ordered_keys = sorted(meta.keys(), key=lambda item: (item[0], item[1]))
    return [(meta[key], grouped[key]) for key in ordered_keys]


def plot_conn_temp_comparison(
    grouped_runs: List[Tuple[Dict[str, float], List[str]]],
    out_path: str,
    n_bins: int = 60,
    show_mass: bool = True,
) -> None:
    if not grouped_runs:
        raise ValueError("No connectivity/temperature runs recognised.")

    first_meta, first_files = grouped_runs[0]
    first_path = first_files[0]
    reference = _load_npz(first_path)
    dx = None
    if "voxel_spacing" in reference:
        spacing = np.asarray(reference["voxel_spacing"], dtype=np.float64)
        if spacing.size:
            dx = float(np.mean(spacing))
    if dx is None:
        dx = 1.0 / float(first_meta["res"])
    dx3 = dx ** 3

    run_entries = []
    all_radius_arrays: List[np.ndarray] = []
    all_volume_arrays: List[np.ndarray] = []
    all_mass_arrays: List[np.ndarray] = []

    for meta, paths in grouped_runs:
        volumes: List[np.ndarray] = []
        masses: List[np.ndarray] = []
        for path in paths:
            data = _load_npz(path)
            if "volume" not in data or "mass" not in data:
                raise KeyError(f"Required fields 'volume' or 'mass' missing in {path}")
            volumes.append(np.asarray(data["volume"], dtype=np.float64))
            masses.append(np.asarray(data["mass"], dtype=np.float64))

        concat_volume = np.concatenate(volumes, axis=0) if volumes else np.array([])
        concat_mass = np.concatenate(masses, axis=0) if masses else np.array([])
        radius = np.cbrt(concat_volume) / dx
        volume_scaled = concat_volume / dx3
        mass_scaled = concat_mass / dx3

        run_entries.append(
            {
                "meta": meta,
                "radius": radius,
                "volume": volume_scaled,
                "mass": mass_scaled,
            }
        )
        all_radius_arrays.append(radius)
        all_volume_arrays.append(volume_scaled)
        all_mass_arrays.append(mass_scaled)

    r_bins = _prepare_bins(all_radius_arrays, n_bins=n_bins)
    v_bins = _prepare_bins(all_volume_arrays, n_bins=n_bins)
    m_bins = _prepare_bins(all_mass_arrays, n_bins=n_bins) if show_mass else None

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.9, len(run_entries)))

    # Precompute curves and fits for annotation/overlay
    for entry, color in zip(run_entries, colors):
        entry["color"] = color
        entry["radius_curve"] = _compute_cumulative_products(entry["radius"], r_bins)
        entry["volume_curve"] = _compute_cumulative_products(entry["volume"], v_bins)
        entry["mass_curve"] = _compute_cumulative_products(entry["mass"], m_bins) if show_mass else None

        r_slope, r_intercept, r_window = _fit_powerlaw_segment(r_bins, entry["radius_curve"])
        v_slope, v_intercept, v_window = _fit_powerlaw_segment(v_bins, entry["volume_curve"])
        if show_mass and entry["mass_curve"] is not None:
            m_slope, m_intercept, m_window = _fit_powerlaw_segment(m_bins, entry["mass_curve"])
        else:
            m_slope = m_intercept = np.nan
            m_window = None

        entry.update(
            radius_slope=r_slope,
            radius_intercept=r_intercept,
            radius_window=r_window,
            volume_slope=v_slope,
            volume_intercept=v_intercept,
            volume_window=v_window,
            mass_slope=m_slope,
            mass_intercept=m_intercept,
            mass_window=m_window,
        )

    n_cols = 3 if show_mass else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Radius panel
    ax = axes[0]
    for entry in run_entries:
        curve = entry["radius_curve"]
        label = entry["meta"]["label"]
        if not np.isnan(entry["radius_slope"]):
            label = f"{label} (α={entry['radius_slope']:.2f})"
        ax.plot(r_bins, curve, color=entry["color"], lw=2, label=label)
        if not np.isnan(entry["radius_slope"]):
            _plot_fit_overlay(
                ax,
                r_bins,
                entry["radius_slope"],
                entry["radius_intercept"],
                entry["radius_window"],
                base_color=entry["color"],
                label=None,
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective radius (Δx)")
    ax.set_ylabel("N(>r_eff) × r_eff")
    ax.set_title("Cumulative Radius Scaling")
    ax.legend(title="Connectivity / T", fontsize="small")

    # Volume panel
    ax = axes[1]
    for entry in run_entries:
        curve = entry["volume_curve"]
        ax.plot(v_bins, curve, color=entry["color"], lw=2)
        if not np.isnan(entry["volume_slope"]):
            fit = _plot_fit_overlay(
                ax,
                v_bins,
                entry["volume_slope"],
                entry["volume_intercept"],
                entry["volume_window"],
                base_color=entry["color"],
                label=None,
                zorder=5,
            )
            if fit is not None:
                xfit, yfit = fit
                x_text = xfit[-1]
                y_text = yfit[-1]
                ax.text(
                    x_text,
                    y_text,
                    f"α={entry['volume_slope']:.2f}",
                    color=entry["color"],
                    fontsize=9,
                    ha="right",
                    va="bottom",
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Volume (Δx³)")
    ax.set_ylabel("N(>V) × V")
    ax.set_title("Cumulative Volume Scaling")

    # Mass panel
    if show_mass and m_bins is not None:
        ax = axes[2]
        for entry in run_entries:
            curve = entry["mass_curve"]
            ax.plot(m_bins, curve, color=entry["color"], lw=2)
            if not np.isnan(entry["mass_slope"]):
                fit = _plot_fit_overlay(
                    ax,
                    m_bins,
                    entry["mass_slope"],
                    entry["mass_intercept"],
                    entry["mass_window"],
                    base_color=entry["color"],
                    label=None,
                    zorder=5,
                )
                if fit is not None:
                    xfit, yfit = fit
                    x_text = xfit[-1]
                    y_text = yfit[-1]
                    ax.text(
                        x_text,
                        y_text,
                        f"α={entry['mass_slope']:.2f}",
                        color=entry["color"],
                        fontsize=9,
                        ha="right",
                        va="bottom",
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mass (ρ₀ Δx³)")
        ax.set_ylabel("N(>M) × M")
        ax.set_title("Cumulative Mass Scaling")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cumulative clump scaling products.")
    parser.add_argument(
        "--input-root",
        default="clump_out/res_05120",
        help="Root directory containing clump NPZ files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=5120,
        help="Resolution used when constructing default file patterns and spacing.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=35,
        help="Simulation step to focus on when constructing default file patterns.",
    )
    parser.add_argument(
        "--output",
        default="clump_out/res_05120/cumulative_scaling_step00035.png",
        help="Output PNG path for the cumulative plots.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of logarithmic bins for cumulative products.",
    )
    parser.add_argument(
        "--skip-mass",
        action="store_true",
        help="If set, only plot the volume cumulative curve.",
    )
    parser.add_argument(
        "--file-pattern",
        default=None,
        help=(
            "Glob pattern relative to input root. Defaults to "
            "'res{resolution:05d}_*step{step:05d}*.npz'."
        ),
    )
    parser.add_argument(
        "--compare-conn-temp",
        action="store_true",
        help=(
            "Force connectivity/temperature comparison mode. If not provided, the "
            "script automatically compares when multiple matching (conn, T) pairs "
            "are detected."
        ),
    )
    args = parser.parse_args()

    pattern = args.file_pattern
    if pattern is None:
        pattern = f"res{args.resolution:05d}_*step{args.step:05d}*.npz"

    files = _find_npz_files(args.input_root, pattern)
    if not files:
        raise SystemExit(f"No NPZ files matching pattern '{pattern}' under {args.input_root}")

    grouped_runs = _group_by_conn_temp(files)
    compare_mode = args.compare_conn_temp or len(grouped_runs) > 1

    if compare_mode and grouped_runs:
        plot_conn_temp_comparison(
            grouped_runs,
            args.output,
            n_bins=args.bins,
            show_mass=not args.skip_mass,
        )
    else:
        plot_cumulative_products(files, args.output, n_bins=args.bins, show_mass=not args.skip_mass)

    # Produce individual plots for each (conn, T) run
    output_dir = os.path.dirname(args.output) or "."
    for meta, paths in grouped_runs:
        temp_token = f"{meta['temp']:.2f}".replace(".", "p")
        indiv_name = (
            f"cumulative_scaling_step{meta['step']:05d}_conn{meta['conn']:02d}_T{temp_token}.png"
        )
        indiv_output = os.path.join(output_dir, indiv_name)
        plot_cumulative_products(paths, indiv_output, n_bins=args.bins, show_mass=not args.skip_mass)

    print(f"Wrote cumulative plots to {args.output}")


if __name__ == "__main__":
    main()
