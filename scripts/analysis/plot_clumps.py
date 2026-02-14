from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Optional, Tuple


def _load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _hist_log(ax, data, bins=50, label=None, xlabel=None, ylabel=None):
    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return
    lo, hi = np.nanmin(data), np.nanmax(data)
    if lo <= 0:
        lo = np.percentile(data, 1)
    edges = np.logspace(np.log10(lo), np.log10(hi), bins)
    ax.hist(data, bins=edges, histtype='stepfilled', alpha=0.85)
    ax.set_xscale('log')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _hist2d(ax, x, y, bins=100, xlog=True, ylog=True, xlabel=None, ylabel=None, xedges=None):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if xlog:
        pos = x > 0
        x = x[pos]
        y = y[pos]
    if ylog:
        pos = y > 0
        x = x[pos]
        y = y[pos]
    if x.size == 0 or y.size == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return

    # Prepare bin edges in linear space; if log axes requested, edges are log-spaced
    if xedges is not None:
        x_edges = np.asarray(xedges)
    else:
        if xlog:
            xmin = np.min(x)
            xmax = np.max(x)
            x_edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)
        else:
            x_edges = np.linspace(np.min(x), np.max(x), bins)

    if ylog:
        ymin = np.min(y)
        ymax = np.max(y)
        y_edges = np.logspace(np.log10(ymin), np.log10(ymax), bins)
    else:
        y_edges = np.linspace(np.min(y), np.max(y), bins)

    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges])
    positive = H[H > 0]
    vmin = np.nanmin(positive) if positive.size else 1.0
    vmax = np.nanmax(positive) if positive.size else 1.0
    im = ax.pcolormesh(xe, ye, H.T, shading='auto', cmap='viridis',
                       norm=LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1.0)))
    plt.colorbar(im, ax=ax, label='counts')
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel or 'x')
    ax.set_ylabel(ylabel or 'y')


def _load_size(npz_path: str, use_volume: bool) -> np.ndarray:
    with np.load(npz_path) as d:
        key = "volume" if use_volume else "cell_count"
        if key not in d:
            raise KeyError(f"{key} missing in {npz_path}")
        arr = np.asarray(d[key], dtype=np.float64)
    mask = np.isfinite(arr) & (arr > 0)
    return arr[mask]


def _size_axis_labels(use_volume: bool) -> tuple[str, str, str, str]:
    if use_volume:
        return "V", "V [Δx^3]", "V · dN / dlog V", "volume"
    return "N_cell", "cell count", "N_cell · dN / dlog N_cell", "cell_count"


def _compensated_cumulative(values: np.ndarray, n_points: int = 300) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None, None
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
        return None, None
    if np.isclose(vmin, vmax):
        thresholds = np.array([vmin], dtype=np.float64)
        compensated = np.array([vmin * vals.size], dtype=np.float64)
        return thresholds, compensated
    thresholds = np.logspace(np.log10(vmin), np.log10(vmax), n_points)
    sorted_vals = np.sort(vals)
    counts = sorted_vals.size - np.searchsorted(sorted_vals, thresholds, side="left")
    compensated = thresholds * counts.astype(np.float64)
    return thresholds, compensated


def _plot_binned_mean(ax, x: np.ndarray, y: np.ndarray, n_bins: int = 50, xlog: bool = True) -> None:
    """Overlay mean(y) in bins of x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if xlog:
        mask = mask & (x > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return

    xmin = np.min(x)
    xmax = np.max(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
        return

    if xlog:
        edges = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.digitize(x, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    if not np.any(valid):
        return

    sums = np.bincount(bin_idx[valid], weights=y[valid], minlength=n_bins).astype(np.float64)
    counts = np.bincount(bin_idx[valid], minlength=n_bins).astype(np.float64)
    means = np.divide(sums, counts, out=np.full(n_bins, np.nan, dtype=np.float64), where=counts > 0)
    m = np.isfinite(means)
    if np.count_nonzero(m) < 2:
        return

    ax.plot(centers[m], means[m], color='black', linewidth=2.0, alpha=0.9, zorder=5)
    ax.plot(centers[m], means[m], color='white', linewidth=1.2, alpha=0.95, zorder=6)


def plot_histogram_comparison(primary_path: str, secondary_path: str, outdir: str,
                              use_volume: bool, labels: Optional[Tuple[str, str]] = None) -> None:
    size_primary = _load_size(primary_path, use_volume)
    size_secondary = _load_size(secondary_path, use_volume)
    _, size_xlabel, size_ylabel, sum_label = _size_axis_labels(use_volume)

    if size_primary.size == 0 or size_secondary.size == 0:
        print("[plot_clumps] Skipping histogram comparison; one input has no positive sizes.")
        return

    combined = np.concatenate([size_primary, size_secondary])
    xmin = combined.min()
    xmax = combined.max()
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 80)

    hist_primary, _ = np.histogram(size_primary, bins=bins)
    hist_secondary, _ = np.histogram(size_secondary, bins=bins)
    log_width = np.log(bins[1:]) - np.log(bins[:-1])
    v_mid = np.sqrt(bins[1:] * bins[:-1])
    spec_primary = np.divide(v_mid * hist_primary, log_width,
                             out=np.full_like(hist_primary, np.nan, dtype=np.float64),
                             where=log_width > 0)
    spec_secondary = np.divide(v_mid * hist_secondary, log_width,
                               out=np.full_like(hist_secondary, np.nan, dtype=np.float64),
                               where=log_width > 0)
    ratio = np.divide(
        spec_secondary,
        spec_primary,
        out=np.full_like(hist_secondary, np.nan, dtype=np.float64),
        where=(spec_primary > 0) & (spec_secondary > 0),
    )
    total_primary = np.sum(size_primary)
    total_secondary = np.sum(size_secondary)

    base_primary = os.path.splitext(os.path.basename(primary_path))[0]
    base_secondary = os.path.splitext(os.path.basename(secondary_path))[0]
    if labels:
        label_primary, label_secondary = labels
    else:
        label_primary = base_primary
        label_secondary = base_secondary
    label_primary = f"{label_primary} (∑ {sum_label} = {total_primary:.3e})"
    label_secondary = f"{label_secondary} (∑ {sum_label} = {total_secondary:.3e})"

    centers = bins[:-1]
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, dpi=150)

    ax_top.step(centers, spec_primary, where='post', label=label_primary, color='tab:blue')
    ax_top.step(centers, spec_secondary, where='post', label=label_secondary, color='tab:orange')
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel(size_ylabel)
    ax_top.legend()

    ax_bottom.step(centers, ratio, where='post', color='tab:purple')
    ax_bottom.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax_bottom.set_xscale('log')
    ax_bottom.set_xlabel(size_xlabel)
    ax_bottom.set_ylabel('ratio (secondary / primary)')

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    compare_name = f"{base_primary}_size_hist_compare.png"
    fig.savefig(os.path.join(outdir, compare_name), bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_clumps] Wrote comparison histogram {compare_name}")


def find_ell_bin_edges(r_min: int, r_max: int, n_ell_bins: int) -> np.ndarray:
    """Compute integer-rounded geometric bin edges with exactly n_ell_bins bins.

    Uses a binary search on the number of geometric points, rounding to integers and taking uniques,
    to obtain exactly n_ell_bins+1 edges when possible. Falls back to the closest result.
    """
    n_points_low = n_ell_bins + 1
    n_points_high = max(n_points_low + 1, 3 * n_ell_bins)

    best = None
    while n_points_low <= n_points_high:
        n_points_mid = (n_points_low + n_points_high) // 2
        edges = np.unique(np.around(np.geomspace(max(1, r_min), max(r_min + 1, r_max), n_points_mid)).astype(int))
        best = edges
        if len(edges) < n_ell_bins + 1:
            n_points_low = n_points_mid + 1
        elif len(edges) > n_ell_bins + 1:
            n_points_high = n_points_mid - 1
        else:
            break

    if best is None:
        best = np.arange(r_min, r_max + 1)
    if len(best) != n_ell_bins + 1:
        print(f"Warning: Could not find exactly {n_ell_bins + 1} unique bin edges. Using {len(best)} instead.")
    return best


def make_pngs(npz_path: str,
              outdir: str,
              use_volume: bool = False,
              mass_weighted: bool = False,
              prefix: str | None = None,
              compensated_cumulative: bool = False):
    d = _load(npz_path)
    size = d['volume'] if use_volume else d['cell_count']
    _, size_xlabel, size_ylabel, _ = _size_axis_labels(use_volume)
    shape_metrics_valid = d.get('shape_metrics_valid')
    if shape_metrics_valid is not None:
        shape_metrics_valid = np.asarray(shape_metrics_valid, dtype=bool)
    else:
        shape_metrics_valid = np.ones(size.shape, dtype=bool)
    if mass_weighted:
        vx_std = d.get('vx_std_massw')
        vy_std = d.get('vy_std_massw')
        vz_std = d.get('vz_std_massw')
    else:
        vx_std = d.get('vx_std')
        vy_std = d.get('vy_std')
        vz_std = d.get('vz_std')
    velocity_std_scalar = d.get('velocity_std')

    have_velocity = False
    vdisp = None
    if vx_std is not None and vy_std is not None and vz_std is not None:
        vdisp = np.sqrt(vx_std**2 + vy_std**2 + vz_std**2)
        have_velocity = True
    elif velocity_std_scalar is not None:
        vdisp = velocity_std_scalar.astype(np.float64, copy=False)
        have_velocity = True

    os.makedirs(outdir, exist_ok=True)
    base = prefix or (os.path.splitext(os.path.basename(npz_path))[0])

    # 1) Size distribution (differential spectrum or compensated cumulative)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    size_symbol = "V" if use_volume else "N_cell"
    if compensated_cumulative:
        x, y = _compensated_cumulative(size, n_points=300)
        if x is None or y is None:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            ax.plot(x[mask], y[mask], alpha=0.9)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(size_xlabel)
        ax.set_ylabel(f'{size_symbol} · N(>{size_symbol})')
    else:
        size_pos = np.asarray(size, dtype=np.float64)
        size_pos = size_pos[np.isfinite(size_pos) & (size_pos > 0)]
        if size_pos.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            lo = float(np.min(size_pos))
            hi = float(np.max(size_pos))
            # Choose bins: integer-rounded geometric edges for integer sizes; logspace for floats
            if np.isclose(lo, hi):
                edges_f = np.array([lo, lo * (1.0 + 1e-6)], dtype=np.float64)
            elif (np.issubdtype(size.dtype, np.integer)) or np.allclose(size_pos, np.round(size_pos)):
                r_min = int(max(1, np.floor(lo)))
                r_max = int(np.ceil(hi))
                if r_max <= r_min:
                    edges_f = np.array([lo, hi], dtype=np.float64)
                else:
                    edges_f = find_ell_bin_edges(r_min, r_max, n_ell_bins=60).astype(np.float64, copy=False)
            else:
                edges_f = np.logspace(np.log10(lo), np.log10(hi), 60)

            edges_f = np.unique(edges_f[np.isfinite(edges_f)])
            if edges_f.size < 2:
                edges_f = np.array([lo, hi if hi > lo else lo * (1.0 + 1e-6)], dtype=np.float64)

            counts, _ = np.histogram(size_pos, bins=edges_f)
            log_width = np.log(edges_f[1:]) - np.log(edges_f[:-1])
            # Use float edges for midpoint calculation to avoid int64 overflow on large bins.
            v_mid = np.sqrt(edges_f[1:] * edges_f[:-1])
            spectrum = np.divide(v_mid * counts, log_width,
                                 out=np.full_like(counts, np.nan, dtype=np.float64),
                                 where=log_width > 0)
            ax.step(edges_f[:-1], spectrum, where='post', alpha=0.9)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(size_xlabel)
            ax.set_ylabel(size_ylabel)
    fig.savefig(os.path.join(outdir, f"{base}_size_hist.png"), bbox_inches='tight')
    plt.close(fig)

    # 2) Size vs velocity dispersion - REMOVED due to numerical issues with large coordinate values

    # 3) Area vs size joint distribution
    area = d.get('area')
    volume_for_area = d.get('volume')
    if area is not None and volume_for_area is not None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        vol = volume_for_area.astype(np.float64, copy=False)
        pos = vol > 0
        vol = vol[pos]
        area_pos = area[pos]
        if vol.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
        else:
            ratio = area_pos / np.power(vol, 8.0 / 9.0)
            # Filter out non-positive ratios before log scaling
            mask = ratio > 0
            vol = vol[mask]
            ratio = ratio[mask]
            if vol.size == 0 or ratio.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
            else:
                lo = np.nanmin(vol)
                hi = np.nanmax(vol)
                xedges = np.logspace(np.log10(lo), np.log10(hi), 60)
                _hist2d(ax, vol, ratio, bins=100, xlog=True, ylog=True,
                        xlabel='clump volume', ylabel='area / volume$^{8/9}$', xedges=xedges)
        fig.savefig(os.path.join(outdir, f"{base}_area_over_vol89_vs_volume.png"), bbox_inches='tight')
        plt.close(fig)

    # 4) Velocity dispersion vs volume - REMOVED due to numerical issues

    # 5) Mass distribution (differential spectrum or compensated cumulative)
    mass = d.get('mass')
    if mass is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        mass_pos = mass[np.isfinite(mass) & (mass > 0)]
        if mass_pos.size > 0:
            if compensated_cumulative:
                x, y = _compensated_cumulative(mass_pos, n_points=300)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                ax.plot(x[mask], y[mask], alpha=0.9)
            else:
                lo = np.nanmin(mass_pos)
                hi = np.nanmax(mass_pos)
                edges = np.logspace(np.log10(lo), np.log10(hi), 60)
                counts, _ = np.histogram(mass_pos, bins=edges)
                log_width = np.log(edges[1:]) - np.log(edges[:-1])
                m_mid = np.sqrt(edges[1:] * edges[:-1])
                spectrum = np.divide(m_mid * counts, log_width,
                                     out=np.full_like(counts, np.nan, dtype=np.float64),
                                     where=log_width > 0)
                ax.step(edges[:-1], spectrum, where='post', alpha=0.9)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('M [code units]')
            if compensated_cumulative:
                ax.set_ylabel('M · N(>M)')
            else:
                ax.set_ylabel('M · dN / dlog M')
        else:
            ax.text(0.5, 0.5, "No mass data", ha='center', va='center')
        fig.savefig(os.path.join(outdir, f"{base}_mass_spectrum.png"), bbox_inches='tight')
        plt.close(fig)

    # 6) Surface area distribution (differential spectrum or compensated cumulative)
    area = d.get('area')
    if area is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        area_pos = area[np.isfinite(area) & (area > 0)]
        if area_pos.size > 0:
            if compensated_cumulative:
                x, y = _compensated_cumulative(area_pos, n_points=300)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                ax.plot(x[mask], y[mask], alpha=0.9)
            else:
                lo = np.nanmin(area_pos)
                hi = np.nanmax(area_pos)
                edges = np.logspace(np.log10(lo), np.log10(hi), 60)
                counts, _ = np.histogram(area_pos, bins=edges)
                log_width = np.log(edges[1:]) - np.log(edges[:-1])
                a_mid = np.sqrt(edges[1:] * edges[:-1])
                spectrum = np.divide(a_mid * counts, log_width,
                                     out=np.full_like(counts, np.nan, dtype=np.float64),
                                     where=log_width > 0)
                ax.step(edges[:-1], spectrum, where='post', alpha=0.9)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('A [Δx^2]')
            if compensated_cumulative:
                ax.set_ylabel('A · N(>A)')
            else:
                ax.set_ylabel('A · dN / dlog A')
        else:
            ax.text(0.5, 0.5, "No area data", ha='center', va='center')
        fig.savefig(os.path.join(outdir, f"{base}_area_spectrum.png"), bbox_inches='tight')
        plt.close(fig)

    # 7) Shape metrics vs size (sphericity, compactness, triaxiality, elongation)
    shape_metrics = [
        ('sphericity', r'Sphericity $\Phi=\pi^{1/3}(6V)^{2/3}/A$', (0, 1)),
        ('compactness', r'Compactness $C=36\pi V^2/A^3$', (0, 1)),
        ('triaxiality', r'Triaxiality $T=(a^2-b^2)/(a^2-c^2)$', (0, 1)),
        ('elongation', r'Elongation $E=a/c$', (1, 1e3)),  # reasonable range for axis ratio
    ]
    has_shape = any(d.get(m[0]) is not None for m in shape_metrics)
    if has_shape:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
        axes = axes.flatten()
        for ax, (key, label, ylim) in zip(axes, shape_metrics):
            metric = d.get(key)
            if metric is None:
                ax.text(0.5, 0.5, f"No {key} data", ha='center', va='center', transform=ax.transAxes)
                continue
            # For elongation, use log scale and filter reasonable values
            is_elongation = (key == 'elongation')
            mask = np.isfinite(size) & np.isfinite(metric) & (size > 0) & shape_metrics_valid
            if key in ('sphericity', 'compactness', 'triaxiality'):
                mask = mask & (metric >= 0) & (metric <= 1)
            elif is_elongation:
                mask = mask & (metric >= 1) & (metric < 1e6)
            x = size[mask]
            y = metric[mask]
            if x.size > 0:
                _hist2d(ax, x, y, bins=80, xlog=True, ylog=is_elongation, xlabel='cell_count', ylabel=label)
                _plot_binned_mean(ax, x, y, n_bins=50, xlog=True)
                if ylim and not is_elongation:
                    ax.set_ylim(ylim)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{base}_shape_vs_size.png"), bbox_inches='tight')
        plt.close(fig)

    # 8) Axis ratios vs size (b/a, c/a, and c/b)
    axis_ratios = d.get('axis_ratios')
    if axis_ratios is not None and axis_ratios.ndim == 2 and axis_ratios.shape[1] >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
        ba = np.asarray(axis_ratios[:, 0], dtype=np.float64)
        ca = np.asarray(axis_ratios[:, 1], dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            cb = np.divide(ca, ba, out=np.full_like(ca, np.nan), where=ba > 0)
        ratio_series = [
            ('b/a (intermediate/major)', ba),
            ('c/a (minor/major)', ca),
            ('c/b (minor/intermediate)', cb),
        ]
        for ax, (rlabel, ratio) in zip(axes, ratio_series):
            mask = np.isfinite(size) & np.isfinite(ratio) & (size > 0) & shape_metrics_valid
            mask = mask & (ratio > 0) & (ratio <= 1)
            x = size[mask]
            y = ratio[mask]
            if x.size > 0:
                _hist2d(ax, x, y, bins=80, xlog=True, ylog=False, xlabel='cell_count', ylabel=rlabel)
                _plot_binned_mean(ax, x, y, n_bins=50, xlog=True)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{base}_axis_ratios_vs_size.png"), bbox_inches='tight')
        plt.close(fig)

    # 9) Minkowski shapefinders vs size
    # Only plot if we have the shapefinder data (computed for interior clumps)
    minkowski_metrics = [
        ('thickness', 'Thickness T', True),   # log scale
        ('breadth', 'Breadth B', True),       # log scale
        ('length', 'Length L', True),         # log scale
        ('planarity', 'Planarity P', False),  # linear 0-1
        ('filamentarity', 'Filamentarity F', False),  # linear 0-1
    ]
    has_minkowski = any(d.get(m[0]) is not None for m in minkowski_metrics)
    if has_minkowski:
        # Check how many clumps have valid Minkowski data
        minkowski_computed = d.get('minkowski_computed')
        if minkowski_computed is not None:
            minkowski_computed = np.asarray(minkowski_computed, dtype=bool)
            n_computed = minkowski_computed.sum()
            n_total = minkowski_computed.shape[0]
        else:
            # Fallback: count finite thickness values
            thickness = d.get('thickness')
            if thickness is not None:
                n_computed = np.isfinite(thickness).sum()
                n_total = thickness.shape[0]
            else:
                n_computed, n_total = 0, 0

        if n_computed > 10:  # Only plot if we have enough data
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
            axes = axes.flatten()

            for ax, (key, label, use_log) in zip(axes[:5], minkowski_metrics):
                metric = d.get(key)
                if metric is None:
                    ax.text(0.5, 0.5, f"No {key} data", ha='center', va='center', transform=ax.transAxes)
                    continue
                if use_log:
                    mask = np.isfinite(size) & np.isfinite(metric) & (size > 0) & (metric > 0)
                else:
                    mask = np.isfinite(size) & np.isfinite(metric) & (size > 0) & (metric >= 0) & (metric <= 1)
                if minkowski_computed is not None:
                    mask = mask & minkowski_computed
                x = size[mask]
                y = metric[mask]
                if x.size > 10:
                    _hist2d(ax, x, y, bins=60, xlog=True, ylog=use_log, xlabel='cell_count', ylabel=label)
                    if not use_log:
                        ax.set_ylim(0, 1)
                else:
                    ax.text(0.5, 0.5, f"Insufficient data\n({x.size} points)", ha='center', va='center', transform=ax.transAxes)

            # 6th panel: Euler characteristic histogram
            euler = d.get('euler_characteristic')
            ax = axes[5]
            if euler is not None:
                euler_finite = euler[np.isfinite(euler)]
                if euler_finite.size > 10:
                    # Clip to percentile range to avoid outliers dominating the view
                    p1, p99 = np.percentile(euler_finite, [1, 99])
                    # Ensure we include χ=1 (sphere) and χ=0 (torus) reference lines
                    xmin = min(p1, -2)
                    xmax = max(p99, 3)
                    euler_clipped = euler_finite[(euler_finite >= xmin) & (euler_finite <= xmax)]
                    n_clipped = euler_finite.size - euler_clipped.size
                    ax.hist(euler_clipped, bins=np.arange(xmin, xmax + 1, 1), histtype='stepfilled',
                            alpha=0.85, edgecolor='black', linewidth=0.5)
                    ax.set_xlabel('Euler characteristic χ')
                    ax.set_ylabel('Count')
                    ax.axvline(1, color='red', linestyle='--', alpha=0.7, label='χ=1 (sphere)')
                    ax.axvline(0, color='orange', linestyle='--', alpha=0.7, label='χ=0 (torus)')
                    ax.legend(fontsize=8)
                    if n_clipped > 0:
                        ax.text(0.98, 0.98, f'{n_clipped:,} outliers clipped', fontsize=7,
                                ha='right', va='top', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "Insufficient Euler data", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No Euler data", ha='center', va='center', transform=ax.transAxes)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{base}_minkowski_vs_size.png"), bbox_inches='tight')
            plt.close(fig)

    # 10) Planarity-Filamentarity (P-F) diagram
    planarity = d.get('planarity')
    filamentarity = d.get('filamentarity')
    if planarity is not None and filamentarity is not None:
        mask = np.isfinite(planarity) & np.isfinite(filamentarity)
        mask = mask & (planarity >= 0) & (planarity <= 1) & (filamentarity >= 0) & (filamentarity <= 1)
        if 'minkowski_computed' in d:
            mask = mask & np.asarray(d['minkowski_computed'], dtype=bool)
        P = planarity[mask]
        F = filamentarity[mask]
        if P.size > 10:
            fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
            _hist2d(ax, P, F, bins=60, xlog=False, ylog=False, xlabel='Planarity P', ylabel='Filamentarity F')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            # Add reference points
            ax.plot(0, 0, 'r*', markersize=12, label='Sphere (P=0, F=0)')
            ax.plot(1, 0, 'g^', markersize=10, label='Pancake (P=1, F=0)')
            ax.plot(0, 1, 'bs', markersize=10, label='Filament (P=0, F=1)')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{base}_PF_diagram.png"), bbox_inches='tight')
            plt.close(fig)

    # 11) Integrated curvature vs size
    curvature = d.get('integrated_curvature')
    if curvature is not None:
        mask = np.isfinite(size) & np.isfinite(curvature) & (size > 0) & (curvature > 0)
        if 'minkowski_computed' in d:
            mask = mask & np.asarray(d['minkowski_computed'], dtype=bool)
        x = size[mask]
        y = curvature[mask]
        if x.size > 10:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            _hist2d(ax, x, y, bins=60, xlog=True, ylog=True, xlabel='cell_count', ylabel='Integrated curvature C')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{base}_curvature_vs_size.png"), bbox_inches='tight')
            plt.close(fig)

    print(f"Wrote PNGs to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to per-node or master npz')
    ap.add_argument('--outdir', default=None, help='output directory for PNGs (default next to input)')
    ap.add_argument('--prefix', default=None, help='prefix for output filenames (default: input basename)')
    ap.add_argument('--use-volume', action='store_true', help='use volume as clump size (default cell_count)')
    ap.add_argument('--mass-weighted', action='store_true', help='use mass-weighted stds')
    ap.add_argument('--compare', default=None,
                    help='optional secondary npz to compare size histogram against (stitched vs unstitched)')
    ap.add_argument('--compare-labels', nargs=2, metavar=('PRIMARY', 'SECONDARY'),
                    help='legend labels for --compare (defaults to basenames)')
    ap.add_argument('--compare-outdir', default=None,
                    help='output directory for comparison plot (defaults to --outdir)')
    ap.add_argument('--compensated-cumulative', action='store_true',
                    help='use compensated cumulative distributions for size/mass/area panels')
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.input) or '.'
    make_pngs(
        args.input,
        outdir,
        use_volume=args.use_volume,
        mass_weighted=args.mass_weighted,
        prefix=args.prefix,
        compensated_cumulative=args.compensated_cumulative,
    )

    if args.compare:
        compare_outdir = args.compare_outdir or outdir
        labels = tuple(args.compare_labels) if args.compare_labels else None
        plot_histogram_comparison(args.input, args.compare, compare_outdir, args.use_volume, labels)


if __name__ == '__main__':
    main()
