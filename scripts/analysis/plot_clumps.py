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
    if label:
        ax.set_title(label)


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


def plot_histogram_comparison(primary_path: str, secondary_path: str, outdir: str,
                              use_volume: bool, labels: Optional[Tuple[str, str]] = None) -> None:
    size_primary = _load_size(primary_path, use_volume)
    size_secondary = _load_size(secondary_path, use_volume)

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
        where=(spec_primary > 0),
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
    label_primary = f"{label_primary} (∑ volume = {total_primary:.3e})"
    label_secondary = f"{label_secondary} (∑ volume = {total_secondary:.3e})"

    centers = bins[:-1]
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, dpi=150)

    ax_top.step(centers, spec_primary, where='post', label=label_primary, color='tab:blue')
    ax_top.step(centers, spec_secondary, where='post', label=label_secondary, color='tab:orange')
    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel('V · dN / dlog V')
    ax_top.legend()
    ax_top.set_title('Clump size histogram comparison')

    ax_bottom.step(centers, ratio, where='post', color='tab:purple')
    ax_bottom.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax_bottom.set_xscale('log')
    ax_bottom.set_xlabel('V [Δx^3]')
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


def make_pngs(npz_path: str, outdir: str, use_volume: bool = False, mass_weighted: bool = False, prefix: str | None = None):
    d = _load(npz_path)
    size = d['volume'] if use_volume else d['cell_count']
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

    # 1) Size spectrum: V * dN/dlogV
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    # Choose bins: integer-rounded geometric edges for integer sizes; logspace for floats
    if (np.issubdtype(size.dtype, np.integer)) or np.allclose(size, np.round(size)):
        r_min = int(max(1, np.nanmin(size)))
        r_max = int(np.nanmax(size))
        edges = find_ell_bin_edges(r_min, r_max, n_ell_bins=60)
    else:
        lo = np.nanmin(size[size > 0]) if np.any(size > 0) else 1.0
        hi = np.nanmax(size)
        edges = np.logspace(np.log10(lo), np.log10(hi), 60)
    counts, _ = np.histogram(size, bins=edges)
    log_width = np.log(edges[1:]) - np.log(edges[:-1])
    v_mid = np.sqrt(edges[1:] * edges[:-1])
    spectrum = np.divide(v_mid * counts, log_width,
                         out=np.full_like(counts, np.nan, dtype=np.float64),
                         where=log_width > 0)
    ax.step(edges[:-1], spectrum, where='post', alpha=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('V [Δx^3]')
    ax.set_ylabel('V · dN / dlog V')
    ax.set_title('Clump size spectrum (K={})'.format(size.shape[0]))
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
        ax.set_title('Normalized area vs volume (8/9 power)')
        fig.savefig(os.path.join(outdir, f"{base}_area_over_vol89_vs_volume.png"), bbox_inches='tight')
        plt.close(fig)

    # 4) Velocity dispersion vs volume - REMOVED due to numerical issues

    # 5) Mass spectrum: M * dN/dlogM
    mass = d.get('mass')
    if mass is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        mass_pos = mass[np.isfinite(mass) & (mass > 0)]
        if mass_pos.size > 0:
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
            ax.set_ylabel('M · dN / dlog M')
            ax.set_title('Clump mass spectrum')
        else:
            ax.text(0.5, 0.5, "No mass data", ha='center', va='center')
        fig.savefig(os.path.join(outdir, f"{base}_mass_spectrum.png"), bbox_inches='tight')
        plt.close(fig)

    # 6) Surface area spectrum: A * dN/dlogA
    area = d.get('area')
    if area is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        area_pos = area[np.isfinite(area) & (area > 0)]
        if area_pos.size > 0:
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
            ax.set_ylabel('A · dN / dlog A')
            ax.set_title('Clump surface area spectrum')
        else:
            ax.text(0.5, 0.5, "No area data", ha='center', va='center')
        fig.savefig(os.path.join(outdir, f"{base}_area_spectrum.png"), bbox_inches='tight')
        plt.close(fig)

    # 7) Shape metrics vs size (sphericity, compactness, triaxiality, elongation)
    shape_metrics = [
        ('sphericity', 'Sphericity', (0, 1)),
        ('compactness', 'Compactness', (0, 1)),
        ('triaxiality', 'Triaxiality T', (0, 1)),
        ('elongation', 'Elongation', (1, 1e3)),  # reasonable range for axis ratio
    ]
    has_shape = any(d.get(m[0]) is not None for m in shape_metrics)
    if has_shape:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
        axes = axes.flatten()
        for ax, (key, label, ylim) in zip(axes, shape_metrics):
            metric = d.get(key)
            if metric is None:
                ax.text(0.5, 0.5, f"No {key} data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{label} vs Size')
                continue
            # For elongation, use log scale and filter reasonable values
            is_elongation = (key == 'elongation')
            if is_elongation:
                mask = np.isfinite(size) & np.isfinite(metric) & (size > 0) & (metric > 0) & (metric < 1e6)
            else:
                mask = np.isfinite(size) & np.isfinite(metric) & (size > 0)
            x = size[mask]
            y = metric[mask]
            if x.size > 0:
                _hist2d(ax, x, y, bins=80, xlog=True, ylog=is_elongation, xlabel='cell_count', ylabel=label)
                if ylim and not is_elongation:
                    ax.set_ylim(ylim)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} vs Size')
        fig.suptitle('Shape metrics vs clump size', y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{base}_shape_vs_size.png"), bbox_inches='tight')
        plt.close(fig)

    # 8) Axis ratios vs size (b/a and c/a)
    axis_ratios = d.get('axis_ratios')
    if axis_ratios is not None and axis_ratios.ndim == 2 and axis_ratios.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        ratio_labels = [('b/a (intermediate/major)', 0), ('c/a (minor/major)', 1)]
        for ax, (rlabel, idx) in zip(axes, ratio_labels):
            ratio = axis_ratios[:, idx]
            mask = np.isfinite(size) & np.isfinite(ratio) & (size > 0) & (ratio > 0)
            x = size[mask]
            y = ratio[mask]
            if x.size > 0:
                _hist2d(ax, x, y, bins=80, xlog=True, ylog=False, xlabel='cell_count', ylabel=rlabel)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{rlabel} vs Size')
        fig.suptitle('Axis ratios vs clump size', y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{base}_axis_ratios_vs_size.png"), bbox_inches='tight')
        plt.close(fig)

    # 9) Minkowski functionals - REMOVED
    # Minkowski shapefinder values computed in stitcher are unreliable because:
    # - Euler characteristic isn't additive across stitched fragments
    # - Integrated curvature requires voxel-level boundary information not preserved through stitching

    print(f"Wrote PNGs to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to per-node or master npz')
    ap.add_argument('--outdir', default=None, help='output directory for PNGs (default next to input)')
    ap.add_argument('--use-volume', action='store_true', help='use volume as clump size (default cell_count)')
    ap.add_argument('--mass-weighted', action='store_true', help='use mass-weighted stds')
    ap.add_argument('--compare', default=None,
                    help='optional secondary npz to compare size histogram against (stitched vs unstitched)')
    ap.add_argument('--compare-labels', nargs=2, metavar=('PRIMARY', 'SECONDARY'),
                    help='legend labels for --compare (defaults to basenames)')
    ap.add_argument('--compare-outdir', default=None,
                    help='output directory for comparison plot (defaults to --outdir)')
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.input) or '.'
    make_pngs(args.input, outdir, use_volume=args.use_volume, mass_weighted=args.mass_weighted)

    if args.compare:
        compare_outdir = args.compare_outdir or outdir
        labels = tuple(args.compare_labels) if args.compare_labels else None
        plot_histogram_comparison(args.input, args.compare, compare_outdir, args.use_volume, labels)


if __name__ == '__main__':
    main()
