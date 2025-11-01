from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
        vx_std = d.get('vx_std_massw', None)
        vy_std = d.get('vy_std_massw', None)
        vz_std = d.get('vz_std_massw', None)
    else:
        vx_std = d.get('vx_std', None)
        vy_std = d.get('vy_std', None)
        vz_std = d.get('vz_std', None)

    if vx_std is None or vy_std is None or vz_std is None:
        raise KeyError('Velocity std fields not found in npz')
    vdisp = np.sqrt(vx_std**2 + vy_std**2 + vz_std**2)

    os.makedirs(outdir, exist_ok=True)
    base = prefix or (os.path.splitext(os.path.basename(npz_path))[0])

    # 1) Size histogram
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    # Choose bins: integer-rounded geometric edges for integer sizes; logspace for floats
    if (np.issubdtype(size.dtype, np.integer)) or np.allclose(size, np.round(size)):
        r_min = int(max(1, np.nanmin(size)))
        r_max = int(np.nanmax(size))
        edges = find_ell_bin_edges(r_min, r_max, n_ell_bins=60)
        ax.hist(size, bins=edges, histtype='stepfilled', alpha=0.85)
    else:
        lo = np.nanmin(size[size > 0]) if np.any(size > 0) else 1.0
        hi = np.nanmax(size)
        edges = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.hist(size, bins=edges, histtype='stepfilled', alpha=0.85)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('clump size ({})'.format('volume' if use_volume else 'cell_count'))
    ax.set_ylabel('count')
    ax.set_title('Clump size distribution (K={})'.format(size.shape[0]))
    fig.savefig(os.path.join(outdir, f"{base}_size_hist.png"), bbox_inches='tight')
    plt.close(fig)

    # 2) Size vs velocity dispersion
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    # X-axis edges consistent with size histogram
    if (np.issubdtype(size.dtype, np.integer)) or np.allclose(size, np.round(size)):
        r_min = int(max(1, np.nanmin(size)))
        r_max = int(np.nanmax(size))
        xedges = find_ell_bin_edges(r_min, r_max, n_ell_bins=60)
    else:
        lo = np.nanmin(size[size > 0]) if np.any(size > 0) else 1.0
        hi = np.nanmax(size)
        xedges = np.logspace(np.log10(lo), np.log10(hi), 60)
    _hist2d(ax, size, vdisp, bins=100, xlog=True, ylog=True,
            xlabel='clump size ({})'.format('volume' if use_volume else 'cell_count'),
            ylabel='velocity dispersion (std |v|)', xedges=xedges)
    ax.set_title('Size vs velocity dispersion')
    fig.savefig(os.path.join(outdir, f"{base}_size_vs_vdisp.png"), bbox_inches='tight')
    plt.close(fig)

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
            ratio = area_pos / np.power(vol, 2.0 / 3.0)
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
                        xlabel='clump volume', ylabel='area / volume$^{2/3}$', xedges=xedges)
        ax.set_title('Normalized area vs volume')
        fig.savefig(os.path.join(outdir, f"{base}_area_over_vol23_vs_volume.png"), bbox_inches='tight')
        plt.close(fig)

    print(f"Wrote PNGs to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to per-node or master npz')
    ap.add_argument('--outdir', default=None, help='output directory for PNGs (default next to input)')
    ap.add_argument('--use-volume', action='store_true', help='use volume as clump size (default cell_count)')
    ap.add_argument('--mass-weighted', action='store_true', help='use mass-weighted stds')
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.input) or '.'
    make_pngs(args.input, outdir, use_volume=args.use_volume, mass_weighted=args.mass_weighted)


if __name__ == '__main__':
    main()
