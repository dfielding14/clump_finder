#!/usr/bin/env python
"""
box_counting.py

Box-counting algorithm for measuring the fractal dimension of 3D structures.

The box-counting dimension is defined as:
    D = lim(ε→0) [log N(ε) / log(1/ε)]

where N(ε) is the number of boxes of size ε needed to cover the object.

In practice, we fit log N(ε) vs log(1/ε) over a range of box sizes,
and the slope gives the fractal dimension.

For a solid 3D object: D = 3
For a 2D surface: D = 2
For a fractal surface (e.g., turbulent interface): 2 < D < 3
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy import stats
from scipy import ndimage


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels from a 3D binary mask.

    Surface voxels are those that are True and adjacent to at least one False voxel.

    Parameters
    ----------
    mask : 3D bool array
        Binary mask of the object

    Returns
    -------
    3D bool array with True only at surface voxels
    """
    mask = mask.astype(bool)

    # A voxel is on the surface if it's True but has at least one False neighbor
    # This is equivalent to: mask AND (dilated(NOT mask))
    # Or: mask AND NOT eroded(mask)
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded

    return surface


def box_count_3d(mask: np.ndarray, box_sizes: Optional[np.ndarray] = None,
                 min_box_size: int = 1, max_box_size: Optional[int] = None) -> Dict:
    """Count boxes needed to cover a 3D binary mask at different scales.

    Parameters
    ----------
    mask : 3D bool array
        Binary mask of the object
    box_sizes : array, optional
        Specific box sizes to use. If None, uses powers of 2.
    min_box_size : int
        Minimum box size in voxels
    max_box_size : int, optional
        Maximum box size. If None, uses min(shape) // 2.

    Returns
    -------
    dict with:
        box_sizes : array of box sizes used
        counts : array of box counts N(ε) for each size
        log_inv_size : log(1/ε) values
        log_counts : log(N) values
    """
    mask = mask.astype(bool)
    shape = np.array(mask.shape)

    if max_box_size is None:
        max_box_size = min(shape) // 2

    if box_sizes is None:
        # Use powers of 2 from min to max
        sizes = []
        s = min_box_size
        while s <= max_box_size:
            sizes.append(s)
            s *= 2
        box_sizes = np.array(sizes, dtype=int)

    counts = np.zeros(len(box_sizes), dtype=np.int64)

    for i, box_size in enumerate(box_sizes):
        counts[i] = _count_boxes(mask, box_size)

    # Filter out zeros (can happen if box_size > object extent)
    valid = counts > 0

    return {
        'box_sizes': box_sizes[valid],
        'counts': counts[valid],
        'log_inv_size': np.log(1.0 / box_sizes[valid].astype(float)),
        'log_counts': np.log(counts[valid].astype(float))
    }


def _count_boxes(mask: np.ndarray, box_size: int) -> int:
    """Count non-empty boxes of given size covering the mask.

    Uses efficient reshaping when box_size divides evenly,
    otherwise falls back to strided iteration.
    """
    shape = np.array(mask.shape)

    # Pad to make dimensions divisible by box_size
    pad_shape = ((box_size - (s % box_size)) % box_size for s in shape)
    padded = np.pad(mask, [(0, p) for p in pad_shape], mode='constant',
                    constant_values=False)

    new_shape = np.array(padded.shape)
    n_boxes = new_shape // box_size

    # Reshape into boxes and check if any voxel is True
    reshaped = padded.reshape(
        n_boxes[0], box_size,
        n_boxes[1], box_size,
        n_boxes[2], box_size
    )

    # A box is "occupied" if it contains at least one True voxel
    box_occupied = reshaped.any(axis=(1, 3, 5))

    return int(np.sum(box_occupied))


def fit_box_counting(bc_result: Dict, min_points: int = 3) -> Dict:
    """Fit the box-counting data to extract fractal dimension.

    Parameters
    ----------
    bc_result : dict
        From box_count_3d
    min_points : int
        Minimum number of points needed for fit

    Returns
    -------
    dict with:
        valid : bool
        dimension : fractal dimension D
        dimension_err : standard error on D
        intercept : fit intercept
        r_squared : coefficient of determination
        n_points : number of points in fit
    """
    log_inv_eps = bc_result['log_inv_size']
    log_N = bc_result['log_counts']

    if len(log_inv_eps) < min_points:
        return {
            'valid': False,
            'reason': 'insufficient_points',
            'n_points': len(log_inv_eps)
        }

    # Linear regression: log(N) = D * log(1/ε) + c
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_inv_eps, log_N)

    return {
        'valid': True,
        'dimension': slope,
        'dimension_err': std_err,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'n_points': len(log_inv_eps)
    }


def box_counting_analysis(mask: np.ndarray,
                          min_box_size: int = 1,
                          max_box_size: Optional[int] = None,
                          surface_only: bool = True) -> Dict:
    """Complete box-counting analysis for a single 3D mask.

    Parameters
    ----------
    mask : 3D bool array
        Binary mask of the object
    min_box_size : int
        Minimum box size
    max_box_size : int, optional
        Maximum box size
    surface_only : bool
        If True, measure dimension of the surface (2 < D < 3 for fractal).
        If False, measure dimension of the volume (D ~ 3 for solid).

    Returns
    -------
    dict with box counting results and fitted dimension
    """
    if surface_only:
        # Extract surface voxels for fractal dimension of the interface
        target = extract_surface(mask)
    else:
        target = mask.astype(bool)

    bc = box_count_3d(target, min_box_size=min_box_size, max_box_size=max_box_size)
    fit = fit_box_counting(bc)

    return {
        'box_sizes': bc['box_sizes'],
        'counts': bc['counts'],
        'log_inv_size': bc['log_inv_size'],
        'log_counts': bc['log_counts'],
        'dimension': fit.get('dimension', np.nan),
        'dimension_err': fit.get('dimension_err', np.nan),
        'r_squared': fit.get('r_squared', np.nan),
        'valid': fit.get('valid', False),
        'n_points': fit.get('n_points', 0),
        'surface_only': surface_only
    }


def analyze_largest_clumps(labels: np.ndarray,
                           n_clumps: int = 5,
                           min_volume: int = 1000,
                           min_box_size: int = 1,
                           max_box_size: Optional[int] = None,
                           surface_only: bool = True,
                           verbose: bool = True) -> Dict:
    """Run box-counting analysis on the N largest clumps.

    Parameters
    ----------
    labels : 3D int array
        Label array (0=background, 1..K = clumps)
    n_clumps : int
        Number of largest clumps to analyze
    min_volume : int
        Minimum volume (voxels) to consider
    min_box_size : int
        Minimum box size for box counting
    max_box_size : int, optional
        Maximum box size (if None, uses bounding box size / 2)
    surface_only : bool
        If True (default), measure fractal dimension of the SURFACE.
        If False, measure dimension of the volume.
    verbose : bool
        Print progress

    Returns
    -------
    dict with:
        clump_ids : IDs of analyzed clumps
        volumes : volumes in voxels
        bounding_boxes : (min, max) coordinates for each clump
        dimensions : fractal dimensions of the surface (2 < D < 3 for fractal)
        dimension_errs : standard errors
        r_squared : fit quality
        box_counts : list of (box_sizes, counts) for each clump
    """
    from scipy import ndimage

    # Find all unique clump IDs (excluding background)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return {'valid': False, 'reason': 'no_clumps'}

    # Calculate volumes
    volumes = ndimage.sum_labels(np.ones_like(labels), labels, unique_ids)
    volumes = np.array(volumes)

    # Filter by minimum volume
    valid_mask = volumes >= min_volume
    valid_ids = unique_ids[valid_mask]
    valid_volumes = volumes[valid_mask]

    if len(valid_ids) == 0:
        return {'valid': False, 'reason': 'no_clumps_above_threshold'}

    # Sort by volume (largest first) and take top N
    sort_idx = np.argsort(valid_volumes)[::-1]
    selected_ids = valid_ids[sort_idx[:n_clumps]]
    selected_volumes = valid_volumes[sort_idx[:n_clumps]]

    if verbose:
        mode = "SURFACE" if surface_only else "VOLUME"
        print(f"Analyzing {len(selected_ids)} largest clumps (of {len(valid_ids)} above threshold)")
        print(f"Measuring {mode} fractal dimension")

    results = {
        'clump_ids': [],
        'volumes': [],
        'bounding_boxes': [],
        'dimensions': [],
        'dimension_errs': [],
        'r_squared': [],
        'box_counts': [],
        'surface_only': surface_only,
        'valid': True
    }

    for i, (cid, vol) in enumerate(zip(selected_ids, selected_volumes)):
        if verbose:
            print(f"  Clump {cid} (volume={vol} voxels)...")

        # Extract mask for this clump
        mask = (labels == cid)

        # Find bounding box
        where = np.argwhere(mask)
        bb_min = where.min(axis=0)
        bb_max = where.max(axis=0) + 1

        # Extract submask (for efficiency)
        submask = mask[bb_min[0]:bb_max[0],
                       bb_min[1]:bb_max[1],
                       bb_min[2]:bb_max[2]]

        # Determine max box size from bounding box
        bb_size = bb_max - bb_min
        clump_max_box = min(bb_size) // 2 if max_box_size is None else max_box_size
        clump_max_box = max(clump_max_box, 2)  # At least 2

        # Run box counting on the surface
        bc_result = box_counting_analysis(submask,
                                          min_box_size=min_box_size,
                                          max_box_size=clump_max_box,
                                          surface_only=surface_only)

        results['clump_ids'].append(int(cid))
        results['volumes'].append(int(vol))
        results['bounding_boxes'].append((tuple(bb_min), tuple(bb_max)))
        results['dimensions'].append(bc_result['dimension'])
        results['dimension_errs'].append(bc_result['dimension_err'])
        results['r_squared'].append(bc_result['r_squared'])
        results['box_counts'].append({
            'box_sizes': bc_result['box_sizes'],
            'counts': bc_result['counts']
        })

        if verbose:
            if bc_result['valid']:
                print(f"    D = {bc_result['dimension']:.3f} +/- {bc_result['dimension_err']:.3f} "
                      f"(R² = {bc_result['r_squared']:.4f})")
            else:
                print(f"    Box counting failed")

    # Convert to arrays
    results['clump_ids'] = np.array(results['clump_ids'])
    results['volumes'] = np.array(results['volumes'])
    results['dimensions'] = np.array(results['dimensions'])
    results['dimension_errs'] = np.array(results['dimension_errs'])
    results['r_squared'] = np.array(results['r_squared'])

    # Summary statistics
    valid_dims = results['dimensions'][~np.isnan(results['dimensions'])]
    if len(valid_dims) > 0:
        results['mean_dimension'] = float(np.mean(valid_dims))
        results['std_dimension'] = float(np.std(valid_dims))
        if verbose:
            print(f"\nMean fractal dimension: {results['mean_dimension']:.3f} "
                  f"+/- {results['std_dimension']:.3f}")

    return results


def plot_box_counting(results: Dict, output_path: str):
    """Generate box-counting diagnostic plots.

    Parameters
    ----------
    results : dict
        From analyze_largest_clumps
    output_path : str
        Path to save the plot
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    n_clumps = len(results['clump_ids'])

    # Create figure with subplots
    n_cols = min(3, n_clumps)
    n_rows = (n_clumps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_clumps == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_clumps):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        bc = results['box_counts'][i]
        box_sizes = bc['box_sizes']
        counts = bc['counts']

        # Plot data points
        ax.loglog(1.0 / box_sizes, counts, 'bo', markersize=8)

        # Plot fit line
        D = results['dimensions'][i]
        D_err = results['dimension_errs'][i]
        if not np.isnan(D):
            x_fit = np.array([1.0 / box_sizes.max(), 1.0 / box_sizes.min()])
            # log(N) = D * log(1/eps) + c => N = c' * (1/eps)^D
            c = counts[0] / (1.0 / box_sizes[0])**D
            y_fit = c * x_fit**D
            ax.loglog(x_fit, y_fit, 'r-', lw=2,
                      label=f'D = {D:.2f} ± {D_err:.2f}')

        ax.set_xlabel('1 / box size')
        ax.set_ylabel('N(boxes)')
        ax.set_title(f'Clump {results["clump_ids"][i]}\n'
                     f'V = {results["volumes"][i]} voxels')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_clumps, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    # Add reference lines annotation
    surface_mode = results.get('surface_only', True)
    if surface_mode:
        ref_text = 'Reference: D=2.0 (smooth surface), D=2.33 (Kolmogorov), D=2.67 (fractal D_s=8/3)'
    else:
        ref_text = 'Reference: D=2 (sheet), D=3 (solid volume)'
    fig.text(0.02, 0.02, ref_text, fontsize=9, style='italic')

    mode_str = "Surface" if surface_mode else "Volume"
    plt.suptitle(f'Box-Counting {mode_str} Fractal Dimension\n'
                 f'Mean D = {results.get("mean_dimension", np.nan):.2f} ± '
                 f'{results.get("std_dimension", 0):.2f}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved box-counting plot to {output_path}")


if __name__ == '__main__':
    # Quick test with a synthetic fractal-like object
    import argparse

    parser = argparse.ArgumentParser(description='Box-counting fractal dimension')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to npz file with labels array')
    parser.add_argument('--labels-key', type=str, default='labels',
                        help='Key for labels in npz file')
    parser.add_argument('--n-clumps', type=int, default=5,
                        help='Number of largest clumps to analyze')
    parser.add_argument('--min-volume', type=int, default=1000,
                        help='Minimum clump volume in voxels')
    parser.add_argument('--output', type=str, default='box_counting.npz',
                        help='Output file path')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output plot path')

    args = parser.parse_args()

    # Load labels
    print(f"Loading labels from {args.labels}...")
    data = np.load(args.labels)
    if args.labels_key in data:
        labels = data[args.labels_key]
    else:
        labels = data[list(data.keys())[0]]

    print(f"Labels shape: {labels.shape}, K={labels.max()}")

    # Run analysis
    results = analyze_largest_clumps(
        labels,
        n_clumps=args.n_clumps,
        min_volume=args.min_volume
    )

    if results['valid']:
        # Save results
        np.savez(args.output,
                 clump_ids=results['clump_ids'],
                 volumes=results['volumes'],
                 dimensions=results['dimensions'],
                 dimension_errs=results['dimension_errs'],
                 r_squared=results['r_squared'],
                 mean_dimension=results.get('mean_dimension', np.nan),
                 std_dimension=results.get('std_dimension', np.nan))
        print(f"\nSaved results to {args.output}")

        # Generate plot
        plot_path = args.plot or args.output.replace('.npz', '.png')
        plot_box_counting(results, plot_path)
    else:
        print(f"Analysis failed: {results.get('reason', 'unknown')}")
