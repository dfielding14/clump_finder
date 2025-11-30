from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('agg')

from local_label import label_3d
import metrics as M
from matplotlib.colors import LogNorm

try:
    from plot_clumps import make_pngs
    _HAS_PLOT_CLUMPS = True
except ImportError:
    try:
        from scripts.analysis.plot_clumps import make_pngs
        _HAS_PLOT_CLUMPS = True
    except ImportError:
        _HAS_PLOT_CLUMPS = False
        make_pngs = None


def generate_powerlaw_field(shape=(128, 128, 128), beta=-5.0/3.0, seed=None,
                            k_min=4, k_max_fraction=1.0/8.0):
    """Generate a 3-D real field with isotropic power spectrum P(k) ~ k^beta.

    Parameters
    ----------
    shape : tuple
        Grid dimensions (nz, ny, nx)
    beta : float
        Power spectrum slope
    seed : int, optional
        Random seed
    k_min : float
        Minimum wavenumber (cuts large-scale modes). Default 4.
    k_max_fraction : float
        Fraction of Nyquist to use as k_max cutoff. Default 1/8 means
        k_max = N/8, so minimum structure size is ~8 cells.

    Returns zero-mean, unit-variance field (float32).
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    kk[0, 0, 0] = 1.0  # avoid div0 at DC

    # k_max cutoff: zero out modes with |k| > k_max
    # This ensures minimum structure size is ~8 cells
    N_ref = max(nx, ny, nz)
    k_max = N_ref * k_max_fraction

    # Amplitude ~ k^{beta/2}
    amp = kk ** (beta / 2.0)
    # Apply k_min and k_max cutoffs
    amp[kk < k_min] = 0.0
    amp[kk > k_max] = 0.0

    # Random complex field with Hermitian symmetry
    phase = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    F = amp * phase
    F[0, 0, 0] = 0.0

    # Inverse FFT to real space (real-valued field)
    f = np.fft.ifftn(F).real
    f -= f.mean()
    s = f.std()
    if s > 0:
        f /= s
    return f.astype(np.float32)


def generate_anisotropic_field(shape=(128, 128, 128), beta=-5.0/3.0,
                                anisotropy=(1.0, 1.0, 1.0), seed=None,
                                k_min=4, k_max_fraction=1.0/8.0):
    """Generate a 3-D real field with anisotropic power spectrum.

    The power spectrum is P(k_eff) ~ k_eff^beta where k_eff is an effective
    wavenumber with anisotropic scaling:

        k_eff = sqrt((kx/ax)^2 + (ky/ay)^2 + (kz/az)^2)

    Larger scaling factors stretch structures along that axis (make them
    elongated in that direction in real space).

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions (nz, ny, nx)
    beta : float
        Power spectrum slope (default -5/3 for Kolmogorov)
    anisotropy : tuple of float
        Scaling factors (ax, ay, az) for k-space. Examples:
        - (1, 1, 1): Isotropic
        - (3, 1, 1): Structures elongated along x (prolate)
        - (3, 3, 1): Structures flattened in z (oblate/pancake)
        - (2, 1.5, 1): Triaxial
    seed : int, optional
        Random seed for reproducibility
    k_min : float
        Minimum wavenumber (cuts large-scale modes). Default 4.
    k_max_fraction : float
        Fraction of Nyquist to use as k_max cutoff. Default 1/8 means
        k_max = N/8, so minimum structure size is ~8 cells.

    Returns
    -------
    ndarray
        Zero-mean, unit-variance field (float32)
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

    # Compute isotropic k for the cutoff
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)

    ax, ay, az = anisotropy
    # Effective wavenumber with anisotropic scaling
    # Dividing k by a larger factor REDUCES the effective k,
    # which INCREASES the amplitude, creating elongated structures
    keff = np.sqrt((KX/ax)**2 + (KY/ay)**2 + (KZ/az)**2)
    keff[0, 0, 0] = 1.0  # avoid div0 at DC

    # k_max cutoff: zero out modes with |k| > k_max
    # Use isotropic k for cutoff to ensure minimum size in all directions
    N_ref = max(nx, ny, nz)
    k_max = N_ref * k_max_fraction

    # Amplitude ~ k_eff^{beta/2}
    amp = keff ** (beta / 2.0)
    # Apply k_min and k_max cutoffs based on isotropic k
    amp[kk < k_min] = 0.0
    amp[kk > k_max] = 0.0

    # Random complex field
    phase = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    F = amp * phase
    F[0, 0, 0] = 0.0

    # Inverse FFT to real space
    f = np.fft.ifftn(F).real
    f -= f.mean()
    s = f.std()
    if s > 0:
        f /= s
    return f.astype(np.float32)


# Predefined anisotropy configurations for batch testing
ANISOTROPY_CONFIGS = {
    'isotropic':     (1.0, 1.0, 1.0),   # Baseline: roughly spherical, T≈0.5
    'prolate':       (3.0, 1.0, 1.0),   # Elongated cigars along x, T→1
    'super_prolate': (8.0, 1.0, 1.0),   # Very elongated filaments along x
    'oblate':        (2.0, 2.0, 1.0),   # Flattened pancakes in z, T→0
    'super_oblate':  (8.0, 8.0, 1.0),   # Very flat sheets in z
    'triaxial_A':    (2.0, 1.5, 1.0),   # Mixed, T≈0.3
    'triaxial_B':    (3.0, 2.0, 1.0),   # Mixed, T≈0.6
}


def rescale_to_range(x: np.ndarray, lo: float, hi: float, robust: bool = True) -> np.ndarray:
    """Rescale array to [lo, hi]. If robust, use 1st–99th percentiles to avoid outliers."""
    x = x.astype(np.float64, copy=False)
    if robust:
        a, b = np.percentile(x, [1, 99])
    else:
        a, b = float(np.min(x)), float(np.max(x))
    if b <= a:
        return np.full_like(x, 0.5 * (lo + hi))
    y = (x - a) / (b - a)
    y = np.clip(y, 0.0, 1.0)
    return (lo + y * (hi - lo)).astype(np.float32)


def run_single_config(N, beta, anisotropy, temp_threshold, connectivity, outdir,
                      config_name=None, seed_base=1):
    """Run clump finding for a single anisotropy configuration.

    Parameters
    ----------
    N : int
        Grid size (N x N x N)
    beta : float
        Power spectrum slope
    anisotropy : tuple of float
        (ax, ay, az) anisotropy factors
    temp_threshold : float
        Temperature threshold for cold gas
    connectivity : int
        Connectivity for labeling (6 or 26)
    outdir : str
        Output directory
    config_name : str, optional
        Name for this configuration (used in output filenames)
    seed_base : int
        Base seed for random number generation

    Returns
    -------
    dict
        Summary statistics for this run
    """
    import matplotlib.pyplot as plt

    dx = dy = dz = 1.0 / N
    origin = (0.0, 0.0, 0.0)

    # Generate density field with anisotropy
    print(f"  Generating anisotropic density field (anisotropy={anisotropy})...")
    f_rho = generate_anisotropic_field((N, N, N), beta=beta,
                                        anisotropy=anisotropy, seed=seed_base)

    # Map to density range [1, 100]
    dens = rescale_to_range(np.exp(f_rho * 0.5), 1.0, 100.0, robust=True)

    # Derive temperature from constant pressure: P ~ rho * T => T ~ 1/rho
    # Normalize so T ranges from 0.01 (high density) to 1.0 (low density)
    T_raw = 1.0 / dens
    temp = rescale_to_range(T_raw, 0.01, 1.0, robust=True)

    # Generate velocity fields (isotropic, independent of density anisotropy)
    velx = generate_powerlaw_field((N, N, N), beta=beta, seed=seed_base + 2)
    vely = generate_powerlaw_field((N, N, N), beta=beta, seed=seed_base + 3)
    velz = generate_powerlaw_field((N, N, N), beta=beta, seed=seed_base + 4)
    velx = rescale_to_range(velx, -1.0, 1.0, robust=True)
    vely = rescale_to_range(vely, -1.0, 1.0, robust=True)
    velz = rescale_to_range(velz, -1.0, 1.0, robust=True)

    # Threshold on temperature
    Tthr = float(temp_threshold)
    mask = temp < Tthr

    # Label connected components
    tile = min(128, N)
    labels = label_3d(mask, tile_shape=(tile, tile, tile),
                      connectivity=connectivity, halo=0)
    K = int(labels.max())
    print(f"  Found {K} clumps")

    if K == 0:
        print(f"  Warning: No clumps found! Try adjusting threshold.")
        return {'K': 0, 'config_name': config_name, 'anisotropy': anisotropy}

    # Compute metrics
    Vc = dx * dy * dz
    cell_count = M.num_cells(labels, K=K)
    vol = M.volumes(cell_count, dx, dy, dz)
    mass = M.masses(labels, dens, dx, dy, dz, K=K)
    cvol, cmass = M.centroids(labels, dens, dx, dy, dz, origin,
                               ((0, N), (0, N), (0, N)), K=K)
    area = M.exposed_area(labels, dx, dy, dz, K=K)

    weight_vol = np.full(dens.shape, Vc, dtype=np.float64)
    weight_mass = (dens.astype(np.float64, copy=False)) * Vc
    pressure = dens * temp

    stats = {}
    for name, arr in (
        ("rho", dens),
        ("T", temp),
        ("vx", velx),
        ("vy", vely),
        ("vz", velz),
        ("pressure", pressure),
    ):
        mu, sd, sk, ku = M.per_label_stats(labels, arr, weights=weight_vol,
                                            K=K, excess_kurtosis=False)
        stats[f"{name}_mean"] = mu
        stats[f"{name}_std"] = sd
        stats[f"{name}_skew"] = sk
        stats[f"{name}_kurt"] = ku

        mu_m, sd_m, sk_m, ku_m = M.per_label_stats(labels, arr, weights=weight_mass,
                                                    K=K, excess_kurtosis=False)
        stats[f"{name}_mean_massw"] = mu_m
        stats[f"{name}_std_massw"] = sd_m
        stats[f"{name}_skew_massw"] = sk_m
        stats[f"{name}_kurt_massw"] = ku_m

    # Shape metrics (mass-weighted covariance)
    W = np.zeros(K + 1, dtype=np.float64)
    Sx = np.zeros(K + 1, dtype=np.float64)
    Sy = np.zeros(K + 1, dtype=np.float64)
    Sz = np.zeros(K + 1, dtype=np.float64)
    Sxx = np.zeros(K + 1, dtype=np.float64)
    Syy = np.zeros(K + 1, dtype=np.float64)
    Szz = np.zeros(K + 1, dtype=np.float64)
    Sxy = np.zeros(K + 1, dtype=np.float64)
    Sxz = np.zeros(K + 1, dtype=np.float64)
    Syz = np.zeros(K + 1, dtype=np.float64)

    xi = origin[0] + (np.arange(0, N) + 0.5) * dx
    yj = origin[1] + (np.arange(0, N) + 0.5) * dy
    zk = origin[2] + (np.arange(0, N) + 0.5) * dz

    for i in range(N):
        L = labels[i, :, :].ravel()
        w = (dens[i, :, :].ravel().astype(np.float64, copy=False)) * Vc
        W += np.bincount(L, weights=w, minlength=K + 1)
        x = xi[i]
        Sx += x * np.bincount(L, weights=w, minlength=K + 1)
        Sxx += (x * x) * np.bincount(L, weights=w, minlength=K + 1)

    for j in range(N):
        L = labels[:, j, :].ravel()
        w = (dens[:, j, :].ravel().astype(np.float64, copy=False)) * Vc
        y = yj[j]
        Sy += y * np.bincount(L, weights=w, minlength=K + 1)
        Syy += (y * y) * np.bincount(L, weights=w, minlength=K + 1)

    for k in range(N):
        L = labels[:, :, k].ravel()
        w = (dens[:, :, k].ravel().astype(np.float64, copy=False)) * Vc
        z = zk[k]
        Sz += z * np.bincount(L, weights=w, minlength=K + 1)
        Szz += (z * z) * np.bincount(L, weights=w, minlength=K + 1)

    for j in range(N):
        y = yj[j]
        for i in range(N):
            L = labels[i, j, :]
            w = (dens[i, j, :].astype(np.float64, copy=False)) * Vc
            if L.size == 0:
                continue
            Sxy += (xi[i] * y) * np.bincount(L, weights=w, minlength=K + 1)

    for k in range(N):
        z = zk[k]
        for i in range(N):
            L = labels[i, :, k]
            w = (dens[i, :, k].astype(np.float64, copy=False)) * Vc
            Sxz += (xi[i] * z) * np.bincount(L, weights=w, minlength=K + 1)

    for k in range(N):
        z = zk[k]
        for j in range(N):
            L = labels[:, j, k]
            w = (dens[:, j, k].astype(np.float64, copy=False)) * Vc
            Syz += (yj[j] * z) * np.bincount(L, weights=w, minlength=K + 1)

    W = W[1:]
    Sx, Sy, Sz = Sx[1:], Sy[1:], Sz[1:]
    Sxx, Syy, Szz = Sxx[1:], Syy[1:], Szz[1:]
    Sxy, Sxz, Syz = Sxy[1:], Sxz[1:], Syz[1:]

    mu_x = Sx / (W + 1e-300)
    mu_y = Sy / (W + 1e-300)
    mu_z = Sz / (W + 1e-300)
    Cxx = Sxx / (W + 1e-300) - mu_x * mu_x
    Cyy = Syy / (W + 1e-300) - mu_y * mu_y
    Czz = Szz / (W + 1e-300) - mu_z * mu_z
    Cxy = Sxy / (W + 1e-300) - mu_x * mu_y
    Cxz = Sxz / (W + 1e-300) - mu_x * mu_z
    Cyz = Syz / (W + 1e-300) - mu_y * mu_z

    principal_axes_lengths = np.zeros((K, 3), dtype=np.float64)
    axis_ratios = np.zeros((K, 2), dtype=np.float64)
    orientation = np.zeros((K, 3, 3), dtype=np.float64)
    triaxiality = np.zeros(K, dtype=np.float64)
    elongation = np.zeros(K, dtype=np.float64)

    for idx in range(K):
        C = np.array([[Cxx[idx], Cxy[idx], Cxz[idx]],
                      [Cxy[idx], Cyy[idx], Cyz[idx]],
                      [Cxz[idx], Cyz[idx], Czz[idx]]], dtype=np.float64)
        C = (C + C.T) * 0.5
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        a = np.sqrt(max(vals[0], 0.0))
        b = np.sqrt(max(vals[1], 0.0))
        c = np.sqrt(max(vals[2], 0.0))
        principal_axes_lengths[idx, :] = (a, b, c)
        axis_ratios[idx, :] = (b / (a + 1e-300), c / (a + 1e-300))
        orientation[idx, :, :] = vecs

        # Triaxiality: T = (a² - b²) / (a² - c²)
        a2, b2, c2 = a**2, b**2, c**2
        if a2 > c2 + 1e-10:
            triaxiality[idx] = (a2 - b2) / (a2 - c2)
        else:
            triaxiality[idx] = 0.5  # degenerate case

        # Elongation: a/c (with sensible minimum for c to avoid huge values)
        # Use at least 1% of a as minimum c to avoid numerical issues
        c_safe = max(c, 0.01 * a, 1e-6 * dx)
        elongation[idx] = a / c_safe

    bbox_ijk = M.compute_bboxes(labels, ((0, N), (0, N), (0, N)), K=K)

    # Save outputs
    os.makedirs(outdir, exist_ok=True)
    rank_ids = np.arange(1, K + 1, dtype=np.int32)

    prefix = f"clumps_{config_name}" if config_name else "clumps"
    out = {
        "labels": labels,  # 3D label array for fractal analysis
        "label_ids": rank_ids,
        "cell_count": cell_count,
        "num_cells": cell_count,
        "volume": vol,
        "mass": mass,
        "area": area,
        "centroid_vol": cvol,
        "centroid_mass": cmass,
        "bbox_ijk": bbox_ijk,
        "voxel_spacing": np.array([dx, dy, dz], dtype=np.float64),
        "origin": np.array(origin, dtype=np.float64),
        "connectivity": np.int32(connectivity),
        "temperature_threshold": np.float64(Tthr),
        "rank": np.int32(0),
        "node_bbox_ijk": np.array([0, N, 0, N, 0, N], dtype=np.int64),
        "periodic": np.array([True, True, True], dtype=bool),
        "principal_axes_lengths": principal_axes_lengths,
        "axis_ratios": axis_ratios,
        "orientation": orientation,
        "triaxiality": triaxiality,
        "elongation": elongation,
        "anisotropy": np.array(anisotropy, dtype=np.float64),
        "config_name": config_name or "custom",
    }
    out.update(stats)

    npz_path = os.path.join(outdir, f"{prefix}_rank00000.npz")
    np.savez(npz_path, **out)
    print(f"  Wrote {npz_path}")

    # Generate plots (if available)
    if _HAS_PLOT_CLUMPS and make_pngs is not None:
        make_pngs(npz_path, outdir, use_volume=False, mass_weighted=False,
                  prefix=prefix)

    # Map slices at mid-planes - create multi-panel figure showing all 3 planes
    mid = N // 2

    # Create 2x3 figure: top row = density, bottom row = temperature
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

    # Density slices
    im0 = axes[0, 0].imshow(dens[mid, :, :], origin='lower', cmap='viridis',
                             norm=LogNorm(vmin=1.0, vmax=100.0))
    axes[0, 0].set_title(f'Density XY (z={mid})')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(dens[:, mid, :], origin='lower', cmap='viridis',
                             norm=LogNorm(vmin=1.0, vmax=100.0))
    axes[0, 1].set_title(f'Density XZ (y={mid})')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('z')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(dens[:, :, mid], origin='lower', cmap='viridis',
                             norm=LogNorm(vmin=1.0, vmax=100.0))
    axes[0, 2].set_title(f'Density YZ (x={mid})')
    axes[0, 2].set_xlabel('y'); axes[0, 2].set_ylabel('z')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    # Temperature slices (cold = blue/dark, hot = yellow/bright in magma)
    im3 = axes[1, 0].imshow(temp[mid, :, :], origin='lower', cmap='magma',
                             norm=LogNorm(vmin=0.01, vmax=1.0))
    axes[1, 0].set_title(f'Temperature XY (z={mid})')
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    im4 = axes[1, 1].imshow(temp[:, mid, :], origin='lower', cmap='magma',
                             norm=LogNorm(vmin=0.01, vmax=1.0))
    axes[1, 1].set_title(f'Temperature XZ (y={mid})')
    axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('z')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    im5 = axes[1, 2].imshow(temp[:, :, mid], origin='lower', cmap='magma',
                             norm=LogNorm(vmin=0.01, vmax=1.0))
    axes[1, 2].set_title(f'Temperature YZ (x={mid})')
    axes[1, 2].set_xlabel('y'); axes[1, 2].set_ylabel('z')
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)

    aniso_str = f"({anisotropy[0]:.1f}, {anisotropy[1]:.1f}, {anisotropy[2]:.1f})"
    fig.suptitle(f'{config_name or "custom"} - Anisotropy {aniso_str}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_slices.png'), bbox_inches='tight')
    plt.close()

    # Also save the mask (cold gas) overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Show temperature with cold gas mask overlay
    for i, (ax, sl_temp, sl_mask, title) in enumerate([
        (axes[0], temp[mid, :, :], mask[mid, :, :], f'XY (z={mid})'),
        (axes[1], temp[:, mid, :], mask[:, mid, :], f'XZ (y={mid})'),
        (axes[2], temp[:, :, mid], mask[:, :, mid], f'YZ (x={mid})'),
    ]):
        ax.imshow(sl_temp, origin='lower', cmap='magma',
                  norm=LogNorm(vmin=0.01, vmax=1.0), alpha=0.7)
        # Overlay cold gas mask in blue
        cold_overlay = np.ma.masked_where(~sl_mask, sl_mask.astype(float))
        ax.imshow(cold_overlay, origin='lower', cmap='Blues', alpha=0.5,
                  vmin=0, vmax=1)
        ax.set_title(f'Cold Gas {title}')
        ax.set_xlabel('col'); ax.set_ylabel('row')

    fig.suptitle(f'{config_name} - Cold gas (T < {Tthr}) overlay', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_cold_gas.png'), bbox_inches='tight')
    plt.close()

    # Return summary statistics
    summary = {
        'config_name': config_name or 'custom',
        'anisotropy': anisotropy,
        'K': K,
        'triaxiality_mean': float(np.mean(triaxiality)),
        'triaxiality_std': float(np.std(triaxiality)),
        'triaxiality_median': float(np.median(triaxiality)),
        'elongation_mean': float(np.mean(elongation)),
        'elongation_std': float(np.std(elongation)),
        'elongation_median': float(np.median(elongation)),
        'total_volume': float(np.sum(vol)),
        'total_mass': float(np.sum(mass)),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(
        description='Generate synthetic turbulence data and find clumps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single isotropic run (default)
  python synthetic_powerlaw_test.py --N 256

  # Single run with custom anisotropy (prolate structures along x)
  python synthetic_powerlaw_test.py --N 256 --anisotropy 3 1 1 --outdir prolate_test

  # Batch mode: run all predefined configurations
  python synthetic_powerlaw_test.py --N 256 --batch-anisotropy --outdir aniso_batch

  # Large-scale test
  python synthetic_powerlaw_test.py --N 512 --anisotropy 5 1 1 --outdir super_prolate
''')
    ap.add_argument('--N', type=int, default=128,
                    help='Grid size N (creates N×N×N domain)')
    ap.add_argument('--beta', type=float, default=-5.0/3.0,
                    help='Power spectrum slope (default -5/3 Kolmogorov)')
    ap.add_argument('--temp-threshold', type=float, default=0.1,
                    help='Temperature threshold for cold gas (default 0.1)')
    ap.add_argument('--connectivity', type=int, default=6,
                    help='Connectivity for labeling: 6 or 26')
    ap.add_argument('--outdir', default='./clump_out_synth',
                    help='Output directory')
    ap.add_argument('--anisotropy', type=float, nargs=3, default=None,
                    metavar=('AX', 'AY', 'AZ'),
                    help='Anisotropy factors (ax, ay, az). Larger values stretch '
                         'structures along that axis. Examples: "3 1 1" for prolate, '
                         '"3 3 1" for oblate.')
    ap.add_argument('--batch-anisotropy', action='store_true',
                    help='Run all predefined anisotropy configurations')
    ap.add_argument('--seed', type=int, default=1,
                    help='Base random seed')
    args = ap.parse_args()

    N = args.N

    if args.batch_anisotropy:
        # Batch mode: run all configurations
        print(f"Running batch anisotropy test with N={N}")
        print(f"Configurations: {list(ANISOTROPY_CONFIGS.keys())}")

        summaries = []
        for name, aniso in ANISOTROPY_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"Configuration: {name} {aniso}")
            print('='*60)
            config_outdir = os.path.join(args.outdir, name)
            summary = run_single_config(
                N=N,
                beta=args.beta,
                anisotropy=aniso,
                temp_threshold=args.temp_threshold,
                connectivity=args.connectivity,
                outdir=config_outdir,
                config_name=name,
                seed_base=args.seed,
            )
            summaries.append(summary)

        # Save batch summary and create comparison plots
        print(f"\n{'='*60}")
        print("Batch Summary")
        print('='*60)
        print(f"{'Config':<15} {'Anisotropy':<15} {'K':>6} {'T_mean':>8} {'E_mean':>8}")
        print('-'*60)
        for s in summaries:
            aniso_str = f"({s['anisotropy'][0]:.0f},{s['anisotropy'][1]:.0f},{s['anisotropy'][2]:.0f})"
            print(f"{s['config_name']:<15} {aniso_str:<15} {s['K']:>6} "
                  f"{s['triaxiality_mean']:>8.3f} {s['elongation_mean']:>8.2f}")

        # Save summary to file
        summary_path = os.path.join(args.outdir, 'batch_summary.npz')
        np.savez(summary_path, summaries=summaries)
        print(f"\nWrote batch summary to {summary_path}")

        # Create comparison plots
        create_batch_comparison_plots(summaries, args.outdir)

    else:
        # Single configuration mode
        anisotropy = tuple(args.anisotropy) if args.anisotropy else (1.0, 1.0, 1.0)
        config_name = 'custom' if args.anisotropy else 'isotropic'

        print(f"Running single configuration: {config_name} {anisotropy}")
        run_single_config(
            N=N,
            beta=args.beta,
            anisotropy=anisotropy,
            temp_threshold=args.temp_threshold,
            connectivity=args.connectivity,
            outdir=args.outdir,
            config_name=config_name,
            seed_base=args.seed,
        )


def create_batch_comparison_plots(summaries, outdir):
    """Create comparison plots across all batch configurations."""
    import matplotlib.pyplot as plt

    if not summaries or all(s['K'] == 0 for s in summaries):
        print("No clumps found in any configuration, skipping comparison plots")
        return

    # Filter out configs with no clumps
    valid = [s for s in summaries if s['K'] > 0]
    if not valid:
        return

    names = [s['config_name'] for s in valid]
    T_means = [s['triaxiality_mean'] for s in valid]
    T_stds = [s['triaxiality_std'] for s in valid]
    E_means = [s['elongation_mean'] for s in valid]
    E_stds = [s['elongation_std'] for s in valid]
    Ks = [s['K'] for s in valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Triaxiality comparison
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, T_means, yerr=T_stds, capsize=5, color='steelblue', alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='T=0.5 (isotropic)')
    ax.axhline(0.0, color='blue', linestyle=':', alpha=0.5, label='T=0 (oblate)')
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='T=1 (prolate)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Triaxiality T')
    ax.set_title('Mean Triaxiality by Configuration')
    ax.legend(loc='best', fontsize=8)

    # Elongation comparison
    ax = axes[1]
    ax.bar(x, E_means, yerr=E_stds, capsize=5, color='coral', alpha=0.8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='E=1 (isotropic)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Elongation E = a/c')
    ax.set_title('Mean Elongation by Configuration')
    ax.legend(loc='best', fontsize=8)

    # Clump count
    ax = axes[2]
    ax.bar(x, Ks, color='green', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Number of Clumps')
    ax.set_title('Clump Count by Configuration')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'batch_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Wrote comparison plot to {os.path.join(outdir, 'batch_comparison.png')}")


if __name__ == '__main__':
    main()
