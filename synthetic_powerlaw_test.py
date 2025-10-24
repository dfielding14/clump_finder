from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('agg')

from local_label import label_3d
import metrics as M
from plot_clumps import make_pngs
from matplotlib.colors import LogNorm


def generate_powerlaw_field(shape=(128, 128, 128), beta=-5.0/3.0, seed=None):
    """Generate a 3-D real field with isotropic power spectrum P(k) ~ k^beta.

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

    # Amplitude ~ k^{beta/2}
    amp = kk ** (beta / 2.0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=128)
    ap.add_argument('--beta', type=float, default=-5.0/3.0)
    ap.add_argument('--temp-threshold', type=float, default=0.1, help='temperature threshold (default geometric mean of 1 and 0.01)')
    ap.add_argument('--connectivity', type=int, default=6)
    ap.add_argument('--outdir', default='./clump_out_synth')
    args = ap.parse_args()

    N = args.N
    dx = dy = dz = 1.0 / N
    origin = (0.0, 0.0, 0.0)

    # Generate raw fields
    f_rho = generate_powerlaw_field((N, N, N), beta=args.beta, seed=1)
    f_T   = generate_powerlaw_field((N, N, N), beta=args.beta, seed=2)
    velx  = generate_powerlaw_field((N, N, N), beta=args.beta, seed=3)
    vely  = generate_powerlaw_field((N, N, N), beta=args.beta, seed=4)
    velz  = generate_powerlaw_field((N, N, N), beta=args.beta, seed=5)

    # Map to requested ranges:
    # density in [1, 100], temperature in [0.01, 1], velocities in [-1, 1]
    # For density and temperature, use exp to enforce positivity then robust-rescale to range.
    dens = rescale_to_range(np.exp(f_rho * 0.5), 1.0, 100.0, robust=True)
    temp = rescale_to_range(np.exp(f_T * 0.5),  0.01, 1.0, robust=True)
    # velocities symmetric in [-1,1]
    velx = rescale_to_range(velx, -1.0, 1.0, robust=True)
    vely = rescale_to_range(vely, -1.0, 1.0, robust=True)
    velz = rescale_to_range(velz, -1.0, 1.0, robust=True)

    # Threshold
    # Temperature threshold: geometric mean of high and low = sqrt(1 * 0.01) = 0.1
    Tthr = float(args.temp_threshold)
    mask = temp < Tthr

    # Label
    labels = label_3d(mask, tile_shape=(128, 128, 128), connectivity=args.connectivity, halo=0)
    K = int(labels.max())

    # Metrics
    Vc = dx * dy * dz
    cell_count = M.num_cells(labels, K=K)
    vol = M.volumes(cell_count, dx, dy, dz)
    mass = M.masses(labels, dens, dx, dy, dz, K=K)
    cvol, cmass = M.centroids(labels, dens, dx, dy, dz, origin, ((0, N), (0, N), (0, N)), K=K)
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
        mu, sd, sk, ku = M.per_label_stats(labels, arr, weights=weight_vol, K=K, excess_kurtosis=False)
        stats[f"{name}_mean"] = mu
        stats[f"{name}_std"] = sd
        stats[f"{name}_skew"] = sk
        stats[f"{name}_kurt"] = ku

        mu_m, sd_m, sk_m, ku_m = M.per_label_stats(labels, arr, weights=weight_mass, K=K, excess_kurtosis=False)
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

    bbox_ijk = M.compute_bboxes(labels, ((0, N), (0, N), (0, N)), K=K)

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    rank_ids = np.arange(1, K + 1, dtype=np.int32)
    out = {
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
        "connectivity": np.int32(args.connectivity),
        "temperature_threshold": np.float64(Tthr),
        "rank": np.int32(0),
        "node_bbox_ijk": np.array([0, N, 0, N, 0, N], dtype=np.int64),
        "periodic": np.array([True, True, True], dtype=bool),
        "principal_axes_lengths": principal_axes_lengths,
        "axis_ratios": axis_ratios,
        "orientation": orientation,
    }
    out.update(stats)

    npz_path = os.path.join(out_dir, "clumps_rank00000.npz")
    np.savez(npz_path, **out)
    print(f"Wrote {npz_path}  K={K}")

    # Plots: per-clump PNGs and 2D slices of state variables
    make_pngs(npz_path, out_dir, use_volume=False, mass_weighted=False, prefix="clumps_synth")

    # Map maker: slice at mid-plane k=N//2 for each variable
    k = N // 2
    # Density map (log)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5), dpi=150)
    im = plt.imshow(dens[k, :, :], origin='lower', cmap='viridis', norm=LogNorm(vmin=1.0, vmax=100.0))
    plt.colorbar(im, label='density')
    plt.title('Density (k mid-slice)')
    plt.savefig(os.path.join(out_dir, 'map_density.png'), bbox_inches='tight'); plt.close()

    # Temperature map (log)
    plt.figure(figsize=(6, 5), dpi=150)
    im = plt.imshow(temp[k, :, :], origin='lower', cmap='magma', norm=LogNorm(vmin=0.01, vmax=1.0))
    plt.colorbar(im, label='temperature')
    plt.title('Temperature (k mid-slice)')
    plt.savefig(os.path.join(out_dir, 'map_temperature.png'), bbox_inches='tight'); plt.close()

    # Velocities (linear, diverging)
    for name, arr in [('velx', velx), ('vely', vely), ('velz', velz)]:
        plt.figure(figsize=(6, 5), dpi=150)
        im = plt.imshow(arr[k, :, :], origin='lower', cmap='coolwarm', vmin=-1.0, vmax=1.0)
        plt.colorbar(im, label=name)
        plt.title(f'{name} (k mid-slice)')
        plt.savefig(os.path.join(out_dir, f'map_{name}.png'), bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
