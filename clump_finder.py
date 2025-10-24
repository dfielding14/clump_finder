from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import yaml
from mpi4py import MPI

from io_bridge import IOConfig, load_subvolume
from io_bridge import query_domain_shape
from local_label import label_3d
import metrics as M


def compute_dims(size: int) -> Tuple[int, int, int]:
    # Near-cubic factorization using MPI helper
    dims = MPI.Compute_dims(size, [0, 0, 0])
    return dims[0], dims[1], dims[2]


def split_axis(n: int, p: int, coord: int) -> Tuple[int, int]:
    base = n // p
    rem = n % p
    start = coord * base + min(coord, rem)
    length = base + (1 if coord < rem else 0)
    return start, start + length


def parse_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--excess-kurtosis", action="store_true", dest="excess_kurtosis")
    ap.add_argument("--auto-aggregate-plot", action="store_true",
                    help="After all ranks finish, aggregate on rank 0 and emit PNG plots")
    ap.add_argument("--assert-nres-from-data", action="store_true",
                    help="Override/validate Nres by reading dataset shape; warns if mismatch.")
    args = ap.parse_args()

    cfg = parse_config(args.config)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify one rank per node if requested
    if cfg.get("verify_one_rank_per_node", True):
        local = comm.Split_type(MPI.COMM_TYPE_SHARED)
        if local.Get_size() != 1 and rank == 0:
            print("WARNING: Detected more than one rank per node; baseline assumes one rank per node.")

    # Grid resolution and spacing
    N = int(cfg.get("Nres", 0))
    # Optionally assert/override from data (rank 0)
    if args.assert_nres_from_data or cfg.get("assert_nres_from_data", False):
        if rank == 0:
            try:
                (Nz, Ny, Nx), lvl = query_domain_shape(str(cfg["dataset_path"]), int(cfg.get("step", 0)))
                if not (Nx == Ny == Nz):
                    print(f"WARNING: dataset shape not cubic: (Nz,Ny,Nx)=({Nz},{Ny},{Nx}). Using Nx={Nx} for Nres.")
                if N > 0 and N != Nx:
                    print(f"WARNING: Nres ({N}) != dataset Nx ({Nx}); overriding to Nx.")
                N = int(Nx)
            except Exception as e:
                if rank == 0:
                    print(f"WARNING: Failed to query dataset shape: {e}. Using configured Nres={N}.")
        N = comm.bcast(N if rank == 0 else None, root=0)
    if N <= 0:
        raise ValueError("Nres must be provided and > 0 (or use --assert-nres-from-data)")
    dx = cfg.get("dx")
    dy = cfg.get("dy")
    dz = cfg.get("dz")
    if dx is None or dy is None or dz is None:
        dx = dy = dz = 1.0 / N
    origin = tuple(cfg.get("origin", [0.0, 0.0, 0.0]))

    # Cartesian decomposition (periodic on all axes)
    px, py, pz = compute_dims(size)
    cart = comm.Create_cart(dims=(px, py, pz), periods=(True, True, True), reorder=False)
    coords = cart.Get_coords(rank)
    i0, i1 = split_axis(N, px, coords[0])
    j0, j1 = split_axis(N, py, coords[1])
    k0, k1 = split_axis(N, pz, coords[2])
    node_bbox = ((i0, i1), (j0, j1), (k0, k1))

    if args.dry_run:
        if rank == 0:
            print(f"World size={size} dims={(px,py,pz)}")
        print(f"rank={rank} coords={coords} bbox={node_bbox}")
        return

    # I/O config
    io_cfg = IOConfig(
        dataset_path=str(cfg["dataset_path"]),
        step=int(cfg.get("step", 0)),
        gamma=float(cfg.get("gamma", 5.0 / 3.0)),
        field_dtype=np.dtype(cfg.get("field_dtype", "float32")),
        ghost_width=int(cfg.get("ghost_width", 1)),
        level_suffix=None,
    )

    # Load subvolume
    t0 = time.time()
    fields = load_subvolume(node_bbox=node_bbox, cfg=io_cfg)
    dens = fields["dens"]
    temp = fields["temp"]
    velx = fields["velx"]
    vely = fields["vely"]
    velz = fields["velz"]
    halo = int(max(0, io_cfg.ghost_width))
    s = slice(halo, -halo if halo > 0 else None)
    dens_i = dens[s, s, s]
    temp_i = temp[s, s, s]
    velx_i = velx[s, s, s]
    vely_i = vely[s, s, s]
    velz_i = velz[s, s, s]
    t_load = time.time()

    # Threshold mask
    Tthr = float(cfg.get("temperature_threshold", 1.0))
    mask = temp < Tthr  # include halo in mask for completeness

    # Labeling
    tile_shape = tuple(cfg.get("tile_shape", [128, 128, 128]))
    connectivity = int(cfg.get("connectivity", 6))
    labels = label_3d(mask, tile_shape=tile_shape, connectivity=connectivity, halo=halo)
    t_label = time.time()

    K = int(labels.max())

    # Metrics
    Vc = dx * dy * dz
    cell_count = M.num_cells(labels, K=K)
    vol = M.volumes(cell_count, dx, dy, dz)
    mass = M.masses(labels, dens_i, dx, dy, dz, K=K)

    cvol, cmass = M.centroids(labels, dens_i, dx, dy, dz, origin, node_bbox, K=K)
    area = M.exposed_area(labels, dx, dy, dz, K=K)

    # Stats for variables (volume-weighted and mass-weighted)
    # weights: volume -> Vc; mass -> dens*Vc
    weight_vol = np.full(dens_i.shape, Vc, dtype=np.float64)
    weight_mass = (dens_i.astype(np.float64, copy=False)) * Vc

    pressure = dens_i * temp_i

    stats = {}
    for name, arr in (
        ("rho", dens_i),
        ("T", temp_i),
        ("vx", velx_i),
        ("vy", vely_i),
        ("vz", velz_i),
        ("pressure", pressure),
    ):
        mu, sd, sk, ku = M.per_label_stats(labels, arr, weights=weight_vol, K=K, excess_kurtosis=args.excess_kurtosis)
        stats[f"{name}_mean"] = mu
        stats[f"{name}_std"] = sd
        stats[f"{name}_skew"] = sk
        stats[f"{name}_kurt"] = ku

        mu_m, sd_m, sk_m, ku_m = M.per_label_stats(labels, arr, weights=weight_mass, K=K, excess_kurtosis=args.excess_kurtosis)
        stats[f"{name}_mean_massw"] = mu_m
        stats[f"{name}_std_massw"] = sd_m
        stats[f"{name}_skew_massw"] = sk_m
        stats[f"{name}_kurt_massw"] = ku_m

    # Shape metrics (mass-weighted covariance)
    # Compute first and second moments using mass weights
    # Accumulate sums required for covariance: Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz, W
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

    (i0, i1), (j0, j1), (k0, k1) = node_bbox
    xi = origin[0] + (np.arange(i0, i1) + 0.5) * dx
    yj = origin[1] + (np.arange(j0, j1) + 0.5) * dy
    zk = origin[2] + (np.arange(k0, k1) + 0.5) * dz

    # Accumulate by looping over slices to keep memory steady
    for i in range(labels.shape[0]):
        L = labels[i, :, :].ravel()
        w = (dens_i[i, :, :].ravel().astype(np.float64, copy=False)) * Vc
        W += np.bincount(L, weights=w, minlength=K + 1)
        x = xi[i]
        Sx += x * np.bincount(L, weights=w, minlength=K + 1)
        Sxx += (x * x) * np.bincount(L, weights=w, minlength=K + 1)

    for j in range(labels.shape[1]):
        L = labels[:, j, :].ravel()
        w = (dens_i[:, j, :].ravel().astype(np.float64, copy=False)) * Vc
        y = yj[j]
        Sy += y * np.bincount(L, weights=w, minlength=K + 1)
        Syy += (y * y) * np.bincount(L, weights=w, minlength=K + 1)

    for k in range(labels.shape[2]):
        L = labels[:, :, k].ravel()
        w = (dens_i[:, :, k].ravel().astype(np.float64, copy=False)) * Vc
        z = zk[k]
        Sz += z * np.bincount(L, weights=w, minlength=K + 1)
        Szz += (z * z) * np.bincount(L, weights=w, minlength=K + 1)

    # Cross terms Sxy, Sxz, Syz: compute by looping over (i,j) rows and summing over k
    # This is heavier but still keeps memory bounded per plane.
    for j in range(labels.shape[1]):
        y = yj[j]
        for i in range(labels.shape[0]):
            L = labels[i, j, :]
            w = (dens_i[i, j, :].astype(np.float64, copy=False)) * Vc
            if L.size == 0:
                continue
            Sxy += (xi[i] * y) * np.bincount(L, weights=w, minlength=K + 1)

    for k in range(labels.shape[2]):
        z = zk[k]
        for i in range(labels.shape[0]):
            L = labels[i, :, k]
            w = (dens_i[i, :, k].astype(np.float64, copy=False)) * Vc
            Sxz += (xi[i] * z) * np.bincount(L, weights=w, minlength=K + 1)

    for k in range(labels.shape[2]):
        z = zk[k]
        for j in range(labels.shape[1]):
            L = labels[:, j, k]
            w = (dens_i[:, j, k].astype(np.float64, copy=False)) * Vc
            Syz += (yj[j] * z) * np.bincount(L, weights=w, minlength=K + 1)

    # Drop background
    W = W[1:]
    Sx, Sy, Sz = Sx[1:], Sy[1:], Sz[1:]
    Sxx, Syy, Szz = Sxx[1:], Syy[1:], Szz[1:]
    Sxy, Sxz, Syz = Sxy[1:], Sxz[1:], Syz[1:]

    # Means and covariances
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
        # ensure symmetry and numerical stability
        C = (C + C.T) * 0.5
        vals, vecs = np.linalg.eigh(C)
        # sort descending
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # RMS extents
        a = np.sqrt(max(vals[0], 0.0))
        b = np.sqrt(max(vals[1], 0.0))
        c = np.sqrt(max(vals[2], 0.0))
        principal_axes_lengths[idx, :] = (a, b, c)
        axis_ratios[idx, :] = (b / (a + 1e-300), c / (a + 1e-300))
        orientation[idx, :, :] = vecs

    # Bounding boxes (global indices, [min,max))
    bbox_ijk = M.compute_bboxes(labels, node_bbox, K=K)

    # Filter out clumps smaller than the configured minimum cell count
    min_cells = int(cfg.get("min_clump_cells", 64))
    K_orig = cell_count.shape[0]
    if K_orig:
        keep = cell_count >= min_cells
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"rank={rank} dropping {dropped} clumps smaller than {min_cells} cells")
    else:
        keep = np.zeros(0, dtype=bool)

    cell_count = cell_count[keep]
    vol = vol[keep]
    mass = mass[keep]
    area = area[keep]
    cvol = cvol[keep]
    cmass = cmass[keep]
    principal_axes_lengths = principal_axes_lengths[keep]
    axis_ratios = axis_ratios[keep]
    orientation = orientation[keep]
    bbox_ijk = bbox_ijk[keep]
    stats = {
        k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (K_orig,) else v)
        for k, v in stats.items()
    }

    K = int(cell_count.shape[0])

    # Prepare output
    out_dir = cfg.get("output_dir", "./clump_out")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.get("log_dir", "./logs"), exist_ok=True)

    rank_ids = np.arange(1, K_orig + 1, dtype=np.int32)[keep]
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
        "connectivity": np.int32(connectivity),
        "temperature_threshold": np.float64(Tthr),
        "rank": np.int32(rank),
        "node_bbox_ijk": np.array([i0, i1, j0, j1, k0, k1], dtype=np.int64),
        "periodic": np.array([True, True, True], dtype=bool),
        "principal_axes_lengths": principal_axes_lengths,
        "axis_ratios": axis_ratios,
        "orientation": orientation,
    }
    out.update(stats)

    part_path = os.path.join(out_dir, f"clumps_rank{rank:05d}.npz")
    np.savez(part_path, **out)

    t_done = time.time()

    # Write per-rank metadata sidecar (JSON)
    import json, subprocess
    def _git_rev():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"
    meta = {
        "rank": int(rank),
        "coords": tuple(coords),
        "cart_dims": (int(px), int(py), int(pz)),
        "node_bbox_ijk": [int(i0), int(i1), int(j0), int(j1), int(k0), int(k1)],
        "K": int(K),
        "times": {
            "load": float(t_load - t0),
            "label": float(t_label - t_load),
            "reduce": float(t_done - t_label),
        },
        "dataset": {
            "path": str(cfg.get("dataset_path")),
            "step": int(cfg.get("step", 0)),
        },
        "grid": {
            "Nres": int(N),
            "dx": float(dx), "dy": float(dy), "dz": float(dz),
            "origin": list(origin),
            "periodic": [True, True, True],
        },
        "labeling": {
            "tile_shape": list(tile_shape),
            "connectivity": int(connectivity),
            "temperature_threshold": float(Tthr),
        },
        "git_rev": _git_rev(),
        "config": cfg,
        "output_npz": os.path.basename(part_path),
    }
    with open(os.path.join(out_dir, f"clumps_rank{rank:05d}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if rank == 0 or cfg.get("profile", False) or args.profile:
        print(f"rank={rank} times: load={t_load-t0:.2f}s label={t_label-t_load:.2f}s reduce={t_done-t_label:.2f}s K={K}")

    # Optional aggregate + plot on rank 0
    comm.Barrier()
    if args.auto_aggregate_plot and rank == 0:
        try:
            from aggregate_results import aggregate
            from plot_clumps import make_pngs
            master = os.path.join(out_dir, "clumps_master.npz")
            aggregate(out_dir, master)
            make_pngs(master, out_dir, use_volume=False, mass_weighted=False)
            print(f"Aggregated and wrote PNG plots to {out_dir}")
        except Exception as e:
            print(f"Auto-aggregate-plot failed: {e}")


if __name__ == "__main__":
    main()
