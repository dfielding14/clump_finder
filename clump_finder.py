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
    ap.add_argument("--extra-stats", action="store_true",
                    help="Compute extended per-clump statistics and shape diagnostics.")
    args = ap.parse_args()

    cfg = parse_config(args.config)
    extra_stats = bool(cfg.get("extra_stats", False)) or args.extra_stats

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
    if cfg.get("assert_nres_from_data", False):
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

    # Ensure overlap width and ghost zones are available
    ghost_cfg_raw = cfg.get("ghost_width", 1)
    try:
        ghost_cfg = int(ghost_cfg_raw)
    except Exception:
        ghost_cfg = 1
    overlap_cfg_raw = cfg.get("overlap_width", None)
    if overlap_cfg_raw is None:
        overlap_width = max(1, ghost_cfg)
    else:
        overlap_width = int(overlap_cfg_raw)
    if overlap_width < 1:
        raise ValueError("overlap_width must be >= 1 for overlap-exact stitching")
    ghost_width = max(ghost_cfg, overlap_width)

    # I/O config
    io_cfg = IOConfig(
        dataset_path=str(cfg["dataset_path"]),
        step=int(cfg.get("step", 0)),
        gamma=float(cfg.get("gamma", 5.0 / 3.0)),
        field_dtype=np.dtype(cfg.get("field_dtype", "float32")),
        ghost_width=ghost_width,
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
    if halo < 1:
        raise ValueError("ghost_width must be >= 1 to support overlap-exact stitching")
    s = slice(halo, -halo if halo > 0 else None)
    dens_i = dens[s, s, s]
    temp_i = temp[s, s, s]
    velx_i = velx[s, s, s]
    vely_i = vely[s, s, s]
    velz_i = velz[s, s, s]
    t_load = time.time()

    # Threshold mask with configurable field and comparator
    cut_by = str(cfg.get("cut_by", "temperature")).lower()
    cut_op = str(cfg.get("cut_op", "lt")).lower()
    if cut_by not in ("temperature", "density"):
        raise ValueError("cut_by must be 'temperature' or 'density'")
    if cut_op not in ("lt", "gt"):
        raise ValueError("cut_op must be 'lt' or 'gt'")

    if cut_by == "temperature":
        thr = float(cfg.get("temperature_threshold", 1.0))
        field = temp
    else:
        thr = float(cfg.get("density_threshold", 1.0))
        field = dens
    mask = (field < thr) if cut_op == "lt" else (field > thr)  # include halo in mask for completeness

    # Labeling
    tile_shape = tuple(cfg.get("tile_shape", [128, 128, 128]))
    connectivity = int(cfg.get("connectivity", 6))
    labels_ext = label_3d(mask, tile_shape=tile_shape, connectivity=connectivity, halo=0)
    if halo > 0:
        labels_core = labels_ext[halo:-halo, halo:-halo, halo:-halo]
    else:
        labels_core = labels_ext
    core_vals = np.unique(labels_core)
    core_vals = core_vals[core_vals != 0]
    lut = np.zeros(int(labels_ext.max()) + 1, dtype=np.uint32)
    lut[core_vals] = np.arange(1, core_vals.size + 1, dtype=np.uint32)
    labels_core = lut[labels_core]
    labels_ext = lut[labels_ext]
    labels = labels_core
    t_label = time.time()

    K = int(labels.max())

    # Metrics
    Vc = dx * dy * dz
    cell_count = M.num_cells(labels, K=K)
    vol = M.volumes(cell_count, dx, dy, dz)
    mass = M.masses(labels, dens_i, dx, dy, dz, K=K)

    cvol, cmass = M.centroids(labels, dens_i, dx, dy, dz, origin, node_bbox, K=K)
    area = M.exposed_area(labels, dx, dy, dz, K=K)

    # Always provide velocity magnitude mean/std for stitched diagnostics
    speed = np.sqrt(
        velx_i.astype(np.float64, copy=False) ** 2
        + vely_i.astype(np.float64, copy=False) ** 2
        + velz_i.astype(np.float64, copy=False) ** 2
    )
    v_mean, v_std, _, _ = M.per_label_stats(labels, speed, K=K, excess_kurtosis=False)

    # Bounding boxes (global indices, [min,max))
    bbox_ijk = M.compute_bboxes(labels, node_bbox, K=K)

    extra_out: dict[str, np.ndarray] = {}
    if extra_stats:
        kurtosis_excess = bool(cfg.get("excess_kurtosis", False))
        Vc64 = float(Vc)
        weight_vol = np.full(labels.shape, Vc64, dtype=np.float64)
        weight_mass = dens_i.astype(np.float64, copy=False) * Vc64
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
            mu, sd, sk, ku = M.per_label_stats(labels, arr, weights=weight_vol, K=K, excess_kurtosis=kurtosis_excess)
            stats[f"{name}_mean"] = mu
            stats[f"{name}_std"] = sd
            stats[f"{name}_skew"] = sk
            stats[f"{name}_kurt"] = ku

            mu_m, sd_m, sk_m, ku_m = M.per_label_stats(labels, arr, weights=weight_mass, K=K, excess_kurtosis=kurtosis_excess)
            stats[f"{name}_mean_massw"] = mu_m
            stats[f"{name}_std_massw"] = sd_m
            stats[f"{name}_skew_massw"] = sk_m
            stats[f"{name}_kurt_massw"] = ku_m

        # Shape diagnostics via mass-weighted covariance tensor
        (i0, i1), (j0, j1), (k0, k1) = node_bbox
        xi = origin[0] + (np.arange(i0, i1) + 0.5) * dx
        yj = origin[1] + (np.arange(j0, j1) + 0.5) * dy
        zk = origin[2] + (np.arange(k0, k1) + 0.5) * dz

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

        for i in range(labels.shape[0]):
            L = labels[i, :, :].ravel()
            if L.size == 0:
                continue
            w = (dens_i[i, :, :].ravel().astype(np.float64, copy=False)) * Vc64
            binc = np.bincount(L, weights=w, minlength=K + 1)
            W += binc
            x = xi[i]
            Sx += x * binc
            Sxx += (x * x) * binc

        for j in range(labels.shape[1]):
            L = labels[:, j, :].ravel()
            w = (dens_i[:, j, :].ravel().astype(np.float64, copy=False)) * Vc64
            binc = np.bincount(L, weights=w, minlength=K + 1)
            y = yj[j]
            Sy += y * binc
            Syy += (y * y) * binc

        for k in range(labels.shape[2]):
            L = labels[:, :, k].ravel()
            w = (dens_i[:, :, k].ravel().astype(np.float64, copy=False)) * Vc64
            binc = np.bincount(L, weights=w, minlength=K + 1)
            z = zk[k]
            Sz += z * binc
            Szz += (z * z) * binc

        for j in range(labels.shape[1]):
            y = yj[j]
            for i in range(labels.shape[0]):
                L = labels[i, j, :]
                if L.size == 0:
                    continue
                w = (dens_i[i, j, :].astype(np.float64, copy=False)) * Vc64
                Sxy += (xi[i] * y) * np.bincount(L, weights=w, minlength=K + 1)

        for k in range(labels.shape[2]):
            z = zk[k]
            for i in range(labels.shape[0]):
                L = labels[i, :, k]
                w = (dens_i[i, :, k].astype(np.float64, copy=False)) * Vc64
                Sxz += (xi[i] * z) * np.bincount(L, weights=w, minlength=K + 1)

        for k in range(labels.shape[2]):
            z = zk[k]
            for j in range(labels.shape[1]):
                L = labels[:, j, k]
                w = (dens_i[:, j, k].astype(np.float64, copy=False)) * Vc64
                Syz += (yj[j] * z) * np.bincount(L, weights=w, minlength=K + 1)

        W = W[1:]
        Sx = Sx[1:]; Sy = Sy[1:]; Sz = Sz[1:]
        Sxx = Sxx[1:]; Syy = Syy[1:]; Szz = Szz[1:]
        Sxy = Sxy[1:]; Sxz = Sxz[1:]; Syz = Syz[1:]

        small = 1e-300
        mu_x = Sx / (W + small)
        mu_y = Sy / (W + small)
        mu_z = Sz / (W + small)

        Cxx = Sxx / (W + small) - mu_x * mu_x
        Cyy = Syy / (W + small) - mu_y * mu_y
        Czz = Szz / (W + small) - mu_z * mu_z
        Cxy = Sxy / (W + small) - mu_x * mu_y
        Cxz = Sxz / (W + small) - mu_x * mu_z
        Cyz = Syz / (W + small) - mu_y * mu_z

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
            axis_ratios[idx, :] = (b / (a + small), c / (a + small))
            orientation[idx, :, :] = vecs

        # Tier 0: Derived shape metrics from existing V, S, principal axes
        derived_metrics = M.derived_shape_metrics(vol, area, principal_axes_lengths)

        # Tier 1a: Euler characteristic
        euler_chi = M.euler_characteristic_fast(labels, K=K)

        # Tier 1b: Bounding box shape metrics
        bbox_metrics = M.bbox_shape_metrics(bbox_ijk, principal_axes_lengths)

        # Tier 2: Minkowski functionals (selective, above size threshold)
        minkowski_min_cells = int(cfg.get("minkowski_min_cells", 1000))
        minkowski_metrics = M.compute_minkowski_functionals(
            labels, vol, area, euler_chi,
            dx=dx, dy=dy, dz=dz, K=K,
            min_cells=minkowski_min_cells,
            cell_count=cell_count
        )

        presence = np.zeros((K, 6), dtype=bool)
        for idx, arr in enumerate((labels[0, :, :],
                                   labels[-1, :, :],
                                   labels[:, 0, :],
                                   labels[:, -1, :],
                                   labels[:, :, 0],
                                   labels[:, :, -1])):
            u = np.unique(arr)
            u = u[(u > 0) & (u <= K)]
            presence[u - 1, idx] = True

        pair_bits = np.zeros((K,), dtype=np.uint16)
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                 (1, 2), (1, 3), (1, 4), (1, 5),
                 (2, 3), (2, 4), (2, 5),
                 (3, 4), (3, 5),
                 (4, 5)]
        for bit, (a, b) in enumerate(pairs):
            both = presence[:, a] & presence[:, b]
            pair_bits |= (both.astype(np.uint16) << np.uint16(bit))

        extra_out.update(stats)
        extra_out.update({
            "principal_axes_lengths": principal_axes_lengths,
            "axis_ratios": axis_ratios,
            "orientation": orientation,
            # Tier 0: Derived shape metrics
            "triaxiality": derived_metrics["triaxiality"],
            "sphericity": derived_metrics["sphericity"],
            "compactness": derived_metrics["compactness"],
            "r_eff": derived_metrics["r_eff"],
            "elongation": derived_metrics["elongation"],
            # Tier 1a: Euler characteristic
            "euler_characteristic": euler_chi,
            # Tier 1b: Bounding box shape metrics
            "bbox_lengths": bbox_metrics["bbox_lengths"],
            "bbox_elongation": bbox_metrics["bbox_elongation"],
            "bbox_flatness": bbox_metrics["bbox_flatness"],
            "curvature_flag": bbox_metrics["curvature_flag"],
            # Tier 2: Minkowski functionals and shapefinders
            "integrated_curvature": minkowski_metrics["integrated_curvature"],
            "thickness": minkowski_metrics["thickness"],
            "breadth": minkowski_metrics["breadth"],
            "length": minkowski_metrics["length"],
            "filamentarity": minkowski_metrics["filamentarity"],
            "planarity": minkowski_metrics["planarity"],
            "minkowski_computed": minkowski_metrics["minkowski_computed"],
            "minkowski_min_cells": np.int32(minkowski_min_cells),
            # Stitching metadata
            "face_presence": presence,
            "face_pair_bits": pair_bits,
            "shell_t": np.int32(3),
            "shell_xneg": labels[0:3, :, :].astype(np.uint32, copy=False),
            "shell_xpos": labels[-3:, :, :].astype(np.uint32, copy=False),
            "shell_yneg": labels[:, 0:3, :].astype(np.uint32, copy=False),
            "shell_ypos": labels[:, -3:, :].astype(np.uint32, copy=False),
            "shell_zneg": labels[:, :, 0:3].astype(np.uint32, copy=False),
            "shell_zpos": labels[:, :, -3:].astype(np.uint32, copy=False),
        })
    ni_c, nj_c, nk_c = labels.shape
    if overlap_width != 1:
        raise NotImplementedError("overlap_width > 1 not yet supported")
    if overlap_width > halo:
        raise ValueError("overlap_width cannot exceed ghost_width")
    i_start = halo
    i_end = halo + ni_c
    j_start = halo
    j_end = halo + nj_c
    k_start = halo
    k_end = halo + nk_c
    ovlp_xneg = labels_ext[i_start, j_start:j_end, k_start:k_end].astype(np.uint32, copy=False)
    ovlp_xpos = labels_ext[i_end - 1, j_start:j_end, k_start:k_end].astype(np.uint32, copy=False)
    ovlp_yneg = labels_ext[i_start:i_end, j_start, k_start:k_end].astype(np.uint32, copy=False)
    ovlp_ypos = labels_ext[i_start:i_end, j_end - 1, k_start:k_end].astype(np.uint32, copy=False)
    ovlp_zneg = labels_ext[i_start:i_end, j_start:j_end, k_start].astype(np.uint32, copy=False)
    ovlp_zpos = labels_ext[i_start:i_end, j_start:j_end, k_end - 1].astype(np.uint32, copy=False)

    # Defer min-clump filtering to global post-stitch stage
    min_cells_cfg = cfg.get("min_clump_cells", 1)
    try:
        min_cells = int(min_cells_cfg) if min_cells_cfg is not None else 1
    except Exception:
        min_cells = 1
    if min_cells > 1 and rank == 0:
        print(f"NOTE: deferring min_clump_cells={min_cells} filtering until after stitching.")

    K = int(cell_count.shape[0])

    # Prepare output
    out_dir = cfg.get("output_dir", "./clump_out")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.get("log_dir", "./logs"), exist_ok=True)

    rank_ids = np.arange(1, K + 1, dtype=np.int32)
    # For stitching: export boundary face maps from the UNFILTERED labels to preserve thin bridges
    face_xneg = labels[0, :, :].astype(np.uint32, copy=False)
    face_xpos = labels[-1, :, :].astype(np.uint32, copy=False)
    face_yneg = labels[:, 0, :].astype(np.uint32, copy=False)
    face_ypos = labels[:, -1, :].astype(np.uint32, copy=False)
    face_zneg = labels[:, :, 0].astype(np.uint32, copy=False)
    face_zpos = labels[:, :, -1].astype(np.uint32, copy=False)

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
        "velocity_mean": v_mean,
        "velocity_std": v_std,
        # Threshold provenance
        "cut_by": np.array(cut_by),
        "cut_op": np.array(cut_op),
        "threshold": np.float64(thr),
        # legacy field for backward compatibility
        "temperature_threshold": np.float64(thr if cut_by == "temperature" else cfg.get("temperature_threshold", np.nan)),
        "rank": np.int32(rank),
        "node_bbox_ijk": np.array([i0, i1, j0, j1, k0, k1], dtype=np.int64),
        "periodic": np.array([True, True, True], dtype=bool),
        "overlap_width": np.int32(overlap_width),
        "ovlp_xneg": ovlp_xneg,
        "ovlp_xpos": ovlp_xpos,
        "ovlp_yneg": ovlp_yneg,
        "ovlp_ypos": ovlp_ypos,
        "ovlp_zneg": ovlp_zneg,
        "ovlp_zpos": ovlp_zpos,
        # Stitching faces
        "face_xneg": face_xneg,
        "face_xpos": face_xpos,
        "face_yneg": face_yneg,
        "face_ypos": face_ypos,
        "face_zneg": face_zneg,
        "face_zpos": face_zpos,
    }
    if extra_out:
        out.update(extra_out)

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
            "cut_by": cut_by,
            "cut_op": cut_op,
            "threshold": float(thr),
            "temperature_threshold": float(thr if cut_by == "temperature" else cfg.get("temperature_threshold", float('nan'))),
            "ghost_width": int(ghost_width),
        },
        "stitching": {
            "overlap_width": int(overlap_width),
            "min_clump_cells_deferred": int(min_cells),
        },
        "extra_stats": bool(extra_stats),
        "git_rev": _git_rev(),
        "config": cfg,
        "output_npz": os.path.basename(part_path),
    }
    with open(os.path.join(out_dir, f"clumps_rank{rank:05d}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if rank == 0 or cfg.get("profile", False):
        print(f"rank={rank} times: load={t_load-t0:.2f}s label={t_label-t_load:.2f}s reduce={t_done-t_label:.2f}s K={K}")


if __name__ == "__main__":
    main()
