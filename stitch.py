"""
Minimal overlap-exact stitcher.

Reads per-rank clump files from disk, unifies labels that share voxels on the
overlap planes, and aggregates per-clump metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Tuple

import numpy as np


class DSU:
    __slots__ = ("p",)

    def __init__(self):
        self.p: Dict[np.uint64, np.uint64] = {}

    def find(self, x: np.uint64) -> np.uint64:
        p = self.p
        while p.get(x, x) != x:
            p[x] = p.get(p[x], p[x])
            x = p[x]
        return x

    def union(self, a: np.uint64, b: np.uint64):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _load_meta(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _gid(rank: int, local_id: int) -> np.uint64:
    return (np.uint64(rank) << np.uint64(32)) | np.uint64(local_id)


def _neighbor(coords: Tuple[int, int, int], dims: Tuple[int, int, int], axis: int, sign: int,
              periodic: Tuple[bool, bool, bool]):
    c = list(coords)
    c[axis] += sign
    if 0 <= c[axis] < dims[axis]:
        return tuple(c)
    if periodic[axis]:
        c[axis] = (c[axis] + dims[axis]) % dims[axis]
        return tuple(c)
    return None


def index_parts(input_dir: str):
    metas = sorted(glob.glob(os.path.join(input_dir, "clumps_rank*.meta.json")))
    if not metas:
        raise FileNotFoundError("No clumps_rank*.meta.json")
    ranks = {}
    cart_dims = None
    periodic = (True, True, True)
    for m in metas:
        d = _load_meta(m)
        r = int(d["rank"])
        ranks[r] = {
            "coords": tuple(d["coords"]),
            "bbox": tuple(d["node_bbox_ijk"]),
            "npz": os.path.join(input_dir, d["output_npz"]),
            "meta": m,
        }
        if cart_dims is None:
            cart_dims = tuple(d["cart_dims"])
        if "grid" in d and "periodic" in d["grid"]:
            periodic = tuple(bool(x) for x in d["grid"]["periodic"])
    return ranks, cart_dims, periodic


def build_edges(ranks: dict,
                cart_dims: Tuple[int, int, int],
                periodic: Tuple[bool, bool, bool],
                dx: float, dy: float, dz: float):
    dsu = DSU()
    edge_counts = {"x": {}, "y": {}, "z": {}}  # dict[(gid_a,gid_b)] = count
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}

    def add_edges(axis_key: str, r: int, rn: int, a: np.ndarray, b: np.ndarray):
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return
        A = a[mask].astype(np.uint64, copy=False)
        B = b[mask].astype(np.uint64, copy=False)
        for la, lb in zip(A, B):
            ga = _gid(r, int(la))
            gb = _gid(rn, int(lb))
            dsu.union(ga, gb)
            key = (ga, gb) if ga < gb else (gb, ga)
            edge_counts[axis_key][key] = edge_counts[axis_key].get(key, 0) + 1

    for r, info in ranks.items():
        npz = _load_npz(info["npz"])
        coords = tuple(info["coords"])

        ncoords = _neighbor(coords, cart_dims, axis=0, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = _load_npz(ranks[rn]["npz"])
            add_edges("x", r, rn, npz["face_xpos"], npz_n["face_xneg"])

        ncoords = _neighbor(coords, cart_dims, axis=1, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = _load_npz(ranks[rn]["npz"])
            add_edges("y", r, rn, npz["face_ypos"], npz_n["face_yneg"])

        ncoords = _neighbor(coords, cart_dims, axis=2, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = _load_npz(ranks[rn]["npz"])
            add_edges("z", r, rn, npz["face_zpos"], npz_n["face_zneg"])

    face_area = {"x": dy * dz, "y": dx * dz, "z": dx * dy}
    return dsu, edge_counts, face_area


def _merge_by_overlap_planes(ranks: dict,
                             cart_dims: Tuple[int, int, int],
                             periodic: Tuple[bool, bool, bool],
                             dsu: DSU | None = None) -> DSU:
    """Unify labels that share identical global voxels on overlap planes."""
    if dsu is None:
        dsu = DSU()
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}
    cache: Dict[int, Dict[str, np.ndarray]] = {}

    def arr(r: int, key: str) -> np.ndarray:
        d = cache.get(r)
        if d is None:
            d = _load_npz(ranks[r]["npz"])
            cache[r] = d
        if key not in d:
            raise KeyError(f"Missing '{key}' in {ranks[r]['npz']}; re-run clump export with overlap metadata.")
        return d[key]

    def pair(axis: int, key_pos: str, key_neg: str):
        for r, info in ranks.items():
            coords = tuple(info["coords"])
            ncoords = _neighbor(coords, cart_dims, axis=axis, sign=+1, periodic=periodic)
            if ncoords is None:
                continue
            rn = by_coords[ncoords]
            A = arr(r, key_pos)
            B = arr(rn, key_neg)
            if A.shape != B.shape:
                raise ValueError(f"Overlap plane shape mismatch between ranks {r} and {rn} on axis {axis}")
            mask = (A > 0) & (B > 0)
            if not mask.any():
                continue
            La = A[mask].astype(np.uint64, copy=False)
            Lb = B[mask].astype(np.uint64, copy=False)
            for la, lb in zip(La, Lb):
                dsu.union(_gid(r, int(la)), _gid(rn, int(lb)))

    pair(axis=0, key_pos="ovlp_xpos", key_neg="ovlp_xneg")
    pair(axis=1, key_pos="ovlp_ypos", key_neg="ovlp_yneg")
    pair(axis=2, key_pos="ovlp_zpos", key_neg="ovlp_zneg")
    return dsu


def _combine_weighted_stats(G: int, parts: Dict, roots: dict, root_to_idx: dict,
                             stat_name: str, weight_key: str = "cell_count"):
    """Combine weighted mean and std across ranks using parallel variance formula.

    Returns (mean, std) arrays of shape (G,).
    """
    mean_key = f"{stat_name}_mean"
    std_key = f"{stat_name}_std"

    # Check if this stat exists
    sample = next(iter(parts.values()))
    if mean_key not in sample:
        return None, None

    # Accumulators: sum of weights, sum of weighted values, sum of weighted squared values
    W = np.zeros(G, dtype=np.float64)
    S1 = np.zeros(G, dtype=np.float64)  # sum(w * x)
    S2 = np.zeros(G, dtype=np.float64)  # sum(w * x^2)

    for r, d in parts.items():
        lids = d["label_ids"].astype(np.int64)
        gids = (np.uint64(r) << np.uint64(32)) + lids.astype(np.uint64)
        idx = np.array([root_to_idx[roots[g]] for g in gids], dtype=np.int64)

        w = d[weight_key].astype(np.float64)
        mu = d[mean_key].astype(np.float64)
        sigma = d[std_key].astype(np.float64)

        np.add.at(W, idx, w)
        np.add.at(S1, idx, w * mu)
        np.add.at(S2, idx, w * (sigma**2 + mu**2))  # E[X^2] = Var(X) + E[X]^2

    small = 1e-300
    mean = S1 / (W + small)
    var = S2 / (W + small) - mean**2
    np.maximum(var, 0.0, out=var)
    std = np.sqrt(var)

    return mean, std


def stitch_reduce(input_dir: str, output_path: str):
    ranks, cart_dims, periodic = index_parts(input_dir)
    any_npz = _load_npz(next(iter(ranks.values()))["npz"])
    dx, dy, dz = (float(any_npz["voxel_spacing"][0]),
                  float(any_npz["voxel_spacing"][1]),
                  float(any_npz["voxel_spacing"][2]))

    dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
    dsu = _merge_by_overlap_planes(ranks, cart_dims, periodic, dsu=dsu)

    all_gids: list[np.uint64] = []
    parts: Dict[int, Dict[str, np.ndarray]] = {}
    for r, info in ranks.items():
        d = _load_npz(info["npz"])
        lids = d["label_ids"].astype(np.int64)
        gids = (_gid(r, 0) + lids.astype(np.uint64))
        all_gids.extend(list(gids))
        parts[r] = d

    roots = {g: dsu.find(g) for g in all_gids}
    uniq_roots = sorted(set(roots.values()))
    root_to_idx = {rt: i for i, rt in enumerate(uniq_roots)}
    G = len(uniq_roots)

    # Track which global clumps were stitched (span multiple ranks)
    # A clump is "stitched" if multiple local labels map to the same root
    stitched_count = np.zeros(G, dtype=np.int32)  # number of local labels per global clump
    for g in all_gids:
        stitched_count[root_to_idx[roots[g]]] += 1
    is_stitched = stitched_count > 1  # True if clump spans multiple ranks

    # Check if extra stats are available
    has_extra_stats = "vx_mean" in any_npz

    cell_count = np.zeros(G, dtype=np.int64)
    volume = np.zeros(G, dtype=np.float64)
    mass = np.zeros(G, dtype=np.float64)
    Sxv = np.zeros(G, dtype=np.float64)
    Syv = np.zeros(G, dtype=np.float64)
    Szv = np.zeros(G, dtype=np.float64)
    Sxm = np.zeros(G, dtype=np.float64)
    Sym = np.zeros(G, dtype=np.float64)
    Szm = np.zeros(G, dtype=np.float64)
    bbox = np.zeros((G, 6), dtype=np.int64)
    bbox[:, 0::2] = np.iinfo(np.int64).max
    bbox[:, 1::2] = np.iinfo(np.int64).min
    area = np.zeros(G, dtype=np.float64)
    speed_w = np.zeros(G, dtype=np.float64)
    speed_w2 = np.zeros(G, dtype=np.float64)

    # Extra stats accumulators (if available)
    if has_extra_stats:
        euler_chi = np.zeros(G, dtype=np.int64)
        # Covariance tensor sums (mass-weighted)
        cov_W = np.zeros(G, dtype=np.float64)
        cov_Sx = np.zeros(G, dtype=np.float64)
        cov_Sy = np.zeros(G, dtype=np.float64)
        cov_Sz = np.zeros(G, dtype=np.float64)
        cov_Sxx = np.zeros(G, dtype=np.float64)
        cov_Syy = np.zeros(G, dtype=np.float64)
        cov_Szz = np.zeros(G, dtype=np.float64)
        cov_Sxy = np.zeros(G, dtype=np.float64)
        cov_Sxz = np.zeros(G, dtype=np.float64)
        cov_Syz = np.zeros(G, dtype=np.float64)

    for r, d in parts.items():
        lids = d["label_ids"].astype(np.int64)
        gids = (_gid(r, 0) + lids.astype(np.uint64))
        idx = np.array([root_to_idx[roots[g]] for g in gids], dtype=np.int64)

        cc = d["cell_count"].astype(np.int64)
        vol = d["volume"].astype(np.float64)
        ms = d["mass"].astype(np.float64)
        ar = d["area"].astype(np.float64)
        np.add.at(cell_count, idx, cc)
        np.add.at(volume, idx, vol)
        np.add.at(mass, idx, ms)
        np.add.at(area, idx, ar)
        if "velocity_mean" in d and "velocity_std" in d:
            v_mean_local = d["velocity_mean"].astype(np.float64, copy=False)
            v_std_local = d["velocity_std"].astype(np.float64, copy=False)
            np.add.at(speed_w, idx, v_mean_local * cc)
            np.add.at(speed_w2, idx, (v_std_local * v_std_local + v_mean_local * v_mean_local) * cc)

        cv = d["centroid_vol"].astype(np.float64)
        cm = d["centroid_mass"].astype(np.float64)
        np.add.at(Sxv, idx, cv[:, 0] * vol)
        np.add.at(Syv, idx, cv[:, 1] * vol)
        np.add.at(Szv, idx, cv[:, 2] * vol)
        np.add.at(Sxm, idx, cm[:, 0] * ms)
        np.add.at(Sym, idx, cm[:, 1] * ms)
        np.add.at(Szm, idx, cm[:, 2] * ms)

        bb = d["bbox_ijk"].astype(np.int64)
        np.minimum.at(bbox[:, 0], idx, bb[:, 0])
        np.minimum.at(bbox[:, 2], idx, bb[:, 2])
        np.minimum.at(bbox[:, 4], idx, bb[:, 4])
        np.maximum.at(bbox[:, 1], idx, bb[:, 1])
        np.maximum.at(bbox[:, 3], idx, bb[:, 3])
        np.maximum.at(bbox[:, 5], idx, bb[:, 5])

        # Extra stats
        if has_extra_stats:
            if "euler_characteristic" in d:
                np.add.at(euler_chi, idx, d["euler_characteristic"].astype(np.int64))

            # Accumulate raw covariance tensor sums (if available)
            if "cov_W" in d:
                # New format: raw sums stored directly
                np.add.at(cov_W, idx, d["cov_W"].astype(np.float64))
                np.add.at(cov_Sx, idx, d["cov_Sx"].astype(np.float64))
                np.add.at(cov_Sy, idx, d["cov_Sy"].astype(np.float64))
                np.add.at(cov_Sz, idx, d["cov_Sz"].astype(np.float64))
                np.add.at(cov_Sxx, idx, d["cov_Sxx"].astype(np.float64))
                np.add.at(cov_Syy, idx, d["cov_Syy"].astype(np.float64))
                np.add.at(cov_Szz, idx, d["cov_Szz"].astype(np.float64))
                np.add.at(cov_Sxy, idx, d["cov_Sxy"].astype(np.float64))
                np.add.at(cov_Sxz, idx, d["cov_Sxz"].astype(np.float64))
                np.add.at(cov_Syz, idx, d["cov_Syz"].astype(np.float64))
            elif "principal_axes_lengths" in d:
                # Legacy fallback: approximate from principal axes (inaccurate for stitched)
                np.add.at(cov_W, idx, ms)
                np.add.at(cov_Sx, idx, cm[:, 0] * ms)
                np.add.at(cov_Sy, idx, cm[:, 1] * ms)
                np.add.at(cov_Sz, idx, cm[:, 2] * ms)
                pal = d["principal_axes_lengths"].astype(np.float64)
                np.add.at(cov_Sxx, idx, (pal[:, 0]**2 + cm[:, 0]**2) * ms)
                np.add.at(cov_Syy, idx, (pal[:, 1]**2 + cm[:, 1]**2) * ms)
                np.add.at(cov_Szz, idx, (pal[:, 2]**2 + cm[:, 2]**2) * ms)

    for axis_key, ec in edge_counts.items():
        af = face_area[axis_key]
        for (ga, gb), cnt in ec.items():
            ra = roots.get(ga, dsu.find(ga))
            rb = roots.get(gb, dsu.find(gb))
            if ra == rb:
                i = root_to_idx[ra]
                area[i] -= 2.0 * af * float(cnt)

    small = 1e-300
    centroid_vol = np.stack([Sxv / (volume + small),
                             Syv / (volume + small),
                             Szv / (volume + small)], axis=1)
    centroid_mass = np.stack([Sxm / (mass + small),
                              Sym / (mass + small),
                              Szm / (mass + small)], axis=1)
    speed_mean = speed_w / (cell_count + small)
    speed_var = speed_w2 / (cell_count + small) - speed_mean * speed_mean
    np.maximum(speed_var, 0.0, out=speed_var)
    speed_std = np.sqrt(speed_var)

    out = {
        "gid": np.array(uniq_roots, dtype=np.uint64),
        "cell_count": cell_count,
        "volume": volume,
        "mass": mass,
        "area": area,
        "centroid_vol": centroid_vol,
        "centroid_mass": centroid_mass,
        "velocity_mean": speed_mean,
        "velocity_std": speed_std,
        "bbox_ijk": bbox.astype(np.int32),
        "voxel_spacing": np.array([dx, dy, dz], dtype=np.float64),
        "connectivity": np.int32(6),
        # Stitching metadata
        "is_stitched": is_stitched,  # True if clump spans multiple ranks
        "n_fragments": stitched_count,  # Number of local labels that were merged
    }

    # Add extra stats if available
    if has_extra_stats:
        # Component-wise velocity (volume-weighted)
        for comp in ["vx", "vy", "vz"]:
            mu, sigma = _combine_weighted_stats(G, parts, roots, root_to_idx, comp, "volume")
            if mu is not None:
                out[f"{comp}_mean"] = mu
                out[f"{comp}_std"] = sigma

        # Thermodynamic stats (volume-weighted)
        for stat in ["rho", "T", "pressure"]:
            mu, sigma = _combine_weighted_stats(G, parts, roots, root_to_idx, stat, "volume")
            if mu is not None:
                out[f"{stat}_mean"] = mu
                out[f"{stat}_std"] = sigma

        # Mass-weighted versions
        for stat in ["rho", "T", "vx", "vy", "vz", "pressure"]:
            mu, sigma = _combine_weighted_stats(G, parts, roots, root_to_idx, f"{stat}_massw", "mass")
            if mu is not None:
                out[f"{stat}_mean_massw"] = mu
                out[f"{stat}_std_massw"] = sigma

        # Euler characteristic (additive)
        out["euler_characteristic"] = euler_chi

        # Compute combined principal axes from covariance sums
        mu_x = cov_Sx / (cov_W + small)
        mu_y = cov_Sy / (cov_W + small)
        mu_z = cov_Sz / (cov_W + small)
        Cxx = cov_Sxx / (cov_W + small) - mu_x**2
        Cyy = cov_Syy / (cov_W + small) - mu_y**2
        Czz = cov_Szz / (cov_W + small) - mu_z**2
        Cxy = cov_Sxy / (cov_W + small) - mu_x * mu_y
        Cxz = cov_Sxz / (cov_W + small) - mu_x * mu_z
        Cyz = cov_Syz / (cov_W + small) - mu_y * mu_z

        # Check if we have full covariance data (new format with cov_Sxy etc.)
        has_full_cov = np.any(cov_Sxy != 0) or np.any(cov_Sxz != 0) or np.any(cov_Syz != 0)

        principal_axes_lengths = np.zeros((G, 3), dtype=np.float64)
        axis_ratios = np.zeros((G, 2), dtype=np.float64)
        orientation = np.zeros((G, 3, 3), dtype=np.float64)

        for i in range(G):
            if has_full_cov:
                # Full covariance tensor available - compute proper eigendecomposition
                C = np.array([[Cxx[i], Cxy[i], Cxz[i]],
                              [Cxy[i], Cyy[i], Cyz[i]],
                              [Cxz[i], Cyz[i], Czz[i]]], dtype=np.float64)
                C = (C + C.T) * 0.5  # ensure symmetry
                vals, vecs = np.linalg.eigh(C)
                order = np.argsort(vals)[::-1]
                vals = vals[order]
                vecs = vecs[:, order]
                orientation[i] = vecs
            else:
                # Diagonal only (legacy fallback)
                vals = np.array([Cxx[i], Cyy[i], Czz[i]])
                vals = np.sort(vals)[::-1]

            a = np.sqrt(max(vals[0], 0.0))
            b = np.sqrt(max(vals[1], 0.0))
            c = np.sqrt(max(vals[2], 0.0))
            principal_axes_lengths[i] = (a, b, c)
            axis_ratios[i] = (b / (a + small), c / (a + small))

        out["principal_axes_lengths"] = principal_axes_lengths
        out["axis_ratios"] = axis_ratios
        if has_full_cov:
            out["orientation"] = orientation

        # Shape metrics are now valid for all clumps if we have full covariance data
        out["shape_metrics_valid"] = np.ones(G, dtype=bool) if has_full_cov else ~is_stitched

        # Derived shape metrics
        r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / (area + small)
        compactness = 36.0 * np.pi * volume**2 / (area**3 + small)

        a = principal_axes_lengths[:, 0]
        b = principal_axes_lengths[:, 1]
        c = principal_axes_lengths[:, 2]
        triaxiality = (a**2 - b**2) / (a**2 - c**2 + small)
        elongation = a / (c + small)

        out["r_eff"] = r_eff
        out["sphericity"] = sphericity
        out["compactness"] = compactness
        out["triaxiality"] = triaxiality
        out["elongation"] = elongation

        # Minkowski shapefinders (from V, S, C)
        C_integrated = euler_chi * 4.0 * np.pi  # Approximate
        thickness = volume / (area + small)
        breadth = area / (C_integrated + small)
        length_mink = C_integrated / (4.0 * np.pi)

        out["integrated_curvature"] = C_integrated
        out["thickness"] = thickness
        out["breadth"] = breadth
        out["length"] = length_mink

        # Shapefinders
        T1 = thickness
        T2 = breadth
        T3 = length_mink
        planarity = (T2 - T1) / (T2 + T1 + small)
        filamentarity = (T3 - T2) / (T3 + T2 + small)
        out["planarity"] = planarity
        out["filamentarity"] = filamentarity

    # Prepare output paths
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base_path = output_path.replace('.npz', '')
    if base_path == output_path:
        base_path = output_path  # no .npz extension

    # 1. Primary output: basic clump properties
    primary_out = {
        "gid": np.array(uniq_roots, dtype=np.uint64),
        "cell_count": cell_count,
        "volume": volume,
        "mass": mass,
        "area": area,
        "centroid_vol": centroid_vol,
        "centroid_mass": centroid_mass,
        "velocity_mean": speed_mean,
        "velocity_std": speed_std,
        "bbox_ijk": bbox.astype(np.int32),
        "voxel_spacing": np.array([dx, dy, dz], dtype=np.float64),
        "connectivity": np.int32(6),
        "is_stitched": is_stitched,
        "n_fragments": stitched_count,
    }
    np.savez(output_path, **primary_out)

    # Additional files only if extra stats available
    if has_extra_stats:
        # 2. Thermodynamic moments
        thermo_out = {"gid": np.array(uniq_roots, dtype=np.uint64)}
        for comp in ["vx", "vy", "vz"]:
            mu, sigma = _combine_weighted_stats(G, parts, roots, root_to_idx, comp, "volume")
            if mu is not None:
                thermo_out[f"{comp}_mean"] = mu
                thermo_out[f"{comp}_std"] = sigma
        for stat in ["rho", "T", "pressure"]:
            mu, sigma = _combine_weighted_stats(G, parts, roots, root_to_idx, stat, "volume")
            if mu is not None:
                thermo_out[f"{stat}_mean"] = mu
                thermo_out[f"{stat}_std"] = sigma
        for stat in ["rho", "T", "vx", "vy", "vz", "pressure"]:
            mu, sigma = _combine_weighted_stats(G, parts, roots, root_to_idx, f"{stat}_massw", "mass")
            if mu is not None:
                thermo_out[f"{stat}_mean_massw"] = mu
                thermo_out[f"{stat}_std_massw"] = sigma
        np.savez(f"{base_path}_thermo.npz", **thermo_out)

        # 3. Shape metrics
        shape_out = {
            "gid": np.array(uniq_roots, dtype=np.uint64),
            "principal_axes_lengths": principal_axes_lengths,
            "axis_ratios": axis_ratios,
            "shape_metrics_valid": out["shape_metrics_valid"],
            "r_eff": r_eff,
            "sphericity": sphericity,
            "compactness": compactness,
            "triaxiality": triaxiality,
            "elongation": elongation,
            "euler_characteristic": euler_chi,
            "integrated_curvature": C_integrated,
            "thickness": thickness,
            "breadth": breadth,
            "length": length_mink,
            "planarity": planarity,
            "filamentarity": filamentarity,
        }
        if has_full_cov:
            shape_out["orientation"] = orientation
        np.savez(f"{base_path}_shape.npz", **shape_out)

        # 4. Raw moment sums (for potential reanalysis)
        moments_out = {
            "gid": np.array(uniq_roots, dtype=np.uint64),
            "cov_W": cov_W,
            "cov_Sx": cov_Sx,
            "cov_Sy": cov_Sy,
            "cov_Sz": cov_Sz,
            "cov_Sxx": cov_Sxx,
            "cov_Syy": cov_Syy,
            "cov_Szz": cov_Szz,
            "cov_Sxy": cov_Sxy,
            "cov_Sxz": cov_Sxz,
            "cov_Syz": cov_Syz,
        }
        np.savez(f"{base_path}_moments.npz", **moments_out)

    return primary_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz + .meta.json")
    ap.add_argument("--output", required=True, help="stitched npz path (also creates _thermo, _shape, _moments)")
    args = ap.parse_args()
    out = stitch_reduce(args.input, args.output)
    print(f"Stitched {out['gid'].size} global clumps -> {args.output}")


if __name__ == "__main__":
    main()
