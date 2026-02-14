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
from typing import Dict, Iterable, Tuple

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
    periodic = None
    seen_coords = set()
    for m in metas:
        d = _load_meta(m)
        r = int(d["rank"])
        if r in ranks:
            raise ValueError(f"Duplicate rank metadata encountered: rank={r}")
        min_cells_deferred = int(d.get("stitching", {}).get("min_clump_cells_deferred", 1))
        coords = tuple(d["coords"])
        if coords in seen_coords:
            raise ValueError(f"Duplicate Cartesian coordinates in metadata: coords={coords}")
        seen_coords.add(coords)
        dims = tuple(d["cart_dims"])
        if cart_dims is None:
            cart_dims = dims
        elif cart_dims != dims:
            raise ValueError(f"Inconsistent cart_dims across rank metadata: {cart_dims} vs {dims}")
        p_here = tuple(bool(x) for x in d.get("grid", {}).get("periodic", (True, True, True)))
        if periodic is None:
            periodic = p_here
        elif periodic != p_here:
            raise ValueError(f"Inconsistent periodic flags across rank metadata: {periodic} vs {p_here}")
        ranks[r] = {
            "coords": coords,
            "bbox": tuple(d["node_bbox_ijk"]),
            "npz": os.path.join(input_dir, d["output_npz"]),
            "meta": m,
            "min_clump_cells_deferred": min_cells_deferred,
        }
    if periodic is None:
        periodic = (True, True, True)
    return ranks, cart_dims, periodic


def build_edges(ranks: dict,
                cart_dims: Tuple[int, int, int],
                periodic: Tuple[bool, bool, bool],
                dx: float, dy: float, dz: float):
    dsu = DSU()
    edge_counts = {"x": {}, "y": {}, "z": {}}  # dict[(gid_a,gid_b)] = count
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}
    cache: Dict[int, Dict[str, np.ndarray]] = {}

    def load_part(rank_id: int) -> Dict[str, np.ndarray]:
        d = cache.get(rank_id)
        if d is None:
            d = _load_npz(ranks[rank_id]["npz"])
            cache[rank_id] = d
        return d

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
        npz = load_part(r)
        coords = tuple(info["coords"])

        ncoords = _neighbor(coords, cart_dims, axis=0, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = load_part(rn)
            add_edges("x", r, rn, npz["face_xpos"], npz_n["face_xneg"])

        ncoords = _neighbor(coords, cart_dims, axis=1, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = load_part(rn)
            add_edges("y", r, rn, npz["face_ypos"], npz_n["face_yneg"])

        ncoords = _neighbor(coords, cart_dims, axis=2, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = load_part(rn)
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
                             stat_name: str | None = None, weight_key: str = "cell_count",
                             mean_key: str | None = None, std_key: str | None = None):
    """Combine weighted mean and std across ranks using parallel variance formula.

    Returns (mean, std) arrays of shape (G,).
    """
    if mean_key is None:
        if stat_name is None:
            raise ValueError("Either stat_name or mean_key/std_key must be provided")
        mean_key = f"{stat_name}_mean"
    if std_key is None:
        if stat_name is None:
            raise ValueError("Either stat_name or mean_key/std_key must be provided")
        std_key = f"{stat_name}_std"

    have_pair = {r: (mean_key in d and std_key in d) for r, d in parts.items()}
    if not any(have_pair.values()):
        return None, None
    if not all(have_pair.values()):
        missing_ranks = sorted(r for r, ok in have_pair.items() if not ok)
        raise ValueError(
            f"Inconsistent statistics fields '{mean_key}/{std_key}' across ranks; "
            f"missing on ranks {missing_ranks}"
        )
    missing_weights = sorted(r for r, d in parts.items() if weight_key not in d)
    if missing_weights:
        raise ValueError(f"Missing required weight field '{weight_key}' on ranks {missing_weights}")

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


def _uniform_key_presence(parts: Dict[int, Dict[str, np.ndarray]], key: str) -> bool:
    """Return True if key exists on all parts, False if absent on all, else raise."""
    present = sorted(r for r, d in parts.items() if key in d)
    if not present:
        return False
    if len(present) != len(parts):
        missing = sorted(r for r in parts if r not in present)
        raise ValueError(f"Inconsistent optional field '{key}' across ranks; missing on ranks {missing}")
    return True


def _uniform_group_presence(parts: Dict[int, Dict[str, np.ndarray]], keys: Iterable[str],
                            group_name: str) -> bool:
    """Return True if all keys exist on all parts, False if absent on all, else raise."""
    keys = tuple(keys)
    fully_present = sorted(r for r, d in parts.items() if all(k in d for k in keys))
    if not fully_present:
        if any(any(k in d for k in keys) for d in parts.values()):
            bad = sorted(r for r, d in parts.items() if any(k in d for k in keys) and not all(k in d for k in keys))
            raise ValueError(f"Incomplete {group_name} fields on ranks {bad}; expected keys={keys}")
        return False
    if len(fully_present) != len(parts):
        missing = sorted(r for r in parts if r not in fully_present)
        raise ValueError(f"Inconsistent {group_name} fields across ranks; missing on ranks {missing}")
    return True


def stitch_reduce(input_dir: str, output_path: str):
    ranks, cart_dims, periodic = index_parts(input_dir)
    deferred_values = sorted({int(info.get("min_clump_cells_deferred", 1)) for info in ranks.values()})
    if len(deferred_values) > 1:
        raise ValueError(f"Inconsistent min_clump_cells_deferred across rank metadata: {deferred_values}")
    min_clump_cells = deferred_values[0] if deferred_values else 1
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
        if "voxel_spacing" not in d:
            raise KeyError(f"Missing required field 'voxel_spacing' in {info['npz']}")
        spacing = np.asarray(d["voxel_spacing"], dtype=np.float64).ravel()
        if spacing.shape[0] < 3:
            raise ValueError(f"Invalid voxel_spacing shape in {info['npz']}: {spacing.shape}")
        if not np.allclose(spacing[:3], np.array([dx, dy, dz], dtype=np.float64), rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Inconsistent voxel_spacing across parts: expected {[dx, dy, dz]}, "
                f"rank {r} has {spacing[:3].tolist()}"
            )
        lids = d["label_ids"].astype(np.int64)
        gids = (_gid(r, 0) + lids.astype(np.uint64))
        all_gids.extend(list(gids))
        parts[r] = d

    cov_base_keys = ("cov_W", "cov_Sx", "cov_Sy", "cov_Sz", "cov_Sxx", "cov_Syy", "cov_Szz")
    cov_cross_keys = ("cov_Sxy", "cov_Sxz", "cov_Syz")
    has_cov_base = _uniform_group_presence(parts, cov_base_keys, "covariance-base")
    has_cov_cross = _uniform_group_presence(parts, cov_cross_keys, "covariance-cross")
    if has_cov_cross and not has_cov_base:
        raise ValueError("Found covariance cross terms without covariance base terms")
    # Legacy principal-axis fallback is only relevant when covariance sums are unavailable.
    has_legacy_axes = _uniform_key_presence(parts, "principal_axes_lengths") if not has_cov_base else False
    has_euler = _uniform_key_presence(parts, "euler_characteristic")
    has_shape_stats = has_cov_base or has_legacy_axes

    global_i0 = min(int(info["bbox"][0]) for info in ranks.values())
    global_i1 = max(int(info["bbox"][1]) for info in ranks.values())
    global_j0 = min(int(info["bbox"][2]) for info in ranks.values())
    global_j1 = max(int(info["bbox"][3]) for info in ranks.values())
    global_k0 = min(int(info["bbox"][4]) for info in ranks.values())
    global_k1 = max(int(info["bbox"][5]) for info in ranks.values())
    domain_cells = np.array([global_i1 - global_i0, global_j1 - global_j0, global_k1 - global_k0], dtype=np.float64)
    domain_lengths = domain_cells * np.array([dx, dy, dz], dtype=np.float64)
    origin = np.asarray(any_npz.get("origin", np.zeros(3, dtype=np.float64)), dtype=np.float64).ravel()
    if origin.size < 3:
        origin = np.pad(origin, (0, 3 - origin.size), mode="constant")
    domain_start = origin[:3] + np.array([global_i0 * dx, global_j0 * dy, global_k0 * dz], dtype=np.float64)

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

    cell_count = np.zeros(G, dtype=np.int64)
    volume = np.zeros(G, dtype=np.float64)
    mass = np.zeros(G, dtype=np.float64)
    Sxv = np.zeros(G, dtype=np.float64)
    Syv = np.zeros(G, dtype=np.float64)
    Szv = np.zeros(G, dtype=np.float64)
    Sxm = np.zeros(G, dtype=np.float64)
    Sym = np.zeros(G, dtype=np.float64)
    Szm = np.zeros(G, dtype=np.float64)
    Sxv_cos = np.zeros(G, dtype=np.float64)
    Syv_cos = np.zeros(G, dtype=np.float64)
    Szv_cos = np.zeros(G, dtype=np.float64)
    Sxv_sin = np.zeros(G, dtype=np.float64)
    Syv_sin = np.zeros(G, dtype=np.float64)
    Szv_sin = np.zeros(G, dtype=np.float64)
    Sxm_cos = np.zeros(G, dtype=np.float64)
    Sym_cos = np.zeros(G, dtype=np.float64)
    Szm_cos = np.zeros(G, dtype=np.float64)
    Sxm_sin = np.zeros(G, dtype=np.float64)
    Sym_sin = np.zeros(G, dtype=np.float64)
    Szm_sin = np.zeros(G, dtype=np.float64)
    bbox = np.zeros((G, 6), dtype=np.int64)
    bbox[:, 0::2] = np.iinfo(np.int64).max
    bbox[:, 1::2] = np.iinfo(np.int64).min
    area = np.zeros(G, dtype=np.float64)
    speed_w = np.zeros(G, dtype=np.float64)
    speed_w2 = np.zeros(G, dtype=np.float64)

    if has_euler:
        euler_chi = np.zeros(G, dtype=np.int64)
    if has_shape_stats:
        # Covariance tensor sums (mass-weighted). For legacy fallback these are
        # reconstructed approximately from principal axes.
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
        if periodic[0] and domain_lengths[0] > 0.0:
            theta_v = 2.0 * np.pi * (cv[:, 0] - domain_start[0]) / domain_lengths[0]
            theta_m = 2.0 * np.pi * (cm[:, 0] - domain_start[0]) / domain_lengths[0]
            np.add.at(Sxv_cos, idx, vol * np.cos(theta_v))
            np.add.at(Sxv_sin, idx, vol * np.sin(theta_v))
            np.add.at(Sxm_cos, idx, ms * np.cos(theta_m))
            np.add.at(Sxm_sin, idx, ms * np.sin(theta_m))
        if periodic[1] and domain_lengths[1] > 0.0:
            theta_v = 2.0 * np.pi * (cv[:, 1] - domain_start[1]) / domain_lengths[1]
            theta_m = 2.0 * np.pi * (cm[:, 1] - domain_start[1]) / domain_lengths[1]
            np.add.at(Syv_cos, idx, vol * np.cos(theta_v))
            np.add.at(Syv_sin, idx, vol * np.sin(theta_v))
            np.add.at(Sym_cos, idx, ms * np.cos(theta_m))
            np.add.at(Sym_sin, idx, ms * np.sin(theta_m))
        if periodic[2] and domain_lengths[2] > 0.0:
            theta_v = 2.0 * np.pi * (cv[:, 2] - domain_start[2]) / domain_lengths[2]
            theta_m = 2.0 * np.pi * (cm[:, 2] - domain_start[2]) / domain_lengths[2]
            np.add.at(Szv_cos, idx, vol * np.cos(theta_v))
            np.add.at(Szv_sin, idx, vol * np.sin(theta_v))
            np.add.at(Szm_cos, idx, ms * np.cos(theta_m))
            np.add.at(Szm_sin, idx, ms * np.sin(theta_m))

        bb = d["bbox_ijk"].astype(np.int64)
        np.minimum.at(bbox[:, 0], idx, bb[:, 0])
        np.minimum.at(bbox[:, 2], idx, bb[:, 2])
        np.minimum.at(bbox[:, 4], idx, bb[:, 4])
        np.maximum.at(bbox[:, 1], idx, bb[:, 1])
        np.maximum.at(bbox[:, 3], idx, bb[:, 3])
        np.maximum.at(bbox[:, 5], idx, bb[:, 5])

        if has_euler:
            np.add.at(euler_chi, idx, d["euler_characteristic"].astype(np.int64))

        if has_shape_stats:
            if has_cov_base:
                np.add.at(cov_W, idx, d["cov_W"].astype(np.float64))
                np.add.at(cov_Sx, idx, d["cov_Sx"].astype(np.float64))
                np.add.at(cov_Sy, idx, d["cov_Sy"].astype(np.float64))
                np.add.at(cov_Sz, idx, d["cov_Sz"].astype(np.float64))
                np.add.at(cov_Sxx, idx, d["cov_Sxx"].astype(np.float64))
                np.add.at(cov_Syy, idx, d["cov_Syy"].astype(np.float64))
                np.add.at(cov_Szz, idx, d["cov_Szz"].astype(np.float64))
                if has_cov_cross:
                    np.add.at(cov_Sxy, idx, d["cov_Sxy"].astype(np.float64))
                    np.add.at(cov_Sxz, idx, d["cov_Sxz"].astype(np.float64))
                    np.add.at(cov_Syz, idx, d["cov_Syz"].astype(np.float64))
            else:
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
    if periodic[0] and domain_lengths[0] > 0.0:
        ang_v = np.arctan2(Sxv_sin, Sxv_cos)
        ang_m = np.arctan2(Sxm_sin, Sxm_cos)
        cand_v = domain_start[0] + np.mod(ang_v, 2.0 * np.pi) * (domain_lengths[0] / (2.0 * np.pi))
        cand_m = domain_start[0] + np.mod(ang_m, 2.0 * np.pi) * (domain_lengths[0] / (2.0 * np.pi))
        unambig_v = np.hypot(Sxv_cos, Sxv_sin) > (1e-12 * (volume + small))
        unambig_m = np.hypot(Sxm_cos, Sxm_sin) > (1e-12 * (mass + small))
        centroid_vol[:, 0] = np.where(unambig_v, cand_v, centroid_vol[:, 0])
        centroid_mass[:, 0] = np.where(unambig_m, cand_m, centroid_mass[:, 0])
    if periodic[1] and domain_lengths[1] > 0.0:
        ang_v = np.arctan2(Syv_sin, Syv_cos)
        ang_m = np.arctan2(Sym_sin, Sym_cos)
        cand_v = domain_start[1] + np.mod(ang_v, 2.0 * np.pi) * (domain_lengths[1] / (2.0 * np.pi))
        cand_m = domain_start[1] + np.mod(ang_m, 2.0 * np.pi) * (domain_lengths[1] / (2.0 * np.pi))
        unambig_v = np.hypot(Syv_cos, Syv_sin) > (1e-12 * (volume + small))
        unambig_m = np.hypot(Sym_cos, Sym_sin) > (1e-12 * (mass + small))
        centroid_vol[:, 1] = np.where(unambig_v, cand_v, centroid_vol[:, 1])
        centroid_mass[:, 1] = np.where(unambig_m, cand_m, centroid_mass[:, 1])
    if periodic[2] and domain_lengths[2] > 0.0:
        ang_v = np.arctan2(Szv_sin, Szv_cos)
        ang_m = np.arctan2(Szm_sin, Szm_cos)
        cand_v = domain_start[2] + np.mod(ang_v, 2.0 * np.pi) * (domain_lengths[2] / (2.0 * np.pi))
        cand_m = domain_start[2] + np.mod(ang_m, 2.0 * np.pi) * (domain_lengths[2] / (2.0 * np.pi))
        unambig_v = np.hypot(Szv_cos, Szv_sin) > (1e-12 * (volume + small))
        unambig_m = np.hypot(Szm_cos, Szm_sin) > (1e-12 * (mass + small))
        centroid_vol[:, 2] = np.where(unambig_v, cand_v, centroid_vol[:, 2])
        centroid_mass[:, 2] = np.where(unambig_m, cand_m, centroid_mass[:, 2])
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
        mu, sigma = _combine_weighted_stats(
            G, parts, roots, root_to_idx,
            weight_key="mass",
            mean_key=f"{stat}_mean_massw",
            std_key=f"{stat}_std_massw",
        )
        if mu is not None:
            out[f"{stat}_mean_massw"] = mu
            out[f"{stat}_std_massw"] = sigma

    if has_euler:
        out["euler_characteristic"] = euler_chi

    if has_shape_stats:
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

        has_full_cov = has_cov_base and has_cov_cross

        principal_axes_lengths = np.zeros((G, 3), dtype=np.float64)
        axis_ratios = np.zeros((G, 2), dtype=np.float64)
        orientation = np.zeros((G, 3, 3), dtype=np.float64)
        shape_metrics_valid = np.ones(G, dtype=bool) if has_full_cov else ~is_stitched

        for i in range(G):
            if has_full_cov:
                # Full covariance tensor available - compute proper eigendecomposition
                C = np.array([[Cxx[i], Cxy[i], Cxz[i]],
                              [Cxy[i], Cyy[i], Cyz[i]],
                              [Cxz[i], Cyz[i], Czz[i]]], dtype=np.float64)
                C = (C + C.T) * 0.5  # ensure symmetry
                # Skip eigendecomposition if covariance has inf/nan (overflow)
                if not np.all(np.isfinite(C)):
                    principal_axes_lengths[i] = (np.nan, np.nan, np.nan)
                    axis_ratios[i] = (np.nan, np.nan)
                    orientation[i] = np.eye(3)
                    shape_metrics_valid[i] = False
                    continue
                try:
                    vals, vecs = np.linalg.eigh(C)
                except np.linalg.LinAlgError:
                    principal_axes_lengths[i] = (np.nan, np.nan, np.nan)
                    axis_ratios[i] = (np.nan, np.nan)
                    orientation[i] = np.eye(3)
                    shape_metrics_valid[i] = False
                    continue
                order = np.argsort(vals)[::-1]
                vals = vals[order]
                vecs = vecs[:, order]
                orientation[i] = vecs
            else:
                # Diagonal only (legacy fallback)
                vals = np.array([Cxx[i], Cyy[i], Czz[i]])
                vals = np.sort(vals)[::-1]

            # Raw RMS extents from eigenvalues
            a_raw = np.sqrt(max(vals[0], 0.0))
            b_raw = np.sqrt(max(vals[1], 0.0))
            c_raw = np.sqrt(max(vals[2], 0.0))

            V_clump = volume[i]
            # Minimum axis length: half a voxel (cells have finite extent)
            axis_min = 0.5 * (dx + dy + dz) / 3.0

            # For degenerate cases (point-like clumps with no spatial extent),
            # use cube approximation: a = b = c = V^(1/3)
            if a_raw < 1e-10 or V_clump <= 0:
                side = V_clump ** (1.0 / 3.0) if V_clump > 0 else 0.0
                a, b, c = side, side, side
            else:
                # Normalize so a*b*c = V while preserving shape ratios
                abc_raw = a_raw * b_raw * c_raw
                if abc_raw > 1e-30:
                    scale = (V_clump / abc_raw) ** (1.0 / 3.0)
                    a = a_raw * scale
                    b = b_raw * scale
                    c = c_raw * scale
                else:
                    # Near-degenerate (very thin filament): use cube approximation
                    side = V_clump ** (1.0 / 3.0)
                    a, b, c = side, side, side

            # Enforce minimum b,c based on finite cell size, then recalculate a
            # to preserve volume. This prevents unphysical elongation for thin
            # filaments where covariance eigenvalues approach zero.
            if c < axis_min:
                c = axis_min
            if b < axis_min:
                b = axis_min
            # Recalculate a to preserve a*b*c = V
            if b * c > 0:
                a = V_clump / (b * c)

            principal_axes_lengths[i] = (a, b, c)
            axis_ratios[i] = (b / (a + small), c / (a + small))

        out["principal_axes_lengths"] = principal_axes_lengths
        out["axis_ratios"] = axis_ratios
        if has_full_cov:
            out["orientation"] = orientation

        out["shape_metrics_valid"] = shape_metrics_valid

        # Derived shape metrics
        r_eff = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / (area + small)
        compactness = 36.0 * np.pi * volume**2 / (area**3 + small)

        a = principal_axes_lengths[:, 0]
        b = principal_axes_lengths[:, 1]
        c = principal_axes_lengths[:, 2]
        triaxiality = (a**2 - b**2) / (a**2 - c**2 + small)
        # Elongation: axes already have minimum bounds from cell size constraint
        elongation = a / (c + small)

        out["r_eff"] = r_eff
        out["sphericity"] = sphericity
        out["compactness"] = compactness
        out["triaxiality"] = triaxiality
        out["elongation"] = elongation

        # Minkowski shapefinders REMOVED
        # The approximation C_integrated = euler_chi * 4*pi doesn't work for computing
        # proper shapefinders (breadth, length, planarity, filamentarity).
        # Euler characteristic is kept for reference but integrated curvature
        # requires voxel-level boundary information not preserved through stitching.

    # Apply deferred min_clump_cells filtering after global stitching.
    n_before_min_cells = int(out["gid"].shape[0])
    out["min_clump_cells_deferred"] = np.int32(min_clump_cells)
    if min_clump_cells > 1:
        keep = out["cell_count"] >= int(min_clump_cells)
        for key, val in list(out.items()):
            if isinstance(val, np.ndarray) and val.shape[:1] == (n_before_min_cells,):
                out[key] = val[keep]
    out["min_clump_cells_applied"] = np.int32(min_clump_cells)
    out["n_before_min_cells"] = np.int32(n_before_min_cells)
    out["n_after_min_cells"] = np.int32(int(out["gid"].shape[0]))

    # Write everything to single output file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **out)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz + .meta.json")
    ap.add_argument("--output", required=True, help="stitched npz path (also creates _thermo, _shape, _moments)")
    args = ap.parse_args()
    out = stitch_reduce(args.input, args.output)
    print(f"Stitched {out['gid'].size} global clumps -> {args.output}")


if __name__ == "__main__":
    main()
