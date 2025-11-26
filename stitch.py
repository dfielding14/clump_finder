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
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz + .meta.json")
    ap.add_argument("--output", required=True, help="stitched npz path")
    args = ap.parse_args()
    out = stitch_reduce(args.input, args.output)
    print(f"Stitched {out['gid'].size} global clumps -> {args.output}")


if __name__ == "__main__":
    main()
