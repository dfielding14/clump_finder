"""
stitch.py — connectivity-6 stitcher with area correction and exact centroid merge

Builds global clumps by unifying per-rank labels that touch across node faces.
Reads only per-rank .npz files and JSON sidecars; no MPI required.
"""

from __future__ import annotations

import os
import glob
import json
import argparse
from typing import Dict, Tuple, List

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
                dx: float, dy: float, dz: float,
                mode: str = "face"):
    dsu = DSU()
    edge_counts = {"x": {}, "y": {}, "z": {}}  # dict[(gid_a,gid_b)] = count

    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}

    # Neighbor map for pair-closure: neighbors[r][label][face_idx] = set of neighbor gids across that face
    face_index = {"x-": 0, "x+": 1, "y-": 2, "y+": 3, "z-": 4, "z+": 5}
    neighbors: Dict[int, Dict[int, List[set]]] = {}

    def _union_inplane_face(r: int, face: np.ndarray):
        if face.size == 0:
            return
        # right neighbors
        a = face[:, :-1]; b = face[:, 1:]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            for la, lb in zip(a[m], b[m]):
                dsu.union(_gid(r, int(la)), _gid(r, int(lb)))
        # down neighbors
        a = face[:-1, :]; b = face[1:, :]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            for la, lb in zip(a[m], b[m]):
                dsu.union(_gid(r, int(la)), _gid(r, int(lb)))

    def add_edges(axis_key: str, r: int, rn: int, a: np.ndarray, b: np.ndarray):
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return
        A = a[mask].astype(np.uint64, copy=False)
        B = b[mask].astype(np.uint64, copy=False)
        for la, lb in zip(A, B):
            ga = _gid(r, int(la))
            gb = _gid(rn, int(lb))
            if ga == 0 or gb == 0:
                continue
            dsu.union(ga, gb)
            key = (ga, gb) if ga < gb else (gb, ga)
            edge_counts[axis_key][key] = edge_counts[axis_key].get(key, 0) + 1
            # record neighbor for pair-closure
            if mode in ("boundary", "face"):
                # map axis to face sides
                if axis_key == "x":
                    fa_a, fa_b = face_index["x+"], face_index["x-"]
                elif axis_key == "y":
                    fa_a, fa_b = face_index["y+"], face_index["y-"]
                else:
                    fa_a, fa_b = face_index["z+"], face_index["z-"]
                mra = neighbors.setdefault(r, {})
                mrb = neighbors.setdefault(rn, {})
                la_i = int(la); lb_i = int(lb)
                lst_a = mra.setdefault(la_i, [set(), set(), set(), set(), set(), set()])
                lst_b = mrb.setdefault(lb_i, [set(), set(), set(), set(), set(), set()])
                lst_a[fa_a].add(gb)
                lst_b[fa_b].add(ga)

    for r, info in ranks.items():
        npz = _load_npz(info["npz"])
        coords = tuple(info["coords"])
        # In-rank shell unions (optional)
        if mode == "shell":
            fxm = npz["face_xneg"]; fxp = npz["face_xpos"]  # (nj, nk)
            fym = npz["face_yneg"]; fyp = npz["face_ypos"]  # (ni, nk)
            fzm = npz["face_zneg"]; fzp = npz["face_zpos"]  # (ni, nj)
            nj, nk = fxm.shape
            ni = fym.shape[0]
            # In-plane unions
            _union_inplane_face(r, fxm); _union_inplane_face(r, fxp)
            _union_inplane_face(r, fym); _union_inplane_face(r, fyp)
            _union_inplane_face(r, fzm); _union_inplane_face(r, fzp)
            # Edge unions between faces sharing an edge
            # X- with Y- (j=0, i=0) along k
            for la, lb in zip(fxm[0, :], fym[0, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X- with Y+ (j=nj-1, i=0)
            for la, lb in zip(fxm[nj-1, :], fyp[0, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X+ with Y- (j=0, i=ni-1)
            for la, lb in zip(fxp[0, :], fym[ni-1, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X+ with Y+ (j=nj-1, i=ni-1)
            for la, lb in zip(fxp[nj-1, :], fyp[ni-1, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X- with Z- (k=0, i=0) along j
            for la, lb in zip(fxm[:, 0], fzm[0, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X- with Z+ (k=nk-1, i=0)
            for la, lb in zip(fxm[:, nk-1], fzp[0, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X+ with Z- (k=0, i=ni-1)
            for la, lb in zip(fxp[:, 0], fzm[ni-1, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # X+ with Z+ (k=nk-1, i=ni-1)
            for la, lb in zip(fxp[:, nk-1], fzp[ni-1, :]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # Y- with Z- (j=0, k=0) along i
            for la, lb in zip(fym[:, 0], fzm[:, 0]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # Y- with Z+ (j=0, k=nk-1)
            for la, lb in zip(fym[:, nk-1], fzp[:, 0]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # Y+ with Z- (j=nj-1, k=0)
            for la, lb in zip(fyp[:, 0], fzm[:, nj-1]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # Y+ with Z+ (j=nj-1, k=nk-1)
            for la, lb in zip(fyp[:, nk-1], fzp[:, nj-1]):
                if la>0 and lb>0 and la!=lb:
                    dsu.union(_gid(r,int(la)), _gid(r,int(lb)))
            # Corners: unify the three labels at each corner
            corners = [
                (fxm[0, 0],      fym[0, 0],      fzm[0, 0]),
                (fxm[0, nk-1],   fym[0, nk-1],   fzp[0, 0]),
                (fxm[nj-1, 0],   fyp[0, 0],      fzm[0, nj-1]),
                (fxm[nj-1, nk-1],fyp[0, nk-1],   fzp[0, nj-1]),
                (fxp[0, 0],      fym[ni-1, 0],   fzm[ni-1, 0]),
                (fxp[0, nk-1],   fym[ni-1, nk-1],fzp[ni-1, 0]),
                (fxp[nj-1, 0],   fyp[ni-1, 0],   fzm[ni-1, nj-1]),
                (fxp[nj-1, nk-1],fyp[ni-1, nk-1],fzp[ni-1, nj-1])
            ]
            for a,b,c in corners:
                if a>0 and b>0 and a!=b: dsu.union(_gid(r,int(a)), _gid(r,int(b)))
                if a>0 and c>0 and a!=c: dsu.union(_gid(r,int(a)), _gid(r,int(c)))
                if b>0 and c>0 and b!=c: dsu.union(_gid(r,int(b)), _gid(r,int(c)))
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
    # Pair-closure using face_pair_bits if present
    if mode == "boundary":
        for r, info in ranks.items():
            npz = _load_npz(info["npz"])
            bits = npz.get("face_pair_bits", None)
            if bits is None:
                continue
            # bits length may correspond to K_orig; neighbors may include labels > len(bits)
            lab_map = neighbors.get(r, {})
            for lab, face_sets in lab_map.items():
                if lab <= 0:
                    continue
                if lab - 1 < 0 or lab - 1 >= int(bits.shape[0]):
                    continue
                v = int(np.uint16(bits[lab - 1]))
                if v == 0:
                    continue
                pairs = [(0,1),(0,2),(0,3),(0,4),(0,5),
                         (1,2),(1,3),(1,4),(1,5),
                         (2,3),(2,4),(2,5),
                         (3,4),(3,5),
                         (4,5)]
                for bit_idx, (fa, fb) in enumerate(pairs):
                    if (v >> bit_idx) & 1:
                        Sa = face_sets[fa]
                        Sb = face_sets[fb]
                        if Sa and Sb:
                            for ga in Sa:
                                for gb in Sb:
                                    dsu.union(ga, gb)
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
                if la == 0 or lb == 0:
                    continue
                dsu.union(_gid(r, int(la)), _gid(rn, int(lb)))

    pair(axis=0, key_pos="ovlp_xpos", key_neg="ovlp_xneg")
    pair(axis=1, key_pos="ovlp_ypos", key_neg="ovlp_yneg")
    pair(axis=2, key_pos="ovlp_zpos", key_neg="ovlp_zneg")
    return dsu

def build_boundary_graph(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
                         dx: float, dy: float, dz: float):
    """Boundary-face DSU yielding label-level unions.

    Nodes: implicit (rank,label,face). Unions:
      - in-rank: connect all faces a label touches
      - cross-rank: connect opposing face pixels
    Returns a DSU over (rank,label) keys plus edge_counts and face_area for area correction.
    """
    bdsu = DSU()

    def _gfid(rank: int, lab: int, face_key: str) -> np.uint64:
        # pack: [ rank:24 | label:32 | face:8 ]
        face_idx = {"x-": 0, "x+": 1, "y-": 2, "y+": 3, "z-": 4, "z+": 5}[face_key]
        return (np.uint64(rank) << np.uint64(40)) | (np.uint64(lab) << np.uint64(8)) | np.uint64(face_idx)

    def u_b(r1: int, l1: int, f1: str, r2: int, l2: int, f2: str):
        if l1 <= 0 or l2 <= 0:
            return
        a = _gfid(r1, int(l1), f1)
        b = _gfid(r2, int(l2), f2)
        bdsu.union(a, b)

    faces = {}
    label_faces = {}
    for r, info in ranks.items():
        d = _load_npz(info["npz"])
        f = {
            "x-": d["face_xneg"], "x+": d["face_xpos"],
            "y-": d["face_yneg"], "y+": d["face_ypos"],
            "z-": d["face_zneg"], "z+": d["face_zpos"],
        }
        faces[r] = f
        for k, arr in f.items():
            u = np.unique(arr)
            u = u[u > 0]
            for lab in u:
                label_faces.setdefault((r, int(lab)), set()).add(k)

    # in-rank unions between faces touched by same label
    for (r, lab), fset in label_faces.items():
        fs = list(fset)
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                u_b(r, lab, fs[i], r, lab, fs[j])

    edge_counts = {"x": {}, "y": {}, "z": {}}
    face_area = {"x": dy * dz, "y": dx * dz, "z": dx * dy}
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}

    # cross-rank unions and edge_counts
    for r, info in ranks.items():
        coords = tuple(info["coords"])
        # x
        ncoords = _neighbor(coords, cart_dims, axis=0, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            a = faces[r]["x+"]; b = faces[rn]["x-"]
            m = (a > 0) & (b > 0)
            if m.any():
                A = a[m]; B = b[m]
                for la, lb in zip(A, B):
                    u_b(r, int(la), "x+", rn, int(lb), "x-")
                    ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
                    key = (ga, gb) if ga < gb else (gb, ga)
                    edge_counts["x"][key] = edge_counts["x"].get(key, 0) + 1
        # y
        ncoords = _neighbor(coords, cart_dims, axis=1, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            a = faces[r]["y+"]; b = faces[rn]["y-"]
            m = (a > 0) & (b > 0)
            if m.any():
                A = a[m]; B = b[m]
                for la, lb in zip(A, B):
                    u_b(r, int(la), "y+", rn, int(lb), "y-")
                    ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
                    key = (ga, gb) if ga < gb else (gb, ga)
                    edge_counts["y"][key] = edge_counts["y"].get(key, 0) + 1
        # z
        ncoords = _neighbor(coords, cart_dims, axis=2, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            a = faces[r]["z+"]; b = faces[rn]["z-"]
            m = (a > 0) & (b > 0)
            if m.any():
                A = a[m]; B = b[m]
                for la, lb in zip(A, B):
                    u_b(r, int(la), "z+", rn, int(lb), "z-")
                    ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
                    key = (ga, gb) if ga < gb else (gb, ga)
                    edge_counts["z"][key] = edge_counts["z"].get(key, 0) + 1

    # Collapse boundary DSU into label-level DSU
    ldsu = DSU()
    buckets: Dict[np.uint64, set] = {}
    for (r, lab), fset in label_faces.items():
        for fk in fset:
            root = bdsu.find(_gfid(r, lab, fk))
            buckets.setdefault(root, set()).add((r, lab))
    for items in buckets.values():
        items = list(items)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ra, la) = items[i]; (rb, lb) = items[j]
                ldsu.union(_gid(ra, la), _gid(rb, lb))

    return ldsu, edge_counts, face_area


def build_shell_exact(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
                      dx: float, dy: float, dz: float):
    """Exact unions using 3-layer shells (t=3):
    - Unify labels within each shell stack along all 3 axes
    - Unify overlapping cells between shells along edges/corners
    - Unify opposing outermost layers across ranks (periodic)
    - Compute edge_counts for area correction from face maps
    Returns label-level DSU and edge_counts/face_area
    """
    dsu = DSU()
    edge_counts = {"x": {}, "y": {}, "z": {}}
    face_area = {"x": dy * dz, "y": dx * dz, "z": dx * dy}
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}

    shells = {}
    dims = {}
    faces = {}
    for r, info in ranks.items():
        d = _load_npz(info["npz"])
        fxm = d["face_xneg"]; fxp = d["face_xpos"]
        fym = d["face_yneg"]; fyp = d["face_ypos"]
        fzm = d["face_zneg"]; fzp = d["face_zpos"]
        faces[r] = {"x-": fxm, "x+": fxp, "y-": fym, "y+": fyp, "z-": fzm, "z+": fzp}
        # dims
        nj, nk = fxm.shape
        ni = fym.shape[0]
        dims[r] = (ni, nj, nk)
        shells[r] = {
            "x-": d["shell_xneg"], "x+": d["shell_xpos"],
            "y-": d["shell_yneg"], "y+": d["shell_ypos"],
            "z-": d["shell_zneg"], "z+": d["shell_zpos"],
        }

    def U(r, la, rb, lb):
        if la > 0 and lb > 0:
            dsu.union(_gid(r, int(la)), _gid(rb, int(lb)))

    # In-rank unions inside each shell stack
    def unify_stack_inrank(r, arr):
        a = arr
        td, s1, s2 = a.shape
        # along thickness
        if td > 1:
            m = (a[:-1, :, :] > 0) & (a[1:, :, :] > 0)
            La = a[:-1, :, :][m]; Lb = a[1:, :, :][m]
            for la, lb in zip(La, Lb): U(r, la, r, lb)
        # along first in-plane axis
        if s1 > 1:
            m = (a[:, :-1, :] > 0) & (a[:, 1:, :] > 0)
            La = a[:, :-1, :][m]; Lb = a[:, 1:, :][m]
            for la, lb in zip(La, Lb): U(r, la, r, lb)
        # along second in-plane axis
        if s2 > 1:
            m = (a[:, :, :-1] > 0) & (a[:, :, 1:] > 0)
            La = a[:, :, :-1][m]; Lb = a[:, :, 1:][m]
            for la, lb in zip(La, Lb): U(r, la, r, lb)

    # Overlap unions for edges between shell stacks
    for r, sh in shells.items():
        ni, nj, nk = dims[r]
        xmn, xmp = sh["x-"], sh["x+"]  # (t,nj,nk)
        ymn, ymp = sh["y-"], sh["y+"]  # (ni,t,nk)
        zmn, zmp = sh["z-"], sh["z+"]  # (ni,nj,t)

        # unify within each stack
        for a in (xmn, xmp, ymn, ymp, zmn, zmp):
            unify_stack_inrank(r, a)

        t = xmn.shape[0]
        # x- with y-
        for di in range(t):
            for dj in range(t):
                La = xmn[di, dj, :]; Lb = ymn[di, dj, :]
                m = (La > 0) & (Lb > 0)
                for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
        # x- with y+
        for di in range(t):
            for dj in range(t):
                j = nj - t + dj
                La = xmn[di, j, :]; Lb = ymp[di, dj, :]
                m = (La > 0) & (Lb > 0)
                for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
        # x+ with y-
        for di in range(t):
            i = ni - t + di
            for dj in range(t):
                La = xmp[di, dj, :]; Lb = ymn[i, dj, :]
                m = (La > 0) & (Lb > 0)
                for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
        # x+ with y+
        for di in range(t):
            i = ni - t + di
            for dj in range(t):
                La = xmp[di, nj - t + dj, :]; Lb = ymp[i, dj, :]
                m = (La > 0) & (Lb > 0)
                for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)

        # x- with z- ; x- with z+
        for di in range(t):
            La = xmn[di, :, 0:t]; Lb = zmn[di, :, 0:t]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
            La = xmn[di, :, nk - t:nk]; Lb = zmp[di, :, 0:t]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
        # x+ with z- ; x+ with z+
        for di in range(t):
            i = ni - t + di
            La = xmp[di, :, 0:t]; Lb = zmn[i, :, 0:t]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
            La = xmp[di, :, nk - t:nk]; Lb = zmp[i, :, 0:t]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)

        # y- with z- ; y- with z+
        for dj in range(t):
            La = ymn[:, dj, 0:t]; Lb = zmn[:, 0:t, dj]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
            La = ymn[:, dj, nk - t:nk]; Lb = zmp[:, 0:t, dj]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
        # y+ with z- ; y+ with z+
        for dj in range(t):
            j = nj - t + dj
            La = ymp[:, dj, 0:t]; Lb = zmn[:, j, 0:t]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)
            La = ymp[:, dj, nk - t:nk]; Lb = zmp[:, j, 0:t]
            m = (La > 0) & (Lb > 0)
            for la, lb in zip(La[m], Lb[m]): U(r, la, r, lb)

    # Cross-rank unions on opposing outermost layers (and edge_counts)
    for r, info in ranks.items():
        coords = tuple(info["coords"])
        ni, nj, nk = dims[r]
        t = shells[r]["x-"].shape[0]
        # X axis
        ncoords = _neighbor(coords, cart_dims, axis=0, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            a = shells[r]["x+"][t-1, :, :]; b = shells[rn]["x-"][0, :, :]
            m = (a > 0) & (b > 0)
            if m.any():
                A = a[m]; B = b[m]
                for la, lb in zip(A, B): U(r, la, rn, lb)
            # edge counts from face maps
            am = faces[r]["x+"]; bm = faces[rn]["x-"]
            mm = (am > 0) & (bm > 0)
            if mm.any():
                AA = am[mm]; BB = bm[mm]
                for la, lb in zip(AA, BB):
                    ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
                    key = (ga, gb) if ga < gb else (gb, ga)
                    edge_counts["x"][key] = edge_counts["x"].get(key, 0) + 1
        # Y axis
        ncoords = _neighbor(coords, cart_dims, axis=1, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            a = shells[r]["y+"][:, t-1, :]; b = shells[rn]["y-"][:, 0, :]
            m = (a > 0) & (b > 0)
            if m.any():
                A = a[m]; B = b[m]
                for la, lb in zip(A, B): U(r, la, rn, lb)
            am = faces[r]["y+"]; bm = faces[rn]["y-"]
            mm = (am > 0) & (bm > 0)
            if mm.any():
                AA = am[mm]; BB = bm[mm]
                for la, lb in zip(AA, BB):
                    ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
                    key = (ga, gb) if ga < gb else (gb, ga)
                    edge_counts["y"][key] = edge_counts["y"].get(key, 0) + 1
        # Z axis
        ncoords = _neighbor(coords, cart_dims, axis=2, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            a = shells[r]["z+"][:, :, t-1]; b = shells[rn]["z-"][:, :, 0]
            m = (a > 0) & (b > 0)
            if m.any():
                A = a[m]; B = b[m]
                for la, lb in zip(A, B): U(r, la, rn, lb)
            am = faces[r]["z+"]; bm = faces[rn]["z-"]
            mm = (am > 0) & (bm > 0)
            if mm.any():
                AA = am[mm]; BB = bm[mm]
                for la, lb in zip(AA, BB):
                    ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
                    key = (ga, gb) if ga < gb else (gb, ga)
                    edge_counts["z"][key] = edge_counts["z"].get(key, 0) + 1

    return dsu, edge_counts, face_area

def stitch_reduce(input_dir: str, output_path: str, mode: str = "face"):
    ranks, cart_dims, periodic = index_parts(input_dir)
    any_npz = _load_npz(next(iter(ranks.values()))["npz"])
    dx, dy, dz = (float(any_npz["voxel_spacing"][0]),
                  float(any_npz["voxel_spacing"][1]),
                  float(any_npz["voxel_spacing"][2]))

    if mode == "boundary":
        dsu, edge_counts, face_area = build_boundary_graph(ranks, cart_dims, periodic, dx, dy, dz)
    elif mode == "exact":
        # Build exact unions from 3-layer shells if present; fallback to boundary graph if missing
        have_shells = True
        try:
            # probe one rank
            any_npz = _load_npz(next(iter(ranks.values()))["npz"])
            _ = any_npz["shell_t"]
        except Exception:
            have_shells = False
        if have_shells:
            dsu, edge_counts, face_area = build_shell_exact(ranks, cart_dims, periodic, dx, dy, dz)
        else:
            dsu, edge_counts, face_area = build_boundary_graph(ranks, cart_dims, periodic, dx, dy, dz)
    elif mode == "overlap-exact":
        dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz, mode="face")
        dsu = _merge_by_overlap_planes(ranks, cart_dims, periodic, dsu=dsu)
    else:
        dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz, mode=mode)

    all_gids: List[np.uint64] = []
    parts = {}
    for r, info in ranks.items():
        d = _load_npz(info["npz"])
        lids = d["label_ids"].astype(np.int64)
        gids = (_gid(r, 0) + lids.astype(np.uint64))
        all_gids.extend(list(gids))
        parts[r] = d

    roots = {}
    for g in all_gids:
        roots[g] = dsu.find(g)

    uniq_roots = sorted(set(roots.values()))
    root_to_idx = {rt: i for i, rt in enumerate(uniq_roots)}
    G = len(uniq_roots)

    cell_count = np.zeros(G, dtype=np.int64)
    volume = np.zeros(G, dtype=np.float64)
    mass = np.zeros(G, dtype=np.float64)
    Sxv = np.zeros(G, dtype=np.float64); Syv = np.zeros(G, dtype=np.float64); Szv = np.zeros(G, dtype=np.float64)
    Sxm = np.zeros(G, dtype=np.float64); Sym = np.zeros(G, dtype=np.float64); Szm = np.zeros(G, dtype=np.float64)
    bbox = np.zeros((G, 6), dtype=np.int64)
    bbox[:, 0::2] = np.iinfo(np.int64).max
    bbox[:, 1::2] = np.iinfo(np.int64).min
    area = np.zeros(G, dtype=np.float64)

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
    centroid_vol = np.stack([Sxv / (volume + small), Syv / (volume + small), Szv / (volume + small)], axis=1)
    centroid_mass = np.stack([Sxm / (mass + small),   Sym / (mass + small),   Szm / (mass + small)], axis=1)

    out = {
        "gid": np.array(uniq_roots, dtype=np.uint64),
        "cell_count": cell_count,
        "volume": volume,
        "mass": mass,
        "area": area,
        "centroid_vol": centroid_vol,
        "centroid_mass": centroid_mass,
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
    ap.add_argument("--stitch-mode",
                    choices=["face", "shell", "boundary", "exact", "overlap-exact"],
                    default="overlap-exact",
                    help="stitching strategy; 'overlap-exact' uses shared overlap planes for exact unions")
    args = ap.parse_args()
    out = stitch_reduce(args.input, args.output, mode=args.stitch_mode)
    print(f"Stitched {out['gid'].size} global clumps -> {args.output}")


if __name__ == "__main__":
    main()
