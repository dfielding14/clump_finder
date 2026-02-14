from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import metrics as M
from local_label import label_3d
from stitch import stitch_reduce


def _split_axis(n: int, p: int) -> list[tuple[int, int]]:
    base, rem = divmod(n, p)
    out: list[tuple[int, int]] = []
    start = 0
    for coord in range(p):
        length = base + (1 if coord < rem else 0)
        end = start + length
        out.append((start, end))
        start = end
    return out


def _extract_with_halo(arr: np.ndarray, i0: int, i1: int, j0: int, j1: int, k0: int, k1: int,
                       halo: int = 1) -> np.ndarray:
    n = arr.shape[0]
    ii = (np.arange(i0 - halo, i1 + halo) % n)
    jj = (np.arange(j0 - halo, j1 + halo) % n)
    kk = (np.arange(k0 - halo, k1 + halo) % n)
    return arr[np.ix_(ii, jj, kk)]


def center_from_fraction(n: int, frac: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(frac[0]) * float(n), float(frac[1]) * float(n), float(frac[2]) * float(n))


def _periodic_axis_delta(n: int, center: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) + 0.5
    return ((x - float(center) + 0.5 * n) % n) - 0.5 * n


def make_periodic_cloud_mask(n: int,
                             center: tuple[float, float, float],
                             kind: str = "sphere",
                             radius: float | None = None,
                             axes: tuple[float, float, float] | None = None,
                             rotation_deg_z: float = 0.0) -> np.ndarray:
    """Return a periodic single-cloud mask in an N^3 domain.

    The cloud can be a sphere or ellipsoid. For ellipsoids, optional rotation is
    applied around the z axis.
    """
    cx, cy, cz = center
    dx = _periodic_axis_delta(n, cx)
    dy = _periodic_axis_delta(n, cy)
    dz = _periodic_axis_delta(n, cz)

    kind = str(kind).lower()
    if kind == "sphere":
        if radius is None or float(radius) <= 0.0:
            raise ValueError("sphere requires radius > 0")
        rr = float(radius) * float(radius)
        return (dx[:, None, None] ** 2 + dy[None, :, None] ** 2 + dz[None, None, :] ** 2) <= rr

    if kind != "ellipsoid":
        raise ValueError("kind must be 'sphere' or 'ellipsoid'")
    if axes is None:
        raise ValueError("ellipsoid requires axes=(a,b,c)")
    a, b, c = (float(axes[0]), float(axes[1]), float(axes[2]))
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise ValueError("ellipsoid axes must be > 0")

    theta = np.deg2rad(float(rotation_deg_z))
    if abs(theta) < 1e-14:
        term_xy = (dx[:, None] / a) ** 2 + (dy[None, :] / b) ** 2
    else:
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        dx2 = dx[:, None]
        dy2 = dy[None, :]
        u = cos_t * dx2 + sin_t * dy2
        v = -sin_t * dx2 + cos_t * dy2
        term_xy = (u / a) ** 2 + (v / b) ** 2
    term_z = (dz[None, None, :] / c) ** 2
    return (term_xy[:, :, None] + term_z) <= 1.0


def _periodic_component_count(mask: np.ndarray) -> int:
    """Count connected components with periodic face-wrap unions."""
    labels = label_3d(mask, tile_shape=(128, 128, 128), connectivity=6, halo=0)
    k0 = int(labels.max())
    if k0 == 0:
        return 0

    parent = np.arange(k0 + 1, dtype=np.int64)

    def uf_find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def uf_union(a: int, b: int):
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            parent[rb] = ra

    def merge_periodic_faces(a: np.ndarray, b: np.ndarray):
        m = (a > 0) & (b > 0)
        if not m.any():
            return
        aa = a[m]
        bb = b[m]
        for la, lb in zip(aa, bb):
            if la != lb:
                uf_union(int(la), int(lb))

    merge_periodic_faces(labels[0, :, :], labels[-1, :, :])
    merge_periodic_faces(labels[:, 0, :], labels[:, -1, :])
    merge_periodic_faces(labels[:, :, 0], labels[:, :, -1])

    roots = set()
    for lab in np.unique(labels):
        if lab > 0:
            roots.add(uf_find(int(lab)))
    return len(roots)


def _write_parts_for_mask(parts_dir: str,
                          mask: np.ndarray,
                          px: int,
                          py: int,
                          pz: int,
                          use_halo: bool = True,
                          overlap: int = 1):
    n = int(mask.shape[0])
    if mask.shape != (n, n, n):
        raise ValueError(f"mask must be cubic, got shape={mask.shape}")
    if overlap != 1:
        raise NotImplementedError("overlap width > 1 is not supported in this synthetic writer")
    if overlap < 1:
        raise ValueError("overlap must be >= 1")

    out_dir = Path(parts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("clumps_rank*.npz"):
        old.unlink()
    for old in out_dir.glob("clumps_rank*.meta.json"):
        old.unlink()

    dx = dy = dz = 1.0
    ix = _split_axis(n, px)
    iy = _split_axis(n, py)
    iz = _split_axis(n, pz)

    rank = 0
    for cx, (i0, i1) in enumerate(ix):
        for cy, (j0, j1) in enumerate(iy):
            for cz, (k0, k1) in enumerate(iz):
                if use_halo:
                    halo = 1
                    mask_h = _extract_with_halo(mask, i0, i1, j0, j1, k0, k1, halo=halo)
                else:
                    halo = 0
                    mask_h = mask[i0:i1, j0:j1, k0:k1]

                labels_ext = label_3d(mask_h, tile_shape=(128, 128, 128), connectivity=6, halo=0)
                labels = labels_ext[halo:-halo, halo:-halo, halo:-halo] if halo > 0 else labels_ext

                core_vals = np.unique(labels)
                core_vals = core_vals[core_vals != 0]
                lut = np.zeros(int(labels_ext.max()) + 1, dtype=np.uint32)
                if core_vals.size:
                    lut[core_vals] = np.arange(1, core_vals.size + 1, dtype=np.uint32)
                labels = lut[labels]
                labels_ext = lut[labels_ext]
                k_local = int(core_vals.size)

                dens_i = np.ones(labels.shape, dtype=np.float32)
                cell = M.num_cells(labels, K=k_local)
                vol = M.volumes(cell, dx, dy, dz)
                mass = M.masses(labels, dens_i, dx, dy, dz, K=k_local)
                cvol, cmass = M.centroids(
                    labels,
                    dens_i,
                    dx,
                    dy,
                    dz,
                    (0.0, 0.0, 0.0),
                    ((i0, i1), (j0, j1), (k0, k1)),
                    K=k_local,
                )
                area = M.exposed_area(labels, dx, dy, dz, K=k_local)
                bbox = M.compute_bboxes(labels, ((i0, i1), (j0, j1), (k0, k1)), K=k_local)

                rank_ids = np.arange(1, k_local + 1, dtype=np.int32)
                face_xneg = labels[0, :, :].astype(np.uint32, copy=False)
                face_xpos = labels[-1, :, :].astype(np.uint32, copy=False)
                face_yneg = labels[:, 0, :].astype(np.uint32, copy=False)
                face_ypos = labels[:, -1, :].astype(np.uint32, copy=False)
                face_zneg = labels[:, :, 0].astype(np.uint32, copy=False)
                face_zpos = labels[:, :, -1].astype(np.uint32, copy=False)

                if overlap > halo:
                    raise ValueError("overlap cannot exceed halo width")
                ni_c, nj_c, nk_c = labels.shape
                i_start = halo
                i_end = halo + ni_c
                j_start = halo
                j_end = halo + nj_c
                k_start = halo
                k_end = halo + nk_c
                ov_xneg = labels_ext[i_start:i_start + overlap, j_start:j_end, k_start:k_end].astype(np.uint32, copy=False)[0]
                ov_xpos = labels_ext[i_end - overlap:i_end, j_start:j_end, k_start:k_end].astype(np.uint32, copy=False)[-1]
                ov_yneg = labels_ext[i_start:i_end, j_start:j_start + overlap, k_start:k_end].astype(np.uint32, copy=False)[:, 0, :]
                ov_ypos = labels_ext[i_start:i_end, j_end - overlap:j_end, k_start:k_end].astype(np.uint32, copy=False)[:, -1, :]
                ov_zneg = labels_ext[i_start:i_end, j_start:j_end, k_start:k_start + overlap].astype(np.uint32, copy=False)[:, :, 0]
                ov_zpos = labels_ext[i_start:i_end, j_start:j_end, k_end - overlap:k_end].astype(np.uint32, copy=False)[:, :, -1]

                out = {
                    "label_ids": rank_ids,
                    "cell_count": cell,
                    "volume": vol,
                    "mass": mass,
                    "area": area,
                    "centroid_vol": cvol,
                    "centroid_mass": cmass,
                    "bbox_ijk": bbox,
                    "voxel_spacing": np.array([dx, dy, dz], dtype=np.float64),
                    "origin": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    "overlap_width": np.int32(overlap),
                    "ovlp_xneg": ov_xneg,
                    "ovlp_xpos": ov_xpos,
                    "ovlp_yneg": ov_yneg,
                    "ovlp_ypos": ov_ypos,
                    "ovlp_zneg": ov_zneg,
                    "ovlp_zpos": ov_zpos,
                    "face_xneg": face_xneg,
                    "face_xpos": face_xpos,
                    "face_yneg": face_yneg,
                    "face_ypos": face_ypos,
                    "face_zneg": face_zneg,
                    "face_zpos": face_zpos,
                }
                np.savez(out_dir / f"clumps_rank{rank:05d}.npz", **out)
                meta = {
                    "rank": rank,
                    "coords": (cx, cy, cz),
                    "cart_dims": (px, py, pz),
                    "node_bbox_ijk": [i0, i1, j0, j1, k0, k1],
                    "grid": {"periodic": [True, True, True]},
                    "stitching": {"min_clump_cells_deferred": 1},
                    "output_npz": f"clumps_rank{rank:05d}.npz",
                }
                with open(out_dir / f"clumps_rank{rank:05d}.meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                rank += 1


def assert_single_cloud_result(result: dict[str, Any]):
    stitched = result["stitched"]
    expected_cells = int(result["expected_cells"])
    got_k = int(stitched["gid"].shape[0])
    if got_k != 1:
        raise AssertionError(f"Expected one stitched cloud, got {got_k}")
    got_cells = int(stitched["cell_count"][0])
    if got_cells != expected_cells:
        raise AssertionError(f"Cell count mismatch: stitched={got_cells}, expected={expected_cells}")
    got_vol = float(stitched["volume"][0])
    if not np.isclose(got_vol, float(expected_cells)):
        raise AssertionError(f"Volume mismatch: stitched={got_vol}, expected={float(expected_cells)}")


def run_single_cloud_case(n: int,
                          center: tuple[float, float, float],
                          partition: tuple[int, int, int],
                          kind: str = "sphere",
                          radius: float | None = None,
                          axes: tuple[float, float, float] | None = None,
                          rotation_deg_z: float = 0.0,
                          use_halo: bool = True,
                          overlap: int = 1,
                          keep_outputs: bool = False,
                          out_dir: str | None = None) -> dict[str, Any]:
    """Generate a single-cloud periodic volume, run clump finding + stitching, return diagnostics."""
    mask = make_periodic_cloud_mask(
        n,
        center=center,
        kind=kind,
        radius=radius,
        axes=axes,
        rotation_deg_z=rotation_deg_z,
    )

    baseline_components = _periodic_component_count(mask)
    if baseline_components != 1:
        raise AssertionError(
            f"Synthetic cloud must be single connected component under periodic connectivity; got {baseline_components}"
        )

    if out_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="stitch_cloud_")
        should_cleanup = not keep_outputs
    else:
        tmpdir = str(out_dir)
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        should_cleanup = False

    try:
        px, py, pz = partition
        _write_parts_for_mask(tmpdir, mask, px, py, pz, use_halo=use_halo, overlap=overlap)
        stitched_path = os.path.join(tmpdir, "stitched.npz")
        stitched = stitch_reduce(tmpdir, stitched_path)
        return {
            "n": int(n),
            "center": tuple(float(c) for c in center),
            "partition": tuple(int(x) for x in partition),
            "kind": str(kind),
            "radius": None if radius is None else float(radius),
            "axes": None if axes is None else tuple(float(a) for a in axes),
            "rotation_deg_z": float(rotation_deg_z),
            "baseline_components": int(baseline_components),
            "expected_cells": int(mask.sum()),
            "stitched": stitched,
            "parts_dir": tmpdir,
            "stitched_path": stitched_path,
        }
    finally:
        if should_cleanup:
            shutil.rmtree(tmpdir, ignore_errors=True)
