from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stitch import stitch_reduce


def _moment_fields(rank_shift: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "vx_mean": np.array([1.0 + rank_shift], dtype=np.float64),
        "vx_std": np.array([0.1], dtype=np.float64),
        "vy_mean": np.array([2.0], dtype=np.float64),
        "vy_std": np.array([0.2], dtype=np.float64),
        "vz_mean": np.array([3.0], dtype=np.float64),
        "vz_std": np.array([0.3], dtype=np.float64),
        "rho_mean": np.array([4.0], dtype=np.float64),
        "rho_std": np.array([0.4], dtype=np.float64),
        "T_mean": np.array([5.0], dtype=np.float64),
        "T_std": np.array([0.5], dtype=np.float64),
        "pressure_mean": np.array([6.0], dtype=np.float64),
        "pressure_std": np.array([0.6], dtype=np.float64),
        "rho_mean_massw": np.array([4.1], dtype=np.float64),
        "rho_std_massw": np.array([0.41], dtype=np.float64),
        "T_mean_massw": np.array([5.1], dtype=np.float64),
        "T_std_massw": np.array([0.51], dtype=np.float64),
        "vx_mean_massw": np.array([1.1 + rank_shift], dtype=np.float64),
        "vx_std_massw": np.array([0.11], dtype=np.float64),
        "vy_mean_massw": np.array([2.1], dtype=np.float64),
        "vy_std_massw": np.array([0.21], dtype=np.float64),
        "vz_mean_massw": np.array([3.1], dtype=np.float64),
        "vz_std_massw": np.array([0.31], dtype=np.float64),
        "pressure_mean_massw": np.array([6.1], dtype=np.float64),
        "pressure_std_massw": np.array([0.61], dtype=np.float64),
    }


def _cov_fields(with_invalid: bool = False) -> dict[str, np.ndarray]:
    sx = np.array([np.inf], dtype=np.float64) if with_invalid else np.array([0.5], dtype=np.float64)
    return {
        "cov_W": np.array([1.0], dtype=np.float64),
        "cov_Sx": sx,
        "cov_Sy": np.array([0.5], dtype=np.float64),
        "cov_Sz": np.array([0.5], dtype=np.float64),
        "cov_Sxx": np.array([1.0], dtype=np.float64),
        "cov_Syy": np.array([1.0], dtype=np.float64),
        "cov_Szz": np.array([1.0], dtype=np.float64),
        "cov_Sxy": np.array([0.0], dtype=np.float64),
        "cov_Sxz": np.array([0.0], dtype=np.float64),
        "cov_Syz": np.array([0.0], dtype=np.float64),
        "euler_characteristic": np.array([1], dtype=np.int64),
    }


def _write_rank(tmpdir: Path,
                rank: int,
                coords: tuple[int, int, int],
                cart_dims: tuple[int, int, int],
                node_bbox: tuple[int, int, int, int, int, int],
                periodic: tuple[bool, bool, bool] = (True, True, True),
                overrides: dict[str, np.ndarray] | None = None) -> None:
    i0, i1, j0, j1, k0, k1 = node_bbox
    cx = i0 + 0.5 * (i1 - i0)
    cy = j0 + 0.5 * (j1 - j0)
    cz = k0 + 0.5 * (k1 - k0)
    out = {
        "label_ids": np.array([1], dtype=np.int32),
        "cell_count": np.array([1], dtype=np.int64),
        "volume": np.array([1.0], dtype=np.float64),
        "mass": np.array([1.0], dtype=np.float64),
        "area": np.array([6.0], dtype=np.float64),
        "centroid_vol": np.array([[cx, cy, cz]], dtype=np.float64),
        "centroid_mass": np.array([[cx, cy, cz]], dtype=np.float64),
        "bbox_ijk": np.array([[i0, i1, j0, j1, k0, k1]], dtype=np.int32),
        "voxel_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "origin": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "velocity_mean": np.array([1.0], dtype=np.float64),
        "velocity_std": np.array([0.1], dtype=np.float64),
        "face_xneg": np.array([[1]], dtype=np.uint32),
        "face_xpos": np.array([[1]], dtype=np.uint32),
        "face_yneg": np.array([[1]], dtype=np.uint32),
        "face_ypos": np.array([[1]], dtype=np.uint32),
        "face_zneg": np.array([[1]], dtype=np.uint32),
        "face_zpos": np.array([[1]], dtype=np.uint32),
        "ovlp_xneg": np.array([[1]], dtype=np.uint32),
        "ovlp_xpos": np.array([[1]], dtype=np.uint32),
        "ovlp_yneg": np.array([[1]], dtype=np.uint32),
        "ovlp_ypos": np.array([[1]], dtype=np.uint32),
        "ovlp_zneg": np.array([[1]], dtype=np.uint32),
        "ovlp_zpos": np.array([[1]], dtype=np.uint32),
    }
    if overrides:
        out.update(overrides)

    np.savez(tmpdir / f"clumps_rank{rank:05d}.npz", **out)
    meta = {
        "rank": rank,
        "coords": coords,
        "cart_dims": cart_dims,
        "node_bbox_ijk": [i0, i1, j0, j1, k0, k1],
        "grid": {"periodic": list(periodic)},
        "stitching": {"min_clump_cells_deferred": 1},
        "output_npz": f"clumps_rank{rank:05d}.npz",
    }
    with open(tmpdir / f"clumps_rank{rank:05d}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_moments_without_covariance_do_not_emit_shape_outputs(tmp_path: Path) -> None:
    _write_rank(
        tmp_path,
        rank=0,
        coords=(0, 0, 0),
        cart_dims=(1, 1, 1),
        node_bbox=(0, 1, 0, 1, 0, 1),
        periodic=(False, False, False),
        overrides=_moment_fields(),
    )
    out = stitch_reduce(str(tmp_path), str(tmp_path / "stitched.npz"))
    assert "vx_mean" in out
    assert "principal_axes_lengths" not in out
    assert "axis_ratios" not in out
    assert "shape_metrics_valid" not in out
    assert "euler_characteristic" not in out


def test_inconsistent_moment_fields_raise_value_error(tmp_path: Path) -> None:
    _write_rank(
        tmp_path,
        rank=0,
        coords=(0, 0, 0),
        cart_dims=(2, 1, 1),
        node_bbox=(0, 1, 0, 1, 0, 1),
        periodic=(False, False, False),
        overrides=_moment_fields(rank_shift=1.0),
    )
    _write_rank(
        tmp_path,
        rank=1,
        coords=(1, 0, 0),
        cart_dims=(2, 1, 1),
        node_bbox=(1, 2, 0, 1, 0, 1),
        periodic=(False, False, False),
        overrides=None,
    )
    with pytest.raises(ValueError, match="vx_mean/vx_std"):
        stitch_reduce(str(tmp_path), str(tmp_path / "stitched.npz"))


def test_invalid_covariance_uses_identity_orientation(tmp_path: Path) -> None:
    _write_rank(
        tmp_path,
        rank=0,
        coords=(0, 0, 0),
        cart_dims=(1, 1, 1),
        node_bbox=(0, 1, 0, 1, 0, 1),
        periodic=(False, False, False),
        overrides=_cov_fields(with_invalid=True),
    )
    out = stitch_reduce(str(tmp_path), str(tmp_path / "stitched.npz"))
    assert "orientation" in out
    assert np.allclose(out["orientation"][0], np.eye(3))
    assert not bool(out["shape_metrics_valid"][0])


def test_periodic_centroid_wrap_is_handled(tmp_path: Path) -> None:
    _write_rank(
        tmp_path,
        rank=0,
        coords=(0, 0, 0),
        cart_dims=(2, 1, 1),
        node_bbox=(0, 1, 0, 1, 0, 1),
        overrides={"centroid_vol": np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
                   "centroid_mass": np.array([[0.5, 0.5, 0.5]], dtype=np.float64),
                   "bbox_ijk": np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int32)},
    )
    _write_rank(
        tmp_path,
        rank=1,
        coords=(1, 0, 0),
        cart_dims=(2, 1, 1),
        node_bbox=(9, 10, 0, 1, 0, 1),
        overrides={"centroid_vol": np.array([[9.5, 0.5, 0.5]], dtype=np.float64),
                   "centroid_mass": np.array([[9.5, 0.5, 0.5]], dtype=np.float64),
                   "bbox_ijk": np.array([[9, 10, 0, 1, 0, 1]], dtype=np.int32)},
    )
    out = stitch_reduce(str(tmp_path), str(tmp_path / "stitched.npz"))
    assert int(out["gid"].shape[0]) == 1
    x = float(out["centroid_vol"][0, 0])
    dist_to_seam = min(abs(x - 0.0), abs(x - 10.0))
    assert dist_to_seam < 1.0
    assert abs(x - 5.0) > 2.0


def test_inconsistent_cart_dims_metadata_raises(tmp_path: Path) -> None:
    _write_rank(
        tmp_path,
        rank=0,
        coords=(0, 0, 0),
        cart_dims=(2, 1, 1),
        node_bbox=(0, 1, 0, 1, 0, 1),
    )
    _write_rank(
        tmp_path,
        rank=1,
        coords=(1, 0, 0),
        cart_dims=(1, 2, 1),
        node_bbox=(1, 2, 0, 1, 0, 1),
    )
    with pytest.raises(ValueError, match="Inconsistent cart_dims"):
        stitch_reduce(str(tmp_path), str(tmp_path / "stitched.npz"))
