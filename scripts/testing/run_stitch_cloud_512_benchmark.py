#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stitch import stitch_reduce
from stitch_cloud_test_utils import center_from_fraction
from stitch_cloud_test_utils import make_periodic_cloud_mask


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


def _parse_partitions(text: str) -> list[tuple[int, int, int]]:
    out = []
    for item in text.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid partition '{item}', expected PxPyPz like 2x2x2")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def _default_locations() -> list[tuple[float, float, float]]:
    return [
        (0.50, 0.50, 0.50),
        (0.05, 0.50, 0.50),
        (0.95, 0.50, 0.50),
        (0.50, 0.05, 0.50),
        (0.50, 0.95, 0.50),
        (0.50, 0.50, 0.05),
        (0.50, 0.50, 0.95),
        (0.96, 0.95, 0.04),
        (0.04, 0.96, 0.95),
        (0.95, 0.04, 0.96),
    ]


def _default_aspects() -> list[tuple[float, float]]:
    return [
        (1.00, 1.00),
        (0.95, 0.95),
        (0.90, 0.90),
        (0.90, 0.70),
        (0.85, 0.65),
        (0.80, 0.60),
        (0.75, 0.55),
        (0.70, 0.50),
        (0.65, 0.45),
        (0.60, 0.40),
    ]


def _global_cov_sums(mask: np.ndarray, center: tuple[float, float, float]) -> dict[str, float]:
    n = int(mask.shape[0])
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("Cloud mask is empty")
    cell = pts.astype(np.float64) + 0.5
    coords = []
    for axis in range(3):
        d = ((cell[:, axis] - center[axis] + 0.5 * n) % n) - 0.5 * n
        coords.append(center[axis] + d)
    x, y, z = coords
    w = float(x.shape[0])
    return {
        "cov_W": w,
        "cov_Sx": float(x.sum()),
        "cov_Sy": float(y.sum()),
        "cov_Sz": float(z.sum()),
        "cov_Sxx": float((x * x).sum()),
        "cov_Syy": float((y * y).sum()),
        "cov_Szz": float((z * z).sum()),
        "cov_Sxy": float((x * y).sum()),
        "cov_Sxz": float((x * z).sum()),
        "cov_Syz": float((y * z).sum()),
    }


def _bbox_from_mask(core: np.ndarray, i0: int, j0: int, k0: int) -> np.ndarray:
    ii = np.where(np.any(core, axis=(1, 2)))[0]
    jj = np.where(np.any(core, axis=(0, 2)))[0]
    kk = np.where(np.any(core, axis=(0, 1)))[0]
    return np.array([[i0 + int(ii[0]), i0 + int(ii[-1]) + 1,
                      j0 + int(jj[0]), j0 + int(jj[-1]) + 1,
                      k0 + int(kk[0]), k0 + int(kk[-1]) + 1]], dtype=np.int32)


def _exposed_area_single(core: np.ndarray) -> float:
    if not core.any():
        return 0.0
    area = 0
    area += int(np.count_nonzero(core[0, :, :]))
    area += int(np.count_nonzero(core[-1, :, :]))
    area += int(np.count_nonzero(core[:, 0, :]))
    area += int(np.count_nonzero(core[:, -1, :]))
    area += int(np.count_nonzero(core[:, :, 0]))
    area += int(np.count_nonzero(core[:, :, -1]))
    area += int(np.count_nonzero(core[:-1, :, :] != core[1:, :, :]))
    area += int(np.count_nonzero(core[:, :-1, :] != core[:, 1:, :]))
    area += int(np.count_nonzero(core[:, :, :-1] != core[:, :, 1:]))
    return float(area)


def _write_parts_fast(parts_dir: str,
                      mask: np.ndarray,
                      center: tuple[float, float, float],
                      partition: tuple[int, int, int],
                      global_cov: dict[str, float]) -> int:
    n = int(mask.shape[0])
    px, py, pz = partition
    ix = _split_axis(n, px)
    iy = _split_axis(n, py)
    iz = _split_axis(n, pz)

    out_dir = Path(parts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_cells = int(mask.sum())
    if total_cells <= 0:
        raise ValueError("Cloud mask has zero foreground cells")

    pre_fragments = 0
    rank = 0
    for cx, (i0, i1) in enumerate(ix):
        for cy, (j0, j1) in enumerate(iy):
            for cz, (k0, k1) in enumerate(iz):
                core = mask[i0:i1, j0:j1, k0:k1]
                labels = core.astype(np.uint32, copy=False)
                ext = _extract_with_halo(mask, i0, i1, j0, j1, k0, k1, halo=1)
                labels_ext = ext.astype(np.uint32, copy=False)

                count = int(core.sum())
                k_local = 1 if count > 0 else 0
                if k_local == 1:
                    pre_fragments += 1

                if k_local == 1:
                    wfrac = float(count) / float(total_cells)
                    label_ids = np.array([1], dtype=np.int32)
                    cell_count = np.array([count], dtype=np.int64)
                    volume = np.array([float(count)], dtype=np.float64)
                    mass = np.array([float(count)], dtype=np.float64)
                    area = np.array([_exposed_area_single(core)], dtype=np.float64)
                    centroid = np.array([[center[0], center[1], center[2]]], dtype=np.float64)
                    bbox = _bbox_from_mask(core, i0, j0, k0)
                    cov = {
                        k: np.array([float(v) * wfrac], dtype=np.float64)
                        for k, v in global_cov.items()
                    }
                    vmean = np.array([0.0], dtype=np.float64)
                    vstd = np.array([0.0], dtype=np.float64)
                else:
                    label_ids = np.zeros((0,), dtype=np.int32)
                    cell_count = np.zeros((0,), dtype=np.int64)
                    volume = np.zeros((0,), dtype=np.float64)
                    mass = np.zeros((0,), dtype=np.float64)
                    area = np.zeros((0,), dtype=np.float64)
                    centroid = np.zeros((0, 3), dtype=np.float64)
                    bbox = np.zeros((0, 6), dtype=np.int32)
                    cov = {k: np.zeros((0,), dtype=np.float64) for k in global_cov}
                    vmean = np.zeros((0,), dtype=np.float64)
                    vstd = np.zeros((0,), dtype=np.float64)

                ni_c, nj_c, nk_c = labels.shape
                i_start = 1
                i_end = 1 + ni_c
                j_start = 1
                j_end = 1 + nj_c
                k_start = 1
                k_end = 1 + nk_c
                ov_xneg = labels_ext[i_start, j_start:j_end, k_start:k_end]
                ov_xpos = labels_ext[i_end - 1, j_start:j_end, k_start:k_end]
                ov_yneg = labels_ext[i_start:i_end, j_start, k_start:k_end]
                ov_ypos = labels_ext[i_start:i_end, j_end - 1, k_start:k_end]
                ov_zneg = labels_ext[i_start:i_end, j_start:j_end, k_start]
                ov_zpos = labels_ext[i_start:i_end, j_start:j_end, k_end - 1]

                out = {
                    "label_ids": label_ids,
                    "cell_count": cell_count,
                    "volume": volume,
                    "mass": mass,
                    "area": area,
                    "centroid_vol": centroid,
                    "centroid_mass": centroid.copy(),
                    "bbox_ijk": bbox,
                    "voxel_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float64),
                    "origin": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    "velocity_mean": vmean,
                    "velocity_std": vstd,
                    "ovlp_xneg": ov_xneg.astype(np.uint32, copy=False),
                    "ovlp_xpos": ov_xpos.astype(np.uint32, copy=False),
                    "ovlp_yneg": ov_yneg.astype(np.uint32, copy=False),
                    "ovlp_ypos": ov_ypos.astype(np.uint32, copy=False),
                    "ovlp_zneg": ov_zneg.astype(np.uint32, copy=False),
                    "ovlp_zpos": ov_zpos.astype(np.uint32, copy=False),
                    "face_xneg": labels[0, :, :].astype(np.uint32, copy=False),
                    "face_xpos": labels[-1, :, :].astype(np.uint32, copy=False),
                    "face_yneg": labels[:, 0, :].astype(np.uint32, copy=False),
                    "face_ypos": labels[:, -1, :].astype(np.uint32, copy=False),
                    "face_zneg": labels[:, :, 0].astype(np.uint32, copy=False),
                    "face_zpos": labels[:, :, -1].astype(np.uint32, copy=False),
                }
                out.update(cov)
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
    return pre_fragments


def _build_case_grid(n: int,
                     radius_min: float,
                     radius_max: float,
                     major_axis: float) -> list[dict]:
    locs = _default_locations()
    radii = np.linspace(float(radius_min), float(radius_max), 10)
    aspects = _default_aspects()

    cases: list[dict] = []
    for ir, r in enumerate(radii):
        for iloc, loc in enumerate(locs):
            cases.append({
                "kind": "sphere",
                "size_idx": ir,
                "loc_idx": iloc,
                "radius": float(r),
                "axes": (float(r), float(r), float(r)),
                "center_frac": loc,
                "rotation_deg_z": 0.0,
                "truth_ba": 1.0,
                "truth_ca": 1.0,
                "truth_elongation": 1.0,
            })

    for ia, (ba, ca) in enumerate(aspects):
        a = float(major_axis)
        b = a * float(ba)
        c = a * float(ca)
        for iloc, loc in enumerate(locs):
            cases.append({
                "kind": "ellipsoid",
                "aspect_idx": ia,
                "loc_idx": iloc,
                "radius": None,
                "axes": (a, b, c),
                "center_frac": loc,
                "rotation_deg_z": float((ia * 17) % 90),
                "truth_ba": float(ba),
                "truth_ca": float(ca),
                "truth_elongation": (1.0 / float(ca)) if ca > 0 else np.nan,
            })
    return cases


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return float("nan")
    d = x[m] - y[m]
    return float(np.sqrt(np.mean(d * d)))


def _make_summary_plot(rows: list[dict], out_png: Path):
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return

    kinds = np.array([r["kind"] for r in ok])
    parts = np.array([r["partition"] for r in ok])
    pre = np.array([float(r["pre_clumps"]) for r in ok], dtype=np.float64)
    post = np.array([float(r["post_clumps"]) for r in ok], dtype=np.float64)
    truth_ba = np.array([float(r["truth_ba"]) for r in ok], dtype=np.float64)
    truth_ca = np.array([float(r["truth_ca"]) for r in ok], dtype=np.float64)
    truth_el = np.array([float(r["truth_elongation"]) for r in ok], dtype=np.float64)
    meas_ba = np.array([float(r["meas_ba"]) for r in ok], dtype=np.float64)
    meas_ca = np.array([float(r["meas_ca"]) for r in ok], dtype=np.float64)
    meas_el = np.array([float(r["meas_elongation"]) for r in ok], dtype=np.float64)

    part_labels = sorted({p for p in parts})
    xpos = np.arange(len(part_labels), dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=160)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    for i, p in enumerate(part_labels):
        m = parts == p
        ms = m & (kinds == "sphere")
        me = m & (kinds == "ellipsoid")
        if np.any(ms):
            j = (np.random.default_rng(0).random(np.count_nonzero(ms)) - 0.5) * 0.16
            ax1.scatter(np.full(np.count_nonzero(ms), i - 0.14) + j, pre[ms], s=9, alpha=0.45, color="tab:blue")
            ax2.scatter(np.full(np.count_nonzero(ms), i - 0.14) + j, post[ms], s=9, alpha=0.45, color="tab:blue")
        if np.any(me):
            j = (np.random.default_rng(1).random(np.count_nonzero(me)) - 0.5) * 0.16
            ax1.scatter(np.full(np.count_nonzero(me), i + 0.14) + j, pre[me], s=9, alpha=0.45, color="tab:orange")
            ax2.scatter(np.full(np.count_nonzero(me), i + 0.14) + j, post[me], s=9, alpha=0.45, color="tab:orange")

    ax1.set_xticks(xpos)
    ax1.set_xticklabels(part_labels, rotation=30, ha="right")
    ax1.set_title("Pre-stitch Fragment Count")
    ax1.set_ylabel("local clumps before stitching")
    ax1.grid(alpha=0.25)

    ax2.set_xticks(xpos)
    ax2.set_xticklabels(part_labels, rotation=30, ha="right")
    ax2.set_title("Post-stitch Clump Count")
    ax2.set_ylabel("global clumps after stitching")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.grid(alpha=0.25)

    for kind, color in (("sphere", "tab:blue"), ("ellipsoid", "tab:orange")):
        m = kinds == kind
        ax3.scatter(truth_ba[m], meas_ba[m], s=10, alpha=0.5, color=color, label=kind)
        ax4.scatter(truth_ca[m], meas_ca[m], s=10, alpha=0.5, color=color, label=kind)
        ax5.scatter(truth_el[m], meas_el[m], s=10, alpha=0.5, color=color, label=kind)

    for ax, title, xlab, ylab, xarr, yarr in (
        (ax3, "Axis Ratio b/a", "truth b/a", "measured b/a", truth_ba, meas_ba),
        (ax4, "Axis Ratio c/a", "truth c/a", "measured c/a", truth_ca, meas_ca),
        (ax5, "Elongation a/c", "truth elongation", "measured elongation", truth_el, meas_el),
    ):
        lo = float(np.nanmin(np.concatenate([xarr, yarr])))
        hi = float(np.nanmax(np.concatenate([xarr, yarr])))
        lo = max(0.0, lo - 0.05 * (hi - lo + 1e-12))
        hi = hi + 0.05 * (hi - lo + 1e-12)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)

    ax3.legend(loc="lower right", fontsize=8, frameon=False)

    total = len(rows)
    n_ok = len(ok)
    n_fail = total - n_ok
    post_ok = int(np.count_nonzero(post == 1.0))
    txt = [
        f"Total runs: {total}",
        f"Successful runs: {n_ok}",
        f"Failed runs: {n_fail}",
        f"Post-stitch single-clump successes: {post_ok}/{n_ok}",
        f"RMSE b/a: {_rmse(meas_ba, truth_ba):.4f}",
        f"RMSE c/a: {_rmse(meas_ca, truth_ca):.4f}",
        f"RMSE elongation: {_rmse(meas_el, truth_el):.4f}",
    ]
    ax6.axis("off")
    ax6.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=10)
    ax6.set_title("Suite Summary")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run large N=512 synthetic stitch benchmark suite.")
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--partitions", default="1x1x1,2x2x2,4x2x2",
                    help="Comma-separated partition list, e.g. 1x1x1,2x2x2,4x2x2")
    ap.add_argument("--radius-min", type=float, default=12.0)
    ap.add_argument("--radius-max", type=float, default=110.0)
    ap.add_argument("--major-axis", type=float, default=96.0,
                    help="Major ellipsoid semi-axis length (cells)")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="Optional cap on total partitioned runs (0 = all)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "clump_out" / "stitch_cloud_512_suite"))
    ap.add_argument("--keep-case-dirs", action="store_true",
                    help="Keep per-run part directories under outdir/cases/")
    args = ap.parse_args()

    n = int(args.N)
    partitions = _parse_partitions(args.partitions)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    case_root = outdir / "cases"
    if args.keep_case_dirs:
        case_root.mkdir(parents=True, exist_ok=True)

    cases = _build_case_grid(
        n=n,
        radius_min=float(args.radius_min),
        radius_max=float(args.radius_max),
        major_axis=float(args.major_axis),
    )
    total_runs_planned = len(cases) * len(partitions)
    if args.max_runs > 0:
        total_runs_planned = min(total_runs_planned, int(args.max_runs))

    rows: list[dict] = []
    t0 = time.time()
    run_idx = 0

    for case_idx, case in enumerate(cases):
        center = center_from_fraction(n, case["center_frac"])
        kind = case["kind"]
        if kind == "sphere":
            mask = make_periodic_cloud_mask(n, center=center, kind="sphere", radius=case["radius"])
        else:
            mask = make_periodic_cloud_mask(
                n,
                center=center,
                kind="ellipsoid",
                axes=case["axes"],
                rotation_deg_z=case["rotation_deg_z"],
            )
        global_cov = _global_cov_sums(mask, center)

        for partition in partitions:
            if args.max_runs > 0 and run_idx >= args.max_runs:
                break
            run_idx += 1
            pstr = f"{partition[0]}x{partition[1]}x{partition[2]}"
            tag = f"case{case_idx:03d}_{kind}_loc{case['loc_idx']:02d}_part{pstr}"
            if kind == "sphere":
                tag = f"{tag}_r{case['size_idx']:02d}"
            else:
                tag = f"{tag}_ar{case['aspect_idx']:02d}"

            t_case = time.time()
            status = "ok"
            err = ""
            pre_clumps = np.nan
            post_clumps = np.nan
            meas_ba = np.nan
            meas_ca = np.nan
            meas_el = np.nan
            n_frag = np.nan

            if args.keep_case_dirs:
                case_dir = case_root / tag
                case_dir.mkdir(parents=True, exist_ok=True)
                parts_dir = str(case_dir)
                cleanup = False
            else:
                temp_ctx = tempfile.TemporaryDirectory(prefix="stitch512_")
                parts_dir = temp_ctx.name
                cleanup = True

            try:
                pre_clumps = _write_parts_fast(parts_dir, mask, center, partition, global_cov)
                stitched = stitch_reduce(parts_dir, os.path.join(parts_dir, "stitched.npz"))
                post_clumps = int(stitched["gid"].shape[0])
                if post_clumps > 0:
                    n_frag = int(stitched["n_fragments"][0])
                if post_clumps > 0 and "principal_axes_lengths" in stitched:
                    a, b, c = stitched["principal_axes_lengths"][0]
                    eps = 1e-30
                    meas_ba = float(b / (a + eps))
                    meas_ca = float(c / (a + eps))
                    meas_el = float(a / (c + eps))
            except Exception as exc:
                status = "fail"
                err = str(exc)
            finally:
                if cleanup:
                    temp_ctx.cleanup()

            rows.append({
                "tag": tag,
                "status": status,
                "error": err,
                "kind": kind,
                "n": n,
                "partition": pstr,
                "pre_clumps": pre_clumps,
                "post_clumps": post_clumps,
                "n_fragments": n_frag,
                "truth_a": float(case["axes"][0]),
                "truth_b": float(case["axes"][1]),
                "truth_c": float(case["axes"][2]),
                "truth_ba": float(case["truth_ba"]),
                "truth_ca": float(case["truth_ca"]),
                "truth_elongation": float(case["truth_elongation"]),
                "meas_ba": meas_ba,
                "meas_ca": meas_ca,
                "meas_elongation": meas_el,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "case_runtime_s": float(time.time() - t_case),
            })

            elapsed = time.time() - t0
            done = len(rows)
            rate = done / max(elapsed, 1e-9)
            remain = max(total_runs_planned - done, 0)
            eta = remain / max(rate, 1e-9)
            print(
                f"[{done:04d}/{total_runs_planned:04d}] {tag} status={status} "
                f"pre={pre_clumps} post={post_clumps} "
                f"runtime={time.time()-t_case:.2f}s ETA={eta/60.0:.1f}m",
                flush=True,
            )

        if args.max_runs > 0 and run_idx >= args.max_runs:
            break

    csv_path = outdir / "suite_results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    json_path = outdir / "suite_summary.json"
    ok_rows = [r for r in rows if r["status"] == "ok"]
    fail_rows = [r for r in rows if r["status"] != "ok"]
    summary = {
        "total_runs": len(rows),
        "ok_runs": len(ok_rows),
        "failed_runs": len(fail_rows),
        "planned_runs": total_runs_planned,
        "elapsed_s": float(time.time() - t0),
        "partitions": [f"{p[0]}x{p[1]}x{p[2]}" for p in partitions],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_path = outdir / "suite_summary.png"
    _make_summary_plot(rows, plot_path)

    print("\nDone.")
    print(f"Results CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary plot: {plot_path}")
    print(f"OK={len(ok_rows)} FAIL={len(fail_rows)}")
    return 0 if len(fail_rows) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
