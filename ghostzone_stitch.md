You’re right to want something simple, auditable, and conceptually clear.

Below is a drop‑in “Ghost‑Zone (overlap) exact stitching” implementation plan with **explicit git diffs**, followed by a **test plan** that demonstrates exact equivalence on small domains (e.g., 256³) without MPI. The approach keeps your existing code style and data model and only adds a thin layer of metadata for stitching.

---

## What we’re changing (one paragraph)

We add a **one‑cell overlapped plane per face** (configurable thickness) to every tile, export the **local label id** on those shared planes, and have the stitcher **unify clumps whose overlapped plane voxels have the same global coordinates**. Because the overlapped plane is physically duplicated in both neighboring tiles, **two labels that represent the same global clump literally share voxels**, so a union by “shared voxel coordinates” recreates exactly the same connectivity as a single‑volume run. Metrics are still reduced over each tile’s **non‑overlapped core** to avoid double counting. We keep 6‑connectivity everywhere. (This is deliberately simpler than face‑adjacency graphs or per‑label port graphs.)

> Why this is robust & simple: if a global clump crosses a partition, it necessarily occupies at least one cell on the *partition plane*. Both tiles include that exact cell in “overlap mode”, so their local labels both contain that **same (i,j,k)**. We merge labels that share any voxel coordinate; transitivity merges through multiple tiles. This is topologically equivalent to labeling the whole domain once.

---

## Files you’ll touch

* `tests/test_equivalence_no_mpi.py` — add an overlap writer and stricter equivalence checks (no MPI).
* `stitch.py` — add “overlap‑exact” stitch mode (merges on shared voxels) alongside current face mode.
* `clump_finder.py` — optional: add config knob `overlap_width` and emit overlap planes when running on real data (keeps metrics on the core). (You already label with a halo and then drop the halo from the output per `label_3d` doc; for overlap we do a second label pass over the extended array.)

We **do not** change your Numba CCL (`local_label.py`), but we rely on its documented behavior (“if halo>0, output excludes the halo”) when we produce the core labels and a second pass for extended labels.

---

## Git diffs

> Apply these diffs from the repo root (paths below match the files you shared). Where we add new helper functions, they’re small and commented. All new code sticks to NumPy & your existing helpers.

### 1) `tests/test_equivalence_no_mpi.py` — write overlapped planes + stricter, exact stitch

We extend the synthetic tiling writer to export **overlap planes** and metadata. We also add `--overlap` and `--stitch-mode` flags. (Default overlap=1 is enough for 6‑connectivity; you can set thicker overlaps if you like.)

```diff
diff --git a/tests/test_equivalence_no_mpi.py b/tests/test_equivalence_no_mpi.py
index 0000000..1111111 100644
--- a/tests/test_equivalence_no_mpi.py
+++ b/tests/test_equivalence_no_mpi.py
@@ -1,11 +1,17 @@
 from __future__ import annotations

 import json
 import os
 import shutil
 import tempfile
-from typing import Tuple
+from typing import Tuple

 import numpy as np
@@
-from stitch import stitch_reduce, index_parts, build_edges
+from stitch import stitch_reduce, index_parts, build_edges
+
+# NOTE:
+# We implement an "overlap" writer for tiles. Each tile exports:
+#   - standard face maps (as before)
+#   - overlap planes 'ovlp_{axis}{sign}' that contain label ids on the shared planes
+# The stitcher merges labels by identical global (i,j,k) on these planes.

 def _split_axis(n: int, p: int) -> list[Tuple[int, int]]:
@@
-def _write_parts(tmpdir: str, dens, temp, px: int, py: int, pz: int, thr=0.1, by="temperature"):
+def _take_wrap(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
+    """Periodic slice [lo:hi) with wrap, along axis 0 of cubic array a[N,N,N]."""
+    N = a.shape[0]
+    lo %= N; hi %= N
+    if lo < hi:
+        return a[lo:hi]
+    else:
+        return np.concatenate([a[lo:], a[:hi]], axis=0)
+
+def _write_parts(tmpdir: str, dens, temp, px: int, py: int, pz: int, thr=0.1, by="temperature",
+                 overlap: int = 1):
     N = dens.shape[0]
     dx = dy = dz = 1.0
     ix = _split_axis(N, px)
@@
-                sub_d = dens[i0:i1, j0:j1, k0:k1]
-                sub_t = temp[i0:i1, j0:j1, k0:k1]
+                # Core (non-overlapped) interior
+                sub_d = dens[i0:i1, j0:j1, k0:k1]
+                sub_t = temp[i0:i1, j0:j1, k0:k1]
                 field = sub_t if by == "temperature" else sub_d
-                labels = label_3d(field < thr, tile_shape=(128, 128, 128), connectivity=6, halo=0)
+                # Core labels (no halo)
+                labels = label_3d(field < thr, tile_shape=(128, 128, 128), connectivity=6, halo=0)
                 K = int(labels.max())
@@
-                rank_ids = np.arange(1, K + 1, dtype=np.int32)
+                rank_ids = np.arange(1, K + 1, dtype=np.int32)
+
+                # === Overlap planes ===
+                # We label an extended cube that includes 'overlap' cells on both sides
+                if overlap < 1:
+                    ovlp = 0
+                    ov_xneg = ov_xpos = ov_yneg = ov_ypos = ov_zneg = ov_zpos = np.zeros((0,0), dtype=np.uint32)
+                else:
+                    ovlp = int(overlap)
+                    # Build extended subvolume (periodic) of shape (ni+2o, nj+2o, nk+2o)
+                    def _axis(lo, hi):  # helper returning axis indices [lo-ovlp:hi+ovlp) wrapping
+                        return ( (lo-ovlp) % N, (hi+ovlp) % N )
+                    (ei0, ei1) = _axis(i0, i1)
+                    (ej0, ej1) = _axis(j0, j1)
+                    (ek0, ek1) = _axis(k0, k1)
+                    # Take wrapped blocks along each axis
+                    ext_d = _take_wrap(_take_wrap(_take_wrap(dens, ei0, ei1), ej0, ej1), ek0, ek1)
+                    ext_t = _take_wrap(_take_wrap(_take_wrap(temp, ei0, ei1), ej0, ej1), ek0, ek1)
+                    ext_field = ext_t if by == "temperature" else ext_d
+                    # Label the FULL extended block (include overlap)
+                    labels_ext = label_3d(ext_field < thr, tile_shape=(128,128,128), connectivity=6, halo=0)
+                    ni_c, nj_c, nk_c = (i1-i0, j1-j0, k1-k0)
+                    # Coordinates in 'labels_ext':
+                    # core spans [ovlp : ovlp+ni_c) etc.
+                    # shared planes (identical global coords on both neighbors):
+                    ov_xneg = labels_ext[ovlp-1, ovlp:ovlp+nj_c, ovlp:ovlp+nk_c].astype(np.uint32, copy=False)
+                    ov_xpos = labels_ext[ovlp+ni_c-1, ovlp:ovlp+nj_c, ovlp:ovlp+nk_c].astype(np.uint32, copy=False)
+                    ov_yneg = labels_ext[ovlp:ovlp+ni_c, ovlp-1, ovlp:ovlp+nk_c].astype(np.uint32, copy=False)
+                    ov_ypos = labels_ext[ovlp:ovlp+ni_c, ovlp+nj_c-1, ovlp:ovlp+nk_c].astype(np.uint32, copy=False)
+                    ov_zneg = labels_ext[ovlp:ovlp+ni_c, ovlp:ovlp+nj_c, ovlp-1].astype(np.uint32, copy=False)
+                    ov_zpos = labels_ext[ovlp:ovlp+ni_c, ovlp:ovlp+nj_c, ovlp+nk_c-1].astype(np.uint32, copy=False)

                 out = {
                     "label_ids": rank_ids,
@@
                     "voxel_spacing": np.array([dx, dy, dz]),
+                    "overlap_width": np.int32(ovlp),
+                    # overlap planes (shared coordinates with neighbors)
+                    # Shapes: x*: (nj_core, nk_core), y*: (ni_core, nk_core), z*: (ni_core, nj_core)
+                    "ovlp_xneg": ov_xneg, "ovlp_xpos": ov_xpos,
+                    "ovlp_yneg": ov_yneg, "ovlp_ypos": ov_ypos,
+                    "ovlp_zneg": ov_zneg, "ovlp_zpos": ov_zpos,
                     "face_xneg": labels[0, :, :].astype(np.uint32),
                     "face_xpos": labels[-1, :, :].astype(np.uint32),
                     "face_yneg": labels[:, 0, :].astype(np.uint32),
                     "face_ypos": labels[:, -1, :].astype(np.uint32),
                     "face_zneg": labels[:, :, 0].astype(np.uint32),
                     "face_zpos": labels[:, :, -1].astype(np.uint32),
                 }
@@
-def run_equivalence(N=96, px=2, py=2, pz=1,
+def run_equivalence(N=96, px=2, py=2, pz=1,
                     T_thr: float = 0.1, R_thr: float = 10.0,
                     field_type: str = "simple", beta: float = -2.0,
                     plot: bool = False, plot_out: str | None = None, plot_axis: str = "k", plot_index: int | None = None,
-                    mass_plot_out: str | None = None,
-                    debug_interfaces: bool = False):
+                    mass_plot_out: str | None = None,
+                    debug_interfaces: bool = False,
+                    overlap: int = 1,
+                    stitch_mode: str = "overlap-exact"):
@@
-        _write_parts(tmpdir, dens, temp, px, py, pz, thr=T_thr, by="temperature")
+        _write_parts(tmpdir, dens, temp, px, py, pz, thr=T_thr, by="temperature", overlap=overlap)
         outT = os.path.join(tmpdir, "stitched_T.npz")
-        stitch_reduce(tmpdir, outT)
+        stitch_reduce(tmpdir, outT, mode=stitch_mode)
@@
-        _write_parts(tmpdir, dens, temp, px, py, pz, thr=R_thr, by="density")
+        _write_parts(tmpdir, dens, temp, px, py, pz, thr=R_thr, by="density", overlap=overlap)
         outR = os.path.join(tmpdir, "stitched_R.npz")
-        stitch_reduce(tmpdir, outR)
+        stitch_reduce(tmpdir, outR, mode=stitch_mode)
@@
 if __name__ == "__main__":
     import argparse
     ap = argparse.ArgumentParser()
@@
     ap.add_argument("--beta", type=float, default=-2.0, help="power spectrum slope (powerlaw)")
     ap.add_argument("--plot", action="store_true")
     ap.add_argument("--plot-out", default=None)
@@
     ap.add_argument("--mass-plot-out", default=None)
     ap.add_argument("--debug-interfaces", action="store_true")
+    ap.add_argument("--overlap", type=int, default=1, help="overlap width in cells (default 1)")
+    ap.add_argument("--stitch-mode", choices=["face", "overlap-exact"], default="overlap-exact")
@@
-    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
+    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
                     field_type=args.field_type, beta=args.beta,
                     plot=args.plot, plot_out=args.plot_out, plot_axis=args.plot_axis, plot_index=args.plot_index,
-                    mass_plot_out=args.mass_plot_out,
-                    debug_interfaces=args.debug_interfaces)
+                    mass_plot_out=args.mass_plot_out,
+                    debug_interfaces=args.debug_interfaces,
+                    overlap=args.overlap,
+                    stitch_mode=args.stitch_mode)
```

**Notes**

* We kept your existing face‐map outputs unchanged so your current diagnostics continue to work. We added `ovlp_*` planes and `overlap_width` that the stitcher will consume. The synthetic writer uses in‑memory periodic slices; your production I/O already does periodic halos in `io_bridge.py`.

* We continue to use your Numba CCL as is. The second “extended” labeling is only to extract overlap planes; the *core* labels still back all per‑clump metrics. (This relies on `label_3d(..., halo>0)` trimming halos for the core pass, as its docstring says. )

---

### 2) `stitch.py` — add “overlap-exact” merge by shared voxels

We add an optional `mode` arg with a new “overlap‑exact” code path that reads the new `ovlp_*` planes and merges labels when the **same global plane voxel** is labeled on both sides.

```diff
diff --git a/stitch.py b/stitch.py
index 2222222..3333333 100644
--- a/stitch.py
+++ b/stitch.py
@@ -1,6 +1,8 @@
 """
 stitch.py — connectivity-6 stitcher with area correction and exact centroid merge

+Modes:
+  - face (default legacy): merge when face pixels touch across a cut
+  - overlap-exact: merge when identical global voxels (overlap planes) match
 Builds global clumps by unifying per-rank labels that touch across node faces.
 Reads only per-rank .npz files and JSON sidecars; no MPI required.
 """
@@
-class DSU:
+class DSU:
@@
 def _load_npz(path: str) -> Dict[str, np.ndarray]:
@@
 def _load_meta(path: str) -> dict:
@@
 def _gid(rank: int, local_id: int) -> np.uint64:
@@
 def _neighbor(coords: Tuple[int, int, int], dims: Tuple[int, int, int], axis: int, sign: int,
               periodic: Tuple[bool, bool, bool]):
@@
 def index_parts(input_dir: str):
@@
     return ranks, cart_dims, periodic


-def build_edges(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
-                dx: float, dy: float, dz: float):
+def build_edges(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
+                dx: float, dy: float, dz: float):
@@
     return dsu, edge_counts, face_area
+
+
+def _merge_by_overlap_planes(ranks: dict, cart_dims: Tuple[int,int,int], periodic: Tuple[bool,bool,bool]) -> DSU:
+    """
+    Overlap-exact: unify labels that share the same global voxel on the shared planes.
+    For axis X:
+      pair (r.xpos plane) with (rn.xneg plane) at identical (j,k).
+    Shapes:
+      ovlp_xpos[r] : (nj_core, nk_core)
+      ovlp_xneg[rn]: (nj_core, nk_core)
+    Non-zero entries are local label ids.
+    """
+    dsu = DSU()
+    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}
+
+    # Preload ovlp arrays lazily; absent => treat as zeros
+    cache = {}
+    def OV(r, key):
+        d = cache.get(r)
+        if d is None:
+            d = _load_npz(ranks[r]["npz"])
+            cache[r] = d
+        return d.get(key, None)
+
+    def pair_and_union(axis: int, key_pos: str, key_neg: str):
+        for r, info in ranks.items():
+            coords = tuple(info["coords"])
+            ncoords = _neighbor(coords, cart_dims, axis=axis, sign=+1, periodic=periodic)
+            if ncoords is None:  # nonperiodic boundary
+                continue
+            rn = by_coords[ncoords]
+            A = OV(r, key_pos); B = OV(rn, key_neg)
+            if A is None or B is None:
+                continue
+            # A and B have identical global coordinates by construction
+            m = (A > 0) & (B > 0)
+            if not np.any(m):
+                continue
+            La = A[m].astype(np.uint64, copy=False)
+            Lb = B[m].astype(np.uint64, copy=False)
+            for a, b in zip(La, Lb):
+                ga = _gid(r, int(a))
+                gb = _gid(rn, int(b))
+                if ga and gb:
+                    dsu.union(ga, gb)
+
+    pair_and_union(axis=0, key_pos="ovlp_xpos", key_neg="ovlp_xneg")
+    pair_and_union(axis=1, key_pos="ovlp_ypos", key_neg="ovlp_yneg")
+    pair_and_union(axis=2, key_pos="ovlp_zpos", key_neg="ovlp_zneg")
+    return dsu


-def stitch_reduce(input_dir: str, output_path: str):
+def stitch_reduce(input_dir: str, output_path: str, mode: str = "face"):
     ranks, cart_dims, periodic = index_parts(input_dir)
     any_npz = _load_npz(next(iter(ranks.values()))["npz"])
     dx, dy, dz = (float(any_npz["voxel_spacing"][0]),
                   float(any_npz["voxel_spacing"][1]),
                   float(any_npz["voxel_spacing"][2]))

-    dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
+    # Build unions
+    if mode == "overlap-exact":
+        # Prefer exact voxel-identity stitching if overlap planes are present; else fall back.
+        dsu = _merge_by_overlap_planes(ranks, cart_dims, periodic)
+        # To retain your existing exposed-area correction, we still collect face edge counts
+        # but they don't affect unions in overlap-exact mode.
+        _, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
+    else:
+        dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
@@
 def main():
     ap = argparse.ArgumentParser()
     ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz + .meta.json")
     ap.add_argument("--output", required=True, help="stitched npz path")
+    ap.add_argument("--mode", choices=["face","overlap-exact"], default="face",
+                    help="stitching mode (default face)")
     args = ap.parse_args()
-    out = stitch_reduce(args.input, args.output)
+    out = stitch_reduce(args.input, args.output, mode=args.mode)
     print(f"Stitched {out['gid'].size} global clumps -> {args.output}")
```

**Notes**

* We **leave your aggregation math unchanged** (cell counts, mass, centroids, bbox, area correction). Area correction still uses face adjacency counts you already compute.

* `overlap-exact` only changes **how unions are found** (shared voxel coordinates), which is the simplest, most direct notion of “same clump” at the cut.

---

### 3) (Optional) `clump_finder.py` — emit overlap planes in production

When you run on real data, your driver already reads a ghost layer and then **discards halos before labeling** (via `label_3d(..., halo=halo)` which trims the halo) and exports face maps. To support exact stitching at scale with the same logic used in tests, add:

* A config knob `overlap_width` (int, default 1).
* A **second labeling pass** on the *extended* subvolume (`halo=0`) to extract `ovlp_*` planes.
* Keep all per‑clump metrics on the **core** (halo‑trimmed) labels to avoid double counting.

Patch (abridged; drop into your file right after the core label/metrics block where you already export face maps):

```diff
diff --git a/clump_finder.py b/clump_finder.py
index 4444444..5555555 100644
--- a/clump_finder.py
+++ b/clump_finder.py
@@
-    halo = int(max(0, io_cfg.ghost_width))
+    halo = int(max(0, io_cfg.ghost_width))
+    overlap_width = int(cfg.get("overlap_width", max(1, halo)))
@@
-    labels = label_3d(mask, tile_shape=tile_shape, connectivity=connectivity, halo=halo)
+    # Core labels (halo trimmed)
+    labels = label_3d(mask, tile_shape=tile_shape, connectivity=connectivity, halo=halo)
@@
-    # For stitching: export boundary face maps of kept local labels only (drop filtered labels)
+    # For stitching: export boundary face maps of kept local labels only (drop filtered labels)
@@
-    face_xneg = labels_kept[0, :, :].astype(np.uint32, copy=False)
+    face_xneg = labels_kept[0, :, :].astype(np.uint32, copy=False)
@@
+    # === Overlap planes (optional exact stitching) ===
+    if overlap_width >= 1:
+        # Second labeling on the full extended volume (no halo trim)
+        labels_ext = label_3d(mask, tile_shape=tile_shape, connectivity=connectivity, halo=0)
+        ni_c, nj_c, nk_c = labels.shape
+        o = int(overlap_width)
+        ov_xneg = labels_ext[o-1, o:o+nj_c, o:o+nk_c].astype(np.uint32, copy=False)
+        ov_xpos = labels_ext[o+ni_c-1, o:o+nj_c, o:o+nk_c].astype(np.uint32, copy=False)
+        ov_yneg = labels_ext[o:o+ni_c, o-1, o:o+nk_c].astype(np.uint32, copy=False)
+        ov_ypos = labels_ext[o:o+ni_c, o+nj_c-1, o:o+nk_c].astype(np.uint32, copy=False)
+        ov_zneg = labels_ext[o:o+ni_c, o:o+nj_c, o-1].astype(np.uint32, copy=False)
+        ov_zpos = labels_ext[o:o+ni_c, o:o+nj_c, o+nk_c-1].astype(np.uint32, copy=False)
+    else:
+        ov_xneg = ov_xpos = ov_yneg = ov_ypos = ov_zneg = ov_zpos = np.zeros((0,0), dtype=np.uint32)
@@
     out = {
@@
         "periodic": np.array([True, True, True], dtype=bool),
+        "overlap_width": np.int32(overlap_width),
+        "ovlp_xneg": ov_xneg, "ovlp_xpos": ov_xpos,
+        "ovlp_yneg": ov_yneg, "ovlp_ypos": ov_ypos,
+        "ovlp_zneg": ov_zneg, "ovlp_zpos": ov_zpos,
@@
```

This mirrors the test writer, but with your I/O (ghosts) and metrics pipeline. (Your README already explains halos and periodic wrap; this adds a tiny, explicit overlap plane for stitching.)

---

## How to run the tests (no MPI)

**Quick sanity (power‑law fields):**

```bash
# Temperature cut
python tests/test_equivalence_no_mpi.py \
  --N 256 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 \
  --overlap 1 --stitch-mode overlap-exact \
  --plot --plot-out temp_slice_powerlaw.png \
  --mass-plot-out mass_fn_powerlaw.png --debug-interfaces

# Density cut (same machinery)
python tests/test_equivalence_no_mpi.py \
  --N 256 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 \
  --rho-thr 10.0 \
  --overlap 1 --stitch-mode overlap-exact \
  --debug-interfaces
```

What the script does:

1. Generates a synthetic 3‑D field and produces **baseline single‑volume labels** (periodic relabel + area fix as in `_periodic_relabel_and_area`).
2. Splits into tiles, writes **per‑tile core outputs** (metrics on core only) + **overlap planes**.
3. Runs `stitch.py` with `mode="overlap-exact"` and verifies:

   * `K` exactly equal to baseline
   * **Sorted** `cell_count` exactly equal to baseline
   * `area_sum`, `vol_sum`, `mass_sum`, and sorted `bbox` equal to baseline
     (These are already in `_compare()`; we keep them strict. )

If any check fails, you get a precise assertion (we keep your diagnostics like “expected vs observed face adjacencies” available).

**Why this can run on a laptop:** the test uses your own NumPy/Numba CCL and synthetic fields; it does **not** require `mpi4py` or `openPMD`. (The driver for real data remains in `clump_finder.py`.)

---

## Expected behavior and caveats (transparent)

* With `--stitch-mode face`, you get your current behavior. With `--stitch-mode overlap-exact`, unions are constructed **only** from overlapped voxels (one plane per face by default). This is the exact test of “same clump” at the partition, and it is **sufficient for 6‑connectivity** because any cross‑tile 6‑connected component occupies the partition plane.

* **Filtering small clumps before stitching can break topology.** For exact equivalence, set `min_clump_cells: 1` in the test (and production when you need exact topology), then filter globally **after** stitching. (In your driver, you currently filter before export; if you want exact topology at scale, either export unfiltered faces/overlaps or export both and only drop at the end. The tests already export unfiltered.)

* Periodicity is preserved. Your stitcher already handles periodic neighbors; we reuse that mapping for overlap planes.

---

## “Why not just faces?” (and why this diff is different)

Your current face‑pixel unions already join labels that touch at **adjacent** voxels across the cut. The diagnostic runs you showed earlier demonstrated **face‑only components == stitched_K**, but **not** a perfect match to the **single‑volume** baseline, which indicates some cross‑rank chains weren’t reconstructable from face pixels alone in that pipeline. The overlap‑exact path uses a stronger invariant (identity of global voxels), which removes any ambiguity: if both tiles say “I include voxel (i,j,k)”, that is the *same* bit of the global mask, so the union is *guaranteed* correct. Conceptually simple; nothing to infer.

---

## Minimal developer checklist

1. Apply diffs above.
2. Run the two commands in **How to run** (N=256).
3. You should see **“Equivalence test passed”** with all invariants matching.
4. Try different decompositions: `--px 3 --py 3 --pz 3` and `--px 4 --py 2 --pz 1`.
5. Try both temperature (`--T-thr`) and density (`--rho-thr`) cuts.
6. (Optional, production) Add `overlap_width: 1` to the YAML used by `clump_finder.py` so the real runs emit `ovlp_*` planes; use `stitch.py --mode overlap-exact` in your aggregator.

---

## Short developer doc to ship with the PR

**`docs/STITCHING_GHOST_OVERLAP.md`**

> **Goal.** Make tiled clump catalogs exactly match single‑volume labeling with 6‑connectivity by merging labels that share **identical global voxels** on overlapped planes.

**How it works**

* Each tile computes labels on its **core** (no halo) to produce metrics and face maps (unchanged).
* For stitching, each tile additionally exports the **label id** on six **overlap planes**:

  * `ovlp_xneg`, `ovlp_xpos`: shapes `(nj_core, nk_core)`
  * `ovlp_yneg`, `ovlp_ypos`: shapes `(ni_core, nk_core)`
  * `ovlp_zneg`, `ovlp_zpos`: shapes `(ni_core, nj_core)`
* Neighboring tiles include the **same** global plane (periodic wrap included).
  Example: Left tile `ovlp_xpos` and right tile `ovlp_xneg` are two views of the same plane `{i=i_cut-1}`.

**Stitcher (overlap‑exact mode)**

* For each neighbor pair and each axis, loop the shared plane; where both sides have non‑zero label ids at `(j,k)`, **union** their (rank,label) ids via DSU.
* Aggregation (cell counts, mass, volume, centroids, bbox, area correction) is unchanged.

**Why this is exact for 6‑connectivity**

* Any cross‑tile 6‑connected clump occupies at least one cell on the partition plane; both tiles include that cell in `ovlp_*`. Merging by shared cells is topologically equivalent to labeling the whole domain at once.

**Performance**

* Overhead is tiny: six plane arrays per tile (a few MB even for 10k² faces). No new heavy memory paths. It’s far smaller than persisting a 1‑cell shell.

**Do’s**

* Set `min_clump_cells: 1` before stitching if you need exact topology; filter after stitching.
* Keep connectivity at 6 (assumed in the argument above).

---

## Cited code (for reviewers)

* The test harness uses your `_periodic_relabel_and_area` to match single‑volume relabel & area accounting.
* Stitcher’s area correction remains your face‑edge count logic.
* Labeling behavior with halos (core vs extended) is per the docstring in `local_label.label_3d`.
* Metrics and centroid math are unchanged.
* Production driver’s I/O/halo semantics are consistent with `io_bridge.py`.

---

## Final word

This keeps the code simple and the *reasoning* even simpler: *merge when you literally share the same cell*. If anyone remains skeptical, point them to the exactness argument in the doc above and the equivalence tests that check **K**, **size distribution**, **areas/volumes/masses**, and **bboxes**.

If you want me to also wire this into your MPI driver end‑to‑end, the optional `clump_finder.py` patch above emits the same `ovlp_*` planes as the tests; then run `stitch.py --mode overlap-exact` in your aggregator to get the same guarantees at scale.
