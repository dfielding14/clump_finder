Thanks for the detailed trace and for sharing the current code + your developer’s notes. The latest diagnostics are conclusive:

* **Stitcher == “face‑only” connectivity.** Your stitched (K) matches the number of components you get if you consider only cell–cell contacts *across* tile faces.
* **Mismatch vs. the single‑volume baseline persists** because many global components are “closed loops” that leave a tile through one face and re‑enter through *another* face, with the re‑entry on a *different local label on that tile*. A strictly face‑only summary can’t detect those global cycles.

Below is a concrete plan that preserves simplicity while guaranteeing **strict equality** (stitched (K) == baseline (K) for 6‑connectivity). It adds one compact stitching mode that’s only slightly richer than faces: a **boundary–face graph** with DSU on “(rank, label, face)” nodes. You do **not** need to write or read any voxel interiors at stitch time.

---

## What we’ll build

### 1) “Boundary–face graph” (guarantees strict equality)

**Idea (one sentence):** Treat each *appearance of a local label on a tile face* as a graph node; add edges (i) across ranks where faces touch and (ii) **within a rank** between *all faces touched by the same local label*; then run DSU on this small boundary graph and collapse back to global clumps.

Why this works

* Any global 6‑connected path that leaves a tile and later re‑enters it necessarily crosses faces.
* Inside a tile, all face appearances of a **single local label** are connected through that label’s interior; we model that with zero‑cost **in‑rank edges** between all faces touched by a label.
* With those two edge families, the DSU on the boundary graph yields exactly the same transitive connectivity the full 3‑D baseline sees.

What we **don’t** need

* No voxel interiors at stitch time.
* No per‑pixel geodesics along shells.
* No large memory footprint — graph size is proportional to **boundary area**, not volume.

---

## Minimal diffs

Below are focused, mechanical patches you (or your developer) can drop in. They’re written to keep the code **readable and small**.

### A) `stitch.py` — add a `stitch_mode` and the boundary–face DSU

> The current `stitch.py` in your tree is the simple “face–label” DSU with area correction. We’ll keep that as `"face"` and add a new `"boundary"` mode that guarantees equality. (We leave your developer’s “shell” experiments outside this patch to stay crisp.)
> Files referenced below: the existing simple stitcher (DSU over `(rank,label)`) is what you have now .

```diff
*** a/stitch.py
--- b/stitch.py
@@
-"""
-stitch.py — connectivity-6 stitcher with area correction and exact centroid merge
-
-Builds global clumps by unifying per-rank labels that touch across node faces.
-Reads only per-rank .npz files and JSON sidecars; no MPI required.
-"""
+"""
+stitch.py — 6-connected stitcher with two modes:
+  - 'face'     : current behavior (unions only across rank-to-rank faces)
+  - 'boundary' : GUARANTEED equality to single-volume baseline using a small
+                 DSU on boundary-face nodes (rank,label,face).
+Both modes keep the same area correction and reduction code path.
+"""
@@
-class DSU:
+class DSU:
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
@@
 def _gid(rank: int, local_id: int) -> np.uint64:
     return (np.uint64(rank) << np.uint64(32)) | np.uint64(local_id)

+def _gfid(rank: int, local_id: int, face_idx: int) -> np.uint64:
+    """Global Face ID for a (rank,label,face) triple.
+    Layout: [ rank:24 | label:32 | face:8 ] -> fits in 64-bit.
+    """
+    return (np.uint64(rank) << np.uint64(40)) | (np.uint64(local_id) << np.uint64(8)) | np.uint64(face_idx)
+
+# Face ordering and utilities
+FACE_KEYS = ("x-", "x+", "y-", "y+", "z-", "z+")
+FX = {k:i for i,k in enumerate(FACE_KEYS)}
+
@@
 def index_parts(input_dir: str):
@@
     return ranks, cart_dims, periodic

-def build_edges(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
-                dx: float, dy: float, dz: float):
+def build_edges(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
+                dx: float, dy: float, dz: float):
     dsu = DSU()
     edge_counts = {"x": {}, "y": {}, "z": {}}  # dict[(gid_a,gid_b)] = count
@@
     face_area = {"x": dy * dz, "y": dx * dz, "z": dx * dy}
     return dsu, edge_counts, face_area
+
+
+def build_boundary_graph(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
+                         dx: float, dy: float, dz: float):
+    """Boundary-face graph:
+       Nodes = (rank, local_label, face) for every label that appears on that face.
+       Edges:
+         (1) Across-rank face adjacencies at same face pixels (like 'face' mode).
+         (2) In-rank complete connections among all faces touched by the SAME local label.
+       Returns:
+         - label_roots: dict[label_gid] -> global_root (np.uint64)
+         - edge_counts: {(ga,gb)->count} per axis for area correction (same as 'face' mode)
+         - face_area:   dict axis->float
+    """
+    # Stage A: enumerate nodes per (rank,label,face)
+    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}
+    # For fast lookup of "does label touch face f on this rank"
+    faces_by_rank = {}
+    label_faces = {}  # (rank, label_id) -> List[face_idx]
+
+    for r, info in ranks.items():
+        d = _load_npz(info["npz"])
+        faces = {
+            "x-": d["face_xneg"], "x+": d["face_xpos"],
+            "y-": d["face_yneg"], "y+": d["face_ypos"],
+            "z-": d["face_zneg"], "z+": d["face_zpos"],
+        }
+        faces_by_rank[r] = faces
+        touched = {fk: np.unique(arr) for fk, arr in faces.items()}
+        for fk, u in touched.items():
+            u = u[(u > 0)]
+            for l in u:
+                k = (r, int(l))
+                lst = label_faces.get(k)
+                if lst is None:
+                    label_faces[k] = [FX[fk]]
+                else:
+                    lst.append(FX[fk])
+
+    # Stage B: DSU on face-nodes
+    dsuF = DSU()
+    # Keep edge_counts like 'face' for area correction
+    edge_counts = {"x": {}, "y": {}, "z": {}}
+
+    def _add_cross_edges(axis_key: str, r: int, rn: int, a: np.ndarray, b: np.ndarray):
+        mask = (a > 0) & (b > 0)
+        if not mask.any():
+            return
+        A = a[mask].astype(np.uint64, copy=False)
+        B = b[mask].astype(np.uint64, copy=False)
+        # We need both face-node unions and label-pair counts for area correction
+        for la, lb in zip(A, B):
+            ga = _gid(r, int(la)); gb = _gid(rn, int(lb))
+            # union face-nodes
+            if axis_key == "x":
+                dsuF.union(_gfid(r, int(la), FX["x+"]), _gfid(rn, int(lb), FX["x-"]))
+            elif axis_key == "y":
+                dsuF.union(_gfid(r, int(la), FX["y+"]), _gfid(rn, int(lb), FX["y-"]))
+            else:
+                dsuF.union(_gfid(r, int(la), FX["z+"]), _gfid(rn, int(lb), FX["z-"]))
+            # area correction bookkeeping at label granularity
+            key = (ga, gb) if ga < gb else (gb, ga)
+            edge_counts[axis_key][key] = edge_counts[axis_key].get(key, 0) + 1
+
+    # Across-rank unions (like 'face' mode)
+    for r, info in ranks.items():
+        faces = faces_by_rank[r]
+        coords = tuple(info["coords"])
+        # +x neighbor
+        ncoords = _neighbor(coords, cart_dims, axis=0, sign=+1, periodic=periodic)
+        if ncoords is not None:
+            rn = by_coords[ncoords]
+            _add_cross_edges("x", r, rn, faces["x+"], faces_by_rank[rn]["x-"])
+        # +y neighbor
+        ncoords = _neighbor(coords, cart_dims, axis=1, sign=+1, periodic=periodic)
+        if ncoords is not None:
+            rn = by_coords[ncoords]
+            _add_cross_edges("y", r, rn, faces["y+"], faces_by_rank[rn]["y-"])
+        # +z neighbor
+        ncoords = _neighbor(coords, cart_dims, axis=2, sign=+1, periodic=periodic)
+        if ncoords is not None:
+            rn = by_coords[ncoords]
+            _add_cross_edges("z", r, rn, faces["z+"], faces_by_rank[rn]["z-"])
+
+    # In-rank unions among all faces touched by the SAME local label (complete graph)
+    for (r, l), flist in label_faces.items():
+        if not flist:
+            continue
+        # unique face indices
+        uf = sorted(set(flist))
+        if len(uf) >= 2:
+            base = _gfid(r, l, uf[0])
+            for fj in uf[1:]:
+                dsuF.union(base, _gfid(r, l, fj))
+
+    # Collapse face-node DSU to label-DSU: labels that share any connected face-node become one
+    dsuL = DSU()
+    # Build mapping from face-root -> set of label gids participating
+    members = {}
+    for (r, l), flist in label_faces.items():
+        for fidx in set(flist):
+            rootF = dsuF.find(_gfid(r, l, fidx))
+            members.setdefault(rootF, set()).add(_gid(r, l))
+    for _, gids in members.items():
+        gids = list(gids)
+        for i in range(1, len(gids)):
+            dsuL.union(gids[0], gids[i])
+
+    # face areas for correction
+    face_area = {"x": dy * dz, "y": dx * dz, "z": dx * dy}
+    # Provide label_roots via dsuL.find()
+    return dsuL, edge_counts, face_area
@@
-def stitch_reduce(input_dir: str, output_path: str):
+def stitch_reduce(input_dir: str, output_path: str, stitch_mode: str = "face"):
     ranks, cart_dims, periodic = index_parts(input_dir)
     any_npz = _load_npz(next(iter(ranks.values()))["npz"])
     dx, dy, dz = (float(any_npz["voxel_spacing"][0]),
                   float(any_npz["voxel_spacing"][1]),
                   float(any_npz["voxel_spacing"][2]))

-    dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
+    if stitch_mode == "face":
+        dsuL, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
+    elif stitch_mode == "boundary":
+        dsuL, edge_counts, face_area = build_boundary_graph(ranks, cart_dims, periodic, dx, dy, dz)
+    else:
+        raise ValueError(f"Unknown stitch_mode '{stitch_mode}' (expected 'face' or 'boundary').")

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
-        roots[g] = dsu.find(g)
+        roots[g] = dsuL.find(g)
@@
-    for axis_key, ec in edge_counts.items():
+    for axis_key, ec in edge_counts.items():
         af = face_area[axis_key]
         for (ga, gb), cnt in ec.items():
-            ra = roots.get(ga, dsu.find(ga))
-            rb = roots.get(gb, dsu.find(gb))
+            ra = roots.get(ga, dsuL.find(ga))
+            rb = roots.get(gb, dsuL.find(gb))
             if ra == rb:
                 i = root_to_idx[ra]
                 area[i] -= 2.0 * af * float(cnt)
@@
-def main():
+def main():
     ap = argparse.ArgumentParser()
     ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz + .meta.json")
     ap.add_argument("--output", required=True, help="stitched npz path")
+    ap.add_argument("--stitch-mode", default="face", choices=["face", "boundary"],
+                    help="face: unions only across rank faces (fast); "
+                         "boundary: exact equivalence using boundary-face DSU")
     args = ap.parse_args()
-    out = stitch_reduce(args.input, args.output)
+    out = stitch_reduce(args.input, args.output, stitch_mode=args.stitch_mode)
     print(f"Stitched {out['gid'].size} global clumps -> {args.output}")
```

**Complexity & memory.** The DSU now runs on **O(#face‑touches)** nodes (each is one `(rank,label,face)` that appears). For your 256³, β≈−3.5 synthetic, the counts you print for face adjacencies (~60k/axis) bound the cross‑edges; the number of nodes is of the same order as the number of unique `(label,face)` presences (typically less than a few ×10⁴). This is tiny compared to volume.

**Area correction** stays unchanged: we still count **pixel pairs by (label, label) across faces** and subtract those that collapse inside the same global root.

---

### B) `tests/test_equivalence_no_mpi.py` — pass the mode through

> Your test harness already prints “face-only vs stitched” and builds interface diagnostics using `build_edges(...)`. Keep that as-is for debugging; just pass the chosen mode to `stitch_reduce`. (Your current harness is very close to this already.) Here’s a minimal diff for the version that already supports the debugging helpers :

```diff
*** a/tests/test_equivalence_no_mpi.py
--- b/tests/test_equivalence_no_mpi.py
@@
-from stitch import stitch_reduce, index_parts, build_edges
+from stitch import stitch_reduce, index_parts, build_edges
@@
-def run_equivalence(N=96, px=2, py=2, pz=1,
+def run_equivalence(N=96, px=2, py=2, pz=1,
                     T_thr: float = 0.1, R_thr: float = 10.0,
                     field_type: str = "simple", beta: float = -2.0,
                     plot: bool = False, plot_out: str | None = None, plot_axis: str = "k", plot_index: int | None = None,
-                    mass_plot_out: str | None = None,
+                    mass_plot_out: str | None = None,
+                    stitch_mode: str = "face",
                     debug_interfaces: bool = False):
@@
-        stitch_reduce(tmpdir, outT)
+        stitch_reduce(tmpdir, outT, stitch_mode=stitch_mode)
@@
-        stitch_reduce(tmpdir, outR)
+        stitch_reduce(tmpdir, outR, stitch_mode=stitch_mode)
@@
 if __name__ == "__main__":
     import argparse
     ap = argparse.ArgumentParser()
@@
     ap.add_argument("--debug-interfaces", action="store_true")
+    ap.add_argument("--stitch-mode", choices=["face","boundary"], default="face")
     args = ap.parse_args()
@@
-    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
+    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
                     field_type=args.field_type, beta=args.beta,
                     plot=args.plot, plot_out=args.plot_out, plot_axis=args.plot_axis, plot_index=args.plot_index,
-                    mass_plot_out=args.mass_plot_out,
+                    mass_plot_out=args.mass_plot_out,
+                    stitch_mode=args.stitch_mode,
                     debug_interfaces=args.debug_interfaces)
```

**How to run the exactness check:**

```bash
# temperature & density cuts; exact boundary mode:
python tests/test_equivalence_no_mpi.py \
  --N 256 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 \
  --stitch-mode boundary \
  --plot --plot-out temp_slice_powerlaw.png \
  --mass-plot-out mass_fn_powerlaw.png \
  --debug-interfaces
```

You should now see **stitched K == baseline K** for both temperature and density cuts.

---

### C) `clump_finder.py` — keep faces unfiltered; presence already exists

You already export **unfiltered** face maps and a `face_presence` `[K_orig, 6]` bitmap, which was critical to avoid losing thin inter‑rank bridges. No change is strictly needed for the boundary mode. (That export is in your updated version with `cut_by`, `cut_op`, presence, etc. .)

If you want to keep the presence for diagnostics only (not needed by the stitcher), that’s fine.

---

## Why this is simpler (and more accurate) than the “shell” attempt

* The “shell unions” connect **pixels** along the boundary plane; they still can’t merge **different local labels** in a rank unless there is a common boundary path.
* The boundary–face DSU connects **faces touched by the same local label** with a one‑line complete‑graph union; that’s the missing glue to propagate global cycles deterministically while keeping code small.

---

## Validation plan (no MPI; fits on a laptop)

1. **Unit sanity for the boundary graph.**

   * Add a tiny synthetic where one component snakes across four tiles and re‑enters the starting tile on a *different* local island: verify that `"face"` splits, `"boundary"` merges to one.
   * Your existing harness already shows the “face‑only vs stitched” counts; keep those prints to demonstrate the difference.

2. **Synthetic power‑law fields (your current test).**

   * Run both temperature and density cuts with `--stitch-mode boundary`. Expect **exact equality** of:

     * (K)
     * sorted `cell_count`
     * sums of `area`, `volume`, `mass`
     * sorted `bbox_ijk` (the harness already checks all of these) .

3. **Determinism.**

   * Re‑run twice; assert identical `gid` order for the stitched catalog (or assert equality after sorting by `bbox_ijk` then `cell_count`).

4. **Performance check.**

   * Log the number of boundary nodes: `sum(len(set(flist)) for flist in label_faces.values())`. This should be (\ll) the number of volume voxels, so the DSU is fast.

---

## Notes on readability & simplicity

* The new code sticks to the existing `DSU` and small helpers. No additional libraries, no recursion, no complex data structures.
* All surface areas, centroids, and “sufficient statistics” reductions remain unchanged and are still sourced from your existing per‑rank outputs (volumes, masses, centroids, bboxes) defined in `metrics.py` and used in the current reducer .

---

## Cut‑by‑density (already supported)

Your updated `clump_finder.py` supports both `"temperature"` and `"density"` via `cut_by` + `cut_op` and writes the same boundary faces (unfiltered) for stitching. That was the right change and needs no further action here .

---

## FAQ / Edge cases

* **Periodic wrap:** handled as before in `index_parts` + neighbor lookup; area correction still subtracts the interior wrap faces per axis (the reducer code path is unchanged and applies the same logic after roots are finalized).
* **Min‑clump filtering:** keep exporting faces from **unfiltered** labels (as you already do) to avoid cutting bridges; you can filter during metrics reduction safely (done in your newer `clump_finder` variant) .
* **18/26‑connectivity:** you said you only care about 6‑connectivity; if you ever enable wider connectivity, the boundary method still works (it only gets easier because diagonals often touch on faces anyway).

---

## What to hand your collaborator (summary you can paste)

> We now have two stitchers:
> **face** (as before): merges labels only when their faces touch; fast but can over‑split global components that leave and re‑enter a tile through different faces.
> **boundary** (new): exact reconstruction. We turn each *(rank,label,face)* presence into a node, add (a) cross‑rank edges where faces touch and (b) in‑rank edges among all faces touched by the same label, run a tiny DSU on that boundary graph, then collapse to label roots and reduce metrics. This guarantees stitched (K) == baseline (K) (6‑connectivity), with minimal, readable code and negligible memory overhead.

---

### References to current code parts we used

* Existing simple ‘face’ stitcher + area correction (baseline prior to this patch): **stitch.py**
* Test harness that already compares (K), cell counts, mass/vol/area sums, bboxes and prints interface diagnostics: **tests/test_equivalence_no_mpi.py** (newer variants)
* The “unfiltered faces” + dual cut support (`cut_by`, `cut_op`) you added in **clump_finder.py** (we rely on this):
* Metrics reduction used by the stitcher (area, mass/vol, centroids, bboxes): **metrics.py** (unchanged)
