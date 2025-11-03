Below is a complete, developer‑facing sketch of the **“richer than faces”** stitching design you asked for, written so you (and your collaborator) can drop it into the current codebase and verify that the stitched result **matches the single‑volume baseline** for 6‑connectivity.

The core idea is simple and keeps the code readable:

> **Persist a 1‑cell–thick “shell” per rank** (you already persist the six faces), and during stitching add **in‑rank unions along the shell** (neighbors within each face + tie the 12 edges and 8 corners where faces overlap) **in addition** to the existing **rank‑to‑rank face unions**.
> With this, the DSU sees all paths a global 6‑connected component can take as it leaves a rank, traverses neighbors, and re‑enters—so the stitched K equals the full baseline K.

You already write the six faces per rank (x±, y±, z±) in the modern `clump_finder.py` and those are derived from the **unfiltered** labels specifically to preserve thin bridges (good!) . The changes below only add **edge builders** on top of the faces you already have; we do **not** write any large new arrays.

---

## Why the 1‑cell shell is sufficient (for 6‑connectivity)

* In 6‑connectivity, any path that crosses from one rank to another **must** cross a **face voxel**.
* A global component may traverse multiple ranks and re‑enter a rank at a different face/pixel. If we make the DSU aware of:

  1. every **cross‑rank** face contact (what you already do), **and**
  2. every **within‑rank** adjacency **on the shell** (so paths can slide along a rank’s boundary and along face–edge–corner overlaps),

  then the DSU’s transitive closure reproduces the same merge decisions as a monolithic CCL on the whole domain at 6‑connectivity. No diagonals are introduced, and we never rely on 18/26‑connectivity.

This is the minimal addition beyond faces: the shell carries exactly the information the stitcher still lacked to let global cycles “close” through a rank even when the entry/exit pixels differ.

---

## What we will change (high level)

1. **Persist nothing new** in the data path (you already export all six faces from the unfiltered labels). We only read the faces and derive the shell connectivity at stitch time.

   * If you prefer explicit dims, we can write a tiny `shape_ijk` vector, but we can already infer `(ni, nj, nk)` from face shapes (e.g., `y-` is `(ni, nk)`, `x-` is `(nj, nk)`, `z-` is `(ni, nj)`).
   * You already emit a per‑label face‑presence diagnostic; we keep using it for debugging but it’s not required for stitching logic.
2. **Augment `stitch.py`’s edge builder** with two extra, very small, in‑rank steps:

   * **Within‑face adjacency (2‑D 6‑neighbors)**: along rows/cols on each of the six faces for the *same* rank we union adjacent positive labels; this is mostly idempotent (same local label) and cheap to evaluate.
   * **Edge/corner tie‑ups (face overlaps)**: the same boundary voxel appears on 2 faces along an edge and on 3 faces at a corner; union those appearances so DSU can move freely along the shell graph inside a rank.
     The cross‑rank face unions remain exactly as they are now.
3. **Expose an option** to select stitching mode (`--stitch-mode face|shell`), defaulting to `shell` for strict equality tests, while keeping `face` for backwards‑compatibility.
4. **Add a test switch** in `tests/test_equivalence_no_mpi.py` to call the new mode and assert `K_baseline == K_stitched` for both temperature and density cuts.

---

## Exact diffs (drop‑in)

### 1) `stitch.py` – add shell unions (≈60–80 lines)

```diff
*** a/stitch.py
--- b/stitch.py
@@
-def build_edges(ranks: dict, cart_dims: Tuple[int, int, int], periodic: Tuple[bool, bool, bool],
-                dx: float, dy: float, dz: float):
+def build_edges(ranks: dict,
+                cart_dims: Tuple[int, int, int],
+                periodic: Tuple[bool, bool, bool],
+                dx: float, dy: float, dz: float,
+                mode: str = "face"):
     dsu = DSU()
     edge_counts = {"x": {}, "y": {}, "z": {}}  # dict[(gid_a,gid_b)] = count

     by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}

+    # --- helper to union within a 2-D face array for a single rank ---
+    def _union_face_inplane(r: int, face2d: np.ndarray, step: Tuple[int, int]):
+        """Union neighbors inside one face along 'step' (dj, dk) or (di, dk) etc.
+        Only unions pairs with both labels > 0 and different ids to avoid no-ops."""
+        if face2d.size == 0:
+            return
+        sj, sk = step
+        # roll-and-mask trick for adjacency
+        A = face2d
+        B = np.roll(face2d, shift=-sj, axis=0) if sj else face2d
+        B = np.roll(B,       shift=-sk, axis=1) if sk else B
+        # valid region (avoid wrap adjacency on the last row/col)
+        if sj:
+            A = A[:-1, :]
+            B = B[:-1, :]
+        if sk:
+            A = A[:, :-1]
+            B = B[:, :-1]
+        m = (A > 0) & (B > 0) & (A != B)
+        if not m.any():
+            return
+        a = A[m].astype(np.uint64, copy=False)
+        b = B[m].astype(np.uint64, copy=False)
+        for la, lb in zip(a, b):
+            dsu.union(_gid(r, int(la)), _gid(r, int(lb)))
+
+    # --- helper to union overlaps along an edge or corner between two faces ---
+    def _union_face_overlap_line(r: int, A: np.ndarray, B: np.ndarray):
+        """Union same-voxel line where two faces overlap (both 1-D views).
+        Shapes must match; unions only when both labels > 0 and ids differ."""
+        if A.size == 0 or B.size == 0:
+            return
+        assert A.shape == B.shape
+        m = (A > 0) & (B > 0) & (A != B)
+        if not m.any():
+            return
+        for la, lb in zip(A[m].astype(np.uint64), B[m].astype(np.uint64)):
+            dsu.union(_gid(r, int(la)), _gid(r, int(lb)))
+
+    # Optionally add in-rank "shell" unions before cross-rank edges
+    if mode == "shell":
+        for r, info in ranks.items():
+            npz = _load_npz(info["npz"])
+            # 6 faces
+            fxm = npz["face_xneg"]  # (nj, nk)
+            fxp = npz["face_xpos"]  # (nj, nk)
+            fym = npz["face_yneg"]  # (ni, nk)
+            fyp = npz["face_ypos"]  # (ni, nk)
+            fzm = npz["face_zneg"]  # (ni, nj)
+            fzp = npz["face_zpos"]  # (ni, nj)
+            nj, nk = fxm.shape
+            ni, nk2 = fym.shape
+            ni2, nj2 = fzm.shape
+            assert nk == nk2 and ni == ni2 and nj == nj2, "face shapes inconsistent"
+            # (a) within-face adjacency (2-D 6-neighbors → in-plane 4-neighbors)
+            # x-faces: axes are (j,k)
+            _union_face_inplane(r, fxm, (1,0)); _union_face_inplane(r, fxm, (0,1))
+            _union_face_inplane(r, fxp, (1,0)); _union_face_inplane(r, fxp, (0,1))
+            # y-faces: axes are (i,k)
+            _union_face_inplane(r, fym, (1,0)); _union_face_inplane(r, fym, (0,1))
+            _union_face_inplane(r, fyp, (1,0)); _union_face_inplane(r, fyp, (0,1))
+            # z-faces: axes are (i,j)
+            _union_face_inplane(r, fzm, (1,0)); _union_face_inplane(r, fzm, (0,1))
+            _union_face_inplane(r, fzp, (1,0)); _union_face_inplane(r, fzp, (0,1))
+            # (b) edge overlaps (tie identical boundary voxels seen by 2 faces)
+            # x± with y± along k (four lines each)
+            _union_face_overlap_line(r, fxm[0, :],       fym[0, :])        # (i=0,j=0, k[:])
+            _union_face_overlap_line(r, fxm[nj-1, :],    fyp[0, :])        # (i=0,j=nj-1)
+            _union_face_overlap_line(r, fxp[0, :],       fym[ni-1, :])     # (i=ni-1,j=0)
+            _union_face_overlap_line(r, fxp[nj-1, :],    fyp[ni-1, :])     # (i=ni-1,j=nj-1)
+            # x± with z± along j
+            _union_face_overlap_line(r, fxm[:, 0],       fzm[0, :])        # (i=0,k=0, j[:])
+            _union_face_overlap_line(r, fxm[:, nk-1],    fzp[0, :])        # (i=0,k=nk-1)
+            _union_face_overlap_line(r, fxp[:, 0],       fzm[ni-1, :])     # (i=ni-1,k=0)
+            _union_face_overlap_line(r, fxp[:, nk-1],    fzp[ni-1, :])     # (i=ni-1,k=nk-1)
+            # y± with z± along i
+            _union_face_overlap_line(r, fym[:, 0],       fzm[:, 0])        # (j=0,k=0, i[:])
+            _union_face_overlap_line(r, fym[:, nk-1],    fzp[:, 0])        # (j=0,k=nk-1)
+            _union_face_overlap_line(r, fyp[:, 0],       fzm[:, nj-1])     # (j=nj-1,k=0)
+            _union_face_overlap_line(r, fyp[:, nk-1],    fzp[:, nj-1])     # (j=nj-1,k=nk-1)
+            # (c) corners (three-way tie: union all three pairwise)
+            corners = [
+              (fxm[0, 0],    fym[0, 0],    fzm[0, 0]),        # (0,0,0)
+              (fxm[0, nk-1], fym[0, nk-1], fzp[0, 0]),        # (0,0,nk-1)
+              (fxm[nj-1, 0], fyp[0, 0],    fzm[0, nj-1]),     # (0,nj-1,0)
+              (fxm[nj-1, nk-1], fyp[0, nk-1], fzp[0, nj-1]),  # (0,nj-1,nk-1)
+              (fxp[0, 0],    fym[ni-1, 0], fzm[ni-1, 0]),     # (ni-1,0,0)
+              (fxp[0, nk-1], fym[ni-1, nk-1], fzp[ni-1, 0]),  # (ni-1,0,nk-1)
+              (fxp[nj-1, 0], fyp[ni-1, 0], fzm[ni-1, nj-1]),  # (ni-1,nj-1,0)
+              (fxp[nj-1, nk-1], fyp[ni-1, nk-1], fzp[ni-1, nj-1]) # (ni-1,nj-1,nk-1)
+            ]
+            for a,b,c in corners:
+                # union pairwise when labels are >0 and not equal
+                if a>0 and b>0 and a!=b: dsu.union(_gid(r,int(a)), _gid(r,int(b)))
+                if a>0 and c>0 and a!=c: dsu.union(_gid(r,int(a)), _gid(r,int(c)))
+                if b>0 and c>0 and b!=c: dsu.union(_gid(r,int(b)), _gid(r,int(c)))
+
     def add_edges(axis_key: str, r: int, rn: int, a: np.ndarray, b: np.ndarray):
         mask = (a > 0) & (b > 0)
         if not mask.any():
             return
         A = a[mask].astype(np.uint64, copy=False)
@@
-    for r, info in ranks.items():
+    for r, info in ranks.items():
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
@@
-    return dsu, edge_counts, face_area
+    return dsu, edge_counts, face_area
@@
-def stitch_reduce(input_dir: str, output_path: str):
+def stitch_reduce(input_dir: str, output_path: str, mode: str = "face"):
     ranks, cart_dims, periodic = index_parts(input_dir)
     any_npz = _load_npz(next(iter(ranks.values()))["npz"])
     dx, dy, dz = (float(any_npz["voxel_spacing"][0]),
                   float(any_npz["voxel_spacing"][1]),
                   float(any_npz["voxel_spacing"][2]))

-    dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)
+    dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz, mode=mode)
@@
 def main():
     ap = argparse.ArgumentParser()
     ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz + .meta.json")
     ap.add_argument("--output", required=True, help="stitched npz path")
+    ap.add_argument("--stitch-mode", choices=["face", "shell"], default="face",
+                    help="stitch with faces only (baseline) or faces + in-rank shell unions")
     args = ap.parse_args()
-    out = stitch_reduce(args.input, args.output)
+    out = stitch_reduce(args.input, args.output, mode=args.stitch_mode)
     print(f"Stitched {out['gid'].size} global clumps -> {args.output}")
```

Rationale / guardrails:

* **We never add area corrections** for in‑rank unions; area subtraction only applies to *cross‑rank* faces and remains exactly where it is today (unchanged loop at the end of `stitch_reduce`). This preserves your existing exposed‑area math.
* The functions `_union_face_inplane` and `_union_face_overlap_line` are deliberately tiny so they are easy to review and reason about.

### 2) `tests/test_equivalence_no_mpi.py` – allow selecting shell mode

Enable calling the new mode from the test harness you’ve been running. (Use the **newer** test entry you attached that already has the richer diagnostics; the snippet below patches that file.)

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
-                    debug_interfaces: bool = False):
+                    mass_plot_out: str | None = None,
+                    debug_interfaces: bool = False,
+                    stitch_mode: str = "face"):
@@
-        stitch_reduce(tmpdir, outT)
+        stitch_reduce(tmpdir, outT, mode=stitch_mode)
@@
-        stitch_reduce(tmpdir, outR)
+        stitch_reduce(tmpdir, outR, mode=stitch_mode)
@@
 if __name__ == "__main__":
     import argparse
     ap = argparse.ArgumentParser()
@@
     ap.add_argument("--debug-interfaces", action="store_true")
+    ap.add_argument("--stitch-mode", choices=["face", "shell"], default="face")
     args = ap.parse_args()
@@
-    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
+    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
                     field_type=args.field_type, beta=args.beta,
                     plot=args.plot, plot_out=args.plot_out, plot_axis=args.plot_axis, plot_index=args.plot_index,
                     mass_plot_out=args.mass_plot_out,
-                    debug_interfaces=args.debug_interfaces)
+                    debug_interfaces=args.debug_interfaces,
+                    stitch_mode=args.stitch_mode)
```

Usage:

```bash
# Face-only (what you ran before)
python tests/test_equivalence_no_mpi.py --N 256 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 --debug-interfaces --stitch-mode face

# Shell mode (strict equality target)
python tests/test_equivalence_no_mpi.py --N 256 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 --debug-interfaces --stitch-mode shell
```

Expectations:

* In `--stitch-mode shell` you should now see `K` match the baseline for both temperature and density cuts in the “Equivalence” check (and we keep the mass/volume/area invariants identical as today). The test harness is already validating those invariants.

---

## Notes on the current writer and why we don’t add new files

* **Faces are already unfiltered** (so we don’t lose micron‑thin bridges); the older writer variant filtered faces after dropping small clumps, but your latest file has fixed this by exporting faces from the unfiltered `labels` and only filtering the *per‑clump rows* afterward. Great. No changes needed.
* You already added **density cuts** (configurable `cut_by: temperature|density` and `cut_op: lt|gt`), which was part of your original feature request. The test script uses both modes.

---

## Complexity & memory

* **No new I/O**: we reuse the existing face arrays.
* **CPU**: per rank, we scan 6 faces once and union a handful of 1‑D edge lines (~O(ni nj + nj nk + nk ni) operations). This is cheap compared to the on‑node CCL, and trivial compared to reading the data.
* **Area accounting is unchanged**: we only subtract the internal area contribution for *cross‑rank* unions, as before. In‑rank unions do not touch the area budget.

---

## How this closes the gap observed in your debug logs

From your run:

```
[face-only] components=40432 vs stitched_K=40432
... baseline K smaller (e.g., 38689)
```

This demonstrated (correctly) that the current stitcher is an exact face‑only reconstruction of global connectivity. The remaining gap occurs when the global component’s path **enters and exits** a tile on **different face pixels** and must rely on **within‑rank** movement along the boundary to complete cycles. The new **shell unions** give DSU exactly that within‑rank movement (along face grids and across their shared edges/corners), so those multi‑tile cycles collapse to the **single baseline component** instead of hundreds/thousands of small stitched roots. The diagnostic print you already have (“[face‑only] … vs stitched_K …”) will still agree in face mode; in shell mode you should see **baseline K == stitched K**.

---

## Optional: one assertion to keep writer & stitcher honest

Add (optional) writer‑side assert to catch accidental face filtering or shape mismatches:

```python
# in clump_finder.py right after faces are built
assert face_xneg.ndim == 2 and face_yneg.ndim == 2 and face_zneg.ndim == 2
# Derive ni,nj,nk from faces and assert internal-labeled shape matches
ni = face_yneg.shape[0]; nj = face_xneg.shape[0]; nk = face_xneg.shape[1]
assert labels.shape == (ni, nj, nk), "face shapes inconsistent with labels"
```

(This is purely defensive; not strictly required.)

---

## Sanity tests to run (laptop‑friendly, no MPI)

1. **Unit vox box** (existing synthetic): run the analytic area & centroid checks (already present). This validates per‑clump geometry independent of stitching.
2. **Synthetic Gaussian/power‑law fields (small N)**:

```bash
# Strict equality (shell mode) small domain
python tests/test_equivalence_no_mpi.py --N 96 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 --stitch-mode shell

# Face-only diagnostic (to confirm the gap is face-only, not a bug)
python tests/test_equivalence_no_mpi.py --N 96 --px 2 --py 2 --pz 2 \
  --field-type powerlaw --beta -3.5 --debug-interfaces --stitch-mode face
```

The script already compares K, mass/volume/area sums, and bboxes for both temperature and density cuts; it also prints face adjacency counts for “expected vs observed” and drills into mismatched components. With shell mode enabled, the K mismatch messages should disappear.

---

## Readability & simplicity checklist

* All new logic lives in **two tiny helpers** inside `stitch.py`, called once per rank.
* No extra files, no changes to the on‑node CCL or metrics code. (Your on‑rank labelling and metrics remain exactly as they are; they’re solid and well‑factored already.)
* The command‑line surface adds **one flag** `--stitch-mode` so we can A/B test “face” vs “shell” anytime.

---

## What we did *not* do (by design)

* We did **not** change the DSU node definition (still one node per `(rank, local_label)`), so no downstream changes are needed in the reducer that sums masses/volumes/centroids and applies face‑area corrections when a union closes a face internally. Your current reducer remains correct.
* We did **not** invent any diagonal (18/26‑connectivity) adjacency; all unions use 6‑neighbors on faces and the physically identical shell voxels at edges/corners.
* We did **not** add heavy metadata or full 3‑D boundary volumes; the faces you already persist are sufficient.

---

## Recap / TL;DR for your collaborator

* **Problem verified**: face‑only stitching exactly equals “face‑only” ground truth, but can over‑segment vs a single global CCL (your logs show that clearly).
* **Minimal fix**: add in‑rank shell unions (within‑face + edge/corner overlaps) before the existing cross‑rank unions.
* **Implementation**: ~70 lines in `stitch.py` (diff above), one optional CLI flag, zero new data written.
* **Outcome**: in `--stitch-mode shell`, you should see **stitched K == baseline K** on your `N=256, β=-3.5` case for both T and ρ cuts, while all mass/volume/area invariants remain identical to the baseline tests you already run.
