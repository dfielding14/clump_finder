Great debugging trail — the new test and the subgraph printouts are doing exactly what we hoped: they narrowed this to *stitching logic vs what the baseline expects* rather than “we’re missing interfaces”.

Below I’ll (1) interpret your logs, (2) point out one real correctness bug we **must** fix in the production writer, (3) explain where I agree/disagree with the inference about “isolated nodes”, and (4) give concrete diffs and a crisp plan to settle this conclusively with face‑presence diagnostics and a face‑only baseline comparator.

---

## 1) What your logs actually tell us

* **Interface coverage is complete.**
  In both T‑cut and ρ‑cut runs, “Expected face adjacencies == Observed” in (x, y, z). That means `build_edges` saw every mask-level contact across all inter‑rank faces; we’re not dropping edges in the stitcher.

* **One huge baseline component is where the K mismatch comes from.**
  Your prints show a single baseline label that maps to **thousands** of stitched roots (`14681 → 1744` for T; `16813 → 1962` for ρ). That’s the delta between `K_baseline` and `K_stitched`. The subgraph restricted to that baseline component, however, shows only **~tens** of connected components via face edges (`connected components: 19` with many degree‑0 “nodes”).

* **Translation:** the DSU unions are there, but many per‑rank fragments that your *diagnostic subgraph* associates with the big baseline label do **not** participate in any face union (degree 0). Those will **remain as separate stitched roots** by construction (they never union with anything). That’s why stitched `K` is larger.

---

## 2) One real correctness bug to fix immediately (production writer)

In the current **production** driver (`clump_finder.py`), face maps are written **after** applying `min_clump_cells` filtering. That zeroes out small labels on the faces:

```python
# current code (production)
rank_ids = np.arange(1, K_orig + 1, dtype=np.int32)[keep]
if K_orig:
    lut_faces = np.zeros(K_orig + 1, dtype=np.uint32)
    lut_faces[rank_ids] = rank_ids.astype(np.uint32)
    labels_kept = lut_faces[labels]
else:
    labels_kept = labels
face_xneg = labels_kept[0, :, :].astype(np.uint32)
...
```

If a globally connected clump uses a **narrow bridge** that is small *within a particular rank*, filtering erases that bridge from the face maps → the DSU can’t “see” it → the stitched object splits. This will **definitely** inflate stitched `K` at scale.

> **Fix**: always build face maps from the **raw, pre‑filter** label array so every inter‑rank bridge is preserved; keep filtering only for the per‑rank metrics rows. (The DSU is happy to union via labels that don’t appear in `label_ids` — they act as “transit vertices” that make two kept labels end up in the same root.)

**Drop‑in diff (safe, minimal):**

```diff
*** a/clump_finder.py
--- b/clump_finder.py
@@
-    rank_ids = np.arange(1, K_orig + 1, dtype=np.int32)[keep]
-    # For stitching: export boundary face maps of kept local labels only (drop filtered labels)
-    if K_orig:
-        lut_faces = np.zeros(K_orig + 1, dtype=np.uint32)
-        # keep original IDs to align with label_ids
-        lut_faces[rank_ids] = rank_ids.astype(np.uint32)
-        labels_kept = lut_faces[labels]
-    else:
-        labels_kept = labels
-    face_xneg = labels_kept[0, :, :].astype(np.uint32, copy=False)
-    face_xpos = labels_kept[-1, :, :].astype(np.uint32, copy=False)
-    face_yneg = labels_kept[:, 0, :].astype(np.uint32, copy=False)
-    face_ypos = labels_kept[:, -1, :].astype(np.uint32, copy=False)
-    face_zneg = labels_kept[:, :, 0].astype(np.uint32, copy=False)
-    face_zpos = labels_kept[:, :, -1].astype(np.uint32, copy=False)
+    rank_ids = np.arange(1, K_orig + 1, dtype=np.int32)[keep]
+    # For stitching: export boundary face maps from the UNFILTERED labels.
+    # This preserves thin inter-rank bridges even if they are dropped from per-rank metrics.
+    face_xneg = labels[0, :, :].astype(np.uint32, copy=False)
+    face_xpos = labels[-1, :, :].astype(np.uint32, copy=False)
+    face_yneg = labels[:, 0, :].astype(np.uint32, copy=False)
+    face_ypos = labels[:, -1, :].astype(np.uint32, copy=False)
+    face_zneg = labels[:, :, 0].astype(np.uint32, copy=False)
+    face_zpos = labels[:, :, -1].astype(np.uint32, copy=False)
```

This change aligns the production outputs with the “no‑filter” test writer you’re using now (the test writer already dumps all labels).

---

## 3) Where I agree/disagree with the “isolated node” interpretation

Your dev wrote:

> “Proxy local labels can arbitrarily split a rank’s actual local label into multiple pieces… Isolated nodes ‘touches=none’ appear in the subgraph, but in reality those pieces are fused with face‑touching parts.”

I **agree** on the practical point: the current diagnostic subgraph re‑labels *baseline‑mask‑only* per rank (`label_3d(base_labels==b)`), which is not the exact set of node labels the stitcher uses. That can make the graph look more fragmented than the stitcher’s actual vertex set.

I **disagree** with the implied mechanism for a real split in the stitched catalog:

> In a 6‑connected grid partitioned into axis‑aligned blocks, **every** portion of a global connected component that is not wholly contained in a block **must** intersect that block’s boundary somewhere.
> *Sketch:* If a subset of the component inside a block had empty intersection with the block boundary, there is no 6‑connected path from it to any exterior voxel; it would be a global component by itself — contradiction.

So a truly “face‑invisible” fragment **cannot** belong to a cross‑rank baseline component. If the diagnostic reports such fragments, that’s telling us the *diagnostic node set* (produced by re‑labeling `base_labels==b`) is too fine‑grained, not that the stitcher is missing unions.

**What this means concretely:** the path to reconcile “1744 stitched roots” vs “19 subgraph components” is to construct the subgraph using exactly the **labels and faces that the stitcher sees**, not proxies. (I give a patch below.)

---

## 4) Concrete next steps (patches you can drop in)

### 4.1 Add **per‑label face‑presence** bitsets (cheap, decisive)

Emit six boolean arrays per rank indicating whether each local label appears on each face. This lets the test quantify **how many** local labels that the baseline maps to its “big” component have *zero face presence*. If that count is non‑zero, those labels are provably un‑unionable by any face‑only stitcher — they are not part of the cross‑rank connectivity information set.

**Patch (`clump_finder.py`)** — add after face maps:

```diff
*** a/clump_finder.py
--- b/clump_finder.py
@@
     face_zpos = labels[:, :, -1].astype(np.uint32, copy=False)

+    # Per-label presence on faces (indices 1..K_orig in label space)
+    Kall = int(labels.max())
+    pres_xneg = np.zeros(Kall + 1, dtype=bool); pres_xneg[np.unique(face_xneg)] = True
+    pres_xpos = np.zeros(Kall + 1, dtype=bool); pres_xpos[np.unique(face_xpos)] = True
+    pres_yneg = np.zeros(Kall + 1, dtype=bool); pres_yneg[np.unique(face_yneg)] = True
+    pres_ypos = np.zeros(Kall + 1, dtype=bool); pres_ypos[np.unique(face_ypos)] = True
+    pres_zneg = np.zeros(Kall + 1, dtype=bool); pres_zneg[np.unique(face_zneg)] = True
+    pres_zpos = np.zeros(Kall + 1, dtype=bool); pres_zpos[np.unique(face_zpos)] = True
+    # Drop background (0) for compact per-label arrays aligned with 1..K_orig
+    face_presence = np.stack(
+        [pres_xneg[1:], pres_xpos[1:], pres_yneg[1:], pres_ypos[1:], pres_zneg[1:], pres_zpos[1:]],
+        axis=1
+    )  # shape [K_orig, 6], bool
@@
     out = {
         "label_ids": rank_ids,
@@
         "face_zpos": face_zpos,
+        "face_presence": face_presence,  # [K_orig,6] bool ordered as [x-,x+,y-,y+,z-,z+]
     }
```

This is tiny (~6 bits per label). It doesn’t change the stitcher at all; it just lets the test print:
“of the labels that baseline assigns to **b=14681**, N have **no** face presence → not stitchable by faces”.

### 4.2 Make the diagnostic subgraph use the **real** stitcher vertices

Replace the “proxy local labels” in the test with the actual labels the stitcher uses at faces. That eliminates spurious degree‑0 nodes.

**Diff (`tests/test_equivalence_no_mpi.py`)** — update `_subgraph_component_connectivity` to build the node set from face maps:

```diff
*** a/tests/test_equivalence_no_mpi.py
--- b/tests/test_equivalence_no_mpi.py
@@
-    # Gather local labels per rank and node set for this baseline component
+    # Gather REAL per-rank label IDs from the NPZ that overlap baseline b_label on any face.
     ranks, cart_dims, periodic = index_parts(parts_dir)
-    local_labels = {}
-    node_ids = {}
-    node_index = 0
+    # map (r,local_id) -> node index
+    node_ids = {}; node_index = 0
+    # Keep per-rank small dict of face slices to recover per-face labels later
+    faces = {}
     for r, info in ranks.items():
-        cx, cy, cz = info["coords"]
-        (i0, i1), (j0, j1), (k0, k1) = ((ix[cx][0], ix[cx][1]), (iy[cy][0], iy[cy][1]), (iz[cz][0], iz[cz][1]))
-        sub_mask = (base_labels[i0:i1, j0:j1, k0:k1] == b_label)
-        # Skip if component absent in this rank
-        if not sub_mask.any():
-            continue
-        # Recompute local labels deterministically
-        loc = label_3d((base_labels[i0:i1, j0:j1, k0:k1] == b_label), tile_shape=(128,128,128), connectivity=6, halo=0)
-        local_labels[r] = loc
-        # Record nodes for labels present on this rank
-        labs = np.unique(loc)
-        labs = labs[labs > 0]
-        for l in labs:
-            node_ids[(r, int(l))] = node_index
-            node_index += 1
+        with np.load(info["npz"]) as d:
+            f = {
+                "x-": d["face_xneg"], "x+": d["face_xpos"],
+                "y-": d["face_yneg"], "y+": d["face_ypos"],
+                "z-": d["face_zneg"], "z+": d["face_zpos"],
+            }
+        faces[r] = f
+        # Determine which of these face labels belong to baseline component b_label
+        cx, cy, cz = info["coords"]
+        (i0, i1), (j0, j1), (k0, k1) = (ix[cx], iy[cy], iz[cz])
+        belongs = set()
+        if np.any(base_labels[i0,   j0:j1, k0:k1] == b_label): belongs.update(np.unique(f["x-"]))
+        if np.any(base_labels[i1-1, j0:j1, k0:k1] == b_label): belongs.update(np.unique(f["x+"]))
+        if np.any(base_labels[i0:i1, j0,   k0:k1] == b_label): belongs.update(np.unique(f["y-"]))
+        if np.any(base_labels[i0:i1, j1-1, k0:k1] == b_label): belongs.update(np.unique(f["y+"]))
+        if np.any(base_labels[i0:i1, j0:j1, k0   ] == b_label): belongs.update(np.unique(f["z-"]))
+        if np.any(base_labels[i0:i1, j0:j1, k1-1 ] == b_label): belongs.update(np.unique(f["z+"]))
+        for l in np.array(list(belongs), dtype=np.int64):
+            if l > 0 and (r, int(l)) not in node_ids:
+                node_ids[(r, int(l))] = node_index; node_index += 1
@@
-    # Build edges by scanning interfaces restricted to baseline label
+    # Build edges by scanning interfaces; use ONLY real face labels
     by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}
     for r, info in ranks.items():
-        if r not in local_labels:
-            continue
+        if r not in faces:
+            continue
         cx, cy, cz = info["coords"]
         (i0, i1), (j0, j1), (k0, k1) = ((ix[cx][0], ix[cx][1]), (iy[cy][0], iy[cy][1]), (iz[cz][0], iz[cz][1]))
         # X interface
         ncoords = ((cx+1)%px, cy, cz)
         rn = by_coords[ncoords]
-        if rn in local_labels:
-            a_mask = (base_labels[i1-1, j0:j1, k0:k1] == b_label)
-            b_mask = (base_labels[ix[ncoords[0]][0], j0:j1, k0:k1] == b_label)
-            m = a_mask & b_mask
-            if m.any():
-                A = local_labels[r][(i1-1)-i0, :, :][m]
-                B = local_labels[rn][(ix[ncoords[0]][0])-ix[ncoords[0]][0], :, :][m]  # left face is index 0
-                for la, lb in zip(A, B):
-                    if la>0 and lb>0:
-                        uni(node_ids[(r,int(la))], node_ids[(rn,int(lb))]); ex += 1
+        a_mask = (base_labels[i1-1, j0:j1, k0:k1] == b_label)
+        b_mask = (base_labels[ix[ncoords[0]][0], j0:j1, k0:k1] == b_label)
+        m = a_mask & b_mask
+        if m.any():
+            A = faces[r]["x+"][m]; B = faces[rn]["x-"][m]
+            for la, lb in zip(A, B):
+                if la>0 and lb>0 and (r,int(la)) in node_ids and (rn,int(lb)) in node_ids:
+                    uni(node_ids[(r,int(la))], node_ids[(rn,int(lb))]); ex += 1
@@
         # Y interface
-        if rn in local_labels:
-            a_mask = (base_labels[i0:i1, j1-1, k0:k1] == b_label)
-            b_mask = (base_labels[i0:i1, iy[ncoords[1]][0], k0:k1] == b_label)
-            m = a_mask & b_mask
-            if m.any():
-                A = local_labels[r][:, (j1-1)-j0, :][m]
-                B = local_labels[rn][:, 0, :][m]
-                for la, lb in zip(A, B):
-                    if la>0 and lb>0:
-                        uni(node_ids[(r,int(la))], node_ids[(rn,int(lb))]); ey += 1
+        a_mask = (base_labels[i0:i1, j1-1, k0:k1] == b_label)
+        b_mask = (base_labels[i0:i1, iy[ncoords[1]][0], k0:k1] == b_label)
+        m = a_mask & b_mask
+        if m.any():
+            A = faces[r]["y+"][m]; B = faces[rn]["y-"][m]
+            for la, lb in zip(A, B):
+                if la>0 and lb>0 and (r,int(la)) in node_ids and (rn,int(lb)) in node_ids:
+                    uni(node_ids[(r,int(la))], node_ids[(rn,int(lb))]); ey += 1
@@
         # Z interface
-        if rn in local_labels:
-            a_mask = (base_labels[i0:i1, j0:j1, k1-1] == b_label)
-            b_mask = (base_labels[i0:i1, j0:j1, iz[ncoords[2]][0]] == b_label)
-            m = a_mask & b_mask
-            if m.any():
-                A = local_labels[r][:, :, (k1-1)-k0][m]
-                B = local_labels[rn][:, :, 0][m]
-                for la, lb in zip(A, B):
-                    if la>0 and lb>0:
-                        uni(node_ids[(r,int(la))], node_ids[(rn,int(lb))]); ez += 1
+        a_mask = (base_labels[i0:i1, j0:j1, k1-1] == b_label)
+        b_mask = (base_labels[i0:i1, j0:j1, iz[ncoords[2]][0]] == b_label)
+        m = a_mask & b_mask
+        if m.any():
+            A = faces[r]["z+"][m]; B = faces[rn]["z-"][m]
+            for la, lb in zip(A, B):
+                if la>0 and lb>0 and (r,int(la)) in node_ids and (rn,int(lb)) in node_ids:
+                    uni(node_ids[(r,int(la))], node_ids[(rn,int(lb))]); ez += 1
```

This makes the “Nodes=…, Edges=…, components=…” printout reflect the **same graph** the stitcher is using, so “19 vs 1744” should collapse to the same number.

### 4.3 Quantify “face‑invisible” labels in the test

After the change above, add a quick report in `_diagnose_component_mismatch`:

```diff
*** a/tests/test_equivalence_no_mpi.py
--- b/tests/test_equivalence_no_mpi.py
@@
     for r, info in ranks.items():
         ...
         with np.load(info["npz"]) as d:
             lids = d["label_ids"].astype(np.int64)
+            if "face_presence" in d.files:
+                presence = d["face_presence"].astype(bool)  # [K_orig,6]
+            else:
+                presence = None
@@
-            s.add(root)
+            s.add(root)
+            # count truly face-invisible cases (if presence is available)
+            if presence is not None and l<=presence.shape[0]:
+                if not np.any(presence[l-1]):  # no face presence on this rank
+                    # you can tally per-baseline-label how many such labels exist
+                    pass
```

(Feel free to add an accumulator and print the counts right after the “worst offender” inspection.)

---

## 5) A “face‑only” baseline comparator (so we test the exact information model)

To put this beyond doubt in review with the team:

* Keep your existing, strict “baseline single volume (fully informed) vs stitched (faces‑only)” comparison.
* **Also** add a second comparator that reconstructs a *baseline graph from faces only* (using the same `build_edges` over the full‑volume mask split into px×py×pz blocks) and counts components with a DSU. That number should **exactly** equal stitched `K` if the stitcher is correct.

You already have all the pieces in `tests/test_equivalence_no_mpi.py` (`index_parts`, `build_edges` are imported and used), so this is a tiny addition.

---

## 6) Why this resolves the disagreement

* The **production** bug (faces built post‑filter) would cause exactly the kind of K inflation you worry about at scale. The diff in §2 eliminates that root cause in the real pipeline.

* In the **test**, your diagnostics currently mix two vertex sets (proxy per‑rank pieces from `base_labels==b` vs the stitcher’s real per‑rank label IDs present on faces). Switching to the latter removes spurious “isolated nodes”. The counts printed by the subgraph will then match what DSU actually does in `stitch.py`.

* **Face‑presence** bitsets let us *measure* the gap between “fully informed baseline” and “what faces alone can recover”. If any baseline‑assigned per‑rank fragment has zero presence on all faces, no face‑only stitcher can ever merge it — and that’s a scientifically honest finding to share with skeptical collaborators.

---

## 7) Sanity: the stitcher itself is doing the right thing

The current stitcher:

* Builds DSU over `(rank,label)` using **all** non‑zero face label matches (wraps periodic axes),
* Accumulates additive stats and corrects area by subtracting `2·a_face·count` for interior faces,
* Recomputes centroids exactly from weighted sums.
  This is the minimal, deterministic algorithm we outlined earlier, and your “expected == observed interface counts” corroborate `build_edges` is complete.

---

## 8) Documentation nits (so the repo narrative matches the behavior)

Add to the README:

* Stitching is **6‑connectivity only**, face unions only; components that never touch a partition face are indistinguishable from single‑rank components in the stitched catalog.
* Face maps are emitted **pre‑filter** by design, so thin bridges are preserved in stitching even if they are dropped from per‑rank metrics.

---

## 9) TL;DR recommendations

1. **Apply the face‑maps pre‑filter fix** in `clump_finder.py` (diff in §2). This is a correctness requirement for production.
2. **Adopt face‑presence bitsets** (diff in §4.1) and print them in the test.
3. **Switch the diagnostic subgraph to the stitcher’s real vertices** (diff in §4.2).
4. Optionally add a **face‑only baseline comparator** (same DSU, same edges) for an apples‑to‑apples check.

Once these land, re‑run your `N=256, β=-3.5` test. You should see:

* Interface counts still match (good).
* The “worst offender” baseline label decomposes into **the same** number of face‑connected components in both the diagnostic subgraph and the stitched result.
* Any residual mismatch with the fully‑informed baseline `K` will be quantified by the count of **face‑invisible** fragments (if any), which is exactly the information shortfall of a face‑only stitcher — and is a clear, defensible result to explain to collaborators.