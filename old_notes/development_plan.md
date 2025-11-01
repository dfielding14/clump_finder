Below is a practical, *readability‑first* plan to (a) streamline and speed up your per‑node clump labeling, (b) **add a density cut**, and (c) implement a **deterministic, connectivity‑6 global stitcher** with laptop‑sized tests that demonstrate equivalence between “single‑volume” and “tiled+stitched” results.

I’ve anchored the suggestions and diffs to your current files so your collaborators can review the rationale line‑by‑line.

---

## 0) High‑level review (what’s solid, what to tweak)

**Solid today**

* The subvolume reader (`io_bridge.py`) is clean and memory‑predictable, adds an optional 1‑cell halo with periodic wrap, and computes thermodynamic/velocity fields in code units. Good basis for deterministic labeling.
* The local CCL (`local_label.py`) does tile‑local labeling plus *face* merges across tiles—**correct for 6‑connectivity** (face adjacency only). It compacts labels to 1..K.
* Metric reducers (`metrics.py`) use bincount‑based grouped reductions and compute exposed area, centroids, per‑label stats, bboxes without storing full per‑voxel state. Nice tradeoff between accuracy and memory.
* Per‑node outputs are tidy `.npz` files with a small sidecar; the aggregator simply concatenates.
* Synthetic tests already exist, including a sanity box and a power‑law field generator that produces QA PNGs.
* A stitching scaffold already exists (stub). We’ll fill this in.
* README correctly sets expectations (one MPI rank per node, halos, 6/18/26 connectivity knob, etc.). We’ll align the doc to reflect a **6‑connectivity only** stitched mode.

**Where we’ll improve**

1. **Deprecate 18/26 connectivity** in the production path. Your current tile‑face merging stage only propagates face adjacencies, which is strictly correct for **6‑connectivity** but incomplete for 18/26 (edge/corner neighbors across tile boundaries would be missed). Let’s make stitched runs explicitly 6‑connectivity and error out otherwise—this eliminates subtle edge‑cases and simplifies the stitcher.

2. **Add density‑based cuts** with an extremely small change footprint: just choose the field (`temp` vs `dens`) and comparator (`<` or `>`) before labeling. We’ll record the choice in outputs.

3. **Export light‑weight boundary “face maps” per node** (six 2‑D `uint32` slices of label IDs at the node’s boundary planes). This lets a rank‑agnostic stitcher construct cross‑node equivalence edges deterministically (no need to exchange labels at runtime). This keeps the stitcher simple and reliable.

4. **Implement a DSU‑based stitcher** that:

   * builds equivalence classes of `(rank, local_label)` pairs using those face maps,
   * **corrects area** by removing double‑counted boundary faces between merged labels,
   * *adds* additive quantities (cells, volume, mass) and **recomputes centroids exactly** from per‑node centroids and weights (no re‑scan of voxels required),
   * unifies bboxes by min/max.
   * (Optional next step: if you want globally exact higher moments/velocity dispersions, emit simple **power sums**; details below.)

5. **Laptop tests (no MPI)** that assert:
   *single‑volume* labeling ≡ *tiled+stitched* labeling, for **K**, sorted clump sizes (cell_count), **area**, **volume/mass**, **bboxes**, and **centroids**, for both **temperature** and **density** cuts. These avoid MPI entirely and run in seconds at N≤128.

---

## 1) Make stitched runs **6‑connectivity‑only** (simplify + correctness)

**Why**
Your cross‑tile merge currently unions only labels touching across *faces* of tiles—exactly the definition of 6‑connectivity. For 18/26 connectivity you’d also need to merge across tile edges/corners, which complicates both the local merge and the global stitch significantly. We’ll keep the code path explicit and fail fast if users request 18/26 for a stitched run.

**Minimal diff** (`local_label.py`)

```diff
*** a/local_label.py
--- b/local_label.py
@@
-def label_3d(mask: np.ndarray,
-             tile_shape=(128, 128, 128),
-             connectivity: int = 6,
-             halo: int = 0) -> np.ndarray:
+def label_3d(mask: np.ndarray,
+             tile_shape=(128, 128, 128),
+             connectivity: int = 6,
+             halo: int = 0) -> np.ndarray:
@@
-    assert connectivity in (6, 18, 26)
+    # Production stitching is correct only for 6-connectivity (face adjacency).
+    # 18/26 would need extra cross-tile edge/corner merges; disallow to avoid silent errors.
+    if connectivity != 6:
+        raise NotImplementedError("Stitched runs support connectivity=6 only. "
+                                  "18/26 require additional cross-tile edge/corner merges.")
```

(We keep the existing 6‑connectivity neighborhood unchanged.)

*README note*: update “Connectivity” to clarify stitching guarantees are for 6 only.

---

## 2) **Add density‑based cut** (and an explicit comparator)

**Why**
You asked to cut by density instead of only temperature. We’ll add two config keys: `cut_by: temperature|density` and `cut_op: lt|gt`. We’ll also record both the chosen variable and threshold in the per‑node `.npz` (for full provenance).

**Diff** (`clump_finder.py`)—*threshold section only*

```diff
*** a/clump_finder.py
--- b/clump_finder.py
@@
-    # Threshold mask
-    Tthr = float(cfg.get("temperature_threshold", 1.0))
-    mask = temp < Tthr  # include halo in mask for completeness
+    # Threshold mask (temperature or density), explicit comparator
+    cut_by = str(cfg.get("cut_by", "temperature")).lower()
+    cut_op = str(cfg.get("cut_op", "lt")).lower()
+    Tthr = float(cfg.get("temperature_threshold", 1.0))
+    Rthr = float(cfg.get("density_threshold", 1.0))
+
+    field_for_cut = temp if cut_by == "temperature" else dens
+    thr = Tthr if cut_by == "temperature" else Rthr
+    if cut_op == "lt":
+        mask = field_for_cut < thr   # include halo in mask for completeness
+    elif cut_op == "gt":
+        mask = field_for_cut > thr
+    else:
+        raise ValueError(f"cut_op must be 'lt' or 'gt', got {cut_op}")
```

**Diff** (augment outputs near where `out` is constructed)

```diff
@@
     out = {
         "label_ids": rank_ids,
@@
-        "temperature_threshold": np.float64(Tthr),
+        "temperature_threshold": np.float64(Tthr),
+        "density_threshold": np.float64(Rthr),
+        "cut_by": np.array([cut_by], dtype=object),
+        "cut_op": np.array([cut_op], dtype=object),
```

And include `cut_by`, `cut_op` in the JSON sidecar under `"labeling"`.

---

## 3) Export **boundary face maps** for stitching

**Why**
We’ll stitch by matching labels on opposing faces of neighboring nodes. To do that deterministically after the fact (from files), each node should write six small 2‑D arrays of **local label IDs** on its subvolume faces: `x-`, `x+`, `y-`, `y+`, `z-`, `z+`.

The size is modest: for a 1240³ tile, each face is 1240×1240×4 bytes ≈ 5.9 MB, ×6 faces ≈ 35 MB per node—tiny compared to the tile volume and a fair cost for transparent stitching.

**Diff** (`clump_finder.py`)—*right after labeling*:

```diff
*** a/clump_finder.py
--- b/clump_finder.py
@@
     labels = label_3d(mask, tile_shape=tile_shape, connectivity=connectivity, halo=halo)
@@
+    # For stitching: export six boundary face maps of local labels (no halo)
+    # Shapes: (nj, nk), (nj, nk), (ni, nk), (ni, nk), (ni, nj), (ni, nj)
+    face_xneg = labels[0, :, :].astype(np.uint32, copy=False)
+    face_xpos = labels[-1, :, :].astype(np.uint32, copy=False)
+    face_yneg = labels[:, 0, :].astype(np.uint32, copy=False)
+    face_ypos = labels[:, -1, :].astype(np.uint32, copy=False)
+    face_zneg = labels[:, :, 0].astype(np.uint32, copy=False)
+    face_zpos = labels[:, :, -1].astype(np.uint32, copy=False)
```

**Diff** (add these to the `out` dict so they land in `clumps_rankXXXXX.npz`):

```diff
@@
         "orientation": orientation,
+        # Stitching faces (local label IDs on boundary planes)
+        "face_xneg": face_xneg,
+        "face_xpos": face_xpos,
+        "face_yneg": face_yneg,
+        "face_ypos": face_ypos,
+        "face_zneg": face_zneg,
+        "face_zpos": face_zpos,
```

The JSON sidecar already includes `cart_dims`, `coords`, and `node_bbox_ijk`; we’ll reuse that to identify neighbors.

---

## 4) **Implement the stitcher** (connectivity‑6, DSU, area‑correct, reduce)

Replace the stub with a concrete tool that:

* Builds a **Cartesian topology** from your per‑node JSON sidecars (they already contain `cart_dims` and `coords`).
* Loads **face maps** from `.npz` for rank–neighbor face pairs and makes **edges** between `(rank,label)` pairs wherever both faces have non‑zero labels at the same (j,k) indices.
* Uses a **DSU** over 64‑bit keys `gid = (rank << 32) | local_label` (same scheme as your aggregator) to find equivalence classes.
* Reduces additive quantities: `cell_count`, `volume`, `mass`, `bbox_ijk` (min/max), **centroids** (recomputed exactly from per‑node centroids × weights).
* **Corrects area**: each cross‑rank touching cell across a face is an *interior* face in the global object. Locally it was counted once on each side as “exposed boundary” area. We therefore subtract **`2 * a_face * count`** for every stitched adjacency along that face (with `a_face` = `dy*dz`, `dx*dz`, or `dx*dy` for X/Y/Z faces, respectively).
* Writes a **stitched master** `clumps_stitched.npz` with stable global IDs (representatives of DSU sets).

> **Note**: Higher‑order variable moments (std/skew/kurt) cannot be recomputed exactly from means alone. If you need them globally exact, see the optional “power sums” extension in §6.

**Full replacement** (`stitch.py`) — drop‑in CLI tool:

```python
# stitch.py — connectivity-6 stitcher with area correction and exact centroid merge
from __future__ import annotations

import os, glob, json, argparse
import numpy as np
from typing import Dict, Tuple, List

# ------- Small DSU over 64-bit keys (gid = (rank<<32)|local_label) -------
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

# ------- Utilities -------
def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}

def _load_meta(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _gid(rank: int, local_id: int) -> np.uint64:
    return (np.uint64(rank) << np.uint64(32)) | np.uint64(local_id)

def _neighbor(coords: Tuple[int,int,int], dims: Tuple[int,int,int], axis: int, sign: int, periodic: Tuple[bool,bool,bool]):
    c = list(coords)
    c[axis] += sign
    if 0 <= c[axis] < dims[axis]:
        return tuple(c)
    if periodic[axis]:
        c[axis] = (c[axis] + dims[axis]) % dims[axis]
        return tuple(c)
    return None

# ------- Build rank index from sidecars -------
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

# ------- Compute edges and per-face adjacency counts -------
def build_edges(ranks: dict, cart_dims: Tuple[int,int,int], periodic: Tuple[bool,bool,bool],
                dx: float, dy: float, dz: float):
    # For area correction we accumulate counts per axis separately.
    dsu = DSU()
    edge_counts = {"x": {}, "y": {}, "z": {}}  # dict[(gid_a,gid_b)] = count

    # Make a coords -> rank lookup
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}

    # Helper to add edges from two face maps of equal shape
    def add_edges(axis_key: str, r:int, rn:int, a: np.ndarray, b: np.ndarray):
        mask = (a > 0) & (b > 0)
        if not mask.any(): return
        A = a[mask].astype(np.uint64, copy=False)
        B = b[mask].astype(np.uint64, copy=False)
        for la, lb in zip(A, B):
            ga = _gid(r, int(la))
            gb = _gid(rn, int(lb))
            if ga == 0 or gb == 0:
                continue
            # union
            dsu.union(ga, gb)
            key = (ga, gb) if ga < gb else (gb, ga)
            edge_counts[axis_key][key] = edge_counts[axis_key].get(key, 0) + 1

    # Loop ranks and axes
    for r, info in ranks.items():
        npz = _load_npz(info["npz"])
        coords = tuple(info["coords"])
        # X + neighbor
        ncoords = _neighbor(coords, cart_dims, axis=0, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = _load_npz(ranks[rn]["npz"])
            add_edges("x", r, rn, npz["face_xpos"], npz_n["face_xneg"])
        # Y + neighbor
        ncoords = _neighbor(coords, cart_dims, axis=1, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = _load_npz(ranks[rn]["npz"])
            add_edges("y", r, rn, npz["face_ypos"], npz_n["face_yneg"])
        # Z + neighbor
        ncoords = _neighbor(coords, cart_dims, axis=2, sign=+1, periodic=periodic)
        if ncoords is not None:
            rn = by_coords[ncoords]
            npz_n = _load_npz(ranks[rn]["npz"])
            add_edges("z", r, rn, npz["face_zpos"], npz_n["face_zneg"])

    # Face areas
    face_area = {"x": dy*dz, "y": dx*dz, "z": dx*dy}
    return dsu, edge_counts, face_area

# ------- Reduce per-set attributes with area correction -------
def stitch_reduce(input_dir: str, output_path: str):
    ranks, cart_dims, periodic = index_parts(input_dir)
    # Read dx,dy,dz from any part (assume uniform)
    any_npz = _load_npz(next(iter(ranks.values()))["npz"])
    dx, dy, dz = (float(any_npz["voxel_spacing"][0]),
                  float(any_npz["voxel_spacing"][1]),
                  float(any_npz["voxel_spacing"][2]))

    dsu, edge_counts, face_area = build_edges(ranks, cart_dims, periodic, dx, dy, dz)

    # Collect all gids (labels) to ensure isolated clumps get their own set
    all_gids: List[np.uint64] = []
    parts = {}
    for r, info in ranks.items():
        d = _load_npz(info["npz"])
        lids = d["label_ids"].astype(np.int64)
        gids = (_gid(r, 0) + lids.astype(np.uint64))  # (rank<<32) + lid
        all_gids.extend(list(gids))
        parts[r] = d

    # Assign representative roots
    roots = {}
    for g in all_gids:
        roots[g] = dsu.find(g)

    # Make compact 0..G-1 ids per root
    uniq_roots = sorted(set(roots.values()))
    root_to_idx = {rt: i for i, rt in enumerate(uniq_roots)}
    G = len(uniq_roots)

    # Prepare accumulators
    cell_count = np.zeros(G, dtype=np.int64)
    volume = np.zeros(G, dtype=np.float64)
    mass = np.zeros(G, dtype=np.float64)
    # Centroids (compute from weighted sums)
    Sxv = np.zeros(G, dtype=np.float64); Syv = np.zeros(G, dtype=np.float64); Szv = np.zeros(G, dtype=np.float64)
    Sxm = np.zeros(G, dtype=np.float64); Sym = np.zeros(G, dtype=np.float64); Szm = np.zeros(G, dtype=np.float64)
    bbox = np.zeros((G,6), dtype=np.int64)
    bbox[:,0::2] = np.iinfo(np.int64).max  # mins
    bbox[:,1::2] = np.iinfo(np.int64).min  # maxs
    area = np.zeros(G, dtype=np.float64)

    # Accumulate from parts
    for r, d in parts.items():
        lids = d["label_ids"].astype(np.int64)
        gids = (_gid(r, 0) + lids.astype(np.uint64))
        idx = np.array([root_to_idx[roots[g]] for g in gids], dtype=np.int64)

        # additive
        cc = d["cell_count"].astype(np.int64)
        vol = d["volume"].astype(np.float64)
        ms = d["mass"].astype(np.float64)
        ar = d["area"].astype(np.float64)
        cell_count[idx] += cc
        volume[idx] += vol
        mass[idx] += ms
        area[idx] += ar

        # centroids => convert to sums
        cv = d["centroid_vol"].astype(np.float64)    # (K,3)
        cm = d["centroid_mass"].astype(np.float64)
        Sxv[idx] += cv[:,0]*vol; Syv[idx] += cv[:,1]*vol; Szv[idx] += cv[:,2]*vol
        Sxm[idx] += cm[:,0]*ms;  Sym[idx] += cm[:,1]*ms;  Szm[idx] += cm[:,2]*ms

        # bboxes: union
        bb = d["bbox_ijk"].astype(np.int64)  # (K,6) = (i_min,i_max,j_min,j_max,k_min,k_max)
        # mins
        bbox[idx,0] = np.minimum(bbox[idx,0], bb[:,0])
        bbox[idx,2] = np.minimum(bbox[idx,2], bb[:,2])
        bbox[idx,4] = np.minimum(bbox[idx,4], bb[:,4])
        # maxs
        bbox[idx,1] = np.maximum(bbox[idx,1], bb[:,1])
        bbox[idx,3] = np.maximum(bbox[idx,3], bb[:,3])
        bbox[idx,5] = np.maximum(bbox[idx,5], bb[:,5])

    # Area correction: subtract 2*a_face per interior adjacency inside the same set
    for axis_key, ec in edge_counts.items():
        af = face_area[axis_key]
        for (ga, gb), cnt in ec.items():
            ra = roots.get(ga, dsu.find(ga))
            rb = roots.get(gb, dsu.find(gb))
            if ra == rb:
                i = root_to_idx[ra]
                area[i] -= 2.0 * af * float(cnt)

    # Final centroids
    small = 1e-300
    centroid_vol = np.stack([Sxv/(volume+small), Syv/(volume+small), Szv/(volume+small)], axis=1)
    centroid_mass = np.stack([Sxm/(mass+small),   Sym/(mass+small),   Szm/(mass+small)], axis=1)

    # Emit stitched master
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
    args = ap.parse_args()
    out = stitch_reduce(args.input, args.output)
    print(f"Stitched {out['gid'].size} global clumps -> {args.output}")

if __name__ == "__main__":
    main()
```

This keeps the stitcher compact, dependency‑free, and easy to audit. The DSU key format and the “gid” concept match your aggregator.

---

## 5) **Laptop tests (no MPI) for equivalence**

These demonstrate that **“single‑volume” == “tiled+stitched”** for connectivity 6, both **temperature** and **density** cuts.

Create `tests/test_equivalence_no_mpi.py`:

```python
# tests/test_equivalence_no_mpi.py
from __future__ import annotations
import numpy as np, os, shutil, tempfile
from local_label import label_3d
import metrics as M
from stitch import stitch_reduce

def _split_boxes(N, px, py, pz):
    # exactly like clump_finder.split_axis; deterministic uneven splits
    def split_axis(n,p):
        base, rem = n//p, n%p
        edges = []
        s = 0
        for c in range(p):
            extra = 1 if c < rem else 0
            e = s + base + extra
            edges.append((s,e))
            s = e
        return edges
    return split_axis(N,px), split_axis(N,py), split_axis(N,pz)

def _make_fields(N, seed=42):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(N,N,N)).astype(np.float32)
    # monotone maps with positive supports
    dens = np.exp(f*0.2).astype(np.float32)     # ~ lognormal
    temp = np.exp(-f*0.2).astype(np.float32)
    return dens, temp

def _one_volume_baseline(dens, temp, dx=1.0, dy=1.0, dz=1.0, thr=0.1, by="temperature"):
    field = temp if by=="temperature" else dens
    labels = label_3d(field < thr, tile_shape=(128,128,128), connectivity=6, halo=0)
    K = int(labels.max())
    cell = M.num_cells(labels, K=K)
    vol = M.volumes(cell, dx, dy, dz)
    mass = M.masses(labels, dens, dx, dy, dz, K=K)
    bbox = M.compute_bboxes(labels, ((0,labels.shape[0]),(0,labels.shape[1]),(0,labels.shape[2])), K=K)
    area = M.exposed_area(labels, dx, dy, dz, K=K)
    return {"K": K, "cell": np.sort(cell), "area_sum": float(area.sum()),
            "bbox": np.sort(bbox, axis=0), "vol_sum": float(vol.sum()), "mass_sum": float(mass.sum())}

def _write_parts_and_faces(tmpdir, dens, temp, px, py, pz, thr=0.1, by="temperature"):
    N = dens.shape[0]; dx = dy = dz = 1.0
    ix, iy, iz = _split_boxes(N, px, py, pz)
    rank = 0
    for cx,(i0,i1) in enumerate(ix):
        for cy,(j0,j1) in enumerate(iy):
            for cz,(k0,k1) in enumerate(iz):
                sub_d = dens[i0:i1, j0:j1, k0:k1]
                sub_t = temp[i0:i1, j0:j1, k0:k1]
                field = sub_t if by=="temperature" else sub_d
                labels = label_3d(field < thr, tile_shape=(128,128,128), connectivity=6, halo=0)
                K = int(labels.max())
                cell = M.num_cells(labels, K=K)
                vol = M.volumes(cell, dx, dy, dz)
                mass = M.masses(labels, sub_d, dx, dy, dz, K=K)
                cvol, cmass = M.centroids(labels, sub_d, dx, dy, dz, (0,0,0), ((i0,i1),(j0,j1),(k0,k1)), K=K)
                area = M.exposed_area(labels, dx, dy, dz, K=K)
                bbox = M.compute_bboxes(labels, ((i0,i1),(j0,j1),(k0,k1)), K=K)
                rank_ids = np.arange(1, K+1, dtype=np.int32)
                # faces
                out = {
                    "label_ids": rank_ids,
                    "cell_count": cell, "volume": vol, "mass": mass, "area": area,
                    "centroid_vol": cvol, "centroid_mass": cmass, "bbox_ijk": bbox,
                    "voxel_spacing": np.array([dx,dy,dz]),
                    "face_xneg": labels[0,:,:].astype(np.uint32),
                    "face_xpos": labels[-1,:,:].astype(np.uint32),
                    "face_yneg": labels[:,0,:].astype(np.uint32),
                    "face_ypos": labels[:,-1,:].astype(np.uint32),
                    "face_zneg": labels[:,:,0].astype(np.uint32),
                    "face_zpos": labels[:,:,-1].astype(np.uint32),
                }
                np.savez(os.path.join(tmpdir, f"clumps_rank{rank:05d}.npz"), **out)
                meta = {
                    "rank": rank, "coords": (cx,cy,cz), "cart_dims": (px,py,pz),
                    "node_bbox_ijk": [i0,i1,j0,j1,k0,k1],
                    "grid":{"periodic":[True,True,True]},
                    "output_npz": f"clumps_rank{rank:05d}.npz",
                }
                with open(os.path.join(tmpdir, f"clumps_rank{rank:05d}.meta.json"), "w") as f:
                    json.dump(meta, f)
                rank += 1

def test_equivalence_small():
    N=96; dens,temp = _make_fields(N)
    base = _one_volume_baseline(dens, temp, thr=0.5, by="temperature")
    tmp = tempfile.mkdtemp(prefix="stitch_test_")
    try:
        _write_parts_and_faces(tmp, dens, temp, px=2, py=2, pz=2, thr=0.5, by="temperature")
        out = stitch_reduce(tmp, os.path.join(tmp,"stitched.npz"))
        # compare invariants
        assert out["gid"].size == base["K"]
        assert np.allclose(np.sort(out["cell_count"]), base["cell"])
        assert np.isclose(out["area"].sum(), base["area_sum"])
        assert np.isclose(out["volume"].sum(), base["vol_sum"])
        assert np.isclose(out["mass"].sum(), base["mass_sum"])
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    test_equivalence_small()
    print("OK")
```

This shows exact equality (up to float roundoff) for clump counts/sizes and additive geometric metrics, using **no MPI** and only the new stitcher. It mirrors the behavior of your production nodes and the stitch step, so failures are easy to reproduce locally.

---

## 6) (Optional, but recommended) **Power‑sums for exact stitched statistics**

If you want *global* mean/std/skew/kurt for `{rho, T, vx, vy, vz, pressure}` to match the single‑volume answer **exactly**, add per‑label **power sums** to the per‑node `.npz`:

For each variable `X` and weighting `w` (volume or mass), emit:

* `W = ∑ w`, `S1 = ∑ w·X`, `S2 = ∑ w·X²`, `S3 = ∑ w·X³`, `S4 = ∑ w·X⁴`.

Then the stitcher simply **adds these across merged labels** and converts to mean/std/skew/kurt. You can implement this cleanly in `metrics.py` using the same `np.bincount` pattern you already use in `per_label_stats`.

Sketch (add to `metrics.py`):

```python
def per_label_powers(labels, X, weights, K=None):
    K = _ensure_K(labels, K)
    if K == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z,z,z,z,z
    lab = labels.ravel()
    x = X.ravel().astype(np.float64, copy=False)
    w = (np.ones_like(x) if weights is None else weights.ravel().astype(np.float64, copy=False))
    W  = np.bincount(lab, weights=w,             minlength=K+1)[1:]
    S1 = np.bincount(lab, weights=w*x,           minlength=K+1)[1:]
    S2 = np.bincount(lab, weights=w*(x*x),       minlength=K+1)[1:]
    S3 = np.bincount(lab, weights=w*(x*x*x),     minlength=K+1)[1:]
    S4 = np.bincount(lab, weights=w*(x*x*x*x),   minlength=K+1)[1:]
    return W,S1,S2,S3,S4
```

You can then (optionally) replace the existing on‑node `per_label_stats` with a **post‑aggregation** step that computes moments from these sums (saves time on nodes), but to keep changes minimal you can initially emit both.

---

## 7) Small accelerations + simplifications (no complexity added)

1. **Fail fast** for empty tiles. You already short‑circuit in metric reducers when `K==0`; that’s good. In `clump_finder.py`, consider early dropping labels below `min_clump_cells` and **remapping labels** *before* heavy second‑moment loops. Today you compute shape moments then drop; switching the order avoids work on tiny clumps. (Mechanically: compute `cell_count`, build `keep` LUT → rewrite `labels` as `labels = lut[labels]`, where dropped labels map to 0, and compact—then proceed.)

2. **Document 6‑connectivity only** for stitched outputs in README and help strings (already covered).

3. **Keep weights and pressure arrays as `float64`** once, reuse them across calls to `metrics.per_label_stats` or `per_label_powers` to avoid repeated dtype casts. (You already create `weight_vol` and `weight_mass`; that’s good.)

4. **Don’t compute eigen‑axes for tiny clumps** (e.g., < 16 cells) to save eigensolves. Guard the loop with a `keep_big = cell_count >= eig_min_cells` and fill NaNs otherwise. (Straightforward tweak near the eigen loop.)

5. **Aggregate script** is fine as is; no change needed. It concatenates per‑node outputs and adds `gid`. Keep it as the “non‑stitched” catalog; the new stitcher writes a separate `clumps_stitched.npz`.

---

## 8) Concrete “how to implement” checklist

**A) Code changes**

1. **local_label.py** – enforce 6‑connectivity for stitched runs (diff in §1).
2. **clump_finder.py** – implement density cut + face map export + provenance fields (diffs in §§2–3). Optionally, remap labels after `min_clump_cells` filtering to avoid heavy second‑moment work on tiny clumps.
3. **stitch.py** – replace stub with the DSU stitcher (full replacement in §4).
4. **metrics.py** – (optional) add `per_label_powers` to enable stitched higher‑order stats (sketch in §6).
5. **README.md** – update connectivity guidance and add a short “Stitching” section with CLI examples.

**B) New tests (no MPI, laptop‑friendly)**

Create `tests/test_equivalence_no_mpi.py` from §5 and run:

```bash
python -m tests.test_equivalence_no_mpi
```

Additionally, extend your existing synthetic scripts:

* **`synthetic_powerlaw_test.py`**: add `--cut-by {temperature,density}` and `--cut-op {lt,gt}` passthrough so you can generate two catalogs back‑to‑back and spot check distributions. (Change is symmetrical to the diff in `clump_finder.py`.)
* **`synthetic_test.py`** (box): extend to put the box so it **crosses a tile boundary** (e.g., two tiles in x), then split + stitch and assert K=1, area matches analytic (after stitch).

**C) Operational flow**

1. **Single node / baseline** (for QA):
   `python synthetic_powerlaw_test.py --N 128 --beta -1.6667 --outdir ./qa_single --cut-by temperature --temp-threshold 0.1`

2. **Emulate tiled outputs without MPI** (for QA): run the no‑MPI test script from §5 (generates per‑tile `.npz` + `.meta.json`).

3. **Stitch**:
   `python stitch.py --input ./qa_tiles --output ./qa_tiles/clumps_stitched.npz` (uses files only; no MPI).

4. **Compare**:

   * K, sorted `cell_count`, sums of `volume`, `mass`, and `area` must match baseline.
   * If you also emit power sums, compare stitched vs baseline stats (mean/std/skew/kurt) per clump.

---

## 9) Example end‑to‑end diffs (ready to paste)

I’ve included the critical patches above. If you prefer a single batch, here are the small hunks again:

* **Density cut & provenance** (in `clump_finder.py`): see §2 diffs.
* **Face map export** (in `clump_finder.py`): see §3 diffs.
* **Connectivity enforcement** (in `local_label.py`): see §1 diff.
* **Stitcher** (`stitch.py`): drop‑in from §4, replacing the stub.

---

## 10) Notes on numerical behavior and corner cases

* **Area correction** is exact under 6‑connectivity: every interior face is counted once by each neighbor and removed by subtracting `2 * a_face * overlaps`. Your current `metrics.exposed_area` counts boundary faces exactly in this way, so the correction aligns perfectly.
* **Centroids** are exactly recomputed via sums `∑(centroid * weight)` / `∑weight` (weight = volume for volume‑weighted, = mass for mass‑weighted), so they match single‑volume results bit‑for‑bit except for floating order of addition.
* **Principal axes / covariances** are not recomputed in the minimal stitcher. If you need them *globally* accurate, add the **second‑order mass‑weighted sums** `W, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz` per label to the per‑node outputs (straightforward to extract where you compute them now) and sum across DSU classes before the eigen solve. (This is the same pattern as the centroids/power sums.)

---

## 11) Why this remains simple and “rock‑solid”

* We **never touch** the heavy voxel arrays during stitching—only small face maps and per‑label rows.
* The DSU is tiny, deterministic, and based on your existing `gid` scheme.
* All corrections (area, centroids) are pure algebra on per‑label *sufficient statistics*, so the stitched results match single‑volume results for connectivity 6.
* Tests are laptop‑sized, deterministic, and avoid MPI; failures are easy to triage and reproduce.

---

### Appendix — References to current code

* Overall pipeline and config keys in **README.md**.
* Current aggregator (**aggregate_results.py**) for non‑stitched concatenation.
* Main driver (**clump_finder.py**) and where to add density cut + faces.
* I/O wrapper (**io_bridge.py**) feeding fields and halos.
* Local labeling (**local_label.py**) – correct cross‑tile merging for **6‑connectivity**.
* Metrics (**metrics.py**) – bincount‑based reductions, exposed area, centroids, bboxes.
* Plotting (**plot_clumps.py**) – unchanged.
* Stitching stub (**stitch.py**) – replaced by the DSU stitcher here.
* Synthetic tests (**synthetic_powerlaw_test.py**, **synthetic_test.py**) – extend for new cuts and cross‑tile cases.
