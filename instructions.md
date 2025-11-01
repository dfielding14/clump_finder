# Clump Finder (Ultra-Scale, Node-Parallel) — Design & Implementation Guide

**Goal:** Identify connected components (“clumps”) of gas with temperature below a user-specified threshold in very large 3‑D hydro simulations. Compute geometry and statistics for each clump, save per‑node results as `.npz`, then assemble a master catalog. Baseline assumes **per‑node independence** (no cross‑node stitching), with a design note for future stitching.

**Key constraints & choices**

* **Scale:** Target up to ~10,000³ overall, but baseline runs one **~1,000³ subvolume per node** (as stated).
* **Parallelism:** One **MPI rank per node**, with **56 CPU threads** inside the rank for intra‑node acceleration (shared memory). (Alternative per‑core MPI mode is supported but optional.)
* **Connectivity:** Default **6‑connected** (face neighbors) to match “exterior faces” interpretation and to align with skimage’s `connectivity=1` in 3‑D.
* **Data I/O & units:** Use `clump_finder/io_bridge.py` (openPMD) to read datasets directly (based on your examples). It returns `dens,temp,velx,vely,velz` in code units; optional ghost zones are read from neighboring domain cells with periodic wrap at the global domain boundary (not across subvolume edges).
* **Outputs:** Per‑node `.npz` files + a master combined `.npz`. Per‑clump metrics include volume, mass, surface area (exposed faces), mass/volume‑weighted centroids, and descriptive stats (mean/std/skew/kurtosis) for `rho, T, vx, vy, vz, pressure` (with `pressure = rho * T` in your units).
* **Scheduler:** Provide a **SLURM job script** for Frontier (you will supply site examples; we include a clean template to adapt).

---

## 1) Repository Layout

High-level organization:

```
clump_find_drummond/
├── README.md
├── instructions.md
├── clump_finder.py                         # MPI entrypoint (driver)
├── io_bridge.py                            # openPMD I/O wrapper (code units; optional ghost zones)
├── local_label.py                          # node-local 3D connected components (Numba)
├── metrics.py                              # per-label reductions, centroids, shapes, area
├── stitch.py                               # (future) cross-node merge design + helpers (feature-flagged)
├── configs/
│   ├── base/                               # canonical defaults (e.g., configs/base/config.yaml)
│   ├── presets/                            # resolution presets (config_nXXXX.yaml)
│   └── runs/                               # sweep- or campaign-specific groups
├── jobs/
│   └── slurm/
│       └── frontier/                       # site-ready sbatch scripts
├── scripts/
│   └── analysis/                           # aggregation, PCA, correlations, plotting helpers
├── analysis/                               # long-lived analysis artifacts (logs/, PNGs, etc.)
├── clump_out/                              # run outputs (git-ignored)
├── logs/                                   # runtime logs (git-ignored)
└── requirements.txt
```

> **Important (updated):** `io_bridge.py` is implemented using `openpmd-api` directly (based on your provided snippets). It returns `dens`, `temp`, `velx`, `vely`, `velz` in code units. Dataset axis order is `[nz,ny,nx]` and outputs are `[i,j,k]` with `i→x`, `j→y`, `k→z`. Optional ghost zones overlap neighbor subvolumes and wrap at the global domain boundary only (no local subvolume wrap or clamping/zeros).

---

## 2) Configuration (configs/base/config.yaml)

Example keys (tune as needed; agent should implement a parser):

```yaml
# Input/output
dataset_path: /path/to/snapshot_or_brickset
output_dir: ./clump_out
log_dir: ./logs

# Selection
temperature_threshold: 100.0          # in your converted units
connectivity: 6                        # 6 or 18 or 26; default 6

# Geometry
# Grids are cubic. User supplies Nres (e.g., 640, 1280, 2560, 5120, 10240).
# Domain length L=1, so dx=dy=dz=L/Nres (auto if left null).
Nres: 2560
dx: null
dy: null
dz: null
origin: [0.0, 0.0, 0.0]                # global origin for cell centers
periodic: [true, true, true]           # all axes periodic

# Parallelism
intra_node_threads: 56                 # per your node
mpi_mode: "one_rank_per_node"          # alt: "many_ranks_per_node"
verify_one_rank_per_node: true         # assert local shared-memory size == 1

# Tiling inside a node (to keep caches hot; not the MPI split)
tile_shape: [128, 128, 128]            # internal bricks for labeling (auto-adjust at edges)

# Memory / precision
field_dtype: float32                   # load fields as float32 to reduce memory
accum_dtype: float64                   # accumulations in float64 for numerical stability

# Output content
save_pressure: false                   # compute pressure on-the-fly as rho*T; don't store field
save_per_clump_voxel_bbox: true       # store i/j/k min/max per clump (helps future stitching)

# Future stitching
ghost_width: 1                         # include 1-cell ghost zones; wrap at global domain edges only
stitch_flag: false                     # baseline off (no cross-node merging)

# Misc
profile: false
```

---

## 3) Data Access

In `io_bridge.py` (implemented; based on your provided readers and openPMD usage):

* Provide a **single function**:

```python
def load_subvolume(node_bbox, cfg):
    """
    node_bbox: ((i0, i1), (j0, j1), (k0, k1)) in global indices (exclusive upper bounds).
    Returns dict of NumPy arrays (C-order) with dtype per config:
      {
        'dens': (ni, nj, nk),
        'temp': (ni, nj, nk),
        'velx': (ni, nj, nk),
        'vely': (ni, nj, nk),
        'velz': (ni, nj, nk),
        # Aliases also provided: rho->dens, T->temp, vx->velx, vy->vely, vz->velz
      }
    Values are in code units. If cfg.ghost_width > 0, include ghost zones (1 cell) per face
    on all faces with periodic wrap-around.
    """
```

* Keep memory in check: **load only the subvolume assigned to this node**, plus optional **ghost zones** (1 cell per face if `ghost_width=1`), filled from neighboring domain cells (wrap only at global edges).
* All outputs are in code units (no external unit conversion here).

---

## 4) Parallelization Strategy

### Baseline (recommended)

* **MPI:** 1 rank **per node** (use `--ntasks-per-node=1`). Optionally assert at runtime that `comm.Split_type(MPI.COMM_TYPE_SHARED).Get_size() == 1` (see `verify_one_rank_per_node`).
* **Threads:** Inside the rank, use **Numba** (`njit(parallel=True)`) to fan out to **56 CPU threads**.

  * Set `NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
  * Also set `OMP_NUM_THREADS` to the same to avoid oversubscription by any BLAS/OpenMP dependencies.
* Rationale: eliminates Python shared-memory complexity while giving full node throughput.

### Alternative (optional)

* Many MPI ranks per node (e.g., 56). Use **MPI shared memory windows** for the label array and implement a coordinated brick merge. Higher complexity; keep behind a flag.

---

## 5) Algorithm Overview

### 5.1 Thresholding

Create a boolean mask `M = (T < T_threshold)`.

### 5.2 Local 3‑D Connected Components (within a node)

Implement a **block‑wise union‑find** CCL tailored for huge volumes:

1. **Partition** the subvolume into 3‑D **tiles** (e.g., 128³).
2. For each tile (in parallel with Numba):

   * Run a fast **two‑pass CCL** (6‑connected).
   * Produce **provisional labels** local to the tile (start labels at 1; 0 is background).
3. **Merge across tile boundaries:** For each pair of face‑adjacent tiles, examine the 1‑cell shared face; build equivalence pairs `(label_A, label_B)` for touching components; feed to a **union‑find** with path compression (Numba or vectorized).
4. **Relabel** the entire node array with the flattened representatives so all clumps are uniquely labeled **within the node**.

   * Ensure labels are **compacted** to `1..K` to keep later reductions cheap.
   * Use `uint32` labels (more than enough per node).

> **Note:** This behaves like `skimage.measure.label(..., connectivity=1)` but is **deeply parallel and tiled**.

### 5.3 Per‑Clump Reductions (geometry & stats)

Let `labels` be the final node‑local label array (`uint32`) and `K = labels.max()`.

**Cell geometry**

* Cell centers:
  `x = x0 + (i + 0.5) * dx`, `y = y0 + (j + 0.5) * dy`, `z = z0 + (k + 0.5) * dz` (use `origin` from config and local/global index mapping).
* Cell volume: `Vc = dx * dy * dz`. (Support anisotropic spacings.)

**Mass & volume**

* `num_cells[ℓ] = bincount(labels)` excluding 0.
* `V[ℓ] = num_cells[ℓ] * Vc`.
* `M[ℓ] = Σ (rho * Vc)` grouped by label.

**Centroids**

* **Volume‑weighted:**
  `r̄_vol = (Σ r * Vc) / V`
* **Mass‑weighted:**
  `r̄_mass = (Σ r * rho * Vc) / M`
* Do this per axis using grouped sums (`np.bincount` or a Numba scatter‑add), with `float64` accumulators.

**Surface area (exposed faces only)**

* For each axis (`x`, `y`, `z`), count faces where a clump **touches background**:

  * Example along `+x`: `exposed_px = (labels > 0) & (shift(labels, -1, axis=0) == 0)`
    Along `-x`: `exposed_mx = (labels > 0) & (shift(labels, +1, axis=0) == 0)`
  * Similarly for `y` and `z`.
* Per label, sum:

  * `A_x = (exposed_px + exposed_mx) * (dy * dz)`
  * `A_y = (exposed_py + exposed_my) * (dx * dz)`
  * `A_z = (exposed_pz + exposed_mz) * (dx * dy)`
* Total area: `A = A_x + A_y + A_z`.
* (This definition ignores **clump–clump** interfaces; only **exterior** surfaces count, as requested.)

**Descriptive stats for state variables**
Compute **mean, std, skewness, kurtosis** for each of:
`rho, T, vx, vy, vz, pressure` (with `pressure = rho*T`, computed on the fly).

* Use numerically stable grouped reductions. Two options:

  1. **Sufficient statistics** per label via grouped sums of powers

     * For variable `X`, compute per label:
       `S1 = Σ X,  S2 = Σ X²,  S3 = Σ X³,  S4 = Σ X⁴`
       Then derive mean/var/skew/kurt.
     * Use `float64` accumulators; avoid overflow.
     * For **weighting**, compute both **volume‑weighted** (each cell weight = `Vc`) and **mass‑weighted** (weights `w = rho * Vc`).
     * Save both sets by default (mass‑weighted arrays have `_massw` suffix).

  2. **Welford/Kahan** streaming with Numba in tiles (also fine; a bit more code).

**Shape (principal axes)**

* For each clump, form the **mass‑weighted** covariance matrix of positions:

  * `μ = r̄_mass`, `w = rho * Vc`
  * `C = (Σ w * (r - μ)(r - μ)^T) / Σ w`
* Eigen‑decompose `C → (λ₁ ≥ λ₂ ≥ λ₃, e⃗₁,e⃗₂,e⃗₃)`.
* Report and save:

  * `principal_axes_lengths` `[K,3]` (RMS extents; a=√λ₁, b=√λ₂, c=√λ₃)
  * `axis_ratios` `[K,2]` (q=b/a, s=c/a)
  * `orientation` `[K,3,3]` (eigenvectors)
* This gives a clean ellipsoidal summary (near-equivalent to inertia‑tensor principal axes for uniform density, but mass‑weighted is more physically meaningful here).

**Bounding boxes**

* Save integer index mins/maxes `(i_min..i_max, j_min..j_max, k_min..k_max)` per clump using inclusive mins and exclusive maxes. Aids future stitching.

---

## 6) Output Formats

### Per‑node `.npz` (`clumps_rank%05d.npz`)

Store compact arrays keyed by label id (length `K` each). Suggested schema:

* `label_ids` — `int32` `[K]` (1..K)
* `num_cells` — `int64` `[K]` (also saved as `cell_count` for clarity)
* `volume` — `float64` `[K]`
* `mass` — `float64` `[K]`
* `area` — `float64` `[K]`
* `centroid_vol` — `float64` `[K,3]`   (x,y,z)
* `centroid_mass` — `float64` `[K,3]`
* For each var in `{rho,T,vx,vy,vz,pressure}` (volume‑weighted):

  * `<var>_mean`, `<var>_std`, `<var>_skew`, `<var>_kurt` — `float64` `[K]`
* Mass‑weighted analogs with `_massw` suffix (included by default).
* `bbox_ijk` — `int32` `[K,6]`  (i_min,i_max,j_min,j_max,k_min,k_max) using [min,max) in **global** indices.
* `principal_axes_lengths` — `float64` `[K,3]`
* `axis_ratios` — `float64` `[K,2]`
* `orientation` — `float64` `[K,3,3]`
* `voxel_spacing` — `float64` `[3]`  (dx,dy,dz) where `dx=dy=dz=1/Nres`
* `origin` — `float64` `[3]`
* `connectivity` — `int32` scalar
* `temperature_threshold` — `float64` scalar
* `rank` — `int32` scalar
* `node_bbox_ijk` — `int64` `[6]`   (global subvolume this node processed)
* `periodic` — `bool` `[3]`

> Keep arrays dense; do **not** pickle Python dicts in `.npz`.

### Master `.npz` (`clumps_master.npz`)

* Concatenate per‑node arrays with a **global clump id**:

  * `gid = (rank << 32) | local_label`  (store as `uint64`)
* Same schema as above (including shape metrics `principal_axes_lengths`, `axis_ratios`, `orientation`) but with `gid` and `rank` retained; **no deduplication** across nodes in baseline (stitching disabled).

---

## 7) Frontier SLURM Template (to adapt)

`jobs/slurm/frontier/frontier_clump.sbatch`:

```bash
#!/bin/bash
#SBATCH -J clumps
#SBATCH -A <your_project>
#SBATCH -N <nodes>                      # total nodes
#SBATCH --ntasks-per-node=1             # one MPI rank per node
#SBATCH --cpus-per-task=56              # threads inside rank
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# -- Load your site modules here (you will supply concrete examples) --
# module restore
# module load cray-python
# source /path/to/your/venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

CONF=${CONF:-configs/base/config.yaml}

srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} \
     python -u clump_finder.py --config "${CONF}"

# optional separate aggregation step if you prefer:
# if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
#     python -u scripts/analysis/aggregate_results.py --input ./clump_out --output ./clump_out/clumps_master.npz
# fi
```

---

## 8) Driver Flow (`clump_finder.py`)

**MPI rank 0** computes a 3‑D node decomposition (`node_bbox_ijk`) across the global grid using a near‑cubic split: use `MPI.Dims_create(comm.size, 3)` to obtain `(px,py,pz)` and a periodic Cartesian communicator (`Create_cart(dims=(px,py,pz), periods=(True,True,True))`). Broadcast per‑rank bboxes.

Per rank (node):

1. **Load subvolume** via `io_bridge.load_subvolume(node_bbox, cfg)` (optionally with ghost zones; ghost cells overlap with neighbors and wrap at global domain edges only). Returns `dens,temp,velx,vely,velz` in code units.
2. **Threshold**: `M = T < threshold`.
3. **Tile** the subvolume (internal bricks).
4. **Local labeling** (`local_label.label_3d`): returns compacted `labels` with `K` clumps.
5. **Compute metrics** (`metrics.*`):

   * `volume`, `mass`, `area`, `centroid_vol`, `centroid_mass`
   * stats for `rho, T, vx, vy, vz, pressure` (pressure computed on the fly) — both volume‑ and mass‑weighted by default
   * `bbox_ijk` (global; [min,max) per axis), axis lengths/ratios/orientations (saved as dedicated arrays)
6. **Write per‑node `.npz`** (`clumps_rank%05d.npz`).
7. Optional: Rank 0 runs `scripts/analysis/aggregate_results.py` to build `clumps_master.npz`.

**CLI** (implement with `argparse`):

```
python clump_finder.py --config configs/base/config.yaml [--dry-run] [--profile]
```

---

## 9) Implementation Details & Pseudocode

### 9.1 Tiled 3‑D CCL (Numba)

* **Two‑pass per tile:** scan forward assigning provisional labels (checking previously visited neighbors in 3‑D: `(-1,0,0),(0,-1,0),(0,0,-1)` for 6‑connectivity on a raster order) and recording equivalences; second pass resolves to minimal representative per tile.
* **Global merge:** examine tile faces; where both sides are foreground and labels differ, union them.

**Sketch** (`local_label.py`):

```python
# Numba-friendly UF
@njit
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit
def uf_union(parent, a, b):
    ra, rb = uf_find(parent, a), uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra

def label_3d(mask, tile_shape=(128,128,128)):
    """
    mask: boolean 3-D array (C-order), ghost zones included if present.
    Returns labels (uint32) compacted to 1..K on the **interior** (exclude ghost zones in output).
    """
    # 1) split into tiles
    # 2) Numba-parallel first pass per tile with local uf
    # 3) collect boundary equivalences, then global uf merge
    # 4) second pass: remap to representatives
    # 5) compact labels -> 1..K (using unique + inverse or a dictionary)
    # 6) drop ghost zones and return labels
    ...
```

> **Note:** Keep intermediate arrays `int32`/`uint32`. Avoid Python objects in Numba regions.

### 9.2 Grouped Reductions (`metrics.py`)

* Avoid Python loops over clumps. Use:

  * `np.bincount(idx, weights=...)` for `S1,S2,S3,S4`.
  * Or Numba parallel scatter‑adds over tiles (larger code, similar speed).
* Use `float64` accumulators (`accum_dtype`) for numerical stability.

**Sketch:**

```python
def per_label_stats(labels, X, weights=None, K=None, accum_dtype=np.float64):
    """
    Compute mean/std/skew/kurt per label for variable X.
    weights=None => volume-weighted by uniform cell volume
    """
    if K is None: K = labels.max()
    lab = labels.ravel()
    x = X.ravel().astype(accum_dtype, copy=False)

    if weights is None:
        w = np.ones_like(x, dtype=accum_dtype)
    else:
        w = weights.ravel().astype(accum_dtype, copy=False)

    W = np.bincount(lab, weights=w, minlength=K+1)[1:]  # drop background
    S1 = np.bincount(lab, weights=w*x, minlength=K+1)[1:]
    mu = S1 / W

    # central moments via Σ w*(x-mu)^p
    xc = x - mu[lab-1]
    S2c = np.bincount(lab, weights=w*xc*xc, minlength=K+1)[1:]
    S3c = np.bincount(lab, weights=w*xc*xc*xc, minlength=K+1)[1:]
    S4c = np.bincount(lab, weights=w*xc*xc*xc*xc, minlength=K+1)[1:]

    var = S2c / W
    std = np.sqrt(var)
    skew = (S3c / W) / (std**3 + 1e-30)
    kurt = (S4c / W) / (var**2 + 1e-30)  # excess not subtracted; decide & document
    return mu, std, skew, kurt
```

> Decide whether to report **Fisher excess kurtosis** (`kurt-3`). Document your choice; either is fine if consistent.

### 9.3 Surface Area

Compute **exposed faces only** (neighbor label = 0). Use shifts and `bincount`:

```python
def exposed_area(labels, dx, dy, dz):
    lab = labels
    K = lab.max()
    face = np.zeros((K+1,), dtype=np.float64)  # accumulate per label

    # X faces
    a_x = dy*dz
    mask = (lab > 0)
    ex = mask & (np.roll(lab, -1, axis=0) == 0)
    em = mask & (np.roll(lab, +1, axis=0) == 0)
    face += a_x * np.bincount(lab[ex], minlength=K+1)
    face += a_x * np.bincount(lab[em], minlength=K+1)

    # Y faces
    a_y = dx*dz
    ey = mask & (np.roll(lab, -1, axis=1) == 0)
    emy= mask & (np.roll(lab, +1, axis=1) == 0)
    face += a_y * np.bincount(lab[ey], minlength=K+1)
    face += a_y * np.bincount(lab[emy], minlength=K+1)

    # Z faces
    a_z = dx*dy
    ez = mask & (np.roll(lab, -1, axis=2) == 0)
    emz= mask & (np.roll(lab, +1, axis=2) == 0)
    face += a_z * np.bincount(lab[ez], minlength=K+1)
    face += a_z * np.bincount(lab[emz], minlength=K+1)

    return face[1:]  # drop background
```

### 9.4 Shape (principal axes)

* Gather per‑label mass‑weighted first and second moments of coordinates using the same grouped strategy. Build covariance matrices and eigendecompose per clump (vectorized over `K` if memory permits; otherwise loop per clump—`K` is typically much smaller than number of cells).

---

## 10) Aggregation (`scripts/analysis/aggregate_results.py`)

* Scan `output_dir` for `clumps_rank*.npz`.
* For each, build a **global id** `gid = (rank << 32) | label_id`.
* Concatenate arrays; write `clumps_master.npz`.
* Include a small JSON sidecar (optional) with run metadata and config snapshot for provenance.

---

## 11) Performance & Memory Notes

* **Memory math (uniform grid):**
  `N = 1000³ = 1e9` cells per node (your assumption).
  One `float32` field: ~4.0 GB. Six fields (rho,T,vx,vy,vz,pressure) would be ~24 GB;
  but **compute pressure on the fly** → ~20 GB plus:

  * `labels` (`uint32`): ~4 GB
  * Working arrays/equivalences: a few GB
    **Total:** ~25–30 GB typical (headroom recommended). If memory tight:
  * Load fields **one‑at‑a‑time** for reductions (two‑pass after labels).
  * Use tiles to limit resident working sets.
* **Threading:** Ensure only **one** thread team is active (Numba). Set BLAS threads to 1.
* **I/O:** Prefer contiguous reads; if data are chunked on disk, align node subvolumes to storage chunks.

---

## 12) Validation & Sanity Checks

* Small synthetic volumes (e.g., 128³) with known clumps to validate:

  * Label counts vs. `skimage.label` (on tiny cases).
  * Area for a single axis‑aligned box: should match analytical `2*(ab+bc+ca)`.
  * Mass vs. direct sum; centroids vs. geometry center.
* Check that per‑node counts sum to expected background/foreground cell totals.

---

## 13) Future Work: Cross‑Node Stitching (Design Note)

Not required for baseline, but prepare for later:

1. **Ghost zone strategy:** Read **1‑cell ghost zones** at node boundaries. Keep a mapping from **interior labels** to any touching **ghost labels** along each of the 6 faces.
2. **Boundary adjacency export:** For each face that abuts a neighbor node, write `(rank, local_label, face_id, face_coords)` pairs where the clump touches the boundary.
3. **Distributed union‑find:** Rank 0 collects adjacency edges `(rankA,labelA) ~ (rankB,labelB)`, computes global components (DSU), broadcasts a **global relabel map**.
4. **Relabel & re‑reduce:** Either (a) re‑scan nodes and re‑reduce using global ids, or (b) reduce locally and then **merge clump rows** that share a global id (merging stats additively using sufficient statistics).
5. **Periodicity:** If `periodic[axis]=true`, also stitch across the domain’s periodic faces.

Keep a `stitch_flag` to toggle this path later.

---

## 14) Minimal Requirements (likely already in your venv)

* `numpy` (≥1.23)
* `scipy` (optional; only if you prefer for eigen/ stats; `numpy.linalg.eigh` is fine)
* `numba` (for parallel loops)
* `mpi4py`
* `pyyaml` (for config)
* `tqdm` (optional, progress)
* **(Do not add heavy deps; reuse your existing I/O stack.)**

`requirements.txt` (if needed):

```
numpy
numba
mpi4py
pyyaml
tqdm
```

---

## 15) Command Examples

```bash
# Dry run (discover geometry; no compute)
python clump_finder.py --config configs/base/config.yaml --dry-run

# Production (via SLURM template above)
sbatch jobs/slurm/frontier/frontier_clump.sbatch

# Aggregate later (if not done in-job)
python scripts/analysis/aggregate_results.py --input ./clump_out --output ./clump_out/clumps_master.npz

# Plot clump size and size-vs-velocity dispersion (from a .npz)
python scripts/analysis/plot_clumps.py --input ./clump_out/clumps_master.npz --outdir ./clump_out

# End-to-end: compute then auto-aggregate+plot (single command)
python clump_finder.py --config configs/base/config.yaml --auto-aggregate-plot
```

---

## 16) Edge Cases & Choices (documented defaults)

* **Connectivity:** default 6; also support 18/26 (area def. changes with connectivity).
* **Kurtosis:** report **Pearson** by default; `--excess-kurtosis` switches to excess.
* **Pressure:** computed as `rho*T` in code units (no extra field I/O).
* **Background label:** `0`. Clumps labeled `1..K`.
* **Centroids:** both **volume‑** and **mass‑weighted** are saved by default.
* **Units:** all stats in **code units** (I/O returns code units).

---

## 17) Extras

* Include a synthetic 128³ test generator to validate labeling/area/centroids on tiny cases.
* Provide a stub `stitch.py` with `stitch_flag` and clear TODOs for future cross‑node merging.

---

## 18) Plotting

`scripts/analysis/plot_clumps.py` creates a multi-page PDF with:

- Histogram of clump sizes (`cell_count` by default; `--use-volume` for volume).
- 2D histograms of size versus velocity dispersion (`sqrt(vx_std²+vy_std²+vz_std²)`) and per-component stds.

Examples:

```bash
python scripts/analysis/plot_clumps.py --input ./clump_out/clumps_rank00000.npz
python scripts/analysis/plot_clumps.py --input ./clump_out/clumps_master.npz --use-volume --mass-weighted
```

You can also run the driver with `--auto-aggregate-plot` to automatically aggregate and emit a PDF on rank 0.

---

## 19) Synthetic Power-Law Test (128³)

`synthetic_powerlaw_test.py` generates 3-D fields for `dens, temp, velx, vely, velz` with an isotropic power spectrum `P(k) ~ k^β` (default `β=-5/3`), thresholds on `temp` (default 30th percentile), runs the clump finder (single-rank), writes a per-node `.npz`, and produces a PDF of clump sizes and size-vs-velocity dispersion.

Usage:

```bash
python synthetic_powerlaw_test.py --N 128 --beta -1.6667 --temp-quantile 0.3 --outdir ./clump_out_synth
```

Outputs:

- `clump_out_synth/clumps_rank00000.npz`
- `clump_out_synth/clumps_synth_plots.pdf`

---

### Done

This spec keeps the implementation **focused, scalable, and minimal**: one MPI rank per node, Numba‑accelerated tiled CCL, grouped reductions for metrics, clean `.npz` outputs, and a straightforward SLURM template. It leaves hooks for **future cross‑node stitching** without complicating the baseline.
