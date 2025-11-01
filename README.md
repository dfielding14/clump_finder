# Clump Finder (Node-Parallel, Numba + MPI)

Identify connected cold clumps (T < threshold) in large 3‑D hydro data, compute per‑clump geometry and statistics, write per‑node `.npz`, and aggregate to a master catalog. Designed for one MPI rank per node with ~56 CPU threads via Numba.

## Conceptual Guide

The driver follows a simple but memory‑aware pipeline so we can work on Frontier’s CPU nodes without GPU dependencies:

1. **Domain decomposition.** We launch exactly one MPI rank per node. `tile_shape` chooses the cell block a rank processes locally. For uniform grids we set tiles to the largest cube that fits in memory (e.g., 640³, 1280³, etc.); higher resolutions split the domain into equal tiles. This keeps each rank’s working set predictable and allows us to scale by adding nodes.

2. **I/O and halos.** `io_bridge.py` reads the OpenPMD snapshot directly into float32 arrays (`dens`, `temp`, `vel*`) plus an optional 1‑cell ghost layer so that connected components at tile boundaries see their immediate neighbors. We keep ghost handling simple—only global periodic wrap, no cross‑tile stitching—so each rank can proceed independently.

3. **Labeling.** `local_label.py` runs a Numba‑accelerated 3‑D connected components on the full tile (including halo) using the configured connectivity (6, 18, or 26 neighbors). Connectivity is the main science lever: tighter (6) keeps clumps compact, wider (26) fuses tenuous bridges.

4. **Filtering.** We drop labels smaller than `min_clump_cells` (default 4³). This avoids filling the catalog with noise while keeping the main statistics stable.

5. **Metrics and statistics.** `metrics.py` reduces per‑label quantities: cell counts, volumes/masses, exposed surface area, centroids, velocity statistics, and principal axes. We ingest data as float32 to save RAM, but accumulations (weights, covariances) stay in float64 to preserve accuracy on large tiles.

6. **Persistence.** Each rank writes one `.npz` plus a JSON meta file (bounding box, timings, config snapshot). An optional master step (`scripts/analysis/aggregate_results.py`) concatenates per-rank outputs and writes a global catalog.

7. **Visualization.** `plot_clumps.py` generates quick‑look PNGs: size histogram, size vs velocity dispersion, and a diagnostic of surface area versus volume (`area / volume^{2/3}`).

Key design choices:

- **Single rank per node** to match the clump finder’s shared‑memory structure and Frontier’s CPU layout (56 cores). Thread counts are wired to SLURM environment variables.
- **Large tiles over small ones.** Memory scaling is the dominant cost. Matching each rank’s tile to the node’s memory limits (e.g., 1280³ fits comfortably within 512 GB) yields the best wall‑clock time without the overhead of stitching.
- **Float32 inputs, float64 accumulators.** This balances memory pressure and numerical stability.
- **Explicit SLURM templates per resolution.** Each dataset gets its own config and sbatch file so we can set the right tile size, node count, and queueing parameters without editing shared files.

## Quick Start

- Install deps (site modules or pip):
  - See `requirements.txt`

- Configure `configs/base/config.yaml`:
  - Set `dataset_path`, `step`, `Nres`, `temperature_threshold`.
  - Optional: set `assert_nres_from_data: true` to validate/override `Nres` from the dataset shape.

- Dry run (no I/O):
```
python clump_finder.py --config configs/base/config.yaml --dry-run
```

- Run (single node):
```
python clump_finder.py --config configs/base/config.yaml --auto-aggregate-plot
```

- Aggregate later and plot:
```
python scripts/analysis/aggregate_results.py --input ./clump_out --output ./clump_out/clumps_master.npz --sidecar ./clump_out/master_meta.json
python scripts/analysis/plot_clumps.py --input ./clump_out/clumps_master.npz --outdir ./clump_out
```

- Synthetic 128³ power-law test (no I/O):
```
python synthetic_powerlaw_test.py --N 128 --beta -1.6667 --outdir ./clump_out_synth
```

## Outputs

Per‑node `.npz` contains:
- label_ids, num_cells/cell_count, volume, mass, area
- centroid_vol, centroid_mass
- stats for rho, T, vx, vy, vz, pressure (both volume- and mass-weighted)
- bbox_ijk (global [min,max) per axis)
- principal_axes_lengths, axis_ratios, orientation
- rank, node_bbox_ijk, voxel_spacing, origin, connectivity, temperature_threshold
- Sidecar per-rank: `clumps_rankXXXXX.meta.json` with config snapshot, timings, git commit

Master `.npz` (`clumps_master.npz`) contains concatenated arrays plus `gid` and `rank`.
Sidecar (optional): JSON with part list and clump count.

## Frontier (ORNL) Notes

- Use one rank per node:
```
sbatch jobs/slurm/frontier/frontier_clump.sbatch
```
- Edit the SLURM script to load your site modules or virtual env.
- Environment hints:
  - `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`
  - `NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK`
  - `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`

## Configuration Keys (highlights)

- dataset_path, step
- output_dir, log_dir
- Nres (cubic grid), origin (default [0,0,0])
- periodic: [true,true,true]
- temperature_threshold (code units)
- connectivity: 6|18|26 (default: 6)
- tile_shape: [128,128,128]
- ghost_width: 1 (include ghost zones from neighbors; wrap at global domain edges only)
- field_dtype: float32, accum_dtype: float64
- assert_nres_from_data: true|false
- verify_one_rank_per_node: true|false

## Plotting

- Size histogram (log–log) and size vs velocity dispersion (log–log, LogNorm) PNGs.
- For cell_count, size bins use integer-rounded geometric edges; for volume, log-spaced bins.

## Notes

- Pressure is computed on the fly as rho*T (code units).
- Memory at scale: label array ~4 bytes/cell; consider streaming stats for very large runs.
- Future: cross-node stitching is stubbed in `stitch.py`.
