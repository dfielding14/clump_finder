# Clump Finder (Node-Parallel, Numba + MPI)

Identify connected cold clumps (T < threshold) in large 3‑D hydro data, compute per‑clump geometry and statistics, write per‑node `.npz`, and aggregate to a master catalog. Designed for one MPI rank per node with ~56 CPU threads via Numba.

## Quick Start

- Install deps (site modules or pip):
  - See `clump_finder/requirements.txt`

- Configure `clump_finder/config.yaml`:
  - Set `dataset_path`, `step`, `Nres`, `temperature_threshold`.
  - Optional: `assert_nres_from_data: true` to validate/override `Nres` from the dataset shape.

- Dry run (no I/O):
```
python clump_finder.py --config config.yaml --dry-run
```

- Run (single node):
```
python clump_finder.py --config config.yaml --auto-aggregate-plot
```

- Aggregate later and plot:
```
python aggregate_results.py --input ./clump_out --output ./clump_out/clumps_master.npz --sidecar ./clump_out/master_meta.json
python plot_clumps.py --input ./clump_out/clumps_master.npz --outdir ./clump_out
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
sbatch slurm/frontier_clump.sbatch
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
- emit_halo: true (periodic wrap halo)
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

