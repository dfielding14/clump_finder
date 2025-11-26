# Job Scripts

`jobs/slurm/frontier/` collects the site-tuned sbatch files. Each script wires in:
- the appropriate config from `configs/presets/` or `configs/runs/`
- shared helper calls to `clump_finder.py`, `scripts/analysis/aggregate_results.py`, and `scripts/analysis/plot_clumps.py`

Use `sbatch jobs/slurm/frontier/<script>.sbatch` from the repo root.

- `frontier_clump.sbatch` is the default stitching smoke test. It:
  1. Exercises the synthetic equivalence harness (`tests/test_equivalence_no_mpi.py`).
  2. Launches a 4-node run of the n1280 dataset (`configs/config_n1280.yaml`) to generate rank-local clump catalogs.
  3. Aggregates rank outputs, stitches them with `stitch.py --input <dir> --output clumps_stitched.npz`, re-plots the master catalog (including an overlaid stitched-vs-unstitched size histogram with ratio panel), and renders stitched diagnostics under `clump_out/clumpn1280_step35/stitched_plots/`.
