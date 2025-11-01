# Job Scripts

`jobs/slurm/frontier/` collects the site-tuned sbatch files. Each script wires in:
- the appropriate config from `configs/presets/` or `configs/runs/`
- shared helper calls to `clump_finder.py`, `scripts/analysis/aggregate_results.py`, and `scripts/analysis/plot_clumps.py`

Use `sbatch jobs/slurm/frontier/<script>.sbatch` from the repo root.
