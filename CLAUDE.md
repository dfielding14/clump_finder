# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A node-parallel 3D connected component labeler for identifying cold clumps in large hydrodynamic simulations. Uses MPI (one rank per node) with Numba multi-threaded labeling. Designed for Frontier's CPU nodes (56 cores each, no GPU).

## Key Commands

```bash
# Run clump finder (single node)
python clump_finder.py --config configs/base/config.yaml

# Run with extended statistics (shape diagnostics, thermodynamic moments)
python clump_finder.py --config <config.yaml> --extra-stats

# Aggregate per-rank outputs into master catalog
python scripts/analysis/aggregate_results.py \
    --input ./clump_out --output ./clump_out/clumps_master.npz \
    --sidecar ./clump_out/master_meta.json

# Stitch across node boundaries (6-connectivity only)
python stitch.py --input ./clump_out --output ./clump_out/clumps_stitched.npz

# Generate diagnostic plots
python scripts/analysis/plot_clumps.py --input <npz> --outdir <dir>

# Compare unstitched vs stitched
python scripts/analysis/plot_clumps.py --input <master.npz> --compare <stitched.npz> \
    --compare-labels "Unstitched" "Stitched"

# Run equivalence test (no MPI, synthetic data)
python tests/test_equivalence_no_mpi.py --N 96 --px 2 --py 2 --pz 1

# Synthetic power-law test
python synthetic_powerlaw_test.py --N 128 --beta -1.6667 --outdir ./clump_out_synth
```

## Frontier SLURM Usage

```bash
# Submit production job (adjusts CONF env var for different configs)
CONF=configs/config_n1280.yaml sbatch jobs/slurm/frontier/frontier_clump.sbatch

# Required environment settings (set in sbatch templates):
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Architecture

### Data Flow
1. **I/O** (`io_bridge.py`) - Reads openPMD/ADIOS2 snapshots, computes temperature/velocity from conservative variables, handles periodic ghost zones
2. **Labeling** (`local_label.py`) - Numba-accelerated 3D connected components using two-pass union-find. Only 6-connectivity is supported for stitched runs
3. **Metrics** (`metrics.py`) - Per-label statistics: cell counts, volumes, masses, centroids, exposed surface area, bounding boxes, velocity moments
4. **Driver** (`clump_finder.py`) - MPI Cartesian decomposition, orchestrates I/O, labeling, metrics, writes per-rank `.npz` + `.meta.json`
5. **Stitching** (`stitch.py`) - Post-hoc merging of per-rank catalogs via DSU on boundary face labels
6. **Analysis** (`scripts/analysis/`) - Aggregation, plotting, PCA, correlation analysis

### Key Design Decisions
- **One MPI rank per node**: Matches Frontier's 56-core CPU layout; Numba handles parallelism within node
- **Float32 inputs, float64 accumulators**: Balances memory pressure and numerical stability
- **6-connectivity enforced**: 18/26-connected modes are disabled because they require cross-tile edge/corner merges not implemented in the stitcher
- **min_clump_cells filtering deferred**: Applied after stitching to avoid breaking boundary connections

### Configuration Structure
- `configs/base/` - Template configs
- `configs/presets/` - Resolution-specific presets (n1280, n2560, n5120, n10240)
- `configs/runs/` - Production run configs with step numbers

### Output Format
Per-rank files: `clumps_rankXXXXX.npz` + `clumps_rankXXXXX.meta.json`

Key arrays in `.npz`:
- `cell_count`, `volume`, `mass`, `area` - Basic geometry
- `centroid_vol`, `centroid_mass` - Weighted centroids
- `velocity_mean`, `velocity_std` - Speed statistics
- `bbox_ijk` - Global bounding boxes [i_min, i_max, j_min, j_max, k_min, k_max]
- `face_*`, `ovlp_*` - Boundary data for stitching

### Coordinate Conventions
- openPMD stores data as [z, y, x]; code transposes to [i, j, k] = [x, y, z]
- `node_bbox` uses global indices with exclusive upper bounds: `((i0, i1), (j0, j1), (k0, k1))`
