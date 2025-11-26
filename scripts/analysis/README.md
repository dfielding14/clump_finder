# Analysis Scripts

| Script | Purpose |
| --- | --- |
| `aggregate_results.py` | Stitch per-rank `.npz` files into a catalog and optional JSON sidecar. |
| `analyze_correlations.py` | Compute correlation matrices for clump properties. |
| `analyze_embedding.py` | Build TSNE/UMAP embeddings for exploratory analysis. |
| `analyze_pca.py` | Principal component / factor analysis of clump features. |
| `plot_clumps.py` | Generate diagnostic PNGs (size histogram, size vs velocity dispersion, surface area ratio). |
| `plot_cumulative_product.py` | Quick-look cumulative statistics visualizations. |

Invoke with `python scripts/analysis/<name>.py [args...]` from the repo root.

Most scripts auto-discover clump features via `feature_utils.py`. The default feature lists align with the lean catalog output (`volume`, `mass`, `area`, `cell_count`, `velocity_std`, and `velocity_mean`). If you pass feature names that require the richer statistics (e.g. `rho_mean`, `pressure_std`, velocity components), rerun `clump_finder.py` with `--extra-stats` (or set `extra_stats: true` in the config) so those arrays are written into each `clumps_master.npz`.
