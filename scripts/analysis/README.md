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
