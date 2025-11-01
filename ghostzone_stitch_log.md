# Ghostzone Stitch Implementation Log

## 2025-10-31 22:37 CET
- Reviewed `ghostzone_stitch.md` to align on the overlap-exact approach.
- Logged new constraints from the user: overlap must be mandatory, and min clump filtering must happen post-stitch.

## 2025-10-31 22:43 CET
- Added mandatory overlap support in `clump_finder.py`: enforced ghost/overlap width >=1, generated mapped overlap planes, and deferred `min_clump_cells` to post-stitch metadata.
- Extended `stitch.py` with `_merge_by_overlap_planes`, new `overlap-exact` mode, and CLI defaults.
- Updated `tests/test_equivalence_no_mpi.py` to emit overlap planes, require overlap >=1, and drive new stitch mode.

## 2025-10-31 22:43 CET
- Ran `python -m compileall` on updated modules to sanity-check syntax.

## 2025-10-31 22:58 CET
- Audited progress against `ghostzone_stitch.md`; core diffs applied but equivalence harness still failing (`K` mismatches).
- Executed `python tests/test_equivalence_no_mpi.py --N 96 --px 2 --py 2 --pz 2 --overlap 1` (fails with stitched_K <=> baseline_K mismatches).
- Ran debug variant on `N=32` to inspect interface diagnostics; observed significant component splits highlighting overlap-plane alignment issues that still need resolution.

## 2025-10-31 23:18 CET
- Fixed per-tile halo extraction in `_write_parts` to pull overlap data from the full field instead of the local sub-block, restoring exact cell coverage.
- Updated `stitch_reduce` aggregation to use `np.add.at`/`np.minimum.at`/`np.maximum.at`, eliminating double-count loss when multiple labels map to the same DSU root.
- After fixes, `python tests/test_equivalence_no_mpi.py --N 96 --px 2 --py 2 --pz 2 --overlap 1` now passes (T and density cuts).

## 2025-11-01 00:15 CET
- Augmented diagnostics when equivalence checks fail: `_compare` now returns detail objects, `_diagnose_failure` prints K/cell-count stats, and volume & surface ratio histograms are emitted per failure case.
- Added automated plots of clump volume distributions and area/volume^{2/3} with min-ratio warnings (<6 threshold), written into the temp workspace for inspection.

## 2025-11-01 00:28 CET
- Fixed `local_label.label_3d` global tiling bug: background voxels in tiles beyond the first were offset by `gbase`, causing massive overcounts once N exceeded `tile_shape`. Limited the offset to positive labels only.
- Re-validated `_write_parts` total cell coverage; per-tile sums now match the baseline for large grids (e.g., N=320).
- Large-scale equivalence runs (N=320 with (2,2,2) and (2,2,5) decompositions) now pass with the enhanced diagnostics staying quiet.
