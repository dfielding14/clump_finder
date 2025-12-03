#!/usr/bin/env python3
"""Check Minkowski functional data in output files."""
import numpy as np
import sys

npz_path = sys.argv[1] if len(sys.argv) > 1 else "clump_out/n10240_sweep/conn6_T0p02_final_step00037/clumps_master.npz"
d = np.load(npz_path)

cell_count = d["cell_count"]
print(f"Total clumps: {cell_count.shape[0]:,}")
print()

# Check flags
minkowski_computed = d.get("minkowski_computed")
touches_boundary = d.get("touches_boundary")

if minkowski_computed is not None:
    n_computed = minkowski_computed.sum()
    print(f"minkowski_computed: {n_computed:,} / {minkowski_computed.shape[0]:,} ({100*minkowski_computed.mean():.2f}%)")

if touches_boundary is not None:
    n_boundary = touches_boundary.sum()
    print(f"touches_boundary: {n_boundary:,} / {touches_boundary.shape[0]:,} ({100*touches_boundary.mean():.2f}%)")

print()

# Check euler and curvature
euler = d.get("euler_characteristic")
curv = d.get("integrated_curvature")

if euler is not None:
    finite = np.isfinite(euler)
    print(f"euler_characteristic: finite={finite.sum():,}")
    euler_finite = euler[finite]
    if euler_finite.size > 0:
        print(f"  range: [{euler_finite.min():.4g}, {euler_finite.max():.4g}]")
        uniq = np.unique(euler_finite)
        print(f"  unique values (first 10): {uniq[:10]}")

if curv is not None:
    finite = np.isfinite(curv)
    nonzero = finite & (curv != 0)
    print(f"integrated_curvature: finite={finite.sum():,}, nonzero={nonzero.sum():,}")
    if nonzero.any():
        print(f"  range (nonzero): [{curv[nonzero].min():.4g}, {curv[nonzero].max():.4g}]")

print()

# Check shapefinders
for key in ["thickness", "breadth", "length", "planarity", "filamentarity"]:
    arr = d.get(key)
    if arr is not None:
        finite = np.isfinite(arr)
        n_finite = finite.sum()
        if n_finite > 0:
            print(f"{key}: finite={n_finite:,}, range=[{arr[finite].min():.4g}, {arr[finite].max():.4g}]")
        else:
            print(f"{key}: no finite values")

# Check why so few are computed
if minkowski_computed is not None and touches_boundary is not None:
    # Clumps that touch boundary should not have Minkowski computed
    interior = ~touches_boundary
    print(f"\nInterior clumps (not touching boundary): {interior.sum():,}")
    print(f"Minkowski computed on interior: {(minkowski_computed & interior).sum():,}")
