#!/bin/bash
# Quick 256³ test to validate the full pipeline before running 512³
# Tests synthetic data generation + fractal P-A analysis with beta=-3 and beta=-4

set -e  # Exit on error

echo "=============================================="
echo "256³ Test: Synthetic Data + Fractal Analysis"
echo "Beta = -3 and -4, k_min = 4, k_max = N/8"
echo "=============================================="

# Beta = -3
OUTDIR_B3="./test_256_beta3"
echo ""
echo ">>> Generating 256³ isotropic synthetic data (beta=-3)..."
time python synthetic_powerlaw_test.py --N 256 --beta -3 --outdir "$OUTDIR_B3"

# Beta = -4
OUTDIR_B4="./test_256_beta4"
echo ""
echo ">>> Generating 256³ isotropic synthetic data (beta=-4)..."
time python synthetic_powerlaw_test.py --N 256 --beta -4 --outdir "$OUTDIR_B4"

# Check that labels were saved
echo ""
echo ">>> Checking NPZ file contents..."
for OUTDIR in "$OUTDIR_B3" "$OUTDIR_B4"; do
    echo "Checking $OUTDIR..."
    python -c "
import numpy as np
npz = np.load('$OUTDIR/clumps_isotropic_rank00000.npz')
print('  Keys:', list(npz.keys())[:6], '...')
if 'labels' in npz:
    labels = npz['labels']
    print(f'  labels shape: {labels.shape}')
    print(f'  labels max (K): {labels.max()}')
else:
    print('  ERROR: labels not found!')
    exit(1)
"
done

# Run fractal analysis on both
echo ""
echo ">>> Running fractal P-A analysis (beta=-3)..."
time python run_fractal_analysis.py \
    --labels "$OUTDIR_B3/clumps_isotropic_rank00000.npz" \
    --labels-key labels \
    --output "$OUTDIR_B3/fractal_analysis.npz" \
    --solidity-threshold 0.9 \
    --save-slices

echo ""
echo ">>> Running fractal P-A analysis (beta=-4)..."
time python run_fractal_analysis.py \
    --labels "$OUTDIR_B4/clumps_isotropic_rank00000.npz" \
    --labels-key labels \
    --output "$OUTDIR_B4/fractal_analysis.npz" \
    --solidity-threshold 0.9 \
    --save-slices

echo ""
echo "=============================================="
echo "Test complete!"
echo "=============================================="
echo "Outputs:"
echo "  Beta=-3: $OUTDIR_B3/"
echo "    - clumps_isotropic_rank00000.npz  (3D labels)"
echo "    - fractal_analysis.npz/.png"
echo "    - fractal_analysis_slices/"
echo ""
echo "  Beta=-4: $OUTDIR_B4/"
echo "    - clumps_isotropic_rank00000.npz  (3D labels)"
echo "    - fractal_analysis.npz/.png"
echo "    - fractal_analysis_slices/"
echo "=============================================="
