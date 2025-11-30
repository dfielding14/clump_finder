#!/bin/bash
# Run 512³ anisotropic tests with different beta values
# Then run fractal perimeter-area analysis on each configuration

set -e  # Exit on error

echo "=============================================="
echo "512³ Anisotropic Tests + Fractal Analysis"
echo "=============================================="

# Beta = -3
echo ""
echo ">>> Running beta=-3 synthetic test..."
time python synthetic_powerlaw_test.py --N 512 --beta -3 --batch-anisotropy --outdir ./aniso_batch_512_beta3

# Beta = -4
echo ""
echo ">>> Running beta=-4 synthetic test..."
time python synthetic_powerlaw_test.py --N 512 --beta -4 --batch-anisotropy --outdir ./aniso_batch_512_beta4

echo ""
echo "=============================================="
echo "Fractal P-A Analysis"
echo "=============================================="

# Function to run fractal analysis on a configuration
run_fractal_analysis() {
    local base_dir=$1
    local config=$2
    local labels_file="${base_dir}/${config}/clumps_${config}_rank00000.npz"
    local output_file="${base_dir}/${config}/fractal_analysis.npz"

    if [ -f "$labels_file" ]; then
        echo ""
        echo ">>> Analyzing ${config} in ${base_dir}..."
        python run_fractal_analysis.py \
            --labels "$labels_file" \
            --labels-key labels \
            --output "$output_file" \
            --solidity-threshold 0.9 \
            --save-slices
    else
        echo "Skipping ${config}: labels file not found"
    fi
}

# List of configurations to analyze
CONFIGS="isotropic prolate super_prolate oblate super_oblate triaxial_A triaxial_B"

# Run fractal analysis on beta=-3 results
echo ""
echo ">>> Beta=-3 fractal analysis..."
for config in $CONFIGS; do
    run_fractal_analysis "./aniso_batch_512_beta3" "$config"
done

# Run fractal analysis on beta=-4 results
echo ""
echo ">>> Beta=-4 fractal analysis..."
for config in $CONFIGS; do
    run_fractal_analysis "./aniso_batch_512_beta4" "$config"
done

echo ""
echo "=============================================="
echo "All tests complete!"
echo "=============================================="
echo "Synthetic data results:"
echo "  ./aniso_batch_512_beta3/"
echo "  ./aniso_batch_512_beta4/"
echo ""
echo "Fractal analysis outputs (per config):"
echo "  fractal_analysis.npz    - Numerical results"
echo "  fractal_analysis.png    - P-A scatter plot"
echo "  fractal_analysis_slices/ - Example slice images"
echo "=============================================="
