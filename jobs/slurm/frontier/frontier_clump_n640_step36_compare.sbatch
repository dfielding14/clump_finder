#!/bin/bash
#SBATCH -J n640_conn_temp_scan
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module reset
module load cray-python/3.11.7
source /ccs/home/dfielding/SFunctor/venv_frontier/bin/activate

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

CONFIG_DIR="configs/runs/n640_step36_compare"
CONFIGS=(
    "config_conn6_T0p10.yaml"
    "config_conn6_T0p02.yaml"
    "config_conn18_T0p10.yaml"
    "config_conn18_T0p02.yaml"
    "config_conn26_T0p10.yaml"
    "config_conn26_T0p02.yaml"
)

mkdir -p logs/n640_step36

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=BLOCK

echo "======================================"
echo "Connectivity/threshold sweep starting at $(date)"
echo "Configs: ${CONFIGS[*]}"
echo "======================================"

for cfg in "${CONFIGS[@]}"; do
    CONF_PATH="${CONFIG_DIR}/${cfg}"
    echo ""
    echo ">>> Running ${CONF_PATH} at $(date)"
    srun -N 1 -n 1 --cpu-bind=cores \
        python -u clump_finder.py --config "${CONF_PATH}"

    OUT_DIR=$(python - "${CONF_PATH}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get("output_dir", "./clump_out"))
PY
)

    MASTER_NPZ="${OUT_DIR}/clumps_master.npz"
    MASTER_JSON="${OUT_DIR}/clumps_master.meta.json"
    echo "Aggregating results in ${OUT_DIR}"
    python -u scripts/analysis/aggregate_results.py \
        --input "${OUT_DIR}" \
        --output "${MASTER_NPZ}" \
        --sidecar "${MASTER_JSON}"

    echo "Generating plots for ${MASTER_NPZ}"
    python -u scripts/analysis/plot_clumps.py \
        --input "${MASTER_NPZ}" \
        --outdir "${OUT_DIR}"
done

echo ""
echo "Sweep complete at $(date)"
