#!/bin/bash
#SBATCH -J clumpn2560_conn6_T0p02_36
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -N 64
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

CONF_PATH="configs/runs/n2560_batch/config_conn6_T0p02_step00036.yaml"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=BLOCK

echo ">>> Running ${CONF_PATH} at $(date)"
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} --cpu-bind=cores \
     python -u clump_finder.py --config "${CONF_PATH}" --extra-stats

OUT_DIR=$(python - "${CONF_PATH}" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir'))
PY
)

MASTER_NPZ="${OUT_DIR}/clumps_master.npz"
MASTER_JSON="${OUT_DIR}/clumps_master.meta.json"
STITCHED_NPZ="${OUT_DIR}/clumps_stitched.npz"
STITCHED_PLOTS="${OUT_DIR}/stitched_plots"

echo "Aggregating per-rank outputs at $(date)"
python -u scripts/analysis/aggregate_results.py \
    --input "${OUT_DIR}" \
    --output "${MASTER_NPZ}" \
    --sidecar "${MASTER_JSON}"

echo "Running overlap-exact stitching at $(date)"
python -u stitch.py \
    --input "${OUT_DIR}" \
    --output "${STITCHED_NPZ}"

echo "Generating comparison plots at $(date)"
python -u scripts/analysis/plot_clumps.py \
    --input "${MASTER_NPZ}" \
    --outdir "${OUT_DIR}" \
    --compare "${STITCHED_NPZ}" \
    --compare-labels "Unstitched" "Stitched"

echo "Plotting stitched catalog at $(date)"
python -u scripts/analysis/plot_clumps.py \
    --input "${STITCHED_NPZ}" \
    --outdir "${STITCHED_PLOTS}"

echo "Done at $(date)"
