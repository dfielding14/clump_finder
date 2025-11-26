#!/bin/bash
#SBATCH -J clump_postprocess_conn6
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module reset
module load cray-python/3.11.7
source /ccs/home/dfielding/SFunctor/venv_frontier/bin/activate

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

log_dir="logs/postprocess"
mkdir -p "${log_dir}"

LOG_FILE="${log_dir}/postprocess_${SLURM_JOB_ID}.log"

echo "[$(date)] Starting clump post-processing on ${HOSTNAME}" | tee "${LOG_FILE}"

bash scripts/postprocess/postprocess_clumps.sh 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date)] Post-processing complete" | tee -a "${LOG_FILE}"
