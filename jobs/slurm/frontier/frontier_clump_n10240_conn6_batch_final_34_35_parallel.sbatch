#!/bin/bash
#SBATCH -J clumpn10240_conn6_T0p02_final_34_35_par
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -N 512
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

CONFIG_DIR="configs/runs/n10240_batch"
CONFIGS=(
    config_conn6_T0p02_final_step00034.yaml
    config_conn6_T0p02_final_step00035.yaml
)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=BLOCK

for cfg in "${CONFIGS[@]}"; do
    CONF_PATH="${CONFIG_DIR}/${cfg}"
    echo ""
    echo ">>> Parallel clump finding for ${CONF_PATH} at $(date)"
    srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} --cpu-bind=cores \
         python -u clump_finder.py --config "${CONF_PATH}"
done

echo "Parallel stage complete for ${SLURM_JOB_NAME}. Aggregate, stitch, and plot on the login node after this job finishes."
