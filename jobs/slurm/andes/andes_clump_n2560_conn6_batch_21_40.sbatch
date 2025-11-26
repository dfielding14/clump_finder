#!/bin/bash
#SBATCH -J clumpn2560_conn6_T0p02_21_40
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

CONFIG_DIR="configs/runs/n2560_batch"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for step in $(seq 21 40); do
    STEP_PAD=$(printf "%05d" $step)
    CONF_PATH="${CONFIG_DIR}/config_conn6_T0p02_step${STEP_PAD}.yaml"
    if [[ ! -f "${CONF_PATH}" ]]; then
        echo "Config not found: ${CONF_PATH}, skipping"
        continue
    fi
    echo ""
    echo ">>> Clump finding for ${CONF_PATH} at $(date)"
    srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} --cpu-bind=cores \
         python -u clump_finder.py --config "${CONF_PATH}"
done

echo "Clump finding complete for ${SLURM_JOB_NAME}."
