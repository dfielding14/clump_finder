#!/bin/bash
#SBATCH -J clumpn1280_cf
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

mkdir -p logs
mkdir -p clump_out/clumpn1280_step35

echo "======================================"
echo "Clump finder run (n1280) starting at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes : ${SLURM_NNODES}"
echo "Tasks : ${SLURM_NTASKS}"
echo "CPUs  : ${SLURM_CPUS_PER_TASK}"
echo "Python: $(which python)"
python --version
echo "Workdir: ${PWD}"
echo "Config : ${CONF:-config_n1280.yaml}"
echo "======================================"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=BLOCK

CONF=${CONF:-config_n1280.yaml}
OUT_DIR="clump_out/clumpn1280_step35"
MASTER_NPZ="${OUT_DIR}/clumps_master.npz"
MASTER_JSON="${OUT_DIR}/clumps_master.meta.json"

srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} --cpu-bind=cores \
     python -u clump_finder.py --config "${CONF}"

echo "Aggregating per-rank outputs at $(date)"
python -u aggregate_results.py \
    --input "${OUT_DIR}" \
    --output "${MASTER_NPZ}" \
    --sidecar "${MASTER_JSON}"

echo "Generating plots at $(date)"
python -u plot_clumps.py \
    --input "${MASTER_NPZ}" \
    --outdir "${OUT_DIR}"
