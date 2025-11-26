#!/bin/bash
#SBATCH -J clump_snap36_triple
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -N 3
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

echo "======================================"
echo "Concurrent clump finder run starting at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes : ${SLURM_NNODES}"
echo "Tasks : ${SLURM_NTASKS}"
echo "CPUs  : ${SLURM_CPUS_PER_TASK}"
echo "Python: $(which python)"
python --version
echo "Workdir: ${PWD}"
echo "Project root: ${PROJECT_ROOT}"
echo "======================================"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=BLOCK

declare -A SNAP36_CONFIGS=(
    [n320]="configs/runs/n320_batch/config_conn6_T0p02_step00036.yaml"
    [n640]="configs/runs/n640_batch/config_conn6_T0p02_step00036.yaml"
    [n1280]="configs/runs/n1280_batch/config_conn6_T0p02_step00036.yaml"
)

declare -A SRUN_PIDS=()

run_resolution() {
    local label="$1"
    local config="$2"

    if [[ ! -f "${config}" ]]; then
        echo "[${label}] Missing config ${config}" >&2
        return 1
    fi

    echo ""
    echo "[${label}] Launching clump_finder with ${config} at $(date)"
    srun --exclusive -N1 -n1 --cpu-bind=cores \
        python -u clump_finder.py --config "${config}" &
    SRUN_PIDS["${label}"]=$!
}

for label in "${!SNAP36_CONFIGS[@]}"; do
    run_resolution "${label}" "${SNAP36_CONFIGS[${label}]}"
done

wait_status=0
for label in "${!SRUN_PIDS[@]}"; do
    pid=${SRUN_PIDS[${label}]}
    if ! wait "${pid}"; then
        echo "[${label}] clump_finder.py failed (PID ${pid})." >&2
        wait_status=1
    else
        echo "[${label}] clump_finder.py completed successfully at $(date)."
    fi
done

if [[ ${wait_status} -ne 0 ]]; then
    echo "One or more runs failed; see logs for details." >&2
    exit ${wait_status}
fi

echo ""
echo "All snap 36 runs finished at $(date)."
echo "Run post-processing (aggregate/stitch/plots) once outputs are synced."
