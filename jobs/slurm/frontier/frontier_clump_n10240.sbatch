#!/bin/bash
#SBATCH -J clumpn10240_cf
#SBATCH -A AST207
#SBATCH -p batch
#SBATCH -N 512
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module reset
module load cray-python/3.11.7
source /ccs/home/dfielding/SFunctor/venv_frontier/bin/activate

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p clump_out/clumpn10240_step35

echo "======================================"
echo "Clump finder run (n10240) starting at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes : ${SLURM_NNODES}"
echo "Tasks : ${SLURM_NTASKS}"
echo "CPUs  : ${SLURM_CPUS_PER_TASK}"
echo "Python: $(which python)"
python --version
echo "Workdir: ${PWD}"
echo "Config : ${CONF:-configs/presets/config_n10240.yaml}"
echo "======================================"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPICH_OFI_NIC_POLICY=BLOCK

BASE_CONF=${CONF:-configs/presets/config_n10240.yaml}
DATASET_ROOT="/lustre/orion/ast207/proj-shared/mpturb/clumpn5120to10240_cont_prepeak"
STEPS=("35" "36" "37")

for STEP in "${STEPS[@]}"; do
    DATASET_PATH="${DATASET_ROOT}/parthenon.opmd.final.000${STEP}.bp/"
    OUT_DIR="clump_out/clumpn10240_final_step${STEP}"
    LOG_DIR="logs/clumpn10240_final_step${STEP}"
    mkdir -p "${OUT_DIR}"
    mkdir -p "${LOG_DIR}"

    TMP_CONF=$(mktemp)
    python - "${BASE_CONF}" "${DATASET_PATH}" "${STEP}" "${OUT_DIR}" "${LOG_DIR}" "${TMP_CONF}" <<'PY'
import sys, yaml
base, dataset, step, out_dir, log_dir, tmp = sys.argv[1:]
step = int(step)
with open(base, "r") as f:
    cfg = yaml.safe_load(f)
cfg["dataset_path"] = dataset
cfg["step"] = step
cfg["output_dir"] = out_dir
cfg["log_dir"] = log_dir
with open(tmp, "w") as f:
    yaml.safe_dump(cfg, f)
PY

    echo ""
    echo ">>> Processing dataset ${DATASET_PATH} at $(date)"
    srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} --cpu-bind=cores \
         python -u clump_finder.py --config "${TMP_CONF}"

    MASTER_NPZ="${OUT_DIR}/clumps_master.npz"
    MASTER_JSON="${OUT_DIR}/clumps_master.meta.json"

    echo "Aggregating per-rank outputs at $(date)"
    python -u scripts/analysis/aggregate_results.py \
        --input "${OUT_DIR}" \
        --output "${MASTER_NPZ}" \
        --sidecar "${MASTER_JSON}"

    echo "Generating plots at $(date)"
    python -u scripts/analysis/plot_clumps.py \
        --input "${MASTER_NPZ}" \
        --outdir "${OUT_DIR}"

    rm -f "${TMP_CONF}"
done

echo ""
echo "Completed processing of final snapshots at $(date)"
