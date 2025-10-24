#!/bin/bash
#SBATCH -J clumps
#SBATCH -A <your_project>
#SBATCH -N <nodes>                      # total nodes
#SBATCH --ntasks-per-node=1             # one MPI rank per node
#SBATCH --cpus-per-task=56              # threads inside rank
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# -- Load your site modules here (placeholders; supply concrete examples) --
# module restore
# module load cray-python
# source /path/to/your/venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

CONF=${CONF:-config.yaml}

srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} \
     python -u clump_finder.py --config "${CONF}"

# optional separate aggregation step if you prefer:
# if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
#     python -u aggregate_results.py --input ./clump_out --output ./clump_out/clumps_master.npz
# fi
