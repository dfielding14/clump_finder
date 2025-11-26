#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sbatch-script> [delay-seconds]" >&2
    exit 1
fi

TARGET_SCRIPT=$1
DELAY=${2:-10}

if [[ ! -f ${TARGET_SCRIPT} ]]; then
    echo "chain_submit: target script ${TARGET_SCRIPT} not found" >&2
    exit 1
fi

if [[ ${DELAY} -gt 0 ]]; then
    echo "chain_submit: sleeping ${DELAY}s before submitting ${TARGET_SCRIPT}."
    sleep "${DELAY}"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "chain_submit: submitting ${TARGET_SCRIPT} with dependency after SLURM job ${SLURM_JOB_ID}."
    sbatch --dependency=afterany:${SLURM_JOB_ID} "${TARGET_SCRIPT}"
else
    echo "chain_submit: submitting ${TARGET_SCRIPT}."
    sbatch "${TARGET_SCRIPT}"
fi
