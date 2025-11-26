#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "[process_rank_worker] Usage: $0 <path-to-dir-list>" >&2
    exit 1
fi

LIST_FILE="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mapfile -t DIRS < "${LIST_FILE}"
TOTAL=${#DIRS[@]}
if (( TOTAL == 0 )); then
    exit 0
fi

NTASKS=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
if (( NTASKS <= 0 )); then
    NTASKS=1
fi

for (( idx=RANK; idx<TOTAL; idx+=NTASKS )); do
    DIR="${DIRS[idx]}"
    if [[ -n "${DIR}" ]]; then
        bash scripts/postprocess/process_one_dir.sh "${DIR}"
    fi
done
