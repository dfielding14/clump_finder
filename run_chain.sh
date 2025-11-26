#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 job1.sbatch [job2.sbatch ...]" >&2
    exit 1
fi

for script in "$@"; do
    if [[ ! -f "${script}" ]]; then
        echo "run_chain: missing sbatch script ${script}" >&2
        exit 1
    fi
done

for script in "$@"; do
    echo "run_chain: submitting ${script} and waiting for completion..."
    if ! sbatch --wait "${script}"; then
        echo "run_chain: ${script} failed. Stopping chain." >&2
        exit 1
    fi
done

echo "run_chain: all jobs completed successfully."
