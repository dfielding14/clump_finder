#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "[process_one_dir] Usage: $0 <directory>" >&2
    exit 1
fi

DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON:-python}
export PYTHONPATH="${REPO_ROOT}/scripts/analysis${PYTHONPATH:+:${PYTHONPATH}}"

log() {
    printf '[process_one_dir] %s\n' "$*" >&2
}

if [[ ! -d ${DIR} ]]; then
    log "Skipping missing directory ${DIR}"
    exit 0
fi

log "Processing ${DIR}"

${PYTHON_BIN} -u scripts/analysis/aggregate_results.py \
    --input "${DIR}" \
    --output "${DIR}/clumps_master.npz" \
    --sidecar "${DIR}/clumps_master.meta.json" >&2

${PYTHON_BIN} -u stitch.py \
    --input "${DIR}" \
    --output "${DIR}/clumps_stitched.npz" >&2

clump_count=$(${PYTHON_BIN} - "${DIR}" <<'PY'
import json, os, sys, numpy as np
base = sys.argv[1]
meta_path = os.path.join(base, "clumps_master.meta.json")
count = None
if os.path.isfile(meta_path):
    try:
        with open(meta_path) as fh:
            count = int(json.load(fh).get("clump_count"))
    except Exception:
        count = None
if count is None:
    npz_path = os.path.join(base, "clumps_master.npz")
    if os.path.isfile(npz_path):
        with np.load(npz_path) as data:
            if data.files:
                count = int(np.asarray(data[data.files[0]]).shape[0])
            else:
                count = 0
    else:
        count = 0
print(count if count is not None else 0)
PY
)

if (( clump_count > 0 )); then
    ${PYTHON_BIN} -u scripts/analysis/plot_clumps.py \
        --input "${DIR}/clumps_master.npz" \
        --outdir "${DIR}" \
        --compare "${DIR}/clumps_stitched.npz" \
        --compare-labels "Unstitched" "Stitched" >&2

    ${PYTHON_BIN} -u scripts/analysis/plot_clumps.py \
        --input "${DIR}/clumps_stitched.npz" \
        --outdir "${DIR}/stitched_plots" >&2
else
    log "No clumps detected in ${DIR}; skipping plot generation."
fi

echo "${clump_count}"
