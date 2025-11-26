#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-clump_out}
export PYTHONPATH="${REPO_ROOT}/scripts/analysis${PYTHONPATH:+:${PYTHONPATH}}"

log() {
    printf '[corr-login] %s\n' "$*" >&2
}

count_from_npz() {
    local npz_path="$1"
    if [[ ! -f "${npz_path}" ]]; then
        echo 0
        return
    fi
    ${PYTHON_BIN} - "${npz_path}" <<'PY'
import sys, numpy as np
path = sys.argv[1]
with np.load(path) as data:
    if not data.files:
        print(0)
    else:
        first_key = data.files[0]
        print(int(np.asarray(data[first_key]).shape[0]))
PY
}

collect_stitched() {
    local root="$1"
    local pattern="$2"
    local files=()
    shopt -s nullglob
    for dir in "${root}"/${pattern}; do
        if [[ -d "${dir}" ]]; then
            local npz="${dir}/clumps_stitched.npz"
            if [[ -f "${npz}" ]]; then
                local count
                count=$(count_from_npz "${npz}")
                if (( count > 0 )); then
                    files+=("${npz}")
                fi
            fi
        fi
    done
    shopt -u nullglob
    printf '%s\n' "${files[@]}"
}

run_for_root() {
    local root="$1"
    local pattern="$2"
    local out_dir="$3"

    if [[ ! -d "${root}" ]]; then
        log "Skipping missing root ${root}"
        return
    fi

    mapfile -t stitched < <(collect_stitched "${root}" "${pattern}")
    if ((${#stitched[@]} == 0)); then
        log "No stitched catalogs with clumps under ${root} (${pattern}); skipping."
        return
    fi

    mkdir -p "${out_dir}"

    local tmpdir
    tmpdir=$(mktemp -d correlation_tmp.login.XXXXXX)

    for f in "${stitched[@]}"; do
        local base
        base="$(basename "$(dirname "${f}")")"
        ln -sf "${REPO_ROOT}/${f}" "${tmpdir}/${base}.npz"
    done

    log "Running analyze_correlations.py for ${root}"
    ${PYTHON_BIN} -u scripts/analysis/analyze_correlations.py \
        --input-root "${tmpdir}" \
        --file-pattern "*.npz" \
        --output-dir "${out_dir}" >&2

    rm -rf "${tmpdir}"
}

run_for_root "${OUTPUT_ROOT}/n5120_sweep" 'conn6_T0p02_step*' "${OUTPUT_ROOT}/n5120_sweep/correlation_scan"
run_for_root "${OUTPUT_ROOT}/n10240_sweep" 'conn6_T0p02*_step*' "${OUTPUT_ROOT}/n10240_sweep/correlation_scan"

log "Correlation scans complete."
