#!/usr/bin/env bash
#
# Orchestrate clump post-processing:
#   - aggregate per-rank outputs
#   - stitch catalogs
#   - generate plot_clumps diagnostics
#   - build correlation scans from stitched catalogs with clumps
#
# Supports parallel execution via SLURM (set SLURM_NTASKS > 1); each rank
# processes a disjoint subset of directories through process_one_dir.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-clump_out}
export PYTHONPATH="${REPO_ROOT}/scripts/analysis${PYTHONPATH:+:${PYTHONPATH}}"

log() {
    printf '[postprocess] %s\n' "$*" >&2
}

run_corr_scan() {
    local tag="$1"; shift
    local files=("$@")

    if ((${#files[@]} == 0)); then
        log "No stitched catalogs with clumps for ${tag}; skipping correlation scan."
        return
    fi

    local tmpdir
    tmpdir=$(mktemp -d "${tag##*/}.XXXXXX")
    trap 'rm -rf "${tmpdir}"' EXIT

    for f in "${files[@]}"; do
        local base
        base="$(basename "$(dirname "${f}")")"
        ln -sf "${f}" "${tmpdir}/${base}.npz"
    done

    ${PYTHON_BIN} -u scripts/analysis/analyze_correlations.py \
        --input-root "${tmpdir}" \
        --file-pattern "*.npz" \
        --output-dir "${tag}" >&2

    rm -rf "${tmpdir}"
    trap - EXIT
}

get_clump_count() {
    local dir="$1"
    ${PYTHON_BIN} - "${dir}" <<'PY'
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
}

# Gather directory lists
mapfile -t dirs_5120 < <(find "${OUTPUT_ROOT}/n5120_sweep" -maxdepth 1 -mindepth 1 -type d -name 'conn6_T0p02_step*' | sort)
mapfile -t dirs_10240_step < <(find "${OUTPUT_ROOT}/n10240_sweep" -maxdepth 1 -mindepth 1 -type d -name 'conn6_T0p02_step*' | sort)
mapfile -t dirs_10240_final < <(find "${OUTPUT_ROOT}/n10240_sweep" -maxdepth 1 -mindepth 1 -type d -name 'conn6_T0p02_final_step*' | sort)

all_dirs=("${dirs_5120[@]}" "${dirs_10240_step[@]}" "${dirs_10240_final[@]}")

if ((${#all_dirs[@]} == 0)); then
    log "No directories found under ${OUTPUT_ROOT}; nothing to do."
    exit 0
fi

log "Preparing to process ${#all_dirs[@]} directories."

NTASKS=${SLURM_NTASKS:-1}
if (( NTASKS > 1 )); then
    list_file=$(mktemp postprocess_dirs.XXXXXX)
    trap 'rm -f "${list_file}"' EXIT
    printf "%s\n" "${all_dirs[@]}" > "${list_file}"
    log "Dispatching work across ${NTASKS} ranks."
    srun --ntasks="${NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash scripts/postprocess/process_rank_worker.sh "${list_file}"
    rm -f "${list_file}"
    trap - EXIT
else
    for dir in "${all_dirs[@]}"; do
        bash scripts/postprocess/process_one_dir.sh "${dir}"
    done
fi

# Collect stitched catalogs with clumps
declare -a stitched_5120=()
declare -a stitched_10240=()

for dir in "${dirs_5120[@]}"; do
    if (( $(get_clump_count "${dir}") > 0 )); then
        stitched_5120+=("${dir}/clumps_stitched.npz")
    fi
done
for dir in "${dirs_10240_step[@]}" "${dirs_10240_final[@]}"; do
    if (( $(get_clump_count "${dir}") > 0 )); then
        stitched_10240+=("${dir}/clumps_stitched.npz")
    fi
done

run_corr_scan "${OUTPUT_ROOT}/n5120_sweep/correlation_scan" "${stitched_5120[@]}"
run_corr_scan "${OUTPUT_ROOT}/n10240_sweep/correlation_scan" "${stitched_10240[@]}"

log "Post-processing complete."
