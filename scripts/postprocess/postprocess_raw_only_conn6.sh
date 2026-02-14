#!/usr/bin/env bash
#
# Post-process raw-only clump outputs for conn6_T0p02 sweeps.
#
# Finds directories that contain clumps_rank*.npz but no clumps_master/stitched
# outputs. Single-rank outputs are aggregated and plotted without stitching;
# multi-rank outputs are stitched and fully processed.
#
# Optional env vars:
#   OUTPUT_ROOT   (default: clump_out)
#   RESOLUTIONS   (default: "320 640 1280 2560")
#   DRY_RUN=1     (list targets, no processing)
#   PYTHON        (override python binary)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT=${OUTPUT_ROOT:-clump_out}
RESOLUTIONS_STR=${RESOLUTIONS:-"320 640 1280 2560"}
log() {
    printf '[postprocess-raw] %s\n' "$*" >&2
}

read -r -a RESOLUTIONS <<< "${RESOLUTIONS_STR}"

declare -a SINGLE_DIRS=()
declare -a MULTI_DIRS=()

for res in "${RESOLUTIONS[@]}"; do
    sweep_dir="${OUTPUT_ROOT}/n${res}_sweep"
    if [[ ! -d "${sweep_dir}" ]]; then
        log "Skipping missing sweep directory ${sweep_dir}"
        continue
    fi

    while IFS= read -r -d '' dir; do
        shopt -s nullglob
        rank_files=("${dir}/clumps_rank"*.npz)
        shopt -u nullglob

        if (( ${#rank_files[@]} == 0 )); then
            continue
        fi

        if [[ -f "${dir}/clumps_master.npz" ]]; then
            continue
        fi

        if compgen -G "${dir}/*clumps_stitched*.npz" > /dev/null; then
            continue
        fi

        if (( ${#rank_files[@]} > 1 )); then
            MULTI_DIRS+=("${dir}")
        else
            SINGLE_DIRS+=("${dir}")
        fi
    done < <(find "${sweep_dir}" -maxdepth 1 -mindepth 1 -type d -name 'conn6_T0p02_step*' -print0 | sort -z)

done

if (( ${#SINGLE_DIRS[@]} == 0 && ${#MULTI_DIRS[@]} == 0 )); then
    log "No raw-only directories found under ${OUTPUT_ROOT}."
    exit 0
fi

log "Found ${#SINGLE_DIRS[@]} single-rank and ${#MULTI_DIRS[@]} multi-rank raw-only directories."

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if (( ${#SINGLE_DIRS[@]} > 0 )); then
        printf 'single-rank: %s\n' "${SINGLE_DIRS[@]}"
    fi
    if (( ${#MULTI_DIRS[@]} > 0 )); then
        printf 'multi-rank: %s\n' "${MULTI_DIRS[@]}"
    fi
    exit 0
fi

NTASKS=${SLURM_NTASKS:-1}
if (( NTASKS > 1 )); then
    if (( ${#MULTI_DIRS[@]} > 0 )); then
        list_file=$(mktemp postprocess_raw_only_multi.XXXXXX)
        trap 'rm -f "${list_file}"' EXIT
        printf "%s\n" "${MULTI_DIRS[@]}" > "${list_file}"
        log "Dispatching multi-rank stitching across ${NTASKS} ranks."
        srun --ntasks="${NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
            bash scripts/postprocess/process_rank_worker.sh "${list_file}"
        rm -f "${list_file}"
        trap - EXIT
    fi

    if (( ${#SINGLE_DIRS[@]} > 0 )); then
        list_file=$(mktemp postprocess_raw_only_single.XXXXXX)
        trap 'rm -f "${list_file}"' EXIT
        printf "%s\n" "${SINGLE_DIRS[@]}" > "${list_file}"
        log "Dispatching single-rank aggregation across ${NTASKS} ranks."
        srun --ntasks="${NTASKS}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
            bash scripts/postprocess/process_rank_worker_no_stitch.sh "${list_file}"
        rm -f "${list_file}"
        trap - EXIT
    fi
else
    for dir in "${MULTI_DIRS[@]}"; do
        bash scripts/postprocess/process_one_dir.sh "${dir}"
    done
    for dir in "${SINGLE_DIRS[@]}"; do
        bash scripts/postprocess/process_one_dir_no_stitch.sh "${dir}"
    done
fi

log "Raw-only postprocessing complete."
