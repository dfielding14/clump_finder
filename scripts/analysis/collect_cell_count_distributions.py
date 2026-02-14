#!/usr/bin/env python3
"""
Collect cell-count distributions for a sweep.

For each snapshot directory in a sweep (e.g. clump_out/n320_sweep),
load the merged clump catalog (prefer stitched if present) and extract
cell counts. Save a single NPZ with per-snapshot arrays.
"""

import argparse
import os
import re
import sys
import numpy as np

STEP_RE = re.compile(r"step0*(\d+)")


def find_snapshot_dirs(sweep_dir: str):
    entries = []
    for name in os.listdir(sweep_dir):
        full = os.path.join(sweep_dir, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith("conn6_T0p02"):
            continue
        if "step" not in name:
            continue
        step_match = STEP_RE.search(name)
        step_num = int(step_match.group(1)) if step_match else -1
        entries.append((step_num, name, full))
    entries.sort(key=lambda x: (x[0], x[1]))
    return entries


def load_cell_counts(npz_path: str):
    data = np.load(npz_path)
    if "cell_count" in data.files:
        key = "cell_count"
    elif "num_cells" in data.files:
        key = "num_cells"
    elif "volume" in data.files:
        key = "volume"
    else:
        raise KeyError(f"No cell-count-like key found in {npz_path}. Keys: {data.files}")
    counts = np.asarray(data[key])
    return key, counts


def main():
    parser = argparse.ArgumentParser(description="Collect per-snapshot cell counts for a sweep.")
    parser.add_argument("--sweep-dir", required=True, help="Path to nXXX_sweep directory")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument(
        "--prefer-stitched",
        action="store_true",
        default=False,
        help="Use clumps_stitched.npz when available (default: clumps_master.npz)",
    )
    args = parser.parse_args()

    sweep_dir = os.path.abspath(args.sweep_dir)
    if not os.path.isdir(sweep_dir):
        print(f"Sweep directory not found: {sweep_dir}", file=sys.stderr)
        return 1

    snapshots = find_snapshot_dirs(sweep_dir)
    if not snapshots:
        print(f"No snapshot dirs found under {sweep_dir}", file=sys.stderr)
        return 1

    labels = []
    source_files = []
    count_keys = []
    payload = {}
    missing = 0

    for _, name, path in snapshots:
        stitched = os.path.join(path, "clumps_stitched.npz")
        master = os.path.join(path, "clumps_master.npz")
        use_path = None
        if args.prefer_stitched and os.path.isfile(stitched):
            use_path = stitched
        elif os.path.isfile(master):
            use_path = master
        elif os.path.isfile(stitched):
            use_path = stitched

        if use_path is None:
            print(f"Warning: no merged NPZ found in {path}", file=sys.stderr)
            missing += 1
            continue

        key, counts = load_cell_counts(use_path)
        labels.append(name)
        source_files.append(use_path)
        count_keys.append(key)
        payload[f"counts__{name}"] = counts

    if not labels:
        print("No usable snapshots found; nothing to save.", file=sys.stderr)
        return 1

    payload["snapshot_labels"] = np.array(labels, dtype=object)
    payload["source_files"] = np.array(source_files, dtype=object)
    payload["cell_count_key"] = np.array(count_keys, dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **payload)

    print(f"Wrote {args.output} ({len(labels)} snapshots, missing {missing})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
