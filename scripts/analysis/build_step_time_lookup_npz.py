#!/usr/bin/env python3
"""
Build per-sweep NPZ lookup for step -> simulation time.

Outputs an NPZ with arrays aligned to clump_cell_count_distribution_evolution_nXXX.npz:
  - snapshot_labels
  - step_numbers
  - sim_time
  - variant ("main" or "final")
"""

import argparse
import os
import re
import sys
import yaml
import numpy as np
import adios2

STEP_RE = re.compile(r"step0*(\d+)")


def read_time(dataset_path: str, step: int) -> float:
    path = dataset_path.rstrip("/")
    fr = adios2.FileReader(path)
    try:
        attr = f"/data/{step}/time"
        time_val = fr.read_attribute(attr)
    finally:
        fr.close()

    if isinstance(time_val, (list, tuple)) and len(time_val) == 1:
        return float(time_val[0])
    try:
        return float(time_val)
    except Exception:
        return float(time_val[()])


def scan_configs(config_dir: str):
    configs = []
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith(".yaml"):
            continue
        if "conn6_T0p02" not in name:
            continue
        if "step" not in name:
            continue
        configs.append(os.path.join(config_dir, name))
    return configs


def load_snapshot_labels(npz_path: str):
    if not npz_path:
        return None
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    if "snapshot_labels" in data.files:
        return [str(x) for x in data["snapshot_labels"].tolist()]
    labels = [k[len("counts__"):] for k in data.files if k.startswith("counts__")]
    labels.sort()
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build per-sweep step->time lookup NPZ")
    parser.add_argument("--config-dir", required=True, help="configs/runs/nXXX_batch")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument(
        "--align-with",
        default=None,
        help="Optional cell-count evolution NPZ to align snapshot order",
    )
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        print(f"Config directory not found: {config_dir}", file=sys.stderr)
        return 1

    label_order = load_snapshot_labels(args.align_with)

    entries = {}
    for cfg_path in scan_configs(config_dir):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        step = int(cfg.get("step"))
        dataset_path = cfg.get("dataset_path")
        if not dataset_path:
            print(f"Warning: missing dataset_path in {cfg_path}", file=sys.stderr)
            continue

        filename = os.path.basename(cfg_path)
        variant = "final" if "final" in filename else "main"

        output_dir = cfg.get("output_dir")
        if output_dir:
            label = os.path.basename(output_dir.rstrip("/"))
        else:
            label = f"conn6_T0p02_step{step:05d}"
            if variant == "final":
                label = f"conn6_T0p02_final_step{step:05d}"

        try:
            time_val = read_time(dataset_path, step)
        except Exception as e:
            print(f"Warning: failed to read time for {cfg_path}: {e}", file=sys.stderr)
            continue

        entries[label] = {
            "step": step,
            "time": time_val,
            "variant": variant,
        }

    if not entries:
        print("No entries found; nothing to write.", file=sys.stderr)
        return 1

    if label_order is None:
        label_order = sorted(entries.keys(), key=lambda k: ("final" in k, k))

    snapshot_labels = []
    step_numbers = []
    sim_time = []
    variants = []

    for label in label_order:
        if label not in entries:
            print(f"Warning: label {label} not found in configs", file=sys.stderr)
            continue
        snapshot_labels.append(label)
        step_numbers.append(entries[label]["step"])
        sim_time.append(entries[label]["time"])
        variants.append(entries[label]["variant"])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        snapshot_labels=np.array(snapshot_labels, dtype=object),
        step_numbers=np.array(step_numbers, dtype=int),
        sim_time=np.array(sim_time, dtype=float),
        variant=np.array(variants, dtype=object),
    )

    print(f"Wrote {args.output} ({len(snapshot_labels)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
