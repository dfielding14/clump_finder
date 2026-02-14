#!/usr/bin/env python3
"""
Build lookup dictionaries mapping step -> simulation time for each sweep.

Reads config files in configs/runs/n*_batch, extracts dataset_path and step,
then queries the openPMD (ADIOS2) attribute /data/<step>/time to get the
simulation time.

Outputs a JSON file with structure:
{
  "n320": {
    "step00021": {"step": 21, "time": 5.25, "variant": "main", ...},
    ...
  },
  "n10240": {
    "step00034": {...},
    "final_step00034": {...}
  }
}
"""

import argparse
import json
import os
import re
import sys
import yaml
import adios2

SWEEPS = [320, 640, 1280, 2560, 5120, 10240]

STEP_RE = re.compile(r"step0*(\d+)")


def read_time(dataset_path: str, step: int) -> float:
    path = dataset_path.rstrip("/")
    fr = adios2.FileReader(path)
    try:
        attr = f"/data/{step}/time"
        time_val = fr.read_attribute(attr)
    finally:
        fr.close()

    # adios2 returns numpy scalars or lists
    if isinstance(time_val, (list, tuple)) and len(time_val) == 1:
        return float(time_val[0])
    try:
        return float(time_val)
    except Exception:
        return float(time_val[()])


def scan_sweep_configs(config_dir: str):
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


def main():
    parser = argparse.ArgumentParser(description="Build step->time lookup for sweeps")
    parser.add_argument(
        "--repo-root",
        default="/lustre/orion/ast207/proj-shared/mpturb/clump_find_drummond",
        help="Repository root",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: clump_out/step_time_lookup.json)",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(repo_root, "clump_out", "step_time_lookup.json")

    lookup = {}
    missing = []

    for n in SWEEPS:
        config_dir = os.path.join(repo_root, "configs", "runs", f"n{n}_batch")
        if not os.path.isdir(config_dir):
            continue
        entries = {}
        for cfg_path in scan_sweep_configs(config_dir):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            step = int(cfg.get("step"))
            dataset_path = cfg.get("dataset_path")
            output_dir = cfg.get("output_dir")
            if not dataset_path:
                missing.append(cfg_path)
                continue

            filename = os.path.basename(cfg_path)
            variant = "final" if "final" in filename else "main"

            try:
                time_val = read_time(dataset_path, step)
            except Exception as e:
                print(f"Warning: failed to read time for {cfg_path}: {e}", file=sys.stderr)
                missing.append(cfg_path)
                continue

            key = f"step{step:05d}"
            if n == 10240 and variant == "final":
                key = f"final_step{step:05d}"
            entry = {
                "step": step,
                "time": time_val,
                "variant": variant,
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "config": cfg_path,
            }
            entries[key] = entry

        lookup[f"n{n}"] = entries

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    print(f"Wrote {out_path}")
    if missing:
        print(f"Warnings for {len(missing)} configs (see stderr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
