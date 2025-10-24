from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def aggregate(input_dir: str, output_path: str) -> Dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(input_dir, "clumps_rank*.npz")))
    if len(files) == 0:
        raise FileNotFoundError("No clumps_rank*.npz found in input directory")

    arrays: Dict[str, List[np.ndarray]] = {}
    gids: List[np.ndarray] = []
    ranks: List[np.ndarray] = []

    for f in files:
        d = load_npz(f)
        rank = int(d.get("rank", np.array([-1])))
        local_ids = d["label_ids"].astype(np.uint64)
        gid = (np.uint64(rank) << np.uint64(32)) | local_ids.astype(np.uint64)
        gids.append(gid)
        ranks.append(np.full_like(local_ids, rank, dtype=np.int32))

        # Only carry per-clump arrays (first dimension equals K)
        K = int(local_ids.shape[0])
        for k, v in d.items():
            if k in ("rank", "label_ids"):
                continue
            if isinstance(v, np.ndarray) and v.shape[:1] == (K,):
                arrays.setdefault(k, []).append(v)

    out: Dict[str, np.ndarray] = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    out["gid"] = np.concatenate(gids, axis=0)
    out["rank"] = np.concatenate(ranks, axis=0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory with clumps_rank*.npz files")
    ap.add_argument("--output", required=True, help="output master npz path")
    ap.add_argument("--sidecar", default=None, help="optional JSON sidecar for metadata/config")
    args = ap.parse_args()

    out = aggregate(args.input, args.output)

    if args.sidecar:
        parts = sorted(glob.glob(os.path.join(args.input, "clumps_rank*.npz")))
        meta = {
            "source_dir": os.path.abspath(args.input),
            "num_parts": len(parts),
            "parts": [os.path.basename(p) for p in parts],
            "output": os.path.abspath(args.output),
            "clump_count": int(out.get("gid", np.array([])).size),
        }
        with open(args.sidecar, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Wrote {args.output} with {out['gid'].size} clumps.")


if __name__ == "__main__":
    main()
