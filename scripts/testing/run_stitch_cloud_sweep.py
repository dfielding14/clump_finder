#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stitch_cloud_test_utils import assert_single_cloud_result
from stitch_cloud_test_utils import center_from_fraction
from stitch_cloud_test_utils import run_single_cloud_case


CENTER_PRESETS = {
    "interior": (0.50, 0.50, 0.50),
    "wrap_x": (0.04, 0.50, 0.50),
    "wrap_xy": (0.95, 0.06, 0.50),
    "wrap_xyz": (0.96, 0.95, 0.04),
}


def _parse_axes_frac(text: str) -> tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("--axes-frac must contain three comma-separated values")
    return (vals[0], vals[1], vals[2])


def _parse_partitions(text: str) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for item in text.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid partition '{item}', expected format PxPyPz like 2x2x1")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def _parse_centers(text: str) -> list[tuple[float, float, float]]:
    centers = []
    for key in (x.strip() for x in text.split(",")):
        if key not in CENTER_PRESETS:
            raise ValueError(f"Unknown center preset '{key}'. Choices: {sorted(CENTER_PRESETS)}")
        centers.append(CENTER_PRESETS[key])
    return centers


def _shape_list(shape: str) -> list[str]:
    shape = shape.lower()
    if shape == "both":
        return ["sphere", "ellipsoid"]
    if shape in ("sphere", "ellipsoid"):
        return [shape]
    raise ValueError("--shape must be one of: sphere, ellipsoid, both")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep synthetic single-cloud stitch tests.")
    ap.add_argument("--N", nargs="+", type=int, default=[64], help="Domain sizes (e.g. --N 64 128 256)")
    ap.add_argument("--shape", default="both", choices=["sphere", "ellipsoid", "both"])
    ap.add_argument("--radius-frac", type=float, default=0.20, help="Sphere radius as fraction of N")
    ap.add_argument(
        "--axes-frac",
        default="0.22,0.14,0.10",
        help="Ellipsoid semi-axes as fractions of N: ax,ay,az",
    )
    ap.add_argument("--rotation-deg-z", type=float, default=20.0, help="Ellipsoid rotation angle about z")
    ap.add_argument(
        "--centers",
        default="interior,wrap_x,wrap_xy,wrap_xyz",
        help="Comma-separated center presets: interior,wrap_x,wrap_xy,wrap_xyz",
    )
    ap.add_argument(
        "--partitions",
        default="1x1x1,2x2x2",
        help="Comma-separated partition specs (e.g. 1x1x1,2x2x2,4x2x2)",
    )
    ap.add_argument("--max-cases", type=int, default=0, help="Optional hard limit on total cases")
    ap.add_argument(
        "--keep-failures",
        default=None,
        help="Directory to save failing-case parts and stitched outputs",
    )
    args = ap.parse_args()

    axes_frac = _parse_axes_frac(args.axes_frac)
    centers = _parse_centers(args.centers)
    partitions = _parse_partitions(args.partitions)
    shapes = _shape_list(args.shape)
    keep_failures_dir = Path(args.keep_failures) if args.keep_failures else None
    if keep_failures_dir is not None:
        keep_failures_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0
    failures: list[dict] = []

    for n in args.N:
        for kind in shapes:
            for center_frac in centers:
                center = center_from_fraction(int(n), center_frac)
                for partition in partitions:
                    if args.max_cases > 0 and total >= args.max_cases:
                        break
                    total += 1
                    tag = f"N{n}_{kind}_c{center_frac[0]:.2f}-{center_frac[1]:.2f}-{center_frac[2]:.2f}_p{partition[0]}x{partition[1]}x{partition[2]}"
                    kwargs = {
                        "n": int(n),
                        "center": center,
                        "partition": partition,
                        "kind": kind,
                    }
                    if kind == "sphere":
                        kwargs["radius"] = float(args.radius_frac) * float(n)
                    else:
                        kwargs["axes"] = tuple(float(a) * float(n) for a in axes_frac)
                        kwargs["rotation_deg_z"] = float(args.rotation_deg_z)

                    try:
                        result = run_single_cloud_case(**kwargs)
                        assert_single_cloud_result(result)
                        passed += 1
                        print(f"[PASS] {tag}")
                    except Exception as exc:
                        fail_info = {"case": tag, "error": str(exc)}
                        if keep_failures_dir is not None:
                            case_dir = keep_failures_dir / tag
                            case_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                rerun = dict(kwargs)
                                rerun["keep_outputs"] = True
                                rerun["out_dir"] = str(case_dir)
                                run_single_cloud_case(**rerun)
                            except Exception:
                                # We still keep metadata for debugging even if rerun fails.
                                pass
                            fail_info["saved_dir"] = str(case_dir)
                        failures.append(fail_info)
                        print(f"[FAIL] {tag}: {exc}")
                if args.max_cases > 0 and total >= args.max_cases:
                    break
            if args.max_cases > 0 and total >= args.max_cases:
                break
        if args.max_cases > 0 and total >= args.max_cases:
            break

    print(f"\nSummary: passed={passed} failed={len(failures)} total={total}")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"  - {item['case']}: {item['error']}")
            if "saved_dir" in item:
                print(f"    saved: {item['saved_dir']}")
        report_path = None
        if keep_failures_dir is not None:
            report_path = keep_failures_dir / "sweep_failures.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2)
            print(f"Wrote failure report: {report_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
