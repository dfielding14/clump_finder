from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stitch_cloud_test_utils import assert_single_cloud_result
from stitch_cloud_test_utils import center_from_fraction
from stitch_cloud_test_utils import run_single_cloud_case


SMALL_PARTITIONS = [
    (1, 1, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (3, 2, 1),
]

SMALL_CASES = [
    {"name": "sphere_n32_center", "n": 32, "kind": "sphere", "radius_frac": 0.20, "center_frac": (0.50, 0.50, 0.50)},
    {"name": "sphere_n32_wrap_x", "n": 32, "kind": "sphere", "radius_frac": 0.20, "center_frac": (0.05, 0.50, 0.50)},
    {"name": "sphere_n48_wrap_xyz", "n": 48, "kind": "sphere", "radius_frac": 0.22, "center_frac": (0.96, 0.95, 0.04)},
    {"name": "sphere_n64_wrap_xy", "n": 64, "kind": "sphere", "radius_frac": 0.24, "center_frac": (0.94, 0.07, 0.50)},
    {
        "name": "ellipsoid_n32_center",
        "n": 32,
        "kind": "ellipsoid",
        "axes_frac": (0.24, 0.16, 0.11),
        "rotation_deg_z": 0.0,
        "center_frac": (0.50, 0.50, 0.50),
    },
    {
        "name": "ellipsoid_n48_wrap_x_rot",
        "n": 48,
        "kind": "ellipsoid",
        "axes_frac": (0.23, 0.15, 0.10),
        "rotation_deg_z": 25.0,
        "center_frac": (0.04, 0.50, 0.50),
    },
    {
        "name": "ellipsoid_n64_wrap_xyz_rot",
        "n": 64,
        "kind": "ellipsoid",
        "axes_frac": (0.22, 0.14, 0.09),
        "rotation_deg_z": 40.0,
        "center_frac": (0.95, 0.96, 0.05),
    },
]

LARGE_CASES = [
    {
        "name": "sphere_n256_center",
        "n": 256,
        "kind": "sphere",
        "radius_frac": 0.20,
        "center_frac": (0.50, 0.50, 0.50),
        "partitions": [(1, 1, 1), (2, 2, 2), (4, 2, 2)],
    },
    {
        "name": "sphere_n256_wrap_xyz",
        "n": 256,
        "kind": "sphere",
        "radius_frac": 0.18,
        "center_frac": (0.96, 0.95, 0.04),
        "partitions": [(1, 1, 1), (2, 2, 2)],
    },
    {
        "name": "ellipsoid_n256_center_rot",
        "n": 256,
        "kind": "ellipsoid",
        "axes_frac": (0.22, 0.14, 0.10),
        "rotation_deg_z": 15.0,
        "center_frac": (0.50, 0.50, 0.50),
        "partitions": [(1, 1, 1), (2, 2, 2), (4, 2, 2)],
    },
    {
        "name": "ellipsoid_n256_wrap_x_rot",
        "n": 256,
        "kind": "ellipsoid",
        "axes_frac": (0.20, 0.13, 0.09),
        "rotation_deg_z": 35.0,
        "center_frac": (0.04, 0.50, 0.50),
        "partitions": [(1, 1, 1), (2, 2, 2)],
    },
]

STRESS_CASES = [
    {
        "name": "sphere_n512_center",
        "n": 512,
        "kind": "sphere",
        "radius_frac": 0.19,
        "center_frac": (0.50, 0.50, 0.50),
        "partitions": [(1, 1, 1), (2, 2, 2)],
    },
    {
        "name": "ellipsoid_n512_wrap_xyz_rot",
        "n": 512,
        "kind": "ellipsoid",
        "axes_frac": (0.21, 0.13, 0.09),
        "rotation_deg_z": 30.0,
        "center_frac": (0.96, 0.95, 0.04),
        "partitions": [(1, 1, 1), (2, 2, 2)],
    },
]


def _run_case(spec: dict, partition: tuple[int, int, int]):
    n = int(spec["n"])
    center = center_from_fraction(n, spec["center_frac"])
    kwargs = {
        "n": n,
        "center": center,
        "partition": partition,
        "kind": spec["kind"],
    }
    if spec["kind"] == "sphere":
        kwargs["radius"] = float(spec["radius_frac"]) * float(n)
    else:
        kwargs["axes"] = tuple(float(a) * float(n) for a in spec["axes_frac"])
        kwargs["rotation_deg_z"] = float(spec.get("rotation_deg_z", 0.0))

    result = run_single_cloud_case(**kwargs)
    assert_single_cloud_result(result)


@pytest.mark.parametrize("spec", SMALL_CASES, ids=[c["name"] for c in SMALL_CASES])
def test_single_cloud_stitch_small(spec: dict):
    for partition in SMALL_PARTITIONS:
        _run_case(spec, partition)


@pytest.mark.large
@pytest.mark.parametrize("spec", LARGE_CASES, ids=[c["name"] for c in LARGE_CASES])
def test_single_cloud_stitch_large(spec: dict):
    for partition in spec["partitions"]:
        _run_case(spec, partition)


@pytest.mark.stress
@pytest.mark.parametrize("spec", STRESS_CASES, ids=[c["name"] for c in STRESS_CASES])
def test_single_cloud_stitch_stress(spec: dict):
    for partition in spec["partitions"]:
        _run_case(spec, partition)
