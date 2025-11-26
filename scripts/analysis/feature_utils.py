#!/usr/bin/env python3
"""Helpers for loading and normalising clump feature arrays."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np

FeatureDict = Dict[str, np.ndarray]
FeatureGetter = Callable[[np.lib.npyio.NpzFile], np.ndarray]


def _velocity_mean_component(data: np.lib.npyio.NpzFile, idx: int) -> np.ndarray:
    if "velocity_mean" not in data:
        raise KeyError("velocity_mean")
    vec = np.asarray(data["velocity_mean"], dtype=np.float64)
    if vec.ndim != 2 or vec.shape[1] <= idx:
        raise KeyError("velocity_mean")
    return vec[:, idx]


def _velocity_speed(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "velocity_mean" not in data:
        raise KeyError("velocity_mean")
    vec = np.asarray(data["velocity_mean"], dtype=np.float64)
    if vec.ndim == 1:
        return vec
    if vec.ndim == 2:
        return np.linalg.norm(vec, axis=1)
    raise KeyError("velocity_mean")


def _cell_count(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "cell_count" in data:
        return np.asarray(data["cell_count"], dtype=np.float64)
    if "num_cells" in data:
        return np.asarray(data["num_cells"], dtype=np.float64)
    raise KeyError("cell_count")


DERIVED_FEATURE_GETTERS: Dict[str, FeatureGetter] = {
    "velocity_mean_mag": _velocity_speed,
    "velocity_speed": _velocity_speed,
    "speed": _velocity_speed,
    "velocity_mean_x": lambda d: _velocity_mean_component(d, 0),
    "velocity_mean_y": lambda d: _velocity_mean_component(d, 1),
    "velocity_mean_z": lambda d: _velocity_mean_component(d, 2),
    "vx_mean": lambda d: _velocity_mean_component(d, 0),
    "vy_mean": lambda d: _velocity_mean_component(d, 1),
    "vz_mean": lambda d: _velocity_mean_component(d, 2),
    "velocity_std": lambda d: np.asarray(d["velocity_std"], dtype=np.float64),
    "vx_std": lambda d: np.asarray(d["velocity_std"], dtype=np.float64),
    "vy_std": lambda d: np.asarray(d["velocity_std"], dtype=np.float64),
    "vz_std": lambda d: np.asarray(d["velocity_std"], dtype=np.float64),
    "cell_count": _cell_count,
    "num_cells": _cell_count,
}


def _flatten_feature(name: str, value: np.ndarray) -> np.ndarray:
    if value.ndim == 1:
        return value
    if name == "principal_axes_lengths":
        return np.max(value, axis=1)
    if name == "axis_ratios":
        return value[:, 0]
    if value.ndim == 2 and value.shape[1] == 1:
        return value[:, 0]
    raise ValueError(f"Cannot flatten feature {name} with shape {value.shape}")


def _extract_feature(data: np.lib.npyio.NpzFile, feature: str) -> np.ndarray:
    if feature in data:
        arr = np.asarray(data[feature], dtype=np.float64)
        return _flatten_feature(feature, arr) if arr.ndim > 1 else arr
    if feature in DERIVED_FEATURE_GETTERS:
        arr = np.asarray(DERIVED_FEATURE_GETTERS[feature](data), dtype=np.float64)
        return _flatten_feature(feature, arr) if arr.ndim > 1 else arr
    raise KeyError(feature)


def load_features(files: Iterable[str], features: List[str]) -> FeatureDict:
    """Load and stack requested features from multiple clump master NPZ files."""
    if not features:
        raise ValueError("No features requested.")

    stacked: Dict[str, List[np.ndarray]] = {feat: [] for feat in features}
    missing: set[str] = set()

    for path in files:
        with np.load(path) as data:
            for feat in features:
                try:
                    arr = _extract_feature(data, feat)
                except KeyError:
                    missing.add(feat)
                    continue
                stacked[feat].append(arr.astype(np.float64, copy=False))

    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(
            f"Missing features: {missing_list}. "
            "Re-run clump_finder with --extra-stats or choose available features."
        )

    return {feat: np.concatenate(arrays, axis=0) for feat, arrays in stacked.items()}


def available_features(path: str) -> List[str]:
    """Return available raw feature names for inspection (first file is representative)."""
    with np.load(path) as data:
        keys = set(data.files)
    keys.update(DERIVED_FEATURE_GETTERS.keys())
    return sorted(keys)


__all__ = ["load_features", "available_features"]
