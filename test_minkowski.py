#!/usr/bin/env python3
"""Test Minkowski functional computation for simple geometric shapes."""

import numpy as np

# Import the functions to test
from metrics import (
    _get_curvature_weights,
    _get_euler_lut_6conn,
    minkowski_functionals_single_pass,
    minkowski_shapefinders,
    exposed_area,
)


def create_cube(size, grid_size=None):
    """Create a 3D label array with a centered cube."""
    if grid_size is None:
        grid_size = size + 4  # Add padding
    labels = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)
    start = (grid_size - size) // 2
    labels[start:start+size, start:start+size, start:start+size] = 1
    return labels


def create_sphere(radius, grid_size=None):
    """Create a 3D label array with a centered sphere."""
    if grid_size is None:
        grid_size = int(radius * 2.5 + 4)
    labels = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)
    center = grid_size // 2
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                r2 = (i - center)**2 + (j - center)**2 + (k - center)**2
                if r2 <= radius**2:
                    labels[i, j, k] = 1
    return labels


def test_curvature_weights():
    """Test that curvature weights are reasonable."""
    print("=" * 60)
    print("Testing curvature weights lookup table")
    print("=" * 60)

    weights = _get_curvature_weights(1.0, 1.0, 1.0)

    # Config 0 (empty) and 255 (full) should have 0 curvature
    print(f"Config 0 (empty):    C = {weights[0]:.6f} (expected: 0.0)")
    print(f"Config 255 (full):   C = {weights[255]:.6f} (expected: 0.0)")
    assert abs(weights[0]) < 1e-10, "Empty config should have 0 curvature"
    assert abs(weights[255]) < 1e-10, "Full config should have 0 curvature"

    # Single corner (config 1): should have positive curvature (convex)
    # 3 edges emanating from this corner, each contributes +π/4
    # But divided by 2 for double-counting
    expected_single = 3 * (np.pi / 4.0) / 2.0
    print(f"Config 1 (corner):   C = {weights[1]:.6f} (expected: {expected_single:.6f}, 3 convex edges)")

    # Config 254 (all but one corner): should have negative curvature (concave)
    # This is the complement of config 1
    print(f"Config 254 (7 on):   C = {weights[254]:.6f} (expected: {-expected_single:.6f}, 3 concave edges)")

    # Symmetry check: complementary configs should have opposite curvature
    for config in range(128):
        complement = 255 - config
        if abs(weights[config] + weights[complement]) > 1e-10:
            print(f"  Warning: config {config} and {complement} not symmetric: "
                  f"{weights[config]:.6f} vs {weights[complement]:.6f}")

    print()
    return True


def test_cube():
    """Test Minkowski functionals for a cube."""
    print("=" * 60)
    print("Testing cube")
    print("=" * 60)

    for size in [2, 3, 4, 5, 8, 10]:
        labels = create_cube(size)

        # Compute Minkowski functionals
        euler, curvature = minkowski_functionals_single_pass(labels, K=1)
        surface = exposed_area(labels, 1.0, 1.0, 1.0, K=1)
        volume = np.sum(labels == 1)

        # Analytical values for a cube of side L (in voxel units)
        L = size
        expected_volume = L**3
        expected_surface = 6 * L**2
        expected_euler = 1  # Simply connected
        # For a discrete cube: 12 edges × L × (π/4) = 3πL
        expected_curvature = 3 * np.pi * L

        print(f"\nCube size {size}×{size}×{size}:")
        print(f"  Volume:    {volume:10.2f} (expected: {expected_volume:.2f})")
        print(f"  Surface:   {surface[0]:10.2f} (expected: {expected_surface:.2f})")
        print(f"  Euler:     {euler[0]:10.2f} (expected: {expected_euler:.2f})")
        print(f"  Curvature: {curvature[0]:10.2f} (expected: {expected_curvature:.2f})")

        # Compute shapefinders
        try:
            sf = minkowski_shapefinders(
                volume=np.array([volume], dtype=np.float64),
                area=surface,
                curvature=curvature,
                euler_chi=euler
            )
            print(f"  Shapefinders: thickness={sf['thickness'][0]:.3f}, breadth={sf['breadth'][0]:.3f}, length={sf['length'][0]:.3f}")
            print(f"                planarity={sf['planarity'][0]:.3f}, filamentarity={sf['filamentarity'][0]:.3f}")
        except Exception as e:
            print(f"  Shapefinders error: {e}")


def test_sphere():
    """Test Minkowski functionals for a discretized sphere."""
    print("\n" + "=" * 60)
    print("Testing sphere")
    print("=" * 60)

    for radius in [5, 10, 15, 20]:
        labels = create_sphere(radius)

        # Compute Minkowski functionals
        euler, curvature = minkowski_functionals_single_pass(labels, K=1)
        surface = exposed_area(labels, 1.0, 1.0, 1.0, K=1)
        volume = np.sum(labels == 1)

        # Analytical values for a sphere of radius R (continuous)
        R = radius
        expected_volume = (4/3) * np.pi * R**3
        expected_surface = 4 * np.pi * R**2
        expected_euler = 2  # For closed surface
        expected_curvature = 4 * np.pi * R

        print(f"\nSphere radius {radius}:")
        print(f"  Volume:    {volume:10.2f} (continuous: {expected_volume:.2f}, ratio: {volume/expected_volume:.3f})")
        print(f"  Surface:   {surface[0]:10.2f} (continuous: {expected_surface:.2f}, ratio: {surface[0]/expected_surface:.3f})")
        print(f"  Euler:     {euler[0]:10.2f} (expected: {expected_euler:.2f})")
        print(f"  Curvature: {curvature[0]:10.2f} (continuous: {expected_curvature:.2f}, ratio: {curvature[0]/expected_curvature:.3f})")

        # Compute shapefinders
        try:
            sf = minkowski_shapefinders(
                volume=np.array([volume], dtype=np.float64),
                area=surface,
                curvature=curvature,
                euler_chi=euler
            )
            print(f"  Shapefinders: thickness={sf['thickness'][0]:.3f}, breadth={sf['breadth'][0]:.3f}, length={sf['length'][0]:.3f}")
            print(f"                planarity={sf['planarity'][0]:.3f}, filamentarity={sf['filamentarity'][0]:.3f}")
            print(f"  (For a sphere: P≈0, F≈0, thickness≈breadth≈length)")
        except Exception as e:
            print(f"  Shapefinders error: {e}")


def test_euler_lut():
    """Test the Euler characteristic lookup table."""
    print("\n" + "=" * 60)
    print("Testing Euler LUT")
    print("=" * 60)

    lut = _get_euler_lut_6conn()

    # Key configurations
    print(f"Config 0 (empty):  χ = {lut[0]:.4f} (expected: 0)")
    print(f"Config 255 (full): χ = {lut[255]:.4f} (expected: 0)")
    print(f"Config 1 (corner): χ = {lut[1]:.4f} (expected: 1/8 = 0.125)")
    print(f"Config 3 (edge):   χ = {lut[3]:.4f} (expected: 0)")
    print(f"Config 15 (face):  χ = {lut[15]:.4f} (expected: 0)")

    # Sum over all configs
    total = lut.sum()
    print(f"\nSum of all LUT entries: {total:.4f}")


def test_single_voxel():
    """Test a single isolated voxel."""
    print("\n" + "=" * 60)
    print("Testing single voxel")
    print("=" * 60)

    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[2, 2, 2] = 1

    euler, curvature = minkowski_functionals_single_pass(labels, K=1)
    surface = exposed_area(labels, 1.0, 1.0, 1.0, K=1)
    volume = 1

    print(f"Single voxel:")
    print(f"  Volume:    {volume}")
    print(f"  Surface:   {surface[0]:.4f} (expected: 6)")
    print(f"  Euler:     {euler[0]:.4f} (expected: 1)")
    # Single voxel has 8 corners, each with config=1, each contributes curvature
    # Each corner has 3 convex edges at π/4, divided by 2 = 3π/8
    # 8 corners × 3π/8 = 3π
    expected_curv = 3 * np.pi
    print(f"  Curvature: {curvature[0]:.4f} (expected: {expected_curv:.4f} = 3π)")


def main():
    test_euler_lut()
    test_curvature_weights()
    test_single_voxel()
    test_cube()
    test_sphere()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
