# Clump Shape Metrics Reference

This document describes the shape and morphology metrics computed for each clump, including mathematical definitions, physical interpretation, and known limitations.

## Table of Contents
1. [Basic Quantities](#basic-quantities)
2. [Covariance Tensor & Principal Axes](#covariance-tensor--principal-axes)
3. [Shape Metrics from Principal Axes](#shape-metrics-from-principal-axes)
4. [Minkowski Functionals](#minkowski-functionals)
5. [Minkowski Shapefinders](#minkowski-shapefinders)
6. [Known Limitations](#known-limitations)

---

## Basic Quantities

### Cell Count
- **Definition**: Number of voxels in the clump
- **Symbol**: `N`

### Volume
- **Definition**: `V = N × dx × dy × dz` (physical volume)
- **Units**: Length³

### Surface Area
- **Definition**: `A = Σ (exposed faces) × face_area`
- **Method**: Count faces between labeled and unlabeled voxels
- **Units**: Length²

### Mass
- **Definition**: `M = Σ ρ_i × dV` (sum of density × voxel volume)
- **Units**: Mass

### Centroid
- **Volume-weighted**: `x̄_vol = Σ x_i / N`
- **Mass-weighted**: `x̄_mass = Σ ρ_i x_i / Σ ρ_i`

---

## Covariance Tensor & Principal Axes

### Mass-Weighted Covariance Tensor

The covariance tensor describes the spatial distribution of mass:

```
C_ij = (1/M) Σ_k m_k (x_i^k - μ_i)(x_j^k - μ_j)
```

where:
- `m_k` = mass of voxel k
- `x_i^k` = position of voxel k in dimension i
- `μ_i` = mass-weighted centroid in dimension i

**Relation to Moment of Inertia Tensor:**

The moment of inertia tensor `I` and covariance tensor `C` share the same eigenvectors (principal axes). Their eigenvalues are related by:
- `I_1 = M(λ_2 + λ_3)`
- `I_2 = M(λ_1 + λ_3)`
- `I_3 = M(λ_1 + λ_2)`

where λ₁ ≥ λ₂ ≥ λ₃ are covariance eigenvalues. For shape ratios, both tensors give equivalent results.

### Principal Axis Lengths

Eigendecomposition of C gives eigenvalues λ₁ ≥ λ₂ ≥ λ₃. The raw RMS extents are:

```
a_raw = √λ₁, b_raw = √λ₂, c_raw = √λ₃
```

We then **normalize** so that `a × b × c = V` (volume):

```
scale = (V / (a_raw × b_raw × c_raw))^(1/3)
a = a_raw × scale
b = b_raw × scale
c = c_raw × scale
```

**Why normalize?** Raw eigenvalues give RMS extents (standard deviations), not physical dimensions. A uniform box of side L has σ = L/√12 ≈ 0.29L. Normalizing to volume gives more intuitive absolute lengths while **preserving all ratios** (b/a, c/a, triaxiality, elongation unchanged).

**Physical meaning**: After normalization, a, b, c represent the semi-axes of an equivalent "ellipsoid-like" shape with the same volume and orientation.

---

## Shape Metrics from Principal Axes

### Triaxiality (T)

```
T = (a² - b²) / (a² - c²)
```

| Value | Shape |
|-------|-------|
| T = 0 | Oblate (pancake): a = b > c |
| T = 0.5 | Triaxial |
| T = 1 | Prolate (cigar): a > b = c |

**Limitation**: Undefined when a = c (perfect sphere). We add ε to denominator.

### Elongation

```
elongation = a / c
```

| Value | Shape |
|-------|-------|
| 1 | Spherical |
| 3-10 | Moderately elongated |
| >100 | Highly filamentary |

**Limitation**: For thin structures (1 voxel thick), c → 0 leading to infinite elongation. We clamp `c ≥ 0.5` (half voxel) to get meaningful values. A 1×1×N filament thus has elongation ≈ 2N.

### Axis Ratios

```
b/a = intermediate / longest    (0 to 1)
c/a = shortest / longest        (0 to 1)
```

### Sphericity (Isoperimetric Ratio)

```
ψ = π^(1/3) × (6V)^(2/3) / A
```

| Value | Shape |
|-------|-------|
| 1 | Perfect sphere (minimizes A for given V) |
| <1 | More surface area than a sphere |

**Physical meaning**: Measures how efficiently the clump encloses volume.

### Compactness

```
compactness = 36π V² / A³
```

Normalized isoperimetric compactness. This equals 1 for a sphere and is < 1 for less compact shapes.

### Effective Radius

```
r_eff = (3V / 4π)^(1/3)
```

Radius of a sphere with the same volume.

---

## Minkowski Functionals

The four Minkowski functionals in 3D completely characterize the morphology of a shape:

### M₀: Volume
```
V = N × dx × dy × dz
```

### M₁: Surface Area
```
A = Σ (exposed faces)
```

### M₂: Integrated Mean Curvature
```
C = ∫ (κ₁ + κ₂)/2 dA
```

For discrete voxels, computed by summing contributions from edges:
- Each edge between voxels contributes `(π - θ)/2 × edge_length`
- θ = dihedral angle (π/2 for 90° edges)

**Physical meaning**: Total "bending" of the surface. For a sphere of radius R: C = 4πR.

### M₃: Euler Characteristic (χ)
```
χ = V - E + F - C  (vertices - edges + faces - cells in boundary complex)
```

For discrete voxels, computed using the 2×2×2 corner contribution method.

| Value | Topology |
|-------|----------|
| χ = 1 | Simply connected (no holes) |
| χ = 0 | One tunnel (torus-like) |
| χ < 0 | Multiple tunnels/cavities |

---

## Minkowski Shapefinders

Following Sahni et al. (1998), we define characteristic length scales:

### Thickness (T)
```
T = 3V / A
```
Related to the "thinnest" dimension.

### Breadth (B)
```
B = A / C
```
Related to the "middle" dimension.

### Length (L)
```
L = C / (4π)
```
Related to the "longest" dimension.

**For a sphere of radius R**: T = B = L = R (all equal).

### Planarity (P)
```
P = (B - T) / (B + T)
```

| Value | Shape |
|-------|-------|
| P ≈ 0 | Isotropic (sphere-like or filament-like) |
| P ≈ 1 | Planar (pancake/sheet) |

### Filamentarity (F)
```
F = (L - B) / (L + B)
```

| Value | Shape |
|-------|-------|
| F ≈ 0 | Isotropic (sphere-like or pancake-like) |
| F ≈ 1 | Filamentary (elongated tube) |

### P-F Diagram

The (P, F) plane classifies shapes:
- **(0, 0)**: Sphere
- **(1, 0)**: Pancake (oblate)
- **(0, 1)**: Filament (prolate)
- **(1, 1)**: Ribbon (both planar and elongated)

---

## Known Limitations

### 1. Discrete Voxel Effects

All metrics are computed on discrete voxel data. For small clumps (< 100 cells), discretization noise dominates.

### 2. Covariance Tensor Degeneracy

For thin structures:
- **1D filament** (1×1×N): Two eigenvalues ≈ 0
- **2D sheet** (1×N×M): One eigenvalue ≈ 0

This causes:
- Triaxiality → undefined (0/0)
- Elongation → infinity

**Mitigation**: We clamp c ≥ 0.5 for elongation calculation.

### 3. Minkowski Functionals for Boundary Clumps

Clumps touching domain boundaries have incomplete surfaces. We only compute Minkowski functionals for **interior clumps** (fully contained within node's tile).

### 4. Stitched Clumps

After stitching clumps across nodes:
- ✅ Volume, Area, Mass, Centroids — correctly aggregated
- ✅ Covariance tensor — correctly aggregated (additive moments)
- ✅ Triaxiality, Elongation, Sphericity — recomputed from aggregated data
- ❌ Integrated Curvature, Euler characteristic — NOT preserved (requires voxel-level boundary info)
- ❌ Minkowski Shapefinders (T, B, L, P, F) — NOT available for stitched clumps

### 5. Connectivity Dependence

6-connectivity (face neighbors only) produces more, smaller clumps than 18 or 26-connectivity. Shape metrics depend on connectivity choice.

---

## References

- Sahni, V., Sathyaprakash, B. S., & Shandarin, S. F. (1998). "Shapefinders: A new shape diagnostic for large-scale structure." *ApJ*, 495, L5.
- Schmalzing, J., & Buchert, T. (1997). "Beyond genus statistics: A unifying approach to the morphology of cosmic structure." *ApJ*, 482, L1.
