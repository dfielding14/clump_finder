from __future__ import annotations

import numpy as np


def _ensure_K(labels: np.ndarray, K: int | None = None) -> int:
    if K is None:
        K = int(labels.max())
    return K


def num_cells(labels: np.ndarray, K: int | None = None) -> np.ndarray:
    K = _ensure_K(labels, K)
    lab = labels.ravel()
    cnt = np.bincount(lab, minlength=K + 1).astype(np.int64)
    return cnt[1:]


def volumes(cell_count: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    Vc = dx * dy * dz
    return cell_count.astype(np.float64) * Vc


def masses(labels: np.ndarray, dens: np.ndarray, dx: float, dy: float, dz: float, K: int | None = None) -> np.ndarray:
    K = _ensure_K(labels, K)
    Vc = dx * dy * dz
    lab = labels.ravel()
    w = (dens.ravel().astype(np.float64, copy=False)) * Vc
    M = np.bincount(lab, weights=w, minlength=K + 1)
    return M[1:]


def centroids(labels: np.ndarray,
              dens: np.ndarray,
              dx: float, dy: float, dz: float,
              origin: tuple[float, float, float],
              node_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
              K: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute volume- and mass-weighted centroids per label.

    Returns (centroid_vol[K,3], centroid_mass[K,3]).
    """
    K = _ensure_K(labels, K)
    if K == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    (i0, i1), (j0, j1), (k0, k1) = node_bbox
    ni, nj, nk = labels.shape
    Vc = dx * dy * dz

    # Prepare accumulators
    Wvol = np.zeros(K + 1, dtype=np.float64)
    Sxv = np.zeros(K + 1, dtype=np.float64)
    Syv = np.zeros(K + 1, dtype=np.float64)
    Szv = np.zeros(K + 1, dtype=np.float64)

    Wmass = np.zeros(K + 1, dtype=np.float64)
    Sxm = np.zeros(K + 1, dtype=np.float64)
    Sym = np.zeros(K + 1, dtype=np.float64)
    Szm = np.zeros(K + 1, dtype=np.float64)

    # Coordinates per axis (global positions of cell centers)
    xi = origin[0] + (np.arange(i0, i1) + 0.5) * dx
    yj = origin[1] + (np.arange(j0, j1) + 0.5) * dy
    zk = origin[2] + (np.arange(k0, k1) + 0.5) * dz

    # Accumulate along i for x-moments (volume-weighted) and mass-weighted using planes
    for i in range(ni):
        L = labels[i, :, :].ravel()
        if L.size == 0:
            continue
        # volume-weighted counts per plane
        cnt = np.bincount(L, minlength=K + 1)
        Wvol += cnt * Vc
        Sxv += (xi[i] * Vc) * cnt

        # mass-weighted per plane
        wplane = (dens[i, :, :].ravel().astype(np.float64, copy=False)) * Vc
        Wmass += np.bincount(L, weights=wplane, minlength=K + 1)
        Sxm += (xi[i]) * np.bincount(L, weights=wplane, minlength=K + 1)

    # Accumulate along j for y-moments
    for j in range(nj):
        L = labels[:, j, :].ravel()
        cnt = np.bincount(L, minlength=K + 1)
        Syv += (yj[j] * Vc) * cnt
        wplane = (dens[:, j, :].ravel().astype(np.float64, copy=False)) * Vc
        Sym += (yj[j]) * np.bincount(L, weights=wplane, minlength=K + 1)

    # Accumulate along k for z-moments
    for k in range(nk):
        L = labels[:, :, k].ravel()
        cnt = np.bincount(L, minlength=K + 1)
        Szv += (zk[k] * Vc) * cnt
        wplane = (dens[:, :, k].ravel().astype(np.float64, copy=False)) * Vc
        Szm += (zk[k]) * np.bincount(L, weights=wplane, minlength=K + 1)

    # Drop background and form centers
    small = 1e-300
    Wvol = Wvol[1:]
    Wmass = Wmass[1:]
    cx_vol = (Sxv[1:] / (Wvol + small))
    cy_vol = (Syv[1:] / (Wvol + small))
    cz_vol = (Szv[1:] / (Wvol + small))

    cx_mass = (Sxm[1:] / (Wmass + small))
    cy_mass = (Sym[1:] / (Wmass + small))
    cz_mass = (Szm[1:] / (Wmass + small))

    return np.stack([cx_vol, cy_vol, cz_vol], axis=1), np.stack([cx_mass, cy_mass, cz_mass], axis=1)


def exposed_area(labels: np.ndarray, dx: float, dy: float, dz: float, K: int | None = None) -> np.ndarray:
    """Compute exterior exposed face area per label (background neighbor = 0).

    No wrap-around within the node; boundary faces of the subvolume count as exposed.
    """
    K = _ensure_K(labels, K)
    if K == 0:
        return np.zeros((0,), dtype=np.float64)
    lab = labels
    ni, nj, nk = lab.shape
    face = np.zeros((K + 1,), dtype=np.float64)

    # X faces (between i and i+1)
    a_x = dy * dz
    # interior faces
    a = lab[:-1, :, :]
    b = lab[1:, :, :]
    ex = (a > 0) & (b == 0)
    em = (b > 0) & (a == 0)
    if ex.any():
        face += a_x * np.bincount(a[ex], minlength=K + 1)
    if em.any():
        face += a_x * np.bincount(b[em], minlength=K + 1)
    # boundary faces at i=0 and i=ni-1
    if ni > 0:
        edge0 = lab[0, :, :]
        edge1 = lab[ni - 1, :, :]
        face += a_x * np.bincount(edge0[edge0 > 0], minlength=K + 1)
        face += a_x * np.bincount(edge1[edge1 > 0], minlength=K + 1)

    # Y faces
    a_y = dx * dz
    a = lab[:, :-1, :]
    b = lab[:, 1:, :]
    ex = (a > 0) & (b == 0)
    em = (b > 0) & (a == 0)
    if ex.any():
        face += a_y * np.bincount(a[ex], minlength=K + 1)
    if em.any():
        face += a_y * np.bincount(b[em], minlength=K + 1)
    if nj > 0:
        edge0 = lab[:, 0, :]
        edge1 = lab[:, nj - 1, :]
        face += a_y * np.bincount(edge0[edge0 > 0], minlength=K + 1)
        face += a_y * np.bincount(edge1[edge1 > 0], minlength=K + 1)

    # Z faces
    a_z = dx * dy
    a = lab[:, :, :-1]
    b = lab[:, :, 1:]
    ex = (a > 0) & (b == 0)
    em = (b > 0) & (a == 0)
    if ex.any():
        face += a_z * np.bincount(a[ex], minlength=K + 1)
    if em.any():
        face += a_z * np.bincount(b[em], minlength=K + 1)
    if nk > 0:
        edge0 = lab[:, :, 0]
        edge1 = lab[:, :, nk - 1]
        face += a_z * np.bincount(edge0[edge0 > 0], minlength=K + 1)
        face += a_z * np.bincount(edge1[edge1 > 0], minlength=K + 1)

    return face[1:]


def per_label_stats(labels: np.ndarray,
                    X: np.ndarray,
                    weights: np.ndarray | None = None,
                    K: int | None = None,
                    excess_kurtosis: bool = False) -> tuple[np.ndarray, ...]:
    """Compute mean/std/skew/kurt per label for variable X.

    weights=None => unweighted (volume-weighted handled by multiplying by Vc outside).
    Uses numerically-stable grouped central moments via bincounts.
    """
    K = _ensure_K(labels, K)
    if K == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z, z

    lab = labels.ravel()
    x = X.ravel().astype(np.float64, copy=False)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = weights.ravel().astype(np.float64, copy=False)

    W = np.bincount(lab, weights=w, minlength=K + 1)[1:]
    S1 = np.bincount(lab, weights=w * x, minlength=K + 1)[1:]
    mu = S1 / (W + 1e-300)

    # central moments
    # Map mu to each element by label index: mu_idx = mu[lab-1] where lab>0
    # Create xc only for lab>0 to reduce overhead
    sel = lab > 0
    lab1 = lab[sel] - 1
    xsel = x[sel]
    wsel = w[sel]
    xc = xsel - mu[lab1]
    S2c = np.bincount(lab1, weights=wsel * (xc * xc), minlength=K)
    S3c = np.bincount(lab1, weights=wsel * (xc * xc * xc), minlength=K)
    S4c = np.bincount(lab1, weights=wsel * (xc * xc * xc * xc), minlength=K)

    var = S2c / (W + 1e-300)
    std = np.sqrt(var)
    skew = (S3c / (W + 1e-300)) / (std**3 + 1e-300)
    kurt = (S4c / (W + 1e-300)) / (var**2 + 1e-300)  # Pearson by default
    if excess_kurtosis:
        kurt = kurt - 3.0
    return mu, std, skew, kurt


def compute_bboxes(labels: np.ndarray,
                   node_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
                   K: int | None = None) -> np.ndarray:
    """Compute per-label bounding boxes in global indices [min,max) for i,j,k.

    Returns int32 array of shape [K,6]: (i_min,i_max,j_min,j_max,k_min,k_max)
    """
    K = _ensure_K(labels, K)
    if K == 0:
        return np.zeros((0, 6), dtype=np.int32)
    (i0, i1), (j0, j1), (k0, k1) = node_bbox
    ni, nj, nk = labels.shape
    # Initialize mins/maxs
    imin = np.full(K + 1, i1, dtype=np.int64)
    jmin = np.full(K + 1, j1, dtype=np.int64)
    kmin = np.full(K + 1, k1, dtype=np.int64)
    imax = np.full(K + 1, i0, dtype=np.int64)
    jmax = np.full(K + 1, j0, dtype=np.int64)
    kmax = np.full(K + 1, k0, dtype=np.int64)

    # Along i
    for i in range(ni):
        plane = labels[i, :, :]
        u = np.unique(plane)
        u = u[u != 0]
        if u.size == 0:
            continue
        gi = i0 + i
        imin[u] = np.minimum(imin[u], gi)
        imax[u] = np.maximum(imax[u], gi + 1)

    # Along j
    for j in range(nj):
        plane = labels[:, j, :]
        u = np.unique(plane)
        u = u[u != 0]
        if u.size == 0:
            continue
        gj = j0 + j
        jmin[u] = np.minimum(jmin[u], gj)
        jmax[u] = np.maximum(jmax[u], gj + 1)

    # Along k
    for k in range(nk):
        plane = labels[:, :, k]
        u = np.unique(plane)
        u = u[u != 0]
        if u.size == 0:
            continue
        gk = k0 + k
        kmin[u] = np.minimum(kmin[u], gk)
        kmax[u] = np.maximum(kmax[u], gk + 1)

    # Drop background
    out = np.stack([imin[1:], imax[1:], jmin[1:], jmax[1:], kmin[1:], kmax[1:]], axis=1).astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Tier 0: Derived shape metrics from existing V, S, principal axes
# ---------------------------------------------------------------------------

def derived_shape_metrics(volume: np.ndarray,
                          area: np.ndarray,
                          principal_axes_lengths: np.ndarray) -> dict[str, np.ndarray]:
    """Compute derived shape metrics from existing volume, surface area, and principal axes.

    Parameters
    ----------
    volume : (K,) array
        Clump volumes.
    area : (K,) array
        Clump surface areas.
    principal_axes_lengths : (K, 3) array
        Principal axis lengths (a >= b >= c) from inertia tensor eigenvalues.

    Returns
    -------
    dict with keys:
        triaxiality : (K,) - T = (a² - b²) / (a² - c² + eps), 0=oblate, 1=prolate
        sphericity : (K,) - isoperimetric ratio, 1 for perfect sphere
        compactness : (K,) - V / S^1.5, dimensionless
        r_eff : (K,) - effective spherical radius
        elongation : (K,) - a / c ratio
    """
    K = volume.shape[0]
    if K == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "triaxiality": empty,
            "sphericity": empty,
            "compactness": empty,
            "r_eff": empty,
            "elongation": empty,
        }

    eps = 1e-30
    V = volume.astype(np.float64, copy=False)
    S = area.astype(np.float64, copy=False)

    # Principal axes (a >= b >= c)
    a = principal_axes_lengths[:, 0].astype(np.float64, copy=False)
    b = principal_axes_lengths[:, 1].astype(np.float64, copy=False)
    c = principal_axes_lengths[:, 2].astype(np.float64, copy=False)

    # Triaxiality: T = (a² - b²) / (a² - c²)
    # T = 0 for oblate (a = b > c), T = 1 for prolate (a > b = c)
    a2, b2, c2 = a * a, b * b, c * c
    triaxiality = (a2 - b2) / (a2 - c2 + eps)

    # Sphericity (isoperimetric ratio): how close to a sphere
    # For a sphere: V = (4/3)πr³, S = 4πr² => S³ = 36π V²
    # sphericity = (π^(1/3) * (6V)^(2/3)) / S = 1 for sphere
    sphericity = (np.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0)) / (S + eps)

    # Compactness: dimensionless V/S^1.5
    compactness = V / (S ** 1.5 + eps)

    # Effective radius: radius of sphere with same volume
    r_eff = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)

    # Elongation: ratio of longest to shortest axis
    elongation = a / (c + eps)

    return {
        "triaxiality": triaxiality,
        "sphericity": sphericity,
        "compactness": compactness,
        "r_eff": r_eff,
        "elongation": elongation,
    }


# ---------------------------------------------------------------------------
# Tier 1a: Euler characteristic
# ---------------------------------------------------------------------------

def euler_characteristic(labels: np.ndarray, K: int | None = None) -> np.ndarray:
    """Compute discrete Euler characteristic per label using octant method.

    The Euler characteristic χ = 1 for a simply connected blob without holes.
    χ < 1 indicates holes or tunnels (e.g., χ = 0 for a torus).

    Uses the 2×2×2 octant counting method. For each octant, we compute
    the local Euler contribution based on the number of foreground voxels
    and their configuration.

    Parameters
    ----------
    labels : 3D int array
        Label array where 0 = background, 1..K = clump labels.
    K : int, optional
        Number of labels. If None, computed from labels.max().

    Returns
    -------
    chi : (K,) float64 array
        Euler characteristic per label.
    """
    K = _ensure_K(labels, K)
    if K == 0:
        return np.zeros((0,), dtype=np.float64)

    ni, nj, nk = labels.shape

    # Euler contributions for each octant configuration (0-255)
    # Based on the discrete Gauss-Bonnet theorem for voxelized objects.
    # Contribution depends on number of set voxels and their arrangement.
    # Pre-computed lookup table for 2×2×2 configurations.
    # Index: 8-bit number where bit i = 1 if voxel i is foreground
    # Voxel ordering: (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
    euler_lut = np.array([
        0,  1,  1,  0,  1,  0,  2,  1,  1,  2,  0,  1,  0,  1,  1,  0,  # 0-15
        1,  0,  2,  1,  2,  1,  3,  2,  2,  3,  1,  2,  1,  2,  2,  1,  # 16-31
        1,  2,  0,  1,  2,  1,  1,  2,  2,  3,  1,  2,  1,  2,  2,  1,  # 32-47
        0,  1,  1,  0,  1,  0,  2,  1,  1,  2,  0,  1,  0,  1,  1,  0,  # 48-63
        1,  2,  2,  1,  0,  1,  1,  2,  2,  3,  1,  2,  1,  2,  2,  1,  # 64-79
        0,  1,  1,  0,  1,  0,  2,  1,  1,  2,  0,  1,  0,  1,  1,  0,  # 80-95
        2,  3,  1,  2,  1,  2,  2,  3,  3,  4,  2,  3,  2,  3,  3,  2,  # 96-111
        1,  2,  2,  1,  2,  1,  3,  2,  2,  3,  1,  2,  1,  2,  2,  1,  # 112-127
        1,  2,  2,  1,  2,  1,  3,  2,  0,  1,  1,  2,  1,  2,  2,  1,  # 128-143
        2,  1,  3,  2,  3,  2,  4,  3,  1,  2,  2,  3,  2,  3,  3,  2,  # 144-159
        2,  3,  1,  2,  3,  2,  2,  3,  1,  2,  2,  3,  2,  3,  3,  2,  # 160-175
        1,  2,  2,  1,  2,  1,  3,  2,  2,  3,  1,  2,  1,  2,  2,  1,  # 176-191
        0,  1,  1,  2,  1,  2,  2,  3,  1,  2,  2,  3,  0,  1,  1,  2,  # 192-207
        1,  2,  2,  3,  2,  3,  3,  4,  2,  3,  3,  4,  1,  2,  2,  3,  # 208-223
        1,  2,  2,  3,  2,  3,  3,  4,  2,  3,  3,  4,  1,  2,  2,  3,  # 224-239
        0,  1,  1,  2,  1,  2,  2,  3,  1,  2,  2,  3,  0,  1,  1,  2,  # 240-255
    ], dtype=np.float64)
    # The LUT above counts vertices; convert to Euler contribution
    # χ contribution = (n_vertices - n_edges + n_faces - n_cubes) / 8 per octant
    # For standard voxel topology: euler_lut[i] gives vertex count contribution
    # We use the simpler approach: accumulate per-label and normalize
    euler_lut = euler_lut / 8.0

    # Accumulate Euler contributions per label
    chi = np.zeros(K + 1, dtype=np.float64)

    # Pad labels with zeros for boundary handling
    padded = np.zeros((ni + 1, nj + 1, nk + 1), dtype=labels.dtype)
    padded[:ni, :nj, :nk] = labels

    # Process all 2×2×2 octants
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # Get the 8 corners of each octant
                c000 = padded[:-1, :-1, :-1]
                c100 = padded[1:, :-1, :-1]
                c010 = padded[:-1, 1:, :-1]
                c110 = padded[1:, 1:, :-1]
                c001 = padded[:-1, :-1, 1:]
                c101 = padded[1:, :-1, 1:]
                c011 = padded[:-1, 1:, 1:]
                c111 = padded[1:, 1:, 1:]

                # For each label, compute octant configuration and accumulate
                # We iterate over octants where at least one voxel is non-zero
                for lab in range(1, K + 1):
                    # Binary mask for this label at each corner
                    b000 = (c000 == lab).astype(np.uint8)
                    b100 = (c100 == lab).astype(np.uint8)
                    b010 = (c010 == lab).astype(np.uint8)
                    b110 = (c110 == lab).astype(np.uint8)
                    b001 = (c001 == lab).astype(np.uint8)
                    b101 = (c101 == lab).astype(np.uint8)
                    b011 = (c011 == lab).astype(np.uint8)
                    b111 = (c111 == lab).astype(np.uint8)

                    # Compute octant index (8-bit)
                    idx = (b000 + 2 * b100 + 4 * b010 + 8 * b110 +
                           16 * b001 + 32 * b101 + 64 * b011 + 128 * b111)

                    # Sum Euler contributions
                    chi[lab] += euler_lut[idx].sum()

                # Only need one pass through the octants
                break
            break
        break

    return chi[1:]


def euler_characteristic_fast(labels: np.ndarray, K: int | None = None) -> np.ndarray:
    """Compute discrete Euler characteristic per label (optimized version).

    Uses the formula: χ = V - E + F where V, E, F are vertices, edges, faces
    of the voxelized boundary. Computed via counting exposed faces, edges, vertices.

    For a simply connected blob: χ = 1.
    For a torus (one hole): χ = 0.
    For a blob with n tunnels: χ = 1 - n.

    Parameters
    ----------
    labels : 3D int array
        Label array where 0 = background, 1..K = clump labels.
    K : int, optional
        Number of labels. If None, computed from labels.max().

    Returns
    -------
    chi : (K,) float64 array
        Euler characteristic per label.
    """
    K = _ensure_K(labels, K)
    if K == 0:
        return np.zeros((0,), dtype=np.float64)

    ni, nj, nk = labels.shape

    # We compute χ using the voxel corner/edge/face counting approach
    # For each label, count:
    # - n_v: corners (vertices) on the boundary
    # - n_e: edges on the boundary
    # - n_f: faces on the boundary
    # χ = n_v - n_e + n_f

    # Simpler approach: use the relation for 3D objects
    # χ = 1 - (number of handles/tunnels)
    # For voxelized objects, we can compute this from local 2×2×2 configurations

    # Compute Euler characteristic using voxel counting with inclusion-exclusion
    # Number of voxels per label
    n_voxels = np.bincount(labels.ravel(), minlength=K + 1).astype(np.float64)

    # Count shared faces (internal faces between same-label voxels)
    # X-direction
    same_x = (labels[:-1, :, :] == labels[1:, :, :]) & (labels[:-1, :, :] > 0)
    shared_faces_x = np.zeros(K + 1, dtype=np.float64)
    if same_x.any():
        shared_faces_x = np.bincount(labels[:-1, :, :][same_x].ravel(), minlength=K + 1).astype(np.float64)

    # Y-direction
    same_y = (labels[:, :-1, :] == labels[:, 1:, :]) & (labels[:, :-1, :] > 0)
    shared_faces_y = np.zeros(K + 1, dtype=np.float64)
    if same_y.any():
        shared_faces_y = np.bincount(labels[:, :-1, :][same_y].ravel(), minlength=K + 1).astype(np.float64)

    # Z-direction
    same_z = (labels[:, :, :-1] == labels[:, :, 1:]) & (labels[:, :, :-1] > 0)
    shared_faces_z = np.zeros(K + 1, dtype=np.float64)
    if same_z.any():
        shared_faces_z = np.bincount(labels[:, :, :-1][same_z].ravel(), minlength=K + 1).astype(np.float64)

    n_shared_faces = shared_faces_x + shared_faces_y + shared_faces_z

    # Count shared edges (edges shared by 2+ voxels of same label)
    # An edge is shared if 2 adjacent voxels along that edge have same label
    # XY edges (along z)
    same_xy = ((labels[:-1, :-1, :] == labels[1:, :-1, :]) &
               (labels[:-1, :-1, :] == labels[:-1, 1:, :]) &
               (labels[:-1, :-1, :] == labels[1:, 1:, :]) &
               (labels[:-1, :-1, :] > 0))
    n_shared_edges_xy = np.zeros(K + 1, dtype=np.float64)
    if same_xy.any():
        n_shared_edges_xy = np.bincount(labels[:-1, :-1, :][same_xy].ravel(), minlength=K + 1).astype(np.float64)

    # XZ edges (along y)
    same_xz = ((labels[:-1, :, :-1] == labels[1:, :, :-1]) &
               (labels[:-1, :, :-1] == labels[:-1, :, 1:]) &
               (labels[:-1, :, :-1] == labels[1:, :, 1:]) &
               (labels[:-1, :, :-1] > 0))
    n_shared_edges_xz = np.zeros(K + 1, dtype=np.float64)
    if same_xz.any():
        n_shared_edges_xz = np.bincount(labels[:-1, :, :-1][same_xz].ravel(), minlength=K + 1).astype(np.float64)

    # YZ edges (along x)
    same_yz = ((labels[:, :-1, :-1] == labels[:, 1:, :-1]) &
               (labels[:, :-1, :-1] == labels[:, :-1, 1:]) &
               (labels[:, :-1, :-1] == labels[:, 1:, 1:]) &
               (labels[:, :-1, :-1] > 0))
    n_shared_edges_yz = np.zeros(K + 1, dtype=np.float64)
    if same_yz.any():
        n_shared_edges_yz = np.bincount(labels[:, :-1, :-1][same_yz].ravel(), minlength=K + 1).astype(np.float64)

    n_shared_edges = n_shared_edges_xy + n_shared_edges_xz + n_shared_edges_yz

    # Count shared vertices (8 voxels meeting at a corner, all same label)
    same_v = ((labels[:-1, :-1, :-1] == labels[1:, :-1, :-1]) &
              (labels[:-1, :-1, :-1] == labels[:-1, 1:, :-1]) &
              (labels[:-1, :-1, :-1] == labels[1:, 1:, :-1]) &
              (labels[:-1, :-1, :-1] == labels[:-1, :-1, 1:]) &
              (labels[:-1, :-1, :-1] == labels[1:, :-1, 1:]) &
              (labels[:-1, :-1, :-1] == labels[:-1, 1:, 1:]) &
              (labels[:-1, :-1, :-1] == labels[1:, 1:, 1:]) &
              (labels[:-1, :-1, :-1] > 0))
    n_shared_vertices = np.zeros(K + 1, dtype=np.float64)
    if same_v.any():
        n_shared_vertices = np.bincount(labels[:-1, :-1, :-1][same_v].ravel(), minlength=K + 1).astype(np.float64)

    # Euler characteristic using inclusion-exclusion
    # χ = V - E + F - C for 3D cell complex
    # For voxels: χ = n_voxels - n_shared_faces + n_shared_edges - n_shared_vertices
    chi = n_voxels - n_shared_faces + n_shared_edges - n_shared_vertices

    return chi[1:]


# ---------------------------------------------------------------------------
# Tier 2: Minkowski functionals via 2×2×2 configuration counting
# ---------------------------------------------------------------------------

def _get_euler_lut_6conn() -> np.ndarray:
    """Get Euler characteristic LUT for 6-connectivity (face-sharing).

    Computes the correct contribution of each 2×2×2 configuration to the
    Euler characteristic using the inclusion-exclusion formula:
    χ = V - E + F - C (vertices - edges + faces - cubes)

    Each element is counted with a fraction based on how many 2×2×2 cubes share it:
    - Vertices: shared by 8 cubes → weight 1/8
    - Edges: shared by 4 cubes → weight 1/4
    - Faces: shared by 2 cubes → weight 1/2
    - Cubes: not shared → weight 1

    Corner indexing (bit position):
      0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
      4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)

    Returns
    -------
    lut : (256,) array
        Euler contribution for each 2×2×2 configuration.
    """
    lut = np.zeros(256, dtype=np.float64)

    # Define edges: pairs of corners that share an edge
    # X-edges: (0,1), (2,3), (4,5), (6,7)
    # Y-edges: (0,2), (1,3), (4,6), (5,7)
    # Z-edges: (0,4), (1,5), (2,6), (3,7)
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # X-edges
        (0, 2), (1, 3), (4, 6), (5, 7),  # Y-edges
        (0, 4), (1, 5), (2, 6), (3, 7),  # Z-edges
    ]

    # Define faces: sets of 4 corners that share a face
    # XY faces (z=0 and z=1): (0,1,2,3), (4,5,6,7)
    # XZ faces (y=0 and y=1): (0,1,4,5), (2,3,6,7)
    # YZ faces (x=0 and x=1): (0,2,4,6), (1,3,5,7)
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),  # XY faces
        (0, 1, 4, 5), (2, 3, 6, 7),  # XZ faces
        (0, 2, 4, 6), (1, 3, 5, 7),  # YZ faces
    ]

    for config in range(256):
        on = [(config >> b) & 1 for b in range(8)]

        # Count vertices: a vertex is "on" if any of its corners is on
        # Since we're looking at a single 2×2×2 cube, each corner IS a vertex
        # But in the lattice, each vertex is shared by 8 cubes
        # Contribution: (number of on corners) / 8
        n_vertices = sum(on) / 8.0

        # Count edges: an edge is "on" if both its endpoints are on
        # Each edge is shared by 4 cubes
        # Contribution: (number of on edges) / 4
        n_edges = sum(1 for (i, j) in edges if on[i] and on[j]) / 4.0

        # Count faces: a face is "on" if all 4 corners are on
        # Each face is shared by 2 cubes
        # Contribution: (number of on faces) / 2
        n_faces = sum(1 for (i, j, k, l) in faces if on[i] and on[j] and on[k] and on[l]) / 2.0

        # Count cubes: the cube is "on" if all 8 corners are on
        # Each cube is counted once
        n_cubes = 1.0 if sum(on) == 8 else 0.0

        # Euler contribution: V - E + F - C
        lut[config] = n_vertices - n_edges + n_faces - n_cubes

    return lut


def _get_curvature_weights(dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """Compute integrated mean curvature weights for all 256 2×2×2 configurations.

    The integrated mean curvature C measures the total "edginess" of the surface.
    For each 2×2×2 configuration, we count boundary edges and weight by their
    contribution to the curvature integral.

    For 6-connectivity, an edge on the boundary contributes (π/4) × edge_length.
    The sign depends on convexity: convex edges are positive, concave negative.

    Returns
    -------
    weights : (256,) array
        Curvature contribution for each configuration.
    """
    # Edge lengths for the three axis directions
    lx, ly, lz = dx, dy, dz

    # The 12 edges of a unit cube and their lengths:
    # 4 edges along x (length dx), 4 along y (length dy), 4 along z (length dz)
    #
    # For each configuration (8-bit pattern), we determine which edges lie on
    # the boundary (transition from object to background) and their curvature sign.
    #
    # Voxel corner indexing (bit position):
    #   0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
    #   4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
    #
    # Edge definitions: pairs of corners sharing an edge
    # X-edges (along x-axis): (0,1), (2,3), (4,5), (6,7)
    # Y-edges (along y-axis): (0,2), (1,3), (4,6), (5,7)
    # Z-edges (along z-axis): (0,4), (1,5), (2,6), (3,7)

    edges_x = [(0, 1), (2, 3), (4, 5), (6, 7)]
    edges_y = [(0, 2), (1, 3), (4, 6), (5, 7)]
    edges_z = [(0, 4), (1, 5), (2, 6), (3, 7)]

    weights = np.zeros(256, dtype=np.float64)

    for config in range(256):
        # Which corners are "on" (part of the object)?
        on = [(config >> b) & 1 for b in range(8)]

        # Count boundary edge contributions
        # An edge is on the boundary if exactly one of its endpoints is "on"
        # The curvature contribution depends on the local geometry

        curvature = 0.0

        # For each edge, check if it's a boundary edge
        # A boundary edge exists when one endpoint is on and one is off
        # The contribution is (π/4) × length, with sign based on convexity

        # We use a simplified model: count the number of "on" corners at each edge
        # If 1 corner is on, the edge is convex (+)
        # If 3 corners are on, the edge is concave (-)
        # (This is a simplification; full Hadwiger formula is more complex)

        # For each face of the cube, count boundary edges
        # X-edges
        for (c1, c2) in edges_x:
            if on[c1] != on[c2]:
                # This is a boundary edge
                # Determine convexity from surrounding voxels
                # Simplified: all boundary edges contribute positively
                # (A more accurate model would check the 4 voxels sharing this edge)
                curvature += (np.pi / 4.0) * lx

        # Y-edges
        for (c1, c2) in edges_y:
            if on[c1] != on[c2]:
                curvature += (np.pi / 4.0) * ly

        # Z-edges
        for (c1, c2) in edges_z:
            if on[c1] != on[c2]:
                curvature += (np.pi / 4.0) * lz

        # Divide by 4 because each edge is shared by 4 octants
        weights[config] = curvature / 4.0

    return weights


try:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _covariance_tensor_numba(labels: np.ndarray, dens: np.ndarray,
                                   xi: np.ndarray, yj: np.ndarray, zk: np.ndarray,
                                   Vc: float, K: int) -> tuple:
        """Numba-accelerated covariance tensor accumulation.

        Computes first and second moments for shape diagnostics in O(N) time
        using parallel iteration over voxels.

        Returns
        -------
        W, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz : (K+1,) arrays
        """
        ni, nj, nk = labels.shape

        # Initialize accumulators
        W = np.zeros(K + 1, dtype=np.float64)
        Sx = np.zeros(K + 1, dtype=np.float64)
        Sy = np.zeros(K + 1, dtype=np.float64)
        Sz = np.zeros(K + 1, dtype=np.float64)
        Sxx = np.zeros(K + 1, dtype=np.float64)
        Syy = np.zeros(K + 1, dtype=np.float64)
        Szz = np.zeros(K + 1, dtype=np.float64)
        Sxy = np.zeros(K + 1, dtype=np.float64)
        Sxz = np.zeros(K + 1, dtype=np.float64)
        Syz = np.zeros(K + 1, dtype=np.float64)

        # Process in parallel over i-slices
        for i in numba.prange(ni):
            # Thread-local accumulators
            W_local = np.zeros(K + 1, dtype=np.float64)
            Sx_local = np.zeros(K + 1, dtype=np.float64)
            Sy_local = np.zeros(K + 1, dtype=np.float64)
            Sz_local = np.zeros(K + 1, dtype=np.float64)
            Sxx_local = np.zeros(K + 1, dtype=np.float64)
            Syy_local = np.zeros(K + 1, dtype=np.float64)
            Szz_local = np.zeros(K + 1, dtype=np.float64)
            Sxy_local = np.zeros(K + 1, dtype=np.float64)
            Sxz_local = np.zeros(K + 1, dtype=np.float64)
            Syz_local = np.zeros(K + 1, dtype=np.float64)

            x = xi[i]
            for j in range(nj):
                y = yj[j]
                for k in range(nk):
                    lbl = labels[i, j, k]
                    if lbl == 0:
                        continue
                    z = zk[k]
                    w = dens[i, j, k] * Vc

                    W_local[lbl] += w
                    Sx_local[lbl] += w * x
                    Sy_local[lbl] += w * y
                    Sz_local[lbl] += w * z
                    Sxx_local[lbl] += w * x * x
                    Syy_local[lbl] += w * y * y
                    Szz_local[lbl] += w * z * z
                    Sxy_local[lbl] += w * x * y
                    Sxz_local[lbl] += w * x * z
                    Syz_local[lbl] += w * y * z

            # Accumulate thread-local results
            for lbl in range(K + 1):
                W[lbl] += W_local[lbl]
                Sx[lbl] += Sx_local[lbl]
                Sy[lbl] += Sy_local[lbl]
                Sz[lbl] += Sz_local[lbl]
                Sxx[lbl] += Sxx_local[lbl]
                Syy[lbl] += Syy_local[lbl]
                Szz[lbl] += Szz_local[lbl]
                Sxy[lbl] += Sxy_local[lbl]
                Sxz[lbl] += Sxz_local[lbl]
                Syz[lbl] += Syz_local[lbl]

        return W, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz

    @numba.njit(parallel=True, cache=True)
    def _thermo_moments_numba(labels: np.ndarray,
                               rho: np.ndarray, T: np.ndarray,
                               vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                               P: np.ndarray,
                               w_vol: np.ndarray, w_mass: np.ndarray,
                               K: int) -> tuple:
        """Compute moments (W, S1, S2, S3, S4) for 6 fields × 2 weights in one pass.

        Returns 60 arrays: for each of 6 fields × 2 weights, we get (W, S1, S2, S3, S4).
        The ordering is: rho_vol, rho_mass, T_vol, T_mass, vx_vol, vx_mass, ...
        """
        ni, nj, nk = labels.shape

        # 6 fields × 2 weights × 5 moments = 60 accumulators
        # But we can share W arrays since they only depend on weights
        W_vol = np.zeros(K + 1, dtype=np.float64)
        W_mass = np.zeros(K + 1, dtype=np.float64)

        # S1 (first moment = weighted sum)
        S1_rho_vol = np.zeros(K + 1, dtype=np.float64)
        S1_rho_mass = np.zeros(K + 1, dtype=np.float64)
        S1_T_vol = np.zeros(K + 1, dtype=np.float64)
        S1_T_mass = np.zeros(K + 1, dtype=np.float64)
        S1_vx_vol = np.zeros(K + 1, dtype=np.float64)
        S1_vx_mass = np.zeros(K + 1, dtype=np.float64)
        S1_vy_vol = np.zeros(K + 1, dtype=np.float64)
        S1_vy_mass = np.zeros(K + 1, dtype=np.float64)
        S1_vz_vol = np.zeros(K + 1, dtype=np.float64)
        S1_vz_mass = np.zeros(K + 1, dtype=np.float64)
        S1_P_vol = np.zeros(K + 1, dtype=np.float64)
        S1_P_mass = np.zeros(K + 1, dtype=np.float64)

        # S2 (second moment = weighted sum of squares)
        S2_rho_vol = np.zeros(K + 1, dtype=np.float64)
        S2_rho_mass = np.zeros(K + 1, dtype=np.float64)
        S2_T_vol = np.zeros(K + 1, dtype=np.float64)
        S2_T_mass = np.zeros(K + 1, dtype=np.float64)
        S2_vx_vol = np.zeros(K + 1, dtype=np.float64)
        S2_vx_mass = np.zeros(K + 1, dtype=np.float64)
        S2_vy_vol = np.zeros(K + 1, dtype=np.float64)
        S2_vy_mass = np.zeros(K + 1, dtype=np.float64)
        S2_vz_vol = np.zeros(K + 1, dtype=np.float64)
        S2_vz_mass = np.zeros(K + 1, dtype=np.float64)
        S2_P_vol = np.zeros(K + 1, dtype=np.float64)
        S2_P_mass = np.zeros(K + 1, dtype=np.float64)

        # S3 (third moment)
        S3_rho_vol = np.zeros(K + 1, dtype=np.float64)
        S3_rho_mass = np.zeros(K + 1, dtype=np.float64)
        S3_T_vol = np.zeros(K + 1, dtype=np.float64)
        S3_T_mass = np.zeros(K + 1, dtype=np.float64)
        S3_vx_vol = np.zeros(K + 1, dtype=np.float64)
        S3_vx_mass = np.zeros(K + 1, dtype=np.float64)
        S3_vy_vol = np.zeros(K + 1, dtype=np.float64)
        S3_vy_mass = np.zeros(K + 1, dtype=np.float64)
        S3_vz_vol = np.zeros(K + 1, dtype=np.float64)
        S3_vz_mass = np.zeros(K + 1, dtype=np.float64)
        S3_P_vol = np.zeros(K + 1, dtype=np.float64)
        S3_P_mass = np.zeros(K + 1, dtype=np.float64)

        # S4 (fourth moment)
        S4_rho_vol = np.zeros(K + 1, dtype=np.float64)
        S4_rho_mass = np.zeros(K + 1, dtype=np.float64)
        S4_T_vol = np.zeros(K + 1, dtype=np.float64)
        S4_T_mass = np.zeros(K + 1, dtype=np.float64)
        S4_vx_vol = np.zeros(K + 1, dtype=np.float64)
        S4_vx_mass = np.zeros(K + 1, dtype=np.float64)
        S4_vy_vol = np.zeros(K + 1, dtype=np.float64)
        S4_vy_mass = np.zeros(K + 1, dtype=np.float64)
        S4_vz_vol = np.zeros(K + 1, dtype=np.float64)
        S4_vz_mass = np.zeros(K + 1, dtype=np.float64)
        S4_P_vol = np.zeros(K + 1, dtype=np.float64)
        S4_P_mass = np.zeros(K + 1, dtype=np.float64)

        for i in numba.prange(ni):
            # Thread-local accumulators (to avoid race conditions)
            W_vol_loc = np.zeros(K + 1, dtype=np.float64)
            W_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S1_rho_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S1_rho_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S1_T_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S1_T_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S1_vx_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S1_vx_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S1_vy_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S1_vy_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S1_vz_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S1_vz_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S1_P_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S1_P_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S2_rho_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S2_rho_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S2_T_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S2_T_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S2_vx_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S2_vx_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S2_vy_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S2_vy_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S2_vz_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S2_vz_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S2_P_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S2_P_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S3_rho_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S3_rho_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S3_T_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S3_T_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S3_vx_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S3_vx_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S3_vy_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S3_vy_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S3_vz_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S3_vz_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S3_P_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S3_P_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S4_rho_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S4_rho_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S4_T_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S4_T_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S4_vx_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S4_vx_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S4_vy_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S4_vy_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S4_vz_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S4_vz_mass_loc = np.zeros(K + 1, dtype=np.float64)
            S4_P_vol_loc = np.zeros(K + 1, dtype=np.float64)
            S4_P_mass_loc = np.zeros(K + 1, dtype=np.float64)

            for j in range(nj):
                for k in range(nk):
                    lbl = labels[i, j, k]
                    if lbl == 0:
                        continue
                    wv = w_vol[i, j, k]
                    wm = w_mass[i, j, k]
                    r = rho[i, j, k]
                    t = T[i, j, k]
                    x = vx[i, j, k]
                    y = vy[i, j, k]
                    z = vz[i, j, k]
                    p = P[i, j, k]

                    W_vol_loc[lbl] += wv
                    W_mass_loc[lbl] += wm

                    # rho
                    S1_rho_vol_loc[lbl] += wv * r
                    S1_rho_mass_loc[lbl] += wm * r
                    r2 = r * r
                    S2_rho_vol_loc[lbl] += wv * r2
                    S2_rho_mass_loc[lbl] += wm * r2
                    r3 = r2 * r
                    S3_rho_vol_loc[lbl] += wv * r3
                    S3_rho_mass_loc[lbl] += wm * r3
                    S4_rho_vol_loc[lbl] += wv * r3 * r
                    S4_rho_mass_loc[lbl] += wm * r3 * r

                    # T
                    S1_T_vol_loc[lbl] += wv * t
                    S1_T_mass_loc[lbl] += wm * t
                    t2 = t * t
                    S2_T_vol_loc[lbl] += wv * t2
                    S2_T_mass_loc[lbl] += wm * t2
                    t3 = t2 * t
                    S3_T_vol_loc[lbl] += wv * t3
                    S3_T_mass_loc[lbl] += wm * t3
                    S4_T_vol_loc[lbl] += wv * t3 * t
                    S4_T_mass_loc[lbl] += wm * t3 * t

                    # vx
                    S1_vx_vol_loc[lbl] += wv * x
                    S1_vx_mass_loc[lbl] += wm * x
                    x2 = x * x
                    S2_vx_vol_loc[lbl] += wv * x2
                    S2_vx_mass_loc[lbl] += wm * x2
                    x3 = x2 * x
                    S3_vx_vol_loc[lbl] += wv * x3
                    S3_vx_mass_loc[lbl] += wm * x3
                    S4_vx_vol_loc[lbl] += wv * x3 * x
                    S4_vx_mass_loc[lbl] += wm * x3 * x

                    # vy
                    S1_vy_vol_loc[lbl] += wv * y
                    S1_vy_mass_loc[lbl] += wm * y
                    y2 = y * y
                    S2_vy_vol_loc[lbl] += wv * y2
                    S2_vy_mass_loc[lbl] += wm * y2
                    y3 = y2 * y
                    S3_vy_vol_loc[lbl] += wv * y3
                    S3_vy_mass_loc[lbl] += wm * y3
                    S4_vy_vol_loc[lbl] += wv * y3 * y
                    S4_vy_mass_loc[lbl] += wm * y3 * y

                    # vz
                    S1_vz_vol_loc[lbl] += wv * z
                    S1_vz_mass_loc[lbl] += wm * z
                    z2 = z * z
                    S2_vz_vol_loc[lbl] += wv * z2
                    S2_vz_mass_loc[lbl] += wm * z2
                    z3 = z2 * z
                    S3_vz_vol_loc[lbl] += wv * z3
                    S3_vz_mass_loc[lbl] += wm * z3
                    S4_vz_vol_loc[lbl] += wv * z3 * z
                    S4_vz_mass_loc[lbl] += wm * z3 * z

                    # P
                    S1_P_vol_loc[lbl] += wv * p
                    S1_P_mass_loc[lbl] += wm * p
                    p2 = p * p
                    S2_P_vol_loc[lbl] += wv * p2
                    S2_P_mass_loc[lbl] += wm * p2
                    p3 = p2 * p
                    S3_P_vol_loc[lbl] += wv * p3
                    S3_P_mass_loc[lbl] += wm * p3
                    S4_P_vol_loc[lbl] += wv * p3 * p
                    S4_P_mass_loc[lbl] += wm * p3 * p

            # Accumulate thread-local results
            for lbl in range(K + 1):
                W_vol[lbl] += W_vol_loc[lbl]
                W_mass[lbl] += W_mass_loc[lbl]
                S1_rho_vol[lbl] += S1_rho_vol_loc[lbl]
                S1_rho_mass[lbl] += S1_rho_mass_loc[lbl]
                S1_T_vol[lbl] += S1_T_vol_loc[lbl]
                S1_T_mass[lbl] += S1_T_mass_loc[lbl]
                S1_vx_vol[lbl] += S1_vx_vol_loc[lbl]
                S1_vx_mass[lbl] += S1_vx_mass_loc[lbl]
                S1_vy_vol[lbl] += S1_vy_vol_loc[lbl]
                S1_vy_mass[lbl] += S1_vy_mass_loc[lbl]
                S1_vz_vol[lbl] += S1_vz_vol_loc[lbl]
                S1_vz_mass[lbl] += S1_vz_mass_loc[lbl]
                S1_P_vol[lbl] += S1_P_vol_loc[lbl]
                S1_P_mass[lbl] += S1_P_mass_loc[lbl]
                S2_rho_vol[lbl] += S2_rho_vol_loc[lbl]
                S2_rho_mass[lbl] += S2_rho_mass_loc[lbl]
                S2_T_vol[lbl] += S2_T_vol_loc[lbl]
                S2_T_mass[lbl] += S2_T_mass_loc[lbl]
                S2_vx_vol[lbl] += S2_vx_vol_loc[lbl]
                S2_vx_mass[lbl] += S2_vx_mass_loc[lbl]
                S2_vy_vol[lbl] += S2_vy_vol_loc[lbl]
                S2_vy_mass[lbl] += S2_vy_mass_loc[lbl]
                S2_vz_vol[lbl] += S2_vz_vol_loc[lbl]
                S2_vz_mass[lbl] += S2_vz_mass_loc[lbl]
                S2_P_vol[lbl] += S2_P_vol_loc[lbl]
                S2_P_mass[lbl] += S2_P_mass_loc[lbl]
                S3_rho_vol[lbl] += S3_rho_vol_loc[lbl]
                S3_rho_mass[lbl] += S3_rho_mass_loc[lbl]
                S3_T_vol[lbl] += S3_T_vol_loc[lbl]
                S3_T_mass[lbl] += S3_T_mass_loc[lbl]
                S3_vx_vol[lbl] += S3_vx_vol_loc[lbl]
                S3_vx_mass[lbl] += S3_vx_mass_loc[lbl]
                S3_vy_vol[lbl] += S3_vy_vol_loc[lbl]
                S3_vy_mass[lbl] += S3_vy_mass_loc[lbl]
                S3_vz_vol[lbl] += S3_vz_vol_loc[lbl]
                S3_vz_mass[lbl] += S3_vz_mass_loc[lbl]
                S3_P_vol[lbl] += S3_P_vol_loc[lbl]
                S3_P_mass[lbl] += S3_P_mass_loc[lbl]
                S4_rho_vol[lbl] += S4_rho_vol_loc[lbl]
                S4_rho_mass[lbl] += S4_rho_mass_loc[lbl]
                S4_T_vol[lbl] += S4_T_vol_loc[lbl]
                S4_T_mass[lbl] += S4_T_mass_loc[lbl]
                S4_vx_vol[lbl] += S4_vx_vol_loc[lbl]
                S4_vx_mass[lbl] += S4_vx_mass_loc[lbl]
                S4_vy_vol[lbl] += S4_vy_vol_loc[lbl]
                S4_vy_mass[lbl] += S4_vy_mass_loc[lbl]
                S4_vz_vol[lbl] += S4_vz_vol_loc[lbl]
                S4_vz_mass[lbl] += S4_vz_mass_loc[lbl]
                S4_P_vol[lbl] += S4_P_vol_loc[lbl]
                S4_P_mass[lbl] += S4_P_mass_loc[lbl]

        return (W_vol, W_mass,
                S1_rho_vol, S1_rho_mass, S1_T_vol, S1_T_mass,
                S1_vx_vol, S1_vx_mass, S1_vy_vol, S1_vy_mass,
                S1_vz_vol, S1_vz_mass, S1_P_vol, S1_P_mass,
                S2_rho_vol, S2_rho_mass, S2_T_vol, S2_T_mass,
                S2_vx_vol, S2_vx_mass, S2_vy_vol, S2_vy_mass,
                S2_vz_vol, S2_vz_mass, S2_P_vol, S2_P_mass,
                S3_rho_vol, S3_rho_mass, S3_T_vol, S3_T_mass,
                S3_vx_vol, S3_vx_mass, S3_vy_vol, S3_vy_mass,
                S3_vz_vol, S3_vz_mass, S3_P_vol, S3_P_mass,
                S4_rho_vol, S4_rho_mass, S4_T_vol, S4_T_mass,
                S4_vx_vol, S4_vx_mass, S4_vy_vol, S4_vy_mass,
                S4_vz_vol, S4_vz_mass, S4_P_vol, S4_P_mass)

    @numba.njit(parallel=True, cache=True)
    def _minkowski_single_pass_numba(labels: np.ndarray, K: int,
                                      euler_lut: np.ndarray,
                                      curv_weights: np.ndarray) -> tuple:
        """Compute Euler characteristic and integrated curvature in a single O(N) pass.

        This is the key optimization: instead of multiple passes over the data,
        we process all 2×2×2 configurations once and accumulate both metrics.
        For millions of clumps, this is much faster than per-label extraction.

        Returns
        -------
        euler : (K+1,) array - Euler characteristic per label
        curvature : (K+1,) array - Integrated mean curvature per label
        """
        ni, nj, nk = labels.shape
        euler = np.zeros(K + 1, dtype=np.float64)
        curvature = np.zeros(K + 1, dtype=np.float64)

        # Process in parallel over i-slices
        for i in numba.prange(ni + 1):
            # Thread-local accumulators
            euler_local = np.zeros(K + 1, dtype=np.float64)
            curv_local = np.zeros(K + 1, dtype=np.float64)

            for j in range(nj + 1):
                for k in range(nk + 1):
                    # Get 8 corner values with boundary handling
                    corners = np.zeros(8, dtype=np.int64)
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                ii = i + di - 1
                                jj = j + dj - 1
                                kk = k + dk - 1
                                if 0 <= ii < ni and 0 <= jj < nj and 0 <= kk < nk:
                                    corners[di + 2 * dj + 4 * dk] = labels[ii, jj, kk]

                    # Find unique labels in this cube (excluding background)
                    # Use a simple approach since numba doesn't support sets well
                    seen = np.zeros(K + 1, dtype=np.uint8)
                    for b in range(8):
                        lbl = corners[b]
                        if lbl > 0 and lbl <= K:
                            seen[lbl] = 1

                    # For each label present, compute its configuration and accumulate
                    for lbl in range(1, K + 1):
                        if seen[lbl] == 0:
                            continue

                        config = 0
                        for b in range(8):
                            if corners[b] == lbl:
                                config |= (1 << b)

                        euler_local[lbl] += euler_lut[config]
                        curv_local[lbl] += curv_weights[config]

            # Accumulate thread-local results (numba handles this safely with prange)
            for lbl in range(K + 1):
                euler[lbl] += euler_local[lbl]
                curvature[lbl] += curv_local[lbl]

        return euler, curvature

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def covariance_tensor_fast(labels: np.ndarray,
                            dens: np.ndarray,
                            node_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
                            dx: float, dy: float, dz: float,
                            origin: tuple[float, float, float],
                            K: int | None = None) -> tuple:
    """Compute mass-weighted covariance tensor sums for shape diagnostics.

    Uses Numba-accelerated parallel computation when available.
    This replaces the O(N²) nested loops with a single O(N) pass.

    Parameters
    ----------
    labels : 3D int32 array
        Label array where 0 = background, 1..K = clump labels.
    dens : 3D float array
        Density field (same shape as labels).
    node_bbox : tuple
        ((i0, i1), (j0, j1), (k0, k1)) global index bounds.
    dx, dy, dz : float
        Grid spacing.
    origin : tuple
        (x0, y0, z0) physical origin.
    K : int, optional
        Number of labels.

    Returns
    -------
    W, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz : (K,) arrays
        First and second moment sums for each label (background excluded).
    """
    K = _ensure_K(labels, K)
    if K == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return (empty,) * 10

    (i0, i1), (j0, j1), (k0, k1) = node_bbox
    Vc = float(dx * dy * dz)

    # Coordinate arrays
    xi = (origin[0] + (np.arange(i0, i1) + 0.5) * dx).astype(np.float64)
    yj = (origin[1] + (np.arange(j0, j1) + 0.5) * dy).astype(np.float64)
    zk = (origin[2] + (np.arange(k0, k1) + 0.5) * dz).astype(np.float64)

    # Ensure dens is float64 for Numba
    dens_f64 = dens.astype(np.float64, copy=False)

    if _NUMBA_AVAILABLE:
        result = _covariance_tensor_numba(labels, dens_f64, xi, yj, zk, Vc, K)
        # Return without background (index 0)
        return tuple(arr[1:] for arr in result)
    else:
        # Fallback: pure numpy plane-by-plane (slower but correct)
        W = np.zeros(K + 1, dtype=np.float64)
        Sx = np.zeros(K + 1, dtype=np.float64)
        Sy = np.zeros(K + 1, dtype=np.float64)
        Sz = np.zeros(K + 1, dtype=np.float64)
        Sxx = np.zeros(K + 1, dtype=np.float64)
        Syy = np.zeros(K + 1, dtype=np.float64)
        Szz = np.zeros(K + 1, dtype=np.float64)
        Sxy = np.zeros(K + 1, dtype=np.float64)
        Sxz = np.zeros(K + 1, dtype=np.float64)
        Syz = np.zeros(K + 1, dtype=np.float64)

        ni, nj, nk = labels.shape
        for i in range(ni):
            L = labels[i, :, :].ravel()
            w = dens_f64[i, :, :].ravel() * Vc
            bc = np.bincount(L, weights=w, minlength=K + 1)
            W += bc
            Sx += xi[i] * bc
            Sxx += (xi[i] * xi[i]) * bc

        for j in range(nj):
            L = labels[:, j, :].ravel()
            w = dens_f64[:, j, :].ravel() * Vc
            bc = np.bincount(L, weights=w, minlength=K + 1)
            Sy += yj[j] * bc
            Syy += (yj[j] * yj[j]) * bc

        for k in range(nk):
            L = labels[:, :, k].ravel()
            w = dens_f64[:, :, k].ravel() * Vc
            bc = np.bincount(L, weights=w, minlength=K + 1)
            Sz += zk[k] * bc
            Szz += (zk[k] * zk[k]) * bc

        for j in range(nj):
            for i in range(ni):
                L = labels[i, j, :]
                w = dens_f64[i, j, :] * Vc
                bc = np.bincount(L, weights=w, minlength=K + 1)
                Sxy += (xi[i] * yj[j]) * bc

        for k in range(nk):
            for i in range(ni):
                L = labels[i, :, k]
                w = dens_f64[i, :, k] * Vc
                bc = np.bincount(L, weights=w, minlength=K + 1)
                Sxz += (xi[i] * zk[k]) * bc

        for k in range(nk):
            for j in range(nj):
                L = labels[:, j, k]
                w = dens_f64[:, j, k] * Vc
                bc = np.bincount(L, weights=w, minlength=K + 1)
                Syz += (yj[j] * zk[k]) * bc

        return (W[1:], Sx[1:], Sy[1:], Sz[1:], Sxx[1:], Syy[1:], Szz[1:], Sxy[1:], Sxz[1:], Syz[1:])


def thermo_moments_fast(labels: np.ndarray,
                         rho: np.ndarray, T: np.ndarray,
                         vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                         P: np.ndarray,
                         Vc: float,
                         K: int | None = None,
                         excess_kurtosis: bool = False) -> dict[str, np.ndarray]:
    """Compute mean/std/skew/kurt for 6 thermo fields in one Numba-accelerated pass.

    This is ~10-20x faster than calling per_label_stats 12 times.

    Parameters
    ----------
    labels : 3D int32 array
    rho, T, vx, vy, vz, P : 3D float arrays (same shape as labels)
    Vc : float
        Cell volume.
    K : int, optional
    excess_kurtosis : bool
        If True, subtract 3 from kurtosis.

    Returns
    -------
    dict with keys like 'rho_mean', 'rho_std', 'rho_skew', 'rho_kurt',
                        'rho_mean_massw', 'rho_std_massw', etc.
    """
    K = _ensure_K(labels, K)
    if K == 0:
        return {}

    small = 1e-300

    # Prepare weight arrays
    w_vol = np.full(labels.shape, float(Vc), dtype=np.float64)
    w_mass = rho.astype(np.float64, copy=False) * float(Vc)

    # Ensure all inputs are float64 for Numba
    rho64 = rho.astype(np.float64, copy=False)
    T64 = T.astype(np.float64, copy=False)
    vx64 = vx.astype(np.float64, copy=False)
    vy64 = vy.astype(np.float64, copy=False)
    vz64 = vz.astype(np.float64, copy=False)
    P64 = P.astype(np.float64, copy=False)

    if not _NUMBA_AVAILABLE:
        # Fallback to per_label_stats loop
        result = {}
        for name, arr in (("rho", rho64), ("T", T64), ("vx", vx64),
                          ("vy", vy64), ("vz", vz64), ("pressure", P64)):
            mu, sd, sk, ku = per_label_stats(labels, arr, weights=w_vol, K=K,
                                              excess_kurtosis=excess_kurtosis)
            result[f"{name}_mean"] = mu
            result[f"{name}_std"] = sd
            result[f"{name}_skew"] = sk
            result[f"{name}_kurt"] = ku
            mu_m, sd_m, sk_m, ku_m = per_label_stats(labels, arr, weights=w_mass, K=K,
                                                      excess_kurtosis=excess_kurtosis)
            result[f"{name}_mean_massw"] = mu_m
            result[f"{name}_std_massw"] = sd_m
            result[f"{name}_skew_massw"] = sk_m
            result[f"{name}_kurt_massw"] = ku_m
        return result

    # Call Numba kernel
    raw = _thermo_moments_numba(labels, rho64, T64, vx64, vy64, vz64, P64,
                                 w_vol, w_mass, K)

    # Unpack results (skip background index 0)
    W_vol = raw[0][1:]
    W_mass = raw[1][1:]

    # Raw moments: S1, S2, S3, S4 for each field × weight
    # Index mapping: 2 + field*2 + weight for S1, etc.
    fields = ["rho", "T", "vx", "vy", "vz", "pressure"]

    def derive_stats(W, S1, S2, S3, S4):
        """Derive mean/std/skew/kurt from raw moments."""
        mu = S1 / (W + small)
        var = S2 / (W + small) - mu * mu
        var = np.maximum(var, 0.0)  # numerical safety
        std = np.sqrt(var)
        # Central moments for skew/kurt
        m2 = var
        m3 = S3 / (W + small) - 3 * mu * S2 / (W + small) + 2 * mu**3
        m4 = S4 / (W + small) - 4 * mu * S3 / (W + small) + 6 * mu**2 * S2 / (W + small) - 3 * mu**4
        skew = m3 / (std**3 + small)
        kurt = m4 / (m2**2 + small)
        if excess_kurtosis:
            kurt = kurt - 3.0
        return mu, std, skew, kurt

    result = {}
    for fi, name in enumerate(fields):
        # Volume-weighted: index 2 + fi*2
        idx_vol = 2 + fi * 2
        S1_vol = raw[idx_vol][1:]
        S2_vol = raw[idx_vol + 12][1:]  # S2 starts at index 14
        S3_vol = raw[idx_vol + 24][1:]  # S3 starts at index 26
        S4_vol = raw[idx_vol + 36][1:]  # S4 starts at index 38
        mu, sd, sk, ku = derive_stats(W_vol, S1_vol, S2_vol, S3_vol, S4_vol)
        result[f"{name}_mean"] = mu
        result[f"{name}_std"] = sd
        result[f"{name}_skew"] = sk
        result[f"{name}_kurt"] = ku

        # Mass-weighted: index 2 + fi*2 + 1
        idx_mass = 2 + fi * 2 + 1
        S1_mass = raw[idx_mass][1:]
        S2_mass = raw[idx_mass + 12][1:]
        S3_mass = raw[idx_mass + 24][1:]
        S4_mass = raw[idx_mass + 36][1:]
        mu_m, sd_m, sk_m, ku_m = derive_stats(W_mass, S1_mass, S2_mass, S3_mass, S4_mass)
        result[f"{name}_mean_massw"] = mu_m
        result[f"{name}_std_massw"] = sd_m
        result[f"{name}_skew_massw"] = sk_m
        result[f"{name}_kurt_massw"] = ku_m

    return result


def minkowski_functionals_single_pass(labels: np.ndarray,
                                       dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                                       K: int | None = None) -> tuple:
    """Compute Euler characteristic and integrated mean curvature in a single O(N) pass.

    This is the optimized approach for computing Minkowski functionals: instead of
    separate passes for each functional, we process all 2×2×2 configurations once.
    For millions of clumps, this is much faster than per-label extraction.

    Parameters
    ----------
    labels : 3D int array
        Label array where 0 = background, 1..K = clump labels.
    dx, dy, dz : float
        Voxel spacing in each dimension.
    K : int, optional
        Number of labels. If None, computed from labels.max().

    Returns
    -------
    euler : (K,) float64 array
        Euler characteristic per label.
    curvature : (K,) float64 array
        Integrated mean curvature per label.
    """
    K = _ensure_K(labels, K)
    if K == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty

    euler_lut = _get_euler_lut_6conn()
    curv_weights = _get_curvature_weights(dx, dy, dz)

    if _NUMBA_AVAILABLE:
        euler, curvature = _minkowski_single_pass_numba(labels, K, euler_lut, curv_weights)
        return euler[1:], curvature[1:]
    else:
        # Fallback: pure numpy (slower, O(K*N))
        ni, nj, nk = labels.shape
        euler = np.zeros(K + 1, dtype=np.float64)
        curvature = np.zeros(K + 1, dtype=np.float64)

        # Pad labels
        pad = np.zeros((ni + 2, nj + 2, nk + 2), dtype=labels.dtype)
        pad[1:-1, 1:-1, 1:-1] = labels

        # Get all 8 corners
        c = np.stack([
            pad[:-1, :-1, :-1], pad[1:, :-1, :-1],
            pad[:-1, 1:, :-1], pad[1:, 1:, :-1],
            pad[:-1, :-1, 1:], pad[1:, :-1, 1:],
            pad[:-1, 1:, 1:], pad[1:, 1:, 1:]
        ], axis=-1)  # shape: (ni+1, nj+1, nk+1, 8)

        # For each label, compute configuration and accumulate
        for lbl in range(1, K + 1):
            mask = (c == lbl).astype(np.uint8)
            config = (mask[..., 0] + 2 * mask[..., 1] + 4 * mask[..., 2] + 8 * mask[..., 3] +
                      16 * mask[..., 4] + 32 * mask[..., 5] + 64 * mask[..., 6] + 128 * mask[..., 7])
            euler[lbl] = euler_lut[config].sum()
            curvature[lbl] = curv_weights[config].sum()

        return euler[1:], curvature[1:]


def euler_characteristic_lut(labels: np.ndarray, K: int | None = None) -> np.ndarray:
    """Compute Euler characteristic using 2×2×2 configuration LUT (Numba-accelerated).

    This is faster than euler_characteristic_fast for large grids and many labels,
    as it uses the single-pass Numba approach with proper topology LUT.

    Parameters
    ----------
    labels : 3D int array
        Label array where 0 = background, 1..K = clump labels.
    K : int, optional
        Number of labels. If None, computed from labels.max().

    Returns
    -------
    chi : (K,) float64 array
        Euler characteristic per label.
    """
    euler, _ = minkowski_functionals_single_pass(labels, 1.0, 1.0, 1.0, K)
    return euler


def integrated_mean_curvature(labels: np.ndarray,
                               dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                               K: int | None = None) -> np.ndarray:
    """Compute integrated mean curvature per label using 2×2×2 configuration counting.

    The integrated mean curvature C is the third Minkowski functional.
    It measures the total "edginess" of the surface and is related to the
    integral of mean curvature over the surface.

    Note: For efficiency when computing both Euler and curvature, use
    minkowski_functionals_single_pass() instead.

    Parameters
    ----------
    labels : 3D int array
        Label array where 0 = background, 1..K = clump labels.
    dx, dy, dz : float
        Voxel spacing in each dimension.
    K : int, optional
        Number of labels. If None, computed from labels.max().

    Returns
    -------
    curvature : (K,) float64 array
        Integrated mean curvature per label.
    """
    _, curvature = minkowski_functionals_single_pass(labels, dx, dy, dz, K)
    return curvature


def minkowski_shapefinders(volume: np.ndarray,
                            area: np.ndarray,
                            curvature: np.ndarray,
                            euler_chi: np.ndarray) -> dict[str, np.ndarray]:
    """Compute Minkowski shapefinders from the four Minkowski functionals.

    The shapefinders (T, B, L) characterize the thickness, breadth, and length
    of an object in a way that is robust to curvature. From these, we derive
    filamentarity F and planarity P.

    Parameters
    ----------
    volume : (K,) array
        Minkowski functional W0 = volume.
    area : (K,) array
        Minkowski functional W1 = surface area / 3.
    curvature : (K,) array
        Minkowski functional W2 = integrated mean curvature / 3.
    euler_chi : (K,) array
        Minkowski functional W3 = Euler characteristic.

    Returns
    -------
    dict with keys:
        thickness : (K,) - T = W0 / W1 (characteristic thickness)
        breadth : (K,) - B = W1 / W2 (characteristic breadth)
        length : (K,) - L = W2 / W3 (characteristic length)
        filamentarity : (K,) - F = (B - T) / (B + T), 0 for sphere, 1 for filament
        planarity : (K,) - P = (L - B) / (L + B), 0 for sphere, 1 for sheet
    """
    K = volume.shape[0]
    if K == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "thickness": empty,
            "breadth": empty,
            "length": empty,
            "filamentarity": empty,
            "planarity": empty,
        }

    eps = 1e-30

    # Convert to Minkowski functional convention
    # W0 = V, W1 = S/3, W2 = C/3, W3 = χ
    W0 = volume.astype(np.float64, copy=False)
    W1 = area.astype(np.float64, copy=False) / 3.0
    W2 = curvature.astype(np.float64, copy=False) / 3.0
    W3 = euler_chi.astype(np.float64, copy=False)

    # Shapefinders
    # T = W0 / W1 (thickness: volume per unit area)
    thickness = W0 / (W1 + eps)

    # B = W1 / W2 (breadth: area per unit curvature)
    breadth = W1 / (np.abs(W2) + eps)

    # L = W2 / W3 (length: curvature per Euler number)
    # Use abs to handle negative curvature
    length = np.abs(W2) / (np.abs(W3) + eps)

    # Filamentarity: F = (B - T) / (B + T)
    # F = 0 for a sphere (B = T), F → 1 for a thin filament (B >> T)
    filamentarity = (breadth - thickness) / (breadth + thickness + eps)

    # Planarity: P = (L - B) / (L + B)
    # P = 0 for a sphere (L = B), P → 1 for a thin sheet (L >> B)
    planarity = (length - breadth) / (length + breadth + eps)

    return {
        "thickness": thickness,
        "breadth": breadth,
        "length": length,
        "filamentarity": filamentarity,
        "planarity": planarity,
    }


def compute_minkowski_functionals(labels: np.ndarray,
                                   volume: np.ndarray,
                                   area: np.ndarray,
                                   euler_chi: np.ndarray,
                                   dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
                                   K: int | None = None,
                                   min_cells: int = 1000,
                                   cell_count: np.ndarray | None = None,
                                   skip_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute Minkowski functionals and shapefinders for clumps above size threshold.

    This is the main entry point for Tier 2 shape metrics. It computes the
    integrated mean curvature and derives shapefinders (T, B, L, F, P) for
    clumps with cell_count >= min_cells. Smaller clumps get NaN values.

    Parameters
    ----------
    labels : 3D int array
        Label array where 0 = background, 1..K = clump labels.
    volume : (K,) array
        Pre-computed volumes per clump.
    area : (K,) array
        Pre-computed surface areas per clump.
    euler_chi : (K,) array
        Pre-computed Euler characteristics per clump.
    dx, dy, dz : float
        Voxel spacing.
    K : int, optional
        Number of labels.
    min_cells : int, default=1000
        Minimum cell count to compute Minkowski functionals.
        Clumps below this threshold get NaN values.
    cell_count : (K,) array, optional
        Cell counts per clump. If None, will be computed from labels.
    skip_mask : (K,) bool array, optional
        If provided, clumps where skip_mask[i] is True will be skipped
        (e.g., boundary-touching clumps that span multiple nodes).

    Returns
    -------
    dict with keys:
        integrated_curvature : (K,) - integrated mean curvature per clump
        thickness : (K,) - shapefinder T
        breadth : (K,) - shapefinder B
        length : (K,) - shapefinder L
        filamentarity : (K,) - F = (B-T)/(B+T)
        planarity : (K,) - P = (L-B)/(L+B)
        minkowski_computed : (K,) bool - True if Minkowski was computed for this clump
    """
    K = _ensure_K(labels, K)
    if K == 0:
        empty = np.zeros((0,), dtype=np.float64)
        empty_bool = np.zeros((0,), dtype=bool)
        return {
            "integrated_curvature": empty,
            "thickness": empty,
            "breadth": empty,
            "length": empty,
            "filamentarity": empty,
            "planarity": empty,
            "minkowski_computed": empty_bool,
        }

    # Get cell counts if not provided
    if cell_count is None:
        cell_count = num_cells(labels, K=K)

    # Determine which clumps meet the threshold AND are not in skip_mask
    compute_mask = cell_count >= min_cells
    if skip_mask is not None:
        compute_mask = compute_mask & ~skip_mask
    n_compute = np.sum(compute_mask)

    # Initialize outputs with NaN
    curvature_out = np.full(K, np.nan, dtype=np.float64)
    thickness_out = np.full(K, np.nan, dtype=np.float64)
    breadth_out = np.full(K, np.nan, dtype=np.float64)
    length_out = np.full(K, np.nan, dtype=np.float64)
    filamentarity_out = np.full(K, np.nan, dtype=np.float64)
    planarity_out = np.full(K, np.nan, dtype=np.float64)

    if n_compute > 0:
        # Get indices of labels to compute
        compute_indices = np.where(compute_mask)[0]

        # Create a masked labels array: zero out labels we don't need to compute
        # This dramatically reduces K_effective for the O(N × K) Numba loop
        labels_masked = labels.copy()
        label_to_new = np.zeros(K + 1, dtype=np.int32)
        for new_idx, old_idx in enumerate(compute_indices):
            label_to_new[old_idx + 1] = new_idx + 1  # +1 because labels are 1-indexed
        labels_masked = label_to_new[labels_masked]
        K_eff = len(compute_indices)

        # Compute integrated mean curvature only for the masked subset
        curvature_subset = integrated_mean_curvature(labels_masked, dx, dy, dz, K=K_eff)

        # Map euler_chi to the subset for shapefinder computation
        euler_subset = euler_chi[compute_indices]
        volume_subset = volume[compute_indices]
        area_subset = area[compute_indices]

        # Compute shapefinders for qualifying clumps
        shapefinders = minkowski_shapefinders(volume_subset, area_subset, curvature_subset, euler_subset)

        # Store results back into full arrays
        curvature_out[compute_mask] = curvature_subset
        thickness_out[compute_mask] = shapefinders["thickness"]
        breadth_out[compute_mask] = shapefinders["breadth"]
        length_out[compute_mask] = shapefinders["length"]
        filamentarity_out[compute_mask] = shapefinders["filamentarity"]
        planarity_out[compute_mask] = shapefinders["planarity"]

    return {
        "integrated_curvature": curvature_out,
        "thickness": thickness_out,
        "breadth": breadth_out,
        "length": length_out,
        "filamentarity": filamentarity_out,
        "planarity": planarity_out,
        "minkowski_computed": compute_mask,
    }


# ---------------------------------------------------------------------------
# Tier 1b: Bounding box aspect ratios and curvature flag
# ---------------------------------------------------------------------------

def bbox_shape_metrics(bbox_ijk: np.ndarray,
                       principal_axes_lengths: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute shape metrics from bounding boxes.

    Parameters
    ----------
    bbox_ijk : (K, 6) array
        Bounding boxes [i_min, i_max, j_min, j_max, k_min, k_max] per clump.
    principal_axes_lengths : (K, 3) array, optional
        Principal axis lengths (a >= b >= c) from inertia tensor.
        If provided, computes curvature flag by comparing bbox vs inertia elongation.

    Returns
    -------
    dict with keys:
        bbox_lengths : (K, 3) - sorted bbox dimensions (L >= M >= S)
        bbox_elongation : (K,) - L / S ratio
        bbox_flatness : (K,) - M / L ratio
        curvature_flag : (K,) bool - True if bbox is much more cubic than inertia suggests
                         (only if principal_axes_lengths provided)
    """
    K = bbox_ijk.shape[0]
    if K == 0:
        empty = np.zeros((0,), dtype=np.float64)
        empty3 = np.zeros((0, 3), dtype=np.float64)
        result = {
            "bbox_lengths": empty3,
            "bbox_elongation": empty,
            "bbox_flatness": empty,
        }
        if principal_axes_lengths is not None:
            result["curvature_flag"] = np.zeros((0,), dtype=bool)
        return result

    eps = 1e-30

    # Extract bbox dimensions: (i_max - i_min, j_max - j_min, k_max - k_min)
    lengths = np.zeros((K, 3), dtype=np.float64)
    lengths[:, 0] = bbox_ijk[:, 1] - bbox_ijk[:, 0]  # i extent
    lengths[:, 1] = bbox_ijk[:, 3] - bbox_ijk[:, 2]  # j extent
    lengths[:, 2] = bbox_ijk[:, 5] - bbox_ijk[:, 4]  # k extent

    # Sort so L >= M >= S (descending order)
    bbox_lengths = np.sort(lengths, axis=1)[:, ::-1]
    L = bbox_lengths[:, 0]
    M = bbox_lengths[:, 1]
    S = bbox_lengths[:, 2]

    # Bbox elongation: longest / shortest
    bbox_elongation = L / (S + eps)

    # Bbox flatness: middle / longest
    bbox_flatness = M / (L + eps)

    result = {
        "bbox_lengths": bbox_lengths,
        "bbox_elongation": bbox_elongation,
        "bbox_flatness": bbox_flatness,
    }

    # Curvature flag: compare bbox elongation to inertia-based elongation
    # If bbox is much more cubic than inertia suggests, clump is probably curved
    if principal_axes_lengths is not None:
        a = principal_axes_lengths[:, 0]
        c = principal_axes_lengths[:, 2]
        inertia_elongation = a / (c + eps)

        # Heuristic: if bbox_elongation < 0.7 * inertia_elongation, flag as curved
        # This catches cases where a curved filament has high inertia elongation
        # but fits in a more compact bounding box
        curvature_flag = bbox_elongation < (0.7 * inertia_elongation)
        result["curvature_flag"] = curvature_flag

    return result


# ---------------------------------------------------------------------------
# 2D Perimeter Functions (for fractal surface analysis)
# ---------------------------------------------------------------------------

def crack_perimeter_2d(mask: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> float:
    """Compute perimeter by counting foreground-background edges (crack perimeter).

    This is the exact perimeter of the pixelated shape: sum of all cell edges
    that separate foreground from background.

    Parameters
    ----------
    mask : 2D bool array
        Binary mask (True = foreground)
    dx, dy : float
        Pixel sizes in physical units

    Returns
    -------
    perimeter : float
        Total perimeter length in physical units

    Notes
    -----
    Handles multiple connected components and holes correctly.
    """
    mask = mask.astype(bool)
    if mask.size == 0 or not mask.any():
        return 0.0

    # Horizontal edges (between vertically adjacent pixels)
    # An edge exists where mask[i,j] != mask[i+1,j]
    h_edges = np.sum(mask[:-1, :] != mask[1:, :]) * dx

    # Vertical edges (between horizontally adjacent pixels)
    v_edges = np.sum(mask[:, :-1] != mask[:, 1:]) * dy

    # Boundary edges (edge of domain counts if pixel is foreground)
    top = np.sum(mask[0, :]) * dx
    bottom = np.sum(mask[-1, :]) * dx
    left = np.sum(mask[:, 0]) * dy
    right = np.sum(mask[:, -1]) * dy

    perimeter = h_edges + v_edges + top + bottom + left + right

    return float(perimeter)


def contour_perimeter_2d(mask: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> float:
    """Compute perimeter via marching squares contour extraction.

    This gives a slightly smoothed perimeter that interpolates through
    pixel centers.

    Parameters
    ----------
    mask : 2D bool array
        Binary mask
    dx, dy : float
        Pixel sizes in physical units

    Returns
    -------
    perimeter : float
        Total perimeter length (sum over all contours)

    Notes
    -----
    Requires scikit-image. Falls back to crack_perimeter if unavailable.
    """
    mask = mask.astype(bool)
    if mask.size == 0 or not mask.any():
        return 0.0

    try:
        from skimage import measure
    except ImportError:
        # Fallback to crack perimeter if skimage not available
        return crack_perimeter_2d(mask, dx, dy)

    # Pad to ensure contours close at boundaries
    padded = np.pad(mask.astype(float), 1, mode='constant', constant_values=0)

    # Find contours at 0.5 level
    contours = measure.find_contours(padded, level=0.5)

    total_perimeter = 0.0
    for contour in contours:
        # Contour is (N, 2) array of (row, col) coordinates
        # Convert to physical units and compute arc length
        phys_coords = contour * np.array([dy, dx])  # (row, col) -> (y, x)
        segments = np.diff(phys_coords, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        total_perimeter += np.sum(lengths)

    return float(total_perimeter)


def measure_2d_slice(mask: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                     min_area_pixels: int = 10) -> dict:
    """Measure area, perimeter, and solidity for a 2D binary mask.

    Handles multiple connected components by summing for totals,
    and returns individual component measurements for filtering.

    Solidity = Area / Convex_Hull_Area.
    High solidity (>0.9) indicates a clean cross-section (oval/blob).
    Low solidity indicates a complex shape (folded ribbon/U-shape).

    Parameters
    ----------
    mask : 2D bool array
        Binary mask
    dx, dy : float
        Pixel sizes in physical units
    min_area_pixels : int
        Ignore components smaller than this

    Returns
    -------
    dict with keys:
        valid : bool - True if slice contains measurable structure
        area : float - total area (physical units)
        perimeter_crack : float - crack perimeter
        perimeter_contour : float - contour perimeter
        n_components : int - number of connected components
        component_areas : list - individual component areas
        component_perimeters : list - individual component perimeters (crack)
        component_solidities : list - individual component solidities
    """
    from scipy import ndimage

    mask = mask.astype(bool)

    # Label connected components
    labeled, n_components = ndimage.label(mask)

    if n_components == 0:
        return {
            'valid': False,
            'area': 0.0,
            'perimeter_crack': 0.0,
            'perimeter_contour': 0.0,
            'n_components': 0,
            'component_areas': [],
            'component_perimeters': [],
            'component_solidities': []
        }

    # Try to import convex_hull_image for solidity calculation
    try:
        from skimage.morphology import convex_hull_image
        has_convex_hull = True
    except ImportError:
        has_convex_hull = False

    # Measure each component
    component_areas = []
    component_perimeters_crack = []
    component_perimeters_contour = []
    component_solidities = []

    for i in range(1, n_components + 1):
        comp_mask = (labeled == i)
        area_pixels = np.sum(comp_mask)

        if area_pixels < min_area_pixels:
            continue

        area = area_pixels * dx * dy
        p_crack = crack_perimeter_2d(comp_mask, dx, dy)
        p_contour = contour_perimeter_2d(comp_mask, dx, dy)

        # Calculate Solidity = Area / Convex_Hull_Area
        if has_convex_hull:
            try:
                chull = convex_hull_image(comp_mask)
                chull_area_pixels = np.sum(chull)
                solidity = area_pixels / chull_area_pixels if chull_area_pixels > 0 else 0.0
            except Exception:
                solidity = 1.0  # Fallback for edge cases (single pixel, etc.)
        else:
            solidity = 1.0  # No skimage, assume solid

        component_areas.append(area)
        component_perimeters_crack.append(p_crack)
        component_perimeters_contour.append(p_contour)
        component_solidities.append(solidity)

    if len(component_areas) == 0:
        return {
            'valid': False,
            'area': 0.0,
            'perimeter_crack': 0.0,
            'perimeter_contour': 0.0,
            'n_components': 0,
            'component_areas': [],
            'component_perimeters': [],
            'component_solidities': []
        }

    return {
        'valid': True,
        'area': sum(component_areas),
        'perimeter_crack': sum(component_perimeters_crack),
        'perimeter_contour': sum(component_perimeters_contour),
        'n_components': len(component_areas),
        'component_areas': component_areas,
        'component_perimeters': component_perimeters_crack,
        'component_solidities': component_solidities
    }

