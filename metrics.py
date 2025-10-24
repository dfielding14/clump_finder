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

