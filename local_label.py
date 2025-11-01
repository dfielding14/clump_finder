from __future__ import annotations

import numpy as np
from numba import njit


@njit(inline='always')
def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(inline='always')
def uf_union(parent, a, b):
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _neighbor_offsets(connectivity: int) -> np.ndarray:
    """Return half-neighborhood offsets for the requested connectivity.

    Offsets are shaped (M,3) with entries (di,dj,dk) relative to current voxel.
    Scanning order is i (x), then j (y), then k (z).
    """
    if connectivity == 6:
        offs = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    elif connectivity == 18:
        offs = [
            # same slice (dz=0), previous row/col
            (-1, 0, 0), (0, -1, 0), (-1, -1, 0), (1, -1, 0),
            # previous slice (dz=-1), exclude 3D corners (dx and dy both non-zero)
            (0, 0, -1), (-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1),
        ]
    elif connectivity == 26:
        offs = [
            # same slice (dz=0)
            (-1, 0, 0), (0, -1, 0), (-1, -1, 0), (1, -1, 0),
            # previous slice (dz=-1), all 3x3 neighbors
            (-1, -1, -1), (0, -1, -1), (1, -1, -1),
            (-1, 0, -1),  (0, 0, -1),  (1, 0, -1),
            (-1, 1, -1),  (0, 1, -1),  (1, 1, -1),
        ]
    else:
        raise ValueError("connectivity must be 6, 18, or 26")
    return np.asarray(offs, dtype=np.int64)


@njit
def _ccl_tile(mask_tile: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    """Label a boolean tile using two-pass union-find CCL with given neighbor set.

    Returns a label array of same shape (uint32) with labels starting at 1; 0 is background.
    """
    ni, nj, nk = mask_tile.shape
    labels = np.zeros((ni, nj, nk), dtype=np.uint32)

    # Worst-case label count within the tile is number of voxels; to cap memory,
    # we grow parent lazily by allocating a generous upper bound: number of voxels // 2 + 1
    # but for simplicity here allocate full upper bound + 1 (small tiles make this ok).
    max_labels = ni * nj * nk + 1
    parent = np.arange(max_labels, dtype=np.int64)
    next_label = 1

    # First pass: assign and union
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                if not mask_tile[i, j, k]:
                    continue
                # Collect neighbor labels
                lbl = 0
                for t in range(neigh.shape[0]):
                    di = i + neigh[t, 0]
                    dj = j + neigh[t, 1]
                    dk = k + neigh[t, 2]
                    if di < 0 or dj < 0 or dk < 0 or di >= ni or dj >= nj or dk >= nk:
                        continue
                    nb = labels[di, dj, dk]
                    if nb != 0:
                        if lbl == 0 or nb < lbl:
                            lbl = nb
                if lbl == 0:
                    lbl = next_label
                    next_label += 1
                labels[i, j, k] = lbl

                # Union with any different neighbor labels
                for t in range(neigh.shape[0]):
                    di = i + neigh[t, 0]
                    dj = j + neigh[t, 1]
                    dk = k + neigh[t, 2]
                    if di < 0 or dj < 0 or dk < 0 or di >= ni or dj >= nj or dk >= nk:
                        continue
                    nb = labels[di, dj, dk]
                    if nb != 0 and nb != lbl:
                        uf_union(parent, lbl, nb)

    # Second pass: compress to representatives
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                l = labels[i, j, k]
                if l != 0:
                    labels[i, j, k] = uf_find(parent, l)

    return labels


def _compact_labels_inplace(labels: np.ndarray) -> int:
    """Relabel positive labels in-place to compact 1..K. Return K.

    This runs in Python/NumPy (not Numba) for simplicity.
    """
    if labels.size == 0:
        return 0
    u = np.unique(labels)
    u = u[u != 0]
    if u.size == 0:
        return 0
    lut = np.zeros(int(u.max()) + 1, dtype=np.uint32)
    lut[u] = np.arange(1, u.size + 1, dtype=np.uint32)
    labels[labels > 0] = lut[labels[labels > 0]]
    return int(u.size)


def label_3d(mask: np.ndarray,
             tile_shape=(128, 128, 128),
             connectivity: int = 6,
             halo: int = 0) -> np.ndarray:
    """Label a 3-D boolean volume with optional halo.

    - mask: boolean array, possibly with 1-cell halo on each face.
    - tile_shape: interior tiling for per-tile CCL.
    - connectivity: 6, 18, or 26.
    - halo: number of halo cells (0 or 1). If >0, output excludes the halo.

    Returns labels (uint32) of shape interior-only if halo>0; same shape as mask otherwise.
    Labels are compacted to 1..K within the node-local subvolume (no global stitching).
    """
    # Production stitching and cross-tile merges are correct only for 6-connectivity
    # (face adjacency). 18/26 would require additional cross-tile edge/corner merges.
    if connectivity != 6:
        raise NotImplementedError(
            "Stitched runs support connectivity=6 only. 18/26 require additional "
            "cross-tile edge/corner merges."
        )
    neigh = _neighbor_offsets(connectivity)

    if halo > 0:
        s = slice(halo, -halo)
        M = mask[s, s, s]
    else:
        M = mask

    ni, nj, nk = M.shape
    li, lj, lk = tile_shape
    labels = np.zeros((ni, nj, nk), dtype=np.uint32)

    # Pass 1: per-tile labeling with local compaction
    tile_labels_bases = []  # tuples of (i0,i1,j0,j1,k0,k1,gbase)
    gbase = 0
    for k0 in range(0, nk, lk):
        k1 = min(k0 + lk, nk)
        for j0 in range(0, nj, lj):
            j1 = min(j0 + lj, nj)
            for i0 in range(0, ni, li):
                i1 = min(i0 + li, ni)
                tile = M[i0:i1, j0:j1, k0:k1]
                tlabels = _ccl_tile(tile, neigh)
                # compact local labels to 1..Kt
                Kt = _compact_labels_inplace(tlabels)
                if Kt > 0:
                    tlabels = tlabels.astype(np.uint32, copy=False)
                    if gbase != 0:
                        mask_pos = tlabels > 0
                        if mask_pos.any():
                            tlabels = tlabels.copy()
                            tlabels[mask_pos] += np.uint32(gbase)
                    labels[i0:i1, j0:j1, k0:k1] = tlabels
                    tile_labels_bases.append((i0, i1, j0, j1, k0, k1, gbase))
                    gbase += Kt
                else:
                    labels[i0:i1, j0:j1, k0:k1] = 0

    total_labels = gbase
    if total_labels == 0:
        return labels

    # Global merge across tile faces using a DSU of size total_labels+1
    parent = np.arange(total_labels + 1, dtype=np.int64)

    # Helper to union touching labels across +i, +j, +k tile faces
    def merge_faces_i(i0, i1, j0, j1, k0, k1):
        # +i face with neighbor tile starting at i1 if exists
        if i1 >= ni:
            return
        a = labels[i1 - 1, j0:j1, k0:k1]
        b = labels[i1, j0:j1, k0:k1]
        # touching if both foreground and labels differ
        mask_touch = (a != 0) & (b != 0)
        La = a[mask_touch]
        Lb = b[mask_touch]
        for idx in range(La.size):
            if La[idx] != Lb[idx]:
                uf_union(parent, int(La[idx]), int(Lb[idx]))

    def merge_faces_j(i0, i1, j0, j1, k0, k1):
        if j1 >= nj:
            return
        a = labels[i0:i1, j1 - 1, k0:k1]
        b = labels[i0:i1, j1, k0:k1]
        mask_touch = (a != 0) & (b != 0)
        La = a[mask_touch]
        Lb = b[mask_touch]
        for idx in range(La.size):
            if La[idx] != Lb[idx]:
                uf_union(parent, int(La[idx]), int(Lb[idx]))

    def merge_faces_k(i0, i1, j0, j1, k0, k1):
        if k1 >= nk:
            return
        a = labels[i0:i1, j0:j1, k1 - 1]
        b = labels[i0:i1, j0:j1, k1]
        mask_touch = (a != 0) & (b != 0)
        La = a[mask_touch]
        Lb = b[mask_touch]
        for idx in range(La.size):
            if La[idx] != Lb[idx]:
                uf_union(parent, int(La[idx]), int(Lb[idx]))

    for (i0, i1, j0, j1, k0, k1, _gb) in tile_labels_bases:
        merge_faces_i(i0, i1, j0, j1, k0, k1)
        merge_faces_j(i0, i1, j0, j1, k0, k1)
        merge_faces_k(i0, i1, j0, j1, k0, k1)

    # Relabel to representatives
    if total_labels > 0:
        flat = labels.ravel()
        for idx in range(flat.size):
            l = flat[idx]
            if l != 0:
                flat[idx] = uf_find(parent, int(l))
        labels = flat.reshape(labels.shape)
        _compact_labels_inplace(labels)

    return labels
