from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Tuple

import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from local_label import label_3d
import metrics as M
from stitch import stitch_reduce, index_parts, build_edges


def _split_axis(n: int, p: int) -> list[Tuple[int, int]]:
    base, rem = divmod(n, p)
    s = 0
    out = []
    for c in range(p):
        extra = 1 if c < rem else 0
        e = s + base + extra
        out.append((s, e))
        s = e
    return out


def _make_fields(N: int, seed: int = 42, field_type: str = "simple", beta: float = -2.0,
                 T_limits: tuple[float, float] = (0.01, 1.0), R_limits: tuple[float, float] = (1.0, 100.0)):
    """Generate synthetic fields with target ranges.

    field_type:
      - "simple": standard normal in real space.
      - "powerlaw": Gaussian random field with isotropic P(k) ~ k^beta.

    The mapping preserves morphology via a monotone transform:
      temp = T_lo * (T_hi/T_lo) ** U,
      dens = R_lo * (R_hi/R_lo) ** (1 - U),
    where U = sigmoid(f) = 1/(1+exp(-f)). This makes temp low where dens is high (anticorrelated).
    """
    if field_type == "simple":
        rng = np.random.default_rng(seed)
        f = rng.normal(size=(N, N, N)).astype(np.float32)
    elif field_type == "powerlaw":
        f = _make_powerlaw_scalar_field(N, beta=beta, seed=seed)
    else:
        raise ValueError("field_type must be 'simple' or 'powerlaw'")

    # Monotone [0,1] map preserving level sets
    U = 1.0 / (1.0 + np.exp(-f.astype(np.float64)))
    T_lo, T_hi = float(T_limits[0]), float(T_limits[1])
    R_lo, R_hi = float(R_limits[0]), float(R_limits[1])
    T_ratio = T_hi / T_lo
    R_ratio = R_hi / R_lo
    temp = T_lo * np.power(T_ratio, U)
    dens = R_lo * np.power(R_ratio, (1.0 - U))
    return dens.astype(np.float32), temp.astype(np.float32)


def _make_powerlaw_scalar_field(N: int, beta: float = -2.0, seed: int = 42) -> np.ndarray:
    """Return a real Gaussian random field with P(k) ~ k^beta using rfftn/irfftn."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    kz = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kk = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    kk[0, 0, 0] = np.inf
    amp = np.power(kk, beta / 2.0)
    amp[~np.isfinite(amp)] = 0.0
    noise = rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape)
    Fk = noise * amp
    # Specify axes explicitly to avoid NumPy 2.0 deprecation when providing 's'
    f = np.fft.irfftn(Fk, s=(N, N, N), axes=(0, 1, 2)).astype(np.float32, copy=False)
    f = f / (f.std() + 1e-12)
    return f


def _periodic_relabel_and_area(labels: np.ndarray, dx: float, dy: float, dz: float) -> tuple[np.ndarray, np.ndarray]:
    """Relabel labels to enforce periodic connectivity and compute corrected exposed area.

    Returns (labels_reindexed, area_per_label) where labels are compacted 1..K.
    """
    ni, nj, nk = labels.shape
    K0 = int(labels.max())
    if K0 == 0:
        return labels, np.zeros((0,), dtype=np.float64)
    # DSU over existing labels
    parent = np.arange(K0 + 1, dtype=np.int64)
    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def uf_union(a, b):
        ra, rb = uf_find(a), uf_find(b)
        if ra != rb:
            parent[rb] = ra

    # Merge across periodic faces where both sides are foreground
    A, B = labels[0, :, :], labels[ni - 1, :, :]
    m = (A > 0) & (B > 0)
    if m.any():
        La, Lb = A[m], B[m]
        for a, b in zip(La, Lb):
            if a != b:
                uf_union(int(a), int(b))
    A, B = labels[:, 0, :], labels[:, nj - 1, :]
    m = (A > 0) & (B > 0)
    if m.any():
        La, Lb = A[m], B[m]
        for a, b in zip(La, Lb):
            if a != b:
                uf_union(int(a), int(b))
    A, B = labels[:, :, 0], labels[:, :, nk - 1]
    m = (A > 0) & (B > 0)
    if m.any():
        La, Lb = A[m], B[m]
        for a, b in zip(La, Lb):
            if a != b:
                uf_union(int(a), int(b))

    # Map to representatives then compact
    flat = labels.ravel()
    if flat.size:
        lut = np.arange(K0 + 1, dtype=np.int64)
        for l in np.unique(flat):
            if l != 0:
                lut[l] = uf_find(int(l))
        mapped = lut[flat]
        # compact 1..K
        u = np.unique(mapped)
        u = u[u != 0]
        lut2 = np.zeros(int(u.max()) + 1, dtype=np.int64)
        lut2[u] = np.arange(1, u.size + 1, dtype=np.int64)
        relabeled = lut2[mapped].reshape(labels.shape).astype(np.uint32)
    else:
        relabeled = labels

    # Area with periodic correction: start from non-periodic area and subtract interior wrap faces
    area = M.exposed_area(relabeled, dx, dy, dz, K=int(relabeled.max()))
    K = int(relabeled.max())
    if K > 0:
        # X wrap faces
        a_face = dy * dz
        A = relabeled[0, :, :]
        B = relabeled[ni - 1, :, :]
        m = (A > 0) & (B > 0) & (A == B)
        if m.any():
            cnt = np.bincount(A[m], minlength=K + 1)
            area -= 2.0 * a_face * cnt[1:]
        # Y wrap
        a_face = dx * dz
        A = relabeled[:, 0, :]
        B = relabeled[:, nj - 1, :]
        m = (A > 0) & (B > 0) & (A == B)
        if m.any():
            cnt = np.bincount(A[m], minlength=K + 1)
            area -= 2.0 * a_face * cnt[1:]
        # Z wrap
        a_face = dx * dy
        A = relabeled[:, :, 0]
        B = relabeled[:, :, nk - 1]
        m = (A > 0) & (B > 0) & (A == B)
        if m.any():
            cnt = np.bincount(A[m], minlength=K + 1)
            area -= 2.0 * a_face * cnt[1:]

    return relabeled, area


def _baseline_single_volume(dens, temp, dx=1.0, dy=1.0, dz=1.0, thr=0.1, by="temperature", return_masses: bool = False,
                            periodic: bool = True):
    field = temp if by == "temperature" else dens
    labels = label_3d(field < thr, tile_shape=(128, 128, 128), connectivity=6, halo=0)
    if periodic:
        labels, area = _periodic_relabel_and_area(labels, dx, dy, dz)
    K = int(labels.max())
    cell = M.num_cells(labels, K=K)
    vol = M.volumes(cell, dx, dy, dz)
    mass = M.masses(labels, dens, dx, dy, dz, K=K)
    bbox = M.compute_bboxes(labels, ((0, labels.shape[0]), (0, labels.shape[1]), (0, labels.shape[2])), K=K)
    if not periodic:
        area = M.exposed_area(labels, dx, dy, dz, K=K)
    out = {
        "K": K,
        "cell_sorted": np.sort(cell),
        "area_sum": float(area.sum()),
        "vol_sum": float(vol.sum()),
        "mass_sum": float(mass.sum()),
        "bbox_sorted": np.sort(bbox, axis=0),
    }
    if return_masses:
        out["masses"] = mass
    return out


def _extract_with_halo(arr: np.ndarray, i0:int, i1:int, j0:int, j1:int, k0:int, k1:int, halo:int = 1) -> np.ndarray:
    """Extract a subvolume with periodic halo padding of width `halo`."""
    N = arr.shape[0]
    # ranges with wrap
    ii = (np.arange(i0 - halo, i1 + halo) % N)
    jj = (np.arange(j0 - halo, j1 + halo) % N)
    kk = (np.arange(k0 - halo, k1 + halo) % N)
    return arr[np.ix_(ii, jj, kk)]


def _write_parts(tmpdir: str,
                 dens,
                 temp,
                 px: int,
                 py: int,
                 pz: int,
                 thr: float = 0.1,
                 by: str = "temperature",
                 use_halo: bool = True,
                 overlap: int = 1):
    N = dens.shape[0]
    dx = dy = dz = 1.0
    ix = _split_axis(N, px)
    iy = _split_axis(N, py)
    iz = _split_axis(N, pz)
    if overlap < 1:
        raise ValueError("overlap must be >= 1 for exact stitching")
    rank = 0
    for cx, (i0, i1) in enumerate(ix):
        for cy, (j0, j1) in enumerate(iy):
            for cz, (k0, k1) in enumerate(iz):
                sub_d = dens[i0:i1, j0:j1, k0:k1]
                sub_t = temp[i0:i1, j0:j1, k0:k1]
                if use_halo:
                    halo = 1
                    fld_h = _extract_with_halo(temp if by == "temperature" else dens,
                                               i0, i1, j0, j1, k0, k1, halo=halo)
                else:
                    halo = 0
                    fld_h = (sub_t if by == "temperature" else sub_d)
                labels_ext = label_3d(fld_h < thr, tile_shape=(128, 128, 128), connectivity=6, halo=0)
                if halo > 0:
                    labels = labels_ext[halo:-halo, halo:-halo, halo:-halo]
                else:
                    labels = labels_ext
                core_vals = np.unique(labels)
                core_vals = core_vals[core_vals != 0]
                lut = np.zeros(int(labels_ext.max()) + 1, dtype=np.uint32)
                lut[core_vals] = np.arange(1, core_vals.size + 1, dtype=np.uint32)
                labels = lut[labels]
                labels_ext = lut[labels_ext]
                K = int(core_vals.size)
                cell = M.num_cells(labels, K=K)
                vol = M.volumes(cell, dx, dy, dz)
                mass = M.masses(labels, sub_d, dx, dy, dz, K=K)
                cvol, cmass = M.centroids(labels, sub_d, dx, dy, dz, (0, 0, 0), ((i0, i1), (j0, j1), (k0, k1)), K=K)
                area = M.exposed_area(labels, dx, dy, dz, K=K)
                bbox = M.compute_bboxes(labels, ((i0, i1), (j0, j1), (k0, k1)), K=K)

                rank_ids = np.arange(1, K + 1, dtype=np.int32)
                # Faces + 15-bit face-pair bits per label (unfiltered labels)
                face_xneg = labels[0, :, :].astype(np.uint32)
                face_xpos = labels[-1, :, :].astype(np.uint32)
                face_yneg = labels[:, 0, :].astype(np.uint32)
                face_ypos = labels[:, -1, :].astype(np.uint32)
                face_zneg = labels[:, :, 0].astype(np.uint32)
                face_zpos = labels[:, :, -1].astype(np.uint32)
                presence = np.zeros((K, 6), dtype=bool)
                for idx, arr in enumerate((face_xneg, face_xpos, face_yneg, face_ypos, face_zneg, face_zpos)):
                    u = np.unique(arr); u = u[(u > 0) & (u <= K)]
                    presence[u - 1, idx] = True
                pair_bits = np.zeros((K,), dtype=np.uint16)
                pairs = [(0,1),(0,2),(0,3),(0,4),(0,5),
                         (1,2),(1,3),(1,4),(1,5),
                         (2,3),(2,4),(2,5),
                         (3,4),(3,5),
                         (4,5)]
                for bit,(a,b) in enumerate(pairs):
                    both = presence[:,a] & presence[:,b]
                    pair_bits |= (both.astype(np.uint16) << np.uint16(bit))

                # Overlap planes: use halo layers labelled in labels_ext
                ovlp = int(overlap)
                if ovlp != 1:
                    raise NotImplementedError("overlap width >1 not yet supported in synthetic tiler")
                if ovlp > halo:
                    raise ValueError("overlap cannot exceed halo width in synthetic tiler")
                ni_c, nj_c, nk_c = labels.shape
                i_start = halo
                i_end = halo + ni_c
                j_start = halo
                j_end = halo + nj_c
                k_start = halo
                k_end = halo + nk_c
                ov_xneg = labels_ext[i_start:i_start + ovlp, j_start:j_end, k_start:k_end].astype(np.uint32, copy=False)
                ov_xpos = labels_ext[i_end - ovlp:i_end, j_start:j_end, k_start:k_end].astype(np.uint32, copy=False)
                ov_yneg = labels_ext[i_start:i_end, j_start:j_start + ovlp, k_start:k_end].astype(np.uint32, copy=False)
                ov_ypos = labels_ext[i_start:i_end, j_end - ovlp:j_end, k_start:k_end].astype(np.uint32, copy=False)
                ov_zneg = labels_ext[i_start:i_end, j_start:j_end, k_start:k_start + ovlp].astype(np.uint32, copy=False)
                ov_zpos = labels_ext[i_start:i_end, j_start:j_end, k_end - ovlp:k_end].astype(np.uint32, copy=False)
                ov_xneg = ov_xneg[0]
                ov_xpos = ov_xpos[-1]
                ov_yneg = ov_yneg[:, 0, :]
                ov_ypos = ov_ypos[:, -1, :]
                ov_zneg = ov_zneg[:, :, 0]
                ov_zpos = ov_zpos[:, :, -1]

                out = {
                    "label_ids": rank_ids,
                    "cell_count": cell,
                    "volume": vol,
                    "mass": mass,
                    "area": area,
                    "centroid_vol": cvol,
                    "centroid_mass": cmass,
                    "bbox_ijk": bbox,
                    "voxel_spacing": np.array([dx, dy, dz]),
                    "overlap_width": np.int32(ovlp),
                    "ovlp_xneg": ov_xneg,
                    "ovlp_xpos": ov_xpos,
                    "ovlp_yneg": ov_yneg,
                    "ovlp_ypos": ov_ypos,
                    "ovlp_zneg": ov_zneg,
                    "ovlp_zpos": ov_zpos,
                    "face_xneg": face_xneg,
                    "face_xpos": face_xpos,
                    "face_yneg": face_yneg,
                    "face_ypos": face_ypos,
                    "face_zneg": face_zneg,
                    "face_zpos": face_zpos,
                    "face_pair_bits": pair_bits,
                    # shells for exact mode (t=3)
                    "shell_t": np.int32(3),
                    "shell_xneg": labels[0:3, :, :].astype(np.uint32),
                    "shell_xpos": labels[-3:, :, :].astype(np.uint32),
                    "shell_yneg": labels[:, 0:3, :].astype(np.uint32),
                    "shell_ypos": labels[:, -3:, :].astype(np.uint32),
                    "shell_zneg": labels[:, :, 0:3].astype(np.uint32),
                    "shell_zpos": labels[:, :, -3:].astype(np.uint32),
                }
                np.savez(os.path.join(tmpdir, f"clumps_rank{rank:05d}.npz"), **out)
                meta = {
                    "rank": rank,
                    "coords": (cx, cy, cz),
                    "cart_dims": (px, py, pz),
                    "node_bbox_ijk": [i0, i1, j0, j1, k0, k1],
                    "grid": {"periodic": [True, True, True]},
                    "overlap_width": ovlp,
                    "output_npz": f"clumps_rank{rank:05d}.npz",
                }
                with open(os.path.join(tmpdir, f"clumps_rank{rank:05d}.meta.json"), "w") as f:
                    json.dump(meta, f)
                rank += 1


def _compare(baseline: dict, stitched_npz: str) -> bool:
    with np.load(stitched_npz) as d:
        cc = d["cell_count"]
        area = d["area"]
        volume = d["volume"]
        mass = d["mass"]
        bbox = d["bbox_ijk"]
    try:
        assert cc.size == baseline["K"], f"K mismatch: {cc.size} vs {baseline['K']}"
        assert np.allclose(np.sort(cc), baseline["cell_sorted"]) , "cell_count mismatch"
        assert np.isclose(area.sum(), baseline["area_sum"]) , "area sum mismatch"
        assert np.isclose(volume.sum(), baseline["vol_sum"]) , "volume sum mismatch"
        assert np.isclose(mass.sum(), baseline["mass_sum"]) , "mass sum mismatch"
        # bbox union check: sort rows and compare per-column sorted extrema
        assert np.allclose(np.sort(bbox, axis=0), baseline["bbox_sorted"]) , "bbox mismatch"
        return True
    except AssertionError as e:
        print(f"Equivalence check FAILED: {e}")
        return False


def _diagnose_interfaces(parts_dir: str, dens: np.ndarray, temp: np.ndarray,
                         px: int, py: int, pz: int, by: str, thr: float):
    """Print diagnostics comparing expected face adjacencies vs stitcher's edges.

    - Expected adjacencies come from the threshold mask on the full volume.
    - Observed adjacencies come from the stitcher's face map edges.
    """
    # Build expected counts from mask
    field = temp if by == "temperature" else dens
    mask = field < thr
    N = mask.shape[0]
    ix = _split_axis(N, px)
    iy = _split_axis(N, py)
    iz = _split_axis(N, pz)
    # helper to count across a split axis with periodic wrap
    def count_axis(axis: int) -> int:
        total = 0
        if axis == 0:
            for cx, (i0, i1) in enumerate(ix):
                j0, j1 = 0, N
                k0, k1 = 0, N
                # neighbor in +x with wrap
                nx = (cx + 1) % px
                i0n, i1n = ix[nx]
                a = mask[i1 - 1, :, :]
                b = mask[i0n, :, :]
                total += int(np.count_nonzero(a & b))
        elif axis == 1:
            for cy, (j0, j1) in enumerate(iy):
                # neighbor in +y with wrap
                ny = (cy + 1) % py
                j0n, j1n = iy[ny]
                a = mask[:, j1 - 1, :]
                b = mask[:, j0n, :]
                total += int(np.count_nonzero(a & b))
        else:
            for cz, (k0, k1) in enumerate(iz):
                nz = (cz + 1) % pz
                k0n, k1n = iz[nz]
                a = mask[:, :, k1 - 1]
                b = mask[:, :, k0n]
                total += int(np.count_nonzero(a & b))
        return total

    exp_x = count_axis(0)
    exp_y = count_axis(1)
    exp_z = count_axis(2)

    # Use stitcher's edge builder to count observed edges
    ranks, cart_dims, periodic = index_parts(parts_dir)
    # derive spacing from any part (not needed for counts)
    any_npz = next(iter(ranks.values()))
    dsu, edge_counts, _ = build_edges(ranks, cart_dims, periodic, 1.0, 1.0, 1.0)
    obs_x = sum(edge_counts["x"].values())
    obs_y = sum(edge_counts["y"].values())
    obs_z = sum(edge_counts["z"].values())

    print(f"  Expected face adjacencies (x,y,z): {exp_x}, {exp_y}, {exp_z}")
    print(f"  Observed  face adjacencies (x,y,z): {obs_x}, {obs_y}, {obs_z}")


def _face_only_component_count(mask: np.ndarray, px: int, py: int, pz: int, use_halo: bool = True) -> int:
    """Compute number of components using only face information across tiles.

    - Labels each subvolume locally (6-connected) to define nodes.
    - Builds DSU edges where both faces are True across adjacent tiles.
    - Returns total DSU component count across all non-zero local labels.
    """
    N = mask.shape[0]
    ix = _split_axis(N, px); iy = _split_axis(N, py); iz = _split_axis(N, pz)
    # Assign a global integer id to each (rank, local_label)
    node_base = {}
    total_nodes = 0
    local_labels = {}
    # Label each tile locally
    r = 0
    for cx,(i0,i1) in enumerate(ix):
        for cy,(j0,j1) in enumerate(iy):
            for cz,(k0,k1) in enumerate(iz):
                if use_halo:
                    subh = _extract_with_halo(mask, i0, i1, j0, j1, k0, k1, halo=1)
                    loc = label_3d(subh, tile_shape=(128,128,128), connectivity=6, halo=1)
                else:
                    sub = mask[i0:i1, j0:j1, k0:k1]
                    loc = label_3d(sub, tile_shape=(128,128,128), connectivity=6, halo=0)
                Kp = int(loc.max())
                local_labels[r] = loc
                node_base[r] = total_nodes
                total_nodes += Kp
                r += 1
    if total_nodes == 0:
        return 0
    # DSU
    parent = list(range(total_nodes))
    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def u(a,b):
        ra, rb = f(a), f(b)
        if ra != rb:
            parent[rb] = ra
    # Map coords to rank
    def rank_of(cx,cy,cz):
        return (cx*py + cy)*pz + cz
    # Build cross-tile edges
    for cx,(i0,i1) in enumerate(ix):
        for cy,(j0,j1) in enumerate(iy):
            for cz,(k0,k1) in enumerate(iz):
                r = rank_of(cx,cy,cz)
                loc = local_labels[r]
                base = node_base[r]
                # x neighbor
                nx = (cx+1)%px
                rn = rank_of(nx,cy,cz)
                locn = local_labels[rn]
                a = loc[-1,:,:]
                b = locn[0,:,:]
                m = (a>0) & (b>0)
                if m.any():
                    for la, lb in zip(a[m], b[m]):
                        u(base + int(la) - 1, node_base[rn] + int(lb) - 1)
                # y neighbor
                ny = (cy+1)%py
                rn = rank_of(cx,ny,cz)
                locn = local_labels[rn]
                a = loc[:, -1, :]
                b = locn[:, 0, :]
                m = (a>0) & (b>0)
                if m.any():
                    for la, lb in zip(a[m], b[m]):
                        u(base + int(la) - 1, node_base[rn] + int(lb) - 1)
                # z neighbor
                nz = (cz+1)%pz
                rn = rank_of(cx,cy,nz)
                locn = local_labels[rn]
                a = loc[:, :, -1]
                b = locn[:, :, 0]
                m = (a>0) & (b>0)
                if m.any():
                    for la, lb in zip(a[m], b[m]):
                        u(base + int(la) - 1, node_base[rn] + int(lb) - 1)
    return len({f(i) for i in range(total_nodes)})


def _diagnose_component_mismatch(parts_dir: str, dens: np.ndarray, temp: np.ndarray,
                                 px: int, py: int, pz: int, by: str, thr: float,
                                 max_reports: int = 5):
    """Diagnose if labels belonging to the same baseline component map to multiple DSU roots.

    Prints a few example baseline component IDs with >1 stitched roots.
    """
    # Baseline periodic labeling
    field = temp if by == "temperature" else dens
    labels = label_3d(field < thr, tile_shape=(128, 128, 128), connectivity=6, halo=0)
    labels, _ = _periodic_relabel_and_area(labels, 1.0, 1.0, 1.0)
    N = labels.shape[0]
    ix = _split_axis(N, px); iy = _split_axis(N, py); iz = _split_axis(N, pz)

    # Stitch DSU
    ranks, cart_dims, periodic = index_parts(parts_dir)
    dsu, _, _ = build_edges(ranks, cart_dims, periodic, 1.0, 1.0, 1.0)
    # Count total stitched roots from all gids
    all_gids = []
    for r, info in ranks.items():
        with np.load(info["npz"]) as d:
            lids = d["label_ids"].astype(np.int64)
        all_gids.extend([(np.uint64(r) << np.uint64(32)) | np.uint64(l) for l in lids])
    uniq_roots = {int(dsu.find(g)) for g in all_gids}

    # For each part, re-label local and map (rank,label)->baseline_label via majority vote
    root_sets_per_baseline = {}
    for r, info in ranks.items():
        cx, cy, cz = info["coords"]
        (i0, i1), (j0, j1), (k0, k1) = ((ix[cx][0], ix[cx][1]), (iy[cy][0], iy[cy][1]), (iz[cz][0], iz[cz][1]))
        sub = (temp if by == "temperature" else dens)[i0:i1, j0:j1, k0:k1]
        loc = label_3d(sub < thr, tile_shape=(128, 128, 128), connectivity=6, halo=0)
        Kp = int(loc.max())
        if Kp == 0:
            continue
        # Map local -> baseline majority
        base_slice = labels[i0:i1, j0:j1, k0:k1]
        # For each local label, find most frequent baseline label
        for l in range(1, Kp + 1):
            m = (loc == l)
            if not m.any():
                continue
            bl = base_slice[m]
            # Count baseline labels; ignore 0
            bl = bl[bl > 0]
            if bl.size == 0:
                continue
            vals, cnts = np.unique(bl, return_counts=True)
            b_major = int(vals[np.argmax(cnts)])
            gid = (np.uint64(r) << np.uint64(32)) | np.uint64(l)
            root = int(dsu.find(gid))
            s = root_sets_per_baseline.setdefault(b_major, set())
            s.add(root)

    # Report mismatches where a baseline label maps to >1 stitched root
    bad = [(b, len(s)) for b, s in root_sets_per_baseline.items() if len(s) > 1]
    bad.sort(key=lambda x: x[1], reverse=True)
    if not bad:
        covered_roots = set().union(*root_sets_per_baseline.values()) if root_sets_per_baseline else set()
        print("  No baseline components split across multiple stitched roots.")
        print(f"  Stitched unique roots: {len(uniq_roots)}; roots covered by baseline-map: {len(covered_roots)}")
        missing = uniq_roots - covered_roots
        if missing:
            print(f"  WARNING: {len(missing)} stitched roots not mapped to any baseline component (unexpected)")
        return
    print(f"  Baseline components split across multiple stitched roots: {len(bad)} examples")
    for b, n in bad[:max_reports]:
        print(f"    baseline_label={b} -> stitched_roots={n}")

    # Deep dive into the worst offender
    b_top, _ = bad[0]
    print(f"  [DETAIL] Inspecting baseline_label={b_top}")
    _inspect_baseline_component(parts_dir, labels, b_top, px, py, pz)


def _inspect_baseline_component(parts_dir: str, base_labels: np.ndarray, b_label: int,
                                px: int, py: int, pz: int):
    N = base_labels.shape[0]
    ix = _split_axis(N, px); iy = _split_axis(N, py); iz = _split_axis(N, pz)
    maskb = (base_labels == b_label)
    # Count cross-partition adjacencies restricted to this baseline label
    def count_axis(axis: int) -> int:
        total = 0
        if axis == 0:
            for cx, (i0, i1) in enumerate(ix):
                nx = (cx + 1) % px
                i0n, i1n = ix[nx]
                a = maskb[i1 - 1, :, :]
                b = maskb[i0n, :, :]
                total += int(np.count_nonzero(a & b))
        elif axis == 1:
            for cy, (j0, j1) in enumerate(iy):
                ny = (cy + 1) % py
                j0n, j1n = iy[ny]
                a = maskb[:, j1 - 1, :]
                b = maskb[:, j0n, :]
                total += int(np.count_nonzero(a & b))
        else:
            for cz, (k0, k1) in enumerate(iz):
                nz = (cz + 1) % pz
                k0n, k1n = iz[nz]
                a = maskb[:, :, k1 - 1]
                b = maskb[:, :, k0n]
                total += int(np.count_nonzero(a & b))
        return total
    cx = count_axis(0); cy = count_axis(1); cz = count_axis(2)
    print(f"    Baseline component b={b_label} boundary-adjacencies (x,y,z): {cx}, {cy}, {cz}")
    # If zero across all, then it shouldn't span ranks; else unions should connect it.
    _subgraph_component_connectivity(parts_dir, base_labels, b_label, px, py, pz)


def _subgraph_component_connectivity(parts_dir: str, base_labels: np.ndarray, b_label: int,
                                     px: int, py: int, pz: int):
    """Build the stitching subgraph for a single baseline component and report connectivity.
    Edges-first version: we only instantiate nodes that actually appear in a baseline-gated edge."""
    N = base_labels.shape[0]
    ix = _split_axis(N, px); iy = _split_axis(N, py); iz = _split_axis(N, pz)

    # Gather face labels and build nodes/edges lazily as edges are discovered
    ranks, cart_dims, periodic = index_parts(parts_dir)
    faces = {}
    node_ids = {}
    parent = []
    deg = []
    def add_node(r:int, l:int) -> int:
        key = (r, int(l))
        if key in node_ids:
            return node_ids[key]
        idx = len(parent)
        node_ids[key] = idx
        parent.append(idx)
        deg.append(0)
        return idx

    # Nodes and edges are built lazily; if no edges are found later, we will report accordingly.

    def fnd(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def uni(a,b):
        ra, rb = fnd(a), fnd(b)
        if ra != rb:
            parent[rb] = ra

    # Count edges per axis
    ex = ey = ez = 0
    # Preload faces for all ranks so neighbor lookups are always available
    for rr, info2 in ranks.items():
        with np.load(info2["npz"]) as d:
            faces[rr] = {
                "x-": d["face_xneg"], "x+": d["face_xpos"],
                "y-": d["face_yneg"], "y+": d["face_ypos"],
                "z-": d["face_zneg"], "z+": d["face_zpos"],
            }

    # Build edges by scanning interfaces; use face labels and baseline mask to gate
    by_coords = {tuple(v["coords"]): r for r, v in ranks.items()}
    for r, info in ranks.items():
        cx, cy, cz = info["coords"]
        (i0, i1), (j0, j1), (k0, k1) = (ix[cx], iy[cy], iz[cz])
        # X interface
        ncoords = ((cx+1)%px, cy, cz)
        rn = by_coords[ncoords]
        a_mask = (base_labels[i1-1, j0:j1, k0:k1] == b_label)
        b_mask = (base_labels[ix[ncoords[0]][0], j0:j1, k0:k1] == b_label)
        m = a_mask & b_mask
        if m.any():
            A = faces[r]["x+"][m]; B = faces[rn]["x-"][m]
            for la, lb in zip(A, B):
                if la>0 and lb>0:
                    ia = add_node(r, la); ib = add_node(rn, lb)
                    uni(ia, ib); ex += 1
                    deg[ia] += 1; deg[ib] += 1
        # Y interface
        ncoords = (cx, (cy+1)%py, cz)
        rn = by_coords[ncoords]
        a_mask = (base_labels[i0:i1, j1-1, k0:k1] == b_label)
        b_mask = (base_labels[i0:i1, iy[ncoords[1]][0], k0:k1] == b_label)
        m = a_mask & b_mask
        if m.any():
            A = faces[r]["y+"][m]; B = faces[rn]["y-"][m]
            for la, lb in zip(A, B):
                if la>0 and lb>0:
                    ia = add_node(r, la); ib = add_node(rn, lb)
                    uni(ia, ib); ey += 1
                    deg[ia] += 1; deg[ib] += 1
        # Z interface
        ncoords = (cx, cy, (cz+1)%pz)
        rn = by_coords[ncoords]
        a_mask = (base_labels[i0:i1, j0:j1, k1-1] == b_label)
        b_mask = (base_labels[i0:i1, j0:j1, iz[ncoords[2]][0]] == b_label)
        m = a_mask & b_mask
        if m.any():
            A = faces[r]["z+"][m]; B = faces[rn]["z-"][m]
            for la, lb in zip(A, B):
                if la>0 and lb>0:
                    ia = add_node(r, la); ib = add_node(rn, lb)
                    uni(ia, ib); ez += 1
                    deg[ia] += 1; deg[ib] += 1

    node_count = len(parent)
    if node_count == 0:
        print("    [subgraph] No baseline-gated face edges for this component.")
        return
    comps = len({fnd(i) for i in range(node_count)})
    iso = [idx for idx,d in enumerate(deg) if d==0]
    print(f"    [subgraph] Nodes={node_count} Edges(x,y,z)=({ex},{ey},{ez}) -> connected components: {comps}; isolated nodes: {len(iso)}")
    if iso:
        rev = {v:k for k,v in node_ids.items()}
        for idx in iso[:5]:
            r,l = rev[idx]
            t = []
            if (faces[r]["x-"] == l).any(): t.append('x-')
            if (faces[r]["x+"] == l).any(): t.append('x+')
            if (faces[r]["y-"] == l).any(): t.append('y-')
            if (faces[r]["y+"] == l).any(): t.append('y+')
            if (faces[r]["z-"] == l).any(): t.append('z-')
            if (faces[r]["z+"] == l).any(): t.append('z+')
            print(f"      isolated node rank={r} local={l} faces={','.join(t) if t else 'none'}")



def run_equivalence(N=96, px=2, py=2, pz=1,
                    T_thr: float = 0.1, R_thr: float = 10.0,
                    field_type: str = "simple", beta: float = -2.0,
                    debug_interfaces: bool = False):
    dens, temp = _make_fields(N, field_type=field_type, beta=beta,
                              T_limits=(0.01, 1.0), R_limits=(1.0, 100.0))
    base_T = _baseline_single_volume(dens, temp, thr=T_thr, by="temperature", return_masses=False)
    base_R = _baseline_single_volume(dens, temp, thr=R_thr, by="density", return_masses=False)

    tmpdir = tempfile.mkdtemp(prefix="tiles_")
    try:
        _write_parts(tmpdir, dens, temp, px, py, pz,
                     thr=T_thr, by="temperature", use_halo=True, overlap=1)
        outT = os.path.join(tmpdir, "stitched_T.npz")
        stitch_reduce(tmpdir, outT)
        ok_T = _compare(base_T, outT)
        if debug_interfaces:
            print("[DEBUG] Temperature-cut interface diagnostics:")
            _diagnose_interfaces(tmpdir, dens, temp, px, py, pz, by="temperature", thr=T_thr)
            _diagnose_component_mismatch(tmpdir, dens, temp, px, py, pz, by="temperature", thr=T_thr)
            # Face-only baseline comparator for T-cut
            face_only_K = _face_only_component_count((temp < T_thr), px, py, pz, use_halo=True)
            with np.load(outT) as d:
                stitched_K = int(d["cell_count"].shape[0])
            print(f"  [face-only] components={face_only_K} vs stitched_K={stitched_K}")

        # clean parts and redo for density cut
        for f in os.listdir(tmpdir):
            if f.startswith("clumps_rank"):
                os.remove(os.path.join(tmpdir, f))
        _write_parts(tmpdir, dens, temp, px, py, pz,
                     thr=R_thr, by="density", use_halo=True, overlap=1)
        outR = os.path.join(tmpdir, "stitched_R.npz")
        stitch_reduce(tmpdir, outR)
        ok_R = _compare(base_R, outR)
        if debug_interfaces:
            print("[DEBUG] Density-cut interface diagnostics:")
            _diagnose_interfaces(tmpdir, dens, temp, px, py, pz, by="density", thr=R_thr)
            _diagnose_component_mismatch(tmpdir, dens, temp, px, py, pz, by="density", thr=R_thr)
            face_only_K = _face_only_component_count((dens < R_thr), px, py, pz, use_halo=True)
            with np.load(outR) as d:
                stitched_K = int(d["cell_count"].shape[0])
            print(f"  [face-only] components={face_only_K} vs stitched_K={stitched_K}")
        if ok_T and ok_R:
            print("Equivalence test passed (single-volume == tiled+stitched) for T and density cuts.")
        else:
            print("Equivalence test did not fully match; see messages above.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--py", type=int, default=2)
    ap.add_argument("--pz", type=int, default=1)
    ap.add_argument("--T-thr", dest="T_thr", type=float, default=0.1, help="temperature cut (lt)")
    ap.add_argument("--rho-thr", dest="R_thr", type=float, default=10.0, help="density cut (lt)")
    ap.add_argument("--thr", type=float, default=None, help="deprecated: sets both T and rho thresholds")
    ap.add_argument("--field-type", choices=["simple", "powerlaw"], default="simple")
    ap.add_argument("--beta", type=float, default=-2.0, help="power spectrum slope (powerlaw)")
    ap.add_argument("--debug-interfaces", action="store_true")
    ap.add_argument("--overlap", type=int, default=1, help="kept for compatibility; must be 1")
    args = ap.parse_args()

    # Backward-compat: --thr overrides if provided
    T_thr = args.T_thr if args.thr is None else args.thr
    R_thr = args.R_thr if args.thr is None else args.thr
    if args.overlap != 1:
        raise ValueError("--overlap must be 1; overlap-exact stitching expects a 1-cell halo")
    run_equivalence(N=args.N, px=args.px, py=args.py, pz=args.pz, T_thr=T_thr, R_thr=R_thr,
                    field_type=args.field_type, beta=args.beta,
                    debug_interfaces=args.debug_interfaces)
