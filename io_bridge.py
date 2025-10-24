"""
io_bridge.py

Thin I/O wrapper for reading subvolumes from AthenaPK openPMD outputs.

- Axis order in the dataset is [z, y, x] (openPMD convention).
- Returned arrays are shaped [ni, nj, nk] with i->x (fastest), j->y, k->z.
- Values are in code units (no external unit conversion).
- Computes temperature and velocities from conservative variables.
- Supports optional 1-cell periodic halos on each face (wrap-around).

Primary API
-----------

    from io_bridge import IOConfig, load_subvolume

    cfg = IOConfig(
        dataset_path="./parthenon.opmd.00010.bp/",  # or a pattern accepted by openPMD
        step=10,                                      # iteration index
        gamma=5.0/3.0,
        field_dtype=np.float32,
        emit_halo=True
    )

    # node_bbox uses global indices with exclusive upper bounds:
    # ((i0,i1), (j0,j1), (k0,k1)) where i->x, j->y, k->z
    out = load_subvolume(node_bbox=((0, 128), (0, 128), (0, 128)), cfg=cfg)
    dens, temp = out["dens"], out["temp"]

Notes
-----
- We auto-detect the mesh level suffix by inspecting available record names.
- Grids are assumed cubic, but code infers full shape from the dataset.
- Halos wrap around periodic boundaries; no clamping or zero-filling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np

try:
    import openpmd_api as opmd
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "io_bridge.py requires openpmd_api. Install and load site modules appropriately."
    ) from exc


@dataclass
class IOConfig:
    """Configuration for load_subvolume.

    - dataset_path: path or pattern accepted by openPMD Series (e.g., ./parthenon.opmd.00010.bp/)
    - step: iteration index to read from the series
    - gamma: ratio of specific heats, used for temperature computation
    - field_dtype: dtype for returned arrays (float32 recommended to limit memory)
    - ghost_width: integer count of ghost zones to include on each face (0 or 1). Ghosts are
      filled from neighboring domain cells with periodic wrap at the global domain boundary.
    - level_suffix: optional explicit suffix for record names (e.g., "lvl5"). If None, auto-detect.
    """

    dataset_path: str
    step: int
    gamma: float = 5.0 / 3.0
    field_dtype: np.dtype = np.float32
    ghost_width: int = 1
    level_suffix: Optional[str] = None


BBox3D = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _detect_level_suffix(it: "opmd.Iteration") -> str:
    """Infer level suffix by inspecting available mesh record names.

    We look for a density record that starts with 'cons_cons_density_' and
    take the trailing part as the level suffix (e.g., 'lvl5').
    """
    for name in it.meshes:
        if name.startswith("cons_cons_density_"):
            return name.split("cons_cons_density_")[-1]
    # Fallback: try to find a substring with '_lvl'
    for name in it.meshes:
        if "_lvl" in name:
            # Take text after '_lvl' up to next underscore (if any), rebuild as 'lvlX'
            lvl = name.split("_lvl")[-1].split("_")[0]
            return f"lvl{lvl}" if not lvl.startswith("lvl") else lvl
    raise ValueError("Could not auto-detect level suffix from meshes.")


def _series_and_iteration(cfg: IOConfig):
    series = opmd.Series(cfg.dataset_path, opmd.Access.read_only)
    it = series.iterations[cfg.step]
    return series, it


def _domain_shape(it: "opmd.Iteration", level_suffix: str) -> Tuple[int, int, int]:
    """Return (Nz, Ny, Nx) from the density record component shape."""
    comp = it.meshes[f"cons_cons_density_{level_suffix}"][opmd.Record_Component.SCALAR]
    return tuple(int(x) for x in comp.shape)


def _slices_pbc(n: int, i0: int, i1: int, halo: int) -> List[Tuple[int, int, int]]:
    """Periodic slice decomposition for a 1D interval with optional halo.

    Given a base interval [i0, i1) and halo cells on both sides, return up to two
    segments in domain coordinates: [(s0, s1, dest0), ...] where each segment
    corresponds to a domain slice [s0:s1] and 'dest0' is the starting index along
    the destination axis where this segment should be placed.
    """
    if n <= 0:
        raise ValueError("Axis length must be positive")
    base_lo = i0 - halo
    base_len = (i1 - i0) + 2 * halo

    # Normalize to [0, n)
    start = base_lo % n
    if base_len <= 0:
        return []

    first_len = min(n - start, base_len)
    segs = [(start, start + first_len, 0)]
    remaining = base_len - first_len
    if remaining > 0:
        segs.append((0, remaining, first_len))
    return segs


def _read_into(
    series: "opmd.Series",
    it: "opmd.Iteration",
    record_name: str,
    zsegs: List[Tuple[int, int, int]],
    ysegs: List[Tuple[int, int, int]],
    xsegs: List[Tuple[int, int, int]],
    dest: np.ndarray,
    field_dtype: np.dtype,
) -> None:
    """Read record slices into the destination array with shape [ni, nj, nk].

    - record_name: e.g., 'cons_cons_density_lvl5'
    - zsegs/ysegs/xsegs: lists of (start, end, dest_start) per axis in domain coords
    - dest: preallocated array of shape (ni, nj, nk) with dtype=field_dtype

    Note: openPMD exposes arrays in [z, y, x]. We transpose slices to [x, y, z]
    to match our [i, j, k] ordering before placing into 'dest'.
    """
    comp = it.meshes[record_name][opmd.Record_Component.SCALAR]

    for (z0, z1, dz0) in zsegs:
        for (y0, y1, dy0) in ysegs:
            for (x0, x1, dx0) in xsegs:
                arr = comp[z0:z1, y0:y1, x0:x1]
                series.flush()  # populate 'arr'
                # arr shape: (lenZ, lenY, lenX) -> transpose to (lenX, lenY, lenZ)
                a = np.asarray(arr).transpose(2, 1, 0)

                xi, xj, xk = a.shape  # (lenX, lenY, lenZ)
                di0, dj0, dk0 = dx0, dy0, dz0
                di1, dj1, dk1 = di0 + xi, dj0 + xj, dk0 + xk
                dest[di0:di1, dj0:dj1, dk0:dk1] = a.astype(field_dtype, copy=False)


def load_subvolume(
    node_bbox: BBox3D,
    cfg: IOConfig,
) -> Dict[str, np.ndarray]:
    """Load a 3-D subvolume with optional 1-cell periodic halos.

    Parameters
    ----------
    node_bbox: ((i0, i1), (j0, j1), (k0, k1))
        Global integer index bounds (exclusive upper). i->x, j->y, k->z.

    cfg: IOConfig
        Contains dataset path, step, gamma, dtype, halo toggle, and optional level suffix.

    Returns
    -------
    dict
        {
          'dens':  (ni', nj', nk') float32/float64,
          'temp':  (ni', nj', nk'),
          'velx':  (ni', nj', nk'),
          'vely':  (ni', nj', nk'),
          'velz':  (ni', nj', nk'),
          # Convenience aliases for potential downstream compatibility:
          'rho':   dens,
          'T':     temp,
          'vx':    velx,
          'vy':    vely,
          'vz':    velz,
        }
        ni'/nj'/nk' include ghost cells if cfg.ghost_width > 0.
    """
    (ib0, ib1), (jb0, jb1), (kb0, kb1) = node_bbox

    series, it = _series_and_iteration(cfg)
    level_suffix = cfg.level_suffix or _detect_level_suffix(it)
    Nz, Ny, Nx = _domain_shape(it, level_suffix)

    halo = int(max(0, cfg.ghost_width))
    ni, nj, nk = (ib1 - ib0), (jb1 - jb0), (kb1 - kb0)
    ni_t, nj_t, nk_t = ni + 2 * halo, nj + 2 * halo, nk + 2 * halo

    # Build periodic slice segments for each axis in domain [0..N)
    xsegs = _slices_pbc(Nx, ib0, ib1, halo)
    ysegs = _slices_pbc(Ny, jb0, jb1, halo)
    zsegs = _slices_pbc(Nz, kb0, kb1, halo)

    fdtype = np.dtype(cfg.field_dtype)

    # Allocate destination arrays (i, j, k) ~ (x, y, z)
    dens = np.empty((ni_t, nj_t, nk_t), dtype=fdtype)
    Etot = np.empty((ni_t, nj_t, nk_t), dtype=fdtype)
    m1   = np.empty((ni_t, nj_t, nk_t), dtype=fdtype)
    m2   = np.empty((ni_t, nj_t, nk_t), dtype=fdtype)
    m3   = np.empty((ni_t, nj_t, nk_t), dtype=fdtype)

    # Record names
    dens_name = f"cons_cons_density_{level_suffix}"
    etot_name = f"cons_cons_total_energy_density_{level_suffix}"
    m1_name   = f"cons_cons_momentum_density_1_{level_suffix}"
    m2_name   = f"cons_cons_momentum_density_2_{level_suffix}"
    m3_name   = f"cons_cons_momentum_density_3_{level_suffix}"

    # Read all fields into buffers
    _read_into(series, it, dens_name, zsegs, ysegs, xsegs, dens, fdtype)
    _read_into(series, it, etot_name, zsegs, ysegs, xsegs, Etot, fdtype)
    _read_into(series, it, m1_name,   zsegs, ysegs, xsegs, m1,   fdtype)
    _read_into(series, it, m2_name,   zsegs, ysegs, xsegs, m2,   fdtype)
    _read_into(series, it, m3_name,   zsegs, ysegs, xsegs, m3,   fdtype)

    # Compute velocities and temperature (code units)
    # Safe divide by density
    eps = np.array(1e-30, dtype=fdtype)
    dens_safe = np.maximum(dens, eps)
    velx = m1 / dens_safe
    vely = m2 / dens_safe
    velz = m3 / dens_safe

    KE = 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / dens_safe
    pres = (cfg.gamma - 1.0) * (Etot - KE)
    # Avoid negative or zero temperature due to numerical roundoff
    pres = np.maximum(pres, eps)
    temp = pres / dens_safe

    # Clean up temporary arrays early (helpful for big subvolumes)
    del Etot, m1, m2, m3
    series.flush()  # ensure all outstanding reads are finalized before closing

    # Expose preferred names and aliases
    out = {
        "dens": dens,
        "temp": temp,
        "velx": velx,
        "vely": vely,
        "velz": velz,
        # Aliases for potential downstream compatibility
        "rho": dens,
        "T": temp,
        "vx": velx,
        "vy": vely,
        "vz": velz,
    }
    return out


__all__ = ["IOConfig", "load_subvolume"]


def query_domain_shape(dataset_path: str, step: int, level_suffix: Optional[str] = None) -> tuple[tuple[int, int, int], str]:
    """Return ((Nz,Ny,Nx), level_suffix) for a dataset path and step.

    Opens the series read-only, detects level suffix if not provided.
    """
    series = opmd.Series(dataset_path, opmd.Access.read_only)
    it = series.iterations[step]
    lvl = level_suffix or _detect_level_suffix(it)
    shape = _domain_shape(it, lvl)
    series.flush()
    return (shape, lvl)
