"""
stitch.py (stub)

Design scaffolding for future cross-node stitching (disabled by default).

When enabled:
- Export boundary adjacency for each node (labels touching node faces).
- Gather edges on rank 0, build DSU over (rank,label) pairs.
- Broadcast global relabel map and either:
  (a) re-reduce locally, or (b) merge per-clump rows using sufficient statistics.

Current implementation: placeholders / TODOs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class StitchConfig:
    enabled: bool = False


def export_boundary_adjacency(labels, node_bbox, periodic):
    """Placeholder: return list of boundary touch records.

    Each record would include (rank, local_label, face_id, face_coords).
    """
    return []


def dsu_merge(edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    """Placeholder DSU over (rank,label) pairs."""
    return {}

