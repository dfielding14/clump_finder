from __future__ import annotations

import numpy as np

from local_label import label_3d
import metrics as M


def make_box_mask(shape=(32, 32, 32), box=((8, 24), (10, 22), (6, 28))):
    mask = np.zeros(shape, dtype=bool)
    (i0, i1), (j0, j1), (k0, k1) = box
    mask[i0:i1, j0:j1, k0:k1] = True
    return mask


def main():
    shape = (64, 64, 64)
    box = ((8, 40), (12, 44), (6, 30))
    mask = make_box_mask(shape, box)

    labels = label_3d(mask, tile_shape=(64, 64, 64), connectivity=6, halo=0)
    K = int(labels.max())
    assert K == 1, f"expected 1 clump, got {K}"

    Nres = 64
    dx = dy = dz = 1.0 / Nres
    origin = (0.0, 0.0, 0.0)
    dens = np.ones(shape, dtype=np.float32)
    node_bbox = ((0, shape[0]), (0, shape[1]), (0, shape[2]))

    # Area check: axis-aligned box area = 2*(ab + bc + ca)
    a = (box[0][1] - box[0][0]) * dx
    b = (box[1][1] - box[1][0]) * dy
    c = (box[2][1] - box[2][0]) * dz
    A_true = 2.0 * (a * b + b * c + c * a)
    A = M.exposed_area(labels, dx, dy, dz, K=K)[0]
    print(f"Area numerical={A:.8f} analytical={A_true:.8f}")

    cvol, cmass = M.centroids(labels, dens, dx, dy, dz, origin, node_bbox, K=K)
    # Center of the box analytically
    cx = (box[0][0] + box[0][1]) * 0.5 * dx
    cy = (box[1][0] + box[1][1]) * 0.5 * dy
    cz = (box[2][0] + box[2][1]) * 0.5 * dz
    print(f"Centroid vol={cvol[0]} analytical={[cx, cy, cz]}")


if __name__ == "__main__":
    main()

