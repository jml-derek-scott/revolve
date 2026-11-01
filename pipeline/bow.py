"""Satin bow seated on the back of an open dress shell.

Hunyuan's 384³ grid keeps the dress and drops the bow down to a few millimetres
of relief, so there is no separate component to cut off. The bow is the smooth
dark region of the back plate, built as a height field just outside the shell.
Sequins stay on the shell; they are not part of this mesh.
"""
from __future__ import annotations

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

EDGE_M = 0.005
LOOP_BULGE_M = 0.018
TAIL_BULGE_M = 0.008
THICK_M = 0.0025
GRID_STEP = 4
SHELL_MATCH_M = 0.035


def satin_loop_mask(rgb: np.ndarray, garment: np.ndarray) -> np.ndarray:
    """Smooth dark cloth in the upper part of the garment.

    A sequin window has a high local standard deviation. Satin does not.
    The band stops at 62% of the garment height so a dark sequin skirt is
    not absorbed into the bow.
    """
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    cloth = np.asarray(garment, dtype=bool)
    if gray.shape != cloth.shape:
        raise ValueError("rgb and garment mask differ in shape")
    mean = cv2.blur(gray, (15, 15))
    square = cv2.blur(gray * gray, (15, 15))
    std = np.sqrt(np.clip(square - mean * mean, 0, None))
    ys = np.nonzero(cloth)[0]
    if len(ys) == 0:
        return np.zeros(cloth.shape, dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max())
    band = np.zeros(cloth.shape, dtype=bool)
    band[y0 : y0 + max(int(0.62 * (y1 - y0 + 1)), 1), :] = True
    satin = cloth & band & (std < 14.0) & (gray < 70.0)
    cleaned = satin.astype(np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if count <= 1:
        return np.zeros(cloth.shape, dtype=bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def add_bow_tails(loops: np.ndarray, garment: np.ndarray) -> np.ndarray:
    """Two tapered ribbons hanging from the bottom centre of the loop mask."""
    cloth = np.asarray(garment, dtype=bool)
    mask = np.asarray(loops, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return mask.copy()
    y_top, y_bot = int(ys.min()), int(ys.max())
    height = max(y_bot - y_top, 1)
    bottom = ys > y_bot - 0.12 * height
    cx = float(np.median(xs[bottom]))
    width = max(int(xs.max() - xs.min()), 1)
    gys = np.nonzero(cloth)[0]
    if len(gys) == 0:
        return mask.copy()
    length = int(0.32 * max(int(gys.max() - gys.min()), 1))
    half = max(int(0.16 * width), 8)
    gap = max(int(0.03 * width), 2)
    y_end = min(y_bot + length, int(gys.max()) - 2)
    left = np.array(
        [
            [cx - gap, y_bot - 0.05 * height],
            [cx - gap - 1.3 * half, y_bot + 0.18 * length],
            [cx - gap - 0.55 * half, y_end],
            [cx - 0.15 * half, y_bot + 0.55 * length],
        ],
        dtype=np.int32,
    )
    right = np.array(
        [
            [cx + gap, y_bot - 0.02 * height],
            [cx + gap + 1.15 * half, y_bot + 0.15 * length],
            [cx + gap + 0.35 * half, y_end],
            [cx + 0.2 * half, y_bot + 0.45 * length],
        ],
        dtype=np.int32,
    )
    drawn = mask.astype(np.uint8)
    cv2.fillPoly(drawn, [left, right], 1)
    drawn = (drawn.astype(bool) & cloth) | mask
    close = cv2.morphologyEx(drawn.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return close.astype(bool) & (cloth | mask)


def build_bow(
    rgb: np.ndarray,
    garment: np.ndarray,
    view: dict,
    shell: trimesh.Trimesh,
    ref: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return the bow mesh and the knot point the straps meet at."""
    loops = satin_loop_mask(rgb, garment)
    full = add_bow_tails(loops, garment)
    if int(full.sum()) < 50:
        raise RuntimeError("bow mask is empty")
    mesh = _heightfield(full, loops, view, shell, ref)
    knot = _knot(loops, view, shell, ref)
    print(
        f"  bow {len(mesh.faces)} faces, knot={np.round(knot, 3)}, "
        f"mask {int(loops.sum())} loop px",
        flush=True,
    )
    return mesh, knot


def _knot(loops: np.ndarray, view: dict, shell: trimesh.Trimesh, ref: trimesh.Trimesh) -> np.ndarray:
    ys, xs = np.nonzero(loops)
    if len(ys) == 0:
        raise RuntimeError("bow loops are empty")
    # The knot is the centre of the loops, not the top edge and not the tails.
    py = float(np.median(ys))
    px = float(np.median(xs))
    point = _world_points(np.array([px]), np.array([py]), view, ref)[0]
    back_z = _back_z(np.array([point[:2]]), shell)[0]
    point[2] = back_z - LOOP_BULGE_M
    return point


def _heightfield(
    full: np.ndarray,
    loops: np.ndarray,
    view: dict,
    shell: trimesh.Trimesh,
    ref: trimesh.Trimesh,
) -> trimesh.Trimesh:
    ys, xs = np.nonzero(full)
    gy = np.arange(int(ys.min()), int(ys.max()) + 1, GRID_STEP)
    gx = np.arange(int(xs.min()), int(xs.max()) + 1, GRID_STEP)
    present = full[np.ix_(gy, gx)]
    loop_here = loops[np.ix_(gy, gx)]
    dist = cv2.distanceTransform(full.astype(np.uint8), cv2.DIST_L2, 3)
    sampled = dist[np.ix_(gy, gx)]
    peak = max(float(sampled.max()), 1.0)
    bulge = np.where(
        loop_here,
        EDGE_M + (LOOP_BULGE_M - EDGE_M) * (sampled / peak),
        EDGE_M + (TAIL_BULGE_M - EDGE_M) * (sampled / peak),
    )
    py = np.broadcast_to(gy[:, None], present.shape).ravel()
    px = np.broadcast_to(gx[None, :], present.shape).ravel()
    world = _world_points(px.astype(np.float64), py.astype(np.float64), view, ref)
    back_z, near = _back_z(world[:, :2], shell, return_distance=True)
    world[:, 2] = back_z - bulge.ravel()
    usable = present.ravel() & near
    # A cell is a quad only when every corner lands on the shell.
    ny, nx = present.shape
    on_shell = usable.reshape(ny, nx)
    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            if not on_shell[iy : iy + 2, ix : ix + 2].all():
                continue
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    if not faces:
        raise RuntimeError("bow height field produced no faces on the shell")
    mesh = trimesh.Trimesh(vertices=world, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    if float(np.mean(mesh.face_normals[:, 2])) > 0:
        mesh.invert()
    return _thicken(mesh, THICK_M)


def _world_points(px: np.ndarray, py: np.ndarray, view: dict, ref: trimesh.Trimesh) -> np.ndarray:
    """Inverse of the back-camera projection in texture_bake.project."""
    mlo, mhi = ref.bounds
    height = max(float(mhi[1] - mlo[1]), 1e-6)
    scale = (float(view["bottom"]) - float(view["top"])) / height
    if abs(scale) < 1e-8:
        raise ValueError("view has no vertical span")
    mid_x = 0.5 * float(mlo[0] + mhi[0])
    # Back view stores px = cx - (x - mid_x) * scale.
    x = mid_x - (np.asarray(px, dtype=np.float64) - float(view["cx"])) / scale
    y = float(mlo[1]) + (float(view["bottom"]) - np.asarray(py, dtype=np.float64)) / scale
    return np.column_stack([x, y, np.zeros(len(x))])


def _back_z(xy: np.ndarray, shell: trimesh.Trimesh, return_distance: bool = False):
    verts = np.asarray(shell.vertices, dtype=np.float64)
    back = verts[:, 2] <= float(np.median(verts[:, 2]))
    pool = verts[back] if int(back.sum()) >= 8 else verts
    tree = cKDTree(pool[:, :2])
    dist, index = tree.query(np.asarray(xy, dtype=np.float64), k=1)
    z = pool[np.atleast_1d(index), 2]
    if return_distance:
        return z, np.atleast_1d(dist) <= SHELL_MATCH_M
    return z


def _thicken(mesh: trimesh.Trimesh, depth: float) -> trimesh.Trimesh:
    """A thin shell so the bow reads from the side, not only from the back."""
    outer = np.asarray(mesh.vertices, dtype=np.float64)
    inner = outer.copy()
    inner[:, 2] += depth
    n = len(outer)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    inner_faces = faces[:, ::-1] + n
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    keys, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = keys[counts == 1]
    walls = []
    for a, b in boundary:
        walls.append([int(a), int(b), int(b) + n])
        walls.append([int(a), int(b) + n, int(a) + n])
    wall_faces = np.asarray(walls, dtype=np.int64) if walls else np.zeros((0, 3), dtype=np.int64)
    out = trimesh.Trimesh(
        vertices=np.vstack([outer, inner]),
        faces=np.vstack([faces, inner_faces, wall_faces]),
        process=False,
    )
    return out
