"""Procedural spaghetti straps. Voxel reconstructors drop 1–2 mm tubes.

Anchors: top front corners of the garment (bandeau) to the mannequin shoulders.
Output is a pair of thin quad ribbons, concatenated into one mesh.
"""
from __future__ import annotations

import numpy as np
import trimesh


def generate_procedural_straps(
    garment: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    strap_width_m: float = 0.006,
    num_samples: int = 20,
    arch_m: float = 0.012,
) -> trimesh.Trimesh:
    """Two ribbons from the bandeau corners to the mannequin shoulders."""
    gv = np.asarray(garment.vertices, dtype=np.float64)
    mv = np.asarray(mannequin.vertices, dtype=np.float64)
    left_bust, right_bust = _bandeau_corners(gv)
    left_sh, right_sh = _shoulder_anchors(mv)
    straps = [
        _ribbon(left_bust, left_sh, strap_width_m, num_samples, arch_m, sign=-1.0),
        _ribbon(right_bust, right_sh, strap_width_m, num_samples, arch_m, sign=1.0),
    ]
    out = trimesh.util.concatenate(straps)
    print(
        f"  procedural straps {len(out.faces)} faces, "
        f"bust x={left_bust[0]:.3f}/{right_bust[0]:.3f} "
        f"shoulder x={left_sh[0]:.3f}/{right_sh[0]:.3f}",
        flush=True,
    )
    return out


def attach_straps(garment: trimesh.Trimesh, mannequin: trimesh.Trimesh, **kwargs) -> trimesh.Trimesh:
    """Concatenate procedural straps onto the fitted garment."""
    return trimesh.util.concatenate([garment, generate_procedural_straps(garment, mannequin, **kwargs)])


def generate_over_shoulder_straps(
    garment: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    knot: np.ndarray,
    strap_width_m: float = 0.006,
    samples: int = 28,
) -> trimesh.Trimesh:
    """Two ribbons from the bandeau, over the shoulder crowns, meeting at `knot`."""
    gv = np.asarray(garment.vertices, dtype=np.float64)
    mv = np.asarray(mannequin.vertices, dtype=np.float64)
    left_bust, right_bust = _bandeau_corners(gv)
    left_sh, right_sh = _shoulder_crowns(mv)
    knot = np.asarray(knot, dtype=np.float64)
    straps = [
        _ribbon_through([left_bust, _lift_crown(left_sh, -1.0), knot], strap_width_m, samples),
        _ribbon_through([right_bust, _lift_crown(right_sh, 1.0), knot], strap_width_m, samples),
    ]
    out = trimesh.util.concatenate(straps)
    print(
        f"  over-shoulder straps {len(out.faces)} faces, "
        f"shoulder x={left_sh[0]:.3f}/{right_sh[0]:.3f} y={left_sh[1]:.3f}, "
        f"knot={np.round(knot, 3)}",
        flush=True,
    )
    return out


def _lift_crown(point: np.ndarray, sign: float) -> np.ndarray:
    """Sit the crest just outside the shoulder skin."""
    lifted = np.asarray(point, dtype=np.float64).copy()
    lifted[1] += 0.008
    lifted[0] += 0.006 * sign
    return lifted


def _shoulder_crowns(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lateral shoulder, not the neck.

    On this mannequin the neck is the highest point below the head. The
    acromion sits lower and further out, near 80–88% of body height.
    """
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
    span = max(y1 - y0, 1e-6)
    band = (verts[:, 1] >= y0 + 0.80 * span) & (verts[:, 1] <= y0 + 0.88 * span)
    pool = verts[band]
    if len(pool) < 8:
        pool = verts

    def pick(sign: float) -> np.ndarray:
        side = pool[pool[:, 0] * sign > 0.09]
        if len(side) < 5:
            side = pool[pool[:, 0] * sign > 0]
        if len(side) == 0:
            side = pool
        y_cut = float(np.percentile(side[:, 1], 60))
        top = side[side[:, 1] >= y_cut]
        return top[int(np.argmax(top[:, 0] * sign))]

    return pick(-1.0), pick(1.0)


def _ribbon_through(points: list[np.ndarray], width: float, n: int) -> trimesh.Trimesh:
    curve = _resample_polyline(points, n)
    t = np.linspace(0.0, 1.0, n)
    # Crest clears the deltoid; the ends stay on the bandeau and the knot.
    curve[:, 1] += np.sin(np.pi * t) * 0.006
    return _ribbon_curve(curve, width)


def _resample_polyline(points: list[np.ndarray], n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    if length < 1e-8:
        return np.repeat(pts[:1], n, axis=0)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, length, n)
    out = np.empty((n, 3), dtype=np.float64)
    for i, target in enumerate(targets):
        k = int(np.searchsorted(dist, target, side="right") - 1)
        k = min(max(k, 0), len(pts) - 2)
        span = max(float(dist[k + 1] - dist[k]), 1e-8)
        a = (target - dist[k]) / span
        out[i] = (1.0 - a) * pts[k] + a * pts[k + 1]
    return out


def _ribbon_curve(curve: np.ndarray, width: float) -> trimesh.Trimesh:
    n = len(curve)
    tangent = np.diff(curve, axis=0, append=curve[-1:])
    tangent[-1] = curve[-1] - curve[-2]
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(tangent, up)
    norms = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(norms > 1e-8, side / np.maximum(norms, 1e-8), np.array([1.0, 0.0, 0.0]))
    half = side * (width * 0.5)
    verts = np.empty((n * 2, 3), dtype=np.float64)
    verts[0::2] = curve - half
    verts[1::2] = curve + half
    faces = []
    for i in range(n - 1):
        v0, v1 = 2 * i, 2 * i + 1
        v2, v3 = 2 * (i + 1), 2 * (i + 1) + 1
        faces.append([v0, v1, v3])
        faces.append([v0, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)


def _bandeau_corners(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_max = float(verts[:, 1].max())
    top = verts[verts[:, 1] > y_max - 0.06]
    if len(top) < 8:
        top = verts
    z_cut = float(np.median(top[:, 2]))
    front = top[top[:, 2] >= z_cut]
    if len(front) < 4:
        front = top
    return front[np.argmin(front[:, 0])], front[np.argmax(front[:, 0])]


def _shoulder_anchors(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
    span = max(y1 - y0, 1e-6)
    # Shoulders sit ~82–88% up this mannequin.
    band = (verts[:, 1] > y0 + 0.82 * span) & (verts[:, 1] < y0 + 0.90 * span)
    pool = verts[band]
    if len(pool) < 8:
        pool = verts
    left = pool[pool[:, 0] < 0]
    right = pool[pool[:, 0] > 0]
    if len(left) == 0:
        left = pool
    if len(right) == 0:
        right = pool
    # Prefer the front of the shoulder (positive Z), then the outermost X.
    def pick(side: np.ndarray, want_max_x: bool) -> np.ndarray:
        z_cut = float(np.percentile(side[:, 2], 60))
        front = side[side[:, 2] >= z_cut]
        if len(front) < 3:
            front = side
        idx = int(np.argmax(front[:, 0]) if want_max_x else np.argmin(front[:, 0]))
        return front[idx]

    return pick(left, False), pick(right, True)


def _ribbon(
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    n: int,
    arch_m: float,
    sign: float,
) -> trimesh.Trimesh:
    t = np.linspace(0.0, 1.0, n)[:, None]
    curve = (1.0 - t) * start + t * end
    # Arch forward and slightly outward so the strap clears the deltoid.
    s = np.sin(np.pi * t.ravel())
    curve[:, 2] += s * arch_m
    curve[:, 0] += s * (0.008 * sign)
    tangent = np.diff(curve, axis=0, prepend=curve[:1])
    tangent[-1] = curve[-1] - curve[-2]
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(tangent, up)
    norms = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(norms > 1e-8, side / np.maximum(norms, 1e-8), np.array([1.0, 0.0, 0.0]))
    half = side * (width * 0.5)
    verts = np.empty((n * 2, 3), dtype=np.float64)
    verts[0::2] = curve - half
    verts[1::2] = curve + half
    faces = []
    for i in range(n - 1):
        v0, v1 = 2 * i, 2 * i + 1
        v2, v3 = 2 * (i + 1), 2 * (i + 1) + 1
        faces.append([v0, v1, v3])
        faces.append([v0, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
