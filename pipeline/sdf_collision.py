"""Clearance against the outer skin of a double-skinned body.

`female_base_mesh.glb` is a joined shell about 5 mm thick with scrambled
winding. `mesh.contains()`, vertex normals, and a winding-number distance all
point the wrong way on it. This field rasterizes the triangles, then flood-fills
from outside the grid. Unreached voxels — the cavity and the shell itself —
are inside, so the inner skin is not a second zero crossing.

SDF is negative inside and positive outside. The gradient is the normalized
direction of increasing distance (outward).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    map_coordinates,
)

GRID_STEP = 0.005
DEFAULT_CLEARANCE = 0.004
MAX_STEP = 0.02
PUSH_ITERS = 8
LAPLACIAN_ITERS = 3
PIN_OUTSIDE = 0.008
ENERGY_TARGET = 1e-4
PAD = 0.08


class SdfField:
    """Trilinear SDF and its unit outward gradient on a regular grid."""

    def __init__(self, sdf: np.ndarray, origin: np.ndarray, step: float):
        self.sdf = np.ascontiguousarray(sdf, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.step = float(step)
        self.grad = _unit_gradient(self.sdf, self.step)

    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(sdf, unit_gradient)` for world-space points `(N, 3)`."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return np.zeros(0, np.float32), np.zeros((0, 3), np.float32)
        index = (pts - self.origin) / self.step - 0.5
        coords = np.ascontiguousarray(index.T)
        sdf = map_coordinates(self.sdf, coords, order=1, mode="nearest").astype(np.float32)
        grad = np.stack(
            [
                map_coordinates(self.grad[..., axis], coords, order=1, mode="nearest")
                for axis in range(3)
            ],
            axis=1,
        ).astype(np.float32)
        length = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad / np.maximum(length, 1e-8)
        grad[length[:, 0] < 1e-5] = 0
        return sdf, grad


def body_field(
    mesh: trimesh.Trimesh,
    *,
    source: Path | None = None,
    cache: Path | None = None,
    step: float = GRID_STEP,
) -> SdfField:
    """Occupancy SDF of a closed body. Cached as `.sdf.npz` next to `source`."""
    cache_path = Path(cache) if cache is not None else None
    if cache_path is None and source is not None:
        cache_path = Path(source).with_suffix(".sdf.npz")
    if cache_path is not None and cache_path.exists():
        loaded = _load_cache(cache_path, mesh, step, source)
        if loaded is not None:
            print(f"  sdf cache {cache_path.name}", flush=True)
            return loaded
    field, inside, how = _build_body_field(mesh, step)
    print(
        f"  sdf grid {field.sdf.shape} step {step * 1000:.0f} mm "
        f"inside {int(inside.sum())} ({how})",
        flush=True,
    )
    if cache_path is not None:
        _save_cache(cache_path, field, mesh, source)
        print(f"  wrote {cache_path}", flush=True)
    return field


def shell_field(mesh: trimesh.Trimesh, *, step: float = GRID_STEP) -> SdfField:
    """Thin band around an open garment. Negative on the cloth, positive off it.

    An open shell has no cavity to flood-fill. The band is the rasterized
    surface dilated by one voxel, so the zero level sits just off the cloth
    on both sides. Callers that layer garments flip this gradient to agree
    with the body field, which keeps the outer garment from being pushed
    into the body.
    """
    origin, shape = _grid_bounds(mesh.bounds, step)
    occupied = _rasterize_surface(mesh.vertices, mesh.faces, origin, step, shape)
    band = binary_dilation(occupied, iterations=1)
    sdf = _sdf_from_inside(band, step)
    return SdfField(sdf, origin, step)


def push_clearance(
    garment: trimesh.Trimesh,
    body: trimesh.Trimesh,
    *,
    clearance_m: float = DEFAULT_CLEARANCE,
    source: Path | None = None,
    cache: Path | None = None,
    obstacle: trimesh.Trimesh | None = None,
    obstacle_clearance_m: float | None = None,
) -> dict:
    """Push `garment` vertices out to `clearance_m`. Updates the mesh in place.

    `obstacle` is an inner garment already in the same pose. Where the two
    overlap, vertices also stay `obstacle_clearance_m` off that shell, on the
    side that points out of the body.
    """
    field = body_field(body, source=source, cache=cache)
    extra = None
    extra_clearance = clearance_m if obstacle_clearance_m is None else float(obstacle_clearance_m)
    if obstacle is not None:
        print(f"  obstacle shell ({len(obstacle.faces)} faces)", flush=True)
        extra = shell_field(obstacle)
    report = _solve(garment, field, clearance_m, extra, extra_clearance if extra is not None else None)
    orient_outward(garment, field)
    _print_report(report, clearance_m)
    return report


def orient_outward(mesh: trimesh.Trimesh, field: SdfField) -> int:
    """Flip faces whose normal disagrees with the outward SDF gradient.

    The mannequin's own winding is left alone. This only retargets the garment
    about to be exported.
    """
    if len(mesh.faces) == 0:
        return 0
    if hasattr(mesh, "_cache"):
        mesh._cache.clear()
    centers = mesh.triangles_center
    _, grad = field.sample(centers)
    normals = _face_normals(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    strength = np.linalg.norm(grad, axis=1)
    dots = np.einsum("ij,ij->i", normals, grad)
    flip = (strength > 0.3) & (dots < 0)
    if not flip.any():
        return 0
    faces = np.asarray(mesh.faces).copy()
    faces[flip] = faces[flip][:, ::-1]
    mesh.faces = faces
    return int(flip.sum())


def _solve(garment, field, clearance, extra, extra_clearance) -> dict:
    vertices = np.array(garment.vertices, dtype=np.float64, copy=True)
    faces = np.asarray(garment.faces)
    edges = _unique_edges(faces)
    moved = np.zeros(len(vertices), dtype=bool)
    for _ in range(PUSH_ITERS):
        step_moved = _push_step(vertices, field, clearance)
        if extra is not None:
            step_moved |= _push_step(
                vertices, extra, extra_clearance, align_to=field
            )
        if not step_moved.any():
            break
        moved |= step_moved
    if len(edges):
        moved |= _push_edge_midpoints(vertices, edges, field, clearance)
        if extra is not None:
            moved |= _push_edge_midpoints(
                vertices, edges, extra, extra_clearance, align_to=field
            )
    _laplacian(vertices, edges, moved, field)
    # Smoothing can tug a vertex back through the skin. Project vertices again,
    # then faces whose centers still clip even though the corners are clear.
    for _ in range(PUSH_ITERS):
        step_moved = _push_step(vertices, field, clearance)
        step_moved |= _push_face_centers(vertices, faces, field, clearance)
        if extra is not None:
            step_moved |= _push_step(vertices, extra, extra_clearance, align_to=field)
            step_moved |= _push_face_centers(
                vertices, faces, extra, extra_clearance, align_to=field
            )
        if not step_moved.any():
            break
    garment.vertices = vertices
    sdf, _ = field.sample(vertices)
    penetration = np.maximum(clearance - sdf, 0.0)
    report = {
        "energy": float(np.mean(penetration**2)),
        "inside": int(np.count_nonzero(sdf < clearance)),
        "vertices": int(len(vertices)),
        "clearance_m": float(clearance),
    }
    if extra is not None:
        extra_sdf, _ = extra.sample(vertices)
        extra_pen = np.maximum(extra_clearance - extra_sdf, 0.0)
        report["obstacle_energy"] = float(np.mean(extra_pen**2))
        report["obstacle_inside"] = int(np.count_nonzero(extra_sdf < extra_clearance))
    return report


def _push_step(vertices, field, clearance, align_to: SdfField | None = None) -> np.ndarray:
    sdf, grad = field.sample(vertices)
    if align_to is not None:
        _, body_grad = align_to.sample(vertices)
        oppose = np.einsum("ij,ij->i", grad, body_grad) < 0
        grad = grad.copy()
        grad[oppose] *= -1
    length = np.linalg.norm(grad, axis=1)
    pen = (sdf < clearance) & (length > 0.5)
    if not pen.any():
        return pen
    delta = (clearance - sdf)[:, None] * grad
    dist = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, MAX_STEP / np.maximum(dist, 1e-12))
    vertices[pen] += delta[pen] * scale[pen, None]
    return pen


def _push_face_centers(vertices, faces, field, clearance, align_to=None) -> np.ndarray:
    """Push a face whose center is inside even when its vertices already clear."""
    moved = np.zeros(len(vertices), dtype=bool)
    if len(faces) == 0:
        return moved
    centers = vertices[faces].mean(axis=1)
    sdf, grad = field.sample(centers)
    if align_to is not None:
        _, body_grad = align_to.sample(centers)
        oppose = np.einsum("ij,ij->i", grad, body_grad) < 0
        grad = grad.copy()
        grad[oppose] *= -1
    length = np.linalg.norm(grad, axis=1)
    hit = (sdf < clearance) & (length > 0.5)
    if not hit.any():
        return moved
    delta = (clearance - sdf)[:, None] * grad
    dist = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, MAX_STEP / np.maximum(dist, 1e-12))
    delta = delta * scale[:, None]
    shift = np.zeros_like(vertices)
    weight = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        ids = faces[hit, corner]
        np.add.at(shift, ids, delta[hit])
        np.add.at(weight, ids, 1.0)
    ok = weight > 0
    vertices[ok] += shift[ok] / weight[ok, None]
    moved[ok] = True
    return moved


def _push_edge_midpoints(vertices, edges, field, clearance, align_to=None) -> np.ndarray:
    """Push both ends of an edge whose midpoint is inside while the ends are not."""
    e0, e1 = edges[:, 0], edges[:, 1]
    mid = 0.5 * (vertices[e0] + vertices[e1])
    sdf_m, grad_m = field.sample(mid)
    if align_to is not None:
        _, body_grad = align_to.sample(mid)
        oppose = np.einsum("ij,ij->i", grad_m, body_grad) < 0
        grad_m = grad_m.copy()
        grad_m[oppose] *= -1
    sdf0, _ = field.sample(vertices[e0])
    sdf1, _ = field.sample(vertices[e1])
    length = np.linalg.norm(grad_m, axis=1)
    hit = (sdf_m < clearance) & (sdf0 >= clearance) & (sdf1 >= clearance) & (length > 0.5)
    moved = np.zeros(len(vertices), dtype=bool)
    if not hit.any():
        return moved
    delta = (clearance - sdf_m)[:, None] * grad_m
    dist = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, MAX_STEP / np.maximum(dist, 1e-12))
    delta = delta * scale[:, None]
    vertices[e0[hit]] += delta[hit]
    vertices[e1[hit]] += delta[hit]
    moved[e0[hit]] = True
    moved[e1[hit]] = True
    return moved


def _laplacian(vertices, edges, moved, field: SdfField) -> None:
    """Two to three uniform averages. Vertices already 8 mm out stay pinned."""
    if not moved.any() or len(edges) == 0:
        return
    for _ in range(LAPLACIAN_ITERS):
        sdf, _ = field.sample(vertices)
        free = moved & (sdf <= PIN_OUTSIDE)
        if not free.any():
            return
        neighbor = _neighbor_mean(vertices, edges)
        vertices[free] = neighbor[free]


def _neighbor_mean(vertices, edges) -> np.ndarray:
    n = len(vertices)
    acc = np.zeros_like(vertices)
    deg = np.zeros(n, dtype=np.float64)
    np.add.at(acc, edges[:, 0], vertices[edges[:, 1]])
    np.add.at(acc, edges[:, 1], vertices[edges[:, 0]])
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    mean = vertices.copy()
    ok = deg > 0
    mean[ok] = acc[ok] / deg[ok, None]
    return mean


def _build_body_field(mesh: trimesh.Trimesh, step: float):
    origin, shape = _grid_bounds(mesh.bounds, step)
    occupied = _rasterize_surface(mesh.vertices, mesh.faces, origin, step, shape)
    inside, how = _inside_mask(occupied)
    sdf = _sdf_from_inside(inside, step)
    return SdfField(sdf, origin, step), inside, how


def _inside_mask(occupied: np.ndarray) -> tuple[np.ndarray, str]:
    """Flood from outside. Unreached voxels are the cavity plus the shell."""
    filled = binary_fill_holes(occupied)
    if _is_solid(filled, occupied):
        return filled, "flood"
    sealed = binary_fill_holes(binary_closing(occupied, iterations=1))
    if _is_solid(sealed, occupied):
        return sealed, "sealed cracks"
    thick = binary_fill_holes(binary_dilation(occupied, iterations=1))
    return thick, "dilated surface"


def _is_solid(filled: np.ndarray, occupied: np.ndarray) -> bool:
    cavity = int(np.count_nonzero(filled & ~occupied))
    return cavity > max(100, int(occupied.sum()) // 5)


def _sdf_from_inside(inside: np.ndarray, step: float) -> np.ndarray:
    # distance_transform_edt measures to the nearest zero. Invert each mask
    # so outside voxels store distance-to-inside and inside voxels store
    # distance-to-outside.
    dist_out = distance_transform_edt(~inside, sampling=step)
    dist_in = distance_transform_edt(inside, sampling=step)
    sdf = np.where(inside, -dist_in, dist_out).astype(np.float32)
    return sdf


def _unit_gradient(sdf: np.ndarray, step: float) -> np.ndarray:
    parts = np.gradient(sdf.astype(np.float32), step, step, step)
    grad = np.stack(parts, axis=-1).astype(np.float32)
    length = np.linalg.norm(grad, axis=-1, keepdims=True)
    unit = grad / np.maximum(length, 1e-8)
    unit[length[..., 0] < 1e-5] = 0
    return np.ascontiguousarray(unit, dtype=np.float32)


def _grid_bounds(bounds, step: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    lo = np.asarray(bounds[0], dtype=np.float64) - PAD
    hi = np.asarray(bounds[1], dtype=np.float64) + PAD
    shape = tuple(int(np.ceil((hi[axis] - lo[axis]) / step)) + 1 for axis in range(3))
    return lo, shape


def _rasterize_surface(vertices, faces, origin, step, shape) -> np.ndarray:
    occupied = np.zeros(shape, dtype=bool)
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if len(faces) == 0:
        return occupied
    _mark_points(occupied, verts, origin, step, shape)
    edges = _unique_edges(faces)
    if len(edges):
        _mark_segments(occupied, verts, edges, origin, step, shape)
    tris = verts[faces]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    edge_len = np.linalg.norm(
        np.stack([v1 - v0, v2 - v1, v0 - v2], axis=1), axis=2
    ).max(axis=1)
    needed = np.clip(np.ceil(edge_len / (step * 0.45)).astype(int) + 1, 2, 256)
    prev = 0
    for n in (4, 8, 16, 32, 64, 128, 256):
        sel = np.nonzero((needed > prev) & (needed <= n))[0]
        prev = n
        if len(sel) == 0:
            continue
        _mark_triangles(occupied, v0[sel], v1[sel], v2[sel], origin, step, shape, n)
    return occupied


def _mark_triangles(occupied, v0, v1, v2, origin, step, shape, n: int) -> None:
    coord = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(coord, coord, indexing="ij")
    bary = np.stack([uu.ravel(), vv.ravel()], axis=1)
    bary = bary[bary.sum(axis=1) <= 1.0 + 1e-6]
    # Keep each batch under a few million samples.
    batch = max(1, 2_000_000 // max(len(bary), 1))
    for start in range(0, len(v0), batch):
        sl = slice(start, start + batch)
        pts = (
            v0[sl, None, :]
            + bary[None, :, 0, None] * (v1[sl] - v0[sl])[:, None, :]
            + bary[None, :, 1, None] * (v2[sl] - v0[sl])[:, None, :]
        )
        _mark_points(occupied, pts.reshape(-1, 3), origin, step, shape)


def _mark_segments(occupied, vertices, edges, origin, step, shape) -> None:
    a = vertices[edges[:, 0]]
    b = vertices[edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    needed = np.clip(np.ceil(length / (step * 0.45)).astype(int) + 1, 2, 256)
    prev = 0
    for n in (4, 8, 16, 32, 64, 128, 256):
        sel = np.nonzero((needed > prev) & (needed <= n))[0]
        prev = n
        if len(sel) == 0:
            continue
        t = np.linspace(0.0, 1.0, n)
        batch = max(1, 2_000_000 // n)
        for start in range(0, len(sel), batch):
            sl = sel[start : start + batch]
            pts = a[sl, None, :] * (1.0 - t)[None, :, None] + b[sl, None, :] * t[None, :, None]
            _mark_points(occupied, pts.reshape(-1, 3), origin, step, shape)


def _mark_points(occupied, pts, origin, step, shape) -> None:
    idx = np.floor((np.asarray(pts) - origin) / step).astype(np.int32)
    ok = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < shape[0])
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < shape[1])
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < shape[2])
    )
    idx = idx[ok]
    if len(idx) == 0:
        return
    occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _face_normals(vertices, faces) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)


def _fingerprint(mesh, step, source: Path | None) -> dict:
    fp = {
        "n_verts": np.int64(len(mesh.vertices)),
        "n_faces": np.int64(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64),
        "step": np.float64(step),
        "mtime_ns": np.int64(0),
        "size": np.int64(0),
    }
    if source is not None and Path(source).exists():
        stat = Path(source).stat()
        fp["mtime_ns"] = np.int64(stat.st_mtime_ns)
        fp["size"] = np.int64(stat.st_size)
    return fp


def _save_cache(path: Path, field: SdfField, mesh, source) -> None:
    fp = _fingerprint(mesh, field.step, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sdf=field.sdf,
        origin=field.origin,
        **fp,
    )


def _load_cache(path: Path, mesh, step, source) -> SdfField | None:
    try:
        data = np.load(path)
    except (OSError, ValueError):
        return None
    fp = _fingerprint(mesh, step, source)
    if int(data["n_verts"]) != int(fp["n_verts"]) or int(data["n_faces"]) != int(fp["n_faces"]):
        return None
    if abs(float(data["step"]) - step) > 1e-9:
        return None
    if not np.allclose(data["bounds"], fp["bounds"], atol=1e-4):
        return None
    if int(fp["mtime_ns"]) and int(data["mtime_ns"]) != int(fp["mtime_ns"]):
        return None
    if int(fp["size"]) and int(data["size"]) != int(fp["size"]):
        return None
    return SdfField(data["sdf"], data["origin"], float(data["step"]))


def _print_report(report: dict, clearance: float) -> None:
    flag = "ok" if report["energy"] < ENERGY_TARGET else "above target"
    print(
        f"  clearance {clearance * 1000:.1f} mm  energy {report['energy']:.3e} "
        f"({flag}, target < {ENERGY_TARGET:.0e})  "
        f"inside {report['inside']}/{report['vertices']}",
        flush=True,
    )
    if "obstacle_energy" in report:
        print(
            f"  obstacle energy {report['obstacle_energy']:.3e}  "
            f"inside {report['obstacle_inside']}/{report['vertices']}",
            flush=True,
        )


def _self_check() -> None:
    """Double-skin box: inner wall is inside, not a second zero. A penetrating mesh clears."""
    outer = trimesh.creation.box(extents=(0.40, 0.50, 0.24))
    inner = trimesh.creation.box(extents=(0.34, 0.44, 0.18))
    body = trimesh.util.concatenate([outer, inner])
    field = body_field(body, step=GRID_STEP)
    center = field.sample(np.array([[0.0, 0.0, 0.0]]))[0][0]
    gap = field.sample(np.array([[0.185, 0.0, 0.0]]))[0][0]
    outside = field.sample(np.array([[0.28, 0.0, 0.0]]))[0][0]
    if not (center < -0.01 and gap < -0.002 and outside > 0.02):
        raise SystemExit(
            f"sdf signs failed center={center:.4f} gap={gap:.4f} outside={outside:.4f}"
        )
    penetrating = trimesh.creation.box(extents=(0.30, 0.30, 0.16))
    report = push_clearance(penetrating, body, clearance_m=DEFAULT_CLEARANCE)
    if report["energy"] >= ENERGY_TARGET:
        raise SystemExit(f"penetration energy {report['energy']:.3e} not under {ENERGY_TARGET:.0e}")
    print("sdf self-check ok", flush=True)


if __name__ == "__main__":
    _self_check()
