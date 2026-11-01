"""Open shell for a fitted garment, taken from the mannequin rather than Hunyuan.

A closed reconstruction copies torn hems and cannot be simulated later. For a
fitted piece the body is the shape: offset the outer skin, cut the neck,
armholes, and hem, and ease vertices toward the Hunyuan guide only where that
guide is already outside the clearance. Those cuts stay open. Face winding
follows the outward SDF gradient. The mannequin mesh is not repaired.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pipeline.sdf_collision import (
    DEFAULT_CLEARANCE,
    SdfField,
    _face_normals,
    _unique_edges,
    body_field,
    orient_outward,
)

OFFSET_M = 0.008
GUIDE_RADIUS = 0.05
EDIT_STEPS = 4
ARM_GAP = 0.025
WELD_M = 0.002
PINHOLE_VERTS = 12


def fitted_shell(
    mannequin: trimesh.Trimesh,
    guide: trimesh.Trimesh,
    y_min: float,
    y_max: float,
    *,
    offset_m: float = OFFSET_M,
    clearance_m: float = DEFAULT_CLEARANCE,
    source: Path | None = None,
    cache: Path | None = None,
) -> trimesh.Trimesh:
    """Return a non-watertight shell. `guide` is the aligned Hunyuan garment."""
    field = body_field(mannequin, source=source, cache=cache)
    keep = _outer_band(mannequin, field, y_min, y_max)
    if not keep.any():
        raise RuntimeError(
            f"fitted shell kept no faces in y=[{y_min:.3f}, {y_max:.3f}]"
        )
    shell = mannequin.copy()
    shell.update_faces(keep)
    shell.remove_unreferenced_vertices()
    shell = _weld_cracks(shell, WELD_M)
    pinholes = _cap_pinholes(shell, PINHOLE_VERTS)
    if pinholes:
        print(f"  capped {pinholes} pinholes shorter than {PINHOLE_VERTS} verts", flush=True)
    _offset_outward(shell, field, offset_m)
    _edit_toward_guide(shell, guide, field, clearance_m)
    flipped = orient_outward(shell, field)
    boundary = _boundary_edge_count(shell.faces)
    print(
        f"  fitted shell {len(shell.faces)} faces, {boundary} boundary edges, "
        f"flipped {flipped} faces, y={y_min:.2f}–{y_max:.2f}",
        flush=True,
    )
    if boundary == 0:
        raise RuntimeError("fitted shell is closed; neck, armholes, and hem must stay open")
    return shell


def _outer_band(mannequin, field: SdfField, y_min: float, y_max: float) -> np.ndarray:
    centers = np.asarray(mannequin.triangles_center)
    sdf, _ = field.sample(centers)
    band = (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max)
    # The skins sit inside one grid cell and the mesh is cracked into patches,
    # so a single SDF threshold mixes inner and outer faces. Patches of the
    # outer skin share a higher (less negative) mean. The gap to the inner
    # skin is the split.
    keep = _outer_skin_faces(mannequin, sdf, band)
    arm = _arm_faces(centers, keep)
    if arm.any() and arm.sum() < 0.7 * int(keep.sum()):
        keep = keep & ~arm
        print(f"  cut {int(arm.sum())} arm faces", flush=True)
    elif arm.sum() >= 0.7 * max(int(keep.sum()), 1):
        print("  arm cut skipped (it would remove most of the shell)", flush=True)
    return keep


def _outer_skin_faces(mesh: trimesh.Trimesh, sdf: np.ndarray, band: np.ndarray) -> np.ndarray:
    idx = np.nonzero(band)[0]
    keep = np.zeros(len(mesh.faces), dtype=bool)
    if len(idx) == 0:
        return keep
    adj = mesh.face_adjacency
    a, b = adj[:, 0], adj[:, 1]
    both = band[a] & band[b]
    remap = np.full(len(mesh.faces), -1, np.int32)
    remap[idx] = np.arange(len(idx))
    aa, bb = remap[a[both]], remap[b[both]]
    n = len(idx)
    if len(aa) == 0:
        labels = np.arange(n, dtype=np.int32)
        ncomp = n
    else:
        rows = np.concatenate([aa, bb])
        cols = np.concatenate([bb, aa])
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        ncomp, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=ncomp)
    sums = np.zeros(ncomp, dtype=np.float64)
    np.add.at(sums, labels, sdf[idx])
    means = sums / np.maximum(counts, 1)
    large = counts >= 200
    if not large.any():
        large = np.ones(ncomp, dtype=bool)
    ordered = np.sort(means[large])[::-1]
    if len(ordered) < 2:
        chosen = np.ones(ncomp, dtype=bool)
    else:
        gaps = ordered[:-1] - ordered[1:]
        k = int(np.argmax(gaps))
        if float(gaps[k]) < 0.0015:
            chosen = np.ones(ncomp, dtype=bool)
        else:
            threshold = 0.5 * (float(ordered[k]) + float(ordered[k + 1]))
            chosen = means >= threshold
            print(
                f"  outer skin sdf >= {threshold * 1000:.1f} mm "
                f"({int(chosen.sum())} patches)",
                flush=True,
            )
    keep[idx[chosen[labels]]] = True
    return keep


def _weld_cracks(mesh: trimesh.Trimesh, radius: float) -> trimesh.Trimesh:
    """Join vertices across authoring cracks. The inner skin is already gone."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if len(verts) == 0:
        return mesh
    parent = np.arange(len(verts))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = cKDTree(verts).query_pairs(radius, output_type="ndarray")
    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(len(verts))])
    uniq, mapped = np.unique(roots, return_inverse=True)
    acc = np.zeros((len(uniq), 3), dtype=np.float64)
    weight = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(acc, mapped, verts)
    np.add.at(weight, mapped, 1.0)
    welded = acc / weight[:, None]
    faces = mapped[faces]
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[good]
    # Drop duplicate triangles created when a crack collapses.
    order = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]
    faces = faces[np.sort(order)]
    print(f"  welded cracks within {radius * 1000:.0f} mm ({len(verts)} → {len(welded)} verts)", flush=True)
    return trimesh.Trimesh(welded, faces, process=False)


def _cap_pinholes(mesh: trimesh.Trimesh, max_loop: int) -> int:
    """Close cracks of a few vertices. Neck, armhole, and hem loops are longer and stay open."""
    verts = np.array(mesh.vertices, dtype=np.float64, copy=True)
    faces = [np.asarray(mesh.faces)]
    loops = _boundary_loops(np.asarray(mesh.faces))
    capped = 0
    extra = []
    for loop in loops:
        if len(loop) < 3 or len(loop) > max_loop:
            continue
        center = verts[loop].mean(axis=0)
        ci = len(verts)
        verts = np.vstack([verts, center])
        for i, v0 in enumerate(loop):
            extra.append([v0, loop[(i + 1) % len(loop)], ci])
        capped += 1
    if capped == 0:
        return 0
    mesh.vertices = verts
    mesh.faces = np.vstack([faces[0], np.asarray(extra, dtype=np.int64)])
    return capped


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen: set[int] = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        # Walk a cycle. A chain (an open crack with endpoints) is still one component.
        loop = [start]
        seen.add(start)
        prev = None
        cur = start
        while True:
            nxts = [v for v in adj[cur] if v != prev]
            if not nxts or nxts[0] in seen:
                break
            prev, cur = cur, nxts[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3 and loop[0] in adj[loop[-1]]:
            loops.append(loop)
    return loops


def _arm_faces(points: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Faces past the torso/arm gap in |x|. A gap near the midline is the crotch, not an arm."""
    arm = np.zeros(len(points), dtype=bool)
    if not active.any():
        return arm
    y0 = float(points[active, 1].min())
    y1 = float(points[active, 1].max())
    for y in np.arange(y0, y1, 0.01):
        band = active & (np.abs(points[:, 1] - y) < 0.006)
        if int(band.sum()) < 12:
            continue
        ax = np.sort(np.abs(points[band, 0]))
        gaps = np.diff(ax)
        if len(gaps) == 0 or float(gaps.max()) <= ARM_GAP:
            continue
        cut = float(ax[int(np.argmax(gaps))])
        if cut < 0.08:
            continue
        outer = band & (np.abs(points[:, 0]) > cut + 0.002)
        # A real arm sits well past the gap. A 1 cm hole in the face samples does not.
        outer_x = np.abs(points[outer, 0])
        if int(outer.sum()) < 3 or float(np.median(outer_x) - cut) < 0.02:
            continue
        arm[outer] = True
    return arm


def _offset_outward(shell: trimesh.Trimesh, field: SdfField, offset_m: float) -> None:
    vertices = np.array(shell.vertices, dtype=np.float64, copy=True)
    _, grad = field.sample(vertices)
    weak = np.linalg.norm(grad, axis=1) < 0.5
    if weak.any():
        mid = shell.bounds.mean(axis=0)
        radial = vertices - mid
        radial[:, 1] = 0.0
        radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8
        grad = grad.copy()
        grad[weak] = radial[weak]
    shell.vertices = vertices + offset_m * grad


def _edit_toward_guide(
    shell: trimesh.Trimesh,
    guide: trimesh.Trimesh,
    field: SdfField,
    clearance_m: float,
) -> None:
    """Laplacian steps toward guide vertices that already clear the body.

    Guide vertices inside the clearance are not targets, so a boundary loop is
    not pulled onto cloth that still intersects the body.
    """
    guide_sdf, _ = field.sample(guide.vertices)
    outside = guide_sdf > clearance_m
    if not outside.any():
        print("  guide has no vertices outside the clearance; shell stays on the body", flush=True)
        return
    tree = cKDTree(np.asarray(guide.vertices)[outside])
    vertices = np.array(shell.vertices, dtype=np.float64, copy=True)
    dist, idx = tree.query(vertices, distance_upper_bound=GUIDE_RADIUS)
    has = np.isfinite(dist) & (dist <= GUIDE_RADIUS)
    if not has.any():
        print("  no shell vertex within 5 cm of an exterior guide vertex", flush=True)
        return
    targets = np.asarray(guide.vertices)[outside][idx[has]]
    edges = _unique_edges(np.asarray(shell.faces))
    boundary = _boundary_vertices(shell.faces, len(vertices))
    for _ in range(EDIT_STEPS):
        attracted = vertices.copy()
        attracted[has] = vertices[has] + 0.45 * (targets - vertices[has])
        mean = _neighbor_mean(attracted, edges)
        nxt = attracted
        interior = ~boundary
        nxt[interior] = 0.5 * attracted[interior] + 0.5 * mean[interior]
        nxt[boundary & ~has] = vertices[boundary & ~has]
        vertices = nxt
    shell.vertices = vertices
    print(f"  laplacian edit moved {int(has.sum())}/{len(vertices)} vertices toward the guide", flush=True)


def _neighbor_mean(vertices, edges) -> np.ndarray:
    acc = np.zeros_like(vertices)
    deg = np.zeros(len(vertices), dtype=np.float64)
    if len(edges) == 0:
        return vertices.copy()
    np.add.at(acc, edges[:, 0], vertices[edges[:, 1]])
    np.add.at(acc, edges[:, 1], vertices[edges[:, 0]])
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    mean = vertices.copy()
    ok = deg > 0
    mean[ok] = acc[ok] / deg[ok, None]
    return mean


def _boundary_vertices(faces, n_verts: int) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    mask = np.zeros(n_verts, dtype=bool)
    boundary = uniq[counts == 1]
    if len(boundary):
        mask[boundary.reshape(-1)] = True
    return mask


def _boundary_edge_count(faces) -> int:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(np.asarray(edges), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def _self_check() -> None:
    """A Y-cropped sphere is an open band and its faces point outward."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=0.25)
    guide = trimesh.creation.icosphere(subdivisions=2, radius=0.30)
    shell = fitted_shell(body, guide, y_min=-0.05, y_max=0.12)
    if _boundary_edge_count(shell.faces) == 0:
        raise SystemExit("shell self-check: expected open boundary loops")
    field = body_field(body)
    normals = _face_normals(np.asarray(shell.vertices), np.asarray(shell.faces))
    _, grad = field.sample(shell.triangles_center)
    dots = np.einsum("ij,ij->i", normals, grad)
    if float(np.median(dots)) <= 0:
        raise SystemExit(f"shell self-check: normals face inward (median dot {np.median(dots):.3f})")
    print("shell self-check ok", flush=True)


if __name__ == "__main__":
    _self_check()
