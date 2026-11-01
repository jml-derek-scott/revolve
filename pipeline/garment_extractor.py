"""Cut the garment from a clothed reconstruction and register it to the mannequin.

Uses orthographic depth buffers, not normals or mesh.contains(). Never calls
trimesh.split(). Weld a working copy before adjacency / connected components.
"""
from pathlib import Path

import cv2
import numpy as np
import trimesh

DEPTH_CELL = 0.004  # fraction of mesh height per depth-buffer cell
CLEARANCE = 0.004  # metres the garment must sit outside the mannequin skin
CLOSE_STEPS = 12  # face rings for hole closing
TORSO_ARM_Y = 1.32  # metres; below this, arms are stripped from the torso map
HIP_Y = 1.05  # metres; push only above here so the skirt is not flattened onto the pelvis
BACK_COVER = 0.07  # metres: only cloth already this close to the back skin is pulled out
HIP_X = 0.35  # metres from midline; drop bags / detached side lumps after placement
SAMPLE_BUILD = 1_500_000
SAMPLE_MANNEQUIN = 3_000_000


def dress_band(views: dict, mannequin: trimesh.Trimesh) -> dict:
    """Bust-to-hem span on the mannequin, from the front photo's placement.

    `hem` and `top` are the photo length. `hip_y` and `torso_top` pad that
    span by a couple of centimetres for the clearance push.
    """
    mlo, mhi = mannequin.bounds
    span = float(mhi[1] - mlo[1])
    vf = views["front"]
    p_h = max(float(vf["bottom"] - vf["top"]), 1.0)
    if "g_top" in vf and "g_bottom" in vf:
        g_top, g_bottom = float(vf["g_top"]), float(vf["g_bottom"])
    else:
        ys = np.nonzero(vf["garment"])[0]
        g_top = float(ys.min()) if len(ys) else float(vf["top"])
        g_bottom = float(ys.max()) if len(ys) else float(vf["bottom"])
    foot_to_hem = max(0.0, min(0.85, (float(vf["bottom"]) - g_bottom) / p_h))
    height_ratio = max(0.15, min(1.0, (g_bottom - g_top) / p_h))
    hem = float(mlo[1]) + span * foot_to_hem
    top = hem + span * height_ratio
    return {"hem": hem, "top": top, "hip_y": hem - 0.02, "torso_top": top + 0.04}


def bandeau_top(view: dict, mannequin: trimesh.Trimesh, fraction: float = 0.35) -> float:
    """Body height of the bandeau, ignoring thin straps above it.

    `dress_band` uses the top garment pixel, which on this dress is a spaghetti
    strap. The shell should stop where the cloth becomes wide.
    """
    rows = np.asarray(view["garment"]).sum(axis=1)
    if rows.size == 0 or float(rows.max()) <= 0:
        raise ValueError("garment mask is empty")
    wide = np.nonzero(rows >= fraction * float(rows.max()))[0]
    if len(wide) == 0:
        raise ValueError("garment mask has no wide rows")
    return _photo_row_to_y(int(wide.min()), view, mannequin)


def _photo_row_to_y(row: int, view: dict, mannequin: trimesh.Trimesh) -> float:
    mlo, mhi = mannequin.bounds
    span = float(mhi[1] - mlo[1])
    p_h = max(float(view["bottom"] - view["top"]), 1.0)
    foot = float(np.clip((float(view["bottom"]) - float(row)) / p_h, 0.0, 0.95))
    return float(mlo[1]) + span * foot


def _sdf_source(mannequin) -> Path | None:
    if isinstance(mannequin, (str, Path)):
        return Path(mannequin)
    return None


def load_mesh(source) -> trimesh.Trimesh:
    """Load a GLB path, bytes, or existing Trimesh as a single mesh."""
    if isinstance(source, trimesh.Trimesh):
        return source
    if isinstance(source, (bytes, bytearray)):
        from io import BytesIO

        loaded = trimesh.load(BytesIO(source), file_type="glb", force="mesh")
    else:
        loaded = trimesh.load(source, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
    return loaded


def extract_garment(build: trimesh.Trimesh, views: dict) -> trimesh.Trimesh:
    """Keep faces whose centers fall in the 2D garment mask of the nearer side."""
    lo, hi = build.bounds
    height = hi[1] - lo[1]
    if height <= 0:
        raise ValueError("build mesh has zero height")
    centers = build.triangles_center
    cell = DEPTH_CELL * height
    gw, gh = int((hi[0] - lo[0]) / cell) + 1, int(height / cell) + 1
    pts, _ = trimesh.sample.sample_surface(build, SAMPLE_BUILD)

    def grid(xy):
        return (
            ((xy[:, 1] - lo[1]) / cell).astype(int).clip(0, gh - 1),
            ((xy[:, 0] - lo[0]) / cell).astype(int).clip(0, gw - 1),
        )

    zmax, zmin = np.full((gh, gw), -np.inf), np.full((gh, gw), np.inf)
    r, c = grid(pts)
    np.maximum.at(zmax, (r, c), pts[:, 2])
    np.minimum.at(zmin, (r, c), pts[:, 2])

    r, c = grid(centers)
    z = centers[:, 2]
    front_side = (zmax[r, c] - z) <= (z - zmin[r, c])
    keep = np.zeros(len(centers), bool)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    for side, sign, on_side in (("front", 1, front_side), ("back", -1, ~front_side)):
        v = views[side]
        s = (v["bottom"] - v["top"]) / height
        px = (v["cx"] + sign * (centers[:, 0] - (lo[0] + hi[0]) / 2) * s).round().astype(int)
        py = (v["bottom"] - (centers[:, 1] - lo[1]) * s).round().astype(int)
        H, W = v["garment"].shape
        # Close holes in sheer / lace so the skirt isn't Swiss-cheesed by u2net.
        gmask = cv2.morphologyEx(v["garment"].astype(np.uint8), cv2.MORPH_CLOSE, close_k).astype(bool)
        m = on_side & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        keep[m] = gmask[py[m], px[m]]

    welded_build = trimesh.Trimesh(build.vertices, build.faces, process=False)
    welded_build.merge_vertices(merge_tex=True, merge_norm=True)
    adj = welded_build.face_adjacency
    for grow in (True, False):
        for _ in range(CLOSE_STEPS):
            a, b = keep[adj[:, 0]], keep[adj[:, 1]]
            nxt = keep.copy()
            if grow:
                nxt[adj[a & ~b, 1]] = True
                nxt[adj[b & ~a, 0]] = True
            else:
                nxt[adj[a & ~b, 0]] = False
                nxt[adj[b & ~a, 1]] = False
            keep = nxt

    garment = build.copy()
    garment.update_faces(keep)
    garment.remove_unreferenced_vertices()
    if len(garment.faces) == 0:
        raise RuntimeError("garment mask kept zero faces")

    welded = trimesh.Trimesh(garment.vertices, garment.faces, process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    labels = trimesh.graph.connected_component_labels(
        welded.face_adjacency, node_count=len(welded.faces)
    )
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    garment.update_faces(labels == largest)
    garment.remove_unreferenced_vertices()
    if len(garment.faces) == 0:
        raise RuntimeError("largest garment component is empty")
    return garment


def align_to_mannequin(
    mesh: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    build_bounds,
) -> trimesh.Trimesh:
    """Scale/translate a reconstruction so its full-person bbox matches the mannequin."""
    lo, hi = build_bounds
    height = hi[1] - lo[1]
    mlo, mhi = mannequin.bounds
    scale = (mhi[1] - mlo[1]) / height
    mesh = mesh.copy()
    mesh.apply_translation(-np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]))
    mesh.apply_scale(scale)
    mesh.apply_translation([(mlo[0] + mhi[0]) / 2, mlo[1], (mlo[2] + mhi[2]) / 2])
    return mesh


def place_on_mannequin(
    garment: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    build_bounds,
    push: bool = True,
    collision: str = "sdf",
    clearance_m: float = CLEARANCE,
    source: Path | None = None,
) -> trimesh.Trimesh:
    """Scale/translate the full-person bbox onto the mannequin, then clear the body."""
    garment = align_to_mannequin(garment, mannequin, build_bounds)
    if push:
        apply_collision(
            garment,
            mannequin,
            collision=collision,
            clearance_m=clearance_m,
            source=source,
        )
    return garment


def apply_collision(
    garment: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    *,
    collision: str = "sdf",
    clearance_m: float = CLEARANCE,
    hip_y: float = HIP_Y,
    torso_top: float = TORSO_ARM_Y,
    cover_back: bool = False,
    strip_arms: bool = True,
    source: Path | None = None,
    obstacle: trimesh.Trimesh | None = None,
    obstacle_clearance_m: float | None = None,
) -> dict | None:
    """Move `garment` off the body. `zpush` is the old depth-buffer push."""
    if collision == "zpush":
        _push_out_of_torso(
            garment,
            mannequin,
            hip_y=hip_y,
            torso_top=torso_top,
            strip_arms=strip_arms,
        )
        if cover_back:
            cover_back_bulge(garment, mannequin)
        return None
    if collision == "sdf":
        from pipeline.sdf_collision import push_clearance

        return push_clearance(
            garment,
            mannequin,
            clearance_m=clearance_m,
            source=source,
            obstacle=obstacle,
            obstacle_clearance_m=obstacle_clearance_m,
        )
    raise ValueError(f"unknown collision {collision!r}")


def _push_out_of_torso(
    garment: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    hip_y: float = HIP_Y,
    torso_top: float = TORSO_ARM_Y,
    strip_arms: bool = True,
) -> None:
    mlo, _mhi = mannequin.bounds
    mp, _ = trimesh.sample.sample_surface(mannequin, SAMPLE_MANNEQUIN)
    torso = np.ones(len(mp), bool)
    if not strip_arms:
        torso_top = float(mannequin.bounds[1][1])
    for y0 in np.arange(mlo[1], torso_top, 0.01):
        band = np.abs(mp[:, 1] - y0) < 0.005
        ax = np.sort(np.abs(mp[band, 0]))
        if len(ax) < 2:
            continue
        gaps = np.diff(ax)
        if strip_arms and gaps.max() > 0.012:
            torso[band & (np.abs(mp[:, 0]) > ax[gaps.argmax()])] = False
    mp = mp[torso]
    tcell = 0.005
    tx0, ty0 = mp[:, 0].min(), mp[:, 1].min()
    tr, tc = ((mp[:, 1] - ty0) / tcell).astype(int), ((mp[:, 0] - tx0) / tcell).astype(int)
    front_z = np.full((tr.max() + 1, tc.max() + 1), -np.inf)
    back_z = np.full_like(front_z, np.inf)
    np.maximum.at(front_z, (tr, tc), mp[:, 2])
    np.minimum.at(back_z, (tr, tc), mp[:, 2])

    V = garment.vertices
    gr = ((V[:, 1] - ty0) / tcell).astype(int)
    gc = ((V[:, 0] - tx0) / tcell).astype(int)
    on_grid = (gr >= 0) & (gr < front_z.shape[0]) & (gc >= 0) & (gc < front_z.shape[1])
    fz = np.full(len(V), -np.inf)
    bz = np.full(len(V), np.inf)
    fz[on_grid], bz[on_grid] = front_z[gr[on_grid], gc[on_grid]], back_z[gr[on_grid], gc[on_grid]]
    inside = (V[:, 2] < fz + CLEARANCE) & (V[:, 2] > bz - CLEARANCE)
    inside_torso = inside & (V[:, 1] > hip_y)
    to_front = inside_torso & ((fz - V[:, 2]) <= (V[:, 2] - bz))
    to_back = inside_torso & ~to_front
    V[to_front, 2] = fz[to_front] + CLEARANCE
    V[to_back, 2] = bz[to_back] - CLEARANCE
    garment.vertices = V


def cover_back_bulge(garment: trimesh.Trimesh, mannequin: trimesh.Trimesh) -> int:
    """Pull the back of a fitted dress out past the mannequin's back.

    The torso push only moves vertices that already sit between the front and
    back skin, and only from the hips to the armpits. The seat and the shoulder
    blades bulge past that band, so the dress stays in front of them and the
    body shows through. Vertices on the back half of the body, within
    BACK_COVER of the back skin, move to CLEARANCE behind that skin. Cloth
    that already flares past the body stays where it is. Arms are left out of
    the depth map so the dress is not pulled onto them.
    """
    mlo, mhi = mannequin.bounds
    mp, _ = trimesh.sample.sample_surface(mannequin, SAMPLE_MANNEQUIN)
    keep = np.ones(len(mp), bool)
    for y0 in np.arange(mlo[1], mhi[1], 0.01):
        band = np.abs(mp[:, 1] - y0) < 0.005
        ax = np.sort(np.abs(mp[band, 0]))
        if len(ax) < 2:
            continue
        gaps = np.diff(ax)
        if gaps.max() > 0.012:
            keep[band & (np.abs(mp[:, 0]) > ax[gaps.argmax()])] = False
    mp = mp[keep]
    cell = 0.005
    tx0, ty0 = mp[:, 0].min(), mp[:, 1].min()
    tr = ((mp[:, 1] - ty0) / cell).astype(int)
    tc = ((mp[:, 0] - tx0) / cell).astype(int)
    back_z = np.full((tr.max() + 1, tc.max() + 1), np.inf)
    np.minimum.at(back_z, (tr, tc), mp[:, 2])

    V = garment.vertices
    gr = ((V[:, 1] - ty0) / cell).astype(int)
    gc = ((V[:, 0] - tx0) / cell).astype(int)
    on = (gr >= 0) & (gr < back_z.shape[0]) & (gc >= 0) & (gc < back_z.shape[1])
    bz = np.full(len(V), np.inf)
    bz[on] = back_z[gr[on], gc[on]]
    # The mannequin faces +Z and is centered on z = 0, so the back half is z < 0.
    gap = V[:, 2] - bz
    pull = on & np.isfinite(bz) & (V[:, 2] < 0) & (gap > -CLEARANCE) & (gap < BACK_COVER)
    V[pull, 2] = bz[pull] - CLEARANCE
    garment.vertices = V
    return int(pull.sum())


def fit_garment(
    build,
    views: dict,
    mannequin: Path | trimesh.Trimesh,
    push: bool = True,
    collision: str = "sdf",
    clearance_m: float = CLEARANCE,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Extract + place. Returns (mannequin_mesh, garment_mesh)."""
    build_mesh = load_mesh(build)
    mannequin_mesh = load_mesh(mannequin)
    bounds = build_mesh.bounds
    garment = extract_garment(build_mesh, views)
    garment = place_on_mannequin(
        garment,
        mannequin_mesh,
        bounds,
        push=push,
        collision=collision,
        clearance_m=clearance_m,
        source=_sdf_source(mannequin),
    )
    mid_x = (mannequin_mesh.bounds[0][0] + mannequin_mesh.bounds[1][0]) / 2
    lateral = np.abs(garment.triangles_center[:, 0] - mid_x) < HIP_X
    if lateral.any() and not lateral.all():
        garment.update_faces(lateral)
        garment.remove_unreferenced_vertices()
    return mannequin_mesh, garment


def align_ghost_to_mannequin(
    mesh: trimesh.Trimesh,
    mannequin: trimesh.Trimesh,
    views: dict,
    build_bounds=None,
) -> trimesh.Trimesh:
    """Scale and position a garment-only (ghost) 3D mesh onto the mannequin.

    Unlike `align_to_mannequin` (which assumes the 3D mesh is a full human figure
    from head to toe), a ghost-mannequin reconstruction spans only the isolated
    garment (`g_top` to `g_bottom` in 2D). We use the 2D garment-to-person height
    ratio and hem-to-foot offset so the dress lands at the right torso height.
    """
    lo, hi = mesh.bounds if build_bounds is None else build_bounds
    build_height = float(hi[1] - lo[1])
    if build_height <= 0:
        raise ValueError("ghost build mesh has zero height")

    mlo, mhi = mannequin.bounds
    mannequin_height = float(mhi[1] - mlo[1])

    vf = views["front"]
    p_height = float(max(vf["bottom"] - vf["top"], 1))
    if "g_top" in vf and "g_bottom" in vf:
        g_top, g_bottom = float(vf["g_top"]), float(vf["g_bottom"])
    else:
        ys_g = np.nonzero(vf["garment"])[0]
        g_top = float(ys_g.min()) if len(ys_g) else float(vf["top"])
        g_bottom = float(ys_g.max()) if len(ys_g) else float(vf["bottom"])

    g_height = max(g_bottom - g_top, 1.0)
    height_ratio = float(np.clip(g_height / p_height, 0.15, 1.0))
    foot_to_hem_ratio = float(np.clip((vf["bottom"] - g_bottom) / p_height, 0.0, 0.85))

    target_garment_height = mannequin_height * height_ratio
    target_hem_y = float(mlo[1]) + mannequin_height * foot_to_hem_ratio
    scale = target_garment_height / build_height

    # Match torso Z center in the garment's vertical band (avoids toe-forward bbox bias)
    mv = mannequin.vertices
    band = (mv[:, 1] >= target_hem_y) & (mv[:, 1] <= target_hem_y + target_garment_height)
    band &= np.abs(mv[:, 0] - (mlo[0] + mhi[0]) / 2) < 0.22
    if band.any():
        target_z = float(0.5 * (mv[band, 2].min() + mv[band, 2].max()))
    else:
        target_z = float((mlo[2] + mhi[2]) / 2)
    target_x = float((mlo[0] + mhi[0]) / 2)

    out = mesh.copy()
    out.apply_translation(-np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]))
    out.apply_scale(scale)
    out.apply_translation([target_x, target_hem_y, target_z])
    return out


def fit_ghost_garment(
    build,
    views: dict,
    mannequin: Path | trimesh.Trimesh,
    push: bool = True,
    hip_y: float = HIP_Y,
    torso_top: float = TORSO_ARM_Y,
    collision: str = "sdf",
    clearance_m: float = CLEARANCE,
    obstacle: trimesh.Trimesh | None = None,
    obstacle_clearance_m: float | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Place a ghost-reconstructed garment mesh onto the mannequin without 3D mask carving.

    `hip_y` and `torso_top` bound the band that is pushed outside the skin.
    The defaults leave a skirt below the hips alone. A mini slip passes the
    hem and the bust so the whole dress sits on the body.
    """
    build_mesh = load_mesh(build)
    mannequin_mesh = load_mesh(mannequin)
    garment = build_mesh.copy()

    # Prune tiny disconnected crumbs using a welded copy (never trimesh.split)
    if len(garment.faces) > 0:
        welded = trimesh.Trimesh(garment.vertices, garment.faces, process=False)
        welded.merge_vertices(merge_tex=True, merge_norm=True)
        labels = trimesh.graph.connected_component_labels(
            welded.face_adjacency, node_count=len(welded.faces)
        )
        sizes = np.bincount(labels)
        largest = int(np.argmax(sizes))
        min_keep = max(200, int(0.02 * sizes[largest]))
        garment.update_faces(sizes[labels] >= min_keep)
        garment.remove_unreferenced_vertices()

    garment = align_ghost_to_mannequin(garment, mannequin_mesh, views, build_mesh.bounds)
    if push:
        apply_collision(
            garment,
            mannequin_mesh,
            collision=collision,
            clearance_m=clearance_m,
            hip_y=hip_y,
            torso_top=torso_top,
            cover_back=(collision == "zpush"),
            source=_sdf_source(mannequin),
            obstacle=obstacle,
            obstacle_clearance_m=obstacle_clearance_m,
        )

    mid_x = (mannequin_mesh.bounds[0][0] + mannequin_mesh.bounds[1][0]) / 2
    lateral = np.abs(garment.triangles_center[:, 0] - mid_x) < HIP_X
    if lateral.any() and not lateral.all():
        garment.update_faces(lateral)
        garment.remove_unreferenced_vertices()
    return mannequin_mesh, garment

