"""Project front, back, left, and right photos into a UV atlas.

A texel’s weight is `clip(|n · view|, 0, 1)`, so a facing photo wins over a
grazing one. Texels no photo painted copy the nearest painted texel on the
same UV island. TELEA runs only when that island has no painted texel.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

TEX = 2048
DEPTH_CELL = 0.004  # metres per depth-buffer cell
DEPTH_EPS = 0.015  # metres
FACING_MIN = 0.05  # wrap waist band; hard cutoff below this left holes
WEIGHT_EPS = 1e-3  # clip(|n · view|, 0, 1); grazing views lose the blend
INPAINT_RADIUS = 7
UV_DILATE = 8
SAMPLE = 2_000_000
NEUTRAL_RGB = 220  # fabric fallback when the reconstruction has no atlas
MANNEQUIN_RGB = (232, 229, 224)  # wrap-style skin: unpainted stays this, no TELEA smear


def _has_uvs(mesh: trimesh.Trimesh) -> bool:
    vis = getattr(mesh, "visual", None)
    uv = getattr(vis, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return False
    return float(np.ptp(uv, axis=0).min()) > 0.01


XATLAS_FACE_LIMIT = 80_000  # xatlas on TRELLIS-scale meshes can hang for minutes


def planar_unwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Front/back orthographic UVs, with vertices split so sides don't share charts.

    Front faces pack into u in [0, 0.49]; mirrored back faces into [0.51, 1].
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    lo, hi = mesh.bounds
    width = max(hi[0] - lo[0], 1e-6)
    height = max(hi[1] - lo[1], 1e-6)
    mid_z = 0.5 * (lo[2] + hi[2])
    front_face = mesh.triangles_center[:, 2] >= mid_z
    verts = V[F].reshape(-1, 3)
    faces = np.arange(len(F) * 3, dtype=np.int64).reshape(-1, 3)
    side = np.repeat(front_face, 3)
    u = (verts[:, 0] - lo[0]) / width
    v = (verts[:, 1] - lo[1]) / height
    uv = np.empty((len(verts), 2), dtype=np.float64)
    uv[side, 0] = 0.01 + 0.48 * u[side]
    uv[~side, 0] = 0.51 + 0.48 * (1.0 - u[~side])
    uv[:, 1] = 0.01 + 0.98 * v
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uv)
    return out


def cylindrical_unwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Continuous cylindrical UV unwrap around the vertical (Y) axis.

    Unlike `planar_unwrap` (which crushes 90-degree lateral side triangles to near-zero
    horizontal width), cylindrical unwrapping gives uniform texel density across front,
    left flank, back, and right flank. Triangles crossing the back seam (atan2 = +-pi)
    are unwrapped onto the consistent side of the seam.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    lo, hi = mesh.bounds
    height = max(hi[1] - lo[1], 1e-6)
    cx = 0.5 * (lo[0] + hi[0])
    cz = 0.5 * (lo[2] + hi[2])

    verts = V[F].reshape(-1, 3)
    faces = np.arange(len(F) * 3, dtype=np.int64).reshape(-1, 3)

    # theta in [-pi, pi]: 0 is +Z (front), +-pi is -Z (back), +pi/2 is +X, -pi/2 is -X
    theta = np.arctan2(verts[:, 0] - cx, verts[:, 2] - cz).reshape(-1, 3)
    # Prevent triangles straddling the -Z seam (+-pi) from stretching across the whole atlas
    span = theta.max(axis=1) - theta.min(axis=1)
    seam = span > np.pi
    if seam.any():
        neg = seam[:, None] & (theta < 0)
        theta = np.where(neg, theta + 2.0 * np.pi, theta)

    u = (theta.reshape(-1) + np.pi) / (2.0 * np.pi)
    u = 0.01 + 0.98 * np.clip(u, 0.0, 1.0)
    v = 0.01 + 0.98 * np.clip((verts[:, 1] - lo[1]) / height, 0.0, 1.0)
    uv = np.stack([u, v], axis=-1).astype(np.float64)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uv)
    return out


def unwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """xatlas when the mesh is small enough; otherwise a fast planar front/back atlas."""
    if len(mesh.faces) > XATLAS_FACE_LIMIT:
        print(f"  xatlas skipped ({len(mesh.faces)} faces > {XATLAS_FACE_LIMIT}); using planar UVs", flush=True)
        return planar_unwrap(mesh)
    import xatlas

    print(f"  xatlas unwrap ({len(mesh.faces)} faces)…", flush=True)
    vmapping, indices, uvs = xatlas.parametrize(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    faces = indices.reshape(-1, 3) if indices.ndim == 1 else indices
    kwargs = {}
    vc = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if vc is not None:
        kwargs["vertex_colors"] = np.asarray(vc)[vmapping]
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[vmapping],
        faces=faces,
        process=False,
        **kwargs,
    )
    out.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs, dtype=np.float64))
    if "vertex_colors" in kwargs:
        out.visual.vertex_colors = kwargs["vertex_colors"]
    return out


def ensure_uvs(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if _has_uvs(mesh):
        return mesh
    return unwrap(mesh)


def _rasterize_uv(
    mesh: trimesh.Trimesh,
    tex: int = TEX,
    contest: str = "last",
    planar: bool = False,
):
    """Rasterize 3D position + normal into the UV atlas.

    `contest="outer"`: front chart (u<0.5) keeps max z, back chart keeps min z,
    so a planar unwrap of a clothed body uses the outer garment, not the torso
    underneath.
    `contest="radial"`: keeps the outermost surface radially from the vertical
    centerline (for cylindrical unwraps or double-skinned mannequins).
    `last` is overwrite (fine for single-skin author UVs / xatlas).
    """
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    pos = np.full((tex, tex, 3), np.nan, np.float32)
    nrm = np.zeros((tex, tex, 3), np.float32)
    zbuf = np.full((tex, tex), np.nan, np.float32)
    tuv = np.stack([uv[:, 0] * tex - 0.5, (1 - uv[:, 1]) * tex - 0.5], -1)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    lo_b, hi_b = mesh.bounds
    cx_mid = float(0.5 * (lo_b[0] + hi_b[0]))
    cz_mid = float(0.5 * (lo_b[2] + hi_b[2]))
    n_faces = len(faces)
    outer = contest == "outer"
    radial_contest = contest == "radial"
    for i, f in enumerate(faces):
        if i and i % 40000 == 0:
            print(f"  uv raster {i}/{n_faces}", flush=True)
        a, b, c = tuv[f]
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int).clip(0, tex - 1)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int).clip(0, tex - 1)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        q = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)
        v0, v1, v2 = b - a, c - a, q - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        w1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
        w2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
        w0 = 1 - w1 - w2
        inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
        if not inside.any():
            continue
        w = np.stack([w0, w1, w2], -1)[inside]
        tx, ty = q[inside, 0].astype(int), q[inside, 1].astype(int)
        pts3 = w @ verts[f]
        tri = verts[f]
        fn = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        fn = fn / (np.linalg.norm(fn) + 1e-9)
        if planar:
            front_chart = float(uv[f, 0].mean()) < 0.5
            if front_chart and fn[2] < 0:
                fn = -fn
            elif not front_chart and fn[2] > 0:
                fn = -fn
        else:
            # Orient face normal outward from the horizontal torso/limb axis so
            # double-skinned mannequins with scrambled winding still face outward.
            tri_c = tri.mean(axis=0)
            rad_xz = np.array([tri_c[0] - cx_mid, 0.0, tri_c[2] - cz_mid], dtype=np.float32)
            if float(np.dot(fn, rad_xz)) < 0:
                fn = -fn
        if outer:
            front_chart = float(uv[f, 0].mean()) < 0.5
            zs = pts3[:, 2]
            old = zbuf[ty, tx]
            empty = np.isnan(old)
            better = empty | ((zs > old) if front_chart else (zs < old))
            if not better.any():
                continue
            pos[ty[better], tx[better]] = pts3[better]
            nrm[ty[better], tx[better]] = fn
            zbuf[ty[better], tx[better]] = zs[better]
        elif radial_contest:
            r2 = (pts3[:, 0] - cx_mid) ** 2 + (pts3[:, 2] - cz_mid) ** 2
            old = zbuf[ty, tx]
            empty = np.isnan(old)
            better = empty | (r2 >= old)
            if not better.any():
                continue
            pos[ty[better], tx[better]] = pts3[better]
            nrm[ty[better], tx[better]] = fn
            zbuf[ty[better], tx[better]] = r2[better]
        else:
            pos[ty, tx] = pts3
            nrm[ty, tx] = fn
    covered = ~np.isnan(pos[..., 0])
    nrm[covered] /= np.linalg.norm(nrm[covered], axis=1, keepdims=True) + 1e-9
    return pos, nrm, covered


def _depth_buffers(mesh: trimesh.Trimesh):
    pts, _ = trimesh.sample.sample_surface(mesh, SAMPLE)
    x0, y0 = mesh.bounds[0][0], mesh.bounds[0][1]
    gw = int((mesh.bounds[1][0] - x0) / DEPTH_CELL) + 1
    gh = int((mesh.bounds[1][1] - y0) / DEPTH_CELL) + 1

    def cell(xy):
        return (
            ((xy[:, 1] - y0) / DEPTH_CELL).astype(int).clip(0, gh - 1),
            ((xy[:, 0] - x0) / DEPTH_CELL).astype(int).clip(0, gw - 1),
        )

    zmax = np.full((gh, gw), -np.inf)
    zmin = np.full((gh, gw), np.inf)
    r, c = cell(pts)
    np.maximum.at(zmax, (r, c), pts[:, 2])
    np.minimum.at(zmin, (r, c), pts[:, 2])
    return zmax, zmin, cell, x0, y0


def _material_image(mesh: trimesh.Trimesh) -> Image.Image | None:
    vis = getattr(mesh, "visual", None)
    mat = getattr(vis, "material", None)
    img = None
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None and vis is not None:
        img = getattr(vis, "image", None)
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert("RGB")


def _fallback_rgb(val) -> np.ndarray:
    a = np.asarray(val, np.uint8).reshape(-1)
    if a.size == 1:
        return np.full(3, int(a[0]), np.uint8)
    return a[:3]


def overlay_lighten(rgb: np.ndarray, gamma: float = 0.7, amount: float = 0.55) -> np.ndarray:
    """Open crushed blacks, then Photoshop-overlay the original photo on top.

    `gamma` < 1 lifts shadows. `amount` is the overlay mix (0 = gamma only).
    """
    x = np.clip(np.asarray(rgb, np.float32) / 255.0, 0, 1)
    lifted = np.clip(np.power(x, gamma), 0, 1)
    if amount <= 0:
        return lifted * 255.0
    lo = 2.0 * lifted * x
    hi = 1.0 - 2.0 * (1.0 - lifted) * (1.0 - x)
    over = np.where(lifted < 0.5, lo, hi)
    out = (1.0 - amount) * lifted + amount * over
    return np.clip(out, 0, 1) * 255.0


def _torso_x_span_by_y(mesh: trimesh.Trimesh, y_min: float, y_max: float, n_bins: int = 256):
    """Compute torso/skirt horizontal [x_min(y), x_max(y)] on the 3D mesh, stripping arms."""
    V = np.asarray(mesh.vertices)
    ys = np.linspace(y_min, y_max, n_bins)
    dy = max((y_max - y_min) / max(n_bins - 1, 1), 0.005)
    x_lo = np.full(n_bins, np.nan, dtype=np.float32)
    x_hi = np.full(n_bins, np.nan, dtype=np.float32)
    mid_x = float(0.5 * (mesh.bounds[0][0] + mesh.bounds[1][0]))

    for i, y0 in enumerate(ys):
        band = np.abs(V[:, 1] - y0) <= dy * 1.25
        if not band.any():
            continue
        xs = V[band, 0]
        # Strip arms in the upper/mid body (where arms hang separate from torso)
        if y0 < 1.33 and len(xs) >= 8:
            ax = np.sort(np.abs(xs - mid_x))
            gaps = np.diff(ax)
            if len(gaps) and gaps.max() > 0.014:
                cut = ax[int(gaps.argmax())]
                xs = xs[np.abs(xs - mid_x) <= cut]
        if len(xs):
            x_lo[i] = float(xs.min())
            x_hi[i] = float(xs.max())

    # Interpolate any missing bins
    valid = ~np.isnan(x_lo)
    if valid.any():
        idx = np.arange(n_bins)
        x_lo = np.interp(idx, idx[valid], x_lo[valid]).astype(np.float32)
        x_hi = np.interp(idx, idx[valid], x_hi[valid]).astype(np.float32)
    else:
        x_lo[:] = float(mesh.bounds[0][0])
        x_hi[:] = float(mesh.bounds[1][0])
    return ys, x_lo, x_hi


def _project_px_with_silhouette_warp(
    P: np.ndarray,
    py: np.ndarray,
    px_linear: np.ndarray,
    sign: int,
    mask_2d: np.ndarray,
    ys_grid: np.ndarray,
    x3d_lo: np.ndarray,
    x3d_hi: np.ndarray,
    y_lo: float | None = None,
    y_hi: float | None = None,
    px_per_m: float | None = None,
) -> np.ndarray:
    """Warp horizontal pixel coordinate `px` per scanline so the 3D torso width
    maps edge-to-edge onto the 2D garment mask silhouette at that row `py`.

    Restricted to the waist–hip band (`y_lo`/`y_hi`) so halter straps and open
    shoulders keep their 2D cut instead of being stretched onto the deltoids.
    Rows where the 2D garment is much narrower than the 3D torso are left on
    linear projection (intentional dress silhouette, not a registration error).
    """
    H, W = mask_2d.shape
    px_out = px_linear.copy()
    ok_y = (py >= 0) & (py < H)
    if y_lo is not None:
        ok_y &= P[:, 1] >= y_lo
    if y_hi is not None:
        ok_y &= P[:, 1] <= y_hi
    if not ok_y.any():
        return px_out

    # Precompute 2D mask [gx_min, gx_max] per image row
    row_has = mask_2d.any(axis=1)
    first_x = np.argmax(mask_2d, axis=1)
    last_x = W - 1 - np.argmax(mask_2d[:, ::-1], axis=1)

    py_clamped = np.clip(py, 0, H - 1)
    has_row = ok_y & row_has[py_clamped]
    if not has_row.any():
        return px_out

    gx_min = first_x[py_clamped[has_row]].astype(np.float32)
    gx_max = last_x[py_clamped[has_row]].astype(np.float32)
    g_width = gx_max - gx_min

    # Only warp rows where the 2D garment span is wide enough to be a torso/skirt band
    valid_band = g_width >= 16.0
    if not valid_band.any():
        return px_out

    idx_pts = np.nonzero(has_row)[0][valid_band]
    gx_min = gx_min[valid_band]
    gx_max = gx_max[valid_band]
    g_width = g_width[valid_band]

    y_vals = P[idx_pts, 1]
    x_vals = P[idx_pts, 0]
    m_lo = np.interp(y_vals, ys_grid, x3d_lo)
    m_hi = np.interp(y_vals, ys_grid, x3d_hi)
    m_span = np.maximum(m_hi - m_lo, 1e-4)

    # Only warp vertices that actually lie within the 3D torso/skirt span [m_lo, m_hi];
    # vertices outside [m_lo, m_hi] are detached arms/hands and must keep linear px
    # so they don't get clamped onto the torso's side seam!
    in_torso = (x_vals >= m_lo - 0.008) & (x_vals <= m_hi + 0.008)
    # Skip rows where the 2D dress is much narrower than the 3D body (halter / open back)
    if px_per_m is not None and px_per_m > 1.0:
        in_torso &= (g_width / px_per_m) >= (0.75 * m_span)
    if not in_torso.any():
        return px_out

    idx_pts = idx_pts[in_torso]
    gx_min = gx_min[in_torso]
    gx_max = gx_max[in_torso]
    x_vals = x_vals[in_torso]
    m_lo = m_lo[in_torso]
    m_span = m_span[in_torso]

    # Normalized horizontal position across the 3D torso in [0, 1]
    u_3d = np.clip((x_vals - m_lo) / m_span, 0.0, 1.0)
    if sign < 0:
        u_3d = 1.0 - u_3d

    # Inset 1.5 px from mask extremes so grazing edges sample pure interior fabric
    inset = np.minimum(2.0, 0.05 * (gx_max - gx_min))
    warped = (gx_min + inset) + u_3d * ((gx_max - gx_min) - 2.0 * inset)
    px_out[idx_pts] = np.round(warped).astype(int)
    return px_out


def _fill_unpainted_islands(tex, covered, assigned, limit=None):
    """Copy the nearest painted texel on the same island. TELEA if the island has none."""
    import cv2
    from scipy.ndimage import distance_transform_edt, label

    holes = covered & (assigned == 0)
    if limit is not None:
        holes = holes & limit
    if not holes.any():
        return tex, assigned
    labels, n_islands = label(covered)
    telea = np.zeros(covered.shape, np.uint8)
    copied = 0
    for lab in range(1, int(n_islands) + 1):
        region = labels == lab
        missing = region & holes
        if not missing.any():
            continue
        painted = region & (assigned > 0)
        if not painted.any():
            telea[missing] = 255
            continue
        ys, xs = np.nonzero(region)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        field = np.ones((y1 - y0, x1 - x0), np.uint8)
        field[painted[y0:y1, x0:x1]] = 0
        _, inds = distance_transform_edt(field, return_indices=True)
        my, mx = np.nonzero(missing[y0:y1, x0:x1])
        tex[y0 + my, x0 + mx] = tex[y0 + inds[0][my, mx], x0 + inds[1][my, mx]]
        assigned[y0 + my, x0 + mx] = 255
        copied += int(my.size)
    n_telea = int(np.count_nonzero(telea))
    if n_telea:
        tex = cv2.inpaint(tex, telea, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        assigned[telea == 255] = 255
    print(
        f"  island fill copied {copied} texels, TELEA {n_telea} on empty islands",
        flush=True,
    )
    return tex, assigned


def bake_photos(
    mesh: trimesh.Trimesh,
    views: dict,
    tex_size: int = TEX,
    force_unwrap: bool = False,
    align_to: trimesh.Trimesh | None = None,
    mask_key: str = "garment",
    inpaint_unpainted: bool = True,
    fallback_rgb=NEUTRAL_RGB,
    unwrap_mode: str = "auto",
    gamma: float = 1.0,
    overlay: float = 0.0,
    also_paint_person: bool = False,
    warp_x_to_silhouette: bool = False,
    keep_mesh_texture: bool = False,
    side_max_x: float | None = None,
    background_max: int | None = None,
) -> trimesh.Trimesh:
    """Project front/back (and optional left/right) photos onto the UV atlas.

    Photo scale/centre come from `align_to` (the mannequin after placement), not
    the garment bbox — a dress is shorter than the person in the photo.

    `mask_key` is `garment` (hybrid) or `person` (full-build, includes skin/hair).
    Texels the photos cannot cover keep the reconstruction's own atlas (or a
    neutral fabric tone), never pitch black.

    `warp_x_to_silhouette`: maps the 3D torso's horizontal width per scanline
    onto the 2D garment mask's left/right boundaries so lateral flanks never
    fall outside the 2D dress mask.

    `background_max`: photo pixels brighter than this are studio background.
    Those samples are left unpainted so the island fill copies cloth instead
    of the white edge.
    """
    import cv2

    fill = _fallback_rgb(fallback_rgb)
    source = _material_image(mesh)
    planar = unwrap_mode == "planar"
    cylindrical = unwrap_mode == "cylindrical"
    if planar:
        mesh = planar_unwrap(mesh)
        uv_src = "planar"
        has_ai = False
        tex = np.full((tex_size, tex_size, 3), fill, np.uint8)
    elif cylindrical:
        mesh = cylindrical_unwrap(mesh)
        uv_src = "cylindrical"
        has_ai = False
        tex = np.full((tex_size, tex_size, 3), fill, np.uint8)
    elif force_unwrap or not _has_uvs(mesh):
        mesh = unwrap(mesh)
        uv_src = "generated"
        has_ai = False
        tex = np.full((tex_size, tex_size, 3), fill, np.uint8)
    else:
        uv_src = "existing"
        if source is None or (not keep_mesh_texture and fallback_rgb == MANNEQUIN_RGB):
            has_ai = False
            tex = np.full((tex_size, tex_size, 3), fill, np.uint8)
        else:
            has_ai = True
            if source.size != (tex_size, tex_size):
                source = source.resize((tex_size, tex_size), Image.Resampling.BILINEAR)
            tex = np.asarray(source, dtype=np.uint8).copy()
    fallback_name = "ai" if has_ai else ("skin" if not inpaint_unpainted else "neutral")
    print(
        f"  baking {len(mesh.faces)} faces, uv={uv_src}, fallback={fallback_name}, "
        f"silhouette_warp={warp_x_to_silhouette}",
        flush=True,
    )
    contest_mode = "outer" if planar else ("radial" if (cylindrical or not has_ai) else "last")
    pos, nrm, covered = _rasterize_uv(
        mesh, tex_size, contest=contest_mode, planar=planar
    )
    zmax, zmin, cell, _x0, _y0 = _depth_buffers(mesh)

    ref = align_to if align_to is not None else mesh
    mlo, mhi = ref.bounds
    height = mhi[1] - mlo[1]
    foot_y = mlo[1]
    mid_x = (mlo[0] + mhi[0]) / 2
    mid_z = (mlo[2] + mhi[2]) / 2

    # Waist–hip only: ~hips (0.45) to just below the armpits (0.66). Shoulders/neck
    # keep linear projection so a strap or halter is not stretched into sleeves.
    warp_y_lo = foot_y + 0.45 * height
    warp_y_hi = foot_y + 0.66 * height
    if warp_x_to_silhouette:
        print(f"  warp band y={warp_y_lo:.2f}–{warp_y_hi:.2f} (waist–hip)", flush=True)

    ys_grid, x3d_lo, x3d_hi = (
        _torso_x_span_by_y(mesh, float(mesh.bounds[0][1]), float(mesh.bounds[1][1]))
        if warp_x_to_silhouette
        else (None, None, None)
    )

    P, N = pos[covered], nrm[covered]
    r, c = cell(P)
    in_warp_band = (P[:, 1] >= warp_y_lo) & (P[:, 1] <= warp_y_hi)

    def project(key, lift=False):
        color = np.zeros((len(P), 3), np.float32)
        weight = np.zeros(len(P), np.float32)
        for side, sign, zbuf in (("front", 1, zmax), ("back", -1, zmin)):
            if side not in views:
                continue
            v = views[side]
            s = (v["bottom"] - v["top"]) / height
            px = (v["cx"] + sign * (P[:, 0] - mid_x) * s).round().astype(int)
            py = (v["bottom"] - (P[:, 1] - foot_y) * s).round().astype(int)
            layer = v[key] if key in v else v["garment"]
            if warp_x_to_silhouette and key == "garment":
                px = _project_px_with_silhouette_warp(
                    P, py, px, sign, layer, ys_grid, x3d_lo, x3d_hi,
                    y_lo=warp_y_lo, y_hi=warp_y_hi, px_per_m=s,
                )
            H, W = layer.shape
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            visible = sign * (P[:, 2] - zbuf[r, c]) > -DEPTH_EPS
            facing = np.clip(sign * N[:, 2], 0, 1)
            # When silhouette warp is active, allow grazing side normals in the
            # waist–hip band to receive color from the nearer camera.
            if warp_x_to_silhouette:
                nearer = in_warp_band & (
                    ((zmax[r, c] - P[:, 2]) <= (P[:, 2] - zmin[r, c])) if sign > 0
                    else ((P[:, 2] - zmin[r, c]) < (zmax[r, c] - P[:, 2]))
                )
                facing_eff = np.where(nearer, np.maximum(facing, 0.18), facing)
                m = ok & (visible | nearer) & (facing_eff > FACING_MIN)
            else:
                facing_eff = facing
                m = ok & visible & (facing > WEIGHT_EPS)
            m[m] &= layer[py[m], px[m]]
            if background_max is not None and m.any():
                pale = v["rgb"][py[m], px[m]].min(axis=1) > background_max
                m[np.flatnonzero(m)[pale]] = False
            w = np.where(m, facing_eff, 0)
            color[m] += w[m, None] * v["rgb"][py[m], px[m]]
            weight += w

        # Optional 4-view lateral projection (+X = left view, -X = right view)
        for side, sign_x in (("left", 1), ("right", -1)):
            if side not in views:
                continue
            v = views[side]
            s = (v["bottom"] - v["top"]) / height
            px = (v["cx"] - sign_x * (P[:, 2] - mid_z) * s).round().astype(int)
            py = (v["bottom"] - (P[:, 1] - foot_y) * s).round().astype(int)
            layer = v[key] if key in v else v["garment"]
            H, W = layer.shape
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            facing_x = np.clip(sign_x * N[:, 0], 0, 1)
            m = ok & (facing_x > WEIGHT_EPS)
            if side_max_x is not None:
                # Keep profile paint on the torso. Past this, the surface is an arm.
                m &= np.abs(P[:, 0] - mid_x) <= side_max_x
            m[m] &= layer[py[m], px[m]]
            if background_max is not None and m.any():
                pale = v["rgb"][py[m], px[m]].min(axis=1) > background_max
                m[np.flatnonzero(m)[pale]] = False
            w = np.where(m, facing_x, 0)
            color[m] += w[m, None] * v["rgb"][py[m], px[m]]
            weight += w

        painted = weight > 0
        color[painted] /= weight[painted, None]
        if lift and painted.any() and (gamma != 1.0 or overlay > 0):
            color[painted] = overlay_lighten(color[painted], gamma=gamma, amount=overlay)
        return color, painted

    color, painted = project(mask_key, lift=True)
    if also_paint_person and "person" in views["front"] and mask_key != "person":
        skin, skin_ok = project("person", lift=False)
        only_skin = skin_ok & ~painted
        color[only_skin] = skin[only_skin]
        painted = painted | skin_ok
        print("  also painted person/skin from photos", flush=True)
    if painted.any() and (gamma != 1.0 or overlay > 0):
        target = "garment" if also_paint_person and mask_key != "person" else "photos"
        print(f"  overlay-lift on {target} gamma={gamma} overlay={overlay}", flush=True)
    print(f"  projection ({painted.mean() * 100:.0f}% of surface painted)", flush=True)

    assigned = np.zeros((tex_size, tex_size), np.uint8)
    ys, xs = np.nonzero(covered)
    tex[ys[painted], xs[painted]] = color[painted].clip(0, 255).astype(np.uint8)
    assigned[ys[painted], xs[painted]] = 255

    if keep_mesh_texture and has_ai:
        # Paint atlas stays on texels the photos do not cover.
        assigned[covered] = 255
    elif inpaint_unpainted:
        unpainted = covered & (assigned == 0)
        tex[unpainted] = fill
        tex, assigned = _fill_unpainted_islands(tex, covered, assigned)
        assigned[covered] = 255
    elif covered.any():
        # Wrap preview: fill unpainted garment texels. Skin stays the body color.
        hole_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        in_garment = np.zeros(len(P), bool)
        for side, sign, zbuf in (("front", 1, zmax), ("back", -1, zmin)):
            if side not in views:
                continue
            v = views[side]
            s = (v["bottom"] - v["top"]) / height
            px = (v["cx"] + sign * (P[:, 0] - mid_x) * s).round().astype(int)
            py = (v["bottom"] - (P[:, 1] - foot_y) * s).round().astype(int)
            g = cv2.dilate(v[mask_key].astype(np.uint8), hole_k).astype(bool)
            if warp_x_to_silhouette and mask_key == "garment":
                px = _project_px_with_silhouette_warp(
                    P, py, px, sign, g, ys_grid, x3d_lo, x3d_hi,
                    y_lo=warp_y_lo, y_hi=warp_y_hi, px_per_m=s,
                )
            H, W = g.shape
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            hit = ok.copy()
            hit[hit] &= g[py[hit], px[hit]]
            in_garment |= hit
        holes = np.zeros((tex_size, tex_size), np.uint8)
        holes[ys[in_garment & ~painted], xs[in_garment & ~painted]] = 255
        if holes.any():
            tex, assigned = _fill_unpainted_islands(
                tex, covered, assigned, limit=holes == 255
            )
        print(
            f"  garment sides ({painted.mean() * 100:.0f}% painted, "
            f"{(in_garment & ~painted).mean() * 100:.0f}% unpainted)",
            flush=True,
        )
        assigned[covered] = 255

    dil = tex.copy()
    mask = assigned.copy()
    if keep_mesh_texture and has_ai:
        mask = np.maximum(mask, (tex.max(axis=2) > 8).astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(UV_DILATE):
        grown = cv2.dilate(dil, kernel)
        new = cv2.dilate(mask, kernel)
        edge = (new == 255) & (mask == 0)
        dil[edge] = grown[edge]
        mask = new
    tex_img = Image.fromarray(dil)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_img,
        metallicFactor=0.05,
        roughnessFactor=0.8,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
    return mesh
