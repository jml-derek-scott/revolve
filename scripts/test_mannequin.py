"""Prototype: project front/back product photos onto a premade mannequin as a texture.

    venv/bin/python scripts/test_mannequin.py <front> <back> [--mannequin mannequins/female_base_mesh.glb]

No diffusion / reconstruction. Steps:
1. rembg person cut-out (for alignment) + u2net_cloth_seg (garment-only mask).
2. Align each photo to the mannequin by person height and horizontal centre (orthographic).
3. Bake a texture in the mannequin's own UVs: front photo for front-facing surfaces, mirrored back
   photo for back-facing ones, blended by normal, with a depth test against self-occlusion.
   Texels without garment coverage keep the plain mannequin colour.
Mannequin is expected y-up, facing +z, feet at y=0 (glTF convention).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rembg
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANNEQUIN_RGB = np.array([232, 229, 224], np.float32)  # off-white
TEX = 2048
DEPTH_CELL = 0.004  # metres per depth-buffer cell
DEPTH_EPS = 0.015  # metres

p = argparse.ArgumentParser()
p.add_argument("front")
p.add_argument("back")
p.add_argument("--mannequin", default=str(ROOT / "mannequins" / "female_base_mesh.glb"))
p.add_argument("--out")
args = p.parse_args()


def timed(label, t0):
    print(f"{label}: {time.perf_counter() - t0:.1f}s", flush=True)


# ---- 1. masks ----
t0 = time.perf_counter()
person_session = rembg.new_session()
cloth_session = rembg.new_session("u2net_cloth_seg")


def load_view(path):
    img = Image.open(path).convert("RGB")
    person = np.asarray(rembg.remove(img, session=person_session, only_mask=True)) > 127
    # cloth_seg predicts upper / lower / full-body masks; the garment is their union.
    masks = cloth_session.predict(img)
    garment = np.any([np.asarray(m.resize(img.size)) > 127 for m in masks], axis=0) & person
    ys, xs = np.nonzero(person)
    cx = np.median(xs[(ys > np.percentile(ys, 25)) & (ys < np.percentile(ys, 50))])  # torso centre
    return {"rgb": np.asarray(img, np.float32), "garment": garment, "top": ys.min(), "bottom": ys.max(), "cx": cx}


views = {"front": load_view(args.front), "back": load_view(args.back)}
for k, v in views.items():
    Image.fromarray((v["garment"] * 255).astype(np.uint8)).save(ROOT / "outputs" / f"mannequin_{k}_garment_mask.png")
timed("masks", t0)

# ---- 2. mannequin geometry in texel space ----
t0 = time.perf_counter()
scene = trimesh.load(args.mannequin)
mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
uv = mesh.visual.uv
height = mesh.bounds[1][1] - mesh.bounds[0][1]
foot_y = mesh.bounds[0][1]

pos = np.full((TEX, TEX, 3), np.nan, np.float32)
nrm = np.zeros((TEX, TEX, 3), np.float32)
tuv = np.stack([uv[:, 0] * TEX - 0.5, (1 - uv[:, 1]) * TEX - 0.5], -1)  # texel coords, row 0 = v 1
vn = mesh.vertex_normals
for f in mesh.faces:
    a, b, c = tuv[f]
    lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int).clip(0, TEX - 1)
    hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int).clip(0, TEX - 1)
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
    pos[ty, tx] = w @ mesh.vertices[f]
    nrm[ty, tx] = w @ vn[f]
covered = ~np.isnan(pos[..., 0])
nrm[covered] /= np.linalg.norm(nrm[covered], axis=1, keepdims=True) + 1e-9
timed(f"uv raster ({covered.mean() * 100:.0f}% of texture used)", t0)

# ---- 3. depth buffers (front = max z, back = min z) for self-occlusion ----
pts, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
x0, y0 = mesh.bounds[0][0], mesh.bounds[0][1]
gw = int((mesh.bounds[1][0] - x0) / DEPTH_CELL) + 1
gh = int((mesh.bounds[1][1] - y0) / DEPTH_CELL) + 1


def cell(xy):
    return (((xy[:, 1] - y0) / DEPTH_CELL).astype(int).clip(0, gh - 1),
            ((xy[:, 0] - x0) / DEPTH_CELL).astype(int).clip(0, gw - 1))


zmax = np.full((gh, gw), -np.inf)
zmin = np.full((gh, gw), np.inf)
r, c = cell(pts)
np.maximum.at(zmax, (r, c), pts[:, 2])
np.minimum.at(zmin, (r, c), pts[:, 2])

# ---- 4. project ----
t0 = time.perf_counter()
P, N = pos[covered], nrm[covered]
r, c = cell(P)
color = np.zeros((len(P), 3), np.float32)
weight = np.zeros(len(P), np.float32)
for side, sign, zbuf in (("front", 1, zmax), ("back", -1, zmin)):
    v = views[side]
    s = (v["bottom"] - v["top"]) / height  # px per metre, person height -> mannequin height
    px = (v["cx"] + sign * P[:, 0] * s).round().astype(int)  # back photo is mirrored in x
    py = (v["bottom"] - (P[:, 1] - foot_y) * s).round().astype(int)
    H, W = v["garment"].shape
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    visible = sign * (P[:, 2] - zbuf[r, c]) > -DEPTH_EPS
    facing = np.clip(sign * N[:, 2], 0, 1)
    m = ok & visible & (facing > 0.05)
    m[m] &= v["garment"][py[m], px[m]]
    w = np.where(m, facing**2, 0)
    color[m] += w[m, None] * v["rgb"][py[m], px[m]]
    weight += w
painted = weight > 0
color[painted] /= weight[painted, None]
color[~painted] = MANNEQUIN_RGB
timed(f"projection ({painted.mean() * 100:.0f}% of surface painted)", t0)

tex = np.tile(MANNEQUIN_RGB, (TEX, TEX, 1))
tex[covered] = color
tex = tex.clip(0, 255).astype(np.uint8)
# bleed colours past UV island edges so mip-mapping doesn't pull in background
dil = tex.copy()
mask = covered.copy()
for _ in range(8):
    grown = cv2.dilate(dil, np.ones((3, 3), np.uint8))
    new = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~mask
    dil[new] = grown[new]
    mask |= new
tex_img = Image.fromarray(dil)

material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_img, metallicFactor=0.0, roughnessFactor=0.8)
mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
out = Path(args.out) if args.out else ROOT / "outputs" / "mannequin_test" / f"{Path(args.front).stem}_mannequin.glb"
out.parent.mkdir(parents=True, exist_ok=True)
mesh.export(out)
tex_img.save(out.with_suffix(".png"))
print(f"wrote {out}")
