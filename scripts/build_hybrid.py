"""Hybrid prototype: garment shape + texture from a full-build mesh, body from the premade mannequin.

    venv/bin/python scripts/build_hybrid.py <full_build.glb> <front.jpg> <back.jpg> [--out out.glb]

1. Project the photos' clothing-segmentation masks onto the full-build mesh (it was generated from the
   front photo, so they line up) and keep only garment faces.
2. Drop everything else (skin, hair, face, bag, shoes).
3. Scale/align the garment onto the mannequin and push any vertices that sink inside the body outwards.
4. Export mannequin (plain off-white) + garment (keeps the full build's own texture) as one GLB.
Both meshes are expected y-up, facing +z.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import rembg
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANNEQUIN_RGBA = [232, 229, 224, 255]
DEPTH_CELL = 0.004  # fraction of mesh height per depth-buffer cell
CLEARANCE = 0.004  # metres the garment must sit outside the mannequin skin
MIN_COMPONENT_FACES = 300
CLOSE_STEPS = 12  # face rings for hole closing

p = argparse.ArgumentParser()
p.add_argument("build")
p.add_argument("front")
p.add_argument("back")
p.add_argument("--mannequin", default=str(ROOT / "mannequins" / "female_base_mesh.glb"))
p.add_argument("--out")
p.add_argument("--no-push", action="store_true", help="debug: skip pushing the garment out of the mannequin")
args = p.parse_args()


def timed(label, t0):
    print(f"{label}: {time.perf_counter() - t0:.1f}s", flush=True)


# ---- 1. masks (same as scripts/test_mannequin.py) ----
t0 = time.perf_counter()
person_session = rembg.new_session()
cloth_session = rembg.new_session("u2net_cloth_seg")


def load_view(path):
    img = Image.open(path).convert("RGB")
    person = np.asarray(rembg.remove(img, session=person_session, only_mask=True)) > 127
    masks = cloth_session.predict(img)
    garment = np.any([np.asarray(m.resize(img.size)) > 127 for m in masks], axis=0) & person
    ys, xs = np.nonzero(person)
    cx = np.median(xs[(ys > np.percentile(ys, 25)) & (ys < np.percentile(ys, 50))])
    return {"garment": garment, "top": ys.min(), "bottom": ys.max(), "cx": cx}


views = {"front": load_view(args.front), "back": load_view(args.back)}
timed("masks", t0)

# ---- 2. garment faces on the full-build mesh ----
t0 = time.perf_counter()
build = trimesh.load(args.build, force="mesh")
lo, hi = build.bounds
height = hi[1] - lo[1]
centers = build.triangles_center

# depth buffers so faces hidden behind e.g. an arm don't sample the torso's mask
cell = DEPTH_CELL * height
gw, gh = int((hi[0] - lo[0]) / cell) + 1, int(height / cell) + 1
pts, _ = trimesh.sample.sample_surface(build, 1_500_000)


def grid(xy):
    return (((xy[:, 1] - lo[1]) / cell).astype(int).clip(0, gh - 1),
            ((xy[:, 0] - lo[0]) / cell).astype(int).clip(0, gw - 1))


zmax, zmin = np.full((gh, gw), -np.inf), np.full((gh, gw), np.inf)
r, c = grid(pts)
np.maximum.at(zmax, (r, c), pts[:, 2])
np.minimum.at(zmin, (r, c), pts[:, 2])

# Assign each face to the photo of the side it's on, by depth rather than normals (TRELLIS normals are
# not reliably oriented): nearer the front surface -> front photo, nearer the back surface -> back photo.
r, c = grid(centers)
z = centers[:, 2]
front_side = (zmax[r, c] - z) <= (z - zmin[r, c])
keep = np.zeros(len(centers), bool)
for side, sign, on_side in (("front", 1, front_side), ("back", -1, ~front_side)):
    v = views[side]
    s = (v["bottom"] - v["top"]) / height
    px = (v["cx"] + sign * (centers[:, 0] - (lo[0] + hi[0]) / 2) * s).round().astype(int)
    py = (v["bottom"] - (centers[:, 1] - lo[1]) * s).round().astype(int)
    H, W = v["garment"].shape
    m = on_side & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    keep[m] = v["garment"][py[m], px[m]]
# Close small holes in the selection (the cloth mask is patchy on sheer/lace areas): grow the kept region
# by CLOSE_STEPS faces, then shrink it back. Holes narrower than ~2*CLOSE_STEPS faces fill in; the outer
# garment edge ends up where it started.
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
# Drop small disconnected bits. Label components instead of split(): split() copies the whole mesh
# (texture included) per component, and a noisy segmentation has thousands of them.
# TRELLIS duplicates vertices along UV seams, which breaks adjacency: label on a position-merged copy
# (merge_vertices only remaps indices, so face order and labels still line up).
welded = trimesh.Trimesh(garment.vertices, garment.faces, process=False)
welded.merge_vertices(merge_tex=True, merge_norm=True)
labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=len(welded.faces))
sizes = np.bincount(labels)
garment.update_faces(sizes[labels] >= MIN_COMPONENT_FACES)
garment.remove_unreferenced_vertices()
n_parts = int((sizes >= MIN_COMPONENT_FACES).sum())
timed(f"garment segmentation ({keep.mean() * 100:.0f}% of faces, {n_parts} of {len(sizes)} parts kept)", t0)

# ---- 3. place on the mannequin ----
t0 = time.perf_counter()
mannequin = trimesh.load(args.mannequin, force="mesh")
mlo, mhi = mannequin.bounds
# whole person -> whole mannequin: same height, feet on the floor, centred in x/z
scale = (mhi[1] - mlo[1]) / height
garment.apply_translation(-np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]))
garment.apply_scale(scale)
garment.apply_translation([(mlo[0] + mhi[0]) / 2, mlo[1], (mlo[2] + mhi[2]) / 2])

# Push garment vertices that sit inside (or too close to) the mannequin's torso out to its surface.
# The mannequin is a thick, double-skinned shell with scrambled winding, so normal- and ray-based inside
# tests fail. Instead use front/back depth maps of the torso (arms removed), like the texture wrap does:
# a point between the torso's front and back surfaces is inside and moves to the nearer one.
mp, _ = trimesh.sample.sample_surface(mannequin, 3_000_000)
torso = np.ones(len(mp), bool)
for y0 in np.arange(mlo[1], 1.32, 0.01):  # below the armpits the arms are separate from the torso
    band = np.abs(mp[:, 1] - y0) < 0.005
    ax = np.sort(np.abs(mp[band, 0]))
    if len(ax) < 2:
        continue
    gaps = np.diff(ax)
    if gaps.max() > 0.012:
        torso[band & (np.abs(mp[:, 0]) > ax[gaps.argmax()])] = False
mp = mp[torso]
TCELL = 0.005
tx0, ty0 = mp[:, 0].min(), mp[:, 1].min()
tr, tc = ((mp[:, 1] - ty0) / TCELL).astype(int), ((mp[:, 0] - tx0) / TCELL).astype(int)
front_z = np.full((tr.max() + 1, tc.max() + 1), -np.inf)
back_z = np.full_like(front_z, np.inf)
np.maximum.at(front_z, (tr, tc), mp[:, 2])
np.minimum.at(back_z, (tr, tc), mp[:, 2])

V = garment.vertices
gr = ((V[:, 1] - ty0) / TCELL).astype(int)
gc = ((V[:, 0] - tx0) / TCELL).astype(int)
on_grid = (gr >= 0) & (gr < front_z.shape[0]) & (gc >= 0) & (gc < front_z.shape[1])
fz = np.full(len(V), -np.inf)
bz = np.full(len(V), np.inf)
fz[on_grid], bz[on_grid] = front_z[gr[on_grid], gc[on_grid]], back_z[gr[on_grid], gc[on_grid]]
inside = (V[:, 2] < fz + CLEARANCE) & (V[:, 2] > bz - CLEARANCE)
to_front = inside & ((fz - V[:, 2]) <= (V[:, 2] - bz))
to_back = inside & ~to_front
if not args.no_push:
    V[to_front, 2] = fz[to_front] + CLEARANCE
    V[to_back, 2] = bz[to_back] - CLEARANCE
    garment.vertices = V
fix = inside
timed(f"placement ({inside.mean() * 100:.0f}% of garment vertices were inside the mannequin and moved out)", t0)

# ---- 4. export ----
mannequin.visual = trimesh.visual.ColorVisuals(mannequin, face_colors=MANNEQUIN_RGBA)
scene = trimesh.Scene()
scene.add_geometry(mannequin, geom_name="mannequin")
scene.add_geometry(garment, geom_name="garment")
out = Path(args.out) if args.out else ROOT / "outputs" / "hybrid_test" / f"{Path(args.front).stem}_hybrid.glb"
out.parent.mkdir(parents=True, exist_ok=True)
scene.export(out)
print(f"wrote {out} (garment {len(garment.vertices)} verts)")
