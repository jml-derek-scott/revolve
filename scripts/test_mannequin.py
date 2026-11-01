"""Project front/back product photos onto a mesh as a texture.

Classic mannequin wrap (no reconstruction):

    poetry run python scripts/test_mannequin.py <front> <back> [--out path.glb]

Better dress extract (SegFormer clothes + LaMa bag/hair fill):

    poetry run python scripts/test_mannequin.py <front> <back> \\
        --cloth segformer --inpaint-occlusion \\
        --out frontend/assets/models/1-wrap.glb

Wrap onto a reconstructed full-build (Hunyuan / TRELLIS) so the garment keeps
real drape while prints stay photo-sharp:

    poetry run python scripts/test_mannequin.py <front> <back> \\
        --mesh outputs/hybrid_src/2-hunyuan.glb \\
        --out frontend/assets/models/2-wrap.glb

Only the garment is painted. Unpainted skin/hair stay mannequin-white.
Mesh is expected y-up, facing +z.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.canonical import DRESS_LIFT_M, TORSO_SIDE_X, load_placed_views  # noqa: E402
from pipeline.export_glb import export_mesh  # noqa: E402
from pipeline.garment_extractor import align_to_mannequin, load_mesh  # noqa: E402
from pipeline.preprocess import load_view, prepare_ghost_mannequin  # noqa: E402
from pipeline.texture_bake import MANNEQUIN_RGB, bake_photos  # noqa: E402
from pipeline.upscale import upscale_image  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("front")
p.add_argument("back")
p.add_argument("--left", help="Optional 90-deg left profile view for 4-view projection")
p.add_argument("--right", help="Optional 270-deg right profile view for 4-view projection")
p.add_argument("--mannequin", default=str(ROOT / "mannequins" / "female_base_mesh.glb"))
p.add_argument("--mesh", help="Mesh to wrap (Hunyuan/TRELLIS GLB). Default: the mannequin.")
p.add_argument("--out")
p.add_argument(
    "--no-align",
    action="store_true",
    help="Leave --mesh in its own frame instead of matching the mannequin bbox",
)
p.add_argument(
    "--cloth",
    default="u2net",
    choices=("u2net", "segformer"),
    help="Garment mask. segformer = Hugging Face clothes parser (dress vs bag vs hair).",
)
p.add_argument(
    "--inpaint-occlusion",
    action="store_true",
    help="Fill bag/hair holes on the dress (LaMa, TELEA fallback) before projecting.",
)
p.add_argument(
    "--ghost",
    "--ghost-prepass",
    dest="ghost_prepass",
    action="store_true",
    help="Run 2D Ghost Mannequin garment isolation + LaMa inpainting before wrapping.",
)
p.add_argument(
    "--silhouette-warp",
    action="store_true",
    help="Row-wise horizontal X-alignment: maps 3D torso width to 2D garment mask width so flanks never miss.",
)
p.add_argument(
    "--unwrap-mode",
    default=None,
    choices=("auto", "planar", "cylindrical"),
    help="UV unwrap mode. Default: 'planar' when --mesh is set (unless --keep-mesh-texture), else 'auto'.",
)
p.add_argument(
    "--keep-mesh-texture",
    action="store_true",
    help="Keep existing mesh texture (e.g. Hunyuan3D-Paint) on unpainted grazing sides instead of mannequin white.",
)
p.add_argument(
    "--upscale",
    type=int,
    default=1,
    help="Real-ESRGAN scale for front/back (and left/right) photos before masking. 1 = off.",
)
p.add_argument(
    "--product-shot",
    action="store_true",
    help="Front/back (and optional sides) are garment-only images on white. Place them with --placement.",
)
p.add_argument(
    "--placement",
    help="placement.json from the dress canonical pass (person box and bust-to-hem).",
)
args = p.parse_args()


def timed(label, t0):
    print(f"{label}: {time.perf_counter() - t0:.1f}s", flush=True)


t0 = time.perf_counter()
front_path = Path(args.front)
back_path = Path(args.back)
left_path = Path(args.left) if args.left else None
right_path = Path(args.right) if args.right else None
if args.upscale and args.upscale > 1:
    front_path = upscale_image(front_path, scale=args.upscale)
    back_path = upscale_image(back_path, scale=args.upscale)
    if left_path is not None:
        left_path = upscale_image(left_path, scale=args.upscale)
    if right_path is not None:
        right_path = upscale_image(right_path, scale=args.upscale)
    timed("upscale", t0)
    t0 = time.perf_counter()

if args.product_shot:
    if not args.placement:
        p.error("--product-shot requires --placement")
    import json

    placement = json.loads(Path(args.placement).read_text())
    sides = ["front", "back"]
    if left_path is not None:
        sides.append("left")
    if right_path is not None:
        sides.append("right")
    shot_dir = front_path.parent
    views = load_placed_views(shot_dir, placement, sides=tuple(sides), lift_m=DRESS_LIFT_M)
    print(f"product shots from {shot_dir} ({', '.join(sides)})", flush=True)
elif args.ghost_prepass:
    ghost_cloth = "segformer" if args.cloth == "u2net" else args.cloth
    views = {
        "front": prepare_ghost_mannequin(front_path, view="front", cloth=ghost_cloth, use_lama=True),
        "back": prepare_ghost_mannequin(back_path, view="back", cloth=ghost_cloth, use_lama=True),
    }
    if left_path is not None:
        views["left"] = prepare_ghost_mannequin(left_path, view="left", cloth=ghost_cloth, use_lama=True)
    if right_path is not None:
        views["right"] = prepare_ghost_mannequin(right_path, view="right", cloth=ghost_cloth, use_lama=True)
else:
    views = {
        "front": load_view(front_path, inpaint_occlusion=args.inpaint_occlusion, cloth=args.cloth),
        "back": load_view(back_path, inpaint_occlusion=args.inpaint_occlusion, cloth=args.cloth),
    }
    if left_path is not None:
        views["left"] = load_view(left_path, inpaint_occlusion=args.inpaint_occlusion, cloth=args.cloth)
    if right_path is not None:
        views["right"] = load_view(right_path, inpaint_occlusion=args.inpaint_occlusion, cloth=args.cloth)
    if args.cloth == "u2net":
        # Close holes in sheer / lace so the wrap doesn't paint skin through the dress.
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        for v in views.values():
            v["garment"] = cv2.morphologyEx(v["garment"].astype(np.uint8), cv2.MORPH_CLOSE, close_k).astype(bool)
timed("masks", t0)

mannequin = load_mesh(args.mannequin)
if args.mesh:
    mesh = load_mesh(args.mesh)
    if not args.no_align:
        mesh = align_to_mannequin(mesh, mannequin, mesh.bounds)
        print(f"aligned mesh to mannequin; bounds {mesh.bounds}", flush=True)
    else:
        print(f"mesh bounds {mesh.bounds}", flush=True)
else:
    mesh = mannequin

if args.unwrap_mode is not None:
    unwrap_mode = args.unwrap_mode
elif args.mesh and not args.keep_mesh_texture:
    unwrap_mode = "planar"
else:
    unwrap_mode = "auto"

t0 = time.perf_counter()
mesh = bake_photos(
    mesh,
    views,
    mask_key="garment",
    inpaint_unpainted=False,
    fallback_rgb=MANNEQUIN_RGB,
    unwrap_mode=unwrap_mode,
    align_to=mannequin if args.mesh and not args.no_align else None,
    warp_x_to_silhouette=args.silhouette_warp,
    keep_mesh_texture=args.keep_mesh_texture,
    side_max_x=TORSO_SIDE_X if args.product_shot and ("left" in views or "right" in views) else None,
)
timed("photo UV bake", t0)

out = Path(args.out) if args.out else ROOT / "outputs" / "mannequin_test" / f"{Path(args.front).stem}_mannequin.glb"
out.parent.mkdir(parents=True, exist_ok=True)
export_mesh(mesh, out)
tex = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
if tex is not None:
    tex.save(out.with_suffix(".png"))
dbg = ROOT / "outputs" / "mannequin_test"
dbg.mkdir(parents=True, exist_ok=True)
for side, v in views.items():
    for key in ("garment", "bag", "occlusion"):
        if key not in v:
            continue
        Image.fromarray((v[key].astype(np.uint8) * 255)).save(dbg / f"{out.stem}_{side}_{key}.png")
print(f"wrote {out}")
