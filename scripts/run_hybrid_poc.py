"""Offline 2-view hybrid PoC: cloud reconstruct → garment extract → photo UV bake.

    poetry run python scripts/run_hybrid_poc.py \\
        --front frontend/assets/templates/1-front.jpg \\
        --back  frontend/assets/templates/1-back.jpg \\
        --provider replicate \\
        --out frontend/assets/models/1-hybrid.glb

TRELLIS person + wrap dress (no mannequin, keep the build atlas on skin/sides):

    poetry run python scripts/run_hybrid_poc.py \\
        --front frontend/assets/templates/1-front-full.jpg \\
        --back  frontend/assets/templates/1-back-full.jpg \\
        --provider skip \\
        --build outputs/hybrid_src/1-build.glb \\
        --no-extract --no-write-build \\
        --cloth segformer --inpaint-occlusion \\
        --unwrap-mode auto \\
        --out frontend/assets/models/1-hybrid.glb

After a new `<n>-hybrid.glb` is written, add n to HAS_HYBRID in frontend/js/app.js.

Full build only (TRELLIS.2, front photo; back is ignored):

    poetry run python scripts/run_hybrid_poc.py \\
        --front frontend/assets/templates/2-front-full.jpg \\
        --back  frontend/assets/templates/2-back-full.jpg \\
        --provider trellis \\
        --no-cutout \\
        --build-only \\
        --save-build outputs/2-build-trellis.glb
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.canonical import DRESS_LIFT_M, TORSO_SIDE_X, load_placed_views  # noqa: E402
from pipeline.export_glb import export_hybrid, export_mesh  # noqa: E402
from pipeline.garment_extractor import (  # noqa: E402
    align_to_mannequin,
    apply_collision,
    dress_band,
    fit_garment,
    fit_ghost_garment,
    load_mesh,
)
from pipeline.preprocess import load_view, person_cutout, prepare_ghost_mannequin  # noqa: E402
from pipeline.reconstruct import reconstruct  # noqa: E402
from pipeline.texture_bake import bake_photos  # noqa: E402


def timed(label, t0):
    print(f"{label}: {time.perf_counter() - t0:.1f}s", flush=True)


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--front", required=True)
p.add_argument("--back", required=True)
p.add_argument(
    "--provider",
    default=os.environ.get("RECONSTRUCT_PROVIDER", "replicate"),
    choices=("modal", "replicate", "trellis", "hf", "skip"),
    help="Cloud reconstructor. trellis is TRELLIS.2 (front only). skip + --build uses a local GLB (no GPU).",
)
p.add_argument(
    "--build-only",
    action="store_true",
    help="Write the reconstructed GLB and stop before garment extract.",
)
p.add_argument("--build", help="Existing clothed GLB; skips reconstruction")
p.add_argument("--mannequin", default=str(ROOT / "mannequins" / "female_base_mesh.glb"))
p.add_argument("--out")
p.add_argument("--save-build", help="Write the reconstructed GLB here (default: outputs/poc_build.glb)")
p.add_argument("--no-cutout", action="store_true", help="Send original photos, not rembg cut-outs")
p.add_argument("--inpaint-occlusion", action="store_true", help="Fill bag/hair holes on the dress (LaMa, TELEA fallback)")
p.add_argument(
    "--cloth",
    default="u2net",
    choices=("u2net", "segformer"),
    help="Garment mask. segformer = Hugging Face clothes parser. Hybrid extract still defaults to u2net.",
)
p.add_argument(
    "--also-paint-person",
    action="store_true",
    help="With --no-extract, also project photo skin/hair. Default: dress only; build atlas stays on the person.",
)
p.add_argument("--no-push", action="store_true")
p.add_argument(
    "--unwrap",
    action="store_true",
    help="Rebuild UVs (planar front/back, or xatlas if small). Default: keep the build's UVs so its texture can be the fallback.",
)
p.add_argument(
    "--unwrap-mode",
    default="auto",
    choices=("auto", "planar", "cylindrical"),
    help="planar = front/back charts. cylindrical = even flank texels for dresses. "
    "auto keeps existing UVs or xatlas.",
)
p.add_argument("--gamma", type=float, default=1.0, help="Shadow lift on painted texels; <1 opens blacks (e.g. 0.7).")
p.add_argument("--overlay", type=float, default=0.0, help="0-1 Photoshop overlay of the photo onto the lifted dress.")
p.add_argument(
    "--no-write-build",
    action="store_true",
    help="Do not overwrite the sibling <n>-build.glb display file.",
)
p.add_argument(
    "--no-extract",
    action="store_true",
    help="Keep the full reconstruction (no mannequin). Bake wrap photos onto that mesh.",
)
p.add_argument(
    "--fallback",
    default=None,
    help="RGB fallback for unpainted texels, e.g. 32,32,36. Default: bake_photos neutral/skin.",
)
p.add_argument(
    "--convention",
    default="none",
    choices=("none", "triposr", "z_up"),
    help="Rotate the reconstructed mesh into glTF y-up before extract",
)
p.add_argument(
    "--ghost",
    "--ghost-prepass",
    dest="ghost_prepass",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run 2D Ghost Mannequin garment isolation before 3D reconstruction and use ghost placement.",
)
p.add_argument(
    "--ghost-dir",
    default=str(ROOT / "outputs" / "ghost_mannequin"),
    help="Directory to write intermediate ghost mannequin images and masks.",
)
p.add_argument(
    "--product-shot",
    action="store_true",
    help="Front/back are garment-only images. Reconstruct those, then fit the shell with fit_ghost_garment.",
)
p.add_argument(
    "--placement",
    help="placement.json from the dress canonical pass.",
)
p.add_argument(
    "--lift",
    type=float,
    default=None,
    help="Metres to raise the dress on the mannequin (photo-row shift). "
    f"Default {DRESS_LIFT_M} m for --product-shot: this mannequin's bust sits "
    "higher than the catalog body ratio, midi or mini. The hem moves the same "
    "amount (still below the knee on a midi). Pass 0 only to compare.",
)
p.add_argument("--left", help="Optional profile, texture only. Not sent to the reconstructor.")
p.add_argument("--right", help="Optional profile, texture only. Not sent to the reconstructor.")
p.add_argument(
    "--collision",
    choices=("sdf", "zpush"),
    default=None,
    help="sdf = outer-skin clearance (default for --ghost and --product-shot). "
    "zpush = the older front/back depth push, kept so a template can be compared.",
)
p.add_argument(
    "--shell",
    choices=("fitted",),
    default=None,
    help="fitted = open shell from the mannequin outer skin. Hunyuan is only the guide. "
    "Omit to export the reconstructed garment.",
)
p.add_argument(
    "--sheer-fill",
    action="store_true",
    help="Fill sheer/hollow regions with gray before Hunyuan so rembg does not "
    "tunnel through white fabric. Writes hunyuan-front/back.png next to the views. "
    "The accepted Sim2 PNGs are not overwritten.",
)
p.add_argument(
    "--remove-background",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Hunyuan rembg. Keep on after --sheer-fill (canvas stays white; gray is the garment).",
)
p.add_argument(
    "--straps",
    action="store_true",
    help="Add procedural spaghetti-strap ribbons after fit, before the photo bake.",
)
args = p.parse_args()

front = Path(args.front)
back = Path(args.back)
out = Path(args.out) if args.out else ROOT / "outputs" / "hybrid_poc" / f"{front.stem}_hybrid.glb"

views = None
ghost_inputs: dict[str, Path] = {}
if args.product_shot and args.ghost_prepass:
    sys.exit("error: --product-shot replaces the segmentation ghost prepass")
if args.product_shot:
    if not args.placement:
        sys.exit("error: --product-shot requires --placement")
    import json

    placement = json.loads(Path(args.placement).read_text())
    sides = ["front", "back"]
    if args.left:
        sides.append("left")
    if args.right:
        sides.append("right")
    lift_m = DRESS_LIFT_M if args.lift is None else args.lift
    views = load_placed_views(front.parent, placement, sides=tuple(sides), lift_m=lift_m)
    print(f"product shots ({', '.join(sides)}) placed from {args.placement}", flush=True)
elif args.ghost_prepass:
    t0 = time.perf_counter()
    ghost_dir = Path(args.ghost_dir)
    ghost_dir.mkdir(parents=True, exist_ok=True)
    front_ghost_res = prepare_ghost_mannequin(front, view="front", cloth=args.cloth, use_lama=True)
    back_ghost_res = prepare_ghost_mannequin(back, view="back", cloth=args.cloth, use_lama=True)
    for side, res in (("front", front_ghost_res), ("back", back_ghost_res)):
        g_path = ghost_dir / f"{side}_ghost.png"
        res["ghost_rgb"].save(g_path)
        res["ghost_rgba"].save(ghost_dir / f"{side}_ghost_alpha.png")
        res["raw_mask_img"].save(ghost_dir / f"{side}_raw_mask.png")
        res["occlusion_mask_img"].save(ghost_dir / f"{side}_occlusion_mask.png")
        ghost_inputs[side] = g_path
    views = {"front": front_ghost_res, "back": back_ghost_res}
    timed(f"ghost mannequin prepass ({ghost_dir})", t0)
elif not args.build_only:
    t0 = time.perf_counter()
    views = {
        "front": load_view(front, inpaint_occlusion=args.inpaint_occlusion, cloth=args.cloth),
        "back": load_view(back, inpaint_occlusion=args.inpaint_occlusion, cloth=args.cloth),
    }
    timed("masks", t0)

if args.build:
    build_path = Path(args.build)
    raw_alt = ROOT / "outputs" / "hybrid_src" / build_path.name
    if raw_alt.exists() and build_path.resolve() != raw_alt.resolve():
        print(f"using raw reconstruction {raw_alt} (not the mannequin-aligned display copy)", flush=True)
        build_path = raw_alt
    else:
        print(f"using existing build {build_path}", flush=True)
    build_mesh = load_mesh(build_path)
else:
    if args.provider == "skip":
        sys.exit("error: --provider skip requires --build <existing.glb>")
    t0 = time.perf_counter()
    if args.ghost_prepass:
        front_in, back_in = ghost_inputs["front"], ghost_inputs["back"]
    elif args.no_cutout or args.product_shot:
        front_in, back_in = front, back
    else:
        tmp = ROOT / "outputs" / "poc_cutouts"
        tmp.mkdir(parents=True, exist_ok=True)
        front_in = tmp / "front.png"
        back_in = tmp / "back.png"
        person_cutout(front).save(front_in)
        person_cutout(back).save(back_in)
    if args.sheer_fill:
        from pipeline.sheer_prepass import prepare_sheer_input

        fill_dir = front.parent
        front_in = prepare_sheer_input(front_in, fill_dir / "hunyuan-front.png")
        back_in = prepare_sheer_input(back_in, fill_dir / "hunyuan-back.png")
    print(
        f"reconstruct via {args.provider} (ghost={args.ghost_prepass}, "
        f"sheer_fill={args.sheer_fill}, rembg={args.remove_background})…",
        flush=True,
    )
    glb_bytes = reconstruct(
        args.provider,
        front_in,
        back_in,
        ghost_prepass=False,
        remove_background=args.remove_background,
    )
    save_build = Path(args.save_build) if args.save_build else ROOT / "outputs" / "poc_build.glb"
    save_build.parent.mkdir(parents=True, exist_ok=True)
    save_build.write_bytes(glb_bytes)
    timed(f"reconstruct ({len(glb_bytes)} bytes → {save_build})", t0)
    build_mesh = load_mesh(glb_bytes)

if args.convention != "none":
    from backend.app.services.mesh_utils import to_y_up

    build_mesh = to_y_up(build_mesh, args.convention)
    print(f"applied convention {args.convention}; bounds {build_mesh.bounds}", flush=True)
else:
    print(f"build bounds {build_mesh.bounds}", flush=True)

if args.build_only:
    if args.convention != "none":
        rotated = Path(args.save_build) if args.save_build else ROOT / "outputs" / "poc_build.glb"
        export_mesh(build_mesh, rotated)
        print(f"re-exported rotated mesh to {rotated}", flush=True)
    print("build-only: stopped before garment extract", flush=True)
    raise SystemExit(0)

bake_kwargs = dict(
    force_unwrap=args.unwrap,
    unwrap_mode=args.unwrap_mode,
    gamma=args.gamma,
    overlay=args.overlay,
)
if args.fallback:
    bake_kwargs["fallback_rgb"] = tuple(int(x) for x in args.fallback.split(","))
if args.product_shot and (args.left or args.right):
    bake_kwargs["side_max_x"] = TORSO_SIDE_X

if args.no_extract:
    # Full reconstruction as-is. Photos scale to this mesh, not the mannequin.
    # Dress pixels overwrite the build atlas; skin/sides keep it (no white fill).
    bake_kwargs.setdefault("mask_key", "garment")
    bake_kwargs.setdefault("also_paint_person", args.also_paint_person)
    bake_kwargs.setdefault("inpaint_unpainted", False)
    t0 = time.perf_counter()
    mesh = bake_photos(build_mesh, views, **bake_kwargs)
    timed("photo UV bake", t0)
    t0 = time.perf_counter()
    export_mesh(mesh, out)
    timed(f"wrote painted full-figure {out}", t0)
    print("done. Hybrid is the Full Build mesh with photo overlay (no mannequin).")
    raise SystemExit(0)

mannequin = load_mesh(args.mannequin)
bake_kwargs["align_to"] = mannequin

collision = args.collision
if collision is None:
    collision = "sdf" if (args.ghost_prepass or args.product_shot) else "zpush"
use_shell = args.shell == "fitted"
band = dress_band(views, mannequin)

t0 = time.perf_counter()
if args.ghost_prepass or args.product_shot:
    fit_kwargs = {"collision": collision}
    if args.product_shot:
        # The default push starts at the hips and stops at the armpits, which
        # hides a mini slip inside the body. Cover this dress from hem to bust.
        fit_kwargs["hip_y"] = band["hip_y"]
        fit_kwargs["torso_top"] = band["torso_top"]
        print(
            f"product-shot push y={band['hip_y']:.2f}–{band['torso_top']:.2f}",
            flush=True,
        )
    mannequin, garment = fit_ghost_garment(
        build_mesh,
        views,
        args.mannequin,
        push=not args.no_push and not use_shell,
        **fit_kwargs,
    )
    timed(f"ghost place (garment {len(garment.vertices)} verts, {len(garment.faces)} faces)", t0)
else:
    mannequin, garment = fit_garment(
        build_mesh,
        views,
        args.mannequin,
        push=not args.no_push and not use_shell,
        collision=collision,
    )
    timed(f"extract+place (garment {len(garment.vertices)} verts, {len(garment.faces)} faces)", t0)

if use_shell:
    from pipeline.garment_deformer import fitted_shell

    t0 = time.perf_counter()
    print(
        f"fitted shell y={band['hem']:.2f}–{band['top']:.2f} (Hunyuan is the guide)",
        flush=True,
    )
    garment = fitted_shell(
        mannequin,
        garment,
        band["hem"],
        band["top"],
        source=Path(args.mannequin),
    )
    if not args.no_push:
        apply_collision(
            garment,
            mannequin,
            collision=collision,
            hip_y=band["hip_y"],
            torso_top=band["torso_top"],
            cover_back=collision == "zpush",
            source=Path(args.mannequin),
        )
    timed(f"fitted shell ({len(garment.vertices)} verts, {len(garment.faces)} faces)", t0)

if args.straps:
    from pipeline.straps import attach_straps

    t0 = time.perf_counter()
    before = len(garment.faces)
    garment = attach_straps(garment, mannequin)
    timed(f"procedural straps (+{len(garment.faces) - before} faces)", t0)

t0 = time.perf_counter()
garment = bake_photos(garment, views, **bake_kwargs)
timed("photo UV bake", t0)

t0 = time.perf_counter()
export_hybrid(mannequin, garment, out)
timed(f"wrote {out}", t0)
if out.name.endswith("-hybrid.glb") and not args.no_write_build:
    build_out = out.with_name(out.name.replace("-hybrid.glb", "-build.glb"))
    raw_dir = ROOT / "outputs" / "hybrid_src"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / build_out.name
    if not raw_path.exists():
        export_mesh(build_mesh, raw_path)
    if args.ghost_prepass:
        # For ghost mode, <n>-build.glb is the standalone textured 3D ghost garment (no mannequin)
        export_mesh(garment, build_out)
        timed(f"wrote ghost build {build_out}", t0)
    else:
        full = align_to_mannequin(build_mesh, mannequin, build_mesh.bounds)
        export_mesh(full, build_out)
        timed(f"wrote full build {build_out}", t0)
print("done. If this is frontend/assets/models/<n>-hybrid.glb, add n to HAS_HYBRID in frontend/js/app.js")
