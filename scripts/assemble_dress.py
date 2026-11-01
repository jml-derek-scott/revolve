"""Open shell, satin bow, over-shoulder straps, cylindrical photo bake.

    poetry run python scripts/assemble_dress.py \
      --config configs/templates/guess_dress.json

The raw Hunyuan GLB must already exist (scripts/run_fitting.py). The shell is
cut again at the bandeau so spaghetti straps do not pull the cloth up over
the shoulders. The bow is a height field of the back plate: the reconstruction
does not keep it as a separable mesh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.bow import add_bow_tails, build_bow, satin_loop_mask  # noqa: E402
from pipeline.export_glb import export_hybrid, export_mesh  # noqa: E402
from pipeline.garment_deformer import fitted_shell  # noqa: E402
from pipeline.garment_extractor import (  # noqa: E402
    align_ghost_to_mannequin,
    bandeau_top,
    dress_band,
    load_mesh,
)
from pipeline.outfit import load_outfit, repo_path  # noqa: E402
from pipeline.prepass import load_fit_views, plate_dir  # noqa: E402
from pipeline.straps import generate_over_shoulder_straps  # noqa: E402
from pipeline.texture_bake import bake_photos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--class", dest="class_id")
    parser.add_argument("--out", help="Default: outputs/guess/<outfit_id>.glb")
    args = parser.parse_args()

    outfit = load_outfit(args.config)
    garment = outfit.select_garment(args.class_id)
    directory = plate_dir(outfit, garment)
    views = load_fit_views(directory)
    mannequin_path = repo_path(outfit.target_mannequin)
    mannequin = load_mesh(mannequin_path)
    raw_path = ROOT / "outputs" / "shape" / outfit.outfit_id / f"{garment.class_id}_raw.glb"
    if not raw_path.exists():
        raise SystemExit(f"missing {raw_path}; run scripts/run_fitting.py first")
    raw = load_mesh(raw_path)
    guide = align_ghost_to_mannequin(raw, mannequin, views, raw.bounds)
    band = dress_band(views, mannequin)
    top = bandeau_top(views["front"], mannequin)
    print(
        f"shell y={band['hem']:.3f}–{top:.3f} (bandeau; photo top was {band['top']:.3f})",
        flush=True,
    )
    shell = fitted_shell(
        mannequin,
        guide,
        band["hem"],
        top,
        clearance_m=garment.clearance_m,
        source=mannequin_path,
    )
    shape_dir = raw_path.parent
    export_mesh(shell, shape_dir / f"{garment.class_id}_shell.glb")

    rgb = np.asarray(views["back"]["rgb"])
    cloth = np.asarray(views["back"]["garment"])
    loops = satin_loop_mask(rgb, cloth)
    preview = np.clip(rgb, 0, 255).astype(np.uint8)
    full = add_bow_tails(loops, cloth)
    tint = preview.copy()
    tint[full] = (tint[full] * 0.45 + np.array([170, 40, 40]) * 0.55).astype(np.uint8)
    Image.fromarray(tint).save(shape_dir / "bow_mask.png")
    bow, knot = build_bow(rgb, cloth, views["back"], shell, mannequin)
    export_mesh(bow, shape_dir / f"{garment.class_id}_bow.glb")
    baked = bake_photos(
        trimesh_concat(shell, bow),
        views,
        align_to=mannequin,
        mask_key="garment",
        unwrap_mode=garment.unwrap_mode,
        inpaint_unpainted=False,
        warp_x_to_silhouette=True,
    )
    straps = generate_over_shoulder_straps(shell, mannequin, knot)
    _pin_straps_black(baked, straps)
    combined = trimesh_concat(baked, straps)
    out = Path(args.out) if args.out else ROOT / "outputs" / "guess" / f"{outfit.outfit_id}.glb"
    export_hybrid(mannequin, combined, out)
    print(f"wrote {out}", flush=True)


def trimesh_concat(*meshes):
    import trimesh

    return trimesh.util.concatenate(list(meshes))


def _pin_straps_black(baked, straps) -> None:
    """Keep the ribbons black. The photo behind them is empty white."""
    import trimesh

    image = baked.visual.material.baseColorTexture.convert("RGB")
    tex = np.array(image)
    tex[:32, :32] = (8, 8, 10)
    baked.visual.material.baseColorTexture = Image.fromarray(tex)
    # glTF v is up, so the top-left of the image is u≈0, v≈1.
    straps.visual = trimesh.visual.TextureVisuals(
        uv=np.tile([[0.006, 0.994]], (len(straps.vertices), 1)),
        material=baked.visual.material,
    )


if __name__ == "__main__":
    main()
