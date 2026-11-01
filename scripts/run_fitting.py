"""One garment: Hunyuan front and back, then an open shell on the mannequin.

    poetry run python scripts/run_fitting.py \\
      --config configs/templates/dress_slip.json \\
      --provider replicate

Plates come from scripts/make_canonical_views.py --config. Sheer outfits send
the gray vision plates. The raw GLB and the open shell are written under
outputs/shape/<outfit_id>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.export_glb import export_mesh  # noqa: E402
from pipeline.garment_deformer import fitted_shell  # noqa: E402
from pipeline.garment_extractor import align_ghost_to_mannequin, dress_band, load_mesh  # noqa: E402
from pipeline.outfit import load_outfit, repo_path  # noqa: E402
from pipeline.prepass import load_fit_views, plate_dir, recon_plate_paths  # noqa: E402
from pipeline.reconstruct import reconstruct  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--class", dest="class_id", help="Required when the outfit has more than one garment.")
    parser.add_argument("--provider", required=True, choices=("replicate", "modal"))
    parser.add_argument("--plates", help="Plate directory. Default: outputs/canonical/<outfit_id>/<class_id>/")
    parser.add_argument("--out", help="Open shell GLB. Default: outputs/shape/<outfit_id>/<class>_shell.glb")
    parser.add_argument("--save-build", help="Raw GLB. Default: outputs/shape/<outfit_id>/<class>_raw.glb")
    parser.add_argument(
        "--convention",
        default="none",
        choices=("none", "z_up", "triposr"),
        help="Rotate the raw mesh to y-up before fitting.",
    )
    parser.add_argument(
        "--remove-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask Hunyuan to cut the white canvas. Default on for ghost plates.",
    )
    args = parser.parse_args()

    outfit = load_outfit(args.config)
    garment = outfit.select_garment(args.class_id)
    directory = Path(args.plates) if args.plates else plate_dir(outfit, garment)
    front, back = recon_plate_paths(directory)
    print(f"reconstruct {front.name} + {back.name} via {args.provider}", flush=True)
    glb = reconstruct(
        args.provider,
        front,
        back,
        remove_background=args.remove_background,
    )
    shape_dir = ROOT / "outputs" / "shape" / outfit.outfit_id
    raw_path = Path(args.save_build) if args.save_build else shape_dir / f"{garment.class_id}_raw.glb"
    shell_path = Path(args.out) if args.out else shape_dir / f"{garment.class_id}_shell.glb"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(glb)
    print(f"wrote {raw_path}", flush=True)

    mesh = load_mesh(glb)
    if args.convention != "none":
        from backend.app.services.mesh_utils import to_y_up

        mesh = to_y_up(mesh, args.convention)
        export_mesh(mesh, raw_path)
        print(f"re-exported {args.convention} mesh to {raw_path}", flush=True)

    views = load_fit_views(directory)
    mannequin_path = repo_path(outfit.target_mannequin)
    mannequin = load_mesh(mannequin_path)
    guide = align_ghost_to_mannequin(mesh, mannequin, views, mesh.bounds)
    band = dress_band(views, mannequin)
    print(
        f"fitted shell y={band['hem']:.2f}–{band['top']:.2f} "
        f"clearance {garment.sdf_clearance_mm:.1f} mm",
        flush=True,
    )
    shell = fitted_shell(
        mannequin,
        guide,
        band["hem"],
        band["top"],
        clearance_m=garment.clearance_m,
        source=mannequin_path,
    )
    export_mesh(shell, shell_path)
    print(f"wrote {shell_path}", flush=True)


if __name__ == "__main__":
    main()
