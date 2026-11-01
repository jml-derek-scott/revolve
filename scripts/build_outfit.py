"""Stack outfit shells with layered SDF clearance and export one GLB.

    poetry run python scripts/build_outfit.py \\
      --config configs/templates/hoodie_jeans_layered.json

Each class reads outputs/shape/<outfit_id>/<class>_raw.glb unless --shell
CLASS=path.glb overrides it. Plates come from the canonical prepass. The
first layer sits sdf_clearance_mm off the body. Each later layer sits that
far off the body and off the shell under it. --placed skips alignment when
the GLB is already in mannequin space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.export_glb import export_hybrid  # noqa: E402
from pipeline.garment_extractor import align_ghost_to_mannequin, load_mesh  # noqa: E402
from pipeline.outfit import load_outfit, repo_path  # noqa: E402
from pipeline.prepass import load_fit_views, plate_dir  # noqa: E402
from pipeline.sdf_collision import push_clearance  # noqa: E402
from pipeline.texture_bake import bake_photos  # noqa: E402


def _shell_overrides(items: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--shell must be CLASS=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--shell must be CLASS=PATH, got {item!r}")
        overrides[name] = Path(path)
    return overrides


def _shell_path(outfit_id: str, class_id: str, overrides: dict[str, Path]) -> Path:
    if class_id in overrides:
        return overrides[class_id]
    return ROOT / "outputs" / "shape" / outfit_id / f"{class_id}_raw.glb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--shell",
        action="append",
        default=[],
        metavar="CLASS=PATH",
        help="Existing GLB for one class. Repeat per garment.",
    )
    parser.add_argument(
        "--placed",
        action="store_true",
        help="Shells are already in mannequin space. Skip alignment.",
    )
    parser.add_argument("--out", help="Default: outputs/fitted/<outfit_id>.glb")
    parser.add_argument(
        "--out-root",
        default=str(ROOT / "outputs" / "canonical"),
        help="Parent of <outfit_id>/<class_id>/ plates.",
    )
    args = parser.parse_args()

    outfit = load_outfit(args.config)
    overrides = _shell_overrides(args.shell)
    mannequin_path = repo_path(outfit.target_mannequin)
    mannequin = load_mesh(mannequin_path)
    previous: trimesh.Trimesh | None = None
    parts: dict[str, trimesh.Trimesh] = {}
    for garment in outfit.ordered_garments():
        path = _shell_path(outfit.outfit_id, garment.class_id, overrides)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        mesh = load_mesh(path)
        extent = mesh.bounds[1] - mesh.bounds[0]
        if int(np.argmax(extent)) != 1:
            raise SystemExit(f"{garment.class_id} shell is not y-up (extent {np.round(extent, 3)})")
        views = load_fit_views(
            plate_dir(outfit, garment, Path(args.out_root)),
            lift_m=garment.lift_m,
        )
        if not args.placed:
            mesh = align_ghost_to_mannequin(mesh, mannequin, views, mesh.bounds)
        push_clearance(
            mesh,
            mannequin,
            clearance_m=garment.clearance_m,
            source=mannequin_path,
            obstacle=previous,
            obstacle_clearance_m=garment.clearance_m if previous is not None else None,
        )
        mesh = bake_photos(
            mesh,
            views,
            align_to=mannequin,
            mask_key="garment",
            unwrap_mode=garment.unwrap_mode,
            inpaint_unpainted=True,
            background_max=garment.background_max,
        )
        previous = mesh
        parts[garment.class_id] = mesh
        print(
            f"  {garment.class_id} clearance {garment.sdf_clearance_mm:.1f} mm "
            f"unwrap {garment.unwrap_mode}",
            flush=True,
        )
    out = Path(args.out) if args.out else ROOT / "outputs" / "fitted" / f"{outfit.outfit_id}.glb"
    export_hybrid(mannequin, parts, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
