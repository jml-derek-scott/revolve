"""Fit painted hoodie and jeans shells onto the textured male body.

    poetry run python scripts/build_male_outfit.py

Reads outputs/canonical/5/{hoodie,jeans}-painted.glb when those exist, otherwise
the grey shells. Projects the real plates on top and keeps an existing Paint
atlas on the sides. Scales each shell to the full figure (hair to shoes).

Jeans sit 4 mm off the body. The hoodie sits 4 mm off the body, and 4 mm off
the jeans where the two overlap. Arms stay in the distance field, so the
sleeves are pushed off the arms. Writes frontend/assets/models/5-hybrid.glb.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.canonical import CLASS_LABELS  # noqa: E402
from pipeline.clothes_seg import parse  # noqa: E402
from pipeline.garment_extractor import align_ghost_to_mannequin, load_mesh  # noqa: E402
from pipeline.sdf_collision import push_clearance  # noqa: E402
from pipeline.texture_bake import bake_photos  # noqa: E402
from scripts.bake_male_bodies import BAKE_GEOMS, MODEL, _world_mesh  # noqa: E402

OUT = ROOT / "frontend/assets/models/5-hybrid.glb"
CANON = ROOT / "outputs/canonical/5"
FRONT = ROOT / "frontend/assets/templates/5-front-full.jpg"
MALE_SDF = ROOT / "mannequins" / "male_body_bake.sdf.npz"
CLEARANCE = 0.004


def _plate_view(path: Path) -> dict:
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image)
    garment = np.any(rgb < 242, axis=2)
    ys, xs = np.nonzero(garment)
    if len(ys) == 0:
        raise ValueError(f"no garment pixels in {path}")
    return {
        "rgb": rgb.astype(np.float32),
        "garment": garment,
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "cx": float(np.median(xs)),
    }


def _body_view(class_name: str) -> dict:
    """Person frame from the catalog front, garment span from this class."""
    image = Image.open(FRONT).convert("RGB")
    parsed = parse(image)
    person = parsed["person"]
    cloth = parsed["labels"] == CLASS_LABELS[class_name]
    ys = np.nonzero(person)[0]
    gys = np.nonzero(cloth)[0]
    if len(ys) == 0 or len(gys) == 0:
        raise ValueError(f"no person or {class_name} in {FRONT}")
    return {
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "g_top": int(gys.min()),
        "g_bottom": int(gys.max()),
    }


def _source(name: str) -> Path:
    painted = CANON / f"{name}-painted.glb"
    shell = CANON / f"{name}-shell.glb"
    path = painted if painted.exists() else shell
    if not path.exists():
        raise SystemExit(f"missing {painted.name} and {shell.name}")
    print(f"{name} from {path.name}", flush=True)
    return path


def _fit(name: str, figure: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = load_mesh(_source(name))
    extent = mesh.bounds[1] - mesh.bounds[0]
    if int(np.argmax(extent)) != 1:
        raise SystemExit(f"{name} shell is not y-up (extent {np.round(extent, 3)})")
    views = {
        "front": _plate_view(CANON / name / "front.png"),
        "back": _plate_view(CANON / name / "back.png"),
    }
    painted = (CANON / f"{name}-painted.glb").exists()
    mesh = bake_photos(
        mesh,
        views,
        mask_key="garment",
        inpaint_unpainted=False,
        unwrap_mode="auto",
        keep_mesh_texture=painted,
    )
    place = {"front": _body_view(name)}
    # Scale against the whole figure (hair to shoes). Clearance runs after
    # both shells are placed, against the body (and the jeans, for the hoodie).
    mesh = align_ghost_to_mannequin(mesh, figure, place, mesh.bounds)
    print(f"  {name} y={mesh.bounds[0][1]:.2f}–{mesh.bounds[1][1]:.2f}", flush=True)
    return mesh


def main() -> None:
    scene = trimesh.load(MODEL, force="scene")
    parts = {name: _world_mesh(scene, name) for name in scene.geometry}
    body = trimesh.util.concatenate([parts[name] for name in BAKE_GEOMS])
    figure = trimesh.util.concatenate(list(parts.values()))
    print(f"body bounds {np.round(body.bounds, 3)}", flush=True)
    print(f"figure bounds {np.round(figure.bounds, 3)}", flush=True)
    out = trimesh.Scene()
    for name, mesh in parts.items():
        out.add_geometry(mesh, geom_name=name[:48])
    jeans = _fit("jeans", figure)
    push_clearance(jeans, body, clearance_m=CLEARANCE, cache=MALE_SDF)
    hoodie = _fit("hoodie", figure)
    push_clearance(
        hoodie,
        body,
        clearance_m=CLEARANCE,
        cache=MALE_SDF,
        obstacle=jeans,
        obstacle_clearance_m=CLEARANCE,
    )
    out.add_geometry(jeans, geom_name="jeans")
    out.add_geometry(hoodie, geom_name="hoodie")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.export(OUT)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
