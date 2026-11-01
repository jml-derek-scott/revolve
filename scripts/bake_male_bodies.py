"""Bake template 5 onto both male bodies.

    poetry run python scripts/bake_male_bodies.py

Basic is the plain gray mannequin. Model is the textured person. Body, face,
and eyes on the model ship at 0.001 scale while hair and shoes are at 0.01, so
those parts are scaled by 10 before the bake. Only the body, arms, and legs
receive the garment photos.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.export_glb import export_mesh  # noqa: E402
from pipeline.garment_extractor import load_mesh  # noqa: E402
from pipeline.preprocess import load_view  # noqa: E402
from pipeline.texture_bake import bake_photos  # noqa: E402

FRONT = ROOT / "frontend/assets/templates/5-front-full.jpg"
BACK = ROOT / "frontend/assets/templates/5-back-full.jpg"
BASIC = ROOT / "mannequins/mannequin_male.glb"
MODEL = ROOT / "mannequins/mannequin_male_model.glb"
OUT_BASIC = ROOT / "frontend/assets/models/5-wrap-basic.glb"
OUT_MODEL = ROOT / "frontend/assets/models/5-wrap-model.glb"

BAKE_GEOMS = {"body_body2_0", "body_arm_0", "body_leg_0"}


def _world_mesh(scene: trimesh.Scene, geom_name: str) -> trimesh.Trimesh:
    geom = scene.geometry[geom_name]
    node = scene.graph.geometry_nodes[geom_name][0]
    transform, _ = scene.graph.get(node)
    scale = float(np.linalg.norm(transform[:3, :3], axis=0).max())
    if scale < 0.002:
        fix = np.eye(4)
        fix[:3, :3] *= 10.0
        transform = fix @ transform
    mesh = geom.copy()
    mesh.apply_transform(transform)
    # These parts store UVs in UDIM tiles (u = 1–4). The bake rasterizer
    # expects 0–1, and each part already has its own texture for that tile.
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is not None and len(uv) == len(mesh.vertices):
        mesh.visual.uv = np.mod(np.asarray(uv, dtype=np.float64), 1.0)
    return mesh


def bake_basic(views: dict) -> None:
    mesh = load_mesh(BASIC)
    print(f"basic bounds {np.round(mesh.bounds, 3)}", flush=True)
    t0 = time.perf_counter()
    mesh = bake_photos(mesh, views, mask_key="garment", inpaint_unpainted=False, unwrap_mode="auto")
    print(f"basic bake {time.perf_counter() - t0:.1f}s", flush=True)
    export_mesh(mesh, OUT_BASIC)
    tex = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    if tex is not None:
        tex.save(OUT_BASIC.with_suffix(".png"))
    print(f"wrote {OUT_BASIC}", flush=True)


def bake_model(views: dict) -> None:
    scene = trimesh.load(MODEL, force="scene")
    parts = {name: _world_mesh(scene, name) for name in scene.geometry}
    align = trimesh.util.concatenate([parts[name] for name in BAKE_GEOMS])
    print(f"model body bounds {np.round(align.bounds, 3)}", flush=True)
    out = trimesh.Scene()
    t0 = time.perf_counter()
    for name, mesh in parts.items():
        if name in BAKE_GEOMS:
            mesh = bake_photos(
                mesh,
                views,
                mask_key="garment",
                inpaint_unpainted=False,
                unwrap_mode="auto",
                align_to=align,
                keep_mesh_texture=True,
            )
        out.add_geometry(mesh, geom_name=name[:48])
    print(f"model bake {time.perf_counter() - t0:.1f}s", flush=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    out.export(OUT_MODEL)
    print(f"wrote {OUT_MODEL}", flush=True)


def main() -> None:
    t0 = time.perf_counter()
    views = {
        "front": load_view(FRONT, cloth="segformer"),
        "back": load_view(BACK, cloth="segformer"),
    }
    print(f"masks {time.perf_counter() - t0:.1f}s", flush=True)
    bake_basic(views)
    bake_model(views)


if __name__ == "__main__":
    main()
