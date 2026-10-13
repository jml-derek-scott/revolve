"""Placeholder mesh builder used until real models are wired in.

Builds a thin textured slab (front image on one face, back image on the other) so the
upload -> GLB -> viewer path can be exercised end to end.
"""
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


def _quad(image: Path, z: float, flip: bool) -> trimesh.Trimesh:
    img = Image.open(image)
    aspect = img.width / img.height
    w, h = aspect, 1.0
    v = np.array([[-w, -h, z], [w, -h, z], [w, h, z], [-w, h, z]]) / 2
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    if flip:
        f = f[:, ::-1]
        uv[:, 0] = 1 - uv[:, 0]
    return trimesh.Trimesh(
        v, f, visual=TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=img)), process=False
    )


def build_placeholder_glb(images: list[Path], out_path: Path) -> Path:
    parts = [_quad(images[0], 0.02, flip=False)]
    if len(images) > 1:
        parts.append(_quad(images[1], -0.02, flip=True))
    trimesh.Scene(parts).export(out_path)
    return out_path
