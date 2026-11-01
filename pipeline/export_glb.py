"""Assemble mannequin + textured garment into one GLB."""
from pathlib import Path

import trimesh

MANNEQUIN_RGBA = [232, 229, 224, 255]


def export_mesh(mesh: trimesh.Trimesh, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    return out


def export_hybrid(
    mannequin: trimesh.Trimesh,
    garment: trimesh.Trimesh | dict[str, trimesh.Trimesh],
    out: Path,
) -> Path:
    """One mannequin node, plus `garment` or one node per name in a dict."""
    mannequin = mannequin.copy()
    mannequin.visual = trimesh.visual.ColorVisuals(mannequin, face_colors=MANNEQUIN_RGBA)
    scene = trimesh.Scene()
    scene.add_geometry(mannequin, geom_name="mannequin")
    parts = garment if isinstance(garment, dict) else {"garment": garment}
    for name, mesh in parts.items():
        scene.add_geometry(mesh, geom_name=str(name)[:48])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out)
    return out
