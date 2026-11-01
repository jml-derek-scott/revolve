"""UV-seam weld. Does not call trimesh.split."""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

from pipeline.mesh_utils import component_labels, weld_copy


def _uv_seam_quad() -> trimesh.Trimesh:
    """Two triangles, one geometric edge, vertices split by different UVs."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    uv = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 1.0],
            [0.6, 1.0],
        ],
        dtype=np.float64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv)
    return mesh


def test_weld_joins_uv_seam_and_leaves_source():
    mesh = _uv_seam_quad()
    welded = weld_copy(mesh)
    assert len(mesh.vertices) == 6
    assert len(welded.vertices) == 4
    labels = component_labels(mesh)
    assert len(mesh.vertices) == 6
    assert len(set(labels.tolist())) == 1
    assert len(labels) == 2
