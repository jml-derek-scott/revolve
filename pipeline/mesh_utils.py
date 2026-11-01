"""Weld and component labels for AI meshes. Never call trimesh.split()."""
from __future__ import annotations

import numpy as np
import trimesh


def weld_copy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Copy, then merge vertices split on UV seams or flipped normals.

    The source mesh is left unchanged. Topology checks use the copy.
    """
    copy = mesh.copy()
    copy.merge_vertices(merge_tex=True, merge_norm=True)
    return copy


def component_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    """One label per face on a welded copy.

    Uses face adjacency after weld. Isolated faces stay their own component.
    """
    welded = weld_copy(mesh)
    count = len(welded.faces)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    labels = trimesh.graph.connected_component_labels(
        welded.face_adjacency, node_count=count
    )
    return np.asarray(labels, dtype=np.int64)
