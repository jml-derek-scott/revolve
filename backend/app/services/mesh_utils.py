import numpy as np
import trimesh

# TripoSR meshes are z-up with the subject facing +x and width along y.
_TSR_TO_YUP = np.array(
    [
        [0, 1, 0, 0],  # X' (right)  = y
        [0, 0, 1, 0],  # Y' (up)     = z
        [1, 0, 0, 0],  # Z' (front)  = x
        [0, 0, 0, 1],
    ],
    dtype=float,
)

# Generic z-up, facing -y (Blender-style; common for multi-view model exports).
_ZUP_TO_YUP = np.array(
    [
        [1, 0, 0, 0],  # X' (right)  = x
        [0, 0, 1, 0],  # Y' (up)     = z
        [0, -1, 0, 0],  # Z' (front) = -y
        [0, 0, 0, 1],
    ],
    dtype=float,
)

_CONVENTIONS = {"triposr": _TSR_TO_YUP, "z_up": _ZUP_TO_YUP}


def to_y_up(mesh: trimesh.Trimesh, convention: str = "triposr") -> trimesh.Trimesh:
    """Rotate a z-up mesh into glTF convention (y-up, facing +z).

    convention: "triposr" (z-up, facing +x) or "z_up" (z-up, facing -y). Check a new
    model's output against these before picking one.
    """
    mesh = mesh.copy()
    mesh.apply_transform(_CONVENTIONS[convention])
    return mesh
