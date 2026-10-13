import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import TRIPOSR_DIR, TRIPOSR_MC_RESOLUTION
from .base import Reconstructor
from .mesh_utils import to_y_up

_FOREGROUND_RATIO = 0.85


class SingleViewReconstructor(Reconstructor):
    """Front-only: TripoSR (stabilityai/TripoSR) on the MPS device.

    Output is a vertex-coloured GLB. Texture baking (xatlas/moderngl) is not wired in.
    """

    name = "single_view"
    n_images = 1

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._rembg_session = None
        self._lock = threading.Lock()  # one inference at a time on the GPU

    def _load(self):
        if self._model is not None:
            return
        if str(TRIPOSR_DIR) not in sys.path:
            sys.path.insert(0, str(TRIPOSR_DIR))
        import rembg
        from torchmcubes import marching_cubes
        from tsr.system import TSR

        model = TSR.from_pretrained(
            "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
        )
        model.renderer.set_chunk_size(8192)
        model.to(self.device)
        model.set_marching_cubes_resolution(TRIPOSR_MC_RESOLUTION)
        # torchmcubes has no MPS support: run marching cubes on CPU (the helper moves results back).
        model.isosurface_helper.mc_func = lambda level, thr: marching_cubes(level.cpu(), thr)
        self._rembg_session = rembg.new_session()
        self._model = model

    def _preprocess(self, path: Path) -> Image.Image:
        from tsr.utils import remove_background, resize_foreground

        img = Image.open(path).convert("RGBA")
        if np.array(img)[..., 3].min() == 255:  # no transparency: cut out the background
            img = remove_background(img, self._rembg_session)
        img = resize_foreground(img, _FOREGROUND_RATIO)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
        return Image.fromarray((arr * 255).astype(np.uint8))

    def reconstruct(self, image_path: Path):
        """Returns the raw TripoSR trimesh (z-up, facing +x)."""
        with self._lock:
            self._load()
            image = self._preprocess(image_path)
            with torch.no_grad():
                codes = self._model([image], device=self.device)
            return self._model.extract_mesh(codes, True, resolution=TRIPOSR_MC_RESOLUTION)[0]

    def generate(self, images: list[Path], out_path: Path) -> Path:
        self._check_images(images)
        to_y_up(self.reconstruct(images[0])).export(out_path)
        return out_path
