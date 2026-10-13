import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Reconstructor
from .mesh_utils import to_y_up

_IMAGE_SIZE = 512  # square input resolution for the multi-view model
_FOREGROUND_RATIO = 0.85  # fraction of the canvas the subject's longest side fills
_BG_GREY = 0.5


class MultiViewReconstructor(Reconstructor):
    """Front + back views -> one mesh from a native multi-view model.

    Unlike the old stitcher, both views condition a single inference pass, so there is
    no per-view mesh to merge. The model itself is not wired in yet: see the TODOs in
    _load() and reconstruct(). Until then generate() raises NotImplementedError (the
    API returns 501).
    """

    name = "multi_view"
    n_images = 2
    # Axis convention of the model's raw mesh, passed to to_y_up(). Set to match the model.
    output_convention = "triposr"  # InstantMesh: z-up facing +x (verified)

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._rembg_session = None
        self._lock = threading.Lock()  # one inference at a time on the GPU

    def _load(self) -> None:
        if self._model is not None:
            return
        import rembg

        self._rembg_session = rembg.new_session()

        # TODO: load a reconstructor that takes several posed input views, so the real
        # back photo is used. Note SF3D, Zero123++ and Unique3D all take ONE image.
        # Planned: InstantMesh (TencentARC/InstantMesh) - Zero123++ generates 6 views
        # (azimuths 30..330, so none at 0/180), the real front/back are added as extra
        # views at azimuth 0/180, then the InstantMesh LRM turns all posed views into a
        # mesh. Its raw mesh is z-up facing +x, i.e. output_convention "triposr".
        # Alternative: LGM (4 views, but needs a CUDA-only Gaussian rasterizer).
        # Device split that works on this machine (see scripts/test_instantmesh.py):
        #   - Zero123++ on CPU, float32: on MPS it OOMs the ~10 GB budget.
        #   - LRM forward_planes on self.device (MPS).
        #   - FlexiCubes extract_mesh on CPU: needs one ~15 GB buffer.
        # nvdiffrast (CUDA-only) must be stubbed; vertex-coloured export never uses it.
        self._model = None

    def _prepare_views(self, paths: list[Path]) -> list[Image.Image]:
        """Cut out each view and place them on identical canvases.

        Both views share one scale (set by the larger subject), are horizontally
        centred and bottom-aligned, so the subject has the same size and ground line
        in front and back.
        """
        cutouts = [self._cutout(p) for p in paths]
        boxes = [c.getchannel("A").getbbox() for c in cutouts]
        if any(b is None for b in boxes):
            raise ValueError("no foreground found in one of the views")
        crops = [c.crop(b) for c, b in zip(cutouts, boxes)]

        longest = max(max(c.size) for c in crops)
        scale = _FOREGROUND_RATIO * _IMAGE_SIZE / longest
        bottom = round((1 + _FOREGROUND_RATIO) / 2 * _IMAGE_SIZE)  # shared ground line

        views = []
        for crop in crops:
            w, h = (max(1, round(s * scale)) for s in crop.size)
            crop = crop.resize((w, h), Image.LANCZOS)
            canvas = Image.new("RGBA", (_IMAGE_SIZE, _IMAGE_SIZE), (0, 0, 0, 0))
            canvas.alpha_composite(crop, ((_IMAGE_SIZE - w) // 2, bottom - h))
            views.append(canvas)
        return views

    def _cutout(self, path: Path) -> Image.Image:
        import rembg

        img = Image.open(path).convert("RGBA")
        if np.array(img)[..., 3].min() == 255:  # no transparency: cut out the background
            img = rembg.remove(img, session=self._rembg_session)
        return img

    def _preprocess(self, paths: list[Path]) -> torch.Tensor:
        """Returns a (V, 3, H, W) float tensor on self.device, alpha-composited on grey.

        TODO: adapt to the chosen model's input spec (normalisation, background colour,
        camera poses per view: front = azimuth 0, back = azimuth 180; side views
        come from Zero123++). See scripts/test_instantmesh.py.
        """
        batch = []
        for view in self._prepare_views(paths):
            arr = np.asarray(view).astype(np.float32) / 255.0
            rgb = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * _BG_GREY
            batch.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(batch).to(self.device)

    def reconstruct(self, images: list[Path]):
        """Returns the model's raw trimesh in its native axis convention."""
        with self._lock:
            self._load()
            views = self._preprocess(images)
            if self._model is None:
                raise NotImplementedError(
                    "multi-view model not wired in yet; send only a front image for now"
                )
            with torch.no_grad():
                # TODO: run the multi-view reconstructor on all posed views at once and
                # return a trimesh.Trimesh.
                # Move outputs to CPU before converting to numpy.
                raise NotImplementedError(views.shape)

    def generate(self, images: list[Path], out_path: Path) -> Path:
        self._check_images(images)
        to_y_up(self.reconstruct(images), self.output_convention).export(out_path)
        return out_path
