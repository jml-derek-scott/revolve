from pathlib import Path

from .base import Reconstructor
from .multi_view import MultiViewReconstructor
from .single_view import SingleViewReconstructor

# Each backend is independent; models load lazily on first use.
_RECONSTRUCTORS: dict[int, Reconstructor] = {
    r.n_images: r for r in (SingleViewReconstructor(), MultiViewReconstructor())
}


def select_reconstructor(n_images: int) -> Reconstructor:
    """1 image -> single-view pipeline, 2 images (front + back) -> multi-view pipeline."""
    try:
        return _RECONSTRUCTORS[n_images]
    except KeyError:
        raise ValueError(f"unsupported number of images: {n_images}") from None


def run(images: list[Path], out_path: Path) -> tuple[Reconstructor, Path]:
    r = select_reconstructor(len(images))
    return r, r.generate(images, out_path)
