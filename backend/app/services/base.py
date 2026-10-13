from abc import ABC, abstractmethod
from pathlib import Path

import torch


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Reconstructor(ABC):
    """A 3D reconstruction backend: image(s) in, GLB file out."""

    name: str
    n_images: int  # how many views generate() expects, in order (front, back, ...)

    def __init__(self) -> None:
        self.device = get_device()

    def _check_images(self, images: list[Path]) -> None:
        if len(images) != self.n_images:
            raise ValueError(f"{self.name} expects {self.n_images} image(s), got {len(images)}")

    @abstractmethod
    def generate(self, images: list[Path], out_path: Path) -> Path:
        """images: [front] or [front, back]. Must write a .glb to out_path."""
