"""Fill sheer / hollow regions so Hunyuan does not treat them as background.

Ideal product shots sit on #FFFFFF. Sheer panels and the empty interior of a
hollow dress are also near-white, so `remove_background=True` carves tunnels
through the garment. This pass keeps the accepted PNGs untouched and writes a
reconstruction-only copy:

- Dark cloth (bandeau, slip, overlay, straps) stays as photographed.
- Enclosed holes (midriff sheer, hollow interior) are filled with gray.
- The region between spaghetti straps stays open (it touches the top border).
- Thin straps are dilated a few pixels so a 384³ occupancy grid can see them.
- Exterior background stays white so the provider can still cut the canvas.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Dark fill, not mid-gray: Hunyuan rembg treated #808080 as a second object
# and still tunneled the midriff. One dark silhouette on white is the point.
GRAY = (24, 24, 24)
# Studio white. Walkable background for the flood that finds the silhouette.
BG_LUMA = 248
# Kept as photographed (opaque bandeau, slip, strap). Lighter cloth is sheer
# and is replaced by gray so Hunyuan does not see a window.
OPAQUE_LUMA = 48
STRAP_DILATE_PX = 11
CLOSE_PX = 3
# Only the top of the frame is strap territory. Dilating the whole opaque
# region closed the midi slit.
STRAP_ZONE = 0.28


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def cloth_mask(rgb: np.ndarray) -> np.ndarray:
    """True on the garment, including sheer. False on studio white.

    Flood from the corners through near-white. Enclosed light panels (midriff)
    are not reachable and stay in the silhouette. The neckline between straps
    is open to the top border, so it stays background.
    """
    luma = _luma(rgb)
    walkable = luma >= BG_LUMA
    h, w = walkable.shape
    flood = np.where(walkable, np.uint8(0), np.uint8(255))
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 128)
    silhouette = flood != 128
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX))
    return cv2.morphologyEx(silhouette.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def prepare_sheer_input(
    image_path: Path,
    out_path: Path,
    *,
    gray: tuple[int, int, int] = GRAY,
    strap_dilate_px: int = STRAP_DILATE_PX,
) -> Path:
    """Write a Hunyuan input next to the accepted view. Does not overwrite it.

    The whole silhouette is one dark fill. Pasting the original sheer pixels
    back recreates the midriff window that rembg tunnels through.
    """
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    luma = _luma(rgb)
    silhouette = cloth_mask(rgb)
    h = luma.shape[0]
    strap_zone = np.zeros_like(silhouette)
    strap_zone[: int(h * STRAP_ZONE)] = True
    straps = (luma < OPAQUE_LUMA) & strap_zone
    if strap_dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (strap_dilate_px, strap_dilate_px)
        )
        straps = cv2.dilate(straps.astype(np.uint8), k).astype(bool)

    canvas = np.full_like(rgb, 255)
    canvas[silhouette | straps] = np.array(gray, dtype=np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print(
        f"  sheer fill {out_path.name}: silhouette {silhouette.mean()*100:.1f}% "
        f"straps {straps.mean()*100:.1f}%",
        flush=True,
    )
    return out_path
