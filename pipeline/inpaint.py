"""Occlusion fill: LaMa on Replicate, TELEA if the API is down."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

# allen-act/lama 404s; this is the live LaMa cog (image + mask).
LAMA_MODEL = "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72"


def _png_bytes(arr: np.ndarray) -> io.BytesIO:
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    buf.name = "image.png"
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _read_output(output) -> np.ndarray:
    if output is None:
        raise RuntimeError("LaMa returned nothing")
    if isinstance(output, (list, tuple)):
        return _read_output(output[0])
    data = None
    if hasattr(output, "read"):
        data = output.read()
    elif isinstance(output, str) and output.startswith("http"):
        import urllib.request

        data = urllib.request.urlopen(output, timeout=60).read()
    elif isinstance(output, (bytes, bytearray)):
        data = bytes(output)
    else:
        url = getattr(output, "url", None)
        if url:
            import urllib.request

            data = urllib.request.urlopen(url, timeout=60).read()
    if not data:
        raise RuntimeError(f"could not read LaMa output: {type(output)}")
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def telea(rgb: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    src = np.clip(rgb, 0, 255).astype(np.uint8)
    hole = (mask.astype(np.uint8) * 255)
    return cv2.inpaint(src, hole, radius, cv2.INPAINT_TELEA)


def lama_replicate(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import time
    from pipeline.reconstruct import _replicate_client

    client = _replicate_client()
    print(f"  LaMa via Replicate ({mask.mean() * 100:.1f}% of frame)", flush=True)
    for attempt in range(3):
        try:
            output = client.run(
                LAMA_MODEL,
                input={
                    "image": _png_bytes(rgb),
                    "mask": _png_bytes((mask.astype(np.uint8) * 255)),
                },
            )
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                print("  LaMa rate-limited (429); waiting 10s to retry...", flush=True)
                time.sleep(10.5)
                continue
            raise
    filled = _read_output(output)
    if filled.shape[:2] != rgb.shape[:2]:
        filled = np.asarray(
            Image.fromarray(filled).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)
        )
    return filled


def inpaint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill `mask` in `rgb`. Tries LaMa, then TELEA."""
    if not mask.any():
        return np.clip(rgb, 0, 255).astype(np.uint8)
    try:
        return lama_replicate(rgb, mask)
    except Exception as exc:
        print(f"  LaMa failed ({exc}); TELEA fallback", flush=True)
        return telea(rgb, mask)
