"""2× photo upscale via Replicate Real-ESRGAN. Cache under outputs/wrap_upscale/.

On 429 / failure, returns the original path so wrap can still run with the
source photo instead of inventing pixels.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "wrap_upscale"
REAL_ESRGAN_MODEL = "nightmareai/real-esrgan"


def _img_buf(path: Path) -> io.BytesIO:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    buf.name = path.name
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def upscale_image(path: Path, scale: int = 2) -> Path:
    """Return a cached 2× PNG, or `path` unchanged if upscale fails."""
    src = Path(path)
    if scale <= 1:
        return src
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{src.stem}_x{scale}.png"
    if cached.exists() and cached.stat().st_size > 0:
        print(f"  upscale cache hit {cached.name}", flush=True)
        return cached

    from pipeline.reconstruct import _read_url, _replicate_client

    client = _replicate_client()
    print(f"  Real-ESRGAN x{scale} via Replicate ({src.name})…", flush=True)
    output = None
    for attempt in range(3):
        try:
            output = client.run(
                REAL_ESRGAN_MODEL,
                input={
                    "image": _img_buf(src),
                    "scale": scale,
                    "face_enhance": False,
                },
            )
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                print("  Real-ESRGAN rate-limited (429); waiting 10s to retry...", flush=True)
                time.sleep(10.5)
                continue
            print(f"  Real-ESRGAN failed ({exc}); using original {src.name}", flush=True)
            return src

    try:
        if isinstance(output, (list, tuple)):
            output = output[0]
        data = output.read() if hasattr(output, "read") else _read_url(str(getattr(output, "url", None) or output))
        cached.write_bytes(data)
        print(f"  wrote {cached} ({cached.stat().st_size} bytes)", flush=True)
        return cached
    except Exception as exc:
        print(f"  could not save upscale ({exc}); using original {src.name}", flush=True)
        return src
