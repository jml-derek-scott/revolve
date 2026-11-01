"""Cloud 3D reconstruction providers: modal | replicate | trellis | hf.

Returns raw GLB bytes. No automatic failover — pick one with RECONSTRUCT_PROVIDER
or --provider. Tokens come from the environment (.env loaded by the CLI).

The Hugging Face Space client is a small HTTP helper rather than `gradio_client`:
a current gradio-client needs huggingface-hub>=0.19, which conflicts with the
TripoSR-pinned transformers==4.35.0. If the Space API hides session state
(common), download a GLB from the Space UI and pass --provider skip --build.
"""
from __future__ import annotations

import base64
import json
import os
import uuid
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

APP_NAME = "garment-3d-reconstruction"
FUNCTION_NAME = "generate_3d_mesh"
REPLICATE_MODEL = (
    "tencent/hunyuan3d-2mv:71798fbc3c9f7b7097e3bb85496e5a797d8b8f616b550692e7c3e176a8e9e5db"
)
# Version pin: POST /models/fishwowater/trellis2/predictions 404s; the version endpoint works.
TRELLIS_MODEL = (
    "fishwowater/trellis2:52e1ad6852599ea10ce8e257635a3c11485cba51c181ea5173e34d9b2955b226"
)
HF_SPACE = "https://microsoft-trellis-2.hf.space"


def reconstruct(
    provider: str,
    front: Path,
    back: Path,
    ghost_prepass: bool = False,
    ghost_dir: Path | None = None,
    use_lama: bool = True,
    remove_background: bool = True,
) -> bytes:
    provider = provider.lower().strip()
    if ghost_prepass:
        from pipeline.preprocess import prepare_ghost_mannequin

        out_dir = Path(ghost_dir) if ghost_dir else Path("outputs/ghost_mannequin")
        out_dir.mkdir(parents=True, exist_ok=True)
        front_res = prepare_ghost_mannequin(front, view="front", use_lama=use_lama)
        back_res = prepare_ghost_mannequin(back, view="back", use_lama=use_lama)
        front_ghost = out_dir / "front_ghost.png"
        back_ghost = out_dir / "back_ghost.png"
        front_res["ghost_rgb"].save(front_ghost)
        back_res["ghost_rgb"].save(back_ghost)
        front_res["ghost_rgba"].save(out_dir / "front_ghost_alpha.png")
        back_res["ghost_rgba"].save(out_dir / "back_ghost_alpha.png")
        front_res["raw_mask_img"].save(out_dir / "front_raw_mask.png")
        back_res["raw_mask_img"].save(out_dir / "back_raw_mask.png")
        front_res["occlusion_mask_img"].save(out_dir / "front_occlusion_mask.png")
        back_res["occlusion_mask_img"].save(out_dir / "back_occlusion_mask.png")
        front, back = front_ghost, back_ghost

    if provider == "replicate":
        return via_replicate(front, back, remove_background=remove_background)
    if provider == "trellis":
        return via_trellis(front, back)
    if provider == "modal":
        return via_modal(front, back)
    if provider == "hf":
        return via_hf_space(front, back)
    raise ValueError(f"unknown provider {provider!r}; use modal, replicate, trellis, or hf")


def _png_bytes(path: Path) -> bytes:
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _read_url(url: str, headers: dict | None = None) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _replicate_client():
    """Client bound to the token currently in .env, not a previously exported one."""
    from pipeline.env import load_env

    load_env()
    token = (os.environ.get("REPLICATE_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN is empty; put it in .env")
    import httpx
    import replicate

    # Default read timeout is 30s, which drops the "wait for GPU" request.
    return replicate.Client(
        api_token=token,
        timeout=httpx.Timeout(30.0, read=120.0, write=120.0, connect=30.0),
    )


def via_replicate(front: Path, back: Path, remove_background: bool = True) -> bytes:
    import time

    client = _replicate_client()

    for attempt in range(3):
        try:
            with open(front, "rb") as f_front, open(back, "rb") as f_back:
                output = client.run(
                    REPLICATE_MODEL,
                    wait=False,
                    input={
                        "front_image": f_front,
                        "back_image": f_back,
                        "file_type": "glb",
                        "remove_background": bool(remove_background),
                        "target_face_num": 80000,
                        "octree_resolution": 384,
                    },
                )
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                print("  Replicate rate-limited (429); waiting 11s to retry...", flush=True)
                time.sleep(11.0)
                continue
            raise
    return _coerce_glb(output)


def via_trellis(front: Path, back: Path) -> bytes:
    """Single-image TRELLIS.2 on Replicate. The back photo is unused."""
    _ = back
    client = _replicate_client()

    with open(front, "rb") as f_front:
        output = client.run(
            TRELLIS_MODEL,
            wait=False,
            input={
                "image": f_front,
                "preprocess_image": True,
                "generate_video": False,
                "generate_model": True,
                # Space resolution "512" is pipeline_type "512" (not the 1024 cascade).
                "pipeline_type": "512",
                "decimation_target": 100000,
                "texture_size": 2048,
            },
        )
    return _coerce_glb(output)


def _coerce_glb(output) -> bytes:
    if output is None:
        raise RuntimeError("replicate returned no output")
    if isinstance(output, dict):
        model = output.get("model_file")
        if model is None:
            raise RuntimeError(
                f"replicate output has no model_file: {sorted(output)}"
            )
        return _coerce_glb(model)
    model = getattr(output, "model_file", None)
    if model is not None:
        return _coerce_glb(model)
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "read"):
        data = output.read()
        return data if isinstance(data, bytes) else _read_url(str(data))
    url = getattr(output, "url", None) or str(output)
    if str(url).startswith("http"):
        return _read_url(str(url))
    path = Path(str(url))
    if path.is_file():
        return path.read_bytes()
    raise RuntimeError(f"could not read GLB from replicate output: {type(output)}")


def via_modal(front: Path, back: Path) -> bytes:
    import modal

    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    glb = fn.remote(_png_bytes(front), _png_bytes(back))
    if not glb:
        raise RuntimeError("Modal generate_3d_mesh returned empty bytes")
    return glb


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _hf_post(path: str, body: dict) -> dict:
    req = Request(
        f"{HF_SPACE}{path}",
        data=json.dumps(body).encode(),
        headers=_hf_headers(),
        method="POST",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HF Space POST {path} failed: {e.code} {detail}") from e


def _hf_sse(path: str, timeout: int = 180):
    req = Request(f"{HF_SPACE}{path}", headers=_hf_headers())
    with urlopen(req, timeout=timeout) as resp:
        event = None
        chunks: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                chunks.append(line[5:].lstrip())
            elif line == "":
                if event == "complete" and chunks:
                    return json.loads("\n".join(chunks))
                if event == "error":
                    raise RuntimeError("HF Space error: " + "\n".join(chunks)[:500])
                event = None
                chunks = []
    raise RuntimeError(f"HF Space SSE {path} ended without a complete event")


def _hf_call(api_name: str, data: list, session_hash: str, timeout: int = 180):
    posted = _hf_post(
        f"/gradio_api/call/{api_name}",
        {"data": data, "session_hash": session_hash},
    )
    event_id = posted.get("event_id")
    if not event_id:
        raise RuntimeError(f"HF Space {api_name} returned no event_id: {posted}")
    return _hf_sse(f"/gradio_api/call/{api_name}/{event_id}", timeout=timeout)


def _filedata(path: Path) -> dict:
    img = Image.open(path).convert("RGBA")
    img.thumbnail((1024, 1024))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "path": None,
        "url": f"data:image/png;base64,{b64}",
        "orig_name": path.name,
        "mime_type": "image/png",
        "meta": {"_type": "gradio.FileData"},
    }


def via_hf_space(front: Path, back: Path) -> bytes:
    """Single-image TRELLIS.2 Space. Uses the front photo only (back is unused)."""
    _ = back
    session = uuid.uuid4().hex
    try:
        _hf_call("start_session", [], session, timeout=30)
    except Exception:
        pass
    _hf_call(
        "image_to_3d",
        [
            _filedata(front),
            0,
            "512",
            7.5,
            0.7,
            12,
            5.0,
            7.5,
            0.5,
            12,
            3.0,
            1.0,
            0.0,
            12,
            3.0,
        ],
        session,
        timeout=180,
    )
    result = _hf_call("extract_glb", [100000, 2048], session, timeout=180)
    if isinstance(result, list):
        result = result[0]
    url = None
    if isinstance(result, dict):
        url = result.get("url") or result.get("path")
    elif isinstance(result, str) and result.startswith("http"):
        url = result
    if not url:
        raise RuntimeError(
            "TRELLIS.2 Space did not return a GLB (the public API hides session "
            "state). Download one from https://huggingface.co/spaces/microsoft/TRELLIS.2 "
            "and rerun with --provider skip --build <file.glb>."
        )
    if not str(url).startswith("http"):
        url = f"{HF_SPACE}/gradio_api/file={url}"
    return _read_url(url, headers=_hf_headers())
