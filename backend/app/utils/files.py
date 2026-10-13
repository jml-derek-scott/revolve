import io
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..config import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES


async def save_upload(file: UploadFile, dest_dir: Path, stem: str) -> Path:
    """Validate an uploaded image and save it as PNG under dest_dir."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"{stem}: unsupported type {file.content_type!r}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{stem}: file exceeds {MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{stem}: not a valid image")
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{stem}.png"
    img.convert("RGBA").save(path)
    return path


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]
