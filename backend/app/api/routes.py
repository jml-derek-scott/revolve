from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..config import OUTPUT_DIR, UPLOAD_DIR
from ..services import pipeline
from ..services.base import get_device
from ..utils.files import new_job_id, save_upload

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "device": str(get_device())}


@router.post("/generate")
async def generate(
    front: UploadFile = File(..., description="Front view image"),
    back: UploadFile | None = File(None, description="Optional back view image"),
):
    job_id = new_job_id()
    job_dir = UPLOAD_DIR / job_id
    images = [await save_upload(front, job_dir, "front")]
    if back is not None and back.filename:
        images.append(await save_upload(back, job_dir, "back"))

    out_path = OUTPUT_DIR / f"{job_id}.glb"
    try:
        reconstructor, _ = await run_in_threadpool(pipeline.run, images, out_path)
    except NotImplementedError as e:
        raise HTTPException(501, str(e))

    return {
        "job_id": job_id,
        "pipeline": reconstructor.name,
        "device": str(reconstructor.device),
        "glb_url": f"/outputs/{job_id}.glb",
    }
