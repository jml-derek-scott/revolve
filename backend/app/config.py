import os
from pathlib import Path

# Must be set before torch is imported anywhere so unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUT_DIR = ROOT_DIR / "outputs"
FRONTEND_DIR = ROOT_DIR / "frontend"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}

for _d in (UPLOAD_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

VENDOR_DIR = ROOT_DIR / "vendor"
TRIPOSR_DIR = VENDOR_DIR / "TripoSR"
TRIPOSR_MC_RESOLUTION = int(os.environ.get("TRIPOSR_MC_RESOLUTION", "256"))
