"""Standalone TripoSR smoke test: poetry run python scripts/test_triposr.py <image> [mps|cpu]"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "TripoSR"))

import numpy as np
import torch
from PIL import Image
from tsr.system import TSR

image_path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else "mps"

t0 = time.perf_counter()
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192)
model.to(device)
print(f"load: {time.perf_counter() - t0:.1f}s")

img = Image.open(image_path).convert("RGBA")
arr = np.array(img).astype(np.float32) / 255.0
arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5  # composite on grey
image = Image.fromarray((arr * 255).astype(np.uint8))

t0 = time.perf_counter()
with torch.no_grad():
    codes = model([image], device=device)
print(f"encode: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
from torchmcubes import marching_cubes
model.set_marching_cubes_resolution(128)
model.isosurface_helper.mc_func = lambda level, thr: marching_cubes(level.cpu(), thr)  # MC is CPU-only here
meshes = model.extract_mesh(codes, True, resolution=128)
print(f"extract_mesh: {time.perf_counter() - t0:.1f}s")
out = ROOT / "outputs" / "triposr_test.glb"
meshes[0].export(out)
print(f"wrote {out} ({len(meshes[0].vertices)} verts)")
