"""Standalone InstantMesh smoke test on MPS/CPU, independent of the backend.

    venv-instantmesh/bin/python scripts/test_instantmesh.py <front> [--back <back>] [--device mps|cpu]
        [--diffusion-device cpu|mps]

Stage 1: Zero123++ generates 6 views from the front image (azimuths 30..330, never 0/180).
Stage 2: the InstantMesh LRM reconstructs a mesh from those posed views.
With --back, each --modes entry reconstructs from the same generated grid (see --help),
with real photos posed at azimuth 0 / 180, elevation 0. The LRM was trained on the 6 fixed Zero123++ poses, so this is an experiment.

Runs in its own venv (venv-instantmesh/) so the TripoSR environment is untouched.
First run downloads ~4 GB of weights into the Hugging Face cache.
"""
import argparse
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
IM_DIR = ROOT / "vendor" / "InstantMesh"
sys.path.insert(0, str(IM_DIR))

# nvdiffrast is CUDA-only and imported at module level all over InstantMesh, but it is only
# used for rendering / texture baking. Vertex-coloured mesh extraction never touches it.
_dr = types.ModuleType("nvdiffrast.torch")
_dr.RasterizeCudaContext = lambda device=None: None
sys.modules["nvdiffrast"] = types.ModuleType("nvdiffrast")
sys.modules["nvdiffrast.torch"] = _dr

import numpy as np
import rembg
import torch
import trimesh
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2

from src.utils.camera_util import FOV_to_intrinsics, spherical_camera_pose
from src.utils.infer_util import remove_background, resize_foreground
from src.utils.train_util import instantiate_from_config

p = argparse.ArgumentParser()
p.add_argument("front")
p.add_argument("--back")
p.add_argument(
    "--modes",
    default="front,add,replace",
    help="with --back: front (6 generated), add (+ real 0/180 = 8), "
    "replace (real back replaces generated 150/210 = 6)",
)
p.add_argument("--device", default="mps", help="device for the LRM")
p.add_argument(
    "--diffusion-device", default="cpu", help="Zero123++ OOMs on a ~10 GB MPS budget"
)
p.add_argument("--steps", type=int, default=75)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
device = torch.device(args.device)
out_dir = ROOT / "outputs" / "instantmesh_test"
out_dir.mkdir(parents=True, exist_ok=True)


def timed(label, t0):
    print(f"{label}: {time.perf_counter() - t0:.1f}s", flush=True)


def cameras(azimuths, elevations, radius=4.0, fov=30.0):
    """Same encoding as src.utils.camera_util.get_zero123plus_input_cameras, any poses."""
    c2ws = spherical_camera_pose(np.array(azimuths, float), np.array(elevations, float), radius)
    c2ws = c2ws.float().flatten(-2)
    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(len(azimuths), 1, 1).float().flatten(-2)
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    return torch.cat([c2ws[:, :12], intrinsics], dim=-1).unsqueeze(0)


rembg_session = rembg.new_session()


def cutout(path):
    return resize_foreground(remove_background(Image.open(path), rembg_session), 0.85)


# ---- Stage 1: Zero123++ (float32: fp16 is unreliable on MPS) ----
t0 = time.perf_counter()
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline=str(IM_DIR / "zero123plus"),
    torch_dtype=torch.float32,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)
unet_ckpt = hf_hub_download("TencentARC/InstantMesh", "diffusion_pytorch_model.bin")
pipeline.unet.load_state_dict(torch.load(unet_ckpt, map_location="cpu"), strict=True)
pipeline = pipeline.to(args.diffusion_device)
timed("load zero123++", t0)

front = cutout(args.front)
t0 = time.perf_counter()
grid = pipeline(
    front,
    num_inference_steps=args.steps,
    generator=torch.Generator(device="cpu").manual_seed(args.seed),
).images[0]
timed(f"zero123++ ({args.steps} steps, {args.diffusion_device})", t0)
grid.save(out_dir / f"{Path(args.front).stem}_views.png")
del pipeline
if args.diffusion_device == "mps":
    torch.mps.empty_cache()

grid_views = torch.from_numpy(np.asarray(grid, dtype=np.float32) / 255.0).permute(2, 0, 1)
grid_views = rearrange(grid_views, "c (n h) (m w) -> (n m) c h w", n=3, m=2)  # (6, 3, 320, 320)
GRID_AZ = [30, 90, 150, 210, 270, 330]
GRID_EL = [20, -10, 20, -10, 20, -10]


def to_view(img):
    a = np.asarray(img.convert("RGBA").resize((320, 320), Image.LANCZOS), np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4])  # white background, like the grid
    return torch.from_numpy(rgb).permute(2, 0, 1)


# mode -> (views, azimuths, elevations)
inputs = {}
modes = args.modes.split(",") if args.back else ["front"]
if args.back:
    real_front, real_back = to_view(front), to_view(cutout(args.back))
for mode in modes:
    if mode == "front":
        inputs[mode] = (grid_views, GRID_AZ, GRID_EL)
    elif mode == "add":
        inputs[mode] = (
            torch.cat([grid_views, torch.stack([real_front, real_back])]),
            GRID_AZ + [0, 180],
            GRID_EL + [0, 0],
        )
    elif mode == "replace":
        keep = [0, 1, 4, 5]  # drop the back-facing 150 / 210 views
        inputs[mode] = (
            torch.cat([grid_views[keep], torch.stack([real_front, real_back])]),
            [GRID_AZ[i] for i in keep] + [0, 180],
            [GRID_EL[i] for i in keep] + [0, 0],
        )
    else:
        raise SystemExit(f"unknown mode {mode}")

# ---- Stage 2: InstantMesh LRM ----
t0 = time.perf_counter()
config = OmegaConf.load(IM_DIR / "configs" / "instant-mesh-large.yaml")
model = instantiate_from_config(config.model_config)
ckpt = hf_hub_download("TencentARC/InstantMesh", "instant_mesh_large.ckpt")
state = torch.load(ckpt, map_location="cpu")["state_dict"]
model.load_state_dict({k[14:]: v for k, v in state.items() if k.startswith("lrm_generator.")})
model = model.to(device).eval()
timed("load lrm", t0)

planes = {}
with torch.no_grad():
    for mode, (views, az, el) in inputs.items():
        t0 = time.perf_counter()
        images = v2.functional.resize(views.unsqueeze(0).to(device), 320, interpolation=3, antialias=True)
        planes[mode] = model.forward_planes(images.clamp(0, 1), cameras(az, el).to(device)).cpu()
        timed(f"forward_planes [{mode}] ({len(az)} views)", t0)

    # FlexiCubes extraction at grid_res 128 needs a single ~15 GB buffer: too big for MPS.
    cpu = torch.device("cpu")
    model = model.to(cpu)
    model.init_flexicubes_geometry(cpu, fovy=30.0)
    for mode, pl in planes.items():
        t0 = time.perf_counter()
        vertices, faces, colors = model.extract_mesh(pl, use_texture_map=False)
        timed(f"extract_mesh [{mode}]", t0)
        # Raw mesh is z-up facing +x (same as TripoSR); [1, 2, 0] gives glTF y-up facing +z.
        mesh = trimesh.Trimesh(vertices[:, [1, 2, 0]], faces, vertex_colors=colors, process=False)
        out = out_dir / f"{Path(args.front).stem}_{mode}.glb"
        mesh.export(out)
        print(f"wrote {out} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)", flush=True)
