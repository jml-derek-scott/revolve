"""Modal A10G worker: Hunyuan3D-2mv shape, plus Hunyuan3D-Paint on an existing mesh.

Deploy:
    make modal-secret    # once, from HF_TOKEN in .env
    make modal-deploy    # first image build can take 20–40 min
    make modal-prefetch  # optional, warms the HF volume

Shape:
    poetry run modal run cloud/modal_app.py --front path/front.png --back path/back.png

Paint an existing GLB (keeps that shape; front + back photos condition the texture):
    poetry run modal run cloud/modal_app.py --front path/front.png --back path/back.png \\
        --mesh outputs/3-build-hunyuan.glb --out outputs/3-build-painted.glb
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

APP_NAME = "garment-3d-reconstruction"
HF_CACHE = "/root/.cache/huggingface"
REPO = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
MODEL_ID = "tencent/Hunyuan3D-2mv"
SUBFOLDER = "hunyuan3d-dit-v2-mv"

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
        "ninja-build",
        "cmake",
    )
    .run_commands(
        "pip install --upgrade pip && "
        "pip install torch==2.4.1 torchvision==0.19.1 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "numpy<2.0.0",
        "Pillow",
        "trimesh",
        "huggingface_hub",
        "transformers>=4.46.0,<5.0",
        "diffusers>=0.30.0,<0.35.0",
        "einops",
        "omegaconf",
        "opencv-python-headless",
        "tqdm",
        "accelerate",
        "safetensors",
        "ninja",
        "pybind11",
        "scikit-image",
        "scipy",
        "einops",
        "xatlas",
        "rembg",
    )
    .run_commands(
        f"git clone --depth 1 {REPO} /opt/Hunyuan3D-2 && "
        "pip install --no-deps -e /opt/Hunyuan3D-2"
    )
    .run_commands(
        # Paint's CUDA rasterizer. A10G is sm_86. Modal's image sets CXX=clang++,
        # which is not installed; the CUDA devel image has g++.
        "cd /opt/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && "
        "CXX=g++ CC=gcc TORCH_CUDA_ARCH_LIST=8.6 python3 setup.py install && "
        "cd /opt/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && "
        "CXX=g++ CC=gcc TORCH_CUDA_ARCH_LIST=8.6 python3 setup.py install"
    )
    .run_commands("pip install pygltflib")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_CACHE": HF_CACHE,
            "PYTHONPATH": "/opt/Hunyuan3D-2",
        }
    )
)

_fn_kwargs = dict(
    image=image,
    volumes={HF_CACHE: hf_cache},
    secrets=[hf_secret],
)


@app.function(**_fn_kwargs, timeout=30 * 60)
def prefetch():
    """Download Hunyuan3D-2mv (+ VAE repo) into the persistent HF volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    snapshot_download("tencent/Hunyuan3D-2")
    print("prefetch done")


_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    # Import the pipeline module directly so we skip postprocessors (pymeshlab).
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    kwargs = dict(subfolder=SUBFOLDER, use_safetensors=True, device="cuda")
    try:
        _pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID, variant="fp16", **kwargs
        )
    except Exception:
        _pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID, **kwargs)
    from hy3dgen.shapegen.preprocessors import MVImageProcessorV2

    _pipe.image_processor = MVImageProcessorV2(size=512)
    return _pipe


@app.function(
    **_fn_kwargs,
    gpu="A10G",
    timeout=10 * 60,
    scaledown_window=60,
    max_containers=1,
)
def generate_3d_mesh(front_bytes: bytes, back_bytes: bytes) -> bytes:
    """Shape-only multi-view reconstruction. No Hunyuan Paint / CUDA rasterizer."""
    from PIL import Image

    front = Image.open(io.BytesIO(front_bytes)).convert("RGBA")
    back = Image.open(io.BytesIO(back_bytes)).convert("RGBA")
    pipe = _get_pipe()
    mesh = pipe(
        image={"front": front, "back": back},
        num_inference_steps=30,
        octree_resolution=384,
        num_chunks=20000,
        output_type="trimesh",
        mc_algo="mc",
    )[0]
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue()


_paint = None


def _get_paint():
    global _paint
    if _paint is not None:
        return _paint
    # Paint turbo still publishes a .bin. transformers>=4.57 blocks torch.load
    # on torch<2.6; the rasterizer is built against 2.4.1, so allow this load.
    # The file is the tencent/Hunyuan3D-2 checkpoint, not an upload.
    import transformers.modeling_utils as modeling_utils
    from transformers.utils import import_utils

    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

    _paint = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-paint-v2-0-turbo",
    )
    # 1024 fits an A10G with delight + turbo paint loaded together.
    _paint.config.render_size = 1024
    _paint.config.texture_size = 1024
    _paint.render = MeshRender(default_resolution=1024, texture_size=1024)
    return _paint


@app.function(
    **_fn_kwargs,
    gpu="A10G",
    timeout=25 * 60,
    scaledown_window=120,
    max_containers=1,
)
def paint_3d_mesh(glb_bytes: bytes, front_bytes: bytes, back_bytes: bytes) -> bytes:
    """Texture an existing mesh with Hunyuan3D-Paint turbo (front + back photos)."""
    import trimesh
    from PIL import Image

    mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh")
    images = [Image.open(io.BytesIO(raw)).convert("RGBA") for raw in (front_bytes, back_bytes)]
    textured = _get_paint()(mesh, image=images)
    buf = io.BytesIO()
    textured.export(buf, file_type="glb")
    return buf.getvalue()


def _cutout_png(path: str) -> bytes:
    """Local person cut-out. The Modal image does not ship onnxruntime."""
    from PIL import Image
    from rembg import remove

    image = Image.open(path).convert("RGB")
    cut = remove(image)
    buf = io.BytesIO()
    cut.save(buf, format="PNG")
    return buf.getvalue()


@app.local_entrypoint()
def main(
    front: str,
    back: str,
    out: str = "outputs/modal_test.glb",
    mesh: str = "",
):
    if mesh:
        glb = paint_3d_mesh.remote(
            Path(mesh).read_bytes(),
            _cutout_png(front),
            _cutout_png(back),
        )
    else:
        glb = generate_3d_mesh.remote(Path(front).read_bytes(), Path(back).read_bytes())
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(glb)
    print(f"wrote {dest} ({len(glb)} bytes)")
