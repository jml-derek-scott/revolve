"""AI Model Pre-Pass for Texture Wrap: load photos into a 2D/multi-view model before wrapping.

Supports three AI pre-pass strategies prior to running `bake_photos`:
1. `vton`: Render the 3D mannequin's exact orthographic front/back silhouette and run
   Virtual Try-On (`cuuupid/idm-vton` on Replicate) so the garment conforms 1:1 to the
   mannequin's pose and width before UV projection.
2. `multiview`: Generate +-90 degree lateral profile views (`left` and `right`) from the
   ghost-isolated front/back garment images using Zero123++ (`jd7h/zero123plusplus` on
   Replicate, or local `venv-instantmesh` if available) and feed all 4 views (`front`,
   `back`, `left`, `right`) into `bake_photos`.
3. `ghost-warp`: Run the 2D Ghost Mannequin isolation (`prepare_ghost_mannequin`) +
   row-wise 3D silhouette registration (`warp_x_to_silhouette=True`).

Usage:
    # Rebuild Template 3 wrap with Ghost Pre-Pass + Silhouette Warp:
    poetry run python scripts/test_wrap_prepass.py \\
        --front frontend/assets/templates/3-front-full.jpg \\
        --back  frontend/assets/templates/3-back-full.jpg \\
        --mode ghost-warp \\
        --out frontend/assets/models/3-wrap.glb

    # Run VTON pre-pass on the canonical mannequin silhouette first:
    poetry run python scripts/test_wrap_prepass.py \\
        --front frontend/assets/templates/3-front-full.jpg \\
        --back  frontend/assets/templates/3-back-full.jpg \\
        --mode vton \\
        --out outputs/3-wrap-vton.glb
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.export_glb import export_mesh  # noqa: E402
from pipeline.garment_extractor import align_to_mannequin, load_mesh  # noqa: E402
from pipeline.preprocess import load_view, prepare_ghost_mannequin  # noqa: E402
from pipeline.reconstruct import _read_url, _replicate_client  # noqa: E402
from pipeline.texture_bake import MANNEQUIN_RGB, bake_photos  # noqa: E402
from pipeline.upscale import upscale_image  # noqa: E402

IDM_VTON_MODEL = (
    "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4"
)
ZERO123_REPLICATE_MODEL = (
    "jd7h/zero123plusplus:c69c6559a29011b576f1f0371b3bc1fb285c880452f8c0a8b68e91068c8f2062"
)


def render_mannequin_silhouette(
    mannequin: trimesh.Trimesh,
    side: str = "front",
    width: int = 768,
    height: int = 1024,
) -> Image.Image:
    """Render a clean shaded 2D orthographic image of `mannequin` from `front` (+Z) or `back` (-Z)."""
    sign = 1.0 if side == "front" else -1.0
    mlo, mhi = mannequin.bounds
    m_height = float(mhi[1] - mlo[1])
    mid_x = float(0.5 * (mlo[0] + mhi[0]))

    # Margin so head/feet match typical studio framing (top=8%, bottom=92%)
    pad_top = int(0.08 * height)
    pad_bot = int(0.92 * height)
    scale = (pad_bot - pad_top) / max(m_height, 1e-6)
    cx = width * 0.5

    pts, face_idx = trimesh.sample.sample_surface(mannequin, 2_500_000)
    fn = mannequin.face_normals[face_idx].copy()
    # Orient normal outward from vertical axis for consistent shading on double-skinned mannequin
    rad = np.stack([pts[:, 0] - mid_x, np.zeros(len(pts)), pts[:, 2]], axis=-1)
    flip = np.sum(fn * rad, axis=-1) < 0
    fn[flip] = -fn[flip]

    px = np.round(cx + sign * (pts[:, 0] - mid_x) * scale).astype(int)
    py = np.round(pad_bot - (pts[:, 1] - mlo[1]) * scale).astype(int)
    ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, pts, fn = px[ok], py[ok], pts[ok], fn[ok]

    zbuf = np.full((height, width), -np.inf, dtype=np.float32)
    z_val = sign * pts[:, 2]
    np.maximum.at(zbuf, (py, px), z_val)

    vis = z_val >= (zbuf[py, px] - 0.004)
    px, py, fn = px[vis], py[vis], fn[vis]

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    light_dir = np.array([0.15 * sign, 0.25, 0.95 * sign], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    ndl = np.clip(np.sum(fn * light_dir[None, :], axis=-1), 0.0, 1.0)
    shade = (0.55 + 0.42 * ndl)[:, None] * np.array(MANNEQUIN_RGB, dtype=np.float32)[None, :]
    canvas[py, px] = np.clip(shade, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def _img_buf(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    buf.name = "image.png"
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def run_vton_prepass(
    mannequin: trimesh.Trimesh,
    ghost_res: dict,
    side: str,
    description: str,
    out_dir: Path,
) -> Path:
    """Dress the rendered 3D mannequin silhouette with the isolated ghost garment via IDM-VTON."""
    mannequin_img = render_mannequin_silhouette(mannequin, side=side)
    m_path = out_dir / f"mannequin_{side}.png"
    mannequin_img.save(m_path)

    import httpx
    import replicate
    import os
    from pipeline.env import load_env

    load_env()
    token = (os.environ.get("REPLICATE_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN is empty; put it in .env")
    client = replicate.Client(
        api_token=token,
        timeout=httpx.Timeout(30.0, read=300.0, write=120.0, connect=30.0),
    )
    print(f"  [vton] dressing mannequin ({side}) via Replicate {IDM_VTON_MODEL.split(':')[0]}...", flush=True)
    output = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            output = client.run(
                IDM_VTON_MODEL,
                input={
                    "human_img": _img_buf(mannequin_img),
                    "garm_img": _img_buf(ghost_res["ghost_rgb"]),
                    "garment_des": description,
                    "category": "dresses",
                    "crop": False,
                    "steps": 30,
                },
            )
            break
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) and attempt < 2:
                print("  [vton] rate-limited (429); waiting 10s to retry...", flush=True)
                time.sleep(10.5)
                continue
            raise
    if output is None:
        raise last_exc or RuntimeError("VTON returned nothing")
    url = getattr(output, "url", None) or str(output)
    data = output.read() if hasattr(output, "read") else _read_url(str(url))
    out_path = out_dir / f"vton_{side}.png"
    out_path.write_bytes(data)
    return out_path


def run_multiview_side_prepass(
    ghost_front: Image.Image,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Generate 6 Zero123++ views on Replicate and extract the +-90 degree lateral views."""
    client = _replicate_client()
    print("  [multiview] synthesizing 6 views via Zero123++ on Replicate...", flush=True)
    output = client.run(
        ZERO123_REPLICATE_MODEL,
        input={
            "image": _img_buf(ghost_front),
            "remove_background": False,
        },
    )
    if isinstance(output, (list, tuple)):
        output = output[0]
    url = getattr(output, "url", None) or str(output)
    data = output.read() if hasattr(output, "read") else _read_url(str(url))
    grid = Image.open(io.BytesIO(data)).convert("RGB")
    grid_path = out_dir / "zero123_grid.png"
    grid.save(grid_path)

    # Zero123++ v1.2 2x3 grid: (row 0, col 1) is azimuth 90°, (row 2, col 0) is azimuth 270°
    gw, gh = grid.size
    vw, vh = gw // 2, gh // 3
    left_img = grid.crop((vw, 0, 2 * vw, vh))
    right_img = grid.crop((0, 2 * vh, vw, 3 * vh))
    left_path = out_dir / "side_left_90.png"
    right_path = out_dir / "side_right_270.png"
    left_img.save(left_path)
    right_img.save(right_path)
    return left_path, right_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument(
        "--mode",
        default="ghost-warp",
        choices=("ghost-warp", "vton", "multiview"),
        help="Pre-pass mode: ghost-warp (local SegFormer+LaMa+silhouette warp), vton (IDM-VTON on mannequin), or multiview (Zero123++ side views).",
    )
    parser.add_argument("--mannequin", default=str(ROOT / "mannequins" / "female_base_mesh.glb"))
    parser.add_argument("--mesh", help="Optional 3D mesh to wrap onto (e.g. outputs/3-build-painted.glb)")
    parser.add_argument("--keep-mesh-texture", action="store_true")
    parser.add_argument("--unwrap-mode", default=None, choices=("auto", "planar", "cylindrical"))
    parser.add_argument("--description", default="sleeveless mini dress", help="Garment description for VTON")
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="Real-ESRGAN scale for front/back photos before ghost/VTON. 1 = off.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    work_dir = ROOT / "outputs" / "wrap_prepass" / out_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    mannequin = load_mesh(args.mannequin)
    front_path = Path(args.front)
    back_path = Path(args.back)
    if args.upscale and args.upscale > 1:
        t0 = time.perf_counter()
        front_path = upscale_image(front_path, scale=args.upscale)
        back_path = upscale_image(back_path, scale=args.upscale)
        print(f"upscale: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    front_ghost = prepare_ghost_mannequin(front_path, view="front", cloth="segformer", use_lama=True)
    back_ghost = prepare_ghost_mannequin(back_path, view="back", cloth="segformer", use_lama=True)
    front_ghost["ghost_rgb"].save(work_dir / "front_ghost.png")
    back_ghost["ghost_rgb"].save(work_dir / "back_ghost.png")
    print(f"ghost prepass: {time.perf_counter() - t0:.1f}s", flush=True)

    views = {"front": front_ghost, "back": back_ghost}

    if args.mode == "vton":
        try:
            vton_f = run_vton_prepass(mannequin, front_ghost, "front", args.description, work_dir)
            vton_b = run_vton_prepass(mannequin, back_ghost, "back", args.description, work_dir)
            views = {
                "front": load_view(vton_f, cloth="segformer"),
                "back": load_view(vton_b, cloth="segformer"),
            }
        except Exception as exc:
            print(f"  [vton] failed ({exc}); wrapping ghost photos instead", flush=True)
    elif args.mode == "multiview":
        left_p, right_p = run_multiview_side_prepass(front_ghost["ghost_rgb"], work_dir)
        views["left"] = load_view(left_p, cloth="u2net")
        views["right"] = load_view(right_p, cloth="u2net")

    if args.mesh:
        mesh = load_mesh(args.mesh)
        mesh = align_to_mannequin(mesh, mannequin, mesh.bounds)
    else:
        mesh = mannequin

    unwrap_mode = (
        args.unwrap_mode
        if args.unwrap_mode is not None
        else ("planar" if (args.mesh and not args.keep_mesh_texture) else "auto")
    )

    t0 = time.perf_counter()
    mesh = bake_photos(
        mesh,
        views,
        mask_key="garment",
        inpaint_unpainted=False,
        fallback_rgb=MANNEQUIN_RGB,
        unwrap_mode=unwrap_mode,
        align_to=mannequin if args.mesh else None,
        warp_x_to_silhouette=True,
        keep_mesh_texture=args.keep_mesh_texture,
    )
    print(f"bake_photos: {time.perf_counter() - t0:.1f}s", flush=True)

    export_mesh(mesh, out_path)
    tex = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    if tex is not None:
        tex.save(work_dir / f"{out_path.stem}_atlas.png")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
