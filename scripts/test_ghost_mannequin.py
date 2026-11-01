"""Standalone CLI test script for the 2D Ghost Mannequin garment isolation stage.

Usage:
    poetry run python scripts/test_ghost_mannequin.py \\
        --front test_images/front-4.jpg \\
        --back test_images/back-4.jpg \\
        --out-dir outputs/ghost_test
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.preprocess import prepare_ghost_mannequin  # noqa: E402

logger = logging.getLogger(__name__)


def _resolve_photo(path_str: str, default_side: str) -> Path:
    """Resolve `test_images/front-4.jpg` to `frontend/assets/templates/4-front-full.jpg` if needed."""
    p = Path(path_str)
    if p.exists():
        return p
    if not p.is_absolute() and (ROOT / p).exists():
        return ROOT / p
    # Match patterns like front-4.jpg or back-3.jpg to frontend/assets/templates/<n>-<side>-full.jpg
    m = re.search(r"(front|back)[-_]?(\d+)", p.stem, re.IGNORECASE)
    if m:
        side, num = m.group(1).lower(), m.group(2)
        for suffix in ("-full.jpg", ".jpg", ".png"):
            cand = ROOT / "frontend" / "assets" / "templates" / f"{num}-{side}{suffix}"
            if cand.exists():
                logger.info("Resolved %s -> %s", path_str, cand)
                return cand
    fallback = ROOT / "frontend" / "assets" / "templates" / f"4-{default_side}-full.jpg"
    if fallback.exists():
        logger.info("Fallback %s -> %s", path_str, fallback)
        return fallback
    raise FileNotFoundError(f"Photo not found: {path_str}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--front",
        default="test_images/front-4.jpg",
        help="Path to front photo (default: test_images/front-4.jpg -> 4-front-full.jpg)",
    )
    parser.add_argument(
        "--back",
        default="test_images/back-4.jpg",
        help="Path to back photo (default: test_images/back-4.jpg -> 4-back-full.jpg)",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/ghost_mannequin",
        help="Output directory for masks and ghost images (default: outputs/ghost_mannequin)",
    )
    parser.add_argument(
        "--use-lama",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Replicate LaMa inpainting with OpenCV TELEA fallback (default: True)",
    )
    parser.add_argument(
        "--cloth",
        default="segformer",
        choices=("segformer", "u2net"),
        help="Clothing segmentation backend (default: segformer)",
    )
    parser.add_argument(
        "--erode-px",
        type=int,
        default=2,
        help="Boundary erosion radius in pixels to eliminate skin halos (default: 2)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    front_path = _resolve_photo(args.front, "front")
    back_path = _resolve_photo(args.back, "back")

    for side, photo_path in (("front", front_path), ("back", back_path)):
        t0 = time.perf_counter()
        print(f"[{side}] isolating ghost mannequin from {photo_path}...", flush=True)
        res = prepare_ghost_mannequin(
            photo_path,
            view=side,
            cloth=args.cloth,
            use_lama=args.use_lama,
            erode_px=args.erode_px,
        )
        raw_mask_out = out_dir / f"{side}_raw_mask.png"
        occ_mask_out = out_dir / f"{side}_occlusion_mask.png"
        ghost_out = out_dir / f"{side}_ghost.png"
        alpha_out = out_dir / f"{side}_ghost_alpha.png"

        res["raw_mask_img"].save(raw_mask_out)
        res["occlusion_mask_img"].save(occ_mask_out)
        res["ghost_rgb"].save(ghost_out)
        res["ghost_rgba"].save(alpha_out)

        dt = time.perf_counter() - t0
        print(
            f"[{side}] done in {dt:.1f}s -> {ghost_out.name}, {alpha_out.name}, "
            f"{raw_mask_out.name}, {occ_mask_out.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
