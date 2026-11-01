"""Ideal garment images, before any 3D reconstruction.

Template 5 (hoodie and jeans), cut from the real photos. Hands are filled;
the print is not redrawn:

    poetry run python scripts/make_canonical_views.py

Template 1 dress. Front and back are edits of the catalog photos. Left and
right are profiles of the accepted front, for texture only:

    poetry run python scripts/make_canonical_views.py --dress

Look at the PNGs before sending a front/back pair to Hunyuan.

Outfit config (ghost v1 / v2 from the garment material):

    poetry run python scripts/make_canonical_views.py \\
      --config configs/templates/hoodie_jeans_layered.json \\
      --front path/front.jpg --back path/back.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.env import load_env  # noqa: E402

load_env()

from pipeline.canonical import (  # noqa: E402
    T2_BACK_FROM_FRONT,
    T2_EMPTY,
    T2_PROMPTS,
    strip_body,
    write_dress_views,
    write_outfit_plates,
)
from pipeline.outfit import load_outfit  # noqa: E402
from pipeline.prepass import write_prepass  # noqa: E402

FRONT = ROOT / "frontend/assets/templates/5-front-full.jpg"
BACK = ROOT / "frontend/assets/templates/5-back-full.jpg"
OUT = ROOT / "outputs/canonical/5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        help="Outfit JSON or YAML. Writes outputs/canonical/<outfit_id>/<class_id>/.",
    )
    parser.add_argument(
        "--out-root",
        default=str(ROOT / "outputs" / "canonical"),
        help="With --config, parent of <outfit_id>/. Ignored by --dress and template 5.",
    )
    parser.add_argument(
        "--dress",
        action="store_true",
        help="Template 1 leopard lace dress from test_images/front.jpg and back.jpg",
    )
    parser.add_argument(
        "--template",
        type=int,
        choices=(2,),
        help="Template 2 black midi. Uses the catalog full photos and T2_PROMPTS.",
    )
    parser.add_argument("--front", default=str(ROOT / "test_images" / "front.jpg"))
    parser.add_argument("--back", default=str(ROOT / "test_images" / "back.jpg"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "canonical" / "1" / "dress"))
    parser.add_argument(
        "--views",
        default="front,back,left,right",
        help="Comma-separated views to generate. Profiles need an existing front.png.",
    )
    parser.add_argument(
        "--strip-body",
        action="store_true",
        help="Template 2 only. Edit the existing front and back to remove a leftover body, then rebuild left and right.",
    )
    args = parser.parse_args()
    if args.config:
        outfit = load_outfit(args.config)
        write_prepass(outfit, Path(args.front), Path(args.back), Path(args.out_root))
        return
    if args.template == 2:
        views = tuple(part.strip() for part in args.views.split(",") if part.strip())
        out = ROOT / "outputs/canonical/2/dress"
        if args.strip_body:
            for side in ("front", "back"):
                strip_body(out / f"{side}.png", T2_EMPTY)
            write_dress_views(
                ROOT / "frontend/assets/templates/2-front-full.jpg",
                ROOT / "frontend/assets/templates/2-back-full.jpg",
                out,
                views=tuple(side for side in views if side in ("left", "right")) or ("left", "right"),
                prompts=T2_PROMPTS,
                label="template 2",
            )
            return
        write_dress_views(
            ROOT / "frontend/assets/templates/2-front-full.jpg",
            ROOT / "frontend/assets/templates/2-back-full.jpg",
            ROOT / "outputs/canonical/2/dress",
            views=views,
            prompts=T2_PROMPTS,
            label="template 2",
            back_from_front=T2_BACK_FROM_FRONT,
        )
        return
    if args.dress:
        views = tuple(part.strip() for part in args.views.split(",") if part.strip())
        write_dress_views(Path(args.front), Path(args.back), Path(args.out), views=views)
        return
    write_outfit_plates(FRONT, BACK, OUT)


if __name__ == "__main__":
    main()
