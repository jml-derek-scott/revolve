"""Prepass routing. No Replicate, Modal, or SegFormer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from pipeline.outfit import GarmentSpec, layer_fits, load_defaults, load_outfit
from pipeline.prepass import paste_graphic, plan_prepass
from pipeline.sheer_prepass import GRAY, prepare_sheer_input

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "configs" / "templates"


def test_opaque_is_ghost_v1_without_kontext():
    outfit = load_outfit(TEMPLATES / "hoodie_jeans_layered.json")
    plans = {garment.class_id: plan_prepass(garment) for garment in outfit.garments}
    assert plans["jeans"].mode == "ghost_v1"
    assert plans["jeans"].use_kontext is False
    assert plans["jeans"].vision_plate is False
    assert plans["hoodie"].mode == "ghost_v1"
    assert plans["hoodie"].use_kontext is False
    hoodie = next(garment for garment in outfit.garments if garment.class_id == "hoodie")
    assert hoodie.preserve_graphics is True


def test_guess_dress_is_opaque_ghost_v2():
    outfit = load_outfit(TEMPLATES / "guess_dress.json")
    garment = outfit.garments[0]
    plan = plan_prepass(garment)
    assert garment.material_type == "opaque"
    assert garment.prepass_mode == "ghost_v2"
    assert plan.mode == "ghost_v2"
    assert plan.use_kontext is True
    assert plan.vision_plate is False
    assert "front" in garment.prompts and "back" in garment.prompts
    assert garment.background_max == 100
    assert garment.lift_m == pytest.approx(0.12)
    assert garment.empty_prompt
    assert outfit.target_mannequin.endswith("female_base_mesh.glb")


def test_sheer_is_ghost_v2_with_vision_plate():
    outfit = load_outfit(TEMPLATES / "dress_slip.json")
    plan = plan_prepass(outfit.garments[0])
    assert plan.mode == "ghost_v2"
    assert plan.use_kontext is True
    assert plan.vision_plate is True


def test_open_leg_reads_template2_plates():
    """Shop Open Leg is template 2. outfit_id 2 is outputs/canonical/2/dress/."""
    outfit = load_outfit(TEMPLATES / "open_leg.json")
    garment = outfit.garments[0]
    plan = plan_prepass(garment)
    assert outfit.outfit_id == "2"
    assert garment.class_id == "dress"
    assert garment.material_type == "sheer"
    assert garment.prepass_mode == "ghost_v2"
    assert plan.vision_plate is True
    assert garment.lift_m == pytest.approx(0.10)
    assert garment.unwrap_mode == "cylindrical"
    assert garment.background_max == 100
    assert garment.prompts is None
    assert garment.empty_prompt is None
    assert outfit.target_mannequin.endswith("female_base_mesh.glb")


def test_preserve_graphics_refuses_ghost_v2():
    with pytest.raises(ValidationError, match="refuses ghost_v2"):
        GarmentSpec(
            class_id="hoodie",
            category="top",
            prepass_mode="ghost_v2",
            preserve_graphics=True,
            graphic_region_bbox=(0.35, 0.25, 0.65, 0.55),
        )


def test_layered_hoodie_clearance_is_body_and_jeans():
    outfit = load_outfit(TEMPLATES / "hoodie_jeans_layered.json")
    fits = {fit.class_id: fit for fit in layer_fits(outfit)}
    assert fits["jeans"].clearance_m == pytest.approx(0.004)
    assert fits["jeans"].obstacle_class_id is None
    assert fits["hoodie"].clearance_m == pytest.approx(0.004)
    assert fits["hoodie"].obstacle_class_id == "jeans"
    assert fits["hoodie"].obstacle_clearance_m == pytest.approx(0.004)


def test_defaults_match_sheer_fill():
    defaults = load_defaults()
    assert defaults.clearance_mm == 4
    assert defaults.background_rgb == (255, 255, 255)
    assert defaults.sheer_fill_rgb == GRAY
    assert defaults.dress_unwrap == "cylindrical"


def test_sheer_vision_plate_is_dark_gray(tmp_path: Path):
    rgb = np.full((64, 64, 3), 255, np.uint8)
    rgb[22:42, 22:42] = (120, 20, 20)
    src = tmp_path / "front.png"
    Image.fromarray(rgb).save(src)
    out = prepare_sheer_input(src, tmp_path / "front-recon.png")
    filled = np.asarray(Image.open(out).convert("RGB"))
    cloth = np.all(rgb < 250, axis=2)
    assert cloth.any()
    assert np.all(filled[cloth] == np.array(GRAY, dtype=np.uint8))
    assert tuple(filled[0, 0]) == (255, 255, 255)


def test_product_view_ignores_offwhite_corner(tmp_path: Path):
    """A warm JPEG corner just under the white cut must not shift the dress."""
    rgb = np.full((280, 280, 3), 255, np.uint8)
    rgb[40:160, 40:120] = (8, 8, 8)
    rgb[200:240, 200:240] = (246, 244, 241)
    path = tmp_path / "front.png"
    Image.fromarray(rgb).save(path)
    from pipeline.canonical import load_product_view

    view = load_product_view(
        path,
        {
            "width": 400,
            "height": 400,
            "top": 0,
            "bottom": 399,
            "cx": 200,
            "g_top": 10,
            "g_bottom": 250,
        },
    )
    ys, xs = np.nonzero(view["garment"])
    aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert aspect == pytest.approx(80 / 120, abs=0.05)
    assert view["rgb"][ys, xs].mean() < 40


def test_paste_graphic_keeps_source_pixels():
    source = Image.new("RGB", (20, 20), (255, 255, 255))
    for x in range(8, 14):
        for y in range(4, 10):
            source.putpixel((x, y), (10, 20, 30))
    plate = Image.new("RGB", (20, 20), (255, 255, 255))
    pasted = paste_graphic(plate, source, (0.4, 0.2, 0.7, 0.5))
    assert pasted.getpixel((8, 4)) == (10, 20, 30)
    assert pasted.getpixel((0, 0)) == (255, 255, 255)
