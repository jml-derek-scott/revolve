"""Choose a 2D prepass from an outfit config and write the plates.

Ghost v1 keeps catalog pixels. Ghost v2 rewrites a sheer garment with the
existing Flux prompts. Sheer vision plates are a second pair for Hunyuan only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.outfit import GarmentSpec, OutfitSpec, repo_path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PrepassPlan:
    mode: str
    use_kontext: bool
    vision_plate: bool


def plan_prepass(garment: GarmentSpec) -> PrepassPlan:
    """Opaque stays ghost v1. Sheer is a Kontext rewrite plus a gray vision plate."""
    if garment.preserve_graphics and garment.prepass_mode == "ghost_v2":
        raise ValueError(f"{garment.class_id}: preserve_graphics refuses ghost_v2")
    if garment.material_type == "sheer":
        if garment.preserve_graphics:
            raise ValueError(f"{garment.class_id}: preserve_graphics refuses a sheer rewrite")
        return PrepassPlan(mode="ghost_v2", use_kontext=True, vision_plate=True)
    if garment.prepass_mode == "ghost_v2":
        return PrepassPlan(mode="ghost_v2", use_kontext=True, vision_plate=False)
    return PrepassPlan(mode="ghost_v1", use_kontext=False, vision_plate=False)


def paste_graphic(
    plate: Image.Image,
    source: Image.Image,
    bbox: tuple[float, float, float, float],
) -> Image.Image:
    """Copy a normalized source box onto the plate. Origin is the top left."""
    x0, y0, x1, y1 = bbox
    sw, sh = source.size
    src_box = (
        int(round(x0 * sw)),
        int(round(y0 * sh)),
        int(round(x1 * sw)),
        int(round(y1 * sh)),
    )
    crop = source.crop(src_box)
    pw, ph = plate.size
    dst = (
        int(round(x0 * pw)),
        int(round(y0 * ph)),
        int(round(x1 * pw)),
        int(round(y1 * ph)),
    )
    width, height = dst[2] - dst[0], dst[3] - dst[1]
    if width > 0 and height > 0 and crop.size != (width, height):
        crop = crop.resize((width, height), Image.Resampling.LANCZOS)
    out = plate.copy()
    if width > 0 and height > 0:
        out.paste(crop, (dst[0], dst[1]))
    return out


def plate_dir(outfit: OutfitSpec, garment: GarmentSpec, out_root: Path | None = None) -> Path:
    root = Path(out_root) if out_root is not None else ROOT / "outputs" / "canonical"
    return root / outfit.outfit_id / garment.class_id


def recon_plate_paths(directory: Path) -> tuple[Path, Path]:
    """Sheer vision plates when they exist, otherwise the texture plates."""

    def pick(side: str) -> Path:
        recon = directory / f"{side}-recon.png"
        if recon.exists():
            return recon
        plain = directory / f"{side}.png"
        if plain.exists():
            return plain
        raise FileNotFoundError(f"missing {side} plate in {directory}")

    return pick("front"), pick("back")


def texture_view(path: Path, span: dict | None = None) -> dict:
    """Catalog-framed plate. Optional span overrides the person and garment rows."""
    image = Image.open(path).convert("RGB")
    rgb = np.asarray(image)
    garment = np.any(rgb < 242, axis=2)
    ys, xs = np.nonzero(garment)
    if len(ys) == 0:
        raise ValueError(f"no garment pixels in {path}")
    view = {
        "rgb": rgb.astype(np.float32),
        "garment": garment,
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "cx": float(np.median(xs)),
        "g_top": int(ys.min()),
        "g_bottom": int(ys.max()),
    }
    if span:
        for key in ("top", "bottom", "cx", "g_top", "g_bottom"):
            if key in span:
                view[key] = span[key]
    return view


def load_fit_views(directory: Path, lift_m: float = 0.0) -> dict:
    """Views for alignment and baking.

    Ghost v1 plates stay in the catalog frame. Ghost v2 product shots are
    pasted back into that frame with the placement file. `lift_m` moves the
    garment box up on the body without changing the person frame.
    """
    placement_path = directory / "placement.json"
    placement = json.loads(placement_path.read_text()) if placement_path.exists() else {}
    framed = placement.get("framed", "product" if "front" in placement else "catalog")
    if framed == "product":
        from pipeline.canonical import load_placed_views

        sides = tuple(
            side for side in ("front", "back", "left", "right") if (directory / f"{side}.png").exists()
        )
        return load_placed_views(directory, placement, sides=sides or ("front", "back"), lift_m=lift_m)
    views = {}
    for side in ("front", "back", "left", "right"):
        path = directory / f"{side}.png"
        if not path.exists():
            continue
        views[side] = texture_view(path, placement.get(side))
    if "front" not in views or "back" not in views:
        raise FileNotFoundError(f"front.png and back.png are required in {directory}")
    return views


def _class_span(image_path: Path, class_name: str) -> dict:
    from pipeline.canonical import CLASS_LABELS
    from pipeline.clothes_seg import parse

    image = Image.open(image_path).convert("RGB")
    parsed = parse(image)
    if class_name not in CLASS_LABELS:
        raise ValueError(f"no SegFormer label for {class_name!r}")
    person = parsed["person"]
    cloth = parsed["labels"] == CLASS_LABELS[class_name]
    return _span_from_masks(person, cloth, image_path, class_name)


def _plate_span(photo_path: Path, plate_path: Path) -> dict:
    """Person box from the catalog photo, garment box from the white plate."""
    from pipeline.clothes_seg import parse

    image = Image.open(photo_path).convert("RGB")
    parsed = parse(image)
    plate = np.asarray(Image.open(plate_path).convert("RGB"))
    cloth = np.any(plate < 242, axis=2)
    return _span_from_masks(parsed["person"], cloth, plate_path, "plate")


def _span_from_masks(person: np.ndarray, cloth: np.ndarray, source: Path, name: str) -> dict:
    ys, xs = np.nonzero(person)
    gys, gxs = np.nonzero(cloth)
    if len(ys) == 0 or len(gys) == 0:
        raise ValueError(f"no person or {name} in {source}")
    torso = (ys > np.percentile(ys, 25)) & (ys < np.percentile(ys, 50))
    cx = float(np.median(xs[torso]) if torso.any() else np.median(xs))
    height, width = person.shape
    return {
        "width": int(width),
        "height": int(height),
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "cx": cx,
        "g_top": int(gys.min()),
        "g_bottom": int(gys.max()),
        "g_left": int(gxs.min()),
        "g_right": int(gxs.max()),
    }


def _mark_framed(directory: Path, framed: str) -> None:
    path = directory / "placement.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data["framed"] = framed
    path.write_text(json.dumps(data, indent=2))


def write_prepass(
    outfit: OutfitSpec,
    front: Path,
    back: Path,
    out_root: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Write texture plates, and sheer recon plates, under outputs/canonical/<outfit_id>/."""
    from pipeline.canonical import CLASS_LABELS, outfit_plate, write_dress_views
    from pipeline.sheer_prepass import prepare_sheer_input

    front = repo_path(front)
    back = repo_path(back)
    written: dict[str, dict[str, Path]] = {}
    for garment in outfit.ordered_garments():
        plan = plan_prepass(garment)
        dest = plate_dir(outfit, garment, out_root)
        dest.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if plan.mode == "ghost_v1":
            from pipeline.canonical import dress_plate

            if garment.class_id not in CLASS_LABELS:
                raise ValueError(f"no SegFormer label for {garment.class_id!r}")
            for side, path in (("front", front), ("back", back)):
                photo = Image.open(path).convert("RGB")
                if garment.category == "dress":
                    plate, hole = dress_plate(photo)
                else:
                    plate, hole = outfit_plate(photo, garment.class_id)
                if garment.preserve_graphics and garment.graphic_region_bbox is not None:
                    plate = paste_graphic(plate, photo, garment.graphic_region_bbox)
                out = dest / f"{side}.png"
                plate.save(out)
                Image.fromarray(hole.astype(np.uint8) * 255).save(dest / f"{side}-hole.png")
                paths[side] = out
                print(f"  [{garment.class_id} {side}] ghost v1 -> {out}", flush=True)
            placement = {
                "framed": "catalog",
                "front": _plate_span(front, dest / "front.png"),
                "back": _plate_span(back, dest / "back.png"),
            }
            placement_path = dest / "placement.json"
            placement_path.write_text(json.dumps(placement, indent=2))
            paths["placement"] = placement_path
        else:
            from pipeline.canonical import strip_body

            write_dress_views(
                front,
                back,
                dest,
                views=("front", "back"),
                prompts=garment.prompts,
                label=garment.class_id,
            )
            if garment.empty_prompt:
                for side in ("front", "back"):
                    strip_body(dest / f"{side}.png", garment.empty_prompt)
            _mark_framed(dest, "product")
            paths["front"] = dest / "front.png"
            paths["back"] = dest / "back.png"
            paths["placement"] = dest / "placement.json"
        if plan.vision_plate:
            for side in ("front", "back"):
                recon = prepare_sheer_input(dest / f"{side}.png", dest / f"{side}-recon.png")
                paths[f"{side}_recon"] = recon
        written[garment.class_id] = paths
    return written
