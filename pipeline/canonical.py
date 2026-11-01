"""Ideal garment images, before any 3D call.

The catalog photo is the reference. Flux Kontext rewrites it into the picture a
reconstructor expects: one garment, fully in frame, white background. The back
edit starts from the real back photo. A dress is sent as the full photo,
because cutting "skin" out of sheer mesh deletes the print.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.clothes_seg import BELT, DRESS, PANTS, SCARF, SKIRT, UPPER, grow_sheer, parse

KONTEXT_MODEL = "black-forest-labs/flux-kontext-pro"

PROMPTS = {
    "hoodie": {
        "front": (
            "Edit this into a clean studio product photo of only this beige hoodie on a plain white background. "
            "Remove the person, the jeans, the shoes, and the hands. "
            "Keep the exact beige color, the hood, the drawstrings, the ribbed cuffs and hem, and the chest graphic "
            "of a teddy bear wearing a cap, with the GUESS wordmark and the red triangle logo. "
            "Do not redesign, recolor, or redraw that graphic. "
            "Show the full hoodie, front view, centered, nothing cropped."
        ),
        "back": (
            "Edit this into a clean studio product photo of only the back of this beige hoodie on a plain white background. "
            "Remove the person, the jeans, the shoes, and the hands. "
            "The back is plain beige fabric. Do not add a logo, wordmark, teddy, triangle, or any print. "
            "Keep the hood, the drawstrings, and the ribbed hem and cuffs. "
            "Show the full hoodie, back view, centered, nothing cropped."
        ),
    },
    "jeans": {
        "front": (
            "Edit this into a clean studio product photo of only these dark blue jeans on a plain white background. "
            "Remove the person, the hoodie, the shoes, and the hands. "
            "Keep the exact denim color and the five-pocket cut. Do not add graphics. "
            "Show the full jeans, front view, centered, nothing cropped."
        ),
        "back": (
            "Edit this into a clean studio product photo of only the back of these dark blue jeans on a plain white background. "
            "Remove the person, the hoodie, the shoes, and the hands. "
            "Keep the yellow arcuate stitching on both back pockets, the small triangle mark on the right pocket, "
            "and the brown leather patch on the waistband. Do not add any other logo or red marks. "
            "Show the full jeans, back view, centered, nothing cropped."
        ),
    },
}

CLASS_LABELS = {
    "hoodie": UPPER,
    "jeans": PANTS,
    "dress": DRESS,
    "skirt": SKIRT,
}

# The catalog body ratio lands this slip's lace under the mannequin's bust.
# 10 cm up puts that lace on the chest. Side photos paint the flanks only
# inside this half-width, so the arms stay bare.
DRESS_LIFT_M = 0.10
BODY_M = 1.803
TORSO_SIDE_X = 0.18

# Template 1. The full catalog photo is the input. A class plate would punch
# the leopard mesh out, because SegFormer calls that sheer region skin.
DRESS_PROMPTS = {
    "front": (
        "Remove the woman and leave only the garment she is wearing, on a plain white background. "
        "Delete her skin, hair, face, arms, hands, legs, shoes, jewelry, and the black clutch. "
        "This is a short mini slip, and the hem stays at the upper thigh, exactly as short as in the photo. "
        "Do not lengthen it. It is not a midi, not a gown, and it has no second skirt. "
        "Keep two thin black spaghetti straps, a black floral lace bust, leopard-print mesh from the bust "
        "straight down to the hem, and a short scalloped black lace trim on that hem. "
        "Fill the spot the clutch covered with the same short leopard mesh and lace trim. "
        "Do not add an empire waist, a belt, a longer lace overlay, a train, or a bodysuit. "
        "Empty dress, no body inside. Front view, centered, full garment in frame."
    ),
    "back": (
        "Remove the woman and leave only the back of the short mini slip she is wearing, on a plain white background. "
        "Delete her skin, hair, arms, legs, and heels. "
        "The hem stays at the upper thigh, exactly as short as in the photo. Do not lengthen it. "
        "It is fitted, the same width from the bust to the hem, not flared and not A-line. "
        "Keep two thin black spaghetti straps, a horizontal black floral lace band across the upper back, "
        "leopard-print mesh below that band, and a short scalloped black lace trim at the upper-thigh hem. "
        "Do not add a solid black bodice, a waist seam, pleats, side flares, or a longer skirt. "
        "Empty dress, no body inside. Back view, centered, full garment in frame."
    ),
    "left": (
        "Turn this same short mini slip ninety degrees to an exact left side profile, on a plain white background. "
        "The hem stays at the upper thigh. Do not lengthen it, and do not add a lace overlay skirt or a train. "
        "Keep the thin spaghetti straps, the black floral lace bust, the leopard mesh, and the short scalloped lace hem. "
        "No person and no body inside the dress. Full garment, centered, nothing cropped."
    ),
    "right": (
        "Turn this same short mini slip ninety degrees to an exact right side profile, on a plain white background. "
        "The hem stays at the upper thigh. Do not lengthen it, and do not add a lace overlay skirt or a train. "
        "Keep the thin spaghetti straps, the black floral lace bust, the leopard mesh, and the short scalloped lace hem. "
        "No person and no body inside the dress. Full garment, centered, nothing cropped."
    ),
}

# Template 2. Solid black, not a print. The overlay is midi and sheer; a short
# opaque slip sits under it. The first reconstructions treated this as a floral
# dress and a fused person. These prompts are the restart.
T2_PROMPTS = {
    "front": (
        "Remove the woman and leave only the dress she is wearing, on a plain white background. "
        "Delete her skin, hair, face, arms, hands, legs, shoes, jewelry, and the black handbag. "
        "This is a solid black midi. It has no floral print, no pattern, and no color other than black. "
        "Keep two thin spaghetti straps and a straight opaque bandeau across the bust. "
        "Under the bust is a sheer black panel, then a short opaque black slip, then a sheer black overlay "
        "whose hem falls to mid-calf, exactly as long as in the photo. Do not lengthen it to the floor and do not shorten it to a mini. "
        "Keep one high slit in the sheer overlay over the left leg, up to the top of the thigh. "
        "Fill the spot the handbag covered with the same sheer black overlay. "
        "Do not add sleeves, a belt, a lining print, or a second skirt. "
        "The dress is empty. No legs, knees, feet, or skin, including through the sheer fabric and the slit. "
        "Sheer and open areas show the plain white background. "
        "Front view, centered, full garment in frame."
    ),
    "back": (
        "Remove the woman and leave only the back of the black midi dress, floating on a plain white background. "
        "Delete her completely: hair, back, arms, hands, legs, and shoes. "
        "Solid black, no floral print. "
        "Keep two thin spaghetti straps, a straight opaque bandeau with a small vertical keyhole at the center, "
        "a sheer black band under the bandeau, a short opaque black slip, and one continuous sheer overlay to mid-calf. "
        "Do not lengthen it to the floor and do not shorten it to a mini. "
        "The back skirt has no slit and no center opening. "
        "The sheer skirt is a single piece of fabric. Do not cut leg-shaped holes out of it, and do not leave two white columns where the legs were. "
        "Where the fabric is sheer, the white background shows through evenly. "
        "Empty dress. No person, no mannequin, no legs. Centered, full garment in frame."
    ),
    "left": (
        "Turn this same black midi dress ninety degrees to an exact left side profile, on a plain white background. "
        "Solid black, no floral print. Keep the thin spaghetti straps, the straight bandeau, the sheer panel, "
        "the short opaque slip, and the sheer overlay hem at mid-calf. Do not lengthen or shorten it. "
        "The dress stays empty. Sheer fabric shows the white background, not skin or legs. "
        "Full garment, centered, nothing cropped."
    ),
    "right": (
        "Turn this same black midi dress ninety degrees to an exact right side profile, on a plain white background. "
        "Solid black, no floral print. Keep the thin spaghetti straps, the straight bandeau, the sheer panel, "
        "the short opaque slip, the high slit, and the sheer overlay hem at mid-calf. Do not lengthen or shorten it. "
        "The dress stays empty. Sheer fabric shows the white background, not skin or legs. "
        "Full garment, centered, nothing cropped."
    ),
}

# The catalog back keeps a figure inside the sheer skirt. Turning the accepted
# empty front, the same way the side profiles are made, does not.
T2_BACK_FROM_FRONT = (
    "Turn this same empty black dress around to an exact back view, on a plain white background. "
    "Keep it empty. Do not add a person, a mannequin, or legs. "
    "The back has two thin spaghetti straps, a straight bandeau with a small vertical keyhole at the center, "
    "a sheer band under the bandeau, a short opaque black slip, and one continuous sheer overlay to mid-calf. "
    "The back skirt has no slit and no leg-shaped holes. Sheer fabric shows the white background evenly. "
    "Centered, full garment in frame."
)

# Second edit for a sheer dress. The first pass keeps the cloth and also the
# legs you can see through it. This pass only removes that body.
T2_EMPTY = (
    "Edit this into a photo of only the black dress, floating on a plain white background. "
    "Keep the dress the same: straps, bandeau, short slip, sheer overlay, hem, and slit. "
    "The dress is hollow. Sheer panels and the open slit are transparent to the white background. "
    "Do not add a mannequin, a head, arms, or legs."
)


# SegFormer B2 clothes: arms sit in front of the hoodie pocket and the jeans.
LEFT_ARM = 14
RIGHT_ARM = 15


def outfit_plate(image: Image.Image, class_name: str) -> tuple[Image.Image, np.ndarray]:
    """One garment on white. LaMa fills only arm pixels that sit on the cloth.

    The chest graphic and the pocket stitching stay the catalog pixels. A close
    of the class mask finds the hand-sized gaps; anything that is not an arm
    is left alone, so the other garment and the studio background are not filled.
    """
    from pipeline.inpaint import inpaint

    label_id = CLASS_LABELS[class_name]
    parsed = parse(image.convert("RGB"))
    labels = parsed["labels"]
    cloth = labels == label_id
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    closed = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    arms = np.isin(labels, (LEFT_ARM, RIGHT_ARM))
    hole = closed & ~cloth & arms & ~parsed["face"] & ~parsed["hair"]
    rgb = np.asarray(image.convert("RGB"))
    filled = inpaint(rgb, hole) if int(hole.sum()) > 25 else rgb
    out = np.full_like(rgb, 255)
    keep = cloth | hole
    if keep.any():
        out[keep] = filled[keep]
    return Image.fromarray(out, mode="RGB"), hole


# Satin bust, sequin skirt, and the back bow. SegFormer splits those across
# DRESS / SKIRT / UPPER / SCARF. Sparkle pixels that look like skin stay in
# the closed silhouette; only a hand sitting on the cloth is filled.
DRESS_PLATE_IDS = (DRESS, SKIRT, UPPER, SCARF, BELT)


def dress_plate(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Opaque dress on white. Real sequin pixels. LaMa only on a hand inside the cloth."""
    from pipeline.inpaint import inpaint

    parsed = parse(image.convert("RGB"))
    labels = parsed["labels"]
    rgb = np.asarray(image.convert("RGB"))
    drop = parsed["face"] | parsed["hair"] | parsed["bag"]
    cloth = np.isin(labels, DRESS_PLATE_IDS) & ~drop
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Black satin/sequin labeled as hair. Brown hair is lighter and stays out.
    dark = (gray < 45) & parsed["person"] & ~parsed["face"] & ~parsed["bag"]
    near = cv2.dilate(cloth.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    cloth = cloth | (dark & near)
    sparkle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(cloth.astype(np.uint8), cv2.MORPH_CLOSE, sparkle).astype(bool)
    arms = np.isin(labels, (LEFT_ARM, RIGHT_ARM))
    hole = closed & ~cloth & arms
    filled = inpaint(rgb, hole) if int(hole.sum()) > 25 else rgb
    keep = closed | hole
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = (
        (ycrcb[..., 1] >= 133)
        & (ycrcb[..., 1] <= 173)
        & (ycrcb[..., 2] >= 77)
        & (ycrcb[..., 2] <= 127)
    )
    ring = keep & ~cv2.erode(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    keep = keep & ~(ring & skin & ~hole)
    out = np.full_like(rgb, 255)
    if keep.any():
        out[keep] = filled[keep]
    return Image.fromarray(out, mode="RGB"), hole


def write_outfit_plates(front_path: Path, back_path: Path, out_dir: Path) -> None:
    """Hoodie and jeans plates from the real photos. No image-model rewrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in ("hoodie", "jeans"):
        dest_dir = out_dir / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for side, path in (("front", front_path), ("back", back_path)):
            photo = Image.open(path).convert("RGB")
            plate, hole = outfit_plate(photo, class_name)
            dest = dest_dir / f"{side}.png"
            plate.save(dest)
            Image.fromarray(hole.astype(np.uint8) * 255).save(dest_dir / f"{side}-hole.png")
            print(
                f"  [{class_name} {side}] hole {hole.mean() * 100:.2f}% -> {dest}",
                flush=True,
            )


def class_plate(image: Image.Image, class_name: str) -> Image.Image:
    """Garment pixels for one class on white. Interior holes in the mask are closed."""
    label_id = CLASS_LABELS[class_name]
    parsed = parse(image.convert("RGB"))
    mask = parsed["labels"] == label_id
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    closed &= ~parsed["face"] & ~parsed["hair"] & ~parsed["bag"]
    rgb = np.asarray(image.convert("RGB"))
    out = np.full_like(rgb, 255)
    if closed.any():
        out[closed] = rgb[closed]
    return Image.fromarray(out, mode="RGB")


def _png_buf(image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    buf.name = "image.png"
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def kontext_edit(image: Image.Image, prompt: str, safety_tolerance: int = 2) -> Image.Image:
    """One Kontext edit. Retries a rate limit. Does not invent a second view."""
    from pipeline.reconstruct import _read_url, _replicate_client

    client = _replicate_client()
    output = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            output = client.run(
                KONTEXT_MODEL,
                input={
                    "prompt": prompt,
                    "input_image": _png_buf(image),
                    "aspect_ratio": "match_input_image",
                    "output_format": "png",
                    "prompt_upsampling": False,
                    "safety_tolerance": safety_tolerance,
                },
            )
            break
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) and attempt < 2:
                print("  Kontext rate-limited (429); waiting 12s", flush=True)
                time.sleep(12.0)
                continue
            raise
    if output is None:
        raise last_exc or RuntimeError("Kontext returned nothing")
    if hasattr(output, "read"):
        data = output.read()
    else:
        url = getattr(output, "url", None) or str(output)
        data = _read_url(str(url))
    return Image.open(io.BytesIO(data)).convert("RGB")


def write_class_views(
    front_path: Path,
    back_path: Path,
    class_name: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Plate plus Kontext front and back for one class. Returns the ideal pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ideal = {}
    for side, path in (("front", front_path), ("back", back_path)):
        photo = Image.open(path).convert("RGB")
        plate = class_plate(photo, class_name)
        plate_path = out_dir / f"{side}-plate.png"
        plate.save(plate_path)
        print(f"  [{class_name} {side}] Kontext from {plate_path.name}", flush=True)
        edited = kontext_edit(plate, PROMPTS[class_name][side])
        dest = out_dir / f"{side}.png"
        edited.save(dest)
        ideal[side] = dest
        print(f"  wrote {dest}", flush=True)
    return ideal["front"], ideal["back"]


def measure_placement(image_path: Path) -> dict:
    """Person box and dress bust-to-hem on a catalog photo.

    SegFormer labels the body. Sheer growth pulls the leopard mesh back into
    the dress so the hem is the lace, not the skin tone under it. The numbers
    are in this photo's pixels. A centered product shot has no feet, so the
    wrap and the ghost fit paste the dress back into this frame.
    """
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    parsed = parse(image)
    person = parsed["person"]
    garment = grow_sheer(
        rgb, parsed["garment"], person, parsed["bag"], parsed["hair"], parsed["face"]
    )
    garment = garment & person & ~parsed["bag"] & ~parsed["hair"] & ~parsed["face"]
    ys, xs = np.nonzero(person)
    gys, gxs = np.nonzero(garment)
    if len(ys) == 0 or len(gys) == 0:
        raise ValueError(f"no person or dress found in {image_path}")
    torso = (ys > np.percentile(ys, 25)) & (ys < np.percentile(ys, 50))
    cx = float(np.median(xs[torso]) if torso.any() else np.median(xs))
    h, w = person.shape
    placement = {
        "width": int(w),
        "height": int(h),
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "cx": cx,
        "g_top": int(gys.min()),
        "g_bottom": int(gys.max()),
        "g_left": int(gxs.min()),
        "g_right": int(gxs.max()),
    }
    p_h = max(placement["bottom"] - placement["top"], 1)
    g_h = placement["g_bottom"] - placement["g_top"]
    print(
        f"  {image_path.name}: person {placement['top']}–{placement['bottom']} "
        f"dress {placement['g_top']}–{placement['g_bottom']} "
        f"height_ratio {g_h / p_h:.2f} "
        f"hem_above_feet {(placement['bottom'] - placement['g_bottom']) / p_h:.2f}",
        flush=True,
    )
    return placement


def strip_body(path: Path, prompt: str) -> Path:
    """Second Kontext edit. Removes a body left inside an otherwise finished garment shot."""
    image = Image.open(path).convert("RGB")
    print(f"  [strip] {path.name}", flush=True)
    edited = kontext_edit(image, prompt, safety_tolerance=5)
    edited.save(path)
    print(f"  wrote {path}", flush=True)
    return path


def write_dress_views(
    front_path: Path,
    back_path: Path,
    out_dir: Path,
    views: tuple[str, ...] = ("front", "back", "left", "right"),
    prompts: dict[str, str] | None = None,
    label: str = "dress",
    back_from_front: str | None = None,
) -> dict[str, Path]:
    """Kontext product shots, plus the body placement.

    Front is edited from the real front. Back is edited from the real back.
    Left and right are edited from the accepted front, and are for texture
    only. `prompts` defaults to the template 1 slip. Returns the written
    paths. Does not call a reconstructor.
    """
    prompts = prompts or DRESS_PROMPTS
    out_dir.mkdir(parents=True, exist_ok=True)
    placement_path = out_dir / "placement.json"
    if placement_path.exists() and views and set(views) <= {"left", "right"}:
        placement = json.loads(placement_path.read_text())
    else:
        placement = {
            "front": measure_placement(front_path),
            "back": measure_placement(back_path),
        }
        placement_path.write_text(json.dumps(placement, indent=2))
        print(f"  wrote {placement_path}", flush=True)

    written: dict[str, Path] = {}
    front_ideal = out_dir / "front.png"
    if "front" in views:
        photo = Image.open(front_path).convert("RGB")
        print(f"  [{label} front] Kontext from the catalog photo", flush=True)
        edited = kontext_edit(photo, prompts["front"])
        edited.save(front_ideal)
        written["front"] = front_ideal
        print(f"  wrote {front_ideal}", flush=True)
    if "back" in views:
        if back_from_front:
            if not front_ideal.exists():
                raise FileNotFoundError(f"accepted front is required for the back: {front_ideal}")
            photo = Image.open(front_ideal).convert("RGB")
            print(f"  [{label} back] Kontext turn of the accepted front", flush=True)
            edited = kontext_edit(photo, back_from_front, safety_tolerance=5)
        else:
            photo = Image.open(back_path).convert("RGB")
            print(f"  [{label} back] Kontext from the catalog photo", flush=True)
            edited = kontext_edit(photo, prompts["back"])
        dest = out_dir / "back.png"
        edited.save(dest)
        written["back"] = dest
        print(f"  wrote {dest}", flush=True)
    if any(side in views for side in ("left", "right")):
        if not front_ideal.exists():
            raise FileNotFoundError(f"accepted front is required for profiles: {front_ideal}")
        accepted = Image.open(front_ideal).convert("RGB")
        for side in ("left", "right"):
            if side not in views:
                continue
            print(f"  [{label} {side}] Kontext profile from the accepted front", flush=True)
            edited = kontext_edit(accepted, prompts[side])
            dest = out_dir / f"{side}.png"
            edited.save(dest)
            written[side] = dest
            print(f"  wrote {dest}", flush=True)
    print("  look at these PNGs before any 3D call", flush=True)
    return written


def _garment_on_white(rgb: np.ndarray) -> np.ndarray:
    """True where the pixel is not the studio white behind a product shot."""
    return np.any(rgb < 242, axis=-1)


def _drop_background_islands(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Drop specks, and drop detached islands that are still studio white.

    A warm JPEG corner can sit just under the white cut and pass as cloth.
    That island stretches the crop, so the real garment is pasted off the
    torso midline and the bake samples the background. The largest component
    stays even when the garment itself is light.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    keep_area = max(80, int(0.01 * int(areas.max())))
    keep_ids = []
    for i, area in enumerate(areas):
        comp_id = i + 1
        if int(area) < keep_area:
            continue
        if comp_id != largest:
            cols = rgb[labels == comp_id]
            darkest = float(np.percentile(cols.min(axis=1), 5))
            if darkest > 230:
                continue
        keep_ids.append(comp_id)
    if not keep_ids:
        keep_ids = [largest]
    return np.isin(labels, keep_ids)


def load_product_view(path: Path, placement: dict) -> dict:
    """Paste a centered garment-on-white shot back into the catalog person frame.

    `placement` is one side of placement.json (the original photo's person box
    and dress bust-to-hem). The baker maps that person box onto the mannequin,
    so the dress lands on the torso instead of stretching to the feet.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"))
    mask = _drop_background_islands(rgb, _garment_on_white(rgb))
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError(f"no garment pixels in {path}")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgb[y0:y1, x0:x1]
    crop_m = mask[y0:y1, x0:x1]

    height = int(placement["height"])
    width = int(placement["width"])
    g_top = int(placement["g_top"])
    g_bottom = int(placement["g_bottom"])
    target_h = max(g_bottom - g_top, 1)
    target_w = max(int(round(crop.shape[1] * (target_h / crop.shape[0]))), 1)
    patch = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    patch_m = cv2.resize(crop_m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0

    canvas = np.full((height, width, 3), 255, np.uint8)
    garment = np.zeros((height, width), bool)
    x_left = int(round(float(placement["cx"]) - target_w / 2))
    _paste(canvas, garment, patch, patch_m, x_left, g_top)
    return {
        "rgb": canvas.astype(np.float32),
        "garment": garment,
        "person": garment,
        "top": int(placement["top"]),
        "bottom": int(placement["bottom"]),
        "cx": float(placement["cx"]),
        "g_top": g_top,
        "g_bottom": g_bottom,
    }


def _paste(
    canvas: np.ndarray,
    mask: np.ndarray,
    patch: np.ndarray,
    patch_m: np.ndarray,
    x: int,
    y: int,
) -> None:
    height, width = mask.shape
    ph, pw = patch_m.shape
    x1, y1 = x + pw, y + ph
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(width, x1), min(height, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    sub_m = patch_m[sy0:sy1, sx0:sx1]
    canvas[dy0:dy1, dx0:dx1][sub_m] = patch[sy0:sy1, sx0:sx1][sub_m]
    mask[dy0:dy1, dx0:dx1][sub_m] = True


def shift_placement(placement: dict, lift_m: float) -> dict:
    """Move the dress box up in the photo so it lands higher on the body.

    Image y grows downward, so a positive lift subtracts pixels from the
    bust and hem rows. The person box stays put.
    """
    if lift_m <= 0:
        return placement
    shifted = dict(placement)
    span = max(int(shifted["bottom"]) - int(shifted["top"]), 1)
    delta = int(round(lift_m * span / BODY_M))
    shifted["g_top"] = max(0, int(shifted["g_top"]) - delta)
    shifted["g_bottom"] = max(shifted["g_top"] + 1, int(shifted["g_bottom"]) - delta)
    return shifted


def load_placed_views(
    directory: Path,
    placement: dict,
    sides: tuple[str, ...] = ("front", "back"),
    lift_m: float = 0.0,
) -> dict:
    """Product-shot views keyed the way bake_photos expects."""
    views = {}
    for side in sides:
        frame = placement["front"] if side in ("left", "right") else placement[side]
        views[side] = load_product_view(directory / f"{side}.png", shift_placement(frame, lift_m))
    if lift_m:
        front = views["front"]
        print(
            f"  dress lifted {lift_m:.2f} m (photo rows {front['g_top']}–{front['g_bottom']})",
            flush=True,
        )
    return views
