"""Person / garment masks and optional inpaint of bag/hair/hands on the fabric."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import rembg
from PIL import Image

logger = logging.getLogger(__name__)

_sessions: dict = {}


def sessions() -> dict:
    if "person" not in _sessions:
        _sessions["person"] = rembg.new_session()
    return _sessions


def _cloth_session():
    if "cloth" not in _sessions:
        _sessions["cloth"] = rembg.new_session("u2net_cloth_seg")
    return _sessions["cloth"]


def person_cutout(path: Path) -> Image.Image:
    """RGBA cut-out of the person (alpha from rembg)."""
    img = Image.open(path).convert("RGBA")
    alpha = np.array(img)[..., 3]
    if alpha.min() < 255:
        return img
    return rembg.remove(img.convert("RGB"), session=sessions()["person"])


def load_view(
    path: Path,
    inpaint_occlusion: bool = False,
    cloth: str = "u2net",
) -> dict:
    """Return rgb, garment mask, and alignment stats for one product photo.

    `cloth`: `u2net` (rembg cloth-seg, hybrid default) or `segformer`
    (Hugging Face clothes parser: dress vs bag vs hair).

    Occlusion: bag, plus hair over a dilated garment, is inpainted (LaMa, then
    TELEA) and treated as garment so the wrap does not leave a hole.
    """
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img)
    person = np.asarray(rembg.remove(img, session=sessions()["person"], only_mask=True)) > 127
    h, w = person.shape
    bag = np.zeros((h, w), bool)
    hair = np.zeros((h, w), bool)
    occlusion = np.zeros((h, w), bool)

    if cloth == "segformer":
        from pipeline.clothes_seg import grow_sheer, parse as parse_clothes

        parsed = parse_clothes(img)
        garment = parsed["garment"] & person
        bag = parsed["bag"] & person
        hair = parsed["hair"] & person
        face = parsed["face"]
        garment = grow_sheer(rgb, garment, person, bag, hair, face)
        # SegFormer (128² logits) drops 1px spaghetti straps; u2net often keeps them.
        u2 = np.any(
            [np.asarray(m.resize(img.size)) > 127 for m in _cloth_session().predict(img)],
            axis=0,
        )
        ys_p = np.nonzero(person)[0]
        y_cut = int(np.percentile(ys_p, 42)) if len(ys_p) else 0
        gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        upper = np.zeros_like(garment)
        upper[:y_cut] = True
        straps = u2 & person & upper & (gray < 90) & ~bag & ~hair
        if straps.any():
            print(f"  u2net strap assist +{straps.mean() * 100:.2f}% of frame", flush=True)
            garment = garment | straps
        # Only holes on the dress: bag overlapping the hip, hair over the neckline.
        # The whole clutch silhouette is not garment.
        near_bag = cv2.dilate(
            garment.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ).astype(bool)
        near_hair = cv2.dilate(
            garment.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        ).astype(bool)
        occlusion = (bag & near_bag) | (hair & near_hair)
        occlusion &= person & ~garment
        print(
            f"  {path.name}: segformer garment {garment.mean() * 100:.1f}% "
            f"bag {bag.mean() * 100:.1f}% hair {hair.mean() * 100:.1f}%",
            flush=True,
        )
    elif cloth == "u2net":
        masks = _cloth_session().predict(img)
        garment = np.any([np.asarray(m.resize(img.size)) > 127 for m in masks], axis=0) & person
        if inpaint_occlusion:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            near = cv2.dilate(garment.astype(np.uint8), kernel).astype(bool)
            occlusion = person & ~garment & near
    else:
        raise ValueError(f"unknown cloth parser {cloth!r}; use u2net or segformer")

    if inpaint_occlusion and occlusion.any():
        from pipeline.inpaint import inpaint

        rgb = inpaint(rgb, occlusion)
        garment = garment | occlusion

    ys, xs = np.nonzero(person)
    if len(ys) == 0:
        raise ValueError(f"no person found in {path}")
    torso = (ys > np.percentile(ys, 25)) & (ys < np.percentile(ys, 50))
    cx = np.median(xs[torso]) if torso.any() else np.median(xs)
    return {
        "rgb": np.asarray(rgb, dtype=np.float32),
        "garment": garment,
        "person": person,
        "bag": bag,
        "hair": hair,
        "occlusion": occlusion,
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "cx": float(cx),
    }


def prepare_ghost_mannequin(
    image_path: Path,
    view: str = "front",
    cloth: str = "segformer",
    use_lama: bool = True,
    erode_px: int = 2,
) -> dict:
    """Isolate the garment on a clean white background with occlusion inpainting.

    Executes the 3-stage Ghost Mannequin pre-pass:
    1. Person & Cloth Segmentation (rembg + SegFormer/u2net_cloth_seg), explicitly
       excluding human limbs/face/shoes/hair/bag and stripping outer skin-tone fringes.
    2. Occlusion Detection & Inpainting (LaMa on Replicate with TELEA fallback),
       pre-filling non-garment surroundings with nearest garment colors so skin never
       bleeds into the inpainted fabric.
    3. Background & Alpha Boundary Cleanup (1-2px edge erosion + feathering,
       composited onto #FFFFFF and paired RGBA).
    """
    from scipy.ndimage import distance_transform_edt

    path = Path(image_path)
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    person = np.asarray(rembg.remove(img, session=sessions()["person"], only_mask=True)) > 127
    h, w = person.shape

    ys_p, xs_p = np.nonzero(person)
    if len(ys_p) == 0:
        raise ValueError(f"no person found in {path}")
    p_top, p_bottom = int(ys_p.min()), int(ys_p.max())
    p_height = max(p_bottom - p_top, 1)

    bag = np.zeros((h, w), bool)
    hair = np.zeros((h, w), bool)
    face = np.zeros((h, w), bool)
    limbs_shoes = np.zeros((h, w), bool)

    u2_masks = _cloth_session().predict(img)
    u2_garment = np.any([np.asarray(m.resize(img.size)) > 127 for m in u2_masks], axis=0) & person

    above_ankles = np.zeros_like(person)
    above_ankles[: int(p_top + 0.80 * p_height)] = True

    if cloth == "segformer":
        from pipeline.clothes_seg import parse as parse_clothes

        parsed = parse_clothes(img)
        labels = parsed["labels"]
        seg_garment = parsed["garment"] & person
        bag = parsed["bag"] & person
        hair = parsed["hair"] & person
        face = parsed["face"] & person
        # ATR labels: 9=left_shoe, 10=right_shoe, 11=face, 12=left_leg, 13=right_leg, 14=left_arm, 15=right_arm
        limbs_shoes = np.isin(labels, (9, 10, 11, 12, 13, 14, 15)) & person
        garment = (seg_garment | (u2_garment & ~limbs_shoes)) & above_ankles & ~bag & ~hair & ~face
    elif cloth == "u2net":
        garment = u2_garment & above_ankles
    else:
        raise ValueError(f"unknown cloth parser {cloth!r}; use segformer or u2net")

    # Sample the model's skin color in CIE LAB space to strip skin-colored border fringes
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    skin_samples = limbs_shoes & ~garment & ~bag & ~hair
    if not skin_samples.any():
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        skin_samples = (
            person
            & ~garment
            & (ycrcb[..., 1] >= 133)
            & (ycrcb[..., 1] <= 175)
            & (ycrcb[..., 2] >= 75)
            & (ycrcb[..., 2] <= 130)
        )

    skin_like = np.zeros((h, w), bool)
    if skin_samples.any():
        skin_med = np.median(lab[skin_samples], axis=0)
        garment_core = cv2.erode(
            garment.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        ).astype(bool)
        g_med = np.median(lab[garment_core], axis=0) if garment_core.any() else np.array([40.0, 128.0, 128.0])
        d_skin = np.linalg.norm(lab - skin_med[None, None, :], axis=-1)
        d_garm = np.linalg.norm(lab - g_med[None, None, :], axis=-1)
        skin_like = (d_skin < 22.0) & (d_skin < d_garm * 0.85)

        # Strip skin-like pixels along the outer 18px perimeter of the garment mask
        inner_core = cv2.erode(
            garment.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        ).astype(bool)
        outer_band = garment & ~inner_core
        lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
        smooth_skin_edge = outer_band & skin_like & (lap < 18.0)
        garment = garment & ~smooth_skin_edge

    # Recover thin dark spaghetti straps connected to the top of the bodice
    ys_g = np.nonzero(garment)[0]
    if len(ys_g):
        g_top_init = int(ys_g.min())
        g_bot_init = int(ys_g.max())
        strap_band = np.zeros_like(garment)
        strap_band[max(p_top, int(g_top_init - 0.12 * p_height)) : int(g_top_init + 0.25 * (g_bot_init - g_top_init))] = True
        dark_strap = (gray < 85) & ~skin_like & person & strap_band & ~hair & ~face & ~bag & u2_garment
        seed = garment.copy()
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(45):
            grown = cv2.dilate(seed.astype(np.uint8), k5).astype(bool)
            nxt = seed | (grown & dark_strap)
            if np.array_equal(nxt, seed):
                break
            seed = nxt
        garment = seed

    # Keep only significant garment components (drops detached finger/arm fragments)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(garment.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        keep_cc = np.zeros_like(garment)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= max(250, int(0.02 * max_area)):
                keep_cc |= cc_labels == idx
        garment = keep_cc

    raw_garment_mask = garment.copy()

    # Detect occlusions: where bag or hair overlaps or indents the garment bounds
    within_span = np.zeros_like(garment)
    ys_g, xs_g = np.nonzero(garment)
    if len(ys_g):
        g_ymin, g_ymax = int(ys_g.min()), int(ys_g.max())
        g_xmin, g_xmax = int(xs_g.min()), int(xs_g.max())
        within_span[g_ymin : g_ymax + 1, g_xmin : g_xmax + 1] = True

    near_garment = cv2.dilate(
        garment.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    ).astype(bool)
    bag_occ = bag & near_garment & within_span
    hair_occ = hair & near_garment & within_span
    closed_internal = (
        cv2.morphologyEx(
            garment.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        ).astype(bool)
        & within_span
    )
    occlusion = (bag_occ | hair_occ | (closed_internal & ~garment & ~limbs_shoes & ~skin_like)) & person & ~garment
    if occlusion.any():
        occlusion = (
            cv2.dilate(occlusion.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
            & within_span
            & ~limbs_shoes
        )

    logger.info(
        "%s (%s): raw garment %.1f%%, occlusion %.2f%%",
        path.name,
        view,
        raw_garment_mask.mean() * 100,
        occlusion.mean() * 100,
    )
    print(
        f"  {path.name} ({view}): garment {raw_garment_mask.mean() * 100:.1f}%, "
        f"occlusion {occlusion.mean() * 100:.2f}%",
        flush=True,
    )

    # Pre-fill non-garment pixels with nearest garment pixel before inpainting so LaMa/TELEA
    # never pulls peach skin tones or white background into the occluded garment region.
    pure_garment = raw_garment_mask & ~occlusion & ~skin_like
    if not pure_garment.any():
        pure_garment = raw_garment_mask
    _, nearest_idx = distance_transform_edt(~pure_garment, return_indices=True)
    fabric_canvas = rgb[nearest_idx[0], nearest_idx[1]].copy()
    fabric_canvas[raw_garment_mask & ~occlusion] = rgb[raw_garment_mask & ~occlusion]

    inpainted_rgb = fabric_canvas.copy()
    if int(occlusion.sum()) > 25:
        from pipeline.inpaint import inpaint, telea

        if use_lama:
            inpainted_rgb = np.array(inpaint(fabric_canvas, occlusion), copy=True)
        else:
            inpainted_rgb = np.array(telea(fabric_canvas, occlusion), copy=True)
        garment = garment | occlusion
    elif occlusion.any():
        garment = garment | occlusion

    # Neutralize any remaining skin-toned peek-through inside sheer lace so the isolated
    # ghost dress doesn't carry warm peach skin tint in the lace cutouts.
    interior_skin = garment & skin_like
    if interior_skin.any():
        _, lace_idx = distance_transform_edt(~(garment & ~skin_like), return_indices=True)
        nearest_lace = inpainted_rgb[lace_idx[0], lace_idx[1]].astype(np.float32)
        inpainted_rgb[interior_skin] = (
            0.75 * nearest_lace[interior_skin] + 0.25 * inpainted_rgb[interior_skin].astype(np.float32)
        ).round().clip(0, 255).astype(np.uint8)

    # Boundary cleanup: erode 1-2px on thick regions to eliminate skin halos while preserving thin straps
    mask_u8 = garment.astype(np.uint8) * 255
    if erode_px > 0:
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        thin_straps = (dist > 0) & (dist <= float(erode_px + 1.5)) & (gray < 95)
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        eroded = cv2.erode(mask_u8, ek) > 127
        clean_mask = eroded | thin_straps
    else:
        clean_mask = garment

    # Re-prune any tiny crumbs created by erosion
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        keep_cc = np.zeros_like(clean_mask)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= max(120, int(0.01 * max_area)):
                keep_cc |= cc_labels == idx
        clean_mask = keep_cc

    # Refill exterior of inpainted_rgb from clean_mask so feathered edges only blend pure garment color
    _, clean_idx = distance_transform_edt(~clean_mask, return_indices=True)
    edge_safe_rgb = inpainted_rgb[clean_idx[0], clean_idx[1]].copy()

    # Smooth alpha with subtle Gaussian feathering along the garment boundary
    alpha_f = cv2.GaussianBlur(clean_mask.astype(np.float32), (3, 3), 0.8)
    alpha_f = np.where(clean_mask, np.clip(alpha_f, 0.6, 1.0), 0.0)
    alpha_u8 = (alpha_f * 255.0).round().astype(np.uint8)

    # Composite onto pure white (#FFFFFF) for Hunyuan3D-2mv and RGBA for alpha output
    white_bg = np.full_like(edge_safe_rgb, 255, dtype=np.uint8)
    comp_rgb = (
        edge_safe_rgb.astype(np.float32) * alpha_f[..., None]
        + white_bg.astype(np.float32) * (1.0 - alpha_f[..., None])
    ).round().clip(0, 255).astype(np.uint8)
    comp_rgba = np.dstack([edge_safe_rgb, alpha_u8])

    ys_g, xs_g = np.nonzero(clean_mask)
    if len(ys_g) == 0:
        raise ValueError(f"ghost garment mask is empty for {path}")
    g_top, g_bottom = int(ys_g.min()), int(ys_g.max())

    torso = (ys_p > np.percentile(ys_p, 25)) & (ys_p < np.percentile(ys_p, 50))
    cx = float(np.median(xs_p[torso]) if torso.any() else np.median(xs_g))

    return {
        "ghost_rgb": Image.fromarray(comp_rgb, mode="RGB"),
        "ghost_rgba": Image.fromarray(comp_rgba, mode="RGBA"),
        "raw_mask_img": Image.fromarray(raw_garment_mask.astype(np.uint8) * 255, mode="L"),
        "occlusion_mask_img": Image.fromarray(occlusion.astype(np.uint8) * 255, mode="L"),
        "rgb": edge_safe_rgb.astype(np.float32),
        "garment": clean_mask,
        "person": person,
        "bag": bag,
        "hair": hair,
        "occlusion": occlusion,
        "top": p_top,
        "bottom": p_bottom,
        "g_top": g_top,
        "g_bottom": g_bottom,
        "cx": cx,
    }
