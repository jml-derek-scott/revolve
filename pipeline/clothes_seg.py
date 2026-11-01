"""Hugging Face SegFormer clothes parser (dress vs bag vs hair vs skin)."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# ATR / human-parsing labels on this checkpoint.
BACKGROUND = 0
HAIR = 2
UPPER = 4
SKIRT = 5
PANTS = 6
DRESS = 7
BELT = 8
FACE = 11
BAG = 16
SCARF = 17

GARMENT_IDS = (UPPER, SKIRT, PANTS, DRESS, BELT, SCARF)

_model = None
_processor = None
_device = None


def _pick_device():
    # SegFormer on MPS returns almost-all-background logits on this torch 2.2
    # build. CPU is ~1s for B2 and matches the published classes.
    import torch

    return torch.device("cpu")


def _get():
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    _device = _pick_device()
    print(f"  loading {MODEL_ID} on {_device}", flush=True)
    _processor = SegformerImageProcessor.from_pretrained(MODEL_ID, do_reduce_labels=False)
    _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
    _model.to(_device)
    _model.eval()
    return _model, _processor, _device


def parse(image: Image.Image) -> dict:
    """Return boolean masks at the photo's native size."""
    import torch
    import torch.nn.functional as F

    image = image.convert("RGB")
    model, processor, device = _get()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = (
        F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        .argmax(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.int32)
    )
    garment = np.isin(labels, GARMENT_IDS)
    return {
        "labels": labels,
        "garment": garment,
        "bag": labels == BAG,
        "hair": labels == HAIR,
        "face": labels == FACE,
        "person": (labels != BACKGROUND),
    }


def grow_sheer(
    rgb: np.ndarray,
    garment: np.ndarray,
    person: np.ndarray,
    bag: np.ndarray,
    hair: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    """Pull leopard/lace pixels next to the dress that SegFormer called skin."""
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    near = cv2.dilate(garment.astype(np.uint8), kernel).astype(bool)
    blocked = bag | hair | face | ~person | garment
    cand = near & ~blocked

    ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
    skin = (ycrcb[..., 1] >= 133) & (ycrcb[..., 1] <= 173) & (ycrcb[..., 2] >= 77) & (ycrcb[..., 2] <= 127)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    lap = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    textured = lap > 12.0
    dark = gray < 90
    extra = cand & (dark | textured | ~skin)

    # Thin straps: dark pixels in the upper torso, grown out from the bodice.
    ys = np.nonzero(person)[0]
    if len(ys):
        y_cut = int(np.percentile(ys, 48))
        upper = np.zeros_like(garment)
        upper[:y_cut] = True
        dark_upper = (gray < 70) & person & upper & ~hair & ~face & ~bag
        seed = garment.copy()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for _ in range(80):
            grown = cv2.dilate(seed.astype(np.uint8), k).astype(bool)
            nxt = garment | (grown & dark_upper)
            if np.array_equal(nxt, seed):
                break
            seed = nxt
        extra = extra | (seed & ~garment)

    if extra.any():
        print(f"  sheer/strap grow +{extra.mean() * 100:.1f}% of frame", flush=True)
    return garment | extra
