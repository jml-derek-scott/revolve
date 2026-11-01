"""Bow mask, bandeau height, and over-shoulder straps. No cloud calls."""
from __future__ import annotations

import numpy as np
import trimesh

from pipeline.bow import add_bow_tails, satin_loop_mask
from pipeline.garment_extractor import bandeau_top, dress_band
from pipeline.straps import generate_over_shoulder_straps


def _speckled_dress():
    rgb = np.full((200, 160, 3), 255, np.uint8)
    garment = np.zeros((200, 160), dtype=bool)
    garment[40:180, 30:130] = True
    rng = np.random.default_rng(0)
    cloth = rgb.copy()
    speck = rng.integers(0, 255, size=(140, 100), dtype=np.uint8)
    cloth[40:180, 30:130, :] = speck[:, :, None]
    # Smooth dark bow in the upper half of the garment.
    yy, xx = np.ogrid[:200, :160]
    bow = ((yy - 70) / 22) ** 2 + ((xx - 80) / 34) ** 2 <= 1
    cloth[bow] = (12, 12, 14)
    garment[bow] = True
    return cloth, garment, bow


def test_satin_mask_keeps_the_bow_and_drops_sequins():
    rgb, garment, bow = _speckled_dress()
    mask = satin_loop_mask(rgb, garment)
    assert mask[70, 80]
    assert mask[bow].mean() > 0.4
    skirt = np.zeros_like(garment)
    skirt[150:175, 40:120] = True
    assert not mask[skirt].any()


def test_tails_extend_below_the_loops():
    rgb, garment, bow = _speckled_dress()
    loops = satin_loop_mask(rgb, garment)
    full = add_bow_tails(loops, garment)
    assert full.sum() > loops.sum()
    assert int(np.nonzero(full)[0].max()) > int(np.nonzero(loops)[0].max())


def test_bandeau_ignores_strap_tips():
    garment = np.zeros((100, 40), dtype=bool)
    garment[10:30, 18:22] = True  # straps
    garment[30:80, 5:35] = True  # bodice
    view = {"garment": garment, "top": 0, "bottom": 100, "g_top": 10, "g_bottom": 80}
    body = trimesh.creation.box(extents=(0.4, 1.8, 0.3))
    body.apply_translation([0, 0.9, 0])
    band = dress_band({"front": view}, body)
    top = bandeau_top(view, body)
    assert top < band["top"]
    assert top > band["hem"]


def test_straps_meet_at_the_knot():
    garment = trimesh.creation.box(extents=(0.3, 0.5, 0.12))
    garment.apply_translation([0, 1.05, 0])
    body = trimesh.creation.box(extents=(0.34, 1.7, 0.2))
    body.apply_translation([0, 0.85, 0])
    knot = np.array([0.0, 1.15, -0.08])
    straps = generate_over_shoulder_straps(garment, body, knot, samples=16)
    verts = np.asarray(straps.vertices)
    assert len(straps.faces) > 20
    assert np.linalg.norm(verts - knot, axis=1).min() < 0.008
    assert verts[:, 1].max() > garment.bounds[1][1]
