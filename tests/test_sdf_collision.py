"""SDF clearance on synthetic boxes. No mannequin file and no mesh.contains."""
from __future__ import annotations

import numpy as np
import trimesh

from pipeline.sdf_collision import (
    DEFAULT_CLEARANCE,
    ENERGY_TARGET,
    GRID_STEP,
    _self_check,
    body_field,
    push_clearance,
    shell_field,
)


def test_self_check():
    _self_check()


def test_penetrating_box_clears_the_body():
    body = trimesh.creation.box(extents=(0.40, 0.50, 0.24))
    garment = trimesh.creation.box(extents=(0.30, 0.30, 0.16))
    report = push_clearance(garment, body, clearance_m=DEFAULT_CLEARANCE)
    sdf, _ = body_field(body).sample(garment.vertices)
    assert report["energy"] < ENERGY_TARGET
    assert float(np.min(sdf)) >= DEFAULT_CLEARANCE - GRID_STEP


def test_outer_box_stays_off_body_and_inner_shell():
    body = trimesh.creation.box(extents=(0.40, 0.50, 0.24))
    inner = trimesh.creation.box(extents=(0.32, 0.36, 0.18))
    push_clearance(inner, body, clearance_m=DEFAULT_CLEARANCE)
    outer = trimesh.creation.box(extents=(0.30, 0.30, 0.16))
    report = push_clearance(
        outer,
        body,
        clearance_m=DEFAULT_CLEARANCE,
        obstacle=inner,
        obstacle_clearance_m=DEFAULT_CLEARANCE,
    )
    body_sdf, _ = body_field(body).sample(outer.vertices)
    shell_sdf, _ = shell_field(inner).sample(outer.vertices)
    assert report["energy"] < ENERGY_TARGET
    assert report["obstacle_energy"] < ENERGY_TARGET
    assert float(np.min(body_sdf)) >= DEFAULT_CLEARANCE - GRID_STEP
    assert float(np.min(shell_sdf)) >= DEFAULT_CLEARANCE - GRID_STEP
