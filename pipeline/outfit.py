"""Outfit and garment config. Clearance in the file is millimetres."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]

MaterialType = Literal["opaque", "sheer"]
PrepassMode = Literal["ghost_v1", "ghost_v2"]
UnwrapMode = Literal["planar", "cylindrical"]
Category = Literal["top", "bottom", "dress"]


class PipelineDefaults(BaseModel):
    """Values from configs/default_pipeline.yaml. Garment fields default to these."""

    background_rgb: tuple[int, int, int] = (255, 255, 255)
    clearance_mm: float = 4.0
    sheer_fill_rgb: tuple[int, int, int] = (24, 24, 24)
    dress_unwrap: UnwrapMode = "cylindrical"


class GarmentSpec(BaseModel):
    class_id: str
    category: Category
    layer_order: int = 1
    material_type: MaterialType = "opaque"
    prepass_mode: PrepassMode = "ghost_v1"
    sdf_clearance_mm: float = 4.0
    lift_m: float = 0.0
    unwrap_mode: UnwrapMode = "cylindrical"
    # Photo pixels brighter than this are the studio background, not cloth.
    # None keeps every pixel the garment mask accepts.
    background_max: int | None = None
    preserve_graphics: bool = False
    graphic_region_bbox: tuple[float, float, float, float] | None = None
    prompts: dict[str, str] | None = None
    empty_prompt: str | None = None

    @field_validator("graphic_region_bbox", mode="before")
    @classmethod
    def _bbox(cls, value):
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("graphic_region_bbox is x0, y0, x1, y1")
        x0, y0, x1, y1 = (float(v) for v in value)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("graphic_region_bbox must be normalized and ordered")
        return (x0, y0, x1, y1)

    @model_validator(mode="after")
    def _graphics(self):
        if self.preserve_graphics and self.prepass_mode == "ghost_v2":
            raise ValueError(f"{self.class_id}: preserve_graphics refuses ghost_v2")
        if self.preserve_graphics and self.material_type == "sheer":
            raise ValueError(f"{self.class_id}: preserve_graphics refuses a sheer rewrite")
        if self.preserve_graphics and self.graphic_region_bbox is None:
            raise ValueError(f"{self.class_id}: preserve_graphics requires graphic_region_bbox")
        return self

    @property
    def clearance_m(self) -> float:
        """Metres for push_clearance. 4 mm in the file is 0.004 m."""
        return self.sdf_clearance_mm / 1000.0


class OutfitSpec(BaseModel):
    outfit_id: str
    target_mannequin: str
    garments: list[GarmentSpec] = Field(min_length=1)
    gender: str | None = None

    def ordered_garments(self) -> list[GarmentSpec]:
        return sorted(self.garments, key=lambda garment: (garment.layer_order, garment.class_id))

    def select_garment(self, class_id: str | None) -> GarmentSpec:
        if class_id is None:
            if len(self.garments) != 1:
                names = ", ".join(garment.class_id for garment in self.ordered_garments())
                raise ValueError(f"outfit has {names}; pass --class")
            return self.garments[0]
        for garment in self.garments:
            if garment.class_id == class_id:
                return garment
        raise ValueError(f"no garment {class_id!r} in {self.outfit_id}")


class LayerFit(BaseModel):
    """Body clearance, plus the previous layer when this garment stacks on it."""

    class_id: str
    clearance_m: float
    obstacle_class_id: str | None = None
    obstacle_clearance_m: float | None = None


def layer_fits(outfit: OutfitSpec) -> list[LayerFit]:
    """First layer is clearance_m off the body. Each later layer is also that far off the one under it."""
    fits: list[LayerFit] = []
    previous: GarmentSpec | None = None
    for garment in outfit.ordered_garments():
        fits.append(
            LayerFit(
                class_id=garment.class_id,
                clearance_m=garment.clearance_m,
                obstacle_class_id=None if previous is None else previous.class_id,
                obstacle_clearance_m=None if previous is None else garment.clearance_m,
            )
        )
        previous = garment
    return fits


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_outfit(path: str | Path) -> OutfitSpec:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return OutfitSpec.model_validate(data)


def load_defaults(path: str | Path | None = None) -> PipelineDefaults:
    path = Path(path) if path is not None else ROOT / "configs" / "default_pipeline.yaml"
    data = yaml.safe_load(path.read_text())
    return PipelineDefaults.model_validate(data)
