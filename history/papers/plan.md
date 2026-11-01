# GUESS ATTEMPT PLAN

There is a new set of images, FRONT: `/Users/eris/code/3DViewer/test_images/guess_front.png` and BACK: `/Users/eris/code/3DViewer/test_images/guess_back.png`

## 1. ControlNet & Flux Kontext Inpainting Prompts
When using ControlNet depth/normal-guided diffusion or Flux Kontext to generate canonical plates or side-fill textures, the prompts must explicitly enforce graphic preservation and background neutrality.   

### A. Front Canonical Plate Inpainting (Arm-Gaps & Graphic Lock)Used for LaMa/ControlNet inpainting when arms occlude the garment body or graphic.   

Positive Prompt:
A studio product shot of a [MENS/WOMENS] [COLOR] [GARMENT_TYPE, e.g., teddy bear graphic hoodie], flat lay, isolated on a seamless pure white background (#FFFFFF). Clean fabric texture, preserving the exact front chest graphic without altering the logo, stitch lines, or print details. Natural fabric drape, unoccluded front surface, professional e-commerce catalog lighting, 8k resolution, photorealistic.   

Negative Prompt:
Human body, skin, hands, arms, face, background objects, altered graphic, redrawn print, extra logos, shadows on background, noise, blur, texture distortion, asymmetry.   

### B. Back Canonical Plate Generation (Turn-from-Front)Used when generating or completing the rear view from the accepted front view.   Positive Prompt:
Rear view studio product shot of a [MENS/WOMENS] [COLOR] [GARMENT_TYPE], isolated on a pure white background (#FFFFFF). Matching fabric weave, seam alignment, clean hood and neck construction, plain back panel without front graphics, smooth draping, high-definition e-commerce asset.   

### C. Lateral Flank Inpainting (ControlNet Depth/Normal Guided)
Used for filling grazing-angle voids ($\pm 90^\circ$ under-arm and side seam zones) across lateral camera projections.   Positive Prompt:
Side profile view of a [GARMENT_TYPE], continuous fabric pattern, seamless side seam construction, uniform material texture, diffuse studio lighting, isolated on pure white background (#FFFFFF).   

## 2. Multi-Garment Outfit & Layering Annotation Prompt (Garments2Look Schema)

For pipeline ingestion, pass the following structured JSON specification to enforce decoupled layering, graphic constraints, and collision clearance.

```json
{
  "outfit_id": "outfit_new_01",
  "gender": "male",
  "target_mannequin": "female_mannequin.glb",
  "garments": [
    {
      "class_id": "jeans",
      "category": "bottom",
      "layer_order": 1,
      "sdf_clearance_mm": 4.0,
      "preserve_graphics": false,
      "mask_method": "segformer_b2_clothes"
    },
    {
      "class_id": "hoodie",
      "category": "top",
      "layer_order": 2,
      "sdf_clearance_mm": 8.0,
      "preserve_graphics": true,
      "graphic_region_bbox": [0.35, 0.25, 0.65, 0.55],
      "mask_method": "segformer_lama_arm_fill"
    }
  ],
  "styling": {
    "tucked_in": false,
    "hood_position": "down",
    "arm_drape_lift_m": 0.0
  }
}
```

---

## 3. Pipeline Execution Pipeline Prompts / Commands

Execute the following three-stage pipeline commands to convert the raw image pair into a multi-garment, SDF-fitted 3D outfit[cite: 12, 15, 20]:

### Stage 1: Generate Canonical Plates (Cut Real Pixels for Graphics)
```bash
poetry run python scripts/make_canonical_views.py \
  --front test_images/new_front.jpg \
  --back test_images/new_back.jpg \
  --preserve-graphic \
  --out outputs/canonical/new_set/
```

### Stage 2: Male Texture Wrap (Mannequin Projection)
```bash
poetry run python scripts/bake_male_bodies.py \
  --front outputs/canonical/new_set/front.png \
  --back outputs/canonical/new_set/back.png \
  --out frontend/assets/models/new_set-wrap-model.glb
```

### Stage 3: Modular Garment SDF Collision Fitting & Cylindrical Bake
```bash
poetry run python scripts/build_male_outfit.py \
  --front outputs/canonical/new_set/hoodie/front.png \
  --back outputs/canonical/new_set/hoodie/back.png \
  --left outputs/canonical/new_set/hoodie/left.png \
  --right outputs/canonical/new_set/hoodie/right.png \
  --collision sdf \
  --clearance-body 4 \
  --clearance-inter-garment 4 \
  --unwrap-mode cylindrical \
  --out frontend/assets/models/new_set-hybrid.glb
```