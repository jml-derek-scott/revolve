To build the interactive 3D model from guess_front.png and guess_back.png, you need an execution plan that operationalizes your architectural guidelines while pulling the specific strengths of the two external repositories.The strategy is to use Hunyuan3D-2mv (via Replicate) as a blunt instrument to get the coarse shape, use Cloth2Tex for its pristine UV-mapped base templates, use Garment3DGen to deform that template to match the Hunyuan shape, and finally use Modal for the heavy ControlNet texture inpainting.   Here is your end-to-end execution plan.Phase 1: Pipeline Initialization & 2D Pre-PassFirst, explicitly declare the garment materials and layering hierarchy so the pipeline knows which 2D and 3D collision rules to apply.   Create config_guess.json:JSON{
  "outfit_id": "guess_look_01",
  "gender": "male",
  "target_mannequin": "mannequins/male_base_mesh.glb",
  "garments": [
    {
      "class_id": "jeans",
      "category": "bottom",
      "layer_order": 1,
      "material_type": "opaque",
      "prepass_mode": "ghost_v1",
      "sdf_clearance_mm": 4.0,
      "unwrap_mode": "planar"
    },
    {
      "class_id": "hoodie",
      "category": "top",
      "layer_order": 2,
      "material_type": "opaque",
      "prepass_mode": "ghost_v1",
      "sdf_clearance_mm": 8.0,
      "preserve_graphics": true,
      "graphic_region_bbox": [0.35, 0.25, 0.65, 0.55],
      "unwrap_mode": "cylindrical"
    }
  ]
}
Execute Ghost v1 (Opaque Pre-pass):
Since this is a standard opaque outfit, run your segmentation and LaMa inpainting locally on your Mac Pro to isolate the garments onto a #FFFFFF canvas.Bashpoetry run python scripts/make_canonical_views.py \
  --front test_images/guess_front.png \
  --back test_images/guess_back.png \
  --config config_guess.json \
  --out outputs/canonical/guess/
Phase 2: Coarse Shape GenerationDo not attempt to run Hunyuan3D-2mv locally on your Apple Silicon MPS, as it will exhaust memory. Dispatch this to Replicate.Bashpoetry run python scripts/generate_shape.py \
  --front outputs/canonical/guess/hoodie/front.png \
  --back outputs/canonical/guess/hoodie/back.png \
  --provider replicate \
  --model tencent/hunyuan3d-2mv \
  --out outputs/shape/guess/hoodie_raw.glb
Phase 3: Topology Conditioning (Integrating the Repositories)The hoodie_raw.glb you just generated is watertight, non-manifold, and unsuitable for simulation. You will now use Cloth2Tex and Garment3DGen to fix it.Extract Template from Cloth2Tex:
Navigate to the Cloth2Tex repository and pull the hoodie base template. This template is guaranteed to have fewer than 10,000 vertices (specifically 8,537 vertices and 15,874 faces), unified UV mapping, and simulation-ready edge loops.   Deform using Garment3DGen:
Pass the pristine Cloth2Tex hoodie template and your raw Hunyuan mesh into Garment3DGen's Neural Jacobian Fields. This will warp the perfect topology of the template to match the unique physical shape of your catalog photos.   Bashpoetry run python Garment3DGen/NeuralJacobianFields/PoissonSystem.py \
  --source_template cloth2tex/meshes/hoodie.obj \
  --target_hull outputs/shape/guess/hoodie_raw.glb \
  --out outputs/shape/guess/hoodie_clean.glb
Phase 4: SDF Collision & Mannequin FittingRun your collision solver locally on the Mac Pro to fit the cleaned meshes onto the rigged mannequin without intersections.   Bashpoetry run python scripts/build_outfit.py \
  --config config_guess.json \
  --collision sdf \
  --out outputs/fitted/guess_fitted.glb
Note: Because the jeans are layer_order: 1, they will push 4mm off the mannequin. The hoodie (layer_order: 2) will push 4mm off the jeans, achieving a total 8mm clearance without clipping.   Phase 5: Texture Bake & ControlNet Flank InpaintingBecause orthographic projection leaves ~17% grazing-angle voids under the arms and along the side seams, you must run the ControlNet inpainting pass. Because your Mac Pro lacks CUDA, this must be deployed to Modal.   Deploy to Modal:
Ensure your flank_inpaint.py script is wrapped in a Modal @app.function(gpu="A10G") decorator.Execute Inpainting:Bashmodal run cloud/flank_inpaint.py \
  --mesh outputs/fitted/guess_fitted.glb \
  --front outputs/canonical/guess/hoodie/front.png \
  --back outputs/canonical/guess/hoodie/back.png \
  --prompt "seamless lateral seam, grey cotton hoodie, highly detailed fabric texture"
This script will project the front and back photos, render the lateral depth/normal maps, run the StableDiffusionControlNetInpaintPipeline, and bake the seamless result back into the UV atlas. 