# AGENTS.md — Working in 3DViewer

This file orients AI agents (Cursor, Claude Code, etc.) on **how product photos become 3D GLBs** and where to modify the pipeline. Human-oriented product context lives in `README.md`; low-level dev notes in `CLAUDE.md`.

## 1. What This Repo Is
- **Demo (Primary):** Pre-built `frontend/assets/models/*.glb` loaded by the static UI. 
- **Live API (Secondary):** `POST /api/generate` scaffold for live reconstruction.
- **Submodules (New):** We now integrate Alibaba's `Cloth2Tex` (for base topological templates) and `Garment3DGen` (for Neural Jacobian Field deformation) to ensure output meshes are simulation-ready.

## 2. Hardware Constraints & API Routing
Target machine: **Mac Pro, AMD GPU, PyTorch MPS** (`torch.device("mps")`).
**Strict Rule:** No custom CUDA kernels (e.g., `nvdiffrast`, GPU marching cubes) run locally.
- **Local (MPS):** 2D Ghost pre-passes, SDF collision solving, procedural geometry (straps), and basic mesh operations.
- **Replicate:** Route fast inference here. Used for `tencent/hunyuan3d-2mv` (Stage 1 Shape generation), `allenhooo/lama` (occlusions), and `Flux Kontext` (photo rewriting).
- **Modal:** Route custom CUDA-dependent rasterization here. Used for `Hunyuan3DPaintPipeline` and ControlNet Depth/Normal flank inpainting (`flank_inpaint.py`) on A10G workers.
- **Hugging Face:** Use for `mattmdjaga/segformer_b2_clothes` and TRELLIS endpoints.

## 3. Core Architectural Paradigms
- **2D Pre-Pass First:** NEVER pass raw catalog photos of humans directly to 3D reconstructors (unless doing a `--no-extract` hybrid overlay).
  - *Ghost v1 (Opaque):* SegFormer + rembg + LaMa -> composite onto pure `#FFFFFF`.
  - *Ghost v2 / Sim2 (Sheer/Lace/Prints):* SegFormer fails on sheer/leopard prints. Use AI Photo Rewriting (Flux Kontext) to generate an "empty dress" product shot.
  - *Sheer-Fill Prepass:* Reconstructors tunnel through sheer fabrics. Fill enclosed sheer hollows with dark gray RGB `(24, 24, 24)` exclusively for the reconstructor's vision.
  - *Graphic Preservation:* Cut canonical plates of real pixels for chest graphics/logos to prevent AI rewriting.
- **Shape vs. Texture Separation:** Reconstructors (Hunyuan3D-2mv) build the coarse geometric hull using BOTH front and back photos (to capture open backs/cowls). Texturing is handled via orthographic camera pixel projection + AI inpainting for flanks.
- **Collision & Clearances (SDF):** NEVER use standard ray-casting (`mesh.contains()`) or vertex normal collision checks on the mannequin (it has inverted winding/double-shells). ALWAYS use Layered Signed Distance Fields (SDF). 
  - Base clearance: 4mm outside the skin.
  - Multi-garment layering: Stack sequentially (e.g., jeans 4mm off body, hoodie 4mm off jeans = 8mm total).
- **Procedural Thin Geometry:** 1-2px spaghetti straps are erased by 384³ diffusion voxel grids. Treat straps as a deterministic post-processing step (generate 3D quad strips between bust and shoulder anchors).

## 4. Submodule Integration (Topology Pipeline)
Raw AI generative meshes (Hunyuan3D) are watertight, non-manifold, and unsuitable for physical simulation.
1. **Cloth2Tex:** Pull pristine base templates (e.g., `<10k` vertices, perfect UVs, unified edge loops) from this repository for standard apparel categories.
2. **Garment3DGen:** Use its Neural Jacobian Fields (`PoissonSystem.py`) to deform the `Cloth2Tex` template against the raw `Hunyuan` 3D hull. This yields a mesh that matches the photo's shape but retains open boundaries (neck, waist, sleeves) for simulation.

## 5. Mesh Topology Rules
- AI meshes (Hunyuan, TRELLIS) split vertices on UV boundaries. ALWAYS call `mesh.merge_vertices(merge_tex=True, merge_norm=True)` on temporary copies prior to topological component analysis.
- NEVER use `trimesh.split()` on textured models; it clones texture memory thousands of times and exhausts RAM. Use `trimesh.graph.connected_component_labels`.
- **Texture Baking:** Planar wraps fail on side flanks. Default to `cylindrical_unwrap` for dresses, skirts, and complex seam gaps to prevent tearing. Avoid OpenCV TELEA for complex fabric patterns.

## 6. Code Map
- `frontend/js/app.js` - UI template picker (ensure `?src=` drops on template change to prevent UI blockers).
- `scripts/make_canonical_views.py` - Material-aware 2D pre-pass (Ghost v1/v2, Sheer-Fill).
- `scripts/build_outfit.py` - SDF Collision and multi-shell layering solver.
- `scripts/run_hybrid_poc.py` - Procedural strap generation and hybrid extraction.
- `cloud/flank_inpaint.py` - Modal A10G ControlNet dual Depth+Normal inpainter for 90° grazing angle voids.
- `Garment3DGen/` & `cloth2tex/` - Integrated topology conditioning submodules.

## 7. Tooling for Agents
- **D2 (Diagrams):** CLI `d2 _docs/diagrams/<name>.d2 _docs/diagrams/svg/<name>.svg`.
- **MCPs:** `user-d2` (compile-d2), `user-context7` (API docs), `cursor-ide-browser`.
- **Assumptions to Avoid:** `multi_view.py` is not InstantMesh yet. There are no unit tests. The mannequin has a double-skin gap.
