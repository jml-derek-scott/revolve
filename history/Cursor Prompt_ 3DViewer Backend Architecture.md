# **Cursor Implementation Prompt: 3DViewer Backend**

**Context:** You are building the production backend for "3DViewer", an automated microservice pipeline that ingests front and back 2D e-commerce catalog photographs and outputs simulation-ready, modular 3D garment shells fitted onto standardized, unskinned rigged mannequins (mannequins/female\_base\_mesh.glb and mannequins/male\_base\_mesh.glb). Target environment is Python 3.10+ using Poetry.

## **1\. Core Architecture & Strategic Paradigms**

Your primary objective is to extract clean, modular 3D garment shells that can be swapped onto a standardized, rigged female or male mannequin. You strictly enforce a **"2D Pre-pass First"** rule: never pass raw catalog photos of humans directly into 3D reconstructors unless doing a \--no-extract hybrid overlay.

> * **Ghost v1 (Opaque Garments):** Use a 2D segmentation pipeline (rembg \+ SegFormer \+ LaMa occlusion inpainting) to eliminate skin, faces, hair, and accessories. Composite the floating garment onto a pure \#FFFFFF background.  
> * **Ghost v2 / Sim2 (Sheer, Lace, & Complex Prints):** Never use masks on sheer mesh or prints (e.g., leopard print), as SegFormer will falsely classify it as "skin" and destroy the garment. Instead, use AI Photo Rewriting (Flux Kontext) to generate an "empty dress" product shot.  
> * **The Sheer-Fill Prepass:** Reconstructors tunnel through sheer fabrics, treating white backgrounds seen through cloth as holes. Before sending a sheer garment to 3D, you must fill enclosed hollow areas with dark gray RGB(24, 24, 24\) exclusively for the reconstructor's vision.  
> * **Graphic Preservation:** Never use AI photo-rewriting on specific chest graphics or logos (e.g., a printed teddy bear). Cut canonical plates of the real pixels and map them back modularly.

## **2\. Shape vs. Texture Separation**

Treat shape generation and texturing as entirely decoupled services.

> * **3D Shape:** Generate geometry via multi-view conditioning (Hunyuan3D-2mv). You MUST feed both front and back photos to the reconstructor to ensure real design elements (open backs, cowls, neck ties) are generated physically, rather than hallucinated as closed backs.  
> * **Texturing & Flanks:** Combine high-fidelity orthographic camera pixel projection on facing surface normals with AI inpainting for lateral flanks/grazing angles (\~17% under-arm and hip seams). For complex seam gaps, utilize cylindrical 4-view bakes rather than planar wraps to prevent tearing on the flanks.

## **3\. Fitting & Clearances**

> * **Layered SDF Collision:** Rely on Signed Distance Fields (SDF) for collision clearance, pushing garment vertices \~4mm outside the skin (CLEARANCE).  
> * **Multi-Garment Modularity:** When assembling multiple overlapping pieces (e.g., hoodie and jeans), apply layered SDF clearance: lift the bottom layer 4mm off the body, and the top layer 4mm off the bottom layer.

## **4\. Hard Hardware & Geometry Constraints**

> * **Development Environment:** Mac Pro with AMD GPU using PyTorch MPS (torch.device("mps")). It lacks custom CUDA kernel support (no local nvdiffrast or GPU marching cubes). All heavy multi-view diffusion and CUDA rasterization must dispatch to cloud GPU workers.  
> * **Base Mannequin Geometry:** The standard mannequin has a double-skinned shell (\~5mm gap) and scrambled/inverted winding. NEVER use standard ray-casting (mesh.contains()) or vertex normal collision checks. Always use SDF or 2D orthographic depth-buffer bounds for alignment and skin-containment tests.  
> * **Mesh Topology Rules:**  
  * AI meshes (Hunyuan, TRELLIS) split vertices on UV boundaries. Always call mesh.merge\_vertices(merge\_tex=True, merge\_norm=True) on temporary copies prior to topological component analysis.  
  * NEVER use trimesh.split() on textured models; it clones texture memory thousands of times and exhausts RAM. Use trimesh.graph.connected\_component\_labels instead.  
  * Inpainting: Avoid OpenCV TELEA for complex fabric patterns; use diffusion-based inpainting for structural gaps.

## **5\. API Integrations & Service Routing**

You have access to Replicate, Modal, and Hugging Face. Route tasks intelligently:

> * **Replicate:** Use for fast inference of tencent/hunyuan3d-2mv (Stage 1 Shape generation), allenhooo/lama (occlusion removal), and Flux models (Kontext empty-dress generation).  
> * **Modal:** Use for custom CUDA-dependent rasterization tasks, specifically compiling and running Hunyuan3DPaintPipeline (Hunyuan3D-Paint v2.0 Turbo) on A10G workers to texture gray-seam flanks.  
> * **Hugging Face:** Use for accessing specialized SegFormer models (mattmdjaga/segformer\_b2\_clothes) and TRELLIS endpoints.

## **6\. Cursor Instructions & Working Style**

> * Write production-grade, defensive Python 3.10+ code compatible with Poetry and the existing pipeline/, scripts/, and cloud/ packages.  
> * Provide clear, actionable CLI scripts and unit tests for every step before modifying production modules.  
> * When troubleshooting texture seam tearing, grazing-angle voids, or cloth-to-body clipping, diagnose the mathematical root cause (UV mapping, depth buffers, surface normals, or mask dilation) before providing code fixes.  
> * Always write and run Python code to calculate and verify 3D camera projection matrices, depth-buffer bounds, and UV math.

## **7\. Target Repository File Tree**

`3DViewer-Backend/`  
`├── pyproject.toml`  
`├── README.md`  
`├── configs/`  
`│   ├── default_pipeline.yaml`  
`│   └── templates/`  
`│       ├── dress_slip.json`  
`│       └── hoodie_jeans_layered.json`  
`├── mannequins/`  
`│   ├── female_base_mesh.glb`  
`│   └── male_base_mesh.glb`  
`├── src/`  
`│   ├── __init__.py`  
`│   ├── config.py                     # Pydantic Settings & Outfit/Garment schemas`  
`│   ├── prepass/`  
`│   │   ├── __init__.py`  
`│   │   ├── ghost_v1.py               # Rembg + SegFormer + LaMa inpainting`  
`│   │   ├── ghost_v2.py               # Flux Kontext empty-dress rewriting`  
`│   │   ├── sheer_fill.py             # Enclosed hollow (24, 24, 24) pre-render fill`  
`│   │   └── graphic_extractor.py      # Logo/print plate extraction & preserving`  
`│   ├── reconstruction/`  
`│   │   ├── __init__.py`  
`│   │   └── hunyuan_client.py         # Replicate/Modal multi-view shape caller`  
`│   ├── geometry/`  
`│   │   ├── __init__.py`  
`│   │   ├── sdf_collision.py          # Voxel flood-fill SDF & clearance solver`  
`│   │   ├── garment_deformer.py       # Fitted open-shell extractor & Laplacian align`  
`│   │   └── mesh_utils.py             # Safe vertex merge, component labeling, normal repairs`  
`│   ├── texturing/`  
`│   │   ├── __init__.py`  
`│   │   ├── projection.py             # 4-view orthographic camera normal-weighted bake`  
`│   │   └── cylindrical_unwarp.py     # Seam-gap cylindrical flank unwrap`  
`│   ├── cloud/`  
`│   │   ├── __init__.py`  
`│   │   └── modal_app.py              # Modal A10G worker for Hunyuan-Paint v2.0`  
`│   └── pipeline.py                   # Master End-to-End Orchestrator`  
`├── scripts/`  
`│   ├── make_canonical_views.py       # CLI: 2D Pre-pass runner`  
`│   ├── run_fitting.py                # CLI: 3D Shape + SDF collision runner`  
`│   └── build_outfit.py               # CLI: End-to-end full outfit builder`  
`└── tests/`  
    `├── test_sdf_collision.py`  
    `├── test_mesh_invariants.py`  
    `└── test_prepass_routing.py`

## **8\. Key Upgrades & Diagnostic Context**

> * **Ghost v1 vs. Ghost v2 Paradigm:** Explicitly splits out the handling of opaque vs. sheer/printed fabrics. Treating a sheer leopard print the same as an opaque satin dress will cause the pipeline to fail.  
> * **Sheer-Fill Prepass:** Includes the exact RGB value (24, 24, 24\) to fill hollows, directly solving the issue where Hunyuan tunnels through sheer midriffs.  
> * **Cylindrical Bakes:** Planar wraps fail on the sides. Utilizing cylindrical 4-view bakes drastically improves texture troubleshooting.  
> * **SDF Clearances & Multi-Garment Layering:** Updated collision rules prioritize Signed Distance Fields (SDF) with a 4mm push, and detail how to stack a hoodie over jeans without mesh intersection.  
> * **API Routing:** Replicate is utilized for raw Hunyuan-2mv shape and Flux image rewriting; Modal is strictly needed for the custom CUDA rasterizer required by Hunyuan-Paint; Hugging Face is leveraged for SegFormer weights.