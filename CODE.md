# 3DViewer — Technical Architecture & Codebase Guide

## 1. Executive Summary & Goal
The **3DViewer** application turns 2D e-commerce product photos (1 front photo + 1 back photo) into interactive 3D GLB models displayed on a clean, faceless mannequin under customizable HDRI lighting environments (Three.js and Google `<model-viewer>`).

### Hardware & Environment Constraints
* **Primary Development Environment:** Mac Pro with AMD GPU running PyTorch **MPS** (`torch.device("mps")` with `PYTORCH_ENABLE_MPS_FALLBACK=1`).
* **Hardware Limit:** PyTorch MPS lacks support for custom CUDA kernels (e.g., `nvdiffrast`, `torchmcubes` GPU acceleration). Marching cubes and multi-view diffusion steps fallback to CPU or run on cloud CUDA workers.
* **Architecture Split:**
  * **Static Demo (Frontend):** Pre-built models (`frontend/assets/models/<template>-{wrap|build|hybrid}.glb`) served statically via FastAPI.
  * **Live API (Backend):** `POST /api/generate` runs live single-view reconstruction via TripoSR; multi-view reconstruction routes to cloud workers or local scaffolds.
  * **Offline CLI Pipelines (`scripts/` & `pipeline/`):** Cloud reconstruction, garment extraction, mannequin alignment, and photo-texture baking scripts.

---

## 2. Summary of Evaluated 3D Models & Reconstructors

| Model | Type | Provider / Runtime | Strengths | Major Drawbacks / Key Issue |
|---|---|---|---|---|
| **TripoSR** | Single-View | Local (MPS/CPU via `vendor/TripoSR`) | Fast (~10s local), simple marching cubes on CPU. | Low resolution, vertex-color only, ignores back photo. |
| **Zero123++ v1.2 + InstantMesh** | Multi-View | Local (`venv-instantmesh/`) | Runs 100% locally on desktop Mac. | Extreme CPU bottleneck (~6.5 min/item), low mesh resolution, blurry textures. |
| **Microsoft TRELLIS.2** | Single-View | Cloud (Replicate / HF Spaces) | Clean geometry, sharp surfaces, fast (~45s). | **Invents the back:** Single-view only. It hallucinates back features rather than using the real back photo (unusable for e-commerce garments with specific zippers/cutouts). |
| **Tencent Hunyuan3D-2 / 2mv** | Multi-View | Cloud (Modal / Replicate) | **Multi-view native:** Accepts both front + back photos directly as input conditioning. | **Harsh Seam Issues:** Creates visible geometry and texture seams where front and back multi-view fusion projections meet along the lateral flanks (90° ribs/hips). |
| **Hunyuan3D-Paint** | Texture Gen | Cloud (Modal A10G) | Re-textures existing 3D meshes using front/back images. | Blurry textures, artificial shading bake-in, alignment mismatches on fitted meshes. |

---

## 3. The Four Model Generation Pipelines

```
                          ┌─────────────────────────────┐
                          │   Front & Back Photos       │
                          └──────────────┬──────────────┘
                                         │
       ┌────────────────┬────────────────┼────────────────┬────────────────┐
       ▼                ▼                ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Texture Wrap │ │ InstantMesh  │ │  TRELLIS.2   │ │ Hunyuan3D-2  │ │    Hybrid    │
│  (Mannequin) │ │  (Local CPU) │ │ (Cloud HF/Rep│ │(Cloud Modal) │ │ (PoC Pipeline│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │                │
       │ No 3D shape    │ ~6.5 min CPU   │ Single-view    │ Multi-view     │ Garment cut +│
       │ Photos projected│ Low res       │ Invents back   │ Seam artifacts │ Photo projection│
       ▼                ▼                ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  <n>-wrap.glb│ │ <n>-build.glb│ │ <n>-build.glb│ │ <n>-build.glb│ │<n>-hybrid.glb│
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### A. Texture Wrap (`scripts/test_mannequin.py`) — No 3D Reconstruction
* **Mechanism:** Projects 2D front and back photos onto the fixed UV map of `mannequins/female_base_mesh.glb`.
* **Segmentation:** `u2net_cloth_seg` extracts clothing masks from `rembg` person cutouts.
* **Occlusion & Projection:** Orthographic depth buffers ($z_{\max}, z_{\min}$) project visible texels. Grazing-angle texels ($|n_z| \le 0.05$) are left unpainted (staying mannequin off-white `[232, 229, 224]`).
* **Result:** Sharp prints and colours in ~10s, but **zero garment shape** (skirts and loose hems are painted directly onto the mannequin legs).

### B. Local Full Build (`scripts/test_instantmesh.py`)
* **Mechanism:** Zero123++ v1.2 generates 6 synthetic surrounding views; the back view is replaced with the real back photo cutout. InstantMesh LRM predicts triplanes, and FlexiCubes extracts a mesh on CPU.
* **Result:** ~6.5 minutes on CPU due to MPS memory limits; coarse geometry and vertex colors only.

### C. Cloud Full Build (TRELLIS.2 & Hunyuan3D-2)
* **TRELLIS.2:** Single-image reconstruction running on Hugging Face Spaces or Replicate in ~45s. High geometric quality, but **invents the back of the garment**, ignoring the actual back photo.
* **Hunyuan3D-2mv:** Multi-view shape generation running on Modal A10G (`cloud/modal_app.py`) or Replicate. Respects front and back photos, but **produces severe texture/geometry seam artifacts** along the side flanks where front/back projections merge.

### D. The Hybrid Pipeline (`pipeline/` & `scripts/run_hybrid_poc.py`)
* **Core Concept:** Combine the 3D drape/silhouette of a cloud build (Hunyuan3D-2 or TRELLIS) with the clean premade mannequin body, then layer sharp real-photo textures over the finished garment mesh.
* **Stages:**
  1. `pipeline/preprocess.py`: Extracts garment masks via `rembg` / `u2net_cloth_seg` and inpaints occlusions (hair, hands).
  2. `pipeline/reconstruct.py`: Calls cloud GPU provider (Modal, Replicate, TRELLIS) to obtain raw clothed GLB.
  3. `pipeline/garment_extractor.py`: Depth-projects mesh triangles against 2D garment masks, welds UV seams, applies morphological ring closing (dilate 12/erode 12 rings), aligns floor/height to the mannequin, and pushes interior garment vertices outward (+4 mm buffer).
  4. `pipeline/texture_bake.py`: Rasterizes UV coordinates, calculates front/back orthographic photo projections, and inpaints unassigned lateral side-seam texels via OpenCV TELEA.
  5. `pipeline/export_glb.py`: Assembles the mannequin and garment sub-meshes into a unified GLB with PBR materials.
* **Current Limitations:**
  * Garment extraction and torso alignment often leave side gaps or awkward mesh boundaries.
  * **Texture projection onto the 3D garment is weak:** Flat photo projection onto complex 3D drape creates unpainted side voids, stretched textures at grazing angles, and blurry TELEA seam smudges.

---

## 4. Current Struggles & Technical Bottlenecks

1. **Weak Texture Generation:**
   * AI texture generators (e.g. Hunyuan3D-Paint) output blurry, low-resolution textures with baked-in lighting, shadows, and unnatural skin bleeding.
   * Direct 2D photo projection onto 3D meshes lacks side-view (90° profile) information.
2. **Side Seams & Grazing Angles:**
   * Front and back camera angles do not capture the sides of the garment under the arms or around the hips.
   * Texels where surface normals are perpendicular to the camera ($|n_z| \le 0.25$) become unassigned voids. Algorithmically inpainting these areas with TELEA leads to unnatural color streaking on patterned fabrics (floral, lace, logos).
3. **TRELLIS Single-View vs. Hunyuan Multi-View Tradeoff:**
   * **TRELLIS.2** yields superior mesh fidelity but invents the back.
   * **Hunyuan3D-2** accepts both views but introduces visible boundary seam defects.
4. **Local Hardware Constraints:**
   * Mac Pro AMD GPU (MPS) lacks CUDA kernel support, forcing heavy 3D steps onto CPU and preventing local CUDA texture rasterization (`nvdiffrast`).

---

## 5. Critical Mesh Quirks (Read Before Editing Geometry Code)

* **Double-Skinned Mannequin Shell (`mannequins/female_base_mesh.glb`):** Has an outer and inner skin ~5 mm apart with scrambled face winding. Standard normals and ray-casting (`mesh.contains()`) fail. Always use 2D orthographic depth buffers ($z_{\max}, z_{\min}$).
* **Split Vertices on UV Seams:** AI-generated meshes (TRELLIS, Hunyuan) split vertices along UV seam edges. Always run `mesh.merge_vertices(merge_tex=True, merge_norm=True)` on temporary copies before evaluating face adjacency or topological components.
* **Never Call `trimesh.split()`:** Splitting textured AI meshes duplicates materials and 2K/4K texture maps thousands of times, exhausting RAM. Use `trimesh.graph.connected_component_labels` and update face masks instead.

---

## 6. Cloud Providers & Integrations

### A. Modal (`modal.com` / `cloud/modal_app.py`)
* **Role:** Serverless A10G GPU worker running Hunyuan3D-2mv shape generation and Hunyuan3D-Paint texture generation.
* **Setup:** Uses a persistent `modal.Volume("hf-model-cache")` mounted to `/root/.cache/huggingface` to avoid re-downloading multi-gigabyte checkpoints on cold starts.
* **CLI Execution:** `poetry run modal run cloud/modal_app.py --front <front.png> --back <back.png>`.

### B. Replicate (`pipeline/reconstruct.py`)
* **Role:** Zero-infrastructure API fallback for `tencent/hunyuan3d-2mv`, `fishwowater/trellis2`, and `allen-act/lama` (inpaint).
* **Token:** Set via `REPLICATE_API_TOKEN` in `.env`.

### C. Hugging Face (`pipeline/reconstruct.py` & `cloud/modal_app.py`)
* **HF Hub & Tokens:** `HF_TOKEN` in `.env` authenticates model weight downloads via `huggingface_hub.snapshot_download`.
* **HF Spaces API:** HTTP/SSE client interfacing directly with `microsoft/TRELLIS.2` Space endpoint.

---

## 7. Future Research & Leveraging Hugging Face

We can significantly expand our use of **Hugging Face** across infrastructure, model evaluation, fine-tuning, and texture generation:

1. **Hugging Face Dedicated Inference Endpoints:**
   * Instead of relying on custom Modal code or third-party Replicate wrappers, deploy dedicated Hugging Face Inference Endpoints (e.g. NVIDIA A10G / A100 instances) directly from Hugging Face model repositories.
   * Provides native auto-scaling, private networking, and direct container management with standard REST APIs.
2. **HF Spaces for Rapid Model Benchmarking:**
   * Proactively monitor and interface with emerging 3D reconstruction and garment generation Spaces on HF Hub (e.g. LGM, Era3D, CRM, Tripo3D, Marigold depth models, Clothed3D).
   * Benchmark new open-source models as soon as they are published without spending time on local CUDA setup.
3. **HF Hub as Model & Dataset Registry:**
   * Store and version fine-tuned pipeline weights, mannequin assets, and standardized e-commerce benchmark image sets directly on Hugging Face Datasets & Repositories.
4. **Diffusion-Based Texture Synthesis via `diffusers`:**
   * Replace basic OpenCV TELEA inpainting with Hugging Face `diffusers` pipelines (e.g., SDXL Inpainting, Flux Inpainting, ControlNet).
   * Project front/back photos onto the mesh, then use a controlled Diffusion Inpainting pipeline to synthesize natural fabric continuation across the 90° lateral side seams.
5. **Fine-Tuning Open Multi-View Models on HF:**
   * Fine-tune multi-view diffusion models (Hunyuan3D-2 or Zero123++) on structured e-commerce clothing photo datasets uploaded to HF, specifically tuning the model to eliminate lateral seam artifacts on front/back pairs.
