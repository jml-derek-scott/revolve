# 3D Viewer: Product Photos to 3D Outfits

Turn a garment's ordinary product photos (one front shot and one back shot) into an interactive 3D model
that shoppers can spin around, shown on a clean, faceless mannequin under customizable HDRI lighting environments.

> **Current State:** The project is a hybrid prototype and exploration sandbox. Most early models were built locally on a Mac Pro (AMD GPU, PyTorch MPS), while recent high-quality multi-view and single-view generation experiments run on cloud GPUs (**Modal**, **Replicate**, and **Hugging Face**).

Template 3, texture wrap, in the two lighting environments:

| ex1 · Beach | ex2 · Nightclub |
| :---: | :---: |
| <img src="_docs/ex1.png" alt="Template 3 texture wrap in the Beach environment" width="460"> | <img src="_docs/ex2.png" alt="Template 3 texture wrap in the Nightclub environment" width="460"> |

---

## 1. Executive Summary & Codebase Architecture

The app is structured into a static demo viewer, a FastAPI service, offline CLI build scripts, and cloud GPU workers:

* **Frontend (`frontend/`):** Static web viewer (no build step) powered by **Three.js** and Google's `<model-viewer>`. Displays 5 real product templates across three build methods (*Texture wrap*, *Full build*, and *Hybrid*) in two relightable HDRI environments (*Beach* daylight and *Nightclub* low-tint light).
* **Backend (`backend/app/`):** FastAPI server serving the frontend UI, static model assets, and an `/api/generate` endpoint (runs local TripoSR single-view reconstruction or dispatches multi-view requests).
* **Pipeline & Scripts (`scripts/` & `pipeline/`):** Python scripts for processing images, running cloud reconstruction, extracting garments, registering meshes onto the mannequin, and baking photo textures onto UV maps.
* **Cloud GPU Worker (`cloud/modal_app.py`):** Modal serverless container running **Hunyuan3D-2mv** shape generation and **Hunyuan3D-Paint** texture generation on NVIDIA A10G GPUs with persistent Hugging Face volume caching.

---

## 2. Core Model Comparison: TRELLIS vs. Hunyuan3D vs. Hybrid

We evaluated multiple 3D reconstruction paradigms to turn 2D photos into 3D GLB assets:

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

### A. Texture Wrap (Photo Projection on Fixed Mannequin)
* **How it works:** Projects 2D front/back photos onto `mannequins/female_base_mesh.glb` using `u2net_cloth_seg` clothing segmentation masks and orthographic depth buffers (~10 s on CPU).
* **Pros:** Sharp real-photo prints, fast, clean mannequin body.
* **Cons:** **Zero garment shape.** Loose skirts, flared hems, and coats are painted directly onto the mannequin's legs.

### B. Microsoft TRELLIS.2 (Single-View Cloud Build)
* **How it works:** Single-image reconstruction model running on a cloud GPU (via Hugging Face Spaces / Replicate) in ~45 s.
* **Pros:** Exceptionally clean geometry, realistic cloth drape, and high-resolution baked UV textures.
* **The Key Issue — TRELLIS "invents" the back:** Because TRELLIS.2 is strictly single-view, it only accepts the front photo and **hallucinates/invents the back of the garment**. E-commerce garments have specific back details (zippers, keyholes, cutouts, prints) that must match the real product, so inventing the back is non-viable for production.

### C. Tencent Hunyuan3D-2 / Hunyuan3D-2mv (Multi-View Cloud Build)
* **How it works:** Native multi-view 3D reconstruction running on Modal A10G GPUs (`cloud/modal_app.py`) or Replicate.
* **Pros:** **Multi-view native.** Directly accepts both the front AND back product photos as conditioning inputs, honoring actual product features on both sides.
* **The Key Issue — Visible Seam Artifacts:** Hunyuan3D-2 creates **severe texture and geometry seams** along the side flanks (90° rib/hip boundaries) where the front and back multi-view diffusion projections merge.

### D. The Hybrid Approach (Garment Shape + Mannequin Body + Photo Projection)
* **How it is supposed to work:**
  1. Take the 3D garment geometry and realistic drape from a reconstructed 3D model (e.g. Hunyuan3D-2 or TRELLIS).
  2. Cut out the garment geometry and fit/align it onto the clean faceless mannequin (`female_base_mesh.glb`).
  3. Layer/project high-resolution, sharp 2D photo textures from the original product shots over the finished 3D garment mesh.
* **Why it is currently struggling / not working cleanly:**
  * Garment extraction and torso-alignment often leave side gaps or mesh clipping against the mannequin torso.
  * **Texture projection onto the 3D garment is weak:** Flat 2D photo projection onto 3D curved geometry leaves blank unpainted voids at 90° grazing angles (under arms, lateral hips). Basic algorithmic inpainting (OpenCV TELEA) creates blurry smudges and streaking across the side seams instead of continuous fabric patterns.

---

## 3. Major Struggles & Current Weak Areas

1. **Weak Texture Generation:**
   * AI-baked textures from models like Hunyuan3D-Paint or InstantMesh are blurry, low-resolution, and suffer from artificial lighting/shadow bake-in and skin-color bleeding.
   * Direct 2D photo projection onto 3D meshes lacks 90° side-angle profile information.
2. **Side Seams & Grazing Angles:**
   * Standard front and back product shots do not capture the sides of the garment under the arms or around the hips.
   * Texels where surface normals are perpendicular to the camera ($|n_z| \le 0.25$) become unassigned voids. Basic inpainting smudges colors rather than continuing patterns.
3. **Local Mac Hardware Constraints:**
   * Desktop Mac AMD GPUs (PyTorch MPS) lack CUDA kernel support (e.g. `nvdiffrast`), forcing heavy 3D steps (FlexiCubes, Zero123++ diffusion, marching cubes) onto CPU.

---

## 4. Providers Used

* **Modal (`modal.com`):** Serverless A10G GPU infrastructure (`cloud/modal_app.py`) running Hunyuan3D-2mv shape generation and Hunyuan3D-Paint texture generation, backed by a persistent `modal.Volume` for caching Hugging Face model weights across cold starts.
* **Replicate:** Cloud API hosting `tencent/hunyuan3d-2mv`, `fishwowater/trellis2`, and `allen-act/lama` inpainting.
* **Hugging Face:**
  * **HF Hub:** Stores base model weights downloaded via `huggingface_hub.snapshot_download` with `HF_TOKEN`.
  * **HF Spaces:** Direct HTTP/SSE client interfacing with the public `microsoft/TRELLIS.2` Space endpoint.

---

## 5. Things to Improve & Future Research (Leveraging Hugging Face)

We believe **Hugging Face** offers massive potential to solve our current bottlenecks in texture generation, model hosting, and pipeline evaluation:

### 1. Hugging Face Dedicated Inference Endpoints
* Deploy custom multi-view reconstruction and texture-generation containers onto **Hugging Face Dedicated Inference Endpoints** (NVIDIA A10G / A100 GPUs).
* Replaces custom Modal code or third-party wrappers with native auto-scaling infrastructure, direct API keys, and enterprise reliability.

### 2. Hugging Face Spaces for Prototyping & Benchmarking
* Continuously evaluate new open-source 3D and garment models as soon as they drop on HF Spaces (e.g., LGM, Era3D, CRM, Tripo3D, Marigold depth estimators, Clothed3D).
* Build an automated test suite that queries new HF Spaces and compares generated GLB geometry against our standard benchmark products.

### 3. Diffusion-Based Texture Synthesis via `diffusers`
* Replace basic OpenCV TELEA inpainting with Hugging Face **`diffusers` pipelines** (e.g., SDXL Inpainting, Flux Inpainting, ControlNet).
* Project front and back photos onto the 3D garment mesh, then pass unpainted side-seam zones through a controlled Diffusion Inpainting pipeline to synthesize seamless, realistic fabric continuation across the 90° lateral flanks.

### 4. Fine-Tuning Open Multi-View Models on HF Hub
* Upload our curated e-commerce garment dataset (front/back photos + ground-truth 3D meshes) to **Hugging Face Datasets**.
* Fine-tune open multi-view diffusion models (such as Hunyuan3D-2 or Zero123++) to eliminate lateral seam artifacts on front/back clothing pairs and host the fine-tuned weights on the Hugging Face Model Hub.

---

## 6. Running the App Locally

```bash
./run.sh          # starts FastAPI + static viewer on http://127.0.0.1:8000
make run          # same, via Poetry
```

Or manually:

```bash
poetry run uvicorn backend.app.main:app --port 8000
```

### Offline Hybrid PoC Script

To run the offline hybrid reconstruction pipeline:

```bash
# Using Replicate as the cloud provider
poetry run python scripts/run_hybrid_poc.py \
  --front frontend/assets/templates/1-front.jpg \
  --back  frontend/assets/templates/1-back.jpg \
  --provider replicate \
  --out frontend/assets/models/1-hybrid.glb

# Using local mesh (no cloud GPU)
poetry run python scripts/run_hybrid_poc.py \
  --front frontend/assets/templates/1-front.jpg \
  --back  frontend/assets/templates/1-back.jpg \
  --provider skip \
  --build frontend/assets/models/1-build.glb \
  --out frontend/assets/models/1-hybrid.glb
```

---

## 7. Project Layout

| Path | Description |
|---|---|
| `frontend/` | Static demo UI (`index.html`, `js/app.js`, `js/three-viewer.js`, `js/mvp-info.js`) |
| `frontend/assets/models/` | Pre-built GLB models (`<template>-wrap.glb`, `-build.glb`, `-hybrid.glb`) |
| `backend/app/` | FastAPI server (`main.py`, `api/routes.py`, `services/`) |
| `pipeline/` | Hybrid pipeline modules (`preprocess.py`, `reconstruct.py`, `garment_extractor.py`, `texture_bake.py`, `export_glb.py`) |
| `cloud/` | Modal serverless GPU app (`modal_app.py`) |
| `scripts/` | CLI test & build scripts (`test_mannequin.py`, `test_instantmesh.py`, `build_hybrid.py`, `run_hybrid_poc.py`) |
| `mannequins/` | Premade mannequin mesh assets (`female_base_mesh.glb`) |

For technical architectural diagrams, see **[AGENTS.md](AGENTS.md)**. For low-level mesh gotchas and developer details, see **[CODE.md](CODE.md)** and **[CLAUDE.md](CLAUDE.md)**.
