# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Image → 3D mesh (GLB) generator with a static web viewer. Not a git repo; part of the `~/code` workspace (unrelated to the JewelML services there). Targets a Mac Pro with an AMD GPU using PyTorch **MPS** (not CUDA).

## Setup and running

- Poetry owns the main env (`.venv/`, in-project). InstantMesh stays on `venv-instantmesh/`. There is no test suite or linter configured.
- Install: `make install` (needs Poetry + Python 3.11). `torchmcubes` is built against the installed torch as a post-step (`--no-build-isolation`). TripoSR is cloned into `vendor/TripoSR` (not pip-installable).
- Run the server from the repo root: `./run.sh` or `make run` (or `poetry run uvicorn backend.app.main:app --reload`; the package is imported as `backend.app`, and relative imports are used throughout). It serves the frontend at `/`, the API under `/api`, and generated GLBs at `/outputs/<job_id>.glb`.
- `poetry run python scripts/verify_mps.py` checks that MPS is usable. `poetry run python scripts/test_triposr.py <image> [mps|cpu]` is a standalone TripoSR smoke test, independent of the backend.
- Modal (cloud GPUs): `make modal-setup` (or `poetry run python -m modal setup`). Then `make modal-secret` to create `huggingface-secret` from `HF_TOKEN` in `.env`, `make modal-deploy` / `make modal-prefetch` for Hunyuan3D-2mv. Package is declared in `pyproject.toml`; do not `pip install modal`. Hybrid PoC CLI: `poetry run python scripts/run_hybrid_poc.py --front … --back … --provider replicate|modal|hf|skip`.
- `scraper/` holds Playwright scripts that download product images (e.g. Revolve) into `scraper/revolve_images/`; they are one-off scripts with hardcoded URLs.

## Architecture

Request flow: `POST /api/generate` (multipart: `front`, optional `back`) → [routes.py](backend/app/api/routes.py) saves uploads to `uploads/<job_id>/` → `pipeline.run` in a threadpool → GLB written to `outputs/<job_id>.glb` → response returns `glb_url`, which the frontend loads. `GET /api/health` reports the torch device.

Reconstructors live in `backend/app/services/` and subclass `Reconstructor` (`base.py`: `generate(images, out_path) -> Path`, `get_device()` prefers MPS, else CPU). `pipeline.select_reconstructor` picks by image count: 1 → `SingleViewReconstructor`, 2 → `MultiViewReconstructor`. Both are module-level singletons so models load once.

- `single_view.py`: real TripoSR. Lazy-loads the model on first request, serializes inference with a lock, adds `vendor/TripoSR` to `sys.path`, and runs marching cubes on CPU because `torchmcubes` has no MPS support. Output is vertex-coloured (no texture baking).
- `multi_view.py`: independent native multi-view scaffold (no TripoSR dependency). Real rembg cut-out + shared-scale front/back alignment; model loading and inference are TODO stubs that raise `NotImplementedError`. `mesh_utils.to_y_up(mesh, convention)` converts z-up outputs to glTF y-up. `NotImplementedError` from a reconstructor surfaces as HTTP 501.

`config.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and must be imported before torch is used anywhere so unsupported MPS ops fall back to CPU. `TRIPOSR_MC_RESOLUTION` (env, default 256) controls mesh resolution. Uploads are limited to 15 MB of PNG/JPEG/WebP.

Frontend (`frontend/`, no build step): a demo picker, not an uploader. `js/app.js` lists templates 1-5 (`assets/thumbs/`, source photos in `assets/templates/`) and loads pre-built `assets/models/<n>-wrap.glb` (mannequin projection, `scripts/test_mannequin.py`) or `<n>-build.glb` (InstantMesh, `scripts/test_instantmesh.py`; only 1-3 exist, see `HAS_BUILD`; template 1's is a TRELLIS.2 cloud build) or `<n>-hybrid.glb` (`scripts/build_hybrid.py`: garment cut from a full build onto the mannequin; template 1 only, see `HAS_HYBRID`) into model-viewer or `js/three-viewer.js`. Styled with Tailwind v4 (browser CDN build, `@apply` components in a `text/tailwindcss` style block in `index.html`) and Boxicons; the section info modals are a native `<dialog>` filled from `INFO` in `app.js`. Served statically by FastAPI; `/api/generate` still exists but the UI no longer calls it.

The design plan is in `_docs/` (a .txt and .docx of the same content).

Mesh gotchas (cost real debugging time): `mannequins/female_base_mesh.glb` is a thick double-skinned shell (outer + inner skin ~5 mm apart, joined) with scrambled face winding, so face/vertex normals and ray-parity inside tests are wrong on it. TRELLIS meshes also have unreliable normals and duplicate vertices along UV seams (weld before computing connectivity), and `trimesh.split()` on them can exhaust memory (it copies the texture per component). `build_hybrid.py` therefore uses depth maps, not normals.
