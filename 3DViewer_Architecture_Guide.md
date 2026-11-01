# 3DViewer: Lead Architect Implementation Guide & Design Limits

This document outlines the strict production guidelines, architectural limits, and modular pipeline requirements for transitioning the 3DViewer engine from monolithic body extraction to a robust, skeletally animated multi-class system.

## 1. The Core Paradigm: Multi-Class Modular Assemblies
We no longer extract a single watertight 3D body from Hunyuan3D-2mv. E-commerce apparel must drape correctly on a rigged mannequin. To achieve this, the pipeline enforces a **Ghost Mannequin + Boundary Capping** workflow.

### Architectural Steps:
1. **2D Segmentation & Layer Separation:** Run `rembg` and SegFormer to isolate distinct clothing elements (e.g., upper `hoodie`, lower `jeans`) and remove skin, hair, and footwear.
2. **Modular 3D Shape Reconstruction:** Send isolated 2D masks to Hunyuan3D-2mv (or TRELLIS for single-view) to generate independent garment shells (one for the hoodie, one for the jeans).
3. **Manifold Boundary Capping:** Do not use simple depth buffer cuts (`garment_extractor.py`) which produce razor-thin, single-sided polygon tearing. Detect open non-manifold boundary loops (cuffs, hems, waistbands) and extrude inward rims.
4. **Skeletal Weight Transfer (LBS):** Transfer `JOINTS_0` and `WEIGHTS_0` via $k$-Nearest Neighbor inverse-distance interpolation from the base mannequin (`mannequins/female_base_mesh.glb`) to the extracted garment shells.

## 2. Strict Hardware & Mesh Constraints
The development environment is a Mac Pro using PyTorch MPS (`torch.device("mps")`).

**MANDATORY RULES:**
* **No Custom CUDA Kernels:** `nvdiffrast` and `torchmcubes` will not compile on MPS. Heavy diffusion and marching cubes must stay on cloud A10G workers (Modal or Replicate).
* **Do NOT Use Ray-Casting:** The `female_base_mesh.glb` contains a double-skinned 5mm shell with inverted normals. Methods like `trimesh.contains()` or ray intersection will fail. You **must** use orthographic 2D depth buffers ($z_{\max}$, $z_{\min}$) to check mesh containment and compute depth offsets.
* **Vertex Merging Before Topology Operations:** AI reconstructors split vertices along UV bounds. Always call `mesh.merge_vertices(merge_tex=True, merge_norm=True)` on a cloned copy before running connected component analysis.
* **No `trimesh.split()`:** It clones 4K texture data thousands of times per fragmented face, crashing the Mac's RAM. Use `trimesh.graph.connected_component_labels(mesh.face_adjacency)` to filter masks.

## 3. Resolving Side-Seam Tears & Grazing Angles
The fundamental math behind the 17% unpainted side flank (blind spots) is due to orthographic camera projection missing grazing angles. 

* For normal vector $N = [n_x, n_y, n_z]$, the facing threshold is $|n_z| > 0.05$.
* Because 16% of a human torso lies at a grazing angle ($|n_z| < 0.25$), linear 2D photo projection leaves blank vertical strips under the arms and along the hips.
* **Fix:** Use continuous **cylindrical UV unwrapping** `cylindrical_unwrap()` combined with bounded `_project_px_with_silhouette_warp` for the waist and hip regions, and rely on multi-view AI texture baking (`Hunyuan3D-Paint` on Modal) to synthesize the missing side textures natively, rather than smudging with OpenCV TELEA.

## 4. Required Cloud Infrastructure
* **Modal (modal.com):** A10G serverless execution for `Hunyuan3D-2mv` geometry and `Hunyuan3D-Paint` texturing. Ensure the HF cache volume is pre-fetched.
* **Replicate:** Fallback API for LaMa inpainting (`allenhooo/lama`) and secondary shape generation.
* **Hugging Face Spaces:** API integration for bleeding-edge single-view testing (e.g., TRELLIS.2), though single-view is prohibited for final catalog use since it hallucinates the back.
