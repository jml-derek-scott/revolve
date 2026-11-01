# **Cursor Task Prompt: Production Pipeline Refactoring for Decoupled 3D Garment Generation & Collision Handling**

You are acting as an elite 3D Graphics and Computer Vision Engineer. Your objective is to refactor and upgrade our current 2D-photo-to-3D-garment pipeline (pipeline/ and scripts/). We are shifting away from carving fused human meshes or applying crude orthographic depth pushes to a **Decoupled 3D Garment Architecture** inspired by Garment3DGen, ReFU, DMap, and ControlNet multi-view diffusion.

## **1\. System Context & Architectural Goals**

* **Problem:** Our existing approach in pipeline/garment\_extractor.py and scripts/run\_hybrid\_poc.py uses basic depth-buffer pushes and flat photo projections. This causes body poke-through, tears along garment hemlines, and severe blurry seam artifacts across the 90° grazing flanks under the arms and lateral hips.  
* **Goal:** Build an end-to-end, modular production pipeline that generates clean, unposed 3D garment meshes with preserved open topologies (arm, neck, and waist openings), registers them to mannequins/female\_base\_mesh.glb using a continuous SDF Repulsive Force solver\[cite: 8, 9, 13\], and inpaints missing side-seam textures using depth/normal-guided diffusion\[cite: 7, 8\].

## **2\. Technical Directives & Module Implementation Specs**

### **Module 1: Preprocessing & Canonical View Isolation (pipeline/canonical\_isolation.py)**

Replace ad-hoc calls in pipeline/preprocess.py with a unified isolation class:

1. Load front and back catalog photos (front.jpg, back.jpg).  
2. Use **Flux Kontext** (black-forest-labs/flux-kontext-pro) or **SegFormer** (mattmdjaga/segformer\_b2\_clothes) combined with **LaMa** to isolate the dress/top/bottom on a solid white background (\#FFFFFF).  
3. Extract clothing bounding boxes, calculate the bust-to-hem ratio, and output isolated normalized PNGs (outputs/canonical/front\_clean.png, back\_clean.png) and placement.json.

### **Module 2: Topology-Preserving Base Mesh Deformation (pipeline/garment\_deformer.py)**

Replace whole-person generation models (e.g., standard TRELLIS or fused Hunyuan meshes) with a template deformation solver inspired by Garment3DGen:

1. **Template Selection:** Load an unposed canonical base garment mesh ${M}_{base}$ (e.g., dress, top, or skirt template from mannequins/templates/) with clean UVs and open boundary loops.  
2. **Pseudo-GT Guidance:** Call Hunyuan3D-2mv (cloud/modal\_app.py or Replicate API) on the isolated front/back photos to generate a coarse shape target ${M}_{guide}$.  
3. **Jacobian Mesh Deformation Solver:**  
   * Parameterize ${M}_{base}$ using per-triangle Jacobians ${J}_{i}\in {R}^{3\times 3}$ initialized to identity.  
   * Optimize ${J}_{i}$ via Poisson surface deformation using a weighted multi-term PyTorch loss:

   * ${L}_{total}={w}_{CD}{L}_{CD}+{w}_{Lap}{L}_{Lap}+{w}_{triag}{L}_{triag}+{w}_{2D}{L}_{2D}+{w}_{E}{L}_{E}$  
   * ${L}_{CD}$: One-directional Chamfer Distance from sampled points on ${M}_{base}$ to ${M}_{guide}$.  
   * ${L}_{Lap}$: Differential Laplacian smoothing to preserve uniform vertex density.  
   * ${L}_{triag}$: Penalty on inverse triangle areas to prevent degenerate/collapsed faces.  
   * ${L}_{2D}$: Multi-view silhouette L1 loss between differentiable renders of ${M}_{base}$ and ${M}_{guide}$.  
   * ${L}_{E}$: FashionCLIP cosine distance loss between rendered views and input product photos.  
4. Export the resulting unposed, manifold garment mesh ${M}_{garment}$ preserving original neck, arm, and hem topology.

### **Module 3: SDF Repulsive Force Unit (ReFU) Collision Solver (pipeline/sdf\_collision.py)**

Completely replace the primitive depth-buffer offset in fit\_ghost\_garment with a PyTorch/Trimesh implementation of ReFU:

1. **SDF Precomputation:** Precompute a high-resolution 3D grid Signed Distance Field $SDF(x)$ and continuous normalized gradient field $\nabla SDF(x)=\frac{\nabla f(x)}{\parallel \nabla f(x)\parallel }$ for mannequins/female\_base\_mesh.glb\[cite: 8, 13, 14\].  
2. **Dynamic Repulsion Loop:**  
   * Place ${M}_{garment}$ onto the mannequin mesh.  
   * For each garment vertex ${v}_{i}$: query $SDF({v}_{i})$.  
   * If $SDF({v}_{i})<\epsilon$ (where clearance $\epsilon =0.004m$ / 4mm)\[cite: 5, 8, 13\]:

   * ${v}_{i}\leftarrow {v}_{i}+{\alpha }_{i}\cdot (\epsilon -SDF({v}_{i}))\cdot \nabla SDF({v}_{i})$  
   * where ${\alpha }_{i}\geq 1.0$ is a localized scale factor to prevent edge-edge (EE) collisions.  
3. Apply 2-3 iterations of light constrained Laplacian smoothing on displaced vertices to keep cloth drapes fluid and wrinkle-smooth\[cite: 8, 13\].

### **Module 4: Multi-View Flank Diffusion Inpainting (pipeline/flank\_inpainting.py)**

Replace simple TELEA blurring (pipeline/texture\_bake.py) with ControlNet-guided diffusion texture inpainting across the 90° grazing flanks\[cite: 7, 8\]:

1. **Orthographic Projection:** Project catalog front ($+Z$) and back ($-Z$) photos onto ${M}_{garment}$ UV atlas.  
2. **Grazing-Angle Masking:** Mark texels where surface normal $z$\-component \$\\vert{}n\_z\\vert{} \\le 0.25\$ or occlusion depth masks fail as unpainted/void.  
3. **Lateral View Generation:** Render depth and normal maps from lateral side cameras ($+X$ left profile, $-X$ right profile).  
4. **Diffusion Inpainting:** Pass the unpainted lateral regions through diffusers StableDiffusionControlNetInpaintPipeline (using depth/normal ControlNets) conditioned on the garment text prompt (e.g., "seamless leopard mesh pattern continuation")\[cite: 4, 8\].  
5. Bake synthesized lateral textures back onto the main 2K UV atlas to yield a completely continuous, artifact-free GLB export.

## **3\. Step-by-Step Deliverables Requested from Cursor**

1. **pipeline/sdf\_collision.py**: A clean PyTorch module implementing the SDFCollisionSolver class with precomputed grid interpolation for female\_base\_mesh.glb\[cite: 2, 8, 13\].  
2. **pipeline/garment\_deformer.py**: A Jacobian-based deformation class GarmentDeformer taking template mesh ${M}_{base}$ and guide mesh ${M}_{guide}$ to output an open-topology garment.  
3. **pipeline/flank\_inpainting.py**: The ControlNet depth-guided inpainting pipeline for filling side-seam texture voids.  
4. **Refactored scripts/run\_hybrid\_poc.py**: Update CLI arguments and pipeline execution flow to link Stage 1 \-\> Stage 2 \-\> Stage 3 \-\> Stage 4 into a single seamless run.

Maintain complete type hints, clear docstrings, error handling for local GPU/CPU execution, and integration with Trimesh and PyTorch. Proceed with code implementation

