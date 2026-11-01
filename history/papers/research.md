# **3D Garment Virtual Try-On: Codebase Expansion & Research Implementation**

## **ControlNet Research Papers in 3D Garment Reconstruction**

Below are the direct links to the relevant research papers demonstrating the integration of ControlNet into 3D garment pipelines:

> * **Adding Conditional Control to Text-to-Image Diffusion Models** (ControlNet Foundation): [https\://arxiv.org/html/2302.05543v3](https://arxiv.org/html/2302.05543v3)  
> * **Garment3DGen: 3D Garment Stylization and Texture Generation**: [https\://arxiv.org/html/2403.18816v3](https://arxiv.org/html/2403.18816v3)  
> * **Cloth2Tex: A Customized Cloth Texture Generation Pipeline for 3D Virtual Try-On**: [https\://arxiv.org/abs/2308.04288](https://arxiv.org/abs/2308.04288)  
> * **DressCode: Autoregressively Sewing and Generating Garments from Text Guidance**: [https\://arxiv.org/html/2401.16465v4](https://arxiv.org/html/2401.16465v4)

## **Strategic Codebase Expansion**

Based on the recent pipeline evaluations, here is how you can directly improve your current codebase by resolving the bottlenecks identified across your five templates.

### **1\. Standardize Sheer and Mesh Fabric Handling**

> * **The Problem:** When processing the garment in Template2.md, the remove\_background: true parameter caused the Hunyuan3D-2mv endpoint to treat sheer regions on a white background as empty space, creating topological tunnels through the torso \[cite: 4, 7\]. Furthermore, the SegFormer model in Template1.md incorrectly labeled the sheer leopard mesh as skin \[cite: 5\].  
> * **The Solution:** Never pass raw background removal commands to the reconstructor for sheer items \[cite: 7\]. Instead, run the sheer\_prepass.py script to composite the garment onto a neutral gray canvas (\#808080) before making the Replicate calls \[cite: 4, 7\]. For patterned sheer items, utilize Flux Kontext to rewrite the catalog photo rather than attempting a ghost mask cut \[cite: 5\]. Opaque garments, however, should continue using the 2D isolation mask approach defined in Template4.md \[cite: 2\].

### **2\. Migrate from Monolithic Meshes to Multi-Shell Architecture**

> * **The Problem:** As detailed in Garment 3D Virtual Try-On Pipeline Failure Analysis & Architecture Recommendations.md, a single 80k watertight mesh cannot accurately represent layered outfits \[cite: 7\]. In Template5.md, the overlapping hoodie and jeans cannot be effectively rigged as a single solid \[cite: 1\]. Similarly, the three independent layers of the dress in Template2.md collapse when merged \[cite: 4, 7\].  
> * **The Solution:** Transition to a multi-shell architecture \[cite: 7\]. You should deploy the build\_male\_outfit.py logic from Template5.md to generate modular shells \[cite: 1\]. Enforce layered SDF clearance by sequentially resolving collisions between the garment and the body, and then between the overlapping garments \[cite: 1\]. Inner undergarments should derive their geometry from a fitted shell based on the mannequin, while outer draped layers should utilize the canonical guide mesh \[cite: 7\].

### **3\. Upgrade UV Baking and View Coverage**

> * **The Problem:** The planar 4-view bake used on the slip in Template1.md left white holes and texture smearing on the flanks \[cite: 5, 6\]. Additionally, front-only reconstructors will incorrectly invent closed backs for open garments, as was the case before the chartreuse halter integration in Template3.md \[cite: 3\].  
> * **The Solution:** Update pipeline/texture\_bake.py to default to cylindrical\_unwrap for dresses and skirts \[cite: 7\]. The t1-sidefill-handoff.md file confirms that switching to a cylindrical bake achieved 96% surface coverage with zero TELEA inpainting required \[cite: 6\]. Always ensure the back photo is fed to the reconstructor to maintain correct open-back geometries, as highlighted in Template3.md \[cite: 3\].

### **4\. Implement Procedural Details for Thin Geometry**

> * **The Problem:** Diffusion-based voxel grids (specifically the 384³ octree occupancy grid) completely erased the 1-2px spaghetti straps on the dress in Template2.md \[cite: 4, 7\].  
> * **The Solution:** Treat thin straps as a deterministic post-processing step rather than a generative one, as advised in Garment 3D Virtual Try-On Pipeline Failure Analysis & Architecture Recommendations.md \[cite: 7\]. Add the generate\_procedural\_straps() script to run\_hybrid\_poc.py to explicitly generate 3D quad strips anchored between the mannequin's shoulders and the garment's bust points \[cite: 7\].

### **5\. Resolve Immediate UI Blockers**

> * **The Problem:** According to t1-sidefill-handoff.md, the front-end application becomes stuck on the Template 2 view because the modelUrl() function allows the ?src= query parameter to persist in the address bar \[cite: 6\].  
> * **The Solution:** Edit frontend/js/app.js to fix this issue \[cite: 6\]. Ensure the application either drops the src parameter from the URL when a user selects a new template, or explicitly ignores it once the state.template or state.method state variables change \[cite: 6\].