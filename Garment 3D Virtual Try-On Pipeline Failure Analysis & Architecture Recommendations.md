# **3D Garment Virtual Try-On Pipeline: Failure Analysis & Strategic Recommendations**

This document provides an executive diagnostic audit, technical root-cause analysis, and an actionable architectural roadmap following the Stage B pipeline failure on the layered black midi dress (SKU Template 2). Specific implementation code, mathematical formulations, and research paper integrations are detailed below to transition the pipeline into a production-grade 3D garment reconstruction system.

## **1\. Diagnostic Audit: Root Cause Analysis of Failure 1**

The Stage B failure was initially characterized as a shape/fitting error. However, geometric and topological inspection confirms that the collision solver (SDF energy \~6.8e-11) and UV baking performed correctly on the geometry provided. The failure was an **input contract and topological representation collapse** during cloud 3D reconstruction in Hunyuan3D-2mv.

| Symptom Observed | Apparent Failure Point | Actual Root Cause & Topological Mechanism |
| :---- | :---- | :---- |
| **White rectangle on upper chest / midriff** | UV unwrap or TELEA texture fill failure. | **Background Removal Tunneling:** Sheer black fabric on white background resulted in \#FFFFFF pixels. Hunyuan's remove\_background: True classified sheer regions as background, carving a topological tunnel (Euler char \-26) through the torso. |
| **Missing spaghetti straps** | Garment fitter / mesh decimation cutting thin geometry. | **Voxel/Grid Resolution Erasure:** 1-2px black straps do not survive background removal and a 384³ octree occupancy grid. Only 433 vertices existed above the bandeau pre-fitting. |
| **Lumpy hip/seat blob in profile** | SDF over-push / collision solver displacement. | **Single-Volume Hallucination:** Two-view Hunyuan generated an opaque, solid 3D capsule volume for a multi-layered, draped cloth garment. |
| **Strapless tube body** | Wrong mannequin placement or improper vertical lift. | **Single Mesh Monolithic Fusion:** One closed, watertight 80k shell cannot represent 3 separate layers (opaque bandeau, inner slip, sheer midi overlay with slit). |
| **Side texture smearing** | Missing left/right profile camera views. | **Planar Projection on Bad Topology:** Planar/orthographic bake projected front/back photos onto lumpy, incorrect flank geometry. |

## **2\. Key Architectural Recommendations**

### **Recommendation A: Input Contract & Sheer Fabric Handling**

Never pass raw remove\_background: True calls to Hunyuan/Replicate when processing sheer, lace, or mesh garments on plain white backgrounds. Sheer fabric mixes with white background pixels, triggering background deletion inside garment bounds.

> * **Neutral Underlay Prepass:** Composite the ideal garment photo onto a neutral gray (\#808080) background or a synthetic non-human ghost bust before calling multi-view 3D generators.  
> * **Explicit Alpha Masking:** Pass pre-computed alpha masks derived from SegFormer / LaMa rather than relying on automatic thresholding by 3D providers.

### **Recommendation B: Abandon Watertight Monoliths for Layered Outfits**

A single 80k watertight mesh is topologically incapable of modeling multi-layered outfits featuring slits, sheer overlays, and independent under-slips.

> * **Decoupled Open Shell Decomposition:** Transition to multi-shell representations. Use fitted\_shell (derived from the target mannequin's outer surface offset) for body-conforming undergarments/bandeaus, and category-template deformation for draped/slit overlays.  
> * **Pattern/Drape Path Integration:** For loose midi skirts with high slits, utilize flat 2D pattern coordinate methods (such as DMap or GarmentCode/Dress-1-to-3) rather than generic image-to-3D mesh generators.

### **Recommendation C: Procedural Spaghetti Strap Generation**

Sub-pixel or 1-2mm geometry (spaghetti straps) will never reliably survive diffusion-based 3D reconstruction and voxel grids. Treat thin straps as a deterministic post-processing step.

> * Generate 3D straps procedurally as quadratic Bézier curves or 2D quad strips anchored between bust points on the garment mesh and shoulder anchor points on the mannequin body.

### **Recommendation D: UV Atlas & Texture Projection Improvements**

> * Replace planar\_unwrap with cylindrical\_unwrap to ensure uniform texel density across lateral flanks.  
> * Incorporate ControlNet depth-guided diffusion inpainting for unpainted lateral voids instead of relying solely on TELEA island filling.

## **3\. Implementation Blueprint & Production Code**

### **1\. Procedural Spaghetti Strap Generator**

import numpy as np  
import trimesh

def generate\_procedural\_straps(  
    garment\_mesh: trimesh.Trimesh,  
    mannequin\_mesh: trimesh.Trimesh,  
    strap\_width\_m: float \= 0.006,  
    num\_samples: int \= 16  
) \-\> trimesh.Trimesh:  
    """Generates explicit 3D quad strips for thin spaghetti straps connecting   
    the garment bust line to mannequin shoulder anchor points."""  
      
    \# 1\. Identify left/right bust anchor points on the upper garment rim  
    v\_garment \= np.asarray(garment\_mesh.vertices)  
    upper\_mask \= v\_garment\[:, 1\] \> (v\_garment\[:, 1\].max() \- 0.05)  
    upper\_verts \= v\_garment\[upper\_mask\]  
      
    left\_bust \= upper\_verts\[np.argmin(upper\_verts\[:, 0\])\]  \# \-X side  
    right\_bust \= upper\_verts\[np.argmax(upper\_verts\[:, 0\])\] \# \+X side  
      
    \# 2\. Define corresponding shoulder anchor points on mannequin  
    v\_mann \= np.asarray(mannequin\_mesh.vertices)  
    shoulder\_y \= v\_mann\[:, 1\].max() \- 0.08  
    shoulder\_mask \= np.abs(v\_mann\[:, 1\] \- shoulder\_y) \< 0.02  
    shoulder\_verts \= v\_mann\[shoulder\_mask\]  
      
    left\_shoulder \= shoulder\_verts\[np.argmin(shoulder\_verts\[:, 0\])\]  
    right\_shoulder \= shoulder\_verts\[np.argmax(shoulder\_verts\[:, 0\])\]  
      
    straps \= \[\]  
    for start, end in \[(left\_bust, left\_shoulder), (right\_bust, right\_shoulder)\]:  
        t \= np.linspace(0, 1, num\_samples)\[:, None\]  
        \# Linear interpolation with slight outward arch  
        curve \= (1 \- t) \* start \+ t \* end  
        curve\[:, 2\] \+= np.sin(np.pi \* t.ravel()) \* 0.005 \# mild outward lift  
          
        \# Build quad ribbon  
        verts \= \[\]  
        faces \= \[\]  
        normal \= np.array(\[0.0, 0.0, 1.0\])  
        binormal \= np.array(\[1.0, 0.0, 0.0\]) \* (strap\_width\_m / 2.0)  
          
        for p in curve:  
            verts.append(p \- binormal)  
            verts.append(p \+ binormal)  
              
        for i in range(num\_samples \- 1):  
            v0, v1 \= 2 \* i, 2 \* i \+ 1  
            v2, v3 \= 2 \* (i \+ 1), 2 \* (i \+ 1\) \+ 1  
            faces.append(\[v0, v1, v3\])  
            faces.append(\[v0, v3, v2\])  
              
        straps.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))  
          
    return trimesh.util.concatenate(straps)

### **2\. Sheer Fabric Pre-processing & Reconstruct Integration**

def prepare\_sheer\_canonical\_input(image\_path: Path, mask\_path: Path, out\_path: Path) \-\> Path:  
    """Composites sheer/transparent garment regions onto a neutral gray canvas   
    to prevent background-removal tools from tunneling through white fabric."""  
    from PIL import Image  
    import numpy as np

    img \= Image.open(image\_path).convert("RGBA")  
    mask \= Image.open(mask\_path).convert("L")  
      
    \# Neutral gray background canvas (\#808080)  
    bg \= Image.new("RGBA", img.size, (128, 128, 128, 255))  
      
    \# Composite garment onto neutral canvas using garment segmentation mask  
    composite \= Image.composite(img, bg, mask)  
    composite.convert("RGB").save(out\_path)  
    return out\_path

## **4\. Integration with Research Literature**

> * **Dress-1-to-3 (2025):** Leverages 2D diffusion priors and differentiable CIPC cloth physics. Primary lesson: optimization of 2D sewing patterns against multi-view normal/RGB guidance produces simulation-ready separated garments.  
> * **DMap (SIGGRAPH 2025):** Maps single-view images to 2D pattern coordinates to reconstruct non-watertight, manifold garments without boundary tears or open holes.  
> * **ReFU (2022):** Uses 3D Signed Distance Fields to push intersecting vertices along $\nabla f(x)$, guaranteeing collision clearance ($f(x)\geq \epsilon$) without tearing mesh boundaries.  
> * **GO-MLVTON (2026):** Multi-layer occlusion learning for virtual try-on, isolating inner and outer garments before composite fitting.

## **5\. Implementation Roadmap & Action Plan**

> 1. **Phase 1 (Immediate \- Input Contract & Prep):** Update canonical.py and reconstruct.py to composite sheer dress views onto neutral gray (\#808080) canvases prior to Replicate calls. Disable remove\_background: True when handling sheer SKU categories.  
> 2. **Phase 2 (Multi-Shell Architecture):** Refactor Stage B in run\_hybrid\_poc.py to separate the dress into two discrete geometry shells:  
   * Inner Opaque Bandeau / Slip: Generated via fitted\_shell from mannequin body skin.  
   * Outer Sheer Midi Overlay: Generated via canonical guide mesh with an explicit slit boundary cut.  
> 3. **Phase 3 (Procedural Straps):** Integrate generate\_procedural\_straps() as a post-generation pass in run\_hybrid\_poc.py before exporting final GLB assets.  
> 4. **Phase 4 (Cylindrical Texture Baking):** Upgrade texture\_bake.py to default to cylindrical\_unwrap for dresses and skirts to ensure seam-free lateral flank projection.