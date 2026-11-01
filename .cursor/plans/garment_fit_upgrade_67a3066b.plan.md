---
name: Garment fit upgrade
overview: The new research note confirms the three-stage pipeline and adds a real collision formula, a flank blend rule, and a layering rule. Garment3DGen, DMap, and Dress-1-to-3 do not replace Hunyuan in this pass. The work is an analytic outer-skin push with an edge check, an open fitted shell, Replicate flank inpainting blended by view angle, and stacked clearance for the hoodie-over-jeans outfit.
todos:
  - id: sdf-solver
    content: "Add pipeline/sdf_collision.py: occupancy SDF (negative inside, outward gradient), analytic push to 4 mm, edge-midpoint pass, constrained Laplacian, report mean squared penetration"
    status: pending
  - id: wire-collision
    content: Use the solver from fit_ghost_garment, fit_garment, and build_male_outfit.py with a per-layer clearance; keep --collision zpush
    status: pending
  - id: open-shell
    content: "Add pipeline/garment_deformer.py: offset outer skin, cut neck/arm/hem, Laplacian-edit only outside the body, export open boundary loops with outward winding"
    status: pending
  - id: flank-inpaint
    content: "Add pipeline/flank_inpainting.py: Replicate ControlNet on flank depth/normals, composite onto the atlas with cos-theta weights, exemplar/TELEA fallback"
    status: pending
  - id: compare-template1
    content: Rebuild template 1 (sdf+shell+flank vs zpush) and check penetration energy, open hem, side print, and template 5 layer order in the viewer
    status: pending
isProject: false
---

# Better garment meshes from the research notes

## What the new note changes

The architecture note matches the pipeline we already planned: isolate the garment, push it off the body along an SDF gradient, then fill the side seams. Four details change the implementation.

- **Use the production push, not the neural ReFU layer.** Section 3 quotes ReFU (Tan et al., arXiv 2207.13871): move a vertex only when `f(x) < 0`, by `x - α f(x) * unit(∇f)`, with `α` predicted by a network from a garment latent. That network is a layer inside a pose-dependent garment model. We do not have its weights, and a fixed `α` would only put cloth on the skin (`f = 0`), which still clips at hems and under arms. Section 4 is the formula to code: while `SDF < ε`, `v += (ε - SDF) * unit(∇SDF)`, with `ε` in the 3–5 mm band. We keep `ε = 4 mm`. Sign convention: negative inside, gradient points outward. The learned `α > 1` exists to clear edge-edge hits that a vertex query misses. The stand-in is a second pass on edge midpoints, not a fake MLP.
- **Blend the flank, do not paste it.** Section 5 asks for weights from the view angle, `cos θ`. Catalog projection keeps the texels that face the photo. The ControlNet flank fills only as that facing falls off.
- **Layers get their own clearance.** Template 5 already fits a hoodie and jeans as separate meshes in [scripts/build_male_outfit.py](scripts/build_male_outfit.py), and both go through the same Z push. The solver takes a clearance argument. Jeans sit 4 mm off the body. The hoodie sits 4 mm off the jeans where they overlap, and 4 mm off the body everywhere else.
- **The named generators are not drop-in replacements.** See below. Hunyuan on Replicate stays the guide mesh.

## Which papers we can use

- **ReFU.** Real collision layer. Implement the section 4 analytic push plus an edge-midpoint pass. Do not train or invent `g(k(z), f)`.
- **Garment3DGen** (Sarafianos et al., 3DV 2025, [nsarafianos/Garment3DGen](https://github.com/nsarafianos/Garment3DGen)). This is the Jacobian deformer: a template is pulled toward a pseudo-ground-truth mesh (they use InstantMesh; ours would be Hunyuan), with CLIP losses, on CUDA (`nvdiffrast`, PyTorch3D). About 1800 steps. Their own note says the template and the target have to be the same kind of garment, and a skirt template will not become a shirt. The published examples are pieces like a long sleeve, not this slip. It does not replace the fitted shell for template 1. If a later product has a matching template, run their repo as one Modal job. Do not reimplement it locally.
- **DMap** (Li et al., SIGGRAPH 2025, [liren2515/DMap](https://github.com/liren2515/DMap)). Pattern-coordinate diffusion. It wants a SMPL body, a single image, and predicted normals. [mannequins/female_base_mesh.glb](mannequins/female_base_mesh.glb) is not SMPL, and we have a real back photo their single-view path does not take. Right long-term generator for loose clothes with real pattern UVs. Not this pass.
- **Dress-1-to-3** (SIGGRAPH 2025). Sewing patterns plus a differentiable cloth simulator. The project page has no training or inference code. Take the requirement it shares with the note: the garment stays a separate open mesh, not a fused person. Do not block on it.
- **GarmentCode, SMPLicit, LayerNet, GO-MLVTON.** Pattern DSLs and layered implicits on SMPL. Same limitation as DMap. The layering rule is already covered by stacked clearance on template 5.

## Where each stage runs

Flux Kontext, LaMa, and Hunyuan3D-2mv already go to Replicate ([pipeline/canonical.py](pipeline/canonical.py), [pipeline/inpaint.py](pipeline/inpaint.py), [pipeline/reconstruct.py](pipeline/reconstruct.py)). ControlNet flank fill is the same kind of call. The SDF solve and the open shell are NumPy on this Mac. SegFormer already runs locally in [pipeline/clothes_seg.py](pipeline/clothes_seg.py). No new isolation module.

```mermaid
flowchart LR
  photos[Front and back photos]
  canon[Existing canonical isolation]
  guide[Hunyuan on Replicate]
  shell[Open fitted shell]
  sdf[Clearance push]
  bake[Photo projection]
  flank[ControlNet on Replicate]
  glb[hybrid GLB]
  photos --> canon --> guide
  canon --> shell
  guide --> sdf
  shell --> sdf --> bake --> flank --> glb
```

## What we will build

The Z push stays behind `--collision zpush` so template 1 can be compared.

### 1. Outer-skin clearance — `pipeline/sdf_collision.py`

The mannequin is a joined double skin about 5 mm thick with scrambled winding. Do not use `mesh.contains()`, vertex normals, or a winding-number SDF ([PLAN.md](PLAN.md), [CLAUDE.md](CLAUDE.md)).

- Rasterize both skins into an occupancy grid. Flood-fill from outside the bounding box. Unreached voxels (cavity and shell) are inside. The zero level is the outer surface, so the inner skin is not a second zero crossing.
- Grid step 5 mm. Cache distance and normalized gradient as a `.npz` next to the mannequin. Trilinear lookup.
- For each vertex with `SDF < ε`: `v += (ε - SDF) * unit(∇SDF)`, `ε = 0.004`. Cap a single step at about 2 cm. Up to 8 iterations.
- Edge pass: if an edge midpoint is inside `ε` while its endpoints are not, push both endpoints along the midpoint gradient. This is the edge-edge case ReFU handles with a learned scale.
- Then 2–3 uniform Laplacian iterations on vertices that moved. Vertices already outside by more than about 8 mm stay pinned, so a flare is not pulled onto the body.
- `clearance_m` is an argument. A second garment can pass a larger clearance, or the inner garment’s own distance field, in the overlap band.
- After the solve, report penetration energy: the mean of `max(ε - SDF, 0) ** 2`. The note’s target is below `1e-4`. Also report how many vertices are still inside `ε`.

Call this from `fit_ghost_garment` and `fit_garment` after alignment. In [scripts/build_male_outfit.py](scripts/build_male_outfit.py), jeans use 4 mm against the body. The hoodie uses 4 mm against the body, and 4 mm outside the jeans where the two overlap. Arms stay in the field. The gradient pushes sleeves off the arm, which is why the hoodie no longer needs `strip_arms`.

### 2. Open shell for fitted garments — `pipeline/garment_deformer.py`

Template 1’s slip is fitted, and Garment3DGen has no slip template, so the shell comes from the outer skin:

- Offset about 8 mm along the SDF gradient.
- Cut neck, armholes, and hem from the bust-to-hem span in `placement.json` and the Y band `fit_ghost_garment` already computes. Those loops stay boundary edges. Do not cap them.
- Where the Hunyuan guide is outside the body by more than the clearance, move only those vertices toward the guide with a few Laplacian-editing steps. Do not pull loops onto guide vertices inside the body.
- On export, the garment is non-watertight on purpose. Face winding points outward (toward positive SDF). Do not try to repair the mannequin’s winding.

Hunyuan remains the guide (`--product-shot`, front and back only). A Modal run of Garment3DGen is the follow-on when a product has a same-class template and this shell misses the flare.

### 3. Flank inpaint — `pipeline/flank_inpainting.py`

Same Replicate client pattern as [pipeline/inpaint.py](pipeline/inpaint.py). No local `diffusers` weights.

- After [pipeline/texture_bake.py](pipeline/texture_bake.py) projects front, back, and any profiles, mark texels with `|n_z| <= 0.25` that no photo painted.
- Render left and right orthographic views: color so far, depth, and normals. The inpaint mask is the unpainted flank in that view.
- Call a Replicate depth or normal ControlNet inpaint. The text names the garment. The conditioning image is the real photo or the Flux canonical view, so the print continues from pixels we have.
- Composite back onto the atlas with view-angle weights: catalog weight `clip(|n · view|, 0, 1)`, flank weight `1 - that`. No hard seam between the photo and the fill.
- If Replicate fails, copy the nearest painted texel on the same UV island, then TELEA.
- Profile photos stay out of Hunyuan.

### 4. CLI

In [scripts/run_hybrid_poc.py](scripts/run_hybrid_poc.py):

- `--collision sdf|zpush` (default `sdf` on the ghost / product-shot path).
- `--shell fitted` to export the open shell instead of the raw Hunyuan mesh.
- `--flank replicate|local` (default `replicate` when a token is present).

## How we know it worked

Rebuild template 1 with the current product-shot command (placement, left, right, `--no-write-build`): once `--collision zpush`, once `--collision sdf --shell fitted --flank replicate`.

- Penetration energy under `1e-4`, and no body through the bust, hips, or underarms.
- Neck, armholes, and hem are open loops. The hem stays a mini at the photo length. Garment normals point outward.
- Side texels continue the leopard mesh. The join between photo and flank follows the facing angle, not a stripe.
- Template 5: jeans clear the legs, hoodie clears both the body and the jeans, sleeves clear the arms.

## Explicitly not in this pass

- A new `canonical_isolation.py`.
- Vendoring Garment3DGen, DMap, or Dress-1-to-3. No SMPL fit. No learned ReFU scale.
- ControlNet weights in the Poetry env.
- `POST /api/generate`. This stays an offline script.
