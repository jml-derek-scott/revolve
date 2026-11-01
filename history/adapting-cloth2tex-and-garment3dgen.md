# Adapting Cloth2Tex and Garment3DGen

This note explains the two checkouts under `repomix/cloth2tex` and `repomix/Garment3DGen`: what each one actually does, which papers the code is built on, which templates are usable, and how both fit a pipeline that turns a front and back catalog photo into a garment shell on a rigged mannequin.

The shell has to be a separate mesh from the body. A reconstructor that sees the original photo builds one watertight blob of the model plus the clothes. A rig cannot swing a skirt without dragging the legs, and it cannot pose an arm without dragging the sleeve. Both papers exist to avoid that. They keep a garment with its own neck, cuffs, waist, and hem, and they leave the body as another mesh.

Our current outfit flow already does the pieces that belong on the Mac: Ghost plates in `pipeline/prepass.py`, category configs in `pipeline/outfit.py`, and layered SDF fitting in `scripts/build_outfit.py`. Neither repo's optimizer runs locally. Both need CUDA.

## Cloth2Tex

**Paper.** Daiheng Gao, Xu Chen, Xindi Zhang, Qi Wang, Ke Sun, Bang Zhang, Liefeng Bo, and Qixing Huang. *Cloth2Tex: A Customized Cloth Texture Generation Pipeline for 3D Virtual Try-On.* arXiv:2308.04288, 3DV 2024. Alibaba XR Lab, ETH Zurich, MPI-IS, UT Austin.

The paper has two phases. Phase I registers a parametric garment mesh to a catalog image and bakes a coarse texture. Phase II completes the fine texture with an image-translation network trained on pictures synthesized by a latent diffusion model. The public tree contains Phase I only. The README says the landmark detector and the automatic scale are also unreleased.

### What the released fitter does

`phase1_inference.py` does not reconstruct a mesh from pixels. The caller passes a category id (`--g`). That id selects five things that all have to agree:

- a template OBJ at `template/reference/<category>/mesh/mesh.obj`
- a hand-tuned scale (`per_scale_dict`)
- a UV mask under `template/uv_mask/`
- a landmark name list (`landmark_order_dict`)
- the vertex indices of those landmarks on the front and the back (`vertex_number_dict`)

The images are already a cut-out garment. For each sample the script reads:

- `template/reference/<category>/bg/<i>_1.jpg` and `<i>_2.jpg`, the front and back on a background
- `mask/<i>_1.jpg` and `<i>_2.jpg`, the binary silhouettes
- `square/<i>_1.json` and `<i>_2.json`, LabelMe point files

There are three optimization stages, all on CUDA through PyTorch3D.

**1. A rigid nudge.** The template is translated in Y until one projected landmark (the code comment says the left side of the neckline) sits on the matching 2D point. This is a loop of fixed steps, not a learned alignment.

**2. An embedded deformation graph.** `lib/deformation_graph.py` builds a coarse graph with qslim-style downsampling, then blends each vertex from its nine nearest graph nodes. The optimizer moves a rotation and a translation on every node. `models/deform_model.py` scores the result by rendering the silhouette from the front and the back:

- mask MSE, plus 0.3 times a mask IoU
- MSE between the projected landmark vertices and the LabelMe points
- an edge loss between the rendered outline and the mask outline
- mesh normal consistency
- an as-rigid-as-possible penalty on the graph

The graph and the ARAP term are the classical ones: Sumner, Schmid, and Pauly, *Embedded Deformation for Shape Manipulation* (SIGGRAPH 2007), and Sorkine and Alexa, *As-Rigid-As-Possible Surface Modeling* (Symposium on Geometry Processing 2007). The renderer is PyTorch3D. The paper calls this a neural mesh renderer; the shipped script is test-time optimization, not a trained network.

`models/gcnn/` is a CoMA-style Chebyshev graph convolution (Ranjan et al., *Generating 3D Faces using Convolutional Mesh Autoencoders*, ECCV 2018). `phase1_inference.py` never imports it. The fitter that actually runs is the deformation graph.

**3. A UV bake.** The mesh is frozen. A free 512×512 RGB image is optimized so that differentiable renders of the front and the back match the photos under an L1 photometric loss, plus total variation on the texture. A garment UV mask keeps paint off the padding. ResNeXt and VGG feature extractors are constructed in `Trainer` and unused by this loss.

`template/comparison/pix2surf/` holds comparison assets for Pix2Surf (Mir, Alldieck, and Pons-Moll, ECCV 2020). They are not an inference path.

### Categories the fitter can run

Vertex counts and default scales are the tables in `phase1_inference.py`.

| Id | Garment | Vertices | Default scale | Landmarks |
| --- | --- | ---: | ---: | --- |
| `1_wy` | hoodie (卫衣) | 9648 | 1.1 | neck, side, shoulder, sleeve, plus a derived mid point and an optional neck point |
| `2_Polo` | polo | 8922 | 0.8 | same top set |
| `3_Tshirt` | t-shirt | 8523 | 0.9 | same top set |
| `4_shorts` | shorts | 8767 | 0.75 | waist, crotch, leg openings |
| `5_trousers` | trousers | 9323 | 0.75 | waist, crotch, leg openings |
| `6_zipup` | zip-up | 8537 | 1.1 | top set |
| `7_windcoat` | windcoat | 9881 | 0.65 | top set |
| `9_jacket` | jacket | 8168 | 1.0 | top set, plus the optional neck point |
| `11_skirt` | skirt | 6116 | 1.0 | waist and hem |

The code never spells the abbreviations out. From the neckline comment and from where the points sit in the LabelMe files: `ln`/`rn` are the neckline, `lh`/`rh` are the side points whose midpoint is stored as `mh`, `lso`/`rso` and `lpo`/`lpi`/`rpo`/`rpi` are the shoulder and sleeve points, `lw`/`mw`/`rw` are the waist, `cc` is the crotch, `llo`/`lli`/`rli`/`rlo` are the leg openings, and `ll`/`ml`/`rl` are the hem.

`10_dress` is on disk as photos, masks, and LabelMe points, in a root set plus `long/` and `short/`. The points are the skirt set: waist (`lw`, `mw`, `rw`) and hem (`ll`, `ml`, `rl`). There is no `mesh.obj`, and `10_dress` is absent from every lookup table in `phase1_inference.py`. Those folders are reference imagery. They are not a dress you can fit or skin.

## Garment3DGen

**Paper.** Nikolaos Sarafianos, Tuur Stuyck, Xiaoyu Xiang, Yilei Li, Jovan Popovic, and Rakesh Ranjan. *Garment3DGen: 3D Garment Stylization and Texture Generation.* arXiv:2403.18816, 2024.

The paper stylizes the shape and the texture of a garment so the result can sit on a parametric body and be simulated. The released loop is the shape half. Texture painting is unfinished.

### What the released loop does

`loop.py`, driven by `example_config.yml`, starts from a clean template and pulls it onto a second mesh. In the paper that second mesh comes from Wonder3D (Long et al.) plus Instant-NSR. The checkout has switched the target to InstantMesh (Xu et al.) and hardcodes `use_target_mesh = True`. The example pair is `meshes/longsleeve.obj` deformed toward `meshes_target/jacket_sdf_new.obj`.

The deformation is a Neural Jacobian Field. Noam Aigerman and Thibault Groueix, *Neural Jacobian Fields: Learning Intrinsic Mappings of Arbitrary Meshes* (SIGGRAPH 2022), store a 3×3 matrix on every triangle and recover vertex positions by solving a Poisson equation. `NeuralJacobianFields/SourceMesh.py` and `PoissonSystem.py` are that solve. TextDeformer (William Gao, Noam Aigerman, et al., SIGGRAPH 2023) is the same representation, optimized so differentiable renders match a CLIP embedding. Garment3DGen cites both, and swaps in FashionCLIP (Chia et al., 2022) so the embedding is about clothes.

Each of the 1800 Adam steps does this:

1. Turn the current Jacobians into vertices with the Poisson solve.
2. Build a mesh that keeps the source faces and UVs (`base=load_mesh`) and a flat gray albedo. The comment in the loop says the texture path still needs work.
3. Render the deformed mesh and the target mesh from a random camera batch with nvdiffrast (Laine et al.).
4. Score them:
   - FashionCLIP L1 between the two renders (`clip_weight` 2.5 in the example config)
   - pixel L1 between the renders (`delta_clip_weight` 5)
   - chamfer distance on 10,000 surface samples, multiplied by 25
   - edge length, normal consistency, uniform Laplacian, and a triangle-area penalty (PyTorch3D)
   - a penalty that pulls each Jacobian toward the identity (`regularize_jacobians_weight` 0.15)

After step 1000 the CLIP weight and the multi-view consistency weight are set to 0, and the Jacobian penalty drops to 0.025. The tail of the run is geometric. Faces, UVs, and open boundaries stay on the source template. The target mesh is only a shape to match.

The README states the limit that matters for classification: the source and the target have to be the same kind of garment, and they have to start near each other. A skirt template will not become a shirt. The target should also keep the arms off the torso, so the reconstructor does not fuse a sleeve to the body.

### Templates it ships

Source meshes in `meshes/`:

- `tshirt.obj`
- `tanktop.obj`
- `poncho.obj`
- `longsleeve.obj` (the only one with an `.mtl` beside it)
- `dress_shortsleeve.obj`

`meshes_target/jacket_sdf_new.obj` is an example target hull, not a template. These sources have no landmark-to-vertex table and no per-category scale. They are deformation sources. `dress_shortsleeve.obj` is the one dress mesh in either checkout.

## Which templates to use

Use both libraries. They fill different holes.

Cloth2Tex is the better default for tight, standard pieces. Those OBJs already have UVs, a known vertex count around 6k–10k, and a landmark-to-vertex table, and they were built to register to a catalog front and back. That covers hoodie, polo, t-shirt, shorts, trousers, zip-up, windcoat, jacket, and skirt.

Garment3DGen is the library that actually ships a dress, and it also has a tank and a poncho, which Cloth2Tex does not. Its meshes can be deformed onto a hull. They cannot be dropped into `phase1_inference.py`, because that script needs landmark indices and a scale.

A dress photo set with no mesh cannot be skinned. A dress mesh with no landmark table can still be deformed onto a hull. So a dress job starts from `dress_shortsleeve.obj` (or a later dress template we add), and a jeans-and-tee job starts from the Cloth2Tex trousers and t-shirt.

## How this maps onto our pipeline

Classification picks the template and the motion model before any 3D step. `pipeline/outfit.py` already has `top`, `bottom`, and `dress`, plus a material and a layer order. The decision that still has to be made per piece is the template id, and whether the piece is body-skinned or free-hanging.

A t-shirt or a pair of jeans takes the mannequin's skinning weights and follows the body. The existing fit already holds it 4 mm off the skin, and a later layer 4 mm off the shell under it. A dress skins at the bodice or the waist only. The skirt hangs, and it is simulated or driven on its own. Classifying a dress as a tee throws the skirt away, and the Jacobian fit will not invent one. That is the same limit as the Garment3DGen README.

The adapted stages:

1. **Classify.** Map each piece to a template id from the union of the two libraries, and mark it body-skinned or free-hanging. SegFormer clothes labels (`mattmdjaga/segformer_b2_clothes`) are a start. The label that matters is the template plus the motion, because a dress and a long tee can look similar to a segmenter and have to move differently.

2. **Remove the person.** Keep the Ghost plates. Opaque pieces stay Ghost v1: segment, cut out, and composite onto white, so the catalog pixels survive. Dresses, sheer fabric, and prints that a segmenter would call skin stay Ghost v2: rewrite an empty garment, as `configs/templates/guess_dress.json` already does. Both papers assume this has already happened. Cloth2Tex reads `bg/` and `mask/`. Garment3DGen reads a garment image with no body in it.

3. **Make three empty-garment views.** Ideal front, ideal back, and an ideal side, each of the garment alone. Neither paper makes a side image. Cloth2Tex never sees a side, so the side of its mesh is whatever the template's rest shape already is. Garment3DGen only has a side because Wonder3D or InstantMesh hallucinated one inside the target mesh. We should generate the side on purpose, as a product shot of the empty garment, and use it as a silhouette and as a texture source for the flank. The side has to be the garment alone. A side that still shows an arm welds that arm into the sleeve.

4. **Build a coarse hull and keep it as a target.** A multi-view reconstructor (Hunyuan3D-2mv on Replicate, or a three-view model if the side is a real conditioning image) turns the three plates into a hull. That hull is watertight. It is the Garment3DGen target mesh. It is not the asset we export, because it cannot be animated separately from the body.

5. **Deform the template onto the hull, on Modal.** Prefer the geometric half of the Garment3DGen loop: chamfer, render L1, edge, normal, and Laplacian, with the Jacobian penalty, and with the CLIP weight left at 0. Connectivity, UVs, and the open boundaries survive. Cloth2Tex's deformation graph is the alternative when the category is one of the nine fitted ones and the only constraints are front and back masks. nvdiffrast, the PyTorch3D rasterizer, and the Poisson solve stay on a CUDA worker. The Mac keeps classification, Ghost plates, and the SDF fit.

6. **Fit, then bake.** `scripts/build_outfit.py` already pushes each shell off the mannequin with a layered signed distance field, which is the right collision test on `female_base_mesh.glb` (the mannequin is a double skin with scrambled winding). Bake the ideal front and back into the template UV. Use the ideal side, plus the existing flank inpaint, for the grazing angles under the arms and along the side seams. That bake replaces Cloth2Tex's unreleased Phase II and the unfinished texture code in `loop.py`.

Skinning follows the class flag from step 1. Tops, jeans, and jackets copy weights from the nearest mannequin skin. Dresses and skirts copy weights on the bodice or the waistband and leave the hem free.

## Leave these out

The FashionCLIP stylization loop is for changing a design. The armor clip in the Garment3DGen README is that mode. Catalog fidelity wants the geometric fit, with the CLIP weight at 0.

Cloth2Tex's per-category scale is a hand constant, and its landmark detector was never released. Until we label landmarks ourselves, silhouette and chamfer against the hull are the constraints to trust. The vertex-index tables are only valid on the Cloth2Tex meshes they were written for.

Neither template matches a sequin mini with spaghetti straps and a back bow, which is the garment in `configs/templates/guess_dress.json`. A 384³ diffusion grid erases straps that are a pixel or two wide, so straps stay a procedural strip between shoulder and bust anchors. A bow is its own small mesh, or a displacement on the dress template. A plain `dress_shortsleeve` will not grow either of them during the Jacobian fit.
