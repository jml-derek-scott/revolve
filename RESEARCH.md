This demo compares two ways of turning a product's front and back photos into a 3D model. **Texture wrap** is fast and clean but flattens garments onto the body. **Full build** gets real garment shape but, as you can see, looks rough. This note covers what we tried, why the full build looks the way it does, which models could do much better, and a hybrid of both approaches that we think is the most promising path to a real product.

## Where we are today

### Texture wrap (mannequin projection)
The front and back photos are projected straight onto a premade, faceless mannequin. A clothing-segmentation model (rembg's `u2net_cloth_seg`) masks out everything that isn't the garment, so the model's skin, hair and bag stay mannequin-white.

* **Speed:** about 10 seconds per product, on CPU (measured).
* **Strengths:** consistent, clean, brand-safe look; fitted pieces (slips, bodysuits, halters) read well; the real photo pixels give sharp prints.
* **Weaknesses:** the garment has no geometry of its own, so skirts, flared hems and long coats get painted onto the legs and read as shorts or leggings. Anything hidden in the photo (behind a bag, under hair) leaves a hole, thin straps get dropped by the mask, and the sides of the body, which no photo sees, show streaks.

### Full build (InstantMesh)
The whole figure is reconstructed as a new mesh. Zero123++ imagines six views around the person from the front photo, then InstantMesh's reconstruction model builds a mesh from those views.

* **Speed:** about 6.5 minutes per product (measured): ~6 min of view generation on CPU, ~2 s reconstruction on the GPU, 10-20 s mesh extraction on CPU.
* **Strengths:** real silhouette; skirts hang and hems flare.
* **Weaknesses:** lumpy surfaces, blurry colour, smeared face, and it reproduces the person, not a mannequin.

## Why the full build looks bad
None of these are bugs in our code. They're limits of the models and of the hardware we ran them on.

1. **The back is imagined, not seen.** Zero123++ only ever looks at the front photo. On every product we tested, it invented a back that copied the front (the leg slit appeared on the back, long hair disappeared). We fixed the worst of it by replacing its two back-facing views with the real back photo ("replace" mode). That clearly helped, and it's why the open back of template 3 shows up, but the other four views are still invented.
2. **The reconstructor was trained on fixed camera angles.** InstantMesh expects exactly Zero123++'s six poses (30°, 90°, 150°... at ±20° elevation). Our real photos are at 0° and 180°, which it never saw in training. Simply adding them as extra views barely changed anything.
3. **Low resolution everywhere.** The views are 320×320 pixels and the mesh is extracted on a 128³ grid, so fine details (lace, straps, seams) can't survive, and colour is stored per vertex rather than as a sharp texture.
4. **These are general-object models.** They were trained on furniture, toys and figurines, not on clothed people. Human bodies and thin, sheer, layered fabric are their weakest case.
5. **The hardware fights us.** On this Mac's AMD GPU (MPS backend, ~10 GB limit) the view generator runs out of memory and has to run on CPU, half-precision is unreliable, and some libraries (nvdiffrast, used for texture baking) are CUDA-only. That's why it's slow and why we export vertex colours instead of a proper texture.

## What we already tried

| Approach | Result |
| --- | --- |
| TripoSR, front photo only | Fast and working, but the back is a guess. |
| TripoSR twice (front + back) and stitch the halves | Removed: duplicate limbs and torsos where the halves overlapped. |
| InstantMesh, front photo only | Coherent single body, but the back copies the front. |
| InstantMesh + real photos added as extra views | Barely better: the model trusts its six trained angles. |
| InstantMesh, real back photo *replaces* the imagined back views | Best full build: backs now match the product. |
| Texture projection onto a premade mannequin | Fast and clean; fails on skirts and loose fabric. |

We also checked SF3D, Zero123++ and Unique3D as multi-view options. All three take a *single* image, so none of them can use the back photo. LGM takes four views but needs a CUDA-only renderer.

## Candidate models to test next
These are published models that fit the problem better than InstantMesh. We haven't run them yet, and all of them realistically need an NVIDIA GPU.

### Multi-view, general purpose
* **Hunyuan3D-2 (Tencent), including its multi-view variant.** Generates shape and a proper texture separately, at much higher quality than InstantMesh. The multi-view variant is designed to take several real views (front, back, sides), so our real back photo would be used directly instead of being bolted on.
* **TRELLIS (Microsoft).** A newer image-to-3D model with noticeably cleaner geometry and textures than the 2024 generation; worth a side-by-side.

### Built for clothed humans
These start from a parametric body model (SMPL-X), so they understand that a person is underneath the clothes. That fixes most of the lumpy-body problems and lets us swap the person for a mannequin later.

* **ECON / ICON.** Reconstruct a clothed person from one photo by fitting a body first, then adding clothing detail from predicted surface normals. Good at loose clothing.
* **SiTH.** Single image to a textured human, with a diffusion step that hallucinates the back. Ideally we would replace that hallucinated back with our real back photo, the same trick that worked for InstantMesh.
* **PIFuHD.** Older, but a well-known baseline for high-detail clothed humans.

### Built for garments specifically
* **Sewing-pattern estimation** (research such as Sewformer and DressCode). Predicts the garment's flat pattern pieces from a photo. A pattern can then be draped on our mannequin with cloth simulation, giving physically plausible skirts and folds. Most of this is still research-grade.
* **Body-pose estimation** (e.g. HMR 2.0 / 4D-Humans). Not a reconstructor, but useful for posing the mannequin to match the photo so projected textures line up.

### Training our own
Our catalog is a real advantage here. Every product already comes with 3-5 studio photos from consistent angles (front, three-quarter, back, detail).

* **Fine-tune a multi-view model on our own photo angles.** Teach it the exact poses our studio shoots, so real photos become first-class inputs instead of guesses.
* **Use the three-quarter shots.** We aren't using them at all yet. They're real side information, exactly where both approaches are weakest today.
* **Garment-on-mannequin dataset.** A few hundred products scanned or modelled once on our mannequin would be enough to fine-tune a reconstructor and to measure quality objectively.

## The hybrid approach (recommended)
The two methods fail in opposite ways. Texture wrap has **great surface and no shape**; full build has **shape and poor surface**. A hybrid takes the shape from one and the surface from the other.

### How it would work
1. **Body: always the mannequin.** Fixed, clean, faceless, identical for every product.
2. **Garment shape: from reconstruction.** Run the full build (ideally a better model from above), align it to the mannequin, and keep only the garment. The clothing-segmentation masks we already compute tell us which part of the mesh is garment. Where the garment sits on the body (fitted tops), snap it onto the mannequin's surface; where it hangs free (skirts, hems), keep the reconstructed shape.
3. **Garment surface: from the real photos.** Project the actual front and back photos onto the garment mesh, exactly as texture wrap does today, instead of using the reconstructor's blurry colours. Prints, lace and logos stay sharp.
4. **Fill the gaps.** Use the three-quarter photo for the sides, and an image-inpainting model for anything no photo sees (behind the bag, under hair).

### A cheaper version first
Instead of reconstructing every garment, build a small library of garment shapes (mini skirt, A-line midi, maxi, flared hem, long coat), fit the closest one to each product's silhouette, and texture it from the photos. That covers most of a dress catalog with no heavy ML per product, and the full reconstruction can be reserved for unusual pieces.

### Why this is the most promising route
* Every product sits on the same mannequin, so the catalog looks consistent.
* The things people judge (print, colour, fabric texture) come from real photos and stay sharp.
* Reconstruction errors only affect the garment's silhouette, which is far more forgiving than a whole body and face.
* It degrades gracefully: if reconstruction fails for a product, fall back to plain texture wrap.

## Making it a product

### Infrastructure
* **Run reconstruction on NVIDIA GPUs.** Most candidate models, and proper texture baking, assume CUDA. A single cloud GPU would also cut the 6.5 minutes to well under a minute.
* **Build offline, per product.** The catalog is finite and photos rarely change, so generate each model once when a product is added and serve the finished file. Shoppers never wait.
* **Job queue instead of a live request.** Builds take minutes; queue them, retry failures, and track progress.

### Quality control
* Render each finished model from the photo angles and compare it with the real photos automatically; flag low scores for human review.
* Keep a fixed benchmark set of products (like templates 1-5) to compare every new model or setting against.

### Viewer and delivery
* Compress models (Draco geometry, KTX2 textures) so they load quickly on phones.
* model-viewer is the simpler choice for a product page (AR support on phones for free); Three.js is the better choice if we want custom lighting and scenes like the nightclub.

## Suggested next steps
1. Fix the texture-wrap gaps (bag and hair holes, dropped straps, side seams) and add the three-quarter photo. That makes fitted garments shippable quickly.
2. Rent a cloud GPU and run Hunyuan3D-2 multi-view and one clothed-human model (ECON or SiTH) on templates 1-5, side by side with InstantMesh.
3. Prototype the hybrid with the best result: mannequin body, reconstructed skirt, photo-projected texture.
4. In parallel, try the garment-shape library on the three dresses with skirts.

Measured numbers come from our runs on the Mac Pro (AMD GPU). The candidate models are from published research and haven't been tested on our products yet, so expect surprises in both directions.