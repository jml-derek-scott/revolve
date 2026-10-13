// Long-form "MVP info" write-up shown in the sidebar's MVP INFO modal.
// Facts marked "measured" come from our own runs on the Mac Pro (AMD GPU, PyTorch MPS);
// models under "Candidates" are from published work and have NOT been tested here yet, except TRELLIS.2
// (template 1's full build, run on the Hugging Face Space, ~45 s).

export const MVP_INFO = `
<p class="lede">This demo compares two ways of turning a product's front and back photos into a 3D model.
<strong>Texture wrap</strong> is fast and clean but flattens garments onto the body. <strong>Full build</strong>
gets real garment shape but, as you can see, looks rough. This note covers what we tried, why the full build looks
the way it does, which models could do much better, and a hybrid of both approaches that we think is the most
promising path to a real product.</p>

<h2>Where we are today</h2>

<h3>Texture wrap (mannequin projection)</h3>
<p>The front and back photos are projected straight onto a premade, faceless mannequin. A clothing-segmentation
model (rembg's <code>u2net_cloth_seg</code>) masks out everything that isn't the garment, so the model's skin,
hair and bag stay mannequin-white.</p>
<ul>
  <li><strong>Speed:</strong> about 10 seconds per product, on CPU (measured).</li>
  <li><strong>Strengths:</strong> consistent, clean, brand-safe look; fitted pieces (slips, bodysuits, halters) read well; the real photo pixels give sharp prints.</li>
  <li><strong>Weaknesses:</strong> the garment has no geometry of its own, so skirts, flared hems and long coats get painted onto the legs and read as shorts or leggings. Anything hidden in the photo (behind a bag, under hair) leaves a hole, thin straps get dropped by the mask, and the sides of the body, which no photo sees, show streaks.</li>
</ul>

<h3>Local Hardware GPU Full build (InstantMesh)</h3>
<p>The whole figure is reconstructed as a new mesh. Zero123++ imagines six views around the person from the front
photo, then InstantMesh's reconstruction model builds a mesh from those views.</p>
<ul>
  <li><strong>Speed:</strong> about 6.5 minutes per product (measured): ~6 min of view generation on CPU, ~2 s reconstruction on the GPU, 10-20 s mesh extraction on CPU.</li>
  <li><strong>Strengths:</strong> real silhouette; skirts hang and hems flare.</li>
  <li><strong>Weaknesses:</strong> lumpy surfaces, blurry colour, smeared face, and it reproduces the person, not a mannequin.</li>
</ul>

<h3>Trellis + Cloud GPU Test</h3>
<p class="note"><strong>Template 1 is different:</strong> its full build was made with
<a class="text-sky-400 underline hover:text-sky-300" href="https://huggingface.co/spaces/microsoft/TRELLIS.2" target="_blank" rel="noopener">Microsoft TRELLIS.2</a>
on a cloud GPU (Hugging Face Space). It took <strong>about 45 seconds</strong>, front photo only, and produced a
textured mesh. Templates 2 and 3 are still InstantMesh on our Mac, so you can compare the two directly.</p>

<h2>Why #2 and #3 full build looks bad</h2>
<p>None of these are bugs in our code. They're limits of the models and of the hardware we ran them on. My Mac has a AMD GPU which is not ideal for these AI Models.</p>
<ol>
  <li><strong>The back is imagined, not seen.</strong> Zero123++ only ever looks at the front photo. On every product we tested, it invented a back that copied the front (the leg slit appeared on the back, long hair disappeared). We fixed the worst of it by replacing its two back-facing views with the real back photo ("replace" mode). That clearly helped, and it's why the open back of template 3 shows up, but the other four views are still invented.</li>
  <li><strong>The reconstructor was trained on fixed camera angles.</strong> InstantMesh expects exactly Zero123++'s six poses (30°, 90°, 150°... at ±20° elevation). Our real photos are at 0° and 180°, which it never saw in training. Simply adding them as extra views barely changed anything.</li>
  <li><strong>Low resolution everywhere.</strong> The views are 320×320 pixels and the mesh is extracted on a 128³ grid, so fine details (lace, straps, seams) can't survive, and colour is stored per vertex rather than as a sharp texture.</li>
  <li><strong>These are general-object models.</strong> They were trained on furniture, toys and figurines, not on clothed people. Human bodies and thin, sheer, layered fabric are their weakest case.</li>
  <li><strong>The hardware fights us.</strong> On this Mac's AMD GPU (MPS backend, ~10 GB limit) the view generator runs out of memory and has to run on CPU, half-precision is unreliable, and some libraries (nvdiffrast, used for texture baking) are CUDA-only. That's why it's slow and why we export vertex colours instead of a proper texture.</li>
</ol>

<h2>What we already tried</h2>
<table>
  <thead><tr><th>Approach</th><th>Result</th></tr></thead>
  <tbody>
    <tr><td>TripoSR, front photo only</td><td>Fast and working, but the back is a guess.</td></tr>
    <tr><td>TripoSR twice (front + back) and stitch the halves</td><td>Removed: duplicate limbs and torsos where the halves overlapped.</td></tr>
    <tr><td>InstantMesh, front photo only</td><td>Coherent single body, but the back copies the front.</td></tr>
    <tr><td>InstantMesh + real photos added as extra views</td><td>Barely better: the model trusts its six trained angles.</td></tr>
    <tr><td>InstantMesh, real back photo <em>replaces</em> the imagined back views</td><td>Best full build: backs now match the product.</td></tr>
    <tr><td>TRELLIS.2 on a cloud GPU, front photo only (template 1)</td><td>About 45 s on the Hugging Face Space; textured mesh, far cleaner than InstantMesh. The back is still the model's guess.</td></tr>
    <tr><td>Texture projection onto a premade mannequin</td><td>Fast and clean; fails on skirts and loose fabric.</td></tr>
  </tbody>
</table>
<p class="note">We also checked SF3D, Zero123++ and Unique3D as multi-view options. All three take a <em>single</em> image,
so none of them can use the back photo. LGM takes four views but needs a CUDA-only renderer.</p>

<h2>Candidate models to test next</h2>
<p>These are published models that fit the problem better than InstantMesh. We haven't run them yet, and all of
them realistically need an NVIDIA GPU.</p>

<h3>Multi-view, general purpose</h3>
<ul>
  <li><strong>Hunyuan3D-2 (Tencent), including its multi-view variant.</strong> Generates shape and a proper texture separately, at much higher quality than InstantMesh. The multi-view variant is designed to take several real views (front, back, sides), so our real back photo would be used directly instead of being bolted on.</li>
  <li><strong>TRELLIS.2 (Microsoft). Tested on template 1.</strong> We ran the
  <a class="text-sky-400 underline hover:text-sky-300" href="https://huggingface.co/spaces/microsoft/TRELLIS.2" target="_blank" rel="noopener">public Hugging Face Space</a>
  on the front photo: about 45 seconds on a cloud GPU, with a real texture instead of vertex colours. It still
  takes a single image, so the back is invented; pairing it with our real back photo (via the hybrid below) is the
  obvious next step.</li>
</ul>

<h3>Built for clothed humans</h3>
<p>These start from a parametric body model (SMPL-X), so they understand that a person is underneath the clothes.
That fixes most of the lumpy-body problems and lets us swap the person for a mannequin later.</p>
<ul>
  <li><strong>ECON / ICON.</strong> Reconstruct a clothed person from one photo by fitting a body first, then adding clothing detail from predicted surface normals. Good at loose clothing.</li>
  <li><strong>SiTH.</strong> Single image to a textured human, with a diffusion step that hallucinates the back. Ideally we would replace that hallucinated back with our real back photo, the same trick that worked for InstantMesh.</li>
  <li><strong>PIFuHD.</strong> Older, but a well-known baseline for high-detail clothed humans.</li>
</ul>

<h3>Built for garments specifically</h3>
<ul>
  <li><strong>Sewing-pattern estimation</strong> (research such as Sewformer and DressCode). Predicts the garment's flat pattern pieces from a photo. A pattern can then be draped on our mannequin with cloth simulation, giving physically plausible skirts and folds. Most of this is still research-grade.</li>
  <li><strong>Body-pose estimation</strong> (e.g. HMR 2.0 / 4D-Humans). Not a reconstructor, but useful for posing the mannequin to match the photo so projected textures line up.</li>
</ul>

<h3>Training our own</h3>
<p>Our catalog is a real advantage here. Every product already comes with 3-5 studio photos from consistent angles
(front, three-quarter, back, detail).</p>
<ul>
  <li><strong>Fine-tune a multi-view model on our own photo angles.</strong> Teach it the exact poses our studio shoots, so real photos become first-class inputs instead of guesses.</li>
  <li><strong>Use the three-quarter shots.</strong> We aren't using them at all yet. They're real side information, exactly where both approaches are weakest today.</li>
  <li><strong>Garment-on-mannequin dataset.</strong> A few hundred products scanned or modelled once on our mannequin would be enough to fine-tune a reconstructor and to measure quality objectively.</li>
</ul>

<h2>The hybrid approach (recommended)</h2>
<p>The two methods fail in opposite ways. Texture wrap has <strong>great surface and no shape</strong>; full build
has <strong>shape and poor surface</strong>. A hybrid takes the shape from one and the surface from the other.</p>

<h3>First prototype: template 1</h3>
<p>Select template 1 and click <strong>Hybrid</strong> to see it. It uses only what we already had: the
TRELLIS.2 full build and the clothing masks from texture wrap. It runs in about 6 seconds on top of the build
(<code>scripts/build_hybrid.py</code>):</p>
<ol>
  <li>Project the front and back clothing masks onto the TRELLIS mesh and keep only the garment faces, closing small holes where the mask is patchy.</li>
  <li>Throw away the rest of the person (skin, hair, face, bag, shoes).</li>
  <li>Scale the garment to the mannequin and push it outward wherever it sinks into the body. The TRELLIS person was slimmer than the mannequin, so about half the garment had to move.</li>
  <li>Keep the TRELLIS texture on the garment for now.</li>
</ol>
<p><strong>Result:</strong> the slip keeps its real shape (cups, straps, hem) on a clean, faceless mannequin, and the
back matches the back photo. <strong>Still rough:</strong> gaps at the sides, where neither photo sees the garment;
skin-coloured patches in sheer areas, because TRELLIS baked the model's skin into the fabric; and the
garment's proportions come from the person, not the mannequin. Projecting the real photos onto the garment
(step 3 below) would fix the colour problems.</p>
<p class="note">Two mesh problems made this harder than it should have been, and any production pipeline needs to
handle them: our mannequin is modelled as a thick double-skinned shell with inconsistent face orientation, and
TRELLIS meshes have unreliable normals and split vertices along texture seams. The prototype avoids depending on
normals entirely and uses depth maps instead.</p>

<h3>How it would work</h3>
<ol>
  <li><strong>Body: always the mannequin.</strong> Fixed, clean, faceless, identical for every product.</li>
  <li><strong>Garment shape: from reconstruction.</strong> Run the full build (ideally a better model from above), align it to the mannequin, and keep only the garment. The clothing-segmentation masks we already compute tell us which part of the mesh is garment. Where the garment sits on the body (fitted tops), snap it onto the mannequin's surface; where it hangs free (skirts, hems), keep the reconstructed shape.</li>
  <li><strong>Garment surface: from the real photos.</strong> Project the actual front and back photos onto the garment mesh, exactly as texture wrap does today, instead of using the reconstructor's blurry colours. Prints, lace and logos stay sharp.</li>
  <li><strong>Fill the gaps.</strong> Use the three-quarter photo for the sides, and an image-inpainting model for anything no photo sees (behind the bag, under hair).</li>
</ol>

<h3>A cheaper version first</h3>
<p>Instead of reconstructing every garment, build a small library of garment shapes (mini skirt, A-line midi, maxi,
flared hem, long coat), fit the closest one to each product's silhouette, and texture it from the photos. That
covers most of a dress catalog with no heavy ML per product, and the full reconstruction can be reserved for
unusual pieces.</p>

<h3>Why this is the most promising route</h3>
<ul>
  <li>Every product sits on the same mannequin, so the catalog looks consistent.</li>
  <li>The things people judge (print, colour, fabric texture) come from real photos and stay sharp.</li>
  <li>Reconstruction errors only affect the garment's silhouette, which is far more forgiving than a whole body and face.</li>
  <li>It degrades gracefully: if reconstruction fails for a product, fall back to plain texture wrap.</li>
</ul>

<h2>Making it a product</h2>

<h3>Infrastructure</h3>
<ul>
  <li><strong>Run reconstruction on NVIDIA GPUs.</strong> Most candidate models, and proper texture baking, assume CUDA. A single cloud GPU also cuts the time dramatically: TRELLIS.2 took about 45 seconds on the Hugging Face Space, versus about 6.5 minutes for InstantMesh on our Mac.</li>
  <li><strong>Build offline, per product.</strong> The catalog is finite and photos rarely change, so generate each model once when a product is added and serve the finished file. Shoppers never wait.</li>
  <li><strong>Job queue instead of a live request.</strong> Builds take minutes; queue them, retry failures, and track progress.</li>
</ul>

<h3>Quality control</h3>
<ul>
  <li>Render each finished model from the photo angles and compare it with the real photos automatically; flag low scores for human review.</li>
  <li>Keep a fixed benchmark set of products (like templates 1-5) to compare every new model or setting against.</li>
</ul>

<h3>Viewer and delivery</h3>
<ul>
  <li>Compress models (Draco geometry, KTX2 textures) so they load quickly on phones.</li>
  <li>model-viewer is the simpler choice for a product page (AR support on phones for free); Three.js is the better choice if we want custom lighting and scenes like the nightclub.</li>
</ul>

<h2>Suggested next steps</h2>
<ol>
  <li>Fix the texture-wrap gaps (bag and hair holes, dropped straps, side seams) and add the three-quarter photo. That makes fitted garments shippable quickly.</li>
  <li>Run TRELLIS.2 on templates 2-5 too (it took about 45 s for template 1), then rent a cloud GPU for Hunyuan3D-2 multi-view and one clothed-human model (ECON or SiTH) and compare all of them side by side.</li>
  <li>Prototype the hybrid with the best result: mannequin body, reconstructed skirt, photo-projected texture.</li>
  <li>In parallel, try the garment-shape library on the three dresses with skirts.</li>
</ol>
<p class="note">Measured numbers come from our runs on the Mac Pro (AMD GPU), plus one TRELLIS.2 run on the Hugging Face
Space. The other candidate models are from published research and haven't been tested on our products yet, so expect
surprises in both directions.</p>
`;
