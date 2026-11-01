import { ThreeViewer } from "./three-viewer.js?v=sim2i";
import { MVP_INFO } from "./mvp-info.js";

const HDRI = {
  beach: "assets/hdri/umhlanga_sunrise_1k.hdr",
  nightclub: "assets/hdri/disco.hdr",
};

// Pre-built models: assets/models/<template>-<method>.glb
const TEMPLATES = [1, 2, 3, 4, 5];
const GHOST_TEMPLATE = 4; // 2D Ghost Mannequin isolation pre-pass before Hunyuan3D-2mv
const HAS_WRAP = new Set([1, 3, 4, 5]);
const HAS_BUILD = new Set([1, 3, 4, 5]);
const HAS_HYBRID = new Set([1, 3, 4, 5]); // add n after scripts/run_hybrid_poc.py writes assets/models/<n>-hybrid.glb
// Ideal views from Flux Kontext. Sides appear under the catalog photos.
const SIM2 = {
  1: {
    dir: "/outputs/canonical/1/dress",
    sides: ["front", "back", "left", "right"],
    mesh: "/outputs/canonical/1/dress-fitted.glb",
  },
  2: {
    dir: "/outputs/canonical/2/dress",
    sides: ["front", "back", "left", "right"],
    mesh: "/outputs/canonical/2/dress-fitted.glb",
  },
};
const METHODS = {
  wrap: "Front/back photos projected onto a premade mannequin (~10 s to build).",
  build: "Cloud GPU 3D reconstruction (TRELLIS.2 or Hunyuan3D-2mv).",
  hybrid: "Cloud 3D reconstruction combined with sharp front/back photo projection.",
};
const WRAP_HINT = {
  1: "Ideal dress photos projected onto the stock mannequin, lifted onto the chest. Profile photos fill the sides; the arms stay bare.",
  2: "No wrap mesh. Sim2 images stay under the catalog photos. Hybrid is the Hunyuan shell from those views.",
  3: "2× upscaled photos, Ghost isolation, waist–hip silhouette warp on the stock mannequin. Halter cut stays sleeveless; hips/waist wrap around. Open back is still painted onto the body.",
  4: "Photos projected onto the stock mannequin (baseline wrap). Compare with Ghost build & Ghost + Mannequin.",
  5: "Original photos projected onto the male body. This is not the generated-image reconstruction. Model is the textured person. Basic is the plain mannequin.",
};
const BUILD_HINT = {
  1: "Whole figure from Microsoft TRELLIS.2 (Hugging Face Space cloud GPU, ~45 s, front photo only).",
  2: "No full-build mesh. Sim2 images stay under the catalog photos. Hybrid is the Hunyuan shell from those views.",
  3: "Whole figure from Hunyuan3D-2mv (Replicate, front + back photos), textured with Hunyuan3D-Paint on Modal A10G.",
  4: "Ghost Mannequin 3D Build: person, hair & bag stripped in 2D first (SegFormer + LaMa), then Hunyuan3D-2mv reconstructs only the isolated dress.",
  5: "The man, hoodie, and jeans as one Hunyuan3D-2mv mesh from the front and back photos, textured with Hunyuan3D-Paint.",
};
const HYBRID_HINT = {
  1: "Hunyuan3D-2mv dress shell from the ideal front and back, fitted on the white mannequin as its own mesh. Full build is still the TRELLIS person. This Hybrid file is the original planar bake. Side fill is a cylindrical 4-view preview of the same shell that paints the flanks from the Sim2 profiles.",
  2: "Hunyuan3D-2mv shell from the accepted Sim2 front and back, lifted onto the chest and SDF-fitted. Texture from all four Sim2 views. Preview under outputs/, not a demo file.",
  3: "Hunyuan3D-2mv Full Build + Paint, wrap dress overlaid. Skin/hair keep Paint; no mannequin, no white fill.",
  4: "Ghost + Mannequin: the ghost dress scaled onto the mannequin. The back is pulled out past the seat and shoulder blades so the body does not show through.",
  5: "Same full figure as Full build. Real hoodie and jeans photos on the front and back; face, hair, hands, and the sides keep the Paint texture. No separate mannequin.",
};

const $ = (s) => document.querySelector(s);
const mv = $("#mv");
const three = new ThreeViewer($("#three"), (txt) => ($("#stats").textContent = txt));

let state = { template: 1, method: "wrap", env: "beach", renderer: "three", body: "model", sim2Preview: false };
let loadedUrl = null;

const ASSET_REV = "sim2j";
const MALE_TEMPLATE = 5;

function sim2Mesh() {
  return SIM2[state.template]?.mesh || null;
}

function hasSideFill() {
  // Distinct from Hybrid: T1 keeps 1-hybrid.glb and offers the cylindrical preview separately.
  return Boolean(sim2Mesh() && HAS_HYBRID.has(state.template));
}

function modelUrl() {
  const src = new URLSearchParams(location.search).get("src");
  if (src) return src;
  const sim2 = sim2Mesh();
  if (state.sim2Preview && sim2) return `${sim2}?v=${ASSET_REV}`;
  // T2 has no demo hybrid; Hybrid loads the Sim2 mesh. T1 Hybrid stays on 1-hybrid.glb.
  if (sim2 && state.method === "hybrid" && !HAS_HYBRID.has(state.template)) return `${sim2}?v=${ASSET_REV}`;
  const variant = state.template === MALE_TEMPLATE && state.method === "wrap" ? `-${state.body}` : "";
  return `assets/models/${state.template}-${state.method}${variant}.glb?v=${ASSET_REV}`;
}

$("#templates").innerHTML = TEMPLATES.map(
  (n) => `<button data-t="${n}" class="group relative w-14 shrink-0 snap-start cursor-pointer overflow-hidden rounded-md border border-zinc-700 transition hover:border-zinc-400 hover:shadow-lg [&.active]:ring-2 [&.active]:ring-blue-500">
    <img src="assets/thumbs/${n}.png?v=${ASSET_REV}" alt="Template ${n}" class="aspect-square w-full object-cover opacity-75 transition group-hover:scale-110 group-hover:opacity-100 group-[.active]:opacity-100">
    <span class="absolute left-1 top-0.5 text-xs font-semibold [text-shadow:0_0_3px_#000]">${n}</span></button>`
).join("");

const templateStrip = $("#templates");
function templateStride() {
  const card = templateStrip.querySelector("button");
  if (!card) return 64;
  const gap = parseFloat(getComputedStyle(templateStrip).columnGap) || 6;
  return card.getBoundingClientRect().width + gap;
}
function updateTemplateArrows() {
  const max = templateStrip.scrollWidth - templateStrip.clientWidth;
  const left = templateStrip.scrollLeft;
  $("#templates-prev").disabled = left <= 8;
  $("#templates-next").disabled = max <= 8 || left >= max - 8;
}
function revealTemplate(n) {
  const card = templateStrip.querySelector(`button[data-t="${n}"]`);
  if (!card) return;
  const pad = 4;
  const left = card.offsetLeft;
  const right = left + card.offsetWidth;
  const viewLeft = templateStrip.scrollLeft;
  const viewRight = viewLeft + templateStrip.clientWidth;
  if (left < viewLeft + pad) templateStrip.scrollTo({ left: Math.max(0, left - pad), behavior: "smooth" });
  else if (right > viewRight - pad) templateStrip.scrollTo({ left: right - templateStrip.clientWidth + pad, behavior: "smooth" });
}
$("#templates-prev").addEventListener("click", () => {
  templateStrip.scrollBy({ left: -templateStride() * 2, behavior: "smooth" });
});
$("#templates-next").addEventListener("click", () => {
  templateStrip.scrollBy({ left: templateStride() * 2, behavior: "smooth" });
});
templateStrip.addEventListener("scroll", updateTemplateArrows, { passive: true });
templateStrip.addEventListener("wheel", (e) => {
  if (templateStrip.scrollWidth <= templateStrip.clientWidth + 1) return;
  if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
  e.preventDefault();
  templateStrip.scrollLeft += e.deltaY;
}, { passive: false });
window.addEventListener("resize", updateTemplateArrows);

function methodLabel(template, method) {
  if (template === GHOST_TEMPLATE) {
    return { wrap: "texture wrap", build: "ghost build", hybrid: "ghost + mannequin" }[method];
  }
  return { wrap: "texture wrap", build: "full build", hybrid: "hybrid" }[method];
}

function meshAvailable() {
  if (new URLSearchParams(location.search).get("src")) return true;
  if (state.sim2Preview && sim2Mesh()) return true;
  if (state.method === "wrap") return HAS_WRAP.has(state.template);
  if (state.method === "build") return HAS_BUILD.has(state.template);
  if (state.method === "hybrid") return HAS_HYBRID.has(state.template) || Boolean(sim2Mesh());
  return false;
}

async function loadModel() {
  if (!meshAvailable()) {
    loadedUrl = null;
    mv.removeAttribute("src");
    three.clear();
    $("#status").textContent = `Template ${state.template}. Sim2 images only — no mesh yet.`;
    return;
  }
  const url = modelUrl();
  if (url === loadedUrl) return;
  loadedUrl = url;
  $("#status").textContent = "Loading…";
  try {
    mv.src = url;
    await three.load(url);
    if (url === loadedUrl) {
      const src = new URLSearchParams(location.search).get("src");
      const body = state.template === MALE_TEMPLATE && state.method === "wrap" ? `, ${state.body} body` : "";
      $("#status").textContent = src
        ? src
        : state.sim2Preview
          ? `Template ${state.template}, side fill.`
          : `Template ${state.template}, ${methodLabel(state.template, state.method)}${body}.`;
    }
  } catch (err) {
    $("#status").textContent = `Error: ${err.message}`;
  }
}

function apply() {
  if (state.body === "basic") state.body = "model";
  const canSim2 = Boolean(sim2Mesh());
  if (!hasSideFill()) state.sim2Preview = false;
  if (state.method === "build" && !HAS_BUILD.has(state.template)) state.method = canSim2 ? "hybrid" : "wrap";
  if (state.method === "hybrid" && !HAS_HYBRID.has(state.template) && !canSim2) state.method = "wrap";
  if (state.method === "wrap" && !HAS_WRAP.has(state.template) && canSim2) state.method = "hybrid";
  const url = HDRI[state.env];
  // Nightclub in model-viewer stays dark: it lights the model from the disco skybox
  // (the "neutral" environment is effectively ignored while a skybox is set).
  mv.environmentImage = state.env === "nightclub" ? "neutral" : url;
  mv.skyboxImage = url;
  mv.exposure = state.env === "nightclub" ? 2 : 1.0;
  three.setEnvironment(state.env, url);

  $("#mv").hidden = state.renderer !== "mv";
  $("#three").hidden = state.renderer !== "three";
  three.setActive(state.renderer === "three");
  if (state.renderer === "mv") $("#stats").textContent = "model-viewer: see browser DevTools for FPS/memory.";

  const isGhost = state.template === GHOST_TEMPLATE;
  const buildBtn = $('#method button[data-m="build"]');
  const hybridBtn = $('#method-hybrid button[data-m="hybrid"]');
  buildBtn.textContent = isGhost ? "Ghost build" : "Full build";
  hybridBtn.textContent = isGhost ? "Ghost + Mannequin" : "Hybrid";

  const mark = (sel, key, val) =>
    document.querySelectorAll(`${sel} button`).forEach((b) => b.classList.toggle("active", b.dataset[key] === String(val)));
  mark("#templates", "t", state.template);
  revealTemplate(state.template);
  updateTemplateArrows();
  mark("#method", "m", state.sim2Preview ? "" : state.method);
  mark("#method-hybrid", "m", state.sim2Preview ? "" : state.method);
  mark("#env", "env", state.env);
  mark("#renderer", "r", state.renderer);
  mark("#body-style", "body", state.body);
  const basicBtn = document.querySelector('#body-style button[data-body="basic"]');
  if (basicBtn) basicBtn.disabled = true;
  $("#body-style-wrap").hidden = !(state.template === MALE_TEMPLATE && state.method === "wrap");

  const canWrap = HAS_WRAP.has(state.template);
  const canBuild = HAS_BUILD.has(state.template);
  const wrapBtn = $('#method button[data-m="wrap"]');
  wrapBtn.disabled = !canWrap;
  buildBtn.disabled = !canBuild;
  hybridBtn.disabled = !HAS_HYBRID.has(state.template) && !canSim2;
  const sideFillBtn = $('#method-hybrid button[data-preview="sim2"]');
  const canSideFill = hasSideFill();
  sideFillBtn.hidden = !canSideFill;
  sideFillBtn.classList.toggle("active", state.sim2Preview);

  $("#method-hint").textContent = state.sim2Preview
    ? "Cylindrical 4-view bake of the same Hunyuan shell, using the Sim2 left and right photos. Hybrid still holds the original planar file."
    : state.method === "wrap" && WRAP_HINT[state.template]
      ? WRAP_HINT[state.template]
      : state.method === "build" && BUILD_HINT[state.template]
        ? BUILD_HINT[state.template]
        : state.method === "hybrid" && HYBRID_HINT[state.template]
          ? HYBRID_HINT[state.template]
          : !canBuild
            ? `${METHODS.wrap} Full build not generated for this template.`
            : METHODS[state.method];
  $("#photos").innerHTML = ["front", "back"]
    .map((side) => `<button data-photo="${side}" class="group relative flex-1 cursor-zoom-in overflow-hidden rounded bg-white ring-blue-500 transition hover:ring-2" aria-label="View ${side} photo full size">
      <img src="assets/templates/${state.template}-${side}.jpg?v=${ASSET_REV}" alt="" class="h-28 w-full object-contain transition group-hover:scale-105">
      <span class="absolute inset-0 flex items-center justify-center bg-black/0 text-2xl text-white opacity-0 transition group-hover:bg-black/30 group-hover:opacity-100"><i class="bx bx-zoom-in"></i></span>
      <span class="block text-center text-[10px] text-zinc-500">${side}</span></button>`)
    .join("");
  const ghostRow = $("#ghost-photos");
  ghostRow.hidden = !isGhost;
  ghostRow.innerHTML = isGhost
    ? ["front", "back"]
        .map(
          (side) => `<button data-ghost-photo="${side}" class="group relative w-16 shrink-0 cursor-zoom-in overflow-hidden rounded bg-white ring-blue-500 transition hover:ring-2" aria-label="View ${side} ghost cutout full size">
          <img src="assets/templates/${state.template}-${side}-ghost.png" alt="" class="h-9 w-full object-contain transition group-hover:scale-105">
          <span class="absolute inset-0 flex items-center justify-center bg-black/0 text-base text-white opacity-0 transition group-hover:bg-black/30 group-hover:opacity-100"><i class="bx bx-zoom-in"></i></span>
          <span class="block text-center text-[9px] leading-tight text-zinc-500">ghost ${side}</span></button>`
        )
        .join("") +
      `<button type="button" data-info="ghost" class="ml-0.5 flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-base text-zinc-400 transition hover:border-zinc-500 hover:bg-zinc-700 hover:text-zinc-100" aria-label="How the ghost images were made" title="How the ghost images were made"><i class="bx bx-info-circle"></i></button>`
    : "";
  const sim2 = SIM2[state.template];
  const sim2Row = $("#sim2-photos");
  sim2Row.hidden = !sim2;
  sim2Row.innerHTML = sim2
    ? sim2.sides
        .map(
          (side) => `<button data-sim2-photo="${side}" class="group relative w-12 shrink-0 cursor-zoom-in overflow-hidden rounded bg-white ring-blue-500 transition hover:ring-2" aria-label="View Sim2 ${side} full size">
          <img src="${sim2.dir}/${side}.png?v=${ASSET_REV}" alt="" class="h-9 w-full object-contain transition group-hover:scale-105">
          <span class="absolute inset-0 flex items-center justify-center bg-black/0 text-base text-white opacity-0 transition group-hover:bg-black/30 group-hover:opacity-100"><i class="bx bx-zoom-in"></i></span>
          <span class="block text-center text-[9px] leading-tight text-zinc-500">Sim2 ${side}</span></button>`
        )
        .join("") +
      `<button type="button" data-info="sim2" class="ml-0.5 flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-base text-zinc-400 transition hover:border-zinc-500 hover:bg-zinc-700 hover:text-zinc-100" aria-label="How the Sim2 images were made" title="How the Sim2 images were made"><i class="bx bx-info-circle"></i></button>`
    : "";
  loadModel();
}

const on = (sel, key, field, parse = (v) => v) =>
  $(sel).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b?.dataset[key]) {
      if (field === "template" || field === "method") state.sim2Preview = false;
      state[field] = parse(b.dataset[key]);
      apply();
    }
  });
on("#templates", "t", "template", Number);
on("#method", "m", "method");
$("#method-hybrid").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.preview === "sim2") {
    if (!hasSideFill()) return;
    state.sim2Preview = true;
    apply();
    return;
  }
  if (b.dataset.m) {
    state.sim2Preview = false;
    state.method = b.dataset.m;
    apply();
  }
});
on("#env", "env", "env");
on("#body-style", "body", "body");
$("#renderer").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b?.dataset.r || b.dataset.r === state.renderer) return;
  state.renderer = b.dataset.r;
  state.env = "beach"; // switching viewer always starts from the beach environment
  apply();
});

apply();

// ---- info modals ----
const INFO = {
  model: {
    title: "Model: how the 3D model is built",
    body: `
      <div class="space-y-5">
        <section class="space-y-2">
          <h4 class="text-xs font-semibold uppercase tracking-wider text-sky-400">The Three Build Modes</h4>
          <div class="grid gap-2.5 sm:grid-cols-3">
            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
              <div class="font-semibold text-zinc-100">1. Texture wrap</div>
              <p class="mt-1 text-xs leading-relaxed text-zinc-400">
                Front and back photos are projected onto a mesh using 2D garment masks and orthographic depth buffers (~10 s on local CPU).
                Only garment pixels are painted; skin, hair, and bags stay mannequin-white. Templates <b>#1, #3, #4</b> wrap onto the stock female mannequin
                (<code>female_base_mesh.glb</code>); <b>#5</b> wraps onto a male body (Basic or Model); <b>#2</b> wraps onto its <b>Hunyuan3D-2mv</b> 3D mesh so the midi skirt keeps its real drape and slit.
              </p>
            </div>
            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
              <div class="font-semibold text-zinc-100">2. Full build <span class="text-xs font-normal text-zinc-400">(Ghost build on #4)</span></div>
              <p class="mt-1 text-xs leading-relaxed text-zinc-400">
                Cloud GPU 3D reconstruction using <b>Microsoft TRELLIS.2</b> (single-view front photo with 2K texture on <b>#1 &amp; #2</b>)
                or <b>Tencent Hunyuan3D-2mv</b> (2-view front + back photos on <b>#3 &amp; #4</b>). Captures real 3D garment silhouette,
                flared hems, cowls, and open backs.
              </p>
            </div>
            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
              <div class="font-semibold text-zinc-100">3. Hybrid <span class="text-xs font-normal text-zinc-400">(Ghost + Mannequin on #4)</span></div>
              <p class="mt-1 text-xs leading-relaxed text-zinc-400">
                Combines cloud 3D geometry (<b>TRELLIS.2</b> or <b>Hunyuan3D-2mv</b>) with sharp front/back photo projection (<code>scripts/run_hybrid_poc.py</code>).
                On <b>#1</b>, a Hunyuan dress shell from model-free photos sits on the white mannequin as its own mesh.
                On <b>#2–#3</b>, the real photo dress is overlaid onto the textured 3D figure (keeping the AI texture on skin and 90° sides).
                On <b>#4</b>, the isolated 3D Ghost dress is fitted onto the clean mannequin body.
              </p>
            </div>
          </div>
        </section>

        <section class="space-y-2">
          <h4 class="text-xs font-semibold uppercase tracking-wider text-sky-400">Services &amp; AI Models Used</h4>
          <div class="grid gap-2.5 sm:grid-cols-2">
            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-xs leading-relaxed">
              <div class="mb-1.5 font-semibold text-zinc-100">Cloud &amp; Local Services</div>
              <ul class="list-disc space-y-1 pl-4 text-zinc-300">
                <li><b class="text-zinc-100">Replicate:</b> Cloud GPU API running <code>black-forest-labs/flux-kontext-pro</code> (model-free dress photos on #1), <code>tencent/hunyuan3d-2mv</code> (2-view 3D shape on #1–#4), <code>fishwowater/trellis2</code> (single-view 3D build on #2), and <code>allenhooo/lama</code> (occlusion inpainting on #4).</li>
                <li><b class="text-zinc-100">Modal (NVIDIA A10G GPU):</b> Serverless container (<code>cloud/modal_app.py</code>) with persistent HF volume caching. Runs <b>Hunyuan3D-2mv</b> shape generation and custom CUDA rasterizers for <b>Hunyuan3D-Paint</b> (#2, #3).</li>
                <li><b class="text-zinc-100">Hugging Face (Spaces &amp; Hub):</b> Runs the <a class="text-sky-400 underline hover:text-sky-300" href="https://huggingface.co/spaces/microsoft/TRELLIS.2" target="_blank" rel="noopener">Microsoft TRELLIS.2 Space</a> (#1) and hosts model checkpoints (<code>tencent/Hunyuan3D-2</code>, <code>mattmdjaga/segformer_b2_clothes</code>).</li>
                <li><b class="text-zinc-100">Local Mac Pro (AMD GPU / PyTorch MPS + CPU):</b> Runs 2D segmentation, depth-buffer registration, UV unwrapping (<code>xatlas</code> / planar), and orthographic photo baking.</li>
              </ul>
            </div>
            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-xs leading-relaxed">
              <div class="mb-1.5 font-semibold text-zinc-100">AI &amp; Vision Models</div>
              <ul class="list-disc space-y-1 pl-4 text-zinc-300">
                <li><b class="text-zinc-100">Tencent Hunyuan3D-2mv</b> (<code>hunyuan3d-dit-v2-mv</code>): Native multi-view flow-matching 3D reconstructor conditioned on both front and back photos (#2, #3, #4).</li>
                <li><b class="text-zinc-100">Hunyuan3D-Paint v2.0 Turbo:</b> Multi-view diffusion texture baker that paints untextured 3D meshes from front/back photos and fills 90° lateral sides (#2, #3).</li>
                <li><b class="text-zinc-100">Microsoft TRELLIS.2:</b> Single-image 3D shape + 2K PBR texture reconstructor (~45 s on cloud GPU; front photo only, invents the back; used on #1 &amp; #2).</li>
                <li><b class="text-zinc-100">SegFormer B2 Clothes</b> (<code>mattmdjaga/segformer_b2_clothes</code>): Human/garment semantic parser separating dress from hair, face, limbs, shoes, and bags (#1, #4).</li>
                <li><b class="text-zinc-100">rembg (<code>u2net</code> &amp; <code>u2net_cloth_seg</code>):</b> Person cutout, cloth mask segmentation, and thin spaghetti-strap recovery (#1–#5).</li>
                <li><b class="text-zinc-100">LaMa Inpainting</b> (<code>allenhooo/lama</code> + OpenCV <code>INPAINT_TELEA</code>): Fills bag/hair occlusions over fabric and heals grazing-angle UV seams (#1, #4).</li>
              </ul>
            </div>
          </div>
        </section>

        <section class="space-y-2.5">
          <h4 class="text-xs font-semibold uppercase tracking-wider text-sky-400">How Each Template Was Built</h4>
          <div class="space-y-2.5 text-xs leading-relaxed">
            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3.5">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <span class="text-sm font-semibold text-zinc-100">Template #1 — Leopard Lace Mini Slip Dress</span>
                <span class="rounded bg-purple-950/80 px-2 py-0.5 text-[11px] font-medium text-purple-300">HF Space (TRELLIS.2) + Replicate (Flux Kontext &amp; Hunyuan3D-2mv)</span>
              </div>
              <ul class="mt-2 space-y-1 text-zinc-300">
                <li><b class="text-zinc-100">Texture wrap (<code>1-wrap.glb</code>):</b> <b>Flux Kontext</b> rewrites the catalog front and back into photos of only the leopard lace slip. Those are projected onto the stock mannequin, shifted up so the lace covers the chest. Left and right profiles paint the flanks only, inside the torso, so the arms stay mannequin-white. The dress is still paint on the body.</li>
                <li><b class="text-zinc-100">Full build (<code>1-build.glb</code>):</b> Whole figure reconstructed by <b>Microsoft TRELLIS.2</b> on the Hugging Face Space cloud GPU (~45 s, front photo only; back is hallucinated by TRELLIS) with its 2K baked UV texture, scaled to mannequin height.</li>
                <li><b class="text-zinc-100">Hybrid (<code>1-hybrid.glb</code>):</b> <b>Hunyuan3D-2mv</b> reconstructs only the dress from the ideal front and back (side profiles are texture, not shape). The shell is pushed out onto the white mannequin from hem to bust and exported as two meshes, <code>mannequin</code> and <code>garment</code>, so they can be rigged later. The TRELLIS full build is left as it was.</li>
                <li><b class="text-zinc-100">Side fill (<code>outputs/canonical/1/dress-fitted.glb</code>):</b> Same Hunyuan shell as Hybrid, but a cylindrical 4-view bake that paints the flanks from the Sim2 left and right photos. Preview only; it does not replace the demo Hybrid file.</li>
              </ul>
            </div>

            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3.5">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <span class="text-sm font-semibold text-zinc-100">Template #2 — Black Sheer Side-Slit Midi</span>
                <span class="rounded bg-sky-950/80 px-2 py-0.5 text-[11px] font-medium text-sky-300">Replicate (Flux Kontext) — Sim2</span>
              </div>
              <ul class="mt-2 space-y-1 text-zinc-300">
                <li><b class="text-zinc-100">Sim2:</b> The old wrap, full build, and hybrid meshes were deleted. <b>Flux Kontext</b> rewrites the catalog front and back into garment-only shots, then turns the front into left and right profiles. Those four images sit under the catalog photos. No 3D mesh until they are accepted.</li>
              </ul>
            </div>

            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3.5">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <span class="text-sm font-semibold text-zinc-100">Template #3 — Chartreuse Halter Mini (Deep Cowl &amp; Open Back)</span>
                <span class="rounded bg-sky-950/80 px-2 py-0.5 text-[11px] font-medium text-sky-300">Replicate (Hunyuan3D-2mv) + Modal A10G (Hunyuan3D-Paint)</span>
              </div>
              <ul class="mt-2 space-y-1 text-zinc-300">
                <li><b class="text-zinc-100">Texture wrap (<code>3-wrap.glb</code>):</b> Front/back photos projected onto the stock mannequin (<code>rembg</code> + <code>u2net_cloth_seg</code>, local CPU). Flat projection — the cowl drape and open back do not exist as 3D geometry.</li>
                <li><b class="text-zinc-100">Full build (<code>3-build.glb</code>):</b> 2-view 3D figure generated by <b>Hunyuan3D-2mv</b> on <b>Replicate</b> (conditioned on both front and back photos so the open back is preserved), then textured on <b>Modal (A10G)</b> using <b>Hunyuan3D-Paint v2.0 Turbo</b> (1024² atlas) for continuous side-flank colour.</li>
                <li><b class="text-zinc-100">Hybrid (<code>3-hybrid.glb</code>):</b> <b>Hunyuan3D-2mv + Hunyuan3D-Paint Full Build with Photo Wrap Overlay</b> (<code>--no-extract</code>). Projects the sharp front and back dress photos over the painted Hunyuan figure while skin, hair, and grazing side seams retain the <b>Hunyuan3D-Paint</b> atlas (no mannequin, no white fill).</li>
              </ul>
            </div>

            <div class="rounded-lg border border-emerald-800/60 bg-emerald-950/20 p-3.5">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <span class="text-sm font-semibold text-zinc-100">Template #4 — Black Lace-Trim Satin Slip (2D Ghost Mannequin Pipeline)</span>
                <span class="rounded bg-emerald-950/90 px-2 py-0.5 text-[11px] font-medium text-emerald-300">SegFormer + Replicate (LaMa &amp; Hunyuan3D-2mv) + Mannequin Fit</span>
              </div>
              <ul class="mt-2 space-y-1 text-zinc-300">
                <li><b class="text-zinc-100">Texture wrap (<code>4-wrap.glb</code>):</b> Baseline photo projection onto the stock mannequin (<code>rembg</code> + <code>u2net_cloth_seg</code>) for comparison against the Ghost pipeline.</li>
                <li><b class="text-zinc-100">Ghost build (<code>4-build.glb</code>):</b> Instead of reconstructing the clothed person and carving the mesh afterward, <code>prepare_ghost_mannequin</code> isolates the dress in 2D first (<code>rembg</code> + <b>SegFormer</b> + <code>u2net_cloth_seg</code> strap recovery, CIE LAB skin-fringe stripping, <b>LaMa</b> bag/hair inpainting on <b>Replicate</b>, and 2 px boundary erosion onto pure <code>#FFFFFF</code>). <b>Hunyuan3D-2mv</b> (on Replicate) then reconstructs <i>only</i> the hollow 3D dress, and the ghost cutouts are UV-baked onto it.</li>
                <li><b class="text-zinc-100">Ghost + Mannequin (<code>4-hybrid.glb</code>):</b> Places the ghost-reconstructed 3D dress onto the faceless mannequin (<code>female_base_mesh.glb</code>) via <code>fit_ghost_garment</code>. The dress is scaled from the 2D garment-to-person height, then the back of the shell is pulled out past the mannequin's seat and shoulder blades (4 mm past the back skin) so the body does not poke through.</li>
              </ul>
            </div>

            <div class="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3.5">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <span class="text-sm font-semibold text-zinc-100">Template #5 — Beige Hoodie and Jeans</span>
                <span class="rounded bg-zinc-800 px-2 py-0.5 text-[11px] font-medium text-zinc-300">Local CPU Baseline</span>
              </div>
              <ul class="mt-2 space-y-1 text-zinc-300">
                <li><b class="text-zinc-100">Texture wrap:</b> Front/back photos of the beige hoodie and jeans projected onto a male body. <b>Basic</b> (<code>5-wrap-basic.glb</code>) is the plain gray mannequin. <b>Model</b> (<code>5-wrap-model.glb</code>) keeps the textured face, hair, and shoes. The Body switch shows only on this template.</li>
                <li><b class="text-zinc-100">Full build &amp; Hybrid:</b> Not generated for this template (serves as a wrap-only baseline).</li>
              </ul>
            </div>
          </div>
        </section>

        <p class="text-xs text-zinc-500">All models are pre-built and served statically so switching in the viewer is instant.</p>
      </div>`,
  },
  renderer: {
    title: "Renderer: what draws the model",
    body: `
      <p><b class="text-zinc-100">model-viewer.</b> Google's &lt;model-viewer&gt; web component: drop-in, mobile-friendly,
      AR-ready, with good default lighting. It lights the model only from the environment image and can't add
      custom lights.</p>
      <p><b class="text-zinc-100">Three.js.</b> A general 3D engine with full control. Here it adds faint extra lights
      (barely pink and blue in the nightclub) and reports live FPS, triangle and memory stats below.</p>
      <p class="text-zinc-500">Same model files in both; compare how each one looks and performs.</p>`,
  },
  env: {
    title: "Environment: the scene around the model",
    body: `
      <p>Each environment is an HDR panorama used as the background and as the main light source, so the model
      picks up its colour and brightness.</p>
      <p><b class="text-zinc-100">Beach.</b> Bright outdoor daylight: a neutral, well-lit look for judging colour.</p>
      <p><b class="text-zinc-100">Nightclub.</b> A dark disco interior: tests how the garment reads in low, tinted light.</p>
      <p class="text-zinc-500">Switching renderer resets the environment to Beach.</p>`,
  },
  sim2: {
    title: "Sim2 — ideal garment images",
    body: `
      <p>Catalog photos are not reconstruction inputs. <b>Sim2</b> is a Flux Kontext pass
      (<code>black-forest-labs/flux-kontext-pro</code>) that rewrites them before any 3D model runs.
      Entry: <code>scripts/make_canonical_views.py --dress</code> (template 1) or <code>--template 2</code>.</p>
      <p><b class="text-zinc-100">Front</b> is edited from the catalog front. <b class="text-zinc-100">Back</b> is edited from the catalog back on template 1, and from the accepted empty front on template 2 (catalog-back edits left a body in the sheer skirt). <b class="text-zinc-100">Left</b> and <b class="text-zinc-100">right</b> are 90° turns of that accepted front. Sides are texture only; Hunyuan gets front and back.</p>
      <p>On template 1 the Hybrid button still loads the original planar bake (<code>1-hybrid.glb</code>). <b class="text-zinc-100">Side fill</b> in the Model section loads a cylindrical 4-view preview that paints the flanks from those profiles, without replacing the demo file.</p>
      <p class="text-zinc-500">If the hem, print, or straps move, change the prompt and run Sim2 again. A bad image is not fixed in 3D.</p>`,
  },
  ghost: {
    title: "2D Ghost Mannequin Isolation Pre-Pass",
    body: `
      <p>Feeding full-body model photos directly into a 3D reconstructor creates a full human figure that must be
      destructively carved apart in 3D. For <b>Template 4</b>, we isolate the garment in 2D <i>before</i> any 3D
      generation starts (<code>pipeline/preprocess.py → prepare_ghost_mannequin</code>):</p>
      <p><b class="text-zinc-100">1. Person &amp; Cloth Segmentation.</b> <code>rembg</code> cuts the model out from the studio
      background, then Hugging Face <code>mattmdjaga/segformer_b2_clothes</code> (assisted by <code>u2net_cloth_seg</code>
      for thin spaghetti straps) separates the dress from human limbs, face, hair, shoes, and the handbag.</p>
      <p><b class="text-zinc-100">2. Skin-Fringe Stripping &amp; Lace Recovery.</b> We sample the model's skin tone in CIE LAB
      colour space and strip warm peach skin fringes along the outer perimeter and inside sheer lace cutouts so skin
      never bakes into the fabric.</p>
      <p><b class="text-zinc-100">3. Occlusion Detection &amp; LaMa Inpainting.</b> Where hair, hands, or the burgundy bag
      overlap the dress silhouette, we build an occlusion mask, pre-fill surrounding pixels with nearest garment tones,
      and run <b>LaMa</b> on Replicate (with OpenCV <code>INPAINT_TELEA</code> fallback) to synthesize clean satin/lace.</p>
      <p><b class="text-zinc-100">4. Alpha Erosion &amp; Pure-White Composite.</b> A 2 px morphological erosion + Gaussian
      feather removes edge halos while keeping thin straps intact, compositing the floating dress onto pure
      <code>#FFFFFF</code> for <b>Hunyuan3D-2mv</b>.</p>`,
  },
};
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-info]");
  if (!btn || !INFO[btn.dataset.info]) return;
  const { title, body } = INFO[btn.dataset.info];
  $("#info-title").textContent = title;
  $("#info-body").innerHTML = body;
  $("#info-body").scrollTop = 0;
  $("#info-modal").showModal();
});
// click on the backdrop closes the modal
$("#info-modal").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });

// ---- full-size photo lightbox ----
$("#photos").addEventListener("click", (e) => {
  const b = e.target.closest("[data-photo]");
  if (!b) return;
  $("#photo-full").src = `assets/templates/${state.template}-${b.dataset.photo}-full.jpg?v=${ASSET_REV}`;
  $("#photo-caption").textContent = `Template ${state.template}, ${b.dataset.photo}`;
  $("#photo-modal").showModal();
});
$("#ghost-photos").addEventListener("click", (e) => {
  const b = e.target.closest("[data-ghost-photo]");
  if (!b) return;
  $("#photo-full").src = `assets/templates/${state.template}-${b.dataset.ghostPhoto}-ghost.png`;
  $("#photo-caption").textContent = `Template ${state.template}, ${b.dataset.ghostPhoto} (2D Ghost Mannequin cutout)`;
  $("#photo-modal").showModal();
});
$("#sim2-photos").addEventListener("click", (e) => {
  const b = e.target.closest("[data-sim2-photo]");
  if (!b) return;
  const sim2 = SIM2[state.template];
  if (!sim2) return;
  const side = b.dataset.sim2Photo;
  $("#photo-full").src = `${sim2.dir}/${side}.png?v=${ASSET_REV}`;
  $("#photo-caption").textContent = `Template ${state.template}, Sim2 ${side}`;
  $("#photo-modal").showModal();
});
$("#photo-modal").addEventListener("click", (e) => e.currentTarget.close()); // click anywhere closes

// ---- MVP info (long-form write-up) ----
$("#mvp-body").innerHTML = MVP_INFO;
$("#mvp-open").addEventListener("click", () => {
  $("#mvp-body").scrollTop = 0;
  $("#mvp-modal").showModal();
});
$("#mvp-modal").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });
