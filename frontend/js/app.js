import { ThreeViewer } from "./three-viewer.js";
import { MVP_INFO } from "./mvp-info.js";

const HDRI = {
  beach: "assets/hdri/umhlanga_sunrise_1k.hdr",
  nightclub: "assets/hdri/disco.hdr",
};

// Pre-built models: assets/models/<template>-<method>.glb
const TEMPLATES = [1, 2, 3, 4, 5];
const TRELLIS_TEMPLATE = 1; // its full build came from TRELLIS.2 on a cloud GPU, not InstantMesh
const HAS_BUILD = new Set([1, 2, 3]);
const HAS_HYBRID = new Set([1]); // scripts/build_hybrid.py on the TRELLIS.2 build // full builds take ~6.5 min each; not generated for 4-5
const METHODS = {
  wrap: "Front/back photos projected onto a premade mannequin (~10 s to build).",
  build: "Local CPU + AMD GPU InstantMesh Reconstruction from the photos. (~6.5 min to build).",
  hybrid: "Garment cut out of the TRELLIS.2 full build and placed on the mannequin (~6 s on top of the build).",
};

const $ = (s) => document.querySelector(s);
const mv = $("#mv");
const three = new ThreeViewer($("#three"), (txt) => ($("#stats").textContent = txt));

let state = { template: 1, method: "wrap", env: "beach", renderer: "three" };
let loadedUrl = null;

$("#templates").innerHTML = TEMPLATES.map(
  (n) => `<button data-t="${n}" class="group relative cursor-pointer overflow-hidden rounded-md border border-zinc-700 transition hover:-translate-y-0.5 hover:border-zinc-400 hover:shadow-lg [&.active]:ring-2 [&.active]:ring-blue-500">
    <img src="assets/thumbs/${n}.png" alt="Template ${n}" class="aspect-square w-full object-cover opacity-75 transition group-hover:scale-110 group-hover:opacity-100 group-[.active]:opacity-100">
    <span class="absolute left-1 top-0.5 text-xs font-semibold [text-shadow:0_0_3px_#000]">${n}</span></button>`
).join("");

async function loadModel() {
  const url = `assets/models/${state.template}-${state.method}.glb`;
  if (url === loadedUrl) return;
  loadedUrl = url;
  $("#status").textContent = "Loading…";
  try {
    mv.src = url;
    await three.load(url);
    if (url === loadedUrl) $("#status").textContent = `Template ${state.template}, ${{ wrap: "texture wrap", build: "full build", hybrid: "hybrid" }[state.method]}.`;
  } catch (err) {
    $("#status").textContent = `Error: ${err.message}`;
  }
}

function apply() {
  if (state.method === "build" && !HAS_BUILD.has(state.template)) state.method = "wrap";
  if (state.method === "hybrid" && !HAS_HYBRID.has(state.template)) state.method = "wrap";
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

  const mark = (sel, key, val) =>
    document.querySelectorAll(`${sel} button`).forEach((b) => b.classList.toggle("active", b.dataset[key] === String(val)));
  mark("#templates", "t", state.template);
  mark("#method", "m", state.method);
  mark("#method-hybrid", "m", state.method);
  mark("#env", "env", state.env);
  mark("#renderer", "r", state.renderer);

  const canBuild = HAS_BUILD.has(state.template);
  $('#method button[data-m="build"]').disabled = !canBuild;
  $('#method-hybrid button[data-m="hybrid"]').disabled = !HAS_HYBRID.has(state.template);
  $("#method-hint").textContent = !canBuild
    ? `${METHODS.wrap} Full build not generated for this template.`
    : state.method === "build" && state.template === TRELLIS_TEMPLATE
      ? "Whole figure built with Microsoft TRELLIS.2 on a cloud GPU (~45 s), not InstantMesh."
      : METHODS[state.method];
  $("#photos").innerHTML = ["front", "back"]
    .map((side) => `<button data-photo="${side}" class="group relative flex-1 cursor-zoom-in overflow-hidden rounded bg-white ring-blue-500 transition hover:ring-2" aria-label="View ${side} photo full size">
      <img src="assets/templates/${state.template}-${side}.jpg" alt="" class="h-28 w-full object-contain transition group-hover:scale-105">
      <span class="absolute inset-0 flex items-center justify-center bg-black/0 text-2xl text-white opacity-0 transition group-hover:bg-black/30 group-hover:opacity-100"><i class="bx bx-zoom-in"></i></span>
      <span class="block text-center text-[10px] text-zinc-500">${side}</span></button>`)
    .join("");
  loadModel();
}

const on = (sel, key, field, parse = (v) => v) =>
  $(sel).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b?.dataset[key]) { state[field] = parse(b.dataset[key]); apply(); }
  });
on("#templates", "t", "template", Number);
on("#method", "m", "method");
on("#method-hybrid", "m", "method");
on("#env", "env", "env");
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
      <p><b class="text-zinc-100">Texture wrap.</b> The product's front and back photos are projected onto a premade,
      faceless mannequin. Only the garment is painted; the model's skin, hair and accessories are masked out.
      Fast (~10 s per product) and consistent, but the garment has no shape of its own: skirts and loose fabric
      end up painted onto the legs.</p>
      <p><b class="text-zinc-100">Full build.</b> The whole figure, body and garment, is reconstructed as a new mesh
      with InstantMesh: Zero123++ imagines side views from the front photo, the real back photo replaces the
      imagined back views, and the mesh is built from all of them. Real garment shape (skirts hang, hems flare),
      but slow (~6.5 min per product), lumpier, and it reproduces the person, not a mannequin.</p>
      <p><b class="text-zinc-100">Hybrid (template 1 only).</b> The garment is cut out of the TRELLIS.2 full build
      using the same clothing masks as texture wrap, the person is thrown away, and the garment is placed on the
      mannequin, pushed outward wherever it would sink into the body. Real garment shape on a clean mannequin.
      It adds ~6 s on top of the full build. Weak spots: the sides (no photo sees them) and sheer areas, where
      TRELLIS baked the model's skin into the fabric colour.</p>
      <p><b class="text-zinc-100">Exception: template 1.</b> Its full build was made with
      <a class="text-sky-400 underline hover:text-sky-300" href="https://huggingface.co/spaces/microsoft/TRELLIS.2" target="_blank" rel="noopener">Microsoft TRELLIS.2</a>
      on a cloud GPU instead of InstantMesh on our Mac. It took about 45 seconds and comes with a proper texture.
      Compare it with templates 2-3 to see the difference.</p>
      <p class="text-zinc-500">Both are pre-built here, so switching is instant. Full builds exist for templates 1-3 only.</p>`,
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
};
document.querySelectorAll("[data-info]").forEach((btn) =>
  btn.addEventListener("click", () => {
    const { title, body } = INFO[btn.dataset.info];
    $("#info-title").textContent = title;
    $("#info-body").innerHTML = body;
    $("#info-modal").showModal();
  })
);
// click on the backdrop closes the modal
$("#info-modal").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });

// ---- full-size photo lightbox ----
$("#photos").addEventListener("click", (e) => {
  const b = e.target.closest("[data-photo]");
  if (!b) return;
  $("#photo-full").src = `assets/templates/${state.template}-${b.dataset.photo}-full.jpg`;
  $("#photo-caption").textContent = `Template ${state.template}, ${b.dataset.photo}`;
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
