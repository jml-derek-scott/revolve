import { ThreeViewer } from "./three-viewer.js?v=shop2";

const HDRI = {
  beach: "assets/hdri/umhlanga_sunrise_1k.hdr",
  nightclub: "assets/hdri/disco.hdr",
};

const ENV_ON =
  "env-btn active flex items-center justify-center gap-1.5 py-2.5 px-3 bg-white text-black font-mono text-xs tracking-wider uppercase font-semibold border border-white transition-all cursor-pointer rounded-none";
const ENV_OFF =
  "env-btn flex items-center justify-center gap-1.5 py-2.5 px-3 bg-neutral-900 text-neutral-400 hover:text-white font-mono text-xs tracking-wider uppercase font-semibold border border-neutral-800 hover:border-neutral-600 transition-all cursor-pointer rounded-none";
const THUMB_ON =
  "garment-thumb relative w-[60px] h-[60px] shrink-0 border-2 border-white overflow-hidden group cursor-pointer transition-all rounded-none";
const THUMB_OFF =
  "garment-thumb relative w-[60px] h-[60px] shrink-0 border border-neutral-800 hover:border-neutral-400 overflow-hidden group cursor-pointer transition-all rounded-none";

const $ = (id) => document.getElementById(id);

let items = [];
let current = -1;
let viewingShell = false;
let environment = "beach";
let rotating = true;

const viewer = new ThreeViewer($("canvas-container"), () => {
  const stats = viewer.lastStats;
  $("stat-fps").textContent = String(stats.fps);
  $("stat-heap").textContent = stats.heap == null ? "–" : `${(stats.heap / 2 ** 20).toFixed(1)}MB`;
  if (current >= 0) {
    $("stat-verts").textContent = stats.vertices.toLocaleString();
    $("stat-polys").textContent = stats.polygons.toLocaleString();
  }
});
viewer.setActive(true);
viewer.setAutoRotate(true);
viewer.renderer.domElement.className = "block h-full w-full cursor-grab active:cursor-grabbing";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function chipHtml(item) {
  const chips = [];
  if (item.current) chips.push({ label: "Current", current: true });
  for (const label of item.chips || []) chips.push({ label, current: false });
  return chips
    .map(
      ({ label, current: isCurrent }) =>
        `<span class="chip${isCurrent ? " chip-current" : ""}">${escapeHtml(label)}</span>`
    )
    .join("");
}

function paintItem(index) {
  const item = items[index];
  const title = viewingShell ? `${item.name} · shell` : item.name;
  $("active-garment-title").textContent = title;
  $("model-chips").innerHTML = chipHtml(item);
  $("model-description-text").textContent = item.details;
  $("macro-thumb-0").src = item.thumbnail1;
  $("macro-thumb-0").alt = `${item.name}, front`;
  $("macro-thumb-1").src = item.thumbnail2;
  $("macro-thumb-1").alt = `${item.name}, back`;
  $("spec-name").textContent = item.name;
  $("spec-chips").innerHTML = chipHtml(item);
  $("spec-details").textContent = item.details;
  $("spec-file").textContent = viewingShell && item.shell ? item.shell.file : item.file;
  $("spec-long").innerHTML = renderSpec(item);
  renderIdeals(item);
  renderShellCard(item);
  document.querySelectorAll(".garment-thumb").forEach((thumb, i) => {
    thumb.className = i === index ? THUMB_ON : THUMB_OFF;
  });
}

function renderSpec(item) {
  const spec = item.spec;
  if (!spec) return "";
  const models = (spec.models || [])
    .map((line) => `<li class="text-neutral-300">${escapeHtml(line)}</li>`)
    .join("");
  const steps = (spec.steps || [])
    .map(
      (step) => `<div class="border border-neutral-800 bg-neutral-900/50 p-3">
        <div class="font-mono text-[11px] uppercase tracking-widest text-white">${escapeHtml(step.title)}</div>
        <p class="mt-1.5 font-mono text-[11px] text-neutral-400 leading-relaxed">${escapeHtml(step.body)}</p>
      </div>`
    )
    .join("");
  return `
    <div>
      <span class="text-[11px] font-mono uppercase tracking-widest text-neutral-400 block mb-2">Models &amp; services</span>
      <ul class="list-disc space-y-1 pl-4 font-mono text-[11px] leading-relaxed">${models}</ul>
    </div>
    <div>
      <span class="text-[11px] font-mono uppercase tracking-widest text-neutral-400 block mb-2">How this mesh was made</span>
      <div class="space-y-2">${steps}</div>
    </div>`;
}

function renderIdeals(item) {
  const ideals = item.ideals || [];
  $("spec-ideals").innerHTML = ideals
    .map(
      (photo, i) => `<button type="button" data-ideal="${i}" class="group border border-neutral-800 bg-white hover:border-white transition-colors cursor-pointer rounded-none">
        <img src="${escapeHtml(photo.src)}" alt="${escapeHtml(photo.label)}" class="h-28 w-full object-contain">
        <span class="block bg-neutral-950 py-1 text-center font-mono text-[9px] uppercase tracking-widest text-neutral-400 group-hover:text-white">${escapeHtml(photo.label)}</span>
      </button>`
    )
    .join("");
}

function renderShellCard(item) {
  const wrap = $("spec-shell-wrap");
  if (!item.shell) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  $("spec-shell-thumb").src = item.shell.thumb;
  $("spec-shell-thumb").alt = item.shell.label;
  $("spec-shell-label").textContent = item.shell.label;
}

async function showMesh(url) {
  $("stat-note").textContent = "LOADING";
  try {
    const shown = await viewer.load(url);
    if (!shown) return;
    $("stat-note").textContent = viewingShell ? "SHELL" : "";
    const counts = viewer.meshCounts();
    $("stat-verts").textContent = counts.vertices.toLocaleString();
    $("stat-polys").textContent = counts.polygons.toLocaleString();
  } catch (err) {
    $("stat-note").textContent = "COULD NOT LOAD";
    console.error(err);
  }
}

async function selectGarment(index) {
  if (index < 0 || index >= items.length) return;
  const same = index === current && !viewingShell;
  if (same) return;
  const item = items[index];
  current = index;
  viewingShell = false;
  paintItem(index);
  await showMesh(item.file);
}

async function loadShell() {
  const item = items[current];
  if (!item?.shell) return;
  closeModal("modal-specs");
  viewingShell = true;
  paintItem(current);
  await showMesh(item.shell.file);
}

function setEnvironment(name) {
  if (!HDRI[name]) return;
  environment = name;
  $("btn-env-beach").className = name === "beach" ? ENV_ON : ENV_OFF;
  $("btn-env-nightclub").className = name === "nightclub" ? ENV_ON : ENV_OFF;
  viewer.setEnvironment(name, HDRI[name]);
}

function openPhoto(src, side, title, description) {
  const photo = $("modal-img-src");
  photo.classList.remove("hidden");
  photo.src = src;
  photo.alt = `${title}, ${side.toLowerCase()}`;
  $("modal-img-title").textContent = title;
  $("modal-img-side").textContent = side;
  $("modal-img-desc").textContent = description;
  $("modal-macro").classList.remove("hidden");
}

function openImageModal(which) {
  const item = items[current];
  if (!item) return;
  const full = which === 0 ? item.staticimage1 : item.staticimage2;
  const side = which === 0 ? "Catalog front" : "Catalog back";
  openPhoto(full, side, item.name, item.details);
}

function openSpecsModal() {
  if (current >= 0) paintItem(current);
  $("modal-specs").classList.remove("hidden");
}

function openControlsModal() {
  $("modal-controls").classList.remove("hidden");
}

function closeModal(id) {
  $(id).classList.add("hidden");
}

function toggleSidebar(open) {
  $("sidebar").classList.toggle("-translate-x-full", !open);
  $("sidebar-backdrop").classList.toggle("hidden", !open);
}

function scrollCarousel(direction) {
  $("carousel-track").scrollBy({ left: direction * 140, behavior: "smooth" });
}

function resetCamera() {
  viewer.resetView();
}

function toggleAutoRotate() {
  rotating = !rotating;
  viewer.setAutoRotate(rotating);
  $("rotate-status-label").textContent = `ROTATE: ${rotating ? "ON" : "OFF"}`;
  $("rotate-icon").classList.toggle("animate-spin-slow", rotating);
}

function buildCarousel() {
  $("carousel-track").innerHTML = items
    .map(
      (item, i) => `<button type="button" class="${i === 0 ? THUMB_ON : THUMB_OFF}" data-index="${i}" title="${escapeHtml(item.name)}" aria-label="${escapeHtml(item.name)}">
        <img src="${escapeHtml(item.thumb || item.thumbnail1)}" alt="" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300">
        ${item.current ? '<span class="pointer-events-none absolute bottom-0 inset-x-0 bg-emerald-500 py-px text-center font-mono text-[7px] font-bold uppercase tracking-wider text-emerald-950">Current</span>' : ""}
      </button>`
    )
    .join("");
}

Object.assign(window, {
  selectGarment,
  loadShell,
  setEnvironment,
  openImageModal,
  openSpecsModal,
  openControlsModal,
  closeModal,
  toggleSidebar,
  scrollCarousel,
  resetCamera,
  toggleAutoRotate,
});

$("carousel-track").addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (button) selectGarment(Number(button.dataset.index));
});

$("spec-ideals").addEventListener("click", (event) => {
  const button = event.target.closest("[data-ideal]");
  if (!button || current < 0) return;
  const item = items[current];
  const photo = item.ideals?.[Number(button.dataset.ideal)];
  if (!photo) return;
  openPhoto(photo.src, photo.label, item.name, item.details);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  for (const id of ["modal-macro", "modal-specs", "modal-controls"]) closeModal(id);
  toggleSidebar(false);
});

const loaded = await fetch("items.json?v=shop6");
if (!loaded.ok) throw new Error(`items.json ${loaded.status}`);
items = await loaded.json();
buildCarousel();
setEnvironment(environment);
await selectGarment(0);
