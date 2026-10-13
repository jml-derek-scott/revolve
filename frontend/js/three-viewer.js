import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";

export class ThreeViewer {
  constructor(container, onStats) {
    this.container = container;
    this.onStats = onStats;
    this.active = false;
    this.model = null;
    this.hdrCache = {};

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(devicePixelRatio);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.position.set(0, 0, 2.2);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.lights = new THREE.Group(); // per-environment extra lights
    this.scene.add(this.lights);

    new ResizeObserver(() => this.resize()).observe(container);
    this.frames = 0;
    this.last = performance.now();
    this.loop = this.loop.bind(this);
    this.renderer.setAnimationLoop(this.loop);
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setActive(on) {
    this.active = on;
    if (on) this.resize();
  }

  async setEnvironment(name, url) {
    this.lights.clear();
    // The HDR (scene.environment) is the main light, as in model-viewer; these are faint accents.
    if (name === "nightclub") {
      // Barely-pink and barely-blue accents so the model reads against the dark HDRI.
      const pink = new THREE.DirectionalLight(0xffe8f0, 0.35); pink.position.set(-2, 2, 3);
      const blue = new THREE.DirectionalLight(0xe8f0ff, 0.35); blue.position.set(2, 1.5, 3);
      this.lights.add(pink, blue);
      this.renderer.toneMappingExposure = 1.08;
    } else {
      // Warm ~5500K sun.
      const sun = new THREE.DirectionalLight(0xffe4c4, 1.0); sun.position.set(3, 4, 2);
      this.lights.add(sun);
      this.renderer.toneMappingExposure = 1.0;
    }
    this.envUrl = url;
    if (!this.hdrCache[url]) {
      this.hdrCache[url] = new RGBELoader().loadAsync(url).then((t) => {
        t.mapping = THREE.EquirectangularReflectionMapping;
        return t;
      });
    }
    const hdr = await this.hdrCache[url];
    if (this.envUrl !== url) return; // a newer setEnvironment() superseded this one
    this.scene.environment = this.scene.background = hdr;
  }

  async load(url) {
    this.loading = url;
    const gltf = await new GLTFLoader().loadAsync(url);
    if (this.loading !== url) return; // a newer load() superseded this one
    if (this.model) this.scene.remove(this.model);
    this.model = gltf.scene;
    this.model.traverse((o) => { if (o.material) o.material.side = THREE.DoubleSide; });
    this.scene.add(this.model);
    this.frame();
  }

  // Point the camera at the model's bounding box so any size/origin fits the view.
  frame() {
    const box = new THREE.Box3().setFromObject(this.model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const dist = (Math.max(size.y, size.x / this.camera.aspect) / 2 / Math.tan(fov / 2)) * 1.25;
    this.controls.target.copy(center);
    this.camera.position.set(center.x, center.y, center.z + dist);
    this.camera.near = dist / 100;
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
  }

  loop(now) {
    if (!this.active) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.frames++;
    if (now - this.last >= 1000) {
      const i = this.renderer.info;
      const heap = performance.memory ? ` | heap ${(performance.memory.usedJSHeapSize / 2 ** 20).toFixed(0)} MB` : "";
      this.onStats(`Three.js: ${this.frames} FPS | ${i.render.triangles} tris | ${i.memory.textures} tex${heap}`);
      this.frames = 0;
      this.last = now;
    }
  }
}
