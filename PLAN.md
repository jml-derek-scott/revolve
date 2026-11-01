# TASK SPECIFICATION: 2-View Hybrid Garment 3D Reconstruction Pipeline

## 1. System Overview & Constraints
* **Inputs:** Exactly two e-commerce product photos: Front (0° azimuth, 0° elevation) and Back (180° azimuth, 0° elevation).
* **Goal:** Produce an interactive `.glb` outfit model featuring a clean, faceless mannequin wearing an extracted 3D garment mesh with sharp real-photo UV textures and realistic drape.
* **Mannequin Asset:** `mannequins/female_base_mesh.glb`.
* **Core Strategy (Hybrid):** Combine 3D silhouette and drape from 3D AI reconstruction running on Modal (TRELLIS.2 / Hunyuan3D-2 MV) with the clean standard mannequin body, projecting the original front and back high-resolution photos onto the garment UV space[cite: 1, 2, 3, 5].

---

## 2. Critical Engine & Mesh Constraints (Must Follow)
1. **Double-Skinned Mannequin Shell:** `mannequins/female_base_mesh.glb` has dual inner and outer surfaces ~5 mm apart with non-manifold winding. Do NOT use `mesh.contains()` or vertex normal ray-casting. Always determine visibility and depth using orthographic 2D z-buffers ($z_{\max}$ and $z_{\min}$).
2. **Split Vertices on UV Seams:** AI-generated meshes (such as TRELLIS.2 outputs) split vertex indices along UV seams. Always invoke `mesh.merge_vertices(merge_tex=True, merge_norm=True)` on a working copy before evaluating face adjacency or topological components.
3. **Never Call `trimesh.split()`:** Splitting textured AI meshes deep-copies materials and textures thousands of times, causing memory exhaustion. Group connected geometry exclusively with `trimesh.graph.connected_component_labels` and update faces via masking.
4. **Coordinate Convention:** Input pipeline uses glTF convention: $+Y$ is Up, $+Z$ is Front (facing user), $+X$ is Model Left.

---

## 3. Architecture & Data Flow

```
Front Photo (0°) ──┐
├─► [Stage 1: 2D Preprocessing] ──► Clean Inpainted Garment Masks
Back Photo (180°) ─┘
│
├─► [Stage 2: 3D Hull/Mesh Generation (Cloud/CUDA)] ──► Raw Clothed Mesh
│
▼
[Stage 3: Hybrid Garment Segmentation & Registration]
├── Depth-project mesh triangles against 2D garment masks
├── Morphological seam repair (dilate 12 / erode 12 rings)
├── Align height/floor to mannequins/female_base_mesh.glb
└── Push torso-penetrating vertices outward (+4 mm buffer)
│
▼
[Stage 4: UV Projection & Side-Seam Synthesis]
├── Parameterize clean UV atlas (xatlas / Smart UV)
├── Rasterize P(x, y, z) and N(nx, ny, nz) in UV space
├── Front/Back projection with grazing-angle cutoff (|nz| >= 0.25)
├── UV-space inpainting across untextured lateral side seams
└── 8 px margin dilation for mipmap padding
│
▼
[Stage 5: glTF PBR Assembly & Export] ──► final-hybrid.glb (< 3MB target)
```

---

## 4. Implementation Tasks

### Task 1: 2D Preprocessing & Occlusion Inpainting (`pipeline/preprocess.py`)
* Extract full-person silhouette using `rembg` (default session).
* Extract clothing silhouette using `rembg` session `u2net_cloth_seg`.
* Compute horizontal torso centerline $c_x$ and bounding top/bottom indices.
* Identify occluding regions on the clothing (e.g., hair over chest/shoulders, crossed hands) by subtracting the clothing mask from the full-body mask[cite: 1, 5].
* Inpaint occluded clothing sections in the 2D source photo using `cv2.inpaint(img, occlusion_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)` prior to texture mapping.
* **Occlusion Cleanup:** If hair, jewelry, or hands cross over the garment silhouette, compute the occlusion mask and inpaint the missing fabric:
  * Fast local mode: `cv2.inpaint(img, occlusion_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)`[cite: 5].
  * High-fidelity API mode (via `REPLICATE_API_TOKEN`): Call a hosted LaMa endpoint (`replicate.run("allen-act/lama", input={"image": open(img_path, "rb"), "mask": open(mask_path, "rb")})`) to synthesize realistic fabric textures across occluded areas before UV projection.

### Task 2 Update: Multi-Provider 3D Reconstruction (`backend/app/services/multi_view.py`)
* Support a three-tiered provider strategy in `MultiViewReconstructor`:
  1. **Primary (Modal):** Custom serverless worker running multi-view reconstruction (TRELLIS.2 / Hunyuan3D-2) with weights cached from Hugging Face Hub[cite: 1, 3, 5].
  2. **Direct API Fallback (Replicate):** Calls `tencent/hunyuan3d-2` or `firtoz/trellis` using `replicate.run()` when Modal container is cold or during quick local prototyping.
  3. **Research Fallback (Hugging Face Spaces):** Direct query to `microsoft/TRELLIS.2` via `gradio_client.Client`[cite: 1, 2, 5].

### Task 2: Modal Cloud 3D Reconstruction Worker with Hugging Face Integration (`cloud/modal_app.py` & `backend/app/services/multi_view.py`)
* **Modal Environment & Weight Caching (`cloud/modal_app.py`):**
  * Define `app = modal.App("garment-3d-reconstruction")`.
  * Build a container image (`modal.Image`) with CUDA 12 support, PyTorch, `nvdiffrast`, and multi-view dependencies.
  * Attach a persistent `modal.Volume("hf-hub-cache")` mounted to `/root/.cache/huggingface` so checkpoints (TRELLIS / Hunyuan3D-2) download once via `snapshot_download` and persist across cold starts.
  * Inject `modal.Secret.from_name("huggingface-secret")` to authenticate Hugging Face Hub access.
* **Inference Function (`cloud/modal_app.py`):**
  * Decorate the reconstruction function with `@app.function(gpu="A10G", timeout=180, container_idle_timeout=60, volumes={"/root/.cache/huggingface": hf_volume}, secrets=[modal.Secret.from_name("huggingface-secret")])`.
  * Accept `front_bytes: bytes` and `back_bytes: bytes`, execute 3D reconstruction, and return the raw output `.glb` as `bytes`.
* **Backend Client Integration (`backend/app/services/multi_view.py`):**
  * Replace the `NotImplementedError` in `MultiViewReconstructor.reconstruct()`.
  * Convert the pre-aligned front and back PIL images from `_prepare_views()` to PNG byte streams (`io.BytesIO`).
  * Call `modal.Function.lookup("garment-3d-reconstruction", "generate_3d_mesh").remote(front_bytes, back_bytes)`.
  * Parse the returned `.glb` bytes directly into a `trimesh.Scene` and pass it down to Stage 3 for garment extraction.
* **Hugging Face Spaces Fallback Mode (`backend/app/services/multi_view.py`):**
  * Add a configurable fallback provider using `gradio_client`.
  * If Modal is unreachable or when running quick containerless experiments, query the `microsoft/TRELLIS.2` Space directly via `gradio_client.Client`.



### Task 3: Garment Extraction & Torso Registration (`pipeline/garment_extractor.py`)
* Load the reconstructed 3D mesh returned from the Modal worker[cite: 1, 5].
* Sample points across surface to compute front ($z_{\max}$) and back ($z_{\min}$) orthographic depth buffers[cite: 1, 5].
* Assign mesh triangles to front or back by z-proximity, project triangle centers to image coordinates, and retain faces whose centers fall within the 2D garment mask[cite: 1, 5].
* Repair mesh topology:
  ```python
  garment_mesh.merge_vertices(merge_tex=True, merge_norm=True)  #[cite: 1]
  # Dilate face selection by 12 rings, erode by 12 rings to bridge lace/holes[cite: 1]
  # Prune disconnected components with face count < 300 using connected_component_labels[cite: 1]
  ```
* Register garment to mannequins/female_base_mesh.glb:
  * Scale and translate garment bounding box to match mannequin floor $y_{\min}$ and total height[cite: 1].
  * Build front and back depth maps of the mannequin torso (strip arms below $y = 1.32\text{ m}$)[cite: 1].
  * Detect garment vertices lying inside the torso boundary and offset outward: $z_{\text{front}} = z_{\text{torso}} + 4\text{ mm}$, $z_{\text{back}} = z_{\text{torso}} - 4\text{ mm}$[cite: 1].

### Task 3: Angle-Gated UV Projection & Seam Inpainting (pipeline/texture_bake.py)

Generate new UV coordinates for the extracted garment mesh using xatlas (or Blender Headless unwrapping) to avoid overlapping projection seams.
In UV space (2048×2048), rasterize triangle 3D world positions $P(x, y, z)$ and surface normals $N(n_x, n_y, n_z)$ using barycentric coordinates[cite: 1].
Implement angle-gated photo projection:For each UV texel $(u, v)$ with world normal $N$:Front facing condition: $n_z > 0.25$ and $\vert{}P_z - z_{\text{buf\_front}}\vert{} < 0.015\text{ m}$[cite: 1].
Back facing condition: $n_z < -0.25$ and $\vert{}P_z - z_{\text{buf\_back}}\vert{} < 0.015\text{ m}$[cite: 1].
Grazing angles ($\vert{}n_z\vert{} \le 0.25$) are flagged as UNASSIGNED (side-seam zone).
Synthesize side seams:
  Apply 
  cv2.inpaint(uv_texture, unassigned_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA) to blend color smoothly across the left and right rib/hip boundaries.
Run an 8-pixel UV island dilation on the filled texture map to eliminate black border fringes during Three.js mipmapping[cite: 1].

### Task 4: Scene Assembly & Export (pipeline/export_glb.py)

Assemble composite scene in Trimesh:Sub-mesh 1: mannequin with off-white vertex color [232, 229, 224, 255][cite: 1].
Sub-mesh 2: garment with the newly baked PBR UV diffuse map[cite: 1].
Set standard PBR fabric attributes: roughnessFactor = 0.8, metallicFactor = 0.05.
Export scene as .glb[cite: 1].
Run gltf-transform optimize (or Draco mesh compression) to ensure final .glb size remains under 3 MB.   

## 5. Algorithmic Reference Implementations

### 5.1 Modal Serverless App (cloud/modal_app.py)

```
### 5.1 Modal Serverless App with Hugging Face Caching (`cloud/modal_app.py`)
import io
import modal

app = modal.App("garment-3d-reconstruction")

# Mount persistent volume so HF checkpoints are cached across cold starts
hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "numpy<2.0.0",
        "Pillow",
        "trimesh",
        "ninja",
        "huggingface_hub",
        "gradio_client",
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=180,
    container_idle_timeout=60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_3d_mesh(front_bytes: bytes, back_bytes: bytes) -> bytes:
    from huggingface_hub import snapshot_download
    from PIL import Image

    # 1. Pull / verify weights from Hugging Face Hub (stored in persistent volume)
    model_dir = snapshot_download(repo_id="microsoft/TRELLIS-image-large")

    front_img = Image.open(io.BytesIO(front_bytes)).convert("RGB")
    back_img = Image.open(io.BytesIO(back_bytes)).convert("RGB")

    # 2. Run multi-view reconstruction with the downloaded HF weights
    # glb_output_path = run_trellis_or_hunyuan(model_dir, front_img, back_img)

    with open("/tmp/output.glb", "rb") as f:
        return f.read()
```

### 5.2 FastAPI Modal Integration (backend/app/services/multi_view.py)

```python
import io
import modal
from PIL import Image
from backend.app.services.single_view import SingleViewReconstructor  # Base class[cite: 1]

class MultiViewReconstructor:
    def __init__(self):
        self._fn = None

    def _get_remote_fn(self):
        if self._fn is None:
            self._fn = modal.Function.lookup("garment-3d-reconstruction", "generate_3d_mesh")
        return self._fn

    def reconstruct(self, front_img: Image.Image, back_img: Image.Image) -> bytes:
        """
        Calls Modal cloud worker with aligned front and back cutouts.
        """
        # Prepare front & back in-memory buffers
        front_buf = io.BytesIO()
        back_buf = io.BytesIO()
        front_img.save(front_buf, format="PNG")
        back_img.save(back_buf, format="PNG")

        remote_fn = self._get_remote_fn()
        glb_bytes = remote_fn.remote(front_buf.getvalue(), back_buf.getvalue())
        return glb_bytes
```

### 5.3 Angle-Gated UV Projection

```python
import numpy as np
import cv2

def compute_grazing_aware_projection(
    uv_positions: np.ndarray,   # (H, W, 3) 3D surface coordinates per UV texel[cite: 1, 5]
    uv_normals: np.ndarray,     # (H, W, 3) 3D surface normals per UV texel[cite: 1, 5]
    uv_valid_mask: np.ndarray,  # (H, W) bool, true if UV texel is covered by mesh[cite: 1, 5]
    front_img: np.ndarray,      # (H_in, W_in, 3)
    back_img: np.ndarray,       # (H_in, W_in, 3)
    front_zbuf: np.ndarray,     # (H_grid, W_grid) depth buffer[cite: 1, 5]
    back_zbuf: np.ndarray,      # (H_grid, W_grid) depth buffer[cite: 1, 5]
    normal_threshold: float = 0.25
) -> np.ndarray:
    tex_h, tex_w = uv_positions.shape[:2]
    out_rgb = np.zeros((tex_h, tex_w, 3), dtype=np.uint8)
    assigned_mask = np.zeros((tex_h, tex_w), dtype=np.uint8)

    nz = uv_normals[:, :, 2]  # Z normal component in glTF space (+Z = front)[cite: 1, 5]

    # Front projection (|nz| > threshold)
    front_valid = uv_valid_mask & (nz >= normal_threshold)
    # Check depth visibility against front z-buffer (|z - zbuf| < 1.5cm)[cite: 1, 5]
    # out_rgb[front_valid] = front_img[py, px]
    # assigned_mask[front_valid] = 255

    # Back projection
    back_valid = uv_valid_mask & (nz <= -normal_threshold)
    # Check depth visibility against back z-buffer[cite: 1, 5]
    # out_rgb[back_valid] = back_img[py, px]
    # assigned_mask[back_valid] = 255

    # Fill lateral side-seam void where normal was grazing (|nz| < threshold)[cite: 5]
    seam_mask = (uv_valid_mask & (assigned_mask == 0)).astype(np.uint8) * 255
    if np.any(seam_mask):
        out_rgb = cv2.inpaint(out_rgb, seam_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)  #[cite: 5]
        assigned_mask[uv_valid_mask] = 255

    # Dilate UV boundaries by 8 pixels to prevent mipmap seam bleeding[cite: 1, 5]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_rgb = out_rgb.copy()
    for _ in range(8):  #[cite: 1, 5]
        dilated_mask = cv2.dilate(assigned_mask, kernel)
        edge_pixels = (dilated_mask == 255) & (assigned_mask == 0)
        # Propagate boundary texels
        assigned_mask = dilated_mask

    return dilated_rgb
```

### 5.4 Hugging Face Spaces Fallback Client

```python

### 5.4 Hugging Face Spaces Direct Fallback (`pipeline/hf_client.py`)
from gradio_client import Client, handle_file

def generate_via_hf_spaces(front_image_path: str, hf_token: str = None) -> str:
    """
    Fallback 3D generation directly via Hugging Face Space (e.g. microsoft/TRELLIS.2)
    without running a custom GPU container.
    """
    client = Client("microsoft/TRELLIS.2", hf_token=hf_token)  #[cite: 1, 2]
    result = client.predict(
        image=handle_file(front_image_path),
        api_name="/predict"
    )
    # Returns the local path to the downloaded .glb
    return result

```

### 5.5 Replicate Direct Fallback Client (`pipeline/replicate_client.py`)

```python
import io
import replicate
import requests

def generate_via_replicate(front_image_path: str, model_id: str = "tencent/hunyuan3d-2") -> bytes:
    """
    Zero-container 3D model generation fallback via Replicate API.
    Returns raw GLB bytes directly.
    """
    with open(front_image_path, "rb") as img_file:
        output = replicate.run(
            model_id,
            input={"image": img_file}
        )
    
    # Replicate returns a FileOutput or URL pointing to the .glb
    glb_url = str(output) if isinstance(output, str) else output.url
    response = requests.get(glb_url)
    response.raise_for_status()
    return response.content
```

6. Acceptance Criteria

- Cloud Execution Speed: Modal reconstruction step finishes in under 60 seconds (warm container) on an A10G GPU.   
- Geometry Drape: Skirts, flared hems, and loose outerwear retain independent 3D separation from the mannequin legs (no flat painted limbs).   
- Surface Sharpness: Front and back fabric patterns (florals, logos, lace) match raw photo sharpness without diffuse blurring[cite: 1, 2, 5].
- Flank Quality: 90° lateral side profiles are smoothly interpolated via seam inpainting without horizontal streaking or stretch artifacts[cite: 1, 3, 5].
- Body Penetration: Zero clipping of the mannequin mesh through the chest, back, or waist (minimum +4 mm clearance maintained)[cite: 1, 5].
- Local Runtime: Steps 3 through 5 (extraction, projection, assembly) execute on local CPU/MPS in under 20 seconds[cite: 1, 2, 5].