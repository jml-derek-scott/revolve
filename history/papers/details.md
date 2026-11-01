# Why ControlNet is the critical choice

Filling sheer holes with gray solves input segmentations for Hunyuan3D, but orthographic photo projection inevitably produces severe grazing-angle void artifacts along side seams and flanks. Basic spatial algorithms (like Telea or Navier-Stokes) blur these regions and create visible seams. ControlNet depth + normal-guided diffusion synthesizes geometrically consistent, photorealistic fabric textures over lateral voids conditioned directly on the target GLB geometry

Act as a 3D Graphics & Computer Vision Engineer. Implement a Python post-processing pass using PyTorch, Diffusers, and PyTorch3D/Trimesh to resolve 90° grazing-angle texture voids (lateral flanks, underarms, side seams) on our unposed 3D garment GLB meshes.

### Workflow Requirements:
1. Camera Setup & Rendering:
   - Position orthographic/perspective cameras at lateral side views (+X and -X directions relative to the garment mesh).
   - Project the front (+Z) and back (-Z) catalog photos onto the mesh UV atlas to identify painted vs. unpainted texels.
   - For each lateral camera view (+X, -X), render three synchronized 2D map channels at 1024x1024:
     a) Current Projection Render (RGB image with unprojected/stretched flank areas).
     b) Inpainting Mask M_void (Binary mask where 1 = unpainted/stretched grazing-angle pixels, 0 = valid front/back projected pixels).
     c) Geometry Conditioning: Surface Normal Map N_side and Normalized Depth Map D_side rendered directly from the 3D GLB mesh.

2. ControlNet Inpainting Engine:
   - Instantiate a Stable Diffusion / SDXL ControlNet Inpainting Pipeline (e.g., diffusers StableDiffusionControlNetInpaintPipeline).
   - Pass N_side and D_side as ControlNet conditioning inputs alongside M_void and the RGB render.
   - Condition the diffusion generation using the garment's text prompt (e.g., "seamless lateral seam, [garment description], detailed fabric texture, photorealistic, 8k") with a low strength (0.75-0.85) to blend boundaries seamlessly into existing front/back projected regions.

3. UV Re-projection & Seamless Blending:
   - Back-project the inpainted lateral RGB images back onto the mesh UV texture atlas.
   - Apply a distance-transform feathering blend along mask edges (M_void) to eliminate hard seam boundaries between camera projection zones and diffusion-generated zones.

Deliverable: A modular Python module `flank_inpaint.py` containing a class `FlankTextureInpainter` with an `inpaint_side_seams(glb_path, front_img, back_img, prompt)` method that outputs the fully textured, seam-free GLB mesh.

## Production Implementation Blueprint

Below is the execution module for `flank_inpaint.py` using diffusers and PyTorch3D / Open3D:

```python
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler

class FlankTextureInpainter:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Load Depth & Normal ControlNets
        controlnet_depth = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
        )
        controlnet_normal = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_normalbae", torch_dtype=torch.float16
        )
        
        # Initialize Multi-ControlNet Inpainting Pipeline
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=[controlnet_depth, controlnet_normal],
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def inpaint_flank_view(
        self, 
        rendered_rgb: Image.Image, 
        mask_void: Image.Image, 
        depth_map: Image.Image, 
        normal_map: Image.Image, 
        prompt: str
    ) -> Image.Image:
        """
        Inpaints lateral 90 degree grazing angle voids using dual Depth + Normal ControlNet guidance.
        """
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        # Process inpainting
        inpainted_image = self.pipe(
            prompt=f"{prompt}, seamless fabric texture, high quality, highly detailed",
            negative_prompt="blurry, distorted, non-matching pattern, holes, seams, background bleed",
            image=rendered_rgb,
            mask_image=mask_void,
            control_image=[depth_map, normal_map],
            strength=0.85,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator
        ).images[0]
        
        # Feather mask edges to eliminate boundaries
        mask_np = np.array(mask_void.convert("L"))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=2)
        blurred_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (15, 15), 0)
        blurred_mask_3ch = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

        # Composite blending
        orig_np = np.array(rendered_rgb).astype(np.float32)
        inpaint_np = np.array(inpainted_image).astype(np.float32)
        blended_np = inpaint_np * blurred_mask_3ch + orig_np * (1.0 - blurred_mask_3ch)
        
        return Image.fromarray(blended_np.astype(np.uint8))

```