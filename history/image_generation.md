# The Strazza method — image generation

Canonical front and back views for 3D reconstruction are **generated**, not cut out of the catalog photo.

A posed e-commerce shot is the wrong picture for Hunyuan3D-2mv: the body is in the way, thin straps disappear in a segmentation mask, and the garment leans with the model. The Strazza method asks an image editor (Flux Kontext Pro on Replicate) to rewrite that shot into an empty, straight, garment-only product photo. That photo is what the reconstructor sees.

First accepted run: Guess black sequin mini, 30 Oct 2026. Plates: `outputs/canonical/guess_dress/dress/{front,back}.png`. Session notes: [docs/handoff_guess_dress.md](../docs/handoff_guess_dress.md).

## Rule

**Input is the original catalog photograph.** Not a SegFormer plate, not rembg, not a white-background cutout.

Ghost v1 (keep class pixels, drop the rest, paste on `#FFFFFF`) is a different method. It is the wrong first step when:

- Straps or other 1–2 px parts are labeled skin and vanish.
- The body is not standing straight, so the cloth silhouette is posed.
- A hand or bag sits on the garment and punching it out leaves a hole.

Feeding those broken cutouts to Kontext makes it complete a crooked, strapless blob. Give it the woman in the photo so the prompt can remove her and straighten the dress.

## Pipeline in this repo

```mermaid
flowchart LR
  catalog[Catalog front and back] --> kontext["Flux Kontext Pro"]
  kontext --> empty[Empty straight product shots]
  empty --> strip["Second edit if a body remains"]
  strip --> plates[Canonical front.png / back.png]
  plates --> hunyuan[Hunyuan later]
```

1. Outfit JSON sets `prepass_mode: ghost_v2` and `material_type: opaque` unless the cloth is actually sheer (sheer also writes a gray `(24,24,24)` vision plate; opaque does not).
2. Per-garment `prompts.front` / `prompts.back` and optional `empty_prompt` live on the garment spec ([configs/templates/guess_dress.json](../configs/templates/guess_dress.json)).
3. `scripts/make_canonical_views.py --config … --front … --back …` calls `write_prepass` → `write_dress_views` → `kontext_edit` (`pipeline/canonical.py`). Model: `black-forest-labs/flux-kontext-pro`. Token: `REPLICATE_API_TOKEN` in `.env`.
4. Front is edited from the real front. Back is edited from the real back. Do not invent the back by turning the front when the back has unique geometry (this dress: the satin bow).
5. If `empty_prompt` is set, `strip_body` runs a second Kontext pass on each written PNG to drop a leftover figure.

```bash
poetry run python scripts/make_canonical_views.py \
  --config configs/templates/guess_dress.json \
  --front test_images/guess_front.png \
  --back test_images/guess_back.png
```

Look at the PNGs. Then stop. Shape generation is a later stage.

## What the prompt has to do

The prompt is the method. It must:

- Remove the person completely (skin, hair, face, arms, hands, legs, shoes). Empty garment, no mannequin inside.
- Plain white background, full garment in frame, centered.
- Hang **straight and symmetric**, as a studio product shot, not leaning with the model’s hip.
- Name the design parts that a mask would drop or a rewrite would invent: straps, bandeau, hem length, prints, bows.
- Keep fabric as that fabric (sequins stay sequins, not smooth satin or a printed fake).
- Fill only the occlusion the body caused (hand on hip, bag on the skirt) with the same cloth.

Guess front and back copy is in `guess_dress.json`. Do not reuse leopard-slip or hoodie prompts.

## Shape vs texture

These generated plates are the **shape** input: both views, so Hunyuan sees straps, an open or closed back, and extra cloth like a bow.

Photographic pixels from the catalog can still be projected onto the fitted mesh later. Kontext will redraw sparkle and may close an open back. That is the cost of a usable hull. Do not treat the generated PNG as the only texture source unless you choose to.

## What this is not

- Not Ghost v1 segmentation.
- Not Flux-from-noise text-to-image. Kontext **edits** the catalog frame (`input_image` + prompt, `aspect_ratio: match_input_image`).
- Not sheer-fill. Gray fill is only for reconstructors tunneling through sheer cloth.
- Not left/right profiles until front and back are accepted.
