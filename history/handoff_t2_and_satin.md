# Handoff — Redo Open Leg (T2) and the satin slip (31 Oct 2026)

Give a **new T2 session** these files, in this order:

1. **This handoff** — what to do.
2. [Template2.md](Template2.md) — what we already did. That page is the “don’t run this Hybrid again” record, with an old-vs-new table. It is not the recipe.
3. [shells.md](shells.md) — the two T2 hulls, and why `dress-shell-whitebg.glb` is the dress.
4. [stage_one.md](stage_one.md) — how the four plates were made. Do not rerun those commands.
5. [handoff_guess_dress.md](handoff_guess_dress.md) — the 3D path that worked (keep Hunyuan, SDF 4 mm, cylindrical bake). Output of that path: `outputs/guess/guess_dress_hull.glb`.

The 2D plates are good. The 3D is not. Do not regenerate the ideals. Do not run Ghost v1 on T2. Do not call `fitted_shell`. Do not run `run_hybrid_poc.py --sheer-fill`. Use the Guess hull path: keep Hunyuan’s dress, seat it on the mannequin, bake the plates onto that mesh.

Shop names (http://127.0.0.1:8000/shop.html): **Open Leg** and **satin slip**. Open Leg is original template 2. Section 2 (satin / original T4) is a different garment; skip it on a T2-only session. Original template 5 is the hoodie/jeans piece we parked; that is not this work.

Config for the T2 redo: [configs/templates/open_leg.json](../configs/templates/open_leg.json). `outfit_id` is `"2"` so `build_outfit.py` reads `outputs/canonical/2/dress/`. There are no prompts on that file so a `--config` prepass cannot call Kontext. `dress_slip.json` is a different folder (`outputs/canonical/dress_slip/…`) — do not use it.

---

## 1. Open Leg (T2)

### Look at these first

Accepted ideals (Flux Kontext, Ghost v2). These are the product. They already show the bandeau, sheer midriff, short opaque slip, sheer overlay to mid-calf, high left slit, and two hairline straps:

- `outputs/canonical/2/dress/front.png`
- `outputs/canonical/2/dress/back.png` (Kontext turn of the accepted front, not the catalog back)
- `outputs/canonical/2/dress/left.png`
- `outputs/canonical/2/dress/right.png`
- `outputs/canonical/2/dress/placement.json`

Catalog: `frontend/assets/templates/2-front.jpg`, `2-back.jpg`. Method: [docs/Template2.md](Template2.md), [docs/shells.md](shells.md).

Two Hunyuan hulls already exist. Render both before calling Replicate.

| File | Input | What it is |
| --- | --- | --- |
| `outputs/canonical/2/dress-shell-whitebg.glb` | accepted white PNGs | 29 Oct 12:32. Euler −26. Extra tunnels where Hunyuan saw white through the midriff **and** the slit. This is the dress. |
| `outputs/canonical/2/dress-shell.glb` | `hunyuan-front.png` / `hunyuan-back.png` | 29 Oct 12:40. Euler 0. One tube, neck through hem. This is a filled slab. |

Shop Open Leg is `outputs/canonical/2/dress-fitted.glb` — the **filled** shell after SDF fit and a cylindrical 4-view bake. Leave it in place so the failure stays visible. The specs modal currently points the “Hunyuan shell” tile at the filled `dress-shell.glb`. That is the wrong hull.

### What we did

```text
catalog front
  → Flux Kontext  →  front.png
  → turn of front →  back.png, left.png, right.png
front/back
  → sheer_prepass (RGB 24,24,24)  →  hunyuan-front.png / hunyuan-back.png
  → Hunyuan3D-2mv Replicate, front+back only, remove_background
  → dress-shell.glb
  → fit_ghost_garment + SDF 4 mm + cylindrical bake of all four ideals
  → dress-fitted.glb
```

The ideals were never the problem. A later white-background Hunyuan run (`dress-shell-whitebg.glb`) was copied aside, then overwritten by the sheer-fill run. Shop kept the fill.

Command that matches the shop mesh (the failure; full old-vs-new table in [Template2.md](Template2.md)):

```bash
.venv/bin/python scripts/run_hybrid_poc.py \
  --front outputs/canonical/2/dress/front.png \
  --back  outputs/canonical/2/dress/back.png \
  --left  outputs/canonical/2/dress/left.png \
  --right outputs/canonical/2/dress/right.png \
  --placement outputs/canonical/2/dress/placement.json \
  --product-shot --provider replicate --sheer-fill \
  --save-build outputs/canonical/2/dress-shell.glb \
  --no-write-build --unwrap-mode cylindrical \
  --collision sdf \
  --out outputs/canonical/2/dress-fitted.glb
```

### Why it looks awful

The viewer complaints are all shape, not texture.

**Square.** `pipeline/sheer_prepass.py` does not fill “enclosed hollows only”. It floods the garment silhouette, including the sheer overlay, and paints the **whole** silhouette `(24, 24, 24)`. Look at `outputs/canonical/2/dress/hunyuan-front.png`: a solid black midi with two strap nubs. Hunyuan then built that rectangle. A three-layer dress (bandeau / slip / overlay) became one occupancy blob at 384³.

**Slit is a solid piece; mannequin should show through.** The high slit in the photo is white (or near-white sheer over white) **inside** the overlay silhouette. Once the silhouette is filled, there is no gap for Hunyuan to keep. The shop mesh is Euler 0: one tunnel from neck to hem, no leg window. The whitebg hull is the opposite: Hunyuan treated every sheer window as background, so the slit **and** the midriff are holes (Euler −26). Shop did not use that hull.

**Straps.** 1–2 px spaghetti on white. The occupancy grid cannot see them. Sheer-fill dilates only the top 28% of the frame (`STRAP_ZONE`) so the slit does not close, and even then Hunyuan returns nubs or a fused loop. Guess already treats straps as a post-pass (`pipeline/straps.py`, ~6 mm ribbons). T2 never attached those.

What is **not** the failure, unlike Guess: T2 did **not** run `fitted_shell`. `fit_ghost_garment` kept Hunyuan. Baking the good ideals onto the **wrong** hull cannot open a slit that is not in the mesh.

### Do this instead

1. Open the four ideals. If they still match the catalog (black midi, one high left slit, empty, no floral, no body in the skirt), do not run Flux again.
2. Render `dress-shell-whitebg.glb` and `dress-shell.glb`. Orbit front, back, and the slitted side. The filled shell should look like a black column. The whitebg shell should show windows.
3. **Start from `dress-shell-whitebg.glb`.** Do not call Hunyuan until you have rejected that file. Copy it if you need a working name under `outputs/shape/open_leg/dress_raw.glb`.
4. If the whitebg midriff is a hole through the torso (mannequin chest visible where the opaque slip should sit), do **not** re-run today’s `sheer_prepass`. That function fills the overlay and closes the slit. A new vision plate may fill **only** the enclosed midriff window, leave the slit as background, and leave the strap gap open. Only then call Hunyuan3D-2mv, front+back, `--no-cutout` / `--product-shot`, `--build-only`. Never send left/right into Hunyuan.
5. Fit with the Guess path, not `run_fitting.py` and not `run_hybrid_poc.py`:

```bash
poetry run python scripts/build_outfit.py \
  --config configs/templates/open_leg.json \
  --shell dress=outputs/canonical/2/dress-shell-whitebg.glb \
  --out-root outputs/canonical \
  --out outputs/canonical/2/dress_hull.glb
```

`open_leg.json` already has cylindrical unwrap, SDF 4 mm, `lift_m` 0.10, `background_max` 100. `outfit_id` `"2"` makes `load_fit_views` read `outputs/canonical/2/dress/`. Do not `fitted_shell`. Do not pass `--sheer-fill`. Do not run `make_canonical_views.py --config` on this file (`material_type: sheer` would still write a vision plate).

6. After the aligned hull is on the body, look at the straps. If they are nubs or a neck loop, concatenate `pipeline/straps.py` (~6 mm) from bandeau corners over the shoulders. Do this **before** the bake, or bake the hull and paint the ribbons with the same black.
7. Bake **the accepted ideals**, including left and right. Those profiles already exist. They are texture, not shape. Island fill for any remaining grazing texels. Do not TELEA the sheer overlay if a nearest-pixel copy from the same UV island will do.
8. Export `mannequin` + `garment` to a **new** file, e.g. `outputs/canonical/2/dress_hull.glb`. Point shop Open Leg at that file when it orbits clean. Leave `dress-fitted.glb` alone.

Check: overlay hem at mid-calf, short slip above it, high left slit as **missing faces** so the mannequin calf shows through, two straps, 4 mm off the outer skin, not a square tube.

### Do not repeat

- Do not regenerate `front.png` / `back.png` / `left.png` / `right.png`.
- Do not run `sheer_prepass` as it exists and then reconstruct. That is the square dress.
- Do not send left/right to Hunyuan.
- Do not Kontext-edit the catalog back; it left a body in the sheer skirt.
- Do not `fitted_shell`, `run_fitting.py`, or `assemble_dress.py`.
- Do not copy `dress-fitted.glb` over the whitebg shell.
- Do not `trimesh.split()` on the textured hull.

---

## 2. Satin slip (shop “satin slip”, original T4)

### Look at these first

Ghost v1 plates (real satin pixels, person cut in 2D). Opaque enough that a mask is legal:

- `frontend/assets/templates/4-front-ghost.png`
- `frontend/assets/templates/4-back-ghost.png`

Catalog: `frontend/assets/templates/4-front.jpg`, `4-back.jpg`. Method: [docs/Template4.md](Template4.md).

Shop file: `frontend/assets/models/4-hybrid.glb` (Ghost + mannequin, 27 Oct). Leave it.

There is **no** `outputs/canonical/4/` plate folder, **no** left/right, and **no** `outputs/shape/.../dress_raw.glb`. What exists as a dress-only mesh is:

- `frontend/assets/models/4-build.glb` — Hunyuan from the ghost cutouts, ghost pixels UV-baked, **no mannequin**. This is the shell. Render it.

`outputs/4-build-ghost-raw.glb` and `outputs/hybrid_src/4-build.glb` are cited in Template4 and are not on disk now. If `4-build.glb` is a hollow slip with straps, keep it. Do not call Hunyuan until you have looked at it and rejected it.

### What we did

```text
catalog 4-front.jpg / 4-back.jpg
  → rembg + SegFormer B2 + u2net_cloth_seg straps
  → CIE LAB skin-fringe strip, LaMa on bag/hair
  → composite on #FFFFFF  →  4-*-ghost.png
  → Hunyuan3D-2mv Replicate, front+back ghost cutouts
  → 4-build.glb (dress only)
  → fit_ghost_garment (depth-buffer / 4 mm push, back pulled past seat)
  → bake ghost front/back only
  → 4-hybrid.glb
```

No cylindrical 4-view. No side plates. `write_dress_views` was never pointed at this dress.

### Why there are holes

The ghost cutouts only exist as front and back. `bake_photos` weights by facing normal, so under-arm and hip flanks get ~0. Those texels stay empty, get a neutral fill, or TELEA. On dark satin that reads as holes or muddy patches. Guess and leopard fixed the same class of gap with a cylindrical atlas plus (for leopard) left/right profiles. This slip never got either.

A missing `outputs/canonical/4/dress-shell.glb` is a bookkeeping problem, not a missing dress. `4-build.glb` is that hull with the ghost bake already on it. Fit and re-bake a copy; do not start from `4-hybrid.glb` if the hybrid already stretched the dress onto the body with the old z-push.

Straps may still be thin after Ghost v1 recovery. If the hull’s straps are unusable, same procedural ribbons as T2.

### Do this instead

1. Render `4-build.glb`. Confirm it is a dress, not a person. If it is a person, only then reconstruct from the ghost PNGs (`--ghost-prepass` or `--no-cutout` on the ghost files), `--build-only`, save `outputs/shape/satin_slip/dress_raw.glb`.
2. Do not rerun SegFormer / LaMa unless a ghost plate is actually missing.
3. Generate **texture-only** profiles from the accepted ghost front. Do not overwrite `4-front-ghost.png` / `4-back-ghost.png`:

```bash
# write_dress_views(..., views=("left", "right")) from the ghost front
# into outputs/canonical/satin_slip/dress/left.png and right.png
# copy the ghost plates in as front.png / back.png and write placement.json
```

Prompts: opaque black/brown satin mini, lace hem, two spaghetti straps, empty, 90° profile. Not leopard, not the Guess sequin, not the T2 midi.

4. Align + SDF 4 mm + cylindrical bake of front, back, left, right via `build_outfit.py`. Same mannequin, same 4 mm, same “bake onto the Hunyuan dress” rule as Guess.
5. Procedural straps only if the hull loops are broken.
6. Export `outputs/satin_slip/satin_slip_hull.glb`. Point shop satin slip at that file when the flanks are cloth. Leave `4-hybrid.glb`.

Check: lace hem, straps, no white/black punctures on the hips or under the arms, 4 mm off the outer skin, body visible at neck and armholes, not through random holes in the satin.

### Do not repeat

- Do not reconstruct the clothed catalog person and carve in 3D.
- Do not run Ghost v1 on T2 or Guess (sheer / sequin).
- Do not `fitted_shell`.
- Do not planar-wrap the flanks and hope TELEA invents satin.
- Do not send left/right to Hunyuan.

---

## Shared pipeline (what “new” means here)

Guess already proved this sequence on an opaque sequin mini. Both redos reuse it. The only T2-specific fork is **which hull** (whitebg, not sheer-filled). The only satin-specific fork is **making side plates that T4 never had**.

```text
accepted 2D plates
  → Hunyuan3D-2mv front+back  →  dress-only hull (keep it)
  → align_ghost_to_mannequin + load_fit_views
  → push_clearance 4 mm (SDF, not mesh.contains)
  → optional procedural straps
  → bake_photos cylindrical, all plates including sides
  → export_hybrid(mannequin, garment)
```

| Piece | Where |
| --- | --- |
| Align | `align_ghost_to_mannequin` in `pipeline/garment_extractor.py` |
| SDF | `push_clearance` in `pipeline/sdf_collision.py` |
| Bake | `bake_photos` in `pipeline/texture_bake.py`, `unwrap_mode="cylindrical"` |
| Straps | `pipeline/straps.py` |
| Export | `export_hybrid` in `pipeline/export_glb.py` |
| CLI | `scripts/build_outfit.py` |
| Do not use | `fitted_shell`, `scripts/run_fitting.py`, `scripts/assemble_dress.py` |
| T2 filled vision plates | `outputs/canonical/2/dress/hunyuan-*.png` — diagnosis only |
| T2 filled shop mesh | `outputs/canonical/2/dress-fitted.glb` — do not patch |
| Satin shop mesh | `frontend/assets/models/4-hybrid.glb` — do not patch |

Poetry owns `.venv`. `REPLICATE_API_TOKEN` is in `.env` if a new hull is actually required. Modal fallback: `poetry run modal run cloud/modal_app.py --front … --back …`.

Open the new GLBs in the shop or:

```
http://127.0.0.1:8000/?src=/outputs/canonical/2/dress_hull.glb
http://127.0.0.1:8000/?src=/outputs/satin_slip/satin_slip_hull.glb
```
