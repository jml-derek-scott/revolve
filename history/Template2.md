# Template 2 — what we already did (do not repeat the 3D)

This page is the **old T2 pipeline**. It is here so a new session does not run the same Hybrid again and call that a redo.

The 2D plates are keepers. The 3D is not. Shop **Open Leg** still loads the failed Hybrid on purpose.

| Start the new 3D from | Leave closed unless you are diagnosing |
| --- | --- |
| [handoff_t2_and_satin.md](handoff_t2_and_satin.md) | this page |
| [shells.md](shells.md) (the whitebg hull) | `run_hybrid_poc.py --sheer-fill` |
| [handoff_guess_dress.md](handoff_guess_dress.md) + `scripts/build_outfit.py` | `fitted_shell`, `run_fitting.py` |
| [stage_one.md](stage_one.md) (plates only) | Flux / Kontext on these four PNGs |

Solid black midi: opaque bandeau, sheer midriff, short opaque slip, sheer overlay to mid-calf, high slit on the left leg. Catalog shots used to look floral in reconstructions; they are not.

---

## Old vs new (read this first)

| Step | What we did (this page, 29 Oct) | What to do now |
| --- | --- | --- |
| Ideal photos | Flux Kontext empty midi. Back is a **turn of the accepted front**. Left/right from that front. | **Keep** `outputs/canonical/2/dress/{front,back,left,right}.png` and `placement.json`. Do not regenerate. |
| Placement | SegFormer on the **catalog** full photos (`967×1450`), not on `front.png`. | Keep that JSON. Do not rewrite it to the PNG size. |
| Hunyuan input | `--sheer-fill` wrote `hunyuan-front.png` / `hunyuan-back.png`. The whole silhouette is RGB `(24, 24, 24)`. | **Do not run** `pipeline/sheer_prepass.py` as it exists. It filled the overlay and closed the slit. Start from the white-background hull. |
| Hull we reconstructed first | White Sim2 PNGs → `dress-shell-whitebg.glb` (12:32). Euler −26. Midriff **and** slit are holes. | **This is the dress.** Fit this file. Do not call Hunyuan until you have rendered it and rejected it. |
| Hull Hybrid used | Sheer-fill PNGs → `dress-shell.glb` (12:40). Euler 0. One tube, neck through hem. | Leave it on disk. Do not fit it again. That is the square column in the shop. |
| Fitter | `run_hybrid_poc.py --product-shot` → `fit_ghost_garment` (also drops faces with `|x| > 0.35 m`). | `scripts/build_outfit.py --config configs/templates/open_leg.json`. `align_ghost_to_mannequin` + SDF 4 mm. No `HIP_X` cut. No `fitted_shell`. |
| Lift | First fit `--lift 0` (bandeau on the ribcage). Then refit the **filled** shell with `DRESS_LIFT_M` 0.10 m. | `lift_m` 0.10 is already in `open_leg.json`. Apply it to the **whitebg** hull, not to `dress-shell.glb`. |
| Bake | Cylindrical 4-view of the four ideals onto the filled hull. | Same bake, same ideals, onto the whitebg hull. Texture cannot open a slit that is not in the mesh. |
| Straps | Occupancy nubs. Never ran `pipeline/straps.py`. | After the hull is on the body: procedural ribbons only if the loops are unusable. |
| Output | `dress-fitted.glb` → Hybrid / shop Open Leg. | **New** file, e.g. `outputs/canonical/2/dress_hull.glb`. Do not overwrite `dress-fitted.glb`. |

`dress_slip.json` is a sibling sheer stub (`outfit_id: dress_slip`). It does not point at `outputs/canonical/2/dress/`. The config for this redo is `configs/templates/open_leg.json` (`outfit_id: "2"` so plates resolve to that folder). It has no prompts on purpose: a `--config` prepass would call Kontext.

---

## What’s on disk

| File | Role |
| --- | --- |
| `outputs/canonical/2/dress/front.png` | Accepted empty front. Keep. |
| `…/back.png` | Kontext turn of that front (`T2_BACK_FROM_FRONT`). Keep. Not an edit of the catalog back. |
| `…/left.png`, `…/right.png` | Texture only. Never Hunyuan inputs. Keep. |
| `…/placement.json` | Person + dress box in catalog pixels. Keep. |
| `…/hunyuan-front.png`, `hunyuan-back.png` | 29 Oct 12:38. Solid dark midi. Diagnosis only. |
| `outputs/canonical/2/dress-shell-whitebg.glb` | 12:32. Hunyuan of the **white** plates. The hull to start from. |
| `outputs/canonical/2/dress-fitted-whitebg.glb` | Same hull after the first fit/bake. Compare, not the product. |
| `outputs/canonical/2/dress-shell.glb` | 12:40. Hunyuan of the gray fill. The slab. |
| `outputs/canonical/2/dress-fitted-lift0.glb` | Filled hull, `--lift 0`. |
| `outputs/canonical/2/dress-fitted.glb` | Filled hull, lift 0.10 m, cylindrical bake. **Shop Hybrid. Leave it.** |

Catalog: `frontend/assets/templates/2-{front,back}.jpg` and `-full.jpg`.

---

## What’s in the viewer (old Hybrid)

| Button | File | What it is |
| --- | --- | --- |
| Texture wrap | — | Disabled. `HAS_WRAP` does not include 2. |
| Full build | — | Disabled. `HAS_BUILD` does not include 2. |
| Hybrid | `outputs/canonical/2/dress-fitted.glb` | The **filled** shell, SDF-fitted, cylindrical 4-view. Not in `frontend/assets/models/`. |
| Sim2 thumbs | the four PNGs above | Accepted empty-dress photos. These are still right. |

`HAS_HYBRID` does not include 2. Hybrid still works because `modelUrl()` loads `SIM2[2].mesh` when there is no demo hybrid. Shop Open Leg (`frontend/items.json`) also points at `dress-fitted.glb`. The specs “Hunyuan shell” tile still points at the **filled** `dress-shell.glb` — that is the wrong hull for the redo.

There is no Side fill button (it would be the same file).

---

## Stage 1 we keep (do not rerun)

**Code:** `T2_PROMPTS`, `T2_BACK_FROM_FRONT`, `T2_EMPTY` in `pipeline/canonical.py`. Entry: `scripts/make_canonical_views.py --template 2`.

```bash
poetry run python scripts/make_canonical_views.py --template 2
```

That command **overwrites** the accepted PNGs. Do not run it unless a file is missing.

How each view was made (full prompts and Kontext settings: [stage_one.md](stage_one.md)):

| View | Source | Prompt |
| --- | --- | --- |
| Front | `2-front-full.jpg` | `T2_PROMPTS["front"]` |
| Back | accepted `front.png` | `T2_BACK_FROM_FRONT`, `safety_tolerance=5` |
| Left / right | accepted `front.png` | `T2_PROMPTS["left"]` / `["right"]` |
| `placement.json` | both **catalog** photos | `measure_placement` |

`--strip-body` runs `T2_EMPTY` on existing front/back, then rebuilds left/right. Only if a leftover figure is actually in the PNG.

Trap that is still true: editing the **catalog back** left a body in the sheer skirt. Do not “fix” the back from `2-back-full.jpg`.

---

## Stage 2 we already ran (do not rerun)

This is the Hybrid in the shop. Command that produced `dress-shell.glb` (12:40) and, after a 0.10 m refit, `dress-fitted.glb` (12:46):

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

The reconstruct itself used `--lift 0`. The bandeau sat under the bust, so the filled shell was copied to `dress-fitted-lift0.glb` and refit with `--provider skip --build dress-shell.glb --lift 0.10`. Today’s default lift is already 0.10 m (`DRESS_LIFT_M`). Re-running the block above would call Hunyuan **again** and overwrite `dress-shell.glb`.

The whitebg hull, earlier the same day:

```bash
poetry run python scripts/run_hybrid_poc.py \
  --front outputs/canonical/2/dress/front.png \
  --back  outputs/canonical/2/dress/back.png \
  --provider replicate \
  --no-cutout \
  --build-only \
  --save-build outputs/canonical/2/dress-shell.glb
```

Those bytes were copied to `dress-shell-whitebg.glb` before sheer-fill overwrote `dress-shell.glb`.

### Sheer fill (why the Hybrid is a square)

**Code:** `pipeline/sheer_prepass.py`. Flag `--sheer-fill` on `run_hybrid_poc.py`.

The accepted PNGs were not overwritten. The copies Hunyuan saw:

- Flood the garment silhouette from the corners.
- Paint **the whole silhouette** `(24, 24, 24)`, not only the enclosed midriff window. Look at `hunyuan-front.png`: a solid black midi with strap nubs.
- Neckline between straps stays open (touches the top border).
- Straps in the top 28% of the frame are dilated 11 px. Dilating the whole dress closed the slit.

Hunyuan3D-2mv (`tencent/hunyuan3d-2mv`, Replicate, `remove_background: true`, 80k faces, octree 384) then built that rectangle. Euler went from −26 (windows) to 0 (one tunnel). Fit and bake succeeded on a bad shell: SDF energy looked fine, the chest is a white slab, the slit is a solid panel.

A later “only fill enclosed hollows” pass is allowed **only** if the whitebg hull is rejected because the midriff is a hole through the torso **and** a new function can fill that window without filling the slit. Today’s `sheer_prepass` cannot.

### Fit and bake of the Hybrid

`fit_ghost_garment` + SDF 4 mm + cylindrical `bake_photos` of all four ideals. Product-shot mode pushes hem through bust (`dress_band`). `--no-write-build` so a `2-build.glb` is not invented.

That fitter also drops faces whose center is more than `HIP_X` (0.35 m) off the midline. `build_outfit.py` does not. Use `build_outfit.py`.

---

## Why it looks awful (shape, not texture)

- **Square.** The vision plate is a filled occupancy blob. Three layers (bandeau / slip / overlay) became one 384³ column.
- **Slit is a solid piece.** The photo slit is white inside the overlay. Once the silhouette is filled, Hunyuan has no gap to keep. Mannequin calf should show through **missing faces**.
- **Straps.** 1–2 px on white. The grid cannot see them. Fill dilates nubs; it does not make wearable loops.
- **Not** `fitted_shell`. T2 kept Hunyuan. Painting good ideals onto the **wrong** hull cannot open a slit that is not in the mesh.

---

## What was deleted earlier (still gone)

| Old idea | Why it went |
| --- | --- |
| TRELLIS.2 full build (`2-build.glb`) | Seamless texture, invented back. Floral print and a back slit were guesses. |
| Wrap onto a Hunyuan person (`2-wrap.glb`) | Person mesh + catalog photos is not the empty-dress path. |
| Hunyuan+Paint then `--no-extract` overlay (T3 finish) | Wrong once the goal was an empty dress on the white mannequin. |

---

## Dead ends (do not redo)

- Regenerating `front.png` / `back.png` / `left.png` / `right.png`.
- Kontext-edit of the **catalog back**.
- `run_hybrid_poc.py --sheer-fill` (or `write_prepass` on `open_leg.json` / `dress_slip.json`). That is the square dress.
- Sending left/right into Hunyuan.
- Fitting `dress-shell.glb` again, or copying `dress-fitted.glb` over the whitebg hull.
- `fitted_shell`, `scripts/run_fitting.py`, `scripts/assemble_dress.py`.
- Ghost v1 / SegFormer cut on this midi.
- Treating a clean SDF energy as “the dress is right”.
- `trimesh.split()` on a textured hull.
- Copying `dress-fitted.glb` into `frontend/assets/models/2-hybrid.glb` unless Hybrid should become a demo asset (`HAS_HYBRID` would then need `2`).

---

## What is still true (steal this, not the Hybrid command)

1. Accept the four Sim2 images before any 3D. They are already accepted.
2. Back-from-front when a catalog-back edit cannot drop the body through sheer.
3. Hunyuan gets **front and back only**. Sides are texture.
4. Cylindrical unwrap + four views for flanks. Planar wrap is not the T2 failure; the hull is.
5. SDF clearance on `female_base_mesh.glb` (not `mesh.contains()`).
6. Two-node export (`mannequin` + `garment`).
7. `--no-write-build` if you ever touch `run_hybrid_poc.py` again.

The redo is: keep (1)–(7), **drop sheer-fill**, start from `dress-shell-whitebg.glb`, fit with `build_outfit.py`. Commands live in the handoff, not here.
