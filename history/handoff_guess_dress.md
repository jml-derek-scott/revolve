# Handoff — Guess sequin dress, start the 3D over (30 Oct 2026)

The 2D plates are good. Hunyuan already turned them into a dress. The last 3D session threw that dress away. Start over from the hull. Do not regenerate the plates. Do not call Hunyuan again unless you have looked at `dress_raw.glb` and rejected it.

## Look at these first

Accepted plates (Strazza / Flux Kontext on the catalog photos, not SegFormer cutouts):

- `outputs/canonical/guess_dress/dress/front.png`
- `outputs/canonical/guess_dress/dress/back.png`
- `outputs/canonical/guess_dress/dress/placement.json`

Catalog sources: `test_images/guess_front.png`, `test_images/guess_back.png`. Method notes: [strazza/image_generation.md](../strazza/image_generation.md). Outfit config: [configs/templates/guess_dress.json](../configs/templates/guess_dress.json).

The reconstruction that matches those plates:

- `outputs/shape/guess_dress/dress_raw.glb` (Hunyuan3D-2mv, Replicate, 30 Oct, 1.5 MB)

Render it before changing anything. Front: straps, straight bandeau, sequin mini. Side: a strap loop over the empty neck, and ribbon tails down the back. That mesh is the garment.

## What the last session shipped, and why it is bad

`outputs/guess/guess_dress.glb` is a torn black decal on the mannequin. Do not patch it. Do not treat it as a starting point.

`scripts/run_fitting.py` wrote two files. Only the raw hull is usable.

- `dress_raw.glb` — the dress. Keep.
- `dress_shell.glb` — a band of `mannequins/female_base_mesh.glb`. Discard.

`fitted_shell` in `pipeline/garment_deformer.py` does not keep Hunyuan. It cuts the mannequin’s outer skin between two Y values and nudges those vertices toward the hull. The viewer then showed the body’s hips with a shredded photo on them. A body has no skirt, no bandeau edge, and no bow, so the dress existed only as paint.

The bake made the holes. The shell is as wide as the mannequin. About half of the front faces missed the dress in the photo and sampled white. `warp_x_to_silhouette` only covers the waist–hip band and did not fix it.

`outputs/shape/guess_dress/dress_bow.glb` is a height field cut from the back plate (`pipeline/bow.py`, `scripts/assemble_dress.py`). It is not the bow on the hull. Do not concatenate it onto a mannequin shell and call that the garment.

`front-hole.png` and `back-hole.png` in the plate folder are leftovers. They are not the plates.

## Do this instead

1. Load `dress_raw.glb`. Align it with `align_ghost_to_mannequin` and `load_fit_views` on the plate directory. The hull is garment-only, so the ghost aligner is the right one. The mannequin is `mannequins/female_base_mesh.glb`.
2. Push the aligned dress outside the body with the SDF (`push_clearance`, 4 mm). The mannequin is a double skin with scrambled winding. Do not use `mesh.contains()` or vertex normals for the collision test.
3. Project `front.png` and `back.png` onto **that** mesh (`bake_photos`, `unwrap_mode="cylindrical"`). The hull’s silhouette matches the plates, so the photo should land on cloth instead of white.
4. Look at the aligned hull before adding straps or a bow. The raw mesh already has a strap loop and hanging tails. Add procedural straps (`pipeline/straps.py`, about 6 mm) only if those loops are unusable. Do not build another bow until the hull’s own tails have been seen in mannequin space.
5. Export a new file, `outputs/guess/guess_dress_hull.glb`, via `export_hybrid` (`mannequin` + `garment`). Leave `outputs/guess/guess_dress.glb` in place so the failure stays visible.

Open it in the viewer and orbit front, back, and one side before calling it done:

`http://127.0.0.1:8000/?src=/outputs/guess/guess_dress_hull.glb`

Check: sequin body, satin bandeau, two straps, hem at the upper thigh, bow or tails as mesh on the back, 4 mm off the outer skin, no white panels where the dress should be.

## Do not repeat

- Do not run `scripts/run_fitting.py` or `scripts/assemble_dress.py` as the build. The first one replaces the dress with the mannequin. The second one bakes that shell and glues on the fake bow.
- Do not call `fitted_shell` for this outfit.
- Do not regenerate the plates. Do not run Ghost v1. Do not sheer-fill with RGB `(24, 24, 24)`. The cloth is opaque sequin.
- Do not use hoodie, jeans, or template-1 leopard prompts.
- Do not call `trimesh.split()` on the textured hull. Weld a copy with `merge_vertices(merge_tex=True, merge_norm=True)` before any component check. `pipeline/mesh_utils.py` already does this.
- Kontext closed the upper back of the plate with sequin. The catalog back is open above the bow. The plates were accepted anyway. Do not “fix” them unless asked.

## Code map

| Piece | Where |
| --- | --- |
| Outfit | `configs/templates/guess_dress.json` |
| Plates | `pipeline/canonical.py`, `scripts/make_canonical_views.py --config` |
| Align a garment-only hull | `align_ghost_to_mannequin` in `pipeline/garment_extractor.py` |
| SDF push | `push_clearance` in `pipeline/sdf_collision.py` |
| Bake | `bake_photos` in `pipeline/texture_bake.py` |
| Export | `export_hybrid` in `pipeline/export_glb.py` |
| The failed shell | `fitted_shell` in `pipeline/garment_deformer.py` |
| The failed assembly | `scripts/assemble_dress.py`, `pipeline/bow.py` |

Poetry owns `.venv`. `poetry run pytest tests/ -q` does not call Replicate. `REPLICATE_API_TOKEN` is already in `.env` if a new hull is actually required. Modal fallback: `poetry run modal run cloud/modal_app.py --front … --back …`.
