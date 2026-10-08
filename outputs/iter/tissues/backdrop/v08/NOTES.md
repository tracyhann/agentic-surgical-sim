# backdrop v08: static liver and surroundings of shot A (current version)

`python -m r2s.tissue.backdrop v08 --baseline --holdout` (code `r2s/tissue/backdrop.py`, params `VERSIONS['v08']`,
about 4 min on CPU, 7 min with the baseline and hold-out evaluations; SAM masks cached in `../masks.npz` and `../masks_strands.npz`, copied to `masks.npz` here).

## What it is
One static, textured surface of everything that does not move. It is fused from all 251 views, and the region that
is never visible (the gallbladder bed and the area under the instruments) is filled and flagged.
`verts4d` repeats `rest_verts` in every frame because the surface is static. Every vertex is attached to the world
(`attach_idx` = all, `attach_to` = 'world').

## Approach
1. **Static pixels per frame.** A pixel is static unless it falls in one of these dilated masks: gallbladder incl.
   neck/duct (12 px), the clip's strand (10 px), instruments (12 px), the pink strands beside the cystic duct (own SAM
   2.1 mask, 10 px; prompts `STRAND_PROMPTS`), a 45 px disk at the grasper's jaw end (the tented sheet), or a 12 px
   image border. About 50 % of the pixels remain static (`../v07/work/static_mask.jpg`).
2. **Depth fusion.** A virtual reference camera sits at the mean pose (canvas 842x496 px covering every frame's
   footprint, `ref_*` in model.npz). Every frame's static pixels are back-projected with its depth and camera,
   re-projected into the reference camera and binned in 2 px cells. Per cell: the median over frames, using sharp
   frames (k >= 62) wherever at least 5 of them see the cell, then 2 rounds of outlier rejection
   (> max(2.5 MAD, 2.5 mm)). Spread = 1.4826 MAD across frames.
3. **Surface.** One sparse solve of a confidence-weighted height field (data weight from the number of views and the
   spread; cells seen only in the blurred pan get weight x0.03) with a thin-plate regulariser. The thin plate was
   chosen over membrane and harmonic fill by a hold-out test. The same solve fills the hidden cells. The gallbladder
   bed is constrained to lie >= 3 mm behind the 90th-percentile gallbladder front depth, so the backdrop is never in
   front of the visible gallbladder. The surface is meshed as a regular grid every 6 canvas px.
4. **Texture.** An 842x496 image over the reference canvas (`texture.png`, UV = canvas pixel / canvas size, v up as in
   OBJ). Each texel is the median colour of the frames that see it as static, unoccluded (depth test, 6 mm) and
   not specular, after per-frame colour gains. Sharp frames are preferred and blended with the pan median where few
   sharp frames see the texel. Texels with < 3 observations are push-pull filled and get liver-like high-frequency
   detail (`texture_seen.png` marks the observed texels). `vertex_rgb` = texture at the vertices.
5. **Camera registration (optional extra).** The clip's cameras between bundle-adjusted keyframes are interpolated.
   Rendering the backdrop shows a misalignment of 0.6 px at keyframes but about 3 px between them (up to 14 px in
   the pan). Every non-keyframe camera is registered to the textured backdrop (ECC -> small rotation, 2 rounds), and
   the backdrop is re-fused with these cameras. The result is in `cams_refined.npz`: R per frame (rows = camera axes,
   as `views.R`), pos and f unchanged, correction median 0.30 deg and max 1.9 deg. The contract metrics below use
   the clip's cameras; the refined ones are reported alongside.

## Iterations (numbers: post-pan frames 62-250, median; static mask excluding the pink strands; clip cameras /
refined cameras; spread = median MAD across frames)

| version | change | depth res. mm | photo L1 | local NCC | gallbladder violation | spread mm | area cm2 |
|---|---|---|---|---|---|---|---|
| old pipeline | frame-0 depth + keyframe patches (all frames, v02 mask) | 5.87 | 25.9 | 0.32 | 21 % | - | - |
| v01 | median fusion, harmonic fill, Telea texture | 2.18 | 8.04 | 0.740 | 11.5 % | 3.70 | 124 |
| v02 | wider masks + sheet disk, sharp-first, rejection, gallbladder bound, push-pull | 2.18 | 8.22 | 0.740 | 2.1 % | 3.06 | 120 |
| v03 | confidence-weighted surface fit, domain >= 6 views, texture detail | 2.32 | 8.28 | 0.739 | 0 | 3.07 | 105 |
| v04 | + liver template (test only, not used) | 2.32 | 8.27 | 0.739 | 0 | 3.07 | 105 |
| v05 | thin plate (from the hold-out test), texel >= 3 views | 2.32 | 8.33 | 0.739 | 0 | 3.07 | 112 |
| v06 | frames registered to the backdrop, re-fused | 2.34 / 2.27 | 8.37 / 7.55 | 0.734 / 0.808 | 0 | 3.02 | 113 |
| v07 | pan-only cells nearly free (edge fin) | 2.35 / 2.27 | 8.38 / 7.56 | 0.734 / 0.808 | 0 | 3.02 | 109 |
| **v08** | pink strands excluded, 2 registration rounds | **2.34 / 2.26** | **8.34 / 7.41** | **0.731 / 0.809** | **0** | **3.01** | **109** |

Notes on the table:
- **v01's lower depth residual.** v01 is the plain median of the per-frame depth maps, so it scores lowest against
  those maps, but it intersects the gallbladder and has fins at the canvas edge. The later versions trade about
  0.15 mm of depth residual for valid geometry.
- **What the depth residual measures.** It is measured against per-frame depth maps that disagree with each other by
  3 mm (MAD), mostly as smooth low-frequency warps that change from frame to frame. It is not ground truth.
- **Independent checks.** These do not involve the texture:
  - *Warp consistency.* Frame k is lifted with a depth and re-projected into frame k+17 or k+53, 24 pairs including
    non-keyframes. Fused surface vs each frame's own depth map, local NCC:
    - clip cameras: 0.645 vs 0.615 (L1 11.09 vs 11.00)
    - refined cameras: 0.709 vs 0.697 (L1 9.74 vs 9.89)

    The fused surface is at least as multi-view consistent as the depth maps, and the refined cameras are better
    than the interpolated ones.
  - *Held-out keyframes.* The 26 keyframes were left out of depth and texture fusion, then evaluated. Compared with
    the full model on the same frames: L1 8.55 vs 8.33, local NCC 0.734 vs 0.747, depth 1.94 vs 1.90 mm
    (`quality_holdout.json`). The model generalises to unseen views.

## Contract metrics (all 251 frames, clip cameras, evaluation mask of v02; quality.json)
| metric | value |
|---|---|
| silhouette IoU with the frame minus instruments | median 0.9998 (min 0.94 at frame 0, where the right edge is outside the domain) |
| coverage of static pixels | 1.00 |
| depth residual | median 2.51 mm (keyframes 1.90, between keyframes about 2.4, pan 3.97) |
| photometric (all frames) | L1 9.27, NCC 0.957, local NCC 0.68; after the pan 8.80 / 0.963 / 0.71 |
| photometric (pan frames) | local NCC 0.47 (motion blur) |
| front violation (> 5 mm in front of the video depth) | 10 % |
| gallbladder violation | 0 |
| gap to the visible gallbladder front | median 9.0 mm (p10 4.9) |
| liver part | IoU 0.75, boundary F 0.27 against the own SAM liver mask (which itself flips at the top centre) |
| mesh | 9269 vertices, 18110 faces, 1 component, open sheet, min edge 0.49 mm, 0 degenerate faces, 109 cm2 |
| old pipeline (same evaluation) | depth 5.87 mm, L1 25.9, local NCC 0.32, gallbladder violation 21 % |

## Liver template (v04): does not help
I fitted a BodyParts3D liver (with its gallbladder, same atlas frame) as a similarity transform to the observed liver
patch and the gallbladder front surface. In a hold-out band along the rim of the bed, it predicted depth worse than
the smooth surface: 2.96 mm vs 1.19 mm (and 0.77 mm with the final thin plate). As a weak prior in the fit it gave
2.26 mm. The visible patch is about 5x4 cm of a 20 cm organ, so the pose is unconstrained: the fit scale was 1.3 and
the pose was 60-77 deg away from the anatomical prior. The template is not used; the fitted mesh is kept in
`../v04/liver_template.npz` for reference only. The liver is a label on the backdrop (`label` == 1, 40 % of the
vertices).

## For the integrator
- **Mesh.** `model.npz` holds `rest_verts` (9269, 3) in world m and `faces` (18110, 3), oriented toward the camera.
  `backdrop.obj` + `backdrop.mtl` + `texture.png` are the same mesh with UVs.
- **Texture.** `texture.png` (842x496 RGB) with `uv` (OBJ convention, v up), plus `vertex_rgb` if per-vertex colour
  is preferred.
- **Flags.**
  - `observed` / `filled`: 74 % / 26 %.
  - `bed`: 14 %, filled and under the gallbladder.
  - `bed_weight`: 0 at the rim of the bed, 1 in its interior.
  - `ray_dir`: unit direction from the reference camera through each vertex.
  - `confidence`: data weight of the cell, 0 for filled cells and small for pan-only cells.
  - `spread_mm`, `n_obs`.
- **Gallbladder bed.** It lies behind the gallbladder: centroid (-0.026, 0.051, 0.032) m, bbox x -0.051..0.011,
  y 0.035..0.068, z 0.011..0.058 (`quality.json: gallbladder_bed`). It sits a median of 9 mm (p10 4.9 mm) behind
  the visible gallbladder front surface. If the gallbladder model is thicker than that, push the bed back with
  `X + d * bed_weight[:, None] * ray_dir`; this keeps the rim attached.
- **Collisions.** Use `hfield_elev` (99x145, normalised), `hfield_size`, `hfield_pos` and `hfield_quat` as a MuJoCo
  hfield. I checked it with mj_ray against the mesh: <= 0.1 mm (0.6 mm at one steep point). `hfield_valid` marks
  grid cells that the surface actually covers. The mesh itself is visual-only; MuJoCo would convexify it.
- **Cameras.** `cams_refined.npz` registers the frames between keyframes; it improves every tissue's image metrics.
  To adopt it, set `V.R = R` for all tracks, e.g. via `r2s.tissue.backdrop.with_cameras(V, R)`. The world frame is
  unchanged because the keyframes are kept fixed.

## Known problems
- **Pan frames 0-61 fit worse.** Local NCC 0.47 and depth 4 mm. The video is motion-blurred and the depth maps of
  those frames are inconsistent.
- **Low-confidence canvas edges.** Cells seen only in the pan are thin-plate extrapolations. A thin wing at the top
  right and a fold at the bottom centre are visible from oblique views (`views3d.jpg`). Use `confidence` to find them.
- **Front violations of 7-10 %.** They come from per-frame depth-map warps, not from the surface (the residual maps
  are smooth blobs that change from frame to frame).
- **Gallbladder bed shape is unobserved.** It is a smooth thin-plate continuation of the rim, pushed behind the
  gallbladder; no data constrains a deeper fossa.
- **Unstable liver label.** The liver/other boundary at the top centre follows the SAM liver mask, which includes or
  excludes that region depending on the frame (boundary F 0.27).
- **Conservative sheet zone.** It is a fixed 45 px disk around the grasper's jaw end. Pixels of the translucent sheet
  outside the disk are rejected only by the robust median.
