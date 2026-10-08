# membrane v06: sheet made consistent with gallbladder v10 and the interaction grasper (superseded by v07: refined cameras, remeshed, gallbladder v11)

`python -m r2s.tissue.membrane v06` (code `r2s/tissue/membrane.py`, `CONFIGS['v06']`; about 65 s on CPU). The mask,
template and video fit are the same as v05 (see ../v05/NOTES.md for the pipeline, SAM prompts and iterations v01-v05).
Reads `gallbladder/v10/model.npz` and `interaction/v05/model.npz` (read-only).

## Why
Integration rounds r01-r04 showed that the lower half of v05 lies behind the v10 gallbladder front surface in many
frames. v10 sits about 3 mm in front of the video depth under the sheet. As a result the simulation hid the sheet (sim
IoU 0.36 vs recon 0.81). The v05 apex was also up to 15 mm from the grasper TCP.

## What changed vs v05
1. **Gallbladder depth.** v10 is rasterised in every frame: per-pixel nearest depth, taking the smaller of the
   perspective-interpolated depth and the flat per-face depth used by `quality.render`.
2. **Base row on the v10 surface (vertices 400-424).**
   - Each base point keeps its image pixel and takes the v10 depth there, minus 0.25 mm. This happens before and after
     the temporal smoothing; a final sigma 1.5 frames follows.
   - 98.2 % of base rays hit v10. The rest move to the nearest pixel that v10 covers (at most 40 px); only if none is
     that close do they go to the closest surface point.
3. **Apex.** It lies on the camera ray of the image apex (where the sheet leaves the jaws), at the point closest to
   the grasper's jaw segment (interaction v05 `grasper_left_tcp` -> `grasper_left_tip`, 3 mm). Depth comes from the
   instrument, the image position from the video. Smoothed sigma 1.5 frames.
4. **No sheet vertex behind v10.** Where the v10 depth (3x3 minimum) at a vertex's pixel is in front of it, the vertex
   moves forward along its camera ray to v10 depth - 1 mm. The silhouette does not change.
   - The push is spread over the grid (Laplacian) and over time (sigma 2), but never below what each vertex needs.
   - The jaw row stays fixed (it is attached to the grasper) and the base row is already on the surface.
5. **Rest.** The frame minimising max(p99 stretch, 1/p1 compression) over the clip: frame 175.

Tried inside this version (scratch numbers):
- Apex exactly at `grasp_point`: IoU 0.768. The TCP is 3 mm proximal of the jaw tips and projects 14 px from where
  the sheet leaves the jaws.
- Apex on the jaw segment, nearest to the image apex in projection, with misses snapped to the closest 3D surface
  point: IoU 0.751. The 3D snap shifted outline points by up to 38 px.

## Results (keyframes unless stated; before = v05)
| metric | v05 | v06 |
|---|---|---|
| IoU vs our mask, mean (min) | 0.808 (0.674) | 0.783 (0.635) |
| boundary F 4 px / 8 px | 0.512 / 0.780 | 0.460 / 0.709 |
| IoU every frame (pan 0-62 / 63-250) | 0.803 (0.744 / 0.822) | 0.779 (0.718 / 0.799) |
| sheet pixels hidden by v10 (render z-test), mean (max) | 26.4 % (53.9 %) | 0.54 % (1.9 %) |
| IoU of the visible (not hidden) sheet vs our mask | 0.613 | 0.784 |
| non-base vertices behind v10, all frames: mean (max) | 14.9 % (34 %) | 0 % (0 %) |
| all vertices behind v10: strictly / by > 1 mm | 16.6 % / 8.4 % | 1.45 % / 0.12 % |
| base vertex -> v10 surface, per-frame median (max) | - | 0.43 mm (0.9 mm; worst 1.5 mm) |
| apex -> grasp point (TCP), all frames: median (max) | 4.0 (14.9) mm | 3.3 (4.4) mm |
| apex -> jaw segment TCP-tip: median (max) | 3.7 (14.9) mm | 1.05 (3.6) mm |
| apex -> image apex | 1.7 px | 1.2 px median (max 5.8) |
| stretch max, mean (max) / p95 / p05 | 2.63 (4.4) / 1.49 / 0.68 | 3.45 (6.8) / 2.01 / 0.65 |
| area | 1.4-5.7 cm2 | 2.0-7.4 cm2 |
| mean vertex accel | 0.15 mm/frame² | 0.23 mm/frame² |
| depth residual vs video, lower / upper half (median) | 1.11 / 1.25 mm | 1.69 / 1.81 mm |

Notes on these numbers:
- The remaining 1.45 % "behind" are base vertices that lie on the surface, within the 3x3 sampling tolerance.
- The apex sits near the jaw tip (u = 0.91 of the TCP->tip segment), so its distance to the TCP is about 3 mm by
  construction.
- The interaction grasp point lies 2.4 mm farther than my old image-depth apex at the median (-8 to +14 mm), and its
  projection is 14 px from where the sheet leaves the jaws.
- The front push: per-frame maximum 2.9 mm on average (max 7.0 mm), and 88 % of the vertices move. The sheet now drapes
  over v10, which raises the area, the stretch and the video-depth residual. That is expected, because v10 lies in
  front of the video depth under the sheet.
- Accel: v10 itself moves at 0.14 mm/frame² and the interaction grasp point at 0.44.

Visual check: `work/sheet_with_gb_before.jpg` (v05 + v10) shows the sheet cut by the gallbladder, with ragged
outlines. `work/sheet_with_gb_after.jpg` shows the whole sheet in front of the gallbladder in every keyframe, with
clean outlines.

## Rest shape: a less-lifted frame or slack?
Edge length / rest length over the whole clip (p99, p1) for candidate rest frames:

| rest frame | p99 | p1 |
|---|---|---|
| 0 (start) | 4.18 | 0.20 |
| 70 (least lifted) | 3.87 | 0.36 |
| 120 | 1.90 | 0.23 |
| 175 (chosen) | 2.94 | 0.42 |

- No single frame works well. Vertices are ray / arc-length samples, not material points. The visible tent recruits
  sheet as it is lifted (area x3.7), and the base slides along the gallbladder.
- A less-lifted rest (frame 70) means up to 3.9x stretch, and therefore a large pull on the gallbladder unless the
  membrane is very soft.
- My suggestion: keep rest = frame 175 (it is balanced) and either use a soft in-plane membrane, or add slack by
  scaling the rest edge lengths by 1.3-1.5. With 1.5x slack, p99 stretch is 1.96; frames with compression then buckle,
  which is harmless with no bending stiffness.
- If the simulation starts from verts4d[0], use rest = verts4d[0] to avoid the initial jump. In that case expect up to
  4x stretch later.

## Known problems
- Slivers: smallest angle 0.48 deg at rest and 2.2 deg at the 1st percentile over time. They sit in the jaw row (5 mm
  over 24 segments) and the right-side columns, where the fan's generators converge. The index layout was kept so
  that attach indices stay 0-24 / 400-424.
- IoU vs our mask is 0.78 (was 0.81). Snapping missed base points to v10 shifts the outline where v10 does not cover
  the sheet's foot, and the pan frames 0-62 are worst (v10 itself fits poorly there).
- The sheet depth now depends on v10. If the gallbladder track changes, rerun `python -m r2s.tissue.membrane v06`
  (the source paths are in `CONFIGS['v06']`).
- Mask flicker and the polar-fan limitations are the same as in v05.

## model.npz
- Contract arrays: `rest_verts` (425, 3) = frame 175, `faces` (768, 3), `verts4d` (251, 425, 3) with no NaN.
- `attach_idx` / `attach_to`: vertices 0-24 -> 'grasper', 400-424 -> 'gallbladder'.
- Extra arrays:
  - as in v05: `grid_shape`, `apex` (251, 3; the jaw-row centre = apex used), `base` (251, 25, 3), `apex2d`,
    `depth_offset`.
  - `apex_image3d`: the v05-style image-depth apex.
  - `grasp_point`: copy of interaction v05.
  - `jaw_u`: nearest TCP->tip fraction.
  - `base_hit`: (251, 25), True where the base ray hit v10.
  - `front_push`: (251, 425), forward move in m.
- `apex_raw` / `base_raw` are the per-frame observations before smoothing (base already cast on v10).
