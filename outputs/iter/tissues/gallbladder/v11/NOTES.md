# gallbladder v11 (integrator request: refined cameras, behind the sheet)

`python -m r2s.tissue.gallbladder v11` (`CFG['v11']`, about 15 min on CPU including the band evaluation).

## What changed vs v10
- Cameras: `views.load('chole_a', 'sift', cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz')`. These are
  rotations registered to the backdrop between keyframes (0 deg at the keyframes; up to 1.9 deg in the pan). Pos and
  f are unchanged. Masks and depth maps are unchanged.
- **Rest, faces, tets and attach_idx/attach_to are byte-identical to v10** (`rest_from='v10'`, attachments copied).
  The bed anchor uses v10's 'backdrop' set. Only `verts4d` (and `fit_parts_mm`) is refitted. All v10 4D settings are
  kept.
- New term: the gallbladder stays behind the tented sheet. The membrane v06 `verts4d` is rasterised per frame
  (perspective-correct depth). Every surface sample that projects into the sheet must be at least 0.3 mm behind it
  (Charbonnier hinge, weight 3, mean over the footprint samples).

## Per frame band (every 3rd frame, 84 frames: keyframes and in-between frames; work/bands.npz)
IoU / boundary F (4 px) / depth residual (median, mm) against the shared SAM mask. Own body+neck mask in brackets.

| band | v10, clip cams | v11, clip cams | v10, refined cams | v11, refined cams |
|---|---|---|---|---|
| 0-61 (pan) | 0.751 / 0.298 / 1.48 (0.833) | 0.743 / 0.288 / 1.47 (0.825) | 0.744 / 0.268 / 1.52 (0.820) | 0.750 / 0.296 / 1.43 (0.831) |
| 62-129 | 0.791 / 0.314 / 1.07 (0.840) | 0.792 / 0.314 / 1.10 (0.841) | 0.787 / 0.320 / 1.16 (0.833) | 0.794 / 0.327 / 1.06 (0.841) |
| 130-250 | 0.799 / 0.317 / 1.13 (0.864) | 0.800 / 0.323 / 1.09 (0.864) | 0.797 / 0.303 / 1.15 (0.862) | 0.801 / 0.328 / 1.11 (0.865) |
| IoU at keyframes / in between | 0.791 / 0.784 | 0.791 / 0.783 | 0.791 / 0.780 | 0.791 / 0.786 |

- **Gallbladder in front of the sheet.** Measured as the share of overlap pixels where the gallbladder is > 0.5 mm in
  front: v10 0 %, v11 <= 1.5 % in any frame (band means 0.06 / 0.18 / 0.01 %). Worst point 2.2 mm.
- **Keyframe metrics through the refined cameras.** Shared mask: IoU 0.790, BF 0.333, depth 1.13 mm. Own mask:
  0.855 / 0.383 / 1.01 mm.
- **Health.** No inverted tet. Min det F 0.24 (v10: 0.07). Max principal stretch 2.9. Volume 0.91-1.06 of rest.
  Max vertex speed 3.2 mm/frame.
- **v11 - v10 vertex distance.** Median 0.3-0.5 mm, p95 1.1-1.6 mm. 59 vertices of the neck end differ by
  5-12 mm in frames 181-217. There v10's neck hooked down over the duct region, and v11 follows the mask: BF
  +0.05..0.17, IoU +0.01-0.02 in those frames (work/diag_neck.jpg).
- **Bed ('backdrop') vertices.** Median difference to v10 0.23-0.28 mm. The excursion from rest is the same as v10
  (median 1.7, p95 5.7, max 14.7 mm). The max is on the top/back of the neck near the probe, frames 70-75.

## Reading
- **"v10 IoU 0.2-0.4 in the pan" was the static rest shape, not v10's 4D.** The 4D is about 0.75 there.
- **The refined cameras help only a little.** Pan frames gain BF +0.03, depth -0.09 mm and IoU +0.006 over v10 seen
  through the same cameras. In-between frames gain +0.006 IoU.
- **What limits the pan is the depth.** In frames 0-42, 22-32 % of the mesh pixels sit > 6 mm behind the measured
  depth and are cut by the visibility test (6-11 % after frame 48). The p10 of measured - mesh is -5..-9 mm (later
  -1..-2 mm). The blurred frames' depth maps scatter. Following them would mean moving the bed 5-8 mm frame to frame,
  and the integrator drives the bed along verts4d, so I kept the bed anchor.

## For the integrator
- Drop-in replacement for v10: same vertex indexing, tets, attachments and textures.
- Membrane v06's base row was snapped onto v10. Its distance to v11 is median 0.58 mm, p95 1.18 mm, max 2.25 mm
  (to v10: 0.54 / 0.99 / 1.53 mm). Re-snapping the base row to v11 would make them exactly consistent.
