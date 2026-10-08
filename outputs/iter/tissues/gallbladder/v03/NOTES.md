# gallbladder v03

`python -m r2s.tissue.gallbladder v03` (`CFG['v03']` on top of v02; masks as v01).

## Change vs v02 (why)
- v02's pose search put the template's neck out of view on the left. New anatomical landmark: the template's neck
  tip (template vertices within 5 mm of the narrow +x end, centroid) must project onto the body/duct junction pixel
  (`neck_junction`: the body-mask pixel closest to the `strand` mask, found in 250/251 frames; checked visually in
  v02/work/junction.jpg: it sits where the dark neck meets the pink duct/strands). Weight 1 (mm-equivalent) in the rest
  fit (all keyframes), 0.3 in every 4D frame (on the surface nodes within 5 mm of the rest neck tip).
- Softer: tet ARAP 30 (v02 50), bed anchor 0.02 / mm^2 (v02 0.05).

## Result
| metric | shared SAM mask | body+neck mask |
|---|---|---|
| silhouette IoU mean | 0.779 | 0.833 (min 0.760) |
| boundary F (4 px) mean | 0.301 | 0.340 |
| depth residual median | 1.50 mm | 1.44 mm |

- Rest: neck tip within 13 px (median) of the junction over 26 keyframes. Scale (0.92, 1.17, 1.70) -> 5.8 x 3.9 x
  4.5 cm (PCA extents), 30.8 ml; template 6.1 x 3.4 x 2.1 cm, 15.4 ml. Thickness sits at the upper bound.
- 1598 nodes, 5833 tets, no degenerate; 4D: no inverted tets, min det F 0.11, max principal stretch 1.96 (local),
  edge stretch p95 <= 1.15, volume 0.92-1.08. Rigid part <= 6.6 deg / 3.6 mm, non-rigid max 4-12 mm.
- Max vertex speed 4.2 mm/frame (frames 99-100, 160-162), p99 0.95.

## What I saw
- Contact sheet: the neck now points to the duct (right), body fills the lower left, outline follows the mask.
- views3d: neck correct; the lattice cut a notch between neck and body (lumpy).
- diag (work/diag.jpg): the visible-only test of r2s.quality drops mesh pixels > 6 mm behind the measured depth;
  this removes strips along the left outline (liver side, where the depth map blends towards the nearer liver) and at
  the tent; the 7x7-eroded depth points never constrain the outline.
- The mask still contains part of the tented sheet between the grasper and the probe (beyond the 60 px apex).

Next (v04): depth points up to the outline (3x3 erosion), 1500 per frame, depth weight 1.5; lattice smoothness 10;
temporal weight 0.3, Gaussian sigma 1.5, 100 iterations per frame.
