# gallbladder v02

`python -m r2s.tissue.gallbladder v02` (`CFG['v02']` on top of BASE; masks and prompts as v01, see v01/NOTES.md).

## Change vs v01 (why)
- v01 tets had 0.03 mm edges (organ.remesh snapping) and volume-weighted ARAP gave those slivers no stiffness
  (inversions in 50/51 sampled frames, 27x stretch) -> marching-cubes remesh projected onto the fitted surface
  (`remesh_projected`, no edge below 35 % of the median), 650 input faces; per-tet (uniform) ARAP weight 50,
  inversion barrier (det F < 0.3) weight 100.
- v01 spun the organ 40-50 deg (rigid part of verts4d vs rest) -> soft anchor of the bed side (surface vertices with
  normal . mean view > 0.35) to the rest positions, 0.05 / mm^2.
- Rest fit on all 26 keyframes 0..250 (frames < 90 weight 0.3 after a first try at 0.5) for a wider baseline.
- First v02 run inflated the template's thinnest axis 3.45x (24 -> 70 mm thick) -> stiff hinge on the scale per
  template axis outside [0.85, 1.7] (anatomical range: 5.2-10.3 cm long, 2.9-5.8 cm wide, 1.8-3.5 cm thick) and a
  stronger lattice size prior (2.0).

## Result
| metric | shared SAM mask | body+neck mask |
|---|---|---|
| silhouette IoU mean | 0.760 | 0.793 (min 0.646) |
| boundary F (4 px) mean | 0.254 | 0.282 |
| depth residual median | 1.54 mm | 1.53 mm |

- Health: 1495 nodes, 5530 tets, 2204 surface faces, no degenerate / inverted tets; over the 4D: min det F 0.24,
  0 inverted tets in all 51 sampled frames, max principal stretch 1.12, edge stretch p95 <= 1.14.
- Rigid part vs rest: <= 8 deg, <= 3.4 mm (v01: 51 deg). Volume 0.92-1.08. Max vertex speed 2.9 mm/frame.
- Rest: scale (1.20, 1.22, 1.70) -> 7.2 x 4.3 x 4.1 cm, 39.6 ml; static multi-view IoU 0.59.

## What I saw
- views3d / rest sheet: the pose search put the template's neck out to the LEFT (out of view, a thin tail) and the
  fundus at the duct: the silhouettes alone do not fix the head-tail direction. The 4D outline is a round blob with no
  neck at the duct junction. Wrong anatomy although the tets are healthy.
- Signed depth (measured - static rest) per keyframe varies -8.6 .. +4.8 mm while the background (liver, right organs)
  agrees across frames within 1-7 % depth scale: the gallbladder really moves towards / away from the scope (lifted by
  the grasper), so the bed anchor must stay soft.

Next (v03): neck landmark (template neck tip -> body/duct junction pixel = body pixel closest to the strand mask) in
the rest fit and weakly in 4D; ARAP 30, anchor 0.02.
