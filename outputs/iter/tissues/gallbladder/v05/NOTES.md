# gallbladder v05 (failed: tets explode)

`python -m r2s.tissue.gallbladder v05` (`CFG['v05']` on top of v04; masks as v01).

## Change vs v04 (why)
- Inversion barrier relu(0.3 - det F)^2 summed over tets with weight 0.5 (v02-v04 averaged it: ~1/6000 per tet).
- Template length scale bounded to >= 1.0 (v04 rest 5.3 cm long, 4.9 cm wide: too round; the fundus continues past
  the lower image border in every frame).

## Result
| metric | shared SAM mask | body+neck mask |
|---|---|---|
| silhouette IoU mean | 0.783 | 0.839 |
| boundary F (4 px) mean | 0.295 | 0.338 |
| depth residual median | 1.56 mm | 1.47 mm |

- Rest: scale (0.995, 1.16, 1.71) -> 6.1 x 4.8 x 4.0 cm, 32.7 ml (more gallbladder-like proportions).
- Temporal: max speed 2.9 mm/frame, centroid range <= 2.4 mm.
- BROKEN: 60 inverted tets, in all 51 sampled frames; max principal stretch 144, edge stretch max 33.

## What I saw
- Diagnosis (tet quality = volume / longest edge^3): ~6 % of TetGen's tets are slivers (< 0.01; regular tet 0.118;
  smallest 1e-4 mm^3). det F of a sliver is hypersensitive to its nodes; the summed barrier drove those nodes and a
  local region blew up (155 tets affected, including well-shaped neighbours). v04's 5 inversions were slivers too.
- TetGen's own quality switch (mindihedral 15, minratio 1.6) brings slivers only to ~3.7 %; peeling sliver caps on
  the boundary (>= 2 boundary faces) and quality-checked interior smoothing brings them to 2.3 %.

Next (v06): that tet pipeline + ARAP / volume / barrier weight per tet min(1, quality / 0.02).
