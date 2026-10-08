# gallbladder v06

`python -m r2s.tissue.gallbladder v06` (`CFG['v06']` on top of v05; masks as v01).

## Change vs v05 (why)
v05's tets exploded around TetGen slivers (see v05/NOTES.md). Now:
- own TetGen call with mindihedral 15, minratio 1.6 on a 500-face projected remesh (16 cells across the thinnest
  extent); boundary sliver caps (quality < 0.01 with >= 2 boundary faces) peeled; interior nodes moved by Laplacian
  steps only where the worst incident tet quality improves (`clean_tets`). Slivers 6 % -> 2.3 %, min quality
  1e-4 -> 1.5e-3 (quality = volume / longest edge^3, regular tet 0.118).
- tet ARAP / volume / summed barrier weighted per tet by min(1, quality / 0.02): remaining slivers cannot dominate.

## Result
| metric | shared SAM mask | body+neck mask |
|---|---|---|
| silhouette IoU mean (min) | 0.791 (0.697) | 0.847 (0.749) |
| boundary F (4 px) mean | 0.314 | 0.355 |
| depth residual median | 1.33 mm | 1.30 mm |

- Rest: scale (0.995, 1.16, 1.71) of the template axes; 6.0 x 4.9 x 3.8 cm (PCA extents), 32.2 ml (template 6.1 x
  3.4 x 2.1 cm, 15.4 ml); lattice max offset 4.7 mm; rest vs pose+scale template mean 1.8 mm, max 6.9 mm. Neck tip
  within 14 px of the duct junction (median of 26 keyframes). Static multi-view IoU 0.58, depth 4.1 mm.
- Tets: 1332 nodes, 5302 tets, 1720 surface faces, min edge 0.18 mm, watertight, no degenerate tets.
- 4D: no inverted tet in any of 51 sampled frames (min det F 0.27); max principal stretch 2.4 (local), edge stretch
  p95 <= 1.15; volume 0.92-1.06 of rest. Rigid part <= 11.8 deg / 4.2 mm, non-rigid rms <= 3.6 mm.
- Temporal: max vertex speed 3.1 mm/frame, p99 0.9, max acceleration 1.4 mm/frame^2.

## What I saw
- Contact sheet: body + neck follow the video; the neck points to the duct junction in all frames. From frame 140
  the top of the mesh still climbs into the tent up to the grasper jaw: the mask keeps the tent below the 60 px apex.
- views3d: pear shape, neck to the right; bed side (blue) away from the scope.
- Weakest keyframes: 10-20 (blurred pan) and 160-170 (BF 0.18-0.2).

Next (v07): tent apex unknown out to 90 px from the jaw; denser surface samples (7000) and coverage pixels (1800)
for a tighter outline; Gaussian sigma 1.0.
