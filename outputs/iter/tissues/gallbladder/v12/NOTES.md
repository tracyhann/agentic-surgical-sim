# gallbladder v12 (integrator feedback r12/r13: neck over the duct, inverted tets in the simulation)

`python -m r2s.tissue.gallbladder v12` (`CFG['v12']`, about 12 min on CPU including evaluation and the MuJoCo
check). It uses the refined cameras (backdrop v08 `cams_refined.npz`) and stays behind membrane v07.
**New rest shape, tets and indexing (1383 nodes): this is NOT a byte-identical replacement for v11.**

## Changes
1. **Duct pixels are background.** The ducts v10 masks (`duct | strands`) are removed from both `body` and
   `unknown`. Before, they were unknown, so the template neck could grow over the duct without penalty.
2. **Neck tip pinned in 3D.** The target is the duct's proximal end, ducts v10 `duct_centerline4d[:, 0]`
   (observed junction, video depth).
   - Rest fit: the template's neck cap (vertices within 2 mm of the narrow end, centroid), weight 1 (mm, Charbonnier),
     on all 26 keyframes.
   - 4D: surface nodes within 3 mm of the rest neck tip, weight 0.5.
   - This replaces v03-v11's 2D landmark (the junction pixel next to the `strand` mask).
3. **Rest refitted** through the refined cameras with these masks (same scheme as v06: pose + anisotropic scale +
   lattice, 26 keyframes).
   - 5.2 x 4.1 x 3.3 cm, 28.6 ml (v10/v11: 6.0 x 4.9 x 3.8 cm, 32.2 ml); template scale (0.99, 1.21, 1.70).
   - The neck is shorter because it no longer covers the duct.
   - Neck cap to the duct's proximal end: 2.3 mm median over the keyframes.
4. **Uniform tet mesh** (`bcc_tets` / `bcc_for_count`) instead of TetGen:
   - body-centred cubic lattice, h = 3.8 mm, tuned for about 1330 nodes; tets with centroid inside kept;
     non-manifold spikes removed; boundary nodes snapped onto the surface;
   - quality-checked smoothing (moves that flatten a tet are undone);
   - Adam on the worst tets' mean ratio, with boundary nodes re-projected;
   - flat caps (two boundary faces and mean ratio < 0.35, or volume < 0.3 x median) peeled.
   - Volume 1.6 % below the fitted surface. Every tet is about the same size.
5. **4D** as v11: refined cameras, behind the sheet (membrane v07), same weights, plus the 3D neck pin.

## Tet quality (rest)
Mean ratio = 6 sqrt(2) V / l_rms^3, 1 for a regular tet.

| | nodes / tets / surface faces | mean ratio min / p1 / p5 / median | volume / median: min / p1 / p5 | min dihedral: min / p1 / p5 (deg) |
|---|---|---|---|---|
| v11 (TetGen) | 1332 / 5302 / 1720 | 0.030 / 0.147 / 0.266 / 0.663 | 0.0004 / 0.0037 / 0.026 | 1.6 / 8.9 / 16.6 |
| v12 (BCC) | 1383 / 6186 / 1222 | 0.102 / 0.552 / 0.680 / 0.920 | 0.087 / 0.312 / 0.486 | 6.5 / 30.5 / 38.6 |

Minimum surface edge: 1.6 mm (v11: 0.18 mm).

## MuJoCo check (`mujoco_check`)
Settings as in scene4d: dim-3 flex, young 1200, poisson 0.45, damping 0.002, Euler, ts 6.25e-5 s, vertex mass
6e-5 kg, bed vertices on 20 N/m springs.
- Push: a capsule (r 2.5 mm) along the interaction v06 probe shaft at frames 60 / 180 / 250. It aims at the surface
  vertex nearest the probe tip, moves from 1 mm outside to 5 mm into the surface in 0.3 s, then holds 0.1 s.
- Bed: bed springs moved to their frame-0 4D positions over 0.3 s.

Inverted tets (max during the run / after the hold):

| | push @60 | push @180 | push @250 | bed -> frame 0 |
|---|---|---|---|---|
| v11 | 7 / 13 | 88 / 78 | 62 / 57 | 1 / 1 |
| v12 | 0 / 0 | 6 / 6 | 0 / 0 | 3 / 2 |

All runs stay finite, about 5 s each. This check has no sheet contact or grasper, so it does not reproduce the
380-500 inversions of the full scene. It isolates the probe contact, which is where the small TetGen tets fold.

## Neck and duct (refined cameras, every 3rd frame)
| band | duct mask covered by the mesh (v11 -> v12) | neck tip - junction depth, median (range) | tip -> junction, 3D / image |
|---|---|---|---|
| 0-61 | 21.8 % -> 0.3 % | v11 -12.6 (-15.1..-10.2) mm -> v12 -1.2 (-8.6..+10.2) mm | 13.0 -> 5.8 mm / 23 -> 26 px |
| 62-129 | 22.7 % -> 1.8 % | -9.4 (-15.4..-7.6) -> -2.4 (-9.0..+2.8) mm | 11.0 -> 5.7 mm / 25 -> 23 px |
| 130-250 | 16.3 % -> 0.2 % | -5.1 (-7.4..-2.6) -> -1.2 (-4.5..+2.6) mm | 5.9 -> 3.1 mm / 18 -> 13 px |

Negative depth difference = the tip is nearer the scope than the junction. The junction's video depth is about
78-82 mm.

## Silhouettes per band (refined cameras, every 3rd frame)
IoU / boundary F (4 px) / depth residual (median, mm). "Own" = this version's masks (duct = background) for both
models.

| band | v11 shared | v12 shared | v11 own | v12 own |
|---|---|---|---|---|
| 0-61 | 0.750 / 0.296 / 1.43 | 0.786 / 0.283 / 0.98 | 0.816 / 0.319 | 0.832 / 0.304 |
| 62-129 | 0.794 / 0.327 / 1.06 | 0.805 / 0.315 / 0.89 | 0.826 / 0.363 | 0.824 / 0.312 |
| 130-250 | 0.801 / 0.328 / 1.11 | 0.794 / 0.261 / 0.98 | 0.859 / 0.378 | 0.859 / 0.312 |

- Better IoU in the pan, and depth is better everywhere.
- **But boundary F drops 0.05-0.07 from frame 62 on.** At frames 130-250 the full silhouette (before the visibility
  test) scores BF 0.27 vs v11 0.34, so the outline itself is less precise, mostly along the left (liver-side) edge
  (work/diag_v11_v12.jpg). Likely cause: the uniform mesh is stiffer under the same tet ARAP weight. v13 tests a
  softer ARAP on this mesh.
- **4D health.** No inverted tet (min det F 0.20), volume 0.95-1.04 of rest, edge stretch p95 <= 1.20.
- **Rigid motion.** Up to 16.9 deg / 6.7 mm (v11 11.7 / 3.9): the 3D neck pin pulls the neck along with the duct's end.
- **Sheet.** Gallbladder > 0.5 mm in front of the sheet on <= 0.4 % of overlap pixels (band means). The worst point,
  3.3 mm at frame 50, is in the pan.

## For the integrator
- New `rest_verts` / `faces` / `tets` / `attach_idx`: textures, the ducts' glue vertex (v244 in v10/v11) and the
  membrane base must be re-mapped. Attachments: backdrop 214, ducts 10 (within 6 mm of the neck tip), membrane 47.
- The neck tip now sits at the duct's proximal end (3 mm median in 130-250). Gluing the duct start to the nearest
  'ducts' vertex should need almost no offset.
