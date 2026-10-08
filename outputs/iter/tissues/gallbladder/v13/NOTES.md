# gallbladder v13 (recommended; same rest / tets / indexing as v12)

`python -m r2s.tissue.gallbladder v13` (`CFG['v13']` = v12 with `rest_from='v12'`, tet ARAP 15 instead of 30, 130
iterations per frame instead of 100; about 15 min).
`rest_verts`, `faces`, `tets`, `attach_idx` and `attach_to` are byte-identical to v12. See v12/NOTES.md for the
changes vs v11:
- duct masks counted as background;
- neck tip pinned in 3D to the duct's proximal end;
- uniform BCC tet mesh;
- rest refitted through the refined cameras.

## Why
v12's outline was less precise than v11's (boundary F 0.31 vs 0.38 at 130-250, also on the full silhouette). The
uniform BCC mesh is stiffer under the same per-tet ARAP weight. Halving the weight recovers the outline.

## Per band (refined cameras, every 3rd frame, keyframes and in-between frames)
IoU / boundary F (4 px) / depth residual (median, mm). Shared = clip SAM mask. Own = this version's masks (duct =
background) for both models.

| band | v11 shared | v13 shared | v11 own | v12 own | v13 own |
|---|---|---|---|---|---|
| 0-61 | 0.750 / 0.296 / 1.43 | 0.799 / 0.342 / 0.87 | 0.816 / 0.319 | 0.832 / 0.304 | 0.855 / 0.367 |
| 62-129 | 0.794 / 0.327 / 1.06 | 0.825 / 0.358 / 0.70 | 0.826 / 0.363 | 0.824 / 0.312 | 0.848 / 0.376 |
| 130-250 | 0.801 / 0.328 / 1.11 | 0.822 / 0.297 / 0.85 | 0.859 / 0.378 | 0.859 / 0.312 | 0.888 / 0.363 |

Keyframes (r2s.quality): shared IoU 0.821 / BF 0.342 / depth 0.80 mm; own 0.869 / 0.390 / 0.74 mm.

## Neck and duct (refined cameras)
| band | duct mask covered (v11 -> v13) | neck tip - junction depth, median (range) | tip -> junction, 3D / image |
|---|---|---|---|
| 0-61 | 21.8 % -> 0.9 % | -12.6 -> -2.1 (-5.9..+9.3) mm | 13.0 -> 4.9 mm / 23 -> 20 px |
| 62-129 | 22.7 % -> 1.8 % | -9.4 -> -2.8 (-9.5..+1.4) mm | 11.0 -> 5.6 mm / 25 -> 21 px |
| 130-250 | 16.3 % -> 0.2 % | -5.1 -> -1.9 (-4.7..+2.0) mm | 5.9 -> 4.0 mm / 18 -> 19 px |

Junction = ducts v10 `duct_centerline4d[:, 0]` (video depth 77-82 mm). Negative = the tip is nearer the scope.

## Mesh, MuJoCo check
Same tets as v12 (BCC, 1383 nodes / 6186 tets / 1222 surface faces):
- mean ratio min 0.10, p1 0.55, p5 0.68;
- volume / median min 0.09, p1 0.31, p5 0.49;
- min dihedral p1 30.5 deg.
- v11: mean ratio 0.03 / 0.15 / 0.27; volume / median 0.0004 / 0.004 / 0.026.

MuJoCo push test (capsule r 2.5 mm, 5 mm into the surface along the probe shaft, 0.3 s + 0.1 s hold; young 1200,
poisson 0.45, scene4d settings). Inverted tets (max / end):

| | push @60 | push @180 | push @250 | bed -> frame 0 |
|---|---|---|---|---|
| v11 | 7 / 13 | 88 / 78 | 62 / 57 | 1 / 1 |
| v12 / v13 | 0 / 0 | 6 / 6 | 0 / 0 | 3 / 2 |

v12/NOTES.md has the method and the caveat: no sheet or grasper in this check.

## 4D health
- No inverted tet in any sampled frame (min det F 0.24). Volume 0.95-1.04 of rest. Edge stretch p95 <= 1.25,
  max principal stretch 2.6.
- Rigid part up to 13.5 deg / 5.6 mm (v11 11.7 / 3.9), non-rigid rms <= 2.7 mm.
- Max vertex speed 4.9 mm/frame (v11 3.2).
- Behind the sheet (membrane v07): > 0.5 mm in front on <= 0.26 % of overlap pixels per band (max 1.5 % in one
  frame). The deepest points are 5-8 mm at frames 45-81, at the sheet's footprint edge in the pan / probe phase.

## For the integrator
- New indexing vs v10/v11, so the following must be re-mapped:
  - textures;
  - the ducts' glue vertex;
  - the membrane base.
- Attachments: backdrop 214, ducts 10 (within 6 mm of the neck tip), membrane 47.
- The duct's proximal end now lies about 4-5 mm from the neck tip, which is at the junction depth within about 2 mm.
- If the full scene still inverts tets, the remaining candidates are the sheet contact and the driven bed. The
  isolated probe contact no longer inverts them.
