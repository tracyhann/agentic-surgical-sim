# gallbladder v10 (FINAL of this track)

`python -m r2s.tissue.gallbladder v10` (code `r2s/tissue/gallbladder.py`, `CFG['v10']`): v06's rest shape and tets
(`rest_from='v06'`) with v07's 4D data settings.

## What it is
- Masks (`masks.npz`: body, unknown, sheet): clip SAM `gallbladder` mask minus instruments (9x9 dilation), `strand`
  (cystic duct / pink strands, 9x9 dilation) and the apex of the tented peritoneal sheet (SAM 2.1 `sheet` object within
  90 px of the grasper jaw tip); those pixels are unknown. Sheet prompts: v01/NOTES.md (`sheet_prompts`).
- Rest (from v06): BodyParts3D gallbladder, pose + anisotropic scale + 4x4x4 lattice fitted jointly to 26 keyframes
  (0..250, own cameras; frames < 90 weight 0.3): silhouette outside / coverage, depth points, neck tip -> duct
  junction landmark, scale hinge [1.0-1.7 length, 0.85-1.7 width / thickness]. Scale (0.995, 1.16, 1.71) of the
  template axes; 6.0 x 4.9 x 3.8 cm (PCA extents), 32.2 ml; template 6.1 x 3.4 x 2.1 cm, 15.4 ml. The thickness sits
  at the bound (data wants it thicker). Rest vs pose+scale template: mean 1.8 mm, max 6.9 mm.
- Tets (from v06): 1332 nodes, 5302 tets, 1720 outward surface faces (indices into the same nodes), watertight,
  min edge 0.18 mm, no degenerate tets; 2.3 % slivers (volume / longest edge^3 < 0.01, min 0.0015).
- 4D: per frame, node positions fitted to that frame's silhouette (7000 surface samples, 1800 body pixels) and depth
  (1500 points up to 1 px from the outline) + tet ARAP 30 (per tet, sliver-weighted) + volume 20 + inversion
  barrier + soft bed anchor 0.02/mm^2 + neck landmark 0.3 + temporal 0.3; frames 120 -> 250 and 120 -> 0;
  Gaussian sigma 1 frame.

## Result
| metric | shared SAM mask | body+neck mask (90 px apex) |
|---|---|---|
| silhouette IoU mean (min) | 0.789 (0.699) | 0.852 (0.746) |
| boundary F (4 px) mean | 0.320 | 0.366 |
| depth residual median | 1.13 mm | 1.04 mm |

Baseline (old pipeline, simulated, SIFT): IoU 0.54, boundary F 0.12, depth 4.2 mm.
- 4D health: no inverted tet in 51 sampled frames (min det F 0.07); principal stretch p99 1.44, max 2.55 in
  well-shaped tets (7.1 only in a sliver); edge stretch p95 <= 1.15; volume 0.91-1.06 of rest.
- Motion: rigid part vs rest <= 11.7 deg / 3.9 mm, non-rigid rms <= 3.6 mm (max 12 mm); vertex speed max 3.1,
  p99 1.16 mm/frame; centroid range 3.9 / 5.1 / 4.6 mm (x / y / z).
- Rest held static over all keyframes: IoU 0.58 (0.2-0.4 in frames 0-80, 0.65-0.8 later), depth 4.1 mm.

## Iterations (all in ../vNN/NOTES.md)
| v | change | IoU shared / own | BF shared / own | depth mm shared | tets inverted | verdict |
|---|---|---|---|---|---|---|
| 01 | first build | 0.811 / 0.860 | 0.304 / 0.353 | 1.27 | 50/51 frames, stretch 27x | organ spins 51 deg, neck hooks |
| 02 | clean tets, uniform ARAP, bed anchor, all keyframes, scale bounds | 0.760 / 0.793 | 0.254 / 0.282 | 1.54 | 0 | neck points out of view (wrong) |
| 03 | neck -> duct junction landmark, softer ARAP / anchor | 0.779 / 0.833 | 0.301 / 0.340 | 1.50 | 0 | anatomy right |
| 04 | depth points to the outline, smoother lattice / time | 0.792 / 0.838 | 0.309 / 0.345 | 1.35 | 15 (slivers) | too round (5.3 cm long) |
| 05 | summed barrier, length >= template | 0.783 / 0.839 | 0.295 / 0.338 | 1.56 | 60, stretch 144x | slivers exploded |
| 06 | TetGen q15 + sliver peeling, sliver-weighted energies | 0.791 / 0.847 | 0.314 / 0.355 | 1.33 | 0 | healthy |
| 07 | 90 px tent apex, denser samples | 0.792 / 0.859* | 0.335 / 0.393* | 1.25 | 378 (0.2 mm tet cluster) | interior broken |
| 08 | surface edge collapse before TetGen | 0.749 / 0.819* | 0.294 / 0.332* | 1.49 | 2 slivers | clean but coarse (1788 tets) |
| 09 | finer v08 tets | - | - | - | - | tetrahedralisation failed |
| 10 | v06 mesh + v07 4D data | 0.789 / 0.852* | 0.320 / 0.366* | 1.13 | 0 | **final** |

`*` own mask with the 90 px tent apex unknown (v01-v06: 60 px), so own-mask numbers of v07+ are not comparable with
v01-v06; the shared-mask columns are.

## For the integrator
- Frames: world = MuJoCo (metres, z up), as `views.load('chole_a', 'sift')`. `rest_verts` is the rest shape already
  placed where it is observed (no extra transform). `verts4d[k]` (251, 1332, 3) is the same vertex set in video
  frame k; `faces` (outward, signed volume +32.2 ml) index those vertices (surface subset; interior nodes are only in
  `tets`). `fit_parts_mm` (251, 3): per-frame outside / coverage / depth residuals (mm, Charbonnier).
- `attach_idx` / `attach_to` (suggestions): 'backdrop' 316 vertices = back side (vertex normal . mean scope view
  > 0.35), the hidden side on the liver bed, fix to the backdrop (the 4D fit pulls them softly to rest); 'ducts' 50
  vertices within 6 mm of the neck tip (where the cystic duct continues; the tip projects within ~14 px of the
  body/duct junction); 'membrane' 68 front vertices that project into the SAM tent sheet in frame 150 (where the sheet
  leaves the body; upper-left of the body, below the grasper). `long_axis_to_neck` = unit vector fundus -> neck.
- `masks.npz['sheet']` is a SAM 2.1 mask of the whole tented sheet (useful to the membrane track).
- The video does not constrain the region under the tent apex (unknown): the mesh top there is the template's guess.

## Known problems
- Early frames (0-90, blurred pan): the static rest explains them poorly (IoU 0.2-0.4); the measured gallbladder depth
  moves -9..+5 mm relative to it while the background agrees to 1-7 % depth scale, so the 4D shows real lift plus
  depth noise; the motion there (rotation up to 12 deg) is less trustworthy than after frame 90.
- Thickness at the scale bound (3.8 cm): the data push for a rounder organ than the template.
- Tet mesh: TetGen's quality refinement grades to sliver / tiny tets on this surface; v06's mesh is clean but a finer
  one could not be made robustly (v07-v09). 123 slivers carry reduced ARAP weight; in MuJoCo they may need care.
- Boundary F stays ~0.32-0.37 at 4 px: SAM outlines are jagged and the surface is 1720 faces.
