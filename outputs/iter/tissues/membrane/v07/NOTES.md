# membrane v07: refined cameras, remeshed sheet without slivers, consistent with gallbladder v11 (superseded by v08: gallbladder v13, held row at the TCP)

`python -m r2s.tissue.membrane v07` (code `r2s/tissue/membrane.py`, `CONFIGS['v07']`; about 4.5 min on CPU, most of it
the per-band evaluation). Pipeline as in v06 (../v06/NOTES.md); v01-v05 in ../v05/NOTES.md.

Inputs, read-only:
- Cameras: `backdrop/v08/cams_refined.npz`. Only R changes. Applied as a shallow copy of `views.load('chole_a', 'sift')`
  with `V.R` replaced, which is equivalent to `views.load(..., cams=...)`.
- Gallbladder: `gallbladder/v11` (it falls back to v10 if v11 is missing).
- Grasper: `interaction/v05` (TCP and jaw tip).

## Changes vs v06
1. **Refined cameras.** Every 3D step (apex, base lifting, depth refinement, gallbladder casting, temporal smoothing in
   world) uses the refined rotations.
2. **Pan frames 0-62.** A scratch test (unsmoothed vs smoothed per-frame fit, both camera sets) showed two things.
   - The refined cameras alone add only about 0.005-0.008 IoU.
   - World-space temporal smoothing is what costs IoU in the pan (0.885 per frame vs 0.745 smoothed).
   - So the base uses lighter smoothing there (median 5, sigma 1.5 instead of median 9, sigma 3). The weight ramps back
     to normal between frames 55 and 70, and the blend happens before the base is cast onto the gallbladder.
3. **Remeshed sheet** (`design_mesh` + `improve_mesh`):
   - The (s, t) domain is triangulated on the ruled surface of a design frame.
   - Boundary samples are every 1.3 mm of 3D arc length: jaw line 5 vertices (was 25 over 5 mm), sides, and the base.
     Every one of the 25 observed base knots is a vertex.
   - Inside: Poisson-disk samples 1.3 mm apart on the surface, Delaunay in rescaled (s, t), then tangential smoothing
     in 3D and Delaunay edge flips in 3D. Every triangle stays positively oriented in (s, t).
   - After the depth refinement and the front push, the triangulation is polished once more on the final rest
     surface. Every frame is then resampled at the new (s, t) by barycentric interpolation (silhouettes unchanged) and
     cleared from the gallbladder again.
   - Design frame = rest frame = 162. It is the post-pan frame (>= 62) that minimises max(p99 stretch, 1/p1 compression).
4. **Gallbladder v11.** Base cast onto v11 and every vertex pushed in front of v11, as in v06 against v10.

Tried and rejected inside this version:
- Exact ray-cast snapping of the base after the fit: the base ended within 0.25 mm of the surface and 0 % of vertices
  were behind. But rest min angle fell to 5.7 deg and max stretch to 12 (35 max), because vertices toggled between ray
  hits and misses.
- Exact casting inside the base fit: rest min angle 8.2 deg, 5 % of base rays missed, IoU -0.006.
- Design frame chosen over all frames: it picked frame 15 (pan), and rest min angle was 1.6 deg.

## Results (v06 numbers are re-measured with v11 and the refined cameras where marked)
| metric | v06 | v07 |
|---|---|---|
| IoU / BF 4 px, frames 0-61 (every frame) | 0.723 / 0.367 (clip cams 0.720 / 0.333) | **0.756 / 0.420** |
| IoU / BF 4 px, frames 62-129 | 0.747 / 0.479 (clip 0.742 / 0.460) | 0.743 / 0.467 |
| IoU / BF 4 px, frames 130-250 | 0.823 / 0.509 (clip 0.823 / 0.507) | 0.821 / 0.515 |
| IoU / BF 4 px, all frames | 0.778 / 0.466 | **0.784 / 0.478** |
| IoU / BF 4 px, keyframes | 0.783 / 0.460 (v10, clip cams) | 0.790 / 0.480 |
| smallest triangle angle at rest | 0.48 deg | **19.6 deg** (0 faces < 15 deg) |
| faces < 15 deg in verts4d, per frame (p1 angle) | many (2.2 deg) | 66 of 395 (4.1 deg) |
| mesh | 425 v / 768 f grid, min edge 0.14 mm | 237 v / 395 f, min edge 0.60 mm, 1 component |
| non-base vertices behind gallbladder (exact ray cast) | 0 % (v10) | 0 % (v11), all frames |
| base vertices behind v11 by > 1 mm (of all vertices) | - | 0.15 % mean |
| sheet pixels hidden by the gallbladder (render z-test) | 0.37 % (v11) | 0.34 % (max 1.1 %) |
| visible-sheet IoU | 0.784 (v11) | 0.793 |
| base -> gallbladder surface, per-frame median | 0.43 mm (v10) | 0.39 mm (v11; per-frame max <= 2.2 mm) |
| apex -> image apex / -> jaw segment / -> TCP (median) | 1.2 px / 1.05 / 3.3 mm | 1.0 px / 1.1 / 3.4 mm |
| stretch max mean (max) / p95 / p05 | 3.45 (6.8) / 2.0 / 0.65 | 4.5 (10.1) / 2.2 / 0.53 |
| area | 2.0-7.4 cm2 | 1.8-6.7 cm2 |
| mean vertex accel by band | 0.28 / 0.21 / 0.21 mm/frame² | 0.35 / 0.25 / 0.24 |

What the numbers say:
- The gain is in the pan, from the lighter smoothing. Mid-clip (62-129) is 0.004 IoU lower.
- The refined cameras alone change v06 by +0.002 IoU over the clip.
- The price is more motion in the base. That comes from the lighter pan smoothing, and from the base being a larger
  share of a coarser mesh (52 of 237 vertices).

Visual check:
- `work/sheet_with_gb_after.jpg`: the sheet is in front of v11 in all keyframes.
- `views3d.jpg`: the remeshed rest sheet is clean and isotropic.

## For the integrator (changes vs v06)
- **New topology.** 237 vertices and 395 faces; `rest_verts` = frame 162. Use `attach_idx` / `attach_to`:
  - 5 jaw vertices -> 'grasper' (1.25 mm apart over 5 mm),
  - 52 base vertices -> 'gallbladder' (on v11).
  - The old indices 0-24 / 400-424 are no longer valid.
- **Bending.** All rest triangles are >= 19.6 deg, so the shell bending term can include every face.
- **Base drive.** Drive the base along `verts4d` as before. The base now sits on v11 (median 0.39 mm).
- **Sheet coordinates.** `st` (237, 2) holds each vertex's (s, t): s = 0 at the jaws, s = 1 at the base. Other
  extras are as in v06.
- **Stretch is not physical.** The 4D vertices are ray / arc-length samples, not material points, so recruitment and
  base sliding show up as stretch (p95 about 2.2). Keep the soft edge springs, or add 1.3-1.5x slack.

## Known problems
- Frames 62-129 are 0.004 IoU below v06, and the base moves more (accel above).
- verts4d triangles distort over time: 66 of 395 faces are under 15 deg per frame on average, because the
  (s, t) -> 3D map changes as the base lengthens (52-88 mm).
- In about 1 % of base casts the ray misses v11 (just outside its outline). Those vertices go to the nearest pixel
  v11 covers, so the outline shifts by a few pixels there.
- Mask flicker and the polar-fan limits are as in v05.
