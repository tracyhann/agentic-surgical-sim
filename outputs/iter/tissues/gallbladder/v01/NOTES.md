# gallbladder v01

`python -m r2s.tissue.gallbladder v01` (code: `r2s/tissue/gallbladder.py`, config `CFG['v01']` = BASE).

## What it does
1. Masks (`masks.npz`: `body`, `unknown`, `sheet`; (251, 360, 640) bool).
   - `sheet`: SAM 2.1 object (views.segment_object) of the tented peritoneal sheet, prompted at frames 0, 50, 100, 150,
     200, 250 relative to the grasper jaw tip (tx, ty) = grasper-mask pixel farthest along image direction (0.45, 0.89):
     pos (tx+6, ty+22), (tx+14, ty+45); neg (tx+10, min(ty+190, 340)) [body], (tx-90, ty+40) [liver],
     (tx+60, ty-40) [above]. Exact prompts: `sheet_prompts(V)` in the code. The SAM mask covers the whole translucent
     cone from the jaw down over the upper-left body (the sheet lies on the body there).
   - `unknown` = instruments (9x9 dilation) + `strand` (9x9 dilation) + tent apex (sheet pixels within 60 px of the jaw
     tip). `body` = clip SAM `gallbladder` mask minus unknown, small specks dropped.
2. Rest: BodyParts3D gallbladder (3000 faces, principal frame, fundus at -x, neck at +x) -> rotation, translation,
   log-scale per axis fitted jointly to keyframes 60..250 (step 10, 20 views, each through its own camera R, f, pos):
   outside-silhouette distance (px -> mm at organ depth, x2), coverage of body pixels by the projected surface (x2),
   visible body points at measured depth -> nearest front-facing surface sample (mm). Points hidden more than 8 mm
   behind the measured depth may leave the silhouette (hidden parts come from the template). 16 starts (long axis
   along the observed long axis, 8 rolls x 2 directions), best 3 refined, then a 4x4x4 lattice (smoothness + size).
3. Tets: organ.remesh (1000 faces, 20 cells, sample snapping) + TetGen -> 2074 nodes, 7953 tets, 2924 surface faces.
4. 4D: per frame, all node positions (mm) by Adam (300 iterations at the start frame 120, then 80 per frame, warm
   start with constant-velocity prediction), frames 120 -> 250 then 120 -> 0. Energy: data (same three terms, one view)
   + 20 x volume-weighted tet ARAP + 20 x (det F - 1)^2 + 0.1 x |X_t - X_t-1|^2. Gaussian sigma 1 frame over time.

## Result
| metric | shared SAM mask | body+neck mask |
|---|---|---|
| silhouette IoU mean (min) | 0.811 (0.748) | 0.860 (0.793) |
| boundary F (4 px) mean | 0.304 | 0.353 |
| depth residual median | 1.27 mm | 1.19 mm |

Baseline (old pipeline, simulated): IoU 0.54, boundary F 0.12, depth 4.2 mm.
- Rest: scale (0.98, 1.26, 1.41) of the template axes; 6.0 x 4.2 x 2.9 cm, 24.6 ml (template 6.1 x 3.4 x 2.1 cm,
  15.4 ml). Lattice max offset 9.5 mm; rest vs scaled template mean 1.9 mm, max 6.7 mm.
- Rest held static over all keyframes (multi-view consistency): IoU 0.60 mean, 0.15-0.33 in frames 0-80 (!),
  0.70-0.83 in frames 100-250; depth 3.4 mm median.
- Temporal: max vertex speed 3.8 mm/frame, p99 1.2 mm/frame. Volume 0.95-1.02 of rest.

## What I saw / problems
- Contact sheet: outline follows the video well. From frame 180 the neck end turns into a thin hook curling down
  along the right edge of the mask (the mask there includes the dark region right of the body); not anatomical.
- Health: min surface edge 0.033 mm (organ.remesh's sample snapping collapses edges) -> slivers; volume-weighted ARAP
  gives those slivers no weight: tets invert in 50/51 sampled frames, max principal stretch 27x.
- Rigid decomposition of verts4d vs rest: rotation 40-50 deg in frames 0-80 and 20-30 deg at 190-250. The free
  deformation explains the silhouettes by spinning the organ; the background points (liver / right organs) agree
  between frames within ~1-2 mm (5-7 mm at frames 0 and 70-80), so the cameras do not need such a spin: implausible for
  an organ attached to its liver bed.

Next (v02): tets without tiny edges, per-tet (uniform) ARAP + inversion barrier, soft anchor of the back (bed) side
to the rest, rest fit over all keyframes 0..250 (early pan frames at half weight).
