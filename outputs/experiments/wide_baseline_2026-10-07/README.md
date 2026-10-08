# Wide-baseline multi-view: VGGT and SIFT, 2026-10-07

Follow-up to `../multiview_2026-10-07`. The keyframe bundle adjustment (BA) there links keyframes only through optical
flow chained over up to 3 keyframes, so views far apart are not tied together. Question: does a wide-baseline
source fix that? Tried VGGT-1B (feed-forward multi-view network, Wang et al. CVPR 2025) as planned, plus SIFT
matches as a classical reference.

## Setup

`r2s/multiview.py` modes (flow tracks are always used; keyframes every 10 frames):

| mode | initial values | wide-baseline correspondences |
|---|---|---|
| `ba` | rotation + zoom track, single-view depth calibration | none (= step 1) |
| `sift` | same | SIFT between all keyframe pairs >= 2 apart, specular highlights masked, ratio 0.75, MAGSAC 1 px |
| `vggt_tracks` | same | VGGT tracks of static corners from every 4th keyframe (one pass each, query frame first), MAGSAC 3 px |
| `vggt` | VGGT poses, focal length, depth (scaled by shaft widths; Depth Anything affine fitted to it) | VGGT tracks |

Runs: `python -m r2s.multiview <clip>+<mode> <mode>`, full pipeline `python -m r2s.batch <clip>+<mode>`; outputs in
`outputs/variants/<clip>+<mode>/` (perception symlinked from the base clip). Evaluation: `python -m r2s.views_eval
<clip> <dir>` (this folder: `alignment.json`, `alignment.txt`, `warp_frame0_to_*.jpg`).

Two measures, both on static anatomy (instruments, gallbladder / pedicle and strands masked):
- **held-out geometry**: ORB matches between all keyframe pairs (a detector none of the modes uses), MAGSAC-verified;
  median reprojection error of each solution (its poses + its depth);
- **image alignment**: NCC of keyframe i warped into keyframe j, on pairs all solutions can compare.

## Geometry

Held-out ORB reprojection, median px (keyframe gap 1-2 / 3-9 / >= 10 / pairs with frame 0):

| | shot A | shot B |
|---|---|---|
| no camera motion | 9.9 / 20.2 / 31.2 / 107 | 14.3 / 41.7 / 136 / 4.6 |
| single view (rotation + zoom) | 3.4 / 4.5 / 5.8 / 6.4 | 5.2 / 9.3 / 9.6 / 3.2 |
| VGGT alone | 4.4 / 7.2 / 10.2 / 38.2 | 5.1 / 7.6 / 15.4 / 7.3 |
| BA, flow (step 1) | 1.9 / 3.6 / 4.7 / **5.5** | 2.2 / 4.1 / 8.3 / 1.9 |
| BA + VGGT tracks | 2.4 / 3.7 / 4.1 / 7.5 | 3.1 / 5.4 / 6.7 / 2.9 |
| BA from VGGT + tracks | 2.4 / 3.7 / 4.2 / 6.9 | 3.2 / 5.6 / 7.6 / 2.7 |
| BA + SIFT | **1.9 / 2.9 / 3.4** / 6.1 | **2.1 / 3.3 / 4.5 / 1.8** |

Image NCC (10 frames / 30 frames / frame 0 to all / pairs >= 100 frames apart):

| | shot A | shot B |
|---|---|---|
| single view | 0.914 / 0.866 / 0.541 / 0.718 | 0.809 / 0.641 / 0.531 / 0.423 |
| VGGT alone | 0.890 / 0.826 / 0.263 / 0.662 | 0.780 / 0.592 / 0.498 / 0.373 |
| BA, flow | 0.947 / 0.902 / **0.589** / 0.762 | 0.877 / 0.694 / 0.427 / 0.345 |
| BA + VGGT tracks | 0.934 / 0.886 / 0.538 / 0.763 | 0.837 / 0.656 / **0.622** / 0.431 |
| BA from VGGT + tracks | 0.935 / 0.886 / 0.446 / 0.749 | 0.841 / 0.662 / 0.610 / 0.400 |
| BA + SIFT | 0.946 / 0.904 / 0.571 / **0.783** | 0.876 / 0.698 / 0.582 / **0.452** |

Focal length (vertical field of view), same scope in both shots:

| | shot A | shot B |
|---|---|---|
| BA, flow | 609 px (32.9°) | 594 px (33.7°) |
| BA + SIFT | 542 px (36.8°) | 594 px (33.7°) |
| VGGT alone | 1027 px (19.9°) | 980 px (20.8°) |
| BA + VGGT tracks | 647 px (31.1°) | 923 px (22.1°) |
| BA from VGGT + tracks | 1109 px (18.4°) | 1310 px (15.7°) |

SIFT finds 19 613 verified matches on shot A and 4 819 on shot B (277).

## VGGT on laparoscopic video

- Out of domain: track visibility median 0.15-0.3 and confidence 0.02 after the query frame (all below the usual 0.5
  cut-off), depth confidence near its floor. These outputs are therefore not used as filters.
- Tracks lag the true motion: against forward-backward-checked optical flow over 60 frames, VGGT tracks are 5-10 px
  off with a consistent bias along the pan (-4 to -10 px in x); denser keyframes (every 5) and 12 tracker iterations
  bring it to 4-8 px.
- It explains the scope motion as translation with a narrow field of view: <= 4° rotation and ~20° FOV, against 13-17°
  rotation and ~33° from the flow-based BA. Its depth disagrees with the instrument shafts (shot B spread 0.38 in log
  scale) and the BA residual on the shaft rulers grows from 0.09 / 0.32 (flow) to 0.15-0.21 / 0.51-0.54 with VGGT.
- Precision is not the cause (bfloat16 and float32 give the same cameras); 518 px width, 7 passes of 26 keyframes take
  42 s on the M5 Max (MPS), 6 GB.

## Simulation (organ conditions; outline IoU / tissue tracking, corrected metric)

Tracking (re-seeded every 25 frames, see `../template_keyframes_2026-10-07/README.md`): pooled median px, and in
brackets the change against the static baseline over the motion phases (seeds where the static baseline is >= 15 px
off). An earlier version of this table used the frame-0-seeded metric (35 samples on shot A) and is superseded.

| shot A | measured | ellipsoid | template | template + fit |
|---|---|---|---|---|
| single view (report) | 0.531 / 8.9 (+1 %) | 0.613 / 9.1 (+5 %) | 0.548 / 9.1 (+7 %) | 0.509 / 8.9 (-3 %) |
| BA, flow | 0.471 / 10.7 (+6 %) | 0.512 / 10.9 (+20 %) | 0.424 / 10.4 (+29 %) | 0.506 / 11.6 (+9 %) |
| BA + SIFT | 0.498 / 9.3 (+3 %) | 0.546 / 10.2 (+26 %) | 0.516 / 10.2 (+15 %) | 0.521 / 10.3 (+13 %) |
| BA + VGGT tracks | 0.524 / 9.7 (-2 %) | 0.598 / 12.0 (+19 %) | 0.403 / 10.3 (+34 %) | 0.503 / 13.0 (+47 %) |

| shot B | measured | ellipsoid |
|---|---|---|
| single view (report) | 0.346 / 13.9 (+3 %) | 0.262 / 16.6 (+19 %) |
| BA, flow | 0.270 / 11.2 (no motion phase) | 0.110 / 26.4 |
| BA + SIFT | 0.298 / 12.0 (no motion phase) | 0.124 / 34.7 |
| BA + VGGT tracks | 0.246 / 13.9 (-2 %) | 0.212 / 20.9 (+56 %) |

All rows re-run after the anatomy fix (`../anatomy_fix_2026-10-07`); before it, the measured organ gained -17 to -22 %
in the multi-view geometries and those numbers are superseded.

Depth error is not listed: each geometry is scored against its own depth. Static baselines differ between rows
because the camera model changes where the unmoved organ projects.

Robustness fixes made on the way (no effect on the report's single-view results): instrument pitch limit 1.4 -> 1.55
rad (a trocar almost straight above its target), shape-fit scales bounded at 0.1 only when an unbounded fit collapses,
one more time-step halving in the simulator fallback. Shot B + SIFT measured needed 0.016 ms; shot A + VGGT tracks
template stays unstable from the first step (probe joint).

## Reading

1. VGGT does not transfer to this laparoscopic video. On its own its cameras are worse than the single-view rotation
   track (frame-0 alignment 0.26 vs 0.54 on shot A), and as initial values it pulls the BA into a narrow-FOV solution
   whose focal length differs between two shots of the same scope (1109 vs 1310 px).
2. Wide-baseline constraints do help the geometry. SIFT matches cut the held-out long-range error by 28 % (shot A
   4.7 -> 3.4 px) and 45 % (shot B 8.3 -> 4.5 px) without hurting short range, give the best alignment of frames
   >= 100 apart, and keep a consistent ~34° field of view. VGGT tracks help less (4.1 / 6.7 px) and cost short-range
   accuracy because of their bias.
3. Downstream, better geometry does not give better tissue motion: on shot A the measured organ stays at the static
   level in every geometry (-2 to +6 %) and the templates get worse (+15 to +34 %); shot B is not reproduced in any
   geometry. (Before the anatomy fix the measured organ seemed to gain -17 to -22 % here.)
4. Frame 0 remains the weak point on shot A (no mode beats the flow BA there): it sits in a fast, blurred pan with few
   matches, and everything is expressed relative to it. A sharper reference keyframe would be the next thing to try.
