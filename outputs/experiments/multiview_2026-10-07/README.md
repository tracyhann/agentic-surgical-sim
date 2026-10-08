# Multi-view joint estimation (keyframe bundle adjustment), 2026-10-07

Question: the reconstructions use one view (frame 0) for geometry and texture plus single-image depth per frame.
The scope moves, so the clip holds several real views of the static anatomy. Does solving them jointly help?

Method (`r2s/multiview.py`): keyframes every 10 frames; correspondences on static anatomy from dense optical flow
chained over up to 3 keyframes (forward-backward checked); unknowns per keyframe = 6-DoF pose, inverse-depth affine of
Depth Anything (1/z = a d + b) and a smooth 4x6 multiplicative depth correction; one focal length per clip; residuals =
reprojection (Huber), cross-view depth agreement, instrument shafts of known diameter for scale. L-BFGS on poses and
focal length, then Adam on everything. Non-keyframes interpolated. Run with `"multiview": true` in a clip config.

| | shot A | shot B |
|---|---|---|
| reprojection median (before → after) | 5.1 → 2.1 px | 9.2 → 2.7 px |
| depth disagreement between keyframes | 7.1 → 3.1 % | 12.7 → 3.6 % |
| focal length (assumed 414 px, 47 deg) | 615 px, 32.6 deg | 594 px, 33.7 deg |
| max scope translation | 17 mm | 9.5 mm |
| alignment NCC, keyframes 10 / 20 / 30 frames apart (rotation+zoom → joint) | 0.912/0.893/0.868 → 0.944/0.923/0.902 | 0.808/0.710/0.641 → 0.876/0.773/0.694 |
| alignment NCC with frame 0 (long range) | about equal | equal or worse (frame 97: 0.16 → 0.07) |

Simulation (`summary.json` here vs the single-view run in the report). Note: the track errors below use the old
frame-0-seeded metric (35 samples on shot A) and are superseded; the same geometry under the corrected metric is in
`../wide_baseline_2026-10-07/README.md` (row "BA, flow").

| shot A | outline IoU | track error (px) |
|---|---|---|
| measured | 0.540 → 0.587 | 19.6 → 15.2 |
| primitive | 0.536 → 0.522 | 27.2 → 26.0 |
| template | 0.569 → 0.490 | 16.1 → 20.6 |
| template_fit | 0.572 → 0.539 | 24.2 → 20.6 |

Shot B: measured 0.346 → 0.270 IoU, 15.2 → 13.1 px track (its static baseline moved from 14.1 to 9.1 px);
primitive worse.

Reading: joint estimation makes neighbouring views consistent (local alignment, depth agreement) and both shots
independently agree on a ~33 deg vertical field of view (the 47 deg guess was too wide). It does not fix long-range
drift: flow chains break during fast pans, so nothing ties frame 0 to frames seconds later. The measured organ gains
from the more consistent depth; the templates lose. (Correction, same day: an earlier version said the scene comes out
~1.45x deeper; it does not - the organ sits at 6.7-8.2 cm vs 6.2-9.0 cm single view, i.e. flatter, and the cause of the
template loss is not established; see ../wide_baseline_2026-10-07.) Next: wide-baseline correspondences (MASt3R / VGGT) for global consistency.
