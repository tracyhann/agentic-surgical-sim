# membrane v09: v08 re-snapped to gallbladder v12's 4D (the one the simulation uses) (current version)

`python -m r2s.tissue.membrane v09` (code `r2s/tissue/membrane.py`, `CONFIGS['v09']` = v08's settings with the changes
below; about 2.8 min on CPU).

Inputs, read-only: refined cameras (backdrop v08), `gallbladder/v12` verts4d, `interaction/v06` TCP / tip, and v08's
mesh. The pipeline, the held row at the TCP with its tongue, and the masks are as in v08 (../v08/NOTES.md).

## Changes vs v08
1. **Base and clearance against v12.** Base cast onto v12 in every frame. Every non-base vertex sits at least 1 mm
   in front of v12 along its camera ray (exact ray-mesh test).
2. **Exact base depth.** The depth map used for casting is conservative: up to about 1-2 mm in front of the true
   surface where it is slanted. So base depths are corrected to the exact ray-mesh intersection (minus the 0.25 mm
   offset). The correction is smoothed over 1 frame (sigma), so hit / miss changes do not jerk.
3. **Clearance tapered near the base.** The 1 mm clearance drops to the 0.25 mm base offset over the last 10 % of s.
   Without the taper, a vertex next to the base was pushed out of line and left a 4.8 deg sliver at rest.
4. **Same vertices and attachments as v08.** 237 vertices, `attach_idx` / `attach_to` identical. The rest-frame
   polish (frame 162) flipped 28 of the 395 faces.
   - Without the polish, the smallest rest angle was 13.6 deg (2 faces below 15).
   - Two optional mesh repairs were added (`polish_per_vertex`, `repair_slivers`). Neither was needed once the
     clearance was tapered.

## Results
| | v08 | v09 |
|---|---|---|
| IoU / BF 4 px, frames 0-61 (every frame, refined cams) | 0.756 / 0.413 | 0.757 / 0.418 |
| IoU / BF 4 px, frames 62-129 | 0.744 / 0.451 | 0.746 / 0.451 |
| IoU / BF 4 px, frames 130-250 | 0.823 / 0.516 | 0.818 / 0.507 |
| IoU / BF 4 px, all frames | 0.785 / 0.473 | 0.783 / 0.470 |
| base -> v12 surface over all frames x base vertices (median / p99 / max) | 0.42 / 1.29 / 2.79 mm | **0.19 / 0.56 / 1.44 mm** (> 1 mm: 0.02 %) |
| non-base vertices behind v12 | 0.17 % mean, max 2.7 % | **0 in every frame** |
| sheet pixels hidden by v12 (render z-test) | 0.39 % | 0.43 % (max 1.7 %) |
| held-row centre -> v06 TCP | - | 0.00 mm median, 0.93 mm max |
| smallest rest angle | 18.2 deg | 20.4 deg |
| stretch max / p95 | 3.1 / 1.86 | 2.9 / 1.92 |

How the base distance is measured: the closest point on v12's 4D surface (250k surface samples, about 0.15 mm
resolution), for every base vertex in every frame, through the refined cameras. The data are in
`work/base_to_v12.json`. The integration anatomy check found 1.9-2.0 mm median for v08 against v12. My own
measurement of v08 is 0.42 mm, so that check probably measures differently (another camera set or the simulated
state); worth a recheck with v09.

`attach_gb_face` / `attach_gb_bary` / `attach_gb_dist_mm` now refer to v12's faces at the rest frame (0.19 mm median,
0.74 mm max).

## Known problems
As in v08. In addition, 0.8 % of vertices lie slightly behind v12 per frame. These are base vertices within the
0.25 mm offset; none is more than 1 mm behind, apart from a few at the edge of v12's outline.
