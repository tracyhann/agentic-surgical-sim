# membrane v08: v07 re-snapped to gallbladder v13, held row at the grasper TCP (superseded by v09: same, against gallbladder v12)

`python -m r2s.tissue.membrane v08` (code `r2s/tissue/membrane.py`, `CONFIGS['v08']`; about 2.7 min on CPU).

Inputs, read-only:
- Refined cameras: `backdrop/v08/cams_refined.npz`.
- `gallbladder/v13`.
- `interaction/v06` (`grasper_left_tcp`, `grasper_left_tip`).
- v07's mesh (`st`, `faces`).

The pipeline is v07's (../v07/NOTES.md), with these changes.

## Changes vs v07
1. **Same vertices as v07.** 237 vertices with the same (s, t) and the same `attach_idx` / `attach_to` (5 grasper,
   52 gallbladder); still 395 faces.
   - With v13 and interaction v06, frame 162 of the plain ruled surface already had triangles of 6.3 deg.
   - So the triangulation is polished again on the final rest surface (frame 162) with 3D Delaunay flips. 60 of the
     395 faces were flipped; the vertices are unchanged.
   - A polish without flips (to keep v07's faces exactly) failed: smallest rest angle 3.4 deg.
2. **Gallbladder v13.** Base cast onto v13, all non-base vertices pushed at least 1 mm in front of it.
   - The "behind" test is now an exact ray-mesh intersection.
   - New arrays for attaching by position: `attach_gb_face` / `attach_gb_bary` (index into v13 `faces`, barycentric)
     and `attach_gb_dist_mm`, one per base vertex at the rest frame. The distance is 0.36 mm median, 0.70 mm max.
3. **Held row at the grasper TCP.**
   - Evidence from the video, through the refined cameras:
     - The v06 TCP projects inside the grasper mask in 100 % of frames (hidden between the jaws).
     - The sheet leaves the jaws 13.9 px from it, median (where the sheet pixels meet the end of the grasper mask).
     - The TCP lies 2.6 mm farther than the video depth at that point, median.
   - Putting the whole fan origin at the TCP cost IoU: 0.755 -> 0.69 in frames 0-61 and 0.762 -> 0.721 in 62-129
     (scratch test). The jaw tip as origin did no better.
   - So the jaw row (s = 0) sits at the TCP, and a short tongue joins it to the v07-style sheet: offset
     (TCP - apex) * (1 - s/0.15)^2, mostly hidden under the jaws. The sheet beyond s = 0.15 is unchanged.
   - Frames 69-87: v13 bulges up next to the TCP. There the jaw row's end vertices move forward along their rays by
     up to 2.3 mm, to 0.25 mm in front of v13. The held-row centre then shifts by up to 0.78 mm.

## Results (refined cameras; IoU / BF 4 px vs our mask, every frame)
| | v07 | v08 |
|---|---|---|
| frames 0-61 | 0.756 / 0.420 | 0.756 / 0.413 |
| frames 62-129 | 0.743 / 0.467 | 0.744 / 0.451 |
| frames 130-250 | 0.821 / 0.515 | 0.823 / 0.516 |
| all | 0.784 / 0.478 | 0.785 / 0.473 |
| keyframes | 0.790 / 0.480 | 0.792 / 0.480 |
| held-row centre -> interaction v06 TCP | 3.15 mm median (max 4.26) | 0.00 mm median (max 0.78) |
| non-base vertices behind v13 (exact ray cast) | 1.0 % mean, max 11 % | **0 in every frame** |
| all vertices behind v13: strictly / by > 1 mm | 5.6 % / 1.9 % | 1.5 % / 0.02 % (base, within its 0.25 mm offset) |
| sheet pixels hidden by v13 (render z-test) | 0.71 % | 0.22 % (max 1.1 %) |
| base -> v13 surface, per-frame median | - | 0.40 mm (per-frame max <= 1.95 mm) |
| smallest rest angle | 19.6 deg (v07 mesh) | 18.2 deg, 0 faces < 15 deg |
| stretch max mean (max) / p95 / p05 | 4.5 (10.1) / 2.2 / 0.53 | 3.1 (4.9) / 1.86 / 0.52 |
| mean vertex accel by band | 0.35 / 0.25 / 0.24 | 0.36 / 0.27 / 0.26 mm/frame² |

- The image apex now projects 13.9 px from the held row. That is expected, because the held row is the TCP inside
  the jaws.
- Front push: per-frame maximum 3.0 mm on average (max 6.5 mm). This includes the pan frames where v13 sits 5-8 mm in
  front of the v07 sheet.

## For the integrator
- **Vertices and attachments** are v07's: 237 vertices, `attach_idx` / `attach_to` unchanged. **Use the new
  `faces`**, because 60 triangles were flipped.
- **Held row:** the 5 'grasper' vertices sit on the v06 TCP (grasp_target = tcp agrees with the 4D now).
- **Base:** the 52 'gallbladder' vertices sit on the v13 surface. For each one, `attach_gb_face` / `attach_gb_bary`
  give its face and barycentric coordinates on v13 at the rest frame (162).
- **Bending:** every rest triangle is >= 18 deg.

## Known problems
- Mask flicker and the polar-fan limits (../v05/NOTES.md) remain.
- The (s, t) samples are not material points, so the 4D stretch is resampling noise (p95 1.9).
- Frames 69-87: the held row bends by up to 2.3 mm where v13 meets the TCP.
- About 48 of 395 faces drop below 15 deg per frame in verts4d (rest is fine).
