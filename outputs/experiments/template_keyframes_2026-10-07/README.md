# Organ template fitted to several keyframes, 2026-10-07

Question (from the user): can the multi-view work be combined with the organ template? Until now the template
(BodyParts3D gallbladder, pose + anisotropic scale) was fitted to frame 0 only; multi-view geometry reached it only
through frame 0's depth. Here later keyframes join the fit, through their own cameras.

## Method (`r2s/organ.py`: `KeyView`, `keyframe_views`; `r2s/__main__.py`: `keyframe_views`)

Per extra keyframe k, the same kinds of residuals as for frame 0: visible organ points at their depth (camera k) to the
front-facing template surface; template samples projecting outside the visible organ outline (instrument / strand
pixels are unknown, samples hidden behind the visible surface are free); the organ outline (minus the parts against
instruments or the border) covered by the projected template. Extra keyframes share a total weight equal to frame 0.
The fit starts from the frame-0 solution. Two conditions, run only when named (`python -m r2s <clip> organ template_mv`):

- `template_mv` (rigid): keyframes (every 5 frames) where the frame-0 organ held still keeps >= 75 % of its frame-0
  outline IoU. The gallbladder is pulled from the start of shot A, so this leaves frames 5, 10, 30 (single view;
  11 keyframes in the SIFT geometry, where the late frames happen to pass).
- `template_mvd` (deformation-compensated): every 10th frame where the existing `template` simulation matches the
  outline (IoU >= 0.6); the candidate template is displaced by that simulation's motion at the keyframe (4 nearest
  rest vertices, inverse distance) before it is compared. 11 keyframes (single view), 4 (SIFT).

## Results, shot A (outline IoU / depth error mm / tissue tracking)

Re-run after the anatomy fix (`../anatomy_fix_2026-10-07`: the gallbladder neck now belongs to the organ). Tracking:
pooled median px and the motion-phase change against static (seeds 0, 50, 75 in both geometries).

| single-view geometry | keyframes | IoU | depth | track px | motion phases |
|---|---|---|---|---|---|
| template (frame 0) | 1 | **0.548** | 9.7 | 9.1 | +7 % |
| template_mv | 2 (frame 5) | 0.528 | **8.6** | **8.8** | **+1 %** |
| template_mvd | 5 (20, 110, 230, 240) | 0.535 | 8.9 | 9.3 | +4 % |
| measured (reference) | 1 | 0.531 | 13.9 | 8.9 | +1 % |

| SIFT geometry | keyframes | IoU | track px | motion phases |
|---|---|---|---|---|
| template | 1 | 0.516 | 10.2 | +15 % |
| template_mv | 2 (frame 35) | 0.453 | 10.3 | +16 % |
| template_mvd | 7 | 0.483 | 10.0 | +34 % |
| measured | 1 | 0.498 | 9.3 | +3 % |

Before the anatomy fix (neck segmented as a strand) the same comparison gave: template -20 %, template_mv -24 %
(4 keyframes), template_mvd -15 %, measured -8 % in the single-view geometry. `templates_rest.jpg`: the three
single-view fits at rest.

## Reading

- With the whole gallbladder (body and neck) as the organ, its outline departs from frame 0 within a few frames, so the
  rigid fit finds one usable extra keyframe; the deformation-compensated fit four. Both move the results only a little
  (depth 9.7 -> 8.6-8.9 mm; motion phases +7 -> +1 / +4 %) and none beats the static baseline; in the SIFT geometry
  they are worse.
- Compensating the deformation with the simulation's own motion cannot help while that motion does not match the
  video. For the template to use the whole clip, the deformation has to be part of the fit (template + simulated
  deformation fitted to all observations), not fit-then-simulate.

## Tracking metric correction (applies to every result in the project)

The earlier tracking error seeded organ texture tracks once, at frame 0. Shot A opens with a fast, blurred pan:
132 tracks at frame 0, 6 left at frame 5, none after frame 110, so shot A's numbers rested on 35 samples (the 6 s
window had 446, shot B 500). `evaluate.organ_tracks` now re-seeds every 25 frames and follows each batch for 2 s;
`evaluate()` binds every batch to the simulated surface at its seed frame and reports the pooled median
(`track_err_px`), per-seed medians (`track_by_seed`) and the old frame-0 seed for reference. Shot A now has ~3000
samples from 10 seeds. Motion phases (`evaluate.motion_seeds` / `track_motion`) are chosen on the measured condition's
static baseline only, so the selection does not favour any simulation.
