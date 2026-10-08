# Disconnected anatomy fixed (gallbladder neck, strand ties, grasp), 2026-10-07

The user looked at shot A in the 3D viewer (geometry mode) and saw it "broken, completely wrong". Three defects, all
fixed, all clips and geometry variants re-run.

| defect | cause | fix |
|---|---|---|
| gallbladder cut in two at the neck | the dark neck right of the probe tip (running into the cystic duct) was segmented as `strand`, so the body became a 3D organ and the neck a separate flat ribbon | `r2s/clips/chole_a.json`, `chole_6s.json`: neck prompts moved to the gallbladder (plus a frame-71 prompt in shot A); the strand keeps only the thin pink strands |
| strand pulled out into a spike / torn | only 1 of 321 strand vertices was within 12 px of the organ, so the strand hung on one vertex; the organ's motion dragged it into a needle (and tore it in the SIFT geometry, edges stretched up to 8.5x) | `scene.strand_ribbon`: a band of vertices at the organ end is tied (>= 8; 35 in shot A, 58 in the 6 s window) |
| grasper holding the organ at a distance | grasped vertices were fixed to the jaws at their current offset, so the 2-11 mm reconstruction gap stayed (the ellipsoid hung on 1 vertex 11 mm away) | `sim._run`: the patch within 8 mm of the closest surface point is drawn into the jaws over 0.25 s (`grasp_ramp`) |

Robustness fixes made during the re-run: duplicate vertex snaps in `organ.remesh` (gave zero-volume tetrahedra and NaNs
at step 0) and a degenerate-mesh check in `organ.to_tets`; `instruments.ik` clamps targets just outside the joint
ranges (<= 5 mm / 0.05 rad) instead of failing; the shared organ-track file is written atomically (parallel evals).

## Effect (motion phases: tissue tracking against static, seeds where static is >= 15 px off)

| | before | after |
|---|---|---|
| 6 s window, measured / ellipsoid / template / template+fit | -49 / -44 / -34 / -33 % | -34 / -33 / -32 / -34 % |
| shot A, measured / ellipsoid / template / template+fit | -8 / -8 / -20 / -17 % | +1 / +5 / +7 / -3 % |
| shot A depth error, measured / ellipsoid / template (mm) | 11.7 / 11.9 / 10.5 | 13.9 / 9.4 / 9.7 |
| shot A outline IoU, measured / ellipsoid / template | 0.540 / 0.536 / 0.569 | 0.531 / 0.613 / 0.548 |
| shot B, measured | +3 % | +3 % (no grasper, segmentation unchanged) |

## Which change did it (shot A)

- Old segmentation + new grasp and ties (`outputs/variants/chole_a+oldseg`, prep pre-filled with the old masks):
  measured -12 %, ellipsoid -1 %, template -19 %, template+fit -12 %; close to before. The simulation fixes are not
  what removed the gain.
- New simulations scored on the old texture points: measured +12 %, template +6 %; so it is not just that the score
  now includes the neck (the fastest-moving part near the jaws): the simulations of the connected organ themselves
  do not follow the pull.
- New segmentation without the strand tie (`ribbon_tie: false`): measured +1 %, template +6 %; the tie is not the cause.

Reading: once the gallbladder is reconstructed as one organ, no organ body reproduces the 10 s shot A's pull better
than holding it still; the 6 s window still is (about a third better). The earlier "template best on the long shot"
rested on the broken reconstruction. The simulation parameters (stiffness, attachment springs, grasp location) are
guesses; fitting them to the video is the next step.
