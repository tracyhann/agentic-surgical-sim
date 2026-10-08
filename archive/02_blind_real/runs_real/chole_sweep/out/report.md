# chole_sweep – reconstruction report

## Reading of the video
Laparoscopic cholecystectomy, static camera. `grasper_left` (dark shaft, upper left, pointing steeply away from the
camera) holds the gallbladder neck throughout and retracts it up/left (largest lift around frames 80–95), then relaxes.
`probe_right` (bright straight shaft, upper right, jaws closed) first dips onto the right flank of the gallbladder
(frames 5–20), sweeps upward beside the neck along the purple peritoneal/Calot tissue (frames 25–50), withdraws out of
view (frames 55–68), re-enters, sweeps up again next to the neck (frames 75–95) and ends resting on the tissue to the
right of the neck (frames 100–150).

## Method
- Frames extracted with OpenCV (`source/frames/f*.png`, `sheet.png`, `sheet2.png`).
- Probe tip tracked automatically in all 151 frames (bright low-saturation blob touching the image border, PCA line,
  distal end); shaft width near the tip ≈ 15 px (blooming probably inflates it slightly). Grasper tip positions and
  tissue outlines were read by eye from the gridded sheets (8 key frames).
- Camera: pinhole, fovy **assumed** 60° (f ≈ 312 px), placed at (0, −0.09, 0.06) looking at the origin. With 5 mm ≈ 15 px
  the probe tip is ≈ 0.10 m from the camera; the grasper shaft is ≈ 30 px wide where it leaves the frame, so it is at
  ≈ 0.055 m there, i.e. it runs steeply away from the camera. All pixel measurements are unprojected at those depths.
- RCMs: grasper = 0.12 m back along the 3-D shaft line (tip → frame-exit point); probe = 0.20 m back along its image
  direction at constant depth (image direction changes only ≈ 0.1 rad over the clip, so the pivot is far away).
- Tissue (rigid bodies + spring joints, no flex): gallbladder body (24 mm-radius ellipsoid, 20 g, ball joint with
  stiffness to the bed at its base), a stretchable neck (slide spring 4 N/m + three ball-jointed tapering capsules, the
  last one 2 mm radius with a small bulb distal to the jaws so the neck is held by pinch + shape), and a separate
  purple "Calot/peritoneal strand" (6 ball-jointed capsules rooted at static anatomy near the duodenum, spring-loaded)
  lying beside the neck where the probe sweeps. Static: bed plane, tilted liver backdrop, a duodenum bump.
- Policy: reads `gb_neck3`, `gb_body`, `strand0` poses from `data`; the grasper closes on the measured neck point and
  then follows the video's relative tip path; the probe path (29 tracked key points, IK through the RCM) is shifted by
  the tissue displacement. A per-step rate limiter guarantees the slew limits. `build.py` regenerates everything.

## Physical replay (final package)
`validate` prints OK. In the replay the grasper keeps hold of the neck for the whole episode and lifts it
(neck segments move up to 18 mm, peak at t ≈ 3.75 s, matching the lift around frame 90); the gallbladder body tilts
≈ 3 mm; the probe's second sweep and final rest deflect the strand by up to 9 mm (tip segment). The probe leaves and
re-enters the view at about the right times. `source/frames/replay_compare_sheet.png` shows 8 side-by-side samples.

## Limitations / not observable
- Field of view and absolute depth are not observable; everything scales with the assumed fovy and the 15 px width
  estimate. The grasper RCM depth in particular rests on one rough width reading (≈ 30 px).
- The real gallbladder is a continuous soft sac, the neck being continuous with the purple peritoneum; here they are a
  jointed chain and a separate strand, so tissue deformation is only qualitatively similar (no surface indentation).
- The first probe contact (flank poke, frames 5–20) produces very little body motion in the simulation; the first
  upward sweep (frames 25–50) only lightly touches the strand. Masses and stiffnesses are plausible guesses, not measured.
- The probe's fast re-entry (frames 70–75) exceeds the insertion slew limit and is therefore slightly delayed.
- The grasper starts with the jaws 0.5 mm clear of the neck and closes in the first 0.1 s (video starts already
  grasped). Robustness to displaced layouts was not tested; a 1 cm shift of the neck may make the open jaws miss it.
- Instruments differ in appearance from the video (the real left tool has a different jaw shape).
