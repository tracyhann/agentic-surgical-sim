# interaction v06: v05 refitted through the refined cameras (current version)

Run with `python -m r2s.tissue.interaction v06` (settings in `VERSIONS['v06']`, about 40 s on CPU).
- The model, fit, array names and meanings are the same as v05; see `../v05/NOTES.md`.
- What changed:
  - Cameras: `views.load('chole_a', 'sift', cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz')`. These
    rotations are registered between keyframes (median correction 0.3 deg, max 1.9 deg). Position and focal length
    are unchanged.
  - The sheet cross-check now uses the membrane v06 mask.
  - New analysis: per-band comparison with v05 on the same cameras.
  - New analysis: independent evidence on the probe's depth.
- Size of the change from v05: tips move by a median of 0.35 mm (max 2.1 mm), ports by 2.6 mm (grasper) and 3.3 mm
  (probe). Contact frames are unchanged (0-113 and 142-250). The probe's indentation in contact is a median of
  +5.2 mm.

## Tip reprojection (px) and shaft angle (deg), median per frame band
"clip cams" is v05 as delivered. The other three columns are measured through the refined cameras.

| instrument / band | v05, clip cams | old tracks | v05 | v06 |
|---|---|---|---|---|
| grasper tip, 0-61 | 1.98 | 40.8 | 4.38 | **2.14** |
| grasper tip, 62-129 | 1.34 | 30.7 | 2.91 | **1.26** |
| grasper tip, 130-250 | 2.75 | 10.1 | 3.86 | **2.79** |
| grasper angle, 0-61 / 62-129 / 130-250 | 1.4 / 1.6 / 4.3 | 2.0 / 1.9 / 7.4 | 1.4 / 1.5 / 4.4 | 1.5 / 1.4 / 4.6 |
| probe tip, 0-61 | 0.80 | 31.8 | 3.47 | **0.79** |
| probe tip, 62-129 | 0.62 | 33.4 | 2.36 | **0.58** |
| probe tip, 130-250 | 0.31 | 27.4 | 1.78 | **0.31** |
| probe angle, 0-61 / 62-129 / 130-250 | 0.7 / 1.9 / 1.5 | 0.8 / 1.2 / 1.1 | 0.9 / 2.0 / 1.3 | 0.8 / 1.7 / 1.5 |

Whole clip, v06: grasper IoU 0.84, probe IoU 0.76, kappa 0.985 / 0.877, no joint clamped.

The refined cameras do not improve the instrument fit. They only shift the projections: v05 tracks seen through
them lose 1-3 px, and refitting (v06) recovers v05's accuracy. Use v06 together with the refined cameras.

## Probe depth: independent evidence
All numbers are mm along the viewing ray, relative to the v06 tip, with + = away from the camera, over the 223
contact frames. Per-frame data are in `work/probe_depth_evidence.npz`.

1. **Width ruler alone** (5 mm shaft, kappa = 1): median +14.9 mm (p10 +10.6).
   - This puts the shaft behind the tissue it is seen against, so the cue is biased. Either the probe is about 4.4 mm
     thick or its mask is about 1-2 px too narrow.
   - It cannot be used for the absolute depth.
2. **Free space (upper bound).** The distal half of the visible shaft must lie in front of the tissue beside it.
   - Measured: V.depth 8 px beyond the mask edge, the 20th percentile over samples, minus the shaft radius.
   - Moving the probe away from the camera is allowed by at most -0.1 mm (median), and p10 is -4.4 mm. At the v06 depth
     the shaft already touches or cuts the tissue beside it in 51 % of contact frames.
   - This bound favours moving the probe toward the camera.
3. **Sheet in front (lower bound).** In 127 contact frames the membrane-v06 sheet covers at least half of the ring
   around the probe tip.
   - The tip lies under the sheet there, so it must stay behind it.
   - Allowed movement toward the camera: median 5.7 mm (p10 0.1 mm).
4. **Fraction of contact frames where both bounds hold**, by shift:

   | shift (mm) | +2 | 0 | -1 | -2 | -3 | -4 | -6 |
   |---|---|---|---|---|---|---|---|
   | both bounds hold | 0.35 | 0.44 | 0.51 | 0.57 | 0.64 | 0.68 | 0.67 |

   The plateau runs from -3 to -6 mm.
5. **Rounded end visible.** At the distal end the mask narrows to 0.30 of the shaft width (median), as a rounded cap
   seen whole would. So the tip is not hidden behind a dimple's rim.
6. **Depth map only** (V.depth on the probe's own tip pixels minus the tissue ring): +0.3 mm. Monocular smoothing
   biases this towards 0, so it is weak evidence.

**Verdict.** A shift of -2 mm (toward the camera) is consistent with this evidence. The evidence favours -2 to -5 mm;
-2 mm is the conservative end. After the shift the indentation in contact is about +3 mm.

Caveat: bounds 2 and 3 use V.depth of the tissue and the sheet. They are independent of the probe's own pixels and of
the gallbladder 4D, but not of the depth map. The width cue alone disagrees by about 15 mm and is not trusted.
