# bead_cup — reconstruction report

## Reading of the video
A straight laparoscopic grasper enters from the right, reaches a **blue cylindrical bead** lying on a flat red
table, closes its jaws on it (~frame 85–100), lifts it, carries it over a **cream ring-shaped cup** (12-sided
wall, open top) and releases it inside (~frame 205), then retracts to its starting pose. A green bead is a
distractor and is never touched.

## Method and measurements (all from pixels; frames in `source/frames/`)
- **Focal length / pitch**: the table is a rectangle. Its left and right edges (fitted lines) meet at the
  vanishing point (280, −51) px and the far edge vanishes at (≈5930, −51). Orthogonality of the two directions
  with the principal point at the image centre gives f ≈ 373.5 px → **fovy ≈ 65.4°**, and the horizon row gives a
  **downward pitch of ≈ 38°**. Check: back-projected table sides come out parallel (y = 0.072/0.073 and
  −0.083/−0.083 m at two depths each).
- **Metric scale and RCM**: in 17 frames (35–130) the two silhouette edges of the 5.0 mm shaft over the table were
  fitted with sub-pixel lines (residual ≈ 0.15 px). The two tangent planes of a cylinder of known radius give the
  3-D axis in metric camera coordinates; the least-squares common point of all axes is the **RCM** (axes pass
  within 0.1–0.8 mm of it, except two short-segment frames at 4–6 mm).
  The axis at the grasp frames intersected with the bead's pixel ray (gap 0.6 mm) puts the bead 70.4 mm below the
  camera → **camera 73.2 mm above the table**. The RCM is at about the camera's height, ~7 cm to its right.
- **World frame**: origin on the table under the camera, z up, +y along the table's far edge (pointing image-left),
  +x away from the camera. Camera (0, 0, 0.0732) looking at (0.0935, −0.0079, 0); RCM (0.0252, −0.0660, 0.0725).
- **Objects** (pixel rays intersected with the table plane):
  - bead: Ø 8.0 mm × 5.5 mm cylinder at (0.0862, 0.0129); mass 0.5 g (assumed, small plastic bead).
  - cup: outer Ø 22.5 mm, height 7.5 mm, wall 1.5 mm, no visible floor (table seen through it), at (0.105, −0.0267).
  - distractor: Ø 8 mm × 6 mm green cylinder at (0.112, 0.0245).
  - table: 138 × 156 mm slab (edges back-projected: x 0.022…0.159, y −0.083…0.073).
- **Initial instrument pose**: jaw-tip midpoint pixel (453,189) and clevis centre pixel (550,163) in frame 0, with
  the 22 mm clevis-to-tip distance, give TCP = RCM + (0.0447, 0.0280, −0.0409) (reprojection error 0.4 px);
  jaws fully open, roll 0.

## Behaviour (`policy.py`)
`plan` reads the bead and cup positions from `data` and the RCM from the model, builds Cartesian TCP waypoints
(start → 12 mm above bead → bead centre +1 mm → close → lift 26 mm → over cup centre → lower to 15 mm → open →
retract → start) and converts them with the closed-form IK, subdividing each segment until the per-step change is
within 60 % of the slew limits. Phase durations follow the video (281 steps vs 284 frames).

## Physical replay
- `validate`: OK. `replay`: bead goes from (0.0862, 0.0129) to (0.105, −0.0266), i.e. 0.1 mm from the cup centre,
  resting on the table inside the wall, at rest; max height 28 mm; distractor does not move.
- Rendered rollout vs video: bead / cup / distractor bounding boxes agree within 0–2 px at frame 0 and at the end;
  the carried bead agrees within ~2–6 px at frames 170 and 200. The sim lifts a little faster than the video
  around frame 130 (bead ≈ 12 px higher there).
- Displaced layouts (my own test, bead or cup moved 1 cm in 8 directions each): 17/17 end with the bead inside the
  cup, slew limits respected.

## Not observable / assumed
- Masses, friction and contact parameters (chosen: 0.5 g, friction 0.8).
- Whether the cup has a floor, and the table thickness (10 mm slab assumed; only its top face matters).
- Absolute scale rests on the 5 mm shaft diameter measured at ~10 px width; I estimate a few percent uncertainty
  in all lengths. fovy depends on the small tilt of the far table edge (≈ ±3° uncertainty).
- The exact jaw opening during approach and the distractor's height are approximate.
