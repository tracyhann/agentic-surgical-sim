# ROSMA post-and-sleeve hand-off: reconstruction report

## Reading of the video
A white skills board with 12 vertical posts (two outer columns of 4, two inner pairs) sits on a clear box. Purple and
pink sleeves stay on right-hand posts. **psm_right** grasps the yellow sleeve on the front-right post by its top rim
(frames 0-60), lifts it off (62-80), turns it roughly horizontal with its wrist and carries it to mid-air above the
board centre (~98). **psm_left** grasps the free end (108-165), psm_right lets go (~172), psm_left turns the sleeve to
hang vertically (176-196), carries it over the **front-left post** (204-218), lowers it part-way and releases (~228);
the sleeve drops onto the post (230). Roles: object `sleeve_yellow`, target `post_L1`.

## Measurements (all from pixels; images in `source/frames/`, scripts `source/calib*.py`, `source/gen.py`)
- Metric reference: 8 mm shaft. Clevis/shaft of psm_right next to the sleeve is ~22-24 px wide while the sleeve is
  ~65 px wide -> sleeve OD ~= 20-22 mm (used 20 mm across flats), length ~= 30 mm (ratio from the image).
  That fixes the board width at ~200 mm.
- Camera: least-squares fit of position / look-at to the 4 board-top corners and to 4 posts (base + top, assumed
  normal to the board), for several fixed vertical FOVs. Residual 3.6 px at 35 deg, growing to 8.6 px at 60 deg; at
  35 deg the board comes out square (200 x 197 mm) and the back-projected post bases form a regular 40 mm grid
  (columns at x = -63, -22, +19, +59 mm), which I take as confirmation. Result: `pos = (0.008, -0.403, 0.154)`,
  `lookat = (-0.002, 0.136, 0)`, `fovy = 35 deg`.
- The image has ~4.5 deg of roll (front board edge slopes, verticals lean asymmetrically). The protocol camera cannot
  roll, so the roll is absorbed as a 0.079 rad tilt of the board about world y (right side higher). Gravity is
  therefore ~4.5 deg off the board normal in the model; in reality the camera is probably the rolled thing.
- Posts: diameter ~7 mm, height 43 mm (fit). Sleeve: OD 20 mm, bore 14 mm (bore looked ~45-50 % of OD; made slightly
  generous so the 2.4 mm jaw fits between post and wall), 1.5 g (foam-like, ~6 cm^3), modelled as an octagonal ring
  of 8 boxes (flat facets for the jaws).
- Hand-off point: sleeve centre at pixel ~(505, 292) -> about x = 0.00, z = 0.08-0.09 m in front of the board centre.
- RCMs: both shafts stay nearly parallel across the clip (the pivots are well outside the image) and I could not
  triangulate them: no measurable taper of the shaft width. They are placed left / right of the board,
  `(-0.17, -0.084, 0.20)` and `(0.17, -0.067, 0.20)`, 0.2-0.25 m from the tips. **This is the least certain part**:
  the real shafts look shallower (~20-30 deg) than mine (35-57 deg); the height was chosen so that a wrist-less
  instrument can do the task (see below).

## Behaviour (policy.py)
The supplied instrument has no wrist, the real one uses it twice (sleeve vertical -> horizontal -> vertical). I
reproduce those re-orientations with the roll joint: a jaw that pinches the sleeve wall at an angle to the sleeve axis
swings the sleeve over a cone when the shaft rolls by pi.
1. psm_right straddles the near wall of the sleeve at the top rim (one jaw in the bore beside the post), closes, lifts
   52 mm along the post, then rolls by pi while carrying: the sleeve ends up roughly horizontal, free end towards
   psm_left, as in the video.
2. The hand-off height is solved so that psm_left, entering the free end, holds the sleeve at exactly the jaw/axis
   angle that makes it parallel to the destination post at the placement pose after its own roll by pi.
3. psm_left enters the bore along the sleeve axis, closes; psm_right opens and backs out along its shaft.
4. psm_left rolls by pi (sleeve hangs), moves over `post_L1`, corrects its position twice from the measured sleeve
   pose, lowers until the sleeve is 13 mm over the post, opens; the sleeve drops to the board around the post.
`plan()` is closed-loop: it rolls a copy of the simulation forward segment by segment and computes each waypoint from
the simulated sleeve / post poses, so it adapts to displaced layouts. Event times follow the video (lift ~4.3 s,
hand-off ~7-11.6 s, release ~15.9 s; 342 steps = 17.1 s).

## Verification
- `validate`: OK. `replay`: sleeve ends on the board around `post_L1` (final centre within ~3 mm of the post axis, i.e.
  inside the bore clearance, axis parallel to the post); side-by-side frames agree with the video in each phase.
- Own displacement test (sleeve + source post and/or destination post moved by up to 1 cm, 12 random layouts):
  12 / 12 end with the sleeve on the destination post.

## Not observable / known deviations
- Depth of the tool tips while parked and the true RCMs (single view, no taper); the parked poses are approximate.
- Shaft inclination differs from the video (see RCMs); tool-tip paths match the phases but the jaws approach from
  steeper directions, and the instrument appearance (wrist, jaw shape) is not reproduced.
- The sleeve is an octagonal prism, not a round tube; it starts 2.5 mm off-centre on its post so that a jaw fits in
  the gap. Mass, friction and stiffness are plausible guesses (rigid, 1.5 g).
- The sleeve is put down flipped end-for-end relative to the video's wrist manoeuvre (symmetric object).
- Other sleeves (purple, pink, far yellow) are visual only; the transparent box under the board is a visual block.
- Lens distortion ignored; FOV is only weakly constrained (35-45 deg fit almost equally well).
