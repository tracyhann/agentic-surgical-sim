# Post-and-sleeve handover: reconstruction report

Status: **success**. In physical replay the right instrument lifts the yellow sleeve off the front-right post,
hands it to the left instrument in mid-air, and the left instrument drops it over the front-left post.

## Result of physical replay (my scene, my controls)
- `validate`: `OK`.
- Replay of `actions.npy` and of `plan()` (`--policy`) give the same outcome: sleeve starts at
  (0.0567, -0.0567, 0.0125) and ends at (-0.0567, -0.0581, 0.0125), upright, resting on the board around the
  target post (1.4 mm off the post axis). Held by R for 8.7 s, by L for 6.05 s, max centre height 75 mm, duration 21.0 s.
- Target post displaced by up to 1 cm (9 layouts: ±1 cm in x, in y, and ±7 mm diagonals): the sleeve ended on the
  post, upright, 1.4 mm off axis in every one; all plans passed the reachability and speed checks.
- The supplied `surgsim2.py replay` crashes on this machine (`make_renderer() got an unexpected keyword argument
  'vig'`). I ran the same `replay()` function through a wrapper that drops that argument; the tool files are
  unmodified. `replay/` holds the outputs of that run.
- Side-by-side frames: `source/frames/replay_vs_video.png` (left video, right rollout).

## Method and measurements (all from pixels)
World frame: origin at the centre of the 4x4 post grid on the board's top surface, +x to image right, +y away
from the camera, +z up. `source/fit_and_build_scene.py` is the script that fits the camera and writes the scene.

**Camera.** Least-squares fit of camera position, look-at point, focal length, grid pitch and post height to:
the four base points of the left post column, the base of the front-right post, the top of the front-left post,
and the 67 px width of the yellow sleeve (assumed 20 mm, see below). The grid was assumed square. Residual 2.4 px RMS.
- focal length 628 px, vertical FOV 62.9 deg; position (-0.015, -0.227, 0.085), look-at (0.027, 0.205, 0).
- The strong perspective is supported independently by the visible lean of the left-hand posts.
- A first fit with a fixed long focal length (1200 px) needed a non-square grid (38 x 56 mm) and was rejected.

**Metric scale.** Shaft diameter (8.4 mm) against sleeve width in frames where they are at similar depth:
grasp (frame 75: collar about 26 px, sleeve 67 px) and handover (frame 180: L shaft about 30 px, R shaft about 22 px,
sleeve about 62 px between them). These give a sleeve outer diameter of 17-24 mm; I used 20 mm.

| quantity | value | basis |
|---|---|---|
| sleeve outer diameter | 20 mm | shaft-width ratio above |
| sleeve inner diameter | 13 mm | hole is about 60-65 % of the outer ellipse on the purple sleeve |
| sleeve height | 25 mm | about 80 px of visible side |
| sleeve mass | 0.84 g | foam density 170 kg/m3 (assumed) |
| post diameter / height | 6 mm / 36 mm | 20-23 px wide; height from the fit |
| grid pitch | 37.8 mm | fit |
| board | 178 mm square, 5 mm thick | front corners back-projected; back edge assumed (square) |

**Layout.** 4 columns: left column 4 bare posts, two middle columns with posts in the 2nd and 3rd rows, right column
4 posts. Source = front-right post (`post_source`), target = front-left post (`post_target`). The purple, pink
and rear yellow sleeves are drawn as visual-only cylinders.

**Trocars.** Shaft lines from several frames were intersected in the image: L at about (-290, 210) px, R at about
(1155, 130) px (three R lines agree within about 10 px). Depth along those rays came from shaft-width change:
L_rcm = (-0.156, -0.089, 0.094), R_rcm = (0.237, -0.013, 0.131).

## Behaviour (`policy.py`)
`plan()` reads the sleeve pose, both post positions/heights and the trocar positions from the model/data and builds
key-framed tool-tip tracks, timed to the video (grasp about 5 s, handover 10-14 s, placement about 19 s):
1. R moves from its rest pose to the far side of the sleeve rim, pinches the wall (one blade in the hole, one
   outside), lifts the sleeve clear of the posts and carries it to mid-air between the two posts.
2. L pinches the opposite side of the rim from above; R opens and withdraws.
3. L carries the sleeve over the target post, lowers it until the blade tips are 5 mm above the post top, releases;
   the sleeve slides down the post. Both arms return to the rest poses seen in the video.

Grasp detail that made it work: the tool is rotated by the hold half-angle so the outer blade lies flat on the
outer wall, and the jaw command stays just below servo saturation. With a symmetric, saturated pinch the sleeve
swung about 17 deg in the jaws and the handover missed.

## Differences from the video / not observable
- **Handover orientation.** In the video R tips the sleeve to near-horizontal and L takes the other end. In my
  rollout the sleeve hangs upright and L takes the opposite side of the rim. Position, height and timing of the
  handover follow the video; the sleeve attitude does not.
- **Depth of everything off the board** (trocars, rest poses, handover point) rests on shaft widths of 18-38 px
  with blurred edges; expect errors of a few centimetres along the viewing direction. The R trocar had to be moved
  along its ray from the first estimate (about 33 cm depth) to 22.5 cm so the source post is within the 0.30 m
  insertion range.
- Metric scale is uncertain by roughly ±15 % (sleeve diameter 17-24 mm from the two shaft comparisons).
- The camera fit assumes a square grid; the board's back edge is hidden, so its depth is assumed.
- The posts lean in the image more than a centred, roll-free pinhole predicts on the left side; some roll,
  principal-point offset or lens distortion is present and not modelled.
- Sleeve mass, friction and foam compliance cannot be seen; the sleeve is rigid in the simulation.
- The other sleeves on the board do not collide with anything in the simulation.
