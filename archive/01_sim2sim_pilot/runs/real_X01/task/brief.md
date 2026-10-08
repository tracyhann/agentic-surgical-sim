# Reconstruct this real robotic-surgery training video as a MuJoCo simulation
Input: only `/tmp/surgrun_X01/task/video.mp4` (1024x768, 316 frames at 15 fps). Deliver a package at `/tmp/surgrun_X01/out`.
Time budget: about 45 minutes of wall-clock in total, after which the process is killed and whatever is in
`/tmp/surgrun_X01/out` is evaluated. Get a package that passes `validate` early (within ~20 minutes), then refine it with physical
replay. Keep the package valid at all times while refining.

The video is a real recording of a da Vinci Research Kit (two patient-side manipulators with 8 mm wristed
graspers) performing the "post and sleeve" skill task on a peg board, filmed by a fixed external camera. It is a
screen capture: the top 24 pixel rows are a desktop panel and a mouse cursor may be visible; ignore both.

Task: the instrument on the image right picks the yellow sleeve off its post, hands it over in mid-air to the instrument on the image left, which places it over another post.

## Given
- `/tmp/surgrun_X01/out/instrument_psm/`: the two manipulators. `README.md` (read it first), the `L_`/`R_` include files.
  Use these files unchanged. `L_` is the instrument entering from the image left, `R_` from the image right.
- `/tmp/surgrun_X01/task/tools/surgsim2.py` (validate / replay; it imports `surgsim.py` next to it).
- Python: `/tmp/surgrun_X01/.venv/bin/python` has mujoco, numpy, scipy, opencv-python-headless, imageio, pillow. To look at the video, extract
  frames to PNG and open them with the Read tool.

## What to reconstruct
Estimate from the pixels: the camera (pose and vertical field of view; pinhole, square pixels, principal point at
the image centre, no distortion), the board, the posts, the manipulated sleeve (a hollow foam cylinder: outer and
inner radius, height, plausible mass), the post it starts on and the post it ends on, and where each instrument
enters (its trocar / remote centre of motion; the shaft lines in the image all pass through it). There is no depth,
calibration or metric marker; the instruments' known dimensions (instrument_psm/README.md) are your metric reference.

Then write the behaviour of BOTH instruments that performs the task shown in the video in YOUR scene, under physics.

World frame: metres, z up, gravity (0, 0, -9.81). Choose the origin and the horizontal axes freely. The camera, the
trocars and all bodies are expressed in this one frame.

## scene.xml rules
- `<compiler angle="radian"/>` and exactly
  `<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>`.
- Both instruments exactly as in instrument_psm/README.md. No other actuators.
- The sleeve: a worldbody child with exactly one `<freejoint/>`, mass 0.1-50 g, colliding geoms (a hollow cylinder
  can be built from boxes arranged around the hole). It may move only through contact with the jaws and gravity:
  no mocap, equality constraints, tendons, flex, attachments, object actuators or state overrides.
- Posts and board: static bodies. The post the sleeve starts on and the post it ends on must each be their own
  body (no joints).

## Behaviour
Controls are tool-tip poses for both arms at dt = 0.05 s (format "tcp_pose_2arm"), see instrument_psm/README.md.
Row 0 is the initial configuration: the instruments are placed there, then the objects settle for 0.25 s. Targets
are interpolated within each step; after the last row the final targets are held for 1 s.

`policy.py` must define `plan(model, data) -> np.ndarray` of shape (T, 16): given your compiled scene
(`mujoco.MjModel`, `mujoco.MjData`) after the settle, read the poses it needs from `data` (e.g.
`data.body("sleeve").xipos`) and return the control stream. It will also be run on displaced layouts of your scene
(the target post moved by up to 1 cm), so compute the motion from the state instead of hard-coding it.
`actions.npy` (float64, (T, 16)) must be what `plan` returns on your submitted scene.

## protocol.json
```jsonc
{
  "protocol_version": "surg-0.2", "status": "success",
  "task": {"instruction": "YOUR reading of what happens in the video", "source_video": "source/video.mp4"},
  "model_path": "scene.xml",
  "camera": {"pos": [x, y, z], "lookat": [x, y, z], "fovy_deg": f, "width": 1024, "height": 768},  // up = +z, no roll
  "instrument": {"L_rcm_pos": [x, y, z], "R_rcm_pos": [x, y, z]},           // == the trocar body positions
  "roles": {"object": ["<sleeve body>"], "source": ["<post it starts on>"], "target": ["<post it ends on>"]},
  "actions": {"path": "actions.npy", "dt": 0.05, "format": "tcp_pose_2arm"},
  "policy": {"path": "policy.py"}
}
```
Also deliver `report.md` (method, measurements, what you could not observe); `source/video.mp4` is already there.
Put measurement images in `/tmp/surgrun_X01/out/source/frames/`. `"status": "infeasible"` with a diagnosis in report.md is a valid
delivery.

## Check your work
- validate: `/tmp/surgrun_X01/.venv/bin/python /tmp/surgrun_X01/task/tools/surgsim2.py validate /tmp/surgrun_X01/out` must print `OK`.
- replay: `/tmp/surgrun_X01/.venv/bin/python /tmp/surgrun_X01/task/tools/surgsim2.py replay /tmp/surgrun_X01/out` writes `/tmp/surgrun_X01/out/replay/compare.mp4` and `compare_000..100.png`
  (left: source video, right: your rollout rendered from your declared camera) and `summary.json` (including how
  long each arm held the sleeve). `--policy` replays `plan()` instead of `actions.npy`.

Execute the reconstructed scene with your controls and inspect the rendered rollout alongside the source video. Use
the observed interaction and the replay diagnostics to revise the scene and the controls. Package validation alone
does not replace physical execution and visual inspection. A completed replay is not a task judgement; reference
annotations and scores are unavailable.

## Rules
Write only inside `/tmp/surgrun_X01/out` and `/tmp/surg_real_X01/`. The only file about this episode you may read is `/tmp/surgrun_X01/task/video.mp4`, plus the
supplied instrument and tool files. Do not search for, list or read anything else on this machine, and do not use
the network. Everything metric must come from the pixels.
