# Reconstruct this laparoscopic video as a MuJoCo simulation
Input: only `{video}` ({W}x{H}, {T} frames at {fps} fps, an endoscope view). Deliver a package at `{out}`.
Time budget: about {minutes} minutes of wall-clock in total, after which the process is killed and whatever is in
`{out}` is evaluated. Get a package that passes `validate` early (within ~15 minutes), then refine it with physical
replay. Keep the package valid at all times while refining.

Task: {task_hint}

## Given
- `{out}/instrument/`: the laparoscopic grasper seen in the video. `README.md` (read it first),
  `instrument_body.xml`, `instrument_actuators.xml`, `instrument_contacts.xml`, `kinematics.py` (closed-form FK/IK).
  Use these files unchanged.
- `{tools}/surgsim.py`: validation and physical replay (below).
- Python: `{py}` has mujoco, numpy, scipy, opencv-python-headless, imageio, pillow. To look at the video, extract
  frames to PNG and open them with the Read tool.

## What to reconstruct
Estimate from the pixels: the endoscope camera (pose and vertical field of view; pinhole, square pixels, principal
point at the image centre, no lens distortion; the darkened corners are optical vignetting), the support surface,
every task-relevant object (geometry, size, pose, plausible mass), the static receiving structure, and where the
instrument enters the body (the trocar / remote centre of motion). There is no depth, calibration or metric marker;
the instrument's known dimensions (instrument/README.md) are your metric reference.

Then write the instrument behaviour that performs the task shown in the video in YOUR scene, under physics.

World frame: metres, z up, gravity (0, 0, -9.81). Choose the origin and the horizontal axes freely. The camera, the
RCM and all bodies are expressed in this one frame.

## scene.xml rules
- `<compiler angle="radian"/>` and `<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>`.
- Instrument exactly as in instrument/README.md: `<body name="trocar" pos="...">` (direct child of worldbody, no
  rotation) containing the body include, plus the actuator and contact includes. No other actuators.
- Manipulated object(s): worldbody children with exactly one `<freejoint/>`, mass 0.1-50 g, geoms that collide.
  They may move only through contact with the instrument and gravity: no mocap, equality constraints (weld/connect),
  tendons, flex, attachments, object actuators or state overrides.
- Receiver/target: a static body (no joints).

## Behaviour
Controls are joint targets `[yaw, pitch, insertion, roll, jaw]` at dt = 0.05 s (format "joint_targets"), see
instrument/README.md. Row 0 is the initial configuration: the instrument is placed there, then the objects settle
for 0.25 s before the episode starts. Targets are linearly interpolated within each step; after the last row the
final targets are held for 1 s. Per-step change limits: 0.08 rad (yaw, pitch), 0.006 m (insertion), 0.15 rad (roll).
Grasping is physical: an object moves only if the closing jaws actually squeeze it between them.

`policy.py` must define `plan(model, data) -> np.ndarray` of shape (T, 5): given your compiled scene
(`mujoco.MjModel`, `mujoco.MjData`) after the settle, read the poses it needs from `data` (e.g.
`data.body("name").xpos`) and return the control stream. It will also be run on displaced layouts of your scene
(the object or the target moved by up to 1 cm), so compute the motion from the state instead of hard-coding it.
`actions.npy` (float64, (T, 5)) must be what `plan` returns on your submitted scene. `policy.py` may import
`instrument/kinematics.py` (add the package's `instrument` directory to `sys.path` relative to `__file__`).

## protocol.json
```jsonc
{{
  "protocol_version": "surg-0.1", "status": "success",
  "task": {{"instruction": "YOUR reading of what happens in the video", "source_video": "source/video.mp4"}},
  "model_path": "scene.xml",
  "camera": {{"pos": [x, y, z], "lookat": [x, y, z], "fovy_deg": f, "width": {W}, "height": {H}}},  // up = +z, no roll
  "instrument": {{"rcm_pos": [x, y, z]}},                                                      // == trocar body pos
  "roles": {{"object": ["<manipulated body>"], "target": ["<receiver body>"]}},
  "actions": {{"path": "actions.npy", "dt": 0.05, "format": "joint_targets"}},
  "policy": {{"path": "policy.py"}}
}}
```
Also deliver `report.md` (method, measurements, what you could not observe); `source/video.mp4` is already there.
Put measurement images in `{out}/source/frames/`. `"status": "infeasible"` with a diagnosis in report.md is a valid
delivery.

## Check your work
- validate: `{py} {tools}/surgsim.py validate {out}` must print `OK`.
- replay: `{py} {tools}/surgsim.py replay {out}` writes `{out}/replay/compare.mp4` and `compare_000..100.png`
  (left: source video, right: your rollout rendered from your declared camera) and `summary.json`.
  `--policy` replays `plan()` instead of `actions.npy`.

Execute the reconstructed scene with your controls and inspect the rendered rollout alongside the source video. Use
the observed interaction and the replay diagnostics to revise the scene and the controls. Package validation alone
does not replace physical execution and visual inspection. A completed replay is not a task judgement; reference
annotations and scores are unavailable.

## Rules
Write only inside `{out}` and `/tmp/{name}/`. The only file about this episode you may read is `{video}`, plus the
supplied instrument and tool files. Do not search for, list or read anything else on this machine, and do not use
the network. Everything metric must come from the pixels.
