# Reconstruct this real surgical video as a MuJoCo simulation
Input: only `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/task/video.mp4` (640x360, 151 frames at 25 fps). This is a REAL recording: laparoscopic cholecystectomy (gallbladder removal) in a patient; the laparoscope is static in this clip.
Deliver a package at `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out`.
Time budget: about 45 minutes of wall-clock in total, after which the process is killed and whatever is in
`/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out` is evaluated. Get a package that passes `validate` early (within ~15 minutes), then refine it with physical
replay. Keep the package valid at all times while refining.

Task: a grasper from the upper left holds the gallbladder neck while a straight instrument from the upper right sweeps the tissue beside it. Reproduce both instrument motions and the response of the gallbladder/tissue they touch. `grasper_left` is the upper-left instrument, `probe_right` the straight upper-right one (keep its jaws closed if it acts as a probe).

## Given
- `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out/instrument/`: the instrument models for this task: grasper_left, probe_right. `README.md` (read it first), the
  include files `<name>_body.xml`, `<name>_actuators.xml`, `<name>_contacts.xml`, and `kinematics.py`.
  Use these files unchanged. They are simplified straight graspers on a remote centre of motion; real instruments
  may look different (wrists, other jaw shapes). Match the motion of the tool tips, not the instrument appearance.
- `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/task/tools/surgsim.py`: validation and physical replay (below).
- Python: `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/.venv/bin/python` has mujoco, numpy, scipy, opencv-python-headless, imageio, pillow. To look at the video, extract
  frames to PNG and open them with the Read tool.

## What to reconstruct
Estimate from the pixels: the camera (it is static during this clip; pinhole, square pixels, principal point at the
image centre, assume no lens distortion; estimate pose and vertical field of view), the support surfaces, the
task-relevant objects or tissue (geometry, size, pose, plausible mass and stiffness), the static structures they
interact with, and the RCM of every instrument you use. There is no depth, calibration or metric marker; the
instrument shaft diameter (grasper_left 5 mm, probe_right 5 mm) is your metric reference. Appearance (texture, lighting) does not have to
match; geometry and motion do.

Then write the instrument behaviour that reproduces what happens in the video in YOUR scene, under physics, with
roughly the video's timing (the clip lasts 6.0 s).

World frame: metres, z up, gravity (0, 0, -9.81). Choose the origin and horizontal axes freely. The camera, the RCMs
and all bodies are expressed in this one frame.

## scene.xml rules (soft profile)
- `<compiler angle="radian"/>` and `<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>`
  (keep the default friction cone and impratio).
- Every instrument exactly as in instrument/README.md: `<body name="<name>_trocar" pos="..." euler="0 0 heading">`
  (worldbody child, rotation about z only) containing the body include, plus the actuator and contact includes.
  All listed instruments must be present (park an unused one out of view). No other actuators, no mocap bodies.
- Tissue may be modelled with flex/flexcomp, composite, or rigid bodies joined by joints/tendons. It may be
  attached to static anatomy (worldbody or static bodies) with equality constraints, tendons or flex pins.
- Nothing may couple tissue to an instrument except contact: no equality constraint or tendon may touch an
  instrument body, and there are no object actuators or state overrides.
- `roles.object` names the tissue that the instruments manipulate (a body or a flex name).

## Behaviour
Controls are joint targets, 5 columns per instrument in the order grasper_left, probe_right:
`[yaw, pitch, insertion, roll, jaw]` at dt = 0.05 s (format "joint_targets"). Row 0 is the initial configuration:
the instruments are placed there, then the scene settles for 0.25 s before the episode starts. Targets are linearly
interpolated within each step; after the last row the final targets are held for 1 s. Per-step change limits:
0.08 rad (yaw, pitch), 0.006 m (insertion), 0.15 rad (roll). Contact is physical: things move only if the
instruments actually push or squeeze them.

`policy.py` must define `plan(model, data) -> np.ndarray` (T, 10): given your compiled scene after the settle,
read the poses it needs from `data` and return the control stream. It will also be run on displaced layouts of your
scene (task objects moved by up to 1 cm), so compute the motion from the state where you can. `actions.npy`
(float64) must be what `plan` returns on your submitted scene.

## protocol.json
```jsonc
{
  "protocol_version": "surg-0.2", "status": "success",
  "task": {"instruction": "YOUR reading of what happens in the video", "source_video": "source/video.mp4"},
  "model_path": "scene.xml",
  "camera": {"pos": [x, y, z], "lookat": [x, y, z], "fovy_deg": f, "width": 640, "height": 360},  // up = +z, no roll
  "instruments": [{"name": "grasper_left", "rcm_pos": [x, y, z], "heading": h}, {"name": "probe_right", "rcm_pos": [x, y, z], "heading": h}],          // same order; rcm_pos == trocar pos, heading == trocar euler z
  "roles": {"object": ["<the tissue body or flex>"], "target": []},
  "actions": {"path": "actions.npy", "dt": 0.05, "format": "joint_targets"},
  "policy": {"path": "policy.py"}
}
```
Also deliver `report.md` (method, measurements, what you could not observe); `source/video.mp4` is already there.
Put measurement images in `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out/source/frames/`. `"status": "infeasible"` with a diagnosis in report.md is a valid
delivery.

## Check your work
- validate: `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/.venv/bin/python /private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/task/tools/surgsim.py validate /private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out` must print `OK`.
- replay: `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/.venv/bin/python /private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/task/tools/surgsim.py replay /private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out` writes `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out/replay/compare.mp4` and `compare_000..100.png`
  (left: source video, right: your rollout rendered from your declared camera, time-normalised) and `summary.json`.
  `--policy` replays `plan()` instead of `actions.npy`.

Execute the reconstructed scene with your controls and inspect the rendered rollout alongside the source video. Use
the observed interaction and the replay diagnostics to revise the scene and the controls. Package validation alone
does not replace physical execution and visual inspection. A completed replay is not a judgement; reference
annotations and scores are unavailable.

## Rules
Write only inside `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/out` and `/tmp/surgreal_chole_sweep/`. The only file about this episode you may read is `/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind/chole_sweep/task/video.mp4`, plus the
supplied instrument and tool files. Do not search for, list or read anything else on this machine, and do not use
the network. Everything metric must come from the pixels.
