# Two dVRK-style patient-side manipulators (public instrument contract)

Two 8 mm EndoWrist-style graspers, `L_` (enters from the image left) and `R_` (enters from the image right).
Use the XML files unchanged; only the two trocar positions are yours to estimate.

## Including them
```xml
<compiler angle="radian"/>
<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
<worldbody>
  ...
  <body name="L_trocar" pos="X Y Z"><include file="instrument_psm/L_body.xml"/></body>
  <body name="R_trocar" pos="X Y Z"><include file="instrument_psm/R_body.xml"/></body>
</worldbody>
<actuator><include file="instrument_psm/L_actuators.xml"/><include file="instrument_psm/R_actuators.xml"/></actuator>
<contact><include file="instrument_psm/L_contacts.xml"/><include file="instrument_psm/R_contacts.xml"/></contact>
```
Each `<P>trocar` is a direct child of worldbody with no rotation; its position is the remote centre of motion
(RCM, the port the shaft always passes through), usually far outside the camera view along the shaft line.

## Kinematic chain (per arm)
| joint | type | range | meaning |
|---|---|---|---|
| `yaw` | hinge about world +z at the RCM | ±3.14 | shaft heading (0 = shaft along world +y) |
| `pitch` | hinge about the yawed −x axis | −0.8 … 1.45 | downward tilt of the shaft |
| `insertion` | slide along the shaft | 0 … 0.30 m | depth beyond the RCM |
| `roll` | hinge about the shaft | ±4.5 | spins the wrist |
| `wrist_pitch` | hinge at the shaft tip | ±1.57 | bends the wrist |
| `wrist_yaw` | hinge 9.1 mm distal of `wrist_pitch` | ±1.57 | swings the jaws; also the jaw hinge axis |
| `jaw_a`, `jaw_b` | hinges about the `wrist_yaw` axis | −0.05 … 0.7 | each jaw opens half of the total jaw angle |

## Geometry (metric references visible in the video)
- shaft: black cylinder, diameter 8.4 mm
- wrist: light grey, 9.1 mm from `wrist_pitch` to `wrist_yaw`
- jaws: two light-grey blades 10 mm long, 1.6 mm thick, 2.8 mm tall, hinged at `wrist_yaw`
- the tool tip (site `<P>tcp`) is 10.2 mm beyond the `wrist_yaw` axis, between the jaw tips
- the instrument is gravity-compensated; the jaw servo torque is limited, so closing squeezes but does not crush.
  Jaw contacts use friction 1.5. Grasps are physical: an object moves only if the jaws actually squeeze it.

## Tool-tip frame and control
The `<P>tcp` site frame: +y points from the wrist out along the jaws, +x is the direction the jaws open
along (jaw_a swings towards +x, jaw_b towards −x), +z = x × y is the hinge axis.

Controls are absolute tool-tip poses in the world frame, one row per 0.05 s for both arms:
`[L: x y z qw qx qy qz jaw,  R: x y z qw qx qy qz jaw]` (16 values). `quat` is the orientation of the tcp site
frame (MuJoCo order w x y z, unit norm); `jaw` is the total jaw opening angle in rad (0 = closed, up to 1.2;
−0.1 squeezes harder). Each row is converted to joint targets by damped-least-squares IK (warm-started from the
previous row) and the joint servos track them under physics. Rows must be reachable (IK error ≤ 1 mm / 3°) and
change slowly enough (per-step joint changes ≤ 0.08 rad yaw/pitch, 6 mm insertion, 0.3 rad roll/wrist).

To build a quaternion from the desired jaw-opening axis `x` and jaw-pointing axis `y`:
```python
z = np.cross(x, y); q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).flatten())
```
`surgsim2.joint_targets(model, actions)` exposes the same IK if you want to check reachability yourself.
