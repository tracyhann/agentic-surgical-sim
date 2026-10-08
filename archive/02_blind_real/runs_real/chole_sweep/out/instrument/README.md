# Laparoscopic instrument(s) (public contract)

Instruments in this task: grasper_left, probe_right
Each is a straight grasper on a remote centre of motion (RCM). Use the supplied files unchanged.

## Including instrument `<p>`
```xml
<compiler angle="radian"/>
<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>
<worldbody>
  ...
  <body name="<p>_trocar" pos="X Y Z" euler="0 0 HEADING">   <!-- worldbody child; rotation about z only; pos = RCM -->
    <include file="instrument/<p>_body.xml"/>
  </body>
</worldbody>
<actuator><include file="instrument/<p>_actuators.xml"/></actuator>
<contact><include file="instrument/<p>_contacts.xml"/></contact>
```
Declare the same `rcm_pos` and `heading` in protocol.json. The RCM is the pivot the shaft always passes through
(a trocar port, or the remote centre of a surgical robot arm). Estimate it from where the shaft lines converge.

## Kinematics (see `kinematics.py`, all functions take `heading`)
| joint | type | range | meaning |
|---|---|---|---|
| `<p>_yaw` | hinge about +z at the RCM | ±1.57 rad | heading of the shaft relative to the trocar's +y |
| `<p>_pitch` | hinge about the yawed −x axis | 0 … 1.4 rad | downward tilt of the shaft (0 = horizontal) |
| `<p>_insertion` | slide along the shaft | 0 … 0.25 m | depth beyond the RCM |
| `<p>_roll` | hinge about the shaft | ±3.14 rad | spins the jaws |
| `<p>_jaw_left`, `<p>_jaw_right` | slides, symmetric | 0 … 0.007 m | per-jaw opening |

`TCP = RCM + (insertion + 0.019) d`, d the shaft direction. The TCP (site `<p>_tcp`) is the grasp point, 3 mm
proximal of the jaw tips. At roll = 0 the jaws open horizontally, perpendicular to the shaft heading.
This model has no wrist: real instruments with a wrist are approximated by matching the tool-tip position.

## Geometry
- shaft: cylinder, diameter grasper_left 5 mm, probe_right 5 mm; it passes through the RCM and is 300 mm long
- clevis: 0.4 mm wider than the shaft, 6 mm long, at the shaft tip
- jaws: two boxes 20 mm long × 3.6 mm × 2.4 mm; gap between inner faces = 2 × (0.0001 + opening)
- gravity-compensated; jaw contacts use friction 1.5

## Control
Position servos. The package's control stream has 5 columns per instrument, in the order grasper_left, probe_right:
`[yaw, pitch, insertion, roll, jaw]` joint targets at dt = 0.05 s; `jaw` drives both jaws (0 closed, 0.007 open,
0.06 N per jaw). Per-step change limits: 0.08 rad (yaw, pitch), 0.006 m (insertion), 0.15 rad (roll).
