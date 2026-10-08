# Laparoscopic grasper (public instrument contract)

A straight 5 mm laparoscopic grasper inserted through a trocar. Use these files unchanged.

## Including it
```xml
<compiler angle="radian"/>
<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>
<worldbody>
  ...
  <body name="trocar" pos="X Y Z">            <!-- direct child of worldbody, no rotation; pos = RCM -->
    <include file="instrument/instrument_body.xml"/>
  </body>
</worldbody>
<actuator><include file="instrument/instrument_actuators.xml"/></actuator>
<contact><include file="instrument/instrument_contacts.xml"/></contact>
```
The trocar origin is the remote centre of motion (RCM): the abdominal-wall port the shaft always passes through.
Its position is the only instrument parameter you estimate (from the visible shaft lines in the video).

## Kinematics (see `kinematics.py`)
| joint | type | range | meaning |
|---|---|---|---|
| `yaw` | hinge about world +z at the RCM | ±1.57 rad | heading of the shaft |
| `pitch` | hinge about the yawed −x axis | 0 … 1.4 rad | downward tilt of the shaft (0 = horizontal) |
| `insertion` | slide along the shaft | 0 … 0.25 m | depth beyond the RCM |
| `roll` | hinge about the shaft | ±3.14 rad | spins the jaws |
| `jaw_left`, `jaw_right` | slides, symmetric | 0 … 0.007 m | per-jaw opening |

With yaw = pitch = 0 the shaft points along world +y. Shaft direction
`d = (-sin(yaw) cos(pitch), cos(yaw) cos(pitch), -sin(pitch))`, and
`TCP = RCM + (insertion + 0.019) d`. The TCP (site `tcp`) is the grasp point, 3 mm proximal of the jaw tips.
At roll = 0 the jaws open along `(cos(yaw), sin(yaw), 0)`: horizontal, perpendicular to the shaft heading.
`kinematics.ik(rcm, tcp)` returns (yaw, pitch, insertion).

## Geometry (metric references visible in the video)
- shaft: cylinder, diameter 5.0 mm (dark), length 300 mm (it passes through the RCM)
- clevis: diameter 5.4 mm, 6 mm long (light grey) at the shaft tip
- jaws: two boxes 20 mm long × 3.6 mm wide × 2.4 mm thick (light grey); the gap between their inner faces is
  `2 × (0.0001 + opening)`, i.e. 0.2 mm closed, 14.2 mm fully open
- the instrument is gravity-compensated; jaw contacts use friction 1.5 (torsional 0.02)

## Control
Position servos on yaw, pitch, insertion, roll and both jaws. The package's control stream is
`[yaw, pitch, insertion, roll, jaw]` joint targets at dt = 0.05 s; `jaw` drives both jaws (0 = closed,
0.007 = fully open). The jaw servo force is limited to 0.06 N per jaw, so a closed jaw squeezes but does not crush.
Per-step change limits: 0.08 rad (yaw, pitch), 0.006 m (insertion), 0.15 rad (roll).
