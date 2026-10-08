"""Closed-form kinematics of the supplied laparoscopic grasper (world frame, metres, radians).

The trocar frame is world-aligned and sits at the RCM `rcm`. With yaw = pitch = 0 the shaft points along +y.
  shaft direction  d(yaw, pitch) = (-sin(yaw) cos(pitch), cos(yaw) cos(pitch), -sin(pitch))
  TCP              = rcm + (insertion + TCP_OFFSET) * d
`roll` spins the jaws about the shaft and does not move the TCP. At roll = 0 the jaws open along
  (cos(yaw), sin(yaw), 0), i.e. horizontally and perpendicular to the shaft's horizontal heading.
"""
import numpy as np

TCP_OFFSET = 0.019        # TCP (grasp point between the distal jaw halves) beyond the shaft origin
JAW_OPEN, JAW_CLOSED = 0.007, 0.0   # per-jaw opening command (m); the jaws touch at ~0.0002 m


def shaft_dir(yaw, pitch):
    return np.array([-np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])


def fk(rcm, yaw, pitch, insertion):
    return np.asarray(rcm, float) + (insertion + TCP_OFFSET) * shaft_dir(yaw, pitch)


def jaw_axis(yaw, pitch, roll):
    """Unit vector along which the jaws open (world frame)."""
    d = shaft_dir(yaw, pitch)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.cross(x, d)
    return np.cos(roll) * x - np.sin(roll) * z


def ik(rcm, tcp):
    """TCP position -> (yaw, pitch, insertion). Raises if the point is unreachable."""
    v = np.asarray(tcp, float) - np.asarray(rcm, float)
    dist = np.linalg.norm(v)
    pitch = np.arcsin(np.clip(-v[2] / dist, -1, 1))
    yaw = np.arctan2(-v[0], v[1])
    insertion = dist - TCP_OFFSET
    if not (0 <= pitch <= 1.4 and -1.57 <= yaw <= 1.57 and 0 <= insertion <= 0.25):
        raise ValueError(f'unreachable TCP {tcp}: yaw={yaw:.3f} pitch={pitch:.3f} insertion={insertion:.4f}')
    return yaw, pitch, insertion
