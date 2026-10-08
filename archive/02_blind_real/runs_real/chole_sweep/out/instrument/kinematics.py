"""Closed-form kinematics of a supplied laparoscopic instrument (world frame, metres, radians).

The trocar body sits at the RCM `rcm` and may be rotated about world z by `heading` (its euler="0 0 heading").
In the trocar frame, with yaw = pitch = 0 the shaft points along +y.
  shaft direction (trocar frame)  d = (-sin(yaw) cos(pitch), cos(yaw) cos(pitch), -sin(pitch)), then rotated by heading
  TCP = rcm + (insertion + TCP_OFFSET) * d
`roll` spins the jaws about the shaft and does not move the TCP. At roll = 0 the jaws open horizontally,
perpendicular to the shaft's horizontal heading.
"""
import numpy as np

TCP_OFFSET = 0.019        # TCP (grasp point between the distal jaw halves) beyond the shaft origin
JAW_OPEN, JAW_CLOSED = 0.007, 0.0   # per-jaw opening command (m); the jaws touch at ~0.0002 m


def _rz(h):
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shaft_dir(yaw, pitch, heading=0.0):
    return _rz(heading) @ np.array([-np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])


def fk(rcm, yaw, pitch, insertion, heading=0.0):
    return np.asarray(rcm, float) + (insertion + TCP_OFFSET) * shaft_dir(yaw, pitch, heading)


def jaw_axis(yaw, pitch, roll, heading=0.0):
    """Unit vector along which the jaws open (world frame)."""
    d = shaft_dir(yaw, pitch, heading)
    x = _rz(heading) @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.cos(roll) * x - np.sin(roll) * np.cross(x, d)


def ik(rcm, tcp, heading=0.0):
    """TCP position -> (yaw, pitch, insertion). Raises if the point is unreachable."""
    v = _rz(-heading) @ (np.asarray(tcp, float) - np.asarray(rcm, float))
    dist = np.linalg.norm(v)
    pitch = np.arcsin(np.clip(-v[2] / dist, -1, 1))
    yaw = np.arctan2(-v[0], v[1])
    insertion = dist - TCP_OFFSET
    if not (0 <= pitch <= 1.4 and -1.57 <= yaw <= 1.57 and 0 <= insertion <= 0.25):
        raise ValueError(f'unreachable TCP {tcp}: yaw={yaw:.3f} pitch={pitch:.3f} insertion={insertion:.4f}')
    return yaw, pitch, insertion
