"""Reference (scripted) policy: reads the initial object/target poses from the simulator state."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'instrument'))
from kinematics import ik, jaw_axis, JAW_OPEN

RCM = np.array([0.075, -0.07, 0.085])
TASK = 'bead_cup'
HOVER = np.array([0.035, -0.012, 0.032])
DT = 0.05


def _seg(p0, p1, speed):
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / (speed * DT))))
    return [p0 + (p1 - p0) * (k + 1) / n for k in range(n)]


def plan(model, data):
    obj = data.body('bead').xipos.copy()
    tgt = data.body('cup').xpos.copy()
    if TASK == 'bead_cup':
        grasp = obj + np.array([0, 0, 0.0008])
        place = np.array([tgt[0], tgt[1], 0.016])
    else:
        yaw = ik(RCM, obj)[0]
        u = jaw_axis(yaw, 0.0, 0.0)
        grasp = np.array([obj[0], obj[1], 0.0040]) + 0.009 * u
        place = np.array([tgt[0], tgt[1], 0.0105])
        for _ in range(5):                      # the ring centre sits 9 mm from the TCP along the jaw axis
            yaw = ik(RCM, place)[0]
            place[:2] = tgt[:2] + 0.009 * jaw_axis(yaw, 0.0, 0.0)[:2]
    lift_z = 0.030 if TASK != 'bead_cup' else 0.026
    OPEN = JAW_OPEN if TASK == 'bead_cup' else 0.003
    tcp, jaw = [HOVER.copy()], [OPEN]
    def go(p, speed, j):
        for q in _seg(tcp[-1], np.asarray(p, float), speed):
            tcp.append(q); jaw.append(j)
    def hold(n, j0, j1):
        for k in range(n):
            tcp.append(tcp[-1].copy()); jaw.append(j0 + (j1 - j0) * (k + 1) / n)
    hold(6, OPEN, OPEN)
    go(grasp + [0, 0, 0.014], 0.025, OPEN)
    go(grasp, 0.010, OPEN)
    hold(10, OPEN, 0.0)
    hold(6, 0.0, 0.0)
    go([grasp[0], grasp[1], lift_z], 0.012, 0.0)
    go([place[0], place[1], lift_z], 0.022, 0.0)
    go(place, 0.008, 0.0)
    hold(4, 0.0, 0.0)
    hold(8, 0.0, OPEN)
    go(place + [0, 0, 0.015], 0.015, OPEN)
    go(HOVER, 0.025, OPEN)
    out = []
    for p, j in zip(tcp, jaw):
        y, pi, ins = ik(RCM, p)
        out.append([y, pi, ins, 0.0, j])
    return np.array(out)
