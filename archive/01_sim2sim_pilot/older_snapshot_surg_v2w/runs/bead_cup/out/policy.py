"""Pick the bead with the grasper and drop it into the cup. Motion is computed from the scene state."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument'))
import kinematics as K

SLEW = np.array([0.08, 0.08, 0.006]) * 0.6        # stay well inside the per-step limits
INIT_TCP_REL = np.array([0.0447, 0.0280, -0.0409])               # initial TCP relative to the RCM (from video frame 0)
OPEN, CLOSED = K.JAW_OPEN, K.JAW_CLOSED


def _segment(rcm, p0, p1, min_steps):
    """Cartesian straight line p0 -> p1 as joint rows, with enough steps to respect the slew limits."""
    n = max(int(min_steps), 1)
    while True:
        q = np.array([K.ik(rcm, p0 + (p1 - p0) * s) for s in np.linspace(0, 1, n + 1)])
        if np.all(np.abs(np.diff(q, axis=0)) <= SLEW) or n > 400:
            return q[1:]
        n = int(np.ceil(n * 1.3)) + 1


def plan(model, data):
    rcm = np.array(model.body('trocar').pos, float)
    bead = np.array(data.body('bead').xpos, float)
    cup = np.array(data.body('cup').xpos, float)
    bead_h = 2 * float(model.geom('bead_geom').size[1])
    table_z = bead[2] - bead_h / 2

    start = rcm + INIT_TCP_REL
    grasp = np.array([bead[0], bead[1], table_z + bead_h / 2 + 0.001])
    above = grasp + [0, 0, 0.012]
    lift = grasp + [0, 0, 0.026]
    over = np.array([cup[0], cup[1], table_z + 0.026])
    drop = np.array([cup[0], cup[1], table_z + 0.015])

    rows = []

    def move(p0, p1, steps, jaw):
        for q in _segment(rcm, p0, p1, steps):
            rows.append([q[0], q[1], q[2], 0.0, jaw])

    def hold(steps, jaw0, jaw1):
        q = rows[-1][:4]
        for s in np.linspace(0, 1, steps + 1)[1:]:
            rows.append(q + [jaw0 + (jaw1 - jaw0) * s])

    q0 = K.ik(rcm, start)
    rows.append([q0[0], q0[1], q0[2], 0.0, OPEN])
    move(start, above, 55, OPEN)        # reach
    move(above, grasp, 25, OPEN)        # descend around the bead
    hold(5, OPEN, OPEN)
    hold(12, OPEN, CLOSED)              # squeeze
    hold(6, CLOSED, CLOSED)
    move(grasp, lift, 30, CLOSED)       # lift
    move(lift, over, 45, CLOSED)        # carry over the cup
    move(over, drop, 15, CLOSED)        # lower into the mouth
    hold(4, CLOSED, CLOSED)
    hold(8, CLOSED, OPEN)               # release
    hold(10, OPEN, OPEN)
    move(drop, over, 15, OPEN)          # retract
    move(over, start, 50, OPEN)
    return np.array(rows, dtype=np.float64)
