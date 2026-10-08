"""Scripted lift -> handoff -> place, reading the sleeve and post positions from the simulator state."""
import numpy as np, mujoco

R_MID = 0.00675          # sleeve wall mid-radius
TIP_BELOW_RIM = 0.006    # how far the jaw tips reach below the rim
DT = 0.05


def quat(x_axis, y_axis):
    x = np.asarray(x_axis, float); y = np.asarray(y_axis, float)
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).flatten())
    return q


def plan(model, data):
    c = data.body('sleeve').xipos.copy()
    rim = c[2] + 0.018 / 2
    dst = data.body('post_dst').xpos.copy()
    post_top = 0.025
    down = [0, 0, -1]
    qR, qL = quat([1, 0, 0], down), quat([-1, 0, 0], down)
    grip_z = rim - TIP_BELOW_RIM
    lift = post_top + 0.018 - TIP_BELOW_RIM + 0.008          # sleeve bottom clears the post top
    H = np.array([0.0, 0.0, lift + 0.01])                  # hand-off point (sleeve centre)
    L0, R0 = np.array([-0.06, 0.01, 0.07]), np.array([0.07, 0.01, 0.06])
    OPEN, SHUT = 0.6, -0.1
    keys = []                                               # (duration s, L pos, L jaw, R pos, R jaw)
    gR = c + [R_MID, 0, 0]; gR[2] = grip_z
    keys += [(0.6, L0, OPEN, R0, OPEN),
             (2.0, L0, OPEN, gR + [0, 0, 0.02], OPEN), (1.0, L0, OPEN, gR, OPEN), (0.6, L0, OPEN, gR, SHUT),
             (0.4, L0, OPEN, gR, SHUT), (1.5, L0, OPEN, [gR[0], gR[1], lift], SHUT)]
    hR = H + [R_MID, 0, -0.01 - (0.018 / 2 - TIP_BELOW_RIM) + 0.018 / 2]
    hR = np.array([H[0] + R_MID, H[1], H[2] + 0.018 / 2 - TIP_BELOW_RIM])
    hL = np.array([H[0] - R_MID, H[1], hR[2]])
    keys += [(2.0, L0, OPEN, hR, SHUT), (2.0, hL + [0, 0, 0.02], OPEN, hR, SHUT), (1.0, hL, OPEN, hR, SHUT),
             (0.6, hL, SHUT, hR, SHUT), (0.4, hL, SHUT, hR, SHUT), (0.6, hL, SHUT, hR, OPEN),
             (1.0, hL, SHUT, hR + [0.01, 0, 0.02], OPEN)]
    pL = np.array([dst[0] - R_MID, dst[1], hL[2]])
    downL = np.array([pL[0], pL[1], 0.002 + 0.018 - TIP_BELOW_RIM])
    keys += [(2.5, pL, SHUT, R0, OPEN), (2.0, downL, SHUT, R0, OPEN), (0.4, downL, SHUT, R0, OPEN),
             (0.6, downL, OPEN, R0, OPEN), (1.5, downL + [0, 0, 0.03], OPEN, R0, OPEN), (1.0, L0, OPEN, R0, OPEN)]
    rows, prev = [], None
    for dur, Lp, Lj, Rp, Rj in keys:
        cur = (np.asarray(Lp, float), Lj, np.asarray(Rp, float), Rj)
        if prev is None:
            prev = cur
        n = max(1, int(round(dur / DT)))
        for k in range(1, n + 1):
            w = k / n
            Lx = prev[0] + (cur[0] - prev[0]) * w; Rx = prev[2] + (cur[2] - prev[2]) * w
            Lw = prev[1] + (cur[1] - prev[1]) * w; Rw = prev[3] + (cur[3] - prev[3]) * w
            rows.append(np.concatenate([Lx, qL, [Lw], Rx, qR, [Rw]]))
        prev = cur
    return np.array(rows)
