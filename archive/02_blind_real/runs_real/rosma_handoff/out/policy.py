"""Sleeve hand-off policy: closed-loop planning by rolling the scene forward inside plan().

psm_right pinches the wall of the sleeve at its top rim (one jaw in the bore, one outside), lifts it off its post,
rolls 180 deg while carrying (the instrument has no wrist: the roll swings the sleeve from hanging to roughly
horizontal, as the wrist does in the video); psm_left pinches the wall at the free end, psm_right lets go, psm_left
rolls 180 deg (sleeve swings to hanging), carries it over the destination post, lowers it part-way and drops it.
Every waypoint is computed from the simulated state.
"""
import copy
import numpy as np
import mujoco
from instrument.kinematics import fk, ik, shaft_dir, _rz

NAMES = ['psm_left', 'psm_right']
ARM = ['yaw', 'pitch', 'insertion', 'roll']
SLEW = np.tile([0.08, 0.08, 0.006, 0.15, 1.0], 2)
DT = 0.05
OBJ, SRC_HINT, DST = 'sleeve_yellow', None, 'post_L1'
SH, WALL_R, POST_H = 0.030, 0.0085, 0.043      # sleeve length, wall mid radius, post height
OPEN, HOLD = 0.0045, 0.0
T_TOTAL = 342


class Roll:
    def __init__(self, model, data):
        self.m, self.d = model, copy.copy(data)
        self.nsub = int(round(DT / model.opt.timestep))
        B = mujoco.mjtObj.mjOBJ_BODY
        self.rcm, self.head = {}, {}
        for p in NAMES:
            t = mujoco.mj_name2id(model, B, p + '_trocar')
            self.rcm[p] = model.body_pos[t].copy()
            q = model.body_quat[t]
            self.head[p] = 2 * np.arctan2(q[3], q[0])
        A = mujoco.mjtObj.mjOBJ_ACTUATOR
        self.aid = {(p, j): mujoco.mj_name2id(model, A, f'{p}_{j}') for p in NAMES for j in ARM + ['jaw_left', 'jaw_right']}
        row = []
        for p in NAMES:
            row += [self.d.ctrl[self.aid[p, j]] for j in ARM] + [self.d.ctrl[self.aid[p, 'jaw_left']]]
        self.rows = [np.array(row, float)]
        self.obj = mujoco.mj_name2id(model, B, OBJ)

    def _apply(self, a):
        for k, p in enumerate(NAMES):
            for i, j in enumerate(ARM):
                self.d.ctrl[self.aid[p, j]] = a[5 * k + i]
            self.d.ctrl[self.aid[p, 'jaw_left']] = self.d.ctrl[self.aid[p, 'jaw_right']] = a[5 * k + 4]

    def push(self, row):
        a = self.rows[-1]
        for s in range(self.nsub):
            self._apply(a + (row - a) * (s + 1) / self.nsub)
            mujoco.mj_step(self.m, self.d)
        self.rows.append(np.array(row, float))

    def tcp_cmd(self, p):
        k = NAMES.index(p)
        r = self.rows[-1][5 * k:5 * k + 5]
        return fk(self.rcm[p], r[0], r[1], r[2], self.head[p])

    def move(self, n, roll={}, **tg):
        """tg: psm_left=(tcp or None, jaw or None), ... ; straight TCP lines, n rows (more if slew-limited)."""
        start = self.rows[-1].copy()
        p0 = {p: self.tcp_cmd(p) for p in tg}
        while True:
            seq = []
            for i in range(1, n + 1):
                s = i / n
                s = s * s * (3 - 2 * s)
                row = start.copy()
                for p, (tcp, jaw) in tg.items():
                    k = NAMES.index(p)
                    if tcp is not None:
                        row[5 * k:5 * k + 3] = ik(self.rcm[p], p0[p] + s * (np.asarray(tcp) - p0[p]), self.head[p])
                    if p in roll:
                        row[5 * k + 3] = start[5 * k + 3] + (i / n) * (roll[p] - start[5 * k + 3])
                    if jaw is not None:
                        row[5 * k + 4] = start[5 * k + 4] + (i / n) * (jaw - start[5 * k + 4])
                seq.append(row)
            dd = np.abs(np.diff(np.array([start] + seq), axis=0))
            if np.all(dd <= SLEW * 0.95) or n > 400:
                break
            n += 2
        for r in seq:
            self.push(r)

    def wait(self, n):
        for _ in range(max(n, 0)):
            self.push(self.rows[-1].copy())

    def until(self, k):
        self.wait(k - (len(self.rows) - 1))

    # ---- state readers
    def sleeve(self):
        c = self.d.xpos[self.obj].copy()
        R = self.d.xmat[self.obj].reshape(3, 3).copy()
        ax = R[:, 2] * (1 if R[2, 2] >= 0 else -1)     # axis, pointing to the upper end
        return c, R, ax

    def frame(self, p, tcp):
        yaw, pitch, _ = ik(self.rcm[p], tcp, self.head[p])
        d = shaft_dir(yaw, pitch, self.head[p])
        x = _rz(self.head[p]) @ np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return np.stack([x, d, np.cross(x, d)], axis=1)

    def carried(self, p, v, tcp_old, tcp_new, flip):
        """World vector v rigidly attached to the jaws of p, after moving the TCP (and rolling by pi if flip)."""
        loc = self.frame(p, tcp_old).T @ v
        if flip:
            loc = loc * np.array([-1, 1, -1])
        return self.frame(p, tcp_new) @ loc

    def wall_grasp(self, want, depth=0.004, shift=0.0008, away_from=None):
        """Grasp point on the sleeve wall whose outward normal is closest to `want`, `depth` inside the upper rim
        (or inside the end farthest from `away_from`)."""
        c, R, ax = self.sleeve()
        if away_from is not None and np.linalg.norm(c + ax * SH / 2 - away_from) < np.linalg.norm(c - ax * SH / 2 - away_from):
            ax = -ax
        best = max((np.array([np.cos(i * np.pi / 4), np.sin(i * np.pi / 4), 0]) for i in range(8)),
                   key=lambda v: (R @ v) @ want)
        nrm = R @ best
        return c + (WALL_R + shift) * nrm + (SH / 2 - depth) * ax, nrm


def plan(model, data):
    r = Roll(model, data)
    try:
        _plan(r, model, data)
    except ValueError:        # a waypoint became unreachable (e.g. the sleeve was lost): hold the last targets
        pass
    r.until(T_TOTAL - 1)
    return np.array(r.rows)[:T_TOTAL]


def _plan(r, model, data):
    B = mujoco.mjtObj.mjOBJ_BODY
    board = mujoco.mj_name2id(model, B, 'board')
    up = data.xmat[board].reshape(3, 3)[:, 2].copy()
    dst = mujoco.mj_name2id(model, B, DST)
    dst_top = data.xpos[dst] + POST_H * up
    L, Rt = 'psm_left', 'psm_right'
    DEPTH = 0.004

    def axis(p, g):
        v = g - r.rcm[p]
        return v / np.linalg.norm(v)

    # --- psm_right: straddle the near wall of the sleeve from above, along the shaft
    g, _ = r.wall_grasp(np.array([0, -1.0, 0]), DEPTH)
    pre = g - 0.024 * axis(Rt, g)
    r.move(8, psm_right=(pre, OPEN))
    r.until(30)
    r.move(20, psm_right=(g, None))
    r.move(10, psm_right=(None, HOLD))
    r.until(85)
    # --- lift along the post
    r.move(22, psm_right=(r.tcp_cmd(Rt) + 0.052 * up, None))
    c, _, ax = r.sleeve()
    t0 = r.tcp_cmd(Rt)
    # --- choose the hand-off height: after psm_right rolls by pi the sleeve must lie so that psm_left, entering the
    #     free end along its shaft, holds it at the angle that makes it parallel to the post at the placement pose
    place_tcp = dst_top + (0.006 + SH - DEPTH) * up + np.array([0, WALL_R, 0])
    beta_p = np.arccos(np.clip(axis(L, place_tcp) @ (-up), -1, 1))
    yh = r.rcm[L][1] + WALL_R
    best = None
    for z in np.linspace(0.06, 0.16, 41):
        hc = np.array([0.002, yh, z])
        t = hc.copy()
        try:
            for _ in range(6):
                t = hc - r.carried(Rt, c - t0, t0, t, True)
            a2 = r.carried(Rt, ax, t0, t, True)             # points to the end held by psm_right
            gl = hc - (SH / 2 - DEPTH) * a2
            gl[1] = r.rcm[L][1]
            ik(r.rcm[L], gl, r.head[L])
        except ValueError:
            continue
        err = abs(np.arccos(np.clip(axis(L, gl) @ a2, -1, 1)) - beta_p)
        if best is None or err < best[0]:
            best = (err, hc, t)
    _, hand, t = best
    k = NAMES.index(Rt)
    # sweep the sleeve through the camera side (-y), away from the other posts
    sgn = 1.0
    r.move(30, roll={Rt: r.rows[-1][5 * k + 3] + sgn * 3.14}, psm_right=(t, None))
    r.wait(2)
    c, _, _ = r.sleeve()
    r.move(6, psm_right=(r.tcp_cmd(Rt) + hand - c, None))
    r.wait(2)
    # --- psm_left: straddle the wall at the free end
    want = np.array([0, -1.0, 0])
    def left_grasp():
        # enter the bore along the sleeve axis; centre the inclined jaw in the opening
        g, nrm = r.wall_grasp(want, DEPTH, away_from=r.tcp_cmd(Rt))
        c, _, _ = r.sleeve()
        e = g - c
        e = e - (e @ nrm) * nrm
        a_in = -e / np.linalg.norm(e)                      # sleeve axis, pointing into the free end
        d = axis(L, g)
        w = d - (d @ a_in) * a_in
        sb = np.linalg.norm(w)
        w /= sb
        g = g - w * (0.003 * sb - DEPTH * sb / max(d @ a_in, 0.3)) / 2
        return g, a_in
    g, a_in = left_grasp()
    r.move(14, psm_left=(g - 0.024 * a_in, OPEN))
    g, a_in = left_grasp()
    r.move(16, psm_left=(g, None))
    r.move(10, psm_left=(None, HOLD))
    r.until(232)
    r.move(8, psm_right=(None, OPEN))
    r.wait(2)
    tr = r.tcp_cmd(Rt)
    r.move(10, psm_right=(tr - 0.03 * axis(Rt, tr), None))
    # --- psm_left rolls by pi (sleeve swings to hanging) while moving towards the destination post
    kl = NAMES.index(L)
    mid = place_tcp + 0.02 * up
    r.move(30, roll={L: r.rows[-1][5 * kl + 3] + 3.14}, psm_left=(mid, None),
           psm_right=(tr - 0.03 * axis(Rt, tr) + np.array([0.02, 0, 0.0]), None))
    r.wait(2)
    for h, n in ((0.008, 10), (0.006, 6)):
        c, _, ax = r.sleeve()
        bottom = c - SH / 2 * ax
        r.move(n, psm_left=(r.tcp_cmd(L) + dst_top + h * up - bottom, None))
    c, _, ax = r.sleeve()
    bottom = c - SH / 2 * ax
    r.move(12, psm_left=(r.tcp_cmd(L) + dst_top - 0.013 * up - bottom, None))
    r.until(316)
    r.move(6, psm_left=(None, OPEN))
    r.wait(3)
    tl = r.tcp_cmd(L)
    r.move(14, psm_left=(tl - 0.035 * axis(L, tl), None))
