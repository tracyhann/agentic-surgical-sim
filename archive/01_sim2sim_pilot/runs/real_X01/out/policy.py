"""Open-loop two-arm plan for the post-and-sleeve handover, computed from the settled scene state."""
import numpy as np
import mujoco

DT = 0.05
R_OUT = 0.010        # sleeve outer radius
H_SLEEVE = 0.025
HINGE = 0.0075       # distance of the jaw hinge above the rim, measured along the jaws
TH = 0.2235          # half jaw angle at which the wall (3.5 mm) is pinched at that hinge height
OPEN, HOLD = 0.54, 2 * (TH - 0.04)
TILT = 0.35          # jaws lean this far from vertical, away from the arm's trocar
TCP = 0.0102         # hinge -> tcp
L_REST0, L_REST1 = (-0.0272, -0.0683, 0.0815), (-0.0401, -0.0679, 0.0773)
R_REST0, R_REST1 = (0.1019, 0.0250, 0.0731), (0.1111, 0.0131, 0.0697)


def _pose(rim, az, tsign):
    """Pose for pinching the wall at the outer rim point `rim` (radial direction az).

    The tool is rotated by TH about the hinge axis so that the outer blade lies flat on the outer wall while the
    inner blade meets the inner rim edge: the hanging sleeve then rests against the flat blade and stays upright.
    """
    xb = np.array([np.cos(az), np.sin(az), 0.0]); t = tsign * np.array([-np.sin(az), np.cos(az), 0.0])
    yb = -np.cos(TILT) * np.array([0, 0, 1.0]) + np.sin(TILT) * t
    x = np.cos(TH) * xb + np.sin(TH) * yb; y = np.cos(TH) * yb - np.sin(TH) * xb; z = np.cross(x, y)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).flatten())
    return np.asarray(rim) + 0.0002 * xb - HINGE * yb + TCP * y, q


def _az(rcm, p, side, ref=None, snap=None):
    """Azimuth of the rim point whose radial direction is perpendicular to the shaft heading (side = +1: left of it)."""
    d = np.arctan2(p[1] - rcm[1], p[0] - rcm[0]) + side * np.pi / 2
    if ref is not None:
        d = ref + (d - ref + np.pi) % (2 * np.pi) - np.pi
    if snap is not None:
        d = snap[0] + np.round((d - snap[0]) / snap[1]) * snap[1]
    return d


def _smooth(s):
    return s * s * (3 - 2 * s)


class _Track:
    """Piecewise key-framed track of (pos, azimuth, jaw) sampled at DT."""
    def __init__(self, p, az, jaw, tsign):
        self.tsign = tsign
        self.keys = [(0.0, np.array(p, float), float(az), float(jaw))]

    def to(self, t, p=None, az=None, jaw=None):
        _, p0, a0, j0 = self.keys[-1]
        self.keys.append((float(t), p0 if p is None else np.array(p, float), a0 if az is None else float(az),
                          j0 if jaw is None else float(jaw)))

    def sample(self, T):
        out = np.zeros((T, 8)); k = 0
        for i in range(T):
            t = i * DT
            while k + 1 < len(self.keys) - 1 and t > self.keys[k + 1][0]:
                k += 1
            (t0, p0, a0, j0), (t1, p1, a1, j1) = self.keys[k], self.keys[min(k + 1, len(self.keys) - 1)]
            s = _smooth(np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1))
            a = a0 + (a1 - a0) * s
            out[i, :3], out[i, 3:7] = _pose(p0 + (p1 - p0) * s, a, self.tsign)
            out[i, 7] = j0 + (j1 - j0) * s
        return out


def _near(a, ref):
    return ref + (a - ref + np.pi) % (2 * np.pi) - np.pi


def plan(model, data):
    sl = data.body("sleeve")
    c = np.array(sl.xipos); Rm = np.array(sl.xmat).reshape(3, 3)
    top = c[2] + H_SLEEVE / 2
    tgt = np.array(data.body("post_target").xpos)
    post_h = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == model.body("post_target").id:
            post_h = max(post_h, data.geom_xpos[g][2] + model.geom_size[g][1])
    src_top = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == model.body("post_source").id:
            src_top = max(src_top, data.geom_xpos[g][2] + model.geom_size[g][1])
    Lr = np.array(model.body("L_trocar").pos); Rr = np.array(model.body("R_trocar").pos)

    # R grasps the far side of the rim (radial direction perpendicular to its shaft), snapped to a wall segment
    yaw_s = np.arctan2(Rm[1, 0], Rm[0, 0]); seg = 2 * np.pi / 16
    aR0 = _az(Rr, c, -1, snap=(yaw_s, seg))
    uR0 = np.array([np.cos(aR0), np.sin(aR0), 0])
    gR = np.array([c[0], c[1], top]) + R_OUT * uR0

    # handover: in mid-air between the two posts, slightly behind the front row (as in the video), sleeve hanging vertically
    zc = max(post_h, src_top) + 0.027 + H_SLEEVE        # rim height while carried (bottom clears the posts)
    hc = 0.5 * (c[:2] + tgt[:2]) + np.array([0.0, 0.031])
    aRh = _az(Rr, hc, -1, ref=aR0, snap=(aR0, seg))
    uRh = np.array([np.cos(aRh), np.sin(aRh), 0])
    hR = np.array([hc[0], hc[1], zc]) + R_OUT * uRh
    aL = aRh + np.pi; uL = -uRh
    hL = np.array([hc[0], hc[1], zc]) + R_OUT * uL

    # place: sleeve axis over the target post, released once the jaw tips are just above the post top
    aLp = aL
    pL_hi = np.array([tgt[0], tgt[1], zc]) + R_OUT * uL
    pL_lo = pL_hi.copy(); pL_lo[2] = post_h + 0.005
    up = np.array([0, 0, 1.0])

    # rest poses of the two tool tips seen at the start / end of the video (back-projected pixels)
    rest = lambda tcp, az: np.asarray(tcp) - _pose(np.zeros(3), az, 1)[0]
    R0, R1 = rest(R_REST0, aR0), rest(R_REST1, aRh)
    L0, L1 = rest(L_REST0, aL), rest(L_REST1, aL)

    R = _Track(R0, aR0, OPEN, 1)
    R.to(2.8, gR + 0.016 * up)
    R.to(4.6, gR)
    R.to(5.6, jaw=HOLD)
    R.to(8.0, gR + (zc - top) * up)
    R.to(10.3, hR, az=aRh)
    R.to(13.4)
    R.to(14.2, jaw=OPEN)
    R.to(15.0, hR + 0.02 * up)
    R.to(17.5, R1)

    L = _Track(L0, aL, OPEN, 1)
    L.to(9.0)
    L.to(11.0, hL + 0.018 * up)
    L.to(12.3, hL)
    L.to(13.2, jaw=HOLD)
    L.to(14.6)
    L.to(17.0, pL_hi, az=aLp)
    L.to(18.6, pL_lo)
    L.to(19.1, jaw=OPEN)
    L.to(19.5, pL_lo + 0.02 * up)
    L.to(20.0, L1)

    T = int(round(20.05 / DT))
    return np.hstack([L.sample(T), R.sample(T)])
