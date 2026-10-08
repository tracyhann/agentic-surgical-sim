"""Peg transfer policy: grasp the ring wall, lift it off its peg, carry it over the target peg, lower, release.
All motion is computed from the simulation state (ring pose, target peg pose, other peg positions)."""
import os, sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument'))
import kinematics as K

SLEW = np.array([0.08, 0.08, 0.006, 0.15, 0.007]) * 0.9
RING_APOTHEM = 0.0105      # centre -> mid-wall of a flat
GRASP_Z = 0.0045           # TCP height above the ring's bottom face while grasping
JAW_PRE = 0.0036           # per-jaw opening used to straddle the wall
CLEAR = 0.012              # clearance of the ring bottom above the peg top while carrying


def _yaw_of(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def plan(model, data):
    rcm = data.body('trocar').xpos.copy()
    ring = data.body('ring').xpos.copy()
    ring_yaw = _yaw_of(data.body('ring').xquat)
    tgt = data.body('target_peg')
    gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == tgt.id][0]
    peg_xy = data.geom_xpos[gid][:2].copy()
    peg_top = data.geom_xpos[gid][2] + model.geom_size[gid][1]
    pegs = [data.geom_xpos[g][:2].copy() for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('peg')]
    h = 0.003                                   # ring half height
    z_bot = ring[2] - h
    tcp0 = data.site('tcp').xpos.copy()
    y0, p0, i0 = K.ik(rcm, tcp0)
    jaw0 = float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'jaw_left')]])

    # choose the flat of the octagon whose normal is best aligned with the jaw opening axis and far from pegs
    best = None
    for k in range(8):
        a = ring_yaw + k * np.pi / 4
        n = np.array([np.cos(a), np.sin(a)])
        g = ring[:2] + RING_APOTHEM * n
        yaw = np.arctan2(-(g[0] - rcm[0]), g[1] - rcm[1])
        u = np.array([np.cos(yaw), np.sin(yaw)])
        align = abs(n @ u)
        v = np.array([-np.sin(yaw), np.cos(yaw)])          # shaft heading (horizontal)
        blocked = False
        for p in pegs:                                    # a peg inside the footprint swept by the open jaws / shaft
            du, dv = (p - g) @ u, (p - g) @ v
            if abs(du) < 0.0078 and -0.024 < dv < 0.006:
                blocked = True
        tilt_ok = True
        score = align + (0.3 if n @ u > 0 else 0.0) - (10.0 if blocked else 0.0)
        if best is None or score > best[0]:
            best = (score, g, yaw)
    g = best[1]
    grasp = np.array([g[0], g[1], z_bot + GRASP_Z])
    yaw_g = best[2]
    z_carry = max(peg_top + CLEAR + GRASP_Z, grasp[2] + 0.02)
    lift = np.array([g[0], g[1], z_carry])
    # ring centre relative to the TCP rotates with the shaft yaw: solve for the TCP that puts the centre on the peg
    off = ring[:2] - g
    place_xy = peg_xy - off
    for _ in range(10):
        yaw_p = np.arctan2(-(place_xy[0] - rcm[0]), place_xy[1] - rcm[1])
        d = yaw_p - yaw_g
        Rz = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
        place_xy = peg_xy - Rz @ off
    over = np.array([place_xy[0], place_xy[1], z_carry])
    z_rel = (data.geom_xpos[gid][2] - model.geom_size[gid][1]) + 0.006 + GRASP_Z
    low = np.array([place_xy[0], place_xy[1], z_rel])
    above = np.array([g[0], g[1], grasp[2] + 0.022])

    # (tcp, jaw, nominal steps) - timing follows the video (20 fps == one control row per frame)
    wps = [(tcp0, jaw0, 0), (tcp0, JAW_PRE, 8), (above, JAW_PRE, 26), (grasp, JAW_PRE, 22), (grasp, 0.0, 12),
           (grasp, 0.0, 4), (lift, 0.0, 40), (over, 0.0, 48), (low, 0.0, 44), (low, K.JAW_OPEN * 0.8, 12),
           (over, K.JAW_OPEN * 0.8, 24), (tcp0, K.JAW_OPEN * 0.8, 52), (tcp0, jaw0, 16)]
    rows = [np.array([y0, p0, i0, 0.0, jaw0])]
    for (a, ja, _), (b, jb, n) in zip(wps[:-1], wps[1:]):
        while True:
            seg = []
            for s in range(1, n + 1):
                t = s / n
                t = t * t * (3 - 2 * t)
                y, p, ins = K.ik(rcm, a + (b - a) * t)
                seg.append([y, p, ins, 0.0, ja + (jb - ja) * s / n])
            seg = np.array(seg)
            dd = np.abs(np.diff(np.vstack([rows[-1], seg]), axis=0))
            if np.all(dd <= SLEW) or n > 300:
                break
            n = int(n * 1.3) + 1
        rows.extend(list(seg))
    return np.array(rows, dtype=np.float64)
