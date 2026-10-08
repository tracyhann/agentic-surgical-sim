"""Control stream for chole_sweep: grasper_left holds/retracts the gallbladder neck, probe_right sweeps beside it.
Targets are computed from the tissue poses found in `data` (neck grasp point, gallbladder body, strand root)."""
import json, os, sys
import numpy as np
import mujoco
_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_D, 'instrument'))
import kinematics as K
CONST = json.load(open(os.path.join(_D, 'policy_const.json')))
SLEW = np.array([0.08, 0.08, 0.006, 0.15, 0.007]) * 0.98
T = 121          # 6.0 s at 0.05 s
FPS_RATIO = 1.25  # video frames per control step
JAW0 = 0.0026    # initial per-jaw opening: just clear of the 2 mm-radius neck


def _interp(track, k):
    tr = np.asarray(track, float)
    return np.array([np.interp(k * FPS_RATIO, tr[:, 0], tr[:, i]) for i in (1, 2, 3)])


def _safe_ik(rcm, p, h):
    v = K._rz(-h) @ (np.asarray(p) - np.asarray(rcm)); dist = np.linalg.norm(v)
    return np.array([np.clip(np.arctan2(-v[0], v[1]), -1.57, 1.57), np.clip(np.arcsin(np.clip(-v[2] / dist, -1, 1)), 0, 1.4),
                     np.clip(dist - K.TCP_OFFSET, 0, 0.25)])


def plan(model, data):
    def bpos(name, default):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[i].copy() if i >= 0 else np.asarray(default, float)
    neck = bpos('gb_neck3', CONST['NECK0'])
    # the scene may be displaced: shift the probe path with the tissue it touches
    shift = 0.5 * (bpos('gb_body', CONST['BODY0']) - CONST['BODY0']) + 0.5 * (bpos('strand0', CONST['STRAND0']) - CONST['STRAND0'])
    g_rcm, g_h, p_rcm, p_h = CONST['G_RCM'], CONST['G_H'], CONST['P_RCM'], CONST['P_H']
    a = np.zeros((T, 10))
    a[0, :3] = _safe_ik(g_rcm, CONST['NECK0'], g_h); a[0, 4] = JAW0
    a[0, 5:8] = _safe_ik(p_rcm, CONST['PT'][0][1:], p_h)
    for k in range(1, T):
        g = np.zeros(5); p = np.zeros(5)
        # the neck sags a little once released by the settle; grasp where it actually is, then follow the video path
        g[:3] = _safe_ik(g_rcm, neck + _interp(CONST['GT'], max(k - 3, 0)), g_h)
        g[4] = 0.0 if k >= 2 else JAW0
        p[:3] = _safe_ik(p_rcm, _interp(CONST['PT'], k) + shift, p_h)
        tgt = np.concatenate([g, p])
        lim = np.tile(SLEW, 2)
        a[k] = a[k - 1] + np.clip(tgt - a[k - 1], -lim, lim)
    return a
