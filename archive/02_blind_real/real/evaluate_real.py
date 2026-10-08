"""Score a real-video reconstruction (post and sleeve, ROSMA) against real measurements.

  python evaluate_real.py <package_dir> [--trial X01 --t0 5 --t1 26] [--render] [--rollouts]

Function (in the package's own scene): lifted off the source post / mid-air handoff R->L / placed over the target.
Image-space fidelity (pixels, 1024x768): the package's sleeve and tool tips projected through its own camera vs
the real sleeve (colour tracking) and the real tool tips (dVRK kinematics through the hand-eye calibration).
Time is normalised over the clip (the reconstruction may run at a different pace than the demonstration).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2, mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'real'))
import surgsim2 as S2  # noqa: E402
import rosma  # noqa: E402

W, H = 1024, 768
ARM_OF = {'L_': 'PSM2', 'R_': 'PSM1'}       # image-left arm = PSM2, image-right = PSM1


def body_geoms(m, b):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]


def _AABB(m, d, gs):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gs:
        R = d.geom_xmat[g].reshape(3, 3)
        s, t = m.geom_size[g], int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            ext = np.abs(R) @ s
        elif t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            a = R[:, 2]
            ext = s[1] * np.abs(a) + s[0] * np.sqrt(np.clip(1 - a ** 2, 0, 1)) + (s[0] if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0)
        else:
            ext = np.full(3, m.geom_rbound[g])
        lo, hi = np.minimum(lo, d.geom_xpos[g] - ext), np.maximum(hi, d.geom_xpos[g] + ext)
    return lo, hi


def hole_radius(m, d, gs, centre):
    best = np.inf
    for g in gs:
        r = np.linalg.norm(d.geom_xpos[g][:2] - centre[:2])
        if r < 1e-4:
            continue
        half = m.geom_size[g][:2].min() if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX) else m.geom_size[g][0]
        best = min(best, r - half)
    return best


def project(cam, P):
    R = S2.S1.cam_axes(cam['pos'], cam['lookat'])
    pc = (np.atleast_2d(P) - np.asarray(cam['pos'])) @ R.T
    f = cam['height'] / 2 / np.tan(np.radians(cam['fovy_deg']) / 2)
    sx, sy = W / cam['width'], H / cam['height']
    return np.stack([(f * pc[:, 0] / pc[:, 2] + cam['width'] / 2) * sx, (f * pc[:, 1] / pc[:, 2] + cam['height'] / 2) * sy], 1)


def gt_tracks(trial, t0, t1):
    cal = json.load(open(ROOT / 'real' / 'calib.json'))
    K = np.array([[cal['f'], 0, W / 2], [0, cal['f'], H / 2], [0, 0, 1.0]])
    kin = rosma.load_kin(trial)
    tv = np.arange(int(round(t0 * 15)), int(round(t1 * 15)) + 1) / 15
    tips = {}
    for P, arm in ARM_OF.items():
        c = cal['arms'][arm]
        tips[P] = cv2.projectPoints(rosma.at_video_time(kin, arm, tv), np.array(c['rvec']), np.array(c['tvec']), K, None)[0].reshape(-1, 2)
    tr = json.load(open(ROOT / 'real' / f'sleeve_track_{trial}_yellow.json'))
    sl = np.array([p['uv'] if p['uv'] else [np.nan, np.nan] for p in tr if t0 - 1e-6 <= p['t'] <= t1 + 1e-6])
    return tv, tips, sl


def resample(x, n=100):
    s = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(s, np.arange(len(x)), x[:, k]) for k in range(x.shape[1])], 1)


def run(pkg, render=False, edit=None, policy=False):
    proto, spec, model, data = S2.S1.build(pkg, edit=edit)
    acts = np.load(Path(pkg) / proto['actions'].get('path', 'actions.npy'))
    if policy:
        q, jaw, _ = S2.joint_targets(model, acts[:1])
        S2.reset(model, data, q[0], jaw[0])
        acts = np.asarray(S2.S1.load_policy(pkg)(model, data), float)
        bad = S2.check_actions(model, acts, 'plan')
        if bad:
            raise ValueError(bad[0])
    q, jaw, _ = S2.joint_targets(model, acts[:1])
    S2.reset(model, data, q[0], jaw[0])
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['object'][0])
    src = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['source'][0])
    tgt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['target'][0])
    og = body_geoms(model, ob)
    lo, hi = _AABB(model, data, og)
    bottom_off = data.xipos[ob][2] - lo[2]
    src_top = _AABB(model, data, body_geoms(model, src))[1][2]
    rec = S2.execute(model, data, acts, [proto['roles']['object'][0]],
                     render=S2.S1.make_renderer(model, proto['camera'], vig=False) if render else None)
    com = rec['pose'][proto['roles']['object'][0]][:, :3]
    held = np.array([c[0] for c in rec['contacts']])
    lifted = bool(np.any(com[:, 2] - bottom_off > src_top + 0.001))
    tlo, thi = _AABB(model, data, body_geoms(model, tgt))
    tc = (tlo + thi) / 2
    handoff = False
    if (held & 2).any():
        first_R = np.argmax((held & 2) > 0)
        later = np.arange(len(held)) > first_R
        handoff = bool(np.any(later & (held == 1) & (com[:, 2] - bottom_off > src_top)))
    r_hole = hole_radius(model, data, og, data.xipos[ob])
    placed = bool(np.linalg.norm(data.xipos[ob][:2] - tc[:2]) < r_hole - 0.5 * (thi[0] - tlo[0])
                  and data.xipos[ob][2] - bottom_off < thi[2] and held[-1] == 0)
    return dict(proto=proto, rec=rec, n_act=len(acts), lifted=lifted, handoff=handoff, placed=placed)


def score(pkg, trial='X01', t0=5.0, t1=26.0, render=False):
    errs = S2.validate(pkg)
    res = dict(package=str(pkg), build=not errs, validate_errors=errs)
    if errs:
        return res
    r = run(pkg, render=render)
    cam = r['proto']['camera']
    res.update(lifted=r['lifted'], handoff=r['handoff'], placed=r['placed'],
               progress=round((r['lifted'] + r['handoff'] + r['placed']) / 3, 3), success=r['lifted'] and r['handoff'] and r['placed'])
    tv, tips, sl = gt_tracks(trial, t0, t1)
    n = r['n_act']                                    # exclude the 1 s hold after the last row
    rec = r['rec']
    a_sl = project(cam, rec['pose'][r['proto']['roles']['object'][0]][:n, :3])
    ok = ~np.isnan(sl[:, 0])
    g_sl = resample(sl[ok])
    a_sl_r = resample(a_sl)
    img = dict(sleeve_start_px=float(np.linalg.norm(a_sl[0] - sl[ok][0])), sleeve_end_px=float(np.linalg.norm(a_sl[-1] - sl[ok][-1])),
               sleeve_traj_px=float(np.linalg.norm(a_sl_r - g_sl, axis=1).mean()))
    for k, P in enumerate(S2.ARMS):
        a_tip = project(cam, rec['tcp'][:n, k])
        img[f'{P}tip_start_px'] = float(np.linalg.norm(a_tip[0] - tips[P][0]))
        img[f'{P}tip_traj_px'] = float(np.linalg.norm(resample(a_tip) - resample(tips[P]), axis=1).mean())
    res['image_space'] = {k: round(v, 1) for k, v in img.items()}
    res['clip_s'], res['rollout_s'] = round(t1 - t0, 2), round(n * S2.DT, 2)
    if render:
        import imageio.v2 as imageio
        src = [f for f in imageio.get_reader(Path(pkg) / 'source' / 'video.mp4')]
        frames = rec['frames'][:n]
        m = max(len(src), len(frames))
        out = []
        for k in range(m):
            i, j = min(int(k * len(src) / m), len(src) - 1), min(int(k * len(frames) / m), len(frames) - 1)
            a = src[i].copy()
            if ok[i]:
                cv2.circle(a, tuple(sl[i].astype(int)), 9, (0, 255, 0), 2)
            cv2.circle(a, tuple(a_sl[j].astype(int)), 9, (255, 0, 255), 2)
            for k2, P in enumerate(S2.ARMS):
                cv2.circle(a, tuple(tips[P][i].astype(int)), 6, (60, 160, 255), 2)
                cv2.circle(a, tuple(project(cam, rec['tcp'][j, k2])[0].astype(int)), 6, (255, 140, 0), 2)
            b = frames[j] if frames[j].shape == a.shape else cv2.resize(frames[j], (a.shape[1], a.shape[0]))
            out.append(np.concatenate([a, b], 1))
        imageio.mimsave(Path(pkg).parent / 'eval_compare.mp4', out, fps=15, macro_block_size=1)
    return res


def rollouts(pkg):
    proto = S2.S1.load_protocol(pkg)
    R = S2.S1.cam_axes(proto['camera']['pos'], proto['camera']['lookat'])
    dirs = {'right': R[0] * [1, 1, 0], 'far': R[2] * [1, 1, 0]}
    dirs = {k: v / np.linalg.norm(v) for k, v in dirs.items()}
    dirs.update({'left': -dirs['right'], 'near': -dirs['far']})
    tgt = proto['roles']['target'][0]
    rows = []
    for dn, dv in dirs.items():
        for mm in (5, 10):
            def edit(spec, dv=dv, mm=mm):
                b = spec.body(tgt)
                b.pos = list(np.asarray(b.pos) + dv * mm / 1000)
            try:
                r = run(pkg, edit=edit, policy=True)
                rows.append(dict(layout=f'target_{dn}_{mm}mm', lifted=r['lifted'], handoff=r['handoff'], placed=r['placed']))
            except Exception as e:
                rows.append(dict(layout=f'target_{dn}_{mm}mm', error=f'{type(e).__name__}: {e}'[:200], placed=False))
            print(rows[-1], flush=True)
    return dict(n=len(rows), placed_rate=float(np.mean([r['placed'] for r in rows])), rows=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg')
    ap.add_argument('--trial', default='X01')
    ap.add_argument('--t0', type=float, default=5.0)
    ap.add_argument('--t1', type=float, default=26.0)
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--rollouts', action='store_true')
    a = ap.parse_args()
    res = score(Path(a.pkg), a.trial, a.t0, a.t1, render=a.render)
    if a.rollouts and res.get('build'):
        res['rollouts'] = rollouts(Path(a.pkg))
    print(json.dumps(res, indent=1, default=str))
