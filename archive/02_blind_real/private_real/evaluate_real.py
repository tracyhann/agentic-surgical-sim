"""Score a real-video package against the hidden GT.

  python evaluate_real.py <instance> [<package_dir>]       (default package: runs_real/<inst>/out)
rosma_handoff: task success in the agent's own scene; sleeve 2D track error (px) vs colour-segmented GT; tool-tip 3D
               trajectories vs dVRK kinematics after similarity alignment (shape error, recovered metric scale).
chole_sweep:   tool-tip 2D error (px) vs auto-tracked probe tip and hand-annotated grasper keyframes; tissue response.
Writes eval.json and eval_overlay.mp4 (source frames; GT in green, agent projection in red | agent rollout).
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import mujoco
import cv2
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent


def load_tool(inst):
    p = ROOT / 'runs_real' / inst / 'task' / 'tools' / 'surgsim.py'
    spec = importlib.util.spec_from_file_location(f'surgsim_{inst}', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resample(x, n=100):
    s = np.linspace(0, len(x) - 1, n)
    return np.stack([np.interp(s, np.arange(len(x)), x[:, k]) for k in range(x.shape[1])], 1)


def umeyama(A, B, scale=True):
    """Best s, R, t with s R A + t ~ B (rows are points). Returns aligned A and s."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum() * len(A) if scale else 1.0
    return (s * (R @ a.T)).T + mb, s


def geoms_of(model, b):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]


def hole_radius(model, data, gs, centre_xy):
    best = np.inf
    for g in gs:
        d = np.linalg.norm(data.geom_xpos[g][:2] - centre_xy)
        if d > 1e-4:
            half = model.geom_size[g][:2].min() if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX) else model.geom_rbound[g]
            best = min(best, d - half)
    return best


def sleeve_success(model, data, proto, S):
    ob = S._id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['object'][0])
    tb = S._id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['target'][0])
    com = data.xipos[ob]
    tg = geoms_of(model, tb)
    top = max(data.geom_xpos[g][2] + (model.geom_size[g][1] if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER) else model.geom_rbound[g]) for g in tg)
    axis = np.mean([data.geom_xpos[g][:2] for g in tg], 0)
    post_r = min(model.geom_size[g][0] for g in tg)
    r_hole = hole_radius(model, data, geoms_of(model, ob), com[:2])
    around = np.isfinite(r_hole) and np.linalg.norm(com[:2] - axis) < max(r_hole - post_r, 0.002) and com[2] < top
    return bool(around), dict(hole_radius_mm=round(float(r_hole) * 1000, 2) if np.isfinite(r_hole) else None,
                              dist_to_post_axis_mm=round(float(np.linalg.norm(com[:2] - axis)) * 1000, 2))


def overlay(frames_src, gt_tracks, ag_tracks, rollout, path):
    n = len(frames_src)
    out = []
    for k, f in enumerate(frames_src):
        f = f.copy()
        for tr, col in ((gt_tracks, (0, 255, 0)), (ag_tracks, (255, 0, 0))):
            for name, t in tr.items():
                idx = min(int(k * len(t) / n), len(t) - 1)
                if np.isfinite(t[idx]).all():
                    cv2.circle(f, tuple(int(c) for c in t[idx]), 8, col, 2)
                    cv2.putText(f, name, (int(t[idx][0]) + 9, int(t[idx][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        r = rollout[min(int(k * len(rollout) / n), len(rollout) - 1)]
        if r.shape != f.shape:
            r = cv2.resize(r, (f.shape[1], f.shape[0]))
        out.append(np.concatenate([f, r], 1))
    imageio.mimsave(path, out, fps=15, macro_block_size=1, quality=7)


def evaluate(inst, pkg):
    S = load_tool(inst)
    gt = np.load(ROOT / 'private_real' / 'gt' / inst / 'gt.npz')
    errs = S.validate(pkg)
    res = dict(instance=inst, package=str(pkg), build=not errs, validate_errors=errs)
    if errs:
        return res
    proto, spec, model, data = S.build(pkg)
    cam = proto['camera']
    acts = np.load(Path(pkg) / proto['actions'].get('path', 'actions.npy'))
    bodies = [b for b in proto['roles'].get('object', []) + proto['roles'].get('target', []) if S._id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
    S.reset(model, data, acts[0])
    flex0 = data.flexvert_xpos.copy() if model.nflex else None
    rec = S.execute(model, data, acts, bodies, render=S.make_renderer(model, cam))
    n = rec['n_actions']
    src = [f for f in imageio.get_reader(Path(pkg) / 'source' / 'video.mp4')]
    res['duration_s'] = dict(agent=round(n * S.DT, 2), video=round(len(src) / S.CONFIG['fps'], 2))
    tcp = {p: v[:n] for p, v in rec['tcp'].items()}
    if inst == 'rosma_handoff':
        ok, info = sleeve_success(model, data, proto, S)
        ob = proto['roles']['object'][0]
        sp = S.project(cam, rec['pose'][ob][:n, :3])
        g = gt['sleeve_px']
        e = np.linalg.norm(resample(sp) - resample(g), axis=1)
        res.update(success=ok, success_detail=info,
                   sleeve_px=dict(median=round(float(np.median(e)), 1), mean=round(float(e.mean()), 1),
                                  start=round(float(np.linalg.norm(sp[0] - g[0])), 1), end=round(float(np.linalg.norm(sp[-1] - g[-1])), 1)))
        tips = {}
        for name, key in (('psm_left', 'psm2'), ('psm_right', 'psm1')):
            A, B = resample(tcp[name]), resample(gt[key])
            al_s, s = umeyama(A, B, True)
            al_r, _ = umeyama(A, B, False)
            tips[name] = dict(shape_rmse_mm=round(float(np.sqrt(((al_s - B) ** 2).sum(1).mean()) * 1000), 1),
                              rigid_rmse_mm=round(float(np.sqrt(((al_r - B) ** 2).sum(1).mean()) * 1000), 1),
                              scale_to_real=round(float(s), 3),
                              path_len_mm=dict(agent=round(float(np.linalg.norm(np.diff(tcp[name], axis=0), axis=1).sum()) * 1000, 1),
                                               real=round(float(np.linalg.norm(np.diff(gt[key], axis=0), axis=1).sum()) * 1000, 1)))
        res['tool_tips_vs_kinematics'] = tips
        overlay(src, {'sleeve': g}, {'sleeve': sp}, rec['frames'], Path(pkg).parent / 'eval_overlay.mp4')
    else:
        pr = S.project(cam, tcp['probe_right'])
        gl = S.project(cam, tcp['grasper_left'])
        gp = gt['probe_px']
        ok = np.isfinite(gp[:, 0])
        idx = np.round(np.linspace(0, len(pr) - 1, len(gp))).astype(int)
        e = np.linalg.norm(pr[idx][ok] - gp[ok], axis=1)
        keys = gt['grasper_keys']
        kidx = np.round(keys[:, 0] / (len(gp) - 1) * (len(gl) - 1)).astype(int)
        eg = np.linalg.norm(gl[kidx] - keys[:, 1:], axis=1)
        res.update(probe_tip_px=dict(median=round(float(np.median(e)), 1), mean=round(float(e.mean()), 1), n=int(ok.sum())),
                   grasper_tip_px=dict(per_key=eg.round(1).tolist(), mean=round(float(eg.mean()), 1)))
        if flex0 is not None:
            d = np.linalg.norm(data.flexvert_xpos - flex0, axis=1)
            res['tissue'] = dict(kind='flex', n_vertices=int(len(d)), max_disp_mm=round(float(d.max()) * 1000, 1),
                                 median_disp_mm=round(float(np.median(d)) * 1000, 2))
        elif bodies:
            p = rec['pose'][bodies[0]][:, :3]
            res['tissue'] = dict(kind='body', max_disp_mm=round(float(np.linalg.norm(p - p[0], axis=1).max()) * 1000, 1))
        gtr = {'probe': gp, 'grasper': np.full((len(gp), 2), np.nan)}
        for k, u, v in keys:
            gtr['grasper'][int(k)] = (u, v)
        overlay(src, gtr, {'probe': pr, 'grasper': gl}, rec['frames'], Path(pkg).parent / 'eval_overlay.mp4')
    (Path(pkg).parent / 'eval.json').write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    inst = sys.argv[1]
    pkg = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'runs_real' / inst / 'out'
    print(json.dumps(evaluate(inst, pkg), indent=1, default=str))
