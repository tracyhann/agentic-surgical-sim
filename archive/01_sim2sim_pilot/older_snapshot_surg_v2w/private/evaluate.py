"""Score a package against the hidden reference (Video2World-style: build, function, geometry, dynamics).

  python evaluate.py <instance> <package_dir> [--render]
All geometric/dynamic comparisons are made in each package's own OpenCV camera frame, so the agent's choice of
world origin and horizontal axes does not matter; only what it implies relative to the endoscope does.
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco

PRIV = Path(__file__).resolve().parent
sys.path.insert(0, str(PRIV))
from make_reference import S  # noqa: E402  (public surgsim module, same executor as the agent's tool)


# ---------------------------------------------------------------- scene geometry helpers
def body_geoms(model, b):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == b]


def geom_aabb(model, data, gs, frame=None):
    """AABB of geoms in world axes, or in the axes of rotation matrix `frame` (e.g. the object's body frame)."""
    F = np.eye(3) if frame is None else np.asarray(frame).reshape(3, 3)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gs:
        R = F.T @ data.geom_xmat[g].reshape(3, 3)
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            ext = model.geom_rbound[g] * np.ones(3)
        else:
            s = model.geom_size[g]
            t, G = int(model.geom_type[g]), mujoco.mjtGeom
            h = None
            if t == int(G.mjGEOM_BOX):
                h = s
            elif t in (int(G.mjGEOM_CYLINDER), int(G.mjGEOM_CAPSULE)):
                a = R[:, 2]                                    # exact extent of a cylinder along each axis
                ext = s[1] * np.abs(a) + s[0] * np.sqrt(np.clip(1 - a ** 2, 0, 1)) + (s[0] if t == int(G.mjGEOM_CAPSULE) else 0)
                h = None
            elif t == int(G.mjGEOM_SPHERE):
                h = np.full(3, s[0])
            elif t == int(G.mjGEOM_ELLIPSOID):
                h = s
            else:
                h = np.full(3, model.geom_rbound[g])
            if h is not None:
                ext = np.abs(R) @ h
        c = F.T @ data.geom_xpos[g]
        lo = np.minimum(lo, c - ext)
        hi = np.maximum(hi, c + ext)
    return lo, hi


def hole_radius(model, data, gs, centre_xy):
    """Inner radius of a ring/cup-like wall set around centre_xy (min over walls of distance - half thickness)."""
    best = np.inf
    for g in gs:
        d = np.linalg.norm(data.geom_xpos[g][:2] - centre_xy)
        lo, hi = geom_aabb(model, data, [g])
        half = 0.5 * min(hi[0] - lo[0], hi[1] - lo[1])
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = model.geom_size[g][:2].min()
        if d > 1e-4:
            best = min(best, d - half)
    return best


def entity_state(model, data, proto):
    """Camera-independent description of the task entities in the current state (world frame)."""
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['object'][0])
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, proto['roles']['target'][0])
    og, tg = body_geoms(model, ob), body_geoms(model, tb)
    olo, ohi = geom_aabb(model, data, og, frame=data.xmat[ob])      # object size in its own body frame
    tlo, thi = geom_aabb(model, data, tg)
    tcen = np.array([(tlo[0] + thi[0]) / 2, (tlo[1] + thi[1]) / 2, thi[2]])   # top-centre of the target
    return dict(obj_com=data.xipos[ob].copy(), obj_size=np.sort(ohi - olo), tgt_top=tcen,
                tgt_size=np.sort(thi - tlo), ob=ob, tb=tb, og=og, tg=tg)


def success(model, data, proto, family):
    e = entity_state(model, data, proto)
    com = e['obj_com']
    tcp = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')]
    if family == 'bead_cup':
        r_in = hole_radius(model, data, e['tg'], e['tgt_top'][:2])
        if not np.isfinite(r_in) or r_in <= 0:
            r_in = 0.35 * min(e['tgt_size'][:2])
        inside = np.linalg.norm(com[:2] - e['tgt_top'][:2]) < r_in and com[2] < e['tgt_top'][2]
    else:  # ring around the destination peg
        r_hole = hole_radius(model, data, e['og'], com[:2])
        peg_r = 0.5 * e['tgt_size'][0]
        inside = np.isfinite(r_hole) and np.linalg.norm(com[:2] - e['tgt_top'][:2]) < r_hole - peg_r \
            and com[2] < e['tgt_top'][2] - 0.002
    instr = S.subtree(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trocar'))
    held = any((model.geom_bodyid[c.geom1] == e['ob'] and model.geom_bodyid[c.geom2] in instr) or
               (model.geom_bodyid[c.geom2] == e['ob'] and model.geom_bodyid[c.geom1] in instr)
               for c in data.contact[:data.ncon])
    return bool(inside and not held)


def segment(p, thr=0.002):
    d0 = np.linalg.norm(p - p[0], axis=1)
    dT = np.linalg.norm(p - p[-1], axis=1)
    on = np.argmax(d0 > thr) if (d0 > thr).any() else 0
    end = len(p) - 1 - np.argmax(dT[::-1] > thr) if (dT > thr).any() else len(p) - 1
    end = max(end + 1, on + 1)
    return p[on:end + 1]


def resample(p, n=50):
    s = np.linspace(0, len(p) - 1, n)
    return np.stack([np.interp(s, np.arange(len(p)), p[:, k]) for k in range(3)], 1)


def run(pkg, family, render=False):
    """Execute a package and return its camera-frame description + task outcome."""
    proto, spec, model, data = S.build(pkg)
    cam = proto['camera']
    acts = np.load(Path(pkg) / proto['actions'].get('path', 'actions.npy'))
    S.reset(model, data, acts[0])
    init = entity_state(model, data, proto)
    rcm = np.asarray(proto['instrument']['rcm_pos'], float)
    rec = S.execute(model, data, acts, proto['roles']['object'] + proto['roles']['target'],
                    render=S.make_renderer(model, cam) if render else None)
    ob = init['ob']
    # CoM trajectory: body pose -> CoM via the body's local inertial offset
    ipos = model.body_ipos[ob]
    traj = []
    for row in rec['pose'][proto['roles']['object'][0]]:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, row[3:])
        traj.append(row[:3] + R.reshape(3, 3) @ ipos)
    traj = np.array(traj)
    final = entity_state(model, data, proto)
    ok = success(model, data, proto, family)
    lifted = traj[:, 2].max() - traj[0, 2] > 0.005
    d0 = np.linalg.norm(init['obj_com'][:2] - init['tgt_top'][:2])
    near = np.linalg.norm(final['obj_com'][:2] - init['tgt_top'][:2]) < max(0.01, 0.25 * d0)
    w2c = lambda p: S.world_to_cam(cam, p)
    return dict(proto=proto, success=ok, progress=(int(lifted) + int(near) + int(ok)) / 3, lifted=bool(lifted), near_target=bool(near),
                obj0=w2c(init['obj_com']), tgt=w2c(init['tgt_top']), rcm=w2c(rcm), obj_size=init['obj_size'],
                traj=np.array([w2c(p) for p in traj]), fovy=cam['fovy_deg'], frames=rec['frames'])


def score(inst, pkg, render=False):
    gt = run(PRIV / 'gt' / inst / 'package', inst)
    errs = S.validate(pkg)
    res = dict(instance=inst, package=str(pkg), build=not errs, validate_errors=errs)
    if errs:
        res.update(success=False, progress=0.0, v2w_score=0.0)
        return res
    try:
        ag = run(pkg, inst, render=render)
    except Exception as e:
        res.update(build=False, validate_errors=[f'execution failed: {e}'], success=False, progress=0.0, v2w_score=0.0)
        return res
    mm = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 1000)
    geo = dict(obj_pos_mm=mm(ag['obj0'], gt['obj0']), target_pos_mm=mm(ag['tgt'], gt['tgt']),
               rcm_pos_mm=mm(ag['rcm'], gt['rcm']),
               obj_size_mm=float(np.abs(ag['obj_size'] - gt['obj_size']).mean() * 1000),
               fovy_err_deg=abs(ag['fovy'] - gt['fovy']))
    sa, sg = segment(ag['traj']), segment(gt['traj'])
    da, dg = resample(sa - sa[0]), resample(sg - sg[0])
    dyn = dict(t_ape_mm=float(np.minimum(np.linalg.norm(da - dg, axis=1), 0.1).mean() * 1000),
               terminal_disp_err_mm=mm(ag['traj'][-1] - ag['traj'][0], gt['traj'][-1] - gt['traj'][0]),
               final_pos_err_mm=mm(ag['traj'][-1], gt['traj'][-1]))
    g = np.mean([np.exp(-geo['obj_pos_mm'] / 10), np.exp(-geo['target_pos_mm'] / 10), np.exp(-geo['rcm_pos_mm'] / 20),
                 np.exp(-geo['obj_size_mm'] / 3), np.exp(-geo['fovy_err_deg'] / 10)])
    d = np.exp(-dyn['t_ape_mm'] / 10)
    v2w = 100 * (0.4 * ag['success'] + 0.2 * ag['progress'] + 0.2 * g + 0.2 * d)
    res.update(success=ag['success'], progress=round(ag['progress'], 3), lifted=ag['lifted'], near_target=ag['near_target'],
               geometry={k: round(v, 2) for k, v in geo.items()}, dynamics={k: round(v, 2) for k, v in dyn.items()},
               geometry_score=round(float(g), 3), dynamics_score=round(float(d), 3), v2w_score=round(float(v2w), 1),
               reference_success=gt['success'])
    if render and ag['frames']:
        import imageio.v2 as imageio
        src = [f for f in imageio.get_reader(PRIV / 'gt' / inst / 'package' / 'source' / 'video.mp4')]
        n = max(len(src), len(ag['frames']))
        comp = [np.concatenate([src[min(int(k * len(src) / n), len(src) - 1)],
                                ag['frames'][min(int(k * len(ag['frames']) / n), len(ag['frames']) - 1)]], 1) for k in range(n)]
        imageio.mimsave(Path(pkg).parent / f'eval_compare_{inst}.mp4', comp, fps=20, macro_block_size=1)
    return res


if __name__ == '__main__':
    r = score(sys.argv[1], Path(sys.argv[2]), render='--render' in sys.argv)
    print(json.dumps(r, indent=1, default=str))
