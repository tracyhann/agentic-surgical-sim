"""surgsim2: two-arm dVRK-style PSM packages with Cartesian tool-tip control.

  python surgsim2.py validate OUT
  python surgsim2.py replay OUT [--policy]

Action row (16 values) = [L: x y z qw qx qy qz jaw,  R: x y z qw qx qy qz jaw]: absolute tool-tip (site
`<P>tcp`) pose in the world frame and total jaw opening angle (rad, 0 = closed, 1.2 = wide open), at dt = 0.05 s.
Each row is converted to joint targets by damped-least-squares IK on the 6 arm joints (warm-started from the
previous row), then executed by the joint servos under physics. A completed replay is not a task judgement.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surgsim as S1  # camera / rendering / policy loading helpers

TIMESTEP, DT, SETTLE, TAIL = 0.0005, 0.05, 0.25, 1.0
ARMS = ('L_', 'R_')
ARM_JOINTS = ('yaw', 'pitch', 'insertion', 'roll', 'wrist_pitch', 'wrist_yaw')
SLEW = np.array([0.08, 0.08, 0.006, 0.30, 0.30, 0.30])     # max joint-target change per control step
JAW_MAX, JAW_SQUEEZE = 1.2, -0.1
INSTR_DIR = Path(__file__).resolve().parent.parent / 'instrument_psm'
INSTR_BODIES = ('yaw_link', 'pitch_link', 'shaft', 'roll_link', 'wrist_pitch_link', 'wrist_yaw_link', 'jaw_a', 'jaw_b')
INSTR_GEOMS = ('shaft_geom', 'collar_geom', 'clevis_geom', 'jaw_a_geom', 'jaw_b_geom')
INSTR_ACTS = ARM_JOINTS + ('jaw_a', 'jaw_b')


def _id(m, kind, name):
    return mujoco.mj_name2id(m, kind, name)


# ---------------------------------------------------------------- IK
class ArmIK:
    def __init__(self, model, P):
        self.m, self.d, self.P = model, mujoco.MjData(model), P
        self.jid = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, P + j) for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jid])
        self.lo, self.hi = model.jnt_range[self.jid, 0], model.jnt_range[self.jid, 1]
        self.site = _id(model, mujoco.mjtObj.mjOBJ_SITE, P + 'tcp')
        self.rcm = model.body_pos[_id(model, mujoco.mjtObj.mjOBJ_BODY, P + 'trocar')].copy()

    def seed(self, pos):
        v = np.asarray(pos) - self.rcm
        dist = np.linalg.norm(v)
        return np.clip([np.arctan2(-v[0], v[1]), np.arcsin(np.clip(-v[2] / dist, -1, 1)), dist - 0.0193, 0, 0, 0],
                       self.lo, self.hi)

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos, quat, q0, iters=60):
        Rt = np.zeros(9)
        mujoco.mju_quat2Mat(Rt, np.asarray(quat, float) / np.linalg.norm(quat))
        Rt = Rt.reshape(3, 3)
        q = np.array(q0, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            ep = np.asarray(pos) - p
            er = 0.5 * sum(np.cross(R[:, k], Rt[:, k]) for k in range(3))
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], 0.02 * jr[:, self.dadr]])
            e = np.concatenate([ep, 0.02 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(q)
        err_r = np.degrees(np.arccos(np.clip((np.trace(R.T @ Rt) - 1) / 2, -1, 1)))
        return q, float(np.linalg.norm(np.asarray(pos) - p)), float(err_r)


def joint_targets(model, actions):
    """IK for every row -> (T, 2, 6) joint targets, (T, 2) jaw, worst position/orientation error."""
    actions = np.asarray(actions, float)
    iks = {P: ArmIK(model, P) for P in ARMS}
    out, jaw, worst = np.zeros((len(actions), 2, 6)), np.zeros((len(actions), 2)), [0.0, 0.0]
    for a, P in enumerate(ARMS):
        q = None
        for k, row in enumerate(actions):
            r = row[8 * a: 8 * a + 8]
            q, ep, er = iks[P].solve(r[:3], r[3:7], iks[P].seed(r[:3]) if q is None else q, iters=200 if q is None else 60)
            out[k, a], jaw[k, a] = q, r[7]
            worst = [max(worst[0], ep), max(worst[1], er)]
    return out, jaw, worst


def apply(model, data, q, jaw):
    for a, P in enumerate(ARMS):
        for i, j in enumerate(ARM_JOINTS):
            data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, P + j)] = q[a, i]
        for j in ('jaw_a', 'jaw_b'):
            data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, P + j)] = np.clip(jaw[a], JAW_SQUEEZE, JAW_MAX) / 2


def reset(model, data, q0, jaw0):
    mujoco.mj_resetData(model, data)
    for a, P in enumerate(ARMS):
        for i, j in enumerate(ARM_JOINTS):
            data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, P + j)]] = q0[a, i]
        for j in ('jaw_a', 'jaw_b'):
            data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, P + j)]] = max(jaw0[a], 0) / 2
    apply(model, data, q0, jaw0)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(SETTLE / model.opt.timestep))):
        mujoco.mj_step(model, data)


def execute(model, data, actions, bodies, render=None):
    q, jaw, _ = joint_targets(model, actions)
    nsub = int(round(DT / model.opt.timestep))
    reset(model, data, q[0], jaw[0])
    bid = [_id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
    sid = [_id(model, mujoco.mjtObj.mjOBJ_SITE, P + 'tcp') for P in ARMS]
    rec = dict(t=[], pose={b: [] for b in bodies}, tcp=[], frames=[], contacts=[])

    def log(t):
        rec['t'].append(t)
        for b, i in zip(bodies, bid):
            rec['pose'][b].append(np.concatenate([data.xipos[i], data.xquat[i]]))
        rec['tcp'].append(np.stack([data.site_xpos[s] for s in sid]))
        rec['contacts'].append(contact_summary(model, data, bid))
        if render is not None:
            rec['frames'].append(render(data))
    log(0.0)
    n_tail = int(round(TAIL / DT))
    qs = np.concatenate([q, np.repeat(q[-1:], n_tail, 0)])
    js = np.concatenate([jaw, np.repeat(jaw[-1:], n_tail, 0)])
    for k in range(1, len(qs)):
        for s in range(nsub):
            w = (s + 1) / nsub
            apply(model, data, qs[k - 1] + (qs[k] - qs[k - 1]) * w, js[k - 1] + (js[k] - js[k - 1]) * w)
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f'simulation diverged at control step {k}')
        log(k * DT)
    rec['pose'] = {b: np.array(v) for b, v in rec['pose'].items()}
    rec['tcp'], rec['t'] = np.array(rec['tcp']), np.array(rec['t'])
    return rec


def contact_summary(model, data, bids):
    """For each tracked body: which arms' jaws touch it (bitmask L=1, R=2)."""
    arm_of = {}
    for a, P in enumerate(ARMS):
        for b in ('jaw_a', 'jaw_b'):
            arm_of[_id(model, mujoco.mjtObj.mjOBJ_BODY, P + b)] = 1 << a
    out = [0] * len(bids)
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        for k, b in enumerate(bids):
            if b1 == b and b2 in arm_of:
                out[k] |= arm_of[b2]
            if b2 == b and b1 in arm_of:
                out[k] |= arm_of[b1]
    return out


# ---------------------------------------------------------------- validation
def reference_instrument():
    xml = '<mujoco><compiler angle="radian"/><option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/><worldbody>'
    for P in ARMS:
        xml += f'<body name="{P}trocar"><include file="{INSTR_DIR}/{P}body.xml"/></body>'
    xml += '</worldbody><actuator>' + ''.join(f'<include file="{INSTR_DIR}/{P}actuators.xml"/>' for P in ARMS)
    xml += '</actuator><contact>' + ''.join(f'<include file="{INSTR_DIR}/{P}contacts.xml"/>' for P in ARMS) + '</contact></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)


def check_actions(model, a, name='actions'):
    a = np.asarray(a, float)
    if a.ndim != 2 or a.shape[1] != 16:
        return [f'{name}: shape {a.shape}, expected (T, 16)']
    if not (10 <= len(a) <= 2400) or not np.all(np.isfinite(a)):
        return [f'{name}: need 10..2400 finite rows']
    errs = []
    for k in (3, 11):
        if np.abs(np.linalg.norm(a[:, k:k + 4], axis=1) - 1).max() > 1e-3:
            errs.append(f'{name}: quaternion columns {k}..{k+3} must be unit (w x y z)')
    if a[:, [7, 15]].min() < JAW_SQUEEZE - 1e-9 or a[:, [7, 15]].max() > JAW_MAX + 1e-9:
        errs.append(f'{name}: jaw outside [{JAW_SQUEEZE}, {JAW_MAX}]')
    if errs:
        return errs
    q, _, (ep, er) = joint_targets(model, a)
    if ep > 0.001 or er > 3.0:
        errs.append(f'{name}: some poses are unreachable (worst IK error {ep*1000:.1f} mm / {er:.1f} deg)')
    step = np.abs(np.diff(q, axis=0)).max(axis=(0, 1)) if len(q) > 1 else np.zeros(6)
    over = [f'{j} {s:.3f}>{l}' for j, s, l in zip(ARM_JOINTS, step, SLEW) if s > l + 1e-9]
    if over:
        errs.append(f'{name}: joint motion per 0.05 s step too fast ({", ".join(over)}); move the tip more slowly')
    return errs


def validate(out):
    out = Path(out)
    try:
        proto = S1.load_protocol(out)
    except Exception as e:
        return [f'protocol.json unreadable: {e}']
    errs = [f'protocol.json: missing "{k}"' for k in ('status', 'task', 'model_path', 'camera', 'instrument', 'roles', 'actions')
            if k not in proto]
    if errs:
        return errs
    if proto['status'] == 'infeasible':
        return [] if (out / 'report.md').exists() else ['status infeasible requires report.md']
    for f in ('report.md', 'source/video.mp4', 'policy.py', proto['actions'].get('path', 'actions.npy')):
        if not (out / f).exists():
            errs.append(f'missing {f}')
    try:
        _, spec, model, data = S1.build(out)
    except Exception as e:
        return errs + [f'scene does not compile: {e}']
    o = model.opt
    if abs(o.timestep - TIMESTEP) > 1e-12 or o.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST \
            or not np.allclose(o.gravity, (0, 0, -9.81)) or o.cone != mujoco.mjtCone.mjCONE_ELLIPTIC or o.impratio != 10:
        errs.append('option must be exactly timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10"')
    if model.nmocap or model.neq or model.ntendon or model.nflex:
        errs.append('no mocap bodies, equality constraints, tendons or flex allowed')
    if model.nu != 16:
        errs.append(f'exactly the 16 supplied instrument actuators are allowed (found {model.nu})')
    ref = reference_instrument()
    R = mujoco.mjtObj
    for P in ARMS:
        tro = _id(model, R.mjOBJ_BODY, P + 'trocar')
        if tro < 0:
            errs.append(f'no body named "{P}trocar"'); continue
        if model.body_parentid[tro] != 0 or not np.allclose(model.body_quat[tro], [1, 0, 0, 0]):
            errs.append(f'{P}trocar must be a direct child of worldbody with identity orientation')
        rcm = proto['instrument'].get(P + 'rcm_pos')
        if rcm is None or not np.allclose(model.body_pos[tro], rcm, atol=1e-6):
            errs.append(f'instrument.{P}rcm_pos must equal the {P}trocar body pos')
        for kind, names, fields in ((R.mjOBJ_BODY, INSTR_BODIES, ('body_pos', 'body_quat', 'body_mass', 'body_gravcomp')),
                                    (R.mjOBJ_JOINT, ARM_JOINTS + ('jaw_a', 'jaw_b'), ('jnt_range', 'jnt_axis')),
                                    (R.mjOBJ_GEOM, INSTR_GEOMS, ('geom_size', 'geom_friction', 'geom_solref', 'geom_priority')),
                                    (R.mjOBJ_ACTUATOR, INSTR_ACTS, ('actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange'))):
            for n in names:
                i, j = _id(model, kind, P + n), _id(ref, kind, P + n)
                if i < 0:
                    errs.append(f'instrument element "{P}{n}" missing'); continue
                for f in fields:
                    if not np.allclose(getattr(model, f)[i], getattr(ref, f)[j], rtol=1e-6, atol=1e-9):
                        errs.append(f'instrument "{P}{n}" field {f} differs from the supplied model')
    instr = set()
    for P in ARMS:
        t = _id(model, R.mjOBJ_BODY, P + 'trocar')
        if t >= 0:
            instr |= S1.subtree(model, t)
    roles = proto['roles']
    for role in ('object', 'target', 'source'):
        if not roles.get(role):
            errs.append(f'roles.{role} must list one body')
    for b in roles.get('object', []):
        i = _id(model, R.mjOBJ_BODY, b)
        if i < 0 or i in instr or model.body_jntnum[i] != 1 or model.body_parentid[i] != 0 \
                or model.jnt_type[model.body_jntadr[i]] != mujoco.mjtJoint.mjJNT_FREE:
            errs.append(f'object "{b}" must be a worldbody child with exactly one free joint'); continue
        if not 1e-4 <= model.body_subtreemass[i] <= 0.05:
            errs.append(f'object "{b}" mass outside [0.1, 50] g')
    for role in ('target', 'source'):
        for b in roles.get(role, []):
            i = _id(model, R.mjOBJ_BODY, b)
            if i < 0 or i in instr or model.body_jntnum[i] != 0:
                errs.append(f'{role} "{b}" must be a static body (no joints)')
    act = proto['actions']
    if act.get('dt') != DT or act.get('format') != 'tcp_pose_2arm':
        errs.append(f'actions must declare dt={DT}, format="tcp_pose_2arm"')
    if errs:
        return errs
    errs += check_actions(model, np.load(out / act.get('path', 'actions.npy')))
    try:
        q, jaw, _ = joint_targets(model, np.load(out / act.get('path', 'actions.npy'))[:1])
        reset(model, data, q[0], jaw[0])
        errs += check_actions(model, S1.load_policy(out)(model, data), 'policy.plan()')
    except Exception as e:
        errs.append(f'policy.plan(model, data) failed: {type(e).__name__}: {e}')
    return errs


def replay(out, use_policy=False):
    import imageio.v2 as imageio
    out = Path(out)
    proto, spec, model, data = S1.build(out)
    cam = proto['camera']
    actions = np.load(out / proto['actions'].get('path', 'actions.npy'))
    if use_policy:
        q, jaw, _ = joint_targets(model, actions[:1])
        reset(model, data, q[0], jaw[0])
        actions = np.asarray(S1.load_policy(out)(model, data), float)
    bodies = proto['roles']['object']
    rec = execute(model, data, actions, bodies, render=S1.make_renderer(model, cam, vig=False))
    rd = out / 'replay'
    rd.mkdir(exist_ok=True)
    imageio.mimsave(rd / 'rollout.mp4', rec['frames'], fps=int(1 / DT), macro_block_size=1)
    src = [f for f in imageio.get_reader(out / 'source' / 'video.mp4')]
    n = max(len(src), len(rec['frames']))
    import cv2
    comp = []
    for k in range(n):
        a = src[min(int(k * len(src) / n), len(src) - 1)]
        b = rec['frames'][min(int(k * len(rec['frames']) / n), len(rec['frames']) - 1)]
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        comp.append(np.concatenate([a, b], axis=1))
    imageio.mimsave(rd / 'compare.mp4', comp, fps=int(1 / DT), macro_block_size=1)
    for qq in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        imageio.imwrite(rd / f'compare_{int(qq * 100):03d}.png', comp[min(int(qq * (n - 1)), n - 1)])
    b = bodies[0]
    p = rec['pose'][b][:, :3]
    held = [c[0] for c in rec['contacts']]
    summ = dict(n_actions=len(actions), duration_s=float(rec['t'][-1]),
                object=dict(initial=p[0].round(4).tolist(), final=p[-1].round(4).tolist(), max_height=float(p[:, 2].max().round(4))),
                held_by_L_s=round(sum(1 for h in held if h & 1) * DT, 2), held_by_R_s=round(sum(1 for h in held if h & 2) * DT, 2),
                tcp_initial=rec['tcp'][0].round(4).tolist(), tcp_final=rec['tcp'][-1].round(4).tolist())
    (rd / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(f'wrote {rd}/rollout.mp4, compare.mp4 (left: source, right: your rollout), compare_*.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('validate', 'replay'))
    ap.add_argument('out')
    ap.add_argument('--policy', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'validate':
        errs = validate(a.out)
        print('OK' if not errs else 'INVALID:\n- ' + '\n- '.join(errs))
        sys.exit(1 if errs else 0)
    replay(a.out, a.policy)


if __name__ == '__main__':
    main()
