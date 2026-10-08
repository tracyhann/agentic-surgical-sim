"""surgsim: validate and physically replay a surgical video-to-sim package.

  python surgsim.py validate OUT              structural + rule checks (must print OK and exit 0)
  python surgsim.py replay OUT [--policy]     execute the actions (or policy.plan) under physics, render from the
                                              declared endoscope camera, write OUT/replay/ (rollout.mp4, compare.mp4,
                                              compare_*.png key frames, summary.json)

A completed replay is NOT a task judgement; it only reports what your scene and controls physically did.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import mujoco

TIMESTEP, DT, SETTLE, TAIL = 0.0005, 0.05, 0.25, 1.0
GRAVITY = (0.0, 0.0, -9.81)
ARM_JOINTS = ('yaw', 'pitch', 'insertion', 'roll')
JAWS = ('jaw_left', 'jaw_right')
SLEW = np.array([0.08, 0.08, 0.006, 0.15, 0.007])        # max change per 0.05 s control step
LO = np.array([-1.57, 0.0, 0.0, -3.14, 0.0])
HI = np.array([1.57, 1.4, 0.25, 3.14, 0.007])
INSTR_BODIES = ('yaw_link', 'pitch_link', 'shaft', 'roll_link', 'jaw_left', 'jaw_right')
INSTR_JOINTS = ARM_JOINTS + JAWS
INSTR_GEOMS = ('shaft_geom', 'clevis_geom', 'jaw_left_geom', 'jaw_right_geom')
INSTR_DIR = Path(__file__).resolve().parent.parent / 'instrument'


# ---------------------------------------------------------------- camera
def cam_axes(pos, lookat, up=(0, 0, 1)):
    """OpenCV camera axes (rows: x right, y down, z forward) in world coordinates."""
    z = np.asarray(lookat, float) - np.asarray(pos, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def world_to_cam(cam, p):
    R = cam_axes(cam['pos'], cam['lookat'])
    return (np.asarray(p, float) - np.asarray(cam['pos'], float)) @ R.T


def intrinsics(cam):
    f = cam['height'] / 2 / np.tan(np.radians(cam['fovy_deg']) / 2)
    return np.array([[f, 0, cam['width'] / 2], [0, f, cam['height'] / 2], [0, 0, 1]])


def add_endoscope(spec, cam):
    """Endoscope camera + co-located light (the scope carries its own light)."""
    R = cam_axes(cam['pos'], cam['lookat'])
    mat = np.stack([R[0], -R[1], -R[2]], axis=1)               # mujoco camera: x right, y up, looks along -z
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    spec.worldbody.add_camera(name='endo_eval', pos=list(cam['pos']), quat=list(q), fovy=float(cam['fovy_deg']))
    light = spec.worldbody.add_light(name='endo_light', pos=list(cam['pos']), dir=list(R[2]))
    light.diffuse = [0.9, 0.85, 0.8]
    light.specular = [0.4, 0.4, 0.4]
    light.cutoff = 70
    light.exponent = 2
    light.castshadow = True
    spec.visual.headlight.ambient = [0.15, 0.15, 0.15]
    spec.visual.headlight.diffuse = [0.1, 0.1, 0.1]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.global_.offwidth = max(int(cam['width']), 640)
    spec.visual.global_.offheight = max(int(cam['height']), 480)


def vignette(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    g = np.clip(1.15 - 0.55 * r ** 2, 0.25, 1.0)
    return (img * g[..., None]).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- loading
def load_protocol(out):
    return json.loads((Path(out) / 'protocol.json').read_text())


def build(out, cam=None, edit=None):
    """Compile the package scene (+ endoscope camera). `edit(spec)` may modify the spec before compiling."""
    out = Path(out)
    proto = load_protocol(out)
    spec = mujoco.MjSpec.from_file(str(out / proto.get('model_path', 'scene.xml')))
    add_endoscope(spec, cam or proto['camera'])
    if edit:
        edit(spec)
    model = spec.compile()
    return proto, spec, model, mujoco.MjData(model)


def reference_instrument():
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="{TIMESTEP}" integrator="implicitfast"/>
      <worldbody><body name="trocar"><include file="{INSTR_DIR}/instrument_body.xml"/></body></worldbody>
      <actuator><include file="{INSTR_DIR}/instrument_actuators.xml"/></actuator>
      <contact><include file="{INSTR_DIR}/instrument_contacts.xml"/></contact></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def subtree(model, root):
    ids = {root}
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == root:
                ids.add(b)
                break
            p = model.body_parentid[p]
    return ids


# ---------------------------------------------------------------- execution
def _id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


def apply(model, data, a):
    for i, n in enumerate(ARM_JOINTS):
        data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)] = a[i]
    for n in JAWS:
        data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)] = a[4]


def reset(model, data, a0):
    mujoco.mj_resetData(model, data)
    for i, n in enumerate(ARM_JOINTS):
        data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = a0[i]
    for n in JAWS:
        data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = a0[4]
    apply(model, data, a0)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(SETTLE / model.opt.timestep))):
        mujoco.mj_step(model, data)


def execute(model, data, actions, bodies, render=None, a0_reset=True):
    """Run the action stream with linear interpolation of the joint targets inside each control step.
    Records (per 0.05 s) tracked-body poses, TCP and joint state; renders a frame per step if `render` is given."""
    actions = np.asarray(actions, float)
    nsub = int(round(DT / model.opt.timestep))
    if a0_reset:
        reset(model, data, actions[0])
    bid = [_id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
    tcp = _id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    rec = dict(t=[], pose={b: [] for b in bodies}, tcp=[], frames=[])

    def log(t):
        rec['t'].append(t)
        for b, i in zip(bodies, bid):
            rec['pose'][b].append(np.concatenate([data.xpos[i], data.xquat[i]]))
        rec['tcp'].append(data.site_xpos[tcp].copy())
        if render is not None:
            rec['frames'].append(render(data))
    log(0.0)
    seq = list(actions) + [actions[-1]] * int(round(TAIL / DT))
    for k in range(1, len(seq)):
        a, b = seq[k - 1], seq[k]
        for s in range(nsub):
            apply(model, data, a + (b - a) * (s + 1) / nsub)
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f'simulation diverged at control step {k}')
        log(k * DT)
    rec['pose'] = {b: np.array(v) for b, v in rec['pose'].items()}
    rec['tcp'] = np.array(rec['tcp'])
    rec['t'] = np.array(rec['t'])
    rec['n_actions'] = len(actions)
    return rec


def make_renderer(model, cam, vig=True):
    r = mujoco.Renderer(model, int(cam['height']), int(cam['width']))
    cid = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'endo_eval')

    def render(data):
        r.update_scene(data, camera=cid)
        return vignette(r.render()) if vig else r.render()
    return render


# ---------------------------------------------------------------- validation
def check_actions(a, name='actions'):
    errs = []
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 5:
        return [f'{name}: shape {a.shape}, expected (T, 5) = [yaw, pitch, insertion, roll, jaw]']
    if not (10 <= len(a) <= 1200):
        errs.append(f'{name}: T={len(a)} outside [10, 1200]')
    if not np.all(np.isfinite(a)):
        return errs + [f'{name}: non-finite values']
    bad = np.where((a < LO - 1e-9) | (a > HI + 1e-9))
    if len(bad[0]):
        errs.append(f'{name}: {len(bad[0])} values outside joint ranges, e.g. row {bad[0][0]} col {bad[1][0]} = {a[bad][0]:.4f}')
    step = np.abs(np.diff(a, axis=0))
    over = np.where(step > SLEW + 1e-9)
    if len(over[0]):
        errs.append(f'{name}: slew limit {SLEW.tolist()} per step exceeded {len(over[0])}x, e.g. row {over[0][0]+1} col {over[1][0]} '
                    f'(|delta|={step[over][0]:.4f})')
    return errs


def load_policy(out):
    p = Path(out) / 'policy.py'
    spec = importlib.util.spec_from_file_location('agent_policy', p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(out)))
    spec.loader.exec_module(mod)
    return mod.plan


def validate(out):
    out = Path(out)
    errs = []
    try:
        proto = load_protocol(out)
    except Exception as e:
        return [f'protocol.json unreadable: {e}']
    for k in ('status', 'task', 'model_path', 'camera', 'instrument', 'roles', 'actions'):
        if k not in proto:
            errs.append(f'protocol.json: missing "{k}"')
    if errs:
        return errs
    if proto['status'] == 'infeasible':
        return [] if (out / 'report.md').exists() else ['status infeasible requires report.md']
    if proto['status'] != 'success':
        errs.append('status must be "success" or "infeasible"')
    cam = proto['camera']
    for k in ('pos', 'lookat', 'fovy_deg', 'width', 'height'):
        if k not in cam:
            errs.append(f'camera: missing "{k}"')
    if errs:
        return errs
    if not 20 <= cam['fovy_deg'] <= 120:
        errs.append('camera.fovy_deg outside [20, 120]')
    for f in ('report.md', 'source/video.mp4', 'policy.py', proto['actions'].get('path', 'actions.npy')):
        if not (out / f).exists():
            errs.append(f'missing {f}')
    try:
        _, spec, model, data = build(out)
    except Exception as e:
        return errs + [f'scene does not compile: {e}']
    if abs(model.opt.timestep - TIMESTEP) > 1e-12 or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        errs.append(f'option must be timestep={TIMESTEP} integrator=implicitfast')
    if not np.allclose(model.opt.gravity, GRAVITY):
        errs.append(f'gravity must be {GRAVITY} (world z is up)')
    if model.nmocap or model.neq or model.ntendon or model.nflex:
        errs.append('no mocap bodies, equality constraints, tendons or flex allowed')
    if model.nu != 6:
        errs.append(f'exactly the 6 supplied instrument actuators are allowed (found {model.nu})')
    tro = _id(model, mujoco.mjtObj.mjOBJ_BODY, 'trocar')
    if tro < 0:
        return errs + ['no body named "trocar"']
    if model.body_parentid[tro] != 0 or not np.allclose(model.body_quat[tro], [1, 0, 0, 0]):
        errs.append('trocar must be a direct child of worldbody with identity orientation')
    if not np.allclose(model.body_pos[tro], proto['instrument'].get('rcm_pos', [np.nan] * 3), atol=1e-6):
        errs.append('instrument.rcm_pos must equal the trocar body pos')
    ref = reference_instrument()
    R = mujoco.mjtObj
    def same(kind, names, fields):
        for n in names:
            i, j = _id(model, kind, n), _id(ref, kind, n)
            if i < 0:
                errs.append(f'instrument element "{n}" missing (include the supplied files unchanged)')
                continue
            for f in fields:
                if not np.allclose(getattr(model, f)[i], getattr(ref, f)[j], rtol=1e-6, atol=1e-9):
                    errs.append(f'instrument "{n}" field {f} differs from the supplied model')
    same(R.mjOBJ_BODY, INSTR_BODIES, ('body_pos', 'body_quat', 'body_mass', 'body_gravcomp'))
    same(R.mjOBJ_JOINT, INSTR_JOINTS, ('jnt_range', 'jnt_axis'))
    same(R.mjOBJ_GEOM, INSTR_GEOMS, ('geom_size', 'geom_friction', 'geom_solref', 'geom_contype', 'geom_conaffinity'))
    same(R.mjOBJ_ACTUATOR, INSTR_JOINTS, ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange'))
    instr = subtree(model, tro)
    roles = proto['roles']
    for role in ('object', 'target'):
        if not roles.get(role):
            errs.append(f'roles.{role} must list at least one body')
    for b in roles.get('object', []):
        i = _id(model, R.mjOBJ_BODY, b)
        if i < 0:
            errs.append(f'roles.object body "{b}" not found')
            continue
        if i in instr:
            errs.append(f'object "{b}" is inside the instrument')
        if model.body_jntnum[i] != 1 or model.jnt_type[model.body_jntadr[i]] != mujoco.mjtJoint.mjJNT_FREE \
                or model.body_parentid[i] != 0:
            errs.append(f'object "{b}" must be a worldbody child with exactly one free joint')
        if not 1e-4 <= model.body_subtreemass[i] <= 0.05:
            errs.append(f'object "{b}" mass {model.body_subtreemass[i]:.5f} kg outside [1e-4, 0.05]')
        gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == i]
        if not any((model.geom_contype[g] & 1) or (model.geom_conaffinity[g] & 2) for g in gs):
            errs.append(f'object "{b}" has no geom that collides with the instrument')
    for b in roles.get('target', []):
        i = _id(model, R.mjOBJ_BODY, b)
        if i < 0:
            errs.append(f'roles.target body "{b}" not found')
        elif i in instr or model.body_jntnum[i] != 0:
            errs.append(f'target "{b}" must be a static body (no joints) outside the instrument')
    act = proto['actions']
    if act.get('dt') != DT or act.get('format') != 'joint_targets':
        errs.append(f'actions must declare dt={DT}, format="joint_targets"')
    try:
        errs += check_actions(np.load(out / act.get('path', 'actions.npy')))
    except Exception as e:
        errs.append(f'actions unreadable: {e}')
    try:
        reset(model, data, np.load(out / act.get('path', 'actions.npy'))[0])
        plan = load_policy(out)(model, data)
        errs += check_actions(plan, 'policy.plan()')
    except Exception as e:
        errs.append(f'policy.plan(model, data) failed: {type(e).__name__}: {e}')
    return errs


# ---------------------------------------------------------------- replay
def replay(out, use_policy=False):
    import imageio.v2 as imageio
    out = Path(out)
    proto, spec, model, data = build(out)
    cam = proto['camera']
    actions = np.load(out / proto['actions'].get('path', 'actions.npy'))
    if use_policy:
        reset(model, data, actions[0])
        actions = np.asarray(load_policy(out)(model, data), float)
    bodies = proto['roles']['object'] + proto['roles']['target']
    rec = execute(model, data, actions, bodies, render=make_renderer(model, cam))
    rd = out / 'replay'
    rd.mkdir(exist_ok=True)
    imageio.mimsave(rd / 'rollout.mp4', rec['frames'], fps=int(1 / DT), macro_block_size=1)
    src = [f for f in imageio.get_reader(out / 'source' / 'video.mp4')]
    n = max(len(src), len(rec['frames']))
    comp = []
    for k in range(n):
        a = src[min(int(k * len(src) / n), len(src) - 1)]
        b = rec['frames'][min(int(k * len(rec['frames']) / n), len(rec['frames']) - 1)]
        if a.shape != b.shape:
            import cv2
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        comp.append(np.concatenate([a, b], axis=1))
    imageio.mimsave(rd / 'compare.mp4', comp, fps=int(1 / DT), macro_block_size=1)
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        imageio.imwrite(rd / f'compare_{int(q * 100):03d}.png', comp[min(int(q * (n - 1)), n - 1)])
    summ = dict(n_actions=len(actions), duration_s=float(rec['t'][-1]), objects={})
    for b in proto['roles']['object']:
        p = rec['pose'][b][:, :3]
        summ['objects'][b] = dict(initial=p[0].round(4).tolist(), final=p[-1].round(4).tolist(),
                                  max_height=float(p[:, 2].max().round(4)),
                                  final_speed=float(np.linalg.norm(p[-1] - p[-2]) / DT))
    for b in proto['roles']['target']:
        summ.setdefault('targets', {})[b] = rec['pose'][b][0, :3].round(4).tolist()
    summ['tcp_initial'], summ['tcp_final'] = rec['tcp'][0].round(4).tolist(), rec['tcp'][-1].round(4).tolist()
    (rd / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(f'wrote {rd}/rollout.mp4, compare.mp4 (left: source, right: your rollout), compare_*.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('validate', 'replay'))
    ap.add_argument('out')
    ap.add_argument('--policy', action='store_true', help='replay policy.plan() instead of actions.npy')
    a = ap.parse_args()
    if a.cmd == 'validate':
        errs = validate(a.out)
        print('OK' if not errs else 'INVALID:\n- ' + '\n- '.join(errs))
        sys.exit(1 if errs else 0)
    replay(a.out, a.policy)


if __name__ == '__main__':
    main()
