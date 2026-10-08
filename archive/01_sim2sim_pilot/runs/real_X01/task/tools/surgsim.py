"""surgsim v2: validate and physically replay a video-to-sim package (one or more laparoscopic instruments).

  python surgsim.py validate OUT              structural + rule checks (must print OK and exit 0)
  python surgsim.py replay OUT [--policy]     execute actions.npy (or policy.plan) under physics, render from the declared
                                              camera, write OUT/replay/ (rollout.mp4, compare.mp4, compare_*.png, summary.json)

The task configuration (instrument names and sizes, rule profile, source video format) is read from task_config.json
next to this file. A completed replay is NOT a task judgement; it only reports what your scene and controls did.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'task_config.json').read_text()) if (HERE / 'task_config.json').exists() else \
    dict(profile='rigid', instruments=[dict(name='tool', shaft_d=0.005)], fps=20)
TIMESTEP, DT, SETTLE, TAIL = 0.0005, 0.05, 0.25, 1.0
GRAVITY = (0.0, 0.0, -9.81)
ARM = ('yaw', 'pitch', 'insertion', 'roll')
SLEW = np.array([0.08, 0.08, 0.006, 0.15, 0.007])        # max change per 0.05 s control step, per instrument
LO = np.array([-1.57, 0.0, 0.0, -3.14, 0.0])
HI = np.array([1.57, 1.4, 0.25, 3.14, 0.007])
NAMES = [i['name'] for i in CONFIG['instruments']]


# ---------------------------------------------------------------- instrument generator
def instrument_xml(p, shaft_d):
    r = shaft_d / 2
    jaw = 'friction="1.5 0.02 0.0002" condim="4" priority="1" solref="0.001 1" solimp="0.95 0.99 0.0005" contype="2" conaffinity="1"'
    body = f"""<mujoco>
  <!-- Include INSIDE <body name="{p}_trocar" pos="x y z" euler="0 0 heading">. The trocar origin is the RCM. -->
  <body name="{p}_yaw_link" gravcomp="1">
    <joint name="{p}_yaw" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.02" armature="0.0005"/>
    <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>
    <body name="{p}_pitch_link" gravcomp="1">
      <joint name="{p}_pitch" type="hinge" axis="-1 0 0" range="0 1.4" damping="0.02" armature="0.0005"/>
      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>
      <body name="{p}_shaft" gravcomp="1">
        <joint name="{p}_insertion" type="slide" axis="0 1 0" range="0 0.25" damping="1" armature="0.01"/>
        <geom name="{p}_shaft_geom" type="cylinder" fromto="0 -0.30 0 0 -0.003 0" size="{r:.5f}" mass="0.04"
              rgba="0.12 0.12 0.14 1" contype="2" conaffinity="1"/>
        <body name="{p}_roll_link" gravcomp="1">
          <joint name="{p}_roll" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="0.005" armature="0.0002"/>
          <geom name="{p}_clevis_geom" type="cylinder" fromto="0 -0.003 0 0 0.003 0" size="{r + 0.0002:.5f}" mass="0.004"
                rgba="0.62 0.62 0.66 1" contype="2" conaffinity="1"/>
          <site name="{p}_tcp" pos="0 0.019 0" size="0.0008" rgba="1 1 0 0"/>
          <body name="{p}_jaw_left" pos="0 0.012 0" gravcomp="1">
            <joint name="{p}_jaw_left" type="slide" axis="-1 0 0" range="0 0.007" damping="0.2" armature="0.001"/>
            <geom name="{p}_jaw_left_geom" type="box" pos="-0.0013 0 0" size="0.0012 0.010 0.0018" mass="0.002"
                  rgba="0.78 0.78 0.82 1" {jaw}/>
          </body>
          <body name="{p}_jaw_right" pos="0 0.012 0" gravcomp="1">
            <joint name="{p}_jaw_right" type="slide" axis="1 0 0" range="0 0.007" damping="0.2" armature="0.001"/>
            <geom name="{p}_jaw_right_geom" type="box" pos="0.0013 0 0" size="0.0012 0.010 0.0018" mass="0.002"
                  rgba="0.78 0.78 0.82 1" {jaw}/>
          </body>
        </body>
      </body>
    </body>
  </body>
</mujoco>
"""
    act = f"""<mujoco>
  <!-- Include INSIDE <actuator>. Position servos; ctrl = joint target. -->
  <position name="{p}_yaw" joint="{p}_yaw" kp="4" kv="0.15" ctrlrange="-1.57 1.57" forcerange="-1 1"/>
  <position name="{p}_pitch" joint="{p}_pitch" kp="4" kv="0.15" ctrlrange="0 1.4" forcerange="-1 1"/>
  <position name="{p}_insertion" joint="{p}_insertion" kp="300" kv="8" ctrlrange="0 0.25" forcerange="-4 4"/>
  <position name="{p}_roll" joint="{p}_roll" kp="0.5" kv="0.01" ctrlrange="-3.14 3.14" forcerange="-0.2 0.2"/>
  <position name="{p}_jaw_left" joint="{p}_jaw_left" kp="60" kv="0.5" ctrlrange="0 0.007" forcerange="-0.06 0.06"/>
  <position name="{p}_jaw_right" joint="{p}_jaw_right" kp="60" kv="0.5" ctrlrange="0 0.007" forcerange="-0.06 0.06"/>
</mujoco>
"""
    con = f"""<mujoco>
  <!-- Include INSIDE <contact>. -->
  <exclude body1="{p}_jaw_left" body2="{p}_jaw_right"/>
  <exclude body1="{p}_jaw_left" body2="{p}_roll_link"/>
  <exclude body1="{p}_jaw_right" body2="{p}_roll_link"/>
</mujoco>
"""
    return body, act, con


def write_instruments(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for i in CONFIG['instruments']:
        b, a, c = instrument_xml(i['name'], i['shaft_d'])
        (d / f"{i['name']}_body.xml").write_text(b)
        (d / f"{i['name']}_actuators.xml").write_text(a)
        (d / f"{i['name']}_contacts.xml").write_text(c)


def instr_elements(p):
    return dict(bodies=[f'{p}_{b}' for b in ('yaw_link', 'pitch_link', 'shaft', 'roll_link', 'jaw_left', 'jaw_right')],
                joints=[f'{p}_{j}' for j in ARM + ('jaw_left', 'jaw_right')],
                geoms=[f'{p}_{g}' for g in ('shaft_geom', 'clevis_geom', 'jaw_left_geom', 'jaw_right_geom')])


def reference_instruments():
    tmp = HERE / '_ref_instruments'
    write_instruments(tmp)
    wb = ''.join(f'<body name="{n}_trocar"><include file="{tmp}/{n}_body.xml"/></body>' for n in NAMES)
    ac = ''.join(f'<include file="{tmp}/{n}_actuators.xml"/>' for n in NAMES)
    co = ''.join(f'<include file="{tmp}/{n}_contacts.xml"/>' for n in NAMES)
    return mujoco.MjModel.from_xml_string(f'<mujoco><compiler angle="radian"/><option timestep="{TIMESTEP}"/>'
                                          f'<worldbody>{wb}</worldbody><actuator>{ac}</actuator><contact>{co}</contact></mujoco>')


# ---------------------------------------------------------------- camera
def cam_axes(pos, lookat, up=(0, 0, 1)):
    """OpenCV camera axes (rows: x right, y down, z forward) in world coordinates."""
    z = np.asarray(lookat, float) - np.asarray(pos, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def world_to_cam(cam, p):
    return (np.asarray(p, float) - np.asarray(cam['pos'], float)) @ cam_axes(cam['pos'], cam['lookat']).T


def project(cam, p):
    """World points (..., 3) -> pixel coordinates (..., 2) for the declared pinhole camera."""
    c = world_to_cam(cam, p)
    f = cam['height'] / 2 / np.tan(np.radians(cam['fovy_deg']) / 2)
    return np.stack([f * c[..., 0] / c[..., 2] + cam['width'] / 2, f * c[..., 1] / c[..., 2] + cam['height'] / 2], -1)


def add_camera(spec, cam):
    R = cam_axes(cam['pos'], cam['lookat'])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([R[0], -R[1], -R[2]], axis=1).flatten())
    spec.worldbody.add_camera(name='eval_cam', pos=list(cam['pos']), quat=list(q), fovy=float(cam['fovy_deg']))
    spec.visual.global_.offwidth = max(int(cam['width']), 640)
    spec.visual.global_.offheight = max(int(cam['height']), 480)


# ---------------------------------------------------------------- loading
def load_protocol(out):
    return json.loads((Path(out) / 'protocol.json').read_text())


def build(out, edit=None):
    out = Path(out)
    proto = load_protocol(out)
    spec = mujoco.MjSpec.from_file(str(out / proto.get('model_path', 'scene.xml')))
    add_camera(spec, proto['camera'])
    if edit:
        edit(spec)
    model = spec.compile()
    return proto, spec, model, mujoco.MjData(model)


def subtree(model, root):
    ids = set()
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == root:
                ids.add(b)
                break
            p = model.body_parentid[p]
    return ids


def _id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


# ---------------------------------------------------------------- execution
def apply(model, data, a):
    for k, p in enumerate(NAMES):
        blk = a[5 * k:5 * k + 5]
        for i, j in enumerate(ARM):
            data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{p}_{j}')] = blk[i]
        for j in ('jaw_left', 'jaw_right'):
            data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{p}_{j}')] = blk[4]


def reset(model, data, a0):
    mujoco.mj_resetData(model, data)
    for k, p in enumerate(NAMES):
        blk = a0[5 * k:5 * k + 5]
        for i, j in enumerate(ARM):
            data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{p}_{j}')]] = blk[i]
        for j in ('jaw_left', 'jaw_right'):
            data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{p}_{j}')]] = blk[4]
    apply(model, data, a0)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(SETTLE / model.opt.timestep))):
        mujoco.mj_step(model, data)


def execute(model, data, actions, bodies=(), render=None):
    """Run the action stream (targets linearly interpolated inside each 0.05 s step); record per step the poses of
    `bodies`, every instrument TCP, and (optionally) a rendered frame."""
    actions = np.asarray(actions, float)
    nsub = int(round(DT / model.opt.timestep))
    reset(model, data, actions[0])
    bid = [_id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
    tcp = [_id(model, mujoco.mjtObj.mjOBJ_SITE, f'{p}_tcp') for p in NAMES]
    rec = dict(t=[], pose={b: [] for b in bodies}, tcp={p: [] for p in NAMES}, frames=[])

    def log(t):
        rec['t'].append(t)
        for b, i in zip(bodies, bid):
            rec['pose'][b].append(np.concatenate([data.xipos[i], data.xquat[i]]))
        for p, i in zip(NAMES, tcp):
            rec['tcp'][p].append(data.site_xpos[i].copy())
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
    rec['tcp'] = {p: np.array(v) for p, v in rec['tcp'].items()}
    rec['t'] = np.array(rec['t'])
    rec['n_actions'] = len(actions)
    return rec


def make_renderer(model, cam, segmentation=False):
    r = mujoco.Renderer(model, int(cam['height']), int(cam['width']))
    if segmentation:
        r.enable_segmentation_rendering()
    cid = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'eval_cam')

    def render(data):
        r.update_scene(data, camera=cid)
        return r.render().copy()
    return render


# ---------------------------------------------------------------- validation
def check_actions(a, name='actions'):
    n = 5 * len(NAMES)
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != n:
        return [f'{name}: shape {a.shape}, expected (T, {n}) = [yaw, pitch, insertion, roll, jaw] per instrument in order {NAMES}']
    errs = []
    if not (10 <= len(a) <= 2400):
        errs.append(f'{name}: T={len(a)} outside [10, 2400]')
    if not np.all(np.isfinite(a)):
        return errs + [f'{name}: non-finite values']
    lo, hi, sl = np.tile(LO, len(NAMES)), np.tile(HI, len(NAMES)), np.tile(SLEW, len(NAMES))
    bad = np.where((a < lo - 1e-9) | (a > hi + 1e-9))
    if len(bad[0]):
        errs.append(f'{name}: {len(bad[0])} values outside joint ranges, e.g. row {bad[0][0]} col {bad[1][0]} = {a[bad][0]:.4f}')
    over = np.where(np.abs(np.diff(a, axis=0)) > sl + 1e-9)
    if len(over[0]):
        errs.append(f'{name}: per-step slew limit exceeded {len(over[0])}x, e.g. row {over[0][0] + 1} col {over[1][0]}')
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
    try:
        proto = load_protocol(out)
    except Exception as e:
        return [f'protocol.json unreadable: {e}']
    errs = [f'protocol.json: missing "{k}"' for k in ('status', 'task', 'model_path', 'camera', 'instruments', 'roles', 'actions')
            if k not in proto]
    if errs:
        return errs
    if proto['status'] == 'infeasible':
        return [] if (out / 'report.md').exists() else ['status infeasible requires report.md']
    if proto['status'] != 'success':
        errs.append('status must be "success" or "infeasible"')
    cam = proto['camera']
    errs += [f'camera: missing "{k}"' for k in ('pos', 'lookat', 'fovy_deg', 'width', 'height') if k not in cam]
    if errs:
        return errs
    if not 10 <= cam['fovy_deg'] <= 120:
        errs.append('camera.fovy_deg outside [10, 120]')
    if [i.get('name') for i in proto['instruments']] != NAMES:
        errs.append(f'instruments must be listed in the order {NAMES}')
    for f in ('report.md', 'policy.py', proto['actions'].get('path', 'actions.npy')):
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
    if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL or model.opt.impratio != 1:
        errs.append('keep the default friction cone (pyramidal) and impratio (1)')
    if model.nmocap:
        errs.append('no mocap bodies allowed')
    if model.nu != 6 * len(NAMES):
        errs.append(f'exactly the {6 * len(NAMES)} supplied instrument actuators are allowed (found {model.nu})')
    ref = reference_instruments()
    R = mujoco.mjtObj
    instr = set()
    for ins in proto['instruments']:
        p = ins['name']
        tro = _id(model, R.mjOBJ_BODY, f'{p}_trocar')
        if tro < 0:
            errs.append(f'no body named "{p}_trocar"')
            continue
        instr |= subtree(model, tro) | {tro}
        h = float(ins.get('heading', 0.0))
        q = np.array([np.cos(h / 2), 0, 0, np.sin(h / 2)])
        if model.body_parentid[tro] != 0 or not (np.allclose(model.body_quat[tro], q, atol=1e-6) or np.allclose(model.body_quat[tro], -q, atol=1e-6)):
            errs.append(f'{p}_trocar must be a worldbody child rotated only about z by instruments[].heading ({h:.4f} rad)')
        if not np.allclose(model.body_pos[tro], ins.get('rcm_pos', [np.nan] * 3), atol=1e-6):
            errs.append(f'instruments[{p}].rcm_pos must equal the {p}_trocar body pos')
        el = instr_elements(p)
        for kind, names, fields in ((R.mjOBJ_BODY, el['bodies'], ('body_pos', 'body_quat', 'body_mass', 'body_gravcomp')),
                                    (R.mjOBJ_JOINT, el['joints'], ('jnt_range', 'jnt_axis')),
                                    (R.mjOBJ_GEOM, el['geoms'], ('geom_size', 'geom_friction', 'geom_solref', 'geom_contype', 'geom_conaffinity')),
                                    (R.mjOBJ_ACTUATOR, el['joints'], ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange'))):
            for n in names:
                i, j = _id(model, kind, n), _id(ref, kind, n)
                if i < 0:
                    errs.append(f'instrument element "{n}" missing (include the supplied files unchanged)')
                    continue
                for f in fields:
                    if not np.allclose(getattr(model, f)[i], getattr(ref, f)[j], rtol=1e-6, atol=1e-9):
                        errs.append(f'instrument "{n}" field {f} differs from the supplied model')
    soft = CONFIG['profile'] == 'soft'
    if not soft and (model.neq or model.ntendon or model.nflex):
        errs.append('rigid profile: no equality constraints, tendons or flex allowed')
    if soft:   # anatomy may be attached to static anatomy, never to an instrument
        for e in range(model.neq):
            t, o1, o2 = model.eq_type[e], model.eq_obj1id[e], model.eq_obj2id[e]
            bodies = []
            if t in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
                bodies = [o1, o2] if model.eq_objtype[e] == R.mjOBJ_BODY else \
                    [model.site_bodyid[o1], model.site_bodyid[o2] if o2 >= 0 else 0]
            elif t == mujoco.mjtEq.mjEQ_JOINT:
                bodies = [model.jnt_bodyid[o1]] + ([model.jnt_bodyid[o2]] if o2 >= 0 else [])
            if any(b in instr for b in bodies):
                errs.append('an equality constraint touches an instrument: tissue may move only through contact')
        for t in range(model.ntendon):
            for w in range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t]):
                ob = model.wrap_objid[w]
                wt = model.wrap_type[w]
                b = model.jnt_bodyid[ob] if wt == mujoco.mjtWrap.mjWRAP_JOINT else \
                    model.site_bodyid[ob] if wt == mujoco.mjtWrap.mjWRAP_SITE else model.geom_bodyid[ob] if ob >= 0 else 0
                if b in instr:
                    errs.append('a tendon touches an instrument')
    roles = proto['roles']
    if not roles.get('object'):
        errs.append('roles.object must list at least one body (or flex, soft profile)')
    for b in roles.get('object', []):
        i = _id(model, R.mjOBJ_BODY, b)
        if i < 0 and soft and _id(model, R.mjOBJ_FLEX, b) >= 0:
            continue
        if i < 0:
            errs.append(f'roles.object "{b}" not found')
            continue
        if i in instr:
            errs.append(f'object "{b}" is inside an instrument')
        if not soft:
            if model.body_jntnum[i] != 1 or model.jnt_type[model.body_jntadr[i]] != mujoco.mjtJoint.mjJNT_FREE or model.body_parentid[i] != 0:
                errs.append(f'object "{b}" must be a worldbody child with exactly one free joint')
            if not 1e-4 <= model.body_subtreemass[i] <= 0.05:
                errs.append(f'object "{b}" mass {model.body_subtreemass[i]:.5f} kg outside [1e-4, 0.05]')
    for b in roles.get('target', []):
        i = _id(model, R.mjOBJ_BODY, b)
        if i < 0:
            errs.append(f'roles.target body "{b}" not found')
        elif i in instr or model.body_jntnum[i] != 0:
            errs.append(f'target "{b}" must be a static body (no joints) outside the instruments')
    act = proto['actions']
    if act.get('dt') != DT or act.get('format') != 'joint_targets':
        errs.append(f'actions must declare dt={DT}, format="joint_targets"')
    try:
        a = np.load(out / act.get('path', 'actions.npy'))
        errs += check_actions(a)
        reset(model, data, a[0])
        errs += check_actions(load_policy(out)(model, data), 'policy.plan()')
    except Exception as e:
        errs.append(f'actions / policy.plan(model, data) failed: {type(e).__name__}: {e}')
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
    bodies = [b for b in proto['roles'].get('object', []) + proto['roles'].get('target', [])
              if _id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
    rec = execute(model, data, actions, bodies, render=make_renderer(model, cam))
    rd = out / 'replay'
    rd.mkdir(exist_ok=True)
    imageio.mimsave(rd / 'rollout.mp4', rec['frames'], fps=int(1 / DT), macro_block_size=1)
    src = [f for f in imageio.get_reader(out / 'source' / 'video.mp4')]
    n = max(len(src), len(rec['frames']))
    comp = []
    import cv2
    for k in range(n):
        a = src[min(int(k * len(src) / n), len(src) - 1)]
        b = rec['frames'][min(int(k * len(rec['frames']) / n), len(rec['frames']) - 1)]
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        comp.append(np.concatenate([a, b], axis=1))
    imageio.mimsave(rd / 'compare.mp4', comp, fps=int(1 / DT), macro_block_size=1)
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        imageio.imwrite(rd / f'compare_{int(q * 100):03d}.png', comp[min(int(q * (n - 1)), n - 1)])
    summ = dict(n_actions=len(actions), duration_s=float(rec['t'][-1]), bodies={},
                tcp={p: dict(initial=v[0].round(4).tolist(), final=v[-1].round(4).tolist()) for p, v in rec['tcp'].items()})
    for b in bodies:
        p = rec['pose'][b][:, :3]
        summ['bodies'][b] = dict(initial=p[0].round(4).tolist(), final=p[-1].round(4).tolist(), max_height=float(p[:, 2].max().round(4)))
    (rd / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(f'wrote {rd}/rollout.mp4, compare.mp4 (left: source, right: your rollout, time-normalised), compare_*.png')


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
