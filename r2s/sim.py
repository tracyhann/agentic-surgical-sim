"""MuJoCo scene from a built clip + one organ condition; simulation of the recovered instrument motion.

  organ       3D flex (tetrahedra), textured with the projected frame-0 canvas; vertices on the bed side hang on
              springs whose rest points follow the observed organ motion; vertices cut by the image border are held
  strands     2D textured flex ribbons; far ends fixed, organ end tied to the nearest organ vertex
  instruments position-servoed laparoscopic instruments through their fitted ports; a holding instrument holds the
              organ through grasp constraints, a touching one acts by contact (aimed at most `press` into the tissue)
Explicit Euler (flex elasticity is not integrated by the implicit integrators).
"""
import json
import numpy as np
import mujoco
import cv2
import imageio.v2 as imageio
from .config import load_json, save_json
from .scene import Canvas
from . import instruments as INS

PARAMS = dict(young=300.0, poisson=0.45, k_back=0.5, k_drive=20.0, friction=0.3, ts=0.000125, dt=0.05, settle=0.3,
              grasp_n=80, grasp_r=0.008, grasp_ramp=0.25, press=0.0015, ribbon_k=8.0, ribbon_young=800.0, vertex_mass=6e-5)
COLORS = {'dark': ('0.36 0.44 0.45 1', '0.55 0.58 0.60 1'), 'bright': ('0.93 0.94 0.95 1', '0.93 0.94 0.95 1'),
          'davinci': ('0.74 0.79 0.86 1', '0.80 0.82 0.85 1')}


def load_scene(clip):
    z = np.load(clip.scene / 'scene.npz')
    S = {k: z[k] for k in z.files}
    S['meta'] = load_json(clip.scene / 'scene.json')
    S['ribbons'] = [{k.split('_', 1)[1]: S[k] for k in S if k.startswith(f'ribbon{j}_')} for j in range(int(S['n_ribbons']))]
    return S


def canvas_of(clip, S):
    return Canvas(clip.camera(), S['cam_R'][0], float(S['cam_f'][0]), int(S['M']))


def bed_displacements(cv, organ, bedT, bed3d=None):
    """Per frame and driven vertex: where its spring rest point moves (the observed image translation at its depth,
    or, for a moving scope, the observed 3D translation)."""
    drive = np.nonzero(organ['anchors'] | organ['pinned'])[0]
    if bed3d is not None and np.abs(bed3d).max() > 0:
        return drive, np.repeat(bed3d[:, None, :], len(drive), 1).astype(np.float32)
    X = organ['X'][drive]
    _, z = cv.project(X)
    q = organ['px'][drive]
    out = np.zeros((len(bedT), len(drive), 3), np.float32)
    for k, T in enumerate(bedT):
        out[k] = cv.unproject(q[:, 0] + T[0], q[:, 1] + T[1], z) - X
    return drive, out


def scene_xml(clip, S, organ, P, cond_dir):
    cv = canvas_of(clip, S)
    cam = cv.cam
    drive_bed = bool(np.abs(S['bedT']).max() > 0.5 or ('bed3d' in S and np.abs(S['bed3d']).max() > 1e-4))
    bodies = []

    def vbody(name, p, k, mass):
        if k is None:
            return f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
        j = ''.join(f'<joint type="slide" axis="{a}" stiffness="{k}"/>' for a in ('1 0 0', '0 1 0', '0 0 1'))
        return (f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" gravcomp="1">{j}'
                f'<inertial pos="0 0 0" mass="{mass}" diaginertia="1e-9 1e-9 1e-9"/></body>')
    for i, p in enumerate(organ['X']):
        if organ['pinned'][i]:
            k = P['k_drive'] if drive_bed else None
        elif organ['anchors'][i]:
            k = P['k_back']
        else:
            k = 0
        bodies.append(vbody(f'ov{i}', p, k, P['vertex_mass']))
    uv = ' '.join('%.5f %.5f' % cv.uv(u, v) for u, v in organ['px'])
    names = ' '.join(f'ov{i}' for i in range(len(organ['X'])))
    organ_flex = (f'<flex name="organ" dim="3" radius="0.0004" body="{names}" vertex="{" ".join("0 0 0" for _ in organ["X"])}" '
                  f'element="{" ".join(" ".join(map(str, t)) for t in organ["tets"])}" texcoord="{uv}" material="m_tissue">'
                  f'<elasticity young="{P["young"]}" poisson="{P["poisson"]}" damping="0.002"/>'
                  f'<contact condim="3" friction="{P["friction"]}" solref="0.004 1" margin="0.0025" selfcollide="none"/></flex>')
    movable = np.nonzero(~organ['pinned'])[0]
    eqs = []
    rib_xml, rib_flex = [], []
    for j, R in enumerate(S['ribbons']):
        for i, p in enumerate(R['X']):
            rib_xml.append(vbody(f'r{j}v{i}', p, None if R['pinned'][i] else P['ribbon_k'], 1e-4))
        ruv = ' '.join('%.5f %.5f' % cv.uv(u, v) for u, v in R['px'])
        rib_flex.append(f'<flex name="ribbon{j}" dim="2" radius="0.0004" body="{" ".join(f"r{j}v{i}" for i in range(len(R["X"])))}" '
                        f'vertex="{" ".join("0 0 0" for _ in R["X"])}" element="{" ".join(map(str, R["tri"].flatten()))}" texcoord="{ruv}" material="m_ribbon">'
                        f'<elasticity young="{P["ribbon_young"]}" poisson="0.4" thickness="0.002" damping="0.002" elastic2d="both"/>'
                        f'<contact condim="3" friction="0.05" solref="0.004 1" margin="0.001" contype="4" conaffinity="2" selfcollide="none"/></flex>')
        for i in (np.nonzero(R['tied'])[0] if P.get('ribbon_tie', True) else []):
            dn = np.linalg.norm(organ['X'][movable] - R['X'][i], axis=1)
            eqs.append(f'<connect body1="r{j}v{i}" body2="ov{movable[int(np.argmin(dn))]}" anchor="0 0 0" solref="0.004 1"/>')
    inst_meta = S['meta']['instruments']
    ib, ia, ic, mats = [], [], [], []
    for ins in clip.instruments:
        name = ins['name']
        shaft, jaw = COLORS[ins.get('color', 'bright')]
        b, a, c = INS.mjcf(name, ins['shaft_d'], shaft, jaw, bool(ins.get('holds')))
        rcm, heading = inst_meta[name]['rcm'], inst_meta[name]['heading']
        ib.append(f'<body name="{name}_trocar" pos="{rcm[0]:.5f} {rcm[1]:.5f} {rcm[2]:.5f}" euler="0 0 {heading:.5f}">{b}</body>')
        ia.append(a)
        ic.append(c)
        mats.append(f'<material name="{name}_shaft_mat" specular="0.9" shininess="0.95" emission="0.3"/>'
                    f'<material name="{name}_jaw_mat" specular="1" shininess="0.95" emission="0.3"/>')
        if ins.get('holds'):
            eqs += [f'<connect name="{name}_grasp{i}" body1="ov0" body2="{name}_roll_link" anchor="0 0 0" active="false" solref="0.004 1"/>'
                    for i in range(P['grasp_n'])]
    q0 = cam.mj_quat(S['cam_R'][0])
    n_patch = int(S['meta'].get('backdrop_patches', 0))
    if n_patch:      # moving scope: one textured patch per keyframe, each in its own place
        back_assets = ''.join(f'<texture name="t_back{j}" type="2d" file="tex_back_{j}.png"/>'
                              f'<material name="m_back{j}" texture="t_back{j}" emission="0.7" specular="0.05" shininess="0.3"/>'
                              f'<mesh name="backdrop{j}" file="backdrop_{j}.obj"/>' for j in range(n_patch))
        back_geoms = ''.join(f'<geom name="backdrop{j}" type="mesh" mesh="backdrop{j}" material="m_back{j}" contype="0" conaffinity="0"/>'
                             for j in range(n_patch))
    else:
        back_assets = '<mesh name="backdrop" file="backdrop.obj"/>'
        back_geoms = '<geom name="backdrop" type="mesh" mesh="backdrop" material="m_back" contype="0" conaffinity="0"/>'
    return f"""<mujoco model="{clip.name}">
  <compiler angle="radian" meshdir="{clip.scene}" texturedir="{clip.scene}"/>
  <option timestep="{P['ts']}" integrator="Euler" gravity="0 0 -9.81"/>
  <visual><global offwidth="{cam.W}" offheight="{cam.H}"/><quality shadowsize="2048"/>
    <headlight ambient="0.05 0.05 0.05" diffuse="0.1 0.1 0.1" specular="0 0 0"/></visual>
  <asset>
    <texture name="t_tissue" type="2d" file="tex_tissue.png"/>
    <texture name="t_back" type="2d" file="tex_back.png"/>
    <texture name="t_ribbon" type="2d" file="tex_ribbon.png"/>
    <material name="m_tissue" texture="t_tissue" emission="0.7" specular="0.15" shininess="0.8"/>
    <material name="m_back" texture="t_back" emission="0.7" specular="0.05" shininess="0.3"/>
    <material name="m_ribbon" texture="t_ribbon" emission="0.7" specular="0.3" shininess="0.9"/>
    {''.join(mats)}
    {back_assets}
  </asset>
  <worldbody>
    <camera name="endo" pos="{cam.pos[0]:.5f} {cam.pos[1]:.5f} {cam.pos[2]:.5f}" quat="{' '.join(f'{x:.6f}' for x in q0)}" fovy="{cam.fovy:.4f}"/>
    <light name="scope" pos="{cam.pos[0]:.5f} {cam.pos[1]:.5f} {cam.pos[2]:.5f}" dir="{' '.join(f'{x:.5f}' for x in cam.R[2])}"
           diffuse="0.4 0.4 0.4" specular="0.6 0.6 0.6" cutoff="60" exponent="1" castshadow="false"/>
    {back_geoms}
    {''.join(bodies)}
    {''.join(rib_xml)}
    {''.join(ib)}
  </worldbody>
  <deformable>{organ_flex}{''.join(rib_flex)}</deformable>
  <equality>{''.join(eqs)}</equality>
  <actuator>{''.join(ia)}</actuator>
  <contact>{''.join(ic)}</contact>
</mujoco>
"""


def camera_look(img):
    """Endoscope response: bloom around over-exposed metal and wet tissue, slight optical softness."""
    f = img.astype(np.float32)
    bright = np.clip((f.mean(2) - 170) / 60, 0, 1)[..., None] * f
    out = cv2.GaussianBlur(f, (0, 0), 0.5) + 0.3 * cv2.GaussianBlur(bright, (0, 0), 3.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply(m, d, clip, a, ctrl=True):
    for k, ins in enumerate(clip.instruments):
        p, blk = ins['name'], a[5 * k:5 * k + 5]
        for i, j in enumerate(INS.ARM):
            if ctrl:
                d.ctrl[m.actuator(f'{p}_{j}').id] = blk[i]
            else:
                d.qpos[m.joint(f'{p}_{j}').qposadr[0]] = blk[i]
        for j in ('jaw_left', 'jaw_right'):
            if ctrl:
                d.ctrl[m.actuator(f'{p}_{j}').id] = blk[4]
            else:
                d.qpos[m.joint(f'{p}_{j}').qposadr[0]] = blk[4]


def cam_pose(S, k):
    """(R, f, pos) of video frame k from a loaded scene (pos None for a scope that only rotates)."""
    return S['cam_R'][k], S['cam_f'][k], (S['cam_pos'][k] if 'cam_pos' in S else None)


def set_camera(m, cam, S, t, fps):
    k = int(np.clip(round(t * fps), 0, len(S['cam_f']) - 1))
    cid = m.camera('endo').id
    m.cam_quat[cid] = cam.mj_quat(S['cam_R'][k])
    m.cam_fovy[cid] = np.degrees(2 * np.arctan(cam.H / 2 / S['cam_f'][k]))
    if 'cam_pos' in S:
        m.cam_pos[cid] = S['cam_pos'][k]


def run(clip, cond, organ, params=None, render=True, actions=None, tag='', log=print, retries=3):
    """Simulate; if the explicit integration goes unstable, halve the time step and start again."""
    P = dict(PARAMS, **clip.get('sim', {}).get('params', {}), **(params or {}))
    for attempt in range(retries + 1):
        try:
            return _run(clip, cond, organ, P, render, actions, tag, log)
        except RuntimeError as e:
            if attempt == retries:
                raise
            log(f'[sim] {e}; retrying with time step {P["ts"] / 2 * 1000:.4f} ms')
            P = dict(P, ts=P['ts'] / 2)


def _run(clip, cond, organ, P, render, actions, tag, log):
    S = load_scene(clip)
    cv = canvas_of(clip, S)
    cam = cv.cam
    out = clip.cond_dir(cond)
    acts = (S['actions'] if actions is None else np.asarray(actions, float)).copy()
    xml = scene_xml(clip, S, organ, P, out)
    (out / f'scene{tag}.xml').write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    n_org = len(organ['X'])
    fps = clip['fps']
    meta = S['meta']['instruments']
    touch = [(k, ins) for k, ins in enumerate(clip.instruments) if not ins.get('holds')]
    hold = [(k, ins) for k, ins in enumerate(clip.instruments) if ins.get('holds')]

    def retarget(k, press):
        """A touching instrument never aims more than `press` below the organ surface along its viewing ray."""
        n_ret = 0
        ck = int(np.clip(round(k * P['dt'] * fps), 0, len(S['cam_f']) - 1))
        Rk, fk, pk = cam_pose(S, ck)
        ck_pos = cam.pos if pk is None else pk
        vpx, vz = cam.project(d.flexvert_xpos[:n_org], Rk, fk, pk)
        for j, ins in touch:
            Pr, h = np.array(meta[ins['name']]['rcm']), meta[ins['name']]['heading']
            tip = INS.fk(Pr, *acts[k, 5 * j:5 * j + 3], h)
            px, zt = cam.project(tip, Rk, fk, pk)
            close = np.linalg.norm(vpx - px, axis=1) < 12
            if close.any() and zt > vz[close].min() + press:
                ray = (tip - ck_pos) / np.linalg.norm(tip - ck_pos)
                new_tip = ck_pos + ray * (vz[close].min() + press) / (ray @ Rk[2])
                try:
                    acts[k, 5 * j:5 * j + 3] = INS.ik(Pr, new_tip, h)
                    if k:
                        acts[k, 5 * j:5 * j + 5] = acts[k - 1, 5 * j:5 * j + 5] + np.clip(acts[k, 5 * j:5 * j + 5] - acts[k - 1, 5 * j:5 * j + 5], -INS.SLEW, INS.SLEW)
                    n_ret += 1
                except ValueError:
                    pass
        return n_ret
    # bed drive
    drive_idx, drive_disp = bed_displacements(cv, organ, S['bedT'], S.get('bed3d'))
    adrs = np.array([m.jnt_qposadr[m.body_jntadr[m.body(f'ov{i}').id]] if m.body_jntnum[m.body(f'ov{i}').id] else -1 for i in drive_idx])
    has = adrs >= 0

    def drive(t):
        if not has.any():
            return
        x = np.clip(t * fps, 0, len(drive_disp) - 1)
        k0 = int(x)
        k1, w = min(k0 + 1, len(drive_disp) - 1), x - int(x)
        disp = (1 - w) * drive_disp[k0] + w * drive_disp[k1]
        for a_, dsp in zip(adrs[has], disp[has]):
            m.qpos_spring[a_:a_ + 3] = dsp
    # start: touching instruments slightly above the tissue and backed out of any initial penetration
    retarget(0, -0.0015)
    apply(m, d, clip, acts[0], ctrl=False)
    apply(m, d, clip, acts[0])
    mujoco.mj_forward(m, d)
    backoff = {}
    for j, ins in touch:
        bodies = {m.body(f'{ins["name"]}_{b}').id for b in ('roll_link', 'jaw_left', 'jaw_right', 'shaft')}
        pen = lambda: min([d.contact[i].dist for i in range(d.ncon) if any(g >= 0 and m.geom_bodyid[g] in bodies for g in d.contact[i].geom)] + [1.0])
        b = 0.0
        while pen() < 0.0003 and b < 0.03:
            b += 0.001
            acts[0, 5 * j + 2] -= 0.001
            apply(m, d, clip, acts[0], ctrl=False)
            mujoco.mj_forward(m, d)
        ramp = min(20, len(acts) - 1)
        for k in range(1, ramp):
            acts[k, 5 * j + 2] -= b * (1 - k / ramp)
        backoff[ins['name']] = round(b * 1000, 1)
    apply(m, d, clip, acts[0])
    mujoco.mj_forward(m, d)
    # grasp: the patch of organ vertices within grasp_r of the surface point closest to the holding instrument's TCP
    # follows its jaws (from its grasp frame). The reconstruction leaves a gap between jaws and tissue, so the patch is
    # drawn into the jaws over grasp_ramp seconds (its closest point ends at the TCP) instead of hanging at a distance.
    grasped = {}
    free = np.nonzero(~organ['pinned'])[0]
    pulls = []                                     # [start time, equality ids, offsets now, offsets in the jaws]

    def grasp(ins):
        name = ins['name']
        tcp = d.site(f'{name}_tcp').xpos.copy()
        vx = d.flexvert_xpos[free]
        order = np.argsort(np.linalg.norm(vx - tcp, axis=1))
        p0 = vx[order[0]]
        near = [free[o] for o in order if np.linalg.norm(vx[o] - p0) < P['grasp_r']][:P['grasp_n']]
        rl = m.body(f'{name}_roll_link').id
        Rb = d.xmat[rl].reshape(3, 3)
        eids, off0, off1 = [], [], []
        for e, i in enumerate(near):
            eid = m.equality(f'{name}_grasp{e}').id
            m.eq_obj1id[eid] = m.body(f'ov{i}').id
            m.eq_data[eid, 0:3] = 0
            eids.append(eid)
            off0.append(Rb.T @ (d.flexvert_xpos[i] - d.xpos[rl]))
            off1.append(Rb.T @ (d.flexvert_xpos[i] + (tcp - p0) - d.xpos[rl]))
            m.eq_data[eid, 3:6] = off0[-1]
            d.eq_active[eid] = 1
        pulls.append([d.time, np.array(eids), np.array(off0), np.array(off1)])
        grasped[name] = dict(n=len(near), gap_mm=round(float(np.linalg.norm(p0 - tcp)) * 1000, 2),
                             step=len(traj) if traj else 0)

    def pull():
        for g in pulls:
            if g[0] is None:
                continue
            f = min(1.0, (d.time - g[0]) / P['grasp_ramp'])
            m.eq_data[g[1], 3:6] = g[2] + f * (g[3] - g[2])
            if f >= 1.0:
                g[0] = None
    traj = []
    grasp_step = {ins['name']: int(round(ins.get('grasp_frame', 0) / fps / P['dt'])) for _, ins in hold}
    for _, ins in hold:
        if grasp_step[ins['name']] == 0:
            grasp(ins)
    for _ in range(int(P['settle'] / P['ts'])):
        pull()
        mujoco.mj_step(m, d)
    rend = mujoco.Renderer(m, cam.H, cam.W) if render else None
    frames = []

    def log_state():
        traj.append(d.flexvert_xpos.copy())
        if rend is not None:
            set_camera(m, cam, S, (len(traj) - 1) * P['dt'], fps)
            rend.update_scene(d, camera='endo')
            frames.append(camera_look(rend.render()))
    log_state()
    nsub = int(round(P['dt'] / P['ts']))
    n_ret = 0
    for k in range(1, len(acts)):
        for _, ins in hold:
            if grasp_step[ins['name']] == k:
                grasp(ins)
        n_ret += retarget(k, P['press'])
        a, b = acts[k - 1], acts[k]
        for s in range(nsub):
            apply(m, d, clip, a + (b - a) * (s + 1) / nsub)
            drive((k - 1) * P['dt'] + (s + 1) * P['ts'])
            pull()
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise RuntimeError(f'{clip.name}/{cond}: unstable at control step {k}')
        log_state()
    traj = np.array(traj, np.float32)
    org = traj[:, :n_org]
    np.savez_compressed(out / f'traj{tag}.npz', flex=traj, actions=acts, n_organ=n_org)
    info = dict(cond=cond, params=P, n_organ_vertices=n_org, n_tets=int(len(organ['tets'])), grasp=grasped, initial_backoff_mm=backoff,
                retargeted_steps=n_ret, organ_max_disp_mm=round(float(np.linalg.norm(org - org[0], axis=2).max()) * 1000, 2),
                organ_p95_disp_mm=round(float(np.percentile(np.linalg.norm(org - org[0], axis=2).max(0), 95)) * 1000, 2))
    save_json(out / f'sim{tag}.json', info)
    if render:
        imageio.mimsave(out / f'scope{tag}.mp4', frames, fps=int(round(1 / P['dt'])), macro_block_size=1, quality=8)
    log(f'[sim] {clip.name}/{cond}{tag}: {n_org} organ vertices, max displacement {info["organ_max_disp_mm"]} mm, grasp {grasped}')
    return info


def set_state(m, clip, d, flex_k, act):
    """Re-pose a saved state for rendering: vertex bodies from their positions, instruments from their joints."""
    apply(m, d, clip, act, ctrl=False)
    for v, b in enumerate(m.flex_vertbodyid):
        if m.body_jntnum[b]:
            a = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[a:a + 3] = flex_k[v] - m.body_pos[b]
    mujoco.mj_forward(m, d)


def save_organ(path, organ):
    np.savez(path, **{k: v for k, v in organ.items() if isinstance(v, np.ndarray)},
             kind=organ['kind'], fit=json.dumps(organ.get('fit', {})))


def load_organ(path):
    z = np.load(path, allow_pickle=False)
    o = {k: z[k] for k in z.files}
    o['kind'] = str(o['kind'])
    o['fit'] = json.loads(str(o['fit']))
    return o
