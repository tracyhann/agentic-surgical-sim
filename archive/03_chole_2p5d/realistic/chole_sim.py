"""Assemble the photoreal MuJoCo scene from chole_recon.py outputs, simulate the instrument motions, render.

  python chole_sim.py [--k-gb 2.5] [--k-fold 6] [--young 4000]
Writes realistic/chole/{scene.xml, sim.mp4, compare.mp4, sim_frames.npy, flex_traj.npy}.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import mujoco
import cv2
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chole_recon as C  # noqa: E402
sys.path.insert(0, str(C.ROOT / 'tools'))
import surgsim  # noqa: E402

OUT = C.OUT
INTEG = 'Euler'
DT, TS, SETTLE = 0.05, 0.00025, 0.3    # Euler at 0.25 ms: stable with the soft shell, contact and position servos
GRASP_N = 80
RIM_K = None              # border-cut rim: None = pinned; a number = spring (N/m) to its rest position
RIM_DRIVE_K = 20.0        # N/m, springs of the cut rim when the anchors move
MOVE_BED = True           # if the geometry has 'bed_disp' (observed global motion of the organ), its anchors follow it
RIBBON_K, RIBBON_E = 8.0, 800.0     # strand ribbon: per-vertex rest spring (N/m) and shell modulus (Pa)
STEEL = 'specular="1" shininess="0.95"'


def instrument_xml(name, shaft_rgba, jaw_rgba):
    b, a, c = surgsim.instrument_xml(name, 0.005)
    b = b.replace('rgba="0.12 0.12 0.14 1"', f'rgba="{shaft_rgba}" material="{name}_shaft_mat"')
    b = b.replace('rgba="0.62 0.62 0.66 1"', f'rgba="{jaw_rgba}" material="{name}_jaw_mat"')
    b = b.replace('rgba="0.78 0.78 0.82 1"', f'rgba="{jaw_rgba}" material="{name}_jaw_mat"')
    if name == 'grasper_left':               # holds the neck through the grasp constraint: no contact needed
        b = b.replace('contype="2" conaffinity="1"', 'contype="0" conaffinity="0"')
    else:                                    # the probe touches tissue with its tip only; its shaft rides above it
        b = b.replace(f'size="0.00250" mass="0.04"\n              rgba="{shaft_rgba}" material="{name}_shaft_mat" contype="2" conaffinity="1"',
                      f'size="0.00250" mass="0.04"\n              rgba="{shaft_rgba}" material="{name}_shaft_mat" contype="0" conaffinity="0"')
    inner = lambda s: s.split('<mujoco>', 1)[1].rsplit('</mujoco>', 1)[0]
    return inner(b), inner(a), inner(c)


def scene_xml(g, k_gb, k_fold, young, friction=0.3):
    X, tri, px = g['sheet_X'], g['sheet_tri'], g['sheet_px']
    rim = g['sheet_rim'] | g['sheet_bottom']
    in_gb = g['sheet_in_gb']
    N = len(X)
    # volumetric tissue: the textured surface is the front of a slab extruded away from the camera by the local
    # thickness (gallbladder dome up to 2.5 cm, peritoneum 3 mm); the back rests on the bed through soft springs,
    # the rim is pinned front and back, the front is held only by the tissue's own elasticity
    ray = (X - C.CAM_POS) / np.linalg.norm(X - C.CAM_POS, axis=1, keepdims=True)
    XB = X + ray * g['sheet_thick'][:, None]
    bodies = []
    moving = MOVE_BED and 'bed_disp' in g.files and RIM_K is None
    def vbody(name, p, k):
        if k is None and moving and name[:2] in ('fv', 'fb'):
            k = RIM_DRIVE_K                    # driven rim: stiff springs whose rest point follows the observed motion
        if k is None:
            return f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
        j = ''.join(f'<joint type="slide" axis="{a}" stiffness="{k}"/>' for a in ('1 0 0', '0 1 0', '0 0 1'))
        return (f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" gravcomp="1">{j}'
                f'<inertial pos="0 0 0" mass="0.00006" diaginertia="1e-9 1e-9 1e-9"/></body>')
    for i, p in enumerate(X):
        bodies.append(vbody(f'fv{i}', p, (RIM_K if rim[i] else 0)))
    for i, p in enumerate(XB):
        bodies.append(vbody(f'fb{i}', p, RIM_K if rim[i] else (k_gb if in_gb[i] else k_fold)))
    tets = []
    allX = np.r_[X, XB]
    for tr_ in tri:
        v0, v1, v2 = sorted(int(v) for v in tr_)
        for tet in ((v0, v1, v2, v2 + N), (v0, v1, v1 + N, v2 + N), (v0, v0 + N, v1 + N, v2 + N)):
            Pt = allX[list(tet)]
            if np.linalg.det(np.stack([Pt[1] - Pt[0], Pt[2] - Pt[0], Pt[3] - Pt[0]])) < 0:
                tet = (tet[0], tet[2], tet[1], tet[3])
            tets.append(tet)
    uv = ' '.join('%.5f %.5f' % C.uv(u, v) for u, v in np.r_[px, px])   # front and back share the projected texture
    names = ' '.join([f'fv{i}' for i in range(N)] + [f'fb{i}' for i in range(N)])
    R = C.R_CW
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([R[0], -R[1], -R[2]], axis=1).flatten())
    gb_b, gb_a, gb_c = instrument_xml('grasper_left', '0.36 0.44 0.45 1', '0.78 0.80 0.82 1')
    pr_b, pr_a, pr_c = instrument_xml('probe_right', '0.93 0.94 0.95 1', '0.93 0.94 0.95 1')
    meta = json.loads((OUT / 'build_meta.json').read_text())['instruments']
    tro = lambda n, b: (f'<body name="{n}_trocar" pos="{" ".join(f"{x:.5f}" for x in meta[n]["rcm"])}" '
                        f'euler="0 0 {meta[n]["heading"]:.5f}">{b}</body>')
    eqs = ''.join(f'<connect name="grasp{i}" body1="fv0" body2="grasper_left_roll_link" anchor="0 0 0" active="false" '
                  f'solref="0.004 1"/>' for i in range(GRASP_N))
    # strand ribbon: textured 2D flex over the purple band, floating 3.5 mm above the tissue; far ends pinned,
    # neck end tied to the nearest movable tissue vertices, interior held near its rest pose by soft springs
    movable_tissue = [i for i in range(len(X)) if not rim[i]]
    RX, Rtri, Rpx = g['ribbon_X'], g['ribbon_tri'], g['ribbon_px']
    rb = []
    for i, p in enumerate(RX):
        if g['ribbon_pinned'][i]:
            rb.append(f'<body name="rv{i}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>')
        else:
            j = ''.join(f'<joint type="slide" axis="{a}" stiffness="{RIBBON_K}"/>' for a in ('1 0 0', '0 1 0', '0 0 1'))
            rb.append(f'<body name="rv{i}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" gravcomp="1">{j}'
                      f'<inertial pos="0 0 0" mass="0.0001" diaginertia="1e-9 1e-9 1e-9"/></body>')
    strand_xml = ''.join(rb)
    ruv = ' '.join('%.5f %.5f' % C.uv(u, v) for u, v in Rpx)
    strand_flex = (f'<flex name="ribbon" dim="2" radius="0.0004" body="{" ".join(f"rv{i}" for i in range(len(RX)))}" '
                   f'vertex="{" ".join("0 0 0" for _ in RX)}" element="{" ".join(map(str, Rtri.flatten()))}" texcoord="{ruv}" material="m_ribbon">'
                   f'<elasticity young="{RIBBON_E}" poisson="0.4" thickness="0.002" damping="0.002" elastic2d="both"/>'
                   f'<contact condim="3" friction="0.05" solref="0.004 1" margin="0.001" contype="4" conaffinity="2" selfcollide="none"/></flex>')   # touches instruments, not tissue
    for i in np.nonzero(g['ribbon_tied'])[0]:
        dn = np.linalg.norm(X[movable_tissue] - RX[i], axis=1)
        eqs += f'<connect body1="rv{i}" body2="fv{movable_tissue[int(np.argmin(dn))]}" anchor="0 0 0" solref="0.004 1"/>'
    return f"""<mujoco model="chole_realistic">
  <compiler angle="radian" meshdir="{OUT}" texturedir="{OUT}"/>
  <option timestep="{TS}" integrator="{INTEG}" gravity="0 0 -9.81"/>
  <visual><global offwidth="{C.W}" offheight="{C.H}"/><quality shadowsize="2048"/>
    <headlight ambient="0.05 0.05 0.05" diffuse="0.1 0.1 0.1" specular="0 0 0"/></visual>
  <asset>
    <texture name="t_tissue" type="2d" file="tex_tissue.png"/>
    <texture name="t_back" type="2d" file="tex_back.png"/>
    <material name="m_tissue" texture="t_tissue" emission="0.7" specular="0.15" shininess="0.8"/>
    <material name="m_back" texture="t_back" emission="0.7" specular="0.05" shininess="0.3"/>
    <material name="grasper_left_shaft_mat" specular="0.8" shininess="0.9" emission="0.35"/>
    <material name="grasper_left_jaw_mat" specular="1" shininess="0.95" emission="0.35"/>
    <material name="probe_right_shaft_mat" specular="1" shininess="0.98" emission="0.25"/>
    <material name="probe_right_jaw_mat" specular="1" shininess="0.98" emission="0.25"/>
    <texture name="t_ribbon" type="2d" file="tex_ribbon.png"/>
    <material name="m_ribbon" texture="t_ribbon" emission="0.7" specular="0.3" shininess="0.9"/>
    <mesh name="backdrop" file="backdrop.obj"/>
  </asset>
  <worldbody>
    <camera name="endo" pos="{' '.join(f'{x:.5f}' for x in C.CAM_POS)}" quat="{' '.join(f'{x:.6f}' for x in q)}" fovy="{C.FOVY}"/>
    <light name="scope" pos="{' '.join(f'{x:.5f}' for x in C.CAM_POS)}" dir="{' '.join(f'{x:.5f}' for x in R[2])}"
           diffuse="0.4 0.4 0.4" specular="0.6 0.6 0.6" cutoff="60" exponent="1" castshadow="true"/>
    <geom name="backdrop" type="mesh" mesh="backdrop" material="m_back" contype="0" conaffinity="0"/>
    {''.join(bodies)}
    {strand_xml}
    {tro('grasper_left', gb_b)}
    {tro('probe_right', pr_b)}
  </worldbody>
  <deformable>
    <flex name="tissue" dim="3" radius="0.0004" body="{names}" vertex="{' '.join('0 0 0' for _ in range(2 * N))}"
          element="{' '.join(' '.join(map(str, tt)) for tt in tets)}" texcoord="{uv}" material="m_tissue">
      <elasticity young="{young}" poisson="0.45" damping="0.002"/>
      <contact condim="3" friction="{friction}" solref="0.004 1" margin="0.0025" selfcollide="none"/>
    </flex>
    {strand_flex}
  </deformable>
  <equality>{eqs}</equality>
  <actuator>{gb_a}{pr_a}</actuator>
  <contact>{gb_c}{pr_c}</contact>
</mujoco>
"""


def camera_look(img):
    """Endoscope camera response: highlight bloom around over-exposed metal and wet tissue, slight optical softness."""
    f = img.astype(np.float32)
    lum = f.mean(2)
    bright = np.clip((lum - 170) / 60, 0, 1)[..., None] * f
    glow = cv2.GaussianBlur(bright, (0, 0), 3.0)
    out = cv2.GaussianBlur(f, (0, 0), 0.5) + 0.3 * glow
    return np.clip(out, 0, 255).astype(np.uint8)


ARM = ('yaw', 'pitch', 'insertion', 'roll')
NAMES = ('grasper_left', 'probe_right')


def apply(m, d, a):
    for k, p in enumerate(NAMES):
        blk = a[5 * k:5 * k + 5]
        for i, j in enumerate(ARM):
            d.ctrl[m.actuator(f'{p}_{j}').id] = blk[i]
        for j in ('jaw_left', 'jaw_right'):
            d.ctrl[m.actuator(f'{p}_{j}').id] = blk[4]


def set_qpos(m, d, a):
    for k, p in enumerate(NAMES):
        blk = a[5 * k:5 * k + 5]
        for i, j in enumerate(ARM):
            d.qpos[m.joint(f'{p}_{j}').qposadr[0]] = blk[i]
        for j in ('jaw_left', 'jaw_right'):
            d.qpos[m.joint(f'{p}_{j}').qposadr[0]] = blk[4]


def run(k_gb=0.5, k_fold=8.0, young=500.0, friction=0.3, render=True, actions=None, tag='', press=0.0015, compare=True, orbit=None):
    g = np.load(OUT / 'geometry.npz')
    acts = (np.load(OUT / 'actions.npy') if actions is None else np.asarray(actions, float)).copy()
    xml = scene_xml(g, k_gb, k_fold, young, friction)
    (OUT / 'scene.xml').write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    meta = json.loads((OUT / 'build_meta.json').read_text())['instruments']['probe_right']
    P_r, h_r = np.array(meta['rcm']), meta['heading']

    def retarget(k, press):
        # contact-aware probe: never aim the tip more than `press` below the current tissue surface along its ray
        y, p_, ins = acts[k, 5:8]
        tip = C.K.fk(P_r, y, p_, ins, h_r)
        ck = int(np.clip(round(k * DT * C.FPS), 0, len(g['cam_f']) - 1))
        px, zt = C.project(tip, g['cam_R'][ck], g['cam_f'][ck])
        vpx, vz = C.project(d.flexvert_xpos[:len(g['sheet_X'])], g['cam_R'][ck], g['cam_f'][ck])   # tissue only: slide under strands
        close = np.linalg.norm(vpx - px, axis=1) < 12
        if close.any() and zt > vz[close].min() + press:
            ray = (tip - C.CAM_POS) / np.linalg.norm(tip - C.CAM_POS)
            new_tip = C.CAM_POS + ray * (vz[close].min() + press) / (ray @ g['cam_R'][ck][2])
            try:
                acts[k, 5:8] = C.K.ik(P_r, new_tip, h_r)
                if k:
                    acts[k, 5:10] = acts[k - 1, 5:10] + np.clip(acts[k, 5:10] - acts[k - 1, 5:10], -C.SLEW, C.SLEW)
                return 1
            except ValueError:
                return 0
        return 0
    bed_drive = []                         # (qpos address, vertex) of spring-anchored vertices: back layer and cut rim
    if MOVE_BED and 'bed_disp' in g.files:
        for i in range(len(g['sheet_X'])):
            for pre in (('fv', 'fb') if g['sheet_rim'][i] else ('fb',)):
                b = m.body(f'{pre}{i}').id
                if m.body_jntnum[b]:
                    bed_drive.append((m.jnt_qposadr[m.body_jntadr[b]], i))

    def drive_bed(t):
        if not bed_drive:
            return
        x = np.clip(t * C.FPS, 0, len(g['bed_disp']) - 1)
        k0 = int(x)
        k1, w = min(k0 + 1, len(g['bed_disp']) - 1), x - int(x)
        disp = (1 - w) * g['bed_disp'][k0] + w * g['bed_disp'][k1]
        for adr, i in bed_drive:
            m.qpos_spring[adr:adr + 3] = disp[i]
    retarget(0, -0.0015)                   # start with the probe 1.5 mm above the tissue: under the strand, never inside tissue
    set_qpos(m, d, acts[0])
    apply(m, d, acts[0])
    mujoco.mj_forward(m, d)
    # initial clearance: no instrument may start inside tissue or strand; back the probe out along its shaft if it does,
    # then let it slide back in over the first second
    probe_bodies = {m.body(f'probe_right_{b}').id for b in ('roll_link', 'jaw_left', 'jaw_right', 'shaft')}
    def probe_pen():
        return min([d.contact[i].dist for i in range(d.ncon)
                    if any(gid >= 0 and m.geom_bodyid[gid] in probe_bodies for gid in d.contact[i].geom)] + [1.0])
    backoff = 0.0
    while probe_pen() < 0.0003 and backoff < 0.03:
        backoff += 0.001
        acts[0, 7] -= 0.001
        set_qpos(m, d, acts[0])
        mujoco.mj_forward(m, d)
    ramp = min(20, len(acts) - 1)
    for k in range(1, ramp):
        acts[k, 7] -= backoff * (1 - k / ramp)
    apply(m, d, acts[0])
    mujoco.mj_forward(m, d)
    # grasp: the 8 movable sheet vertices nearest the grasper TCP at t=0 follow the grasper jaw body
    tcp = d.site('grasper_left_tcp').xpos.copy()
    free = [i for i in range(len(g['sheet_X'])) if not g['sheet_rim'][i]]
    movable = [f'fv{i}' for i in free if m.body(f'fv{i}').jntnum[0] > 0] + \
              [f'fb{i}' for i in free if m.body(f'fb{i}').jntnum[0] > 0]   # whole neck volume
    near, grasp_gap_mm = [], float('nan')
    if movable:                          # (an ablation may freeze every vertex: then there is nothing to grasp)
        vx = np.array([d.body(b).xpos for b in movable])
        order = np.argsort(np.linalg.norm(vx - tcp, axis=1))
        near = [movable[j] for j in order[:GRASP_N] if np.linalg.norm(vx[j] - tcp) < 0.008] or [movable[order[0]]]
        rl = m.body('grasper_left_roll_link').id
        Rb = d.xmat[rl].reshape(3, 3)
        for e, i in enumerate(near):
            eid = m.equality(f'grasp{e}').id
            m.eq_obj1id[eid] = m.body(i).id
            m.eq_data[eid, 0:3] = 0
            m.eq_data[eid, 3:6] = Rb.T @ (d.body(i).xpos - d.xpos[rl])
            d.eq_active[eid] = 1
        grasp_gap_mm = float(np.linalg.norm(vx[np.argsort(np.linalg.norm(vx - tcp, axis=1))[0]] - tcp) * 1000)
    for _ in range(int(SETTLE / TS)):
        mujoco.mj_step(m, d)
    rend = mujoco.Renderer(m, C.H, C.W) if render else None
    frames, flex = [], []
    cam_id = m.camera('endo').id
    src_t = np.arange(len(g['cam_f'])) / C.FPS

    def set_camera(t):
        k = int(np.clip(round(t * C.FPS), 0, len(g['cam_f']) - 1))
        R = g['cam_R'][k]
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, np.stack([R[0], -R[1], -R[2]], axis=1).flatten())
        m.cam_quat[cam_id] = qq
        m.cam_fovy[cam_id] = np.degrees(2 * np.arctan(C.H / 2 / g['cam_f'][k]))

    orbit_frames = []
    if orbit is not None:                         # an extra free camera (lookat, distance, azimuth, elevation)
        ocam = mujoco.MjvCamera()
        ocam.type = mujoco.mjtCamera.mjCAMERA_FREE
        ocam.lookat[:], ocam.distance, ocam.azimuth, ocam.elevation = orbit

    def log():
        flex.append(d.flexvert_xpos.copy())
        if rend is not None:
            set_camera((len(flex) - 1) * DT)
            rend.update_scene(d, camera='endo')
            frames.append(camera_look(rend.render()))
            if orbit is not None:
                rend.update_scene(d, camera=ocam)
                orbit_frames.append(camera_look(rend.render()))
    log()
    nsub = int(round(DT / TS))
    retargeted = 0
    for k in range(1, len(acts)):
        retargeted += retarget(k, press)
        a, b = acts[k - 1], acts[k]
        for s in range(nsub):
            apply(m, d, a + (b - a) * (s + 1) / nsub)
            drive_bed((k - 1) * DT + (s + 1) * TS)
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise RuntimeError(f'unstable at control step {k}')
        log()
    flex = np.array(flex)
    np.save(OUT / f'flex_traj{tag}.npy', flex)
    np.save(OUT / f'actions_executed{tag}.npy', acts)
    info = dict(probe_initial_backoff_mm=round(backoff * 1000, 1), k_gb=k_gb, k_fold=k_fold, young=young, friction=friction, probe_retargeted_steps=retargeted, grasp_vertices=near, grasp_gap_mm=round(grasp_gap_mm, 2) if grasp_gap_mm == grasp_gap_mm else None,
                tissue_max_disp_mm=round(float(np.linalg.norm(flex - flex[0], axis=2).max()) * 1000, 2),
                tissue_p95_disp_mm=round(float(np.percentile(np.linalg.norm(flex - flex[0], axis=2).max(0), 95)) * 1000, 2))
    if render:
        frames = np.array(frames)
        np.save(OUT / f'sim_frames{tag}.npy', frames)
        if orbit_frames:
            np.save(OUT / f'sim_frames_orbit{tag}.npy', np.array(orbit_frames))
        imageio.mimsave(OUT / f'sim{tag}.mp4', list(frames), fps=20, macro_block_size=1, quality=8)
        if not compare:
            (OUT / f'sim_info{tag}.json').write_text(json.dumps(info, indent=1))
            return info
        src = [f for f in imageio.get_reader(C.ROOT / 'runs_real/chole_sweep/task/video.mp4')]
        n = len(src)
        comp = [np.concatenate([src[k], frames[min(int(round(k / (n - 1) * (len(frames) - 1))), len(frames) - 1)]], 1) for k in range(n)]
        imageio.mimsave(OUT / 'compare.mp4', comp, fps=C.FPS, macro_block_size=1, quality=8)
        for q in (0, 0.33, 0.66, 1.0):
            imageio.imwrite(OUT / f'compare_{int(q * 100):03d}.png', comp[int(q * (n - 1))])
    (OUT / f'sim_info{tag}.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-gb', type=float, default=0.003)
    ap.add_argument('--k-fold', type=float, default=8.0)
    ap.add_argument('--friction', type=float, default=0.3)
    ap.add_argument('--young', type=float, default=1500.0)
    a = ap.parse_args()
    run(a.k_gb, a.k_fold, a.young, a.friction)
