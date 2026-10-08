"""Step 7 of v2: assemble a clip's MuJoCo scene from the agents' models, simulate it with the video's instrument motion,
and check it (against the video, the 4D reconstruction, physics and illegal interpenetration).

    python -m t2s.assemble <clip> rNN [organ=gallbladder:v10,...] [instruments=v01] [background=v08] [param=value ...]
    -> outputs/t2s/<clip>/rounds/rNN/: scene.xml, traj.npz, metrics.json, orbit.jpg, sheet_sim.jpg

Boundary conditions come from the scene_spec and the agents' attachment hints, nothing else:
  organ vertices hinted '<structure> [object]' that the spec says the organ is 'suspended by' / 'attached' to -> soft
  springs to where the 4D puts them in the first simulated frame (the far end of that structure is fixed anatomy);
  'opening:<instrument>:<a-b>' -> that instrument does not collide with that organ (it enters through the declared
  opening; the check verifies it enters there and nowhere else); 'hidden_back' -> free (stopped by the background);
  a jawed instrument whose closed jaws are on an organ in the first frame holds the vertices under its jaws.
Contact groups: organs (1) collide with instruments (2) and the static background (4: thin boxes along the fused
background surface, plus the background agent's convex bodies). Flex-heightfield collision does not work in MuJoCo
3.x (tested: a tet block falls 9 mm through), hence boxes.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from r2s import quality as Q
from . import data as D, views2, instruments as INS
from .checks import seg_tri_hits

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(ts=6.25e-5, settle=0.3, vertex_mass=6e-5, damping=0.002, solref=0.01, friction=0.3, k_anchor=2.0, k_foundation=0.5, k_foundation_free=0.05, k_neck=2.0, neck_n=8,
              grasp_r=0.004, grasp_reach_mm=10.0, grasp_soft_tc=0.1, hold_depth='organ', k_puncture=0.0, puncture_r_mm=6.0, grasp_n=40, grasp_ramp=0.25, bg_patches=800, bg_patch_mm=3.0, bg_thick_mm=3.0,
              bg_reach_mm=30.0, rest='start', gravcomp=1, drag=0.0, first_frame=0, last_frame=-1, world_scale=1.0,
              tool_depth='model',    # 'contact': r09 diagnostic, see contact_consistent_joints
              body_collide=1,        # 0: the background agent's convex bodies are not collidable
              young_scale=1.0)       # r07: x the material table's Young's modulus (a tense fluid-filled sac is not a 1 kPa gel)
# instrument types whose closed jaws hold TISSUE (a needle holder holds the needle / the thread: liver_s4 r05 had it
# 'grasp' 19 liver vertices and drag the lobe tip, r08)
HOLDS_TISSUE = ('grasper', 'dissector')
MATERIAL = {'fluid-filled': (1200.0, 0.45), 'solid parenchyma': (3000.0, 0.45), 'spongy/air-filled': (400.0, 0.3),
            'fatty': (800.0, 0.4), None: (1500.0, 0.4)}


def latest(clip, kind, name=None):
    d = D.OUT / clip / kind / name if name else D.OUT / clip / kind
    vs = sorted(p.name for p in d.glob('v*') if (p / 'model.npz').exists())
    return vs[-1] if vs else None


def load(clip, organ_versions, ins_ver, bg_ver):
    spec = json.loads((D.OUT / clip / 'spec' / 'scene_spec.json').read_text())
    organs = {}
    for name, ver in organ_versions.items():
        z = dict(np.load(D.OUT / clip / 'organs' / name / ver / 'model.npz', allow_pickle=True))
        z['version'] = ver
        organs[name] = z
    ins = np.load(D.OUT / clip / 'instruments' / ins_ver / 'model.npz', allow_pickle=False)
    bg = dict(np.load(D.OUT / clip / 'background' / bg_ver / 'model.npz', allow_pickle=True))
    return spec, organs, ins, bg


def _vbody(name, p, k, mass, gravcomp=1, damping=0.0):
    j = ''.join(f'<joint type="slide" axis="{a}" stiffness="{k}" damping="{damping}"/>' for a in ('1 0 0', '0 1 0', '0 0 1'))
    return (f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" gravcomp="{gravcomp}">{j}'
            f'<inertial pos="0 0 0" mass="{mass}" diaginertia="1e-9 1e-9 1e-9"/></body>')


def _quat_from_z(n):
    """Quaternion (w x y z) rotating +z onto unit vector n."""
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    c = float(z @ n)
    if np.linalg.norm(v) < 1e-9:
        return np.array([1, 0, 0, 0.0]) if c > 0 else np.array([0, 1, 0, 0.0])
    q = np.r_[1 + c, v]
    return q / np.linalg.norm(q)


def background_boxes(bg, organ_pts, P, start_pts, rng=np.random.default_rng(0)):
    """Thin static boxes whose front faces lie on the background surface, near where the organs move."""
    X, F = bg['rest_verts'].astype(float), bg['faces']
    from scipy.spatial import cKDTree
    near = cKDTree(organ_pts).query(X)[0] < P['bg_reach_mm'] / 1000
    nrm = np.zeros_like(X)
    fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    for i in range(3):
        np.add.at(nrm, F[:, i], fn)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    toward = bg['ref_pos'][None] - X
    nrm *= np.sign((nrm * toward).sum(1, keepdims=True))          # normals point to the free (camera) side
    idx = np.nonzero(near)[0]
    if len(idx) > P['bg_patches']:                                  # farthest-point-ish subsample of patch centres
        idx = rng.choice(idx, P['bg_patches'], replace=False)
    hs, th = P['bg_patch_mm'] / 1000, P['bg_thick_mm'] / 2000
    geoms = []
    tree = cKDTree(start_pts)
    for j, i in enumerate(idx):
        # a flat box is tangent at its centre only; push it back until no organ vertex of the first frame lies
        # behind its front face within its footprint (otherwise contact shoves the organ at the start)
        near_i = tree.query_ball_point(X[i], np.sqrt(2) * hs + 0.008)
        back = 0.0
        if near_i:
            Y = start_pts[near_i] - X[i]
            h = Y @ nrm[i]                                          # height of organ points above the patch plane
            ok_fp = np.linalg.norm(Y - h[:, None] * nrm[i], axis=1) < np.sqrt(2) * hs + 0.004   # + a tet's size
            if ok_fp.any():
                back = max(0.0, 0.0005 - float(h[ok_fp].min()))
        c = X[i] - nrm[i] * (th + back)
        q = _quat_from_z(nrm[i])
        geoms.append(f'<geom name="bg{j}" type="box" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" quat="{q[0]:.5f} {q[1]:.5f} '
                     f'{q[2]:.5f} {q[3]:.5f}" size="{hs:.4f} {hs:.4f} {th:.4f}" rgba="0.6 0.5 0.45 0.25" contype="4" conaffinity="1" '
                     f'friction="{P["friction"]}"/>')
    return geoms, len(idx)


def body_geoms(bg, collide=True):
    """The background agent's convex bodies as static mesh geoms (each convex piece its own mesh); collide=False:
    shown only (r07: chole_derot's 'fat' body is a scene-wide wedge under the organ, visual QA)."""
    ct, ca = (4, 1) if collide else (0, 0)
    assets, geoms = [], []
    for i, name in enumerate(bg.get('body_names', [])):
        Xb, Fb, pc = bg[f'body{i}_verts'], bg[f'body{i}_faces'], bg[f'body{i}_piece']
        for p in np.unique(pc):
            vi = np.nonzero(pc == p)[0]
            assets.append(f'<mesh name="b{i}_{p}" vertex="{" ".join(f"{v:.5f}" for v in Xb[vi].ravel())}"/>')
            geoms.append(f'<geom name="body{i}_{p}" type="mesh" mesh="b{i}_{p}" rgba="0.7 0.3 0.3 0.6" contype="{ct}" conaffinity="{ca}"/>')
    return assets, geoms


def organ_rules(spec, organ_name, z):
    """Per vertex the strongest rule: 'anchor' (attached to fixed anatomy per the spec) wins over
    ('opening', instrument); 'hidden_back' and anything the spec does not support -> no rule.
    Openings are also returned per instrument (vertex sets), independently of anchors."""
    conn = spec.get('connections', [])
    rules, openings = {}, {}
    for i, a in zip(np.asarray(z.get('attach_idx', [])).tolist(), np.asarray(z.get('attach_to', [])).astype(str).tolist()):
        if a.startswith('opening:'):
            openings.setdefault(a.split(':')[1], set()).add(i)
            rules.setdefault(i, ('opening', a.split(':')[1]))
        elif a == 'hidden_back':
            rules.setdefault(i, ('foundation', a))        # the unseen back rests on surrounding tissue (elastic bed)
        else:
            base = a.split('[')[0].strip().lower()
            if any(c.get('type') in ('suspended by', 'attached', 'stapled', 'sutured') and
                   any(t.strip() and t.strip() in (str(c.get('a', '')) + ' ' + str(c.get('b', ''))).lower() for t in base.split('+'))
                   for c in conn):
                rules[i] = ('anchor', a)
    rules['_openings'] = openings
    return rules


def build(clip, spec, organs, ins, bg, P, f0):
    s = P['world_scale']
    parts, bodies, flexes, eqs, excl_ins = {}, [], [], [], set()
    organ_specs = {o['name']: o for o in spec.get('organs', [])}
    for name, z in organs.items():
        X0 = z['verts4d'][f0].astype(float) * s
        # rest='start': the first simulated frame's shape is the stress-free state (a free organ released from a
        # different rest shape springs back and drifts: 9 mm in 20 frames for chole_a); 'model': the agent's rest shape
        rest = X0.copy() if P['rest'] == 'start' else z['rest_verts'].astype(float) * s
        rules = organ_rules(spec, name, z)
        E, nu = MATERIAL.get(organ_specs.get(name, {}).get('consistency'), MATERIAL[None])
        E *= P['young_scale']
        free_organ = any(c.get('type') == 'free' and name in (str(c.get('a', '')) + str(c.get('b', ''))).lower()
                         for c in spec.get('connections', []))
        k_found = P['k_foundation_free'] if free_organ else P['k_foundation']   # surrounding tissue / embedding
        for i, p in enumerate(rest):
            r = rules.get(i)
            k = P['k_anchor'] if r and r[0] == 'anchor' else (k_found if r and r[0] == 'foundation' else 0)
            bodies.append(_vbody(f'{name}_{i}', p, k, P['vertex_mass'], P['gravcomp'], P['drag']))
        excl_ins |= set(rules['_openings'])
        names = ' '.join(f'{name}_{i}' for i in range(len(rest)))
        el = ' '.join(' '.join(map(str, t)) for t in z['tets'])
        flexes.append(f'<flex name="{name}" dim="3" radius="0.0004" body="{names}" vertex="{" ".join("0 0 0" for _ in rest)}" '
                      f'element="{el}" rgba="0.9 0.78 0.16 1"><elasticity young="{E}" poisson="{nu}" damping="{P["damping"]}"/>'
                      f'<contact condim="3" friction="{P["friction"]}" solref="{P["solref"]} 1" margin="0" contype="1" conaffinity="6" '
                      f'selfcollide="none"/></flex>')
        parts[name] = dict(rest=rest, start=X0, rules=rules, E=E, nu=nu)
    # instruments
    names = [str(n) for n in ins['names']]
    ib, ia, it, ie, ic = [], [], [], [], []
    for nm in names:
        t = INS.tool_from_npz(ins, nm)
        if s != 1.0:
            t = INS.scaled(t, s) if hasattr(INS, 'scaled') else t
        short = nm.replace('instrument_', '')
        coll = not any(short in o or nm in o for o in excl_ins)
        fr = INS.mjcf(nm, t, np.asarray(ins[f'{nm}__port']) * s, ins[f'{nm}__R0'], collide=True,
                      contype=2 if coll else 8, conaffinity=1 if coll else 0)
        ib.append(fr['body']); ia.append(fr['actuator']); it.append(fr['tendon']); ie.append(fr['equality']); ic.append(fr['contact'])
    # a closed tool whose jaws are near (not on) an organ holds a structure without a model (chole_a: the neck):
    # spring links (spatial tendons) from the tool tip to the organ's nearest vertices stand for that tissue
    tendons, link_info = [], {}
    for nm in names:
        J = np.asarray(ins[f'{nm}__joints'], float)
        tl = INS.tool_from_npz(ins, nm)
        if not tl['jawed'] or tl.get('type') not in HOLDS_TISSUE or J[f0, 4] > 0.35 or P['k_neck'] <= 0:
            continue
        tip = np.asarray(ins[f'{nm}__tip'], float)[f0] * s
        for name, pr in parts.items():
            dd = np.linalg.norm(pr['start'] - tip, axis=1)
            if not (P['grasp_r'] < dd.min() < P['grasp_reach_mm'] / 1000):
                continue
            near = np.argsort(dd)[:P['neck_n']]
            for i in near:
                j = next(j for j, b in enumerate(bodies) if f'name="{name}_{i}"' in b)
                bodies[j] = bodies[j].replace('</body>', f'<site name="{name}_{i}_s" size="0.0005"/></body>')
                tendons.append(f'<spatial name="neck_{nm}_{i}" stiffness="{P["k_neck"]}" springlength="{dd[i]:.5f}" damping="{P["damping"]}">'
                               f'<site site="{nm}_tip"/><site site="{name}_{i}_s"/></spatial>')
            link_info[nm] = dict(organ=name, n=len(near), length_mm=round(float(dd[near].mean()) * 1000, 1))
            break
    # background
    pts = np.concatenate([np.concatenate([z['verts4d'][::10].reshape(-1, 3) * s for z in organs.values()])])
    start_pts = np.concatenate([z['verts4d'][f0].astype(float) * s for z in organs.values()])
    bgeoms, nbox = background_boxes(dict(bg, rest_verts=bg['rest_verts'] * s, ref_pos=bg['ref_pos'] * s), pts, P, start_pts)
    bassets, bbodies = body_geoms({k: (v * s if k.endswith('_verts') else v) for k, v in bg.items()}, bool(P['body_collide']))
    xml = f"""<mujoco model="t2s {clip}">
  <compiler angle="radian"/>
  <option timestep="{P['ts']}" integrator="Euler" gravity="0 0 -9.81"/>
  <visual><global offwidth="960" offheight="540"/></visual>
  <asset>{''.join(bassets)}</asset>
  <worldbody>
    {''.join(bgeoms)}{''.join(bbodies)}
    {''.join(bodies)}
    {''.join(ib)}
  </worldbody>
  <deformable>{''.join(flexes)}</deformable>
  <tendon>{''.join(it)}{''.join(tendons)}</tendon>
  <equality>{''.join(ie)}{''.join(f'<connect name="grasp{j}" body1="world" body2="{names[0]}_roll_link" anchor="0 0 0" active="false" solref="0.004 1"/>' for j in range(P['grasp_n'] * max(1, len(names))))}</equality>
  <contact>{''.join(ic)}</contact>
  <actuator>{''.join(ia)}</actuator>
</mujoco>"""
    return xml, parts, names, dict(n_boxes=nbox, excluded_instruments=sorted(excl_ins), neck_links=link_info)


def hold_consistent_joints(ins, names, tools, organs, V, f0, P):
    """A tool's depth along the line of sight is the weakest part of its fit (port depth, monocular depth), and a
    holding tool drags the organ with it: for a jawed tool closed on an organ at the first frame, replace the
    along-view component of its tip motion by that of the tissue it holds (the organ agent's 4D of the held
    vertices), keeping the image-plane motion; then recompute yaw / pitch / insertion about the same port."""
    out = {}
    for nm in names:
        J = np.asarray(ins[f'{nm}__joints'], float).copy()
        if not tools[nm]['jawed'] or J[f0, 4] > 0.35:
            continue
        tip = np.asarray(ins[f'{nm}__tip'], float)
        for name, z in organs.items():
            rec = z['verts4d'].astype(float)
            dd = np.linalg.norm(rec[f0] - tip[f0], axis=1)
            if dd.min() > P['grasp_reach_mm'] / 1000:
                continue
            held = np.argsort(dd)[:P['grasp_n']]
            held = held[dd[held] < max(P['grasp_r'], dd.min() + 0.002)]
            c = rec[:, held].mean(1)
            port, R0 = np.asarray(ins[f'{nm}__port'], float), np.asarray(ins[f'{nm}__R0'], float)
            new = tip.copy()
            for k in range(len(tip)):
                fwd = V.R[k][2]
                dt_tip, dt_c = tip[k] - tip[f0], c[k] - c[f0]
                new[k] = tip[k] - (dt_tip @ fwd - dt_c @ fwd) * fwd
            dvec = new - port
            ln = np.linalg.norm(dvec, axis=1)
            yaw, pitch = INS.angles_of(R0, dvec / ln[:, None])
            J[:, 0], J[:, 1], J[:, 2] = yaw, pitch, ln
            out[nm] = J
            break
    return out


def contact_consistent_joints(ins, names, organs, V, skip, step=0.02, lam_min=0.5):
    """tool_depth='contact' (r09 diagnostic): an instrument's depth along the line of sight is what its fit knows
    least, and t2s.recon_check finds its shaft inside the organ's 4D in a third of the frames of chole_derot. For a
    tool seen in a frame whose shaft would be inside an organ (no opening declared), slide its tip towards the scope
    along its line of sight - image position and port unchanged, so its projected shaft line is unchanged - until
    the shaft clears the organ by its radius; the slide factor is min-filtered and smoothed over time. This uses the
    organ's RECONSTRUCTION for the one tool coordinate the image does not give (as hold_consistent_joints does for a
    holding tool): it tests whether the physics follows once the contact geometry is consistent, it is not a
    prediction from the instruments alone."""
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d
    from .recon_check import inside
    out, info = {}, {}
    n = V.n
    sdf = {}
    for nm in names:
        if any(o in nm for o in skip):
            continue
        J = np.asarray(ins[f'{nm}__joints'], float).copy()
        tip = np.asarray(ins[f'{nm}__tip'], float)
        vis = np.asarray(ins[f'{nm}__visible'], bool)
        port, R0 = np.asarray(ins[f'{nm}__port'], float), np.asarray(ins[f'{nm}__R0'], float)
        rad = float(json.loads(str(ins[f'{nm}__params']))['radius'])
        lam = np.ones(n)
        for k in np.nonzero(vis)[0]:
            for name, z in organs.items():
                X, F = z['verts4d'][k].astype(float), z['faces']
                if (name, k) not in sdf:
                    sdf[(name, k)] = Q.signed_distance(X, F)
                for l in np.arange(1.0, lam_min - 1e-9, -step):
                    T = V.pos[k] + l * (tip[k] - V.pos[k])
                    d = (T - port) / np.linalg.norm(T - port)
                    S = T - np.linspace(0, 0.06, 25)[:, None] * d
                    if not inside(X, F, S).any() and np.abs(sdf[(name, k)](S)).min() >= rad:
                        break
                lam[k] = min(lam[k], l)
        ls = np.minimum(gaussian_filter1d(minimum_filter1d(lam, 5, mode='nearest'), 1.5, mode='nearest'), lam)
        new = V.pos + ls[:, None] * (tip - V.pos)
        dvec = new - port
        ln = np.linalg.norm(dvec, axis=1)
        yaw, pitch = INS.angles_of(R0, dvec / ln[:, None])
        J[:, 0], J[:, 1], J[:, 2] = yaw, pitch, ln
        out[nm] = J
        sl = np.linalg.norm(new - tip, axis=1) * 1000
        info[nm] = dict(frames_slid=int((lam[vis] < 1).sum()), frames_seen=int(vis.sum()), at_limit=int((lam <= lam_min + 1e-9).sum()),
                        slide_mm_median_when_slid=round(float(np.median(sl[lam < 1])), 1) if (lam < 1).any() else 0.0,
                        slide_mm_max=round(float(sl.max()), 1))
    return out, info


def simulate(clip, spec, organs, ins, bg, P, V, log=print):
    n = V.n
    f0 = int(P['first_frame'])
    f1 = V.n - 1 if P['last_frame'] < 0 else int(P['last_frame'])
    xml, parts, names, info = build(clip, spec, organs, ins, bg, P, f0)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    joints = {nm: np.asarray(ins[f'{nm}__joints'], float) for nm in names}
    tools = {nm: INS.tool_from_npz(ins, nm) for nm in names}
    if P['hold_depth'] == 'organ':
        joints.update(hold_consistent_joints(ins, names, tools, organs, V, f0, P))
    if P['tool_depth'] == 'contact':
        cj, info['tool_depth_contact'] = contact_consistent_joints(ins, names, organs, V, {o for p_ in parts.values() for o in p_['rules']['_openings']})
        joints.update(cj)
        log(f"[assemble] tool depth from contact: {info['tool_depth_contact']}")

    def set_instruments(k, ctrl_only=False):
        for nm in names:
            q = joints[nm][k].copy()
            if P['world_scale'] != 1.0:
                q[2] *= P['world_scale']
            acts = [f'{nm}_{j}' for j in ('yaw', 'pitch', 'insertion', 'roll')] + (['%s_jaw' % nm] if tools[nm]['jawed'] else [])
            for a, v in zip(acts, q):
                d.ctrl[m.actuator(a).id] = v
            if not ctrl_only:
                INS.set_qpos(m, d, nm, q, tools[nm]['jawed'], tools[nm]['jaw_mode'])

    # start: instruments at the first frame, organ vertices at their 4D positions there (rest shape = reference)
    set_instruments(f0)
    for name, p in parts.items():
        for i in range(len(p['rest'])):
            b = m.body(f'{name}_{i}')
            a = m.jnt_qposadr[b.jntadr[0]]
            d.qpos[a:a + 3] = p['start'][i] - p['rest'][i]
            if p['rules'].get(i, (None,))[0] in ('anchor', 'foundation'):   # springs rest at the first frame's positions
                m.qpos_spring[a:a + 3] = p['start'][i] - p['rest'][i]
    mujoco.mj_forward(m, d)
    # grasp: closed jaws on tissue in the first frame hold the organ vertices under them
    grasp, eq_used = {}, 0
    for nm in names:
        if not tools[nm]['jawed'] or tools[nm].get('type') not in HOLDS_TISSUE or joints[nm][f0, 4] > 0.35:
            continue
        tip = d.site(f'{nm}_tip').xpos.copy()
        rl = m.body(f'{nm}_roll_link').id
        Rb = d.xmat[rl].reshape(3, 3)
        held = []
        for name, p in parts.items():
            Xc = np.array([d.xpos[m.body(f'{name}_{i}').id] for i in range(len(p['rest']))])
            dd = np.linalg.norm(Xc - tip, axis=1)
            if dd.min() > (P['grasp_reach_mm'] / 1000 if P['k_neck'] <= 0 else P['grasp_r']):   # neck: tendon links
                continue
            r_hold = max(P['grasp_r'], dd.min() + 0.002)     # the held part may be a structure without its own model
            for i in np.argsort(dd)[:P['grasp_n']]:          # (chole_a: the neck); the link to it is then rigid
                if dd[i] < r_hold and eq_used < m.neq:
                    e = m.equality(f'grasp{eq_used}').id
                    m.eq_obj1id[e] = m.body(f'{name}_{i}').id
                    m.eq_obj2id[e] = rl
                    m.eq_data[e, 0:3] = 0
                    m.eq_data[e, 3:6] = Rb.T @ (Xc[i] - d.xpos[rl])
                    if dd.min() > P['grasp_r']:              # the jaws hold an unmodelled structure (the neck): the
                        m.eq_solref[e, 0] = P['grasp_soft_tc']   # link to the organ is that tissue, not a rigid bar
                    d.eq_active[e] = 1
                    held.append((name, int(i)))
                    eq_used += 1
        if held:
            grasp[nm] = held
    for _ in range(int(P['settle'] / P['ts'])):
        mujoco.mj_step(m, d)
    traj = {name: [] for name in parts}
    tips = {nm: [] for nm in names}
    nsub = int(round(1 / V.fps / P['ts']))
    t0 = time.time()

    def log_state():
        for name in parts:
            f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, name)
            a, c = m.flex_vertadr[f], m.flex_vertnum[f]
            traj[name].append(d.flexvert_xpos[a:a + c].copy())
        for nm in names:
            tips[nm].append(d.site(f'{nm}_tip').xpos.copy())
    log_state()
    # declared openings: the punctured wall grips the instrument's shaft but lets it slide in and out - organ
    # vertices at the puncture are pulled toward the shaft axis (perpendicular springs) while the tool is inside
    punct = []
    for nm in names:
        inside = np.asarray(ins[f'{nm}__tip_inside_organ'], bool) if f'{nm}__tip_inside_organ' in ins.files else None
        for name, pr in parts.items():
            ov = [i for o, ii in pr['rules']['_openings'].items() if o in nm for i in ii]
            if ov and inside is not None and P['k_puncture'] > 0:
                adr = np.array([m.jnt_dofadr[m.body(f'{name}_{i}').jntadr[0]] for i in ov])
                fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, name)
                punct.append(dict(tool=nm, organ=name, verts=np.array(ov), adr=adr, inside=inside,
                                  port=np.asarray(ins[f'{nm}__port'], float) * P['world_scale'], va=m.flex_vertadr[fid]))
    engaged = {}

    def puncture_forces(k):
        for j, pc in enumerate(punct):
            d.qfrc_applied[np.r_[pc['adr'], pc['adr'] + 1, pc['adr'] + 2]] = 0
            if not pc['inside'][k]:
                engaged.pop(j, None)
                continue
            tip = d.site(f"{pc['tool']}_tip").xpos
            u = tip - pc['port']
            u /= np.linalg.norm(u)
            X = d.flexvert_xpos[pc['va'] + pc['verts']]
            r = (X - pc['port']) - ((X - pc['port']) @ u)[:, None] * u
            if j not in engaged:                       # engage the opening vertices close to the shaft
                engaged[j] = np.linalg.norm(r, axis=1) < P['puncture_r_mm'] / 1000
            f = -P['k_puncture'] * r * engaged[j][:, None]
            for c in range(3):
                d.qfrc_applied[pc['adr'] + c] = f[:, c]
    for k in range(f0 + 1, f1 + 1):
        set_instruments(k, ctrl_only=True)
        for sub in range(nsub):
            if punct and sub % 4 == 0:
                puncture_forces(k)
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise RuntimeError(f'unstable at frame {k}')
        log_state()
        if (k - f0) % 50 == 0:
            log(f'[assemble] frame {k}/{f1} {time.time() - t0:.0f} s')
    out = {f'organ_{nm}': np.array(v, np.float32) for nm, v in traj.items()}
    out.update({f'tip_{nm}': np.array(v, np.float32) for nm, v in tips.items()})
    return out, xml, dict(info, grasp={k: len(v) for k, v in grasp.items()}, frames=[f0, f1], seconds=round(time.time() - t0, 1)), m, d


def evaluate(clip, V, organs, ins, bg, traj, P, parts_rules):
    s = P['world_scale']
    f0, f1 = traj['frames']
    ks = list(range(f0, f1 + 1, 10))
    res = {}
    from .background import Background
    B = Background(clip, P.get('background_version'))
    for name, z in organs.items():
        sim = traj[f'organ_{name}'] / s                      # back to the views' frame for scoring
        rec = z['verts4d'][f0:f1 + 1].astype(float)
        F = z['faces']
        sim4 = np.zeros_like(z['verts4d'], dtype=float)
        sim4[f0:f1 + 1] = sim
        r = {}
        if name in V.names:
            iou = Q.silhouette_iou(V, sim4, F, name, ks)
            iou2 = Q.silhouette_iou(V, sim4, F, name, ks, visible_only=False)
            iour = Q.silhouette_iou(V, z['verts4d'], F, name, ks)
            thirds = np.array_split(np.arange(len(ks)), 3)
            r.update(sim_iou=round(float(iou.mean()), 4), sim_iou_2d=round(float(iou2.mean()), 4), recon_iou=round(float(iour.mean()), 4),
                     sim_iou_thirds=[round(float(iou[t].mean()), 3) for t in thirds],
                     recon_iou_thirds=[round(float(iour[t].mean()), 3) for t in thirds])
        e = np.linalg.norm(sim - rec, axis=2).mean(1) * 1000
        e0 = np.linalg.norm(rec[:1] - rec, axis=2).mean(1) * 1000
        rules = parts_rules[name]
        free = np.array([i for i in range(len(z['rest_verts'])) if rules.get(i, (None,))[0] not in ('anchor', 'foundation')])
        ef = np.linalg.norm(sim[:, free] - rec[:, free], axis=2).mean() * 1000
        ef0 = np.linalg.norm(rec[:1, free] - rec[:, free], axis=2).mean() * 1000
        Tt = z['tets']
        vol = lambda X: np.einsum('ij,ij->i', np.cross(X[Tt[:, 1]] - X[Tt[:, 0]], X[Tt[:, 2]] - X[Tt[:, 0]]), X[Tt[:, 3]] - X[Tt[:, 0]])
        sgn = np.sign(vol(z['rest_verts'].astype(float)))
        import trimesh
        vml = lambda X: abs(trimesh.Trimesh(X, F, process=False).volume) * 1e6
        r.update(err_vs_recon_mm=round(float(e.mean()), 3), motion_explained=round(float(1 - e.mean() / max(e0.mean(), 1e-9)), 3),
                 free_motion_explained=round(float(1 - ef / max(ef0, 1e-9)), 3), n_anchor=int(sum(1 for k, x in rules.items() if k != '_openings' and x[0] == 'anchor')),
                 inverted_tets_max=int(max((np.sign(vol(sim[k])) != sgn).sum() for k in range(0, len(sim), 5))),
                 volume_ml=dict(rest=round(vml(z['rest_verts'].astype(float)), 2),
                                sim=[round(min(vml(sim[k]) for k in range(0, len(sim), 10)), 2), round(max(vml(sim[k]) for k in range(0, len(sim), 10)), 2)],
                                recon=[round(min(vml(rec[k]) for k in range(0, len(rec), 10)), 2), round(max(vml(rec[k]) for k in range(0, len(rec), 10)), 2)]))
        # interpenetration: organ behind the background surface (along each frame's camera rays)
        behind = [float((B.ray_distance(sim4[k], k) < -0.002).mean()) for k in ks]
        r['organ_behind_background'] = dict(median=round(float(np.median(behind)), 4), max=round(float(np.max(behind)), 4))
        res[name] = r
    # instruments vs organs (outside declared openings) and vs background
    names = [str(nm) for nm in ins['names']]
    viol = {}
    for nm in names:
        tip = traj[f'tip_{nm}'] / s
        inside, through_bg = [], 0
        for name, z in organs.items():
            opening = any(o in nm for o in parts_rules[name]['_openings'])
            for j, k in enumerate(range(f0, f1 + 1, 5)):
                sd = Q.signed_distance(traj[f'organ_{name}'][k - f0] / s, z['faces'])
                d_in = float(sd(tip[k - f0][None])[0])
                if d_in > 0.001:
                    inside.append(dict(frame=k, organ=name, depth_mm=round(d_in * 1000, 2), legal=bool(opening)))
        viol[nm] = dict(tip_inside_organ_frames=len([v for v in inside if not v['legal']]),
                        tip_inside_through_opening_frames=len([v for v in inside if v['legal']]),
                        max_illegal_depth_mm=max([v['depth_mm'] for v in inside if not v['legal']], default=0.0))
    res['instruments'] = viol
    return res


def orbit(m, d, traj, parts, names, path):
    """Final state from three angles (geometry check for the visual QA)."""
    rend = mujoco.Renderer(m, 360, 480)
    centre = np.mean([v[-1].mean(0) for k, v in traj.items() if k.startswith('organ_')], 0)
    imgs = []
    for az in (0, 70, 150):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = centre
        c.distance = 0.12
        c.azimuth = az
        c.elevation = -25
        rend.update_scene(d, camera=c)
        imgs.append(rend.render())
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(imgs, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv):
    clip, rnd = argv[0], argv[1]
    kv = dict(a.split('=', 1) for a in argv[2:] if '=' in a)
    P = dict(PARAMS)
    for k, v in kv.items():
        if k in P:
            P[k] = type(P[k])(v) if not isinstance(P[k], str) else v
    organ_versions = {}
    if 'organ' in kv:
        for item in kv['organ'].split(','):
            nm, ver = item.split(':')
            organ_versions[nm] = ver
    else:
        for d in sorted((D.OUT / clip / 'organs').glob('*')):
            if latest(clip, 'organs', d.name):
                organ_versions[d.name] = latest(clip, 'organs', d.name)
    ins_ver = kv.get('instruments', latest(clip, 'instruments'))
    bg_ver = kv.get('background', latest(clip, 'background'))
    P['background_version'] = bg_ver
    out = D.OUT / clip / 'rounds' / rnd
    out.mkdir(parents=True, exist_ok=True)
    V = views2.load(clip)
    spec, organs, ins, bg = load(clip, organ_versions, ins_ver, bg_ver)
    traj, xml, info, m, d = simulate(clip, spec, organs, ins, bg, P, V)
    traj['frames'] = info['frames']
    (out / 'scene.xml').write_text(xml)
    np.savez_compressed(out / 'traj.npz', **{k: v for k, v in traj.items()})
    rules = {name: organ_rules(spec, name, z) for name, z in organs.items()}
    ev = evaluate(clip, V, organs, ins, bg, traj, P, rules)
    met = dict(clip=clip, round=rnd, versions=dict(organs=organ_versions, instruments=ins_ver, background=bg_ver),
               params=P, info=info, rules={k: dict(anchor=sum(1 for i, x in v.items() if i != '_openings' and x[0] == 'anchor'),
                                         openings={o: len(ii) for o, ii in v['_openings'].items()}) for k, v in rules.items()},
               eval=ev)
    (out / 'metrics.json').write_text(json.dumps(met, indent=1, default=str))
    orbit(m, d, traj, None, None, out / 'orbit.jpg')
    ks = list(range(info['frames'][0], info['frames'][1] + 1, max(1, (info['frames'][1] - info['frames'][0]) // 7)))[:8]
    layers = []
    for name, z in organs.items():
        sim4 = np.zeros_like(z['verts4d'], dtype=float)
        sim4[info['frames'][0]:info['frames'][1] + 1] = traj[f'organ_{name}'] / P['world_scale']
        layers.append((f'{name} (sim)', sim4, z['faces'], (255, 220, 60)))
    Q.contact_sheet(V, layers, ks, out / 'sheet_sim.jpg')
    print(json.dumps(met['eval'], indent=1)[:2500])


if __name__ == '__main__':
    main(sys.argv[1:])
