"""Interpenetration check of a simulated scene: which 3D objects pass through each other where they should not.

    PYTHONPATH=<code root of the run> python t2s/checks.py <run_dir> [--every 5] [--spec scene_spec.json]

Run it with the r2s code that produced the run (v1 runs: PYTHONPATH=medical-agentic-sim). Writes
<run_dir>/interpenetration.json and prints a summary.

Objects: gallbladder (closed surface of the tet mesh), sheet (open shell), duct tubes, backdrop (static surface),
instrument geoms (posed per control step by replaying the run's actions through its scene.xml).
Rules:
  instrument_in_organ          instrument surface points > tol inside a closed organ. Legal only where the scene
                               spec declares an opening (incision / puncture) of that organ for that instrument.
  instrument_through_sheet     an instrument's axis crosses the open sheet (no opening declared)
  instrument_through_backdrop  an instrument's axis crosses the background surface (liver / fat)
  organ_in_organ               duct nodes / free sheet vertices > tol inside the gallbladder
  organ_through_backdrop       gallbladder surface vertices > tol behind the background surface, along view rays
Grasping is not a violation: the held sheet row is excluded from the sheet test for the instrument holding it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

from r2s import views, quality as Q
from r2s import scene4d as S
from r2s import instruments as INS

TOL = 0.001


def seg_tri_hits(P0, P1, X, F, eps=1e-12):
    """Boolean (n_seg,) whether each segment P0->P1 crosses any triangle (Moller-Trumbore, vectorised)."""
    A, B, C = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    e1, e2 = B - A, C - A
    hits = np.zeros(len(P0), bool)
    for i, (p, q) in enumerate(zip(P0, P1)):
        d = q - p
        h = np.cross(d, e2)
        a = (e1 * h).sum(1)
        ok = np.abs(a) > eps
        f = np.zeros_like(a)
        f[ok] = 1.0 / a[ok]
        s = p - A
        u = f * (s * h).sum(1)
        qv = np.cross(s, e1)
        v = f * (qv @ d)
        t = f * (e2 * qv).sum(1)
        hits[i] = bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).any())
    return hits


def geom_samples(m, d, g, n_axis=9, n_ring=8):
    """Surface sample points (k, 3) and axis points (n_axis, 3) of one geom in world coordinates."""
    t = m.geom_type[g]
    s = m.geom_size[g]
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        z = np.linspace(-s[1], s[1], n_axis)
        axis = c + z[:, None] * R[:, 2]
        ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        ring = s[0] * (np.cos(ang)[:, None] * R[:, 0] + np.sin(ang)[:, None] * R[:, 1])
        pts = (axis[:, None] + ring[None]).reshape(-1, 3)
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts = np.vstack([pts, axis[0] - s[0] * R[:, 2], axis[-1] + s[0] * R[:, 2]])
        return pts, axis
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
        pts = c + corners @ R.T
        long = int(np.argmax(s))
        axis = c + np.linspace(-s[long], s[long], n_axis)[:, None] * R[:, long]
        return np.vstack([pts, axis]), axis
    r = s[0]
    pts = c + r * np.vstack([np.eye(3), -np.eye(3)])
    return np.vstack([pts, c]), c[None]


def run_check(run, every=5, spec=None):
    run = Path(run)
    met = json.loads((run / 'metrics.json').read_text())
    P = dict(S.PARAMS, **met['params'])
    V = views.load('chole_a', 'sift', S.CAMS.get(P.get('cams', 'refined'), P.get('cams')))
    tissues = {t: S.load_tissue(t, met['tissues'].get(t)) for t in S.TISSUES}
    tr = np.load(run / 'traj.npz')
    n = len(tr['t'])
    G, M, B = tissues['gallbladder'], tissues['membrane'], tissues['backdrop']
    gb = tr['gallbladder']
    mem = tr['membrane'] if 'membrane' in tr.files else None
    cables = {k[len('part_ducts_'):]: tr[k] for k in tr.files if k.startswith('part_ducts_')}
    Xb, Fb = S.backdrop_behind(B, G), np.asarray(B['faces'])
    Fm = np.asarray(M['faces']) if M is not None else None
    att = S.attachments(M) if M is not None else {}
    held = np.array([i for i, a in att.items() if a not in ('gallbladder', 'backdrop')], int)
    base = [i for i, a in att.items() if a == 'gallbladder']
    free_sheet = np.setdiff1d(np.arange(len(M['rest_verts'])), np.r_[held, base].astype(int)) if M is not None else []
    openings = (spec or {}).get('openings', [])
    # instruments posed from the run's actions
    m = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    d = mujoco.MjData(m)
    tools, _ = S.tool_tracks(V)
    S.counterfactual(V, tools, P)
    acts, _ = S.control_targets(V, tools, P)
    names = [ins['name'] for ins in V.clip.instruments]
    gids = {nm: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(nm) and m.geom_rgba[g][3] > 0]
            for nm in names}
    holds = {ins['name']: bool(ins.get('holds')) for ins in V.clip.instruments}
    rec = {k: [] for k in ('instrument_in_organ', 'instrument_through_sheet', 'instrument_through_backdrop',
                           'organ_in_organ', 'organ_through_backdrop')}
    for k in range(0, n, every):
        a = acts[min(k, len(acts) - 1)]
        for j, ins in enumerate(V.clip.instruments):
            for i, jn in enumerate(INS.ARM):
                d.qpos[m.joint(f"{ins['name']}_{jn}").qposadr[0]] = a[5 * j + i]
        mujoco.mj_kinematics(m, d)
        frame = int(round(tr['t'][k] * V.fps))
        sd = Q.signed_distance(gb[k], G['faces'])
        for nm in names:
            pts, axes = [], []
            for g in gids[nm]:
                p, ax = geom_samples(m, d, g)
                pts.append(p)
                if len(ax) > 1:
                    axes.append(ax)
            pts = np.vstack(pts)
            inside = sd(pts)
            depth = float(inside.max())
            if depth > TOL and not any(o.get('organ') == 'gallbladder' and o.get('instrument') == nm for o in openings):
                rec['instrument_in_organ'].append(dict(frame=frame, instrument=nm, organ='gallbladder',
                                                       depth_mm=round(depth * 1000, 2), n_points=int((inside > TOL).sum())))
            segs0 = np.vstack([ax[:-1] for ax in axes])
            segs1 = np.vstack([ax[1:] for ax in axes])
            if mem is not None and not holds[nm]:
                if seg_tri_hits(segs0, segs1, mem[k], Fm).any():
                    rec['instrument_through_sheet'].append(dict(frame=frame, instrument=nm))
            if seg_tri_hits(segs0, segs1, Xb, Fb).any():
                rec['instrument_through_backdrop'].append(dict(frame=frame, instrument=nm))
        if cables:
            nodes = np.concatenate([c[k][1:] for c in cables.values()])
            fr = float((sd(nodes) > TOL).mean())
            if fr > 0:
                rec['organ_in_organ'].append(dict(frame=frame, what='duct nodes in gallbladder', fraction=round(fr, 3)))
        if mem is not None and len(free_sheet):
            fr = float((sd(mem[k][free_sheet]) > TOL).mean())
            if fr > 0:
                rec['organ_in_organ'].append(dict(frame=frame, what='sheet in gallbladder', fraction=round(fr, 3)))
        msk, zb = Q.render(V, Xb, Fb, frame)
        q, z = V.project(gb[k], frame)
        u, v = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (u >= 0) & (u < V.W) & (v >= 0) & (v < V.H) & (z > 0)
        behind = np.zeros(len(z), bool)
        behind[ok] = msk[v[ok], u[ok]] & (z[ok] > zb[v[ok], u[ok]] + 0.003)
        surf = np.unique(G['faces'])
        fr = float(behind[surf].mean())
        if fr > 0:
            rec['organ_through_backdrop'].append(dict(frame=frame, what='gallbladder surface behind background', fraction=round(fr, 3)))
    checked = len(range(0, n, every))
    summary = {}
    for key, items in rec.items():
        frames = sorted({it['frame'] for it in items})
        summ = dict(frames_with_violation=len(frames), of_frames_checked=checked)
        if items and 'depth_mm' in items[0]:
            for nm in names:
                dd = [it['depth_mm'] for it in items if it['instrument'] == nm]
                if dd:
                    summ[nm] = dict(frames=len(dd), max_depth_mm=max(dd), median_depth_mm=float(np.median(dd)))
        elif items and 'instrument' in items[0]:
            for nm in names:
                c = sum(it['instrument'] == nm for it in items)
                if c:
                    summ[nm] = dict(frames=c)
        elif items:
            whats = sorted({it['what'] for it in items})
            for w in whats:
                fs = [it['fraction'] for it in items if it['what'] == w]
                summ[w] = dict(frames=len(fs), max_fraction=max(fs), median_fraction=float(np.median(fs)))
        if frames:
            summ['first_last_frame'] = [frames[0], frames[-1]]
        summary[key] = summ
    out = dict(run=str(run), tolerance_mm=TOL * 1000, openings_declared=openings, summary=summary, events=rec)
    (run / 'interpenetration.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    every = int(args[args.index('--every') + 1]) if '--every' in args else 5
    spec = json.loads(Path(args[args.index('--spec') + 1]).read_text()) if '--spec' in args else None
    res = run_check(args[0], every, spec)
    print(json.dumps(res['summary'], indent=1, ensure_ascii=False))
