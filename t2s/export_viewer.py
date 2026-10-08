"""Export a v2 clip for the interactive 3D page (site/v2, three.js; same scene format as r2s/export.py):

    python -m t2s.export_viewer <clip> [organ_<name>=vNN] [instruments=vNN] [background=vNN] [sim=<round/config>] [title=...] [video=0]
    -> site/v2/data/<clip>/{scene.json, recon4d.txt, sim4d.txt, tex_back.jpg, tex_tissue.jpg, tex_ribbon.jpg, video.mp4}
       and the clip's entry in site/v2/data/index.json

Layers: backdrop = the background agent's fused surface (textured); organ = the organ agent's surface (video
texture from its start frame), animated by its 4D ('recon4d') or by the assembly's simulation ('sim4d'); ribbons =
the background agent's bodies as convex hulls in their own colour; instruments = the instrument agent's rigid
models posed per frame about their ports; scope camera per frame. One step per video frame. One clip per process
(memory-light).
"""
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import mujoco
import numpy as np

from r2s.export import quant, r, LO, HI
from . import data as D, views2, instruments as INS

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site' / 'v2' / 'data'
TITLES = dict(chole_a='胆囊减压（镜头 A）', liver_s4='肝左叶尖端缝合', chole_derot='胆囊复位（暂定）')


def latest(clip, kind, name=None):
    d = D.OUT / clip / kind / name if name else D.OUT / clip / kind
    vs = sorted(p.name for p in d.glob('v*') if (p / 'model.npz').exists())
    return vs[-1] if vs else None


def instrument_geoms(ins, n):
    """Geoms (type, size, rgba) of all instruments and their world pose per frame (x y z, qx qy qz qw)."""
    frags = INS.mjcf_from_npz_data(ins) if hasattr(INS, 'mjcf_from_npz_data') else None
    names = [str(s) for s in ins['names']]
    if frags is None:
        frags = {nm: INS.mjcf(nm, INS.tool_from_npz(ins, nm), ins[f'{nm}__port'], ins[f'{nm}__R0']) for nm in names}
    m = mujoco.MjModel.from_xml_string(INS.standalone_xml(frags))
    d = mujoco.MjData(m)
    gids = [g for g in range(m.ngeom) if m.geom_rgba[g][3] > 0]
    rows = []
    for k in range(n):
        for nm in names:
            t = INS.tool_from_npz(ins, nm)
            INS.set_qpos(m, d, nm, np.asarray(ins[f'{nm}__joints'][k], float), t['jawed'], t['jaw_mode'])
        mujoco.mj_kinematics(m, d)
        row = []
        for g in gids:
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[g])
            row += list(d.geom_xpos[g]) + [q[1], q[2], q[3], q[0]]
        rows.append(row)
    geoms = [dict(type=mujoco.mjtGeom(m.geom_type[g]).name.replace('mjGEOM_', '').lower(), size=r(m.geom_size[g]),
                  rgba=r(m.geom_rgba[g], 3)) for g in gids]
    return geoms, r(rows, 5)


def main(argv):
    clip = argv[0]
    kv = dict(a.split('=', 1) for a in argv[1:] if '=' in a)
    out = SITE / clip
    out.mkdir(parents=True, exist_ok=True)
    V = views2.load(clip)
    n, H, W = V.n, V.H, V.W
    # organ (first organ with a model)
    od = sorted(p for p in (D.OUT / clip / 'organs').glob('*') if latest(clip, 'organs', p.name))[0]
    over = kv.get(f'organ_{od.name}') or latest(clip, 'organs', od.name)
    z = np.load(od / over / 'model.npz')
    F = z['faces']
    surf = np.unique(F)
    remap = -np.ones(len(z['rest_verts']), int)
    remap[surf] = np.arange(len(surf))
    ref = int(z['start_frame']) if 'start_frame' in z.files else 0
    q, _ = V.project(z['verts4d'][ref].astype(float), ref)
    uv_org = np.c_[np.clip(q[surf, 0] / W, 0, 1), 1 - np.clip(q[surf, 1] / H, 0, 1)]
    val = V.valid if V.valid.ndim == 2 else V.valid[ref]          # outside the scope image: nearest tissue colour, not black
    bad = cv2.dilate((~val).astype(np.uint8), np.ones((9, 9), np.uint8))
    _, lab = cv2.distanceTransformWithLabels(bad, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(bad == 0)
    lut = np.zeros((lab.max() + 1, 2), int)
    lut[lab[ys, xs]] = np.c_[ys, xs]
    texo = cv2.blur(V.frames[ref][lut[lab][..., 0], lut[lab][..., 1]], (9, 9))
    texo[bad == 0] = V.frames[ref][bad == 0]
    iio.imwrite(out / 'tex_tissue.jpg', texo, quality=86)
    # background surface + bodies
    bver = kv.get('background') or latest(clip, 'background')
    zb = np.load(D.OUT / clip / 'background' / bver / 'model.npz')
    tex = cv2.cvtColor(cv2.imread(str(D.OUT / clip / 'background' / bver / 'texture.png')), cv2.COLOR_BGR2RGB)
    iio.imwrite(out / 'tex_back.jpg', tex, quality=86)
    uvb = zb['uv'].astype(float)
    backdrop = dict(pos=r(zb['rest_verts']), uv=r(np.c_[uvb[:, 0], uvb[:, 1]]), index=zb['faces'].astype(int).ravel().tolist())
    from scipy.spatial import ConvexHull
    ribbons, rib_pos, colors = [], [], []
    names_b = [str(s) for s in zb['body_names']] if 'body_names' in zb.files else []
    for i, nm in enumerate(names_b):
        Xb, pc = zb[f'body{i}_verts'].astype(float), zb[f'body{i}_piece']
        col = zb[f'body{i}_rgb'].reshape(-1, 3).mean(0) if f'body{i}_rgb' in zb.files else np.array([170, 80, 80])
        for p in np.unique(pc):
            P = Xb[pc == p]
            if len(P) > 400:                       # a light hull: the piece's own hull vertices, thinned
                hv = np.unique(ConvexHull(P).simplices)
                P = P[hv[np.linspace(0, len(hv) - 1, min(len(hv), 90)).astype(int)]]
            h = ConvexHull(P)
            vi = np.unique(h.simplices)
            rm = -np.ones(len(P), int)
            rm[vi] = np.arange(len(vi))
            tri = h.simplices.copy()                 # outward winding (for shading)
            nrm = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
            flip = (nrm * h.equations[:, :3]).sum(1) < 0
            tri[flip] = tri[flip][:, ::-1]
            ribbons.append(dict(n=len(vi), index=rm[tri].ravel().tolist(), color=len(colors), name=nm))
            rib_pos.append(P[vi])
        colors.append(col)
    atlas = np.zeros((16, 16 * max(1, len(colors)), 3), np.uint8)
    for j, c in enumerate(colors):
        atlas[:, 16 * j:16 * (j + 1)] = c
    iio.imwrite(out / 'tex_ribbon.jpg', atlas, quality=92)
    for rb in ribbons:
        u = (16 * rb.pop('color') + 8) / atlas.shape[1]
        rb['uv'] = r(np.tile([u, 0.5], (rb['n'], 1)))
    static = np.concatenate(rib_pos) if rib_pos else np.zeros((0, 3))
    # instruments
    iver = kv.get('instruments') or latest(clip, 'instruments')
    ins = np.load(D.OUT / clip / 'instruments' / iver / 'model.npz')
    geoms, gtraj = instrument_geoms(ins, n)
    # conditions
    conditions = {}

    def add(key, X4, metrics):
        pos = np.concatenate([X4[:, surf], np.repeat(static[None], n, 0)], 1)
        (out / f'{key}.txt').write_text(quant(pos))
        conditions[key] = dict(file=f'{key}.txt', steps=n, n_surface=len(surf), uv=r(uv_org), index=remap[F].ravel().tolist(),
                               geoms=geoms, geom_traj=gtraj, metrics=metrics,
                               max_disp_mm=round(float(np.linalg.norm(X4 - X4[:1], axis=2).max()) * 1000, 1))
    qj = json.loads((od / over / 'quality.json').read_text()) if (od / over / 'quality.json').exists() else {}
    add('recon4d', z['verts4d'].astype(float), dict(note='4D reconstruction (organ agent %s)' % over))
    if 'sim' in kv:
        rd = D.OUT / clip / 'rounds' / kv['sim']
        tr = np.load(rd / 'traj.npz')
        f0, f1 = [int(v) for v in tr['frames']]
        Xs = tr[f'organ_{od.name}'].astype(float)
        X4 = np.concatenate([np.repeat(Xs[:1], f0, 0), Xs, np.repeat(Xs[-1:], n - 1 - f1, 0)])
        ev = json.loads((rd / 'metrics.json').read_text())['eval'][od.name]
        add('sim4d', X4, {k: ev.get(k) for k in ('sim_iou', 'sim_iou_2d', 'recon_iou', 'motion_explained', 'inverted_tets_max', 'err_vs_recon_mm')})
    # scope camera per frame
    eq, ef, ep = [], [], []
    for k in range(n):
        qq = V.cam.mj_quat(V.R[k])
        eq += [qq[1], qq[2], qq[3], qq[0]]
        ef.append(float(np.degrees(2 * np.arctan(H / 2 / V.f[k]))))
        ep += list(V.pos[k])
    mid = z['verts4d'][n // 2].astype(float)
    pivot = mid.mean(0)
    right, down, fwd = V.R[n // 2]                 # the free view starts beside the scope, with the image's up as up
    look = np.cos(np.radians(30)) * fwd + np.sin(np.radians(30)) * right + 0.15 * down
    look /= np.linalg.norm(look)
    dist = 1.5 * float(np.linalg.norm(pivot - V.pos[n // 2]))
    scene = dict(name=clip, title=kv.get('title') or TITLES.get(clip, clip), dt=1 / V.fps, fps=V.fps, steps=n, size=[W, H], quant=[LO, HI],
                 backdrop=backdrop, ribbons=ribbons, conditions=conditions,
                 endo=dict(pos=r(ep), quat=r(eq, 6), fov=r(ef, 3)),
                 ports=[r(ins[f'{nm}__port']) for nm in [str(s) for s in ins['names']]],
                 view=dict(target=r(pivot), position=r(pivot - dist * look), up=r(-down, 4)),
                 versions=dict(organ={od.name: over}, instruments=iver, background=bver))
    (out / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')))
    if kv.get('video', '1') != '0':
        w = iio.get_writer(out / 'video.mp4', fps=V.fps, macro_block_size=1, quality=7, output_params=['-g', '5', '-movflags', '+faststart'])
        for f in V.frames:
            w.append_data(f[:H // 2 * 2, :W // 2 * 2])
        w.close()
    ip = SITE / 'index.json'
    idx = [e for e in (json.loads(ip.read_text()) if ip.exists() else []) if e['name'] != clip]
    idx.append(dict(name=clip, title=scene['title'], conditions=list(conditions), steps=n, size=[W, H]))
    order = ['chole_a', 'liver_s4', 'chole_derot']
    idx.sort(key=lambda e: order.index(e['name']) if e['name'] in order else 9)
    ip.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    for p in sorted(out.iterdir()):
        print(f'{p.stat().st_size / 1e6:6.2f} MB  {p.name}')


if __name__ == '__main__':
    main(sys.argv[1:])
