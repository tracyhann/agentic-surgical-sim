"""Export a v2 clip for the interactive 3D page (site/v2/page.html, three.js):

    python -m t2s.export_viewer <clip> [organ_<dir>=vNN] [instruments=vNN] [background=vNN] [sim=<round/config>] [title=...] [video=0]
    -> site/v2/data/<clip>/{scene.json, recon4d.txt, sim4d.txt, tex_back.jpg, tex_tissue.jpg, tex_ribbon.jpg, video.mp4}
       and the clip's entry in site/v2/data/index.json

Every SAM 3 object of the clip is listed in scene.json 'objects' with the 3D model that reconstructs it (r09: all
segmented parts have to be reconstructed): the primary organ (tet model, 4D; 'recon4d' = its fitted 4D, 'sim4d' = the
assembly's simulation), other fitted structures (generic tet models, 4D), the background agent's bodies (convex
pieces, static), labelled regions of the background surface (static), instruments (rigid models posed per frame
about their ports). An object without any model gets kind 'none' and is reported. One step per video frame; one clip
per process (memory-light).
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
TITLES = dict(chole_a='胆囊减压（镜头 A）', liver_s4='肝左叶尖端缝合', chole_derot='胆囊复位')
NAMES_ZH = {
    'gallbladder': '胆囊', 'neck_pedicle': '胆囊颈和蒂（胆囊管、动脉）', 'liver': '肝', 'red strand band': '红色条索', 'blood': '血块',
    'red-brown oval organ (unidentified)': '红褐色椭圆器官（没认出来）', 'peri-gallbladder fat': '胆囊周围脂肪',
    'left triangular ligament': '左三角韧带', 'diaphragm': '膈肌', 'fat / omentum': '脂肪 / 网膜',
    'gauze or sponge at leak site': '漏口处的纱布', 'instrument_grasper': '抓钳', 'instrument_suction_cannula': '吸引管',
    'instrument_grasper_top': '抓钳（上方）', 'instrument_dissector_right': '分离钳（右侧）',
    'instrument_needle_holder_R': '持针器（右）', 'instrument_needle_holder_L': '持针器（左）'}
TYPE_ZH = dict(grasper='抓钳', dissector='分离钳', needle_holder='持针器', suction='吸引管')
PRIMARY = '#D9B840'
PALETTE = ['#C77DB5', '#6FB7E8', '#8FD18A', '#E8915A', '#9E8CF0', '#5FD4C4', '#E0718A', '#B8C46A', '#D6A77A']
STEEL = ['#C9CFD1', '#8FA6B5', '#B9B29E', '#7F8C9A']


def rgb(h):
    return [round(int(h[i:i + 2], 16) / 255, 3) for i in (1, 3, 5)]


def latest(clip, kind, name=None):
    d = D.OUT / clip / kind / name if name else D.OUT / clip / kind
    vs = sorted(p.name for p in d.glob('v[0-9][0-9]') if (p / 'model.npz').exists())
    return vs[-1] if vs else None


def instrument_geoms(ins, n):
    """Geoms (type, size, rgba, instrument name) of all instruments and their world pose per frame (x y z, qx qy qz qw)."""
    names = [str(s) for s in ins['names']]
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
    owner = []
    for g in gids:
        bn = m.body(m.geom_bodyid[g]).name
        owner.append(next((nm for nm in sorted(names, key=len, reverse=True) if bn.startswith(nm)), None))
    geoms = [dict(type=mujoco.mjtGeom(m.geom_type[g]).name.replace('mjGEOM_', '').lower(), size=r(m.geom_size[g]),
                  rgba=r(m.geom_rgba[g], 3), owner=o) for g, o in zip(gids, owner)]
    return geoms, r(rows, 5)


def filled_texture(V, ref):
    """Frame `ref` with the pixels outside the scope image replaced by the nearest tissue colour (not black)."""
    val = V.valid if V.valid.ndim == 2 else V.valid[ref]
    bad = cv2.dilate((~val).astype(np.uint8), np.ones((9, 9), np.uint8))
    _, lab = cv2.distanceTransformWithLabels(bad, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(bad == 0)
    lut = np.zeros((lab.max() + 1, 2), int)
    lut[lab[ys, xs]] = np.c_[ys, xs]
    out = cv2.blur(V.frames[ref][lut[lab][..., 0], lut[lab][..., 1]], (9, 9))
    out[bad == 0] = V.frames[ref][bad == 0]
    return out


def main(argv):
    from scipy.spatial import ConvexHull
    clip = argv[0]
    kv = dict(a.split('=', 1) for a in argv[1:] if '=' in a)
    out = SITE / clip
    out.mkdir(parents=True, exist_ok=True)
    V = views2.load(clip)
    n, H, W = V.n, V.H, V.W
    sam = list(V.names)                                              # every SAM 3 object of the clip
    # ---- fitted structures (organ agent): the primary organ and the rest
    sel = json.loads((D.OUT / clip / 'organs' / 'selection.json').read_text())
    spec_fit = [o['name'].replace(' ', '_') for o in sel['fit'] if not o.get('generic')]
    fitted = {}
    for od in sorted((D.OUT / clip / 'organs').glob('*/')):
        ver = kv.get(f'organ_{od.name}') or latest(clip, 'organs', od.name)
        if not ver:
            continue
        z = np.load(od / ver / 'model.npz')
        q = json.loads((od / ver / 'quality.json').read_text()) if (od / ver / 'quality.json').exists() else {}
        fitted[od.name] = dict(z=z, ver=ver, mask=q.get('mask', od.name.replace('_', ' ')), q=q)
    prim = next((nm for nm in spec_fit if nm in fitted), sorted(fitted)[0])
    P = fitted[prim]
    z = P['z']
    F = z['faces']
    surf = np.unique(F)
    remap = -np.ones(len(z['rest_verts']), int)
    remap[surf] = np.arange(len(surf))
    ref = int(z['start_frame']) if 'start_frame' in z.files else 0
    qq, _ = V.project(z['verts4d'][ref].astype(float), ref)
    uv_org = np.c_[np.clip(qq[surf, 0] / W, 0, 1), 1 - np.clip(qq[surf, 1] / H, 0, 1)]
    iio.imwrite(out / 'tex_tissue.jpg', filled_texture(V, ref), quality=86)
    # ---- background surface (labelled regions) and bodies
    bver = kv.get('background') or latest(clip, 'background')
    zb = np.load(D.OUT / clip / 'background' / bver / 'model.npz')
    tex = cv2.cvtColor(cv2.imread(str(D.OUT / clip / 'background' / bver / 'texture.png')), cv2.COLOR_BGR2RGB)
    iio.imwrite(out / 'tex_back.jpg', tex, quality=86)
    lab_names = [str(x) for x in zb['label_names']] if 'label_names' in zb.files else []
    lab = zb['label'].astype(int) if 'label' in zb.files else np.zeros(len(zb['rest_verts']), int)
    body_names = [str(s) for s in zb['body_names']] if 'body_names' in zb.files else []
    # ---- instruments
    iver = kv.get('instruments') or latest(clip, 'instruments')
    ins = np.load(D.OUT / clip / 'instruments' / iver / 'model.npz')
    ins_names = [str(s) for s in ins['names']]
    geoms, gtraj = instrument_geoms(ins, n)
    qi = json.loads((D.OUT / clip / 'instruments' / iver / 'quality.json').read_text()).get('instruments', {})
    qb = json.loads((D.OUT / clip / 'background' / bver / 'quality.json').read_text()).get('bodies', {})
    iou_of = lambda f: (f['q'].get('bands', {}).get('4d', {}).get('all', {}) or {}).get('iou')
    # ---- the object list: every SAM 3 object and the model that reconstructs it
    objects, pal, steel = [], iter(PALETTE * 3), iter(STEEL * 3)
    by_mask = {f['mask']: nm for nm, f in fitted.items()}
    for nm in sam:
        o = dict(name=nm, label=NAMES_ZH.get(nm, nm))
        if nm in ins_names:
            t = INS.tool_from_npz(ins, nm)
            o.update(kind='instrument', color=rgb(next(steel)), iou=(qi.get(nm, {}).get('iou_vis') or qi.get(nm, {}).get('iou') or {}).get('median'),
                     how=f"刚体模型（{TYPE_ZH.get(t.get('type'), t.get('type'))}，杆径 {2000 * t['radius']:.0f} mm），绕固定穿刺点逐帧拟合位姿")
        elif nm in by_mask:
            f = fitted[by_mask[nm]]
            vol = f['q'].get('volume', {}).get('rest_ml')
            o['iou'] = iou_of(f)
            if by_mask[nm] == prim:
                o.update(kind='organ', color=rgb(PRIMARY), how=f"四面体体模型，从椭球出发，逐帧形变（{len(f['z']['tets'])} 个四面体" + (f"，{vol:.1f} ml）" if vol else '）'))
            else:
                o.update(kind='part', color=rgb(next(pal)), how='通用体模型（管状或膜状结构的近似），逐帧形变' + (f"，{vol:.1f} ml" if vol else ''))
        elif nm in body_names:
            i = body_names.index(nm)
            o.update(kind='body', color=rgb(next(pal)), iou=((qb.get(nm, {}).get('iou') or {}).get('all') or {}).get('median'), how=f"封闭凸体（{len(np.unique(zb[f'body{i}_piece']))} 块），静态，仿真里可碰撞")
        elif nm in lab_names and (lab == lab_names.index(nm)).any():
            o.update(kind='surface', color=rgb(next(pal)), how=f"背景表面上带标签的区域（{int((lab == lab_names.index(nm)).sum())} 个顶点），静态，仿真里可碰撞")
        else:
            o.update(kind='none', color=[0.35, 0.35, 0.35], how='还没有三维模型')
        if o.get('iou') is not None:
            o['iou'] = round(float(o['iou']), 2)
            o['how'] += f"；和掩码的重合 IoU {o['iou']:.2f}"
        objects.append(o)
    oid = {o['name']: i for i, o in enumerate(objects)}
    for g in geoms:
        g['object'] = oid.get(g.pop('owner'), -1)
    # ---- backdrop faces grouped by object
    Fb = zb['faces'].astype(int)
    surf_obj = {l: oid[nm] for l, nm in enumerate(lab_names) if nm in oid and objects[oid[nm]]['kind'] == 'surface'}
    fobj = np.array([surf_obj.get(int(l), -1) for l in lab[Fb[:, 0]]])
    order = np.argsort(fobj, kind='stable')
    Fb, fobj = Fb[order], fobj[order]
    groups, st = [], 0
    for v in np.unique(fobj):
        c = int((fobj == v).sum())
        groups.append(dict(start=st * 3, count=c * 3, object=int(v)))
        st += c
    uvb = zb['uv'].astype(float)
    backdrop = dict(pos=r(zb['rest_verts']), uv=r(np.c_[uvb[:, 0], uvb[:, 1]]), index=Fb.ravel().tolist(), groups=groups)
    # ---- ribbons: other fitted structures (dynamic) and bodies (static), flat colours from the video in an atlas
    ribbons, dyn, colors = [], [], []
    for nm, f in fitted.items():
        if nm == prim:
            continue
        zz = f['z']
        Ff = zz['faces']
        sv = np.unique(Ff)
        rm = -np.ones(len(zz['rest_verts']), int)
        rm[sv] = np.arange(len(sv))
        k0 = int(zz['start_frame']) if 'start_frame' in zz.files else 0
        m = V.mask(f['mask'])[k0] if f['mask'] in sam else None
        col = V.frames[k0][m].mean(0) if m is not None and m.any() else np.array([170, 90, 110])
        ribbons.append(dict(n=len(sv), index=rm[Ff].ravel().tolist(), color=len(colors), object=oid.get(f['mask'], -1), dynamic=True))
        dyn.append(zz['verts4d'].astype(float)[:, sv])
        colors.append(col)
    for i, nm in enumerate(body_names):
        Xb, pc = zb[f'body{i}_verts'].astype(float), zb[f'body{i}_piece']
        col = zb[f'body{i}_rgb'].reshape(-1, 3).mean(0) if f'body{i}_rgb' in zb.files else np.array([170, 80, 80])
        for p in np.unique(pc):
            Pp = Xb[pc == p]
            if len(Pp) > 400:                       # a light hull: the piece's own hull vertices, thinned
                hv = np.unique(ConvexHull(Pp).simplices)
                Pp = Pp[hv[np.linspace(0, len(hv) - 1, min(len(hv), 90)).astype(int)]]
            h = ConvexHull(Pp)
            vi = np.unique(h.simplices)
            rm = -np.ones(len(Pp), int)
            rm[vi] = np.arange(len(vi))
            tri = h.simplices.copy()                 # outward winding (for shading)
            nrm = np.cross(Pp[tri[:, 1]] - Pp[tri[:, 0]], Pp[tri[:, 2]] - Pp[tri[:, 0]])
            flip = (nrm * h.equations[:, :3]).sum(1) < 0
            tri[flip] = tri[flip][:, ::-1]
            ribbons.append(dict(n=len(vi), index=rm[tri].ravel().tolist(), color=len(colors), object=oid.get(nm, -1), dynamic=False))
            dyn.append(np.repeat(Pp[vi][None], n, 0))
        colors.append(col)
    atlas = np.zeros((16, 16 * max(1, len(colors)), 3), np.uint8)
    for j, c in enumerate(colors):
        atlas[:, 16 * j:16 * (j + 1)] = c
    iio.imwrite(out / 'tex_ribbon.jpg', atlas, quality=92)
    for rb in ribbons:
        u = (16 * rb.pop('color') + 8) / atlas.shape[1]
        rb['uv'] = r(np.tile([u, 0.5], (rb['n'], 1)))
    rest = np.concatenate(dyn, 1) if dyn else np.zeros((n, 0, 3))
    # ---- conditions
    conditions = {}

    def add(key, X4, metrics):
        pos = np.concatenate([X4[:, surf], rest], 1)
        (out / f'{key}.txt').write_text(quant(pos))
        conditions[key] = dict(file=f'{key}.txt', steps=n, n_surface=len(surf), uv=r(uv_org), index=remap[F].ravel().tolist(),
                               object=oid.get(P['mask'], -1), geoms=geoms, geom_traj=gtraj, metrics=metrics,
                               max_disp_mm=round(float(np.linalg.norm(X4 - X4[:1], axis=2).max()) * 1000, 1))
    add('recon4d', z['verts4d'].astype(float), dict(note='4D reconstruction (organ agent %s)' % P['ver']))
    if 'sim' in kv:
        rd = D.OUT / clip / 'rounds' / kv['sim']
        tr = np.load(rd / 'traj.npz')
        f0, f1 = [int(v) for v in tr['frames']]
        Xs = tr[f'organ_{prim}'].astype(float)
        X4 = np.concatenate([np.repeat(Xs[:1], f0, 0), Xs, np.repeat(Xs[-1:], n - 1 - f1, 0)])
        ev = json.loads((rd / 'metrics.json').read_text())['eval'][prim]
        add('sim4d', X4, {k: ev.get(k) for k in ('sim_iou', 'sim_iou_2d', 'recon_iou', 'motion_explained', 'inverted_tets_max', 'err_vs_recon_mm')})
    # ---- scope camera per frame, free view
    eq, ef, ep = [], [], []
    for k in range(n):
        qm = V.cam.mj_quat(V.R[k])
        eq += [qm[1], qm[2], qm[3], qm[0]]
        ef.append(float(np.degrees(2 * np.arctan(H / 2 / V.f[k]))))
        ep += list(V.pos[k])
    pivot = z['verts4d'][n // 2].astype(float).mean(0)
    right, down, fwd = V.R[n // 2]                 # the free view starts beside the scope, with the image's up as up
    look = np.cos(np.radians(30)) * fwd + np.sin(np.radians(30)) * right + 0.15 * down
    look /= np.linalg.norm(look)
    dist = 1.5 * float(np.linalg.norm(pivot - V.pos[n // 2]))
    scene = dict(name=clip, title=kv.get('title') or TITLES.get(clip, clip), dt=1 / V.fps, fps=V.fps, steps=n, size=[W, H], quant=[LO, HI],
                 objects=objects, backdrop=backdrop, ribbons=ribbons, conditions=conditions,
                 endo=dict(pos=r(ep), quat=r(eq, 6), fov=r(ef, 3)),
                 ports=[r(ins[f'{nm}__port']) for nm in ins_names],
                 view=dict(target=r(pivot), position=r(pivot - dist * look), up=r(-down, 4)),
                 versions=dict(geometry=V.geometry, organs={nm: f['ver'] for nm, f in fitted.items()}, instruments=iver, background=bver))
    (out / 'scene.json').write_text(json.dumps(scene, separators=(',', ':'), ensure_ascii=False))
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
    print(f"[export] {clip}: " + '; '.join(f"{o['name']} -> {o['kind']}" for o in objects))
    missing = [o['name'] for o in objects if o['kind'] == 'none']
    print(f"[export] {clip}: {len(objects) - len(missing)} / {len(objects)} SAM 3 objects have a 3D model" + (f"; MISSING: {missing}" if missing else ''))
    for p in sorted(out.iterdir()):
        print(f'{p.stat().st_size / 1e6:6.2f} MB  {p.name}')


if __name__ == '__main__':
    main(sys.argv[1:])
