"""Export clips for the interactive web viewer (site/viewer, three.js).

  python -m r2s.export <out_dir> <clip>[:cond,cond...] ...
per clip: <out>/<clip>/scene.json (backdrop, strands, organ surfaces per condition, instruments per condition,
          endoscope pose per step, ports, start view), <cond>.txt (base64 int16 surface + strand positions per step),
          tex_*.jpg, video.mp4;  <out>/index.json lists the clips.
World coordinates are MuJoCo's (metres, z up); uv follow three.js (v from the bottom).
"""
import base64
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from .config import Clip, load_json
from . import sim as SIM, source, evaluate as EV

LO, HI = -0.4, 0.4


def r(a, n=5):
    return np.round(np.asarray(a, float), n).ravel().tolist()


def quant(x):
    q = np.clip(np.round((np.asarray(x) - LO) / (HI - LO) * 65535 - 32768), -32768, 32767).astype('<i2')
    return base64.b64encode(q.tobytes()).decode()


def obj_mesh(path):
    V, VT, F = [], [], []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            V.append([float(x) for x in p[1:4]])
        elif p[0] == 'vt':
            VT.append([float(x) for x in p[1:3]])
        elif p[0] == 'f':
            F.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
    return np.array(V), np.array(VT), np.array(F)


def export_clip(clip, conds, out):
    out.mkdir(parents=True, exist_ok=True)
    S = SIM.load_scene(clip)
    cv = SIM.canvas_of(clip, S)
    cam = cv.cam
    V, VT, F = obj_mesh(clip.scene / 'backdrop.obj')
    ribbons = []
    for R in S['ribbons']:
        u, v = cv.uv(R['px'][:, 0], R['px'][:, 1])
        ribbons.append(dict(n=len(R['X']), uv=r(np.stack([u, 1 - v], 1)), index=R['tri'].astype(int).ravel().tolist()))
    n_rib = sum(len(R['X']) for R in S['ribbons'])
    conditions = {}
    pivot = None
    for c in conds:
        d = clip.cond_dir(c)
        o = SIM.load_organ(d / 'organ.npz')
        z = np.load(d / 'traj.npz')
        flex, acts, n_org = z['flex'], z['actions'], int(z['n_organ'])
        surf = np.unique(o['faces'])
        remap = -np.ones(n_org, int)
        remap[surf] = np.arange(len(surf))
        u, v = cv.uv(o['px'][surf, 0], o['px'][surf, 1])
        sel = np.r_[surf, n_org + np.arange(n_rib)]
        (out / f'{c}.txt').write_text(quant(flex[:, sel]))
        # instruments: every geom's pose per step
        m = mujoco.MjModel.from_xml_path(str(d / 'scene.xml'))
        dd = mujoco.MjData(m)
        gids = [i for i in range(m.ngeom) if any(m.body(m.geom_bodyid[i]).name.startswith(ins['name']) for ins in clip.instruments)]
        tr = []
        for k in range(len(flex)):
            SIM.set_state(m, clip, dd, flex[k], acts[k])
            row = []
            for i in gids:
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, dd.geom_xmat[i])
                row += list(dd.geom_xpos[i]) + [q[1], q[2], q[3], q[0]]
            tr.append(row)
        geoms = [dict(type=mujoco.mjtGeom(m.geom_type[i]).name.replace('mjGEOM_', '').lower(), size=r(m.geom_size[i]),
                      rgba=r(m.geom_rgba[i], 3)) for i in gids]
        ev = load_json(d / 'eval.json')['sim'] if (d / 'eval.json').exists() else {}
        ref = clip.out / 'measured' / 'eval.json'
        tm = EV.track_motion(ev, EV.motion_seeds(load_json(ref)['sim'])) if ref.exists() and ev else None
        conditions[c] = dict(file=f'{c}.txt', steps=len(flex), n_surface=len(surf), uv=r(np.stack([u, 1 - v], 1)),
                             index=remap[o['faces']].ravel().tolist(), geoms=geoms, geom_traj=r(tr, 5),
                             metrics=dict({k: ev.get(k) for k in ('outline_iou', 'depth_err_mm_monocular', 'depth_err_mm_stereo', 'track_err_px')},
                                          track_motion_pct=tm['change_pct'] if tm else None),
                             max_disp_mm=round(float(np.linalg.norm(flex[:, :n_org] - flex[:1, :n_org], axis=2).max()) * 1000, 1))
        if pivot is None:
            org0 = flex[0, :n_org]
            q, z0 = cam.project(org0, S['cam_R'][0], S['cam_f'][0])
            inside = (q[:, 0] >= 0) & (q[:, 0] < cam.W) & (q[:, 1] >= 0) & (q[:, 1] < cam.H)
            pivot = org0[inside].mean(0) if inside.any() else org0.mean(0)
            dist = 1.5 * float(np.median(z0[inside] if inside.any() else z0))
    n = max(v['steps'] for v in conditions.values())
    dt = SIM.PARAMS['dt']
    endo_q, endo_fov = [], []
    for k in range(n):
        kv = min(int(round(k * dt * clip['fps'])), len(S['cam_R']) - 1)
        q = cam.mj_quat(S['cam_R'][kv])
        endo_q += [q[1], q[2], q[3], q[0]]
        endo_fov.append(float(np.degrees(2 * np.arctan(cam.H / 2 / S['cam_f'][kv]))))
    fwd = S['cam_R'][0][2]
    az, el = np.arctan2(fwd[1], fwd[0]) + np.radians(35), -np.arcsin(-fwd[2]) + np.radians(8)
    look = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    meta = load_json(clip.scene / 'scene.json')
    scene = dict(name=out.name, title=clip['title'], dt=dt, fps=clip['fps'], steps=n, size=[cam.W, cam.H], quant=[LO, HI],
                 backdrop=dict(pos=r(V), uv=r(VT), index=F.ravel().tolist()), ribbons=ribbons, conditions=conditions,
                 endo=dict(pos=r(cam.pos), quat=r(endo_q, 6), fov=r(endo_fov, 3)),
                 ports=[r(v['rcm']) for v in meta['instruments'].values()],
                 view=dict(target=r(pivot), position=r(pivot - dist * look)))
    (out / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')))
    for t in ('tex_back', 'tex_tissue', 'tex_ribbon'):
        imageio.imwrite(out / f'{t}.jpg', imageio.imread(clip.scene / f'{t}.png')[..., :3], quality=86)
    frames = source.frames(clip)
    imageio.mimsave(out / 'video.mp4', frames, fps=clip['fps'], macro_block_size=1, quality=7,
                    output_params=['-g', '5', '-movflags', '+faststart'])
    return dict(name=out.name, title=clip['title'], conditions=list(conditions), steps=n, size=[cam.W, cam.H])


def main(out, specs):
    out = Path(out)
    index = []
    for spec in specs:
        name, _, cs = spec.partition(':')
        clip = Clip(name)
        conds = cs.split(',') if cs else [c for c in ('measured', 'primitive', 'template', 'template_fit', 'template_mv') if (clip.out / c / 'traj.npz').exists()]
        folder = name.replace('+', '_')                    # variants: chole_a+sift -> chole_a_sift
        index.append(export_clip(clip, conds, out / folder))
        print('exported', name, conds)
    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
