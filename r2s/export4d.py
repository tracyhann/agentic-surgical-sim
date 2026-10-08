"""Export the per-tissue 4D study (outputs/iter/rounds/) for the 3D viewer (site/viewer), in the viewer's scene format.

    python -m r2s.export4d site/viewer/data sim4d=r27/render=1_mode=sim recon4d=r28/render=0_mode=recon ...

One scene folder <out>/chole_a_4d/ with scene.json (backdrop, ribbons, conditions, endoscope per step, ports, start
view) and one <condition>.txt per run (base64 int16 positions per control step: gallbladder surface, then the
ribbons = sheet + duct tubes). Mapping onto the viewer's layers: 'organ' = gallbladder (tex_tissue.jpg), 'ribbons' =
the peritoneal sheet and the four duct tubes (tex_ribbon.jpg: the sheet's video frame with a strip of the ducts'
video colours below it), 'backdrop' = the fused background (tex_back.jpg). Instruments: the run's actions replayed
through its scene.xml. World coordinates are MuJoCo's; uv follow three.js (v from the bottom). index.json gets the
scene appended (replacing an earlier entry of the same name).
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from . import views, instruments as INS
from . import scene4d as S
from .export import quant, r, LO, HI

NAME = 'chole_a_4d'
TITLE = '胆囊 · 镜头 A · 按组织建模的 4D 仿真'
STRIP = 40                         # px of duct colour under the sheet's frame in tex_ribbon.jpg


def run_inputs(run):
    m = json.loads((run / 'metrics.json').read_text())
    P = dict(S.PARAMS, **m['params'])
    tissues = {t: S.load_tissue(t, m['tissues'].get(t) if t in m['tissues'] else None) for t in S.TISSUES}
    tr = np.load(run / 'traj.npz')
    return m, P, tissues, {k: tr[k] for k in tr.files}


def instrument_traj(V, run, P, n_steps):
    """Geoms of the instruments (type, size, rgba) and their world pose per control step from the run's actions."""
    m = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    d = mujoco.MjData(m)
    tools, _ = S.tool_tracks(V)
    S.counterfactual(V, tools, P)
    acts, _ = S.control_targets(V, tools, P)
    names = [ins['name'] for ins in V.clip.instruments]
    gids = [g for g in range(m.ngeom) if any(m.body(m.geom_bodyid[g]).name.startswith(n) for n in names)
            and m.geom_rgba[g][3] > 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    rows = []
    for k in range(n_steps):
        a = acts[min(k, len(acts) - 1)]
        for j, ins in enumerate(V.clip.instruments):
            for i, jn in enumerate(INS.ARM):
                d.qpos[m.joint(f"{ins['name']}_{jn}").qposadr[0]] = a[5 * j + i]
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


def tissue_uv(V, T):
    """Texture coordinates of a tissue on the video frame closest to its rest shape (as scene4d's textures)."""
    rec = np.asarray(T['verts4d'])
    ref = int(np.argmin(np.linalg.norm(rec - T['rest_verts'][None], axis=2).mean(1)))
    q, _ = V.project(rec[ref], ref)
    return np.c_[np.clip(q[:, 0] / V.W, 0, 1), np.clip(q[:, 1] / V.H, 0, 1)], ref


def main(out, specs):
    out = Path(out)
    dst = out / NAME
    dst.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in specs:
        key, _, path = spec.partition('=')
        runs[key] = S.ITER / 'rounds' / path
    first = next(iter(runs.values()))
    m0, P0, tissues, _ = run_inputs(first)
    V = views.load('chole_a', 'sift', S.CAMS.get(P0['cams'], P0['cams']))
    G, M, B, Dt = tissues['gallbladder'], tissues['membrane'], tissues['backdrop'], tissues['ducts']
    # textures
    uv_g, ref_g = tissue_uv(V, G)
    uv_m, ref_m = tissue_uv(V, M)
    imageio.imwrite(dst / 'tex_tissue.jpg', V.frames[ref_g], quality=86)
    duct_rgb, strand_rgb = S.tube_colors(V, Dt)
    atlas = np.zeros((V.H + STRIP, V.W, 3), np.uint8)
    atlas[:V.H] = V.frames[ref_m]
    atlas[V.H:, :V.W // 2] = duct_rgb
    atlas[V.H:, V.W // 2:] = strand_rgb
    imageio.imwrite(dst / 'tex_ribbon.jpg', atlas, quality=86)
    tex_b = cv2.cvtColor(cv2.imread(str(B['dir'] / 'texture.png')), cv2.COLOR_BGR2RGB)
    imageio.imwrite(dst / 'tex_back.jpg', tex_b, quality=86)
    Xb = S.backdrop_behind(B, G)
    backdrop = dict(pos=r(Xb), uv=r(np.asarray(B['uv'], float)), index=np.asarray(B['faces']).astype(int).ravel().tolist())
    # gallbladder surface (organ layer)
    surf = np.unique(G['faces'])
    remap = -np.ones(len(G['rest_verts']), int)
    remap[surf] = np.arange(len(surf))
    uv_org = np.c_[uv_g[surf, 0], 1 - uv_g[surf, 1]]
    # ribbons: sheet, then one tube per cable (tube topology from scene4d.tube_mesh)
    ribbons = [dict(n=len(M['rest_verts']), uv=r(np.c_[uv_m[:, 0], 1 - uv_m[:, 1] * V.H / (V.H + STRIP)]),
                    index=np.asarray(M['faces']).astype(int).ravel().tolist())]
    parts = {t: S.sim_bodies(tissues[t], t, P0, G) for t in S.SIMULATED}
    tube_faces = []
    for p in parts['ducts']:
        vv, ff = S.tube_mesh(p['X'][None], p['spec']['radius'])
        u = 0.25 if p['name'].endswith('_duct') else 0.75
        ribbons.append(dict(n=vv.shape[1], uv=r(np.tile([u, STRIP / 2 / (V.H + STRIP)], (vv.shape[1], 1))),
                            index=np.asarray(ff).astype(int).ravel().tolist()))
        tube_faces.append(p)
    conditions = {}
    pivot = dist = None
    for key, run in runs.items():
        m, P, tis, traj = run_inputs(run)
        t_ctl = traj['t']
        n = len(t_ctl)
        gb = traj['gallbladder']                                   # (steps, N, 3) at control steps
        mem = traj['membrane']
        tubes = [S.tube_mesh(traj[f'part_{p["name"]}'], p['spec']['radius'])[0] for p in tube_faces]
        pos = np.concatenate([gb[:, surf], mem] + tubes, 1)
        (dst / f'{key}.txt').write_text(quant(pos))
        geoms, gtraj = instrument_traj(V, run, P, n)
        e = m['eval']
        conditions[key] = dict(file=f'{key}.txt', steps=n, n_surface=len(surf), uv=r(uv_org),
                               index=remap[np.asarray(G['faces'])].ravel().tolist(), geoms=geoms, geom_traj=gtraj,
                               metrics=dict(outline_iou=(e['gallbladder'].get('sim_iou') or {}).get('mean'),
                                            depth_err_mm_monocular=(e['gallbladder'].get('sim_depth_mm') or {}).get('mean'),
                                            depth_err_mm_stereo=None, track_err_px=None,
                                            track_motion_pct=(e['gallbladder'].get('track_motion') or {}).get('change_pct')),
                               max_disp_mm=round(float(np.linalg.norm(gb - gb[:1], axis=2).max()) * 1000, 1))
        if pivot is None:
            q, z0 = V.project(gb[0], 0)
            pivot = gb[0].mean(0)
            dist = 1.6 * float(np.median(z0))
    dt = P0['dt']
    n = max(c['steps'] for c in conditions.values())
    kv = [min(int(round(k * dt * V.fps)), V.n - 1) for k in range(n)]
    endo_q, endo_fov, endo_pos = [], [], []
    for k in kv:
        q = V.cam.mj_quat(V.R[k])
        endo_q += [q[1], q[2], q[3], q[0]]
        endo_fov.append(float(np.degrees(2 * np.arctan(V.H / 2 / V.f[k]))))
        endo_pos += list(V.pos[k])
    fwd = V.R[0][2]
    az, el = np.arctan2(fwd[1], fwd[0]) + np.radians(35), -np.arcsin(-fwd[2]) + np.radians(8)
    look = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    tools, _ = S.tool_tracks(V)
    scene = dict(name=NAME, title=TITLE, dt=dt, fps=V.fps, steps=n, size=[V.W, V.H], quant=[LO, HI],
                 backdrop=backdrop, ribbons=ribbons, conditions=conditions,
                 endo=dict(pos=r(endo_pos), quat=r(endo_q, 6), fov=r(endo_fov, 3)),
                 ports=[r(t['rcm']) for t in tools.values()],
                 view=dict(target=r(pivot), position=r(pivot - dist * look)))
    (dst / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')))
    vid = out / 'chole_a' / 'video.mp4'
    if vid.exists():
        shutil.copy(vid, dst / 'video.mp4')
    else:
        imageio.mimsave(dst / 'video.mp4', list(V.frames), fps=V.fps, macro_block_size=1, quality=7,
                        output_params=['-g', '5', '-movflags', '+faststart'])
    idx_p = out / 'index.json'
    idx = [e for e in (json.loads(idx_p.read_text()) if idx_p.exists() else []) if e['name'] != NAME]
    idx.insert(0, dict(name=NAME, title=TITLE, conditions=list(conditions), steps=n, size=[V.W, V.H]))
    idx_p.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    for p in sorted(dst.iterdir()):
        print(f'{p.stat().st_size / 1e6:6.2f} MB  {p.name}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
