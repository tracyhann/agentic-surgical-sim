"""Export a finished 3D run for the interactive web viewer (three.js).

  python export_viewer.py <out_dir>
    scene.json        static meshes (backdrop, tissue surface topology, strand ribbon), uvs, instrument geoms,
                      endoscope pose per step, ports, initial orbit camera, quantisation of the trajectories
    flex_<tag>.txt    base64 of int16 vertex positions, steps x vertices x 3 (tissue front, tissue back, strand)
    tex_*.jpg         the projected textures;  video.mp4  the source clip, keyframe every 5 frames (for scrubbing)
Coordinates are MuJoCo world (metres, z up); uv are three.js convention (v from the image bottom).
"""
import base64, json, sys
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'realistic'))
sys.path.insert(0, str(ROOT / 'real3d'))
import chole_recon as C  # noqa: E402
import chole_sim as S  # noqa: E402
import views as V  # noqa: E402

OUT = ROOT / 'real3d' / 'chole3d'
VARIANTS = (('main', ''), ('retract2x', '_retract2x'), ('press', '_press'))


def r(a, n=5):
    return np.round(np.asarray(a, float), n).ravel().tolist()


def uv3(px):
    u, v = C.uv(px[:, 0], px[:, 1])
    return np.stack([u, 1 - v], 1)


def main(dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    C.OUT = S.OUT = V.OUT = OUT
    g = np.load(OUT / 'geometry.npz')
    meta = json.loads((OUT / 'build_meta.json').read_text())
    m = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
    d = mujoco.MjData(m)
    N, NR = len(g['sheet_X']), len(g['ribbon_X'])
    # tissue: closed surface over 2N vertices (front, back, side walls along the open boundary)
    tri = g['sheet_tri'].astype(int)
    edges = {}
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    walls = []
    for e, occ in edges.items():
        if len(occ) == 1:
            a, b = occ[0]
            walls += [(a, b, b + N), (a, b + N, a + N)]
    tissue_idx = np.r_[tri, tri[:, ::-1] + N, np.array(walls, int)]
    tissue_uv = np.r_[uv3(g['sheet_px']), uv3(g['sheet_px'])]
    # backdrop OBJ (world coordinates, OBJ uv already v-from-bottom)
    V_, VT, F = [], [], []
    for line in (OUT / 'backdrop.obj').read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'v':
            V_.append([float(x) for x in p[1:4]])
        elif p[0] == 'vt':
            VT.append([float(x) for x in p[1:3]])
        elif p[0] == 'f':
            F.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
    # instruments: geoms of the two instruments
    geoms = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(('grasper_left', 'probe_right'))]
    ginfo = [dict(name=m.geom(i).name, type=mujoco.mjtGeom(m.geom_type[i]).name.replace('mjGEOM_', '').lower(),
                  size=r(m.geom_size[i]), rgba=r(m.geom_rgba[i], 3)) for i in geoms]
    # trajectories
    lo, hi = -0.35, 0.35
    q = lambda x: np.clip(np.round((x - lo) / (hi - lo) * 65535 - 32768), -32768, 32767).astype(np.int16)
    variants = {}
    for name, tag in VARIANTS:
        flex = np.load(OUT / f'flex_traj{tag}.npy')
        acts = np.load(OUT / f'actions_executed{tag}.npy')
        (dst / f'flex_{name}.txt').write_text(base64.b64encode(q(flex).astype('<i2').tobytes()).decode())
        tr = []
        for k in range(len(flex)):
            V.set_state(m, d, flex[k], acts[k])
            row = []
            for i in geoms:
                qq = np.zeros(4)
                mujoco.mju_mat2Quat(qq, d.geom_xmat[i])
                row += list(d.geom_xpos[i]) + [qq[1], qq[2], qq[3], qq[0]]        # three.js quaternion order x y z w
            tr.append(row)
        variants[name] = dict(file=f'flex_{name}.txt', steps=len(flex), geoms=r(tr, 5),
                              max_disp_mm=round(float(np.linalg.norm(flex - flex[0], axis=2).max()) * 1000, 1))
    # endoscope: fixed tip, rotation and zoom per step (from the video frame at that time)
    n = len(np.load(OUT / 'flex_traj.npy'))
    endo_q, endo_fov = [], []
    for k in range(n):
        kv = min(int(round(k * S.DT * C.FPS)), len(g['cam_R']) - 1)
        R = g['cam_R'][kv]
        M = np.stack([R[0], -R[1], -R[2]], axis=1)
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, M.flatten())
        endo_q += [qq[1], qq[2], qq[3], qq[0]]
        endo_fov.append(float(np.degrees(2 * np.arctan(C.H / 2 / g['cam_f'][kv]))))
    # initial orbit view = the report's 35 degree view
    cam = V.orbit_cam(g, 35)
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fw = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    target = np.array(cam.lookat)
    scene = dict(
        dt=S.DT, fps=C.FPS, n_steps=n, quant=dict(lo=lo, hi=hi),
        backdrop=dict(pos=r(V_), uv=r(VT), index=np.array(F).ravel().tolist()),
        tissue=dict(n=N, uv=r(tissue_uv), index=tissue_idx.ravel().tolist()),
        ribbon=dict(n=NR, uv=r(uv3(g['ribbon_px'])), index=g['ribbon_tri'].astype(int).ravel().tolist()),
        geoms=ginfo, variants=variants,
        endo=dict(pos=r(C.CAM_POS), quat=r(endo_q, 6), fov=r(endo_fov, 3)),
        ports={k: r(v['rcm']) for k, v in meta['instruments'].items()},
        view=dict(target=r(target), position=r(target - cam.distance * fw)))
    (dst / 'scene.json').write_text(json.dumps(scene, separators=(',', ':')))
    for t in ('tex_back', 'tex_tissue', 'tex_ribbon'):
        imageio.imwrite(dst / f'{t}.jpg', imageio.imread(OUT / f'{t}.png')[..., :3], quality=88)
    src = [f for f in imageio.get_reader(ROOT / 'runs_real/chole_sweep/task/video.mp4')]
    imageio.mimsave(dst / 'video.mp4', src, fps=C.FPS, macro_block_size=1, quality=7,
                    output_params=['-g', '5', '-movflags', '+faststart'])
    return {p.name: p.stat().st_size for p in sorted(dst.iterdir())}


if __name__ == '__main__':
    print(main(sys.argv[1]))
