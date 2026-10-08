"""Re-render saved simulation states from new viewpoints (no re-simulation): tissue / strand vertices from
flex_traj.npy, instruments from actions_executed.npy.

  python views.py          -> real3d/chole3d/views.png   rows: t = 0, 3, 6 s
                              columns: video | simulation through the scope | orbit 35 deg | orbit 70 deg
  python views.py video    -> real3d/chole3d/views.mp4   the same four panels as a 2x2 video, every control step
"""
import sys
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'realistic'))
import chole_recon as C  # noqa: E402
import chole_sim as S  # noqa: E402

OUT = ROOT / 'real3d' / 'chole3d'


def set_state(m, d, flex_k, act):
    S.set_qpos(m, d, act)
    for v, b in enumerate(m.flex_vertbodyid):
        if m.body_jntnum[b]:                                  # three slide joints (x, y, z) from the rest position
            a = m.jnt_qposadr[m.body_jntadr[b]]
            d.qpos[a:a + 3] = flex_k[v] - m.body_pos[b]
    mujoco.mj_forward(m, d)


def orbit_cam(g, turn):
    ys, xs = np.nonzero(g['gb'])
    c = C.unproject(xs.mean(), ys.mean(), float(np.median(g['z'][g['gb']])))
    fwd = C.R_CW[2]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = c
    cam.distance = 0.11
    cam.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0])) + turn
    cam.elevation = -np.degrees(np.arcsin(-fwd[2])) + 8
    return cam


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main(steps=(0, 60, 120), turns=(35, 70), out='views.png'):
    C.OUT = OUT
    S.OUT = OUT
    g = np.load(OUT / 'geometry.npz')
    m = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
    d = mujoco.MjData(m)
    flex = np.load(OUT / 'flex_traj.npy')
    acts = np.load(OUT / 'actions_executed.npy')
    sim = np.load(OUT / 'sim_frames.npy', mmap_mode='r')
    video = [f for f in imageio.get_reader(ROOT / 'runs_real/chole_sweep/task/video.mp4')]
    rend = mujoco.Renderer(m, C.H, C.W)
    rows = []
    for k in steps:
        kv = min(int(round(k * S.DT * C.FPS)), len(video) - 1)
        set_state(m, d, flex[k], acts[k])
        row = [label(video[kv], f'video  t={kv / C.FPS:.1f}s'), label(np.asarray(sim[k]), 'sim (scope)')]
        for t in turns:
            rend.update_scene(d, camera=orbit_cam(g, t))
            row.append(label(S.camera_look(rend.render()), f'sim orbit {t} deg'))
        rows.append(np.concatenate(row, 1))
    grid = np.concatenate(rows, 0)
    imageio.imwrite(OUT / out, grid[::2, ::2])
    return OUT / out


def video(turns=(35, 70), out='views.mp4'):
    C.OUT = OUT
    S.OUT = OUT
    g = np.load(OUT / 'geometry.npz')
    m = mujoco.MjModel.from_xml_path(str(OUT / 'scene.xml'))
    d = mujoco.MjData(m)
    flex = np.load(OUT / 'flex_traj.npy')
    acts = np.load(OUT / 'actions_executed.npy')
    sim = np.load(OUT / 'sim_frames.npy', mmap_mode='r')
    src = [f for f in imageio.get_reader(ROOT / 'runs_real/chole_sweep/task/video.mp4')]
    rend = mujoco.Renderer(m, C.H, C.W)
    cams = [orbit_cam(g, t) for t in turns]
    frames = []
    for k in range(len(flex)):
        set_state(m, d, flex[k], acts[k])
        orb = []
        for cam in cams:
            rend.update_scene(d, camera=cam)
            orb.append(S.camera_look(rend.render()))
        kv = min(int(round(k * S.DT * C.FPS)), len(src) - 1)
        top = np.concatenate([label(src[kv], 'video'), label(np.asarray(sim[k]), 'simulation (scope view)')], 1)
        bot = np.concatenate([label(orb[0], f'simulation, orbit {turns[0]} deg'), label(orb[1], f'simulation, orbit {turns[1]} deg')], 1)
        frames.append(np.concatenate([top, bot], 0))
    imageio.mimsave(OUT / out, frames, fps=20, macro_block_size=1, quality=8)
    return OUT / out


if __name__ == '__main__':
    print(video() if sys.argv[1:] == ['video'] else main())
