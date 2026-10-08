"""Renders of saved simulations (no re-simulation): through the scope, and from orbit cameras around the organ."""
import numpy as np
import cv2
import mujoco
import imageio.v2 as imageio
from . import sim as SIM

LABELS = {'video': 'video', 'measured': 'measured surface', 'primitive': 'primitive (ellipsoid)',
          'template': 'organ template', 'template_fit': 'template + fit', 'depth': 'measured depth',
          'template_mv': 'template, keyframes', 'template_mvd': 'template, keyframes + motion'}


def label(img, text, scale=0.7):
    img = img.copy()
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    return img


class Renderer:
    """Loads one condition's scene and trajectory; renders any step from the scope or an orbit camera."""

    def __init__(self, clip, cond, tag=''):
        self.clip, self.cond = clip, cond
        out = clip.cond_dir(cond)
        self.S = SIM.load_scene(clip)
        self.cam = clip.camera()
        # the scene is rebuilt from the organ and the current visual settings (same physics as simulated)
        organ = SIM.load_organ(out / 'organ.npz')
        self.m = mujoco.MjModel.from_xml_string(SIM.scene_xml(clip, self.S, organ, SIM.PARAMS, out))
        self.d = mujoco.MjData(self.m)
        z = np.load(out / f'traj{tag}.npz')
        self.flex, self.acts, self.n_org = z['flex'], z['actions'], int(z['n_organ'])
        self.r = mujoco.Renderer(self.m, self.cam.H, self.cam.W)
        self.dt = SIM.PARAMS['dt']
        # orbit pivot: the organ's visible centre at frame 0
        org0 = self.flex[0, :self.n_org]
        q, z0 = self.cam.project(org0, self.S['cam_R'][0], self.S['cam_f'][0])
        inside = (q[:, 0] >= 0) & (q[:, 0] < self.cam.W) & (q[:, 1] >= 0) & (q[:, 1] < self.cam.H)
        self.pivot = org0[inside].mean(0) if inside.any() else org0.mean(0)
        self.dist = 1.5 * float(np.median(z0[inside] if inside.any() else z0))

    @property
    def steps(self):
        return len(self.flex)

    def orbit(self, turn, lift=8):
        fwd = self.S['cam_R'][0][2]
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = self.pivot
        c.distance = self.dist
        c.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + turn
        c.elevation = -float(np.degrees(np.arcsin(-fwd[2]))) + lift
        return c

    def render(self, k, view='scope'):
        SIM.set_state(self.m, self.clip, self.d, self.flex[k], self.acts[k])
        if view == 'scope':
            SIM.set_camera(self.m, self.cam, self.S, k * self.dt, self.clip['fps'])
            self.r.update_scene(self.d, camera='endo')
        else:
            self.r.update_scene(self.d, camera=self.orbit(view))
        return SIM.camera_look(self.r.render())


def video_main(clip, cond, frames, out_path, turns=(35, 70)):
    """2x2: video | simulation through the scope / orbit views."""
    R = Renderer(clip, cond)
    out = []
    for k in range(R.steps):
        kv = min(int(round(k * R.dt * clip['fps'])), len(frames) - 1)
        top = np.concatenate([label(frames[kv], 'video'), label(R.render(k), 'simulation, scope view')], 1)
        bot = np.concatenate([label(R.render(k, t), f'simulation, orbit {t} deg') for t in turns], 1)
        out.append(np.concatenate([top, bot], 0))
    imageio.mimsave(out_path, out, fps=int(round(1 / R.dt)), macro_block_size=1, quality=7)
    return out_path


def video_conditions(clip, conds, frames, Zref, out_path, turn=60, scale=0.5):
    """Columns: video + one per condition. Rows: scope view, orbit view (the video column shows the measured depth)."""
    Rs = [Renderer(clip, c) for c in conds]
    n = min(R.steps for R in Rs)
    lo, hi = np.percentile(Zref[0], 3), np.percentile(Zref[0], 97)
    out = []
    for k in range(n):
        kv = min(int(round(k * Rs[0].dt * clip['fps'])), len(frames) - 1)
        dep = cv2.applyColorMap((np.clip((Zref[kv] - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        top = [label(frames[kv], LABELS['video'])] + [label(R.render(k), LABELS[c]) for R, c in zip(Rs, conds)]
        bot = [label(dep, LABELS['depth'])] + [label(R.render(k, turn), f'orbit {turn} deg') for R in Rs]
        img = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
        out.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    imageio.mimsave(out_path, out, fps=int(round(1 / Rs[0].dt)), macro_block_size=1, quality=7)
    return out_path


def video_variants(names, cond, frames, out_path, titles):
    """2x2 through the scope: the video and the same condition simulated in up to three variants of a clip
    (e.g. different camera geometries); names are clip names such as chole_a, chole_a+sift."""
    from .config import Clip
    Rs = [Renderer(Clip(n), cond) for n in names]
    n = min(R.steps for R in Rs)
    fps = Rs[0].clip['fps']
    out = []
    for k in range(n):
        kv = min(int(round(k * Rs[0].dt * fps)), len(frames) - 1)
        tiles = [label(frames[kv], 'video')] + [label(R.render(k), t) for R, t in zip(Rs, titles)]
        tiles += [np.zeros_like(tiles[0])] * (4 - len(tiles))
        out.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0))
    imageio.mimsave(out_path, out, fps=int(round(1 / Rs[0].dt)), macro_block_size=1, quality=7)
    return out_path


def stills(clip, conds, frames, out_path, times=(0.0, 0.5, 1.0), turn=60, scale=0.5):
    """Rows = times; columns = video, then scope + orbit per condition."""
    Rs = [Renderer(clip, c) for c in conds]
    n = min(R.steps for R in Rs)
    rows = []
    for t in times:
        k = int(round(t * (n - 1)))
        kv = min(int(round(k * Rs[0].dt * clip['fps'])), len(frames) - 1)
        row = [label(frames[kv], f'video t={kv / clip["fps"]:.1f}s')]
        for R, c in zip(Rs, conds):
            row += [label(R.render(k), LABELS[c]), label(R.render(k, turn), f'{LABELS[c]}, orbit')]
        rows.append(np.concatenate(row, 1))
    img = np.concatenate(rows, 0)
    imageio.imwrite(out_path, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), quality=85)
    return out_path


def organ_preview(clip, conds, out_path, turns=(60, 120), scale=0.5):
    """Each condition's organ body at rest (before simulation): textured through the scope, then the bare geometry
    (organ in one colour, background dimmed) from orbit cameras."""
    S = SIM.load_scene(clip)
    cam = clip.camera()
    rows = []
    for c in conds:
        o = SIM.load_organ(clip.cond_dir(c) / 'organ.npz')
        m = mujoco.MjModel.from_xml_string(SIM.scene_xml(clip, S, o, SIM.PARAMS, clip.cond_dir(c)))
        d = mujoco.MjData(m)
        SIM.apply(m, d, clip, S['actions'][0], ctrl=False)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, cam.H, cam.W)
        SIM.set_camera(m, cam, S, 0, clip['fps'])
        r.update_scene(d, camera='endo')
        row = [label(SIM.camera_look(r.render()), LABELS[c])]
        # bare geometry: organ material without texture, background and strands greyed
        for fi in range(m.nflex):                     # flexes: own colour, no material
            m.flex_matid[fi] = -1
            m.flex_rgba[fi] = [0.85, 0.72, 0.25, 1] if fi == 0 else [0.62, 0.42, 0.72, 1]
        mb = m.material('m_back').id
        m.mat_texid[mb] = -1
        m.mat_rgba[mb] = [0.42, 0.43, 0.45, 1]
        m.mat_emission[mb] = 0.2
        vis = o['X'][~o['anchors']] if (~o['anchors']).any() else o['X']
        piv = vis.mean(0)
        _, z0 = cam.project(piv, S['cam_R'][0], S['cam_f'][0])
        fwd = S['cam_R'][0][2]
        for t in turns:
            oc = mujoco.MjvCamera()
            oc.type = mujoco.mjtCamera.mjCAMERA_FREE
            oc.lookat[:] = piv
            oc.distance = 1.5 * float(z0)
            oc.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + t
            oc.elevation = -float(np.degrees(np.arcsin(-fwd[2]))) + 8
            r.update_scene(d, camera=oc)
            row.append(label(r.render(), f'geometry, orbit {t} deg'))
        rows.append(np.concatenate(row, 1))
    img = np.concatenate(rows, 0)
    imageio.imwrite(out_path, cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), quality=85)
    return out_path
