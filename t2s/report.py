"""Media for the v2 result page: per clip one video with the original frames, the SAM 3 masks, the 4D reconstruction
rendered through the scope camera (shaded, on black) and outlined on the video; chole_a also gets the simulation.

    python -m t2s.report <clip> [sim=<round/config>]   -> site/v2/media/<clip>.mp4 (+ poster jpg)

Memory-light: one clip per process, frames streamed to the encoder.
"""
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

from . import data as D

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'site' / 'v2' / 'media'
COLORS = [(255, 210, 40), (80, 200, 255), (240, 90, 200), (120, 230, 120), (255, 140, 60), (170, 130, 255),
          (60, 255, 220), (255, 90, 90), (200, 200, 200), (255, 255, 120)]


def latest(clip, kind, name=None):
    d = D.OUT / clip / kind / name if name else D.OUT / clip / kind
    vs = sorted(p.name for p in d.glob('v*') if (p / 'model.npz').exists())
    return vs[-1] if vs else None


def painter(img, V, k, meshes):
    """Flat-shaded triangles of several meshes drawn far-to-near into img (H, W, 3). meshes: [(X, F, rgb), ...]."""
    polys = []
    for X, F, rgb in meshes:
        q, z = V.project(X, k)
        ok = (z[F] > 1e-4).all(1)
        n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        view = X[F].mean(1) - V.pos[k]
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        sh = 0.35 + 0.65 * np.abs((n * view).sum(1))
        for fi in np.nonzero(ok)[0]:
            polys.append((z[F[fi]].mean(), q[F[fi]].astype(np.int32), tuple(float(c * sh[fi]) for c in rgb)))
    polys.sort(key=lambda t: -t[0])
    for _, p, c in polys:
        cv2.fillConvexPoly(img, p, c)
    return img


def tube(a, b, r, n=8, segs=16):
    """Cylinder mesh from a to b in `segs` rings (a shaft that runs past the camera is drawn up to there)."""
    d = b - a
    d = d / (np.linalg.norm(d) + 1e-12)
    u = np.cross(d, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    X = np.vstack([a + (b - a) * t + ring for t in np.linspace(0, 1, segs + 1)])
    F = []
    for s_ in range(segs):
        for j in range(n):
            j2 = (j + 1) % n
            F += [[s_ * n + j, (s_ + 1) * n + j, s_ * n + j2], [s_ * n + j2, (s_ + 1) * n + j, (s_ + 1) * n + j2]]
    return X, np.array(F)


def hull(X):
    from scipy.spatial import ConvexHull
    h = ConvexHull(X)
    return X, h.simplices


def outline(img, mask, rgb, thick=1):
    cnt = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return cv2.drawContours(np.ascontiguousarray(img), cnt, -1, rgb, thick)


def label(img, text):
    cv2.putText(img, text, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(argv):
    clip = argv[0]
    kv = dict(a.split('=', 1) for a in argv[1:] if '=' in a)
    MEDIA.mkdir(parents=True, exist_ok=True)
    frames, valid, info = D.load(clip)
    n, H, W = frames.shape[:3]
    segv = kv.get('seg') or sorted(p.name for p in (D.OUT / clip / 'seg').glob('v*') if (p / 'masks.npz').exists())[-1]
    zm = np.load(D.OUT / clip / 'seg' / segv / 'masks.npz')
    masks = {nm: np.unpackbits(zm[nm], axis=-1)[..., :W].astype(bool) for nm in zm.files}
    has3d = (ROOT / 'outputs' / 'variants' / f'{clip}_t2s+sift' / 'prep' / 'multiview' / 'frames.npz').exists() and (D.OUT / clip / 'organs').exists()
    organs, ins, bodies, sim, V = {}, None, [], None, None
    if has3d:
        from . import views2
        V = views2.load(clip)
        for d in sorted((D.OUT / clip / 'organs').glob('*')):
            ver = kv.get(f'organ_{d.name}') or latest(clip, 'organs', d.name)
            if ver:
                z = np.load(d / ver / 'model.npz')
                organs[d.name] = (z['verts4d'].astype(float), z['faces'], ver)
        iv = kv.get('instruments') or latest(clip, 'instruments')
        if iv:
            ins = np.load(D.OUT / clip / 'instruments' / iv / 'model.npz')
        bv = kv.get('background') or latest(clip, 'background')
        if bv:
            zb = np.load(D.OUT / clip / 'background' / bv / 'model.npz')
            for i in range(len(zb['body_names']) if 'body_names' in zb.files else 0):
                Xb, pc = zb[f'body{i}_verts'].astype(float), zb[f'body{i}_piece']
                for p in np.unique(pc):
                    bodies.append(hull(Xb[pc == p]))
        if 'sim' in kv:
            tr = np.load(D.OUT / clip / 'rounds' / kv['sim'] / 'traj.npz')
            f0 = int(tr['frames'][0])
            sim = ({k[len('organ_'):]: tr[k].astype(float) for k in tr.files if k.startswith('organ_')}, f0, int(tr['frames'][1]))
    rows = 1 + (1 if has3d else 0) + (1 if sim else 0)
    w = iio.get_writer(MEDIA / f'{clip}.mp4', fps=info['fps'], macro_block_size=1, quality=6, output_params=['-movflags', '+faststart'])
    half = (W // 2, H // 2)

    def scene_meshes(k, organ_pos):
        ms = [(X, F, (150, 70, 70)) for X, F in bodies]
        for c, (nm, X) in zip([(235, 200, 60), (120, 200, 240)], organ_pos.items()):
            ms.append((X, organs[nm][1], c))
        if ins is not None:
            for nm in [str(s) for s in ins['names']]:
                if not bool(ins[f'{nm}__visible'][k]) and not bool(ins[f'{nm}__filled'][k]):
                    continue
                tip, dr = ins[f'{nm}__tip'][k], ins[f'{nm}__dir'][k]
                pr = json.loads(str(ins[f'{nm}__params']))
                ms.append((*tube(tip - dr * 0.12, tip, pr['radius']), (190, 195, 200)))
        return ms
    for k in range(n):
        f = frames[k]
        segim = f.astype(np.float32).copy()
        for c, (nm, m) in zip(COLORS * 2, masks.items()):
            segim[m[k]] = 0.5 * segim[m[k]] + 0.5 * np.array(c, np.float32)
        segim = np.clip(segim, 0, 255).astype(np.uint8)
        for c, (nm, m) in zip(COLORS * 2, masks.items()):
            segim = outline(segim, m[k], c)
        tiles = [[label(cv2.resize(f, half), 'video'), label(cv2.resize(segim, half), f'SAM 3 masks ({len(masks)} objects)')]]
        if has3d:
            pos = {nm: v[0][k] for nm, v in organs.items()}
            ms = scene_meshes(k, pos)
            ren = painter(np.zeros((H, W, 3), np.uint8) + 18, V, k, ms)
            over = painter(f.copy(), V, k, ms)
            over = (0.55 * f + 0.45 * over).astype(np.uint8)
            tiles.append([label(cv2.resize(ren, half), '4D reconstruction'), label(cv2.resize(over, half), 'reconstruction on video')])
        if sim:
            kk = min(max(k, sim[1]), sim[2]) - sim[1]
            pos = {nm: X[kk] for nm, X in sim[0].items()}
            ms = scene_meshes(k, pos)
            ren = painter(np.zeros((H, W, 3), np.uint8) + 18, V, k, ms)
            over = (0.55 * f + 0.45 * painter(f.copy(), V, k, ms)).astype(np.uint8)
            tiles.append([label(cv2.resize(ren, half), 'physics simulation'), label(cv2.resize(over, half), 'simulation on video')])
        out = np.concatenate([np.concatenate(r, 1) for r in tiles], 0)
        out = out[:out.shape[0] // 2 * 2, :out.shape[1] // 2 * 2]
        w.append_data(out)
        if k == int(n * 0.6):
            iio.imwrite(MEDIA / f'{clip}_poster.jpg', out, quality=82)
    w.close()
    print(clip, 'rows', rows, 'size', out.shape, round((MEDIA / f'{clip}.mp4').stat().st_size / 1e6, 2), 'MB', {nm: v[2] for nm, v in organs.items()})


if __name__ == '__main__':
    main(sys.argv[1:])
