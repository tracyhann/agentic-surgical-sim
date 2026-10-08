"""Commons clip 'gallbladder torsion' (CC BY 2.0, Boer, Boerma, de Vries Reilingh; J Med Case Rep 2011), 26-32 s.
A grasper from the upper left holds the gallbladder neck; a straight shiny instrument from the upper right sweeps the
tissue. The laparoscope is nearly static in this window (ORB/RANSAC background homography: 15-60 px corner motion per
second vs 100-2500 px elsewhere in the clip). Frames are cropped to the 640x360 picture (letterbox removed).

GT (hidden): probe tip track per frame (bright elongated component entering from the right edge, tip = its lower-left
extreme), grasper jaw-tip keyframes annotated by hand on a pixel grid before any agent run (+-20 px).
"""
import json
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
INST = 'chole_sweep'
F0, F1, FPS = 650, 800, 25
CROP = (59, 419)
GRASPER_KEYS = {650: (238, 150), 700: (215, 95), 750: (190, 80), 800: (215, 70)}   # frame -> (u, v), manual


def probe_tip(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > 170) & (hsv[..., 1] < 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a > 400 and (x + w >= f.shape[1] - 3 or y <= 3) and w > 120:   # long bright component entering at the right/top edge
            if best is None or a > st[best, 4]:
                best = i
    if best is None:
        return np.array([np.nan, np.nan])
    ys, xs = np.nonzero(lab == best)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    d = d if d[0] < 0 else -d                                          # pointing towards the lower-left tip
    s = (pts - c) @ d
    return pts[s > np.percentile(s, 99.5)].mean(0)


def main():
    r = imageio.get_reader(ROOT / 'data/real/commons/gallbladder_torsion_S1.mp4')
    frames = [r.get_data(k)[CROP[0]:CROP[1]] for k in range(F0, F1 + 1)]
    task = ROOT / 'runs_real' / INST / 'task'
    task.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(task / 'video.mp4', frames, fps=FPS, macro_block_size=1, quality=8)
    tips = np.array([probe_tip(f) for f in frames])
    gt = ROOT / 'private_real' / 'gt' / INST
    gt.mkdir(parents=True, exist_ok=True)
    keys = np.array([[k - F0, u, v] for k, (u, v) in GRASPER_KEYS.items()], float)
    np.savez(gt / 'gt.npz', probe_px=tips, grasper_keys=keys)
    vis = []
    for k in (0, 50, 100, 150):
        f = frames[k].copy()
        if np.isfinite(tips[k]).all():
            cv2.circle(f, tuple(int(c) for c in tips[k]), 7, (0, 255, 0), 2)
        if F0 + k in GRASPER_KEYS:
            cv2.circle(f, GRASPER_KEYS[F0 + k], 7, (255, 255, 0), 2)
        vis.append(f)
    imageio.imwrite(gt / 'gt_overlay.png', np.concatenate([np.concatenate(vis[:2], 1), np.concatenate(vis[2:], 1)], 0))
    ann = dict(instance=INST, source='Wikimedia Commons, gallbladder torsion case report video S1 (CC BY 2.0)',
               frames=[F0, F1], fps=FPS, width=640, height=360, probe_tracked_frac=float(np.isfinite(tips[:, 0]).mean()),
               probe_px_start=tips[0].round(1).tolist(), probe_px_end=tips[-1].round(1).tolist(),
               grasper_keys={str(k - F0): v for k, v in GRASPER_KEYS.items()})
    (gt / 'annotation.json').write_text(json.dumps(ann, indent=1))
    print(json.dumps(ann, indent=1))


if __name__ == '__main__':
    main()
