"""ROSMA X01 Post and Sleeve trial 1, 8-25 s: bimanual hand-off of the yellow sleeve. Builds the task video and the GT.

GT (hidden):  sleeve 2D centroid per frame (colour segmentation), PSM1/PSM2 tool-tip 3D positions (kinematics, metres,
each in its own PSM base frame) resampled at the video frame times. Video/kinematics sync comes from the on-screen
clock: 17:02:37.000 falls at video frame 9.5 (checked again at 17:04:36 -> frame 1794.5, no drift). On top of the
clock, the kinematics are shifted by KIN_SHIFT frames: smoothed sleeve-image speed vs PSM tool-tip speed cross-
correlates best at -7 frames for both arms (webcam latency / NTP offset).
"""
import csv, json, shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'data/real/rosma/X01_Post_and_Sleeve_01'
INST = 'rosma_handoff'
F0, F1, FPS = 120, 375, 15
T_ANCHOR, K_ANCHOR = datetime(2020, 2, 4, 17, 2, 37).timestamp(), 9.5
KIN_SHIFT = 7          # frames; see module docstring


def frame_time(k):
    return T_ANCHOR + (k - K_ANCHOR) / FPS


def sleeve_track(frames):
    out = []
    for f in frames:
        m = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_RGB2HSV), (18, 90, 90), (38, 255, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        big = [i for i in range(1, n) if st[i, 4] > 2000]
        out.append(cen[max(big, key=lambda i: st[i, 4])] if big else [np.nan, np.nan])
    return np.array(out)


def kinematics():
    rows = list(csv.reader(open(f'{SRC}.csv')))
    h, body = rows[0], [r for r in rows[1:] if not r[0].startswith('Date')]
    t = np.array([datetime.strptime(r[0], '%Y-%m-%d.%H:%M:%S.%f').timestamp() for r in body])
    X = np.array([[float(v) for v in r[1:]] for r in body])
    col = lambda n: X[:, h.index(n) - 1]
    return t, {p: np.stack([col(f'{p}_position_{a}') for a in 'xyz'], 1) for p in ('PSM1', 'PSM2')}


def main():
    gt = ROOT / 'private_real' / 'gt' / INST
    gt.mkdir(parents=True, exist_ok=True)
    r = imageio.get_reader(f'{SRC}.mp4')
    frames = [r.get_data(k) for k in range(F0, F1 + 1)]
    task = ROOT / 'runs_real' / INST / 'task'
    task.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(task / 'video.mp4', frames, fps=FPS, macro_block_size=1, quality=8)
    sleeve = sleeve_track(frames)
    tk, P = kinematics()
    tf = np.array([frame_time(k + KIN_SHIFT) for k in range(F0, F1 + 1)])
    tips = {p: np.stack([np.interp(tf, tk, v[:, a]) for a in range(3)], 1) for p, v in P.items()}
    # sync check: sleeve image speed vs PSM speeds, cross-correlated over +-1 s
    sm = lambda x: np.convolve(x, np.ones(7) / 7, mode='same')
    sv = sm(np.linalg.norm(np.gradient(sleeve, axis=0), axis=1))
    lags, n = range(-15, 16), len(sv)
    corr = {}
    for p, v in tips.items():
        tv = sm(np.linalg.norm(np.gradient(v, axis=0), axis=1))
        corr[p] = [float(np.corrcoef(sv[max(0, L):n + min(0, L)], tv[max(0, -L):n - max(0, L)])[0, 1]) for L in lags]
    best = {p: list(lags)[int(np.nanargmax(c))] for p, c in corr.items()}
    np.savez(gt / 'gt.npz', sleeve_px=sleeve, psm1=tips['PSM1'], psm2=tips['PSM2'], t=tf - tf[0])
    ann = dict(instance=INST, source='ROSMA X01_Post_and_Sleeve_01 (CC BY 4.0)', frames=[F0, F1], fps=FPS, kin_shift_frames=KIN_SHIFT,
               width=frames[0].shape[1], height=frames[0].shape[0], sync_best_lag_frames=best,
               sync_peak_corr={p: round(max(c), 3) for p, c in corr.items()},
               sleeve_px_start=sleeve[0].round(1).tolist(), sleeve_px_end=sleeve[-1].round(1).tolist(),
               psm1_path_len_m=float(np.linalg.norm(np.diff(tips['PSM1'], axis=0), axis=1).sum()),
               psm2_path_len_m=float(np.linalg.norm(np.diff(tips['PSM2'], axis=0), axis=1).sum()))
    (gt / 'annotation.json').write_text(json.dumps(ann, indent=1))
    print(json.dumps(ann, indent=1))


if __name__ == '__main__':
    main()
