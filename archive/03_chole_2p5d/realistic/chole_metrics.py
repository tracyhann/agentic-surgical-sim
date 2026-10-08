"""How close is the simulated clip to the real one?

  python chole_metrics.py            (after chole_recon.py and chole_sim.py)
tissue      LK feature tracks on the real video (forward-backward checked) vs the same material points in the sim
            (barycentric on the flex sheet; points on the static backdrop stay put); also the 'nothing moves' baseline
image       per-frame SSIM / PSNR of the rendered clip vs the real clip, and of the static first frame vs the real clip
instruments tip pixel error of the simulated TCPs vs the tracked tips
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chole_recon as C  # noqa: E402

OUT = C.OUT


def real_tracks(frames, exclude):
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mask = (~exclude).astype(np.uint8) * 255
    p0 = cv2.goodFeaturesToTrack(g[0], 400, 0.01, 8, mask=mask)
    P = np.full((len(g), len(p0), 2), np.nan, np.float32)
    P[0] = p0[:, 0]
    alive = np.ones(len(p0), bool)
    cur = p0
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for k in range(1, len(g)):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g[k - 1], g[k], cur, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g[k], g[k - 1], nxt, None, **lk)
        fb = np.linalg.norm(back - cur, axis=2)[:, 0]
        alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 1.0)
        P[k][alive] = nxt[alive, 0]
        cur = nxt
    return P


def barycentric(px, tri_px):
    a, b, c = tri_px
    v0, v1, v2 = b - a, c - a, px - a
    d = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(d) < 1e-9:
        return None
    l1 = (v2[0] * v1[1] - v1[0] * v2[1]) / d
    l2 = (v0[0] * v2[1] - v2[0] * v0[1]) / d
    w = np.array([1 - l1 - l2, l1, l2])
    return w if (w >= -1e-6).all() else None


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))


def evaluate(write=True, image=True):
    frames = [f for f in imageio.get_reader(C.ROOT / 'runs_real/chole_sweep/task/video.mp4')]
    n = len(frames)
    tr = C.tracks(frames)
    instr = np.zeros(frames[0].shape[:2], bool)
    for key in ('probe', 'grasper'):
        for m in tr[key]['masks']:
            instr |= m
    instr = cv2.dilate(instr.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    P = np.load(OUT / 'real_tracks.npy')
    g = np.load(OUT / 'geometry.npz')
    flex = np.load(OUT / 'flex_traj.npy')                       # (steps at 20 Hz, nv, 3)
    sim_t = np.arange(len(flex)) * 0.05
    src_t = np.arange(n) / C.FPS
    flex_n = np.stack([np.stack([np.interp(src_t, sim_t, flex[:, v, a]) for a in range(3)], -1) for v in range(flex.shape[1])], 1)
    tri, spx = g['sheet_tri'], g['sheet_px']
    nv_t = len(g['sheet_X'])
    has_ribbon = 'ribbon_tri' in g.files and flex.shape[1] >= nv_t + len(g['ribbon_X'])
    region = {'gallbladder': g['gb'], 'fold': g['fold'] & ~g['gb']}
    rows = []
    for j in range(P.shape[1]):
        p0 = P[0, j]
        w = None
        on_band = has_ribbon and g['band'][int(p0[1]), int(p0[0])]
        tris, pxs, off = (g['ribbon_tri'], g['ribbon_px'], flex.shape[1] - len(g['ribbon_X'])) if on_band else (tri, spx, 0)
        for t in tris:
            w = barycentric(p0, pxs[t])
            if w is not None:
                t = t + off
                break
        if w is not None:
            X = (flex_n[:, t, :] * w[None, :, None]).sum(1)
            sim_px = np.array([C.project(X[k], g['cam_R'][k], g['cam_f'][k])[0] for k in range(n)])
            sim_px += p0 - sim_px[0]                                # same starting pixel
            reg = 'strand' if on_band else 'gallbladder' if region['gallbladder'][int(p0[1]), int(p0[0])] else 'fold'
        else:
            X0 = C.unproject(p0[0], p0[1], g['zb'][int(p0[1]), int(p0[0])])
            sim_px = np.array([C.project(X0, g['cam_R'][k], g['cam_f'][k])[0] for k in range(n)])
            reg = 'static backdrop'
        ok = np.isfinite(P[:, j, 0])
        if ok.sum() < 20:
            continue
        real_disp = np.linalg.norm(P[ok, j] - p0, axis=1)
        rows.append(dict(region=reg, err=float(np.linalg.norm(sim_px[ok] - P[ok, j], axis=1).mean()),
                         static_err=float(real_disp.mean()), real_max_disp=float(real_disp.max()),
                         sim_max_disp=float(np.linalg.norm(sim_px - p0, axis=1).max())))
    tissue = {}
    for reg in ('gallbladder', 'fold', 'strand', 'static backdrop'):
        r = [x for x in rows if x['region'] == reg]
        if r:
            tissue[reg] = dict(n_tracks=len(r), sim_err_px=round(float(np.median([x['err'] for x in r])), 2),
                               static_baseline_px=round(float(np.median([x['static_err'] for x in r])), 2),
                               real_max_disp_px=round(float(np.median([x['real_max_disp'] for x in r])), 1),
                               sim_max_disp_px=round(float(np.median([x['sim_max_disp'] for x in r])), 1))
    allr = rows
    tissue['all'] = dict(n_tracks=len(allr), sim_err_px=round(float(np.median([x['err'] for x in allr])), 2),
                         static_baseline_px=round(float(np.median([x['static_err'] for x in allr])), 2))
    if not image:
        return dict(tissue_tracks=tissue)
    sim_frames = np.load(OUT / 'sim_frames.npy')
    idx = np.round(np.linspace(0, len(sim_frames) - 1, n)).astype(int)
    gray = lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    s_sim = [ssim(gray(sim_frames[i]), gray(f)) for i, f in zip(idx, frames)]
    s_static = [ssim(gray(frames[0]), gray(f)) for f in frames]
    p_sim = [psnr(sim_frames[i], f) for i, f in zip(idx, frames)]
    p_static = [psnr(frames[0], f) for f in frames]
    # strand shape: purple-band masks (same HSV rule, same region) in the real frames vs the rendered frames
    def iou(a, b):
        return float((a & b).sum() / max((a | b).sum(), 1))
    band_sim = [iou(C.band_mask(sim_frames[i]), C.band_mask(f)) for i, f in zip(idx, frames)]
    band_static = [iou(C.band_mask(frames[0]), C.band_mask(f)) for f in frames]
    image = dict(strand_iou_sim=round(float(np.mean(band_sim)), 3), strand_iou_static_frame0=round(float(np.mean(band_static)), 3),
                 ssim_sim=round(float(np.mean(s_sim)), 3), ssim_static_frame0=round(float(np.mean(s_static)), 3),
                 psnr_sim=round(float(np.mean(p_sim)), 2), psnr_static_frame0=round(float(np.mean(p_static)), 2),
                 ssim_sim_first=round(s_sim[0], 3), ssim_sim_last=round(s_sim[-1], 3), ssim_static_last=round(s_static[-1], 3))
    res = dict(tissue_tracks=tissue, image=image)
    if write:
        (OUT / 'metrics.json').write_text(json.dumps(res, indent=1))
        np.save(OUT / 'real_tracks.npy', P)
    return res


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=1))
