"""Step 3, r03: a second camera + metric depth solution for clips whose v1-style multi-view solve is weak
(chole_derot), written as the geometry variant 'sift2' in the same format as r2s.multiview (frames.npz), so that
t2s.views2.load(clip, 'sift2') reads it (importing this module registers the variant with r2s.config).

    PYTHONPATH=. .venv/bin/python -m t2s.geomfix solve <clip> [--tag sift2] [--holdout-far 100]
    PYTHONPATH=. .venv/bin/python -m t2s.geomfix eval <clip> [tags ...]     -> outputs/t2s/<clip>/geometry/

What differs from r2s.multiview mode 'sift' (r02 findings on chole_derot; r2s itself is not changed):
 1. Static pixels. The r2s config written by t2s.geom marks every spec organ as 'strand' (moving) and every other
    object as 'static': in chole_derot the liver (20 % of the image, static) was excluded and the neck / pedicle (pulled
    by the dissector) and the fat were used. Here: movers = instruments + organs + background bodies that move, as the
    background agent classified them (flow residual test); static = scope area minus the movers dilated by 15 px,
    minus specular highlights for features; after a first solve, pixels whose observed flow disagrees with the
    camera-induced flow (moving tissue the masks miss) are dropped too.
 2. Focal length fixed to another clip's solve of the same video and scope (chole_a: 568.9 px) instead of fitted.
 3. Metric gauge. Reprojection and relative depth are scale-free; only the rulers (5 mm shafts) hold the scale, weakly,
    and the old solve kept the single-view calibration made with the assumed 47 deg (414 px), i.e. depth 28 % nearer
    than the shafts imply at its own focal length. Here the depth starts from the calibration at the fixed focal
    length and, after each solve, the whole solution (depth and translations) is multiplied by the robust median of
    (shaft depth from apparent width) / (model depth on the shaft) over all ruler samples.
 4. Wide-baseline matches: SIFT with relaxed thresholds (6000 features, contrast 0.01, ratio 0.8, F-MAGSAC 1.5 px,
    >= 15 inliers) on the new static pixels between all keyframe pairs, then guided matching (descriptor match among
    the keypoints within 12 px of the position predicted by the previous solution, then F-verified) and a check of
    every wide match against the previous solution's transfer (> 8 px: dropped).
 5. Keyframes every 5 frames (r2s: 10), flow tracks up to 4 keyframes ahead.
 6. Every frame gets its own pose and depth affine (+ smooth correction grid), fitted to the static structure of the
    two bracketing keyframes (flow tracks from both, reprojection + relative depth), instead of interpolating poses
    and the depth affine between keyframes (Depth Anything's normalisation changes from frame to frame when the large
    gallbladder turns: that interpolation is what made the per-frame depth scale spread 0.74-1.27).

Evaluation (eval): every geometry is read through its frames.npz + the shared Depth Anything output (what the other
agents load), on one common static mask (labels only: liver + unlabelled minus dilated movers), keyframes every 10
frames as in v1:
  probe      held-out ORB matches (a detector the solve does not use) between keyframe pairs, F-verified; median
             transfer error (px) through frame i's depth and both cameras, by keyframe gap 1-2 / 3-9 / >= 10
  ruler      (depth map on the shaft - f D / width) / (f D / width), all frames, signed and absolute median
  scale      per frame: median over static points of the other frames (every 3rd) of (their depth in this frame) /
             (this frame's depth there): p5 / p95 / min / max and MAD of the per-frame values
  photo      frame k warped into k+10 on static pixels in both: NCC and local NCC (15x15), median over k
  freespace  5 mm shaft cross-sections (axis depth f D / w): fraction where the depth map 16 px beside the mask is
             > 2 mm in front of the shaft's front surface (tissue in front of a shaft that is seen in front of it),
             and the scale factor that would leave 10 % behind (the instrument agent's kappa rule)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

from r2s import config as RC
from r2s import multiview as MV
from r2s import perception
from r2s.scene import dil
from . import data as D
from .geom import r2s_name

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'sift2'
RC.VARIANTS.setdefault(VARIANT, {'multiview': 'geomfix'})       # read by r2s.perception.metric_depth (truthy)
RC.VARIANT_TITLES.setdefault(VARIANT, '多视角·SIFT2 (t2s.geomfix)')
for _t in ('sift2_holdout', 'sift2_r1', 'sift2_r2'):              # diagnostic variants of the same solve
    RC.VARIANTS.setdefault(_t, {'multiview': 'geomfix'})
    RC.VARIANT_TITLES.setdefault(_t, _t)

CFG = dict(every=5, window=4, n_pts=500, dil_px=15, grid=(4, 6), focal_from='chole_a', motion_ratio=1.5,
           sift_nf=6000, sift_ct=0.01, ratio=0.8, f_thresh=1.5, min_inliers=15, guided_radius=12.0, guided_ratio=0.85,
           wide_check_px=8.0, rounds=2, iters=300, adam_iters=2000, w_ruler=3.0, motion_px=4.0, motion_gap=3,
           frame_w_depth=10.0, frame_min_obs=40, smooth=True, sigma_acc_mm=0.25, sigma_acc_deg=0.1, smooth_iters=500)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------ inputs
class Ctx:
    """Frames, masks, Depth Anything output and the base camera of a clip (the shared perception)."""

    def __init__(self, clip):
        self.name = clip
        self.base = RC.Clip(r2s_name(clip))
        self.frames, self.valid, self.info = D.load(clip)
        self.n, self.H, self.W = self.frames.shape[:3]
        self.masks = perception.load_masks(self.base)
        self.names = self.base.objects
        self.disp = np.load(self.base.prep / 'disp.npy').astype(np.float32)
        self.cams = dict(np.load(self.base.prep / 'cams.npz'))
        self.cam = self.base.camera()
        self.instruments = [i['mask'] for i in self.base.instruments]
        self.shaft_d = {i['mask']: i['shaft_d'] for i in self.base.instruments}
        self.movers = movers(clip, self.base)
        mi = [self.names.index(m) for m in self.movers]
        self.moving = np.any(self.masks[:, mi], 1)
        self.gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in self.frames])

    def static(self, dil_px=CFG['dil_px']):
        return np.stack([self.valid & ~dil(self.moving[k], dil_px) for k in range(self.n)])

    def specular(self, k):
        hsv = cv2.cvtColor(self.frames[k], cv2.COLOR_RGB2HSV)
        return dil((hsv[..., 2] > 230) & (hsv[..., 1] < 60), 9)


def movers(clip, base):
    """Objects that move: instruments + organs + background bodies whose flow residual is >= motion_ratio x that of
    the unlabelled pixels (the background agent's classification, latest version); without it: r2s roles."""
    bg = sorted((D.OUT / clip / 'background').glob('v*/quality.json'))
    out = [i['mask'] for i in base.instruments]
    if bg:
        q = json.loads(bg[-1].read_text())
        for n, o in q['objects'].items():
            r = (o.get('motion') or {}).get('ratio')
            if o['kind'] == 'organ' or (o['kind'] in ('body', 'candidate') and r is not None and r >= CFG['motion_ratio']):
                out.append(n)
    else:
        out += base.role('organ') + base.role('strand')
    return [n for n in dict.fromkeys(out) if n in base.objects]


def reference_focal(clip):
    src = CFG['focal_from']
    if not src or src == clip:
        return None
    rep = json.loads((RC.OUTPUTS / 'variants' / f'{r2s_name(src)}+sift' / 'prep' / 'multiview' / 'report.json').read_text())
    return float(rep['focal_px'])


def bsample(img, p):
    """Bilinear sample of a (H, W) image at float pixel positions p (N, 2) (numpy: no size limit, edges clamped)."""
    H, W = img.shape
    x = np.clip(p[:, 0], 0, W - 1.000001)
    y = np.clip(p[:, 1], 0, H - 1.000001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    return (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x0 + 1] * wx * (1 - wy) + img[y0 + 1, x0] * (1 - wx) * wy
            + img[y0 + 1, x0 + 1] * wx * wy)


def corr_field(c, H, W):
    """A depth-correction grid as a (H, W) field. Coarse grids (the bundle adjustment's, nodes on the image corners,
    KeyframeBA.correction) are upsampled corner-aligned; fine grids (written to frames.npz) with cv2.resize as
    r2s.multiview.depth_frames does (at 1/4 resolution the two conventions agree to < 1 px)."""
    gy, gx = c.shape
    if gy >= 20:
        return cv2.resize(c.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    y = np.linspace(0, gy - 1, H)
    x = np.linspace(0, gx - 1, W)
    y0 = np.clip(np.floor(y).astype(int), 0, gy - 2)
    x0 = np.clip(np.floor(x).astype(int), 0, gx - 2)
    wy, wx = (y - y0)[:, None], (x - x0)[None, :]
    return (c[y0][:, x0] * (1 - wy) * (1 - wx) + c[y0][:, x0 + 1] * (1 - wy) * wx + c[y0 + 1][:, x0] * wy * (1 - wx)
            + c[y0 + 1][:, x0 + 1] * wy * wx)


def fine_corr(corr, H, W, f=4):
    """(n, gy, gx) corner-aligned grids -> (n, H/f, W/f) grids that cv2.resize turns into the same field."""
    h, w = H // f, W // f
    out = np.zeros((len(corr), h, w), np.float32)
    for k, c in enumerate(corr):
        F = corr_field(c, H, W)
        out[k] = cv2.resize(F, (w, h), interpolation=cv2.INTER_AREA)
    return out


# ------------------------------------------------------------------ a solution at keyframe or frame level
class Sol:
    """Cameras x_k = R_k X + t_k (X in keyframe-0 / frame-0 camera coordinates), one focal length, and the depth model
    1/z = a_k d + b_k times exp(bilinear corr_k) on the Depth Anything output d, for frames `idx`."""

    def __init__(self, idx, R, t, f, a, b, corr, disp, W, H):
        self.idx, self.R, self.t, self.f, self.a, self.b, self.corr = list(idx), R, t, float(f), a, b, corr
        self.disp, self.W, self.H = disp, W, H
        self.pos = {k: i for i, k in enumerate(self.idx)}

    def depth(self, k):
        i = self.pos[k]
        z = 1.0 / np.clip(self.a[i] * self.disp[k] + self.b[i], 1.0, None)
        c = self.corr[i]
        if c.shape[0] > 1:
            z = z * np.exp(corr_field(c, self.H, self.W))
        return z

    def transfer(self, k, j, P, z=None):
        """Pixels P of frame k (with depth z, default its model depth) into frame j: positions and depth there."""
        i, l = self.pos[k], self.pos[j]
        z = bsample(self.depth(k), P) if z is None else z
        x = np.stack([(P[:, 0] - self.W / 2) * z / self.f, (P[:, 1] - self.H / 2) * z / self.f, z], 1)
        X = (x - self.t[i]) @ self.R[i]
        y = X @ self.R[l].T + self.t[l]
        return np.stack([self.f * y[:, 0] / y[:, 2] + self.W / 2, self.f * y[:, 1] / y[:, 2] + self.H / 2], 1), y[:, 2]

    def scaled(self, s):
        return Sol(self.idx, self.R, self.t * s, self.f, self.a / s, self.b / s, self.corr, self.disp, self.W, self.H)


def sol_from_frames(path, disp, W, H, cam):
    """A per-frame solution from a frames.npz (world cameras) in the relative convention of Sol (frame 0 = identity)."""
    z = np.load(path)
    R_w, pos = z['R'].astype(float), z['pos'].astype(float)
    n = len(R_w)
    R = np.array([R_w[k] @ R_w[0].T for k in range(n)])                 # frame-0 camera -> frame-k camera
    c = (pos - pos[0]) @ R_w[0].T                                        # centres in frame-0 camera coordinates
    t = -np.einsum('kij,kj->ki', R, c)
    corr = z['corr'] if z['corr'].shape[1] > 1 else np.zeros((n, 1, 1))
    return Sol(range(n), R, t, float(np.median(z['f'])), z['a'], z['b'], corr, disp, W, H)


# ------------------------------------------------------------------ correspondences
def sift_features(ctx, static, keys):
    sift = cv2.SIFT_create(nfeatures=CFG['sift_nf'], contrastThreshold=CFG['sift_ct'])
    clahe = cv2.createCLAHE(2.0, (8, 8))
    out = []
    for k in keys:
        kp, de = sift.detectAndCompute(clahe.apply(ctx.gray[k]), (static[k] & ~ctx.specular(k)).astype(np.uint8) * 255)
        out.append((np.array([q.pt for q in kp]).reshape(-1, 2), de))
    return out


def f_verified(A, B, thresh):
    if len(A) < 8:
        return np.zeros(len(A), bool)
    try:
        return MV.verified(A, B, thresh)
    except cv2.error:
        return np.zeros(len(A), bool)


def wide_matches(feats, pairs):
    """Ratio-test SIFT matches between keyframe pairs (i, j), F-verified."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out = []
    for i, j in pairs:
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 8 or len(dj) < 8:
            continue
        good = [m[0] for m in bf.knnMatch(di, dj, k=2) if len(m) == 2 and m[0].distance < CFG['ratio'] * m[1].distance]
        if len(good) < CFG['min_inliers']:
            continue
        A, B = pi[[g.queryIdx for g in good]], pj[[g.trainIdx for g in good]]
        inl = f_verified(A, B, CFG['f_thresh'])
        if inl.sum() >= CFG['min_inliers']:
            out.append((i, j, A[inl], B[inl]))
    return out


def guided_matches(feats, sol, keys, static, pairs):
    """Descriptor matches among the keypoints of keyframe j within guided_radius px of where the previous solution
    puts keypoint p of keyframe i (best vs second best inside the radius), F-verified."""
    out = []
    r = CFG['guided_radius']
    for i, j in pairs:
        (pi, di), (pj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(pi) < 8 or len(pj) < 8:
            continue
        q, z = sol.transfer(keys[i], keys[j], pi)
        W, H = sol.W, sol.H
        ok = (z > 0) & (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1)
        if ok.sum() < CFG['min_inliers']:
            continue
        qi = q[ok].astype(int)
        ok[np.nonzero(ok)[0]] = static[keys[j]][qi[:, 1], qi[:, 0]]
        if ok.sum() < CFG['min_inliers']:
            continue
        tree = cKDTree(pj)
        A, B = [], []
        for a in np.nonzero(ok)[0]:
            nb = tree.query_ball_point(q[a], r)
            if not nb:
                continue
            d = np.linalg.norm(dj[nb] - di[a], axis=1)
            o = np.argsort(d)
            if len(o) == 1 or d[o[0]] < CFG['guided_ratio'] * d[o[1]]:
                A.append(pi[a])
                B.append(pj[nb[o[0]]])
        if len(A) < CFG['min_inliers']:
            continue
        A, B = np.array(A), np.array(B)
        inl = f_verified(A, B, CFG['f_thresh'])
        if inl.sum() >= CFG['min_inliers']:
            out.append((i, j, A[inl], B[inl]))
    return out


def stack_matches(ms):
    if not ms:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros((0, 2))
    return (np.concatenate([np.full(len(A), i) for i, j, A, B in ms]), np.concatenate([np.full(len(A), j) for i, j, A, B in ms]),
            np.concatenate([A for i, j, A, B in ms]), np.concatenate([B for i, j, A, B in ms]))


def rulers_at(ctx, keys):
    rows = []
    for i, k in enumerate(keys):
        for ins in ctx.instruments:
            for u, v, invw in perception.shaft_samples(ctx.masks[k, ctx.names.index(ins)], 1.0, 1.0, ctx.H, ctx.W):
                rows.append((i, float(ctx.disp[k, int(v), int(u)]), invw, ctx.shaft_d[ins], u, v))
    return np.array(rows) if rows else np.zeros((0, 6))


# ------------------------------------------------------------------ keyframe bundle adjustment (fixed focal length)
class BA2(MV.KeyframeBA):
    """r2s.multiview.KeyframeBA with the focal length held fixed (same scope as a well-solved clip)."""

    def __init__(self, *a, fix_f=True, w_ruler=3.0, **kw):
        super().__init__(*a, **kw)
        self.fix_f, self.w_ruler = fix_f, w_ruler
        if fix_f:
            self.logf.requires_grad_(False)

    def residuals(self, w_depth=10.0, w_ruler=None):
        return super().residuals(w_depth, self.w_ruler if w_ruler is None else w_ruler)

    def fit(self, iters=300, adam_iters=2000, log=print):
        pose = [self.rvec, self.tvec] + ([] if self.fix_f else [self.logf])
        opt = torch.optim.LBFGS(pose, lr=0.5, max_iter=iters, history_size=50, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            L = self.loss()
            L.backward()
            return L
        opt.step(closure)
        groups = [dict(params=[self.rvec], lr=2e-4), dict(params=[self.tvec], lr=2e-5),
                  dict(params=[self.a], lr=2e-3), dict(params=[self.b], lr=1e-2)]
        if not self.fix_f:
            groups.append(dict(params=[self.logf], lr=2e-4))
        if self.corr.requires_grad:
            groups.append(dict(params=[self.corr], lr=2e-3))
        adam = torch.optim.Adam(groups)
        for _ in range(adam_iters):
            adam.zero_grad()
            L = self.loss()
            L.backward()
            adam.step()

    def solution(self, keys, disp, W, H):
        R, t = self.poses()
        return Sol(keys, R.detach().numpy(), t.detach().numpy(), float(torch.exp(self.logf.detach())),
                   self.a.detach().numpy().copy(), self.b.detach().numpy().copy(), self.corr.detach().numpy().copy(), disp, W, H)


def ba_stats(ba, is_wide, far):
    with torch.no_grad():
        r_proj, r_depth, r_ruler = ba.residuals()
    e = r_proj.norm(dim=1).numpy()
    med = lambda v: round(float(np.median(v)), 3) if len(v) else None
    return dict(reproj_px_median=med(e[~is_wide]), reproj_px_median_wide=med(e[is_wide]),
                reproj_px_median_wide_far=med(e[is_wide & far]),
                depth_rel_median=med(np.abs(r_depth.numpy() / 10)),
                ruler_rel_median=med(np.abs(r_ruler.numpy() / ba.w_ruler)) if len(r_ruler) else None,
                ruler_signed_median=med(r_ruler.numpy() / ba.w_ruler) if len(r_ruler) else None)


def gauge(sol, ctx, frames=None):
    """Scale factor s that makes the median ruler ratio exactly 1: z_shaft / z_model over all shaft samples."""
    ratios = []
    for k in (sol.idx if frames is None else frames):
        Z = sol.depth(k)
        for ins in ctx.instruments:
            for u, v, zs in perception.shaft_samples(ctx.masks[k, ctx.names.index(ins)], sol.f, ctx.shaft_d[ins], ctx.H, ctx.W):
                ratios.append(zs / Z[int(v), int(u)])            # zs = f D / apparent width
    return float(np.median(ratios)), len(ratios)


# ------------------------------------------------------------------ moving pixels the masks miss
def motion_reject(ctx, sol, static, gap=CFG['motion_gap']):
    """Static pixels of frame k whose observed flow to k+gap differs from the camera-induced flow by more than
    max(motion_px, 3 x the frame's median) -> not static (dilated 9 px). Returns the new static masks and the
    rejected fraction per frame."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    H, W = ctx.H, ctx.W
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([uu.ravel(), vv.ravel()], 1).astype(float)
    out = static.copy()
    frac = np.zeros(ctx.n)
    border_free = cv2.erode(ctx.valid.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    for k in range(ctx.n):
        j = k + gap if k + gap < ctx.n else k - gap
        if k not in sol.pos or j not in sol.pos:
            continue
        fw = dis.calc(ctx.gray[k], ctx.gray[j], None)
        bw = dis.calc(ctx.gray[j], ctx.gray[k], None)
        fl = fw.reshape(-1, 2)
        Q = P + fl
        qx, qy = np.clip(Q[:, 0], 0, W - 1).astype(int), np.clip(Q[:, 1], 0, H - 1).astype(int)
        fb = np.linalg.norm(fl + bw[qy, qx], axis=1).reshape(H, W)
        q, z = sol.transfer(k, j, P)
        res = np.linalg.norm(Q - q, axis=1).reshape(H, W)
        s = static[k]
        if s.sum() < 500:
            continue
        # only where the flow is reliable (forward-backward consistent, away from the scope border and from the
        # movers' outlines, where flow smears across occlusion boundaries) does a disagreement mean motion
        reliable = (fb < 1.0) & border_free & ~dil(ctx.moving[k] | ctx.moving[j], 25)
        thr = max(CFG['motion_px'], 3 * float(np.median(res[s & reliable]))) if (s & reliable).sum() > 500 else np.inf
        bad = dil(s & reliable & (res > thr), 9) & s
        frac[k] = float(bad.sum() / s.sum())
        out[k] = s & ~bad
    return out, frac


# ------------------------------------------------------------------ per-frame poses and depth
def frame_tracks(ctx, static, k0, k1, n_pts):
    """Corners of keyframe k0 followed forward to every frame up to k1 and corners of k1 followed backward to k0
    (DIS flow, forward-backward checked, static pixels). Returns {frame: [(key, P_key, P_frame), ...]}."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    H, W = ctx.H, ctx.W
    out = {t: [] for t in range(k0 + 1, k1)}
    for key, step in ((k0, 1), (k1, -1)):
        if key >= ctx.n:                          # the tail after the last keyframe: forward tracks only
            continue
        p = cv2.goodFeaturesToTrack(ctx.gray[key], n_pts, 0.005, 7, mask=static[key].astype(np.uint8) * 255)
        if p is None:
            continue
        p0 = p[:, 0].astype(np.float64)
        cur, alive = p0.copy(), np.ones(len(p0), bool)
        t = key
        while True:
            nt = t + step
            if nt == (k1 if step > 0 else k0):
                break
            fw = dis.calc(ctx.gray[t], ctx.gray[nt], None)
            bw = dis.calc(ctx.gray[nt], ctx.gray[t], None)
            xi, yi = np.clip(cur[:, 0], 0, W - 1).astype(int), np.clip(cur[:, 1], 0, H - 1).astype(int)
            q = cur + fw[yi, xi]
            inside = (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1)
            qx, qy = np.clip(q[:, 0], 0, W - 1).astype(int), np.clip(q[:, 1], 0, H - 1).astype(int)
            back = q + bw[qy, qx]
            alive &= inside & (np.linalg.norm(back - cur, axis=1) < 1.0) & static[nt][qy, qx]
            cur = q
            t = nt
            if alive.sum():
                out[t].append((key, p0[alive], cur[alive]))
    return out


def frame_obs(ctx, ks, k, obs, f, grid):
    """Keyframe 3D points (frame-0 camera coordinates) tracked into frame k: X (M, 3), P (M, 2), Depth Anything d
    (M,) at P, and the bilinear nodes / weights (M, 4) of the correction grid at P (corner-aligned)."""
    W, H = ctx.W, ctx.H
    X, P = [], []
    for key, Pk, Pt in obs:
        z = bsample(ks.depth(key), Pk)
        i = ks.pos[key]
        x = np.stack([(Pk[:, 0] - W / 2) * z / f, (Pk[:, 1] - H / 2) * z / f, z], 1)
        X.append((x - ks.t[i]) @ ks.R[i])
        P.append(Pt)
    X, P = np.concatenate(X), np.concatenate(P)
    d = bsample(ctx.disp[k], P)
    gy, gx = grid
    cx = np.clip(P[:, 0] / W * (gx - 1), 0, gx - 1 - 1e-6)
    cy = np.clip(P[:, 1] / H * (gy - 1), 0, gy - 1 - 1e-6)
    x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
    wx, wy = cx - x0, cy - y0
    nodes = np.stack([y0 * gx + x0, y0 * gx + x0 + 1, (y0 + 1) * gx + x0, (y0 + 1) * gx + x0 + 1], 1)
    wts = np.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy], 1)
    return X, P, d, nodes, wts


def solve_frame(ctx, ks, k, fo, init, f, grid):
    """Pose (rotation vector, centre in frame-0 camera coordinates) and depth model (a, b, corr grid) of frame k from
    keyframe 3D points tracked into it: reprojection (px) + relative depth (x frame_w_depth), Huber; the correction
    grid is pulled weakly to the interpolated one and kept smooth."""
    W, H = ctx.W, ctx.H
    X, P, d, nodes, wts = fo
    gy, gx = grid
    Bm = np.zeros((len(P), gy * gx))
    for c in range(4):
        np.add.at(Bm, (np.arange(len(P)), nodes[:, c]), wts[:, c])
    rv0, c0, a0, b0, corr0 = init
    th0 = np.r_[rv0, c0 * 100, a0, b0, corr0.ravel()]
    wd = CFG['frame_w_depth']

    def unpack(th):
        return th[:3], th[3:6] / 100, th[6], th[7], th[8:]

    def fun(th):
        rv, c, a, b, cr = unpack(th)
        R = Rot.from_rotvec(rv).as_matrix()
        y = (X - c) @ R.T
        q = np.stack([f * y[:, 0] / y[:, 2] + W / 2, f * y[:, 1] / y[:, 2] + H / 2], 1)
        zm = np.exp(Bm @ cr) / np.maximum(a * d + b, 1.0)
        r_p = (q - P).ravel()
        r_d = wd * (y[:, 2] - zm) / zm
        cg = cr.reshape(gy, gx)
        r_c = np.r_[2.0 * (cr - corr0.ravel()), 2.0 * np.diff(cg, axis=0).ravel(), 2.0 * np.diff(cg, axis=1).ravel()]
        return np.r_[r_p, r_d, r_c * np.sqrt(len(P) / cr.size)]
    r = least_squares(fun, th0, loss='huber', f_scale=2.0, x_scale='jac', max_nfev=200)
    rv, c, a, b, cr = unpack(r.x)
    res = fun(r.x)
    e = np.linalg.norm(res[:2 * len(P)].reshape(-1, 2), axis=1)
    return rv, c, a, b, cr.reshape(gy, gx), float(np.median(e)), float(np.median(np.abs(res[2 * len(P):3 * len(P)]) / wd)), len(P)


def per_frame(ctx, ks, static, grid, log=print):
    """Every frame's camera and depth model: keyframes from the keyframe solution, frames in between solved against
    the two bracketing keyframes (interpolated values where too few tracks survive)."""
    n, f = ctx.n, ks.f
    keys = np.array(ks.idx)
    rel = Rot.from_matrix(ks.R)
    centres = np.array([-ks.R[i].T @ ks.t[i] for i in range(len(keys))])
    tt = np.clip(np.arange(n), keys[0], keys[-1])
    R_int = Slerp(keys, rel)(tt).as_rotvec()
    c_int = np.stack([np.interp(tt, keys, centres[:, j]) for j in range(3)], 1)
    a_int, b_int = np.interp(tt, keys, ks.a), np.interp(tt, keys, ks.b)
    corr_int = np.stack([np.stack([np.interp(tt, keys, ks.corr[:, y, x]) for x in range(grid[1])], -1) for y in range(grid[0])], 1)
    rv, c, a, b, corr = R_int.copy(), c_int.copy(), a_int.copy(), b_int.copy(), corr_int.copy()
    err, derr, nobs, solved = np.zeros(n), np.zeros(n), np.zeros(n, int), np.zeros(n, bool)
    for i, k in enumerate(keys):
        solved[k] = True
    fos = {}
    segs = [(int(keys[i]), int(keys[i + 1])) for i in range(len(keys) - 1)]
    if keys[-1] < n - 1:
        segs.append((int(keys[-1]), n))
    for k0, k1 in segs:
        tr = frame_tracks(ctx, static, k0, k1, CFG['n_pts'])
        for k in range(k0 + 1, k1):
            obs = tr[k]
            m = sum(len(o[1]) for o in obs)
            if m < CFG['frame_min_obs']:
                continue
            fos[k] = frame_obs(ctx, ks, k, obs, f, grid)
            out = solve_frame(ctx, ks, k, fos[k], (R_int[k], c_int[k], a_int[k], b_int[k], corr_int[k]), f, grid)
            rv[k], c[k], a[k], b[k], corr[k], err[k], derr[k], nobs[k] = out
            solved[k] = True
    rep = dict(frames_solved=int(solved.sum()), frames_interpolated=int((~solved).sum()),
               reproj_px_median=round(float(np.median(err[nobs > 0])), 3) if (nobs > 0).any() else None,
               depth_rel_median=round(float(np.median(derr[nobs > 0])), 4) if (nobs > 0).any() else None,
               obs_per_frame_median=int(np.median(nobs[nobs > 0])) if (nobs > 0).any() else 0)
    if CFG['smooth']:
        rep['independent'] = dict(jitter(rv, c), reproj_px_median=rep['reproj_px_median'], depth_rel_median=rep['depth_rel_median'])
        rv, c, a, b, corr, err, sm = smooth_frames(ctx, fos, keys, (rv, c, a, b, corr), corr_int, f, grid, nobs)
        rep.update(sm)
    rep.update(jitter(rv, c))
    R = Rot.from_rotvec(rv).as_matrix()
    t = -np.einsum('kij,kj->ki', R, c)
    sol = Sol(range(n), R, t, f, a, b, corr, ctx.disp, ctx.W, ctx.H)
    rep['fit_err_px'] = err
    return sol, rep


def jitter(rv, c):
    """Camera acceleration per frame^2: centre (mm) and rotation (deg), median and p90."""
    ac = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1) * 1000
    ar = np.degrees(np.linalg.norm(rv[2:] - 2 * rv[1:-1] + rv[:-2], axis=1))
    return dict(centre_acc_mm_median=round(float(np.median(ac)), 3), centre_acc_mm_p90=round(float(np.percentile(ac, 90)), 3),
                rot_acc_deg_median=round(float(np.median(ar)), 3), rot_acc_deg_p90=round(float(np.percentile(ar, 90)), 3))


def smooth_frames(ctx, fos, keys, init, corr_int, f, grid, nobs):
    """All non-keyframes jointly (torch, L-BFGS): the per-frame data terms of solve_frame plus a prior on the camera's
    acceleration (centre sigma_acc_mm per frame^2, rotation sigma_acc_deg), weighted like the median frame's
    observations. Keyframes stay at the keyframe solution (their depth defines the 3D points)."""
    dt = torch.float64
    n = len(init[0])
    gy, gx = grid
    W, H = ctx.W, ctx.H
    free = torch.tensor([k not in set(int(x) for x in keys) and k in fos for k in range(n)])
    rv0, c0, a0, b0, cr0 = [torch.tensor(np.asarray(v, float), dtype=dt) for v in init]
    cr0 = cr0.reshape(n, gy * gx)
    ci = torch.tensor(corr_int.reshape(n, gy * gx), dtype=dt)
    rv, c = rv0.clone().requires_grad_(True), (c0 * 1000).clone().requires_grad_(True)       # centres in mm
    a, b, cr = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True), cr0.clone().requires_grad_(True)
    fr = sorted(fos)
    Fi = torch.tensor(np.concatenate([np.full(len(fos[k][0]), k) for k in fr]))
    X = torch.tensor(np.concatenate([fos[k][0] for k in fr]) * 1000, dtype=dt)
    P = torch.tensor(np.concatenate([fos[k][1] for k in fr]), dtype=dt)
    d = torch.tensor(np.concatenate([fos[k][2] for k in fr]), dtype=dt)
    nodes = torch.tensor(np.concatenate([fos[k][3] for k in fr]))
    wts = torch.tensor(np.concatenate([fos[k][4] for k in fr]), dtype=dt)
    cnt = torch.tensor(np.maximum(nobs, 1), dtype=dt)
    nmed = float(np.median(nobs[nobs > 0]))
    sc, sr = CFG['sigma_acc_mm'], np.radians(CFG['sigma_acc_deg'])
    wd = CFG['frame_w_depth']
    hub = torch.nn.functional.huber_loss
    fm = free[:, None].to(dt)

    def params():
        return (rv0 + (rv - rv0) * fm, c0 * 1000 + (c - c0 * 1000) * fm, a0 + (a - a0) * free.to(dt),
                b0 + (b - b0) * free.to(dt), cr0 + (cr - cr0) * fm)

    def terms():
        RV, C, A, B, CR = params()
        R = MV.rodrigues(RV)
        y = torch.einsum('nij,nj->ni', R[Fi], X - C[Fi])
        q = torch.stack([f * y[:, 0] / y[:, 2] + W / 2, f * y[:, 1] / y[:, 2] + H / 2], 1)
        corr = (CR[Fi].gather(1, nodes) * wts).sum(1)
        zm = torch.exp(corr) * 1000 / (A[Fi] * d + B[Fi]).clamp_min(1.0)
        r_p = q - P
        r_d = wd * (y[:, 2] - zm) / zm
        G = CR.reshape(n, gy, gx)
        r_c = (4.0 * ((CR - ci) ** 2).sum(1) + 4.0 * (torch.diff(G, dim=1) ** 2).sum((1, 2))
               + 4.0 * (torch.diff(G, dim=2) ** 2).sum((1, 2))) * cnt / (gy * gx)
        acc_c = ((C[2:] - 2 * C[1:-1] + C[:-2]) ** 2).sum(1) / (sc ** 2)
        acc_r = ((RV[2:] - 2 * RV[1:-1] + RV[:-2]) ** 2).sum(1) / (sr ** 2)
        return r_p, r_d, r_c, acc_c, acc_r

    def loss():
        r_p, r_d, r_c, acc_c, acc_r = terms()
        z = torch.zeros_like
        return (hub(r_p, z(r_p), delta=2.0, reduction='sum') + hub(r_d, z(r_d), delta=2.0, reduction='sum')
                + r_c[free].sum() + nmed * (acc_c.sum() + acc_r.sum()))
    opt = torch.optim.LBFGS([rv, c, a, b, cr], lr=1.0, max_iter=CFG['smooth_iters'], history_size=50,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        L = loss()
        L.backward()
        return L
    L0 = float(loss())
    opt.step(closure)
    with torch.no_grad():
        RV, C, A, B, CR = params()
        r_p, r_d, *_ = terms()
        e = r_p.norm(dim=1).numpy()
        Fn = Fi.numpy()
        err = np.zeros(n)
        for k in fr:
            err[k] = float(np.median(e[Fn == k]))
    out = (RV.numpy(), C.numpy() / 1000, A.numpy(), B.numpy(), CR.numpy().reshape(n, gy, gx), err)
    return out + (dict(smooth_loss=[round(L0, 1), round(float(loss()), 1)], reproj_px_median=round(float(np.median(e)), 3),
                       depth_rel_median=round(float(np.median(np.abs(r_d.numpy()) / wd)), 4)),)


def write_frames(clip_v, sol, ctx, fit_err):
    """frames.npz in the r2s.multiview format (world cameras of the base clip's world frame)."""
    cam = ctx.cam
    n = ctx.n
    R = np.array([sol.R[k] @ cam.R for k in range(n)])
    centres = np.array([-sol.R[k].T @ sol.t[k] for k in range(n)])
    pos = cam.pos + centres @ cam.R
    params = np.c_[np.array([Rot.from_matrix(sol.R[k]).as_rotvec() for k in range(n)]), np.zeros(n)]
    out = clip_v.prep / 'multiview'
    out.mkdir(parents=True, exist_ok=True)
    corr = fine_corr(sol.corr, ctx.H, ctx.W) if sol.corr.shape[1] < 20 else sol.corr
    np.savez(out / 'frames.npz', R=R, f=np.full(n, sol.f), pos=pos, params=params, fit_err_px=fit_err,
             a=sol.a, b=sol.b, corr=corr.astype(np.float32))
    return out / 'frames.npz'


# ------------------------------------------------------------------ the solve
def solve(clip, tag=VARIANT, holdout_far=None, rounds=None, log=log):
    t_start = time.time()
    ctx = Ctx(clip)
    clip_v = RC.Clip(f'{r2s_name(clip)}+{tag}')
    out = clip_v.prep / 'multiview'
    out.mkdir(parents=True, exist_ok=True)
    n, H, W = ctx.n, ctx.H, ctx.W
    keys = list(range(0, n, CFG['every']))
    f_ref = reference_focal(clip)
    f0 = f_ref if f_ref else float(np.median(ctx.cams['f']))
    static = ctx.static()
    log(f'[geomfix] {clip} -> {tag}: movers {ctx.movers}; static {static.mean():.3f} of the image (r2s static '
        f'{MV.static_masks(ctx.base, ctx.masks).mean():.3f}); {len(keys)} keyframes; focal {f0:.1f} px '
        f'{"fixed (from " + CFG["focal_from"] + ")" if f_ref else "fitted"}')
    rulers = rulers_at(ctx, keys)
    # initial values: the rotation + zoom track and the single-view calibration rescaled to the focal length
    m = RC.load_json(ctx.base.prep / 'metric.json')
    bsv = np.load(ctx.base.prep / 'metric_b.npy')
    fr = ctx.cams['f'][keys] / f0
    R0rel = np.array([Rot.from_matrix(ctx.cams['R'][k] @ ctx.cams['R'][0].T).as_rotvec() for k in keys])
    a0, b0, t0 = m['a'] * fr, bsv[keys] * fr, None
    rep = dict(clip=clip, tag=tag, cfg=CFG, movers=ctx.movers, keyframes=keys, focal_px=f0, focal_fixed=bool(f_ref),
               holdout_far_frames=holdout_far, rounds=[])
    feats = sift_features(ctx, static, keys)
    pairs = [(i, j) for i in range(len(keys)) for j in range(i + 2, len(keys))]
    if holdout_far:                  # wide matches this far apart are kept out of every round and saved as a probe
        far_pairs = [(i, j) for i, j in pairs if keys[j] - keys[i] >= holdout_far]
        pairs = [(i, j) for i, j in pairs if keys[j] - keys[i] < holdout_far]
        hm = wide_matches(feats, far_pairs)
        np.savez(out / 'heldout_wide.npz', frames=np.array([(keys[i], keys[j]) for i, j, A, B in hm]).reshape(-1, 2),
                 pair=np.concatenate([np.full(len(A), p) for p, (i, j, A, B) in enumerate(hm)]),
                 A=np.concatenate([A for *_, A, B in hm]), B=np.concatenate([B for *_, A, B in hm]))
        log(f'[geomfix] held out: {sum(len(A) for *_, A, B in hm)} ratio-test SIFT matches in {len(hm)} pairs >= {holdout_far} frames apart')
    sol = None
    for rnd in range(rounds or CFG['rounds']):
        t0r = time.time()
        if rnd > 0:
            static, mfrac = motion_reject(ctx, sol_frames, ctx.static())
            feats = sift_features(ctx, static, keys)
            log(f'[geomfix] round {rnd}: moving pixels rejected from static: median {np.median(mfrac):.3f}, '
                f'p90 {np.percentile(mfrac, 90):.3f} per frame')
        I, J, Pi, Pj = MV.chain_tracks(ctx.frames, static, keys, window=CFG['window'], n_pts=CFG['n_pts'])
        n_flow = len(I)
        ms = wide_matches(feats, pairs)
        n_ratio = sum(len(A) for *_, A, B in ms)
        if sol is not None:                               # guided matches + check against the previous solution
            gm = guided_matches(feats, sol, keys, static, pairs)
            ms = ms + gm
            kept = []
            for i, j, A, B in ms:                         # outliers of the pair under the previous solution
                q, z = sol.transfer(keys[i], keys[j], A)
                e = np.linalg.norm(q - B, axis=1)
                ok = (z > 0) & (e < max(CFG['wide_check_px'], 3 * float(np.median(e))))
                if ok.sum() >= CFG['min_inliers'] // 2:
                    kept.append((i, j, A[ok], B[ok]))
            ms = kept
        WI, WJ, WPi, WPj = stack_matches(ms)
        WI, WJ = WI.astype(int), WJ.astype(int)
        gaps_f = np.abs(np.array(keys)[WJ] - np.array(keys)[WI]) if len(WI) else np.zeros(0)
        log(f'[geomfix] round {rnd}: {n_flow} flow correspondences, {len(WI)} wide ({n_ratio} ratio-test before checks; '
            f'{int((gaps_f >= 50).sum())} >= 50 frames apart, {int((gaps_f >= 100).sum())} >= 100), '
            f'{len(set(zip(WI.tolist(), WJ.tolist())))} pairs, {len(rulers)} ruler samples')
        I2, J2 = np.r_[I, WI].astype(int), np.r_[J, WJ].astype(int)
        Pi2, Pj2 = np.r_[Pi, WPi], np.r_[Pj, WPj]
        is_wide = np.r_[np.zeros(n_flow, bool), np.ones(len(WI), bool)]
        far = np.abs(np.array(keys)[J2] - np.array(keys)[I2]) >= 100
        if sol is not None:                               # warm start from the previous round
            R0rel = np.array([Rot.from_matrix(R).as_rotvec() for R in sol.R])
            a0, b0, t0 = sol.a, sol.b, sol.t
        ba = BA2(W, H, keys, ctx.disp[keys], I2, J2, Pi2, Pj2, rulers[:, :4], f0, R0rel, a0, b0, grid=CFG['grid'],
                 rpx=rulers[:, 4:6], t0=t0, fix_f=bool(f_ref), w_ruler=CFG['w_ruler'])
        if sol is not None:
            with torch.no_grad():
                ba.corr.copy_(torch.tensor(sol.corr))
        before = ba_stats(ba, is_wide, far)
        ba.fit(iters=CFG['iters'], adam_iters=CFG['adam_iters'])
        after = ba_stats(ba, is_wide, far)
        sol = ba.solution(keys, ctx.disp, W, H)
        s, ns = gauge(sol, ctx)
        sol = sol.scaled(s)
        log(f'[geomfix] round {rnd}: keyframe BA {before} -> {after}; gauge x{s:.3f} ({ns} shaft samples) '
            f'({time.time() - t0r:.0f} s)')
        sol_frames, pf = per_frame(ctx, sol, static, CFG['grid'])
        s2, ns2 = gauge(sol_frames, ctx, frames=range(0, n, 2))
        sol, sol_frames = sol.scaled(s2), sol_frames.scaled(s2)
        centres = np.array([-sol.R[i].T @ sol.t[i] for i in range(len(keys))])
        rr = dict(round=rnd, n_flow=n_flow, n_wide=int(len(WI)), n_wide_ratio_test=n_ratio,
                  n_wide_50plus_frames=int((gaps_f >= 50).sum()), n_wide_100plus_frames=int((gaps_f >= 100).sum()),
                  n_wide_pairs=int(len(set(zip(WI.tolist(), WJ.tolist())))), n_rulers=int(len(rulers)),
                  before=before, after=after, gauge_keyframes=round(s, 4), gauge_all_frames=round(s2, 4),
                  per_frame={k: v for k, v in pf.items() if k != 'fit_err_px'},
                  max_translation_mm=round(float(np.linalg.norm(centres, axis=1).max()) * 1000, 1),
                  a_range=[round(float(sol.a.min()), 3), round(float(sol.a.max()), 3)], seconds=round(time.time() - t0r))
        rep['rounds'].append(rr)
        log(f'[geomfix] round {rnd}: per-frame {rr["per_frame"]}; final gauge x{s2:.3f}')
        np.savez(out / f'keyframes_r{rnd}.npz', keys=np.array(keys), R=sol.R, t=sol.t, f=sol.f, a=sol.a, b=sol.b,
                 corr=sol.corr, I=I2, J=J2, Pi=Pi2, Pj=Pj2, is_wide=is_wide)
    np.savez(out / 'keyframes.npz', keys=np.array(keys), R=sol.R, t=sol.t, f=sol.f, a=sol.a, b=sol.b, corr=sol.corr,
             I=I2, J=J2, Pi=Pi2, Pj=Pj2, is_wide=is_wide)
    write_frames(clip_v, sol_frames, ctx, pf['fit_err_px'])
    np.savez_compressed(out / 'static.npz', static=np.packbits(static, axis=-1), shape=np.array(static.shape))
    rep['seconds'] = round(time.time() - t_start)
    RC.save_json(out / 'report.json', rep)
    log(f'[geomfix] {clip} {tag}: done in {rep["seconds"]} s -> {out}')
    return rep


# ------------------------------------------------------------------ evaluation
def load_geometry(clip, tag, ctx):
    g = RC.Clip(f'{r2s_name(clip)}+{tag}')
    return sol_from_frames(g.prep / 'multiview' / 'frames.npz', ctx.disp, ctx.W, ctx.H, ctx.cam)


def probe_matches(ctx, static, keys, min_inliers=20):
    """Held-out ORB matches (as r2s.views_eval.probe_matches, plus a mutual-nearest-neighbour check: without it
    many-to-one matches on repeated texture pass the F test)."""
    ak = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
    feats = []
    for k in keys:
        kp, de = ak.detectAndCompute(ctx.gray[k], (static[k] & ~ctx.specular(k)).astype(np.uint8) * 255)
        feats.append((np.array([q.pt for q in kp]).reshape(-1, 2), de))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (pi, di), (pj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(di) < min_inliers or len(dj) < min_inliers:
                continue
            back = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(dj, di, k=1) if m}
            good = [m[0] for m in bf.knnMatch(di, dj, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance
                    and back.get(m[0].trainIdx) == m[0].queryIdx]           # ratio test + mutual nearest neighbour
            if len(good) < min_inliers:
                continue
            A, B = pi[[g.queryIdx for g in good]], pj[[g.trainIdx for g in good]]
            inl = f_verified(A, B, 1.0)
            if inl.sum() >= min_inliers:
                out[(i, j)] = (A[inl], B[inl])
    return out


def probe_error(sol, probes, keys):
    err = {}
    for (i, j), (A, B) in probes.items():
        q, z = sol.transfer(keys[i], keys[j], A)
        err[(i, j)] = np.linalg.norm(q - B, axis=1)
    pick = lambda c: np.concatenate([e for (i, j), e in err.items() if c(i, j)] or [np.zeros(0)])
    med = lambda v: round(float(np.median(v)), 2) if len(v) else None
    return dict(gap_1_2=med(pick(lambda i, j: j - i < 3)), gap_3_9=med(pick(lambda i, j: 3 <= j - i < 10)),
                gap_10plus=med(pick(lambda i, j: j - i >= 10)), with_frame0=med(pick(lambda i, j: i == 0)),
                n_1_2=int(len(pick(lambda i, j: j - i < 3))), n_3_9=int(len(pick(lambda i, j: 3 <= j - i < 10))),
                n_10plus=int(len(pick(lambda i, j: j - i >= 10))),
                p90_10plus=round(float(np.percentile(pick(lambda i, j: j - i >= 10), 90)), 2) if len(pick(lambda i, j: j - i >= 10)) else None)


def shaft_sections(m, H, W, keep=(0.05, 0.70)):
    """perception.shaft_samples with the cross-section's normal and width: (u, v, width px, nx, ny) per sample."""
    ys, xs = np.nonzero(m)
    if len(xs) < 300:
        return []
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    d = np.linalg.svd(P - c)[2][0]
    s = (P - c) @ d
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    border = lambda q: min(q[0], W - 1 - q[0], q[1], H - 1 - q[1])
    if border(c + d * hi) < border(c + d * lo):
        d, s, lo, hi = -d, -s, -hi, -lo
    nrm = np.array([-d[1], d[0]])
    out = []
    for t in np.linspace(lo + keep[0] * (hi - lo), lo + keep[1] * (hi - lo), 24):
        sel = np.abs(s - t) < 1.5
        if sel.sum() < 4:
            continue
        o = (P[sel] - c) @ nrm
        w = np.ptp(o) + 1
        q = c + d * t + nrm * (o.max() + o.min()) / 2               # centre of the cross-section
        if 5 < w < 120 and 0 <= q[0] < W and 0 <= q[1] < H:
            out.append((q[0], q[1], w, nrm[0], nrm[1]))
    return out


def ruler_and_free_space(ctx, sol, frames, off=16, tol=0.002, D=0.005):
    rel, s_free, behind, who = [], [], [], []
    ins_all = np.any(ctx.masks[:, [ctx.names.index(i) for i in ctx.instruments]], 1)
    for k in frames:
        Z = sol.depth(k)
        for ins in ctx.instruments:
            mk = ctx.masks[k, ctx.names.index(ins)]
            for u, v, w, nx, ny in shaft_sections(mk, ctx.H, ctx.W):
                zs = sol.f * D / w
                rel.append((Z[int(v), int(u)] - zs) / zs)
                front = zs - D / 2
                for sg in (-1, 1):
                    px = int(round(u + sg * nx * (w / 2 + off)))
                    py = int(round(v + sg * ny * (w / 2 + off)))
                    if 0 <= px < ctx.W and 0 <= py < ctx.H and ctx.valid[py, px] and not ins_all[k, py, px]:
                        zt = Z[py, px]
                        s_free.append((zt + tol) / front)
                        behind.append(zt + tol < front)
                        who.append(ins)
    rel, s_free, behind, who = np.array(rel), np.array(s_free), np.array(behind), np.array(who)
    per = {i: dict(frac_behind=round(float(np.mean(behind[who == i])), 4), n=int((who == i).sum()),
                   s_quantiles_5_10_25_50=[round(float(x), 3) for x in np.quantile(s_free[who == i], [0.05, 0.1, 0.25, 0.5])])
           for i in ctx.instruments if (who == i).any()}
    return (dict(signed_median=round(float(np.median(rel)), 4), abs_median=round(float(np.median(np.abs(rel))), 4),
                 p90_abs=round(float(np.percentile(np.abs(rel), 90)), 4), n=int(len(rel))),
            dict(frac_behind=round(float(np.mean(behind)), 4), n=int(len(behind)),
                 kappa_10pct=round(float(min(np.quantile(s_free, 0.10), 1.0)), 4),
                 s_quantiles_5_10_25_50=[round(float(x), 3) for x in np.quantile(s_free, [0.05, 0.1, 0.25, 0.5])],
                 per_instrument=per))


def static_scale(ctx, sol, static, step_px=8, src_every=3):
    """Per frame k: median over static points of the other frames of (their depth in k) / (k's depth there)."""
    H, W = ctx.H, ctx.W
    vv, uu = np.mgrid[step_px // 2:H:step_px, step_px // 2:W:step_px]
    pts = {}
    for j in range(0, ctx.n, src_every):
        m = static[j][vv, uu]
        P = np.stack([uu[m], vv[m]], 1).astype(float)
        z = sol.depth(j)[vv[m], uu[m]]
        i = sol.pos[j]
        x = np.stack([(P[:, 0] - W / 2) * z / sol.f, (P[:, 1] - H / 2) * z / sol.f, z], 1)
        pts[j] = (x - sol.t[i]) @ sol.R[i]
    s = np.full(ctx.n, np.nan)
    for k in range(ctx.n):
        X = np.concatenate([p for j, p in pts.items() if abs(j - k) >= 3])
        i = sol.pos[k]
        y = X @ sol.R[i].T + sol.t[i]
        ok = y[:, 2] > 1e-3
        q = np.stack([sol.f * y[ok, 0] / y[ok, 2] + W / 2, sol.f * y[ok, 1] / y[ok, 2] + H / 2], 1)
        zz = y[ok, 2]
        inn = (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1)
        q, zz = q[inn], zz[inn]
        qi = q.astype(int)
        on = static[k][qi[:, 1], qi[:, 0]]
        if on.sum() < 200:
            continue
        Zk = sol.depth(k)
        s[k] = np.median(zz[on] / bsample(Zk, q[on]))
    ok = np.isfinite(s)
    v = s[ok]
    return dict(p5=round(float(np.percentile(v, 5)), 4), p95=round(float(np.percentile(v, 95)), 4),
                min=round(float(v.min()), 4), max=round(float(v.max()), 4), median=round(float(np.median(v)), 4),
                mad=round(float(np.median(np.abs(v - np.median(v)))), 4), frames=int(ok.sum())), s


def local_ncc(a, b, m, win=15):
    """Mean over masked pixels of the windowed NCC of images a, b (only masked pixels enter the windows)."""
    mf = m.astype(np.float32)
    box = lambda x: cv2.boxFilter(x, -1, (win, win), normalize=False)
    n = box(mf)
    sa, sb = box(a * mf), box(b * mf)
    saa, sbb, sab = box(a * a * mf), box(b * b * mf), box(a * b * mf)
    ok = m & (n > win * win * 0.5)
    n_ = np.maximum(n, 1)
    va = saa - sa * sa / n_
    vb = sbb - sb * sb / n_
    cab = sab - sa * sb / n_
    r = cab / np.sqrt(np.maximum(va * vb, 1e-6))
    good = ok & (va > n_ * 4) & (vb > n_ * 4)                 # windows with some texture (std > 2 grey levels)
    return float(np.mean(r[good])) if good.any() else np.nan


def photometric(ctx, sol, static, gap=10, every=2):
    H, W = ctx.H, ctx.W
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([uu.ravel(), vv.ravel()], 1).astype(float)
    nccs, lnccs = [], []
    for k in range(0, ctx.n - gap, every):
        j = k + gap
        # backward warp: every pixel of j looks up frame k through j's depth -> k
        q, z = sol.transfer(j, k, P)
        ok = (z > 0) & (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1) & static[j].ravel()
        qi = np.clip(q.round().astype(int), 0, [W - 1, H - 1])
        ok &= static[k][qi[:, 1], qi[:, 0]]
        if ok.sum() < 0.03 * H * W:
            continue
        warped = cv2.remap(ctx.gray[k].astype(np.float32), q[:, 0].reshape(H, W).astype(np.float32),
                           q[:, 1].reshape(H, W).astype(np.float32), cv2.INTER_LINEAR)
        tgt = ctx.gray[j].astype(np.float32)
        m = ok.reshape(H, W)
        a, b = warped[m] - warped[m].mean(), tgt[m] - tgt[m].mean()
        nccs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        lnccs.append(local_ncc(warped, tgt, m))
    return dict(ncc_median=round(float(np.median(nccs)), 4), local_ncc_median=round(float(np.nanmedian(lnccs)), 4),
                pairs=len(nccs), gap_frames=gap)


def evaluate(clip, tags, log=log):
    ctx = Ctx(clip)
    st_eval = ctx.static()                                # common static mask: labels only
    keys = list(range(0, ctx.n, 10))
    probes = probe_matches(ctx, st_eval, keys)
    log(f'[eval] {clip}: {len(probes)} ORB probe pairs, {sum(len(A) for A, B in probes.values())} matches, '
        f'{sum(len(A) for (i, j), (A, B) in probes.items() if j - i >= 10)} at >= 10 keyframes')
    res = dict(clip=clip, movers=ctx.movers, static_fraction=round(float(st_eval.mean()), 4), probe_pairs=len(probes),
               geometries={})
    hp = RC.OUTPUTS / 'variants' / f'{r2s_name(clip)}+sift2_holdout' / 'prep' / 'multiview' / 'heldout_wide.npz'
    heldout = None
    if hp.exists():
        z = np.load(hp)
        heldout = [(fi, fj, z['A'][z['pair'] == p], z['B'][z['pair'] == p]) for p, (fi, fj) in enumerate(z['frames'])]
    scales = {}
    for tag in tags:
        p = RC.OUTPUTS / 'variants' / f'{r2s_name(clip)}+{tag}' / 'prep' / 'multiview'
        if not (p / 'frames.npz').exists():
            continue
        sol = load_geometry(clip, tag, ctx)
        t0 = time.time()
        r = dict(focal_px=round(sol.f, 1))
        rep = json.loads((p / 'report.json').read_text())
        if 'rounds' in rep:
            last = rep['rounds'][-1]
            r['in_sample'] = dict(last['after'], n_wide=last['n_wide'], n_wide_100plus_frames=last['n_wide_100plus_frames'],
                                  per_frame=last['per_frame'])
        else:
            r['in_sample'] = dict(rep['after'], n_wide=rep.get('n_wide'), n_wide_100plus_frames=rep.get('n_wide_10plus_keyframes'))
        r['probe'] = probe_error(sol, probes, keys)
        r['ruler'], r['freespace'] = ruler_and_free_space(ctx, sol, range(ctx.n))
        r['scale'], scales[tag] = static_scale(ctx, sol, st_eval)
        r['photo'] = photometric(ctx, sol, st_eval)
        r['photo_30'] = photometric(ctx, sol, st_eval, gap=30, every=4)
        Zs = [np.median(sol.depth(k)[st_eval[k]]) for k in range(0, ctx.n, 5)]
        r['static_depth_mm_median'] = round(float(np.median(Zs)) * 1000, 1)
        centres = np.array([-sol.R[k].T @ sol.t[k] for k in range(ctx.n)])
        r['jitter'] = jitter(np.array([Rot.from_matrix(R).as_rotvec() for R in sol.R]), centres)
        if heldout is not None:
            e = []
            for fi, fj, A, B in heldout:
                q, z = sol.transfer(int(fi), int(fj), A)
                e.append(np.linalg.norm(q - B, axis=1))
            e = np.concatenate(e)
            r['heldout_sift'] = dict(median_px=round(float(np.median(e)), 2), p75_px=round(float(np.percentile(e, 75)), 2),
                                     p90_px=round(float(np.percentile(e, 90)), 2), n=int(len(e)), pairs=len(heldout),
                                     source=str(hp))
        r['scope_path_mm'] = round(float(np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1))) * 1000, 1)
        r['scope_extent_mm'] = round(float(np.linalg.norm(centres - centres[0], axis=1).max()) * 1000, 1)
        res['geometries'][tag] = r
        log(f'[eval] {clip} {tag} ({time.time() - t0:.0f} s): ' + json.dumps({k: v for k, v in r.items() if k != 'in_sample'}))
    return res, scales


# ------------------------------------------------------------------ figures
def warped_image(ctx, sol, k, j, static):
    """Frame k rendered into frame j (backward warp through j's depth), where both see static anatomy."""
    H, W = ctx.H, ctx.W
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([uu.ravel(), vv.ravel()], 1).astype(float)
    q, z = sol.transfer(j, k, P)
    ok = (z > 0) & (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1) & static[j].ravel()
    qi = np.clip(q.round().astype(int), 0, [W - 1, H - 1])
    ok &= static[k][qi[:, 1], qi[:, 0]]
    img = cv2.remap(ctx.frames[k], q[:, 0].reshape(H, W).astype(np.float32), q[:, 1].reshape(H, W).astype(np.float32),
                    cv2.INTER_LINEAR)
    return img, ok.reshape(H, W)


def checker_figure(ctx, sols, pairs, path, block=40):
    """Rows: frame pairs (k -> j); columns: real frame j, then each geometry's warp of k checkerboarded with j
    (edges continue across the squares where the geometry is right; non-static parts dimmed)."""
    H, W = ctx.H, ctx.W
    yy, xx = np.mgrid[0:H, 0:W]
    chk = (yy // block + xx // block) % 2 == 0
    st = ctx.static()
    rows = []
    for k, j in pairs:
        cells = []
        real = ctx.frames[j].copy()
        lab = lambda im, t: (cv2.rectangle(im, (0, 0), (W, 24), (0, 0, 0), -1),
                             cv2.putText(im, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA))
        r0 = real.copy()
        lab(r0, f'real frame {j}')
        cells.append(r0)
        for tag, sol in sols.items():
            img, cov = warped_image(ctx, sol, k, j, st)
            out = np.where((chk & cov)[..., None], img, real)
            out = np.where(cov[..., None], out, (real * 0.35).astype(np.uint8)).copy()
            lab(out, f'{tag}: frame {k} warped into {j}')
            cells.append(out)
        rows.append(np.concatenate(cells, 1))
    grid = np.concatenate(rows, 0)
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def series_figure(series, path, w=1100, h=180, title=''):
    """Line plot (cv2) of per-frame series {label: (values, colour)} with a 1.0 reference line."""
    vals = np.concatenate([v[np.isfinite(v)] for v, c in series.values()])
    lo, hi = min(0.6, float(np.nanmin(vals))), max(1.4, float(np.nanmax(vals)))
    img = np.full((h + 40, w + 60, 3), 255, np.uint8)
    X = lambda i, n: int(50 + i * w / max(n - 1, 1))
    Y = lambda v: int(20 + (hi - v) / (hi - lo) * h)
    for v in (0.8, 1.0, 1.2):
        cv2.line(img, (50, Y(v)), (50 + w, Y(v)), (200, 200, 200) if v != 1.0 else (120, 120, 120), 1)
        cv2.putText(img, f'{v:.1f}', (8, Y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    for li, (lab, (v, col)) in enumerate(series.items()):
        n = len(v)
        pts = [(X(i, n), Y(x)) for i, x in enumerate(v) if np.isfinite(x)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, col, 2, cv2.LINE_AA)
        cv2.putText(img, lab, (60 + 260 * li, h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(img, title, (60, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main(argv):
    cmd, clip = argv[0], argv[1]
    if cmd == 'solve':
        tag = argv[argv.index('--tag') + 1] if '--tag' in argv else VARIANT
        hf = int(argv[argv.index('--holdout-far') + 1]) if '--holdout-far' in argv else None
        solve(clip, tag, holdout_far=hf)
    elif cmd == 'eval':
        tags = [a for a in argv[2:] if not a.startswith('--')] or ['sift', VARIANT]
        res, scales = evaluate(clip, tags)
        out = Path(argv[argv.index('--outdir') + 1]) if '--outdir' in argv else D.OUT / clip / 'geometry'
        out.mkdir(parents=True, exist_ok=True)
        name = argv[argv.index('--out') + 1] if '--out' in argv else 'quality.json'
        RC.save_json(out / name, res)
        np.savez(out / name.replace('.json', '_scales.npz'), **scales)
    elif cmd == 'figures':
        tags = [a for a in argv[2:] if not a.startswith('--')] or ['sift', VARIANT]
        out = Path(argv[argv.index('--outdir') + 1]) if '--outdir' in argv else D.OUT / clip / 'geometry'
        ctx = Ctx(clip)
        sols = {t: load_geometry(clip, t, ctx) for t in tags}
        n = ctx.n
        checker_figure(ctx, sols, [(40, 50), (20, 60), (0, 100), (100, 200), (140, 240)], out / 'warp_checker.jpg')
        cols = [(200, 60, 40), (40, 120, 220), (40, 160, 60), (150, 60, 170)]
        sc = {}
        for f in sorted(out.glob('quality*_scales.npz')):
            z = np.load(f)
            for t in z.files:
                if t in tags:
                    sc[t] = z[t]
        if sc:
            series_figure({t: (v, cols[i % 4]) for i, (t, v) in enumerate(sc.items())}, out / 'static_scale.png',
                          title='per-frame depth scale of the static surface vs the other frames (1 = consistent)')


if __name__ == '__main__':
    main(sys.argv[1:])
