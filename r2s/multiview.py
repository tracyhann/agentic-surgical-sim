"""Multi-view joint estimation: bundle adjustment over keyframes of a moving scope.

The scope moves (1-2 cm in the gallbladder shots, at 5-9 cm from the tissue), so the clip holds several real views
of the same static anatomy. Over K keyframes we solve jointly for
  - each keyframe's camera pose (6 DoF; keyframe 0 fixed),
  - one focal length for the clip (the Commons video has no calibration; 47 deg was a guess),
  - each keyframe's depth: inverse depth affine in the Depth Anything output, 1/z = a_k d + b_k,
from correspondences on static anatomy (dense flow chained across frames, forward-backward checked):
  reprojection   a point seen in keyframe i at its modelled depth must land on its track in keyframe j (px, Huber)
  depth          its depth in keyframe j must agree with keyframe j's modelled depth there (relative)
  rulers         instrument shafts of known diameter fix the absolute scale (relative depth error)
Non-keyframes then get their pose by PnP against the nearest keyframe's 3D points and their own depth affine.

Wide-baseline correspondences (MODES) tie keyframes far apart: SIFT matches or VGGT point tracks (vggt_views.py),
and VGGT can provide the initial values.

  python -m r2s.multiview <clip> [mode]   -> outputs/<clip>/prep/multiview/ (keyframe solution, report)
"""
import sys
import time
import numpy as np
import cv2
import torch
from .config import Clip, save_json, load_json
from .camera import fill_smooth
from .perception import shaft_samples
from .scene import dil
from scipy.spatial.transform import Rotation as Rot


def static_masks(clip, masks):
    moving = [clip.obj_index(i['mask']) for i in clip.instruments] + [clip.obj_index(s) for s in clip.role('strand')] + \
             [clip.obj_index(clip.organ)]
    return np.stack([~dil(np.any(masks[k, moving], 0), 15) for k in range(len(masks))])


def chain_tracks(frames, static, keys, window=3, n_pts=500, fb_tol=1.0):
    """Corners of each keyframe followed frame by frame (dense flow, forward-backward checked, static pixels only) up
    to `window` keyframes ahead. Returns observations (i, j, p_i, p_j) between keyframe indices i < j."""
    n = len(frames)
    H, W = frames[0].shape[:2]
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    key_index = {k: i for i, k in enumerate(keys)}
    active = []                                   # [start key index, start px (N,2), current px (N,2), alive (N,)]
    obs = []
    for t in range(n):
        if t in key_index:
            i = key_index[t]
            p = cv2.goodFeaturesToTrack(g[t], n_pts, 0.005, 9, mask=static[t].astype(np.uint8) * 255)
            if p is not None:
                active.append([i, p[:, 0].astype(np.float64), p[:, 0].astype(np.float64), np.ones(len(p), bool)])
            for tr in active:                     # record every live track at this keyframe
                if tr[0] < i:
                    ok = tr[3]
                    for a, b in zip(tr[1][ok], tr[2][ok]):
                        obs.append((tr[0], i, a, b))
            active = [tr for tr in active if i - tr[0] < window and tr[3].any()]
        if t == n - 1 or not active:
            continue
        fw = dis.calc(g[t], g[t + 1], None)
        bw = dis.calc(g[t + 1], g[t], None)
        for tr in active:
            p = tr[2]
            xi = np.clip(p[:, 0], 0, W - 1).astype(int)
            yi = np.clip(p[:, 1], 0, H - 1).astype(int)
            q = p + fw[yi, xi]
            inside = (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1)
            qx = np.clip(q[:, 0], 0, W - 1).astype(int)
            qy = np.clip(q[:, 1], 0, H - 1).astype(int)
            back = q + bw[qy, qx]
            tr[3] &= inside & (np.linalg.norm(back - p, axis=1) < fb_tol) & static[t + 1][qy, qx]
            tr[2] = q
    I = np.array([o[0] for o in obs])
    J = np.array([o[1] for o in obs])
    Pi = np.array([o[2] for o in obs])
    Pj = np.array([o[3] for o in obs])
    return I, J, Pi, Pj


def sample(img, p):
    """Bilinear sample of a (H, W) image at float pixel positions."""
    m = cv2.remap(img.astype(np.float32), p[:, 0].astype(np.float32).reshape(-1, 1), p[:, 1].astype(np.float32).reshape(-1, 1),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return m[:, 0]


def rodrigues(r):
    th = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-9)
    k = r / th
    K = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, dtype=r.dtype) + s * K + (1 - c) * (K @ K)


class KeyframeBA:
    def __init__(self, W, H, keys, disp_keys, I, J, Pi, Pj, rulers, f0, R0rel, a0, b0, grid=(4, 6), rpx=None, t0=None):
        dt = torch.float64
        self.W, self.H, self.K = W, H, len(keys)
        self.c = torch.tensor([W / 2, H / 2], dtype=dt)
        self.I, self.J = torch.tensor(I), torch.tensor(J)
        self.Pi, self.Pj = torch.tensor(Pi, dtype=dt), torch.tensor(Pj, dtype=dt)
        di, dj = np.zeros(len(I)), np.zeros(len(J))
        for k in range(self.K):                        # each observation's Depth Anything value in its own keyframe
            si, sj = I == k, J == k
            if si.any():
                di[si] = sample(disp_keys[k], Pi[si])
            if sj.any():
                dj[sj] = sample(disp_keys[k], Pj[sj])
        self.di, self.dj = torch.tensor(di, dtype=dt), torch.tensor(dj, dtype=dt)
        self.rk = torch.tensor(rulers[:, 0].astype(int))          # rulers: (key, d, 1/width)
        self.rd = torch.tensor(rulers[:, 1], dtype=dt)
        self.rinvw = torch.tensor(rulers[:, 2], dtype=dt)
        self.shaft = torch.tensor(rulers[:, 3], dtype=dt)
        self.rpx = torch.tensor(rpx if rpx is not None else np.zeros((len(rulers), 2)), dtype=dt)
        self.rvec = torch.tensor(R0rel, dtype=dt, requires_grad=True)
        self.tvec = torch.tensor(np.zeros((self.K, 3)) if t0 is None else t0, dtype=dt, requires_grad=True)
        self.logf = torch.tensor(np.log(f0), dtype=dt, requires_grad=True)
        self.a = torch.tensor(a0, dtype=dt, requires_grad=True)
        self.b = torch.tensor(b0, dtype=dt, requires_grad=True)
        # smooth multiplicative depth correction per keyframe on a coarse grid (bilinear): the single-image depth has
        # shape errors that one scale and offset per frame cannot absorb
        self.gy, self.gx = grid
        self.corr = torch.zeros((self.K, self.gy, self.gx), dtype=dt, requires_grad=bool(grid[0]))

    def params(self):
        return [self.rvec, self.tvec, self.logf, self.a, self.b] + ([self.corr] if self.corr.requires_grad else [])

    def correction(self, k, p):
        """exp(bilinear grid value) at pixels p (N, 2) of keyframes k (N,)."""
        if not self.corr.requires_grad:
            return torch.ones(len(p), dtype=torch.float64)
        gx = (p[:, 0] / self.W) * (self.gx - 1)
        gy = (p[:, 1] / self.H) * (self.gy - 1)
        x0 = gx.floor().clamp(0, self.gx - 2).long()
        y0 = gy.floor().clamp(0, self.gy - 2).long()
        wx, wy = gx - x0, gy - y0
        c = self.corr
        v = (c[k, y0, x0] * (1 - wx) * (1 - wy) + c[k, y0, x0 + 1] * wx * (1 - wy) +
             c[k, y0 + 1, x0] * (1 - wx) * wy + c[k, y0 + 1, x0 + 1] * wx * wy)
        return torch.exp(v)

    def poses(self):
        mask = torch.ones((self.K, 1), dtype=torch.float64)
        mask[0] = 0                                   # keyframe 0 is the reference
        R = rodrigues(self.rvec * mask)
        return R, self.tvec * mask

    def residuals(self, w_depth=10.0, w_ruler=3.0):
        R, t = self.poses()
        f = torch.exp(self.logf)
        zi = self.correction(self.I, self.Pi) / (self.a[self.I] * self.di + self.b[self.I]).clamp_min(1.0)
        zj = self.correction(self.J, self.Pj) / (self.a[self.J] * self.dj + self.b[self.J]).clamp_min(1.0)
        xi = torch.stack([(self.Pi[:, 0] - self.c[0]) * zi / f, (self.Pi[:, 1] - self.c[1]) * zi / f, zi], 1)
        X = torch.einsum('nji,nj->ni', R[self.I], xi - t[self.I])        # camera-i -> keyframe-0 coordinates
        xj = torch.einsum('nij,nj->ni', R[self.J], X) + t[self.J]
        q = f * xj[:, :2] / xj[:, 2:3] + self.c
        r_proj = q - self.Pj
        r_depth = w_depth * (xj[:, 2] - zj) / zj
        zr = self.correction(self.rk, self.rpx) / (self.a[self.rk] * self.rd + self.b[self.rk]).clamp_min(1.0)
        zs = f * self.shaft * self.rinvw
        r_ruler = w_ruler * (zr - zs) / zs
        return r_proj, r_depth, r_ruler

    def loss(self, delta=2.0):
        r_proj, r_depth, r_ruler = self.residuals()
        hub = torch.nn.functional.huber_loss
        z = torch.zeros_like
        return (hub(r_proj, z(r_proj), delta=delta, reduction='sum') + hub(r_depth, z(r_depth), delta=delta, reduction='sum')
                + hub(r_ruler, z(r_ruler), delta=delta, reduction='sum') * (len(r_proj) / max(len(r_ruler), 1)) * 0.2
                + 50.0 * torch.sum((self.a[1:] - self.a[:-1]) ** 2) * len(r_proj) / self.K
                + self.corr_reg() * len(r_proj) / max(self.corr.numel(), 1))

    def corr_reg(self, smooth=4.0, size=1.0):
        if not self.corr.requires_grad:
            return torch.zeros((), dtype=torch.float64)
        c = self.corr
        lap = torch.diff(c, dim=1).pow(2).sum() + torch.diff(c, dim=2).pow(2).sum()
        temporal = torch.diff(c, dim=0).pow(2).sum()        # neighbouring keyframes see nearly the same scene
        return smooth * lap + size * c.pow(2).sum() + 2.0 * temporal

    def fit(self, iters=300, adam_iters=2000, log=print):
        # 1) poses and focal length (L-BFGS); 2) everything, Adam with per-parameter step sizes (the parameters live on
        # very different scales: rotations ~0.1 rad, translations ~5 mm, inverse-depth offsets ~5, corrections ~0.05)
        opt = torch.optim.LBFGS([self.rvec, self.tvec, self.logf], lr=0.5, max_iter=iters, history_size=50,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            L = self.loss()
            L.backward()
            return L
        opt.step(closure)
        groups = [dict(params=[self.rvec], lr=2e-4), dict(params=[self.tvec], lr=2e-5), dict(params=[self.logf], lr=2e-4),
                  dict(params=[self.a], lr=2e-3), dict(params=[self.b], lr=1e-2)]
        if self.corr.requires_grad:
            groups.append(dict(params=[self.corr], lr=2e-3))
        adam = torch.optim.Adam(groups)
        for it in range(adam_iters):
            adam.zero_grad()
            L = self.loss()
            L.backward()
            adam.step()
        with torch.no_grad():
            r_proj, r_depth, r_ruler = self.residuals()
        return dict(reproj_px_median=float(r_proj.norm(dim=1).median()), depth_rel_median=float((r_depth / 10).abs().median()),
                    ruler_rel_median=float((r_ruler / 3).abs().median()))


# multi-view modes: (initial values, source of wide-baseline correspondences); flow tracks are always used
MODES = {
    'ba': ('track', None),              # rotation + zoom track and single-view depth calibration as initial values
    'sift': ('track', 'sift'),          # + SIFT matches between all keyframe pairs, RANSAC-verified
    'vggt_tracks': ('track', 'vggt'),   # + VGGT point tracks, RANSAC-verified
    'vggt': ('vggt', 'vggt'),           # VGGT cameras, focal length and depth as initial values, + its tracks
}


def sift_matches(frames, static, keys, min_gap=2, min_inliers=25):
    """Wide-baseline correspondences between every pair of keyframes >= min_gap apart: SIFT on static anatomy
    (specular highlights masked), ratio test, fundamental matrix by MAGSAC at 1 px."""
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    feats = []
    for k in keys:
        hsv = cv2.cvtColor(frames[k], cv2.COLOR_RGB2HSV)
        spec = dil((hsv[..., 2] > 230) & (hsv[..., 1] < 60), 9)
        kp, de = sift.detectAndCompute(clahe.apply(cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY)),
                                       (static[k] & ~spec).astype(np.uint8) * 255)
        feats.append((np.array([q.pt for q in kp]).reshape(-1, 2), de))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    obs = [], [], [], []
    for i in range(len(keys)):
        for j in range(i + min_gap, len(keys)):
            (pi, di), (pj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(di) < min_inliers or len(dj) < min_inliers:
                continue
            good = [m[0] for m in bf.knnMatch(di, dj, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < min_inliers:
                continue
            A, B = pi[[g.queryIdx for g in good]], pj[[g.trainIdx for g in good]]
            inl = verified(A, B, 1.0)
            if inl.sum() >= min_inliers:
                for lst, v in zip(obs, (np.full(inl.sum(), i), np.full(inl.sum(), j), A[inl], B[inl])):
                    lst.append(v)
    return tuple(np.concatenate(v) if v else np.zeros((0, 2) if n > 1 else 0) for n, v in enumerate(obs))


def verified(A, B, thresh):
    """Inlier mask of correspondences under a fundamental matrix fitted by MAGSAC."""
    if len(A) < 8:
        return np.zeros(len(A), bool)
    _, inl = cv2.findFundamentalMat(A, B, cv2.USAC_MAGSAC, thresh, 0.999, 5000)
    return np.zeros(len(A), bool) if inl is None else inl.ravel().astype(bool)


def solve(name, mode='ba', every=10, window=3, grid=(4, 6), log=print):
    from . import source, perception
    clip = Clip(name)
    init, wide = MODES['ba' if mode is True else mode]
    out = clip.prep / 'multiview'
    out.mkdir(exist_ok=True)
    frames = source.frames(clip)
    masks = perception.load_masks(clip)
    disp = np.load(clip.prep / 'disp.npy').astype(np.float32)
    cams = dict(np.load(clip.prep / 'cams.npz'))
    cam = clip.camera()
    n, H, W = len(frames), cam.H, cam.W
    keys = list(range(0, n, every))
    static = static_masks(clip, masks)
    t0 = time.time()
    I, J, Pi, Pj = chain_tracks(frames, static, keys, window=window)
    n_flow = len(I)
    log(f'[multiview] {name} ({mode}): {len(keys)} keyframes, {n_flow} flow correspondences ({time.time() - t0:.0f} s)')
    # rulers at keyframes: instrument shafts of known diameter (1/width, so the focal length can change)
    rows = []
    for i, k in enumerate(keys):
        for ins in clip.instruments:
            if not ins.get('ruler', True):
                continue
            for u, v, invw in shaft_samples(masks[k, clip.obj_index(ins['mask'])], 1.0, 1.0, H, W):
                rows.append((i, float(disp[k, int(v), int(u)]), invw, ins['shaft_d'], u, v))
    rows = np.array(rows) if rows else np.zeros((0, 6))
    rulers, rpx = rows[:, :4], rows[:, 4:6]
    rep = dict(mode=mode, init=init, wide=wide, keyframes=keys)
    # wide-baseline correspondences
    if wide == 'sift':
        WI, WJ, WPi, WPj = sift_matches(frames, static, keys)
    elif wide == 'vggt':
        from . import vggt_views
        raw, WI, WJ, WPi, WPj = vggt_views.run(clip, frames, static, keys, log=log)
    if wide:
        I, J, Pi, Pj = np.r_[I, WI].astype(int), np.r_[J, WJ].astype(int), np.r_[Pi, WPi], np.r_[Pj, WPj]
        gaps = np.abs(WJ - WI)
        rep.update(n_wide=int(len(WI)), n_wide_10plus_keyframes=int((gaps >= 10).sum()),
                   n_wide_pairs=int(len(set(zip(WI.tolist(), WJ.tolist())))))
        log(f'[multiview] {len(WI)} {wide} correspondences, {(gaps >= 10).sum()} of them >= 10 keyframes apart')
    # initial values
    if init == 'vggt':
        from . import vggt_views
        g = vggt_views.metric(raw, np.c_[rows[:, 0], rows[:, 4], rows[:, 5], rows[:, 2], rows[:, 3]], W, H)
        ab = np.array([vggt_views.fit_affine(disp[k][static[k]], g['depth'][i][static[k]], g['depth_conf'][i][static[k]])
                       for i, k in enumerate(keys)])
        R0rel = np.array([Rot.from_matrix(R).as_rotvec() for R in g['R']])
        f0, a0, b0, t0 = g['f'], ab[:, 0], ab[:, 1], g['t']
        np.savez_compressed(out / 'vggt_metric.npz', keys=np.array(keys), R=g['R'], t=g['t'], f=g['f'], scale=g['scale'],
                            depth=g['depth'].astype(np.float32), depth_conf=g['depth_conf'].astype(np.float32))
        rep['vggt'] = dict(focal_px=round(g['f'], 1), fovy_deg=round(float(np.degrees(2 * np.arctan(H / 2 / g['f']))), 1),
                           metric_scale=round(g['scale'], 5), shaft_log_spread=round(g['ruler_spread'], 3),
                           max_translation_mm=round(float(np.linalg.norm(g['t'], axis=1).max()) * 1000, 1))
    else:
        m = load_json(clip.prep / 'metric.json')
        R0rel = np.array([Rot.from_matrix(cams['R'][k] @ cams['R'][0].T).as_rotvec() for k in keys])
        f0, a0, b0, t0 = float(cams['f'][0]), np.full(len(keys), m['a']), np.load(clip.prep / 'metric_b.npy')[keys], None
    ba = KeyframeBA(W, H, keys, disp[keys], I, J, Pi, Pj, rulers, f0, R0rel, a0, b0, grid=grid, rpx=rpx, t0=t0)
    is_wide = np.r_[np.zeros(n_flow, bool), np.ones(len(I) - n_flow, bool)]
    far = torch.tensor(np.abs(I - J) >= 10)

    def stats():
        with torch.no_grad():
            r_proj, r_depth, r_ruler = ba.residuals()
        e = r_proj.norm(dim=1)
        return dict(reproj_px_median=float(e[torch.tensor(~is_wide)].median()),
                    reproj_px_median_wide=float(e[torch.tensor(is_wide)].median()) if is_wide.any() else None,
                    reproj_px_median_10plus_keyframes=float(e[far].median()) if bool(far.any()) else None,
                    depth_rel_median=float((r_depth / 10).abs().median()),
                    ruler_rel_median=float((r_ruler / 3).abs().median()) if len(rulers) else None)
    rep['before'] = stats()
    ba.fit(log=log)
    rep['after'] = stats()
    R, t = ba.poses()
    R, t = R.detach().numpy(), t.detach().numpy()
    f = float(torch.exp(ba.logf.detach()))
    a, b = ba.a.detach().numpy(), ba.b.detach().numpy()
    centres = np.array([-R[i].T @ t[i] for i in range(len(keys))])            # in keyframe-0 camera coordinates
    rep.update(n_flow=n_flow, n_rulers=int(len(rulers)), focal_px=round(f, 1),
               fovy_deg=round(float(np.degrees(2 * np.arctan(H / 2 / f))), 1), fovy_assumed_deg=round(cam.fovy, 1),
               max_translation_mm=round(float(np.linalg.norm(centres, axis=1).max()) * 1000, 1),
               a_range=[round(float(a.min()), 3), round(float(a.max()), 3)])
    np.savez(out / 'keyframes.npz', keys=np.array(keys), R=R, t=t, f=f, a=a, b=b, corr=ba.corr.detach().numpy(), I=I, J=J, Pi=Pi, Pj=Pj)
    save_json(out / 'report.json', rep)
    log(f'[multiview] {name}: {rep}')
    return rep


if __name__ == '__main__':
    solve(sys.argv[1], mode=sys.argv[2] if len(sys.argv) > 2 else 'ba')


def keyframe_depth(kb, k_index, disp_k, W, H):
    """Metric depth map of one keyframe from the joint solution (affine + smooth correction)."""
    z = 1.0 / np.clip(kb['a'][k_index] * disp_k + kb['b'][k_index], 1.0, None)
    c = kb['corr'][k_index] if 'corr' in kb and kb['corr'].size else None
    if c is not None and c.shape[0] > 1:
        z = z * np.exp(cv2.resize(c.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR))
    return z


def alignment(clip, frames, static, gaps=(1, 2, 3), kb=None, cams=None, Z=None):
    """Mean gradient-free NCC of keyframe k warped into keyframe k+gap (static anatomy in both), for either the joint
    solution `kb` or a rotation + zoom track `cams` with per-frame depth Z."""
    cam = clip.camera()
    H, W = cam.H, cam.W
    vv, uu = np.mgrid[0:H, 0:W]
    disp = np.load(clip.prep / 'disp.npy').astype(np.float32) if kb is not None else None
    keys = list(kb['keys']) if kb is not None else list(range(0, len(frames), 10))
    out = {}
    for d in gaps:
        vals = []
        for i in range(len(keys) - d):
            k0, k1 = keys[i], keys[i + d]
            if kb is not None:
                f = float(kb['f'])
                z = keyframe_depth(kb, i, disp[k0], W, H)
                x = np.stack([(uu - W / 2) * z / f, (vv - H / 2) * z / f, z], -1).reshape(-1, 3)
                X = (x - kb['t'][i]) @ kb['R'][i]
                y = X @ kb['R'][i + d].T + kb['t'][i + d]
            else:
                f0, f = cams['f'][k0], cams['f'][k1]
                z = Z[k0]
                x = np.stack([(uu - W / 2) * z / f0, (vv - H / 2) * z / f0, z], -1).reshape(-1, 3)
                y = x @ (cams['R'][k1] @ cams['R'][k0].T).T
            q = np.stack([f * y[:, 0] / y[:, 2] + W / 2, f * y[:, 1] / y[:, 2] + H / 2], 1)
            ok = (y[:, 2] > 0) & (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1) & static[k0].ravel()
            qi = q[ok].astype(int)
            okm = static[k1][qi[:, 1], qi[:, 0]]
            a = cv2.cvtColor(frames[k0], cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()[ok][okm]
            b = cv2.cvtColor(frames[k1], cv2.COLOR_RGB2GRAY).astype(np.float32)[qi[okm, 1], qi[okm, 0]]
            a, b = a - a.mean(), b - b.mean()
            vals.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        out[d] = round(float(np.mean(vals)), 4)
    return out


def per_frame(clip, kb=None):
    """Camera and depth of every frame from the keyframe solution: rotations slerped, centres, depth affine and
    correction grids interpolated between keyframes. Writes prep/multiview/frames.npz (the pipeline's cams + depth)."""
    from scipy.spatial.transform import Slerp
    kb = dict(np.load(clip.prep / 'multiview' / 'keyframes.npz')) if kb is None else kb
    n = int(np.load(clip.prep / 'disp.npy', mmap_mode='r').shape[0])
    cam = clip.camera()
    keys = np.array(kb['keys'])
    rel = Rot.from_matrix(kb['R'])                                  # keyframe-0 camera -> keyframe camera
    centres = np.array([-kb['R'][i].T @ kb['t'][i] for i in range(len(keys))])
    t = np.clip(np.arange(n), keys[0], keys[-1])
    R_rel = Slerp(keys, rel)(t)
    c = np.stack([np.interp(t, keys, centres[:, j]) for j in range(3)], 1)
    a = np.interp(t, keys, kb['a'])
    b = np.interp(t, keys, kb['b'])
    corr = kb['corr']
    if corr.size and corr.shape[1] > 1:
        corr_f = np.stack([np.stack([np.interp(t, keys, corr[:, y, x]) for x in range(corr.shape[2])], -1)
                           for y in range(corr.shape[1])], 1)
    else:
        corr_f = np.zeros((n, 1, 1))
    R = np.array([Rr @ cam.R for Rr in R_rel.as_matrix()])         # camera axes in world
    pos = cam.pos + c @ cam.R                                       # camera centres in world
    f = float(kb['f'])
    params = np.c_[R_rel.as_rotvec(), np.zeros(n)]
    np.savez(clip.prep / 'multiview' / 'frames.npz', R=R, f=np.full(n, f), pos=pos, params=params, fit_err_px=np.zeros(n),
             a=a, b=b, corr=corr_f)
    return dict(R=R, f=np.full(n, f), pos=pos, params=params, fit_err_px=np.zeros(n))


def depth_frames(clip, disp=None):
    """(n, H, W) metric depth of every frame from the multi-view solution."""
    z = np.load(clip.prep / 'multiview' / 'frames.npz')
    disp = np.load(clip.prep / 'disp.npy').astype(np.float32) if disp is None else disp
    n, H, W = disp.shape
    out = 1.0 / np.clip(z['a'][:, None, None] * disp + z['b'][:, None, None], 1.0, None)
    if z['corr'].shape[1] > 1:
        for k in range(n):
            out[k] *= np.exp(cv2.resize(z['corr'][k].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR))
    return out.astype(np.float32)
