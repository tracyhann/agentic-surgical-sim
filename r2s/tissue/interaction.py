"""Interaction track of the 4D study: the two instruments of shot A in 3D over time, and what they do to the tissue.

    python -m r2s.tissue.interaction vNN      -> outputs/iter/tissues/interaction/vNN/

Instruments (clip chole_a): 'grasper_left' (mask 'grasper', dark, from the top left) holds the thin peritoneal sheet
and lifts it into a tent above the gallbladder; 'probe_right' (mask 'probe', white rod from the top right) pushes into
the gallbladder neck under that sheet.

Model per instrument: a straight 5 mm shaft through one fixed port P (remote centre of motion) and a per-frame distal
end T_k (the far end of the jaws / the probe tip). Fitted jointly over all frames (least squares) to
  - the far end of the instrument mask along its axis (tip pixel)
  - the mask's centre line (projected shaft line through T_k and P)
  - the apparent shaft width along the visible shaft (5 mm diameter -> depth f * d / w, which also fixes the tilt)
  - temporal smoothness of T_k, and the out-of-view constraint for frames where the instrument is not seen
  - port priors: above the working area, the tip within reach of the MuJoCo insertion range
Then: grasper jaw opening and the point it holds (apex of the tented sheet), the probe's signed indentation into the
tissue (tip depth minus the tissue depth around the tip, from V.depth) and its contact intervals.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .. import views, quality
from .. import instruments as INS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'iter' / 'tissues' / 'interaction'
SHAFT_D = 0.005                          # shaft diameter (m), both instruments
JAW_TIP_BEYOND_TCP = 0.003               # MuJoCo model: jaws end 3 mm distal of the TCP site (instruments.mjcf)
TOOLS = {'grasper_left': 'grasper', 'probe_right': 'probe'}


# ================================================================ 2D measurements from the masks
def _largest_cc(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m.astype(bool)
    keep = 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])
    out = lab == keep[0]
    # keep other big pieces (the shaft can be split by a highlight or by the other instrument)
    for i in keep[1:]:
        if st[i, cv2.CC_STAT_AREA] > 0.15 * st[keep[0], cv2.CC_STAT_AREA]:
            out |= lab == i
    return out


def _border(q, W, H):
    return min(q[0], W - 1 - q[0], q[1], H - 1 - q[1])


cfg_axis = dict(pca_min_elong=3.0, tip_on_axis=False)


def _edge_axis(m, W, H, step=4):
    """Dominant orientation (unit 2D vector, sign arbitrary) of the mask's straight silhouette edges (contour
    tangents, image-border pixels excluded), from a smoothed circular histogram of tangent angles."""
    cnts = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts:
        return None
    C = max(cnts, key=len)[:, 0].astype(float)
    if len(C) < 4 * step:
        return None
    T = np.roll(C, -step, 0) - np.roll(C, step, 0)
    inner = (C[:, 0] > 2) & (C[:, 0] < W - 3) & (C[:, 1] > 2) & (C[:, 1] < H - 3)
    inner &= (np.roll(inner, step) & np.roll(inner, -step))
    if inner.sum() < 10:
        return None
    ang = np.mod(np.arctan2(T[inner, 1], T[inner, 0]), np.pi)
    hist = np.bincount((ang / np.pi * 180).astype(int) % 180, minlength=180).astype(float)
    ker = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    hs = np.convolve(np.r_[hist[-8:], hist, hist[:8]], ker, mode='same')[8:-8]
    a0 = (np.argmax(hs) + 0.5) / 180 * np.pi
    # refine: mean of the tangent angles within 10 deg of the peak
    dd = np.angle(np.exp(2j * (ang - a0))) / 2
    sel = np.abs(dd) < np.radians(10)
    a1 = a0 + (dd[sel].mean() if sel.any() else 0)
    return np.array([np.cos(a1), np.sin(a1)])


def measure_frame(m, other, W, H, min_area=150):
    """Axis, tip, centre line and widths of one instrument mask. `other`: the other instrument's mask (pixels where
    the two touch are not used for widths). Returns None when there is too little mask."""
    if m.sum() < min_area:
        return None
    m = _largest_cc(m)
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    _, sv, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    short = sv[0] / max(sv[1], 1e-9) < cfg_axis['pca_min_elong']
    if short:
        # short, wide mask (only the jaws and a stub of shaft in view): PCA is unreliable, use the direction of the
        # straight silhouette edges instead
        de = _edge_axis(m, W, H)
        if de is not None:
            d = de
        else:
            short = False
    s = (P - c) @ d
    a, b = P[s <= np.percentile(s, 1)].mean(0), P[s >= np.percentile(s, 99)].mean(0)
    bpix = (xs <= 1) | (xs >= W - 2) | (ys <= 1) | (ys >= H - 2)
    if bpix.sum() >= 3:                               # d points from where the mask enters the picture to its end
        if (c - P[bpix].mean(0)) @ d < 0:
            d = -d
    elif _border(b, W, H) < _border(a, W, H):
        d = -d
    nb = np.minimum.reduce([P[:, 0], W - 1 - P[:, 0], P[:, 1], H - 1 - P[:, 1]]) <= 2     # pixels on the border
    for _ in range(2):                                # refine the axis on the centre line (robust)
        nrm = np.array([-d[1], d[0]])
        s, t = (P - c) @ d, (P - c) @ nrm
        bins = np.arange(s.min(), s.max() + 2, 2.0)
        idx = np.digitize(s, bins)
        cs, ct, cw, cb = [], [], [], []
        for i in np.unique(idx):
            sel = idx == i
            if sel.sum() < 3:
                continue
            tt = t[sel]
            cs.append(s[sel].mean()), ct.append(0.5 * (tt.min() + tt.max()))
            cw.append(sel.sum() / 2.0), cb.append(nb[sel].any())
        cs, ct, cw, cb = map(np.array, (cs, ct, cw, cb))
        wmed = np.median(cw[~cb]) if (~cb).sum() >= 3 else np.median(cw)
        # shaft part: whole cross-sections only (not cut by the image border), away from the distal end (jaws, tip
        # taper) by 1.5 widths
        sh = (cs < cs.max() - 1.5 * wmed) & ~cb
        if sh.sum() < 4:
            sh = (cs < cs.max() - 0.5 * wmed) & ~cb
        if sh.sum() < 4:
            break
        A = np.stack([cs[sh], np.ones(sh.sum())], 1)
        wts = np.ones(sh.sum())
        for _ in range(3):
            coef = np.linalg.lstsq(A * wts[:, None], ct[sh] * wts, rcond=None)[0]
            r = ct[sh] - A @ coef
            wts = 1 / np.maximum(1, np.abs(r) / 1.5)
        ang = 0.0 if short else np.clip(np.arctan(coef[0]), -np.radians(8), np.radians(8))
        d = np.cos(ang) * d + np.sin(ang) * nrm
        d /= np.linalg.norm(d)
        c = c + coef[1] * nrm
    nrm = np.array([-d[1], d[0]])
    s, t = (P - c) @ d, (P - c) @ nrm
    smax = s.max()
    far = s >= smax - 2.5
    tip_far = P[far].mean(0)                          # mean of the farthest pixels (can sit off the axis: end cap)
    if cfg_axis['tip_on_axis']:
        # distal end on the shaft axis: the farthest mask pixel along the axis within the shaft core (|t| < w / 3)
        cnt = np.bincount(((s - s.min()) / 2).astype(int))
        wcore = float(np.median(cnt[cnt > 0]) / 2.0)          # median width (px) over 2 px slices along the axis
        core = np.abs(t) < max(3.0, wcore / 3)
        sm = s[core].max() if core.sum() > 3 else smax
        tip = c + d * sm
    else:
        tip = tip_far
    entry = P[s <= s.min() + 2.5].mean(0)
    # centre line samples and widths (perpendicular extent, from the pixel count per 1 px of axis: robust to ragged
    # edges), bins of 3 px
    bins = np.arange(s.min(), smax + 3, 3.0)
    idx = np.digitize(s, bins)
    oth = cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) if other is not None else None
    samples = []
    for i in np.unique(idx):
        sel = idx == i
        if sel.sum() < 4:
            continue
        ss = s[sel].mean()
        tt = t[sel]
        q = c + d * ss + nrm * 0.5 * (tt.min() + tt.max())
        w_ext = tt.max() - tt.min() + 1
        w_cnt = sel.sum() / 3.0
        cut = _border(q, W, H) < 0.6 * w_ext + 2   # the image border cuts the mask here
        occ = bool(oth[P[sel, 1].astype(int), P[sel, 0].astype(int)].any()) if oth is not None else False
        samples.append((q[0], q[1], ss, w_cnt, w_ext, cut, occ))
    S = np.array(samples, float)
    tip_cut = _border(tip, W, H) < 3
    return dict(tip=tip, tip_far=tip_far, dir=d, entry=entry, c=c, smax=smax, samples=S, tip_cut=bool(tip_cut), area=int(m.sum()),
                mask=m)


def measure(V, name):
    """Per-frame 2D measurements of one instrument (name = sim name)."""
    mk = TOOLS[name]
    other = [v for k, v in TOOLS.items() if k != name][0]
    M, O = V.mask(mk), V.mask(other)
    out = []
    for k in range(V.n):
        out.append(measure_frame(M[k], O[k], V.W, V.H))
    return out


def blur_score(V):
    """Per-frame sharpness (variance of the Laplacian), for down-weighting the blurred opening pan."""
    return np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in V.frames])


# ================================================================ geometry helpers
def cam_dir(V, k, X):
    """Camera-frame coordinates of world points X (N, 3) in frame k."""
    return (np.asarray(X, float) - V.pos[k]) @ V.R[k].T


def proj(V, k, X):
    pc = cam_dir(V, k, X)
    z = pc[..., 2]
    return np.stack([V.f[k] * pc[..., 0] / z + V.W / 2, V.f[k] * pc[..., 1] / z + V.H / 2], -1), z


def tissue_depth_ring(V, k, uv, excl, r0=5, r1=14, ahead=None):
    """Median V.depth on tissue pixels in a ring r0..r1 px around uv (instrument pixels excluded)."""
    x0, y0 = int(round(uv[0])), int(round(uv[1]))
    ys, xs = np.mgrid[max(0, y0 - r1):min(V.H, y0 + r1 + 1), max(0, x0 - r1):min(V.W, x0 + r1 + 1)]
    rr = np.hypot(xs - uv[0], ys - uv[1])
    sel = (rr >= r0) & (rr <= r1) & ~excl[ys, xs]
    if ahead is not None:
        sel &= ((xs - uv[0]) * ahead[0] + (ys - uv[1]) * ahead[1]) > -0.3 * rr
    if sel.sum() < 8:
        return np.nan, 0
    return float(np.median(V.depth(k)[ys[sel], xs[sel]])), int(sel.sum())


# ================================================================ joint fit: fixed port + per-frame distal end
def _world_from_cam(V, ks, u, v, z):
    pc = np.stack([(u - V.W / 2) * z / V.f[ks], (v - V.H / 2) * z / V.f[ks], z], -1)
    return V.pos[ks] + np.einsum('ki,kij->kj', pc, V.R[ks])


def _to_cam(V, ks, X):
    return np.einsum('kij,kj->ki', V.R[ks], X - V.pos[ks])


def _proj_k(V, ks, X):
    pc = _to_cam(V, ks, X)
    z = pc[:, 2]
    return np.stack([V.f[ks] * pc[:, 0] / z + V.W / 2, V.f[ks] * pc[:, 1] / z + V.H / 2], -1), z


def pack_obs(V, meas, cfg):
    """Fixed-size observation arrays for the fit (J samples per frame, weight 0 = missing)."""
    n, J = V.n, cfg['n_samples']
    O = dict(tip=np.full((n, 2), np.nan), tip_w=np.zeros(n), C=np.zeros((n, J, 2)), Cw_=np.zeros((n, J)),
             Cpx=np.zeros((n, J, 2)), Ww=np.zeros((n, J)), wobs=np.zeros((n, J)), vis=np.zeros(n, bool),
             zobs=np.ones((n, J)), Wz=np.zeros((n, J)))
    sharp = cfg['sharp']
    depth_k = V.depth

    def spread(idx):
        return idx[np.linspace(0, len(idx) - 1, min(J, len(idx))).round().astype(int)]

    for k, m in enumerate(meas):
        if m is None:
            continue
        O['vis'][k] = True
        S = m['samples']
        wmed = np.median(S[:, 3])
        fw = float(np.clip(sharp[k] / cfg['sharp_ref'], 0.25, 1.0))   # blurred frames (opening pan) count less
        if not m['tip_cut']:
            O['tip'][k] = m['tip']
            O['tip_w'][k] = fw
        ax = (S[:, 5] == 0) & (S[:, 6] == 0)
        if ax.sum() >= 2:
            sel = spread(np.nonzero(ax)[0])
            O['C'][k, :len(sel)] = S[sel, :2]
            # samples near the tip count most: a fixed port seen through an imperfect camera path cannot match the
            # whole visible shaft in every frame, and the distal end is what touches the tissue
            dtip = np.linalg.norm(S[sel, :2] - m['tip'], axis=1)
            O['Cw_'][k, :len(sel)] = fw * (np.exp(-dtip / cfg['axis_decay']) if cfg.get('axis_decay') else 1.0)
        # widths: shaft part only (not cut by the border, not touching the other instrument, away from jaws / tip)
        wsel = ax & (S[:, 2] < m['smax'] - cfg['tip_excl_w'] * wmed) & (S[:, 3] > 6)
        if wsel.sum() >= 2:
            sel = spread(np.nonzero(wsel)[0])
            O['Cpx'][k, :len(sel)] = S[sel, :2]
            O['Ww'][k, :len(sel)] = fw
            O['wobs'][k, :len(sel)] = S[sel, 3]
            if cfg.get('w_dav2', 0) > 0:
                er = cv2.erode(m['mask'].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                Z = depth_k(k)
                for j, (u, v) in enumerate(S[sel, :2]):
                    x, y = int(round(u)), int(round(v))
                    win = (slice(max(0, y - 2), y + 3), slice(max(0, x - 2), x + 3))
                    ok = er[win]
                    if ok.sum() >= 3:
                        O['zobs'][k, j] = np.median(Z[win][ok])
                        O['Wz'][k, j] = fw
    return O


def _line_depth(V, ks, T, e, uv):
    """Depth (camera z) of the points of the 3D lines T_k + t e_k closest to the camera rays through pixels uv
    (n, J, 2); also t."""
    rc = np.stack([(uv[..., 0] - V.W / 2) / V.f[ks][:, None], (uv[..., 1] - V.H / 2) / V.f[ks][:, None],
                   np.ones(uv.shape[:2])], -1)
    r = np.einsum('kji,kil->kjl', rc, V.R[ks])
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    w0 = T - V.pos[ks]
    b = np.einsum('kji,ki->kj', r, e)
    dd = np.einsum('kji,ki->kj', r, w0)
    ee = np.einsum('ki,ki->k', w0, e)[:, None]
    t = (b * dd - ee) / np.maximum(1 - b ** 2, 1e-9)
    X = T[:, None, :] + t[..., None] * e[:, None, :]
    zc = np.einsum('ki,kji->kj', V.R[ks][:, 2, :], X - V.pos[ks][:, None, :])
    return zc, t


def residuals(x, V, O, cfg, part=False):
    n = V.n
    ks = np.arange(n)
    P = x[:3]
    kappa = x[3] if cfg.get('fit_kappa') else cfg['kappa0']
    U = x[4:].reshape(n, 3)
    z = np.exp(U[:, 2])
    T = _world_from_cam(V, ks, U[:, 0], U[:, 1], z)
    e = P - T
    L = np.linalg.norm(e, axis=1)
    e = e / L[:, None]
    out = {}
    # tip pixel
    out['tip'] = ((U[:, :2] - np.nan_to_num(O['tip'])) * O['tip_w'][:, None] / cfg['sig_tip']).ravel()
    # shaft centre line: distance of the mask's centre-line samples to the projected shaft
    a, _ = _proj_k(V, ks, T)
    b, _ = _proj_k(V, ks, T + 0.02 * e)
    dl = b - a
    dl /= np.linalg.norm(dl, axis=1, keepdims=True) + 1e-12
    nl = np.stack([-dl[:, 1], dl[:, 0]], 1)
    out['axis'] = (np.einsum('kji,ki->kj', O['C'] - a[:, None, :], nl) * O['Cw_'] / cfg['sig_axis']).ravel()
    # the visible shaft points away from the tip towards the entry side: projected port direction must agree
    side = np.einsum('kji,ki->kj', O['C'] - a[:, None, :], dl)
    out['side'] = (np.minimum(side, 0) * O['Cw_'] / 10.0).ravel()
    # apparent width: kappa * f * d / z on the shaft line at the sample pixels
    zl, _ = _line_depth(V, ks, T, e, O['Cpx'])
    wpred = kappa * V.f[:, None] * SHAFT_D / np.maximum(zl, 1e-3)
    out['width'] = ((wpred - O['wobs']) * O['Ww'] * cfg['w_width'] / cfg['sig_w']).ravel()
    if cfg.get('w_dav2', 0) > 0:
        # V.depth on the instrument's own pixels (the depth frame of all tissue models): sets the depth level
        out['dav2'] = ((zl - O['zobs']) * O['Wz'] * cfg['w_dav2'] / cfg['sig_dav2']).ravel()
    # smoothness (3D acceleration of the distal end)
    acc = T[2:] - 2 * T[1:-1] + T[:-2]
    out['smooth'] = (acc / cfg['sig_acc']).ravel()
    vel = np.diff(np.log(z))
    out['smooth_z'] = vel / cfg['sig_dlogz']
    # out of view: no mask -> the distal end projects outside the picture (or the far side of the border)
    inside = np.minimum.reduce([U[:, 0], V.W - U[:, 0], U[:, 1], V.H - U[:, 1]]) + 5
    out['outside'] = np.where(~O['vis'], np.maximum(inside, 0) / 5.0, 0.0)
    # port priors: above the working area, within reach, kappa near 1
    top = T[:, 2].max()
    out['port'] = np.array([30 * max(0.0, top + 0.01 - P[2]) / 0.01,
                            max(0.0, L.max() - cfg['reach_max']) / 0.005,
                            max(0.0, cfg['reach_min'] - L.min()) / 0.005,
                            (kappa - cfg['kappa0']) / cfg['sig_kappa']])
    # scene-consistency terms (tissue depth beside the visible shaft must lie behind the shaft)
    if cfg.get('w_free', 0) > 0:
        zb, _ = _line_depth(V, ks, T, e, O['Fpx'])
        gap = O['Fz'] - zb - cfg['free_margin']                  # >= 0 when the shaft is in front of the tissue
        out['free'] = (np.minimum(gap, 0) * O['Fw'] * cfg['w_free'] / 0.002).ravel()
    if cfg.get('w_tipz', 0) > 0:
        out['tipz'] = (np.log(z) - np.log(np.nan_to_num(O['tipz'], nan=1.0))) * O['tipz_w'] * cfg['w_tipz'] / 0.05
    if part:
        return out
    return np.concatenate(list(out.values()))


def jac_sparsity(V, O, cfg):
    n = V.n
    parts = residuals(np.r_[np.zeros(3), 1.0, np.tile([300, 200, np.log(0.08)], n)], V, O, cfg, part=True)
    nv = 4 + 3 * n
    sizes = {k: len(v) for k, v in parts.items()}
    S = lil_matrix((sum(sizes.values()), nv), dtype=int)
    r0 = 0
    for name, sz in sizes.items():
        per = sz // n if sz % n == 0 and name not in ('smooth', 'smooth_z', 'port') else None
        if name == 'smooth':
            for k in range(1, n - 1):
                for c in range(3):
                    rr = r0 + (k - 1) * 3 + c
                    for kk in (k - 1, k, k + 1):
                        S[rr, 4 + 3 * kk:4 + 3 * kk + 3] = 1
        elif name == 'smooth_z':
            for k in range(n - 1):
                S[r0 + k, 4 + 3 * k + 2] = 1
                S[r0 + k, 4 + 3 * (k + 1) + 2] = 1
        elif name == 'port':
            S[r0:r0 + sz, :] = 1
        else:
            for k in range(n):
                S[r0 + k * per:r0 + (k + 1) * per, 0:4] = 1
                S[r0 + k * per:r0 + (k + 1) * per, 4 + 3 * k:4 + 3 * k + 3] = 1
        r0 += sz
    return S.tocsr()


def free_space_obs(V, meas, cfg):
    """Tissue depth beside the visible shaft (both sides, `free_off` px beyond the mask edge, instrument pixels
    excluded): the shaft is seen in front of that tissue, so it cannot be deeper than it."""
    n, J = V.n, cfg['n_samples']
    Fpx, Fz, Fw = np.zeros((n, J, 2)), np.ones((n, J)), np.zeros((n, J))
    for k, m in enumerate(meas):
        if m is None:
            continue
        excl = cv2.dilate((V.mask('grasper')[k] | V.mask('probe')[k]).astype(np.uint8),
                          np.ones((9, 9), np.uint8)).astype(bool)
        S = m['samples']
        ok = (S[:, 5] == 0) & (S[:, 6] == 0)
        d = m['dir']
        nr = np.array([-d[1], d[0]])
        rows = []
        for u, v, s, w, we, cut, occ in S[ok]:
            zs = []
            for sg in (-1, 1):
                q = np.array([u, v]) + sg * nr * (we / 2 + cfg['free_off'])
                x, y = int(round(q[0])), int(round(q[1]))
                if 0 <= x < V.W and 0 <= y < V.H and not excl[y, x]:
                    zs.append(V.depth(k)[y, x])
            if zs:
                rows.append((u, v, max(zs)))       # the farther side: the shaft is in front of both
        if not rows:
            continue
        rows = np.array(rows)
        sel = np.linspace(0, len(rows) - 1, min(J, len(rows))).round().astype(int)
        Fpx[k, :len(sel)] = rows[sel, :2]
        Fz[k, :len(sel)] = rows[sel, 2]
        Fw[k, :len(sel)] = 1.0
    return Fpx, Fz, Fw


def init_state(V, meas, O, name, cfg):
    n = V.n
    old = V.tools[name]
    tip = np.array([m['tip'] if m is not None else [np.nan, np.nan] for m in meas], float)
    q_old, z_old = _proj_k(V, np.arange(n), old['tip'])
    for c in range(2):
        ok = np.isfinite(tip[:, c])
        tip[:, c] = np.interp(np.arange(n), np.nonzero(ok)[0], tip[ok, c])
    # depth of the distal end from the widths, extrapolated along the shaft to the tip (inverse depth is affine in
    # the image coordinate along a 3D line), median filtered
    zt = np.full(n, np.nan)
    for k, m in enumerate(meas):
        if m is None:
            continue
        S = m['samples']
        ok = (S[:, 5] == 0) & (S[:, 6] == 0) & (S[:, 2] < m['smax'] - cfg['tip_excl_w'] * np.median(S[:, 3])) & \
             (S[:, 3] > 6)
        if ok.sum() < 4:
            continue
        co = np.polyfit(S[ok, 2], S[ok, 3], 1)
        wt = np.polyval(co, m['smax'])
        if wt > 5:
            zt[k] = cfg['kappa0'] * V.f[k] * SHAFT_D / wt
    ok = np.isfinite(zt)
    zt = np.interp(np.arange(n), np.nonzero(ok)[0], zt[ok]) if ok.sum() > 5 else z_old
    from scipy.ndimage import median_filter, uniform_filter1d
    zt = uniform_filter1d(median_filter(zt, 15, mode='nearest'), 9, mode='nearest')
    zt = np.clip(zt, 0.04, 0.16)
    return tip, zt


def fit_instrument(V, meas, name, cfg, log=print):
    n = V.n
    O = pack_obs(V, meas, cfg)
    if cfg.get('w_free', 0) > 0:
        O['Fpx'], O['Fz'], O['Fw'] = free_space_obs(V, meas, cfg)
    if cfg.get('w_tipz', 0) > 0:
        O['tipz'], O['tipz_w'] = cfg['tipz'][name]
    tip, zt = init_state(V, meas, O, name, cfg)
    S = jac_sparsity(V, O, cfg)
    best = None
    old = V.tools[name]
    # port starts: the old port, and points beyond the instrument's entry side at several distances
    T0 = _world_from_cam(V, np.arange(n), tip[:, 0], tip[:, 1], zt)
    starts = [old['rcm']]
    for dist in cfg['port_starts']:
        starts.append(T0.mean(0) - dist * np.mean(old['shaft'], 0) / np.linalg.norm(np.mean(old['shaft'], 0)))
    for P0 in starts:
        x0 = np.r_[P0, cfg['kappa0'], np.c_[tip, np.log(zt)].ravel()]
        t0 = time.time()
        r = least_squares(residuals, x0, args=(V, O, cfg), jac_sparsity=S, loss='soft_l1', f_scale=2.0,
                          x_scale='jac', max_nfev=cfg['max_nfev'], method='trf')
        log(f'  [{name}] start P0={np.round(P0, 3)} -> P={np.round(r.x[:3], 4)} kappa={r.x[3]:.3f} '
            f'cost={r.cost:.1f} nfev={r.nfev} ({time.time() - t0:.0f} s)')
        if best is None or r.cost < best.cost:
            best = r
    x = best.x
    U = x[4:].reshape(n, 3)
    T = _world_from_cam(V, np.arange(n), U[:, 0], U[:, 1], np.exp(U[:, 2]))
    P = x[:3]
    sh = (T - P) / np.linalg.norm(T - P, axis=1, keepdims=True)
    parts = residuals(x, V, O, cfg, part=True)
    return dict(tip=T, rcm=P, shaft=sh, kappa=float(x[3]) if cfg.get('fit_kappa') else cfg['kappa0'], cost=float(best.cost),
                parts={k: float(np.sum(v ** 2)) for k, v in parts.items()}, obs=O)


# ================================================================ rendering and metrics
def render_shaft(V, k, T, P, r=SHAFT_D / 2, far=0.30):
    """Silhouette (H, W) bool and per-pixel depth of a straight cylinder of radius r from the distal end T back
    towards the port P (at most `far` m), in frame k. Approximated by the trapezoid between the projected end
    circles plus the disk at the distal end; inverse depth interpolated along the image axis."""
    e = P - T
    L = min(np.linalg.norm(e), far)
    e = e / np.linalg.norm(e)
    T = T + r * e                                     # rounded end: the cap's centre sits one radius behind the tip
    ts = np.linspace(0, L, 200)
    X = T + ts[:, None] * e
    q, z = proj(V, k, X)
    ok = z > 0.01
    ok &= np.abs(q).max(1) < 4000
    if not ok[0]:
        return np.zeros((V.H, V.W), bool), np.full((V.H, V.W), np.inf)
    last = np.nonzero(ok)[0]
    last = last[np.nonzero(np.diff(np.r_[last, -5]) != 1)[0][0]] if len(last) > 1 else 0
    qa, qb, za, zb = q[0], q[last], z[0], z[last]
    ra, rb = V.f[k] * r / za, V.f[k] * r / zb
    d = qb - qa
    Ld = np.linalg.norm(d)
    m = np.zeros((V.H, V.W), np.uint8)
    if Ld > 1e-6:
        d = d / Ld
        nn = np.array([-d[1], d[0]])
        poly = np.array([qa + nn * ra, qb + nn * rb, qb - nn * rb, qa - nn * ra])
        cv2.fillConvexPoly(m, np.round(poly * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    cv2.circle(m, (int(round(qa[0] * 16)), int(round(qa[1] * 16))), int(round(ra * 16)), 1, -1, cv2.LINE_8, 4)
    mask = m.astype(bool)
    ys, xs = np.mgrid[0:V.H, 0:V.W]
    lam = np.clip(((xs - qa[0]) * d[0] + (ys - qa[1]) * d[1]) / max(Ld, 1e-6), 0, 1) if Ld > 1e-6 else 0 * xs
    zmap = 1.0 / ((1 - lam) / za + lam / zb)
    return mask, np.where(mask, zmap, np.inf)


def tool_metrics(V, meas_all, tracks, ks=None):
    """Per frame, per instrument: tip reprojection error (px), shaft axis angle error (deg), centre-line offset (px),
    shaft IoU (5 mm cylinder vs mask, the two instruments z-buffered), 3D speed / acceleration of the distal end."""
    ks = range(V.n) if ks is None else ks
    names = list(tracks)
    res = {nm: dict(tip_px=np.full(V.n, np.nan), ang_deg=np.full(V.n, np.nan), offset_px=np.full(V.n, np.nan),
                    iou=np.full(V.n, np.nan)) for nm in names}
    for k in ks:
        rend = {nm: render_shaft(V, k, tracks[nm]['tip'][k], tracks[nm]['rcm'],
                                 r=tracks[nm].get('kappa', 1.0) * SHAFT_D / 2) for nm in names}
        zmin = np.min([z for _, z in rend.values()], 0)
        for nm in names:
            m = meas_all[nm][k]
            T, sh = tracks[nm]['tip'][k], tracks[nm]['shaft'][k]
            a, za = proj(V, k, T[None])
            b, zb = proj(V, k, (T - 0.02 * sh)[None])
            mk = V.mask(TOOLS[nm])[k]
            vis = rend[nm][0] & (rend[nm][1] <= zmin + 1e-9)
            u = (vis | mk).sum()
            if u > 0 and (mk.sum() > 50 or vis.sum() > 50):
                res[nm]['iou'][k] = (vis & mk).sum() / u
            if m is None:
                continue
            if not m['tip_cut']:
                res[nm]['tip_px'][k] = float(np.linalg.norm(a[0] - m['tip']))
            d2 = a[0] - b[0]                                # projected shaft direction, entry -> tip
            d2 /= np.linalg.norm(d2) + 1e-12
            res[nm]['ang_deg'][k] = float(np.degrees(np.arccos(np.clip(d2 @ m['dir'], -1, 1))))
            S = m['samples']
            ok = (S[:, 5] == 0) & (S[:, 6] == 0)
            nl = np.array([-d2[1], d2[0]])
            res[nm]['offset_px'][k] = float(np.median(np.abs((S[ok, :2] - a[0]) @ nl))) if ok.any() else np.nan
    for nm in names:
        T = tracks[nm]['tip']
        res[nm]['speed_mm'] = np.r_[np.linalg.norm(np.diff(T, axis=0), axis=1), np.nan] * 1000
        res[nm]['acc_mm'] = np.r_[np.nan, np.linalg.norm(T[2:] - 2 * T[1:-1] + T[:-2], axis=1), np.nan] * 1000
    return res


def port_consistency(V, meas, P, kappa=1.0):
    """Independent per-frame 3D shaft lines (distal end and entry-side point, both at the depth given by their own
    apparent width, scaled by kappa) -> their least-squares common point; distances (mm) of the lines from that point
    and from the fitted port P, and distance of P from each frame's back-projected axis plane (mm, depth-free)."""
    A, B, planes = [], [], []
    for k, m in enumerate(meas):
        if m is None:
            continue
        S = m['samples']
        ok = (S[:, 5] == 0) & (S[:, 6] == 0) & (S[:, 2] < m['smax'] - 1.5 * np.median(S[:, 3])) & (S[:, 3] > 6)
        if ok.sum() >= 6:
            co = np.polyfit(S[ok, 2], S[ok, 3], 1)
            s0, s1 = S[ok, 2].min(), m['smax']
            w0, w1 = np.polyval(co, s0), np.polyval(co, s1)
            if w0 > 5 and w1 > 5:
                q0 = m['c'] + m['dir'] * s0
                q1 = m['c'] + m['dir'] * s1
                X0 = V.unproject(q0[0], q0[1], kappa * V.f[k] * SHAFT_D / w0, k)
                X1 = V.unproject(q1[0], q1[1], kappa * V.f[k] * SHAFT_D / w1, k)
                A.append(X1), B.append(X0)
        # plane through the camera centre and the 2D axis
        q0 = m['c'] - 50 * m['dir']
        q1 = m['c'] + 50 * m['dir']
        X0 = V.unproject(q0[0], q0[1], 0.1, k) - V.pos[k]
        X1 = V.unproject(q1[0], q1[1], 0.1, k) - V.pos[k]
        nrm = np.cross(X0, X1)
        nrm /= np.linalg.norm(nrm)
        planes.append(abs((P - V.pos[k]) @ nrm))
    A, B = np.array(A), np.array(B)
    D = (B - A) / np.linalg.norm(B - A, axis=1, keepdims=True)
    M = np.zeros((3, 3))
    rhs = np.zeros(3)
    for a, d in zip(A, D):
        Pm = np.eye(3) - np.outer(d, d)
        M += Pm
        rhs += Pm @ a
    X = np.linalg.solve(M, rhs)

    def dist(p):
        v = p - A
        return np.linalg.norm(v - (v * D).sum(1, keepdims=True) * D, axis=1)
    return dict(common_point=X.tolist(), line_spread_mm=quality.summarize(dist(X) * 1000),
                lines_to_port_mm=quality.summarize(dist(P) * 1000), port_to_common_mm=float(np.linalg.norm(P - X) * 1000),
                plane_dist_mm=quality.summarize(np.array(planes) * 1000), n_lines=int(len(A)))


# ================================================================ interaction facts
def tcp_of(track):
    """MuJoCo TCP site (between the jaws) of a distal end: 3 mm proximal of the jaw tips along the shaft."""
    return track['tip'] - JAW_TIP_BEYOND_TCP * track['shaft']


def probe_indentation(V, track, meas, r0=4, r1=14):
    """Signed indentation (mm, + = the tip is deeper than the tissue around it) = camera depth of the probe tip minus
    the median V.depth of tissue pixels 4-14 px around the tip pixel (instrument pixels excluded, the half plane
    behind the tip along the shaft excluded); plus which tissue surrounds the tip."""
    n = V.n
    ind = np.full(n, np.nan)
    ztip = np.full(n, np.nan)
    ztis = np.full(n, np.nan)
    label = np.full(n, '', dtype=object)
    on_gb = np.full(n, np.nan)
    for k in range(n):
        m = meas[k]
        if m is None or m['tip_cut']:
            continue
        q, z = proj(V, k, track['tip'][k][None])
        q = q[0]
        excl = cv2.dilate((V.mask('grasper')[k] | V.mask('probe')[k]).astype(np.uint8),
                          np.ones((5, 5), np.uint8)).astype(bool)
        zt, cnt = tissue_depth_ring(V, k, m['tip'], excl, r0, r1, ahead=m['dir'])
        if cnt < 8:
            continue
        ztip[k], ztis[k] = z[0], zt
        ind[k] = (z[0] - zt) * 1000
        # tissue labels around the tip
        x0, y0 = int(round(m['tip'][0])), int(round(m['tip'][1]))
        ys, xs = np.mgrid[max(0, y0 - r1):min(V.H, y0 + r1 + 1), max(0, x0 - r1):min(V.W, x0 + r1 + 1)]
        sel = (np.hypot(xs - m['tip'][0], ys - m['tip'][1]) <= r1) & ~excl[ys, xs]
        labs = {nm: float(V.mask(nm)[k][ys[sel], xs[sel]].mean()) for nm in ('gallbladder', 'strand', 'liver')}
        on_gb[k] = labs['gallbladder']
        best = max(labs, key=labs.get)
        label[k] = best if labs[best] > 0.3 else 'other'
    return dict(indent_mm=ind, z_tip=ztip, z_tissue=ztis, label=label, gb_frac=on_gb)


def probe_indent_depthmap(V, meas, r0=4, r1=14):
    """Depth map only: median V.depth on the probe's own distal pixels (last 8 px along the axis, eroded mask) minus
    the tissue ring (mm). Monocular depth smooths thin instruments into their surroundings, so this is biased to 0."""
    out = np.full(V.n, np.nan)
    for k, m in enumerate(meas):
        if m is None or m['tip_cut']:
            continue
        er = cv2.erode(m['mask'].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.nonzero(er)
        s = (np.stack([xs, ys], 1) - m['c']) @ m['dir']
        sel = s >= s.max() - 8
        if sel.sum() < 3:
            continue
        zp = float(np.median(V.depth(k)[ys[sel], xs[sel]]))
        excl = cv2.dilate((V.mask('grasper')[k] | V.mask('probe')[k]).astype(np.uint8),
                          np.ones((5, 5), np.uint8)).astype(bool)
        zt, cnt = tissue_depth_ring(V, k, m['tip'], excl, r0, r1, ahead=m['dir'])
        if cnt >= 8:
            out[k] = (zp - zt) * 1000
    return out


def contact_intervals(flag):
    out, k = [], 0
    while k < len(flag):
        if flag[k]:
            j = k
            while j < len(flag) and flag[j]:
                j += 1
            out.append((int(k), int(j - 1)))
            k = j
        else:
            k += 1
    return out


def clean_runs(flag, min_on=5, min_off=5):
    f = flag.copy()
    for val, mn in ((False, min_off), (True, min_on)):           # close short gaps, then drop short runs
        k = 0
        while k < len(f):
            if f[k] == val:
                j = k
                while j < len(f) and f[j] == val:
                    j += 1
                if j - k < mn and k > 0 and j < len(f):
                    f[k:j] = not val
                k = j
            else:
                k += 1
    return f


def jaw_evidence(V, meas):
    """Distal mask width (max over the last 1.5 shaft widths) / shaft width: open jaws spread wider than the shaft
    (V shape), closed jaws taper."""
    r = np.full(V.n, np.nan)
    for k, m in enumerate(meas):
        if m is None:
            continue
        S = m['samples']
        ok = (S[:, 5] == 0) & (S[:, 6] == 0)
        if ok.sum() < 6:
            continue
        wsh = np.median(S[ok & (S[:, 2] < m['smax'] - 1.5 * np.median(S[ok, 3])), 4]) if \
            (ok & (S[:, 2] < m['smax'] - 1.5 * np.median(S[ok, 3]))).sum() >= 3 else np.median(S[ok, 4])
        dist = ok & (S[:, 2] >= m['smax'] - 1.5 * wsh)
        if dist.sum() >= 2:
            r[k] = float(S[dist, 4].max() / wsh)
    return r


def joint_targets(V, track):
    """yaw, pitch, insertion, roll, (jaw) about the port as r2s.instruments uses them (heading from the port and the
    mean TCP); frames that are just outside the joint ranges are clamped by INS.ik (reported)."""
    T = tcp_of(track)
    P = track['rcm']
    h = INS.heading_for(P, T)
    J = np.zeros((V.n, 4))
    bad = 0
    for k in range(V.n):
        try:
            J[k, :3] = INS.ik(P, T[k], h)
        except ValueError:
            bad += 1
            v = INS._rz(-h) @ (T[k] - P)
            dist = np.linalg.norm(v)
            J[k, :3] = [np.clip(np.arctan2(-v[0], v[1]), -1.57, 1.57), np.clip(np.arcsin(-v[2] / dist), 0, INS.PITCH_MAX),
                        np.clip(dist - INS.TCP_OFFSET, 0, 0.25)]
    # reconstruction error of the clamped joints
    T2 = np.array([INS.fk(P, y, p, i, h) for y, p, i in J[:, :3]])
    return J, h, bad, float(np.abs(np.linalg.norm(T2 - T, axis=1)).max() * 1000)


# ================================================================ outputs
def tube_mesh(T, P, r=SHAFT_D / 2, length=0.12, nseg=12, nring=6):
    """Cylinder verts (nring * nseg + 1, 3) from the distal end back towards the port, capped at the tip."""
    e = (P - T) / np.linalg.norm(P - T)
    a = np.cross(e, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(e, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(e, a)
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    rings = [T + t * e + r * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b) for t in np.linspace(0, length, nring)]
    return np.r_[np.concatenate(rings), T[None]]


def tube_faces(nseg=12, nring=6):
    F = []
    for i in range(nring - 1):
        for j in range(nseg):
            a, b = i * nseg + j, i * nseg + (j + 1) % nseg
            c, d = a + nseg, b + nseg
            F += [[a, b, d], [a, d, c]]
    cap = nring * nseg
    for j in range(nseg):
        F.append([cap, (j + 1) % nseg, j])
    return np.array(F)


def tips_sheet(V, meas, old, new, facts, ks, path, cols=4, scale=0.5):
    tiles = []
    for k in ks:
        img = np.ascontiguousarray(V.frames[k].copy())
        for nm in new:
            mk = V.mask(TOOLS[nm])[k].astype(np.uint8)
            cnt = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(img, cnt, -1, (255, 255, 255), 1)
            for tr, col in ((old[nm], (255, 220, 0)), (new[nm], (0, 255, 120))):
                a, _ = proj(V, k, tr['tip'][k][None])
                b, zb = proj(V, k, (tr['tip'][k] - 0.04 * tr['shaft'][k])[None])
                a, b = a[0], b[0]
                if zb[0] > 0.01:
                    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(np.round(a).astype(int)), 5, col, 2, cv2.LINE_AA)
        # rendered new shafts (5 mm cylinders) outline
        for nm in new:
            m, _ = render_shaft(V, k, new[nm]['tip'][k], new[nm]['rcm'])
            cnt = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(img, cnt, -1, (0, 200, 90), 1)
        gp, _ = proj(V, k, facts['grasp_point'][k][None])
        cv2.drawMarker(img, tuple(np.round(gp[0]).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 10, 2)
        ind = facts['probe_indent_mm'][k]
        txt = f'{k}  probe {ind:+.1f} mm' + (' CONTACT' if facts['probe_contact'][k] else '')
        cv2.putText(img, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    leg = np.zeros((26, sheet.shape[1], 3), np.uint8)
    cv2.putText(leg, 'yellow: old tip/shaft   green: new tip/shaft + 5 mm cylinder outline   white: masks   '
                'magenta: grasp point', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.concatenate([leg, sheet], 0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _ortho(Xs, R, size, pad=30):
    """Orthographic pixel coordinates of lists of world points under rotation R (rows: screen x, screen y, depth)."""
    allp = np.concatenate([np.atleast_2d(x) for x in Xs]) @ R.T
    lo, hi = allp[:, :2].min(0), allp[:, :2].max(0)
    sc = (size - 2 * pad) / max((hi - lo).max(), 1e-6)
    return lambda X: np.c_[(np.atleast_2d(X) @ R.T)[:, 0] - lo[0], hi[1] - (np.atleast_2d(X) @ R.T)[:, 1]] * sc + pad


def tips_zoom(V, meas, old, new, ks, path, half=(90, 60), zoom=2):
    """Crops around each instrument's measured tip (2x): mask outline (white), old tip/shaft (yellow), new tip/shaft
    (green) and the rendered 5 mm cylinder (thin green)."""
    rows = []
    for k in ks:
        tiles = []
        for nm in new:
            m = meas[nm][k]
            img = np.ascontiguousarray(V.frames[k].copy())
            mk = V.mask(TOOLS[nm])[k].astype(np.uint8)
            cv2.drawContours(img, cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1,
                             (255, 255, 255), 1)
            rm, _ = render_shaft(V, k, new[nm]['tip'][k], new[nm]['rcm'])
            cv2.drawContours(img, cv2.findContours(rm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)[0], -1, (0, 200, 90), 1)
            for tr, col in ((old[nm], (255, 220, 0)), (new[nm], (0, 255, 120))):
                a, _ = proj(V, k, tr['tip'][k][None])
                b, zb = proj(V, k, (tr['tip'][k] - 0.03 * tr['shaft'][k])[None])
                if zb[0] > 0.01:
                    cv2.line(img, tuple(np.round(a[0]).astype(int)), tuple(np.round(b[0]).astype(int)), col, 1,
                             cv2.LINE_AA)
                cv2.circle(img, tuple(np.round(a[0]).astype(int)), 3, col, 1, cv2.LINE_AA)
            c = m['tip'] if m is not None else proj(V, k, new[nm]['tip'][k][None])[0][0]
            x0 = int(np.clip(c[0] - half[0], 0, V.W - 2 * half[0]))
            y0 = int(np.clip(c[1] - half[1], 0, V.H - 2 * half[1]))
            crop = img[y0:y0 + 2 * half[1], x0:x0 + 2 * half[0]]
            crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, f'{k} {nm}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        rows.append(np.concatenate(tiles, 1))
    pairs = [np.concatenate(rows[i:i + 2], 1) if i + 1 < len(rows) else
             np.concatenate([rows[i], np.zeros_like(rows[i])], 1) for i in range(0, len(rows), 2)]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(pairs, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def views3d(V, new, old, path, size=420):
    """Rest scene from 3 directions (cv2 drawing): gallbladder points (frame 100), camera path (black), new ports
    (filled triangles) and tip paths (solid), old ports (crosses) and tip paths (thin), shafts every 25 frames."""
    gb = V.points('gallbladder', 100, step=6)
    panels = []
    views_ = (('scope view (frame 0)', np.stack([V.R[0][0], -V.R[0][1], V.R[0][2]])),
              ('side (y right, z up)', np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.]])),
              ('top (x right, y up)', np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.]])))
    for name, R in views_:
        Xs = [gb, V.pos] + [new[nm]['rcm'] for nm in new] + [old[nm]['rcm'] for nm in old] + [new[nm]['tip'] for nm in new]
        f = _ortho(Xs, R, size)
        img = np.full((size, size, 3), 255, np.uint8)
        for q in f(gb).astype(int):
            cv2.circle(img, tuple(q), 1, (120, 140, 60), -1)
        cv2.polylines(img, [f(V.pos).astype(np.int32)], False, (0, 0, 0), 2)
        for nm, col in (('grasper_left', (200, 40, 40)), ('probe_right', (40, 80, 220))):
            tr, ot = new[nm], old[nm]
            for k in range(0, V.n, 25):
                seg = f(np.r_[tr['tip'][k][None], tr['rcm'][None]]).astype(int)
                cv2.line(img, tuple(seg[0]), tuple(seg[1]), tuple(int(c * 0.5 + 127) for c in col), 1, cv2.LINE_AA)
            cv2.polylines(img, [f(tr['tip']).astype(np.int32)], False, col, 2, cv2.LINE_AA)
            cv2.polylines(img, [f(ot['tip']).astype(np.int32)], False, col, 1, cv2.LINE_AA)
            cv2.drawMarker(img, tuple(f(tr['rcm'])[0].astype(int)), col, cv2.MARKER_TRIANGLE_UP, 14, 3)
            cv2.drawMarker(img, tuple(f(ot['rcm'])[0].astype(int)), col, cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(img, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(img)
    img = np.concatenate(panels, 1)
    cv2.putText(img, 'red grasper, blue probe; thick = new tips, thin = old; triangle new port, x old port; black = '
                'scope path; green = gallbladder', (8, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _panel(series, ylim, title, w=1000, h=170, shade=None):
    img = np.full((h, w, 3), 255, np.uint8)
    n = max(len(s[0]) for s in series)
    X = lambda k: int(40 + (w - 50) * k / max(n - 1, 1))
    Y = lambda y: int(h - 15 - (h - 30) * (np.clip(y, *ylim) - ylim[0]) / (ylim[1] - ylim[0]))
    if shade is not None:
        for k in np.nonzero(shade)[0]:
            cv2.line(img, (X(k), 15), (X(k), h - 15), (255, 225, 190), 3)
    if ylim[0] < 0 < ylim[1]:
        cv2.line(img, (40, Y(0)), (w - 10, Y(0)), (150, 150, 150), 1)
    for k in range(0, n, 25):
        cv2.line(img, (X(k), h - 15), (X(k), h - 11), (0, 0, 0), 1)
        cv2.putText(img, str(k), (X(k) - 8, h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    for yv in ylim:
        cv2.putText(img, f'{yv:g}', (2, Y(yv) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    for y, col, thick in series:
        pts = [(X(k), Y(v)) for k, v in enumerate(y) if np.isfinite(v)]
        for a, b in zip(pts[:-1], pts[1:]):
            if b[0] - a[0] <= X(1) - X(0) + 1:
                cv2.line(img, a, b, col, thick, cv2.LINE_AA)
    cv2.putText(img, title, (45, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def series_plot(V, res_old, res_new, facts, path):
    red, blue, lred, lblue = (200, 40, 40), (40, 80, 220), (240, 160, 160), (150, 170, 240)
    G, Pn = 'grasper_left', 'probe_right'
    rows = [
        _panel([(res_old[G]['tip_px'], lred, 1), (res_old[Pn]['tip_px'], lblue, 1), (res_new[G]['tip_px'], red, 2),
                (res_new[Pn]['tip_px'], blue, 2)], (0, 100), 'tip reprojection error (px): light = old, dark = new; '
                                                              'red grasper, blue probe'),
        _panel([(res_old[G]['iou'], lred, 1), (res_old[Pn]['iou'], lblue, 1), (res_new[G]['iou'], red, 2),
                (res_new[Pn]['iou'], blue, 2)], (0, 1), 'shaft IoU (cylinder of diameter kappa * 5 mm vs mask)'),
        _panel([(res_old[G]['ang_deg'], lred, 1), (res_old[Pn]['ang_deg'], lblue, 1), (res_new[G]['ang_deg'], red, 2),
                (res_new[Pn]['ang_deg'], blue, 2)], (0, 20), 'shaft axis angle error (deg)'),
        _panel([(facts['probe_indent_raw'], lblue, 1), (facts['probe_indent_mm'], blue, 2)], (-20, 20),
               'probe indentation (mm, + = tip deeper than tissue around it); shaded = contact',
               shade=facts['probe_contact']),
        _panel([(facts['grasper_tissue_gap_mm'], red, 2), (10 * (facts['jaw_ratio'] - 1), (40, 160, 40), 1)], (-20, 20),
               'red: grasper jaw-tip depth - tissue around it (mm); green: 10 x (distal width / shaft width - 1)'),
    ]
    img = np.concatenate(rows, 0)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ================================================================ versions and build
BASE = dict(n_samples=12, sig_tip=2.0, sig_axis=1.5, sig_w=1.5, w_width=1.0, sig_acc=0.0006, sig_dlogz=0.03,
            tip_excl_w=1.5, reach_min=0.06, reach_max=0.26, kappa0=1.0, sig_kappa=1e-3, port_starts=(0.12, 0.2),
            max_nfev=60, sharp_ref=None, w_free=0.0, free_margin=0.0025, free_off=8, w_tipz=0.0,
            contact_tol_mm=2.0)
VERSIONS = {
    # v01: geometry from the masks alone: tip pixel, centre line, shaft width (5 mm, kappa = 1), one fixed port
    'v01': dict(),
    # v02: + edge-based axis for short masks (measurement fix); depth level from V.depth on the instrument's own
    # pixels (the depth frame of the tissue models), widths keep the tilt; kappa (apparent-width scale) fitted
    'v02': dict(fit_kappa=True, sig_kappa=0.25, w_dav2=1.0, sig_dav2=0.006),
    # v03: the tip first: tip sigma 1 px, centre-line samples weighted by their distance to the tip (exp(-d / 120 px)),
    # sigma 2 px; smoothness relaxed (1 mm / frame^2)
    'v03': dict(fit_kappa=True, sig_kappa=0.25, w_dav2=1.0, sig_dav2=0.006, sig_tip=1.0, sig_axis=2.0,
                axis_decay=120.0, sig_acc=0.001),
    # v04: the distal end is measured ON the mask's centre line (farthest core pixel along the axis); the probe's end
    # cap is asymmetric, so the mean of the farthest pixels sat ~5 px off the axis and the fixed-port line could not
    # pass through both. Rendered shaft ends in a rounded cap at the tip.
    'v04': dict(fit_kappa=True, sig_kappa=0.25, w_dav2=1.0, sig_dav2=0.006, sig_tip=1.0, sig_axis=2.0,
                axis_decay=120.0, sig_acc=0.001, tip_on_axis=True),
    # v05: the grasper's jaw-tip depth also from the tissue around its jaws (the sheet apex it holds; sigma ~4 mm),
    # so the grasp point sits in the sheet as V.depth sees it; probe unchanged (its indentation is measured against
    # that depth). + indentation sensitivity and a cross-check with the membrane track's sheet mask.
    'v05': dict(fit_kappa=True, sig_kappa=0.25, w_dav2=1.0, sig_dav2=0.006, sig_tip=1.0, sig_axis=2.0,
                axis_decay=120.0, sig_acc=0.001, tip_on_axis=True, w_tipz=1.0, tipz_for=('grasper_left',)),
    # v06: v05 refitted through the refined cameras (backdrop v08: rotations registered between keyframes,
    # correction median 0.3 deg, max 1.9 deg); sheet cross-check against membrane v06; per-band comparison with v05
    # on the same cameras; independent evidence on the probe's depth
    'v06': dict(fit_kappa=True, sig_kappa=0.25, w_dav2=1.0, sig_dav2=0.006, sig_tip=1.0, sig_axis=2.0,
                axis_decay=120.0, sig_acc=0.001, tip_on_axis=True, w_tipz=1.0, tipz_for=('grasper_left',),
                cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz', membrane='v06', compare_with='v05',
                depth_evidence=True),
}


def build(version, log=print):
    cfg = dict(BASE, **VERSIONS[version])
    out = OUT / version
    (out / 'work').mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    V = views.load('chole_a', 'sift', cams=str(ROOT / cfg['cams']) if cfg.get('cams') else None)
    sharp = blur_score(V)
    cfg['sharp'] = sharp
    cfg['sharp_ref'] = cfg['sharp_ref'] or float(np.percentile(sharp, 60))
    cfg_axis['tip_on_axis'] = bool(cfg.get('tip_on_axis', False))
    meas = {nm: measure(V, nm) for nm in TOOLS}
    log(f'[interaction {version}] measured masks ({time.time() - t0:.0f} s)')
    if cfg.get('w_tipz', 0) > 0:
        cfg['tipz'] = tip_depth_targets(V, meas, cfg)
    old = {nm: dict(tip=V.tools[nm]['tip'], shaft=V.tools[nm]['shaft'], rcm=V.tools[nm]['rcm']) for nm in TOOLS}
    new = {}
    for nm in TOOLS:
        new[nm] = fit_instrument(V, meas[nm], nm, cfg, log=log)
    log(f'[interaction {version}] fitted ({time.time() - t0:.0f} s)')
    res_old = tool_metrics(V, meas, old)
    res_new = tool_metrics(V, meas, new)
    # ---------------- interaction facts
    pi = probe_indentation(V, new['probe_right'], meas['probe_right'])
    from scipy.ndimage import median_filter
    raw = pi['indent_mm']
    okk = np.isfinite(raw)
    filled = np.interp(np.arange(V.n), np.nonzero(okk)[0], raw[okk])
    sm = median_filter(filled, 7, mode='nearest')
    vis_p = np.array([m is not None and not m['tip_cut'] for m in meas['probe_right']])
    contact = clean_runs(vis_p & (sm > -cfg['contact_tol_mm']))
    gq = tcp_of(new['grasper_left'])
    gi = probe_indentation(V, new['grasper_left'], meas['grasper_left'])      # same ring measure at the jaws
    jr = jaw_evidence(V, meas['grasper_left'])
    sa = sheet_apex(V, new['grasper_left'], meas['grasper_left'], cfg.get('membrane', 'v05'))
    # indentation sensitivity: ring radii, and the depth map alone (V.depth on the probe's distal pixels vs the ring)
    sens = {}
    for r0, r1 in ((8, 20), (12, 28)):
        sens[f'ring_{r0}_{r1}'] = probe_indentation(V, new['probe_right'], meas['probe_right'], r0, r1)['indent_mm']
    sens['depth_map_only'] = probe_indent_depthmap(V, meas['probe_right'])
    facts = dict(probe_indent_raw=raw, probe_indent_mm=sm, probe_contact=contact, probe_visible=vis_p,
                 grasp_point=gq, grasper_tissue_gap_mm=gi['indent_mm'], jaw_ratio=jr,
                 probe_label=pi['label'], probe_gb_frac=pi['gb_frac'])
    # ---------------- joints for the MuJoCo instruments
    joints = {}
    for nm in TOOLS:
        J, h, bad, err = joint_targets(V, new[nm])
        joints[nm] = dict(J=J, heading=h, n_clamped=bad, clamp_err_mm=err)
    # ---------------- metrics
    def summ(r):
        return {k: quality.summarize(v) for k, v in r.items()}
    early = np.arange(V.n) < 63
    Q = dict(version=version, cfg={k: v for k, v in cfg.items() if k not in ('sharp', 'tipz')},
             metrics_meaning=dict(
                 tip_px='distance (px) between the projected distal end and the far end of the instrument mask along '
                        'its axis (frames with a mask whose far end is not on the image border)',
                 ang_deg='angle between the projected shaft and the mask axis (deg)',
                 offset_px='median distance of the mask centre-line samples from the projected shaft line (px)',
                 iou='IoU of the rendered cylinder (tip back to the port, both instruments z-buffered) with the '
                     'instrument mask; diameter kappa * 5 mm (the shaft as it appears in the V.depth frame; kappa = 1 '
                     'for the old tracks)',
                 speed_mm='3D speed of the distal end (mm / frame)', acc_mm='3D acceleration (mm / frame^2)',
                 port='independent per-frame 3D shaft lines (from widths) -> spread around their common point, '
                      'distance from the fitted port; plane_dist = distance of the port from each frame\'s back-projected '
                      'axis plane (depth-free)',
                 probe_indent_mm='camera depth of the probe tip minus median V.depth of tissue 4-14 px around it '
                                 '(+ = tip deeper than the surrounding surface)',
                 grasper_tissue_gap_mm='same measure at the grasper jaw tip (the tented sheet / tissue around it)',
                 shaft_tissue_gap_mm='tissue depth beside the visible shaft minus the shaft depth there (negative = '
                                     'shaft behind the tissue it is seen in front of: inconsistent with V.depth)'),
             old={}, new={})
    for nm in TOOLS:
        Q['old'][nm] = summ(res_old[nm])
        Q['new'][nm] = summ(res_new[nm])
        Q['new'][nm]['after_pan'] = {k: quality.summarize(v[~early]) for k, v in res_new[nm].items()}
        Q['old'][nm]['after_pan'] = {k: quality.summarize(v[~early]) for k, v in res_old[nm].items()}
        Q['new'][nm]['port'] = port_consistency(V, meas[nm], new[nm]['rcm'], new[nm]['kappa'])
        Q['old'][nm]['port'] = port_consistency(V, meas[nm], old[nm]['rcm'], 1.0)
        Q['new'][nm]['fit'] = dict(rcm=new[nm]['rcm'].round(4).tolist(), kappa=new[nm]['kappa'],
                                   cost=new[nm]['cost'], parts=new[nm]['parts'],
                                   reach_mm=quality.summarize(np.linalg.norm(new[nm]['tip'] - new[nm]['rcm'], axis=1) * 1000),
                                   joints_clamped=joints[nm]['n_clamped'], joints_clamp_err_mm=joints[nm]['clamp_err_mm'])
        Q['new'][nm]['shaft_tissue_gap_mm'] = shaft_tissue_gap(V, meas[nm], new[nm])
        Q['old'][nm]['shaft_tissue_gap_mm'] = shaft_tissue_gap(V, meas[nm], old[nm])
    Q['probe'] = dict(indent_mm=quality.summarize(sm[contact]), indent_all=quality.summarize(raw),
                      contact_intervals=contact_intervals(contact),
                      contact_frames=int(contact.sum()),
                      push_intervals_gt4mm=contact_intervals(clean_runs(contact & (sm > 4.0))),
                      tissue_at_tip={lab: int((pi['label'][contact] == lab).sum()) for lab in set(pi['label'][contact])},
                      gb_frac_at_tip_contact=quality.summarize(pi['gb_frac'][contact]))
    Q['probe']['sensitivity_in_contact'] = {k: quality.summarize(v[contact]) for k, v in sens.items()}
    Q['grasper'] = dict(jaw_ratio=quality.summarize(jr), jaw_gap_mm=quality.summarize(gi['indent_mm']))
    if sa is not None:
        Q['grasper'][f"sheet_check_membrane_{cfg.get('membrane', 'v05')}"] = dict(
            touch_px=quality.summarize(sa['touch_px']), frames_touching_le5px=int(np.nansum(sa['touch_px'] <= 5)),
            tcp_minus_sheet_depth_mm=quality.summarize(sa['tcp_minus_sheet_mm']),
            tcp_to_sheet_apex_mm=quality.summarize(sa['dist_mm']))
    # ---------------- files
    verts4d = []
    F0 = tube_faces()
    faces = np.r_[F0, F0 + (F0.max() + 1)]
    for k in range(V.n):
        verts4d.append(np.r_[tube_mesh(new['grasper_left']['tip'][k], new['grasper_left']['rcm']),
                             tube_mesh(new['probe_right']['tip'][k], new['probe_right']['rcm'])])
    verts4d = np.array(verts4d)
    arrays = dict(rest_verts=verts4d[0], faces=faces, verts4d=verts4d,
                  mesh_parts=np.array(['grasper_left'] * 73 + ['probe_right'] * 73))
    for nm in TOOLS:
        arrays[f'{nm}_tip'] = new[nm]['tip']
        arrays[f'{nm}_shaft'] = new[nm]['shaft']
        arrays[f'{nm}_rcm'] = new[nm]['rcm']
        arrays[f'{nm}_tcp'] = tcp_of(new[nm])
        arrays[f'{nm}_joints'] = joints[nm]['J']
        arrays[f'{nm}_heading'] = np.array(joints[nm]['heading'])
        arrays[f'{nm}_visible'] = np.array([m is not None for m in meas[nm]])
        arrays[f'{nm}_old_tip'] = old[nm]['tip']
    if sa is not None:
        ap = sa['apex'].copy()
        for c in range(3):
            okc = np.isfinite(ap[:, c])
            ap[:, c] = np.interp(np.arange(V.n), np.nonzero(okc)[0], ap[okc, c])
        arrays['grasp_apex_sheet'] = ap
    arrays.update(grasper_jaw=np.zeros(V.n), grasp_point=gq, probe_indent_mm=sm, probe_indent_raw=np.nan_to_num(raw, nan=np.nan),
                  probe_contact=contact, grasper_jaw_ratio=jr)
    np.savez_compressed(out / 'model.npz', **arrays)
    ks = list(range(0, V.n, 10))
    tips_sheet(V, meas, old, new, facts, ks, out / 'tips_sheet.jpg')
    tips_zoom(V, meas, old, new, list(range(0, V.n, 20)), out / 'tips_zoom.jpg')
    quality.contact_sheet(V, [('grasper_left (5 mm shaft)', verts4d[:, :73], F0, (255, 80, 80)),
                              ('probe_right (5 mm shaft)', verts4d[:, 73:], F0, (80, 160, 255))], V.keyframes,
                          out / 'sheet.jpg')
    views3d(V, new, old, out / 'views3d.jpg')
    series_plot(V, res_old, res_new, facts, out / 'series.png')
    if cfg.get('compare_with'):
        Q['bands'] = band_report(V, meas, old, new, cfg['compare_with'], log=log)
    if cfg.get('depth_evidence'):
        Q['probe_depth_evidence'] = probe_depth_evidence(V, meas['probe_right'], new['probe_right'], contact,
                                                         cfg.get('membrane', 'v05'), out / 'work')
    json.dump(_jsonable(Q), open(out / 'quality.json', 'w'), indent=1)
    np.savez_compressed(out / 'work' / 'per_frame.npz', **{f'{w}_{nm}_{k}': v for w, R in (('old', res_old), ('new', res_new))
                                                       for nm in R for k, v in R[nm].items()},
                        probe_label=pi['label'].astype(str))
    log(f'[interaction {version}] done ({time.time() - t0:.0f} s)')
    return Q


BANDS = ((0, 62), (62, 130), (130, 251))


def band_report(V, meas, old, new, prev_version, log=print):
    """Tip reprojection / shaft angle / centre-line offset / IoU per frame band, for the old tracks, a previous
    version's tracks and this version's, all measured through this V (e.g. the refined cameras)."""
    d = np.load(OUT / prev_version / 'model.npz')
    prev = {nm: dict(tip=d[f'{nm}_tip'], shaft=d[f'{nm}_shaft'], rcm=d[f'{nm}_rcm'],
                     kappa=float(json.load(open(OUT / prev_version / 'quality.json'))['new'][nm]['fit']['kappa']))
            for nm in TOOLS}
    R = {'old': tool_metrics(V, meas, old), prev_version: tool_metrics(V, meas, prev), 'this': tool_metrics(V, meas, new)}
    out = {}
    for nm in TOOLS:
        for key in ('tip_px', 'ang_deg', 'offset_px', 'iou'):
            for lo, hi in BANDS:
                for w, r in R.items():
                    out.setdefault(nm, {}).setdefault(key, {}).setdefault(f'{lo}-{hi - 1}', {})[w] = \
                        float(np.nanmedian(r[nm][key][lo:hi])) if np.isfinite(r[nm][key][lo:hi]).any() else None
    for nm in TOOLS:
        for key in ('tip_px', 'ang_deg'):
            log(f'  {nm} {key}: ' + ' | '.join(f"{b}: " + ' '.join(f"{w} {v:.2f}" for w, v in out[nm][key][b].items()
                                                                   if v is not None) for b in out[nm][key]))
    return out


def probe_depth_evidence(V, meas, track, contact, membrane, work):
    """Independent evidence on the probe's depth along the viewing rays (mm, + = away from the camera), per frame:
    - width_only_mm: where the 5 mm width ruler alone (kappa = 1) puts the tip, relative to the fitted tip
    - away_max_mm: free space. The distal half of the visible shaft is seen in front of the tissue beside it
      (V.depth 8 px beyond the mask edge), so the shaft can move away from the camera by at most the 20th percentile
      of (tissue - shaft - radius) over those samples
    - toward_max_mm: the sheet in front. In frames where the probe tip lies under the sheet (membrane mask covering
      the pixels just around the tip), the tip is behind the sheet: it can move toward the camera by at most
      (tip depth - V.depth of the sheet pixels around the tip)
    - cap_visible: the distal end shows a whole rounded cap (the width over the last radius of the mask grows like a
      hemisphere seen from the side); a tip buried in a dimple would be cut straight by the dimple's rim
    - depth_map_only_mm: V.depth on the probe's own distal pixels - tissue ring (biased to 0 by monocular smoothing)."""
    n = V.n
    ks = np.arange(n)
    e = track['rcm'] - track['tip']
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    r = track.get('kappa', 1.0) * SHAFT_D / 2
    # free space
    Fpx, Fz, Fw = free_space_obs(V, meas, dict(n_samples=12, free_off=8))
    zl, _ = _line_depth(V, ks, track['tip'], e, Fpx)
    away = np.full(n, np.nan)
    for k in ks:
        sel = Fw[k] > 0
        if sel.sum() >= 4:
            g = (Fz[k, sel] - zl[k, sel] - r)[sel.sum() // 2:]          # distal half
            away[k] = np.percentile(g, 20) * 1000
    # width ruler alone
    width_only = np.full(n, np.nan)
    zt = proj(V, 0, track['tip'][:1])[1]
    for k, m in enumerate(meas):
        if m is None or m['tip_cut']:
            continue
        S = m['samples']
        ok = (S[:, 5] == 0) & (S[:, 6] == 0) & (S[:, 2] < m['smax'] - 1.5 * np.median(S[:, 3])) & (S[:, 3] > 6)
        if ok.sum() >= 4:
            co = np.polyfit(S[ok, 2], S[ok, 3], 1)
            wt = np.polyval(co, m['smax'])
            if wt > 5:
                width_only[k] = (V.f[k] * SHAFT_D / wt - proj(V, k, track['tip'][k][None])[1][0]) * 1000
    del zt
    # sheet in front of the tip
    toward = np.full(n, np.nan)
    under = np.zeros(n, bool)
    f = OUT.parent / 'membrane' / membrane / 'masks.npz'
    if f.exists():
        M = np.load(f)['membrane_clean']
        for k, m in enumerate(meas):
            if m is None or m['tip_cut']:
                continue
            x0, y0 = m['tip']
            ys, xs = np.mgrid[max(0, int(y0) - 10):min(V.H, int(y0) + 11), max(0, int(x0) - 10):min(V.W, int(x0) + 11)]
            ring = (np.hypot(xs - x0, ys - y0) <= 10) & ~V.mask('probe')[k][ys, xs]
            sheet = ring & M[k][ys, xs]
            if ring.sum() > 0 and sheet.sum() >= 0.5 * ring.sum():
                under[k] = True
                zs = float(np.median(V.depth(k)[ys[sheet], xs[sheet]]))
                toward[k] = (proj(V, k, track['tip'][k][None])[1][0] - zs) * 1000
    # rounded cap visible at the distal end
    cap = np.full(n, np.nan)
    for k, m in enumerate(meas):
        if m is None or m['tip_cut']:
            continue
        S = m['samples']
        ok = (S[:, 5] == 0) & (S[:, 6] == 0)
        if ok.sum() < 6:
            continue
        wsh = np.median(S[ok, 4][-12:-4]) if ok.sum() >= 12 else np.median(S[ok, 4])
        last = ok & (S[:, 2] >= m['smax'] - 0.5 * wsh)
        if last.sum() >= 2:
            # width at the last third of a radius vs a hemisphere seen side-on (0.94 of the diameter at d = r/3 ...
            # 0 at the end): a cut (buried) end keeps nearly the full width up to the end
            cap[k] = float(S[last, 4][-1] / wsh)
    dm = probe_indent_depthmap(V, meas)
    np.savez_compressed(work / 'probe_depth_evidence.npz', away_max_mm=away, toward_max_mm=toward, under_sheet=under,
                        width_only_mm=width_only, cap_end_width_ratio=cap, depth_map_only_mm=dm)
    c = contact
    return dict(
        meaning='mm along the viewing ray, + = away from the camera, relative to this version\'s probe tip; '
                'contact frames only unless noted',
        width_only_mm=quality.summarize(width_only[c]),
        away_max_mm=quality.summarize(away[c]),
        frac_contact_frames_away_lt_0=float(np.nanmean(away[c] < 0)),
        under_sheet_frames=int(under[c].sum()),
        toward_max_mm=quality.summarize(toward[c & under]),
        cap_end_width_ratio=quality.summarize(cap[c]),
        depth_map_only_indent_mm=quality.summarize(dm[c]),
        shift_minus2mm_consistent_frac=float(np.nanmean(np.where(under[c], toward[c] >= 2.0, True) & (away[c] >= -2.0)
                                                        if c.any() else np.nan)))


def shaft_tissue_gap(V, meas, track, off=8):
    """Median over frames of (tissue depth beside the visible distal half of the shaft - shaft depth) (mm)."""
    Fpx, Fz, Fw = free_space_obs(V, meas, dict(n_samples=12, free_off=off))
    ks = np.arange(V.n)
    e = (track['rcm'] - track['tip'])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    zl, t = _line_depth(V, ks, track['tip'], e, Fpx)
    g = (Fz - zl) * 1000
    dist = Fw > 0
    per = np.array([np.median(g[k, dist[k]][len(g[k, dist[k]]) // 2:]) if dist[k].sum() > 3 else np.nan for k in ks])
    return dict(distal_half=quality.summarize(per), frac_frames_negative=float(np.nanmean(per < 0)))


def tip_depth_targets(V, meas, cfg):
    """Depth targets for the distal end from the tissue around it (V.depth, ring 4-14 px, instrument pixels
    excluded), only for the instruments listed in cfg['tipz_for'] (the grasper: its closed jaws sit in the apex of the
    sheet they hold). Never for the probe, whose indentation is measured against exactly this depth."""
    out = {}
    for nm in TOOLS:
        z = np.full(V.n, np.nan)
        w = np.zeros(V.n)
        if nm in cfg.get('tipz_for', ()):
            for k, m in enumerate(meas[nm]):
                if m is None or m['tip_cut']:
                    continue
                excl = cv2.dilate((V.mask('grasper')[k] | V.mask('probe')[k]).astype(np.uint8),
                                  np.ones((5, 5), np.uint8)).astype(bool)
                zt, cnt = tissue_depth_ring(V, k, m['tip'], excl, 4, 14, ahead=m['dir'])
                if cnt >= 8:
                    z[k], w[k] = zt, 1.0
        out[nm] = (z, w)
    return out


def sheet_apex(V, track, meas, version='v05'):
    """Cross-check with the membrane track (read-only): the sheet pixels (membrane/<version>/masks.npz,
    'membrane_clean') nearest the grasper's distal end; their distance (px) to the jaw tip and their median V.depth
    (the sheet apex as the video depth sees it) vs the TCP depth."""
    f = OUT.parent / 'membrane' / version / 'masks.npz'
    if not f.exists():
        return None
    M = np.load(f)['membrane_clean']
    n = V.n
    touch = np.full(n, np.nan)
    apex = np.full((n, 3), np.nan)
    dz = np.full(n, np.nan)
    tcp = tcp_of(track)
    for k in range(n):
        ys, xs = np.nonzero(M[k])
        if len(xs) == 0:
            continue
        q, z = proj(V, k, track['tip'][k][None])
        dd = np.hypot(xs - q[0, 0], ys - q[0, 1])
        touch[k] = dd.min()
        sel = dd <= dd.min() + 6
        zz = float(np.median(V.depth(k)[ys[sel], xs[sel]]))
        apex[k] = V.unproject(np.median(xs[sel]), np.median(ys[sel]), zz, k)
        dz[k] = (proj(V, k, tcp[k][None])[1][0] - zz) * 1000
    return dict(touch_px=touch, apex=apex, tcp_minus_sheet_mm=dz,
                dist_mm=np.linalg.norm(apex - tcp, axis=1) * 1000)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == '__main__':
    v = sys.argv[1] if len(sys.argv) > 1 else 'v01'
    build(v)
