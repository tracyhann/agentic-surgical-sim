"""Ducts track: the cystic duct and the pink strands of shot A (chole_a) as tubes (generalised cylinders).

    python -m r2s.tissue.ducts vNN [--resegment]

Template = tube: an ordered centreline (S samples, proximal -> distal) + a radius profile (S,), swept into a closed
triangle surface (rings of N_RING vertices + one centre vertex per end cap). Each tube is fitted
  rest: one static centreline jointly to all usable keyframes (multi-view: reprojection chamfer to the mask skeletons
        of every keyframe + the video depth at the skeleton, end-point terms, bending / spacing regularisers);
  4D:   one centreline per video frame (251) jointly, same data terms, temporal smoothness (1st and 2nd differences
        over frames), a soft length term towards the rest length and a weak pull towards the rest shape;
radius profile from the mask width (distance transform at the skeleton) x depth / f, median over frames.

Tubes (see NOTES.md of a version for the anatomical reading):
  duct      the shiny purple-white cord that continues the gallbladder neck and runs down-right (cystic duct);
            proximal end attached to the gallbladder neck, distal end where it disappears into the fat (backdrop)
  strands   the band of thin pink fibres that leave the neck / duct junction and run right / up-right (cystic
            artery and peritoneal strands). The fibres are not separable in the masks: the band is fitted as one
            bundle (centreline + half-width) and laid out as 3 packed fibre tubes strand_1..3 across its width;
            proximal ends at the neck (gallbladder), distal ends on the tissue at right (backdrop)
Per frame the duct section is an area-preserving ellipse following the observed width, and the fibre spacing follows
the band's width (width_scale). The duct also gets a tet mesh (tube_tets). cable_spec(version) gives the integrator
node chains for a cable / string simulation.
v09 (integration feedback): every tube's proximal end rides on one gallbladder-model vertex (CFG gb_version, chosen
among the vertices that model marks 'ducts', closest to the observed junction), hard in rest and every frame; the
distal ends stay free per frame (cfg distal_fixed=True pins them to one world point instead); one common length per
tube is fitted in 4D and the rest is refitted to it; a curvature barrier keeps the bend radius above the tube radius.

Outputs (CONTRACT.md): outputs/iter/tissues/ducts/vNN/{model.npz, quality.json, sheet.jpg, views3d.jpg, masks.npz,
NOTES.md (written by hand)}.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .. import views, quality

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iter/tissues/ducts'
N_RING = 12

# SAM 2.1 prompts (pixel x, y) of the masks this track makes itself. The duct uses the clip's 'strand' mask (it is
# the shiny cord) with the gallbladder mask NOT subtracted (that mask leaks onto the cord in frames ~170-250).
PROMPTS = {
    'strands': [
        {'frame': 0, 'pos': [[360, 195], [390, 192], [415, 195]], 'neg': [[370, 250], [470, 220], [400, 160], [300, 200]]},
        {'frame': 20, 'pos': [[370, 195], [400, 192], [430, 200]], 'neg': [[380, 280], [490, 200], [420, 165], [290, 240]]},
        {'frame': 40, 'pos': [[410, 172], [430, 170], [470, 172]], 'neg': [[430, 235], [545, 175], [450, 130], [300, 200]]},
        {'frame': 100, 'pos': [[430, 158], [460, 152], [490, 163]], 'neg': [[440, 205], [560, 150], [470, 120], [350, 190]]},
        {'frame': 160, 'pos': [[445, 125], [470, 127], [490, 138]], 'neg': [[440, 180], [570, 140], [480, 100], [330, 150]]},
        {'frame': 220, 'pos': [[490, 150], [515, 156], [535, 160]], 'neg': [[480, 215], [585, 130], [500, 120], [400, 160]]},
    ],
}

# per-tube settings: samples along the centreline, which end is proximal, attachments, colour in the sheets
TUBES = {
    'duct': dict(S=24, proximal='gallbladder', attach=('gallbladder', 'backdrop'), rgb=(255, 0, 255),
                 r_range=(0.8e-3, 3.0e-3)),
    'strands': dict(S=20, proximal='left', attach=('gallbladder', 'backdrop'), rgb=(0, 255, 255),
                    r_range=(0.4e-3, 2.0e-3)),
}

CFG = dict(min_area=150, step_px=3.0, junction_px=12, strand_junction_px=45, sigma_px=3.0, sigma_mm=2.5,
           w_end=0.5, w_depth=1.0, w_bend=20.0, w_space=5.0, w_t1=2.0, w_t2=8.0, w_len=20.0, w_anchor=5.0, sigma_anchor_mm=1.5, w_rest=0.002,
           iters_rest=600, iters_4d=900, rest_from=40, n_fibres=3, width_scale=True, s_sigma=4.0, s_range=(0.75, 1.4),
           flat_ends={'duct': (0.15, 0.0), 'strands': (0.0, 0.0)}, fibre_const_r=False,
           gb_version='v12', anchor_from=60, fibre_ramp=0.35, distal_fixed=False, w_curv=200.0, max_kr=0.85,
           cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz', depth_blend=True,
           r_prox_mm={'duct': 1.1, 'strands': 1.0}, bands=((0, 62), (62, 130), (130, 251)),
           gb_front=False, w_front=20.0, front_margin_mm=0.3, anchor_mode='patch',
           fibre_ramp_start=0.0, bridge_frac=0.2, rest_ref='median_frame')


# ----------------------------------------------------------------------------------------------- segmentation
def make_masks(V, path, resegment=False):
    """{'duct': (n, H, W) bool, 'strands': (n, H, W) bool}; SAM results cached in `path` (masks.npz)."""
    path = Path(path)
    have = dict(np.load(path)) if path.exists() else {}
    out = {'duct': V.mask('strand').copy()}
    for name, prompts in PROMPTS.items():
        if name in have and not resegment:
            out[name] = have[name].astype(bool)
            continue
        t = time.time()
        out[name] = views.segment_object(V, prompts)
        print(f'SAM {name}: {time.time() - t:.0f} s')
    np.savez_compressed(path, **{k: v for k, v in out.items()})
    return out


# ----------------------------------------------------------------------------------------------- 2D observations
def clean(m, min_area):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i) if st[i, cv2.CC_STAT_AREA] >= min_area else np.zeros(m.shape, bool)


def longest_path(skel):
    """Ordered (x, y) pixels of the longest geodesic path in a skeleton (two BFS passes)."""
    from collections import deque
    ys, xs = np.nonzero(skel)
    if len(xs) < 3:
        return None
    idx = {(x, y): i for i, (x, y) in enumerate(zip(xs, ys))}
    nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

    def bfs(s):
        dist = np.full(len(xs), -1)
        par = np.full(len(xs), -1)
        dist[s] = 0
        q = deque([s])
        while q:
            i = q.popleft()
            for dx, dy in nb:
                j = idx.get((xs[i] + dx, ys[i] + dy))
                if j is not None and dist[j] < 0:
                    dist[j] = dist[i] + 1
                    par[j] = i
                    q.append(j)
        return dist, par

    d0, _ = bfs(0)
    a = int(np.argmax(d0))
    da, par = bfs(a)
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a:
        path.append(par[path[-1]])
    return np.stack([xs[path], ys[path]], 1).astype(float)


def resample(P, step):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return P[[0, -1]]
    t = np.linspace(0, s[-1], max(int(round(s[-1] / step)) + 1, 2))
    return np.stack([np.interp(t, s, P[:, c]) for c in range(P.shape[1])], 1)


def extend(P, m, end, max_px=40):
    """Prolong the path from its end (0 or -1) along the end tangent while inside the mask (skeletons stop short)."""
    Q = P if end == -1 else P[::-1]
    d = Q[-1] - Q[max(len(Q) - 6, 0)]
    d = d / max(np.linalg.norm(d), 1e-9)
    add = []
    for s in range(1, max_px):
        x, y = Q[-1] + s * d
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < m.shape[1] and 0 <= yi < m.shape[0]) or not m[yi, xi]:
            break
        add.append([x, y])
    if add:
        Q = np.concatenate([Q, np.array(add)], 0)
    return Q if end == -1 else Q[::-1]


def observe(V, M, name, cfg, junction=None):
    """Per frame: ordered skeleton samples (proximal first) with depth, radius (m) and validity flags.
    junction: per-frame (x, y) of the neck end the proximal end is pulled to (for the strands: the duct's junction)."""
    from skimage.morphology import skeletonize
    tube = TUBES[name]
    obs = []
    for k in range(V.n):
        m = clean(M[name][k], cfg['min_area'])
        o = dict(ok=False)
        if m.sum() >= cfg['min_area']:
            sk = skeletonize(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) | False)
            P = longest_path(sk) if sk.sum() >= 3 else None
            if P is not None and len(P) >= 4:
                P = resample(cv2.GaussianBlur(P.reshape(-1, 1, 2).astype(np.float32), (1, 7), 0).reshape(-1, 2)
                             if len(P) > 7 else P, 1.0)
                P = extend(extend(P, m, 0), m, -1)
                gb = V.mask('gallbladder')[k] & ~cv2.dilate(M['duct'][k].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                # proximal end first
                if tube['proximal'] == 'gallbladder':   # the cord always runs from the neck (upper) downwards
                    if P[-1][1] < P[0][1]:
                        P = P[::-1]
                else:
                    if P[-1][0] < P[0][0]:
                        P = P[::-1]
                # pull the proximal end to the neck junction
                jpt = None
                if tube['proximal'] == 'gallbladder' and gb.any():
                    ys, xs = np.nonzero(gb)
                    dd = np.hypot(xs - P[0][0], ys - P[0][1])
                    i = int(np.argmin(dd))
                    if dd[i] <= cfg['junction_px']:
                        jpt = np.array([xs[i], ys[i]], float)
                elif junction is not None and junction[k] is not None:
                    if np.hypot(*(junction[k] - P[0])) <= cfg['strand_junction_px']:
                        jpt = junction[k]
                if jpt is not None and np.hypot(*(jpt - P[0])) > 1:
                    P = np.concatenate([jpt[None], P], 0)
                P = resample(P, cfg['step_px'])
                dtm = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                xi = np.clip(np.round(P[:, 0]).astype(int), 0, V.W - 1)
                yi = np.clip(np.round(P[:, 1]).astype(int), 0, V.H - 1)
                dep = cv2.medianBlur(V.depth(k).astype(np.float32), 5)[yi, xi]
                rad_px = dtm[yi, xi]
                rad = rad_px * dep / V.f[k]
                unk = cv2.dilate((V.occluders(k)).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
                border = lambda p: p[0] < 4 or p[1] < 4 or p[0] > V.W - 5 or p[1] > V.H - 5
                end_ok = [not (border(P[e]) or unk[yi[e], xi[e]]) for e in (0, -1)]
                o = dict(ok=True, uv=P, depth=dep, rad=rad, rad_px=rad_px, end_ok=end_ok, junction=jpt,
                         in_mask=m[yi, xi])
        obs.append(o)
    return obs


# ----------------------------------------------------------------------------------------------- fitting
def _torch_obs(V, obs, ks, rad_prior):
    import torch
    Mmax = max([len(obs[k]['uv']) for k in ks if obs[k]['ok']] + [2])
    T = len(ks)
    s = np.zeros((T, Mmax, 2)); d = np.zeros((T, Mmax)); w = np.zeros((T, Mmax)); r = np.zeros((T, Mmax))
    ends = np.zeros((T, 2, 2)); end_w = np.zeros((T, 2)); fw = np.zeros(T)
    for t, k in enumerate(ks):
        o = obs[k]
        if not o['ok']:
            continue
        m = len(o['uv'])
        s[t, :m] = o['uv']; d[t, :m] = o['depth']; w[t, :m] = 1; r[t, :m] = np.clip(o['rad'], *rad_prior)
        ends[t] = o['uv'][[0, -1]]
        end_w[t] = o['end_ok']
        fw[t] = 1
    f = lambda a: torch.tensor(a, dtype=torch.float64)
    return dict(s=f(s), d=f(d) * 1000, w=f(w), r=f(r) * 1000, ends=f(ends), end_w=f(end_w), fw=f(fw),
                R=f(V.R[ks]), pos=f(V.pos[ks]) * 1000, foc=f(V.f[ks]), unk=[_unknown(V, k) for k in ks],
                dz=f(np.zeros(T)))


def depth_offsets(V, obs, anchor, r0, ks, sigma=3.0):
    """Per frame k in ks: camera depth of the anchor point (anchor (3,) static or (n, 3) per video frame) minus the
    video's depth (+ radius r0) at the first observed skeleton sample (the junction), m; smoothed over time (Gaussian,
    sigma frames), gaps interpolated. Used to tilt the depth targets towards the anchor near the proximal end."""
    A = np.asarray(anchor, float)
    raw = np.full(len(ks), np.nan)
    for t, k in enumerate(ks):
        o = obs[k]
        if o['ok']:
            _, za = V.project((A if A.ndim == 1 else A[k])[None], k)
            raw[t] = za[0] - (o['depth'][0] + r0)
    ok = np.isfinite(raw)
    if not ok.any():
        return np.zeros(len(ks)), raw
    x = np.interp(np.arange(len(ks)), np.nonzero(ok)[0], raw[ok])
    if len(ks) > 7 and sigma > 0:
        g = np.exp(-0.5 * (np.arange(-10, 11) / sigma) ** 2); g /= g.sum()
        x = np.convolve(np.pad(x, 10, mode='edge'), g, mode='valid')
    return x, raw


def _project(X, R, pos, foc, W, H):
    """X (T, S, 3) mm world -> uv (T, S, 2), z (T, S) mm."""
    import torch
    pc = torch.einsum('tij,tsj->tsi', R, X - pos[:, None])
    z = pc[..., 2]
    u = foc[:, None] * pc[..., 0] / z + W / 2
    v = foc[:, None] * pc[..., 1] / z + H / 2
    return torch.stack([u, v], -1), z


def _data_terms(X, O, W, H, cfg, unk_w):
    import torch
    q, z = _project(X, O['R'], O['pos'], O['foc'], W, H)
    D2 = ((q[:, :, None] - O['s'][:, None]) ** 2).sum(-1)                         # (T, S, M)
    big = 1e8 * (1 - O['w'][:, None])
    sig2 = cfg['sigma_px'] ** 2
    rho = lambda x2: sig2 * torch.log1p(x2 / sig2)
    d_cs, j = (D2 + big).min(2)                                                    # curve -> skeleton
    d_sc, _ = (D2 + 1e8 * (unk_w[:, :, None] == 0)).min(1)                         # skeleton -> visible curve
    fw = O['fw']
    nS = X.shape[1]
    e_cs = (rho(d_cs) * unk_w).sum(1) / unk_w.sum(1).clamp(min=1)
    e_sc = (rho(d_sc.clamp(max=1e6)) * O['w']).sum(1) / O['w'].sum(1).clamp(min=1)
    e_end = (rho(((q[:, [0, -1]] - O['ends']) ** 2).sum(-1)) * O['end_w']).sum(1)
    # depth: curve centre = video surface depth at the matched skeleton point + radius
    zt = torch.gather(O['d'] + O['r'], 1, j)
    # depth targets tilted towards the anchor's depth near the proximal end (1 at the anchor, 0 at the distal end)
    zt = zt + O['dz'][:, None] * torch.linspace(1, 0, X.shape[1], dtype=torch.float64)[None]
    near = (d_cs.detach() < 64).double() * unk_w
    sz2 = cfg['sigma_mm'] ** 2
    e_z = (sz2 * torch.log1p((z - zt) ** 2 / sz2) * near).sum(1) / near.sum(1).clamp(min=1)
    E = fw * (e_cs + e_sc + cfg['w_end'] * e_end + cfg['w_depth'] * e_z)
    return E, q, z, d_cs


def _visibility(q, O, W, H):
    """1 where a projected curve point is in the image and not on an instrument (else unknown), no gradient."""
    import torch
    qn = q.detach().numpy()
    out = np.zeros(qn.shape[:2])
    for t in range(len(qn)):
        x = np.round(qn[t, :, 0]).astype(int); y = np.round(qn[t, :, 1]).astype(int)
        inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        ok = inside.copy()
        ok[inside] = ~O['unk'][t][y[inside], x[inside]]
        out[t] = ok
    return torch.tensor(out, dtype=torch.float64)


def gb_depth(V, X, F, ks):
    """z-buffer (mm, inf where empty) of the gallbladder model in frames ks; X (N, 3) static or (n, N, 3) per frame."""
    out = []
    for k in ks:
        m, z = quality.render(V, X, F, k)
        out.append(np.where(m, z * 1000, np.inf).astype(np.float32))
    return out


def _front_term(q, z, O, cfg):
    """Penalty for curve samples (from the 3rd on) that lie behind the gallbladder model along the camera ray: the
    centre must be at least its radius (+ margin) in front of the gallbladder surface seen at that pixel (the video
    shows the tube in front of it)."""
    import torch
    if O.get('zgb') is None:
        return torch.zeros((), dtype=torch.float64)
    qn = q.detach().numpy()
    zg = np.full(qn.shape[:2], np.inf)
    for t in range(len(qn)):
        x = np.round(qn[t, :, 0]).astype(int); y = np.round(qn[t, :, 1]).astype(int)
        ok = (x >= 0) & (x < O['zgb'][t].shape[1]) & (y >= 0) & (y < O['zgb'][t].shape[0])
        zg[t, ok] = O['zgb'][t][y[ok], x[ok]]
    zg[:, :2] = np.inf
    zg = torch.tensor(np.where(np.isfinite(zg), zg, 1e6))
    lim = zg - O['r_front'][None] - cfg['front_margin_mm']
    return cfg['w_front'] * (torch.relu(z - lim) ** 2).sum()


def _regs(X, cfg, L_rest=None):
    import torch
    d1 = X[:, 1:] - X[:, :-1]
    seg = d1.norm(dim=-1)
    L = seg.sum(1)
    e_bend = ((X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]) ** 2).sum((1, 2))
    e_space = ((seg - L[:, None] / seg.shape[1]) ** 2).sum(1)
    E = cfg['w_bend'] * e_bend + cfg['w_space'] * e_space
    if L_rest is not None:
        E = E + cfg['w_len'] * ((L - L_rest) / L_rest) ** 2 * 100
    return E, L


def init_curve(V, obs, k, S):
    o = obs[k]
    P3 = V.unproject(o['uv'][:, 0], o['uv'][:, 1], o['depth'] + o['rad'], k)
    return resample_n(P3, S)


def resample_n(P, S):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], S)
    return np.stack([np.interp(t, s, P[:, c]) for c in range(3)], 1)


def _anchor(X0, A, cfg):
    """Robust 3D pull of the proximal end X0 (T, 3) mm to anchor points A (T, 3) mm."""
    import torch
    s2 = cfg['sigma_anchor_mm'] ** 2
    return cfg['w_anchor'] * s2 * torch.log1p(((X0 - A) ** 2).sum(-1) / s2)


def _blend_start(X, A):
    """Move the start of curve(s) X (..., S, 3) onto A (..., 3), fading the shift linearly along the curve."""
    S = X.shape[-2]
    w = (1 - np.linspace(0, 1, S))[:, None]
    return X + w * (A - X[..., :1, :])


def fit_rest(V, obs, name, cfg, log, anchor=None, fix_start=None, fix_end=None, init=None, L_target=None, dz=False,
             front=None):
    """Static centreline (S, 3) m fitted jointly to all usable keyframes. fix_start / fix_end (3,) m: hard end
    points (projected after every step)."""
    import torch
    S = TUBES[name]['S']
    ks = [k for k in range(cfg['rest_from'], V.n, 5) if obs[k]['ok']]
    O = _torch_obs(V, obs, ks, TUBES[name]['r_range'])
    if dz and fix_start is not None:
        O['dz'] = torch.tensor(depth_offsets(V, obs, fix_start, cfg['r_prox_mm'][name] / 1000, ks)[0] * 1000)
    if front is not None:     # front = (gallbladder rest verts, faces, radius profile (S,) m)
        O['zgb'] = gb_depth(V, front[0], front[1], ks)
        O['r_front'] = torch.tensor(np.asarray(front[2]) * 1000)
    k0 = min(ks, key=lambda k: abs(k - 120))
    X0 = init_curve(V, obs, k0, S) if init is None else np.array(init)
    if fix_start is not None:
        X0 = _blend_start(X0, fix_start)
    if fix_end is not None:
        X0 = _blend_start(X0[::-1], fix_end)[::-1].copy()
    X = torch.tensor(X0 * 1000, dtype=torch.float64).requires_grad_(True)
    fs = None if fix_start is None else torch.tensor(np.asarray(fix_start) * 1000)
    fe = None if fix_end is None else torch.tensor(np.asarray(fix_end) * 1000)
    opt = torch.optim.Adam([X], lr=0.05)
    for it in range(cfg['iters_rest']):
        Xt = X[None].expand(len(ks), S, 3)
        with torch.no_grad():
            q, _ = _project(Xt, O['R'], O['pos'], O['foc'], V.W, V.H)
        vis = _visibility(q, O, V.W, V.H)
        E, q, z, dcs = _data_terms(Xt, O, V.W, V.H, cfg, vis)
        R_, L = _regs(X[None], cfg, None if L_target is None else L_target * 1000)
        loss = E.sum() / O['fw'].sum() + R_.sum() + _front_term(q, z, O, cfg) / len(ks)
        if anchor is not None:
            loss = loss + _anchor(X[None, 0], torch.tensor(anchor[None] * 1000), cfg).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if fs is not None:
                X[0] = fs
            if fe is not None:
                X[-1] = fe
        if it in (0, cfg['iters_rest'] - 1):
            log(f'  rest {name} it {it}: loss {loss.item():.2f} L {L.item():.1f} mm '
                f'reproj {float(dcs.detach().sqrt().mean()):.2f} px')
    return X.detach().numpy() / 1000, ks


def fit_4d(V, obs, name, cfg, rest, log, anchor=None, fix_start=None, share_end=False, free_L0=False, r_curv=None,
           dz=False, front=None):
    """Centreline per frame (n, S, 3) m. fix_start (n, 3) m: hard proximal trajectory (e.g. a gallbladder vertex);
    share_end: one distal point for all frames (fixed in the world, as in the simulation), fitted."""
    import torch
    S = TUBES[name]['S']
    ks = list(range(V.n))
    O = _torch_obs(V, obs, ks, TUBES[name]['r_range'])
    if dz and fix_start is not None:
        O['dz'] = torch.tensor(depth_offsets(V, obs, fix_start, cfg['r_prox_mm'][name] / 1000, ks)[0] * 1000)
    if front is not None:     # front = (gallbladder verts4d, faces, radius profile (S,) m) or a cached z-buffer list
        O['zgb'] = front[3] if len(front) > 3 else gb_depth(V, front[0], front[1], ks)
        O['r_front'] = torch.tensor(np.asarray(front[2]) * 1000)
    Xr = torch.tensor(rest * 1000, dtype=torch.float64)
    L_rest = float((Xr[1:] - Xr[:-1]).norm(dim=-1).sum())
    X0 = np.repeat(rest[None], V.n, 0)
    if fix_start is not None:
        X0 = _blend_start(X0, fix_start[:, None])
    X = torch.tensor(X0 * 1000, dtype=torch.float64).requires_grad_(True)
    fs = None if fix_start is None else torch.tensor(np.asarray(fix_start) * 1000)
    L0 = torch.tensor(L_rest, dtype=torch.float64, requires_grad=free_L0)   # one length for all frames
    opt = torch.optim.Adam([X, L0] if free_L0 else [X], lr=0.05)
    for it in range(cfg['iters_4d']):
        with torch.no_grad():
            q, _ = _project(X, O['R'], O['pos'], O['foc'], V.W, V.H)
        vis = _visibility(q, O, V.W, V.H)
        E, q, z, dcs = _data_terms(X, O, V.W, V.H, cfg, vis)
        R_, L = _regs(X, cfg, L0)
        e_t1 = ((X[1:] - X[:-1]) ** 2).sum((1, 2)).sum()
        e_t2 = ((X[2:] - 2 * X[1:-1] + X[:-2]) ** 2).sum((1, 2)).sum()
        e_rest = ((X - Xr[None]) ** 2).sum((1, 2)).sum()
        loss = (E.sum() + R_.sum() + cfg['w_t1'] * e_t1 + cfg['w_t2'] * e_t2 + cfg['w_rest'] * e_rest
                + _front_term(q, z, O, cfg)) / V.n
        if r_curv is not None:   # curvature barrier: bend radius >= tube radius / cfg['max_kr'] (no self-folding rings)
            seg = (X[:, 1:] - X[:, :-1]).norm(dim=-1)
            kappa = (X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]).norm(dim=-1) / (0.5 * (seg[:, 1:] + seg[:, :-1])) ** 2
            rc = torch.tensor(np.asarray(r_curv)[1:-1] * 1000)
            loss = loss + cfg['w_curv'] * (torch.relu(kappa * rc - cfg['max_kr']) ** 2).sum() / V.n
        if anchor is not None:
            loss = loss + _anchor(X[:, 0], torch.tensor(anchor * 1000), cfg).sum() / V.n
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if fs is not None:
                X[:, 0] = fs
            if share_end:
                X[:, -1] = X[:, -1].mean(0)
        if it in (0, cfg['iters_4d'] // 2, cfg['iters_4d'] - 1):
            log(f'  4d {name} it {it}: loss {loss.item():.2f} L {float(L.mean()):.1f}+-{float(L.std()):.1f} mm '
                f'reproj {float((dcs.detach().sqrt() * O["fw"][:, None]).sum() / O["fw"].sum() / S):.2f} px'
                + (f' L0 {float(L0):.2f} mm' if free_L0 else ''))
    return X.detach().numpy() / 1000, float(L0.detach()) / 1000


def radius_profile(obs, cl4d, V, name, cfg):
    """Radius (S,) m: median over frames of the mask half-width at the skeleton point nearest each curve sample."""
    S = cl4d.shape[1]
    vals = [[] for _ in range(S)]
    for k in range(V.n):
        o = obs[k]
        if not o['ok']:
            continue
        q, _ = V.project(cl4d[k], k)
        d = np.linalg.norm(q[:, None] - o['uv'][None], axis=-1)
        j = d.argmin(1)
        for i in range(S):
            if d[i, j[i]] < 6 and o['in_mask'][j[i]] and o['rad_px'][j[i]] > 0.5:
                vals[i].append(o['rad'][j[i]])
    r = np.array([np.median(v) if len(v) >= 5 else np.nan for v in vals])
    if np.isnan(r).all():
        r[:] = np.mean(TUBES[name]['r_range'])
    idx = np.arange(S)
    r = np.interp(idx, idx[~np.isnan(r)], r[~np.isnan(r)])
    r = np.convolve(np.pad(r, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    # the mask's rounded tips (and the gap to the gallbladder mask) are not a tapering tube: hold the radius constant
    # over the end fractions given in cfg['flat_ends'][name] = (proximal, distal)
    fp, fd = cfg.get('flat_ends', {}).get(name, (0, 0))
    i0, i1 = int(round(fp * (S - 1))), int(round((1 - fd) * (S - 1)))
    r[:i0] = r[i0]
    r[i1 + 1:] = r[i1]
    return np.clip(r, *TUBES[name]['r_range'])


# ----------------------------------------------------------------------------------------------- tube mesh
def tube_faces(S, n=N_RING):
    """Surface of a capped tube: rings i*n + a, cap centres S*n (proximal), S*n + 1 (distal)."""
    F = []
    for i in range(S - 1):
        for a in range(n):
            b = (a + 1) % n
            p, q, r, s = i * n + a, i * n + b, (i + 1) * n + a, (i + 1) * n + b
            F += [[p, r, q], [q, r, s]]
    c0, c1 = S * n, S * n + 1
    for a in range(n):
        b = (a + 1) % n
        F += [[c0, b, a], [c1, (S - 1) * n + a, (S - 1) * n + b]]
    return np.array(F, int)


def tube_tets(S, verts, n=N_RING):
    """Volume of a tube whose vertex list has interior centreline vertices S*n + 2 + (i - 1), i = 1..S-2: each
    segment is a fan of prisms (centre, ring a, ring a+1) split into 3 tets with the same diagonal rule on every
    shared face (conforming); tets oriented to positive volume."""
    cen = lambda i: S * n if i == 0 else (S * n + 1 if i == S - 1 else S * n + 2 + i - 1)
    T = []
    for i in range(S - 1):
        c, c1 = cen(i), cen(i + 1)
        for a in range(n):
            b = (a + 1) % n
            A, B, A1, B1 = i * n + a, i * n + b, (i + 1) * n + a, (i + 1) * n + b
            T += [[c, A, B, B1], [c, A, B1, A1], [c, A1, B1, c1]]
    T = np.array(T, int)
    X = np.asarray(verts)
    v = np.einsum('ij,ij->i', np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), X[T[:, 3]] - X[T[:, 0]])
    T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    return T


def frames_pt(C, n0=None):
    """Parallel-transport normals along a polyline C (S, 3); n0 a preferred first normal."""
    T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    if n0 is None:
        a = np.eye(3)[np.argmin(np.abs(T[0]))]
        n0 = np.cross(T[0], a)
    n = n0 - T[0] * (n0 @ T[0])
    n /= np.linalg.norm(n)
    N = [n]
    for i in range(1, len(C)):
        n = N[-1] - T[i] * (N[-1] @ T[i])
        n /= max(np.linalg.norm(n), 1e-12)
        N.append(n)
    N = np.array(N)
    B = np.cross(T, N)
    return T, N, B


def tube_verts(C, r, n0=None, n=N_RING, interior=False, d_view=None, s=1.0):
    """Rings (S*n), cap centres (2) [, interior centreline vertices (S-2)] of a tube around C with radius r (S,).
    d_view given: ring frame e1 = T x d_view (width direction seen by the scope), e2 = T x e1 (towards the scope), the
    same rule in rest and every frame (no twist between them); s stretches the section along e1 and shrinks it along
    e2 (ellipse s*r x r/s: same area, i.e. a flattened tube seen face-on)."""
    if d_view is None:
        T, N, B = frames_pt(C, n0)
    else:
        T, N = view_frame(C, d_view)
        B = np.cross(T, N)
    ang = 2 * np.pi * np.arange(n) / n
    ring = C[:, None] + r[:, None, None] * (s * np.cos(ang)[None, :, None] * N[:, None]
                                            + np.sin(ang)[None, :, None] / s * B[:, None])
    parts = [ring.reshape(-1, 3), C[[0, -1]]] + ([C[1:-1]] if interior else [])
    return np.concatenate(parts, 0), N[0]


def width_scale(V, obs, cl4d, r, cfg):
    """Per-frame apparent width / model width (n,): observed mask half-width at the skeleton (px -> m at the fitted
    centreline's own depth) over the radius profile; temporally smoothed (Gaussian, sigma cfg['s_sigma'] frames) and
    clipped to cfg['s_range']; frames without observation interpolated."""
    raw = np.full(V.n, np.nan)
    for k in range(V.n):
        o = obs[k]
        if not o['ok']:
            continue
        q, z = V.project(cl4d[k], k)
        d = np.linalg.norm(o['uv'][:, None] - q[None], axis=-1)
        i = d.argmin(1)
        sel = o['in_mask'] & (o['rad_px'] > 1) & (d.min(1) < 4)
        sel[:2] = sel[-2:] = False   # skip the tapered ends
        if sel.sum() >= 5:
            raw[k] = np.median(o['rad_px'][sel] * z[i[sel]] / V.f[k] / r[i[sel]])
    t = np.arange(V.n)
    ok = np.isfinite(raw)
    x = np.interp(t, t[ok], raw[ok])
    g = np.exp(-0.5 * (np.arange(-15, 16) / cfg['s_sigma']) ** 2); g /= g.sum()
    xs = np.convolve(np.pad(x, 15, mode='edge'), g, mode='valid')
    return np.clip(xs, *cfg['s_range']), raw


def view_frame(C, d_view, min_sin=0.35):
    """Unit tangents T and normals N = T x d_view (the width direction seen by the scope) of a polyline C (S, 3).
    Where the curve runs nearly along the line of sight (|T x d_view| < min_sin, e.g. a segment that dives towards
    an anchor in depth) the cross product is unstable: there N is taken from the nearest stable sample and
    re-orthogonalised (no flips -> no twisted rings)."""
    T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    c = np.cross(T, d_view)
    w = np.linalg.norm(c, axis=1)
    ok = w >= min_sin
    if not ok.any():
        ok = w >= w.max() - 1e-9
    idx = np.arange(len(C))
    src = np.array([idx[ok][np.argmin(np.abs(idx[ok] - i))] for i in idx])
    c = c[src]
    N = c - T * (c * T).sum(1, keepdims=True)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return T, N


def fibres(C, r_bundle, d_view, n_f=3, r_min=0.3e-3, spread=None, const_r=False, ramp=0.0, fixed_end=None,
           ramp_start=0.0, ramp_t0=0.0):
    """n_f parallel fibres laid across a fitted bundle (centreline C (S, 3) or (T, S, 3), half-width r_bundle (S,)):
    offsets along b = T x d_view (the band's width direction as seen by the scope); fibre radius r_bundle / n_f."""
    C = np.asarray(C)
    b = np.stack([view_frame(Ck, d_view)[1] for Ck in C.reshape(-1, *C.shape[-2:])]).reshape(C.shape)
    r_f = np.maximum(r_bundle / n_f, r_min)   # adjacent fibres touch (a packed band)
    if const_r:                                # one radius per fibre; the fan still converges where the band narrows
        r_f = np.full_like(r_bundle, max(np.median(r_bundle) / n_f, r_min))
    off = np.maximum(r_bundle - r_f, 0)
    S = len(r_bundle)
    if ramp > 0:     # fibres leave one point (the anchor) and open smoothly over the first `ramp` of the length;
        u = np.clip((np.linspace(0, 1, S) - ramp_t0) / ramp, 0, 1)    # closed along a bridge (t < ramp_t0)       # constant spacing from there to the distal end
        w = ramp_start + (1 - ramp_start) * u * u * (3 - 2 * u)      # monotone: no S-bends
        off = np.full(S, np.median(off[S // 3:])) * w
    s = np.linspace(1, -1, n_f) if n_f > 1 else np.zeros(1)
    sp = 1.0 if spread is None else np.asarray(spread)[:, None, None]   # per-frame fan opening (T,)
    out = [C + sj * sp * off[:, None] * b for sj in s]
    if fixed_end is not None:   # distal points fixed in the world: shift linearly along the fibre
        t = np.linspace(0, 1, S)[:, None]
        out = [F + t * (np.asarray(e) - F[..., -1:, :]) for F, e in zip(out, fixed_end)]
    return out, r_f


# ----------------------------------------------------------------------------------------------- metrics
def _unknown(V, k):
    return cv2.dilate(V.occluders(k).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def signed_dist(P, X, F, n=40000, seed=0):
    """Distance (m) of points P (N, 3) to the surface (X, F) by dense surface sampling, + outside / - inside (normal
    of the nearest sample; faces outward)."""
    from scipy.spatial import cKDTree
    X, F = np.asarray(X, float), np.asarray(F)
    A = np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), n, p=A / A.sum())
    r = rng.random((n, 2))
    r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
    Sm = X[F[fi, 0]] * (1 - r.sum(1))[:, None] + X[F[fi, 1]] * r[:, :1] + X[F[fi, 2]] * r[:, 1:]
    N = np.cross(X[F[fi, 1]] - X[F[fi, 0]], X[F[fi, 2]] - X[F[fi, 0]])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    d, i = cKDTree(Sm).query(np.asarray(P, float))
    return d * np.sign(((P - Sm[i]) * N[i]).sum(-1) + 1e-15)


def frame_metrics(V, verts, faces, mask, k, tol=4):
    """IoU, boundary F (tol px) and depth residual (mm) of one rendered tube against a mask (as quality.py, but for
    this track's own masks); the mesh is cut where the video depth says something is in front (+6 mm)."""
    m, z = quality.render(V, verts, faces, k)
    vis = m & (z < V.depth(k) + 0.006)
    valid = ~_unknown(V, k)
    a, b = vis & valid, mask & valid
    iou = float((a & b).sum() / max((a | b).sum(), 1)) if (a | b).sum() else np.nan
    k3 = np.ones((3, 3), np.uint8)
    bm = (vis.astype(np.uint8) - cv2.erode(vis.astype(np.uint8), k3)).astype(bool) & valid
    br = (mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), k3)).astype(bool) & valid
    if bm.sum() and br.sum():
        dr = cv2.distanceTransform((~br).astype(np.uint8), cv2.DIST_L2, 3)
        dm = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 3)
        p, r = float((dr[bm] <= tol).mean()), float((dm[br] <= tol).mean())
        bf = 2 * p * r / max(p + r, 1e-9)
    else:
        bf = 0.0 if (bm.sum() or br.sum()) else np.nan
    both = m & mask & valid
    dres = float(np.median(np.abs(z[both] - V.depth(k)[both])) * 1000) if both.sum() > 15 else np.nan
    hidden = float(((m & ~vis) & mask & valid).sum() / max((m & mask & valid).sum(), 1))
    # the same IoU with the gallbladder's (clip SAM) pixels outside this mask treated as unknown: the part of a tube
    # that runs over the neck to its attachment is not judged against the video there
    v2 = valid & ~(V.mask('gallbladder')[k] & ~mask)
    a2, b2 = vis & v2, mask & v2
    iou_nogb = float((a2 & b2).sum() / max((a2 | b2).sum(), 1)) if (a2 | b2).sum() else np.nan
    return dict(iou=iou, bf=bf, depth_mm=dres, cut_by_depth=hidden, mask_px=int(b.sum()), mesh_px=int(a.sum()),
                iou_nogb=iou_nogb)


def views3d(V, pieces, path, k_ref=100, size=440):
    """Rest tubes from 3 directions, flat shaded, orthographic: as the scope sees them in frame k_ref, from the
    side (rotated 90 deg about the image vertical) and from above (rotated 90 deg about the image horizontal)."""
    allv = np.concatenate([p[0] for p in pieces], 0)
    c = allv.mean(0)
    span = np.ptp(allv, 0).max() * 1.3
    Rc = V.R[k_ref]                       # rows: camera x right, y down, z forward (world)
    views = [('scope view f%d' % k_ref, Rc[0], -Rc[1], Rc[2]),
             ('side (from image right)', -Rc[2], -Rc[1], -Rc[0]),
             ('from image top', Rc[0], Rc[2], Rc[1])]
    tiles = []
    for name, right, up, fwd in views:
        img = np.full((size, size, 3), 30, np.uint8)
        polys = []
        for Vt, F, col in pieces:
            P = Vt - c
            u = (P @ right / span + 0.5) * size
            v = (0.5 - P @ up / span) * size
            dpt = P @ fwd
            nrm = np.cross(Vt[F[:, 1]] - Vt[F[:, 0]], Vt[F[:, 2]] - Vt[F[:, 0]])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-15)
            shade = 0.35 + 0.65 * np.abs(nrm @ fwd)
            for fi in range(len(F)):
                polys.append((dpt[F[fi]].mean(), np.stack([u[F[fi]], v[F[fi]]], 1), np.array(col) * shade[fi]))
        for _, pts, col in sorted(polys, key=lambda p: -p[0]):
            cv2.fillConvexPoly(img, np.round(pts * 4).astype(np.int32), tuple(float(x) for x in col), cv2.LINE_AA, 2)
        cv2.putText(img, f'{name}  (bar 5 mm)', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        L = 0.005 / span * size
        cv2.line(img, (10, size - 15), (10 + int(L), size - 15), (255, 255, 255), 2)
        tiles.append(img)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def tube_stats(V, cl4d, rest_cl, r, L_rest=None):
    L = np.linalg.norm(np.diff(cl4d, axis=1), axis=2).sum(1)
    L_rest = np.linalg.norm(np.diff(rest_cl, axis=0), axis=1).sum() if L_rest is None else L_rest
    speed = np.linalg.norm(np.diff(cl4d, axis=0), axis=2).max(1) * 1000
    acc = np.linalg.norm(cl4d[2:] - 2 * cl4d[1:-1] + cl4d[:-2], axis=2).max(1) * 1000
    S = cl4d.shape[1]
    smax, s95 = quality.stretch(rest_cl, cl4d, np.array([[i, i + 1] for i in range(S - 1)]))
    return dict(
        length_mm=dict(rest=round(L_rest * 1000, 2), min=round(float(L.min() * 1000), 2),
                       max=round(float(L.max() * 1000), 2), std=round(float(L.std() * 1000), 2),
                       max_strain=round(float(np.abs(L / L_rest - 1).max()), 4),
                       per_keyframe=[round(float(L[k] * 1000), 2) for k in V.keyframes]),
        segment_stretch=dict(max=round(float(smax.max()), 3), p95_mean=round(float(s95.mean()), 3)),
        radius_mm=dict(profile=[round(float(x * 1000), 2) for x in r], min=round(float(r.min() * 1000), 2),
                       max=round(float(r.max() * 1000), 2), median=round(float(np.median(r) * 1000), 2)),
        max_speed_mm_per_frame=round(float(speed.max()), 3),
        p95_speed_mm_per_frame=round(float(np.percentile(speed, 95)), 3),
        max_accel_mm_per_frame2=round(float(acc.max()), 3))


def silhouette_block(V, v4, F, mask, ks, rest=None):
    per = {k: frame_metrics(V, v4, F, mask[k], k) for k in ks}
    out = dict(keyframes=list(ks),
               iou=[per[k]['iou'] for k in ks], boundary_f=[per[k]['bf'] for k in ks],
               depth_mm=[per[k]['depth_mm'] for k in ks], cut_by_depth=[per[k]['cut_by_depth'] for k in ks],
               iou_summary=quality.summarize([per[k]['iou'] for k in ks]),
               boundary_f_summary=quality.summarize([per[k]['bf'] for k in ks]),
               depth_mm_summary=quality.summarize([per[k]['depth_mm'] for k in ks]),
               iou_excl_gallbladder_summary=quality.summarize([per[k]['iou_nogb'] for k in ks]))
    if rest is not None:
        out['static_rest_iou_summary'] = quality.summarize([frame_metrics(V, rest, F, mask[k], k)['iou'] for k in ks])
    out['bands'] = {f'{a}-{b}': dict(iou=quality.summarize([per[k]['iou'] for k in ks if a <= k < b]),
                                     iou_excl_gallbladder=quality.summarize([per[k]['iou_nogb'] for k in ks if a <= k < b]),
                                     boundary_f=quality.summarize([per[k]['bf'] for k in ks if a <= k < b]),
                                     depth_mm=quality.summarize([per[k]['depth_mm'] for k in ks if a <= k < b]))
                    for a, b in ((0, 62), (62, 130), (130, 251))}
    return out


def diag(V, obs, cfg, fit, log, path):
    """Variants of the 4D fit: without the length term (what length do the observations alone imply over time?) and
    without the video depth (does the camera motion alone, i.e. multi-view parallax + temporal smoothness, put the
    tubes at the depth the depth maps say?)."""
    out = {}
    for tag, upd in [('no_length', dict(w_len=0.0, w_rest=0.0)), ('no_depth', dict(w_depth=0.0))]:
        c = dict(cfg, **upd)
        out[tag] = {}
        for nm in TUBES:
            anc = fit['duct'][1][:, 0] if nm == 'strands' else None
            cl, _ = fit_4d(V, obs[nm], nm, c, fit[nm][0], log, anc)
            L = np.linalg.norm(np.diff(cl, axis=1), axis=2).sum(1) * 1000
            dz = []
            for k in V.keyframes:
                _, z0 = V.project(fit[nm][1][k], k)
                _, z1 = V.project(cl[k], k)
                dz.append(float(np.median(z1 - z0)) * 1000)
            out[tag][nm] = dict(length_mm_per_keyframe=[round(float(L[k]), 2) for k in V.keyframes],
                                length_mm=quality.summarize(L),
                                depth_shift_vs_main_mm_per_keyframe=[round(x, 2) for x in dz],
                                mean_abs_shift_vs_main_mm=round(float(np.linalg.norm(cl - fit[nm][1], axis=2).mean() * 1000), 3))
            log(f'diag {tag} {nm}: L {out[tag][nm]["length_mm"]} depth shift {np.round(dz, 1).tolist()}')
    json.dump(out, open(path, 'w'), indent=1)


# ----------------------------------------------------------------------------------------------- for the integrator
def cable_spec(version, n_seg=12):
    """Per tube of a saved version: rest centreline resampled to n_seg + 1 nodes, radius per node (m), the attachment
    of each end and the suggested simulation element (see NOTES.md). Keys:
      name, nodes (n, 3) rest, radius (n,), length_m (rest), nodes4d (251, n, 3), attach (proximal, distal), kind;
      from v09 also: least_stretched_frame, nodes_least_stretched (n, 3) (= nodes4d of the frame with the shortest
      centreline), length4d_m (251,), anchor (gallbladder version, vertex index the proximal node sits on, or None).
    Example:
        for c in cable_spec('v09'): print(c['name'], c['kind'], c['length_m'], c['attach'], c['anchor'])"""
    D = np.load(OUT / version / 'model.npz')
    anc = None
    if 'anchor_gb' in D.files and int(D['anchor_gb'][1]) >= 0:
        off = D['anchor_offset_rest'] if 'anchor_offset_rest' in D.files else np.zeros(3)
        anc = dict(gallbladder=str(D['anchor_gb'][0]), vertex=int(D['anchor_gb'][1]), offset_rest_m=np.asarray(off),
                   on_vertex=bool(np.linalg.norm(off) < 1e-6))

    def res(C, t, s):
        return np.stack([np.interp(t, s, C[:, c]) for c in range(3)], 1)

    out = []
    for name in D['tube_names']:
        C, r = D[f'{name}_rest_centerline'], D[f'{name}_radius']
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        t = np.linspace(0, s[-1], n_seg + 1)
        n4 = []
        for Ck in D[f'{name}_centerline4d']:      # resample every frame by its own arc length (ends stay ends)
            sk = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Ck, axis=0), axis=1))])
            n4.append(res(Ck, np.linspace(0, sk[-1], n_seg + 1), sk))
        n4 = np.stack(n4)
        L4 = np.linalg.norm(np.diff(n4, axis=1), axis=2).sum(1)
        k_ls = int(np.argmin(L4))
        a_t = anc
        if anc is not None and 'anchor_rest_point' in D.files:     # this tube's own rest offset from the vertex
            off = C[0] - D['anchor_rest_point']
            a_t = dict(anc, offset_rest_m=off, on_vertex=bool(np.linalg.norm(off) < 1e-6))
        out.append(dict(name=str(name), nodes=res(C, t, s), radius=np.interp(t, s, r), length_m=float(s[-1]),
                        nodes4d=n4, attach=TUBES['duct' if name == 'duct' else 'strands']['attach'],
                        kind='cable (soft tube, bending + stretch)' if name == 'duct' else
                             'string (thin fibre, tension only; no fibre-fibre collision)',
                        least_stretched_frame=k_ls, nodes_least_stretched=n4[k_ls], length4d_m=L4, anchor=a_t))
    return out


def closest_on_tris(p, X, F):
    """Closest point to p (3,) on the triangles F (M, 3) of vertices X (Ericson); returns point, face index, distance."""
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp, cp = p - b, p - c
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = 1.0 / np.maximum(va + vb + vc, 1e-30)
    q = a + ab * (vb * den)[:, None] + ac * (vc * den)[:, None]
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    q[m] = (b + (c - b) * ((d4 - d3) / np.maximum(d4 - d3 + d5 - d6, 1e-30))[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    q[m] = (a + ac * (d2 / np.maximum(d2 - d6, 1e-30))[:, None])[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    q[m] = (a + ab * (d1 / np.maximum(d1 - d3, 1e-30))[:, None])[m]
    m = (d6 >= 0) & (d5 <= d6); q[m] = c[m]
    m = (d3 >= 0) & (d4 <= d3); q[m] = b[m]
    m = (d1 <= 0) & (d2 <= 0); q[m] = a[m]
    d = np.linalg.norm(q - p, axis=1)
    i = int(np.argmin(d))
    return q[i], i, float(d[i])


def bridge_start(C, A, frac=0.2, max_angle=60.0):
    """Replace the first `frac` of centreline C (S, 3) by a cubic Hermite bridge from the anchor A (3,) to the path
    point P_b = C(s_b), leaving A along the chord A -> P_b and joining the path tangentially; s_b is moved further
    along the path while the chord and the path tangent at s_b differ by more than max_angle (no hook when the anchor
    lies ahead of the path start). Result resampled to S points equally spaced in arc length."""
    C = np.asarray(C, float)
    S = len(C)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    T = np.gradient(C, s, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    ib = max(int(round(frac * (S - 1))), 1)
    while ib < S - 3:
        ch = C[ib] - A
        if np.linalg.norm(ch) < 1e-9 or (ch @ T[ib]) / np.linalg.norm(ch) >= np.cos(np.radians(max_angle)):
            break
        ib += 1
    P1, chord = C[ib], C[ib] - A
    L = np.linalg.norm(chord)
    m0, m1 = chord, T[ib] * L
    u = np.linspace(0, 1, 40)[:, None]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    B = h00 * A + h10 * m0 + h01 * P1 + h11 * m1
    P = np.concatenate([B[:-1], C[ib:]], 0)
    sp = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0, sp[-1], S)
    return np.stack([np.interp(t, sp, P[:, c]) for c in range(3)], 1)


def _bow(C, d_view):
    """Largest deviation (m) of a centreline from its end-to-end chord along the viewing direction (a dip in depth)."""
    t = np.linspace(0, 1, len(C))[:, None]
    dev = C - (C[0] + t * (C[-1] - C[0]))
    return np.abs(dev @ d_view).max()


def end_motion_test(nodes, nodes4d, max_extra=3):
    """Can a cable glued at node 0 and driven at node n-1 reproduce the 4D? Motion explained (as scene4d's metric:
    1 - mean |pred - rec| / mean |rest - rec| over nodes and frames) by
      lin:   rest + displacements of the two end nodes interpolated linearly along the arc (a stretching rope),
      chord: rest moved rigidly with the end chord (minimal rotation about node 0 + end translation),
      drive: lin with extra driven interior nodes chosen greedily (piecewise-linear between driven nodes).
    Returns dict with the scores and the greedy node list."""
    X0, X = np.asarray(nodes, float), np.asarray(nodes4d, float)
    n = len(X0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(X0, axis=0), axis=1))])
    t = s / s[-1]
    D = X - X0[None]
    e0 = np.linalg.norm(D, axis=2).mean()

    def interp(drv):
        drv = sorted(drv)
        P = np.stack([np.stack([np.interp(t, t[drv], D[k, drv, c]) for c in range(3)], 1) for k in range(len(X))])
        return P

    def score(P):
        return round(float(1 - np.linalg.norm(D - P, axis=2).mean() / max(e0, 1e-12)), 3)

    out = dict(static_err_mm=round(float(e0 * 1000), 3), lin=score(interp([0, n - 1])))
    Dp = D.copy()                      # only the proximal end moves (distal fixed): linear fade to the distal end
    out['prox_only'] = score(D[:, :1] * (1 - t)[None, :, None])
    out['dist_only'] = score(D[:, -1:] * t[None, :, None])
    # rigid with the end chord
    c0 = X0[-1] - X0[0]
    P = np.zeros_like(D)
    for k in range(len(X)):
        c1 = X[k, -1] - X[k, 0]
        a, b = c0 / np.linalg.norm(c0), c1 / np.linalg.norm(c1)
        v, cth = np.cross(a, b), float(a @ b)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx / (1 + cth) if cth > -0.999 else -np.eye(3)
        P[k] = (X0 - X0[0]) @ Rm.T + X[k, 0] - X0
    out['chord'] = score(P)
    drv, gains = [0, n - 1], []
    for _ in range(max_extra):
        best = max((i for i in range(1, n - 1) if i not in drv), key=lambda i: score(interp(drv + [i])))
        drv.append(best)
        gains.append((best, score(interp(drv))))
    out['drive'] = gains
    out['per_node_residual_mm_lin'] = [round(float(x * 1000), 2) for x in np.linalg.norm(D - interp([0, n - 1]), axis=2).mean(0)]
    return out


def cable_equilibrium_test(nodes, nodes4d, driven=(0, -1), k_skip=1.0):
    """Quasi-static version of the integrator's cable (chain springs + every-second-node springs, rest lengths from
    `nodes`, no gravity / inertia): per frame the free nodes are put at the spring-energy minimum with the driven nodes
    on their 4D positions (warm start from the previous frame). Returns motion explained vs the 4D (scene4d's metric),
    the per-frame mean node error (mm) and the equilibrium nodes."""
    from scipy.optimize import least_squares
    X0, X = np.asarray(nodes, float), np.asarray(nodes4d, float)
    n = len(X0)
    drv = sorted({d % n for d in driven})
    free = [i for i in range(n) if i not in drv]
    E1 = np.array([[i, i + 1] for i in range(n - 1)])
    E2 = np.array([[i, i + 2] for i in range(n - 2)])
    l1 = np.linalg.norm(X0[E1[:, 0]] - X0[E1[:, 1]], axis=1)
    l2 = np.linalg.norm(X0[E2[:, 0]] - X0[E2[:, 1]], axis=1)
    out, cur = [], X0.copy()
    for k in range(len(X)):
        cur[drv] = X[k, drv]

        prev = cur[free].ravel().copy()

        def res(z):            # springs + a small pull to the previous frame (damping; picks one of equal minima)
            Y = cur.copy()
            Y[free] = z.reshape(-1, 3)
            r1 = np.linalg.norm(Y[E1[:, 0]] - Y[E1[:, 1]], axis=1) - l1
            r2 = np.linalg.norm(Y[E2[:, 0]] - Y[E2[:, 1]], axis=1) - l2
            return np.concatenate([r1, np.sqrt(k_skip) * r2, 0.03 * (z - prev)])
        sol = least_squares(res, prev, method='lm', xtol=1e-12, ftol=1e-12)
        cur[free] = sol.x.reshape(-1, 3)
        out.append(cur.copy())
    Y = np.stack(out)
    e = np.linalg.norm(Y - X, axis=2).mean(1)
    e0 = np.linalg.norm(X0[None] - X, axis=2).mean(1)
    L = np.linalg.norm(np.diff(Y, axis=1), axis=2).sum(1) / l1.sum()
    return dict(motion_explained=round(float(1 - e.mean() / e0.mean()), 3), err_mm=round(float(e.mean() * 1000), 3),
                static_err_mm=round(float(e0.mean() * 1000), 3), length_ratio=[round(float(L.min()), 3), round(float(L.max()), 3)],
                driven=[int(d) for d in drv]), Y


# ----------------------------------------------------------------------------------------------- main
def main(argv):
    import torch
    torch.set_num_threads(4)
    ver = argv[0]
    vdir = OUT / ver
    (vdir / 'work').mkdir(parents=True, exist_ok=True)
    logf = open(vdir / 'work' / 'fit.log', 'w')

    def log(s):
        print(s); logf.write(s + '\n'); logf.flush()

    cfg = dict(CFG)
    for a in argv[1:]:          # --cfg key=python-literal overrides, e.g. --cfg "flat_ends={'duct': (0.15, 0)}"
        if a.startswith('--cfg='):
            import ast
            key, val = a[len('--cfg='):].split('=', 1)
            cfg[key] = ast.literal_eval(val)
    t0 = time.time()
    V = views.load('chole_a', 'sift', cams=str(ROOT / cfg['cams']) if cfg.get('cams') else None)
    log(f'cameras: {V.cams}')
    src = vdir / 'masks.npz'
    if not src.exists():   # reuse the newest earlier version's SAM masks unless asked to resegment
        prev = sorted(p for p in OUT.glob('v*/masks.npz') if p.parent.name < ver)
        if prev and '--resegment' not in argv:
            import shutil
            shutil.copy(prev[-1], src)
    M = make_masks(V, src, resegment='--resegment' in argv)
    ks = V.keyframes

    obs = {}
    obs['duct'] = observe(V, M, 'duct', cfg)
    junction = [o['uv'][0] if o['ok'] and o['junction'] is not None else None for o in obs['duct']]
    obs['strands'] = observe(V, M, 'strands', cfg, junction=junction)
    for nm in obs:
        log(f'{nm}: observed in {sum(o["ok"] for o in obs[nm])}/{V.n} frames, junction in '
            f'{sum(o["ok"] and o["junction"] is not None for o in obs[nm])}')

    # ---- fits: duct, then the strand bundle (proximal end pulled to the duct's proximal end) = v08 ("free")
    fit = {}
    for nm in TUBES:
        anc_rest = fit['duct'][0][0] if nm == 'strands' else None
        anc_4d = fit['duct'][1][:, 0] if nm == 'strands' else None
        rest_cl, ks_rest = fit_rest(V, obs[nm], nm, cfg, log, anc_rest)
        cl4d, L_rest = fit_4d(V, obs[nm], nm, cfg, rest_cl, log, anc_4d)
        r = radius_profile(obs[nm], cl4d, V, nm, cfg)
        fit[nm] = (rest_cl, cl4d, r, L_rest, len(ks_rest))
    free = dict(fit)

    # ---- v09: the proximal ends ride on one gallbladder-model vertex (rest and every frame), the distal ends are one
    # fixed world point (as the simulation treats them); the curve in between is refitted to the video
    anchor = None
    if cfg.get('gb_version'):
        G = np.load(OUT.parent / 'gallbladder' / cfg['gb_version'] / 'model.npz')
        cand = np.asarray(G['attach_idx'])[np.asarray(G['attach_to']) == 'ducts']
        if not len(cand):
            cand = np.unique(G['faces'])
        f0 = cfg['anchor_from']
        md = np.linalg.norm(G['verts4d'][f0:, cand] - free['duct'][1][f0:, None, 0], axis=2).mean(0)
        j = int(cand[np.argmin(md)])
        anchor = dict(version=cfg['gb_version'], vertex=j, rest=G['rest_verts'][j], traj=G['verts4d'][:, j],
                      G=G, mean_dist_to_free_junction_mm=round(float(md.min() * 1000), 2))
        log(f'anchor: gallbladder {cfg["gb_version"]} vertex {j} (of {len(cand)} marked "ducts"), mean distance to the '
            f'observed junction (frames >= {f0}) {md.min() * 1000:.1f} mm')
        if cfg['anchor_mode'] == 'bridge':
            # v11: proximal ends ON the gallbladder model (one of its 'ducts' vertices, every frame); the first
            # bridge_frac of each centreline is replaced by a smooth bridge from that vertex to the observed path
            Gv, Gr = G['verts4d'], G['rest_verts']
            mdj = np.linalg.norm(Gv[:, cand] - free['duct'][1][:, None, 0], axis=2).mean(0)
            jb = int(cand[np.argmin(mdj)])
            anchor = dict(version=cfg['gb_version'], vertex=jb, rest=Gr[jb], traj=Gv[:, jb], G=G, mode='bridge',
                          offset_rest=np.zeros(3), mean_dist_to_free_junction_mm=round(float(mdj.min() * 1000), 2))
            log(f'bridge mode: gallbladder {cfg["gb_version"]} vertex {jb} (of {len(cand)} "ducts"), mean distance to the '
                f'observed junction {mdj.min() * 1000:.1f} mm, bridge over the first {cfg["bridge_frac"]:.0%}')
            for nm in TUBES:
                rb_ = bridge_start(free[nm][0], Gr[jb], cfg['bridge_frac'])
                cb_ = np.stack([bridge_start(free[nm][1][k], Gv[k, jb], cfg['bridge_frac']) for k in range(V.n)])
                fit[nm] = (rb_, cb_, free[nm][2], float(np.linalg.norm(np.diff(rb_, axis=0), axis=1).sum()), free[nm][4])
        if cfg['anchor_mode'] in ('patch', 'patch_fit'):
            # v11: proximal ends ON the gallbladder model's duct-attachment patch (its triangles with >= 2 vertices
            # marked 'ducts'): per frame the patch point closest to the observed junction, smoothed over time and
            # put back onto the patch; the first bridge_frac of each centreline bridges from it to the observed path
            Gv, Gr, GF = G['verts4d'], G['rest_verts'], np.asarray(G['faces'])
            Fp = GF[np.isin(GF, cand).sum(1) >= 2]
            J = free['duct'][1][:, 0]
            Q = np.stack([closest_on_tris(J[k], Gv[k], Fp)[0] for k in range(V.n)])
            g = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2); g /= g.sum()
            Qs = np.stack([np.convolve(np.pad(Q[:, c], 9, mode='edge'), g, mode='valid') for c in range(3)], 1)
            res_ = [closest_on_tris(Qs[k], Gv[k], Fp) for k in range(V.n)]
            Q = np.stack([r_[0] for r_ in res_])
            qr, fr_i, _ = closest_on_tris(free['duct'][0][0], Gr, Fp)
            gap = np.linalg.norm(Q - J, axis=1)
            anchor = dict(version=cfg['gb_version'], vertex=int(cand[np.argmin(np.linalg.norm(Gr[cand] - qr, axis=1))]),
                          rest=qr, traj=Q, G=G, mode='patch', offset_rest=np.zeros(3),
                          face4d=np.array([int(np.nonzero((GF == Fp[r_[1]]).all(1))[0][0]) for r_ in res_]),
                          mean_dist_to_free_junction_mm=round(float(gap.mean() * 1000), 2))
            anchor['offset_rest'] = qr - Gr[anchor['vertex']]
            log(f'patch mode: {len(Fp)} gallbladder {cfg["gb_version"]} faces, gap observed junction -> patch point '
                f'{quality.summarize(gap * 1000)} mm, bridge over the first {cfg["bridge_frac"]:.0%}')
            for nm in (TUBES if cfg['anchor_mode'] == 'patch' else ()):
                cb_ = np.stack([bridge_start(free[nm][1][k], Q[k], cfg['bridge_frac']) for k in range(V.n)])
                if cfg['rest_ref'] == 'median_frame':
                    # rest = the 4D shape of the frame (>= 62, after the pan) whose length is the median, moved so its
                    # start sits on the patch point of the gallbladder's rest shape (no slack / pre-stretch built in)
                    L4 = np.linalg.norm(np.diff(cb_, axis=1), axis=2).sum(1)
                    kk = 62 + int(np.argmin(np.abs(L4[62:] - np.median(L4[62:]))))
                    rb_ = cb_[kk] - Q[kk] + qr
                    log(f'  {nm}: rest = frame {kk} shape (length {L4[kk] * 1000:.2f} mm) moved onto the rest patch point')
                else:
                    rb_ = bridge_start(free[nm][0], qr, cfg['bridge_frac'])
                fit[nm] = (rb_, cb_, free[nm][2], float(np.linalg.norm(np.diff(rb_, axis=0), axis=1).sum()), free[nm][4])
        if cfg['anchor_mode'] == 'junction':
            # v10: the proximal ends stay where the video shows the duct leaving the neck (also in depth); the glue to
            # the gallbladder is the v10 vertex nearest to the rest proximal end, with the rest offset kept
            p0 = free['duct'][0][0]
            jg = int(np.argmin(np.linalg.norm(G['rest_verts'] - p0, axis=1)))
            anchor = dict(version=cfg['gb_version'], vertex=jg, rest=G['rest_verts'][jg], traj=G['verts4d'][:, jg],
                          G=G, mode='junction', offset_rest=p0 - G['rest_verts'][jg],
                          mean_dist_to_free_junction_mm=round(float(md.min() * 1000), 2))
            drift = np.linalg.norm(free['duct'][1][:, 0] - (G['verts4d'][:, jg] + anchor['offset_rest']), axis=1)
            anchor['glue_drift_mm'] = quality.summarize(drift * 1000)
            log(f'junction mode: glue vertex {jg}, rest offset {np.linalg.norm(anchor["offset_rest"]) * 1000:.1f} mm, '
                f'drift of the observed junction from vertex+offset: {anchor["glue_drift_mm"]}')
        fr4 = fr_rest = None
        if cfg['gb_front']:
            t1 = time.time()
            z4 = gb_depth(V, G['verts4d'], G['faces'], range(V.n))
            log(f'gallbladder z-buffers: {time.time() - t1:.0f} s')
        for nm in (TUBES if cfg['anchor_mode'] in ('gb_vertex', 'patch_fit') else ()):
            bl = cfg['depth_blend']
            if cfg['gb_front']:
                rf = free[nm][2] * (1.0 if nm == 'duct' else 0.5)    # strands: a fibre's radius ~ bundle / 3
                fr_rest = (G['rest_verts'], G['faces'], rf)
                fr4 = (G['verts4d'], G['faces'], rf, z4)
            r1, ks_rest = fit_rest(V, obs[nm], nm, cfg, log, fix_start=anchor['rest'], init=free[nm][0], dz=bl,
                                   front=fr_rest)
            fixed = cfg['distal_fixed']
            r_c = free[nm][2] * (1.25 if nm == 'duct' else 1.0)      # duct: allow for the widened elliptical section
            cl4d, L0 = fit_4d(V, obs[nm], nm, cfg, r1, log, fix_start=anchor['traj'], share_end=fixed, free_L0=True,
                              r_curv=r_c, dz=bl, front=fr4)
            # rest: same anchor, the 4D's common length (and the 4D's fixed distal point if the distal end is fixed)
            rest_cl, _ = fit_rest(V, obs[nm], nm, cfg, log, fix_start=anchor['rest'],
                                  fix_end=cl4d[0, -1] if fixed else None, init=r1, L_target=L0, dz=bl,
                                  front=fr_rest)
            L_rest = float(np.linalg.norm(np.diff(rest_cl, axis=0), axis=1).sum())
            log(f'  {nm}: 4D common length {L0 * 1000:.2f} mm, rest length {L_rest * 1000:.2f} mm')
            fit[nm] = (rest_cl, cl4d, free[nm][2], L_rest, len(ks_rest))

    if '--diag' in argv:
        diag(V, obs, cfg, fit, log, vdir / 'work' / 'diag.json')

    # ---- output tubes: the duct (with interior centreline vertices -> tets) + the fibres of the strand bundle
    d_view = V.R[:, 2].mean(0); d_view /= np.linalg.norm(d_view)
    s_duct, s_duct_raw = width_scale(V, obs['duct'], fit['duct'][1], fit['duct'][2], cfg)
    s_str, s_str_raw = width_scale(V, obs['strands'], fit['strands'][1], fit['strands'][2], cfg)
    if not cfg['width_scale']:
        s_duct[:], s_str[:] = 1.0, 1.0
    pieces = [dict(name='duct', group='duct', rest=fit['duct'][0], cl4d=fit['duct'][1], r=fit['duct'][2], interior=True,
                   s=s_duct)]
    rb, cb, r_b = fit['strands'][0], fit['strands'][1], fit['strands'][2]
    rt0 = cfg['bridge_frac'] if cfg.get('anchor_mode') == 'patch' else 0.0
    rr = cfg['fibre_ramp'] - rt0 * 0.5 if rt0 else cfg['fibre_ramp']
    f_rest, r_f = fibres(rb, r_b, d_view, cfg['n_fibres'], const_r=cfg['fibre_const_r'], ramp=rr,
                         ramp_start=cfg['fibre_ramp_start'], ramp_t0=rt0)
    f_4d, _ = fibres(cb, r_b, d_view, cfg['n_fibres'], spread=s_str, const_r=cfg['fibre_const_r'],
                     ramp=rr, ramp_start=cfg['fibre_ramp_start'], ramp_t0=rt0, fixed_end=[f[-1] for f in f_rest] if (anchor is not None and cfg['distal_fixed']
                                                                       and cfg['anchor_mode'] == 'gb_vertex') else None)
    # order fibres from image-top to image-bottom in frame 100
    vy = [V.project(fr[100], 100)[0][:, 1].mean() for fr in f_4d]
    for j, i in enumerate(np.argsort(vy)):
        pieces.append(dict(name=f'strand_{j + 1}', group='strands', rest=f_rest[i], cl4d=f_4d[i], r=r_f, interior=False,
                           s=np.ones(V.n)))

    model, Q = {}, dict(config=cfg, groups={}, tubes={})
    rest_all, v4_all, faces_all, tets_all, att_idx, att_to, tube_id = [], [], [], [], [], [], []
    off = 0
    for ti, P in enumerate(pieces):
        S = len(P['rest'])
        F = tube_faces(S)
        rv, n0 = tube_verts(P['rest'], P['r'], interior=P['interior'], d_view=d_view)
        v4 = np.stack([tube_verts(P['cl4d'][k], P['r'], interior=P['interior'], d_view=d_view, s=P['s'][k])[0]
                       for k in range(V.n)])
        if P['interior']:
            tets_all.append(tube_tets(S, rv) + off)
        P.update(F=F, rv=rv, v4=v4, off=off)
        model[f'{P["name"]}_rest_centerline'] = P['rest']
        model[f'{P["name"]}_centerline4d'] = P['cl4d'].astype(np.float32)
        model[f'{P["name"]}_radius'] = P['r']
        rest_all.append(rv); v4_all.append(v4); faces_all.append(F + off)
        prox = np.r_[np.arange(N_RING), S * N_RING]
        dist = np.r_[(S - 1) * N_RING + np.arange(N_RING), S * N_RING + 1]
        a0, a1 = TUBES[P['group']]['attach']
        att_idx += list(prox + off) + list(dist + off)
        att_to += [a0] * len(prox) + [a1] * len(dist)
        tube_id += [ti] * len(rv)
        off += len(rv)
        Q['tubes'][P['name']] = dict(tube_stats(V, P['cl4d'], P['rest'], P['r']), mesh_health=quality.mesh_health(rv, F))
    model['strands_bundle_rest_centerline'] = rb
    model['strands_bundle_centerline4d'] = cb.astype(np.float32)
    model['strands_bundle_halfwidth'] = r_b
    model['duct_section_scale4d'] = s_duct
    model['strands_spread4d'] = s_str
    Q['width_scale'] = dict(
        duct=dict(raw=quality.summarize(s_duct_raw), used=quality.summarize(s_duct),
                  per_keyframe=[round(float(s_duct[k]), 3) for k in ks]),
        strands=dict(raw=quality.summarize(s_str_raw), used=quality.summarize(s_str),
                     per_keyframe=[round(float(s_str[k]), 3) for k in ks]))

    # ---- per group: silhouettes vs this track's masks
    layers = []
    for g in TUBES:
        G = [P for P in pieces if P['group'] == g]
        v4 = np.concatenate([P['v4'] for P in G], 1)
        rv = np.concatenate([P['rv'] for P in G], 0)
        F = np.concatenate([P['F'] + (P['off'] - G[0]['off']) for P in G], 0)
        layers.append((g, v4, F, TUBES[g]['rgb']))
        blk = silhouette_block(V, v4, F, M[g], ks, rest=rv)
        reproj = []
        cl = fit[g][1]
        for k in range(V.n):
            o = obs[g][k]
            if o['ok']:
                q, _ = V.project(cl[k], k)
                reproj.append(float(np.linalg.norm(o['uv'][:, None] - q[None], axis=-1).min(1).mean()))
        blk.update(centerline_reproj_px=quality.summarize(reproj), observed_frames=int(sum(o['ok'] for o in obs[g])),
                   rest_fit_frames=fit[g][4], fitted=dict(tube_stats(V, fit[g][1], fit[g][0], fit[g][2], fit[g][3])))
        Q['groups'][g] = blk
        log(f'{g}: IoU {blk["iou_summary"]["mean"]} BF {blk["boundary_f_summary"]["mean"]} depth '
            f'{blk["depth_mm_summary"].get("median")} mm static-rest IoU {blk["static_rest_iou_summary"]["mean"]} '
            f'reproj {blk["centerline_reproj_px"]["mean"]} px L {blk["fitted"]["length_mm"]["rest"]} mm '
            f'(std {blk["fitted"]["length_mm"]["std"]}) r {blk["fitted"]["radius_mm"]["median"]} mm')
    # the bundle envelope (one tube of the band's half-width) as a diagnostic
    Fb = tube_faces(len(rb))
    v4b = np.stack([tube_verts(cb[k], r_b)[0] for k in range(V.n)])
    Q['strands_envelope'] = {k: v for k, v in silhouette_block(V, v4b, Fb, M['strands'], ks).items()
                             if k.endswith('summary')}

    # ---- attachment geometry
    d_c = fit['duct'][1]
    Q['strand_to_duct_proximal_mm'] = quality.summarize(np.linalg.norm(cb[:, 0] - d_c[:, 0], axis=1) * 1000)
    gbd = []
    for k in ks:
        q, _ = V.project(d_c[k, :1], k)
        dt = cv2.distanceTransform((~V.mask('gallbladder')[k]).astype(np.uint8), cv2.DIST_L2, 5)
        x, y = np.clip(np.round(q[0]).astype(int), 0, [V.W - 1, V.H - 1])
        gbd.append(float(dt[y, x]))
    Q['duct_proximal_to_gallbladder_mask_px'] = quality.summarize(gbd)
    # clearance (m -> mm) between the duct and the fibres and among the fibres, away from the shared proximal end:
    # min over sample pairs of centre distance - (r_i + r_j); < 0 = interpenetration
    fib = [P for P in pieces if P['group'] == 'strands']
    cd, cf = [], []
    for k in range(0, V.n, 5):
        Dk = pieces[0]['cl4d'][k][3:]
        rd = pieces[0]['r'][3:] * pieces[0]['s'][k]
        for P in fib:
            dd = np.linalg.norm(Dk[:, None] - P['cl4d'][k][None, 3:], axis=-1) - rd[:, None] - P['r'][None, 3:]
            cd.append(dd.min())
        for i in range(len(fib)):     # fibres converge at both ends by construction: middle samples only
            for j in range(i + 1, len(fib)):
                A, B = fib[i]['cl4d'][k][3:-3], fib[j]['cl4d'][k][3:-3]
                dd = np.linalg.norm(A - B, axis=-1) - fib[i]['r'][3:-3] - fib[j]['r'][3:-3]
                cf.append(dd.min())
    Q['clearance_mm'] = dict(duct_to_fibres=quality.summarize(np.array(cd) * 1000),
                             fibre_to_fibre=quality.summarize(np.array(cf) * 1000))
    # cross-track check (read-only): proximal ends -> gallbladder model surface (signed: + outside), before (the free
    # v08-style fit) and after (anchored), rest and per keyframe; length / end-to-end range over time
    gbs = sorted((OUT.parent / 'gallbladder').glob('v*/model.npz'))
    Gp = OUT.parent / 'gallbladder' / cfg['gb_version'] / 'model.npz' if cfg.get('gb_version') else (gbs[-1] if gbs else None)
    if Gp is not None and Gp.exists():
        G = np.load(Gp)
        chk = dict(model=str(Gp.relative_to(ROOT)), anchor_vertex=None if anchor is None else anchor['vertex'],
                   anchor_mode=cfg.get('anchor_mode'),
                   glue_offset_rest_mm=None if anchor is None else round(float(np.linalg.norm(anchor.get('offset_rest', np.zeros(3))) * 1000), 2),
                   glue_drift_mm=None if anchor is None else anchor.get('glue_drift_mm'),
                   anchor_mean_dist_to_observed_junction_mm=None if anchor is None else anchor['mean_dist_to_free_junction_mm'])
        for tag, curves in [('before', {'duct': free['duct'], 'strands': free['strands']}),
                            ('after', {P['name']: (P['rest'], P['cl4d']) for P in pieces})]:
            chk[tag] = {}
            for nm, c in curves.items():
                d = [float(signed_dist(c[1][k, :1], G['verts4d'][k], G['faces'])[0]) * 1000 for k in ks]
                chk[tag][nm] = dict(per_keyframe=[round(x, 2) for x in d], abs=quality.summarize(np.abs(d)),
                                    rest=round(float(signed_dist(c[0][:1], G['rest_verts'], G['faces'])[0]) * 1000, 2))
        Q['proximal_to_gallbladder_surface_mm'] = chk
        # overlap with the gallbladder model: tube pixels hidden behind it in a joint render (the video shows them),
        # and centreline samples (after the anchor) inside it in 3D
        ov = {}
        for g in TUBES:
            Gp_ = [P for P in pieces if P['group'] == g]
            v4 = np.concatenate([P['v4'] for P in Gp_], 1)
            F_ = np.concatenate([P['F'] + (P['off'] - Gp_[0]['off']) for P in Gp_], 0)
            hid, ins = [], []
            for k in ks:
                mg, zg = quality.render(V, G['verts4d'], G['faces'], k)
                m, z = quality.render(V, v4, F_, k)
                hid.append(float((m & mg & (zg < z - 0.0005)).sum() / max(m.sum(), 1)))
                d = signed_dist(Gp_[0]['cl4d'][k][1:], G['verts4d'][k], G['faces'], n=20000)
                ins.append(float((d < 0).mean()))
            ov[g] = dict(hidden_by_gallbladder=quality.summarize(hid), inside_gallbladder=quality.summarize(ins),
                         hidden_per_keyframe=[round(x, 3) for x in hid])
        Q['gallbladder_overlap'] = ov
        log('overlap with gallbladder: ' + '; '.join(f'{g} hidden {v["hidden_by_gallbladder"]["mean"]} inside '
                                                       f'{v["inside_gallbladder"]["mean"]} (max {v["inside_gallbladder"]["max"]})'
                                                       for g, v in ov.items()))
        log('proximal -> gallbladder surface |d| median/max (mm): ' + '; '.join(
            f'{tag} {nm} {v["abs"]["median"]}/{v["abs"]["max"]} rest {v["rest"]}' for tag in ('before', 'after')
            for nm, v in chk[tag].items()))
    lt = {}
    for P in pieces:
        L = np.linalg.norm(np.diff(P['cl4d'], axis=1), axis=2).sum(1)
        L0 = np.linalg.norm(np.diff(P['rest'], axis=0), axis=1).sum()
        e2e = np.linalg.norm(P['cl4d'][:, -1] - P['cl4d'][:, 0], axis=1)
        k_ls = int(np.argmin(L))
        P['least_stretched_frame'] = k_ls
        lt[P['name']] = dict(rest_mm=round(L0 * 1000, 2), min_mm=round(float(L.min() * 1000), 2),
                             max_mm=round(float(L.max() * 1000), 2), ratio_range=[round(float(L.min() / L0), 3),
                                                                                    round(float(L.max() / L0), 3)],
                             end_to_end_mm=[round(float(e2e.min() * 1000), 2), round(float(e2e.max() * 1000), 2)],
                             rest_end_to_end_mm=round(float(np.linalg.norm(P['rest'][-1] - P['rest'][0]) * 1000), 2),
                             least_stretched_frame=k_ls,
                             ratio_by_band={f'{a_}-{b_}': [round(float(L[a_:b_].min() / L0), 3), round(float(L[a_:b_].max() / L0), 3)]
                                            for a_, b_ in ((0, 62), (62, 130), (130, 251))},
                             depth_bow_mm=dict(rest=round(float(_bow(P['rest'], d_view) * 1000), 2),
                                               median_4d=round(float(np.median([_bow(c, d_view) for c in P['cl4d']]) * 1000), 2),
                                               max_4d=round(float(max(_bow(c, d_view) for c in P['cl4d']) * 1000), 2)),
                             distal_motion_mm=round(float(np.linalg.norm(P['cl4d'][:, -1] - P['rest'][-1], axis=1).max() * 1000), 3))
    Q['length_over_time'] = lt
    # can a cable glued at node 0 and driven at node 12 reproduce the 4D? (13 nodes as cable_spec)
    emt = {}
    for P in pieces:
        C = P['rest']
        sr = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        nodes = np.stack([np.interp(np.linspace(0, sr[-1], 13), sr, C[:, c]) for c in range(3)], 1)
        n4 = []
        for Ck in P['cl4d']:
            sk = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Ck, axis=0), axis=1))])
            n4.append(np.stack([np.interp(np.linspace(0, sk[-1], 13), sk, Ck[:, c]) for c in range(3)], 1))
        n4 = np.stack(n4)
        emt[P['name']] = end_motion_test(nodes, n4)
    Q['end_motion_test'] = emt
    if '--cable-test' in argv:      # quasi-static cable driven by the 4D at node 0 + node 12 (+ a mid node)
        ct = {}
        for P in pieces:
            if P['name'] not in ('duct', 'strand_2'):
                continue
            C = P['rest']
            sr = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
            nodes = np.stack([np.interp(np.linspace(0, sr[-1], 13), sr, C[:, c]) for c in range(3)], 1)
            n4 = []
            for Ck in P['cl4d']:
                sk = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Ck, axis=0), axis=1))])
                n4.append(np.stack([np.interp(np.linspace(0, sk[-1], 13), sk, Ck[:, c]) for c in range(3)], 1))
            n4 = np.stack(n4)
            mid = emt[P['name']]['drive'][0][0]
            ct[P['name']] = dict(ends=cable_equilibrium_test(nodes, n4, (0, -1))[0],
                                 ends_mid=cable_equilibrium_test(nodes, n4, (0, -1, mid))[0])
            log(f'cable test {P["name"]}: {ct[P["name"]]}')
        Q['cable_equilibrium_test'] = ct
    log('end-motion test (motion explained): ' + '; '.join(
        f'{n} lin {v["lin"]} chord {v["chord"]} prox-only {v["prox_only"]} +drive {v["drive"]}' for n, v in emt.items()))
    log('length over time: ' + '; '.join(f'{n} {v["min_mm"]}-{v["max_mm"]} (rest {v["rest_mm"]}, e2e {v["end_to_end_mm"]})'
                                         for n, v in lt.items()))

    rest_verts = np.concatenate(rest_all, 0)
    faces = np.concatenate(faces_all, 0)
    tets = np.concatenate(tets_all, 0)
    verts4d = np.concatenate(v4_all, 1)
    assert np.isfinite(verts4d).all()
    np.savez_compressed(vdir / 'model.npz', rest_verts=rest_verts, faces=faces, tets=tets,
                        verts4d=verts4d.astype(np.float32), attach_idx=np.array(att_idx), attach_to=np.array(att_to),
                        tube_id=np.array(tube_id), tube_names=np.array([P['name'] for P in pieces]),
                        least_stretched_frame=np.array([P.get('least_stretched_frame', 0) for P in pieces]),
                        anchor_gb=np.array([cfg.get('gb_version') or '', str(-1 if anchor is None else anchor['vertex'])]),
                        anchor_offset_rest=np.zeros(3) if anchor is None else np.asarray(anchor.get('offset_rest', np.zeros(3))),
                        anchor_rest_point=np.full(3, np.nan) if anchor is None else np.asarray(anchor['rest']),
                        anchor_point4d=np.zeros((0, 3)) if anchor is None else np.asarray(anchor['traj']),
                        anchor_face4d=np.asarray(anchor.get('face4d', np.zeros(0, int))) if anchor is not None else np.zeros(0, int),
                        **model)
    Q['mesh_health'] = quality.mesh_health(rest_verts, faces)
    Q['duct_tets_health'] = quality.mesh_health(pieces[0]['rv'], pieces[0]['F'], tets - pieces[0]['off'])
    inv4 = [int((np.einsum('ij,ij->i', np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                           X[tets[:, 3]] - X[tets[:, 0]]) <= 0).sum()) for X in verts4d]
    Q['duct_tets_inverted_4d_max'] = int(max(inv4))
    ious = []
    for k in ks:
        m, z = quality.render(V, verts4d, faces, k)
        vis = m & (z < V.depth(k) + 0.006)
        valid = ~_unknown(V, k)
        real = (M['duct'][k] | M['strands'][k]) & valid
        a = vis & valid
        ious.append(float((a & real).sum() / max((a | real).sum(), 1)))
    Q['union_iou_summary'] = quality.summarize(ious)
    Q['runtime_s'] = round(time.time() - t0, 1)
    json.dump(Q, open(vdir / 'quality.json', 'w'), indent=1, default=lambda o: None if o != o else o)
    quality.contact_sheet(V, layers, ks, vdir / 'sheet.jpg')
    _zoom_sheet(V, layers, M, ks, vdir / 'work' / 'sheet_zoom.jpg')
    views3d(V, [(P['rv'], P['F'], TUBES[P['group']]['rgb']) for P in pieces], vdir / 'views3d.jpg')
    log(f'width scale {Q["width_scale"]["duct"]["used"]} / {Q["width_scale"]["strands"]["used"]}; clearance '
        f'{Q["clearance_mm"]}; GB check {Q.get("duct_proximal_to_gallbladder_model_mm", {}).get("median")} mm')
    log(f'union IoU {Q["union_iou_summary"]["mean"]}  envelope IoU {Q["strands_envelope"]["iou_summary"]["mean"]} '
        f'strand->duct proximal {Q["strand_to_duct_proximal_mm"]["median"]} mm  tets {Q["duct_tets_health"]} '
        f'inverted 4d {Q["duct_tets_inverted_4d_max"]}  done in {Q["runtime_s"]} s')


def _zoom_sheet(V, layers, M, ks, path, box=(300, 80, 580, 330), cols=6):
    x0, y0, x1, y1 = box
    tiles = []
    for k in ks:
        img = V.frames[k].copy()
        for nm, v4, F, rgb in layers:
            m, z = quality.render(V, v4, F, k)
            m &= z < V.depth(k) + 0.006
            cnt = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            img = cv2.drawContours(np.ascontiguousarray(img), cnt, -1, rgb, 1)
            cm = cv2.findContours(M[nm][k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            img = cv2.drawContours(img, cm, -1, (255, 255, 255), 1)
        t = img[y0:y1, x0:x1].copy()
        cv2.putText(t, f'f{k}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == '__main__':
    main(sys.argv[1:])
