"""Photoreal real-to-sim of the laparoscopic clip (runs_real/chole_sweep/task/video.mp4, 640x360, 25 fps, 6 s).

Pipeline
  1. tracks      instrument tips + image-axis directions per frame (colour segmentation; probe = bright rod,
                 grasper = low-saturation shaft), gaps interpolated, lightly smoothed
  2. geometry    pinhole camera (fovy 47 deg, ~75 deg horizontal) and a depth prior: smooth background relief,
                 gallbladder as a dome over its outline, peritoneal fold as a shallow dome (no learned depth)
  3. texture     frame 0 with both instruments inpainted -> projected onto every surface (UV = image coordinates),
                 so the first rendered frame reproduces the video from the original viewpoint
  4. scene       static textured backdrop mesh (visual only) + deformable textured flex sheet for gallbladder,
                 neck and peritoneal fold (per-vertex anchoring springs to the bed, shell elasticity, rim pinned)
  5. instruments RCM of each instrument fitted to the observed shaft directions, tips placed on the tissue surface,
                 inverse kinematics -> joint targets at 20 Hz; the grasper holds the neck through a grasp constraint
                 (as in SOFA/SurRoL grasping), the probe deforms tissue only through contact
  6. simulate + render from the endoscope camera; metrics vs the video in metrics.py
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'realistic' / 'chole'
sys.path.insert(0, str(ROOT / 'tools'))
import kinematics as K  # noqa: E402

W, H, FPS = 640, 360, 25
PAD = 112                      # texture is reflect-padded by PAD px so camera motion never reveals clamped texels
FOVY = 47.0
F = (H / 2) / np.tan(np.radians(FOVY / 2))
CAM_POS = np.array([0.0, 0.0, 0.10])
CAM_LOOK = CAM_POS + np.array([0.0, np.cos(np.radians(40)), -np.sin(np.radians(40))])
GB_POLY = [(105, 360), (100, 290), (115, 230), (150, 185), (190, 165), (205, 120), (225, 105), (250, 130),
           (262, 175), (300, 205), (330, 250), (345, 300), (330, 360)]
# the purple tubular structure (peritoneum-covered pedicle) traced on frame 0: a Y from the neck to two anchors
STRANDS = {'A': dict(pts=[(505, 128), (490, 140), (460, 165), (430, 185), (400, 192), (360, 205), (320, 220), (285, 230)], r=0.005),
           'B': dict(pts=[(430, 185), (455, 205), (475, 235), (490, 265), (500, 295)], r=0.0035)}
FOLD_POLY = [(250, 140), (300, 118), (380, 92), (460, 108), (500, 150), (490, 205), (430, 235), (360, 245),
             (300, 215), (262, 180)]


# ---------------------------------------------------------------- camera
def cam_axes():
    z = CAM_LOOK - CAM_POS
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


R_CW = cam_axes()


def unproject(u, v, z):
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    pc = np.stack([(u - W / 2) * z / F, (v - H / 2) * z / F, z], -1)
    return CAM_POS + pc @ R_CW


def project(p, R=None, f=None):
    R = R_CW if R is None else R
    f = F if f is None else f
    pc = (np.asarray(p, float) - CAM_POS) @ R.T
    return np.stack([f * pc[..., 0] / pc[..., 2] + W / 2, f * pc[..., 1] / pc[..., 2] + H / 2], -1), pc[..., 2]


def unproject_k(u, v, z, R, f):
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    pc = np.stack([(u - W / 2) * z / f, (v - H / 2) * z / f, z], -1)
    return CAM_POS + pc @ R


def lk_tracks(frames, exclude, n=400):
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    p0 = cv2.goodFeaturesToTrack(g[0], n, 0.01, 8, mask=(~exclude).astype(np.uint8) * 255)
    P = np.full((len(g), len(p0), 2), np.nan, np.float32)
    P[0] = p0[:, 0]
    alive, cur = np.ones(len(p0), bool), p0
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for k in range(1, len(g)):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(g[k - 1], g[k], cur, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g[k], g[k - 1], nxt, None, **lk)
        alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - cur, axis=2)[:, 0] < 1.0)
        P[k][alive] = nxt[alive, 0]
        cur = nxt
    return P


def camera_track(P, static_mask):
    """Scope motion as rotation + zoom about the fixed scope tip, fitted to tracks on the static backdrop."""
    from scipy.spatial.transform import Rotation as Rot
    p0 = P[0]
    bg = np.array([static_mask[int(y), int(x)] for x, y in p0])
    rays = np.c_[(p0 - [W / 2, H / 2]) / F, np.ones(len(p0))]
    pars, err, par = [np.zeros(4)], [0.0], np.zeros(4)
    for k in range(1, len(P)):
        ok = bg & np.isfinite(P[k, :, 0])

        def f(x):
            q = rays[ok] @ Rot.from_rotvec(x[:3]).as_matrix().T
            return (np.c_[F * np.exp(x[3]) * q[:, 0] / q[:, 2] + W / 2, F * np.exp(x[3]) * q[:, 1] / q[:, 2] + H / 2] - P[k, ok]).ravel()
        sol = least_squares(f, par, loss='huber', f_scale=3.0)
        par = sol.x
        pars.append(par.copy())
        err.append(float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1))))
    pars = fill_smooth(np.array(pars), 9)
    Rs = [Rot.from_rotvec(x[:3]).as_matrix() @ R_CW for x in pars]
    fs = [F * np.exp(x[3]) for x in pars]
    return dict(R=np.array(Rs), f=np.array(fs), params=pars, fit_err_px=np.array(err))


# ---------------------------------------------------------------- 1. tracks
def _axis_tip(mask, pick):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    d = pick(d)
    s = (pts - c) @ d
    return pts[s > np.percentile(s, 99.3)].mean(0), d


def probe_obs(f):
    h = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    m = ((h[..., 2] > 170) & (h[..., 1] < 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    c = [i for i in range(1, n) if st[i, 4] > 400 and (st[i, 0] + st[i, 2] >= W - 3 or st[i, 1] <= 3) and st[i, 2] > 120]
    if not c:
        return None
    i = max(c, key=lambda i: st[i, 4])
    tip, d = _axis_tip(lab == i, lambda d: d if d[0] < 0 else -d)
    return tip, d, lab == i


def grasper_obs(f):
    h = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(int)
    m = ((h[..., 1] < 60) & (h[..., 2] > 45) & (h[..., 2] < 175)).astype(np.uint8)
    m[:, 330:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    c = [i for i in range(1, n) if st[i, 1] <= 2 and st[i, 4] > 1200]
    if not c:
        return None
    i = max(c, key=lambda i: st[i, 4])
    tip, d = _axis_tip(lab == i, lambda d: d if d[1] > 0 else -d)
    return tip, d, lab == i


def fill_smooth(x, w=7):
    x = np.array(x, float)
    for k in range(x.shape[1]):
        ok = np.isfinite(x[:, k])
        x[:, k] = np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok, k])
    pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode='edge')
    return np.stack([np.convolve(pad[:, k], np.ones(w) / w, mode='valid') for k in range(x.shape[1])], 1)


def tracks(frames):
    out = {}
    for name, fn in (('probe', probe_obs), ('grasper', grasper_obs)):
        tip, d, masks = [], [], []
        for f in frames:
            o = fn(f)
            tip.append(o[0] if o else [np.nan] * 2)
            d.append(o[1] if o else [np.nan] * 2)
            masks.append(o[2] if o else np.zeros(f.shape[:2], bool))
        out[name] = dict(tip=fill_smooth(tip), dir=fill_smooth(d), raw_tip=np.array(tip), masks=masks)
        out[name]['dir'] /= np.linalg.norm(out[name]['dir'], axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------- 2. geometry
def poly_mask(poly):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def dome(mask):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sqrt(np.clip(dt / max(dt.max(), 1), 0, 1))


def depth_map():
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    un, vn = u / W, v / H
    z = 0.078 + 0.024 * un                                      # liver on the left is closest, right organs further
    z += 0.016 * np.clip((0.33 - vn) / 0.33, 0, 1) * np.clip((un - 0.25) / 0.2, 0, 1)   # dark recess under the liver
    z -= 0.006 * np.clip((vn - 0.8) / 0.2, 0, 1)                # foreground tissue at the bottom
    z = cv2.GaussianBlur(z, (0, 0), 15)
    gb, fold = poly_mask(GB_POLY), poly_mask(FOLD_POLY)
    zb = z.copy()
    z = z - 0.020 * dome(gb) - 0.006 * dome(fold & ~gb)
    return z, zb, gb, fold


# ---------------------------------------------------------------- 3. texture
def clean_plate(frames, tr, cams, hole, every=2, cover_px=15, first=None, pct=30):
    """Fill `hole` (frame-0 pixels hidden by instruments) with a temporal percentile of the same scene points in other
    frames (warped by the scope rotation+zoom) where no instrument covers them. `first`: use only the earliest that
    many uncovered samples (the scene changes later, e.g. strands lifted into the gap). Returns the filled image and
    the pixels that were never uncovered."""
    v0, u0 = np.nonzero(hole)
    X = unproject(u0, v0, 0.1)
    samples = [[] for _ in range(len(u0))]
    for k in range(0, len(frames), every):
        uk = project(X, cams['R'][k], cams['f'][k])[0]
        inside = (uk[:, 0] >= 0) & (uk[:, 0] < W - 1) & (uk[:, 1] >= 0) & (uk[:, 1] < H - 1)
        cover = cv2.dilate((tr["probe"]["masks"][k] | tr["grasper"]["masks"][k]).astype(np.uint8), np.ones((cover_px, cover_px), np.uint8))
        for i in np.nonzero(inside)[0]:
            x, y = int(uk[i, 0]), int(uk[i, 1])
            if not cover[y, x]:
                samples[i].append(frames[k][y, x])
    out = frames[0].copy()
    never = np.zeros((H, W), np.uint8)
    for i, smp in enumerate(samples):
        if len(smp) >= 3:
            smp = smp[:first] if first else smp
            out[v0[i], u0[i]] = np.percentile(np.array(smp), pct, axis=0)   # bloom only brightens: lean dark
        else:
            never[v0[i], u0[i]] = 1
    return out, never


def band_mask(f):
    """Purple strand pixels (HSV), restricted to the region of the traced strands."""
    h = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(int)
    m = (((h[..., 0] >= 115) & (h[..., 0] <= 175)) & (h[..., 1] >= 50) & (h[..., 2] >= 20) & (h[..., 2] <= 200)).astype(np.uint8)
    roi = np.zeros_like(m)
    for S_ in STRANDS.values():
        cv2.polylines(roi, [np.array(S_['pts'], np.int32)], False, 1, 60)
    m = cv2.morphologyEx(m * roi, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


def strand_points(z, step=9):
    """Resample each traced strand every `step` px and lift it onto the tissue surface (centre one radius in front)."""
    out = {}
    for name, S_ in STRANDS.items():
        P = np.array(S_['pts'], float)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        sacc = np.r_[0, np.cumsum(seg)]
        ss = np.linspace(0, sacc[-1], max(3, int(sacc[-1] / step) + 1))
        px = np.stack([np.interp(ss, sacc, P[:, 0]), np.interp(ss, sacc, P[:, 1])], 1)
        zz = z[np.clip(px[:, 1].astype(int), 0, H - 1), np.clip(px[:, 0].astype(int), 0, W - 1)] - S_['r'] - 0.0035   # clear of the tissue's 2.5 mm contact margin
        out[name] = dict(px=px, X=unproject(px[:, 0], px[:, 1], zz), r=S_['r'])
    return out


RIBBON_ANCHORS = [(505, 128), (500, 295)]          # far ends of the strand, fixed to the surrounding anatomy
RIBBON_NECK = (255, 182)                           # near end, attached to the gallbladder neck (under the probe)


def ribbon_sheet(band, z, step=8, lift=0.006):
    """Textured ribbon over the purple band, `lift` in front of the tissue surface; flags for pinned / tied vertices."""
    inner = cv2.erode(band.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pts = [(u, v) for v in range(2, H - 2, step) for u in range(int(step / 2) * ((v // step) % 2) + 2, W - 2, step) if inner[v, u]]
    cnt = max(cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    rim = [tuple(map(int, q)) for q in cnt[::max(1, len(cnt) // 90)]]
    if pts:                                        # no rim point closer than 0.6 step to the grid: avoids slivers
        G = np.array(pts, float)
        rim = [r for r in rim if np.min(np.linalg.norm(G - r, axis=1)) > 0.6 * step]
    P = np.array(pts + rim, float)
    tri = Delaunay(P)

    def min_angle(t):
        a, b, c = P[t]
        ang = []
        for p0, p1, p2 in ((a, b, c), (b, c, a), (c, a, b)):
            u, v = p1 - p0, p2 - p0
            ang.append(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))
        return min(ang)
    keep = [t for t in tri.simplices if band[int(np.clip(P[t].mean(0)[1], 0, H - 1)), int(np.clip(P[t].mean(0)[0], 0, W - 1))]
            and min_angle(t) > 18]
    T = np.array(keep)
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, T = P[used], remap[T]
    zz = z[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)] - lift
    X = unproject(P[:, 0], P[:, 1], zz)
    pinned = np.zeros(len(P), bool)
    for a in RIBBON_ANCHORS:
        pinned |= np.linalg.norm(P - a, axis=1) < 16
    tied = np.linalg.norm(P - RIBBON_NECK, axis=1) < 16
    return dict(px=P, X=X, tri=T, pinned=pinned, tied=tied & ~pinned)


def textures(frame0, tr, frames=None, cams=None):
    frame0 = frame0.copy()
    frame0[-4:] = frame0[-8:-4][::-1]            # dark letterbox remnant in the last rows would show as a seam
    frame0[:2] = frame0[2:4][::-1]
    frame0[:, -3:] = frame0[:, -6:-3][:, ::-1]
    m = (tr['probe']['masks'][0] | tr['grasper']['masks'][0]).astype(np.uint8)
    m = cv2.dilate(m, np.ones((17, 17), np.uint8))       # include the specular bloom around the polished probe
    if frames is not None:
        f0 = [frame0] + list(frames[1:])
        plate, never = clean_plate(f0, tr, cams, m.astype(bool))
        plate = cv2.GaussianBlur(plate, (0, 0), 0.7)
        # feathered blend of the plate into frame 0 (no Poisson: it washes large holes out)
        a = cv2.GaussianBlur(cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
        frame0 = (a * plate + (1 - a) * frame0).astype(np.uint8)
        m = cv2.dilate(never, np.ones((3, 3), np.uint8))
    clean = cv2.inpaint(frame0, m, 7, cv2.INPAINT_TELEA) if m.any() else frame0
    ribbon_tex = clean.copy()                     # the strand keeps its real appearance on its own ribbon mesh
    # ...and is removed from the tissue underneath (what the probe uncovers when it lifts the strand)
    bm = cv2.dilate(band_mask(frame0).astype(np.uint8), np.ones((7, 7), np.uint8))
    clean = cv2.inpaint(clean, bm, 9, cv2.INPAINT_TELEA)
    soft = poly_mask(GB_POLY) | poly_mask(FOLD_POLY)
    back = cv2.inpaint(clean, cv2.dilate(soft.astype(np.uint8), np.ones((5, 5), np.uint8)), 15, cv2.INPAINT_TELEA)
    back = cv2.addWeighted(back, 0.75, cv2.GaussianBlur(back, (0, 0), 6), 0.25, 0)
    return clean, back, ribbon_tex


# ---------------------------------------------------------------- 4. meshes
def uv(u, v):
    return (u + PAD) / (W + 2 * PAD), (v + PAD) / (H + 2 * PAD)


def backdrop_obj(path, zb, step=8, margin=PAD):
    us = np.arange(-margin, W + margin + 1, step)
    vs = np.arange(-margin, H + margin + 1, step)
    zz = cv2.copyMakeBorder(zb.astype(np.float32), margin, margin, margin, margin, cv2.BORDER_REPLICATE)
    lines, idx = [], {}
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            p = unproject(u, v, zz[int(np.clip(v + margin, 0, zz.shape[0] - 1)), int(np.clip(u + margin, 0, zz.shape[1] - 1))] + 0.004)
            lines.append('v %.6f %.6f %.6f' % tuple(p))
            tu, tv = uv(u, v)
            lines.append('vt %.6f %.6f' % (tu, 1 - tv))
            idx[i, j] = len(idx) + 1
    for j in range(len(vs) - 1):
        for i in range(len(us) - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            lines.append(f'f {a}/{a} {c}/{c} {b}/{b}')
            lines.append(f'f {a}/{a} {d}/{d} {c}/{c}')
    Path(path).write_text('\n'.join(lines))


def flex_sheet(z, gb, fold, step=11):
    soft = gb | fold
    inner = cv2.erode(soft.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pts = [(u, v) for v in range(4, H - 2, step) for u in range(int(step / 2) * ((v // step) % 2) + 2, W - 2, step) if inner[v, u]]
    cnt = cv2.findContours(soft.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    rim = max(cnt, key=len)[:, 0, :]
    rim = rim[::max(1, len(rim) // 110)]
    rim = [tuple(map(int, p)) for p in rim if 0 < p[1] < H - 1 or True]
    G = np.array(pts, float)
    rim = [r for r in rim if np.min(np.linalg.norm(G - r, axis=1)) > 0.6 * step]   # no slivers along the boundary
    P = np.array(pts + rim, float)
    is_rim = np.r_[np.zeros(len(pts), bool), np.ones(len(rim), bool)]
    tri = Delaunay(P)

    def min_angle(t):
        out = []
        for i0, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            u_, v_ = P[t[i1]] - P[t[i0]], P[t[i2]] - P[t[i0]]
            out.append(np.degrees(np.arccos(np.clip(u_ @ v_ / (np.linalg.norm(u_) * np.linalg.norm(v_) + 1e-9), -1, 1))))
        return min(out)
    keep = []
    for t in tri.simplices:
        c = P[t].mean(0).astype(int)
        if soft[np.clip(c[1], 0, H - 1), np.clip(c[0], 0, W - 1)] and min_angle(t) > 12:
            keep.append(t)
    T = np.array(keep)
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, is_rim, T = P[used], is_rim[used], remap[T]
    zz = z[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)]
    X = unproject(P[:, 0], P[:, 1], zz)
    in_gb = gb[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)]
    bottom = P[:, 1] >= H - 3                                       # cut by the image border: anchored firmly
    dm = dome(gb)
    thick = 0.003 + 0.022 * dm[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)]
    return dict(px=P, X=X, tri=T, rim=is_rim & ~bottom, bottom=bottom, in_gb=in_gb, thick=thick)


# ---------------------------------------------------------------- 5. instruments
SHAFT_D = 0.005


def width_along(mask, tip, d, ts):
    """Apparent shaft width (px) at image distances ts behind the tip: length of the mask run across the axis."""
    nrm = np.array([-d[1], d[0]])
    out = []
    for t in ts:
        p = tip - d * t
        run = 0
        for sgn in (1, -1):
            for s in range(0 if sgn > 0 else 1, 45):
                q = p + sgn * nrm * s
                if not (0 <= q[0] < W and 0 <= q[1] < H) or not mask[int(q[1]), int(q[0])]:
                    break
                run += 1
        out.append(run)
    return np.array(out, float)


def shaft_lines(tr, name, every=3):
    """Per frame a 3D shaft line from apparent widths (depth = F * D / width), robustly fitted."""
    lines = {}
    for k in range(0, len(tr[name]['masks']), every):
        m, tip, d = tr[name]['masks'][k], tr[name]['raw_tip'][k], None
        if not m.any() or not np.isfinite(tip).all():
            continue
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1).astype(float)
        dd = np.linalg.svd(pts - pts.mean(0))[2][0]
        d = dd if np.dot(dd, tr[name]['dir'][k]) > 0 else -dd
        ts = np.arange(14, 260, 6)
        ts = ts[[0 <= (tip - d * t)[0] < W and 0 <= (tip - d * t)[1] < H for t in ts]]
        w = width_along(m, tip, d, ts)
        ok = (w >= 6) & (w <= 70)
        if ok.sum() < 5:
            continue
        ts, w = ts[ok], w[ok]
        # robust: 1/w is affine in image distance only approximately; fit z(t) with a median line
        z = F * SHAFT_D / w
        A = np.c_[ts, np.ones_like(ts)]
        coef = np.linalg.lstsq(A, z, rcond=None)[0]
        res = np.abs(A @ coef - z)
        keep = res < max(np.median(res) * 2.5, 0.004)
        coef = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        px = tip[None] - d[None] * ts[keep, None]
        P3 = unproject(px[:, 0], px[:, 1], A[keep] @ coef)
        c = P3.mean(0)
        u = np.linalg.svd(P3 - c)[2][0]
        tip3 = unproject(tip[0], tip[1], coef[1])                # z at t = 0
        u = u if np.dot(u, c - tip3) > 0 else -u               # pointing from the tip towards the RCM
        lines[k] = dict(c=c, u=u, tip3=tip3, tip_depth=coef[1], slope=coef[0])
    return lines


def rcm_from_lines(lines):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for L in lines.values():
        Pm = np.eye(3) - np.outer(L['u'], L['u'])
        A += Pm
        b += Pm @ L['c']
    P = np.linalg.solve(A, b)
    resid = np.mean([np.linalg.norm((np.eye(3) - np.outer(L['u'], L['u'])) @ (P - L['c'])) for L in lines.values()])
    return P, float(resid)


def tip_world(tip_px, z, offset):
    zz = z[np.clip(tip_px[:, 1].astype(int), 0, H - 1), np.clip(tip_px[:, 0].astype(int), 0, W - 1)]
    return unproject(tip_px[:, 0], tip_px[:, 1], zz + offset)


def fit_rcm(T, dirs, init):
    """RCM P such that the projected shaft (P -> tip) matches the observed image direction (pointing to the tip)."""
    def res(P):
        r = []
        for t, d in zip(T[::3], dirs[::3]):
            a, _ = project(t)
            b, _ = project(t + 0.01 * (P - t) / np.linalg.norm(P - t))
            e = a - b
            e /= np.linalg.norm(e) + 1e-12
            r.append(e[0] * d[1] - e[1] * d[0])
        dist = np.linalg.norm(P - T.mean(0))
        r.append(5 * max(0.0, 0.10 - dist))
        r.append(5 * max(0.0, dist - 0.20))
        return np.array(r)
    s = least_squares(res, init, x_scale=0.02)
    return s.x, float(np.degrees(np.arcsin(np.clip(np.abs(s.fun[:-2]), 0, 1))).mean())


SLEW = np.array([0.08, 0.08, 0.006, 0.15, 0.007])        # per 0.05 s step, as in the benchmark


def joint_targets(P, Tseq, heading, jaw):
    out = []
    for t in Tseq:
        y, p, ins = K.ik(P, t, heading)
        out.append([y, p, ins, 0.0, jaw])
    out = np.array(out)
    for k in range(1, len(out)):                              # rate-limit (no teleporting instruments)
        out[k] = out[k - 1] + np.clip(out[k] - out[k - 1], -SLEW, SLEW)
    return out


def heading_for(P, T):
    v = T.mean(0) - P
    return float(np.arctan2(-v[0], v[1]))     # shaft along trocar +y at yaw 0  ->  heading = atan2(-vx, vy)


def fit_rcm_ports(tips, dirs2d, init_px, cams=None, fwd_range=(-0.06, 0.03)):
    """RCM (port) for 3D tips: projected shaft directions must match the observed 2D axes; the port lies near the
    endoscope's depth (abdominal wall, camera-forward coordinate in [-0.06, 0.03] m) and 8-22 cm from the tips."""
    def res(P):
        r = []
        for k in range(0, len(tips), 2):
            t, d = tips[k], dirs2d[k]
            Rk, fk = (cams['R'][k], cams['f'][k]) if cams is not None else (None, None)
            a, _ = project(t, Rk, fk)
            b, _ = project(t + 0.005 * (P - t) / np.linalg.norm(P - t), Rk, fk)
            e = a - b
            e /= np.linalg.norm(e) + 1e-12
            r.append(e[0] * d[1] - e[1] * d[0])
        fwd = (P - CAM_POS) @ R_CW[2]
        dist = np.linalg.norm(P - tips.mean(0))
        r += [3 * max(0, fwd - fwd_range[1]), 3 * max(0, fwd_range[0] - fwd), 3 * max(0, 0.08 - dist), 3 * max(0, dist - 0.22)]
        return np.array(r)
    best = None
    for z0 in (0.0, 0.03, 0.06, float(np.mean(fwd_range))):
        init = unproject(*init_px, max(z0, 0.01) + 0.02) if z0 else CAM_POS + (unproject(*init_px, 0.05) - CAM_POS)
        s = least_squares(res, init, x_scale=0.02)
        if best is None or s.cost < best.cost:
            best = s
    ang = np.degrees(np.arcsin(np.clip(np.abs(best.fun[:-4]), 0, 1)))
    return best.x, float(ang.mean())


def surface_capped(tips, cams, zscale, press=0.003):
    """Move each tip toward the camera along its ray until it is at most `press` below the tissue surface."""
    z0 = depth_map()[0] * zscale
    out = []
    for k, t in enumerate(tips):
        ray = t - CAM_POS
        dist = np.linalg.norm(ray)
        ray /= dist
        hit = dist
        for s_ in np.linspace(0.04, dist, 160):
            q = CAM_POS + ray * s_
            px, zc = project(q)
            u, v = int(np.clip(px[0], 0, W - 1)), int(np.clip(px[1], 0, H - 1))
            if zc >= z0[v, u]:
                hit = s_
                break
        out.append(CAM_POS + ray * min(dist, hit + press))
    return np.array(out)


def tip_on_ray(P, tip_px):
    """Point on the camera ray of tip_px closest to the line through the RCM with the ray's best-fitting depth."""
    out = []
    for u, v in tip_px:
        r = unproject(u, v, 1.0) - CAM_POS
        r /= np.linalg.norm(r)
        out.append(r)
    return np.array(out)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [f for f in imageio.get_reader(ROOT / 'runs_real/chole_sweep/task/video.mp4')]
    tr = tracks(frames)
    instr = np.zeros((H, W), bool)
    for key in ('probe', 'grasper'):
        for m in tr[key]['masks']:
            instr |= m
    instr = cv2.dilate(instr.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    P_lk = lk_tracks(frames, instr)
    np.save(OUT / 'real_tracks.npy', P_lk)
    _, _, gb0, fold0 = depth_map()
    cams = camera_track(P_lk, ~(gb0 | fold0))
    inst, t_src = {}, np.arange(len(frames)) / FPS
    for name, key, init_px in (('probe_right', 'probe', (900, -200)), ('grasper_left', 'grasper', (60, -250))):
        lines = shaft_lines(tr, key)                         # used for the tip depth (apparent shaft width)
        ks = np.array(sorted(lines))
        depth = np.array([lines[k]['tip_depth'] for k in ks])
        # widths become unreliable once the grasper retracts and only a stub of shaft is visible: take its depth from
        # the first 80 frames; clamp both instruments to +-1.5 cm around their median
        med = float(np.median(depth[ks < 80] if key == 'grasper' else depth))
        depth = np.clip(depth, med - 0.015, med + 0.015)
        if key == 'grasper':
            depth = np.full_like(depth, med) + 0.25 * (depth - med)
        tip_depth = fill_smooth(np.interp(np.arange(len(frames)), ks, depth)[:, None], 15)[:, 0]
        tips = np.array([unproject_k(tr[key]['tip'][k, 0], tr[key]['tip'][k, 1], tip_depth[k], cams['R'][k], cams['f'][k])
                         for k in range(len(frames))])
        P, err = fit_rcm_ports(tips, tr[key]['dir'], init_px, cams)
        dirs = (tips - P) / np.linalg.norm(tips - P, axis=1, keepdims=True)       # RCM -> tip
        T = tips + (0.022 * dirs if key == 'grasper' else -0.003 * dirs)          # model TCP
        inst[name] = dict(rcm=P, rcm_resid_mm=err, tips=tips, T=T, tip_depth=tip_depth, n_lines=len(lines))
    # depth prior calibrated to the instrument contacts in frame 0 (probe tip on the gallbladder, grasper on the neck)
    z0, _, gb, fold = depth_map()
    pt = tr['probe']['tip'][0]
    gt = project(inst['grasper_left']['T'][0])[0]
    ratios = [inst['probe_right']['tip_depth'][0] / z0[int(pt[1]), int(pt[0])],
              project(inst['grasper_left']['T'][0])[1] / z0[int(np.clip(gt[1], 0, H - 1)), int(np.clip(gt[0], 0, W - 1))]]
    scale = float(ratios[0])                      # metric scale from the probe tip touching the gallbladder
    z, zb = z0 * scale, depth_map()[1] * scale
    # the grasper holds the neck: its jaws sit INSIDE the tissue (the neck wraps them in the video), so push the
    # whole grasper path along the viewing rays until the frame-0 grasp point is 6 mm behind the tissue surface,
    # then refit its port
    I = inst['grasper_left']
    gpx, gd = project(I['T'][0])
    zs = z[int(np.clip(gpx[1], 0, H - 1)), int(np.clip(gpx[0], 0, W - 1))]
    f_ = (zs + 0.006) / gd
    tips_g = CAM_POS + (I['tips'] - CAM_POS) * f_
    Pg, eg = fit_rcm_ports(tips_g, tr['grasper']['dir'], (60, -250), cams)
    dg = (tips_g - Pg) / np.linalg.norm(tips_g - Pg, axis=1, keepdims=True)
    I.update(rcm=Pg, rcm_resid_mm=eg, tips=tips_g, T=tips_g + 0.022 * dg, depth_shift=float(f_))
    # the probe sweeps over the tissue: its tip may press at most 2 mm below the (scaled) surface; refit its port
    I = inst['probe_right']
    capped = surface_capped(I['tips'], cams, scale, press=0.002)
    I['cap_moved_mm'] = float(np.linalg.norm(capped - I['tips'], axis=1).max() * 1000)
    P2, err2 = fit_rcm_ports(capped, tr['probe']['dir'], (900, -200), cams)
    # where the probe is not detected for > 5 consecutive frames it has been withdrawn out of view: slide it back
    # along its own shaft towards the (already fitted) port over ~1 s; the port is not refitted to these frames
    raw = np.isfinite(tr['probe']['raw_tip'][:, 0])
    gap, k = np.zeros(len(raw), bool), 0
    while k < len(raw):
        if not raw[k]:
            j = k
            while j < len(raw) and not raw[j]:
                j += 1
            if j - k > 5:
                gap[k:j] = True
            k = j
        else:
            k += 1
    w = np.clip(np.convolve(gap.astype(float), np.ones(25) / 25, mode='same') * 1.6, 0, 1)
    capped = capped + (0.45 * w)[:, None] * (P2 - capped)
    d2 = (capped - P2) / np.linalg.norm(capped - P2, axis=1, keepdims=True)
    I.update(rcm=P2, rcm_resid_mm=err2, tips=capped, T=capped - 0.003 * d2, withdrawn_frames=int(gap.sum()))
    # visibility: tissue seen behind an instrument must lie behind it (ignore 12 px around the probe tip contact)
    occ = np.full((H, W), np.inf)
    for name, key in (('probe_right', 'probe'), ('grasper_left', 'grasper')):
        I = inst[name]
        for k in range(0, len(frames), 2):
            m = tr[key]['masks'][k]
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            # depth of the shaft line at each mask pixel: intersect the pixel ray with the shaft line (closest point)
            a, u = I['rcm'], (I['tips'][k] - I['rcm']) / np.linalg.norm(I['tips'][k] - I['rcm'])
            rays = unproject(xs, ys, 1.0) - CAM_POS
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            w0 = CAM_POS - a
            bb, dd, ee = rays @ u, rays @ w0, w0 @ u
            sr = (bb * ee - dd) / np.maximum(1 - bb ** 2, 1e-9)
            zc = ((CAM_POS + rays * sr[:, None]) - CAM_POS) @ R_CW[2]
            if key == 'probe':
                far = np.hypot(xs - tr['probe']['tip'][k][0], ys - tr['probe']['tip'][k][1]) > 12
                xs, ys, zc = xs[far], ys[far], zc[far]
            np.minimum.at(occ, (ys, xs), zc)
    occ_d = cv2.erode(np.where(np.isfinite(occ), occ, 9).astype(np.float32), np.ones((7, 7), np.uint8))
    need = np.isfinite(cv2.dilate(np.where(np.isfinite(occ), 1, 0).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(float)) & (occ_d < 9)
    z_vis = np.where(need, np.maximum(z, occ_d + 0.005), z)
    z_vis = cv2.GaussianBlur(z_vis.astype(np.float32), (0, 0), 4)
    pushed = float(np.percentile(np.maximum(z_vis - z, 0)[need], 95)) if need.any() else 0.0
    z = np.maximum(z, z_vis)
    zb = np.maximum(zb, z)
    clean, back, ribbon_tex = textures(frames[0], tr, frames, cams)
    imageio.imwrite(OUT / 'tex_ribbon.png', cv2.copyMakeBorder(ribbon_tex, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT))
    imageio.imwrite(OUT / 'tex_tissue.png', cv2.copyMakeBorder(clean, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT))
    back_pad = cv2.copyMakeBorder(clean, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)   # outside the picture: mirrored real
    back_pad[PAD:PAD + H, PAD:PAD + W] = back                                         # texture, not the smeared fill
    imageio.imwrite(OUT / 'tex_back.png', back_pad)
    backdrop_obj(OUT / 'backdrop.obj', zb)
    sheet = flex_sheet(z, gb, fold)
    strands = strand_points(z)
    # the strand continues under the probe, which occludes it in frame 0: extend the band into the occluded pixels
    band0 = band_mask(frames[0])
    occl = cv2.dilate(tr['probe']['masks'][0].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    reach = cv2.dilate(band0.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    neck_link = np.zeros((H, W), np.uint8)
    cv2.line(neck_link, RIBBON_NECK, (300, 222), 1, 26)
    band_ext = band0 | (occl & reach) | neck_link.astype(bool)
    band_ext = cv2.morphologyEx(band_ext.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    ribbon = ribbon_sheet(band_ext, z)
    t_ctl = np.arange(0, t_src[-1] + 1e-9, 0.05)
    acts = []
    for name in ('grasper_left', 'probe_right'):
        I = inst[name]
        I['heading'] = heading_for(I['rcm'], I['T'])
        Ti = np.stack([np.interp(t_ctl, t_src, I['T'][:, k]) for k in range(3)], 1)
        acts.append(joint_targets(I['rcm'], Ti, I['heading'], 0.0))
    acts = np.concatenate(acts, 1)
    np.save(OUT / 'actions.npy', acts)
    np.savez(OUT / 'geometry.npz', z=z, zb=zb, gb=gb, fold=fold, sheet_px=sheet['px'], sheet_X=sheet['X'],
             sheet_tri=sheet['tri'], sheet_thick=sheet['thick'], sheet_rim=sheet['rim'], sheet_bottom=sheet['bottom'], sheet_in_gb=sheet['in_gb'],
             probe_tip_px=tr['probe']['tip'], grasper_tip_px=tr['grasper']['tip'],
             probe_raw=tr['probe']['raw_tip'], grasper_raw=tr['grasper']['raw_tip'],
             probe_T=inst['probe_right']['T'], grasper_T=inst['grasper_left']['T'],
             cam_R=cams['R'], cam_f=cams['f'], cam_fit_err=cams['fit_err_px'], band=band_mask(frames[0]),
             **{f'strand{k}_X': v['X'] for k, v in strands.items()}, **{f'strand{k}_px': v['px'] for k, v in strands.items()},
             **{f'strand{k}_r': v['r'] for k, v in strands.items()},
             ribbon_px=ribbon['px'], ribbon_X=ribbon['X'], ribbon_tri=ribbon['tri'], ribbon_pinned=ribbon['pinned'], ribbon_tied=ribbon['tied'], band_ext=band_ext)
    meta = dict(fovy=FOVY, cam_pos=CAM_POS.tolist(), cam_lookat=CAM_LOOK.tolist(), depth_scale=round(scale, 3),
                camera_motion=dict(max_rot_deg=round(float(np.degrees(np.linalg.norm(cams['params'][:, :3], axis=1)).max()), 2),
                                   zoom_range=[round(float(np.exp(cams['params'][:, 3]).min()), 3), round(float(np.exp(cams['params'][:, 3]).max()), 3)],
                                   fit_err_px_median=round(float(np.median(cams['fit_err_px'][1:])), 2)),
                visibility_push_p95_mm=round(pushed * 1000, 1), n_flex_vertices=len(sheet['X']), n_flex_triangles=len(sheet['tri']),
                instruments={k: dict(rcm=v['rcm'].round(4).tolist(), heading=round(v['heading'], 4), axis_fit_err_deg=round(v['rcm_resid_mm'], 2), rcm_cam=((v['rcm'] - CAM_POS) @ R_CW.T).round(3).tolist(),
                                     n_lines=v['n_lines'], tip_depth_m=[round(float(v['tip_depth'].min()), 3), round(float(v['tip_depth'].max()), 3)],
                                     rcm_to_tip_m=round(float(np.linalg.norm(v['rcm'] - v['tips'].mean(0))), 3), cap_moved_mm=round(v.get('cap_moved_mm', 0.0), 1), withdrawn_frames=v.get('withdrawn_frames', 0), depth_shift=round(v.get('depth_shift', 1.0), 3))
                             for k, v in inst.items()},
                actions_shape=list(acts.shape))
    (OUT / 'build_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


if __name__ == '__main__':
    build()
