"""Everything of the scene that does not depend on how the organ is modelled.

  camera     scope rotation + zoom per frame (tracks on anatomy that is neither organ, strand nor instrument)
  panorama   texture and depth on a canvas that extends the frame-0 picture by M px on each side: every canvas pixel
             takes the earliest frames that see it uncovered (frame 0 itself where it is uncovered there), so camera
             pans and instruments moving away reveal real tissue instead of mirrored or smeared texture
  backdrop   static mesh of the measured surface behind the organ (organ, strands and instruments filled in)
  organ obs  amodal organ mask at frame 0 (parts hidden by instruments / strands taken from early frames), its
             measured front depth, a thickness estimate from the silhouette, the observed bed motion
  strands    textured ribbons over each strand mask at their own depth (far ends fixed, a band at the organ end tied)
  actions    instrument ports and joint targets (instruments.py)
"""
import numpy as np
import cv2
import imageio.v2 as imageio
from scipy.ndimage import maximum_filter
from scipy.spatial import Delaunay
from .camera import camera_track, camera_track_6dof, fill_smooth, pose
from .config import save_json
from . import instruments as INS


def dil(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((r, r), np.uint8)).astype(bool)


# ---------------------------------------------------------------- canvas geometry
class Canvas:
    """Frame-0 image coordinates extended by M px; texture uv of a canvas pixel."""

    def __init__(self, cam, R0, f0, M):
        self.cam, self.R0, self.f0, self.M = cam, R0, f0, int(M)
        self.Wc, self.Hc = cam.W + 2 * self.M, cam.H + 2 * self.M

    def uv(self, u, v):
        return (np.asarray(u, float) + self.M) / self.Wc, (np.asarray(v, float) + self.M) / self.Hc

    def unproject(self, u, v, z):
        return self.cam.unproject(u, v, z, self.R0, self.f0)

    def project(self, p):
        return self.cam.project(p, self.R0, self.f0)

    def frame_to_canvas(self, img):
        """Paste a frame-sized image into a canvas-sized one (zeros outside)."""
        out = np.zeros((self.Hc, self.Wc) + img.shape[2:], img.dtype)
        out[self.M:self.M + self.cam.H, self.M:self.M + self.cam.W] = img
        return out

    def lookup(self, img, u, v):
        """Sample a canvas image at frame-0 pixel coordinates (clamped)."""
        x = np.clip(np.asarray(u, float).astype(int) + self.M, 0, self.Wc - 1)
        y = np.clip(np.asarray(v, float).astype(int) + self.M, 0, self.Hc - 1)
        return img[y, x]


def canvas_margin(cam, cams, lo=96, hi=480):
    over = 0.0
    for k in range(0, len(cams['R']), 3):
        c = cam.unproject([0, cam.W, 0, cam.W], [0, 0, cam.H, cam.H], 1.0, cams['R'][k], cams['f'][k])
        q, _ = cam.project(c, cams['R'][0], cams['f'][0])
        over = max(over, -q[:, 0].min(), q[:, 0].max() - cam.W, -q[:, 1].min(), q[:, 1].max() - cam.H)
    return int(np.clip(over + 48, lo, hi))


def panorama(cv, cams, frames, cover, Z=None, first=5, pct=30, every=1):
    """Colour (and frame-0 depth) for every canvas pixel from the earliest frames that see it uncovered."""
    cam = cv.cam
    vv, uu = np.mgrid[-cv.M:cam.H + cv.M, -cv.M:cam.W + cv.M]
    rays = cv.unproject(uu.ravel(), vv.ravel(), 1.0)
    N = len(rays)
    col = np.full((first, N, 3), np.nan, np.float32)
    zz = np.full((first, N), np.nan, np.float32)
    cnt = np.zeros(N, int)
    exact = np.zeros(N, bool)
    for k in range(0, len(frames), every):
        q, zc = cam.project(rays, cams['R'][k], cams['f'][k])
        u, v = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (zc > 0) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H) & (cnt < first)
        idx = np.nonzero(ok)[0]
        idx = idx[~cover[k][v[idx], u[idx]]]
        if not len(idx):
            continue
        c = frames[k][v[idx], u[idx]].astype(np.float32)
        z0 = None
        if Z is not None:
            X = cam.unproject(u[idx], v[idx], Z[k][v[idx], u[idx]], cams['R'][k], cams['f'][k])
            z0 = (X - cam.pos) @ cv.R0[2]          # depth along the frame-0 view axis
        if k == 0:                                 # uncovered in frame 0: exactly frame 0
            col[:, idx] = c[None]
            if z0 is not None:
                zz[:, idx] = z0[None]
            exact[idx] = True
            cnt[idx] = first
        else:
            col[cnt[idx], idx] = c
            if z0 is not None:
                zz[cnt[idx], idx] = z0
            cnt[idx] += 1
    seen = cnt > 0
    out = np.zeros((N, 3), np.float32)
    with np.errstate(all='ignore'):
        out[seen] = np.nanpercentile(col[:, seen], pct, axis=0)
    out[exact] = col[0, exact]
    img = out.reshape(cv.Hc, cv.Wc, 3).clip(0, 255).astype(np.uint8)
    never = (~seen).reshape(cv.Hc, cv.Wc)
    M, H, W = cv.M, cam.H, cam.W
    outside = np.ones_like(never)
    outside[M:M + H, M:M + W] = False
    # never seen outside the frame-0 picture (only orbit views show it): mirrored real texture, edge depth;
    # never seen inside it (always under an instrument): inpainted
    mirror = cv2.copyMakeBorder(img[M:M + H, M:M + W], M, M, M, M, cv2.BORDER_REFLECT)
    img[never & outside] = mirror[never & outside]
    if (never & ~outside).any():
        img = cv2.inpaint(img, dil(never & ~outside, 3).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    depth = None
    if Z is not None:
        with np.errstate(all='ignore'):
            d = np.nanmedian(zz, axis=0)
        d = d.reshape(cv.Hc, cv.Wc)
        inner = ~np.isfinite(d) & ~outside
        if inner.any():
            d[M:M + H, M:M + W] = inpaint_depth(np.where(inner, 0, d)[M:M + H, M:M + W], inner[M:M + H, M:M + W])
        edge = cv2.copyMakeBorder(d[M:M + H, M:M + W].astype(np.float32), M, M, M, M, cv2.BORDER_REPLICATE)
        d = np.where(np.isfinite(d), d, edge)
        depth = d.astype(np.float32)
    return img, depth, never


def fill_from_views(cam, cams, frames, Z0, hole, cover, first=5, pct=30):
    """Frame-0 pixels in `hole` (hidden then): their 3D points (frame-0 depth, hole already inpainted) looked up in the
    earliest later frames where nothing covers them; works for a translating scope. Returns image, never-seen mask."""
    H, W = hole.shape
    v0, u0 = np.nonzero(hole)
    X = cam.unproject(u0, v0, Z0[v0, u0], *pose(cams, 0))
    col = np.full((first, len(u0), 3), np.nan, np.float32)
    cnt = np.zeros(len(u0), int)
    for k in range(1, len(frames)):
        q, z = cam.project(X, *pose(cams, k))
        u, v = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (cnt < first)
        idx = np.nonzero(ok)[0]
        idx = idx[~cover[k][v[idx], u[idx]]]
        col[cnt[idx], idx] = frames[k][v[idx], u[idx]]
        cnt[idx] += 1
    img = frames[0].copy()
    seen = cnt > 0
    with np.errstate(all='ignore'):
        img[v0[seen], u0[seen]] = np.nanpercentile(col[:, seen], pct, axis=0).clip(0, 255).astype(np.uint8)
    never = np.zeros((H, W), bool)
    never[v0[~seen], u0[~seen]] = True
    if never.any():
        img = cv2.inpaint(img, dil(never, 3).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    return img, never


def inpaint_depth(z, hole, radius=15):
    scale = 1000.0
    out = cv2.inpaint(np.clip(z * scale, 0, 65535).astype(np.uint16), hole.astype(np.uint8), radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / scale


# ---------------------------------------------------------------- organ observations
def amodal(masks, organ_i, hiders, cams, cam, Z0, n_early=60):
    """Organ extent at frame 0 including what `hiders` (probe, strands) cover then, from the early frames."""
    H, W = cam.H, cam.W
    org0 = masks[0, organ_i]
    cov0 = dil(np.any(masks[0, hiders], axis=0), 5) if hiders else np.zeros_like(org0)
    vv, uu = np.mgrid[0:H, 0:W]
    X0 = cam.unproject(uu.ravel(), vv.ravel(), Z0.ravel(), *pose(cams, 0))
    seen = np.zeros(H * W, bool)
    for k in range(2, min(n_early, len(masks)), 2):
        q, _ = cam.project(X0, *pose(cams, k))
        x, y = q[:, 0].round().astype(int), q[:, 1].round().astype(int)
        ins = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        seen[ins] |= masks[k, organ_i][y[ins], x[ins]]
    am = org0 | (seen.reshape(H, W) & cov0)
    am = cv2.morphologyEx(am.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnt = cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    am = cv2.drawContours(np.zeros_like(am), [max(cnt, key=cv2.contourArea)], -1, 1, -1).astype(bool)
    return am, am & ~org0


def extrapolate_depth(z0, known, fill, behind=None):
    """Depth of `fill` pixels from the depth of `known` pixels only (normalised convolution, growing radius);
    optionally kept at least `behind` (per pixel, metres) behind the measured depth."""
    z = z0.copy()
    w = known.astype(np.float32)
    zf = z0 * w
    done = np.zeros_like(fill)
    for sig in (8, 16, 32, 64, 128):
        num, den = cv2.GaussianBlur(zf, (0, 0), sig), cv2.GaussianBlur(w, (0, 0), sig)
        sel = fill & (den > 1e-3) & ~done
        z[sel] = num[sel] / den[sel]
        done |= sel
    if behind is not None:
        z[fill] = np.maximum(z[fill], z0[fill] + behind[fill])
    return z


def silhouette_thickness(mask, z, F, lo=0.002, hi=0.03):
    """Thickness of a body seen in silhouette, as if its cross-sections were round: 2 sqrt(R^2 - (R - d)^2) with d
    the distance to the outline and R the local half-width."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    R = maximum_filter(dt, size=41)
    t_px = 2 * np.sqrt(np.clip(R ** 2 - (R - dt) ** 2, 0, None))
    return np.clip(t_px * z / F, lo, hi)


def bed_motion(cam, cams, frames, masks, organ_i, holder_is, occluder_is, neck_px=60):
    """Global organ translation per frame (frame-0 pixels): median frame-to-frame dense flow of organ pixels away
    from holding instruments, occluders and specular highlights, scope motion removed, accumulated."""
    n = len(frames)
    if 'pos' in cams:                       # the image-translation estimate assumes a scope that only rotates
        return np.zeros((n, 2))
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    T = np.zeros((n, 2))
    to0 = lambda q, kk: cam.project(cam.unproject(q[:, 0], q[:, 1], 0.1, cams['R'][kk], cams['f'][kk]), cams['R'][0], cams['f'][0])[0]
    for k in range(1, n):
        sel = masks[k - 1, organ_i].copy()
        for h in holder_is:
            sel &= ~dil(masks[k - 1, h], 2 * neck_px + 1)
        if occluder_is:
            sel &= ~dil(np.any(masks[k - 1, occluder_is], 0) | np.any(masks[k, occluder_is], 0), 9)
        sel &= cv2.cvtColor(frames[k - 1], cv2.COLOR_RGB2HSV)[..., 2] < 220
        sel = cv2.erode(sel.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ys, xs = np.nonzero(sel)
        if len(xs) < 50:
            T[k] = T[k - 1]
            continue
        fl = dis.calc(gray[k - 1], gray[k], None)
        idx = np.random.default_rng(k).choice(len(xs), min(400, len(xs)), replace=False)
        a = np.stack([xs[idx], ys[idx]], 1).astype(float)
        b = a + fl[ys[idx], xs[idx]]
        T[k] = T[k - 1] + np.median(to0(b, k) - to0(a, k - 1), axis=0)
    return fill_smooth(T, 9) - fill_smooth(T, 9)[0]


def bed_motion_3d(cam, cams, frames, masks, Z, organ_i, holder_is, occluder_is, neck_px=60):
    """Global organ translation (world, metres) for a moving scope: frame-to-frame dense flow of organ pixels away
    from holding instruments, occluders and highlights, lifted to 3D with each frame's metric depth and camera pose;
    the median 3D step, accumulated and smoothed."""
    n = len(frames)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    T = np.zeros((n, 3))
    for k in range(1, n):
        sel = masks[k - 1, organ_i].copy()
        for h in holder_is:
            sel &= ~dil(masks[k - 1, h], 2 * neck_px + 1)
        if occluder_is:
            sel &= ~dil(np.any(masks[k - 1, occluder_is], 0) | np.any(masks[k, occluder_is], 0), 9)
        sel &= cv2.cvtColor(frames[k - 1], cv2.COLOR_RGB2HSV)[..., 2] < 220
        sel = cv2.erode(sel.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ys, xs = np.nonzero(sel)
        if len(xs) < 50:
            T[k] = T[k - 1]
            continue
        fl = dis.calc(gray[k - 1], gray[k], None)
        idx = np.random.default_rng(k).choice(len(xs), min(400, len(xs)), replace=False)
        a = np.stack([xs[idx], ys[idx]], 1).astype(float)
        b = a + fl[ys[idx], xs[idx]]
        ok = (b[:, 0] >= 0) & (b[:, 0] < cam.W) & (b[:, 1] >= 0) & (b[:, 1] < cam.H)
        a, b = a[ok], b[ok]
        Xa = cam.unproject(a[:, 0], a[:, 1], Z[k - 1][a[:, 1].astype(int), a[:, 0].astype(int)], *pose(cams, k - 1))
        Xb = cam.unproject(b[:, 0], b[:, 1], Z[k][b[:, 1].astype(int), b[:, 0].astype(int)], *pose(cams, k))
        T[k] = T[k - 1] + np.median(Xb - Xa, axis=0)
    return fill_smooth(T, 9) - fill_smooth(T, 9)[0]


# ---------------------------------------------------------------- strands
def ribbon_sheet(cv, band, z, step=7):
    """Textured triangle sheet over a mask at depth z (frame-0 pixels; slivers removed)."""
    H, W = band.shape
    inner = cv2.erode(band.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pts = [(u, v) for v in range(2, H - 2, step) for u in range(int(step / 2) * ((v // step) % 2) + 2, W - 2, step) if inner[v, u]]
    cnt = max(cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    rim = [tuple(map(int, q)) for q in cnt[::max(1, len(cnt) // 90)]]
    if pts:
        G = np.array(pts, float)
        rim = [r for r in rim if np.min(np.linalg.norm(G - r, axis=1)) > 0.6 * step]
    P = np.array(pts + rim, float)
    T = np.array([t for t in Delaunay(P).simplices
                  if band[int(np.clip(P[t].mean(0)[1], 0, H - 1)), int(np.clip(P[t].mean(0)[0], 0, W - 1))] and min_angle(P[t]) > 18])
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, T = P[used], remap[T]
    zz = z[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)]
    return dict(px=P, X=cv.unproject(P[:, 0], P[:, 1], zz), tri=T)


def min_angle(tri_pts):
    a, b, c = tri_pts
    out = []
    for p0, p1, p2 in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = p1 - p0, p2 - p0
        out.append(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))
    return min(out)


def strand_ribbon(cv, mask, z, organ_mask, n_anchors=2):
    """Ribbon over a strand: vertices near its far ends (farthest from the organ) fixed, near the organ tied to it."""
    from scipy.cluster.vq import kmeans2
    R = ribbon_sheet(cv, mask, z)
    P = R['px']
    ys, xs = np.nonzero(mask)
    dorg = cv2.distanceTransform((~organ_mask).astype(np.uint8), cv2.DIST_L2, 5)
    dv_all = dorg[ys, xs]
    far = np.stack([xs, ys], 1)[dv_all > np.percentile(dv_all, 92)].astype(float)
    anchors = kmeans2(far, n_anchors, minit='++', seed=0)[0] if len(far) > 10 else far
    pinned = np.zeros(len(P), bool)
    for a in anchors:
        pinned |= np.linalg.norm(P - a, axis=1) < 14
    dv = dorg[np.clip(P[:, 1].astype(int), 0, cv.cam.H - 1), np.clip(P[:, 0].astype(int), 0, cv.cam.W - 1)]
    # the end nearest the organ is tied to it: a band of vertices (not a single one, which the organ's motion would
    # pull out of the sheet as a spike); the strand often stops short of the organ mask (jaws on the neck in between)
    dv_free = np.where(pinned, np.inf, dv)
    tied = (dv_free <= dv_free.min() + 20) & ~pinned
    if tied.sum() < 8:
        tied[np.argsort(dv_free)[:8]] = True
    R.update(pinned=pinned, tied=tied & ~pinned)
    return R


# ---------------------------------------------------------------- backdrop mesh
def seen_mask(cv, cams):
    """Canvas pixels that some frame of the clip actually looked at (union of the frames' footprints)."""
    m = np.zeros((cv.Hc, cv.Wc), np.uint8)
    cam = cv.cam
    for k in range(len(cams['R'])):
        c = cam.unproject([0, cam.W, cam.W, 0], [0, 0, cam.H, cam.H], 1.0, cams['R'][k], cams['f'][k])
        q, _ = cv.project(c)
        cv2.fillConvexPoly(m, (q + cv.M).astype(np.int32), 1)
    return m.astype(bool)


def backdrop_obj(path, cv, zb, step=8, behind=0.004, seen=None):
    """Background mesh over the canvas; only where some frame looked (no geometry is invented outside)."""
    us = np.arange(-cv.M, cv.cam.W + cv.M + 1, step)
    vs = np.arange(-cv.M, cv.cam.H + cv.M + 1, step)
    seen_d = None if seen is None else dil(seen, 2 * step + 1)
    ok = (lambda u, v: True) if seen is None else (lambda u, v: bool(cv.lookup(seen_d, u, v)))
    lines, idx = [], {}
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            if not ok(u, v):
                continue
            p = cv.unproject(u, v, float(cv.lookup(zb, u, v)) + behind)
            lines.append('v %.6f %.6f %.6f' % tuple(p))
            tu, tv = cv.uv(u, v)
            lines.append('vt %.6f %.6f' % (tu, 1 - tv))
            idx[i, j] = len(idx) + 1
    for j in range(len(vs) - 1):
        for i in range(len(us) - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if not all(c in idx for c in corners):
                continue
            a, b, c, d = (idx[x] for x in corners)
            lines.append(f'f {a}/{a} {c}/{c} {b}/{b}')
            lines.append(f'f {a}/{a} {d}/{d} {c}/{c}')
    path.write_text('\n'.join(lines))


def keyframe_patches(out, cam, cams, frames, masks, Z, hide_is, zb0, every=25, step=8, behind=0.004):
    """Background for a scope that translates: textured depth patches from keyframes, each placed with its own pose;
    a keyframe adds only what the earlier patches do not already cover (with a few pixels of overlap)."""
    n, H, W = len(frames), cam.H, cam.W
    keys = list(range(0, n, every)) + ([n - 1] if (n - 1) % every else [])
    patches = []
    for k in keys:
        R, f, pos = pose(cams, k)
        cov = np.zeros((H, W), np.uint8)
        for X, F in patches:
            q, z = cam.project(X, R, f, pos)
            for t in F:
                if (z[t] > 0).all():
                    cv2.fillConvexPoly(cov, q[t].astype(np.int32), 1)
        new = ~cv2.erode(cov, np.ones((9, 9), np.uint8)).astype(bool)
        if patches and new.mean() < 0.02:
            continue
        hide = dil(np.any(masks[k, hide_is], 0), 15)
        img = cv2.inpaint(frames[k], hide.astype(np.uint8), 9, cv2.INPAINT_TELEA)
        z = zb0 if k == 0 else cv2.GaussianBlur(inpaint_depth(Z[k], dil(hide, 7)), (0, 0), 4)
        us = np.r_[np.arange(0, W - 1, step), W - 1]
        vs = np.r_[np.arange(0, H - 1, step), H - 1]
        grow = dil(new, 2 * step + 1)
        idx, V, VT, F = {}, [], [], []
        for j, v in enumerate(vs):
            for i, u in enumerate(us):
                if grow[v, u]:
                    idx[i, j] = len(V)
                    V.append(cam.unproject(u, v, z[v, u] + behind, R, f, pos))
                    VT.append((u / (W - 1), 1 - v / (H - 1)))
        for j in range(len(vs) - 1):
            for i in range(len(us) - 1):
                c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                if all(x in idx for x in c):
                    a, b, cc, d = (idx[x] for x in c)
                    F += [(a, cc, b), (a, d, cc)]
        if not F:
            continue
        m = len(patches)
        V, F = np.array(V), np.array(F)
        lines = ['v %.6f %.6f %.6f' % tuple(x) for x in V] + ['vt %.6f %.6f' % t for t in VT] + \
                [f'f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}' for a, b, c in F]
        (out / f'backdrop_{m}.obj').write_text('\n'.join(lines))
        imageio.imwrite(out / f'tex_back_{m}.png', img)
        patches.append((V, F))
    return len(patches), keys


# ---------------------------------------------------------------- build
def track_camera(clip, frames, masks, Z=None):
    """Scope motion; everything that is not organ, strand or instrument counts as static anatomy."""
    cam = clip.camera()
    n = len(frames)
    moving_is = [clip.obj_index(i['mask']) for i in clip.instruments] + [clip.obj_index(s) for s in clip.role('strand')] + \
                [clip.obj_index(clip.organ)]
    if clip['camera'].get('static'):
        return dict(R=np.repeat(cam.R[None], n, 0), f=np.full(n, cam.F), params=np.zeros((n, 4)), fit_err_px=np.zeros(n))
    static = np.stack([~dil(np.any(masks[k, moving_is], 0), 15) for k in range(n)])
    if clip['camera'].get('motion') == '6dof':
        return camera_track_6dof(cam, frames, static, Z)
    return camera_track(cam, frames, None, static_masks=static)


def build(clip, frames, masks, Z, cams, log=print):
    cam = clip.camera()
    n, H, W = len(frames), cam.H, cam.W
    oi = clip.obj_index(clip.organ)
    instr_is = [clip.obj_index(i['mask']) for i in clip.instruments]
    strand_is = [clip.obj_index(s) for s in clip.role('strand')]
    holder_is = [clip.obj_index(i['mask']) for i in clip.instruments if i.get('holds')]
    log(f'[scene] camera: max rotation {np.degrees(np.linalg.norm(cams["params"][:, :3], axis=1)).max():.1f} deg, '
        f'zoom {np.exp(cams["params"][:, 3]).min():.3f}-{np.exp(cams["params"][:, 3]).max():.3f}, median fit {np.median(cams["fit_err_px"][1:]):.1f} px')
    R0, f0 = cams['R'][0], cams['f'][0]
    six = 'pos' in cams
    M = 0 if six else canvas_margin(cam, cams)
    cv = Canvas(cam, R0, f0, M)
    cover_i = np.stack([dil(np.any(masks[k, instr_is], 0), 15) for k in range(n)])
    if six:
        # a translating scope sees the scene from different places: no single-viewpoint panorama. The organ and strand
        # textures come from frame 0 (instruments inpainted); the background is built from keyframe patches below
        zpano = inpaint_depth(Z[0], dil(cover_i[0], 9))
        pano, never = fill_from_views(cam, cams, frames, zpano, cover_i[0], cover_i)
        if strand_is:
            cover_s = np.stack([cover_i[k] | dil(np.any(masks[k, strand_is], 0), 7) for k in range(n)])
            pano_s, never_s = fill_from_views(cam, cams, frames, zpano, cover_s[0], cover_s, pct=50)
        else:
            pano_s, never_s = pano, never
        log(f'[scene] moving scope: frame-0 textures filled from later views ({never.mean() * 100:.1f} % never uncovered), '
            'keyframe background patches')
    else:
        # --- panoramas: instruments removed (texture + depth), and instruments + strands removed (tissue under strands)
        cover_s = np.stack([cover_i[k] | dil(np.any(masks[k, strand_is], 0), 7) for k in range(n)]) if strand_is else cover_i
        pano, zpano, never = panorama(cv, cams, frames, cover_i, Z, first=5, pct=30)
        pano_s, _, never_s = panorama(cv, cams, frames, cover_s, None, first=5, pct=50) if strand_is else (pano, None, never)
        log(f'[scene] canvas margin {M} px; never seen {never.mean() * 100:.1f} % of canvas')
    # --- organ observations at frame 0
    am, added = amodal(masks, oi, [clip.obj_index(i['mask']) for i in clip.instruments if not i.get('holds')] + strand_is, cams, cam, Z[0])
    z0 = cv2.bilateralFilter(Z[0].astype(np.float32), 9, 0.004, 5)
    hid_by = np.zeros((H, W), np.float32)
    for ins in clip.instruments:
        if not ins.get('holds'):
            hid_by[masks[0, clip.obj_index(ins['mask'])]] = 0.003
    for s in strand_is:
        hid_by[masks[0, s] & (hid_by == 0)] = 0.002
    z0 = extrapolate_depth(z0, masks[0, oi], added, hid_by)
    thick = silhouette_thickness(am, z0, f0)
    # --- background surface: organ, strands and (widely dilated) instruments filled in
    zc = zpano.copy()
    org_c = cv.frame_to_canvas(am)
    str_c = cv.frame_to_canvas(np.any(masks[0, strand_is], 0)) if strand_is else np.zeros_like(org_c)
    ins_c = cv.frame_to_canvas(dil(np.any(masks[0, instr_is], 0), 25))
    zc[cv.M:cv.M + H, cv.M:cv.M + W] = np.where(np.any(masks[0, instr_is], 0), zc[cv.M:cv.M + H, cv.M:cv.M + W], z0)
    hole = dil(org_c | str_c | ins_c, 7)
    zb = cv2.GaussianBlur(inpaint_depth(zc, hole), (0, 0), 6)
    zb = np.maximum(zb, np.where(org_c, cv.frame_to_canvas(z0 + thick), zb))
    # --- textures
    tex_ribbon = pano
    under = cv.frame_to_canvas(dil(np.any(masks[0, strand_is], 0), 7)) if strand_is else np.zeros_like(org_c)
    tex_tissue = pano.copy()
    good = under & ~never_s
    tex_tissue[good] = pano_s[good]
    rest = under & never_s
    if rest.any():
        tex_tissue = cv2.inpaint(tex_tissue, rest.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    tex_back = cv2.inpaint(tex_tissue, dil(org_c | str_c, 5).astype(np.uint8), 15, cv2.INPAINT_TELEA)
    for name, img in (('tex_tissue', tex_tissue), ('tex_back', tex_back), ('tex_ribbon', tex_ribbon)):
        imageio.imwrite(clip.scene / f'{name}.png', img)
    backdrop_obj(clip.scene / 'backdrop.obj', cv, zb, seen=seen_mask(cv, cams))
    n_patches = 0
    if six:
        n_patches, keys = keyframe_patches(clip.scene, cam, cams, frames, masks, Z, instr_is + strand_is + [oi], zb)
        log(f'[scene] background: {n_patches} keyframe patches (keyframes {keys})')
    # --- strands
    ribbons = [strand_ribbon(cv, masks[0, s], z0, am) for s in strand_is]
    # --- instruments
    inst = INS.recover(clip, cam, cams, masks, Z)
    acts = INS.actions(inst, n, clip['fps'])
    for k, v in inst.items():
        log(f'[scene] {k}: port {v["port_mode"]} (width ratio {v["width_ratio"]:.2f}), axis error {v["axis_err_deg"]:.2f} deg, '
            f'visible {v["visible"].mean() * 100:.0f} % of frames')
    # --- observed global organ motion
    occl = [clip.obj_index(i['mask']) for i in clip.instruments if not i.get('holds')] + strand_is
    use_bed = clip.get('sim', {}).get('bed_motion', True)
    bedT = bed_motion(cam, cams, frames, masks, oi, holder_is, occl) if use_bed and not six else np.zeros((n, 2))
    bed3d = bed_motion_3d(cam, cams, frames, masks, Z, oi, holder_is, occl) if use_bed and six else np.zeros((n, 3))
    log(f'[scene] organ bed motion: max {np.abs(bedT).max():.1f} px, {np.linalg.norm(bed3d, axis=1).max() * 1000:.1f} mm (3D)')
    np.savez(clip.scene / 'scene.npz', cam_R=cams['R'], cam_f=cams['f'], cam_fit_err=cams['fit_err_px'], M=M,
             **({'cam_pos': cams['pos']} if six else {}),
             organ_mask=am, organ_added=added, organ_visible=masks[0, oi], z0=z0, thick=thick, zb=zb, bedT=bedT, bed3d=bed3d, actions=acts,
             **{f'ribbon{j}_{k}': v for j, r in enumerate(ribbons) for k, v in r.items()}, n_ribbons=len(ribbons))
    meta = dict(canvas_margin=M, backdrop_patches=n_patches, camera_moves=six, organ_amodal_added_px=int(added.sum()), organ_depth_cm=[round(float(np.percentile(z0[am], 5)) * 100, 1), round(float(np.percentile(z0[am], 95)) * 100, 1)],
                organ_max_thickness_cm=round(float(thick[am].max()) * 100, 1), bed_motion_max_px=round(float(np.abs(bedT).max()), 1),
                bed_motion_max_mm=round(float(np.linalg.norm(bed3d, axis=1).max()) * 1000, 1),
                camera=dict(max_translation_mm=round(float(np.linalg.norm(cams['pos'] - cams['pos'][0], axis=1).max()) * 1000, 1) if six else 0.0,
                            max_rot_deg=round(float(np.degrees(np.linalg.norm(cams['params'][:, :3], axis=1)).max()), 2),
                            zoom=[round(float(np.exp(cams['params'][:, 3]).min()), 3), round(float(np.exp(cams['params'][:, 3]).max()), 3)],
                            median_fit_px=round(float(np.median(cams['fit_err_px'][1:])), 2)),
                instruments={k: dict(rcm=v['rcm'].round(4).tolist(), heading=round(v['heading'], 4), axis_err_deg=round(v['axis_err_deg'], 2),
                                     port_mode=v['port_mode'], width_ratio=round(v['width_ratio'], 2), visible_frac=round(float(v['visible'].mean()), 2))
                             for k, v in inst.items()},
                ribbons=[dict(vertices=len(r['X']), pinned=int(r['pinned'].sum()), tied=int(r['tied'].sum())) for r in ribbons],
                n_frames=n, actions_shape=list(acts.shape))
    save_json(clip.scene / 'scene.json', meta)
    return meta
