"""Backdrop of shot A: one static, textured surface of everything that does not move (liver + surroundings), fused from
all 251 views, with the liver as a labelled part (optionally template-fitted).

    python -m r2s.tissue.backdrop vNN            build + evaluate  -> outputs/iter/tissues/backdrop/vNN/
    python -m r2s.tissue.backdrop masks          (re)segment the liver with SAM 2.1 (cached in tissues/backdrop/masks.npz)

Pipeline (see NOTES.md of each version for the iteration history)
1. static pixels per frame: not gallbladder, strand, instruments (dilated) and not the zone around the grasper where the
   peritoneal sheet is tented; optional temporal-consistency rejection of pixels that disagree with the fused surface.
2. fusion in a virtual reference camera (mean pose of the clip, canvas wide enough for every frame's footprint): every
   frame's static pixels are back-projected with that frame's camera and depth, re-projected into the reference
   camera and binned; per cell a robust median across frames (+ spread = 1.4826 MAD across frames).
3. surface = height field over the reference canvas (depth along the reference optical axis), hidden cells
   (behind the gallbladder in every frame = gallbladder bed, or always under an instrument) filled smoothly
   (membrane / biharmonic-like relaxation from the observed rim), meshed as a regular triangle grid.
4. texture = image over the reference canvas (UV = canvas pixel / canvas size): per texel the median colour of the
   frames that see that surface point as static and unoccluded (sharp frames preferred over the blurred opening pan),
   unobserved texels inpainted.
5. per-frame evaluation with an own z-buffer rasteriser (perspective-correct barycentrics): depth residual, coverage,
   photometric L1 / NCC of the textured render against the video, free-space violations, multi-view spread.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from r2s import views, quality

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iter/tissues/backdrop'
LIVER_STL = ROOT / 'data/templates/bodyparts3d/liver.stl'

# SAM 2.1 prompts (x, y) for the liver: the pink organ on the left and top (the clip's own 'liver' mask only covers a
# sliver at the top right and is not used).
LIVER_PROMPTS = [
    dict(frame=0, pos=[[30, 170], [150, 30], [25, 300]], neg=[[200, 280], [560, 250], [470, 200], [600, 60], [330, 300]]),
    dict(frame=100, pos=[[60, 150], [40, 300], [120, 40], [280, 40]],
         neg=[[220, 250], [580, 150], [430, 300], [300, 345], [470, 60], [380, 210]]),
    dict(frame=200, pos=[[60, 200], [100, 60], [40, 320]], neg=[[250, 280], [600, 280], [430, 320], [480, 60]]),
]


# pink strands between the gallbladder neck and the right organs (not in the clip's masks; they move with the neck)
STRAND_PROMPTS = [
    dict(frame=0, pos=[[370, 185], [400, 180], [355, 190]], neg=[[450, 240], [380, 130], [300, 300]]),
    dict(frame=100, pos=[[465, 150], [490, 155], [440, 145]], neg=[[470, 100], [520, 200], [400, 280], [560, 130]]),
    dict(frame=200, pos=[[490, 140], [515, 145], [470, 130]], neg=[[500, 190], [560, 100], [420, 260]]),
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def dil(m, r):
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def ero(m, r):
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(m.astype(np.uint8), k).astype(bool)


# ---------------------------------------------------------------- masks
def liver_masks(V, redo=False):
    path = OUT / 'masks.npz'
    if path.exists() and not redo:
        d = np.load(path)
        return np.unpackbits(d['liver'], axis=-1)[..., :V.W].astype(bool)
    m = views.segment_object(V, LIVER_PROMPTS)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, liver=np.packbits(m, axis=-1), prompts=json.dumps(LIVER_PROMPTS))
    return m


def sheet_zone(V, k, r):
    """Disk of radius r px around the grasper's jaw end in frame k (the tented peritoneal sheet hangs there)."""
    m = V.mask('grasper')[k]
    z = np.zeros((V.H, V.W), bool)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return z
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    a = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ a
    # the shaft enters at the image border: orient the axis away from the border end
    e0, e1 = P[np.argmin(t)], P[np.argmax(t)]
    bd = lambda p: min(p[0], p[1], V.W - 1 - p[0], V.H - 1 - p[1])
    tip = e1 if bd(e1) > bd(e0) else e0
    cv2.circle(z.view(np.uint8), (int(tip[0]), int(tip[1])), int(r), 1, -1)
    return z


def static_masks(V, p):
    """(n, H, W) bool: pixels of static anatomy (conservative)."""
    gb, st, gr, pr = (V.mask(n) for n in ('gallbladder', 'strand', 'grasper', 'probe'))
    pink = strand_masks(V) if p.get('pink_r') is not None else None
    S = np.zeros((V.n, V.H, V.W), bool)
    b = p.get('border', 3)
    for k in range(V.n):
        mov = dil(gb[k], p['gb_r']) | dil(st[k], p['strand_r']) | dil(gr[k] | pr[k], p['tool_r'])
        if pink is not None:
            mov |= dil(pink[k], p['pink_r'])
        if p.get('sheet_r'):
            mov |= sheet_zone(V, k, p['sheet_r'])
        mov[:b] = mov[-b:] = True
        mov[:, :b] = mov[:, -b:] = True
        S[k] = ~mov
    return S


def specular(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return dil((hsv[..., 2] > 225) & (hsv[..., 1] < 90), 2)


def sharpness(V):
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in V.frames]
    return np.array([cv2.Laplacian(cv2.GaussianBlur(x, (3, 3), 0), cv2.CV_32F).var() for x in g])


# ---------------------------------------------------------------- reference camera / canvas
class Ref:
    """Virtual reference camera (mean pose of the clip) with a canvas that covers every frame's footprint.
    Canvas pixel (uc, vc) <-> reference image coordinate (uc + ou, vc + ov) of the 640x360 camera model."""

    def __init__(self, V, margin=6, cell=2):
        U, _, Wt = np.linalg.svd(V.R.mean(0))
        R = U @ Wt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Wt
        self.R, self.pos, self.f, self.cam = R, V.pos.mean(0), float(np.median(V.f)), V.cam
        us, vs = [], []
        for k in range(V.n):
            ys, xs = np.mgrid[0:V.H:12, 0:V.W:12]
            ys, xs = np.r_[ys.ravel(), 0, 0, V.H - 1, V.H - 1], np.r_[xs.ravel(), 0, V.W - 1, 0, V.W - 1]
            q, _ = self.cam.project(V.unproject(xs, ys, V.depth(k)[ys, xs], k), self.R, self.f, self.pos)
            us.append(q[:, 0]), vs.append(q[:, 1])
        us, vs = np.concatenate(us), np.concatenate(vs)
        self.ou = cell * np.floor((us.min() - margin) / cell)
        self.ov = cell * np.floor((vs.min() - margin) / cell)
        self.Wc = int(cell * np.ceil((us.max() + margin - self.ou) / cell))
        self.Hc = int(cell * np.ceil((vs.max() + margin - self.ov) / cell))

    def project(self, X):
        q, z = self.cam.project(X, self.R, self.f, self.pos)
        return q[..., 0] - self.ou, q[..., 1] - self.ov, z

    def unproject(self, uc, vc, z):
        return self.cam.unproject(np.asarray(uc, float) + self.ou, np.asarray(vc, float) + self.ov, z, self.R, self.f, self.pos)

    def as_dict(self):
        return dict(R=self.R, pos=self.pos, f=self.f, ou=self.ou, ov=self.ov, Wc=self.Wc, Hc=self.Hc,
                    cx=self.cam.W / 2, cy=self.cam.H / 2)


# ---------------------------------------------------------------- depth fusion
def sorted_median(D):
    """Median and count over axis 0 ignoring NaN (D float32 (n, m))."""
    cnt = np.isfinite(D).sum(0)
    S = np.sort(D, 0)                    # NaN last
    i = np.arange(D.shape[1])
    lo, hi = np.maximum((cnt - 1) // 2, 0), np.maximum(cnt // 2, 0)
    med = 0.5 * (S[lo, i] + S[hi, i])
    med[cnt == 0] = np.nan
    return med, cnt


def fuse_depth(V, ref, S, p, liver=None):
    cs = p['cell']
    Hg, Wg = ref.Hc // cs, ref.Wc // cs
    m = Hg * Wg
    D = np.full((V.n, m), np.nan, np.float32)
    foot = np.zeros(m, np.int32)
    gbz = np.full((V.n, m), np.nan, np.float32)           # gallbladder front depth (reference z) per frame
    liv = np.zeros(m, np.float32)
    ys, xs = np.mgrid[0:V.H, 0:V.W]
    ys, xs = ys.ravel(), xs.ravel()
    gbm = V.mask('gallbladder')
    for k in range(V.n):
        X = V.unproject(xs, ys, V.depth(k)[ys, xs], k)
        uc, vc, zr = ref.project(X)
        ci, cj = np.floor(uc / cs).astype(int), np.floor(vc / cs).astype(int)
        inb = (ci >= 0) & (ci < Wg) & (cj >= 0) & (cj < Hg)
        cell = cj * Wg + ci
        foot[np.unique(cell[inb])] += 1
        st = inb & S[k].ravel()
        c = cell[st]
        n_ = np.bincount(c, minlength=m)
        s_ = np.bincount(c, zr[st], minlength=m)
        hit = n_ > 0
        D[k, hit] = s_[hit] / n_[hit]
        if liver is not None:
            liv += np.bincount(c, liver[k].ravel()[st], minlength=m) / np.maximum(n_, 1)
        g = inb & ero(gbm[k], 4).ravel()
        if g.any():
            n_ = np.bincount(cell[g], minlength=m)
            s_ = np.bincount(cell[g], zr[g], minlength=m)
            gbz[k, n_ > 0] = s_[n_ > 0] / n_[n_ > 0]
    return dict(D=D, foot=foot, gbz=gbz, liv=liv, Hg=Hg, Wg=Wg)


def robust_fuse(F, p, w=None):
    """Median across frames, then (optionally) reject observations far from it and re-median."""
    D = F['D'].copy()
    if w is not None:                                    # drop low-weight (blurred) frames where enough sharp ones see
        D_sharp = D[w]
        cnt_sharp = np.isfinite(D_sharp).sum(0)
        use_blur = cnt_sharp < p.get('min_sharp', 5)
        D[np.ix_(~w, ~use_blur)] = np.nan
    med, cnt = sorted_median(D)
    for _ in range(p.get('reject_iters', 0)):
        mad = sorted_median(np.abs(D - med))[0] * 1.4826
        thr = np.maximum(p['reject_k'] * np.nan_to_num(mad, nan=1.0), p['reject_min'])
        bad = np.abs(D - med) > thr
        D[bad] = np.nan
        med, cnt = sorted_median(D)
    mad = sorted_median(np.abs(D - med))[0] * 1.4826
    return med, cnt, mad, D


def grid_laplacian(dom):
    """4-neighbour graph Laplacian over the cells of `dom` (H, W bool): (L sparse (n, n), index image)."""
    import scipy.sparse as sp
    H, W = dom.shape
    idx = -np.ones((H, W), np.int64)
    idx[dom] = np.arange(dom.sum())
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[:H - dy, :W - dx]
        b = idx[dy:, dx:]
        ok = (a >= 0) & (b >= 0)
        rows += [a[ok], b[ok]]
        cols += [b[ok], a[ok]]
    r, c = np.concatenate(rows), np.concatenate(cols)
    n = int(dom.sum())
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    return L.tocsr(), idx


def fill_surface(Z, known, dom, mode='harmonic', lower=None, iters=3):
    """Fill the unknown cells of Z inside dom smoothly from the known ones (harmonic or biharmonic); optionally keep
    the filled cells at or behind `lower` (reference depth)."""
    import scipy.sparse.linalg as spla
    L, idx = grid_laplacian(dom)
    A = L if mode == 'harmonic' else (L.T @ L).tocsr()
    z = Z[dom].astype(float)
    fixed = known[dom].copy()
    for it in range(iters):
        h = ~fixed
        if not h.any():
            break
        zf = z.copy()
        zf[h] = 0
        rhs = -(A[h][:, fixed] @ zf[fixed])
        z[h] = spla.spsolve(A[h][:, h].tocsc(), rhs)
        if lower is None:
            break
        lo = lower[dom]
        viol = h & np.isfinite(lo) & (z < lo)
        if not viol.any():
            break
        z[viol] = lo[viol]
        fixed = fixed | viol
    out = np.full(Z.shape, np.nan)
    out[dom] = z
    return out


def fit_surface(Z, w, dom, lam1, lam2, lower=None, iters=4):
    """Confidence-weighted smooth height field over dom: min sum w (z - Z)^2 + lam1 |grad z|^2 + lam2 |lap z|^2;
    cells with w = 0 are filled (membrane + thin plate); optionally z >= lower (heavy-weight constraint, iterated)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    L, idx = grid_laplacian(dom)
    Q = (lam1 * L + lam2 * (L.T @ L)).tocsr()
    d = Z[dom].astype(float)
    ww = w[dom].astype(float)
    lo = lower[dom] if lower is not None else None
    for it in range(iters):
        A = (sp.diags(ww) + Q).tocsc()
        z = spla.spsolve(A, ww * d)
        if lo is None:
            break
        viol = np.isfinite(lo) & (z < lo - 1e-4)
        if not viol.any():
            break
        d = np.where(viol, lo + 5e-4, d)
        ww = np.where(viol, np.maximum(ww, 20.0), ww)
    out = np.full(Z.shape, np.nan)
    out[dom] = z
    return out


def norm_smooth(Z, known, sigma):
    if sigma <= 0:
        return Z
    a = cv2.GaussianBlur(np.where(known, Z, 0).astype(np.float32), (0, 0), sigma)
    b = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    out = Z.copy()
    out[known] = (a / np.maximum(b, 1e-6))[known]
    return out


def largest_cc(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    if n <= 1:
        return m
    return lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])


def fill_holes(m):
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    out = m.copy()
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    for i in range(1, n):
        if i not in edge:
            out[lab == i] = True
    return out


def sample_grid(G, cs, uc, vc, interp=cv2.INTER_LINEAR):
    """Values of a cell grid G (cell size cs canvas px) at canvas coords."""
    mx = ((np.asarray(uc, np.float32) - (cs - 1) / 2) / cs).astype(np.float32)
    my = ((np.asarray(vc, np.float32) - (cs - 1) / 2) / cs).astype(np.float32)
    return remap_pts(G.astype(np.float32), mx, my, interp)


def remap_pts(img, x, y, interp=cv2.INTER_LINEAR):
    """cv2.remap at arbitrary-shaped point arrays (works around the 32767 map size limit)."""
    sh = np.shape(x)
    x, y = np.asarray(x, np.float32).ravel(), np.asarray(y, np.float32).ravel()
    n, w = len(x), 4096
    pad = (-n) % w
    xm = np.r_[x, np.zeros(pad, np.float32)].reshape(-1, w)
    ym = np.r_[y, np.zeros(pad, np.float32)].reshape(-1, w)
    out = cv2.remap(img, xm, ym, interp, borderMode=cv2.BORDER_REPLICATE)
    out = out.reshape((-1,) + img.shape[2:])[:n]
    return out.reshape(sh + img.shape[2:])


# ---------------------------------------------------------------- rasteriser
def raster(q, z, F, H, W, max_box=512):
    """Z-buffer rasterisation: q (N, 2) pixel coords, z (N,) camera depth, F (M, 3).
    Returns zbuf (H, W) (inf = empty), fid (H, W) (-1), bary (H, W, 3) perspective-correct barycentrics."""
    F = np.asarray(F)
    P, Z = q[F], z[F]
    ok = (Z > 1e-4).all(1) & np.isfinite(P).all((1, 2))
    x0 = np.floor(P[..., 0].min(1)).clip(-1, W)
    x1 = np.ceil(P[..., 0].max(1)).clip(-1, W)
    y0 = np.floor(P[..., 1].min(1)).clip(-1, H)
    y1 = np.ceil(P[..., 1].max(1)).clip(-1, H)
    ok &= (x1 >= 0) & (x0 <= W - 1) & (y1 >= 0) & (y0 <= H - 1)
    x0, y0 = np.maximum(x0, 0), np.maximum(y0, 0)
    x1, y1 = np.minimum(x1, W - 1), np.minimum(y1, H - 1)
    size = np.maximum(x1 - x0, y1 - y0) + 1
    ok &= size <= max_box
    buckets = np.where(ok, 2 ** np.ceil(np.log2(np.maximum(size, 1))).astype(int), 0)
    PIX, ZZ, FI, B = [], [], [], []
    for b in np.unique(buckets[ok]):
        fs = np.nonzero(buckets == b)[0]
        oy, ox = np.mgrid[0:b, 0:b]
        ox, oy = ox.ravel(), oy.ravel()
        chunk = max(1, int(3e6 // (b * b)))
        for s in range(0, len(fs), chunk):
            f = fs[s:s + chunk]
            px = x0[f, None] + ox[None]
            py = y0[f, None] + oy[None]
            A, Bv, C = P[f, 0], P[f, 1], P[f, 2]
            v0x, v0y = (Bv - A)[:, 0:1], (Bv - A)[:, 1:2]
            v1x, v1y = (C - A)[:, 0:1], (C - A)[:, 1:2]
            v2x, v2y = px - A[:, 0:1], py - A[:, 1:2]
            d = v0x * v1y - v1x * v0y
            d = np.where(np.abs(d) < 1e-12, np.nan, d)
            b1 = (v2x * v1y - v1x * v2y) / d
            b2 = (v0x * v2y - v2x * v0y) / d
            b0 = 1 - b1 - b2
            e = -1e-6
            ins = (b0 >= e) & (b1 >= e) & (b2 >= e) & (px <= x1[f, None]) & (py <= y1[f, None])
            if not ins.any():
                continue
            fi = np.broadcast_to(f[:, None], ins.shape)[ins]
            b0, b1, b2 = b0[ins], b1[ins], b2[ins]
            zf = Z[fi]
            iz = b0 / zf[:, 0] + b1 / zf[:, 1] + b2 / zf[:, 2]
            zp = 1 / iz
            PIX.append((py[ins] * W + px[ins]).astype(np.int64))
            ZZ.append(zp)
            FI.append(fi)
            B.append(np.stack([b0 / zf[:, 0], b1 / zf[:, 1], b2 / zf[:, 2]], 1) * zp[:, None])
    zbuf = np.full(H * W, np.inf)
    fid = np.full(H * W, -1, np.int64)
    bary = np.zeros((H * W, 3))
    if PIX:
        pix, zz, fi, bb = np.concatenate(PIX), np.concatenate(ZZ), np.concatenate(FI), np.concatenate(B)
        o = np.lexsort((zz, pix))
        pix, zz, fi, bb = pix[o], zz[o], fi[o], bb[o]
        first = np.r_[True, pix[1:] != pix[:-1]]
        zbuf[pix[first]] = zz[first]
        fid[pix[first]] = fi[first]
        bary[pix[first]] = bb[first]
    return zbuf.reshape(H, W), fid.reshape(H, W), bary.reshape(H, W, 3)


def interp_attr(F, fid, bary, attr):
    """Per-pixel attribute (H, W, C) from per-vertex attr (N, C)."""
    m = fid >= 0
    out = np.zeros(fid.shape + (attr.shape[1],), np.float32)
    f = np.asarray(F)[fid[m]]
    out[m] = (bary[m][..., None] * attr[f]).sum(1)
    return out


def render_textured(q, z, F, uv, tex, H, W):
    """-> mask, depth, rgb (H, W, 3 float), fid. uv in [0, 1] with v up (OBJ convention)."""
    zb, fid, bary = raster(q, z, F, H, W)
    m = fid >= 0
    UV = interp_attr(F, fid, bary, uv)
    th, tw = tex.shape[:2]
    mx = (UV[..., 0] * tw - 0.5).astype(np.float32)
    my = ((1 - UV[..., 1]) * th - 0.5).astype(np.float32)
    rgb = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    rgb[~m] = 0
    return m, zb, rgb, fid


# ---------------------------------------------------------------- texture
def masked_median(C, sel):
    """Per-column median of the valid (< 999) entries of C[sel] (k, N, 3) uint16 -> (N, 3) float, count (N,)."""
    C = C[sel]
    cnt = (C[..., 0] < 999).sum(0)
    Cs = np.sort(C, 0)
    i = np.arange(C.shape[1])
    lo, hi = np.maximum((cnt - 1) // 2, 0), np.maximum(cnt // 2, 0)
    med = 0.5 * (Cs[lo, i].astype(np.float32) + Cs[hi, i])
    med[cnt == 0] = np.nan
    return med, cnt


def push_pull(img, known, levels=9):
    """Fill unknown pixels with a smooth multi-scale average of the known ones (push-pull)."""
    img = img.astype(np.float32)
    w = known.astype(np.float32)
    pyr = [(img * w[..., None], w)]
    for _ in range(levels):
        a, b = pyr[-1]
        if min(b.shape) < 4:
            break
        pyr.append((cv2.pyrDown(a), cv2.pyrDown(b)))
    a, b = pyr[-1]
    est = a / np.maximum(b, 1e-6)[..., None]
    for a, b in pyr[-2::-1]:
        up = cv2.pyrUp(est, dstsize=(b.shape[1], b.shape[0]))
        cur = a / np.maximum(b, 1e-6)[..., None]
        t = np.clip(b * 4, 0, 1)[..., None]           # trust the level where it has enough support
        est = t * cur + (1 - t) * up
    out = img.copy()
    out[~known] = est[~known]
    return out


def fuse_texture(V, ref, Zc, domc, S, p, sharp_w):
    """Texture over the reference canvas: per texel the median colour of the frames that see the point as static
    and unoccluded (depth test); per-frame colour gains equalised against a first fused texture; sharp frames
    preferred, blended with the blurred-pan median where few sharp frames see the texel; holes push-pull filled."""
    ys, xs = np.nonzero(domc)
    X = ref.unproject(xs, ys, Zc[ys, xs])
    ks = np.arange(0, V.n, p.get('tex_every', 2))
    N = len(xs)
    cols = np.full((len(ks), N, 3), 999, np.uint16)
    for i, k in enumerate(ks):
        q, z = V.project(X, k)
        xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        inb = (xi >= 0) & (xi < V.W) & (yi >= 0) & (yi < V.H) & (z > 0)
        ok = np.zeros(N, bool)
        dk = V.depth(k)
        st = S[k] & ~specular(V.frames[k])
        ok[inb] = st[yi[inb], xi[inb]] & (np.abs(z[inb] - dk[yi[inb], xi[inb]]) < p['vis_tol'])
        c = remap_pts(V.frames[k], q[:, 0], q[:, 1])
        cols[i, ok] = c[ok]
    allk = np.ones(len(ks), bool)
    gains = np.ones((len(ks), 3), np.float32)
    if p.get('tex_gain'):
        ref_col, _ = masked_median(cols, allk)
        for i in range(len(ks)):
            v = (cols[i, :, 0] < 999) & np.isfinite(ref_col[:, 0])
            if v.sum() > 500:
                g = np.median(cols[i, v].astype(np.float32) + 1, 0) / np.median(ref_col[v] + 1, 0)
                gains[i] = np.clip(g, 0.6, 1.6)
                c = np.clip(cols[i, v] / gains[i], 0, 255).astype(np.uint16)
                cols[i, v] = c
    good = sharp_w[ks]
    med_s, cnt_s = masked_median(cols, good)
    med_a, cnt_a = masked_median(cols, allk)
    if p.get('tex_blend'):
        w = np.clip(cnt_s / p.get('tex_min_sharp', 4), 0, 1)[:, None]
        res = np.where(cnt_s[:, None] > 0, w * np.nan_to_num(med_s) + (1 - w) * np.nan_to_num(med_a), med_a)
    else:
        res = np.where((cnt_s >= p.get('tex_min_sharp', 3))[:, None], med_s, med_a)
    done = cnt_a >= p.get('tex_min_obs', 1)
    tex = np.zeros((ref.Hc, ref.Wc, 3), np.float32)
    seen = np.zeros((ref.Hc, ref.Wc), bool)
    nobs = np.zeros((ref.Hc, ref.Wc), np.int32)
    tex[ys[done], xs[done]] = res[done]
    seen[ys[done], xs[done]] = True
    nobs[ys, xs] = cnt_a
    if p.get('tex_fill', 'telea') == 'pushpull':
        tex = push_pull(tex, seen)
        if p.get('tex_detail'):
            tex = add_detail(tex, seen, domc, p)
        tex = np.clip(tex, 0, 255).astype(np.uint8)
    else:
        tex = cv2.inpaint(np.clip(tex, 0, 255).astype(np.uint8), (~seen).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    return tex, seen, nobs, gains


def add_detail(tex, seen, domc, p, seed=0):
    """Hidden texels get high-frequency detail copied from random observed tiles (low frequency stays push-pull)."""
    rng = np.random.default_rng(seed)
    hp = np.clip(tex - cv2.GaussianBlur(tex, (0, 0), 6), -p.get('detail_clip', 255), p.get('detail_clip', 255))
    T = p.get('detail_tile', 48)
    H, W = seen.shape
    ok = cv2.erode(seen.astype(np.uint8), np.ones((T, T), np.uint8)).astype(bool)   # tiles fully observed
    cy, cx = np.nonzero(ok[T // 2:H - T // 2, T // 2:W - T // 2])
    if len(cy) == 0:
        return tex
    det = np.zeros_like(tex)
    wsum = np.zeros(seen.shape, np.float32)
    win = np.outer(np.hanning(T), np.hanning(T)).astype(np.float32) + 1e-3
    hole = ~seen & domc
    for y in range(0, H, T // 2):
        for x in range(0, W, T // 2):
            if not hole[y:y + T, x:x + T].any():
                continue
            j = rng.integers(len(cy))
            sy, sx = cy[j], cx[j]
            src = hp[sy:sy + T, sx:sx + T]
            h, w = min(T, H - y), min(T, W - x)
            det[y:y + h, x:x + w] += src[:h, :w] * win[:h, :w, None]
            wsum[y:y + h, x:x + w] += win[:h, :w]
    det /= np.maximum(wsum, 1e-6)[..., None]
    a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 4)[..., None]   # fade in at the rim
    return tex + a * det * p.get('detail_gain', 0.8)


# ---------------------------------------------------------------- mesh
def grid_mesh(ref, Zc, domc, step):
    vs = np.arange(0, ref.Hc, step)
    us = np.arange(0, ref.Wc, step)
    if us[-1] != ref.Wc - 1:
        us = np.r_[us, ref.Wc - 1]
    if vs[-1] != ref.Hc - 1:
        vs = np.r_[vs, ref.Hc - 1]
    UU, VV = np.meshgrid(us, vs)
    inside = domc[VV, UU]
    idx = -np.ones(UU.shape, np.int64)
    idx[inside] = np.arange(inside.sum())
    uc, vc = UU[inside], VV[inside]
    X = ref.unproject(uc, vc, Zc[vc, uc])
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    q = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    F = np.concatenate([np.stack([a[q], c[q], b[q]], 1), np.stack([a[q], d[q], c[q]], 1)])
    used = np.zeros(len(X), bool)
    used[F.ravel()] = True
    remap = -np.ones(len(X), np.int64)
    remap[used] = np.arange(used.sum())
    F = remap[F]
    uc, vc, X = uc[used], vc[used], X[used]
    uv = np.stack([(uc + 0.5) / ref.Wc, 1 - (vc + 0.5) / ref.Hc], 1)
    return X, F, uv, uc, vc


def orient_faces(X, F, ref):
    """Make face normals point towards the reference camera."""
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    toward = ref.pos - X[F].mean(1)
    flip = (n * toward).sum(1) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


# ---------------------------------------------------------------- evaluation
def boundary_f(a, b, valid, tol=4):
    k3 = np.ones((3, 3), np.uint8)
    ba = (a.astype(np.uint8) - cv2.erode(a.astype(np.uint8), k3)).astype(bool) & valid
    bb = (b.astype(np.uint8) - cv2.erode(b.astype(np.uint8), k3)).astype(bool) & valid
    if ba.sum() == 0 or bb.sum() == 0:
        return 0.0
    db = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 3)
    da = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 3)
    pr, rc = float((db[ba] <= tol).mean()), float((da[bb] <= tol).mean())
    return 2 * pr * rc / max(pr + rc, 1e-9)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def local_ncc(A, B, m, r=7):
    """Mean of windowed NCC (window 2r+1) over pixels of m (grey images float)."""
    w = (2 * r + 1, 2 * r + 1)
    mf = m.astype(np.float32)
    n = cv2.boxFilter(mf, -1, w, normalize=False) + 1e-6
    mu_a = cv2.boxFilter(A * mf, -1, w, normalize=False) / n
    mu_b = cv2.boxFilter(B * mf, -1, w, normalize=False) / n
    saa = cv2.boxFilter(A * A * mf, -1, w, normalize=False) / n - mu_a ** 2
    sbb = cv2.boxFilter(B * B * mf, -1, w, normalize=False) / n - mu_b ** 2
    sab = cv2.boxFilter(A * B * mf, -1, w, normalize=False) / n - mu_a * mu_b
    c = sab / np.sqrt(np.maximum(saa, 1e-3) * np.maximum(sbb, 1e-3))
    sel = m & (n > 0.6 * w[0] * w[1]) & (saa > 4) & (sbb > 4)
    return float(np.mean(c[sel])) if sel.sum() > 100 else np.nan


def render_frame(V, meshes, k):
    """Composite of several textured meshes [(X, F, uv, tex), ...] seen from frame k."""
    M = np.zeros((V.H, V.W), bool)
    Zr = np.full((V.H, V.W), np.inf)
    RGB = np.zeros((V.H, V.W, 3), np.float32)
    FID = np.full((V.H, V.W), -1, np.int64)
    for j, (X, F, uv, tex) in enumerate(meshes):
        q, z = V.project(X, k)
        m, zb, rgb, fid = render_textured(q, z, F, uv, tex, V.H, V.W)
        sel = m & (zb < Zr)
        M |= sel
        Zr[sel] = zb[sel]
        RGB[sel] = rgb[sel]
        FID[sel] = fid[sel] + (j << 32)
    return M, Zr, RGB, FID


def evaluate(V, meshes, S, ks, liver=None, face_label=None):
    keys = ['coverage', 'depth_med_mm', 'depth_bias_mm', 'depth_p90_mm', 'front_violation', 'gb_violation', 'gb_gap_mm', 'photo_l1', 'photo_ncc',
            'photo_lncc', 'frame_iou', 'liver_iou', 'liver_bf']
    per = {k: [] for k in keys}
    gbm = V.mask('gallbladder')
    for k in ks:
        M, Zr, RGB, FID = render_frame(V, meshes, k)
        D = V.depth(k)
        St = S[k]
        both = M & St
        per['coverage'].append(both.sum() / max(St.sum(), 1))
        dz = (Zr - D)[both]
        per['depth_med_mm'].append(np.median(np.abs(dz)) * 1e3 if len(dz) > 50 else np.nan)
        per['depth_bias_mm'].append(np.median(dz) * 1e3 if len(dz) > 50 else np.nan)
        per['depth_p90_mm'].append(np.percentile(np.abs(dz), 90) * 1e3 if len(dz) > 50 else np.nan)
        per['front_violation'].append(float((dz < -0.005).mean()) if len(dz) else np.nan)
        G = ero(gbm[k], 3) & M
        per['gb_violation'].append(float((Zr[G] < D[G] - 0.002).mean()) if G.sum() > 50 else np.nan)
        per['gb_gap_mm'].append(float(np.median(Zr[G] - D[G]) * 1e3) if G.sum() > 50 else np.nan)
        ph = both & ~specular(V.frames[k])
        fr = V.frames[k].astype(np.float32)
        per['photo_l1'].append(float(np.abs(RGB - fr)[ph].mean()) if ph.sum() > 50 else np.nan)
        ga = cv2.cvtColor(RGB.astype(np.float32), cv2.COLOR_RGB2GRAY)
        gb_ = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        per['photo_ncc'].append(ncc(ga[ph], gb_[ph]) if ph.sum() > 50 else np.nan)
        per['photo_lncc'].append(local_ncc(ga, gb_, ph))
        valid = ~dil(V.occluders(k), 2)
        per['frame_iou'].append(float((M & valid).sum() / valid.sum()))
        if liver is not None and face_label is not None:
            lab = np.zeros((V.H, V.W), bool)
            sel = FID >= 0
            lab[sel] = face_label[(FID[sel] & 0xffffffff)] == 1
            vis = lab & (Zr < D + 0.006)
            vl = ~dil(V.occluders(k) | gbm[k] | V.mask('strand')[k], 3)
            a, b = vis & vl, liver[k] & vl
            per['liver_iou'].append(float((a & b).sum() / max((a | b).sum(), 1)))
            per['liver_bf'].append(boundary_f(vis, liver[k], vl))
    return {k: np.array(v, float) for k, v in per.items() if len(v)}


def old_backdrop():
    """The old pipeline's backdrop in the same (SIFT) geometry: frame-0 depth backdrop + keyframe patches."""
    import imageio.v2 as imageio
    d = ROOT / 'outputs/variants/chole_a+sift/scene'
    out = []
    for j in range(20):
        p = d / f'backdrop_{j}.obj'
        if not p.exists():
            break
        X, T, F = [], [], []
        for line in p.read_text().splitlines():
            t = line.split()
            if not t:
                continue
            if t[0] == 'v':
                X.append([float(x) for x in t[1:4]])
            elif t[0] == 'vt':
                T.append([float(x) for x in t[1:3]])
            elif t[0] == 'f':
                F.append([int(x.split('/')[0]) - 1 for x in t[1:4]])
        tex = imageio.imread(d / f'tex_back_{j}.png')[..., :3]
        out.append((np.array(X), np.array(F), np.array(T, np.float32), tex))
    return out


# ---------------------------------------------------------------- liver template
GB_STL = ROOT / 'data/templates/bodyparts3d/gallbladder.stl'
# BodyParts3D frame (x patient left, y posterior, z cranial) -> world prior for a supine patient seen from the
# umbilical port (world x ~ patient left, y ~ cranial, z ~ anterior/up)
R_BP3D = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])


def rotvec_to_R(r):
    th = np.linalg.norm(r)
    return np.eye(3) if th < 1e-12 else rodrigues(r / th, th)


def render_ref_depth(ref, X, F):
    q, z = ref.cam.project(X, ref.R, ref.f, ref.pos)
    q = q - np.array([ref.ou, ref.ov])
    zb, fid, _ = raster(q, z, F, ref.Hc, ref.Wc, max_box=256)
    return np.where(fid >= 0, zb, np.nan)


def fit_liver_template(M, p, holdout=None, log=log, seed=0):
    """Similarity fit of the BodyParts3D liver (with its gallbladder, same frame) to the observed liver part of the
    backdrop and the observed gallbladder front surface, keeping the liver behind every observed static surface.
    holdout: canvas bool mask of observed cells to leave out of the data (for the hidden-shape test)."""
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    V, ref = M['V'], M['ref']
    lv = trimesh.load(LIVER_STL)
    gb = trimesh.load(GB_STL)
    LX, LF = np.asarray(lv.vertices) / 1e3, np.asarray(lv.faces)
    GX = np.asarray(gb.vertices) / 1e3
    rng = np.random.default_rng(seed)
    Ls = np.asarray(trimesh.sample.sample_surface(lv, 30000, seed=seed)[0]) / 1e3
    Gs = np.asarray(trimesh.sample.sample_surface(gb, 4000, seed=seed)[0]) / 1e3
    c0 = GX.mean(0)
    tL, tG = cKDTree(Ls - c0), cKDTree(Gs - c0)
    # data: observed liver cells of the reference canvas (world points), observed gallbladder points
    cs = p['cell']
    known = M['known']
    Hg, Wg = known.shape
    liv_cell = M['liv_cell'] > 0.5
    sel = known & liv_cell
    if holdout is not None:
        sel &= ~holdout
    cj, ci = np.nonzero(sel)
    o = rng.choice(len(cj), min(3000, len(cj)), replace=False)
    cj, ci = cj[o], ci[o]
    uc, vc = ci * cs + (cs - 1) / 2, cj * cs + (cs - 1) / 2
    PL = ref.unproject(uc, vc, M['Zfit'][cj, ci])
    PG = np.concatenate([V.points('gallbladder', k, step=6, erode=4) for k in range(150, 251, 25)])
    PG = PG[rng.choice(len(PG), min(3000, len(PG)), replace=False)]
    # free space: observed cells (any label) must not be behind the template surface
    fj, fi = np.nonzero(known)
    o = rng.choice(len(fj), min(6000, len(fj)), replace=False)
    fj, fi = fj[o], fi[o]
    PF = ref.unproject(fi * cs + (cs - 1) / 2, fj * cs + (cs - 1) / 2, M['Zfit'][fj, fi])
    dirF = PF - ref.pos
    dirF /= np.linalg.norm(dirF, axis=1, keepdims=True)
    view = ref.R[2]
    gc = np.median(PG, 0) + 0.010 * view

    def unpack(x):
        R = rotvec_to_R(x[:3]) @ R_BP3D
        s = np.exp(x[6])
        return R, x[3:6], s

    def residuals(x, w_g=0.5, w_f=2.0):
        R, t, s = unpack(x)
        # world -> template frame (centred at the template gallbladder): Xt = R^T (Xw - t) / s
        dl = tL.query(((PL - t) @ R) / s)[0] * s
        dg = tG.query(((PG - t) @ R) / s)[0] * s
        # free space: a point slightly in front of each observed surface point (2 mm towards the camera) must be
        # outside the liver -> penalise if it lies inside (approximate with distance to nearest template sample on
        # the inner side is costly; use the ray test via the fitted depth instead in evaluation)
        return np.concatenate([np.minimum(dl, 0.02), w_g * np.minimum(dg, 0.02)])

    # coarse search around the anatomical prior
    best = []
    s_grid = np.log([0.55, 0.7, 0.85, 1.0])
    for i in range(1500):
        r = rng.normal(size=3)
        r = r / np.linalg.norm(r) * np.radians(70) * rng.random() ** (1 / 3)
        for ls in s_grid:
            R = rotvec_to_R(r) @ R_BP3D
            t = gc
            x = np.r_[r, t, ls]
            c = np.median(residuals(x))
            best.append((c, x))
    best.sort(key=lambda a: a[0])
    cands = []
    for c, x in best[:12]:
        sol = least_squares(lambda x: residuals(x), x, loss='soft_l1', f_scale=0.003, max_nfev=200, diff_step=1e-3)
        cands.append((float(np.median(np.abs(residuals(sol.x)))), sol.x))
    cands.sort(key=lambda a: a[0])
    # choose the best candidate that keeps the observed surfaces visible (template not in front of them)
    out = []
    for c, x in cands[:6]:
        R, t, s = unpack(x)
        X = (LX - c0) * s @ R.T + t
        zt = render_ref_depth(ref, X, LF)
        zt_cells = zt[(np.arange(Hg) * cs + cs // 2)[:, None].clip(0, ref.Hc - 1), (np.arange(Wg) * cs + cs // 2)[None].clip(0, ref.Wc - 1)]
        fv = known & np.isfinite(zt_cells) & (zt_cells < M['Zfit'] - 0.005)
        fsv = float(fv.sum() / max(known.sum(), 1))
        dl = tL.query(((PL - t) @ R) / s)[0] * s
        dg = tG.query(((PG - t) @ R) / s)[0] * s
        out.append(dict(cost=c, x=x, X=X, F=LF, zt=zt_cells, freespace_violation=fsv,
                        liver_dist_mm=float(np.median(dl) * 1e3), gb_dist_mm=float(np.median(dg) * 1e3),
                        scale=float(s), rot_from_prior_deg=float(np.degrees(np.linalg.norm(x[:3])))))
    out.sort(key=lambda d: d['cost'] + 0.02 * d['freespace_violation'])
    log('liver template fits', [(round(d['liver_dist_mm'], 2), round(d['gb_dist_mm'], 2), round(d['freespace_violation'], 3),
                                 round(d['scale'], 2), round(d['rot_from_prior_deg'])) for d in out])
    return out[0]


def liver_cell_grid(Fz, known, Hg, Wg):
    """Fraction of a cell's static observations labelled liver; hidden cells take the nearest observed value."""
    from scipy import ndimage
    liv = np.where(Fz['cnt'] > 0, Fz['liv'] / np.maximum(Fz['cnt'], 1), np.nan).reshape(Hg, Wg)
    li = np.where(known & np.isfinite(liv), liv, np.nan)
    _, (iy, ix) = ndimage.distance_transform_edt(~np.isfinite(li), return_indices=True)
    return li[iy, ix]


def liver_template_stage(M, p, log=log):
    """Template fit + hold-out test of the hidden shape: a band of observed liver cells along the rim of the hidden
    region (gallbladder bed) is left out; its depth is predicted (a) by the smooth fit alone, (b) by the template
    alone, (c) by the smooth fit with the template as a weak prior in unknown cells; error vs the fused median depth.
    Returns the full-data fit, the test numbers and the bed surface of the chosen variant."""
    known, dom, w, Z = M['known'], M['dom'], M['w'], M['Zmed']
    liv = M['liv_cell'] > 0.5
    hidden = dom & ~known
    dist = cv2.distanceTransform((~hidden).astype(np.uint8), cv2.DIST_L2, 5)
    band = known & liv & (dist <= p.get('holdout_band', 30))
    res = dict(holdout_cells=int(band.sum()))
    truth = Z[band]
    # (a) smooth fit without the band
    wa = np.where(band, 0.0, w)
    Za = fit_surface(np.where(known & ~band, Z, 0), wa, dom, p['lam1'], p['lam2'], M['lower'])
    res['smooth_mm'] = float(np.median(np.abs(Za[band] - truth)) * 1e3)
    # (b) template fitted without the band
    Mh = dict(M, known=known & ~band, Zfit=Za)
    T = fit_liver_template(Mh, p, holdout=band, log=log)
    zt = T['zt']
    okb = band & np.isfinite(zt)
    res['template_mm'] = float(np.median(np.abs(zt[okb] - Z[okb])) * 1e3) if okb.any() else None
    res['template_coverage'] = float(okb.sum() / max(band.sum(), 1))
    # (c) smooth fit + template prior in the unknown cells
    wt = p.get('tpl_weight', 0.05)
    prior = (~(known & ~band)) & dom & np.isfinite(zt)
    Zc_ = fit_surface(np.where(prior, zt, np.where(known & ~band, Z, 0)), np.where(prior, wt, wa), dom, p['lam1'], p['lam2'], M['lower'])
    res['smooth+template_mm'] = float(np.median(np.abs(Zc_[band] - truth)) * 1e3)
    res['template_fit_holdout'] = {k: T[k] for k in ('liver_dist_mm', 'gb_dist_mm', 'freespace_violation', 'scale', 'rot_from_prior_deg')}
    log('hold-out test', res)
    # full-data fit
    Tf = fit_liver_template(M, p, log=log)
    zt = Tf['zt']
    prior = hidden & np.isfinite(zt)
    Zbed = fit_surface(np.where(prior, zt, np.where(known, Z, 0)), np.where(prior, wt, w), dom, p['lam1'], p['lam2'], M['lower'])
    res['template_fit'] = {k: Tf[k] for k in ('liver_dist_mm', 'gb_dist_mm', 'freespace_violation', 'scale', 'rot_from_prior_deg')}
    res['bed_template_coverage'] = float(prior.sum() / max(hidden.sum(), 1))
    return dict(test=res, X=Tf['X'], F=Tf['F'], Zbed=Zbed, x=Tf['x'])


# ---------------------------------------------------------------- build
VERSIONS = {
    # v01: plain fusion: dilated SAM masks, median over all frames, harmonic fill, median texture
    'v01': dict(gb_r=8, strand_r=8, tool_r=6, sheet_r=0, cell=2, grid=6, min_obs=3, smooth=1.0, fill='harmonic',
                gb_gap=None, vis_tol=0.006, tex_every=2, sharp_from=62, reject_iters=0),
    # v02: wider exclusion (depth bleeds at instrument / gallbladder edges), sheet zone, robust rejection, sharp
    # frames preferred for depth, domain = cells seen by >= 15 frames, gallbladder lower bound for the fill,
    # texture gains + sharp/blur blending + push-pull fill
    # v03: confidence-weighted smooth height-field fit (data weight from #views and spread, pan-only cells down-weighted;
    # membrane + thin-plate regulariser fills the hidden cells in the same solve), domain >= 6 frames, wider border,
    # liver-like detail in the filled texture
    'v03': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=12, cell=2, grid=6, min_obs=5, min_foot=6,
                smooth=0, fill='fit', fit_sigma=0.003, pan_only_w=0.2, lam1=0.3, lam2=3.0, gb_gap=0.003,
                vis_tol=0.006, tex_every=2, sharp_from=62, depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5,
                reject_min=0.0025, tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_fill='pushpull',
                tex_detail=True),
    # v04: v03 + BodyParts3D liver template (with its gallbladder) fitted to the observed liver + gallbladder,
    # hold-out test of the hidden shape (does the template predict the rim of the gallbladder bed?)
    'v04': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=12, cell=2, grid=6, min_obs=5, min_foot=6,
                smooth=0, fill='fit', fit_sigma=0.003, pan_only_w=0.2, lam1=0.3, lam2=3.0, gb_gap=0.003,
                vis_tol=0.006, tex_every=2, sharp_from=62, depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5,
                reject_min=0.0025, tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_fill='pushpull',
                tex_detail=True, detail_gain=0.6, liver_template=True, holdout_band=30, tpl_weight=0.05,
                bed_from_template=False),
    # v05: pure thin-plate regulariser (lam1=0; chosen by the hold-out band test), no template, texture texels need
    # >= 3 observations, clipped detail; warp-consistency metric, labelled contact sheet, held-out keyframes
    'v05': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=12, cell=2, grid=6, min_obs=5, min_foot=6,
                smooth=0, fill='fit', fit_sigma=0.003, pan_only_w=0.2, lam1=0.0, lam2=3.0, gb_gap=0.003,
                vis_tol=0.006, tex_every=2, sharp_from=62, depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5,
                reject_min=0.0025, tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_min_obs=3, tex_fill='pushpull',
                tex_detail=True, detail_gain=0.6, detail_clip=12),
    # v06: v05 + per-frame camera rotations registered to the backdrop (ECC), backdrop re-fused with them (1 round)
    'v06': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=12, cell=2, grid=6, min_obs=5, min_foot=6,
                smooth=0, fill='fit', fit_sigma=0.003, pan_only_w=0.2, lam1=0.0, lam2=3.0, gb_gap=0.003,
                vis_tol=0.006, tex_every=2, sharp_from=62, depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5,
                reject_min=0.0025, tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_min_obs=3, tex_fill='pushpull',
                tex_detail=True, detail_gain=0.6, detail_clip=12, refine_cams=1),
    # v07: v06 with cells seen only in the blurred pan nearly free (weight 0.03): the thin plate extends the sharp
    # surface there instead of following the pan frames' inconsistent depth (fins at the canvas edge)
    'v07': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=12, cell=2, grid=6, min_obs=5, min_foot=6,
                smooth=0, fill='fit', fit_sigma=0.003, pan_only_w=0.03, lam1=0.0, lam2=3.0, gb_gap=0.003,
                vis_tol=0.006, tex_every=2, sharp_from=62, depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5,
                reject_min=0.0025, tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_min_obs=3, tex_fill='pushpull',
                tex_detail=True, detail_gain=0.6, detail_clip=12, refine_cams=1),
    # v08: v07 + the pink strands next to the cystic duct excluded (own SAM mask, dilated 10 px; they moved with the
    # neck and lit up in the residual maps), two rounds of camera registration
    'v08': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, pink_r=10, border=12, cell=2, grid=6, min_obs=5,
                min_foot=6, smooth=0, fill='fit', fit_sigma=0.003, pan_only_w=0.03, lam1=0.0, lam2=3.0, gb_gap=0.003,
                vis_tol=0.006, tex_every=2, sharp_from=62, depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5,
                reject_min=0.0025, tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_min_obs=3, tex_fill='pushpull',
                tex_detail=True, detail_gain=0.6, detail_clip=12, refine_cams=2),
    'v02': dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=8, cell=2, grid=6, min_obs=5, min_foot=15,
                smooth=1.0, fill='harmonic', gb_gap=0.003, vis_tol=0.006, tex_every=2, sharp_from=62,
                depth_sharp=True, min_sharp=5, reject_iters=2, reject_k=2.5, reject_min=0.0025,
                tex_gain=True, tex_blend=True, tex_min_sharp=4, tex_fill='pushpull'),
}


def label_cells(F, med_ok, Hg, Wg):
    liv = np.where(F['cnt'] > 0, F['liv'] / np.maximum(F['cnt'], 1), 0).reshape(Hg, Wg)
    return liv


def with_cameras(V, R):
    """A shallow copy of the views with other camera rotations (positions, focal, depth maps unchanged)."""
    import copy
    Vr = copy.copy(V)
    Vr.R = np.asarray(R, float)
    return Vr


def refine_cameras(V, X, F, uv, tex, S, iters=2, log=log, fix_keyframes=True):
    """Register every frame to the textured backdrop: 2D Euclidean ECC between the render and the video on static
    pixels, converted into a small camera rotation (shift dx, dy -> yaw/pitch about the camera axes, in-plane angle
    -> roll); iterated. Only the rotation changes (the clip's cameras between BA keyframes are interpolated)."""
    R = V.R.copy()
    stats = np.zeros((V.n, 4))
    for it in range(iters):
        Vr = with_cameras(V, R)
        for k in range(V.n):
            if fix_keyframes and k in V.keyframes:      # bundle-adjusted keyframes stay as they are
                continue
            q, z = Vr.project(X, k)
            m, zb, rgb, _ = render_textured(q, z, F, uv, tex, V.H, V.W)
            a = cv2.GaussianBlur(cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY), (0, 0), 1.5)
            b = cv2.GaussianBlur(cv2.cvtColor(V.frames[k], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
            msk = (S[k] & m).astype(np.uint8)
            W = np.eye(2, 3, dtype=np.float32)
            try:
                cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_EUCLIDEAN,
                                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), msk, 5)
            except cv2.error:
                stats[k] = (np.nan, 0, 0, 0)
                continue
            c = np.array([V.W / 2, V.H / 2, 1.0])
            dx, dy = W @ c - c[:2]
            th = np.arctan2(W[1, 0], W[0, 0])
            f = V.f[k]
            if np.hypot(dx, dy) > 40:                # implausible: keep the camera
                stats[k] = (cc, dx, dy, th)
                continue
            beta, alpha = np.arctan(dx / f), -np.arctan(dy / f)
            Ry = rodrigues([0, 1, 0], beta)
            Rx = rodrigues([1, 0, 0], alpha)
            Rz = rodrigues([0, 0, 1], th)
            R[k] = Rz @ Ry @ Rx @ R[k]
            stats[k] = (cc, dx, dy, th)
        sh = np.hypot(stats[:, 1], stats[:, 2])
        log(f'camera refinement iter {it}: median shift {np.nanmedian(sh):.2f} px, p90 {np.nanpercentile(sh, 90):.2f}, '
            f'max {np.nanmax(sh):.1f}')
    ang = np.degrees([np.arccos(np.clip((np.trace(R[k] @ V.R[k].T) - 1) / 2, -1, 1)) for k in range(V.n)])
    return R, dict(correction_deg=ang, last_shift_px=np.hypot(stats[:, 1], stats[:, 2]), ecc=stats[:, 0])


def build(ver, p, log=log, V=None):
    out = OUT / ver
    (out / 'work').mkdir(parents=True, exist_ok=True)
    V = views.load('chole_a', 'sift') if V is None else V
    liver = liver_masks(V)
    t0 = time.time()
    S = static_masks(V, p)
    if p.get('holdout_kf'):
        S[V.keyframes] = False                      # keyframes contribute neither depth nor colour
    log('static fraction', round(float(S.mean()), 3))
    ref = Ref(V, cell=p['cell'])
    log('canvas', ref.Wc, ref.Hc, 'offset', ref.ou, ref.ov)
    Fz = fuse_depth(V, ref, S, p, liver)
    Hg, Wg, cs = Fz['Hg'], Fz['Wg'], p['cell']
    sharp = np.arange(V.n) >= p['sharp_from']
    med, cnt, mad, Dk = robust_fuse(Fz, p, sharp if p.get('depth_sharp') else None)
    Fz['cnt'] = np.isfinite(Fz['D']).sum(0)
    known = (cnt >= p['min_obs']).reshape(Hg, Wg)
    Z = med.reshape(Hg, Wg).astype(np.float64)
    dom = fill_holes(largest_cc(Fz['foot'].reshape(Hg, Wg) >= p.get('min_foot', 1)))
    known &= dom
    Zs = norm_smooth(np.where(known, Z, 0), known, p['smooth'])
    lower = None
    if p.get('gb_gap') is not None:
        g = Fz['gbz']
        gcnt = np.isfinite(g).sum(0)
        lower = np.full(Hg * Wg, np.nan)
        ok = gcnt >= 3
        lower[ok] = np.nanpercentile(g[:, ok], 90, axis=0) + p['gb_gap']
        lower = lower.reshape(Hg, Wg)
    if p['fill'] == 'fit':
        cnt_sharp = np.isfinite(Fz['D'][sharp]).sum(0).reshape(Hg, Wg)
        madg = np.nan_to_num(mad.reshape(Hg, Wg), nan=0.02)
        w = np.minimum(cnt.reshape(Hg, Wg), 50) / 50 / (1 + (madg / p['fit_sigma']) ** 2)
        w = np.where(cnt_sharp >= p.get('min_sharp', 5), w, w * p.get('pan_only_w', 0.2))
        w = np.where(known, w, 0.0)
        Zf = fit_surface(np.where(known, Z, 0), w, dom, p['lam1'], p['lam2'], lower)
    else:
        w = known.astype(float)
        Zf = fill_surface(np.where(known, Zs, 0), known, dom, p['fill'], lower=lower)
    TPL = None
    if p.get('liver_template'):
        TPL = liver_template_stage(dict(V=V, ref=ref, known=known, dom=dom, Zfit=Zf, Zmed=Z, w=w, lower=lower,
                                        liv_cell=liver_cell_grid(Fz, known, Hg, Wg)), p, log)
        if p.get('bed_from_template'):
            Zf = TPL['Zbed']
    if p.get('fill_smooth'):
        hid = dom & ~known
        sm = norm_smooth(np.where(dom, Zf, 0), dom, p['fill_smooth'])
        Zf[hid] = sm[hid]
    log('fused: known cells', int(known.sum()), 'hidden', int((dom & ~known).sum()), f'{time.time() - t0:.0f}s')
    # canvas-resolution depth and domain
    vc_, uc_ = np.mgrid[0:ref.Hc, 0:ref.Wc]
    Zc = sample_grid(np.where(dom, Zf, np.nanmedian(Zf)), cs, uc_, vc_)
    domc = sample_grid(dom.astype(np.float32), cs, uc_, vc_, cv2.INTER_NEAREST) > 0.5
    knownc = sample_grid(known.astype(np.float32), cs, uc_, vc_, cv2.INTER_NEAREST) > 0.5
    tex, seen, nobs, gains = fuse_texture(V, ref, Zc, domc, S, p, sharp)
    log('gains', np.round(gains[::10, 1], 2))
    log('texture: seen', round(float(seen[domc].mean()), 3), f'{time.time() - t0:.0f}s')
    X, F, uv, uc, vc = grid_mesh(ref, Zc, domc, p['grid'])
    F = orient_faces(X, F, ref)
    ci, cj = np.clip(uc // cs, 0, Wg - 1), np.clip(vc // cs, 0, Hg - 1)
    observed = known[cj, ci]
    liv = np.where(Fz['cnt'] > 0, Fz['liv'] / np.maximum(Fz['cnt'], 1), np.nan).reshape(Hg, Wg)
    # labels of hidden cells: nearest observed cell's label
    from scipy import ndimage
    li = np.where(known & np.isfinite(liv), liv, np.nan)
    _, (iy, ix) = ndimage.distance_transform_edt(~np.isfinite(li), return_indices=True)
    li = li[iy, ix]
    label = (li[cj, ci] > 0.5).astype(np.int8)
    gcnt = np.isfinite(Fz['gbz']).sum(0).reshape(Hg, Wg)
    bed = (~observed) & (dil(gcnt >= 3, 2)[cj, ci])
    bedcell = dom & ~known & dil(gcnt >= 3, 2)
    dcell = cv2.distanceTransform(bedcell.astype(np.uint8), cv2.DIST_L2, 5)
    t_ = np.clip(dcell / p.get('bed_ramp', 20), 0, 1)
    bed_weight = ((3 * t_ ** 2 - 2 * t_ ** 3))[cj, ci] * bed
    confidence = np.asarray(w)[cj, ci]
    spread = (mad.reshape(Hg, Wg) * 1e3)[cj, ci]
    nob = cnt.reshape(Hg, Wg)[cj, ci]
    vcol = tex[vc, uc]
    face_label = (label[F].sum(1) >= 2).astype(np.int8)
    model = dict(TPL=TPL, bed_weight=bed_weight, confidence=confidence, Zfit=Zf, Zmed=Z, w=w, lower=lower, liv_cell=liver_cell_grid(Fz, known, Hg, Wg),
                 X=X, F=F, uv=uv.astype(np.float32), tex=tex, seen=seen, observed=observed, bed=bed, label=label,
                 face_label=face_label, spread=spread, nobs=nob, vcol=vcol, ref=ref, Zc=Zc, domc=domc, knownc=knownc,
                 mad=mad.reshape(Hg, Wg), cnt=cnt.reshape(Hg, Wg), known=known, dom=dom, S=S, liver=liver, V=V,
                 tex_nobs=nobs)
    log('mesh', len(X), 'verts', len(F), 'faces', f'{time.time() - t0:.0f}s')
    return model


def save(ver, p, M, metrics_frames=None):
    import imageio.v2 as imageio
    out = OUT / ver
    V, ref = M['V'], M['ref']
    X, F = M['X'].astype(np.float32), M['F'].astype(np.int32)
    r = ref.as_dict()
    ray = X - ref.pos
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    hf = hfield(ref, M['X'], M['F'])
    np.savez(out / 'model.npz', rest_verts=X, faces=F, verts4d=np.repeat(X[None], V.n, 0),
             uv=M['uv'], vertex_rgb=M['vcol'].astype(np.uint8), observed=M['observed'], filled=~M['observed'],
             bed=M['bed'], bed_weight=M['bed_weight'].astype(np.float32), ray_dir=ray.astype(np.float32),
             confidence=M['confidence'].astype(np.float32),
             label=M['label'], spread_mm=M['spread'].astype(np.float32), n_obs=M['nobs'].astype(np.int16),
             attach_idx=np.arange(len(X)), attach_to=np.array('world'),
             **{f'ref_{k}': np.asarray(v) for k, v in r.items()}, **{f'hfield_{k}': v for k, v in hf.items()})
    if M.get('TPL') is not None:
        T = M['TPL']
        np.savez(out / 'liver_template.npz', verts=T['X'].astype(np.float32), faces=T['F'].astype(np.int32),
                 params=T['x'], note='BodyParts3D liver after the similarity fit (world m); x = rotvec (from the '
                 'anatomical prior), t, log scale')
    lv = np.load(OUT / 'masks.npz')
    extra = {}
    if p.get('pink_r') is not None:
        ps = np.load(OUT / 'masks_strands.npz')
        extra = dict(pink_strands=ps['pink_strands'], prompts_pink_strands=ps['prompts'])
    np.savez_compressed(out / 'masks.npz', liver=lv['liver'], prompts_liver=lv['prompts'], packed='np.packbits(axis=-1)',
                        **extra)
    imageio.imwrite(out / 'texture.png', M['tex'])
    imageio.imwrite(out / 'texture_seen.png', (M['seen'] * 255).astype(np.uint8))
    write_obj(out / 'backdrop.obj', X, F, M['uv'], 'texture.png')


def hfield(ref, X, F, res=0.0008):
    """MuJoCo height field of the surface for collisions, in a frame looking back at the reference camera:
    frame axes x = ref x (image right), y = -ref y (image up), z = -ref z (towards the camera); elevation at
    (x, y) = z_far - depth along the reference axis. MuJoCo convention: elev[i, j] at x = (2j/(ncol-1) - 1) rx,
    y = (2i/(nrow-1) - 1) ry; size = (rx, ry, elevation range, base thickness); data normalised to [0, 1]."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree
    pc = (X - ref.pos) @ ref.R.T                    # ref camera coords (x right, y down, z forward)
    hx, hy, hz = pc[:, 0], -pc[:, 1], pc[:, 2]
    x0, x1, y0, y1 = hx.min(), hx.max(), hy.min(), hy.max()
    ncol, nrow = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
    gx, gy = np.meshgrid(np.linspace(x0, x1, ncol), np.linspace(y0, y1, nrow))
    z = LinearNDInterpolator(np.stack([hx, hy], 1), hz)(gx, gy)
    # outside the surface: nearest value (MuJoCo needs a full grid; flagged in hfield_valid)
    valid = np.isfinite(z)
    d, i = cKDTree(np.stack([gx[valid], gy[valid]], 1)).query(np.stack([gx[~valid], gy[~valid]], 1))
    z[~valid] = z[valid][i]
    zfar = z.max() + 0.002
    elev = zfar - z
    er = float(elev.max())
    Rf = np.stack([ref.R[0], -ref.R[1], -ref.R[2]])  # rows = hfield frame axes in world
    centre = ref.pos + ((x0 + x1) / 2) * ref.R[0] - ((y0 + y1) / 2) * ref.R[1] + zfar * ref.R[2]
    import mujoco
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rf.T.flatten())
    return dict(elev=(elev / er).astype(np.float32), valid=valid, size=np.array([(x1 - x0) / 2, (y1 - y0) / 2, er, 0.005]),
                pos=centre, quat=quat, R=Rf)


def write_obj(path, X, F, uv, tex_name):
    mtl = path.with_suffix('.mtl')
    mtl.write_text(f'newmtl backdrop\nKd 1 1 1\nmap_Kd {tex_name}\n')
    lines = [f'mtllib {mtl.name}', 'usemtl backdrop'] + ['v %.6f %.6f %.6f' % tuple(x) for x in X] + \
            ['vt %.6f %.6f' % tuple(t) for t in uv] + [f'f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}' for a, b, c in F]
    path.write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------- pictures
def render_sheet(V, meshes, S, ks, path, scale=0.5):
    rows = []
    for k in ks:
        M, Zr, RGB, _ = render_frame(V, meshes, k)
        fr = V.frames[k]
        rgb = np.clip(RGB, 0, 255).astype(np.uint8)
        diff = np.abs(RGB - fr.astype(np.float32)).mean(2)
        dimg = cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
        dimg[~S[k]] = (dimg[~S[k]] * 0.3).astype(np.uint8)
        dz = np.where(M, np.abs(Zr - V.depth(k)), 0) * 1e3
        zimg = cv2.applyColorMap(np.clip(dz * 25, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
        zimg[~S[k]] = (zimg[~S[k]] * 0.3).astype(np.uint8)
        cnt = cv2.findContours(S[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        rgb = cv2.drawContours(rgb.copy(), cnt, -1, (0, 255, 0), 1)
        tiles = [fr.copy(), rgb, dimg, zimg]
        for t, s in zip(tiles, (f'video {k}', 'render (static outline)', '|rgb diff| x3', '|depth diff| 0-10mm')):
            cv2.putText(t, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.concatenate([cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles], 1))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def rodrigues(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def vertex_normals(X, F):
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    vn = np.zeros_like(X)
    for i in range(3):
        np.add.at(vn, F[:, i], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def views3d(M, path, extra=None):
    """Rest shape from 3 angles: textured (top) and shaded geometry with filled cells tinted blue, liver magenta (bottom)."""
    V, ref = M['V'], M['ref']
    X, F, uv, tex = M['X'], M['F'], M['uv'], M['tex']
    c = np.median(X, 0)
    vn = vertex_normals(X, F)
    top, bot = [], []
    for ang, axis in ((0, ref.R[1]), (-40, ref.R[1]), (40, ref.R[0])):
        Rt = rodrigues(axis, np.radians(ang))
        Rv = ref.R @ Rt.T
        pv = c + Rt @ (ref.pos - c)
        f = ref.f * 0.62
        q, z = V.cam.project(X, Rv, f, pv)
        m, zb, rgb, fid = render_textured(q, z, F, uv, tex, V.H, V.W)
        top.append(np.clip(rgb, 0, 255).astype(np.uint8))
        light = -Rv[2]
        sh = np.abs(vn @ light) * 0.75 + 0.25
        base = np.ones((len(X), 3)) * 200
        base[M['label'] == 1] = (220, 140, 200)
        base[~M['observed']] = (120, 150, 230)
        if extra is not None:
            base[extra] = (240, 200, 80)
        col = base * sh[:, None]
        zb2, fid2, bary2 = raster(q, z, F, V.H, V.W)
        img = interp_attr(F, fid2, bary2, col.astype(np.float32))
        img[fid2 < 0] = 30
        bot.append(np.clip(img, 0, 255).astype(np.uint8))
        cv2.putText(top[-1], f'view {ang:+d} deg', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bot[0], 'grey observed, blue filled, magenta liver', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    sheet = cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def canvas_pictures(M, path):
    """Canvas (reference view) maps: texture, observation count, spread (MAD across frames), fused depth."""
    tex = M['tex'].copy()
    known, dom = M['known'], M['dom']
    cs = M['tex'].shape[0] / known.shape[0]
    up = lambda a: cv2.resize(a, (tex.shape[1], tex.shape[0]), interpolation=cv2.INTER_NEAREST)
    cnt = up(M['cnt'].astype(np.float32))
    cimg = cv2.applyColorMap(np.clip(cnt, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    mad = up(np.nan_to_num(M['mad'] * 1e3).astype(np.float32))
    mimg = cv2.applyColorMap(np.clip(mad * 40, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
    Zc = M['Zc'] * 1e3
    lo, hi = np.percentile(Zc[M['domc']], [2, 98])
    zimg = cv2.applyColorMap(np.clip((Zc - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    hid = up((dom & ~known).astype(np.uint8)).astype(bool)
    for im in (tex, cimg, mimg, zimg):
        im[~M['domc']] = 0
    cnt_ = cv2.findContours(hid.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for im, s in ((tex, 'texture (cyan: filled region)'), (cimg, 'frames observing (0-255)'), (mimg, 'spread MAD 0-6 mm'),
                  (zimg, f'depth {lo:.0f}-{hi:.0f} mm')):
        cv2.drawContours(im, cnt_, -1, (0, 255, 255), 1)
        cv2.putText(im, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    sheet = np.concatenate([np.concatenate([tex, cimg], 1), np.concatenate([mimg, zimg], 1)], 0)
    sheet = cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- report
# one evaluation mask for every version (the static pixels of v02's conservative masks), so numbers are comparable
EVAL_MASK = dict(gb_r=12, strand_r=10, tool_r=12, sheet_r=45, border=8)


def report(ver, p, M, baseline=None):
    V = M['V']
    S = static_masks(V, EVAL_MASK)
    out = OUT / ver
    meshes = [(M['X'], M['F'], M['uv'], M['tex'])]
    ks_all = list(range(V.n))
    t0 = time.time()
    per = evaluate(V, meshes, S, ks_all, M['liver'], M['face_label'])
    log('evaluated', len(ks_all), 'frames', f'{time.time() - t0:.0f}s')
    kf = V.keyframes
    summ = {k: quality.summarize(v) for k, v in per.items()}
    summ_kf = {k: quality.summarize(v[kf]) for k, v in per.items()}
    pan = np.arange(V.n) < p['sharp_from']
    summ_pan = {k: quality.summarize(v[pan]) for k, v in per.items()}
    summ_post = {k: quality.summarize(v[~pan]) for k, v in per.items()}
    known = M['known']
    mad = M['mad'][known] * 1e3
    q = dict(
        version=ver, params=p, eval_mask=EVAL_MASK,
        definitions=dict(
            coverage='fraction of the frame\'s static pixels covered by the rendered backdrop',
            depth_med_mm='median |z_render - z_video| over static covered pixels (mm), per frame',
            depth_bias_mm='signed median (z_render - z_video): negative = backdrop in front of the video surface',
            depth_p90_mm='90th percentile of |z_render - z_video|',
            front_violation='fraction of static covered pixels where the backdrop is >5 mm in front of the video depth',
            gb_violation='fraction of visible gallbladder pixels where the backdrop is >2 mm in front of the gallbladder',
            gb_gap_mm='median depth gap (mm) between the backdrop and the visible gallbladder front surface (room for the gallbladder body)',
            photo_l1='mean |rgb_render - rgb_video| (0-255) over static covered non-specular pixels',
            photo_ncc='global NCC of grey render vs grey video over the same pixels',
            photo_lncc='mean 15x15 windowed NCC over the same pixels (texture alignment)',
            frame_iou='silhouette IoU of the backdrop with the whole frame minus instruments (the backdrop should be behind everything)',
            liver_iou='IoU of the visible liver-labelled backdrop part with the SAM liver mask (gallbladder/strand/instrument pixels ignored)',
            liver_bf='boundary F (4 px) of the same',
            spread='multi-view consistency: 1.4826*MAD across frames of the fused reference depth per observed cell (mm)'),
        summary_all_frames=summ, summary_keyframes=summ_kf, summary_pan_frames=summ_pan, summary_after_pan=summ_post,
        keyframes=dict(frames=kf, silhouette_iou=per['frame_iou'][kf].round(4).tolist(),
                       boundary_f_liver=per['liver_bf'][kf].round(4).tolist(),
                       depth_residual_mm=per['depth_med_mm'][kf].round(3).tolist()),
        per_frame={k: np.round(v, 4).tolist() for k, v in per.items()},
        spread_mm=dict(median=round(float(np.median(mad)), 3), p90=round(float(np.percentile(mad, 90)), 3),
                       p99=round(float(np.percentile(mad, 99)), 3)),
        surface=dict(n_verts=int(len(M['X'])), n_faces=int(len(M['F'])),
                     filled_vertex_fraction=round(float((~M['observed']).mean()), 4),
                     bed_vertex_fraction=round(float(M['bed'].mean()), 4),
                     liver_vertex_fraction=round(float((M['label'] == 1).mean()), 4),
                     texture_seen_fraction=round(float(M['seen'][M['domc']].mean()), 4),
                     canvas=[int(M['ref'].Wc), int(M['ref'].Hc)]),
        mesh_health=quality.mesh_health(M['X'], M['F']),
        verts4d='static: verts4d repeats rest_verts in every frame',
    )
    if M.get('Vr') is not None:
        per_r = evaluate(M['Vr'], meshes, S, ks_all, M['liver'], M['face_label'])
        q_r = {k: quality.summarize(v) for k, v in per_r.items()}
        q_r_kf = {k: quality.summarize(v[kf]) for k, v in per_r.items()}
        q_r_post = {k: quality.summarize(v[~pan]) for k, v in per_r.items()}
        wc_r = warp_consistency(M['Vr'], M['X'], M['F'], S, [(k, k + d) for d in (17, 53) for k in range(63, V.n - d, 13)])
        mid = [15, 45, 75, 105, 145, 185, 225, 245]      # between BA keyframes (keyframe cameras are not changed)
        render_sheet(M['Vr'], meshes, S, mid, out / 'render_sheet_midframes_refined_cams.jpg')
        render_sheet(V, meshes, S, mid, out / 'render_sheet_midframes_clip_cams.jpg')
    wc = warp_consistency(V, M['X'], M['F'], S, [(k, k + d) for d in (17, 53) for k in range(63, V.n - d, 13)])
    q['warp_consistency'] = wc
    if p.get('pink_r') is not None:
        strict = dict(EVAL_MASK, pink_r=p['pink_r'])
        Ss = static_masks(V, strict)
        Vs = M.get('Vr') if M.get('Vr') is not None else V
        per_s = evaluate(Vs, meshes, Ss, ks_all, M['liver'], M['face_label'])
        q['strict_mask'] = dict(note='evaluation mask also excluding the pink strands (masks_strands.npz), cameras: '
                                + ('cams_refined.npz' if M.get('Vr') is not None else 'clip'), mask=strict,
                                summary_all_frames={k: quality.summarize(v) for k, v in per_s.items()},
                                summary_after_pan={k: quality.summarize(v[~pan]) for k, v in per_s.items()})
    q['definitions']['warp_consistency'] = ('static pixels of frame k lifted with a depth (fused backdrop rendered '
        'into k, or frame k\'s own depth map), projected into frame j, colours compared (15x15 local NCC, L1) on the '
        'same pixel set: tests the geometry + cameras without the texture; fused > depth map means the fused '
        'surface is more multi-view consistent than the per-frame depth')
    if M.get('Vr') is not None:
        q['with_refined_cameras'] = dict(note='same model evaluated with cams_refined.npz (frames registered to the backdrop)',
                                         summary_all_frames=q_r, summary_keyframes=q_r_kf, summary_after_pan=q_r_post,
                                         warp_consistency=wc_r, per_frame={k: np.round(v, 4).tolist() for k, v in per_r.items()})
    bed = M['bed']
    if bed.any():
        q['gallbladder_bed'] = dict(n_verts=int(bed.sum()), centroid_world=M['X'][bed].mean(0).round(4).tolist(),
                                    bbox_world=[M['X'][bed].min(0).round(4).tolist(), M['X'][bed].max(0).round(4).tolist()],
                                    lower_bound='bed >= 90th percentile of the gallbladder front depth (reference view) + gb_gap')
    if M.get('TPL') is not None:
        q['liver_template'] = M['TPL']['test']
    if baseline is not None:
        q['baseline_old_pipeline'] = baseline
    (out / 'quality.json').write_text(json.dumps(q, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    sk = [0, 30, 60, 100, 140, 180, 220, 250]
    render_sheet(V, meshes, S, sk, out / 'render_sheet.jpg')
    views3d(M, out / 'views3d.jpg')
    canvas_pictures(M, out / 'canvas.jpg')
    # contract contact sheet: quality.contact_sheet on a coarse version of the same surface (its rasteriser is per-face)
    Xc, Fc, _, ucc, vcc = grid_mesh(M['ref'], M['Zc'], M['domc'], 12)
    cs = p['cell']
    ci, cj = np.clip(ucc // cs, 0, M['known'].shape[1] - 1), np.clip(vcc // cs, 0, M['known'].shape[0] - 1)
    obs_c = M['known'][cj, ci][Fc].sum(1) >= 2
    liv_c = (M['liv_cell'][cj, ci] > 0.5)[Fc].sum(1) >= 2
    layers = [('liver (observed)', Xc, Fc[obs_c & liv_c], (255, 80, 220)),
              ('other static (observed)', Xc, Fc[obs_c & ~liv_c], (90, 200, 255)),
              ('filled (bed, under tools)', Xc, Fc[~obs_c], (255, 230, 60))]
    quality.contact_sheet(V, layers, kf[::2], out / 'sheet.jpg')
    log('pictures', f'{time.time() - t0:.0f}s')
    return q


def warp_consistency(V, X, F, S, pairs):
    res = {'fused_lncc': [], 'depthmap_lncc': [], 'fused_l1': [], 'depthmap_l1': [], 'pairs': []}
    for k, j in pairs:
        q, z = V.project(X, k)
        zb, fid, _ = raster(q, z, F, V.H, V.W)
        gk = cv2.cvtColor(V.frames[k], cv2.COLOR_RGB2GRAY).astype(np.float32)
        warped, oks = {}, []
        for name, Dk in (('fused', zb), ('depthmap', V.depth(k))):
            m = S[k] & (fid >= 0) & ~specular(V.frames[k])
            ys, xs = np.nonzero(m)
            qj, zj = V.project(V.unproject(xs, ys, Dk[ys, xs], k), j)
            xi, yi = np.round(qj[:, 0]).astype(int), np.round(qj[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < V.W) & (yi >= 0) & (yi < V.H) & (zj > 0)
            ok[ok] &= S[j][yi[ok], xi[ok]] & ~specular(V.frames[j])[yi[ok], xi[ok]]
            img = np.zeros((V.H, V.W, 3), np.float32)
            img[ys, xs] = remap_pts(V.frames[j], qj[:, 0], qj[:, 1]).astype(np.float32)
            okm = np.zeros((V.H, V.W), bool)
            okm[ys[ok], xs[ok]] = True
            warped[name] = img
            oks.append(okm)
        common = oks[0] & oks[1]
        if common.sum() < 500:
            continue
        res['pairs'].append([k, j])
        for name, img in warped.items():
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            res[f'{name}_lncc'].append(local_ncc(g, gk, common))
            res[f'{name}_l1'].append(float(np.abs(img - V.frames[k].astype(np.float32))[common].mean()))
    out = {k: (quality.summarize(v) if k != 'pairs' else v) for k, v in res.items()}
    out['per_pair'] = {k: np.round(v, 4).tolist() for k, v in res.items() if k != 'pairs'}
    return out


def baseline_metrics(V, S, ks):
    per = evaluate(V, old_backdrop(), S, ks)
    return {k: quality.summarize(v) for k, v in per.items()}


def main(argv):
    ver = argv[0]
    p = VERSIONS[ver]
    M = build(ver, p)
    cams = None
    if p.get('refine_cams'):
        V0 = M['V']
        Se = static_masks(V0, EVAL_MASK)
        for r in range(p['refine_cams']):
            R, info = refine_cameras(M['V'], M['X'], M['F'], M['uv'], M['tex'], Se, iters=2)
            M = build(ver, p, V=with_cameras(V0, R))
            log(f'round {r}: correction median {np.median(info["correction_deg"]):.3f} deg, max {info["correction_deg"].max():.2f}')
        info['correction_deg'] = np.degrees([np.arccos(np.clip((np.trace(R[k] @ V0.R[k].T) - 1) / 2, -1, 1))
                                             for k in range(V0.n)])
        cams = dict(R=R, pos=V0.pos, f=V0.f, **info)
        np.savez(OUT / ver / 'cams_refined.npz', **cams,
                 note='per-frame camera rotations registered to the backdrop (rows = camera axes in world, as views.R); '
                      'pos and f unchanged; correction_deg = angle to the clip cameras')
        M['Vr'] = M['V']
        M['V'] = V0                                      # contract metrics use the clip's own cameras
    save(ver, p, M)
    base = None
    if '--baseline' in argv:
        Se = static_masks(M['V'], EVAL_MASK)
        base = baseline_metrics(M['V'], Se, list(range(M['V'].n)))
        base['note'] = 'outputs/variants/chole_a+sift/scene backdrop_*.obj (frame-0 depth + keyframe patches), same static masks'
        render_sheet(M['V'], old_backdrop(), Se, [0, 30, 60, 100, 140, 180, 220, 250], OUT / ver / 'work/render_sheet_old.jpg')
    q = report(ver, p, M, base)
    if '--holdout' in argv:
        ph = dict(p, holdout_kf=True)
        Mh = build(ver, ph)
        Se = static_masks(Mh['V'], EVAL_MASK)
        kf = Mh['V'].keyframes
        per = evaluate(Mh['V'], [(Mh['X'], Mh['F'], Mh['uv'], Mh['tex'])], Se, kf, Mh['liver'], Mh['face_label'])
        hq = dict(note='keyframes (every 10th frame) left out of depth + texture fusion, evaluated on them',
                  heldout_keyframes={k: quality.summarize(v) for k, v in per.items()},
                  same_frames_in_full_model=q['summary_keyframes'])
        (OUT / ver / 'quality_holdout.json').write_text(json.dumps(hq, indent=1))
        q['holdout_keyframes'] = hq['heldout_keyframes']
        (OUT / ver / 'quality.json').write_text(json.dumps(q, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        log('HOLDOUT', {k: v.get('median') for k, v in hq['heldout_keyframes'].items()})
        log('FULL   ', {k: v.get('median') for k, v in q['summary_keyframes'].items()})
    s = q['summary_all_frames']
    log('SUMMARY', ver, {k: s[k].get('median') for k in s}, 'spread', q['spread_mm'], q['mesh_health'])
    if base:
        log('BASELINE', {k: base[k].get('median') for k in base if isinstance(base[k], dict)})


def strand_masks(V, redo=False):
    path = OUT / 'masks_strands.npz'
    if path.exists() and not redo:
        d = np.load(path)
        return np.unpackbits(d['pink_strands'], axis=-1)[..., :V.W].astype(bool)
    m = views.segment_object(V, STRAND_PROMPTS)
    np.savez_compressed(path, pink_strands=np.packbits(m, axis=-1), prompts=json.dumps(STRAND_PROMPTS))
    return m


if __name__ == '__main__':
    if sys.argv[1:2] == ['masks']:
        V = views.load('chole_a', 'sift')
        m = liver_masks(V, redo=True)
        log('liver mask fraction', m.mean(axis=(1, 2))[::25].round(3))
    elif sys.argv[1:2] == ['strands']:
        V = views.load('chole_a', 'sift')
        m = strand_masks(V, redo=True)
        log('pink strand mask fraction', m.mean(axis=(1, 2))[::25].round(4))
    else:
        main(sys.argv[1:])
