"""Step 5 of v2 (PLAN_V2.md): the static background of a clip, as a COLLISION surface plus small closed 3D bodies.

    PYTHONPATH=. .venv/bin/python -m t2s.background <clip> [vNN] [--holdout] [--seg vNN]

-> outputs/t2s/<clip>/background/vNN/{model.npz, occupancy.npz, texture.png, background.obj, quality.json, NOTES.md,
   sheet.jpg, views3d.jpg, canvas.jpg, work/}

v1's backdrop (r2s/tissue/backdrop.py, outputs/iter/tissues/backdrop/v08) was a visual-only depth field; the
gallbladder sank into it in every frame and it had fins at the canvas edge. Here:

0. objects    every tracked object of the newest finished segmentation is classified automatically (spec + masks):
              instrument; organ (primary / secondary organ of the scene_spec that its own agent models: excluded, the
              surface is filled BEHIND it); surface (context organs and background items that are large and move no
              more than the static pixels, e.g. the liver of the chole clips, omentum, diaphragm wall); body
              candidate (small background items: blood, fat, gauze, unidentified organs). A candidate becomes a body
              when it protrudes from the surface fitted without it, otherwise it is flat and stays surface texture.
              Motion of every object = optical flow minus the flow the camera alone would cause (median EPE), as a
              ratio to the unlabelled static pixels.
1. surface    static pixels of all frames -> metric depth -> world -> binned in a virtual reference camera (mean pose,
              canvas = union of all footprints); per cell robust median over frames, spread = MAD. One sparse solve of
              a confidence-weighted thin-plate height field over a single hole-free domain (one component by
              construction) fills the hidden cells; it is constrained to stay >= margin BEHIND the back of every organ
              in every frame (organ agent's 4D model where one exists, else front depth + thickness from the mask
              width), so organs cannot start inside it. Meshed as a regular grid; a closed slab (surface + back + side
              walls, watertight) and a MuJoCo height field are derived from it.
2. bodies     each body = 1..k convex pieces (k = smallest number of k-means clusters of its canvas footprint whose 2D
              hulls cover the footprint well), each the convex hull of the fused multi-view front points and the
              surface points under the footprint (closed, sits on the surface). Static unless a per-frame translation
              explains the video clearly better (then verts4d).
3. texture    one reference-canvas texture for surface and bodies: per texel the median colour of the frames that see
              it as background, unoccluded and not specular (v1 recipe, push-pull fill of unseen texels).
4. checks     per frame (and per third of the clip): coverage of static pixels, depth residual, photometric L1 / NCC /
              local NCC (also on held-out frames with --holdout), front violations against static pixels, organs,
              instruments and the organs' 4D models (vertices behind the surface along the camera rays), fragment
              and steep-face counts; per body IoU / depth residual; pictures sheet.jpg / views3d.jpg / canvas.jpg.
5. contract   model.npz: rest_verts, faces, uv (+ texture.png), labels, flags, reference camera, hfield_*, slab_*;
              bodies body<i>_{verts, faces, piece, uv, rgb[, verts4d]}, body_names, body_static.
              occupancy.npz: per frame (half resolution) the depth of the first background hit along every camera ray
              (surface / body id) and the video depth; `Background(clip).ray_distance(X, k)` gives the signed
              distance of world points to the background along frame k's rays (> 0 free, < 0 behind / inside).
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from r2s import quality
from r2s.tissue import backdrop as B
from . import views2
from . import data as D

OUT = D.OUT
dil, ero = B.dil, B.ero


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- versions (cumulative)
BASE = dict(
    seg=None,                 # segmentation version (None = newest finished one)
    cell=2, grid=6,           # canvas cell (px) for the fusion, mesh step (canvas px)
    tool_r=10, organ_r=10, body_r=6, border=8,   # px: dilation of instrument / organ / body masks, image border
    min_obs=5, min_foot=6, dom_pad=6,            # cells: observations for a known cell, footprint, domain pad around known
    reject_iters=2, reject_k=2.5, reject_min=0.0025,
    fit_sigma=0.003, lam1=0.0, lam2=3.0,          # thin plate (v1 v05 hold-out choice)
    organ_margin=0.003,       # m: surface at least this far behind the back of every organ (all frames)
    organ_every=2,            # frames: organ back depths sampled every n frames
    thick_min=0.003, thick_max=0.030,              # m: fallback organ thickness clamp (front depth + thickness)
    enforce_w=0.05,           # lower bound enforced in cells whose data weight is below this (else: conflict reported)
    sharp_rel=0.6,            # frames with Laplacian sharpness < sharp_rel x median are 'blurred' (down-weighted)
    big_area=0.10,            # objects covering more than this fraction of the scope on average are surface, not bodies
    motion_ratio=1.5,         # flow residual / unlabelled residual above this = the object moves
    protrude_mm=2.0,          # body candidates must stand out of the surface by this much (median), else flat
    body_min_obs=3, body_gap=0.001, piece_cover=0.7, max_pieces=6, carve_tol=0.003, carve_frac=0.35,
    spike_mm=3.0, spike_r=2,  # cells further than this from their 5x5 local median get no data weight
    move_gain=0.03,           # a body is moving if per-frame translations improve its IoU by this much (median)
    body_min_iou=0.3,         # a carved body must reproduce its own mask at least this well (median IoU), else surface
    vis_tol=0.006, tex_every=2, tex_min_sharp=4, tex_min_obs=3, detail_gain=0.6, detail_clip=12,
    flow_reject=False,        # also drop pixels whose flow disagrees with the camera-only flow (untracked movers)
    flow_gap=4, flow_px=6.0, flow_k=3.0,
    border_in=None,           # px: interior border; cells seen by >= min_obs interior pixels use only those
    scale_iters=0,            # rounds of per-frame depth scale against the multi-view median
    fs_iters=0, fs_tol=0.004, fs_frac=0.2,   # visibility check against the organs (rounds, tolerance, fraction of frames)
    fs_instruments=False, shaft_d=0.005,      # ... and against the instruments (their front depth + shaft diameter)
    holdout_mod=None,         # (m, r): frames with k % m == r left out of the fusion (hold-out evaluation)
)
VERSIONS = {
    # v01: first pass (bodies from the fused canvas front + the surface under them; lam2 3; no spike rejection) --
    # rerunning it with the current code gives v02-like bodies
    'v01': dict(BASE, lam2=3.0, spike_mm=None, enforce_w=0.25),
    # v02: bodies by multi-view space carving, spike cells dropped, smooth organ lower bound enforced only where the
    # data are weak, stronger thin plate
    'v02': dict(BASE, lam2=30.0),
    'v02a': dict(BASE, lam2=3.0),
    # v03: per-frame depth scale against the multi-view median (2 rounds), interior pixels (24 px border) preferred,
    # pixels whose flow disagrees with the camera-only flow (untracked movers) dropped, tube thickness from the medial
    # axis
    'v03': dict(BASE, lam2=30.0, scale_iters=2, border_in=24, flow_reject=True),
    # v04: visibility check of the surface against the visible organs (cells where an organ is repeatedly seen behind
    # the surface lose their static data and get the organ lower bound), flow threshold 2x the static residual
    'v04': dict(BASE, lam2=30.0, scale_iters=2, border_in=24, flow_reject=True, flow_k=2.0, fs_iters=3),
    # v05: v04 + a carved body must reproduce its own mask (median IoU >= 0.3), else it stays surface; hold-out run
    'v05': dict(BASE, lam2=30.0, scale_iters=2, border_in=24, flow_reject=True, flow_k=2.0, fs_iters=3, body_min_iou=0.3),
    # v06: v05 + visibility check also against the instruments; collision height field as 6 x 4 tiles along the camera
    # rays (z-buffered, uncovered cells at the far plane) instead of one interpolated field; newest organ models
    'v06': dict(BASE, lam2=30.0, scale_iters=2, border_in=24, flow_reject=True, flow_k=2.0, fs_iters=3, body_min_iou=0.3,
                fs_instruments=True),
    # v07: v06, but cells contradicted by a visible organ are pushed only behind that organ's FRONT (+3 mm), not behind
    # the organ model's back (chole_derot: v06 pushed 28 % of the observed cells 13 mm back)
    'v07': dict(BASE, lam2=30.0, scale_iters=2, border_in=24, flow_reject=True, flow_k=2.0, fs_iters=3, body_min_iou=0.3,
                fs_instruments=True, fs_min_push=True),
    # v08: v07 with the visibility pushes smoothed like the organ bound (5x5 closing + median; v07 left per-cell spikes,
    # worst in chole_derot) and taken from the mean, not the maximum, of the organ / instrument fronts seen behind
    'v08': dict(BASE, lam2=30.0, scale_iters=2, border_in=24, flow_reject=True, flow_k=2.0, fs_iters=3, body_min_iou=0.3,
                fs_instruments=True, fs_min_push=True, fs_mean=True, fs_smooth=True),
}


# ---------------------------------------------------------------- inputs
def seg_versions(clip):
    d = OUT / clip / 'seg'
    return sorted(p.name for p in d.glob('v[0-9][0-9]') if (p / 'masks.npz').exists() and (p / 'qc.json').exists())


def load_masks(clip, V, ver=None):
    """{name: (n, H, W) bool} of every tracked object of a segmentation version; instrument pixels (dilated 1 px) are
    removed from the other objects (instruments are always in front)."""
    ver = ver or seg_versions(clip)[-1]
    q = json.loads((OUT / clip / 'seg' / ver / 'qc.json').read_text())
    z = np.load(OUT / clip / 'seg' / ver / 'masks.npz')
    M = {n: np.unpackbits(z[n], axis=-1)[..., :V.W].astype(bool) for n in q['objects']}
    ins = [n for n in M if n.startswith('instrument_')]
    if ins:
        cov = np.any([M[n] for n in ins], 0)
        k3 = np.ones((3, 3), np.uint8)
        cov = np.stack([cv2.dilate(c.astype(np.uint8), k3).astype(bool) for c in cov])
        for n in M:
            if n not in ins:
                M[n] &= ~cov
    return ver, M


def latest_organ_models(clip, pin=None):
    """{mask name: dict(organ, version, verts4d (n, N, 3), faces)} of the organ agent's newest finished models
    (or the versions pinned as {organ dir: version})."""
    out = {}
    d = OUT / clip / 'organs'
    for od in sorted(p for p in d.glob('*') if p.is_dir()) if d.exists() else []:
        vs = sorted(p for p in od.glob('v[0-9][0-9]') if (p / 'model.npz').exists())
        if pin and od.name in pin:
            vs = [od / pin[od.name]]
        if not vs:
            continue
        z = np.load(vs[-1] / 'model.npz')
        if 'verts4d' not in z.files:
            continue
        q = json.loads((vs[-1] / 'quality.json').read_text()) if (vs[-1] / 'quality.json').exists() else {}
        mask = q.get('mask') or od.name
        out[mask] = dict(organ=od.name, version=vs[-1].name, verts4d=z['verts4d'].astype(np.float64),
                         faces=z['faces'].astype(np.int64),
                         depth_scale=z['depth_scale'] if 'depth_scale' in z.files else None)
    return out


def flow_residual(V, k, j, dis=None):
    """|video optical flow k->j - flow the camera motion alone would cause with frame k's depth| (H, W) px."""
    dis = dis or cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = dis.calc(cv2.cvtColor(V.frames[k], cv2.COLOR_RGB2GRAY), cv2.cvtColor(V.frames[j], cv2.COLOR_RGB2GRAY), None)
    ys, xs = np.mgrid[0:V.H, 0:V.W]
    q, _ = V.project(V.unproject(xs.ravel(), ys.ravel(), V.depth(k).ravel(), k), j)
    pf = q.reshape(V.H, V.W, 2) - np.stack([xs, ys], -1)
    return np.linalg.norm(fl - pf, axis=-1)


def motion_stats(V, M, ins, gap=4, every=6):
    """Per object: median flow residual over its (eroded, instrument-free) pixels, and the same for unlabelled pixels."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    anyobj = np.any(list(M.values()), 0)
    res = {n: [] for n in list(M) + ['_unlabelled']}
    for k in range(0, V.n - gap, every):
        j = k + gap
        e = flow_residual(V, k, j, dis)
        near_tool = dil(ins[k] | ins[j], 7)
        for n, m in M.items():
            if n.startswith('instrument_'):
                continue
            s = ero(m[k], 4) & V.valid & ~near_tool
            if s.sum() > 200:
                res[n].append(float(np.median(e[s])))
        s = ero(~anyobj[k], 6) & V.valid
        if s.sum() > 200:
            res['_unlabelled'].append(float(np.median(e[s])))
    base = np.median(res['_unlabelled']) if res['_unlabelled'] else np.nan
    out = {}
    for n, v in res.items():
        if v:
            thirds = [round(float(np.median(a)), 2) for a in np.array_split(np.array(v), 3)]
            out[n] = dict(epe_px=round(float(np.median(v)), 2), ratio=round(float(np.median(v) / base), 2), thirds=thirds)
    return out


def classify(clip, V, M, motion, p):
    """kind per object: instrument / organ / surface / candidate (body or flat, decided after the surface fit)."""
    spec = json.loads((OUT / clip / 'spec' / 'scene_spec.json').read_text())
    sp = OUT / clip / 'organs' / 'selection.json'
    sel = json.loads(sp.read_text()) if sp.exists() else dict(fit=[], skipped=[])
    by_mask = {}
    for e in sel.get('fit', []) + sel.get('skipped', []):
        if e.get('mask'):
            by_mask.setdefault(e['mask'], []).append(e)
    spec_organs = {o['name']: o for o in spec.get('organs', [])}
    valid = V.valid.mean()
    out = {}
    for n, m in M.items():
        area = float(m.mean() / valid)
        mo = motion.get(n, {})
        ratio = mo.get('ratio', np.nan)
        ents = by_mask.get(n, [])
        if n not in by_mask and n in spec_organs:
            o = spec_organs[n]
            ents = [dict(name=n, role=o.get('role'), consistency=o.get('consistency'))]
        roles = {e.get('role') for e in ents}
        cons = {e.get('consistency') for e in ents}
        if n.startswith('instrument_'):
            kind, why = 'instrument', 'tracked instrument'
        elif 'primary' in roles:
            kind, why = 'organ', f"primary organ of the spec ({', '.join(e['name'] for e in ents)}): own agent, moves"
        elif ents and area > p['big_area'] and ratio < p['motion_ratio'] and roles <= {'secondary', 'context'} \
                and not (cons & {'tubular', 'fluid-filled'}):
            kind, why = 'surface', (f"{'/'.join(sorted(r for r in roles if r))} organ ({', '.join(e['name'] for e in ents)}), "
                                    f"large ({area:.0%} of the scope) and static (flow residual x{ratio:.2f} of the "
                                    f"unlabelled pixels): part of the background wall")
        elif ents and roles & {'secondary'}:
            kind, why = 'organ', f"secondary organ ({', '.join(e['name'] for e in ents)}, {'/'.join(c for c in cons if c)}): own agent"
        elif ents and ratio >= p['motion_ratio']:
            kind, why = 'organ', f"context organ that moves (flow residual x{ratio:.2f}): left to the organ agents"
        elif area > p['big_area']:
            kind, why = 'surface', f"large background item ({area:.0%} of the scope, flow residual x{ratio:.2f}): part of the surface"
        else:
            kind, why = 'candidate', (f"small background item ({area:.1%} of the scope, flow residual x{ratio:.2f}): body "
                                      f"if it stands out of the surface")
        out[n] = dict(kind=kind, why=why, area=round(area, 4), motion=mo)
    return out


# ---------------------------------------------------------------- lifting all pixels into the reference canvas
class Lift:
    """Every pixel of every frame lifted with its metric depth and binned in the reference canvas (cell index and
    reference depth), cached for the surface, the bodies and the lower bound."""

    def __init__(self, V, ref, cs):
        self.Hg, self.Wg = ref.Hc // cs, ref.Wc // cs
        m = self.Hg * self.Wg
        ys, xs = np.mgrid[0:V.H, 0:V.W]
        ys, xs = ys.ravel(), xs.ravel()
        self.cell = np.full((V.n, V.H * V.W), -1, np.int32)
        self.zr = np.zeros((V.n, V.H * V.W), np.float32)
        self.foot = np.zeros(m, np.int32)
        for k in range(V.n):
            uc, vc, zr = ref.project(V.unproject(xs, ys, V.depth(k)[ys, xs], k))
            ci, cj = np.floor(uc / cs).astype(np.int64), np.floor(vc / cs).astype(np.int64)
            inb = (ci >= 0) & (ci < self.Wg) & (cj >= 0) & (cj < self.Hg) & V.valid.ravel() & (zr > 0)
            c = np.where(inb, cj * self.Wg + ci, -1)
            self.cell[k], self.zr[k] = c, zr
            self.foot[np.unique(c[inb])] += 1

    def fuse(self, S):
        """Per frame and cell: mean reference depth of the selected pixels (n, cells), NaN where none."""
        n, m = S.shape[0], self.Hg * self.Wg
        Dm = np.full((n, m), np.nan, np.float32)
        for k in range(n):
            s = S[k].ravel() & (self.cell[k] >= 0)
            c = self.cell[k][s]
            if not len(c):
                continue
            cnt = np.bincount(c, minlength=m)
            sm = np.bincount(c, self.zr[k][s], minlength=m)
            hit = cnt > 0
            Dm[k, hit] = sm[hit] / cnt[hit]
        return Dm

    def count(self, S, w=None):
        """Per cell: number of frames in which a selected pixel lands there (or sum of weights per frame)."""
        m = self.Hg * self.Wg
        out = np.zeros(m, np.float32)
        for k in range(S.shape[0]):
            s = S[k].ravel() & (self.cell[k] >= 0)
            u = np.unique(self.cell[k][s])
            out[u] += 1 if w is None else w[k]
        return out


def robust_median(Dm, p, use=None):
    """Median over frames per cell with outlier rejection (> max(k MAD, min)); `use` (n,) bool restricts to frames
    where enough of them see the cell (sharp frames first)."""
    Dm = Dm.copy()
    if use is not None:
        cnt_use = np.isfinite(Dm[use]).sum(0)
        Dm[np.ix_(~use, cnt_use >= p['min_obs'])] = np.nan
    med, cnt = B.sorted_median(Dm)
    for _ in range(p['reject_iters']):
        mad = B.sorted_median(np.abs(Dm - med))[0] * 1.4826
        thr = np.maximum(p['reject_k'] * np.nan_to_num(mad, nan=1.0), p['reject_min'])
        Dm[np.abs(Dm - med) > thr] = np.nan
        med, cnt = B.sorted_median(Dm)
    mad = B.sorted_median(np.abs(Dm - med))[0] * 1.4826
    return med, cnt, mad, Dm


# ---------------------------------------------------------------- organ backs (lower bound of the surface)
def far_depth(ref, X, F, Hc, Wc):
    """Largest reference depth of a closed mesh per canvas pixel (its back surface as seen from the reference camera),
    NaN where it does not cover; by rasterising C - z."""
    q, z = ref.cam.project(X, ref.R, ref.f, ref.pos)
    q = q - np.array([ref.ou, ref.ov])
    C = float(z.max()) + 0.05
    zb, fid, _ = B.raster(q, C - z, F, Hc, Wc, max_box=1024)
    return np.where(fid >= 0, C - zb, np.nan)


def organ_lower_bound(V, ref, L, M, cls, models, p):
    """Per canvas cell: max over frames of the back depth of every organ + margin (NaN where no organ ever covers).
    Organs with a 4D model: the model's back surface; others: video front depth + thickness (2 x the largest inscribed
    radius of the mask, metric, clamped)."""
    cs, Hg, Wg = p['cell'], L.Hg, L.Wg
    lower = np.full(Hg * Wg, -np.inf)
    src = {}
    organs = [n for n, c in cls.items() if c['kind'] == 'organ']
    ks = list(range(0, V.n, p['organ_every']))
    if p.get('holdout_mod'):
        ks = [k for k in ks if k % p['holdout_mod'][0] != p['holdout_mod'][1]]
    for n in organs:
        mdl = models.get(n)
        if mdl is not None:
            src[n] = f"organ model {mdl['organ']}/{mdl['version']} (4D back surface)"
            for k in ks:
                fz = far_depth(ref, mdl['verts4d'][k], mdl['faces'], ref.Hc, ref.Wc)
                g = fz[cs // 2::cs, cs // 2::cs][:Hg, :Wg].ravel()
                ok = np.isfinite(g)
                lower[ok] = np.maximum(lower[ok], g[ok] + p['organ_margin'])
            continue
        thick = []
        for k in ks:
            m0 = M[n][k] & V.valid
            m = ero(m0, 4)                         # off the mask edge: depth there bleeds into the background
            if m.sum() < 30:
                continue
            dt = cv2.distanceTransform(m0.astype(np.uint8), cv2.DIST_L2, 5)
            ridge = (dt >= 2) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)     # medial axis (approx.)
            dz = cv2.medianBlur(V.depth(k).astype(np.float32), 5)
            ys, xs = np.nonzero(m)
            z = dz[ys, xs]
            rad = np.median(dt[ridge]) if ridge.sum() > 5 else dt.max()                 # typical half width
            th = float(np.clip(2 * rad * np.median(z) / V.f[k], p['thick_min'], p['thick_max']))
            thick.append(th)
            sub = slice(None, None, 2)
            X = V.unproject(xs[sub], ys[sub], z[sub] + th, k)
            uc, vc, zr = ref.project(X)
            ci, cj = np.floor(uc / cs).astype(int), np.floor(vc / cs).astype(int)
            inb = (ci >= 0) & (ci < Wg) & (cj >= 0) & (cj < Hg)
            c = cj[inb] * Wg + ci[inb]
            np.maximum.at(lower, c, zr[inb] + p['organ_margin'])
        src[n] = (f"no organ model: video front depth + thickness {np.median(thick) * 1e3:.1f} mm (2 x the median "
                  f"inscribed radius along the medial axis, median over frames)") if thick else 'never visible'
    lower[~np.isfinite(lower)] = np.nan
    return smooth_lower(lower.reshape(Hg, Wg)), src


def smooth_lower(lower):
    """Smooth, spike-free lower bound: grey closing over 2 cells, then the median of 5x5 cells (NaN-aware via fill)."""
    fin = np.isfinite(lower)
    if not fin.any():
        return lower
    lo = np.where(fin, lower, 0).astype(np.float32)
    lo = cv2.morphologyEx(lo, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lo = cv2.medianBlur(lo, 5)
    fin2 = cv2.morphologyEx(fin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return np.where(fin2 & (lo > 0), lo, np.nan)


# ---------------------------------------------------------------- surface
def fit_surface(Z, w, dom, lam1, lam2, lower=None, enforce=None, iters=6):
    """Confidence-weighted thin-plate height field over dom (v1's fit), with z >= lower enforced where `enforce`."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    L, idx = B.grid_laplacian(dom)
    Q = (lam1 * L + lam2 * (L.T @ L)).tocsr()
    d = Z[dom].astype(float)
    ww = w[dom].astype(float)
    lo = None
    if lower is not None:
        lo = np.where(enforce[dom] if enforce is not None else True, lower[dom], np.nan)
    z = None
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


def domain(known, foot, p):
    """One hole-free connected domain: footprint cells near known cells (no extrapolated wings), largest component."""
    pad = dil(known, p['dom_pad'])
    d = (foot >= p['min_foot']) & pad
    d = B.largest_cc(d)
    d = B.fill_holes(d)
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    return B.fill_holes(B.largest_cc(d))


def local_median(Z, known, r=2):
    """Median of the known cells in a (2r+1)^2 window (NaN where none): spike reference."""
    from scipy import ndimage
    A = np.where(known, Z, np.nan)
    return ndimage.generic_filter(A, np.nanmedian, size=2 * r + 1, mode='nearest')


def surface_solve(Z, cnt, mad, cnt_sharp, dom, known, lower, p):
    """Data weights (views, spread, sharp frames), spike cells (> spike_mm from the local median) dropped, then the
    thin-plate fit with the organ lower bound enforced in cells without trustworthy data."""
    import warnings
    madg = np.nan_to_num(mad, nan=0.02)
    w = np.minimum(cnt, 50) / 50 / (1 + (madg / p['fit_sigma']) ** 2)
    w = np.where(cnt_sharp >= p['min_obs'], w, w * 0.2)
    w = np.where(known, w, 0.0)
    if p.get('spike_mm'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lm = local_median(Z, known, p.get('spike_r', 2))
        spike = known & np.isfinite(lm) & (np.abs(Z - lm) > p['spike_mm'] * 1e-3)
        w = np.where(spike, 0.0, w)
    enforce = (w < p['enforce_w'])
    Zf = fit_surface(np.where(known, Z, 0), w, dom, p['lam1'], p['lam2'], lower, enforce)
    return Zf, w, enforce


# ---------------------------------------------------------------- bodies
def convex_pieces(P, vol, p, seed=0):
    """Split a voxel set (centres (N, 3), voxel volume) into the fewest k-means clusters whose convex hulls are
    mostly filled by their voxels (sum voxel volume / sum hull volume >= piece_cover)."""
    from scipy.spatial import ConvexHull
    from scipy.cluster.vq import kmeans2
    best, best_r = np.zeros(len(P), int), -1
    for k in range(1, p['max_pieces'] + 1):
        lab = np.zeros(len(P), int) if k == 1 else kmeans2(P, k, minit='++', seed=np.random.default_rng(seed))[1]
        hv = 0.0
        for i in range(k):
            Q = P[lab == i]
            if len(Q) < 8:
                continue
            try:
                hv += ConvexHull(Q).volume
            except Exception:
                pass
        r = len(P) * vol / max(hv, 1e-15)
        if r > best_r + 0.02:
            best, best_r = lab, r
        if r >= p['piece_cover']:
            break
    return best, min(best_r, 1.0)


def hull_mesh(P):
    import trimesh
    h = trimesh.convex.convex_hull(P)
    return np.asarray(h.vertices), np.asarray(h.faces)


def carve_body(V, name, Mb, M, cls, ins, ref, Zsurf, hold, p):
    """Static closed body by multi-view space carving in the reference camera frame. A voxel is kept if, over the
    frames that see it, it is inside the body's silhouette (mask dilated 2 px) and not in front of the body's video
    depth (> tol), it never sits in front of / on another visible surface (organ, static, other item), most of the
    time (carve fraction < carve_frac), and it lies in front of the static surface (which it may touch, body_gap).
    Thickness is capped at the body's larger lateral extent. Returns the body (convex pieces) and its numbers."""
    from scipy import ndimage
    tol, cs = p['carve_tol'], p['cell']
    ks = [k for k in range(0, V.n, 2) if not hold[k]]
    pts = []
    for k in ks:
        m = ero(Mb[k], 3) & V.valid & ~dil(ins[k], 6)
        if m.sum() < 30:
            continue
        ys, xs = np.nonzero(m[::2, ::2])
        ys, xs = ys * 2, xs * 2
        dz = cv2.medianBlur(V.depth(k).astype(np.float32), 5)
        pts.append(V.unproject(xs, ys, dz[ys, xs], k))
    info = dict(name=name, frames_with_points=len(pts))
    if len(pts) < 3:
        return None, info
    P = np.vstack(pts)
    pc = (P - ref.pos) @ ref.R.T
    lo, hi = np.percentile(pc, 1, 0), np.percentile(pc, 99, 0)
    lat = max(hi[0] - lo[0], hi[1] - lo[1])
    lo[:2] -= 0.002
    hi[:2] += 0.002
    lo[2] -= 0.004
    hi[2] = lo[2] + min(lat + 0.008, 0.045)
    s = float(np.clip(max(hi - lo) / 64, 0.0003, 0.0012))
    ax = [np.arange(lo[i] + s / 2, hi[i], s) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
    shape = G.shape[:3]
    G = G.reshape(-1, 3)
    # in front of the static surface (touching allowed): reference depth <= surface depth + gap
    zr = G[:, 2]
    uc = ref.f * G[:, 0] / zr + ref.cam.W / 2 - ref.ou
    vc = ref.f * G[:, 1] / zr + ref.cam.H / 2 - ref.ov
    ci, cj = np.floor(uc / cs).astype(int), np.floor(vc / cs).astype(int)
    Hg, Wg = Zsurf.shape
    inb = (ci >= 0) & (ci < Wg) & (cj >= 0) & (cj < Hg)
    zs = np.full(len(G), np.inf)
    zs[inb] = Zsurf[cj[inb], ci[inb]]
    zs[~np.isfinite(zs)] = np.inf
    front_of_surface = zr <= zs + p['body_gap']
    Xw = ref.pos + G @ ref.R
    sup = np.zeros(len(G), np.int32)
    carve = np.zeros(len(G), np.int32)
    organs = [n for n, c in cls.items() if c['kind'] == 'organ']
    for k in ks:
        q, z = V.project(Xw, k)
        xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < V.W) & (yi >= 0) & (yi < V.H) & (z > 0)
        ok[ok] &= V.valid[yi[ok], xi[ok]]
        if not ok.any():
            continue
        xi, yi, zz = xi[ok], yi[ok], z[ok]
        d = cv2.medianBlur(V.depth(k).astype(np.float32), 5)[yi, xi]
        inb_ = dil(Mb[k], 2)[yi, xi]
        tool = dil(ins[k], 3)[yi, xi]
        org = np.zeros(len(xi), bool)
        for n in organs:
            org |= M[n][k][yi, xi]
        infront = zz < d - tol
        on = np.abs(zz - d) <= tol
        c = np.zeros(len(xi), bool)
        sp = np.zeros(len(xi), bool)
        c |= inb_ & infront & ~tool
        sp |= inb_ & ~infront & ~tool
        c |= ~inb_ & ~tool & infront
        c |= ~inb_ & ~tool & ~org & on
        idx = np.nonzero(ok)[0]
        carve[idx[c]] += 1
        sup[idx[sp]] += 1
    keep = front_of_surface & (sup >= p['body_min_obs']) & (carve <= p['carve_frac'] * (carve + sup))
    vox = keep.reshape(shape)
    vox = ndimage.binary_opening(vox, iterations=1)
    lab, nl = ndimage.label(vox)
    if nl == 0:
        info.update(voxels=0)
        return None, info
    sizes = ndimage.sum(vox, lab, range(1, nl + 1))
    vox = np.isin(lab, 1 + np.nonzero(sizes >= 0.1 * sizes.max())[0])
    vox = ndimage.binary_fill_holes(vox)
    Gk = G.reshape(shape + (3,))[vox]
    vol = s ** 3
    info.update(voxels=int(vox.sum()), voxel_mm=round(s * 1e3, 3), volume_ml=round(float(vox.sum() * vol * 1e6), 3),
                components=int(nl), carve_tol_mm=tol * 1e3)
    if vox.sum() < 30:
        return None, info
    # protrusion: per column (x, y), front-most voxel vs the surface depth along the reference ray
    cols = {}
    for x, y, z in Gk:
        key = (round(x / s), round(y / s))
        if key not in cols or z < cols[key][0]:
            cols[key] = (z, x, y)
    F_ = np.array(list(cols.values()))
    zf_ = F_[:, 0]
    u_ = ref.f * F_[:, 1] / zf_ + ref.cam.W / 2 - ref.ou
    v_ = ref.f * F_[:, 2] / zf_ + ref.cam.H / 2 - ref.ov
    ci, cj = np.clip(np.floor(u_ / cs).astype(int), 0, Wg - 1), np.clip(np.floor(v_ / cs).astype(int), 0, Hg - 1)
    pr = Zsurf[cj, ci] - zf_
    pr = pr[np.isfinite(pr)]
    info.update(protrude_med_mm=round(float(np.median(pr)) * 1e3, 2) if len(pr) else None,
                protrude_p90_mm=round(float(np.percentile(pr, 90)) * 1e3, 2) if len(pr) else None)
    lab, fill = convex_pieces(Gk, vol, p)
    Xs, Fs, piece = [], [], []
    off = 0
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s / 2
    for i in np.unique(lab):
        Q = Gk[lab == i]
        if len(Q) < 8:
            continue
        Qc = (Q[:, None] + corners[None]).reshape(-1, 3)
        Xh, Fh = hull_mesh(ref.pos + Qc @ ref.R)
        Xs.append(Xh)
        Fs.append(Fh + off)
        piece.append(np.full(len(Xh), len(piece)))
        off += len(Xh)
    X, F, piece = np.vstack(Xs), np.vstack(Fs), np.concatenate(piece)
    info.update(pieces=int(piece.max() + 1), hull_fill=round(float(fill), 3))
    return dict(name=name, X=X, F=F, piece=piece), info


def subdivide_body(X, F, piece, max_edge=0.0015):
    """Subdivide each convex piece's faces (loop-free midpoint split) so that per-vertex colour / UV is dense."""
    import trimesh
    Xo, Fo, Po = [], [], []
    off = 0
    for i in np.unique(piece):
        vi = np.nonzero(piece == i)[0]
        remap = -np.ones(len(X), int)
        remap[vi] = np.arange(len(vi))
        f = remap[F[np.all(np.isin(F, vi), 1)]]
        x, f = trimesh.remesh.subdivide_to_size(X[vi], f, max_edge=max_edge, max_iter=6)
        m = trimesh.Trimesh(x, f, process=True)
        m.fix_normals()
        Xo.append(np.asarray(m.vertices))
        Fo.append(np.asarray(m.faces) + off)
        Po.append(np.full(len(m.vertices), i))
        off += len(m.vertices)
    return np.vstack(Xo), np.vstack(Fo), np.concatenate(Po)


def body_motion(V, body, Mb, occl, p):
    """Per-frame translation of a body that would explain the video (image shift of the mask vs the render, depth
    offset along the view); returns (verts4d, gain info). The body counts as moving only if this clearly helps."""
    X, F = body['X'], body['F']
    T = np.zeros((V.n, 3))
    iou_s, iou_m = [], []
    for k in range(V.n):
        q, z = V.project(X, k)
        zb, fid, _ = B.raster(q, z, F, V.H, V.W)
        m = fid >= 0
        obs = Mb[k] & ~occl[k]
        if obs.sum() < 50 or m.sum() < 50:
            continue
        ys, xs = np.nonzero(obs)
        yr, xr = np.nonzero(m & ~occl[k])
        if len(xr) < 50:
            continue
        du, dv = xs.mean() - xr.mean(), ys.mean() - yr.mean()
        both = obs & m
        dz = float(np.median(V.depth(k)[both] - zb[both])) if both.sum() > 30 else 0.0
        zc = float(np.median(zb[m]))
        R = V.R[k]
        T[k] = (du * zc / V.f[k]) * R[0] + (dv * zc / V.f[k]) * R[1] + dz * R[2]
    from scipy.ndimage import gaussian_filter1d
    Ts = gaussian_filter1d(T, 3, axis=0, mode='nearest')
    for k in range(0, V.n, 3):
        obs = Mb[k] & ~occl[k]
        if obs.sum() < 50:
            continue
        for Tk, acc in ((np.zeros(3), iou_s), (Ts[k], iou_m)):
            q, z = V.project(X + Tk, k)
            _, fid, _ = B.raster(q, z, F, V.H, V.W)
            m = (fid >= 0) & ~occl[k]
            acc.append((m & obs).sum() / max((m | obs).sum(), 1))
    gain = float(np.median(np.array(iou_m) - np.array(iou_s))) if iou_s else 0.0
    return X[None] + Ts[:, None], dict(iou_static=round(float(np.median(iou_s)), 4) if iou_s else None,
                                         iou_translated=round(float(np.median(iou_m)), 4) if iou_m else None,
                                         gain=round(gain, 4), max_shift_mm=round(float(np.linalg.norm(Ts, axis=1).max()) * 1e3, 2))


# ---------------------------------------------------------------- texture
def fuse_texture(V, ref, Zt, domc, S, p, sharp, ks):
    """v1 recipe on the first-hit reference depth Zt (surface or body front): per texel the median colour of the frames
    that see it as background, unoccluded (|z - video depth| < vis_tol) and not specular; per-frame gains; push-pull."""
    ys, xs = np.nonzero(domc)
    X = ref.unproject(xs, ys, Zt[ys, xs])
    N = len(xs)
    cols = np.full((len(ks), N, 3), 999, np.uint16)
    for i, k in enumerate(ks):
        q, z = V.project(X, k)
        xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        inb = (xi >= 0) & (xi < V.W) & (yi >= 0) & (yi < V.H) & (z > 0)
        ok = np.zeros(N, bool)
        st = S[k] & ~B.specular(V.frames[k])
        ok[inb] = st[yi[inb], xi[inb]] & (np.abs(z[inb] - V.depth(k)[yi[inb], xi[inb]]) < p['vis_tol'])
        c = B.remap_pts(V.frames[k], q[:, 0], q[:, 1])
        cols[i, ok] = c[ok]
    allk = np.ones(len(ks), bool)
    ref_col, _ = B.masked_median(cols, allk)
    for i in range(len(ks)):
        v = (cols[i, :, 0] < 999) & np.isfinite(ref_col[:, 0])
        if v.sum() > 500:
            g = np.clip(np.median(cols[i, v].astype(np.float32) + 1, 0) / np.median(ref_col[v] + 1, 0), 0.6, 1.6)
            cols[i, v] = np.clip(cols[i, v] / g, 0, 255).astype(np.uint16)
    good = sharp[ks]
    med_s, cnt_s = B.masked_median(cols, good)
    med_a, cnt_a = B.masked_median(cols, allk)
    w = np.clip(cnt_s / p['tex_min_sharp'], 0, 1)[:, None]
    res = np.where(cnt_s[:, None] > 0, w * np.nan_to_num(med_s) + (1 - w) * np.nan_to_num(med_a), med_a)
    done = cnt_a >= p['tex_min_obs']
    tex = np.zeros((ref.Hc, ref.Wc, 3), np.float32)
    seen = np.zeros((ref.Hc, ref.Wc), bool)
    tex[ys[done], xs[done]] = res[done]
    seen[ys[done], xs[done]] = True
    tex = B.push_pull(tex, seen)
    tex = B.add_detail(tex, seen, domc, dict(detail_gain=p['detail_gain'], detail_clip=p['detail_clip']))
    return np.clip(tex, 0, 255).astype(np.uint8), seen


def canvas_uv(ref, X):
    uc, vc, _ = ref.project(X)
    return np.stack([(uc + 0.5) / ref.Wc, 1 - (vc + 0.5) / ref.Hc], 1).astype(np.float32)


# ---------------------------------------------------------------- slab (closed version of the surface)
def slab(ref, X, F, depth_extra=0.010):
    """Closed solid: the surface, a copy pushed back along the reference rays to a common far depth, and side walls
    along the boundary edges. Watertight if the surface is an edge-manifold disk."""
    pc = (X - ref.pos) @ ref.R.T
    zfar = pc[:, 2].max() + depth_extra
    Xb = ref.pos + (pc * (zfar / pc[:, 2])[:, None]) @ ref.R
    n = len(X)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(E, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = E[cnt[inv.ravel()] == 1]                      # directed boundary edges (orientation of their face)
    walls = np.concatenate([np.stack([bnd[:, 1], bnd[:, 0], bnd[:, 0] + n], 1),
                            np.stack([bnd[:, 1], bnd[:, 0] + n, bnd[:, 1] + n], 1)])
    Fs = np.concatenate([F, F[:, ::-1] + n, walls])
    return np.vstack([X, Xb]), Fs


# ---------------------------------------------------------------- rendering helpers
def render_all(V, surf, bodies, k, Vt=None):
    """Composite of the surface and the bodies seen from frame k: mask, depth, rgb, owner (0 surface, i+1 body i)."""
    meshes = [(surf['X'], surf['F'], surf['uv'], surf['tex'])]
    for b in bodies:
        Xb = b['verts4d'][k] if b.get('verts4d') is not None else b['X']
        meshes.append((Xb, b['F'], b['uv'], surf['tex']))
    M_, Zr, RGB, FID = B.render_frame(V if Vt is None else Vt, meshes, k)
    owner = np.where(M_, FID >> 32, -1)
    return M_, Zr, RGB, owner


# ---------------------------------------------------------------- build
class ScaledViews:
    """The clip's views with every frame's metric depth multiplied by a per-frame scale (about the camera centre)."""

    def __init__(self, V, s):
        self._V, self.s = V, np.asarray(s, float)

    def __getattr__(self, a):
        if a == '_V' or a.startswith('__'):          # copy.copy / pickle probe attributes before __init__ ran
            raise AttributeError(a)
        return getattr(self._V, a)

    def depth(self, k):
        return self._V.depth(k) * self.s[k]


class CamViews(ScaledViews):
    """Views with other camera rotations (positions, focal and depth maps unchanged): for the registration experiment."""

    def __init__(self, V, R):
        super().__init__(V, np.ones(V.n))
        self.R = np.asarray(R, float)

    def project(self, X, k):
        return self.cam.project(np.asarray(X, float), self.R[k], self.f[k], self.pos[k])

    def unproject(self, u, v, z, k):
        return self.cam.unproject(u, v, z, self.R[k], self.f[k], self.pos[k])


def camera_experiment(R, out, p, rounds=2):
    """Optional (not the contract): register every frame's camera rotation to the textured background (v1's ECC step,
    keyframes included), re-fuse with the registered cameras, and evaluate with them. Shows how much of the
    background's misfit is camera error. Writes work/cams_registered.npz and work/quality_registered_cams.json."""
    V0 = R['V_raw']
    Rr = V0.R.copy()
    Rk = R
    for r in range(rounds):
        Se = Rk['S']
        Vc = CamViews(V0, Rr)
        Rr, info = B.refine_cameras(Vc, Rk['surf']['X'], Rk['surf']['F'], Rk['surf']['uv'],
                                    Rk['surf']['tex'], Se, iters=2, log=log, fix_keyframes=False)
        Rk = build(R['clip'], R['ver'], p, V=CamViews(V0, Rr))
    ang = np.degrees([np.arccos(np.clip((np.trace(Rr[k] @ V0.R[k].T) - 1) / 2, -1, 1)) for k in range(V0.n)])
    per, per_body, _ = evaluate(Rk)
    keys = ['depth_med_mm', 'depth_med_raw_mm', 'photo_l1', 'photo_ncc', 'photo_lncc', 'front_organ', 'organ_model_behind', 'coverage']
    q = dict(note='cameras registered to the background (rotation only, ECC, keyframes included), background re-fused and '
                  'evaluated with them; organ models unchanged (fitted with the clip cameras)',
             correction_deg=dict(median=round(float(np.median(ang)), 3), p90=round(float(np.percentile(ang, 90)), 3),
                                 max=round(float(ang.max()), 3)),
             summary={k: quality.summarize(per[k]) for k in keys},
             bodies={n: dict(iou=quality.summarize(v['iou']), depth_med_mm=quality.summarize(v['depth_med_mm'])) for n, v in per_body.items()},
             spread_mm=round(float(np.nanmedian(Rk['mad'][Rk['known']]) * 1e3), 3))
    np.savez(out / 'work' / 'cams_registered.npz', R=Rr, pos=V0.pos, f=V0.f, correction_deg=ang,
             note='rows = camera axes in world (as views2.Views.R); pos and f unchanged')
    (out / 'work' / 'quality_registered_cams.json').write_text(json.dumps(q, indent=1))
    sheet(Rk, np.linspace(0, V0.n - 1, 6).astype(int), out / 'work' / 'sheet_registered_cams.jpg')
    log('REGISTERED CAMERAS', {k: v.get('median') for k, v in q['summary'].items()}, q['correction_deg'], 'spread', q['spread_mm'])
    return q


def border_mask(V, b):
    bd = np.zeros((V.H, V.W), bool)
    bd[:b] = bd[-b:] = True
    bd[:, :b] = bd[:, -b:] = True
    return V.valid & ~bd


def moving_masks(V, S, p):
    """Pixels whose optical flow disagrees with the flow the camera alone would cause (untracked moving tissue:
    strands pulled by the organ, a tented sheet ...): residual > max(flow_px, flow_k x the frame's static median)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = np.zeros((V.n, V.H, V.W), bool)
    g = p['flow_gap']
    thr = np.zeros(V.n)
    for k in range(V.n):
        j = k + g if k + g < V.n else k - g
        e = flow_residual(V, k, j, dis)
        st = S[k]
        thr[k] = max(p['flow_px'], p['flow_k'] * np.median(e[st])) if st.sum() > 500 else np.inf
        out[k] = dil(e > thr[k], 4)
    return out, thr


def frame_scales(Dm, med, cnt, use):
    """Per-frame depth scale that brings the frame's static cells onto the multi-view median (median ratio)."""
    ok = cnt >= 10
    r = Dm[:, ok] / med[ok]
    ratio = np.nanmedian(np.where(np.isfinite(r), r, np.nan), 1)
    ratio = np.where(np.isfinite(ratio) & use, ratio, np.nan)
    s = 1 / ratio
    s = np.where(np.isfinite(s), s, np.nanmedian(s))
    return s / np.median(s)


def free_space(L, org, valid, Zf, hold, p):
    """Visibility check of the static surface against the organs: per cell, the number of frames in which a visible
    organ pixel (eroded 3 px) lies more than fs_tol BEHIND the surface (impossible if the surface were static
    background there: the organ is seen through it), and the number of frames with organ pixels in the cell."""
    m = L.Hg * L.Wg
    n_bad = np.zeros(m, np.int32)
    n_org = np.zeros(m, np.int32)
    zbad = np.full(m, -np.inf)
    zf = Zf.ravel()
    for k in range(org.shape[0]):
        if hold[k]:
            continue
        o = (ero(org[k], 3) & valid).ravel() & (L.cell[k] >= 0)
        c = L.cell[k][o]
        if not len(c):
            continue
        cnt = np.bincount(c, minlength=m)
        zo = np.bincount(c, L.zr[k][o], minlength=m) / np.maximum(cnt, 1)
        hit = cnt > 0
        n_org[hit] += 1
        bad = hit & np.isfinite(zf) & (zo > zf + p['fs_tol'])
        n_bad[bad] += 1
        if p.get('fs_mean'):
            zbad[bad] = np.where(np.isfinite(zbad[bad]), zbad[bad] + zo[bad], zo[bad])     # sum, divided below
        else:
            zbad[bad] = np.maximum(zbad[bad], zo[bad])
    if p.get('fs_mean'):
        zbad = np.where(n_bad > 0, zbad / np.maximum(n_bad, 1), np.nan)
    zbad[~np.isfinite(zbad)] = np.nan
    return n_bad.reshape(L.Hg, L.Wg), n_org.reshape(L.Hg, L.Wg), zbad.reshape(L.Hg, L.Wg)


def build(clip, ver, p, V=None, log=log):
    t0 = time.time()
    V = V or views2.load(clip)
    seg, M = load_masks(clip, V, p.get('seg'))
    log(clip, 'seg', seg, 'objects', list(M))
    ins_names = [n for n in M if n.startswith('instrument_')]
    ins = np.any([M[n] for n in ins_names], 0) if ins_names else np.zeros((V.n, V.H, V.W), bool)
    motion = motion_stats(V, M, ins)
    cls = classify(clip, V, M, motion, p)
    for n, c in cls.items():
        log(f'  {n:40s} {c["kind"]:10s} {c["why"]}')
    models = latest_organ_models(clip, p.get('organ_pin'))
    models = {n: m for n, m in models.items() if n in cls and cls[n]['kind'] == 'organ'}
    log('organ models', {n: f"{m['organ']}/{m['version']}" for n, m in models.items()})
    organs = [n for n, c in cls.items() if c['kind'] == 'organ']
    cands = [n for n, c in cls.items() if c['kind'] == 'candidate']
    org = np.any([M[n] for n in organs], 0) if organs else np.zeros_like(ins)
    sh = B.sharpness(V)
    sharp = sh >= p['sharp_rel'] * np.median(sh)
    hold = np.zeros(V.n, bool)
    if p.get('holdout_mod'):
        a, r = p['holdout_mod']
        hold = np.arange(V.n) % a == r
    # static pixels of the surface: not instruments / organs / body candidates, inside the scope, off the border;
    # S_in: the same away from the image border (depth maps are least reliable there), preferred where enough
    bmask = np.zeros_like(ins)
    for n in cands:
        bmask |= M[n]
    valid = border_mask(V, p['border'])
    valid_in = border_mask(V, p.get('border_in') or p['border'])
    S = np.zeros_like(ins)
    for k in range(V.n):
        S[k] = valid & ~dil(ins[k], p['tool_r']) & ~dil(org[k], p['organ_r']) & ~dil(bmask[k], p['body_r'])
    S[hold] = False
    mov_thr = None
    if p.get('flow_reject'):
        mov, mov_thr = moving_masks(V, S, p)
        log('moving pixels (flow) removed from the static set:', round(float((S & mov).sum() / max(S.sum(), 1)), 3),
            'threshold px median', round(float(np.median(mov_thr[np.isfinite(mov_thr)])), 2), f'{time.time() - t0:.0f}s')
        S &= ~mov
    S_in = S & valid_in[None]
    log('static fraction', round(float(S[~hold].mean()), 3), f'{time.time() - t0:.0f}s')
    ref = B.Ref(V, cell=p['cell'])
    scale = np.ones(V.n)
    use = sharp & ~hold
    for it in range(p.get('scale_iters', 0) + 1):
        SV = ScaledViews(V, scale)
        L = Lift(SV, ref, p['cell'])
        Hg, Wg = L.Hg, L.Wg
        Dm_in, Dm_all = L.fuse(S_in), L.fuse(S)
        cnt_in = np.isfinite(Dm_in[use]).sum(0)
        Dm = np.where((cnt_in >= p['min_obs'])[None], Dm_in, Dm_all)
        med, cnt, mad, Dk = robust_median(Dm, p, use)
        if it < p.get('scale_iters', 0):
            ds = frame_scales(Dm, med, cnt, ~hold)
            scale = scale * ds
            log(f'per-frame depth scale round {it}: change p5-p95 {np.percentile(ds, 5):.3f}-{np.percentile(ds, 95):.3f}; '
                f'MAD median {np.nanmedian(mad[cnt >= p["min_obs"]]) * 1e3:.2f} mm')
    log('canvas', ref.Wc, ref.Hc, 'cells', Hg, Wg, 'MAD median', round(float(np.nanmedian(mad[cnt >= p['min_obs']]) * 1e3), 2),
        'mm', f'{time.time() - t0:.0f}s')
    cnt_sharp = np.isfinite(Dk[sharp]).sum(0).reshape(Hg, Wg)
    Z, cnt, mad = med.reshape(Hg, Wg).astype(float), cnt.reshape(Hg, Wg), mad.reshape(Hg, Wg)
    foot = L.foot.reshape(Hg, Wg)
    known0 = cnt >= p['min_obs']
    lower, lower_src = organ_lower_bound(SV, ref, L, M, cls, models, p)
    log('organ lower bound', lower_src, f'{time.time() - t0:.0f}s')
    # candidate footprints (cells seen as the candidate in >= body_min_obs frames)
    cand_cells = {}
    for n in cands:
        Sb = np.stack([ero(M[n][k], 2) & valid & ~dil(ins[k], p['tool_r']) & ~dil(org[k], 4) for k in range(V.n)])
        Sb[hold] = False
        cand_cells[n] = Sb
    dom_cells = known0 | np.any([L.count(Sb).reshape(Hg, Wg) >= p['body_min_obs'] for Sb in cand_cells.values()], 0) \
        if cands else known0
    dom = domain(dom_cells, foot, p)
    known = known0 & dom
    Zf, w, enforce = surface_solve(Z, cnt, mad, cnt_sharp, dom, known, lower, p)
    # body candidates: carved, then the protrusion test against the surface fitted without them, the mask test and
    # the motion test
    occl_all = np.stack([dil(ins[k] | org[k], 3) | ~V.valid for k in range(V.n)])
    bodies, body_info, flat = [], {}, []
    for n in cands:
        bdy, info = carve_body(SV, n, M[n], M, cls, ins, ref, np.where(dom, Zf, np.nan), hold, p)
        cls[n]['body_test'] = info
        log('candidate', n, info)
        if bdy is None or info.get('protrude_med_mm') is None or info['protrude_med_mm'] < p['protrude_mm']:
            flat.append(n)
            cls[n]['kind'] = 'flat'
            cls[n]['why'] += f" -> flat (protrudes {info.get('protrude_med_mm')} mm < {p['protrude_mm']} mm): surface texture"
            continue
        v4, mi = body_motion(SV, bdy, M[n], occl_all, p)
        if mi['iou_static'] is None or mi['iou_static'] < p['body_min_iou']:
            flat.append(n)
            cls[n]['kind'] = 'flat'
            cls[n]['why'] += (f" -> not a body: the carved shape does not reproduce its mask (IoU {mi['iou_static']} < "
                              f"{p['body_min_iou']}; multi-view inconsistent): left to the surface")
            continue
        flow_ratio = cls[n]['motion'].get('ratio', 0)
        bdy['static'] = not (mi['gain'] >= p['move_gain'] and flow_ratio >= p['motion_ratio'])
        bdy['T'] = None if bdy['static'] else (v4[:, 0] - bdy['X'][0])
        bdy['motion'] = mi
        cls[n]['motion_fit'] = mi
        log('body', n, 'static' if bdy['static'] else 'MOVING', mi)
        cls[n]['kind'] = 'body'
        cls[n]['why'] += f" -> body (protrudes {info['protrude_med_mm']} mm, {info.get('pieces')} convex piece(s), IoU {mi['iou_static']})"
        bodies.append(bdy)
        body_info[n] = info
    if flat:                                          # flat candidates are surface after all: re-fuse with them
        for k in range(V.n):
            if hold[k]:
                continue
            fm = np.zeros((V.H, V.W), bool)
            for n in flat:
                fm |= M[n][k]
            add = fm & valid & ~dil(ins[k], p['tool_r']) & ~dil(org[k], p['organ_r'])
            for bd_ in bodies:
                add &= ~dil(M[bd_['name']][k], p['body_r'])
            S[k] |= add
            S_in[k] |= add & valid_in
        Dm_in, Dm_all = L.fuse(S_in), L.fuse(S)
        cnt_in = np.isfinite(Dm_in[use]).sum(0)
        Dm = np.where((cnt_in >= p['min_obs'])[None], Dm_in, Dm_all)
        med, cnt, mad, Dk = robust_median(Dm, p, use)
        cnt_sharp = np.isfinite(Dk[sharp]).sum(0).reshape(Hg, Wg)
        Z, cnt, mad = med.reshape(Hg, Wg).astype(float), cnt.reshape(Hg, Wg), mad.reshape(Hg, Wg)
        known = (cnt >= p['min_obs']) & dom
        Zf, w, enforce = surface_solve(Z, cnt, mad, cnt_sharp, dom, known, lower, p)
    # visibility: cells where visible organs are repeatedly seen BEHIND the surface are not static background (moved,
    # or mislabelled organ parts): their static data are dropped and the organ lower bound applies; refit, repeat
    contra = np.zeros_like(known)
    for it in range(p.get('fs_iters', 0)):
        n_bad, n_org, zbad = free_space(L, org, valid, Zf, hold, p)
        new = known & (n_bad >= 3) & (n_bad >= p['fs_frac'] * np.maximum(n_org, 1)) & ~contra
        n_ins = 0
        if p.get('fs_instruments'):
            # instruments (eroded 3 px; their video depth is their front): surface behind front + shaft + 1 mm
            nb_i, no_i, zb_i = free_space(L, ins, valid, Zf, hold, dict(p, fs_tol=p['fs_tol'] + 0.002))
            new_i = dom & (nb_i >= 3) & (nb_i >= p['fs_frac'] * np.maximum(no_i, 1)) & ~contra
            lower = np.where(new_i, np.fmax(lower, zb_i + p['shaft_d'] + 0.001), lower)
            new = new | (new_i & known)
            n_ins = int(new_i.sum())
        if not new.any() and not n_ins:
            break
        contra |= new
        known = known & ~contra
        if p.get('fs_min_push'):
            # minimal push: behind the organ FRONT seen there (+ margin), not behind the organ model's guessed back
            # (with inconsistent cameras the static data may be right and the organ placement wrong)
            lower = np.where(new, zbad + p['organ_margin'], lower)
        else:
            lower = np.where(new, np.fmax(lower, zbad + p['organ_margin']), lower)    # behind the organ seen there
        if p.get('fs_smooth'):
            lower = smooth_lower(lower)
        Zf, w, enforce = surface_solve(Z, np.where(contra, 0, cnt), mad, cnt_sharp, dom, known, lower, p)
        log(f'free-space round {it}: {int(new.sum())} cells contradicted by visible organs / instruments '
            f'({int(contra.sum())} total; instrument cells {n_ins})')
    conflict = known & np.isfinite(lower) & (Zf < lower - p['organ_margin']) & ~enforce    # organ back behind the surface
    log('surface: known', int(known.sum()), 'hidden', int((dom & ~known).sum()), 'lower-bound conflicts in observed cells',
        int(conflict.sum()), f'{time.time() - t0:.0f}s')
    # per-cell label (which object the static observations came from)
    lab_names = ['unlabelled'] + [n for n in M if cls[n]['kind'] in ('surface', 'flat')]
    votes = np.zeros((len(lab_names), Hg * Wg), np.float32)
    for k in range(0, V.n, 2):
        if hold[k]:
            continue
        s = S[k].ravel() & (L.cell[k] >= 0)
        labk = np.zeros(V.H * V.W, np.int32)
        for i, n in enumerate(lab_names[1:], 1):
            labk[M[n][k].ravel()] = i
        votes += np.stack([np.bincount(L.cell[k][s & (labk == i)], minlength=Hg * Wg) for i in range(len(lab_names))])
    lab_cell = votes.argmax(0).reshape(Hg, Wg)
    from scipy import ndimage
    has = known & (votes.sum(0).reshape(Hg, Wg) > 0)
    _, (iy, ix) = ndimage.distance_transform_edt(~has, return_indices=True)
    lab_cell = lab_cell[iy, ix]
    # canvas resolution
    vc_, uc_ = np.mgrid[0:ref.Hc, 0:ref.Wc]
    cs = p['cell']
    Zc = B.sample_grid(np.where(dom, Zf, np.nanmedian(Zf[dom])), cs, uc_, vc_)
    domc = B.sample_grid(dom.astype(np.float32), cs, uc_, vc_, cv2.INTER_NEAREST) > 0.5
    X, F, uv, uc, vc = B.grid_mesh(ref, Zc, domc, p['grid'])
    F = B.orient_faces(X, F, ref)
    ci, cj = np.clip(uc // cs, 0, Wg - 1), np.clip(vc // cs, 0, Hg - 1)
    # bodies: dense closed meshes, first-hit depth in the canvas for the texture
    Zt = Zc.copy()
    for bdy in bodies:
        Xb, Fb, Pb = subdivide_body(bdy['X'], bdy['F'], bdy['piece'])
        bdy.update(X=Xb, F=Fb, piece=Pb, uv=canvas_uv(ref, Xb))
        nz = far_depth(ref, Xb, Fb, ref.Hc, ref.Wc)          # (only for its coverage)
        q, z = ref.cam.project(Xb, ref.R, ref.f, ref.pos)
        zb, fid, _ = B.raster(q - np.array([ref.ou, ref.ov]), z, Fb, ref.Hc, ref.Wc, max_box=1024)
        hit = (fid >= 0) & np.isfinite(nz)
        Zt[hit] = np.minimum(Zt[hit], zb[hit])
    for bdy in bodies:
        bdy['verts4d'] = None if bdy['static'] else bdy['X'][None] + bdy['T'][:, None]
    # texture over the first-hit canvas, from surface + body pixels
    S_tex = S.copy()
    for bdy in bodies:
        for k in range(V.n):
            if not hold[k]:
                S_tex[k] |= ero(M[bdy['name']][k], 2) & valid & ~dil(ins[k], p['tool_r']) & ~dil(org[k], 4)
    ks = np.array([k for k in range(0, V.n, p['tex_every']) if not hold[k]])
    domt = domc | (Zt < Zc - 1e-6)
    tex, seen = fuse_texture(SV, ref, Zt, domt, S_tex, p, sharp, ks)
    log('texture seen', round(float(seen[domc].mean()), 3), f'{time.time() - t0:.0f}s')
    surf = dict(X=X, F=F, uv=uv.astype(np.float32), tex=tex, observed=known[cj, ci], confidence=w[cj, ci],
                spread_mm=(np.nan_to_num(mad, nan=0) * 1e3)[cj, ci], n_obs=cnt[cj, ci],
                label=lab_cell[cj, ci], label_names=lab_names, enforce=enforce[cj, ci],
                under_organ=(np.isfinite(lower) & ~known)[cj, ci], lower=lower)
    # per-organ conflicts: observed static cells behind which an organ's back would lie (organ thicker than the room)
    conflict_by = {}
    if conflict.any():
        for n in organs:
            lo_n, _ = organ_lower_bound(SV, ref, L, M, {m: dict(c, kind='organ' if m == n else 'x') for m, c in cls.items()},
                                        models, p)
            conflict_by[n] = int((known & np.isfinite(lo_n) & (Zf < lo_n - p['organ_margin']) & ~enforce).sum())
    return dict(V=SV, V_raw=V, scale=scale, mov_thr=mov_thr, conflict_by=conflict_by, contra=contra,
                clip=clip, ver=ver, p=p, seg=seg, M=M, cls=cls, motion=motion, models=models, lower_src=lower_src,
                ref=ref, S=S, hold=hold, sharp=sharp, ins=ins, org=org, organs=organs, Z=Z, Zf=Zf, Zc=Zc, dom=dom,
                domc=domc, known=known, cnt=cnt, mad=mad, w=w, lower=lower, conflict=conflict, seen=seen,
                surf=surf, bodies=bodies, body_info=body_info, t_build=time.time() - t0)


# ---------------------------------------------------------------- evaluation
def thirds(n):
    a = np.array_split(np.arange(n), 3)
    return {f'{s[0]}-{s[-1]}': s for s in a}


def evaluate(R, ks=None, cams=None):
    """Per-frame metrics (see quality.json definitions)."""
    V, M, surf, bodies = R['V'], R['M'], R['surf'], R['bodies']
    ks = range(V.n) if ks is None else ks
    ins, org = R['ins'], R['org']
    bd = np.zeros((V.H, V.W), bool)
    b = 8
    bd[:b] = bd[-b:] = True
    bd[:, :b] = bd[:, -b:] = True
    keys = ['coverage', 'depth_med_mm', 'depth_p90_mm', 'depth_bias_mm', 'front_static', 'front_organ', 'front_instrument',
            'photo_l1', 'photo_ncc', 'photo_lncc', 'organ_model_behind', 'organ_model_behind_mm',
            'depth_med_raw_mm', 'depth_p90_raw_mm', 'front_static_raw', 'front_organ_raw']
    Vraw = R.get('V_raw', V)
    per = {k: np.full(V.n, np.nan) for k in keys}
    per_body = {b_['name']: dict(iou=np.full(V.n, np.nan), depth_med_mm=np.full(V.n, np.nan),
                                 front_organ=np.full(V.n, np.nan)) for b_ in bodies}
    body_names = [b_['name'] for b_ in bodies]
    per_organ = {n: np.full(V.n, np.nan) for n in R['organs']}
    for k in ks:
        Mk, Zr, RGB, owner = render_all(V, surf, bodies, k)
        Dk = V.depth(k)
        bod = np.zeros((V.H, V.W), bool)
        for n in body_names:
            bod |= M[n][k]
        E = V.valid & ~bd & ~dil(ins[k], 8) & ~dil(org[k], 8)
        both = Mk & E
        per['coverage'][k] = both.sum() / max(E.sum(), 1)
        if both.sum() > 50:
            dz = (Zr - Dk)[both]
            per['depth_med_mm'][k] = np.median(np.abs(dz)) * 1e3
            per['depth_p90_mm'][k] = np.percentile(np.abs(dz), 90) * 1e3
            per['depth_bias_mm'][k] = np.median(dz) * 1e3
            per['front_static'][k] = float((dz < -0.005).mean())
            dzr = (Zr - Vraw.depth(k))[both]
            per['depth_med_raw_mm'][k] = np.median(np.abs(dzr)) * 1e3
            per['depth_p90_raw_mm'][k] = np.percentile(np.abs(dzr), 90) * 1e3
            per['front_static_raw'][k] = float((dzr < -0.005).mean())
            ph = both & ~B.specular(V.frames[k])
            fr = V.frames[k].astype(np.float32)
            per['photo_l1'][k] = float(np.abs(RGB - fr)[ph].mean())
            ga = cv2.cvtColor(RGB.astype(np.float32), cv2.COLOR_RGB2GRAY)
            gb = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            per['photo_ncc'][k] = B.ncc(ga[ph], gb[ph])
            per['photo_lncc'][k] = B.local_ncc(ga, gb, ph)
        O = ero(org[k], 3) & Mk & V.valid
        if O.sum() > 50:
            per['front_organ'][k] = float((Zr[O] < Dk[O] - 0.002).mean())
            per['front_organ_raw'][k] = float((Zr[O] < Vraw.depth(k)[O] - 0.002).mean())
        for n in R['organs']:
            On = ero(M[n][k], 3) & Mk & V.valid
            if On.sum() > 50:
                per_organ[n][k] = float((Zr[On] < Dk[On] - 0.002).mean())
        I = ero(ins[k], 2) & Mk & V.valid
        if I.sum() > 50:
            per['front_instrument'][k] = float((Zr[I] < Dk[I] - 0.002).mean())
        # organ 4D models: vertices behind the background along this frame's rays
        behind, depth_b = [], []
        for n, mdl in R['models'].items():
            Xo = mdl['verts4d'][k]
            q, z = V.project(Xo, k)
            xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
            inb = (xi >= 0) & (xi < V.W) & (yi >= 0) & (yi < V.H) & (z > 0)
            zr_ = np.full(len(Xo), np.inf)
            zr_[inb] = Zr[yi[inb], xi[inb]]
            cov = np.isfinite(zr_)
            bh = cov & (z > zr_ + 0.001)
            behind.append(bh.sum() / max(cov.sum(), 1))
            depth_b.append(float(np.max(z[bh] - zr_[bh])) * 1e3 if bh.any() else 0.0)
        if behind:
            per['organ_model_behind'][k] = float(np.max(behind))
            per['organ_model_behind_mm'][k] = float(np.max(depth_b))
        # bodies
        knowable = ~dil(ins[k] | org[k], 3) & V.valid
        for i, n in enumerate(body_names):
            vis = (owner == i + 1) & knowable
            obs = M[n][k] & knowable
            if obs.sum() + vis.sum() > 50:
                per_body[n]['iou'][k] = (vis & obs).sum() / max((vis | obs).sum(), 1)
            bb = vis & obs
            if bb.sum() > 30:
                per_body[n]['depth_med_mm'][k] = np.median(np.abs(Zr[bb] - Dk[bb])) * 1e3
            Ob = (owner == i + 1) & ero(org[k], 3)
            if Ob.sum() > 30:
                per_body[n]['front_organ'][k] = float((Zr[Ob] < Dk[Ob] - 0.002).mean())
    return per, per_body, per_organ


def summarize_thirds(v, n, ks=None):
    v = np.asarray(v, float)
    out = dict(all=quality.summarize(v))
    for name, s in thirds(n).items():
        out[name] = quality.summarize(v[s])
    for key in ('all',) + tuple(thirds(n)):
        x = v[thirds(n)[key]] if key != 'all' else v
        x = x[np.isfinite(x)]
        if len(x):
            out[key]['p90'] = round(float(np.percentile(x, 90)), 4)
    return out


def mesh_checks(X, F, ref, lowconf=None):
    import trimesh
    m = trimesh.Trimesh(X, F, process=False)
    comps = m.split(only_watertight=False)
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    c = X[F].mean(1)
    ray = c - ref.pos
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    cosv = np.abs((n / np.maximum(a[:, None], 1e-15) * ray).sum(1))
    steep = cosv < 0.17                           # > 80 deg between the face normal and the reference ray
    # spikes: vertices far from the mean of their neighbours
    E = m.edges_unique
    nb = np.zeros_like(X)
    deg = np.zeros(len(X))
    np.add.at(nb, E[:, 0], X[E[:, 1]])
    np.add.at(nb, E[:, 1], X[E[:, 0]])
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    lap = np.linalg.norm(X - nb / np.maximum(deg, 1)[:, None], axis=1)
    lc = {}
    if lowconf is not None:
        flc = lowconf[F].any(1)
        lc = dict(steep_lowconf_fraction=round(float(a[steep & flc].sum() / a.sum()), 4))
    return dict(components=len(comps), component_faces=sorted([len(cc.faces) for cc in comps], reverse=True)[:5],
                steep_face_fraction=round(float(a[steep].sum() / a.sum()), 4), **lc,
                spike_vertices_3mm=int((lap > 0.003).sum()), laplacian_p99_mm=round(float(np.percentile(lap, 99)) * 1e3, 3))


# ---------------------------------------------------------------- outputs
def save(R, out):
    import imageio.v2 as imageio
    V, ref, surf = R['V'], R['ref'], R['surf']
    X, F = surf['X'].astype(np.float32), surf['F'].astype(np.int32)
    ray = X - ref.pos
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    hf = hfield_zbuf(ref, surf['X'], surf['F'])
    tiles = hfield_tiles(ref, surf['X'], surf['F'])
    hk = {'hfields_n': np.int32(len(tiles))}
    for i, h in enumerate(tiles):
        for k_, v_ in h.items():
            hk[f'hfield{i}_{k_}'] = v_
    Xs, Fs = slab(ref, surf['X'], surf['F'])
    bk = {}
    for i, b in enumerate(R['bodies']):
        bk[f'body{i}_verts'] = b['X'].astype(np.float32)
        bk[f'body{i}_faces'] = b['F'].astype(np.int32)
        bk[f'body{i}_piece'] = b['piece'].astype(np.int16)
        bk[f'body{i}_uv'] = b['uv']
        bk[f'body{i}_rgb'] = vertex_rgb(surf['tex'], b['uv'])
        if b.get('verts4d') is not None:
            bk[f'body{i}_verts4d'] = b['verts4d'].astype(np.float32)
    np.savez(out / 'model.npz', rest_verts=X, faces=F, uv=surf['uv'], vertex_rgb=vertex_rgb(surf['tex'], surf['uv']),
             observed=surf['observed'], filled=~surf['observed'], under_organ=surf['under_organ'],
             confidence=surf['confidence'].astype(np.float32), spread_mm=surf['spread_mm'].astype(np.float32),
             n_obs=surf['n_obs'].astype(np.int16), label=surf['label'].astype(np.int8),
             label_names=np.array(surf['label_names']), ray_dir=ray.astype(np.float32),
             attach_idx=np.arange(len(X)), attach_to=np.array('world'),
             slab_verts=Xs.astype(np.float32), slab_faces=Fs.astype(np.int32),
             body_names=np.array([b['name'] for b in R['bodies']]),
             body_static=np.array([b['static'] for b in R['bodies']], bool),
             **bk, **{f'ref_{k}': np.asarray(v) for k, v in ref.as_dict().items()},
             **{f'hfield_{k}': v for k, v in hf.items()}, **hk,
             ref_surface_depth=np.where(R['dom'], R['Zf'], np.nan).astype(np.float32), ref_cell=np.int32(R['p']['cell']),
             depth_scale=np.asarray(R.get('scale', np.ones(V.n)), np.float32))
    imageio.imwrite(out / 'texture.png', surf['tex'])
    imageio.imwrite(out / 'work' / 'texture_seen.png', (R['seen'] * 255).astype(np.uint8))
    B.write_obj(out / 'background.obj', surf['X'], surf['F'], surf['uv'], 'texture.png')
    for i, b in enumerate(R['bodies']):
        B.write_obj(out / f'body{i}.obj', b['X'], b['F'], b['uv'], 'texture.png')
    return dict(slab=quality.mesh_health(Xs, Fs))


def hfield_zbuf(ref, X, F, res=0.0006):
    """MuJoCo height field of the surface (frame and conventions as v1's backdrop.hfield: x = ref x, y = -ref y,
    z = -ref z, elevation = z_far - depth along the reference axis), but sampled by an orthographic z-buffer of the
    mesh: where the surface folds over along the axis the FRONT-most sheet is kept (everything behind the surface is
    solid anyway), instead of interpolating scattered vertices (which mixes front and back on steep walls)."""
    from scipy.spatial import cKDTree
    import mujoco
    pc = (X - ref.pos) @ ref.R.T
    hx, hy, hz = pc[:, 0], -pc[:, 1], pc[:, 2]
    x0, x1, y0, y1 = hx.min(), hx.max(), hy.min(), hy.max()
    ncol, nrow = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
    rx, ry = (x1 - x0) / (ncol - 1), (y1 - y0) / (nrow - 1)
    q = np.stack([(hx - x0) / rx, (hy - y0) / ry], 1)
    zb, fid, _ = B.raster(q, hz, F, nrow, ncol, max_box=4096)
    valid = fid >= 0
    z = np.where(valid, zb, np.nan)
    gy, gx = np.mgrid[0:nrow, 0:ncol]
    if (~valid).any():
        _, i = cKDTree(np.stack([gx[valid], gy[valid]], 1)).query(np.stack([gx[~valid], gy[~valid]], 1))
        z[~valid] = z[valid][i]
    zfar = z.max() + 0.002
    elev = zfar - z
    er = float(elev.max())
    Rf = np.stack([ref.R[0], -ref.R[1], -ref.R[2]])
    centre = ref.pos + ((x0 + x1) / 2) * ref.R[0] - ((y0 + y1) / 2) * ref.R[1] + zfar * ref.R[2]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rf.T.flatten())
    return dict(elev=(elev / er).astype(np.float32), valid=valid, size=np.array([(x1 - x0) / 2, (y1 - y0) / 2, er, 0.005]),
                pos=centre, quat=quat, R=Rf)


def hfield_frame(X, F, origin, d, xref, res=0.0006):
    """Orthographic z-buffer height field of a mesh along the unit direction d (looking from `origin`): frame axes
    z = -d (towards the origin), x = xref made orthogonal, y = z x x; elevation = z_far - depth along d."""
    from scipy.spatial import cKDTree
    import mujoco
    zt = -d / np.linalg.norm(d)
    xt = xref - (xref @ zt) * zt
    xt /= np.linalg.norm(xt)
    yt = np.cross(zt, xt)
    P = X - origin
    hx, hy, hz = P @ xt, P @ yt, P @ (-zt)
    x0, x1, y0, y1 = hx.min(), hx.max(), hy.min(), hy.max()
    ncol, nrow = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
    rx, ry = (x1 - x0) / (ncol - 1), (y1 - y0) / (nrow - 1)
    zb, fid, _ = B.raster(np.stack([(hx - x0) / rx, (hy - y0) / ry], 1), hz, F, nrow, ncol, max_box=4096)
    valid = fid >= 0
    zfar = float(zb[valid].max()) + 0.002
    # cells the surface does not cover: elevation 0 (at the far plane), not the nearest value -- a nearest fill would
    # put phantom solid in front of the neighbouring tiles / outside the surface
    elev = np.where(valid, zfar - zb, 0.0)
    er = max(float(elev.max()), 1e-4)
    Rf = np.stack([xt, yt, zt])
    centre = origin + ((x0 + x1) / 2) * xt + ((y0 + y1) / 2) * yt - zfar * zt
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rf.T.flatten())
    return dict(elev=(elev / er).astype(np.float32), valid=valid, size=np.array([(x1 - x0) / 2, (y1 - y0) / 2, er, 0.005]),
                pos=centre, quat=quat, R=Rf)


def hfield_tiles(ref, X, F, nt=(6, 4), overlap=0.04, res=0.0006):
    """The surface as nt[0] x nt[1] height fields, each looking along the reference camera ray through its tile centre.
    One height field along the camera axis turns walls that run along the camera rays at the edge of the view into
    overhangs (filled solid up to ~8 mm laterally); per-tile axes keep every ray within a few degrees of its tile's
    axis. Tiles overlap by `overlap` of the image span (the solid is the union)."""
    q, _ = ref.cam.project(X[F].mean(1), ref.R, ref.f, ref.pos)
    u, v = q[:, 0], q[:, 1]
    ue = np.linspace(u.min(), u.max(), nt[0] + 1)
    ve = np.linspace(v.min(), v.max(), nt[1] + 1)
    du, dv = overlap * (u.max() - u.min()), overlap * (v.max() - v.min())
    out = []
    for i in range(nt[0]):
        for j in range(nt[1]):
            sel = (u >= ue[i] - du) & (u <= ue[i + 1] + du) & (v >= ve[j] - dv) & (v <= ve[j + 1] + dv)
            if sel.sum() < 20:
                continue
            Fi = F[sel]
            used = np.unique(Fi)
            remap = -np.ones(len(X), int)
            remap[used] = np.arange(len(used))
            uc, vc = (ue[i] + ue[i + 1]) / 2, (ve[j] + ve[j + 1]) / 2
            d = ref.cam.unproject(uc, vc, 1.0, ref.R, ref.f, ref.pos) - ref.pos
            out.append(hfield_frame(X[used], remap[Fi], ref.pos, d / np.linalg.norm(d), ref.R[0], res))
    return out


def hfield_mjcf(fields, name='bg'):
    """MJCF snippet (asset, worldbody) for a list of height fields; the data must be set after loading:
    model.hfield_data[model.hfield_adr[i]: ...] = elev.ravel()."""
    asset, body = [], []
    for i, h in enumerate(fields):
        nrow, ncol = h['elev'].shape
        sz = h['size']
        asset.append(f'<hfield name="{name}{i}" nrow="{nrow}" ncol="{ncol}" size="{sz[0]} {sz[1]} {sz[2]} {sz[3]}"/>')
        body.append(f'<geom name="{name}{i}" type="hfield" hfield="{name}{i}" pos="{" ".join(map(str, h["pos"]))}" '
                    f'quat="{" ".join(map(str, h["quat"]))}"/>')
    return '\n'.join(asset), '\n'.join(body)


def load_hfields(z, prefix='hfield'):
    if f'{prefix}s_n' in z.files:
        return [dict(elev=z[f'{prefix}{i}_elev'], size=z[f'{prefix}{i}_size'], pos=z[f'{prefix}{i}_pos'], quat=z[f'{prefix}{i}_quat'])
                for i in range(int(z[f'{prefix}s_n']))]
    return [dict(elev=z[f'{prefix}_elev'], size=z[f'{prefix}_size'], pos=z[f'{prefix}_pos'], quat=z[f'{prefix}_quat'])]


def hfield_check(out, n=4000, seed=0, tiles=True):
    """Load model.npz's height field(s) into MuJoCo and cast rays from the reference camera through surface vertices:
    hit distance vs the mesh (does the collision surface reproduce the mesh?)."""
    import mujoco
    z = np.load(out / 'model.npz')
    fields = load_hfields(z) if tiles else [dict(elev=z['hfield_elev'], size=z['hfield_size'], pos=z['hfield_pos'], quat=z['hfield_quat'])]
    a, b = hfield_mjcf(fields)
    m = mujoco.MjModel.from_xml_string(f'<mujoco><asset>{a}</asset><worldbody>{b}</worldbody></mujoco>')
    for i, h in enumerate(fields):
        nr, nc = h['elev'].shape
        m.hfield_data[m.hfield_adr[i]: m.hfield_adr[i] + nr * nc] = h['elev'].ravel()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    from scipy.spatial import cKDTree
    X = z['rest_verts'].astype(float)
    F = z['faces']
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, cnt = np.unique(E, axis=0, return_counts=True)
    bv = np.zeros(len(X), bool)
    bv[u[cnt == 1].ravel()] = True
    interior = cKDTree(X[bv]).query(X)[0] > 0.002           # > 2 mm from the surface's outer boundary
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    pos = z['ref_pos'].astype(float)
    err = np.full(len(idx), np.inf)
    gid = np.zeros(1, np.int32)
    for j, i in enumerate(idx):
        v = X[i] - pos
        L = np.linalg.norm(v)
        t = mujoco.mj_ray(m, d, pos, v / L, None, 1, -1, gid)
        if t >= 0:
            err[j] = (t - L) * 1e3
    out = dict(height_fields=len(fields), rays=len(idx))
    for name, sel in (('all', np.ones(len(idx), bool)), ('interior', interior[idx])):
        e = err[sel]
        hit = np.isfinite(e)
        out[name] = dict(rays=int(sel.sum()), missed=round(float((~hit).mean()), 4),
                         err_med_mm=round(float(np.median(np.abs(e[hit]))), 3),
                         early_hits_3mm=round(float((e[hit] < -3).sum() / len(e)), 4),
                         late_hits_3mm=round(float((e[hit] > 3).sum() / len(e)), 4),
                         err_p99_mm=round(float(np.percentile(np.abs(e[hit]), 99)), 3))
    return out


def vertex_rgb(tex, uv):
    th, tw = tex.shape[:2]
    return B.remap_pts(tex, uv[:, 0] * tw - 0.5, (1 - uv[:, 1]) * th - 0.5).astype(np.uint8)


def occupancy(R, out, scale=2):
    """Camera-ray occupancy per frame at 1/scale resolution: depth of the first background hit (surface or body),
    its owner (0 surface, i+1 body i, -1 none), and the video depth."""
    V = R['V']
    h, w = V.H // scale, V.W // scale
    zb = np.full((V.n, h, w), np.inf, np.float16)
    own = np.full((V.n, h, w), -1, np.int8)
    vz = np.zeros((V.n, h, w), np.float16)
    for k in range(V.n):
        Mk, Zr, _, owner = render_all(V, R['surf'], R['bodies'], k)
        zb[k] = np.where(Mk, Zr, np.inf)[scale // 2::scale, scale // 2::scale][:h, :w]
        own[k] = owner[scale // 2::scale, scale // 2::scale][:h, :w]
        vz[k] = R.get('V_raw', V).depth(k)[scale // 2::scale, scale // 2::scale][:h, :w]
    np.savez_compressed(out / 'occupancy.npz', bg_depth=zb, owner=own, video_depth=vz, scale=scale,
                        depth_scale=np.asarray(R.get('scale', np.ones(V.n)), np.float32),
                        body_names=np.array([b['name'] for b in R['bodies']]),
                        note='per frame k, pixel (y, x) of the half-resolution grid = full-res pixel (scale*y + scale//2, '
                             'scale*x + scale//2): video_depth = the raw metric video depth (the background was fused with '
                             'depth_scale[k] x video_depth); bg_depth = camera depth (m, along the optical axis) of the first '
                             'background hit (inf: none), owner 0 surface / i+1 body i / -1 none. Anything at a camera '
                             'depth < bg_depth along that ray is in free space; organs must stay in front of it.')


class Background:
    """The contract as an object: load a background version and query it.

        bg = Background('chole_a')                      # newest version
        d = bg.ray_distance(X, k)                       # (N,) m, > 0: in front of the background along frame k's rays
        d = bg.ref_distance(X)                          # same along the reference camera's rays (the height field)
    """

    def __init__(self, clip, ver=None):
        d = OUT / clip / 'background'
        ver = ver or sorted(p.name for p in d.glob('v[0-9][0-9]') if (p / 'model.npz').exists())[-1]
        self.dir = d / ver
        z = np.load(self.dir / 'model.npz')
        self.z = z
        self.X, self.F = z['rest_verts'], z['faces']
        self.V = views2.load(clip)
        self.ref_R, self.ref_pos, self.ref_f = z['ref_R'], z['ref_pos'], float(z['ref_f'])
        self.ref_ou, self.ref_ov = float(z['ref_ou']), float(z['ref_ov'])
        self.Zref, self.cell = z['ref_surface_depth'], int(z['ref_cell'])
        self.bodies = [(str(n), z[f'body{i}_verts'], z[f'body{i}_faces'], z[f'body{i}_piece'],
                        z[f'body{i}_verts4d'] if f'body{i}_verts4d' in z.files else None)
                       for i, n in enumerate(z['body_names'])]
        o = np.load(self.dir / 'occupancy.npz')
        self.occ, self.occ_scale = o['bg_depth'], int(o['scale'])

    def ref_distance(self, X):
        cam = self.V.cam
        q, zr = cam.project(np.asarray(X, float), self.ref_R, self.ref_f, self.ref_pos)
        ci = np.floor((q[:, 0] - self.ref_ou) / self.cell).astype(int)
        cj = np.floor((q[:, 1] - self.ref_ov) / self.cell).astype(int)
        H, W = self.Zref.shape
        ok = (ci >= 0) & (ci < W) & (cj >= 0) & (cj < H)
        zs = np.full(len(zr), np.nan)
        zs[ok] = self.Zref[cj[ok], ci[ok]]
        return zs - zr

    def ray_distance(self, X, k):
        q, z = self.V.project(np.asarray(X, float), k)
        s = self.occ_scale
        xi, yi = np.floor(q[:, 0] / s).astype(int), np.floor(q[:, 1] / s).astype(int)
        h, w = self.occ.shape[1:]
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        zb = np.full(len(z), np.nan)
        zb[ok] = self.occ[k, yi[ok], xi[ok]].astype(float)
        zb[~np.isfinite(zb)] = np.nan
        return zb - z


# ---------------------------------------------------------------- pictures
LABEL_RGB = [(160, 160, 160), (255, 120, 200), (90, 200, 255), (255, 200, 80), (120, 255, 140), (200, 140, 255)]
BODY_RGB = [(255, 230, 40), (40, 255, 230), (255, 90, 60), (140, 255, 40), (255, 140, 255), (60, 140, 255)]


def sheet(R, ks, path, scale=0.5):
    """Per frame: video | textured render (surface + bodies) where the video shows background, organs / instruments
    greyed | labels (surface parts, bodies outlined, red = background in front of an organ / instrument by > 2 mm) |
    |depth diff| 0-10 mm."""
    V, surf, bodies, M = R['V'], R['surf'], R['bodies'], R['M']
    rows = []
    for k in ks:
        Mk, Zr, RGB, owner = render_all(V, surf, bodies, k)
        fr = V.frames[k]
        fg = R['ins'][k] | R['org'][k]
        rgb = np.clip(RGB, 0, 255).astype(np.uint8)
        rgb[fg] = (0.35 * fr[fg] + 0.65 * 60).astype(np.uint8)
        lab = fr.astype(np.float32) * 0.55
        # surface labels by vertex label -> per pixel via render of label colours
        q, z = V.project(surf['X'], k)
        zb, fid, bary = B.raster(q, z, surf['F'], V.H, V.W)
        col = np.array([LABEL_RGB[i % len(LABEL_RGB)] for i in surf['label']], np.float32)
        col[~surf['observed']] = (90, 120, 255)
        cimg = B.interp_attr(surf['F'], fid, bary, col)
        sv = (owner == 0) & ~fg
        lab[sv] = 0.45 * lab[sv] / 0.55 + 0.55 * cimg[sv]
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        for i, b in enumerate(bodies):
            vis = (owner == i + 1).astype(np.uint8)
            cnt = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(lab, cnt, -1, BODY_RGB[i % len(BODY_RGB)], 2)
            mc = cv2.findContours(M[b['name']][k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(lab, mc, -1, (255, 255, 255), 1)
        Dk = V.depth(k)
        bad = Mk & ero(fg, 3) & (Zr < Dk - 0.002)
        lab[bad] = (255, 30, 30)
        oc = cv2.findContours(R['org'][k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(lab, oc, -1, (255, 255, 255), 1)
        dz = np.where(Mk & ~fg, np.abs(Zr - Dk), 0) * 1e3
        zimg = cv2.applyColorMap(np.clip(dz * 25, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
        zimg[fg] = (zimg[fg] * 0.25).astype(np.uint8)
        tiles = [fr.copy(), rgb, lab, zimg]
        for t, s in zip(tiles, (f'video {k}', 'render (organs/tools grey)', 'parts, bodies, red=in front', '|dz| 0-10 mm')):
            cv2.putText(t, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.concatenate([cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles], 1))
    img = np.concatenate(rows, 0)
    y = 14
    names = surf['label_names'] + ['filled (hidden)'] + [b['name'] for b in bodies]
    cols = [LABEL_RGB[i % len(LABEL_RGB)] for i in range(len(surf['label_names']))] + [(90, 120, 255)] + \
        [BODY_RGB[i % len(BODY_RGB)] for i in range(len(bodies))]
    x0 = img.shape[1] // 2 + 4
    for n, c in zip(names, cols):
        cv2.putText(img, n[:38], (x0, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
        y += 13
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def views3d(R, path, organ_frame=None):
    """Surface + bodies (+ the organ 4D models at one frame, translucent red) from the reference view and 3 oblique
    angles: textured (top) and shaded geometry (bottom: grey observed, blue filled, orange filled behind organs,
    body colours)."""
    V, ref, surf, bodies = R['V'], R['ref'], R['surf'], R['bodies']
    X, F, uv, tex = surf['X'], surf['F'], surf['uv'], surf['tex']
    c = np.median(X, 0)
    k_org = organ_frame if organ_frame is not None else V.n // 2
    top, bot = [], []
    for ang, axis in ((0, ref.R[1]), (-55, ref.R[1]), (55, ref.R[1]), (60, ref.R[0])):
        Rt = B.rodrigues(axis, np.radians(ang))
        Rv = ref.R @ Rt.T
        pv = c + Rt @ (ref.pos - c)
        f = ref.f * 0.62
        meshes = [(X, F, uv, tex)] + [(b['X'], b['F'], b['uv'], tex) for b in bodies]
        Mt = np.zeros((V.H, V.W), bool)
        Zt = np.full((V.H, V.W), np.inf)
        RGB = np.zeros((V.H, V.W, 3), np.float32)
        shade = np.full((V.H, V.W, 3), 30, np.float32)
        geo = [(X, F, None)] + [(b['X'], b['F'], i) for i, b in enumerate(bodies)]
        for (Xm, Fm, uvm, texm), (_, _, bi) in zip(meshes, geo):
            q, z = V.cam.project(Xm, Rv, f, pv)
            m, zb, rgb, fid = B.render_textured(q, z, Fm, uvm, texm, V.H, V.W)
            zb2, fid2, bary2 = B.raster(q, z, Fm, V.H, V.W)
            vn = B.vertex_normals(Xm, Fm)
            sh = np.abs(vn @ (-Rv[2])) * 0.75 + 0.25
            if bi is None:
                base = np.ones((len(Xm), 3)) * 200
                base[~surf['observed']] = (120, 150, 230)
                base[surf['under_organ']] = (240, 160, 90)
            else:
                base = np.tile(np.array(BODY_RGB[bi % len(BODY_RGB)], float), (len(Xm), 1))
            col = interp = B.interp_attr(Fm, fid2, bary2, (base * sh[:, None]).astype(np.float32))
            sel = m & (zb < Zt)
            Mt |= sel
            Zt[sel] = zb[sel]
            RGB[sel] = rgb[sel]
            shade[sel] = col[sel]
        # organ models at one frame (red, translucent, z-tested)
        for n, mdl in R['models'].items():
            q, z = V.cam.project(mdl['verts4d'][k_org], Rv, f, pv)
            zb, fid, _ = B.raster(q, z, mdl['faces'], V.H, V.W)
            m = fid >= 0
            front = m & (zb < Zt)
            back = m & ~front
            for img in (RGB, shade):
                img[front] = 0.45 * img[front] + 0.55 * np.array([230, 40, 40])
                img[back] = 0.8 * img[back] + 0.2 * np.array([230, 40, 40])
        top.append(np.clip(RGB, 0, 255).astype(np.uint8))
        bot.append(np.clip(shade, 0, 255).astype(np.uint8))
        cv2.putText(top[-1], f'view {ang:+d} deg', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bot[0], f'grey seen, blue filled, orange behind organs; red: organ models (frame {k_org})', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def canvas_pictures(R, path):
    """Reference-canvas maps: texture, frames observing, spread (MAD), fused surface depth; outline = hidden cells,
    magenta = cells behind organs (lower bound), red = lower-bound conflicts in observed cells."""
    tex = R['surf']['tex'].copy()
    up = lambda a: cv2.resize(a.astype(np.float32), (tex.shape[1], tex.shape[0]), interpolation=cv2.INTER_NEAREST)
    cimg = cv2.applyColorMap(np.clip(up(R['cnt']), 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    mimg = cv2.applyColorMap(np.clip(up(np.nan_to_num(R['mad'] * 1e3)) * 40, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
    Zc = R['Zc'] * 1e3
    lo, hi = np.percentile(Zc[R['domc']], [2, 98])
    zimg = cv2.applyColorMap(np.clip((Zc - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    hid = up(R['dom'] & ~R['known']) > 0.5
    behind = up(R['dom'] & np.isfinite(R['lower'])) > 0.5
    conf = up(R['conflict']) > 0.5
    for im in (tex, cimg, mimg, zimg):
        im[~R['domc']] = 0
    c1 = cv2.findContours(hid.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    c2 = cv2.findContours(behind.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for im, s in ((tex, 'texture'), (cimg, 'frames observing (0-255)'), (mimg, 'spread MAD 0-6 mm'), (zimg, f'depth {lo:.0f}-{hi:.0f} mm')):
        cv2.drawContours(im, c1, -1, (0, 255, 255), 1)
        cv2.drawContours(im, c2, -1, (255, 0, 255), 1)
        im[conf] = (255, 0, 0)
        cv2.putText(im, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    img = np.concatenate([np.concatenate([tex, cimg], 1), np.concatenate([mimg, zimg], 1)], 0)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- report
DEFS = dict(
    coverage="fraction of the frame's static pixels (scope, 8 px border, instruments and organs dilated 8 px removed) covered by the rendered surface + bodies",
    depth_med_mm='median |z_render - z_video| over static covered pixels (mm), per frame; p90 likewise',
    depth_bias_mm='signed median z_render - z_video (negative = background in front of the video surface)',
    front_static='fraction of static covered pixels where the background is > 5 mm in front of the video depth',
    front_organ='fraction of visible organ pixels (eroded 3 px) where the background is > 2 mm in front of the organ video depth',
    front_instrument='same for instrument pixels (video depth on thin shafts is unreliable: indicative)',
    organ_model_behind="fraction of an organ 4D model's vertices that lie > 1 mm behind the background along the frame's camera rays (max over organs); v1: 15-24 % for the gallbladder",
    organ_model_behind_mm='largest such penetration depth (mm)',
    photo='L1 (0-255), global NCC and 15x15 local NCC of the textured render vs the video on static covered non-specular pixels',
    body_iou="IoU of the body's visible render with its SAM mask, pixels near instruments / organs ignored",
    body_depth_med_mm='median |z_body - z_video| where both cover',
)


def report(R, out, per, per_body, per_organ, extra):
    V = R['V']
    n = V.n
    ev = [k for k in range(n) if not R['hold'][k]]
    sel = lambda v: np.where(R['hold'], np.nan, v)
    q = dict(clip=R['clip'], version=R['ver'], seg=R['seg'], params=R['p'], definitions=DEFS,
             objects={k: {kk: vv for kk, vv in c.items()} for k, c in R['cls'].items()},
             organ_lower_bound=R['lower_src'],
             organ_models={nm: f"{m['organ']}/{m['version']}" for nm, m in R['models'].items()},
             summary={k: summarize_thirds(sel(v), n) for k, v in per.items()},
             front_organ_per_organ={k: summarize_thirds(sel(v), n) for k, v in per_organ.items()},
             bodies={b['name']: dict(static=bool(b['static']), pieces=int(b['piece'].max() + 1),
                                     mesh=quality.mesh_health(b['X'], b['F']),
                                     motion=b['motion'], build=R['body_info'][b['name']],
                                     iou=summarize_thirds(sel(per_body[b['name']]['iou']), n),
                                     depth_med_mm=summarize_thirds(sel(per_body[b['name']]['depth_med_mm']), n),
                                     front_organ=summarize_thirds(sel(per_body[b['name']]['front_organ']), n))
                     for b in R['bodies']},
             surface=dict(mesh=quality.mesh_health(R['surf']['X'], R['surf']['F']), checks=mesh_checks(R['surf']['X'], R['surf']['F'], R['ref'], (R['surf']['n_obs'] < 10) & ~R['surf']['under_organ']),
                          observed_vertex_fraction=round(float(R['surf']['observed'].mean()), 4),
                          behind_organ_vertex_fraction=round(float(R['surf']['under_organ'].mean()), 4),
                          lower_bound_conflict_cells=int(R['conflict'].sum()),
                          lower_bound_conflict_fraction_of_observed=round(float(R['conflict'].sum() / max(R['known'].sum(), 1)), 4),
                          spread_mm=dict(median=round(float(np.nanmedian(R['mad'][R['known']]) * 1e3), 3),
                                         p90=round(float(np.nanpercentile(R['mad'][R['known']], 90) * 1e3), 3)),
                          texture_seen=round(float(R['seen'][R['domc']].mean()), 4),
                          labels={nm: round(float((R['surf']['label'] == i).mean()), 4) for i, nm in enumerate(R['surf']['label_names'])}),
             per_frame={k: np.round(v, 4).tolist() for k, v in per.items()},
             build_seconds=round(R['t_build']), **extra)
    sc = np.asarray(R.get('scale', np.ones(n)))
    q['depth_scale'] = dict(note='per-frame scale of the video depth that brings the static pixels onto the multi-view '
                                 'median (the background is fused with scaled depth; metrics *_raw use the raw depth)',
                            p5=round(float(np.percentile(sc, 5)), 4), p95=round(float(np.percentile(sc, 95)), 4),
                            per_frame=np.round(sc, 4).tolist())
    for nm, m in R['models'].items():
        if m.get('depth_scale') is not None and len(m['depth_scale']) == n:
            q['depth_scale'][f'corr_with_organ_{nm}_scale'] = round(float(np.corrcoef(sc, m['depth_scale'])[0, 1]), 3)
    q['surface']['lower_bound_conflict_cells_by_organ'] = R.get('conflict_by', {})
    if R.get('contra') is not None:
        q['surface']['visibility_contradicted_cells'] = int(R['contra'].sum())
        q['surface']['visibility_contradicted_fraction'] = round(float(R['contra'].sum() / max((R['known'] | R['contra']).sum(), 1)), 4)
    if R.get('mov_thr') is not None:
        t = R['mov_thr'][np.isfinite(R['mov_thr'])]
        q['flow_reject'] = dict(threshold_px_median=round(float(np.median(t)), 2), threshold_px_p90=round(float(np.percentile(t, 90)), 2))
    (out / 'quality.json').write_text(json.dumps(q, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    (out / 'quality.json').write_text(json.dumps(q, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    return q


def fmt(d, key='median', nd=2):
    v = d.get(key) if isinstance(d, dict) else None
    return '-' if v is None else f'{v:.{nd}f}'


def write_notes(q, out, extra_text=''):
    """NOTES.md from quality.json (numbers) plus free text (visual checks, problems)."""
    S = q['summary']
    th = [k for k in S['coverage'] if k != 'all']
    L = [f"# {q['clip']} / background {q['version']}", '',
         f"`PYTHONPATH=. .venv/bin/python -m t2s.background {q['clip']} {q['version']}{' --holdout' if 'holdout' in q else ''}` "
         f"(t2s/background.py; segmentation {q['seg']}; organ models {q['organ_models'] or 'none'}; build {q['build_seconds']} s)", '',
         '## Objects', '', '| object | kind | why |', '|---|---|---|']
    for n, c in q['objects'].items():
        L.append(f"| {n} | {c['kind']} | {c['why']} |")
    L += ['', 'Organ lower bound (surface kept >= %.0f mm behind the back of every organ in every frame):' % (q['params']['organ_margin'] * 1e3)]
    for n, v in q['organ_lower_bound'].items():
        L.append(f'- {n}: {v}')
    L += ['', '## Surface (per third of the clip: median; p90 in brackets)', '',
          '| metric | ' + ' | '.join(['all'] + th) + ' |', '|---' * (len(th) + 2) + '|']
    rows = [('coverage of static pixels', 'coverage', 3), ('depth residual mm (fused depth scale)', 'depth_med_mm', 2),
            ('depth residual mm (raw video depth)', 'depth_med_raw_mm', 2), ('depth p90 mm (raw)', 'depth_p90_raw_mm', 2),
            ('photometric L1', 'photo_l1', 2), ('photometric NCC', 'photo_ncc', 3), ('local NCC 15x15', 'photo_lncc', 3),
            ('front of static pixels > 5 mm (raw)', 'front_static_raw', 3), ('front of organ pixels > 2 mm', 'front_organ', 3),
            ('front of instrument pixels > 2 mm', 'front_instrument', 3),
            ('organ 4D model vertices behind the surface', 'organ_model_behind', 4)]
    for label, k, nd in rows:
        if k not in S:
            continue
        cells = [f"{fmt(S[k][t], 'median', nd)} ({fmt(S[k][t], 'p90', nd)})" for t in ['all'] + th]
        L.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    if q.get('front_organ_per_organ'):
        L += ['', 'Front of organ pixels per organ (median / p90 over frames): ' + ', '.join(
            f"{n} {fmt(v['all'], 'median', 3)} / {fmt(v['all'], 'p90', 3)}" for n, v in q['front_organ_per_organ'].items())]
    su = q['surface']
    L += ['', f"Mesh: {su['mesh']['n_verts']} vertices, {su['mesh']['n_faces']} faces, {su['checks']['components']} component(s), "
              f"area {su['mesh']['area_cm2']} cm2, min edge {su['mesh']['min_edge_mm']} mm; steep faces (> 80 deg to the "
              f"reference ray) {su['checks']['steep_face_fraction']:.1%} of the area, of which in low-confidence cells "
              f"{su['checks'].get('steep_lowconf_fraction', float('nan')):.1%}; spike vertices (> 3 mm off the neighbour mean) "
              f"{su['checks']['spike_vertices_3mm']}; closed slab watertight: {q.get('slab', {}).get('watertight')}.",
          f"Observed {su['observed_vertex_fraction']:.1%} of the vertices, filled behind organs {su['behind_organ_vertex_fraction']:.1%}; "
          f"multi-view spread (MAD) median {su['spread_mm']['median']} mm, p90 {su['spread_mm']['p90']} mm; texture seen "
          f"{su['texture_seen']:.1%}; labels {su['labels']}.",
          f"Organ thicker than the room behind it (observed static cells where an organ's back would lie behind the "
          f"surface; not pushed, reported): {su['lower_bound_conflict_cells']} cells "
          f"({su['lower_bound_conflict_fraction_of_observed']:.1%} of the observed cells), by organ "
          f"{su.get('lower_bound_conflict_cells_by_organ')}.",
          f"Per-frame depth scale (static pixels onto the multi-view median): p5 {q['depth_scale']['p5']}, p95 {q['depth_scale']['p95']}"
          + ''.join(f"; {k} {v}" for k, v in q['depth_scale'].items() if k.startswith('corr')) + '.']
    if 'holdout' in q:
        h = q['holdout']
        L += ['', '## Held-out frames (k % 10 == 5 left out of all fusion)', '', '| metric | held-out model | full model, same frames |', '|---|---|---|']
        for k in h['heldout_model']:
            L.append(f"| {k} | {fmt(h['heldout_model'][k], 'median', 3)} | {fmt(h['full_model_same_frames'][k], 'median', 3)} |")
        for n, v in h.get('heldout_bodies', {}).items():
            L.append(f"| body {n}: IoU / depth mm | {fmt(v['iou'], 'median', 3)} / {fmt(v['depth_med_mm'], 'median', 2)} | |")
    if q['bodies']:
        L += ['', '## Bodies', '', '| body | static | pieces | volume ml | protrudes mm (median / p90) | IoU ' + ' / '.join(['all'] + th) +
              ' | depth mm ' + ' / '.join(['all'] + th) + ' | in front of organs |', '|---|---|---|---|---|---|---|---|']
        for n, b in q['bodies'].items():
            L.append(f"| {n} | {b['static']} | {b['pieces']} | {b['build'].get('volume_ml')} | "
                     f"{b['build'].get('protrude_med_mm')} / {b['build'].get('protrude_p90_mm')} | "
                     + ' / '.join(fmt(b['iou'][t], 'median', 3) for t in ['all'] + th) + ' | '
                     + ' / '.join(fmt(b['depth_med_mm'][t], 'median', 2) for t in ['all'] + th) + ' | '
                     + f"{fmt(b['front_organ']['all'], 'median', 3)} |")
        L.append('')
        L.append('Motion test per body (per-frame translation fitted to mask + depth; moving only if it raises the IoU by '
                 f">= {q['params']['move_gain']} AND the flow residual is >= {q['params']['motion_ratio']} x the static one): "
                 + '; '.join(f"{n}: {b['motion']}" for n, b in q['bodies'].items()))
    L += ['', extra_text.strip(), '']
    (out / 'NOTES.md').write_text('\n'.join(L))


def run(clip, ver, holdout=False, refine_cams=False):
    p = dict(VERSIONS[ver])
    out = OUT / clip / 'background' / ver
    (out / 'work').mkdir(parents=True, exist_ok=True)
    R = build(clip, ver, p)
    t0 = time.time()
    extra = save(R, out)
    try:
        extra['hfield_check'] = dict(tiles=hfield_check(out), single=hfield_check(out, tiles=False))
        log('hfield check', extra['hfield_check'])
    except Exception as e:                            # noqa: BLE001 (report, do not fail the run)
        extra['hfield_check'] = dict(error=repr(e))
    per, per_body, per_organ = evaluate(R)
    log('evaluated', f'{time.time() - t0:.0f}s')
    if holdout:
        ph = dict(p, holdout_mod=(10, 5), seg=R['seg'],                  # same inputs as the full model
                  organ_pin={m['organ']: m['version'] for m in R['models'].values()})
        Rh = build(clip, ver, ph, V=R['V'])
        hk = [k for k in range(R['V'].n) if k % 10 == 5]
        perh, per_bh, _ = evaluate(Rh, hk)
        keys = ['photo_l1', 'photo_ncc', 'photo_lncc', 'depth_med_mm', 'depth_p90_mm', 'coverage']
        extra['holdout'] = dict(note='frames k % 10 == 5 left out of depth, body and texture fusion, evaluated on them',
                                frames=hk,
                                heldout_model={k: quality.summarize(perh[k][hk]) for k in keys},
                                full_model_same_frames={k: quality.summarize(per[k][hk]) for k in keys},
                                heldout_bodies={n: dict(iou=quality.summarize(v['iou'][hk]),
                                                        depth_med_mm=quality.summarize(v['depth_med_mm'][hk]))
                                                for n, v in per_bh.items()})
        log('holdout', {k: v.get('median') for k, v in extra['holdout']['heldout_model'].items()})
    q = report(R, out, per, per_body, per_organ, extra)
    extra_txt = (out / 'work' / 'observations.md').read_text() if (out / 'work' / 'observations.md').exists() else ''
    write_notes(q, out, extra_txt)
    occupancy(R, out)
    V = R['V']
    ks = np.linspace(0, V.n - 1, 6).astype(int)
    sheet(R, ks, out / 'sheet.jpg')
    views3d(R, out / 'views3d.jpg')
    canvas_pictures(R, out / 'canvas.jpg')
    log('pictures', f'{time.time() - t0:.0f}s')
    if refine_cams:
        camera_experiment(R, out, p)
    s = q['summary']
    log('SUMMARY', clip, ver, {k: (s[k]['all'].get('median'), s[k]['all'].get('p90')) for k in s},
        q['surface']['checks'], {n: (b['iou']['all'].get('median'), b['depth_med_mm']['all'].get('median'), b['static'])
                                 for n, b in q['bodies'].items()})
    return R, q


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    clip = args[0]
    ver = args[1] if len(args) > 1 else sorted(VERSIONS)[-1]
    if '--seg' in sys.argv:
        VERSIONS[ver]['seg'] = sys.argv[sys.argv.index('--seg') + 1]
    run(clip, ver, holdout='--holdout' in sys.argv, refine_cams='--refine-cams' in sys.argv)
