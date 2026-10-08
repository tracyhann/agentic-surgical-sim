"""Membrane track: the peritoneal sheet the left grasper lifts into a tent above the gallbladder (shot A, chole_a).

    python -m r2s.tissue.membrane vNN          # fit 3D + 4D with CONFIGS[vNN], evaluate, write outputs
    python -m r2s.tissue.membrane vNN --seg    # only (re)segment with SAM 2.1 and write the mask overlay

Model: a ruled shell between the line gripped in the jaws (row 0, attached to the grasper) and a base curve on the
gallbladder (last row), base = polar envelope of the sheet mask around the image apex lifted with the video depth,
interior depth pulled toward the video depth along camera rays only where it is meaningful; smoothed in time.
Current version v09: v08's mesh (rest triangles >= 15 deg) re-snapped to gallbladder v12's 4D (exact base depth,
no other vertex behind it), refined cameras, held jaw row at the interaction v06 grasper TCP with a short tongue to
where the sheet leaves the jaws. Iterations: outputs/iter/tissues/membrane/v05 ... v09 NOTES.md.

Outputs (outputs/iter/tissues/membrane/vNN/, CONTRACT.md): model.npz, quality.json, sheet.jpg, views3d.jpg,
masks.npz (SAM 2.1 mask of the sheet, prompts in NOTES.md / PROMPTS below), work/ (scratch).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from r2s import views, quality as Q

ROOT = Path(__file__).resolve().parents[2] / 'outputs/iter/tissues/membrane'
CLIP = 'chole_a'

# SAM 2.1 point prompts of the tented sheet (x, y pixels of the 640x360 frame). Positives on the translucent sheet just
# below the grasper jaws and along its glossy ridges; negatives on the grasper shaft, the probe, the liver / fat behind
# and the gallbladder body away from the tent.
PROMPTS = [
    {"frame": 0, "pos": [[105, 200], [115, 250]], "neg": [[60, 300], [230, 60], [50, 60], [300, 200]]},
    {"frame": 40, "pos": [[175, 190], [200, 230], [165, 150]], "neg": [[150, 330], [120, 60], [280, 100], [330, 250]]},
    {"frame": 100, "pos": [[210, 130], [225, 170], [245, 215], [275, 160]],
     "neg": [[140, 300], [300, 40], [160, 30], [440, 120], [100, 120]]},
    {"frame": 150, "pos": [[215, 115], [225, 150], [240, 200], [270, 170]],
     "neg": [[130, 300], [300, 30], [165, 30], [420, 110], [100, 120]]},
    {"frame": 200, "pos": [[240, 130], [260, 180], [280, 140], [300, 110]],
     "neg": [[140, 320], [330, 40], [180, 20], [460, 140], [120, 120]]},
    {"frame": 250, "pos": [[215, 120], [250, 140], [265, 180], [300, 130]],
     "neg": [[120, 320], [330, 60], [185, 40], [470, 160], [120, 120]]},
]


# per-version settings (each version stays reproducible: python -m r2s.tissue.membrane vNN)
CONFIGS = {
    'v01': dict(prune=False, p_fit=True, resample=False, sa=2.0, sb=4.0),
    'v02': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=2.0, sb=4.0),
    'v03': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=2.0, sb=4.0, depth_refine=True,
                lam=2.0, w_upper=0.2, sd=3.0, rest='median_area'),
    'v04': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=2.0, sb=4.0, depth_refine=True,
                lam=2.0, w_upper=0.2, sd=3.0, rest='median_area', grid=(16, 24), pct=98),
    'v05': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=1.5, sb=3.0, depth_refine=True,
                lam=2.0, w_upper=0.2, sd=3.0, rest='median_area', grid=(16, 24), pct=98),
    # v06: integration fix. Base on the gallbladder v10 surface, no vertex behind it, apex = interaction grasp point.
    'v06': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=1.5, sb=3.0, depth_refine=True,
                lam=2.0, w_upper=0.2, sd=3.0, rest='balanced', grid=(16, 24), pct=98,
                gb_src='gallbladder/v10', grasp_src='interaction/v05', apex_from='ray_near_jaws',
                base_off=0.00025, margin=0.001, compare='v05', sa_apex=1.5, sb_after=1.5, sp_time=2.0,
                xcheck=dict(gallbladder='gallbladder/v10', interaction='interaction/v05')),
    # v07: refined cameras (backdrop v08), remeshed sheet (no slivers), same gallbladder / grasper consistency as v06
    'v07': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=1.5, sb=3.0, depth_refine=True,
                lam=2.0, w_upper=0.2, sd=3.0, rest='balanced', grid=(16, 24), pct=98,
                gb_src=('gallbladder/v11', 'gallbladder/v10'), grasp_src='interaction/v05', apex_from='ray_near_jaws',
                base_off=0.00025, margin=0.001, compare='v06', sa_apex=1.5, sb_after=1.5, sp_time=2.0,
                cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz', remesh=dict(h=0.0013, iters=20),
                pan_light=dict(sb=1.5, med=5, k0=55, k1=70), rest_from=62, exact_cast=False,
                xcheck=dict(gallbladder='gallbladder/v10', interaction='interaction/v05')),
    # v08: v07's mesh (same 237 vertices / 395 faces) re-snapped to gallbladder v13, grasper interaction v06; the held
    # jaw row sits at the grasper TCP and a short tongue (s < tongue_s0) joins it to the sheet where the video shows it
    # leaving the jaws
    'v08': dict(mask_from='v01', prune=True, p_fit=False, resample=True, sa=1.5, sb=3.0, depth_refine=True,
                lam=2.0, w_upper=0.2, sd=3.0, rest='fixed', rest_frame=162, grid=(16, 24), pct=98,
                gb_src='gallbladder/v13', grasp_src='interaction/v06', apex_from='ray_near_jaws',
                base_off=0.00025, margin=0.001, compare='v07', sa_apex=1.5, sb_after=1.5, sp_time=2.0,
                cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz', mesh_from='v07', tongue_s0=0.15,
                polish=True, polish_flips=True, jaw_clear=0.00025,
                pan_light=dict(sb=1.5, med=5, k0=55, k1=70), exact_cast=False,
                xcheck=dict(gallbladder='gallbladder/v13', interaction='interaction/v06')),
}
# v09: v08 (same vertices, faces, attachments) re-snapped to gallbladder v12's 4D, which the simulation uses
CONFIGS['v09'] = dict(CONFIGS['v08'], gb_src='gallbladder/v12', compare='v08', mesh_from='v08', polish=True,
                      base_exact_smooth=1.0, polish_per_vertex=True, sliver_repair=15.0, margin_taper=0.1,
                      xcheck=dict(gallbladder='gallbladder/v12', interaction='interaction/v06'))
# other tracks' outputs used only as cross-checks in quality.json (read-only; versions recorded there)
XCHECK = dict(gallbladder='gallbladder/v01', interaction='interaction/v01')


def vdir(version):
    d = ROOT / version
    (d / 'work').mkdir(parents=True, exist_ok=True)
    return d


def segment(V, prompts=PROMPTS):
    return views.segment_object(V, prompts)


def overlay_masks(V, M, ks, path, prompts=(), cols=4, scale=0.5, extra=None):
    """Mask outline (cyan) + tint over the frames; prompt points of that frame (green +, red -)."""
    tiles = []
    for k in ks:
        img = V.frames[k].astype(np.float32).copy()
        m = M[k]
        img[m] = 0.6 * img[m] + 0.4 * np.array([0, 255, 255], np.float32)
        img = np.ascontiguousarray(img)
        cnt = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(img, cnt, -1, (0, 255, 255), 1)
        for p in prompts:
            if p['frame'] == k:
                for q in p['pos']:
                    cv2.drawMarker(img, tuple(map(int, q)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                for q in p.get('neg', []):
                    cv2.drawMarker(img, tuple(map(int, q)), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        if extra is not None:
            extra(img, k)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.putText(img, f'frame {k}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def load_or_segment(V, d, prompts=PROMPTS, force=False, mask_from=None):
    f = d / 'masks.npz'
    if mask_from and not f.exists() and not force:
        import shutil
        shutil.copy(ROOT / mask_from / 'masks.npz', f)
    if f.exists() and not force:
        return np.load(f)['membrane']
    M = segment(V, prompts)
    np.savez_compressed(f, membrane=M, prompts=json.dumps(prompts))
    return M


# ----------------------------------------------------------------------------------------------------------------------
# observations per frame
NS, NT = 16, 12           # grid rows (apex -> base) and columns (left -> right side of the tent)
JAW_W = 0.005             # width of the gripped line in the jaws (m)
P_GRID = np.round(np.arange(1.0, 3.01, 0.25), 2)


def clean_mask(m):
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(bool)


def grasper_end(V, k):
    ys, xs = np.nonzero(V.mask('grasper')[k])
    if len(xs) < 50:
        return None
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    d = np.linalg.svd(P - c, full_matrices=False)[2][0]
    d = d if d[1] > 0 else -d                       # the shaft enters from the top: distal = down
    pr = (P - c) @ d
    return P[pr > np.percentile(pr, 99)].mean(0)


def _median_depth(D, x, y, r=2):
    x, y = int(round(np.clip(x, r, D.shape[1] - r - 1))), int(round(np.clip(y, r, D.shape[0] - r - 1)))
    return float(np.median(D[y - r:y + r + 1, x - r:x + r + 1]))


def shaft_dir2d(V, k):
    ys, xs = np.nonzero(V.mask('grasper')[k])
    if len(xs) < 50:
        return None
    P = np.c_[xs, ys].astype(float)
    d = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    return d if d[1] > 0 else -d


def observe(V, M, prune=False, pct=97):
    """Per frame: cleaned mask, image apex (sheet pixels nearest the grasper's distal end), video depth there, and the
    polar envelope of the mask around the apex (NT+1 base points) with the video depth just inside it."""
    n = V.n
    Mc = np.zeros_like(M)
    apex2d, apex_z, gend_all = np.zeros((n, 2)), np.zeros(n), np.zeros((n, 2))
    base2d, base_z = np.zeros((n, NT + 1, 2)), np.zeros((n, NT + 1))
    prev = None
    for k in range(n):
        m = clean_mask(M[k])
        Mc[k] = m
        ge = grasper_end(V, k)
        ge = prev if ge is None else ge
        prev = ge
        gend_all[k] = ge
        ys, xs = np.nonzero(m)
        Qp = np.c_[xs, ys].astype(float)
        dd = np.linalg.norm(Qp - ge, axis=1)
        a2 = Qp[np.argsort(dd)[:30]].mean(0)
        if prune:
            # sheet pixels proximal of the jaws along the shaft are the instrument (jaws / shaft not in its mask)
            sd = shaft_dir2d(V, k)
            if sd is not None:
                keep = (Qp - a2) @ sd > -6
                m[ys[~keep], xs[~keep]] = False
                Mc[k] = m = clean_mask(m)
                ys, xs = np.nonzero(m)
                Qp = np.c_[xs, ys].astype(float)
                a2 = Qp[np.argsort(np.linalg.norm(Qp - ge, axis=1))[:30]].mean(0)
        apex2d[k] = a2
        D = V.depth(k)
        near = (np.linalg.norm(Qp - a2, axis=1) < 10)
        apex_z[k] = float(np.median(D[ys[near], xs[near]]))
        q = Qp - a2
        r = np.linalg.norm(q, axis=1)
        ax = q[r > 8].mean(0)
        ax /= np.linalg.norm(ax)
        ap = np.array([-ax[1], ax[0]])
        phi = np.arctan2(q @ ap, q @ ax)
        sel = r > 8
        lo, hi = np.percentile(phi[sel], [1, 99])
        phis = np.linspace(lo, hi, NT + 1)
        hw = max((hi - lo) / NT / 2, np.radians(2))
        R = np.full(NT + 1, np.nan)
        for j, f in enumerate(phis):
            b = sel & (np.abs(phi - f) < hw)
            if b.sum() >= 5:
                R[j] = np.percentile(r[b], pct)
        ok = np.isfinite(R)
        R = np.interp(np.arange(NT + 1), np.nonzero(ok)[0], R[ok])
        from scipy.ndimage import median_filter
        R = median_filter(R, 3, mode='nearest')
        dirs = np.cos(phis)[:, None] * ax + np.sin(phis)[:, None] * ap
        base2d[k] = a2 + R[:, None] * dirs
        for j in range(NT + 1):
            x, y = a2 + max(R[j] - 4, 0) * dirs[j]
            base_z[k, j] = _median_depth(D, x, y)
    return dict(masks=Mc, apex2d=apex2d, apex_z=apex_z, gend=gend_all, base2d=base2d, base_z=base_z)


# ----------------------------------------------------------------------------------------------------------------------
# template: tent between an apex segment (jaws) and a base curve on the gallbladder
def grid_faces(ns=NS, nt=NT):
    idx = np.arange((ns + 1) * (nt + 1)).reshape(ns + 1, nt + 1)
    F = []
    for i in range(ns):
        for j in range(nt):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            F += [[a, c, b], [b, c, d]]
    return np.array(F, np.int32), idx


S_GRID = np.linspace(0, 1, NS + 1)
T_GRID = np.linspace(0, 1, NT + 1)
FACES, IDX = grid_faces()
# per-vertex sheet coordinates: s (0 = jaw line, 1 = base curve), t (0 = left side, 1 = right side); TOP = jaw-line
# vertices (attached to the grasper), BASE = base-curve vertices (attached to the gallbladder), both ordered by t
SV, TV = np.repeat(S_GRID, NT + 1), np.tile(T_GRID, NS + 1)
TOP, BASE = IDX[0], IDX[-1]
MESH_KIND = 'grid'


def set_grid(ns, nt):
    """Structured (ns+1) x (nt+1) grid (v01-v06); nt+1 is also the number of observed base points."""
    global NS, NT, S_GRID, T_GRID, FACES, IDX, SV, TV, TOP, BASE, MESH_KIND
    NS, NT = ns, nt
    S_GRID, T_GRID = np.linspace(0, 1, NS + 1), np.linspace(0, 1, NT + 1)
    FACES, IDX = grid_faces(NS, NT)
    SV, TV = np.repeat(S_GRID, NT + 1), np.tile(T_GRID, NS + 1)
    TOP, BASE = IDX[0], IDX[-1]
    MESH_KIND = 'grid'


def set_mesh(st, faces):
    """Unstructured triangulation of the (s, t) sheet domain (v07+); the observation keeps NT+1 base points."""
    global FACES, SV, TV, TOP, BASE, MESH_KIND
    SV, TV = st[:, 0].copy(), st[:, 1].copy()
    SV[np.isclose(SV, 0, atol=1e-9)], SV[np.isclose(SV, 1, atol=1e-9)] = 0.0, 1.0
    FACES = np.asarray(faces, np.int32)
    TOP = np.nonzero(SV == 0)[0][np.argsort(TV[SV == 0])]
    BASE = np.nonzero(SV == 1)[0][np.argsort(TV[SV == 1])]
    MESH_KIND = 'remeshed'


def interp_curve(B, t):
    """Polyline B (m, 3) with uniform knots on [0, 1] evaluated at t (k,)."""
    u = np.linspace(0, 1, len(B))
    return np.stack([np.interp(t, u, B[:, c]) for c in range(3)], -1)


def tent(A, B, p, jaw_w=JAW_W):
    """A (3,) apex, B (m, 3) base curve (uniform in t), p profile exponent (1: ruled surface between the jaw line
    and the base, >1: sides pinched toward the apex). Returns the sheet vertices at (SV, TV)."""
    Bc = B.mean(0)
    e = B[-1] - B[0]
    e /= max(np.linalg.norm(e), 1e-9)
    s, t = SV[:, None], TV[:, None]
    return A + s * (Bc - A) + s ** p * (interp_curve(B, TV) - Bc) + (1 - s) * (t - 0.5) * jaw_w * e


def fast_sil(V, X, k):
    q, z = V.project(X, k)
    img = np.zeros((V.H, V.W), np.uint8)
    tri = np.round(q[FACES]).astype(np.int32)
    cv2.fillPoly(img, list(tri), 1)
    return img.astype(bool)


def iou(a, b, valid=None):
    if valid is not None:
        a, b = a & valid, b & valid
    return float((a & b).sum() / max((a | b).sum(), 1))


def resample_curve(B, n=None):
    n = NT + 1 if n is None else n
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
    u = np.linspace(0, L[-1], n)
    return np.stack([np.interp(u, L, B[:, c]) for c in range(3)], 1)


def fit_frames(V, O, p_fit=True, resample=False):
    n = V.n
    A = np.array([V.unproject(*O['apex2d'][k], O['apex_z'][k], k) for k in range(n)])
    B = np.array([V.unproject(O['base2d'][k, :, 0], O['base2d'][k, :, 1], O['base_z'][k], k) for k in range(n)])
    if resample:
        B = np.stack([resample_curve(b) for b in B])
    P = np.ones(n)
    for k in range(n if p_fit else 0):
        valid = ~V.occluders(k)
        best = max(P_GRID, key=lambda p: iou(fast_sil(V, tent(A[k], B[k], p), k), O['masks'][k], valid))
        P[k] = best
    return A, B, P


def smooth(A, B, P, sa=2.0, sb=4.0, sp=4.0):
    from scipy.ndimage import median_filter, gaussian_filter1d
    A = gaussian_filter1d(median_filter(A, (5, 1), mode='nearest'), sa, axis=0, mode='nearest')
    B = gaussian_filter1d(median_filter(B, (9, 1, 1), mode='nearest'), sb, axis=0, mode='nearest')
    P = gaussian_filter1d(median_filter(P, 9, mode='nearest'), sp, mode='nearest')
    return A, B, P


def grid_laplacian():
    E = edges_of(FACES)
    N = len(SV)
    L = np.zeros((N, N))
    np.add.at(L, (E[:, 0], E[:, 0]), 1)
    np.add.at(L, (E[:, 1], E[:, 1]), 1)
    L[E[:, 0], E[:, 1]] -= 1          # each edge appears once in E
    L[E[:, 1], E[:, 0]] -= 1
    return L


def depth_offsets(V, X, k, mask, lam=2.0, w_upper=0.2):
    """Per-vertex depth offsets along the camera rays of frame k (the silhouette in frame k does not change) that pull
    the sheet toward the video depth where it is meaningful: weight 1 on the lower half of the tent (sheet on / just
    above the gallbladder), w_upper on the upper half (translucent sheet in front of a far background), 0 off the
    mask or on instruments; the jaw row and the base row stay fixed; a grid Laplacian keeps the offsets smooth."""
    q, z = V.project(X, k)
    D = V.depth(k)
    xi = np.clip(np.round(q[:, 0]).astype(int), 0, V.W - 1)
    yi = np.clip(np.round(q[:, 1]).astype(int), 0, V.H - 1)
    occ = cv2.dilate(V.occluders(k).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    w = np.where(SV >= 0.5, 1.0, w_upper) * (mask[yi, xi] & ~occ[yi, xi])
    d = np.array([_median_depth(D, a, b) for a, b in zip(xi, yi)])
    fixed = (SV == 0) | (SV == 1)
    A = np.diag(w) + lam * grid_laplacian()
    rhs = w * (d - z)
    free = ~fixed
    delta = np.zeros(len(X))
    delta[free] = np.linalg.solve(A[np.ix_(free, free)], rhs[free])
    return delta, q, z


def refine_depth(V, verts4d, masks, lam, w_upper, sd):
    from scipy.ndimage import gaussian_filter1d
    n = len(verts4d)
    deltas, qs, zs = np.zeros(verts4d.shape[:2]), np.zeros(verts4d.shape[:2] + (2,)), np.zeros(verts4d.shape[:2])
    for k in range(n):
        deltas[k], qs[k], zs[k] = depth_offsets(V, verts4d[k], k, masks[k], lam, w_upper)
    deltas = gaussian_filter1d(deltas, sd, axis=0, mode='nearest')
    out = np.stack([V.unproject(qs[k, :, 0], qs[k, :, 1], zs[k] + deltas[k], k) for k in range(n)])
    return out, deltas


# ----------------------------------------------------------------------------------------------------------------------
# evaluation
class MV:
    """views.Views with the membrane mask added (name 'membrane') and the jaw pixels (not in the grasper mask) counted
    as instrument pixels (unknown)."""

    def __init__(self, V, M, jaw_xy, jaw_r=10):
        self._V, self._M, self._jaw, self._r = V, M, jaw_xy, jaw_r

    def __getattr__(self, a):
        return getattr(self._V, a)

    def mask(self, name):
        return self._M if name == 'membrane' else self._V.mask(name)

    def occluders(self, k):
        u = self._V.occluders(k).copy()
        cv2.circle(u.view(np.uint8), tuple(int(round(c)) for c in self._jaw[k]), self._r, 1, -1)
        return u


def tri_area(X, F=None):
    F = FACES if F is None else F
    return 0.5 * np.linalg.norm(np.cross(X[..., F[:, 1], :] - X[..., F[:, 0], :], X[..., F[:, 2], :] - X[..., F[:, 0], :]), axis=-1).sum(-1)


def edges_of(F):
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    return np.unique(E, axis=0)


def evaluate(V, MVv, O, verts4d, rest, ks, xcheck=None):
    XC = xcheck or XCHECK
    sil = Q.silhouette_iou(MVv, verts4d, FACES, 'membrane', ks)
    sil_raw = Q.silhouette_iou(MVv, verts4d, FACES, 'membrane', ks, visible_only=False)
    bf4 = Q.boundary_f(MVv, verts4d, FACES, 'membrane', ks, tol=4)
    bf8 = Q.boundary_f(MVv, verts4d, FACES, 'membrane', ks, tol=8)
    dres = Q.depth_residual(MVv, verts4d, FACES, 'membrane', ks)
    # depth residual split by the grid row of the visible face: lower half (sheet lying on / near the gallbladder,
    # depth meaningful) vs upper half (free tent between the jaws and the organ, the translucent sheet in front of a
    # far background: the video depth there is not the sheet's)
    s_of_face = SV[FACES].mean(1)
    dlow, dup, dsign_up = [], [], []
    for k in ks:
        m, z, ids = Q.render(MVv, verts4d, FACES, k, with_ids=True)
        ok = m & O['masks'][k] & ~MVv.occluders(k)
        sf = np.where(ids >= 0, s_of_face[np.maximum(ids, 0)], -1)
        lo, up = ok & (sf >= 0.5), ok & (sf < 0.5) & (sf >= 0)
        D = V.depth(k)
        dlow.append(float(np.median(np.abs(z[lo] - D[lo]))) if lo.sum() > 30 else np.nan)
        dup.append(float(np.median(np.abs(z[up] - D[up]))) if up.sum() > 30 else np.nan)
        dsign_up.append(float(np.median(D[up] - z[up])) if up.sum() > 30 else np.nan)
    n = V.n
    apex_mesh = verts4d[:, TOP].mean(1)
    tip = V.tools['grasper_left']['tip']
    apex_px = np.array([V.project(apex_mesh[k:k + 1], k)[0][0] for k in range(n)])
    tip_px = np.array([V.project(tip[k:k + 1], k)[0][0] for k in range(n)])
    apex_img_err = np.linalg.norm(apex_px - O['apex2d'], axis=1)
    tip_img_err = np.linalg.norm(tip_px - O['apex2d'], axis=1)
    apex_tip_3d = np.linalg.norm(apex_mesh - tip, axis=1)
    # base on the gallbladder: |depth of base vertices - video depth| where they project onto gallbladder pixels
    gbz = []
    for k in ks:
        q, z = V.project(verts4d[k, BASE], k)
        xi, yi = np.clip(np.round(q[:, 0]).astype(int), 0, V.W - 1), np.clip(np.round(q[:, 1]).astype(int), 0, V.H - 1)
        on = V.mask('gallbladder')[k][yi, xi]
        gbz.append(float(np.median(np.abs(z[on] - V.depth(k)[yi[on], xi[on]]))) if on.sum() >= 3 else np.nan)
    area = tri_area(verts4d) * 1e4
    E = edges_of(FACES)
    smax, s95 = Q.stretch(rest, verts4d, E)
    L0 = np.linalg.norm(rest[E[:, 0]] - rest[E[:, 1]], axis=1)
    s05 = np.percentile(np.linalg.norm(verts4d[:, E[:, 0]] - verts4d[:, E[:, 1]], axis=2) / L0, 5, axis=1)
    on_gb = []
    for k in range(n):
        q, _ = V.project(verts4d[k, BASE], k)
        xi, yi = np.clip(np.round(q[:, 0]).astype(int), 0, V.W - 1), np.clip(np.round(q[:, 1]).astype(int), 0, V.H - 1)
        gbd = cv2.dilate(V.mask('gallbladder')[k].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        on_gb.append(float(gbd[yi, xi].mean()))
    xc = {}
    try:
        G = np.load(ROOT.parent / XC['gallbladder'] / 'model.npz')
        gd = [np.median(Q.chamfer(verts4d[k, BASE], G['verts4d'][k], G['faces'], n_samples=30000)) for k in ks]
        xc['base_to_gallbladder_model_mm'] = dict(source=XC['gallbladder'], note='median distance base vertices -> GB mesh surface', **Q.summarize(np.array(gd) * 1000))
        ah = [Q.chamfer(apex_mesh[k:k + 1], G['verts4d'][k], G['faces'], n_samples=30000)[0] for k in ks]
        xc['apex_height_above_gallbladder_model_mm'] = dict(source=XC['gallbladder'], **Q.summarize(np.array(ah) * 1000))
    except Exception as e:  # other track not available
        xc['base_to_gallbladder_model_mm'] = dict(error=str(e))
    try:
        I = np.load(ROOT.parent / XC['interaction'] / 'model.npz')
        T = I['grasper_left_tip']
        tpx = np.array([V.project(T[k:k + 1], k)[0][0] for k in range(n)])
        dz = np.array([V.project(T[k:k + 1], k)[1][0] - V.project(apex_mesh[k:k + 1], k)[1][0] for k in range(n)])
        xc['apex_to_interaction_tip_mm'] = dict(source=XC['interaction'], **Q.summarize(np.linalg.norm(T - apex_mesh, axis=1) * 1000))
        xc['interaction_tip_to_image_jaw_px'] = dict(**Q.summarize(np.linalg.norm(tpx - O['apex2d'], axis=1)))
        xc['interaction_tip_minus_apex_depth_mm'] = dict(note='>0: their tip is farther from the camera', **Q.summarize(dz * 1000))
    except Exception as e:
        xc['apex_to_interaction_tip_mm'] = dict(error=str(e))
    acc = np.linalg.norm(verts4d[2:] - 2 * verts4d[1:-1] + verts4d[:-2], axis=2).mean(1) * 1000
    lift = np.array([np.linalg.norm(apex_mesh[k] - verts4d[k, BASE].mean(0)) for k in range(n)]) * 1000
    S = Q.summarize
    r4 = lambda x: [None if not np.isfinite(v) else round(float(v), 4) for v in x]
    out = dict(
        keyframes=ks,
        silhouette_iou=dict(per_keyframe=r4(sil), **S(sil)),
        silhouette_iou_raw=dict(note='projection without the video-depth visibility cut', per_keyframe=r4(sil_raw), **S(sil_raw)),
        boundary_f_4px=dict(per_keyframe=r4(bf4), **S(bf4)),
        boundary_f_8px=dict(per_keyframe=r4(bf8), **S(bf8)),
        depth_residual_mm=dict(note='all sheet pixels; translucent: see split', per_keyframe=r4(dres * 1000), **S(dres * 1000)),
        depth_residual_lower_mm=dict(note='lower half of the tent (on/near the gallbladder): meaningful', per_keyframe=r4(np.array(dlow) * 1000), **S(np.array(dlow) * 1000)),
        depth_residual_upper_mm=dict(note='upper half (free sheet in front of the far background): not meaningful', per_keyframe=r4(np.array(dup) * 1000), **S(np.array(dup) * 1000)),
        video_minus_mesh_depth_upper_mm=dict(note='>0: video depth sees behind the sheet', **S(np.array(dsign_up) * 1000)),
        base_on_gallbladder_mm=dict(note='|z base vertex - video depth| on gallbladder pixels', per_keyframe=r4(np.array(gbz) * 1000), **S(np.array(gbz) * 1000)),
        apex_to_image_jaw_px=dict(note='projected mesh apex vs sheet apex in the image (all frames)', **S(apex_img_err)),
        grasper_tip_to_image_jaw_px=dict(note='projected V.tools grasper tip vs sheet apex in the image', **S(tip_img_err)),
        apex_to_grasper_tip_mm=dict(note='3D distance mesh apex - V.tools grasper tip (all frames)', **S(apex_tip_3d * 1000)),
        area_cm2=dict(per_keyframe=r4(area[ks]), **S(area)),
        lift_mm=dict(note='apex to base-centroid distance', **S(lift)),
        stretch_max=dict(note='max edge length / rest length per frame', **S(smax)),
        stretch_p95=dict(**S(s95)),
        stretch_p05=dict(note='5th percentile edge ratio (<1: compressed vs rest)', **S(s05)),
        base_on_gallbladder_fraction=dict(note='fraction of base vertices projecting on the (dilated) GB mask', **S(np.array(on_gb))),
        crosscheck=xc,
        accel_mm_per_frame2=dict(note='mean vertex acceleration (temporal smoothness)', **S(acc)),
        mesh_health=Q.mesh_health(rest, FACES),
    )
    series = dict(s05=s05, apex_img_err=apex_img_err, tip_img_err=tip_img_err, apex_tip_3d=apex_tip_3d, area=area, smax=smax,
                  s95=s95, lift=lift, apex_px=apex_px, tip_px=tip_px)
    return out, series


def views3d(V, verts, path, k=150, size=420):
    """Rest shape from 3 angles (orthographic, flat-shaded, painter's order) with the gallbladder's visible points of
    frame k in grey for context; red = jaw (grasper) vertices, green = base (gallbladder) vertices."""
    G = V.points('gallbladder', k, step=6)
    c = verts.mean(0)
    tiles = []
    for el, az in [(25, -60), (80, -90), (5, 0)]:
        e, a = np.radians(el), np.radians(az)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(np.pi / 2 - e), -np.sin(np.pi / 2 - e)], [0, np.sin(np.pi / 2 - e), np.cos(np.pi / 2 - e)]])
        R = Rx @ Rz          # world -> view (x right, y up, z toward viewer)
        sc = size / 0.07     # 70 mm across
        to2 = lambda X: np.c_[size / 2 + (X - c) @ R.T[:, 0] * sc, size / 2 - (X - c) @ R.T[:, 1] * sc]
        img = np.full((size, size, 3), 255, np.uint8)
        for p in to2(G).astype(int):
            if 0 <= p[0] < size and 0 <= p[1] < size:
                img[p[1], p[0]] = (170, 170, 170)
        Xv = (verts - c) @ R.T
        F = FACES
        nrm = np.cross(Xv[F[:, 1]] - Xv[F[:, 0]], Xv[F[:, 2]] - Xv[F[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        shade = 0.35 + 0.65 * np.abs(nrm[:, 2])
        q = to2(verts)
        for fi in np.argsort(Xv[F].mean(1)[:, 2]):
            col = tuple(int(v) for v in np.array([60, 190, 220]) * shade[fi])
            cv2.fillConvexPoly(img, np.round(q[F[fi]]).astype(np.int32), col)
            cv2.polylines(img, [np.round(q[F[fi]]).astype(np.int32)], True, (40, 90, 110), 1)
        for p in q[TOP].astype(int):
            cv2.circle(img, tuple(p), 3, (230, 30, 30), -1)
        for p in q[BASE].astype(int):
            cv2.circle(img, tuple(p), 3, (30, 160, 30), -1)
        # world axes
        for v, col, lab in [((1, 0, 0), (200, 0, 0), 'x'), ((0, 1, 0), (0, 150, 0), 'y'), ((0, 0, 1), (0, 0, 200), 'z')]:
            o = np.array([30, size - 30])
            d = np.array([np.array(v) @ R.T[:, 0], -(np.array(v) @ R.T[:, 1])]) * 22
            cv2.arrowedLine(img, tuple(o), tuple((o + d).astype(int)), col, 2)
            cv2.putText(img, lab, tuple((o + d * 1.2).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(img, f'elev {el} azim {az}  (70 mm)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(img)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def mask_sheet(V, O, verts4d, ks, path, series):
    """Mesh silhouette (magenta) vs our mask (cyan), image apex (yellow), grasper tip of V.tools (red)."""
    def extra(img, k):
        m = fast_sil(V, verts4d[k], k)
        cnt = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(img, cnt, -1, (255, 0, 255), 2)
        cv2.circle(img, tuple(int(c) for c in O['apex2d'][k]), 5, (255, 255, 0), 2)
        cv2.circle(img, tuple(int(c) for c in series['tip_px'][k]), 5, (255, 0, 0), 2)
        q, _ = V.project(verts4d[k, BASE], k)
        for a in q.astype(int):
            cv2.circle(img, tuple(a), 2, (0, 255, 0), -1)
    return overlay_masks(V, O['masks'], ks, path, extra=extra)


# ----------------------------------------------------------------------------------------------------------------------
# v06: consistency with the gallbladder model (other track, read-only) and the interaction grasp point
def zbuffer(V, X, F, k):
    """Nearest depth per pixel of a mesh in frame k, the smaller of the perspective-interpolated depth and the flat
    per-face mean depth that r2s.quality.render uses (conservative for 'is the sheet in front')."""
    q, z = V.project(X, k)
    zb = np.full((V.H, V.W), np.inf, np.float32)
    T, Z = q[F], z[F]
    lo, hi = np.floor(T.min(1)).astype(int), np.ceil(T.max(1)).astype(int)
    ok = (Z > 1e-4).all(1) & (hi[:, 0] >= 0) & (lo[:, 0] < V.W) & (hi[:, 1] >= 0) & (lo[:, 1] < V.H)
    for fi in np.nonzero(ok)[0]:
        x0, y0 = max(lo[fi, 0], 0), max(lo[fi, 1], 0)
        x1, y1 = min(hi[fi, 0], V.W - 1), min(hi[fi, 1], V.H - 1)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a, b, c = T[fi]
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-9:
            continue
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
        dz = np.minimum(1 / (w0 / Z[fi, 0] + w1 / Z[fi, 1] + w2 / Z[fi, 2]), Z[fi].mean())
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (dz < sub)
        sub[upd] = dz[upd]
    return zb


def sample_min(D, q, r=1):
    """Smallest depth in a (2r+1)^2 window at pixel positions q (inf outside the image / the mesh)."""
    out = np.full(len(q), np.inf)
    xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x, y = xi + dx, yi + dy
            ok = (x >= 0) & (x < D.shape[1]) & (y >= 0) & (y < D.shape[0])
            out[ok] = np.minimum(out[ok], D[y[ok], x[ok]])
    return out


class GBSurface:
    """The gallbladder model of another track: per-frame depth maps (float16) and closest surface points."""

    def __init__(self, V, src, n_samples=60000, seed=0):
        z = np.load(ROOT.parent / src / 'model.npz')
        self.src, self.X4, self.F = src, z['verts4d'], z['faces']
        self.D = np.stack([zbuffer(V, self.X4[k], self.F, k).astype(np.float16) for k in range(V.n)])
        rng = np.random.default_rng(seed)
        X = self.X4[0]
        a = np.linalg.norm(np.cross(X[self.F[:, 1]] - X[self.F[:, 0]], X[self.F[:, 2]] - X[self.F[:, 0]]), axis=1)
        self.fi = rng.choice(len(self.F), n_samples, p=a / a.sum())
        r = rng.random((n_samples, 2))
        r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
        self.bary = np.c_[1 - r.sum(1), r]

    def samples(self, k):
        X = self.X4[k][self.F[self.fi]]
        return np.einsum('nij,ni->nj', X, self.bary)

    def closest(self, P, k):
        from scipy.spatial import cKDTree
        S = self.samples(k)
        d, i = cKDTree(S).query(P)
        return S[i], d

    def depth_at(self, V, X, k):
        q, z = V.project(X, k)
        return q, z, sample_min(self.D[k].astype(np.float32), q)


def ray_hit(V, GB, X, k):
    """Exact first intersection of the camera rays of frame k through the points X with the gallbladder model
    (Moller-Trumbore against all faces); returns optical-axis depth (inf where the ray misses)."""
    G = GB.X4[k]
    v0, v1, v2 = G[GB.F[:, 0]], G[GB.F[:, 1]], G[GB.F[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    o = V.pos[k]
    zc = V.R[k][2]                                     # optical axis (world)
    out = np.full(len(X), np.inf)
    for i, x in enumerate(X):
        dvec = x - o
        dvec = dvec / (dvec @ zc)                      # parametrise by optical-axis depth
        pv = np.cross(dvec, e2)
        det = (e1 * pv).sum(1)
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        tv = o - v0
        u = (tv * pv).sum(1) * inv
        qv = np.cross(tv, e1)
        v = (qv @ dvec) * inv
        t = (e2 * qv).sum(1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        if hit.any():
            out[i] = t[hit].min()
    return out


def snap_base_exact(V, GB, verts4d, off):
    """Final placement of the base vertices: on the gallbladder surface along their camera ray (exact ray cast),
    off in front of it; vertices whose ray misses the model are left where they are."""
    out = verts4d.copy()
    for k in range(len(verts4d)):
        q, z = V.project(verts4d[k, BASE], k)
        zh = ray_hit(V, GB, verts4d[k, BASE], k)
        hit = np.isfinite(zh)
        if hit.any():
            out[k, BASE[hit]] = V.unproject(q[hit, 0], q[hit, 1], zh[hit] - off, k)
    return out          # rays that miss (just outside the model's outline) keep the nearest-pixel placement


def cast_on_gb(V, GB, Bw, k, off, snap_px=40, exact=False):
    """Base points onto the gallbladder surface along the camera rays of frame k (image outline kept). A point whose
    ray misses the gallbladder model moves to the nearest pixel the model covers (within snap_px; the outline shifts
    by that much), else to the closest surface point in 3D. exact: depth from an exact ray-mesh intersection instead
    of the (conservative, up to ~1 mm in front) depth map."""
    from scipy.ndimage import distance_transform_edt
    q, z, zg = GB.depth_at(V, Bw, k)
    if exact:
        zg = ray_hit(V, GB, Bw, k)
    hit = np.isfinite(zg)
    X = Bw.copy()
    if hit.any():
        X[hit] = V.unproject(q[hit, 0], q[hit, 1], zg[hit] - off, k)
    if (~hit).any():
        D = GB.D[k].astype(np.float32)
        dist, (iy, ix) = distance_transform_edt(~np.isfinite(D), return_indices=True)
        for j in np.nonzero(~hit)[0]:
            x, y = int(np.clip(round(q[j, 0]), 0, V.W - 1)), int(np.clip(round(q[j, 1]), 0, V.H - 1))
            if dist[y, x] <= snap_px:
                yy, xx = iy[y, x], ix[y, x]
                zz = D[yy, xx]
                if exact:
                    ze = ray_hit(V, GB, V.unproject(xx, yy, zz, k)[None], k)[0]
                    zz = ze if np.isfinite(ze) else zz
                X[j] = V.unproject(xx, yy, zz - off, k)
            else:
                X[j] = GB.closest(Bw[j:j + 1], k)[0][0]
    return X, hit


def enforce_front(V, GB, verts4d, margin, sp_time=0.0, jaw_clear=None, taper=None):
    """Move sheet vertices forward along their camera ray wherever the gallbladder model is in front of them, to
    gallbladder depth - margin. The push is spread smoothly over the grid and over time (sp_time frames) but never
    below what each vertex needs. The jaw row stays (attached to the grasper); the base row is already on the surface."""
    from scipy.ndimage import gaussian_filter1d
    E = edges_of(FACES)
    N = verts4d.shape[1]
    Adj = np.zeros((N, N))
    Adj[E[:, 0], E[:, 1]] = 1
    Adj[E[:, 1], E[:, 0]] = 1
    deg = Adj.sum(1)
    frozen = (SV == 0) | (SV == 1)
    n = len(verts4d)
    need, qs, zs = np.zeros((n, N)), np.zeros((n, N, 2)), np.zeros((n, N))
    for k in range(n):
        qs[k], zs[k], zg = GB.depth_at(V, verts4d[k], k)
        m_v = margin if taper is None else taper[0] + (margin - taper[0]) * np.clip((1 - SV) / taper[1], 0, 1)
        need[k] = np.maximum(np.where(np.isfinite(zg), zs[k] - (zg - m_v), 0.0), 0)
    need[:, frozen] = 0

    def spread(push):
        for _ in range(30):
            push = np.maximum(need, 0.5 * push + 0.5 * (push @ Adj.T) / deg)
            push[:, frozen] = 0
        return push
    push = spread(need.copy())
    for _ in range(3 if sp_time else 0):
        push = spread(np.maximum(need, gaussian_filter1d(push, sp_time, axis=0, mode='nearest')))
    if jaw_clear is not None:
        # held jaw row: only where the gallbladder model is in front of it, move it forward to jaw_clear in front
        # (no spreading; smoothed in time but never below what is needed)
        jaw = SV == 0
        zg_all = np.stack([GB.depth_at(V, verts4d[k], k)[2] for k in range(n)])
        need_j = np.maximum(np.where(np.isfinite(zg_all), zs - (zg_all - jaw_clear), 0.0), 0)[:, jaw]
        pj = np.maximum(need_j, gaussian_filter1d(need_j, 1.5, axis=0, mode='nearest'))
        push[:, jaw] = pj
    out = np.stack([V.unproject(qs[k, :, 0], qs[k, :, 1], zs[k] - push[k], k) for k in range(n)])
    return out, push


def apex_ray_near_jaws(V, tcp, tip, apex2d, sigma=1.5, nu=61):
    """Apex = the point of the image apex's camera ray closest to the grasper's jaw segment (TCP -> jaw tip,
    interaction track): the depth comes from the instrument, the image position from the video. u = position of the
    nearest jaw-segment point (0 = TCP, 1 = tip). Smoothed in time (sigma frames)."""
    from scipy.ndimage import gaussian_filter1d
    us = np.linspace(0, 1, nu)
    A, u = np.zeros_like(tcp), np.zeros(len(tcp))
    for k in range(len(tcp)):
        o = V.pos[k]
        r = V.unproject(apex2d[k, 0], apex2d[k, 1], 1.0, k) - o
        r /= np.linalg.norm(r)
        P = tcp[k] + us[:, None] * (tip[k] - tcp[k])
        t = (P - o) @ r
        dist = np.linalg.norm(o + t[:, None] * r - P, axis=1)
        i = np.argmin(dist)
        A[k], u[k] = o + t[i] * r, us[i]
    return (gaussian_filter1d(A, sigma, axis=0, mode='nearest') if sigma else A), u


def apex_on_jaws(V, tcp, tip, apex2d, sigma=1.5, nu=31):
    """Apex = the point of the grasper's jaw segment (TCP -> jaw tip, interaction track) whose projection is nearest
    the image apex (where the sheet leaves the jaws); u in [0, 1] along the segment, smoothed in time."""
    from scipy.ndimage import gaussian_filter1d, median_filter
    us = np.linspace(0, 1, nu)
    u = np.zeros(len(tcp))
    for k in range(len(tcp)):
        P = tcp[k] + us[:, None] * (tip[k] - tcp[k])
        q, _ = V.project(P, k)
        u[k] = us[np.argmin(np.linalg.norm(q - apex2d[k], axis=1))]
    u = gaussian_filter1d(median_filter(u, 5, mode='nearest'), 2, mode='nearest')
    A = tcp + u[:, None] * (tip - tcp)
    return gaussian_filter1d(A, sigma, axis=0, mode='nearest') if sigma else A, u


def behind_stats(V, GB, verts4d, tol=0.0, sel=None):
    """Per frame: fraction of sheet vertices (or of the subset sel) behind the gallbladder model along their camera
    ray (z > z_hit + tol, z_hit = exact first intersection of the ray with the model)."""
    out = []
    for k in range(len(verts4d)):
        _, z = V.project(verts4d[k], k)
        zg = ray_hit(V, GB, verts4d[k], k)
        b = np.isfinite(zg) & (z > zg + tol)
        out.append(float((b if sel is None else b[sel]).mean()))
    return np.array(out)


def occlusion_stats(V, GB, verts4d, masks, ks, MVv, faces=None):
    """Keyframes, r2s.quality.render z-test of sheet vs gallbladder model (as in the round contact sheets):
    fraction of the sheet's pixels hidden by the gallbladder, and IoU of the still-visible sheet with our mask."""
    hid, iouv = [], []
    for k in ks:
        ms, zs = Q.render(V, verts4d, FACES if faces is None else faces, k)
        mg, zg = Q.render(V, GB.X4, GB.F, k)
        hidden = ms & mg & (zg < zs)
        hid.append(float(hidden.sum() / max(ms.sum(), 1)))
        iouv.append(iou(ms & ~hidden, masks[k], ~MVv.occluders(k)))
    return np.array(hid), np.array(iouv)


def min_angles(X, F=None):
    F = FACES if F is None else F
    P = X[..., F, :]
    ang = []
    for i in range(3):
        u, v = P[..., (i + 1) % 3, :] - P[..., i, :], P[..., (i + 2) % 3, :] - P[..., i, :]
        c = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-15)
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.min(ang, 0)


def rest_candidates(verts4d, step=5, lo=0):
    """For candidate rest frames r: 99th / 1st percentile over all frames and edges of edge length / rest length."""
    E = edges_of(FACES)
    L = np.linalg.norm(verts4d[:, E[:, 0]] - verts4d[:, E[:, 1]], axis=2)
    res = {}
    for r in range(lo, len(verts4d), step):
        ratio = L / np.maximum(L[r], 1e-9)
        res[r] = (float(np.percentile(ratio, 99)), float(np.percentile(ratio, 1)))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# v07: remeshed sheet (no slivers) and per-band metrics
def _in_poly(P, poly):
    """Even-odd point-in-polygon test, P (n, 2), poly (m, 2) closed implicitly."""
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    a, b = poly, np.roll(poly, -1, 0)
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    xint = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / np.where(b[:, 1] == a[:, 1], 1e-12, b[:, 1] - a[:, 1])
    return (cond & (x < xint)).sum(1) % 2 == 1


def _seg_dist(P, poly):
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((P[:, None] - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-18), 0, 1)
    return np.linalg.norm(P[:, None] - (a + t[..., None] * ab), axis=-1).min(1)


def design_mesh(A, B, h=0.0013, jaw_w=JAW_W, iters=20, seed=0):
    """Triangulate the sheet domain (s, t) in [0,1]^2 so that the triangles are well shaped on the ruled surface
    X(s, t) = (1-s) jawline(t) + s B(t) of the design frame. Boundary (jaw line, sides, base) sampled every h of 3D
    arc length; inside, Poisson-disk samples with spacing h measured on the surface; Delaunay in (s, t) rescaled by
    the mean generator length / mean width; then improve_mesh (3D tangential smoothing + 3D Delaunay edge flips).
    Returns st (N, 2) (boundary first) and faces (M, 3), counter-clockwise in (s, t)."""
    from scipy.spatial import Delaunay, cKDTree
    e = B[-1] - B[0]
    e /= np.linalg.norm(e)

    def X(s_, t_):
        s_, t_ = np.asarray(s_, float), np.asarray(t_, float)
        return (1 - s_)[:, None] * (A + (t_ - 0.5)[:, None] * jaw_w * e) + s_[:, None] * interp_curve(B, t_)

    def side(sfn, tfn):
        u = np.linspace(0, 1, 800)
        P = X(sfn(u), tfn(u))
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        m = max(1, int(round(L[-1] / h)))
        ui = np.interp(np.linspace(0, L[-1], m + 1), L, u)
        return np.c_[sfn(ui), tfn(ui)][:-1]
    def base_side():
        # every knot of the base polyline is a vertex (the outline follows the observed base exactly), each knot
        # interval subdivided to about h
        kn = np.linspace(0, 1, len(B))
        seg = np.linalg.norm(np.diff(B, axis=0), axis=1)
        tt = np.concatenate([np.linspace(kn[j], kn[j + 1], max(1, int(round(seg[j] / h))) + 1)[:-1] for j in range(len(B) - 1)] + [[1.0]])
        tt = tt[::-1][:-1]                      # right -> left, drop t = 0 (start of the left side)
        return np.c_[np.ones_like(tt), tt]
    bst = np.concatenate([side(lambda u: 0 * u, lambda u: u), side(lambda u: u, lambda u: 0 * u + 1),
                          base_side(), side(lambda u: 1 - u, lambda u: 0 * u)])
    # Poisson-disk interior samples on the surface
    sf, tf = np.meshgrid(np.linspace(0.005, 0.995, 161), np.linspace(0.002, 0.998, 321), indexing='ij')
    cst = np.c_[sf.ravel(), tf.ravel()]
    rng = np.random.default_rng(seed)
    cst = cst[rng.permutation(len(cst))]
    C3 = X(cst[:, 0], cst[:, 1])
    acc3 = list(X(bst[:, 0], bst[:, 1]))
    acc = []
    tree_pts = np.array(acc3)
    tree = cKDTree(tree_pts)
    pending = []
    for i, p3 in enumerate(C3):
        if tree.query(p3)[0] < h:
            continue
        if pending and np.min(np.linalg.norm(np.array(pending) - p3, axis=1)) < h:
            continue
        pending.append(p3)
        acc.append(cst[i])
        if len(pending) >= 64:
            tree_pts = np.concatenate([tree_pts, np.array(pending)])
            tree = cKDTree(tree_pts)
            pending = []
    st = np.concatenate([bst, np.array(acc).reshape(-1, 2)])
    P3 = X(st[:, 0], st[:, 1])
    Hm = np.mean(np.linalg.norm(B - (A + (np.linspace(0, 1, len(B)) - 0.5)[:, None] * jaw_w * e), axis=1))
    Wm = 0.5 * (jaw_w + np.linalg.norm(np.diff(B, axis=0), axis=1).sum())
    T = Delaunay(st * [Hm, Wm]).simplices
    a2 = np.cross(st[T[:, 1]] - st[T[:, 0]], st[T[:, 2]] - st[T[:, 0]])
    T = T[np.abs(a2) > 1e-9]
    T = _orient(st, T)
    st, T = improve_mesh(st, T, X, fixed=np.arange(len(st)) < len(bst), iters=iters)
    return st, T


def _orient(st, T):
    a = np.cross(st[T[:, 1]] - st[T[:, 0]], st[T[:, 2]] - st[T[:, 0]])
    T = T.copy()
    T[a < 0] = T[a < 0][:, [0, 2, 1]]
    return T


def _angles3(X, T):
    P = X[T]
    out = []
    for i in range(3):
        u, v = P[:, (i + 1) % 3] - P[:, i], P[:, (i + 2) % 3] - P[:, i]
        out.append(np.arccos(np.clip((u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-18), -1, 1)))
    return np.stack(out, 1)          # angle at each corner (rad)


def repair_slivers(st, T, X, fixed, min_deg=15.0, rounds=40):
    """Move the free vertices of triangles below min_deg (on the surface X) toward the (s, t) centroid of their
    neighbours, step by step, while no triangle inverts in (s, t) and the worst incident angle improves."""
    st = st.copy()
    nbr = [set() for _ in range(len(st))]
    inc = [[] for _ in range(len(st))]
    for f, (a, b, c) in enumerate(T):
        nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
        inc[a].append(f); inc[b].append(f); inc[c].append(f)
    for _ in range(rounds):
        P = X(st[:, 0], st[:, 1])
        ang = np.degrees(_angles3(P, T)).min(1)
        bad = np.nonzero(ang < min_deg)[0]
        if not len(bad):
            break
        moved = False
        for f in bad:
            for v in T[f]:
                if fixed[v]:
                    continue
                fs = np.array(inc[v])
                worst0 = np.degrees(_angles3(X(st[:, 0], st[:, 1]), T[fs])).min()
                target = st[list(nbr[v])].mean(0)
                for a in (1.0, 0.5, 0.25):
                    cand = st.copy()
                    cand[v] = st[v] + a * (target - st[v])
                    o = np.cross(cand[T[fs, 1]] - cand[T[fs, 0]], cand[T[fs, 2]] - cand[T[fs, 0]])
                    if (o <= 0).any():
                        continue
                    if np.degrees(_angles3(X(cand[:, 0], cand[:, 1]), T[fs])).min() > worst0 + 1e-6:
                        st, moved = cand, True
                        break
        if not moved:
            break
    return st


def improve_mesh(st, T, X, fixed, iters=20, flips=True, per_vertex=False):
    """Improve triangle angles on the surface X(s, t): tangential smoothing of the free vertices toward the 3D
    centroid of their neighbours (moved in (s, t) through the local Jacobian), then edge flips by the 3D Delaunay
    criterion (opposite angles > 180 deg), keeping every triangle positively oriented in (s, t)."""
    st = st.copy()
    T = T.copy()
    eps = 1e-4
    for it in range(iters):
        P = X(st[:, 0], st[:, 1])
        nbr = [set() for _ in range(len(st))]
        for a, b, c in T:
            nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
        free = np.nonzero(~fixed)[0]
        Js = (X(np.clip(st[free, 0] + eps, 0, 1), st[free, 1]) - X(np.clip(st[free, 0] - eps, 0, 1), st[free, 1])) / (2 * eps)
        Jt = (X(st[free, 0], np.clip(st[free, 1] + eps, 0, 1)) - X(st[free, 0], np.clip(st[free, 1] - eps, 0, 1))) / (2 * eps)
        new = st.copy()
        for n_, i in enumerate(free):
            d3 = P[list(nbr[i])].mean(0) - P[i]
            J = np.c_[Js[n_], Jt[n_]]
            new[i] = np.clip(st[i] + 0.7 * np.linalg.lstsq(J, d3, rcond=None)[0], 1e-4, 1 - 1e-4)
        # accept only if no triangle flips in (s, t) (per_vertex: undo just the moves around flipped triangles)
        a_new = np.cross(new[T[:, 1]] - new[T[:, 0]], new[T[:, 2]] - new[T[:, 0]])
        if per_vertex:
            for _ in range(10):
                bad = a_new <= 1e-12
                if not bad.any():
                    break
                vb = np.unique(T[bad])
                new[vb] = st[vb]
                a_new = np.cross(new[T[:, 1]] - new[T[:, 0]], new[T[:, 2]] - new[T[:, 0]])
            if (a_new > 0).all():
                st = new
        elif (a_new > 0).all():
            st = new
        # Delaunay edge flips in 3D
        for _ in range(50 if flips else 0):
            P = X(st[:, 0], st[:, 1])
            ang = _angles3(P, T)
            emap = {}
            for f, tri in enumerate(T):
                for i in range(3):
                    a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
                    emap.setdefault((min(a, b), max(a, b)), []).append((f, i))
            flipped, done = 0, set()
            for (a, b), lst in emap.items():
                if len(lst) != 2:
                    continue
                (f1, i1), (f2, i2) = lst
                if f1 in done or f2 in done or ang[f1, i1] + ang[f2, i2] <= np.pi + 1e-6:
                    continue
                c, dd = T[f1, i1], T[f2, i2]
                t1, t2 = np.array([c, dd, a]), np.array([dd, c, b])
                o1 = np.cross(st[t1[1]] - st[t1[0]], st[t1[2]] - st[t1[0]])
                o2 = np.cross(st[t2[1]] - st[t2[0]], st[t2[2]] - st[t2[0]])
                if o1 < 0:
                    t1 = t1[[0, 2, 1]]; o1 = -o1
                if o2 < 0:
                    t2 = t2[[0, 2, 1]]; o2 = -o2
                # the quad must stay convex in (s, t): the new triangles must not overlap (areas add up)
                tot = abs(np.cross(st[T[f1, 1]] - st[T[f1, 0]], st[T[f1, 2]] - st[T[f1, 0]])) + abs(np.cross(st[T[f2, 1]] - st[T[f2, 0]], st[T[f2, 2]] - st[T[f2, 0]]))
                if abs(o1 + o2 - tot) > 1e-9 * max(tot, 1e-12) + 1e-14 or min(o1, o2) < 1e-12:
                    continue
                T[f1], T[f2] = t1, t2
                done |= {f1, f2}
                flipped += 1
            T = _orient(st, T)
            if not flipped:
                break
    return st, T


def band_metrics(MVv, verts4d, faces, bands=((0, 62), (62, 130), (130, 251)), step=1):
    """IoU and boundary F (4 px) vs our mask on every `step`-th frame, summarised per frame band."""
    ks = list(range(0, MVv.n, step))
    iou_ = Q.silhouette_iou(MVv, verts4d, faces, 'membrane', ks)
    bf = Q.boundary_f(MVv, verts4d, faces, 'membrane', ks, tol=4)
    out = {}
    for a, b in bands:
        sel = [i for i, k in enumerate(ks) if a <= k < b]
        out[f'{a}-{b - 1}'] = dict(iou=round(float(np.mean(iou_[sel])), 4), bf4=round(float(np.mean(bf[sel])), 4),
                                   iou_min=round(float(np.min(iou_[sel])), 4), n=len(sel))
    out['all'] = dict(iou=round(float(iou_.mean()), 4), bf4=round(float(bf.mean()), 4), n=len(ks))
    return out, iou_, bf


def plot_series(path, curves, n, title=''):
    """Tiny cv2 line chart: curves = [(label, y (n,), (r, g, b)), ...] each scaled to its own min..max (printed)."""
    Wd, Hh, pad = 900, 90, 4
    img = np.full((Hh * len(curves) + 24, Wd, 3), 255, np.uint8)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for i, (lab, y, col) in enumerate(curves):
        y = np.asarray(y, float)
        lo, hi = np.nanmin(y), np.nanmax(y)
        y0 = 24 + i * Hh
        cv2.rectangle(img, (0, y0), (Wd - 1, y0 + Hh - 1), (220, 220, 220), 1)
        for k in range(0, n, 25):
            x = int(k / (n - 1) * (Wd - 1))
            cv2.line(img, (x, y0), (x, y0 + Hh - 1), (235, 235, 235), 1)
        pts = np.c_[np.arange(len(y)) / (n - 1) * (Wd - 1), y0 + Hh - pad - (y - lo) / max(hi - lo, 1e-9) * (Hh - 2 * pad)]
        ok = np.isfinite(pts[:, 1])
        cv2.polylines(img, [pts[ok].astype(np.int32)], False, tuple(int(c) for c in col), 1, cv2.LINE_AA)
        cv2.putText(img, f'{lab}  [{lo:.3g} .. {hi:.3g}]', (6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(img, 'frames 0..250, grid every 25', (Wd - 230, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def gb_report(V, GB, cfg, O, MVv, verts4d, pre_front, extra, As, ks, series, rest_c, k_rest, d):
    S = Q.summarize
    pz = np.load(ROOT / cfg['compare'] / 'model.npz')
    prev, pF = pz['verts4d'].astype(float), pz['faces']
    p_top = pz['attach_idx'][pz['attach_to'] == 'grasper']
    p_inner = np.ones(prev.shape[1], bool)
    p_inner[pz['attach_idx'][pz['attach_to'] == 'gallbladder']] = False
    rep = dict(source=GB.src, grasp=cfg['grasp_src'] + ':' + cfg['apex_from'], compare=cfg['compare'],
               margin_mm=cfg['margin'] * 1000, base_offset_mm=cfg['base_off'] * 1000)
    beh = {}
    for lab, X in [(cfg['compare'], prev), ('before_front_fix', pre_front), ('this', verts4d)]:
        b0, b1 = behind_stats(V, GB, X), behind_stats(V, GB, X, tol=0.001)
        beh[lab] = (b0, b1)
        rep[f'vertices_behind_gb_frac_{lab}'] = dict(note='fraction of sheet vertices behind the GB surface along the ray, per frame',
                                                   frames_with_any=int((b0 > 0).sum()), **S(b0))
        rep[f'vertices_behind_gb_by_1mm_frac_{lab}'] = S(b1)
        inner = p_inner if lab == cfg['compare'] else SV < 1
        rep[f'non_base_vertices_behind_gb_frac_{lab}'] = dict(note='same, without the base row (which lies on the surface)',
                                                            **S(behind_stats(V, GB, X, sel=inner)))
    for lab, X, F_ in [(cfg['compare'], prev, pF), ('this', verts4d, FACES)]:
        hid, iv = occlusion_stats(V, GB, X, O['masks'], ks, MVv, F_)
        rep[f'sheet_pixels_hidden_by_gb_{lab}'] = dict(note='keyframes, quality.render z-test', per_keyframe=[round(float(v), 3) for v in hid], **S(hid))
        rep[f'visible_sheet_iou_{lab}'] = dict(note='IoU with our mask of the sheet part not hidden by the GB model', **S(iv))
    apex_prev = prev[:, p_top].mean(1)
    apex_now = verts4d[:, TOP].mean(1)
    GP = extra['grasp_point']
    dprev = np.linalg.norm(apex_prev - GP, axis=1) * 1000
    dnow = np.linalg.norm(apex_now - GP, axis=1) * 1000
    dz = np.array([V.project(GP[k:k + 1], k)[1][0] - V.project(extra['apex_image3d'][k:k + 1], k)[1][0] for k in range(V.n)]) * 1000
    if 'jaw_u' in extra:
        rep['apex_jaw_u'] = dict(note='nearest jaw-segment point to the apex, 0 = grasp point (TCP), 1 = jaw tip (3 mm)', **S(extra['jaw_u']))
        Igr = np.load(ROOT.parent / cfg['grasp_src'] / 'model.npz')
        tcp, tip = Igr['grasper_left_tcp'], Igr['grasper_left_tip']
        us = np.linspace(0, 1, 61)
        dseg = np.array([np.min(np.linalg.norm(tcp[k] + us[:, None] * (tip[k] - tcp[k]) - apex_now[k], axis=1)) for k in range(V.n)]) * 1000
        rep['apex_to_jaw_segment_mm_after'] = dict(note='jaw-row centre to the nearest point of the TCP -> tip segment', **S(dseg))
        dseg0 = np.array([np.min(np.linalg.norm(tcp[k] + us[:, None] * (tip[k] - tcp[k]) - apex_prev[k], axis=1)) for k in range(V.n)]) * 1000
        rep['apex_to_jaw_segment_mm_before'] = S(dseg0)
        series.update(apex_seg_prev=dseg0, apex_seg_now=dseg)
    rep['apex_to_grasp_point_mm_before'] = dict(note=f'{cfg["compare"]} jaw-row centre to the grasp point, per frame', **S(dprev))
    rep['apex_to_grasp_point_mm_after'] = S(dnow)
    rep['grasp_point_minus_image_apex_depth_mm'] = dict(note='>0: grasp point farther than our image-depth apex', **S(dz))
    gpx = np.array([V.project(GP[k:k + 1], k)[0][0] for k in range(V.n)])
    rep['grasp_point_to_image_apex_px'] = S(np.linalg.norm(gpx - O['apex2d'], axis=1))
    apx = np.array([V.project(apex_now[k:k + 1], k)[0][0] for k in range(V.n)])
    rep['apex_to_image_apex_px_after'] = S(np.linalg.norm(apx - O['apex2d'], axis=1))
    bd = np.array([np.median(GB.closest(verts4d[k, BASE], k)[1]) for k in range(V.n)]) * 1000
    bdx = np.array([np.max(GB.closest(verts4d[k, BASE], k)[1]) for k in range(V.n)]) * 1000
    rep['base_to_gb_surface_mm'] = dict(note='median (max) over the 25 base vertices, per frame; ~0.3 mm = dense-sample resolution', median=S(bd), max=S(bdx))
    rep['base_ray_hits_gb_frac'] = dict(note='base vertices placed along their ray (rest snapped to the closest surface point)', **S(extra['base_hit'].mean(1)))
    pu = extra['front_push'] * 1000
    rep['front_push_mm'] = dict(note='forward move along the ray to clear the GB, per frame max', **S(pu.max(1)),
                                frac_vertices_moved=S((pu > 1e-3).mean(1)))
    if rest_c is not None:
        sel = {r: rest_c[r] for r in sorted(set([k_rest, 0, int(np.argmin(tri_area(verts4d))), int(np.argmax(tri_area(verts4d))) // 5 * 5]) & set(rest_c))}
        rep['rest_choice'] = dict(note='edge length / rest length over all frames: (p99, p1) for candidate rest frames; chosen = min max(p99, 1/p1)',
                                  chosen=int(k_rest), candidates={str(r): [round(a, 2), round(b, 2)] for r, (a, b) in sel.items()})
    series.update(behind_prev=beh[cfg['compare']][0], behind_pre=beh['before_front_fix'][0], behind_now=beh['this'][0],
                  apex_gp_prev=dprev, apex_gp_now=dnow)
    Q.contact_sheet(V, [('gallbladder ' + GB.src, GB.X4, GB.F, (230, 200, 40)), ('membrane ' + cfg['compare'], prev, pF, (0, 255, 255))],
                    ks[::3], d / 'work/sheet_with_gb_before.jpg')
    Q.contact_sheet(V, [('gallbladder ' + GB.src, GB.X4, GB.F, (230, 200, 40)), ('membrane this', verts4d, FACES, (0, 255, 255))],
                    ks[::3], d / 'work/sheet_with_gb_after.jpg')
    return rep


def main(argv):
    version = argv[0]
    d = vdir(version)
    V = views.load(CLIP, 'sift')
    if '--seg' in argv:
        M = load_or_segment(V, d, CONFIGS.get(version, {}).get('prompts', PROMPTS), force=True)
        overlay_masks(V, M, list(range(0, V.n, 10)), d / 'work/masks_overlay.jpg', CONFIGS.get(version, {}).get('prompts', PROMPTS))
        print('mask area per keyframe', [int(M[k].sum()) for k in range(0, V.n, 10)])
        return
    cfg = CONFIGS[version]
    V_clip = V
    if cfg.get('cams'):
        # same as views.load(CLIP, 'sift', cams=...) (only the rotations change), sharing the big arrays
        import copy
        V = copy.copy(V_clip)
        V.R = np.load(ROOT.parents[3] / cfg['cams'])['R'][:V.n].astype(float)
        V.cams = cfg['cams']
    if isinstance(cfg.get('gb_src'), (tuple, list)):
        cfg = dict(cfg, gb_src=next(g for g in cfg['gb_src'] if (ROOT.parent / g / 'model.npz').exists()))
        cfg['xcheck'] = dict(cfg['xcheck'], gallbladder=cfg['gb_src'])
    set_grid(*cfg.get('grid', (16, 12)))
    M = load_or_segment(V, d, mask_from=cfg.get('mask_from'))
    O = observe(V, M, prune=cfg['prune'], pct=cfg.get('pct', 97))
    mf = np.load(d / 'masks.npz')
    if 'membrane_clean' not in mf.files:
        np.savez_compressed(d / 'masks.npz', membrane=mf['membrane'], prompts=mf['prompts'], membrane_clean=O['masks'])
    A, B, P = fit_frames(V, O, p_fit=cfg['p_fit'], resample=cfg['resample'])
    GB, extra = None, {}
    if cfg.get('gb_src'):
        # apex = the interaction track's grasp point (grasper TCP between the jaws); base on the gallbladder model
        GB = GBSurface(V, cfg['gb_src'])
        Igr = np.load(ROOT.parent / cfg['grasp_src'] / 'model.npz')
        extra['apex_image3d'] = A.copy()
        extra['grasp_point'] = Igr['grasp_point'].astype(float)
        if cfg['apex_from'] == 'ray_near_jaws':
            A, extra['jaw_u'] = apex_ray_near_jaws(V, Igr['grasper_left_tcp'], Igr['grasper_left_tip'], O['apex2d'],
                                                   cfg.get('sa_apex', 1.5))
        elif cfg['apex_from'] == 'jaw_segment':
            A, extra['jaw_u'] = apex_on_jaws(V, Igr['grasper_left_tcp'], Igr['grasper_left_tip'], O['apex2d'],
                                             cfg.get('sa_apex', 1.5))
        else:
            A = Igr[cfg['apex_from']].astype(float)
        ex = cfg.get('exact_cast', False)
        B = np.stack([cast_on_gb(V, GB, B[k], k, cfg['base_off'], exact=ex)[0] for k in range(V.n)])
        _, Bs, Ps = smooth(A, B, P, sa=cfg['sa'], sb=cfg['sb'])
        As = A.copy()
        if cfg.get('pan_light'):
            # lighter temporal smoothing of the base during the fast pan (frames < k0, ramp to k1): world-space
            # smoothing across a fast-moving scope mixes views and costs ~0.05 IoU there (scratch test)
            pl = cfg['pan_light']
            from scipy.ndimage import median_filter, gaussian_filter1d
            Bl = gaussian_filter1d(median_filter(B, (pl['med'], 1, 1), mode='nearest'), pl['sb'], axis=0, mode='nearest')
            w = np.clip((pl['k1'] - np.arange(V.n)) / (pl['k1'] - pl['k0']), 0, 1)[:, None, None]
            Bs = w * Bl + (1 - w) * Bs
        cast = [cast_on_gb(V, GB, Bs[k], k, cfg['base_off'], exact=ex) for k in range(V.n)]
        Bs = np.stack([c[0] for c in cast])
        if cfg.get('sb_after'):
            from scipy.ndimage import gaussian_filter1d
            Bs = gaussian_filter1d(Bs, cfg['sb_after'], axis=0, mode='nearest')
        extra['base_hit'] = np.stack([c[1] for c in cast])
    else:
        As, Bs, Ps = smooth(A, B, P, sa=cfg['sa'], sb=cfg['sb'])
    verts4d = np.stack([tent(As[k], Bs[k], Ps[k]) for k in range(V.n)])
    k_design = None
    if cfg.get('remesh'):
        # design frame = the balanced rest frame of the ruled grid sheet; triangulate the sheet domain there
        rc = rest_candidates(verts4d, lo=cfg.get('rest_from', 0))
        k_design = min(rc, key=lambda r: max(rc[r][0], 1 / rc[r][1]))
        st, T = design_mesh(As[k_design], Bs[k_design], **cfg['remesh'])
        set_mesh(st, T)
        extra['st'] = st
        verts4d = np.stack([tent(As[k], Bs[k], Ps[k]) for k in range(V.n)])
        # (base vertices interpolate the base polyline, which lies on the gallbladder surface; chord error ~0.01 mm)
    elif cfg.get('mesh_from'):
        zm = np.load(ROOT / cfg['mesh_from'] / 'model.npz')
        set_mesh(zm['st'], zm['faces'])
        extra['st'] = zm['st']
        k_design = cfg.get('rest_frame')
        verts4d = np.stack([tent(As[k], Bs[k], Ps[k]) for k in range(V.n)])
    if cfg.get('tongue_s0') and GB is not None:
        # held jaw row at the grasper TCP; the offset TCP - apex fades out quadratically up to s = tongue_s0, so the
        # sheet beyond that is unchanged (it leaves the jaws where the video shows it)
        tcp = Igr['grasper_left_tcp'].astype(float)
        extra['apex_image_ray'] = As.copy()
        wv = np.clip(1 - SV / cfg['tongue_s0'], 0, 1) ** 2
        verts4d = verts4d + wv[None, :, None] * (tcp - As)[:, None, :]
        As = tcp.copy()
    deltas = np.zeros(verts4d.shape[:2])
    if cfg.get('depth_refine'):
        verts4d, deltas = refine_depth(V, verts4d, O['masks'], cfg['lam'], cfg['w_upper'], cfg['sd'])
    if GB is not None:
        pre_front = verts4d.copy()
        verts4d, pushes = enforce_front(V, GB, verts4d, cfg['margin'], cfg.get('sp_time', 0), cfg.get('jaw_clear'),
                                            (cfg['base_off'], cfg['margin_taper']) if cfg.get('margin_taper') else None)
        extra['front_push'] = pushes.astype(np.float32)
        if cfg.get('base_exact_smooth'):
            # base depth corrected to the exact ray-mesh intersection (the depth map is conservative by up to ~1-2 mm
            # on slanted surface); the correction is smoothed over time so hit / miss changes do not jerk
            from scipy.ndimage import gaussian_filter1d
            corr = np.zeros((V.n, len(BASE)))
            qz = []
            for k in range(V.n):
                q, z = V.project(verts4d[k, BASE], k)
                zh = ray_hit(V, GB, verts4d[k, BASE], k)
                corr[k] = np.where(np.isfinite(zh), (zh - cfg['base_off']) - z, 0.0)
                qz.append((q, z))
            corr = gaussian_filter1d(corr, cfg['base_exact_smooth'], axis=0, mode='nearest')
            for k, (q, z) in enumerate(qz):
                verts4d[k, BASE] = V.unproject(q[:, 0], q[:, 1], z + corr[k], k)
            extra['base_depth_corr'] = corr.astype(np.float32)
    area = tri_area(verts4d)
    rest_c = None
    if cfg.get('rest') == 'balanced':
        rest_c = rest_candidates(verts4d, lo=cfg.get('rest_from', 0))
        k_rest = min(rest_c, key=lambda r: max(rest_c[r][0], 1 / rest_c[r][1]))
        if k_design is not None:
            k_rest = k_design            # the triangulation was designed on that frame
    elif cfg.get('rest') == 'median_area':
        k_rest = int(np.argsort(area)[len(area) // 2])
    else:
        k_rest = int(np.argmin(area))
    if cfg.get('rest') == 'fixed':
        k_rest = int(cfg['rest_frame'])
        rest_c = rest_candidates(verts4d, lo=k_rest % 5)
    if (cfg.get('remesh') and cfg.get('polish', True)) or (cfg.get('mesh_from') and cfg.get('polish')):
        # improve the triangles on the final rest surface (with depth offsets and front push), then resample every
        # frame at the new (s, t) by barycentric interpolation in the old triangulation and clear the gallbladder again
        from scipy.spatial import Delaunay
        st_old = np.c_[SV, TV]
        tri = Delaunay(st_old)
        def bary(q):
            si = tri.find_simplex(np.clip(q, 0, 1))
            Tm = tri.transform[si]
            b = np.einsum('nij,nj->ni', Tm[:, :2], np.clip(q, 0, 1) - Tm[:, 2])
            return tri.simplices[si], np.c_[b, 1 - b.sum(1)]
        Xr = verts4d[k_rest]
        def Xfun(s_, t_):
            idx, w = bary(np.c_[s_, t_])
            return np.einsum('nk,nkj->nj', w, Xr[idx])
        fixed_v = (SV == 0) | (SV == 1) | (TV == 0) | (TV == 1)
        st_new, T_new = improve_mesh(st_old, FACES, Xfun, fixed=fixed_v, iters=30,
                                     flips=cfg.get('polish_flips', True), per_vertex=cfg.get('polish_per_vertex', False))
        if cfg.get('sliver_repair'):
            st_new = repair_slivers(st_new, T_new, Xfun, fixed_v, min_deg=cfg['sliver_repair'])
        idx, w = bary(st_new)
        verts4d = np.einsum('nk,fnkj->fnj', w, verts4d[:, idx])
        deltas = np.einsum('nk,fnk->fn', w, deltas[:, idx])
        for key in ('front_push',):
            if key in extra:
                extra[key] = np.einsum('nk,fnk->fn', w, extra[key][:, idx]).astype(np.float32)
        set_mesh(st_new, T_new)
        extra['st'] = st_new
        if GB is not None:
            verts4d, push2 = enforce_front(V, GB, verts4d, cfg['margin'], cfg.get('sp_time', 0), cfg.get('jaw_clear'),
                                            (cfg['base_off'], cfg['margin_taper']) if cfg.get('margin_taper') else None)
            extra['front_push'] = (extra['front_push'] + push2).astype(np.float32)
    rest = verts4d[k_rest].copy()
    if GB is not None:
        # where each base vertex sits on the gallbladder model at the rest frame (face of GB faces + barycentric),
        # for attaching by position
        Sg = GB.samples(k_rest)
        from scipy.spatial import cKDTree
        dg, ig = cKDTree(Sg).query(rest[BASE])
        extra['attach_gb_face'] = GB.fi[ig].astype(np.int32)
        extra['attach_gb_bary'] = GB.bary[ig].astype(np.float32)
        extra['attach_gb_dist_mm'] = (dg * 1000).astype(np.float32)
    ks = V.keyframes
    MVv = MV(V, O['masks'], O['apex2d'])
    out, series = evaluate(V, MVv, O, verts4d, rest, ks, xcheck=cfg.get('xcheck'))
    out['about'] = ('Against our SAM mask of the sheet (masks.npz membrane_clean: closed/opened, largest component, '
                    'instrument part above the jaws pruned). Grasper/probe pixels and a 10 px disc at the jaws are unknown '
                    '(ignored). silhouette_iou / boundary_f: r2s.quality on keyframes (every 10th frame); boundary F '
                    'tolerance 4 and 8 px. depth residual: median |z_mesh - z_video| (the depth refinement pulls the sheet '
                    'toward the video depth, so this is partly a fit residual, not an independent check). stretch: edge '
                    'length / rest edge length. crosscheck: against other tracks\' outputs (read-only, version given).')
    out['rest_frame'] = k_rest
    out['profile_p'] = Q.summarize(Ps)
    top, base = TOP, BASE
    attach_idx = np.concatenate([top, base])
    attach_to = np.array(['grasper'] * len(top) + ['gallbladder'] * len(base))
    np.savez_compressed(d / 'model.npz', rest_verts=rest, faces=FACES, verts4d=verts4d.astype(np.float32),
                        attach_idx=attach_idx, attach_to=attach_to,
                        **({'grid_shape': np.array([NS + 1, NT + 1])} if MESH_KIND == 'grid' else {}),
                        apex=As, base=Bs, profile_p=Ps, apex_raw=A, base_raw=B, apex2d=O['apex2d'], depth_offset=deltas.astype(np.float32), **extra)
    if GB is not None:
        out['gallbladder_consistency'] = gb_report(V, GB, cfg, O, MVv, verts4d, pre_front, extra, As, ks, series, rest_c,
                                                   k_rest, d)
    out['min_angle_deg'] = dict(note='smallest triangle angle (MuJoCo shells dislike slivers)',
                                rest=round(float(min_angles(rest).min()), 2),
                                p1_over_time=round(float(np.percentile(min_angles(verts4d), 1)), 2),
                                min_over_time=round(float(min_angles(verts4d).min()), 2),
                                faces_below_15deg_rest=int((min_angles(rest) < 15).sum()),
                                faces_below_15deg_over_time_mean=round(float((min_angles(verts4d) < 15).sum(1).mean()), 2),
                                n_faces=int(len(FACES)), mesh=MESH_KIND)
    out['cameras'] = str(getattr(V, 'cams', 'clip'))
    if cfg.get('band_compare', True) and cfg.get('cams'):
        # IoU / BF per frame band on every frame: this version and the previous one, with the refined and clip cameras
        bands = {}
        pz = np.load(ROOT / cfg['compare'] / 'model.npz')
        bands['this_refined_cams'], i_this, b_this = band_metrics(MVv, verts4d, FACES)
        bands[cfg['compare'] + '_refined_cams'], _, _ = band_metrics(MVv, pz['verts4d'], pz['faces'])
        bands[cfg['compare'] + '_clip_cams'], _, _ = band_metrics(MV(V_clip, O['masks'], O['apex2d']), pz['verts4d'], pz['faces'])
        out['bands'] = dict(note='mean IoU / boundary F (4 px) vs our mask over every frame of each band', **bands)
        series.update(iou_every_frame=i_this, bf4_every_frame=b_this)
    json.dump(out, open(d / 'quality.json', 'w'), indent=1)
    Q.contact_sheet(V, [('membrane', verts4d, FACES, (0, 255, 255))], ks, d / 'sheet.jpg')
    views3d(V, rest, d / 'views3d.jpg')
    mask_sheet(V, O, verts4d, ks, d / 'work/mask_vs_mesh.jpg', series)
    # references: per-frame fit before temporal smoothing (upper bound of a smooth model) and the mask's own
    # frame-to-frame consistency; IoU of every frame (fast silhouette, instruments ignored)
    iou_all = np.array([iou(fast_sil(V, verts4d[k], k), O['masks'][k], ~MVv.occluders(k)) for k in range(V.n)])
    raw4d = np.stack([tent(A[k], B[k], P[k]) for k in range(V.n)])
    iou_raw = np.array([iou(fast_sil(V, raw4d[k], k), O['masks'][k], ~MVv.occluders(k)) for k in range(V.n)])
    self1 = np.array([iou(O['masks'][k], O['masks'][k + 1]) for k in range(V.n - 1)])
    out['reference'] = dict(
        iou_all_frames=dict(note='this model, every frame (fast silhouette, no visibility cut)', **Q.summarize(iou_all)),
        iou_all_frames_pan_0_62=Q.summarize(iou_all[:63]), iou_all_frames_63_250=Q.summarize(iou_all[63:]),
        iou_per_frame_fit_unsmoothed=dict(note='same template fitted per frame, no temporal smoothing', **Q.summarize(iou_raw)),
        mask_self_iou_adjacent=dict(note='IoU of our mask between consecutive frames (flicker + motion)', **Q.summarize(self1)))
    json.dump(out, open(d / 'quality.json', 'w'), indent=1)
    series.update(iou_all=iou_all, iou_raw=iou_raw)
    np.savez_compressed(d / 'work/series.npz', **series)
    plot_series(d / 'work/timeseries.jpg', [
        ('IoU every frame (smoothed model)', iou_all, (0, 120, 200)), ('IoU per-frame fit (unsmoothed)', iou_raw, (120, 120, 120)),
        ('sheet area cm2', series['area'], (0, 150, 0)), ('stretch max', series['smax'], (200, 0, 0)),
        ('apex vs image jaw px', series['apex_img_err'], (150, 0, 150)), ('V.tools tip vs image jaw px', series['tip_img_err'], (200, 100, 0)),
        ('apex - V.tools tip mm (3D)', series['apex_tip_3d'] * 1000, (0, 0, 0)), ('lift mm (apex-base centroid)', series['lift'], (0, 150, 150))],
        V.n, f'membrane {version}')
    for key, v in out['reference'].items():
        print(f'{key:34s} mean {v.get("mean")}  min {v.get("min")}')
    for key in ['silhouette_iou', 'silhouette_iou_raw', 'boundary_f_4px', 'boundary_f_8px', 'depth_residual_mm',
                'depth_residual_lower_mm', 'depth_residual_upper_mm', 'video_minus_mesh_depth_upper_mm',
                'base_on_gallbladder_mm', 'apex_to_image_jaw_px', 'grasper_tip_to_image_jaw_px',
                'apex_to_grasper_tip_mm', 'area_cm2', 'lift_mm', 'stretch_max', 'stretch_p95', 'stretch_p05',
                'base_on_gallbladder_fraction', 'accel_mm_per_frame2']:
        v = out[key]
        print(f'{key:34s} mean {v.get("mean")}  median {v.get("median")}  min {v.get("min")}  max {v.get("max")}')
    for kk, v in out['crosscheck'].items():
        print(f'{kk:34s}', {a: b for a, b in v.items() if a in ('mean', 'median', 'max', 'error')})
    print('mesh', out['mesh_health'], 'rest frame', k_rest, 'min angle', out['min_angle_deg'])
    if 'bands' in out:
        for kk, v in out['bands'].items():
            print('band', kk, v)
    if 'gallbladder_consistency' in out:
        print(json.dumps(out['gallbladder_consistency'], indent=0)[:4000])
    print('IoU per keyframe', out['silhouette_iou']['per_keyframe'])


if __name__ == '__main__':
    main(sys.argv[1:])
