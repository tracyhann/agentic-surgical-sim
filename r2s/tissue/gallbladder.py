"""Gallbladder (body + neck) of shot A: BodyParts3D template fitted to the multi-view observations (rest shape), then
deformed in every frame (4D). Contract: outputs/iter/CONTRACT.md; records: outputs/iter/tissues/gallbladder/vNN/NOTES.md.

    python -m r2s.tissue.gallbladder v01

Pipeline
  1. masks     body+neck mask per frame = the clip's SAM gallbladder mask minus the instruments, the pink strands /
               cystic duct (`strand`) and the apex of the translucent peritoneal tent the grasper lifts (SAM 2.1 object
               prompted just below the grasper jaws, kept within APEX_PX of the jaw tip). Those pixels are *unknown*
               (the organ may or may not be behind them), never background.
  2. rest      template (principal frame, metres) -> rotation + translation + anisotropic scale fitted jointly to the
               silhouettes and depth points of many keyframes through their own cameras (torch, 16 starts), then a
               4x4x4 free-form lattice (smoothness + size regularised), again on all those keyframes.
  3. tets      remesh + TetGen of the rest surface (r2s.organ.remesh / TetGen worker), no degenerate tets.
  4. 4D        every frame: node positions of the tet mesh fitted to that frame's silhouette (outside + coverage) and
               depth, with tet ARAP + volume terms against the rest shape and a temporal term to the previous frame;
               frames run outward from a well-observed start frame; a light temporal Gaussian at the end.
  5. quality   r2s.quality metrics against the shared SAM mask and against the body+neck mask here, plus temporal
               smoothness, volume, ARAP strain, template distance, multi-view consistency of the rest shape.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import organ, quality, views

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iter/tissues/gallbladder'
torch.set_num_threads(4)                     # other tracks share the machine

# ---------------------------------------------------------------- configuration per version
BASE = dict(
    apex_px=60,                 # tent apex: sheet pixels within this distance of the grasper jaw tip are unknown
    rest_keys=list(range(60, 251, 10)),
    rest_iters=120, rest_ffd_iters=200, ffd_smooth=3.0, ffd_size=0.3,
    scale_prior=0.02,           # weight of (log scale)^2 (mm-equivalent)
    scale_bounds=None,          # (lo, hi) plausible scale per template axis, enforced by a stiff hinge
    w_sil=2.0, w_cov=1.0, w_depth=1.0,
    hidden_mm=8.0,              # a surface point this far behind the measured depth is hidden by what is in front
    surf_faces=1000, cells=20,  # tet mesh resolution
    start_frame=120, iters_first=300, iters=80, lr=0.08,
    w_arap=20.0, w_vol=20.0, w_temp=0.1, smooth_sigma=1.0,
    arap_uniform=False, w_barrier=0.0, w_anchor=0.0, anchor_cos=0.35, early_weight=1.0, tet_snap='organ',
    w_neck=0.0, w_neck4d=0.0,   # neck landmark: template neck tip projects onto the body/duct junction pixel
    depth_erode=7, n_pts4d=800, # observed depth points: erosion of the body mask (px), number per frame in 4D
    barrier_sum=False,          # inversion barrier summed over tets (v02-v04: averaged, i.e. ~1/6000 per tet)
    tetgen_q=None,              # (mindihedral, minratio): own TetGen call + sliver peeling / interior smoothing
    sliver_weight=False,        # ARAP / volume / barrier weight per tet min(1, quality / 0.02); 0 below this quality
    n_surf4d=4000, n_cov4d=1200,  # surface samples / covered body pixels per frame in 4D
    tet_mode='v06',             # tet clean-up generation (see tetrahedralize)
    rest_from=None,             # reuse rest shape + tets of an earlier version (skip steps 2-3)
    cams=None,                  # camera file for views.load (None = the clip's SIFT cameras)
    membrane=None, w_sheet=0.0, sheet_margin=0.3,   # keep the surface >= margin (mm) behind this sheet's verts4d
    duct_masks=None,            # npz with 'duct' / 'strands' masks: not gallbladder (negative evidence)
    neck_target=None,           # npz with duct_centerline4d: neck tip pinned in 3D to the duct's proximal end
    neck_cap_mm=5.0, neck_nodes_mm=5.0, bcc_nodes=1330, band_ref='v10', mujoco_check=False,
    w_gvol=0.0, gvol_tol=0.005, # total volume within (1 +- tol) of rest; beyond: w * ((|V/V0 - 1| - tol) / 0.01)^2
)
CFG = {
    'v01': dict(),
    # tets without tiny edges, tet ARAP weighted per tet (not per volume) + inversion barrier, back side anchored to the
    # bed, rest fit on all keyframes (early pan frames at half weight)
    'v02': dict(arap_uniform=True, w_barrier=100.0, w_arap=50.0, w_anchor=0.05, surf_faces=650, cells=18,
                rest_keys=list(range(0, 251, 10)), early_weight=0.3, tet_snap='project',
                scale_bounds=(0.85, 1.7), ffd_size=2.0),
    # v02 put the template's neck out of view on the left: anatomical landmark (neck tip -> body/duct junction);
    # softer ARAP and bed anchor (v02 lost 0.07 IoU)
    'v03': dict(w_neck=1.0, w_neck4d=0.3, w_arap=30.0, w_anchor=0.02),
    # v03 lost pixels to the visible-only test (mesh > 6 mm behind the measured depth near the outline, where the
    # erosion left no depth points): depth points up to 1 px from the outline, more of them, depth weight 1.5;
    # smoother lattice (v03 rest had a notch), stronger temporal term (speed peaks 4 mm/frame at 99-100, 160-162)
    'v04': dict(depth_erode=3, n_pts4d=1500, w_depth=1.5, ffd_smooth=10.0, w_temp=0.3, smooth_sigma=1.5, iters=100),
    # v04: 15 inverted tets in 13/51 sampled frames (the averaged barrier is negligible per tet) -> summed barrier;
    # rest came out 5.3 x 4.9 x 4.2 cm (too round for a gallbladder, length scale 0.88) -> length scale >= 1.0
    'v05': dict(barrier_sum=True, w_barrier=0.5, scale_bounds=((1.0, 0.85, 0.85), (1.7, 1.7, 1.7))),
    # v05 exploded around slivers (det F of a 1e-4 mm^3 tet is hypersensitive; the summed barrier drove it): TetGen
    # mindihedral 15 / minratio 1.6, peel boundary sliver caps, smooth interior nodes, down-weight remaining slivers
    'v06': dict(tetgen_q=(15, 1.6), surf_faces=500, cells=16, sliver_weight=1e-9),
    # v06 mesh still climbs into the tent up to the grasper jaw (only 60 px of the tent apex were unknown): 90 px;
    # boundary F 0.31-0.36: denser surface samples / coverage pixels (spacing ~5 px -> ~3.5 px), Gaussian sigma 1.0
    # (first v07 run exploded at frame 150: a 7e-5 sliver; tets below quality 0.003 now carry no energy)
    'v07': dict(apex_px=90, n_surf4d=7000, n_cov4d=1800, smooth_sigma=1.0, sliver_weight=0.003),
    # v07 failed: TetGen filled spots around short surface edges (decimated marching cubes, down to 7 % of the median
    # edge) with 0.2 mm tets that inverted (359 well-shaped tets). Same config; surface edges < 45 % of the median are
    # collapsed before TetGen and the tet mesh with the largest min/median edge ratio (>= 0.1 accepted) is used.
    'v08': dict(tet_mode='v08'),
    # v08 tets are clean but coarse (646 surface faces, fit IoU -0.03) and its 2 zero-weight slivers invert: finer
    # remesh, every tet weighted again (min quality is 0.0017 after the edge collapse, as in v06)
    'v09': dict(surf_faces=800, cells=20, sliver_weight=1e-9),
    # v09 failed (no finer clean tet mesh: the collapse breaks watertightness, TetGen grades down to tiny tets).
    # v06's healthy rest + tets with v07's 4D data settings (90 px tent apex, 7000 samples, 1800 coverage pixels)
    'v10': dict(rest_from='v06', tet_mode='v06', surf_faces=500, cells=16),
    # integrator request: v10's rest / tets / indexing, 4D refitted through the backdrop's refined cameras, and kept
    # behind the tented sheet (membrane v06) in every frame
    'v11': dict(rest_from='v10', cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz',
                membrane='outputs/iter/tissues/membrane/v06/model.npz', w_sheet=3.0),
    # integrator feedback r12/r13: duct masks are background (the neck grew over the duct), neck tip pinned in 3D to
    # the duct's proximal end (video depth at the junction), uniform BCC tet mesh (sim inverted 380-500 small tets);
    # rest refitted through the refined cameras, behind membrane v07
    'v12': dict(rest_from=None, cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz',
                membrane='outputs/iter/tissues/membrane/v07/model.npz', w_sheet=3.0,
                duct_masks='outputs/iter/tissues/ducts/v10/masks.npz',
                neck_target='outputs/iter/tissues/ducts/v10/model.npz', w_neck=1.0, w_neck4d=0.5,
                neck_cap_mm=2.0, neck_nodes_mm=3.0, tet_mode='bcc', bcc_nodes=1330, band_ref='v11', mujoco_check=True),
    # v12's outline is less precise (boundary F 130-250: 0.31 vs v11 0.38, also on the full silhouette): the uniform
    # BCC mesh is stiffer under the same tet ARAP weight -> v12's rest / tets / indexing, softer ARAP, more iterations
    'v13': dict(rest_from='v12', w_arap=15.0, iters=130, mujoco_check=False),
    # integration r16/r17: v12 simulates better than v13; the 4D volume breathes +-4 % while the sim keeps it.
    # v12 mesh, total-volume term (+-0.5 % free, stiff beyond), 130 iterations; ARAP 22 (v14) / 30 (v15)
    'v14': dict(rest_from='v12', w_arap=22.0, iters=130, w_gvol=5.0, band_ref='v12', mujoco_check=False),
    'v15': dict(rest_from='v12', w_arap=30.0, iters=130, w_gvol=5.0, band_ref='v12', mujoco_check=False),
}


def cfg_of(ver):
    c = dict(BASE)
    for v in sorted(CFG):
        if v <= ver:
            c.update(CFG[v])
    return c


# ---------------------------------------------------------------- 1. masks
SHEET_KEYS = [0, 50, 100, 150, 200, 250]


def grasper_tips(V):
    """Jaw tip pixel per frame: the grasper-mask pixel farthest along the shaft's image direction (down-right)."""
    g = V.mask('grasper')
    d = np.array([0.45, 0.89])
    tips = np.full((V.n, 2), np.nan)
    for k in range(V.n):
        ys, xs = np.nonzero(g[k])
        if len(xs):
            i = np.argmax(xs * d[0] + ys * d[1])
            tips[k] = xs[i], ys[i]
    for c in range(2):
        ok = np.isfinite(tips[:, c])
        tips[:, c] = np.interp(np.arange(V.n), np.nonzero(ok)[0], tips[ok, c])
    return tips


def sheet_prompts(V):
    tips = grasper_tips(V)
    out = []
    for k in SHEET_KEYS:
        tx, ty = (int(round(a)) for a in tips[k])
        out.append(dict(frame=k, pos=[[tx + 6, ty + 22], [tx + 14, ty + 45]],
                        neg=[[tx + 10, min(ty + 190, 340)], [max(tx - 90, 5), ty + 40], [tx + 60, ty - 40]]))
    return out


def make_masks(V, apex_px, cache=None, negative=None):
    """body (n, H, W) bool, unknown (n, H, W) bool, sheet (n, H, W) bool (SAM 2.1 tent sheet, raw). negative: (n, H, W)
    pixels that are certainly not gallbladder (v12: the ducts track's duct + strands masks); removed from body and
    unknown, so they count as background."""
    if cache is not None and Path(cache).exists():
        sheet = np.load(cache)['sheet']
    else:
        sheet = views.segment_object(V, sheet_prompts(V))
    tips = grasper_tips(V)
    yy, xx = np.mgrid[:V.H, :V.W]
    k5 = np.ones((9, 9), np.uint8)
    body = np.zeros((V.n, V.H, V.W), bool)
    unknown = np.zeros_like(body)
    gb, strand = V.mask('gallbladder'), V.mask('strand')
    for k in range(V.n):
        near = (xx - tips[k, 0]) ** 2 + (yy - tips[k, 1]) ** 2 < apex_px ** 2
        apex = sheet[k] & near
        u = cv2.dilate(V.occluders(k).astype(np.uint8), k5).astype(bool)
        u |= cv2.dilate(strand[k].astype(np.uint8), k5).astype(bool)
        u |= apex
        b = gb[k] & ~u
        # keep the largest blobs only (drop specks the cuts leave behind)
        n_c, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8))
        if n_c > 2:
            big = 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])
            keep = [i for i in big if st[i, cv2.CC_STAT_AREA] >= 0.05 * st[big[0], cv2.CC_STAT_AREA]]
            b = np.isin(lab, keep)
        if negative is not None:
            b, u = b & ~negative[k], u & ~negative[k]
        body[k], unknown[k] = b, u
    return body, unknown, sheet


def neck_junction(V, body):
    """(n, 2) pixel where the body+neck mask meets the cystic duct / strands (`strand` mask): the body pixel closest
    to the strand mask; NaN where there is no strand."""
    st = V.mask('strand')
    out = np.full((V.n, 2), np.nan)
    for k in range(V.n):
        if st[k].sum() < 150 or not body[k].any():
            continue
        dt = cv2.distanceTransform((~st[k]).astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.nonzero(body[k])
        i = np.argmin(dt[ys, xs])
        if dt[ys[i], xs[i]] < 25:
            out[k] = xs[i], ys[i]
    return out


class OwnMasks:
    """Views proxy whose mask('gb_own') is the body+neck mask and whose occluders() are the unknown pixels, so the
    shared quality functions evaluate against it."""

    def __init__(self, V, body, unknown):
        self.V, self.body, self.unknown = V, body, unknown

    def __getattr__(self, a):
        return getattr(self.V, a)

    def mask(self, name):
        return self.body if name == 'gb_own' else self.V.mask(name)

    def occluders(self, k):
        return self.unknown[k]


# ---------------------------------------------------------------- torch helpers
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)


def rodrigues(r):
    th = torch.sqrt((r * r).sum() + 1e-12)
    k = r / th
    K = torch.zeros(3, 3)
    K = torch.stack([torch.stack([0 * th, -k[2], k[1]]), torch.stack([k[2], 0 * th, -k[0]]),
                     torch.stack([-k[1], k[0], 0 * th])])
    return torch.eye(3) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def rho(r, eps=0.5):
    """Charbonnier (mm): quadratic below eps, linear above."""
    return torch.sqrt(r * r + eps * eps) - eps


class FrameObs:
    """One frame's observations as tensors: camera, distance to the allowed region (body | unknown), measured depth,
    body pixels (coverage) and visible body points at their measured depth (mm, world)."""

    def __init__(self, V, k, body, unknown, n_cov=1200, n_pts=800, seed=0, junction=None, erode=7):
        rng = np.random.default_rng(seed + k)
        self.k = k
        self.junction = self.neck3d = None
        if junction is not None and np.isfinite(junction[k]).all():
            if junction.shape[1] == 3:                     # a 3D target (m): the duct's proximal end
                self.neck3d = T(junction[k] * 1000)
            else:
                self.junction = T(junction[k])
        self.R, self.pos, self.f = T(V.R[k]), T(V.pos[k] * 1000), float(V.f[k])
        self.H, self.W = V.H, V.W
        allowed = body[k] | unknown[k]
        dt = cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 5)
        self.dt = T(dt)[None, None]
        D = V.depth(k).astype(np.float32) * 1000
        self.D = T(D)[None, None]
        ys, xs = np.nonzero(body[k])
        self.area = len(xs)
        i = rng.choice(len(xs), min(n_cov, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
        self.cov = T(np.c_[xs[i], ys[i]])
        er = cv2.erode(body[k].astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool)
        ys, xs = np.nonzero(er)
        i = rng.choice(len(xs), min(n_pts, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
        P = V.unproject(xs[i], ys[i], V.depth(k)[ys[i], xs[i]], k) * 1000
        self.pts = T(P)
        self.zmed = float(np.median(D[body[k]])) if body[k].any() else 80.0
        self.px2mm = self.zmed / self.f

    def project(self, X):
        pc = (X - self.pos) @ self.R.T
        z = pc[:, 2].clamp(min=1.0)
        u = self.f * pc[:, 0] / z + self.W / 2
        v = self.f * pc[:, 1] / z + self.H / 2
        return u, v, pc[:, 2]

    def sample(self, img, u, v):
        g = torch.stack([u / (self.W - 1) * 2 - 1, v / (self.H - 1) * 2 - 1], -1)[None, None]
        return torch.nn.functional.grid_sample(img, g, align_corners=True, padding_mode='border')[0, 0, 0]

    sheet_z = sheet_m = None

    def sheet_violation(self, S, margin):
        """For surface samples that project into the sheet: how far (mm) they are in front of sheet depth + margin.
        Returns (violations of the samples in the sheet footprint, number of such samples)."""
        u, v, z = self.project(S)
        g = torch.stack([u / (self.W - 1) * 2 - 1, v / (self.H - 1) * 2 - 1], -1)[None, None]
        with torch.no_grad():
            m = torch.nn.functional.grid_sample(self.sheet_m, g, mode='nearest', align_corners=True)[0, 0, 0] > 0.5
            zs = torch.nn.functional.grid_sample(self.sheet_z, g, mode='nearest', align_corners=True)[0, 0, 0]
            m &= (u >= 0) & (u <= self.W - 1) & (v >= 0) & (v <= self.H - 1)
        return torch.relu(zs[m] + margin - z[m]), int(m.sum())

    def neck_loss(self, tip):
        """Pixel distance (as mm at organ depth) between the projected neck tip (3,) and the junction pixel."""
        if self.neck3d is not None:
            return rho(torch.sqrt(((tip - self.neck3d) ** 2).sum() + 1e-6))
        if self.junction is None:
            return tip.sum() * 0
        u, v, _ = self.project(tip[None])
        return rho(torch.sqrt((u[0] - self.junction[0]) ** 2 + (v[0] - self.junction[1]) ** 2 + 1e-6) * self.px2mm)

    def losses(self, S, Nrm, hidden_mm=8.0):
        """S (M, 3) surface samples (mm), Nrm (M, 3) their outward normals. Returns (outside, coverage, depth) in mm."""
        u, v, z = self.project(S)
        inside = (u >= 0) & (u <= self.W - 1) & (v >= 0) & (v <= self.H - 1) & (z > 1)
        with torch.no_grad():
            dmeas = self.sample(self.D, u, v)
            hidden = z > dmeas + hidden_mm
        sel = inside & ~hidden
        dt = self.sample(self.dt, u, v)
        l_out = (rho(dt[sel] * self.px2mm)).sum() / max(len(S), 1) * 4 if sel.any() else S.sum() * 0
        # coverage: every body pixel near a projected surface point (hinge at 2 px: sample spacing)
        q = torch.stack([u[inside], v[inside]], 1)
        if len(self.cov) and len(q):
            d = torch.cdist(self.cov, q).min(1)[0]
            l_cov = rho(torch.relu(d - 2.0) * self.px2mm).mean()
        else:
            l_cov = S.sum() * 0
        # depth: observed visible points to the nearest front-facing surface sample
        with torch.no_grad():
            front = ((Nrm * (S - self.pos)).sum(1) < 0) & inside
        if len(self.pts) and front.sum() > 20:
            d = torch.cdist(self.pts, S[front]).min(1)[0]
            l_dep = rho(d).mean()
        else:
            l_dep = S.sum() * 0
        return l_out, l_cov, l_dep


class Surface:
    """Fixed barycentric samples on a triangle mesh (faces F); samples and normals from vertex positions X (torch)."""

    def __init__(self, Xnp, F, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        A = np.linalg.norm(np.cross(Xnp[F[:, 1]] - Xnp[F[:, 0]], Xnp[F[:, 2]] - Xnp[F[:, 0]]), axis=1)
        fi = rng.choice(len(F), n, p=A / A.sum())
        r = rng.random((n, 2))
        r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
        self.F = torch.as_tensor(F, dtype=torch.long)
        self.fi = torch.as_tensor(fi, dtype=torch.long)
        self.b = T(np.c_[1 - r.sum(1), r])

    def __call__(self, X):
        tri = X[self.F[self.fi]]
        S = (tri * self.b[:, :, None]).sum(1)
        with torch.no_grad():
            n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
            n = n / (n.norm(dim=1, keepdim=True) + 1e-9)
        return S, n


def data_loss(obs_list, S, Nrm, c, weights=None):
    tot, parts = 0.0, np.zeros(3)
    for j, o in enumerate(obs_list):
        lo, lc, ld = o.losses(S, Nrm, c['hidden_mm'])
        w = 1.0 if weights is None else weights[j]
        tot = tot + w * (c['w_sil'] * lo + c['w_cov'] * c['w_sil'] * lc + c['w_depth'] * ld)
        parts += [float(lo), float(lc), float(ld)]
    return tot / len(obs_list), parts / len(obs_list)


# ---------------------------------------------------------------- 2. rest shape
def lattice_weights(Vs, lo, hi, n=4):
    """Trilinear weights (N, n^3) of points in the box [lo, hi] for an n^3 lattice."""
    t = (Vs - lo) / (hi - lo + 1e-9) * (n - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
    w = t - i0
    Wm = np.zeros((len(Vs), n ** 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * (w[:, 2] if dz else 1 - w[:, 2])
                idx = ((i0[:, 0] + dx) * n + (i0[:, 1] + dy)) * n + (i0[:, 2] + dz)
                np.add.at(Wm, (np.arange(len(Vs)), idx), wt)
    return Wm


def initial_poses(P, view, V0):
    from scipy.spatial.transform import Rotation as Rot
    c = P.mean(0)
    _, _, ax = np.linalg.svd(P - c, full_matrices=False)
    out = []
    for sgn in (1, -1):
        a = sgn * ax[0]
        base = Rot.align_vectors([a], [[1.0, 0, 0]])[0]
        for roll in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            Rm = Rot.from_rotvec(a * roll) * base
            half = np.abs((V0 @ Rm.as_matrix().T) @ view).max()
            out.append((Rm.as_rotvec(), c + view * half * 0.6))
    return out


def scale_hinge(s, c):
    """Stiff penalty outside the plausible scale range of the template axes (anatomical size)."""
    if c['scale_bounds'] is None:
        return 0.0
    lo, hi = T(np.log(c['scale_bounds'][0])), T(np.log(c['scale_bounds'][1]))
    return 100.0 * ((torch.relu(lo - s) + torch.relu(s - hi)) ** 2).sum()


def neck_tip_index(V0, cap=5.0):
    """Template vertices of the neck tip (principal frame: the narrow end is +x, the fundus -x), within cap mm."""
    return np.nonzero(V0[:, 0] > V0[:, 0].max() - cap)[0]


def fit_rest(V, body, unknown, c, log, junction=None):
    V0, F0 = organ.load_shape('template', 'gallbladder', faces=3000)       # metres, principal frame
    V0 = V0 * 1000                                                         # mm
    surf = Surface(V0, F0, n=3000)
    tip_i = torch.as_tensor(neck_tip_index(V0, c['neck_cap_mm']), dtype=torch.long)
    obs = [FrameObs(V, k, body, unknown, n_cov=800, n_pts=500, junction=junction, erode=c['depth_erode'])
           for k in c['rest_keys']]
    wts = [c['early_weight'] if k < 90 else 1.0 for k in c['rest_keys']]
    P = np.concatenate([o.pts.numpy() for o in obs])
    view = np.mean([V.R[k][2] for k in c['rest_keys']], 0)
    V0t = T(V0)

    def shape(r, t, s, off=None, Wl=None):
        Xs = V0t * torch.exp(s)
        if off is not None:
            Xs = Xs + Wl @ off
        return Xs @ rodrigues(r).T + t

    def run(r0, t0, s0, iters, sub, sw):
        r = torch.tensor(r0, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t0, dtype=torch.float32, requires_grad=True)
        s = torch.tensor(s0, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([dict(params=[r], lr=0.02), dict(params=[t], lr=1.0), dict(params=[s], lr=0.02)])
        for it in range(iters):
            opt.zero_grad()
            X = shape(r, t, s)
            S, N = surf(X)
            L, parts = data_loss(sub, S, N, c, sw)
            L = L + c['scale_prior'] * (s * s).sum() + scale_hinge(s, c)
            if c['w_neck'] > 0:
                tip = X[tip_i].mean(0)
                L = L + c['w_neck'] * sum(o.neck_loss(tip) for o in sub) / len(sub)
            L.backward()
            opt.step()
        return r.detach().numpy(), t.detach().numpy(), s.detach().numpy(), float(L), parts

    t0 = time.time()
    sub = obs[::3]
    cands = []
    for r0, tr0 in initial_poses(P, view, V0):
        cands.append(run(r0, tr0, np.zeros(3), c['rest_iters'] // 2, sub, wts[::3]))
    cands.sort(key=lambda x: x[3])
    log(f'[rest] pose search: best losses {[round(x[3], 3) for x in cands[:5]]} ({time.time() - t0:.0f} s)')
    best = None
    for cand in cands[:3]:
        res = run(cand[0], cand[1], cand[2], c['rest_iters'], obs, wts)
        if best is None or res[3] < best[3]:
            best = res
    r, t, s, Lp, parts = best
    log(f'[rest] pose+scale: loss {Lp:.3f} parts(out,cov,depth mm) {np.round(parts, 3)} scale {np.exp(s).round(3)} '
        f'({time.time() - t0:.0f} s)')
    # free-form lattice on top
    lo, hi = V0.min(0), V0.max(0)
    Wl = T(lattice_weights(V0, lo, hi))
    off = torch.zeros(64, 3, requires_grad=True)
    rt = torch.tensor(r, requires_grad=True)
    tt = torch.tensor(t, requires_grad=True)
    st = torch.tensor(s, requires_grad=True)
    opt = torch.optim.Adam([dict(params=[off], lr=0.3), dict(params=[rt], lr=0.005), dict(params=[tt], lr=0.3),
                            dict(params=[st], lr=0.005)])
    for it in range(c['rest_ffd_iters']):
        opt.zero_grad()
        X = shape(rt, tt, st, off, Wl)
        S, N = surf(X)
        L, parts = data_loss(obs, S, N, c, wts)
        o = off.reshape(4, 4, 4, 3)
        lap = sum((torch.diff(o, n=2, dim=a) ** 2).mean() for a in range(3))
        if c['w_neck'] > 0:
            L = L + c['w_neck'] * sum(o.neck_loss(X[tip_i].mean(0)) for o in obs) / len(obs)
        L = L + c['scale_prior'] * (st * st).sum() + scale_hinge(st, c) + c['ffd_smooth'] * lap + c['ffd_size'] * (off ** 2).mean() / 100
        L.backward()
        opt.step()
    log(f'[rest] + lattice: loss {float(L):.3f} parts {np.round(parts, 3)} max offset {float(off.abs().max()):.1f} mm '
        f'({time.time() - t0:.0f} s)')
    with torch.no_grad():
        Xr = shape(rt, tt, st, off, Wl).numpy() / 1000
        Xpose = shape(rt, tt, st).numpy() / 1000
        tip_w = Xr[tip_i.numpy()].mean(0)
    if junction is not None:
        err = []
        for o in obs:
            if o.neck3d is not None:
                err.append(float(torch.linalg.norm(T(tip_w * 1000) - o.neck3d)))
            elif o.junction is not None:
                u, v, _ = o.project(T(tip_w * 1000)[None])
                err.append(float(torch.sqrt((u[0] - o.junction[0]) ** 2 + (v[0] - o.junction[1]) ** 2)))
        log(f'[rest] neck tip vs junction: median {np.median(err):.1f} ({"mm, 3D" if junction.shape[1] == 3 else "px"}) '
            f'over {len(err)} keyframes')
    info = dict(neck_tip=tip_w.round(5).tolist(), scale=np.exp(st.detach().numpy()).round(3).tolist(), rotvec=rt.detach().numpy().round(4).tolist(),
                t_mm=tt.detach().numpy().round(2).tolist(), lattice_max_offset_mm=round(float(off.abs().max()), 2),
                fit_parts_mm=dict(outside=round(parts[0], 3), coverage=round(parts[1], 3), depth=round(parts[2], 3)))
    return Xr, F0, Xpose, info


# ---------------------------------------------------------------- 3. tets
def remesh_projected(X, F, faces, cells, pairs=True):
    """Marching-cubes remesh (organ.remesh without its sample snapping, which collapses edges to ~0.03 mm), then
    every vertex moved to the closest point of the original surface; edges that would get shorter than 35 % of the
    median keep their marching-cubes positions."""
    import trimesh
    from scipy.spatial import cKDTree
    Vr, Fr = organ.remesh(X, F, faces, cells, snap=False)
    m = trimesh.Trimesh(X, F, process=False)
    S, fi = trimesh.sample.sample_surface(m, 100000, seed=0)          # closest point ~ nearest sample's tangent plane
    n = m.face_normals[fi]
    _, nn = cKDTree(S).query(Vr)
    cp = Vr - ((Vr - S[nn]) * n[nn]).sum(1, keepdims=True) * n[nn]
    E = trimesh.Trimesh(Vr, Fr, process=False).edges_unique
    med = np.median(np.linalg.norm(Vr[E[:, 0]] - Vr[E[:, 1]], axis=1))
    out = cp.copy()
    for _ in range(8):
        L = np.linalg.norm(out[E[:, 0]] - out[E[:, 1]], axis=1)
        bad = set(E[L < 0.35 * med].ravel().tolist())
        # also non-adjacent vertices projected onto (nearly) the same point (two sheets of a thin fold); TetGen
        # refines such spots into clusters of 0.2 mm tets that invert under any motion (v07)
        if pairs:
            bad |= {i for pr in cKDTree(out).query_pairs(0.35 * med) for i in pr}
        if not bad:
            break
        bad = np.array(sorted(bad))
        out[bad] = 0.5 * (out[bad] + Vr[bad])
    return out, Fr


def tet_quality(N, T):
    """Volume / longest edge^3 per tet (regular tet 0.118; slivers -> 0)."""
    P = N[T]
    vol = np.linalg.det(np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], 2)) / 6
    E = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)], 1)
    return vol / E.max(1) ** 3


def _tetgen_q(V, F, q, mindihedral, minratio):
    import tetgen
    try:
        nodes, elems = tetgen.TetGen(V, F).tetrahedralize(order=1, mindihedral=mindihedral, minratio=minratio,
                                                          quality=True)[:2]
        q.put((np.asarray(nodes, float), np.asarray(elems, int)))
    except Exception as e:                                  # noqa: BLE001
        q.put(e)


def clean_tets(N, T, thr=0.01, smooth_iters=20):
    """Peel sliver caps (quality < thr with >= 2 faces on the boundary; volume ~ 0), then Laplacian moves of interior
    nodes accepted only where they raise the worst incident tet quality."""
    for _ in range(20):
        bset = {tuple(sorted(f)) for f in organ.boundary_faces(T)}
        q = tet_quality(N, T)
        nb = np.array([sum(tuple(sorted(f)) in bset for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]), (t[0], t[2], t[3]),
                                                             (t[1], t[2], t[3]))) for t in T])
        rm = (q < thr) & (nb >= 2)
        if not rm.any():
            break
        T = T[~rm]
    used = np.unique(T)
    remap = -np.ones(len(N), int)
    remap[used] = np.arange(len(used))
    N, T = N[used], remap[T]
    surf = np.unique(organ.boundary_faces(T))
    inner = np.setdiff1d(np.arange(len(N)), surf)
    E = np.unique(np.sort(np.concatenate([T[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), 1), axis=0)
    cnt = np.zeros(len(N))
    np.add.at(cnt, E[:, 0], 1)
    np.add.at(cnt, E[:, 1], 1)
    for _ in range(smooth_iters):
        S = np.zeros_like(N)
        np.add.at(S, E[:, 0], N[E[:, 1]])
        np.add.at(S, E[:, 1], N[E[:, 0]])
        cand = N.copy()
        cand[inner] = 0.5 * N[inner] + 0.5 * S[inner] / np.maximum(cnt[inner, None], 1)
        q0, q1 = tet_quality(N, T), tet_quality(cand, T)
        w0, w1 = np.full(len(N), np.inf), np.full(len(N), np.inf)
        for j in range(4):
            np.minimum.at(w0, T[:, j], q0)
            np.minimum.at(w1, T[:, j], q1)
        ok = np.zeros(len(N), bool)
        ok[inner] = w1[inner] > w0[inner]
        N = np.where(ok[:, None], cand, N)
    return N, T


def collapse_short(V, F, ratio=0.45, iters=10):
    """Collapse surface edges shorter than `ratio` x median to their midpoint (the decimated marching-cubes surface has
    edges down to 7 % of the median; TetGen's quality refinement fills those spots with sub-millimetre tets)."""
    import trimesh
    for _ in range(iters):
        m = trimesh.Trimesh(V, F, process=False)
        E, L = m.edges_unique, m.edges_unique_length
        short = np.argsort(L)
        short = short[L[short] < ratio * np.median(L)]
        if not len(short):
            break
        rep, used, Vn = np.arange(len(V)), np.zeros(len(V), bool), V.copy()
        for e in short:
            a, b = E[e]
            if used[a] or used[b]:
                continue
            used[a] = used[b] = True
            Vn[a] = 0.5 * (V[a] + V[b])
            rep[b] = a
        F2 = rep[F]
        keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
        m2 = trimesh.Trimesh(Vn, F2[keep], process=False)
        m2.remove_unreferenced_vertices()
        V, F = np.asarray(m2.vertices), np.asarray(m2.faces)
    return V, F


def tet_measures(N, T):
    """Per tet: mean-ratio quality 6*sqrt(2)*V / l_rms^3 (1 = regular, 0 = flat), volume / median volume, smallest
    dihedral angle (deg)."""
    P = N[T]
    vol = np.linalg.det(np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], 2)) / 6
    E = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)], 1)
    lrms = np.sqrt((E ** 2).mean(1))
    q = 6 * np.sqrt(2) * vol / lrms ** 3
    # dihedral angles from face normals
    fn = []
    for f in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
        n = np.cross(P[:, f[1]] - P[:, f[0]], P[:, f[2]] - P[:, f[0]])
        fn.append(n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-15))
    dih = []
    for a in range(4):
        for b in range(a + 1, 4):
            dih.append(np.degrees(np.pi - np.arccos(np.clip((fn[a] * fn[b]).sum(1), -1, 1))))
    return q, vol / np.median(vol), np.min(np.stack(dih, 1), 1)


def tet_report(N, T):
    q, vr, dmin = tet_measures(N, T)
    pc = lambda x: {'min': round(float(x.min()), 4), 'p1': round(float(np.percentile(x, 1)), 4),
                    'p5': round(float(np.percentile(x, 5)), 4), 'median': round(float(np.median(x)), 4)}
    return dict(n_nodes=len(N), n_tets=len(T), mean_ratio=pc(q), volume_over_median=pc(vr), min_dihedral_deg=pc(dmin))


class SurfaceSDF:
    """Signed distance (m) to a closed surface: magnitude from a dense surface sample (KD-tree, tangent-plane
    corrected), sign from a filled voxel grid. closest(x) = nearest surface point."""

    def __init__(self, X, F, n=200000, pitch=None):
        import trimesh
        from scipy.spatial import cKDTree
        m = trimesh.Trimesh(X, F, process=False)
        S, fi = trimesh.sample.sample_surface_even(m, n, seed=0)
        self.S, self.n = S, m.face_normals[fi]
        self.tree = cKDTree(S)
        pitch = pitch or float(np.sort(m.extents)[0]) / 60
        vox = m.voxelized(pitch).fill()
        self.occ = vox.matrix
        self.T = np.linalg.inv(vox.transform)

    def inside(self, x):
        ijk = np.round(np.c_[x, np.ones(len(x))] @ self.T.T)[:, :3].astype(int)
        ok = ((ijk >= 0) & (ijk < np.array(self.occ.shape))).all(1)
        out = np.zeros(len(x), bool)
        out[ok] = self.occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out

    def closest(self, x):
        _, i = self.tree.query(x)
        d = ((x - self.S[i]) * self.n[i]).sum(1, keepdims=True)
        return x - d * self.n[i]

    def __call__(self, x):
        d = np.linalg.norm(x - self.closest(x), axis=1)
        return np.where(self.inside(x), -d, d)


def manifold_tets(N, T, iters=50):
    """Drop tets until the boundary is a closed 2-manifold: tets with >= 3 boundary faces (spikes), then tets at
    non-manifold boundary edges / vertices (the ones with most boundary faces first)."""
    import trimesh
    for _ in range(iters):
        bf = organ.boundary_faces(T)
        bset = {tuple(sorted(f)) for f in bf}
        nb = np.array([sum(tuple(sorted(f)) in bset for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]), (t[0], t[2], t[3]),
                                                             (t[1], t[2], t[3]))) for t in T])
        if (nb >= 3).any():
            T = T[nb < 3]
            continue
        m = trimesh.Trimesh(N, bf, process=False)
        if m.is_watertight and len(m.split(only_watertight=False)) == 1:
            # non-manifold vertices: a vertex whose boundary-face fan is not one disc
            E = np.sort(np.concatenate([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [0, 2]]]), 1)
            V_ = len(np.unique(bf))
            if V_ - len(np.unique(E, axis=0)) + len(bf) == 2 * len(m.split(only_watertight=False)):
                return T
            bad_v = _nonmanifold_vertices(bf)
        else:
            E = np.sort(np.concatenate([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [0, 2]]]), 1)
            ue, cnt = np.unique(E, axis=0, return_counts=True)
            bad_v = np.unique(ue[cnt != 2])
            if not len(bad_v):
                bad_v = _nonmanifold_vertices(bf)
        if not len(bad_v):
            return T
        hit = np.isin(T, bad_v).any(1)
        score = nb + hit * 10
        cand = np.nonzero(hit & (nb >= 1))[0]
        if not len(cand):
            cand = np.nonzero(hit)[0]
        T = np.delete(T, cand[np.argmax(score[cand])] if len(cand) else [], axis=0) if len(cand) else T
    return T


def _nonmanifold_vertices(bf):
    """Boundary vertices whose incident boundary faces do not form a single edge-connected fan."""
    from collections import defaultdict
    inc = defaultdict(list)
    for fi, f in enumerate(bf):
        for v in f:
            inc[v].append(fi)
    bad = []
    for v, fs in inc.items():
        # union faces sharing an edge through v
        par = {f: f for f in fs}
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        others = {f: set(bf[f]) - {v} for f in fs}
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                if others[a] & others[b]:
                    par[find(a)] = find(b)
        if len({find(f) for f in fs}) > 1:
            bad.append(v)
    return np.array(bad, int)


def bcc_tets(X, F, h, smooth_iters=30, log=None, warp=0.25, peel_q=0.35, peel_v=0.3):
    """Uniform tet mesh of a closed surface: body-centred cubic lattice (spacing h), tets with their centroid inside
    kept, boundary nodes snapped onto the surface, then quality-checked smoothing (interior: Laplacian; boundary:
    tangential Laplacian + re-projection) and peeling of flat boundary caps. Every tet is about the same size."""
    sdf = SurfaceSDF(X, F)
    lo, hi = X.min(0) - h, X.max(0) + h
    n = np.ceil((hi - lo) / h).astype(int) + 1
    gi = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'), -1).reshape(-1, 3)
    corners = lo + gi * h
    nc = n - 1
    ci = np.stack(np.meshgrid(*[np.arange(k) for k in nc], indexing='ij'), -1).reshape(-1, 3)
    centres = lo + (ci + 0.5) * h
    Pidx = lambda i, j, k: (i * n[1] + j) * n[2] + k
    Cidx = lambda i, j, k: len(corners) + (i * nc[1] + j) * nc[2] + k
    N = np.r_[corners, centres]
    tets = []
    for ax in range(3):
        o = [1 if a == ax else 0 for a in range(3)]
        rng = [np.arange(nc[a] - o[a]) for a in range(3)]
        I, J, K = [a.ravel() for a in np.meshgrid(*rng, indexing='ij')]
        c1 = Cidx(I, J, K)
        c2 = Cidx(I + o[0], J + o[1], K + o[2])
        # the shared face lies at index +1 along ax; its 4 corners in cyclic order
        b = [I + o[0], J + o[1], K + o[2]]
        u, w = [a for a in range(3) if a != ax]
        def corner(du, dw):
            q = [b[0].copy(), b[1].copy(), b[2].copy()]
            q[u] = q[u] + du
            q[w] = q[w] + dw
            return Pidx(*q)
        cyc = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]
        for e in range(4):
            tets.append(np.stack([c1, c2, cyc[e], cyc[(e + 1) % 4]], 1))
    Tt = np.concatenate(tets)
    keep = sdf.inside(N[Tt].mean(1))
    Tt = Tt[keep]
    used = np.unique(Tt)
    remap = -np.ones(len(N), int)
    remap[used] = np.arange(len(used))
    N, Tt = N[used].copy(), remap[Tt]
    if warp > 0:                       # isosurface-stuffing style: lattice nodes close to the surface go onto it
        d = sdf(N)
        near = np.abs(d) < warp * h
        N[near] = sdf.closest(N[near])
    Tt = organ.orient_tets(N, Tt)
    Tt = Tt[tet_measures(N, Tt)[0] > 0.02]
    Tt = manifold_tets(N, Tt)
    for rnd in range(3):
        bf = organ.boundary_faces(Tt)
        bnd = np.unique(bf)
        N[bnd] = sdf.closest(N[bnd])
        # flat boundary caps (all 4 nodes on the surface, >= 2 boundary faces, near zero volume) are removed
        q, _, _ = tet_measures(N, Tt)
        bset = {tuple(sorted(f)) for f in bf}
        nb = np.array([sum(tuple(sorted(f)) in bset for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]), (t[0], t[2], t[3]),
                                                             (t[1], t[2], t[3]))) for t in Tt])
        bad = ((q < 0.15) & (nb >= 2)) | ((q < 0.05) & (nb >= 1))
        if not bad.any():
            break
        Tt = manifold_tets(N, Tt[~bad])
        used = np.unique(Tt)
        remap = -np.ones(len(N), int)
        remap[used] = np.arange(len(used))
        N, Tt = N[used], remap[Tt]
    # quality-checked smoothing
    E = np.unique(np.sort(np.concatenate([Tt[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), 1), axis=0)
    bnd = np.zeros(len(N), bool)
    bnd[np.unique(organ.boundary_faces(Tt))] = True
    bf = organ.boundary_faces(Tt)
    Eb = np.unique(np.sort(np.concatenate([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [0, 2]]]), 1), axis=0)
    def nbr_mean(Ed, mask):
        S = np.zeros_like(N)
        c = np.zeros(len(N))
        np.add.at(S, Ed[:, 0], N[Ed[:, 1]])
        np.add.at(S, Ed[:, 1], N[Ed[:, 0]])
        np.add.at(c, Ed[:, 0], 1)
        np.add.at(c, Ed[:, 1], 1)
        return S / np.maximum(c, 1)[:, None]
    for it in range(smooth_iters):
        cand = N.copy()
        mi = nbr_mean(E, ~bnd)
        cand[~bnd] = 0.5 * N[~bnd] + 0.5 * mi[~bnd]
        mb = nbr_mean(Eb, bnd)
        cand[bnd] = sdf.closest(0.6 * N[bnd] + 0.4 * mb[bnd])
        q0 = tet_measures(N, Tt)[0]
        q1 = tet_measures(cand, Tt)[0]
        w0, w1 = np.full(len(N), np.inf), np.full(len(N), np.inf)
        for j in range(4):
            np.minimum.at(w0, Tt[:, j], q0)
            np.minimum.at(w1, Tt[:, j], q1)
        ok = w1 >= w0 - 1e-9
        Nn = np.where(ok[:, None], cand, N)
        for _ in range(20):                        # joint moves of neighbours can still flatten a tet: undo them
            qn = tet_measures(Nn, Tt)[0]
            worse = (qn < np.minimum(q0, 0.3)) & (qn < q0)
            if not worse.any():
                break
            back = np.unique(Tt[worse])
            Nn[back] = N[back]
        N = Nn
    N = optimise_tets(N, Tt, sdf)
    # flat or small caps on the surface (all four nodes on it, two boundary faces) are peeled off
    for _ in range(4):
        bf = organ.boundary_faces(Tt)
        bset = {tuple(sorted(f)) for f in bf}
        nb = np.array([sum(tuple(sorted(f)) in bset for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]), (t[0], t[2], t[3]),
                                                             (t[1], t[2], t[3]))) for t in Tt])
        q, vr, _ = tet_measures(N, Tt)
        bad = (nb >= 2) & ((q < peel_q) | (vr < peel_v))
        if not bad.any():
            break
        Tt = manifold_tets(N, Tt[~bad])
        used = np.unique(Tt)
        remap = -np.ones(len(N), int)
        remap[used] = np.arange(len(used))
        N, Tt = N[used], remap[Tt]
        N = optimise_tets(N, Tt, sdf, iters=60)
    for _ in range(10):                            # anything still (nearly) flat or inverted: peel if on the boundary
        q = tet_measures(N, Tt)[0]
        if q.min() > 0.05:
            break
        Tt = manifold_tets(N, Tt[q > 0.05])
        used = np.unique(Tt)
        remap = -np.ones(len(N), int)
        remap[used] = np.arange(len(used))
        N, Tt = N[used], remap[Tt]
    if log:
        log(f'[bcc] h {h * 1000:.2f} mm: {tet_report(N, Tt)}')
    return N, Tt, organ.boundary_faces(Tt)


def optimise_tets(N, Tt, sdf, iters=150, target=0.7):
    """Raise the worst tets' mean-ratio quality: Adam on node positions (mm) minimising sum relu(target - q)^2, with
    boundary nodes re-projected onto the surface after every step; a step that lowers the minimum quality is undone."""
    bnd = np.zeros(len(N), bool)
    bnd[np.unique(organ.boundary_faces(Tt))] = True
    Tl = torch.as_tensor(Tt, dtype=torch.long)
    X = torch.tensor(N * 1000, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([X], lr=0.03)
    best = N.copy()
    qbest = tet_measures(N, Tt)[0].min()
    for it in range(iters):
        opt.zero_grad()
        P = X[Tl]
        vol = torch.det(torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], 2)) / 6
        E2 = sum(((P[:, a] - P[:, b]) ** 2).sum(1) for a in range(4) for b in range(a + 1, 4)) / 6
        q = 6 * np.sqrt(2) * vol / E2 ** 1.5
        L = (torch.relu(target - q) ** 2).sum()
        L.backward()
        opt.step()
        with torch.no_grad():
            Xn = X.numpy() / 1000
            Xn[bnd] = sdf.closest(Xn[bnd])
            X.copy_(torch.as_tensor(Xn * 1000))
        qmin = tet_measures(Xn, Tt)[0].min()
        if qmin >= qbest:
            qbest, best = qmin, Xn.copy()
    return best


def bcc_for_count(X, F, target_nodes, log=None):
    """bcc_tets with the lattice spacing that gives about target_nodes nodes."""
    import trimesh
    vol = abs(trimesh.Trimesh(X, F, process=False).volume)
    h = (2 * vol / target_nodes) ** (1 / 3)                      # 2 nodes per cube in a BCC lattice
    for _ in range(4):
        N, Tt, Fb = bcc_tets(X, F, h)
        r = len(N) / target_nodes
        if abs(r - 1) < 0.06:
            break
        h *= r ** (1 / 3)
    if log:
        log(f'[bcc] h {h * 1000:.2f} mm: {tet_report(N, Tt)}')
    return N, Tt, Fb


def tetrahedralize(X, F, faces, cells, snap='organ', tetq=None, mode='v08'):
    """mode 'v06': projected remesh -> TetGen(tetq) -> clean_tets, first candidate without degenerate tets (what v06
    used); 'v08': + vertex-pair check, short-edge collapse, candidate with the largest min/median tet edge ratio."""
    if tetq is not None and mode == 'v06':
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        for fc, cl in ((faces, cells), (int(faces * 0.8), int(cells * 0.85)), (int(faces * 0.6), int(cells * 0.7))):
            Vr, Fr = remesh_projected(X, F, fc, cl, pairs=False)
            q = ctx.Queue()
            p = ctx.Process(target=_tetgen_q, args=(Vr, Fr, q, tetq[0], tetq[1]))
            p.start()
            try:
                r = q.get(timeout=300)
            except Exception:                               # noqa: BLE001
                r = None
            p.join(10)
            if p.is_alive():
                p.kill()
            if r is None or isinstance(r, Exception):
                continue
            nodes, elems = clean_tets(r[0], organ.orient_tets(*r))
            if not organ.degenerate(nodes, elems) and (tet_quality(nodes, elems) > 0).all():
                return nodes, elems, organ.boundary_faces(elems)
        raise RuntimeError('tetrahedralisation failed')
    if tetq is not None:                                    # TetGen with stricter quality + sliver clean-up
        import multiprocessing as mp
        import trimesh
        ctx = mp.get_context('spawn')
        best = None
        for fc, cl in ((faces, cells), (faces, int(cells * 0.88)), (int(faces * 0.8), int(cells * 0.8)),
                       (int(faces * 0.6), int(cells * 0.7))):
            Vr, Fr = remesh_projected(X, F, fc, cl)
            Vr, Fr = collapse_short(Vr, Fr)
            if not trimesh.Trimesh(Vr, Fr, process=False).is_watertight:
                continue
            q = ctx.Queue()
            p = ctx.Process(target=_tetgen_q, args=(Vr, Fr, q, tetq[0], tetq[1]))
            p.start()
            try:
                r = q.get(timeout=300)
            except Exception:                               # noqa: BLE001
                r = None
            p.join(10)
            if p.is_alive():
                p.kill()
            if r is None or isinstance(r, Exception):
                continue
            nodes, elems = r
            elems = organ.orient_tets(nodes, elems)
            nodes, elems = clean_tets(nodes, elems)
            if organ.degenerate(nodes, elems) or not (tet_quality(nodes, elems) > 0).all():
                continue
            P = nodes[elems]
            Le = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)], 1)
            ratio = Le.min() / np.median(Le)                # clusters of tiny tets (v07) invert under any motion
            if best is None or ratio > best[0]:
                best = (ratio, nodes, elems)
            if ratio >= 0.1:
                break
        if best is None:
            raise RuntimeError('tetrahedralisation failed')
        _, nodes, elems = best
        return nodes, elems, organ.boundary_faces(elems)
    return _tetrahedralize_v01(X, F, faces, cells, snap)


def _tetrahedralize_v01(X, F, faces, cells, snap='organ'):
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    for fc, cl, sn in ((faces, cells, True), (int(faces * 0.8), int(cells * 0.85), True), (faces, cells, False)):
        if snap == 'project' and sn:
            Vr, Fr = remesh_projected(X, F, fc, cl)
        else:
            Vr, Fr = organ.remesh(X, F, fc, cl, sn)
        q = ctx.Queue()
        p = ctx.Process(target=organ._tetgen_worker, args=(Vr, Fr, q))
        p.start()
        try:
            r = q.get(timeout=300)
        except Exception:                                   # noqa: BLE001
            r = None
        p.join(10)
        if p.is_alive():
            p.kill()
        if r is not None and not isinstance(r, Exception) and not organ.degenerate(*r):
            nodes, elems = r
            elems = organ.orient_tets(nodes, elems)
            return nodes, elems, organ.boundary_faces(elems)
    raise RuntimeError('tetrahedralisation failed')


# ---------------------------------------------------------------- 4. 4D
def raster_depth(V, X, F, k):
    """Per-pixel nearest depth (m, perspective-correct barycentric) of a triangle mesh in frame k; inf elsewhere."""
    q, z = V.project(X, k)
    zb = np.full((V.H, V.W), np.inf, np.float32)
    for f in F:
        p, zz = q[f], z[f]
        if (zz <= 1e-4).any():
            continue
        x0, x1 = max(int(np.floor(p[:, 0].min())), 0), min(int(np.ceil(p[:, 0].max())), V.W - 1)
        y0, y1 = max(int(np.floor(p[:, 1].min())), 0), min(int(np.ceil(p[:, 1].max())), V.H - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(d) < 1e-9:
            continue
        l0 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / d
        l1 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / d
        l2 = 1 - l0 - l1
        ins = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        if not ins.any():
            continue
        zi = 1.0 / (l0 / zz[0] + l1 / zz[1] + l2 / zz[2])
        yy, xx = ys[ins], xs[ins]
        zb[yy, xx] = np.minimum(zb[yy, xx], zi[ins])
    return zb


def sheet_depths(V, path):
    """(n, H, W) depth (m) of the membrane track's verts4d in every frame, inf where the sheet is not."""
    m = np.load(ROOT / path)
    return np.stack([raster_depth(V, m['verts4d'][k], m['faces'], k) for k in range(V.n)])


class TetARAP:
    def __init__(self, X0, tets, uniform=False, sliver_weight=False):
        self.T = torch.as_tensor(tets, dtype=torch.long)
        if sliver_weight:               # slivers below 0.003 get no energy at all: their det F is hypersensitive
            tq = tet_quality(np.asarray(X0, float), np.asarray(tets))
            self.qw = T(np.where(tq < sliver_weight, 0.0, np.minimum(1.0, tq / 0.02)))
        else:
            self.qw = torch.ones(len(tets))
        X0 = T(X0)
        Dm = torch.stack([X0[self.T[:, i]] - X0[self.T[:, 0]] for i in (1, 2, 3)], 2)
        self.vol = torch.det(Dm).abs() / 6
        self.w = torch.full_like(self.vol, 1.0 / len(self.vol)) if uniform else self.vol / self.vol.sum()
        self.w = self.w * self.qw
        self.Dm_inv = torch.linalg.inv(Dm)

    def F(self, X):
        Ds = torch.stack([X[self.T[:, i]] - X[self.T[:, 0]] for i in (1, 2, 3)], 2)
        return Ds @ self.Dm_inv

    def __call__(self, X):
        Fm = self.F(X)
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(Fm)
            d = torch.det(U @ Vh)
            U[:, :, 2] *= d[:, None]
            R = U @ Vh
        e_arap = (self.w * ((Fm - R) ** 2).sum((1, 2))).sum()
        det = torch.det(Fm)
        e_vol = (self.w * (det - 1) ** 2).sum()
        self.barrier = (torch.relu(0.3 - det) ** 2).mean()
        self.barrier_sum = (self.qw * torch.relu(0.3 - det) ** 2).sum()
        return e_arap, e_vol


def bed_vertices(V, X, faces, cos):
    """Surface vertices whose outward normal points away from the cameras (the hidden back, lying on the liver bed)."""
    import trimesh
    vn = trimesh.Trimesh(X, faces, process=False).vertex_normals
    surf = np.unique(faces)
    view = np.mean(V.R[:, 2], 0)
    return surf[(vn[surf] @ view) > cos]


def fit_4d(V, body, unknown, nodes, tets, faces, c, log, junction=None, neck_tip=None, bed_idx=None, sheet_z=None):
    X0 = T(nodes * 1000)
    surf_i = np.unique(faces)
    tip_nodes = torch.as_tensor(surf_i[np.linalg.norm(nodes[surf_i] - neck_tip, axis=1) < c['neck_nodes_mm'] / 1000],
                                dtype=torch.long) \
        if neck_tip is not None else None
    arap = TetARAP(nodes * 1000, tets, c['arap_uniform'], c['sliver_weight'])
    bed = torch.as_tensor(bed_vertices(V, nodes, faces, c['anchor_cos']) if bed_idx is None else bed_idx,
                          dtype=torch.long)
    sheet_stats = np.zeros((V.n, 2))
    Fs = torch.as_tensor(faces, dtype=torch.long)
    vol_of = lambda X_: (X_[Fs[:, 0]] * torch.cross(X_[Fs[:, 1]], X_[Fs[:, 2]], dim=1)).sum() / 6
    vol0 = float(vol_of(X0))
    surf = Surface(nodes * 1000, faces, n=c['n_surf4d'])
    n = V.n
    out = np.zeros((n, len(nodes), 3), np.float32)
    parts_all = np.zeros((n, 3))
    s0 = c['start_frame']
    order = [list(range(s0, n)), list(range(s0, -1, -1))]
    t0 = time.time()
    for seq in order:
        prev = prev2 = None
        for j, k in enumerate(seq):
            o = FrameObs(V, k, body, unknown, junction=junction, erode=c['depth_erode'], n_pts=c['n_pts4d'],
                         n_cov=c['n_cov4d'])
            if sheet_z is not None:
                zs = sheet_z[k]
                o.sheet_m = T(np.isfinite(zs).astype(np.float32))[None, None]
                o.sheet_z = T(np.where(np.isfinite(zs), zs * 1000, 0).astype(np.float32))[None, None]
            init = X0.clone() if prev is None else (prev.clone() if prev2 is None else prev + 0.5 * (prev - prev2))
            X = init.clone().requires_grad_(True)
            opt = torch.optim.Adam([X], lr=c['lr'])
            iters = c['iters_first'] if prev is None else c['iters']
            for it in range(iters):
                opt.zero_grad()
                S, N = surf(X)
                lo, lc, ld = o.losses(S, N, c['hidden_mm'])
                L = c['w_sil'] * lo + c['w_cov'] * c['w_sil'] * lc + c['w_depth'] * ld
                ea, ev = arap(X)
                L = L + c['w_arap'] * ea + c['w_vol'] * ev + c['w_barrier'] * (arap.barrier_sum if c['barrier_sum'] else arap.barrier)
                if c['w_anchor'] > 0:
                    L = L + c['w_anchor'] * ((X[bed] - X0[bed]) ** 2).sum(1).mean()
                if c['w_neck4d'] > 0 and tip_nodes is not None and len(tip_nodes):
                    L = L + c['w_neck4d'] * o.neck_loss(X[tip_nodes].mean(0))
                if prev is not None:
                    L = L + c['w_temp'] * ((X - prev) ** 2).sum(1).mean()
                if c['w_gvol'] > 0:
                    dv = torch.abs(vol_of(X) / vol0 - 1)
                    L = L + c['w_gvol'] * (torch.relu(dv - c['gvol_tol']) / 0.01) ** 2
                if sheet_z is not None and c['w_sheet'] > 0:
                    viol, nfp = o.sheet_violation(S, c['sheet_margin'])
                    L = L + c['w_sheet'] * rho(viol).sum() / max(nfp, 1)
                L.backward()
                opt.step()
            Xd = X.detach()
            parts_all[k] = [float(lo), float(lc), float(ld)]
            if sheet_z is not None:
                with torch.no_grad():
                    viol, nfp = o.sheet_violation(surf(Xd)[0], 0.0)
                sheet_stats[k] = [float((viol > 0).float().mean()) if nfp else 0.0,
                                  float(viol.max()) if nfp else 0.0]
            out[k] = Xd.numpy() / 1000
            prev2, prev = prev, Xd
            if k % 25 == 0:
                log(f'[4d] frame {k}: out {parts_all[k][0]:.3f} cov {parts_all[k][1]:.3f} depth {parts_all[k][2]:.3f} '
                    f'arap {float(ea):.4f} vol {float(ev):.4f} sheet viol frac {sheet_stats[k][0]:.3f} max '
                    f'{sheet_stats[k][1]:.2f} mm ({time.time() - t0:.0f} s)')
    if c['smooth_sigma'] > 0:
        from scipy.ndimage import gaussian_filter1d
        out = gaussian_filter1d(out, c['smooth_sigma'], axis=0, mode='nearest').astype(np.float32)
    return out, parts_all


# ---------------------------------------------------------------- 5. quality
def volume(X, F):
    a, b, cc = X[..., F[:, 0], :], X[..., F[:, 1], :], X[..., F[:, 2], :]
    return np.einsum('...ij,...ij->...i', a, np.cross(b, cc)).sum(-1) / 6


def evaluate(V, P, rest, verts4d, faces, tets, info, log):
    ks = V.keyframes
    q = {}
    for label, VV, name in (('shared_mask', V, 'gallbladder'), ('body_neck_mask', P, 'gb_own')):
        iou = quality.silhouette_iou(VV, verts4d, faces, name, ks)
        bf = quality.boundary_f(VV, verts4d, faces, name, ks)
        dr = quality.depth_residual(VV, verts4d, faces, name, ks) * 1000
        q[label] = dict(iou=quality.summarize(iou), boundary_f=quality.summarize(bf), depth_residual_mm=quality.summarize(dr),
                        per_keyframe=dict(frames=ks, iou=np.round(iou, 4).tolist(), boundary_f=np.round(bf, 4).tolist(),
                                          depth_residual_mm=np.round(dr, 2).tolist()))
        log(f'[quality] {label}: IoU {q[label]["iou"]["mean"]} BF {q[label]["boundary_f"]["mean"]} '
            f'depth {q[label]["depth_residual_mm"]["median"]} mm')
    # multi-view consistency of the static rest shape (no 4D deformation)
    iou_r = quality.silhouette_iou(P, rest, faces, 'gb_own', ks)
    dr_r = quality.depth_residual(P, rest, faces, 'gb_own', ks) * 1000
    q['rest_shape_static'] = dict(note='rest shape held fixed in all keyframes vs body+neck mask (multi-view consistency)',
                                  iou=quality.summarize(iou_r), depth_residual_mm=quality.summarize(dr_r),
                                  per_keyframe_iou=np.round(iou_r, 4).tolist())
    # temporal smoothness
    vel = np.linalg.norm(np.diff(verts4d, axis=0), axis=2) * 1000          # mm / frame
    acc = np.linalg.norm(np.diff(verts4d, n=2, axis=0), axis=2) * 1000
    cen = verts4d.mean(1)
    q['temporal'] = dict(max_vertex_speed_mm_per_frame=round(float(vel.max()), 3),
                         p99_vertex_speed_mm_per_frame=round(float(np.percentile(vel, 99)), 3),
                         mean_vertex_speed_mm_per_frame=round(float(vel.mean()), 3),
                         max_vertex_accel_mm_per_frame2=round(float(acc.max()), 3),
                         centroid_path_mm=round(float(np.linalg.norm(np.diff(cen, axis=0), axis=1).sum() * 1000), 2),
                         centroid_range_mm=(np.ptp(cen, 0) * 1000).round(2).tolist())
    # volume preservation
    v0 = volume(rest, faces)
    vt = volume(verts4d, faces)
    q['volume'] = dict(rest_ml=round(float(v0) * 1e6, 2), ratio_min=round(float((vt / v0).min()), 4),
                       ratio_max=round(float((vt / v0).max()), 4))
    # strain / inversion
    arap = TetARAP(rest * 1000, tets, True)
    good = T(tet_quality(rest * 1000, tets) >= 0.01).bool()
    dets, strains, ninv, ninv_good = [], [], [], []
    for k in range(0, V.n, 5):
        Fm = arap.F(T(verts4d[k] * 1000))
        S = torch.linalg.svdvals(Fm)
        dets.append(float(torch.det(Fm).min()))
        ninv.append(int((torch.det(Fm) <= 0).sum()))
        ninv_good.append(int(((torch.det(Fm) <= 0) & good).sum()))
        strains.append(float((S.max(1)[0] - 1).abs().max()))
    edges = np.unique(np.sort(np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), 1), axis=0)
    smax, s95 = quality.stretch(rest, verts4d, edges)
    q['deformation'] = dict(min_tet_det=round(min(dets), 4), inverted_tet_frames=int(sum(d <= 0 for d in dets)),
                            max_inverted_tets=int(max(ninv)), sampled_frames=len(dets),
                            max_inverted_wellshaped_tets=int(max(ninv_good)),
                            slivers_rest=int((~good).sum()),
                            max_principal_stretch=round(max(strains), 4), edge_stretch_max=round(float(smax.max()), 4),
                            edge_stretch_p95_max=round(float(s95.max()), 4))
    # rigid part of the motion (Procrustes of verts4d onto rest) and the non-rigid remainder
    rd = []
    for k in ks:
        A, B = rest, verts4d[k]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rm = Vt.T @ D @ U.T
        res = np.linalg.norm(B - ((A - ca) @ Rm.T + cb), axis=1) * 1000
        rd.append([np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))), np.linalg.norm(cb - ca) * 1000,
                   np.sqrt((res ** 2).mean()), res.max()])
    rd = np.array(rd)
    q['rigid_vs_nonrigid'] = dict(note='per keyframe: rotation (deg) and centroid shift (mm) of the best rigid fit of '
                                       'verts4d onto rest, rms / max non-rigid remainder (mm)',
                                  rot_deg=rd[:, 0].round(1).tolist(), shift_mm=rd[:, 1].round(1).tolist(),
                                  nonrigid_rms_mm=rd[:, 2].round(2).tolist(), nonrigid_max_mm=rd[:, 3].round(1).tolist())
    q['mesh_health'] = quality.mesh_health(rest, faces, tets)
    q['template'] = info
    log(f'[quality] temporal {q["temporal"]}  volume {q["volume"]}  deformation {q["deformation"]}')
    return q


BANDS = ((0, 62), (62, 130), (130, 251))


def frame_metrics(V, P, verts, faces, k, sheet_verts=None, sheet_faces=None, duct=None):
    """r2s.quality's IoU / boundary F (4 px) / depth residual of one frame from a single render, against the shared
    mask (V) and the body+neck mask (P); plus how much of the mesh lies in front of the sheet (if given)."""
    m, z = quality.render(V, verts, faces, k)
    vis = m & (z < V.depth(k) + 0.006)
    k3 = np.ones((3, 3), np.uint8)
    out = {}
    for lab, VV, real in (('shared', V, V.mask('gallbladder')[k]), ('own', P, P.body[k])):
        valid = ~quality._unknown(VV, k)
        a, b = vis & valid, real & valid
        out[f'iou_{lab}'] = float((a & b).sum() / max((a | b).sum(), 1))
        bm = (vis.astype(np.uint8) - cv2.erode(vis.astype(np.uint8), k3)).astype(bool) & valid
        br = (real.astype(np.uint8) - cv2.erode(real.astype(np.uint8), k3)).astype(bool) & valid
        if bm.sum() and br.sum():
            dr = cv2.distanceTransform((~br).astype(np.uint8), cv2.DIST_L2, 3)
            dm = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 3)
            pr, rc = float((dr[bm] <= 4).mean()), float((dm[br] <= 4).mean())
            out[f'bf_{lab}'] = 2 * pr * rc / max(pr + rc, 1e-9)
        else:
            out[f'bf_{lab}'] = 0.0
        both = m & real & valid
        out[f'depth_{lab}_mm'] = float(np.median(np.abs(z[both] - V.depth(k)[both]))) * 1000 if both.sum() > 30 else np.nan
    if sheet_verts is not None:
        zs = raster_depth(V, sheet_verts, sheet_faces, k)
        zg = raster_depth(V, verts, faces, k)
        ov = np.isfinite(zs) & np.isfinite(zg)
        d = (zs - zg)[ov] * 1000                                     # > 0: gallbladder in front of the sheet
        out['sheet_overlap_px'] = int(ov.sum())
        out['sheet_front_frac'] = float((d > 0.5).mean()) if ov.any() else 0.0
        out['sheet_front_max_mm'] = float(d.max()) if ov.any() else 0.0
    if duct is not None:                                             # duct mask pixels the mesh covers
        out['duct_covered'] = float((m & duct).sum() / max(duct.sum(), 1))
        out['duct_covered_visible'] = float((vis & duct).sum() / max(duct.sum(), 1))
    return out


def neck_tip_track(V, verts4d, tip_idx, target):
    """Per frame: camera depth (mm) of the neck tip (mean of tip_idx vertices), of the duct's proximal end (target,
    video depth at the junction), their 3D distance (mm) and image distance (px)."""
    tip = verts4d[:, tip_idx].mean(1)
    out = np.zeros((V.n, 4))
    for k in range(V.n):
        q1, z1 = V.project(tip[k][None], k)
        q2, z2 = V.project(target[k][None], k)
        out[k] = z1[0] * 1000, z2[0] * 1000, np.linalg.norm(tip[k] - target[k]) * 1000, np.linalg.norm(q1 - q2)
    return out


def band_report(c, body, unknown, faces, verts4d, ver, log, step=3, ref='v10', duct=None):
    """Per frame band (pan 0-62, 62-130, 130-250): the reference version and this one through the clip cameras and
    the refined cameras (every `step`-th frame, keyframes and in-between frames), sheet penetration, duct coverage."""
    cams = {'clip': None}
    if c['cams']:
        cams['refined'] = str(ROOT / c['cams'])
    r = np.load(OUT / ref / 'model.npz')
    models = {ref: (r['verts4d'], r['faces']), ver: (verts4d, faces)}
    mem = np.load(ROOT / c['membrane']) if c['membrane'] else None
    frames = list(range(0, 251, step))
    rep, per = {}, {}
    for cl, cp in cams.items():
        V = views.load('chole_a', 'sift', cams=cp)
        P = OwnMasks(V, body, unknown)
        for mv, (X, FF) in models.items():
            rows = [frame_metrics(V, P, X[k], FF, k, None if mem is None else mem['verts4d'][k],
                                  None if mem is None else mem['faces'], None if duct is None else duct[k])
                    for k in frames]
            keys = rows[0].keys()
            arr = {kk: np.array([r[kk] for r in rows], float) for kk in keys}
            per[f'{cl}/{mv}'] = arr
            fr = np.array(frames)
            out = {}
            for a, b in BANDS:
                sel = (fr >= a) & (fr < b)
                out[f'{a}-{b - 1}'] = {kk: round(float(np.nanmedian(v[sel]) if 'depth' in kk or 'max' in kk
                                                      else np.nanmean(v[sel])), 4) for kk, v in arr.items()}
            kf = fr % 10 == 0
            out['keyframes_iou_shared'] = round(float(arr['iou_shared'][kf].mean()), 4)
            out['between_keyframes_iou_shared'] = round(float(arr['iou_shared'][~kf].mean()), 4)
            rep[f'{cl}_cams/{mv}'] = out
            b = out
            log(f'[bands] {cl} cams, {mv}: ' + ' | '.join(
                f"{k_}: IoU {b[k_]['iou_shared']:.3f}/{b[k_]['iou_own']:.3f} BF {b[k_]['bf_shared']:.3f}/{b[k_]['bf_own']:.3f} "
                f"depth {b[k_]['depth_shared_mm']:.2f} mm" + (f" sheet-front {b[k_]['sheet_front_frac']:.3f}" if mem is not None else '')
                + (f" duct-covered {b[k_]['duct_covered']:.3f}" if duct is not None else '')
                for k_ in ('0-61', '62-129', '130-250')))
    np.savez_compressed(OUT / ver / 'work' / 'bands.npz', frames=np.array(frames),
                        **{k.replace('/', '__') + '__' + kk: v for k, arr in per.items() for kk, v in arr.items()})
    rep['note'] = (f'every {step}rd frame; mean IoU / BF, median depth residual and max sheet values per band; '
                   'shared = V.mask("gallbladder"), own = masks.npz body; sheet_front_frac = share of pixels where '
                   f'both meshes project with the gallbladder > 0.5 mm in front of the membrane sheet ({c["membrane"]}); '
                   'duct_covered = share of the ducts track duct-mask pixels inside the mesh silhouette (_visible: '
                   'after the 6 mm visibility test); own masks = this version\'s masks.npz for both models')
    return rep


def views3d(rest, faces, path, V, attach=()):
    """Rest shape from 3 directions (orthographic, flat shaded, painter's order): the mean scope view, from the side
    (scope view turned 90 deg about world z) and from above (world -z). Attachment vertices as coloured dots."""
    X = rest * 1000
    c = X.mean(0)
    view = np.mean(V.R[:, 2], 0)
    view /= np.linalg.norm(view)
    def frame(z):
        z = z / np.linalg.norm(z)
        up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z])          # rows: right, down, forward
    side = np.array([-view[1], view[0], 0.0])
    dirs = [('scope view', frame(view)), ('side', frame(side)), ('top (world -z)', frame(np.array([0, 0, -1.0])))]
    S, pad = 420, 30
    r = np.abs(X - c).max()
    tiles = []
    for name, Rv in dirs:
        img = np.full((S, S, 3), 245, np.uint8)
        P = (X - c) @ Rv.T
        q = (P[:, :2] / r * (S / 2 - pad) + S / 2).astype(np.int32)
        tri = P[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        light = np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])
        shade = 0.25 + 0.75 * np.abs(nrm @ light)
        order = np.argsort(-tri[:, :, 2].mean(1))
        for fi in order:
            col = (np.array([70, 110, 40]) * shade[fi] + 20).clip(0, 255)
            cv2.fillConvexPoly(img, q[faces[fi]], tuple(int(a) for a in col), cv2.LINE_AA)
        import trimesh
        vn = trimesh.Trimesh(X, faces, process=False).vertex_normals @ Rv.T
        for lab, idx, colr in attach:
            for i in idx:
                if vn[i, 2] < 0:                              # dots on the side facing this view only
                    cv2.circle(img, tuple(int(a) for a in q[i]), 2, colr, -1)
        cv2.putText(img, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        bar = int(10 / r * (S / 2 - pad))
        cv2.line(img, (10, S - 12), (10 + bar, S - 12), (0, 0, 0), 2)
        cv2.putText(img, '1 cm', (14 + bar, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(img)
    out = np.concatenate(tiles, 1)
    y = 40
    for lab, idx, colr in attach:
        cv2.putText(out, f'{lab} ({len(idx)})', (S * 3 - 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 1, cv2.LINE_AA)
        y += 18
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def attachments(V, rest, faces, nodes_n, sheet, neck_tip=None):
    """Suggested attachment vertices (indices into rest_verts):
    'backdrop'   surface vertices facing away from all cameras (normal . view > 0.35): the hidden back side lying on
                 the liver bed;
    'ducts'      neck end: surface vertices within 6 mm of the neck tip (the end the duct continues from);
    'membrane'   front surface vertices that project into the tented sheet's footprint in frame 150 (where the sheet
                 leaves the body)."""
    import trimesh
    vn = trimesh.Trimesh(rest, faces, process=False).vertex_normals
    surf = np.unique(faces)
    back = bed_vertices(V, rest, faces, 0.35)
    # neck tip: extreme surface vertex along the rest shape's long axis towards the observed duct (to the right in frame 150)
    c = rest[surf].mean(0)
    _, _, ax = np.linalg.svd(rest[surf] - c, full_matrices=False)
    a = ax[0] if (V.project(c + ax[0] * 0.01, 150)[0][0] > V.project(c, 150)[0][0]) else -ax[0]
    s = (rest[surf] - c) @ a
    tip = rest[surf[np.argmax(s)]] if neck_tip is None else np.asarray(neck_tip)
    neck = surf[np.linalg.norm(rest[surf] - tip, axis=1) < 0.006]
    q, z = V.project(rest[surf], 150)
    u = np.clip(q[:, 0].astype(int), 0, V.W - 1)
    v = np.clip(q[:, 1].astype(int), 0, V.H - 1)
    inside = (q[:, 0] >= 0) & (q[:, 0] < V.W) & (q[:, 1] >= 0) & (q[:, 1] < V.H)
    fr = np.einsum('ij,ij->i', vn[surf], rest[surf] - V.pos[150])
    sheet_v = surf[inside & (fr < 0) & sheet[150][v, u]]
    idx = np.r_[back, neck, sheet_v]
    to = np.array(['backdrop'] * len(back) + ['ducts'] * len(neck) + ['membrane'] * len(sheet_v))
    return idx, to, a


# ---------------------------------------------------------------- MuJoCo check
def mujoco_check(rest, tets, bed_idx, push=None, bed_target=None, T_end=0.3, young=1200.0, poisson=0.45, log=None):
    """Quick MuJoCo test of the tet mesh with scene4d's gallbladder settings (dim-3 flex, Euler, ts 6.25e-5 s, vertex
    bodies on slide joints, bed vertices on 20 N/m springs): `push` = (point, direction): a capsule (r 2.5 mm) whose
    tip starts 1 mm before `point` moves 6 mm along `direction` (5 mm into the tissue) over T_end; `bed_target` =
    bed vertex positions the bed springs move to (linear ramp over T_end). Returns inverted-tet counts."""
    import mujoco
    X = np.asarray(rest, float)
    bed = set(int(i) for i in bed_idx)
    bodies = []
    for i, p in enumerate(X):
        k = 20.0 if i in bed else 0.0
        j = ''.join(f'<joint type="slide" axis="{a}" stiffness="{k}"/>' for a in ('1 0 0', '0 1 0', '0 0 1'))
        bodies.append(f'<body name="v{i}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" gravcomp="1">{j}'
                      f'<inertial pos="0 0 0" mass="6e-5" diaginertia="1e-9 1e-9 1e-9"/></body>')
    probe = ''
    if push is not None:
        pt, dr = np.asarray(push[0], float), np.asarray(push[1], float) / np.linalg.norm(push[1])
        z = dr
        x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [0, 1.0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, np.stack([x, y, z], 1).ravel())
        start = pt - dr * 0.001 - dr * 0.0025
        probe = (f'<body name="probe" mocap="true" pos="{start[0]:.6f} {start[1]:.6f} {start[2]:.6f}" '
                 f'quat="{qt[0]:.6f} {qt[1]:.6f} {qt[2]:.6f} {qt[3]:.6f}"><geom type="capsule" fromto="0 0 -0.02 0 0 0" '
                 f'size="0.0025" rgba="0.8 0.8 0.8 1"/></body>')
    el = ' '.join(' '.join(map(str, t)) for t in tets)
    xml = f"""<mujoco><option timestep="6.25e-5" integrator="Euler" gravity="0 0 -9.81"/>
<worldbody>{''.join(bodies)}{probe}</worldbody>
<deformable><flex name="gb" dim="3" radius="0.0004" body="{' '.join(f'v{i}' for i in range(len(X)))}"
 vertex="{' '.join('0 0 0' for _ in X)}" element="{el}"><elasticity young="{young}" poisson="{poisson}" damping="0.002"/>
<contact condim="3" friction="0.3" solref="0.004 1" selfcollide="none"/></flex></deformable></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bed_l = np.array(sorted(bed))
    adr = np.array([m.jnt_qposadr[m.body(f'v{i}').jntadr[0]] for i in bed_l]) if len(bed_l) else np.zeros(0, int)
    idx = (adr[:, None] + np.arange(3)[None]).ravel()
    disp = (np.asarray(bed_target, float) - X[bed_l]) if bed_target is not None else None
    T_ = np.asarray(tets)
    def n_inv():
        P = d.flexvert_xpos[T_]
        v = np.einsum('ij,ij->i', np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])
        return int((v <= 0).sum()), bool(np.isfinite(P).all())
    steps = int(round(T_end / m.opt.timestep))
    counts, t0 = [], time.time()
    mujoco.mj_forward(m, d)
    s0 = None
    for st in range(steps):
        f = (st + 1) / steps
        if push is not None:
            d.mocap_pos[0] = start + dr * 0.006 * f
        if disp is not None:
            m.qpos_spring[idx] = (disp * f).ravel()
        mujoco.mj_step(m, d)
        if st % 80 == 0 or st == steps - 1:
            c, ok = n_inv()
            counts.append(c)
            if not ok:
                break
    # settle 0.1 s at the end position
    for st in range(int(0.1 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    c_end, ok = n_inv()
    out = dict(max_inverted=int(max(counts)), inverted_end_push=int(counts[-1]), inverted_after_settle=c_end,
               finite=ok, n_tets=len(T_), seconds=round(time.time() - t0, 1))
    if log:
        log(f'[mujoco] {out}')
    return out


def mujoco_report(c, nodes, tets, faces, idx, to, verts4d, log, ref='v11', frames=(60, 180, 250)):
    """mujoco_check for the reference version and this one: probe capsule pushed 5 mm into the surface vertex
    nearest the interaction track's probe tip (along its shaft) at the given frames' probe positions (rest shapes),
    and bed springs moved to frame 0 of the 4D."""
    I = np.load(ROOT / 'outputs/iter/tissues/interaction/v06/model.npz')
    r = np.load(OUT / ref / 'model.npz')
    models = {ref: (r['rest_verts'].astype(float), r['tets'], r['faces'], r['attach_idx'], r['attach_to'], r['verts4d']),
              'this': (nodes, tets, faces, idx, to, verts4d)}
    out = {}
    for nm, (N, Tt, F_, ai, at, X4) in models.items():
        bed = ai[at == 'backdrop']
        su = np.unique(F_)
        res = {}
        for k in frames:
            tip, sh = I['probe_right_tip'][k], I['probe_right_shaft'][k]
            j = su[np.argmin(np.linalg.norm(N[su] - tip, axis=1))]
            res[f'push_frame{k}'] = mujoco_check(N, Tt, bed, push=(N[j], sh))
        res['bed_to_frame0'] = mujoco_check(N, Tt, bed, bed_target=X4[0][bed])
        res['tets'] = tet_report(N, Tt)
        out[nm] = res
        log(f'[mujoco] {nm}: ' + ', '.join(f"{k_}: max {v_['max_inverted']} end {v_['inverted_after_settle']}"
                                           for k_, v_ in res.items() if k_ != 'tets'))
    out['note'] = ('young 1200 Pa, poisson 0.45, damping 0.002, ts 6.25e-5 s Euler, vertex mass 6e-5 kg, bed springs '
                   '20 N/m; push: capsule r 2.5 mm, tip from 1 mm outside to 5 mm into the surface in 0.3 s, then 0.1 s '
                   'hold; counts = inverted tets (signed volume <= 0)')
    return out


# ---------------------------------------------------------------- main
def build(ver):
    c = cfg_of(ver)
    d = OUT / ver
    (d / 'work').mkdir(parents=True, exist_ok=True)
    logf = open(d / 'work' / 'log.txt', 'w')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()
    log(f'[{ver}] config {json.dumps(c)}')
    t0 = time.time()
    V = views.load('chole_a', 'sift', cams=str(ROOT / c['cams']) if c['cams'] else None)
    # masks: reuse the sheet SAM masks of an earlier version when present (same prompts)
    cache = None
    for prev in sorted(OUT.glob('v*/masks.npz')):
        cache = prev
    if cache is None and (OUT / 'v01/work/sheet_try.npz').exists():
        cache = OUT / 'v01/work/sheet_try.npz'
    negative = None
    if c['duct_masks']:
        dm = np.load(ROOT / c['duct_masks'])
        negative = dm['duct'] | dm['strands']
    body, unknown, sheet = make_masks(V, c['apex_px'], cache, negative)
    np.savez_compressed(d / 'masks.npz', body=body, unknown=unknown, sheet=sheet)
    log(f'[masks] body area per keyframe {body.sum((1, 2))[::10].tolist()} ({time.time() - t0:.0f} s)')
    P = OwnMasks(V, body, unknown)

    junction = neck_junction(V, body) if not c['neck_target'] else \
        np.load(ROOT / c['neck_target'])['duct_centerline4d'][:, 0].astype(float)
    log(f'[masks] body/duct junction found in {int(np.isfinite(junction[:, 0]).sum())} frames')
    if c['rest_from']:
        src = np.load(OUT / c['rest_from'] / 'model.npz')
        nodes, tets, faces = src['rest_verts'].astype(float), src['tets'], src['faces']
        info = json.loads((OUT / c['rest_from'] / 'quality.json').read_text())['template']
        info['rest_from'] = c['rest_from']
        Xpose = None
        log(f'[rest] rest shape and tets from {c["rest_from"]}')
    else:
        Xr, F0, Xpose, info = fit_rest(V, body, unknown, c, log, junction)
        if c['tet_mode'] == 'bcc':
            nodes, tets, faces = bcc_for_count(Xr, F0, c['bcc_nodes'], log)
        else:
            nodes, tets, faces = tetrahedralize(Xr, F0, c['surf_faces'], c['cells'], c['tet_snap'], c['tetgen_q'],
                                                c['tet_mode'])
    tq = tet_quality(nodes * 1000, tets)
    log(f'[tets] quality vol/maxedge^3: median {np.median(tq):.3f}, < 0.01: {(tq < 0.01).mean() * 100:.1f} %, '
        f'min {tq.min():.5f}')
    log(f'[tets] {len(nodes)} nodes, {len(tets)} tets, {len(faces)} surface faces, degenerate {organ.degenerate(nodes, tets)}')
    # template size bookkeeping
    import trimesh
    V0, F0t = organ.load_shape('template', 'gallbladder', faces=3000)
    mt = trimesh.Trimesh(V0, F0t)
    mr = trimesh.Trimesh(nodes, faces)
    ext_r = np.ptp((nodes[np.unique(faces)] - nodes.mean(0)) @ np.linalg.svd(nodes - nodes.mean(0), full_matrices=False)[2].T, 0)
    from scipy.spatial import cKDTree
    info.update(template_extent_cm=(np.ptp(V0, 0) * 100).round(2).tolist(), rest_extent_cm=(ext_r * 100).round(2).tolist(),
                template_volume_ml=round(float(mt.volume) * 1e6, 2), rest_volume_ml=round(float(mr.volume) * 1e6, 2))
    if Xpose is not None:
        d_pose = cKDTree(Xpose).query(trimesh.sample.sample_surface(mr, 5000, seed=0)[0])[0]
        info.update(rest_vs_scaled_template_mm=dict(mean=round(float(d_pose.mean()) * 1000, 2),
                                                    max=round(float(d_pose.max()) * 1000, 2)))
    log(f'[rest] {info}')

    sheet_z = None
    if c['membrane']:
        t1 = time.time()
        sheet_z = sheet_depths(V, c['membrane'])
        log(f'[sheet] membrane depth rasterised in {time.time() - t1:.0f} s')
    bed_idx = None
    if c['rest_from']:                  # identical indexing / attachments as the source version
        idx, to, axis = src['attach_idx'], src['attach_to'], src['long_axis_to_neck']
        bed_idx = idx[to == 'backdrop']
    verts4d, parts = fit_4d(V, body, unknown, nodes, tets, faces, c, log, junction, np.array(info['neck_tip']),
                            bed_idx, sheet_z)
    assert np.isfinite(verts4d).all()
    if not c['rest_from']:
        idx, to, axis = attachments(V, nodes, faces, len(nodes), sheet, info.get('neck_tip'))
    np.savez_compressed(d / 'model.npz', rest_verts=nodes.astype(np.float32), faces=faces.astype(np.int32),
                        tets=tets.astype(np.int32), verts4d=verts4d, attach_idx=idx.astype(np.int32), attach_to=to,
                        fit_parts_mm=parts.astype(np.float32), long_axis_to_neck=axis)
    q = evaluate(V, P, nodes, verts4d, faces, tets, info, log)
    if c['cams'] or c['membrane']:
        ref = c.get('band_ref', 'v10')
        q['bands'] = band_report(c, body, unknown, faces, verts4d, ver, log, ref=ref,
                                 duct=None if negative is None else np.load(ROOT / c['duct_masks'])['duct'])
    if c['neck_target']:
        tgt = np.load(ROOT / c['neck_target'])['duct_centerline4d'][:, 0].astype(float)
        Vr = views.load('chole_a', 'sift', cams=str(ROOT / c['cams']) if c['cams'] else None)
        rows = {}
        for nm, X4, R0, F_, tipc in ((c['band_ref'], *[np.load(OUT / c['band_ref'] / 'model.npz')[k_] for k_ in
                                                        ('verts4d', 'rest_verts', 'faces')],
                                      json.loads((OUT / c['band_ref'] / 'quality.json').read_text())['template']['neck_tip']),
                                     (ver, verts4d, nodes, faces, info['neck_tip'])):
            su = np.unique(F_)
            ti = su[np.linalg.norm(R0[su] - np.asarray(tipc), axis=1) < 0.003]
            if not len(ti):
                ti = su[[np.argmin(np.linalg.norm(R0[su] - np.asarray(tipc), axis=1))]]
            tr = neck_tip_track(Vr, X4, ti, tgt)
            rows[nm] = {f'{a}-{b - 1}': dict(tip_depth_mm=round(float(np.median(tr[a:b, 0])), 2),
                                             junction_depth_mm=round(float(np.median(tr[a:b, 1])), 2),
                                             tip_minus_junction_depth_mm=dict(
                                                 median=round(float(np.median(tr[a:b, 0] - tr[a:b, 1])), 2),
                                                 min=round(float((tr[a:b, 0] - tr[a:b, 1]).min()), 2),
                                                 max=round(float((tr[a:b, 0] - tr[a:b, 1]).max()), 2)),
                                             dist3d_mm_median=round(float(np.median(tr[a:b, 2])), 2),
                                             dist_px_median=round(float(np.median(tr[a:b, 3])), 1))
                        for a, b in BANDS}
            log(f'[neck] {nm}: ' + ' | '.join(f"{k_}: tip-junction depth {v_['tip_minus_junction_depth_mm']['median']:+.1f} mm, "
                                              f"3D {v_['dist3d_mm_median']:.1f} mm, {v_['dist_px_median']:.0f} px"
                                              for k_, v_ in rows[nm].items()))
        q['neck_tip_vs_duct_junction'] = dict(note='neck tip = mean of surface vertices within 3 mm of the rest neck tip; '
                                                   'junction = ducts v10 duct_centerline4d[:, 0] (video depth); '
                                                   'camera depth along the optical axis, refined cameras; negative = '
                                                   'tip nearer the scope', **rows)
    if c.get('mujoco_check'):
        q['mujoco_check'] = mujoco_report(c, nodes, tets, faces, idx, to, verts4d, log)
    q['attach_counts'] = {k: int((to == k).sum()) for k in np.unique(to)}
    q['runtime_s'] = round(time.time() - t0, 1)
    q['metric_notes'] = {
        'shared_mask': 'r2s.quality against V.mask("gallbladder") (includes the tent apex and some strand pixels)',
        'body_neck_mask': 'same metrics against masks.npz body, unknown = instruments + strands + tent apex',
        'temporal': 'vertex speed / acceleration of verts4d in mm per frame (25 fps)',
        'volume': 'closed-surface volume of verts4d relative to the rest shape',
        'deformation': 'tet deformation gradients of verts4d vs rest (det <= 0 = inverted), edge stretch ratios',
        'template': 'fitted scale per template principal axis, extents, volume, rest-vs-(pose+scale template) distance'}
    (d / 'quality.json').write_text(json.dumps(q, indent=1))
    quality.contact_sheet(V, [('gallbladder ' + ver, verts4d, faces, (60, 255, 60))], V.keyframes, d / 'sheet.jpg')
    quality.contact_sheet(P, [('rest (static) ' + ver, nodes, faces, (255, 200, 0))], V.keyframes[::2], d / 'work' / 'rest_sheet.jpg')
    attach = [(lab, idx[to == lab], cc) for lab, cc in (('backdrop', (40, 90, 220)), ('ducts', (220, 40, 40)),
                                                        ('membrane', (240, 160, 0)))]
    views3d(nodes, faces, d / 'views3d.jpg', V, attach)
    log(f'[{ver}] done in {time.time() - t0:.0f} s')
    return q


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'v01')
