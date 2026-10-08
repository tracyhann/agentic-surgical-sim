"""The deformable organ, four ways (the experiment's conditions). All produce the same structure: a tetrahedral mesh
(X, tets), per-vertex frame-0 pixel coordinates (texture), anchored vertices (attached to the bed by springs) and
pinned vertices (cut by the image border).

  measured      the measured front surface (depth over the amodal mask), extruded backwards by a thickness inferred
                from the silhouette                                                        [the 3D pipeline so far]
  primitive     an ellipsoid fitted to the same observations (pose + 3 radii)             [what a coding agent builds]
  template      an organ template (BodyParts3D) fitted by pose + anisotropic scale
  template_fit  the template, then bent by a smooth free-form deformation to match the observations

Fitting uses frame-0 observations: visible organ points at their measured depth, and the amodal silhouette
(template parts that project inside the picture must stay inside it; the silhouette must be covered). Parts outside
the picture are free: the template is the only source for them.

Multi-keyframe variants (experiment, not default conditions) add the same kinds of residuals from later keyframes,
seen through their own cameras (instrument / strand pixels count as unknown):
  template_mv   rigid: only keyframes where the organ still matches its frame-0 outline
  template_mvd  deformation-compensated: the template at keyframe k is displaced by the motion of the existing
                'template' simulation at k (nearest rest vertices), so keyframes of a pulled organ can be used
"""
import numpy as np
import cv2
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.transform import Rotation as Rot
from .config import TEMPLATES


# ---------------------------------------------------------------- common
def boundary_faces(tets):
    faces = {}
    for t in tets:
        for f in ((t[0], t[2], t[1]), (t[0], t[1], t[3]), (t[1], t[2], t[3]), (t[0], t[3], t[2])):
            key = tuple(sorted(f))
            faces[key] = None if key in faces else f
    return np.array([f for f in faces.values() if f is not None], int)


def orient_tets(X, tets):
    out = []
    for t in tets:
        P = X[list(t)]
        if np.linalg.det(np.stack([P[1] - P[0], P[2] - P[0], P[3] - P[0]])) < 0:
            t = (t[0], t[2], t[1], t[3])
        out.append(tuple(int(i) for i in t))
    return np.array(out, int)


def raster_depth(cv, X, faces, R=None, f=None):
    """Nearest and farthest depth of a closed surface per frame pixel (+inf / -inf where it does not project)."""
    cam = cv.cam
    q, z = cam.project(X, cv.R0 if R is None else R, cv.f0 if f is None else f)
    near = np.full((cam.H, cam.W), np.inf, np.float32)
    far = np.full((cam.H, cam.W), -np.inf, np.float32)
    for t in faces:
        poly = q[t].astype(np.int32)
        if (poly[:, 0].max() < 0) or (poly[:, 0].min() >= cam.W) or (poly[:, 1].max() < 0) or (poly[:, 1].min() >= cam.H):
            continue
        tmp = np.zeros((cam.H, cam.W), np.uint8)
        cv2.fillConvexPoly(tmp, poly, 1)
        sel = tmp.astype(bool)
        zt = float(z[t].mean())
        near[sel] = np.minimum(near[sel], zt)
        far[sel] = np.maximum(far[sel], zt)
    return near, far


def finish(cv, X, tets, kind, pinned=None, anchors=None):
    """Common fields: pixel coordinates (texture), surface, anchors (back half along the view, or out of view)."""
    tets = orient_tets(X, tets)
    faces = boundary_faces(tets)
    q, z = cv.project(X)
    if anchors is None:
        near, far = raster_depth(cv, X, faces)
        u = np.clip(q[:, 0].astype(int), 0, cv.cam.W - 1)
        v = np.clip(q[:, 1].astype(int), 0, cv.cam.H - 1)
        inside = (q[:, 0] >= 0) & (q[:, 0] < cv.cam.W) & (q[:, 1] >= 0) & (q[:, 1] < cv.cam.H)
        mid = 0.5 * (near[v, u] + far[v, u])
        anchors = ~inside | (z > mid + 1e-4)
    pinned = np.zeros(len(X), bool) if pinned is None else pinned
    px = np.clip(q, -cv.M + 1, np.array([cv.cam.W, cv.cam.H]) + cv.M - 2)
    return dict(kind=kind, X=X, tets=tets, faces=faces, px=px, anchors=anchors & ~pinned, pinned=pinned)


# ---------------------------------------------------------------- measured
def sheet(mask, step):
    """Triangulated sheet of image points over a mask (grid + subsampled outline, slivers removed)."""
    H, W = mask.shape
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pts = [(u, v) for v in range(4, H - 2, step) for u in range(int(step / 2) * ((v // step) % 2) + 2, W - 2, step) if inner[v, u]]
    cnt = max(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
    rim = [tuple(map(int, p)) for p in cnt[::max(1, len(cnt) // 110)]]
    G = np.array(pts, float)
    rim = [r for r in rim if np.min(np.linalg.norm(G - r, axis=1)) > 0.6 * step]
    P = np.array(pts + rim, float)
    is_rim = np.r_[np.zeros(len(pts), bool), np.ones(len(rim), bool)]
    from .scene import min_angle
    T = np.array([t for t in Delaunay(P).simplices
                  if mask[int(np.clip(P[t].mean(0)[1], 0, H - 1)), int(np.clip(P[t].mean(0)[0], 0, W - 1))] and min_angle(P[t]) > 12])
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    return P[used], remap[T], is_rim[used]


def measured(cv, obs, step=10):
    am, z0, thick = obs['organ_mask'], obs['z0'], obs['thick']
    H, W = am.shape
    P, tri, rim = sheet(am, step)
    u, v = np.clip(P[:, 0].astype(int), 0, W - 1), np.clip(P[:, 1].astype(int), 0, H - 1)
    Xf = cv.unproject(P[:, 0], P[:, 1], z0[v, u])
    ray = (Xf - cv.cam.pos) / np.linalg.norm(Xf - cv.cam.pos, axis=1, keepdims=True)
    Xb = Xf + ray * thick[v, u][:, None]
    N = len(P)
    tets = []
    for t in tri:
        a, b, c = sorted(int(i) for i in t)
        tets += [(a, b, c, c + N), (a, b, b + N, c + N), (a, a + N, b + N, c + N)]
    on_border = (P[:, 0] <= 3) | (P[:, 0] >= W - 4) | (P[:, 1] <= 3) | (P[:, 1] >= H - 4)
    pinned = np.r_[rim & on_border, rim & on_border]
    anchors = np.r_[np.zeros(N, bool), np.ones(N, bool)]          # the back layer rests on the bed
    out = finish(cv, np.r_[Xf, Xb], np.array(tets), 'measured', pinned=pinned, anchors=anchors)
    out['px'] = np.r_[P, P]                                        # front and back share the projected texture
    return out


# ---------------------------------------------------------------- shape fitting
def load_shape(kind, template=None, faces=900):
    """Unit-free shape in its own principal frame (centroid at 0, axes = principal axes, metres for templates)."""
    if kind == 'primitive':
        m = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
    else:
        m = trimesh.load(TEMPLATES / f'{template}.stl')
        m.apply_scale(1e-3)                                    # BodyParts3D is in millimetres
        m = max(m.split(only_watertight=False), key=lambda b: len(b.faces))
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    V = m.vertices - m.vertices.mean(0)
    _, _, axes = np.linalg.svd(V, full_matrices=False)
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return V @ axes.T, np.asarray(m.faces)


class ShapeFit:
    """Pose (rotation, translation), anisotropic scale and an optional 4x4x4 trilinear free-form deformation of a
    shape, fitted to visible points (metres) and an amodal silhouette (frame-0 pixels)."""

    def __init__(self, cv, V, F, pts, sil, z_med, zvis=None, n_samples=2500, seed=0):
        self.cv, self.V, self.F = cv, V, F
        self.zvis = zvis                  # measured depth of what is visible at frame 0 (occlusion test)
        rng = np.random.default_rng(seed)
        area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        self.sf = rng.choice(len(F), n_samples, p=area / area.sum())
        r = rng.random((n_samples, 2))
        flip = r.sum(1) > 1
        r[flip] = 1 - r[flip]
        self.sb = np.c_[1 - r.sum(1), r]
        self.pts = pts
        self.sil = sil
        H, W = sil.shape
        self.dt_out = cv2.distanceTransform((~sil).astype(np.uint8), cv2.DIST_L2, 5)
        cnt = max(cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0, :]
        cnt = cnt[(cnt[:, 0] > 3) & (cnt[:, 0] < W - 4) & (cnt[:, 1] > 3) & (cnt[:, 1] < H - 4)]
        self.contour = cnt[::max(1, len(cnt) // 200)].astype(float)
        self.px2m = z_med / cv.f0
        self.lo, self.hi = V.min(0), V.max(0)
        self.ffd = None
        self.views = []

    # deformation ---------------------------------------------------------
    def lattice_offsets(self, V, off):
        """Trilinear interpolation of 4x4x4 lattice offsets over the shape's bounding box."""
        t = (V - self.lo) / (self.hi - self.lo + 1e-9) * 3
        i0 = np.clip(np.floor(t).astype(int), 0, 2)
        w = t - i0
        out = np.zeros_like(V)
        L = off.reshape(4, 4, 4, 3)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * (w[:, 2] if dz else 1 - w[:, 2])
                    out += wt[:, None] * L[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out

    def transform(self, x, off=None):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        S = np.exp(x[6:9])
        V = self.V * S
        if off is not None:
            V = V + self.lattice_offsets(self.V, off)
        return V @ R.T + x[3:6]

    # residuals -------------------------------------------------------------
    def residuals(self, Vw, w_sil=2.0):
        F = self.F
        S = (Vw[F[self.sf]] * self.sb[:, :, None]).sum(1)
        nrm = np.cross(Vw[F[:, 1]] - Vw[F[:, 0]], Vw[F[:, 2]] - Vw[F[:, 0]])[self.sf]
        front = np.einsum('ij,ij->i', nrm, S - self.cv.cam.pos) < 0
        Sf = S[front] if front.sum() > 20 else S
        d_pts, _ = cKDTree(Sf).query(self.pts)
        q, _ = self.cv.project(S)
        H, W = self.sil.shape
        inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        r_out = np.zeros(len(S))
        qi = q[inside].astype(int)
        r_out[inside] = self.dt_out[qi[:, 1], qi[:, 0]]
        if self.zvis is not None:         # outside the outline is fine where the visible surface hides it
            zs = (S[inside] - self.cv.cam.pos) @ self.cv.R0[2]
            hidden = zs > self.zvis[qi[:, 1], qi[:, 0]] + 0.008     # the backdrop is drawn 4 mm behind the surface
            r_out[np.nonzero(inside)[0][hidden]] = 0.0
        d_cov, _ = cKDTree(q[inside] if inside.sum() > 10 else q).query(self.contour)
        out = [d_pts, w_sil * self.px2m * r_out / np.sqrt(len(S) / len(self.pts)),
               w_sil * self.px2m * d_cov / np.sqrt(len(self.contour) / len(self.pts) + 1e-9)]
        for v in self.views:
            out.append(v.residuals(S, nrm, self.px2m, len(self.pts), w_sil))
        return np.concatenate(out)

    def fit_pose(self, x0s, scale_prior=0.0, iters=60, min_scale=0.1):
        xs = np.r_[0.3, 0.3, 0.3, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
        f = lambda x: np.r_[self.residuals(self.transform(x)), scale_prior * x[6:9]]

        def run(bounds):
            kw = dict(loss='soft_l1', f_scale=0.003, x_scale=xs)
            if bounds is not None:
                kw['bounds'] = bounds
                clip = lambda x: np.clip(np.asarray(x, float), bounds[0] + 1e-9, bounds[1] - 1e-9)
            else:
                clip = lambda x: x
            best = None
            for x0 in x0s:
                s = least_squares(f, clip(x0), max_nfev=iters, **kw)
                if best is None or s.cost < best.cost:
                    best = s
            return least_squares(f, best.x, max_nfev=200, **kw)
        s = run(None)
        if np.exp(s.x[6:9]).min() < min_scale:
            # a thin strand-like observation can flatten the shape into a disk that cannot be tetrahedralised: refit
            # with the scales bounded (0.1 = 2 mm radius for the primitive, a tenth of anatomical size for templates)
            s = run((np.r_[np.full(6, -np.inf), np.full(3, np.log(min_scale))], np.r_[np.full(6, np.inf), np.full(3, np.log(20.0))]))
        return s.x, float(np.median(np.abs(self.residuals(self.transform(s.x))[:len(self.pts)])))

    def fit_ffd(self, x, smooth=6.0, size=1.5, iters=40):
        """Lattice offsets (in the shape frame, before rotation) on top of a fitted pose and scale."""
        ext = (self.hi - self.lo) * np.exp(x[6:9])
        L = np.zeros((4, 4, 4, 3))

        def reg(off):
            o = off.reshape(4, 4, 4, 3)
            lap = []
            for ax in range(3):
                lap.append(np.diff(o, n=2, axis=ax).ravel())
            return np.r_[smooth * np.concatenate(lap), size * off] / np.sqrt(len(off)) * 0.02 / max(ext.max(), 1e-3)
        f = lambda off: np.r_[self.residuals(self.transform(x, off)), reg(off)]
        s = least_squares(f, L.ravel(), loss='soft_l1', f_scale=0.003, max_nfev=iters, x_scale=0.003)
        return s.x, float(np.median(np.abs(self.residuals(self.transform(x, s.x))[:len(self.pts)])))


class KeyView:
    """Observations of one later keyframe for the multi-keyframe fit: visible organ outline and points through that
    keyframe's camera; instrument / strand pixels are unknown. Optional deformation: displacement of rest positions
    at this keyframe, interpolated from a simulated organ (rest vertices X, displacements D)."""

    def __init__(self, cam, R, f, pos, organ, unknown, Z, weight=1.0, X=None, D=None, n_pts=400, seed=0):
        self.cam, self.R, self.f, self.pos, self.Z, self.w = cam, R, f, pos, Z, weight
        H, W = organ.shape
        allowed = organ | unknown
        self.dt_out = cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 5)
        cs = cv2.findContours(organ.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cnt = max(cs, key=len)[:, 0, :] if cs else np.zeros((0, 2), int)
        near_unknown = cv2.dilate(unknown.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        keep = (cnt[:, 0] > 3) & (cnt[:, 0] < W - 4) & (cnt[:, 1] > 3) & (cnt[:, 1] < H - 4)
        keep[keep] &= ~near_unknown[cnt[keep, 1], cnt[keep, 0]]     # outline against instruments is not the organ's
        cnt = cnt[keep]
        self.contour = cnt[::max(1, len(cnt) // 150)].astype(float)
        ys, xs = np.nonzero(cv2.erode(organ.astype(np.uint8), np.ones((5, 5), np.uint8)))
        idx = np.random.default_rng(seed).choice(len(xs), min(n_pts, len(xs)), replace=False)
        self.pts = cam.unproject(xs[idx], ys[idx], Z[ys[idx], xs[idx]], R, f, pos)
        self.tree = cKDTree(X) if X is not None else None
        self.D = D

    def displace(self, S):
        if self.tree is None:
            return S
        d, i = self.tree.query(S, k=4)
        w = 1.0 / (d + 1e-4)
        return S + (self.D[i] * w[..., None]).sum(1) / w.sum(1, keepdims=True)

    def residuals(self, S, nrm, px2m, n_ref, w_sil):
        Sk = self.displace(S)
        pos = self.cam.pos if self.pos is None else self.pos
        front = np.einsum('ij,ij->i', nrm, Sk - pos) < 0
        Sf = Sk[front] if front.sum() > 20 else Sk
        d_pts, _ = cKDTree(Sf).query(self.pts)
        q, z = self.cam.project(Sk, self.R, self.f, self.pos)
        H, W = self.dt_out.shape
        inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H) & (z > 0)
        r_out = np.zeros(len(Sk))
        qi = q[inside].astype(int)
        r_out[inside] = self.dt_out[qi[:, 1], qi[:, 0]]
        hidden = z[inside] > self.Z[qi[:, 1], qi[:, 0]] + 0.008                  # behind what is visible there
        r_out[np.nonzero(inside)[0][hidden]] = 0.0
        d_cov = cKDTree(q[inside] if inside.sum() > 10 else q).query(self.contour)[0] if len(self.contour) else np.zeros(0)
        r = np.r_[d_pts * np.sqrt(n_ref / len(self.pts)), w_sil * px2m * r_out / np.sqrt(len(Sk) / n_ref),
                  w_sil * px2m * d_cov / np.sqrt(max(len(self.contour), 1) / n_ref)]
        return self.w * r


def keyframe_views(clip, cams, masks, Z, keys, weights, sim=None, fps=None, dt=None):
    """KeyView per keyframe. sim = (rest X (n, 3), flex (steps, >= n, 3)) for the deformation-compensated fit."""
    oi = clip.obj_index(clip.organ)
    hide = [clip.obj_index(i['mask']) for i in clip.instruments] + [clip.obj_index(s) for s in clip.role('strand')]
    cam = clip.camera()
    views = []
    for k, w in zip(keys, weights):
        organ = masks[k, oi] & ~np.any(masks[k, hide], 0)
        if organ.sum() < 300:
            continue
        unknown = cv2.dilate(np.any(masks[k, hide], 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        X = D = None
        if sim is not None:
            X0, flex = sim
            ks = min(int(round(k / fps / dt)), len(flex) - 1)
            X, D = X0, flex[ks, :len(X0)] - X0
        pos = cams['pos'][k] if 'pos' in cams else None
        views.append(KeyView(cam, cams['R'][k], float(cams['f'][k]), pos, organ, unknown, Z[k], w, X, D, seed=k))
    return views


def initial_poses(V, pts, view, scale0):
    """Long axis of the shape along the observed long axis (both ways), 8 rolls; centred behind the visible points."""
    c = pts.mean(0)
    _, _, ax = np.linalg.svd(pts - c, full_matrices=False)
    long_obs = ax[0]
    out = []
    for sgn in (1, -1):
        a = sgn * long_obs
        # rotation taking the shape's x axis (its long axis) onto a
        base = Rot.align_vectors([a], [[1.0, 0, 0]])[0]
        for roll in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            R = Rot.from_rotvec(a * roll) * base
            half_depth = np.abs(((V * np.exp(scale0)) @ R.as_matrix().T) @ view).max()
            t = c + view * half_depth                      # visible points lie on the near side of the shape
            out.append(np.r_[R.as_rotvec(), t, scale0])
    return out


def _tetgen_worker(V, F, q):
    import tetgen
    try:
        nodes, elems = tetgen.TetGen(V, F).tetrahedralize(order=1, mindihedral=8, minratio=2.2, quality=True)[:2]
        q.put((np.asarray(nodes, float), np.asarray(elems, int)))
    except Exception as e:                      # noqa: BLE001
        q.put(e)


def remesh(V, F, faces=650, cells=16, snap=True):
    """Closed, non-self-intersecting surface: voxelise (about `cells` voxels across the thinnest extent), fill,
    marching cubes, light smoothing, decimation."""
    m = trimesh.Trimesh(V, F, process=True)
    pitch = float(np.sort(m.extents)[0]) / cells
    vox = m.voxelized(pitch).fill()
    mc = vox.marching_cubes
    mc.apply_transform(vox.transform)
    trimesh.smoothing.filter_taubin(mc, iterations=10)
    if len(mc.faces) > faces:
        mc = mc.simplify_quadric_decimation(face_count=faces)
    mc.remove_unreferenced_vertices()
    if not snap:
        return np.asarray(mc.vertices, float), np.asarray(mc.faces, np.int32)
    # voxelisation marks every voxel the surface touches: the marching-cubes surface sits up to a voxel outside.
    # Snap the vertices back onto the original surface (keeps the clean topology, restores the shape)
    dense, _ = trimesh.sample.sample_surface(m, 30000, seed=0)
    _, nn = cKDTree(dense).query(mc.vertices)
    V = np.asarray(dense[nn], float)
    # two vertices snapped onto the same sample would leave zero-length edges (degenerate tetrahedra): keep the first,
    # leave the others where marching cubes put them
    _, first = np.unique(nn, return_index=True)
    dup = np.ones(len(nn), bool)
    dup[first] = False
    V[dup] = np.asarray(mc.vertices, float)[dup]
    return V, np.asarray(mc.faces, np.int32)


def to_tets(V, F):
    """Remesh, then TetGen in a child process (a crash there must not take the pipeline down); coarser on failure,
    and as a last resort without snapping onto the surface (snapping can fold faces; up to a voxel larger)."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    for faces, cells, snap in ((450, 14, True), (320, 11, True), (220, 9, True), (450, 14, False), (320, 11, False)):
        Vr, Fr = remesh(V, F, faces, cells, snap)
        q = ctx.Queue()
        p = ctx.Process(target=_tetgen_worker, args=(Vr, Fr, q))
        p.start()
        try:
            r = q.get(timeout=300)              # read before join: a large result blocks the child until it is read
        except Exception:                       # noqa: BLE001  (child crashed or hung)
            r = None
        p.join(10)
        if p.is_alive():
            p.kill()
        if r is not None and not isinstance(r, Exception) and not degenerate(*r):
            return r
    raise RuntimeError('tetrahedralisation failed')


def degenerate(nodes, elems, rel=1e-6):
    """Any (near) zero-volume tetrahedron: the explicit simulation turns those into NaNs at the first step."""
    X, T = nodes, elems
    v = np.abs(np.einsum('ij,ij->i', np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), X[T[:, 3]] - X[T[:, 0]]))
    return bool((v < rel * np.median(v)).any())


def fit_surface(cv, obs, kind, template=None, log=print):
    """The fitted closed surface (V, F) and fit info of a primitive / template / template_fit body."""
    vis = obs['organ_visible']
    z0 = obs['z0']
    ys, xs = np.nonzero(cv2.erode(vis.astype(np.uint8), np.ones((5, 5), np.uint8)))
    idx = np.random.default_rng(0).choice(len(xs), min(600, len(xs)), replace=False)
    pts = cv.unproject(xs[idx], ys[idx], z0[ys[idx], xs[idx]])
    V, F = load_shape('primitive' if kind == 'primitive' else 'template', template)
    fit = ShapeFit(cv, V, F, pts, obs['organ_mask'], float(np.median(z0[vis])), zvis=obs.get('zvis'))
    if kind == 'primitive':
        ext_obs = np.ptp((pts - pts.mean(0)) @ np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2].T, axis=0)
        scale0 = np.log(np.maximum(ext_obs / np.maximum(np.ptp(V, axis=0), 1e-6), 0.2))
        scale0[2] = np.log(max(np.exp(scale0[1]) * 0.8, 0.2))
        prior = 0.0
    else:
        scale0 = np.zeros(3)
        prior = 0.004
    x, err = fit.fit_pose(initial_poses(V, pts, cv.R0[2], scale0), scale_prior=prior)
    info = dict(kind=kind, template=template, scale=np.exp(x[6:9]).round(3).tolist(), pose_fit_median_mm=round(err * 1000, 2))
    Vw = fit.transform(x)
    if kind == 'template_fit':
        off, err2 = fit.fit_ffd(x)
        Vw = fit.transform(x, off)
        info.update(ffd_fit_median_mm=round(err2 * 1000, 2), ffd_max_offset_mm=round(float(np.abs(off).max()) * 1000, 1))
    log(f'[organ] {kind}: {info}')
    return Vw, F, info


def fitted(cv, obs, kind, template=None, log=print, views=None):
    vis = obs['organ_visible']
    z0 = obs['z0']
    ys, xs = np.nonzero(cv2.erode(vis.astype(np.uint8), np.ones((5, 5), np.uint8)))
    idx = np.random.default_rng(0).choice(len(xs), min(600, len(xs)), replace=False)
    pts = cv.unproject(xs[idx], ys[idx], z0[ys[idx], xs[idx]])
    view = cv.R0[2]
    V, F = load_shape('primitive' if kind == 'primitive' else 'template', template)
    fit = ShapeFit(cv, V, F, pts, obs['organ_mask'], float(np.median(z0[vis])), zvis=obs.get('zvis'))
    # initial scale: primitive -> observed extent; template -> anatomical size (scale 1)
    if kind == 'primitive':
        ext_obs = np.ptp((pts - pts.mean(0)) @ np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2].T, axis=0)
        scale0 = np.log(np.maximum(ext_obs / np.maximum(np.ptp(V, axis=0), 1e-6), 0.2))
        scale0[2] = np.log(max(np.exp(scale0[1]) * 0.8, 0.2))
        prior = 0.0
    else:
        scale0 = np.zeros(3)
        prior = 0.004                      # mild pull towards anatomical proportions
    x, err = fit.fit_pose(initial_poses(V, pts, view, scale0), scale_prior=prior)
    if views:                              # multi-keyframe fit, started from the frame-0 fit
        x_f0 = x
        fit.views = views
        x, _ = fit.fit_pose([x_f0], scale_prior=prior)
        fit.views = []
        err = float(np.median(np.abs(fit.residuals(fit.transform(x))[:len(fit.pts)])))
    info = dict(kind=kind, template=template, scale=np.exp(x[6:9]).round(3).tolist(), pose_fit_median_mm=round(err * 1000, 2))
    if views:
        info.update(n_views=len(views), frame0_scale=np.exp(x_f0[6:9]).round(3).tolist(),
                    shift_mm=round(float(np.linalg.norm(x[3:6] - x_f0[3:6])) * 1000, 1))
    Vw = fit.transform(x)
    if kind == 'template_fit':
        off, err2 = fit.fit_ffd(x)
        Vw = fit.transform(x, off)
        info.update(ffd_fit_median_mm=round(err2 * 1000, 2), ffd_max_offset_mm=round(float(np.abs(off).max()) * 1000, 1))
    log(f'[organ] {kind}: {info}')
    nodes, elems = to_tets(Vw, F)
    out = finish(cv, nodes, elems, kind)
    out['fit'] = info
    out['surface_fit'] = (Vw, F)
    return out


def build(cv, obs, kind, template=None, log=print, views=None):
    if kind == 'measured':
        return measured(cv, obs)
    return fitted(cv, obs, kind, template, log, views=views)
