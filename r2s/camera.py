"""Pinhole endoscope camera with a fixed tip: per-frame rotation + zoom (fitted to tracked background points).

World frame: metres, z up. The scope tip sits at `pos` and looks 40 deg down the +y axis in frame 0; camera axes are
the rows of R (x right, y down, z forward), as in OpenCV.
"""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


class Cam:
    def __init__(self, W, H, f, pos=(0.0, 0.0, 0.10), down_deg=40.0):
        self.W, self.H, self.F = int(W), int(H), float(f)
        self.pos = np.asarray(pos, float)
        a = np.radians(down_deg)
        z = np.array([0.0, np.cos(a), -np.sin(a)])
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
        self.R = np.stack([x, np.cross(z, x), z])

    @property
    def fovy(self):
        return float(np.degrees(2 * np.arctan(self.H / 2 / self.F)))

    def unproject(self, u, v, z, R=None, f=None, pos=None):
        R = self.R if R is None else R
        f = self.F if f is None else f
        pos = self.pos if pos is None else pos
        u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
        pc = np.stack([(u - self.W / 2) * z / f, (v - self.H / 2) * z / f, z], -1)
        return pos + pc @ R

    def project(self, p, R=None, f=None, pos=None):
        R = self.R if R is None else R
        f = self.F if f is None else f
        pos = self.pos if pos is None else pos
        pc = (np.asarray(p, float) - pos) @ R.T
        return np.stack([f * pc[..., 0] / pc[..., 2] + self.W / 2, f * pc[..., 1] / pc[..., 2] + self.H / 2], -1), pc[..., 2]

    def mj_quat(self, R):
        """MuJoCo camera quaternion (w x y z) for camera rows R (MuJoCo cameras look along -z with +y up)."""
        import mujoco
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.stack([R[0], -R[1], -R[2]], axis=1).flatten())
        return q


def pose(cams, k):
    """(R, f, pos) of frame k; pos is None (the fixed scope tip) unless the clip's camera translates."""
    pos = cams.get('pos') if isinstance(cams, dict) else None
    return cams['R'][k], cams['f'][k], (None if pos is None else pos[k])


def fill_smooth(x, w=7):
    """Interpolate NaNs per column, then a centred moving average of width w."""
    x = np.array(x, float)
    if x.ndim == 1:
        return fill_smooth(x[:, None], w)[:, 0]
    for k in range(x.shape[1]):
        ok = np.isfinite(x[:, k])
        x[:, k] = np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok, k]) if ok.any() else 0.0
    pad = np.pad(x, ((w // 2, w // 2), (0, 0)), mode='edge')
    return np.stack([np.convolve(pad[:, k], np.ones(w) / w, mode='valid') for k in range(x.shape[1])], 1)


def lk_tracks(frames, exclude, n=400, mask=None):
    """Forward-backward checked Lucas-Kanade tracks of corners in frame 0 (NaN once lost)."""
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    allow = ~exclude if mask is None else (mask & ~exclude)
    p0 = cv2.goodFeaturesToTrack(g[0], n, 0.01, 8, mask=allow.astype(np.uint8) * 255)
    P = np.full((len(g), 0 if p0 is None else len(p0), 2), np.nan, np.float32)
    if p0 is None:
        return P
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


def camera_track(cam, frames, static, refresh=25, static_masks=None):
    """Scope rotation + zoom per frame from KLT tracks on static anatomy. Tracks are re-seeded every `refresh` frames
    (long clips, pans) and chained: each segment is fitted relative to the pose already known at its first frame.
    static: (H, W) mask for frame 0, or `static_masks` (n, H, W) per frame."""
    n = len(frames)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    pars = np.zeros((n, 4))
    err = np.zeros(n)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def rays_of(p, par):
        """Frame-0 camera rays (unit z) of pixels p seen with parameters par."""
        Rk = Rot.from_rotvec(par[:3]).as_matrix()
        fk = cam.F * np.exp(par[3])
        r = np.c_[(p - [cam.W / 2, cam.H / 2]) / fk, np.ones(len(p))]
        return r @ Rk          # camera-k rays expressed in frame-0 camera coordinates

    s = 0
    while s < n - 1:
        e = min(n - 1, s + refresh)
        allow = (static_masks[s] if static_masks is not None else (static if s == 0 else None))
        if allow is None:
            allow = np.ones((cam.H, cam.W), bool)
        p0 = cv2.goodFeaturesToTrack(g[s], 400, 0.01, 8, mask=allow.astype(np.uint8) * 255)
        if p0 is None or len(p0) < 8:
            pars[s + 1:e + 1] = pars[s]
            s = e
            continue
        base = rays_of(p0[:, 0].astype(float), pars[s])
        cur, alive = p0, np.ones(len(p0), bool)
        par = pars[s].copy()
        for k in range(s + 1, e + 1):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(g[k - 1], g[k], cur, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g[k], g[k - 1], nxt, None, **lk)
            alive &= (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - cur, axis=2)[:, 0] < 1.0)
            if static_masks is not None:
                q = nxt[:, 0]
                inside = (q[:, 0] >= 0) & (q[:, 0] < cam.W) & (q[:, 1] >= 0) & (q[:, 1] < cam.H)
                ok_mask = np.zeros(len(q), bool)
                ok_mask[inside] = static_masks[k][q[inside, 1].astype(int), q[inside, 0].astype(int)]
                alive &= ok_mask
            cur = nxt
            ok = alive
            if ok.sum() < 6:
                pars[k] = par
                continue

            def f(x):
                Rk = Rot.from_rotvec(x[:3]).as_matrix()
                q = base[ok] @ Rk.T
                fk = cam.F * np.exp(x[3])
                return (np.c_[fk * q[:, 0] / q[:, 2] + cam.W / 2, fk * q[:, 1] / q[:, 2] + cam.H / 2] - nxt[ok, 0]).ravel()
            sol = least_squares(f, par, loss='huber', f_scale=3.0)
            par = sol.x
            pars[k] = par
            err[k] = float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)))
        s = e
    pars = fill_smooth(pars, 9)
    Rs = np.array([Rot.from_rotvec(x[:3]).as_matrix() @ cam.R for x in pars])
    fs = cam.F * np.exp(pars[:, 3])
    return dict(R=Rs, f=fs, params=pars, fit_err_px=err)


def camera_track_dense(cam, frames, static_masks, grid=8):
    """Scope rotation + zoom from dense optical flow between consecutive frames (robust to blur and fast pans):
    per frame, the incremental rotation and zoom that best explain the flow of static pixels (Huber), chained."""
    n = len(frames)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    vv, uu = np.mgrid[grid // 2:cam.H:grid, grid // 2:cam.W:grid]
    P0 = np.stack([uu.ravel(), vv.ravel()], 1).astype(float)
    Rrel = [np.eye(3)]                      # frame-0 camera coordinates -> frame-k camera coordinates
    fk = [cam.F]
    err = np.zeros(n)
    for k in range(1, n):
        fl = dis.calc(g[k - 1], g[k], None)
        ok = static_masks[k - 1][P0[:, 1].astype(int), P0[:, 0].astype(int)]
        p = P0[ok]
        q = p + fl[p[:, 1].astype(int), p[:, 0].astype(int)]
        inside = (q[:, 0] >= 0) & (q[:, 0] < cam.W) & (q[:, 1] >= 0) & (q[:, 1] < cam.H)
        p, q = p[inside], q[inside]
        if len(p) < 20:
            Rrel.append(Rrel[-1]); fk.append(fk[-1])
            continue
        r = np.c_[(p - [cam.W / 2, cam.H / 2]) / fk[-1], np.ones(len(p))]

        def f(x):
            qq = r @ Rot.from_rotvec(x[:3]).as_matrix().T
            ff = fk[-1] * np.exp(x[3])
            return (np.c_[ff * qq[:, 0] / qq[:, 2] + cam.W / 2, ff * qq[:, 1] / qq[:, 2] + cam.H / 2] - q).ravel()
        sol = least_squares(f, np.zeros(4), loss='huber', f_scale=1.5)
        Rrel.append(Rot.from_rotvec(sol.x[:3]).as_matrix() @ Rrel[-1])
        fk.append(fk[-1] * np.exp(sol.x[3]))
        err[k] = float(np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)))
    pars = np.array([np.r_[Rot.from_matrix(R).as_rotvec(), np.log(f / cam.F)] for R, f in zip(Rrel, fk)])
    pars = fill_smooth(pars, 5)
    Rs = np.array([Rot.from_rotvec(x[:3]).as_matrix() @ cam.R for x in pars])
    return dict(R=Rs, f=cam.F * np.exp(pars[:, 3]), params=pars, fit_err_px=err)


def camera_track_6dof(cam, frames, static_masks, Z, grid=8):
    """Scope rotation AND translation (fixed focal length): frame-to-frame PnP (RANSAC) of static pixels whose 3D
    position comes from the previous frame's metric depth, dense flow for the matches; chained, then smoothed.
    For clips where the scope moves enough for parallax to matter (a rotation + zoom model then fails)."""
    n = len(frames)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    K = np.array([[cam.F, 0, cam.W / 2], [0, cam.F, cam.H / 2], [0, 0, 1.0]])
    vv, uu = np.mgrid[grid // 2:cam.H:grid, grid // 2:cam.W:grid]
    P0 = np.stack([uu.ravel(), vv.ravel()], 1).astype(float)
    T = np.eye(4)                                  # camera-0 coordinates -> camera-k coordinates
    rv, tv, err = [np.zeros(3)], [np.zeros(3)], np.zeros(n)
    for k in range(1, n):
        p = P0[static_masks[k - 1][P0[:, 1].astype(int), P0[:, 0].astype(int)]]
        fl = dis.calc(g[k - 1], g[k], None)
        q = p + fl[p[:, 1].astype(int), p[:, 0].astype(int)]
        z = Z[k - 1][p[:, 1].astype(int), p[:, 0].astype(int)]
        Xc = np.c_[(p[:, 0] - cam.W / 2) * z / cam.F, (p[:, 1] - cam.H / 2) * z / cam.F, z]
        X0 = (np.linalg.inv(T) @ np.c_[Xc, np.ones(len(Xc))].T).T[:, :3]
        ok, r, t, inl = cv2.solvePnPRansac(X0, q, K, None, rvec=cv2.Rodrigues(T[:3, :3])[0], tvec=T[:3, 3:].copy(),
                                           useExtrinsicGuess=True, reprojectionError=2.0, iterationsCount=200)
        if ok and inl is not None and len(inl) > 30:
            R, _ = cv2.Rodrigues(r)
            Tn = np.eye(4)
            Tn[:3, :3], Tn[:3, 3] = R, t[:, 0]
            step = np.linalg.norm((np.linalg.inv(Tn) @ [0, 0, 0, 1])[:3] - (np.linalg.inv(T) @ [0, 0, 0, 1])[:3])
            if step < 0.005:                        # reject jumps > 5 mm per frame
                T = Tn
            pr, _ = cv2.projectPoints(X0[inl[:, 0]], cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)
            err[k] = float(np.median(np.linalg.norm(pr[:, 0] - q[inl[:, 0]], axis=1)))
        rv.append(cv2.Rodrigues(T[:3, :3])[0][:, 0])
        tv.append((np.linalg.inv(T) @ [0, 0, 0, 1])[:3])          # camera centre in camera-0 coordinates
    rv = fill_smooth(np.array(rv), 5)
    c0 = fill_smooth(np.array(tv), 5)
    Rs, pos = [], []
    for r, c in zip(rv, c0):
        Rk0 = Rot.from_rotvec(r).as_matrix()       # camera-0 -> camera-k
        Rs.append(Rk0 @ cam.R)                     # camera-k axes in world
        pos.append(cam.pos + c @ cam.R)            # camera-k centre in world
    params = np.c_[rv, np.zeros(n)]
    return dict(R=np.array(Rs), f=np.full(n, cam.F), pos=np.array(pos), params=params, fit_err_px=err)


def refine_to_keyframes(cam, frames, static_masks, Z, cams, every=10, n_pts=500):
    """Re-estimate every pose against the latest keyframe instead of the previous frame (less drift): corners of
    the keyframe, lifted to 3D with its metric depth and pose, are projected into the frame with the predicted pose,
    refined there by Lucas-Kanade (started at the prediction, so large motions converge), and PnP gives the pose."""
    n = len(frames)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    K = np.array([[cam.F, 0, cam.W / 2], [0, cam.F, cam.H / 2], [0, 0, 1.0]])
    lk = dict(winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

    def w2c(k):                                    # world -> camera k (rows R, centre pos)
        R, p = cams['R'][k], cams['pos'][k]
        return R, -R @ p
    Rw = [cams['R'][k].copy() for k in range(n)]
    Pw = [cams['pos'][k].copy() for k in range(n)]
    err = np.zeros(n)
    ref = 0
    ref_pts = None

    def lift(k):
        p = cv2.goodFeaturesToTrack(g[k], n_pts, 0.01, 7, mask=static_masks[k].astype(np.uint8) * 255)
        if p is None:
            return None, None
        p = p[:, 0]
        z = Z[k][p[:, 1].astype(int), p[:, 0].astype(int)]
        return p, cam.unproject(p[:, 0], p[:, 1], z, Rw[k], cams['f'][k], Pw[k])
    ref_px, ref_X = lift(0)
    for k in range(1, n):
        # predicted pose: the original relative motion from the reference applied to the refined reference pose
        R0r, Rk0 = cams['R'][ref], cams['R'][k]
        dR = Rk0 @ R0r.T
        Rpred = dR @ Rw[ref]
        Ppred = Pw[ref] + (cams['pos'][k] - cams['pos'][ref])
        q0, z = cam.project(ref_X, Rpred, cams['f'][k], Ppred)
        ok = (z > 0) & (q0[:, 0] > 2) & (q0[:, 0] < cam.W - 3) & (q0[:, 1] > 2) & (q0[:, 1] < cam.H - 3)
        if ok.sum() >= 30:
            q, st, _ = cv2.calcOpticalFlowPyrLK(g[ref], g[k], np.ascontiguousarray(ref_px[ok].reshape(-1, 1, 2), np.float32), np.ascontiguousarray(q0[ok].reshape(-1, 1, 2), np.float32), **lk)
            good = st[:, 0] == 1
            Xg, qg = ref_X[ok][good], q[good, 0]
            if len(Xg) >= 30:
                R_, t_ = w2c(ref)
                rv0 = cv2.Rodrigues(Rpred)[0]
                tv0 = (-Rpred @ Ppred)[:, None]
                okp, rv, tv, inl = cv2.solvePnPRansac(Xg, qg, K, None, rvec=rv0, tvec=tv0, useExtrinsicGuess=True,
                                                      reprojectionError=2.0, iterationsCount=300)
                if okp and inl is not None and len(inl) >= 25:
                    R, _ = cv2.Rodrigues(rv)
                    Rw[k], Pw[k] = R, -R.T @ tv[:, 0]
                    pr, _ = cv2.projectPoints(Xg[inl[:, 0]], rv, tv, K, None)
                    err[k] = float(np.median(np.linalg.norm(pr[:, 0] - qg[inl[:, 0]], axis=1)))
                else:
                    Rw[k], Pw[k] = Rpred, Ppred
            else:
                Rw[k], Pw[k] = Rpred, Ppred
        else:
            Rw[k], Pw[k] = Rpred, Ppred
        if k % every == 0:                         # new keyframe
            ref = k
            ref_px, ref_X = lift(k)
            if ref_px is None:
                ref = ref
    rv = fill_smooth(np.array([Rot.from_matrix(R @ cam.R.T).as_rotvec() for R in Rw]), 3)
    pos = fill_smooth(np.array(Pw), 3)
    Rs = np.array([Rot.from_rotvec(r).as_matrix() @ cam.R for r in rv])
    return dict(R=Rs, f=np.full(n, cam.F), pos=pos, params=np.c_[rv, np.zeros(n)], fit_err_px=err)


def alignment_ncc(cam, frames, Z, cams, k, ref=0):
    """NCC of image gradients between frame k and frame `ref` warped into k (through ref's depth and both poses)."""
    from .camera import pose as _pose  # noqa
    H, W = cam.H, cam.W
    vv, uu = np.mgrid[0:H, 0:W]
    X = cam.unproject(uu.ravel(), vv.ravel(), Z[ref].ravel(), *pose(cams, ref))
    q, z = cam.project(X, *pose(cams, k))
    ok = (z > 0) & (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1)
    a = cv2.cvtColor(frames[ref], cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()[ok]
    b = cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY).astype(np.float32)[q[ok, 1].astype(int), q[ok, 0].astype(int)]
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), float(ok.mean())
