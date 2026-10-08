"""Metrics of one condition against the video (every `step` frames).

  outline IoU    simulated organ surface (projected with that frame's scope camera) vs the SAM organ mask, both
                 restricted to pixels no instrument or strand covers in the real frame
  depth error    median |z_sim - z_ref| (mm) where both see the organ; z_ref = the calibrated monocular depth that
                 the reconstruction itself was built from, and, when the dataset has one, an independent reference
                 (EndoNeRF: stereo depth)
  track error    real organ feature tracks (KLT, frame 0 onwards) vs the simulated surface points under them at
                 frame 0, followed through the simulation and projected (px); independent of any depth estimate
Baselines from the same scene: 'frozen' (organ held in its frame-0 shape) is a separate simulation; 'static'
(frame-0 state, only the scope moves) is computed here.
"""
import numpy as np
import cv2
from .camera import lk_tracks, pose
from .organ import boundary_faces


def raster(cam, X, faces, R, f, pos=None, with_ids=False):
    q, z = cam.project(X, R, f, pos)
    mask = np.zeros((cam.H, cam.W), bool)
    zbuf = np.full((cam.H, cam.W), np.inf, np.float32)
    ids = np.full((cam.H, cam.W), -1, np.int32)
    zt = z[faces].mean(1)
    for fi in np.argsort(-zt):                        # far to near
        t = faces[fi]
        if (z[t] <= 0).any():
            continue
        poly = q[t].astype(np.int32)
        if (poly[:, 0].max() < 0) or (poly[:, 0].min() >= cam.W) or (poly[:, 1].max() < 0) or (poly[:, 1].min() >= cam.H):
            continue
        tmp = np.zeros((cam.H, cam.W), np.uint8)
        cv2.fillConvexPoly(tmp, poly, 1)
        sel = tmp.astype(bool)
        zbuf[sel] = zt[fi]
        ids[sel] = fi
        mask |= sel
    return (mask, zbuf, ids) if with_ids else (mask, zbuf)


def backdrop_depth(cam, cv, zb, R, f, behind=0.004):
    """Depth (in camera k) of the static background surface at every pixel of frame k."""
    vv, uu = np.mgrid[0:cam.H, 0:cam.W]
    rays = cam.unproject(uu.ravel(), vv.ravel(), 1.0, R, f)
    q, _ = cv.project(rays)                       # same ray seen in the frame-0 canvas
    z0 = cv.lookup(zb, q[:, 0], q[:, 1]) + behind
    X = cv.unproject(q[:, 0], q[:, 1], z0)
    return ((X - cam.pos) @ R[2]).reshape(cam.H, cam.W)


def bind_tracks(cam, X0, faces, R0, f0, pos0, starts):
    """Barycentric coordinates of each track start on the visible face under it (frame 0)."""
    _, _, ids = raster(cam, X0, faces, R0, f0, pos0, with_ids=True)
    q, _ = cam.project(X0, R0, f0, pos0)
    out = []
    for i, (u, v) in enumerate(starts):
        fi = ids[int(np.clip(v, 0, cam.H - 1)), int(np.clip(u, 0, cam.W - 1))]
        if fi < 0:
            continue
        a, b, c = q[faces[fi]]
        T = np.array([[a[0] - c[0], b[0] - c[0]], [a[1] - c[1], b[1] - c[1]]])
        try:
            l1, l2 = np.linalg.solve(T, np.array([u - c[0], v - c[1]]))
        except np.linalg.LinAlgError:
            continue
        out.append((i, fi, np.array([l1, l2, 1 - l1 - l2])))
    return out


def evaluate(clip, organ, flex, frames, masks, Zref, cams, dt, refs=None, step=5, tracks=None, cv=None, zb=None):
    """flex: (steps, n_flex, 3) simulated vertex positions (organ first). refs: {name: (Z, valid)} extra depth refs.
    cv, zb: canvas and background depth; organ parts behind the background surface do not count as visible."""
    cam = clip.camera()
    n_org = len(organ['X'])
    faces = organ['faces']
    oi = clip.obj_index(clip.organ)
    hide = [clip.obj_index(i['mask']) for i in clip.instruments] + [clip.obj_index(s) for s in clip.role('strand')]
    fps = clip['fps']
    sim_t = np.arange(len(flex)) * dt
    ious, dz = [], {'monocular': []}
    for name in (refs or {}):
        dz[name] = []
    for k in range(0, len(frames), step):
        ks = int(np.argmin(np.abs(sim_t - k / fps)))
        if masks[k, oi].sum() < 300:                    # organ out of view in the video: nothing to compare
            continue
        m, zbuf = raster(cam, flex[ks, :n_org], faces, *pose(cams, k))
        if 'pos' in cams:                               # moving scope: the measured surface itself occludes
            m &= zbuf < Zref[k] + 0.004
        elif cv is not None:
            m &= zbuf < backdrop_depth(cam, cv, zb, cams['R'][k], cams['f'][k])
        vis = ~np.any(masks[k, hide], 0)
        real = masks[k, oi] & vis
        m &= vis
        ious.append(float((m & real).sum() / max((m | real).sum(), 1)))
        both = m & real
        dz['monocular'].append(float(np.median(np.abs(zbuf[both] - Zref[k][both]))) * 1000 if both.any() else np.nan)
        for name, (Zr, valid) in (refs or {}).items():
            b2 = both & valid[k]
            dz[name].append(float(np.median(np.abs(zbuf[b2] - Zr[k][b2]))) * 1000 if b2.sum() > 50 else np.nan)
    res = dict(outline_iou=round(float(np.mean(ious)), 3), iou_curve=[round(x, 3) for x in ious])
    for name, v in dz.items():
        res[f'depth_err_mm_{name}'] = round(float(np.nanmean(v)), 2)
        res[f'depth_curve_{name}'] = [round(x, 2) for x in v]
    # tracks on the organ, seeded every few frames; each is bound to the simulated surface at its seed frame and
    # followed over its horizon (the simulated point vs the real texture; static = the point held where it was)
    if tracks is None:
        tracks = organ_tracks(frames, masks, clip)
    P, seed = tracks
    errs, errs_static, errs0 = [], [], []
    by_seed = []                                            # (seed frame, median sim, median static, samples)
    n_bound = 0
    for s0 in np.unique(seed):
        n_before = len(errs)
        idx = np.nonzero(seed == s0)[0]
        ks0 = int(np.argmin(np.abs(sim_t - s0 / fps)))
        bound = bind_tracks(cam, flex[ks0, :n_org], faces, *pose(cams, s0), P[s0, idx])
        n_bound += len(bound)
        for k in range(s0 + step, len(frames), step):
            live = np.isfinite(P[k, idx, 0])
            if not live.any():
                break
            ks = int(np.argmin(np.abs(sim_t - k / fps)))
            for i, fi, w in bound:
                if not live[i]:
                    continue
                p = (flex[ks, :n_org][faces[fi]] * w[:, None]).sum(0)
                p0 = (flex[ks0, :n_org][faces[fi]] * w[:, None]).sum(0)
                e = np.linalg.norm(cam.project(p, *pose(cams, k))[0] - P[k, idx[i]])
                errs.append(e)
                errs_static.append(np.linalg.norm(cam.project(p0, *pose(cams, k))[0] - P[k, idx[i]]))
                if s0 == 0:
                    errs0.append(e)
        if len(errs) - n_before >= 10:
            by_seed.append([int(s0), round(float(np.median(errs[n_before:])), 2),
                            round(float(np.median(errs_static[n_before:])), 2), len(errs) - n_before])
    res.update(track_err_px=round(float(np.median(errs)), 2) if errs else None,
               track_err_px_static=round(float(np.median(errs_static)), 2) if errs_static else None,
               n_tracks=n_bound, n_track_samples=len(errs), n_track_seeds=int(len(np.unique(seed))),
               track_err_px_frame0_seed=round(float(np.median(errs0)), 2) if errs0 else None, n_track_samples_frame0_seed=len(errs0),
               track_by_seed=by_seed, track_seeds_better_than_static=int(sum(m < st for _, m, st, _ in by_seed)))
    return res


def organ_tracks(frames, masks, clip, n=300, every=25, horizon=50):
    """Forward-backward checked LK tracks of organ texture, seeded at frame 0 and every `every` frames after while the
    organ is in view, each followed for up to `horizon` frames. Seeding once at frame 0 is not enough: a fast, blurred
    pan at the start of a clip (shot A) loses nearly all tracks within a few frames. Returns P (frames, tracks, 2; NaN
    when not tracked) and each track's seed frame."""
    oi = clip.obj_index(clip.organ)
    hide = [clip.obj_index(i['mask']) for i in clip.instruments] + [clip.obj_index(s) for s in clip.role('strand')]
    H, W = frames[0].shape[:2]
    blocks, seeds = [], []
    for s0 in range(0, len(frames) - 1, every):
        if masks[s0, oi].sum() < 300:
            continue
        excl = cv2.dilate(np.any(masks[s0, hide], 0).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        organ = cv2.erode(masks[s0, oi].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        end = min(len(frames), s0 + horizon + 1)
        Q = lk_tracks(frames[s0:end], excl, n=n, mask=organ)
        if Q.shape[1] == 0:
            continue
        # a track that lands on an instrument or strand afterwards is occluded there: drop those samples
        for t in range(1, len(Q)):
            ok = np.isfinite(Q[t, :, 0])
            q = Q[t, ok].astype(int)
            inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            ids = np.nonzero(ok)[0]
            bad = np.zeros(len(ids), bool)
            bad[~inside] = True
            bad[inside] = np.any(masks[s0 + t, hide][:, q[inside, 1], q[inside, 0]], 0)
            Q[t, ids[bad]] = np.nan
        P = np.full((len(frames), Q.shape[1], 2), np.nan, np.float32)
        P[s0:end] = Q
        blocks.append(P)
        seeds.append(np.full(Q.shape[1], s0))
    if not blocks:
        return np.zeros((len(frames), 0, 2), np.float32), np.zeros(0, int)
    return np.concatenate(blocks, 1), np.concatenate(seeds)


MOTION_PX = 15.0     # a seed is in a motion phase when the static baseline is off by >= this much (px, median)


def motion_seeds(ref_sim):
    """Seed frames of the motion phases, chosen on the static baseline of a reference condition only (measured)."""
    return [s for s, _, st, _ in ref_sim.get('track_by_seed', []) if st >= MOTION_PX]


def track_motion(sim, seeds):
    """Mean over motion-phase seeds of the per-seed median track error, simulation and static, and the change in %."""
    d = {s: (m, st) for s, m, st, _ in sim.get('track_by_seed', [])}
    ss = [s for s in seeds if s in d]
    if not ss:
        return None
    m = float(np.mean([d[s][0] for s in ss]))
    st = float(np.mean([d[s][1] for s in ss]))
    return dict(sim_px=round(m, 2), static_px=round(st, 2), change_pct=round((m / st - 1) * 100, 1), seeds=ss)

