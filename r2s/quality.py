"""Shared quality checks for tissue models and 4D simulations against the video (outputs/iter/).

All functions take a views.Views V and meshes in world metres. A mesh is (verts (N, 3), faces (M, 3) int); a 4D
mesh is verts per frame (n, N, 3) with one face list.

    render(V, verts, faces, k)                  -> visible mask (H, W), depth (H, W) of a mesh seen from frame k
    silhouette_iou(V, verts, faces, name, ks)   -> per-frame IoU with the object's mask; pixels covered by
                                                   instruments are ignored (unknown), and so are pixels where
                                                   another listed object is in front (`hidden_by`)
    boundary_f(V, verts, faces, name, ks, tol)  -> per-frame boundary F-score (outline within tol px)
    depth_residual(V, verts, faces, name, ks)   -> per-frame median |z_mesh - z_video| (m) on pixels both cover
    chamfer(P_obs, verts, faces)                -> distances (m) from observed points to the mesh surface
    mesh_health(verts, faces, tets=None)        -> watertight, components, min edge, degenerate faces / tets,
                                                   min dihedral angle of tets
    stretch(rest, verts_t, edges)               -> per-frame max and p95 edge stretch ratio
    contact_sheet(V, layers, ks, path)          -> video frames with each layer's visible part tinted and outlined;
                                                   layers = [(label, verts (or per-frame verts), faces, rgb), ...]
    summarize(per_frame)                        -> dict(mean, median, p10, min, n)
    signed_distance(verts, faces)               -> f(points) = signed distance (m) to a closed surface, > 0 inside
"""
import numpy as np
import cv2


def _verts_at(verts, k):
    v = np.asarray(verts)
    return v[k] if v.ndim == 3 else v


def render(V, verts, faces, k, with_ids=False):
    """Z-buffered rasterisation of a triangle mesh into frame k (painter's order per face, flat depth per face)."""
    X = _verts_at(verts, k)
    q, z = V.project(X, k)
    mask = np.zeros((V.H, V.W), bool)
    zbuf = np.full((V.H, V.W), np.inf, np.float32)
    ids = np.full((V.H, V.W), -1, np.int32)
    F = np.asarray(faces)
    zt = z[F].mean(1)
    ok = (z[F] > 1e-4).all(1)
    for fi in np.nonzero(ok)[0][np.argsort(-zt[ok])]:
        poly = q[F[fi]].astype(np.int32)
        if poly[:, 0].max() < 0 or poly[:, 0].min() >= V.W or poly[:, 1].max() < 0 or poly[:, 1].min() >= V.H:
            continue
        tmp = np.zeros((V.H, V.W), np.uint8)
        cv2.fillConvexPoly(tmp, poly, 1)
        sel = tmp.astype(bool) & (zt[fi] < zbuf)
        zbuf[sel] = zt[fi]
        ids[sel] = fi
        mask |= sel
    return (mask, zbuf, ids) if with_ids else (mask, zbuf)


def _unknown(V, k, hidden_by=()):
    u = V.occluders(k).copy()
    for n in hidden_by:
        u |= V.mask(n)[k]
    return cv2.dilate(u.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def silhouette_iou(V, verts, faces, name, ks, hidden_by=(), visible_only=True):
    """visible_only: compare against what the video shows (the mesh is also cut where the video's depth says something
    else is in front of it)."""
    out = []
    for k in ks:
        m, z = render(V, verts, faces, k)
        if visible_only:
            m &= z < V.depth(k) + 0.006
        real = V.mask(name)[k]
        valid = ~_unknown(V, k, hidden_by)
        a, b = m & valid, real & valid
        out.append(float((a & b).sum() / max((a | b).sum(), 1)))
    return np.array(out)


def boundary_f(V, verts, faces, name, ks, tol=4, hidden_by=()):
    k3 = np.ones((3, 3), np.uint8)
    out = []
    for k in ks:
        m, z = render(V, verts, faces, k)
        m &= z < V.depth(k) + 0.006
        real = V.mask(name)[k]
        valid = ~_unknown(V, k, hidden_by)
        bm = (m.astype(np.uint8) - cv2.erode(m.astype(np.uint8), k3)).astype(bool) & valid
        br = (real.astype(np.uint8) - cv2.erode(real.astype(np.uint8), k3)).astype(bool) & valid
        if bm.sum() == 0 or br.sum() == 0:
            out.append(0.0)
            continue
        dr = cv2.distanceTransform((~br).astype(np.uint8), cv2.DIST_L2, 3)
        dm = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 3)
        prec = float((dr[bm] <= tol).mean())
        rec = float((dm[br] <= tol).mean())
        out.append(2 * prec * rec / max(prec + rec, 1e-9))
    return np.array(out)


def depth_residual(V, verts, faces, name, ks, hidden_by=()):
    out = []
    for k in ks:
        m, z = render(V, verts, faces, k)
        both = m & V.mask(name)[k] & ~_unknown(V, k, hidden_by)
        out.append(float(np.median(np.abs(z[both] - V.depth(k)[both]))) if both.sum() > 30 else np.nan)
    return np.array(out)


def chamfer(P_obs, verts, faces, n_samples=20000, seed=0):
    """Distances (m) from observed points to a dense sample of the mesh surface."""
    from scipy.spatial import cKDTree
    X, F = np.asarray(verts), np.asarray(faces)
    a = np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), n_samples, p=a / a.sum())
    r = rng.random((n_samples, 2))
    r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
    S = X[F[fi, 0]] * (1 - r.sum(1))[:, None] + X[F[fi, 1]] * r[:, :1] + X[F[fi, 2]] * r[:, 1:]
    return cKDTree(S).query(P_obs)[0]


def mesh_health(verts, faces, tets=None):
    import trimesh
    X, F = np.asarray(verts, float), np.asarray(faces)
    m = trimesh.Trimesh(X, F, process=False)
    e = m.edges_unique_length
    area = m.area_faces
    out = dict(n_verts=len(X), n_faces=len(F), watertight=bool(m.is_watertight),
               components=int(len(m.split(only_watertight=False))), min_edge_mm=round(float(e.min()) * 1000, 3),
               degenerate_faces=int((area < 1e-12).sum()), area_cm2=round(float(area.sum()) * 1e4, 2))
    if m.is_watertight:
        out['volume_ml'] = round(float(abs(m.volume)) * 1e6, 2)
    if tets is not None:
        T = np.asarray(tets)
        v = np.einsum('ij,ij->i', np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), X[T[:, 3]] - X[T[:, 0]]) / 6
        out.update(n_tets=len(T), degenerate_tets=int((np.abs(v) < 1e-6 * np.median(np.abs(v))).sum()),
                   inverted_tets=int((v < 0).sum()))
    return out


def stretch(rest, verts_t, edges):
    E = np.asarray(edges)
    L0 = np.linalg.norm(rest[E[:, 0]] - rest[E[:, 1]], axis=1)
    L = np.linalg.norm(verts_t[:, E[:, 0]] - verts_t[:, E[:, 1]], axis=2) / np.maximum(L0, 1e-9)
    return L.max(1), np.percentile(L, 95, axis=1)


def summarize(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(mean=round(float(x.mean()), 4), median=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
                min=round(float(x.min()), 4), max=round(float(x.max()), 4), n=int(len(x)))


def contact_sheet(V, layers, ks, path, cols=4, scale=0.5, alpha=0.45):
    """Each layer's visible part (z-buffered against the other layers and the video's depth) tinted over the frame,
    with its outline; the real object masks are not drawn (compare against the video underneath)."""
    tiles = []
    for k in ks:
        img = V.frames[k].astype(np.float32).copy()
        rendered = []
        for label, verts, faces, rgb in layers:
            m, z = render(V, verts, faces, k)
            rendered.append((m, z, np.array(rgb, np.float32), label))
        zmin = np.min([np.where(m, z, np.inf) for m, z, _, _ in rendered], 0) if rendered else None
        for m, z, rgb, label in rendered:
            vis = m & (z <= zmin + 1e-6)
            img[vis] = (1 - alpha) * img[vis] + alpha * rgb
            cnt = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            img = cv2.drawContours(np.ascontiguousarray(img), cnt, -1, tuple(float(c) for c in rgb), 1)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.putText(img, f'frame {k}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    y = 18
    for label, _, _, rgb in layers:
        cv2.putText(sheet, label, (sheet.shape[1] - 260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(c) for c in rgb), 1, cv2.LINE_AA)
        y += 18
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


_FLOW_CACHE = {}


def video_flow(V, k, gap):
    """Dense optical flow (DIS, medium) from frame k to frame k + gap of the video, cached."""
    key = (id(V), k, gap)
    if key not in _FLOW_CACHE:
        g0 = cv2.cvtColor(V.frames[k], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(V.frames[k + gap], cv2.COLOR_RGB2GRAY)
        _FLOW_CACHE[key] = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g0, g1, None)
    return _FLOW_CACHE[key]


def flow_epe(V, verts, faces, name, ks, gap=5, hidden_by=()):
    """Local motion check independent of any 3D reconstruction: on pixels where the mesh is visible and the video
    shows the object (frame k), compare the mesh's projected motion k -> k+gap (camera + tissue motion) with the
    video's optical flow. Returns per-frame median end-point error (px) for the mesh and for the same mesh held
    still in its frame-k shape (camera motion only)."""
    F = np.asarray(faces)
    err, err_static = [], []
    for k in ks:
        if k + gap >= V.n:
            continue
        Xk, Xg = _verts_at(verts, k), _verts_at(verts, k + gap)
        m, z, ids = render(V, Xk, F, k, with_ids=True)
        sel = m & (z < V.depth(k) + 0.006) & V.mask(name)[k] & ~_unknown(V, k, hidden_by)
        if sel.sum() < 50:
            continue
        q0, _ = V.project(Xk, k)
        q1, _ = V.project(Xg, k + gap)
        q1s, _ = V.project(Xk, k + gap)
        fid = ids[sel]
        mv = (q1 - q0)[F[fid]].mean(1)                 # face-average vertex motion at each pixel
        ms = (q1s - q0)[F[fid]].mean(1)
        fl = video_flow(V, k, gap)[sel]
        err.append(float(np.median(np.linalg.norm(mv - fl, axis=1))))
        err_static.append(float(np.median(np.linalg.norm(ms - fl, axis=1))))
    return np.array(err), np.array(err_static)


def signed_distance(verts, faces, n_samples=40000, seed=0):
    """Signed distance to a closed triangle surface from a dense surface sample (no spatial index library needed):
    distance to the nearest sample, sign from that sample's outward face normal. Positive = inside."""
    from scipy.spatial import cKDTree
    X, F = np.asarray(verts, float), np.asarray(faces)
    a = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    outward = np.sign(np.einsum('ij,ij->', X[F[:, 0]], a)) or 1.0
    nrm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-18) * outward
    A = np.linalg.norm(a, axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), n_samples, p=A / A.sum())
    r = rng.random((n_samples, 2))
    r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
    P = X[F[fi, 0]] * (1 - r.sum(1))[:, None] + X[F[fi, 1]] * r[:, :1] + X[F[fi, 2]] * r[:, 1:]
    tree = cKDTree(P)

    def f(Q):
        Q = np.atleast_2d(Q)
        d, i = tree.query(Q)
        return -np.sign(np.einsum('ij,ij->i', Q - P[i], nrm[fi[i]])) * d
    return f
