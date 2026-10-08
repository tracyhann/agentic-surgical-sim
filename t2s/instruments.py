"""Step 6 of v2: the instrument agent. A small library of rigid parametric instruments, their per-frame pose about ONE
fixed port per instrument (yaw / pitch / insertion / roll) and jaw opening, fitted to the instrument masks through the
clip's cameras; a self-check; and the contract for the assembly step (model.npz + the MJCF builder below).

    PYTHONPATH=. .venv/bin/python -m t2s.instruments <clip> [--ver vNN]   -> outputs/t2s/<clip>/instruments/vNN/

Model (all lengths in metres)
  A straight cylindrical shaft (radius R, from the spec's shaft diameter) through a port P (remote centre of motion,
  outside the picture) and a tip: two jaws hinged at a pivot `jaw_len` behind the tip point (grasper, curved Maryland
  dissector, single-action needle holder) or a rounded end (suction cannula).
  Tip frame T: origin at the tip point (distal end of the closed tool, on the shaft axis), z distal along the shaft,
  x the jaw opening direction, y the jaw hinge axis.
      R_T = R0 . Ry(yaw) . Rx(pitch) . Rz(roll),   tip = P + insertion * R_T[:, 2]
  R0 is the port frame (z0 = mean shaft direction port -> tip), fixed per instrument. jaw = total opening angle
  between the two jaws (double action: each jaw jaw/2; single action: one jaw moves by jaw).
  MuJoCo (mjcf()): port body at P with orientation R0 -> yaw hinge (y) -> pitch hinge (x) -> insertion slide (z) ->
  roll hinge (z) -> jaw hinges at the pivot; joint values = the model.npz 'joints' columns.

Fit (per clip, per instrument; scipy least squares, sparse Jacobian)
  unknowns: P (3) + per frame yaw, pitch, insertion, roll, jaw; globals jls (jaw length / nominal, a shape
  parameter) and kappa (the whole tool's size / nominal, see "scale" below).
  data terms per frame (masks of t2s.views2, cameras of the clip's multi-view solution):
   - mask boundary -> model: signed distance (px) of points sampled on the mask outline to the projected model
     (union of tapered capsules). Outline points on the scope border or against the other instrument are not used;
     points at an OCCLUDED far end (tip inside / behind tissue, cut by the border, or against the other instrument)
     only say "the model reaches at least this far" (one-sided).
   - model -> mask: distance (px) to the mask of points on the projected model outline that should be visible
     (inside the scope image, not against the other instrument, not beyond an occluded far end).
   - metric depth on shaft pixels (low weight, joint stage only). NOTE circularity: the clip's metric scale was
     fitted to these same shafts' apparent widths (5 mm rulers), so the depth map holds no independent scale for
     them; the silhouette width already fixes the depth. The term only adds the depth map's shape along the shaft.
   - free space: the visible shaft lies in front of the tissue just beside it (depth map 16 px outside the mask).
   - hidden length: where the instrument agent recorded the tip inside an organ (chole_a suction cannula), the true
     tip is beyond the visible end by a prior length, capped by the organ's far wall (latest organ model).
   - absent frames (outside the instrument agent's visible ranges; masks there are tracker drift and ignored): the
     distal part of the model must project outside the scope image.
  temporal: 3D acceleration of the tip (tighter along the line of sight, where depth comes from the width alone) and
  of the shaft direction; roll and jaw velocity and acceleration (all fps-scaled).
  priors: the port is outside the picture in every frame, >= 3 cm from the scope, 5-32 cm from the tip, weakly
  ~15 cm from the mean tip (only matters where the data leave the port's distance open).
  Stages: A pose + port at the nominal size, silhouettes only, from several port starts -> B roll / jaw from a grid
  of states per frame and a Viterbi pass (roll held constant where it is seen in too few frames: closed straight
  jaws are axisymmetric) -> C everything jointly (+ depth term, + jaw length) -> scale (below) -> line search of the
  port along its least-determined direction (eigenvector of the axis-plane constraints), refit, profile.
  Scale: silhouettes cannot separate the tool's size from its distance (with the port moving along). The size is
  the nominal diameter unless, at that size, more than 10 % of the free-space samples put the shaft behind the
  tissue beside it; then the tool and its port are scaled about the scope to the largest factor that brings this to
  10 % (kappa < 1: the depth map is nearer than the 5 mm ruler implies there) and refitted.
  Gaps: frames without a usable mask are bridged by the temporal terms when the gap is short (<= 0.5 s, `filled`);
  in long gaps (`long_gap`) the instrument is withdrawn along its shaft until it is out of view.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 't2s'

# ================================================================ library
SHAFT_LEN = 0.40                       # shaft length behind the jaw pivot (reaches past the port)
LIBRARY = {
    # jaw radii are fractions of the shaft radius; curve = bend of the distal half of each jaw (rad, perpendicular to
    # the opening plane: Maryland); jaw_max = largest total opening (rad)
    'grasper': dict(jaw_len=0.016, jaw_rb=0.95, jaw_rt=0.75, curve=0.0, jaw_mode='double', jaw_max=1.2),
    'dissector': dict(jaw_len=0.018, jaw_rb=0.75, jaw_rt=0.25, curve=0.5, jaw_mode='double', jaw_max=1.6),
    'needle_holder': dict(jaw_len=0.012, jaw_rb=0.80, jaw_rt=0.40, curve=0.0, jaw_mode='single', jaw_max=0.8),
    'suction': dict(jaw_len=0.0, jaw_rb=0.0, jaw_rt=0.0, curve=0.0, jaw_mode=None, jaw_max=0.0),
}
ALIASES = {'grasper': 'grasper', 'forceps': 'grasper', 'dissector': 'dissector', 'maryland': 'dissector',
           'needle holder': 'needle_holder', 'needle_holder': 'needle_holder', 'suction cannula': 'suction',
           'suction': 'suction', 'suction/irrigation': 'suction', 'irrigation': 'suction'}


def lib_type(name):
    s = str(name).lower()
    for k, v in ALIASES.items():
        if k in s:
            return v
    return 'grasper'


def make_tool(kind, diameter=0.005, **over):
    """Parametric model of one instrument type (see LIBRARY), shaft diameter in metres."""
    t = dict(LIBRARY[kind])
    t.update(type=kind, radius=diameter / 2, shaft_len=SHAFT_LEN)
    t.update(over)
    t['jawed'] = t['jaw_mode'] is not None
    t['roll_period'] = float(np.pi if (t['jaw_mode'] == 'double' and t['curve'] == 0) else 2 * np.pi)
    return t


def tool_params(t):
    out = {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v) for k, v in t.items()}
    out['tip_len'] = float(t['jaw_len'] if t['jawed'] else t['radius'])     # pivot (or cap centre) -> tip point
    return out


def jaw_geometry(t):
    """Per-jaw capsule chain in the jaw's own frame (hinge at the origin, closed jaw along +z), for the jaw on the +x
    side: points (3, 3) base / mid / tip and radii (3,)."""
    R, jl = t['radius'], t['jaw_len']
    rb, rt = t['jaw_rb'] * R, t['jaw_rt'] * R
    ob, ot = 0.0, 0.0                                       # closed: the two jaws coincide (an axisymmetric tapered tip)
    om = 0.5 * (ob + ot)
    k = t['curve']
    mid = np.array([om, 0.0, jl / 2])
    tip = mid + (jl / 2) * np.array([0.0, np.sin(k), np.cos(k)]) + np.array([ot - om, 0, 0])
    return np.array([[ob, 0.0, 0.0], mid, tip]), np.array([rb, 0.5 * (rb + rt), rt])


def local_prims(t, jaw):
    """Primitives (tapered capsules) of the tool in its tip frame for jaw openings jaw (N,):
    A, B (N, np, 3) end points, ra, rb (np,) radii. Primitive 0 is the shaft (A = proximal end)."""
    jaw = np.atleast_1d(np.asarray(jaw, float))
    N = len(jaw)
    R, jl = t['radius'], t['jaw_len']
    if not t['jawed']:
        A = np.tile([0.0, 0.0, -R - t['shaft_len']], (N, 1, 1))
        B = np.tile([0.0, 0.0, -R], (N, 1, 1))
        return A, B, np.array([R]), np.array([R])
    pts, rad = jaw_geometry(t)
    th_a = jaw / 2 if t['jaw_mode'] == 'double' else jaw
    th_b = jaw / 2 if t['jaw_mode'] == 'double' else np.zeros(N)
    As, Bs, ra, rb = [np.tile([0.0, 0.0, -jl - t['shaft_len']], (N, 1))], [np.tile([0.0, 0.0, -jl], (N, 1))], [R], [R]
    for sgn, th in ((1.0, th_a), (-1.0, th_b)):
        p = pts * np.array([sgn, 1.0, 1.0])
        c, s = np.cos(sgn * th), np.sin(sgn * th)          # rotation about +y by sgn * th
        x = p[None, :, 0] * c[:, None] + p[None, :, 2] * s[:, None]
        z = -p[None, :, 0] * s[:, None] + p[None, :, 2] * c[:, None]
        q = np.stack([x, np.broadcast_to(p[None, :, 1], x.shape), z - jl], -1)     # (N, 3, 3)
        for i in range(2):
            As.append(q[:, i]), Bs.append(q[:, i + 1]), ra.append(rad[i]), rb.append(rad[i + 1])
    return np.stack(As, 1), np.stack(Bs, 1), np.array(ra), np.array(rb)


# ================================================================ kinematics
def _rx(a):
    a = np.asarray(a, float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _ry(a):
    a = np.asarray(a, float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rz(a):
    a = np.asarray(a, float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def tip_frame(R0, yaw, pitch, roll):
    """(N, 3, 3) rotation of the tip frame in world."""
    return R0 @ _ry(yaw) @ _rx(pitch) @ _rz(roll)


def port_frame(z0, x_hint):
    z0 = z0 / np.linalg.norm(z0)
    x0 = x_hint - (x_hint @ z0) * z0
    x0 /= np.linalg.norm(x0)
    return np.stack([x0, np.cross(z0, x0), z0], 1)          # columns x0, y0, z0


def angles_of(R0, d):
    """yaw, pitch of unit shaft directions d (N, 3) in port frame R0 (inverse of tip_frame's z column)."""
    dl = np.asarray(d, float) @ R0                       # R0^T d
    return np.arctan2(dl[..., 0], dl[..., 2]), -np.arcsin(np.clip(dl[..., 1], -1, 1))


def jaw_tip_world(t, Q, R0):
    """(n, 3) distal end of the tool as built: the midpoint of the two jaw tips (curved jaws bend off the axis), or
    the end of the rounded cap."""
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    O = Q['P'][None] + Q['L'][:, None] * RT[:, :, 2]
    if not t['jawed']:
        return O
    A, B, _, _ = local_prims(t, Q['jaw'])
    loc = 0.5 * (B[:, 2] + B[:, 4])
    return O + np.einsum('nij,nj->ni', RT, loc)


def unwrap_roll(r, period):
    return np.unwrap(np.asarray(r, float) * (2 * np.pi / period)) * (period / (2 * np.pi))


# ================================================================ projection of the model
Z_NEAR = 0.008


def project_prims(t, P, R0, q, cam):
    """Projected primitives of N items. q: dict yaw, pitch, L, roll, jaw (N,); cam: dict R (N,3,3), f (N,), pos (N,3),
    W, H. Returns dict c0, c1 (N, np, 2), r0, r1 (N, np) px, z0, z1 (N, np), O (N, 3) tip, RT (N, 3, 3), ok (N, np)."""
    RT = tip_frame(R0, q['yaw'], q['pitch'], q['roll'])
    O = P[None] + q['L'][:, None] * RT[:, :, 2]
    A, B, ra, rb = local_prims(t, q['jaw'])
    XA = O[:, None] + np.einsum('nij,npj->npi', RT, A)
    XB = O[:, None] + np.einsum('nij,npj->npi', RT, B)
    Rc, pos = cam['R'], cam['pos']
    CA = np.einsum('nij,npj->npi', Rc, XA - pos[:, None])
    CB = np.einsum('nij,npj->npi', Rc, XB - pos[:, None])
    za, zb = CA[..., 2], CB[..., 2]
    ok = (za > Z_NEAR) | (zb > Z_NEAR)
    # clip at the near plane (the shaft runs back past the scope)
    ra_ = np.broadcast_to(ra, za.shape).copy()
    rb_ = np.broadcast_to(rb, zb.shape).copy()
    fa = (za < Z_NEAR) & (zb > Z_NEAR)
    if fa.any():
        s = ((Z_NEAR - zb) / (za - zb))[fa]
        CA[fa] = CB[fa] + s[:, None] * (CA[fa] - CB[fa])
        ra_[fa] = rb_[fa] + s * (ra_[fa] - rb_[fa])
    fb = (zb < Z_NEAR) & (za > Z_NEAR)
    if fb.any():
        s = ((Z_NEAR - za) / (zb - za))[fb]
        CB[fb] = CA[fb] + s[:, None] * (CB[fb] - CA[fb])
        rb_[fb] = ra_[fb] + s * (rb_[fb] - ra_[fb])
    za = np.maximum(CA[..., 2], Z_NEAR)
    zb = np.maximum(CB[..., 2], Z_NEAR)
    f = cam['f'][:, None]
    c0 = np.stack([f * CA[..., 0] / za, f * CA[..., 1] / za], -1) + [cam['W'] / 2, cam['H'] / 2]
    c1 = np.stack([f * CB[..., 0] / zb, f * CB[..., 1] / zb], -1) + [cam['W'] / 2, cam['H'] / 2]
    r0 = np.where(ok, f * ra_ / za, 0.0)
    r1 = np.where(ok, f * rb_ / zb, 0.0)
    c0 = np.where(ok[..., None], c0, -1e4)
    c1 = np.where(ok[..., None], c1, -1e4)
    return dict(c0=c0, c1=c1, r0=r0, r1=r1, z0=za, z1=zb, O=O, RT=RT, XA=XA, XB=XB, ok=ok)


def seg_sd(p, c0, c1, r0, r1):
    """Signed distance (px) of points p (..., M, 2) to tapered capsules (..., np, 2)/(..., np): (..., M, np)."""
    d = c1 - c0
    L2 = np.maximum((d ** 2).sum(-1), 1e-9)
    v = p[..., :, None, :] - c0[..., None, :, :]
    s = np.clip((v * d[..., None, :, :]).sum(-1) / L2[..., None, :], 0, 1)
    w = v - s[..., None] * d[..., None, :, :]
    return np.sqrt((w ** 2).sum(-1) + 1e-12) - (r0[..., None, :] + s * (r1 - r0)[..., None, :])


def outline_points(t, pr, cam, n_jaw=4):
    """Points on the projected model outline (N, K, 2) and the primitive they belong to (K,). Shaft: two sides at
    fixed 3D stations behind the pivot (dense near the tip); rounded end / jaw tips: cap points."""
    N = len(pr['O'])
    out, own = [], []
    # shaft stations (3D) -> exact projected sides
    stations = np.array([0.0, 0.003, 0.007, 0.012, 0.018, 0.025, 0.035, 0.05, 0.065, 0.085, 0.11, 0.14, 0.18, 0.23])
    base = pr['XB'][:, 0]                                   # distal end of the shaft primitive (pivot / cap centre)
    d = pr['RT'][:, :, 2]
    X = base[:, None] - stations[None, :, None] * d[:, None]
    Xc = np.einsum('nij,nsj->nsi', cam['R'], X - cam['pos'][:, None])
    z = np.maximum(Xc[..., 2], Z_NEAR)
    f = cam['f'][:, None]
    c = np.stack([f * Xc[..., 0] / z, f * Xc[..., 1] / z], -1) + [cam['W'] / 2, cam['H'] / 2]
    rr = f * t['radius'] / z
    ax = pr['c1'][:, 0] - pr['c0'][:, 0]
    ax /= np.linalg.norm(ax, axis=-1, keepdims=True) + 1e-9
    nrm = np.stack([-ax[:, 1], ax[:, 0]], -1)
    bad = (Xc[..., 2] < Z_NEAR)[..., None]
    for sg in (1, -1):
        p = c + sg * rr[..., None] * nrm[:, None]
        out.append(np.where(bad, -1e4, p))
        own += [0] * len(stations)
    if not t['jawed']:                                       # rounded end: cap points
        cc, r1 = pr['c1'][:, 0], pr['r1'][:, 0]
        for a in np.radians([-60, -30, 0, 30, 60]):
            v = np.cos(a) * ax + np.sin(a) * nrm
            out.append((cc + r1[:, None] * v)[:, None]), own.append(0)
    else:
        for j in range(1, pr['c0'].shape[1]):
            c0, c1, r0, r1 = pr['c0'][:, j], pr['c1'][:, j], pr['r0'][:, j], pr['r1'][:, j]
            e = c1 - c0
            e /= np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9
            nn = np.stack([-e[:, 1], e[:, 0]], -1)
            for s in np.linspace(0.25, 1.0, n_jaw):
                cc = c0 + s * (c1 - c0)
                rr_ = r0 + s * (r1 - r0)
                for sg in (1, -1):
                    out.append((cc + sg * rr_[:, None] * nn)[:, None]), own.append(j)
            if j % 2 == 0:                                   # distal segment: tip cap
                out.append((c1 + r1[:, None] * e)[:, None]), own.append(j)
    return np.concatenate(out, 1), np.array(own)


def _clip_seg(a, b, ra, rb, lo, hi):
    """Clip the 2D segment a-b (radii ra, rb, linear along it) to the box lo..hi; None if outside."""
    t0, t1 = 0.0, 1.0
    d = b - a
    for i in range(2):
        for p, q in ((-d[i], a[i] - lo[i]), (d[i], hi[i] - a[i])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    if t0 > t1:
        return None
    return a + t0 * d, a + t1 * d, ra + t0 * (rb - ra), ra + t1 * (rb - ra)


def render(pr, i, H, W):
    """Raster silhouette (H, W) bool of item i of projected primitives (segments clipped to a box around the image)."""
    m = np.zeros((H, W), np.uint8)
    sh = 2
    k = 1 << sh
    lo, hi = np.array([-800.0, -800.0]), np.array([W + 800.0, H + 800.0])
    for j in range(pr['c0'].shape[1]):
        if not pr['ok'][i, j]:
            continue
        a, b, ra, rb = pr['c0'][i, j], pr['c1'][i, j], pr['r0'][i, j], pr['r1'][i, j]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        cl = _clip_seg(a, b, ra, rb, lo, hi)
        if cl is None:
            continue
        a, b, ra, rb = cl
        d = b - a
        L = np.linalg.norm(d)
        if L > 1e-6:
            d /= L
            n = np.array([-d[1], d[0]])
            poly = np.array([a + n * ra, b + n * rb, b - n * rb, a - n * ra])
            cv2.fillConvexPoly(m, np.round(poly * k).astype(np.int32), 1, cv2.LINE_8, sh)
        for c, r in ((a, ra), (b, rb)):
            if r > 0.3:
                cv2.circle(m, (int(round(c[0] * k)), int(round(c[1] * k))), int(round(r * k)), 1, -1, cv2.LINE_8, sh)
    return m.astype(bool)


def axis_depth(O, d, cam, uv):
    """Camera depth of the points of the 3D lines O + s d closest to the rays through pixels uv (N, J, 2), and the
    line parameter s (N, J) (negative = behind the tip point, towards the port)."""
    rc = np.stack([(uv[..., 0] - cam['W'] / 2) / cam['f'][:, None], (uv[..., 1] - cam['H'] / 2) / cam['f'][:, None],
                   np.ones(uv.shape[:2])], -1)
    r = np.einsum('nji,nil->njl', rc, cam['R'])
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    w0 = O - cam['pos']
    b = np.einsum('nji,ni->nj', r, d)
    dd = np.einsum('nji,ni->nj', r, w0)
    ee = np.einsum('ni,ni->n', w0, d)[:, None]
    s = (b * dd - ee) / np.maximum(1 - b ** 2, 1e-9)
    X = O[:, None] + s[..., None] * d[:, None]
    zc = np.einsum('ni,nji->nj', cam['R'][:, 2, :], X - cam['pos'][:, None])
    return zc, s


# ================================================================ 2D measurements from the masks
def _clean(m):
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        keep = 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])
        out = lab == keep[0]
        for i in keep[1:]:
            if st[i, cv2.CC_STAT_AREA] > 0.15 * st[keep[0], cv2.CC_STAT_AREA]:
                out |= lab == i
        m = out.astype(np.uint8)
    # fill holes (specular highlights) and close small notches
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    f = np.zeros_like(m)
    cv2.drawContours(f, cnts, -1, 1, -1)
    return f.astype(bool)


def edge_axis(m, bz, step=4):
    """Dominant orientation (unit 2D, sign arbitrary) of the straight parts of a mask outline (contour tangents,
    border pixels excluded): the shaft direction of a short stub where the centre line is too short to fit."""
    cnts = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts:
        return None
    C = max(cnts, key=len)[:, 0].astype(float)
    if len(C) < 4 * step:
        return None
    T = np.roll(C, -step, 0) - np.roll(C, step, 0)
    inner = ~bz[C[:, 1].astype(int), C[:, 0].astype(int)]
    inner &= np.roll(inner, step) & np.roll(inner, -step)
    if inner.sum() < 10:
        return None
    ang = np.mod(np.arctan2(T[inner, 1], T[inner, 0]), np.pi)
    hist = np.bincount((ang / np.pi * 180).astype(int) % 180, minlength=180).astype(float)
    ker = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    hs = np.convolve(np.r_[hist[-8:], hist, hist[:8]], ker, mode='same')[8:-8]
    a0 = (np.argmax(hs) + 0.5) / 180 * np.pi
    dd = np.angle(np.exp(2j * (ang - a0))) / 2
    sel = np.abs(dd) < np.radians(10)
    a1 = a0 + (dd[sel].mean() if sel.any() else 0)
    return np.array([np.cos(a1), np.sin(a1)])


def measure_frame(m, other, bz, prev_dir=None, min_area=150, M=64, Jz=10):
    """Axis, far end, widths, outline samples and shaft pixels of one instrument mask (None: too little mask).
    bz: border zone (scope border + image edge), other: the other instruments' masks."""
    if m.sum() < min_area:
        return None
    m = _clean(m)
    H, W = m.shape
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    _, sv, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    elong = float(sv[0] / max(sv[1], 1e-9))
    onb = bz[ys, xs]
    if onb.sum() >= 3:
        if (c - P[onb].mean(0)) @ d < 0:
            d = -d
    elif prev_dir is not None:
        if d @ prev_dir < 0:
            d = -d
    else:
        s = (P - c) @ d
        a, b = P[s <= np.percentile(s, 1)].mean(0), P[s >= np.percentile(s, 99)].mean(0)
        bd = lambda q: min(q[0], W - 1 - q[0], q[1], H - 1 - q[1])
        if bd(b) < bd(a):
            d = -d
    oth = cv2.dilate(other.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    for it in range(2):                                       # centre-line refinement of the axis (shaft part)
        nrm = np.array([-d[1], d[0]])
        s, tt = (P - c) @ d, (P - c) @ nrm
        bins = np.arange(s.min(), s.max() + 3, 3.0)
        idx = np.digitize(s, bins)
        rows = []
        for i in np.unique(idx):
            sel = idx == i
            if sel.sum() < 4:
                continue
            rows.append((s[sel].mean(), 0.5 * (tt[sel].min() + tt[sel].max()), np.ptp(tt[sel]) + 1,
                         bool(onb[sel].any()), bool(oth[ys[sel], xs[sel]].any())))
        S = np.array(rows, float).reshape(-1, 5)
        if len(S) < 4:
            break
        wmed = np.median(S[S[:, 3] == 0, 2]) if (S[:, 3] == 0).sum() >= 3 else np.median(S[:, 2])
        sh = (S[:, 0] < S[:, 0].max() - 1.5 * wmed) & (S[:, 3] == 0) & (S[:, 4] == 0)
        if sh.sum() < 4 or elong < 2.5:
            break
        A = np.stack([S[sh, 0], np.ones(sh.sum())], 1)
        wts = np.ones(sh.sum())
        for _ in range(3):
            coef = np.linalg.lstsq(A * wts[:, None], S[sh, 1] * wts, rcond=None)[0]
            wts = 1 / np.maximum(1, np.abs(S[sh, 1] - A @ coef) / 1.5)
        ang = np.clip(np.arctan(coef[0]), -np.radians(10), np.radians(10))
        d = np.cos(ang) * d + np.sin(ang) * nrm
        d /= np.linalg.norm(d)
        c = c + coef[1] * nrm
    nrm = np.array([-d[1], d[0]])
    s, tt = (P - c) @ d, (P - c) @ nrm
    smax = float(np.percentile(s, 99.8))
    smin = float(s.min())
    bins = np.arange(smin, smax + 3, 3.0)
    idx = np.digitize(s, bins)
    rows = []
    for i in np.unique(idx):
        sel = idx == i
        if sel.sum() < 3:
            continue
        rows.append((s[sel].mean(), 0.5 * (tt[sel].min() + tt[sel].max()), np.ptp(tt[sel]) + 1,
                     bool(onb[sel].any()), bool(oth[ys[sel], xs[sel]].any())))
    S = np.array(rows, float).reshape(-1, 5)
    free = (S[:, 3] == 0) & (S[:, 4] == 0)
    wsh_rows = free & (S[:, 0] < smax - 1.5 * (np.median(S[free, 2]) if free.sum() else 10))
    w_shaft = float(np.median(S[wsh_rows, 2])) if wsh_rows.sum() >= 3 else float(np.median(S[:, 2]))
    end_rows = S[:, 0] > smax - 4
    end_ratio = float(S[end_rows, 2].mean() / max(w_shaft, 1)) if end_rows.any() else 1.0
    far = s >= smax - 3
    end_on_border = bool(onb[far].any() | bz[np.clip(np.round(c + d * (smax + 2))[1], 0, H - 1).astype(int),
                                               np.clip(np.round(c + d * (smax + 2))[0], 0, W - 1).astype(int)])
    end_on_other = bool(oth[ys[far], xs[far]].any())
    # width line for the initial depth (inverse depth is affine along the image of a 3D line -> so is the width)
    wline = None
    if wsh_rows.sum() >= 4 and np.ptp(S[wsh_rows, 0]) > 15:
        A = np.stack([S[wsh_rows, 0], np.ones(wsh_rows.sum())], 1)
        wts = np.ones(wsh_rows.sum())
        for _ in range(3):
            co = np.linalg.lstsq(A * wts[:, None], S[wsh_rows, 2] * wts, rcond=None)[0]
            wts = 1 / np.maximum(1, np.abs(S[wsh_rows, 2] - A @ co) / 1.5)
        wline = (float(co[0]), float(co[1]), float(S[wsh_rows, 0].min()), float(S[wsh_rows, 0].max()))
    # outline samples
    cnts = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    C = np.concatenate([q[:, 0] for q in cnts], 0).astype(float)
    bzd = cv2.dilate(bz.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ok = ~bzd[C[:, 1].astype(int), C[:, 0].astype(int)] & ~oth[C[:, 1].astype(int), C[:, 0].astype(int)]
    C = C[ok]
    if len(C) > M:
        C = C[np.linspace(0, len(C) - 1, M).round().astype(int)]
    bs = (C - c) @ d if len(C) else np.zeros(0)
    end_zone = bs > smax - 0.7 * w_shaft
    # shaft pixels for the depth term (eroded mask, shaft part, near the centre line)
    er = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    eys, exs = np.nonzero(er)
    Z = np.zeros((0, 2))
    if len(exs):
        E = np.stack([exs, eys], 1).astype(float)
        es, et = (E - c) @ d, (E - c) @ nrm
        selz = (es < smax - 1.5 * w_shaft) & (np.abs(et) < 0.25 * w_shaft) & ~bz[eys, exs]
        if selz.sum() >= 3:
            Z = E[selz][np.linspace(0, selz.sum() - 1, min(Jz, selz.sum())).round().astype(int)]
    # axis for the metrics: the centre line where the shaft part is long enough, else the straight outline edges
    shaft_len = float(np.ptp(S[wsh_rows, 0])) if wsh_rows.sum() >= 2 else 0.0
    d_metric = d
    if shaft_len < 2.5 * w_shaft:
        de = edge_axis(m, bz)
        if de is not None:
            d_metric = de if de @ d >= 0 else -de
    return dict(c=c, d=d, d_metric=d_metric, shaft_len=shaft_len, smin=smin, smax=smax, tip=c + d * smax, w_shaft=w_shaft, end_ratio=end_ratio,
                end_on_border=end_on_border, end_on_other=end_on_other, wline=wline, elong=elong, S=S,
                outline=C, end_zone=end_zone, zpx=Z, area=int(m.sum()), mask=m,
                touches_border=bool(onb.sum() >= 3))


def blur_scores(frames):
    return np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])


# ================================================================ observations of one instrument
def observations(V, name, others, cfg, hidden_frames=(), agent_visible=None):
    n, H, W = V.n, V.H, V.W
    bz = cv2.dilate((~V.valid).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    bz[:2], bz[-2:], bz[:, :2], bz[:, -2:] = True, True, True, True
    M_ = V.mask(name)
    O_ = np.any([V.mask(o) for o in others], 0) if others else np.zeros_like(M_)
    meas, prev = [], None
    trimmed = np.zeros(n, bool)
    for a, b in cfg.get('trim', []):
        trimmed[a:b + 1] = True
    # masks outside the instrument agent's visible ranges are tracker drift (the agent checked every frame): ignored
    inr = np.zeros(n, bool)
    for a, b in (agent_visible or [[0, n - 1]]):
        inr[a:b + 1] = True
    drift = np.array([bool(M_[k].any()) for k in range(n)]) & ~inr
    for k in range(n):
        mk = None if (trimmed[k] or drift[k]) else measure_frame(M_[k], O_[k], bz, prev, M=cfg['M'], Jz=cfg['Jz'])
        if mk is not None and mk['area'] < cfg['min_area']:
            mk = None
        meas.append(mk)
        if mk is not None:
            prev = mk['d']
    M, Jz = cfg['M'], cfg['Jz']
    ob = dict(b=np.zeros((n, M, 2)), bw=np.zeros((n, M)), bone=np.zeros((n, M), bool), zpx=np.zeros((n, Jz, 2)),
              zobs=np.ones((n, Jz)), zw=np.zeros((n, Jz)), vis=np.zeros(n, bool), hid=np.zeros(n, bool),
              endpx=np.zeros((n, 2)), axis_d=np.zeros((n, 2)), axis_c=np.zeros((n, 2)), smax=np.zeros(n),
              end_type=np.array(['none'] * n, dtype=object), trimmed=trimmed, wshaft=np.zeros(n),
              hid_prior=np.zeros(n, bool), fw=np.zeros(n))
    Jf = cfg['Jf']
    ob.update(fpx=np.zeros((n, Jf, 2)), fz=np.ones((n, Jf)), fw_=np.zeros((n, Jf)))
    allins = np.any([V.mask(o) for o in V.instrument_names], 0)
    dt = np.full((n, H, W), 255, np.uint8)                   # 2 * distance to the mask (px), 255 = not usable
    hidden_frames = set(hidden_frames)
    sharp = cfg['sharp']
    for k, mk in enumerate(meas):
        if mk is None:
            continue
        ob['vis'][k] = True
        fw = float(np.clip(sharp[k] / cfg['sharp_ref'], 0.3, 1.0))
        ob['fw'][k] = fw
        L = len(mk['outline'])
        ob['b'][k, :L] = mk['outline']
        ob['bw'][k, :L] = fw
        if k in hidden_frames:
            et = 'inside_organ'
        elif mk['end_on_border']:
            et = 'cut'
        elif mk['end_on_other']:
            et = 'other'
        elif mk['end_ratio'] > cfg['blunt_ratio']:
            et = 'blunt'
        else:
            et = 'free'
        ob['end_type'][k] = et
        ob['hid'][k] = et != 'free'
        ob['hid_prior'][k] = et == 'inside_organ'
        if ob['hid'][k]:
            ob['bone'][k, :L] = mk['end_zone']
        Lz = len(mk['zpx'])
        if Lz:
            ob['zpx'][k, :Lz] = mk['zpx']
            ob['zobs'][k, :Lz] = V.depth(k)[mk['zpx'][:, 1].astype(int), mk['zpx'][:, 0].astype(int)]
            ob['zw'][k, :Lz] = fw
        # free space: the farther of the two tissue samples just beside the visible shaft (instrument pixels excluded)
        S = mk['S']
        nr2 = np.array([-mk['d'][1], mk['d'][0]])
        sel = (S[:, 3] == 0) & (S[:, 4] == 0) & (S[:, 0] < mk['smax'] - 1.5 * mk['w_shaft']) & (S[:, 2] > 6)
        if sel.sum() >= 2:
            excl = cv2.dilate(allins[k].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            c2 = mk['c'] + S[sel, :1] * mk['d'] + S[sel, 1:2] * nr2
            zs = np.full((len(c2), 2), np.nan)
            for j, sg in enumerate((-1, 1)):
                p = np.round(c2 + sg * nr2 * (S[sel, 2:3] / 2 + cfg['free_off'])).astype(int)
                ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
                pp = np.clip(p, 0, [W - 1, H - 1])
                ok &= V.valid[pp[:, 1], pp[:, 0]] & ~excl[pp[:, 1], pp[:, 0]]
                zs[ok, j] = V.depth(k)[pp[ok, 1], pp[ok, 0]]
            zf = np.where(np.isfinite(zs).any(1), np.nanmax(np.where(np.isfinite(zs), zs, -np.inf), 1), np.nan)
            okf = np.isfinite(zf)
            if okf.sum():
                ii = np.nonzero(okf)[0]
                ii = ii[np.linspace(0, len(ii) - 1, min(Jf, len(ii))).round().astype(int)]
                ob['fpx'][k, :len(ii)] = c2[ii]
                ob['fz'][k, :len(ii)] = zf[ii]
                ob['fw_'][k, :len(ii)] = fw
        ob['endpx'][k] = mk['tip']
        ob['axis_d'][k], ob['axis_c'][k], ob['smax'][k] = mk['d'], mk['c'], mk['smax']
        ob['wshaft'][k] = mk['w_shaft']
        dd = cv2.distanceTransform((~mk['mask']).astype(np.uint8), cv2.DIST_L2, 5)
        elig = V.valid & ~bz & ~cv2.dilate(O_[k].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if ob['hid'][k]:
            yy, xx = np.mgrid[0:H, 0:W]
            elig &= ((xx - mk['c'][0]) * mk['d'][0] + (yy - mk['c'][1]) * mk['d'][1]) < mk['smax'] - 2
        dt[k] = np.where(elig, np.minimum(np.round(dd * 2), 254), 255).astype(np.uint8)
    ob['dt'] = dt
    # absent: outside the instrument agent's visible ranges -> the model must be out of view there
    ob['absent'] = ~ob['vis'] & ~inr & ~trimmed
    ob['drift'] = drift
    ob['dtin'] = cv2.distanceTransform(V.valid.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    return meas, ob


def dt_lookup(dt, idx, p):
    """Bilinear lookup of the 2x distance maps dt (n, H, W) at points p (N, K, 2) of frames idx (N,): distance (px)
    and usable flag (all four neighbours usable)."""
    n, H, W = dt.shape
    x, y = p[..., 0], p[..., 1]
    inside = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    x0 = np.clip(np.floor(x), 0, W - 2).astype(int)
    y0 = np.clip(np.floor(y), 0, H - 2).astype(int)
    ax, ay = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    k = np.broadcast_to(idx[:, None], x.shape)
    v00, v01 = dt[k, y0, x0].astype(float), dt[k, y0, x0 + 1].astype(float)
    v10, v11 = dt[k, y0 + 1, x0].astype(float), dt[k, y0 + 1, x0 + 1].astype(float)
    use = inside & (v00 < 255) & (v01 < 255) & (v10 < 255) & (v11 < 255)
    v = (v00 * (1 - ax) * (1 - ay) + v01 * ax * (1 - ay) + v10 * (1 - ax) * ay + v11 * ax * ay) / 2
    return np.where(use, v, 0.0), use


# ================================================================ residuals
def cams_of(V, idx):
    return dict(R=V.R[idx], f=V.f[idx], pos=V.pos[idx], W=V.W, H=V.H)


def scaled(t, kappa=1.0, jls=1.0):
    """The tool scaled as a whole by kappa (apparent size / nominal: radius and jaw length) with the jaw length
    further scaled by jls (shape)."""
    if kappa == 1.0 and jls == 1.0:
        return t
    return dict(t, radius=t['radius'] * kappa, jaw_len=t['jaw_len'] * kappa * jls)


def frame_residuals(t, P, R0, q, idx, ob, cam, cfg, parts=False, kappa=1.0, jls=1.0, extras=True):
    """Data residuals of N items (frames idx, poses q): dict of (N, ...) arrays, already divided by their sigmas.
    kappa scales the whole tool (apparent size / nominal), jls the jaw length. extras=False skips the scene terms
    (background, organ coupling: zeros), for the roll / jaw grid."""
    t = scaled(t, kappa, jls)
    pr = project_prims(t, P, R0, q, cam)
    res = {}
    # mask outline -> model
    sd = seg_sd(ob['b'][idx], pr['c0'], pr['c1'], pr['r0'], pr['r1']).min(-1)       # (N, M)
    one = ob['bone'][idx]
    r = np.where(one, np.maximum(sd, 0), sd)
    res['outline'] = r * ob['bw'][idx] / cfg['sig_b']
    # model outline -> mask
    pts, own = outline_points(t, pr, cam)
    dist, use = dt_lookup(ob['dt'], idx, pts)
    # only points on the outline of the union (not inside another primitive)
    sdu = seg_sd(pts, pr['c0'], pr['c1'], pr['r0'], pr['r1'])
    for j in np.unique(own):
        sdu[:, own == j, j] = np.inf
    on_union = sdu.min(-1) > -0.75
    vis = ob['vis'][idx][:, None]
    res['model'] = np.where(use & on_union & vis, dist, 0.0) * ob['fw'][idx][:, None] / cfg['sig_m']
    # depth map on shaft pixels (low weight; see the module note on circularity)
    d = pr['RT'][:, :, 2]
    zl, _ = axis_depth(pr['O'], d, cam, ob['zpx'][idx])
    res['depth'] = (zl - t['radius'] - ob['zobs'][idx]) * ob['zw'][idx] * cfg['w_depth'] / cfg['sig_z']
    # free space: the visible shaft lies in front of the tissue just beside it (one-sided, tolerance free_tol)
    zf, _ = axis_depth(pr['O'], d, cam, ob['fpx'][idx])
    res['free'] = np.maximum(zf - t['radius'] - ob['fz'][idx] + cfg['free_tol'], 0) * ob['fw_'][idx] * cfg['w_free'] / cfg['sig_free']
    # absent frames: the distal part of the model projects outside the scope image
    ab = ob['absent'][idx]
    if ab.any():
        piv = pr['XB'][:, 0]
        X = np.concatenate([pr['XB'][:, 1:], pr['O'][:, None], piv[:, None] - np.array([0.01, 0.025, 0.045])[None, :, None] * d[:, None]], 1)
        Xc = np.einsum('nij,nkj->nki', cam['R'], X - cam['pos'][:, None])
        zc = Xc[..., 2]
        u = cam['f'][:, None] * Xc[..., 0] / np.maximum(zc, 1e-4) + cam['W'] / 2
        v = cam['f'][:, None] * Xc[..., 1] / np.maximum(zc, 1e-4) + cam['H'] / 2
        Hh, Ww = ob['dtin'].shape
        ins = (zc > 0.005) & (u >= 0) & (u < Ww - 1) & (v >= 0) & (v < Hh - 1)
        din = ob['dtin'][np.clip(v, 0, Hh - 1).astype(int), np.clip(u, 0, Ww - 1).astype(int)]
        res['outview'] = np.where(ab[:, None] & ins & (din > 0), din + 3.0, 0.0) / 5.0
    else:
        res['outview'] = np.zeros((len(idx), 3 + (pr['XB'].shape[1] - 1) + 1))
    # hidden length beyond the visible end (tip inside an organ; replaced by the puncture terms when coupled)
    _, s_end = axis_depth(pr['O'], d, cam, ob['endpx'][idx][:, None])
    h = -s_end[:, 0]                                        # tip beyond the visible end along the shaft (m)
    hp = ob['hid_prior'][idx] & (ob.get('cpl') is None)
    h0 = ob['h0'][idx] if 'h0' in ob else np.full(len(idx), cfg['hidden_len'])
    hmax = ob['hmax'][idx] if 'hmax' in ob else np.full(len(idx), np.inf)
    res['hidden'] = np.where(hp, (h - h0) / cfg['sig_hidden'], 0.0)
    res['hidden_max'] = np.where(hp & np.isfinite(hmax), np.maximum(h - hmax, 0) / 0.002, 0.0)
    # port: outside the picture, away from the scope, within reach of the tip
    qP, zP = _proj1(P, cam)
    inside = np.minimum.reduce([qP[:, 0], cam['W'] - qP[:, 0], qP[:, 1], cam['H'] - qP[:, 1]]) + 15
    res['port_out'] = np.where(zP > 0, np.maximum(inside, 0) / 10.0, 0.0)
    res['port_cam'] = np.maximum(0.03 - np.linalg.norm(P[None] - cam['pos'], axis=1), 0) / 0.003
    res['reach'] = (np.maximum(q['L'] - 0.32, 0) + np.maximum(0.05 - q['L'], 0)) / 0.005
    res.update(scene_residuals(t, P, pr, d, idx, ob, cam, cfg) if extras else scene_zeros(len(idx)))
    if parts:
        res['_pr'] = pr
        res['_h'] = h
    return res


BG_STATIONS = np.arange(0.0, 0.2001, 0.005)            # shaft points (m behind the tip point) tested against the background
SCENE_KEYS = (('bg', len(BG_STATIONS)), ('punct', 1), ('in_lo', 1), ('in_hi', 1), ('enter', 1), ('hid2', 1), ('contact', 1))


def scene_zeros(N):
    return {k: np.zeros((N, w)) for k, w in SCENE_KEYS}


def ray_mesh(o, d, Xm, F):
    """Crossings of the rays o + t d (N, 3) with the triangle meshes Xm[F] (N, Nv, 3): t (N, nf) (nan = no hit), and
    the barycentric u, v of the hits."""
    A, B, C = Xm[:, F[:, 0]], Xm[:, F[:, 1]], Xm[:, F[:, 2]]
    e1, e2 = B - A, C - A
    h = np.cross(d[:, None], e2)
    a = (e1 * h).sum(-1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    sv = o[:, None] - A
    u = f * (sv * h).sum(-1)
    qv = np.cross(sv, e1)
    v = f * (qv * d[:, None]).sum(-1)
    tt = f * (e2 * qv).sum(-1)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    return np.where(hit, tt, np.nan), u, v


def exit_distance(S, d, Xm, F, eps=0.0005):
    """Distance along d from the surface point S to the next crossing of the mesh (the far wall); 0 if none."""
    tt, _, _ = ray_mesh(S + eps * d, d, Xm, F)
    tt = np.where(tt > 0, tt, np.nan)
    with np.errstate(all='ignore'):
        m = np.nanmin(tt, 1)
    return np.where(np.isfinite(m), m + eps, 0.0)


def bg_distance(bg, idx, X, cam):
    """Signed distance (m) of points X (N, K, 3) to the background along frame idx's camera rays (> 0: in front;
    nan: outside the occupancy image or behind the scope). Vectorised Background.ray_distance."""
    Xc = np.einsum('nij,nkj->nki', cam['R'], X - cam['pos'][:, None])
    z = Xc[..., 2]
    zz = np.maximum(z, 1e-4)
    u = cam['f'][:, None] * Xc[..., 0] / zz + cam['W'] / 2
    v = cam['f'][:, None] * Xc[..., 1] / zz + cam['H'] / 2
    s = bg['scale']
    h, w = bg['occ'].shape[1:]
    xi, yi = np.floor(u / s).astype(int), np.floor(v / s).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (z > 0.005)
    kk = np.broadcast_to(idx[:, None], xi.shape)
    zb = np.full(z.shape, np.nan)
    zb[ok] = bg['occ'][kk[ok], yi[ok], xi[ok]]
    return zb - z


def scene_residuals(t, P, pr, d, idx, ob, cam, cfg):
    """Scene consistency: shaft in front of the background, and the organ coupling (puncture: the shaft passes through
    one material point S of the organ, the tip lies inside the organ beyond S; contact: the jaws are on the organ)."""
    N = len(idx)
    out = scene_zeros(N)
    O = pr['O']
    if ob.get('bg') is not None and cfg['w_bg'] > 0:
        X = O[:, None] - BG_STATIONS[None, :, None] * d[:, None]
        dist = bg_distance(ob['bg'], idx, X, cam)
        need = t['radius'] + cfg['bg_margin']
        r = np.where(np.isfinite(dist), np.maximum(need - dist, 0.0), 0.0)
        out['bg'] = r * ob['bg_w'][idx][:, None] * cfg['w_bg'] / cfg['sig_bg']
    cp = ob.get('cpl')
    if cp is None:
        return out
    if cp['kind'] == 'puncture':
        ins = cp['inside'][idx]
        if ins.any():
            S = cp['S'][idx]
            v = S - P[None]
            w = v - (v * d).sum(1, keepdims=True) * d
            out['punct'][:, 0] = np.where(ins, np.linalg.norm(w, axis=1), 0.0) / cfg['sig_punct']
            ii = np.nonzero(ins)[0]
            nohit = np.zeros(0, int)
            if cfg['punct_mode'] == 'point':
                # depth along the shaft measured from the puncture point S (the line is pulled through S)
                h = ((O - S) * d).sum(1)
                room = np.zeros(N)
                room[ii] = exit_distance(S[ii], d[ii], cp['X4'][idx[ii]], cp['F']) - cp['margin']
            else:
                # 'line': from where this frame's shaft line actually enters the organ (first crossing from the port)
                # to where it leaves it (next crossing); frames whose line misses the organ fall back to S
                h = ((O - S) * d).sum(1)
                room = np.zeros(N)
                tt, _, _ = ray_mesh(np.repeat(P[None], len(ii), 0), d[ii], cp['X4'][idx[ii]], cp['F'])
                tt = np.sort(np.where(tt > 0, tt, np.inf), 1)
                hit = np.isfinite(tt[:, 1])
                Lk = ((O[ii] - P[None]) * d[ii]).sum(1)
                h[ii[hit]] = Lk[hit] - tt[hit, 0]
                room[ii[hit]] = tt[hit, 1] - tt[hit, 0] - cp['margin']
                nohit = ii[~hit]
                room[nohit] = exit_distance(S[nohit], d[nohit], cp['X4'][idx[nohit]], cp['F']) - cp['margin']
            lo = cp['h_min']
            out['in_lo'][:, 0] = np.where(ins, np.maximum(lo - h, 0.0), 0.0) / cfg['sig_inside']
            if cfg['punct_mode'] != 'point' and len(nohit):
                # the line misses the organ in a frame where the tip must be inside: pull it onto the puncture point
                dist = np.linalg.norm(w[nohit], axis=1)
                out['in_lo'][nohit, 0] = (dist + lo) / cfg['sig_inside']
            out['in_hi'][:, 0] = np.where(ins, np.maximum(h - room, 0.0), 0.0) / cfg['sig_inside']
            h0 = np.clip(np.minimum(cp['h0'], room), lo, None)
            out['hid2'][:, 0] = np.where(ins, (h - h0) / cfg['sig_hidden'], 0.0)
            if cfg['punct_mode'] == 'point':
                nrm = cp['N'][idx]
                out['enter'][:, 0] = np.where(ins, np.maximum((d * nrm).sum(1) + cfg['enter_cos'], 0.0), 0.0) / 0.1
    elif cp['kind'] == 'contact':
        on = cp['on'][idx]
        if on.any():
            tcp = O - cp['tcp_back'] * d
            ii = np.nonzero(on)[0]
            Xs = cp['X4'][idx[ii]][:, cp['surf']]
            dist = np.sqrt(((Xs - tcp[ii, None]) ** 2).sum(-1)).min(1)
            out['contact'][ii, 0] = np.maximum(dist - cfg['contact_tol'], 0.0) / cfg['sig_contact']
    return out


def _proj1(X, cam):
    Xc = np.einsum('nij,nj->ni', cam['R'], X[None] - cam['pos'])
    z = Xc[:, 2]
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return np.stack([cam['f'] * Xc[:, 0] / zz + cam['W'] / 2, cam['f'] * Xc[:, 1] / zz + cam['H'] / 2], -1), z


DATA_KEYS = ('outline', 'model', 'depth', 'free', 'outview', 'hidden', 'hidden_max', 'port_out', 'port_cam', 'reach') + \
    tuple(k for k, _ in SCENE_KEYS)
VARS = ('yaw', 'pitch', 'L', 'roll', 'jaw')


class Problem:
    """Least squares over the globals (port P; optionally kappa = apparent size / nominal, jls = jaw length shape) +
    per-frame (yaw, pitch, L, roll, jaw). `free` selects the per-frame variables; fix_port holds all globals."""

    def __init__(self, V, t, ob, cfg, R0, free=('yaw', 'pitch', 'L'), fix_port=False, fit_kappa=None, fit_jls=None):
        self.V, self.t, self.ob, self.cfg, self.R0 = V, t, ob, cfg, R0
        self.n = V.n
        self.free = [v for v in VARS if v in free and (t['jawed'] or v not in ('roll', 'jaw'))]
        self.fix_port = fix_port
        self.fit_kappa = (cfg['fit_kappa'] if fit_kappa is None else fit_kappa) and not fix_port
        self.fit_jls = (cfg['fit_jls'] if fit_jls is None else fit_jls) and not fix_port and t['jawed']
        self.gnames = [] if fix_port else (['P'] + (['kappa'] if self.fit_kappa else []) + (['jls'] if self.fit_jls else []))
        self.ng = 0 if fix_port else 3 + self.fit_kappa + self.fit_jls
        self.idx = np.arange(self.n)
        self.cam = cams_of(V, self.idx)
        dt = 1.0 / V.fps
        self.sig_acc = cfg['acc_tip'] * dt ** 2
        self.sig_dir = cfg['acc_dir'] * dt ** 2
        self.sig_roll = cfg['acc_roll'] * dt ** 2
        self.sig_jaw = cfg['acc_jaw'] * dt ** 2
        self.sig_vroll = cfg['vel_roll'] * dt
        self.sig_vjaw = cfg['vel_jaw'] * dt

    def set_ref(self, Q):
        RT = tip_frame(self.R0, Q['yaw'], Q['pitch'], Q['roll'])
        T = Q['P'][None] + Q['L'][:, None] * RT[:, :, 2]
        self.Tref = T[self.ob['vis']].mean(0) if self.ob['vis'].any() else T.mean(0)

    def pack(self, P, Q):
        g = []
        for gname in self.gnames:
            g.append(P if gname == 'P' else [Q.get(gname, 1.0)])
        return np.concatenate(g + [Q[v] for v in self.free])

    def unpack(self, x, Q0):
        n = self.n
        P = Q0['P']
        Q = {v: Q0[v].copy() for v in VARS}
        Q['kappa'], Q['jls'] = Q0.get('kappa', 1.0), Q0.get('jls', 1.0)
        o = 0
        for gname in self.gnames:
            if gname == 'P':
                P, o = x[:3], 3
            else:
                Q[gname] = x[o]
                o += 1
        for v in self.free:
            Q[v] = x[o:o + n]
            o += n
        return P, Q

    def residuals(self, x, Q0, parts=False):
        P, Q = self.unpack(x, Q0)
        r = frame_residuals(self.t, P, self.R0, Q, self.idx, self.ob, self.cam, self.cfg, kappa=Q['kappa'], jls=Q['jls'])
        data = np.concatenate([r[k].reshape(self.n, -1) for k in DATA_KEYS], 1)
        RT = tip_frame(self.R0, Q['yaw'], Q['pitch'], Q['roll'])
        d = RT[:, :, 2]
        T = P[None] + Q['L'][:, None] * d
        acc = T[2:] - 2 * T[1:-1] + T[:-2]
        sm = [(acc / self.sig_acc).ravel(),
              ((d[2:] - 2 * d[1:-1] + d[:-2]) / self.sig_dir).ravel(),
              # depth along the line of sight comes from the silhouette width alone and is noisier than the image
              # position: its acceleration gets a tighter sigma
              np.einsum('ki,ki->k', acc, self.V.R[1:-1, 2]) * self.cfg['depth_smooth'] / self.sig_acc]
        if self.t['jawed']:
            sm += [(Q['roll'][2:] - 2 * Q['roll'][1:-1] + Q['roll'][:-2]) / self.sig_roll,
                   (Q['jaw'][2:] - 2 * Q['jaw'][1:-1] + Q['jaw'][:-2]) / self.sig_jaw,
                   np.diff(Q['roll']) / self.sig_vroll, np.diff(Q['jaw']) / self.sig_vjaw]
        # weak priors on the globals: port distance from the (fixed) mean starting tip (only matters when the data
        # leave the port undetermined), apparent radius = nominal
        glob = np.array([(np.linalg.norm(P - self.Tref) - self.cfg['port_dist0']) / self.cfg['sig_port_dist'],
                         (Q['kappa'] - 1.0) / self.cfg['sig_kappa'], np.log(Q['jls']) / self.cfg['sig_jls']])
        if parts:
            out = {k: r[k] for k in DATA_KEYS}
            out['smooth'] = np.concatenate(sm)
            out['glob'] = glob
            return out
        return np.concatenate([data.ravel()] + sm + [glob])

    def sparsity(self, Q0):
        n = self.n
        x0 = self.pack(Q0['P'], Q0)
        D = sum(np.asarray(v).reshape(n, -1).shape[1] for v in
                frame_residuals(self.t, Q0['P'], self.R0, Q0, self.idx, self.ob, self.cam, self.cfg).values()
                if np.ndim(v) >= 1 and np.asarray(v).shape[0] == n)
        self.D = D
        nv = len(x0)
        po = self.ng
        nf = len(self.free)
        cols = lambda k: [po + j * n + k for j in range(nf)]
        n_sm = 7 * (n - 2) + ((2 * (n - 2) + 2 * (n - 1)) if self.t['jawed'] else 0)
        S = lil_matrix((n * D + n_sm + 3, nv), dtype=np.int8)
        for k in range(n):
            for c in cols(k):
                S[k * D:(k + 1) * D, c] = 1
            if po:
                S[k * D:(k + 1) * D, 0:po] = 1
        r0 = n * D
        for blk in range(2):                     # tip and direction acceleration
            for k in range(1, n - 1):
                rows = slice(r0 + (k - 1) * 3, r0 + k * 3)
                for kk in (k - 1, k, k + 1):
                    for c in cols(kk):
                        S[rows, c] = 1
                if po:
                    S[rows, 0:3] = 1
            r0 += 3 * (n - 2)
        for k in range(1, n - 1):                # line-of-sight acceleration
            for kk in (k - 1, k, k + 1):
                for c in cols(kk):
                    S[r0 + k - 1, c] = 1
            if po:
                S[r0 + k - 1, 0:3] = 1
        r0 += n - 2
        if self.t['jawed']:
            for v in ('roll', 'jaw'):
                for k in range(1, n - 1):
                    if v in self.free:
                        j = self.free.index(v)
                        for kk in (k - 1, k, k + 1):
                            S[r0 + k - 1, po + j * n + kk] = 1
                r0 += n - 2
            for v in ('roll', 'jaw'):
                for k in range(n - 1):
                    if v in self.free:
                        j = self.free.index(v)
                        S[r0 + k, po + j * n + k] = 1
                        S[r0 + k, po + j * n + k + 1] = 1
                r0 += n - 1
        if po:
            S[r0, 0:3] = 1
            for j, gname in enumerate(self.gnames[1:]):
                S[r0 + (1 if gname == 'kappa' else 2), 3 + j] = 1
        return S.tocsr()

    def bounds(self):
        n = self.n
        lo, hi = [], []
        for gname in self.gnames:
            if gname == 'P':
                lo.append(np.full(3, -1.0)), hi.append(np.full(3, 1.0))
            else:
                rg = self.cfg[f'{gname}_range']
                lo.append([rg[0]]), hi.append([rg[1]])
        B = dict(yaw=(-1.5, 1.5), pitch=(-1.5, 1.5), L=(0.02, 0.45), roll=(-50.0, 50.0), jaw=(0.0, max(self.t['jaw_max'], 1e-3)))
        for v in self.free:
            lo.append(np.full(n, B[v][0])), hi.append(np.full(n, B[v][1]))
        return np.concatenate(lo), np.concatenate(hi)

    def solve(self, Q0, max_nfev=60, log=print, tag=''):
        if not hasattr(self, 'Tref'):
            self.set_ref(Q0)
        x0 = self.pack(Q0['P'], Q0)
        lo, hi = self.bounds()
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        S = self.sparsity(Q0)
        t0 = time.time()
        r = least_squares(self.residuals, x0, args=(Q0,), jac_sparsity=S, bounds=(lo, hi), loss='soft_l1', f_scale=1.0,
                          x_scale='jac', max_nfev=max_nfev, method='trf')
        P, Q = self.unpack(r.x, Q0)
        Q['P'] = P
        if log:
            log(f'  [{tag}] free={self.free}+{self.gnames} cost '
                f'{0.5 * np.sum(self._rho(self.residuals(x0, Q0))):.1f} -> {r.cost:.1f}, nfev {r.nfev}, {time.time() - t0:.0f} s, '
                f'P={np.round(P * 1000, 1)} mm, kappa={Q["kappa"]:.3f}, jls={Q["jls"]:.3f}')
        return Q, float(r.cost)

    @staticmethod
    def _rho(r):
        return 2 * (np.sqrt(1 + r ** 2) - 1)


# ================================================================ initialisation
def init_lines(V, meas, t):
    """Per-frame 3D shaft lines from the mask axis and the widths (5 mm -> depth): far end E_k and unit direction
    u_k (distal), NaN where the mask gives no width line."""
    n = V.n
    E, U = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    for k, m in enumerate(meas):
        if m is None or m['wline'] is None:
            continue
        a, b, s0, s1 = m['wline']
        se = m['smax']
        sp = max(s0, se - 160)
        we, wp = a * min(se, s1) + b, a * sp + b
        if we < 4 or wp < 4:
            continue
        ze, zp = V.f[k] * 2 * t['radius'] / we, V.f[k] * 2 * t['radius'] / wp
        qe, qp = m['c'] + m['d'] * se, m['c'] + m['d'] * sp
        Xe = V.unproject(qe[0], qe[1], np.clip(ze, 0.01, 0.4), k)
        Xp = V.unproject(qp[0], qp[1], np.clip(zp, 0.01, 0.4), k)
        u = Xe - Xp
        if np.linalg.norm(u) < 1e-4:
            continue
        E[k], U[k] = Xe, u / np.linalg.norm(u)
    return E, U


def common_point(E, U, w=None):
    """Least-squares point closest to the lines E + s U (robust reweighting)."""
    ok = np.isfinite(E[:, 0])
    E, U = E[ok], U[ok]
    w = np.ones(len(E)) if w is None else w[ok]
    X = E.mean(0)
    for _ in range(8):
        Mx, rhs = np.zeros((3, 3)), np.zeros(3)
        for e, u, wi in zip(E, U, w):
            Pm = np.eye(3) - np.outer(u, u)
            Mx += wi * Pm
            rhs += wi * Pm @ e
        X = np.linalg.solve(Mx + 1e-9 * np.eye(3), rhs)
        v = X - E
        dist = np.linalg.norm(v - (v * U).sum(1, keepdims=True) * U, axis=1)
        w = 1 / np.maximum(1, dist / (np.median(dist) + 1e-4))
    return X


def interp_rows(X, ok):
    X = np.array(X, float)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return X
    for c in range(X.shape[1]):
        X[:, c] = np.interp(np.arange(len(X)), idx, X[idx, c])
    return X


def init_pose(V, meas, ob, t, P0, R0, hidden_len=0.0):
    n = V.n
    E, U = init_lines(V, meas, t)
    okE = np.isfinite(E[:, 0])
    # frames with a mask but no width line: far end at the depth interpolated from the neighbours
    if okE.sum() >= 2:
        ze = np.array([((E[k] - V.pos[k]) @ V.R[k][2]) if okE[k] else np.nan for k in range(n)])
        zi = np.interp(np.arange(n), np.nonzero(okE)[0], ze[okE])
        for k in range(n):
            if not okE[k] and ob['vis'][k]:
                q = ob['endpx'][k]
                E[k] = V.unproject(q[0], q[1], zi[k], k)
                okE[k] = True
    E = interp_rows(E, okE)
    if hidden_len:                                             # tip inside an organ: start beyond the visible end
        u = E - P0
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        E = E + np.where(ob['hid_prior'][:, None], hidden_len * u, 0.0)
    d = E - P0
    L = np.linalg.norm(d, axis=1)
    d /= L[:, None]
    yaw, pitch = angles_of(R0, d)
    Q = dict(P=P0.copy(), yaw=yaw, pitch=pitch, L=L, roll=np.zeros(n),
             jaw=np.full(n, 0.0))
    return Q, E, U


# ================================================================ roll / jaw: grid + Viterbi
def roll_jaw_grid(V, t, ob, Q, R0, cfg, n_roll=12, n_jaw=7, chunk=24):
    n = V.n
    t = scaled(t, Q.get('kappa', 1.0), Q.get('jls', 1.0))
    per = t['roll_period']
    rolls = np.linspace(0, per, n_roll, endpoint=False)
    jaws = np.linspace(0, t['jaw_max'], n_jaw)
    RR, JJ = np.meshgrid(rolls, jaws, indexing='ij')
    RR, JJ = RR.ravel(), JJ.ravel()
    S = len(RR)
    C = np.zeros((n, S))
    for a in range(0, n, chunk):
        ks = np.arange(a, min(n, a + chunk))
        idx = np.repeat(ks, S)
        q = dict(yaw=Q['yaw'][idx], pitch=Q['pitch'][idx], L=Q['L'][idx], roll=np.tile(RR, len(ks)), jaw=np.tile(JJ, len(ks)))
        r = frame_residuals(t, Q['P'], R0, q, idx, ob, cams_of(V, idx), cfg, extras=False)
        rr = np.concatenate([r['outline'], r['model']], 1)          # tool already scaled above
        C[ks] = (2 * (np.sqrt(1 + rr ** 2) - 1)).sum(1).reshape(len(ks), S)
    C[~ob['vis']] = 0
    # Viterbi: transition cost from roll (circular) and jaw steps
    dt = 1.0 / V.fps
    dr = np.abs(np.angle(np.exp(1j * (RR[:, None] - RR[None, :]) * 2 * np.pi / per))) * per / (2 * np.pi)
    dj = np.abs(JJ[:, None] - JJ[None, :])
    trans = (dr / (cfg['vel_roll'] * dt)) ** 2 + (dj / (cfg['vel_jaw'] * dt)) ** 2
    D = C[0].copy()
    back = np.zeros((n, S), int)
    for k in range(1, n):
        tot = D[:, None] + trans
        back[k] = np.argmin(tot, 0)
        D = tot[back[k], np.arange(S)] + C[k]
    s = np.zeros(n, int)
    s[-1] = int(np.argmin(D))
    for k in range(n - 1, 0, -1):
        s[k - 1] = back[k, s[k]]
    roll = unwrap_roll(RR[s], per)
    jaw = JJ[s]
    # observability: how much the frame's own cost prefers the chosen jaw (best roll for each jaw) over the others,
    # and the chosen roll over the others at the chosen jaw; roll only counts with open (or curved) jaws
    jaw_obs = np.zeros(n, bool)
    roll_obs = np.zeros(n, bool)
    Cr = C.reshape(n, n_roll, n_jaw)
    for k in range(n):
        if not ob['vis'][k] or ob['hid'][k]:
            continue
        i_r, i_j = np.unravel_index(s[k], (n_roll, n_jaw))
        cj = Cr[k].min(0)
        jaw_obs[k] = (cj.max() - cj[i_j]) > cfg['obs_gain']
        roll_obs[k] = ((Cr[k, :, i_j].max() - Cr[k, i_r, i_j]) > cfg['obs_gain']) and \
            (JJ[s[k]] >= cfg['roll_obs_min_jaw'] or t['curve'] > 0)
    return roll, jaw, dict(cost=C, best=C.min(1), state=s, jaw_observed=jaw_obs, roll_observed=roll_obs)


# ================================================================ gaps
def gap_flags(vis, max_short):
    n = len(vis)
    filled, long_gap = np.zeros(n, bool), np.zeros(n, bool)
    k = 0
    while k < n:
        if vis[k]:
            k += 1
            continue
        j = k
        while j < n and not vis[j]:
            j += 1
        if (j - k) <= max_short and k > 0 and j < n:
            filled[k:j] = True
        else:
            long_gap[k:j] = True
        k = j
    return filled, long_gap


def withdraw_long_gaps(V, t, Q, R0, long_gap, vis, step=0.002):
    """In long gaps: direction held / interpolated between the gap's ends, the tool pulled back along its shaft
    until its tip region projects outside the scope image."""
    n = V.n
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Q.items()}
    if not long_gap.any():
        return Q
    okv = vis & ~long_gap
    if okv.sum() == 0:
        return Q
    for v in ('yaw', 'pitch', 'roll', 'jaw', 'L'):
        Q[v][long_gap] = np.interp(np.nonzero(long_gap)[0], np.nonzero(okv)[0], Q[v][okv])
    bz = ~V.valid
    for k in np.nonzero(long_gap)[0]:
        for _ in range(200):
            q = {v: np.array([Q[v][k]]) for v in VARS}
            pr = project_prims(t, Q['P'], R0, q, cams_of(V, np.array([k])))
            m = render(pr, 0, V.H, V.W)
            if not (m & ~bz).any() or Q['L'][k] <= 0.03:
                break
            Q['L'][k] -= step
    return Q


# ================================================================ metrics
def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(median=round(float(np.median(x)), 4), mean=round(float(x.mean()), 4), p90=round(float(np.percentile(x, 90)), 4),
                max=round(float(x.max()), 4), n=int(len(x)))


def thirds(n):
    a, b = n // 3, 2 * n // 3
    return {f'0-{a - 1}': (0, a), f'{a}-{b - 1}': (a, b), f'{b}-{n - 1}': (b, n)}


def evaluate(V, tools, cfg):
    """Per frame, per instrument: IoU (raw / visible part), far-end error, shaft angle error, centre-line offset,
    port consistency, smoothness, depth consistency. tools[name] = dict(t, Q, R0, ob, meas)."""
    n, H, W = V.n, V.H, V.W
    names = list(tools)
    res = {nm: dict(iou=np.full(n, np.nan), iou_vis=np.full(n, np.nan), tip_px=np.full(n, np.nan),
                    tip_signed_px=np.full(n, np.nan), hidden_px=np.full(n, np.nan), hidden_mm=np.full(n, np.nan),
                    ang_deg=np.full(n, np.nan), offset_px=np.full(n, np.nan), plane_mm=np.full(n, np.nan),
                    line_mm=np.full(n, np.nan), depth_res_mm=np.full(n, np.nan), shaft_behind_tissue=np.full(n, np.nan))
           for nm in names}
    for k in range(n):
        sil, zfun = {}, {}
        for nm in names:
            T = tools[nm]
            q = {v: np.array([T['Q'][v][k]]) for v in VARS}
            cam = cams_of(V, np.array([k]))
            pr = project_prims(T['t'], T['Q']['P'], T['R0'], q, cam)
            sil[nm] = render(pr, 0, H, W) & V.valid
            zfun[nm] = (pr['O'], pr['RT'][:, :, 2], cam)
        # z-buffer the two instruments where their silhouettes overlap
        vis_sil = {nm: sil[nm].copy() for nm in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ov = sil[a] & sil[b]
                if ov.any():
                    ys, xs = np.nonzero(ov)
                    uv = np.stack([xs, ys], 1)[None].astype(float)
                    za, _ = axis_depth(zfun[a][0], zfun[a][1], zfun[a][2], uv)
                    zb, _ = axis_depth(zfun[b][0], zfun[b][1], zfun[b][2], uv)
                    front_a = za[0] <= zb[0]
                    vis_sil[a][ys[~front_a], xs[~front_a]] = False
                    vis_sil[b][ys[front_a], xs[front_a]] = False
        excl = np.zeros((H, W), bool)
        for o in names:
            if tools[o]['meas'][k] is not None:
                excl |= tools[o]['meas'][k]['mask']
        excl = cv2.dilate(excl.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        for nm in names:
            T = tools[nm]
            ob, m = T['ob'], T['meas'][k]
            R = res[nm]
            mk = m['mask'] if m is not None else np.zeros((H, W), bool)
            s = vis_sil[nm]
            if (mk.sum() > 50 or s.sum() > 50) and not ob['trimmed'][k]:
                R['iou'][k] = (s & mk).sum() / max(1, (s | mk).sum())
                if m is not None and ob['hid'][k]:
                    yy, xx = np.mgrid[0:H, 0:W]
                    keep = ((xx - m['c'][0]) * m['d'][0] + (yy - m['c'][1]) * m['d'][1]) < m['smax'] + 1
                    R['iou_vis'][k] = (s & mk & keep).sum() / max(1, ((s & keep) | mk).sum())
                else:
                    R['iou_vis'][k] = R['iou'][k]
            if m is None:
                continue
            O, d, cam = zfun[nm]
            qt, _ = V.project(T['Q']['P'] + T['Q']['L'][k] * d[0], k)
            qb, _ = V.project(T['Q']['P'] + (T['Q']['L'][k] - 0.03) * d[0], k)
            e = qt - qb
            e /= np.linalg.norm(e) + 1e-12
            dm = m['d_metric']
            R['ang_deg'][k] = float(np.degrees(np.arccos(np.clip(e @ dm, -1, 1))))
            # far end of the model silhouette along the mask axis vs the mask's far end
            ys, xs = np.nonzero(s)
            if len(xs):
                sm = float(np.percentile((np.stack([xs, ys], 1) - m['c']) @ m['d'], 99.8))
                sgn = sm - m['smax']
                if ob['hid'][k]:
                    R['hidden_px'][k] = sgn
                else:
                    R['tip_px'][k] = abs(sgn)
                    R['tip_signed_px'][k] = sgn
            S = m['S']
            okr = (S[:, 3] == 0) & (S[:, 4] == 0) & (S[:, 0] < m['smax'] - 1.5 * m['w_shaft'])
            if okr.any():
                pts = m['c'] + S[okr, :1] * m['d'] + S[okr, 1:2] * np.array([-m['d'][1], m['d'][0]])
                nl = np.array([-e[1], e[0]])
                R['offset_px'][k] = float(np.median(np.abs((pts - qt) @ nl)))
            # port consistency, depth-free: distance of the port from the plane through the camera centre and the
            # observed 2D axis
            q0, q1 = m['c'] - 50 * dm, m['c'] + 50 * dm
            X0 = V.unproject(q0[0], q0[1], 0.1, k) - V.pos[k]
            X1 = V.unproject(q1[0], q1[1], 0.1, k) - V.pos[k]
            nr = np.cross(X0, X1)
            nr /= np.linalg.norm(nr)
            R['plane_mm'][k] = abs((T['Q']['P'] - V.pos[k]) @ nr) * 1000
            if ob['hid_prior'][k]:
                _, s_end = axis_depth(O, d, cam, m['tip'][None, None])
                R['hidden_mm'][k] = -s_end[0, 0] * 1000
            if ob['zw'][k].any():
                zl, _ = axis_depth(O, d, cam, ob['zpx'][k][None])
                w = ob['zw'][k] > 0
                R['depth_res_mm'][k] = float(np.median((zl[0] - T['t']['radius'] - ob['zobs'][k])[w]) * 1000)
            # free space: tissue just beside the visible shaft must not lie in front of the shaft
            nr2 = np.array([-m['d'][1], m['d'][0]])
            sel = okr & (S[:, 2] > 6)
            if sel.sum() >= 3:
                c2 = m['c'] + S[sel, :1] * m['d'] + S[sel, 1:2] * nr2
                zs, _ = axis_depth(O, d, cam, c2[None])
                beh = []
                for sg in (-1, 1):
                    p = np.round(c2 + sg * nr2 * (S[sel, 2:3] / 2 + 8)).astype(int)
                    ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
                    p, zz = p[ok], zs[0][ok]
                    ok2 = V.valid[p[:, 1], p[:, 0]] & ~excl[p[:, 1], p[:, 0]]
                    beh += list(V.depth(k)[p[ok2, 1], p[ok2, 0]] < zz[ok2] - T['t']['radius'] - cfg['free_tol'])
                if beh:
                    R['shaft_behind_tissue'][k] = float(np.mean(beh))
        # independent per-frame lines (widths) -> distance from the port
    for nm in names:
        T = tools[nm]
        E, U = init_lines(V, T['meas'], T['t'])
        P = T['Q']['P']
        for k in range(n):
            if np.isfinite(E[k, 0]):
                v = P - E[k]
                res[nm]['line_mm'][k] = float(np.linalg.norm(v - (v @ U[k]) * U[k]) * 1000)
    return res


def smooth_stats(V, Q, R0, mask):
    dt = 1.0 / V.fps
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    d = RT[:, :, 2]
    T = Q['P'][None] + Q['L'][:, None] * d
    v = np.linalg.norm(np.diff(T, axis=0), axis=1)
    a = np.linalg.norm(T[2:] - 2 * T[1:-1] + T[:-2], axis=1)
    w = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1)))
    mv = mask[1:] & mask[:-1]
    ma = mask[2:] & mask[1:-1] & mask[:-2]
    return dict(tip_speed_mm_per_frame=summ(v[mv] * 1000), tip_speed_m_s=summ(v[mv] / dt),
                tip_acc_mm_per_frame2=summ(a[ma] * 1000), tip_acc_m_s2=summ(a[ma] / dt ** 2),
                shaft_turn_deg_per_frame=summ(w[mv]),
                jaw_rate_rad_s=summ(np.abs(np.diff(Q['jaw']))[mv] / dt),
                roll_rate_rad_s=summ(np.abs(np.diff(Q['roll']))[mv] / dt))


# ================================================================ MJCF contract
def _quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def mjcf(name, tool, port, R0, rgba_shaft='0.35 0.38 0.40 1', rgba_jaw='0.75 0.76 0.78 1', collide=True,
         contype=2, conaffinity=1, gains=None, insertion_range=(0.0, 0.45)):
    """MJCF fragments of one instrument about its port: dict(body, actuator, tendon, equality, contact, joints,
    actuators). Angles are in radians: the enclosing model must use <compiler angle="radian"/>. Joints (and position actuators of the same name): <name>_yaw, _pitch (hinges, rad), _insertion (slide,
    m, port -> tip point), _roll (hinge, rad), and the jaw: <name>_jaw = total opening (rad; a fixed tendon over the two
    jaw hinges for double-action jaws, the moving jaw's hinge for single-action). Site <name>_tip = the tip point;
    model.npz 'joints' (n, 5) are the joint values in this order. Geoms: <name>_shaft (cylinder), <name>_tip_cap
    (suction), <name>_jaw_{a,b}{0,1} (capsules)."""
    g = dict(yaw=(4.0, 0.15, 1.0), pitch=(4.0, 0.15, 1.0), insertion=(300.0, 8.0, 4.0), roll=(0.5, 0.01, 0.2),
             jaw=(0.6, 0.02, 0.1))
    g.update(gains or {})
    R, jl, p = tool['radius'], tool['jaw_len'], name
    con = f'contype="{contype}" conaffinity="{conaffinity}"' if collide else 'contype="0" conaffinity="0"'
    q = _quat(R0)
    P = np.asarray(port, float)
    ins_lo, ins_hi = insertion_range
    geoms = []
    if tool['jawed']:
        geoms.append(f'<geom name="{p}_shaft" type="cylinder" fromto="0 0 {-(jl + tool["shaft_len"]):.5f} 0 0 {-jl:.5f}" '
                     f'size="{R:.5f}" mass="0.05" rgba="{rgba_shaft}" {con}/>')
    else:
        geoms.append(f'<geom name="{p}_shaft" type="capsule" fromto="0 0 {-(R + tool["shaft_len"]):.5f} 0 0 {-R:.5f}" '
                     f'size="{R:.5f}" mass="0.05" rgba="{rgba_shaft}" {con}/>')
    jaws = ''
    if tool['jawed']:
        pts, rad = jaw_geometry(tool)
        lim = tool['jaw_max'] / 2 if tool['jaw_mode'] == 'double' else tool['jaw_max']
        jfr = 'friction="1.5 0.02 0.0002" condim="4"'
        for j, (sgn, axis) in enumerate(((1, '0 1 0'), (-1, '0 -1 0'))):
            pp = pts * np.array([sgn, 1.0, 1.0])
            gg = ''.join(f'<geom name="{p}_jaw_{"ab"[j]}{i}" type="capsule" fromto="{" ".join(f"{v:.5f}" for v in pp[i])} '
                         f'{" ".join(f"{v:.5f}" for v in pp[i + 1])}" size="{0.5 * (rad[i] + rad[i + 1]):.5f}" mass="0.002" '
                         f'rgba="{rgba_jaw}" {jfr} {con}/>' for i in range(2))
            moving = tool['jaw_mode'] == 'double' or j == 0
            jt = (f'<joint name="{p}_jaw_{"ab"[j]}" type="hinge" axis="{axis}" range="0 {lim:.4f}" damping="0.01" '
                  f'armature="0.0002"/>') if moving else ''
            jaws += f"""
            <body name="{p}_jaw_{"ab"[j]}" pos="0 0 {-jl:.5f}" gravcomp="1">
              {jt}
              {gg}
            </body>"""
    inertial = '<inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>'
    body = f"""
  <body name="{p}_port" pos="{P[0]:.5f} {P[1]:.5f} {P[2]:.5f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
    <body name="{p}_yaw_link" gravcomp="1">
      <joint name="{p}_yaw" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.02" armature="0.0005"/>
      {inertial}
      <body name="{p}_pitch_link" gravcomp="1">
        <joint name="{p}_pitch" type="hinge" axis="1 0 0" range="-1.5 1.5" damping="0.02" armature="0.0005"/>
        {inertial}
        <body name="{p}_insert_link" gravcomp="1">
          <joint name="{p}_insertion" type="slide" axis="0 0 1" range="{ins_lo} {ins_hi}" damping="1" armature="0.01"/>
          {inertial}
          <body name="{p}_roll_link" gravcomp="1">
            <joint name="{p}_roll" type="hinge" axis="0 0 1" limited="false" damping="0.005" armature="0.0002"/>
            {''.join(geoms)}
            <site name="{p}_tip" pos="0 0 0" size="0.0008" rgba="1 1 0 0"/>{jaws}
          </body>
        </body>
      </body>
    </body>
  </body>"""
    act = ''.join(f'\n  <position name="{p}_{j}" joint="{p}_{j}" kp="{g[j][0]}" kv="{g[j][1]}" forcerange="{-g[j][2]} {g[j][2]}"/>'
                  for j in ('yaw', 'pitch', 'insertion', 'roll'))
    tendon = equality = ''
    contact = ''
    if tool['jawed']:
        if tool['jaw_mode'] == 'double':
            tendon = f"""
  <fixed name="{p}_jaw"><joint joint="{p}_jaw_a" coef="1"/><joint joint="{p}_jaw_b" coef="1"/></fixed>"""
            equality = f'\n  <joint joint1="{p}_jaw_a" joint2="{p}_jaw_b"/>'
            act += f'\n  <position name="{p}_jaw" tendon="{p}_jaw" kp="{g["jaw"][0]}" kv="{g["jaw"][1]}" forcerange="{-g["jaw"][2]} {g["jaw"][2]}"/>'
        else:
            act += f'\n  <position name="{p}_jaw" joint="{p}_jaw_a" kp="{g["jaw"][0]}" kv="{g["jaw"][1]}" forcerange="{-g["jaw"][2]} {g["jaw"][2]}"/>'
        contact = f"""
  <exclude body1="{p}_jaw_a" body2="{p}_jaw_b"/>
  <exclude body1="{p}_jaw_a" body2="{p}_roll_link"/>
  <exclude body1="{p}_jaw_b" body2="{p}_roll_link"/>"""
    joints = [f'{p}_yaw', f'{p}_pitch', f'{p}_insertion', f'{p}_roll'] + ([f'{p}_jaw'] if tool['jawed'] else [])
    return dict(body=body, actuator=act, tendon=tendon, equality=equality, contact=contact, joints=joints,
                actuators=[f'{p}_{j}' for j in ('yaw', 'pitch', 'insertion', 'roll')] + ([f'{p}_jaw'] if tool['jawed'] else []))


def tool_from_npz(z, name):
    """Tool dict of instrument `name` from a model.npz (the contract)."""
    pr = json.loads(str(z[f'{name}__params']))
    return make_tool(pr['type'], 2 * pr['radius'], **{k: pr[k] for k in ('jaw_len', 'jaw_rb', 'jaw_rt', 'curve', 'jaw_mode', 'jaw_max')})


def mjcf_from_npz(path, names=None, **kw):
    """MJCF fragments of the instruments of a model.npz: {name: fragments} (see mjcf())."""
    z = np.load(path, allow_pickle=False)
    names = names or [str(s) for s in z['names']]
    return {nm: mjcf(nm, tool_from_npz(z, nm), z[f'{nm}__port'], z[f'{nm}__R0'], **kw) for nm in names}


def standalone_xml(frags, timestep=0.002):
    """A minimal MuJoCo model with only the instruments (for checks)."""
    b = ''.join(f['body'] for f in frags.values())
    a = ''.join(f['actuator'] for f in frags.values())
    t = ''.join(f['tendon'] for f in frags.values())
    e = ''.join(f['equality'] for f in frags.values())
    c = ''.join(f['contact'] for f in frags.values())
    return f"""<mujoco model="instruments">
 <compiler angle="radian"/>
 <option timestep="{timestep}" gravity="0 0 -9.81"/>
 <worldbody>{b}
 </worldbody>
 <tendon>{t}
 </tendon>
 <equality>{e}
 </equality>
 <contact>{c}
 </contact>
 <actuator>{a}
 </actuator>
</mujoco>"""


def set_qpos(m, d, name, joints_k, jawed, mode):
    """Write one frame's joint values (yaw, pitch, insertion, roll, jaw) of instrument `name` into qpos."""
    for j, jn in enumerate(('yaw', 'pitch', 'insertion', 'roll')):
        d.qpos[m.joint(f'{name}_{jn}').qposadr[0]] = joints_k[j]
    if jawed:
        if mode == 'double':
            d.qpos[m.joint(f'{name}_jaw_a').qposadr[0]] = joints_k[4] / 2
            d.qpos[m.joint(f'{name}_jaw_b').qposadr[0]] = joints_k[4] / 2
        else:
            d.qpos[m.joint(f'{name}_jaw_a').qposadr[0]] = joints_k[4]


def mujoco_check(path, every=5):
    """Load the instruments of model.npz in MuJoCo, set every `every`-th frame's joints, compare the tip site and the
    shaft axis with the contract's tip / dir (max error mm / deg); also step the position servos for 1 s holding
    one frame (tracking error)."""
    import mujoco
    z = np.load(path)
    names = [str(s) for s in z['names']]
    frags = mjcf_from_npz(path)
    m = mujoco.MjModel.from_xml_string(standalone_xml(frags))
    d = mujoco.MjData(m)
    out = {}
    for nm in names:
        t = tool_from_npz(z, nm)
        J = z[f'{nm}__joints']
        e_tip, e_dir = [], []
        for k in range(0, len(J), every):
            set_qpos(m, d, nm, J[k], t['jawed'], t['jaw_mode'])
            mujoco.mj_kinematics(m, d)
            tip = d.site(f'{nm}_tip').xpos
            ax = d.site(f'{nm}_tip').xmat.reshape(3, 3)[:, 2]
            e_tip.append(np.linalg.norm(tip - z[f'{nm}__tip'][k]) * 1000)
            e_dir.append(np.degrees(np.arccos(np.clip(ax @ z[f'{nm}__dir'][k], -1, 1))))
        out[nm] = dict(fk_tip_err_mm=round(float(np.max(e_tip)), 6), fk_dir_err_deg=round(float(np.max(e_dir)), 6))
    # servo replay: position servos fed with the per-frame joint targets at the clip's frame rate
    fps = float(z['fps'])
    mujoco.mj_resetData(m, d)
    for nm in names:
        t = tool_from_npz(z, nm)
        set_qpos(m, d, nm, z[f'{nm}__joints'][0], t['jawed'], t['jaw_mode'])
    steps = max(1, int(round(1.0 / fps / m.opt.timestep)))
    err = {nm: [] for nm in names}
    for k in range(len(z[f'{names[0]}__joints'])):
        for nm in names:
            for j, a in enumerate(frags[nm]['actuators']):
                d.ctrl[m.actuator(a).id] = z[f'{nm}__joints'][k][j]
        for _ in range(steps):
            mujoco.mj_step(m, d)
        for nm in names:
            err[nm].append(np.linalg.norm(d.site(f'{nm}_tip').xpos - z[f'{nm}__tip'][k]) * 1000)
    for nm in names:
        e = np.array(err[nm])
        out[nm]['servo_replay_tip_err_mm'] = dict(median=round(float(np.median(e)), 3), p95=round(float(np.percentile(e, 95)), 3),
                                                  max=round(float(e.max()), 3))
    # servo hold: the middle frame for 1 s
    mujoco.mj_resetData(m, d)
    kmid = len(z[f'{names[0]}__joints']) // 2
    for nm in names:
        t = tool_from_npz(z, nm)
        set_qpos(m, d, nm, z[f'{nm}__joints'][kmid], t['jawed'], t['jaw_mode'])
        for j, a in enumerate(frags[nm]['actuators']):
            d.ctrl[m.actuator(a).id] = z[f'{nm}__joints'][kmid][j]
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    for nm in names:
        out[nm]['servo_hold_tip_drift_mm'] = round(float(np.linalg.norm(d.site(f'{nm}_tip').xpos - z[f'{nm}__tip'][kmid]) * 1000), 3)
    return out


# ================================================================ clip-level driver
DEFAULT = dict(organ_solid=True, w_bg=1.0, bg_margin=0.0005, sig_bg=0.001, sig_punct=0.001, sig_inside=0.0005, enter_cos=0.17,
               contact_tol=0.0025, sig_contact=0.001, couple=False, organ_version=None, puncture_topk=5,
               punct_h_min=0.003, punct_wall=0.001, punct_mode='line', sig_kappa_coupled=0.25, contact_gate=0.005, first_frame=0,
               M=64, Jz=10, Jf=10, min_area=150, blunt_ratio=0.78, sig_b=1.5, sig_m=2.0, sig_z=0.008, w_depth=0.3,
               w_free=1.0, sig_free=0.003, free_tol=0.002, free_off=16, fit_kappa=False, sig_kappa=0.1, kappa_range=(0.6, 1.4),
               kappa_free_q=0.10, kappa_free_min=0.6, kappa_apply_below=0.97,
               fit_jls=True, sig_jls=0.3, jls_range=(0.5, 2.2),
               hidden_len=0.018, sig_hidden=0.004, acc_tip=0.6, depth_smooth=3.0, acc_dir=2.0, acc_roll=40.0, acc_jaw=40.0,
               vel_roll=3.0, vel_jaw=4.0, obs_gain=15.0, roll_obs_min_jaw=0.3, port_dist0=0.15, sig_port_dist=0.06,
               port_starts=(None, 0.10, 0.16, 0.24), max_nfev_a=40, max_nfev_c=80, short_gap_s=0.5,
               profile_mm=(-40, -20, 20, 40))
CLIP_CFG = {
    'liver_s4': dict(trim=[(175, 176)]),
}


def instrument_list(clip, V):
    """[(mask name, library type, diameter m, evidence)] from the instrument agent's prompts and the scene spec."""
    ip = json.loads((OUT / clip / 'seg' / 'instrument_prompts.json').read_text())
    spec = json.loads((OUT / clip / 'spec' / 'scene_spec.json').read_text())
    out = []
    for key, v in ip['instruments'].items():
        nm = f'instrument_{key}'
        if nm not in V.instrument_names:
            continue
        kind = lib_type(v.get('type', key))
        same = [s for s in spec.get('instruments', []) if s.get('shaft_diameter_mm') and lib_type(s.get('type', '')) == kind
                and kind in str(s.get('type', '')).lower().replace(' ', '_')]
        anyd = [s for s in spec.get('instruments', []) if s.get('shaft_diameter_mm')]
        if same:
            diam, src = same[0]['shaft_diameter_mm'], f"scene_spec instrument '{same[0].get('id')}'"
        elif anyd:
            diam, src = anyd[0]['shaft_diameter_mm'], f"scene_spec default (instrument '{anyd[0].get('id')}'; none listed for this type)"
        else:
            diam, src = 5, 'default 5 mm'
        out.append(dict(name=nm, key=key, type=kind, diameter=diam / 1000, diameter_src=src,
                        agent_type=v.get('type'), visible_frames=v.get('visible_frames')))
    return out, ip


def hidden_info(clip, ip, name_key):
    """Frames where the instrument agent recorded the tip inside an organ (puncture / opening)."""
    pu = ip.get('puncture')
    if not pu or pu.get('instrument') != name_key:
        return []
    fr = [tuple(pu['frames'])] + ([tuple(pu['reinsertion']['frames'])] if pu.get('reinsertion') else [])
    return [k for a, b in fr for k in range(a, b + 1)]


def ray_hits(o, d, X, F):
    """Distances along the ray o + s d (s > 0) of its crossings with the triangles F of vertices X (Moller-Trumbore)."""
    A, B, C = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    e1, e2 = B - A, C - A
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    sv = o - A
    u = f * (sv * h).sum(1)
    q = np.cross(sv, e1)
    v = f * (q @ d)
    t = f * (e2 * q).sum(1)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return t[hit]


def organ_far_wall(V, clip, Q, R0, ob, log):
    """Upper bound of the hidden length (m) per frame from the primary organ's 4D mesh (latest organ version): the
    distance along the shaft from the visible end to the organ's far wall, minus 3 mm. Inf when unavailable."""
    hmax = np.full(V.n, np.inf)
    sel = json.loads((OUT / clip / 'organs' / 'selection.json').read_text()) if (OUT / clip / 'organs' / 'selection.json').exists() else None
    if not sel:
        return hmax, None
    org = sel['fit'][0]['name']
    vers = sorted((OUT / clip / 'organs' / org).glob('v*/model.npz'))
    if not vers:
        return hmax, None
    z = np.load(vers[-1])
    F = z['faces']
    used = str(vers[-1].relative_to(ROOT))
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    for k in np.nonzero(ob['hid_prior'])[0]:
        d = RT[k, :, 2]
        O = Q['P'] + Q['L'][k] * d
        cam = cams_of(V, np.array([k]))
        _, s_end = axis_depth(O[None], d[None], cam, ob['endpx'][k][None, None])
        Xe = O + s_end[0, 0] * d                               # the visible end on the shaft axis
        s = ray_hits(Xe - 0.01 * d, d, z['verts4d'][k].astype(float), F) - 0.01
        s = s[s > 0.002]
        if len(s):
            hmax[k] = float(s.max()) - 0.003
    log(f'  organ far wall from {used}: hidden length cap median {np.nanmedian(np.where(np.isfinite(hmax), hmax, np.nan)) * 1000:.1f} mm '
        f'over {np.isfinite(hmax).sum()} / {ob["hid_prior"].sum()} frames')
    return hmax, used


# ================================================================ organ coupling (puncture site / jaws on the organ)
def load_organ(clip, ver=None):
    """The primary organ's 4D model (organ agent): newest finished version unless pinned."""
    sel = json.loads((OUT / clip / 'organs' / 'selection.json').read_text())
    org = sel['fit'][0]['name']
    d = OUT / clip / 'organs' / org
    vs = sorted(p.name for p in d.glob('v[0-9][0-9]') if (p / 'model.npz').exists())
    ver = ver or vs[-1]
    z = np.load(d / ver / 'model.npz')
    F = z['faces'].astype(np.int64)
    return dict(name=org, version=ver, path=str((d / ver).relative_to(ROOT)), X4=z['verts4d'].astype(np.float64), F=F,
                rest=z['rest_verts'].astype(np.float64), surf=np.unique(F))


def face_normals(X4, F, f):
    """Outward unit normal of face f in every frame (the organ meshes are oriented outward: positive volume)."""
    A, B, C = X4[:, F[f, 0]], X4[:, F[f, 1]], X4[:, F[f, 2]]
    n = np.cross(B - A, C - A)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def material_point(organ, f, b, frames=None):
    X4 = organ['X4'] if frames is None else organ['X4'][frames]
    return np.einsum('j,njk->nk', b, X4[:, organ['F'][f]])


def puncture_candidates(organ):
    """Material points on the organ surface: every face at its centroid and 3 interior barycentric points."""
    B = np.array([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    nf = len(organ['F'])
    f = np.repeat(np.arange(nf), len(B))
    b = np.tile(B, (nf, 1))
    rest = np.einsum('nj,njk->nk', b, organ['rest'][organ['F'][f]])
    return f, b, rest


def puncture_proxy(V, organ, ob, frames, cf, cb):
    """Per candidate and frame: the image distance (px) of the candidate from the mask's visible end and from the
    mask's axis line, combined robustly (Cauchy-like rho); summed over `frames`."""
    cost = np.zeros(len(cf))
    rho = lambda x: 2 * (np.sqrt(1 + x ** 2) - 1)
    for k in frames:
        X = np.einsum('nj,njk->nk', cb, organ['X4'][k][organ['F'][cf]])
        q, z = V.project(X, k)
        e, c, dd = ob['endpx'][k], ob['axis_c'][k], ob['axis_d'][k]
        nrm = np.array([-dd[1], dd[0]])
        c1 = np.linalg.norm(q - e, axis=1)
        c2 = np.abs((q - c) @ nrm)
        cost += rho(c1 / 5.0) + rho(c2 / 2.0) + np.where(z > 0, 0, 100.0)
    return cost


def nms_rest(order, rest, k, min_mm=4.0):
    out = []
    for i in order:
        if all(np.linalg.norm(rest[i] - rest[j]) * 1000 >= min_mm for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return out


def puncture_setup(organ, f, b, inside, t, cfg):
    n = organ['X4'].shape[0]
    return dict(kind='puncture', face=int(f), bary=np.asarray(b, float), S=material_point(organ, f, b),
                N=face_normals(organ['X4'], organ['F'], f), inside=inside, X4=organ['X4'], F=organ['F'],
                h_min=cfg['punct_h_min'], margin=t['radius'] + cfg['punct_wall'], h0=cfg['hidden_len'], n=n)


def pose_through(Q, R0, S, inside, organ, cp, t):
    """Start poses: in the inside frames the shaft from the port through S, the tip h0 (capped by the far wall)
    beyond S."""
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Q.items()}
    ii = np.nonzero(inside)[0]
    d = S[ii] - Q['P']
    LS = np.linalg.norm(d, axis=1)
    d /= LS[:, None]
    tex = exit_distance(S[ii], d, organ['X4'][ii], organ['F'])
    h = np.clip(np.minimum(cp['h0'], tex - cp['margin']), cp['h_min'], None)
    Q['L'][ii] = LS + h
    Q['yaw'][ii], Q['pitch'][ii] = angles_of(R0, d)
    return Q


def puncture_diagnostics(V, t, organ, cp, Q, R0, ob):
    """Per frame: distance of the shaft line from the puncture point (mm), tip beyond the puncture (mm), far-wall
    distance (mm), tip inside the organ (parity test), the entry crossing of the shaft line (port -> tip) and its
    distance from the puncture in rest coordinates (mm, 'entry wander'), shaft points behind the background."""
    n = V.n
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    d = RT[:, :, 2]
    P = Q['P']
    T = P + Q['L'][:, None] * d
    S = cp['S']
    v = S - P
    line = np.linalg.norm(v - (v * d).sum(1, keepdims=True) * d, axis=1) * 1000
    h = ((T - S) * d).sum(1) * 1000
    tex = exit_distance(S, d, organ['X4'], organ['F']) * 1000
    # tip inside: parity of crossings ahead of the tip along the shaft
    tt, _, _ = ray_mesh(T, d, organ['X4'], organ['F'])
    inside_geom = (np.nansum(tt > 0, 1) % 2) == 1
    # entry crossing of the line from the port
    tt, u, w = ray_mesh(np.repeat(P[None], n, 0), d, organ['X4'], organ['F'])
    tt = np.where(tt > 0, tt, np.nan)
    wander = np.full(n, np.nan)
    entry_world = np.full((n, 3), np.nan)
    rest_p = np.einsum('j,jk->k', cp['bary'], organ['rest'][organ['F'][cp['face']]])
    for k in range(n):
        if not np.isfinite(tt[k]).any():
            continue
        fi = int(np.nanargmin(tt[k]))
        bb = np.array([1 - u[k, fi] - w[k, fi], u[k, fi], w[k, fi]])
        entry_world[k] = P + tt[k, fi] * d[k]
        wander[k] = np.linalg.norm(bb @ organ['rest'][organ['F'][fi]] - rest_p) * 1000
    # along this frame's own line: tip beyond the entry crossing and the organ's thickness there
    h_line = np.full(n, np.nan)
    thick = np.full(n, np.nan)
    ts = np.sort(np.where(np.isfinite(tt), tt, np.inf), 1)
    hit = np.isfinite(ts[:, 1])
    h_line[hit] = (Q['L'][hit] - ts[hit, 0]) * 1000
    thick[hit] = (ts[hit, 1] - ts[hit, 0]) * 1000
    # organ correction: the vector from the puncture material point to the nearest point of the shaft line (what the
    # organ model would have to move at the puncture for the line to pass through it)
    corr = -(v - (v * d).sum(1, keepdims=True) * d)
    return dict(line_mm=line, h_mm=h, exit_mm=tex, inside_geom=inside_geom, wander_mm=wander, entry_world=entry_world,
                h_line_mm=h_line, thick_mm=thick, correction=corr)


def bg_report(V, bgd, t, Q, R0, frames):
    """Per frame the smallest distance (mm) of the shaft axis points (tip .. 200 mm back) in front of the background
    and the distance from the tip (mm) of the first point behind it (nan: none)."""
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    d = RT[:, :, 2]
    T = Q['P'] + Q['L'][:, None] * d
    X = T[:, None] - BG_STATIONS[None, :, None] * d[:, None]
    idx = np.arange(V.n)
    dist = bg_distance(bgd, idx, X, cams_of(V, idx))
    mind = np.where(np.isfinite(dist).any(1), np.nanmin(np.where(np.isfinite(dist), dist, np.inf), 1), np.nan) * 1000
    first = np.full(V.n, np.nan)
    for k in range(V.n):
        b = np.nonzero(np.isfinite(dist[k]) & (dist[k] < 0))[0]
        if len(b):
            first[k] = BG_STATIONS[b[0]] * 1000
    return dict(min_mm=np.where(frames, mind, np.nan), first_behind_mm=np.where(frames, first, np.nan))


def couple_puncture(V, clip, t, ob, cfg, R0, Q, organ, hidden_frames, log):
    """One puncture site for all insertion phases: choose the organ material point from the masks' visible ends and
    axes (proxy), refit the K best candidates with the shaft through that point while inside, the tip inside the organ
    beyond it (hidden-length prior capped by the far wall), the apparent size free (the organ now gives the depth),
    and keep the best."""
    n = V.n
    inside = np.zeros(n, bool)
    inside[[k for k in hidden_frames if k < n]] = True
    inside &= ~ob['trimmed']
    sel = np.nonzero(inside & ob['vis'] & (np.arange(n) >= cfg['first_frame']))[0]
    cf, cb, crest = puncture_candidates(organ)
    cost = puncture_proxy(V, organ, ob, sel, cf, cb)
    # per insertion phase (runs of inside frames), the phase's own best candidate (diagnostic)
    phases = [(a, b) for a, b in _runs(inside)]
    per_phase = []
    for a, b in phases:
        fr = sel[(sel >= a) & (sel <= b)]
        if len(fr):
            cph = puncture_proxy(V, organ, ob, fr, cf, cb)
            i = int(np.argmin(cph))
            per_phase.append(dict(frames=[int(a), int(b)], face=int(cf[i]), rest_mm=[round(float(x) * 1000, 2) for x in crest[i]],
                                  proxy_px_cost=round(float(cph[i] / len(fr)), 3)))
    sep = (float(np.linalg.norm(np.array(per_phase[0]['rest_mm']) - np.array(per_phase[1]['rest_mm'])))
           if len(per_phase) >= 2 else None)
    log(f"  puncture: {len(cf)} candidate material points on {organ['path']}; phase-wise best points "
        f"{[p['rest_mm'] for p in per_phase]} ({sep if sep is None else round(sep, 1)} mm apart in rest coordinates)")
    order = np.argsort(cost)
    top = nms_rest(order, crest, cfg['puncture_topk'])
    cfgC = dict(cfg, sig_kappa=cfg['sig_kappa_coupled'])
    best = None
    tried = []
    for i in top:
        cp = puncture_setup(organ, cf[i], cb[i], inside, t, cfg)
        ob['cpl'] = cp
        Q0 = pose_through(Q, R0, cp['S'], inside, organ, cp, scaled(t, Q.get('kappa', 1.0)))
        Qi, ci = Problem(V, t, ob, cfgC, R0, free=('yaw', 'pitch', 'L'), fit_kappa=True).solve(
            Q0, max_nfev=cfg['max_nfev_c'], log=log, tag=f"puncture cand face {cf[i]} (proxy {cost[i] / max(1, len(sel)):.2f})")
        tried.append(dict(face=int(cf[i]), bary=[round(float(x), 3) for x in cb[i]], proxy=round(float(cost[i] / max(1, len(sel))), 3),
                          cost=round(ci, 1), rest_mm=[round(float(x) * 1000, 2) for x in crest[i]]))
        if best is None or ci < best[1]:
            best = (Qi, ci, cp, i)
    Q, c, cp, i = best
    ob['cpl'] = cp
    Q, c = Problem(V, t, ob, cfgC, R0, free=('yaw', 'pitch', 'L'), fit_kappa=True).solve(
        Q, max_nfev=2 * cfg['max_nfev_c'], log=log, tag='puncture final')
    rest_p = crest[i]
    nv = organ['surf'][np.argmin(np.linalg.norm(organ['rest'][organ['surf']] - rest_p, axis=1))]
    info = dict(organ=organ['name'], organ_version=organ['version'], organ_path=organ['path'], face=int(cf[i]),
                bary=[float(x) for x in cb[i]], rest_m=[float(x) for x in rest_p], nearest_vertex=int(nv),
                nearest_vertex_dist_mm=round(float(np.linalg.norm(organ['rest'][nv] - rest_p) * 1000), 2),
                inside_frames=[list(map(int, g)) for g in _runs(inside)], candidates=tried, per_phase=per_phase,
                phase_separation_mm=sep, kappa=float(Q['kappa']))
    return Q, cp, info


def couple_contact(V, t, ob, cfg, R0, Q, organ, log):
    """Jaws on the organ: if the organ model reaches the jaws (median distance of the grasp point to the organ surface
    <= contact_gate), the grasp point is pulled onto the surface in every frame with a mask and the size freed."""
    n = V.n
    ts = scaled(t, Q.get('kappa', 1.0), Q.get('jls', 1.0))
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    d = RT[:, :, 2]
    tcp = Q['P'] + (Q['L'][:, None] - 0.35 * ts['jaw_len']) * d
    Xs = organ['X4'][:, organ['surf']]
    dist = np.sqrt(((Xs - tcp[:, None]) ** 2).sum(-1)).min(1) * 1000
    on = ob['vis'] & ~ob['trimmed']
    info = dict(organ=organ['name'], organ_version=organ['version'], tcp_to_surface_mm_before=summ(dist[on]))
    if np.median(dist[on]) > cfg['contact_gate'] * 1000:
        info['applied'] = False
        info['why'] = (f"grasp point {np.median(dist[on]):.1f} mm (median) from the {organ['name']} {organ['version']} surface "
                       f"> gate {cfg['contact_gate'] * 1000:.0f} mm: the model does not reach the jaws (no neck)")
        log(f"  contact: {info['why']}")
        return Q, None, info
    cp = dict(kind='contact', on=on, X4=organ['X4'], surf=organ['surf'], tcp_back=0.35 * ts['jaw_len'])
    ob['cpl'] = cp
    cfgC = dict(cfg, sig_kappa=cfg['sig_kappa_coupled'])
    free = tuple(v for v in VARS if not (v == 'roll' and np.ptp(Q['roll']) == 0))
    Q, c = Problem(V, t, ob, cfgC, R0, free=free, fit_kappa=True).solve(Q, max_nfev=2 * cfg['max_nfev_c'], log=log, tag='contact')
    ts = scaled(t, Q['kappa'], Q.get('jls', 1.0))
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    tcp = Q['P'] + (Q['L'][:, None] - 0.35 * ts['jaw_len']) * RT[:, :, 2]
    dist2 = np.sqrt(((Xs - tcp[:, None]) ** 2).sum(-1)).min(1) * 1000
    info.update(applied=True, tcp_to_surface_mm_after=summ(dist2[on]), kappa=float(Q['kappa']))
    return Q, cp, info


def fit_one(V, clip, inst, others, ip, cfg, log):
    t = make_tool(inst['type'], inst['diameter'])
    hidden_frames = hidden_info(clip, ip, inst['key'])
    meas, ob = observations(V, inst['name'], others, cfg, hidden_frames, inst.get('visible_frames'))
    attach_scene(ob, cfg, V, inst['name'], opening=bool(hidden_frames))
    log(f"[{inst['name']}] type {t['type']} (agent: {inst['agent_type']}), shaft {inst['diameter'] * 1000:.0f} mm "
        f"({inst['diameter_src']}); mask frames {ob['vis'].sum()} / {V.n}; end types "
        f"{ {e: int((ob['end_type'] == e).sum()) for e in ('free', 'blunt', 'cut', 'other', 'inside_organ')} }")
    n = V.n
    E, U = init_lines(V, meas, t)
    okl = np.isfinite(E[:, 0])
    if okl.sum() < 3:
        raise RuntimeError(f"{inst['name']}: fewer than 3 frames with a width line")
    Um = U[okl].mean(0)
    Um /= np.linalg.norm(Um)
    R0 = port_frame(Um, V.R[0][0])
    starts = []
    for s in cfg['port_starts']:
        if s is None:
            Pc = common_point(E, U)
            if np.isfinite(Pc).all() and 0.04 < np.linalg.norm(Pc - np.nanmean(E, 0)) < 0.4 and ((np.nanmean(E, 0) - Pc) @ Um) > 0:
                starts.append(('lines', Pc))
        else:
            starts.append((f'{s * 100:.0f}cm', np.nanmedian(E[okl] - s * U[okl], 0)))
    best = None
    hl = cfg['hidden_len'] if ob['hid_prior'].any() else 0.0
    cfgA = dict(cfg, w_depth=0.0)          # stage A: silhouettes only (no depth map), nominal size
    for tag, P0 in starts:
        Q0, _, _ = init_pose(V, meas, ob, t, P0, R0, hidden_len=hl)
        Q0['kappa'], Q0['jls'] = 1.0, 1.0
        prob = Problem(V, t, ob, cfgA, R0, free=('yaw', 'pitch', 'L'), fit_kappa=False, fit_jls=False)
        Q, cost = prob.solve(Q0, max_nfev=cfg['max_nfev_a'], log=log, tag=f'A {tag}')
        if best is None or cost < best[1]:
            best = (Q, cost, tag)
    Q, costA, tagA = best
    log(f'  stage A best start: {tagA}')
    # organ far wall cap and hidden-length prior (tip inside an organ)
    organ_used = None
    if ob['hid_prior'].any():
        hmax, organ_used = organ_far_wall(V, clip, Q, R0, ob, log)
        ob['hmax'] = hmax
        ob['h0'] = np.minimum(cfg['hidden_len'], np.where(np.isfinite(hmax), np.maximum(hmax, 0.0), np.inf))
        Q, _ = Problem(V, t, ob, cfgA, R0, free=('yaw', 'pitch', 'L'), fit_kappa=False, fit_jls=False).solve(Q, max_nfev=cfg['max_nfev_a'], log=log, tag='A+hidden')
    grid = None
    roll_note = ''
    freeC = VARS
    if t['jawed']:
        roll, jaw, grid = roll_jaw_grid(V, t, ob, Q, R0, cfg)
        Q['roll'], Q['jaw'] = roll, jaw
        log(f"  stage B roll/jaw grid: jaw observed in {grid['jaw_observed'].sum()} frames, roll observed in "
            f"{grid['roll_observed'].sum()}; jaw median {np.degrees(np.median(jaw)):.0f} deg")
        # roll: held constant (and not refitted) when it is seen in too few frames, else interpolated between the
        # frames where it is seen and refitted with the rest
        Q, roll_note = hold_unobserved_roll(Q, grid, ob, t)
        log(f'  {roll_note}')
        freeC = tuple(v for v in VARS if not (v == 'roll' and 'constant' in roll_note))
    Q, costC = Problem(V, t, ob, cfg, R0, free=freeC, fit_kappa=False).solve(Q, max_nfev=cfg['max_nfev_c'], log=log, tag='C all')
    if t['jawed'] and abs(Q['jls'] - 1) > 0.1:              # jaw length changed: redo the roll / jaw states and refit
        roll, jaw, grid = roll_jaw_grid(V, t, ob, Q, R0, cfg)
        Q['roll'], Q['jaw'] = roll, jaw
        Q, roll_note = hold_unobserved_roll(Q, grid, ob, t)
        freeC = tuple(v for v in VARS if not (v == 'roll' and 'constant' in roll_note))
        Q, costC = Problem(V, t, ob, cfg, R0, free=freeC, fit_kappa=False).solve(Q, max_nfev=cfg['max_nfev_c'], log=log, tag='C2 all')
    # scale from free space: the visible shaft must lie in front of the tissue beside it. If, at the nominal size, too
    # many samples put it behind that tissue, the instrument is scaled (with its port, about the scope) to the largest
    # factor <= 1 that leaves at most kappa_free_q of the samples behind, and refitted at that size.
    kfree, kinfo = kappa_from_free_space(V, t, ob, cfg, R0, Q)
    log(f"  free space at nominal size: {kinfo['frac_behind']:.2f} of {kinfo['n']} samples behind the tissue beside the "
        f"shaft (> {cfg['free_tol'] * 1000:.0f} mm); implied scale {kfree:.3f}")
    if kfree < cfg['kappa_apply_below']:
        c = V.pos[ob['vis']].mean(0)
        RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
        Tk = Q['P'] + Q['L'][:, None] * RT[:, :, 2]
        Q['P'] = c + kfree * (Q['P'] - c)
        dn = c + kfree * (Tk - c) - Q['P']
        Q['L'] = np.linalg.norm(dn, axis=1)
        Q['yaw'], Q['pitch'] = angles_of(R0, dn / Q['L'][:, None])
        Q['kappa'] = kfree
        Q, costC = Problem(V, t, ob, cfg, R0, free=freeC, fit_kappa=False).solve(Q, max_nfev=cfg['max_nfev_c'], log=log, tag=f'C scaled {kfree:.3f}')
        kfree2, kinfo2 = kappa_from_free_space(V, t, ob, cfg, R0, Q)
        kinfo['after'] = kinfo2
        log(f"  after scaling: {kinfo2['frac_behind']:.2f} of samples behind")
    kinfo['applied'] = bool(kfree < cfg['kappa_apply_below'])
    prof = port_profile(V, t, ob, cfg, R0, Q, log, freeC)
    # the port's least-determined direction converges slowly in the joint fit: line search along it, then refit
    best = min(prof['rows'], key=lambda r: r['dcost_per_frame'])
    if best['offset_mm'] != 0 and best['dcost_per_frame'] < -0.3:
        Q = shift_port(Q, R0, np.array(prof['direction']) * best['offset_mm'] / 1000)
        Q, costC = Problem(V, t, ob, cfg, R0, free=freeC, fit_kappa=False).solve(Q, max_nfev=cfg['max_nfev_c'], log=log,
                                                                               tag=f"C port {best['offset_mm']:+d} mm")
        prof2 = port_profile(V, t, ob, cfg, R0, Q, log, freeC)
        prof2['line_search_mm'] = best['offset_mm']
        prof = prof2
    couple = None
    if cfg['couple'] and cfg.get('_organ') is not None:
        if hidden_frames:
            Q, cp, couple = couple_puncture(V, clip, t, ob, cfg, R0, Q, cfg['_organ'], hidden_frames, log)
            couple['kind'] = 'puncture'
        elif t['jawed']:
            Q, cp, couple = couple_contact(V, t, ob, cfg, R0, Q, cfg['_organ'], log)
            couple['kind'] = 'contact'
    t_fit = dict(scaled(t, Q['kappa'], Q['jls']), radius_nominal=t['radius'], jaw_len_nominal=t['jaw_len'],
                 kappa=float(Q['kappa']), jls=float(Q['jls']))
    filled, long_gap = gap_flags(ob['vis'] | ob['trimmed'], int(round(cfg['short_gap_s'] * V.fps)))
    Qf = withdraw_long_gaps(V, t_fit, Q, R0, long_gap, ob['vis'])
    return dict(t=t_fit, Q=Qf, Q_fit=Q, R0=R0, ob=ob, meas=meas, filled=filled, long_gap=long_gap, grid=grid,
                hidden_frames=hidden_frames, organ_used=organ_used, start=tagA, profile=prof, roll_note=roll_note,
                kappa_info=kinfo, couple=couple)


def organ_fronts(V, clip, scale, pin=None, log=print):
    """Front depth (m, inf where none) of the clip's organ models (organ agent, newest finished version of each)
    along every frame's camera rays, min-pooled to the occupancy resolution: (n, h, w), and the versions used."""
    from r2s.tissue.gallbladder import raster_depth
    h, w = int(np.ceil(V.H / scale)), int(np.ceil(V.W / scale))
    front, used = np.full((V.n, h, w), np.inf, np.float32), {}
    for d in sorted((OUT / clip / 'organs').glob('*/')):
        vs = sorted(q.name for q in d.glob('v[0-9][0-9]') if (q / 'model.npz').exists())
        if not vs:
            continue
        ver = pin if pin in vs else vs[-1]
        z = np.load(d / ver / 'model.npz')
        X4, F = z['verts4d'].astype(np.float64), z['faces'].astype(np.int64)
        for k in range(V.n):
            zb = np.full((h * scale, w * scale), np.inf, np.float32)
            zb[:V.H, :V.W] = raster_depth(V, X4[k], F, k)
            front[k] = np.minimum(front[k], zb.reshape(h, scale, w, scale).min((1, 3)))
        used[d.name] = ver
    return front, used


def attach_scene(ob, cfg, V=None, name=None, opening=False):
    """Scene the shaft has to stay in front of: the background agent's surface and (r06) - at the pixels where this
    instrument is SEEN, so it occludes whatever lies behind - the front of the organ models, unless the instrument has
    a declared opening into an organ (its tip is then inside by the spec: puncture coupling)."""
    ob['bg'] = cfg.get('_bg')
    ob['bg_w'] = (~ob['trimmed']).astype(float)
    fr = cfg.get('_organ_front')
    if fr is None or V is None or opening:
        return
    sc = fr['scale']
    n, h, w = fr['depth'].shape
    m = np.zeros((n, h * sc, w * sc), bool)
    m[:, :V.H, :V.W] = V.mask(name)
    seen = m.reshape(n, h, sc, w, sc).any((2, 4))
    occ = ob['bg']['occ'] if ob['bg'] is not None else np.full(fr['depth'].shape, np.inf, np.float32)
    ob['bg'] = dict(occ=np.where(seen, np.minimum(occ, fr['depth']), occ).astype(np.float32), scale=sc)


def rebuild_coupling(T, cfg, ob):
    """Re-create the coupling state of a cached fit (reuse mode)."""
    c = T.get('couple')
    if not c or not c.get('kind') or cfg.get('_organ') is None:
        return
    org = cfg['_organ']
    if c['kind'] == 'puncture':
        inside = np.zeros(len(ob['vis']), bool)
        for a, b in c['inside_frames']:
            inside[a:b + 1] = True
        ob['cpl'] = puncture_setup(org, c['face'], c['bary'], inside, T['t'], cfg)
    elif c['kind'] == 'contact' and c.get('applied'):
        ob['cpl'] = dict(kind='contact', on=ob['vis'] & ~ob['trimmed'], X4=org['X4'], surf=org['surf'],
                         tcp_back=0.35 * T['t']['jaw_len'])


def kappa_from_free_space(V, t, ob, cfg, R0, Q):
    """Per free-space sample the scale factor s (about the scope) at which the shaft's front surface just reaches
    the tissue beside it (minus the tolerance); the clip's factor is the kappa_free_q quantile of s, capped at 1."""
    ts = scaled(t, Q.get('kappa', 1.0), Q.get('jls', 1.0))
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    d = RT[:, :, 2]
    O = Q['P'] + Q['L'][:, None] * d
    zl, _ = axis_depth(O, d, cams_of(V, np.arange(V.n)), ob['fpx'])
    w = ob['fw_'] > 0
    if w.sum() < 10:
        return 1.0, dict(n=int(w.sum()), frac_behind=float('nan'))
    front = (zl - ts['radius'])[w]
    s_i = (ob['fz'][w] + cfg['free_tol']) / np.maximum(front, 1e-4)
    k = float(np.clip(np.quantile(s_i, cfg['kappa_free_q']), cfg['kappa_free_min'], 1.0))
    return k * Q.get('kappa', 1.0), dict(n=int(w.sum()), frac_behind=float(np.mean(s_i < 1.0)),
                                         s_quantiles=[round(float(x), 3) for x in np.quantile(s_i, [0.05, 0.1, 0.25, 0.5])])


def hold_unobserved_roll(Q, grid, ob, t, min_frac=0.05):
    """Roll is only seen through open (or curved) jaws. Too few such frames: one constant roll for the clip (the
    circular mean over the observed frames, 0 if none); otherwise unobserved frames are interpolated between
    observed ones (held at the ends)."""
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Q.items()}
    if not t['jawed'] or grid is None:
        return Q, ''
    obs = grid['roll_observed']
    per = t['roll_period']
    n_need = max(5, int(min_frac * ob['vis'].sum()))
    if obs.sum() < n_need:
        if obs.any():
            a = np.angle(np.mean(np.exp(1j * Q['roll'][obs] * 2 * np.pi / per))) * per / (2 * np.pi)
        else:
            a = 0.0
        Q['roll'][:] = a
        return Q, f'roll observed in {obs.sum()} frames (< {n_need}): held constant at {np.degrees(a):.0f} deg'
    r = unwrap_roll(Q['roll'], per)
    idx = np.nonzero(obs)[0]
    Q['roll'] = np.interp(np.arange(len(r)), idx, r[idx])
    return Q, f'roll observed in {obs.sum()} frames: interpolated between them elsewhere'


def shift_port(Q, R0, delta):
    """Move the port by delta keeping every frame's tip point (new directions and insertions)."""
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Q.items()}
    RT = tip_frame(R0, Q['yaw'], Q['pitch'], Q['roll'])
    T = Q['P'] + Q['L'][:, None] * RT[:, :, 2]
    Q['P'] = Q['P'] + delta
    dn = T - Q['P']
    Q['L'] = np.linalg.norm(dn, axis=1)
    Q['yaw'], Q['pitch'] = angles_of(R0, dn / Q['L'][:, None])
    return Q


def port_profile(V, t, ob, cfg, R0, Q, log, free=VARS):
    """Port uncertainty by profiling: the port moved along its least-determined direction (the eigenvector of the
    depth-free axis-plane constraints with the smallest eigenvalue: mostly along the line of sight) by the offsets
    cfg['profile_mm'], per-frame poses refitted with the port fixed; mean cost increase per visible frame."""
    N, C = [], []
    for k in np.nonzero(ob['vis'])[0]:
        c, dd = ob['axis_c'][k], ob['axis_d'][k]
        q0, q1 = c - 50 * dd, c + 50 * dd
        X0 = V.unproject(q0[0], q0[1], 0.1, k) - V.pos[k]
        X1 = V.unproject(q1[0], q1[1], 0.1, k) - V.pos[k]
        nr = np.cross(X0, X1)
        N.append(nr / np.linalg.norm(nr)), C.append(V.pos[k])
    N = np.array(N)
    w, v = np.linalg.eigh(N.T @ N)
    e = v[:, 0]
    base = Problem(V, t, ob, cfg, R0, free=free, fix_port=True)       # fix_port also holds kappa and jls
    base.set_ref(Q)
    c0 = 0.5 * np.sum(base._rho(base.residuals(base.pack(Q['P'], Q), Q)))
    nv = max(1, int(ob['vis'].sum()))
    rows = []
    for off in (0,) + tuple(cfg['profile_mm']):
        Qs = shift_port(Q, R0, off / 1000 * e)
        pb = Problem(V, t, ob, cfg, R0, free=free, fix_port=True)
        pb.Tref = base.Tref
        _, cost = pb.solve(Qs, max_nfev=25, log=None)
        if off == 0:
            c0 = cost
        rows.append(dict(offset_mm=off, dcost_per_frame=round((cost - c0) / nv, 3)))
    log(f'  port profile along {np.round(e, 2)} (axis-plane eigenvalues {np.round(w, 2)}): ' +
        ', '.join(f"{r['offset_mm']:+d} mm: {r['dcost_per_frame']:+.2f}" for r in rows))
    return dict(direction=[round(float(x), 3) for x in e], plane_eigenvalues=[round(float(x), 3) for x in w], rows=rows,
                base_cost_per_frame=round(c0 / nv, 3))


def run(clip, ver='v01', overrides=None, log=print, reuse=False):
    from . import views2
    t0 = time.time()
    V = views2.load(clip)
    cfg = dict(DEFAULT, **CLIP_CFG.get(clip, {}), **(overrides or {}))
    sharp = blur_scores(V.frames)
    cfg['sharp'], cfg['sharp_ref'] = sharp, float(np.percentile(sharp, 60))
    out = OUT / clip / 'instruments' / ver
    (out / 'work').mkdir(parents=True, exist_ok=True)
    cfg['_bg'], cfg['background_version'] = None, None
    if cfg['w_bg'] > 0 and any((OUT / clip / 'background').glob('v[0-9][0-9]/model.npz')):
        from .background import Background
        bgo = Background(clip, cfg.get('background_pin'))
        cfg['_bg'] = dict(occ=bgo.occ.astype(np.float32), scale=bgo.occ_scale)
        cfg['background_version'] = bgo.dir.name
        log(f'[scene] background {bgo.dir.name}: shafts kept in front of it')
    cfg['_organ_front'], cfg['organ_front_versions'] = None, None
    if cfg['organ_solid'] and cfg['w_bg'] > 0 and any((OUT / clip / 'organs').glob('*/v[0-9][0-9]/model.npz')):
        sc = cfg['_bg']['scale'] if cfg['_bg'] is not None else 4
        front, used = organ_fronts(V, clip, sc, cfg['organ_version'], log)
        if cfg['_bg'] is not None and cfg['_bg']['occ'].shape != front.shape:
            log(f"[scene] organ fronts {front.shape} do not match the background occupancy {cfg['_bg']['occ'].shape}: skipped")
        else:
            cfg['_organ_front'], cfg['organ_front_versions'] = dict(depth=front, scale=sc), used
            log(f'[scene] organ models {used}: a shaft is kept in front of them where it is seen (no declared opening)')
    cfg['_organ'] = None
    if cfg['couple']:
        cfg['_organ'] = load_organ(clip, cfg['organ_version'])
        cfg['organ_used'] = cfg['_organ']['path']
        log(f"[scene] organ coupling with {cfg['_organ']['path']}")
    insts, ip = instrument_list(clip, V)
    import pickle
    cache = out / 'work' / 'fit.pkl'
    if reuse and cache.exists():
        tools = pickle.loads(cache.read_bytes())
        for nm, T in tools.items():                            # observations are cheap to rebuild, not cached
            others = [o['name'] for o in insts if o['name'] != nm]
            T['meas'], ob = observations(V, nm, others, cfg, T['hidden_frames'], T['inst'].get('visible_frames'))
            ob.update({k: T['ob_extra'][k] for k in T['ob_extra']})
            attach_scene(ob, cfg, V, nm, opening=bool(T['hidden_frames']))
            rebuild_coupling(T, cfg, ob)
            T['ob'] = ob
        log('[reuse] fitted poses from work/fit.pkl')
    else:
        tools = {}
        for inst in insts:
            others = [o['name'] for o in insts if o['name'] != inst['name']]
            tools[inst['name']] = fit_one(V, clip, inst, others, ip, cfg, log)
            tools[inst['name']]['inst'] = inst
        cache.write_bytes(pickle.dumps({nm: dict({k: v for k, v in T.items() if k not in ('meas', 'ob')},
                                                 ob_extra={k: T['ob'][k] for k in ('h0', 'hmax') if k in T['ob']})
                                        for nm, T in tools.items()}))
    log('[eval]')
    ev = evaluate(V, tools, cfg)
    q = write_outputs(V, clip, ver, out, tools, ev, cfg, insts, t0, log, reused=reuse and cache.exists())
    return q


def joints_of(T):
    Q = T['Q']
    return np.stack([Q['yaw'], Q['pitch'], Q['L'], Q['roll'], Q['jaw']], 1)


def write_outputs(V, clip, ver, out, tools, ev, cfg, insts, t0, log, reused=False):
    n = V.n
    bands = thirds(n)
    arrays = dict(names=np.array(list(tools)), fps=np.float64(V.fps), n=np.int64(n))
    qual = dict(clip=clip, version=ver, n_frames=n, fps=V.fps, instruments={},
                meaning=dict(
                    iou='IoU of the rendered model (both instruments z-buffered, inside the scope image) with the mask',
                    iou_vis='same, with the model beyond an occluded far end (tip inside / behind tissue, cut by the border) not counted',
                    tip_px='free far ends only: |far end of the model silhouette - far end of the mask| along the mask axis (px)',
                    hidden_px='occluded far ends: how far the model extends beyond the visible end (px, along the mask axis)',
                    hidden_mm='frames with the tip inside an organ: tip beyond the visible end along the shaft (mm)',
                    ang_deg='angle between the projected shaft and the mask axis (deg)',
                    offset_px='median distance of the mask centre line (shaft part) from the projected shaft line (px)',
                    plane_mm='port consistency, depth-free: distance of the port from the plane through the camera centre and the frame\'s observed 2D shaft axis (mm)',
                    line_mm='port consistency with depth: distance of the port from the frame\'s independent 3D shaft line (mask axis + widths of a 5 mm shaft) (mm)',
                    depth_res_mm='model shaft surface depth - metric depth map on shaft pixels (median, mm); circular: the map\'s scale comes from these shafts',
                    shaft_behind_tissue='fraction of shaft samples where the depth map puts the tissue just beside the shaft in front of it (by > 2 mm)'))
    for nm, T in tools.items():
        t, Q, ob, R = T['t'], T['Q'], T['ob'], ev[nm]
        RT = tip_frame(T['R0'], Q['yaw'], Q['pitch'], Q['roll'])
        d = RT[:, :, 2]
        tip = Q['P'][None] + Q['L'][:, None] * d
        vis = ob['vis'].copy()
        J = joints_of(T)
        grid = T['grid']
        jaw_obs = grid['jaw_observed'] if grid is not None else np.zeros(n, bool)
        roll_obs = grid['roll_observed'] if grid is not None else np.zeros(n, bool)
        arrays.update({f'{nm}__port': Q['P'], f'{nm}__R0': T['R0'], f'{nm}__tip': tip, f'{nm}__dir': d,
                       f'{nm}__roll': Q['roll'], f'{nm}__jaw': Q['jaw'], f'{nm}__visible': vis,
                       f'{nm}__filled': T['filled'], f'{nm}__long_gap': T['long_gap'], f'{nm}__trimmed': ob['trimmed'],
                       f'{nm}__tip_hidden': ob['hid'], f'{nm}__tip_inside_organ': ob['hid_prior'],
                       f'{nm}__end_type': ob['end_type'].astype(str), f'{nm}__jaw_observed': jaw_obs,
                       f'{nm}__roll_observed': roll_obs, f'{nm}__joints': J, f'{nm}__iou': R['iou'],
                       f'{nm}__params': json.dumps(tool_params(t)),
                       f'{nm}__tcp': tip - (0.35 * t['jaw_len'] if t['jawed'] else 0.0) * d,
                       f'{nm}__jaw_tip': jaw_tip_world(t, Q, T['R0'])})
        use = vis & ~ob['trimmed']
        sm = smooth_stats(V, Q, T['R0'], ~T['long_gap'] & ~ob['trimmed'])
        band = {}
        for bn, (a, b) in bands.items():
            sl = np.zeros(n, bool)
            sl[a:b] = True
            band[bn] = {key: summ(R[key][sl]).get('median') for key in ('iou', 'iou_vis', 'tip_px', 'hidden_px', 'ang_deg',
                                                                         'offset_px', 'plane_mm', 'line_mm')}
            band[bn]['tip_acc_m_s2_p90'] = smooth_stats(V, Q, T['R0'], sl & ~T['long_gap'] & ~ob['trimmed'])['tip_acc_m_s2'].get('p90')
        cam_P = (Q['P'] - V.pos[0]) @ V.R[0].T
        dist_tip = np.linalg.norm(tip - Q['P'], axis=1)
        # depth map vs model depth on shaft pixels, per instrument
        ratio = []
        for k in range(n):
            if T['meas'][k] is None or not ob['zw'][k].any():
                continue
            O = Q['P'] + Q['L'][k] * d[k]
            zl, _ = axis_depth(O[None], d[k][None], cams_of(V, np.array([k])), ob['zpx'][k][None])
            w = ob['zw'][k] > 0
            ratio.append(np.median(ob['zobs'][k][w] / (zl[0][w] - t['radius'])))
        qi = dict(type=t['type'], agent_type=T['inst']['agent_type'], diameter_nominal_mm=t['radius_nominal'] * 2000,
                  diameter_source=T['inst']['diameter_src'], kappa=round(t['kappa'], 4),
                  diameter_apparent_mm=round(t['radius'] * 2000, 3), kappa_free_space=T.get('kappa_info'),
                  jaw_len_scale=round(t.get('jls', 1.0), 3), model=tool_params(t),
                  frames=dict(mask=int(vis.sum()), filled_short_gap=int(T['filled'].sum()), long_gap=int(T['long_gap'].sum()),
                              trimmed=int(ob['trimmed'].sum()), tip_hidden=int(ob['hid'].sum()),
                              tip_inside_organ=int(ob['hid_prior'].sum()),
                              end_types={e: int((ob['end_type'] == e).sum()) for e in ('free', 'blunt', 'cut', 'other', 'inside_organ')},
                              jaw_observed=int(jaw_obs.sum()), roll_observed=int(roll_obs.sum())),
                  long_gaps=[list(map(int, g)) for g in _runs(T['long_gap'])],
                  ignored_mask_frames=[list(map(int, g)) for g in _runs(ob['drift'])],
                  short_gaps=[list(map(int, g)) for g in _runs(T['filled'])],
                  port=dict(world_mm=[round(float(v) * 1000, 1) for v in Q['P']],
                            camera0_mm=[round(float(v) * 1000, 1) for v in cam_P],
                            dist_to_tip_mm=summ(dist_tip[use] * 1000), profile=T['profile'], start=T['start']),
                  iou=summ(R['iou'][use]), iou_vis=summ(R['iou_vis'][use]), tip_px=summ(R['tip_px']),
                  tip_signed_px=summ(R['tip_signed_px']), hidden_px=summ(R['hidden_px']), hidden_mm=summ(R['hidden_mm']),
                  ang_deg=summ(R['ang_deg']), offset_px=summ(R['offset_px']), plane_mm=summ(R['plane_mm']),
                  line_mm=summ(R['line_mm']), depth_res_mm=summ(R['depth_res_mm']),
                  depthmap_over_model_depth=summ(ratio), shaft_behind_tissue=summ(R['shaft_behind_tissue']),
                  jaw_deg=summ(np.degrees(Q['jaw'][use])) if t['jawed'] else None,
                  smooth=sm, bands=band, organ_model_for_hidden_cap=T['organ_used'])
        # scene: background along the shaft, organ coupling
        live = ~T['long_gap'] & ~ob['trimmed']
        if cfg.get('_bg') is not None:
            br = bg_report(V, cfg['_bg'], t, Q, T['R0'], live)
            behind = np.isfinite(br['first_behind_mm'])
            qi['background'] = dict(version=cfg.get('background_version'), frames_checked=int(live.sum()),
                                    frames_axis_behind=int(behind.sum()), frames_axis_behind_list=[list(map(int, g)) for g in _runs(behind)],
                                    first_behind_mm_from_tip=summ(br['first_behind_mm']),
                                    axis_clearance_mm=summ(br['min_mm']),
                                    frames_surface_within_radius=int(np.nansum(br['min_mm'] < t['radius'] * 1000)))
            arrays[f'{nm}__bg_clearance_mm'] = br['min_mm']
        c = T.get('couple')
        if c:
            qi['organ_coupling'] = {k: v for k, v in c.items()}
            if c.get('kind') == 'puncture':
                cp = T['ob']['cpl']
                org = cfg['_organ']
                dg = puncture_diagnostics(V, t, org, cp, Q, T['R0'], ob)
                ins = cp['inside']
                win = ins & (np.arange(n) >= cfg['first_frame'])
                arrays.update({f'{nm}__tip_inside_organ': dg['inside_geom'], f'{nm}__tip_inside_agent': ins,
                               f'{nm}__puncture_world': cp['S'], f'{nm}__puncture_face': np.int64(cp['face']),
                               f'{nm}__puncture_bary': cp['bary'], f'{nm}__puncture_rest': np.array(c['rest_m']),
                               f'{nm}__puncture_vertex': np.int64(c['nearest_vertex']), f'{nm}__puncture_organ': c['organ_path'],
                               f'{nm}__line_to_puncture_mm': np.where(ins, dg['line_mm'], np.nan),
                               f'{nm}__tip_beyond_puncture_mm': np.where(ins, dg['h_mm'], np.nan),
                               f'{nm}__far_wall_mm': np.where(ins, dg['exit_mm'], np.nan),
                               f'{nm}__entry_wander_mm': np.where(ins, dg['wander_mm'], np.nan)})
                fail = win & ~dg['inside_geom']
                qi['organ_coupling'].update(
                    puncture_world_at_first_inside_frame_mm=[round(float(x) * 1000, 2) for x in cp['S'][np.nonzero(win)[0][0]]],
                    puncture_world_path_mm=dict(range=[round(float(x), 1) for x in np.ptp(cp['S'][ins], 0) * 1000]),
                    tip_inside=dict(frames=int(win.sum()), inside=int((win & dg['inside_geom']).sum()),
                                    outside_frames=[list(map(int, g)) for g in _runs(fail)],
                                    frames_before_first_frame=[int(k) for k in np.nonzero(ins & (np.arange(n) < cfg['first_frame']))[0]],
                                    inside_before_first_frame=int((ins & dg['inside_geom'] & (np.arange(n) < cfg['first_frame'])).sum())),
                    line_to_puncture_mm=summ(dg['line_mm'][win]),
                    entry_wander_mm=dict(rms=round(float(np.sqrt(np.nanmean(dg['wander_mm'][win] ** 2))), 2),
                                         max=round(float(np.nanmax(dg['wander_mm'][win])), 2),
                                         median=round(float(np.nanmedian(dg['wander_mm'][win])), 2)),
                    tip_beyond_puncture_mm=summ(dg['h_mm'][win]), far_wall_mm=summ(dg['exit_mm'][win]),
                    tip_past_far_wall_frames=[list(map(int, g)) for g in _runs(win & (dg['h_mm'] > dg['exit_mm']))])
            elif c.get('kind') == 'contact' and c.get('applied'):
                pass
        qual['instruments'][nm] = qi
        np.savez_compressed(out / 'work' / f'{nm}_fit.npz', **{f'Q_{k}': (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in T['Q_fit'].items()},
                            **{f'ev_{k}': v for k, v in R.items() if k != 'sil'},
                            grid_cost=(grid['cost'] if grid is not None else np.zeros(0)))
    # frames where every instrument in view misses its port at once: more likely a camera (pose interpolation) error
    # than instrument motion; such frames get a pose that compromises between the mask and the port
    flags = []
    for nm in tools:
        pm = ev[nm]['plane_mm']
        thr = max(3 * np.nanmedian(pm), 10.0)
        flags.append(np.where(np.isfinite(pm), pm > thr, np.nan))
    F = np.array(flags)
    seen = np.isfinite(F)
    cam_sus = (np.nansum(F, 0) == seen.sum(0)) & (seen.sum(0) >= 2)
    qual['camera_suspect_frames'] = [list(map(int, g)) for g in _runs(cam_sus)]
    qual['camera_suspect_rule'] = 'all instruments in view (>= 2) have plane_mm > max(3 x their median, 10 mm)'
    arrays['camera_suspect'] = cam_sus
    np.savez_compressed(out / 'model.npz', **arrays)
    try:
        qual['mujoco_check'] = mujoco_check(out / 'model.npz')
    except Exception as e:                                     # noqa
        qual['mujoco_check'] = dict(error=repr(e))
    qual['runtime_s'] = round(time.time() - t0, 1)
    qual['reused_fit'] = bool(reused)
    qual['cfg'] = {k: v for k, v in cfg.items() if k not in ('sharp',) and not k.startswith('_')}
    (out / 'quality.json').write_text(json.dumps(qual, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    write_notes(out, qual)
    sheet(V, tools, ev, out / 'sheet.jpg')
    # the worst frames (lowest IoU of the visible part) of every instrument, for the hard cases
    worst = []
    for nm in tools:
        v = np.where(np.isfinite(ev[nm]['iou_vis']), ev[nm]['iou_vis'], 2.0)
        worst += [int(k) for k in np.argsort(v)[:max(2, 8 // len(tools))] if v[k] < 2]
    sheet(V, tools, ev, out / 'sheet_worst.jpg', frames=sorted(set(worst)))
    views3d(V, tools, out / 'views3d.jpg')
    series(V, tools, ev, out / 'series.png')
    log(f'[done] {out} ({qual["runtime_s"]} s)')
    return qual


def _f(x, k='median', nd=2):
    if x is None or not isinstance(x, dict) or x.get(k) is None:
        return '-'
    return f'{x[k]:.{nd}f}'


def write_notes(out, q):
    L = [f"# {q['clip']} / instruments {q['version']}", '',
         f"`PYTHONPATH=. .venv/bin/python -m t2s.instruments {q['clip']} --ver {q['version']}` (t2s/instruments.py; "
         + (f"outputs re-evaluated from the cached fit (--reuse) in {q['runtime_s']} s; fit log ../{q['version']}_run.log"
            if q.get('reused_fit') else f"{q['runtime_s']} s") +
         f"). Files: model.npz (contract), quality.json (all numbers), sheet.jpg (16 frames: mask "
         f"outline green, model outline per instrument, yellow dot = model jaw tip / cap end, green cross = mask far end, line "
         f"towards the port; grey outline = not seen / long gap), sheet_worst.jpg (lowest-IoU frames), series.png (per-frame traces), views3d.jpg (ports, "
         f"shaft lines every 10th frame, tip paths, scope path; camera-0 coordinates).", '']
    for nm, v in q['instruments'].items():
        fr, p = v['frames'], v['port']
        L += [f"## {nm}: {v['type']} (instrument agent: {v['agent_type']})", '',
              f"- Model: shaft {v['diameter_nominal_mm']:.1f} mm nominal ({v['diameter_source']}); size used "
              f"{v['diameter_apparent_mm']:.2f} mm (kappa {v['kappa']:.3f}; kappa < 1 only when, at the nominal size, more than "
              f"{q['cfg']['kappa_free_q']:.0%} of the free-space samples put the shaft behind the tissue beside it: "
              f"{json.dumps(v.get('kappa_free_space'))}); jaw length {v['model']['jaw_len'] * 1000:.1f} mm (shape scale "
              f"{v.get('jaw_len_scale', 1.0)} fitted to the silhouettes), jaw mode {v['model']['jaw_mode']}, curve "
              f"{v['model']['curve']:.2f} rad.",
              f"- Frames: mask {fr['mask']} / {q['n_frames']}, short gaps filled {fr['filled_short_gap']} {v['short_gaps']}, "
              f"long gaps {fr['long_gap']} {v['long_gaps']} (withdrawn out of view), trimmed {fr['trimmed']}; masks ignored "
              f"outside the instrument agent's visible ranges (tracker drift) {v.get('ignored_mask_frames')}; far end "
              f"hidden in {fr['tip_hidden']} ({fr['end_types']}); jaw observed {fr['jaw_observed']}, roll observed "
              f"{fr['roll_observed']}.",
              f"- Port: camera-0 coordinates {p['camera0_mm']} mm (x right, y down, z forward), {_f(p['dist_to_tip_mm'], nd=0)} mm "
              f"from the tip (median); best start {p['start']}. Profile along the least-determined direction "
              f"{p['profile']['direction']} (axis-plane eigenvalues {p['profile']['plane_eigenvalues']}): cost change per "
              f"frame " + ', '.join(f"{r['offset_mm']:+d} mm {r['dcost_per_frame']:+.2f}" for r in p['profile']['rows']) + '.',
              '', '| frames | IoU | IoU visible part | tip px (free end) | hidden px | axis deg | centre-line px | port-plane mm | port-line mm | tip acc p90 m/s2 |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for b, x in v['bands'].items():
            L.append(f"| {b} | {x['iou'] if x['iou'] is not None else '-'} | {x['iou_vis'] if x['iou_vis'] is not None else '-'} | "
                     f"{x['tip_px'] if x['tip_px'] is not None else '-'} | {x['hidden_px'] if x['hidden_px'] is not None else '-'} | "
                     f"{x['ang_deg'] if x['ang_deg'] is not None else '-'} | {x['offset_px'] if x['offset_px'] is not None else '-'} | "
                     f"{x['plane_mm'] if x['plane_mm'] is not None else '-'} | {x['line_mm'] if x['line_mm'] is not None else '-'} | "
                     f"{x['tip_acc_m_s2_p90'] if x['tip_acc_m_s2_p90'] is not None else '-'} |")
        L.append(f"| all | {_f(v['iou'])} | {_f(v['iou_vis'])} | {_f(v['tip_px'])} | {_f(v['hidden_px'])} | {_f(v['ang_deg'])} | "
                 f"{_f(v['offset_px'])} | {_f(v['plane_mm'])} | {_f(v['line_mm'])} | {_f(v['smooth']['tip_acc_m_s2'], 'p90')} |")
        sm = v['smooth']
        L += ['', f"- Smoothness (not in long gaps): tip speed median {_f(sm['tip_speed_m_s'], nd=3)} m/s (max "
                  f"{_f(sm['tip_speed_m_s'], 'max', 3)}), acceleration p90 {_f(sm['tip_acc_m_s2'], 'p90')} m/s2 (max "
                  f"{_f(sm['tip_acc_m_s2'], 'max')}), shaft turn p90 {_f(sm['shaft_turn_deg_per_frame'], 'p90')} deg/frame.",
              f"- Depth consistency (circular for the scale, see the module note): model shaft surface - depth map on shaft "
              f"pixels {_f(v['depth_res_mm'], nd=1)} mm (median); depth map / model depth {_f(v['depthmap_over_model_depth'])}; "
              f"tissue beside the shaft in front of it (> 2 mm) in {_f(v['shaft_behind_tissue'], 'mean')} of samples.",
              (f"- Tip inside the organ ({fr['tip_inside_organ']} frames): {_f(v['hidden_mm'], nd=1)} mm beyond the visible end "
               f"(median; prior {q['cfg']['hidden_len'] * 1000:.0f} +- {q['cfg']['sig_hidden'] * 1000:.0f} mm, capped by the far wall "
               f"of {v['organ_model_for_hidden_cap']})." if fr['tip_inside_organ'] else ''),
              (f"- Jaw opening: median {_f(v['jaw_deg'], nd=0)} deg, p90 {_f(v['jaw_deg'], 'p90', 0)}, max {_f(v['jaw_deg'], 'max', 0)}."
               if v.get('jaw_deg') else ''), '']
    L += [f"## Clip", '', f"- Camera-suspect frames ({q['camera_suspect_rule']}): {q['camera_suspect_frames'] or 'none'}.",
          f"- MuJoCo check of the contract (mjcf_from_npz -> joints -> tip site): {json.dumps(q['mujoco_check'])}", '',
          '## Contract (model.npz)', '',
          '- `names`; per instrument `<name>__port` (3,) world m, `__R0` (3, 3) port frame (columns), `__tip` (n, 3) tip point, '
          '`__dir` (n, 3) unit shaft direction port -> tip, `__roll` (n,), `__jaw` (n,) total opening rad, `__joints` (n, 5) '
          'yaw / pitch / insertion / roll / jaw = the MJCF joint values, `__tcp` (n, 3) grasp point between the jaws (on the axis), '
          '`__jaw_tip` (n, 3) midpoint of the jaw tips (off the axis for curved jaws) / end of the rounded cap, '
          '`__visible` (n,) mask seen, `__filled` short gap bridged, `__long_gap` withdrawn out of view, `__trimmed`, '
          '`__tip_hidden` far end occluded, `__tip_inside_organ`, `__end_type`, `__jaw_observed`, `__roll_observed`, '
          '`__iou`, `__params` (json: type, radius (apparent), jaw_len, jaw radii, curve, jaw_mode, jaw_max, kappa); '
          '`camera_suspect` (n,), `fps`.',
          '- MJCF: `t2s.instruments.mjcf_from_npz(path)` -> per instrument fragments (body, actuator, tendon, equality, '
          'contact) with joints / position actuators `<name>_yaw`, `_pitch`, `_insertion`, `_roll`, `_jaw` and site '
          '`<name>_tip`; `standalone_xml(frags)` wraps them (needs `<compiler angle="radian"/>`).', '']
    (out / 'NOTES.md').write_text('\n'.join(x for x in L if x is not None))


def _runs(flag):
    out, k = [], 0
    while k < len(flag):
        if flag[k]:
            j = k
            while j < len(flag) and flag[j]:
                j += 1
            out.append((k, j - 1))
            k = j
        else:
            k += 1
    return out


# ================================================================ pictures
COLS = [(255, 80, 200), (60, 200, 255), (255, 200, 40), (120, 255, 120)]


def overlay(V, tools, ev, k, scale=1.0):
    im = V.frames[k].copy()
    for i, (nm, T) in enumerate(tools.items()):
        col = COLS[i % len(COLS)]
        m = T['meas'][k]
        if m is not None:
            cs, _ = cv2.findContours(m['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, (40, 255, 40), 1)
        q = {v: np.array([T['Q'][v][k]]) for v in VARS}
        pr = project_prims(T['t'], T['Q']['P'], T['R0'], q, cams_of(V, np.array([k])))
        full = render(pr, 0, V.H, V.W)
        cs, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dashed = T['long_gap'][k] or not T['ob']['vis'][k]
        cv2.drawContours(im, cs, -1, col if not dashed else (150, 150, 150), 1 if dashed else 2)
        tip = T['Q']['P'] + T['Q']['L'][k] * pr['RT'][0, :, 2]
        qt, _ = V.project(tip, k)
        qj, zj = V.project(jaw_tip_world(T['t'], {v: (T['Q'][v][k:k + 1] if v != 'P' else T['Q']['P']) for v in list(VARS) + ['P']}, T['R0'])[0], k)
        if zj > 0 and np.all(np.abs(qj) < 5000):
            cv2.circle(im, (int(qj[0]), int(qj[1])), 4, (255, 255, 0), -1)
        if m is not None:
            cv2.drawMarker(im, (int(m['tip'][0]), int(m['tip'][1])), (40, 255, 40), cv2.MARKER_CROSS, 10, 2)
        # direction to the port
        qP, zP = V.project(T['Q']['P'], k)
        if zP > 0:
            cv2.line(im, (int(qt[0]), int(qt[1])), (int(np.clip(qP[0], -3000, 3000)), int(np.clip(qP[1], -3000, 3000))), col, 1, cv2.LINE_AA)
        txt = f"{nm.replace('instrument_', '')[:12]} " + (f"IoU {ev[nm]['iou'][k]:.2f}" if np.isfinite(ev[nm]['iou'][k]) else '')
        if T['ob']['hid'][k]:
            txt += f" [{T['ob']['end_type'][k]}]"
        if T['long_gap'][k]:
            txt += ' [long gap]'
        cv2.putText(im, txt, (6, 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(im, txt, (6, 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(im, f'f{k}', (V.W - 55, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(im, f'f{k}', (V.W - 55, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im


def sheet(V, tools, ev, path, n_tiles=16, cols=4, frames=None):
    import imageio.v2 as iio
    ks = frames if frames is not None else np.linspace(0, V.n - 1, n_tiles).round().astype(int)
    tiles = [overlay(V, tools, ev, int(k), 0.5) for k in ks]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    iio.imwrite(path, np.concatenate(rows, 0), quality=85)


def _panel(w, h, xr, yr, title, invert_y=False):
    """A blank plot panel and its data -> pixel mapping (equal or free aspect)."""
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 18), (w - 6, h - 22), (180, 180, 180), 1)
    cv2.putText(img, title, (44, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def to_px(x, y):
        u = 40 + (np.asarray(x, float) - xr[0]) / (xr[1] - xr[0] + 1e-12) * (w - 46)
        v = (np.asarray(y, float) - yr[0]) / (yr[1] - yr[0] + 1e-12)
        v = 18 + (v if invert_y else 1 - v) * (h - 40)
        return np.stack([u, v], -1)
    for val, lab in ((yr[0], yr[0]), (yr[1], yr[1])):
        q = to_px(xr[0], val)
        cv2.putText(img, f'{lab:.3g}', (2, int(q[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 80), 1)
    for val in (xr[0], xr[1]):
        q = to_px(val, yr[0] if not invert_y else yr[1])
        cv2.putText(img, f'{val:.3g}', (int(q[0]) - 10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 80), 1)
    return img, to_px


def _poly(img, P, col, th=1):
    P = np.asarray(P, float)
    ok = np.isfinite(P).all(1)
    k = 0
    while k < len(P):
        if not ok[k]:
            k += 1
            continue
        j = k
        while j < len(P) and ok[j]:
            j += 1
        if j - k >= 2:
            cv2.polylines(img, [np.round(P[k:j]).astype(np.int32)], False, col, th, cv2.LINE_AA)
        elif j - k == 1:
            cv2.circle(img, tuple(np.round(P[k]).astype(int)), 1, col, -1)
        k = j


def views3d(V, tools, path, size=420):
    """Ports, shaft lines (every 10th frame) and tip paths in camera-0 coordinates (mm): front (x, y), top (x, z),
    side (z, y). Black: scope centre path."""
    R0c = V.R[0]
    to_c = lambda X: (np.asarray(X) - V.pos[0]) @ R0c.T * 1000
    allp = [to_c(V.pos)]
    data = {}
    for nm, T in tools.items():
        RT = tip_frame(T['R0'], T['Q']['yaw'], T['Q']['pitch'], T['Q']['roll'])
        tip = T['Q']['P'] + T['Q']['L'][:, None] * RT[:, :, 2]
        data[nm] = (to_c(T['Q']['P'][None])[0], to_c(tip), ~T['long_gap'])
        allp += [data[nm][0][None], data[nm][1]]
    A = np.concatenate(allp, 0)
    lo, hi = A.min(0) - 10, A.max(0) + 10
    panels = []
    for title, (i, j), inv in (('camera-0 front: x right, y down (mm)', (0, 1), True), ('top: x right, z forward', (0, 2), False),
                               ('side: z forward, y down', (2, 1), True)):
        span = max(hi[i] - lo[i], hi[j] - lo[j])
        xr = ((lo[i] + hi[i]) / 2 - span / 2, (lo[i] + hi[i]) / 2 + span / 2)
        yr = ((lo[j] + hi[j]) / 2 - span / 2, (lo[j] + hi[j]) / 2 + span / 2)
        img, tp = _panel(size, size, xr, yr, title, invert_y=inv)
        cp = to_c(V.pos)
        _poly(img, tp(cp[:, i], cp[:, j]), (0, 0, 0), 1)
        q = tp(0, 0)
        cv2.drawMarker(img, (int(q[0]), int(q[1])), (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 10, 2)
        for c, (nm, (P, tc, ok)) in enumerate(data.items()):
            col = COLS[c % len(COLS)][::-1]
            qP = tp(P[i], P[j])
            for k in range(0, V.n, 10):
                if ok[k]:
                    qt = tp(tc[k, i], tc[k, j])
                    cv2.line(img, (int(qP[0]), int(qP[1])), (int(qt[0]), int(qt[1])), tuple(int(v * 0.6 + 100) for v in col), 1, cv2.LINE_AA)
            _poly(img, np.where(ok[:, None], tp(tc[:, i], tc[:, j]), np.nan), col, 2)
            cv2.circle(img, (int(qP[0]), int(qP[1])), 6, col, -1)
            cv2.putText(img, nm.replace('instrument_', '') + ' port', (int(qP[0]) + 8, int(qP[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
        panels.append(img)
    cv2.imwrite(str(path), np.concatenate(panels, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def series(V, tools, ev, path, w=1200, h=110):
    """Per-frame traces (one colour per instrument). Grey band: long gap; thick bar at the bottom: tip hidden."""
    rows = [('iou', 'IoU', (0, 1)), ('tip_px', 'tip px (free end)', None), ('ang_deg', 'axis angle deg', None),
            ('plane_mm', 'port-plane mm', None), ('L', 'insertion mm', None), ('jaw', 'jaw deg', None), ('roll', 'roll deg', None)]
    out = []
    for key, title, yr in rows:
        ys = {}
        for nm, T in tools.items():
            if key in ('L', 'jaw', 'roll'):
                if key != 'L' and not T['t']['jawed']:
                    continue
                ys[nm] = T['Q'][key] * (1000 if key == 'L' else 180 / np.pi)
            else:
                ys[nm] = ev[nm][key]
        vals = np.concatenate([v[np.isfinite(v)] for v in ys.values()]) if ys else np.zeros(1)
        if yr is None:
            yr = (float(np.min(vals)) if len(vals) else 0.0, float(np.percentile(vals, 99)) if len(vals) else 1.0)
            if yr[1] - yr[0] < 1e-6:
                yr = (yr[0] - 1, yr[1] + 1)
        img, tp = _panel(w, h, (0, V.n - 1), yr, title)
        for c, (nm, T) in enumerate(tools.items()):
            col = COLS[c % len(COLS)][::-1]
            for a, b in _runs(T['long_gap']):
                p0, p1 = tp(a, yr[1]), tp(b, yr[0])
                ov = img.copy()
                cv2.rectangle(ov, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (225, 225, 225), -1)
                img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
            for a, b in _runs(T['ob']['hid']):
                p0, p1 = tp(a, yr[0]), tp(b, yr[0])
                cv2.line(img, (int(p0[0]), int(p0[1]) - 2 - 3 * c), (int(p1[0]), int(p1[1]) - 2 - 3 * c), col, 2)
            if nm in ys:
                _poly(img, tp(np.arange(V.n), np.clip(ys[nm], yr[0], yr[1])), col, 1)
        out.append(img)
    leg = np.full((22, w, 3), 255, np.uint8)
    for c, nm in enumerate(tools):
        cv2.putText(leg, nm.replace('instrument_', ''), (50 + 220 * c, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLS[c % len(COLS)][::-1], 1)
    cv2.imwrite(str(path), np.concatenate([leg] + out, 0))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('--ver', default='v01')
    ap.add_argument('--reuse', action='store_true', help='re-evaluate the cached fit (work/fit.pkl)')
    ap.add_argument('--couple', action='store_true', help='couple to the organ model (puncture site / jaws on the organ)')
    ap.add_argument('--organ', default=None, help='organ model version (default: newest)')
    ap.add_argument('--first-frame', type=int, default=0, help='first frame the inside-the-organ requirement is reported for')
    ap.add_argument('--set', nargs='*', default=[], help='cfg overrides key=value (python literals)')
    a = ap.parse_args()
    import ast
    ov = dict(couple=a.couple, organ_version=a.organ, first_frame=a.first_frame)
    for kv in a.set:
        k, v = kv.split('=', 1)
        ov[k] = ast.literal_eval(v)
    run(a.clip, a.ver, reuse=a.reuse, overrides=ov)
