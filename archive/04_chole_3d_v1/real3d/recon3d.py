"""3D real-to-sim of the cholecystectomy clip from measured geometry (replaces the hand-made depth prior).

Inputs:  Depth Anything V2 Small relative depth per frame (real3d/disp_rel.npy), SAM 2.1 masks per frame
         (real3d/sam_masks.npz: gallbladder, grasper, probe, strand, liver, right organs), metric calibration from
         the 5 mm instrument shafts (real3d/metric.npz).
Output:  real3d/chole3d/ in the layout chole_sim.py / chole_metrics.py read (geometry.npz, textures, backdrop.obj,
         actions.npy, build_meta.json, real_tracks.npy).

  gallbladder  front surface = metric depth over its SAM mask; back = front + thickness from the silhouette
               (circular cross-section: t = 2 sqrt(R^2 - (R - d)^2), d = distance to the outline, R = local half-width);
               pinned only where it lies against the liver and along the image border
  strand       ribbon over its SAM mask at its own measured depth; far ends pinned, gallbladder end tied
  backdrop     metric depth with the gallbladder, strand and instruments filled in (the bed behind them)
  instruments  tip and axis from the SAM masks, tip depth from the calibrated depth of the instrument's own pixels,
               ports fitted to the observed axes
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'realistic'))
sys.path.insert(0, str(ROOT / 'real3d'))
import chole_recon as C  # noqa: E402
import segment as SG  # noqa: E402
import metric_depth as MD  # noqa: E402

OUT = ROOT / 'real3d' / 'chole3d'
W, H, FPS = C.W, C.H, C.FPS
GB, GR, PR, ST, LV, RO = range(6)


def instrument_track(masks, Z, idx, toward):
    """Per frame: visible distal end (px), image axis (unit, pointing to the end) and the end's metric depth."""
    tips, dirs, depth = [], [], []
    for k in range(len(masks)):
        m = masks[k, idx]
        ys, xs = np.nonzero(m)
        if len(xs) < 200:
            tips.append([np.nan] * 2), dirs.append([np.nan] * 2), depth.append(np.nan)
            continue
        P = np.stack([xs, ys], 1).astype(float)
        c = P.mean(0)
        d = np.linalg.svd(P - c)[2][0]
        d = d if d @ toward > 0 else -d
        s = (P - c) @ d
        tip = P[s > np.percentile(s, 98.5)].mean(0)
        near = P[(s > np.percentile(s, 85)) & (s < np.percentile(s, 97))]
        tips.append(tip), dirs.append(d), depth.append(float(np.median(Z[k][near[:, 1].astype(int), near[:, 0].astype(int)])))
    return C.fill_smooth(np.array(tips)), C.fill_smooth(np.array(dirs)), C.fill_smooth(np.array(depth)[:, None], 9)[:, 0]


def silhouette_thickness(mask, z):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    R = maximum_filter(dt, size=41)
    t_px = 2 * np.sqrt(np.clip(R ** 2 - (R - dt) ** 2, 0, None))
    return np.clip(t_px * z / C.F, 0.002, 0.03)


def amodal_gallbladder(masks, cams, z0, n_early=60):
    """Gallbladder extent at frame 0 including what the probe / strands hide then: pixels under them (frame 0) that
    show gallbladder in the early frames (warped back by the scope rotation+zoom; the grasper is excluded: the neck
    is later pulled up into where its shaft was). Depth there: extrapolated from the visible gallbladder, kept behind
    whatever hid it."""
    gb0 = masks[0, GB]
    cov0 = cv2.dilate((masks[0, PR] | masks[0, ST]).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    vv, uu = np.mgrid[0:H, 0:W]
    X0 = C.unproject(uu.ravel(), vv.ravel(), 0.1)
    seen = np.zeros(H * W, bool)
    for k in range(2, n_early, 2):
        uk = C.project(X0, cams['R'][k], cams['f'][k])[0]
        x, y = uk[:, 0].round().astype(int), uk[:, 1].round().astype(int)
        ins = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        seen[ins] |= masks[k, GB][y[ins], x[ins]]
    am = gb0 | (seen.reshape(H, W) & cov0)
    am = cv2.morphologyEx(am.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnt = cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    am = cv2.drawContours(np.zeros_like(am), [max(cnt, key=cv2.contourArea)], -1, 1, -1).astype(bool)   # filled, one piece
    add = am & ~gb0
    z = z0.copy()
    w = gb0.astype(np.float32)
    zf = z0 * w
    for sig in (8, 16, 32, 64):                                    # normalised convolution: gallbladder depth only
        num, den = cv2.GaussianBlur(zf, (0, 0), sig), cv2.GaussianBlur(w, (0, 0), sig)
        fill = add & (den > 1e-3) & (z == z0)
        z[fill] = num[fill] / den[fill]
    hid_by = np.where(masks[0, PR], 0.003, np.where(masks[0, ST], 0.002, 0.0))
    z[add] = np.maximum(z[add], z0[add] + hid_by[add])             # behind the probe / strand that covered it
    return am, add, z


def bed_motion(frames, masks, cams, px, z_back, neck_px=60):
    """The gallbladder as a whole shifts by 4-6 mm in the clip, its out-of-view part included: forces from out of view
    (bed, fundus retraction) that the scene does not contain. Taken from the video like the scope motion: frame-to-
    frame dense flow (DIS) of gallbladder pixels, re-sampled every frame, away from the grasped neck (the grasper's
    doing is left to the physics), occluders and specular highlights; scope rotation / zoom removed; the median step
    accumulated into one translation per frame (in the frame-0 image), applied to where the tissue is anchored.
    Returns (n_frames, N, 3) anchor displacements and the (n_frames, 2) pixel translation."""
    n = len(frames)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ker = np.ones((2 * neck_px + 1, 2 * neck_px + 1), np.uint8)
    T = np.zeros((n, 2))
    for k in range(1, n):
        sel = masks[k - 1, GB] & ~cv2.dilate(masks[k - 1, GR].astype(np.uint8), ker).astype(bool)
        sel &= ~cv2.dilate((masks[k - 1, PR] | masks[k - 1, ST] | masks[k, PR] | masks[k, ST]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        sel &= cv2.cvtColor(frames[k - 1], cv2.COLOR_RGB2HSV)[..., 2] < 220
        sel = cv2.erode(sel.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ys, xs = np.nonzero(sel)
        if len(xs) < 50:
            T[k] = T[k - 1]
            continue
        fl = dis.calc(gray[k - 1], gray[k], None)
        a = np.stack([xs, ys], 1).astype(float)
        b = a + fl[ys, xs]
        to0 = lambda q, kk: C.project(C.unproject_k(q[:, 0], q[:, 1], 0.1, cams['R'][kk], cams['f'][kk]).T if False else
                                      np.stack([C.unproject_k(u, v, 0.1, cams['R'][kk], cams['f'][kk]) for u, v in q]), cams['R'][0], cams['f'][0])[0]
        idx = np.random.default_rng(k).choice(len(a), min(400, len(a)), replace=False)
        T[k] = T[k - 1] + np.median(to0(b[idx], k) - to0(a[idx], k - 1), axis=0)
    T = C.fill_smooth(T, 9)
    out = np.zeros((n, len(px), 3))
    for i in range(len(px)):
        X0 = C.unproject_k(px[i, 0], px[i, 1], z_back[i], cams['R'][0], cams['f'][0])
        out[:, i] = np.stack([C.unproject_k(px[i, 0] + T[k, 0], px[i, 1] + T[k, 1], z_back[i], cams['R'][0], cams['f'][0]) for k in range(n)]) - X0
    return out, T


def ribbon_from_mask(mask, z, anchors_far, tie_near, step=7):
    """Textured strand ribbon at its own depth: vertices near the far ends pinned, near the gallbladder tied."""
    R = C.ribbon_sheet(mask, z, step=step, lift=0.0)
    P = R['px']
    pinned = np.zeros(len(P), bool)
    for a in anchors_far:
        pinned |= np.linalg.norm(P - a, axis=1) < 14
    dgb = cv2.distanceTransform((~tie_near).astype(np.uint8), cv2.DIST_L2, 5)        # tie_near: gallbladder mask
    dv = dgb[np.clip(P[:, 1].astype(int), 0, H - 1), np.clip(P[:, 0].astype(int), 0, W - 1)]
    tied = (dv < 12) & ~pinned
    if not tied.any():
        tied[np.argsort(dv)[:5]] = True
    R.update(pinned=pinned, tied=tied & ~pinned)
    return R


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    C.OUT = OUT
    frames = [f for f in imageio.get_reader(ROOT / 'runs_real/chole_sweep/task/video.mp4')]
    for f in frames:                     # the crop keeps dark letterbox rows / columns at the edges: the padded
        f[0] = f[1]                      # textures would mirror them into dark lines across the backdrop
        f[-2:] = f[-3]
        f[:, -2:] = f[:, -3:-2]
    n = len(frames)
    masks, names = SG.load()
    disp = np.load(ROOT / 'real3d/disp_rel.npy')
    mt = np.load(ROOT / 'real3d/metric.npz')
    a, b = float(mt['a']), mt['b']
    Z = np.stack([MD.depth(disp[k], a, b[k]) for k in range(n)])
    # --- camera: rotation + zoom fitted to tracks on liver / right organs (SAM), instruments excluded
    instr = np.zeros((H, W), bool)
    for k in range(n):
        instr |= masks[k, GR] | masks[k, PR]
    instr = cv2.dilate(instr.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    P_lk = C.lk_tracks(frames, instr)
    np.save(OUT / 'real_tracks.npy', P_lk)
    cams = C.camera_track(P_lk, masks[0, LV] | masks[0, RO])
    # --- instruments in 3D
    gt, gd, gz = instrument_track(masks, Z, GR, np.array([0.5, 0.85]))
    pt, pd, pz = instrument_track(masks, Z, PR, np.array([-0.85, 0.5]))
    inst = {}
    # grasper: its jaws are buried in the neck, so the measurable thing is the neck apex it holds: the gallbladder
    # pixels nearest the grasper's visible end, at their own measured depth (+4 mm into the tissue)
    apex_px, apex_z = [], []
    for k in range(n):
        ys, xs = np.nonzero(masks[k, GB])
        dd_ = np.hypot(xs - gt[k, 0], ys - gt[k, 1])
        sel = dd_ < np.percentile(dd_, 3) + 2 if len(dd_) else []
        if len(dd_) == 0:
            apex_px.append([np.nan] * 2), apex_z.append(np.nan)
            continue
        apex_px.append([np.median(xs[sel]), np.median(ys[sel])])
        apex_z.append(float(np.median(Z[k][ys[sel], xs[sel]])) + 0.004)
    apex_px = C.fill_smooth(np.array(apex_px), 7)
    apex_z = C.fill_smooth(np.array(apex_z)[:, None], 9)[:, 0]
    for name, tip, d2, zt, init_px, tcp_off in (('grasper_left', apex_px, gd, apex_z, (60, -250), 0.0),
                                                ('probe_right', pt, pd, pz, (900, -200), -0.003)):
        tips = np.array([C.unproject_k(tip[k, 0], tip[k, 1], zt[k], cams['R'][k], cams['f'][k]) for k in range(n)])
        # the probe keeps a constant apparent width along its visible length: its shaft runs roughly parallel to the
        # image plane, so its port lies at about the tips' depth (the grasper widens towards the scope: port near it)
        zr = (float(np.median(zt)) - 0.02, float(np.median(zt)) + 0.02) if name == 'probe_right' else (-0.06, 0.03)
        P, err = C.fit_rcm_ports(tips, d2, init_px, cams, fwd_range=zr)
        dd = (tips - P) / np.linalg.norm(tips - P, axis=1, keepdims=True)
        inst[name] = dict(rcm=P, rcm_resid_mm=err, tips=tips, T=tips + tcp_off * dd, tip_depth=zt)
    # --- geometry of frame 0
    z0 = cv2.bilateralFilter(Z[0].astype(np.float32), 9, 0.004, 5)
    strand = masks[0, ST]
    gb, gb_added, z0 = amodal_gallbladder(masks, cams, z0)
    liver = masks[0, LV]
    # instruments dilated wide: the measured depth bleeds well past their mask edge
    hole = (gb | strand | cv2.dilate((masks[0, GR] | masks[0, PR]).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool))
    scale = 1000.0
    zb = cv2.inpaint(np.clip(z0 * scale, 0, 65535).astype(np.uint16), cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8)), 15,
                     cv2.INPAINT_TELEA).astype(np.float32) / scale
    zb = cv2.GaussianBlur(zb, (0, 0), 6)
    thick = silhouette_thickness(gb, z0)
    zb = np.maximum(zb, np.where(gb, z0 + thick, zb))            # the bed lies behind the gallbladder's back
    sheet = C.flex_sheet(z0, gb, np.zeros_like(gb), step=10)
    px = sheet['px'].astype(int)
    sheet['thick'] = thick[np.clip(px[:, 1], 0, H - 1), np.clip(px[:, 0], 0, W - 1)]
    against_liver = cv2.dilate(liver.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    rim_all = sheet['rim'] | sheet['bottom']
    on_border = (px[:, 0] <= 3) | (px[:, 0] >= W - 4) | (px[:, 1] >= H - 4)
    # only where the image border cuts the organ: the attachment to the liver bed is on the hidden back side
    # (the back vertices' springs), not along the outline that merely touches the liver in the image
    pinned = rim_all & on_border
    sheet['rim'], sheet['bottom'] = pinned, np.zeros_like(pinned)
    zv = z0[np.clip(px[:, 1], 0, H - 1), np.clip(px[:, 0], 0, W - 1)] + sheet['thick']          # anchors sit at the back
    bed_disp, bed_T = bed_motion(frames, masks, cams, sheet['px'], zv)
    sheet['in_gb'] = np.ones(len(px), bool)
    # strand ribbon: far ends = the strand pixels farthest from the gallbladder along each branch
    ys, xs = np.nonzero(strand)
    dgb = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 5)[ys, xs]
    far = np.stack([xs, ys], 1)[dgb > np.percentile(dgb, 92)]
    from scipy.cluster.vq import kmeans2
    anchors = kmeans2(far.astype(float), 2, minit='++', seed=0)[0] if len(far) > 10 else far
    tie = np.stack([xs, ys], 1)[np.argmin(dgb)].astype(float)
    ribbon = ribbon_from_mask(strand, z0, [tuple(x) for x in anchors], gb)
    # --- textures: clean plate with SAM instrument masks; tissue under the strand inpainted
    tr = {'probe': {'masks': [masks[k, PR] for k in range(n)]}, 'grasper': {'masks': [masks[k, GR] for k in range(n)]}}
    f0 = frames[0].copy()
    f0[-4:] = f0[-8:-4][::-1]
    m_in = cv2.dilate((masks[0, GR] | masks[0, PR]).astype(np.uint8), np.ones((13, 13), np.uint8))
    # earliest uncovered views only (later the probe lifts the dark strands into the gap it left); a wide margin and a
    # dark-leaning percentile keep the shaft's glare out
    plate, never = C.clean_plate([f0] + frames[1:], tr, cams, m_in.astype(bool), cover_px=15, first=5, pct=30)
    alpha = cv2.GaussianBlur(cv2.dilate(m_in, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
    f0 = (alpha * plate + (1 - alpha) * f0).astype(np.uint8)
    if never.any():
        f0 = cv2.inpaint(f0, cv2.dilate(never, np.ones((3, 3), np.uint8)), 7, cv2.INPAINT_TELEA)
    ribbon_tex = f0.copy()
    under = cv2.dilate((strand & gb).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    tr_s = {'probe': {'masks': [masks[k, PR] | masks[k, ST] for k in range(n)]}, 'grasper': tr['grasper']}
    plate_s, never_s = C.clean_plate([f0] + frames[1:], tr_s, cams, under, cover_px=7, first=5, pct=50)
    f0s = f0.copy()
    f0s[under & ~never_s.astype(bool)] = plate_s[under & ~never_s.astype(bool)]
    rest = cv2.dilate(strand.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~(under & ~never_s.astype(bool))
    tissue_tex = cv2.inpaint(f0s, rest.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    back = cv2.inpaint(tissue_tex, cv2.dilate((gb | strand).astype(np.uint8), np.ones((5, 5), np.uint8)), 15, cv2.INPAINT_TELEA)
    pad = lambda img: cv2.copyMakeBorder(img, C.PAD, C.PAD, C.PAD, C.PAD, cv2.BORDER_REFLECT)
    imageio.imwrite(OUT / 'tex_tissue.png', pad(tissue_tex))
    imageio.imwrite(OUT / 'tex_ribbon.png', pad(ribbon_tex))
    bp = pad(tissue_tex)
    bp[C.PAD:C.PAD + H, C.PAD:C.PAD + W] = back
    imageio.imwrite(OUT / 'tex_back.png', bp)
    C.backdrop_obj(OUT / 'backdrop.obj', zb)
    # --- actions (20 Hz), rate-limited
    t_src, t_ctl = np.arange(n) / FPS, np.arange(0, (n - 1) / FPS + 1e-9, 0.05)
    acts = []
    for name in ('grasper_left', 'probe_right'):
        I = inst[name]
        I['heading'] = C.heading_for(I['rcm'], I['T'])
        Ti = np.stack([np.interp(t_ctl, t_src, I['T'][:, k]) for k in range(3)], 1)
        acts.append(C.joint_targets(I['rcm'], Ti, I['heading'], 0.0))
    acts = np.concatenate(acts, 1)
    np.save(OUT / 'actions.npy', acts)
    np.savez(OUT / 'geometry.npz', bed_disp=bed_disp.astype(np.float32), bed_T=bed_T, z=z0, zb=zb, gb=gb, fold=np.zeros_like(gb), band=strand, band_ext=strand,
             sheet_px=sheet['px'], sheet_X=sheet['X'], sheet_tri=sheet['tri'], sheet_thick=sheet['thick'],
             sheet_rim=sheet['rim'], sheet_bottom=sheet['bottom'], sheet_in_gb=sheet['in_gb'],
             ribbon_px=ribbon['px'], ribbon_X=ribbon['X'], ribbon_tri=ribbon['tri'], ribbon_pinned=ribbon['pinned'],
             ribbon_tied=ribbon['tied'], cam_R=cams['R'], cam_f=cams['f'], cam_fit_err=cams['fit_err_px'],
             probe_raw=pt, grasper_raw=gt, probe_tip_px=pt, grasper_tip_px=gt,
             probe_T=inst['probe_right']['T'], grasper_T=inst['grasper_left']['T'])
    meta = dict(source='Depth Anything V2 Small + SAM 2.1 tiny', metric_slope=a, fovy=C.FOVY,
                camera_motion=dict(max_rot_deg=round(float(np.degrees(np.linalg.norm(cams['params'][:, :3], axis=1)).max()), 2),
                                   fit_err_px_median=round(float(np.median(cams['fit_err_px'][1:])), 2)),
                gallbladder=dict(depth_cm=[round(float(np.percentile(z0[gb], 5)) * 100, 1), round(float(np.percentile(z0[gb], 95)) * 100, 1)],
                                 max_thickness_cm=round(float(thick[gb].max()) * 100, 1)),
                gb_amodal_added_px=int(gb_added.sum()), bed_max_disp_mm=round(float(np.linalg.norm(bed_disp, axis=2).max()) * 1000, 1), n_flex_vertices=len(sheet['X']), n_flex_triangles=len(sheet['tri']), pinned=int(pinned.sum()),
                ribbon=dict(vertices=len(ribbon['X']), pinned=int(ribbon['pinned'].sum()), tied=int(ribbon['tied'].sum())),
                instruments={k: dict(rcm=v['rcm'].round(4).tolist(), heading=round(v['heading'], 4), axis_fit_err_deg=round(v['rcm_resid_mm'], 2),
                                     tip_depth_cm=[round(float(v['tip_depth'].min()) * 100, 1), round(float(v['tip_depth'].max()) * 100, 1)])
                             for k, v in inst.items()},
                actions_shape=list(acts.shape))
    (OUT / 'build_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    build()
