"""Compare camera + depth solutions of a clip by how well keyframes warp onto each other.

Each solution gives, per keyframe k, a pose in keyframe-0 camera coordinates (x_k = R_k X + t_k), a focal length and
a metric depth map. Keyframe i is unprojected with its own depth, moved into keyframe j and compared with the real
frame j on static anatomy (instruments, organ and strands masked in both): image NCC, higher is better. Short range
(10 and 30 frames) shows local consistency; frame 0 against every later keyframe shows long-range drift.

  static       no camera motion (reference)
  rotation     the single-view track: rotation + zoom about a fixed tip, single-view depth calibration
  vggt_raw     VGGT alone: its poses, focal length and depth, metric scale from instrument shafts
  ba, sift, vggt_tracks, vggt   keyframe bundle adjustment in the multi-view modes of multiview.MODES
               (outputs/variants/<clip>+<mode>; ba falls back to the archived step-1 solution)

  python -m r2s.views_eval <clip> [out.json]
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from .config import Clip, OUTPUTS, save_json
from . import multiview, perception, source

ARCHIVED_BA = OUTPUTS / 'experiments' / 'multiview_2026-10-07'


def solutions(name, frames, keys):
    clip = Clip(name)
    cam = clip.camera()
    H, W = cam.H, cam.W
    disp = np.load(clip.prep / 'disp.npy').astype(np.float32)
    Z1 = perception.metric_depth(clip)
    cams = dict(np.load(clip.prep / 'cams.npz'))
    sol = {}
    sol['static'] = dict(R=np.repeat(np.eye(3)[None], len(keys), 0), t=np.zeros((len(keys), 3)),
                         f=np.full(len(keys), cams['f'][0]), Z=Z1[keys])
    sol['rotation'] = dict(R=np.array([cams['R'][k] @ cams['R'][0].T for k in keys]), t=np.zeros((len(keys), 3)),
                           f=cams['f'][keys], Z=Z1[keys])
    for tag in ('ba', 'sift', 'vggt_tracks', 'vggt'):
        path = OUTPUTS / 'variants' / f'{name}+{tag}' / 'prep' / 'multiview' / 'keyframes.npz'
        if tag == 'ba' and not path.exists():
            path = ARCHIVED_BA / name / 'multiview_solution' / 'keyframes.npz'
        if not path.exists():
            continue
        kb = dict(np.load(path))
        assert list(kb['keys']) == list(keys)
        sol[tag] = dict(R=kb['R'], t=kb['t'], f=np.full(len(keys), float(kb['f'])),
                        Z=np.stack([multiview.keyframe_depth(kb, i, disp[k], W, H) for i, k in enumerate(keys)]))
    p = OUTPUTS / 'variants' / f'{name}+vggt' / 'prep' / 'multiview' / 'vggt_metric.npz'
    if p.exists():
        g = dict(np.load(p))
        sol['vggt_raw'] = dict(R=g['R'], t=g['t'], f=np.full(len(keys), float(g['f'])), Z=g['depth'])
    order = ['static', 'rotation', 'vggt_raw', 'ba', 'vggt_tracks', 'vggt', 'sift']
    return {k: sol[k] for k in order if k in sol}


def warp(s, i, j, W, H):
    """Pixel positions in keyframe j of every pixel of keyframe i, and their depth there."""
    vv, uu = np.mgrid[0:H, 0:W]
    z = s['Z'][i]
    x = np.stack([(uu - W / 2) * z / s['f'][i], (vv - H / 2) * z / s['f'][i], z], -1).reshape(-1, 3)
    X = (x - s['t'][i]) @ s['R'][i]
    y = X @ s['R'][j].T + s['t'][j]
    q = np.stack([s['f'][j] * y[:, 0] / y[:, 2] + W / 2, s['f'][j] * y[:, 1] / y[:, 2] + H / 2], 1)
    return q, y[:, 2]


def ncc(frames, static, keys, s, i, j, min_overlap=0.05):
    H, W = static.shape[1:]
    q, z = warp(s, i, j, W, H)
    ok = (z > 0) & (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1) & static[keys[i]].ravel()
    qi = q[ok].round().astype(int)
    okm = static[keys[j]][qi[:, 1], qi[:, 0]]
    if okm.sum() < min_overlap * H * W:
        return None
    a = cv2.cvtColor(frames[keys[i]], cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()[ok][okm]
    b = cv2.cvtColor(frames[keys[j]], cv2.COLOR_RGB2GRAY).astype(np.float32)[qi[okm, 1], qi[okm, 0]]
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def probe_matches(frames, static, keys, min_inliers=20):
    """Held-out geometric probe: ORB matches between all keyframe pairs (a different detector from the SIFT used
    by the 'sift' mode), ratio test, MAGSAC-verified at 1 px. Returns {(i, j): (A, B)}."""
    ak = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
    feats = []
    for k in keys:
        hsv = cv2.cvtColor(frames[k], cv2.COLOR_RGB2HSV)
        spec = cv2.dilate(((hsv[..., 2] > 230) & (hsv[..., 1] < 60)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        kp, de = ak.detectAndCompute(cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY), (static[k] & ~spec).astype(np.uint8) * 255)
        feats.append((np.array([q.pt for q in kp]).reshape(-1, 2), de))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (pi, di), (pj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(di) < min_inliers or len(dj) < min_inliers:
                continue
            good = [m[0] for m in bf.knnMatch(di, dj, k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < min_inliers:
                continue
            A, B = pi[[g.queryIdx for g in good]], pj[[g.trainIdx for g in good]]
            inl = multiview.verified(A, B, 1.0)
            if inl.sum() >= min_inliers:
                out[(i, j)] = (A[inl], B[inl])
    return out


def probe_error(s, probes, W, H):
    """Median reprojection error (px) of the probe matches under a solution, by keyframe gap."""
    err = {}
    for (i, j), (A, B) in probes.items():
        z = multiview.sample(s['Z'][i], A)
        x = np.stack([(A[:, 0] - W / 2) * z / s['f'][i], (A[:, 1] - H / 2) * z / s['f'][i], z], 1)
        y = ((x - s['t'][i]) @ s['R'][i]) @ s['R'][j].T + s['t'][j]
        q = np.stack([s['f'][j] * y[:, 0] / y[:, 2] + W / 2, s['f'][j] * y[:, 1] / y[:, 2] + H / 2], 1)
        err[(i, j)] = np.linalg.norm(q - B, axis=1)
    pick = lambda c: np.concatenate([e for (i, j), e in err.items() if c(i, j)] or [np.zeros(0)])
    med = lambda v: round(float(np.median(v)), 2) if len(v) else None
    return dict(gap_1_2=med(pick(lambda i, j: j - i < 3)), gap_3_9=med(pick(lambda i, j: 3 <= j - i < 10)),
                gap_10plus=med(pick(lambda i, j: j - i >= 10)), with_frame0=med(pick(lambda i, j: i == 0)),
                n_10plus=int(len(pick(lambda i, j: j - i >= 10))))


def warped_image(frames, keys, s, i, j, valid=None):
    """Keyframe i rendered into keyframe j's view (z-buffered splat onto the four neighbouring pixels, so zoomed-in
    views have no holes), restricted to `valid` source pixels; returns the image and the covered mask."""
    H, W = frames[0].shape[:2]
    q, z = warp(s, i, j, W, H)
    ok = (z > 0) & (q[:, 0] > -1) & (q[:, 0] < W) & (q[:, 1] > -1) & (q[:, 1] < H)
    if valid is not None:
        ok &= valid.ravel()
    src = frames[keys[i]].reshape(-1, 3)[ok]
    q, z = q[ok], z[ok]
    out = np.zeros((H, W, 3), np.uint8)
    zbuf = np.full((H, W), np.inf)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        u = np.floor(q[:, 0]).astype(int) + dx
        v = np.floor(q[:, 1]).astype(int) + dy
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, zz, cc = u[m], v[m], z[m], src[m]
        order = np.argsort(-zz)                            # far first, near overwrites
        u, v, zz, cc = u[order], v[order], zz[order], cc[order]
        closer = zz < zbuf[v, u]
        out[v[closer], u[closer]] = cc[closer]
        zbuf[v[closer], u[closer]] = zz[closer]
    return out, np.isfinite(zbuf)


def evaluate(name, every=10):
    clip = Clip(name)
    frames = source.frames(clip)
    masks = perception.load_masks(clip)
    static = multiview.static_masks(clip, masks)
    keys = list(range(0, len(frames), every))
    sols = solutions(name, frames, keys)
    K = len(keys)
    Ms = {}
    for tag, s in sols.items():
        M = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(K):
                if i != j:
                    v = ncc(frames, static, keys, s, i, j)
                    M[i, j] = np.nan if v is None else v
        Ms[tag] = M
    common = np.all([np.isfinite(M) for M in Ms.values()], 0)   # pairs every solution can compare
    res = {}
    for tag, M in Ms.items():
        Mc = np.where(common, M, np.nan)
        mean = lambda v: round(float(np.nanmean(v)), 4) if np.isfinite(v).any() else None
        res[tag] = dict(gap_10_frames=mean(np.array([Mc[i, i + 1] for i in range(K - 1)])),
                        gap_30_frames=mean(np.array([Mc[i, i + 3] for i in range(K - 3)])),
                        frame0_to_all=mean(Mc[0, 1:]), frame0_to_last_third=mean(Mc[0, 2 * K // 3:]),
                        pairs_100plus_frames=mean(np.array([Mc[i, j] for i in range(K) for j in range(i + 10, K)])),
                        frame0_row=[None if np.isnan(v) else round(float(v), 3) for v in M[0]],
                        overlap_pairs=int(np.isfinite(M).sum()), common_pairs=int(common.sum()))
    probes = probe_matches(frames, static, keys)
    H, W = static.shape[1:]
    for tag, sol in sols.items():
        res[tag]['probe_reproj_px'] = probe_error(sol, probes, W, H)
    return dict(clip=name, keyframes=keys, methods=res, n_probe_pairs=len(probes)), sols, frames, keys, static


LABELS = dict(static='no camera motion', rotation='single view (rotation + zoom)', vggt_raw='VGGT alone',
              ba='BA, flow tracks', vggt_tracks='BA + VGGT tracks', vggt='BA from VGGT + its tracks',
              sift='BA + SIFT matches')


def figure(frames, keys, sols, j, path, i=0, block=48, methods=None, valid=None):
    """Keyframe i warped into keyframe j by each solution, checkerboarded with the real frame j: where the solution
    is right, edges continue across the squares. Only static anatomy of keyframe i is warped; the rest of
    the picture shows the real frame dimmed."""
    H, W = frames[0].shape[:2]
    real = frames[keys[j]]
    tiles = []
    for tag in methods or [m for m in sols if m != 'static']:
        warped, cover = warped_image(frames, keys, sols[tag], i, j, valid)
        yy, xx = np.mgrid[0:H, 0:W]
        chk = ((yy // block + xx // block) % 2 == 0) & cover
        img = np.where(chk[..., None], warped, real)
        img = np.where(cover[..., None], img, (real * 0.35).astype(np.uint8))
        img = img.copy()
        cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(img, LABELS.get(tag, tag), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    top = real.copy()
    cv2.rectangle(top, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(top, f'real frame {keys[j]}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    src = frames[keys[i]].copy()
    cv2.rectangle(src, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(src, f'real frame {keys[i]} (warped below)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cells = [src, top] + tiles
    if len(cells) % 2:
        cells.append(np.zeros_like(real))
    grid = np.concatenate([np.concatenate(cells[r:r + 2], 1) for r in range(0, len(cells), 2)], 0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def table(r):
    rows = ['held-out ORB matches, median reprojection px: gap 1-2 / 3-9 / >=10 keyframes / with frame 0']
    for tag, m in r['methods'].items():
        e = m['probe_reproj_px']
        rows.append(f"{tag:12s} {e['gap_1_2']} / {e['gap_3_9']} / {e['gap_10plus']} ({e['n_10plus']}) / {e['with_frame0']}")
    rows.append('image NCC      10 fr   30 fr   f0->all  f0->last3rd  >=100 fr')
    for tag, m in r['methods'].items():
        rows.append(f"{tag:12s} {m['gap_10_frames']:6.3f}  {m['gap_30_frames']:6.3f}  {m['frame0_to_all']:7.3f}  "
                    f"{m['frame0_to_last_third']:11.3f}  {m['pairs_100plus_frames']}   ({m['overlap_pairs']} pairs overlap)")
    return '\n'.join(rows)


if __name__ == '__main__':
    r, sols, frames, keys, static = evaluate(sys.argv[1])
    print(table(r))
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'alignment.json', r)
        (out / 'alignment.txt').write_text(table(r) + '\n')
        row = np.array([[np.nan if v is None else v for v in m['frame0_row']] for m in r['methods'].values()])
        late = [j for j in range(len(keys) * 2 // 3, len(keys)) if np.isfinite(row[:, j]).all()]
        j = late[len(late) // 2] if late else int(np.nanargmax(np.isfinite(row).all(0) * np.arange(len(keys))))
        figure(frames, keys, sols, j, out / f'warp_frame0_to_{keys[j]}.jpg', valid=static[0],
               methods=[m for m in ('rotation', 'vggt_raw', 'ba', 'sift') if m in sols])
