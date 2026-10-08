"""Metric depth for every frame: Depth Anything V2 relative inverse depth d -> 1/z = a * d + b_k.
One global slope a (the relative depth is consistent within this static-camera clip) and a per-frame offset b_k,
both fitted to instrument shafts of known 5 mm diameter (depth = f_k * 5 mm / apparent width, SAM 2 masks)."""
import sys
import numpy as np
sys.path.insert(0, 'realistic'); sys.path.insert(0, 'real3d')
import segment as SG  # noqa: E402

SHAFT_D = 0.005


def shaft_samples(m, fk):
    ys, xs = np.nonzero(m)
    if len(xs) < 300:
        return []
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    d = np.linalg.svd(P - c)[2][0]
    s, nrm = (P - c) @ d, np.array([-d[1], d[0]])
    out = []
    for t in np.linspace(np.percentile(s, 3), np.percentile(s, 97), 30):
        sel = np.abs(s - t) < 1.5
        if sel.sum() < 4:
            continue
        w = np.ptp((P[sel] - c) @ nrm) + 1
        q = c + d * t
        if 5 < w < 80 and 0 <= q[0] < 640 and 0 <= q[1] < 360:
            out.append((q[0], q[1], fk * SHAFT_D / w))
    return out


def fit(disp, masks, cam_f, use=(1,)):
    """Robust global slope + per-frame offsets. `use`: mask indices of the instruments used as rulers."""
    rows = []
    for k in range(len(disp)):
        for i in use:
            for u, v, z in shaft_samples(masks[k, i], cam_f[k]):
                rows.append((k, disp[k, int(v), int(u)], 1 / z))
    R = np.array(rows)
    K = len(disp)
    a, b = 3.0, np.full(K, 5.0)
    for _ in range(8):                                     # alternate: offsets per frame, then the global slope
        for k in range(K):
            sel = R[:, 0] == k
            if sel.sum() >= 3:
                b[k] = np.median(R[sel, 2] - a * R[sel, 1])
        bb = b[R[:, 0].astype(int)]
        res = R[:, 2] - bb
        w = np.abs(res - a * R[:, 1]) < 3 * np.median(np.abs(res - a * R[:, 1])) + 1e-6
        a = float((R[w, 1] @ res[w]) / (R[w, 1] @ R[w, 1]))
    have = np.array([(R[:, 0] == k).sum() >= 3 for k in range(K)])
    b = np.interp(np.arange(K), np.nonzero(have)[0], b[have])
    b = np.convolve(np.pad(b, 7, mode='edge'), np.ones(15) / 15, mode='valid')     # ~0.6 s smoothing
    z_pred = 1 / (a * R[:, 1] + b[R[:, 0].astype(int)])
    err = np.abs(z_pred - 1 / R[:, 2])
    return a, b, dict(n=len(R), median_err_mm=float(np.median(err) * 1000), p90_err_mm=float(np.percentile(err, 90) * 1000))


def depth(disp_k, a, b_k):
    return 1.0 / np.clip(a * disp_k + b_k, 1.0, None)


if __name__ == '__main__':
    masks, names = SG.load()
    disp = np.load('real3d/disp_rel.npy')
    cam_f = np.load('realistic/chole/geometry.npz')['cam_f']
    for use, lab in (((1,), 'grasper'), ((2,), 'probe'), ((1, 2), 'both')):
        a, b, st = fit(disp, masks, cam_f, use)
        z0 = depth(disp[0], a, b[0])
        rep = {n: f'{np.median(z0[masks[0, i]]) * 100:.1f} cm' for i, n in enumerate(names)}
        gb = z0[masks[0, 0]]
        print(f'{lab:8s} a={a:.3f} b0={b[0]:.2f}  fit err median {st["median_err_mm"]:.1f} mm p90 {st["p90_err_mm"]:.1f} (n={st["n"]})  '
              f'gallbladder {np.percentile(gb, 5) * 100:.1f}-{np.percentile(gb, 95) * 100:.1f} cm | {rep}')
    a, b, st = fit(disp, masks, cam_f, (1, 2))
    np.savez('real3d/metric.npz', a=a, b=b)
