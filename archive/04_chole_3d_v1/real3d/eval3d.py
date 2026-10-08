"""3D check of the reconstruction against the video, frame by frame.

  gallbladder outline   IoU of the simulated gallbladder surface (projected with the scope camera of that frame)
                        vs the SAM 2 gallbladder mask, both restricted to pixels no real instrument / strand hides
  gallbladder depth     median |z_sim - z_measured| (mm) where both cover the pixel; z_measured = calibrated
                        Depth Anything depth of that frame
Compared for: the full simulation, the same simulation with the tissue frozen, and 'frame 0 forever'.
  python eval3d.py      -> real3d/chole3d/eval3d.json (+ runs the frozen simulation)
"""
import json, re, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'realistic'))
sys.path.insert(0, str(ROOT / 'real3d'))
import chole_recon as C  # noqa: E402
import chole_sim as S  # noqa: E402
import segment as SG  # noqa: E402
import metric_depth as MD  # noqa: E402

OUT = ROOT / 'real3d' / 'chole3d'


def raster(Xf, tri, R, f):
    """Mask + z-buffer of a triangle surface seen by camera (R, f)."""
    px, zc = C.project(Xf, R, f)
    mask = np.zeros((C.H, C.W), np.uint8)
    zbuf = np.full((C.H, C.W), np.inf, np.float32)
    order = np.argsort(-zc[tri].mean(1))                     # far to near
    for t in tri[order]:
        poly = px[t].astype(np.int32)
        tmp = np.zeros_like(mask)
        cv2.fillConvexPoly(tmp, poly, 1)
        sel = tmp.astype(bool)
        zbuf[sel] = np.minimum(zbuf[sel], zc[t].mean())
        mask |= tmp
    return mask.astype(bool), zbuf


def score(flex, g, masks, Z, n_front):
    n = len(masks)
    sim_t, src_t = np.arange(len(flex)) * 0.05, np.arange(n) / C.FPS
    ious, dz = [], []
    for k in range(0, n, 5):
        ks = int(np.argmin(np.abs(sim_t - src_t[k])))
        m, zb = raster(flex[ks, :n_front], g['sheet_tri'], g['cam_R'][k], g['cam_f'][k])
        vis = ~(masks[k, 1] | masks[k, 2] | masks[k, 3])            # where neither instruments nor strands hide it
        real = masks[k, 0] & vis
        m = m & vis
        ious.append((m & real).sum() / max((m | real).sum(), 1))
        both = m & real
        dz.append(float(np.median(np.abs(zb[both] - Z[k][both]))) * 1000 if both.any() else np.nan)
    return dict(gb_iou=round(float(np.mean(ious)), 3), gb_depth_err_mm=round(float(np.nanmean(dz)), 2),
                iou_curve=[round(float(x), 3) for x in ious], depth_curve=[round(float(x), 2) for x in dz])


def main():
    C.OUT = OUT
    S.OUT = OUT
    S.TS = 0.000125
    g = np.load(OUT / 'geometry.npz')
    masks, _ = SG.load()
    disp = np.load(ROOT / 'real3d/disp_rel.npy')
    mt = np.load(ROOT / 'real3d/metric.npz')
    Z = np.stack([MD.depth(disp[k], float(mt['a']), float(mt['b'][k])) for k in range(len(disp))])
    N = len(g['sheet_X'])
    full = score(np.load(OUT / 'flex_traj.npy'), g, masks, Z, N)
    static = score(np.repeat(np.load(OUT / 'flex_traj.npy')[:1], len(np.load(OUT / 'flex_traj.npy')), 0), g, masks, Z, N)
    orig = S.scene_xml

    def frozen_xml(*a, **k):
        return re.sub(r'(<body name="(?:fv|fb|rv)\d+" pos="[^"]+") gravcomp="1">(?:<joint[^>]*/>){3}<inertial[^>]*/></body>', r'\1/>', orig(*a, **k))
    S.scene_xml = frozen_xml
    move_bed, S.MOVE_BED = S.MOVE_BED, False                        # frozen means everything: the anchors too
    try:
        S.run(render=False, tag='_frozen', compare=False)
    finally:
        S.scene_xml = orig
        S.MOVE_BED = move_bed
    frozen = score(np.load(OUT / 'flex_traj_frozen.npy'), g, masks, Z, N)
    res = dict(full=full, frozen=frozen, frame0_forever=static)
    (OUT / 'eval3d.json').write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:15s} gallbladder IoU {v['gb_iou']:.3f}   depth error {v['gb_depth_err_mm']:.2f} mm")
    print('IoU over time (full)  ', full['iou_curve'])
    print('IoU over time (frozen)', frozen['iou_curve'])


if __name__ == '__main__':
    main()
