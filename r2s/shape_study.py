"""Static shape study of one object at frame 0: measured surface vs ellipsoid vs organ template vs template + fit.

All four are built from the same monocular observations (SAM mask + calibrated Depth Anything depth) exactly as the
organ conditions are; they are then scored against an independent reference depth where the dataset has one
(EndoNeRF: stereo), on the pixels where the object is visible.

  python -m r2s.shape_study <clip> <object> <template>     -> outputs/<clip>/shape_study/<object>/
"""
import sys
import numpy as np
import cv2
from .config import Clip, save_json
from . import source, perception, organ as ORG, evaluate as EV, sim as SIM, render as RD
from .scene import silhouette_thickness


def main(name, obj, template):
    clip = Clip(name)
    out = clip.out / 'shape_study' / obj
    out.mkdir(parents=True, exist_ok=True)
    frames = source.frames(clip)
    masks = perception.load_masks(clip)
    Z = perception.metric_depth(clip)
    S = SIM.load_scene(clip)
    cv = SIM.canvas_of(clip, S)
    cam = cv.cam
    ref = source.reference_depth(clip)
    i = clip.obj_index(obj)
    hide = [clip.obj_index(x['mask']) for x in clip.instruments]
    mask = masks[0, i] & ~np.any(masks[0, hide], 0)
    z0 = cv2.bilateralFilter(Z[0].astype(np.float32), 9, 0.004, 5)
    M, H, W = cv.M, cam.H, cam.W
    obs = dict(organ_mask=mask, organ_visible=mask, z0=z0, thick=silhouette_thickness(mask, z0, cv.f0),
               zvis=np.where(mask, z0, S['zb'][M:M + H, M:M + W]))
    res, bodies = {}, {}
    for kind in ('measured', 'primitive', 'template', 'template_fit'):
        if kind == 'measured':
            b = ORG.build(cv, obs, kind, template)
        else:                                  # static study: the fitted surface itself, no tetrahedra needed
            V, F, info = ORG.fit_surface(cv, obs, kind, template)
            b = dict(X=V, faces=np.asarray(F), fit=info)
        bodies[kind] = b
        m, zb = EV.raster(cam, b['X'], b['faces'], cv.R0, cv.f0)
        m &= zb < EV.backdrop_depth(cam, cv, S['zb'], cv.R0, cv.f0)
        r = dict(outline_iou=round(float((m & mask).sum() / (m | mask).sum()), 3),
                 depth_err_mm_monocular=round(float(np.median(np.abs(zb[m & mask] - Z[0][m & mask]))) * 1000, 2))
        if ref is not None:
            Zr, valid = ref
            both = m & mask & valid[0]
            r['depth_err_mm_stereo'] = round(float(np.median(np.abs(zb[both] - Zr[0][both]))) * 1000, 2)
            r['n_px'] = int(both.sum())
        r['fit'] = b.get('fit', {})
        res[kind] = r
    if ref is not None:                                   # how far the monocular depth itself is from the stereo
        Zr, valid = ref
        sel = mask & valid[0]
        res['monocular_depth_vs_stereo_mm'] = round(float(np.median(np.abs(Z[0][sel] - Zr[0][sel]))) * 1000, 2)
        res['monocular_scale_ratio'] = round(float(np.median(Zr[0][sel] / Z[0][sel])), 3)
    save_json(out / 'shape_study.json', res)
    # picture: overlay of each body's visible outline + depth error map against the reference
    tiles = []
    for kind, b in bodies.items():
        m, zb = EV.raster(cam, b['X'], b['faces'], cv.R0, cv.f0)
        img = frames[0].astype(float)
        img[m & ~mask] = 0.4 * img[m & ~mask] + 0.6 * np.array([255, 60, 60])
        img[mask & ~m] = 0.4 * img[mask & ~m] + 0.6 * np.array([60, 160, 255])
        img[m & mask] = 0.7 * img[m & mask] + 0.3 * np.array([90, 255, 90])
        tiles.append(RD.label(img.astype(np.uint8), f'{RD.LABELS[kind]}  IoU {res[kind]["outline_iou"]}'))
    img = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    import imageio.v2 as imageio
    imageio.imwrite(out / 'outlines.jpg', cv2.resize(img, None, fx=0.5, fy=0.5), quality=85)
    for kind, b in bodies.items():
        np.savez(out / f'{kind}.npz', X=b['X'], faces=b['faces'])
    print(res)
    return res


if __name__ == '__main__':
    main(*sys.argv[1:4])
