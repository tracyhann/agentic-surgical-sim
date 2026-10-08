"""Interpenetration check of a v2 RECONSTRUCTION, before any simulation (r06): do the fitted instruments pass through
the fitted organ models where the spec declares no opening, and when do they touch them.

    python -m t2s.recon_check <clip> [organ_<name>=vNN] [instruments=vNN] [every=5]
    -> outputs/t2s/<clip>/checks/recon_<instruments>_<organ>-<ver>.json, one line per instrument printed

Per instrument and organ, over the frames where the instrument is seen: the shaft (25 samples over the last 60 mm,
radius from the model) against the organ's closed 4D surface. overlap_mm > 0: the shaft's surface is that deep inside
the organ (illegal unless the organ model carries an 'opening:<instrument>' attachment, as t2s.assemble reads it);
< 0: the gap between them. 'contact' = within 2 mm of the surface.
"""
import json
import sys

import numpy as np

from r2s import quality as Q
from . import data as D


def inside(X, F, P):
    """Points P inside the closed surface (X, F): generalised winding number (sum of the triangles' solid angles,
    Van Oosterom-Strackee). r2s.quality.signed_distance takes the sign from the nearest surface sample's normal,
    which is unreliable for points far from the surface (seen: 'inside by 71 mm' for a 21 mm thick organ)."""
    a, b, c = (X[F[:, i]][None] - P[:, None] for i in range(3))
    la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
    num = np.einsum('pfi,pfi->pf', a, np.cross(b, c))
    den = la * lb * lc + np.einsum('pfi,pfi->pf', a, b) * lc + np.einsum('pfi,pfi->pf', b, c) * la + np.einsum('pfi,pfi->pf', c, a) * lb
    w = np.arctan2(num, den).sum(1) / (2 * np.pi)
    return np.abs(w) > 0.5


def latest(d):
    vs = sorted(p.name for p in d.glob('v[0-9][0-9]') if (p / 'model.npz').exists())
    return vs[-1] if vs else None


def check(clip, organ_versions=None, ins_ver=None, every=5):
    ins_ver = ins_ver or latest(D.OUT / clip / 'instruments')
    zi = np.load(D.OUT / clip / 'instruments' / ins_ver / 'model.npz')
    res = dict(clip=clip, instruments=ins_ver, organs={}, every=every, pairs={})
    for od in sorted((D.OUT / clip / 'organs').glob('*/')):
        ver = (organ_versions or {}).get(od.name) or latest(od)
        if not ver:
            continue
        zo = np.load(od / ver / 'model.npz')
        res['organs'][od.name] = ver
        X4, F = zo['verts4d'].astype(float), zo['faces']
        openings = {str(a).split(':')[1] for a in zo['attach_to'].astype(str) if str(a).startswith('opening:')} if 'attach_to' in zo.files else set()
        for nm in [str(s) for s in zi['names']]:
            tip, dr, vis = zi[f'{nm}__tip'].astype(float), zi[f'{nm}__dir'].astype(float), zi[f'{nm}__visible'].astype(bool)
            rad = float(json.loads(str(zi[f'{nm}__params']))['radius']) if f'{nm}__params' in zi.files else 0.0025
            ks = [k for k in range(0, len(X4), every) if vis[k]]
            ov = []
            for k in ks:
                P = tip[k] - np.linspace(0, 0.06, 25)[:, None] * dr[k] / np.linalg.norm(dr[k])
                dist = np.abs(Q.signed_distance(X4[k], F)(P))                      # distance to the surface
                ov.append(float(np.where(inside(X4[k], F, P), dist, -dist).max()) + rad)
            ov = np.array(ov) * 1000
            legal = any(o in nm for o in openings)
            bad = ov > 1.0
            res['pairs'][f'{nm}|{od.name}'] = dict(
                frames_checked=len(ks), opening_declared=legal, frames_inside=int(bad.sum()),
                frames_inside_illegal=0 if legal else int(bad.sum()),
                max_overlap_mm=round(float(ov.max()), 1), median_overlap_when_inside_mm=round(float(np.median(ov[bad])), 1) if bad.any() else 0.0,
                frames_contact=int((np.abs(ov) <= 2.0).sum()), median_gap_when_outside_mm=round(float(np.median(-ov[~bad])), 1) if (~bad).any() else None,
                inside_frames=[int(k) for k, b in zip(ks, bad) if b][:60])
    out = D.OUT / clip / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    name = f"recon_{ins_ver}_" + '_'.join(f'{o}-{v}' for o, v in res['organs'].items()) + '.json'
    (out / name).write_text(json.dumps(res, indent=1))
    for pair, r in res['pairs'].items():
        print(f"[recon_check] {clip} {ins_ver} {pair}: inside {r['frames_inside']} / {r['frames_checked']} frames"
              f"{' (opening declared)' if r['opening_declared'] else ' (ILLEGAL)' if r['frames_inside'] else ''}, max {r['max_overlap_mm']} mm; "
              f"contact {r['frames_contact']}; median gap outside {r['median_gap_when_outside_mm']} mm")
    return res


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[2:] if '=' in a)
    check(sys.argv[1], {k[6:]: v for k, v in kv.items() if k.startswith('organ_')}, kv.get('instruments'), int(kv.get('every', 5)))
