"""Cross-agent consistency of a reconstruction (r07): the organ agent and the instrument agent fit the same frames
independently, and the organ model grows into the pixels an instrument hides ('unknown' to its silhouette term), so
shafts end up inside it (t2s.recon_check). Rule, from visibility alone: where an instrument is SEEN it is in front of
the organ - unless the spec declares an opening for it. This module enforces the rule on the organ's 4D as a
geometric projection (no refit): per frame, organ nodes whose camera ray passes through a seen shaft and that lie in
front of the shaft's far side are moved back along their ray to behind it; the push is spread over the mesh
(neighbours follow with a decay) so that it becomes a dent, not a spike.

    python -m t2s.consistency <clip> <organ> <ver_in> <ver_out> [instruments=vNN] [decay=0.85] [iters=12]
    -> outputs/t2s/<clip>/organs/<organ>/<ver_out>/ (model.npz with the corrected verts4d, carve.json)
"""
import json
import shutil
import sys

import numpy as np

from . import data as D, views2


def carve(clip, organ, ver_in, ver_out, ins_ver=None, margin=0.0005, decay=0.85, iters=12, length=0.2):
    V = views2.load(clip)
    src, dst = D.OUT / clip / 'organs' / organ / ver_in, D.OUT / clip / 'organs' / organ / ver_out
    z = dict(np.load(src / 'model.npz', allow_pickle=True))
    X4 = z['verts4d'].astype(np.float64).copy()
    T = z['tets'].astype(int)
    n, N = X4.shape[:2]
    iv = sorted(p.parent.name for p in (D.OUT / clip / 'instruments').glob('v[0-9][0-9]/model.npz'))
    ins_ver = ins_ver or iv[-1]
    zi = np.load(D.OUT / clip / 'instruments' / ins_ver / 'model.npz')
    ipp = D.OUT / clip / 'seg' / 'instrument_prompts.json'
    pu = (json.loads(ipp.read_text()).get('puncture') or {}) if ipp.exists() else {}
    tools = []
    for nm in [str(s) for s in zi['names']]:
        if pu.get('instrument') and nm.endswith(str(pu['instrument'])):       # declared opening: may be inside
            continue
        dr = zi[f'{nm}__dir'].astype(float)
        tools.append((nm, zi[f'{nm}__tip'].astype(float), dr / np.linalg.norm(dr, axis=1, keepdims=True),
                      zi[f'{nm}__visible'].astype(bool), float(json.loads(str(zi[f'{nm}__params']))['radius'])))
    # node adjacency (tet edges)
    e = np.unique(np.sort(np.concatenate([T[:, [a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]), 1), axis=0)
    deg = np.bincount(e.ravel(), minlength=N).astype(float)
    sgn0 = np.sign(np.einsum('ij,ij->i', np.cross(X4[0][T[:, 1]] - X4[0][T[:, 0]], X4[0][T[:, 2]] - X4[0][T[:, 0]]), X4[0][T[:, 3]] - X4[0][T[:, 0]]))
    vol = lambda X: np.einsum('ij,ij->i', np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), X[T[:, 3]] - X[T[:, 0]])
    rep = dict(clip=clip, organ=organ, source=ver_in, instruments=ins_ver, geometry=V.geometry, margin_mm=margin * 1000, decay=decay,
               iters=iters, tools=[t[0] for t in tools], frames=[])
    for k in range(n):
        C = V.pos[k]
        X = X4[k]
        ray = X - C
        s = np.linalg.norm(ray, axis=1)
        w = ray / s[:, None]
        need = np.zeros(N)
        for nm, tip, dr, vis, r in tools:
            if not vis[k]:
                continue
            ev = -dr[k]                                              # along the shaft, away from the tip
            b, dd, ee = w @ ev, w @ (C - tip[k]), (C - tip[k]) @ ev
            tc = np.clip((ee - b * dd) / np.clip(1 - b * b, 1e-9, None), 0.0, length)
            A = tip[k] + tc[:, None] * ev
            sc = ((A - C) * w).sum(1)
            dist = np.linalg.norm(C + sc[:, None] * w - A, axis=1)
            R = r + margin
            far = sc + np.sqrt(np.clip(R * R - dist * dist, 0, None))   # where the ray leaves the shaft
            need = np.maximum(need, np.where((dist < R) & (sc > 0.001), far - s, 0.0))
        need = np.clip(need, 0, None)
        if need.max() <= 0:
            rep['frames'].append(dict(frame=k, pushed=0))
            continue
        a = need.copy()
        for _ in range(iters):                                        # neighbours follow with a decay (a dent)
            nb = np.zeros(N)
            np.add.at(nb, e[:, 0], a[e[:, 1]])
            np.add.at(nb, e[:, 1], a[e[:, 0]])
            a = np.maximum(need, decay * nb / np.maximum(deg, 1))
        v0 = np.abs(vol(X)).sum()
        X4[k] = X + a[:, None] * w
        rep['frames'].append(dict(frame=k, pushed=int((need > 0).sum()), max_push_mm=round(float(need.max()) * 1000, 2),
                                  moved_over_1mm=int((a > 0.001).sum()), inverted_tets=int((np.sign(vol(X4[k])) != sgn0).sum()),
                                  volume_ratio=round(float(np.abs(vol(X4[k])).sum() / v0), 4)))
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != 'model.npz':
            shutil.copy2(f, dst / f.name)
    z['verts4d'] = X4.astype(z['verts4d'].dtype)
    np.savez_compressed(dst / 'model.npz', **z)
    fr = [f for f in rep['frames'] if f['pushed']]
    rep['summary'] = dict(frames_changed=len(fr), max_push_mm=max((f['max_push_mm'] for f in fr), default=0.0),
                          inverted_tets_max=max((f['inverted_tets'] for f in fr), default=0),
                          volume_ratio_min=min((f['volume_ratio'] for f in fr), default=1.0),
                          volume_ratio_max=max((f['volume_ratio'] for f in fr), default=1.0))
    (dst / 'carve.json').write_text(json.dumps(rep, indent=1))
    print(f"[consistency] {clip} {organ} {ver_in} -> {ver_out} (instruments {ins_ver}): {rep['summary']}")
    return rep


if __name__ == '__main__':
    kv = dict(a.split('=', 1) for a in sys.argv[5:] if '=' in a)
    carve(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], kv.get('instruments'), decay=float(kv.get('decay', 0.85)),
          iters=int(kv.get('iters', 12)))
