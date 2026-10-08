"""Hold-out check of the 4D study's parameter choices: would the same configuration win if it had been chosen on the
first part of the clip only (pan + lift, frames 0-129) and judged on the rest (probe phase, 130-250)?

    python -m r2s.holdout4d r14 r15 r17 ...      -> outputs/iter/rounds/holdout.json and a table per round

Per run: silhouette IoU of the gallbladder, the sheet and the duct tubes per keyframe, recomputed from traj.npz with
the tissue versions and parameters recorded in its metrics.json, split into the selection part and the test part.
"""
import json
import sys
import numpy as np
from . import views, quality as Q
from . import scene4d as S

SPLIT = 130


def per_frame(V, run):
    m = json.loads((run / 'metrics.json').read_text())
    P = dict(S.PARAMS, **m['params'])
    tr = np.load(run / 'traj.npz')
    t_ctl = tr['t']
    ks = list(range(0, V.n, 10))
    out = {}
    for t in ('gallbladder', 'membrane'):
        T = S.load_tissue(t, m['tissues'][t])
        sim4 = S.to_frames(tr[t], t_ctl, V)
        mk = S.tissue_masks(T, V, t)
        saved = V._masks
        try:
            V._masks = np.concatenate([saved, mk[:, None]], 1)
            V.names = list(V.names) + [f'_{t}']
            out[t] = Q.silhouette_iou(V, sim4, T['faces'], f'_{t}', ks)
        finally:
            V._masks = saved
            V.names = V.names[:-1]
    T = S.load_tissue('ducts', m['tissues']['ducts'])
    parts = S.sim_bodies(T, 'ducts', P, S.load_tissue('gallbladder', m['tissues']['gallbladder']))
    traj = {k: tr[k] for k in tr.files}
    r = S.evaluate_cables(V, T, traj, t_ctl, ks, parts)
    if '_tube' in r:
        Vt, Ft = r['_tube']
        saved = V._masks
        try:
            V._masks = np.concatenate([saved, S.tissue_masks(T, V, 'ducts')[:, None]], 1)
            V.names = list(V.names) + ['_ducts']
            out['ducts'] = Q.silhouette_iou(V, Vt, Ft, '_ducts', ks)
        finally:
            V._masks = saved
            V.names = V.names[:-1]
    return np.array(ks), out


def main(rounds):
    V = views.load('chole_a', 'sift', S.CAMS['refined'])
    res = {}
    for rnd in rounds:
        runs = sorted(p.parent for p in (S.ITER / 'rounds' / rnd).glob('*/traj.npz'))
        rows = {}
        for run in runs:
            ks, pf = per_frame(V, run)
            sel, test = ks < SPLIT, ks >= SPLIT
            rows[run.name] = {t: dict(select=round(float(x[sel].mean()), 4), test=round(float(x[test].mean()), 4))
                              for t, x in pf.items()}
        res[rnd] = rows
        print(f'\n{rnd}')
        for t in ('gallbladder', 'membrane', 'ducts'):
            sc = {n: r[t] for n, r in rows.items() if t in r}
            if not sc:
                continue
            best_sel = max(sc, key=lambda n: sc[n]['select'])
            best_test = max(sc, key=lambda n: sc[n]['test'])
            rank_sel = sorted(sc, key=lambda n: -sc[n]['select'])
            rank_test = sorted(sc, key=lambda n: -sc[n]['test'])
            print(f'  {t:12s} chosen on 0-129: {best_sel}  | best on 130-250: {best_test}  | same: {best_sel == best_test}')
            print(f'  {"":12s} test score of the choice {sc[best_sel]["test"]:.3f} vs best {sc[best_test]["test"]:.3f}; '
                  f'ranks by selection {[rank_test.index(n) + 1 for n in rank_sel]}')
    prev = S.ITER / 'rounds' / 'holdout.json'
    allres = json.loads(prev.read_text()) if prev.exists() else {}
    allres.update(res)
    prev.write_text(json.dumps(allres, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
