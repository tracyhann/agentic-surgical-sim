"""r06: a lean comparison of stored geometries of one clip (t2s.geomfix.evaluate grew to 31 GB on chole_derot and was
stopped): ORB probe error across keyframes, photometric consistency of the static scene (frame k warped into k+gap),
static depth-scale spread and camera jitter, with a hard memory cap; plus the warp checker figure.

    PYTHONPATH=. .venv/bin/python -m t2s.geomcheck <clip> <tag> [<tag> ...] [--figure] [--cap-gb 12]
    -> outputs/t2s/<clip>/geometry/check_r06.json (+ warp_checker.jpg)
"""
import json
import resource
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import data as D, geomfix as G


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9          # bytes on macOS


def main(argv):
    clip = argv[0]
    tags = [a for a in argv[1:] if not a.startswith('--') and not a.replace('.', '').isdigit()]
    cap = float(argv[argv.index('--cap-gb') + 1]) if '--cap-gb' in argv else 12.0
    for t in tags:
        G.register(t)
    ctx = G.Ctx(clip)
    st = ctx.static()
    keys = list(range(0, ctx.n, 10))
    probes = G.probe_matches(ctx, st, keys)
    print(f'[check] {clip}: {len(probes)} probe pairs; rss {rss_gb():.1f} GB', flush=True)
    out = D.OUT / clip / 'geometry'
    res, sols = {}, {}
    for tag in tags:
        sol = sols[tag] = G.load_geometry(clip, tag, ctx)
        r = dict(probe=G.probe_error(sol, probes, keys))
        for name, fn in (('photo_10', lambda: G.photometric(ctx, sol, st, gap=10, every=4)),
                         ('photo_30', lambda: G.photometric(ctx, sol, st, gap=30, every=8)),
                         ('scale', lambda: G.static_scale(ctx, sol, st)[0])):
            r[name] = fn()
            if rss_gb() > cap:
                raise SystemExit(f'[check] memory cap {cap} GB exceeded after {tag} {name} ({rss_gb():.1f} GB)')
        centres = np.array([-sol.R[k].T @ sol.t[k] for k in range(ctx.n)])
        r['jitter'] = G.jitter(np.array([Rot.from_matrix(R).as_rotvec() for R in sol.R]), centres)
        r['scope_path_mm'] = round(float(np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1))) * 1000, 1)
        res[tag] = r
        print(f'[check] {tag}: {json.dumps(r)}; rss {rss_gb():.1f} GB', flush=True)
    (out / 'check_r06.json').write_text(json.dumps(res, indent=1))
    if '--figure' in argv:
        G.checker_figure(ctx, sols, [(40, 50), (20, 60), (0, 100), (100, 200), (140, 240)], out / 'warp_checker.jpg')
        print('[check] figure', out / 'warp_checker.jpg', f'rss {rss_gb():.1f} GB')


if __name__ == '__main__':
    main(sys.argv[1:])
