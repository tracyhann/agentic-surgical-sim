"""One shot of a segmented multi-shot clip through the 3D steps, as its own clip (after r10: the lung video is 9 shots;
every SAM 3 object has to be reconstructed in 3D at least once).

    python -m t2s.shot <src clip> <src seg ver> <new clip> <first> <last>      (frame numbers of the src clip)

Steps, one process at a time (the depth prep alone takes ~10 GB): frames (t2s.data; the new clip has to be in
t2s.data.CLIPS), masks + texts (t2s.seg3.subclip), cameras and depth (t2s.geom), instruments (t2s.instruments v01),
the spec's organs that the selection fits and, with the generic fit, those it skips as tubular / membrane (t2s.organ
v20), background (t2s.background v09), the page's scene and comparison video (t2s.export_viewer, t2s.report). A step
that fails is reported and the rest goes on where it can.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from . import data as D

ROOT = Path(__file__).resolve().parents[1]


def step(log, name, args):
    with open(log, 'a') as f:
        f.write(f'\n===== {name}: {" ".join(args)}\n')
        f.flush()
        p = subprocess.run([sys.executable, '-m'] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=dict(os.environ, PYTHONPATH=str(ROOT)))
    print(f'[shot] {name}: exit {p.returncode}', flush=True)
    return p.returncode == 0


def main(src, src_ver, clip, a, b):
    out = D.OUT / clip
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'shot_run.log'
    log.write_text('')
    if not step(log, 'frames', ['t2s.data', clip]):
        return
    from . import seg3
    names = seg3.subclip(src, clip, int(a), int(b), src_ver)
    if not step(log, 'geometry', ['t2s.geom', clip, 'v01']):
        return
    for d in ('instruments', 'organs', 'background'):
        (out / d).mkdir(exist_ok=True)
    if any(n.startswith('instrument_') for n in names):
        step(log, 'instruments', ['t2s.instruments', clip, '--ver', 'v01'])
    # organs: what the selection fits, and what it skips only for being a tube / membrane (generic fit)
    from . import organ as O, views2
    c = O.cfg_of('v20')
    fit, skipped, _, _ = O.select_organs(clip, c['seg'], list(views2.load(clip).names))
    forced = sorted({s['mask'] for s in skipped if s.get('mask') and 'membrane / tube' in s['reason']} - {o['mask'] for o in fit})
    views2.load.cache_clear()
    if fit or forced:
        step(log, 'organs', ['t2s.organ', clip, '--ver', 'v20'] + [x for m in forced for x in ('--mask', m)] + ([o['name'] for o in fit] if forced else []))
    step(log, 'background', ['t2s.background', clip, 'v09'])
    step(log, 'export', ['t2s.export_viewer', clip])          # also for a shot without any moving model
    step(log, 'video', ['t2s.report', clip, 'seg=v01'])
    sc = ROOT / 'site' / 'v2' / 'data' / clip / 'scene.json'
    if sc.exists():
        ob = json.loads(sc.read_text())['objects']
        print(f"[shot] {clip}: " + '; '.join(f"{o['name']} -> {o['kind']}" for o in ob), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:6])
