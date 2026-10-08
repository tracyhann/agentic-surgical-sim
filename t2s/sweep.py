"""Parallel configurations of one assembly round: python -m t2s.sweep <clip> rNN "k=v,k=v" "k=v" ... [--jobs N]
-> outputs/t2s/<clip>/rounds/rNN/<config>/ (each a t2s.assemble run) and rounds/rNN/sweep.json (main numbers)."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import data as D

ROOT = Path(__file__).resolve().parents[1]


def row(m):
    out = {}
    for name, r in m['eval'].items():
        if name == 'instruments':
            out['illegal_tool_in_organ_frames'] = sum(v['tip_inside_organ_frames'] for v in r.values())
            continue
        out[name] = {k: r.get(k) for k in ('sim_iou', 'sim_iou_2d', 'recon_iou', 'err_vs_recon_mm', 'motion_explained',
                                         'free_motion_explained', 'inverted_tets_max', 'organ_behind_background')}
    return out


def main(argv):
    clip, rnd = argv[0], argv[1]
    jobs = int(argv[argv.index('--jobs') + 1]) if '--jobs' in argv else 4
    cfgs = [a for a in argv[2:] if a != '--jobs' and not a.isdigit()]

    def run(cfg):
        name = cfg.replace(',', '_') or 'default'
        args = [clip, f'{rnd}/{name}'] + [kv for kv in cfg.split(',') if kv]
        out = D.OUT / clip / 'rounds' / rnd / name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'run.log', 'w') as f:
            p = subprocess.run([sys.executable, '-m', 't2s.assemble'] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                               env=dict(os.environ, PYTHONPATH=str(ROOT)))
        return name, p.returncode
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(run, cfgs))
    table = {}
    for name, rc in res:
        mp = D.OUT / clip / 'rounds' / rnd / name / 'metrics.json'
        table[name] = row(json.loads(mp.read_text())) if rc == 0 and mp.exists() else dict(failed=rc)
    (D.OUT / clip / 'rounds' / rnd / 'sweep.json').write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
