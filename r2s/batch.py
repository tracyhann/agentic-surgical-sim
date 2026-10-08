"""python -m r2s.batch <clip> [<clip> ...] [--from organ] [--jobs 4]

Runs the pipeline for whole clips: prep and scene once per clip, then the organ conditions (organ, sim in parallel
processes, eval), then render + compare. Logs go to outputs/logs/<clip>_<stage>[_<cond>].log."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from .config import Clip, OUTPUTS, ROOT, CONDITIONS

STAGES = ['prep', 'scene', 'organ', 'sim', 'eval', 'render', 'compare']


def run(args, log):
    t0 = time.time()
    with open(log, 'w') as f:
        # run from outputs/logs: MuJoCo (MUJOCO_LOG.TXT) and TetGen (_skipped.*) drop files in the working directory
        p = subprocess.run([sys.executable, '-m', 'r2s'] + args, stdout=f, stderr=subprocess.STDOUT, cwd=OUTPUTS / 'logs',
                           env=dict(os.environ, PYTHONPATH=str(ROOT)))
    status = 'ok' if p.returncode == 0 else f'FAILED ({p.returncode})'
    print(time.strftime('%H:%M:%S'), ' '.join(args), status, f'{time.time() - t0:.0f} s', flush=True)
    return p.returncode == 0


def clip_conditions(name):
    clip = Clip(name)
    conds = list(clip.get('conditions', CONDITIONS))
    if not clip['objects'][clip.organ].get('template'):
        conds = [c for c in conds if c in ('measured', 'primitive')]
    return conds


def main(argv):
    start = 'prep'
    jobs = 4
    names = []
    i = 0
    while i < len(argv):
        if argv[i] == '--from':
            start, i = argv[i + 1], i + 2
        elif argv[i] == '--jobs':
            jobs, i = int(argv[i + 1]), i + 2
        else:
            names.append(argv[i])
            i += 1
    logs = OUTPUTS / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    todo = STAGES[STAGES.index(start):]
    for name in names:
        conds = clip_conditions(name)
        for st in ('prep', 'scene', 'organ'):
            if st in todo and not run([name, st] + conds, logs / f'{name}_{st}.log'):
                return 1
        with ThreadPoolExecutor(jobs) as ex:
            if 'sim' in todo:
                list(ex.map(lambda c: run([name, 'sim', c], logs / f'{name}_sim_{c}.log'), conds))
            if 'eval' in todo:
                list(ex.map(lambda c: run([name, 'eval', c], logs / f'{name}_eval_{c}.log'), conds))
            if 'render' in todo:
                list(ex.map(lambda c: run([name, 'render', c], logs / f'{name}_render_{c}.log'), conds))
        if 'compare' in todo:
            run([name, 'compare'] + conds, logs / f'{name}_compare.log')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
