"""Parameter sweep for one round of the 4D study (system identification): runs scene4d with several parameter
sets in parallel into outputs/iter/rounds/<round>/<config>/ and tabulates the main metrics.

    python -m r2s.sweep4d rNN "gb_k_bed=0.05,mem_k=10" "gb_k_bed=0.2" ... [--jobs 6] [tissue=version ...]
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from .config import OUTPUTS, ROOT

ITER = OUTPUTS / 'iter'


def row(m):
    e = m['eval']
    g, mb = e.get('gallbladder', {}), e.get('membrane', {})
    gs, ms = g.get('split', {}), mb.get('split', {})
    f = lambda d, k: (d.get(k) or {}).get('mean')
    return dict(gb_iou=f(g, 'sim_iou'), gb_rigid_mm=f(gs, 'rigid_err'), gb_rot_sim=f(gs, 'rot_sim'), gb_rot_rec=f(gs, 'rot_rec'),
                gb_nonrigid_mm=f(gs, 'nonrigid_err'), gb_track_motion=(g.get('track_motion') or {}).get('change_pct'),
                gb_free_mexp=g.get('free_motion_explained'), gb_free_err=f(g, 'free_err_vs_recon_mm'),
                gb_inverted=g.get('inverted_tets_max'), photo=(m.get('photometric') or {}).get('ncc', {}).get('mean'),
                gb_push=[(g.get('probe_push') or {}).get('sim_mm'), (g.get('probe_push') or {}).get('recon_mm')],
                gb_flow=[f(g, 'flow_epe_px'), f(g, 'flow_epe_static_px'), f(g, 'flow_epe_recon_px')],
                mem_flow=[f(mb, 'flow_epe_px'), f(mb, 'flow_epe_static_px'), f(mb, 'flow_epe_recon_px')],
                mem_iou=f(mb, 'sim_iou'), mem_rigid_mm=f(ms, 'rigid_err'), mem_nonrigid_mm=f(ms, 'nonrigid_err'),
                mem_stretch=mb.get('stretch_max'), duct_iou=f(e.get('ducts', {}), 'sim_iou'),
                duct_free_mexp={k: v.get('free_motion_explained') for k, v in e.get('ducts', {}).items()
                                if isinstance(v, dict) and 'free_motion_explained' in v})


def main(argv):
    rnd = argv[0]
    jobs = 6
    configs, fixed = [], []
    i = 1
    while i < len(argv):
        if argv[i] == '--jobs':
            jobs = int(argv[i + 1])
            i += 2
            continue
        (configs if ',' in argv[i] or argv[i].split('=')[0] not in ('gallbladder', 'membrane', 'ducts', 'backdrop', 'interaction')
         else fixed).append(argv[i])
        i += 1

    def run(cfg):
        name = cfg.replace(',', '_') or 'default'
        args = [f'{rnd}/{name}'] + fixed + [kv for kv in cfg.split(',') if kv]
        with open(ITER / 'rounds' / f'{rnd}_{name}.log', 'w') as f:
            p = subprocess.run([sys.executable, '-m', 'r2s.scene4d'] + args, stdout=f, stderr=subprocess.STDOUT,
                               cwd=OUTPUTS / 'logs', env=dict(__import__('os').environ, PYTHONPATH=str(ROOT)))
        return name, p.returncode
    (ITER / 'rounds' / rnd).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(run, configs))
    table = {}
    for name, rc in results:
        mp = ITER / 'rounds' / rnd / name / 'metrics.json'
        table[name] = row(json.loads(mp.read_text())) if rc == 0 and mp.exists() else dict(failed=rc)
        # keep the per-config log next to its outputs
        lp = ITER / 'rounds' / f'{rnd}_{name}.log'
        if lp.exists() and (ITER / 'rounds' / rnd / name).exists():
            lp.rename(ITER / 'rounds' / rnd / name / 'run.log')
    (ITER / 'rounds' / rnd / 'sweep.json').write_text(json.dumps(table, indent=1))
    for name, r in table.items():
        print(f'{name:40s}', json.dumps(r))


if __name__ == '__main__':
    main(sys.argv[1:])
