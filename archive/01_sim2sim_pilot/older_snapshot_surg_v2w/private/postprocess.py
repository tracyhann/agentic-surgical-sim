"""After the blind runs: score each agent package, run the layout-perturbation data engine, collect agent stats.

  python postprocess.py  ->  results/results.json, results/<inst>/rollouts/, results/<inst>/eval_compare.mp4
"""
import json, shutil, sys
from pathlib import Path

PRIV = Path(__file__).resolve().parent
sys.path.insert(0, str(PRIV))
import evaluate as E  # noqa: E402
import rollouts as RO  # noqa: E402
from make_reference import PUB  # noqa: E402

OUT = PRIV / 'results'


def agent_stats(run):
    rec = json.loads((run / 'agent_run.json').read_text()) if (run / 'agent_run.json').exists() else {}
    n_tools, n_valid, n_replay, first_valid = 0, 0, 0, None
    for line in (run / 'agent_log.jsonl').read_text(errors='replace').splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get('type') == 'assistant':
            for c in ev['message'].get('content', []):
                if c.get('type') == 'tool_use':
                    n_tools += 1
                    cmd = str(c['input'].get('command', ''))
                    n_valid += 'surgsim.py validate' in cmd
                    n_replay += 'surgsim.py replay' in cmd
    rec.update(tool_calls=n_tools, validate_calls=n_valid, replay_calls=n_replay)
    return rec


def main(insts=('bead_cup', 'peg_transfer')):
    OUT.mkdir(exist_ok=True)
    allres = {}
    for inst in insts:
        run = PUB / 'runs' / inst
        pkg = run / 'out'
        res = dict(agent=agent_stats(run))
        res['score'] = E.score(inst, pkg, render=True)
        d = OUT / inst
        d.mkdir(exist_ok=True)
        if (run / f'eval_compare_{inst}.mp4').exists():
            shutil.move(run / f'eval_compare_{inst}.mp4', d / 'eval_compare.mp4')
        if res['score']['build']:
            summ = RO.main(inst, pkg, d / 'rollouts')
            res['rollouts'] = {k: v for k, v in summ.items() if k != 'rows'}
        ref = json.loads((PRIV / 'rollouts_ref' / inst / 'summary.json').read_text())
        res['reference_rollouts'] = {k: v for k, v in ref.items() if k != 'rows'}
        allres[inst] = res
        print(inst, json.dumps(res, indent=1, default=str))
    (OUT / 'results.json').write_text(json.dumps(allres, indent=1, default=str))


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or ('bead_cup', 'peg_transfer'))
