"""Stage a blind real-video task workspace and run one fresh Claude Code session on it.

  python launch_real.py <instance> [--minutes 45] [--model claude-opus-5-5] [--dry-run]
The agent sees only runs_real/<inst>/task (video, brief, tools, instrument) and writes runs_real/<inst>/out.
"""
import argparse, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import imageio.v2 as imageio

PRIV = Path(__file__).resolve().parent
ROOT = PRIV.parent
BLIND = Path('/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad/blind')
LEAK_MARKERS = ('private_real', '/gt/', 'gt.npz', 'annotation.json', 'prepare_', 'data/real', '/phai/medical-agentic-sim')

SOFT_RULES = """- Tissue may be modelled with flex/flexcomp, composite, or rigid bodies joined by joints/tendons. It may be
  attached to static anatomy (worldbody or static bodies) with equality constraints, tendons or flex pins.
- Nothing may couple tissue to an instrument except contact: no equality constraint or tendon may touch an
  instrument body, and there are no object actuators or state overrides.
- `roles.object` names the tissue that the instruments manipulate (a body or a flex name)."""
RIGID_RULES = """- Manipulated object(s): worldbody children with exactly one `<freejoint/>`, mass 0.1-50 g, geoms that collide.
  They may move only through contact with the instruments and gravity: no equality constraints, tendons, flex,
  attachments, object actuators or state overrides.
- Receiver/target: a static body (no joints)."""

INSTANCES = {
    'rosma_handoff': dict(
        config=dict(profile='rigid', instruments=[dict(name='psm_left', shaft_d=0.008), dict(name='psm_right', shaft_d=0.008)], fps=15),
        scene_hint='a webcam in front of a da Vinci Research Kit; two robotic instruments work on a skills board with '
                   'posts and coloured sleeves (the robot has no endoscope; this webcam is the only camera).',
        task_hint='one sleeve is transferred from one post to another post using both instruments; reproduce that '
                  'transfer (the other sleeves stay where they are and are optional in your scene). `psm_left` is the '
                  'instrument entering from the image left, `psm_right` the one from the right.',
        object_hint='"<the transferred sleeve>"', target_hint='"<its destination post>"'),
    'chole_sweep': dict(
        config=dict(profile='soft', instruments=[dict(name='grasper_left', shaft_d=0.005), dict(name='probe_right', shaft_d=0.005)], fps=25),
        scene_hint='laparoscopic cholecystectomy (gallbladder removal) in a patient; the laparoscope is static in this clip.',
        task_hint='a grasper from the upper left holds the gallbladder neck while a straight instrument from the upper '
                  'right sweeps the tissue beside it. Reproduce both instrument motions and the response of the '
                  'gallbladder/tissue they touch. `grasper_left` is the upper-left instrument, `probe_right` the '
                  'straight upper-right one (keep its jaws closed if it acts as a probe).',
        object_hint='"<the tissue body or flex>"', target_hint=''),
}


def stage(inst, minutes):
    I = INSTANCES[inst]
    run = BLIND / inst                      # outside the project tree: the agent cannot stumble on the GT
    task, out, cwd = run / 'task', run / 'out', run / 'cwd'
    task.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / 'runs_real' / inst / 'task' / 'video.mp4', task / 'video.mp4')
    scratch = Path('/tmp') / f'surgreal_{inst}'
    for d in (out, cwd, scratch, task / 'tools'):
        if d.exists():
            shutil.rmtree(d)
    (task / 'tools').mkdir(parents=True)
    shutil.copy(ROOT / 'tools' / 'surgsim.py', task / 'tools' / 'surgsim.py')
    rd = imageio.get_reader(task / 'video.mp4')
    T, fps = rd.count_frames(), I['config']['fps']
    H, W = rd.get_data(0).shape[:2]
    cfg = dict(I['config'], width=W, height=H, frames=T)
    (task / 'tools' / 'task_config.json').write_text(json.dumps(cfg, indent=1))
    sys.path.insert(0, str(task / 'tools'))
    import surgsim
    (out / 'source').mkdir(parents=True)
    surgsim.write_instruments(out / 'instrument')
    shutil.copy(ROOT / 'tools' / 'kinematics.py', out / 'instrument' / 'kinematics.py')
    names = [i['name'] for i in cfg['instruments']]
    shafts = ', '.join(f"{i['name']} {i['shaft_d'] * 1000:.0f} mm" for i in cfg['instruments'])
    (out / 'instrument' / 'README.md').write_text((ROOT / 'tools' / 'instrument_README.md').read_text()
                                                  .format(instrument_list=', '.join(names), shaft_list=shafts))
    shutil.copy(task / 'video.mp4', out / 'source' / 'video.mp4')
    cwd.mkdir()
    scratch.mkdir()
    ins_json = ', '.join(f'{{"name": "{n}", "rcm_pos": [x, y, z], "heading": h}}' for n in names)
    brief = (PRIV / 'brief_real.md').read_text().format(
        video=task / 'video.mp4', W=W, H=H, T=T, fps=fps, out=out, tools=task / 'tools', py=BLIND / '.venv' / 'bin' / 'python',
        name=f'surgreal_{inst}', minutes=minutes, scene_hint=I['scene_hint'], task_hint=I['task_hint'],
        instrument_list=', '.join(names), shaft_list=shafts, duration=T / fps, profile=cfg['profile'],
        profile_rules=SOFT_RULES if cfg['profile'] == 'soft' else RIGID_RULES, ncols=5 * len(names),
        instrument_json=ins_json, object_hint=I['object_hint'], target_hint=I['target_hint'])
    (task / 'brief.md').write_text(brief)
    return run, task, out, cwd, scratch, brief


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('instance')
    ap.add_argument('--minutes', type=int, default=45)
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    run, task, out, cwd, scratch, brief = stage(a.instance, a.minutes)
    if a.dry_run:
        print(brief)
        return
    cmd = [shutil.which('claude') or str(Path.home() / '.local/bin/claude'), '-p', brief, '--output-format', 'stream-json',
           '--verbose', '--model', a.model, '--permission-mode', 'acceptEdits', '--allowedTools', 'Bash,Read,Edit,Write,Glob,Grep',
           '--add-dir', str(task), '--add-dir', str(out), '--add-dir', str(scratch)]
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDECODE') and k != 'CLAUDE_CODE_ENTRYPOINT'}
    log_path = run / 'agent_log.jsonl'
    t0 = time.time()
    rec = dict(instance=a.instance, model=a.model, minutes=a.minutes, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(log_path, 'w') as log:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rec['exit'], rec['timeout'] = p.wait(timeout=a.minutes * 60), False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(5)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            rec['exit'], rec['timeout'] = None, True
    rec['seconds'] = round(time.time() - t0)
    text = log_path.read_text(errors='replace')
    results = []
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get('type') == 'result':
            results.append(dict(cost=ev.get('total_cost_usd'), turns=ev.get('num_turns'), is_error=ev.get('is_error'),
                                result=str(ev.get('result'))[:300]))
    rec.update(results=results, cost_usd=max([r['cost'] or 0 for r in results], default=None),
               leaks=[m for m in LEAK_MARKERS if m in text])
    (run / 'agent_run.json').write_text(json.dumps(rec, indent=1))
    dst = ROOT / 'runs_real' / a.instance
    for sub in ('task', 'out'):
        if (dst / sub).exists():
            shutil.rmtree(dst / sub)
        shutil.copytree(run / sub, dst / sub, ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('agent_log.jsonl', 'agent_run.json'):
        shutil.copy(run / f, dst / f)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
