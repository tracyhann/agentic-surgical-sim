"""Stage a blind task workspace and run one fresh Claude Code session on it (Video2World-style driver).

  python launch.py <instance> [--minutes 45] [--model claude-opus-5-5] [--dry-run]
The agent sees only runs/<inst>/task (video, brief, tools, instrument) and writes runs/<inst>/out.
"""
import argparse, json, os, shutil, signal, subprocess, time
from pathlib import Path
import imageio.v2 as imageio

PRIV = Path(__file__).resolve().parent
from make_reference import PUB  # noqa: E402

HINTS = {
    'bead_cup': 'the grasper moves one object into a receiver.',
    'peg_transfer': 'FLS-style peg transfer: an object is moved from one peg onto another peg.',
}
LEAK_MARKERS = ('surg_private', 'scratchpad', '/gt/', 'rollouts_ref', 'annotation.json', 'make_reference')


def stage(inst, minutes):
    run = PUB / 'runs' / inst
    task, out, cwd = run / 'task', run / 'out', run / 'cwd'
    scratch = Path('/tmp') / f'surg_{inst}'
    for d in (out, cwd, scratch, task / 'tools', task / 'instrument'):
        if d.exists():
            shutil.rmtree(d)
    (task / 'tools').mkdir(parents=True)
    shutil.copy(PUB / 'tools' / 'surgsim.py', task / 'tools' / 'surgsim.py')
    shutil.copytree(PUB / 'instrument', task / 'instrument', ignore=shutil.ignore_patterns('__pycache__'))
    (out / 'source').mkdir(parents=True)
    shutil.copytree(PUB / 'instrument', out / 'instrument', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy(task / 'video.mp4', out / 'source' / 'video.mp4')
    cwd.mkdir()
    scratch.mkdir()
    rd = imageio.get_reader(task / 'video.mp4')
    T = rd.count_frames()
    H, W = rd.get_data(0).shape[:2]
    brief = (PRIV / 'brief_template.md').read_text().format(
        video=task / 'video.mp4', W=W, H=H, T=T, fps=20, out=out, tools=task / 'tools', py=PUB / '.venv' / 'bin' / 'python',
        name=f'surg_{inst}', minutes=minutes, task_hint=HINTS[inst])
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
           '--verbose', '--model', a.model, '--permission-mode', 'acceptEdits',
           '--allowedTools', 'Bash,Read,Edit,Write,Glob,Grep',
           '--add-dir', str(task), '--add-dir', str(out), '--add-dir', str(scratch)]
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDECODE') and k != 'CLAUDE_CODE_ENTRYPOINT'}
    log_path = run / 'agent_log.jsonl'
    t0 = time.time()
    rec = dict(instance=a.instance, model=a.model, minutes=a.minutes, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(log_path, 'w') as log:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rec['exit'] = p.wait(timeout=a.minutes * 60)
            rec['timeout'] = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(5)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            rec['exit'], rec['timeout'] = None, True
    rec['seconds'] = round(time.time() - t0)
    text = log_path.read_text(errors='replace')
    cost, turns = None, None
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get('type') == 'result':
            cost, turns = ev.get('total_cost_usd'), ev.get('num_turns')
    rec.update(cost_usd=cost, num_turns=turns, leaks=[m for m in LEAK_MARKERS if m in text])
    (run / 'agent_run.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
