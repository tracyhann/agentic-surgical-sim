"""Stage an isolated workspace under /tmp for a real-video clip and run one blind Claude Code session on it.

  python launch_real.py X01 --t0 5 --t1 26 [--minutes 45] [--model claude-opus-5-5] [--dry-run]
Nothing in the brief points into this project: the agent gets its own venv, tools and instrument copies.
Afterwards the package and logs are copied to runs/real_<trial>/.
"""
import argparse, json, os, shutil, signal, subprocess, time
from pathlib import Path
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
HINT = ('the instrument on the image right picks the yellow sleeve off its post, hands it over in mid-air to the '
        'instrument on the image left, which places it over another post.')
LEAK = ('phai', 'medical-agentic-sim', 'rosma', 'calib', 'annotations', 'sleeve_track', 'private', 'Post_and_Sleeve')


def stage(trial, t0, t1, minutes):
    ws = Path(f'/tmp/surgrun_{trial}')
    name = f'surg_real_{trial}'
    scratch = Path('/tmp') / name
    for d in (ws, scratch):
        if d.exists():
            shutil.rmtree(d)
    task, out, cwd = ws / 'task', ws / 'out', ws / 'cwd'
    for d in (task / 'tools', out / 'source', cwd, scratch):
        d.mkdir(parents=True)
    for f in ('surgsim_v1.py', 'surgsim2.py'):
        shutil.copy(ROOT / 'tools' / f, task / 'tools' / f)
    ign = shutil.ignore_patterns('generate.py', '__pycache__')
    shutil.copytree(ROOT / 'instrument_psm', task / 'instrument_psm', ignore=ign)
    shutil.copytree(ROOT / 'instrument_psm', out / 'instrument_psm', ignore=ign)
    r = imageio.get_reader(ROOT / 'data' / 'rosma' / f'{trial}_Post_and_Sleeve_01.mp4')
    frames = [r.get_data(n) for n in range(int(round(t0 * 15)), int(round(t1 * 15)) + 1)]
    imageio.mimsave(task / 'video.mp4', frames, fps=15, quality=9, macro_block_size=1)
    shutil.copy(task / 'video.mp4', out / 'source' / 'video.mp4')
    subprocess.run(['uv', 'venv', '-q', '--python', '3.11', str(ws / '.venv')], check=True)
    subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(ws / '.venv' / 'bin' / 'python'), '-r',
                    str(ROOT / 'requirements.txt')], check=True)
    H, W = frames[0].shape[:2]
    brief = (ROOT / 'real' / 'brief_real.md').read_text().format(
        video=task / 'video.mp4', W=W, H=H, T=len(frames), fps=15, out=out, tools=task / 'tools',
        py=ws / '.venv' / 'bin' / 'python', name=name, minutes=minutes, task_hint=HINT)
    (task / 'brief.md').write_text(brief)
    return ws, task, out, cwd, scratch, brief


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trial')
    ap.add_argument('--t0', type=float, default=5.0)
    ap.add_argument('--t1', type=float, default=26.0)
    ap.add_argument('--minutes', type=int, default=45)
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    ws, task, out, cwd, scratch, brief = stage(a.trial, a.t0, a.t1, a.minutes)
    if a.dry_run:
        print(brief)
        return
    cmd = [shutil.which('claude') or str(Path.home() / '.local/bin/claude'), '-p', brief, '--output-format', 'stream-json',
           '--verbose', '--model', a.model, '--permission-mode', 'acceptEdits', '--allowedTools', 'Bash,Read,Edit,Write,Glob,Grep',
           '--add-dir', str(task), '--add-dir', str(out), '--add-dir', str(scratch)]
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDECODE') and k != 'CLAUDE_CODE_ENTRYPOINT'}
    log_path = ws / 'agent_log.jsonl'
    rec = dict(trial=a.trial, clip=[a.t0, a.t1], model=a.model, minutes=a.minutes, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    t0 = time.time()
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
    cost = None
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get('type') == 'result' and ev.get('total_cost_usd') is not None:
            cost = ev['total_cost_usd']
    rec.update(cost_usd=cost, leaks=[m for m in LEAK if m in text])
    dst = ROOT / 'runs' / f'real_{a.trial}'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(out, dst / 'out', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(task, dst / 'task', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy(log_path, dst / 'agent_log.jsonl')
    (dst / 'agent_run.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
