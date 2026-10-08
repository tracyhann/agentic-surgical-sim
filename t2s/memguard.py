"""Memory watchdog for long v2 jobs: stops this project's python jobs before the machine runs out of RAM.

    python -m t2s.memguard [min_free_gb=6] [interval_s=10]      (run in the background while heavy jobs run)

Every interval it reads vm_stat; if free + inactive + speculative memory falls below min_free_gb it sends SIGTERM to
the largest python process whose command line mentions this worktree's t2s / r2s modules, and logs to
outputs/t2s/memguard.log. It never touches other processes.
"""
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'outputs' / 't2s' / 'memguard.log'


def free_gb():
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    page = int(re.search(r'page size of (\d+)', out).group(1))
    get = lambda k: int(re.search(rf'{k}:\s+(\d+)', out).group(1))
    return (get('Pages free') + get('Pages inactive') + get('Pages speculative')) * page / 1e9


def ours():
    out = subprocess.run(['ps', '-axo', 'pid=,rss=,command='], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        pid, rss, cmd = line.strip().split(None, 2)
        if 'python' in cmd and ('t2s.' in cmd or 'r2s' in cmd or 't2s/' in cmd) and 'memguard' not in cmd:
            rows.append((int(rss) / 1e6, int(pid), cmd[:120]))
    return sorted(rows, reverse=True)


def main(min_free=6.0, interval=10.0):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        f = free_gb()
        if f < min_free:
            jobs = ours()
            with open(LOG, 'a') as fh:
                fh.write(f'{time.strftime("%H:%M:%S")} free {f:.1f} GB < {min_free}: jobs {[(round(g, 1), p) for g, p, _ in jobs]}\n')
                if jobs:
                    g, pid, cmd = jobs[0]
                    os.kill(pid, signal.SIGTERM)
                    fh.write(f'  SIGTERM {pid} ({g:.1f} GB) {cmd}\n')
            time.sleep(20)
        time.sleep(interval)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(float(a[0]) if a else 6.0, float(a[1]) if len(a) > 1 else 10.0)
