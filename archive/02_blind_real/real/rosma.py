"""ROSMA loading: kinematics (50 Hz, per-arm RCM frames) aligned to video frames via the on-screen clock."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'rosma'
FPS = 15.0
# video frame index of the first on-screen clock tick and the wall time it shows (read from the frame)
CLOCK = {'X01': dict(tick_frame=10, tick_time='2020-02-04.17:02:37.000')}


def _t(s):
    # the logger writes milliseconds without zero padding ('17:02:37.22' = 37.022 s)
    base, ms = s.rsplit('.', 1)
    return datetime.strptime(base, '%Y-%m-%d.%H:%M:%S').timestamp() + int(ms) / 1000.0


def _monotonic(t):
    # right after a second rollover the seconds field can still show the previous second
    t = t.copy()
    for i in range(1, len(t)):
        while t[i] < t[i - 1] - 0.5:
            t[i] += 1.0
    return t


def load_kin(trial='X01', task='Post_and_Sleeve', rep='01'):
    rows = list(csv.reader(open(DATA / f'{trial}_{task}_{rep}.csv')))
    hdr = rows[0]
    body = [r for r in rows[1:] if len(r) == len(hdr) and not r[0].startswith('Date:')]
    idx = {h: i for i, h in enumerate(hdr)}
    t = _monotonic(np.array([_t(r[0]) for r in body]))
    out = dict(t_wall=t)
    for arm in ('PSM1', 'PSM2'):
        cols = [f'{arm}_position_{a}' for a in 'xyz'] + [f'{arm}_orientation_{a}' for a in 'xyzw']
        cols += [f'{arm}_joint_position_{k}' for k in range(1, 7)]
        out[arm] = np.array([[float(r[idx[c]]) for c in cols] for r in body])   # pos(3) quat xyzw(4) joints(6)
    c = CLOCK[trial]
    out['t_video'] = (t - _t(c['tick_time'])) + c['tick_frame'] / FPS             # seconds into the video
    return out


def at_video_time(kin, arm, tv):
    """Linear interpolation of an arm's position (3) at video times tv."""
    tv = np.atleast_1d(tv)
    return np.stack([np.interp(tv, kin['t_video'], kin[arm][:, k]) for k in range(3)], 1)
