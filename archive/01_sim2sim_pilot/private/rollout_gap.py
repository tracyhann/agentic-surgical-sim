"""Matched-layout gap between agent-world rollouts and reference-world rollouts (camera frame)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
import evaluate as E
S = E.S
from make_reference import PUB
out = {}
for inst in ('bead_cup', 'peg_transfer'):
    cam_a = S.load_protocol(PUB / 'runs' / inst / 'out')['camera']
    cam_g = S.load_protocol(E.PRIV / 'gt' / inst / 'package')['camera']
    rows = []
    for f in sorted((PUB / 'results' / inst / 'rollouts').glob('*.npz')):
        g = PUB / 'results' / 'rollouts_reference' / inst / f.name
        a, b = np.load(f), np.load(g)
        pa = np.array([S.world_to_cam(cam_a, p) for p in a['obj_pose'][:, :3]])
        pg = np.array([S.world_to_cam(cam_g, p) for p in b['obj_pose'][:, :3]])
        sa, sg = E.segment(pa), E.segment(pg)
        da, dg = E.resample(sa - sa[0]), E.resample(sg - sg[0])
        rows.append(dict(layout=f.stem, motion_err_mm=float(np.linalg.norm(da - dg, axis=1).mean() * 1000),
                         final_err_mm=float(np.linalg.norm(pa[-1] - pg[-1]) * 1000),
                         dur_agent_s=float(a['t'][-1]), dur_ref_s=float(b['t'][-1])))
    m = lambda k: float(np.median([r[k] for r in rows]))
    out[inst] = dict(n=len(rows), median_motion_err_mm=round(m('motion_err_mm'), 2), median_final_err_mm=round(m('final_err_mm'), 2),
                     median_duration_agent_s=round(m('dur_agent_s'), 2), median_duration_ref_s=round(m('dur_ref_s'), 2))
print(json.dumps(out, indent=1))
(PUB / 'results' / 'rollout_gap.json').write_text(json.dumps(out, indent=1))
