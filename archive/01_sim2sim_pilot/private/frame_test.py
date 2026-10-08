"""Evaluator invariance check: rigidly move the whole GT world (rotate about z, translate) and rescore."""
import json, re, shutil, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, '.')
import evaluate as E
S = E.S
inst, th = sys.argv[1], np.radians(float(sys.argv[2]))
t = np.array([0.11, -0.04, 0.0])
Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
qR = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
tf = lambda p: (Rz @ np.asarray(p, float) + t).tolist()
src = E.PRIV / 'gt' / inst / 'package'
dst = E.PRIV / 'frame_test' / inst
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
spec = mujoco.MjSpec.from_file(str(src / 'scene.xml'))
for b in spec.worldbody.bodies:
    if b.name == 'trocar':
        b.pos = tf(b.pos); continue
    b.pos = tf(b.pos); q = np.zeros(4); mujoco.mju_mulQuat(q, qR, np.asarray(b.quat)); b.quat = q.tolist()
for g in spec.worldbody.geoms:
    g.pos = tf(g.pos); q = np.zeros(4); mujoco.mju_mulQuat(q, qR, np.asarray(g.quat)); g.quat = q.tolist()
(dst / 'scene.xml').write_text(spec.to_xml())
proto = json.loads((src / 'protocol.json').read_text())
proto['camera']['pos'] = tf(proto['camera']['pos']); proto['camera']['lookat'] = tf(proto['camera']['lookat'])
proto['instrument']['rcm_pos'] = tf(proto['instrument']['rcm_pos'])
(dst / 'protocol.json').write_text(json.dumps(proto, indent=1))
pol = (src / 'policy.py').read_text()
pol = re.sub(r'RCM = np.array\(\[.*?\]\)', f"RCM = np.array({proto['instrument']['rcm_pos']})", pol)
hov = re.search(r'HOVER = np.array\((\[.*?\])\)', pol).group(1)
pol = pol.replace(f'HOVER = np.array({hov})', f'HOVER = np.array({tf(json.loads(hov))})')
(dst / 'policy.py').write_text(pol)
_, _, model, data = S.build(dst)
S.reset(model, data, np.load(src / 'actions.npy')[0])
np.save(dst / 'actions.npy', S.load_policy(dst)(model, data))
r = E.score(inst, dst)
print(json.dumps({k: r.get(k) for k in ['build', 'validate_errors', 'success', 'progress', 'geometry', 'dynamics', 'v2w_score']}))
