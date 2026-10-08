"""Build the hidden reference packages (GT) and render the source videos for the blind agent.

  python make_reference.py [instance ...]
Writes  surg_private/gt/<inst>/package/   a valid surgsim package (scene, scripted policy, actions, GT camera)
        surg_private/gt/<inst>/annotation.json, traj.npz
        <PUB>/runs/<inst>/task/video.mp4
"""
import json, shutil, sys, textwrap
from pathlib import Path
import numpy as np

PRIV = Path(__file__).resolve().parent
PUB = PRIV.parent
sys.path.insert(0, str(PUB / 'tools'))
sys.path.insert(0, str(PUB / 'instrument'))
import surgsim_v1 as S  # noqa: E402

W, H = 640, 480

HEADER = """<mujoco model="{name}">
  <compiler angle="radian"/>
  <option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><quality shadowsize="4096"/></visual>
  <asset>
    <texture name="tissue" type="2d" builtin="gradient" rgb1="{c1}" rgb2="{c2}" width="256" height="256"/>
    <material name="tissue" texture="tissue" texrepeat="3 3" specular="0.6" shininess="0.6" reflectance="0"/>
  </asset>
  <worldbody>
"""
FOOTER = """    <body name="trocar" pos="{rcm}">
      <include file="instrument/instrument_body.xml"/>
    </body>
  </worldbody>
  <actuator><include file="instrument/instrument_actuators.xml"/></actuator>
  <contact><include file="instrument/instrument_contacts.xml"/></contact>
</mujoco>
"""


def v(x):
    return ' '.join(f'{c:.5g}' for c in x)


def ring_geoms(n, r_in, t, h, rgba, name, extra=''):
    """n-gon wall: boxes whose inner faces sit at apothem r_in."""
    side = 2 * (r_in + t) * np.tan(np.pi / n)
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n
        c = (r_in + t / 2) * np.array([np.cos(a), np.sin(a)])
        out.append(f'<geom name="{name}{k}" type="box" size="{t/2:.5g} {side/2:.5g} {h/2:.5g}" pos="{c[0]:.5g} {c[1]:.5g} {h/2:.5g}" '
                   f'euler="0 0 {a:.5g}" rgba="{rgba}" {extra}/>')
    return '\n        '.join(out)


INSTANCES = {
    'bead_cup': dict(
        task='Pick up the blue bead with the grasper and drop it into the white cup.',
        rcm=[0.075, -0.07, 0.085],
        cam=dict(pos=[-0.005, -0.085, 0.075], lookat=[0.003, 0.012, 0.0], fovy_deg=65.0, width=W, height=H),
        tissue=('0.82 0.45 0.45', '0.70 0.33 0.36'),
        objects={'bead': [-0.018, 0.004]}, target='cup', target_pos=[0.022, 0.024],
        dims=dict(bead_r=0.004, bead_hh=0.003, cup_rin=0.0095, cup_t=0.0015, cup_h=0.008),
        hover=[0.035, -0.012, 0.032],
    ),
    'peg_transfer': dict(
        task='FLS peg transfer: lift the orange ring off its peg and place it over the destination peg.',
        rcm=[-0.07, -0.065, 0.085],
        cam=dict(pos=[0.01, -0.09, 0.08], lookat=[0.0, 0.016, 0.0], fovy_deg=60.0, width=W, height=H),
        tissue=('0.40 0.42 0.47', '0.30 0.31 0.36'),
        pegs={'peg1': [-0.022, 0.0], 'peg2': [-0.022, 0.026], 'peg3': [0.0, 0.036], 'peg4': [0.022, 0.026], 'peg5': [0.022, 0.0]},
        objects={'ring': [-0.022, 0.0]}, target='peg4',
        dims=dict(ring_rin=0.008, ring_t=0.002, ring_h=0.005, peg_r=0.0018, peg_h=0.018),
        hover=[-0.025, -0.012, 0.035],
    ),
}


def scene_xml(name, I):
    d = I['dims']
    s = HEADER.format(name=name, c1=I['tissue'][0], c2=I['tissue'][1])
    if name == 'bead_cup':
        s += f"""    <geom name="pad" type="box" size="0.08 0.07 0.005" pos="0 0.01 -0.005" material="tissue"/>
    <body name="bead" pos="{v(I['objects']['bead'] + [d['bead_hh'] + 0.0002])}">
      <freejoint/>
      <geom name="bead_geom" type="cylinder" size="{d['bead_r']} {d['bead_hh']}" density="1200" rgba="0.15 0.35 0.85 1"
            friction="1.2 0.02 0.0002" condim="4"/>
    </body>
    <body name="bead_green" pos="-0.03 0.032 0.0032">
      <freejoint/>
      <geom name="bead_green_geom" type="cylinder" size="0.004 0.003" density="1200" rgba="0.2 0.7 0.3 1" condim="4"/>
    </body>
    <body name="cup" pos="{v(I['target_pos'] + [0])}">
        {ring_geoms(16, d['cup_rin'], d['cup_t'], d['cup_h'], '0.93 0.93 0.88 1', 'cup_wall')}
    </body>
"""
    else:
        s += """    <geom name="board" type="box" size="0.06 0.05 0.003" pos="0 0.014 -0.003" material="tissue"/>
"""
        for pn, p in I['pegs'].items():
            s += f"""    <body name="{pn}" pos="{v(p + [0])}">
      <geom name="{pn}_geom" type="cylinder" size="{d['peg_r']} {d['peg_h']/2}" pos="0 0 {d['peg_h']/2}" rgba="0.92 0.92 0.92 1"/>
    </body>
"""
        s += f"""    <body name="ring" pos="{v(I['objects']['ring'] + [0.0002])}">
      <freejoint/>
        {ring_geoms(8, d['ring_rin'], d['ring_t'], d['ring_h'], '0.95 0.45 0.1 1', 'ring_wall', 'density="1000" friction="1.2 0.02 0.0002" condim="4"')}
    </body>
"""
    return s + FOOTER.format(rcm=v(I['rcm']))


POLICY = '''"""Reference (scripted) policy: reads the initial object/target poses from the simulator state."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'instrument'))
from kinematics import ik, jaw_axis, JAW_OPEN

RCM = np.array({rcm})
TASK = {task!r}
HOVER = np.array({hover})
DT = 0.05


def _seg(p0, p1, speed):
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / (speed * DT))))
    return [p0 + (p1 - p0) * (k + 1) / n for k in range(n)]


def plan(model, data):
    obj = data.body({obj!r}).xipos.copy()
    tgt = data.body({tgt!r}).xpos.copy()
    if TASK == 'bead_cup':
        grasp = obj + np.array([0, 0, 0.0008])
        place = np.array([tgt[0], tgt[1], 0.016])
    else:
        yaw = ik(RCM, obj)[0]
        u = jaw_axis(yaw, 0.0, 0.0)
        grasp = np.array([obj[0], obj[1], 0.0040]) + 0.009 * u
        place = np.array([tgt[0], tgt[1], 0.0105])
        for _ in range(5):                      # the ring centre sits 9 mm from the TCP along the jaw axis
            yaw = ik(RCM, place)[0]
            place[:2] = tgt[:2] + 0.009 * jaw_axis(yaw, 0.0, 0.0)[:2]
    lift_z = 0.030 if TASK != 'bead_cup' else 0.026
    OPEN = JAW_OPEN if TASK == 'bead_cup' else 0.003
    tcp, jaw = [HOVER.copy()], [OPEN]
    def go(p, speed, j):
        for q in _seg(tcp[-1], np.asarray(p, float), speed):
            tcp.append(q); jaw.append(j)
    def hold(n, j0, j1):
        for k in range(n):
            tcp.append(tcp[-1].copy()); jaw.append(j0 + (j1 - j0) * (k + 1) / n)
    hold(6, OPEN, OPEN)
    go(grasp + [0, 0, 0.014], 0.025, OPEN)
    go(grasp, 0.010, OPEN)
    hold(10, OPEN, 0.0)
    hold(6, 0.0, 0.0)
    go([grasp[0], grasp[1], lift_z], 0.012, 0.0)
    go([place[0], place[1], lift_z], 0.022, 0.0)
    go(place, 0.008, 0.0)
    hold(4, 0.0, 0.0)
    hold(8, 0.0, OPEN)
    go(place + [0, 0, 0.015], 0.015, OPEN)
    go(HOVER, 0.025, OPEN)
    out = []
    for p, j in zip(tcp, jaw):
        y, pi, ins = ik(RCM, p)
        out.append([y, pi, ins, 0.0, j])
    return np.array(out)
'''


def build_instance(name):
    import imageio.v2 as imageio
    I = INSTANCES[name]
    root = PRIV / 'gt' / name
    pkg = root / 'package'
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / 'source').mkdir(parents=True)
    shutil.copytree(PUB / 'instrument', pkg / 'instrument')
    (pkg / 'scene.xml').write_text(scene_xml(name, I))
    obj = list(I['objects'])[0]
    (pkg / 'policy.py').write_text(POLICY.format(rcm=I['rcm'], task=name, hover=I['hover'], obj=obj, tgt=I['target']))
    proto = dict(protocol_version='surg-0.1', status='success', task=dict(instruction=I['task'], source_video='source/video.mp4'),
                 model_path='scene.xml', camera=I['cam'], instrument=dict(rcm_pos=I['rcm']),
                 roles=dict(object=[obj], target=[I['target']]),
                 actions=dict(path='actions.npy', dt=S.DT, format='joint_targets'), policy=dict(path='policy.py'))
    (pkg / 'protocol.json').write_text(json.dumps(proto, indent=1))
    (pkg / 'report.md').write_text('Hidden reference package (scripted reference policy with privileged state).\n')
    _, _, model, data = S.build(pkg)
    S.reset(model, data, np.zeros(5) + [0, 0.5, 0.1, 0, 0.007])
    plan = S.load_policy(pkg)
    acts = plan(model, data)
    np.save(pkg / 'actions.npy', acts)
    rec = S.execute(model, data, acts, [obj, I['target']], render=S.make_renderer(model, I['cam']))
    imageio.mimsave(pkg / 'source' / 'video.mp4', rec['frames'], fps=int(1 / S.DT), macro_block_size=1, quality=8)
    task_dir = PUB / 'runs' / name / 'task'
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pkg / 'source' / 'video.mp4', task_dir / 'video.mp4')
    for q in (0, 0.5, 1.0):
        imageio.imwrite(root / f'frame_{int(q*100):03d}.png', rec['frames'][int(q * (len(rec['frames']) - 1))])
    errs = S.validate(pkg)
    p = rec['pose'][obj][:, :3]
    np.savez(root / 'traj.npz', t=rec['t'], obj=rec['pose'][obj], tcp=rec['tcp'], actions=acts)
    ann = dict(instance=name, camera=I['cam'], rcm=I['rcm'], dims=I['dims'], target=I['target'], object=obj,
               frames=len(rec['frames']), validate_errors=errs)
    (root / 'annotation.json').write_text(json.dumps(ann, indent=1))
    print(name, 'frames', len(rec['frames']), 'actions', len(acts), 'validate', errs or 'OK',
          'obj start', p[0].round(4), 'max z', p[:, 2].max().round(4), 'end', p[-1].round(4))


if __name__ == '__main__':
    for n in sys.argv[1:] or INSTANCES:
        build_instance(n)
