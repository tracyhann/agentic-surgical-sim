"""Feasibility check for the two-arm PSM contract: a hand-built post-and-sleeve scene (plausible dimensions, not
measured) and a scripted lift -> mid-air handoff -> place. Writes a valid surgsim2 package and renders it."""
import json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent / 'feasibility'
sys.path.insert(0, str(ROOT / 'tools'))
import surgsim2 as S2  # noqa: E402

SL = dict(r_in=0.0055, t=0.0025, h=0.018)          # sleeve: hole radius, wall, height
POST = dict(r=0.00175, h=0.025)
SRC, DST = (0.04, -0.015), (-0.04, -0.015)


def scene():
    posts = ''
    for i, x in enumerate((-0.065, -0.04, -0.015, 0.015, 0.04, 0.065)):
        for j, y in enumerate((-0.015, 0.015)):
            name = 'post_src' if (x, y) == SRC else 'post_dst' if (x, y) == DST else f'post_{i}{j}'
            posts += (f'    <body name="{name}" pos="{x} {y} 0"><geom type="cylinder" size="{POST["r"]} {POST["h"]/2}" '
                      f'pos="0 0 {POST["h"]/2}" rgba="0.95 0.95 0.95 1"/></body>\n')
    n, side = 12, 2 * (SL['r_in'] + SL['t']) * np.tan(np.pi / 12)
    walls = ''.join(
        f'      <geom type="box" size="{SL["t"]/2:.5f} {side/2:.5f} {SL["h"]/2:.5f}" '
        f'pos="{(SL["r_in"]+SL["t"]/2)*np.cos(a):.5f} {(SL["r_in"]+SL["t"]/2)*np.sin(a):.5f} {SL["h"]/2:.5f}" '
        f'euler="0 0 {a:.5f}" density="300" rgba="0.95 0.85 0.1 1" friction="1.2 0.02 0.0002" condim="4"/>\n'
        for a in np.arange(n) * 2 * np.pi / n)
    return f"""<mujoco model="post_and_sleeve_feasibility">
  <compiler angle="radian"/>
  <option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <worldbody>
    <light pos="0 -0.1 0.4" dir="0 0.2 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="board" type="box" size="0.085 0.05 0.003" pos="0 0 -0.003" rgba="0.92 0.92 0.92 1"/>
{posts}    <body name="sleeve" pos="{SRC[0]} {SRC[1]} 0.0002">
      <freejoint/>
{walls}    </body>
    <body name="L_trocar" pos="-0.2 0.06 0.09"><include file="instrument_psm/L_body.xml"/></body>
    <body name="R_trocar" pos="0.2 0.06 0.09"><include file="instrument_psm/R_body.xml"/></body>
  </worldbody>
  <actuator><include file="instrument_psm/L_actuators.xml"/><include file="instrument_psm/R_actuators.xml"/></actuator>
  <contact><include file="instrument_psm/L_contacts.xml"/><include file="instrument_psm/R_contacts.xml"/></contact>
</mujoco>
"""


POLICY = '''"""Scripted lift -> handoff -> place, reading the sleeve and post positions from the simulator state."""
import numpy as np, mujoco

R_MID = {r_mid}          # sleeve wall mid-radius
TIP_BELOW_RIM = 0.006    # how far the jaw tips reach below the rim
DT = 0.05


def quat(x_axis, y_axis):
    x = np.asarray(x_axis, float); y = np.asarray(y_axis, float)
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], 1).flatten())
    return q


def plan(model, data):
    c = data.body('sleeve').xipos.copy()
    rim = c[2] + {h} / 2
    dst = data.body('post_dst').xpos.copy()
    post_top = {post_h}
    down = [0, 0, -1]
    qR, qL = quat([1, 0, 0], down), quat([-1, 0, 0], down)
    grip_z = rim - TIP_BELOW_RIM
    lift = post_top + {h} - TIP_BELOW_RIM + 0.008          # sleeve bottom clears the post top
    H = np.array([0.0, 0.0, lift + 0.01])                  # hand-off point (sleeve centre)
    L0, R0 = np.array([-0.06, 0.01, 0.07]), np.array([0.07, 0.01, 0.06])
    OPEN, SHUT = 0.6, -0.1
    keys = []                                               # (duration s, L pos, L jaw, R pos, R jaw)
    gR = c + [R_MID, 0, 0]; gR[2] = grip_z
    keys += [(0.6, L0, OPEN, R0, OPEN),
             (2.0, L0, OPEN, gR + [0, 0, 0.02], OPEN), (1.0, L0, OPEN, gR, OPEN), (0.6, L0, OPEN, gR, SHUT),
             (0.4, L0, OPEN, gR, SHUT), (1.5, L0, OPEN, [gR[0], gR[1], lift], SHUT)]
    hR = H + [R_MID, 0, -0.01 - ({h} / 2 - TIP_BELOW_RIM) + {h} / 2]
    hR = np.array([H[0] + R_MID, H[1], H[2] + {h} / 2 - TIP_BELOW_RIM])
    hL = np.array([H[0] - R_MID, H[1], hR[2]])
    keys += [(2.0, L0, OPEN, hR, SHUT), (2.0, hL + [0, 0, 0.02], OPEN, hR, SHUT), (1.0, hL, OPEN, hR, SHUT),
             (0.6, hL, SHUT, hR, SHUT), (0.4, hL, SHUT, hR, SHUT), (0.6, hL, SHUT, hR, OPEN),
             (1.0, hL, SHUT, hR + [0.01, 0, 0.02], OPEN)]
    pL = np.array([dst[0] - R_MID, dst[1], hL[2]])
    downL = np.array([pL[0], pL[1], 0.002 + {h} - TIP_BELOW_RIM])
    keys += [(2.5, pL, SHUT, R0, OPEN), (2.0, downL, SHUT, R0, OPEN), (0.4, downL, SHUT, R0, OPEN),
             (0.6, downL, OPEN, R0, OPEN), (1.5, downL + [0, 0, 0.03], OPEN, R0, OPEN), (1.0, L0, OPEN, R0, OPEN)]
    rows, prev = [], None
    for dur, Lp, Lj, Rp, Rj in keys:
        cur = (np.asarray(Lp, float), Lj, np.asarray(Rp, float), Rj)
        if prev is None:
            prev = cur
        n = max(1, int(round(dur / DT)))
        for k in range(1, n + 1):
            w = k / n
            Lx = prev[0] + (cur[0] - prev[0]) * w; Rx = prev[2] + (cur[2] - prev[2]) * w
            Lw = prev[1] + (cur[1] - prev[1]) * w; Rw = prev[3] + (cur[3] - prev[3]) * w
            rows.append(np.concatenate([Lx, qL, [Lw], Rx, qR, [Rw]]))
        prev = cur
    return np.array(rows)
'''


def build(cam=None):
    PKG.mkdir(exist_ok=True)
    (PKG / 'scene.xml').write_text(scene())
    (PKG / 'policy.py').write_text(POLICY.format(r_mid=SL['r_in'] + SL['t'] / 2, h=SL['h'], post_h=POST['h']))
    cam = cam or dict(pos=[0.0, -0.17, 0.10], lookat=[0.0, 0.0, 0.012], fovy_deg=50.0, width=1024, height=768)
    proto = dict(protocol_version='surg-0.2', status='success', task=dict(instruction='feasibility', source_video='source/video.mp4'),
                 model_path='scene.xml', camera=cam, instrument=dict(L_rcm_pos=[-0.2, 0.06, 0.09], R_rcm_pos=[0.2, 0.06, 0.09]),
                 roles=dict(object=['sleeve'], source=['post_src'], target=['post_dst']),
                 actions=dict(path='actions.npy', dt=0.05, format='tcp_pose_2arm'), policy=dict(path='policy.py'))
    (PKG / 'protocol.json').write_text(json.dumps(proto, indent=1))
    (PKG / 'report.md').write_text('feasibility package\n')
    _, _, model, data = S2.S1.build(PKG)
    acts = np.array([[-0.06, 0.01, 0.07, 1, 0, 0, 0, 0.9, 0.07, 0.01, 0.06, 1, 0, 0, 0, 0.9]])
    q, jaw, _ = S2.joint_targets(model, acts)
    S2.reset(model, data, q[0], jaw[0])
    acts = S2.S1.load_policy(PKG)(model, data)
    np.save(PKG / 'actions.npy', acts)
    return model, data, acts


if __name__ == '__main__':
    import imageio.v2 as imageio
    model, data, acts = build()
    print('rows', len(acts), 'action check:', S2.check_actions(model, acts) or 'OK')
    proto = S2.S1.load_protocol(PKG)
    rec = S2.execute(model, data, acts, ['sleeve'], render=S2.S1.make_renderer(model, proto['camera'], vig=False))
    imageio.mimsave(PKG / 'source' / 'video.mp4', rec['frames'], fps=20, macro_block_size=1)
    p = rec['pose']['sleeve'][:, :3]
    held = np.array([c[0] for c in rec['contacts']])
    for k in range(0, len(p), 20):
        print(f"t={rec['t'][k]:5.2f} sleeve {np.round(p[k]*1000,1)} held L={bool(held[k]&1)} R={bool(held[k]&2)}")
    print('final', np.round(p[-1] * 1000, 1), 'dst', DST)
    for qq in (0.2, 0.45, 0.6, 0.8, 1.0):
        imageio.imwrite(PKG.parent / f'feas_{int(qq*100)}.png', rec['frames'][int(qq * (len(rec['frames']) - 1))])
    print('validate:', S2.validate(PKG) or 'OK')
