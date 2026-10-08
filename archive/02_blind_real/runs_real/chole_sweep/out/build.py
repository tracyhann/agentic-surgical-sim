"""Generates scene.xml, policy.py constants, actions.npy, protocol.json from pixel measurements."""
import json, sys, numpy as np, mujoco
sys.path.insert(0, 'instrument'); import kinematics as K
CAM = dict(pos=[0, -0.09, 0.06], lookat=[0, 0, 0], fovy_deg=60.0, width=640, height=360)
C = np.array(CAM['pos']); f = np.array(CAM['lookat']) - C; f /= np.linalg.norm(f)
r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
F = 180 / np.tan(np.radians(30))
def unproj(px, py, Z): return C + Z * (f + (px - 320) / F * r - (py - 180) / F * u)
v3 = lambda a: ' '.join(f'{x:.5f}' for x in a)

# --- measurements (pixels, depth from shaft width: 5 mm ~ 15 px -> Z ~ 0.10 m at fovy 60)
ZP = 0.100
g_tip0 = unproj(238, 125, 0.100); g_top = unproj(172, 0, 0.055)
gd = (g_tip0 - g_top) / np.linalg.norm(g_tip0 - g_top)
G_RCM = g_tip0 - 0.12 * gd; G_H = float(np.arctan2(-gd[0], gd[1]))
p_tip0 = unproj(279, 224, ZP); pdir = -(0.86 * r + 0.50 * u) ; pdir /= np.linalg.norm(pdir)
P_RCM = p_tip0 - 0.20 * pdir; P_H = float(np.arctan2(-pdir[0], pdir[1]))
n = np.cross(gd, [1, 0, 0]); n = n if n[2] > 0 else -n; n /= np.linalg.norm(n)   # neck axis: perpendicular to grasper shaft
body_c = unproj(207, 263, 0.090); RB = 0.024
floor_z = body_c[2] - RB
anchor = body_c + np.array([0, 0, -RB])
n1a = body_c + 0.014 * (g_tip0 - 0.016 * n - body_c) / np.linalg.norm(g_tip0 - 0.016 * n - body_c)
n1b = g_tip0 - 0.016 * n
s_root = unproj(470, 238, 0.103); s_end = unproj(262, 150, 0.100); NS = 6
sseg = (s_end - s_root) / NS

def chain(i):
    if i == NS: return ''
    pos = s_root if i == 0 else sseg
    return f'''<body name="strand{i}" pos="{v3(pos)}">
      <joint type="ball" stiffness="{0.1 if i==0 else 0.03}" damping="0.003"/>
      <geom type="capsule" fromto="0 0 0 {v3(sseg)}" size="{0.0045-0.0003*i:.4f}" mass="0.0006" rgba="0.42 0.18 0.38 1" friction="0.8"/>
      {chain(i+1)}</body>'''

xml = f'''<mujoco model="chole_sweep">
  <compiler angle="radian"/>
  <option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  
  <worldbody>
    <light pos="0 -0.1 0.2" dir="0 0.5 -1"/>
    <geom name="bed" type="plane" pos="0 0 {floor_z:.5f}" size="0.3 0.3 0.01" rgba="0.75 0.42 0.42 1"/>
    <geom name="liver" type="box" pos="0 0.11 0.02" euler="-0.6 0 0" size="0.25 0.005 0.15" rgba="0.55 0.3 0.33 1"/>
    <geom name="duodenum" type="sphere" pos="{v3(unproj(450, 270, 0.112))}" size="0.012" rgba="0.6 0.25 0.25 1"/>
    <body name="gb_body" pos="{v3(body_c)}">
      <joint name="gb_pivot" type="ball" pos="{v3(anchor - body_c)}" stiffness="0.03" damping="0.004"/>
      <geom name="gb_body_geom" type="ellipsoid" size="{RB} {RB} {RB}" mass="0.02" rgba="0.36 0.33 0.18 1" friction="0.8"/>
      <body name="gb_neck1" pos="{v3(n1a - body_c)}">
        <joint name="gb_stretch" type="slide" axis="{v3((n1b - n1a) / np.linalg.norm(n1b - n1a))}" range="-0.004 0.03" stiffness="4" damping="0.3"/>
        <joint type="ball" stiffness="0.01" damping="0.001"/>
        <geom type="capsule" fromto="0 0 0 {v3(n1b - n1a)}" size="0.006" mass="0.002" rgba="0.38 0.35 0.2 1"/>
        <body name="gb_neck2" pos="{v3(n1b - n1a)}">
          <joint type="ball" stiffness="0.004" damping="0.0005"/>
          <geom type="capsule" fromto="0 0 0 {v3(0.010 * n)}" size="0.0035" mass="0.001" rgba="0.4 0.36 0.22 1"/>
          <body name="gb_neck3" pos="{v3(0.016 * n)}">
            <joint type="ball" stiffness="0.002" damping="0.0003"/>
            <geom type="capsule" fromto="{v3(-0.006 * n)} {v3(0.0065 * n)}" size="0.002" mass="0.0005" rgba="0.42 0.38 0.25 1" friction="1.5"/>
            <geom type="sphere" pos="{v3(0.0068 * n)}" size="0.0042" mass="0.0005" rgba="0.42 0.38 0.25 1" friction="1.5"/>
          </body>
        </body>
      </body>
    </body>
    {chain(0)}
    <body name="grasper_left_trocar" pos="{v3(G_RCM)}" euler="0 0 {G_H:.5f}"><include file="instrument/grasper_left_body.xml"/></body>
    <body name="probe_right_trocar" pos="{v3(P_RCM)}" euler="0 0 {P_H:.5f}"><include file="instrument/probe_right_body.xml"/></body>
  </worldbody>
  <actuator><include file="instrument/grasper_left_actuators.xml"/><include file="instrument/probe_right_actuators.xml"/></actuator>
  <contact><include file="instrument/grasper_left_contacts.xml"/><include file="instrument/probe_right_contacts.xml"/></contact>
</mujoco>'''
open('scene.xml', 'w').write(xml)

# --- tracks (frame, px, py)
GT = [(0,238,125),(30,232,122),(50,222,105),(60,215,95),(90,205,70),(105,222,100),(120,232,110),(150,234,106)]
PT = [(0,279,224),(5,290,254),(10,295,270),(15,283,263),(20,274,258),(25,289,228),(30,265,195),(35,264,183),(40,285,175),
      (45,273,141),(50,282,150),(55,360,80),(60,450,10),(66,450,10),(70,427,21),(75,326,93),(80,259,145),(85,253,141),
      (90,257,107),(95,249,114),(100,286,168),(105,296,172),(110,311,168),(115,312,180),(120,326,180),(130,328,177),
      (135,334,177),(140,329,164),(150,323,165)]
const = dict(G_RCM=G_RCM.tolist(), G_H=G_H, P_RCM=P_RCM.tolist(), P_H=P_H, NECK0=g_tip0.tolist(), BODY0=body_c.tolist(),
             STRAND0=s_root.tolist(),
             GT=[[k] + (unproj(x, y, 0.100) - g_tip0).tolist() for k, x, y in GT],
             PT=[[k] + unproj(x, y, 0.094 if 5 <= k <= 20 else ZP).tolist() for k, x, y in PT])
json.dump(const, open('policy_const.json', 'w'))
proto = dict(protocol_version='surg-0.2', status='success',
  task=dict(instruction='Laparoscopic cholecystectomy: the left grasper holds the gallbladder neck and retracts it up/left while the closed straight instrument from the upper right pokes the gallbladder flank and sweeps the tissue beside the neck (Calot region) upward, withdraws, re-enters and sweeps again.', source_video='source/video.mp4'),
  model_path='scene.xml', camera=CAM,
  instruments=[dict(name='grasper_left', rcm_pos=[round(x, 5) for x in G_RCM], heading=round(G_H, 5)),
               dict(name='probe_right', rcm_pos=[round(x, 5) for x in P_RCM], heading=round(P_H, 5))],
  roles=dict(object=['gb_body', 'gb_neck1', 'gb_neck2', 'gb_neck3'] + [f'strand{i}' for i in range(NS)], target=[]),
  actions=dict(path='actions.npy', dt=0.05, format='joint_targets'), policy=dict(path='policy.py'))
json.dump(proto, open('protocol.json', 'w'), indent=1)
from policy import plan
m = mujoco.MjModel.from_xml_path('scene.xml'); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
a = plan(m, d); np.save('actions.npy', a.astype(np.float64)); print(a.shape, a[0])
