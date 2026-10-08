"""Write the two-arm dVRK-style PSM includes (L_*, R_*). Run once; the XML files are the public contract."""
from pathlib import Path

D = Path(__file__).resolve().parent
BODY = """<mujoco>
  <!-- dVRK-style PSM ({P}): include INSIDE <body name="{P}trocar" pos="x y z"> (world-aligned; pos = RCM). -->
  <body name="{P}yaw_link" gravcomp="1">
    <joint name="{P}yaw" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.02" armature="0.001"/>
    <inertial pos="0 0 0" mass="0.02" diaginertia="2e-6 2e-6 2e-6"/>
    <body name="{P}pitch_link" gravcomp="1">
      <joint name="{P}pitch" type="hinge" axis="-1 0 0" range="-0.8 1.45" damping="0.02" armature="0.001"/>
      <inertial pos="0 0 0" mass="0.02" diaginertia="2e-6 2e-6 2e-6"/>
      <body name="{P}shaft" gravcomp="1">
        <joint name="{P}insertion" type="slide" axis="0 1 0" range="0 0.30" damping="2" armature="0.02"/>
        <geom name="{P}shaft_geom" type="cylinder" fromto="0 -0.40 0 0 -0.002 0" size="0.0042" mass="0.06"
              rgba="0.10 0.10 0.12 1" contype="2" conaffinity="1"/>
        <body name="{P}roll_link" gravcomp="1">
          <joint name="{P}roll" type="hinge" axis="0 1 0" range="-4.5 4.5" damping="0.005" armature="0.0005"/>
          <geom name="{P}collar_geom" type="cylinder" fromto="0 -0.002 0 0 0.001 0" size="0.0044" mass="0.003"
                rgba="0.70 0.70 0.74 1" contype="2" conaffinity="1"/>
          <body name="{P}wrist_pitch_link" gravcomp="1">
            <joint name="{P}wrist_pitch" type="hinge" axis="1 0 0" range="-1.57 1.57" damping="0.002" armature="0.0002"/>
            <geom name="{P}clevis_geom" type="box" pos="0 0.0045 0" size="0.0028 0.0045 0.0030" mass="0.002"
                  rgba="0.72 0.72 0.76 1" contype="2" conaffinity="1"/>
            <body name="{P}wrist_yaw_link" pos="0 0.0091 0" gravcomp="1">
              <joint name="{P}wrist_yaw" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.002" armature="0.0002"/>
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-8 1e-8 1e-8"/>
              <site name="{P}tcp" pos="0 0.0102 0" size="0.0008" rgba="1 1 0 0"/>
              <body name="{P}jaw_a" gravcomp="1">
                <joint name="{P}jaw_a" type="hinge" axis="0 0 -1" range="-0.05 0.7" damping="0.002" armature="0.0002"/>
                <geom name="{P}jaw_a_geom" type="box" pos="0.0009 0.0052 0" size="0.0008 0.0050 0.0014" mass="0.001"
                      rgba="0.80 0.80 0.84 1" friction="1.5 0.02 0.0002" condim="4" priority="1" solref="0.001 1"
                      solimp="0.95 0.99 0.0005" contype="2" conaffinity="1"/>
              </body>
              <body name="{P}jaw_b" gravcomp="1">
                <joint name="{P}jaw_b" type="hinge" axis="0 0 1" range="-0.05 0.7" damping="0.002" armature="0.0002"/>
                <geom name="{P}jaw_b_geom" type="box" pos="-0.0009 0.0052 0" size="0.0008 0.0050 0.0014" mass="0.001"
                      rgba="0.80 0.80 0.84 1" friction="1.5 0.02 0.0002" condim="4" priority="1" solref="0.001 1"
                      solimp="0.95 0.99 0.0005" contype="2" conaffinity="1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </body>
</mujoco>
"""
ACT = """<mujoco>
  <!-- {P} servos; include INSIDE <actuator>. jaw_a/jaw_b are driven together with half the jaw angle each. -->
  <position name="{P}yaw" joint="{P}yaw" kp="8" kv="0.3" ctrlrange="-3.14 3.14" forcerange="-2 2"/>
  <position name="{P}pitch" joint="{P}pitch" kp="8" kv="0.3" ctrlrange="-0.8 1.45" forcerange="-2 2"/>
  <position name="{P}insertion" joint="{P}insertion" kp="400" kv="12" ctrlrange="0 0.30" forcerange="-6 6"/>
  <position name="{P}roll" joint="{P}roll" kp="0.6" kv="0.02" ctrlrange="-4.5 4.5" forcerange="-0.3 0.3"/>
  <position name="{P}wrist_pitch" joint="{P}wrist_pitch" kp="0.4" kv="0.01" ctrlrange="-1.57 1.57" forcerange="-0.1 0.1"/>
  <position name="{P}wrist_yaw" joint="{P}wrist_yaw" kp="0.4" kv="0.01" ctrlrange="-1.57 1.57" forcerange="-0.1 0.1"/>
  <position name="{P}jaw_a" joint="{P}jaw_a" kp="0.05" kv="0.001" ctrlrange="-0.05 0.7" forcerange="-0.0015 0.0015"/>
  <position name="{P}jaw_b" joint="{P}jaw_b" kp="0.05" kv="0.001" ctrlrange="-0.05 0.7" forcerange="-0.0015 0.0015"/>
</mujoco>
"""
CON = """<mujoco>
  <exclude body1="{P}jaw_a" body2="{P}jaw_b"/>
  <exclude body1="{P}jaw_a" body2="{P}wrist_yaw_link"/>
  <exclude body1="{P}jaw_b" body2="{P}wrist_yaw_link"/>
  <exclude body1="{P}jaw_a" body2="{P}wrist_pitch_link"/>
  <exclude body1="{P}jaw_b" body2="{P}wrist_pitch_link"/>
</mujoco>
"""
for P in ('L_', 'R_'):
    (D / f'{P}body.xml').write_text(BODY.format(P=P))
    (D / f'{P}actuators.xml').write_text(ACT.format(P=P))
    (D / f'{P}contacts.xml').write_text(CON.format(P=P))
print('wrote', sorted(p.name for p in D.glob('*.xml')))
