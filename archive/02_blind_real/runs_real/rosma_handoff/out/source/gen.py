import numpy as np, json, sys
from scipy.optimize import least_squares
exec(open('calib2.py').read().split("for fov in")[0])
OUT=sys.argv[1]; fov=35.0
s=least_squares(mk(fov),[0.0,-0.4,0.2,0,0,0.15,0.0,0.045]); q=s.x
pos=q[:3]; look=np.array([q[3],q[4],0.]); D=q[5]; tilt=q[6]; hp=q[7]; Rb=Ry(tilt); n=Rb@[0,0,1.]
R=axes(pos,look); f=H_/2/np.tan(np.radians(fov)/2)
def ray(px): return np.array([(px[0]-W_/2)/f,(px[1]-H_/2)/f,1.])@R
def onboard(px):
    d=ray(px); return Rb.T@(pos-(pos@n)/(d@n)*d)
def onplane_y(px,y):
    d=ray(px); return pos+(y-pos[1])/d[1]*d
bases={'L1':(303,570),'L2':(325,520),'L3':(343,478),'L4':(358,445),'M1':(447,512),'M2':(450,465),
 'U1':(566,505),'U2':(562,465),'R1':(700,543),'R2':(682,500),'R3':(672,462),'R4':(662,432)}
B={k:onboard(v) for k,v in bases.items()}
for k,v in B.items(): print(k,v.round(4))
print('cam',pos,look,'D',D,'tilt',tilt,'hp',hp)
yrow=(Rb@B['R1'])[1]
print('handoff px(610,262) @front row',onplane_y((610,262),yrow), ' left home tip',onplane_y((390,262),yrow),' right tip f240',onplane_y((655,340),yrow))
PH=0.043; SH=0.030; ai,ao=0.0070,0.0100
posts=''.join(f'<body name="post_{k}" pos="{v[0]:.4f} {v[1]:.4f} 0"><geom name="post_{k}_g" type="cylinder" fromto="0 0 0 0 0 {PH}" size="0.0035" rgba="0.95 0.95 0.95 1" friction="0.3 0.005 0.0001"/></body>\n' for k,v in B.items())
def ring(rgba,coll=True,mass=0.0015):
    o=''
    for i in range(8):
        a=i*np.pi/4; r=(ai+ao)/2
        o+=f'<geom type="box" pos="{r*np.cos(a):.5f} {r*np.sin(a):.5f} 0" euler="0 0 {a:.5f}" size="{(ao-ai)/2:.5f} {ao*np.tan(np.pi/8):.5f} {SH/2}" rgba="{rgba}" '+(f'mass="{mass/8}" friction="0.6 0.005 0.0001"' if coll else 'contype="0" conaffinity="0" mass="0"')+'/>\n'
    return o
deco=''
for k,c,z in (('U1','0.5 0.25 0.7 1',0),('U2','0.5 0.25 0.7 1',0),('R2','0.95 0.2 0.6 1',0),('R3','0.95 0.2 0.6 1',0),('R4','0.9 0.85 0.1 1',0)):
    deco+=f'<body name="deco_{k}" pos="{B[k][0]:.4f} {B[k][1]:.4f} {SH/2+z}">'+ring(c,False)+'</body>\n'
sl=Rb@(B['R1']+np.array([0,-0.0025,SH/2+0.0003]))
qb=[np.cos(tilt/2),0,np.sin(tilt/2),0]
rcmR=[0.17,round(float(sl[1]-(ai+ao)/2),4),0.20]; rcmL=[-0.17,round(float(yrow-0.028),4),0.20]
xml=f'''<mujoco model="rosma_handoff">
<compiler angle="radian"/>
<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81"/>
<visual><headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/></visual>
<worldbody>
<light pos="0 -0.3 0.6" dir="0 0.3 -1"/>
<geom name="table" type="plane" pos="0 0 -0.05" size="1 1 0.01" rgba="0.25 0.2 0.15 1"/>
<body name="board" pos="0 0 0" euler="0 {tilt:.5f} 0">
<geom name="board_g" type="box" pos="0 0 -0.004" size="0.1 {D/2:.4f} 0.004" rgba="0.92 0.92 0.95 1" friction="0.8 0.005 0.0001"/>
<geom name="base_g" type="box" pos="0 0 -0.03" size="0.085 {D/2-0.01:.4f} 0.022" rgba="0.5 0.5 0.5 0.5" contype="0" conaffinity="0"/>
{posts}{deco}</body>
<body name="sleeve_yellow" pos="{sl[0]:.5f} {sl[1]:.5f} {sl[2]:.5f}" quat="{qb[0]:.6f} 0 {qb[2]:.6f} 0"><freejoint/>
{ring('0.95 0.9 0.1 1')}</body>
<body name="psm_left_trocar" pos="{rcmL[0]} {rcmL[1]} {rcmL[2]}" euler="0 0 {-np.pi/2:.6f}"><include file="instrument/psm_left_body.xml"/></body>
<body name="psm_right_trocar" pos="{rcmR[0]} {rcmR[1]} {rcmR[2]}" euler="0 0 {np.pi/2:.6f}"><include file="instrument/psm_right_body.xml"/></body>
</worldbody>
<actuator><include file="instrument/psm_left_actuators.xml"/><include file="instrument/psm_right_actuators.xml"/></actuator>
<contact><include file="instrument/psm_left_contacts.xml"/><include file="instrument/psm_right_contacts.xml"/></contact>
</mujoco>'''
open(OUT+'/scene.xml','w').write(xml)
proto={"protocol_version":"surg-0.2","status":"success",
 "task":{"instruction":"psm_right grasps the yellow sleeve on the front-right post by its rim, lifts it off, carries it to the middle above the board, hands it to psm_left, which carries it to the front-left post and drops it over that post.","source_video":"source/video.mp4"},
 "model_path":"scene.xml",
 "camera":{"pos":[round(float(v),4) for v in pos],"lookat":[round(float(v),4) for v in look],"fovy_deg":fov,"width":1024,"height":768},
 "instruments":[{"name":"psm_left","rcm_pos":rcmL,"heading":round(-np.pi/2,6)},{"name":"psm_right","rcm_pos":rcmR,"heading":round(np.pi/2,6)}],
 "roles":{"object":["sleeve_yellow"],"target":["post_L1"]},
 "actions":{"path":"actions.npy","dt":0.05,"format":"joint_targets"},"policy":{"path":"policy.py"}}
json.dump(proto,open(OUT+'/protocol.json','w'),indent=1)
