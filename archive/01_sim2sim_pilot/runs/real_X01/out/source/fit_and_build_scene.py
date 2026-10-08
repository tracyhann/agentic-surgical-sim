import numpy as np, json
from scipy.optimize import least_squares
F=1200.0
def cam_axes(pos,look):
    f=look-pos; f/=np.linalg.norm(f); r=np.cross(f,[0,0,1.]); r/=np.linalg.norm(r); u=np.cross(r,f); return f,r,u
def proj(pos,look,p):
    f,r,u=cam_axes(pos,look); d=np.asarray(p)-pos
    return np.array([512+F*d@r/(d@f), 384-F*d@u/(d@f)])
def unp(x):
    global F
    F=x[8]; x=list(x); x[6]=x[5]
    return np.array(x[0:3]),np.array([x[3],x[4],0.]),x[5],x[6],x[7]
obs_left=[(312,562),(332,518),(350,478),(366,445)]
def res(x):
    pos,look,dx,dy,H=unp(x); r=[]
    for k,o in enumerate(obs_left):
        r+=list(proj(pos,look,[-1.5*dx,(-1.5+k)*dy,0])-o)
    r+=list(proj(pos,look,[1.5*dx,-1.5*dy,0])-(695,545))
    r+=list(proj(pos,look,[-1.5*dx,-1.5*dy,H])-(300,442))
    f,rr,u=cam_axes(pos,look); c=np.array([1.5*dx,-1.5*dy,0.0125])
    w=proj(pos,look,c+0.01*rr)[0]-proj(pos,look,c-0.01*rr)[0]
    r.append((w-67)*3)
    return r
s=least_squares(res,[0,-0.4,0.15,0,0,0.037,0.031,0.038,1000.])
pos,look,dx,dy,H=unp(s.x)
print('F',F,'fovy',2*np.degrees(np.arctan(384/F)));print('cam',pos,look,'dx dy H',dx,dy,H,'rms',np.sqrt(np.mean(s.fun**2)))
f,r,u=cam_axes(pos,look)
def back(px,Z):
    d=f+(px[0]-512)/F*r-(px[1]-384)/F*u; return pos+d*Z
def back_plane(px,z=0):
    d=f+(px[0]-512)/F*r-(px[1]-384)/F*u; t=(z-pos[2])/d[2]; return pos+d*t
Lr=back((-290,210),0.24); Rr=back((1155,130),0.52)
print('rcm',Lr,Rr)
corners=[back_plane(c) for c in [(147,640),(868,598),(752,388)]]
print('corners',corners)
L0=back((402,275),0.30); R0=back((735,290),0.44); print('tips0',L0,R0, 'handover', back((510,290),0.37))
json.dump(dict(F=F,pos=pos.tolist(),look=look.tolist(),dx=dx,dy=dy,H=H,L=Lr.tolist(),R=Rr.tolist(),corners=[c.tolist() for c in corners],L0=L0.tolist(),R0=R0.tolist()),open('fit.json','w'))
Lr=back((-290,210),0.12); Rr=back((1155,130),0.225)
print('rcm',Lr,Rr)
OUT='/tmp/surgrun_X01/out/'
RO,RI,HS,RP=0.010,0.0065,0.025,0.003
N=16
def sleeve_geoms(rgba,prefix,col=True):
    g=''; rm=(RO+RI)/2; hw=rm*np.tan(np.pi/N)*1.08
    for k in range(N):
        a=2*np.pi*k/N
        g+=f'<geom name="{prefix}{k}" type="box" pos="{rm*np.cos(a):.6f} {rm*np.sin(a):.6f} 0" euler="0 0 {a:.6f}" size="{(RO-RI)/2:.5f} {hw:.5f} {HS/2}" rgba="{rgba}" '+('friction="1.0 0.02 0.0002" condim="4" density="170"' if col else 'contype="0" conaffinity="0"')+'/>\n'
    return g
grid=lambda i,j:(( -1.5+i)*dx,(-1.5+j)*dx)
posts=''
decor=''
allp=[(0,j) for j in range(4)]+[(1,1),(1,2),(2,1),(2,2)]+[(3,j) for j in range(4)]
for (i,j) in allp:
    name='post_source' if (i,j)==(3,0) else 'post_target' if (i,j)==(0,0) else f'post_{i}{j}'
    x,y=grid(i,j)
    posts+=f'<body name="{name}" pos="{x:.5f} {y:.5f} 0"><geom name="{name}_g" type="cylinder" fromto="0 0 0 0 0 {H:.4f}" size="{RP}" rgba="0.95 0.95 0.95 1"/></body>\n'
for (i,j),c in {(2,1):'0.45 0.25 0.7 1',(2,2):'0.45 0.25 0.7 1',(3,1):'0.95 0.2 0.6 1',(3,2):'0.95 0.2 0.6 1',(3,3):'0.9 0.85 0.1 1'}.items():
    x,y=grid(i,j)
    decor+=f'<geom type="cylinder" pos="{x:.5f} {y:.5f} {HS/2}" size="{RO} {HS/2}" rgba="{c}" contype="0" conaffinity="0"/>\n'
sx,sy=grid(3,0)
xml=f'''<mujoco model="post_and_sleeve">
<compiler angle="radian"/>
<option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
<visual><global offwidth="1024" offheight="768"/><headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/></visual>
<worldbody>
<light pos="0 -0.2 0.5" dir="0 0.3 -1"/>
<geom name="table" type="box" pos="0 0.05 -0.03" size="0.4 0.4 0.004" rgba="0.25 0.22 0.2 1"/>
<body name="board" pos="0.001 -0.001 0"><geom name="board_g" type="box" pos="0 0 -0.0025" size="0.089 0.089 0.0025" rgba="0.92 0.92 0.92 1"/>
<geom type="box" pos="0 0 -0.0155" size="0.07 0.07 0.0105" rgba="0.6 0.65 0.7 0.5" contype="0" conaffinity="0"/>
{decor}</body>
{posts}
<body name="sleeve" pos="{sx:.5f} {sy:.5f} {HS/2+0.0002}"><freejoint/>
{sleeve_geoms('0.93 0.88 0.12 1','sl')}</body>
<body name="L_trocar" pos="{Lr[0]:.5f} {Lr[1]:.5f} {Lr[2]:.5f}"><include file="instrument_psm/L_body.xml"/></body>
<body name="R_trocar" pos="{Rr[0]:.5f} {Rr[1]:.5f} {Rr[2]:.5f}"><include file="instrument_psm/R_body.xml"/></body>
</worldbody>
<actuator><include file="instrument_psm/L_actuators.xml"/><include file="instrument_psm/R_actuators.xml"/></actuator>
<contact><include file="instrument_psm/L_contacts.xml"/><include file="instrument_psm/R_contacts.xml"/></contact>
</mujoco>'''
open(OUT+'scene.xml','w').write(xml)
proto={"protocol_version":"surg-0.2","status":"success",
"task":{"instruction":"The right instrument grasps the wall of the yellow sleeve on the front-right post, lifts it off, carries it to mid-air above the front of the board and hands it to the left instrument, which carries it to the front-left post and lowers it over that post.","source_video":"source/video.mp4"},
"model_path":"scene.xml",
"camera":{"pos":[round(v,5) for v in pos],"lookat":[round(v,5) for v in look],"fovy_deg":round(float(2*np.degrees(np.arctan(384/F))),3),"width":1024,"height":768},
"instrument":{"L_rcm_pos":[round(v,5) for v in Lr],"R_rcm_pos":[round(v,5) for v in Rr]},
"roles":{"object":["sleeve"],"source":["post_source"],"target":["post_target"]},
"actions":{"path":"actions.npy","dt":0.05,"format":"tcp_pose_2arm"},
"policy":{"path":"policy.py"}}
json.dump(proto,open(OUT+'protocol.json','w'),indent=1)
