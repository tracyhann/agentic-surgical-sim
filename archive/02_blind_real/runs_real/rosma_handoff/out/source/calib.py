import numpy as np, json
from scipy.optimize import least_squares
W_,H_=1024,768
def axes(pos,look):
    z=look-pos; z/=np.linalg.norm(z); x=np.cross(z,[0,0,1.]); x/=np.linalg.norm(x); return np.stack([x,np.cross(z,x),z])
def proj(pos,look,fov,p):
    c=(np.asarray(p,float)-pos)@axes(pos,look).T; f=H_/2/np.tan(np.radians(fov)/2)
    return np.stack([f*c[...,0]/c[...,2]+W_/2,f*c[...,1]/c[...,2]+H_/2],-1)
corn=np.array([[145,645],[868,592],[750,390],[285,425]],float)  # FL FR BR BL
# vertical post: front-right base/top, front-left base/top
posts=np.array([[700,543,700,408],[303,570,296,440]],float)
fov=45.0
def unpack(q):
    pos=q[:3]; look=np.array([q[3],q[4],0.]); return pos,look,q[5],q[6]
def res(q):
    pos,look,D,hp=unpack(q); W=0.2
    P=np.array([[-W/2,-D/2,0],[W/2,-D/2,0],[W/2,D/2,0],[-W/2,D/2,0]])
    r=(proj(pos,look,fov,P)-corn).ravel()
    return r
q=least_squares(res,[0.0,-0.4,0.2,0,0,0.2,0.04]).x
pos,look,D,_=unpack(q); print('rms',np.sqrt(np.mean(res(q)**2)),pos,look,D)
def back(px,z=0.0):
    R=axes(pos,look); f=H_/2/np.tan(np.radians(fov)/2)
    d=np.array([(px[0]-W_/2)/f,(px[1]-H_/2)/f,1.])@R
    t=(z-pos[2])/d[2]; return pos+t*d
bases={'L1':(303,570),'L2':(325,520),'L3':(343,478),'L4':(358,445),'M1':(447,512),'M2':(450,465),
 'U1':(566,505),'U2':(562,465),'R1':(700,543),'R2':(682,500),'R3':(672,462),'R4':(662,432)}
B={k:back(v) for k,v in bases.items()}
for k,v in B.items(): print(k,v.round(4))
# post height: find h so top projects to observed y
for k,top in (('R1',(700,408)),('L1',(296,440))):
    hs=np.linspace(0.01,0.1,901); ys=[proj(pos,look,fov,B[k]+[0,0,h])[1] for h in hs]
    print(k,'height',hs[np.argmin(np.abs(np.array(ys)-top[1]))])
# sleeve width scale at R1: px per m
p=proj(pos,look,fov,np.array([B['R1']+[-.01,0,0.03],B['R1']+[.01,0,0.03]])); print('px for 20mm at R1 top',np.linalg.norm(p[1]-p[0]))
json.dump(dict(pos=pos.tolist(),look=look.tolist(),D=D,B={k:v.tolist() for k,v in B.items()}),open('calib.json','w'))
