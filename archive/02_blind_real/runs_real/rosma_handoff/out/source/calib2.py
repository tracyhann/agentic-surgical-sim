import numpy as np, json
from scipy.optimize import least_squares
exec(open('calib.py').read().split("fov=45.0")[0])
def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
vert=[((700,543),(700,408)),((303,570),(296,440)),((447,512),(438,393)),((566,505),(560,385))]
def mk(fov):
    def res(q):
        pos=q[:3]; look=np.array([q[3],q[4],0.]); D=q[5]; tilt=q[6]; hp=q[7]; W=0.2
        P=np.array([[-W/2,-D/2,0],[W/2,-D/2,0],[W/2,D/2,0],[-W/2,D/2,0]])@Ry(tilt).T
        r=list((proj(pos,look,fov,P)-corn).ravel())
        R=axes(pos,look); f=H_/2/np.tan(np.radians(fov)/2); n=Ry(tilt)@[0,0,1.]
        for b,t in vert:
            d=np.array([(b[0]-W_/2)/f,(b[1]-H_/2)/f,1.])@R
            s=-(pos@n)/(d@n); X=pos+s*d
            r+=list(proj(pos,look,fov,X+hp*n)-np.array(t))
        return np.array(r)
    return res
for fov in (35,40,45,50,55,60):
    s=least_squares(mk(fov),[0.0,-0.4,0.2,0,0,0.15,0.0,0.045]); print(fov,np.sqrt(np.mean(s.fun**2)).round(2),s.x.round(4))
