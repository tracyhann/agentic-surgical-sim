"""Per-arm image projection from hand-marked tips (f fixed; 2D projection is insensitive to f in this setup)."""
import json, sys, numpy as np, cv2, rosma
from scipy.optimize import least_squares
W, H, F = 1024, 768, 650.0
DROP = {(16, 'PSM2'), (24, 'PSM1')}
ann = [a for a in json.load(open('annotations.json')) if (a['t'], a['arm']) not in DROP and not a.get('skip')]
kin = rosma.load_kin()
K = np.array([[F, 0, W/2], [0, F, H/2], [0, 0, 1.]])
out = dict(f=F, arms={})
for arm in ('PSM1', 'PSM2'):
    A = [a for a in ann if a['arm'] == arm]
    X = np.array([rosma.at_video_time(kin, arm, a['t'])[0] for a in A]); U = np.array([a['uv'] for a in A], float)
    ok, rv, tv = cv2.solvePnP(X, U, K, None, flags=cv2.SOLVEPNP_SQPNP)
    res = lambda p: (cv2.projectPoints(X, p[:3], p[3:], K, None)[0].reshape(-1, 2) - U).ravel()
    s = least_squares(res, np.r_[rv.ravel(), tv.ravel()], loss='soft_l1', f_scale=10)
    e = np.linalg.norm(res(s.x).reshape(-1, 2), axis=1)
    out['arms'][arm] = dict(rvec=s.x[:3].tolist(), tvec=s.x[3:].tolist(), n=len(A), resid_px=e.round(1).tolist())
    print(arm, len(A), 'pts, residuals', e.round(1).tolist(), 'median %.1f' % np.median(e))
json.dump(out, open('calib.json', 'w'), indent=1)
