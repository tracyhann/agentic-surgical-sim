"""Camera <-> PSM calibration from hand-marked tool tips: shared pinhole K (f, principal point at centre)
+ one 6-DoF pose per arm (each arm's kinematics live in its own RCM frame)."""
import json
import numpy as np, cv2
from scipy.optimize import least_squares
import rosma

W, H = 1024, 768
ANN = 'annotations.json'


def K_of(f):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def project(params, arm_idx, X):
    f = params[0]
    rv, tv = params[1 + 6 * arm_idx:4 + 6 * arm_idx], params[4 + 6 * arm_idx:7 + 6 * arm_idx]
    uv, _ = cv2.projectPoints(np.asarray(X, float), rv, tv, K_of(f), None)
    return uv.reshape(-1, 2)


def fit(ann, kin, f0=900.0, drop=()):
    arms = ['PSM1', 'PSM2']
    data = {a: [(t, np.array(uv)) for t, a2, uv in ann if a2 == a and (t, a2) not in drop] for a in arms}
    X = {a: np.ascontiguousarray(rosma.at_video_time(kin, a, [t for t, _ in data[a]]), dtype=np.float64) for a in arms}
    U = {a: np.array([uv for _, uv in data[a]], dtype=np.float64) for a in arms}
    p0 = [f0]
    for a in arms:
        ok, rv, tv = cv2.solvePnP(X[a], U[a], K_of(f0), None, flags=cv2.SOLVEPNP_EPNP if len(U[a]) >= 4 else 0)
        ok, rv, tv = cv2.solvePnP(X[a], U[a], K_of(f0), None, rv, tv, useExtrinsicGuess=True)
        p0 += list(rv.ravel()) + list(tv.ravel())
    res = lambda p: np.concatenate([(project(p, i, X[a]) - U[a]).ravel() for i, a in enumerate(arms)])
    sol = least_squares(res, p0, loss='soft_l1', f_scale=8.0)
    r = res(sol.x).reshape(-1, 2)
    per = {}
    k = 0
    for a in arms:
        for t, _ in data[a]:
            per[(t, a)] = float(np.linalg.norm(r[k])); k += 1
    return sol.x, per


def cam_from_arm(params, arm_idx):
    """4x4 transform: arm RCM frame -> OpenCV camera frame."""
    rv, tv = params[1 + 6 * arm_idx:4 + 6 * arm_idx], params[4 + 6 * arm_idx:7 + 6 * arm_idx]
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rv))[0]
    T[:3, 3] = tv
    return T


if __name__ == '__main__':
    ann = [(a['t'], a['arm'], a['uv']) for a in json.load(open(ANN))]
    kin = rosma.load_kin()
    best = None
    for f0 in (600, 800, 1000, 1300, 1700):
        p, per = fit(ann, kin, f0)
        rms = np.sqrt(np.mean(np.square(list(per.values()))))
        if best is None or rms < best[2]:
            best = (p, per, rms)
    p, per, rms = best
    print('focal %.1f px  (vertical fov %.1f deg)  rms %.1f px' % (p[0], np.degrees(2 * np.arctan(H / 2 / p[0])), rms))
    for k, v in sorted(per.items()):
        print('  ', k, '%.1f px' % v)
    for i, a in enumerate(('PSM1', 'PSM2')):
        T = cam_from_arm(p, i)
        print(a, 'RCM in camera frame (m):', T[:3, 3].round(4))
    json.dump(dict(f=p[0], params=list(p)), open('calib.json', 'w'), indent=1)
