import json, sys, numpy as np, cv2, imageio.v2 as imageio, rosma, annot_tools as A
W, H = 1024, 768
cal = json.load(open('calib.json')); K = np.array([[cal['f'], 0, W/2], [0, cal['f'], H/2], [0, 0, 1.]])
kin = rosma.load_kin()
COL = {'PSM1': (255, 60, 60), 'PSM2': (60, 160, 255)}
def proj(arm, t):
    c = cal['arms'][arm]
    return cv2.projectPoints(rosma.at_video_time(kin, arm, t), np.array(c['rvec']), np.array(c['tvec']), K, None)[0].reshape(-1, 2)
def draw(t, f=None):
    f = A.frame(t).copy() if f is None else f
    for arm in ('PSM1', 'PSM2'):
        tr = proj(arm, np.clip(np.arange(t - 1.0, t + 0.01, 1/15), 0.3, 118))
        for a, b in zip(tr[:-1], tr[1:]):
            cv2.line(f, tuple(a.astype(int)), tuple(b.astype(int)), COL[arm], 1)
        cv2.circle(f, tuple(tr[-1].astype(int)), 7, COL[arm], 2)
    cv2.putText(f, f't={t}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    return f
if __name__ == '__main__':
    ts = [float(x) for x in sys.argv[2:]]
    tiles = [cv2.resize(draw(t), None, fx=0.5, fy=0.5) for t in ts]
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    imageio.imwrite(sys.argv[1], np.concatenate([np.concatenate(tiles[i:i+2], 1) for i in range(0, len(tiles), 2)], 0))
