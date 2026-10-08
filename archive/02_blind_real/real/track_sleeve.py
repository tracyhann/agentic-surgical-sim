"""2D track of one coloured sleeve (largest matching blob nearest to the previous position)."""
import json, sys, numpy as np, cv2, imageio.v2 as imageio, rosma
HSV = {'yellow': ((20, 120, 120), (35, 255, 255))}


def track(t0, t1, color, seed_xy, trial='X01'):
    r = imageio.get_reader(rosma.DATA / f'{trial}_Post_and_Sleeve_01.mp4')
    lo, hi = HSV[color]
    prev, out = np.array(seed_xy, float), []
    for n in range(int(t0 * 15), int(t1 * 15) + 1):
        f = r.get_data(n)
        m = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_RGB2HSV), lo, hi)
        m[:30] = 0
        k, lab, st, cen = cv2.connectedComponentsWithStats(m)
        best = None
        for i in range(1, k):
            if st[i, 4] < 150:
                continue
            d = np.linalg.norm(cen[i] - prev)
            if d < 120 and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            out.append(dict(t=n / 15, uv=None)); continue
        i = best[1]
        prev = cen[i]
        x, y, w, h, a = st[i]
        out.append(dict(t=n / 15, uv=cen[i].round(1).tolist(), bbox=[int(x), int(y), int(w), int(h)], area=int(a)))
    return out


if __name__ == '__main__':
    tr = track(5.0, 26.0, 'yellow', (697, 500))
    json.dump(tr, open('sleeve_track_X01_yellow.json', 'w'))
    ok = [p for p in tr if p['uv']]
    print('frames', len(tr), 'tracked', len(ok))
    for p in tr[::15]:
        print(round(p['t'], 1), p.get('uv'), p.get('area'))
