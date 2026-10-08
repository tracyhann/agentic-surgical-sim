import imageio.v2 as imageio, numpy as np, cv2
from rosma import DATA, FPS

_reader = {}
def frame(t, trial='X01'):
    if trial not in _reader:
        _reader[trial] = imageio.get_reader(DATA / f'{trial}_Post_and_Sleeve_01.mp4')
    return _reader[trial].get_data(int(round(t * FPS)))

def grid_full(times, path, step=50, scale=0.5):
    tiles = []
    for t in times:
        f = frame(t).copy()
        for x in range(0, f.shape[1], step):
            cv2.line(f, (x, 0), (x, f.shape[0]), (0, 255, 255) if x % 100 == 0 else (0, 120, 120), 1)
        for y in range(0, f.shape[0], step):
            cv2.line(f, (0, y), (f.shape[1], y), (0, 255, 255) if y % 100 == 0 else (0, 120, 120), 1)
        for x in range(0, f.shape[1], 100):
            cv2.putText(f, str(x), (x + 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for y in range(100, f.shape[0], 100):
            cv2.putText(f, str(y), (2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f, f't={t}s', (800, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        tiles.append(cv2.resize(f, None, fx=scale, fy=scale))
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    imageio.imwrite(path, np.concatenate(rows, 0))

def zoom(t, cx, cy, path, half=60, z=5, step=10):
    f = frame(t)
    x0, y0 = int(cx - half), int(cy - half)
    c = f[max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half]
    c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST).copy()
    for k in range(0, 2 * half + 1, step):
        col = (0, 255, 255) if (x0 + k) % 50 == 0 else (0, 140, 140)
        cv2.line(c, (k * z, 0), (k * z, c.shape[0]), col, 1)
        cv2.putText(c, str(x0 + k), (k * z + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        col = (0, 255, 255) if (y0 + k) % 50 == 0 else (0, 140, 140)
        cv2.line(c, (0, k * z), (c.shape[1], k * z), col, 1)
        cv2.putText(c, str(y0 + k), (2, k * z - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    imageio.imwrite(path, c)
