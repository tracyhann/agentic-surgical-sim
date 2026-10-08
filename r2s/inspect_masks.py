"""python -m r2s.inspect_masks <clip> [frames...]  -> outputs/<clip>/prep/masks_overview.jpg (SAM masks over frames)
and the per-object mask area per frame (to find where SAM loses an object)."""
import sys
import numpy as np
import cv2
import imageio.v2 as imageio
from .config import Clip
from . import source, perception

COLS = np.array([[90, 200, 90], [0, 220, 255], [255, 255, 255], [200, 60, 220], [255, 160, 60], [255, 60, 60], [60, 120, 255]], float)


def main(name, ks=None):
    clip = Clip(name)
    frames = source.frames(clip)
    masks = perception.load_masks(clip)
    n = len(frames)
    ks = ks or [int(round(t * (n - 1))) for t in np.linspace(0, 1, 8)]
    tiles = []
    for k in ks:
        v = frames[k].astype(float)
        for i in range(masks.shape[1]):
            v[masks[k, i]] = 0.5 * v[masks[k, i]] + 0.5 * COLS[i % len(COLS)]
        tiles.append(cv2.putText(v.astype(np.uint8), f'#{k}', (10, 30), 0, 0.9, (255, 255, 0), 2))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    imageio.imwrite(clip.prep / 'masks_overview.jpg', cv2.resize(img, None, fx=0.5, fy=0.5), quality=80)
    area = masks.sum((2, 3))
    for i, o in enumerate(clip.objects):
        print(f'{o:14s}', ' '.join(f'{a // 1000:3d}' for a in area[::max(1, n // 25), i]), ' (k px, every', max(1, n // 25), 'frames)')
    print('legend:', {o: COLS[i % len(COLS)].astype(int).tolist() for i, o in enumerate(clip.objects)})


if __name__ == '__main__':
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or None)
