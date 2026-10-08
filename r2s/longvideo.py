"""Join the per-shot 2x2 videos of a long clip into one, with a title card at each cut.

  python -m r2s.longvideo <out.mp4> <clip>:<cond>:<title> [<clip>:<cond>:<title> ...]
"""
import sys
import numpy as np
import cv2
import imageio.v2 as imageio
from .config import Clip


def card(shape, lines, n):
    img = np.full(shape, 12, np.uint8)
    h = shape[0]
    for i, (text, scale) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y = int(h / 2 + (i - (len(lines) - 1) / 2) * 60)
        cv2.putText(img, text, ((shape[1] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (225, 232, 232), 2, cv2.LINE_AA)
    return [img] * n


def main(out, specs):
    frames = []
    for spec in specs:
        name, cond, title = spec.split(':', 2)
        clip = Clip(name)
        r = imageio.get_reader(clip.cond_dir(cond) / 'views.mp4')
        fps = r.get_meta_data()['fps']
        shot = [f for f in r]
        frames += card(shot[0].shape, [(title, 1.6), (f'{clip["source"]["frames"][0] / 25:.1f}-{clip["source"]["frames"][1] / 25:.1f} s of the source video', 0.9)], int(fps * 1.2))
        frames += shot
    imageio.mimsave(out, frames, fps=fps, macro_block_size=1, quality=7)
    print(out, len(frames), 'frames')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
