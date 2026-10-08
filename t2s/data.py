"""Step 0 of v2: the clips (video, time window, frame stride, crop), their frames, shot-cut check and the scope's valid
image area.

    python -m t2s.data [clip ...]      -> outputs/t2s/<clip>/data/{frames.npz, valid.png, data.json, overview.jpg}

A clip keeps every `stride`-th frame of its window (fps = video fps / stride) so that clips of different length cost
about the same downstream (~250-500 frames).
"""
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 't2s'

CLIPS = {
    # name: video, first / last frame (inclusive, source fps), stride, crop rows (None = all)
    'chole_derot': dict(video='data/videos/commons_chole/gallbladder_torsion_S1.mp4', frames=(94, 586), stride=2,
                        crop_rows=(59, 419), text='data/videos/commons_chole/SOURCE.md'),
    'chole_a': dict(video='data/videos/commons_chole/gallbladder_torsion_S1.mp4', frames=(587, 837), stride=1,
                    crop_rows=(59, 419), text='data/videos/commons_chole/SOURCE.md'),
    'liver_dx': dict(video='data/videos/commons_liver/diagnostic_laparoscopy_S1.mp4', frames=(0, 1495), stride=4,
                     crop_rows=(12, 468), text='data/videos/commons_liver/SOURCE.md'),
    # liver_dx turned out to be five scenes joined by page-curl wipes (prompting agent, r01); scene 4 (36.2-50.2 s:
    # suturing at the left lobe tip / triangular ligament) is the longest continuous one
    'liver_s4': dict(video='data/videos/commons_liver/diagnostic_laparoscopy_S1.mp4', frames=(904, 1256), stride=2,
                     crop_rows=(12, 468), text='data/videos/commons_liver/SOURCE.md'),
    'lung_mln': dict(video='data/videos/commons_lung/vats_lobectomy_S2.mp4', frames=(783, 1552), stride=2,
                     crop_rows=(60, 420), text='data/videos/commons_lung/SOURCE.md'),
}


def read_frames(name):
    c = CLIPS[name]
    r = iio.get_reader(ROOT / c['video'], 'ffmpeg')
    fps = r.get_meta_data()['fps']
    a, b = c['frames']
    out = []
    for k, f in enumerate(r):
        if k < a:
            continue
        if k > b:
            break
        if c['crop_rows']:
            f = f[c['crop_rows'][0]:c['crop_rows'][1]]
        out.append(np.ascontiguousarray(f[..., :3]))
    full = np.stack(out)                       # every source frame of the window (for the cut check)
    return full[::c['stride']], fps / c['stride'], fps, full


def cuts(frames, thr=0.25):
    """Frame indices where a new shot starts (colour-histogram jump between consecutive kept frames)."""
    prev, out = None, []
    for i, f in enumerate(frames):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2HSV), (160, 90))
        h = cv2.calcHist([g], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > thr:
            out.append(i)
        prev = h
    return out


def valid_area(frames, thr=14):
    """Pixels inside the scope image (the round / octagonal border is dark in every frame)."""
    m = np.median(frames[::max(1, len(frames) // 40)].max(-1), 0)
    v = (m > thr).astype(np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(v)
    if n > 1:
        v = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return cv2.erode(v, np.ones((7, 7), np.uint8)).astype(bool)


def prepare(name):
    d = OUT / name / 'data'
    d.mkdir(parents=True, exist_ok=True)
    frames, fps, src_fps, full = read_frames(name)
    c = [int(round(i / CLIPS[name]['stride'])) for i in cuts(full)]      # cuts found at the source frame rate
    del full
    valid = valid_area(frames)
    np.savez_compressed(d / 'frames.npz', frames=frames)
    cv2.imwrite(str(d / 'valid.png'), valid.astype(np.uint8) * 255)
    info = dict(clip=name, **{k: v for k, v in CLIPS[name].items()}, n=len(frames), fps=fps, source_fps=src_fps,
                size=[int(frames.shape[2]), int(frames.shape[1])], cuts_at_frame=c, cuts_at_s=[round(i / fps, 2) for i in c],
                valid_fraction=round(float(valid.mean()), 3))
    (d / 'data.json').write_text(json.dumps(info, indent=1))
    ids = np.linspace(0, len(frames) - 1, 8).astype(int)
    tiles = []
    for i in ids:
        t = cv2.resize(frames[i], (320, int(320 * frames.shape[1] / frames.shape[2])))
        cv2.putText(t, f'{i} / {i / fps:.1f}s', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(t)
    iio.imwrite(d / 'overview.jpg', np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0), quality=82)
    return info


def load(name):
    d = OUT / name / 'data'
    info = json.loads((d / 'data.json').read_text())
    frames = np.load(d / 'frames.npz')['frames']
    valid = cv2.imread(str(d / 'valid.png'), 0) > 0
    return frames, valid, info


if __name__ == '__main__':
    for n in sys.argv[1:] or list(CLIPS):
        i = prepare(n)
        print(n, i['n'], 'frames', round(i['fps'], 2), 'fps', i['size'], 'cuts at s', i['cuts_at_s'], 'valid', i['valid_fraction'])
