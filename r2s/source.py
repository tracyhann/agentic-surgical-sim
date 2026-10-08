"""Source frames of a clip (a video time range or an image sequence) and, where the dataset has one, reference depth."""
import glob
import numpy as np
import imageio.v2 as imageio
from .config import ROOT


def frames(clip):
    s = clip['source']
    if s['type'] == 'video':
        r = imageio.get_reader(ROOT / s['path'], 'ffmpeg')
        a, b = s['frames']
        out = []
        for k, f in enumerate(r):
            if k < a:
                continue
            if k > b:
                break
            if 'crop_rows' in s:
                f = f[s['crop_rows'][0]:s['crop_rows'][1]]
            out.append(np.ascontiguousarray(f[..., :3]))
    else:
        files = sorted(glob.glob(str(ROOT / s['path'] / s['pattern'])))
        a, b = s.get('frames', [0, len(files) - 1])
        out = [np.ascontiguousarray(imageio.imread(p)[..., :3]) for p in files[a:b + 1]]
    if s.get('fix_borders'):
        # the letterbox crop keeps dark rows / columns at the edges; they would show as dark lines in the textures
        for f in out:
            f[0] = f[1]
            f[-2:] = f[-3]
            f[:, -2:] = f[:, -3:-2]
    return out


def reference_depth(clip):
    """Metres per pixel and validity from an external depth source (e.g. EndoNeRF stereo), or None."""
    r = clip.get('reference_depth')
    if not r:
        return None
    files = sorted(glob.glob(str(ROOT / r['path'] / r['pattern'])))
    a, b = clip['source'].get('frames', [0, len(files) - 1])
    Z = np.stack([imageio.imread(p).astype(np.float32) for p in files[a:b + 1]]) * float(r['scale_m'])
    return Z, Z > 0
