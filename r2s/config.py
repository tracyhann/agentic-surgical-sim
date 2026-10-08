"""Clip configuration (r2s/clips/<name>.json) and output layout (outputs/<name>/...)."""
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS = Path(__file__).resolve().parent / 'clips'
OUTPUTS = ROOT / 'outputs'
MODELS = ROOT / 'models'
TEMPLATES = ROOT / 'data' / 'templates' / 'bodyparts3d'
CONDITIONS = ('measured', 'primitive', 'template', 'template_fit')
# experiment variants of a clip, run as "<clip>+<variant>" into outputs/variants/<clip>+<variant>/ (config overrides)
VARIANTS = {m: {'multiview': m} for m in ('ba', 'sift', 'vggt_tracks', 'vggt')}   # multi-view modes, see multiview.MODES
VARIANTS['oldseg'] = {}     # diagnostic: prep pre-filled with the segmentation before 2026-10-07 18:00
VARIANT_TITLES = {'ba': '多视角·光流', 'sift': '多视角·SIFT', 'vggt_tracks': '多视角·VGGT 轨迹', 'vggt': '多视角·VGGT', 'oldseg': '旧分割'}
SHARED_PREP = ('masks.npz', 'disp.npy', 'cams.npz', 'metric.json', 'metric_b.npy', 'organ_tracks.npz')


class Clip:
    """A clip config plus its output folders.
      prep/      frames, SAM masks, relative depth, metric calibration, camera track (shared by all conditions)
      scene/     backdrop, textures, instruments, strands, organ observations (shared by all conditions)
      <cond>/    the organ body of one condition, its MuJoCo scene, trajectory, renders and metrics"""

    def __init__(self, name):
        self.name = name
        self.base, _, self.variant = name.partition('+')
        self.cfg = json.loads((CLIPS / f'{self.base}.json').read_text())
        if self.variant:
            self.cfg.update(VARIANTS[self.variant])
            self.cfg['title'] = f"{self.cfg['title']} · {VARIANT_TITLES[self.variant]}"
        self.out = OUTPUTS / 'variants' / name if self.variant else OUTPUTS / name
        self.prep = self.out / 'prep'
        self.scene = self.out / 'scene'
        for d in (self.prep, self.scene):
            d.mkdir(parents=True, exist_ok=True)
        if self.variant:                          # a variant reuses the base clip's perception (masks, depth, tracks)
            for f in SHARED_PREP:                 # read-only here: every stage writes them only when missing
                src = OUTPUTS / self.base / 'prep' / f
                if src.exists() and not (self.prep / f).exists():
                    (self.prep / f).symlink_to(os.path.relpath(src, self.prep))

    def __getitem__(self, k):
        return self.cfg[k]

    def get(self, k, default=None):
        return self.cfg.get(k, default)

    def cond_dir(self, cond):
        d = self.out / cond
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def objects(self):
        return list(self.cfg['objects'])

    def obj_index(self, name):
        return self.objects.index(name)

    def role(self, role):
        return [n for n, o in self.cfg['objects'].items() if o['role'] == role]

    @property
    def organ(self):
        return self.role('organ')[0]

    @property
    def instruments(self):
        return self.cfg['instruments']

    def camera(self):
        from .camera import Cam
        c = self.cfg['camera']
        W, H = self.cfg['size']
        f = c['f'] if 'f' in c else (H / 2) / np.tan(np.radians(c['fovy'] / 2))
        return Cam(W, H, f)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


def load_json(path):
    return json.loads(Path(path).read_text())
