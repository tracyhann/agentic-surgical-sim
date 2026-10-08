"""A clip seen through its multi-view cameras, for the v2 steps: same interface as r2s.views.Views (frames, masks,
cameras, metric depth, project / unproject, points, occluders), built from the clip's prep/ outputs only (no r2s
scene stage: instruments are the instrument agent's job in v2).

    from t2s import views2
    V = views2.load('chole_derot')          # geometry 'sift' (keyframe bundle adjustment + SIFT)
    V.mask('gallbladder'); V.depth(k); V.project(X, k); V.valid (scope image area)
"""
from functools import lru_cache

import numpy as np

from r2s.config import Clip
from r2s import source, perception
from . import data as D
from .geom import r2s_name


class Views:
    def __init__(self, clip, geometry='sift'):
        self.name = clip
        self.clip = Clip(r2s_name(clip))
        self.gclip = Clip(f'{r2s_name(clip)}+{geometry}') if geometry != 'single' else self.clip
        self.cam = self.clip.camera()
        self.frames, self.valid, self.info = D.load(clip)
        self.n, self.H, self.W = self.frames.shape[:3]
        self.fps = self.info['fps']
        self._masks = perception.load_masks(self.clip)
        self.names = self.clip.objects
        if geometry != 'single':
            mv = np.load(self.gclip.prep / 'multiview' / 'frames.npz')
            self.R, self.f, self.pos = mv['R'].astype(float), mv['f'].astype(float), mv['pos'].astype(float)
        else:
            cams = np.load(self.clip.prep / 'cams.npz')
            self.R, self.f = cams['R'].astype(float), cams['f'].astype(float)
            self.pos = np.repeat(self.cam.pos[None], self.n, 0).astype(float)
        self._depth = perception.metric_depth(self.gclip)
        self.keyframes = list(range(0, self.n, 10))
        self.instrument_names = [i['mask'] for i in self.clip.instruments]
        self.tools = {}

    def mask(self, name):
        return self._masks[:, self.names.index(name)]

    def depth(self, k):
        return self._depth[k]

    def occluders(self, k):
        ins = [self.mask(n)[k] for n in self.instrument_names]
        return (np.any(ins, 0) if ins else np.zeros((self.H, self.W), bool)) | ~self.valid

    def project(self, X, k):
        return self.cam.project(np.asarray(X, float), self.R[k], self.f[k], self.pos[k])

    def unproject(self, u, v, z, k):
        return self.cam.unproject(u, v, z, self.R[k], self.f[k], self.pos[k])

    def points(self, name, k, step=3, erode=2):
        import cv2
        m = (self.mask(name)[k] & self.valid).astype(np.uint8)
        if erode:
            m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        ys, xs = np.nonzero(m[::step, ::step])
        ys, xs = ys * step, xs * step
        return self.unproject(xs, ys, self.depth(k)[ys, xs], k)


@lru_cache(maxsize=4)
def load(clip, geometry='sift'):
    return Views(clip, geometry)
