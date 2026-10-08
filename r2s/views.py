"""One clip seen through one camera geometry: the shared input of the per-tissue models (r2s/tissue/) and the 4D study
(outputs/iter/).

    from r2s import views
    V = views.load('chole_a', geometry='sift')     # 'sift': multi-view keyframe BA + SIFT (default); 'single': rotation + zoom
    V = views.load('chole_a', 'sift', cams='outputs/iter/tissues/backdrop/v08/cams_refined.npz')   # refined rotations

    V.frames            (n, H, W, 3) uint8 video frames;  V.n, V.H, V.W, V.fps
    V.mask(name)        (n, H, W) bool SAM mask of a clip object (instrument pixels removed from soft tissue);
                        V.names lists them (chole_a: gallbladder, grasper, probe, strand, liver, right organs)
    V.R, V.f, V.pos     per-frame camera: rows of R are the camera axes in world (x right, y down, z forward),
                        f focal length (px), pos camera centre (world metres; MuJoCo frame, z up)
    V.depth(k)          (H, W) metric depth along the optical axis of frame k (multi-view refined for 'sift')
    V.project(X, k)     world points (N, 3) -> pixels (N, 2) and depth (N,) in frame k
    V.unproject(u, v, z, k) -> world points (N, 3)
    V.points(name, k, step=3)   world points of an object's visible pixels in frame k at their measured depth
    V.keyframes         every 10th frame
    V.tools             {instrument name: dict(tip (n, 3) world tool-centre point per video frame, rcm (3,),
                        shaft (n, 3) unit direction from port to tip, jaw (n,) opening, mask=object name)}
    V.occluders(k)      (H, W) bool pixels covered by instruments in frame k (unknown for any tissue behind them)

    segment_object(V, prompts) -> (n, H, W) bool: SAM 2.1 masks of a new object (prompts as in the clip configs:
                        [{"frame": k, "pos": [[x, y], ...], "neg": [[x, y], ...]}, ...])
"""
from functools import lru_cache
import numpy as np
from .config import Clip, MODELS


class Views:
    def __init__(self, name, geometry='sift', cams=None):
        from . import source, perception, sim as SIM, instruments as INS
        self.name, self.geometry = name, geometry
        self.clip = Clip(name)
        gclip = Clip(f'{name}+{geometry}') if geometry != 'single' else self.clip
        self.gclip = gclip
        self.cam = self.clip.camera()
        self.frames = np.stack(source.frames(self.clip))
        self.n, self.H, self.W = self.frames.shape[:3]
        self.fps = self.clip['fps']
        self._masks = perception.load_masks(self.clip)
        self.names = self.clip.objects
        S = SIM.load_scene(gclip)
        self.R = S['cam_R'][:self.n].astype(float)
        self.cams = cams or 'clip'
        if cams and cams != 'clip':          # e.g. a cams_refined.npz: per-frame rotations registered between keyframes
            self.R = np.load(cams)['R'][:self.n].astype(float)
        self.f = S['cam_f'][:self.n].astype(float)
        self.pos = (S['cam_pos'][:self.n] if 'cam_pos' in S else np.repeat(self.cam.pos[None], self.n, 0)).astype(float)
        self._depth = perception.metric_depth(gclip)
        self.keyframes = list(range(0, self.n, 10))
        # instruments: joint actions per simulation step (dt) -> tool-centre point per video frame
        dt = SIM.PARAMS['dt']
        acts = S['actions']
        t_steps = np.arange(len(acts)) * dt
        t_frames = np.arange(self.n) / self.fps
        self.tools = {}
        for j, ins in enumerate(self.clip.instruments):
            meta = S['meta']['instruments'][ins['name']]
            rcm, heading = np.array(meta['rcm'], float), float(meta['heading'])
            a = np.stack([np.interp(t_frames, t_steps, acts[:, 5 * j + c]) for c in range(5)], 1)
            tip = np.array([INS.fk(rcm, y, p, i, heading) for y, p, i in a[:, :3]])
            shaft = np.array([INS.shaft_dir(y, p, heading) for y, p in a[:, :2]])
            self.tools[ins['name']] = dict(tip=tip, rcm=rcm, shaft=shaft, jaw=a[:, 4], mask=ins['mask'],
                                           holds=ins.get('holds'))

    def mask(self, name):
        return self._masks[:, self.names.index(name)]

    def depth(self, k):
        return self._depth[k]

    def occluders(self, k):
        return np.any([self.mask(t['mask'])[k] for t in self.tools.values()], 0)

    def project(self, X, k):
        return self.cam.project(np.asarray(X, float), self.R[k], self.f[k], self.pos[k])

    def unproject(self, u, v, z, k):
        return self.cam.unproject(u, v, z, self.R[k], self.f[k], self.pos[k])

    def points(self, name, k, step=3, erode=2):
        import cv2
        m = self.mask(name)[k].astype(np.uint8)
        if erode:
            m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        ys, xs = np.nonzero(m[::step, ::step])
        ys, xs = ys * step, xs * step
        return self.unproject(xs, ys, self.depth(k)[ys, xs], k)


@lru_cache(maxsize=4)
def load(name='chole_a', geometry='sift', cams=None):
    """cams: None / 'clip' = the geometry's cameras; a path to a cams_refined.npz = those rotations."""
    return Views(name, geometry, cams)


def segment_object(V, prompts, device='mps'):
    """SAM 2.1 video masks (n, H, W) of one new object from point prompts."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    model = Sam2VideoModel.from_pretrained(MODELS / 'sam2.1-hiera-tiny').to(device, dtype=torch.float32).eval()
    proc = Sam2VideoProcessor.from_pretrained(MODELS / 'sam2.1-hiera-tiny')
    sess = proc.init_video_session(video=list(V.frames), inference_device=device, dtype=torch.float32)
    out_masks = np.zeros((V.n, V.H, V.W), bool)
    with torch.no_grad():
        for p in prompts:
            pts = [list(map(float, q)) for q in p['pos'] + p.get('neg', [])]
            lab = [1] * len(p['pos']) + [0] * len(p.get('neg', []))
            proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=int(p['frame']), obj_ids=1,
                                                 input_points=[[pts]], input_labels=[[lab]])
            model(inference_session=sess, frame_idx=int(p['frame']))
        for out in model.propagate_in_video_iterator(sess):
            m = proc.post_process_masks([out.pred_masks], original_sizes=[[V.H, V.W]], binarize=True)[0]
            out_masks[out.frame_idx] = m[0, 0].cpu().numpy()
    return out_masks
