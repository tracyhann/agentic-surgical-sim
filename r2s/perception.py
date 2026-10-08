"""Per-frame perception: SAM 2.1 video masks, Depth Anything V2 relative depth, metric scale from instrument shafts.

Cached in outputs/<clip>/prep/: masks.npz (packed bits), disp.npy (float16), metric.json + metric_b.npy.
"""
import time
import numpy as np
from .config import MODELS, save_json, load_json


# ---------------------------------------------------------------- segmentation
def segment(clip, frames, device='mps'):
    """One SAM 2 object per clip object. Prompts: [{"frame": k, "pos": [[x, y], ...], "neg": [...]}, ...]."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    H, W = frames[0].shape[:2]
    model = Sam2VideoModel.from_pretrained(MODELS / 'sam2.1-hiera-tiny').to(device, dtype=torch.float32).eval()
    proc = Sam2VideoProcessor.from_pretrained(MODELS / 'sam2.1-hiera-tiny')
    t0 = time.time()
    sess = proc.init_video_session(video=frames, inference_device=device, dtype=torch.float32)
    names = clip.objects
    with torch.no_grad():
        for oid, name in enumerate(names, start=1):
            for p in clip['objects'][name]['prompts']:
                pts = [list(map(float, q)) for q in p['pos'] + p.get('neg', [])]
                lab = [1] * len(p['pos']) + [0] * len(p.get('neg', []))
                proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=int(p['frame']), obj_ids=oid,
                                                     input_points=[[pts]], input_labels=[[lab]])
                model(inference_session=sess, frame_idx=int(p['frame']))
        masks = np.zeros((len(frames), len(names), H, W), bool)
        for out in model.propagate_in_video_iterator(sess):
            m = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            ids = [int(i) for i in (out.object_ids if hasattr(out, 'object_ids') else range(1, len(names) + 1))]
            for j, oid in enumerate(ids):
                masks[out.frame_idx, oid - 1] = m[j, 0].cpu().numpy()
    np.savez_compressed(clip.prep / 'masks.npz', masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape),
                        names=np.array(names))
    print(f'[segment] {len(frames)} frames in {time.time() - t0:.0f} s; mean area:',
          {n: int(masks[:, i].sum((1, 2)).mean()) for i, n in enumerate(names)})
    return masks


def load_masks(clip, clean=True):
    """SAM masks (n, objects, H, W). clean: instruments are always in front, so organ and strand masks lose the
    instrument pixels (SAM tends to let a wet organ mask run up the shaft of the instrument holding it)."""
    import cv2
    z = np.load(clip.prep / 'masks.npz')
    shp = tuple(z['shape'])
    masks = np.unpackbits(z['masks'], axis=-1)[..., :shp[-1]].astype(bool)
    if clean:
        ins = [clip.obj_index(i['mask']) for i in clip.instruments]
        soft = [clip.obj_index(n) for n in clip.role('organ') + clip.role('strand')]
        k3 = np.ones((3, 3), np.uint8)
        for k in range(len(masks)):
            cover = cv2.dilate(np.any(masks[k, ins], 0).astype(np.uint8), k3).astype(bool)
            for i in soft:
                masks[k, i] &= ~cover
    return masks


# ---------------------------------------------------------------- relative depth
def depth_rel(clip, frames, device='mps'):
    """Depth Anything V2 Small, raw output (relative inverse depth: larger = nearer), resized to the frame."""
    import torch
    import torch.nn.functional as Fnn
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(MODELS / 'Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained(MODELS / 'Depth-Anything-V2-Small-hf').to(device).eval()
    H, W = frames[0].shape[:2]
    out = np.zeros((len(frames), H, W), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for k, f in enumerate(frames):
            inp = proc(images=f, return_tensors='pt').to(device)
            d = model(**inp).predicted_depth[:, None]
            out[k] = Fnn.interpolate(d, size=(H, W), mode='bilinear', align_corners=False)[0, 0].float().cpu().numpy()
    np.save(clip.prep / 'disp.npy', out)
    print(f'[depth] {len(frames)} frames in {time.time() - t0:.0f} s')
    return out


# ---------------------------------------------------------------- metric scale from shafts of known diameter
def shaft_samples(m, fk, shaft_d, H, W, keep=(0.05, 0.70)):
    """(u, v, z) samples along the shaft part of an instrument mask: depth = f * diameter / apparent width.
    Only the proximal part (from the image border inwards, `keep` fractions of the axis) is used: wrists and jaws are
    wider or narrower than the shaft."""
    ys, xs = np.nonzero(m)
    if len(xs) < 300:
        return []
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    d = np.linalg.svd(P - c)[2][0]
    s = (P - c) @ d
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    end_lo, end_hi = c + d * lo, c + d * hi
    border = lambda q: min(q[0], W - 1 - q[0], q[1], H - 1 - q[1])
    if border(end_hi) < border(end_lo):             # make s grow from the entry border towards the tip
        d, s, lo, hi = -d, -s, -hi, -lo
    nrm = np.array([-d[1], d[0]])
    out = []
    for t in np.linspace(lo + keep[0] * (hi - lo), lo + keep[1] * (hi - lo), 24):
        sel = np.abs(s - t) < 1.5
        if sel.sum() < 4:
            continue
        w = np.ptp((P[sel] - c) @ nrm) + 1
        q = c + d * t
        if 5 < w < 120 and 0 <= q[0] < W and 0 <= q[1] < H:
            out.append((q[0], q[1], fk * shaft_d / w))
    return out


def calibrate(clip, disp, masks, cam_f):
    """1/z = a * d + b_k: one slope for the clip, smoothed per-frame offsets; fitted on the instrument shafts."""
    H, W = disp.shape[1:]
    rows = []
    for k in range(len(disp)):
        for ins in clip.instruments:
            if not ins.get('ruler', True):
                continue
            for u, v, z in shaft_samples(masks[k, clip.obj_index(ins['mask'])], cam_f[k], ins['shaft_d'], H, W):
                rows.append((k, float(disp[k, int(v), int(u)]), 1 / z))
    R = np.array(rows)
    K = len(disp)
    a, b = 3.0, np.full(K, 5.0)
    for _ in range(10):
        for k in range(K):
            sel = R[:, 0] == k
            if sel.sum() >= 3:
                b[k] = np.median(R[sel, 2] - a * R[sel, 1])
        bb = b[R[:, 0].astype(int)]
        res = R[:, 2] - bb
        w = np.abs(res - a * R[:, 1]) < 3 * np.median(np.abs(res - a * R[:, 1])) + 1e-6
        a = float((R[w, 1] @ res[w]) / (R[w, 1] @ R[w, 1]))
    have = np.array([(R[:, 0] == k).sum() >= 3 for k in range(K)])
    b = np.interp(np.arange(K), np.nonzero(have)[0], b[have])
    b = np.convolve(np.pad(b, 7, mode='edge'), np.ones(15) / 15, mode='valid')
    z_pred = 1 / (a * R[:, 1] + b[R[:, 0].astype(int)])
    err = np.abs(z_pred - 1 / R[:, 2])
    stats = dict(a=a, n_samples=len(R), frames_with_rulers=int(have.sum()), median_err_mm=float(np.median(err) * 1000),
                 p90_err_mm=float(np.percentile(err, 90) * 1000))
    save_json(clip.prep / 'metric.json', stats)
    np.save(clip.prep / 'metric_b.npy', b)
    print('[metric]', {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    return a, b


def metric_depth(clip, disp=None):
    """(n, H, W) metric depth in metres: from the multi-view solution for clips that use it, otherwise from the
    single-view calibration on instrument shafts."""
    if clip.get('multiview') and (clip.prep / 'multiview' / 'frames.npz').exists():
        from .multiview import depth_frames
        return depth_frames(clip, disp)
    disp = np.load(clip.prep / 'disp.npy').astype(np.float32) if disp is None else disp
    a = load_json(clip.prep / 'metric.json')['a']
    b = np.load(clip.prep / 'metric_b.npy')
    return 1.0 / np.clip(a * disp + b[:, None, None], 1.0, None)
