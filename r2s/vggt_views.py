"""VGGT (Wang et al., CVPR 2025) as a source of wide-baseline geometry for the keyframe BA (multiview.py).

Weights facebook/VGGT-1B (CC BY-NC 4.0) in models/VGGT-1B, code in third_party/vggt. VGGT sees all keyframes at once
and predicts their cameras, depth maps and point tracks in one forward pass. Used for
  - wide-baseline correspondences: tracks of static-anatomy corners from several query keyframes to all others
    (one pass per query keyframe, moved to the front), each keyframe pair RANSAC-verified,
  - initial values (mode 'vggt'): poses and focal length, metric scale from instrument shafts of known diameter, and
    the Depth Anything inverse-depth affine of each keyframe fitted to VGGT's depth.
On laparoscopic video its visibility / confidence outputs collapse (out of domain), so they are not used as filters.
"""
import sys
import time
import types
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .config import ROOT, MODELS

VGGT_CODE = ROOT / 'third_party' / 'vggt'
VGGT_WEIGHTS = MODELS / 'VGGT-1B' / 'model.safetensors'
IMG_W = 518                                     # VGGT input width; height follows the aspect ratio (multiple of 14)


def _chunked_sdpa(q, k, v, dropout_p=0.0, chunk=2048, **kw):
    """Attention over query chunks: the global attention of VGGT over all keyframes' tokens would otherwise
    materialise (heads x N x N) on MPS."""
    if q.shape[-2] <= chunk:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, **kw)
    return torch.cat([F.scaled_dot_product_attention(q[..., i:i + chunk, :], k, v, dropout_p=dropout_p, **kw)
                      for i in range(0, q.shape[-2], chunk)], dim=-2)


def load_model(device):
    if str(VGGT_CODE) not in sys.path:
        sys.path.insert(0, str(VGGT_CODE))
    import vggt.layers.attention as attention
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    shim = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith('_')})
    shim.scaled_dot_product_attention = _chunked_sdpa
    attention.F = shim
    model = VGGT()
    model.load_state_dict(load_file(str(VGGT_WEIGHTS)))
    return model.eval().to(device)


def preprocess(frames):
    """RGB uint8 frames (H, W) -> (S, 3, h, 518) float in [0, 1], plus the pixel scale (sx, sy) original -> VGGT."""
    H, W = frames[0].shape[:2]
    h = int(round(H * IMG_W / W / 14)) * 14
    imgs = np.stack([cv2.resize(f, (IMG_W, h), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32) / 255
    return torch.from_numpy(imgs).permute(0, 3, 1, 2), (IMG_W / W, h / H)


@torch.no_grad()
def predict(model, imgs, device, queries=None, dtype=torch.bfloat16, iters=None):
    """One VGGT pass over (S, 3, h, w) images; the first image is the reference (world frame) and the query frame of
    the tracks. Returns numpy arrays: extrinsic (S, 3, 4) camera-from-world, intrinsic (S, 3, 3), depth (S, h, w),
    depth_conf (S, h, w), and if queries (N, 2) given in VGGT pixels: track (S, N, 2), vis (S, N), conf (S, N)."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = imgs.to(device)[None]
    with torch.autocast(device_type=device.type, dtype=dtype):
        tokens, ps = model.aggregator(x)
    tokens = [None if t is None else t.float() for t in tokens]   # only the layers the heads read are kept
    pose_enc = model.camera_head(tokens)[-1]
    E, K = pose_encoding_to_extri_intri(pose_enc, x.shape[-2:])
    depth, dconf = model.depth_head(tokens, images=x, patch_start_idx=ps)
    out = dict(extrinsic=E[0].cpu().numpy(), intrinsic=K[0].cpu().numpy(), depth=depth[0, ..., 0].cpu().numpy(),
               depth_conf=dconf[0].cpu().numpy())
    if queries is not None and len(queries):
        q = torch.tensor(queries, dtype=torch.float32, device=device)[None]
        tracks, vis, conf = model.track_head(tokens, images=x, patch_start_idx=ps, query_points=q, iters=iters)
        out.update(track=tracks[-1][0].cpu().numpy(), vis=vis[0].cpu().numpy(), conf=conf[0].cpu().numpy())
    del tokens
    if device.type == 'mps':
        torch.mps.empty_cache()
    return out


def corners(gray, mask, n):
    p = cv2.goodFeaturesToTrack(gray, n, 0.005, 9, mask=mask.astype(np.uint8) * 255)
    return np.zeros((0, 2)) if p is None else p[:, 0].astype(np.float64)


def run(clip, frames, static, keys, every_query=4, n_query=600, iters=12, verify_px=3.0, log=print):
    """Raw VGGT geometry of the keyframes (keyframe 0 first, as reference) and verified track correspondences
    (I, J, Pi, Pj) in the clip's pixels."""
    from .multiview import verified
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    t0 = time.time()
    model = load_model(device)
    imgs, (sx, sy) = preprocess([frames[k] for k in keys])
    S, H, W = len(keys), frames[0].shape[0], frames[0].shape[1]
    query_keys = list(range(0, S, every_query))
    log(f'[vggt] model on {device} ({time.time() - t0:.0f} s); {S} keyframes at {imgs.shape[-1]}x{imgs.shape[-2]}, '
        f'query keyframes {query_keys}')
    raw = None
    pairs = {}
    for qi in query_keys:
        order = [qi] + [i for i in range(S) if i != qi]
        P = corners(cv2.cvtColor(frames[keys[qi]], cv2.COLOR_RGB2GRAY), static[keys[qi]], n_query)
        pred = predict(model, imgs[order], device, queries=P * [sx, sy], iters=iters)
        inv = np.argsort(order)                            # back to keyframe order
        if qi == 0:
            raw = {k: v[inv] for k, v in pred.items() if k in ('extrinsic', 'intrinsic', 'depth', 'depth_conf')}
        T = pred['track'][inv] / [sx, sy]
        for j in range(S):
            if j == qi:
                continue
            q = T[j]
            inside = (q[:, 0] >= 0) & (q[:, 0] < W - 1) & (q[:, 1] >= 0) & (q[:, 1] < H - 1)
            qx, qy = np.clip(q[:, 0], 0, W - 1).astype(int), np.clip(q[:, 1], 0, H - 1).astype(int)
            ok = inside & static[keys[j]][qy, qx]
            ok[ok] &= verified(P[ok], q[ok], verify_px)
            if ok.sum() >= 25:
                pairs[(qi, j)] = (P[ok], q[ok])
    log(f'[vggt] {len(query_keys)} passes ({time.time() - t0:.0f} s), {len(pairs)} verified keyframe pairs')
    del model
    if device.type == 'mps':
        torch.mps.empty_cache()
    raw['scale_xy'] = np.array([sx, sy])
    I = np.concatenate([np.full(len(a), i) for (i, j), (a, b) in pairs.items()] or [np.zeros(0, int)]).astype(int)
    J = np.concatenate([np.full(len(a), j) for (i, j), (a, b) in pairs.items()] or [np.zeros(0, int)]).astype(int)
    Pi = np.concatenate([a for a, b in pairs.values()] or [np.zeros((0, 2))])
    Pj = np.concatenate([b for a, b in pairs.values()] or [np.zeros((0, 2))])
    return raw, I, J, Pi, Pj


def metric(raw, rulers_px, W, H):
    """VGGT geometry in the clip's pixels and metres: focal length, rotations, scaled translations and depth maps.
    Scale: instrument shafts (depth = f * diameter / apparent width) against VGGT's depth at the same pixels.
    rulers_px rows: (key index, u, v, 1/width, diameter)."""
    sx, sy = raw['scale_xy']
    K = raw['intrinsic']
    f = float(np.median(np.r_[K[:, 0, 0] / sx, K[:, 1, 1] / sy]))
    D = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) for d in raw['depth']])
    Dc = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) for d in raw['depth_conf']])
    ratios = [f * shaft * invw / D[int(k), int(v), int(u)] for k, u, v, invw, shaft in rulers_px if D[int(k), int(v), int(u)] > 0]
    s = float(np.median(ratios))
    E = raw['extrinsic']
    R = E[:, :, :3] @ E[0, :, :3].T                       # relative to keyframe 0 (VGGT already puts it at identity)
    t = s * (E[:, :, 3] - np.einsum('kij,j->ki', R, E[0, :, 3]))
    return dict(f=f, R=R, t=t, depth=s * D, depth_conf=Dc, scale=s, ruler_spread=float(np.std(np.log(ratios))))


def fit_affine(disp_k, z_k, w_k):
    """1/z = a d + b on pixels with weights w (weighted least squares, one robust re-weighting)."""
    d, y, w = disp_k.ravel(), 1.0 / z_k.ravel(), w_k.ravel()
    for _ in range(2):
        A = np.stack([d, np.ones_like(d)], 1) * np.sqrt(w)[:, None]
        a, b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        r = np.abs(a * d + b - y)
        w = w * (r < 3 * np.median(r) + 1e-9)
    return float(a), float(b)
