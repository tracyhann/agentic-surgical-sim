"""Step 2 of v2: SAM 3 video segmentation from the text prompts of a clip's scene_spec, plus its self-check.

    python -m t2s.seg3 <clip> [vNN] [--prompts "a,b,c"]   -> outputs/t2s/<clip>/seg/vNN/
        masks.npz      one (n, H, W) bool array per prompt (union of that prompt's instances), packed along W
        instances.npz  per prompt: instance count per frame and mean detection score per frame
        qc.json        coverage, temporal stability, instance counts, scores; for chole_a also IoU vs the v1 SAM 2.1 masks
        sheet.jpg      every prompt's mask over keyframes

Prompts come from scene_spec.json: organs, instruments and background entries (their "sam_prompt").
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import data as D

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'sam3'
COLORS = [(255, 210, 40), (80, 200, 255), (240, 90, 200), (120, 230, 120), (255, 140, 60), (170, 130, 255),
          (60, 255, 220), (255, 90, 90), (200, 200, 200), (255, 255, 120)]


def prompts_of(spec):
    out = []
    for key in ('organs', 'instruments', 'background'):
        for o in spec.get(key, []):
            p = (o.get('sam_prompt') or o.get('name') or o.get('type') or '').strip()
            if p and p not in out:
                out.append(p)
    if 'surgical instrument' not in out:          # generic instrument concept (SAM 3 finds it; specific tool names vary)
        out.append('surgical instrument')
    return out


def segment(frames, prompts, device=None, log=print, instances_of=()):
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Sam3VideoModel.from_pretrained(MODEL).to(device, dtype=torch.float32).eval()
    proc = Sam3VideoProcessor.from_pretrained(MODEL)
    n, H, W = frames.shape[:3]
    masks = {p: np.zeros((n, H, W), bool) for p in prompts}
    count = {p: np.zeros(n, np.int16) for p in prompts}
    score = {p: np.zeros(n, np.float32) for p in prompts}
    inst = {}                                   # (prompt, object id) -> (n, H, W) for prompts in instances_of
    t0 = time.time()
    with torch.no_grad():
        sess = proc.init_video_session(video=list(frames), inference_device=device, processing_device='cpu',
                                       video_storage_device='cpu', dtype=torch.float32)
        proc.add_text_prompt(sess, prompts)
        for out in model.propagate_in_video_iterator(sess):
            r = proc.postprocess_outputs(sess, out)
            k = out.frame_idx
            ids = r['object_ids'].tolist()
            for p, pids in r['prompt_to_obj_ids'].items():
                if p not in masks:
                    continue
                for oid in pids:
                    if oid in ids:
                        j = ids.index(oid)
                        mj = r['masks'][j].cpu().numpy()
                        masks[p][k] |= mj
                        if p in instances_of:
                            inst.setdefault((p, oid), np.zeros((n, H, W), bool))[k] = mj
                        count[p][k] += 1
                        score[p][k] = max(score[p][k], float(r['scores'][j]))
            if k % 50 == 0:
                log(f'[seg3] frame {k}/{n} {time.time() - t0:.0f} s')
    if instances_of:
        return masks, count, score, inst
    return masks, count, score


def track_points(frames, objects, device=None, log=print):
    """SAM 3's interactive tracker (SAM 2 style) from point prompts: objects = {name: [{"frame": k, "pos": [[x, y], ...],
    "neg": [[x, y], ...]}, ...]} -> {name: (n, H, W) bool}. Used for objects the text prompt does not find; the points
    come from the prompting agent (vision), not from a person."""
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Sam3TrackerVideoModel.from_pretrained(MODEL).to(device, dtype=torch.float32).eval()
    proc = Sam3TrackerVideoProcessor.from_pretrained(MODEL)
    n, H, W = frames.shape[:3]
    names = list(objects)
    out = {nm: np.zeros((n, H, W), bool) for nm in names}
    t0 = time.time()
    with torch.no_grad():
        sess = proc.init_video_session(video=list(frames), inference_device=device, processing_device='cpu',
                                       video_storage_device='cpu', dtype=torch.float32)
        first = n
        for oid, nm in enumerate(names, start=1):
            for pr in objects[nm]:
                pts = [list(map(float, q)) for q in pr['pos'] + pr.get('neg', [])]
                lab = [1] * len(pr['pos']) + [0] * len(pr.get('neg', []))
                proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=int(pr['frame']), obj_ids=oid,
                                                     input_points=[[pts]], input_labels=[[lab]])
                model(inference_session=sess, frame_idx=int(pr['frame']))
                first = min(first, int(pr['frame']))
        for rev in (False, True):                      # forward from the first prompt, then backward to frame 0
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=first, reverse=rev):
                ms = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                for j, oid in enumerate(sess.obj_ids):
                    out[names[oid - 1]][o.frame_idx] = ms[j, 0].cpu().numpy()
            log(f'[track] {"backward" if rev else "forward"} done {time.time() - t0:.0f} s')
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def qc(masks, count, score, valid):
    out = {}
    for p, m in masks.items():
        area = m[:, valid].mean(1)
        present = area > 0.002
        tiou = [iou(m[k], m[k + 1]) for k in range(len(m) - 1) if present[k] and present[k + 1]]
        da = np.abs(np.diff(area))[present[:-1] & present[1:]] / np.maximum(area[:-1][present[:-1] & present[1:]], 1e-6)
        out[p] = dict(frames_present=round(float(present.mean()), 3), area_mean=round(float(area[present].mean()), 4) if present.any() else 0,
                      temporal_iou_median=round(float(np.nanmedian(tiou)), 3) if tiou else None,
                      temporal_iou_p10=round(float(np.nanpercentile(tiou, 10)), 3) if tiou else None,
                      area_jump_p90=round(float(np.percentile(da, 90)), 3) if len(da) else None,
                      instances_median=float(np.median(count[p][present])) if present.any() else 0,
                      instances_max=int(count[p].max()), score_median=round(float(np.median(score[p][present])), 3) if present.any() else 0)
    return out


def compare_v1(masks, valid):
    """chole_a only: IoU per frame against the v1 SAM 2.1 (hand-prompted) masks of the same frames."""
    v1 = ROOT.parent / 'medical-agentic-sim' / 'outputs' / 'chole_a' / 'prep' / 'masks.npz'
    if not v1.exists():
        return None
    z = np.load(v1)
    m1 = np.unpackbits(z['masks'], axis=-1)[..., :valid.shape[1]].astype(bool) if z['masks'].dtype == np.uint8 else z['masks']
    names = json.loads((ROOT / 'r2s' / 'clips' / 'chole_a.json').read_text())['objects']
    names = list(names)
    pairs = dict(gallbladder='gallbladder', grasper='grasper', probe='probe', liver='liver')
    res = {}
    for p, m in masks.items():
        key = next((v for k, v in pairs.items() if k in p.lower()), None)
        if key is None or key not in names:
            continue
        ref = m1[:, names.index(key)]
        res[p] = dict(v1_object=key, iou_median=round(float(np.nanmedian([iou(m[k] & valid, ref[k] & valid) for k in range(len(m))])), 3))
    return res


def sheet(frames, masks, path, n_tiles=8):
    ids = np.linspace(0, len(frames) - 1, n_tiles).astype(int)
    tiles = []
    for k in ids:
        img = frames[k].astype(np.float32).copy()
        for c, (p, m) in zip(COLORS, masks.items()):
            img[m[k]] = 0.55 * img[m[k]] + 0.45 * np.array(c, np.float32)
            cnt = cv2.findContours(m[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            img = cv2.drawContours(np.ascontiguousarray(img), cnt, -1, c, 1)
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.putText(img, f'frame {k}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, n_tiles, 4)], 0)
    y = 16
    for c, p in zip(COLORS, masks):
        cv2.putText(grid, p, (grid.shape[1] - 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        y += 16
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def grid_keyframes(clip, n_keys=5, step=40):
    """Keyframes with a labelled pixel grid for the prompting agent: outputs/t2s/<clip>/seg/keyframes/k<frame>.jpg."""
    frames, valid, info = D.load(clip)
    d = D.OUT / clip / 'seg' / 'keyframes'
    d.mkdir(parents=True, exist_ok=True)
    keys = np.linspace(0, len(frames) - 1, n_keys + 2)[1:-1].astype(int).tolist()
    for k in keys:
        img = cv2.resize(frames[k], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)     # 2x so the labels stay legible
        H, W = frames[k].shape[:2]
        for x in range(0, W, step):
            cv2.line(img, (2 * x, 0), (2 * x, 2 * H - 1), (255, 255, 255), 1)
            cv2.putText(img, str(x), (2 * x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        for y in range(0, H, step):
            cv2.line(img, (0, 2 * y), (2 * W - 1, 2 * y), (255, 255, 255), 1)
            cv2.putText(img, str(y), (3, 2 * y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(d / f'k{k:04d}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(d / f'k{k:04d}_plain.jpg'), cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (d / 'keys.json').write_text(json.dumps(dict(keys=keys, size=[int(frames.shape[2]), int(frames.shape[1])], grid_px=step,
                                                 note='grid images are 2x upscaled; labels are original-frame pixels')))
    return keys


def keep_points(prev, name, k, neg, W, n=2, clear=50):
    """Positive points for an object at frame k from its previous mask: the deepest interior points at least
    `clear` px from every negative point (so a repair that cuts a neighbour away does not delete the object)."""
    if name not in prev.files:
        return []
    m = np.unpackbits(prev[name][k], axis=-1)[..., :W].astype(np.uint8)
    if not m.any():
        return []
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    for x, y in neg:
        m[(xx - x) ** 2 + (yy - y) ** 2 < clear ** 2] = 0
    out = []
    for _ in range(n):
        if not m.any():
            break
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        out.append([int(x), int(y)])
        m[(xx - x) ** 2 + (yy - y) ** 2 < (2 * clear) ** 2] = 0
    return out


def combine(clip, ver, min_frames=0.0, per_shot=False):
    """vNN = every object tracked from agent point prompts in ONE SAM 3 tracker session: the instrument agent's
    instruments (seg/instrument_prompts.json) and the prompting agent's organs / tissue (seg/prompts.json).
    Instruments are in front of tissue (their pixels are removed from the tissue masks)."""
    frames, valid, info = D.load(clip)
    out = D.OUT / clip / 'seg' / ver
    out.mkdir(parents=True, exist_ok=True)
    seg = D.OUT / clip / 'seg'
    pj = json.loads((seg / 'prompts.json').read_text())
    ij = json.loads((seg / 'instrument_prompts.json').read_text()) if (seg / 'instrument_prompts.json').exists() else {'instruments': {}}
    ins = {f'instrument_{k}': list(v['prompts']) for k, v in ij['instruments'].items() if v.get('prompts')}
    org = {nm: list(o['prompts']) for nm, o in pj['objects'].items() if o.get('prompts')}
    blank, subtract = {}, {}
    W = frames.shape[2]
    prev = None
    for rp in sorted(seg.glob('repairs_r*.json')):          # QA repairs: extra prompts, new objects, blanked ranges
        rj = json.loads(rp.read_text())
        rv = rj.get('reviewed_version')                  # keep-points come from the version the QA looked at
        prev = np.load(seg / rv / 'masks.npz') if rv and (seg / rv / 'masks.npz').exists() else prev
        for a in rj.get('add_prompts', []):
            tgt = ins if a['object'].startswith('instrument_') else org
            pos = a.get('pos', [])
            if not pos and prev is not None:          # a negative-only prompt makes SAM drop the object (r02):
                pos = keep_points(prev, a['object'], a['frame'], a.get('neg', []), W)    # keep it with positives
            tgt.setdefault(a['object'], []).append(dict(frame=a['frame'], pos=pos, neg=a.get('neg', [])))
        for o, rr in rj.get('blank', {}).items():
            blank.setdefault(o, []).extend(rr)
        for o, others in rj.get('subtract', {}).items():
            subtract.setdefault(o, []).extend(others)
    for d in (ins, org):                                     # a tracker object needs at least one positive point
        for nm in [nm for nm, pr in d.items() if not any(q['pos'] for q in pr)]:
            d.pop(nm)
    t0 = time.time()
    n = len(frames)
    shots = ij.get('shots') if per_shot else None
    if shots and isinstance(shots, dict):
        shots = [v['frames'] for v in shots.values()]
    if shots:                                                # memory reset at every cut: track each shot on its own
        tracked = {nm: np.zeros((n,) + valid.shape, bool) for nm in {**ins, **org}}
        for a, b in shots:
            sub = {nm: [dict(q, frame=q['frame'] - a) for q in pr if a <= q['frame'] <= b] for nm, pr in {**ins, **org}.items()}
            sub = {nm: pr for nm, pr in sub.items() if any(q['pos'] for q in pr)}
            if sub:
                r = track_points(frames[a:b + 1], sub)
                for nm, m in r.items():
                    tracked[nm][a:b + 1] = m
    else:
        tracked = track_points(frames, {**ins, **org})
    for o, rr in blank.items():
        if o in tracked:
            for a, b in rr:
                tracked[o][a:b + 1] = False
    union = np.zeros((n,) + valid.shape, bool)
    masks = {}
    for nm in ins:
        masks[nm] = tracked[nm] & valid
        union |= masks[nm]
    for nm in org:
        masks[nm] = tracked[nm] & valid & ~union
    for nm, others in subtract.items():                     # explicit precedence between overlapping tissues
        for o in others:
            if nm in masks and o in masks:
                masks[nm] &= ~masks[o]
    for nm in ins:
        pass
    np.savez_compressed(out / 'masks.npz', **{p: np.packbits(m, axis=-1) for p, m in masks.items()})
    cnt = {p: m.any((1, 2)).astype(np.int16) for p, m in masks.items()}
    res = dict(clip=clip, version=ver, source=dict(organs='prompts.json', instruments='instrument_prompts.json'), n=n,
               width=int(frames.shape[2]), seconds=round(time.time() - t0, 1), objects=list(masks),
               repairs=[p.name for p in sorted(seg.glob('repairs_r*.json'))], per_shot=bool(shots),
               confidence={**{nm: pj['objects'].get(nm, {}).get('confidence', 'repair') for nm in org},
                           **{f'instrument_{k}': v.get('confidence') for k, v in ij['instruments'].items() if v.get('prompts')}},
               shots=ij.get('shots') or pj.get('shots'),
               qc=qc(masks, cnt, {p: np.ones(n, np.float32) for p in masks}, valid))
    if clip == 'chole_a':
        res['vs_v1_sam21'] = compare_v1(masks, valid)
    (out / 'qc.json').write_text(json.dumps(res, indent=1))
    sheet(frames, masks, out / 'sheet.jpg')
    return res


def probe_text(clip, ver='v01'):
    """What SAM 3 recognises from the spec's names alone: per prompt, presence score and best instance score / area on
    the 5 prompting keyframes (image model). Cheap documentation of the text-only route (outputs/t2s/<clip>/seg/vNN/)."""
    from transformers import Sam3Model, Sam3Processor
    frames, valid, info = D.load(clip)
    spec = json.loads((D.OUT / clip / 'spec' / 'scene_spec.json').read_text())
    prompts = prompts_of(spec)
    keys = json.loads((D.OUT / clip / 'seg' / 'keyframes' / 'keys.json').read_text())['keys']
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = Sam3Model.from_pretrained(MODEL).to(dev).eval()
    proc = Sam3Processor.from_pretrained(MODEL)
    res = {}
    for p in prompts:
        rows = []
        for k in keys:
            inp = proc(images=frames[k], text=p, return_tensors='pt').to(dev)
            with torch.no_grad():
                o = model(**inp)
            pres = float(o.presence_logits.sigmoid().item()) if getattr(o, 'presence_logits', None) is not None else None
            r = proc.post_process_instance_segmentation(o, threshold=0.5, mask_threshold=0.5, target_sizes=[frames[k].shape[:2]])[0]
            sc = r['scores'].cpu().numpy()
            rows.append(dict(frame=k, presence=round(pres, 3) if pres is not None else None, n=int(len(sc)),
                             best=round(float(sc.max()), 3) if len(sc) else 0.0,
                             area=round(float(r['masks'][int(np.argmax(sc))].float().mean()), 3) if len(sc) else 0.0))
        res[p] = dict(found_in=sum(r['n'] > 0 for r in rows), of=len(rows), presence_median=float(np.median([r['presence'] for r in rows])),
                      frames=rows)
    out = D.OUT / clip / 'seg' / ver
    out.mkdir(parents=True, exist_ok=True)
    (out / 'text_probe.json').write_text(json.dumps(dict(clip=clip, threshold=0.5, prompts=res), indent=1))
    return res

def main(argv):
    clip = argv[0]
    if '--combine' in argv:
        print(json.dumps(combine(clip, next((a for a in argv[1:] if a.startswith('v')), 'v02'), per_shot='--per-shot' in argv), indent=1))
        return
    ver = next((a for a in argv[1:] if a.startswith('v')), 'v01')
    frames, valid, info = D.load(clip)
    spec = json.loads((D.OUT / clip / 'spec' / 'scene_spec.json').read_text())
    prompts = argv[argv.index('--prompts') + 1].split(',') if '--prompts' in argv else prompts_of(spec)
    out = D.OUT / clip / 'seg' / ver
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    masks, count, score = segment(frames, prompts)
    for p in masks:
        masks[p] &= valid
    np.savez_compressed(out / 'masks.npz', **{p: np.packbits(m, axis=-1) for p, m in masks.items()})
    np.savez_compressed(out / 'instances.npz', **{f'{p}__count': count[p] for p in prompts}, **{f'{p}__score': score[p] for p in prompts})
    res = dict(clip=clip, version=ver, prompts=prompts, n=len(frames), width=int(frames.shape[2]), seconds=round(time.time() - t0, 1),
               qc=qc(masks, count, score, valid))
    if clip == 'chole_a':
        res['vs_v1_sam21'] = compare_v1(masks, valid)
    (out / 'qc.json').write_text(json.dumps(res, indent=1))
    sheet(frames, masks, out / 'sheet.jpg')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])


def qa_strips(clip, ver, n=12):
    """For the visual QA agent: one row per object, n frames across the clip, the object's mask tinted on the frame."""
    frames, valid, info = D.load(clip)
    z = np.load(D.OUT / clip / 'seg' / ver / 'masks.npz')
    W = frames.shape[2]
    ids = np.linspace(0, len(frames) - 1, n).astype(int)
    rows = []
    for c, nm in zip(COLORS * 3, z.files):
        m = np.unpackbits(z[nm], axis=-1)[..., :W].astype(bool)
        tiles = []
        for k in ids:
            img = frames[k].astype(np.float32).copy()
            img[m[k]] = 0.45 * img[m[k]] + 0.55 * np.array(c, np.float32)
            cnt = cv2.findContours(m[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            img = cv2.drawContours(np.ascontiguousarray(img), cnt, -1, c, 2)
            t = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (192, int(192 * frames.shape[1] / W)))
            cv2.putText(t, str(k), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append(t)
        row = np.concatenate(tiles, 1)
        bar = np.full((18, row.shape[1], 3), 25, np.uint8)
        cv2.putText(bar, nm, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        rows += [bar, row]
    path = D.OUT / clip / 'seg' / ver / 'strips.jpg'
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path
