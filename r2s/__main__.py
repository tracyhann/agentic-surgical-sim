"""python -m r2s <clip> <stage> [conditions...]

stages
  prep      source frames -> SAM 2 masks, Depth Anything V2 depth, scope motion, metric calibration   (prep/)
  scene     panorama textures, backdrop, organ observations, strands, instrument ports and actions   (scene/)
  organ     build the organ body of each condition                                                   (<cond>/organ.npz)
  sim       simulate each condition                                                                  (<cond>/traj.npz, scope.mp4)
  eval      metrics of each condition and of the static baseline                                     (<cond>/eval.json)
  render    2x2 video (video, scope, two orbit views) of each condition                              (<cond>/views.mp4)
  compare   condition grid video, stills and summary table                                           (compare/)
  all       everything above that is missing
conditions: measured primitive template template_fit (default: all)
"""
import sys
import os
import time
import numpy as np
from .config import Clip, CONDITIONS, save_json, load_json
from . import source, perception, scene, organ as ORG, sim as SIM, evaluate as EV, render as RD


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def prep(clip, frames):
    if not (clip.prep / 'masks.npz').exists():
        perception.segment(clip, frames)
    masks = perception.load_masks(clip)
    if not (clip.prep / 'disp.npy').exists():
        perception.depth_rel(clip, frames)
    if not (clip.prep / 'cams.npz').exists():
        Zc = None
        if clip['camera'].get('motion') == '6dof':      # the 6-DoF tracker needs metric depth: calibrate first
            if not (clip.prep / 'metric.json').exists():
                perception.calibrate(clip, np.load(clip.prep / 'disp.npy').astype(np.float32), masks,
                                     np.full(len(frames), clip.camera().F))
            Zc = perception.metric_depth(clip)
        cams = scene.track_camera(clip, frames, masks, Zc)
        np.savez(clip.prep / 'cams.npz', **cams)
    cams = dict(np.load(clip.prep / 'cams.npz'))
    if clip.get('multiview'):                      # joint keyframe solution: cameras (6 DoF, one focal length) + depth
        from . import multiview
        if not (clip.prep / 'multiview' / 'frames.npz').exists():
            if not (clip.prep / 'metric.json').exists():
                perception.calibrate(clip, np.load(clip.prep / 'disp.npy').astype(np.float32), masks, cams['f'])
            multiview.solve(clip.name, mode=clip['multiview'], log=log)
            multiview.per_frame(clip)
        mv = dict(np.load(clip.prep / 'multiview' / 'frames.npz'))
        cams = {k: mv[k] for k in ('R', 'f', 'pos', 'params', 'fit_err_px')}
    if not (clip.prep / 'metric.json').exists():
        perception.calibrate(clip, np.load(clip.prep / 'disp.npy').astype(np.float32), masks, cams['f'])
    return masks, cams


def observations(clip):
    z = np.load(clip.scene / 'scene.npz')
    obs = {k: z[k] for k in ('organ_mask', 'organ_visible', 'z0', 'thick')}
    M, (H, W) = int(z['M']), obs['z0'].shape
    obs['zvis'] = np.minimum(z['zb'][M:M + H, M:M + W], np.where(obs['organ_mask'], obs['z0'], np.inf))
    return obs


MULTI_KEYFRAME = ('template_mv', 'template_mvd')      # experiment conditions, run only when named


def keyframe_views(clip, c, masks, Z, every=5):
    """Later keyframes for the multi-keyframe template fit (organ.KeyView).
    template_mv: keyframes where the frame-0 organ held still keeps >= 75 % of its frame-0 outline overlap (rigid fit);
    template_mvd: every 10th frame where the 'template' simulation matches the outline (IoU >= 0.6), displaced by
    that simulation's motion."""
    S = SIM.load_scene(clip)
    cams = dict(R=S['cam_R'], f=S['cam_f'], **({'pos': S['cam_pos']} if 'cam_pos' in S else {}))
    n = len(masks)
    if c == 'template_mv':
        curve = load_json(clip.cond_dir('measured') / 'eval.json')['static']['iou_curve']
        sim = None
        cand = list(range(every, n, every))
        keep = [k for k in cand if curve[k // 5] >= 0.75 * curve[0]]
    else:
        td = clip.cond_dir('template')
        curve = load_json(td / 'eval.json')['sim']['iou_curve']
        sim = (SIM.load_organ(td / 'organ.npz')['X'], np.load(td / 'traj.npz')['flex'])
        keep = [k for k in range(10, n, 10) if curve[k // 5] >= 0.6]
    assert len(curve) == (n + 4) // 5, 'iou curve does not cover every 5th frame'
    log(f'[organ] {c}: keyframes {keep}')
    return ORG.keyframe_views(clip, cams, masks, Z, keep, [1.0 / max(len(keep), 1)] * len(keep), sim=sim,
                              fps=clip['fps'], dt=SIM.PARAMS['dt'])


def conditions_of(clip, args):
    conds = [a for a in args if a in CONDITIONS + MULTI_KEYFRAME] or list(clip.get('conditions', CONDITIONS))
    return [c for c in conds if c == 'measured' or c == 'primitive' or clip['objects'][clip.organ].get('template')]


def main(argv):
    name, stage, rest = argv[0], argv[1], argv[2:]
    clip = Clip(name)
    t0 = time.time()
    frames = source.frames(clip)
    log(f'{name}: {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}')
    stages = ['prep', 'scene', 'organ', 'sim', 'eval', 'render', 'compare'] if stage == 'all' else [stage]
    conds = conditions_of(clip, rest)
    force = '--force' in rest or stage != 'all'
    masks, cams = prep(clip, frames)
    Z = perception.metric_depth(clip)
    cv = None
    for st in stages:
        if st == 'scene' and (force or not (clip.scene / 'scene.npz').exists()):
            meta = scene.build(clip, frames, masks, Z, cams, log=log)
            log('[scene]', {k: meta[k] for k in ('canvas_margin', 'organ_amodal_added_px', 'organ_depth_cm', 'bed_motion_max_px')})
        if st in ('organ', 'sim', 'eval', 'render', 'compare') and cv is None:
            cv = SIM.canvas_of(clip, SIM.load_scene(clip))
        for c in conds if st in ('organ', 'sim', 'eval', 'render') else []:
            d = clip.cond_dir(c)
            if st == 'organ' and (force or not (d / 'organ.npz').exists()):
                views = keyframe_views(clip, c, masks, Z) if c in MULTI_KEYFRAME else None
                o = ORG.build(cv, observations(clip), c, clip['objects'][clip.organ].get('template'), log=log, views=views)
                SIM.save_organ(d / 'organ.npz', o)
                log(f'[organ] {c}: {len(o["X"])} vertices, {len(o["tets"])} tets, {int(o["anchors"].sum())} anchored, {int(o["pinned"].sum())} pinned')
            if st == 'sim' and (force or not (d / 'traj.npz').exists()):
                SIM.run(clip, c, SIM.load_organ(d / 'organ.npz'), log=log)
            if st == 'eval' and (force or not (d / 'eval.json').exists()):
                run_eval(clip, c, frames, masks, Z)
            if st == 'render' and (force or not (d / 'views.mp4').exists()):
                RD.video_main(clip, c, frames, d / 'views.mp4')
                log(f'[render] {c}: views.mp4')
        if st == 'compare':
            compare(clip, conds, frames, Z)
    log(f'{name} {stage}: done in {time.time() - t0:.0f} s')


def run_eval(clip, c, frames, masks, Z):
    d = clip.cond_dir(c)
    o = SIM.load_organ(d / 'organ.npz')
    flex = np.load(d / 'traj.npz')['flex']
    S = SIM.load_scene(clip)
    cams = dict(R=S['cam_R'], f=S['cam_f'], **({'pos': S['cam_pos']} if 'cam_pos' in S else {}))
    tp = clip.prep / 'organ_tracks.npz'
    if not tp.exists():                            # conditions are evaluated in parallel: write atomically
        P, seed = EV.organ_tracks(frames, masks, clip)
        tmp = tp.with_name(f'{tp.stem}.{os.getpid()}.tmp.npz')
        np.savez_compressed(tmp, P=P, seed=seed)
        os.replace(tmp, tp)
    z = np.load(tp)
    tracks = (z['P'], z['seed'])
    refs = {}
    ref = source.reference_depth(clip)
    if ref is not None:
        refs[clip['reference_depth']['name']] = ref
    dt = SIM.PARAMS['dt']
    cv = SIM.canvas_of(clip, S)
    kw = dict(tracks=tracks, cv=cv, zb=S['zb'])
    res = dict(sim=EV.evaluate(clip, o, flex, frames, masks, Z, cams, dt, refs, **kw),
               static=EV.evaluate(clip, o, np.repeat(flex[:1], len(flex), 0), frames, masks, Z, cams, dt, refs, **kw))
    save_json(d / 'eval.json', res)
    keys = [k for k in res['sim'] if 'curve' not in k and k not in ('n_tracks', 'track_err_px_static')]
    log(f'[eval] {c}: ' + ', '.join(f'{k} {res["sim"][k]} (static {res["static"][k]})' for k in keys))


def compare(clip, conds, frames, Z):
    out = clip.out / 'compare'
    out.mkdir(exist_ok=True)
    have = [c for c in conds if (clip.cond_dir(c) / 'eval.json').exists()]
    table = {c: dict(load_json(clip.cond_dir(c) / 'eval.json'), organ=load_json(clip.cond_dir(c) / 'sim.json')) for c in have}
    save_json(out / 'summary.json', table)
    RD.video_conditions(clip, have, frames, Z, out / 'conditions.mp4')
    RD.stills(clip, have, frames, out / 'conditions.jpg')
    RD.organ_preview(clip, have, out / 'organs_rest.jpg')
    log(f'[compare] {clip.name}: {", ".join(have)} -> compare/')


if __name__ == '__main__':
    main(sys.argv[1:])
