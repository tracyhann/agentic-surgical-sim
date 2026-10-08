"""Step 4 of v2 (PLAN_V2.md): organs from a geometric primitive, fitted jointly to many frames and views, then tracked
per frame (4D). Generic: works for any organ named in a clip's scene_spec, no per-organ tuning or anatomical template.

    PYTHONPATH=. .venv/bin/python -m t2s.organ <clip> [organ ...] [--ver vNN]
    PYTHONPATH=. .venv/bin/python -m t2s.organ --selftest        synthetic superquadric scene (fit + tets + 4D)

-> outputs/t2s/<clip>/organs/<organ>/vNN/{model.npz, quality.json, NOTES.md, sheet.jpg, views3d.jpg, work/}
   (vNN = method version below, the same for every organ; an organ may start at a later version)

Pipeline (current method v15; CFG below records each version and why)
  0. selection  spec organs with role primary / secondary that have a SAM mask (same name, or the prompting agent's
                notes naming them); thin membranes and tubes are skipped (other agents) and reported, as are context
                organs. Volume rule from the consistency and the spec's structured fields: fluid-filled -> ~constant
                volume, unless an opening (kind != none) or a drain / decompress action targets the organ ('drained':
                volume may only fall over time); anything else 'solid' (elastic, local volume term only).
  1. masks      body = organ mask in the scope area minus instruments (dilated); unknown (no evidence either way) =
                instruments, outside the scope, other objects in front of the organ (video depth along the shared
                border nearer by > 3 mm), see-through objects (named / described as translucent), the junction band of
                structures the spec attaches to the organ, thin flaps of the organ mask (morphological opening);
                everything else = background. Pixels claimed by the organ and another object stay the organ's (and
                are logged as a segmentation conflict); frames whose mask area collapses are 'unobserved'.
  2. primitive  superquadric (centre, rotation, 3 radii, 2 exponents in 0.5-1.6) on an icosphere (radial form), fitted
                to the silhouettes (outside + coverage) and depth points of the keyframes around the start frame,
                then each keyframe's rigid pose tracked outward from it, then everything jointly over all keyframes
                (pose priors on size and change between keyframes; start keyframe = gauge). Video depth is trusted up
                to a per-frame scale (prior sd 10 %); depth points far behind the surface are taken as seen through
                it. Unseen thickness: prior radius along the view >= 0.8 x the smaller radius across it, largest /
                smallest radius <= 4; where the background agent's surface exists (outputs/t2s/<clip>/background/
                vNN/occupancy.npz) no surface point may lie behind it (rest fit and 4D).
  3. rest       Bernstein free-form deformation (5^3 control points in the primitive's frame; smoothness + size +
                thickness priors) on top, same data. Rest shape = the shape at the start keyframe's pose.
  4. tets       uniform BCC tet lattice inside the rest surface (v1 gallbladder v12: no slivers), ~1300 nodes.
  5. 4D         per frame: node positions with tet ARAP + local volume + the global-volume rule + inversion barrier +
                temporal term + per-frame depth scale, fitted to that frame's silhouette and depth; outward from the
                start frame; unobserved frames held (re-acquired from the nearest keyframe pose after >= 5 of them);
                temporal Gaussian.
  6. quality    IoU / boundary F / depth residual per third of the clip for the primitive, the static rest shape and
                the 4D mesh (own masks, raw SAM mask, organ + attached-structure masks); temporal smoothness, volume
                over time, mesh health, inverted tets (rest, 4D), share of vertices behind the background surface;
                sheet.jpg (4D over the video), views3d.jpg (rest
                mesh, 3 directions, attachment hints), work/baselines.jpg (outlines of all three models).
  attach hints  vertices near each attached structure's mask (median distance over frames), the hidden back, and
                instrument entry points from the instrument agent's 'puncture' record (seg/instrument_prompts.json).
"""
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from r2s import quality
from r2s.organ import boundary_faces
from r2s.tissue.gallbladder import (Surface, TetARAP, bcc_for_count, rho, rodrigues, tet_measures, tet_report,
                                    views3d)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 't2s'
torch.set_num_threads(4)                     # other agents share the machine
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)

# ---------------------------------------------------------------- method versions (cumulative, same for all organs)
BASE = dict(
    seg='v02',
    # masks
    ins_dilate=9,               # px: instrument masks dilated (SAM edges, shadows)
    front_margin_mm=3.0,        # another object is in front if its depth along the shared border is this much nearer
    border_px=15,               # width of the shared-border band for that test
    thin_frac=0.08,             # thin flaps: opening radius = thin_frac * equivalent radius of the organ mask
    junction_px=25,             # attached structures (spec connections): their pixels this close to the organ = unknown
    merge_parts=False,          # merge SAM objects the prompting agent says may be part of this organ (e.g. its neck)
    min_area=0.004,             # keyframes where the organ covers less of the scope area are not used in the rest fit
    # data terms (mm)
    hidden_mm=8.0, w_sil=2.0, w_cov=1.0, w_depth=1.5, depth_erode=3,
    see_mm=10.0,                # a depth point this far BEHIND the surface is seen through the organ: ignored
    w_dscale=0.0, dscale_sd=0.1,  # per-frame scale of the video depth (about the camera), prior sd (log); 0 = off
    # primitive
    sq_starts=4, sq_iters=(60, 250), sq_exp=(0.3, 1.7), w_iso=0.05, w_exp=0.02,
    # depth extent (unseen back): the radius along the mean view direction may not fall below thick_ratio x the
    # smaller in-plane radius (round cross-section; data alone prefer a plate on the visible surface)
    w_thick=0.0, thick_ratio=0.8,
    w_aspect=0.0, max_aspect=4.0,   # largest / smallest radius above max_aspect costs w_aspect per log^2 (no plates)
    w_pose=0.1, pose_r_deg=10.0, pose_t_mm=5.0, w_pose_smooth=0.3,
    pose_track=False,           # shape from a window around the start keyframe, keyframe poses tracked outward from
    track_win=2,                # it, then everything jointly (start keyframe pose = global pose; smoothness on the
                                # pose acceleration instead of the offset size: organs that turn a lot)
    # free-form deformation
    ffd_n=5, ffd_iters=250, ffd_smooth=3.0, ffd_size=0.3, ffd_pad=1.25,
    ffd_thick=False,            # thickness prior also on the deformed shape (extent along the view vs across it)
    # tets
    nodes=1300,
    # 4D
    start=None, iters_first=300, iters=110, lr=0.08, w_arap=22.0, w_temp=0.3, w_barrier=0.5, smooth_sigma=1.0,
    n_surf4d=7000, n_cov4d=1800, n_pts4d=1500,
    # volume rule per consistency: local det term, global term (+- tol free, stiff beyond)
    # band: allowed range of volume / rest; monotone: forward in time the volume may only fall (by > tol);
    # rate: free change per frame
    vol_rules=dict(fluid=dict(w_vol=5.0, w_gvol=5.0, gvol_tol=0.005, band=(0.995, 1.005), monotone=False, rate=0.005),
                   drained=dict(w_vol=5.0, w_gvol=5.0, gvol_tol=0.005, band=(0.5, 1.05), monotone=True, rate=0.01),
                   solid=dict(w_vol=20.0, w_gvol=0.0, gvol_tol=0.0, band=(0.0, 9.0), monotone=False, rate=1.0)),
    eval_step=3,
    vol_cumulative=False,       # monotone volume rule against the running extreme (else frame to frame)
    dropout_frac=0.0,           # mask area below this fraction of the clip median = mask dropout (frame unobserved)
    w_bg=0.0, bg_margin_mm=1.0, # surface samples behind the background agent's surface (+ margin) cost w_bg (mm, rho)
    reacquire_gap=1,            # unobserved frames in a row after which the 4D restarts from the nearest keyframe pose
    w_ext=0.0, ext_max=0.6,     # border completion: where the organ mask runs into the image / scope border, an ellipse
                                # fitted to the rest of its outline continues it outside the view (coverage targets,
                                # weight w_ext x the coverage weight; at most ext_max x the mask's equivalent diameter)
    one_opening=False,          # all entry episodes of the instrument agent's puncture record = one opening region
)
CFG = {
    'v01': dict(),
    # v01 with the parts the prompting agent says may belong to the organ merged into its mask (chole: neck_pedicle)
    'v02': dict(merge_parts=True),
    # v01/v02 primitives degenerated: a 4 mm plate (v02) and a box (exponents at 0.33, v01) on the visible surface ->
    # thickness prior along the view, exponents kept in 0.5-1.6 with a stronger pull to 1
    'v03': dict(merge_parts=False, w_thick=5.0, sq_exp=(0.5, 1.6), w_exp=1.0),
    # v03 rest too small and a compromise between frames: the geometry's video depth changes scale between frames
    # (frame 120 -> 130: median depth 86 -> 75 mm, same mask area, camera moved 3 mm) -> depth trusted up to a
    # per-frame scale (prior sd 10 %, keyframe mean anchored at 1), in the rest fit and in 4D
    'v04': dict(w_dscale=0.3),
    # chole_a v04: BF 0.25-0.47 (outline loose) -> softer ARAP, more iterations; the 'drained' volume was monotone
    # only frame to frame (0.5 % per frame accumulated: 1.04 mid-clip) -> against the running extreme; rest thinner
    # after the FFD (1.9 cm) -> thickness prior also on the deformed shape
    'v05': dict(ffd_thick=True, w_arap=15.0, iters=130, vol_cumulative=True),
    # keyframe poses tracked outward from the start keyframe (rotations of tens of degrees cannot be found by
    # gradient descent from one global pose: chole_derot); start keyframe = gauge; each tracking step starts from the
    # previous keyframe's pose and is pulled to it and (weakly) to zero
    'v06': dict(pose_track=True, w_pose=0.03, w_pose_smooth=0.1),
    # v06 with the objects the prompting agent says may be part of the organ merged into its mask (chole: the
    # neck_pedicle, 'merge the two masks if a single gallbladder including its neck is wanted'); thin parts of the
    # merged mask (the duct) stay unknown via the thin-flap opening
    'v07': dict(merge_parts=True),
    # chole_derot v06: the SAM mask drops out for ~35 frames (165-200, area 0.3 -> 0.0-0.04 of the scope) while the
    # gallbladder is in view; fitting the empty mask crushed the mesh (edge stretch 7.8, depth residual 20 mm after
    # it) -> frames whose mask area is < dropout_frac x the clip median are unobserved: keyframes skipped in the rest
    # fit, in 4D the mesh is held (elastic + temporal terms only) and re-acquired after the gap from the rest fit's
    # nearest keyframe pose. (The derot loss turned out to be a segmentation conflict, not a dropout: neck_pedicle
    # covers the gallbladder in 165-200 and the junction rule made it unknown -> pixels both masks claim stay the
    # organ's, conflicts are logged.)
    'v08': dict(merge_parts=False, dropout_frac=0.25),
    # variant of v08 (as v07 of v06): parts the prompting agent says may belong to the organ merged into its mask
    'v09': dict(merge_parts=True),
    # v09's primitive became a 2.4 mm plate (radii 1.2 / 21 / 28 mm): the thickness prior only looks along the mean
    # view at the start pose, and the tracked poses turned the plate -> aspect guard on the radii (any direction)
    'v10': dict(w_aspect=5.0, merge_parts=False),
    # variant of v10: parts the prompting agent says may belong to the organ merged into its mask
    'v11': dict(merge_parts=True),
    # same method as v10 / v11, refitted on the repaired segmentation seg/v04 (chole_a, chole_derot; 2026-10-08)
    'v12': dict(merge_parts=False, seg='v04'),
    'v13': dict(merge_parts=True),
    # background agent (liver_s4 background v08): 20 % of the liver v10 vertices lie up to 24 mm behind the observed
    # background (right rim, underside) -> the hidden back may not pass behind the background agent's surface (its
    # per-frame first-hit depth along every camera ray, occupancy.npz) in the rest fit and in 4D. The thickness prior
    # stays (0.8; with 0.5 the gallbladder thinned to 1.3 cm): where a background lies close behind, its penalty
    # (w_bg 2, rho in mm) outweighs the thickness prior (w_thick 5 per log^2), so a lobe tip can still be thin
    'v14': dict(merge_parts=False, w_bg=2.0),
    # chole_a v14: the 1-frame mask dropout at frame 90 triggered a re-acquisition from the keyframe pose at frame 89
    # (vertex jump 4.7 mm / frame) -> re-acquire only after a gap of >= reacquire_gap frames
    'v15': dict(reacquire_gap=5),
    # coordinator / assembly r03: the gallbladder ends where the image ends (8.8 ml, all outlines run along the bottom
    # edge; hydropic per the spec, v1 template ~28 ml) -> border completion (pixels outside the image / scope area were
    # already unknown, but nothing asked the shape to continue there); one shared puncture region; neck hint at the
    # grasper's jaw tip (instrument agent v01)
    'v16': dict(w_ext=0.5, one_opening=True),
}


def cfg_of(ver):
    c = json.loads(json.dumps(BASE))
    for v in sorted(CFG):
        if v <= ver:
            c.update(CFG[v])
    return c


# ---------------------------------------------------------------- 0. selection from the spec
SKIP_CONSISTENCY = ('membrane', 'tubular', 'tube', 'sheet', 'fold')
STOP = {'the', 'of', 'and', 'or', 'a', 'an', 'right', 'left', 'with', 'at', 'in'}


def _tokens(s):
    return set(re.findall(r'[a-z]+', (s or '').lower())) - STOP


def load_texts(clip, seg='v02'):
    d = OUT / clip
    spec = json.loads((d / 'spec' / 'scene_spec.json').read_text())
    qc = json.loads((d / 'seg' / seg / 'qc.json').read_text()) if (d / 'seg' / seg / 'qc.json').exists() else {}
    pp = d / 'seg' / 'prompts.json'
    prompts = json.loads(pp.read_text()) if pp.exists() else {}
    return spec, qc, prompts


def mask_descriptions(prompts):
    """{mask name: the prompting agent's text about it} (object notes + 'added_objects' lines)."""
    out = {m: o.get('notes', '') for m, o in prompts.get('objects', {}).items()}
    for line in prompts.get('added_objects') or []:
        m, _, txt = line.partition(':')
        out[m.strip()] = out.get(m.strip(), '') + ' ' + txt
    return out


def find_mask(name, masks, desc):
    """The SAM object that shows spec organ `name`: same name, one name containing the other, or the prompting
    agent's text for the object naming every word of it (e.g. 'cystic duct' -> neck_pedicle, 'neck + cystic duct')."""
    cand = [m for m in masks if not m.startswith('instrument_')]
    n = name.lower().strip()
    for m in cand:
        if m.lower() == n:
            return m
    for m in cand:
        ml = m.lower().replace('_', ' ')
        if n in ml or (len(ml) > 3 and ml in n):
            return m
    tok = _tokens(name)
    for m in cand:                   # multi-word names only ('liver' would match any text that mentions the liver)
        if len(tok) >= 2 and tok <= _tokens(desc.get(m, '') + ' ' + m.replace('_', ' ')):
            return m
    return None


def volume_rule(clip_spec, organ):
    """'fluid' (closed, fluid-filled), 'drained' (fluid-filled and the spec says it is opened / emptied in this clip)
    or 'solid' (anything else: elastic), with the spec sentences that decided it."""
    cons = (organ.get('consistency') or '').lower()
    if 'fluid' not in cons and 'cyst' not in cons and 'hollow' not in cons:
        return 'solid', [f"consistency '{organ.get('consistency')}'"]
    why = [f"consistency '{organ.get('consistency')}'"]
    drain_words = ('deflat', 'drain', 'decompress', 'aspirat', 'empt', 'loses volume', 'not conserved', 'puncture')
    txt = ' '.join(str(organ.get(k) or '') for k in ('expected_motion', 'notes', 'pathology')).lower()
    said = [w for w in drain_words if w in txt]
    if said:                       # free text is only reported (it may say 'decompression comes in the next phase')
        why.append(f'organ text mentions {said} (not used: only openings / actions decide)')
    hits = []
    for o in clip_spec.get('openings') or []:
        if o.get('organ') == organ['name'] and (o.get('kind') or 'none') not in ('none', ''):
            why.append(f"opening '{o.get('kind')}' declared ({o.get('confidence')})")
            hits.append('opening')
    for a in clip_spec.get('actions') or []:
        if organ['name'] in str(a.get('target', '')) and any(w in str(a.get('verb', '')).lower() for w in drain_words):
            why.append(f"action '{a.get('verb')}' on it")
            hits.append('action')
    return ('drained' if hits else 'fluid'), why


def seg_version(clip, names):
    """The latest seg/vNN whose object list is the one the views were loaded with (prep/masks.npz holds no version)."""
    for d in sorted((OUT / clip / 'seg').glob('v*/qc.json'), reverse=True):
        if json.loads(d.read_text()).get('objects') == list(names):
            return d.parent.name
    return None


def select_organs(clip, seg='v02', masks=None):
    spec, qc, prompts = load_texts(clip, seg)
    masks = masks if masks is not None else qc.get('objects', [])
    desc = mask_descriptions(prompts)
    fit, skipped = [], []
    for o in spec.get('organs', []):
        m = find_mask(o['name'], masks, desc)
        cons = (o.get('consistency') or '').lower()
        rec = dict(name=o['name'], role=o.get('role'), consistency=o.get('consistency'), mask=m)
        if o.get('role') not in ('primary', 'secondary'):
            skipped.append({**rec, 'reason': f"role '{o.get('role')}' (only primary / secondary organs are fitted)"})
        elif m is None:
            skipped.append({**rec, 'reason': 'no SAM mask names it'})
        elif any(w in cons for w in SKIP_CONSISTENCY):
            skipped.append({**rec, 'reason': f"consistency '{o.get('consistency')}': membrane / tube agents"})
        elif any(r['mask'] == m for r in fit):
            skipped.append({**rec, 'reason': f'mask {m} already fitted as another organ'})
        else:
            rule, why = volume_rule(spec, o)
            see_through = any(w in (cons + ' ' + str(o.get('notes', ''))).lower() for w in ('translucent', 'transparent', 'see-through'))
            fit.append({**rec, 'volume_rule': rule, 'volume_why': why, 'see_through': see_through})
    # SAM objects that are not organs of the spec but that the agent tied to one (e.g. neck_pedicle = neck + duct)
    return fit, skipped, spec, prompts


def merge_candidates(organ_name, organ_mask, masks, desc):
    """SAM objects that the prompting agent says may be part of this organ: the organ's notes name the object in a
    sentence with 'merge', or the object's notes call it the organ's neck / part of the organ."""
    out = []
    own = desc.get(organ_mask, '')
    for m in masks:
        if m == organ_mask or m.startswith('instrument_'):
            continue
        sent = [x for x in re.split(r'(?<=[.;])\s', own) if m in x]
        if any('merge' in x.lower() for x in sent):
            out.append(m)
            continue
        t = desc.get(m, '').lower()
        if re.search(rf'{re.escape(organ_name.lower())} (neck|part)|part of the {re.escape(organ_name.lower())}', t):
            out.append(m)
    return out


def attached_masks(spec, organ_name, masks, desc):
    """{mask: (partner name, connection type, confidence)} for spec connections of the organ that are not 'free'."""
    out = {}
    for cn in spec.get('connections', []):
        if organ_name not in (cn.get('a'), cn.get('b')) or cn.get('type') in ('free',):
            continue
        other = cn['b'] if cn['a'] == organ_name else cn['a']
        m = find_mask(other, masks, desc)
        if m and m not in out:
            out[m] = (other, cn.get('type'), cn.get('confidence'))
    return out


def see_through_objects(spec, masks, desc):
    """SAM objects that may be see-through (video depth on them is the depth of what is behind): named peritoneum /
    pleura / membrane / sheet, or called translucent / transparent in the spec or the prompting agent's notes.
    (A thin membrane per the spec is not enough: the diaphragm is a muscular sheet.)"""
    see = ('translucent', 'transparent', 'see-through', 'see through')
    out = []
    for o in spec.get('organs', []):
        txt = ' '.join(str(o.get(k) or '') for k in ('consistency', 'notes', 'pathology')).lower()
        if any(w in txt for w in see):
            m = find_mask(o['name'], masks, desc)
            if m:
                out.append(m)
    for m in masks:
        if m.startswith('instrument_'):
            continue
        if any(w in m.lower() for w in ('membrane', 'peritone', 'sheet', 'pleura')) or \
                any(w in desc.get(m, '').lower().split('.')[0] for w in see):
            out.append(m)
    return sorted(set(out))


# ---------------------------------------------------------------- 1. masks
def prepare_masks(V, organ, others, see_through, c, log, merge=(), junction=()):
    """body (n, H, W), unknown (n, H, W) and per-object lists of frames in which it was in front of the organ.
    merge: objects whose masks are added to the organ's; junction: attached structures (their pixels within
    junction_px of the organ are unknown: the boundary between organ and attached structure is not observable)."""
    n, H, W = V.n, V.H, V.W
    valid = getattr(V, 'valid', np.ones((H, W), bool))
    om = V.mask(organ).copy()
    for m in merge:
        om |= V.mask(m)
    others = [o for o in others if o not in merge]
    kj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c['junction_px'] + 1,) * 2)
    ins_names = [m for m in V.names if m.startswith('instrument_')] or list(getattr(V, 'instrument_names', []))
    ins = np.zeros((n, H, W), bool)
    for m in ins_names:
        ins |= V.mask(m)
    kd = np.ones((c['ins_dilate'], c['ins_dilate']), np.uint8)
    kb = np.ones((c['border_px'], c['border_px']), np.uint8)
    body = np.zeros((n, H, W), bool)
    unknown = np.zeros((n, H, W), bool)
    front = {o: [] for o in others}
    thin_px = np.zeros(n)
    overlap = {}
    eq_r = np.sqrt(max(np.median(om.sum((1, 2))), 1) / np.pi)
    r_thin = max(2, int(round(c['thin_frac'] * eq_r)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_thin + 1, 2 * r_thin + 1))
    for k in range(n):
        u = cv2.dilate(ins[k].astype(np.uint8), kd).astype(bool) | ~valid
        b = om[k] & ~u
        D = V.depth(k)
        for o in others:
            mo = V.mask(o)[k]
            ov = (mo & om[k]).sum()
            if ov > 0.3 * max(om[k].sum(), 1):
                overlap.setdefault(o, []).append(k)
            mo = mo & ~om[k]              # pixels both masks claim stay the organ's (SAM tracks can overlap)
            if not mo.any():
                continue
            if o in see_through:
                u |= mo
                continue
            if o in junction:
                u |= mo & cv2.dilate(b.astype(np.uint8), kj).astype(bool)
            ring = cv2.dilate(b.astype(np.uint8), kb).astype(bool) & mo & ~b
            nb = b & cv2.dilate(mo.astype(np.uint8), kb).astype(bool)
            if ring.sum() < 20 or nb.sum() < 20:
                continue
            if np.median(D[ring]) * 1000 < np.median(D[nb]) * 1000 - c['front_margin_mm']:
                u |= mo
                front[o].append(k)
        b &= ~u
        op = cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)
        thin = b & ~op
        thin_px[k] = thin.sum()
        u |= thin
        b = op
        n_c, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8))
        if n_c > 2:                                    # drop specks the cuts leave behind
            big = 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])
            keep = [i for i in big if st[i, cv2.CC_STAT_AREA] >= 0.05 * st[big[0], cv2.CC_STAT_AREA]]
            drop = b & ~np.isin(lab, keep)
            b &= ~drop
            u |= drop
        body[k], unknown[k] = b, u
    log(f'[masks] {organ} (+ merged {list(merge)}): thin-flap opening radius {r_thin} px (unknown {thin_px.mean():.0f} '
        f'px / frame); objects in front (frames): {({o: len(v) for o, v in front.items() if v})}; see-through: '
        f'{see_through}; junction unknown: {list(junction)}')
    for o, ks in overlap.items():
        runs = np.split(np.array(ks), np.nonzero(np.diff(ks) > 1)[0] + 1)
        log(f'[masks] SEGMENTATION CONFLICT: {o} covers > 30 % of the {organ} mask in {len(ks)} frames '
            f'{[(int(r[0]), int(r[-1])) for r in runs]} (kept as {organ})')
    return body, unknown, dict(merged=list(merge), overlap_frames={o: [int(x) for x in v] for o, v in overlap.items()},
                               front={o: v for o, v in front.items() if v}, thin_radius_px=r_thin,
                               thin_px_mean=round(float(thin_px.mean()), 1), see_through=see_through,
                               junction_unknown=list(junction))


def border_completion(body_k, valid, unk_k, ext_max=0.6, step=3):
    """Pixels OUTSIDE the image / scope area (x, y may be < 0 or >= W, H) where the organ continues: if the organ's
    outline runs into the image or scope border, an ellipse fitted to the rest of the outline (points away from the
    border, the scope edge and unknown pixels) is continued outside; kept only if the ellipse explains the visible
    mask (IoU > 0.6) and only up to ext_max x the mask's equivalent diameter beyond the border. (N, 2) or None."""
    H, W = body_k.shape
    b = body_k.astype(np.uint8)
    if b.sum() < 200:
        return None
    cnts = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    pts = max(cnts, key=len)[:, 0]
    edge = np.zeros((H, W), np.uint8)
    edge[:3], edge[-3:], edge[:, :3], edge[:, -3:] = 1, 1, 1, 1
    open_ = cv2.dilate((edge.astype(bool) | ~valid).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    unk = cv2.dilate(unk_k.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    at_border = open_[pts[:, 1], pts[:, 0]]
    closed = pts[~at_border & ~unk[pts[:, 1], pts[:, 0]]]
    if at_border.sum() < 10 or len(closed) < 20:
        return None
    (cx, cy), (a1, a2), ang = cv2.fitEllipse(closed.astype(np.float32))
    pad = int(max(H, W))
    canvas = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
    cv2.ellipse(canvas, ((cx + pad, cy + pad), (a1, a2), ang), 1, -1)
    inside = canvas[pad:pad + H, pad:pad + W].astype(bool) & valid
    vis = body_k.astype(bool)
    iou = (inside & vis).sum() / max((inside | vis).sum() - (inside & unk_k).sum(), 1)
    if iou < 0.6:
        return None
    region = canvas.astype(bool)
    region[pad:pad + H, pad:pad + W] &= ~valid                 # only where the view does not reach
    far = ext_max * 2 * np.sqrt(vis.sum() / np.pi)
    inview = np.zeros_like(canvas)
    inview[pad:pad + H, pad:pad + W] = valid
    dist = cv2.distanceTransform((1 - inview).astype(np.uint8), cv2.DIST_L2, 5)
    region &= dist <= far
    ys, xs = np.nonzero(region[::step, ::step])
    if len(xs) < 5:
        return None
    return np.c_[xs * step - pad, ys * step - pad].astype(np.float32)


class OwnMasks:
    """Views proxy: mask('own') = body, occluders() = unknown (for r2s.quality and contact sheets)."""

    def __init__(self, V, body, unknown):
        self.V, self.body, self.unknown = V, body, unknown

    def __getattr__(self, a):
        return getattr(self.V, a)

    def mask(self, name):
        return self.body if name == 'own' else self.V.mask(name)

    def occluders(self, k):
        return self.unknown[k]


# ---------------------------------------------------------------- observations of one frame (torch, mm)
class Obs:
    """Camera, distance to the allowed region (body | unknown), measured depth, body pixels (coverage) and visible
    body points at their measured depth (world, mm). use_depth=False: see-through organ, no depth points."""

    def __init__(self, V, k, body, unknown, n_cov=800, n_pts=500, erode=3, use_depth=True, seed=0, see_mm=10.0,
                 bg=None, bg_margin=1.0, ext=None):
        rng = np.random.default_rng(seed + k)
        self.k = k
        self.R, self.pos, self.f = T(V.R[k]), T(V.pos[k] * 1000), float(V.f[k])
        self.H, self.W = V.H, V.W
        allowed = body[k] | unknown[k]
        self.dt = T(cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 5))[None, None]
        D = V.depth(k).astype(np.float32) * 1000
        self.D = T(D)[None, None]
        ys, xs = np.nonzero(body[k])
        self.area = len(xs)
        i = rng.choice(len(xs), min(n_cov, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
        self.cov = T(np.c_[xs[i], ys[i]])
        er = cv2.erode(body[k].astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool) if erode else body[k]
        er &= np.isfinite(D) & (D > 1)
        ys, xs = np.nonzero(er)
        i = rng.choice(len(xs), min(n_pts, len(xs)), replace=False) if len(xs) and use_depth else np.zeros(0, int)
        self.pts = T(V.unproject(xs[i], ys[i], V.depth(k)[ys[i], xs[i]], k) * 1000) if len(i) else torch.zeros(0, 3)
        self.see_mm = see_mm
        self.ext = None               # border completion targets (pixels outside the view)
        if ext is not None and ext[k] is not None and len(ext[k]):
            e = ext[k]
            self.ext = T(e[rng.choice(len(e), min(n_cov // 2, len(e)), replace=False)])
        self.B = None                 # background first-hit depth (mm) along this frame's rays, coarse grid
        if bg is not None:
            b = np.asarray(bg[k], np.float32) * 1000
            self.B = T(np.where(np.isfinite(b), b, 1e6))[None, None]
            self.bg_margin = bg_margin
        self.zmed = float(np.median(D[body[k]])) if body[k].any() else 80.0
        self.px2mm = self.zmed / self.f

    def project(self, X):
        pc = (X - self.pos) @ self.R.T
        z = pc[:, 2].clamp(min=1.0)
        return self.f * pc[:, 0] / z + self.W / 2, self.f * pc[:, 1] / z + self.H / 2, pc[:, 2]

    def sample(self, img, u, v):
        g = torch.stack([u / (self.W - 1) * 2 - 1, v / (self.H - 1) * 2 - 1], -1)[None, None]
        return torch.nn.functional.grid_sample(img, g, align_corners=True, padding_mode='border')[0, 0, 0]

    def ext_loss(self, S):
        """Coverage of the border completion targets by projected surface samples (any sample in front of the camera)."""
        if self.ext is None:
            return S.sum() * 0
        u, v, z = self.project(S)
        q = torch.stack([u[z > 1], v[z > 1]], 1)
        return rho(torch.relu(torch.cdist(self.ext, q).min(1)[0] - 3.0) * self.px2mm).mean()

    def bg_loss(self, S):
        """Surface samples (all, hidden ones included) behind the background surface + margin (mm, rho, mean)."""
        if self.B is None:
            return S.sum() * 0
        u, v, z = self.project(S)
        inside = (u >= 0) & (u <= self.W - 1) & (v >= 0) & (v <= self.H - 1) & (z > 1)
        g = torch.stack([u / self.W * 2 - 1, v / self.H * 2 - 1], -1)[None, None]
        with torch.no_grad():
            zb = torch.nn.functional.grid_sample(self.B, g, mode='nearest', align_corners=False, padding_mode='border')[0, 0, 0]
        return (rho(torch.relu(z - zb - self.bg_margin)) * inside).mean()

    def losses(self, S, Nrm, hidden_mm=8.0, ls=None):
        """S (M, 3) surface samples (mm), Nrm (M, 3) outward normals -> (outside, coverage, depth), mm.
        ls: log scale of this frame's video depth (about the camera centre; None = 0)."""
        u, v, z = self.project(S)
        inside = (u >= 0) & (u <= self.W - 1) & (v >= 0) & (v <= self.H - 1) & (z > 1)
        sig = torch.exp(ls) if ls is not None else None
        with torch.no_grad():
            hidden = z > self.sample(self.D, u, v) * (1.0 if sig is None else float(sig)) + hidden_mm
        sel = inside & ~hidden
        zero = S.sum() * 0
        l_out = rho(self.sample(self.dt, u, v)[sel] * self.px2mm).sum() / max(len(S), 1) * 4 if sel.any() else zero
        q = torch.stack([u[inside], v[inside]], 1)
        if len(self.cov) and len(q):
            l_cov = rho(torch.relu(torch.cdist(self.cov, q).min(1)[0] - 2.0) * self.px2mm).mean()
        else:
            l_cov = zero
        with torch.no_grad():
            front = ((Nrm * (S - self.pos)).sum(1) < 0) & inside
        if len(self.pts) and front.sum() > 20:
            P = self.pts if sig is None else self.pos + sig * (self.pts - self.pos)
            d, i = torch.cdist(P, S[front]).min(1)
            with torch.no_grad():     # points far behind the surface along the view: seen through it, not on it
                behind = (P - self.pos).norm(dim=1) - (S[front][i] - self.pos).norm(dim=1)
                w = (behind < self.see_mm).float()
            l_dep = (rho(d) * w).sum() / w.sum().clamp(min=1.0)
        else:
            l_dep = zero
        return l_out, l_cov, l_dep


def data_loss(obs, S, N, c, use_depth=True, ls=None):
    tot, parts = 0.0, np.zeros(3)
    for o in obs:
        lo, lc, ld = o.losses(S, N, c['hidden_mm'], ls)
        tot = tot + c['w_sil'] * (lo + c['w_cov'] * lc) + (c['w_depth'] * ld if use_depth else 0.0)
        if o.B is not None and c['w_bg'] > 0:
            tot = tot + c['w_bg'] * o.bg_loss(S)
        if o.ext is not None and c['w_ext'] > 0:
            tot = tot + c['w_sil'] * c['w_cov'] * c['w_ext'] * o.ext_loss(S)
        parts += [float(lo), float(lc), float(ld)]
    return tot / len(obs), parts / len(obs)


# ---------------------------------------------------------------- 2. superquadric primitive
def icosphere(sub):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=sub)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def sq_local(Dt, s, e):
    """Superquadric points (local frame, mm) along unit directions Dt: radii exp(s), exponents e = (e1 along local
    z, e2 in the xy plane); inside-outside F(x) = (|x/a1|^(2/e2) + |y/a2|^(2/e2))^(e2/e1) + |z/a3|^(2/e1), and F is
    homogeneous of degree 2/e1, so the point on the surface along d is d * F(d)^(-e1/2)."""
    a = torch.exp(s)
    x2, y2, z2 = ((Dt / a) ** 2 + 1e-12).unbind(1)
    f = (x2 ** (1 / e[1]) + y2 ** (1 / e[1])) ** (e[1] / e[0]) + z2 ** (1 / e[0])
    return Dt * f[:, None] ** (-e[0] / 2)


def exps(q, c):
    lo, hi = c['sq_exp']
    return lo + (hi - lo) * torch.sigmoid(q + float(np.log((1 - lo) / (hi - 1))))   # q = 0 -> e = 1 (ellipsoid)


class Posed:
    """World positions of local shape points: global pose (rotvec r, centre t, mm) and per-keyframe offsets."""

    @staticmethod
    def world(Xl, r, t):
        return Xl @ rodrigues(r).T + t

    @staticmethod
    def offset(X, t, dr, dt):
        return (X - t) @ rodrigues(dr).T + t + dt


def dscale_prior(ls, c):
    """Per-frame log scale of the video depth: prior sd dscale_sd; the mean over keyframes anchored at 0."""
    return c['w_dscale'] * ((ls / c['dscale_sd']) ** 2).mean() + (ls.mean() / 0.02) ** 2


def pose_prior(dR, dT, c):
    """Size of the per-keyframe offsets (w_pose) and their change between consecutive keyframes (w_pose_smooth)."""
    deg = np.radians(c['pose_r_deg'])
    p = ((dR / deg) ** 2).sum(1).mean() + ((dT / c['pose_t_mm']) ** 2).sum(1).mean()
    s = (((torch.diff(dR, dim=0)) / deg) ** 2).sum(1).mean() + \
        ((torch.diff(dT, dim=0) / c['pose_t_mm']) ** 2).sum(1).mean() if len(dR) > 1 else 0.0
    return c['w_pose'] * p + c['w_pose_smooth'] * s


def init_starts(obs, view, n_starts):
    """Starting poses from the depth points of all keyframes: PCA frame, in-plane rolls about the view direction."""
    from scipy.spatial.transform import Rotation as Rot
    P = np.concatenate([getattr(o, 'pts_init', o.pts).numpy() for o in obs])
    c0 = np.median(P, 0)
    lo, hi = np.percentile(P, 5, 0), np.percentile(P, 95, 0)
    _, _, ax = np.linalg.svd(P - c0, full_matrices=False)
    w = view / np.linalg.norm(view)
    a1 = ax[0] - (ax[0] @ w) * w
    a1 /= np.linalg.norm(a1)
    out = []
    for roll in np.linspace(0, np.pi, n_starts, endpoint=False):
        x = np.cos(roll) * a1 + np.sin(roll) * np.cross(w, a1)
        y = np.cross(w, x)
        Rm = np.stack([x, y, w], 1)                                         # columns: local axes in world
        r = np.maximum(np.percentile(np.abs((P - c0) @ Rm), 95, 0), 3.0)
        r[2] = min(r[0], r[1])                                              # depth extent: unobserved, as the smaller
        out.append((Rot.from_matrix(Rm).as_rotvec(), c0 + w * r[2] * 0.6, np.log(r[:3])))
    return out


def thickness_prior(r, s, view_t, c):
    """One-sided: log radius along the view below log(thick_ratio x smaller in-plane radius) costs w_thick per log^2."""
    if c['w_thick'] <= 0:
        return 0.0
    w = (rodrigues(r).T @ view_t) ** 2                     # share of each local axis along the view
    s_view = (w * s).sum()
    s_in = (s + 10 * w).min()                               # smallest radius among the axes across the view
    return c['w_thick'] * torch.relu(s_in + float(np.log(c['thick_ratio'])) - s_view) ** 2


def fit_primitive(obs, view, c, log, use_depth=True, j0=0):
    """Superquadric jointly on all keyframe observations; returns dict of numpy params (r, t, s, q, dR, dT, ls).
    j0: index of the start keyframe (pose_track: its offset is the gauge, fixed at 0)."""
    Dl, Fl = icosphere(3)
    Dt = T(Dl)
    view_t = T(view / np.linalg.norm(view))
    surf = Surface(Dl * 30, Fl, n=3000)
    K = len(obs)
    t0 = time.time()

    use_ds = c['w_dscale'] > 0 and use_depth

    gauge = torch.ones(K, 1)
    if c['pose_track']:
        gauge[j0] = 0.0

    def run(r0, t0_, s0, q0, dR0, dT0, ls0, sub, iters, free_exp, free_pose, lr=1.0):
        ls = torch.tensor(ls0, dtype=torch.float32, requires_grad=bool(free_pose and use_ds))
        r = torch.tensor(r0, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t0_, dtype=torch.float32, requires_grad=True)
        s = torch.tensor(s0, dtype=torch.float32, requires_grad=True)
        q = torch.tensor(q0, dtype=torch.float32, requires_grad=bool(free_exp))
        dR = torch.tensor(dR0, dtype=torch.float32, requires_grad=bool(free_pose))
        dT = torch.tensor(dT0, dtype=torch.float32, requires_grad=bool(free_pose))
        groups = [dict(params=[r], lr=0.02 * lr), dict(params=[t], lr=1.0 * lr), dict(params=[s], lr=0.02 * lr)]
        if free_exp:
            groups.append(dict(params=[q], lr=0.03 * lr))
        if free_pose:
            groups += [dict(params=[dR], lr=0.01 * lr), dict(params=[dT], lr=0.5 * lr)]
            if use_ds:
                groups.append(dict(params=[ls], lr=0.005 * lr))
        opt = torch.optim.Adam(groups)
        for it in range(iters):
            opt.zero_grad()
            e = exps(q, c)
            X = Posed.world(sq_local(Dt, s, e), r, t)
            L, parts = 0.0, np.zeros(3)
            for j in sub:
                Xk = Posed.offset(X, t, dR[j] * gauge[j], dT[j] * gauge[j]) if free_pose or c['pose_track'] else X
                S, N = surf(Xk)
                lj, pj = data_loss([obs[j]], S, N, c, use_depth, ls[j] if use_ds else None)
                L, parts = L + lj / len(sub), parts + pj / len(sub)
            L = L + c['w_iso'] * ((s - s.mean()) ** 2).sum() + c['w_exp'] * ((e - 1) ** 2).sum() + \
                thickness_prior(r, s, view_t, c) + \
                c['w_aspect'] * torch.relu(s.max() - s.min() - float(np.log(c['max_aspect']))) ** 2
            if free_pose:
                L = L + pose_prior(dR * gauge, dT * gauge, c) + (dscale_prior(ls, c) if use_ds else 0.0)
            L.backward()
            opt.step()
        return [x.detach().numpy().copy() for x in (r, t, s, q, dR * gauge, dT * gauge, ls)], float(L), parts

    def track(p, order):
        """Pose (and depth scale) of each keyframe in `order`, shape fixed, started from the previous one's."""
        r, t, s, q = (T(x) for x in p[:4])
        dR, dT, ls = p[4].copy(), p[5].copy(), p[6].copy()
        with torch.no_grad():
            X = Posed.world(sq_local(Dt, s, exps(q, c)), r, t)
        prev = None
        for j in order:
            a = torch.tensor(dR[prev] if prev is not None else dR[j], requires_grad=True)
            b = torch.tensor(dT[prev] if prev is not None else dT[j], requires_grad=True)
            g = torch.tensor(float(ls[prev] if prev is not None else ls[j]), requires_grad=use_ds)
            opt = torch.optim.Adam([dict(params=[a], lr=0.02), dict(params=[b], lr=0.5)] +
                                   ([dict(params=[g], lr=0.005)] if use_ds else []))
            for it in range(c['sq_iters'][0]):
                opt.zero_grad()
                S, N = surf(Posed.offset(X, t, a, b))
                L, _ = data_loss([obs[j]], S, N, c, use_depth, g if use_ds else None)
                L = L + c['w_pose'] * ((a / np.radians(c['pose_r_deg'])) ** 2).sum() + \
                    c['w_pose'] * ((b / c['pose_t_mm']) ** 2).sum()
                if prev is not None:      # change from the previous keyframe, as in the joint pose prior
                    L = L + c['w_pose_smooth'] * (((a - T(dR[prev])) / np.radians(c['pose_r_deg'])) ** 2).sum() + \
                        c['w_pose_smooth'] * (((b - T(dT[prev])) / c['pose_t_mm']) ** 2).sum()
                if use_ds:
                    L = L + c['w_dscale'] * (g / c['dscale_sd']) ** 2
                L.backward()
                opt.step()
            dR[j], dT[j], ls[j] = a.detach().numpy(), b.detach().numpy(), float(g)
            prev = j
        return [p[0], p[1], p[2], p[3], dR, dT, ls]

    z3 = np.zeros((K, 3), np.float32)
    if c['pose_track']:                      # shape from the keyframes around the start, then poses tracked outward
        win = [j for j in range(K) if abs(j - j0) <= c['track_win']]
        sub, allk = win, win
    else:
        sub, allk = list(range(0, K, 2)), list(range(K))
    cands = []
    for r0, t0_, s0 in init_starts([obs[j] for j in allk], view, c['sq_starts']):
        cands.append(run(r0, t0_, s0, np.zeros(2), z3, z3, np.zeros(K, np.float32), sub, c['sq_iters'][0], False, False))
    cands.sort(key=lambda x: x[1])
    log(f'[primitive] ellipsoid starts: losses {[round(x[1], 3) for x in cands]} ({time.time() - t0:.0f} s)')
    p, L, parts = cands[0]
    p, L, parts = run(*p, allk, c['sq_iters'][1] // 2, True, False)
    static = dict(r=p[0], t=p[1], s=p[2], e=exps(T(p[3]), c).numpy())
    log(f'[primitive] static superquadric: loss {L:.3f} parts(out,cov,depth mm) {np.round(parts, 3)} radii '
        f'{np.exp(p[2]).round(1)} mm exponents {exps(T(p[3]), c).numpy().round(2)} ({time.time() - t0:.0f} s)')
    if c['pose_track']:
        p = track(p, list(range(j0 + 1, K)))
        p = track(p, list(range(j0 - 1, -1, -1)))
        log(f'[primitive] tracked keyframe poses: rotation {np.degrees(np.linalg.norm(p[4], axis=1)).round(0).tolist()} '
            f'deg ({time.time() - t0:.0f} s)')
    p, L, parts = run(*p, list(range(K)), c['sq_iters'][1], True, True, lr=0.5)
    e = exps(T(p[3]), c).numpy()
    log(f'[primitive] + per-keyframe pose: loss {L:.3f} parts {np.round(parts, 3)} radii {np.exp(p[2]).round(1)} mm '
        f'exponents {e.round(2)}; offsets max {np.degrees(np.linalg.norm(p[4], axis=1)).max():.1f} deg / '
        f'{np.linalg.norm(p[5], axis=1).max():.1f} mm; depth scale {np.exp(p[6]).min():.3f}-{np.exp(p[6]).max():.3f} '
        f'({time.time() - t0:.0f} s)')
    return dict(r=p[0], t=p[1], s=p[2], q=p[3], dR=p[4], dT=p[5], ls=p[6], e=e, loss=L, parts=parts.tolist(),
                static=static)


# ---------------------------------------------------------------- 3. free-form deformation (Bernstein)
def bernstein_weights(U, n):
    """(N, n^3) weights of points U in [0, 1]^3 for a degree n-1 Bernstein lattice (x slowest)."""
    from math import comb
    d = n - 1
    B = [np.stack([comb(d, i) * U[:, a] ** i * (1 - U[:, a]) ** (d - i) for i in range(n)], 1) for a in range(3)]
    return np.einsum('ni,nj,nk->nijk', *B).reshape(len(U), -1)


def extent_prior(X, view_t, c):
    """Thickness prior on a mesh (mm): its extent along the view direction may not fall below thick_ratio x its
    smaller extent across the view (principal axes of the projected vertices)."""
    a = X @ view_t
    Q = X - a[:, None] * view_t
    Q = Q - Q.mean(0)
    with torch.no_grad():
        _, _, ax = torch.linalg.svd(Q, full_matrices=False)
    w_in = torch.stack([(Q @ ax[i]).max() - (Q @ ax[i]).min() for i in range(2)]).min()
    thick = a.max() - a.min()
    return c['w_thick'] * torch.relu(torch.log(c['thick_ratio'] * w_in) - torch.log(thick)) ** 2


def fit_ffd(obs, prim, c, log, use_depth=True, j0=0, view=None):
    """Lattice offsets (mm, local frame) on the primitive, with the global pose and per-keyframe offsets refined."""
    Dl, Fl = icosphere(4)
    e = T(prim['e'])
    Xl = sq_local(T(Dl), T(prim['s']), e).numpy()
    half = np.abs(Xl).max(0) * c['ffd_pad']
    Wl = T(bernstein_weights((Xl + half) / (2 * half), c['ffd_n']))
    Xl_t = T(Xl)
    surf = Surface(Xl, Fl, n=4000)
    n = c['ffd_n']
    off = torch.zeros(n ** 3, 3, requires_grad=True)
    r = torch.tensor(prim['r'], requires_grad=True)
    t = torch.tensor(prim['t'], requires_grad=True)
    dR = torch.tensor(prim['dR'], requires_grad=True)
    dT = torch.tensor(prim['dT'], requires_grad=True)
    use_ds = c['w_dscale'] > 0 and use_depth
    ls = torch.tensor(prim['ls'], dtype=torch.float32, requires_grad=use_ds)
    gauge = torch.ones(len(obs), 1)
    if c['pose_track']:
        gauge[j0] = 0.0
    opt = torch.optim.Adam([dict(params=[off], lr=0.3), dict(params=[r], lr=0.005), dict(params=[t], lr=0.3),
                            dict(params=[dR], lr=0.005), dict(params=[dT], lr=0.25)] +
                           ([dict(params=[ls], lr=0.0025)] if use_ds else []))
    t0 = time.time()
    for it in range(c['ffd_iters']):
        opt.zero_grad()
        X = Posed.world(Xl_t + Wl @ off, r, t)
        L, parts = 0.0, np.zeros(3)
        for j, o in enumerate(obs):
            S, N = surf(Posed.offset(X, t, dR[j] * gauge[j], dT[j] * gauge[j]))
            lj, pj = data_loss([o], S, N, c, use_depth, ls[j] if use_ds else None)
            L, parts = L + lj / len(obs), parts + pj / len(obs)
        g = off.reshape(n, n, n, 3)
        lap = sum((torch.diff(g, n=2, dim=a) ** 2).mean() for a in range(3))
        L = L + c['ffd_smooth'] * lap + c['ffd_size'] * (off ** 2).mean() / 100 + pose_prior(dR * gauge, dT * gauge, c) + \
            (dscale_prior(ls, c) if use_ds else 0.0)
        if c['ffd_thick'] and view is not None:
            L = L + extent_prior(X, T(view / np.linalg.norm(view)), c)
        L.backward()
        opt.step()
    log(f'[rest] + lattice: loss {float(L):.3f} parts {np.round(parts, 3)} max offset {float(off.abs().max()):.1f} mm '
        f'({time.time() - t0:.0f} s)')
    with torch.no_grad():
        Xr = Posed.world(Xl_t + Wl @ off, r, t).numpy()
    return dict(verts_mm=Xr, faces=Fl, off=off.detach().numpy(), r=r.detach().numpy(), t=t.detach().numpy(),
                dR=(dR * gauge).detach().numpy(), dT=(dT * gauge).detach().numpy(), ls=ls.detach().numpy(), loss=float(L),
                parts=parts.tolist(), max_offset_mm=float(off.abs().max()))


def primitive_mesh(prim, sub=4, static=False):
    """The superquadric at its global pose (static=True: the fit before per-keyframe offsets), metres."""
    Dl, Fl = icosphere(sub)
    p = prim['static'] if static else prim
    with torch.no_grad():
        X = Posed.world(sq_local(T(Dl), T(p['s']), T(p['e'])), T(p['r']), T(p['t'])).numpy()
    return X / 1000, Fl


# ---------------------------------------------------------------- 5. 4D
def surface_volume(X, F):
    return (X[F[:, 0]] * torch.cross(X[F[:, 1]], X[F[:, 2]], dim=1)).sum() / 6


def fit_4d(V, body, unknown, nodes, tets, faces, c, rule, log, start, init_pose=None, use_depth=True, ls_keys=None,
           observed=None, reacquire=None, bg=None, ext=None):
    """verts4d (n, N, 3) m, per-frame data parts, per-frame log depth scale. ls_keys: (keyframes, log scales) from
    the rest fit: prior centre of each frame's depth scale (interpolated)."""
    from scipy.ndimage import gaussian_filter1d
    X0 = T(nodes * 1000)
    arap = TetARAP(nodes * 1000, tets, uniform=True)
    vr = c['vol_rules'][rule]
    Fs = torch.as_tensor(faces, dtype=torch.long)
    vol0 = float(surface_volume(X0, Fs))
    surf = Surface(nodes * 1000, faces, n=c['n_surf4d'])
    out = np.zeros((V.n, len(nodes), 3), np.float32)
    parts = np.zeros((V.n, 3))
    use_ds = c['w_dscale'] > 0 and use_depth
    ls_c = np.interp(np.arange(V.n), *ls_keys) if (use_ds and ls_keys is not None) else np.zeros(V.n)
    ls_out = np.zeros(V.n)
    t0 = time.time()
    Xs = X0 if init_pose is None else init_pose
    observed = np.ones(V.n, bool) if observed is None else observed
    for seq, sgn in ((list(range(start, V.n)), 1), (list(range(start, -1, -1)), -1)):
        prev = prev2 = None
        vprev = vext = None
        gap = 0
        for k in seq:
            if not observed[k] and prev is not None:     # mask dropout: hold the shape, no data
                out[k] = prev.numpy() / 1000
                ls_out[k] = ls_out[k - sgn] if 0 <= k - sgn < V.n else 0.0
                parts[k] = np.nan
                gap += 1
                continue
            o = Obs(V, k, body, unknown, n_cov=c['n_cov4d'], n_pts=c['n_pts4d'], erode=c['depth_erode'],
                    use_depth=use_depth, see_mm=c['see_mm'], bg=bg, bg_margin=c['bg_margin_mm'], ext=ext)
            if gap >= c['reacquire_gap'] and reacquire is not None:   # after a long dropout: restart from the rest fit
                prev = prev2 = None
                Xs = reacquire(k)
                log(f'[4d] frame {k}: re-acquired after a mask dropout of {gap} frames')
            elif gap:
                prev2 = None                                  # short gap: continue from the held shape, no extrapolation
            gap = 0
            init = Xs.clone() if prev is None else (prev.clone() if prev2 is None else prev + 0.5 * (prev - prev2))
            X = init.clone().requires_grad_(True)
            ls = torch.tensor(float(ls_c[k]), requires_grad=use_ds)
            opt = torch.optim.Adam([X], lr=c['lr'])
            opt_s = torch.optim.Adam([ls], lr=0.003) if use_ds else None
            for it in range(c['iters_first'] if prev is None else c['iters']):
                opt.zero_grad()
                if opt_s:
                    opt_s.zero_grad()
                S, N = surf(X)
                lo, lc, ld = o.losses(S, N, c['hidden_mm'], ls if use_ds else None)
                L = c['w_sil'] * (lo + c['w_cov'] * lc) + (c['w_depth'] * ld if use_depth else 0.0)
                if o.B is not None and c['w_bg'] > 0:
                    L = L + c['w_bg'] * o.bg_loss(S)
                if o.ext is not None and c['w_ext'] > 0:
                    L = L + c['w_sil'] * c['w_cov'] * c['w_ext'] * o.ext_loss(S)
                ea, ev = arap(X)
                L = L + c['w_arap'] * ea + vr['w_vol'] * ev + c['w_barrier'] * arap.barrier_sum
                if prev is not None:
                    L = L + c['w_temp'] * ((X - prev) ** 2).sum(1).mean()
                if vr['w_gvol'] > 0:
                    ratio = surface_volume(X, Fs) / vol0
                    lo_, hi_ = vr['band']
                    dv = torch.relu(lo_ - ratio) + torch.relu(ratio - hi_)
                    if vprev is not None:
                        dv = dv + torch.relu(torch.abs(ratio - vprev) - vr['rate'])
                        if vr['monotone']:      # forward in time the volume may only fall (backward: only rise)
                            ref = vext if c['vol_cumulative'] else vprev
                            dv = dv + torch.relu(sgn * (ratio - ref) - vr['gvol_tol'])
                    L = L + vr['w_gvol'] * (dv / 0.01) ** 2
                if use_ds:            # depth scale near the rest fit's estimate for this time (sd half the prior)
                    L = L + c['w_dscale'] * ((ls - float(ls_c[k])) / (c['dscale_sd'] / 2)) ** 2
                L.backward()
                opt.step()
                if opt_s:
                    opt_s.step()
            Xd = X.detach()
            ls_out[k] = float(ls)
            parts[k] = [float(lo), float(lc), float(ld)]
            out[k] = Xd.numpy() / 1000
            vprev = float(surface_volume(Xd, Fs)) / vol0
            vext = vprev if vext is None else (min(vext, vprev) if sgn > 0 else max(vext, vprev))
            prev2, prev = prev, Xd
            if k % 25 == 0:
                log(f'[4d] frame {k}: out {parts[k][0]:.3f} cov {parts[k][1]:.3f} depth {parts[k][2]:.3f} '
                    f'arap {float(ea):.4f} vol {vprev:.3f} depth scale {np.exp(ls_out[k]):.3f} ({time.time() - t0:.0f} s)')
    if c['smooth_sigma'] > 0:
        out = gaussian_filter1d(out, c['smooth_sigma'], axis=0, mode='nearest').astype(np.float32)
    return out, parts, ls_out


# ---------------------------------------------------------------- 6. quality
def render(V, X, F, k):
    """quality.render (painter's order, flat depth per face) restricted to each face's bounding box (same output)."""
    q, z = V.project(X, k)
    H, W = V.H, V.W
    mask = np.zeros((H, W), bool)
    zbuf = np.full((H, W), np.inf, np.float32)
    zt = z[F].mean(1)
    ok = (z[F] > 1e-4).all(1)
    for fi in np.nonzero(ok)[0][np.argsort(-zt[ok])]:
        poly = q[F[fi]].astype(np.int32)
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        if x1 < 0 or x0 >= W or y1 < 0 or y0 >= H:
            continue
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W - 1), min(y1, H - 1)
        tmp = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillConvexPoly(tmp, poly - np.array([x0, y0], np.int32), 1)
        zb = zbuf[y0:y1 + 1, x0:x1 + 1]
        sel = tmp.astype(bool) & (zt[fi] < zb)
        zb[sel] = zt[fi]
        mask[y0:y1 + 1, x0:x1 + 1] |= sel
    return mask, zbuf


def mask_metrics(m, z, D, real, unk, scale=1.0):
    """IoU / boundary F (4 px) of the visible part of a rendered mesh (m, z; not > 6 mm behind the video depth D) and
    median |depth residual| (mm; raw video depth, and the video depth times this frame's fitted scale) against mask
    `real` with `unk` ignored (dilated 5x5, as r2s.quality)."""
    vis = m & (z < D * scale + 0.006)
    valid = ~cv2.dilate(unk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    a, b = vis & valid, real & valid
    out = dict(iou=float((a & b).sum() / max((a | b).sum(), 1)))
    k3 = np.ones((3, 3), np.uint8)
    bm = (vis.astype(np.uint8) - cv2.erode(vis.astype(np.uint8), k3)).astype(bool) & valid
    br = (real.astype(np.uint8) - cv2.erode(real.astype(np.uint8), k3)).astype(bool) & valid
    if bm.sum() and br.sum():
        dr = cv2.distanceTransform((~br).astype(np.uint8), cv2.DIST_L2, 3)
        dm = cv2.distanceTransform((~bm).astype(np.uint8), cv2.DIST_L2, 3)
        pr, rc = float((dr[bm] <= 4).mean()), float((dm[br] <= 4).mean())
        out['bf'] = 2 * pr * rc / max(pr + rc, 1e-9)
    else:
        out['bf'] = 0.0
    both = m & real & valid
    ok = both.sum() > 30
    out['depth_mm'] = float(np.median(np.abs(z[both] - D[both]))) * 1000 if ok else np.nan
    out['depth_scaled_mm'] = float(np.median(np.abs(z[both] - scale * D[both]))) * 1000 if ok else np.nan
    return out


def frame_metrics(V, X, F, k, real, unk, scale=1.0):
    m, z = render(V, X, F, k)
    return mask_metrics(m, z, V.depth(k), real, unk, scale)


def thirds(n):
    return [(int(a[0]), int(a[-1])) for a in np.array_split(np.arange(n), 3)]


COLS = ('iou', 'bf', 'depth_mm', 'depth_scaled_mm', 'iou_raw', 'bf_raw')


def band_table(V, models, body, unknown, raw, frames, bands, scale=None, alt=None):
    """{model: {band: {iou, bf, depth_mm, depth_scaled_mm, iou_raw, bf_raw}}}, per-frame arrays, and for each `alt`
    mask {label: {model: {band: {iou, bf}}}} (raw masks with only instruments / outside the scope ignored)."""
    per, per_alt = {}, {lab: {} for lab in (alt or {})}
    for name, (X4, F) in models.items():
        rows, arows = [], {lab: [] for lab in (alt or {})}
        for k in frames:
            X = X4[k] if X4.ndim == 3 else X4
            m, z = render(V, X, F, k)
            D = V.depth(k)
            sc = 1.0 if scale is None else float(scale[k])
            own = mask_metrics(m, z, D, body[k], unknown[k], sc)
            r = mask_metrics(m, z, D, raw[k], V.occluders(k), sc)
            rows.append([own['iou'], own['bf'], own['depth_mm'], own['depth_scaled_mm'], r['iou'], r['bf']])
            for lab, am in (alt or {}).items():
                ra = mask_metrics(m, z, D, am[k], V.occluders(k), sc)
                arows[lab].append([ra['iou'], ra['bf']])
        per[name] = np.array(rows, float)
        for lab in arows:
            per_alt[lab][name] = np.array(arows[lab], float)
    fr = np.array(frames)
    sels = {f'{a}-{b}': (fr >= a) & (fr <= b) for a, b in bands}
    sels['all'] = np.ones(len(fr), bool)

    def summ(x, cols):
        return {cname: (round(float(np.nanmedian(x[:, i])), 2) if 'depth' in cname else round(float(np.nanmean(x[:, i])), 4))
                if np.isfinite(x[:, i]).any() else None for i, cname in enumerate(cols)}
    tab = {name: {bl: summ(arr[sel], COLS) for bl, sel in sels.items()} for name, arr in per.items()}
    atab = {lab: {name: {bl: summ(arr[sel], ('iou', 'bf')) for bl, sel in sels.items()} for name, arr in d.items()}
            for lab, d in per_alt.items()}
    return tab, per, atab


def volume(X, F):
    a, b, cc = X[..., F[:, 0], :], X[..., F[:, 1], :], X[..., F[:, 2], :]
    return np.einsum('...ij,...ij->...i', a, np.cross(b, cc)).sum(-1) / 6


def tet_dets(X, tets):
    P = X[..., tets, :]
    D = np.stack([P[..., 1, :] - P[..., 0, :], P[..., 2, :] - P[..., 0, :], P[..., 3, :] - P[..., 0, :]], -1)
    return np.linalg.det(D)


def evaluate(V, rest, faces, tets, verts4d, body, unknown, raw, prim_static, prim_faces, c, log, alt=None, scale=None,
             observed=None):
    n = V.n
    frames = [k for k in range(0, n, c['eval_step']) if observed is None or observed[k]]   # mask dropouts excluded
    bands = thirds(n)
    models = {'4d': (verts4d, faces), 'static_rest': (rest, faces), 'primitive': (prim_static, prim_faces)}
    tab, per, alt_tab = band_table(V, models, body, unknown, raw, frames, bands, scale, alt)
    for lab in alt_tab:                          # raw IoU against other mask definitions (e.g. organ + neck)
        log(f'[quality] vs raw mask "{lab}": ' + ' | '.join(f"{m}: " + ', '.join(f"{b} {v['iou']:.3f}" for b, v in
                                                                              alt_tab[lab][m].items()) for m in alt_tab[lab]))
    for name in models:
        log(f'[quality] {name}: ' + ' | '.join(f"{b}: IoU {v['iou']:.3f} BF {v['bf']:.3f} depth {v['depth_mm']} / "
                                               f"{v['depth_scaled_mm']} mm (raw IoU {v['iou_raw']:.3f})"
                                               for b, v in tab[name].items()))
    q = dict(bands=tab, band_frames=[f'{a}-{b}' for a, b in bands], bands_other_masks=alt_tab,
             per_frame=dict(frames=frames, **{f'{m}_{col}': np.round(per[m][:, i], 4).tolist() for m in per
                                              for i, col in enumerate(COLS)}))
    vel = np.linalg.norm(np.diff(verts4d, axis=0), axis=2) * 1000
    acc = np.linalg.norm(np.diff(verts4d, n=2, axis=0), axis=2) * 1000
    cen = verts4d.mean(1)
    q['temporal'] = dict(max_vertex_speed_mm_per_frame=round(float(vel.max()), 3),
                         p99_vertex_speed_mm_per_frame=round(float(np.percentile(vel, 99)), 3),
                         mean_vertex_speed_mm_per_frame=round(float(vel.mean()), 3),
                         max_vertex_accel_mm_per_frame2=round(float(acc.max()), 3),
                         p99_vertex_accel_mm_per_frame2=round(float(np.percentile(acc, 99)), 3),
                         centroid_path_mm=round(float(np.linalg.norm(np.diff(cen, axis=0), axis=1).sum() * 1000), 2),
                         centroid_range_mm=(np.ptp(cen, 0) * 1000).round(2).tolist())
    v0 = volume(rest, faces)
    vt = volume(verts4d, faces) / v0
    q['volume'] = dict(rest_ml=round(float(v0) * 1e6, 2), ratio_min=round(float(vt.min()), 4),
                       ratio_max=round(float(vt.max()), 4),
                       ratio_per_band={f'{a}-{b}': round(float(vt[a:b + 1].mean()), 4) for a, b in bands},
                       ratio_every_10th=np.round(vt[::10], 4).tolist())
    d0 = tet_dets(rest, tets)
    dt = tet_dets(verts4d, tets)
    inv = (dt * np.sign(d0) <= 0)
    J = dt / d0
    E = np.unique(np.sort(np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), 1), axis=0)
    smax, s95 = quality.stretch(rest, verts4d, E)
    q['deformation'] = dict(inverted_tets_rest=int((d0 <= 0).sum()), inverted_tets_4d_max=int(inv.sum(1).max()),
                            frames_with_inverted_tets=int((inv.sum(1) > 0).sum()), min_volume_ratio_tet=round(float(J.min()), 4),
                            edge_stretch_max=round(float(smax.max()), 4), edge_stretch_p95_max=round(float(s95.max()), 4))
    rd = []
    for k in frames:
        A, B = rest, verts4d[k]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rm = Vt.T @ Dg @ U.T
        res = np.linalg.norm(B - ((A - ca) @ Rm.T + cb), axis=1) * 1000
        rd.append([np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))), np.linalg.norm(cb - ca) * 1000,
                   np.sqrt((res ** 2).mean())])
    rd = np.array(rd)
    q['rigid_vs_nonrigid'] = dict(rot_deg_max=round(float(rd[:, 0].max()), 1), shift_mm_max=round(float(rd[:, 1].max()), 1),
                                  nonrigid_rms_mm_median=round(float(np.median(rd[:, 2])), 2),
                                  nonrigid_rms_mm_max=round(float(rd[:, 2].max()), 2))
    q['mesh_health'] = quality.mesh_health(rest, faces, tets)
    q['tets'] = tet_report(rest, tets)
    log(f"[quality] temporal {q['temporal']}\n[quality] volume {q['volume']}\n[quality] deformation {q['deformation']}"
        f"\n[quality] mesh {q['mesh_health']}")
    return q, per, frames


def behind_background(V, clip, verts4d, faces, log, tol_mm=1.0):
    """Share of surface vertices more than tol_mm behind the newest background surface along the camera rays (the
    background agent's check), per third of the clip: mean / max share and the deepest vertex (mm)."""
    vers = sorted(p_.parent.name for p_ in (OUT / clip / 'background').glob('v*/occupancy.npz'))
    if not vers:
        return None
    o_ = np.load(OUT / clip / 'background' / vers[-1] / 'occupancy.npz')
    zb_all, sc = o_['bg_depth'], int(o_['scale'])
    surf = np.unique(faces)
    share, deep = np.full(V.n, np.nan), np.full(V.n, np.nan)
    for k in range(V.n):
        q_, z = V.project(verts4d[k][surf], k)
        xi, yi = np.floor(q_[:, 0] / sc).astype(int), np.floor(q_[:, 1] / sc).astype(int)
        ok = (xi >= 0) & (xi < zb_all.shape[2]) & (yi >= 0) & (yi < zb_all.shape[1])
        zb = np.full(len(z), np.nan)
        zb[ok] = zb_all[k, yi[ok], xi[ok]].astype(float)
        d = (zb - z) * 1000
        fin = np.isfinite(d)
        if fin.any():
            share[k] = float((d[fin] < -tol_mm).mean())
            deep[k] = float(-d[fin].min())
    out = dict(background_version=vers[-1], tol_mm=tol_mm,
               per_third={f'{a}-{b}': dict(share_mean=round(float(np.nanmean(share[a:b + 1])), 4),
                                           share_max=round(float(np.nanmax(share[a:b + 1])), 4),
                                           deepest_mm=round(float(np.nanmax(deep[a:b + 1])), 2)) for a, b in thirds(V.n)})
    log(f'[quality] behind background {vers[-1]} (> {tol_mm} mm): {out["per_third"]}')
    return out


# ---------------------------------------------------------------- attachments
def attach_hints(V, spec, organ, masks, desc, rest, faces, verts4d, frames, log, thr_mm=5.0, n_min=8, merged=()):
    """Surface vertices facing each connected structure that has a mask (spec connections other than 'free'):
    per vertex the median over frames (where the structure is visible) of its distance to the structure's
    back-projected pixels; vertices with median < thr_mm, or the n_min closest. Plus the hidden back (facing away
    from every camera: a support surface, not an anatomical attachment)."""
    import trimesh
    from scipy.spatial import cKDTree
    surf = np.unique(faces)
    idx, to, notes = [], [], []
    by_mask = {}
    for cn in spec.get('connections', []):
        if organ not in (cn.get('a'), cn.get('b')):
            continue
        other = cn['b'] if cn['a'] == organ else cn['a']
        typ = cn.get('type', '')
        if typ in ('free',):
            notes.append(f"{other}: '{typ}' ({cn.get('confidence')}) -> no attachment")
            continue
        m = find_mask(other, masks, desc)
        if m is None or m == organ:
            notes.append(f"{other}: '{typ}' ({cn.get('confidence')}), no mask of its own -> no hint")
            continue
        if m in merged:
            notes.append(f"{other}: '{typ}' ({cn.get('confidence')}), its mask {m} is merged into the organ -> no hint")
            continue
        by_mask.setdefault(m, []).append(f"{other} ('{typ}', {cn.get('confidence')})")
    for m, who in by_mask.items():
        D = []
        for k in frames:
            if V.mask(m)[k].sum() < 50:
                continue
            P = V.points(m, k, step=3, erode=1)
            if len(P) < 10:
                continue
            D.append(cKDTree(P).query(verts4d[k][surf])[0] * 1000)
        if not D:
            notes.append(f"{' + '.join(who)}: mask {m} never visible -> no hint")
            continue
        med = np.median(np.array(D), 0)
        sel = surf[med < thr_mm]
        how = f'median distance < {thr_mm} mm over {len(D)} frames'
        if len(sel) < n_min:
            sel = surf[np.argsort(med)[:n_min]]
            how = f'the {n_min} vertices with the smallest median distance ({np.sort(med)[n_min - 1]:.1f} mm) over {len(D)} frames'
        lab = f"{' + '.join(w.split(' (')[0] for w in who)} [{m}]"
        idx += sel.tolist()
        to += [lab] * len(sel)
        notes.append(f"{lab}: {len(sel)} vertices, {how}; spec: {'; '.join(who)}")
    vn = trimesh.Trimesh(rest, faces, process=False).vertex_normals
    view = np.mean(V.R[:, 2], 0)
    back = surf[(vn[surf] @ view) > 0.35]
    idx += back.tolist()
    to += ['hidden_back'] * len(back)
    notes.append(f'hidden_back: {len(back)} surface vertices facing away from every camera (normal . mean view > 0.35); '
                 'a support surface, not an anatomical attachment')
    for s_ in notes:
        log(f'[attach] {s_}')
    return np.array(idx, np.int32), np.array(to), notes


def opening_hints(V, clip, organ_name, mask, rest, faces, verts4d, log, radius_mm=4.0, one=False):
    """Surface vertices at an instrument entry the instrument agent recorded (seg/instrument_prompts.json 'puncture':
    instrument, frame ranges, per-frame entry pixel track): in each tracked frame the front-facing surface vertex
    whose projection is nearest the entry pixel; the hint = rest vertices within radius_mm of the median of those
    vertices' rest positions, per entry episode."""
    import trimesh
    p = OUT / clip / 'seg' / 'instrument_prompts.json'
    if not p.exists():
        return np.zeros(0, np.int32), np.zeros(0, str), []
    pu = json.loads(p.read_text()).get('puncture')
    if not pu or not pu.get('entry_track'):
        return np.zeros(0, np.int32), np.zeros(0, str), []
    surf = np.unique(faces)
    eps = [tuple(pu['frames'])] + ([tuple(pu['reinsertion']['frames'])] if pu.get('reinsertion') else [])
    if one and len(eps) > 1:          # one puncture site entered in several episodes (instrument agent)
        eps = [(min(a for a, _ in eps), max(b for _, b in eps))]
    idx, to, notes = [], [], []
    for j, (a, b) in enumerate(eps):
        hits = []
        for k, u, v in pu['entry_track']:
            if not a <= k <= b:
                continue
            X = verts4d[k]
            vn = trimesh.Trimesh(X, faces, process=False).vertex_normals
            q, z = V.project(X[surf], k)
            front = (vn[surf] * (X[surf] - V.pos[k])).sum(1) < 0
            d = np.linalg.norm(q - np.array([u, v]), axis=1) + np.where(front, 0, 1e9)
            i = np.argmin(d)
            if d[i] < 30:
                hits.append(surf[i])
        if not hits:
            notes.append(f"opening ({pu.get('instrument')}, frames {a}-{b}): no front vertex within 30 px of the entry track")
            continue
        c0 = np.median(rest[hits], 0)
        sel = surf[np.linalg.norm(rest[surf] - c0, axis=1) < radius_mm / 1000]
        lab = f"opening:{pu.get('instrument')}:{a}-{b}"
        idx += sel.tolist()
        to += [lab] * len(sel)
        spread = np.median(np.linalg.norm(rest[hits] - c0, axis=1)) * 1000
        notes.append(f"{lab}: {len(sel)} vertices within {radius_mm} mm of the median entry vertex ({len(hits)} tracked "
                     f"frames, spread {spread:.1f} mm, instrument agent confidence {pu.get('confidence')})")
    for s_ in notes:
        log(f'[attach] {s_}')
    return np.array(idx, np.int32), np.array(to), notes


def instrument_contact_hints(V, clip, rest, faces, verts4d, log, near_mm=12.0, radius_mm=4.0):
    """Surface vertices at the jaw tip of an instrument that stays at the organ (instrument agent's model: newest
    outputs/t2s/<clip>/instruments/vNN/model.npz '<name>__jaw_tip', world m): per frame the surface vertex nearest the
    jaw tip; if that distance is < near_mm in >= half the frames, the hint = rest vertices within radius_mm of the
    median of those vertices' rest positions. Label 'neck:<instrument>' when the instrument agent's notes say it holds
    the neck, else 'grasp:<instrument>' for graspers, else 'contact:<instrument>'."""
    from scipy.spatial import cKDTree
    vers = sorted((OUT / clip / 'instruments').glob('v*/model.npz'))
    if not vers:
        return np.zeros(0, np.int32), np.zeros(0, str), []
    m = np.load(vers[-1], allow_pickle=True)
    ip = OUT / clip / 'seg' / 'instrument_prompts.json'
    notes_ = json.loads(ip.read_text()).get('instruments', {}) if ip.exists() else {}
    surf = np.unique(faces)
    idx, to, notes = [], [], []
    for key in m.files:
        if not key.endswith('__jaw_tip'):
            continue
        name = key[:-len('__jaw_tip')]
        tip = m[key]
        hits, dist = [], []
        for k in range(0, V.n, 3):
            if not np.isfinite(tip[k]).all():
                continue
            d, i = cKDTree(verts4d[k][surf]).query(tip[k])
            dist.append(d * 1000)
            hits.append(surf[i])
        dist = np.array(dist)
        if not len(dist) or np.mean(dist < near_mm) < 0.5:
            continue
        txt = json.dumps(notes_.get(name.replace('instrument_', ''), {})).lower()
        kind = 'neck' if 'neck' in txt else ('grasp' if 'grasp' in name or 'grasp' in txt else 'contact')
        c0 = np.median(rest[hits], 0)
        sel = surf[np.linalg.norm(rest[surf] - c0, axis=1) < radius_mm / 1000]
        lab = f'{kind}:{name}'
        idx += sel.tolist()
        to += [lab] * len(sel)
        th = np.array_split(dist, 3)
        notes.append(f"{lab}: {len(sel)} vertices within {radius_mm} mm of the surface point nearest the jaw tip "
                     f"({vers[-1].parent.name}); jaw tip to surface median per third "
                     f"{[round(float(np.median(t)), 1) for t in th]} mm")
    for s_ in notes:
        log(f'[attach] {s_}')
    return np.array(idx, np.int32), np.array(to), notes


def unseen_volume(V, verts4d, tets):
    """Share of the volume (tets by centroid) that is outside the scope's view area in every frame."""
    valid = getattr(V, 'valid', np.ones((V.H, V.W), bool))
    seen = np.zeros(len(tets), bool)
    for k in range(0, V.n, 2):
        c_ = verts4d[k][tets].mean(1)
        q_, z = V.project(c_, k)
        x, y = np.round(q_[:, 0]).astype(int), np.round(q_[:, 1]).astype(int)
        ok = (x >= 0) & (x < V.W) & (y >= 0) & (y < V.H) & (z > 0)
        ok[ok] = valid[y[ok], x[ok]]
        seen |= ok
    P = verts4d[0][tets]
    vol = np.abs(np.linalg.det(np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], -1))) / 6
    return float(vol[~seen].sum() / vol.sum())


# ---------------------------------------------------------------- driver
def choose_start(V, body, unknown, keys):
    """Keyframe with the largest organ area whose outline touches unknown pixels least."""
    best, score = keys[0], -1
    for k in keys:
        b = body[k].astype(np.uint8)
        edge = (b - cv2.erode(b, np.ones((3, 3), np.uint8))).astype(bool)
        near_u = cv2.dilate(unknown[k].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        s = body[k].sum() * (1 - (edge & near_u).sum() / max(edge.sum(), 1))
        if s > score:
            best, score = k, s
    return int(best)


def build(V, clip, org, ver, spec, prompts, out_dir, overrides=None, log_print=True):
    c = cfg_of(ver)
    c.update(overrides or {})
    out_dir = Path(out_dir)
    (out_dir / 'work').mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / 'work' / 'log.txt', 'w')

    def log(s):
        if log_print:
            print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()
    t00 = time.time()
    name = org['mask']
    rule = org['volume_rule']
    use_depth = not org.get('see_through', False)
    log(f"[{clip}/{org['name']} {ver}] mask {name}, role {org['role']}, consistency {org['consistency']} -> volume "
        f"rule {rule} ({'; '.join(org['volume_why'])}), video depth {'used' if use_depth else 'NOT used (see-through)'}")
    log(f'[config] {json.dumps(c)}')
    desc = mask_descriptions(prompts)
    others = [m for m in V.names if m != name and not m.startswith('instrument_')]
    st = see_through_objects(spec, V.names, desc)
    merge = merge_candidates(org['name'], name, V.names, desc) if c['merge_parts'] else []
    att = {m: v for m, v in attached_masks(spec, org['name'], V.names, desc).items() if m != name and m not in merge}
    body, unknown, minfo = prepare_masks(V, name, others, [m for m in st if m != name], c, log, merge, list(att))
    valid = getattr(V, 'valid', np.ones((V.H, V.W), bool))
    area = body.sum((1, 2)) / valid.sum()
    raw_area = (V.mask(name) & valid).sum((1, 2)) / valid.sum()
    observed = raw_area >= c['dropout_frac'] * np.median(raw_area) if c['dropout_frac'] > 0 else np.ones(V.n, bool)
    if (~observed).any():
        runs = np.split(np.nonzero(~observed)[0], np.nonzero(np.diff(np.nonzero(~observed)[0]) > 1)[0] + 1)
        log(f'[masks] mask dropouts (area < {c["dropout_frac"]} x clip median {np.median(raw_area):.3f}): '
            f'{[(int(r[0]), int(r[-1])) for r in runs]} -> unobserved')
    keys = [k for k in range(0, V.n, 10) if area[k] >= c['min_area'] and observed[k]]
    log(f'[masks] body area fraction per keyframe {np.round(area[::10], 3).tolist()}; {len(keys)} keyframes used')
    np.savez_compressed(out_dir / 'work' / 'masks.npz', body=np.packbits(body, axis=-1), unknown=np.packbits(unknown, axis=-1),
                        shape=np.array(body.shape))
    bg, bg_ver = None, None
    if c['w_bg'] > 0:
        vers = sorted(p_.parent.name for p_ in (OUT / clip / 'background').glob('v*/occupancy.npz'))
        if vers:
            bg_ver = vers[-1]
            o_ = np.load(OUT / clip / 'background' / bg_ver / 'occupancy.npz')
            bg = o_['bg_depth'].astype(np.float32)
            log(f'[background] {bg_ver}: first-hit depth {bg.shape} (1/{int(o_["scale"])} res.); samples behind it + '
                f'{c["bg_margin_mm"]} mm cost w_bg {c["w_bg"]}')
    ext = None
    if c['w_ext'] > 0:
        ext = [border_completion(body[k], valid, unknown[k], c['ext_max']) if observed[k] else None for k in range(V.n)]
        n_ext = sum(e is not None for e in ext)
        log(f'[masks] border completion: ellipse continuation outside the view in {n_ext} / {V.n} frames '
            f'(median {np.median([len(e) for e in ext if e is not None]) * 9 if n_ext else 0:.0f} px)')
    obs = [Obs(V, k, body, unknown, n_cov=800, n_pts=500, erode=c['depth_erode'], use_depth=use_depth,
               see_mm=c['see_mm'], bg=bg, bg_margin=c['bg_margin_mm'], ext=ext) for k in keys]
    if not use_depth:                     # pose initialisation still needs points: the (untrusted) video depth
        for o, k in zip(obs, keys):
            o.pts_init = Obs(V, k, body, unknown, n_pts=300, erode=c['depth_erode']).pts
    view = np.mean([V.R[k][2] for k in keys], 0)
    t1 = time.time()
    start = c['start'] if c['start'] is not None else choose_start(V, body, unknown, keys)
    j0 = keys.index(start) if start in keys else int(np.argmin(np.abs(np.array(keys) - start)))
    prim = fit_primitive(obs, view, c, log, use_depth, j0)
    ffd = fit_ffd(obs, prim, c, log, use_depth, j0, view)
    log(f'[rest] fit time {time.time() - t1:.0f} s')
    rest_surf = ffd['verts_mm'] / 1000
    t1 = time.time()
    nodes, tets, faces = bcc_for_count(rest_surf, ffd['faces'], c['nodes'], log)
    log(f'[tets] {len(nodes)} nodes, {len(tets)} tets, {len(faces)} surface faces ({time.time() - t1:.0f} s); '
        f'{tet_report(nodes, tets)}')
    with torch.no_grad():
        init = Posed.offset(T(nodes * 1000), T(ffd['t']), T(ffd['dR'][j0]), T(ffd['dT'][j0]))
    log(f'[4d] start frame {start} (rule {rule}: {c["vol_rules"][rule]})')
    t1 = time.time()
    def reacquire(k):
        jk = int(np.argmin(np.abs(np.array(keys) - k)))
        with torch.no_grad():
            return Posed.offset(T(nodes * 1000), T(ffd['t']), T(ffd['dR'][jk]), T(ffd['dT'][jk]))
    verts4d, parts, ls4d = fit_4d(V, body, unknown, nodes, tets, faces, c, rule, log, start, init, use_depth,
                                  (np.array(keys), ffd['ls']), observed, reacquire, bg, ext)
    log(f'[4d] {time.time() - t1:.0f} s')
    assert np.isfinite(verts4d).all()
    prim_static, prim_faces = primitive_mesh(prim, 4, static=False)
    raw = V.mask(name).copy()
    for m in merge:
        raw |= V.mask(m)
    raw &= valid
    alt = {}
    if merge:
        alt[f'{name} alone'] = V.mask(name) & valid
    for m in att:
        alt[f'{name} + {m}'] = (raw | V.mask(m)) & valid
    q, per, frames = evaluate(V, nodes, faces, tets, verts4d, body, unknown, raw, prim_static, prim_faces, c, log, alt,
                              np.exp(ls4d), observed)
    mask_names = list(V.names)
    idx, to, notes = attach_hints(V, spec, org['name'], mask_names, desc, nodes, faces, verts4d, frames[::2], log,
                                  merged=merge)
    oi, ot, on = opening_hints(V, clip, org['name'], name, nodes, faces, verts4d, log, one=c['one_opening'])
    ni, nt, nn = instrument_contact_hints(V, clip, nodes, faces, verts4d, log)
    oi, ot, on = np.r_[oi, ni].astype(np.int32), np.r_[ot, nt], on + nn
    idx, to, notes = np.r_[idx, oi].astype(np.int32), np.r_[to, ot], notes + on
    r2 = lambda a, d=2: [round(float(x), d) for x in np.ravel(a)]
    pe = dict(radii_mm=r2(np.exp(prim['s'])), exponents=r2(prim['e'], 3),
              rotvec=r2(prim['r'], 4), centre_mm=r2(prim['t']),
              keyframe_offsets_deg_max=round(float(np.degrees(np.linalg.norm(ffd['dR'], axis=1)).max()), 2),
              keyframe_offsets_mm_max=round(float(np.linalg.norm(ffd['dT'], axis=1).max()), 2),
              video_depth_scale_range=[round(float(np.exp(ls4d).min()), 3), round(float(np.exp(ls4d).max()), 3)],
              ffd_max_offset_mm=round(ffd['max_offset_mm'], 2), fit_parts_mm=dict(primitive=np.round(prim['parts'], 3).tolist(),
                                                                                     rest=np.round(ffd['parts'], 3).tolist()))
    np.savez_compressed(out_dir / 'model.npz', rest_verts=nodes.astype(np.float32), faces=faces.astype(np.int32),
                        tets=tets.astype(np.int32), verts4d=verts4d.astype(np.float32), attach_idx=idx, attach_to=to,
                        start_frame=start, keyframes=np.array(keys), keyframe_dR=ffd['dR'], keyframe_dT_mm=ffd['dT'],
                        primitive_verts=prim_static.astype(np.float32), primitive_faces=prim_faces.astype(np.int32),
                        depth_scale=np.exp(ls4d).astype(np.float32), keyframe_depth_scale=np.exp(ffd['ls']).astype(np.float32),
                        fit_parts_mm=parts.astype(np.float32))
    q['unobserved_frames'] = np.nonzero(~observed)[0].tolist()
    q['behind_background'] = behind_background(V, clip, verts4d, faces, log)
    q['unseen_volume_share'] = round(unseen_volume(V, verts4d, tets), 4)
    ext_r = np.ptp((nodes - nodes.mean(0)) @ np.linalg.svd(nodes - nodes.mean(0), full_matrices=False)[2].T, 0) * 1000
    q['rest_extents_mm'] = [round(float(x), 1) for x in ext_r]
    log(f"[quality] rest extents {q['rest_extents_mm']} mm (principal axes); volume outside the view in every frame: "
        f"{q['unseen_volume_share'] * 100:.1f} %")
    q['segmentation'] = dict(version=seg_version(clip, V.names) if clip in OUT.name or (OUT / clip).exists() else None,
                             objects=list(V.names), note='matched by object list against seg/vNN/qc.json')
    q.update(clip=clip, organ=org['name'], mask=name, version=ver, role=org['role'], consistency=org['consistency'],
             volume_rule=rule, volume_rule_why=org['volume_why'], video_depth_used=use_depth, start_frame=start,
             keyframes=keys, masks=minfo, primitive=pe, attach=dict(counts={k: int((to == k).sum()) for k in np.unique(to)},
                                                                   notes=notes),
             runtime_s=round(time.time() - t00, 1),
             metric_notes=dict(
                 bands='thirds of the clip, every %d-th frame; iou / bf (4 px) / depth_mm (median |mesh - video depth|) '
                       'against this organ\'s own masks (work/masks.npz: body, unknown ignored); *_raw = against the raw '
                       'SAM mask with only instruments / outside-scope ignored' % c['eval_step'],
                 primitive='the fitted superquadric alone at its global pose, held static (no FFD, no 4D)',
                 static_rest='the rest shape (primitive + FFD) at its global pose, held static',
                 temporal='vertex speed / acceleration of verts4d in mm per frame',
                 volume='closed-surface volume of verts4d / rest', deformation='tets of verts4d vs rest (det <= 0 = inverted)'))
    (out_dir / 'quality.json').write_text(json.dumps(q, indent=1, default=float))
    col = (60, 255, 60)
    quality.contact_sheet(V, [(f"{org['name']} {ver} 4D", verts4d, faces, col)], list(range(0, V.n, 10)), out_dir / 'sheet.jpg')
    baselines_sheet(V, body, unknown, prim_static, prim_faces, nodes, faces, verts4d, keys, out_dir / 'work' / 'baselines.jpg')
    att = [(lab, idx[to == lab], cc) for lab, cc in zip(sorted(set(to.tolist()) - {'hidden_back'}),
                                                         [(220, 40, 40), (240, 160, 0), (160, 0, 200), (0, 170, 170),
                                                          (120, 60, 0), (255, 0, 140)])]
    att.append(('hidden_back', idx[to == 'hidden_back'], (40, 90, 220)))
    views3d(nodes, faces, out_dir / 'views3d.jpg', V, att)
    write_notes(out_dir, q, org, ver)
    log(f'[{ver}] done in {time.time() - t00:.0f} s')
    logf.close()
    return q


def baselines_sheet(V, body, unknown, prim, pf, rest, rf, v4, keys, path, cols=4):
    """Every 2nd keyframe: own body mask outline (white), unknown (grey hatch), primitive (blue), static rest (yellow),
    4D (green) outlines (full silhouettes, no visibility test)."""
    tiles = []
    for k in keys[::2]:
        img = V.frames[k].copy()
        img[unknown[k]] = (0.5 * img[unknown[k]] + 0.5 * np.array([90, 90, 90])).astype(np.uint8)
        for m, colr in ((body[k], (255, 255, 255)), (render(V, prim, pf, k)[0], (60, 120, 255)),
                        (render(V, rest, rf, k)[0], (255, 220, 0)), (render(V, v4[k], rf, k)[0], (60, 255, 60))):
            cnt = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            img = cv2.drawContours(np.ascontiguousarray(img), cnt, -1, colr, 2)
        cv2.putText(img, f'frame {k}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    y = 16
    for lab, colr in (('mask (body)', (255, 255, 255)), ('unknown', (150, 150, 150)), ('primitive', (60, 120, 255)),
                      ('static rest', (255, 220, 0)), ('4D', (60, 255, 60))):
        cv2.putText(sheet, lab, (sheet.shape[1] - 150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colr, 1, cv2.LINE_AA)
        y += 16
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_notes(out_dir, q, org, ver):
    b = q['bands']
    rows = []
    for m in ('primitive', 'static_rest', '4d'):
        rows.append(f"| {m} | " + ' | '.join(f"{v['iou']:.3f} / {v['bf']:.3f} / {v['depth_mm']} ({v['depth_scaled_mm']})"
                                            for v in b[m].values())
                    + f" | " + ' / '.join(f"{v['iou_raw']:.3f}" for k, v in b[m].items() if k != 'all') + ' |')
    hdr = ' | '.join(b['4d'].keys())
    other = ''
    for lab, t in q.get('bands_other_masks', {}).items():
        other += f"\nIoU against the raw mask '{lab}' (per third / all): " + '; '.join(
            f"{m} " + ' / '.join(f"{v['iou']:.3f}" for v in t[m].values()) for m in ('primitive', 'static_rest', '4d')) + '\n'
    p = q['primitive']
    t, v, d, mh = q['temporal'], q['volume'], q['deformation'], q['mesh_health']
    txt = f"""# {q['clip']} / {q['organ']} {ver}

`PYTHONPATH=. .venv/bin/python -m t2s.organ {q['clip']} "{q['organ']}" --ver {ver}` (t2s/organ.py; {q['runtime_s']:.0f} s)

- Mask `{q['mask']}`, role {q['role']}, consistency {q['consistency']} -> volume rule **{q['volume_rule']}**
  ({'; '.join(q['volume_rule_why'])}); video depth {'used' if q['video_depth_used'] else 'not used (see-through)'}.
- Merged into the organ mask: {q['masks']['merged']}; attached structures with unknown junction band:
  {q['masks']['junction_unknown']}.
- Unknown pixels: instruments (dilated), outside the scope, objects in front {list(q['masks']['front'])},
  see-through objects {q['masks']['see_through']}, thin flaps (opening r = {q['masks']['thin_radius_px']} px,
  {q['masks']['thin_px_mean']} px / frame).
- Primitive: superquadric radii {[round(x, 1) for x in p['radii_mm']]} mm, exponents {[round(x, 2) for x in p['exponents']]}; per-keyframe rigid offsets up to
  {p['keyframe_offsets_deg_max']} deg / {p['keyframe_offsets_mm_max']} mm; FFD max offset {p['ffd_max_offset_mm']} mm.
- Tets: {q['tets']['n_nodes']} nodes / {q['tets']['n_tets']} tets / {mh['n_faces']} surface faces, mean ratio min
  {q['tets']['mean_ratio']['min']:.3f} (p1 {q['tets']['mean_ratio']['p1']:.3f}), min dihedral {q['tets']['min_dihedral_deg']['min']:.1f} deg;
  rest volume {v['rest_ml']} ml, watertight {mh['watertight']}; extents (principal axes) {q.get('rest_extents_mm')} mm;
  share of the volume outside the view in every frame {q.get('unseen_volume_share')}.
- 4D from frame {q['start_frame']}; video depth scale fitted per frame {q['primitive'].get('video_depth_scale_range')};
  unobserved frames (mask dropout, excluded from the metrics): {len(q.get('unobserved_frames', []))};
  segmentation conflicts (another mask covers > 30 % of the organ's): {({o: len(v) for o, v in q['masks'].get('overlap_frames', {}).items()})}.

## Silhouettes (own masks): IoU / boundary F / depth residual mm (in brackets: against the video depth times the
## frame's fitted depth scale), and IoU against the raw SAM mask

| model | {hdr} | raw IoU per third |
|---|{'---|' * (len(b['4d']) + 1)}
""" + '\n'.join(rows) + f"""

{other}
## 4D health
- Behind the background surface ({(q.get('behind_background') or {}).get('background_version')}, vertices > 1 mm behind along the
  camera rays, per third): {(q.get('behind_background') or {}).get('per_third')}
- Temporal: vertex speed max {t['max_vertex_speed_mm_per_frame']} / p99 {t['p99_vertex_speed_mm_per_frame']} mm per frame,
  acceleration max {t['max_vertex_accel_mm_per_frame2']} / p99 {t['p99_vertex_accel_mm_per_frame2']}; centroid path {t['centroid_path_mm']} mm.
- Volume / rest: {v['ratio_min']} - {v['ratio_max']} (per third {v['ratio_per_band']}).
- Inverted tets: rest {d['inverted_tets_rest']}, 4D max {d['inverted_tets_4d_max']} per frame ({d['frames_with_inverted_tets']} frames);
  smallest tet volume ratio {d['min_volume_ratio_tet']}; edge stretch max {d['edge_stretch_max']} (p95 {d['edge_stretch_p95_max']}).
- Rigid part: up to {q['rigid_vs_nonrigid']['rot_deg_max']} deg / {q['rigid_vs_nonrigid']['shift_mm_max']} mm; non-rigid rms
  median {q['rigid_vs_nonrigid']['nonrigid_rms_mm_median']} mm.

## Attachment hints (model.npz attach_idx / attach_to)
""" + '\n'.join(f'- {s}' for s in q['attach']['notes']) + '\n'
    (out_dir / 'NOTES.md').write_text(txt)


def run(clip, organs=None, ver='v01', overrides=None):
    from t2s import views2
    c = cfg_of(ver)
    V = views2.load(clip)
    fit, skipped, spec, prompts = select_organs(clip, c['seg'], V.names)
    print(f'[select] fit: {[(o["name"], o["mask"], o["volume_rule"]) for o in fit]}')
    for s in skipped:
        print(f"[select] skip {s['name']} (mask {s['mask']}): {s['reason']}")
    sel = {'clip': clip, 'fit': fit, 'skipped': skipped}
    (OUT / clip / 'organs').mkdir(parents=True, exist_ok=True)
    (OUT / clip / 'organs' / 'selection.json').write_text(json.dumps(sel, indent=1))
    res = {}
    for o in fit:
        if organs and o['name'] not in organs:
            continue
        d = OUT / clip / 'organs' / o['name'].replace(' ', '_') / ver
        res[o['name']] = build(V, clip, o, ver, spec, prompts, d, overrides)
    return res


# ---------------------------------------------------------------- synthetic self-test
class SynthViews:
    """A deforming, rotating superquadric seen by an orbiting camera, with an instrument bar in front of it."""

    def __init__(self, n=30, H=240, W=320, f=280.0, seed=0):
        from r2s.camera import Cam
        from r2s.tissue.gallbladder import raster_depth
        from scipy.spatial.transform import Rotation as Rot
        self.n, self.H, self.W, self.fps = n, H, W, 10.0
        self.cam = Cam(W, H, f)
        self.names = ['organ', 'instrument_bar']
        self.instrument_names = ['instrument_bar']
        self.keyframes = list(range(0, n, 10))
        yy, xx = np.mgrid[:H, :W]
        self.valid = (xx - W / 2) ** 2 / (W / 2) ** 2 + (yy - H / 2) ** 2 / (H / 2 * 1.15) ** 2 < 1
        Dl, Fl = icosphere(4)
        a = np.array([0.026, 0.016, 0.013])
        e1, e2 = 0.8, 1.0
        x2, y2, z2 = ((Dl / a) ** 2).T
        fv = (x2 ** (1 / e2) + y2 ** (1 / e2)) ** (e2 / e1) + z2 ** (1 / e1)
        X0 = Dl * fv[:, None] ** (-e1 / 2)
        X0[:, 1:] *= (1 + 0.3 * X0[:, :1] / a[0])                         # pear-like taper (not a superquadric)
        R0 = Rot.from_euler('xyz', [20, -10, 35], degrees=True).as_matrix()
        self.faces = Fl
        self.truth, self.R, self.f, self.pos = [], [], [], []
        frames, masks, depth = [], [], []
        for k in range(n):
            t = k / (n - 1)
            s = 1 - 0.12 * np.sin(np.pi * t)
            Xk = X0 * np.array([1 / np.sqrt(s), 1 / np.sqrt(s), s])          # volume-preserving squash
            Rk = Rot.from_euler('z', 12 * t, degrees=True).as_matrix() @ R0
            Xk = Xk @ Rk.T + np.array([0.003 * t, 0, 0])
            self.truth.append(Xk)
            az = np.radians(-25 + 50 * t)
            el = np.radians(40)
            pos = 0.09 * np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
            z = -pos / np.linalg.norm(pos)
            x = np.cross(z, [0, 0, 1.0])
            x /= np.linalg.norm(x)
            Rc = np.stack([x, np.cross(z, x), z])
            self.R.append(Rc), self.f.append(f), self.pos.append(pos)
            zb = raster_depth(self, Xk, Fl, k)
            org = np.isfinite(zb)
            D = np.where(org, zb, 0.13).astype(np.float32)
            bar = np.zeros((H, W), np.uint8)
            cx = int(W * (0.3 + 0.4 * t))
            cv2.line(bar, (cx - 40, 0), (cx + 10, int(H * 0.55)), 1, 14)
            bar = bar.astype(bool) & self.valid
            D[bar] = 0.05
            img = np.full((H, W, 3), 60, np.uint8)
            img[org] = (120, 140, 60)
            img[bar] = (200, 200, 200)
            frames.append(img)
            masks.append(np.stack([org & ~bar & self.valid, bar]))
            depth.append(D)
        self.R, self.f, self.pos = np.array(self.R), np.array(self.f), np.array(self.pos)
        self.frames, self._masks, self._depth = np.array(frames), np.array(masks), np.array(depth)
        self.truth = np.array(self.truth)

    def mask(self, name):
        return self._masks[:, self.names.index(name)]

    def depth(self, k):
        return self._depth[k]

    def occluders(self, k):
        return self.mask('instrument_bar')[k] | ~self.valid

    def project(self, X, k):
        return self.cam.project(np.asarray(X, float), self.R[k], self.f[k], self.pos[k])

    def unproject(self, u, v, z, k):
        return self.cam.unproject(u, v, z, self.R[k], self.f[k], self.pos[k])

    def points(self, name, k, step=3, erode=2):
        m = (self.mask(name)[k] & self.valid).astype(np.uint8)
        if erode:
            m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        ys, xs = np.nonzero(m[::step, ::step])
        ys, xs = ys * step, xs * step
        return self.unproject(xs, ys, self.depth(k)[ys, xs], k)


def selftest(out_dir=None):
    out_dir = Path(out_dir or ROOT / 'outputs' / 't2s' / '_selftest')
    t0 = time.time()
    V = SynthViews()
    print(f'[selftest] synthetic views {time.time() - t0:.0f} s')
    spec = dict(organs=[dict(name='organ', role='primary', consistency='fluid-filled')], connections=[])
    org = dict(name='organ', mask='organ', role='primary', consistency='fluid-filled', volume_rule='fluid',
               volume_why=['synthetic'], see_through=False)
    q = build(V, 'synthetic', org, 'v10', spec, {}, out_dir,
              overrides=dict(iters_first=200, iters=60, nodes=900, sq_iters=(40, 160), ffd_iters=150, eval_step=1))
    # shape error of the 4D mesh against the truth (surface-to-surface, mm)
    from scipy.spatial import cKDTree
    m = np.load(out_dir / 'model.npz')
    err = []
    for k in range(0, V.n, 5):
        S = m['verts4d'][k][np.unique(m['faces'])]
        err.append(float(np.median(cKDTree(V.truth[k]).query(S)[0]) * 1000))
    print(f"[selftest] 4D IoU {q['bands']['4d']['all']['iou']}, static rest {q['bands']['static_rest']['all']['iou']}, "
          f"primitive {q['bands']['primitive']['all']['iou']}; surface error median per frame (mm) {np.round(err, 2)}; "
          f"volume {q['volume']['ratio_min']}-{q['volume']['ratio_max']}; inverted 4D {q['deformation']['inverted_tets_4d_max']}")
    ok = q['bands']['4d']['all']['iou'] > 0.9 and max(err) < 2.0 and q['deformation']['inverted_tets_4d_max'] == 0
    print('[selftest]', 'PASS' if ok else 'FAIL', f'({time.time() - t0:.0f} s)')
    return ok


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--selftest':
        sys.exit(0 if selftest(a[1] if len(a) > 1 else None) else 1)
    ver = 'v01'
    if '--ver' in a:
        i = a.index('--ver')
        ver = a[i + 1]
        a = a[:i] + a[i + 2:]
    run(a[0], a[1:] or None, ver)
