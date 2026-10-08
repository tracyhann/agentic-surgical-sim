"""Laparoscopic instrument model (MJCF + closed-form kinematics) and instrument motion recovered from the video.

Each instrument is a straight shaft through a fixed port (remote centre of motion, RCM) with yaw / pitch / insertion /
roll and two jaws, driven by position servos. Its port is fitted so that the projected shaft matches the segmented
shaft axis in every frame; its tool-centre point (TCP) follows the tip seen in the video at the calibrated depth.
"""
import numpy as np
from scipy.optimize import least_squares
from .camera import fill_smooth, pose

TCP_OFFSET = 0.019                 # TCP (between the jaws) beyond the shaft origin of the roll link
ARM = ('yaw', 'pitch', 'insertion', 'roll')
PITCH_MAX = 1.55           # rad below horizontal; a trocar can sit almost straight above its target
SLEW = np.array([0.08, 0.08, 0.006, 0.15, 0.007])        # max joint change per 0.05 s control step


# ---------------------------------------------------------------- kinematics
def _rz(h):
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shaft_dir(yaw, pitch, heading=0.0):
    return _rz(heading) @ np.array([-np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])


def fk(rcm, yaw, pitch, insertion, heading=0.0):
    return np.asarray(rcm, float) + (insertion + TCP_OFFSET) * shaft_dir(yaw, pitch, heading)


def ik(rcm, tcp, heading=0.0):
    v = _rz(-heading) @ (np.asarray(tcp, float) - np.asarray(rcm, float))
    dist = np.linalg.norm(v)
    pitch = np.arcsin(np.clip(-v[2] / dist, -1, 1))
    yaw = np.arctan2(-v[0], v[1])
    insertion = dist - TCP_OFFSET
    if not (-0.05 <= pitch <= PITCH_MAX + 0.05 and -1.62 <= yaw <= 1.62 and -0.005 <= insertion <= 0.255):
        raise ValueError(f'unreachable TCP {tcp}: yaw={yaw:.3f} pitch={pitch:.3f} insertion={insertion:.4f}')
    # just outside a joint range (fitted port slightly off): clamp, the tip then misses by at most a few mm
    return float(np.clip(yaw, -1.57, 1.57)), float(np.clip(pitch, 0.0, PITCH_MAX)), float(np.clip(insertion, 0.0, 0.25))


def heading_for(P, T):
    v = T.mean(0) - P
    return float(np.arctan2(-v[0], v[1]))     # shaft along trocar +y at yaw 0


def joint_targets(P, Tseq, heading, jaw=0.0):
    out = []
    for t in Tseq:
        y, p, ins = ik(P, t, heading)
        out.append([y, p, ins, 0.0, jaw])
    out = np.array(out)
    for k in range(1, len(out)):
        out[k] = out[k - 1] + np.clip(out[k] - out[k - 1], -SLEW, SLEW)
    return out


# ---------------------------------------------------------------- MJCF
def mjcf(p, shaft_d, shaft_rgba, jaw_rgba, holds):
    """(body, actuators, contacts) XML fragments. An instrument that holds tissue does so through a grasp
    constraint (no contact); one that only touches tissue collides with its jaws, its shaft rides above the tissue."""
    r = shaft_d / 2
    jaw_con = 'contype="0" conaffinity="0"' if holds else 'contype="2" conaffinity="1"'
    jaw = f'friction="1.5 0.02 0.0002" condim="4" priority="1" solref="0.001 1" solimp="0.95 0.99 0.0005" {jaw_con}'
    body = f"""
  <body name="{p}_yaw_link" gravcomp="1">
    <joint name="{p}_yaw" type="hinge" axis="0 0 1" range="-1.57 1.57" damping="0.02" armature="0.0005"/>
    <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>
    <body name="{p}_pitch_link" gravcomp="1">
      <joint name="{p}_pitch" type="hinge" axis="-1 0 0" range="0 {PITCH_MAX}" damping="0.02" armature="0.0005"/>
      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>
      <body name="{p}_shaft" gravcomp="1">
        <joint name="{p}_insertion" type="slide" axis="0 1 0" range="0 0.25" damping="1" armature="0.01"/>
        <geom name="{p}_shaft_geom" type="cylinder" fromto="0 -0.30 0 0 -0.003 0" size="{r:.5f}" mass="0.04"
              rgba="{shaft_rgba}" material="{p}_shaft_mat" contype="0" conaffinity="0"/>
        <body name="{p}_roll_link" gravcomp="1">
          <joint name="{p}_roll" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="0.005" armature="0.0002"/>
          <geom name="{p}_clevis_geom" type="cylinder" fromto="0 -0.003 0 0 0.003 0" size="{r + 0.0002:.5f}" mass="0.004"
                rgba="{shaft_rgba}" material="{p}_shaft_mat" {jaw_con}/>
          <site name="{p}_tcp" pos="0 {TCP_OFFSET} 0" size="0.0008" rgba="1 1 0 0"/>
          <body name="{p}_jaw_left" pos="0 0.012 0" gravcomp="1">
            <joint name="{p}_jaw_left" type="slide" axis="-1 0 0" range="0 0.007" damping="0.2" armature="0.001"/>
            <geom name="{p}_jaw_left_geom" type="box" pos="-0.0013 0 0" size="0.0012 0.010 0.0018" mass="0.002"
                  rgba="{jaw_rgba}" material="{p}_jaw_mat" {jaw}/>
          </body>
          <body name="{p}_jaw_right" pos="0 0.012 0" gravcomp="1">
            <joint name="{p}_jaw_right" type="slide" axis="1 0 0" range="0 0.007" damping="0.2" armature="0.001"/>
            <geom name="{p}_jaw_right_geom" type="box" pos="0.0013 0 0" size="0.0012 0.010 0.0018" mass="0.002"
                  rgba="{jaw_rgba}" material="{p}_jaw_mat" {jaw}/>
          </body>
        </body>
      </body>
    </body>
  </body>"""
    act = f"""
  <position name="{p}_yaw" joint="{p}_yaw" kp="4" kv="0.15" ctrlrange="-1.57 1.57" forcerange="-1 1"/>
  <position name="{p}_pitch" joint="{p}_pitch" kp="4" kv="0.15" ctrlrange="0 {PITCH_MAX}" forcerange="-1 1"/>
  <position name="{p}_insertion" joint="{p}_insertion" kp="300" kv="8" ctrlrange="0 0.25" forcerange="-4 4"/>
  <position name="{p}_roll" joint="{p}_roll" kp="0.5" kv="0.01" ctrlrange="-3.14 3.14" forcerange="-0.2 0.2"/>
  <position name="{p}_jaw_left" joint="{p}_jaw_left" kp="60" kv="0.5" ctrlrange="0 0.007" forcerange="-0.06 0.06"/>
  <position name="{p}_jaw_right" joint="{p}_jaw_right" kp="60" kv="0.5" ctrlrange="0 0.007" forcerange="-0.06 0.06"/>"""
    con = f"""
  <exclude body1="{p}_jaw_left" body2="{p}_jaw_right"/>
  <exclude body1="{p}_jaw_left" body2="{p}_roll_link"/>
  <exclude body1="{p}_jaw_right" body2="{p}_roll_link"/>"""
    return body, act, con


# ---------------------------------------------------------------- motion from the video
def _border_dist(q, W, H):
    return min(q[0], W - 1 - q[0], q[1], H - 1 - q[1])


def mask_axis(m, W, H):
    """Visible distal end, image axis (unit, pointing to the end) and entry point of an instrument mask, or None.
    The distal end is the axis extreme farther from the image border (instruments enter from the border)."""
    ys, xs = np.nonzero(m)
    if len(xs) < 200:
        return None
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    d = np.linalg.svd(P - c)[2][0]
    s = (P - c) @ d
    a, b = P[s <= np.percentile(s, 1.5)].mean(0), P[s >= np.percentile(s, 98.5)].mean(0)
    if _border_dist(b, W, H) < _border_dist(a, W, H):
        d, a, b, s = -d, b, a, -s
    tip = b
    near = P[(s > np.percentile(s, 85)) & (s < np.percentile(s, 97))]
    return tip, d, a, near


def width_ratio(m, W, H):
    """Apparent shaft width near the entry / near the distal end: > 1 when the shaft comes towards the scope."""
    from .perception import shaft_samples
    s = shaft_samples(m, 1.0, 1.0, H, W, keep=(0.05, 0.95))
    if len(s) < 8:
        return 1.0
    w = 1.0 / np.array([z for _, _, z in s])         # f * d / z with f = d = 1  ->  width
    n = len(w) // 3
    return float(np.median(w[:n]) / np.median(w[-n:]))


def track_2d(masks_i, Z, W, H):
    tips, dirs, entry, depth = [], [], [], []
    for k in range(len(masks_i)):
        r = mask_axis(masks_i[k], W, H)
        if r is None:
            tips.append([np.nan] * 2), dirs.append([np.nan] * 2), entry.append([np.nan] * 2), depth.append(np.nan)
            continue
        tip, d, a, near = r
        tips.append(tip), dirs.append(d), entry.append(a)
        depth.append(float(np.median(Z[k][near[:, 1].astype(int), near[:, 0].astype(int)])) if len(near) else np.nan)
    vis = np.isfinite(np.array(tips)[:, 0])
    return dict(tip=fill_smooth(np.array(tips)), dir=fill_smooth(np.array(dirs)), entry=fill_smooth(np.array(entry)),
                depth=fill_smooth(np.array(depth), 9), visible=vis)


def holding_point(organ_masks, tip_px, Z, offset=0.004):
    """Where a holding instrument's buried jaws are: the organ pixels nearest its visible end, at their own depth plus
    `offset` into the tissue."""
    px, z = [], []
    for k in range(len(organ_masks)):
        ys, xs = np.nonzero(organ_masks[k])
        if len(xs) == 0 or not np.isfinite(tip_px[k]).all():
            px.append([np.nan] * 2), z.append(np.nan)
            continue
        dd = np.hypot(xs - tip_px[k, 0], ys - tip_px[k, 1])
        sel = dd < np.percentile(dd, 3) + 2
        px.append([np.median(xs[sel]), np.median(ys[sel])])
        z.append(float(np.median(Z[k][ys[sel], xs[sel]])) + offset)
    return fill_smooth(np.array(px), 7), fill_smooth(np.array(z), 9)


def fit_port(cam, tips, dirs2d, entry_px, cams, fwd_range):
    """Port (RCM) whose projected shaft matches the observed axis in every frame; within `fwd_range` along the scope's
    forward axis and 8-22 cm from the tips. Returns (port, mean axis error in degrees)."""
    def res(P):
        r, wrong_way, end_on = [], [], []
        for k in range(0, len(tips), 2):
            a, _ = cam.project(tips[k], *pose(cams, k))
            b, _ = cam.project(tips[k] + 0.005 * (P - tips[k]) / np.linalg.norm(P - tips[k]), *pose(cams, k))
            e = a - b
            e_len = np.linalg.norm(e)
            e /= e_len + 1e-12
            d = dirs2d[k]
            r.append(e[0] * d[1] - e[1] * d[0])
            wrong_way.append(max(0.0, -(e @ d)))        # the port must lie behind the tip, on the entry side
            end_on.append(max(0.0, 2.0 - e_len))        # a shaft never points straight into the lens
        fwd = (P - cam.pos) @ cam.R[2]
        dist = np.linalg.norm(P - tips.mean(0))
        r += [3 * max(0, fwd - fwd_range[1]), 3 * max(0, fwd_range[0] - fwd), 3 * max(0, 0.08 - dist), 3 * max(0, dist - 0.22),
              3 * float(np.mean(wrong_way)), float(np.mean(end_on)),
              30 * max(0.0, 0.04 - float(np.linalg.norm(P - cam.pos))),      # ports are on the wall, >= 4 cm from the scope
              0.02 * inside_picture(P),           # the shaft enters from the border: its port is not inside the picture
              30 * max(0.0, top + 0.01 - P[2])]   # trocars sit in the abdominal wall, above the working area
        return np.array(r)

    top = float(tips[:, 2].max())

    def inside_picture(P):
        q, z = cam.project(P, *pose(cams, 0))
        if z <= 0:
            return 0.0
        return float(max(0.0, min(q[0], cam.W - q[0], q[1], cam.H - q[1]) + 20))   # px inside (with a 20 px margin)
    # start points: beyond the entry point of the shaft, outside the picture, at several depths
    out_dir = entry_px - tips_px_mean(cam, tips, cams)
    out_dir /= np.linalg.norm(out_dir) + 1e-9
    init_px = entry_px + out_dir * 400
    best = None
    for z0 in (fwd_range[0] + 0.01, np.mean(fwd_range), fwd_range[1], 0.03, 0.06):
        init = cam.unproject(init_px[0], init_px[1], max(z0, 0.01))
        s = least_squares(res, init, x_scale=0.02)
        if best is None or s.cost < best.cost:
            best = s
    ang = np.degrees(np.arcsin(np.clip(np.abs(best.fun[:-9]), 0, 1)))
    return best.x, float(ang.mean())


def tips_px_mean(cam, tips, cams):
    return np.mean([cam.project(tips[k], *pose(cams, k))[0] for k in range(len(tips))], axis=0)


def recover(clip, cam, cams, masks, Z):
    """3D path (TCP per frame), port and heading of every instrument of the clip."""
    n = len(masks)
    out = {}
    organ_i = clip.obj_index(clip.organ)
    for ins in clip.instruments:
        i = clip.obj_index(ins['mask'])
        tr = track_2d(masks[:, i], Z, cam.W, cam.H)
        if ins.get('holds'):
            hp, hz = holding_point(masks[:, clip.obj_index(ins['holds'])], tr['tip'], Z, ins.get('hold_offset', 0.004))
            # before it grasps (grasp_frame), the instrument follows its own visible end; blended over 5 frames
            g = int(ins.get('grasp_frame', 0))
            w = np.clip((np.arange(n) - g + 5) / 5.0, 0, 1)
            tip_px = w[:, None] * hp + (1 - w[:, None]) * tr['tip']
            zt = w * hz + (1 - w) * tr['depth']
        else:
            tip_px, zt = tr['tip'], tr['depth']
        tips = np.array([cam.unproject(tip_px[k, 0], tip_px[k, 1], zt[k], *pose(cams, k)) for k in range(n)])
        # an instrument whose apparent width stays constant along its length runs parallel to the image plane: its port
        # is at about the depth of its tips; one that widens towards the image border comes from near the scope
        vis_k = np.nonzero(tr['visible'])[0]
        ratio = float(np.median([width_ratio(masks[k, i], cam.W, cam.H) for k in vis_k[::max(1, len(vis_k) // 10)]]))
        entry = np.nanmedian(tr['entry'], axis=0)
        # the port either sits near the scope's depth (shaft widening towards the border) or near the tips' depth
        # (shaft parallel to the image plane): fit both, keep the one whose projected shaft matches the video better
        ranges = {'near_scope': (-0.06, 0.03), 'near_tips': (float(np.median(zt)) - 0.02, float(np.median(zt)) + 0.02)}
        fits = {mode: fit_port(cam, tips, tr['dir'], entry, cams, zr) for mode, zr in ranges.items()
                if ins.get('port') in (None, mode)}
        mode = min(fits, key=lambda mo: fits[mo][1])
        P, err = fits[mode]
        dd = (tips - P) / np.linalg.norm(tips - P, axis=1, keepdims=True)
        T = withdraw(tips + ins.get('tcp_offset', 0.0 if ins.get('holds') else -0.003) * dd, tr['visible'], P)
        out[ins['name']] = dict(rcm=P, axis_err_deg=err, port_mode=mode, width_ratio=ratio, T=T, tip_px=tip_px,
                                tip_depth=zt, visible=tr['visible'], heading=heading_for(P, T[tr['visible']]))
    del organ_i
    return out


def withdraw(T, visible, P, frac=0.45, ramp=12):
    """Out of view for more than 5 frames (before it enters, after it leaves, or in between): the instrument is pulled
    back along its own shaft towards the port, blended in and out over `ramp` frames."""
    gap = np.zeros(len(visible), bool)
    k = 0
    while k < len(visible):
        if not visible[k]:
            j = k
            while j < len(visible) and not visible[j]:
                j += 1
            if j - k > 5 or k == 0 or j == len(visible):
                gap[k:j] = True
            k = j
        else:
            k += 1
    w = np.clip(np.convolve(gap.astype(float), np.ones(ramp) / ramp, mode='same') * 1.6, 0, 1)
    w[gap] = 1.0
    return T + (frac * w)[:, None] * (P - T)


def actions(inst, n_frames, fps, dt=0.05):
    t_src = np.arange(n_frames) / fps
    t_ctl = np.arange(0, t_src[-1] + 1e-9, dt)
    acts = []
    for name, I in inst.items():
        Ti = np.stack([np.interp(t_ctl, t_src, I['T'][:, k]) for k in range(3)], 1)
        acts.append(joint_targets(I['rcm'], Ti, I['heading'], 0.0))
    return np.concatenate(acts, 1)
