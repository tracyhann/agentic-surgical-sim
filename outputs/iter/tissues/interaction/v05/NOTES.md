# interaction v05: instruments vs tissue (shot A, chole_a). This is the current version.

`python -m r2s.tissue.interaction v05` (code `r2s/tissue/interaction.py`, settings `VERSIONS['v05']`; about 35 s on
CPU, no GPU). No new masks were made: the code uses the clip's SAM masks 'grasper' and 'probe'. It also reads one
mask from another track, `membrane/v05/masks.npz` ('membrane_clean'), for a cross-check only.

## What it models
Each instrument is a straight shaft through one fixed port P (the remote centre of motion). Its distal end T_k is
fitted per frame. T_k is the far end of the jaws or of the probe, on the shaft axis. One joint least-squares fit per
instrument (scipy `least_squares`, soft-L1, sparse Jacobian) uses these unknowns: P, kappa (an apparent-width scale),
and (u_k, v_k, log z_k) for the 251 frames. Its terms are:
- **tip**: (u_k, v_k) is the distal end of the mask, measured on the mask's centre line. It is the farthest mask pixel
  along the axis inside the shaft core. sigma = 1 px. Blurred frames get less weight (variance of the Laplacian).
- **axis**: up to 12 centre-line samples of the mask, at their distance to the projected line T_k->P. Each sample is
  weighted by exp(-d_tip / 120 px), sigma = 2 px, so samples near the tip count most.
- **width**: the apparent width at the shaft samples is kappa * f * 5 mm / z on the 3D line. This term sets the tilt.
  It leaves out the image border, the other instrument, and the last 1.5 widths before the tip (jaws, taper).
- **dav2**: V.depth on the instrument's own pixels at the same samples (sigma = 6 mm). This puts the instruments in the
  same depth frame as the tissue models, which are all built on V.depth.
- **tipz** (grasper only): the jaw-tip depth also follows V.depth of the tissue 4-14 px around the jaw tip, which is
  the apex of the sheet the jaws hold (sigma about 4 mm). This term is never used for the probe, because the probe's
  indentation is measured against exactly that depth.
- **smoothness**: 3D acceleration of T sigma = 1 mm/frame^2, and log-depth velocity.
- **out of view**: in frames without a probe mask (117-126 and a few frames up to 136) the tip must project outside
  the picture.
- **port priors**: the port is at least 1 cm above every tip; the port-to-tip distance is between 6 and 26 cm (the
  MuJoCo insertion range).

Axis measurement: PCA on the mask, then a robust line fit on the centre line using only whole cross-sections (no
image-border bins). When the mask is short and wide (PCA elongation < 3, the grasper from about frame 120 on, with only
the jaws and a stub of shaft in view), the axis comes from the dominant direction of the straight silhouette edges
instead.

## Results (v05; old = the tracks in `V.tools`, measured the same way)
| | grasper old | grasper new | probe old | probe new |
|---|---|---|---|---|
| tip reprojection, median / mean (px) | 15.4 / 23.1 | **2.0 / 2.6** | 29.6 / 31.4 | **0.6 / 0.6** |
| tip, after the opening pan (frames >= 63), median (px) | 11.5 | 1.9 | 28.3 | 0.4 |
| shaft axis angle error, median (deg) | 2.2 | 2.3 | 1.0 | 1.5 |
| centre-line offset, median (px) | 3.9 | 1.0 | 4.5 | 3.4 |
| shaft IoU, median / mean | 0.73 / 0.51 | **0.84 / 0.83** | 0.70 / 0.64 | **0.76 / 0.74** |
| port: distance from each frame's back-projected axis plane, median / max (mm) | 3.2 / 21 | 4.1 / 32 | 1.4 / 12.5 | 2.4 / 8.7 |
| port: independent per-frame width lines to port, median (mm) | 46 | 38 | 17.5 | 7.6 |
| 3D acceleration, median / max (mm/frame^2) | 0.17 / 1.1 | 0.41 / 1.2 | 0.22 / 3.9 | 0.68 / 7.1 |
| shaft-tissue gap, distal half (mm; < 0 = shaft behind the tissue it is seen against) | +7.6 | +5.7 | +10.0 | +8.1 |
| kappa (apparent-width scale in the V.depth frame) | - | 0.986 | - | 0.877 |
| port (world m) | (-0.042, 0.033, 0.164) | (-0.021, -0.010, 0.211) | (0.052, 0.085, 0.070) | (0.070, 0.068, 0.096) |
| port-to-tip distance (mm) | | 164-185 | | 65-114 |

The IoU is computed for the rendered cylinder with diameter kappa * 5 mm and a rounded tip, both instruments
z-buffered, against the instrument mask. "Old" uses the old TCP as its tip. The old grasper TCP was the 'holding
point' 4 mm inside the gallbladder, and it projects 60-90 px below the jaws during the opening pan.

## Interaction facts (from the video)
- **Grasper, what it holds**: the closed jaws sit in the apex of the tented peritoneal sheet in every frame. The tent
  hangs from the jaws down to the gallbladder; the jaws never touch the gallbladder body.
  - `grasp_point` is the MuJoCo TCP (3 mm proximal of the jaw tips, between the jaws).
  - `grasp_apex_sheet` is the sheet apex as V.depth sees it. It is the median of the membrane-v05 sheet pixels nearest
    the jaw tip, unprojected at their video depth.
  - TCP to the sheet apex: median 4.6 mm (p10 2.6, max 15.5). TCP minus sheet depth: median +2.9 mm.
  - The sheet mask reaches within 5 px of the jaw tip in 132 of 251 frames (median 4.7 px). The membrane track cut
    its mask near the jaws, so in the other frames it ends a little short of them.
- **Jaw opening**: the jaws are closed on the sheet throughout.
  - The distal mask width divided by the shaft width is median 0.94 (p10 0.87, max 1.15). That is a taper, never the
    V-shape of open jaws.
  - The far end of the mask is the rounded tip of the closed jaws.
  - `grasper_jaw` = 0 for all frames (MuJoCo jaw slide 0 = closed).
- **Probe**: the tip pushes into the gallbladder neck, under the sheet.
  - The tissue around the tip in contact frames is 'gallbladder' in 208 of 223 frames (median fraction 0.99). The
    gallbladder mask includes the neck.
  - Signed indentation = camera depth of the tip minus the median V.depth of the tissue 4-14 px around it. In contact
    it is median **+5.3 mm** (p10 +0.8, max +11.2).
  - The indentation is insensitive to the ring radius: 8-20 px gives +5.1, 12-28 px gives +5.3.
  - **Contact** (smoothed indentation > -2 mm, tip visible, runs of 5 frames or more): frames **0-113 and 142-250**.
  - **Pushing hard** (> +4 mm): frames 0-3, 17-27, 51-68, 76-112, 162-178, 194-201 and 212-250.
  - 114-141: the probe is pulled back and out of view. Where its tip is visible again at the top border (127-136) it
    is 15-18 mm in front of the tissue.
- **Uncertainty of the indentation**: the absolute level depends on the depth frame.
  - With widths only (v01, kappa = 1) the tip lies +15 mm behind the surface, which is physically impossible. The
    distal shaft is then behind the tissue it is seen against in 63 % of frames.
  - With the shaft put into the V.depth frame (v02 on) it is +2..+5 mm.
  - Depth map alone (V.depth on the probe's own tip pixels vs the ring) gives +0.3 mm. Monocular depth smooths thin
    tools into their background, so this measure is biased towards 0.
  - Treat it as "a few mm into the tissue, varying in time". Do not use the absolute number to better than about 5 mm.

## model.npz
- `<name>_tip` (251, 3): distal end, world m. `<name>_shaft` (251, 3): unit vector from the port to the tip.
  `<name>_rcm` (3,): the fixed port. `<name>_tcp` (251, 3) = tip - 3 mm * shaft (the MuJoCo TCP site).
- `<name>_joints` (251, 4): yaw, pitch, insertion and roll (0) from `instruments.ik(rcm, tcp, heading)`, with
  `<name>_heading` = `instruments.heading_for(rcm, tcp)`. All frames are within the joint ranges (0 clamped).
- `<name>_visible` (251,): an instrument mask exists. `<name>_old_tip`: V.tools tip, for comparison.
- `grasper_jaw` (251,) = 0. `grasper_jaw_ratio` (251,) is the evidence for it.
- `grasp_point` (251, 3) = grasper TCP. `grasp_apex_sheet` (251, 3): see above.
- `probe_indent_mm` (251,): median filtered (7 frames), NaN-free (interpolated over 117-126, where there is no probe
  mask). `probe_indent_raw` keeps NaN where there is no measurement. `probe_contact` (251,) bool.
- Contract mesh: `rest_verts`, `faces`, `verts4d` (251, 146, 3) are the two shafts as 12 cm tubes with radius 2.5 mm
  from the tip back towards the port. Vertices 0-72 are the grasper and 73-145 the probe (`mesh_parts`). This mesh is
  for `quality.contact_sheet` only; the integrator should drive the MJCF instruments.
- Files: `sheet.jpg` (contact sheet of the tubes at the keyframes), `tips_sheet.jpg` (old vs new tips and shafts every
  10 frames), `tips_zoom.jpg` (2x crops around both tips every 20 frames), `series.png` (per-frame metrics, indentation
  and contact), `views3d.jpg` (ports, tip paths and shafts in 3 views), `quality.json`, `work/per_frame.npz`.

## For the integrator (MuJoCo, r2s/instruments.py)
- Put each instrument's trocar at `<name>_rcm` with `heading = <name>_heading`. Drive yaw, pitch and insertion with
  `<name>_joints[:, :3]` (per video frame; interpolate to the control rate). Set roll to 0.
- Set the grasper jaw to 0 (closed) and attach the sheet's jaw row to the grasper TCP (`grasp_point`). The sheet apex
  as the video depth sees it is about 4.6 mm away (`grasp_apex_sheet`). Snap one to the other, or accept the gap.
- The probe must collide with the gallbladder neck. Remove the 1.5 mm penetration clamp. The data say the tip goes
  about 5 mm (up to 11 mm) below the surrounding surface during 0-113 and 142-250.
- Slew limits: the probe's insertion changes by up to 12 mm per 0.05 s at 113-117 and 136-139 (fast pull-out and
  re-entry). `INS.SLEW` allows 6 mm, so raise it or accept a lag there. All other joints are within the slew limits.
- Shaft diameter: in the V.depth frame the probe looks like a 4.4 mm shaft (kappa 0.877) and the grasper like a 4.9 mm
  shaft. Either the probe is thinner than 5 mm or V.depth is about 12 % too near around it. A 5 mm probe rendered from
  these tracks looks slightly too thick.

## Iterations (change -> metrics -> what I saw -> next)
Numbers are median tip error (px), median axis angle error (deg), median IoU, and kappa. G = grasper, P = probe.

| version | change | G tip / ang / IoU | P tip / ang / IoU | probe indent in contact | what I saw |
|---|---|---|---|---|---|
| old | V.tools | 13.8 / 12.3* / 0.67 | 28.3 / 0.9 / 0.72 | - | grasper TCP in the gallbladder; probe TCP 25-35 px behind the tip, along the shaft |
| v01 | masks only: tip, centre line, widths (kappa 1), fixed port | 2.1 / 15.2* / 0.79 | 5.9 / 1.2 / 0.85 | **+15.3 mm** | tip 1-2 cm behind the tissue; distal probe shaft behind the tissue next to it in 63 % of frames; mask axis wrong for the short grasper masks (*) |
| v02 | edge-based axis for short masks; V.depth on the instrument pixels sets the depth level; kappa fitted | 1.5 / 3.0 / 0.81 | 5.9 / 1.0 / 0.80 (kappa 0.895) | +2.2 | probe tip about 10 px off the mask end in many frames; the line is pulled by 12 axis samples against 1 tip |
| v03 | tip first: sigma_tip 1 px, axis samples weighted by distance to the tip, smoothness 1 mm | 0.8 / 3.2 / 0.79 | 1.2 / 1.2 / 0.66 | +5.2 | the probe tip is right, but its line is offset 5 px: the mean of the farthest pixels lies off the axis (asymmetric end cap) |
| v04 | tip measured on the centre line; rounded-end render | 1.9 / 2.6 / 0.83 | 0.6 / 1.5 / 0.76 (kappa 0.877) | +5.3 | both tips on the distal ends (tips_zoom); grasper axis still 7 deg off in 137-190; grasper jaw tip 6.8 mm behind the tissue around it |
| v05 | grasper jaw-tip depth also from the tissue around the jaws; indentation sensitivity; membrane cross-check | 2.0 / 2.3 / 0.84 | 0.6 / 1.5 / 0.76 | +5.3 | grasper jaw-tip minus tissue 6.8 -> 3.5 mm; TCP to sheet apex 4.6 mm |

(*) The angle errors of v01 and old (v01 run) used the PCA mask axis, which was wrong for the short grasper masks
(frames 120+). From v02 the axis is measured with the edge method; with it the old grasper scores 2.2 deg.
The tip measure also changed between v03 and v04 (mean of the farthest pixels -> axis end point), so the "old"
numbers differ slightly between the two.

## Known problems
- **Port position along the shaft is weakly determined.** Fits on frame subsets move the port by 10-20 cm along the
  shaft direction; across sides it moves only a few mm. The grasper port moved between 11 and 21 cm from the tip
  across versions. The direction to the port is well determined; its distance comes from the camera parallax (2 cm)
  and the width tilt. All solutions keep the joints inside the MuJoCo ranges.
- **A fixed port does not match every frame.** The grasper axis is 5-10 deg off in frames 150-200 (short mask, only
  the jaws and a stub of shaft in view). The probe's far shaft is offset 5 px in frames 60-117 (IoU about 0.6 there).
  The cause could be camera-pose errors, real motion of the abdominal wall, or a biased mask axis. It is not
  resolved; the tips are fitted first.
- **Absolute depth of the instruments** follows V.depth (kappa 0.88 for the probe). The width ruler alone puts the
  probe 1-2 cm deeper, which conflicts with the tissue it is seen against. The indentation level is therefore
  uncertain by about 5 mm.
- The first about 2.5 s (blurred pan) have lower weight. Tip errors there are still 1-4 px, but the depths are noisier.
- `grasp_apex_sheet` depends on membrane/v05's mask. If that track changes its mask, rerun this version.
