# r2s — real-to-sim of laparoscopic clips

A config-driven pipeline that turns a surgical video clip into a MuJoCo scene whose instruments replay the video's
motion and whose organ deforms under them, then scores the simulation against the video.

```
python -m r2s.batch <clip> [<clip> ...]            # whole pipeline, logs in outputs/logs/
python -m r2s <clip> <stage> [conditions...]       # one stage (prep scene organ sim eval render compare all)
python -m r2s.shape_study <clip> <object> <template>   # static shape study of one object (stereo-scored if available)
python -m r2s.export site/viewer/data <clip>[:conds] ...   # data for the web viewer
python -m r2s.inspect_masks <clip>                 # SAM masks over the clip (to place extra prompts)
python -m r2s.longvideo <out.mp4> <clip>:<cond>:<title> ...   # join per-shot 2x2 videos with title cards
python -m r2s.multiview <clip>+<mode> <mode>       # joint keyframe solution; modes ba / sift / vggt_tracks / vggt
python -m r2s.batch <clip>+<mode>                  # whole pipeline in that geometry -> outputs/variants/<clip>+<mode>/
python -m r2s.views_eval <clip> <dir>              # compare camera geometries (held-out matches, image alignment)
```

## Clips (`r2s/clips/*.json`)

| clip | source | frames |
|---|---|---|
| `chole_6s` | Commons cholecystectomy video, 26–32 s (the earlier 6 s window; regression check) | 650–800 |
| `chole_a` | same video, shot A, 23.5–33.5 s | 587–837 |
| `chole_b` | same video, shot B, 33.5–41.4 s (dissector, camera pan) | 838–1034 |
| `endonerf_pulling` | EndoNeRF `pulling_soft_tissues` (da Vinci prostatectomy, stereo depth) | 0–62 |

A config names the source, camera (field of view or focal length; `static` when the scope does not move), the
objects with their SAM prompts and roles (`organ`, `instrument`, `strand`, `static`), the instruments (mask, shaft
diameter, which organ a holding instrument holds and from which frame) and optionally a reference depth.

## Stages and modules

| stage | module | output (`outputs/<clip>/`) |
|---|---|---|
| prep | `perception.py`, `scene.track_camera` | `prep/`: SAM 2.1 masks, Depth Anything V2 depth, scope rotation + zoom, metric scale from shaft widths |
| scene | `scene.py`, `instruments.py` | `scene/`: panoramic textures + backdrop mesh, organ observations (amodal mask, depth, thickness, bed motion), strand ribbons, instrument ports and joint targets |
| organ | `organ.py` | `<cond>/organ.npz`: the organ body of a condition (tetrahedra, texture coordinates, anchors) |
| sim | `sim.py` | `<cond>/traj.npz`, `scope.mp4`, `sim.json` |
| eval | `evaluate.py` | `<cond>/eval.json` (simulation and static baseline) |
| render, compare | `render.py` | `<cond>/views.mp4`; `compare/conditions.mp4`, `conditions.jpg`, `summary.json` |

## Organ conditions (the template experiment)

| condition | organ body |
|---|---|
| `measured` | measured front surface over the amodal mask, extruded by a thickness inferred from the silhouette |
| `primitive` | ellipsoid fitted to the same observations (pose + three radii): what a coding agent typically builds |
| `template` | organ template (BodyParts3D) fitted by pose + anisotropic scale |
| `template_fit` | the template, then a smooth 4×4×4 free-form deformation |
| `template_mv`, `template_mvd` | experiment, run only when named: the template fitted to several keyframes (rigid / with the `template` simulation's motion compensated); `outputs/experiments/template_keyframes_2026-10-07` |

All four see the same observations (visible surface points at measured depth, amodal outline, occlusion by the
measured surface) and share everything else in the scene.

## Metrics (`evaluate.py`)

- **outline IoU**: simulated organ (occluded by the background where it lies behind it) vs the SAM organ mask,
  restricted to pixels no instrument or strand covers.
- **depth error**: median |z_sim − z_ref| over the visible organ; `monocular` = the calibrated Depth Anything depth
  the scene was built from (not independent), `stereo` = EndoNeRF's stereo depth (independent).
- **track error**: KLT tracks of real organ texture, re-seeded every 25 frames and followed for 2 s, vs the simulated
  surface points bound to them at their seed frame (pixels); independent of any depth estimate. `track_err_px` is the
  pooled median, `track_by_seed` the per-seed medians. **Motion phases** (`motion_seeds`, `track_motion`): seeds where
  the static baseline of the measured condition is >= 15 px off; the change against static there is the headline
  tracking number. (Until 2026-10-07 tracks were seeded once at frame 0; shot A then had 35 samples.)
- `static` baseline: the organ held in its frame-0 state while the scope moves.

## Camera models

Default: the scope tip is fixed; each frame has a rotation and a zoom, fitted to KLT tracks on static anatomy
(`camera.camera_track`); textures come from a panorama on a canvas larger than the picture (`scene.panorama`).
`"camera": {"static": true}` fixes it (EndoNeRF). `"camera": {"motion": "6dof"}` is experimental: rotation and
translation from frame-to-frame PnP on monocular depth (`camera_track_6dof`, `refine_to_keyframes`), keyframe
background patches instead of the panorama. On shot B it aligned worse with frame 0 than rotation + zoom (gradient
NCC 0.45 vs 0.58 at frame 150), so no clip uses it; `camera.alignment_ncc` is the check.

## Results (2026-10-07)

| clip | condition | outline IoU | depth err (mm) | track err (px) | motion phases vs static |
|---|---|---|---|---|---|
| chole_6s | measured / primitive / template / template_fit / static | 0.763 / 0.701 / 0.713 / 0.733 / 0.775 | 5.6 / 5.0 / 5.5 / 5.4 / 6.0 | 9.3 / 11.2 / 11.7 / 10.9 / 12.2 | -34 / -33 / -32 / -34 % |
| chole_a | measured / primitive / template / template_fit / static | 0.531 / 0.613 / 0.548 / 0.509 / 0.523 | 13.9 / 9.4 / 9.7 / 9.8 / 14.1 | 8.9 / 9.1 / 9.1 / 8.9 / 9.5 | +1 / +5 / +7 / -3 % |
| chole_b | measured / primitive / static | 0.346 / 0.262 / 0.388 | 11.4 / 12.6 / 11.3 | 13.9 / 16.6 / 13.7 | +3 / +19 % |
| endonerf_pulling | measured / primitive / static | 0.495 / 0.515 / 0.601 | stereo 18.2 / 19.3 / 15.6 | 6.5 / 5.7 / 5.6 | -58 / -66 % (1 seed) |

Since 2026-10-07 evening (`outputs/experiments/anatomy_fix_2026-10-07`): the gallbladder neck belongs to the organ in
chole_a / chole_6s (it was segmented as a strand, cutting the organ in two), strands are tied to the organ along a band
of vertices (not one), and the grasper draws the tissue patch into its jaws (`grasp_ramp`) instead of holding it at
the reconstruction gap. Earlier numbers (e.g. chole_a template -20 %) rested on the cut organ and are superseded.

EndoNeRF shape study (seminal vesicle, frame 0, vs stereo): measured 15.3 mm, ellipsoid 13.6, template 13.7,
template_fit 14.8; the monocular depth itself is 24 % too far there (7 % over the pulled tissue).

## Multi-view (experimental)

`multiview.py` solves keyframe poses, one focal length and per-keyframe depth jointly (bundle adjustment on static
anatomy: flow tracks, instrument shafts as rulers). Modes add wide-baseline correspondences: `sift` (SIFT between all
keyframe pairs, MAGSAC-verified), `vggt_tracks` (VGGT-1B point tracks, `vggt_views.py`), `vggt` (VGGT also as initial
values). A clip runs in a mode as the variant `<clip>+<mode>` (`config.VARIANTS`; perception symlinked from the base
clip). Findings: flow-only BA gives local consistency and a ~33 deg field of view (`outputs/experiments/multiview_2026-10-07`);
SIFT cuts the held-out long-range error by 30-45 % and is the best geometry; VGGT is out of domain on laparoscopy
(narrow-FOV / translation interpretation, biased tracks) (`outputs/experiments/wide_baseline_2026-10-07`). Better
geometry does not give better tissue motion (shot A measured -2 to +6 % against static, templates worse, shot B not
reproduced), so the single-view pipeline stays the default.
