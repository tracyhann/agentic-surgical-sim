# membrane v05: tented peritoneal sheet (shot A, chole_a); superseded by v06 (depth-consistent with gallbladder v10)

`python -m r2s.tissue.membrane v05` (code `r2s/tissue/membrane.py`, settings `CONFIGS['v05']`; about 30 s on CPU,
the SAM mask is reused from v01).

## What it models
The left grasper holds a thin, translucent peritoneal sheet and lifts it into a tent above the gallbladder. The model
is a single-layer triangulated shell (17 x 25 grid: 425 vertices, 768 faces, open sheet, 1 component):
- row 0 (vertices 0..24) is the line gripped in the jaws: 5 mm wide, centred on the apex A, oriented along the base.
- row 16 (vertices 400..424) is the base curve B(t) on the gallbladder surface.
- in between it is a ruled surface: `X(s,t) = (1-s) * jawline(t) + s * B(t)`, plus a depth offset along each camera ray
  (see step 4). The side columns (t = 0 and t = 1) are free edges.

## Pipeline
1. Mask (`masks.npz`): SAM 2.1 (`views.segment_object`) with the prompts below (made in v01, 3 min on MPS).
   `membrane` = raw SAM output, `membrane_clean` = what all metrics use: 7 px close, 5 px open, largest component,
   holes filled, and pixels more than 6 px proximal of the jaws along the grasper's 2D shaft axis removed (SAM put the
   jaws or shaft into the sheet in some frames, e.g. frame 100).
2. Image apex per frame: the 30 sheet pixels nearest to the grasper mask's distal end (the jaws are not in the grasper
   mask, so the end of the mask is where the sheet enters the jaws). 3D apex = unprojected at the median video depth
   of the sheet pixels within 10 px.
3. Base per frame: polar envelope of the mask around the apex. 25 rays between the 1st and 99th percentile angle of the
   sheet pixels, radius = 98th percentile of the pixels in each ray's bin. Each base point is lifted with the video
   depth 4 px inside the mask (the sheet lies on the gallbladder there, so that depth is the organ surface), then the
   curve is resampled to equal arc length in 3D.
4. Translucent sheet and depth: the video depth is not used for the free part of the tent. The ruled surface between
   the jaw line and the base sets the shape. Then the vertices are moved only along the camera rays (so the silhouette in
   that frame does not change), toward the video depth: weight 1 on the lower half (sheet on or just above the
   gallbladder), 0.2 on the upper half (sheet in front of a far background), 0 off the mask or on instruments,
   jaw and base rows fixed, grid Laplacian regulariser (lambda 2).
5. 4D: apex median 5 + Gaussian sigma 1.5 frames, base median 9 + sigma 3 (world frame), depth offsets sigma 3.
   `verts4d` (251, 425, 3) has no NaN.
6. Rest shape = the frame with the median sheet area (frame 176). Stretch is measured against it.

Extra arrays in model.npz: `grid_shape` (17, 25); `apex` (251,3) smoothed jaw point (world m); `base` (251,25,3)
smoothed base curve; `apex_raw`, `base_raw` per-frame observations before smoothing; `apex2d` (251,2) image apex (px);
`depth_offset` (251,425) ray offsets of step 4 (m); `profile_p` (all 1 = ruled surface).

`attach_idx` / `attach_to`: vertices 0..24 -> 'grasper' (jaw line), vertices 400..424 -> 'gallbladder' (base curve).

### SAM prompts (x, y px; frame 0 is inside the blurred pan)
| frame | positive | negative |
|---|---|---|
| 0 | (105,200) (115,250) | (60,300) (230,60) (50,60) (300,200) |
| 40 | (175,190) (200,230) (165,150) | (150,330) (120,60) (280,100) (330,250) |
| 100 | (210,130) (225,170) (245,215) (275,160) | (140,300) (300,40) (160,30) (440,120) (100,120) |
| 150 | (215,115) (225,150) (240,200) (270,170) | (130,300) (300,30) (165,30) (420,110) (100,120) |
| 200 | (240,130) (260,180) (280,140) (300,110) | (140,320) (330,40) (180,20) (460,140) (120,120) |
| 250 | (215,120) (250,140) (265,180) (300,130) | (120,320) (330,60) (185,40) (470,160) (120,120) |

Positive points are on the translucent sheet below the jaws and along its glossy ridges. Negative points are on the
lower gallbladder body, the liver/fat behind, the grasper shaft, the probe and the neck region. Overlay of every 10th
frame: `../v01/work/masks_overlay.jpg`. The mask is a tent from the jaws to the gallbladder in every frame. Two parts
change from frame to frame: the lower part (frames 120-160 run far down the body) and a branch toward the probe and
neck (frames 190-250). The adjacent-frame self-IoU of the mask is 0.85 (0.78 during the pan).

## Iterations (keyframes 0..250 step 10; all rows measured against the same cleaned mask)
| ver | change | IoU mean (min) | BF 4px | BF 8px | depth res. lower / upper (mm) | stretch max mean (max) | p95 | accel mm/f² |
|---|---|---|---|---|---|---|---|---|
| v01 | polar fan, concave profile exponent p fitted per frame, 13 rays | 0.734 (0.583)* | 0.355* | 0.628 | 1.56 / 1.96 | 6.5 (18.5) | 4.15 | 0.099 |
| v02 | ruled surface (p=1), base resampled by 3D arc length, mask pruned above the jaws | 0.772 (0.615) | 0.415 | 0.688 | 1.50 / 1.91 | 2.34 (3.56) | 2.10 | 0.097 |
| v03 | + depth refinement along camera rays, rest = median-area frame | 0.775 (0.615) | 0.414 | 0.688 | 1.15 / 1.44 | 1.67 (2.52) | 1.37 | 0.116 |
| v04 | 25 rays instead of 13, 98th percentile envelope | 0.795 (0.638) | 0.486 | 0.742 | 1.12 / 1.28 | 2.29 (3.90) | 1.44 | 0.119 |
| v05 | lighter temporal smoothing (apex sigma 1.5, base 3) | **0.808 (0.674)** | **0.512** | **0.780** | 1.11 / 1.25 | 2.63 (4.37) | 1.49 | 0.153 |

(*) v01 was re-scored on the cleaned mask. Its own quality.json uses the unpruned mask (IoU 0.745, BF4 0.390).
v01 -> v02: the stretch outliers (18x) came from the per-frame exponent p changing the tiny near-apex edges like s^p.
The ruled surface removed them and IoU rose. v03: the first depth solve blew up (area 240 cm2) because of a Laplacian
assembly bug (fancy-index += does not accumulate; fixed with np.add.at). After the fix the lower-half depth residual
went from 1.50 to 1.15 mm. v04: finer rays follow the outline (BF4 +0.07). v05: smoothing costs IoU: the unsmoothed
per-frame fit reaches IoU 0.90 against a mask that flickers itself (self-IoU 0.85 adjacent, 0.78 at +-2 frames).
sigma 1.5/3 is the compromise. Acceleration rose from 0.12 to 0.15 mm/frame².

Other v05 numbers (quality.json):
- IoU on every frame is 0.803: 0.744 in the pan (frames 0-62), 0.822 after it.
- Base vertices vs the video depth on gallbladder pixels: median 1.0 mm. 95 % of the base vertices project onto the
  gallbladder mask.
- Projected mesh apex vs the image apex: median 1.7 px, max 10 px.
- Area 1.36-5.73 cm2. Lift (apex to base centroid) 10.5-25 mm.
- Stretch p05 0.68: in the least-lifted frames the sheet is about 30 % compressed vs rest.
- Mesh health: one open component, no degenerate faces, min edge 0.14 mm (jaw row).
- Cross-checks: base -> gallbladder v01 surface median 1.0 mm. Apex height above gallbladder v01 is 6.8 mm mean
  (max 9.6).

## Apex vs grasper tip
- `V.tools['grasper_left']['tip']` projected vs the image apex: median 12.8 px, max 70 px (frame 0). It is 20-40 px off
  in frames 40-110 and 4-12 px after frame 120. In 3D, mesh apex vs that tip: mean 5.7 mm, max 9.8 mm.
- interaction v01 `grasper_left_tip`: 9.5 px median (max 30 px) in the image, but 7.4 mm median (max 21 mm) in 3D.
  The difference is mostly along the viewing ray (depth). Their tip is 1.2 mm farther at the median and up to 18 mm off.
- The apex depth comes from the video depth at the jaws, which is noisy (65-87 mm over the clip, about 1.4 mm/frame
  jitter before smoothing).

## Known problems
- The mask is the weak link: it flickers, and its lower extent and the branch toward the neck vary. A temporally
  smooth model cannot follow both, so per-keyframe IoU ranges 0.67-0.91. Frames 40-50 and the pan are worst.
- A polar fan from the apex cannot represent non-star-shaped masks (the branch toward the probe in frames 190-250).
  The fan fills the gap between that branch and the body.
- Vertex identity is "same ray index / same arc-length fraction", not material. When the visible tent grows (area
  x4 after frame ~110, when the grasper lifts higher) the base vertices slide along the gallbladder. That sliding
  causes the stretch maxima (2.6). Edges are about 1.5x at p95.
- The depth of the free tent is a model assumption (ruled surface plus weak pull). The video depth there is not the
  sheet's, and the depth residuals partly measure the refinement's own fit.
- Single layer: the back face of the fold and the sheet beyond the side edges are not modelled.

## For the integrator
- Attach vertices 0..24 to the grasper jaws, following `apex` (or your corrected jaw point; the offset is above). Attach
  vertices 400..424 to the gallbladder (nearest surface point of the GB mesh at the rest frame; they lie about 1 mm
  from gallbladder v01). The side edges are free.
- Material starting points (not measured, to be tuned): a thin shell 0.2-0.5 mm thick, almost no bending stiffness,
  soft in-plane (MuJoCo flex dim 2, Young's modulus of the order 1e4-1e5 Pa, Poisson 0.4, some damping). The
  observed sheet recruits material as it is lifted (area x4), so a stiff sheet with fixed rest lengths would pull the
  gallbladder hard. Use either a soft membrane or a rest shape with slack. For rendering: translucent (alpha about
  0.4-0.5), glossy specular.
