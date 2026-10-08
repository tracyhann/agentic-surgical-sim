# ducts v08 (final of this track's first pass)

`python -m r2s.tissue.ducts v08` (code: `r2s/tissue/ducts.py`, `CFG` as committed; 40 s on CPU, SAM masks reused
from `masks.npz`). Optional `--diag` (length-free / depth-free 4D variants -> `work/diag.json`, run in v07),
`--cfg=key=literal` overrides, `--resegment` (re-runs SAM).

## Anatomical reading (look at `work/sheet_zoom.jpg`)
- The dark purple **neck** belongs to the gallbladder (clip mask `gallbladder`). It ends at the junction where a shiny
  purple-white **cord** leaves it and runs down-right for ~18 mm until it dives into the fat (lower right). This cord
  continues the neck, is ~3.8 mm thick and is what the clip mask `strand` covers: it is modelled as the **cystic duct**.
  (The brief said "dark purple tube running right/up-right"; in the frames the only tube that continues the neck runs
  down-right. If the integrator reads the cord as the cystic artery instead, nothing in the geometry changes.)
- From the same junction a **band of 2-3 thin pink fibres** runs right / slightly up for ~13 mm to the tissue at right
  (cystic artery / peritoneal strands). The fibres are not separable in the masks (ridge filters only pick the
  specular dots), so the band is fitted as one bundle and laid out as **3 parallel fibre tubes** across its width.
- Distal ends: both sit on the clip's `right organs` region (distance 0 px in all checked frames; liver ~18 px from the
  strand end) -> `backdrop`. Proximal ends -> `gallbladder` (neck).

## Masks (`masks.npz`, (251, 360, 640) bool)
- `duct` = clip SAM mask `strand` unchanged (it is the cord). The clip's `gallbladder` mask overlaps the cord in frames
  ~170-250 (up to 2600 px); the gallbladder track should not use those pixels.
- `strands` = new SAM 2.1 object (`views.segment_object`), prompts (x, y):
  f0 pos (360,195) (390,192) (415,195) neg (370,250) (470,220) (400,160) (300,200);
  f20 pos (370,195) (400,192) (430,200) neg (380,280) (490,200) (420,165) (290,240);
  f40 pos (410,172) (430,170) (470,172) neg (430,235) (545,175) (450,130) (300,200);
  f100 pos (430,158) (460,152) (490,163) neg (440,205) (560,150) (470,120) (350,190);
  f160 pos (445,125) (470,127) (490,138) neg (440,180) (570,140) (480,100) (330,150);
  f220 pos (490,150) (515,156) (535,160) neg (480,215) (585,130) (500,120) (400,160).
  Weak frames: 70-90 (the probe shaft covers the band's left half; only the right part is masked), 20-40 (blur).
- Instrument pixels (`V.occluders`, dilated) are unknown in every term and metric.

## Method
1. Per frame and tube: largest mask component -> skeleton -> longest geodesic path -> smoothed, extended along the
   end tangents to the mask outline, ordered proximal -> distal (duct: upper end; strands: left end). Duct: the
   proximal end is pulled to the nearest gallbladder-mask pixel (outside the dilated duct mask) when within 12 px
   (= the neck junction; found in 204/250 frames). Strands: the left end is joined to the duct junction when within
   45 px (173 frames). Samples every 3 px with video depth (5 px median) and mask half-width (distance transform).
2. Rest (multi-view): one centreline (duct 24, strands 20 samples) fitted jointly to keyframes 40..250 step 5
   (43 views, each through its own camera R, f, pos): robust 2D chamfer both ways (curve <-> skeleton, sigma 3 px),
   end points, depth of the centre = video depth + radius (robust, sigma 2.5 mm), bending + even spacing.
3. 4D: all 251 centrelines jointly (Adam, 900 its), same data terms per frame + temporal 1st/2nd differences + soft
   length towards the rest length + weak pull to the rest; strands' proximal end pulled (robust, 1.5 mm) to the duct's
   proximal end of the same frame.
4. Radius: median over frames of the half-width x depth / f per centreline sample; duct held constant over the
   proximal 15 % (the mask stops short of the neck there). Per frame the duct section is an ellipse s(t)r x r/s(t)
   (same area; s = observed / model width, Gaussian-smoothed, clipped 0.75-1.4) -> follows the apparent widening while
   the probe works at the neck (frames 50-110, s up to 1.26) without changing volume. Strands: the fan opening
   (fibre spacing) scales with the band's observed width s(t) (0.75-1.36).
5. Mesh: rings of 12 + capped ends, ring frame e1 = tangent x mean viewing direction (identical rule in rest and every
   frame, no twist). Duct also gets interior centreline vertices and a conforming tet mesh (prism fans split 3 ways).
   Fibres: 3 tubes at offsets {+1, 0, -1} x (r_bundle - r_f) across the band, r_f = r_bundle / 3 (packed, touching).

## Result (keyframes every 10 frames, 26 views; masks = this track's masks)
| | duct | strands (3 fibres) |
|---|---|---|
| silhouette IoU mean (min) | 0.836 (0.726, f40) | 0.745 (0.465, f80) |
| boundary F (4 px) mean | 0.878 | 0.700 |
| depth residual median (mm) | 1.73 (f30/f40: 6.8 / 10.8) | 1.80 |
| centreline reprojection (px, all frames) mean / median | 2.35 / 1.88 | 2.84 / 2.12 |
| rest (static multi-view) IoU mean | 0.571 | 0.586 |
| length rest / std over time (mm) | 18.28 / 0.11 | 13.36 / 0.18 (fibres 13.4-13.7) |
| radius (mm) | 1.1 at neck, 2.2 mid, 0.8 distal tip; median 1.88 | fibres 0.3-0.67 (band half-width 0.5-2.0) |
| max / p95 vertex speed (mm/frame) | 1.0 / 0.84 | 0.79 / 0.56 |
| max acceleration (mm/frame^2) | 0.50 | 0.52 |
| centreline segment stretch vs rest p95 (max) | 1.09 (1.30) | 1.11-1.16 (1.42) |
Union of both vs union of masks: IoU 0.807. Mesh: 4 watertight components, 1038 verts, 2016 faces, min edge 0.16 mm,
0 degenerate; duct tets 828 (312 nodes), 0 degenerate, 0 inverted in rest and in all 251 frames.
Attachments: strand proximal end -> duct proximal end median 0.36 mm (max 8.8 mm, frames 50-90); duct proximal end ->
gallbladder mask median 1.2 px (max 33 px at f50-80 where the neck mask is broken by the probe); duct proximal end ->
gallbladder track surface: v01 median 1.9 mm (rest 1.0 mm), v02 median 8.6 mm (rest 10.9 mm; their v02 put the
template neck on the wrong side, their NOTES say so). Clearance duct-fibres >= 0.40 mm (no interpenetration).

Diagnostics (v07 `work/diag.json`): without the length term the observed length alone varies 14-22 mm (duct) and
12-24 mm (strands, long only in frames 50-90 where the band's visible left end is ~8 mm from the junction) - mask
extent noise, so the constant length is kept. Without the video depth the 4D fit stays within 2 mm in depth of the
main fit (camera parallax + smoothness agree with the depth maps; weak test, initialised from the rest).

## Iterations (change -> numbers -> what I saw -> next)
| ver | change | duct IoU / BF / depth | strands IoU / BF | union IoU | notes |
|---|---|---|---|---|---|
| v01 | skeleton + depth tube fit, rest (multi-view) + 4D, junction 25 px | 0.799 / 0.815 / 1.80 | 0.720 / 0.685 | 0.771 | duct bridged 20-30 px gaps to a broken neck mask (f50-90); strand start floats (up to 11 mm from duct start) |
| v02 | junction 12 px, strand start pulled to duct start (w 5), wider end-occlusion test | 0.800 / 0.819 / 1.74 | 0.721 / 0.705 | 0.773 | strand-duct start 2.4 -> 0.36 mm median; duct mesh 25 % narrower than mask at f50-110 |
| v03 | strands as 3 fibre tubes across the fitted band; duct tets; camera-view 3D renders | 0.800 / 0.819 / 1.74 | 0.727 / 0.713 | 0.775 | tets healthy; implied radius varies 1.6-2.6 mm over time (mask width, not geometry: length agrees with depth) |
| v04 | elliptical area-preserving duct section s(t), fan spread s(t), strand anchor w 20 | 0.843 / 0.892 / 1.74 | 0.670 / 0.645 | 0.785 | duct fixed; strong anchor drags strands off the band at f50-100 (IoU 0.0-0.4) |
| v05 | anchor back to 5, thinner fibres (r/4) | 0.843 / 0.892 / 1.74 | 0.612 / 0.573 | 0.751 | gaps between fibres cost IoU/BF |
| v06 | packed fibres (r/3) | 0.843 / 0.892 / 1.74 | 0.745 / 0.700 | 0.809 | best numbers; but radius tapers to 0.8 mm at both duct ends (mask tips) |
| v07 | constant-radius ends (duct both, strands distal), constant fibre radius; `--diag` | 0.814 / 0.811 / 1.73 | 0.698 / 0.605 | 0.774 | distal flat end protrudes where the cord dives into fat (BF -0.07); variants in `v07/work/variants`: proximal-only flat = 0.836/0.878 |
| v08 | proximal 15 % flat only, packed fibres; `cable_spec` helper | 0.836 / 0.878 / 1.73 | 0.745 / 0.700 | 0.807 | final |

## For the integrator
- `model.npz`: `rest_verts` (1038, 3) = static multi-view fit (consistent with gallbladder v01's rest at the neck,
  1.0 mm); `verts4d` (251, 1038, 3) float32; `faces` (2016, 3); `tets` (828, 4) duct only; `tube_id` per vertex,
  `tube_names` = duct, strand_1 (image-top) .. strand_3; `attach_idx` (104) + `attach_to` (same length, strings):
  proximal ring + cap of every tube -> `gallbladder`, distal ring + cap -> `backdrop`.
  Extra: `<tube>_rest_centerline` (S, 3), `<tube>_centerline4d` (251, S, 3), `<tube>_radius` (S,);
  `strands_bundle_*` (fitted band), `duct_section_scale4d`, `strands_spread4d` (251,).
  Vertex layout per tube: rings `i*12 + a`, then proximal cap centre, distal cap centre (+ duct interior centres).
- Centreline samples are equally spaced along the curve in every frame; they are not material points (sliding along
  a textureless tube is unobservable), so judge stretch by total length (constant within 1 %), not per segment.
- Simulation: duct as a MuJoCo cable (elasticity `cable` plugin or a capsule chain) of 12-16 segments along
  `duct_rest_centerline` with `duct_radius`, proximal node welded to the gallbladder neck (the neck flex/body vertex
  nearest to the duct's proximal end; gap to gallbladder v01 ~2 mm), distal node fixed to the world. If the
  gallbladder is a flex and contact matters, the duct tets can be a second dim-3 flex connected at the 13 proximal
  attach vertices instead. Strands: 3 thin cables (radius ~0.3-0.7 mm) with very low bending stiffness (strings,
  tension only), proximal ends on the neck next to the duct, distal ends fixed; disable collisions among the fibres
  (they touch by construction) and between fibres and duct near the junction. `ducts.cable_spec('v08')` returns
  node chains, radii and ends. Mass is negligible (0.2 ml in total). Stiffness is not observable from this clip.
- Driving: the duct follows the neck: its proximal end moves up to 9.5 mm (p95 7.7 mm) from its mean position in
  world over the clip (centroid +-4 mm); the 4D centrelines are the targets to compare a simulation against.

## Known problems
- Strands in frames 50-90: the probe covers the band's left half and the visible band starts ~8 mm from the duct
  junction, so the fibres' proximal ends sit up to 8.8 mm from the duct's start (IoU 0.46-0.70 there).
- Blurred pan (frames 0-40): masks are soft; duct depth residual 6.8-10.8 mm at f30-40 (the depth maps jump while
  the temporal smoothness holds the tube in world).
- The static rest explains the moving tissue poorly (IoU 0.57-0.59): the tube centroids move +-4 mm in world over the clip,
  the duct's neck end up to 9.5 mm.
- The duct radius tapers 2.2 -> 0.8 mm over the distal 20 % (visible cord narrowing as it enters the fat); real duct
  calibre beyond is unknown. Fibre count (3) and positions inside the band are a layout, not individually measured.
- Gallbladder-mask leak onto the duct in frames 170-250 (clip mask, not this track's).
