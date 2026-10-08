# ducts v10 - refined cameras, no depth dip, cable end-motion test (integration feedback r07)

`python -m r2s.tissue.ducts v10 --cable-test` (CFG: cams = backdrop v08 `cams_refined.npz`, anchor_mode
'junction', fibre_ramp 0.35, distal ends free; masks = v08's). About 3.5 min, most of it the quasi-static cable test.
Variants in `work/variants/`:
- `gb_vertex290`: v09-style hard anchor on v10 vertex 290, plus depth targets blended towards the anchor.
- `gb_front`: additionally keeps the tubes in front of v10 along camera rays. Rejected: lengths 27-28 mm, depth
  residual 6.5-8 mm.

## 1. Depth dip / curl
- **Diagnosis.** The video depth where the duct leaves the neck is about 82-84 mm; v10's neck tip (vertex 290) is
  4-12 mm nearer the scope in frames 0-125. In v09 the hard anchor therefore made the tubes descend from the anchor
  to the video depth:
  - rest centreline bowed from its chord in depth by 2.6 mm (duct) and 1.9 mm (strands);
  - in 4D by up to 3.6 / 5.0 mm;
  - the strands grew from 13.4 to 19.4 mm.
- **Tried and not enough.** Blending the depth targets toward the anchor depth along the tube changed little
  (bow 2.9 / 2.2 mm). Putting the tubes in front of v10 made things much worse (`gb_front` above). Vertex 290
  is already the best v10 vertex: none follows the junction or keeps the duct's chord length better.
- **v10 fix.** The proximal ends stay where the video shows the duct leaving the neck, depth included, as in v08.
  The glue target is the v10 vertex nearest to the rest proximal end: **v244, rest offset 5.7 mm**. This is stored
  in `anchor_gb`, `anchor_offset_rest`, `anchor_rest_point`, and `cable_spec(...)['anchor']` gives each tube's
  offset.
- **Result.**
  - Depth bow in rest: duct 0.83 mm, strands 0.27 mm. In 4D: max 0.92 / 1.69 mm.
  - Lengths: duct 18.5 mm, fibres 13.3-13.6 mm. Over time 0.99-1.02 (duct) and 0.94-1.09 (fibres) of rest.
  - Fibres converge monotonically (ramp) at the junction: no S-bends.
- **Cost.**
  - Proximal end to v10 surface: median 1.5 mm, max 5.7 mm.
  - The observed junction drifts from (v244 + rest offset) by mean 4.4 mm, max 8.7 mm. This is the v10 neck
    motion mismatch.
  - 20 % of the duct's rendered pixels lie behind v10 (v10's neck covers 9-42 % of the duct mask in 2D: the
    gallbladder track treats the duct region as unknown, so its template neck grows over it).

## 2. Can a cable glued at node 0 and driven at node 12 reproduce the 4D?
Motion explained (scene4d's metric, 1 - mean |pred - recon| / mean |rest - recon|) on the 13-node chains:

| | duct | fibres (strand_1..3) |
|---|---|---|
| linear interpolation of the two end motions | 0.64 | 0.78-0.79 |
| rigid move with the end chord | 0.61 | 0.73-0.76 |
| proximal end only (distal fixed) | 0.46 | 0.41-0.42 |
| quasi-static cable (chain + skip-one springs), ends on their 4D tracks | 0.57 | 0.79 (strand_2) |
| ... plus one interior node driven (duct node 6, fibre node 4) | 0.87 | 0.90 |
| linear, ends + 2 / 3 driven nodes (duct 6,3 / 6,3,9; fibres 4-5,8 / +2) | 0.91 / 0.97 | 0.95 / 0.98 |

With the ends on the reconstruction, the cable model explains 57-79 % of the 4D. The integration got
-0.1..+0.08 because node 0 follows the simulated gallbladder, and v10's neck itself drifts 4.4 mm (mean) from the
junction.

Advice:
- Drive node 0 along `nodes4d[:, 0]` (or glue it to the gallbladder with a soft spring) and node 12 along
  `nodes4d[:, -1]`.
- Add a driven interior node: duct node 6, fibre node 4 (the greedy choice). For about 0.9, add duct 6 + 3 and
  fibres 4 + 8.

## 3. Silhouettes per frame band (keyframes; this track's masks; refined cameras)
| band | duct IoU / BF / depth mm | strands IoU / BF / depth mm |
|---|---|---|
| 0-62 | 0.834 / 0.865 / 3.4 | 0.641 / 0.518 / 1.3 |
| 62-130 | 0.806 / 0.816 / 1.4 | 0.673 / 0.635 / 1.5 |
| 130-250 | 0.849 / 0.902 / 1.7 | 0.766 / 0.797 / 1.8 |
| all | 0.835 / 0.872 / 1.74 | 0.711 / 0.684 / 1.53 |

`gb_vertex290` variant, same bands: duct 0.80 / 0.84 / 0.82 IoU, depth 7.6 / 5.9 / 1.2 mm; strands
0.69 / 0.61 / 0.80. The refined cameras change keyframe numbers only marginally (they register in-between frames).

## Other
- Mesh: 4 watertight tubes. Duct tets: none inverted in rest or any frame.
- Duct-fibre clearance >= 0.5 mm.
- Strands IoU 0.711 vs v08 0.745: the fibres now leave one point and open over 35 % of the length (no hooks).
- `cable_spec` keys are unchanged. `anchor` now has `offset_rest_m` and `on_vertex` (False here).
- Upstream fix that would remove the offset: the gallbladder track pins its neck tip to the video depth at the
  junction (about 7-9 mm further from the scope than now in frames 0-125) and stops the neck growing over the
  visible duct.
