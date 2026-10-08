# ducts v09 - attachments made consistent with gallbladder v10 (integration feedback r01-r04)

`python -m r2s.tissue.ducts v09` (CFG: gb_version 'v10', anchor_from 60, distal_fixed False, fibre_ramp 0.35,
w_curv 200, max_kr 0.85; everything else as v08; masks = v08's). Variant with fixed distal ends:
`work/variants/distal_fixed` (`--cfg=distal_fixed=True`).

## Feedback and what changed
1. Proximal ends vs gallbladder v10. v08 ends were 1.4 mm (duct) / 2.6 mm (strands) median, up to 5.6 mm, from the
   v10 surface per frame, 5.6 mm in rest. Now every tube's proximal end IS v10 vertex 290 (rest_verts[290] in rest,
   verts4d[k, 290] in frame k; hard constraint). 290 is the vertex among the 50 v10 marks 'ducts' closest to the
   observed duct junction (mean 4.5 mm over frames >= 60). The nearest v10 vertex to every proximal node is 290, so
   the integrator's nearest-vertex glue has zero offset.
2. Hooks / 1.2x stretch. Two causes found:
   - v08 fibres converged at both ends (offsets followed the band half-width, which tapers at the mask tips), so the
     outer fibres had S-bends at the ends. Now the fibres leave the anchor together, open smoothly over the first
     35 % of their length and stay parallel to the distal end.
   - The boundary conditions. With the proximal end on v10 vertex 290 and the distal end fixed in the world, the
     reconstruction itself must change length: duct 0.96-1.21 x rest, fibres 0.94-1.16 x with end-to-end 15-27.6 mm,
     i.e. strong bowing (variant `distal_fixed`). The v10 neck tip moves differently from the observed junction by
     2-9 mm, mostly in depth (v10 is 4-12 mm nearer the scope than the depth map in frames 0-125). So v09 keeps the
     distal ends free per frame. They move up to 4.3 mm (duct) and 9.7 mm (fibres) from rest, and the tubes then
     keep their length within 0.96-1.12 x rest.
   - Also: one common 4D length per tube (fitted), rest refitted to that length, and a curvature barrier (bend radius
     >= tube radius / 0.85) so the sharper bend next to the anchor does not fold the rings. v09 without it had 17
     inverted tets in frames 100-103; now 0.
3. `cable_spec` keeps its keys and adds `least_stretched_frame`, `nodes_least_stretched` (nodes4d of the frame with
   the shortest centreline), `length4d_m` (251,) and `anchor` {gallbladder: 'v10', vertex: 290}. `nodes4d` is now
   resampled by each frame's own arc length, so node 0 / node n-1 are always the ends.

## Result (keyframes, this track's masks)
| | v08 duct | v09 duct | v08 strands | v09 strands |
|---|---|---|---|---|
| IoU mean (min) | 0.836 (0.73) | 0.815 (0.69) | 0.745 (0.47) | 0.730 (0.32, f80) |
| boundary F | 0.878 | 0.848 | 0.700 | 0.708 |
| depth residual median mm | 1.73 | 3.17 | 1.80 | 1.88 |
| proximal -> v10 surface, abs median / max mm (rest) | 1.4 / 5.6 (5.6) | 0.14 / 0.34 (0.21) * | 2.6 / 5.6 (5.5) | 0.14 / 0.34 (0.21) * |
| rest length mm | 18.3 | 17.5 | 13.4 | 19.4-19.8 (3 fibres) |
| 4D length / rest | 0.98-1.02 | 0.97-1.12 | 0.98-1.04 | 0.96-1.09 |
| least-stretched frame | - | 101 | - | 100 / 151 |
| distal end motion max mm | 4.3 | 4.3 | 3.5 | 9.7 |
\* The vertex lies exactly on v10. The 0.1-0.3 mm residual is the surface sampling of the distance check.

Mesh: 4 watertight tubes, 1038 verts. Duct tets: 828, none inverted in rest or any frame. Clearance duct-fibres
min -0.23 mm (they share the anchor). Union IoU 0.798 (v08 0.807).

## Known problems / advice
- The fibre rest length grew from 13.4 to 19.5 mm. The anchor sits 5-8 mm from the observed junction, mostly in
  depth, so the bundle now runs back to it. If the gallbladder track pins its neck tip to the video depth at the
  junction, this would shrink back.
- Simulation:
  - Glue proximal node 0 of every cable to v10 vertex 290.
  - Drive the distal node along `nodes4d[:, -1]` (as the liver bed is driven) instead of fixing it. Fixed distal
    ends reproduce the hooks and stretch.
  - Fibres are strings, so model them tension-only: a spatial tendon through the nodes with a max-length limit
    `length_m`, or chain springs without the skip-one springs. Skip-one springs push back under compression and
    buckle into hooks.
  - No collisions among the fibres or between fibres and duct near the anchor.
