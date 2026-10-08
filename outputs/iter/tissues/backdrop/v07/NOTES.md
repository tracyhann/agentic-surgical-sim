# backdrop v07 — pan-only cells nearly free

Change vs v06: data weight of cells that no sharp frame sees x0.03 (was x0.2), so the thin plate continues the
sharp surface there instead of following the blurred pan frames' depth. Added the gallbladder gap metric, per-vertex
bed weight / ray direction / confidence, and a MuJoCo height field export (verified with mj_ray against the mesh:
<= 0.1 mm, 0.6 mm at one steep point).

Metrics: all-frame medians as v06 (depth 2.54 mm, L1 9.11); pan frames unchanged (depth 3.9 mm, local NCC 0.48);
area 113 -> 109 cm2 (the edge fin flattened). Gap backdrop - visible gallbladder front: median 9.0 mm (p10 4.9).
Seen (residual maps with refined cameras): the pink strands beside the cystic duct light up (they move with the
neck and are in no clip mask). -> v08.
