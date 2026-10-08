# backdrop v06 — frames registered to the backdrop

Changes vs v05: every non-keyframe camera is registered to the textured backdrop (2D Euclidean ECC on static pixels
-> small camera rotation: shift dx, dy -> yaw/pitch, in-plane angle -> roll; 2 iterations), then the backdrop is
re-fused with these cameras. Keyframe (bundle-adjusted) cameras are kept (refining them too made keyframe-pair warp
consistency worse). `cams_refined.npz`: R per frame (pos, f unchanged), correction median 0.28 deg, max 1.9 deg.

Metrics with the clip cameras (contract): unchanged (depth 2.55 mm, L1 9.12, local NCC 0.69).
With the refined cameras: L1 8.12, local NCC 0.75 (after pan 0.80, now flat across keyframe offsets 0.78-0.81).
Warp consistency on 24 frame pairs that include non-keyframes (no texture involved): local NCC 0.646 -> 0.709,
L1 11.06 -> 9.72 -> the refined cameras are genuinely more consistent, not just fitted to the texture.
Seen: a thin wing still extends from the top-right canvas edge (cells only the pan sees). -> v07.
