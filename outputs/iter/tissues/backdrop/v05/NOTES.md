# backdrop v05 — thin plate, held-out keyframes, warp consistency

Changes vs v03: lam1 = 0 (pure thin plate, chosen by the hold-out test of v04); texels need >= 3 observations;
filled-texture detail clipped to +-12 grey levels; no template. New checks: keyframes (every 10th frame) left out
of depth and texture fusion and evaluated (`quality_holdout.json`); warp consistency (frame k lifted with a depth
and re-projected into frame j, compared without the texture).

Metrics (all frames, median): depth 2.56 mm, L1 8.99, local NCC 0.70, spread 3.07 mm.
Held-out keyframes vs the same keyframes in the full model: L1 8.34 vs 8.28, local NCC 0.743 vs 0.760,
depth 1.94 vs 1.92 mm -> the surface/texture generalise to unseen frames.
Found: local NCC 0.80 at BA keyframes but 0.69 mid-way between them; ECC registration of the render to the video
gives 0.6 px misalignment at keyframes, ~3 px between them (up to 14 px in the pan): the clip's cameras between
keyframes are interpolated. -> v06.
