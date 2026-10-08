# backdrop v02 — conservative masks, robust fusion, texture fixes

Changes vs v01: masks dilated more (gallbladder 12 px, strand 10, instruments 12, border 8) plus a 45 px disk at
the grasper's jaw end (tented sheet); depth from sharp frames (k >= 62) wherever >= 5 of them see a cell, else all;
2 rounds of outlier rejection (> max(2.5 MAD, 2.5 mm)); domain = cells inside >= 15 frames' footprints; hidden
cells kept >= 3 mm behind the 90th-percentile gallbladder front depth; texture with per-frame colour gains,
sharp/blurred median blending and push-pull hole filling.

Metrics (all frames, median): depth 2.43 mm, L1 8.95, local NCC 0.70, gallbladder violations 1.4 % (was 7 %),
spread 3.06 mm. Seen: bed fill smooth (no starburst) but blurry; the canvas edge still folds (pan-only cells
with inconsistent depth); frame 0 loses its right edge (domain >= 15 frames). -> v03.
