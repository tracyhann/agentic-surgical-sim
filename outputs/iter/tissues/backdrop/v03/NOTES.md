# backdrop v03 — confidence-weighted surface fit

Changes vs v02: one sparse solve for the whole height field: min sum w (z - z_med)^2 + lam1 |grad z|^2 +
lam2 |lap z|^2 with w = min(n,50)/50 / (1 + (MAD/3 mm)^2), x0.2 where no sharp frame sees the cell; hidden
cells w = 0 (filled by the same solve), gallbladder lower bound as heavy constraints; domain >= 6 frames (frame 0
covered again); border 12 px; liver-like high-frequency detail copied into the filled texture.

Metrics (all frames, median): depth 2.55 mm, L1 8.99, local NCC 0.70, gallbladder violations 0, spread 3.07 mm.
Seen: skirt/fin smaller, filled texture no longer a flat blob. Per-frame depth errors are smooth low-frequency
warps of the per-frame depth maps (+-10 mm blobs that change from frame to frame), not defects of the surface.
-> v04 (template), then smoothing weights chosen by a hold-out test.
