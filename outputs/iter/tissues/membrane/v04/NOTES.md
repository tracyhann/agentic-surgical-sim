# membrane v04 (superseded by v05; full log in ../v05/NOTES.md)
- Changes vs v03: 25 rays instead of 13 (grid 17 x 25 = 425 vertices) and a 98th-percentile envelope, to follow the
  outline better.
- Metrics: IoU 0.795 (min 0.638), BF4 0.486 (+0.07), BF8 0.742, depth residual lower/upper 1.12/1.28 mm, stretch max
  mean 2.29 (finer base edges slide more).
- Scratch check: the unsmoothed per-frame fit reaches IoU 0.90, while the mask's own adjacent-frame IoU is 0.85.
  Temporal smoothing costs 0.08-0.11 IoU. Next: lighter smoothing.
