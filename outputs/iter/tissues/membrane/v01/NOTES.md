# membrane v01 (superseded by v05; the full iteration log is in ../v05/NOTES.md)
`python -m r2s.tissue.membrane v01 --seg` (SAM 2.1 mask, prompts in ../v05/NOTES.md and in masks.npz), then
`python -m r2s.tissue.membrane v01`.
- Model: polar fan from the image apex (sheet pixels nearest the grasper mask's distal end) to 13 base points
  (97th-percentile radius of the mask per angle bin), lifted with the video depth. Concave profile
  `X = A + s(Bc-A) + s^p (B-Bc)`, with p fitted per frame to the silhouette (mostly 1.0). Temporal smoothing in world:
  apex sigma 2, base sigma 4. Rest = minimum-area frame (71).
- Metrics: IoU 0.745 on the raw mask (0.734 on the cleaned mask used from v02 on), BF4 0.39, stretch max mean 6.5
  (max 18.5).
- Seen: silhouettes follow the mask, but the per-frame p makes the tiny near-apex edges change like s^p, which gives
  the stretch outliers. The SAM mask includes jaw/shaft pixels above the apex in some frames (e.g. 100).
- Next: ruled surface (p = 1), prune the mask above the jaws, resample the base by arc length.
