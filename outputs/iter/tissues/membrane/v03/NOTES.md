# membrane v03 (superseded by v05; full log in ../v05/NOTES.md)
- Changes vs v02:
  - Per-frame depth offsets along the camera rays (the silhouette is unchanged). Weight 1 on the lower half, 0.2 on the
    upper half (translucent sheet in front of a far background), 0 off the mask or on instruments. Jaw and base rows
    fixed. Laplacian lambda 2. Offsets smoothed sigma 3 frames.
  - Rest = median-area frame (112).
- First run blew up (area up to 240 cm2): the grid Laplacian was assembled with fancy-index += (no accumulation, not
  PSD). Fixed with np.add.at.
- Metrics: IoU 0.775, BF4 0.414, depth residual lower/upper 1.15/1.44 mm (from 1.50/1.91), stretch max mean 1.67,
  p05 0.70.
