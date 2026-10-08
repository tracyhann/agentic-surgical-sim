# membrane v02 (superseded by v05; full log in ../v05/NOTES.md)
- Changes vs v01: ruled surface between the 5 mm jaw line and the base (p = 1). Base curve resampled to equal 3D arc
  length. Mask pruned more than 6 px proximal of the jaws along the 2D shaft axis. masks.npz is a copy of v01's.
- Metrics: IoU 0.772 (min 0.615), BF4 0.415, stretch max mean 2.34 (max 3.56), depth residual lower/upper 1.50/1.91 mm.
- Tested and rejected (scratch): stronger base smoothing or a world-fixed base (multi-view consensus). IoU fell to
  0.62 (sigma 25) and 0.47 (fixed), and stretch did not improve. The mask's extent really changes when the grasper
  lifts more after frame ~110, and it also flickers.
- Next: use the video depth inside the sheet where it is meaningful.
