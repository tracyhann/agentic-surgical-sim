# interaction v02 (superseded by v05; see v05/NOTES.md)
Changes from v01:
- Edge-based mask axis for short masks.
- V.depth on the instrument's own pixels sets the depth level (sigma 6 mm).
- kappa (apparent-width scale) is fitted: grasper 1.000, probe 0.895.
Result: grasper tip 1.5 px, angle 3.0 deg, IoU 0.81. Probe tip 5.9 px, IoU 0.80. Probe indentation in contact +2.2 mm.
Seen: the probe tip is about 10 px off the end of the mask in many frames, because 12 centre-line samples outweigh the
single tip sample.
