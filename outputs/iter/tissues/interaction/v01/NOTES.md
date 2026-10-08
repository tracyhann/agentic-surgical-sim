# interaction v01 (superseded by v05; see v05/NOTES.md for the full record)
Settings: `VERSIONS['v01']`. Fixed port and per-frame distal end, fitted to the mask tip (mean of the farthest pixels),
the centre line, and the apparent widths of a 5 mm shaft (kappa = 1).
Mask axis: PCA plus a centre-line refinement. Both were later changed: border-cut bins excluded, edge-based axis for
short masks. Rerunning `v01` with the current code therefore gives slightly different numbers.
Result: tip error 13.8 -> 2.1 px (grasper) and 28.3 -> 5.9 px (probe); IoU 0.67 -> 0.79 and 0.72 -> 0.85.
Problems:
- The probe tip lies 15 mm (median) behind the tissue around it.
- The distal probe shaft is behind the tissue it is seen against in 63 % of frames. The width ruler and V.depth
  disagree by about 12 % around the probe.
- The grasper axis angle error of 15 deg comes from a wrong PCA axis on the short grasper masks (frames 120+).
- work/ holds the exploration crops (frames, zooms of the jaws and the probe).
