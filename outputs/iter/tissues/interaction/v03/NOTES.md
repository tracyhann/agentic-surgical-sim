# interaction v03 (superseded by v05; see v05/NOTES.md)
Changes from v02: tip first (sigma 1 px), centre-line samples weighted by exp(-d_tip / 120 px) with sigma 2 px,
smoothness 1 mm/frame^2.
Result: tips 0.8 px (grasper) and 1.2 px (probe). But the probe's centre-line offset rose to 5.5 px and its IoU fell to
0.66. The measured tip (mean of the farthest pixels) sits off the axis because the probe's end cap is asymmetric.
Indentation in contact +5.2 mm.
