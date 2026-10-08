# backdrop v04 — BodyParts3D liver template (negative result)

Same surface as v03 plus a similarity fit (rotation within 70 deg of a supine anatomical prior, scale 0.55-1.36,
1500x4 coarse candidates, 12 least-squares refinements) of the BodyParts3D liver together with its gallbladder (same
atlas frame) to the observed liver cells and the observed gallbladder front surface (frames 150-250).
Output: `liver_template.npz` (world m) and `work/liver_template.jpg`.

Hold-out test of the hidden shape: a 60 px band of observed liver cells along the rim of the gallbladder bed is
removed, then predicted. Median |error| vs the fused depth:
- smooth fit alone (lam1 0.3, lam2 3): 1.19 mm
- template alone: 2.96 mm (covers 85 % of the band)
- smooth fit with the template as weak prior in unknown cells: 2.26 mm
Template fit: scale 1.31-1.36, 59-77 deg away from the prior, 1.5-2.2 mm to the liver points, 2.9-3.7 mm to
the gallbladder points, but 3-9 % of observed cells end up behind the template surface; the whole frame lies inside
the template's silhouette (the visible patch is ~5x4 cm of a 20 cm organ), so the pose is unconstrained.
Decision: the template does not help the hidden shape; it is not used in the surface (`bed_from_template=False`).
Sweep of the smoothing weights with the same hold-out band (scratch): pure thin plate (lam1 = 0) best,
0.77 mm (liver band) / 1.61 mm (all observed cells along the rim) vs 1.19 / 2.53 mm with lam1 = 0.3 and
0.93 / 2.40 mm for harmonic fill. -> v05 uses lam1 = 0.
