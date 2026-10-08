# ducts v11 - proximal ends on the gallbladder v12 surface in every frame (integration feedback r14-r25)

`python -m r2s.tissue.ducts v11 --cable-test`. CFG: gb_version 'v12', anchor_mode 'patch', bridge_frac 0.2,
rest_ref 'median_frame', refined cameras as v10, distal ends free as v10. Masks are v08's. Runtime about 4.5 min,
mostly the quasi-static cable test.

Variants in `work/variants/` (quality.json, sheets):

| variant | proximal end | result |
|---|---|---|
| `vertex1381` | the one fixed v12 'ducts' vertex closest on average (4.1 mm mean, 12.5 max) | duct IoU 0.78, strands 0.60, lengths up to 1.44x |
| `patch_fit` | patch point, but a hard-anchored refit as v09 | lengths stable (0.97-1.16) but the depth dip is back: bow 3.9 / 3.4 mm in rest |
| `bridge_0.1`, `bridge_0.3` | other bridge fractions | no material change |

## What v11 does
1. **Attachment patch.** The patch is v12's 18 triangles with at least 2 of the 10 vertices marked 'ducts'.
2. **Anchor point per frame.** For each frame, take the point of that patch closest to the observed duct junction
   (v10's free start, video depth). Smooth it over time (Gaussian, sigma 3 frames) and project it back onto the
   patch (exact closest point on the triangles). Stored as `anchor_point4d` and `anchor_face4d` (face index in
   v12 `faces`). In rest, the point lands on v12 vertex 694 (offset 0).
3. **Bridge.** In every frame, the first 20 % of each centreline (duct and strand bundle) is replaced by a cubic
   Hermite bridge. It runs from the anchor, along the chord, to the observed path and joins it tangentially; the
   rest of the observed path is untouched. The fibres stay together along the bridge and open over the next 25 %.
4. **Rest shape.** Each tube's 4D shape in the frame (>= 62) with the median length (duct frame 248, strands 159),
   moved so its start sits on the anchor point in the rest gallbladder. So the rest carries no slack or
   pre-stretch. The static-rest IoU in quality.json (0.30 / 0.22) is therefore not comparable to earlier versions.

## Result
Proximal end to the v12 surface, exact closest-point distance over all 251 frames:

| | median / max (rest) |
|---|---|
| duct, before (v10 path) | 2.3 / 9.3 mm (4.7) |
| strands, before (v10 path) | 3.1 / 9.3 mm (4.7) |
| all tubes, v11 | 0.000 / 0.000 mm (0.000) |

| band | duct IoU (excl. gallbladder px) / BF / depth mm | strands IoU (excl.) / BF / depth mm |
|---|---|---|
| 0-62 | 0.762 (0.816) / 0.768 / 3.2 | 0.600 (0.615) / 0.474 / 1.3 |
| 62-130 | 0.803 (0.816) / 0.821 / 1.3 | 0.597 (0.601) / 0.542 / 1.6 |
| 130-250 | 0.815 (0.889) / 0.873 / 1.6 | 0.740 (0.764) / 0.767 / 1.9 |
| all | 0.798 (0.852) / 0.833 / 1.69 | 0.669 (0.686) / 0.636 / 1.60 |

"Excl." = IoU with the gallbladder's pixels outside the tube mask treated as unknown. The bridge over the neck is
the main IoU loss against v10 (duct 0.835, strands 0.711).

Depth bow (max deviation from the chord along the view direction):

| | rest | 4D median | 4D max |
|---|---|---|---|
| duct | 0.37 mm | 0.69 mm | 3.2 mm |
| strands | 0.19 mm | 0.65 mm | 2.8 mm |

For comparison: v09 4D max 3.6 / 5.0 mm, v10 0.9 / 1.7 mm. The v11 maxima fall in the pan frames, where the patch is
up to 9.6 mm from the observed junction.

Length over time / rest:

| tube | rest | 0-62 | 62-130 | 130-250 |
|---|---|---|---|---|
| duct | 18.8 mm | 0.99-1.43 | 0.96-1.11 | 0.96-1.04 |
| fibres | 15.6-16.1 mm | 0.97-1.47 | 0.98-1.48 | 0.86-1.08 |

The long fibres in 62-130 are frames 50-90, where the visible band starts about 8 mm from the junction.

Overlap with v12: 3 % of the duct's pixels are hidden behind it (v10 vs v10 gallbladder: 20 %). Duct tets: none
inverted in rest; at most 1 inverted in any frame.

## End-motion test (13-node chains, motion explained vs the 4D)
| | duct | fibres |
|---|---|---|
| linear interpolation of node 0 + node 12 | 0.74 | 0.85 |
| plus one driven node (duct 4, fibres 5) | 0.92 | 0.94 |
| plus two (duct 4,8 / fibres 5,9) | 0.96 | 0.97 |
| proximal end only | 0.31 | 0.42 |
| quasi-static spring chain, ends driven | 0.53 | 0.74 (strand_2) |
| quasi-static, plus one mid node | 0.84 | 0.87 |

The greedy interior node moved: duct 6 -> 4 (the bridge sits in the first fifth). Fibres stay at 5 (v10: 4).

## Known problems / advice
- The bridge absorbs the gap between v12's neck patch and the observed junction (2.1 mm median after frame 130,
  up to 9.6 mm in the pan). This shows as length change in frames 0-130 rather than as a depth dip.
- Driving node 0 along `nodes4d[:, 0]` now keeps the duct on the v12 neck by construction.
- `cable_spec` keys are unchanged. `anchor` = {gallbladder 'v12', vertex 694 (rest), offset 0}; the per-frame
  attachment point is `model.npz['anchor_point4d']`.
