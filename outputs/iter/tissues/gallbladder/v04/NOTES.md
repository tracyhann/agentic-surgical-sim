# gallbladder v04

`python -m r2s.tissue.gallbladder v04` (`CFG['v04']` on top of v03; masks as v01).

## Change vs v03 (why)
- v03 lost outline pixels to the visible-only test (mesh > 6 mm behind the measured depth near the outline):
  observed depth points now up to 1 px from the outline (3x3 erosion instead of 7x7), 1500 per frame (800),
  depth weight 1.5 (1.0).
- Lattice smoothness 10 (3): v03 rest had a notch between neck and body.
- Temporal weight 0.3 (0.1), Gaussian sigma 1.5 frames (1.0), 100 iterations per frame (80).

## Result
| metric | shared SAM mask | body+neck mask |
|---|---|---|
| silhouette IoU mean | 0.792 | 0.838 (min 0.759) |
| boundary F (4 px) mean | 0.309 | 0.345 |
| depth residual median | 1.35 mm | 1.30 mm |

- Rest: scale (0.88, 1.23, 1.71) -> 5.3 x 4.9 x 4.2 cm, 30.0 ml; static multi-view IoU 0.60, depth 3.8 mm.
- Temporal: max speed 3.2 mm/frame (v03 4.2), p99 0.81, max acceleration 1.6 mm/frame^2 (v03 3.3).
- Health: 1785 nodes, 6566 tets; but 15 inverted tets in 13 of 51 sampled frames (min det F -0.88), max principal
  stretch 2.1. Rigid part <= 7.2 deg / 4.1 mm.

## What I saw
- Fit slightly better everywhere, smoother in time.
- The inversion barrier (v02-v04) is averaged over ~6000 tets, i.e. worth ~1/6000 per inverted tet: useless once the
  depth term pulls harder. The rest shape is too round (5.3 cm long vs 4.9 cm wide; length scale 0.88 < template)
  and keeps a concavity between neck and body.

Next (v05): barrier summed over tets (weight 0.5); length scale bounded to >= 1.0 (fundus continues past the lower
image border in every frame, so the visible 5 cm are not the full length).
