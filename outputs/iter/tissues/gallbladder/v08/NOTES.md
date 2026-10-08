# gallbladder v08

`python -m r2s.tissue.gallbladder v08` (`CFG['v08']` = v07 config; masks.npz here: tent apex 90 px).

## Change vs v07 (why)
v07's inverted tets were clusters of sub-millimetre tets that TetGen's quality refinement put around short surface
edges: the decimated marching-cubes surface has edges down to 7 % of the median (checked: 0.074-0.086), and
non-adjacent vertices can be projected close together. Now (`tetrahedralize`, tetgen_q branch):
- `remesh_projected` also reverts non-adjacent vertex pairs closer than 35 % of the median edge;
- `collapse_short`: surface edges < 45 % of the median collapsed to their midpoint before TetGen;
- candidates (500/16, 500/14, 400/12, 300/11 faces/cells) until min/median tet edge >= 0.1 (else the best).
A first v08 run without the collapse failed outright (all candidates had min edge ratio 0.02-0.05).

## Result
| metric | shared SAM mask | body+neck mask (90 px apex) |
|---|---|---|
| silhouette IoU mean | 0.749 | 0.819 |
| boundary F (4 px) mean | 0.294 | 0.332 |
| depth residual median | 1.49 mm | 1.39 mm |

- Tets: 472 nodes, 1788 tets, 646 surface faces, min quality 0.0017, 2.2 % slivers (< 0.01).
- 4D: no well-shaped tet inverts; 2 zero-weight slivers (quality < 0.003, no energy by design) invert in 45/51
  sampled frames. Edge stretch p95 <= 1.16, volume 0.94-1.05.
- Rest 6.1 x 4.3 x 4.1 cm, 31.7 ml. Max vertex speed 3.5 mm/frame, centroid range up to 7 mm.

## What I saw
- Clean, small tet mesh (good for MuJoCo) but coarse surface: outline and depth worse than v06/v07 (shared-mask IoU
  0.749 vs 0.791; part of the drop is the larger unknown tent apex, which the shared mask counts as gallbladder).

Next (v09): finer remesh (800 faces, 20 cells) with the same clean-up, all tets weighted (as in v06).
