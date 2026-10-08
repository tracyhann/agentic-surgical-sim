# gallbladder v07 (failed: tiny-tet cluster inverts)

`python -m r2s.tissue.gallbladder v07` (`CFG['v07']` on top of v06). Masks: `masks.npz` here (tent apex 90 px).

## Change vs v06 (why)
- Tent apex unknown out to 90 px from the grasper jaw tip (v06 mesh climbed into the tent up to the jaw).
- Denser data: 7000 surface samples (4000), 1800 coverage pixels (1200) per frame -> sample spacing ~3.5 px for a
  tighter outline (boundary F was 0.31-0.36); Gaussian sigma 1.0.
- First run exploded at frame 150 around a 7e-5-quality sliver weighted 0.0035: tets below quality 0.003 now carry no
  ARAP / volume / barrier energy (`sliver_weight=0.003`), rerun.

## Result (rerun)
| metric | shared SAM mask | body+neck mask (90 px apex) |
|---|---|---|
| silhouette IoU mean | 0.792 | 0.859 |
| boundary F (4 px) mean | 0.335 | 0.393 |
| depth residual median | 1.25 mm | 1.10 mm |

The outline / depth fit is the best so far, but the volume mesh is broken: up to 378 inverted tets (359 of them
well-shaped), min det F -2688, edge stretch p95 up to 5.

## What I saw / diagnosis
- The inverted tets are a cluster of well-shaped but tiny tets (rest edges ~0.2 mm, typical 2-3 mm) around two
  nodes 0.05 mm apart: `remesh_projected` moved two non-adjacent marching-cubes vertices (a thin fold between neck and
  body) onto nearly the same surface point; it only checked mesh edges. TetGen then refined that spot. Any relative
  motion > 0.2 mm inverts them; surface motion of the nodes was only ~8 mm max, so the surface (and the silhouette
  metrics) is fine, the interior is not usable for simulation.
- Contact sheet: the mesh top still reaches towards the jaw in frames 150-250, less than v06; the apex pixels are now
  unknown (no penalty either way), so this part is unconstrained by the video.

Next (v08): same config; remesh check for near-coincident non-adjacent vertices (query_pairs), and a tet mesh with
any edge < 15 % of the median edge is rejected for the next resolution.
