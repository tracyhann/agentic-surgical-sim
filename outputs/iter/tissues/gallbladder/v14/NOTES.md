# gallbladder v14 (volume-preserving 4D on v12's mesh)

`python -m r2s.tissue.gallbladder v14` (`CFG['v14']`). It reuses v12's rest shape, tets, indexing and attachments
(`rest_from='v12'`, byte-identical) and refits only the 4D. Changes vs v12's 4D:
- a total-volume term `w_gvol=5`: free within +-0.5 % of the rest volume, then 5 x ((|V/V0 - 1| - 0.005) / 0.01)^2;
- tet ARAP 22 (v12 30, v13 15);
- 130 iterations per frame.

v15 (same config with ARAP 30) is worse on every measure and is kept only for the record (../v15).

## Why
Integration r16/r17:
- v12 simulates better than v13 (sim IoU 0.655 vs 0.637).
- Both 4Ds "breathe" 27.0-29.7 ml (+-4 %) while the sim keeps its volume.
- The question: can a volume-preserving 4D keep v13's outline gains at v12-like rigidity?

## Result (refined cameras, every 3rd frame; IoU / BF 4 px / depth residual median mm)
| band | v12 shared | v13 shared | v14 shared | v15 shared |
|---|---|---|---|---|
| 0-61 | 0.786 / 0.283 / 0.98 | 0.799 / 0.342 / 0.87 | 0.785 / 0.310 / 1.02 | 0.776 / 0.292 / 1.09 |
| 62-129 | 0.805 / 0.315 / 0.89 | 0.825 / 0.358 / 0.70 | 0.800 / 0.305 / 1.05 | 0.790 / 0.289 / 1.10 |
| 130-250 | 0.794 / 0.261 / 0.98 | 0.822 / 0.297 / 0.85 | 0.797 / 0.252 / 0.99 | 0.787 / 0.247 / 1.09 |

Own masks: IoU / BF / depth (mm) for v12 -> v14:
- 0-61: 0.832 / 0.304 / 0.93 -> 0.838 / 0.325 / 0.92
- 62-129: 0.824 / 0.312 / 0.78 -> 0.822 / 0.309 / 0.90
- 130-250: 0.859 / 0.312 / 0.90 -> 0.864 / 0.303 / 0.91

| | volume over the clip | rotation vs rest, median / max | shift median / max | non-rigid rms max | max vertex speed |
|---|---|---|---|---|---|
| v12 | 27.12-29.74 ml | 10.4 / 16.9 deg | 3.5 / 6.7 mm | 2.51 mm | 3.8 mm/frame |
| v13 | 27.04-29.75 ml | 7.5 / 13.5 deg | 3.0 / 5.6 mm | 2.73 mm | 4.9 mm/frame |
| v14 | 28.42-28.72 ml (+-0.5 %) | 8.3 / 12.9 deg | 2.8 / 6.0 mm | 2.59 mm | 2.7 mm/frame |
| v15 | 28.42-28.72 ml | 9.0 / 14.0 deg | 2.8 / 6.3 mm | 2.52 mm | 3.2 mm/frame |

- No inverted tets in any sampled frame (min det F 0.28). Edge stretch p95 <= 1.22.
- Duct coverage <= 2.1 % per band. Sheet front <= 0.3 %.

## Reading
- v13's outline gains do not survive the volume constraint. With the volume fixed, v14 fits like v12: same IoU,
  BF +0.03 in the pan, -0.01 later, shared-mask depth +0.0..0.16 mm. Part of v13's gain came from breathing volume, i.e. from
  following per-frame depth noise.
- v14 is the more physical reference:
  - volume constant like the sim's;
  - less rigid rotation than v12;
  - smoothest in time (max vertex speed 2.7 mm/frame).
- Expected effect in the sim: the rigid / non-rigid errors and the free-vertex motion are scored against this 4D,
  and v14 no longer asks for volume changes the sim cannot make.
- Not tested in the sim here. It is a drop-in for v12: same mesh and indexing.
