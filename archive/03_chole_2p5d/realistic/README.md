# Photoreal real-to-sim of a laparoscopic cholecystectomy clip

Source: Wikimedia Commons, case-report video "gallbladder torsion" (CC BY 2.0; Boer, Boerma, de Vries Reilingh),
26–32 s, 640×360, 25 fps (`runs_real/chole_sweep/task/video.mp4`). A grasper holds the gallbladder neck, a straight
probe sweeps the adjacent tissue, the laparoscope drifts and zooms slightly.

## Pipeline (`chole_recon.py` → `chole_sim.py` → `chole_metrics.py`, variants in `chole_variants.py`)
| step | what | how |
|---|---|---|
| instruments 2D | tip + axis per frame | colour segmentation (bright rod; low-saturation grasper shaft), gaps interpolated |
| camera | scope motion | rotation + zoom per frame fitted to 200+ LK tracks on static tissue (median residual 4.5 px) |
| geometry | tissue surface | depth prior (smooth relief, gallbladder dome, peritoneal-fold dome), scaled to the instrument contacts; no learned depth |
| instruments 3D | tip depth, ports (RCM) | apparent shaft width → depth (5 mm shafts); ports fitted to the observed shaft axes near the scope plane (0.7° / 2.0° axis error) |
| appearance | textures | frame 0 projected onto every surface; instruments removed with a clean plate (temporal fill from frames where the tissue is visible) |
| tissue | MuJoCo flex shell | 670 vertices, shell elasticity (`elastic2d="both"`), per-vertex anchoring to the bed, rim pinned, gravity-compensated |
| interaction | grasp + push | grasper holds the neck through a grasp constraint; probe only through contact, aimed never more than 1.5 mm into the current surface |
| rendering | endoscope look | emission-dominant tissue with a scope spotlight for wet highlights, steel/teal instruments, mild bloom |

Tissue parameters were fitted to the real feature tracks (`chole_tune.py`).

## How close is it (`chole/metrics.json`)
| | this reconstruction | baseline |
|---|---|---|
| image SSIM over the clip | **0.586** | static first frame 0.502 · blind-agent primitive scene 0.466 |
| image PSNR over the clip | **15.8 dB** | static first frame 14.9 · blind-agent 9.8 |
| first frame SSIM | 0.85 | — |
| tissue tracks, all (308) | **5.6 px** | nothing moves 13.7 px |
| … static backdrop (241) | 4.9 px | 13.3 px |
| … gallbladder (43) | 23.1 px | 31.4 px |
| … peritoneal fold (24) | 13.1 px | 18.3 px |
| probe / grasper tip | 4.1 / 2.5 px (median) | — |

## Limits
- Depth is a hand-designed prior calibrated at two contact points, not measured; views far from the original
  camera expose the 2.5-D relief.
- Tissue that was never visible (always under the probe near the image corner) is inpainted.
- The gallbladder's own motion is only partly reproduced (23 px vs 31 px baseline); the peritoneal strands are not
  lifted by the probe as in the video.
- One 6 s clip. Euler at 0.25 ms is needed for stability (≈30 s per simulated clip on an M5 Max).

## New interactions (`chole/variants.mp4`)
The same scene driven by actions that never happened in the video: 2× neck retraction, and the probe indenting
the gallbladder body. These render with the same appearance and camera, which is what makes the reconstruction
usable as a data source.
