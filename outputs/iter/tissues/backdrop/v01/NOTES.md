# backdrop v01 — plain multi-view fusion

`python -m r2s.tissue.backdrop v01 --baseline` (code: `r2s/tissue/backdrop.py`, params `VERSIONS['v01']`)

- Static pixels: not gallbladder (dilated 8 px), strand (8), instruments (6), 3 px image border.
- Virtual reference camera = mean pose of the 251 cameras; canvas 834x492 px covering every frame's footprint.
- Every frame's static pixels back-projected with its depth + camera, re-projected into the reference camera,
  binned in 2 px cells; per cell median over frames (count >= 3) -> height field; hidden cells harmonic fill.
- Grid mesh every 6 canvas px (9.8k verts, 19k faces); texture = per-texel median colour over every 2nd frame
  (visible + static + depth test), Telea inpainting of holes.
- Liver mask: own SAM 2.1 masks (`../masks.npz`, prompts in the module: `LIVER_PROMPTS`); the clip's 'liver'
  mask only covers a sliver at the top right.

Metrics (all 251 frames, median; evaluation mask = v02's static mask, same for every version):
depth residual 2.52 mm, photometric L1 8.77, NCC 0.959, local NCC 0.70, coverage 1.00, front violations 10 %,
**gallbladder violations 7 %** (backdrop in front of the visible gallbladder), spread (MAD across frames) 3.7 mm.
Old pipeline (frame-0 depth backdrop + keyframe patches, `work/render_sheet_old.jpg`): depth 5.87 mm, L1 25.9,
NCC 0.70, local NCC 0.32, front violations 22 %, gallbladder violations 21 %.

Seen: starburst artefacts of the Telea inpainting in the gallbladder bed; rectangular seams where only the blurred
pan frames see; flaps/fins along the canvas edge (cells seen by 1-3 pan frames); high spread next to the probe and
the gallbladder (depth bleeds over mask edges). -> v02.
