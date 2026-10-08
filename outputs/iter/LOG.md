# Round log

## r00 baseline (before this study)
- Scene: single organ body per condition (gallbladder template in the SIFT multi-view geometry), strand ribbon, grasper
  pulls the closest gallbladder patch into its jaws, probe limited to 1.5 mm penetration.
- Metrics (gallbladder, keyframes every 10 frames, simulated 4D vs video): silhouette IoU mean 0.54 (min 0.30),
  boundary F (4 px) 0.12, depth residual median 4.2 mm.
- Visual: the gallbladder is a blob that covers the neck, the probe and the strands; no peritoneal sheet; the grasper
  holds the gallbladder body instead of lifting the sheet; the probe does not push in. Grasper TCP projects 60-90 px
  below the visible jaw tips (frames 0, 100).
- Decision: rebuild tissue by tissue (CONTRACT.md), then integrate.

## r00b integrator check (placeholder: the old gallbladder template, no other tissues)
- New: `r2s/scene4d.py` assembles tissue models into MuJoCo (gallbladder tets, sheet / duct shells, static backdrop),
  attachments from the models, grasp with pull-in, probe unclamped with a 2 cm tip capsule; evaluation per tissue
  (sim vs video masks, sim vs the tissue's 4D reconstruction, stretch), texture tracking (motion phases), probe push
  (tissue displacement near the probe tip, sim vs reconstruction), instrument tip lag; contact sheets + 3D side views.
- Placeholder run: 139 s. Texture tracking motion phases +9 % vs static; probe push sim 7.4 mm vs reference 9.6 mm.
- Visual (orbit.jpg): the grasper jaws sit INSIDE the gallbladder body for the whole clip (it should hold the sheet
  above it); the probe only grazes the surface. Confirms the user's point.

## tissue: membrane v05 (sheet track, 5 internal iterations)
- Own SAM mask (prompts 0/40/100/150/200/250), single-layer shell 17x25 between a 5 mm apex line in the jaws and a
  base curve on the gallbladder; depth only pulled along camera rays (translucent: its depth is unreliable).
- Keyframe IoU 0.81 (min 0.67), boundary F 0.51, stretch max 2.6. Mask flickers (consecutive-frame IoU 0.85),
  fast-pan frames 0-62 worst. Sheet area grows 1.4 -> 5.7 cm2 after frame ~110 (base slides on the gallbladder).
- Attach: vertices 0-24 grasper, 400-424 gallbladder. Apex vs old grasper tip: 5.7 mm (3D), 70 px at frame 0.
- My check of sheet.jpg: tent from the jaws onto the gallbladder in every keyframe; frames 190-250 the fan widens
  toward the probe (agent flagged it). Accepted as v05 for round 1.

## tissue: interaction v05 (instrument track, 5 internal iterations)
- Per instrument one least-squares fit: fixed port, per-frame tip, width scale; tip on the mask axis, centre line,
  5 mm shaft widths, V.depth on the instrument; grasper depth also from the sheet around its jaws.
- Tip reprojection median: grasper 15.4 -> 2.0 px, probe 29.6 -> 0.6 px; shaft angle 2.3 / 1.5 deg; shaft IoU
  0.84 / 0.76. Grasper closed on the sheet throughout; grasp point 4.6 mm from the sheet apex. Probe pushes into the
  gallbladder neck: contact frames 0-113, 142-250, indentation median +5 mm (max +11; uncertainty ~5 mm), out of view
  114-141. Port distance along the shaft poorly determined (11-21 cm across versions).
- Integrator updated: joints aim at `<name>_tcp` from the fitted port and heading; insertion slew 6 -> 15 mm/step;
  grasper jaw closed.

## tissue: ducts v08 (duct track, 8 internal iterations)
- The dark neck is part of the gallbladder mask; the tube continuing from it is a shiny purple-white cord (~3.8 mm
  thick) running down-right into fat after 18 mm (= clip mask 'strand') -> "duct"; a band of 2-3 pink fibres leaves
  the same junction to the right for ~13 mm (own SAM mask) -> 3 packed fibre tubes.
- Joint centreline fit over 43 keyframes (rest) and all 251 frames (4D) with smoothness + soft length; elliptical
  area-preserving section; watertight capped tubes; duct tets (828, none inverted in any frame).
- Duct IoU 0.836 / BF 0.878, strands 0.745 / 0.700 (union 0.807); depth residual 1.7 / 1.8 mm; duct length 18.3 mm
  (varies 0.11 mm over the clip). Rest (static) shape vs moving tissue IoU 0.57. Probe hides the band at 50-90.
- Attach: proximal ring + cap -> gallbladder, distal ends -> backdrop. Proximal end 1.9 mm from gallbladder v01.
- My check of sheet.jpg: duct and fibres follow the video in all keyframes. Accepted.

## tissue: backdrop v08 (background track, 8 internal iterations)
- Static pixels (minus dilated gallbladder/neck, strands, instruments, a 45 px disk at the jaws = sheet, own SAM
  pink-strand mask) of all 251 frames fused into a virtual reference camera: robust median per 2 px cell, then a
  confidence-weighted thin-plate height field (also fills the gallbladder bed, kept behind the gallbladder front).
  Median texture over frames; own SAM liver mask (the clip's liver mask is a sliver).
- Frames 62-250 medians: depth residual 2.34 mm (old pipeline 5.87), photometric L1 8.3 (25.9), local NCC 0.73
  (0.32), backdrop in front of the gallbladder 0 % (21 %). Held-out keyframes NCC 0.734 vs 0.747: generalises.
- Liver template rejected (hold-out bed rim error 2.96 mm vs 0.77 mm for the smooth fill).
- Side product: cams_refined.npz, per-frame rotation refinements (clip cameras ~3 px off between keyframes, up to
  14 px in the pan) -> candidate shared improvement for a later round.
- My check of render_sheet.jpg: clean textured background, no seams, bed filled with liver texture. Accepted.

## r00c integration fixes (before round 1)
- Problem 1, speed: sheet-gallbladder flex contact 80 ms/step (1.8 h per round) vs 6.9 ms without -> sheet/duct
  contacts limited to the probe (conaffinity 2); the sheet base is glued to the gallbladder anyway. Revisit later.
- Problem 2, NaN at the first step for every configuration. Isolated: (a) MuJoCo's in-plane shell term ('stretch')
  returns NaN on the sheet (sliver triangles down to 3 deg) and on the thin duct tubes (0.15 mm edges); bending alone
  is stable; edge collapsing chained the apex row into one vertex, voxel clustering made worse slivers (0.3 deg).
  Fix: shells = bend-only shell + a parallel 1D flex of edge springs over the same vertices (original meshes kept);
  ducts simulated as cables (1D flex, 13 nodes per tube from `ducts.cable_spec`, ends glued to the gallbladder /
  fixed in the fat), scored on their 4D centrelines and as rendered tubes. (b) the probe starts 8.5 mm inside the
  unindented rest tissue -> backed off along its shaft until clear and eased back in over 1 s.
- Integrator now follows the interaction track's TCP, port and heading (insertion slew 15 mm/step).
- Short-run stability: all tissues stable; the grasper holds the sheet's 25 apex vertices (gap 6.7 mm, pulled in).

## tissue: gallbladder v10 (gallbladder track, 10 internal iterations)
- Clip SAM mask with instruments, strands/duct and the tented-sheet tip (90 px around the jaws) as unknown;
  BodyParts3D template posed, scaled per axis and smoothly deformed to fit all 26 keyframes jointly through their
  own cameras (silhouette, depth, neck-tip landmark on the neck/duct junction, anatomical scale limits); TetGen with
  sliver clean-up (5302 tets); 4D per-frame fit with ARAP, volume, inversion barrier, soft bed anchor, temporal term.
- Keyframe IoU 0.789 / boundary F 0.320 / depth residual 1.13 mm (baseline 0.54 / 0.12 / 4.2); against its own
  body+neck mask 0.852 / 0.366 / 1.04. No inverted tets in any frame (v07: 378). Volume 0.91-1.06 of rest. Rest
  6.0 x 4.9 x 3.8 cm, 32 ml (template 6.1 x 3.4 x 2.1 cm, 15 ml; distended torsion case). Frames 0-90 (pan) fit
  poorly (IoU 0.2-0.4); region under the tent tip unconstrained; 123 slivers kept with reduced stiffness.
- Attach suggestions: 316 back-side vertices -> backdrop (liver bed), 50 near the neck tip -> ducts, 68 under the
  sheet -> membrane. Integrator: bed vertices get the soft bed spring (gb_k_bed), not the 20 N/m attachment spring.
- My check of sheet.jpg: body and tapering neck follow the video in all keyframes. Accepted.
- Integrator: per-tissue mask keys (gallbladder 'body', sheet 'membrane_clean', ducts 'duct'+'strands').

## r01 first integration (gallbladder v10, membrane v05, ducts v08 as cables, backdrop v08, interaction v05)
- 174 s. Gallbladder sim IoU 0.50 (its 4D recon 0.78, rest held still 0.55), motion explained -0.19, texture
  tracking (motion phases) -28 %, probe push sim 5.3 vs recon 5.0 mm. Sheet IoU 0.34 (recon 0.81), stretch 5.4.
  Ducts IoU 0.46, cables move less like the recon than staying still. Instrument tips follow plans (0.6 / 1.2 mm).
- Visual: the sheet forms a tent from the jaws to the gallbladder (frame 180) and the probe pushes into the neck
  (first time the interaction matches the video qualitatively). Backdrop bed cuts through the gallbladder (it was
  filled for a thinner organ); loose backdrop fragments at the top.
- Diagnosis: the rest shapes are mid-clip states (gallbladder ~frame 136, sheet frame 176); at frame 0 the tissue is
  already 4.2 / 13 mm from rest.

## r02 start from the reconstructed frame-0 state (rest shape stays the elastic reference)
- Change: jointed tissue vertices start at verts4d[0] / nodes4d[0]; backdrop bed pushed behind the gallbladder
  (convex-hull test, visual only).
- Result: no gain (gallbladder motion explained -0.18, tracking +8 %, ducts worse); the tissue relaxes back toward
  rest because nothing holds the frame-0 shape. Sheet apex at frame 0 is 15 mm from the grasper TCP.
- New metric: whole-body (rigid) vs non-rigid split per frame (Procrustes). The reconstructed gallbladder mostly
  swings: rotation 3-11 deg (about the vertical axis, -13 deg at frames 50-75 to +8 deg at 200-250) plus ~2 mm of
  deformation; sim deformation ~1.9 mm (right size), but no matching swing. Reverted to start=rest.

## r03 system identification sweep (7 configs in parallel, r2s/sweep4d.py)
- gb_k_bed 0.05/0.2/0.5 x mem_k 2/20 (+ start=recon). No setting fixes the swing: gallbladder rigid error stays
  3.6-4.2 mm; sim rotates 5.5-8 deg but about the wrong axis. gb_k_bed 0.5 keeps the best tracking (-28..-30 %) and
  duct IoU; mem_k 20 cuts sheet stretch 5.4 -> 3.0. Kept gb_k_bed 0.5, mem_k 20.
- Check: organ rotation vs camera rotation (relative to frame 136): organ up to 14 deg while the scope is within
  ~3 deg -> the swing is organ motion, not a camera artefact; its cause (liver bed / retraction) is outside the model.

## r04 liver bed driven by its reconstructed 4D trajectory (boundary condition from the video)
- Change: springs of the 316 bed vertices pull toward their reconstructed positions over time (qpos_spring updated
  every 20 substeps); new honest metric on the 1016 free vertices only.
- Sweep k_drive 1/5/20 (mem_k 20). k_drive 20: gallbladder IoU 0.49 -> 0.60, rigid error 3.7 -> 1.5 mm,
  non-rigid 2.7 -> 2.0 mm, free-vertex motion explained -0.26 -> +0.35, tracking -22 %, probe push 6.3 vs 5.0 mm;
  sheet IoU 0.32 -> 0.36; duct IoU 0.45 -> 0.35 (to check). Adopted: bed_drive=recon, k_drive=20, mem_k=20.

## r05 gallbladder stiffness sweep (bed driven, mem_k 20)
- gb_young 150 / 300 / 600 / 1200 / 2400 Pa. 1200 best overall: IoU 0.609, rigid 1.32 mm, non-rigid 1.79 mm,
  tracking (motion phases) -36 % (best so far), free-vertex motion explained 0.41, probe push 6.15 vs 4.95 mm,
  sheet IoU 0.39. 2400: tracking -30 %, sheet stretch 3.8. Adopted gb_young 1200.
- Found in r04/r05 overlays: the sheet is hidden behind the gallbladder in many frames, in the recon as well: the
  sheet v05 and gallbladder v10 reconstructions intersect (gallbladder bulges forward under the sheet, where the
  video does not constrain it). Sent back to the sheet track (v06: base on the v10 surface, nothing behind the
  gallbladder along rays, apex at the interaction grasp point) and the duct track (v09: proximal ends on the v10
  neck, fibre hooks / 1.2x stretch). Measured contact cost with real meshes: no sheet contact 0.9 ms/step, with
  sheet-gallbladder contact 17 ms (margins) / 3.7 ms (zero margins) -> enable with zero margins once v06 is in.
- Side check: refined cameras (backdrop v08) change nothing at keyframes (they only register in-between frames),
  so keyframe metrics are identical; deferred.

## r06 sheet v06 (consistent with gallbladder v10): contact and base options
- Sheet track v06: base on the v10 surface (0.43 mm median), no sheet vertex behind v10 (was 15 % per frame), sheet
  pixels hidden by v10 26 % -> 0.5 %, apex 1.05 mm from the jaw segment (was 3.7); IoU vs own mask 0.783; slivers
  down to 0.5 deg.
- Integration problems found and fixed: (1) v06 crossed the probe shaft at frame 0 (probe pushes under the sheet)
  -> the sheet no longer collides with the probe, only (optionally) with the gallbladder; (2) v06 slivers made the
  shell's bending term explode (finite but huge accelerations; earlier isolated tests missed it because MuJoCo
  resets the state after the warning) -> triangles with min angle < 15 deg are left out of the bending shell (their
  edges remain springs; scoring renders the full mesh); (3) gb_young >= 600 needs ts 0.0625 ms (was silently
  halved in r05) -> base ts 0.0625 ms. Integrator now names the exploding body.
- Sweep (contact 0/1 x base glued/driven): contact + driven base (base springs follow the reconstruction, as the
  base slides when more sheet is lifted): sheet IoU 0.39 -> 0.65, sheet rigid error 6.9 -> 2.5 mm, gallbladder
  IoU 0.62, tracking -30 %, free motion explained 0.43, probe push 5.7 vs 5.0 mm. Glued base gives the best
  gallbladder (rigid 1.06 mm, push 5.3) but sheet IoU 0.44. Adopted contact + driven base.

## r07 ducts v09 (proximal ends exactly on gallbladder v10 vertex 290)
- Duct track v09: ends on v10 (0.14 mm median, was 1.4), duct IoU 0.815, strands 0.730; advice: drive distal ends,
  fibres tension-only.
- Sweep: v09 + distal ends driven along the reconstruction (duct_end=recon): duct-group IoU 0.35 -> 0.39; without
  the fibres' bending springs 0.37; softer cables 0.37. Cables still ~ static vs their 4D (motion explained -0.1
  to +0.08); fibres curl near the anchor (anchor 4-12 mm nearer the scope than the video depth, so the rest
  centreline dips in depth). Adopted v09 + duct_end=recon. Ducts are the weakest part; parked.
- Visual (sheet_sim.jpg): the sheet is now a tent from the jaws lying over the gallbladder in all keyframes.

## r08 textured rendering, videos and two new metric families
- New metric 1, optical-flow consistency (independent of any 3D reconstruction): projected motion of the simulated
  surface k -> k+5 vs the video's DIS flow on the object's visible pixels (median end-point error). Gallbladder:
  sim 11.8 px, static 13.2, 4D recon 12.9 (frames >= 100: sim 8.9, static 8.4, recon 9.6) -> the recon fits outlines
  and depth, not texture motion; specular highlights make the flow noisy.
- New metric 2, photometric: textured MuJoCo render through the per-frame scope camera (tissue textures from the
  frame closest to each rest shape, backdrop's fused texture, pale-pink ducts) vs the video on tissue pixels.
  sim L1 34.1 / NCC 0.36; replay of the 4D reconstructions L1 31.1 / NCC 0.42 (upper bound of this renderer);
  tissues at rest L1 32.8 / NCC 0.38. Photometrically the sim is slightly worse than static even though its outline
  IoU is clearly better (0.62 vs 0.55): small position errors smear the projected texture.
- Deliverables per round now: scope.mp4 (render), compare.mp4 (video | render), render_sheet.jpg.
- Visual (render_sheet.jpg): realistic overall (liver backdrop, gallbladder with dark neck, translucent tented sheet,
  instruments); backdrop does not cover the frame-0 view's edge (pan), dark gap at the gallbladder's lower-left edge.

## r09 sheet stiffness sweep (edge springs 5/20/60 N/m x bending 60/600/3000 Pa), no rendering
- Softer sheet (5): sheet IoU 0.69 (best), gallbladder free motion explained 0.48 (best), but edges stretch up to
  5.1x (not physical); stiff (60): stretch 2.4x, sheet IoU 0.62. Bending irrelevant. Kept 20 N/m (IoU 0.65, 3.6x).
- Region analysis of the gallbladder error (r08): bed 0.06 mm (driven), free 2.0 mm (of 3.8 mm recon motion),
  neck/duct zone 3.7 (of 4.6), zone under the sheet 4.7 (of 4.8: none reproduced), near the probe 4.2 (of 5.0).

## r10 sheet base both driven and softly linked to the gallbladder (so the lift reaches the wall)
- Link time constants 0.02/0.05/0.2 s, sheet 10 or 20 N/m: tracking -32..-36 % (slightly better), free motion
  0.37-0.44 (no better), zone-under-sheet error unchanged or worse. Finding: that zone sinks ~3 mm in the sim
  while the recon rises 0.6 mm (time course correlates 0.78), also without sheet contact (-2.7 mm) -> the probe,
  5.8 mm away, presses the wall down. Not adopted.

## r11 probe depth (least certain input, +-5 mm): shift its path toward the camera by 0/2/4/6 mm
- 2 mm: probe push 5.1 vs recon 5.0 mm (was 5.7), rigid 1.24, non-rigid 1.65, free 0.45, zone-under-sheet error
  4.71 -> 4.27 mm, tracking -25.5 % (was -29.4). 4-6 mm: 3D errors a little lower again but push undershoots
  (4.4-4.6) and tracking drops to -19 %. Adopted 2 mm. The remaining zone error sits where the gallbladder track says
  its shape is unconstrained (under the tent).

## r12 counterfactual actions (the use case: data generation) + mesh-health metric
- All stable at ts 0.0625 ms: replay; grasper lifted +10 mm (sheet IoU 0.66 -> 0.69); probe +5 mm deeper (push
  5.1 -> 5.8 mm, gallbladder IoU 0.62 -> 0.59); probe withdrawn (push 3.9 mm: the rest is the driven bed); grasp
  released at frame 125 (sheet IoU 0.66 -> 0.60, the sheet falls). Responses are in the expected directions.
- New metric: inverted tetrahedra per frame. The gallbladder sim inverts up to 470 of 5302 tets (9 %) in the replay
  (415 without probe, 768 with the deeper probe); its 4D recon has none -> physically invalid elements; next round.
- Interaction v06 (refit with refined cameras): tips back to 0.3-2.8 px per band; independent depth bounds (shaft
  must stay in front of the tissue beside it; tip under the sheet) support shifting the probe 2-5 mm toward the
  camera -> the r11 choice (2 mm) is the conservative end.
- Gallbladder v11 (refit with refined cameras; rest mesh, tets and attachments byte-identical to v10): IoU per band
  0-61 / 62-129 / 130-250 = 0.750 / 0.794 / 0.801 (v10 with the same cameras 0.744 / 0.787 / 0.797); kept behind
  the sheet; neck no longer hooks over the duct at 181-217. Correction to my brief: v10's 4D was already ~0.75 in
  the pan (0.2-0.4 was its static rest shape); the pan is limited by the depth maps, not the cameras.
- Ducts v10 (refined cameras): start nodes at the observed junction depth (glue vertex v244 with a 5.7 mm rest
  offset; cable_spec returns it), depth dip 2.6 -> 0.8 mm (duct), fibres back to 13.3 mm; IoU per band duct
  0.83/0.81/0.85, strands 0.64/0.67/0.77. End-motion test on the 4D: interpolating both end motions explains
  0.64 (duct) / 0.78 (fibres) of the motion; with one more driven node 0.87 / 0.90 -> the sims' ~0 came from node 0
  following the simulated neck. Suggestion: drive node 0 along the 4D (or glue softly) and drive duct node 6 /
  fibre node 4.
- Membrane v07 (refined cams + gallbladder v11): retriangulated sheet, smallest rest angle 19.6 deg (v06 0.48),
  237 verts / 395 faces (new topology: use attach_idx/attach_to, 5 jaw -> grasper, 52 base -> gallbladder), less
  temporal smoothing of the base during the pan (frames 0-62). IoU/BF4 0-61: 0.723/0.367 -> 0.756/0.420; 62-129
  0.747/0.479 -> 0.743/0.467; 130-250 0.823/0.509 -> 0.821/0.515; all 0.778/0.466 -> 0.784/0.478. No non-base
  vertex behind v11; base-to-v11 surface 0.39 mm median. Recon stretch max 4.5 (per-frame resampling, not tissue):
  keep edge springs soft.

## r13 gallbladder material / contact sweep against inverted tets (gb_poisson x gb_young x solref, no render)
| poisson / young / solref | gb IoU | rigid / nonrigid mm | free motion expl. | inverted tets (max of 5302) | flow EPE sim / static px | run s |
|---|---|---|---|---|---|---|
| 0.45 / 1200 / 0.004 (r12) | 0.621 | 1.24 / 1.65 | 0.450 | 470 | 11.9 / 13.0 | 411 |
| 0.45 / 1200 / 0.01 | 0.630 | 1.12 / 1.59 | 0.474 | 454 | 11.6 / 13.0 | 562 |
| 0.45 / 2400 / 0.004 | 0.617 | 1.26 / 1.66 | 0.450 | 502 | 11.7 / 12.9 | 517 |
| 0.49 / 1200 / 0.004 | 0.628 | 1.30 / 1.61 | 0.454 | 476 | 11.9 / 12.9 | 1640 |
| 0.49 / 1200 / 0.01 | 0.634 | 1.14 / 1.58 | 0.484 | 378 | 11.4 / 13.0 | 2280 |
| 0.49 / 2400 / 0.004 | 0.590 | 1.84 / 2.02 | 0.302 | 528 | 12.8 / 12.9 | - |
- Material barely moves the inversions (378-528); softer contact (solref 0.01) helps a little everywhere and is
  adopted; poisson 0.49 runs 4x slower (time-step halving) for -76 inversions -> kept 0.45.
- Where they are (traj of 0.49/1200/0.01): the 4D recon has 0 inverted tets in every frame. In the sim, frame 0:
  58, all within ~3 mm of the driven liver-bed vertices (shear between driven bed and free neighbours while the
  sim settles from rest); frame 120: 72 % within 3 mm of the sheet; frames 60/180/250: 30-46 % within 10 mm of the
  probe tip. The inverted ones are the small tets (rest volume 0.15-0.4x median) -> mesh problem, not material:
  asked the gallbladder track for v12 (uniform surface layer) and for the neck to stop covering the duct.

## r14 new tissue versions (membrane v07, ducts v10, refined cameras) + duct node driving (duct_nodes)
New: `duct_nodes` = which cable nodes follow the tube's own 4D (springs k_drive): end (distal only, r13 and before),
ends (+ proximal, not glued to the simulated neck), mid (+ duct node 6 / fibre node 4), mid2 (+ duct 3 / fibre 8),
glue_mid (proximal glued to the gallbladder, distal + middle driven). New metric: free_motion_explained over the
cable nodes the simulation decides itself (the driven ones excluded). `cams=refined` (backdrop v08 rotations)
is the default now.
| config | duct IoU | free-node motion expl. duct / strands 1-3 | gb IoU | inverted | sheet IoU |
|---|---|---|---|---|---|
| end | 0.442 | -0.43 / -0.28 -0.23 -0.22 | 0.627 | 542 | 0.647 |
| ends | 0.451 | 0.10 / 0.34 0.53 0.20 | 0.628 | 503 | 0.652 |
| glue_mid | 0.628 | 0.18 / 0.40 0.39 0.36 | 0.631 | 494 | 0.646 |
| mid | 0.663 | 0.67 / 0.76 0.77 0.76 | 0.624 | 512 | 0.652 |
| mid2 | 0.719 | 0.74 / 0.87 0.88 0.85 | 0.622 | 603 | 0.648 |
| mid + solref 0.01 | 0.642 | 0.62 / 0.76 0.77 0.76 | 0.627 | 494 | 0.645 |
| mid + start=recon | 0.660 | 0.63 / 0.79 0.80 0.72 | 0.626 | 471 | 0.643 |
| mid + cams=clip | 0.663 | same | 0.624 | 512 | 0.652 (flow EPE gb 12.6 vs 11.7, sheet 13.0 vs 12.2) |
- Ducts: IoU 0.44 -> 0.66 and the undriven nodes now follow the video (motion explained -0.4..-0.2 -> 0.67-0.77)
  with 3 of 13 nodes per cable on the 4D; 'mid2' (5 of 13) +0.06 IoU more, but then nearly half the cable is
  prescribed -> adopted 'mid' (cable physics decides 10 of 13 nodes; the driven interior node stands for the
  connective tissue the cable lies on, which is not modelled).
- The refined cameras do not change keyframe metrics (keyframes are the same) but lower the optical-flow error of
  everything by 0.8-1.2 px (the in-between frames are registered) -> confirms the refined cameras.
- Membrane v07 vs v06 in the sim: no change (0.645-0.652); the sim loses 0.13 IoU against its own recon (0.784).
  Diagnosis (traj of 'mid'): the held jaw vertices are 3.3 mm off the recon in every band because the grasp pulls
  the held patch onto the grasper's tool-centre point, while in the 4D the sheet apex sits 3.2 mm from it; the
  error is largest near the apex (t 0-0.4: 3.3-3.9 mm vs 2.2-2.4 mm at the base). -> r15.

## r15 grasp target (held patch onto the TCP vs the 4D's own apex) x sheet stiffness (duct_nodes=mid, solref 0.01)
Caveat found afterwards: scene4d took each tissue's newest folder, and the gallbladder track had just written v12
(work in progress) -> all r15 configs ran gallbladder v12, so r15 vs r14 mixes two changes; inside r15 the
comparison is fair. Fix: scene4d now uses pinned ADOPTED versions (gallbladder v11, membrane v07, ducts v10,
backdrop v08, interaction v06) unless a run names one ('latest' = newest folder).
| grasp target, mem_k, mem_young | sheet IoU | sheet err vs recon mm | sheet rigid / nonrigid | sheet stretch max | gb IoU | gb rigid | gb free mexp | inverted |
|---|---|---|---|---|---|---|---|---|
| recon, 20, 600 | 0.635 | 3.15 | 2.64 / 2.52 | 3.6 | 0.645 | 0.87 | 0.71 | 179 |
| recon, 10, 600 | 0.653 | 3.07 | 2.61 / 2.50 | 4.9 | 0.657 | 0.84 | 0.72 | 150 |
| recon, 5, 600 | 0.654 | 3.12 | 2.72 / 2.56 | 6.3 | 0.655 | 0.82 | 0.73 | 164 |
| recon, 5, 60 | 0.664 | 3.02 | 2.54 / 2.49 | 6.3 | 0.655 | 0.82 | 0.73 | 119 |
| tcp, 5, 600 | **0.680** | **2.93** | 2.32 / 2.55 | 6.3 | 0.655 | 0.81 | 0.73 | 130 |
- The r14 diagnosis was half right: `grasp_target=recon` does put the held vertices on the 4D (jaw error 2.7-3.4 ->
  0.3-2.4 mm in a short test) but the rest of the sheet then fits worse, against the video (IoU -0.026) and even
  against the 4D (err +0.19 mm): the TCP (interaction v06, 0.3-2.8 px) is the better evidence for where the jaws
  hold the sheet; the 4D's apex is the sheet track's least certain part. Negative result: kept `tcp`.
- Softer sheet edges (mem_k 5) again help the silhouette (r09 showed the same) -> adopted mem_k 5 (sheet
  stretch 6.3 is above the recon's 4.5, which itself is resampling noise).
- Gallbladder v12 (picked up by accident): gb rigid error 1.19 -> 0.81 mm, free-vertex motion explained 0.47 ->
  0.73, inverted tets 494 -> 119-179, IoU 0.627 -> 0.655 -> checked under equal settings in r16.
- New defaults: solref 0.01, duct_nodes mid, mem_k 5, grasp_target tcp.
- New instrumentation (from r16): net grasp force on the jaws (equality forces) and probe normal contact force per
  control step; per-band (pan 0-61 / lift 62-129 / probe 130-250) IoU for sim and recon.
- Gallbladder v12 / v13 (hand-back after r15): the ducts v10 `duct | strands` pixels now count as background (not
  unknown) and the neck tip is pinned in 3D to the duct's 4D proximal end. Duct mask covered by the gallbladder
  21.8/22.7/16.3 % -> 0.9/1.8/0.2 % per band; neck tip - junction depth -12.6/-9.4/-5.1 -> -2.1/-2.8/-1.9 mm.
  New tet mesh: uniform lattice instead of TetGen, 1383 nodes / 6186 tets, 0 inverted in any 4D frame; rest
  quality (mean ratio min/p1/p5) 0.03/0.15/0.27 -> 0.10/0.55/0.68, volume/median p1 0.004 -> 0.31, min dihedral p1
  8.9 -> 30.5 deg. Track's own MuJoCo check (capsule r 2.5 mm pushed 5 mm along the probe shaft): inverted tets
  v11 7/88/62 -> v13 0/6/0 at frames 60/180/250. v13 = v12's mesh with the 4D refit at half the ARAP weight (v12
  lost ~0.06 boundary F). Own-mask IoU/BF per band v11 0.816/0.319, 0.826/0.363, 0.859/0.378 -> v13
  0.855/0.367, 0.848/0.376, 0.888/0.363; depth residual 1.43/1.06/1.11 -> 0.87/0.70/0.85 mm. Costs: rigid motion
  up to 13.5 deg / 5.6 mm (v11 11.7 / 3.9), smaller rest volume 28.6 ml (32.2), gallbladder points 5-8 mm in front
  of the sheet in pan frames 45-81 -> asked the sheet track for v08 re-snapped to v13 (and to check its apex,
  r15).

## r16 gallbladder v11 vs v12 under equal settings, time-step convergence, instrument forces
Defaults now: solref 0.01, duct_nodes mid, mem_k 5, grasp_target tcp, refined cameras.
| config | gb IoU (bands pan / lift / probe) | gb rigid / nonrigid mm | gb free mexp | inverted | probe push sim / recon mm | sheet IoU (bands) | run s |
|---|---|---|---|---|---|---|---|
| v11 | 0.620 (0.52 / 0.57 / 0.70) | 1.23 / 1.68 | 0.46 | 490 | 5.3 / 5.1 | 0.653 (0.52 / 0.61 / 0.74) | 305 |
| v12 | **0.655** (0.56 / 0.59 / 0.73) | **0.81 / 0.91** | **0.73** | 130 | 5.9 / 6.5 | **0.680** (0.58 / 0.64 / 0.75) | 382 |
| v12, mem_young 60 | 0.654 | 0.80 / 0.92 | 0.73 | 156 | 6.0 / 6.5 | 0.659 | 460 |
| v12, ts 0.03125 ms | 0.655 (0.56 / 0.59 / 0.73) | 0.83 / 0.92 | 0.73 | 124 | 5.9 / 6.5 | 0.670 | 688 |
- v12's mesh/neck fix is the largest single gain of the gallbladder in the sim so far (rigid error -34 %,
  nonrigid -46 %, inversions -73 %, sheet +0.03 because the sheet no longer has to sit on a neck that covers the
  duct).
- Time-step convergence: halving the step changes every metric by <= 0.01 IoU / 0.03 mm -> results are not
  integration artefacts at ts 0.0625 ms.
- Forces (new): net grasp force on the jaws median 15 mN (max 39), probe normal force median 20-22 mN (max 68-111
  mN). These are low against reported retraction / palpation forces (order 1 N). With gravity compensated
  (the rest shapes are already the loaded shapes), the video only fixes stiffness relative to stiffness:
  scaling every modulus, spring and vertex mass by s leaves the motion unchanged and multiplies every force by s
  -> the absolute force scale is not identifiable from this video; the forces are reported for E = 1.2 kPa.

## r17 gallbladder v13 (v12's mesh, 4D refit with half the ARAP weight) in the sim
| config | gb IoU (bands) | gb rigid / nonrigid | free mexp | inverted | push sim / recon | sheet IoU |
|---|---|---|---|---|---|---|
| v13 | 0.637 (0.52 / 0.57 / 0.73) | 0.93 / 1.18 | 0.62 | 163 | 4.8 / 5.9 | 0.667 |
| v13, gb_young 600 | 0.630 | 0.98 / 1.23 | 0.61 | 220 | 4.7 / 5.9 | 0.668 |
| v13, gb_young 2400 | 0.630 | 0.92 / 1.16 | 0.63 | 107 | 4.8 / 5.9 | 0.661 |
| v13, mem_young 60 | 0.636 | 0.94 / 1.19 | 0.62 | 172 | 4.8 / 5.9 | 0.676 |
| v13, no sheet-gallbladder contact | 0.613 | 0.96 / 1.23 | 0.61 | 285 | 5.1 / 5.9 | 0.641 |
- v13's own 4D fits the video better than v12's (recon IoU per band 0.79 / 0.77 / 0.84 vs 0.77 / 0.75 / 0.82),
  but the simulation of it fits worse (0.637 vs 0.655): the looser 4D has more non-rigid motion and larger rigid
  motion (13.5 deg) that a 1.2 kPa body driven at its bed cannot follow. -> v12 adopted for the sim
  (ADOPTED gallbladder v12); v13 remains the better reconstruction.
- E 1200 Pa stays best (600 and 2400 both -0.007 IoU). Sheet-gallbladder contact on is confirmed: without it
  the gallbladder IoU drops 0.024, the sheet's 0.026 and inversions double (285).
- Pan-band diagnosis (v12): the sim's IoU there (0.45-0.73 per frame) is held down by the visibility cut, not the
  outline: without the depth cut the sim reaches 0.65-0.74 vs recon 0.71-0.81. The sim's surface renders 0.5-3 mm
  behind the video depth (recon -1.7..+0.8 mm). Physical reason: the sim keeps its volume (27.1-28.1 ml, rest
  28.6) while the 4D recon "breathes" 27.3-29.7 ml (+-4 %, not physical for a fluid-filled organ) -> the physics
  removes reconstruction noise, and the visibility cut counts that as error. New metrics: sim_iou_2d /
  recon_iou_2d (no depth cut), volume_ml (rest, sim min-max, recon min-max).
- Membrane v08 (hand-back after r17): v07 re-snapped to gallbladder v13 (v12's rest mesh), same 237 verts and
  attachments, 60 faces flipped for a clean rest mesh (smallest angle 18.2 deg). Apex: moving the whole sheet
  origin onto the TCP costs IoU 0.065 (0-61) / 0.04 (62-129), so only the held row sits on the TCP (median 0 mm,
  max 0.78; v07 3.15 / 4.3) with a short strip under the jaws to where the sheet visibly leaves them. IoU/BF per
  band unchanged (all 0.784/0.478 -> 0.785/0.473). No non-base vertex behind v13 in any frame (v07 vs v13: up to
  11 % in some frames); base to v13 surface 0.40 mm median; stretch max 4.5 -> 3.1.

## r18 adopted scene, textured render: simulation vs kinematic replay of the 4D (mode=recon)
gallbladder v12, membrane v07, ducts v10 (cables coloured with the video's median duct / strand colour).
| | gb IoU (2D, no depth cut) | sheet IoU (2D) | duct IoU | photometric NCC / L1 | gb volume ml |
|---|---|---|---|---|---|
| replay of the 4D | 0.788 (0.771) | 0.787 (0.796) | 0.761 | 0.400 / 32.2 | 27.3-29.6 |
| simulation | 0.655 (0.702) | 0.680 (0.742) | 0.653 | 0.357 / 34.9 | 27.1-28.1 |
- The simulation keeps 83-93 % of the replay's silhouette agreement (2D: 91 % gallbladder, 93 % sheet) while
  being driven only at the liver bed, the sheet base, 3 nodes per cable and the instruments.
- Visual check (render_sheet.jpg): grasper holds the sheet's apex and lifts it into a tent; the probe enters
  under the sheet into the gallbladder neck; ducts now in their video colour. Remaining visible defects: the duct
  tube is thicker and more vertical than the video's fibre fan; black corners where the backdrop does not reach.

## r19 membrane v08 (held row on the TCP, consistent with gallbladder v12/v13's rest mesh)
| config | sheet IoU (bands) | sheet err vs recon mm | sheet rigid / nonrigid | gb IoU | gb inverted | grasp force mN |
|---|---|---|---|---|---|---|
| v07, mem_k 5 (r16) | 0.680 (0.58 / 0.64 / 0.75) | 2.93 | 2.32 / 2.55 | 0.655 | 130 | 15 |
| v08, mem_k 5 | 0.684 (0.57 / 0.67 / 0.75) | 2.71 | 2.21 / 2.36 | 0.655 | 149 | 6 |
| v08, mem_k 10 | **0.696** (0.62 / 0.68 / 0.74) | 2.70 | 2.12 / 2.36 | 0.651 | 125 | 12 |
| v08, mem_young 60 | 0.689 | 2.66 | 2.14 / 2.33 | 0.652 | 98 | 7 |
| v08 + gallbladder v13 | 0.666 | 2.82 | 2.31 / 2.46 | 0.631 | 90 | 7 |
- With the held row on the TCP the sheet tolerates stiffer edges again (mem_k 10 best; with v07, 5 was best
  because the edges had to absorb the 3 mm apex offset) -> adopted membrane v08, mem_k 10.
- v12 beats v13 in the sim again with the v13-consistent sheet (0.651-0.655 vs 0.631) -> v12 stays.
- Ducts (new baseline: the same tubes around the duct track's own 4D centrelines): sim 0.661 vs recon 0.761;
  per band 0.63 / 0.61 / 0.71 vs 0.73 / 0.70 / 0.81.

## r20 cable stiffness / damping (ducts as 1D cables, 3 driven nodes each)
| config | duct IoU (bands) | free-node motion expl. duct / fibres | duct free err mm | duct length ratio max | sheet IoU |
|---|---|---|---|---|---|
| cable_k 0.5 | 0.639 (0.62 / 0.63 / 0.65) | 0.56 / 0.73-0.77 | 0.92 | 1.25 | 0.693 |
| cable_k 2 (r19) | 0.648 (0.63 / 0.61 / 0.68) | 0.64 / 0.76-0.77 | 0.76 | 1.14 | 0.696 |
| cable_k 2, no fibre bending springs | 0.616 | 0.64 / 0.52-0.69 | 0.76 | 1.14 | 0.696 |
| cable_k 8 | 0.658 (0.63 / 0.63 / 0.68) | 0.69 / 0.74-0.81 | 0.66 | 1.05 | 0.694 |
| cable_k 8, damping 0.05 | **0.669** (0.66 / 0.63 / 0.69) | 0.68 / 0.78-0.81 | 0.67 | 1.05 | 0.682 |
- Stiffer cables follow the video better and stop over-stretching (25 % -> 5 %, the 4D's own range is 0.94-1.09);
  the skip-one springs (bending) matter for the fibres (-0.03 IoU without). Adopted cable_k 8, damping 0.05.
- Noise floor: the sheet and gallbladder change by up to 0.014 IoU between these runs although nothing of theirs
  changed (the probe touches the ducts and is slightly deflected) -> differences below ~0.01 IoU are not
  meaningful in single runs; r21 measures this directly with perturbed inputs.
- Gallbladder v14 (hand-back): v12's mesh, 4D refit with a total-volume term (+-0.5 %): volume 28.42-28.72 ml
  (v12 27.12-29.74), rotation median/max 8.3/12.9 deg (v12 10.4/16.9), max vertex speed 2.7 mm/frame (3.8); IoU per
  band 0.785/0.800/0.797 (v12 0.786/0.805/0.794) - v13's outline gain came partly from the volume following noisy
  per-frame depth. To be tried in the sim (r21).

## hold-out check of the choices (r2s/holdout4d.py; select on frames 0-129, test on 130-250; outputs/iter/rounds/holdout.json)
- Structural decisions hold: duct_nodes (r14: the choice on 0-129 is also best on 130-250 for all three tissues),
  grasp target tcp + mem_k for the sheet (r15), gallbladder v12 for the sheet (r16).
- Parameter fine-tuning does not: in r13 (material), r17 (young), r19 (sheet stiffness) the config chosen on the
  first part is not the best on the probe phase, and it loses only 0.002-0.009 IoU (ducts up to 0.03) on the
  test part -> these differences are at the run-to-run noise floor (r20). Parameter tuning has saturated; the
  remaining gap to the 4D replay (gallbladder -0.13, sheet -0.09, ducts -0.09 IoU) needs model changes, not
  parameters.

## r21 gallbladder v14 (volume-preserving 4D) + robustness to instrument-track errors (noise_mm, smooth, RMS per axis)
| config | gb IoU (bands) | gb rigid / nonrigid | gb free mexp | inverted | sheet IoU | sheet rigid | duct IoU |
|---|---|---|---|---|---|---|---|
| v12 (adopted, cable_k 8) | 0.650 (0.56 / 0.59 / 0.73) | 0.80 / 0.90 | 0.73 | 97 | 0.682 | 2.13 | 0.669 |
| v14 | 0.633 (0.52 / 0.56 / 0.73) | 0.88 / 1.04 | 0.65 | 102 | 0.667 | 2.25 | 0.677 |
| noise 1 mm, seeds 0/1/2 | 0.649-0.651 | 0.77-0.84 / 0.91-0.92 | 0.73-0.74 | 120-146 | 0.661-0.678 | 2.15-2.38 | 0.669-0.681 |
| noise 2 mm, seeds 0/1 | 0.643-0.647 | 0.82-0.83 / 0.96 | 0.73 | 154-203 | 0.673-0.674 | 2.46-2.47 | 0.671-0.681 |
- v14 (volume held within +-0.5 % in the 4D) simulates worse than v12 (-0.017 IoU, rigid +0.08 mm): a
  constant-volume 4D is not easier for the physics; its bed motion (which drives the sim) is what differs. v12 stays.
- Robustness: 1 mm track error changes the gallbladder by <= 0.001 IoU and 2 mm by <= 0.008; the sheet is the
  sensitive tissue (up to -0.02 IoU at 1 mm, rigid error +0.3 mm at 2 mm) because it hangs from the grasper;
  inversions grow with noise (97 -> up to 203). The instrument tracks (0.3-2.8 px, ~0.1-1 mm) are accurate
  enough that their error is below the differences we act on.

## r22 adopted scene, textured render + counterfactual actions (data generation)
Adopted: gallbladder v12, membrane v08, ducts v10, backdrop v08, interaction v06; solref 0.01, mem_k 10, duct_nodes
mid, cable_k 8 / damping 0.05, grasp_target tcp, refined cameras.
| run | gb IoU (2D) | gb rigid / nonrigid | inverted | sheet IoU (2D) | duct IoU | photometric NCC / L1 | gb volume ml | grasp / probe force median mN |
|---|---|---|---|---|---|---|---|---|
| replay of the 4D | 0.788 (0.771) | 0 | 0 | 0.788 (0.794) | 0.761 | 0.399 / 32.2 | 27.3-29.6 | - |
| simulation, video's actions | 0.650 (0.697) | 0.80 / 0.90 | 97 | 0.682 (0.747) | 0.669 | 0.358 / 34.5 | 27.2-28.3 | 12 / 19 |
| grasper +10 mm | 0.650 | 0.78 / 0.90 | 97 | 0.695 | 0.682 | 0.367 / 34.3 | 27.2-28.3 | 67 / 19 |
| probe 5 mm deeper | 0.622 | 0.87 / 1.02 | 288 | 0.674 | 0.677 | 0.360 / 34.9 | 25.5-28.0 | 12 / 27 |
| probe withdrawn | 0.652 | 0.73 / 0.85 | 55 | 0.680 | 0.681 | 0.375 / 33.5 | 27.7-28.3 | 12 / 0 |
- Probe effect measured properly (new check: sim minus the probe-withdrawn counterfactual, vertices within 8 mm
  of the tip, along the shaft): indentation max 6.2 mm, median 2.3 mm over 185 contact frames (pan 1.0 / lift 2.1
  / probe phase 2.2 mm mean); 5 mm deeper -> max 7.8, median 3.6 mm and 107 vs 38 vertices moved > 1 mm, volume
  down to 25.5 ml. The old `probe_push` metric (displacement near the tip) mostly measures the driven bed motion
  (5.8 mm even without the probe) -> superseded by this difference.
- Lifting the grasper 10 mm raises the grasp force 12 -> 67 mN and the sheet follows (IoU vs the original video
  +0.013, i.e. the 4D sheet sits a little higher than the sim's). Withdrawing the probe halves the inversions
  (97 -> 55): most remaining inversions are probe-contact elements.
- Photometric NCC is lowest for the sim with the probe (0.358) and highest without it (0.375): the probe-side
  texture smears where the wall is indented; replay 0.399 is this renderer's ceiling.
- Report updated (artifact VT6cePMUES9nUifUZ8cv2y v11): per-tissue models, 4D sim video, round ledger, progress
  chart, counterfactual grid, checks; media built by `site/report/build.py 4d` from r22.

## r23 probe contact softness and gallbladder contact radius vs the remaining inversions
Indentation = sim minus the r22 probe-withdrawn run, vertices within 8 mm of the tip, along the shaft.
| config | gb IoU | gb rigid / nonrigid | inverted | probe indentation max / median mm | sheet IoU |
|---|---|---|---|---|---|
| r22 (probe solref 0.002, flex radius 0.4 mm) | 0.650 | 0.80 / 0.90 | 97 | 6.2 / 2.5 | 0.682 |
| probe solref 0.01 | 0.650 | 0.79 / 0.90 | 105 | 6.2 / 2.5 | 0.683 |
| probe solref 0.02 | 0.650 | 0.79 / 0.90 | 99 | 6.2 / 2.4 | 0.678 |
| gallbladder flex radius 1 mm | 0.629 | 0.90 / 1.03 | 204 | 6.6 / 3.0 | 0.694 |
| both | 0.629 | 0.90 / 1.02 | 198 | 6.6 / 3.0 | 0.691 |
- Negative: softer probe contact changes nothing (the probe is position-controlled, the indentation is set by its
  path); a thicker collision skin pushes the sheet off the wall (sheet +0.01) but doubles the inversions and costs
  the gallbladder 0.02. Kept r22. The ~100 inversions (1.6 %) are where a rigid 2.5 mm capsule indents a 1.2 kPa
  lattice by 6 mm; fixing them needs a nonlinear (inversion-resistant) material, which MuJoCo flex does not offer.

## r24 (analysis) can the liver-bed motion be predicted from the actions? (counterfactual readiness)
The bed is driven by the 4D, so counterfactual runs keep the video's bed motion. Bed displacement field (214 bed
vertices, 4-7 mm range, 63 % of its variance in one mode) regressed linearly on instrument / camera signals,
fitted on frames 0-129 and tested on 130-250 (and reversed):
| predictor | test error mm (forward / reversed split) | zero-motion baseline |
|---|---|---|
| grasper TCP | 3.89 / 4.89 | 2.16 / 4.68 |
| grasper + probe TCP | 3.89 / 7.17 | 2.16 / 4.68 |
| camera position | 3.70 / 7.54 | 2.16 / 4.68 |
| linear drift in time | 7.44 / - | 2.16 / - |
- Negative: no predictor beats assuming no bed motion on held-out frames (in-sample correlations up to 0.8 with
  the grasper are not causal enough to extrapolate). The bed motion stays a measured boundary condition; the
  report says so (counterfactuals keep the video's bed motion).

## r25 anatomy check (connectivity) + gluing the duct start to the gallbladder's neck tip
New metric `anatomy` (r2s/quality.signed_distance, sample-based): gap of the duct start to the gallbladder surface,
duct nodes > 1 mm inside the gallbladder, sheet base gap, free sheet vertices inside, held row to grasper TCP
(median / max over keyframes). Run on r22 first: the duct start floats 4.96 / 10.75 mm off the simulated
gallbladder (the 4D itself: 3.75 / 9.37 mm between ducts v10 and gallbladder v12) -> the duct visibly detaches
from the neck, the kind of torn anatomy the user rejected before.
| duct_nodes | duct IoU | free-node mexp duct / fibres | duct start gap mm (med / max) | duct nodes inside gb % (med / max) | sheet base gap | gb IoU / inverted |
|---|---|---|---|---|---|---|
| mid (r22) | 0.669 | 0.68 / 0.78-0.81 | 4.96 / 10.75 | 0 / 48 | 2.04 / 4.55 | 0.650 / 97 |
| glue_mid (start glued to the neck-tip vertices) | 0.630 | 0.28 / 0.58-0.61 | 3.72 / 5.46 | 25 / 92 | 1.96 / 4.43 | 0.651 / 94 |
| glue_mid2 | 0.701 | 0.53 / 0.66-0.69 | 3.55 / 5.53 | 44 / 88 | 1.93 / 4.52 | 0.654 / 109 |
| glue_mid2 + sheet base also linked | 0.701 | 0.54 / 0.69-0.70 | 3.52 / 5.43 | 29 / 96 | 1.64 / 3.12 | **0.440 / 1536** |
- Gluing does not close the gap: MuJoCo's connect constraint keeps the two bodies' relative offset from the
  reference configuration (the rest offset, ~4-6 mm), it does not pull them together; it only caps the worst
  frames (10.7 -> 5.5 mm) and makes the duct pass through the neck (nodes inside 25-44 %). Linking the sheet base
  to v12 (v08 sits ~2 mm off v12's 4D) wrecks the gallbladder (1536 inversions). Kept 'mid'.
- Real fix belongs in the reconstruction: asked the duct track for v11 with the proximal ends on gallbladder v12's
  duct-attachment vertices in every frame (blended so the v09 curl does not return).

## r26 integrator-side stop-gap: snap the duct start onto gallbladder v12's neck tip (duct_snap)
`snap_cable`: node 0 of every cable moved onto the nearest v12 duct-attachment vertex, in the rest shape and in
every 4D frame, the shift blended out over the first 3 segments (rest of the centreline untouched).
| config | duct IoU | free-node mexp duct / fibres | duct start gap mm (med / max) | duct nodes inside gb % (med / max) | gb IoU |
|---|---|---|---|---|---|
| mid, no snap (r22) | 0.669 | 0.68 / 0.78-0.81 | 4.96 / 10.75 | 0 / 48 | 0.650 |
| snap + mid (start driven along v12's 4D vertex) | 0.622 | 0.46 / 0.64-0.66 | 3.42 / 7.14 | 0 / 33 | 0.650 |
| snap + mid2 | 0.661 | 0.66 / 0.69-0.72 | 3.43 / 7.08 | 0 / 31 | 0.650 |
| snap + glue_mid2 (start glued to the simulated neck) | **0.687** | 0.52 / 0.59-0.60 | **0.44 / 0.76** | 20 / 71 | 0.652 |
- Driving the start along the 4D vertex is not enough (the simulated neck is 3-7 mm off its own 4D there: the
  neck/duct zone is the gallbladder's worst region, r09). Gluing the snapped start to the simulated neck closes
  the gap (0.44 mm) and even raises duct IoU (+0.018); price: the first 2-3 nodes of each cable run inside the
  neck surface (the duct continuing into the neck; hidden in the scope view) and lower free-node motion explained.
- Adopted snap + glue_mid2 for connected anatomy until ducts v11 (proper fix in the reconstruction) arrives.

## r27 render of the connected-anatomy scene (snap + glue_mid2) and r28 counterfactual trajectories under it
- r27 sim: gb IoU 0.652, inverted 92, sheet 0.681, ducts 0.687, photometric NCC 0.365 (r22: 0.358); anatomy: duct
  start gap 0.44 / 0.76 mm, held row 0.08 mm. Probe withdrawn: inverted 41, NCC 0.377.
- r28: replay of the 4D, grasper +10 mm, probe +5 mm with the same settings (trajectories for the 3D viewer).
- 3D viewer updated (artifact B4V3kZTgvRuJdTsaSPLHD7 v8): new default scene chole_a_4d (r2s/export4d.py):
  gallbladder = organ layer, sheet + 4 duct tubes = ribbon layer (atlas: sheet frame + duct colour strip),
  backdrop; conditions sim / 4D replay / lift / deeper / no probe; per-step scope position (multi-view cameras);
  capsule geoms. Checked locally (http.server): scene, instruments and metrics table load, no console errors.
- Tissue hand-backs: ducts v11 (proximal ends exactly on v12's duct-attachment surface in every frame, 0.000 mm;
  20 % bridge onto the observed path; duct IoU per band 0.76/0.80/0.82, strands 0.60/0.60/0.74 (v10 0.835/0.711);
  depth bow 4D max 3.2/2.8 mm in the pan; lengths up to 1.43-1.48x of rest in frames 0-130 (the v12 neck to
  junction gap now shows as bridge length); end-motion test 0.74/0.85, +1 node 0.92/0.94). Membrane v09 (base on
  v12's 4D surface: 0.19 mm median / 1.44 max, v08 vs v12 0.42 / 2.79; IoU unchanged 0.783; 28 faces flipped). Note
  from the sheet track: my anatomy metric measures the sheet base against the *simulated* gallbladder (1.9-2.0 mm
  for v08), its 0.42 mm is against v12's 4D - both true; the sim gallbladder sits 1-2 mm off its 4D near the base.

## r29 ducts v11 + membrane v09 (the reconstructions' own fixes for the disconnects)
| config | duct IoU | duct free mexp duct / fibres | duct start gap (med / max) | duct nodes inside % | sheet IoU | gb IoU / inverted |
|---|---|---|---|---|---|---|
| r27: ducts v10 snapped + glue_mid2, membrane v08 | 0.687 | 0.52 / 0.59-0.60 | 0.44 / 0.76 | 20 / 71 | 0.681 | 0.652 / 92 |
| v11 + v09, glue_mid2 | 0.681 | **0.73** / 0.66-0.69 | 0.42 / 1.29 | 23 / 83 | 0.675 | 0.654 / 95 |
| v11 + v09, glue_mid | 0.642 | 0.60 / 0.59-0.66 | 0.49 / 1.01 | 44 / 98 | 0.676 | 0.651 / 106 |
| v11 + v09, mid (start driven, not glued) | 0.626 | 0.75 / 0.72-0.80 | 4.25 / 7.24 | 27 / 63 | 0.667 | 0.638 / 149 |
| membrane v09 only (ducts v10 snapped) | 0.678 | 0.53 / 0.57-0.61 | 0.46 / 0.83 | 32 / 79 | 0.674 | 0.653 / 121 |
- Adopted ducts v11 + membrane v09 + glue_mid2 without the integrator-side snap (the fix now lives in the
  reconstructions; equal IoU within noise, duct free-node motion explained 0.52 -> 0.73). Starting the duct on the
  4D surface is not enough while it is not glued: the simulated neck is 4-7 mm off its 4D there.
- Sheet base vs the *simulated* gallbladder stays 1.7 mm (vs v12's 4D 0.19 mm): the residual is the sim's own
  error near the base.

## user check (3D viewer, shape mode): "这坨紫色是什么" -> the sheet crumples near the jaws
- The purple wad is the peritoneal sheet. Measured: in the sim the sheet keeps its rest area (0.99-1.24x), while
  in the 4D the visible sheet's area runs 0.56-1.11x of rest (frames 40-80 ~0.56-0.72: less sheet is visible /
  lifted) -> with fixed rest lengths the surplus material has nowhere to go and folds: fold-angle p90 77-110 deg in
  frames 0-80 (4D 23-52), median over keyframes 71 vs 27.5 deg. A fixed-topology sheet cannot represent material
  entering / leaving the view at its base. -> r30: rest lengths follow the 4D (mem_rest 'area' / 'edges',
  flexedge_length0 updated with the drives; verified that MuJoCo honours runtime rest-length changes).
- New metrics: membrane fold_p90_deg (sim / recon), area_vs_recon.

## source check (user asked for the video's own description): it changes the anatomy assumptions
- The authors' legend: derotation 0-23 s, decompression 24-33 s, cholecystectomy 34-42 s; the paper: hydropic,
  necrotic gallbladder with a two-fold torsion, hanging free from the cystic duct and artery with no liver
  attachment (data/videos/commons_chole/SOURCE.md).
- Consequences for the model: (1) shot A is the decompression, so the white instrument entering the gallbladder is
  most likely the decompression instrument (not named in the paper) and the volume may fall; (2) the gallbladder
  had no liver bed: our 214 bed vertices driven along the 4D stand in for a pedicle-suspended organ, which likely
  explains the whole-body swing (r02-r03) that no bed spring reproduced; (3) the dark purple neck / duct colour is
  gangrene. To test next: gallbladder hanging from its pedicle (ducts) instead of a bed; volume loss during
  decompression.
