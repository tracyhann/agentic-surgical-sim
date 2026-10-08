# v2 round log (text-guided, SAM 3, primitive initialisation)

Plan: PLAN_V2.md. v1 (hand-prompted, per-tissue, anatomical template) ends at outputs/iter/LOG.md r31.

## r01 perception: text -> scene spec -> SAM 3 (2026-10-08)
Clips (t2s/data.py): chole_derot (3.8-23.4 s, every 2nd frame, 247 frames, 12.5 fps), chole_a (control, 251 frames
at 25 fps), liver_dx (0-60 s, every 4th frame, 374), lung_mln (31-62 s, every 2nd, 385).
- Text agent -> scene_spec.json per clip (organs with roles / prompts / consistency, connections with evidence and
  confidence, instruments, actions, openings, uncertainties), from the Commons legends and the open-access articles
  only (the human frame notes in SOURCE.md were excluded). Key content: chole: gallbladder hangs from the cystic
  duct + artery, free of the liver bed (high); chole_a: a puncture opening of the gallbladder (medium; instrument
  left open for vision); liver: leak in the left triangular ligament, closed by sutures; lung: one-lung ventilation,
  no breathing motion of the right lung.
- SAM 3 text prompts (t2s/seg3.probe_text, 5 keyframes per clip, score >= 0.5): anatomy names are not recognised
  in any clip (gallbladder, cystic duct, liver, ligament, diaphragm, lung, lymph node, vena cava...: 0/5); only
  "blood" (2-5/5), "pleura" (2/5 lung), "needle" (4/5 chole_a, i.e. the instruments) work. Instruments: "surgical
  instrument" 4/5 chole_a, 2/5 chole_derot, 1/5 liver, 0/5 lung; "metal" / "tool" / "forceps" are no more reliable.
  -> SAM 3 text alone cannot segment laparoscopic anatomy; the route is: text decides WHAT to find, a vision
  prompting agent decides WHERE (points on keyframes), SAM 3's tracker propagates. No human prompts.
- Prompting agents (one per clip, keyframes with a pixel grid): chole: gallbladder (high), an added
  "neck_pedicle" (cystic duct and artery cannot be told apart or separated from the neck: merged, not guessed),
  liver, fat, blood, a red strand band; liver_dx: found the clip is 5 scenes joined by page-curl wipes that the
  histogram cut check missed, and the liver is identifiable only in scene 4 -> new clip liver_s4 (scene 4, 36.2-50.2 s,
  every 2nd frame, 177 frames: suturing the left lobe tip); lung_mln: 9 shots (jump cuts the check missed), longest
  continuous ~6 s, named vessels / airways not identifiable -> prompted only "lymph nodes and fat", "pleura",
  "dissection bed", "white curved cord". Lung is kept as a perception-only clip (too fragmented for multi-view 3D).
- Instrument agents (one per clip) give point prompts per physical instrument and re-check shot continuity.
- Instrument agents: chole_derot: grasper_top (absent 0-12 and 57-113, passes in front of the lens 13-17) and a
  curved dissector_right; one continuous shot. chole_a: a grasper holding the gallbladder neck all clip and a
  **suction cannula** (side holes visible f143-160), not a needle: it is inside the gallbladder in f0-112 and
  f161-250 (the puncture precedes the clip; medium confidence) -> this is the spec's decompression opening.
  liver_s4: two needle holders tying a knot on a stitch placed earlier (no needle pass in the clip); one continuous
  shot; frames 175-176 start the next wipe (trim for 3D).
- Tracking (SAM 3 tracker, all objects of a clip in one session, instruments in front): seg v02.
  | clip | objects (temporal IoU median) | runtime |
  |---|---|---|
  | chole_a | grasper 0.94, suction cannula 0.85, gallbladder 0.96, neck_pedicle 0.86, red strand band 0.91, blood 0.94, liver 0.96 | 20 min |
  | chole_derot | grasper 0.55, dissector 0.55, gallbladder 0.87, neck_pedicle 0.76, liver 0.93, fat 0.91 | 19 min |
  | liver_s4 | needle holder R 0.78 / L 0.76, liver 0.93, ligament 0.86, diaphragm 0.96, omentum 0.95, gauze 0.88 | 11 min |
  | lung_mln | lymph nodes+fat 0.75, pleura 0.90, dissection bed 0.83, white cord 0.62 (no instruments yet) | 19 min |
  chole_a vs v1's hand-prompted SAM 2.1 masks: grasper IoU 0.93; gallbladder + neck_pedicle vs v1 gallbladder 0.75.
- Geometry (t2s/geom.py -> r2s prep + keyframe BA with SIFT): chole_a: 26 keyframes, reprojection 6.0 -> 2.5 px,
  fovy 35 deg (assumed 47), metric scale from 11840 shaft samples (median 5.4 mm).
- Visual QA agent (r01, images only): chole_a and chole_derot "usable after repairs", lung "not usable".
  Problems with frames and fixes: chole_a red strand band swallows an unidentified red-brown oval organ;
  neck_pedicle drops its distal tube (38-101) and proximal segment (162-179, 203-218); liver leaks onto an
  unidentified surface (4-34). chole_derot: identity swap neck_pedicle -> gallbladder body (163-206) and the
  gallbladder swallows the neck lobe (164-246); dissector blips 232-246; fat on shadow 0-50. lung: no instruments;
  masks carried across the 9 cuts (pleura over 0-289, bed in unprompted shots).
  -> r02 repairs: repairs_r02.json per clip (extra positive / negative prompts at the QA's frames, a new object for
  the unidentified oval, blanked ranges), per-shot tracking with memory reset for lung, lung instrument agent.

## r02 segmentation repairs from the r01 visual QA
- seg v03 (repairs_r02.json: QA prompts added; a new object for chole_a's unidentified red-brown oval organ;
  blanked ranges): regression - prompts with only negative points make the SAM 3 tracker DROP the object from that
  frame on (chole_derot gallbladder vanished 206-246; chole_a red strand band present in 8 % of frames).
- seg v04: a negative-only repair now also gets positives taken from the reviewed version's mask (deepest interior
  points >= 50 px from every negative: t2s.seg3.keep_points), plus explicit precedence between overlapping tissues
  (chole_derot: gallbladder minus neck_pedicle; chole_a: red strand band minus oval organ and neck).
  chole_derot: gallbladder present in all frames, neck lobe separate again (identity swap fixed); chole_a: all 8
  objects present in all frames, band 5.4 % of the image (temporal IoU 0.92), oval organ 0.94.
  | clip | v02 -> v04 (temporal IoU median, frames present) |
  |---|---|
  | chole_a | neck_pedicle area 4.8 -> 7.3 % (distal tube + proximal segment back), oval organ new (0.94, 100 %) |
  | chole_derot | gallbladder 0.87 -> 0.86, 100 % present; neck_pedicle 0.76 -> 0.71 (blanked 175-199 where hidden) |
- Instruments in chole_derot stay flickery (temporal IoU 0.55): fast motion and passes in front of the lens.
- prep masks of chole_a / chole_derot switched to v04 (cameras and depth unchanged: they use instruments + static
  pixels only).
- lung_mln: instrument agent added 4 instruments (Harmonic, steel grasper, STORZ grasper, suction) with prompts in
  every shot where visible; v03 = per-shot tracking with memory reset at the 9 cuts + QA repairs (running).

## r02 instrument agent (3D, t2s/instruments.py): rigid library + one port per tool
Library: grasper, curved Maryland dissector, needle holder, suction cannula (shaft radius, jaw length / radii /
curve / max opening; jaw opening = 1 DoF); MuJoCo builder with yaw / pitch / insertion / roll / jaw joints about the
port (forward kinematics vs fitted tip <= 0.007 mm; servo replay tracks the tip at 0.06-0.46 mm median).
Fit per frame: two-way silhouette distance (one-sided at an occluded end), low-weight depth on shaft pixels,
a free-space term (the shaft lies in front of the tissue beside it), hidden-tip prior, temporal smoothness, one
port per clip; size: nominal 5 mm unless > 10 % of free-space samples put the shaft behind tissue (then tool and
port scale together, kappa).
| clip / tool | IoU (visible part) | tip px median (p90) | axis deg | port-plane mm | size (kappa) |
|---|---|---|---|---|---|
| chole_a grasper | 0.92 | 1.5 (2.4) | 2.3 | 7.5 | 5 mm (1.00) |
| chole_a suction cannula | 0.90 | 1.8 (3.0) | 0.75 | 1.4 | 4.76 mm (0.95) |
| chole_derot grasper_top | 0.89 | 1.5 (5.3) | 3.5 | 10 | 3.3 mm (0.65) |
| chole_derot dissector_right | 0.79 | 3.9 (16) | 2.9 | 3.7 | 3.2 mm (0.64) |
| liver_s4 needle_holder_R | 0.90 | 2.2 (9.2) | 3.0 | 10 | 4.7 mm (0.945) |
| liver_s4 needle_holder_L | 0.90 | 1.4 (5.5) | 1.0 | 4.4 | 3.95 mm (0.79) |
- Findings for other steps: (1) chole_derot's metric depth is ~35 % too near for 5 mm shafts (at nominal size
  41-54 % of the shaft samples lie behind the tissue they are seen in front of), consistent with that clip's weak
  multi-view solve (no wide-baseline SIFT matches, ruler residual 0.30) -> the clip's whole world is scaled by
  1/kappa (~1.55) at assembly so that everything (organs, background, cameras, tools) is consistent with 5 mm tools
  (a uniform world scale leaves every projection unchanged). (2) chole_a keyframe 0 sits on the blurred fast pan:
  both tools miss their ports in frames 0-6 -> those frames are excluded from the simulation. (3) The suction
  cannula's hidden tip is inside the gallbladder in 203 frames, placed 15.9 mm (median) past the visible end, capped
  by the organ's far wall -> the spec's decompression opening: legal in assembly only at that entry site.
- Limits: roll observable only through open / curved jaws; port depth along the line of sight is the weakest
  direction; liver_s4 R's port is a lower bound (shaft nearly parallel to the image).
- lung_mln seg v03 (per-shot tracking, memory reset at the 9 cuts, 4 instruments + QA repairs): no carry-over across
  cuts any more (pleura only in C4 / D: 25 % of frames; lymph nodes / fat 37 %; dissection bed 82 %; shot D's bed split
  in two); instruments present only in their shots (Harmonic 93 %, temporal IoU 0.72). Lung stays perception-only.

## r02 background agent (t2s/background.py, v08 per clip)
Roles decided automatically (spec + mask area + a motion test: optical flow minus the camera-induced flow): organs
excluded and filled behind; large static items -> surface (liver in the chole clips; diaphragm + omentum in
liver_s4, kept as surface labels = attachment targets); small items -> closed bodies only if they stand out of the
surface and their carved shape reproduces their own mask (IoU >= 0.3). Surface: static pixels fused in a reference
camera (per-frame depth scale onto the multi-view median), one thin-plate height field without holes, kept 3 mm
behind the back of every organ's 4D model in every frame; collision = 24 per-tile height fields along camera rays
(early ray hits <= 0.1 %) + a watertight slab; query helper Background(clip).ray_distance(X, k).
| clip | coverage | depth residual median per third mm | held-out photometric NCC (local) | in front of organs (median / p90) | organ vertices behind surface | bodies |
|---|---|---|---|---|---|---|
| chole_a | 1.00 | 3.96 / 3.40 / 3.61 | 0.940 (0.585) | 1.0 % / 23 % | 0 % every frame (v1: 15-24 %) | oval organ (IoU 0.81), blood (0.72), red strand band (0.57) |
| liver_s4 | 1.00 | 5.50 / 3.09 / 2.05 | 0.915 (0.442) | 0.04 % / 2.3 % | liver v10: 20 % up to 24 mm (too thick) | gauze (0.64) |
| chole_derot | - | 10.7 (surface ~10 mm behind video depth) | local NCC 0.17 | 0.06 % | 26 % | fat (0.30, unreliable) |
- chole_derot is limited by its cameras and depth (per-frame depth scale 0.74-1.27 needed vs 0.95-1.05 in chole_a;
  textured surface off by several px even at keyframes; re-registering the cameras to the background did not
  converge) - the second independent agent to flag this clip's geometry (instrument agent: depth ~35 % too near).
  -> r03: geometry agent for chole_derot (same video and scope as chole_a: focal can be fixed; denser wide-baseline
  matching on the static liver; stronger shaft rulers; per-frame scale prior).
- liver_s4: the organ agent's liver v10 (a 35 mm deep primitive for a thin lobe tip) goes through observed background
  (20 % of vertices) -> feedback to the organ agent.

## r03 assembly (t2s/assemble.py) on chole_a: building a scene where only the spec's connections hold the organ
Scene: gallbladder (organ agent v10: 1278 nodes / 5828 tets, E 1.2 kPa / nu 0.45 from the spec's "fluid-filled"),
the two instruments driven through their ports (instrument agent v01: tips follow their fits to 0.5-0.7 mm), the
background as static collision (background agent v08) and its three convex bodies. Rules from the spec + attach
hints only: 8 pedicle vertices -> springs to their first-frame positions ("suspended by cystic duct / artery");
25 opening vertices -> the suction cannula does not collide with the gallbladder (declared puncture); closed
grasper jaws at the first frame hold the nearest organ vertices (the jaws hold the neck, which has no model: the
nearest gallbladder vertices are 6.4 mm away). Frames 9-250 (0-8: camera error at keyframe 0).
Debugging (all logged as runs in the session, numbers here):
- MuJoCo 3.x flex-heightfield collision does not work (a tet block falls 9 mm through an hfield; a box or plane
  holds it) -> the background is 500-800 thin boxes tangent to the fused surface near the organ, each pushed back
  until no first-frame organ vertex lies behind its face (unpushed boxes shoved the organ 2.8 mm during settling).
- The agent's rest shape is not the first frame's shape: released from it, the free organ springs back and drifts
  9 mm in 20 frames -> the first simulated frame is the stress-free state (rest='start').
- With only the pedicle springs and the grasp, the organ still drifts 10-15 mm over 60 frames although its 4D
  moves 3.5 mm: nothing holds a free organ against small persistent pushes. Replacing the grasper's along-view
  motion by the held tissue's (its depth is the weakest part of the tool fit) did not help (13 vs 12.5 mm).
  An elastic foundation under the unseen back (the 224 'hidden_back' vertices; surrounding organs in reality),
  springs to the first frame, k >= 0.5 N/m: the drift is gone but the organ then barely moves (0.1-0.3 mm vs the
  4D's 3.5 mm). -> r03 sweep: foundation stiffness 0.05 / 0.2 / 0.5 N/m, soft vs rigid grasp of the neck.
- r03 sweep (frames 9-250; foundation 0.05 / 0.2 / 0.5 N/m; soft vs rigid grasp): sim IoU 0.73-0.74 (2D 0.77-0.78; the
  organ's own 4D 0.82), organ behind background 0 % in every frame (v1: 15-24 %), no illegal tool-in-organ frames
  with the soft grasp (3 with the rigid one, plus 84 inverted tets), but motion explained ~0 (-0.02..+0.01).
- Visual QA r03 (independent, images + its own renders): "stable, clean shape, but nothing physical happens":
  (1) the organ never moves (centroid <= 0.2 mm; the foundation springs act as a liver bed, contradicting the spec);
  (2) the grasper holds nothing: the closed jaws are 6.4 mm (later 10-12 mm) above the organ - the neck between is
  not modelled; (3) the cannula is a ghost (no collision): in 17 frames of phase 1 it exits through the far wall,
  the entry point wanders 6.3 mm rms, and in phase 2 the input tip (instruments v01, capped against organ v09) is
  outside the organ; (4) the organ is 9.2 ml (v1's template fit: ~28 ml) and its fundus is cut off at the image
  edge; (5) gaps in the box support. Also: Q.signed_distance gave false 'inside' hits on a body (use winding).
  -> sent to the organ agent (neck up to the jaws, complete the fundus, one puncture site) and the instrument
  agent (v02: cannula refit jointly with the organ, one shared puncture point, shafts in front of the background).
- Assembly change for r04: the puncture as a trocar-like coupling - while the tool is inside, the opening vertices
  near its shaft are pulled toward the shaft axis (perpendicular springs, k_puncture), so the wall moves with the
  tool's sideways motion and lets it slide in and out.

## r04 puncture coupling (chole_a, frames 9-250)
| foundation N/m, puncture N/m, grasp tc | sim IoU (2D) | err vs 4D mm | motion explained | inverted tets (max of 5828) | illegal tool-in-organ frames |
|---|---|---|---|---|---|
| 0.2, 0 (r03) | 0.734 (0.776) | 3.99 | +0.002 | 2 | 0 |
| 0.2, 2 | 0.714 (0.777) | 4.16 | -0.041 | 1076 | 1 |
| 0.05, 2 | 0.675 (0.746) | 4.26 | -0.067 | 1116 | 0 |
| 0.05, 10 | 0.650 (0.742) | 4.35 | -0.090 | 1128 | 2 |
| 0.2, 2, grasp tc 0.02 | 0.714 (0.770) | 4.02 | -0.006 | 1316 | 0 |
- Negative: pulling the opening vertices toward the shaft axis (which runs inside the organ beyond the puncture)
  drags surface vertices into the interior: ~1100-1300 inverted tets, worse fit, the organ moves but not like the
  video. Reverted (k_puncture 0). The puncture needs a proper hole / a coupling at the entry point only, and an
  instrument fit consistent with the organ (agents asked, r05).

## r05 organ agent hand-back (t2s/organ.py: primitive -> FFD -> uniform tet lattice -> 4D) and the neck link
Generic fitter: organs = spec primary / secondary with a mask (tubes and membranes listed, not fitted); volume rule
from the spec (fluid-filled: near-constant, 'drained' allowed only if the spec declares an opening / drain action);
unknown pixels = instruments, objects in front, see-through objects, junction bands; superquadric fitted to all
keyframes with tracked keyframe poses, thickness prior + aspect cap (unconstrained it collapsed to a plate), FFD;
video depth trusted only up to a per-frame scale (0.86-1.13 needed on chole_a, 0.89-1.12 liver, 0.78-1.12 derot);
background surface as a back limit. Self-test: synthetic deforming superquadric, 4D IoU 0.92, ~1 mm error.
| clip / organ | 4D IoU per third (own masks) | static rest | primitive alone | depth residual mm | volume | inverted tets |
|---|---|---|---|---|---|---|
| chole_a gallbladder v15 | 0.826 / 0.836 / 0.918 | 0.617 / 0.537 / 0.769 | 0.553 / 0.455 / 0.713 | 4.3 / 5.8 / 2.2 (0.7 / 1.7 / 0.4 after scale) | 8.8 ml, 1.04-0.98 of rest | 0 |
| liver_s4 liver v14 | 0.841 / 0.885 / 0.852 | 0.577 / 0.715 / 0.667 | 0.534 / 0.675 / 0.650 | 2.5 / 2.9 / 3.7 | 6.4 ml wedge (v10: 15.3 ml) | 0 |
| chole_derot gallbladder v12 (provisional) | 0.740 / 0.698 / 0.768 | 0.11 / 0.50 / 0.55 | 0.12 / 0.48 / 0.56 | 1.9 / 2.4 / 13.2 | 28.9 ml (+-1 %) | 0 |
- chole_a vs v1 (v1's bands, own masks; different masks and geometry, indicative): v2 primitive 0.841 / 0.830 /
  0.887 vs v1 template 0.832 / 0.824 / 0.859 -> the primitive-initialised fit is at least as good as the template.
- liver v14 answers the background feedback: vertices behind the background 20-25 % (v10) -> 1.6-1.8 %.
- The agent kept the neck out of the closed volume (a merged gallbladder + neck_pedicle fit was worse on every
  metric: own IoU 0.76 vs 0.83) -> assembly models the short neck between the jaws and the organ as a link:
  spatial-tendon springs (k_neck) from the grasper tip to the organ's 8 nearest vertices (7.3 mm long).
- Assembly: elastic foundation under the unseen back only as the spec allows: 'free' organs (gallbladder: "free of
  the liver bed", high) get a weak one (k_foundation_free, surrounding tissue), embedded organs (liver) a normal one.
- Short test (chole_a v15, frames 9-69): with the neck link the organ now moves with the grasper (0.3-0.8 mm) but
  less than its 4D (1.0-2.7 mm); error 1.0-2.7 mm (= roughly the 4D's own motion).
- Still open: chole_a's organ is the visible part only (8.8 ml; v1 template ~28 ml; fundus cut at the image edge)
  -> asked for v16 with the out-of-image part as unknown + smooth extension.
- r05 chole_a sweep (gallbladder v15, frames 9-250, neck link, weak foundation for the 'free' organ):
  | k_neck N/m, foundation N/m | sim IoU (2D) | err vs 4D mm | motion explained | inverted | organ behind bg | illegal |
  |---|---|---|---|---|---|---|
  | 2, 0.05 | 0.723 (0.782) | 2.89 | **+0.084** | 4 | 0 % | 0 |
  | 10, 0.05 | 0.725 (0.781) | 2.91 | +0.079 | 24 | 0 % | 0 |
  | 10, 0.02 | 0.720 (0.769) | 2.96 | +0.061 | 22 | 0 % | 0 |
  | 10, 0 | 0.658 (0.688) | 7.20 | -1.28 (drifts away) | 25 | 0 % | 0 |
  First positive motion explained in v2 (the 4D's own IoU here 0.80). Without any support the free organ drifts
  (-1.28): the surrounding tissue is a real part of the boundary condition, but weak (0.05 N/m per vertex).
- Session restart (2026-10-08 ~10:00): background jobs and agents were stopped; resumed the geometry agent
  (chole_derot: focal fixed to chole_a's, 84k wide matches, held-out long-range error 10 px vs 224 px, ruler
  residual 0.15 vs 0.30 in its r1), the organ agent (v16: fundus beyond the image edge) and the instrument agent
  (v02 done: grasper IoU 0.92, size 5.04 mm; cannula refit in progress); liver_s4 r05 sweep restarted.
- liver_s4 r05 (first assembly; liver v14, two needle holders, background v08; frames 0-174): broken - sim IoU
  0.46-0.51 (the organ's own 4D 0.83), motion explained -0.16..-0.29, 1000-1500 inverted tets, tools inside the
  organ in 8-22 frames, organ behind background <= 0.7 %. The needle holders work in contact with the lobe tip and
  the knot; as rigid colliders they plough through the wedge. To fix in r06 (tool-organ contact at the suture
  site, where the spec declares the suture as the only opening).
- RAM: the machine (51 GB) ran out of memory twice during r02-r05 (three to four fitting agents + SAM 3 tracking +
  4-job sweeps at once). From now on: one heavy job at a time (one agent, sweeps with <= 2 jobs), a watchdog
  (t2s/memguard.py, stops this project's largest job below 8 GB free).
- Result page for v2 (video + reconstruction per clip): t2s/report.py -> site/v2/media/<clip>.mp4 (video | SAM 3 masks |
  4D reconstruction rendered through the scope camera | on video | simulation for chole_a); published as a new
  artifact https://claude.ai/artifact/82NXJveVMwHGb1eLWCdqHc (v1).
- Result page rebuilt as an interactive 3D viewer (the user asked for the reconstruction itself, interactive, not
  rendered videos): t2s/export_viewer.py writes each clip in the v1 viewer's scene format (site/v2/data/<clip>:
  background agent surface + its bodies as convex pieces, organ surface with its 4D ('recon4d') and the round's
  simulation ('sim4d'), instruments posed per frame about their ports, scope camera per frame, the video);
  site/v2/build_page.py patches the v1 viewer (site/viewer/index.html) for these scenes and appends the results
  (site/v2/results.html). Scenes: chole_a (4D + r05 sim), liver_s4 (4D + the broken r05 sim, shown as such),
  chole_derot (4D on the old geometry, provisional). Checked in a local preview (all three scenes, both layers,
  scope view against the video frame, shape mode, no console errors). Published as version 2 of
  https://claude.ai/artifact/82NXJveVMwHGb1eLWCdqHc.

## r06 (2026-10-08)
- Geometry agent hand-back for chole_derot: use `sift2_r1` (outputs/variants/chole_derot_t2s+sift2_r1; r2 and the
  strict hold-out solve never completed). Same evaluation for old / new / chole_a (control): per-frame static depth
  scale p5-p95 0.82-1.25 -> 0.94-1.04 (chole_a 0.95-1.05); ORB probe error at >= 10 keyframes 224 -> 10 px (chole_a
  6 px); static surface depth 53 -> 73 mm; shaft samples behind tissue 68 % -> 24 % (chole_a 17 % by the same test).
  Corrections to the earlier entry: "10 px vs 224 px" is NOT a strict hold-out (ORB on the same static pixels the
  solve matched; 157 matches, p90 still 245 px); the signed ruler residual ~0 is by construction (the solve rescales
  to it). Known defects: per-frame poses jitter (median 1.2 mm and 0.79 deg per frame^2; scope path 332 mm vs 139 mm),
  focal length assumed from chole_a, absolute scale without an error bar, correction grid read centre-aligned but
  fitted corner-aligned (~4 % median depth difference), and it had been checked by numbers only.
  t2s/geomfix.py on disk is the r2 code and does not reproduce sift2_r1 (see its docstring and geometry/NOTES.md).
- First picture of the new geometry and a decision on the jitter (t2s/geomcheck.py: lean comparison, ~2 GB;
  `t2s.geomfix eval` grew to 31 GB in one process and was killed - do not use it). Cameras low-passed over time
  (`t2s.geomfix smooth`, Gaussian on centre + rotation vector together):
  | geometry | probe px, keyframe gap 1-2 / >= 10 | static NCC k->k+10 (local) | depth scale p5-p95 | jitter mm, deg / frame^2 | scope path mm |
  |---|---|---|---|---|---|
  | sift (old) | 3.6 / 170.9 | 0.819 (0.158) | 0.823-1.249 | 0 (interpolated) | 139 |
  | sift2_r1 | 3.1 / 5.4 | 0.884 (0.458) | 0.936-1.036 | 1.20, 0.79 | 332 |
  | sift2_r1s2 (sigma 2 frames) | 4.0 / 8.5 | 0.875 (0.402) | 0.948-1.036 | 0.21, 0.11 | 169 |
  | sift2_r1s4 (sigma 4 frames) | 6.4 / 13.0 | 0.865 (0.309) | 0.949-1.034 | 0.09, 0.05 | 122 |
  Smoothing costs static consistency, so part of the "jitter" is real scope motion or compensates per-frame depth
  errors; sigma 2 removes 5/6 of it for ~1 px. Adopted for the clip: **sift2_r1s2** (t2s/views2.py GEOMETRY; Views
  now carry `.geometry`). Warp checker (geometry/warp_checker.jpg: frame k warped into frame j, checkerboarded with the
  real frame): with the new geometries the static anatomy lines up across 100 frames; the old one has almost no
  valid overlap at that distance. No gross error seen.
- Workspace: the clip's models fitted on the old geometry (organs v06 / v12, instruments v01, background v03-v08)
  moved to outputs/t2s/chole_derot/_geom_sift/; new-geometry versions: instruments v02, organ v15, background v09.
- Instruments v02 (chole_derot, new geometry; t2s.instruments unchanged, 112 s): the instrument agent's own free-space
  test now passes without any size scaling - grasper 0 % of shaft samples behind the tissue beside it (implied scale
  1.00; old geometry: kappa 0.64), dissector 15 % (implied 0.97; old 54 %). Grasper IoU 0.90, tip 1.4 px; dissector
  IoU 0.84, tip 3.0 px. Independent confirmation that the ~35 % scale error is gone.
- Organ v15 (chole_derot gallbladder, new geometry; t2s.organ unchanged, v15 settings, no background model yet so
  w_bg inactive; 721 s, 1.7 GB): 4D IoU 0.79 / 0.80 / 0.79 (old geometry v12: 0.74 / 0.70 / 0.77), depth residual
  4.2 mm mean / 0.8 mm median, rest volume 22.2 ml (58 x 37 x 21 mm; old geometry 28.9 ml at the wrong scale), volume
  kept within 0.994-1.003, 1 inverted tet in one frame, keyframe rotations up to 96 deg relative to the start frame
  (110). Static rest shape alone: IoU 0.54 - the motion matters in this clip. Attachments from the spec: liver bed
  'free' -> none; pedicle (cystic duct + artery) 8 vertices; hidden back 277 vertices.
- Background v09 (= v08 settings, new geometry, with organ v15 and instruments v02; 2 min, 3.5 GB): one connected
  surface, 0 cells contradicted by visible organs / instruments after 2 rounds, 1 body (peri-gallbladder fat).
  Against the old-geometry v08 (same code): fused-surface depth error median 10.5 -> 2.3 mm per frame, local photometric
  NCC of the re-rendered static scene 0.17 -> 0.48, organ-model vertices behind the background 26 % -> 6 %,
  instrument pixels in front-of-surface conflict 1.6 % -> 0 % (medians over frames). The background agent's and the
  instrument agent's own tests, which first exposed the scale problem, both pass now.
- First assembly of chole_derot (r06; organ v15, instruments v02, background v09; frames 0-246, 8 min per run):
  | pedicle anchor N/m | sim IoU (2D) | 4D's own IoU | err vs 4D mm | motion explained | inverted | organ behind bg | illegal tip-in-organ frames |
  |---|---|---|---|---|---|---|---|
  | 2 | 0.441 (0.518) | 0.739 | 8.28 | -0.005 | 208 | 6.8 % | 6 |
  | 20 | 0.440 (0.517) | 0.739 | 8.30 | -0.009 | 195 | 6.7 % | 5 |
  NEGATIVE: the simulated gallbladder stays where it started (centroid moves 0.4-2.2 mm; in the 4D it moves 4-10 mm
  and turns by up to 96 deg), IoU by thirds 0.57 / 0.36 / 0.37. No grasp and no neck link formed (no closed jaw on
  or near the organ in frame 0; the tools push in this clip). So far the answer to "does the free gallbladder's
  rotation follow from the physics" is no.
- Why (new check t2s/recon_check.py: the fitted instruments against the fitted organ 4D, before any simulation;
  inside test by winding number - r2s.quality.signed_distance mis-signs points far from the surface, an earlier
  reading of "4 cm inside" was that artefact): the RECONSTRUCTION is not consistent between agents. Frames where
  the shaft is inside the organ model although no opening is declared:
  | clip, instruments | instrument | inside / frames seen | max depth mm | within 2 mm of the surface | median gap otherwise mm |
  |---|---|---|---|---|---|
  | chole_derot v02 | grasper | 11 / 36 | 11.7 | 3 | 5.3 |
  | chole_derot v02 | dissector | 18 / 44 | 9.2 | 9 | 6.5 |
  | chole_a v02 | grasper | 1 / 51 | 1.1 | 8 | 3.7 |
  | chole_a v02 | suction cannula | 41 / 50 (opening declared) | 10.1 | 4 | 2.5 |
  | liver_s4 v01 | needle holder R | 0 / 35 | - | 0 | 20.9 |
  | liver_s4 v01 | needle holder L | 2 / 20 | 2.2 | 3 | 12.3 |
  An instrument's depth along the view is only weakly observed (silhouette + fixed port), so its shaft ends up
  through the organ in a third of the frames (chole_derot) or far in front of it (liver). The assembly then asks
  MuJoCo to resolve an impossible start instead of a push.
- Fix in the instrument fit (t2s/instruments.py, `organ_solid`, default on): at the pixels where an instrument is
  SEEN it occludes what is behind it, so the first surface behind its shaft is min(background, front of the organ
  models) there (organ fronts rasterised per frame at the occupancy resolution); skipped for an instrument with a
  declared opening (chole_a's cannula). Uses the existing background term, no new solver terms.
- Instruments v03 (chole_derot, with the organ-front rule against organ v15): did NOT resolve it - grasper inside
  9 / 36 frames (max 8.5 mm), dissector 18 / 44 (max 10.0 mm); ports moved by ~15 mm, IoU 0.90 / 0.83 (v02 0.90 /
  0.84). With a fixed port and the image line of the shaft, an instrument has no per-frame depth freedom: only the
  port (3 numbers for the whole clip) can move, so the remaining conflict has to be on the organ's side.

## r07 (2026-10-08)
- Visual QA agent on r06 (images only; report in the session log): (1) the organ 4D is 15-50 px too big exactly where
  instruments hide it (frames 40-60, 130-150, 200-240): those pixels are 'unknown' to the organ fit, so the model
  grows into them for free - this, not the instrument depth, is the tool-in-organ conflict (the instruments are never
  behind the organ in the video: always in front of or beside it); outline polygonal with spikes (10-50, 120);
  (2) the dark purple bulb that turns into view at 165-180 is the gallbladder neck, a third of the body's size, and
  is not in the model (mask 'neck_pedicle' kept separate) - from frame 180 both instruments work on it, so the
  simulation's tools push nothing; (3) background v09: its 'fat' body is a scene-wide wedge overlapping the organ's
  footprint, surface in front of the lower 40 % of the gallbladder at frame 49; (4) instruments fine in 2D (v03's
  jaw 20 deg off at f148, dissector roll flips around 113-150); (5) cameras: no doubled static structures with
  sift2_r1s2, indistinguishable from sift2_r1; (6) what the instruments do: all pushing and prodding (right pole by
  the neck 0-50, dissector lifting / rolling along the lower margin 60-105, grasper on the dome 110-165, both at the
  body / bulb junction 180-246), never a sustained grasp of the sac; largest motion 50->60, 100->110, 165->180.
- Tissue stiffness (assemble `young_scale`; frames 0-60, instruments v03, organ v15):
  | E x | damping, ts | sim IoU | err vs 4D mm | motion explained | inverted |
  |---|---|---|---|---|---|
  | 1 (1.2 kPa) | 0.002, 6.25e-5 | 0.609 | 5.86 | -0.08 | 161 |
  | 10 | 0.0002, 6.25e-5 | 0.619 | 6.77 | -0.25 | 3 |
  | 10 | 0.0002, 3e-5 | 0.622 | 6.67 | -0.23 | 0 |
  | 30 | 0.00007, 3e-5 | 0.624 | 7.03 | -0.30 | 0 |
  (x 10 and x 100 with the damping unchanged blow up in frame 1: the flex damping scales with the stiffness.) A
  stiffer sac no longer crumples under the tools (inverted tets 161 -> 0-3) and now moves when pushed - but in the
  wrong direction, because the tools start inside an oversized organ. Stiffness x 10 with damping / 10 is kept as
  the candidate; the organ has to be fixed first.
- Organ fit, two changes (t2s/organ.py, settings 'v17'): `w_ins` - surface samples inside an instrument's shaft
  (the instrument agent's per-frame pose, frames where it is seen, instruments with a puncture record excepted)
  are penalised in the rest fit and in the 4D; `merge_parts` on (neck_pedicle joins the gallbladder mask). The
  instrument poses are taken from v02 (fitted without any organ) to avoid a circular bias.
- Organ v17 (chole_derot; merged neck, surface-in-shaft w_ins, w_bg active with background v09; 695 s): NOT adopted.
  Shafts still inside (grasper 10 / 36 frames, dissector 16 / 44, max 8.6 / 6.5 mm), 4D IoU 0.73 (v15 0.79), and the
  organ became a plate: 59 x 37 x 12.8 mm, 11.8 ml (v15: 20.7 mm thick, 22.2 ml). Two causes: (a) w_ins looked at
  surface samples inside the shaft and a shaft buried in the model has none; (b) a ratchet between two agents -
  the background behind a free-hanging organ is never seen, the background agent fills it from the previous organ
  model's back surface, and the organ's w_bg then squeezes the next organ against that fill.
- Organ v18 settings: w_ins along the camera rays (front-facing surface samples whose ray passes through a seen
  shaft must lie behind its axis: volumetric, unit-tested), w_bg off, and for organs the spec calls fluid-filled
  the thickness prior asks for a round cross-section (thick_ratio 1.0 instead of 0.8).
- liver_s4, stiffness x 10 (r08, same settings as r05's best otherwise): sim IoU 0.507 -> 0.539, illegal tool-in-organ
  frames 14 -> 0, inverted tets 1008 -> 676, motion explained -0.16 (unchanged), volume down to 4.6 of 6.4 ml. Still
  broken, and the run's own info shows why: needle holder L 'grasps' 19 liver vertices and needle holder R gets a
  neck link to 8 - the assembly's rule "closed jaws on tissue in the first frame hold it" fired for needle holders,
  which hold the needle and the thread (the clip ties a knot), and dragged the lobe tip. Rule changed: only
  instrument types that hold tissue (grasper, dissector) form grasps / neck links (assemble.HOLDS_TISSUE).
- Organ v18 (chole_derot; ray-based w_ins in the rest fit and the 4D, w_bg off, round cross-section; 654 s): NOT
  adopted - the fit ran away: 152 x 136 x 65 mm, 696 ml. A silhouette-preserving retreat (farther + bigger) satisfies
  "behind the instruments" for free, and the per-frame depth scale lets it happen. Shafts still inside (8 / 36,
  27 / 44).
- Geometric projection instead of a refit (t2s/consistency.py: nodes whose camera ray passes through a seen shaft are
  moved back behind it, neighbours follow): v15 -> v19, NOT adopted - the pushes needed are a median 13 mm and 29 mm
  at the 90th percentile per frame, more than the organ is thick (21 mm); 332 inverted tets, volume 0.85-1.09 of
  rest. By the instrument poses, the shafts lie behind most of the organ in the pixels where they are seen in front
  of it.
- Is it one global scale? Organ 4D scaled about each frame's camera centre (silhouettes unchanged), instruments
  fixed (v02): x 1.0 grasper inside 11 / 36, dissector 18 / 44; x 1.2: 1 / 36 and 14 / 44; x 1.3: 0 / 36 (but then
  21 mm from the organ, never touching) and 7 / 44; x 1.5: 0 and 0, both tens of mm away. With v03: x 1.2 gives 0 /
  36 and 5 / 44, again with no contact left. There is no scale at which both instruments touch the organ without
  entering it: the disagreement is frame-dependent, about a centimetre, and neither agent's depth is good enough
  to say which side is wrong (the shafts are metric rulers through their apparent width; the organ's depth is the
  monocular depth map, calibrated on static tissue only). This is the open problem of the clip: contact physics
  needs the tool-organ gap right to about a millimetre, the reconstruction gives it to about a centimetre.
  Rejected organ versions are kept in organs/gallbladder/_rejected/ (v17, v18, v19). In use: organ v15.
- liver_s4 r08b (no needle-holder grasps, stiffness x 10): first clean liver simulation - sim IoU 0.665 (2D 0.679; the
  4D's own 0.827), inverted tets 0 (r05: 1008-1519), volume 6.28-6.30 of 6.41 ml, 1 sampled frame with a tool tip in
  the organ, organ behind background 0.2 %. But the lobe stays where it starts (motion explained -0.10; IoU by
  thirds 0.60 / 0.66 / 0.73): in the 4D it moves ~7 mm on average, and what moves it - breathing and the pull of
  the thread being tied - is not in the model (no suture, no respiration).
- chole_derot r07b, full clip, stiffness x 10 (damping / 10), instruments v03, organ v15:
  | bodies collide | sim IoU (2D) | err vs 4D mm | motion explained | inverted | volume ml (rest 22.15) | illegal tip-in-organ frames |
  |---|---|---|---|---|---|---|
  | yes | 0.459 (0.515) | 8.83 | -0.07 | 9 | 22.0-22.05 | 12 |
  | no | 0.459 (0.515) | 8.79 | -0.07 | 9 | 22.0-22.05 | 12 |
  The stiffer sac keeps its shape and volume (inverted tets 208 -> 9) but still does not follow (IoU by thirds 0.57 /
  0.40 / 0.39), and the background 'fat' wedge is not what holds it (no difference without it). Final status of the
  clip: reconstruction usable per agent, forward simulation negative.
- chole_a r08 (regression + stiffness): the r05 configuration reproduces with the current code (sim IoU 0.723, 2D
  0.781, motion explained +0.083, 1 inverted tet, 0 illegal frames, 0 % behind the background). Stiffness x 10:
  0.716 / +0.055 / 0 inverted - no gain; chole_a keeps the table's 1.2 kPa. Tissue stiffness is not what limits the
  motion in either clip.
- chole_derot r07b, full clip with stiffness x 10 (instruments v03, organ v15):
  | background bodies collide | sim IoU (2D) | err vs 4D mm | motion explained | inverted | organ behind bg | illegal tip-in-organ frames |
  |---|---|---|---|---|---|---|
  | yes | 0.459 (0.515) | 8.83 | -0.07 | 9 | 5.0 % | 12 |
  | no | 0.459 (0.515) | 8.79 | -0.07 | 9 | 5.1 % | 12 |
  The fat wedge the visual QA flagged is not what holds the organ (no difference). Against r06 / r07 at x 1 (IoU
  0.44, 181-208 inverted, 5-6 illegal frames): no crumpling any more, the same failure - the organ stays put.

## r08 (2026-10-08)
- chole_a regression with the current code (organ v15, instruments v02, background v08, frames 9-250, neck link):
  | E x | sim IoU (2D) | err vs 4D mm | motion explained | inverted | organ behind bg | illegal |
  |---|---|---|---|---|---|---|
  | 1 | 0.723 (0.781) | 2.89 | +0.083 | 1 | 0 % | 0 |
  | 10 (damping / 10) | 0.716 (0.764) | 2.98 | +0.055 | 0 | 0 % | 0 |
  r05's best is reproduced (0.723 / 0.782 / +0.084). A stiffer sac does not make the gallbladder follow the grasper
  more (the "one third of the amplitude" problem is not a stiffness problem). chole_a keeps E x 1; x 10 is what the
  two clips with pushing / no grasp need to stop crumpling (liver 1008 -> 0 inverted, chole_derot ~200 -> 9).
- Session note: the desktop session restarted while the r07b / r08 sweeps ran; they finished on their own.

## r09 (2026-10-08)
- Diagnostic for the clip's open problem (assemble `tool_depth=contact`, t2s.assemble.contact_consistent_joints): if
  the contact geometry WERE consistent, would the physics follow? For a tool seen in a frame whose shaft would be
  inside the organ's 4D, its tip is slid towards the scope along its line of sight (image position and port
  unchanged) until the shaft clears the organ; the slide is min-filtered and smoothed over time. This borrows the
  organ's reconstruction for the one tool coordinate the image does not give - as the holding tool of chole_a
  already does (hold_depth='organ') - so it is a test of the physics, not a prediction from the instruments alone.
  Result (chole_derot, organ v15, instruments v03, background v09, frames 0-246): the grasper is slid in 53 of its
  175 seen frames (median 5.9 mm, max 11.2 mm), the dissector in 124 of 215 (median 13.0 mm, max 20.7 mm).
  | E x | sim IoU (2D) | err vs 4D mm | motion explained | inverted | illegal tip-in-organ frames |
  |---|---|---|---|---|---|
  | 1 | 0.446 (0.522) | 8.19 | +0.005 | 31 | 2 |
  | 10 | 0.458 (0.513) | 8.78 | -0.066 | 3 | 4 |
  Same as without the correction (0.437-0.459, -0.07..+0.005): with the instruments made consistent with the
  reconstructed organ, the simulated gallbladder still does not follow it. Fixing the contact geometry alone is NOT
  sufficient for this clip. What remains: the instruments mostly work on the neck / pedicle (visual QA), which is not
  in the organ model, so nothing in the simulation transmits that manipulation; and tools that only just clear the
  reconstructed organ push the simulated one only where it sticks out past the reconstruction.
- Two desktop sessions (forks of the same conversation) have been writing to this worktree since ~13:50: this one
  ran r09 and the final visual QA; the other one rewrote the result page (site/v2/page.html, results.html,
  build_page.py, export_viewer.py). This session does not touch site/v2 after 14:12.

## r10 (2026-10-08)
- Final visual QA agent over the three clips' final states (images only: organ / instrument sheets, sheet_sim.jpg,
  orbit.jpg; chole_a r08 E x 1, liver_s4 r08b, chole_derot r07b and r09 E x 10). Its verdicts:
  - chole_a: usable as a silhouette-level reproduction - right place and the right drift in all 8 panels (7 fine,
    frame 145 somewhat off: jagged apex spilling onto the cannula and the neck; a spurious spike at frame 111). NOT a
    reproduction of the decompression: in 3D the final organ is a plump convex egg while the video shows a
    collapsed, wrinkled cone tented up to the grasper - only the 2D outline matches (the simulation has no fluid
    removal; the organ agent's 4D lets the volume fall, MuJoCo's elastic body keeps it). Grasper jaw tips overlap
    the organ apex with no tissue fold between them. Background v08 has stretched curtains / spikes in off-axis
    views (fake collidable walls). The orbit's view 1 is from behind the background and useless.
  - liver_s4: not a meaningful reproduction - a near-static ellipse; both needle holders end in free space, no
    suture, no contact; it matches because almost nothing moves (5 of 8 panels fine, 3 somewhat off: a lower band
    of 25-35 px and the sutured tip uncovered). The 4D stops 15-30 px short of the sutured tip in frames 20-90 and
    its area grows by about half between frames 30 and 140 (lower lobe excluded early, included late); the rest
    shape is a plump egg where the real lobe edge is a thin wedge.
  - chole_derot: not usable - the same upper-centre blob in all 8 panels (frame 0 fine, 3 somewhat off, 4 clearly
    wrong: 70, 105, 210, 245). The 4D tracks the gross pose but is angular at 120, extends 25-40 px under the tools
    at 40-50, runs up the grasper shaft at 140-150 and covers grasper and liver at 210-240. Instruments v03 fine
    in all 16 panels. r07b and r09 are indistinguishable (the depth correction changed nothing visible).
  - Could not judge: depth / penetration at intermediate frames (orbit = final state only), whether motion comes
    from contact or from anchors, volume change.
- State at the end of the 10-round budget:
  | clip | 4D reconstruction (IoU) | simulation IoU (4D's own) | motion explained | verdict |
  |---|---|---|---|---|
  | chole_a (grasp + puncture) | 0.83 / 0.84 / 0.92 | 0.72 (0.80) | +0.08 | silhouette-level reproduction; no deflation in 3D |
  | liver_s4 (knot tying) | 0.84 / 0.89 / 0.85 | 0.66 (0.83) | -0.10 | clean but static; suture and breathing not modelled |
  | chole_derot (pushing) | 0.79 / 0.80 / 0.79 | 0.46 (0.74) | -0.07 | not reproduced; tool-organ depth off by ~1 cm, neck not modelled |
  Answer to the experiment's question (can organ dynamics be modelled agentically in 3D from text + video): the 4D
  reconstruction, yes, without hand prompts or anatomical templates; forward simulation from the instruments, only
  where an instrument holds the organ (and then at a third of the amplitude); not where it pushes, prods or pulls
  through structures that have no model (neck / pedicle, suture) or where the driver is not an instrument (breathing,
  suction).

## after r10: every SAM 3 object reconstructed in 3D (user request, 2026-10-08)
- Request: "你应该确保所有sam3 seg出来的部位都被3D重建了". Inventory at the end of r10 (SAM 3 objects -> 3D model):
  chole_a 7 of 8 (missing: neck_pedicle), chole_derot 5 of 6 (missing: neck_pedicle), liver_s4 6 of 7 (missing: left
  triangular ligament), lung_mln 0 of 10 (segmentation only). The three missing ones are moving structures the organ
  agent's selection skips as 'tubular' / 'thin membrane' (left to tube / membrane agents that do not exist); liver,
  diaphragm and omentum were reconstructed only as labelled regions of the fused background surface and not shown
  as objects.
- Generic volumetric fit for the skipped structures (t2s.organ `--mask "<SAM object>"`, settings v20 = v15's without
  the background term and without the v16-v18 experiments; one process each, ~1.9 GB, 8-13 min):
  | clip | object | 4D IoU (bands) | rest size mm, volume | inverted tets 4D | note |
  |---|---|---|---|---|---|
  | chole_a | neck_pedicle | 0.71 / 0.79 / 0.82 | 60 x 53 x 11.5, 11.3 ml | - | a flat sheet-like blob beside the gallbladder |
  | chole_derot | neck_pedicle | 0.56 / 0.52 / 0.45 | 22 x 11 x 4.3, 0.21 ml | 4 (20 frames) | poor: the mask is a thin stalk until ~frame 165 and a bulb after; one rest shape cannot be both; mesh torn-looking from 150 on |
  | liver_s4 | left triangular ligament | 0.72 / 0.68 / 0.81 | 15.6 x 7.3 x 6.6, 0.31 ml | 0 | a membrane approximated by a small solid |
  They are reconstructed and shown; they are NOT part of the simulation (assemble is still run with the primary
  organ only), and a tube / membrane is not well described by a tet blob.
- Page: t2s/export_viewer.py rewritten around an object list (every SAM 3 object of the clip -> kind organ / part /
  body / surface / instrument / none, its model and its IoU with the mask); background faces grouped by label so
  that liver / diaphragm / omentum are objects; site/v2/page.html is now the page's own template (no longer patched
  from site/viewer), with an object list (per-object show / hide), a "colour by object" mode and a coverage table
  generated by site/v2/build_page.py. Coverage: chole_a 8 / 8, liver_s4 7 / 7, chole_derot 6 / 6 = 21 / 21.
  The other session's r09 / r10 results are in the page text. Published as version 3 of the artifact.
- Still open: lung_mln's 10 objects. Its shot C2 (frames 174-239, 5.3 s; Harmonic, STORZ grasper, lymph nodes and
  fat, dissection bed, white cord) is now its own clip `lung_c2` (t2s.data.CLIPS, t2s.seg3.subclip from lung_mln
  v03); the geometry step is running. The other 8 shots (1-6 s each) are not processed.
- lung_c2 (shot C2 of lung_mln, 66 frames, 12.5 fps), same modules as the other clips, one process at a time:
  geometry (t2s.geom: depth prep 97 s at ~10-11 GB resident - the heaviest step of the day, run alone; sift
  multi-view: 7 keyframes, 890 flow + 36 SIFT correspondences, none >= 10 keyframes apart, reprojection 35.8 -> 4.4
  px, f 450 px; shaft-ruler error 5.3 mm median) -> instruments v01 (40 s; no ultrasonic-scalpel type in the library:
  the Harmonic is fitted with the grasper model; IoU 0.83 / 0.80; STORZ grasper size scale 0.89) -> organ v20
  ('mediastinal lymph nodes and fat', the spec's primary: 4D IoU 0.70 / 0.60 / 0.86, 22 x 9 x 8 mm, 0.63 ml, 0
  inverted tets; 184 s) -> background v09 (dissection bed and white cord = labelled regions of the surface, no
  bodies; surface depth error 1.3 mm median) -> export. 5 / 5 SAM 3 objects of the shot have a 3D model. recon_check:
  the STORZ grasper's shaft overlaps the fat block's model in 6 of 14 frames (max 4.3 mm) - it is holding it; the
  Harmonic in 1 of 11 (1.1 mm). No simulation. Weaker than the other clips (short shot, little static tissue).
- Coverage now: 26 / 26 objects in the four reconstructed clips. NOT covered: the other 8 shots of lung_mln (1-6 s
  each); 5 of the lung video's 10 SAM 3 objects (steel grasper, suction, mediastinal pleura, the two dissection beds
  of shot D) appear only there and still have masks only.
- Page: lung_c2 is the fourth scene; artifact version 4.

