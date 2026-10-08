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
