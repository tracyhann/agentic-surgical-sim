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
