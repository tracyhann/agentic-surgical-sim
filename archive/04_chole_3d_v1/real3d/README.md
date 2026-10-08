# real3d: measured-3D reconstruction of the cholecystectomy clip

Builds on `realistic/` (simulator `chole_sim.py`, helpers `chole_recon.py`); replaces the hand-made depth prior
with measured depth and segmentation. Models in `models/` (Depth Anything V2 Small, SAM 2.1 hiera-tiny).

    python real3d/segment.py        # SAM 2 masks, 6 objects x 151 frames  -> real3d/sam_masks.npz
    python real3d/metric_depth.py   # DAv2 relative depth -> metres via 5 mm shaft widths  -> real3d/metric.npz
    python real3d/recon3d.py        # geometry, textures, instrument actions, observed bed motion  -> real3d/chole3d/
    python real3d/eval3d.py         # runs the frozen control too  -> chole3d/eval3d.json
    python real3d/views.py [video]  # scope + orbit 35/70 deg renders of a finished run

The main simulation run (writes `chole3d/flex_traj.npy`, `sim_frames.npy`, ...):

    import chole_recon as C, chole_sim as S        # with realistic/ on sys.path
    C.OUT = S.OUT = Path('real3d/chole3d'); S.TS = 0.000125
    S.run(k_gb=0.5, young=300.0)

Current result (visible-region metrics, every 5th frame): gallbladder depth error 4.36 mm
(frozen 5.17, frame 0 throughout 5.09); outline IoU 0.760 (frozen 0.760, frame 0 0.754).

What comes from where:
- depth: DAv2 per frame, `1/z = a*d + b_k` (one slope, smoothed per-frame offsets) fitted on both shafts
- gallbladder: amodal extent (parts hidden by probe / strands at frame 0 taken from early frames), silhouette
  thickness, tetrahedral flex; E = 300 Pa (1500 Pa needs TS 6.25e-5 and scores slightly worse)
- global gallbladder motion (up to ~9 mm, from out of view): frame-to-frame DIS flow, translation only, drives the
  back-spring anchors (`bed_disp`); `S.MOVE_BED = False` turns it off
- grasper follows the measured neck apex; probe port constrained near the tips' depth

Interactive viewer (three.js): `python real3d/export_viewer.py real3d/viewer` writes the data next to
`real3d/viewer/index.html` (serve the folder over http; published as a claude.ai artifact).
