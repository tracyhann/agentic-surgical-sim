# Contract for the per-tissue models (r2s/tissue/*.py -> outputs/iter/tissues/<tissue>/vNN/)

Goal: shot A of the cholecystectomy video (`chole_a`, 251 frames, 25 fps, 640x360) reconstructed tissue by tissue in
3D and over time (4D), each tissue a template + fit on the multi-view camera geometry, so that the scene can be
simulated (MuJoCo) and every piece checked against the video.

## Inputs (read-only)
- `r2s/views.py`: `V = views.load('chole_a', 'sift')` gives frames, SAM masks of the clip objects, per-frame cameras
  (R, f, pos; multi-view keyframe BA + SIFT), metric depth per frame, instrument tool tips. World frame = MuJoCo
  (metres, z up). `views.segment_object(V, prompts)` segments a new object with SAM 2.1.
- `r2s/quality.py`: shared metrics (silhouette IoU, boundary F, depth residual, chamfer, mesh health, stretch) and
  `contact_sheet` overlays. Extend metrics inside your own module if you need more; do not edit shared files.
- Templates: `data/templates/bodyparts3d/{gallbladder,liver}.stl` (mm, BodyParts3D, CC BY-SA 2.1 JP).
  Existing helpers you may import: `r2s.organ` (ShapeFit, remesh, to_tets, degenerate), `r2s.multiview`, `r2s.camera`.

## Outputs of one version vNN of a tissue
`outputs/iter/tissues/<tissue>/vNN/`
- `model.npz`: `rest_verts` (N, 3) world m; `faces` (M, 3); optional `tets` (T, 4) on the same vertices;
  `verts4d` (251, N, 3) the shape fitted in every frame (4D reconstruction; NaN-free, temporally smooth);
  optional `attach_idx` + `attach_to` (which vertices are fixed to what: 'backdrop', 'gallbladder', 'grasper', ...);
  optional extra arrays documented in NOTES.md.
- `quality.json`: at least silhouette IoU and boundary F (per keyframe + summary) of `verts4d`, depth residual (mm),
  mesh health; plus whatever tissue-specific metrics you add (state what each means).
- `sheet.jpg` (`quality.contact_sheet` of verts4d over keyframes) and `views3d.jpg` (rest shape, 3 angles).
- `NOTES.md`: what the version does, the iterations you went through (change -> metrics -> what you saw -> next),
  known problems.
- Masks you made: `masks.npz` + the prompts in NOTES.md. Scratch files: `vNN/work/` (may be deleted later).

## Rules
- Write code only in `r2s/tissue/<your tissue>.py` (plus a `__main__` entry: `python -m r2s.tissue.<tissue> vNN`).
  Do not edit other files in r2s/, the clip configs, or anything under outputs/ outside your tissue folder.
- Do not run `python -m r2s.batch` or `python -m r2s <clip> ...` stages (they overwrite shared outputs).
- Other tracks run at the same time on this machine (18 cores, one GPU via MPS): keep jobs lean.
- Iterate: build, check with several metrics and by looking at the contact sheet, fix, repeat. Record every iteration.
- Report honestly: what works, what does not, numbers.
