# EndoNeRF sample data — pulling_soft_tissues

da Vinci robotic prostatectomy, in vivo, static endoscope, 63 frames, 640x512.
`images/` left RGB, `depth/` stereo depth (STTR-light; 8-bit, millimetres), `masks/` tool masks,
`poses_bounds.npy` (identical poses; focal 569.47 px). `_drive_listing.json`: the Google Drive file ids.
Only the pulling sequence was fetched (the right images and the cutting sequence were not).

Source: Wang et al., "Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery",
MICCAI 2022, https://github.com/med-air/EndoNeRF (sample dataset link in the README).
No license is stated for the sample data: research use only, cite the paper. Downloaded 2026-10-07.
