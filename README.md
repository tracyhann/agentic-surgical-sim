# medical-agentic-sim

Real surgical video → physics simulation. A laparoscopic clip is reconstructed as a MuJoCo scene: measured tissue
geometry with the video's own texture, a deformable organ, and the instruments replaying the motion recovered from the
video. The scene can then be re-simulated with other instrument motions (data for surgical world models), and every
reconstruction is scored against the video it came from.

Current best method (per-tissue models + 4D simulation, 2026-10-08): **[PIPELINE.md](PIPELINE.md)**; round log
`outputs/iter/LOG.md`.

```
r2s/            the pipeline (config-driven; see r2s/README.md)
  clips/        one JSON per clip: source, camera, SAM prompts, instruments
data/
  videos/       source videos with SOURCE.md (Commons cholecystectomy, lung VATS lobectomy, liver diagnostic
                laparoscopy; EndoNeRF prostatectomy, ROSMA)
  templates/    organ templates (BodyParts3D, CC BY-SA)
models/         Depth Anything V2 Small, SAM 2.1 hiera-tiny (Hugging Face format), VGGT-1B (multi-view experiment)
third_party/    vggt code (not installed; see third_party/README.md)
outputs/<clip>/ everything generated per clip: prep/, scene/, one folder per organ condition, compare/
  iter/         per-tissue 4D study: tissues/<tissue>/vNN (models + quality), rounds/rNN (simulations), LOG.md
  experiments/  side experiments with their own README (multiview, wide_baseline, template_keyframes, anatomy_fix; 2026-10-07)
  variants/     pipeline runs of a clip with config overrides, <clip>+<variant> (multi-view geometries)
  logs/         stage logs of the batch runner
site/
  report/       the report artifact (claude.ai/artifact/VT6cePMUES9nUifUZ8cv2y)
  viewer/       the interactive 3D viewer artifact (claude.ai/artifact/B4V3kZTgvRuJdTsaSPLHD7); data/ from r2s.export
archive/        earlier phases, frozen (sim2sim pilot, blind agents, 2.5D and first 3D reconstructions)
```

Setup: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

Run a clip end to end (prep → scene → four organ conditions → simulation → metrics → renders):

```
.venv/bin/python -m r2s.batch chole_a
.venv/bin/python -m r2s.shape_study endonerf_pulling vesicle right_seminal_vesicle
.venv/bin/python -m r2s.longvideo outputs/long/chole_23-41s_measured.mp4 chole_a:measured:"Shot A" chole_b:measured:"Shot B"
.venv/bin/python site/report/build.py          # collect report media and numbers into site/report/
.venv/bin/python -m r2s.export site/viewer/data chole_a chole_a+sift chole_b   # EndoNeRF left out of the site for now
```
