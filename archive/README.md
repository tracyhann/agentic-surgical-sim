# archive — earlier phases (frozen)

Kept as they were when each phase ended. Their scripts use the paths of the time (most expect to run from the
project root with folders such as `realistic/`, `real3d/`, `runs_real/` at top level), so they do not run from here
without adjusting paths. Everything current lives in `r2s/` (code), `outputs/` (results) and `site/` (artifacts).

| folder | phase | what it holds |
|---|---|---|
| `01_sim2sim_pilot/` | Video2World-style pilot, 2026-10-06 | public instrument model + `surgsim` tool, hidden reference scenes and evaluator (`private/`), two blind Claude Code runs (`runs/`), scores and rollouts (`results/`); `older_snapshot_surg_v2w/` is an earlier copy of the same pilot |
| `02_blind_real/` | blind agents on real video | ROSMA hand-off and the cholecystectomy clip handed to blind sessions (`private_real/`, `real/`, `runs_real/`, `results_real/`) |
| `03_chole_2p5d/` | first hand-built reconstruction | hand-made depth prior, 2.5D flex sheet, strands, ablation, counterfactual variants (`realistic/`) |
| `04_chole_3d_v1/` | measured 3D, 6 s clip | Depth Anything V2 + SAM 2 reconstruction of the 6 s clip (`real3d/`), the first viewer export (`viewer_export/`) |

`MUJOCO_LOG.TXT` is MuJoCo's warning log from those runs.
