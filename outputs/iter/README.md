# 4D reconstruction and simulation study, shot A (2026-10-07)

Why: the user saw that the simulated scene is wrong: in the video the left grasper lifts a thin peritoneal sheet into
a tent and the right probe pushes into the gallbladder neck; the reconstruction had no sheet, grasped the gallbladder
body, and clamped the probe to 1.5 mm of penetration. Brief: model each tissue separately (template + fit,
multi-view), check every result with several metrics, fix, simulate the dynamics in 3D/4D, iterate 30 rounds,
keep complete records.

Layout
- `CONTRACT.md`   interface of the per-tissue models
- `LOG.md`        round by round: hypothesis, change, metrics, visual check, decision
- `tissues/<tissue>/vNN/`   per-tissue model versions (gallbladder, membrane, ducts, backdrop, interaction)
- `rounds/rNN/`   integrated scene of round NN: metrics.json, contact sheets, renders, notes

Code: `r2s/views.py` (data access), `r2s/quality.py` (metrics), `r2s/tissue/` (per-tissue models),
`r2s/scene4d.py` (assembly + dynamics + evaluation of a round).
