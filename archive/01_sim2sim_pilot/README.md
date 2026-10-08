# surg_v2w — a surgical Video2World pilot (MuJoCo, sim-rendered source videos)

A blind coding agent sees only an endoscope video, a brief and the instrument model, and must deliver a MuJoCo scene,
a control stream and a state-reading `policy.py`. It is scored against a hidden reference, and its policy is then
replayed on displaced layouts to generate rollouts (the "data engine").

```
instrument/        public laparoscopic grasper (MJCF includes, kinematics.py, README)
tools/surgsim.py   public validate / replay tool the agent uses
runs/<inst>/       task/ (what the agent saw), out/ (its package), agent_log.jsonl, agent_run.json
private/           hidden side: make_reference.py (GT scenes + scripted demo + source video), evaluate.py,
                   rollouts.py, launch.py (blind Claude Code driver), postprocess.py, rollout_gap.py, gt/
results/           results.json, eval_compare.mp4 per instance, *_strip.png, rollouts (agent and reference)
```

Setup: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

Rerun (from `private/`, with `../.venv/bin/python`):
1. `make_reference.py` — GT scenes, scripted demos, source videos into `runs/<inst>/task/`
2. `rollouts.py <inst> gt/<inst>/package ../results/rollouts_reference/<inst>` — reference data engine
3. `launch.py <inst> --minutes 45` — one blind Claude Code session (needs the `claude` CLI)
4. `postprocess.py` then `rollout_gap.py`

`runs/*/task/brief.md` and `runs/*/agent_log.jsonl` record the paths of the original pilot run (2026-10-06).

For further blind runs, move `private/` out of this tree first: the agent's shell can list parent directories,
and the log audit in `launch.py` is the only isolation.
