"""After the blind real-video run: score, render page videos, run the displaced-target data engine.

  python postprocess_real.py X01  ->  results/real_X01/{result.json, eval_compare.mp4, rollout_*.mp4, feasibility.mp4}
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, cv2, imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'real'))
import evaluate_real as E  # noqa: E402


def shrink(src, dst, width):
    r = imageio.get_reader(src)
    fps = r.get_meta_data().get('fps', 15)
    frames = []
    for f in r:
        h = int(round(f.shape[0] * width / f.shape[1] / 2) * 2)
        frames.append(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA))
    imageio.mimsave(dst, frames, fps=fps, quality=8, macro_block_size=1)


def main(trial):
    run = ROOT / 'runs' / f'real_{trial}'
    pkg = run / 'out'
    out = ROOT / 'results' / f'real_{trial}'
    out.mkdir(parents=True, exist_ok=True)
    agent = json.loads((run / 'agent_run.json').read_text())
    res = E.score(pkg, trial, *agent['clip'], render=True)
    if res.get('build'):
        res['rollouts'] = E.rollouts(pkg)
        shrink(run / 'eval_compare.mp4', out / 'eval_compare.mp4', 1536)
        proto, _, model, data = E.S2.S1.build(pkg)
        for dn, mm in (('right', 10), ('far', 10)):
            R = E.S2.S1.cam_axes(proto['camera']['pos'], proto['camera']['lookat'])
            dv = {'right': R[0], 'far': R[2]}[dn] * [1, 1, 0]
            dv /= np.linalg.norm(dv)
            tgt = proto['roles']['target'][0]

            def edit(spec, dv=dv, mm=mm):
                b = spec.body(tgt)
                b.pos = list(np.asarray(b.pos) + dv * mm / 1000)
            try:
                r = E.run(pkg, render=True, edit=edit, policy=True)
                imageio.mimsave(out / f'rollout_target_{dn}_{mm}mm.mp4', [cv2.resize(f, (768, 576)) for f in r['rec']['frames']],
                                fps=20, quality=8, macro_block_size=1)
                res.setdefault('rollout_videos', {})[f'target_{dn}_{mm}mm'] = r['placed']
            except Exception as e:
                res.setdefault('rollout_videos', {})[f'target_{dn}_{mm}mm'] = f'error: {e}'
    feas = ROOT / 'private' / 'real_ref' / 'feasibility' / 'source' / 'video.mp4'
    if feas.exists():
        shrink(feas, out / 'feasibility.mp4', 768)
    res['agent'] = agent
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != 'rollouts'}, indent=1, default=str))
    if 'rollouts' in res:
        print('rollouts placed_rate', res['rollouts']['placed_rate'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'X01')
