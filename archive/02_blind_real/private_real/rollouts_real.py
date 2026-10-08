"""Data engine on a real-video reconstruction: replay the package's policy.plan() on displaced layouts of its scene.

  python rollouts_real.py <instance> [--video]
rosma_handoff: the destination post moved 2/4/6/8/10 mm along image-right/left/far/near (projected on the ground);
               success = sleeve ends around the moved post (evaluate_real.sleeve_success).
chole_sweep:   the tissue body moved the same way; reports whether the rollout stays stable and how far the tissue
               is displaced (no task predicate exists for this clip).
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

PRIV = Path(__file__).resolve().parent
ROOT = PRIV.parent
sys.path.insert(0, str(PRIV))
from evaluate_real import load_tool, sleeve_success  # noqa: E402

MAGS = (0.002, 0.004, 0.006, 0.008, 0.010)


def main(inst, video=False):
    S = load_tool(inst)
    pkg = ROOT / 'runs_real' / inst / 'out'
    proto = S.load_protocol(pkg)
    R = S.cam_axes(proto['camera']['pos'], proto['camera']['lookat'])
    right, depth = R[0] * [1, 1, 0], R[2] * [1, 1, 0]
    dirs = {'right': right / np.linalg.norm(right), 'left': -right / np.linalg.norm(right),
            'far': depth / np.linalg.norm(depth), 'near': -depth / np.linalg.norm(depth)}
    mover = proto['roles']['target'][0] if inst == 'rosma_handoff' else proto['roles']['object'][0]
    a0 = np.load(pkg / proto['actions'].get('path', 'actions.npy'))[0]
    out = ROOT / 'results_real' / inst / 'rollouts'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, v in dirs.items():
        for m in MAGS:
            tag = f'{mover}_{d}_{int(m * 1000)}mm'

            def edit(spec):
                b = spec.body(mover)
                b.pos = list(np.asarray(b.pos) + v * m)
            row = dict(layout=tag, direction=d, mm=int(m * 1000))
            try:
                _, _, model, data = S.build(pkg, edit=edit)
                S.reset(model, data, a0)
                acts = np.asarray(S.load_policy(pkg)(model, data), float)
                bad = S.check_actions(acts, 'plan')
                if bad:
                    raise ValueError(bad[0])
                bodies = [b for b in proto['roles']['object'] + proto['roles'].get('target', [])
                          if S._id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
                rec = S.execute(model, data, acts, bodies,
                                render=S.make_renderer(model, proto['camera']) if video and m == MAGS[-1] and d in ('right', 'far') else None)
                if inst == 'rosma_handoff':
                    row['success'], row['detail'] = sleeve_success(model, data, proto, S)
                else:
                    p = rec['pose'][mover][:, :3]
                    row['success'] = bool(np.all(np.isfinite(p)))
                    row['tissue_max_disp_mm'] = round(float(np.linalg.norm(p - p[0], axis=1).max()) * 1000, 1)
                np.savez_compressed(out / f'{tag}.npz', t=rec['t'], actions=acts,
                                    **{f'tcp_{k}': v_ for k, v_ in rec['tcp'].items()},
                                    **{f'pose_{k}': v_ for k, v_ in rec['pose'].items()})
                if rec['frames']:
                    imageio.mimsave(out / f'{tag}.mp4', rec['frames'], fps=20, macro_block_size=1, quality=7)
            except Exception as e:
                row['success'], row['error'] = False, f'{type(e).__name__}: {e}'[:200]
            rows.append(row)
            print(tag, 'OK' if row['success'] else 'FAIL', row.get('error', ''), flush=True)
    summ = dict(instance=inst, n=len(rows), success_rate=float(np.mean([r['success'] for r in rows])),
                by_mm={mm: float(np.mean([r['success'] for r in rows if r['mm'] == mm])) for mm in sorted({r['mm'] for r in rows})},
                rows=rows)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], video='--video' in sys.argv)
