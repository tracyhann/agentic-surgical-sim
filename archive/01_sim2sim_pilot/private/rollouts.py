"""Data engine: replay a package's policy.plan() on displaced layouts of its own scene and record rollouts.

  python rollouts.py <instance> <package_dir> <out_dir> [--video]
Layouts: the target (and, for bead_cup, also the object) translated by 2/4/6/8/10 mm along four directions that
are defined from the package's own camera (image-right and image-depth projected onto the ground plane), so
different reconstructions are perturbed in the same visual directions.
Each rollout is saved as <out_dir>/<layout>.npz (t, object pose, tcp, actions, success) [+ .mp4].
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco

PRIV = Path(__file__).resolve().parent
sys.path.insert(0, str(PRIV))
import evaluate as E  # noqa: E402
S = E.S

MAGS = (0.002, 0.004, 0.006, 0.008, 0.010)


def directions(cam):
    R = S.cam_axes(cam['pos'], cam['lookat'])
    right = R[0] * [1, 1, 0]
    depth = R[2] * [1, 1, 0]
    right, depth = right / np.linalg.norm(right), depth / np.linalg.norm(depth)
    return {'right': right, 'left': -right, 'far': depth, 'near': -depth}


def layouts(inst):
    movers = ['target'] + (['object'] if inst == 'bead_cup' else [])
    return [(m, d, s) for m in movers for d in ('right', 'left', 'far', 'near') for s in MAGS]


def run_layout(inst, pkg, mover, dname, mag, video=False):
    proto = S.load_protocol(pkg)
    body = proto['roles'][mover][0]
    off = directions(proto['camera'])[dname] * mag

    def edit(spec):
        b = spec.body(body)
        b.pos = list(np.asarray(b.pos) + off)
    _, _, model, data = S.build(pkg, edit=edit)
    a0 = np.load(Path(pkg) / proto['actions'].get('path', 'actions.npy'))[0]
    S.reset(model, data, a0)
    try:
        acts = np.asarray(S.load_policy(pkg)(model, data), float)
        bad = S.check_actions(acts, 'plan')
        if bad:
            return dict(ok=False, error=bad[0])
        rec = S.execute(model, data, acts, [proto['roles']['object'][0]],
                        render=S.make_renderer(model, proto['camera']) if video else None)
    except Exception as e:
        return dict(ok=False, error=f'{type(e).__name__}: {e}')
    ok = E.success(model, data, proto, inst)
    return dict(ok=ok, rec=rec, actions=acts)


def main(inst, pkg, out, video=False):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mover, d, m in layouts(inst):
        tag = f'{mover}_{d}_{int(m * 1000)}mm'
        r = run_layout(inst, pkg, mover, d, m, video=video)
        rows.append(dict(layout=tag, mover=mover, direction=d, mm=int(m * 1000), success=bool(r['ok']), error=r.get('error')))
        if 'rec' in r:
            rec = r['rec']
            obj = list(rec['pose'])[0]
            np.savez_compressed(out / f'{tag}.npz', t=rec['t'], obj_pose=rec['pose'][obj], tcp=rec['tcp'],
                                actions=r['actions'], success=r['ok'])
            if video and rec['frames']:
                import imageio.v2 as imageio
                imageio.mimsave(out / f'{tag}.mp4', rec['frames'], fps=20, macro_block_size=1)
        print(f"{tag:28s} {'OK ' if r['ok'] else 'FAIL'} {r.get('error') or ''}", flush=True)
    summ = dict(instance=inst, package=str(pkg), n=len(rows), success_rate=float(np.mean([r['success'] for r in rows])),
                by_mm={mm: float(np.mean([r['success'] for r in rows if r['mm'] == mm])) for mm in sorted({r['mm'] for r in rows})},
                rows=rows)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != 'rows'}, indent=1))
    return summ


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), video='--video' in sys.argv)
