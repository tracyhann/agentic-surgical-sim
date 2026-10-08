"""New interactions in the reconstructed scene (not in the video): the data-engine use of the real-to-sim.

  python chole_variants.py
  _retract2x : the grasper retracts the neck twice as far as in the video (same timing)
  _press     : the probe leaves its sweep, moves over the gallbladder body and indents it ~6 mm, then lifts off
Each is rendered photoreal from the same moving endoscope as the original clip.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chole_recon as C  # noqa: E402
import chole_sim as S  # noqa: E402

OUT = C.OUT


def main():
    acts = np.load(OUT / 'actions.npy')
    meta = json.loads((OUT / 'build_meta.json').read_text())['instruments']
    g = np.load(OUT / 'geometry.npz')
    n = len(acts)
    res = {}
    # 1) stronger retraction: double the grasper TCP displacement from its start
    Pg, hg = np.array(meta['grasper_left']['rcm']), meta['grasper_left']['heading']
    T = np.array([C.K.fk(Pg, *a[0:3], hg) for a in acts])
    T2 = T[0] + 2.0 * (T - T[0])
    a1 = acts.copy()
    a1[:, 0:5] = C.joint_targets(Pg, T2, hg, 0.0)
    res['retract2x'] = S.run(actions=a1, tag='_retract2x', compare=False)
    # 2) probe presses the gallbladder body: hover -> descend over (210, 270) -> indent 6 mm -> hold -> lift
    Pp, hp = np.array(meta['probe_right']['rcm']), meta['probe_right']['heading']
    Tp = np.array([C.K.fk(Pp, *a[5:8], hp) for a in acts])
    u, v = 215, 270
    surf = C.unproject(u, v, g['z'][v, u])
    ray = (surf - C.CAM_POS) / np.linalg.norm(surf - C.CAM_POS)
    above, deep = surf - 0.012 * ray, surf + 0.006 * ray
    t = np.arange(n) * 0.05
    key_t = [0.0, 1.0, 2.2, 3.4, 4.4, 5.2, t[-1]]
    key_p = [Tp[0], above, above, deep, deep, above, above]
    Tn = np.stack([np.interp(t, key_t, [p[k] for p in key_p]) for k in range(3)], 1)
    a2 = acts.copy()
    a2[:, 5:10] = C.joint_targets(Pp, Tn, hp, 0.0)
    res['press'] = S.run(actions=a2, tag='_press', press=0.0065, compare=False)
    # side-by-side of the original reconstruction and the two variants
    import imageio.v2 as imageio
    import cv2
    vids = [np.load(OUT / f'sim_frames{tag}.npy') for tag in ('', '_retract2x', '_press')]
    m = min(len(x) for x in vids)
    lab = ['reconstruction (as in video)', 'variant: 2x retraction', 'variant: probe presses gallbladder']
    comp = []
    for k in range(m):
        tiles = []
        for x, l in zip(vids, lab):
            f = x[k].copy()
            cv2.putText(f, l, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            tiles.append(f)
        comp.append(np.concatenate(tiles, 1))
    imageio.mimsave(OUT / 'variants.mp4', comp, fps=20, macro_block_size=1, quality=8)
    imageio.imwrite(OUT / 'variants_070.png', comp[int(0.7 * (m - 1))])
    (OUT / 'variants.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in ('tissue_max_disp_mm', 'tissue_p95_disp_mm', 'probe_retargeted_steps')} for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    main()
