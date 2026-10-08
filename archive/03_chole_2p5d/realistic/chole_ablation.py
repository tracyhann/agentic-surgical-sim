"""Does the simulated deformation matter? Re-run the clip with every tissue / strand vertex frozen (same camera
motion, same instruments) and compare the per-region track errors and image metrics with the full model.

  python chole_ablation.py      -> chole/ablation.json
"""
import json, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chole_recon as C  # noqa: E402
import chole_sim as S  # noqa: E402
import chole_metrics as M  # noqa: E402

OUT = C.OUT


def frozen_xml(fn):
    def wrap(*a, **k):
        x = fn(*a, **k)
        return re.sub(r'(<body name="(?:fv|fb|rv)\d+" pos="[^"]+") gravcomp="1">(?:<joint[^>]*/>){3}<inertial[^>]*/></body>', r'\1/>', x)
    return wrap


def main():
    full_info = S.run(render=True)
    full = M.evaluate(write=True)
    for f in ('flex_traj.npy', 'sim_frames.npy'):
        shutil.copy(OUT / f, OUT / f.replace('.npy', '_full.npy'))
    orig = S.scene_xml
    S.scene_xml = frozen_xml(orig)
    try:
        S.run(render=True, compare=False, tag='_frozen')
    finally:
        S.scene_xml = orig
    shutil.copy(OUT / 'flex_traj_frozen.npy', OUT / 'flex_traj.npy')
    shutil.copy(OUT / 'sim_frames_frozen.npy', OUT / 'sim_frames.npy')
    frozen = M.evaluate(write=False)
    for f in ('flex_traj.npy', 'sim_frames.npy'):                      # restore the full model's outputs
        shutil.copy(OUT / f.replace('.npy', '_full.npy'), OUT / f)
    rows = {}
    for reg in full['tissue_tracks']:
        rows[reg] = dict(full=full['tissue_tracks'][reg]['sim_err_px'], frozen=frozen['tissue_tracks'][reg]['sim_err_px'],
                         static=full['tissue_tracks'][reg]['static_baseline_px'], n=full['tissue_tracks'][reg]['n_tracks'])
    rows['image'] = dict(full=[full['image']['ssim_sim'], full['image']['psnr_sim']],
                         frozen=[frozen['image']['ssim_sim'], frozen['image']['psnr_sim']],
                         static=[full['image']['ssim_static_frame0'], full['image']['psnr_static_frame0']])
    rows['strand_iou'] = dict(full=full['image']['strand_iou_sim'], frozen=frozen['image']['strand_iou_sim'],
                              static=full['image']['strand_iou_static_frame0'])
    res = dict(rows=rows, sim=full_info)
    (OUT / 'ablation.json').write_text(json.dumps(res, indent=1))
    for k, v in rows.items():
        print(f'{k:16s} {v}')


if __name__ == '__main__':
    main()
