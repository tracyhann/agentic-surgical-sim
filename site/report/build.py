"""Collect the report's media and numbers from outputs/ into site/report/ (media/ + data.json).

  .venv/bin/python site/report/build.py          # everything (re-renders the long clip)
  .venv/bin/python site/report/build.py 4d       # only the per-tissue 4D study (outputs/iter/)
The page itself (index.html) is written by hand and reads nothing at run time: numbers in data.json are pasted into
it after each rebuild, so the page renders complete without scripts.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
SITE = Path(__file__).resolve().parent
MEDIA = SITE / 'media'
sys.path.insert(0, str(ROOT))


def reencode(src, dst, scale=1.0, quality=6, poster=None, poster_at=0.5):
    r = imageio.get_reader(src)
    fps = r.get_meta_data()['fps']
    frames = [f if scale == 1.0 else cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in r]
    frames = [f[:f.shape[0] // 2 * 2, :f.shape[1] // 2 * 2] for f in frames]
    imageio.mimsave(dst, frames, fps=fps, macro_block_size=1, quality=quality, output_params=['-movflags', '+faststart'])
    if poster:
        imageio.imwrite(poster, frames[int(poster_at * (len(frames) - 1))], quality=82)
    return dst


def metrics(clip):
    s = json.loads((OUT / clip / 'compare' / 'summary.json').read_text())
    keys = ('outline_iou', 'depth_err_mm_monocular', 'depth_err_mm_stereo', 'track_err_px')
    return {c: dict(sim={k: v['sim'].get(k) for k in keys}, static={k: v['static'].get(k) for k in keys},
                    n_vertices=v['organ']['n_organ_vertices'], n_tets=v['organ']['n_tets'],
                    max_disp_mm=v['organ']['organ_max_disp_mm'], ts_ms=v['organ']['params']['ts'] * 1000)
            for c, v in s.items()}


ITER = OUT / 'iter'
FINAL = 'r22'                     # the round whose renders the page shows
TISSUE_SHOWN = dict(gallbladder='v12', membrane='v08', ducts='v10', interaction='v06', backdrop='v08')


def tiles(sheet, frames=(0, 80, 160, 240), cols=4):
    """Crop the keyframe tiles (every 10th frame, `cols` per row) of a tissue track's sheet.jpg."""
    img = cv2.imread(str(sheet))
    h, w = img.shape[0] // 7, img.shape[1] // cols
    return [img[(k // 10) // cols * h:((k // 10) // cols + 1) * h, (k // 10) % cols * w:((k // 10) % cols + 1) * w] for k in frames]


def label(img, text, scale=0.5):
    img = img.copy()
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def study4d():
    import numpy as np
    rnd = ITER / 'rounds' / FINAL
    # tissue models: keyframes 0 / 80 / 160 / 240 of each track's overlay, one row per tissue
    rows = [np.concatenate(tiles(ITER / 'tissues' / t / TISSUE_SHOWN[t] / 'sheet.jpg'), 1) for t in ('gallbladder', 'membrane', 'ducts')]
    cv2.imwrite(str(MEDIA / 'r4d_tissues.jpg'), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 82])
    for t in ('gallbladder', 'membrane'):
        shutil.copy(ITER / 'tissues' / t / TISSUE_SHOWN[t] / 'views3d.jpg', MEDIA / f'r4d_{t}_3d.jpg')
    # simulation vs video, and the counterfactual grid (video frame times, half size tiles)
    reencode(rnd / 'render=1_mode=sim' / 'compare.mp4', MEDIA / 'r4d_compare.mp4', scale=0.75, poster=MEDIA / 'r4d_compare_poster.jpg', poster_at=0.62)
    shutil.copy(rnd / 'render=1_mode=sim' / 'render_sheet.jpg', MEDIA / 'r4d_render_sheet.jpg')
    orbit = cv2.imread(str(rnd / 'render=1_mode=sim' / 'orbit.jpg'))
    cv2.imwrite(str(MEDIA / 'r4d_orbit.jpg'), cv2.resize(orbit, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
    from r2s import source
    from r2s.config import Clip
    video = source.frames(Clip('chole_a'))
    cells = [('video', None), ('simulation, same actions', 'render=1_mode=sim'), ('replay of the 4D', 'render=1_mode=recon'),
             ('grasper lifted +10 mm', 'render=1_cf_lift_mm=10'), ('probe 5 mm deeper', 'render=1_cf_probe_mm=5'),
             ('probe withdrawn', 'render=1_cf_no_probe=1')]
    scopes = {c: imageio.mimread(rnd / c / 'scope.mp4', memtest=False) for _, c in cells if c}
    dt = 0.05
    frames = []
    for k in range(len(video)):
        tl = []
        for name, c in cells:
            img = video[k] if c is None else scopes[c][min(int(round(k / 25 / dt)), len(scopes[c]) - 1)]
            tl.append(label(cv2.resize(np.asarray(img), (320, 180), interpolation=cv2.INTER_AREA), name, 0.42))
        frames.append(np.concatenate([np.concatenate(tl[:3], 1), np.concatenate(tl[3:], 1)], 0))
    imageio.mimsave(MEDIA / 'r4d_counterfactual.mp4', frames, fps=25, macro_block_size=1, quality=6, output_params=['-movflags', '+faststart'])
    imageio.imwrite(MEDIA / 'r4d_counterfactual_poster.jpg', frames[180], quality=82)
    out = {}
    for _, c in cells:
        if c:
            m = json.loads((rnd / c / 'metrics.json').read_text())
            e = m['eval']
            out[c] = dict(tissues=m['tissues'], photometric=m['photometric'],
                          gb={k: e['gallbladder'].get(k) for k in ('sim_iou', 'sim_iou_2d', 'sim_iou_bands', 'split', 'inverted_tets_max', 'probe_push', 'volume_ml', 'free_motion_explained')},
                          sheet={k: e['membrane'].get(k) for k in ('sim_iou', 'sim_iou_2d', 'sim_iou_bands', 'split')},
                          ducts={k: e['ducts'].get(k) for k in ('sim_iou', 'recon_iou')}, forces={k: v for k, v in e['instruments'].items() if 'force' in k})
    return out


def main():
    shutil.rmtree(MEDIA, ignore_errors=True)   # generated; rebuilt from outputs/ every time
    MEDIA.mkdir()
    data = {}
    # 1. long clip: shot A + shot B, measured organ
    long_raw = OUT / 'long' / 'chole_23-41s_measured.mp4'
    long_raw.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'r2s.longvideo', str(long_raw), 'chole_a:measured:Shot A', 'chole_b:measured:Shot B'], cwd=ROOT, check=True)
    reencode(long_raw, MEDIA / 'long.mp4', scale=0.75, poster=MEDIA / 'long_poster.jpg', poster_at=0.3)
    # 2. organ conditions on shot A (and the 6 s window)
    for clip in ('chole_a', 'chole_6s', 'chole_b'):   # EndoNeRF prostatectomy left out of the report for now
        if (OUT / clip / 'compare' / 'conditions.mp4').exists():
            reencode(OUT / clip / 'compare' / 'conditions.mp4', MEDIA / f'{clip}_conditions.mp4', poster=MEDIA / f'{clip}_conditions_poster.jpg')
            shutil.copy(OUT / clip / 'compare' / 'conditions.jpg', MEDIA / f'{clip}_conditions.jpg')
            data[clip] = metrics(clip)
        if (OUT / clip / 'compare' / 'organs_rest.jpg').exists():
            shutil.copy(OUT / clip / 'compare' / 'organs_rest.jpg', MEDIA / f'{clip}_organs_rest.jpg')
        scene = OUT / clip / 'scene' / 'scene.json'
        if scene.exists():
            data.setdefault(clip, {})
            data[f'{clip}_scene'] = json.loads(scene.read_text())
            data[f'{clip}_metric'] = json.loads((OUT / clip / 'prep' / 'metric.json').read_text())
    # 3. multi-view follow-up: shot A, measured organ, single view vs wide-baseline geometries
    mv = OUT / 'experiments' / 'wide_baseline_2026-10-07'
    raw = mv / 'shot_a_measured_geometries.mp4'
    if not raw.exists():
        from r2s import render, source
        from r2s.config import Clip
        render.video_variants(['chole_a', 'chole_a+sift', 'chole_a+vggt_tracks'], 'measured', source.frames(Clip('chole_a')), raw,
                              ['single view (report)', 'multi-view: BA + SIFT', 'multi-view: BA + VGGT tracks'])
    reencode(raw, MEDIA / 'multiview_shot_a.mp4', scale=0.75, poster=MEDIA / 'multiview_shot_a_poster.jpg', poster_at=0.8)
    for c in ('chole_a', 'chole_b'):
        data[f'wide_baseline_{c}'] = json.loads((mv / c / 'alignment.json').read_text())
    # 4. template fitted to several keyframes
    shutil.copy(OUT / 'experiments' / 'template_keyframes_2026-10-07' / 'templates_rest.jpg', MEDIA / 'template_keyframes_rest.jpg')
    # 5. per-tissue models + 4D simulation study (outputs/iter/, LOG.md)
    data['r4d'] = study4d()
    (SITE / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for p in sorted(MEDIA.iterdir()):
        print(f'{p.stat().st_size / 1e6:6.1f} MB  {p.name}')


def main_4d():
    """Only the 4D study's media and numbers (the rest of media/ and data.json is kept)."""
    data = json.loads((SITE / 'data.json').read_text())
    data['r4d'] = study4d()
    (SITE / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for p in sorted(MEDIA.glob('r4d_*')):
        print(f'{p.stat().st_size / 1e6:6.1f} MB  {p.name}')


if __name__ == '__main__':
    main_4d() if sys.argv[1:] == ['4d'] else main()
