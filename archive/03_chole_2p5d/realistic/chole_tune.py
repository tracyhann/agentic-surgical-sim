"""Grid search of the tissue parameters against the real feature tracks (gallbladder + fold), no rendering."""
import itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chole_sim as S, chole_metrics as M  # noqa: E402

rows = []
for k_gb, k_fold, young, fr in itertools.product((0.003, 0.01, 0.04), (4.0, 8.0), (800.0, 1500.0, 2500.0), (0.5,)):
    try:
        info = S.run(k_gb, k_fold, young, fr, render=False)
        t = M.evaluate(write=False, image=False)['tissue_tracks']
        score = (t['gallbladder']['sim_err_px'] * t['gallbladder']['n_tracks'] + t['fold']['sim_err_px'] * t['fold']['n_tracks']) / \
                (t['gallbladder']['n_tracks'] + t['fold']['n_tracks'])
        rows.append(dict(k_gb=k_gb, k_fold=k_fold, young=young, friction=fr, score=round(score, 2), gb=t['gallbladder']['sim_err_px'],
                         fold=t['fold']['sim_err_px'], gb_disp=t['gallbladder']['sim_max_disp_px'], max_mm=info['tissue_max_disp_mm']))
    except Exception as e:
        rows.append(dict(k_gb=k_gb, k_fold=k_fold, young=young, error=str(e)[:80]))
    print(rows[-1], flush=True)
rows = sorted([r for r in rows if 'score' in r], key=lambda r: r['score'])
Path(S.OUT / 'tune.json').write_text(json.dumps(rows, indent=1))
print('BEST', rows[0])
