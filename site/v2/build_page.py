"""Build site/v2/index.html = page.html (the interactive 3D viewer for the scenes in site/v2/data/, written by
t2s/export_viewer.py) with results.html (comparison videos, numbers, round log) inserted at <!--RESULTS-->; the
table of every SAM 3 object and its 3D model (<!--COVERAGE--> in results.html) is generated from the exported scenes.

    python site/v2/build_page.py
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIND = dict(organ='主器官，体模型，逐帧形变', part='体模型，逐帧形变', body='凸体，静态', surface='背景表面的一块，静态', instrument='刚体，逐帧位姿',
            none='没有')


def coverage():
    rows, tot, done = [], 0, 0
    scenes = json.loads((HERE / 'data' / 'index.json').read_text())
    n_scenes = len(scenes)
    for e in scenes:
        sc = json.loads((HERE / 'data' / e['name'] / 'scene.json').read_text())
        obs = sc.get('objects', [])
        tot += len(obs)
        done += sum(o['kind'] != 'none' for o in obs)
        for i, o in enumerate(obs):
            first = f'<td rowspan="{len(obs)}">{html.escape(sc["title"])}</td>' if i == 0 else ''
            iou = '–' if o.get('iou') is None else f'{o["iou"]:.2f}'
            st = '<span class="st no">没有</span>' if o['kind'] == 'none' else KIND[o['kind']]
            rows.append(f'<tr>{first}<td>{html.escape(o["label"])}<br><span class="sam">{html.escape(o["name"])}</span></td>'
                        f'<td>{st}</td><td class="n">{iou}</td></tr>')
    head = '<thead><tr><th>片段</th><th>SAM 3 对象</th><th>三维模型</th><th>和掩码的重合 IoU</th></tr></thead>'
    return (f'<div class="ledger"><table>{head}<tbody>{"".join(rows)}</tbody></table>'
            f'<p class="tcap">做了三维重建的 {n_scenes} 个片段里，{done} / {tot} 个 SAM 3 对象有自己的三维模型。IoU 是模型按内镜相机投影后和 SAM 3 掩码的重合；'
            f'背景表面上的区域没有单独算。</p></div>')


page = (HERE / 'page.html').read_text()
assert page.count('<!--RESULTS-->') == 1
res = (HERE / 'results.html').read_text().replace('<section>', '<section class="res">')
if '<!--COVERAGE-->' in res:
    res = res.replace('<!--COVERAGE-->', coverage())
(HERE / 'index.html').write_text(page.replace('<!--RESULTS-->', res))
print('index.html', len(page) + len(res))
