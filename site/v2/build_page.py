"""Build site/v2/index.html = the interactive 3D viewer (the v1 viewer's code, site/viewer/index.html, patched for the
v2 scenes in site/v2/data/) followed by the v2 results (comparison videos, numbers, round log) from results.html.

    python site/v2/build_page.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / 'viewer' / 'index.html').read_text()


def rep(a, b, count=1):
    global src
    assert src.count(a) >= 1, a[:80]
    src = src.replace(a, b, count)


rep('<title>胆囊重建 3D 查看器</title>', '<title>文字引导的器官 4D 重建</title>')
# header
h0, h1 = src.index('<header>'), src.index('</header>') + len('</header>')
src = src[:h0] + '''<header>
    <div class="eyebrow">medical-agentic-sim · v2 · 第 5 轮 / 共 10 轮 · 2026-10-08</div>
    <h1>文字引导的器官 4D 重建</h1>
    <p class="lead">新方法的进行中记录。每段手术视频先读它配套的文字，再由一组 agent 分工：定位并分割器官和器械、求相机和深度、从基本几何体拟合器官、给器械建刚体模型、重建可碰撞的背景，最后合成物理场景。<b>下面的三维场景就是重建本身</b>：拖动旋转，滚轮缩放，右键平移；切到内镜视角可以和真实视频左右对比。</p>
    <p class="foot"><a href="https://claude.ai/artifact/VT6cePMUES9nUifUZ8cv2y" target="_blank" rel="noopener">上一版方法（v1）的报告</a> · <a href="https://github.com/tracyhann/agentic-surgical-sim/tree/v2-text2sim" target="_blank" rel="noopener">代码和逐轮记录（分支 v2-text2sim）</a></p>
  </header>''' + src[h1:]
# condition buttons
c0 = src.index('<div class="seg" id="segCond">')
c1 = src.index('</div>', c0) + len('</div>')
src = src[:c0] + '''<div class="seg" id="segCond">
        <button data-v="recon4d" aria-pressed="true">4D 重建</button><button data-v="sim4d" aria-pressed="false">物理仿真</button>
      </div>''' + src[c1:]
src = re.sub(r'<legend>器官体[^<]*</legend>', '<legend>显示</legend>', src, count=1)
rep('<fieldset><legend>显示</legend>\n      <div class="seg" id="segMode">', '<fieldset><legend>外观</legend>\n      <div class="seg" id="segMode">') if '<fieldset><legend>显示</legend>\n      <div class="seg" id="segMode">' in src else None
src = re.sub(r"var COND_NAMES = \{[^}]*\};", "var COND_NAMES = { recon4d: '4D 重建', sim4d: '物理仿真' };", src, count=1, flags=re.S)
rep("cond: 'measured'", "cond: 'recon4d'")
rep("载入重建数据…", "载入重建数据…")
# metrics block -> generic table
m0 = src.index('<div class="metrics">')
m1 = src.index('</div>', src.index('id="mcap"')) + len('</div>')
m1 = src.index('</div>', m1) + len('</div>')
src = src[:m0] + '''<div class="metrics">
    <table>
      <thead><tr><th>结果</th><th>轮廓 IoU ↑</th><th>不切深度 ↑</th><th>运动解释率 ↑</th><th>翻转四面体 ↓</th><th>最大位移</th></tr></thead>
      <tbody id="mbody"></tbody>
    </table>
    <div class="mcap" id="mcap">物理仿真一行来自最近一轮整合；轮廓 IoU 是仿真的器官投影和 SAM 3 掩码的重合（每 10 帧一次）。没有仿真的场景只有 4D 重建一行。</div>
  </div>''' + src[m1:]
f0 = src.index('  function metricsTable() {')
f1 = src.index('  function setMode(m) {')
src = src[:f0] + '''  function metricsTable() {
    var conds = Object.keys(D.conditions);
    var f = function (v, d) { return v == null ? '–' : (typeof v === 'number' ? v.toFixed(d) : v); };
    $('mbody').innerHTML = conds.map(function (c) {
      var m = D.conditions[c].metrics || {};
      return '<tr class="' + (c === S.cond ? 'sel' : '') + '"><td>' + COND_NAMES[c] + '</td><td class="n">' + f(m.sim_iou != null ? m.sim_iou : m.recon_iou, 2) + '</td><td class="n">' + f(m.sim_iou_2d, 2) +
        '</td><td class="n">' + (m.motion_explained == null ? '–' : (m.motion_explained > 0 ? '+' : '') + m.motion_explained.toFixed(2)) + '</td><td class="n">' + f(m.inverted_tets_max, 0) + '</td><td class="n">' + D.conditions[c].max_disp_mm.toFixed(1) + ' mm</td></tr>';
    }).join('');
    var NOTES = { chole_a: '这一段的物理仿真是第 5 轮最好的一次：没有非法穿插，但胆囊跟着器械动的幅度只有重建的三分之一左右。',
      liver_s4: '这一段的物理仿真是坏的（上千个四面体翻转，持针器穿进肝里），放在这里是为了能直接看到它怎么坏。4D 重建是可用的。',
      chole_derot: '这一段还是旧的相机和深度（尺度偏近约 35%），所以器械偏粗、胆囊偏大；新几何上的重建和仿真还没做。' };
    $('mcap').textContent = (NOTES[D.name] || '') + ' 轮廓 IoU 是仿真的器官投影和 SAM 3 掩码的重合（每 10 帧一次）；"不切深度"是同一个数但不按深度裁剪被遮挡的部分。4D 重建一行没有仿真指标。';
  }

''' + src[f1:]
# notes -> v2 legend + results
n0 = src.index('<section class="notes">')
n1 = src.index('</section>', n0) + len('</section>')
src = src[:n0] + '''<section class="notes">
    <div>
      <h2>画面里是什么</h2>
      <ul>
        <li><span class="chip" style="background:#D9B840"></span><span><b>器官</b>：从椭球出发、多视角多帧拟合出来的四面体体（胆囊或肝左叶尖端）。"4D 重建"播放逐帧拟合的形状，"物理仿真"播放 MuJoCo 算出来的形状。</span></li>
        <li><span class="chip" style="background:#9E6BB8"></span><span><b>背景里的小实体</b>：血块、条索、纱布、没认出来的器官，各自建成封闭的凸体。</span></li>
        <li><span class="chip" style="background:#9C8A86"></span><span><b>背景表面</b>：所有帧融合出来的静态表面，贴视频纹理，仿真里参与碰撞。</span></li>
        <li><span class="chip" style="background:#C9CFD1"></span><span><b>器械</b>：刚体模型，每把绕一个固定的穿刺点（红点）运动，位姿逐帧拟合。</span></li>
        <li><span class="chip" style="background:#7DB7BA"></span><span><b>内镜</b>：青色线框是内镜视野，位置和朝向来自多视角联合求解。</span></li>
      </ul>
    </div>
    <div>
      <h2>怎么用</h2>
      <ul>
        <li><span></span><span>拖动旋转，滚轮或双指缩放，右键或双指拖动平移。</span></li>
        <li><span></span><span>"几何形状"下最容易看清三维形状；"形变热图"显示相对第一帧的位移。</span></li>
        <li><span></span><span>内镜视角下拖动中间的竖线，左边真实视频，右边重建。</span></li>
        <li><span></span><span><span class="key">空格</span> 播放 / 暂停，<span class="key">←</span> <span class="key">→</span> 逐帧，<span class="key">V</span> 切换视角。</span></li>
      </ul>
    </div>
  </section>

''' + (HERE / 'results.html').read_text().replace('<section>', '<section class="res">') + src[n1:]
# v2 display: lit background bodies (so the convex pieces read as 3D), their own toggle, wipe label follows the layer
rep("M.ribbon = { photo: new THREE.MeshBasicMaterial({ map: tR, side: ds })", "M.ribbon = { photo: new THREE.MeshStandardMaterial({ map: tR, roughness: 0.6, side: ds })")
rep("if (S.mode === 'shape') g.computeVertexNormals();", "if (S.mode !== 'heat') g.computeVertexNormals();")
rep('<span class="wipe-label r">仿真</span>', '<span class="wipe-label r" id="wipeR">重建</span>')
rep("var prev = S.cond; S.cond = c;", "var prev = S.cond; S.cond = c; $('wipeR').textContent = COND_NAMES[c];")
rep('<label><input type="checkbox" id="cBack" checked> 背景组织</label>', '<label><input type="checkbox" id="cBack" checked> 背景表面</label>\n        <label><input type="checkbox" id="cBodies" checked> 背景实体</label>')
rep("(G.ribbons || []).forEach(function (r) { r.wire.visible = wire; });", "var bodies = $('cBodies').checked; (G.ribbons || []).forEach(function (r) { r.mesh.visible = bodies; r.wire.visible = wire && bodies; });")
rep("['cWire', 'cGizmo', 'cBack']", "['cWire', 'cGizmo', 'cBack', 'cBodies']")
rep("这个器官体的数据没有载入", "这一层的数据没有载入")
rep("camFree.position.fromArray(D.view.position); controls.target.fromArray(D.view.target); controls.update();",
    "if (D.view.up) { camFree.up.fromArray(D.view.up); controls.dispose(); controls = new THREE.OrbitControls(camFree, canvas); controls.enableDamping = true; controls.dampingFactor = 0.12; controls.zoomSpeed = 0.9; controls.minDistance = 0.01; controls.maxDistance = 0.8; controls.enabled = S.view === 'free'; }\n    camFree.position.fromArray(D.view.position); controls.target.fromArray(D.view.target); controls.update();")
rep('--good: #9CCB8A;', '--good: #9CCB8A; --warn: #E2B55A;')
# extra styles for the results part (same dark tokens)
rep('</style>', '''  .res { display: grid; gap: 14px; border-top: 1px solid var(--rule); padding-top: 20px; margin-top: 10px; }
  .sec-head { display: grid; gap: 6px; }
  footer { border-top: 1px solid var(--rule); padding-top: 14px; font-size: 12.5px; color: var(--muted); display: grid; gap: 4px; }
  .res h2 { font: 650 22px/1.25 var(--display); font-stretch: 87%; margin: 0; text-wrap: balance; }
  .res h3 { font: 600 15px/1.4 var(--body); margin: 0; }
  .res p { margin: 0; max-width: 74ch; color: var(--muted); }
  .res p b, .res li b { color: var(--ink); font-weight: 500; }
  .tag { font: 500 12px/1 var(--mono); color: var(--accent); letter-spacing: .06em; }
  .res figure { margin: 0; display: grid; gap: 8px; }
  .res video { width: 100%; max-width: 100%; height: auto; display: block; background: #000; border-radius: 4px; border: 1px solid var(--rule); }
  .res figcaption { font-size: 13px; color: var(--muted); max-width: 86ch; }
  .answers { display: grid; gap: 10px; }
  .answer { background: var(--surface); border: 1px solid var(--rule); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 12px 16px; display: grid; gap: 4px; }
  .res ul.plain { margin: 0; padding-left: 18px; display: grid; gap: 6px; max-width: 78ch; color: var(--muted); list-style: disc; }
  .res ul.plain li { display: list-item; }
  .ledger { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; overflow-x: auto; }
  .ledger table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  .ledger th, .ledger td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
  .ledger th { font: 500 11px/1.3 var(--mono); color: var(--muted); letter-spacing: .04em; white-space: nowrap; }
  .ledger tr:last-child td { border-bottom: 0; }
  .ledger td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .tcap { font-size: 12.5px; color: var(--muted); padding: 8px 12px 10px; border-top: 1px solid var(--rule); margin: 0; max-width: none !important; }
  .st { font: 500 11.5px/1 var(--mono); padding: 3px 6px; border-radius: 3px; white-space: nowrap; border: 1px solid currentColor; }
  .st.ok { color: var(--good); } .st.part { color: var(--warn); } .st.no { color: var(--accent); }
</style>''')
(HERE / 'index.html').write_text('<meta charset="utf-8">\n' + src)
print('index.html', len(src))
