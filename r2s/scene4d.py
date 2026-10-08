"""One round of the 4D study (outputs/iter/): assemble the per-tissue models into one MuJoCo scene, drive the
instruments, simulate, and evaluate against the video and against each tissue's own 4D reconstruction.

    python -m r2s.scene4d rNN [gallbladder=vNN membrane=vNN ducts=vNN backdrop=vNN interaction=vNN] [param=value ...]

Tissues are loaded from outputs/iter/tissues/<tissue>/<version>/model.npz (CONTRACT.md); the latest version is used
unless one is named ('none' leaves a tissue out). Simulated: gallbladder (tetrahedra if given, else shell), membrane
and ducts (shells); backdrop is static and visual only. Attachments come from each model's attach_idx / attach_to:
  backdrop     spring to the rest position (k_attach)
  gallbladder  connect to the nearest gallbladder vertex (offset kept)
  grasper      held by the grasper from frame 0, pulled into its jaws over grasp_ramp seconds
Instruments follow the interaction model's tracks (or the old fit); the probe is not clamped: it pushes into tissue
through contact. Outputs in outputs/iter/rounds/rNN/: metrics.json, sheet_sim.jpg, sheet_recon.jpg, orbit.jpg,
traj.npz, scene.xml, run.json.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
from .config import OUTPUTS, save_json
from . import views, quality as Q, instruments as INS
from .sim import COLORS

ITER = OUTPUTS / 'iter'
TISSUES = ('gallbladder', 'membrane', 'ducts', 'backdrop')
SIMULATED = ('gallbladder', 'membrane', 'ducts')
PARAMS = dict(ts=6.25e-5, dt=0.05, settle=0.3, vertex_mass=6e-5, friction=0.3, damping=0.002, solref=0.01,
              merge_mm=0.0, start='rest', bed_drive='recon', k_drive=20.0, mem_contact=1, mem_base='recon', sliver_deg=15.0, duct_end='recon', duct_nodes='glue_mid2', cams='refined', fibre_skip=1, tex_dir='', mode='sim', render=1, base_link_tc=0.05, probe_depth_mm=2.0, cf_lift_mm=0.0, cf_probe_mm=0.0, cf_no_probe=0, cf_release_frame=-1, cable_k=8.0, cable_damping=0.05, probe_ramp=1.0, mem_k=10.0, gb_shell_k=5.0,
              gb_young=1200.0, gb_poisson=0.45, gb_k_bed=0.5,
              mem_young=600.0, mem_thick=0.0006, duct_young=900.0, duct_thick=0.0015,
              k_attach=20.0, grasp_ramp=0.25, grasp_r=0.004, grasp_n=60, probe_tip_len=0.02, grasp_target='tcp', noise_mm=0.0, noise_seed=0, probe_solref=0.002, gb_radius=0.0004, duct_snap=0, mem_rest='edges', mem_pre=1.0)
CAMS = dict(refined=str(ITER / 'tissues/backdrop/v08/cams_refined.npz'), clip=None)
MASK_KEYS = dict(gallbladder=['body'], membrane=['membrane_clean', 'membrane'], ducts=[('duct', 'strands')])
COLOR = dict(gallbladder=(230, 200, 40), membrane=(120, 220, 255), ducts=(200, 90, 220), backdrop=(200, 170, 160))
TISSUE_RGBA = dict(ducts='0.86 0.72 0.76 1')       # rendered colour of untextured tissues (pale pink cords)


# ---------------------------------------------------------------- inputs
def latest(tissue):
    d = ITER / 'tissues' / tissue
    vs = sorted(p for p in d.glob('v*') if (p / 'model.npz').exists()) if d.exists() else []
    return vs[-1].name if vs else None


# versions used when a run names none (a track's newest folder can be work in progress: r15 picked up gallbladder
# v12 while its track was still writing it); 'latest' = newest folder with a model.npz
ADOPTED = dict(gallbladder='v12', membrane='v09', ducts='v11', backdrop='v08', interaction='v06')


def load_tissue(tissue, version=None):
    version = version or ADOPTED.get(tissue) or latest(tissue)
    if version == 'latest':
        version = latest(tissue)
    if version in (None, 'none'):
        return None
    d = Path(version) if '/' in version else ITER / 'tissues' / tissue / version
    z = dict(np.load(d / 'model.npz', allow_pickle=True))
    z['version'], z['dir'] = version, d
    return z


def tissue_masks(T, V, tissue):
    """The tissue track's own masks if it made them, else the clip mask of the same name (None if neither)."""
    p = T['dir'] / 'masks.npz'
    if p.exists():
        z = np.load(p)
        keys = next(([k] if isinstance(k, str) else list(k) for k in MASK_KEYS.get(tissue, []) if
                     all(kk in z.files for kk in ([k] if isinstance(k, str) else k))), None)
        if keys is None:
            keys = ['masks'] if 'masks' in z.files else [z.files[0]]
        m = np.zeros((V.n, V.H, V.W), bool)
        for k in keys:
            mk = z[k]
            if mk.dtype == np.uint8 and mk.shape[-1] != V.W:
                mk = np.unpackbits(mk, axis=-1)[..., :V.W]
            m |= mk.astype(bool)
        return m
    name = {'ducts': 'strand'}.get(tissue, tissue)
    return V.mask(name) if name in V.names else None


def tool_tracks(V, version=None):
    """Per instrument: tcp (n, 3) tool-centre point the joints aim at, tip (n, 3) physical tip, rcm (3,), heading,
    jaw (n,) from the interaction model, else the old fit in V.tools (whose 'tip' is its tool-centre point)."""
    I = load_tissue('interaction', version)
    out = {}
    for name, t in V.tools.items():
        tr = dict(tcp=np.asarray(t['tip'], float), tip=np.asarray(t['tip'], float), rcm=np.asarray(t['rcm'], float),
                  jaw=np.asarray(t['jaw'], float), heading=None, mask=t['mask'], holds=t['holds'])
        if I is not None and f'{name}_tcp' in I:
            tr.update(tcp=np.asarray(I[f'{name}_tcp'], float), tip=np.asarray(I[f'{name}_tip'], float),
                      rcm=np.asarray(I[f'{name}_rcm'], float), heading=float(I[f'{name}_heading']))
            if t['holds']:
                tr['jaw'] = np.zeros(V.n)                  # closed on the sheet throughout
            if 'grasp_point' in I and t['holds']:
                tr['grasp_point'] = np.asarray(I['grasp_point'], float)
        out[name] = tr
    return out, (I['version'] if I is not None else 'old fit')


# ---------------------------------------------------------------- scene
def _vbody(name, p, k, mass):
    if k is None:
        return f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"/>'
    j = ''.join(f'<joint type="slide" axis="{a}" stiffness="{k}"/>' for a in ('1 0 0', '0 1 0', '0 0 1'))
    return (f'<body name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" gravcomp="1">{j}'
            f'<inertial pos="0 0 0" mass="{mass}" diaginertia="1e-9 1e-9 1e-9"/></body>')


def attachments(T):
    """{vertex index: target} from attach_idx / attach_to (attach_to: one string, or one per index)."""
    if 'attach_idx' not in T:
        return {}
    idx = np.asarray(T['attach_idx']).ravel().astype(int)
    to = np.asarray(T['attach_to'])
    to = [str(to)] * len(idx) if to.ndim == 0 else [str(s) for s in to.ravel()]
    return dict(zip(idx.tolist(), to))


def merge_short(X, F, dmin):
    """Shell for simulation: vertices clustered on a dmin voxel grid (MuJoCo's in-plane shell term returns NaN on
    sliver / sub-0.5 mm elements; a grid, unlike edge collapsing, cannot chain a row of close vertices into one).
    Returns X (n', 3), F (m', 3), inv (n,): original vertex -> merged vertex."""
    X = np.asarray(X, float)
    key = np.floor((X - X.min(0)) / dmin).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Xn = np.array([X[inv == k].mean(0) for k in range(inv.max() + 1)])
    Fn = inv[np.asarray(F)]
    Fn = Fn[(Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])]
    Fn = np.unique(np.sort(Fn, 1), axis=0, return_index=True)[1]
    Fn = (inv[np.asarray(F)])[(inv[np.asarray(F)][:, 0] != inv[np.asarray(F)][:, 1]) & (inv[np.asarray(F)][:, 1] != inv[np.asarray(F)][:, 2])
                             & (inv[np.asarray(F)][:, 0] != inv[np.asarray(F)][:, 2])][Fn]
    used = np.unique(Fn)
    remap = -np.ones(len(Xn), int)
    remap[used] = np.arange(len(used))
    keep = remap[inv]
    for i in np.nonzero(keep < 0)[0]:          # a vertex whose triangles all collapsed: follow its nearest kept one
        keep[i] = remap[used[np.argmin(np.linalg.norm(Xn[used] - X[i], axis=1))]]
    return Xn[used], remap[Fn], keep


def snap_cable(c, G, blend=3):
    """Integrator-side stop-gap for a duct start that floats off the gallbladder: move node 0 onto the gallbladder's
    nearest duct-attachment vertex, in the rest shape and in every 4D frame, blending the shift out over the first
    `blend` segments (the rest of the centreline is untouched)."""
    att = attachments(G)
    own = np.array([j for j, w in att.items() if w == 'ducts']) if att else np.arange(len(G['rest_verts']))
    j = int(own[np.argmin(np.linalg.norm(G['rest_verts'][own] - c['nodes'][0], axis=1))])
    w = np.clip(1 - np.arange(len(c['nodes'])) / blend, 0, 1)[:, None]
    c = dict(c)
    c['nodes'] = c['nodes'] + w * (G['rest_verts'][j] - c['nodes'][0])
    c['nodes4d'] = c['nodes4d'] + w[None] * (G['verts4d'][:, j] - c['nodes4d'][:, 0])[:, None]
    c['snapped_to'] = j
    return c


def sim_bodies(T, tissue, P, G=None):
    """The simulated representation of a tissue: list of parts dict(name, X (n, 3), F (elements), dim,
    att {local idx: target}, radius, inv (original vertex -> part vertex) or None)."""
    att = attachments(T)
    if 'tube_names' in T:                     # ducts: one cable per tube from its rest centreline
        from .tissue import ducts as D
        parts = []
        for c in D.cable_spec(T['version']):
            if P['duct_snap'] and G is not None and 'verts4d' in G:
                c = snap_cable(c, G)
            n = len(c['nodes'])
            at = c['attach']
            if isinstance(at, dict):
                ends = (at.get('start') or at.get('proximal'), at.get('end') or at.get('distal'))
            elif isinstance(at, (list, tuple)) and len(at) == 2:
                ends = tuple(at)
            else:
                ends = ('gallbladder', 'backdrop')
            a = {}
            if ends[0]:
                a[0] = str(ends[0])
            if ends[1]:
                a[n - 1] = str(ends[1])
            # nodes that follow the tube's own 4D (duct_nodes): 'end' distal only; 'ends' + proximal (no glue to
            # the simulated neck); 'mid' / 'mid2' + 1 / 2 interior nodes (ducts v10 end-motion test, greedy picks);
            # 'glue_mid' / 'glue_mid2': proximal glued to the gallbladder's neck tip, distal + 1 / 2 interior driven
            mode = P['duct_nodes']
            drv = {n - 1} if a.get(n - 1) == 'backdrop' else set()
            if mode in ('ends', 'mid', 'mid2'):
                drv.add(0)
                a.pop(0, None)
            duct = c['name'] == 'duct'
            if mode in ('mid', 'glue_mid'):
                drv.add(n // 2 if duct else n // 3)
            if mode in ('mid2', 'glue_mid2'):
                drv |= {n // 2, n // 4} if duct else {n // 3, 2 * n // 3}
            # chain edges plus every-second-node edges: the second set gives the cable bending resistance
            # (a 1D flex has stretch springs only); thin fibres can go without (fibre_skip=0: they just hang)
            E = [[i, i + 1] for i in range(n - 1)]
            if c['name'] == 'duct' or P['fibre_skip']:
                E += [[i, i + 2] for i in range(n - 2)]
            parts.append(dict(name=f"{tissue}_{c['name']}", X=c['nodes'], F=np.array(E),
                              dim=1, att=a, radius=float(np.mean(c['radius'])), inv=None, spec=c, driven=sorted(drv)))
        return parts
    if tissue == 'gallbladder' and 'tets' in T:
        return [dict(name=tissue, X=T['rest_verts'], F=T['tets'], dim=3, att=att, radius=P['gb_radius'], inv=None)]
    if P['merge_mm'] > 0:
        X, F, inv = merge_short(T['rest_verts'], T['faces'], P['merge_mm'] / 1000)
        a = {}
        for i, tgt in att.items():
            a.setdefault(int(inv[i]), tgt)
        return [dict(name=tissue, X=X, F=F, dim=2, att=a, radius=0.0004, inv=inv)]
    return [dict(name=tissue, X=T['rest_verts'], F=np.asarray(T['faces']), dim=2, att=att, radius=0.0004, inv=None)]


def bed_anchors(V, X, k=0, frac=0.45):
    """Fallback for the gallbladder: the back part (farthest from the scope in frame 0) rests on the liver bed."""
    _, z = V.project(X, k)
    return z > np.quantile(z, 1 - frac)


def prepare_textures(V, tissues, tex_dir):
    """Video textures: each simulated tissue takes the frame closest to its rest shape (its vertices projected there
    give the texture coordinates, top-down v as MuJoCo flexes expect); the backdrop keeps its fused texture."""
    import cv2
    tex_dir = Path(tex_dir)
    tex_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in ('gallbladder', 'membrane'):
        T = tissues.get(t)
        if T is None or 'verts4d' not in T:
            continue
        rec = np.asarray(T['verts4d'])
        ref = int(np.argmin(np.linalg.norm(rec - T['rest_verts'][None], axis=2).mean(1)))
        q, _ = V.project(rec[ref], ref)
        uv = np.c_[np.clip(q[:, 0] / V.W, 0, 1), np.clip(q[:, 1] / V.H, 0, 1)]
        f = tex_dir / f'tex_{t}.png'
        cv2.imwrite(str(f), cv2.cvtColor(V.frames[ref], cv2.COLOR_RGB2BGR))
        out[t] = (f, uv, ref)
    B = tissues.get('backdrop')
    if B is not None and 'uv' in B and (B['dir'] / 'texture.png').exists():
        f = tex_dir / 'tex_backdrop.png'
        import shutil
        shutil.copy(B['dir'] / 'texture.png', f)
        uv = np.asarray(B['uv'], float)
        out['backdrop'] = (f, np.c_[uv[:, 0], 1 - uv[:, 1]], None)
    return out


def tube_colors(V, T, step=10):
    """Median video colour (RGB 0-255) of the duct and of the strands masks, instrument pixels excluded."""
    z = np.load(T['dir'] / 'masks.npz')
    out = []
    for key in ('duct', 'strands'):
        m = z[key] if key in z.files else z[z.files[0]]
        if m.dtype == np.uint8 and m.shape[-1] != V.W:
            m = np.unpackbits(m, axis=-1)[..., :V.W]
        m = m.astype(bool)
        px = np.concatenate([V.frames[k][m[k] & ~V.occluders(k)] for k in range(0, V.n, step)])
        out.append(np.median(px, 0) if len(px) else np.array([220, 184, 194]))
    return out


def build_xml(V, tissues, tools, P):
    bodies, flexes, eqs, grasps = [], [], [], []
    parts = {}
    tex = prepare_textures(V, tissues, P['tex_dir']) if P['tex_dir'] else {}
    tex_assets = ''.join(f'<texture name="t_{t}" type="2d" file="{f}"/><material name="m_{t}" texture="t_{t}" emission="0.85" '
                         f'specular="{0.5 if t != "backdrop" else 0.05}" shininess="0.8"/>' for t, (f, _, _) in tex.items())
    for t in SIMULATED:
        T = tissues.get(t)
        if T is None:
            continue
        parts[t] = sim_bodies(T, t, P, tissues.get('gallbladder'))
        for part in parts[t]:
            pre, X, att = part['name'] + '_', part['X'], part['att']
            anchored = np.zeros(len(X), bool)
            if t == 'gallbladder' and not any(v == 'backdrop' for v in att.values()):
                anchored = bed_anchors(V, X)
            for i, p in enumerate(X):
                a = att.get(i)
                if t == 'membrane' and a == 'gallbladder' and P['mem_base'] in ('recon', 'both'):
                    bodies.append(_vbody(f'{pre}{i}', p, P['k_drive'], P['vertex_mass']))
                    continue
                if part['dim'] == 1 and i in part['driven']:   # cable nodes on the tube's 4D (duct_nodes), or a
                    k = None if P['duct_end'] == 'fixed' and a == 'backdrop' else P['k_drive']   # fixed end
                elif part['dim'] == 1:
                    k = 0
                elif t == 'gallbladder':
                    bed = a == 'backdrop' or anchored[i]                              # liver bed: soft springs,
                    k = (P['k_drive'] if P['bed_drive'] == 'recon' else P['gb_k_bed']) if bed else 0   # or driven ones
                else:
                    k = P['k_attach'] if a == 'backdrop' else 0
                bodies.append(_vbody(f'{pre}{i}', p, k, P['vertex_mass']))
            names = ' '.join(f'{pre}{i}' for i in range(len(X)))
            rgba = TISSUE_RGBA.get(t) if P['tex_dir'] and t in TISSUE_RGBA else ' '.join(f'{c / 255:.3f}' for c in COLOR[t]) + ' 1'
            elems = part['F']
            if part['dim'] == 2:                  # slivers make the shell's bending term explode: keep them out of
                elems = elems[min_angle(X, elems) >= P['sliver_deg']]   # the shell (their edges stay as springs)
            el = ' '.join(' '.join(map(str, e)) for e in elems)
            look = f'rgba="{rgba}"'
            if P['tex_dir'] and part['dim'] == 1:          # cables: the video's own colour of the tube (no texture
                rgb = tube_colors(V, T)[0 if part['name'].endswith('_duct') else 1] / 255   # coordinates)
                tex_assets += (f'<material name="m_{part["name"]}" rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1" '
                               f'emission="0.85" specular="0.5" shininess="0.8"/>')
                look = f'material="m_{part["name"]}"'
            if t in tex and part['inv'] is None and part['dim'] >= 2:
                uv = tex[t][1]
                look = f'material="m_{t}" texcoord="{" ".join(f"{a:.5f} {b:.5f}" for a, b in uv)}"'
            head = (f'<flex name="{part["name"]}" dim="{part["dim"]}" radius="{part["radius"]:.5f}" body="{names}" '
                    f'vertex="{" ".join("0 0 0" for _ in X)}" element="{el}" {look}>')
            if part['dim'] == 3:
                flexes.append(head + f'<elasticity young="{P["gb_young"]}" poisson="{P["gb_poisson"]}" damping="{P["damping"]}"/>'
                              f'<contact condim="3" friction="{P["friction"]}" solref="{P["solref"]} 1" margin="{0 if P["mem_contact"] else 0.002}" selfcollide="none"/></flex>')
            elif part['dim'] == 2:
                young = P['gb_young'] if t == 'gallbladder' else P['mem_young']
                thick = 0.003 if t == 'gallbladder' else P['mem_thick']
                # MuJoCo's in-plane shell term returns NaN on sliver elements: bending from the shell, in-plane
                # stretch from edge springs on a second (line) flex over the same vertices
                # the probe works under the sheet: the sheet collides with the gallbladder only (when enabled)
                aff, mg = (('1', '0') if P['mem_contact'] else ('0', '0')) if t == 'membrane' else ('2', '0.001')
                flexes.append(head + f'<elasticity young="{young}" poisson="0.4" thickness="{thick}" damping="{P["damping"]}" elastic2d="bend"/>'
                              f'<contact condim="3" friction="0.1" solref="{P["solref"]} 1" margin="{mg}" contype="4" conaffinity="{aff}" selfcollide="none"/></flex>')
                E = edges_of(part['F'])
                k_edge = P['gb_shell_k'] if t == 'gallbladder' else P['mem_k']
                flexes.append(f'<flex name="{part["name"]}_edges" dim="1" radius="0.0002" body="{names}" vertex="{" ".join("0 0 0" for _ in X)}" '
                              f'element="{" ".join(" ".join(map(str, e)) for e in E)}" rgba="0 0 0 0">'
                              f'<edge stiffness="{k_edge}" damping="{P["cable_damping"]}"/><contact contype="0" conaffinity="0"/></flex>')
            else:
                flexes.append(head + f'<edge stiffness="{P["cable_k"]}" damping="{P["cable_damping"]}"/>'
                              f'<contact condim="3" friction="0.1" solref="{P["solref"]} 1" margin="0.001" contype="4" conaffinity="2" selfcollide="none"/></flex>')
            for i, a in att.items():
                if a == 'gallbladder' and t == 'membrane' and P['mem_base'] == 'recon':
                    continue                                  # driven instead (base_driver)
                if a == 'gallbladder' and t == 'membrane' and P['mem_base'] == 'both' and 'gallbladder' in parts:
                    gp = parts['gallbladder'][0]              # driven AND softly linked: the lift reaches the wall
                    j2 = int(np.argmin(np.linalg.norm(gp['X'] - X[i], axis=1)))
                    eqs.append(f'<connect body1="{pre}{i}" body2="{gp["name"]}_{j2}" anchor="0 0 0" solref="{P["base_link_tc"]} 1"/>')
                    continue
                if a == 'gallbladder' and t != 'gallbladder' and 'gallbladder' in parts:
                    gp = parts['gallbladder'][0]
                    cand = np.arange(len(gp['X']))
                    if t == 'ducts':           # the gallbladder model's own duct-attachment vertices (its neck tip)
                        own = [j for j, w in gp['att'].items() if w == 'ducts']
                        cand = np.array(own) if own else cand
                    j2 = int(cand[np.argmin(np.linalg.norm(gp['X'][cand] - X[i], axis=1))])
                    eqs.append(f'<connect body1="{pre}{i}" body2="{gp["name"]}_{j2}" anchor="0 0 0" solref="{P["solref"]} 1"/>')
                elif a in ('grasper', 'grasper_left'):
                    grasps.append(f'{pre}{i}')
    # instruments
    ib, ia, ic, mats = [], [], [], []
    for ins in V.clip.instruments:
        name = ins['name']
        tr = tools[name]
        holds = bool(ins.get('holds'))
        shaft, jaw = COLORS[ins.get('color', 'bright')]
        b, a, c = INS.mjcf(name, ins['shaft_d'], shaft, jaw, holds)
        if not holds:          # a probe pushes with its distal shaft as well as its tip
            r = ins['shaft_d'] / 2
            b = b.replace(f'<site name="{name}_tcp"',
                          f'<geom name="{name}_tipcap" type="capsule" fromto="0 {-P["probe_tip_len"]:.4f} 0 0 0.016 0" size="{r:.5f}" '
                          f'rgba="{shaft}" material="{name}_shaft_mat" contype="2" conaffinity="1" friction="0.3" solref="{P["probe_solref"]} 1"/>'
                          f'<site name="{name}_tcp"', 1)
        heading = tr['heading'] if tr.get('heading') is not None else INS.heading_for(tr['rcm'], tr['tcp'])
        tr['heading'] = heading
        rcm = tr['rcm']
        ib.append(f'<body name="{name}_trocar" pos="{rcm[0]:.5f} {rcm[1]:.5f} {rcm[2]:.5f}" euler="0 0 {heading:.5f}">{b}</body>')
        ia.append(a)
        ic.append(c)
        mats.append(f'<material name="{name}_shaft_mat" specular="0.9" shininess="0.95" emission="0.3"/>'
                    f'<material name="{name}_jaw_mat" specular="1" shininess="0.95" emission="0.3"/>')
        if holds:
            eqs += [f'<connect name="{name}_grasp{i}" body1="world" body2="{name}_roll_link" anchor="0 0 0" active="false" solref="0.004 1"/>'
                    for i in range(P['grasp_n'])]
    # backdrop (static, visual)
    back = ''
    B = tissues.get('backdrop')
    if B is not None:
        Xb, Fb = backdrop_behind(B, tissues.get('gallbladder')), B['faces']
        tc = f' texcoord="{" ".join(f"{a:.5f} {b:.5f}" for a, b in tex["backdrop"][1])}"' if 'backdrop' in tex else ''
        back_asset = (f'<mesh name="backdrop" vertex="{" ".join(f"{v:.5f}" for v in Xb.ravel())}" '
                      f'face="{" ".join(map(str, np.asarray(Fb).ravel()))}"{tc}/>')
        look = 'material="m_backdrop"' if 'backdrop' in tex else f'rgba="{" ".join(f"{c / 255:.3f}" for c in COLOR["backdrop"])} 1"'
        back = f'<geom name="backdrop" type="mesh" mesh="backdrop" {look} contype="0" conaffinity="0"/>'
    else:
        back_asset = ''
    cam = V.cam
    q0 = cam.mj_quat(V.R[0])
    xml = f"""<mujoco model="r4d">
  <compiler angle="radian"/>
  <option timestep="{P['ts']}" integrator="Euler" gravity="0 0 -9.81"/>
  <visual><global offwidth="{cam.W}" offheight="{cam.H}"/><quality shadowsize="2048"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.4 0.4 0.4" specular="0 0 0"/></visual>
  <asset>{''.join(mats)}{tex_assets}{back_asset}</asset>
  <worldbody>
    <camera name="endo" pos="{V.pos[0][0]:.5f} {V.pos[0][1]:.5f} {V.pos[0][2]:.5f}" quat="{' '.join(f'{x:.6f}' for x in q0)}" fovy="{cam.fovy:.4f}"/>
    <camera name="orbit" pos="{V.pos[0][0] + 0.08:.4f} {V.pos[0][1]:.4f} {V.pos[0][2] + 0.03:.4f}"/>
    {back}
    {''.join(bodies)}
    {''.join(ib)}
  </worldbody>
  <deformable>{''.join(flexes)}</deformable>
  <equality>{''.join(eqs)}</equality>
  <actuator>{''.join(ia)}</actuator>
  <contact>{''.join(ic)}</contact>
</mujoco>"""
    return xml, grasps, parts


SLEW = np.array([0.08, 0.08, 0.015, 0.15, 0.007])   # max joint change per control step (probe re-enters fast)


def bed_vertices(T, V, X):
    """Gallbladder vertices resting on the liver bed: attached to 'backdrop' in the model, else the back part."""
    att = attachments(T)
    bed = np.array([att.get(i) == 'backdrop' for i in range(len(X))])
    return bed if bed.any() else bed_anchors(V, X)


def bed_driver(m, tissues, parts, P, V):
    """bed_drive='recon': the springs of the gallbladder's liver-bed vertices pull toward those vertices'
    reconstructed 4D positions (the bed moves as observed), instead of toward their rest positions. Returns
    drive(t) that updates the spring references, or None."""
    if P['bed_drive'] != 'recon' or 'gallbladder' not in parts:
        return None
    T = tissues['gallbladder']
    p = parts['gallbladder'][0]
    if 'verts4d' not in T or p['inv'] is not None:
        return None
    bed = np.nonzero(bed_vertices(T, V, p['X']))[0]
    adr = []
    for i in bed:
        b = m.body(f"{p['name']}_{i}")
        adr.append(m.jnt_qposadr[b.jntadr[0]] if b.jntnum[0] == 3 else -1)
    adr = np.array(adr)
    ok = adr >= 0
    bed, adr = bed[ok], adr[ok]
    disp = T['verts4d'][:, bed] - p['X'][bed][None]               # (frames, n_bed, 3)
    idx = (adr[:, None] + np.arange(3)[None]).ravel()

    def drive(t):
        f = min(t * V.fps, V.n - 1)
        k0 = int(f)
        k1 = min(k0 + 1, V.n - 1)
        w = f - k0
        m.qpos_spring[idx] = ((1 - w) * disp[k0] + w * disp[k1]).ravel()
    return drive


def base_driver(m, tissues, parts, P, V):
    """mem_base='recon': the sheet's base vertices (attached to the gallbladder) follow their reconstructed 4D
    positions through springs (the base slides as more of the sheet is lifted), instead of being glued."""
    if P['mem_base'] not in ('recon', 'both') or 'membrane' not in parts:
        return None
    T = tissues['membrane']
    p = parts['membrane'][0]
    if 'verts4d' not in T or p['inv'] is not None:
        return None
    base = np.array(sorted(i for i, a in p['att'].items() if a == 'gallbladder'))
    if not len(base):
        return None
    adr = np.array([m.jnt_qposadr[m.body(f"{p['name']}_{i}").jntadr[0]] for i in base])
    disp = T['verts4d'][:, base] - p['X'][base][None]
    idx = (adr[:, None] + np.arange(3)[None]).ravel()

    def drive(t):
        f = min(t * V.fps, V.n - 1)
        k0 = int(f)
        k1 = min(k0 + 1, V.n - 1)
        w = f - k0
        m.qpos_spring[idx] = ((1 - w) * disp[k0] + w * disp[k1]).ravel()
    return drive


def duct_end_driver(m, parts, P, V):
    """duct_end='recon': the driven nodes of every cable (part['driven']: the distal end, plus more with duct_nodes)
    follow their reconstructed 4D positions through springs."""
    if P['duct_end'] != 'recon' or 'ducts' not in parts:
        return None
    idx, disp = [], []
    for p in parts['ducts']:
        for i in p['driven']:
            b = m.body(f"{p['name']}_{i}")
            if b.jntnum[0] != 3:
                continue
            a = m.jnt_qposadr[b.jntadr[0]]
            idx += [a, a + 1, a + 2]
            disp.append(p['spec']['nodes4d'][:, i] - p['X'][i][None])
    if not idx:
        return None
    idx, disp = np.array(idx), np.stack(disp, 1)                   # (frames, n_ends, 3)

    def drive(t):
        f = min(t * V.fps, V.n - 1)
        k0 = int(f)
        k1 = min(k0 + 1, V.n - 1)
        w = f - k0
        m.qpos_spring[idx] = ((1 - w) * disp[k0] + w * disp[k1]).ravel()
    return drive


def rest_driver(m, tissues, parts, P, V):
    """mem_rest: the visible sheet gains / loses material at its base as the grasper lifts (its 4D area runs 0.56-1.11x
    of rest); with fixed rest lengths the surplus crumples. 'area': every edge's rest length scaled by
    sqrt(4D area / rest area); 'edges': each edge follows its own 4D length (both smoothed over 3 frames). mem_pre
    multiplies the rest lengths (< 1 = slight pre-tension)."""
    if P['mem_rest'] == 'fixed' or 'membrane' not in parts:
        return None
    from scipy.ndimage import gaussian_filter1d
    T, part = tissues['membrane'], parts['membrane'][0]
    if 'verts4d' not in T or part['inv'] is not None:
        return None
    f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, f"{part['name']}_edges")
    if f < 0:
        return None
    a, n = m.flex_edgeadr[f], m.flex_edgenum[f]
    E = m.flex_edge[a:a + n]
    L0 = m.flexedge_length0[a:a + n].copy()
    rec = np.asarray(T['verts4d'], float)
    if P['mem_rest'] == 'area':
        F = np.asarray(T['faces'])
        area = lambda X: np.linalg.norm(np.cross(X[..., F[:, 1], :] - X[..., F[:, 0], :], X[..., F[:, 2], :] - X[..., F[:, 0], :]), axis=-1).sum(-1)
        sc = gaussian_filter1d(np.sqrt(area(rec) / area(np.asarray(T['rest_verts'], float))), 3)
        L = sc[:, None] * L0[None]
    else:
        L = gaussian_filter1d(np.linalg.norm(rec[:, E[:, 0]] - rec[:, E[:, 1]], axis=2), 3, axis=0)
        L = np.clip(L, 0.5 * L0, 2.0 * L0)
    L = L * P['mem_pre']

    def drive(t):
        fr = min(t * V.fps, V.n - 1)
        k0 = int(fr)
        k1 = min(k0 + 1, V.n - 1)
        w = fr - k0
        m.flexedge_length0[a:a + n] = (1 - w) * L[k0] + w * L[k1]
    return drive


def start_from_recon(m, d, tissues, parts, k=0):
    """Put every jointed tissue vertex at its reconstructed position in frame k (the rest shape stays the elastic
    reference): the tissue then starts where the video shows it, not in its stress-free shape."""
    for t, plist in parts.items():
        T = tissues[t]
        for p in plist:
            if p['dim'] == 1:
                start = p['spec']['nodes4d'][k]
            elif p['inv'] is None and 'verts4d' in T:
                start = T['verts4d'][k]
            else:
                continue
            for i in range(len(p['X'])):
                b = m.body(f"{p['name']}_{i}")
                if b.jntnum[0] == 3:
                    adr = m.jnt_qposadr[b.jntadr[0]]
                    d.qpos[adr:adr + 3] = start[i] - p['X'][i]


def backdrop_behind(B, G, clearance=0.001):
    """Backdrop vertices of the gallbladder bed that lie inside the gallbladder's rest shape are pushed back along
    their view ray until they clear it (the background track filled the bed assuming a thinner organ). Visual only."""
    Xb = np.array(B['rest_verts'], float)
    if G is None or 'ray_dir' not in B:
        return Xb
    from scipy.spatial import Delaunay
    hull = Delaunay(G['rest_verts'])                     # the gallbladder is close to convex: hull as inside test
    cand = np.nonzero(np.asarray(B.get('bed_weight', np.ones(len(Xb)))) > 0)[0]
    ray = np.asarray(B['ray_dir'], float)
    for _ in range(40):
        inside = cand[hull.find_simplex(Xb[cand]) >= 0]
        if not len(inside):
            break
        Xb[inside] += 0.002 * ray[inside]
    return Xb + 0 * clearance


def control_targets(V, tools, P):
    """Joint targets per control step for every instrument (yaw, pitch, insertion, roll, jaw), aimed at the TCP."""
    t_src = np.arange(V.n) / V.fps
    t_ctl = np.arange(0, t_src[-1] + 1e-9, P['dt'])
    blocks = []
    for ins in V.clip.instruments:
        tr = tools[ins['name']]
        tcp = tr['tcp']
        if not tr['holds'] and P['probe_depth_mm']:     # the probe's depth is the least certain input: shift it
            ray = tcp - V.pos                             # toward the camera along each frame's viewing ray
            tcp = tcp - P['probe_depth_mm'] / 1000 * ray / np.linalg.norm(ray, axis=1, keepdims=True)
        Ti = np.stack([np.interp(t_ctl, t_src, tcp[:, k]) for k in range(3)], 1)
        jaw = np.interp(t_ctl, t_src, tr['jaw'])
        out = np.array([list(INS.ik(tr['rcm'], t, tr['heading'])) + [0.0, j] for t, j in zip(Ti, jaw)])
        for k in range(1, len(out)):
            out[k] = out[k - 1] + np.clip(out[k] - out[k - 1], -SLEW, SLEW)
        blocks.append(out)
    return np.concatenate(blocks, 1), t_ctl


def counterfactual(V, tools, P):
    """Instrument actions that differ from the video (data generation): grasper lifted cf_lift_mm higher (world z,
    ramped in over the first second), probe pushed cf_probe_mm deeper along its shaft, probe withdrawn 4 cm
    (cf_no_probe), grasp released at frame cf_release_frame."""
    ramp = np.clip(np.arange(V.n) / V.fps, 0, 1)[:, None]
    for name, tr in tools.items():
        if tr['holds'] and P['cf_lift_mm']:
            tr['tcp'] = tr['tcp'] + ramp * np.array([0, 0, P['cf_lift_mm'] / 1000])
        if not tr['holds'] and (P['cf_probe_mm'] or P['cf_no_probe']):
            shaft = tr['tcp'] - tr['rcm']
            shaft /= np.linalg.norm(shaft, axis=1, keepdims=True)
            dz = -0.04 if P['cf_no_probe'] else P['cf_probe_mm'] / 1000
            tr['tcp'] = tr['tcp'] + ramp * dz * shaft
    if P['noise_mm']:            # input uncertainty: smooth random error (RMS noise_mm per axis, ~0.2 s correlation)
        from scipy.ndimage import gaussian_filter1d
        rng = np.random.default_rng(int(P['noise_seed']))
        for name in sorted(tools):
            e = gaussian_filter1d(rng.standard_normal((V.n, 3)), 5, axis=0)
            tools[name]['tcp'] = tools[name]['tcp'] + e / e.std(0) * P['noise_mm'] / 1000


def apply(m, d, V, a):
    for k, ins in enumerate(V.clip.instruments):
        p, blk = ins['name'], a[5 * k:5 * k + 5]
        for i, j in enumerate(INS.ARM):
            d.ctrl[m.actuator(f'{p}_{j}').id] = blk[i]
        for j in ('jaw_left', 'jaw_right'):
            d.ctrl[m.actuator(f'{p}_{j}').id] = blk[4]


def simulate(V, tissues, tools, P, log=print, render_frames=(0, 60, 120, 180, 240)):
    xml, grasps, parts = build_xml(V, tissues, tools, P)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    acts, t_ctl = control_targets(V, tools, P)

    def place(a):
        for k, ins in enumerate(V.clip.instruments):
            for i, j in enumerate(INS.ARM):
                d.qpos[m.joint(f"{ins['name']}_{j}").qposadr[0]] = a[5 * k + i]
        apply(m, d, V, a)
        mujoco.mj_forward(m, d)
    place(acts[0])
    if P['start'] == 'recon' or P['mode'] == 'recon':
        start_from_recon(m, d, tissues, parts)
        mujoco.mj_forward(m, d)
    # a probe that starts inside the (unindented) rest tissue is backed off along its shaft until clear, then eased
    # back in over probe_ramp seconds, so it pushes the tissue in instead of starting embedded
    backoff = {}
    for k, ins in enumerate(V.clip.instruments):
        if ins.get('holds'):
            continue
        name = ins['name']
        gids = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith(name)}
        b = 0.0
        while b < 0.025 and any(d.contact[c].geom1 in gids or d.contact[c].geom2 in gids for c in range(d.ncon)):
            b += 0.0005
            a = acts[0].copy()
            a[5 * k + 2] -= b
            place(a)
        if b:
            n_r = max(2, int(round(P['probe_ramp'] / P['dt'])))
            for s_ in range(min(n_r, len(acts))):
                acts[s_, 5 * k + 2] -= b * (1 - s_ / n_r)
            backoff[name] = round(b * 1000, 1)
    place(acts[0])
    # grasp: the attached vertices (or, without any, the closest patch of the gallbladder) follow the grasper jaws
    pulls, grasp_info = [], {}
    holder = next((ins['name'] for ins in V.clip.instruments if ins.get('holds')), None)
    if holder:
        rl = m.body(f'{holder}_roll_link').id
        Rb = d.xmat[rl].reshape(3, 3)
        tcp = d.site(f'{holder}_tcp').xpos.copy()
        names = list(dict.fromkeys(grasps))
        if not names and 'gallbladder' in parts:
            gp = parts['gallbladder'][0]
            X = np.array([d.xpos[m.body(f"{gp['name']}_{i}").id] for i in range(len(gp['X']))])
            p0 = X[np.argmin(np.linalg.norm(X - tcp, axis=1))]
            names = [f"{gp['name']}_{i}" for i in np.nonzero(np.linalg.norm(X - p0, axis=1) < P['grasp_r'])[0]]
        names = names[:P['grasp_n']]
        pts = np.array([d.xpos[m.body(n).id] for n in names]) if names else np.zeros((0, 3))
        c = pts.mean(0) if len(pts) else tcp
        eids, off0, off1 = [], [], []
        for e, n in enumerate(names):
            eid = m.equality(f'{holder}_grasp{e}').id
            m.eq_obj1id[eid] = m.body(n).id
            m.eq_data[eid, 0:3] = 0
            p = d.xpos[m.body(n).id]
            eids.append(eid)
            off0.append(Rb.T @ (p - d.xpos[rl]))
            # grasp_target 'tcp': the held patch is pulled onto the tool-centre point; 'recon': each held vertex
            # goes to where the reconstruction has it in frame 0 relative to the jaw (the sheet's apex sits ~3 mm
            # from the TCP in the 4D)
            tgt = p + (tcp - c)
            pn, _, vi = n.rpartition('_')
            T_ = tissues.get(pn)
            if P['grasp_target'] == 'recon' and T_ is not None and 'verts4d' in T_ and parts[pn][0]['inv'] is None:
                tgt = T_['verts4d'][0][int(vi)]
            off1.append(Rb.T @ (tgt - d.xpos[rl]))
            m.eq_data[eid, 3:6] = off0[-1]
            d.eq_active[eid] = 1
        if eids:
            pulls.append([d.time, np.array(eids), np.array(off0), np.array(off1)])
        grasp_info = dict(holder=holder, n=len(names), gap_mm=round(float(np.linalg.norm(tcp - c)) * 1000, 2))
    grasp_info['probe_backoff_mm'] = backoff

    def pull():
        for g in pulls:
            if g[0] is None:
                continue
            f = min(1.0, (d.time - g[0]) / P['grasp_ramp'])
            m.eq_data[g[1], 3:6] = g[2] + f * (g[3] - g[2])
            if f >= 1.0:
                g[0] = None
    drivers = [f for f in (bed_driver(m, tissues, parts, P, V), base_driver(m, tissues, parts, P, V),
                           duct_end_driver(m, parts, P, V), rest_driver(m, tissues, parts, P, V)) if f]
    drive = (lambda t: [f(t) for f in drivers]) if drivers else None
    if drive:
        drive(0.0)
    for _ in range(int(P['settle'] / P['ts']) if P['mode'] == 'sim' else 0):
        pull()
        mujoco.mj_step(m, d)
    part_list = [p for t in parts for p in parts[t]]
    fid = {p['name']: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, p['name']) for p in part_list}
    raw = {p['name']: [] for p in part_list}
    tips = {ins['name']: [] for ins in V.clip.instruments}

    # forces on the instruments (N): grasp = net pull of the held tissue on the jaws (equality constraint forces),
    # probe = summed normal contact force on the probe's geoms
    grasp_eids = np.array([int(e) for g in pulls for e in g[1]], int)
    probe_gids = {g for g in range(m.ngeom) if any((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith(ins['name'])
                                                   for ins in V.clip.instruments if not ins.get('holds'))}
    forces = dict(grasp=[], probe=[])
    f6 = np.zeros(6)

    def log_forces():
        rows = np.nonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) &
                          np.isin(d.efc_id[:d.nefc], grasp_eids))[0]
        fg = d.efc_force[rows].reshape(-1, 3).sum(0) if len(rows) and len(rows) % 3 == 0 else np.zeros(3)
        fp = 0.0
        for c in range(d.ncon):
            if d.contact[c].geom1 in probe_gids or d.contact[c].geom2 in probe_gids:
                mujoco.mj_contactForce(m, d, c, f6)
                fp += f6[0]
        forces['grasp'].append(float(np.linalg.norm(fg)))
        forces['probe'].append(float(fp))

    def log_state():
        log_forces()
        for p in part_list:
            a, n = m.flex_vertadr[fid[p['name']]], m.flex_vertnum[fid[p['name']]]
            raw[p['name']].append(d.flexvert_xpos[a:a + n].copy())
        for name in tips:
            tips[name].append(d.site(f'{name}_tcp').xpos.copy())
    log_state()
    nsub = int(round(P['dt'] / P['ts']))
    if P['mode'] != 'sim':                  # kinematic replay: tissues placed from the reconstruction / at rest
        nsub = 0
    shots = {}
    rend = mujoco.Renderer(m, 360, 480)
    centre = np.mean([v[0].mean(0) for v in raw.values()], 0) if raw else V.pos[0]
    view_dir = V.R[0][2]

    def snap(k_frame):
        imgs = []
        for az_off, el in ((0, -10), (70, -25), (150, -20)):
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = centre
            c.distance = 0.13
            c.azimuth = float(np.degrees(np.arctan2(-view_dir[1], -view_dir[0]))) + az_off
            c.elevation = el
            rend.update_scene(d, camera=c)
            imgs.append(rend.render())
        shots[k_frame] = np.concatenate(imgs, 1)
    step_of = {int(round(kf / V.fps / P['dt'])): kf for kf in render_frames}
    scope = []
    srend = mujoco.Renderer(m, V.H, V.W) if P['tex_dir'] else None
    cid = m.camera('endo').id

    def scope_shot(t):
        if srend is None:
            return
        kf = int(np.clip(round(t * V.fps), 0, V.n - 1))
        m.cam_quat[cid] = V.cam.mj_quat(V.R[kf])
        m.cam_fovy[cid] = np.degrees(2 * np.arctan(V.H / 2 / V.f[kf]))
        m.cam_pos[cid] = V.pos[kf]
        srend.update_scene(d, camera='endo')
        scope.append(srend.render())
    scope_shot(0.0)
    if 0 in step_of:
        snap(step_of[0])
    for k in range(1, len(acts)):
        a, b = acts[k - 1], acts[k]
        if P['mode'] != 'sim':
            for kk, ins in enumerate(V.clip.instruments):
                for i, j in enumerate(INS.ARM):
                    d.qpos[m.joint(f"{ins['name']}_{j}").qposadr[0]] = b[5 * kk + i]
            d.qpos[[m.jnt_qposadr[j] for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').endswith('_jaw_left')]] = 0
            if P['mode'] == 'recon':
                start_from_recon(m, d, tissues, parts, k=int(np.clip(round(k * P['dt'] * V.fps), 0, V.n - 1)))
            mujoco.mj_forward(m, d)
        for s in range(nsub):
            apply(m, d, V, a + (b - a) * (s + 1) / nsub)
            pull()
            if drive and s % 20 == 0:
                drive((k - 1) * P['dt'] + (s + 1) * P['ts'])
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            dof = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].lastinfo)
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[m.dof_jntid[dof]])) if 0 <= dof < m.nv else '?'
            raise RuntimeError(f'unstable at control step {k} (dof {dof}, body {body})')
        log_state()
        scope_shot(k * P['dt'])
        if k in step_of:
            snap(step_of[k])
        if P['cf_release_frame'] >= 0 and k == int(round(P['cf_release_frame'] / V.fps / P['dt'])):
            for g in pulls:
                d.eq_active[g[1]] = 0
    out = {}
    for t, plist in parts.items():
        for p in plist:
            arr = np.array(raw[p['name']], np.float32)
            out[f'part_{p["name"]}'] = arr
            if p['dim'] == 1:
                continue
            out[t] = arr[:, p['inv']] if p['inv'] is not None else arr
    out.update({f'tip_{n}': np.array(v, np.float32) for n, v in tips.items()})
    out.update({f'force_{n}': np.array(v, np.float32) for n, v in forces.items()})
    out['_shots'] = shots
    out['_scope'] = scope
    out['_parts'] = parts
    return out, xml, grasp_info, t_ctl


def run_sim(V, tissues, tools, P, log=print, retries=3):
    for attempt in range(retries + 1):
        try:
            return simulate(V, tissues, tools, P, log) + (P,)
        except RuntimeError as e:
            if attempt == retries:
                raise
            log(f'[scene4d] {e}; retrying with time step {P["ts"] / 2 * 1e3:.4f} ms')
            P = dict(P, ts=P['ts'] / 2)


# ---------------------------------------------------------------- evaluation
def to_frames(x, t_ctl, V):
    """(steps, ...) at control times -> (n_frames, ...) at video frames (nearest step)."""
    idx = np.clip(np.round(np.arange(V.n) / V.fps / (t_ctl[1] - t_ctl[0])).astype(int), 0, len(x) - 1)
    return x[idx]


def procrustes(A, B):
    """Rigid (R, t) with B ~ A @ R.T + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Wt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Wt))])
    R = (U @ D @ Wt).T
    return R, cb - R @ ca


def rigid_split(rest, sim4, rec4, ks):
    """Per frame: the whole-body (rigid) motion of sim and reconstruction relative to rest, and what is left.
    rigid_err: distance between the rest shape moved by the sim's and by the reconstruction's rigid motions;
    rot_*: rotation angles (deg); nonrigid_err: sim vs reconstruction after aligning them rigidly;
    deform_*: non-rigid deformation of sim / reconstruction relative to rest."""
    out = dict(rigid_err=[], rot_sim=[], rot_rec=[], nonrigid_err=[], deform_sim=[], deform_rec=[])
    ang = lambda R: float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    for k in ks:
        Rs, ts = procrustes(rest, sim4[k])
        Rr, tr = procrustes(rest, rec4[k])
        a, b = rest @ Rs.T + ts, rest @ Rr.T + tr
        out['rigid_err'].append(np.linalg.norm(a - b, axis=1).mean() * 1000)
        out['rot_sim'].append(ang(Rs))
        out['rot_rec'].append(ang(Rr))
        R2, t2 = procrustes(sim4[k], rec4[k])
        out['nonrigid_err'].append(np.linalg.norm(rec4[k] - (sim4[k] @ R2.T + t2), axis=1).mean() * 1000)
        out['deform_sim'].append(np.linalg.norm(sim4[k] - a, axis=1).mean() * 1000)
        out['deform_rec'].append(np.linalg.norm(rec4[k] - b, axis=1).mean() * 1000)
    return {k: Q.summarize(v) for k, v in out.items()}


def min_angle(X, F):
    """Smallest interior angle (deg) of each triangle."""
    P = np.asarray(X)[np.asarray(F)]
    ang = []
    for i in range(3):
        u, w = P[:, (i + 1) % 3] - P[:, i], P[:, (i + 2) % 3] - P[:, i]
        c = (u * w).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-15)
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.min(ang, 0)


def edges_of(faces):
    F = np.asarray(faces)
    return np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)


def evaluate(V, tissues, traj, t_ctl, tools, ks, parts=None):
    res = {}
    for t, T in tissues.items():
        if T is not None and parts and t in parts and parts[t][0]['dim'] == 1:
            r = evaluate_cables(V, T, traj, t_ctl, ks, parts[t])
            r.pop('_tube', None)
            res[t] = dict(version=T['version'], **r)
            continue
        if T is None or t not in traj:
            continue
        sim4 = to_frames(traj[t], t_ctl, V)
        F = T['faces']
        mk = tissue_masks(T, V, t)
        r = dict(version=T['version'])
        if mk is not None:
            saved = V._masks
            try:                                   # score against the tissue track's own masks
                V._masks = np.concatenate([saved, mk[:, None]], 1)
                V.names = list(V.names) + [f'_{t}']
                iou = Q.silhouette_iou(V, sim4, F, f'_{t}', ks)
                r['sim_iou'] = Q.summarize(iou)
                r['sim_iou_bands'] = bands(iou, ks)
                # without the visibility cut against the video's depth (the cut drops mesh pixels > 6 mm behind
                # the video surface: a sim surface 3-5 mm behind costs IoU there, r17)
                r['sim_iou_2d'] = Q.summarize(Q.silhouette_iou(V, sim4, F, f'_{t}', ks, visible_only=False))
                r['sim_boundary_f'] = Q.summarize(Q.boundary_f(V, sim4, F, f'_{t}', ks))
                r['sim_depth_mm'] = Q.summarize(Q.depth_residual(V, sim4, F, f'_{t}', ks) * 1000)
                if 'verts4d' in T:
                    iou_r = Q.silhouette_iou(V, T['verts4d'], F, f'_{t}', ks)
                    r['recon_iou'] = Q.summarize(iou_r)
                    r['recon_iou_bands'] = bands(iou_r, ks)
                    r['recon_iou_2d'] = Q.summarize(Q.silhouette_iou(V, T['verts4d'], F, f'_{t}', ks, visible_only=False))
                r['static_iou'] = Q.summarize(Q.silhouette_iou(V, T['rest_verts'], F, f'_{t}', ks))
                fe, fs = Q.flow_epe(V, sim4, F, f'_{t}', [k for k in range(0, V.n - 5, 5)])
                r['flow_epe_px'] = Q.summarize(fe)
                r['flow_epe_static_px'] = Q.summarize(fs)
                r['flow_explained'] = round(float(1 - np.mean(fe) / max(np.mean(fs), 1e-9)), 3) if len(fe) else None
                if 'verts4d' in T:
                    fr, _ = Q.flow_epe(V, T['verts4d'], F, f'_{t}', [k for k in range(0, V.n - 5, 5)])
                    r['flow_epe_recon_px'] = Q.summarize(fr)
            finally:
                V._masks = saved
                V.names = V.names[:-1]
        if 'verts4d' in T:
            rec = np.asarray(T['verts4d'])
            e_sim = np.linalg.norm(sim4 - rec, axis=2).mean(1) * 1000
            e_static = np.linalg.norm(T['rest_verts'][None] - rec, axis=2).mean(1) * 1000
            r['err_vs_recon_mm'] = Q.summarize(e_sim)
            r['err_static_vs_recon_mm'] = Q.summarize(e_static)
            r['motion_explained'] = round(float(1 - e_sim.mean() / max(e_static.mean(), 1e-9)), 3)
            r['split'] = rigid_split(T['rest_verts'], sim4, rec, ks)
            if t == 'gallbladder':
                free = ~bed_vertices(T, V, T['rest_verts'])
                ef = np.linalg.norm(sim4[:, free] - rec[:, free], axis=2).mean(1) * 1000
                e0 = np.linalg.norm(T['rest_verts'][None, free] - rec[:, free], axis=2).mean(1) * 1000
                r['free_err_vs_recon_mm'] = Q.summarize(ef)
                r['free_motion_explained'] = round(float(1 - ef.mean() / max(e0.mean(), 1e-9)), 3)
        if 'tets' in T:                            # a fluid-filled organ keeps its volume: does the 4D / the sim?
            import trimesh
            vml = lambda X: abs(trimesh.Trimesh(X, F, process=False).volume) * 1e6
            vs = [vml(sim4[k]) for k in ks]
            r['volume_ml'] = dict(rest=round(vml(T['rest_verts']), 2), sim=[round(min(vs), 2), round(max(vs), 2)])
            if 'verts4d' in T:
                vr = [vml(T['verts4d'][k]) for k in ks]
                r['volume_ml']['recon'] = [round(min(vr), 2), round(max(vr), 2)]
            Tt = np.asarray(T['tets'])
            vol = lambda X: np.einsum('ij,ij->i', np.cross(X[Tt[:, 1]] - X[Tt[:, 0]], X[Tt[:, 2]] - X[Tt[:, 0]]), X[Tt[:, 3]] - X[Tt[:, 0]])
            s0 = np.sign(vol(T['rest_verts']))
            r['inverted_tets_max'] = int(max(((np.sign(vol(sim4[k])) != s0).sum() for k in range(0, V.n, 5))))
        if t == 'membrane':                       # crumpling: fold angle between neighbouring faces, area vs 4D
            r['fold_p90_deg'] = dict(sim=fold_p90(sim4, F, ks), recon=fold_p90(T['verts4d'], F, ks) if 'verts4d' in T else None)
            if 'verts4d' in T:
                ar = [face_area(sim4[k], F) / face_area(T['verts4d'][k], F) for k in ks]
                r['area_vs_recon'] = dict(median=round(float(np.median(ar)), 3), max=round(float(np.max(ar)), 3))
        mx, p95 = Q.stretch(T['rest_verts'], sim4, edges_of(F))
        r['stretch_max'] = round(float(mx.max()), 2)
        r['stretch_p95_max'] = round(float(p95.max()), 2)
        if t == 'gallbladder':
            r.update(texture_tracking(V, T, traj[t], t_ctl))
            r.update(probe_push(V, T, sim4, tools))
        res[t] = r
    res['anatomy'] = anatomy(V, tissues, traj, t_ctl, ks, parts)
    # instruments: does each simulated tip follow its planned path (or does tissue block it)?
    res['instruments'] = {}
    for n in ('grasp', 'probe'):
        if f'force_{n}' in traj:
            fr = to_frames(traj[f'force_{n}'], t_ctl, V)
            res['instruments'][f'{n}_force_N'] = dict(Q.summarize(fr), bands=bands(fr[ks], ks))
    for name, tr in tools.items():
        if f'tip_{name}' in traj:
            sim_tip = to_frames(traj[f'tip_{name}'], t_ctl, V)
            e = np.linalg.norm(sim_tip - tr['tcp'], axis=1) * 1000
            res['instruments'][name] = dict(tip_lag_mm=Q.summarize(e))
    return res


def face_area(X, F):
    return float(np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1).sum() / 2)


def fold_p90(X4, F, ks):
    """Median over keyframes of the 90th percentile dihedral (fold) angle between neighbouring faces, degrees."""
    F = np.asarray(F)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    fid = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((E[:, 1], E[:, 0]))
    E, fid = E[order], fid[order]
    same = (E[1:] == E[:-1]).all(1)
    pairs = np.stack([fid[:-1][same], fid[1:][same]], 1)
    out = []
    for k in ks:
        X = np.asarray(X4[k], float)
        n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-18
        out.append(np.percentile(np.degrees(np.arccos(np.clip((n[pairs[:, 0]] * n[pairs[:, 1]]).sum(1), -1, 1))), 90))
    return round(float(np.median(out)), 1)


def anatomy(V, tissues, traj, t_ctl, ks, parts):
    """Is the anatomy still connected in the simulation? Gaps (mm) between tissues that are attached in reality:
    duct start to the gallbladder surface, sheet base to it, held sheet row to the grasper TCP; and penetrations:
    duct nodes / free sheet vertices more than 1 mm inside the gallbladder (% of them, worst keyframe)."""
    G, M = tissues.get('gallbladder'), tissues.get('membrane')
    if G is None or 'gallbladder' not in traj:
        return {}
    gb = to_frames(traj['gallbladder'], t_ctl, V)
    cab = {p['name']: to_frames(traj[f'part_{p["name"]}'], t_ctl, V) for p in parts.get('ducts', [])}
    mem = to_frames(traj['membrane'], t_ctl, V) if 'membrane' in traj and M is not None else None
    att = attachments(M) if M is not None else {}
    base = [i for i, a in att.items() if a == 'gallbladder']
    jaw = [i for i, a in att.items() if a not in ('gallbladder', 'backdrop')]
    free = np.setdiff1d(np.arange(len(M['rest_verts'])), base + jaw) if M is not None else []
    tip = to_frames(traj['tip_grasper_left'], t_ctl, V) if 'tip_grasper_left' in traj else None
    r = {k: [] for k in ('duct_start_gap_mm', 'duct_nodes_inside_pct', 'sheet_base_gap_mm', 'sheet_inside_pct', 'jaw_tcp_mm')}
    for k in ks:
        sd = Q.signed_distance(gb[k], G['faces'])
        if cab:
            r['duct_start_gap_mm'].append(max(abs(sd(c[k][:1]))[0] for c in cab.values()) * 1000)
            r['duct_nodes_inside_pct'].append(float((sd(np.concatenate([c[k][1:] for c in cab.values()])) > 0.001).mean() * 100))
        if mem is not None and base:
            r['sheet_base_gap_mm'].append(float(np.median(abs(sd(mem[k][base])))) * 1000)
            r['sheet_inside_pct'].append(float((sd(mem[k][free]) > 0.001).mean() * 100))
            if tip is not None and jaw:
                r['jaw_tcp_mm'].append(float(np.linalg.norm(mem[k][jaw].mean(0) - tip[k])) * 1000)
    return {k: dict(median=round(float(np.median(v)), 2), max=round(float(np.max(v)), 2)) for k, v in r.items() if v}


BANDS = ((0, 62), (62, 130), (130, 251))        # pan / lift / probe phases of chole_a


def bands(x, ks):
    """Mean of a per-keyframe metric in each phase band."""
    x, ks = np.asarray(x, float), np.asarray(ks)
    return [round(float(np.nanmean(x[(ks >= a) & (ks < b)])), 4) if ((ks >= a) & (ks < b)).any() else None for a, b in BANDS]


def tube_mesh(nodes, radius, n_around=8):
    """Tube surface around a polyline (per-frame nodes (n_t, n, 3) or (n, 3)): rings of n_around vertices."""
    def one(C):
        C = np.asarray(C)
        t = np.gradient(C, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        a = np.cross(t, [0, 0, 1.0])
        a[np.linalg.norm(a, axis=1) < 1e-6] = [1, 0, 0]
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = np.cross(t, a)
        ang = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
        r = np.broadcast_to(np.asarray(radius, float), (len(C),))
        return (C[:, None] + r[:, None, None] * (np.cos(ang)[None, :, None] * a[:, None] + np.sin(ang)[None, :, None] * b[:, None])).reshape(-1, 3)
    nodes = np.asarray(nodes)
    n = nodes.shape[-2]
    F = []
    for i in range(n - 1):
        for j in range(n_around):
            a0, a1 = i * n_around + j, i * n_around + (j + 1) % n_around
            b0, b1 = a0 + n_around, a1 + n_around
            F += [[a0, b0, a1], [a1, b0, b1]]
    V = np.stack([one(c) for c in nodes]) if nodes.ndim == 3 else one(nodes)
    return V, np.array(F)


def evaluate_cables(V, T, traj, t_ctl, ks, parts):
    """Ducts simulated as cables: centreline error vs the track's 4D centreline, length change, and the tube
    rendered around the simulated centreline against the duct track's masks (union of its tubes)."""
    res = {}
    sims, specs = [], []
    for p in parts:
        sim = to_frames(traj[f'part_{p["name"]}'], t_ctl, V)
        rec = p['spec']['nodes4d']
        e = np.linalg.norm(sim - rec, axis=2).mean(1) * 1000
        e0 = np.linalg.norm(p['X'][None] - rec, axis=2).mean(1) * 1000
        L0 = np.linalg.norm(np.diff(p['X'], axis=0), axis=1).sum()
        L = np.linalg.norm(np.diff(sim, axis=1), axis=2).sum(1)
        free = np.setdiff1d(np.arange(len(p['X'])), p.get('driven', []))      # nodes the simulation decides
        ef = np.linalg.norm(sim[:, free] - rec[:, free], axis=2).mean()
        ef0 = np.linalg.norm(p['X'][None, free] - rec[:, free], axis=2).mean()
        res[p['name']] = dict(err_vs_recon_mm=Q.summarize(e), err_static_vs_recon_mm=Q.summarize(e0),
                              motion_explained=round(float(1 - e.mean() / max(e0.mean(), 1e-9)), 3),
                              free_nodes=len(free), free_motion_explained=round(float(1 - ef / max(ef0, 1e-9)), 3),
                              free_err_mm=round(float(ef * 1000), 2),
                              length_ratio=[round(float(L.min() / L0), 3), round(float(L.max() / L0), 3)])
        sims.append((sim, p['spec']['radius'], rec))
    mk = tissue_masks(T, V, 'ducts')
    if mk is not None and sims:
        def tubes(which):
            verts, faces, off = [], [], 0
            for c in sims:
                vv, ff = tube_mesh(c[which], c[1])
                verts.append(vv)
                faces.append(ff + off)
                off += vv.shape[1]
            return np.concatenate(verts, 1), np.concatenate(faces)
        Vt, Ft = tubes(0)
        Vr, _ = tubes(2)                      # the same tubes around the track's own 4D centrelines (baseline)
        saved = V._masks
        try:
            V._masks = np.concatenate([saved, mk[:, None]], 1)
            V.names = list(V.names) + ['_ducts']
            iou = Q.silhouette_iou(V, Vt, Ft, '_ducts', ks)
            iou_r = Q.silhouette_iou(V, Vr, Ft, '_ducts', ks)
            res['sim_iou'] = Q.summarize(iou)
            res['recon_iou'] = Q.summarize(iou_r)
            res['sim_iou_bands'], res['recon_iou_bands'] = bands(iou, ks), bands(iou_r, ks)
        finally:
            V._masks = saved
            V.names = V.names[:-1]
        res['_tube'] = (Vt, Ft)
    return res


def texture_tracking(V, T, flex, t_ctl):
    """Organ texture tracks (re-seeded every 25 frames, 2 s) against the simulated surface: pooled median px and the
    change against the static organ over the motion phases (evaluate.track_motion)."""
    from . import evaluate as EV
    z = np.load(V.clip.prep / 'organ_tracks.npz')
    tracks = (z['P'], z['seed'])
    cams = dict(R=V.R, f=V.f, pos=V.pos)
    organ = dict(X=T['rest_verts'], faces=np.asarray(T['faces']))
    dt = float(t_ctl[1] - t_ctl[0])
    kw = dict(tracks=tracks, step=5)
    sim = EV.evaluate(V.clip, organ, flex, list(V.frames), V._masks, V._depth, cams, dt, **kw)
    stat = EV.evaluate(V.clip, organ, np.repeat(flex[:1], len(flex), 0), list(V.frames), V._masks, V._depth, cams, dt, **kw)
    tm = EV.track_motion(sim, EV.motion_seeds(stat))
    return dict(track_px=sim['track_err_px'], track_px_static=stat['track_err_px'],
                track_motion=tm, track_samples=sim['n_track_samples'])


def probe_push(V, T, sim4, tools, radius=0.01):
    """Displacement (mm) of the gallbladder within `radius` of the probe tip, simulated vs reconstructed (4D), in
    the frames where the probe tip is within 5 mm of the reconstructed surface."""
    from scipy.spatial import cKDTree
    probe = next((n for n, t in tools.items() if not t['holds']), None)
    if probe is None or 'verts4d' not in T:
        return {}
    rec = np.asarray(T['verts4d'])
    rest = T['rest_verts']
    sim_d, rec_d, frames = [], [], []
    for k in range(V.n):
        tip = tools[probe]['tip'][k]
        dist, _ = cKDTree(rec[k]).query(tip)
        if dist > 0.005:
            continue
        near = np.linalg.norm(rec[k] - tip, axis=1) < radius
        if near.sum() < 3:
            continue
        rec_d.append(float(np.linalg.norm(rec[k][near] - rest[near], axis=1).mean()) * 1000)
        sim_d.append(float(np.linalg.norm(sim4[k][near] - rest[near], axis=1).mean()) * 1000)
        frames.append(k)
    if not frames:
        return dict(probe_push=dict(frames=0))
    return dict(probe_push=dict(frames=len(frames), sim_mm=round(float(np.mean(sim_d)), 2),
                                recon_mm=round(float(np.mean(rec_d)), 2),
                                abs_err_mm=round(float(np.mean(np.abs(np.array(sim_d) - np.array(rec_d)))), 2)))


def sheets(V, tissues, traj, t_ctl, out, ks, parts=None):
    sim_layers, rec_layers = [], []
    for t in ('backdrop',) + SIMULATED:
        T = tissues.get(t)
        if T is None:
            continue
        if parts and t in parts and parts[t][0]['dim'] == 1:
            for p in parts[t]:
                vv, ff = tube_mesh(to_frames(traj[f'part_{p["name"]}'], t_ctl, V), p['spec']['radius'])
                sim_layers.append((f'{p["name"]} sim', vv, ff, COLOR[t]))
        if t in traj:
            sim_layers.append((f'{t} sim', to_frames(traj[t], t_ctl, V), T['faces'], COLOR[t]))
        elif t == 'backdrop':
            pass
        if 'verts4d' in T and t != 'backdrop':
            rec_layers.append((f'{t} recon', T['verts4d'], T['faces'], COLOR[t]))
    if sim_layers:
        Q.contact_sheet(V, sim_layers, ks, out / 'sheet_sim.jpg')
    if rec_layers:
        Q.contact_sheet(V, rec_layers, ks, out / 'sheet_recon.jpg')


def photometric(V, scope, t_ctl, tissues, out):
    """Textured render through the scope vs the video, on tissue pixels (gallbladder / sheet / duct masks, minus
    instruments): mean L1 (0-255) and NCC per keyframe; writes scope.mp4 (render), compare.mp4 (video | render)
    and render_sheet.jpg."""
    import cv2
    import imageio.v2 as imageio
    frames_idx = np.clip(np.round(t_ctl * V.fps).astype(int), 0, V.n - 1)
    region = np.zeros((V.n, V.H, V.W), bool)
    for t in ('gallbladder', 'membrane', 'ducts'):
        T = tissues.get(t)
        if T is not None:
            mk = tissue_masks(T, V, t)
            if mk is not None:
                region |= mk
    l1, ncc = [], []
    for j, kf in enumerate(frames_idx):
        if kf % 10:
            continue
        sel = region[kf] & ~V.occluders(kf)
        if sel.sum() < 100:
            continue
        a = scope[j].astype(np.float32)[sel]
        b = V.frames[kf].astype(np.float32)[sel]
        l1.append(float(np.abs(a - b).mean()))
        ga, gb = a.mean(1) - a.mean(), b.mean(1) - b.mean()
        ncc.append(float(ga @ gb / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-9)))
    fps = int(round(1 / (t_ctl[1] - t_ctl[0])))
    imageio.mimsave(out / 'scope.mp4', scope, fps=fps, macro_block_size=1, quality=7)
    side = [np.concatenate([V.frames[kf], img], 1) for kf, img in zip(frames_idx, scope)]
    imageio.mimsave(out / 'compare.mp4', side, fps=fps, macro_block_size=1, quality=7)
    tiles = [np.concatenate([V.frames[kf], scope[j]], 1) for j, kf in enumerate(frames_idx) if kf in (0, 60, 120, 180, 240)]
    sheet = cv2.resize(np.concatenate(tiles, 0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / 'render_sheet.jpg'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dict(l1=Q.summarize(l1), ncc=Q.summarize(ncc))


def main(argv):
    rnd = argv[0]
    versions = dict(a.split('=') for a in argv[1:] if '=' in a and a.split('=')[0] in TISSUES + ('interaction',))
    def val(v):
        try:
            return float(v)
        except ValueError:
            return v
    params = {a.split('=')[0]: val(a.split('=')[1]) for a in argv[1:] if '=' in a and a.split('=')[0] in PARAMS}
    out = ITER / 'rounds' / rnd
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P0 = dict(PARAMS, **params)
    V = views.load('chole_a', 'sift', CAMS.get(P0['cams'], P0['cams']))
    tissues = {t: load_tissue(t, versions.get(t)) for t in TISSUES}
    tools, tool_src = tool_tracks(V, versions.get('interaction'))
    P = dict(PARAMS, **params)
    counterfactual(V, tools, P)
    if P.get('render', 1):
        P['tex_dir'] = P['tex_dir'] or str(out / 'textures')
    traj, xml, grasp, t_ctl, P = run_sim(V, tissues, tools, P)
    shots = traj.pop('_shots')
    scope = traj.pop('_scope')
    if scope:
        metrics_photo = photometric(V, scope, t_ctl, tissues, out)
    else:
        metrics_photo = None
    parts = traj.pop('_parts')
    if shots:
        import cv2
        rows = []
        for kf, img in sorted(shots.items()):
            img = np.ascontiguousarray(img)
            cv2.putText(img, f'frame {kf}: scope-side / 70 deg / 150 deg', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rows.append(img)
        cv2.imwrite(str(out / 'orbit.jpg'), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / 'scene.xml').write_text(xml)
    np.savez_compressed(out / 'traj.npz', t=t_ctl, **traj)
    ks = list(range(0, V.n, 10))
    metrics = dict(round=rnd, photometric=metrics_photo, tissues={t: (T['version'] if T is not None else None) for t, T in tissues.items()},
                   tools=tool_src, params=P, grasp=grasp, sim_seconds=round(time.time() - t0, 1),
                   eval=evaluate(V, tissues, traj, t_ctl, tools, ks, parts))
    save_json(out / 'metrics.json', metrics)
    sheets(V, tissues, traj, t_ctl, out, [0, 40, 80, 120, 160, 200, 230, 250], parts)
    print(json.dumps(metrics['eval'], indent=1)[:3000])
    print(f'[scene4d] {rnd} done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main(sys.argv[1:])
