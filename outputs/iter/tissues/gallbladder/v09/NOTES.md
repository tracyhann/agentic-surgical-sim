# gallbladder v09 (failed: no tet mesh)

`CFG['v09']` = v08 + surf_faces 800, cells 20, all tets weighted. Goal: v08's clean tet pipeline at a finer
resolution (v08 had only 646 surface faces and lost ~0.03 IoU).

Result: `tetrahedralize` raised 'tetrahedralisation failed' (work/run.out); no model.

Diagnosis (scratch tests on the v08 rest surface): at 800/20, 800/17, 700/18, 640/16 faces/cells the projected
remesh or the short-edge collapse is not watertight; 600/15 is watertight but self-intersecting. Decimating the
fitted template surface directly (1200-600 faces) is watertight, but TetGen's quality refinement (mindihedral 15,
radius-edge 1.6) grades down to tets with edges 0.1-6 % of the median again. A robust fine tet mesh needs an
isotropic remesher (not available in the venv) or an embedded (coarse control) deformation; not done here.

Decision: keep v06's healthy rest + tets and improve only the 4D data terms (v10).
