# ducts v01 - first build

Tube fit of the duct (= clip `strand` mask, the shiny cord continuing the neck) and the pink strand band (new SAM
object, prompts in v08/NOTES.md; v01 was run with the same `masks.npz` as v02, i.e. with the frame 0/20 prompts - the
first SAM run without them lost the band in frames 0-30). Skeleton -> ordered path -> depth; rest = joint static fit
to keyframes 40..250 step 5 (multi-view); 4D = all frames jointly with temporal smoothness and a length term
(w_len 50); radius = median mask half-width x depth / f. Duct proximal end pulled to the gallbladder mask when within
25 px.

Result: duct IoU 0.799 (min 0.678), BF 0.815, depth 1.80 mm, length 18.3 mm (std 0.07), r 1.86 mm; strands IoU 0.720
(min 0.472), BF 0.685, depth 1.44 mm, length 13.3 mm, half-width 1.65 mm; union IoU 0.771; static rest IoU 0.57/0.59.
Seen: duct bridges a 20-30 px gap to a broken neck mask at f50-90 (the probe is there); strand start floats up to
11 mm from the duct start; duct depth residual 7-11 mm at f30-40 (blurred pan).
Next: junction 12 px; pull the strand start to the duct start. Full log: v08/NOTES.md.
