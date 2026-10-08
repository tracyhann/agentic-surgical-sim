# ducts v07

v06 with constant radius over the end 15 % (duct both ends, strands distal) and one radius per fibre; `--diag`
variants (`work/diag.json`): without length term the observed length varies 14-22 mm (duct), 12-24 mm (strands);
without video depth the 4D fit stays within 2 mm in depth.
Result: duct IoU 0.814, BF 0.811; strands IoU 0.698, BF 0.605 (both worse). Variants in `work/variants/`:
duct flat proximal only = 0.836 / 0.878, flat distal only = 0.821 / 0.824 -> the distal flat end protrudes where the cord
dives into the fat. v08 keeps only the proximal flat end and packed tapered fibres. Full log: v08/NOTES.md.
