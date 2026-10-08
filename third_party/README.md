# third_party

| folder | source | version | license | used by |
|---|---|---|---|---|
| `vggt/` | https://github.com/facebookresearch/vggt (shallow clone, not modified) | a288dd0 (2026-05-18) | VGGT License (code); the weights in `models/VGGT-1B` (facebook/VGGT-1B) are CC BY-NC 4.0, non-commercial | `r2s/vggt_views.py` (imported via `sys.path`, not installed: its requirements pin torch 2.3) |

`models/VGGT-1B` holds only `model.safetensors` (5.03 GB, sha256 f164acf6...467e) from the Hugging Face repo, which
also has the same weights as `model.pt`. A commercially licensed checkpoint (facebook/VGGT-1B-Commercial) exists behind
an application form.
