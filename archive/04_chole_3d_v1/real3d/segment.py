"""SAM 2.1 (tiny) video segmentation of the clip: gallbladder, grasper, probe, strand, liver, right organs.
Prompts are a few clicks per structure on frame 0; masks are propagated through all 151 frames."""
import time
import numpy as np, torch, imageio.v2 as imageio
from transformers import Sam2VideoModel, Sam2VideoProcessor

OBJ = {  # id: (name, positive clicks, negative clicks) on frame 0 (640x360)
    1: ('gallbladder', [(200, 290), (170, 240), (230, 190), (245, 150)], [(60, 150), (330, 90), (420, 190)]),
    2: ('grasper', [(165, 25), (195, 70), (212, 95)], [(250, 150)]),
    3: ('probe', [(480, 90), (560, 40), (380, 150)], [(420, 190)]),
    4: ('strand', [(350, 210), (420, 190), (470, 240), (490, 265)], [(480, 90), (250, 280)]),
    5: ('liver', [(60, 100), (60, 250), (120, 60)], [(200, 290)]),
    6: ('right organs', [(560, 230), (600, 300), (430, 315)], [(420, 190)]),
}


def main():
    dev = 'mps'
    frames = [f for f in imageio.get_reader('runs_real/chole_sweep/task/video.mp4')]
    H, W = frames[0].shape[:2]
    model = Sam2VideoModel.from_pretrained('models/sam2.1-hiera-tiny').to(dev, dtype=torch.float32).eval()
    proc = Sam2VideoProcessor.from_pretrained('models/sam2.1-hiera-tiny')
    t0 = time.time()
    sess = proc.init_video_session(video=frames, inference_device=dev, dtype=torch.float32)
    with torch.no_grad():
        for oid, (name, pos, neg) in OBJ.items():
            proc.add_inputs_to_inference_session(
                inference_session=sess, frame_idx=0, obj_ids=oid,
                input_points=[[[list(map(float, p)) for p in pos + neg]]], input_labels=[[[1] * len(pos) + [0] * len(neg)]])
            model(inference_session=sess, frame_idx=0)
        masks = np.zeros((len(frames), len(OBJ), H, W), bool)
        for out in model.propagate_in_video_iterator(sess):
            m = proc.post_process_masks([out.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            masks[out.frame_idx] = m[:, 0].cpu().numpy()
    names = [v[0] for v in OBJ.values()]
    print('segmented', len(frames), 'frames in', round(time.time() - t0, 1), 's; mean area (px):',
          {n: int(masks[:, i].sum((1, 2)).mean()) for i, n in enumerate(names)})
    np.savez_compressed('real3d/sam_masks.npz', masks=np.packbits(masks, axis=-1), shape=np.array(masks.shape), names=np.array(names))
    return frames, masks


def load():
    z = np.load('real3d/sam_masks.npz')
    shp = tuple(z['shape'])
    return np.unpackbits(z['masks'], axis=-1)[..., :shp[-1]].astype(bool), list(z['names'])


if __name__ == '__main__':
    frames, masks = main()
    SP = "/private/tmp/claude-501/-Users-grandpa-phai-medical-agentic-sim/bbf1e83e-cf52-4deb-8485-0368dcb5b01a/scratchpad"
    cols = np.array([[90, 200, 90], [0, 220, 255], [255, 255, 255], [200, 60, 220], [255, 160, 60], [255, 60, 60]], float)
    tiles = []
    for k in (0, 50, 100, 150):
        v = frames[k].astype(float)
        for i in range(masks.shape[1]):
            v[masks[k, i]] = 0.45 * v[masks[k, i]] + 0.55 * cols[i]
        tiles.append(v.astype(np.uint8))
    imageio.imwrite(SP + '/sam.png', np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)[::2, ::2])
