"""Evaluate a HoloHuman checkpoint on a benchmark with the metrics of Section 4.

    python eval.py --cfg configs/main/coco_w48.yaml --weights output/coco_w48/final_state.pth
    python eval.py --cfg configs/main/coco_w48.yaml --weights ... DATASET.TEST test-dev2017   # json for the server
    python eval.py --cfg configs/eval/crowdpose_cross_w48.yaml --weights output/coco_w48/final_state.pth
    python eval.py --cfg configs/main/mhp2_w48.yaml --weights ...                             # MHP v2.0 val
    python eval.py --cfg configs/eval/mhp1_cross_w48.yaml --weights output/mhp2_w48/final_state.pth
    python eval.py --cfg configs/main/mesh_w48.yaml --weights ...                             # 3DPW test
    python eval.py --cfg configs/eval/panoptic_w48.yaml --weights output/mesh_w48/final_state.pth

Protocol (Section 4): shorter side resized to 512, single scale with flip test, centers with
confidence threshold 0.1 and at most 30 per image, OKS-NMS on the poses.  Metrics: COCO /
CrowdPose OKS-AP (pycocotools, CrowdPose sigmas and crowd-index splits), COCO mask AP of
the person class, MHP AP^p_50 / AP^p_vol / PCP_50, 3DPW MPJPE / PA-MPJPE / PVE with box-IoU
matching, CMU-Panoptic MPJPE per sequence.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from humvis import get_cfg, load_weights
from humvis.config import update_config
from humvis.holohuman import HoloHuman
from humvis.datasets import build_test_records, TestDataset, collate_test, joints
from humvis.datasets.mhp import load_part_png
from humvis.evaluation import predict_image, CocoAccumulator, MhpAccumulator, MeshAccumulator


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--weights', default='', help='checkpoint (overrides TEST.MODEL_FILE)')
    ap.add_argument('--out', default='', help='output directory (default: output/<cfg>/eval_<split>)')
    ap.add_argument('--max-images', type=int, default=0, help='debug: evaluate only the first N images')
    ap.add_argument('--dump-parts', action='store_true', help='write MHP instance predictions as PNGs')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = update_config(get_cfg(), args.cfg, args.opts)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = HoloHuman(cfg, device)
    weights = args.weights or cfg.TEST.MODEL_FILE
    if weights and os.path.isfile(weights):
        load_weights(model, weights, strict=True)
    else:
        print(f'[WARN] no checkpoint at {weights!r}: evaluating random weights')
    model = model.to(device).eval()

    records, info = build_test_records(cfg)
    if args.max_images:
        records = records[:args.max_images]
    name = cfg.CFG_NAME or os.path.splitext(os.path.basename(args.cfg))[0]
    out_dir = args.out or os.path.join(cfg.OUTPUT_DIR, name, f'eval_{cfg.DATASET.DATASET}_{cfg.DATASET.TEST}')
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(TestDataset(cfg, records), batch_size=1, shuffle=False, num_workers=cfg.WORKERS,
                        collate_fn=collate_test)

    # cross-dataset conversions (Section 4): COCO keypoints -> CrowdPose, MHP v2.0 labels -> v1.0
    converter, part_lut = None, None
    if info['type'] == 'crowdpose' and model.heads.num_keypoints == 17:
        converter = joints.coco_to_crowdpose
        print('COCO-trained pose head evaluated on CrowdPose: 17 -> 14 keypoint conversion (Table 11 ‡)')
    if info['type'] == 'mhp' and info['num_parts'] == 18 and model.heads.num_parts == 58:
        part_lut = joints.MHP_V2_TO_V1_LUT if not cfg.TEST.PART_MAPPING else np.load(cfg.TEST.PART_MAPPING)
        print('MHP v2.0-trained part head evaluated on MHP v1.0: 58 -> 18 label mapping (Table 14 ‡)')

    if info['type'] in ('coco', 'crowdpose'):
        sigmas = joints.CROWDPOSE_SIGMAS if info['type'] == 'crowdpose' else cfg.TEST.OKS_SIGMAS
        acc = CocoAccumulator(info['ann_file'], out_dir, sigmas, crowd_splits=(info['type'] == 'crowdpose'))
    elif info['type'] == 'mhp':
        acc = MhpAccumulator(info['num_parts'], dump_dir=os.path.join(out_dir, 'parts') if args.dump_parts else '')
    else:
        acc = MeshAccumulator(cfg, device, cfg.TEST.MATCH_IOU, with_vertices=(info['name'] != 'panoptic'))

    t0 = time.time()
    with torch.no_grad():
        for n, (image, meta) in enumerate(loader):
            res = predict_image(model, cfg, image, meta, converter, part_lut)
            rec = records[meta['index']]
            if info['type'] in ('coco', 'crowdpose'):
                if res:
                    acc.add(meta['id'], res)
            elif info['type'] == 'mhp':
                gts = [load_part_png(p['part']) for p in rec['persons']]
                preds = res.get('parts', np.zeros((0,) + tuple(meta['ori_size']), dtype=np.uint8))
                acc.add(meta['id'], preds, res.get('part_scores', np.zeros(0)), gts)
            else:
                acc.add(res, rec['persons'], rec.get('sequence'))
            if n % 100 == 0:
                print(f'[{n}/{len(loader)}] {(time.time() - t0) / (n + 1):.3f} s/img')

    metrics = acc.summarize()
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'metrics written to {os.path.join(out_dir, "metrics.json")}')


if __name__ == '__main__':
    main()
