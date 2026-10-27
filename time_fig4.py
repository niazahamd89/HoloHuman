"""Efficiency versus the number of persons (Figure 4, Section 4.1.3).

    python time_fig4.py --weights model_files/holohuman_w32.pth [--m 1 5 10 20 30] [--num-images 1]

Protocol: HRNet-W32 with all four heads, 512 x 512 images from the CrowdPose test set, m
person centers instantiated from the CrowdPose annotations (padded with grid centers when
the image has fewer), batch size 1, single scale with flip test, fp32, RTX 3090; latency is
the median of 100 CUDA-synchronized runs after warm-up and memory the peak allocation.
Writes efficiency_vs_m.json (m, latency ms, FPS, memory GB), the input of
tools/make_efficiency_fig.py.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from humvis import get_cfg, load_weights
from humvis.config import update_config
from humvis.holohuman import HoloHuman
from humvis.datasets.coco import crowdpose_records
from humvis.datasets.base import person_center
from humvis.datasets.transforms import normalize_image
from humvis.evaluation.timing import time_forward, pad_centers


def square_image(record, size):
    img = cv2.cvtColor(cv2.imread(record['image'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    sx, sy = size / float(w), size / float(h)
    centers = []
    for p in record['persons']:
        c = person_center(p['keypoints'], p['bbox'])
        centers.append([c[0] * sx, c[1] * sy])
    return img, np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='')
    ap.add_argument('--weights', default='')
    ap.add_argument('--backbone', default='hrnet32')
    ap.add_argument('--m', type=int, nargs='+', default=[1, 5, 10, 20, 30])
    ap.add_argument('--num-images', type=int, default=1, help='CrowdPose test images per m (medians are averaged)')
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--out', default='efficiency_vs_m.json')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cfg = update_config(get_cfg(), a.cfg or None, a.opts)
    cfg.defrost()
    cfg.MODEL.BACKBONE = a.backbone
    cfg.MODEL.HEADS.TASKS = ['pose', 'seg', 'part', 'mesh']
    cfg.TEST.FLIP_TEST = True
    cfg.TEST.MAX_INSTANCES = max(cfg.TEST.MAX_INSTANCES, max(a.m))
    cfg.freeze()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = HoloHuman(cfg, device)
    if a.weights and os.path.isfile(a.weights):
        load_weights(model, a.weights, strict=False)       # timing only: heads absent from the checkpoint stay random
    model = model.to(device).eval()

    records, _ = crowdpose_records(cfg.DATASET.ROOT, 'test')
    records = sorted(records, key=lambda r: -len(r['persons']))     # crowded images first
    stride = cfg.DATASET.INPUT_SIZE // cfg.DATASET.OUTPUT_SIZE
    results = []
    for m in a.m:
        lat, mem = [], []
        for rec in records[:a.num_images]:
            img, centers = square_image(rec, a.size)
            x = torch.from_numpy(normalize_image(img)).unsqueeze(0)
            cells = pad_centers((centers - stride / 2.0) / stride, m, a.size // stride, a.size // stride)
            cells = cells.clamp(0, a.size // stride - 1).to(device)
            median, peak, _ = time_forward(model, x, centers=cells, runs=a.runs)
            lat.append(median)
            mem.append(peak)
        row = {'m': m, 'latency_ms': float(np.mean(lat)), 'fps': 1000.0 / float(np.mean(lat)),
               'memory_gb': float(np.mean(mem))}
        results.append(row)
        print(f"m={m:3d}: {row['latency_ms']:7.1f} ms  {row['fps']:5.1f} FPS  {row['memory_gb']:.2f} GB")
    with open(a.out, 'w') as f:
        json.dump({'backbone': a.backbone, 'size': a.size, 'flip_test': True, 'runs': a.runs,
                   'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                   'rows': results}, f, indent=2)
    print(f'written {a.out}')


if __name__ == '__main__':
    main()
