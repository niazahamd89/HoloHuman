"""Inference time and throughput per task configuration (Figure 5, Table 9; Section 4.3).

    python time_fig5.py --weights output/coco_w32/final_state.pth [--num-images 100]

Protocol: HRNet-W32, 512 x 512 input, batch size 1, single scale with flip test, fp32, on
COCO val images with the persons detected by the C-pose/seg center head (threshold 0.1,
at most 30); every configuration attaches the listed heads to the same trunk (Table 1,
"Timing only"; heads missing from the checkpoint are randomly initialized, which does not
change their cost).  Latency per configuration is the median over the images of the
per-image median of `--runs` CUDA-synchronized runs.  Writes runtime_per_task.json.
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
from humvis.datasets.coco import coco_test_records
from humvis.datasets.transforms import normalize_image
from humvis.evaluation.timing import time_forward

CONFIGS = {'pose': ['pose'], 'seg': ['seg'], 'part': ['part'], 'mesh': ['mesh'],
           'pose+seg': ['pose', 'seg'], 'all': ['pose', 'seg', 'part', 'mesh']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='')
    ap.add_argument('--weights', default='')
    ap.add_argument('--backbone', default='hrnet32')
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--num-images', type=int, default=100)
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--out', default='runtime_per_task.json')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    a = ap.parse_args()

    base = update_config(get_cfg(), a.cfg or None, a.opts)
    records, _ = coco_test_records(base.DATASET.ROOT, 'val2017')
    records = records[:a.num_images]
    images = []
    for rec in records:
        img = cv2.cvtColor(cv2.imread(rec['image'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_LINEAR)
        images.append(torch.from_numpy(normalize_image(img)).unsqueeze(0))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = {}
    for name in a.configs:
        cfg = base.clone()
        cfg.defrost()
        cfg.MODEL.BACKBONE = a.backbone
        cfg.MODEL.HEADS.TASKS = CONFIGS[name]
        cfg.TEST.FLIP_TEST = True
        cfg.freeze()
        model = HoloHuman(cfg, device)
        if a.weights and os.path.isfile(a.weights):
            load_weights(model, a.weights, strict=False)
        model = model.to(device).eval()
        medians, mems, persons = [], [], []
        for x in images:
            median, mem, _ = time_forward(model, x, runs=a.runs, warmup=3)
            with torch.no_grad():
                out = model(x.to(device))
            medians.append(median)
            mems.append(mem)
            persons.append(out['centers'].shape[0] if out else 0)
        lat = float(np.median(medians))
        results[name] = {'heads': CONFIGS[name], 'latency_ms': lat, 'fps': 1000.0 / lat,
                         'memory_gb': float(np.max(mems)), 'mean_persons': float(np.mean(persons)),
                         'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
        print(f"{name:9s} {lat:6.1f} ms  {1000.0 / lat:5.1f} FPS  {results[name]['memory_gb']:.2f} GB  "
              f"{results[name]['params_M']:.2f} M params  ({np.mean(persons):.1f} persons/img)")
        del model
    with open(a.out, 'w') as f:
        json.dump({'backbone': a.backbone, 'size': a.size, 'runs': a.runs, 'num_images': len(images),
                   'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                   'configs': results}, f, indent=2)
    print(f'written {a.out}')


if __name__ == '__main__':
    main()
