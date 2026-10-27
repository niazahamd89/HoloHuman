"""Computational breakdown of HoloHuman (reproduces Table 7 of the paper).

    python profile_cost.py [--backbone hrnet32|hrnet48] [--m 1 5] [--size 512] [--no-smpl] [--cfg <yaml>]

Counts convolution, linear and matrix-multiplication MACs with torch's FlopCounterMode
(1 MAC = 2 FLOPs) for a single image view and `m` persons instantiated at given centers;
BN, activations, pooling, softmax, NMS, bilinear sampling and element-wise products are not
counted, as in Table 7.  With `--cfg` the architecture switches of an ablation configuration
are applied (Params / GMACs columns of Table 4).  Run from the directory that contains the
`humvis` package and `model_files/` (the SMPL files are optional).
"""
import argparse
import os

import torch
from torch.utils.flop_counter import FlopCounterMode

from humvis import get_cfg
from humvis.config import update_config
from humvis.holohuman import HoloHuman

ROWS = [
    ('backbone',                  'HRNet backbone + fusion to F'),
    ('iia.center_head',           'Center head H_center'),
    ('iia.attention',             'IIA: W_Q, W_K, W_V, W_O and QK^T, I V'),
    ('iia.pos_mlp',               'IIA: absolute PE (Table 6 only)'),
    ('gfd.decoder',               'GFD decoder W_g'),
    ('gfd.channel_attention',     'GFD A_c (SE)'),
    ('gfd.spatial_attention',     'GFD A_s'),
    ('gfd.W_f',                   'GFD fusion W_f'),
    ('heads.pose',                'Pose head'),
    ('heads.seg',                 'Seg head'),
    ('heads.part',                'Part head'),
    ('heads.mesh',                'Mesh head (IEF x3)'),
    ('heads.smpl',                'SMPL forward (LBS)'),
]


def count(model, x, m):
    H = W = x.shape[-1] // 4
    g = int(m ** 0.5) + 1
    centers = torch.tensor([[(i % g + 1) * W // (g + 1), (i // g + 1) * H // (g + 1)] for i in range(m)],
                           dtype=torch.float32)
    fc = FlopCounterMode(display=False, depth=None)
    with torch.no_grad(), fc:
        model(x, centers=centers)
    by_module = {}
    for name, ops in fc.get_flop_counts().items():
        by_module[name] = sum(ops.values()) / 2.0          # FLOPs -> MACs
    return by_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='', help='optional configuration (ablation switches, heads)')
    ap.add_argument('--backbone', default='')
    ap.add_argument('--m', type=int, nargs='+', default=[1, 5])
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--no-smpl', action='store_true')
    ap.add_argument('--all-heads', action='store_true', help='attach all four heads regardless of the config')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cfg = update_config(get_cfg(), a.cfg or None, a.opts)
    cfg.defrost()
    if a.backbone:
        cfg.MODEL.BACKBONE = a.backbone
    if a.all_heads or not a.cfg:
        cfg.MODEL.HEADS.TASKS = ['pose', 'seg', 'part', 'mesh']   # Table 7 / Table 4 count all four heads
    cfg.MODEL.DEVICE = 'cpu'
    cfg.TEST.FLIP_TEST = False
    cfg.MODEL.PRETRAINED = ''
    if a.no_smpl:
        cfg.MODEL.SMPL_MODEL = ''
    cfg.freeze()
    model = HoloHuman(cfg).eval()
    x = torch.randn(1, 3, a.size, a.size)

    params = {}
    for pname, p in model.named_parameters():
        for key, _ in ROWS:
            if pname.startswith(key + '.'):
                params[key] = params.get(key, 0) + p.numel()
    runs = {m: count(model, x, m) for m in a.m}
    name = 'HoloHuman'
    print(f'{name}, {cfg.MODEL.BACKBONE}, {a.size}x{a.size}, single view, C={model.backbone.out_channels}, '
          f"C'={model.gfd.out_channels}, heads={list(model.tasks)}"
          + (f', d={model.iia.attention.d}' if model.iia.attention is not None else ', IIA disabled')
          + (f', window={model.gfd.window}' if model.gfd.window else ''))
    header = f"{'Component':40s}" + ''.join(f'  m={m:<3d} GMACs' for m in a.m) + '   Params (M)'
    print(header)
    for key, label in ROWS:
        if key not in params and all(runs[m].get(name + '.' + key, 0.0) == 0 for m in a.m):
            continue
        cells = ''.join(f'{runs[m].get(name + "." + key, 0.0) / 1e9:13.4f}' for m in a.m)
        print(f'{label:40s}{cells}{params.get(key, 0) / 1e6:13.3f}')
    for m in a.m:
        print(f'TOTAL m={m}: {runs[m].get(name, 0.0) / 1e9:.2f} GMACs')
    print(f'TOTAL params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')


if __name__ == '__main__':
    main()
