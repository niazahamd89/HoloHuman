"""Assemble a four-head demo checkpoint from the per-benchmark checkpoints of Table 1.

    python tools/merge_checkpoints.py --trunk output/main_coco_w32/final_state.pth \
        --heads part=output/main_mhp2_w32/final_state.pth mesh=output/main_mesh_w32/final_state.pth \
        --out model_files/holohuman_w32.pth

The backbone, center head, IIA, GFD and the pose / seg heads come from `--trunk`; the listed
heads are copied from their own checkpoints.  This is the "Timing only / all four" configuration
of Table 1 (one trunk with all heads attached) used by the demo, app.py and the timing scripts;
it does not correspond to an accuracy result of the paper, where every benchmark is evaluated
with its own checkpoint.
"""
import argparse

import torch


def load(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trunk', required=True)
    ap.add_argument('--heads', nargs='*', default=[], help='<task>=<checkpoint> pairs (pose, seg, part, mesh)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    merged = load(a.trunk)
    for item in a.heads:
        task, path = item.split('=', 1)
        src = load(path)
        prefix = f'heads.{task}.'
        n = 0
        for k, v in src.items():
            if k.startswith(prefix):
                merged[k] = v
                n += 1
        print(f'{task}: {n} tensors from {path}')
    torch.save({'state_dict': merged}, a.out)
    print(f'written {a.out} ({len(merged)} tensors)')


if __name__ == '__main__':
    main()
