"""Mesh datasets (Section 4, "3DPW" and "CMU-Panoptic").

Training mixture: MuCo-3DHP, COCO with pseudo-ground-truth SMPL fits, 3DPW train.
Evaluation: 3DPW test, CMU-Panoptic (Haggling, Mafia, Ultimatum, Pizza).

All of them are read from the unified annotation files written by tools/prepare_mesh_data.py:

    data/mesh/<name>.pkl  =  list of records
    record = {'image': path (absolute, or relative to data/), 'width', 'height', 'id',
              'persons': [{'bbox': [x, y, w, h],
                           'joints_2d': (14, 3) LSP14 pixels + visibility,
                           'smpl_pose': (72,) or None, 'smpl_shape': (10,) or None,
                           'gender': 'm' | 'f' | 'n',                    (evaluation)
                           'joints_3d': (14, 3) metres, camera frame, or None   (evaluation)
                          }, ...]}
"""
import os
import pickle

import numpy as np


def load_mesh_records(root, name):
    path = os.path.join(root, 'mesh', f'{name}.pkl')
    with open(path, 'rb') as f:
        records = pickle.load(f)
    for r in records:
        if not os.path.isabs(r['image']):
            r['image'] = os.path.join(root, r['image'])
        r.setdefault('ignore', [])
        for p in r['persons']:
            p.setdefault('keypoints', None)
            p.setdefault('segmentation', None)
            p.setdefault('part', None)
            p.setdefault('smpl_pose', None)
            p.setdefault('smpl_shape', None)
            if p.get('joints_2d') is not None:
                p['joints_2d'] = np.asarray(p['joints_2d'], dtype=np.float32).reshape(14, 3)
    return records


def mesh_train_records(root, names, weights=None):
    """Concatenation of the mixture; a dataset with weight w is repeated round(w) times
    (all weights are 1 in the paper's protocol)."""
    weights = weights or [1.0] * len(names)
    records = []
    for name, w in zip(names, weights):
        recs = load_mesh_records(root, name)
        records.extend(recs * max(int(round(w)), 1))
    return records
