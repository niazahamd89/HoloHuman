"""Benchmark loaders (Section 4).  `build_train_dataset(cfg)` returns the augmented training
set of the benchmark in cfg.DATASET.DATASET with the targets of the heads in
cfg.MODEL.HEADS.TASKS; `build_test_records(cfg)` returns the evaluation images and the
information the evaluator in eval.py needs."""
from .base import TrainDataset, TestDataset, collate_train, collate_test
from .coco import coco_records, coco_test_records, crowdpose_records
from .mhp import mhp_v2_records, mhp_v1_records
from .mesh import mesh_train_records, load_mesh_records
from . import joints


def build_train_dataset(cfg):
    name = cfg.DATASET.DATASET.lower()
    root = cfg.DATASET.ROOT
    tasks = list(cfg.MODEL.HEADS.TASKS)
    part_flip = list(cfg.DATASET.PART_FLIP_INDEX) or None
    if name == 'coco':
        records = coco_records(root, cfg.DATASET.TRAIN, with_masks='seg' in tasks)
    elif name == 'crowdpose':
        records, _ = crowdpose_records(root, cfg.DATASET.TRAIN)
    elif name == 'mhp2':
        records = mhp_v2_records(root, cfg.DATASET.TRAIN)
        part_flip = part_flip or joints.MHP_V2_FLIP_INDEX
    elif name == 'mhp1':
        records = mhp_v1_records(root, cfg.DATASET.TRAIN)
        part_flip = part_flip or joints.MHP_V1_FLIP_INDEX
    elif name == 'mesh':
        records = mesh_train_records(root, list(cfg.DATASET.MESH_TRAIN), list(cfg.DATASET.MESH_TRAIN_WEIGHTS))
    else:
        raise ValueError(f'unknown dataset {name!r}')
    return TrainDataset(cfg, records, tasks, part_flip)


def build_test_records(cfg):
    """Returns (records, info) where info['type'] selects the evaluator and carries the
    annotation file / ground truth needed by it."""
    name = cfg.DATASET.DATASET.lower()
    root = cfg.DATASET.ROOT
    split = cfg.DATASET.TEST
    if name == 'coco':
        records, ann_file = coco_test_records(root, split)
        return records, {'type': 'coco', 'ann_file': ann_file, 'split': split}
    if name == 'crowdpose':
        records, ann_file = crowdpose_records(root, split)
        return records, {'type': 'crowdpose', 'ann_file': ann_file, 'split': split}
    if name == 'mhp2':
        return mhp_v2_records(root, split), {'type': 'mhp', 'num_parts': 58, 'split': split}
    if name == 'mhp1':
        return mhp_v1_records(root, split), {'type': 'mhp', 'num_parts': 18, 'split': split}
    if name == 'mesh':
        test_name = cfg.DATASET.MESH_TEST
        return load_mesh_records(root, test_name), {'type': 'mesh', 'name': test_name}
    raise ValueError(f'unknown dataset {name!r}')
