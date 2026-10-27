"""Common training / test datasets.

A benchmark is converted into a list of *records* (one per image):

    {'image': path, 'width': W, 'height': H, 'id': image id,
     'persons': [person, ...], 'ignore': [COCO segmentation of crowd regions, ...]}
    person = {'bbox': [x, y, w, h] (pixels),
              'keypoints': (K, 3) array or None,                   pose
              'segmentation': COCO polygon / RLE or None,           seg
              'part': path of the per-instance part-label PNG or None,   part
              'smpl_pose': (72,) or None, 'smpl_shape': (10,) or None,
              'joints_2d': (14, 3) LSP14 pixels + visibility or None}     mesh

`TrainDataset` applies the augmentation of Section 4 and produces the targets described in
training.py (instances instantiated at the ground-truth centers).  `TestDataset` yields the
network input of Section 4 (short side 512, padded to a multiple of 32) plus the meta data
needed to map predictions back to the image.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pycocotools import mask as mask_utils

from ..utils import test_image_process
from .transforms import (sample_augmentation, affine_matrix, apply_matrix, warp_image, scale_matrix,
                         bbox_after_transform, normalize_image)
from .joints import flip_smpl_pose, rotate_global_orient, J14_FLIP_INDEX


def decode_segmentation(segm, height, width):
    """COCO polygon / RLE -> binary mask (H, W) uint8."""
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segm.get('counts'), list):
        rle = mask_utils.frPyObjects(segm, height, width)
    else:
        rle = segm
    return mask_utils.decode(rle).astype(np.uint8)


def person_center(keypoints, bbox):
    """Ground-truth center r*: center of the box of the visible keypoints when keypoints are
    annotated, otherwise the center of the annotated box."""
    if keypoints is not None:
        vis = keypoints[:, 2] > 0
        if vis.any():
            pts = keypoints[vis, :2]
            return (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)


def person_sigma(cfg, area_cells):
    """Adaptive sigma of Eq. 22: the base sigma scaled with the square root of the person
    area (in output cells), clipped to [0.5, 2] x base."""
    factor = np.clip(np.sqrt(max(area_cells, 1.0)) / float(cfg.DATASET.SIGMA_REF), 0.5, 2.0)
    return float(factor)


class TrainDataset(Dataset):
    def __init__(self, cfg, records, tasks, part_flip_index=None):
        self.cfg = cfg
        self.records = [r for r in records if len(r['persons']) > 0 or not cfg.DATASET.FILTER_IMAGE]
        self.tasks = tuple(tasks)
        self.input_size = cfg.DATASET.INPUT_SIZE
        self.output_size = cfg.DATASET.OUTPUT_SIZE
        self.flip_index = list(cfg.DATASET.FLIP_INDEX)
        self.part_flip = np.asarray(part_flip_index, dtype=np.uint8) if part_flip_index else None
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        self._rng, self._rng_owner = None, None

    @property
    def rng(self):
        """Augmentation generator, derived from torch's seed of the current process: in the
        main process this is cfg.SEED (+ rank) set by train.py, in a data-loader worker it is
        the per-worker seed that PyTorch derives from it.  Different SEED values therefore give
        different augmentation sequences, the same SEED repeats them, and every worker draws
        its own sequence."""
        info = torch.utils.data.get_worker_info()
        owner = (os.getpid(), info.id if info is not None else -1)
        if self._rng is None or self._rng_owner != owner:
            self._rng = np.random.default_rng(torch.initial_seed())
            self._rng_owner = owner
        return self._rng

    def __len__(self):
        return len(self.records)

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.load_image(rec['image'])
        h, w = img.shape[:2]
        aug = sample_augmentation(self.rng, self.cfg)
        M = affine_matrix(w, h, self.input_size, aug['scale'], aug['rot'], aug['flip'], aug['tx'], aug['ty'],
                          self.cfg.DATASET.SCALE_TYPE)
        Mo = scale_matrix(M, self.output_size / float(self.input_size))   # pixels -> output cells
        S_in, S_out = self.input_size, self.output_size
        image = warp_image(img, M, S_in)

        persons = []
        for p in rec['persons']:
            bbox_out = bbox_after_transform(p['bbox'], Mo, S_out)
            kpts = None
            if p.get('keypoints') is not None:
                kpts = np.asarray(p['keypoints'], dtype=np.float32).copy()
                kpts[:, :2] = apply_matrix(kpts[:, :2], Mo)
                if aug['flip']:
                    kpts = kpts[self.flip_index]
                inside = (kpts[:, 0] >= 0) & (kpts[:, 0] < S_out) & (kpts[:, 1] >= 0) & (kpts[:, 1] < S_out)
                kpts[:, 2] = np.where(inside, kpts[:, 2], 0)
            center = person_center(kpts, bbox_out)
            if not (0 <= center[0] < S_out and 0 <= center[1] < S_out):
                continue                                                     # person left the crop
            if bbox_out[2] * bbox_out[3] <= 0:
                continue
            persons.append((p, kpts, center, bbox_out))
        persons = persons[:self.max_instances]
        m = len(persons)

        # ---- targets -------------------------------------------------------------
        t = {}
        centers = np.array([c for _, _, c, _ in persons], dtype=np.float32).reshape(m, 2)
        sig_factor = np.array([person_sigma(self.cfg, b[2] * b[3]) for _, _, _, b in persons],
                              dtype=np.float32).reshape(m)
        t['centers'] = torch.from_numpy(centers)
        t['sigmas'] = torch.from_numpy(sig_factor * float(self.cfg.DATASET.SIGMA))
        t['center_heatmap'] = torch.from_numpy(
            render_center_heatmap(centers, sig_factor * float(self.cfg.DATASET.CENTER_SIGMA), S_out))

        valid = np.ones((1, S_out, S_out), dtype=np.float32)
        for segm in rec.get('ignore', []):
            ign = warp_image(decode_segmentation(segm, h, w), Mo, S_out, cv2.INTER_NEAREST)
            valid[0][ign > 0] = 0.0
        t['valid'] = torch.from_numpy(valid)

        if 'pose' in self.tasks:
            K = self.cfg.DATASET.NUM_KEYPOINTS
            kp = np.zeros((m, K, 3), dtype=np.float32)
            for i, (_, kpts, _, _) in enumerate(persons):
                if kpts is not None:
                    kp[i] = kpts
            t['keypoints'] = torch.from_numpy(kp)
        if 'seg' in self.tasks:
            masks = np.zeros((m, S_out, S_out), dtype=np.uint8)
            for i, (p, _, _, _) in enumerate(persons):
                if p.get('segmentation') is not None:
                    masks[i] = warp_image(decode_segmentation(p['segmentation'], h, w), Mo, S_out, cv2.INTER_NEAREST)
                elif p.get('part') is not None:
                    masks[i] = (self.load_part(p['part'], aug['flip']) > 0).astype(np.uint8)
            t['masks'] = torch.from_numpy(masks)
        if 'part' in self.tasks:
            parts = np.full((m, S_out, S_out), 255, dtype=np.uint8)
            for i, (p, _, _, _) in enumerate(persons):
                if p.get('part') is not None:
                    lab = cv2.imread(p['part'], cv2.IMREAD_UNCHANGED)
                    if lab.ndim == 3:
                        lab = lab[:, :, 0]
                    lab = warp_image(lab.astype(np.uint8), Mo, S_out, cv2.INTER_NEAREST)
                    if aug['flip'] and self.part_flip is not None:
                        lab = self.part_flip[lab]
                    parts[i] = lab
            t['parts'] = torch.from_numpy(parts)
        if 'mesh' in self.tasks:
            pose = np.zeros((m, 72), dtype=np.float32)
            shape = np.zeros((m, 10), dtype=np.float32)
            has = np.zeros((m,), dtype=bool)
            j2d = np.zeros((m, 14, 3), dtype=np.float32)
            for i, (p, _, _, _) in enumerate(persons):
                if p.get('smpl_pose') is not None:
                    po = rotate_global_orient(p['smpl_pose'], aug['rot'])
                    if aug['flip']:
                        po = flip_smpl_pose(po)
                    pose[i], shape[i], has[i] = po, np.asarray(p['smpl_shape'], dtype=np.float32), True
                if p.get('joints_2d') is not None:
                    jj = np.asarray(p['joints_2d'], dtype=np.float32).copy()
                    jj[:, :2] = apply_matrix(jj[:, :2], M)                  # input pixels
                    if aug['flip']:
                        jj = jj[J14_FLIP_INDEX]
                    inside = (jj[:, 0] >= 0) & (jj[:, 0] < S_in) & (jj[:, 1] >= 0) & (jj[:, 1] < S_in)
                    jj[:, 2] = np.where(inside, jj[:, 2], 0)
                    jj[:, :2] = jj[:, :2] / (S_in / 2.0) - 1.0                # normalized [-1, 1]
                    j2d[i] = jj
            t['smpl_pose'], t['smpl_shape'] = torch.from_numpy(pose), torch.from_numpy(shape)
            t['has_smpl'], t['joints_2d'] = torch.from_numpy(has), torch.from_numpy(j2d)

        return torch.from_numpy(normalize_image(image)), t, {'id': rec.get('id', idx), 'num_persons': m}

    def load_part(self, path, flip):
        lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return lab if lab.ndim == 2 else lab[:, :, 0]


def render_center_heatmap(centers, sigmas, size):
    """P* (Section 3.4): Gaussian of sigma_c at every ground-truth center, max-merged, with an
    exact 1 at the rounded center cell.  Returns (1, size, size) float32."""
    hm = np.zeros((1, size, size), dtype=np.float32)
    if len(centers) == 0:
        return hm
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    for (cx, cy), s in zip(centers, sigmas):
        cx, cy = float(np.round(cx)), float(np.round(cy))
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * float(s) ** 2))
        hm[0] = np.maximum(hm[0], g)
    return hm


def collate_train(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, targets, metas


class TestDataset(Dataset):
    """Images of an evaluation split with the Section 4 preprocessing."""

    def __init__(self, cfg, records):
        self.records = records
        self.input_size = cfg.DATASET.INPUT_SIZE

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec['image'], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rec['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        padded, padded_size, resized_size, scale = test_image_process(img, self.input_size)
        meta = {'id': rec.get('id', idx), 'index': idx, 'ori_size': img.shape[:2], 'padded_size': padded_size,
                'resized_size': resized_size, 'scale': scale, 'image': rec['image']}
        return torch.from_numpy(normalize_image(padded)), meta


def collate_test(batch):
    assert len(batch) == 1, 'inference runs one image at a time'
    return batch[0][0].unsqueeze(0), batch[0][1]
