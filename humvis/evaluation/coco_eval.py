"""COCO keypoint / instance-segmentation evaluation with pycocotools (Section 4, "Evaluation
Metrics") and CrowdPose OKS-AP with the easy / medium / hard crowd-index splits."""
import copy
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils


class CocoAccumulator:
    """Collects keypoint and segmentation results in the COCO result format."""

    def __init__(self, ann_file, out_dir, sigmas=None, crowd_splits=False):
        self.ann_file = ann_file
        self.out_dir = out_dir
        self.kpt_results, self.seg_results = [], []
        self.sigmas = sigmas
        self.crowd_splits = crowd_splits
        os.makedirs(out_dir, exist_ok=True)

    def add(self, image_id, res):
        if 'poses' in res:
            for kp, sc in zip(res['poses'], res['pose_scores']):
                self.kpt_results.append({'image_id': int(image_id), 'category_id': 1,
                                         'keypoints': kp.reshape(-1).round(2).tolist(), 'score': float(sc)})
        if 'masks' in res:
            for mk, sc in zip(res['masks'], res['mask_scores']):
                rle = mask_utils.encode(np.asfortranarray(mk.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('ascii')
                self.seg_results.append({'image_id': int(image_id), 'category_id': 1, 'segmentation': rle,
                                         'score': float(sc)})

    def _evaluate(self, coco, results, iou_type, img_ids=None):
        if not results:
            return {}
        dt = coco.loadRes(results)
        ev = COCOeval(coco, dt, iou_type)
        if iou_type == 'keypoints' and self.sigmas is not None:
            ev.params.kpt_oks_sigmas = np.asarray(self.sigmas, dtype=np.float32)
        if img_ids is not None:
            ev.params.imgIds = list(img_ids)
        ev.params.catIds = [1]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        names = ['AP', 'AP50', 'AP75', 'APm', 'APl', 'AR', 'AR50', 'AR75', 'ARm', 'ARl'] if iou_type == 'keypoints' \
            else ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        return {k: float(v) for k, v in zip(names, ev.stats)}

    def summarize(self, tag=''):
        metrics = {}
        kp_file = os.path.join(self.out_dir, f'keypoints{tag}.json')
        with open(kp_file, 'w') as f:
            json.dump(self.kpt_results, f)
        if self.seg_results:
            with open(os.path.join(self.out_dir, f'segm{tag}.json'), 'w') as f:
                json.dump(self.seg_results, f)
        coco = COCO(self.ann_file) if os.path.isfile(self.ann_file) else None
        if coco is None or len(coco.getAnnIds()) == 0:            # e.g. COCO test-dev2017 (image_info only)
            print(f'results written to {self.out_dir} (no ground truth available for this split)')
            return metrics
        if self.kpt_results:
            metrics['keypoints'] = self._evaluate(coco, self.kpt_results, 'keypoints')
            if self.crowd_splits:
                for name, lo, hi in (('easy', 0.0, 0.1), ('medium', 0.1, 0.8), ('hard', 0.8, 1.01)):
                    ids = [i for i, im in coco.imgs.items() if lo <= im.get('crowdIndex', 0.0) < hi]
                    if ids:
                        metrics[f'keypoints_{name}'] = self._evaluate(coco, self.kpt_results, 'keypoints', ids)
        has_masks = any('segmentation' in a for a in coco.anns.values())
        if self.seg_results and has_masks:
            metrics['segm'] = self._evaluate(coco, self.seg_results, 'segm')
        elif self.seg_results:
            print('segmentation results written but not evaluated: the ground truth has no masks')
        return metrics
