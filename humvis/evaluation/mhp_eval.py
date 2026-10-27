"""Instance-level human parsing metrics of MHP v1.0 / v2.0 (Li et al., 2017; Zhao et al.,
2018), as defined in Section 4 "Evaluation Metrics":

  AP^p_t   : average precision of person instances at part-averaged IoU threshold t, where the
             IoU of a predicted and a ground-truth person is the mean over the semantic parts
             present in either of them of the per-part mask IoU;
  AP^p_50  : AP^p at t = 0.5;
  AP^p_vol : mean of AP^p over t = 0.1, 0.2, ..., 0.9;
  PCP_50   : for every ground-truth person matched to a prediction at IoU >= 0.5, the fraction
             of its parts whose IoU exceeds 0.5, averaged over all ground-truth persons
             (unmatched persons count 0).

Predictions are consumed image by image (`add`), keeping only the pairwise IoU statistics,
so the memory does not grow with the image size.  The per-image instance predictions can also
be dumped (`dump_dir`) as <id>_<k>.png label maps plus <id>_scores.txt for the official
MHP evaluation toolkit.
"""
import os

import cv2
import numpy as np

THRESHOLDS = np.arange(0.1, 1.0, 0.1)


def part_ious(pred, gt, num_parts):
    """Per-part IoU (num_parts,) between two label maps (0 = background); NaN for parts absent
    in both."""
    ious = np.full(num_parts, np.nan, dtype=np.float32)
    for c in range(1, num_parts + 1):
        p, g = pred == c, gt == c
        union = np.logical_or(p, g).sum()
        if union == 0:
            continue
        ious[c - 1] = np.logical_and(p, g).sum() / float(union)
    return ious


def voc_ap(rec, prec):
    """Area under the interpolated precision-recall curve (VOC 2010+ style)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class MhpAccumulator:
    def __init__(self, num_parts, dump_dir=''):
        self.num_parts = num_parts
        self.images = []          # per image: dict(scores (n_pred,), iou (n_pred, n_gt), part_iou (n_pred, n_gt, P))
        self.n_gt = 0
        self.dump_dir = dump_dir
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    def add(self, image_id, pred_labels, pred_scores, gt_labels):
        """pred_labels: (n_pred, H, W) uint8 label maps; pred_scores (n_pred,);
        gt_labels: list of (H, W) uint8 label maps (one per annotated person)."""
        n_pred, n_gt = len(pred_scores), len(gt_labels)
        self.n_gt += n_gt
        iou = np.zeros((n_pred, n_gt), dtype=np.float32)
        piou = np.full((n_pred, n_gt, self.num_parts), np.nan, dtype=np.float32)
        for i in range(n_pred):
            for j in range(n_gt):
                pi = part_ious(pred_labels[i], gt_labels[j], self.num_parts)
                piou[i, j] = pi
                iou[i, j] = np.nanmean(pi) if np.isfinite(pi).any() else 0.0
        self.images.append({'scores': np.asarray(pred_scores, dtype=np.float32), 'iou': iou, 'part_iou': piou})
        if self.dump_dir:
            for k, (lab, sc) in enumerate(zip(pred_labels, pred_scores)):
                cv2.imwrite(os.path.join(self.dump_dir, f'{image_id}_{k:02d}.png'), lab)
            with open(os.path.join(self.dump_dir, f'{image_id}_scores.txt'), 'w') as f:
                f.write('\n'.join(f'{float(s):.6f}' for s in pred_scores))

    def _match(self, img, t):
        """Greedy matching by decreasing score at IoU threshold t; returns per-prediction
        (tp flag, matched gt index or -1)."""
        order = np.argsort(-img['scores'])
        used = np.zeros(img['iou'].shape[1], dtype=bool)
        tp = np.zeros(len(order), dtype=bool)
        match = -np.ones(len(order), dtype=int)
        for i in order:
            if img['iou'].shape[1] == 0:
                continue
            cand = np.where(~used & (img['iou'][i] >= t))[0]
            if len(cand):
                j = cand[np.argmax(img['iou'][i, cand])]
                used[j] = True
                tp[i], match[i] = True, j
        return tp, match

    def ap_at(self, t):
        scores, tps = [], []
        for img in self.images:
            tp, _ = self._match(img, t)
            scores.append(img['scores'])
            tps.append(tp)
        if self.n_gt == 0 or not scores:
            return 0.0
        scores, tps = np.concatenate(scores), np.concatenate(tps)
        order = np.argsort(-scores)
        tps = tps[order]
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(~tps)
        rec = tp_cum / float(self.n_gt)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        return voc_ap(rec, prec)

    def pcp_at(self, t=0.5):
        total, count = 0.0, 0
        for img in self.images:
            tp, match = self._match(img, t)
            n_gt = img['iou'].shape[1]
            per_gt = np.zeros(n_gt, dtype=np.float32)
            for i in np.where(tp)[0]:
                pi = img['part_iou'][i, match[i]]
                present = np.isfinite(pi)
                per_gt[match[i]] = (pi[present] > t).mean() if present.any() else 0.0
            total += per_gt.sum()
            count += n_gt
        return float(total / max(count, 1))

    def summarize(self):
        aps = {float(round(t, 1)): self.ap_at(t) for t in THRESHOLDS}
        return {'AP_p_50': aps[0.5] * 100.0, 'AP_p_vol': float(np.mean(list(aps.values()))) * 100.0,
                'PCP_50': self.pcp_at(0.5) * 100.0, 'AP_p_per_threshold': {k: v * 100.0 for k, v in aps.items()}}
