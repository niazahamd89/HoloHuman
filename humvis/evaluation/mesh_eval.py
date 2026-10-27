"""3D mesh metrics (Section 4, "Evaluation Metrics"): MPJPE, PA-MPJPE and PVE in millimetres
on the 14 LSP joints regressed from the SMPL mesh, after aligning the root (pelvis) of the
prediction to the ground truth; PA-MPJPE after a rigid Procrustes alignment; PVE over the
6,890 vertices after root alignment.  Predictions are matched to ground-truth persons by 2D
bounding-box IoU (threshold 0.5); unmatched ground-truth persons count as misses (reported
separately, they do not enter the averages).  For CMU-Panoptic only MPJPE on the available
joints is computed against the released 3D joints."""
import numpy as np
import torch

from ..smpl import SMPL_layer
from ..training import H36M_TO_J14

J14_ROOT = [2, 3]      # right hip, left hip -> pelvis = midpoint


def box_iou(a, b):
    """a (n, 4), b (m, 4) as x, y, w, h -> (n, m)."""
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(ax0[:, None], bx0[None]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(ay0[:, None], by0[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.clip(union, 1e-9, None)


def procrustes(S1, S2):
    """Rigid alignment (rotation, translation, scale) of S1 (J, 3) to S2 (J, 3)."""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum()
    K = X1.T @ X2
    U, s, Vt = np.linalg.svd(K)
    Z = np.eye(3)
    Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var1
    t = mu2 - scale * (R @ mu1)
    return scale * (R @ S1.T).T + t


def root_align(j14):
    return j14 - j14[J14_ROOT, :].mean(axis=0, keepdims=True)


class GroundTruthSMPL:
    """Gendered SMPL meshes of the 3DPW annotations (neutral fallback with a warning)."""

    def __init__(self, cfg, device):
        import os
        regressor = np.load(cfg.MODEL.SMPL_H36M_REGRESSOR)
        self.models = {}
        for g, path in (('m', cfg.MODEL.SMPL_MALE), ('f', cfg.MODEL.SMPL_FEMALE), ('n', cfg.MODEL.SMPL_MODEL)):
            if os.path.isfile(path):
                self.models[g] = SMPL_layer(path, h36m_jregressor=regressor, dtype=torch.float32).to(device)
        if 'm' not in self.models or 'f' not in self.models:
            print('[WARN] gendered SMPL models not found; ground-truth meshes use the neutral model '
                  '(not the 3DPW protocol)')
        self.device = device

    def __call__(self, pose, shape, gender):
        model = self.models.get(gender, self.models['n'])
        pose = torch.as_tensor(pose, dtype=torch.float32, device=self.device).view(1, 72)
        shape = torch.as_tensor(shape, dtype=torch.float32, device=self.device).view(1, 10)
        out = model(pose, shape)
        verts = out.vertices[0].cpu().numpy()
        j14 = out.joints_from_verts[0].cpu().numpy()[H36M_TO_J14]
        return verts, j14


class MeshAccumulator:
    def __init__(self, cfg, device, match_iou=0.5, with_vertices=True):
        self.gt_smpl = GroundTruthSMPL(cfg, device) if with_vertices else None
        self.match_iou = match_iou
        self.mpjpe, self.pa_mpjpe, self.pve = [], [], []
        self.per_sequence = {}
        self.n_gt, self.n_matched = 0, 0

    def add(self, res, gt_persons, sequence=None):
        """res: predictions of predict_image (needs 'boxes', 'joints14', 'vertices');
        gt_persons: records with 'bbox', and 'smpl_pose'/'smpl_shape'/'gender' or 'joints_3d'."""
        self.n_gt += len(gt_persons)
        if not res or 'joints14' not in res or len(gt_persons) == 0:
            return
        gt_boxes = np.asarray([p['bbox'] for p in gt_persons], dtype=np.float32)
        iou = box_iou(np.asarray(res['boxes'], dtype=np.float32), gt_boxes)
        used = np.zeros(len(gt_persons), dtype=bool)
        order = np.argsort(-res['center_scores'][res['keep']]) if 'keep' in res else np.arange(iou.shape[0])
        for i in order:
            cand = np.where(~used & (iou[i] >= self.match_iou))[0]
            if len(cand) == 0:
                continue
            j = cand[np.argmax(iou[i, cand])]
            used[j] = True
            self.n_matched += 1
            gt = gt_persons[j]
            pred_j = res['joints14'][i].astype(np.float64)
            if gt.get('smpl_pose') is not None and self.gt_smpl is not None:
                gt_verts, gt_j = self.gt_smpl(gt['smpl_pose'], gt['smpl_shape'], gt.get('gender', 'n'))
                gt_j, gt_verts = gt_j.astype(np.float64), gt_verts.astype(np.float64)
                valid = np.ones(14, dtype=bool)
                pred_root = pred_j[J14_ROOT].mean(0)
                gt_root = gt_j[J14_ROOT].mean(0)
                pve = np.linalg.norm((res['vertices'][i] - pred_root) - (gt_verts - gt_root), axis=1).mean()
                self.pve.append(pve * 1000.0)
            else:
                gt_j = np.asarray(gt['joints_3d'], dtype=np.float64)
                valid = np.asarray(gt.get('joints_3d_valid', np.ones(14, dtype=bool)), dtype=bool)
            pa = root_align(pred_j)[valid]
            ga = root_align(gt_j)[valid] if valid[J14_ROOT].all() else (gt_j - gt_j[valid].mean(0))[valid]
            if not valid[J14_ROOT].all():
                pa = (pred_j - pred_j[valid].mean(0))[valid]
            mpjpe = np.linalg.norm(pa - ga, axis=1).mean() * 1000.0
            pa_mpjpe = np.linalg.norm(procrustes(pa, ga) - ga, axis=1).mean() * 1000.0
            self.mpjpe.append(mpjpe)
            self.pa_mpjpe.append(pa_mpjpe)
            if sequence is not None:
                self.per_sequence.setdefault(sequence, []).append(mpjpe)

    def summarize(self):
        out = {'MPJPE': float(np.mean(self.mpjpe)) if self.mpjpe else float('nan'),
               'PA-MPJPE': float(np.mean(self.pa_mpjpe)) if self.pa_mpjpe else float('nan'),
               'PVE': float(np.mean(self.pve)) if self.pve else float('nan'),
               'matched': self.n_matched, 'gt_persons': self.n_gt,
               'miss_rate': 1.0 - self.n_matched / max(self.n_gt, 1)}
        if self.per_sequence:
            out['MPJPE_per_sequence'] = {k: float(np.mean(v)) for k, v in self.per_sequence.items()}
        return out
