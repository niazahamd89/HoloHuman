"""Training-time forward of HoloHuman (Section 3.3): the m instances of every image are
instantiated at the ground-truth centers r^(i)*, the IIA loss is computed from the center
heatmap, the descriptors of the image and of its horizontal flip (positives, Algorithm 3)
and the attention matrix, and the task losses are computed from the outputs of the heads
attached to the trunk (cfg.MODEL.HEADS.TASKS).  Only the loss terms of the annotated tasks
are active (Section 4, "Training Data and Checkpoints").

Targets (one dict per image, produced by humvis.datasets; all at the output resolution
H x W = INPUT_SIZE / 4):
    'centers'        (m, 2) float, ground-truth centers (x, y) in cells
    'center_heatmap' (1, H, W) float, P* (Gaussian of sigma_c at every center)
    'valid'          (1, H, W) float, 0 inside crowd / ignore regions (negatives not penalized)
    'sigmas'         (m,) float, per-person sigma_k of Eq. 22 in cells
    'keypoints'      (m, K, 3) float, (x, y, visibility) in cells          [pose]
    'masks'          (m, H, W) uint8, binary instance masks                 [seg]
    'parts'          (m, H, W) uint8, part labels, 0 = background, 255 = ignore   [part]
    'smpl_pose'      (m, 72), 'smpl_shape' (m, 10), 'has_smpl' (m,) bool,
    'joints_2d'      (m, 14, 3) normalized image coordinates in [-1, 1] + visibility   [mesh]
"""
import torch
from torch import nn

from .iia import pool_instance_descriptors
from .heads import weak_perspective_projection
from .losses import (focal_loss, render_gaussians, contrastive_loss, interaction_regularization,
                     dice_loss, part_loss, mesh_loss)

# H36M 17-joint order (regressed with J_regressor_h36m) -> 14 LSP joints
# (R ankle, R knee, R hip, L hip, L knee, L ankle, R wrist, R elbow, R shoulder,
#  L shoulder, L elbow, L wrist, neck, head top)
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]


class HoloHumanTrainer(nn.Module):
    """Wraps a HoloHuman model and returns the loss terms of Eqs. 16-26 for a batch."""

    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.tasks = model.tasks
        self.alpha, self.beta = cfg.LOSS.FOCAL_ALPHA, cfg.LOSS.FOCAL_BETA
        self.use_center_focal = bool(cfg.LOSS.CENTER_FOCAL)
        self.use_contrastive = bool(cfg.LOSS.CONTRASTIVE)
        self.use_interaction = bool(cfg.LOSS.INTERACTION)
        self.lambda_I = float(cfg.LOSS.INTERACTION_WEIGHT)
        self.tau = float(cfg.LOSS.TEMPERATURE)
        self.task_weights = {'pose': cfg.LOSS.POSE_WEIGHT, 'seg': cfg.LOSS.SEG_WEIGHT,
                             'part': cfg.LOSS.PART_WEIGHT, 'mesh': cfg.LOSS.MESH_WEIGHT}
        self.lambda_iia = float(cfg.LOSS.IIA_WEIGHT)
        self.mesh_weights = (cfg.LOSS.MESH_POSE_WEIGHT, cfg.LOSS.MESH_SHAPE_WEIGHT, cfg.LOSS.MESH_REPROJ_WEIGHT)
        self.chunk = max(int(cfg.TRAIN.INSTANCE_CHUNK), 1)
        if 'mesh' in self.tasks and model.heads.smpl is None:
            raise FileNotFoundError('training the mesh head needs the SMPL model files '
                                    f'({cfg.MODEL.SMPL_MODEL}, {cfg.MODEL.SMPL_H36M_REGRESSOR})')

    # ------------------------------------------------------------------
    def task_losses(self, out, F_prime, t, rows):
        """Per-instance task distances dist_t (Eq. 20) for the instance rows of one image."""
        losses = {}
        H, W = t['center_heatmap'].shape[-2:]        # head outputs are on the full map (windows pasted back)
        if 'pose' in self.tasks:
            kpts = t['keypoints'][rows]
            hm_target, num_kpts = render_gaussians(kpts, t['sigmas'][rows], H, W)      # Eq. 22
            per = []
            for i in range(kpts.shape[0]):                                            # Eq. 21
                per.append(focal_loss(out['heatmaps'][i], hm_target[i], self.alpha, self.beta,
                                      num_pos=num_kpts[i], weight=t.get('valid')))
            losses['pose'] = torch.stack(per)
        if 'seg' in self.tasks:
            losses['seg'] = dice_loss(out['masks'][:, 0], t['masks'][rows])             # Eq. 23
        if 'part' in self.tasks:
            losses['part'] = part_loss(out['part_logits'], t['parts'][rows])            # Eq. 24
        if 'mesh' in self.tasks:
            theta = out['theta']
            _, _, joints_h36m = self.model.heads.smpl_forward(theta)
            joints14 = joints_h36m[:, H36M_TO_J14]
            losses['mesh'] = mesh_loss(theta, joints14, weak_perspective_projection, {
                'pose': t['smpl_pose'][rows], 'shape': t['smpl_shape'][rows],
                'has_smpl': t['has_smpl'][rows], 'joints_2d': t['joints_2d'][rows]},
                self.mesh_weights)                                                     # Eq. 25
        return losses

    # ------------------------------------------------------------------
    def forward(self, images, targets):
        """images: (B, 3, S, S) normalized; targets: list of B dicts (see module docstring).
        Returns a dict of scalar losses: 'total' (Eq. 16) and the individual terms."""
        model = self.model
        device = images.device
        B = images.shape[0]
        feats = model.backbone(images)                                                 # Eq. 1
        H, W = feats.shape[-2:]

        # flipped-view pass for the contrastive positives (Algorithm 3), only for the
        # images with at least two annotated persons (otherwise N_i is empty and C_i = 0)
        flip_rows = [b for b in range(B) if self.use_contrastive and model.iia.enabled
                     and targets[b]['centers'].shape[0] >= 2]
        feats_flip = model.backbone(images[flip_rows].flip(-1)) if flip_rows else None

        sums = {'center': 0.0, 'contrastive': 0.0, 'interaction': 0.0}
        for task in self.tasks:
            sums[task] = 0.0
        n_images = 0
        for b in range(B):
            t = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in targets[b].items()}
            centers = t['centers']
            m = centers.shape[0]
            if m == 0:
                # no annotated person: only the center heatmap is supervised (all negatives)
                P = model.iia.center_head(feats[b:b + 1])
                if self.use_center_focal:
                    sums['center'] = sums['center'] + focal_loss(P, t['center_heatmap'][None],
                                                                 self.alpha, self.beta, num_pos=1,
                                                                 weight=t.get('valid'))
                n_images += 1
                continue

            inst = model.iia(feats[b:b + 1], centers=centers)                           # Algorithm 1 at r*
            if self.use_center_focal:
                sums['center'] = sums['center'] + focal_loss(
                    inst['center_heatmap'], t['center_heatmap'][None], self.alpha, self.beta, num_pos=m,
                    weight=t.get('valid'))
            if b in flip_rows:
                k = flip_rows.index(b)
                mirrored = torch.stack((W - 1 - centers[:, 0], centers[:, 1]), dim=1)      # (W_I - x*, y*)
                f_plus = pool_instance_descriptors(feats_flip[k:k + 1], mirrored, model.iia.pool_kernel)
                F_e_plus, _ = model.iia.interact(f_plus, mirrored, H, W)
                sums['contrastive'] = sums['contrastive'] + contrastive_loss(
                    inst['descriptors'], F_e_plus, self.tau)                               # Eq. 19
            if self.use_interaction and model.iia.enabled:
                sums['interaction'] = sums['interaction'] + interaction_regularization(
                    inst['descriptors'], inst['attention'][0])

            # GFD + heads at the ground-truth centers, in chunks
            per_task = {task: [] for task in self.tasks}
            for start in range(0, m, self.chunk):
                rows = torch.arange(start, min(start + self.chunk, m), device=device)
                fb = feats[b:b + 1].expand(rows.numel(), -1, -1, -1)
                out, F_prime = model.decode(fb, inst['descriptors'][rows], centers[rows])   # Algorithm 2
                for task, val in self.task_losses(out, F_prime, t, rows).items():
                    per_task[task].append(val)
            for task in self.tasks:
                sums[task] = sums[task] + torch.cat(per_task[task]).mean()                 # (1/m) sum_i dist_t
            n_images += 1

        n_images = max(n_images, 1)
        losses = {k: (v / n_images if torch.is_tensor(v) else torch.zeros((), device=device))
                  for k, v in sums.items()}
        l_iia = losses['center'] + losses['contrastive'] + self.lambda_I * losses['interaction']   # Eq. 17
        l_gfd = sum(self.task_weights[task] * losses[task] for task in self.tasks)                  # Eq. 20
        losses['iia'] = l_iia
        losses['gfd'] = l_gfd
        losses['total'] = self.lambda_iia * l_iia + l_gfd                                            # Eq. 16
        return losses
