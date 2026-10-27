"""Task-specific heads, Appendix A.2 / Figure 11.  All heads take F'^(i) (C' x H x W,
stride 4), are shared across instances, and run at stride 4:

    H_pose : 3x3 C'->C', ReLU ; 1x1 C'->K   ; sigmoid  -> K keypoint heatmaps H_k^(i)
    H_seg  : 3x3 C'->C', ReLU ; 1x1 C'->1   ; sigmoid  -> binary instance mask M^(i)
    H_part : 3x3 C'->C', ReLU ; 1x1 C'->n+1 ; softmax  -> part map S^(i) (n parts + background)
    H_mesh : GAP -> [feat, Theta] -> FC 1024 -> FC 1024 -> delta Theta, 3 IEF iterations
             Theta^(i) = (theta_pose in R^72 (24 joints, axis-angle), theta_shape in R^10,
                          theta_cam in R^3 (weak-perspective scale and 2D translation))
             -> SMPL forward

cfg.MODEL.HEADS.TASKS selects the heads attached to the trunk (Table 1: one checkpoint per
benchmark, e.g. ['pose', 'seg'] for C-pose/seg); the default attaches all four.
"""
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .smpl import SMPL_layer

NUM_POSE, NUM_SHAPE, NUM_CAM = 72, 10, 3
NUM_THETA = NUM_POSE + NUM_SHAPE + NUM_CAM  # 85
ALL_TASKS = ('pose', 'seg', 'part', 'mesh')


class ConvHead(nn.Module):
    """3x3 convolution (C'->C', ReLU) followed by a 1x1 convolution to `out_channels`."""

    def __init__(self, channels, out_channels, bias_prob=None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.out = nn.Conv2d(channels, out_channels, 1, 1, 0)
        if bias_prob is not None:       # focal-loss heads: sigmoid output ~ bias_prob at initialization
            nn.init.constant_(self.out.bias, -math.log((1.0 - bias_prob) / bias_prob))

    def forward(self, x):
        return self.out(F.relu(self.conv(x), inplace=True))


class MeshHead(nn.Module):
    """Iterative error feedback regressor (Kanazawa et al., 2018) on the pooled F'^(i)."""

    def __init__(self, channels, fc_dim=1024, n_iter=3, init_theta=None):
        super().__init__()
        self.n_iter = n_iter
        self.fc1 = nn.Linear(channels + NUM_THETA, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.out = nn.Linear(fc_dim, NUM_THETA)
        if init_theta is None:
            # rest pose, mean shape, unit scale, zero translation
            init_theta = torch.zeros(NUM_THETA)
            init_theta[NUM_POSE + NUM_SHAPE] = 1.0
        self.register_buffer('init_theta', torch.as_tensor(init_theta, dtype=torch.float32).view(NUM_THETA))

    def forward(self, feats):
        """feats: (n, C', H, W).  Returns Theta: (n, 85)."""
        pooled = feats.mean(dim=(2, 3))                                   # GAP
        theta = self.init_theta.unsqueeze(0).expand(pooled.shape[0], -1).to(pooled.dtype)
        for _ in range(self.n_iter):
            x = torch.cat((pooled, theta), dim=1)
            x = F.relu(self.fc1(x), inplace=True)
            x = F.relu(self.fc2(x), inplace=True)
            theta = theta + self.out(x)
        return theta


def split_theta(theta):
    """Theta (n, 85) -> pose (n, 72), shape (n, 10), cam (n, 3) = (s, t_x, t_y)."""
    return (theta[:, :NUM_POSE], theta[:, NUM_POSE:NUM_POSE + NUM_SHAPE],
            theta[:, NUM_POSE + NUM_SHAPE:])


def weak_perspective_projection(points, cam):
    """Pi_theta_cam (Eq. 25): x_2d = s * (X_xy + t) in normalized image coordinates.
    points: (n, J, 3); cam: (n, 3) = (s, t_x, t_y).  Returns (n, J, 2)."""
    s, t = cam[:, :1, None], cam[:, None, 1:]
    return s * (points[:, :, :2] + t)


def weak_perspective_to_translation(cam, focal, image_size):
    """Perspective translation equivalent to the weak-perspective camera (s, t_x, t_y),
    used only for rendering with a pinhole camera of focal length `focal` (pixels) whose
    principal point is the instance center:  t_z = 2 f / (s * image_size).
    cam: (n, 3); image_size: size (pixels) of the square region the scale s refers to."""
    s = cam[:, 0].abs().clamp(min=1e-3)
    t_z = 2.0 * focal / (s * float(image_size))
    return torch.stack((cam[:, 1], cam[:, 2], t_z), dim=1)


class TaskHeads(nn.Module):
    """H_pose, H_seg, H_part and H_mesh applied to F'^(i) (Algorithm 2, lines 11-15)."""

    def __init__(self, cfg, channels):
        super().__init__()
        self.tasks = tuple(t for t in ALL_TASKS if t in list(cfg.MODEL.HEADS.TASKS))
        assert self.tasks, 'cfg.MODEL.HEADS.TASKS must list at least one of ' + str(ALL_TASKS)
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
        self.num_parts = cfg.DATASET.NUM_PARTS
        self.pose = ConvHead(channels, self.num_keypoints, cfg.MODEL.BIAS_PROB) if 'pose' in self.tasks else None
        self.seg = ConvHead(channels, 1) if 'seg' in self.tasks else None
        self.part = ConvHead(channels, self.num_parts + 1) if 'part' in self.tasks else None

        self.mesh = None
        self.smpl = None
        if 'mesh' in self.tasks:
            init_theta = None
            if cfg.MODEL.HEADS.INIT_THETA and os.path.isfile(cfg.MODEL.HEADS.INIT_THETA):
                init_theta = np.load(cfg.MODEL.HEADS.INIT_THETA).astype(np.float32).reshape(-1)
                assert init_theta.shape[0] == NUM_THETA, init_theta.shape
            self.mesh = MeshHead(channels, cfg.MODEL.HEADS.FC_DIM, cfg.MODEL.HEADS.IEF_ITERS, init_theta)
            # SMPL layer (no learnable parameters); the mesh output is skipped if the files are absent
            if os.path.isfile(cfg.MODEL.SMPL_MODEL):
                if not os.path.isfile(cfg.MODEL.SMPL_H36M_REGRESSOR):
                    raise FileNotFoundError(f'SMPL model found but not the H36M joint regressor '
                                            f'{cfg.MODEL.SMPL_H36M_REGRESSOR}')
                regressor = np.load(cfg.MODEL.SMPL_H36M_REGRESSOR)
                self.smpl = SMPL_layer(cfg.MODEL.SMPL_MODEL, h36m_jregressor=regressor, dtype=torch.float32)
        self.focal = cfg.MODEL.HEADS.FOCAL

    def forward(self, feats):
        """feats: (n, C', H, W) = F'^(i).  Returns a dict with the outputs of the attached heads:
        'heatmaps' (n, K, H, W) sigmoid, 'masks' (n, 1, H, W) sigmoid, 'parts' (n, n_parts+1, H, W)
        softmax and 'part_logits', 'theta' (n, 85)."""
        out = {}
        if self.pose is not None:
            out['heatmaps'] = torch.sigmoid(self.pose(feats))
        if self.seg is not None:
            out['masks'] = torch.sigmoid(self.seg(feats))
        if self.part is not None:
            logits = self.part(feats)
            out['part_logits'] = logits
            out['parts'] = torch.softmax(logits, dim=1)
        if self.mesh is not None:
            out['theta'] = self.mesh(feats)
        return out

    def smpl_forward(self, theta):
        """SMPL forward of the predicted parameters: vertices (n, 6890, 3), joints (n, 24, 3),
        h36m joints (n, 17, 3) regressed from the vertices."""
        pose, shape, _ = split_theta(theta)
        out = self.smpl(pose.contiguous(), shape.contiguous())
        return out.vertices, out.joints, out.joints_from_verts
