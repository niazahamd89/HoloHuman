"""Loss functions of HoloHuman (Sections 3.3-3.4).

    L_total = lambda_IIA * L_IIA + L_GFD                                           (Eq. 16)
    L_IIA   = FL(P, P*) + (1/m) sum_i C_i + (lambda_I/m) sum_ij I_ij ||F_e_i - F_e_j||^2   (Eq. 17)
    FL      = focal loss with alpha = 2, beta = 4, normalized by the N positives     (Eq. 18)
    C_i     = InfoNCE between the descriptor of person i and its flipped-view positive,
              against the other m-1 persons of the image, temperature tau            (Eq. 19)
    L_GFD   = sum_t lambda_t (1/m) sum_i dist_t(T_t^(i), T_t^(i)*)                   (Eq. 20)
    dist_pose = FL on keypoint heatmaps (Eq. 21), N = annotated keypoints
    dist_seg  = Dice loss (Eq. 23)
    dist_part = pixel-wise cross-entropy (Eq. 24)
    dist_mesh = ||theta_pose - *||^2 + ||theta_shape - *||^2 + ||Pi(J_3D) - J_2D*||_1  (Eq. 25);
                only the reprojection term for persons without SMPL annotations
    weights: lambda_IIA = lambda_pose = lambda_seg = lambda_part = 1, lambda_mesh = 0.1 (Eq. 26)
"""
import torch
import torch.nn.functional as F


def focal_loss(pred, target, alpha=2.0, beta=4.0, num_pos=None, eps=1e-6, weight=None):
    """FL(Y, Y*) of Eq. 18 for one tensor pair (any shape, pred in (0, 1) after the sigmoid).
    Positive locations are those with target == 1; the sum is divided by the number of
    positives N (`num_pos`, or counted from the target), at least 1.  `weight` (same shape
    or broadcastable) multiplies the negative term, e.g. 0 inside crowd regions."""
    pred = pred.clamp(eps, 1.0 - eps)
    pos = target.eq(1).to(pred.dtype)
    neg = 1.0 - pos
    if weight is not None:
        neg = neg * weight.to(pred.dtype)
    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - target, beta) * neg
    if num_pos is None:
        num_pos = pos.sum()
    num_pos = torch.as_tensor(num_pos, dtype=pred.dtype, device=pred.device).clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def render_gaussians(keypoints, sigmas, height, width):
    """Ground-truth heatmaps H_k* (Eq. 22): keypoints (m, K, 3) in output cells with
    visibility in the last column (> 0 = annotated), sigmas (m,) in cells.
    Returns targets (m, K, H, W) with exact 1 at the (rounded) keypoint cell, and the
    per-instance count of annotated keypoints (m,)."""
    m, K, _ = keypoints.shape
    device, dtype = keypoints.device, torch.float32
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    kx = keypoints[:, :, 0].round()[:, :, None, None]
    ky = keypoints[:, :, 1].round()[:, :, None, None]
    vis = (keypoints[:, :, 2] > 0).to(dtype)
    s2 = (2.0 * sigmas.to(dtype) ** 2)[:, None, None, None]
    g = torch.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / s2)
    inside = ((kx >= 0) & (kx < width) & (ky >= 0) & (ky < height)).to(dtype)
    g = g * vis[:, :, None, None] * inside
    return g, (vis * inside[:, :, 0, 0]).sum(dim=1)


def contrastive_loss(anchor, positive, tau=0.05):
    """(1/m) sum_i C_i of Eqs. 17 and 19 (Algorithm 3).  anchor, positive: (m, C) descriptors
    of the same m persons from the image and its horizontal flip (positive[i] is the
    mirrored-center descriptor of person i); the negatives of i are the other anchors
    j != i.  Returns 0 for m < 2 (N_i empty)."""
    m = anchor.shape[0]
    if m < 2:
        return anchor.sum() * 0.0
    a = F.normalize(anchor, dim=1)
    p = F.normalize(positive, dim=1)
    pos = (a * p).sum(dim=1) / tau                                   # (m,)
    neg = torch.matmul(a, a.t()) / tau                               # (m, m)
    eye = torch.eye(m, device=a.device, dtype=torch.bool)
    neg = neg.masked_fill(eye, float('-inf'))                        # the anchor is never compared with itself
    logits = torch.cat((pos[:, None], neg), dim=1)                   # positive + (m-1) negatives
    return -F.log_softmax(logits, dim=1)[:, 0].mean()


def interaction_regularization(descriptors, attention):
    """(1/m) sum_i sum_j I_ij ||F̄_i - F̄_j||^2 of Eq. 17 (without lambda_I);
    descriptors (m, C), attention I (m, m).  F̄ is the L2-normalized descriptor."""
    m = descriptors.shape[0]
    if m < 2:
        return descriptors.sum() * 0.0
    nd = F.normalize(descriptors, dim=1)
    d2 = (nd[:, None, :] - nd[None, :, :]).pow(2).sum(dim=-1)          # (m, m)
    return (attention * d2).sum() / m


def dice_loss(pred, target, eps=1e-6):
    """DL(M, M*) of Eq. 23 per instance; pred, target: (m, H, W).  Returns (m,)."""
    pred = pred.flatten(1)
    target = target.flatten(1).to(pred.dtype)
    inter = (pred * target).sum(dim=1)
    denom = pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1)
    return 1.0 - 2.0 * inter / (denom + eps)


def part_loss(logits, target, ignore_index=255):
    """PL(M_p, M_p*) of Eq. 24 per instance: mean pixel-wise cross-entropy.
    logits: (m, n+1, H, W); target: (m, H, W) long with 0 = background.  Returns (m,)."""
    ce = F.cross_entropy(logits, target.long(), ignore_index=ignore_index, reduction='none')
    valid = (target != ignore_index).to(ce.dtype)
    return (ce * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1.0)


def mesh_loss(theta, joints_3d, projection_fn, targets, weights=(1.0, 1.0, 1.0)):
    """dist_mesh of Eq. 25 per instance.
    theta: (m, 85) predicted (pose 72, shape 10, cam 3); joints_3d: (m, J, 3) joints regressed
    from the predicted mesh (root-relative); projection_fn(points, cam) -> (m, J, 2);
    targets: dict with 'pose' (m, 72), 'shape' (m, 10), 'has_smpl' (m,) bool,
    'joints_2d' (m, J, 3) normalized image coordinates in [-1, 1] with visibility.
    Returns (m,)."""
    from .heads import split_theta
    pose, shape, cam = split_theta(theta)
    has_smpl = targets['has_smpl'].to(theta.dtype)
    l_pose = (pose - targets['pose']).pow(2).mean(dim=1) * has_smpl
    l_shape = (shape - targets['shape']).pow(2).mean(dim=1) * has_smpl
    proj = projection_fn(joints_3d, cam)                               # (m, J, 2)
    j2d = targets['joints_2d']
    vis = (j2d[:, :, 2] > 0).to(theta.dtype)
    l_reproj = ((proj - j2d[:, :, :2]).abs().sum(dim=-1) * vis).sum(dim=1) / vis.sum(dim=1).clamp(min=1.0)
    return weights[0] * l_pose + weights[1] * l_shape + weights[2] * l_reproj
