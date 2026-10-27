"""Instance Interaction Attention (IIA), Section 3.1 / Algorithm 1 / Figure 11.

    F  --H_center-->  P  --Peak-->  {r^(i)}  --P_inst-->  {f^(i)}  --stack-->  Z
    Z  --W_Q, W_K, W_V-->  Q, K, V ;  I = softmax(Q K^T / sqrt(d))  ;  F_e = I V W_O

Shapes (per view): F in R^{C x H x W}, P in [0,1]^{1 x H x W}, Z in R^{m x C},
Q, K, V in R^{m x d}, I in R^{m x m}, F_e in R^{m x C}.

Default (all reported results): the attention is computed from descriptor content only
(Eq. 7).  Two switches reproduce the ablation rows:
  * cfg.MODEL.IIA.ENABLED = False      -> F_e^(i) <- f^(i)  (Algorithm 1, line 18; Table 4 rows 0-3,
                                          Tables 2-3 rows without I_ij, Table 6 "No IIA")
  * cfg.MODEL.IIA.POSITIONAL = 'pe' | 'bias' | 'both'  (Table 6): absolute sinusoidal encoding of
    the normalized center added to f^(i) before Eq. 6, and/or the relative-position bias B of Eq. 9.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class CenterHead(nn.Module):
    """H_center (Eq. 2): 3x3 conv C->64 (BatchNorm, ReLU), 3x3 conv 64->1, sigmoid."""

    def __init__(self, in_channels, mid_channels=64, bias_prob=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, 1, 3, 1, 1)
        # the output bias starts at logit(bias_prob) so that P ~ bias_prob everywhere at
        # initialization (focal-loss heads, as in CID / CenterNet; cfg.MODEL.BIAS_PROB)
        nn.init.constant_(self.conv2.bias, -math.log((1.0 - bias_prob) / bias_prob))

    def forward(self, feats):
        x = F.relu(self.bn1(self.conv1(feats)), inplace=True)
        return torch.sigmoid(self.conv2(x))


def peak(center_heatmap, kernel=3, threshold=0.1, max_instances=30):
    """Peak(P) (Eq. 3): local maxima of P after k x k max-pooling NMS and confidence
    thresholding, keeping at most `max_instances`.

    center_heatmap: (1, 1, H, W).  Returns centers (m, 2) as (x, y) in feature-map
    cells and their scores (m,), sorted by decreasing score; m may be 0.
    """
    assert center_heatmap.dim() == 4 and center_heatmap.shape[:2] == (1, 1)
    H, W = center_heatmap.shape[-2:]
    maxm = F.max_pool2d(center_heatmap, kernel, 1, kernel // 2)
    keep = torch.eq(maxm, center_heatmap).to(center_heatmap.dtype)
    scores = (center_heatmap * keep).view(-1)
    k = min(int(max_instances), scores.numel())
    scores, idx = scores.topk(k, dim=0)
    valid = scores > threshold
    scores, idx = scores[valid], idx[valid]
    xs = (idx % W).to(center_heatmap.dtype)
    ys = torch.div(idx, W, rounding_mode='floor').to(center_heatmap.dtype)
    centers = torch.stack((xs, ys), dim=1)
    return centers, scores


def pool_instance_descriptors(feats, centers, kernel=3):
    """P_inst(F, r) (Eq. 4): bilinear sampling of F at r followed by average pooling
    over a k x k neighbourhood (the two operations commute, so the k x k average is
    applied to F first and the result is sampled bilinearly at r).

    feats: (1, C, H, W); centers: (m, 2) as (x, y) in cells.  Returns f: (m, C).
    """
    assert feats.dim() == 4 and feats.shape[0] == 1
    H, W = feats.shape[-2:]
    pooled = F.avg_pool2d(feats, kernel, 1, kernel // 2, count_include_pad=False)
    gx = centers[:, 0] / max(W - 1, 1) * 2 - 1
    gy = centers[:, 1] / max(H - 1, 1) * 2 - 1
    grid = torch.stack((gx, gy), dim=-1).view(1, -1, 1, 2).to(feats.dtype)
    sampled = F.grid_sample(pooled, grid, mode='bilinear', align_corners=True)  # (1, C, m, 1)
    return sampled[0, :, :, 0].t().contiguous()


def sinusoidal_encoding(coords, num_freq):
    """Sinusoidal encoding (Vaswani et al., 2017) of normalized coordinates in [0, 1].
    coords: (m, 2).  Returns (m, 2 * num_freq): for each coordinate, num_freq/2 sines and
    num_freq/2 cosines at geometrically spaced frequencies."""
    half = num_freq // 2
    freq = torch.exp(torch.arange(half, device=coords.device, dtype=coords.dtype)
                     * (-math.log(10000.0) / max(half, 1)))
    arg = coords[:, :, None] * 2.0 * math.pi * 128.0 * freq[None, None, :]   # (m, 2, half)
    enc = torch.cat((torch.sin(arg), torch.cos(arg)), dim=-1)               # (m, 2, num_freq)
    return enc.view(coords.shape[0], -1)


class InstanceInteraction(nn.Module):
    """Single-head self-attention over the m instance descriptors (Eqs. 5-8, Figure 3),
    optionally with the relative-position bias of Eq. 9 (Table 6 only)."""

    def __init__(self, in_channels, d=256, relative_bias=False, bias_hidden=64):
        super().__init__()
        self.d = d
        # linear maps C->d (W_Q, W_K, W_V) and d->C (W_O); the bias terms are included in
        # the 0.49M parameters / 34.40M total of Table 7
        self.W_Q = nn.Linear(in_channels, d)
        self.W_K = nn.Linear(in_channels, d)
        self.W_V = nn.Linear(in_channels, d)
        self.W_O = nn.Linear(d, in_channels)
        # psi of Eq. 9: two-layer MLP 2 -> 64 -> 1 (ReLU) shared across all pairs
        self.psi = None
        if relative_bias:
            self.psi = nn.Sequential(nn.Linear(2, bias_hidden), nn.ReLU(inplace=True),
                                     nn.Linear(bias_hidden, 1))

    def forward(self, Z, rel_offsets=None):
        """Z: (m, C) stacked descriptors; rel_offsets: optional (m, m, 2) normalized
        (r^(i) - r^(j)) / (W, H) for Eq. 9.  Returns F_e (m, C) and I (m, m)."""
        Q, K, V = self.W_Q(Z), self.W_K(Z), self.W_V(Z)
        logits = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.d)
        if self.psi is not None and rel_offsets is not None:
            logits = logits + self.psi(rel_offsets).squeeze(-1)        # + B, Eq. 9
        I = torch.softmax(logits, dim=-1)                       # row-wise softmax, Eq. 7
        F_e = self.W_O(torch.matmul(I, V))                      # Eq. 8
        return F_e, I


class IIA(nn.Module):
    """Localizes the persons on the shared map F and returns one interaction-enhanced
    descriptor F_e^(i) per detected center r^(i) (Algorithm 1)."""

    def __init__(self, cfg, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.enabled = bool(cfg.MODEL.IIA.ENABLED)
        self.positional = str(cfg.MODEL.IIA.POSITIONAL).lower()
        assert self.positional in ('none', 'pe', 'bias', 'both'), self.positional
        self.center_head = CenterHead(in_channels, cfg.MODEL.IIA.CENTER_CHANNELS, cfg.MODEL.BIAS_PROB)
        self.attention = None
        self.pos_mlp = None
        if self.enabled:
            self.attention = InstanceInteraction(
                in_channels, cfg.MODEL.IIA.D,
                relative_bias=self.positional in ('bias', 'both'),
                bias_hidden=cfg.MODEL.IIA.BIAS_HIDDEN)
            if self.positional in ('pe', 'both'):
                self.pe_dim = cfg.MODEL.IIA.PE_DIM
                self.pos_mlp = nn.Sequential(nn.Linear(2 * self.pe_dim, in_channels),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(in_channels, in_channels))
        self.pool_kernel = cfg.MODEL.IIA.POOL_KERNEL
        self.nms_kernel = cfg.TEST.NMS_KERNEL
        self.center_threshold = cfg.TEST.CENTER_THRESHOLD
        self.max_instances = cfg.TEST.MAX_INSTANCES

    def interact(self, f, centers, height, width):
        """f: (m, C) descriptors of one view; returns F_e (m, C) and I (m, m).
        With IIA disabled, F_e = f and I is the identity (no interaction)."""
        m = f.shape[0]
        if not self.enabled:
            return f, torch.eye(m, device=f.device, dtype=f.dtype)
        if self.pos_mlp is not None:                                     # absolute PE (Table 6)
            norm = centers.to(f.dtype) / torch.tensor([max(width - 1, 1), max(height - 1, 1)],
                                                      device=f.device, dtype=f.dtype)
            f = f + self.pos_mlp(sinusoidal_encoding(norm, self.pe_dim))
        rel = None
        if self.attention.psi is not None:                               # relative bias (Eq. 9)
            c = centers.to(f.dtype)
            rel = (c[:, None, :] - c[None, :, :]) / torch.tensor([float(width), float(height)],
                                                                 device=f.device, dtype=f.dtype)
        return self.attention(f, rel)

    def forward(self, feats, centers=None):
        """feats: (V, C, H, W) shared map(s); V = 1, or V = 2 for the flip test where
        feats[1] is the map of the horizontally flipped image flipped back to the
        original orientation.  Persons are localized on the view-averaged P, and the
        attention is computed separately within each view.

        centers: optional (m, 2) (x, y) cells that replace Peak(P) (ground-truth
        centers during training, Section 3.3).

        Returns a dict with
            'center_heatmap' (1, 1, H, W), 'centers' (m, 2), 'scores' (m,),
            'descriptors' (V*m, C) = F_e, 'pooled' (V*m, C) = f (before interaction),
            'view' (V*m,) view index of each row, 'attention' list of V matrices I (m, m).
        or None if no person is detected.
        """
        V, _, H, W = feats.shape
        P = self.center_head(feats)                                   # (V, 1, H, W), Eq. 2
        P_det = P.mean(dim=0, keepdim=True)
        if centers is None:
            centers, scores = peak(P_det, self.nms_kernel, self.center_threshold,
                                   self.max_instances)                # Eq. 3
        else:
            centers = centers.to(feats.dtype)
            scores = P_det[0, 0, centers[:, 1].long().clamp(0, H - 1),
                           centers[:, 0].long().clamp(0, W - 1)]
        m = centers.shape[0]
        if m == 0:
            return None

        descriptors, pooled, views, attention = [], [], [], []
        for v in range(V):
            f = pool_instance_descriptors(feats[v:v + 1], centers, self.pool_kernel)  # Z, Eqs. 4-5
            F_e, I = self.interact(f, centers, H, W)                                  # Eqs. 6-8
            descriptors.append(F_e)
            pooled.append(f)
            views.append(torch.full((m,), v, dtype=torch.long, device=feats.device))
            attention.append(I)

        return {
            'center_heatmap': P_det,
            'centers': centers,
            'scores': scores,
            'descriptors': torch.cat(descriptors, dim=0),
            'pooled': torch.cat(pooled, dim=0),
            'view': torch.cat(views, dim=0),
            'attention': attention,
        }
