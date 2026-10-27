"""Global Feature Distillation (GFD), Section 3.2 / Algorithm 2 / Figure 11.

Per detected person i (weights shared across instances, stride 4, no upsampling):

    G^(i) = W_g * [F, B(F_e^(i)), E(r^(i))]                       (Eq. 10, 1x1, 2C+2 -> C')
    A_c   = sigmoid(W2 d(W1 GAP(G)) + W2 d(W1 GMP(G)))            (Eq. 11, SE bottleneck, r = 16)
    F_c   = A_c (.) G                                             (Eq. 12)
    A_s   = sigmoid(W_s * [AvgPool_c(G), MaxPool_c(G)])           (Eq. 13, 7x7 conv)
    F_s   = A_s (.) G                                             (Eq. 14)
    F'^(i)= ReLU(W_f * [F_c, F_s])                                (Eq. 15, 3x3, 2C' -> C')

Ablation switches (Table 4; Algorithm 2, line 17):
  * cfg.MODEL.GFD.CHANNEL_ATTENTION / SPATIAL_ATTENTION = False drop F_c / F_s from [F_c, F_s];
    with one branch W_f takes C' channels, with none W_f is applied to G itself (row 0).
  * cfg.MODEL.GFD.CENTER_OFFSET = False removes E(r^(i)) from Eq. 10 (W_g on 2C channels, row 5).
  * cfg.MODEL.GFD.WINDOW = S > 0 decodes G^(i) from an S x S window of F centred at r^(i) and
    zero-padded at the border instead of the whole map F (row 6, "Full F" unchecked).
"""
import torch
from torch import nn
import torch.nn.functional as F


def center_offset_encoding(centers, height, width):
    """E(r^(i)): two-channel map of the offset of every location (x, y) from the
    center r^(i) = (x_i, y_i), normalized by the map size:
        E[0](x, y) = (x - x_i) / W,   E[1](x, y) = (y - y_i) / H.
    centers: (n, 2) (x, y) in cells.  Returns (n, 2, H, W).
    """
    device, dtype = centers.device, centers.dtype
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    ex = (xx[None] - centers[:, 0, None, None]) / float(width)
    ey = (yy[None] - centers[:, 1, None, None]) / float(height)
    return torch.stack((ex, ey), dim=1)


def crop_windows(feats, centers, size):
    """S x S window of each instance's map centred at its (rounded) center, zero-padded
    where the window leaves the map.  feats: (n, C, H, W); centers: (n, 2).
    Returns windows (n, C, S, S), the window origins (n, 2) as (x0, y0) in cells, and the
    centers expressed in window coordinates (n, 2)."""
    n, C, H, W = feats.shape
    half = size // 2
    x0 = centers[:, 0].round().long() - half
    y0 = centers[:, 1].round().long() - half
    padded = F.pad(feats, (size, size, size, size))                      # zero padding
    windows = []
    for i in range(n):
        ys, xs = y0[i].item() + size, x0[i].item() + size
        windows.append(padded[i, :, ys:ys + size, xs:xs + size])
    origins = torch.stack((x0, y0), dim=1)
    return torch.stack(windows, dim=0), origins, centers - origins.to(centers.dtype)


class GlobalFeatureDecoder(nn.Module):
    """D(F, F_e^(i), r^(i)) = W_g * [F, B(F_e^(i)), E(r^(i))]  (Eq. 10)."""

    def __init__(self, in_channels, out_channels, center_offset=True):
        super().__init__()
        self.center_offset = center_offset
        self.W_g = nn.Conv2d(2 * in_channels + (2 if center_offset else 0), out_channels, 1, 1, 0)

    def forward(self, feats, descriptors, centers):
        """feats: (n, C, H, W) shared map of the view of each instance;
        descriptors: (n, C) = F_e^(i); centers: (n, 2).  Returns G: (n, C', H, W)."""
        n, C, H, W = feats.shape
        broadcast = descriptors.view(n, C, 1, 1).expand(n, C, H, W)      # B(F_e^(i))
        parts = [feats, broadcast]
        if self.center_offset:
            parts.append(center_offset_encoding(centers, H, W).to(feats.dtype))  # E(r^(i))
        return self.W_g(torch.cat(parts, dim=1))                         # (n, 2C+2, H, W) -> (n, C', H, W)


class ChannelAttention(nn.Module):
    """A_c (Eq. 11): GAP and GMP of G through a shared SE bottleneck C'->C'/r->C', summed, sigmoid."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.W1 = nn.Linear(channels, hidden)
        self.W2 = nn.Linear(hidden, channels)

    def forward(self, G):
        gap = G.mean(dim=(2, 3))
        gmp = G.amax(dim=(2, 3))
        a = self.W2(F.relu(self.W1(gap))) + self.W2(F.relu(self.W1(gmp)))
        return torch.sigmoid(a)[:, :, None, None]                        # (n, C', 1, 1)


class SpatialAttention(nn.Module):
    """A_s (Eq. 13): k x k convolution of the channel-wise average and max maps of G, sigmoid."""

    def __init__(self, kernel=7):
        super().__init__()
        self.W_s = nn.Conv2d(2, 1, kernel, 1, kernel // 2)

    def forward(self, G):
        pooled = torch.cat((G.mean(dim=1, keepdim=True), G.amax(dim=1, keepdim=True)), dim=1)
        return torch.sigmoid(self.W_s(pooled))                            # (n, 1, H, W)


class GFD(nn.Module):
    """Decodes an instance-conditioned global map per person and distills it into the
    final instance-specific representation F'^(i) consumed by the task heads."""

    def __init__(self, cfg, in_channels):
        super().__init__()
        self.in_channels = in_channels
        C_prime = cfg.MODEL.GFD.CHANNELS if cfg.MODEL.GFD.CHANNELS > 0 else in_channels // 2
        self.out_channels = C_prime
        self.window = int(cfg.MODEL.GFD.WINDOW)
        self.decoder = GlobalFeatureDecoder(in_channels, C_prime, bool(cfg.MODEL.GFD.CENTER_OFFSET))
        self.channel_attention = ChannelAttention(C_prime, cfg.MODEL.GFD.REDUCTION) \
            if cfg.MODEL.GFD.CHANNEL_ATTENTION else None
        self.spatial_attention = SpatialAttention(cfg.MODEL.GFD.SPATIAL_KERNEL) \
            if cfg.MODEL.GFD.SPATIAL_ATTENTION else None
        n_branches = int(self.channel_attention is not None) + int(self.spatial_attention is not None)
        self.W_f = nn.Conv2d(max(n_branches, 1) * C_prime, C_prime, 3, 1, 1)

    def forward(self, feats, descriptors, centers):
        """feats: (n, C, H, W); descriptors: (n, C); centers: (n, 2).
        Returns F': (n, C', H, W) (or (n, C', S, S) with a window) and the window origins
        (n, 2) as (x0, y0) in cells, or None when G is decoded from the whole map."""
        origins = None
        if self.window > 0:
            feats, origins, centers = crop_windows(feats, centers, self.window)
        G = self.decoder(feats, descriptors, centers)                     # Eq. 10
        branches = []
        if self.channel_attention is not None:
            branches.append(self.channel_attention(G) * G)               # Eqs. 11-12
        if self.spatial_attention is not None:
            branches.append(self.spatial_attention(G) * G)               # Eqs. 13-14
        x = torch.cat(branches, dim=1) if branches else G
        return F.relu(self.W_f(x), inplace=True), origins                # Eq. 15
