"""Backbones of the backbone analysis (Table 8): ResNet-50 / ResNet-101 and ViT / ViT-L.

The paper specifies the HRNet fusion only (four branches upsampled to stride 4 and
concatenated, Appendix A.2).  For the other backbones the shared map F is formed at the
same stride 4 as follows (choices of this implementation, see README "assumptions"):

  * ResNet-50 / 101 (He et al., 2016): the stage outputs C2-C5 (strides 4-32) are reduced by
    1x1 convolutions (BN, ReLU) to 128 channels each, bilinearly upsampled to stride 4 and
    concatenated: C = 512.
  * ViT-B/16, ViT-L/16 (Dosovitskiy et al., 2020; timm, dynamic input size): the patch tokens
    (stride 16) are reshaped to a map and upsampled by two 4x4 stride-2 deconvolutions
    (BN, ReLU) with 256 channels, as in the ViTPose decoder: C = 256.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ResNetStride4(nn.Module):
    def __init__(self, depth=50, pretrained='', lateral=128):
        super().__init__()
        import torchvision
        ctor = {50: torchvision.models.resnet50, 101: torchvision.models.resnet101}[depth]
        weights = None
        if pretrained == 'torchvision':
            weights = 'IMAGENET1K_V1'
        net = ctor(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        chans = [256, 512, 1024, 2048]
        self.lateral = nn.ModuleList([nn.Sequential(nn.Conv2d(c, lateral, 1, bias=False),
                                                    nn.BatchNorm2d(lateral), nn.ReLU(inplace=True))
                                      for c in chans])
        self.out_channels = lateral * 4
        if pretrained and pretrained != 'torchvision':
            from .backbone import load_pretrained_backbone
            load_pretrained_backbone(net, pretrained)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        size = feats[0].shape[-2:]
        outs = [self.lateral[0](feats[0])]
        for lat, f in zip(self.lateral[1:], feats[1:]):
            outs.append(F.interpolate(lat(f), size=size, mode='bilinear', align_corners=False))
        return torch.cat(outs, dim=1)


class ViTStride4(nn.Module):
    def __init__(self, size='base', pretrained='', decoder_channels=256):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError('the ViT backbones of Table 8 need `pip install timm`') from e
        name = {'base': 'vit_base_patch16_224', 'large': 'vit_large_patch16_224'}[size]
        self.vit = timm.create_model(name, pretrained=(pretrained == 'timm'), num_classes=0,
                                     dynamic_img_size=True)
        embed = self.vit.embed_dim
        if pretrained and pretrained != 'timm':
            from .backbone import load_pretrained_backbone
            load_pretrained_backbone(self.vit, pretrained)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed, decoder_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True))
        self.out_channels = decoder_channels

    def forward(self, x):
        B, _, H, W = x.shape
        tokens = self.vit.forward_features(x)                     # (B, 1 + N, D) with class token
        n_prefix = tokens.shape[1] - (H // 16) * (W // 16)
        fmap = tokens[:, n_prefix:].transpose(1, 2).reshape(B, -1, H // 16, W // 16)
        return self.decoder(fmap)


def build_extra_backbone(cfg):
    name = str(cfg.MODEL.BACKBONE).lower()
    pre = cfg.MODEL.PRETRAINED
    if name in ('resnet50', 'resnet-50', 'rn50'):
        return ResNetStride4(50, pre)
    if name in ('resnet101', 'resnet-101', 'rn101'):
        return ResNetStride4(101, pre)
    if name in ('vit', 'vit_base', 'vit-b'):
        return ViTStride4('base', pre)
    if name in ('vit_large', 'vit-l', 'vitl'):
        return ViTStride4('large', pre)
    raise ValueError(f'unknown backbone {name!r}')
