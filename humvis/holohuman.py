"""HoloHuman: single-stage multi-person human analysis (Figure 2 / Figure 11).

    I --HRNet--> F --IIA--> {(F_e^(i), r^(i))} --GFD--> {F'^(i)} --heads--> pose, mask, parts, mesh

The backbone runs once per image; GFD and the task heads run once per detected person,
with weights shared across instances.  Inference follows Section 4: single scale, optional
horizontal-flip test, centers from Peak(P) with threshold 0.1 and at most 30 instances.
The training-time forward (ground-truth centers, losses) is in training.py.
"""
import torch
from torch import nn

from .backbone import build_backbone
from .iia import IIA
from .gfd import GFD
from .heads import TaskHeads, split_theta, weak_perspective_to_translation


def paste_windows(t, origins, height, width, background=None):
    """Place per-instance window outputs t (n, K, S, S) decoded from windows with origins
    (n, 2) = (x0, y0) back on an (n, K, H, W) canvas.  Outside the window the canvas is
    zero, or one-hot `background` channel for softmax outputs."""
    n, K, S, _ = t.shape
    canvas = t.new_zeros((n, K, height, width))
    if background is not None:
        canvas[:, background] = 1.0
    for i in range(n):
        x0, y0 = int(origins[i, 0]), int(origins[i, 1])
        xs, ys = max(x0, 0), max(y0, 0)
        xe, ye = min(x0 + S, width), min(y0 + S, height)
        if xe > xs and ye > ys:
            canvas[i, :, ys:ye, xs:xe] = t[i, :, ys - y0:ye - y0, xs - x0:xe - x0]
    return canvas


class HoloHuman(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        C = self.backbone.out_channels                      # 480 (W32) or 720 (W48)
        self.iia = IIA(cfg, C)
        self.gfd = GFD(cfg, C)
        self.heads = TaskHeads(cfg, self.gfd.out_channels)  # C' = C/2
        self.tasks = self.heads.tasks

        # inference protocol
        self.stride = cfg.DATASET.INPUT_SIZE // cfg.DATASET.OUTPUT_SIZE
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_index = list(cfg.DATASET.FLIP_INDEX)
        part_flip = list(cfg.DATASET.PART_FLIP_INDEX)
        self.part_flip_index = part_flip if len(part_flip) == cfg.DATASET.NUM_PARTS + 1 else None
        self.instance_chunk = max(int(cfg.TEST.INSTANCE_CHUNK), 1)
        self.input_size = cfg.DATASET.INPUT_SIZE

    # ------------------------------------------------------------------
    def shared_map(self, images):
        """F = F(I) (Eq. 1); with the flip test also the map of the flipped image,
        flipped back to the original orientation.  Returns (V, C, H, W)."""
        if self.flip_test:
            images = torch.cat((images, images.flip(-1)), dim=0)
        feats = self.backbone(images)
        if self.flip_test:
            feats = torch.stack((feats[0], feats[1].flip(-1)), dim=0)
        return feats

    def decode(self, feats, descriptors, centers):
        """GFD + task heads for the instance rows (feats (n, C, H, W) is the map of each
        instance's view).  Window outputs are pasted back on the full map."""
        H, W = feats.shape[-2:]
        F_prime, origins = self.gfd(feats, descriptors, centers)          # Eqs. 10-15
        out = self.heads(F_prime)
        if origins is not None:
            bg = self.heads.num_parts if 'parts' in out else None
            for k in ('heatmaps', 'masks', 'part_logits'):
                if k in out:
                    out[k] = paste_windows(out[k], origins, H, W)
            if 'parts' in out:
                out['parts'] = paste_windows(out['parts'], origins, H, W, background=bg)
        return out, F_prime

    def decode_instances(self, feats, instances):
        """GFD + task heads for every (view, instance) row, in chunks to bound memory."""
        descriptors, view, centers = instances['descriptors'], instances['view'], instances['centers']
        m = centers.shape[0]
        outputs = {}
        for start in range(0, descriptors.shape[0], self.instance_chunk):
            end = min(start + self.instance_chunk, descriptors.shape[0])
            rows = torch.arange(start, end, device=feats.device)
            out, _ = self.decode(feats[view[rows]], descriptors[rows], centers[rows % m])
            for k, v in out.items():
                outputs.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0) for k, v in outputs.items()}

    def merge_views(self, outputs, m):
        """Average the two views of the flip test (keypoint / part channels permuted
        for the flipped view).  Mesh parameters are taken from the original view."""
        if not self.flip_test:
            return outputs
        merged = {}
        if 'heatmaps' in outputs:
            hm, hm_flip = outputs['heatmaps'][:m], outputs['heatmaps'][m:]
            merged['heatmaps'] = (hm + hm_flip[:, self.flip_index]) / 2.0
        if 'masks' in outputs:
            merged['masks'] = (outputs['masks'][:m] + outputs['masks'][m:]) / 2.0
        if 'parts' in outputs:
            parts, parts_flip = outputs['parts'][:m], outputs['parts'][m:]
            if self.part_flip_index is not None:
                parts_flip = parts_flip[:, self.part_flip_index]
            merged['parts'] = (parts + parts_flip) / 2.0
        if 'theta' in outputs:
            merged['theta'] = outputs['theta'][:m]
        return merged

    # ------------------------------------------------------------------
    def forward(self, images, input_scale=1.0, centers=None):
        """images: (1, 3, H_I, W_I) normalized RGB tensor (the short side resized to
        INPUT_SIZE, padded to a multiple of 32).  input_scale maps network input pixels
        to the original image.  centers: optional (m, 2) (x, y) feature-map cells that
        replace Peak(P).

        Returns {} if no person is detected, else a dict with per-instance results in
        original-image coordinates (poses, masks, parts, mesh) plus the raw network
        outputs (centers, scores, feats, heatmaps, theta, attention, center_heatmap).
        Only the outputs of the attached heads (cfg.MODEL.HEADS.TASKS) are present.
        """
        images = images.to(self.device)
        assert images.shape[0] == 1, 'inference runs one image at a time'

        feats = self.shared_map(images)                                   # Eq. 1
        instances = self.iia(feats, centers=centers)                      # Algorithm 1
        if instances is None:
            return {}
        m = instances['centers'].shape[0]

        outputs = self.merge_views(self.decode_instances(feats, instances), m)  # Algorithm 2
        scores = instances['scores']
        results = {
            'centers': instances['centers'],
            'center_scores': scores,
            'feats': torch.nn.functional.normalize(instances['descriptors'][:m], dim=1),
            'attention': instances['attention'][0],
            'center_heatmap': instances['center_heatmap'],
            'mesh_loc': (instances['centers'] * self.stride + self.stride / 2.0) * input_scale,
        }

        if 'heatmaps' in outputs:
            # pose: each keypoint at the maximum of its heatmap, mapped to image coordinates
            heatmaps = outputs['heatmaps']
            n, K, h, w = heatmaps.shape
            vals, inds = heatmaps.view(n, K, -1).max(dim=2)
            x = (inds % w).float()
            y = torch.div(inds, w, rounding_mode='floor').float()
            vals = vals * scores.unsqueeze(1)
            poses = torch.stack((x, y, vals), dim=2)
            poses[:, :, :2] = (poses[:, :, :2] * self.stride + self.stride / 2.0) * input_scale
            results.update({'heatmaps': heatmaps, 'poses': poses, 'pose_scores': vals.mean(dim=1)})
        if 'masks' in outputs:
            results.update({'masks': outputs['masks'], 'mask_scores': scores})
        if 'parts' in outputs:
            results.update({'parts': outputs['parts'], 'part_scores': scores})
        if 'theta' in outputs:
            theta = outputs['theta']
            results['theta'] = theta
            if self.heads.smpl is not None:
                # mesh: SMPL forward of Theta and the camera of the weak-perspective model
                vertices, joints, joints_h36m = self.heads.smpl_forward(theta)
                _, _, cam = split_theta(theta)
                cam_t = weak_perspective_to_translation(cam, self.heads.focal, self.input_size * input_scale)
                results.update({
                    'mesh': vertices + cam_t[:, None, :],
                    'vertices': vertices,
                    'joints_3d': joints,
                    'joints_h36m': joints_h36m,
                    'cam': cam,
                    'cam_t': cam_t,
                })
        return results
