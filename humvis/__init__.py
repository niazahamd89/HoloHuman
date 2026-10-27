"""HumVis: demo wrapper around HoloHuman (pose, body segmentation, part segmentation,
mesh reconstruction from one forward pass).  The public API used by app.py / camera.py /
video.py is unchanged:  HumVis(backbone_name).get_analysis_results(img, tracking)."""
import os

import torchvision
from yacs.config import CfgNode
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from .holohuman import HoloHuman
from .utils import test_image_process, oks_nms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)


def get_cfg() -> CfgNode:
    from .config import _C

    return _C.clone()


def load_weights(model, path, strict=True):
    """Load a HoloHuman checkpoint.  Checkpoints of the previous release (CID-style
    1x1 heads, C'=32, no attention) do not match this architecture and are rejected
    with a list of the mismatching keys."""
    state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    own = model.state_dict()
    mismatched = [k for k, v in state.items() if k in own and tuple(own[k].shape) != tuple(v.shape)]
    if not strict:   # e.g. C-mhp1 initialized from C-part: the 59-channel part output is re-initialized
        state = {k: v for k, v in state.items() if k not in mismatched}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            f'{path} does not match the HoloHuman architecture described in the paper:\n'
            f'  missing keys   : {len(missing)} (e.g. {missing[:5]})\n'
            f'  unexpected keys: {len(unexpected)} (e.g. {unexpected[:5]})\n'
            f'  shape mismatch : {len(mismatched)} (e.g. {mismatched[:5]})\n'
            'Train the architecture of Section 3 / Appendix A.2, or pass strict=False to load '
            'only the matching tensors (the backbone).')
    return missing, unexpected


class HumVis(object):
    def __init__(self, backbone_name='hrnet32', weights=None, flip_test=None, strict=True):
        cfg = get_cfg()
        cfg.defrost()
        cfg.MODEL.BACKBONE = backbone_name
        if flip_test is not None:
            cfg.TEST.FLIP_TEST = bool(flip_test)
        cfg.freeze()
        self.cfg = cfg

        model = HoloHuman(cfg, device)
        weights = cfg.TEST.MODEL_FILE if weights is None else weights
        if weights and os.path.isfile(weights):
            load_weights(model, weights, strict=strict)
        else:
            print(f'[WARN] no checkpoint found at {weights!r}; running with random weights')

        model = model.to(device)
        self.model = model.eval()

        self.init_video_queue()

    def init_video_queue(self):
        self.instances = []

    @torch.no_grad()
    def get_analysis_results(self, img, tracking=False):
        assert len(img.shape) == 3 and img.shape[2] == 3
        # convert bgr to rgb
        img_rgb = img[:, :, ::-1] - np.zeros_like(img)
        ori_size = img.shape[:2]
        img_rgb, padded_size, resized_size, scale_factor = test_image_process(img_rgb)
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        inputs = transforms(img_rgb).unsqueeze(0)
        # one forward pass: pose, mask, part and mesh results for all detected persons
        inputs = inputs.to(device)
        results = self.model(inputs, input_scale=1/scale_factor)
        if 'centers' not in results:
            return img, img, img, img
        n_inst = results['centers'].shape[0]
        h_out, w_out = results['center_heatmap'].shape[-2:]

        # heads that are not attached to this checkpoint (cfg.MODEL.HEADS.TASKS) give empty outputs
        if 'poses' in results:
            poses = results['poses'].cpu().numpy()
            scores = results['pose_scores'].cpu().numpy()
        else:
            poses = np.zeros((n_inst, 17, 3), dtype=np.float32)
            scores = results['center_scores'].cpu().numpy()
        if 'masks' not in results:
            results['masks'] = torch.zeros((n_inst, 1, h_out, w_out), device=results['centers'].device)
        if 'parts' not in results:
            results['parts'] = torch.zeros((n_inst, 2, h_out, w_out), device=results['centers'].device)
            results['parts'][:, 0] = 1.0

        # nms by pose
        keep, _ = oks_nms(poses, scores,  0.9, np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0)

        poses = poses[keep]

        masks = results['masks'][keep]
        # upsample mask
        masks = F.interpolate(masks,
                                size=padded_size,
                                mode='bilinear')[:, :, :resized_size[0], :resized_size[1]]
        masks = F.interpolate(masks,
                                size=ori_size,
                                mode='bilinear')

        masks = masks[:, 0, :, :]
        masks = (masks > 0.5).float()
        masks = masks.cpu().numpy()

        partmaps = results['parts'][keep]
        partmaps = F.interpolate(partmaps,
                                  size=padded_size,
                                  mode='bilinear')[:, :, :resized_size[0], :resized_size[1]]
        partmaps = F.interpolate(partmaps,
                                size=ori_size,
                                mode='bilinear')
        partmaps = partmaps.max(dim=1)[1]
        partmaps = partmaps.cpu().numpy()

        meshes = results['mesh'][keep] if 'mesh' in results else None
        mesh_locs = results['mesh_loc'][keep]

        if poses.shape[0] > 1 and tracking:
            # identity matching on the L2-normalized interaction-enhanced descriptors F_e
            instance_feats = results['feats'][keep]

            # first frame
            if len(self.instances) == 0:
                for i in range(instance_feats.shape[0]):
                    self.instances.append(instance_feats[i].cpu().numpy())
                target_ids = list(range(len(self.instances)))
            else:
                # perform ID match
                src_feats = np.stack(self.instances, axis=0)
                assert src_feats.shape[0] == len(self.instances)
                tgt_feats = instance_feats.cpu().numpy()
                match_mat = np.matmul(src_feats, tgt_feats.T)
                src_ind, tgt_ind = linear_sum_assignment(1 - match_mat)
                target_ids = [-1 for _ in range(tgt_feats.shape[0])]
                for i in range(len(tgt_ind)):
                    target_ids[tgt_ind[i]] = src_ind[i]
                    # update feature
                    self.instances[src_ind[i]] = tgt_feats[tgt_ind[i]]
                # update new feature
                if len(self.instances) < 50:
                    for i in range(len(target_ids)):
                        if target_ids[i] == -1:
                            target_ids[i] = len(self.instances)
                            self.instances.append(tgt_feats[i])
            instance_ids = target_ids
        else:
            instance_ids = list(range(poses.shape[0]))

        from .vis import draw_results
        faces = self.model.heads.smpl.faces_tensor if self.model.heads.smpl is not None else None
        pose_img, mask_img, part_img, mesh_img = draw_results(
            img, poses, masks, partmaps, meshes, mesh_locs, faces,
            focal=self.cfg.MODEL.HEADS.FOCAL, instance_ids=instance_ids)

        return pose_img, mask_img, part_img, mesh_img
