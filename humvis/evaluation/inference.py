"""Per-image inference for the evaluation scripts (Section 4 protocol: shorter side 512,
single scale, flip test, centers from Peak(P) with threshold 0.1 and at most 30 instances).
Returns the predictions of the attached heads in original-image coordinates."""
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import oks_nms
from ..heads import split_theta, weak_perspective_projection
from ..training import H36M_TO_J14


def _to_original(maps, meta, mode='bilinear'):
    """(n, C, h, w) stride-4 maps of the padded input -> (n, C, H_ori, W_ori)."""
    ph, pw = meta['padded_size']
    rh, rw = meta['resized_size']
    oh, ow = meta['ori_size']
    up = F.interpolate(maps, size=(ph, pw), mode=mode, align_corners=False)[:, :, :rh, :rw]
    return F.interpolate(up, size=(oh, ow), mode=mode, align_corners=False)


@torch.no_grad()
def predict_image(model, cfg, image, meta, keypoint_converter=None, part_lut=None):
    """image: (1, 3, H_pad, W_pad) normalized tensor; meta from TestDataset.
    keypoint_converter: optional function (n, K, 3) -> (n, K', 3) (COCO -> CrowdPose).
    part_lut: optional (n_classes_in,) -> class index mapping applied by summing the class
    probabilities of merged labels before the arg-max (MHP v2.0 -> v1.0).
    Returns a dict (possibly empty): 'poses' (n, K, 3) image pixels + score, 'pose_scores' (n,),
    'masks' (n, H, W) bool, 'mask_scores', 'parts' (n, H, W) uint8 labels, 'part_scores',
    'theta' (n, 85), 'vertices' (n, 6890, 3), 'joints14' (n, 14, 3), 'cam' (n, 3),
    'boxes' (n, 4) x, y, w, h in pixels of the projected joints (mesh) / mask / keypoints."""
    out = model(image.to(model.device), input_scale=1.0 / meta['scale'])
    if not out:
        return {}
    res = {'center_scores': out['center_scores'].cpu().numpy()}
    keep = None
    if 'poses' in out:
        poses = out['poses'].cpu().numpy()
        scores = out['pose_scores'].cpu().numpy()
        sigmas = np.array(cfg.TEST.OKS_SIGMAS, dtype=np.float32) / 10.0
        if keypoint_converter is not None:
            poses = keypoint_converter(poses)
            from ..datasets.joints import CROWDPOSE_SIGMAS
            sigmas = np.array(CROWDPOSE_SIGMAS, dtype=np.float32) / 10.0
        if cfg.TEST.OKS_NMS > 0 and poses.shape[0] > 1:
            keep, _ = oks_nms(poses, scores, cfg.TEST.OKS_NMS, sigmas)
            keep = np.asarray(sorted(keep))
        else:
            keep = np.arange(poses.shape[0])
        res['poses'], res['pose_scores'] = poses[keep], scores[keep]
    n = out['centers'].shape[0]
    if keep is None:
        keep = np.arange(n)
    res['keep'] = keep
    if 'masks' in out:
        masks = _to_original(out['masks'][keep], meta)[:, 0]
        res['masks'] = (masks > cfg.TEST.MASK_THRESHOLD).cpu().numpy()
        res['mask_scores'] = out['mask_scores'][keep].cpu().numpy()
    if 'parts' in out:
        probs = out['parts'][keep]
        if part_lut is not None:
            lut = torch.as_tensor(np.asarray(part_lut), dtype=torch.long, device=probs.device)
            merged = probs.new_zeros((probs.shape[0], int(lut.max()) + 1) + probs.shape[2:])
            merged.index_add_(1, lut, probs)
            probs = merged
        probs = _to_original(probs, meta)
        res['parts'] = probs.argmax(dim=1).to(torch.uint8).cpu().numpy()
        res['part_scores'] = out['part_scores'][keep].cpu().numpy()
    if 'theta' in out:
        theta = out['theta'][keep]
        res['theta'] = theta.cpu().numpy()
        res['cam'] = out['cam'][keep].cpu().numpy() if 'cam' in out else None
        if 'vertices' in out:
            res['vertices'] = out['vertices'][keep].cpu().numpy()
            joints14 = out['joints_h36m'][keep][:, H36M_TO_J14]
            res['joints14'] = joints14.cpu().numpy()
            # projected joints -> image pixels, for the box matching of Section 4
            _, _, cam = split_theta(theta)
            proj = weak_perspective_projection(joints14, cam).cpu().numpy()        # normalized [-1, 1]
            size_in = cfg.DATASET.INPUT_SIZE / meta['scale']
            pix = (proj + 1.0) * (size_in / 2.0)
            x0, y0 = pix.min(axis=1).T
            x1, y1 = pix.max(axis=1).T
            res['boxes'] = np.stack((x0, y0, x1 - x0, y1 - y0), axis=1)
    if 'boxes' not in res:
        if 'masks' in res:
            boxes = []
            for mk in res['masks']:
                ys, xs = np.nonzero(mk)
                boxes.append([xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
                             if len(xs) else [0, 0, 0, 0])
            res['boxes'] = np.asarray(boxes, dtype=np.float32)
        elif 'poses' in res:
            p = res['poses']
            x0, y0 = p[:, :, 0].min(axis=1), p[:, :, 1].min(axis=1)
            x1, y1 = p[:, :, 0].max(axis=1), p[:, :, 1].max(axis=1)
            res['boxes'] = np.stack((x0, y0, x1 - x0, y1 - y0), axis=1)
    return res
