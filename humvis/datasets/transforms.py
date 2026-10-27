"""Training augmentation (Section 4): random rotation (+-30 deg), scaling ([0.75, 1.5] of the
short side relative to INPUT_SIZE), horizontal flip (p = 0.5) and translation, followed by a
512 x 512 crop; the same transform is applied to keypoints, masks, part maps and 2D joints.
Test-time preprocessing resizes the shorter side to INPUT_SIZE and pads to a multiple of 32
(utils.test_image_process)."""
import math

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(img_rgb_uint8):
    """HWC uint8 RGB -> CHW float32 normalized with the ImageNet statistics."""
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def sample_augmentation(rng, cfg):
    """Random (scale, rotation, flip, translation) of Section 4."""
    scale = rng.uniform(cfg.DATASET.MIN_SCALE, cfg.DATASET.MAX_SCALE)
    rot = rng.uniform(-cfg.DATASET.MAX_ROTATION, cfg.DATASET.MAX_ROTATION)
    flip = rng.random() < cfg.DATASET.FLIP
    tx = rng.uniform(-cfg.DATASET.MAX_TRANSLATE, cfg.DATASET.MAX_TRANSLATE)
    ty = rng.uniform(-cfg.DATASET.MAX_TRANSLATE, cfg.DATASET.MAX_TRANSLATE)
    return dict(scale=scale, rot=rot, flip=flip, tx=tx, ty=ty)


def affine_matrix(img_w, img_h, out_size, scale=1.0, rot=0.0, flip=False, tx=0.0, ty=0.0,
                  scale_type='short'):
    """3x3 matrix mapping original-image pixels to the out_size x out_size crop.
    The image is resized so that its short (or long) side equals out_size * scale, rotated
    by `rot` degrees about its center, translated by (tx, ty) output pixels, optionally
    mirrored, and centred in the crop."""
    ref = min(img_w, img_h) if scale_type == 'short' else max(img_w, img_h)
    s = out_size * scale / float(ref)
    cx, cy = img_w / 2.0, img_h / 2.0
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    r = math.radians(rot)
    R = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]], dtype=np.float64)
    Fm = np.array([[-1 if flip else 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[1, 0, out_size / 2.0 + tx], [0, 1, out_size / 2.0 + ty], [0, 0, 1]], dtype=np.float64)
    return T2 @ Fm @ R @ S @ T1


def apply_matrix(points, M):
    """points (..., 2) -> transformed (..., 2) with the 3x3 matrix M."""
    pts = np.asarray(points, dtype=np.float64)
    shape = pts.shape
    pts = pts.reshape(-1, 2)
    hom = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    return (hom @ M.T)[:, :2].reshape(shape).astype(np.float32)


def warp_image(img, M, out_size, interpolation=cv2.INTER_LINEAR, border=0):
    return cv2.warpAffine(img, M[:2].astype(np.float32), (out_size, out_size),
                          flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def scale_matrix(M, factor):
    """Compose M with a scaling by `factor` of the output (e.g. 1/4 for stride-4 targets).
    Output cell c corresponds to input pixel 4c + 2 (decoding of HoloHuman.forward), so the
    output grid is offset by half a cell."""
    S = np.array([[factor, 0, -0.5 + 0.5 * factor], [0, factor, -0.5 + 0.5 * factor], [0, 0, 1]], dtype=np.float64)
    return S @ M


def bbox_after_transform(bbox_xywh, M, size):
    """Transformed axis-aligned box (x, y, w, h), clipped to [0, size]."""
    x, y, w, h = bbox_xywh
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float64)
    c = apply_matrix(corners, M)
    x0, y0 = np.clip(c.min(axis=0), 0, size)
    x1, y1 = np.clip(c.max(axis=0), 0, size)
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)
