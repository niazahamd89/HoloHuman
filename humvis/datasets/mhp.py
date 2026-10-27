"""MHP v2.0 (58 part categories) and MHP v1.0 (18 categories), Section 4.

Expected layout (cfg.DATASET.ROOT = data):
    data/mhp_v2/{train,val,test}/images/<id>.jpg
    data/mhp_v2/{train,val}/parsing_annos/<id>_<num_persons:02d>_<k:02d>.png   (one PNG per person,
                                                                            values = category ids 1..58)
    data/mhp_v1/images/<id>.jpg
    data/mhp_v1/annotations/<id>_<num_persons:02d>_<k:02d>.png                 (values 1..18)
    data/mhp_v1/{train,val,test}_list.txt                                       (image ids, one per line)

Every person record carries the path of its part PNG; the box is the extent of its mask.
"""
import glob
import os
import re

import cv2
import numpy as np

_ANNO_RE = re.compile(r'^(?P<id>.+?)_(?P<n>\d{2})_(?P<k>\d{2})\.png$')


def _image_size(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img.shape[1], img.shape[0]


def _mask_bbox(png_path):
    lab = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(png_path)
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    ys, xs = np.nonzero(lab)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def _records_from_dir(img_dir, anno_dir, ids=None, ext='.jpg'):
    by_image = {}
    for path in sorted(glob.glob(os.path.join(anno_dir, '*.png'))):
        m = _ANNO_RE.match(os.path.basename(path))
        if not m:
            continue
        by_image.setdefault(m.group('id'), []).append(path)
    records = []
    for img_id in (sorted(by_image) if ids is None else ids):
        img_path = os.path.join(img_dir, img_id + ext)
        if not os.path.isfile(img_path):
            alt = glob.glob(os.path.join(img_dir, img_id + '.*'))
            if not alt:
                continue
            img_path = alt[0]
        w, h = _image_size(img_path)
        persons = []
        for png in sorted(by_image.get(img_id, [])):
            bbox = _mask_bbox(png)
            if bbox is None:
                continue
            persons.append({'bbox': bbox, 'keypoints': None, 'segmentation': None, 'part': png})
        records.append({'image': img_path, 'width': w, 'height': h, 'id': img_id, 'persons': persons,
                        'ignore': []})
    return records


def mhp_v2_records(root, split):
    base = os.path.join(root, 'mhp_v2', split)
    return _records_from_dir(os.path.join(base, 'images'), os.path.join(base, 'parsing_annos'))


def mhp_v1_records(root, split):
    base = os.path.join(root, 'mhp_v1')
    list_file = os.path.join(base, f'{split}_list.txt')
    ids = None
    if os.path.isfile(list_file):
        with open(list_file) as f:
            ids = [os.path.splitext(line.strip())[0] for line in f if line.strip()]
    return _records_from_dir(os.path.join(base, 'images'), os.path.join(base, 'annotations'), ids)


def load_part_png(path):
    lab = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(path)
    return lab if lab.ndim == 2 else lab[:, :, 0]
