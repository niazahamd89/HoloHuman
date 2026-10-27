"""COCO 2017 person keypoints + instance masks (Section 4, "COCO") and CrowdPose
(Section 4, "CrowdPose"; same JSON format with 14 keypoints).

Expected layout (cfg.DATASET.ROOT = data):
    data/coco/images/{train2017,val2017,test2017}/*.jpg
    data/coco/annotations/person_keypoints_{train2017,val2017}.json, image_info_test-dev2017.json
    data/crowdpose/images/*.jpg
    data/crowdpose/json/crowdpose_{train,val,trainval,test}.json
"""
import os

import numpy as np
from pycocotools.coco import COCO


def coco_records(root, split, with_masks=True, require_keypoints=True):
    """Training records of the person category.  Crowd annotations become ignore regions
    (Section 4: only the person category is used)."""
    ann_file = os.path.join(root, 'coco', 'annotations', f'person_keypoints_{split}.json')
    img_dir = os.path.join(root, 'coco', 'images', split)
    coco = COCO(ann_file)
    cat = coco.getCatIds(catNms=['person'])
    records = []
    for img_id in sorted(coco.getImgIds(catIds=cat)):
        info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat, iscrowd=None))
        persons, ignore = [], []
        for a in anns:
            if a.get('iscrowd', 0):
                ignore.append(a['segmentation'])
                continue
            kpts = np.asarray(a['keypoints'], dtype=np.float32).reshape(-1, 3)
            if require_keypoints and a.get('num_keypoints', int((kpts[:, 2] > 0).sum())) == 0:
                continue
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            persons.append({'bbox': list(a['bbox']), 'keypoints': kpts,
                            'segmentation': a['segmentation'] if with_masks else None,
                            'area': a['area']})
        if persons:
            records.append({'image': os.path.join(img_dir, info['file_name']), 'width': info['width'],
                            'height': info['height'], 'id': img_id, 'persons': persons, 'ignore': ignore})
    return records


def coco_test_records(root, split):
    """Evaluation images of val2017 (annotated) or test-dev2017 (image_info json)."""
    if split.startswith('test'):
        ann_file = os.path.join(root, 'coco', 'annotations', 'image_info_test-dev2017.json')
        img_dir = os.path.join(root, 'coco', 'images', 'test2017')
    else:
        ann_file = os.path.join(root, 'coco', 'annotations', f'person_keypoints_{split}.json')
        img_dir = os.path.join(root, 'coco', 'images', split)
    coco = COCO(ann_file)
    records = [{'image': os.path.join(img_dir, info['file_name']), 'width': info['width'],
                'height': info['height'], 'id': info['id'], 'persons': []}
               for info in coco.loadImgs(sorted(coco.getImgIds()))]
    return records, ann_file


def crowdpose_records(root, split):
    ann_file = os.path.join(root, 'crowdpose', 'json', f'crowdpose_{split}.json')
    img_dir = os.path.join(root, 'crowdpose', 'images')
    coco = COCO(ann_file)
    records = []
    for img_id in sorted(coco.getImgIds()):
        info = coco.loadImgs(img_id)[0]
        persons = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            kpts = np.asarray(a['keypoints'], dtype=np.float32).reshape(-1, 3)
            if (kpts[:, 2] > 0).sum() == 0 or a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            persons.append({'bbox': list(a['bbox']), 'keypoints': kpts, 'segmentation': None})
        records.append({'image': os.path.join(img_dir, info['file_name']), 'width': info['width'],
                        'height': info['height'], 'id': img_id, 'persons': persons, 'ignore': []})
    return records, ann_file
