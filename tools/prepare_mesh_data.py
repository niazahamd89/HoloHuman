"""Convert the mesh benchmarks to the unified annotation files read by humvis/datasets/mesh.py
(data/mesh/<name>.pkl, format documented there).

    # 3DPW (Von Marcard et al., 2018): official sequenceFiles/{train,validation,test}/*.pkl
    python tools/prepare_mesh_data.py 3dpw --root data/3dpw --split train --out data/mesh/3dpw_train.pkl
    python tools/prepare_mesh_data.py 3dpw --root data/3dpw --split test  --out data/mesh/3dpw_test.pkl

    # MuCo-3DHP with the SMPL fits released with I2L-MeshNet / Pose2Mesh (Moon et al.):
    #   MuCo-3DHP.json (COCO-style, 'keypoints_img' / 'keypoints_cam', 'bbox') + smpl_param.json
    python tools/prepare_mesh_data.py muco --root data/muco --out data/mesh/muco.pkl

    # COCO train2017 with pseudo-ground-truth SMPL fits:
    #   --format i2l : coco_smplify_train.json keyed by annotation id ({'pose', 'shape', 'trans'})
    #   --format eft : EFT fits (COCO2014-All-ver01.json, 'parm_pose' as 24x3x3 rotation matrices)
    python tools/prepare_mesh_data.py coco_smpl --root data/coco --fits data/coco/coco_smplify_train.json \
        --format i2l --out data/mesh/coco_smpl.pkl

    # CMU-Panoptic (Joo et al., 2015) from the Panoptic toolbox layout (hdPose3d_stage1_coco19,
    # calibration_<seq>.json, hdImgs/<cam>/), four sequences and the HD views / frame step
    # of the protocol of Jiang et al. (2020):
    python tools/prepare_mesh_data.py panoptic --root data/panoptic --cameras 00_16 00_30 --step 25 \
        --out data/mesh/panoptic.pkl

The records store SMPL parameters in the camera frame (global orientation composed with the
camera rotation), 2D joints in LSP14 order and, for evaluation, gender and 3D joints.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from humvis.datasets.joints import (OPENPOSE18_TO_J14, MUCO17_TO_J14, PANOPTIC19_TO_J14, coco_to_j14,
                                    remap_joints)  # noqa: E402

SMPL_24_TO_J14 = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]   # SMPL joints -> LSP14 (approx. neck/head)


def rotate_pose(pose, R):
    pose = np.asarray(pose, dtype=np.float32).copy()
    rot, _ = cv2.Rodrigues(pose[:3])
    pose[:3] = cv2.Rodrigues(R @ rot)[0].reshape(3)
    return pose


def bbox_from_points(pts, margin=0.1):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    return [float(x0 - margin * w), float(y0 - margin * h), float(w * (1 + 2 * margin)), float(h * (1 + 2 * margin))]


def project(points_cam, K):
    p = points_cam @ K.T
    return p[:, :2] / np.clip(p[:, 2:3], 1e-6, None)


# ---------------------------------------------------------------- 3DPW
def convert_3dpw(root, split, out):
    seq_dir = os.path.join(root, 'sequenceFiles', split)
    records = []
    for pkl in sorted(glob.glob(os.path.join(seq_dir, '*.pkl'))):
        with open(pkl, 'rb') as f:
            seq = pickle.load(f, encoding='latin1')
        name = seq['sequence']
        K = np.asarray(seq['cam_intrinsics'], dtype=np.float64)
        n_frames = len(seq['img_frame_ids'])
        n_actors = len(seq['poses'])
        for fi in range(n_frames):
            frame_id = int(seq['img_frame_ids'][fi])
            img = os.path.join('3dpw', 'imageFiles', name, f'image_{frame_id:05d}.jpg')
            cam = np.asarray(seq['cam_poses'][fi], dtype=np.float64)
            R, t = cam[:3, :3], cam[:3, 3]
            persons = []
            for a in range(n_actors):
                if not seq['campose_valid'][a][fi]:
                    continue
                pose = rotate_pose(seq['poses'][a][fi], R)
                shape = np.asarray(seq['betas'][a][:10], dtype=np.float32)
                j3d_world = np.asarray(seq['jointPositions'][a][fi], dtype=np.float64).reshape(24, 3)
                j3d_cam = j3d_world @ R.T + t
                j2d_proj = project(j3d_cam, K)
                op = np.asarray(seq['poses2d'][a][fi], dtype=np.float32).T          # (18, 3)
                j2d = remap_joints(op[None], OPENPOSE18_TO_J14)[0]
                j2d[:, 2] = (j2d[:, 2] > 0.3).astype(np.float32)
                persons.append({'bbox': bbox_from_points(j2d_proj), 'joints_2d': j2d,
                                'smpl_pose': pose, 'smpl_shape': shape,
                                'gender': str(seq['genders'][a]),
                                'joints_3d': j3d_cam[SMPL_24_TO_J14].astype(np.float32),
                                'joints_3d_valid': np.ones(14, dtype=bool)})
            if persons:
                records.append({'image': img, 'width': int(2 * K[0, 2]), 'height': int(2 * K[1, 2]),
                                'id': f'{name}/{frame_id}', 'persons': persons})
    _save(records, out)


# ---------------------------------------------------------------- MuCo-3DHP
def convert_muco(root, out, ann='MuCo-3DHP.json', fits='smpl_param.json'):
    with open(os.path.join(root, ann)) as f:
        data = json.load(f)
    with open(os.path.join(root, fits)) as f:
        smpl = json.load(f)
    images = {im['id']: im for im in data['images']}
    by_image = {}
    for a in data['annotations']:
        by_image.setdefault(a['image_id'], []).append(a)
    records = []
    for img_id, anns in by_image.items():
        im = images[img_id]
        persons = []
        for a in anns:
            kp = np.asarray(a['keypoints_img'], dtype=np.float32).reshape(-1, 2)[:17]
            vis = np.asarray(a.get('keypoints_vis', [1] * len(kp)), dtype=np.float32).reshape(-1)[:17]
            kp3 = np.concatenate((kp, vis[:, None]), axis=1)
            j2d = remap_joints(kp3[None], MUCO17_TO_J14)[0]
            fit = smpl.get(str(a['id']))
            pose = np.asarray(fit['pose'], dtype=np.float32).reshape(72) if fit else None
            shape = np.asarray(fit['shape'], dtype=np.float32).reshape(10) if fit else None
            persons.append({'bbox': list(map(float, a['bbox'])), 'joints_2d': j2d,
                            'smpl_pose': pose, 'smpl_shape': shape})
        records.append({'image': os.path.join('muco', 'images', im['file_name']), 'width': im['width'],
                        'height': im['height'], 'id': img_id, 'persons': persons})
    _save(records, out)


# ---------------------------------------------------------------- COCO + pseudo-SMPL
def convert_coco_smpl(root, fits, fmt, out, split='train2017'):
    from pycocotools.coco import COCO
    coco = COCO(os.path.join(root, 'annotations', f'person_keypoints_{split}.json'))
    with open(fits) as f:
        data = json.load(f)
    if fmt == 'i2l':
        fit_by_ann = {int(k): {'pose': np.asarray(v['pose'], dtype=np.float32).reshape(72),
                               'shape': np.asarray(v['shape'], dtype=np.float32).reshape(10)}
                      for k, v in data.items()}
    elif fmt == 'eft':
        fit_by_ann = {}
        for d in data['data']:
            rot = np.asarray(d['parm_pose'], dtype=np.float32).reshape(24, 3, 3)
            pose = np.concatenate([cv2.Rodrigues(r)[0].reshape(3) for r in rot])
            fit_by_ann[int(d['annotId'])] = {'pose': pose.astype(np.float32),
                                             'shape': np.asarray(d['parm_shape'], dtype=np.float32).reshape(10)}
    else:
        raise ValueError(fmt)
    records = []
    for img_id in sorted(coco.getImgIds(catIds=[1])):
        im = coco.loadImgs(img_id)[0]
        persons = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)):
            kp = np.asarray(a['keypoints'], dtype=np.float32).reshape(17, 3)
            if (kp[:, 2] > 0).sum() == 0:
                continue
            fit = fit_by_ann.get(a['id'])
            persons.append({'bbox': list(a['bbox']), 'joints_2d': coco_to_j14(kp[None])[0],
                            'smpl_pose': fit['pose'] if fit else None,
                            'smpl_shape': fit['shape'] if fit else None})
        if persons:
            records.append({'image': os.path.join('coco', 'images', split, im['file_name']),
                            'width': im['width'], 'height': im['height'], 'id': img_id, 'persons': persons})
    _save(records, out)


# ---------------------------------------------------------------- CMU-Panoptic
def convert_panoptic(root, sequences, cameras, step, out):
    records = []
    for seq in sequences:
        seq_dir = os.path.join(root, seq)
        with open(os.path.join(seq_dir, f'calibration_{seq}.json')) as f:
            calib = {c['name']: c for c in json.load(f)['cameras']}
        pose_files = sorted(glob.glob(os.path.join(seq_dir, 'hdPose3d_stage1_coco19', 'body3DScene_*.json')))
        for pf in pose_files[::step]:
            frame = int(os.path.basename(pf)[len('body3DScene_'):-len('.json')])
            with open(pf) as f:
                bodies = json.load(f)['bodies']
            for cam_name in cameras:
                c = calib[cam_name]
                R = np.asarray(c['R'], dtype=np.float64)
                t = np.asarray(c['t'], dtype=np.float64).reshape(3) / 100.0        # cm -> m
                K = np.asarray(c['K'], dtype=np.float64)
                persons = []
                for b in bodies:
                    j19 = np.asarray(b['joints19'], dtype=np.float64).reshape(19, 4)
                    xyz = j19[:, :3] / 100.0 @ R.T + t                               # camera frame, metres
                    conf = j19[:, 3]
                    j14 = np.zeros((14, 3), dtype=np.float32)
                    valid = np.zeros(14, dtype=bool)
                    j14[:13] = xyz[PANOPTIC19_TO_J14]
                    valid[:13] = conf[PANOPTIC19_TO_J14] > 0
                    if valid.sum() < 6:
                        continue
                    p2d = project(xyz[conf > 0], K)
                    j2d = np.zeros((14, 3), dtype=np.float32)
                    j2d[:13, :2] = project(xyz[PANOPTIC19_TO_J14], K)
                    j2d[:13, 2] = valid[:13]
                    persons.append({'bbox': bbox_from_points(p2d), 'joints_2d': j2d, 'joints_3d': j14,
                                    'joints_3d_valid': valid, 'gender': 'n'})
                if persons:
                    records.append({'image': os.path.join('panoptic', seq, 'hdImgs', cam_name,
                                                          f'{cam_name}_{frame:08d}.jpg'),
                                    'width': int(2 * K[0, 2]), 'height': int(2 * K[1, 2]),
                                    'id': f'{seq}/{cam_name}/{frame}', 'sequence': seq, 'persons': persons})
    _save(records, out)


def _save(records, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(records, f)
    n = sum(len(r['persons']) for r in records)
    print(f'{out}: {len(records)} images, {n} persons')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('3dpw'); p.add_argument('--root', required=True); p.add_argument('--split', default='train')
    p.add_argument('--out', required=True)
    p = sub.add_parser('muco'); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--ann', default='MuCo-3DHP.json'); p.add_argument('--fits', default='smpl_param.json')
    p = sub.add_parser('coco_smpl'); p.add_argument('--root', required=True); p.add_argument('--fits', required=True)
    p.add_argument('--format', choices=['i2l', 'eft'], default='i2l'); p.add_argument('--out', required=True)
    p = sub.add_parser('panoptic'); p.add_argument('--root', required=True)
    p.add_argument('--sequences', nargs='+', default=['160422_haggling1', '160422_mafia2', '160422_ultimatum1',
                                                      '160906_pizza1'])
    p.add_argument('--cameras', nargs='+', required=True, help='HD views of the protocol, e.g. 00_16 00_30')
    p.add_argument('--step', type=int, default=25, help='frame sampling step of the protocol')
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == '3dpw':
        convert_3dpw(a.root, a.split, a.out)
    elif a.cmd == 'muco':
        convert_muco(a.root, a.out, a.ann, a.fits)
    elif a.cmd == 'coco_smpl':
        convert_coco_smpl(a.root, a.fits, a.format, a.out)
    elif a.cmd == 'panoptic':
        convert_panoptic(a.root, a.sequences, a.cameras, a.step, a.out)


if __name__ == '__main__':
    main()
