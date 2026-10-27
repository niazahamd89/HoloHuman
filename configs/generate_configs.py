"""Writes the configuration file of every checkpoint of Table 1 and of every ablation row of
Tables 2-6 and 8 (Section 4), as overrides of the defaults in humvis/config.py.

    python configs/generate_configs.py            # (re)generates configs/**/*.yaml

Layout
    configs/base.yaml                     shared training / testing settings
    configs/main/<benchmark>_<w32|w48>.yaml       C-pose/seg, C-part, C-mesh, C-crowd, C-mhp1 (Table 1)
    configs/eval/*.yaml                   cross-dataset evaluations (CrowdPose ‡, MHP v1.0 ‡, Panoptic, test-dev)
    configs/ablation/table4/row<k>_<benchmark>.yaml      component ablation (rows 0-7; row 7 = full model)
    configs/ablation/table2_3/iia_as_<benchmark>.yaml    the I_ij + A_s row of Tables 2-3 (not in Table 4)
    configs/ablation/table5/{loss,lambda,tau}_*_<benchmark>.yaml   loss terms, lambda_I and tau sweeps
    configs/ablation/table6/{pe,bias,both}_<benchmark>.yaml        positional variants of IIA
    configs/ablation/table8/<backbone>_coco.yaml                   backbone analysis
Every ablation is trained on HRNet-W48 with the full schedule, one checkpoint per row and
benchmark (Table 1, "Ablation").
"""
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from humvis.datasets.joints import (CROWDPOSE_FLIP_INDEX, CROWDPOSE_SIGMAS, MHP_V2_FLIP_INDEX,
                                    MHP_V1_FLIP_INDEX)  # noqa: E402

PRETRAINED = {'hrnet32': 'model_files/hrnetv2_w32_imagenet_pretrained.pth',
              'hrnet48': 'model_files/hrnetv2_w48_imagenet_pretrained.pth',
              'resnet50': 'torchvision', 'resnet101': 'torchvision', 'vit': 'timm', 'vit_large': 'timm'}

BENCHMARKS = {
    'coco': {'DATASET': {'DATASET': 'coco', 'TRAIN': 'train2017', 'TEST': 'val2017', 'NUM_KEYPOINTS': 17},
             'MODEL': {'HEADS': {'TASKS': ['pose', 'seg']}},
             'note': 'C-pose/seg: pose and segmentation heads trained jointly on COCO train2017 '
                     '(Tables 10, 12, 13, 22; full rows of Tables 2, 4-6 for W48)'},
    'mhp2': {'DATASET': {'DATASET': 'mhp2', 'TRAIN': 'train', 'TEST': 'val', 'NUM_PARTS': 58,
                         'PART_FLIP_INDEX': MHP_V2_FLIP_INDEX},
             'MODEL': {'HEADS': {'TASKS': ['part']}},
             'note': 'C-part: part head trained on MHP v2.0 train (Table 15; Table 14 ‡ cross-dataset; '
                     'full rows of Tables 3-6 for W48)'},
    'mesh': {'DATASET': {'DATASET': 'mesh', 'MESH_TRAIN': ['muco', 'coco_smpl', '3dpw_train'],
                         'MESH_TRAIN_WEIGHTS': [1.0, 1.0, 1.0], 'MESH_TEST': '3dpw_test'},
             'MODEL': {'HEADS': {'TASKS': ['mesh']}},
             'note': 'C-mesh: mesh head trained on MuCo-3DHP + COCO (pseudo-SMPL) + 3DPW train, evaluated on '
                     '3DPW test (Table 16) and CMU-Panoptic (Table 17); full rows of Tables 3-6 for W48'},
    'crowdpose': {'DATASET': {'DATASET': 'crowdpose', 'TRAIN': 'trainval', 'TEST': 'test', 'NUM_KEYPOINTS': 14,
                              'FLIP_INDEX': CROWDPOSE_FLIP_INDEX},
                  'MODEL': {'HEADS': {'TASKS': ['pose']}},
                  'TEST': {'OKS_SIGMAS': CROWDPOSE_SIGMAS},
                  'note': 'C-crowd: pose head trained on CrowdPose trainval with the 14 native keypoints '
                          '(Table 11 §, in-domain)'},
    'mhp1': {'DATASET': {'DATASET': 'mhp1', 'TRAIN': 'train', 'TEST': 'val', 'NUM_PARTS': 18,
                         'PART_FLIP_INDEX': MHP_V1_FLIP_INDEX},
             'MODEL': {'HEADS': {'TASKS': ['part']}},
             'note': 'C-mhp1: part head trained on MHP v1.0 train (18 categories), initialized from C-part '
                     '(Table 14 §, in-domain)'},
}

# Table 4 rows: (IIA, A_s, A_c, E(r), full F)
TABLE4 = {0: (False, False, False, True, True), 1: (False, True, False, True, True),
          2: (False, False, True, True, True), 3: (False, True, True, True, True),
          4: (True, False, False, True, True), 5: (True, True, True, False, True),
          6: (True, True, True, True, False), 7: (True, True, True, True, True)}
TABLE4_RESULTS = {0: '62.6 AP / 46.3 Seg AP / 41.1 AP^p_50 / 91.5 mm', 1: '64.5 / 48.3 / 42.3 / 90.3',
                  2: '66.3 / 50.5 / 44.1 / 84.1', 3: '67.7 / 52.0 / 47.7 / 80.5', 4: '68.0 / 53.7 / 48.5 / 78.8',
                  5: '73.9 / 55.8 / 50.2 / 69.7', 6: '77.4 / 58.8 / 53.5 / 63.8', 7: '79.3 / 61.0 / 55.5 / 58.5'}
TABLE5_LOSS = {'fl_int': (True, False, True), 'fl_ctr': (True, True, False), 'fl_ctr_int': (True, True, True),
               'ctr_int': (False, True, True), 'fl': (True, False, False)}
TABLE5_LAMBDA = [0.0, 0.01, 0.1, 0.5, 1.0]
TABLE5_TAU = [0.01, 0.1, 0.05, 0.5, 1.0]
TABLE6 = {'pe': 'pe', 'bias': 'bias', 'both': 'both'}
TABLE8 = ['resnet50', 'resnet101', 'vit', 'vit_large']


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def write(path, cfg, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg = dict(cfg)
    cfg['CFG_NAME'] = os.path.splitext(os.path.relpath(path, HERE))[0].replace(os.sep, '_')
    with open(path, 'w') as f:
        f.write('# ' + header.replace('\n', '\n# ') + '\n')
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=100)


def model_cfg(backbone, iia=True, As=True, Ac=True, Er=True, full=True, positional='none'):
    return {'MODEL': {'BACKBONE': backbone, 'PRETRAINED': PRETRAINED[backbone],
                      'IIA': {'ENABLED': iia, 'POSITIONAL': positional},
                      'GFD': {'SPATIAL_ATTENTION': As, 'CHANNEL_ATTENTION': Ac, 'CENTER_OFFSET': Er,
                              'WINDOW': 0 if full else 128}}}


def benchmark_cfg(bench, backbone, **model_kw):
    b = BENCHMARKS[bench]
    cfg = {'DATASET': copy.deepcopy(b['DATASET']), 'MODEL': {'HEADS': copy.deepcopy(b['MODEL']['HEADS'])}}
    if 'TEST' in b:
        cfg['TEST'] = copy.deepcopy(b['TEST'])
    deep_update(cfg, model_cfg(backbone, **model_kw))
    return cfg


def main():
    base = {'OUTPUT_DIR': 'output', 'WORKERS': 8, 'PRINT_FREQ': 50, 'AUTO_RESUME': True, 'SEED': 0,
            'MODEL': {'DEVICE': 'cuda'}, 'DATASET': {'ROOT': 'data'},
            'TRAIN': {'OPTIMIZER': 'adam', 'LR': 0.001, 'BACKBONE_LR': 0.0001, 'BATCH_SIZE': 64,
                      'END_EPOCH': 140, 'LR_STEPS': [90, 120], 'LR_FACTOR': 0.1},
            'TEST': {'FLIP_TEST': True, 'CENTER_THRESHOLD': 0.1, 'MAX_INSTANCES': 30, 'NMS_KERNEL': 3}}
    write(os.path.join(HERE, 'base.yaml'), base,
          'Shared settings of Section 4 (every other file lists its overrides; train.py / eval.py merge the\n'
          'file on top of humvis/config.py defaults, so base.yaml documents the protocol: Adam, lr 1e-3 / 1e-4\n'
          '(backbone), batch 64, 140 epochs, x0.1 at 90 and 120, single-scale flip test, threshold 0.1, <= 30).')

    # ---- Table 1 checkpoints ----------------------------------------------------------------------
    for bench in BENCHMARKS:
        for backbone in ('hrnet32', 'hrnet48'):
            w = backbone[-2:]
            cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, backbone))
            if bench == 'mhp1':
                cfg['TRAIN'] = dict(cfg['TRAIN'], INIT_CHECKPOINT=f'output/main_mhp2_w{w}/final_state.pth')
            write(os.path.join(HERE, 'main', f'{bench}_w{w}.yaml'), cfg,
                  f'Table 1, {BENCHMARKS[bench]["note"]}. Backbone HRNet-W{w}.')

    # ---- cross-dataset / server evaluations (same weights, different test set) ----------------------
    for backbone in ('hrnet32', 'hrnet48'):
        w = backbone[-2:]
        cfg = deep_update(copy.deepcopy(base), benchmark_cfg('coco', backbone))
        deep_update(cfg, {'DATASET': {'DATASET': 'crowdpose', 'TEST': 'test'},
                          'TEST': {'MODEL_FILE': f'output/main_coco_w{w}/final_state.pth'}})
        write(os.path.join(HERE, 'eval', f'crowdpose_cross_w{w}.yaml'), cfg,
              f'Table 11 ‡: C-pose/seg (COCO train2017, HRNet-W{w}) evaluated on CrowdPose test without training\n'
              'on CrowdPose; 17 -> 14 keypoints (eval.py converts automatically).')
        cfg = deep_update(copy.deepcopy(base), benchmark_cfg('coco', backbone))
        deep_update(cfg, {'DATASET': {'TEST': 'test-dev2017'},
                          'TEST': {'MODEL_FILE': f'output/main_coco_w{w}/final_state.pth'}})
        write(os.path.join(HERE, 'eval', f'coco_testdev_w{w}.yaml'), cfg,
              f'Tables 10 and 13: C-pose/seg (HRNet-W{w}) on COCO test-dev2017 (writes the json for the server).')
        cfg = deep_update(copy.deepcopy(base), benchmark_cfg('mhp2', backbone))
        deep_update(cfg, {'DATASET': {'DATASET': 'mhp1', 'TEST': 'val'},
                          'TEST': {'MODEL_FILE': f'output/main_mhp2_w{w}/final_state.pth'}})
        write(os.path.join(HERE, 'eval', f'mhp1_cross_w{w}.yaml'), cfg,
              f'Table 14 ‡: C-part (MHP v2.0, HRNet-W{w}) evaluated on MHP v1.0 val / test (DATASET.TEST) with the\n'
              '58 -> 18 label mapping of humvis/datasets/joints.py (MHP_V2_TO_V1).')
        cfg = deep_update(copy.deepcopy(base), benchmark_cfg('mesh', backbone))
        deep_update(cfg, {'DATASET': {'MESH_TEST': 'panoptic'},
                          'TEST': {'MODEL_FILE': f'output/main_mesh_w{w}/final_state.pth'}})
        write(os.path.join(HERE, 'eval', f'panoptic_w{w}.yaml'), cfg,
              f'Table 17: C-mesh (HRNet-W{w}) evaluated on CMU-Panoptic (Haggling, Mafia, Ultimatum, Pizza) '
              'without training on Panoptic.')

    # ---- Table 4 (and the rows of Tables 2-3 that it contains) ----------------------------------------
    for row, (iia, As, Ac, Er, full) in TABLE4.items():
        for bench in ('coco', 'mhp2', 'mesh'):
            cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, 'hrnet48', iia=iia, As=As, Ac=Ac, Er=Er, full=full))
            write(os.path.join(HERE, 'ablation', 'table4', f'row{row}_{bench}.yaml'), cfg,
                  f'Table 4 row {row} (HRNet-W48): I_ij={iia}, A_s={As}, A_c={Ac}, E(r)={Er}, full F={full}.\n'
                  f'Reported: {TABLE4_RESULTS[row]}.' +
                  (' Row 7 is the full model = configs/main/<benchmark>_w48.yaml.' if row == 7 else ''))
    for bench in ('coco', 'mhp2', 'mesh'):
        cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, 'hrnet48', iia=True, As=True, Ac=False))
        write(os.path.join(HERE, 'ablation', 'table2_3', f'iia_as_{bench}.yaml'), cfg,
              'Tables 2-3, row "I_ij + A_s" (HRNet-W48; the other rows are Table 4 rows 1, 2, 3, 4 and 7).')

    # ---- Table 5 ----------------------------------------------------------------------------------------
    for name, (fl, ctr, inter) in TABLE5_LOSS.items():
        for bench in ('coco', 'mhp2', 'mesh'):
            cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, 'hrnet48'))
            cfg['LOSS'] = {'CENTER_FOCAL': fl, 'CONTRASTIVE': ctr, 'INTERACTION': inter,
                           'INTERACTION_WEIGHT': 0.1, 'TEMPERATURE': 0.05}
            write(os.path.join(HERE, 'ablation', 'table5', f'loss_{name}_{bench}.yaml'), cfg,
                  f'Table 5, left block (HRNet-W48): FL={fl}, Ctr={ctr}, Int={inter}, lambda_I=0.1, tau=0.05.' +
                  (' Full configuration = main/<benchmark>_w48.yaml.' if name == 'fl_ctr_int' else ''))
    for lam in TABLE5_LAMBDA:
        for bench in ('coco', 'mhp2', 'mesh'):
            cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, 'hrnet48'))
            cfg['LOSS'] = {'CENTER_FOCAL': True, 'CONTRASTIVE': True, 'INTERACTION': True,
                           'INTERACTION_WEIGHT': lam, 'TEMPERATURE': 0.05}
            write(os.path.join(HERE, 'ablation', 'table5', f'lambda_{lam:g}_{bench}.yaml'), cfg,
                  f'Table 5, middle block (HRNet-W48): lambda_I = {lam:g}, tau = 0.05, all three loss terms.')
    for tau in TABLE5_TAU:
        for bench in ('coco', 'mhp2', 'mesh'):
            cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, 'hrnet48'))
            cfg['LOSS'] = {'CENTER_FOCAL': True, 'CONTRASTIVE': True, 'INTERACTION': True,
                           'INTERACTION_WEIGHT': 0.1, 'TEMPERATURE': tau}
            write(os.path.join(HERE, 'ablation', 'table5', f'tau_{tau:g}_{bench}.yaml'), cfg,
                  f'Table 5, right block (HRNet-W48): tau = {tau:g}, lambda_I = 0.1, all three loss terms.')

    # ---- Table 6 ----------------------------------------------------------------------------------------
    for name, pos in TABLE6.items():
        for bench in ('coco', 'mhp2', 'mesh'):
            cfg = deep_update(copy.deepcopy(base), benchmark_cfg(bench, 'hrnet48', positional=pos))
            write(os.path.join(HERE, 'ablation', 'table6', f'{name}_{bench}.yaml'), cfg,
                  f'Table 6 (HRNet-W48): IIA + {name.upper() if name != "both" else "PE + B"} '
                  f'(MODEL.IIA.POSITIONAL = {pos}). "No IIA" = table4/row3, "content only" = main/<benchmark>_w48.')

    # ---- Table 8 ----------------------------------------------------------------------------------------
    for backbone in TABLE8:
        cfg = deep_update(copy.deepcopy(base), benchmark_cfg('coco', backbone))
        write(os.path.join(HERE, 'ablation', 'table8', f'{backbone}_coco.yaml'), cfg,
              f'Table 8 (COCO keypoint val): backbone {backbone} (humvis/backbones_extra.py); HR-W32 / HR-W48 rows '
              '= main/coco_w32.yaml / main/coco_w48.yaml.')
    print('configs written under', HERE)


if __name__ == '__main__':
    main()
