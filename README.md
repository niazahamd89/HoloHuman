# HoloHuman: training and testing

HoloHuman is a single-stage network for multi-person human analysis. One forward pass predicts,
for every person in the image, the 2D keypoints, the instance mask, the body-part map and the SMPL
body mesh:

```
image ─► HRNet ─► shared map F ─► IIA ─► {F_e(i), r(i)} ─► GFD ─► {F'(i)} ─► pose / seg / part / mesh heads
          (once per image)        (center head, Peak, descriptor pooling,       (once per detected person,
                                   self-attention over the m persons)           weights shared across persons)
```

This package contains the model, the training code with the losses of the paper, the data loaders
of every benchmark, the evaluation code with the official metrics, the configuration file of every
checkpoint and ablation row, and the profiling and timing scripts. The sections below go from
installation to the commands behind each table and figure.

**Contents**

1. [Installation](#1-installation)
2. [Model files](#2-model-files)
3. [Data preparation](#3-data-preparation)
4. [Configuration system](#4-configuration-system)
5. [Training](#5-training)
6. [Testing and evaluation](#6-testing-and-evaluation)
7. [Ablation studies](#7-ablation-studies)
8. [Computational cost and timing](#8-computational-cost-and-timing)
9. [Demo and inference API](#9-demo-and-inference-api)
10. [Reference values from the paper](#10-reference-values-from-the-paper)
11. [Implementation choices the paper leaves open](#11-implementation-choices-the-paper-leaves-open)
12. [Code ↔ paper map](#12-code--paper-map)
13. [Troubleshooting](#13-troubleshooting)

---

## Package layout

```
humvis/
  backbone.py          HRNet-W32 / W48 (shared map F, C = 480 / 720, stride 4); ImageNet initialization
  backbones_extra.py   ResNet-50 / 101 and ViT-B / ViT-L backbones (Table 8)
  iia.py               Instance Interaction Attention: center head, Peak, P_inst, self-attention (Eqs. 2-9)
  gfd.py               Global Feature Distillation: W_g, A_c, A_s, W_f (Eqs. 10-15)
  heads.py             pose / segmentation / part / mesh heads, SMPL forward
  holohuman.py         inference forward (single scale, flip test, Peak threshold 0.1, <= 30 persons)
  losses.py            loss functions of Eqs. 16-26
  training.py          training forward (ground-truth centers, flipped-view positives, all losses)
  config.py            every hyper-parameter with its default (the paper setting)
  datasets/            COCO, CrowdPose, MHP v1.0 / v2.0, mesh mixture; augmentation; keypoint/part conventions
  evaluation/          COCO / CrowdPose AP, MHP AP^p / PCP, 3DPW / Panoptic errors, timing helpers
configs/               base.yaml, main/ (Table 1), eval/ (cross-dataset, test-dev, Panoptic), ablation/
train.py               train one checkpoint
eval.py                evaluate one checkpoint on one benchmark
profile_cost.py        GMACs and parameters per component (Table 7, Table 4 columns)
time_fig4.py           latency / memory versus the number of persons (Figure 4)
time_fig5.py           latency per task configuration (Figure 5, Table 9)
tools/                 prepare_mesh_data.py, make_efficiency_fig.py, merge_checkpoints.py
scripts/               train_main.sh, eval_main.sh, train_ablations.sh, reproduce_cost_and_timing.sh
app.py, camera.py, video.py, test_app.py   demo applications
model_files/           SMPL files, ImageNet backbone weights, trained checkpoints (see model_files/README.txt)
```

---

## 1. Installation

Python 3.9 or newer and PyTorch 2.x with CUDA. The code was exercised with PyTorch 2.14 and
torchvision 0.29.

```bash
pip install torch torchvision                  # the build matching your CUDA version
pip install yacs pycocotools opencv-python scipy pyyaml matplotlib
pip install timm                               # only for the ViT rows of Table 8
pip install pytorch3d flask                    # only for the demo (mesh rendering, web app)
```

All commands are run from the package root (the folder that contains `train.py`).

---

## 2. Model files

Put the following files in `model_files/`:

| File | Needed for | Source |
|---|---|---|
| `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | mesh head (training, testing, demo) | SMPL website (registration required) |
| `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, `basicModel_f_lbs_10_207_0_v1.0.0.pkl` | 3DPW ground-truth meshes for MPJPE / PA-MPJPE / PVE | SMPL website |
| `J_regressor_h36m.npy` | 17-joint regressor used for the mesh joints | released with SPIN / HybrIK / ROMP |
| `hrnetv2_w32_imagenet_pretrained.pth`, `hrnetv2_w48_imagenet_pretrained.pth` | backbone initialization of every checkpoint | official HRNet ImageNet classification release |

The paths are set in `humvis/config.py` (`MODEL.SMPL_MODEL`, `MODEL.SMPL_MALE`, `MODEL.SMPL_FEMALE`,
`MODEL.SMPL_H36M_REGRESSOR`) and in the YAML files (`MODEL.PRETRAINED`). The ResNet backbones use the
torchvision ImageNet weights (`MODEL.PRETRAINED torchvision`), and the ViT backbones the timm weights
(`MODEL.PRETRAINED timm`); both are downloaded automatically.

SMPL is needed only by checkpoints with the mesh head. The pose, segmentation and part checkpoints
train and run without it.

---

## 3. Data preparation

Every dataset lives under `DATASET.ROOT` (default `data/`). Create the following layout, using
symbolic links if the data is stored elsewhere.

### COCO 2017 (pose and instance segmentation)

```
data/coco/images/train2017/*.jpg
data/coco/images/val2017/*.jpg
data/coco/images/test2017/*.jpg                         (test-dev evaluation only)
data/coco/annotations/person_keypoints_train2017.json
data/coco/annotations/person_keypoints_val2017.json
data/coco/annotations/image_info_test-dev2017.json      (test-dev evaluation only)
```

Only the person category is used. Training uses persons with at least one annotated keypoint;
crowd annotations (`iscrowd = 1`) become ignore regions.

### CrowdPose

```
data/crowdpose/images/*.jpg
data/crowdpose/json/crowdpose_trainval.json
data/crowdpose/json/crowdpose_test.json
```

The JSON files are the official CrowdPose annotations (COCO format, 14 keypoints, `crowdIndex` per
image, used for the easy / medium / hard splits).

### MHP v2.0 (58 part categories)

```
data/mhp_v2/train/images/<id>.jpg
data/mhp_v2/train/parsing_annos/<id>_<N:02d>_<k:02d>.png     one PNG per person, values 0..58
data/mhp_v2/val/images/<id>.jpg
data/mhp_v2/val/parsing_annos/<id>_<N:02d>_<k:02d>.png
```

This is the layout of the official LV-MHP v2.0 release: `N` is the number of persons in the image
and `k` the person index.

### MHP v1.0 (18 part categories)

```
data/mhp_v1/images/<id>.jpg
data/mhp_v1/annotations/<id>_<N:02d>_<k:02d>.png             values 0..18
data/mhp_v1/train_list.txt, val_list.txt, test_list.txt      image ids, one per line
```

### Mesh datasets (MuCo-3DHP, COCO pseudo-SMPL, 3DPW, CMU-Panoptic)

The mesh loaders read one unified annotation file per dataset, `data/mesh/<name>.pkl`. Write them
with `tools/prepare_mesh_data.py` from the official releases (image paths in the files are relative
to `data/`):

```bash
# 3DPW: official sequenceFiles/{train,test}/*.pkl and imageFiles/
python tools/prepare_mesh_data.py 3dpw --root data/3dpw --split train --out data/mesh/3dpw_train.pkl
python tools/prepare_mesh_data.py 3dpw --root data/3dpw --split test  --out data/mesh/3dpw_test.pkl

# MuCo-3DHP: MuCo-3DHP.json and smpl_param.json as released with I2L-MeshNet
python tools/prepare_mesh_data.py muco --root data/muco --out data/mesh/muco.pkl

# COCO train2017 with pseudo-ground-truth SMPL fits (I2L-MeshNet format, or EFT with --format eft)
python tools/prepare_mesh_data.py coco_smpl --root data/coco --fits data/coco/coco_smplify_train.json \
    --format i2l --out data/mesh/coco_smpl.pkl

# CMU-Panoptic: Panoptic toolbox layout (hdPose3d_stage1_coco19/, calibration_<seq>.json, hdImgs/)
python tools/prepare_mesh_data.py panoptic --root data/panoptic --cameras <HD views> --step <frame step> \
    --out data/mesh/panoptic.pkl
```

For Panoptic, pass the HD camera views and the frame step of the evaluation protocol you follow. The
default sequences are Haggling, Mafia, Ultimatum and Pizza. Each run prints the number of images and
persons written. The format of the files is documented at the top of `humvis/datasets/mesh.py`.

---

## 4. Configuration system

Configurations are layered:

1. **Defaults** in `humvis/config.py`: every hyper-parameter, set to the full model and the paper
   protocol, with a comment that names the equation or table it belongs to.
2. **A YAML file** passed with `--cfg`: only the settings that differ (backbone, heads, dataset,
   ablation switches). `configs/base.yaml` documents the shared protocol.
3. **Command-line overrides**: any `KEY VALUE` pairs after the other arguments, for example
   `python train.py --cfg configs/main/coco_w48.yaml TRAIN.BATCH_SIZE 32 WORKERS 4`.

The YAML files are generated by `configs/generate_configs.py`. Re-run it after changing it; don't
edit the generated files by hand. Each file starts with a comment naming the table and row it
produces, and sets `CFG_NAME`, which is also the name of its output folder.

| Group | Key | Default | Meaning |
|---|---|---|---|
| Model | `MODEL.BACKBONE` | `hrnet32` | `hrnet32`, `hrnet48`, `resnet50`, `resnet101`, `vit`, `vit_large` |
| | `MODEL.HEADS.TASKS` | all four | heads attached to the trunk, e.g. `[pose, seg]` |
| | `MODEL.PRETRAINED` | `''` | ImageNet backbone weights (set in every YAML) |
| IIA | `MODEL.IIA.ENABLED` | `True` | `False`: F_e(i) = f(i), no interaction (Table 4 rows 0-3) |
| | `MODEL.IIA.D` | 256 | attention width d |
| | `MODEL.IIA.POSITIONAL` | `none` | `pe`, `bias` (Eq. 9), `both` (Table 6) |
| GFD | `MODEL.GFD.CHANNEL_ATTENTION` | `True` | A_c branch |
| | `MODEL.GFD.SPATIAL_ATTENTION` | `True` | A_s branch |
| | `MODEL.GFD.CENTER_OFFSET` | `True` | center-offset encoding E(r) in Eq. 10 |
| | `MODEL.GFD.WINDOW` | 0 | 0 = decode from the whole map F; 128 = 128 x 128 window (Table 4 row 6) |
| Loss | `LOSS.CENTER_FOCAL`, `LOSS.CONTRASTIVE`, `LOSS.INTERACTION` | `True` | the three IIA loss terms (Table 5) |
| | `LOSS.INTERACTION_WEIGHT` | 0.1 | lambda_I |
| | `LOSS.TEMPERATURE` | 0.05 | tau |
| | `LOSS.*_WEIGHT` | 1, 1, 1, 1, 0.1 | lambda_IIA, lambda_pose, lambda_seg, lambda_part, lambda_mesh |
| Data | `DATASET.DATASET` | `coco` | `coco`, `crowdpose`, `mhp2`, `mhp1`, `mesh` |
| | `DATASET.TRAIN`, `DATASET.TEST` | | split names (e.g. `train2017`, `val2017`, `test-dev2017`) |
| | `DATASET.INPUT_SIZE` | 512 | training crop and test short side |
| Training | `TRAIN.BATCH_SIZE` | 64 | total batch, divided over the GPUs |
| | `TRAIN.LR`, `TRAIN.BACKBONE_LR` | 1e-3, 1e-4 | Adam learning rates |
| | `TRAIN.END_EPOCH`, `TRAIN.LR_STEPS` | 140, [90, 120] | schedule (x0.1 at each step) |
| | `TRAIN.INIT_CHECKPOINT` | `''` | initialize the weights from a checkpoint (C-mhp1) |
| | `SEED` | 0 | random seed of the run |
| Testing | `TEST.FLIP_TEST` | `True` in the YAMLs | flip test |
| | `TEST.CENTER_THRESHOLD`, `TEST.MAX_INSTANCES` | 0.1, 30 | person detection |
| | `TEST.MODEL_FILE` | | checkpoint used when `--weights` is not given |

---

## 5. Training

### 5.1 One checkpoint per benchmark

HoloHuman shares the backbone, center head, IIA and GFD across tasks; only the heads are task-specific.
As in the paper (Section 4, Table 1), one checkpoint is trained per benchmark, with the shared
components and the heads of that benchmark, starting from the ImageNet backbone. No module is frozen,
datasets are not mixed across checkpoints, and only the losses of the annotated tasks are active.

| Checkpoint | Config (W32 / W48) | Heads | Training data | Evaluated in |
|---|---|---|---|---|
| C-pose/seg | `configs/main/coco_w32.yaml` / `coco_w48.yaml` | pose, seg | COCO train2017 | Tables 10, 12, 13, 22; Table 11 (cross-dataset) |
| C-crowd | `configs/main/crowdpose_w32.yaml` / `crowdpose_w48.yaml` | pose (14 keypoints) | CrowdPose trainval | Table 11 (in-domain) |
| C-part | `configs/main/mhp2_w32.yaml` / `mhp2_w48.yaml` | part (58 + 1) | MHP v2.0 train | Table 15; Table 14 (cross-dataset) |
| C-mhp1 | `configs/main/mhp1_w32.yaml` / `mhp1_w48.yaml` | part (18 + 1) | MHP v1.0 train, initialized from C-part | Table 14 (in-domain) |
| C-mesh | `configs/main/mesh_w32.yaml` / `mesh_w48.yaml` | mesh | MuCo-3DHP + COCO pseudo-SMPL + 3DPW train | Tables 16, 17 |

### 5.2 Training protocol

| Setting | Value |
|---|---|
| Optimizer | Adam; lr 1e-3 for non-backbone parameters, 1e-4 for the backbone; no weight decay |
| Batch size | 64 images in total |
| Schedule | 140 epochs; lr x0.1 at epochs 90 and 120 |
| Input | 512 x 512 crops; output maps at stride 4 (128 x 128) |
| Augmentation | rotation ±30°, scale [0.75, 1.5], horizontal flip (p = 0.5), translation up to ±40 px |
| Instances | instantiated at the ground-truth person centers (at most 30 per image) |
| Losses | center focal loss, contrastive loss with the flipped-image positive, interaction regularization (λ_I = 0.1, τ = 0.05); focal (pose), Dice (seg), cross-entropy (part), SMPL parameters + 2D reprojection (mesh); weights 1 / 1 / 1 / 1 / 0.1 |
| Checkpoint evaluated | the one after the final (140th) epoch, `final_state.pth` |

### 5.3 Commands

Multi-GPU training (recommended; the batch of 64 is divided over the GPUs):

```bash
torchrun --nproc_per_node 8 train.py --cfg configs/main/coco_w48.yaml
```

Single GPU:

```bash
python train.py --cfg configs/main/coco_w32.yaml
```

A single GPU rarely fits 64 images of 512 x 512 with HRNet-W48. Lowering `TRAIN.BATCH_SIZE` makes the
run fit, but it is then no longer the paper protocol.

All checkpoints of Table 1, in the right order (C-mhp1 needs C-part):

```bash
GPUS=8 scripts/train_main.sh
```

C-mhp1 reads its initialization from `TRAIN.INIT_CHECKPOINT`, set to `output/main_mhp2_w{32,48}/final_state.pth`
in its config. To use another file:

```bash
torchrun --nproc_per_node 8 train.py --cfg configs/main/mhp1_w48.yaml --init path/to/c_part.pth
```

Tensors whose shape differs (the 59-class part output layer of C-part vs. the 19 classes of MHP v1.0)
are skipped and newly initialized; everything else is copied.

### 5.4 Outputs, resuming and logs

Each run writes to `output/<CFG_NAME>/`:

| File | Content |
|---|---|
| `final_state.pth` | weights after the last epoch (evaluate this one) |
| `checkpoint.pth` | weights, optimizer, scheduler and epoch, rewritten every epoch |
| `epoch_<n>.pth` | intermediate weights every `TRAIN.SAVE_EVERY` (10) epochs |
| `config.yaml` | the full merged configuration of the run |
| `train.log` | per-iteration losses: `center`, `contrastive`, `interaction`, one term per task, `iia`, `gfd`, `total` |

`AUTO_RESUME` is on, so re-running the same command after an interruption continues from
`checkpoint.pth`. To resume from a specific file, use `--resume path/to/checkpoint.pth`. To start
from scratch, delete the output folder or pass `AUTO_RESUME False`.

### 5.5 Repeated runs with different seeds

`SEED` sets the initial weights of the non-pretrained layers, the image order and the augmentation
draws. The same seed repeats a run (up to GPU non-determinism); a different seed gives an independent
run of the same configuration. **Give every repeated run its own `CFG_NAME`.** Otherwise the run uses
the folder of seed 0, auto-resume loads the finished seed-0 checkpoint, and nothing new is trained.

```bash
# Table 4, rows 3, 6 and 7 on COCO, seeds 1 and 2 (seed 0 is the original run)
for s in 1 2; do
  torchrun --nproc_per_node 8 train.py --cfg configs/ablation/table4/row3_coco.yaml SEED $s CFG_NAME t4_row3_coco_s$s
  torchrun --nproc_per_node 8 train.py --cfg configs/ablation/table4/row6_coco.yaml SEED $s CFG_NAME t4_row6_coco_s$s
  torchrun --nproc_per_node 8 train.py --cfg configs/main/coco_w48.yaml             SEED $s CFG_NAME t4_row7_coco_s$s
done
```

Evaluate each with the same `CFG_NAME` so that the metrics of seed 0 are not overwritten (Section 6.3).
Report mean ± sample standard deviation (`numpy.std(x, ddof=1)`) over seeds 0, 1 and 2. Set
`CUDNN.DETERMINISTIC True CUDNN.BENCHMARK False` if bit-for-bit repeatability matters more than speed.

### 5.6 Quick checks before a long run

```bash
# a few iterations on a small input, to check the data paths and the losses
python train.py --cfg configs/main/coco_w32.yaml --max-iters 5 \
    TRAIN.BATCH_SIZE 2 TRAIN.END_EPOCH 1 DATASET.INPUT_SIZE 256 DATASET.OUTPUT_SIZE 64 \
    WORKERS 0 CFG_NAME debug_coco
```

`--max-iters N` stops every epoch after N iterations. All losses should be finite. The `pose` and
`center` terms start high and drop within the first iterations; `contrastive` is zero on images with
one person.

---

## 6. Testing and evaluation

### 6.1 Protocol

| Setting | Value |
|---|---|
| Input | shorter image side resized to 512, aspect ratio kept, padded to a multiple of 32 |
| Test-time augmentation | single scale with horizontal flip; no multi-scale testing, no refinement |
| Persons | peaks of the center heatmap after 3 x 3 max-pool NMS, confidence > 0.1, at most 30 |
| Keypoints | maximum of each heatmap channel; OKS-NMS 0.9 over the persons |
| Masks / parts | upsampled to the image; mask threshold 0.5; part label = arg-max |
| Mesh | SMPL from the predicted parameters; predictions matched to ground truth by 2D box IoU ≥ 0.5 |

### 6.2 Commands per benchmark

```bash
# COCO val2017: keypoint AP and mask AP (Tables 22 and 12)
python eval.py --cfg configs/main/coco_w48.yaml --weights output/main_coco_w48/final_state.pth

# COCO test-dev2017: writes the result files for the evaluation server (Tables 10 and 13)
python eval.py --cfg configs/eval/coco_testdev_w48.yaml

# CrowdPose test, COCO-trained model (cross-dataset, Table 11 ‡); 17 -> 14 keypoints is automatic
python eval.py --cfg configs/eval/crowdpose_cross_w48.yaml

# CrowdPose test, CrowdPose-trained model (in-domain, Table 11 §)
python eval.py --cfg configs/main/crowdpose_w48.yaml --weights output/main_crowdpose_w48/final_state.pth

# MHP v2.0 val (Table 15)
python eval.py --cfg configs/main/mhp2_w48.yaml --weights output/main_mhp2_w48/final_state.pth

# MHP v1.0, MHP v2.0-trained model (cross-dataset, Table 14 ‡); 58 -> 18 labels is automatic
python eval.py --cfg configs/eval/mhp1_cross_w48.yaml DATASET.TEST val
python eval.py --cfg configs/eval/mhp1_cross_w48.yaml DATASET.TEST test

# MHP v1.0, MHP v1.0-trained model (in-domain, Table 14 §)
python eval.py --cfg configs/main/mhp1_w48.yaml --weights output/main_mhp1_w48/final_state.pth DATASET.TEST val
python eval.py --cfg configs/main/mhp1_w48.yaml --weights output/main_mhp1_w48/final_state.pth DATASET.TEST test

# 3DPW test: MPJPE, PA-MPJPE, PVE (Table 16)
python eval.py --cfg configs/main/mesh_w48.yaml --weights output/main_mesh_w48/final_state.pth

# CMU-Panoptic: MPJPE per sequence and average (Table 17)
python eval.py --cfg configs/eval/panoptic_w48.yaml
```

Replace `w48` with `w32` for the HRNet-W32 rows. The files in `configs/eval/` already point to the
right checkpoint through `TEST.MODEL_FILE`, so they need no `--weights`. All of the above at once:

```bash
scripts/eval_main.sh
```

### 6.3 Outputs

`eval.py` writes to `output/<CFG_NAME>/eval_<dataset>_<split>/` (or the folder given with `--out`):

| Benchmark | Files |
|---|---|
| all | `metrics.json`: the metrics below, also printed to the console |
| COCO, CrowdPose | `keypoints.json`, `segm.json`: results in the COCO result format |
| MHP (with `--dump-parts`) | `parts/<id>_<k>.png` label maps and `<id>_scores.txt`, for the official MHP toolkit |

To evaluate a repeated run, pass its name so its metrics go to its own folder:

```bash
python eval.py --cfg configs/main/coco_w48.yaml --weights output/t4_row7_coco_s1/final_state.pth CFG_NAME t4_row7_coco_s1
```

**COCO test-dev server.** test-dev has no public ground truth, so `metrics.json` stays empty. Rename
the files following the server convention and upload them as a zip:

```bash
cd output/eval_coco_testdev_w48/eval_coco_test-dev2017
cp keypoints.json person_keypoints_test-dev2017_holohuman_results.json
cp segm.json      detections_test-dev2017_holohuman_results.json
```

### 6.4 Metrics

| Benchmark | Metrics in `metrics.json` |
|---|---|
| COCO keypoints | `keypoints`: AP, AP50, AP75, APm, APl, AR, ... (pycocotools, OKS) |
| COCO segmentation | `segm`: mask AP, AP50, AP75, APs, APm, APl, ... (person class) |
| CrowdPose | `keypoints` (CrowdPose sigmas), and `keypoints_easy`, `keypoints_medium`, `keypoints_hard` (crowd index < 0.1, 0.1-0.8, > 0.8) |
| MHP v1.0 / v2.0 | `AP_p_50`, `AP_p_vol` (mean over IoU 0.1-0.9), `PCP_50`, and AP^p at every threshold |
| 3DPW | `MPJPE`, `PA-MPJPE`, `PVE` in mm (14 LSP joints, root-aligned; PVE over the 6,890 vertices), `matched`, `gt_persons`, `miss_rate` |
| CMU-Panoptic | `MPJPE` and `MPJPE_per_sequence` in mm |

`--max-images N` evaluates only the first N images, which is useful for a quick check of a new setup.

---

## 7. Ablation studies

Every ablation row is a separate checkpoint, trained with HRNet-W48 and the full 140-epoch schedule,
one per benchmark (COCO, MHP v2.0, mesh). The full configuration of every table is the main W48
checkpoint (`configs/main/<benchmark>_w48.yaml`), so it is not trained again.

| Table | Config files | What changes |
|---|---|---|
| Table 4, rows 0-6 | `configs/ablation/table4/row<k>_{coco,mhp2,mesh}.yaml` | `MODEL.IIA.ENABLED`, `MODEL.GFD.SPATIAL_ATTENTION`, `MODEL.GFD.CHANNEL_ATTENTION`, `MODEL.GFD.CENTER_OFFSET`, `MODEL.GFD.WINDOW` |
| Tables 2-3 | rows 1, 2, 3, 4, 7 of Table 4, and `configs/ablation/table2_3/iia_as_*.yaml` (IIA + A_s) | |
| Table 5, loss terms | `configs/ablation/table5/loss_{fl_int,fl_ctr,ctr_int,fl}_*.yaml` | `LOSS.CENTER_FOCAL`, `LOSS.CONTRASTIVE`, `LOSS.INTERACTION` |
| Table 5, λ_I | `configs/ablation/table5/lambda_{0,0.01,0.5,1}_*.yaml` | `LOSS.INTERACTION_WEIGHT` |
| Table 5, τ | `configs/ablation/table5/tau_{0.01,0.1,0.5,1}_*.yaml` | `LOSS.TEMPERATURE` |
| Table 6 | `configs/ablation/table6/{pe,bias,both}_*.yaml`; "No IIA" is Table 4 row 3 | `MODEL.IIA.POSITIONAL` |
| Table 8 | `configs/ablation/table8/{resnet50,resnet101,vit,vit_large}_coco.yaml` | `MODEL.BACKBONE` |

Train and evaluate every row of a table (the full-configuration files are skipped automatically):

```bash
GPUS=8 scripts/train_ablations.sh table4        # or table2_3, table5, table6, table8, all
```

A single row by hand:

```bash
torchrun --nproc_per_node 8 train.py --cfg configs/ablation/table4/row6_coco.yaml
python eval.py --cfg configs/ablation/table4/row6_coco.yaml --weights output/ablation_table4_row6_coco/final_state.pth
```

The Table 4 rows map to the switches as follows (✓ = on):

| Row | IIA | A_s | A_c | E(r) | Full F |
|---|---|---|---|---|---|
| 0 | | | | ✓ | ✓ |
| 1 | | ✓ | | ✓ | ✓ |
| 2 | | | ✓ | ✓ | ✓ |
| 3 | | ✓ | ✓ | ✓ | ✓ |
| 4 | ✓ | | | ✓ | ✓ |
| 5 | ✓ | ✓ | ✓ | | ✓ |
| 6 | ✓ | ✓ | ✓ | ✓ | (128 x 128 window) |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ (full model) |

---

## 8. Computational cost and timing

### 8.1 GMACs and parameters (Table 7, Table 4 columns)

These counts are exact and need neither data nor a GPU:

```bash
python profile_cost.py --m 1 5                         # Table 7: HRNet-W32, 512 x 512, single view, all four heads
python profile_cost.py --backbone hrnet48 --m 1        # Table 4, row 7
python profile_cost.py --backbone hrnet48 --m 1 --all-heads --cfg configs/ablation/table4/row6_coco.yaml   # any row
```

Convolutions, linear layers and matrix products are counted with `torch.utils.flop_counter`
(1 MAC = 2 FLOPs). BatchNorm, activations, pooling, softmax, NMS, sampling and element-wise products
are not counted. With `--cfg`, use `--all-heads` for the Table 4 columns, which count all four heads;
without it, only the heads of that config are attached. The flip test roughly doubles the counts.

Expected output: 92.34 GMACs at m = 1, 278.61 GMACs at m = 5, and 34.40 M parameters (HRNet-W32);
196.05 GMACs and 74.47 M parameters (HRNet-W48 full model).

### 8.2 Latency versus the number of persons (Figure 4)

```bash
python time_fig4.py --weights output/main_coco_w32/final_state.pth --out efficiency_vs_m.json
python tools/make_efficiency_fig.py --json efficiency_vs_m.json --out efficiency_vs_m.pdf
```

HRNet-W32 with all four heads, CrowdPose test images resized to 512 x 512, m = 1, 5, 10, 20, 30
person centers from the annotations (padded with extra centers when an image has fewer), batch 1,
flip test, fp32. Latency is the median of 100 CUDA-synchronized runs after warm-up; memory is the
peak allocation. Heads missing from the checkpoint are randomly initialized, which doesn't change
their cost.

### 8.3 Latency per task configuration (Figure 5, Table 9)

```bash
python time_fig5.py --weights output/main_coco_w32/final_state.pth --out runtime_per_task.json
```

COCO val images at 512 x 512, persons detected by the center head, each configuration (pose, seg,
part, mesh, pose+seg, all four) attached to the same trunk. All timing numbers depend on the GPU;
the paper uses a single RTX 3090.

Everything in this section at once:

```bash
scripts/reproduce_cost_and_timing.sh
```

---

## 9. Demo and inference API

The demo runs all four heads on one trunk. Build a four-head checkpoint from the per-benchmark
checkpoints (trunk and pose/seg heads from C-pose/seg, the other heads from their own checkpoints):

```bash
python tools/merge_checkpoints.py --trunk output/main_coco_w32/final_state.pth \
    --heads part=output/main_mhp2_w32/final_state.pth mesh=output/main_mesh_w32/final_state.pth \
    --out model_files/holohuman_w32.pth
```

This combination is for demonstration and timing only; the paper evaluates every benchmark with its
own checkpoint.

```python
import cv2
from humvis import HumVis

model = HumVis(backbone_name='hrnet32', weights='model_files/holohuman_w32.pth', flip_test=True)
frame = cv2.imread('image.jpg')                                   # BGR
pose_img, mask_img, part_img, mesh_img = model.get_analysis_results(frame, tracking=False)
```

For raw outputs, call the network directly:

```python
from humvis import get_cfg, load_weights
from humvis.holohuman import HoloHuman

cfg = get_cfg(); cfg.defrost(); cfg.MODEL.BACKBONE = 'hrnet32'; cfg.TEST.FLIP_TEST = True; cfg.freeze()
net = HoloHuman(cfg, 'cuda').eval()
load_weights(net, 'model_files/holohuman_w32.pth')
out = net(image_tensor, input_scale=1 / scale)    # see humvis/evaluation/inference.py for the preprocessing
# out['poses'] (m, K, 3), out['masks'], out['parts'], out['theta'] (m, 85), out['mesh'] (m, 6890, 3), ...
```

`app.py` (Flask web demo on port 65432), `camera.py` (webcam) and `video.py` (video files; needs
`ffmpeg` for the final re-encoding) wrap the same `HumVis` API.

---

## 10. Reference values from the paper

These are the values reported in the paper for the full model (single scale, flip test). Use them
to check a trained checkpoint.

| Benchmark (checkpoint) | Metric | HRNet-W32 | HRNet-W48 |
|---|---|---|---|
| COCO val (C-pose/seg) | keypoint AP | 75.6 | 79.3 |
| COCO val (C-pose/seg) | mask AP | 58.0 | 61.0 |
| COCO test-dev (C-pose/seg) | keypoint AP | 73.8 | 77.5 |
| CrowdPose test (C-pose/seg, cross-dataset) | AP | 74.0 | 77.3 |
| CrowdPose test (C-crowd, in-domain) | AP | 76.2 | 79.2 |
| MHP v2.0 val (C-part) | AP^p_50 / AP^p_vol / PCP_50 | 53.2 / 51.9 / 54.1 | 55.5 / 52.6 / 55.7 |
| 3DPW test (C-mesh) | MPJPE / PA-MPJPE / PVE (mm) | 61.6 / 38.4 / 72.7 | 58.5 / 35.6 / 70.3 |
| CMU-Panoptic (C-mesh) | average MPJPE (mm) | 80.0 | 77.6 |

Cost (Table 7): 92.34 GMACs and 34.40 M parameters for HRNet-W32 with all four heads at m = 1.

---

## 11. Implementation choices the paper leaves open

The paper fixes the architecture, the losses and their weights, and the schedule. The details below
are not stated in the paper; they are set here as follows, and each one is configurable.

1. **Ground-truth center r\*** (`datasets/base.py`, `person_center`): the center of the box of the
   visible keypoints; for part or mesh annotations without keypoints, the center of the annotated box.
2. **Gaussian targets** (`DATASET.SIGMA = 2`, `CENTER_SIGMA = 4`, `SIGMA_REF = 32`, in output cells):
   the base σ is scaled by √area / 32 of the person, clipped to [0.5, 2].
3. **Focal-head initialization** (`MODEL.BIAS_PROB = 0.01`): the output bias of the center and
   keypoint heads starts at logit(0.01).
4. **Crowd regions** of COCO add no negatives to the center and keypoint focal losses, and crowd
   persons are not instantiated.
5. **Contrastive positive**: the mirrored center is (W − 1 − x, y) in output cells. The flipped pass
   runs only on images with at least two annotated persons; on the others C_i = 0.
6. **Mesh supervision**: the reprojection term uses the 14 LSP joints regressed from the mesh, in
   normalized coordinates of the 512 x 512 input. The root for the 3D metrics is the midpoint of the
   hips. 3DPW ground-truth meshes use the gendered SMPL models when present.
7. **CrowdPose cross-dataset keypoints** (`datasets/joints.py`): neck = midpoint of the shoulders;
   head top = nose + 0.8 · (nose − neck).
8. **MHP v2.0 → v1.0 labels** (`MHP_V2_TO_V1`): limbs, garments, headwear, eyewear, bags and scarves
   are merged as listed in the file; labels without a v1.0 counterpart become background.
   `TEST.PART_MAPPING` accepts a `.npy` look-up table to use another mapping.
9. **ResNet / ViT feature map** (`backbones_extra.py`): ResNet C2-C5 reduced to 128 channels each,
   upsampled to stride 4 and concatenated (C = 512); ViT patch tokens upsampled by two stride-2
   deconvolutions (C = 256).
10. **Mesh mixture**: the three datasets are concatenated with equal weight (`DATASET.MESH_TRAIN_WEIGHTS`).
11. **Part flipping**: left and right part labels are swapped under horizontal flip, in training and in
    the flip test.
12. **OKS-NMS** with threshold 0.9 (`TEST.OKS_NMS`) is applied to the poses before evaluation.

---

## 12. Code ↔ paper map

| Paper | Code |
|---|---|
| Eq. 1, shared map F | `humvis/backbone.py` (`HRNet`, `build_backbone`) |
| Eq. 2, center head | `humvis/iia.py` (`CenterHead`) |
| Eq. 3, Peak | `humvis/iia.py` (`peak`) |
| Eq. 4, descriptor pooling | `humvis/iia.py` (`pool_instance_descriptors`) |
| Eqs. 5-8, self-attention | `humvis/iia.py` (`InstanceInteraction`) |
| Eq. 9, relative-position bias; absolute PE | `humvis/iia.py` (`psi`, `pos_mlp`; Table 6 only) |
| Eq. 10, decoder W_g with E(r) | `humvis/gfd.py` (`GlobalFeatureDecoder`, `center_offset_encoding`) |
| Eqs. 11-14, A_c and A_s | `humvis/gfd.py` (`ChannelAttention`, `SpatialAttention`) |
| Eq. 15, fusion W_f | `humvis/gfd.py` (`GFD.W_f`) |
| Appendix A.2, task heads | `humvis/heads.py` (`ConvHead`, `MeshHead`, `TaskHeads`) |
| Eqs. 16-17, total and IIA loss | `humvis/training.py` (`HoloHumanTrainer.forward`) |
| Eq. 18, focal loss | `humvis/losses.py` (`focal_loss`) |
| Eq. 19 and Algorithm 3, contrastive loss | `humvis/losses.py` (`contrastive_loss`), `humvis/training.py` (flipped pass) |
| Eqs. 20-25, task losses | `humvis/losses.py` (`dice_loss`, `part_loss`, `mesh_loss`, `render_gaussians`) |
| Eq. 26, loss weights | `humvis/config.py` (`LOSS.*_WEIGHT`) |
| Eq. 27 and Table 7, cost | `profile_cost.py` |
| Section 4, training protocol | `train.py`, `humvis/datasets/transforms.py` |
| Section 4, evaluation metrics | `humvis/evaluation/` |

---

## 13. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError` for the SMPL files when training the mesh head | Put the SMPL model and `J_regressor_h36m.npy` in `model_files/` (Section 2). |
| A run with a new `SEED` finishes immediately | It resumed the finished seed-0 run in the same folder. Pass a new `CFG_NAME` (Section 5.5). |
| `RuntimeError: ... does not match the HoloHuman architecture` in `eval.py` | The checkpoint was trained with other heads or switches than the config. Use the config of the run (`output/<CFG_NAME>/config.yaml`). |
| CUDA out of memory in training | Lower `TRAIN.INSTANCE_CHUNK` (persons decoded per pass), use more GPUs, or as a last resort lower `TRAIN.BATCH_SIZE`. |
| CUDA out of memory in evaluation | Lower `TEST.INSTANCE_CHUNK`; it changes no output. |
| `metrics.json` is empty for COCO | The split has no ground truth (test-dev); upload the result files to the server (Section 6.3). |
| `PVE` is `NaN` and a warning about gendered models | Add the male and female SMPL models for the 3DPW protocol. |
| Panoptic MPJPE far from the paper | Check the HD views and the frame step passed to `prepare_mesh_data.py panoptic`. |
| Seg AP missing on CrowdPose | CrowdPose has no masks; the masks are written but not evaluated. |
| `ImportError: timm` | Needed only for the ViT rows of Table 8: `pip install timm`. |
