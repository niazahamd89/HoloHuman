# Reproducing the tables and figures

Every number of the paper is produced by one configuration file under `configs/` (its header
comment names the table and row) and one of `train.py`, `eval.py`, `profile_cost.py`,
`time_fig4.py`, `time_fig5.py`. All training runs use the protocol of Section 4 (Adam, lr
1e-3 / 1e-4 for the backbone, batch 64, 140 epochs with ×0.1 at epochs 90 and 120, 512×512
crops, ImageNet-initialized HRNet, losses of Sections 3.3–3.4 restricted to the heads of the
checkpoint); all evaluations use single-scale inference with flip test, center threshold 0.1
and at most 30 instances. `GPUS` is the number of GPUs sharing the batch of 64.

## Data

```
data/coco/images/{train2017,val2017,test2017}, data/coco/annotations/person_keypoints_*.json, image_info_test-dev2017.json
data/crowdpose/images, data/crowdpose/json/crowdpose_{trainval,test}.json
data/mhp_v2/{train,val}/{images,parsing_annos}
data/mhp_v1/{images,annotations}, data/mhp_v1/{train,val,test}_list.txt
data/mesh/{muco,coco_smpl,3dpw_train,3dpw_test,panoptic}.pkl      <- tools/prepare_mesh_data.py
```

```
python tools/prepare_mesh_data.py 3dpw --root data/3dpw --split train --out data/mesh/3dpw_train.pkl
python tools/prepare_mesh_data.py 3dpw --root data/3dpw --split test  --out data/mesh/3dpw_test.pkl
python tools/prepare_mesh_data.py muco --root data/muco --out data/mesh/muco.pkl
python tools/prepare_mesh_data.py coco_smpl --root data/coco --fits <pseudo-SMPL fits> --format i2l|eft --out data/mesh/coco_smpl.pkl
python tools/prepare_mesh_data.py panoptic --root data/panoptic --cameras <HD views> --step <frame step> --out data/mesh/panoptic.pkl
```

## Table 1 checkpoints (main results)

| Checkpoint | Train | Evaluate | Reported in |
|---|---|---|---|
| C-pose/seg W32 / W48 | `train.py --cfg configs/main/coco_w{32,48}.yaml` | `eval.py --cfg configs/main/coco_w{32,48}.yaml --weights output/main_coco_w{32,48}/final_state.pth` (COCO val: keypoint AP, mask AP) | Tables 22, 12; full rows of Tables 2, 4–6 (W48) |
| | | `eval.py --cfg configs/eval/coco_testdev_w{32,48}.yaml` → `keypoints.json`, `segm.json` for the COCO server | Tables 10, 13 |
| | | `eval.py --cfg configs/eval/crowdpose_cross_w{32,48}.yaml` (17 → 14 keypoints, CrowdPose test) | Table 11 ‡ |
| C-crowd W32 / W48 | `train.py --cfg configs/main/crowdpose_w{32,48}.yaml` | `eval.py --cfg configs/main/crowdpose_w{32,48}.yaml --weights output/main_crowdpose_w{32,48}/final_state.pth` | Table 11 § |
| C-part W32 / W48 | `train.py --cfg configs/main/mhp2_w{32,48}.yaml` | `eval.py --cfg configs/main/mhp2_w{32,48}.yaml --weights output/main_mhp2_w{32,48}/final_state.pth` (MHP v2.0 val) | Table 15; full rows of Tables 3–6 (W48) |
| | | `eval.py --cfg configs/eval/mhp1_cross_w{32,48}.yaml DATASET.TEST val` and `... DATASET.TEST test` (58 → 18 labels) | Table 14 ‡ |
| C-mhp1 W32 / W48 | `train.py --cfg configs/main/mhp1_w{32,48}.yaml` (initialized from C-part) | `eval.py --cfg configs/main/mhp1_w{32,48}.yaml --weights output/main_mhp1_w{32,48}/final_state.pth DATASET.TEST {val,test}` | Table 14 § |
| C-mesh W32 / W48 | `train.py --cfg configs/main/mesh_w{32,48}.yaml` | `eval.py --cfg configs/main/mesh_w{32,48}.yaml --weights output/main_mesh_w{32,48}/final_state.pth` (3DPW test) | Table 16; full rows of Tables 3–6 (W48) |
| | | `eval.py --cfg configs/eval/panoptic_w{32,48}.yaml` | Table 17 |

`scripts/train_main.sh` and `scripts/eval_main.sh` run all of the above.

## Ablations (HRNet-W48, one checkpoint per row and benchmark)

`scripts/train_ablations.sh <table>` trains and evaluates every file of `configs/ablation/<table>/`;
the full configuration (Table 4 row 7, Table 5 highlighted cells, Table 6 "content only",
Table 8 HR-W48) is `configs/main/<benchmark>_w48.yaml`.

| Table | Configs | Switches |
|---|---|---|
| Table 4 rows 0–6 | `ablation/table4/row<k>_{coco,mhp2,mesh}.yaml` | `MODEL.IIA.ENABLED`, `MODEL.GFD.SPATIAL_ATTENTION`, `MODEL.GFD.CHANNEL_ATTENTION`, `MODEL.GFD.CENTER_OFFSET`, `MODEL.GFD.WINDOW = 128` |
| Tables 2–3 | rows 1, 2, 3, 4, 7 of Table 4 and `ablation/table2_3/iia_as_{coco,mhp2,mesh}.yaml` (I_ij + A_s) | |
| Table 5, loss terms | `ablation/table5/loss_{fl_int,fl_ctr,ctr_int,fl}_<benchmark>.yaml` | `LOSS.CENTER_FOCAL`, `LOSS.CONTRASTIVE`, `LOSS.INTERACTION` |
| Table 5, λ_I sweep | `ablation/table5/lambda_{0,0.01,0.5,1}_<benchmark>.yaml` | `LOSS.INTERACTION_WEIGHT` |
| Table 5, τ sweep | `ablation/table5/tau_{0.01,0.1,0.5,1}_<benchmark>.yaml` | `LOSS.TEMPERATURE` |
| Table 6 | `ablation/table6/{pe,bias,both}_<benchmark>.yaml`; "No IIA" = `table4/row3_*` | `MODEL.IIA.POSITIONAL` |
| Table 8 | `ablation/table8/{resnet50,resnet101,vit,vit_large}_coco.yaml` | `MODEL.BACKBONE` |

Params and GMACs of Table 4 for any row: `python profile_cost.py --backbone hrnet48 --m 1 --cfg <row config>`.

## Cost and timing

```
python profile_cost.py --m 1 5                      # Table 7 (HRNet-W32, 512x512, single view; 92.34 / 278.61 GMACs, 34.40 M)
python profile_cost.py --backbone hrnet48 --m 1     # Table 4 full row (196.05 GMACs, 74.47 M)
python time_fig4.py --weights output/main_coco_w32/final_state.pth     # Figure 4: m = 1, 5, 10, 20, 30 on CrowdPose test
python tools/make_efficiency_fig.py --json efficiency_vs_m.json --out efficiency_vs_m.pdf
python time_fig5.py --weights output/main_coco_w32/final_state.pth     # Figure 5 / Table 9: per-task latency on COCO val
```

Timing numbers depend on the GPU (the paper: RTX 3090, batch 1, fp32, flip test, median of 100
CUDA-synchronized runs after warm-up); the analytic counts of `profile_cost.py` do not.

## Seeds

`SEED` (default 0) seeds Python, NumPy and PyTorch; `python train.py --cfg ... SEED 1 CFG_NAME main_coco_w48_s1`
trains a second run of the same configuration into its own output directory. Data loading and
cuDNN remain non-deterministic unless `CUDNN.DETERMINISTIC True CUDNN.BENCHMARK False`.
