# HoloHuman — code, configurations and scripts

This package contains the HoloHuman architecture of the paper (Section 3, Appendix A.2,
Figure 11), the training and evaluation code of the Section 4 protocol, the configuration
file of every checkpoint of Table 1 and of every ablation row of Tables 2–6 and 8, the
evaluation scripts with the metric implementations, and the profiling and timing scripts
behind Table 7, Figure 4 and Figure 5. `REPRODUCE.md` lists the command behind every
reported number.

The demo entry point is unchanged:

```python
from humvis import HumVis
model = HumVis(backbone_name='hrnet32')            # or 'hrnet48'
pose_img, mask_img, part_img, mesh_img = model.get_analysis_results(frame, tracking=True)
```

`app.py`, `camera.py`, `video.py` and `test_app.py` run without modification.

## Layout

```
humvis/                 model, losses, training forward, datasets, evaluation
  backbone.py           HRNet-W32 / W48 (C = 480 / 720 at stride 4), ImageNet initialization
  backbones_extra.py    ResNet-50 / 101 and ViT-B / ViT-L of Table 8
  iia.py                IIA: center head, Peak, P_inst, self-attention (Eqs. 2–8), Eq. 9 bias / PE (Table 6)
  gfd.py                GFD: W_g, A_c, A_s, W_f (Eqs. 10–15), ablation switches (Table 4)
  heads.py              pose / seg / part / mesh heads (A.2), SMPL forward
  holohuman.py          inference forward (Section 4: flip test, threshold 0.1, ≤ 30 instances)
  losses.py             Eqs. 16–26 (focal, contrastive, interaction, dice, cross-entropy, mesh)
  training.py           training forward: ground-truth centers, flipped-view positives (Algorithm 3)
  datasets/             COCO, CrowdPose, MHP v1.0 / v2.0, mesh mixture; augmentation of Section 4
  evaluation/           pycocotools (COCO, CrowdPose), MHP AP^p / PCP, 3DPW / Panoptic metrics, timing
configs/                base.yaml, main/ (Table 1), eval/ (cross-dataset), ablation/table{4,2_3,5,6,8}/
train.py  eval.py       one checkpoint per benchmark; metrics of Section 4
profile_cost.py         Table 7 (GMACs and parameters per component)
time_fig4.py            Figure 4 (latency / memory versus the number of persons)
time_fig5.py            Figure 5 / Table 9 (latency per task configuration)
tools/                  prepare_mesh_data.py (3DPW, MuCo, COCO-SMPL, Panoptic), make_efficiency_fig.py,
                        merge_checkpoints.py (four-head demo checkpoint)
scripts/                train_main.sh, eval_main.sh, train_ablations.sh, reproduce_cost_and_timing.sh
```

## Code ↔ paper

| File | Paper | Content |
|---|---|---|
| `humvis/backbone.py` | Eq. 1, A.2 "Backbone" | HRNet-W32 / W48, four branches upsampled to stride 4 and concatenated: `F` with C = 480 / 720, selected by `cfg.MODEL.BACKBONE`; `load_pretrained_backbone` for the ImageNet initialization. |
| `humvis/iia.py` | §3.1, Alg. 1, Fig. 3 | `CenterHead` (3×3 C→64, BN, ReLU; 3×3 64→1; sigmoid, Eq. 2), `peak` (3×3 max-pool NMS, threshold 0.1, ≤ 30, Eq. 3), `pool_instance_descriptors` (bilinear sampling + 3×3 average, Eq. 4), `InstanceInteraction` (single head, d = 256, Eqs. 5–8, content only; `psi` of Eq. 9 and the absolute PE only for Table 6). `MODEL.IIA.ENABLED = False` gives F_e = f (Alg. 1, line 18). |
| `humvis/gfd.py` | §3.2, Alg. 2 | `center_offset_encoding` E(r), `GlobalFeatureDecoder` W_g (1×1, 2C+2 → C′, C′ = C/2, Eq. 10), `ChannelAttention` (GAP+GMP, SE r = 16, Eq. 11), `SpatialAttention` (7×7, Eq. 13), fusion W_f (3×3, 2C′ → C′, ReLU, Eq. 15); switches `CHANNEL_ATTENTION`, `SPATIAL_ATTENTION`, `CENTER_OFFSET`, `WINDOW` (128×128 zero-padded window, Table 4 row 6). |
| `humvis/heads.py` | A.2 "Task heads" | pose / seg / part heads (3×3 C′, ReLU; 1×1 to K / 1 / n+1; sigmoid / sigmoid / softmax), mesh head (GAP, FC 1024 ×2, IEF ×3 → Θ ∈ R⁸⁵ = θ_pose 72 + θ_shape 10 + θ_cam 3) → SMPL; `MODEL.HEADS.TASKS` attaches the heads of a checkpoint (Table 1). |
| `humvis/losses.py`, `humvis/training.py` | §3.3–3.4, Alg. 3 | L_total = λ_IIA L_IIA + L_GFD (Eq. 16); focal loss α = 2, β = 4 (Eq. 18); contrastive loss with the flipped-view positive pooled at the mirrored ground-truth center and the other m−1 persons as negatives, τ = 0.05 (Eq. 19); interaction regularization weighted by I_ij, λ_I = 0.1 (Eq. 17); focal / dice / cross-entropy / SMPL-parameter + reprojection task losses (Eqs. 21–25); weights 1, 1, 1, 1, 0.1 (Eq. 26); instances instantiated at the ground-truth centers. |
| `humvis/datasets/` | §4 | 512×512 crops, rotation ±30°, scale [0.75, 1.5], flip 0.5; COCO person keypoints + masks, CrowdPose (14 keypoints), MHP v2.0 (58 parts) / v1.0 (18 parts), mesh mixture (MuCo-3DHP, COCO pseudo-SMPL, 3DPW train); COCO→CrowdPose keypoint conversion and MHP 58→18 label mapping for the cross-dataset rows. |
| `humvis/evaluation/` | §4 "Evaluation Metrics" | COCO / CrowdPose OKS-AP and mask AP with pycocotools (CrowdPose sigmas, easy / medium / hard splits), MHP AP^p_50, AP^p_vol, PCP_50, 3DPW MPJPE / PA-MPJPE / PVE (box-IoU 0.5 matching), Panoptic MPJPE per sequence; latency / memory measurement. |
| `humvis/holohuman.py` | Fig. 2, §4 | backbone → IIA → GFD → heads; single-scale test with flip test; keypoint = maximum of its channel. |
| `humvis/config.py` | §3–4, A.2 | all hyper-parameters above, with comments; `configs/*.yaml` only contain overrides. |
| `profile_cost.py` | Table 7 | per-component GMACs and parameters with `torch.utils.flop_counter`. |
| `time_fig4.py`, `time_fig5.py` | Fig. 4, Fig. 5, Table 9 | median of CUDA-synchronized runs after warm-up, peak memory, batch 1, 512×512, flip test. |

## Setup

```
pip install torch torchvision yacs pycocotools opencv-python scipy pyyaml matplotlib   # timm for the ViT rows
model_files/  basicModel_neutral_lbs_10_207_0_v1.0.0.pkl   SMPL neutral model (mesh head, training and demo)
              basicModel_{m,f}_lbs_10_207_0_v1.0.0.pkl      gendered models (3DPW ground-truth meshes only)
              J_regressor_h36m.npy
              hrnetv2_w32_imagenet_pretrained.pth, hrnetv2_w48_imagenet_pretrained.pth   (ImageNet init.)
data/         coco/, crowdpose/, mhp_v2/, mhp_v1/ in the layouts documented in humvis/datasets/*.py;
              mesh/*.pkl written by tools/prepare_mesh_data.py
```

## Training and evaluation

```
torchrun --nproc_per_node 8 train.py --cfg configs/main/coco_w48.yaml      # C-pose/seg, batch 64 over 8 GPUs
python eval.py --cfg configs/main/coco_w48.yaml --weights output/main_coco_w48/final_state.pth
scripts/train_main.sh ; scripts/eval_main.sh                                # all Table 1 checkpoints
scripts/train_ablations.sh table4                                           # one checkpoint per row and benchmark
scripts/reproduce_cost_and_timing.sh                                        # Table 7, Figures 4 and 5
```

`train.py` writes `output/<CFG_NAME>/final_state.pth` (the checkpoint after the 140th epoch,
which is the one evaluated), `checkpoint.pth` for resuming, `config.yaml` and `train.log`.
Any `KEY VALUE` pair after the config file overrides it (`python train.py --cfg ... TRAIN.BATCH_SIZE 32`).

## Verification (random weights, CPU)

`python profile_cost.py --m 1 5` reproduces Table 7 exactly:

```
HRNet backbone + fusion to F   41.23     Center head        4.54
IIA projections + attention     0.0005   GFD decoder W_g    3.78 (per person)
GFD A_c / A_s                   <0.001 / 0.002
GFD fusion W_f                 16.99     Pose/Seg/Part heads 8.56 / 8.50 / 8.73
Mesh head (IEF×3)               0.004    SMPL forward       0.008
TOTAL m=1: 92.34 GMACs   TOTAL m=5: 278.61 GMACs   params: 34.40 M
```

`python profile_cost.py --backbone hrnet48 --m 1` gives 196.05 GMACs and 74.47 M parameters
(Table 4, row 7), and the parameter counts of the ablation switches match every row of Table 4
(72.55 / 72.55 / 72.56 / 73.73 / 73.29 / 74.47 / 74.47 / 74.47 M for rows 0–7). The training
step, every ablation switch, the evaluators (checked with oracle predictions: COCO AP 1.0,
MHP AP^p 100, 3DPW error 0) and the timing scripts were exercised end to end on synthetic
data; the flip-test path, the forced-center path (`model(x, centers=...)`) and the tracking
path of the demo were exercised with random weights.

## Choices where the paper leaves a detail open

The paper fixes the architecture, losses, weights and schedule; the following implementation
details are not stated in the paper and are set here as follows (all configurable):

1. **Ground-truth center r\*** (`datasets/base.py`, `person_center`): the center of the box of
   the visible keypoints; for part / mesh annotations without keypoints, the center of the
   annotated box.
2. **Gaussian targets** (`DATASET.SIGMA = 2`, `CENTER_SIGMA = 4`, `SIGMA_REF = 32` cells): the
   base σ_k / σ_c are scaled by √area / 32 (area in output cells), clipped to [0.5, 2].
3. **Focal-head initialization** (`MODEL.BIAS_PROB = 0.01`): the output bias of the center and
   keypoint heads starts at logit(0.01), as in CID / CenterNet.
4. **Crowd regions** of COCO (`iscrowd = 1`) do not contribute negatives to the center and
   keypoint focal losses (`valid` mask); crowd persons are not instantiated.
5. **Contrastive positive**: the mirrored center is `(W − 1 − x, y)` in output cells; the flipped
   pass runs only on the images with at least two annotated persons (otherwise C_i = 0).
6. **Mesh supervision**: the reprojection term uses the 14 LSP joints regressed from the mesh
   with `J_regressor_h36m` in normalized image coordinates of the 512×512 input; the root for
   the 3D metrics is the midpoint of the hips. 3DPW ground-truth meshes use the gendered
   models when `MODEL.SMPL_MALE / SMPL_FEMALE` exist.
7. **CrowdPose cross-dataset keypoints** (`datasets/joints.py`): neck = midpoint of the
   shoulders; head top = nose + 0.8 · (nose − neck).
8. **MHP v2.0 → v1.0 label mapping** (`MHP_V2_TO_V1`): limbs, garments, head-wear, eyewear,
   bags and scarves merged as listed in the file; labels without a v1.0 counterpart (ball,
   bottle, umbrella, watch, …) become background. Set `TEST.PART_MAPPING` to a `.npy`
   look-up table to use a different mapping.
9. **ResNet / ViT fusion to stride 4** (`backbones_extra.py`): 1×1 laterals (128 channels) on
   C2–C5 upsampled and concatenated (C = 512); ViT-B/L patch tokens + two stride-2 deconvolutions
   (C = 256).
10. **Mesh mixture sampling**: the three datasets are concatenated with equal weight
    (`DATASET.MESH_TRAIN_WEIGHTS`); the MuCo / COCO pseudo-SMPL annotation files are read in
    the I2L-MeshNet / EFT formats by `tools/prepare_mesh_data.py`.
11. **Part flipping**: left / right part labels are swapped under horizontal flip
    (`MHP_V2_FLIP_INDEX`, `MHP_V1_FLIP_INDEX`), both in training and in the flip test.
12. **OKS-NMS** (`TEST.OKS_NMS = 0.9`) is applied to the pose results before evaluation, as in
    the demo.
