"""Default configuration of HoloHuman.

Values follow the paper: Methodology (Sections 3.1-3.4), Section 4 (training and inference
protocol) and Appendix A.2 (layer-level specification, Figure 11 / Table 7).  Every entry
below is the full-model / paper setting; the ablation rows of Tables 2-6 and 8 are expressed
as overrides of these entries (see configs/generate_configs.py).
"""
from yacs.config import CfgNode as CN

_C = CN()

_C.CFG_NAME = ''
_C.OUTPUT_DIR = 'output'
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.DIST_BACKEND = 'nccl'
_C.AUTO_RESUME = True
_C.PIN_MEMORY = True
_C.VERBOSE = True
_C.DDP = False
_C.SEED = 0                            # seed of the run (-1: not seeded)

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'HoloHuman'
_C.MODEL.DEVICE = 'cpu'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''               # ImageNet-pretrained backbone weights (Section 4)
_C.MODEL.SYNC_BN = False
# 'hrnet32' -> C = 480 (4 branches 32+64+128+256 at stride 4)
# 'hrnet48' -> C = 720 (48+96+192+384)
# 'resnet50' / 'resnet101' / 'vit' / 'vit_large' -> Table 8 (see backbone.py)
_C.MODEL.BACKBONE = 'hrnet32'
_C.MODEL.BIAS_PROB = 0.01

# IIA (Section 3.1): center head H_center, Peak, P_inst, single-head self-attention
_C.MODEL.IIA = CN()
_C.MODEL.IIA.ENABLED = True            # False: F_e^(i) <- f^(i) (Algorithm 1, line 18; Table 4 rows 0-3)
_C.MODEL.IIA.CENTER_CHANNELS = 64      # H_center: 3x3 C->64 (BN, ReLU), 3x3 64->1, sigmoid
_C.MODEL.IIA.D = 256                   # W_Q, W_K, W_V: C->d ; W_O: d->C  (Eq. 6, Eq. 8)
_C.MODEL.IIA.POOL_KERNEL = 3           # P_inst: bilinear sampling + 3x3 average pooling (Eq. 4)
# positional information in IIA (Table 6): 'none' (Eq. 7, all reported results), 'pe'
# (sinusoidal encoding of the normalized center -> MLP -> added to f^(i) before Eq. 6),
# 'bias' (relative-position bias B of Eq. 9, psi: 2 -> 64 -> 1), 'both'
_C.MODEL.IIA.POSITIONAL = 'none'
_C.MODEL.IIA.PE_DIM = 64               # sinusoidal features per coordinate of the absolute encoding
_C.MODEL.IIA.BIAS_HIDDEN = 64          # hidden width of psi in Eq. 9

# GFD (Section 3.2): decoder W_g, channel/spatial attention, fusion W_f
_C.MODEL.GFD = CN()
_C.MODEL.GFD.CHANNELS = -1             # C' ; -1 means C' = C/2 (240 for W32, 360 for W48)
_C.MODEL.GFD.REDUCTION = 16            # r of the SE bottleneck of A_c (Eq. 11)
_C.MODEL.GFD.SPATIAL_KERNEL = 7        # k of W_s in A_s (Eq. 13)
_C.MODEL.GFD.CHANNEL_ATTENTION = True  # A_c (Table 4 column A_c); False drops F_c from [F_c, F_s]
_C.MODEL.GFD.SPATIAL_ATTENTION = True  # A_s (Table 4 column A_s); False drops F_s from [F_c, F_s]
_C.MODEL.GFD.CENTER_OFFSET = True      # E(r^(i)) in Eq. 10 (Table 4 column E(r)); False: W_g on 2C channels
_C.MODEL.GFD.WINDOW = 0                # 0: G^(i) decoded from the whole map F ("Full F" in Table 4);
                                       # 128: from a 128x128 window of F centred at r^(i), zero-padded (row 6)

# Task heads (Appendix A.2): 3x3 C'->C' ReLU, 1x1 to K / 1 / n+1 ; mesh: GAP, FC 1024 x2, IEF x3
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.TASKS = ['pose', 'seg', 'part', 'mesh']   # heads attached to the trunk (Table 1)
_C.MODEL.HEADS.FC_DIM = 1024
_C.MODEL.HEADS.IEF_ITERS = 3
_C.MODEL.HEADS.INIT_THETA = ''         # optional .npy with the initial Theta (85) of the IEF loop
_C.MODEL.HEADS.FOCAL = 5000.           # focal length used only to render the weak-perspective camera

# SMPL files
_C.MODEL.SMPL_MODEL = './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
_C.MODEL.SMPL_H36M_REGRESSOR = './model_files/J_regressor_h36m.npy'
# gendered models, used only to build the 3DPW ground-truth meshes for MPJPE / PA-MPJPE / PVE
_C.MODEL.SMPL_MALE = './model_files/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
_C.MODEL.SMPL_FEMALE = './model_files/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

# ---------------------------------------------------------------------------
# Loss (Sections 3.3-3.4)
# ---------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.IIA_WEIGHT = 1.0               # lambda_IIA (Eq. 16, Eq. 26)
_C.LOSS.POSE_WEIGHT = 1.0              # lambda_pose (Eq. 26)
_C.LOSS.SEG_WEIGHT = 1.0               # lambda_seg
_C.LOSS.PART_WEIGHT = 1.0              # lambda_part
_C.LOSS.MESH_WEIGHT = 0.1              # lambda_mesh
_C.LOSS.FOCAL_ALPHA = 2.0              # alpha of Eq. 18
_C.LOSS.FOCAL_BETA = 4.0               # beta of Eq. 18
_C.LOSS.CENTER_FOCAL = True            # FL(P, P*) of Eq. 17 (Table 5 column FL)
_C.LOSS.CONTRASTIVE = True             # (1/m) sum_i C_i of Eq. 17 (Table 5 column Ctr)
_C.LOSS.INTERACTION = True             # interaction regularization of Eq. 17 (Table 5 column Int)
_C.LOSS.INTERACTION_WEIGHT = 0.1       # lambda_I (Eq. 17; Table 5 sweep)
_C.LOSS.TEMPERATURE = 0.05             # tau (Eq. 19; Table 5 sweep)
_C.LOSS.MESH_POSE_WEIGHT = 1.0         # the three terms of Eq. 25 are unweighted in the paper
_C.LOSS.MESH_SHAPE_WEIGHT = 1.0
_C.LOSS.MESH_REPROJ_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = 'data'
# 'coco' (pose + seg), 'crowdpose' (pose, K = 14), 'mhp2' / 'mhp1' (part), 'mesh' (SMPL mixture)
_C.DATASET.DATASET = 'coco'
_C.DATASET.NUM_KEYPOINTS = 17          # K = 17 on COCO, 14 on CrowdPose
_C.DATASET.NUM_PARTS = 58              # n part categories (n + 1 output channels); n = 58 for MHP v2.0
_C.DATASET.MAX_INSTANCES = 30
_C.DATASET.TRAIN = 'train'
_C.DATASET.TEST = 'val'
_C.DATASET.FILTER_IMAGE = False
_C.DATASET.SIGMA = 2.0                 # base sigma (cells) of the Gaussian targets; scaled by person size
_C.DATASET.CENTER_SIGMA = 4.0          # base sigma_c of the center target P* (cells)
_C.DATASET.SIGMA_REF = 32.0            # person size (sqrt(area) in cells) at which the base sigmas apply;
                                       # sigma scales with sqrt(area)/SIGMA_REF, clipped to [0.5, 2]
_C.DATASET.FLIP = 0.5
# keypoint permutation under horizontal flip (COCO order)
_C.DATASET.FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# part-label permutation under horizontal flip (length n + 1); empty = identity
_C.DATASET.PART_FLIP_INDEX = []
# mesh mixture (Section 4, 3DPW): list of (name, annotation file) pairs and sampling weights
_C.DATASET.MESH_TRAIN = ['muco', 'coco_smpl', '3dpw_train']
_C.DATASET.MESH_TRAIN_WEIGHTS = [1.0, 1.0, 1.0]
_C.DATASET.MESH_TEST = '3dpw_test'     # '3dpw_test' or 'panoptic'

# training data augmentation (Section 4)
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.5
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = 128           # stride 4

# ---------------------------------------------------------------------------
# Training (Section 4, "Implementation Details")
# ---------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 1e-3                     # non-backbone parameters
_C.TRAIN.BACKBONE_LR = 1e-4            # backbone parameters
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BATCH_SIZE = 64               # total batch size (split across GPUs under DDP)
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.LR_STEPS = [90, 120]
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.SHUFFLE = True
_C.TRAIN.RESUME = ''                   # resume (model + optimizer + epoch) from this checkpoint
_C.TRAIN.INIT_CHECKPOINT = ''          # initialize the model weights only (e.g. C-mhp1 from C-part)
_C.TRAIN.CLIP_GRAD = 0.0               # 0: no gradient clipping
_C.TRAIN.AMP = False
_C.TRAIN.INSTANCE_CHUNK = 16           # instances decoded by GFD + heads per pass (memory only)
_C.TRAIN.SAVE_EVERY = 10               # epochs between intermediate checkpoints (final epoch always saved)

# ---------------------------------------------------------------------------
# Testing (Section 4, Appendix A.2)
# ---------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.FLIP_TEST = False              # paper accuracy results: single scale with flip test (True)
_C.TEST.IMAGES_PER_GPU = 1
_C.TEST.MODEL_FILE = 'model_files/holohuman_w32.pth'
_C.TEST.OKS_SCORE = 0.7
_C.TEST.OKS_NMS = 0.9                  # OKS-NMS threshold applied to the pose results before evaluation
_C.TEST.OKS_SIGMAS = [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]

_C.TEST.NMS_KERNEL = 3                 # Peak: 3x3 max-pooling NMS
_C.TEST.CENTER_THRESHOLD = 0.1         # Peak: confidence threshold
_C.TEST.MAX_INSTANCES = 30             # Peak: at most 30 instances per image
_C.TEST.INSTANCE_CHUNK = 8             # instances decoded by GFD + heads per pass (memory only)
_C.TEST.MASK_THRESHOLD = 0.5
_C.TEST.MATCH_IOU = 0.5                # 2D box IoU for matching mesh predictions to GT (3DPW / Panoptic)
_C.TEST.PART_MAPPING = ''              # many-to-one label mapping file (MHP v2.0 -> v1.0, Table 14 cross-dataset)


def update_config(cfg, args_cfg=None, opts=None):
    """Merge a YAML file (and optional `KEY VALUE ...` overrides) into cfg (in place)."""
    cfg.defrost()
    if args_cfg:
        cfg.merge_from_file(args_cfg)
    if opts:
        cfg.merge_from_list(list(opts))
    cfg.freeze()
    return cfg
