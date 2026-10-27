"""Keypoint and part-label conventions of the benchmarks (Section 4)."""
import numpy as np

# ---------------------------------------------------------------- COCO (K = 17)
COCO_KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
COCO_FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
COCO_SIGMAS = [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]

# ---------------------------------------------------------------- CrowdPose (K = 14)
CROWDPOSE_KEYPOINTS = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                       'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                       'right_ankle', 'head_top', 'neck']
CROWDPOSE_FLIP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13]
CROWDPOSE_SIGMAS = [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]
# the 12 CrowdPose body joints that exist in COCO (CrowdPose index -> COCO index)
COCO_TO_CROWDPOSE_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
HEAD_TOP_EXTRAPOLATION = 0.8   # head_top = nose + 0.8 * (nose - neck)  (cross-dataset evaluation, Table 11 ‡)


def coco_to_crowdpose(kpts):
    """17 COCO keypoints (N, 17, 3) -> 14 CrowdPose keypoints (N, 14, 3).  The 12 body joints
    are copied; the neck is the midpoint of the shoulders and the head top is extrapolated
    from the nose along the neck -> nose direction (Section 4, "CrowdPose (cross-dataset)")."""
    kpts = np.asarray(kpts, dtype=np.float32)
    out = np.zeros((kpts.shape[0], 14, 3), dtype=np.float32)
    out[:, :12] = kpts[:, COCO_TO_CROWDPOSE_BODY]
    ls, rs, nose = kpts[:, 5], kpts[:, 6], kpts[:, 0]
    neck_vis = np.minimum(ls[:, 2], rs[:, 2])
    neck = np.concatenate(((ls[:, :2] + rs[:, :2]) / 2.0, neck_vis[:, None]), axis=1)
    out[:, 13] = neck
    head_vis = np.minimum(neck_vis, nose[:, 2])
    head = nose[:, :2] + HEAD_TOP_EXTRAPOLATION * (nose[:, :2] - neck[:, :2])
    out[:, 12] = np.concatenate((head, head_vis[:, None]), axis=1)
    return out


# ---------------------------------------------------------------- 14 LSP joints (mesh supervision / 3D metrics)
LSP14 = ['right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee', 'left_ankle', 'right_wrist',
         'right_elbow', 'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist', 'neck', 'head_top']
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]   # 17 H36M joints -> LSP14
# COCO -> LSP14 (neck synthesized from the shoulders; head top unavailable)
COCO_TO_J14 = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9]
# OpenPose/COCO-18 order used by the 3DPW 'poses2d' annotations -> LSP14 (head top unavailable)
OPENPOSE18_TO_J14 = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1]
# MuCo-3DHP / MPI-INF-3DHP 17-joint order (head_top, neck, r_sho, r_elb, r_wri, l_sho, l_elb, l_wri,
# r_hip, r_knee, r_ank, l_hip, l_knee, l_ank, pelvis, spine, head) -> LSP14
MUCO17_TO_J14 = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]
# CMU Panoptic COCO19 order -> LSP14 (head top unavailable)
PANOPTIC19_TO_J14 = [14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5, 0]
J14_FLIP_INDEX = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]


def coco_to_j14(kpts):
    """(N, 17, 3) COCO keypoints -> (N, 14, 3) LSP14 (neck = mid-shoulders, head top invisible)."""
    kpts = np.asarray(kpts, dtype=np.float32)
    out = np.zeros((kpts.shape[0], 14, 3), dtype=np.float32)
    out[:, :12] = kpts[:, COCO_TO_J14]
    ls, rs = kpts[:, 5], kpts[:, 6]
    out[:, 12, :2] = (ls[:, :2] + rs[:, :2]) / 2.0
    out[:, 12, 2] = np.minimum(ls[:, 2], rs[:, 2])
    return out


def remap_joints(kpts, index, num_out=14):
    """(N, J, 3) -> (N, num_out, 3) by index; missing targets stay invisible."""
    kpts = np.asarray(kpts, dtype=np.float32)
    out = np.zeros((kpts.shape[0], num_out, 3), dtype=np.float32)
    out[:, :len(index)] = kpts[:, index]
    return out


# ---------------------------------------------------------------- SMPL (24 joints, axis-angle)
SMPL_POSE_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]


def flip_smpl_pose(pose):
    """Horizontal flip of SMPL axis-angle pose (72,): swap left/right joints, negate y and z."""
    pose = np.asarray(pose, dtype=np.float32).reshape(24, 3)[SMPL_POSE_FLIP_PERM].copy()
    pose[:, 1:] *= -1
    return pose.reshape(72)


def rotate_global_orient(pose, rot_deg):
    """Rotate the global orientation (first 3 axis-angle values) by an in-plane image
    rotation of rot_deg degrees (counter-clockwise image rotation = rotation about -z)."""
    import cv2
    pose = np.asarray(pose, dtype=np.float32).copy()
    r = np.deg2rad(-rot_deg)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    R, _ = cv2.Rodrigues(pose[:3])
    pose[:3] = cv2.Rodrigues(Rz @ R)[0].reshape(3)
    return pose


# ---------------------------------------------------------------- MHP part labels
MHP_V2_PARTS = ['Cap/hat', 'Helmet', 'Face', 'Hair', 'Left-arm', 'Right-arm', 'Left-hand', 'Right-hand',
                'Protector', 'Bikini/bra', 'Jacket/windbreaker/hoodie', 'Tee-shirt', 'Polo-shirt', 'Sweater',
                'Singlet', 'Torso-skin', 'Pants', 'Shorts/swim-shorts', 'Skirt', 'Stockings', 'Socks',
                'Left-boot', 'Right-boot', 'Left-shoe', 'Right-shoe', 'Left-highheel', 'Right-highheel',
                'Left-sandal', 'Right-sandal', 'Left-leg', 'Right-leg', 'Left-foot', 'Right-foot', 'Coat',
                'Dress', 'Robe', 'Jumpsuit', 'Other-full-body-clothes', 'Headwear', 'Backpack', 'Ball',
                'Bats', 'Belt', 'Bottle', 'Carrybag', 'Cases', 'Sunglasses', 'Eyewear', 'Glove', 'Scarf',
                'Umbrella', 'Wallet/purse', 'Watch', 'Wristband', 'Tie', 'Other-accessory',
                'Other-upper-body-clothes', 'Other-lower-body-clothes']        # labels 1..58
MHP_V1_PARTS = ['hat', 'hair', 'sunglasses', 'upper-clothes', 'skirt', 'pants', 'dress', 'belt',
                'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm',
                'bag', 'scarf', 'torso-skin']                                  # labels 1..18
assert len(MHP_V2_PARTS) == 58 and len(MHP_V1_PARTS) == 18


def _flip_perm(num, pairs):
    perm = list(range(num + 1))
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


MHP_V2_FLIP_INDEX = _flip_perm(58, [(5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33)])
MHP_V1_FLIP_INDEX = _flip_perm(18, [(9, 10), (12, 13), (14, 15)])

# Many-to-one mapping of the 58 MHP v2.0 labels to the 18 MHP v1.0 labels used for the
# cross-dataset evaluation of Table 14 (‡): limb sub-parts are merged into the v1.0 limb
# class, head accessories into hat, eyewear into sunglasses, bags into bag, torso garments
# into upper-clothes, leg garments into pants/skirt/dress.  Labels mapped to 0 have no v1.0
# counterpart and count as background.
MHP_V2_TO_V1 = {0: 0,
                1: 1, 2: 1, 39: 1,                       # cap/hat, helmet, headwear -> hat
                3: 11,                                    # face
                4: 2,                                     # hair
                5: 14, 7: 14, 49: 14,                     # left arm, left hand, glove -> left arm
                6: 15, 8: 15,                             # right arm, right hand -> right arm
                9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 34: 4, 57: 4,   # torso garments -> upper clothes
                16: 18,                                   # torso skin
                17: 6, 18: 6, 20: 6, 21: 6, 58: 6,        # pants, shorts, stockings, socks, other lower -> pants
                19: 5,                                    # skirt
                22: 9, 24: 9, 26: 9, 28: 9,               # left boot/shoe/highheel/sandal -> left shoe
                23: 10, 25: 10, 27: 10, 29: 10,           # right ... -> right shoe
                30: 12, 32: 12,                           # left leg, left foot -> left leg
                31: 13, 33: 13,                           # right leg, right foot -> right leg
                35: 7, 36: 7, 37: 7, 38: 7,               # dress, robe, jumpsuit, other full-body -> dress
                40: 16, 45: 16, 46: 16, 52: 16,           # backpack, carrybag, cases, wallet -> bag
                43: 8,                                    # belt
                47: 3, 48: 3,                             # sunglasses, eyewear -> sunglasses
                50: 17, 55: 17,                           # scarf, tie -> scarf
                41: 0, 42: 0, 44: 0, 51: 0, 53: 0, 54: 0, 56: 0}   # ball, bats, bottle, umbrella, watch, wristband, other
assert sorted(MHP_V2_TO_V1) == list(range(59))
MHP_V2_TO_V1_LUT = np.array([MHP_V2_TO_V1[i] for i in range(59)], dtype=np.uint8)
