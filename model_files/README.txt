Place here:
  basicModel_neutral_lbs_10_207_0_v1.0.0.pkl     SMPL neutral model (smpl.is.tue.mpg.de): mesh head, training, demo
  basicModel_m_lbs_10_207_0_v1.0.0.pkl           SMPL male / female models: only for the 3DPW ground-truth
  basicModel_f_lbs_10_207_0_v1.0.0.pkl           meshes of the MPJPE / PA-MPJPE / PVE evaluation
  J_regressor_h36m.npy                           H36M joint regressor (17 joints)
  hrnetv2_w32_imagenet_pretrained.pth            ImageNet-pretrained HRNet-W32 / W48 (Sun et al., 2019),
  hrnetv2_w48_imagenet_pretrained.pth            backbone initialization of every checkpoint (cfg.MODEL.PRETRAINED)
  holohuman_w32.pth / holohuman_w48.pth          four-head demo checkpoints assembled with tools/merge_checkpoints.py
                                                 from the per-benchmark checkpoints (output/main_*/final_state.pth)
