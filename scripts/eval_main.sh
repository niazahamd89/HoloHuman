#!/usr/bin/env bash
# Evaluate the Table 1 checkpoints with the Section 4 protocol (single scale, flip test):
#   COCO val (Tables 22, 12), COCO test-dev json (Tables 10, 13), CrowdPose cross-dataset and in-domain
#   (Table 11), MHP v2.0 val (Table 15), MHP v1.0 cross-dataset and in-domain (Table 14),
#   3DPW test (Table 16), CMU-Panoptic (Table 17).
set -e
cd "$(dirname "$0")/.."
for w in w32 w48; do
  python eval.py --cfg configs/main/coco_$w.yaml      --weights output/main_coco_$w/final_state.pth
  python eval.py --cfg configs/eval/coco_testdev_$w.yaml
  python eval.py --cfg configs/eval/crowdpose_cross_$w.yaml
  python eval.py --cfg configs/main/crowdpose_$w.yaml --weights output/main_crowdpose_$w/final_state.pth
  python eval.py --cfg configs/main/mhp2_$w.yaml      --weights output/main_mhp2_$w/final_state.pth
  python eval.py --cfg configs/eval/mhp1_cross_$w.yaml DATASET.TEST val
  python eval.py --cfg configs/eval/mhp1_cross_$w.yaml DATASET.TEST test
  python eval.py --cfg configs/main/mhp1_$w.yaml      --weights output/main_mhp1_$w/final_state.pth DATASET.TEST val
  python eval.py --cfg configs/main/mhp1_$w.yaml      --weights output/main_mhp1_$w/final_state.pth DATASET.TEST test
  python eval.py --cfg configs/main/mesh_$w.yaml      --weights output/main_mesh_$w/final_state.pth
  python eval.py --cfg configs/eval/panoptic_$w.yaml
done
