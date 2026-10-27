#!/usr/bin/env bash
# Train the checkpoints of Table 1 (Section 4): C-pose/seg, C-part, C-mesh, C-crowd (W32 and W48),
# then C-mhp1 (initialized from C-part).  GPUS = number of GPUs sharing the batch of 64.
set -e
cd "$(dirname "$0")/.."
GPUS=${GPUS:-8}
run() { torchrun --nproc_per_node "$GPUS" train.py --cfg "$1" "${@:2}"; }
for w in w32 w48; do
  run configs/main/coco_$w.yaml
  run configs/main/mhp2_$w.yaml
  run configs/main/mesh_$w.yaml
  run configs/main/crowdpose_$w.yaml
done
for w in w32 w48; do
  run configs/main/mhp1_$w.yaml          # TRAIN.INIT_CHECKPOINT = output/main_mhp2_$w/final_state.pth
done
