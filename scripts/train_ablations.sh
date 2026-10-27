#!/usr/bin/env bash
# One checkpoint per ablation row and benchmark (Tables 2-6 and 8), HRNet-W48, full schedule.
# Usage: scripts/train_ablations.sh [table4|table2_3|table5|table6|table8|all]
set -e
cd "$(dirname "$0")/.."
GPUS=${GPUS:-8}
TABLES=${1:-all}
[ "$TABLES" = all ] && TABLES="table4 table2_3 table5 table6 table8"
for t in $TABLES; do
  for cfg in configs/ablation/$t/*.yaml; do
    name=$(basename "$cfg" .yaml)
    # the full configuration (table4/row7_*, table5/loss_fl_ctr_int_*, lambda_0.1_*, tau_0.05_*) is the
    # main W48 checkpoint; skip it here and read its metrics from output/main_<benchmark>_w48
    case "$name" in row7_*|loss_fl_ctr_int_*|lambda_0.1_*|tau_0.05_*) continue;; esac
    torchrun --nproc_per_node "$GPUS" train.py --cfg "$cfg"
    python eval.py --cfg "$cfg" --weights "output/$(python -c "import yaml,sys;print(yaml.safe_load(open('$cfg'))['CFG_NAME'])")/final_state.pth"
  done
done
