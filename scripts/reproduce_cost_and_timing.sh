#!/usr/bin/env bash
# Table 7 (analytic cost, any machine), Figure 4 and Figure 5 / Table 9 (RTX 3090 timing).
set -e
cd "$(dirname "$0")/.."
python profile_cost.py --m 1 5                          # Table 7, HRNet-W32: 92.34 / 278.61 GMACs, 34.40 M
python profile_cost.py --backbone hrnet48 --m 1         # Table 4 full row: 196.05 GMACs, 74.47 M
python time_fig4.py --weights output/main_coco_w32/final_state.pth --out efficiency_vs_m.json
python tools/make_efficiency_fig.py --json efficiency_vs_m.json --out efficiency_vs_m.pdf
python time_fig5.py --weights output/main_coco_w32/final_state.pth --out runtime_per_task.json
