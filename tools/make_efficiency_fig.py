"""Figure 4 (latency / FPS and peak memory versus the number of persons m) from the output
of time_fig4.py:

    python tools/make_efficiency_fig.py --json efficiency_vs_m.json --out efficiency_vs_m.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

SERIES, INK, INK2, GRID, REF = '#eb6834', '#0b0b0b', '#52514e', '#d9d8d3', '#8a8984'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='efficiency_vs_m.json')
    ap.add_argument('--out', default='efficiency_vs_m.pdf')
    a = ap.parse_args()
    with open(a.json) as f:
        rows = json.load(f)['rows']
    m = [r['m'] for r in rows]
    latency = [r['latency_ms'] for r in rows]
    memory = [r['memory_gb'] for r in rows]

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.edgecolor': INK2, 'axes.linewidth': 0.6,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                         'pdf.fonttype': 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.62})

    ax1.axhline(40, color=REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.axhline(100, color=REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.text(max(m) + 0.6, 40, 'real-time criterion\n(40 ms, 25 FPS)', ha='right', va='bottom', fontsize=7.5, color=INK2)
    ax1.text(max(m) + 0.6, 100, 'interactive (100 ms, 10 FPS)', ha='right', va='bottom', fontsize=7.5, color=INK2)
    ax1.plot(m, latency, color=SERIES, lw=2, marker='o', ms=5.5, mec=SERIES, mew=0.8, zorder=3)
    for mi, li in ((m[0], latency[0]), (m[-1], latency[-1])):
        ax1.annotate(f'{li:.1f} ms ({1000 / li:.1f} FPS)', (mi, li), xytext=(mi + (0.3 if mi == m[0] else -0.4), li + 12),
                     fontsize=7.5, ha='left' if mi == m[0] else 'right', va='bottom', color=INK)
    slope = (latency[-1] - latency[0]) / max(m[-1] - m[0], 1)
    ax1.text(np.mean(m), np.mean(latency) + 25, f'+{slope:.1f} ms per person', fontsize=7.5, color=INK2, ha='center')
    ax1.set_xlabel('number of persons $m$')
    ax1.set_ylabel('latency (ms)')
    ax1.set_xlim(0, max(m) + 1)
    ax1.set_ylim(0, max(300, max(latency) * 1.15))
    ax1.set_xticks(m)
    _ms_to_fps = lambda y: 1000.0 / np.maximum(np.asarray(y, dtype=float), 1e-3)
    sec = ax1.secondary_yaxis('right', functions=(_ms_to_fps, _ms_to_fps))
    fps_ticks = [25, 16, 10, 5, 4]
    sec.yaxis.set_major_locator(FixedLocator(fps_ticks))
    sec.yaxis.set_major_formatter(FixedFormatter([f'{f:g}' for f in fps_ticks]))
    sec.set_ylabel('throughput (FPS)', labelpad=6)
    sec.tick_params(colors=INK2, labelsize=8)
    ax1.set_title('(a) Inference time', loc='left', fontsize=9)

    if all(np.isfinite(memory)):            # memory is only measured on a GPU
        ax2.plot(m, memory, color=SERIES, lw=2, marker='o', ms=5.5, mec=SERIES, mew=0.8, zorder=3)
        ax2.annotate(f'{memory[0]:.1f} GB', (m[0], memory[0]), xytext=(m[0] + 0.8, memory[0] - 0.04), fontsize=7.5,
                     va='top', color=INK)
        ax2.annotate(f'{memory[-1]:.1f} GB', (m[-1], memory[-1]), xytext=(m[-1] - 0.8, memory[-1] + 0.03), fontsize=7.5,
                     ha='right', va='bottom', color=INK)
        ax2.set_ylim(min(memory) - 0.2, max(memory) + 0.2)
    else:
        ax2.text(0.5, 0.5, 'peak memory not measured\n(no CUDA device)', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=7.5, color=INK2)
    ax2.set_xlabel('number of persons $m$')
    ax2.set_ylabel('peak GPU memory (GB)')
    ax2.set_xlim(0, max(m) + 1)
    ax2.set_xticks(m)
    ax2.set_title('(b) Peak memory', loc='left', fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig.savefig(a.out, bbox_inches='tight')
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
