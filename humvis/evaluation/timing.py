"""Latency / throughput / memory measurement (Section 4, Figures 4 and 5, Table 9): batch size 1,
512 x 512 input, fp32, single scale with flip test, median of CUDA-synchronized runs after
warm-up, peak memory from torch.cuda.max_memory_allocated."""
import time

import numpy as np
import torch


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def time_forward(model, image, centers=None, runs=100, warmup=10):
    """Median latency (ms) of `runs` forward passes of one image (1, 3, S, S), and the
    peak memory (GB) of the measured passes.  centers: optional (m, 2) cells replacing Peak(P)."""
    device = model.device
    image = image.to(device)
    for _ in range(warmup):
        model(image, centers=centers)
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(runs):
        synchronize(device)
        t0 = time.perf_counter()
        model(image, centers=centers)
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else float('nan')
    return float(np.median(times)), mem, times


def grid_centers(m, height, width):
    """m centers on a regular grid (used to pad an image with fewer annotated persons)."""
    g = int(np.ceil(np.sqrt(m)))
    pts = [((i % g + 1) * width / (g + 1), (i // g + 1) * height / (g + 1)) for i in range(m)]
    return torch.tensor(pts, dtype=torch.float32)


def pad_centers(centers, m, height, width):
    """Keep the first m annotated centers, or pad them with grid centers up to m."""
    centers = torch.as_tensor(centers, dtype=torch.float32).view(-1, 2)
    if centers.shape[0] >= m:
        return centers[:m]
    extra = grid_centers(m - centers.shape[0], height, width)
    return torch.cat((centers, extra), dim=0)
