"""Train one HoloHuman checkpoint (Section 4, "Implementation Details" and Table 1).

    python train.py --cfg configs/main/coco_w48.yaml [KEY VALUE ...]
    torchrun --nproc_per_node 8 train.py --cfg configs/main/coco_w48.yaml      # DDP, batch 64 in total

Protocol: Adam, lr 1e-3 for the non-backbone parameters and 1e-4 for the backbone, batch
size 64, 140 epochs with the learning rate divided by 10 at epochs 90 and 120, 512 x 512
crops with random rotation (+-30 deg), scaling ([0.75, 1.5]) and horizontal flip, ImageNet-
initialized backbone, instances instantiated at the ground-truth centers, losses of
Sections 3.3-3.4 restricted to the heads of the checkpoint.  The checkpoint after the final
epoch (final_state.pth) is the one evaluated in the paper.
"""
import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from humvis import get_cfg, load_weights
from humvis.config import update_config
from humvis.backbone import load_pretrained_backbone
from humvis.holohuman import HoloHuman
from humvis.training import HoloHumanTrainer
from humvis.datasets import build_train_dataset, collate_train


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--resume', default='', help='checkpoint to resume from (overrides TRAIN.RESUME)')
    ap.add_argument('--init', default='', help='weights to initialize from (overrides TRAIN.INIT_CHECKPOINT)')
    ap.add_argument('--max-iters', type=int, default=0, help='debug: stop after this many iterations per epoch')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    return ap.parse_args()


def setup_distributed():
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        torch.distributed.init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return True, rank, int(os.environ['WORLD_SIZE'])
    return False, 0, 1


def set_seed(seed, rank=0):
    if seed < 0:
        return
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def build_optimizer(cfg, model):
    """Adam with two parameter groups (Section 4): backbone at BACKBONE_LR, the rest at LR."""
    backbone_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    groups = [{'params': other_params, 'lr': cfg.TRAIN.LR, 'name': 'heads'},
              {'params': backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR, 'name': 'backbone'}]
    if cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        return torch.optim.Adam(groups, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.OPTIMIZER.lower() == 'adamw':
        return torch.optim.AdamW(groups, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    raise ValueError(cfg.TRAIN.OPTIMIZER)


def save_checkpoint(path, model, optimizer, scheduler, epoch, cfg):
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'cfg': cfg.dump()}, path)


def main():
    args = parse_args()
    cfg = update_config(get_cfg(), args.cfg, args.opts)
    distributed, rank, world_size = setup_distributed()
    set_seed(cfg.SEED, rank)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.CFG_NAME or os.path.splitext(os.path.basename(args.cfg))[0])
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, 'train.log'))])
    log = logging.getLogger('train')
    if rank == 0:
        with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        log.info(f'config {args.cfg}: {cfg.MODEL.BACKBONE}, heads {list(cfg.MODEL.HEADS.TASKS)}, '
                 f'dataset {cfg.DATASET.DATASET}/{cfg.DATASET.TRAIN}')

    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    model = HoloHuman(cfg, device)
    if cfg.MODEL.PRETRAINED and os.path.isfile(cfg.MODEL.PRETRAINED):
        n, total = load_pretrained_backbone(model.backbone, cfg.MODEL.PRETRAINED)
        log.info(f'ImageNet initialization: {n}/{total} backbone tensors from {cfg.MODEL.PRETRAINED}')
    init = args.init or cfg.TRAIN.INIT_CHECKPOINT
    if init:
        missing, unexpected = load_weights(model, init, strict=False)
        log.info(f'initialized from {init} (missing {len(missing)}, unexpected {len(unexpected)})')
    if cfg.MODEL.SYNC_BN and distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    trainer = HoloHumanTrainer(cfg, model).to(device)

    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(cfg.TRAIN.LR_STEPS), cfg.TRAIN.LR_FACTOR)
    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    resume = args.resume or cfg.TRAIN.RESUME
    if not resume and cfg.AUTO_RESUME and os.path.isfile(os.path.join(out_dir, 'checkpoint.pth')):
        resume = os.path.join(out_dir, 'checkpoint.pth')
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        log.info(f'resumed from {resume} at epoch {start_epoch}')

    dataset = build_train_dataset(cfg)
    sampler = DistributedSampler(dataset, shuffle=cfg.TRAIN.SHUFFLE) if distributed else None
    per_gpu = max(cfg.TRAIN.BATCH_SIZE // world_size, 1)
    loader = DataLoader(dataset, batch_size=per_gpu, shuffle=(sampler is None and cfg.TRAIN.SHUFFLE),
                        sampler=sampler, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY,
                        collate_fn=collate_train, drop_last=True, persistent_workers=cfg.WORKERS > 0)
    log.info(f'{len(dataset)} training images, batch {cfg.TRAIN.BATCH_SIZE} ({per_gpu} x {world_size} GPUs), '
             f'{len(loader)} iterations per epoch')

    step_module = trainer
    if distributed:
        step_module = nn.parallel.DistributedDataParallel(trainer, device_ids=[device.index],
                                                          find_unused_parameters=True)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.TRAIN.AMP and device.type == 'cuda')

    for epoch in range(start_epoch, cfg.TRAIN.END_EPOCH):
        if sampler is not None:
            sampler.set_epoch(epoch)
        step_module.train()
        t0 = time.time()
        meter = {}
        for it, (images, targets, _) in enumerate(loader):
            if args.max_iters and it >= args.max_iters:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                losses = step_module(images, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses['total']).backward()
            if cfg.TRAIN.CLIP_GRAD > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            for k, v in losses.items():
                meter[k] = meter.get(k, 0.0) + float(v.detach())
            if rank == 0 and (it % cfg.PRINT_FREQ == 0 or it == len(loader) - 1):
                msg = ' '.join(f'{k} {float(v):.4f}' for k, v in losses.items())
                log.info(f'epoch {epoch} [{it}/{len(loader)}] lr {optimizer.param_groups[0]["lr"]:.1e} {msg} '
                         f'({(time.time() - t0) / (it + 1):.2f} s/it)')
        scheduler.step()
        n_it = max(min(len(loader), args.max_iters or len(loader)), 1)
        log.info(f'epoch {epoch} done: ' + ' '.join(f'{k} {v / n_it:.4f}' for k, v in meter.items()))
        if rank == 0:
            save_checkpoint(os.path.join(out_dir, 'checkpoint.pth'), model, optimizer, scheduler, epoch, cfg)
            if (epoch + 1) % cfg.TRAIN.SAVE_EVERY == 0:
                torch.save(model.state_dict(), os.path.join(out_dir, f'epoch_{epoch + 1}.pth'))
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(out_dir, 'final_state.pth'))
        log.info(f'saved {os.path.join(out_dir, "final_state.pth")} (checkpoint after epoch {cfg.TRAIN.END_EPOCH})')
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
