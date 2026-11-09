#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本
16 通道原始输入经预处理得到 15 通道；预测 1 通道 asinh(von Mises 应力)。
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", category=FutureWarning, module="mamba_ssm")

import os
import sys
import time
import argparse
import csv
import json
import multiprocessing as mp
import queue as queue_module
from datetime import datetime, timedelta


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from .config import config
from .transforms import load_stress_stats, transform_settings
from .model import SegMamba
from .dataset import get_dataloaders
from .utils import StressLoss, save_r2_scatter



def parse_args():
    # Filter out --local-rank from sys.argv to avoid argparse errors
    # (torch.distributed.launch passes --local-rank automatically)
    filtered_argv = [arg for arg in sys.argv if not arg.startswith('--local-rank')]
    
    parser = argparse.ArgumentParser(description='Train Stress Mamba Model')
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS, help='DataLoader workers')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--debug', action='store_true', help='Limit training models and validate each epoch; set --epochs/--batch_size explicitly for a short run')
    
    args = parser.parse_args(filtered_argv[1:])
    
    # Get local_rank from environment if set by torch.distributed.launch
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    return args


def setup_distributed():
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank == -1:
        print("[Single GPU mode] No LOCAL_RANK found.", flush=True)
        return 0, 1

    print(
        f"[Pre-init] rank={rank}, local_rank={local_rank}, world_size={world_size}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True
    )

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}")
    )

    return local_rank, dist.get_world_size()


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, rank=0, use_amp=True):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_count = 0

    desc = "Training" if rank == 0 else f"Training Rank {rank}"
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=desc, disable=(rank != 0))):
        input_data = batch['input'].to(device)
        stress_target = batch['stress'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred_stress = model(input_data)
                loss, loss_dict = criterion(pred_stress, stress_target, input_data)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_stress = model(input_data)
            loss, loss_dict = criterion(pred_stress, stress_target, input_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
        
        total_loss += loss_dict['total_loss']
        loss_count += 1
    
    return total_loss / loss_count


def _plot_worker(plot_queue):
    while True:
        item = plot_queue.get()
        if item is None:
            break
        scatter_true, scatter_pred, r2, nrmse, scatter_path, dpi, title = item
        save_r2_scatter(
            scatter_true,
            scatter_pred,
            scatter_path,
            r2=r2,
            nrmse=nrmse,
            dpi=dpi,
            title=title
        )


def validate(model, val_loader, criterion, device, collect_scatter=False, stress_median=1.0, rank=0, world_size=1):
    """在验证 patches 上计算损失和指标（不拼接完整体积）。

    按样本所属 model 分组统计指标和散点；损失仍按 batch 等权平均。
    各卡重复完整验证，按 model 汇总后的指标再对 model 等权平均。
    """
    model.eval()
    total_loss = 0
    total_loss_asinh = 0
    total_loss_boundary = 0
    total_loss_gradient = 0
    loss_count = 0

    model_stats = {}
    
    scatter_data = None
    model_scatter_data = {}
    num_points = config.VAL_SCATTER_NUM_POINTS

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=(rank != 0)):
            input_data = batch['input'].to(device)
            stress_target = batch['stress'].to(device)
            
            model_names = batch['model_name']
            if isinstance(model_names, str):
                model_names = [model_names]
            if len(model_names) != input_data.shape[0]:
                raise ValueError('Every validation patch must have its own model_name')
            indices_by_model = {}
            for sample_idx, name in enumerate(model_names):
                indices_by_model.setdefault(name, []).append(sample_idx)
            
            pred_stress = model(input_data)
            loss, loss_dict = criterion(pred_stress, stress_target, input_data)
            
            total_loss += loss_dict['total_loss']
            total_loss_asinh += loss_dict.get('loss_asinh', 0)
            total_loss_boundary += loss_dict.get('loss_boundary', 0)
            total_loss_gradient += loss_dict.get('loss_gradient', 0)
            loss_count += 1
            
            pred_real = torch.sinh(pred_stress) * stress_median
            true_real = torch.sinh(stress_target) * stress_median

            for model_name, indices in indices_by_model.items():
                pred_flat = pred_real[indices].flatten()
                true_flat = true_real[indices].flatten()
                diff = pred_flat - true_flat

                pred_asinh_flat = pred_stress[indices].flatten()
                true_asinh_flat = stress_target[indices].flatten()
                diff_asinh = pred_asinh_flat - true_asinh_flat

                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'total_count': 0,
                        'sum_abs_error': 0.0,
                        'sum_sq_error': 0.0,
                        'sum_true': 0.0,
                        'sum_true_sq': 0.0,
                        'sum_abs_error_asinh': 0.0,
                        'sum_sq_error_asinh': 0.0,
                        'sum_true_asinh': 0.0,
                        'sum_true_asinh_sq': 0.0,
                    }

                stats = model_stats[model_name]
                stats['total_count'] += true_flat.numel()
                stats['sum_abs_error'] += diff.abs().sum().item()
                stats['sum_sq_error'] += (diff ** 2).sum().item()
                stats['sum_true'] += true_flat.sum().item()
                stats['sum_true_sq'] += (true_flat ** 2).sum().item()
                stats['sum_abs_error_asinh'] += diff_asinh.abs().sum().item()
                stats['sum_sq_error_asinh'] += (diff_asinh ** 2).sum().item()
                stats['sum_true_asinh'] += true_asinh_flat.sum().item()
                stats['sum_true_asinh_sq'] += (true_asinh_flat ** 2).sum().item()

                if collect_scatter and num_points > 0:
                    true_real_cpu = true_flat.detach().cpu()
                    pred_real_cpu = pred_flat.detach().cpu()
                    true_asinh_cpu = true_asinh_flat.detach().cpu()
                    pred_asinh_cpu = pred_asinh_flat.detach().cpu()

                    if model_name not in model_scatter_data:
                        model_scatter_data[model_name] = {
                            'true_real': torch.empty((0,), dtype=torch.float32),
                            'pred_real': torch.empty((0,), dtype=torch.float32),
                            'true_asinh': torch.empty((0,), dtype=torch.float32),
                            'pred_asinh': torch.empty((0,), dtype=torch.float32),
                            'max_points': num_points,
                            'generator': torch.Generator(device='cpu').manual_seed(config.VAL_SCATTER_SEED + hash(model_name) % 1000),
                        }

                    data = model_scatter_data[model_name]
                    current_size = data['true_real'].numel()

                    if current_size < num_points:
                        batch_size = true_real_cpu.numel()
                        points_to_sample = min(batch_size, num_points - current_size)
                        gen = data['generator']

                        if points_to_sample > 0:
                            indices = torch.randperm(batch_size, generator=gen)[:points_to_sample]

                            data['true_real'] = torch.cat([data['true_real'], true_real_cpu[indices]])
                            data['pred_real'] = torch.cat([data['pred_real'], pred_real_cpu[indices]])
                            data['true_asinh'] = torch.cat([data['true_asinh'], true_asinh_cpu[indices]])
                            data['pred_asinh'] = torch.cat([data['pred_asinh'], pred_asinh_cpu[indices]])

    # ===== 分布式统计合并 =====
    if world_size > 1:
        import torch.distributed as dist
        
        all_model_names = list(model_stats.keys())
        
        for model_name in all_model_names:
            stats = model_stats[model_name]
            tensor_stats = torch.tensor([
                stats['total_count'],
                stats['sum_abs_error'],
                stats['sum_sq_error'],
                stats['sum_true'],
                stats['sum_true_sq'],
                stats['sum_abs_error_asinh'],
                stats['sum_sq_error_asinh'],
                stats['sum_true_asinh'],
                stats['sum_true_asinh_sq'],
            ], dtype=torch.float32, device=device)
            
            dist.all_reduce(tensor_stats, op=dist.ReduceOp.SUM)
            
            model_stats[model_name] = {
                'total_count': int(tensor_stats[0].item()),
                'sum_abs_error': float(tensor_stats[1].item()),
                'sum_sq_error': float(tensor_stats[2].item()),
                'sum_true': float(tensor_stats[3].item()),
                'sum_true_sq': float(tensor_stats[4].item()),
                'sum_abs_error_asinh': float(tensor_stats[5].item()),
                'sum_sq_error_asinh': float(tensor_stats[6].item()),
                'sum_true_asinh': float(tensor_stats[7].item()),
                'sum_true_asinh_sq': float(tensor_stats[8].item()),
            }
        
        dist.barrier()

    avg_loss = total_loss / loss_count
    avg_loss_asinh = total_loss_asinh / loss_count
    avg_loss_boundary = total_loss_boundary / loss_count
    avg_loss_gradient = total_loss_gradient / loss_count

    def compute_metrics(stats):
        total_count = stats['total_count']
        if total_count == 0:
            return {'mae': 0, 'rmse': 0, 'r2': 0, 'nrmse': 0, 'r2_asinh': 0, 'nrmse_asinh': 0}
        
        mae = stats['sum_abs_error'] / total_count
        rmse = (stats['sum_sq_error'] / total_count) ** 0.5
        mean_true = stats['sum_true'] / total_count
        ss_tot = stats['sum_true_sq'] - total_count * (mean_true ** 2)
        r2 = 1 - (stats['sum_sq_error'] / (ss_tot + 1e-8))
        nrmse = stats['sum_abs_error'] / (stats['sum_true'] + 1e-8)
        
        mean_true_asinh = stats['sum_true_asinh'] / total_count
        ss_tot_asinh = stats['sum_true_asinh_sq'] - total_count * (mean_true_asinh ** 2)
        r2_asinh = 1 - (stats['sum_sq_error_asinh'] / (ss_tot_asinh + 1e-8))
        nrmse_asinh = stats['sum_abs_error_asinh'] / (stats['sum_true_asinh'] + 1e-8)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'nrmse': nrmse,
            'r2_asinh': r2_asinh,
            'nrmse_asinh': nrmse_asinh
        }

    per_model_metrics = {}
    for model_name in sorted(model_stats.keys()):
        per_model_metrics[model_name] = compute_metrics(model_stats[model_name])

    num_models = len(model_stats)
    if num_models > 0:
        avg_metrics = {
            'mae': sum(m['mae'] for m in per_model_metrics.values()) / num_models,
            'rmse': sum(m['rmse'] for m in per_model_metrics.values()) / num_models,
            'r2': sum(m['r2'] for m in per_model_metrics.values()) / num_models,
            'nrmse': sum(m['nrmse'] for m in per_model_metrics.values()) / num_models,
            'r2_asinh': sum(m['r2_asinh'] for m in per_model_metrics.values()) / num_models,
            'nrmse_asinh': sum(m['nrmse_asinh'] for m in per_model_metrics.values()) / num_models,
            'loss_asinh': avg_loss_asinh,
            'loss_boundary': avg_loss_boundary,
            'loss_gradient': avg_loss_gradient,
        }
    else:
        avg_metrics = {
            'mae': 0, 'rmse': 0, 'r2': 0, 'nrmse': 0,
            'r2_asinh': 0, 'nrmse_asinh': 0,
            'loss_asinh': avg_loss_asinh,
            'loss_boundary': avg_loss_boundary,
            'loss_gradient': avg_loss_gradient,
        }

    if collect_scatter and model_scatter_data:
        scatter_data = {}
        for model_name, data in model_scatter_data.items():
            scatter_data[model_name] = (
                data['true_real'],
                data['pred_real'],
                data['true_asinh'],
                data['pred_asinh']
            )

    return avg_loss, avg_metrics, per_model_metrics, scatter_data


def main():
    args = parse_args()
    
    # CLI defaults were already parsed; --debug alone does not reset args.epochs/batch_size.
    if args.debug:
        config.DEBUG_MODE = True

    from .data_policy import validate_splits, validate_stats, stats_path
    split_manifest = validate_splits(config)
    stats_hash = validate_stats(config, split_manifest)
    stress_percentiles, stress_scale_percentile = load_stress_stats(config.STATS_DATA_DIR, (config.ASINH_PERCENTILE, config.LABEL_WEIGHT_PERCENTILE))
    
    local_rank, world_size = setup_distributed()
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if local_rank == 0:
        print(f"Using device: {device}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all visible devices')}")
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Train data dir: {config.TRAIN_DATA_DIR}")
        print(f"Val data dir: {config.VAL_DATA_DIR}")
        print(f"Train patch dir: {config.TRAIN_PATCH_DIR_NAME}")
        print(f"Val patch dir: {config.VAL_PATCH_DIR_NAME}")
        if stress_scale_percentile is not None:
            print(f"Stress asinh percentile: P{float(config.ASINH_PERCENTILE):g}")
        print(f"Stress scale value: {stress_percentiles[float(config.ASINH_PERCENTILE)]:.6f}")
        print(f"Label weight percentile: P{float(config.LABEL_WEIGHT_PERCENTILE):g}")
        print(f"Label weight value: {stress_percentiles[float(config.LABEL_WEIGHT_PERCENTILE)]:.6f}")
        print(
            f"Label weight: alpha={config.LABEL_WEIGHT_ALPHA}, "
            f"gamma={config.LABEL_WEIGHT_GAMMA}, c={config.LABEL_WEIGHT_CLIP_C}"
        )
        print(f"Max epochs: {args.epochs}")
        
        os.makedirs(config.LOG_DIR, exist_ok=True)
        with open(os.path.join(config.LOG_DIR, 'data_manifest.json'), 'w') as f:
            json.dump({'splits': split_manifest, 'transform_stats_sha256': stats_hash}, f, indent=2)
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        scatter_dir = os.path.join(config.LOG_DIR, config.VAL_SCATTER_DIRNAME)
        os.makedirs(scatter_dir, exist_ok=True)
        metrics_path = os.path.join(config.LOG_DIR, config.VAL_METRICS_FILENAME)
        
        writer = SummaryWriter(config.LOG_DIR)

        plot_queue = None
        plot_process = None
        if config.ASYNC_PLOT:
            plot_queue = mp.Queue(maxsize=config.PLOT_QUEUE_MAXSIZE)
            plot_process = mp.Process(target=_plot_worker, args=(plot_queue,))
            plot_process.daemon = True
            plot_process.start()
    
    max_train_models = config.DEBUG_MAX_TRAIN_MODELS if config.DEBUG_MODE else None

    train_loader, val_loader = get_dataloaders(
        config.TRAIN_DATA_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=(config.PATCH_D, config.PATCH_H, config.PATCH_W),
        stride=(config.STRIDE_D, config.STRIDE_H, config.STRIDE_W),
        use_sliding_window=True,
        split_train_test=False,
        use_prepatched=True,
        patch_dir_name=config.VAL_PATCH_DIR_NAME,
        use_cache=False,
        rank=local_rank,
        world_size=world_size,
        val_model_indices=None,
        num_val_models=0,  # Directories already define whole-model subsets.
        train_data_dir=config.TRAIN_DATA_DIR,
        val_data_dir=config.VAL_DATA_DIR,
        train_patch_dir_name=config.TRAIN_PATCH_DIR_NAME,
        val_patch_dir_name=config.VAL_PATCH_DIR_NAME,
        asinh_percentile=config.ASINH_PERCENTILE,
        max_train_models=max_train_models,
        stats_path=str(stats_path(config))
    )
    
    if local_rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    model = SegMamba(
        in_chans=config.IN_CHANS,
        out_chans=config.OUT_CHANS,
        depths=config.DEPTHS,
        feat_size=config.FEAT_SIZE,
        hidden_size=config.HIDDEN_SIZE
    )
    model = model.to(device)
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    criterion = StressLoss(
        vonmises_weight=config.VONMISES_LOSS_WEIGHT,
        stress_median=stress_percentiles[float(config.ASINH_PERCENTILE)],
        grad_weight=config.GRAD_WEIGHT,
        label_weight_base=stress_percentiles[float(config.LABEL_WEIGHT_PERCENTILE)],
        label_weight_alpha=config.LABEL_WEIGHT_ALPHA,
        label_weight_gamma=config.LABEL_WEIGHT_GAMMA,
        label_weight_clip_c=config.LABEL_WEIGHT_CLIP_C,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    best_metric = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 50)
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, local_rank, config.USE_AMP
        )
        
        if (epoch + 1) % config.VAL_EVERY == 0:
            val_loss, val_metrics, per_model_metrics, scatter_data = validate(
                model,
                val_loader,
                criterion,
                device,
                collect_scatter=True,
                stress_median=stress_percentiles[float(config.ASINH_PERCENTILE)],
                rank=local_rank,
                world_size=world_size
            )
            
            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()
            
            if local_rank == 0:
                print(f"\nTrain Loss: {train_loss:.6f}")
                print(f"Val Loss: {val_loss:.6f}")
                print(f"Val Loss (asinh): {val_metrics.get('loss_asinh', 0):.6f}")
                print(f"Val Loss (boundary): {val_metrics.get('loss_boundary', 0):.6f}")
                print(f"Val Loss (gradient): {val_metrics.get('loss_gradient', 0):.6f}")
                print(f"Val Avg MAE: {val_metrics['mae']:.6f}")
                print(f"Val Avg R2 (real): {val_metrics['r2']:.6f}")
                print(f"Val Avg NRMSE (real): {val_metrics['nrmse']:.6f}")
                print(f"Val Avg R2 (asinh): {val_metrics['r2_asinh']:.6f}")
                print(f"Val Avg NRMSE (asinh): {val_metrics['nrmse_asinh']:.6f}")
                
                print("\nPer-model metrics:")
                for model_name in sorted(per_model_metrics.keys()):
                    m = per_model_metrics[model_name]
                    print(f"  {model_name}: MAE={m['mae']:.6f}, R2={m['r2']:.6f}, NRMSE={m['nrmse']:.6f}")
                
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Metrics/mae', val_metrics['mae'], epoch)
                writer.add_scalar('Metrics/r2_real', val_metrics['r2'], epoch)
                writer.add_scalar('Metrics/nrmse_real', val_metrics['nrmse'], epoch)
                writer.add_scalar('Metrics/r2_asinh', val_metrics['r2_asinh'], epoch)
                writer.add_scalar('Metrics/nrmse_asinh', val_metrics['nrmse_asinh'], epoch)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

                if scatter_data is not None:
                    for model_name in sorted(scatter_data.keys()):
                        model_scatter = scatter_data[model_name]
                        if model_scatter is None:
                            continue
                        scatter_true_real, scatter_pred_real, scatter_true_asinh, scatter_pred_asinh = model_scatter
                        
                        scatter_path_real = os.path.join(
                            scatter_dir,
                            f"epoch_{epoch + 1:04d}_{model_name}_r2_scatter_real.png"
                        )
                        scatter_path_asinh = os.path.join(
                            scatter_dir,
                            f"epoch_{epoch + 1:04d}_{model_name}_r2_scatter_asinh.png"
                        )
                        
                        model_metrics = per_model_metrics[model_name]

                        if config.ASYNC_PLOT and plot_queue is not None:
                            try:
                                plot_queue.put_nowait((
                                    scatter_true_real,
                                    scatter_pred_real,
                                    model_metrics['r2'],
                                    model_metrics['nrmse'],
                                    scatter_path_real,
                                    config.VAL_SCATTER_DPI,
                                    f'{model_name} R2 Scatter (Real)'
                                ))
                                plot_queue.put_nowait((
                                    scatter_true_asinh,
                                    scatter_pred_asinh,
                                    model_metrics['r2_asinh'],
                                    model_metrics['nrmse_asinh'],
                                    scatter_path_asinh,
                                    config.VAL_SCATTER_DPI,
                                    f'{model_name} R2 Scatter (Asinh)'
                                ))
                            except queue_module.Full:
                                pass
                        else:
                            save_r2_scatter(
                                scatter_true_real,
                                scatter_pred_real,
                                scatter_path_real,
                                r2=model_metrics['r2'],
                                nrmse=model_metrics['nrmse'],
                                dpi=config.VAL_SCATTER_DPI,
                                title=f'{model_name} R2 Scatter (Real)'
                            )
                            save_r2_scatter(
                                scatter_true_asinh,
                                scatter_pred_asinh,
                                scatter_path_asinh,
                                r2=model_metrics['r2_asinh'],
                                nrmse=model_metrics['nrmse_asinh'],
                                dpi=config.VAL_SCATTER_DPI,
                                title=f'{model_name} R2 Scatter (Asinh)'
                            )

                if config.SAVE_CHECKPOINT_EVERY > 0 and (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
                    checkpoint_path = os.path.join(
                        config.MODEL_SAVE_PATH,
                        f"checkpoint_epoch_{epoch + 1:04d}.pth"
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'best_metric': best_metric,
                        'transform_stats_sha256': stats_hash,
                        'transform_settings': transform_settings(config, stress_percentiles),
                        'val_metrics': val_metrics
                    }, checkpoint_path)
                    
                    latest_path = os.path.join(config.MODEL_SAVE_PATH, 'latest_checkpoint.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'best_metric': best_metric,
                        'transform_stats_sha256': stats_hash,
                        'transform_settings': transform_settings(config, stress_percentiles),
                        'val_metrics': val_metrics
                    }, latest_path)

                write_header = not os.path.exists(metrics_path)
                with open(metrics_path, mode='a', newline='') as f:
                    writer_csv = csv.writer(f)
                    if write_header:
                        writer_csv.writerow([
                            'epoch',
                            'val_loss',
                            'loss_asinh',
                            'loss_boundary',
                            'loss_gradient',
                            'r2_real',
                            'nrmse_real',
                            'r2_asinh',
                            'nrmse_asinh',
                            'mae',
                            'rmse'
                        ])
                    writer_csv.writerow([
                        epoch + 1,
                        f"{val_loss:.8f}",
                        f"{val_metrics.get('loss_asinh', 0):.8f}",
                        f"{val_metrics.get('loss_boundary', 0):.8f}",
                        f"{val_metrics.get('loss_gradient', 0):.8f}",
                        f"{val_metrics['r2']:.8f}",
                        f"{val_metrics['nrmse']:.8f}",
                        f"{val_metrics['r2_asinh']:.8f}",
                        f"{val_metrics['nrmse_asinh']:.8f}",
                        f"{val_metrics['mae']:.8f}",
                        f"{val_metrics['rmse']:.8f}"
                    ])
                
                # Only 'mae' selects MAE; the default and all other values select val_loss.
                current_metric = val_metrics['mae'] if config.BEST_METRIC == 'mae' else val_loss
                
                if config.SAVE_BEST_MODEL and current_metric < best_metric:
                    best_metric = current_metric
                    save_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'best_metric': best_metric,
                        'transform_stats_sha256': stats_hash,
                        'transform_settings': transform_settings(config, stress_percentiles),
                        'val_metrics': val_metrics
                    }, save_path)
                    print(f"Saved best model to {save_path}")
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        if local_rank == 0:
            print(f"Epoch time: {epoch_time:.2f}s")
            print(f"Estimated remaining: {epoch_time * (args.epochs - epoch - 1) / 60:.1f} min")
    
    if local_rank == 0:
        if config.ASYNC_PLOT and plot_queue is not None:
            plot_queue.put(None)
            plot_process.join()
        writer.close()
        print("\nTraining completed!")
        print(f"Logs saved to {config.LOG_DIR}")
        print(f"Best model saved to {config.MODEL_SAVE_PATH}")
    
    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
