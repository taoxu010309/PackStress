#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本
配置 configs/data.yaml 和 configs/evaluate.yaml 后运行 python evaluate.py。
先在 asinh 空间拼接 patch，再反变换并计算完整体积指标。
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", category=FutureWarning, module="mamba_ssm")

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import time

# 尝试导入matplotlib，如果失败则跳过绘图功能
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, scatter plots will be disabled")
    MATPLOTLIB_AVAILABLE = False

from .config import config
from .transforms import load_stress_stats, transform_settings, validate_checkpoint_transform
from .model import SegMamba
from .dataset import StressDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test Stress Mamba Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None, help='Test data directory (default: config.DATA_DIR)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--plot_scatter', action='store_true', help='Generate scatter plots for R2 visualization')
    parser.add_argument('--sample_ratio', type=float, default=0.2, help='Requested scatter fraction, capped by max_points (default: 0.2)')
    parser.add_argument('--max_points', type=int, default=50000, help='Maximum number of points in scatter plots (default: 50000)')
    return parser.parse_args()


def build_expected_patch_counts(dataset):
    """构建每个模型应处理的patch数量。"""
    expected_patch_counts = {}

    if hasattr(dataset, 'use_prepatched') and dataset.use_prepatched and hasattr(dataset, 'patch_metadata'):
        for sample_path, meta in dataset.patch_metadata.items():
            model_name = os.path.basename(sample_path)
            expected_patch_counts[model_name] = int(meta.get('num_patches', 0))
        return expected_patch_counts

    if hasattr(dataset, 'use_sliding_window') and dataset.use_sliding_window and hasattr(dataset, 'sample_patches'):
        for sample_path, patches in zip(dataset.samples, dataset.sample_patches):
            model_name = os.path.basename(sample_path)
            expected_patch_counts[model_name] = len(patches)
        return expected_patch_counts

    if hasattr(dataset, 'samples'):
        for sample_path in dataset.samples:
            model_name = os.path.basename(sample_path)
            expected_patch_counts[model_name] = 1

    return expected_patch_counts


def test_model(model, test_loader, device, stress_median, output_dir='./test_results',
               plot_scatter=False, sample_ratio=0.2, max_points=50000):
    """
    测试模型 - 重建完整3D体积后计算指标
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        output_dir: 输出目录
        plot_scatter: 是否生成散点图
        sample_ratio: 散点图目标采样比例，点数还受 max_points 限制
        max_points: 散点图最大点数
    
    Returns:
        avg_metrics: 成功计算指标的模型之间等权汇总（mean/std/min/max）
        model_avg_metrics: 每个模型的指标（含test_time_sec）
    """
    model.eval()
    
    # 检查数据集是否支持体积重建
    dataset = test_loader.dataset
    support_volume_reconstruction = hasattr(dataset, 'get_volume_shape') and hasattr(dataset, 'load_full_volume')
    
    if not support_volume_reconstruction:
        print("Warning: Dataset does not support volume reconstruction, falling back to patch-based metrics")
        # 原实现没有 patch 指标回退分支；此处直接返回空字典。
        return {}, {}
    
    # 按模型存储重建数据
    model_reconstructions = {}
    expected_patch_counts = build_expected_patch_counts(dataset)
    model_patch_counts = {}
    model_start_times = {}
    model_test_times = {}
    finalized_models = set()
    sample_path_by_model = {
        os.path.basename(sample_path): sample_path for sample_path in getattr(dataset, 'samples', [])
    }

    def sync_device():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def to_int(x):
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError(f"Expected scalar tensor, got shape {tuple(x.shape)}")
            return int(x.item())
        return int(x)

    def get_triplet(batch_value, sample_idx, batch_size, field_name):
        if batch_value is None:
            return None
        if not isinstance(batch_value, (list, tuple)) or len(batch_value) != 3:
            raise ValueError(f"Expected {field_name} to be list/tuple of length 3, got {type(batch_value)}")

        values = []
        for dim_value in batch_value:
            if isinstance(dim_value, torch.Tensor):
                if dim_value.ndim == 0:
                    values.append(to_int(dim_value))
                elif dim_value.shape[0] == batch_size:
                    values.append(to_int(dim_value[sample_idx]))
                elif dim_value.numel() == 1:
                    values.append(to_int(dim_value.reshape(-1)[0]))
                else:
                    raise ValueError(
                        f"Unexpected tensor shape {tuple(dim_value.shape)} in {field_name} for batch_size={batch_size}"
                    )
            elif isinstance(dim_value, (list, tuple)):
                values.append(to_int(dim_value[sample_idx]))
            else:
                values.append(to_int(dim_value))
        return tuple(values)

    def finalize_model_prediction(model_name):
        if model_name in finalized_models:
            return

        recon_data = model_reconstructions[model_name]
        count_accum = recon_data['count_accum']
        non_zero_mask = count_accum > 0
        if not non_zero_mask.any():
            print(f"Warning: No valid predictions for model {model_name}")
            recon_data['pred_final'] = None
            finalized_models.add(model_name)
            return

        pred_final = torch.zeros_like(recon_data['pred_accum'])
        pred_final[non_zero_mask] = recon_data['pred_accum'][non_zero_mask] / count_accum[non_zero_mask]
        recon_data['pred_final'] = pred_final
        finalized_models.add(model_name)
    
    print("Starting inference loop...")
    print(f"Total batches in test_loader: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0 or batch_idx == len(test_loader) - 1:
                print(f"Processing batch {batch_idx}/{len(test_loader)-1}...")

            input_data = batch['input'].to(device, non_blocking=True)
            model_names = batch['model_name']
            if isinstance(model_names, str):
                model_names = [model_names]
            else:
                model_names = list(model_names)

            patch_coords = batch.get('patch_coords', None)
            patch_size = batch.get('patch_size', None)
            batch_size = int(input_data.shape[0])

            new_model_names = [name for name in model_names if name not in model_start_times]
            if new_model_names:
                sync_device()
                start_time = time.perf_counter()
                for model_name in set(new_model_names):
                    model_start_times[model_name] = start_time
            
            # 获取预测
            pred_stress = model(input_data)  # (B, 1, patch_d, patch_h, patch_w)
            pred_cpu = pred_stress.cpu()

            for sample_idx in range(batch_size):
                model_name = model_names[sample_idx]

                # 初始化或获取该模型的累加器
                if model_name not in model_reconstructions:
                    sample_path = sample_path_by_model.get(model_name)
                    if sample_path is None:
                        print(f"Warning: Cannot find sample path for model {model_name}")
                        continue

                    vol_shape = dataset.get_volume_shape(sample_path)
                    if vol_shape is None:
                        print(f"Warning: Cannot get volume shape for model {model_name}")
                        continue

                    # 创建累加器：预测值累加和计数累加
                    full_shape = (1, 1, vol_shape[0], vol_shape[1], vol_shape[2])
                    model_reconstructions[model_name] = {
                        'pred_accum': torch.zeros(full_shape, dtype=torch.float32, device='cpu'),
                        'count_accum': torch.zeros(full_shape, dtype=torch.float32, device='cpu'),
                        'sample_path': sample_path,
                        'pred_final': None
                    }

                # 将预测值累加到对应位置（处理重叠区域）
                if patch_coords is not None and patch_size is not None:
                    d_start, h_start, w_start = get_triplet(
                        patch_coords, sample_idx, batch_size, 'patch_coords'
                    )
                    pd, ph, pw = get_triplet(
                        patch_size, sample_idx, batch_size, 'patch_size'
                    )

                    d_end = d_start + pd
                    h_end = h_start + ph
                    w_end = w_start + pw

                    # 累加预测值和计数
                    recon_data = model_reconstructions[model_name]
                    pred_patch = pred_cpu[sample_idx:sample_idx + 1]
                    recon_data['pred_accum'][..., d_start:d_end, h_start:h_end, w_start:w_end] += pred_patch
                    recon_data['count_accum'][..., d_start:d_end, h_start:h_end, w_start:w_end] += 1

                    model_patch_counts[model_name] = model_patch_counts.get(model_name, 0) + 1
                    expected_count = expected_patch_counts.get(model_name, 0)
                    if (
                        expected_count > 0
                        and model_patch_counts[model_name] == expected_count
                        and model_name not in finalized_models
                    ):
                        finalize_model_prediction(model_name)
                        sync_device()
                        model_test_times[model_name] = time.perf_counter() - model_start_times[model_name]
                else:
                    print(f"Warning: No patch coordinates for model {model_name}, skipping volume reconstruction")

    # 处理尚未完成拼接的模型（fallback）
    for model_name in model_reconstructions.keys():
        if model_name not in finalized_models:
            finalize_model_prediction(model_name)
            if model_name in model_start_times:
                sync_device()
                model_test_times[model_name] = time.perf_counter() - model_start_times[model_name]
        expected_count = expected_patch_counts.get(model_name, 0)
        actual_count = model_patch_counts.get(model_name, 0)
        if expected_count > 0 and actual_count != expected_count:
            print(
                f"Warning: model {model_name} patch count mismatch "
                f"(actual={actual_count}, expected={expected_count})"
            )
    
    # 计算每个模型的完整体积指标
    volume_metrics = {}
    
    for model_name, recon_data in model_reconstructions.items():
        print(f"\nProcessing model: {model_name}")

        pred_final = recon_data.get('pred_final', None)
        if pred_final is None:
            print(f"Warning: No stitched volume for model {model_name}")
            continue
        
        # 加载完整真实体积
        true_volume = dataset.load_full_volume(recon_data['sample_path'])
        if true_volume is None:
            print(f"Warning: Cannot load full volume for model {model_name}")
            continue
        
        # 保留原来的两个 unsqueeze：当前标签为 (1,1,1,D,H,W)，预测为 (1,1,D,H,W)。
        # 后续分别 flatten 后逐元素比较；这里不改变张量处理。
        true_volume = true_volume.unsqueeze(0).unsqueeze(0)
        
        # 转移到相同设备
        pred_final = pred_final.to(device)
        true_volume = true_volume.to(device)

        pred_real = torch.sinh(pred_final) * stress_median
        true_real = torch.sinh(true_volume) * stress_median
        
        # 计算完整体积指标
        pred_flat = pred_real.flatten()
        true_flat = true_real.flatten()
        pred_flat_asinh = pred_final.flatten()
        true_flat_asinh = true_volume.flatten()
        
        # 计算R²
        mean_true = true_flat.mean()
        ss_tot = ((true_flat - mean_true) ** 2).sum()
        ss_res = ((pred_flat - true_flat) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # 计算NRMSE（总绝对误差 / 真值总和）
        total_abs_error_real = torch.abs(pred_flat - true_flat).sum()
        total_true_real = true_flat.sum()
        nrmse_real = total_abs_error_real / (total_true_real + 1e-8)

        total_abs_error_asinh = torch.abs(pred_flat_asinh - true_flat_asinh).sum()
        total_true_asinh = true_flat_asinh.sum()
        nrmse_asinh = total_abs_error_asinh / (total_true_asinh + 1e-8)
        
        # 存储指标
        volume_metrics[model_name] = {
            'r2': r2.item(),
            'nrmse_real': nrmse_real.item(),
            'nrmse_asinh': nrmse_asinh.item(),
            'test_time_sec': float(model_test_times.get(model_name, 0.0))
        }
        
        print(f"  Volume Metrics:")
        print(f"    R²: {r2.item():.6f}")
        print(f"    NRMSE (real): {nrmse_real.item():.6f}")
        print(f"    NRMSE (asinh): {nrmse_asinh.item():.6f}")
        print(f"    Test Time (patch->stitch): {volume_metrics[model_name]['test_time_sec']:.4f}s")
        
        # 生成散点图
        if plot_scatter and MATPLOTLIB_AVAILABLE:
            print(f"  Generating scatter plot for {model_name}...")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 将数据转移到CPU并转换为numpy数组
            pred_np = pred_real.cpu().numpy().flatten()
            true_np = true_real.cpu().numpy().flatten()
            pred_np_asinh = pred_final.cpu().numpy().flatten()
            true_np_asinh = true_volume.cpu().numpy().flatten()
            
            # 计算需要采样的点数
            total_points = len(pred_np)
            min_sample_points = int(total_points * sample_ratio)
            target_points = min(min_sample_points, max_points)
            
            # 将真值排序后分为 20 段，各段按固定步长取点，拼接后截断到目标点数。
            n_bins = 20
            sorted_indices = np.argsort(true_np)
            bin_size = len(sorted_indices) // n_bins
            selected_indices = []
            
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size if i < n_bins - 1 else len(sorted_indices)
                bin_indices = sorted_indices[start:end]
                points_per_bin = max(1, len(bin_indices) // (max(1, target_points // n_bins)))
                selected_indices.extend(bin_indices[::points_per_bin])
            
            selected_indices = np.array(selected_indices[:target_points])
            pred_sample = pred_np[selected_indices]
            true_sample = true_np[selected_indices]
            print(f"  Sampled {len(pred_sample)} points from {total_points} total points (uniform across {n_bins} bins)")
            
            # 创建散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(true_sample, pred_sample, alpha=0.5, s=1, edgecolors='none')
            
            # 添加y=x参考线
            min_val = min(true_sample.min(), pred_sample.min())
            max_val = max(true_sample.max(), pred_sample.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x (Perfect Prediction)')
            
            # 设置图形属性
            plt.xlabel('True von Mises Stress', fontsize=12)
            plt.ylabel('Predicted von Mises Stress', fontsize=12)
            plt.title(
                f'Scatter Plot: {model_name}\n'
                f'R² = {r2.item():.4f}, NRMSE(real) = {nrmse_real.item():.4f}, NRMSE(asinh) = {nrmse_asinh.item():.4f}',
                fontsize=14
            )
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 添加文本标注
            plt.text(0.05, 0.95, f'Total points: {total_points:,}\nSampled points: {len(pred_sample):,}', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 确保坐标轴比例相同
            plt.axis('equal')
            plt.tight_layout()
            
            # 保存图像
            plot_path = os.path.join(output_dir, f'{model_name}_scatter.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved scatter plot to {plot_path}")
            
            # 计算十区间NRMSE
            print(f"  Computing bin-wise NRMSE (10 bins)...")
            n_bins_nrmse = 10
            bin_edges = np.linspace(true_np.min(), true_np.max(), n_bins_nrmse + 1)
            bin_nrmse = {}
            bin_counts = {}
            
            for i in range(n_bins_nrmse):
                if i < n_bins_nrmse - 1:
                    mask = (true_np >= bin_edges[i]) & (true_np < bin_edges[i+1])
                else:
                    mask = (true_np >= bin_edges[i]) & (true_np <= bin_edges[i+1])
                
                if mask.sum() > 0:
                    pred_bin = pred_np[mask]
                    true_bin = true_np[mask]
                    pred_bin_asinh = pred_np_asinh[mask]
                    true_bin_asinh = true_np_asinh[mask]

                    nrmse_bin_real = np.abs(pred_bin - true_bin).sum() / (true_bin.sum() + 1e-8)
                    nrmse_bin_asinh = np.abs(pred_bin_asinh - true_bin_asinh).sum() / (true_bin_asinh.sum() + 1e-8)
                    bin_key = f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}'

                    bin_nrmse[bin_key] = {
                        'nrmse_real': float(nrmse_bin_real),
                        'nrmse_asinh': float(nrmse_bin_asinh)
                    }
                    bin_counts[bin_key] = int(mask.sum())
            
            print(f"  Bin-wise NRMSE (10 bins):")
            for bin_range, nrmse_val in bin_nrmse.items():
                count = bin_counts[bin_range]
                print(
                    f"    {bin_range}: "
                    f"NRMSE(real)={nrmse_val['nrmse_real']:.6f}, "
                    f"NRMSE(asinh)={nrmse_val['nrmse_asinh']:.6f} (n={count})"
                )
            
            # 保存区间NRMSE到文件
            bin_nrmse_path = os.path.join(output_dir, f'{model_name}_bin_nrmse.json')
            with open(bin_nrmse_path, 'w') as f:
                json.dump({
                    'bin_nrmse': bin_nrmse,
                    'bin_counts': bin_counts,
                    'bin_edges': [float(x) for x in bin_edges]
                }, f, indent=2)
            print(f"  Saved bin NRMSE to {bin_nrmse_path}")
            
        elif plot_scatter and not MATPLOTLIB_AVAILABLE:
            print(f"  Warning: matplotlib not available, skipping scatter plot for {model_name}")
    
    # 转换为与旧版本兼容的格式
    if volume_metrics:
        # 提取所有模型的指标值
        r2_values = [m['r2'] for m in volume_metrics.values()]
        nrmse_real_values = [m['nrmse_real'] for m in volume_metrics.values()]
        nrmse_asinh_values = [m['nrmse_asinh'] for m in volume_metrics.values()]
        test_time_values = [m['test_time_sec'] for m in volume_metrics.values()]
        
        avg_metrics = {
            'r2': {
                'mean': float(np.mean(r2_values)),
                'std': float(np.std(r2_values)),
                'min': float(np.min(r2_values)),
                'max': float(np.max(r2_values))
            },
            'nrmse_real': {
                'mean': float(np.mean(nrmse_real_values)),
                'std': float(np.std(nrmse_real_values)),
                'min': float(np.min(nrmse_real_values)),
                'max': float(np.max(nrmse_real_values))
            },
            'nrmse_asinh': {
                'mean': float(np.mean(nrmse_asinh_values)),
                'std': float(np.std(nrmse_asinh_values)),
                'min': float(np.min(nrmse_asinh_values)),
                'max': float(np.max(nrmse_asinh_values))
            },
            'test_time_sec': {
                'mean': float(np.mean(test_time_values)),
                'std': float(np.std(test_time_values)),
                'min': float(np.min(test_time_values)),
                'max': float(np.max(test_time_values))
            }
        }
        
        model_avg_metrics = {}
        for model_name, metrics in volume_metrics.items():
            model_avg_metrics[model_name] = metrics
        
        return avg_metrics, model_avg_metrics
    else:
        print("Warning: No volume metrics calculated")
        return {}, {}


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    data_dir = args.data_dir if args.data_dir else config.DATA_DIR
    print(f"Using data directory: {data_dir}")
    from .data_policy import validate_splits, validate_stats, stats_path
    split_manifest = validate_splits(config, test_dir=data_dir)
    stats_hash = validate_stats(config, split_manifest)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_dataset = StressDataset(
        data_dir, 
        split='test',
        stats_path=str(stats_path(config)), 
        normalize=True,
        patch_size=(config.PATCH_D, config.PATCH_H, config.PATCH_W),
        stride=(config.STRIDE_D, config.STRIDE_H, config.STRIDE_W),
        use_sliding_window=True,
        use_prepatched=True,
        patch_dir_name=config.TRAIN_PATCH_DIR_NAME,
        asinh_percentile=config.ASINH_PERCENTILE,
        use_cache=True,
        include_stress=False
    )

    pin_memory = device.type == 'cuda'
    prefetch_factor = 2 if args.num_workers > 0 else None
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=prefetch_factor
    )

    print(
        f"DataLoader config: batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, pin_memory={pin_memory}"
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    model = SegMamba(
        in_chans=config.IN_CHANS,
        out_chans=config.OUT_CHANS,
        depths=config.DEPTHS,
        feat_size=config.FEAT_SIZE,
        hidden_size=config.HIDDEN_SIZE
    ).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get('transform_stats_sha256') and checkpoint['transform_stats_sha256'] != stats_hash:
        raise ValueError('Checkpoint and configured training statistics differ')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    stress_percentiles, stress_scale_percentile = load_stress_stats(config.STATS_DATA_DIR, (config.ASINH_PERCENTILE,))
    validate_checkpoint_transform(checkpoint, transform_settings(config, stress_percentiles))
    stress_median = stress_percentiles[float(config.ASINH_PERCENTILE)]

    avg_metrics, model_avg_metrics = test_model(
        model, test_loader, device,
        stress_median,
        output_dir=args.output_dir,
        plot_scatter=args.plot_scatter,
        sample_ratio=args.sample_ratio,
        max_points=args.max_points
    )
    
    print("\n" + "=" * 60)
    print("Overall Metrics:")
    print("=" * 60)
    for metric_name, metric_values in avg_metrics.items():
        print(f"{metric_name}:")
        print(f"  Mean: {metric_values['mean']:.6f}")
        print(f"  Std: {metric_values['std']:.6f}")
        print(f"  Min: {metric_values['min']:.6f}")
        print(f"  Max: {metric_values['max']:.6f}")
    
    results = {
        'overall_metrics': avg_metrics,
        'per_model_metrics': model_avg_metrics
    }
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
