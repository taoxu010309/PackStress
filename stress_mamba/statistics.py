#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and save transform stats for dataset normalization.
Reads 15-channel full input volumes and evenly spaced flattened stress samples from supplied models.
The public entry point supplies training models only; the numerical routine is retained.
"""

import argparse
import json
import os

import numpy as np
import zarr


def compute_transform_stats(model_paths, stress_percentile=50.0, max_samples_per_model=100, extra_percentiles=()):
    """Fit input statistics and sampled stress quantiles on the supplied paths.

    Stress sampling uses a fixed flattened stride, not random or exhaustive sampling;
    max_samples_per_model is an approximate target, not a strict upper bound.
    stress_median is a legacy key for the requested scale percentile;
    stress_p50_value stores the actual sampled median.
    """
    from .transforms import percentile_key
    extra_keys = [(percentile_key(p), float(p)) for p in extra_percentiles]
    idx_x = 0
    idx_y = 1
    idx_z = 2
    idx_t = 3
    idx_mat_params = list(range(4, 10))
    idx_thickness = 10
    idx_grad_x = 13
    idx_grad_y = 14

    z_max = 0.0
    t_max = 0.0
    thickness_max = 0.0
    mat_params_max = np.zeros(6, dtype=np.float64)

    grad_sum = np.zeros(2, dtype=np.float64)
    grad_sum_sq = np.zeros(2, dtype=np.float64)
    grad_count = np.zeros(2, dtype=np.float64)
    grad_raw_min = np.zeros(2, dtype=np.float64) + np.inf
    grad_raw_max = np.zeros(2, dtype=np.float64) - np.inf

    stress_samples = []
    stress_percentile = float(np.clip(stress_percentile, 0.0, 100.0))

    for model_idx, model_path in enumerate(model_paths):
        input_path = os.path.join(model_path, 'input_features.zarr')
        if not os.path.exists(input_path):
            print(f"Warning: missing input_features.zarr in {model_path}")
            continue

        input_zarr = zarr.open(input_path, mode='r')
        
        if len(input_zarr.shape) != 4 or input_zarr.shape[0] != 15:
            raise ValueError(
                f'Expected input shape (15, D, H, W), got {input_zarr.shape} in {model_path}. '
                'Prepare exactly 15 channels in the order documented in README.md.'
            )

        data = input_zarr[:].astype(np.float32)
        
        z_max = max(z_max, float(np.nanmax(data[idx_z])))
        t_max = max(t_max, float(np.nanmax(data[idx_t])))
        thickness_max = max(thickness_max, float(np.nanmax(data[idx_thickness])))

        for i, c in enumerate(idx_mat_params):
            mat_params_max[i] = max(mat_params_max[i], float(np.nanmax(data[c])))

        for g_idx, c in enumerate([idx_grad_x, idx_grad_y]):
            g = data[c]
            mask = ~np.isnan(g)
            if np.any(mask):
                vals = g[mask]
                grad_sum[g_idx] += vals.sum()
                grad_sum_sq[g_idx] += (vals ** 2).sum()
                grad_count[g_idx] += mask.sum()
                grad_raw_min[g_idx] = min(grad_raw_min[g_idx], float(vals.min()))
                grad_raw_max[g_idx] = max(grad_raw_max[g_idx], float(vals.max()))

        del data

        stress_path = os.path.join(model_path, 'vonmises.zarr')
        if os.path.exists(stress_path):
            try:
                stress_zarr = zarr.open(stress_path, mode='r')
                stress_data = stress_zarr[:].astype(np.float32)
                stress_data = np.nan_to_num(stress_data, nan=0.0)
                stress_flat = stress_data.reshape(-1)
                
                step = max(1, len(stress_flat) // max_samples_per_model)
                stress_samples.append(stress_flat[::step])
                
                del stress_data
            except Exception as e:
                print(f"Warning: failed to load stress data from {stress_path}: {e}")

        if (model_idx + 1) % 5 == 0:
            print(f"  Processed {model_idx + 1}/{len(model_paths)} models...")

    grad_mean = np.zeros(2, dtype=np.float64)
    grad_std = np.ones(2, dtype=np.float64)
    for i in range(2):
        if grad_count[i] > 0:
            grad_mean[i] = grad_sum[i] / grad_count[i]
            grad_var = grad_sum_sq[i] / grad_count[i] - grad_mean[i] ** 2
            grad_std[i] = np.sqrt(max(grad_var, 1e-8))

    grad_std_max = np.zeros(2, dtype=np.float64)
    for i in range(2):
        if grad_count[i] > 0:
            max_dist = max(
                abs(grad_raw_max[i] - grad_mean[i]),
                abs(grad_raw_min[i] - grad_mean[i])
            )
            grad_std_max[i] = max_dist / (grad_std[i] + 1e-8)
        else:
            grad_std_max[i] = 1.0

    if z_max <= 0:
        z_max = 1.0
    if t_max <= 0:
        t_max = 1.0
    if thickness_max <= 0:
        thickness_max = 1.0
    mat_params_max = np.where(mat_params_max <= 0, 1.0, mat_params_max)
    grad_std_max = np.where(grad_std_max <= 0, 1.0, grad_std_max)

    if stress_samples:
        stress_concat = np.concatenate(stress_samples, axis=0)
        stress_median = float(np.median(stress_concat))
        stress_scale = float(np.percentile(stress_concat, stress_percentile))
        stress_p50 = float(np.percentile(stress_concat, 50.0))
        stress_p60 = float(np.percentile(stress_concat, 60.0))
        stress_p65 = float(np.percentile(stress_concat, 65.0))
        stress_p70 = float(np.percentile(stress_concat, 70.0))
        stress_p80 = float(np.percentile(stress_concat, 80.0))
        stress_p90 = float(np.percentile(stress_concat, 90.0))
        stress_p95 = float(np.percentile(stress_concat, 95.0))
    else:
        stress_median = 1.0
        stress_scale = 1.0
        stress_p50 = 1.0
        stress_p60 = 1.0
        stress_p65 = 1.0
        stress_p70 = 1.0
        stress_p80 = 1.0
        stress_p90 = 1.0
        stress_p95 = 1.0

    if stress_scale <= 0:
        stress_scale = 1.0
    if stress_p50 <= 0:
        stress_p50 = stress_scale
    if stress_p60 <= 0:
        stress_p60 = stress_scale
    if stress_p65 <= 0:
        stress_p65 = stress_scale
    if stress_p70 <= 0:
        stress_p70 = stress_scale
    if stress_p80 <= 0:
        stress_p80 = stress_scale
    if stress_p90 <= 0:
        stress_p90 = stress_scale
    if stress_p95 <= 0:
        stress_p95 = stress_scale

    stats = {
        'version': 1,
        'z_max': float(z_max),
        't_max': float(t_max),
        'material_params_max': [float(x) for x in mat_params_max.tolist()],
        'thickness_max': float(thickness_max),
        'grad_mean': [float(x) for x in grad_mean.tolist()],
        'grad_std': [float(x) for x in grad_std.tolist()],
        'grad_std_max': [float(x) for x in grad_std_max.tolist()],
        'stress_median': float(stress_scale),
        'stress_scale': float(stress_scale),
        'stress_scale_percentile': float(stress_percentile),
        'stress_p50': float(stress_p50),
        'stress_p60': float(stress_p60),
        'stress_p65': float(stress_p65),
        'stress_p70': float(stress_p70),
        'stress_p80': float(stress_p80),
        'stress_p90': float(stress_p90),
        'stress_p95': float(stress_p95),
        'stress_p50_value': float(stress_median),
        'input_order': [
            'x', 'y', 'z', 'T',
            'material_params(6)', 'thickness', 'interface_mask',
            'interface_distance', 'temp_grad_x', 'temp_grad_y'
        ]
    }

    # Reuse the same sampled stresses and the same percentile formula; no refitting on test data.
    for key, percentile in extra_keys:
        if key not in stats:
            value = float(np.percentile(stress_concat, percentile)) if stress_samples else 1.0
            stats[key] = value if value > 0 else stress_scale
    return stats


def parse_args():
    from .config import config
    parser = argparse.ArgumentParser(description='Fit transform statistics on the configured training models only')
    parser.add_argument('--stress_percentile', type=float, default=config.ASINH_PERCENTILE)
    parser.add_argument('--stress_samples_per_model', type=int, default=100)
    return parser.parse_args()


def main():
    from .config import config
    from .data_policy import validate_splits, stats_path
    args = parse_args()
    splits = validate_splits(config)
    model_paths = [model['path'] for model in splits['train']]
    for model_path in model_paths:
        if not os.path.isdir(os.path.join(model_path, 'vonmises.zarr')):
            raise FileNotFoundError(f'Missing training labels: {model_path}')
    print(f'Fitting statistics on training models only: {[m["name"] for m in splits["train"]]}')
    stats = compute_transform_stats(model_paths, stress_percentile=args.stress_percentile,
                                    max_samples_per_model=max(1, args.stress_samples_per_model),
                                    extra_percentiles=(getattr(config, "ASINH_PERCENTILE", 70),
                                                       getattr(config, "LABEL_WEIGHT_PERCENTILE", 70)))
    stats['fit_provenance'] = {'split':'train',
                               'design_ids':sorted(m['design_id'] for m in splits['train']),
                               'model_names':sorted(m['name'] for m in splits['train'])}
    output_path = stats_path(config)
    output_path.write_text(json.dumps(stats, indent=2)+'\n')
    print(f'Saved {output_path}; validation and test reuse this file without refitting')


if __name__ == '__main__':
    main()
