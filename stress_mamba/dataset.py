#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块
针对16维输入、1维von Mises应力输出数据集
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from .transforms import stress_scale


class StressDataset(Dataset):
    """
    热应力数据集
    原始输入：16 通道；normalize=True 时跳过原始索引 4，返回 15 通道。
    目标：1 通道 von Mises 应力，经非负裁剪和 asinh 变换。
    公共入口显式传入训练集统计文件，验证/测试不重新拟合统计量。
    """
    def __init__(self, data_dir, split='train', transform=None, normalize=True,
                 patch_size=(20, 256, 256), stride=(10, 128, 128), use_sliding_window=True,
                 stats_path=None, use_prepatched=True,
                 patch_dir_name='patches_40x512x512_overlap50', use_cache=True,
                 val_model_indices=None, asinh_percentile=70,
                 max_models=None, include_stress=True):
        """
        Args:
            data_dir: 已按 model 划分的单个集合目录，直接包含 model_* 子目录
            split: 集合标签（'train'、'val' 或 'test'），不改变 data_dir
            transform: 保留的接口参数；当前 __getitem__ 不调用它
            normalize: 是否归一化
            patch_size: 补丁大小 (D, H, W)
            use_sliding_window: 缺少预切 patch 时是否按 stride 切片
            use_prepatched: 优先使用准备好的 patch 文件及其坐标
            include_stress: 是否返回变换后的标签；需要启用 normalize 加载统计量
            val_model_indices: 固定的验证模型索引列表，例如 [0, 1] 表示前2个模型作为验证集
            asinh_percentile: asinh变换使用的分位数
            max_models: 限制使用的最大model数量（用于debug模式）
        """
        data_dir = os.path.realpath(os.path.expanduser(data_dir))
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.patch_size = patch_size
        self.stride = stride
        self.use_sliding_window = use_sliding_window
        self.stats_path = stats_path
        self.use_prepatched = use_prepatched
        self.patch_dir_name = patch_dir_name
        self.use_cache = use_cache
        self.val_model_indices = val_model_indices
        self.asinh_percentile = asinh_percentile
        self.max_models = max_models
        self.include_stress = include_stress
        
        # Read exactly the configured subset root, as used by the model-level
        # preflight and training-statistics code; never descend into nested splits.
        self.split_dir = data_dir
        
        self.model_dirs = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d)) and d.startswith('model_')
        ])
        
        if not self.model_dirs:
            model_name = os.path.basename(data_dir)
            if model_name.startswith('model_') and os.path.exists(os.path.join(data_dir, 'input_features.zarr')):
                self.split_dir = os.path.dirname(data_dir)
                self.model_dirs = [model_name]

        if not self.model_dirs:
            raise ValueError(f'No model_* directories in {data_dir}; specify the prepared subset directory directly')
        
        self.samples = []
        self.sample_patches = []
        self.zarr_cache = {}
        self.patch_index = []
        self.patch_metadata = {}
        self.patch_dir_map = {}
        
        for model_dir in self.model_dirs:
            model_path = os.path.join(self.split_dir, model_dir)
            self.samples.append(model_path)

        if self.max_models is not None and len(self.samples) > self.max_models:
            self.samples = self.samples[:self.max_models]

        if self.val_model_indices is not None and len(self.samples) > 0:
            if self.split == 'train':
                self.samples = [s for i, s in enumerate(self.samples) if i not in self.val_model_indices]
            elif self.split == 'val':
                self.samples = [s for i, s in enumerate(self.samples) if i in self.val_model_indices]



        if self.use_prepatched:
            self._init_prepatched_index()
        else:
            for model_path in self.samples:
                if self.use_sliding_window:
                    patches = self._generate_sliding_window_patches(model_path)
                    self.sample_patches.append(patches)
        
        self.transform_stats = None
        if self.normalize:
            self._load_transform_stats()

    def _build_zarr_cache(self):
        for sample_path in self.samples:
            if sample_path in self.zarr_cache:
                continue
            try:
                cache_entry = {}
                if self.use_prepatched and sample_path in self.patch_dir_map:
                    patch_dir = self.patch_dir_map[sample_path]
                    cache_entry['input_patches'] = zarr.open(
                        os.path.join(patch_dir, 'input_patches.zarr'), mode='r'
                    )
                    if self.include_stress:
                        cache_entry['stress_patches'] = zarr.open(
                            os.path.join(patch_dir, 'vonmises_patches.zarr'), mode='r'
                        )
                else:
                    cache_entry['input'] = zarr.open(
                        os.path.join(sample_path, 'input_features.zarr'), mode='r'
                    )
                    if self.include_stress:
                        cache_entry['stress'] = zarr.open(
                            os.path.join(sample_path, 'vonmises.zarr'), mode='r'
                        )
                self.zarr_cache[sample_path] = cache_entry
            except Exception as e:
                print(f"Error caching zarr for {sample_path}: {e}")

    def _get_cache_entry(self, sample_path):
        if not self.use_cache:
            return {}

        if sample_path in self.zarr_cache:
            return self.zarr_cache[sample_path]

        try:
            cache_entry = {}
            if self.use_prepatched and sample_path in self.patch_dir_map:
                patch_dir = self.patch_dir_map[sample_path]
                cache_entry['input_patches'] = zarr.open(
                    os.path.join(patch_dir, 'input_patches.zarr'), mode='r'
                )
                if self.include_stress:
                    cache_entry['stress_patches'] = zarr.open(
                        os.path.join(patch_dir, 'vonmises_patches.zarr'), mode='r'
                    )
            else:
                cache_entry['input'] = zarr.open(
                    os.path.join(sample_path, 'input_features.zarr'), mode='r'
                )
                if self.include_stress:
                    cache_entry['stress'] = zarr.open(
                        os.path.join(sample_path, 'vonmises.zarr'), mode='r'
                    )
            self.zarr_cache[sample_path] = cache_entry
            return cache_entry
        except Exception as e:
            print(f"Error opening zarr for {sample_path}: {e}")
            return {}

    def _init_prepatched_index(self):
        all_have_patches = True
        for model_path in self.samples:
            patch_dir = os.path.join(model_path, self.patch_dir_name)
            input_patch_path = os.path.join(patch_dir, 'input_patches.zarr')
            stress_patch_path = os.path.join(patch_dir, 'vonmises_patches.zarr')
            meta_path = os.path.join(patch_dir, 'patches_metadata.json')

            if not (os.path.isdir(patch_dir) and os.path.exists(input_patch_path)
                    and os.path.exists(stress_patch_path) and os.path.exists(meta_path)):
                all_have_patches = False
                break

        if not all_have_patches:
            self.use_prepatched = False
            for model_path in self.samples:
                if self.use_sliding_window:
                    patches = self._generate_sliding_window_patches(model_path)
                    self.sample_patches.append(patches)
            return

        for model_path in self.samples:
            patch_dir = os.path.join(model_path, self.patch_dir_name)
            meta_path = os.path.join(patch_dir, 'patches_metadata.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            num_patches = metadata['num_patches']
            d_starts = metadata['d_starts']
            h_starts = metadata['h_starts']
            w_starts = metadata['w_starts']
            patch_shape = metadata['patch_shape_dhw']

            self.patch_metadata[model_path] = {
                'num_patches': num_patches,
                'd_starts': d_starts,
                'h_starts': h_starts,
                'w_starts': w_starts,
                'patch_shape': patch_shape,
            }
            self.patch_dir_map[model_path] = patch_dir

            for patch_idx in range(num_patches):
                self.patch_index.append((model_path, patch_idx))
        
    def _load_transform_stats(self):
        if self.stats_path is None:
            self.stats_path = os.path.join(self.data_dir, 'transform_stats.json')

        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(
                f"Transform stats not found at {self.stats_path}. "
                "Please run compute_transform_stats.py first."
            )

        with open(self.stats_path, 'r', encoding='utf-8') as f:
            self.transform_stats = json.load(f)

        self.z_max = float(self.transform_stats['z_max'])
        self.t_max = float(self.transform_stats['t_max'])
        self.thickness_max = float(self.transform_stats['thickness_max'])
        self.mat_params_max = np.array(self.transform_stats['material_params_max'], dtype=np.float32)
        self.grad_mean = np.array(self.transform_stats['grad_mean'], dtype=np.float32)
        self.grad_std = np.array(self.transform_stats['grad_std'], dtype=np.float32)

        if 'grad_std_max' in self.transform_stats:
            self.grad_std_max = np.array(self.transform_stats['grad_std_max'], dtype=np.float32)
        else:
            self.grad_std_max = np.array(self.transform_stats['grad_std_max_abs'], dtype=np.float32)

        self.stress_median = stress_scale(self.transform_stats, self.asinh_percentile)

    def _generate_sliding_window_patches(self, model_path):
        """生成sliding window patches列表"""
        try:
            input_zarr = zarr.open(os.path.join(model_path, 'input_features.zarr'))
            D, H, W = input_zarr.shape[1], input_zarr.shape[2], input_zarr.shape[3]
            
            patches = []
            d_stride, h_stride, w_stride = self.stride
            d_patch, h_patch, w_patch = self.patch_size
            
            for d_start in range(0, max(1, D - d_patch + 1), d_stride):
                for h_start in range(0, max(1, H - h_patch + 1), h_stride):
                    for w_start in range(0, max(1, W - w_patch + 1), w_stride):
                        patches.append({
                            'd_start': d_start,
                            'h_start': h_start,
                            'w_start': w_start
                        })
            return patches
        except Exception as e:
            print(f"Error generating patches for {model_path}: {e}")
            return []
    
    def __len__(self):
        if self.use_prepatched:
            return len(self.patch_index)
        if self.use_sliding_window:
            return sum(len(p) for p in self.sample_patches)
        return len(self.samples)
    
    def _get_random_patch_coords(self, data_shape):
        """随机生成patch坐标"""
        D, H, W = data_shape
        pd, ph, pw = self.patch_size
        
        d = np.random.randint(0, max(1, D - pd + 1))
        h = np.random.randint(0, max(1, H - ph + 1))
        w = np.random.randint(0, max(1, W - pw + 1))
        
        return d, h, w
    
    def __getitem__(self, idx):
        if self.use_prepatched:
            sample_path, patch_idx = self.patch_index[idx]
            patch_info = None
        elif self.use_sliding_window:
            cumsum = 0
            for sample_idx, patches in enumerate(self.sample_patches):
                if idx < cumsum + len(patches):
                    patch_idx = idx - cumsum
                    sample_path = self.samples[sample_idx]
                    patch_info = patches[patch_idx]
                    break
                cumsum += len(patches)
        else:
            sample_idx = idx
            sample_path = self.samples[sample_idx]
        
        cache_entry = self._get_cache_entry(sample_path)

        if self.use_prepatched:
            if 'input_patches' in cache_entry:
                input_zarr = cache_entry['input_patches']
                stress_zarr = cache_entry['stress_patches'] if self.include_stress else None
            else:
                patch_dir = os.path.join(sample_path, self.patch_dir_name)
                input_zarr = zarr.open(os.path.join(patch_dir, 'input_patches.zarr'), mode='r')
                stress_zarr = zarr.open(os.path.join(patch_dir, 'vonmises_patches.zarr'), mode='r') if self.include_stress else None
        else:
            if 'input' in cache_entry:
                input_zarr = cache_entry['input']
                stress_zarr = cache_entry['stress'] if self.include_stress else None
            else:
                input_zarr = zarr.open(os.path.join(sample_path, 'input_features.zarr'), mode='r')
                stress_zarr = zarr.open(os.path.join(sample_path, 'vonmises.zarr'), mode='r') if self.include_stress else None

        if self.use_prepatched:
            input_data = input_zarr[patch_idx].astype(np.float32)
            if self.include_stress:
                stress_data = stress_zarr[patch_idx].astype(np.float32)
        elif self.use_sliding_window:
            d, h, w = patch_info['d_start'], patch_info['h_start'], patch_info['w_start']
            pd, ph, pw = self.patch_size
            input_data = input_zarr[:, d:d+pd, h:h+ph, w:w+pw].astype(np.float32)
            if self.include_stress:
                stress_data = stress_zarr[:, d:d+pd, h:h+ph, w:w+pw].astype(np.float32)
        else:
            input_data = input_zarr[:].astype(np.float32)
            if self.include_stress:
                stress_data = stress_zarr[:].astype(np.float32)

        input_data = np.nan_to_num(input_data, nan=0.0)
        if self.include_stress:
            stress_data = np.nan_to_num(stress_data, nan=0.0)

        if self.normalize and self.transform_stats is not None:
            idx_x = 0
            idx_y = 1
            idx_z = 2
            idx_t = 3
            idx_mat_params = list(range(5, 11))
            idx_thickness = 11
            idx_interface_mask = 12
            idx_interface_distance = 13
            idx_grad_x = 14
            idx_grad_y = 15

            x = input_data[idx_x]
            y = input_data[idx_y]
            z = input_data[idx_z] / (self.z_max + 1e-8)
            t = input_data[idx_t] / (self.t_max + 1e-8)

            mat_params = input_data[idx_mat_params] / (self.mat_params_max[:, None, None, None] + 1e-8)
            thickness = input_data[idx_thickness] / (self.thickness_max + 1e-8)
            interface_mask = input_data[idx_interface_mask]
            # Raw channel 13 already stores D_int = exp(-d), where d is the
            # nearest material-interface distance. Pass it through unchanged;
            # do not apply exp(-d) a second time (model input channel 12).
            interface_distance = input_data[idx_interface_distance]

            grad_x = (input_data[idx_grad_x] - self.grad_mean[0]) / (self.grad_std[0] + 1e-8)
            grad_y = (input_data[idx_grad_y] - self.grad_mean[1]) / (self.grad_std[1] + 1e-8)
            grad_x = grad_x / (self.grad_std_max[0] + 1e-8)
            grad_y = grad_y / (self.grad_std_max[1] + 1e-8)

            input_data = np.concatenate(
                [
                    x[None, ...],
                    y[None, ...],
                    z[None, ...],
                    t[None, ...],
                    mat_params,
                    thickness[None, ...],
                    interface_mask[None, ...],
                    interface_distance[None, ...],
                    grad_x[None, ...],
                    grad_y[None, ...],
                ],
                axis=0
            )

        if self.include_stress:
            stress_data = np.arcsinh(
                np.clip(stress_data, a_min=0.0, a_max=None) / (self.stress_median + 1e-8)
            )

        input_data = torch.from_numpy(input_data.astype(np.float32))
        if self.include_stress:
            stress_data = torch.from_numpy(stress_data.astype(np.float32))
        
        model_name = os.path.basename(sample_path)

        if self.use_prepatched:
            meta = self.patch_metadata[sample_path]
            d_starts = meta['d_starts']
            h_starts = meta['h_starts']
            w_starts = meta['w_starts']
            patch_shape = meta['patch_shape']
            hw_size = len(h_starts) * len(w_starts)
            d_idx = patch_idx // hw_size
            rem = patch_idx % hw_size
            h_idx = rem // len(w_starts)
            w_idx = rem % len(w_starts)
            d = d_starts[d_idx]
            h = h_starts[h_idx]
            w = w_starts[w_idx]
            pd, ph, pw = patch_shape
            output = {
                'input': input_data,
                'model_name': model_name,
                'patch_coords': (d, h, w),
                'patch_size': (pd, ph, pw)
            }
            if self.include_stress:
                output['stress'] = stress_data
            return output

        if self.use_sliding_window:
            output = {
                'input': input_data,
                'model_name': model_name,
                'patch_coords': (d, h, w),
                'patch_size': (pd, ph, pw)
            }
            if self.include_stress:
                output['stress'] = stress_data
            return output
        else:
            output = {
                'input': input_data,
                'model_name': model_name,
                'patch_coords': None,
                'patch_size': None
            }
            if self.include_stress:
                output['stress'] = stress_data
            return output

    def get_volume_shape(self, model_path):
        """
        获取原始体积形状
        
        Args:
            model_path: 模型路径
            
        Returns:
            tuple: 体积形状 (D, H, W)
        """
        try:
            input_zarr = zarr.open(os.path.join(model_path, 'input_features.zarr'))
            return input_zarr.shape[1:]  # 跳过通道维度，返回 (D, H, W)
        except Exception as e:
            print(f"Error getting volume shape for {model_path}: {e}")
            return None
    
    def load_full_volume(self, model_path, normalize=True):
        """
        直接加载完整真实体积
        
        Args:
            model_path: 模型路径
            normalize: 是否应用与数据集相同的归一化
            
        Returns:
            torch.Tensor: 完整体积张量，形状 (1, D, H, W)
        """
        try:
            stress_zarr = zarr.open(os.path.join(model_path, 'vonmises.zarr'))
            volume = stress_zarr[:].astype(np.float32)  # (1, D, H, W)
            
            if normalize:
                stress_median = 1.0
                if self.transform_stats is not None:
                    stress_median = stress_scale(self.transform_stats, self.asinh_percentile)
                if stress_median <= 0:
                    stress_median = 1.0
                volume = np.arcsinh(
                    np.clip(volume, a_min=0.0, a_max=None) / (stress_median + 1e-8)
                )
            
            return torch.from_numpy(volume)
        except Exception as e:
            print(f"Error loading full volume for {model_path}: {e}")
            return None


def get_dataloaders(data_dir, batch_size=1, num_workers=4, 
                    patch_size=(20, 256, 256), stride=(10, 128, 128),
                    use_sliding_window=True, split_train_test=False,
                    use_prepatched=True, patch_dir_name='patches_40x512x512_overlap50',
                    use_cache=False, rank=0, world_size=1, num_val_models=0,
                    val_model_indices=None,
                    train_data_dir=None, val_data_dir=None,
                    train_patch_dir_name=None, val_patch_dir_name=None,
                    asinh_percentile=70,
                    max_train_models=None, stats_path=None):
    """
    获取训练和验证数据加载器。
    公共入口先检查 model 隔离，再传入三个独立目录中的训练/验证目录，
    split_train_test=False、num_val_models=0、val_model_indices=None。
    其他参数分支为保留的旧接口，不执行公共入口的数据隔离检查。
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        patch_size: 补丁大小 (D, H, W)
        stride: 步长
        use_sliding_window: 是否使用sliding window
        split_train_test: 旧接口，读取 data_dir/train 和 data_dir/test 已分好的集合，不创建划分
        rank: 分布式训练中的进程rank
        world_size: 分布式训练中的进程数
        num_val_models: 固定使用前N个模型作为验证集
        val_model_indices: 指定的验证模型索引列表
        train_data_dir: 训练数据目录
        val_data_dir: 验证数据目录
        train_patch_dir_name: 训练patch目录名称
        val_patch_dir_name: 验证patch目录名称
        asinh_percentile: asinh变换使用的分位数
        max_train_models: 限制训练使用的最大model数量（用于debug模式）
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    train_data_dir = train_data_dir or data_dir
    val_data_dir = val_data_dir or data_dir
    train_patch_dir_name = train_patch_dir_name or patch_dir_name
    val_patch_dir_name = val_patch_dir_name or patch_dir_name
    
    # 如果传入val_model_indices则直接使用，否则按num_val_models逻辑处理
    if val_model_indices is None:
        if num_val_models > 0:
            val_model_indices = list(range(num_val_models))
        else:
            val_model_indices = None  # 各自目录内不再按索引筛选；公共入口使用独立目录
    
    if split_train_test:
        # Legacy caller explicitly requests the two already prepared subdirectories.
        # StressDataset itself never infers or switches between subset roots.
        legacy_stats_path = stats_path or os.path.join(data_dir, 'transform_stats.json')
        train_dataset = StressDataset(
            os.path.join(data_dir, 'train'), split='train', normalize=True,
            stats_path=legacy_stats_path,
            patch_size=patch_size, stride=stride,
            use_sliding_window=use_sliding_window,
            use_prepatched=use_prepatched,
            patch_dir_name=patch_dir_name,
            use_cache=use_cache
        )
        test_dataset = StressDataset(
            os.path.join(data_dir, 'test'), split='test', normalize=True,
            stats_path=legacy_stats_path,
            patch_size=patch_size, stride=stride,
            use_sliding_window=use_sliding_window,
            use_prepatched=use_prepatched,
            patch_dir_name=patch_dir_name,
            use_cache=use_cache
        )
        
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        ) if world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        return train_loader, test_loader
    
    train_dataset = StressDataset(
        train_data_dir, split='train', normalize=True, stats_path=stats_path,
        patch_size=patch_size, stride=stride,
        use_sliding_window=use_sliding_window,
        use_prepatched=use_prepatched,
        patch_dir_name=train_patch_dir_name,
        use_cache=use_cache,
        val_model_indices=val_model_indices,
        asinh_percentile=asinh_percentile,
        max_models=max_train_models
    )
    
    val_dataset = StressDataset(
        val_data_dir, split='val', normalize=True, stats_path=stats_path,
        patch_size=patch_size, stride=stride,
        use_sliding_window=use_sliding_window,
        use_prepatched=use_prepatched,
        patch_dir_name=val_patch_dir_name,
        use_cache=use_cache,
        val_model_indices=val_model_indices,
        asinh_percentile=asinh_percentile
    )
    
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=False
    )
    
    return train_loader, val_loader
