#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
包含von_mises应力计算、损失函数等
"""

import torch
import torch.nn as nn


class StressLoss(nn.Module):
    """
    单通道von Mises应力损失函数
    在 asinh 空间计算最佳配置使用的加权误差四次方均值。
    L = vonmises_weight * mean(w * diff_asinh**4)。
    a 为输入通道 12 的界面特征 exp(-d)，d 为最近材料界面的距离；
    b 为处理后平面梯度通道的幅值。
    w = (1 + a) * (1 + grad_weight * b) * label_weight。
    loss_boundary/loss_gradient 是权重诊断量，不是另外相加的损失项。
    """
    def __init__(
        self,
        vonmises_weight=1.0,
        stress_median=1.0,
        grad_weight=50.0,
        label_weight_base=1.0,
        label_weight_alpha=1.0,
        label_weight_gamma=1.0,
        label_weight_clip_c=5.0,
    ):
        super().__init__()
        self.vonmises_weight = vonmises_weight
        self.stress_median = float(stress_median) if stress_median > 0 else 1.0
        self.grad_weight = grad_weight
        self.label_weight_base = float(label_weight_base) if label_weight_base > 0 else 1.0
        self.label_weight_alpha = float(label_weight_alpha)
        self.label_weight_gamma = float(label_weight_gamma)
        self.label_weight_clip_c = float(label_weight_clip_c) if label_weight_clip_c > 0 else 5.0
        
    def forward(self, pred_stress, true_stress, input_data):
        """
        Args:
            pred_stress: (B, 1, D, H, W) - asinh 空间预测值
            true_stress: (B, 1, D, H, W) - asinh 空间目标值
            input_data: (B, 15, D, H, W) - 输入数据
        
        Returns:
            (loss, diagnostics): 标量损失和包含各项数值的字典
        """
        diff = pred_stress - true_stress
        
        # Model input channel 12 already contains D_int = exp(-d), not raw d.
        # Channel 11 is the interface mask; no additional distance transform here.
        a = input_data[:, 12:13, ...]
        grad_x = input_data[:, 13:14, ...]
        grad_y = input_data[:, 14:15, ...]
        b = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)
        
        weight_base = (1.0 + a) * (1.0 + self.grad_weight * b)

        true_real = torch.sinh(true_stress) * self.stress_median
        label_norm = torch.clamp(true_real / (self.label_weight_base + 1e-8), min=0.0, max=self.label_weight_clip_c)
        label_weight = 1.0 + self.label_weight_alpha * torch.pow(label_norm, self.label_weight_gamma)

        weight = weight_base * label_weight
        
        loss_asinh = (weight * (diff ** 4)).mean()
        
        loss_boundary = a.mean()
        loss_gradient = (self.grad_weight * b).mean()

        total_loss = self.vonmises_weight * loss_asinh

        return total_loss, {
            'loss_asinh': loss_asinh.item(),
            'loss_boundary': loss_boundary.item(),
            'loss_gradient': loss_gradient.item(),
            'total_loss': total_loss.item()
        }


def save_r2_scatter(true_values, pred_values, out_path, r2=None, nrmse=None, dpi=150, title=None):
    """
    保存R2散点图

    Args:
        true_values: 1D数组/张量，真实值
        pred_values: 1D数组/张量，预测值
        out_path: 输出图片路径
        r2: 可选R2值
        nrmse: 可选NRMSE值
        dpi: 图片DPI
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if torch.is_tensor(true_values):
        true_values = true_values.detach().cpu().numpy()
    if torch.is_tensor(pred_values):
        pred_values = pred_values.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.scatter(true_values, pred_values, s=4, alpha=0.25, edgecolors='none')

    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linewidth=1, linestyle='--')

    ax.set_xlabel('True')
    ax.set_ylabel('Pred')
    ax.set_title(title or 'R2 Scatter')
    ax.set_aspect('equal', 'box')

    text_lines = []
    if r2 is not None:
        text_lines.append(f"R2: {r2:.6f}")
    if nrmse is not None:
        text_lines.append(f"NRMSE: {nrmse:.6f}")
    if text_lines:
        ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
