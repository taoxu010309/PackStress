# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    channels_last normalizes the last dimension. The channels_first branch uses
    five-dimensional (B, C, D, H, W) tensors and broadcasts affine parameters over D/H/W.
    """
    def __init__(self, normalized_shape, eps=1e-8, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.LeakyReLU(0.1, inplace=True)

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.LeakyReLU(0.1, inplace=True)

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.LeakyReLU(0.1, inplace=True)

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 3, 3, 2], dims=[64, 128, 256, 512],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem followed by three downsampling convolutions
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=2, stride=2),
              )
        self.downsample_layers.append(stem)
        # 三次下采样：前两次缩小 D/H/W，最后一次仅缩小 H/W。
        for i in range(3):
            if i >= 2:
                downsample_layer = nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        
        # BiMamba-v3 的序列分组数，保留原算法设置。
        # 不是 stage 输出的深度；D=40 时各 stage 深度为 20、10、5、5。
        num_slices_list = [40, 20, 10, 5]
        
        cur = 0
        # 为四个编码 stage 创建模块。
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for _ in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        # 为四个编码 stage 创建模块。
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        # 依次执行四个编码 stage。
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class RegressionOutBlock(nn.Module):
    """
    热应力回归任务的输出块
    """
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        
        # 三层卷积依次缩减通道数，再映射到回归输出。
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(in_channels // 2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(in_channels // 4)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv3 = nn.Conv3d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm3d(in_channels // 8)
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)
        
        # 最终回归输出层 - 无激活函数（回归任务）
        self.final_conv = nn.Conv3d(in_channels // 8, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        
        x = self.final_conv(x)
        
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=16,  # 保留类默认值；公共入口显式传入配置 IN_CHANS=15
        out_chans=1,  # 1维von Mises应力
        depths=[2, 2, 2, 2],
        feat_size=[64, 128, 256, 512],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        
        # Mamba编码器
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        
        # 四个 UNet skip 编码块，另有最深处的 hidden 编码块。
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,  
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 最深处再升到 hidden_size（如果 hidden_size 与 feat_size[3] 不同）
        self.encoder_hidden = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block,
        )

        # UNet解码器部分
        # decoder5: enc_hidden(5,32) → enc4(5,64)，upsample (1,2,2) 表示Z轴不上采样，XY轴2倍上采样
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=(1, 2, 2),  # Z轴1倍，XY轴2倍
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=(2, 2, 2),  # 全部2倍上采样
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=(2, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=(2, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # 热应力回归输出块
        self.out = RegressionOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.feat_size[0], 
            out_channels=self.out_chans
        )

    '''def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x'''

    def forward(self, x_in):
        # 检查张量维数和通道数；空间尺寸的下采样/序列分组约束见 README.md。
        assert x_in.dim() == 5, f"Expected 5D input (B,C,D,H,W), got {x_in.dim()}D with shape {x_in.shape}"
        assert x_in.size(1) == self.in_chans, f"Expected {self.in_chans} input channels, got {x_in.size(1)}"
        
        # 下列形状示例采用 configs/model.yaml 默认值和 (40,512,512) patch。
        # MambaEncoder: 输出4个尺度的特征
        # outs[0]: (B,32,20,256,256)  Stage0输出
        # outs[1]: (B,64,10,128,128)  Stage1输出
        # outs[2]: (B,128,5,64,64)    Stage2输出
        # outs[3]: (B,256,5,32,32)    Stage3输出
        outs = self.vit(x_in)
        
        # UNet Encoder (UnetrBasicBlock)
        # enc1: (B,15,40,512,512) -> (B,32,40,512,512)
        enc1 = self.encoder1(x_in)
        # enc2: (B,32,20,256,256) -> (B,64,20,256,256) 
        enc2 = self.encoder2(outs[0])
        # enc3: (B,64,10,128,128) -> (B,128,10,128,128)
        enc3 = self.encoder3(outs[1])
        # enc4: (B,128,5,64,64) -> (B,256,5,64,64)
        enc4 = self.encoder4(outs[2])
        # enc_hidden: (B,256,5,32,32) -> (B,768,5,32,32)  通道扩展到hidden_size
        enc_hidden = self.encoder_hidden(outs[3])
        
        # UNet Decoder (UnetrUpBlock) - 带Skip Connections
        # dec5: (B,768,5,32,32) + enc4(B,256,5,64,64) -> (B,256,5,64,64)
        dec5 = self.decoder5(enc_hidden, enc4)
        # dec4: (B,256,5,64,64) + enc3(B,128,10,128,128) -> (B,128,10,128,128)
        dec4 = self.decoder4(dec5, enc3)
        # dec3: (B,128,10,128,128) + enc2(B,64,20,256,256) -> (B,64,20,256,256)
        dec3 = self.decoder3(dec4, enc2)
        # dec2: (B,64,20,256,256) + enc1(B,32,40,512,512) -> (B,32,40,512,512)
        dec2 = self.decoder2(dec3, enc1)
        # dec1: (B,32,40,512,512) -> (B,32,40,512,512)
        dec1 = self.decoder1(dec2)
        
        # RegressionOutBlock；此处没有额外的 D 维插值。
        # (B,32,40,512,512) -> (B,1,40,512,512)
        final_out = self.out(dec1)
        
        return final_out

