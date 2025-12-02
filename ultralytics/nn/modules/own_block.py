import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .block import C3k2
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

import einops
from einops import rearrange

import math
import numpy as np



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

__all__= (
    "DepthwiseSeparableConv",
    "WaveletDownsampleWrapper",
    "PSD",
    "GGMix",
    "GatedABlock",
    "GatedA2C2f",
    "AdaptiveGatedC3k2",
)

#region WaveletDownsampleWrapper
class HaarWavelet(nn.Module):
    # 使用例：- [-1, 1, WaveletDownsampleWrapper, [256]]
    def __init__(self, in_channels, grad=True):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels
        self.grad = grad

    def forward(self, x, rev=False):
        if not rev:
            # 正向小波变换
            out = x.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = out.permute(0, 2, 1, 3, 4).reshape([x.shape[0], 4 * self.in_channels, x.shape[2] // 2, x.shape[3] // 2])
        else:
            # 逆向小波变换
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = out.permute(0, 2, 1, 3, 4).reshape([x.shape[0], self.in_channels, x.shape[2] * 2, x.shape[3] * 2])
        return out

class WFD(nn.Module):
    def __init__(self, dim_in, dim, need=False):
        super(WFD, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1, padding=0)
            self.HaarWavelet = HaarWavelet(dim, grad=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False)
            self.dim = dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)
        
        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)

        return a, h+v+d

class WaveletDownsampleWrapper(nn.Module):
    def __init__(self, in_channels, use_conv=True):
        super().__init__()
        k=3
        d=2
        self.wfd = WFD(dim_in=in_channels, dim=in_channels, need=use_conv)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, k, stride=1, padding=((k//2)*d), groups=in_channels, dilation=d))      
    def forward(self, x):
        a, hvd = self.wfd(x)
        B, C, H, W = a.shape
        a_flat = a.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        attn_out, _ = self.attn(a_flat, a_flat, a_flat)  # [B, H*W, C]
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        conv_out = self.conv(hvd)  # [B, C, H, W]

        return attn_out + conv_out  # [B, C, H, W]
    
def autopad(k, p=None):
    """
    Automatically set padding = k//2 if not specified, to preserve spatial dims.
    """
    return k // 2 if p is None else p
#endregion

class PSD(nn.Module):
    # 使用例：- [-1, 1, PSD,  [256, 0.5]] 
    """
    Phase-split downsampling (PSD) block:
      1) 1x1 conv to reduce channels to c = int(c2 * e)
      2) depthwise conv 3x3
      3) spatial downsample by splitting and concatenating 2x2 grids
      4) 1x1 conv to expand back to c2 channels

    Args:
        c1 (int): input channels
        c2 (int): output channels
        e (float): expansion ratio for intermediate channels
    """
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        # intermediate channels
        self.c = int(c2 * e)

        # reduce channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )

        # depthwise conv
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                self.c, self.c,
                kernel_size=3, stride=1,
                padding=autopad(3),
                groups=self.c,
                bias=False
            ),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )

        # expand + merge
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2)
            # no activation by default; add nn.SiLU() if desired
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1x1 -> depthwise
        x = self.dwconv(self.cv1(x))

        # split into 4 sub-samplings
        # top-left, bottom-left, top-right, bottom-right
        x = torch.cat([
            x[..., ::2, ::2],    # even rows, even cols
            x[..., 1::2, ::2],   # odd rows, even cols
            x[..., ::2, 1::2],   # even rows, odd cols
            x[..., 1::2, 1::2],  # odd rows, odd cols
        ], dim=1)

        # final 1x1 conv to c2
        return self.cv2(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bottleneck_ratio=0.5, groups=4):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param padding: 填充
        :param bottleneck_ratio: 用于瓶颈层的通道缩减比例
        :param groups: 逐点卷积的分组数
        """
        super().__init__()
        bottleneck_channels = int(in_channels * bottleneck_ratio)
        
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        
        # 瓶颈层降维
        self.bottleneck = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        
        # 分组逐点卷积
        self.pointwise = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bottleneck(x)
        x = self.pointwise(x)
        return x
    
#region gatemethod
class GatedABlock(nn.Module):
    """
    Gated Area-attention block combining area attention with gated mechanism.
    
    This module integrates:
    - Area-based attention for spatial feature processing
    - Gated mechanism for dynamic feature selection
    - Residual connections for stable training
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio for the FFN
        area (int): Number of areas for spatial division
        gate_ratio (float): Not used in current implementation (kept for compatibility)
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, area=1, gate_ratio=0.5):
        super().__init__()
        
        # Area attention component (from A2C2f)
        self.area_attn = self._create_area_attention(dim, num_heads, area)
        
        # Gated mechanism component - simplified approach
        self.gate_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()  # Direct gating signal
        )
        
        # Gated FFN - processes the gated input
        self.gated_ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1, bias=False),
            nn.BatchNorm2d(int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        # Learnable mixing weight
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
    def _create_area_attention(self, dim, num_heads, area):
        """Create area attention module similar to AAttn"""
        head_dim = dim // num_heads
        all_head_dim = head_dim * num_heads
        
        return nn.ModuleDict({
            'qkv': Conv(dim, all_head_dim * 3, 1, act=False),
            'proj': Conv(all_head_dim, dim, 1, act=False),
            'pe': Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)
        })
    
    def forward(self, x):
        """Forward pass with gated area attention"""
        shortcut = x
        B, C, H, W = x.shape
        
        # Area attention path
        attn_out = self._forward_area_attention(x)
        
        # Gated mechanism path
        gate = self.gate_proj(x)  # Generate gating signal [B, C, H, W]
        gated_input = x * gate  # Apply gating to input
        gated_out = self.gated_ffn(gated_input)
        
        # Adaptive mixing of attention and gated features
        mixed_out = self.alpha * attn_out + (1 - self.alpha) * gated_out
        
        return shortcut + mixed_out
    
    def _forward_area_attention(self, x):
        """Simplified area attention forward pass"""
        B, C, H, W = x.shape
        N = H * W
        
        # Generate Q, K, V through convolution
        qkv = self.area_attn['qkv'](x).flatten(2).transpose(1, 2)  # [B, N, 3C]
        
        # Simplified attention computation
        q, k, v = qkv.chunk(3, dim=-1)
        scale = (C // 8) ** -0.5  # Simplified scaling
        
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.area_attn['proj'](out)
        
        return out


class GatedA2C2f(nn.Module):
    """
    Gated Area-Attention C2f module that combines:
    - A2C2f's area attention mechanism
    - Gated control for dynamic feature selection
    - Backward compatibility with existing YOLO architectures
    
    This module can replace A2C2f in YOLO12 or C3k2 in YOLO11 while
    providing enhanced feature processing capabilities.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of GatedABlock layers
        shortcut (bool): Whether to use shortcut connections
        g (int): Groups for convolutions
        e (float): Channel expansion ratio
        gate_ratio (float): Ratio for gated mechanism (auto-configured in tasks.py)
        area (int): Area division for attention (auto-configured in tasks.py)
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, gate_ratio=0.5, area=1, **kwargs):
        super().__init__()
        
        self.c = int(c2 * e)  # hidden channels
        
        # Ensure dimension compatibility for attention heads - make it divisible by 32
        if self.c % 32 != 0:
            self.c = ((self.c + 31) // 32) * 32  # Round up to nearest multiple of 32
        
        # Input/output projections
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((1 + n) * self.c, c2, 1)
        
        # Gated area attention blocks
        self.m = nn.ModuleList([
            GatedABlock(
                dim=self.c,
                num_heads=self.c // 32,  # Ensure proper head count
                area=area,
                gate_ratio=gate_ratio
            ) for _ in range(n)
        ])
        
        # Learnable residual scaling (similar to A2C2f)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if shortcut else None
        
    def forward(self, x):
        """Forward pass through gated area attention layers"""
        shortcut = x
        
        # Process through gated blocks
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        
        # Concatenate and project
        out = self.cv2(torch.cat(y, 1))
        
        # Apply learnable residual scaling if enabled
        if self.gamma is not None:
            out = shortcut + self.gamma.view(-1, self.gamma.size(0), 1, 1) * out
            
        return out


class AdaptiveGatedC3k2(nn.Module):
    """
    Adaptive Gated C3k2 that can dynamically choose between:
    - Traditional C3k2 behavior for efficiency
    - Gated mechanism for enhanced feature processing
    - Area attention for spatial awareness
    
    This provides a unified interface that can adapt based on:
    - Input feature resolution
    - Computational budget
    - Task requirements
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels  
        n (int): Number of layers
        shortcut (bool): Whether to use shortcut connections
        g (int): Groups for convolutions
        e (float): Channel expansion ratio
        adaptive_mode (str): 'auto', 'traditional', or 'gated' (auto-configured in tasks.py)
        gate_threshold (float): Threshold for adaptive mode switching (auto-configured in tasks.py)
        area (int): Area division for attention (auto-configured in tasks.py)
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, 
                 adaptive_mode='auto', gate_threshold=0.5, area=1, **kwargs):
        super().__init__()
        
        self.adaptive_mode = adaptive_mode
        self.gate_threshold = gate_threshold
        
        # Traditional C3k2 path - explicitly pass keyword arguments to avoid parameter order issues
        self.traditional_path = C3k2(c1=c1, c2=c2, n=n, c3k=False, e=e, g=g, shortcut=shortcut)
        
        # Gated enhancement path - explicitly pass keyword arguments
        default_gate_ratio = 0.5  # Default gate ratio if not provided
        self.gated_path = GatedA2C2f(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e, 
                                    gate_ratio=default_gate_ratio, area=area)
        
        # Adaptive gate controller
        if adaptive_mode == 'auto':
            self.gate_controller = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c1 // 4, 1),
                nn.ReLU(),
                nn.Conv2d(c1 // 4, 1, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        """Adaptive forward pass"""
        if self.adaptive_mode == 'traditional':
            return self.traditional_path(x)
        elif self.adaptive_mode == 'gated':
            return self.gated_path(x)
        else:  # auto mode
            # Compute gating signal based on input characteristics
            gate_signal = self.gate_controller(x)
            gate_value = gate_signal.mean().item()
            
            if gate_value > self.gate_threshold:
                return self.gated_path(x)
            else:
                return self.traditional_path(x)
#endregion            

#region our GGMix

def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='border',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.data_format = data_format
        
        normalized_dim = len(self.normalized_shape)
        param_shape = self.normalized_shape
        
        # 初始化gamma为1.0，beta为0.0
        self.gamma = nn.Parameter(torch.ones(*param_shape) * 1.0)
        self.beta = nn.Parameter(torch.zeros(*param_shape))
        
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
    
    def forward(self, x):
        if self.data_format == "channels_last":
            # 使用F.layer_norm进行channels_last层归一化
            return F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        
        # 自定义实现channels_first
        channel_dim = 1
        
        # 计算通道维度的均值
        mean = x.mean(dim=channel_dim, keepdim=True)
        
        # 计算通道维度的方差，使用epsilon进行数值稳定性
        var = x.var(dim=channel_dim, keepdim=True, unbiased=False)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        reshaped_gamma = self.gamma.view((-1,) + (1,) * (x.ndim - 2))
        reshaped_beta = self.beta.view((-1,) + (1,) * (x.ndim - 2))
        
        return x_normalized * reshaped_gamma + reshaped_beta



class Global_Guidance(nn.Module):
    def __init__(self, dim, window_size=4, k=4,ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        
        # 输入卷积层
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # 输出偏移量
        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        # 输出mask
        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1)
        )

        # 输出通道注意力
        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        # 输出空间注意力
        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        


    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):
        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)

        ca = self.out_CA(x)
        sa = self.out_SA(x)
        
        x = torch.mean(x, keepdim=True, dim=1) 

        x = rearrange(x,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()

        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]

        return mask, offsets, ca, sa


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, N, C = q.shape
        #由外部输入的q, k, v进行线性变换，并reshape为(B, N, num_heads, head_dim)，然后permute为(num_heads, B, N, head_dim)
        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        k = self.k_proj(k).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        v = self.v_proj(v).reshape(B, N, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

    

class GGMix(nn.Module):
    def __init__(self, c1, c2, window_size=4, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()

        if c1 != c2:
            raise ValueError(f"GGMix requires matching input/output channels, but got c1={c1} and c2={c2}.")

        self.dim = c2
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio
        self.requires_img_ori = True

        k = 3
        d = 2

        self.project_v = nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(self.dim, self.dim, bias=bias)
        self.project_k = nn.Linear(self.dim, self.dim, bias=bias)

        self.multihead_attn = MultiHeadAttention(self.dim)
        self.conv_sptial = DepthwiseSeparableConv(self.dim, self.dim)   
        self.project_out = nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=bias)

        self.act = nn.GELU()
        self.route = Global_Guidance(self.dim, window_size, ratio=ratio)

        # 生成global feature
        self.global_predictor = nn.Sequential(
            nn.Conv2d(3, 8, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(8, self.dim + 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, condition_global=None, mask=None, train_mode=True, img_ori=None):
        if img_ori is None:
            raise ValueError("GGMix requires 'img_ori' input but received None.")

        N, C, H, W = x.shape

        global_status = self.global_predictor(img_ori)
        global_status = F.interpolate(global_status, size=(H, W), mode='bilinear', align_corners=False)

        if self.is_deformable:
            patch_status = torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.window_size), torch.linspace(-1, 1, self.window_size)))\
                .type_as(x).unsqueeze(0).repeat(N, 1, H // self.window_size, W // self.window_size)
            
            global_feature = torch.cat([global_status, patch_status], dim=1)
            
        mask, offsets, ca, sa = self.route(global_feature, ratio=self.ratio, train_mode=train_mode)

        q = x
        k = x + flow_warp(x, offsets.permute(0, 2, 3, 1))
        qk = torch.cat([q, k], dim=1)
        v = self.project_v(x)

        vs = v * sa

        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, 'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        N_ = v.shape[1]
        v1, v2 = v * mask, vs * (1 - mask)
        qk1 = qk * mask
    

        v1 = rearrange(v1, 'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, 'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, 'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        # 使用多头自注意力机制
        f_attn = self.multihead_attn(q1, k1, v1)

        f_attn = rearrange(f_attn, '(b n) (dh dw) c -> b n (dh dw c)',
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        
        attn_out = f_attn + v2

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)',
            h=H // self.window_size, w=W // self.window_size, dh=self.window_size, dw=self.window_size
        )

        out = attn_out
        out = self.act(self.conv_sptial(out)) * ca + out
        out = self.project_out(out)

        return out
#endregion
