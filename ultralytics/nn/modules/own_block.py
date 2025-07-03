import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .block import C3k2
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from einops import rearrange


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
    "CED",
    "GatedABlock",
    "GatedA2C2f",
    "AdaptiveGatedC3k2",
)


class HaarWavelet(nn.Module):
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


class CED(nn.Module):
    """
    Channel Expansion + Depthwise (CED) block:
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
        gate_ratio (float): Ratio for gated mechanism
        area (int): Area division for attention
        e (float): Channel expansion ratio
        shortcut (bool): Whether to use shortcut connections
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
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, 
                 adaptive_mode='auto', gate_threshold=0.5, area=1, **kwargs):
        super().__init__()
        
        self.adaptive_mode = adaptive_mode
        self.gate_threshold = gate_threshold
        
        # Traditional C3k2 path - match C3k2's parameter order: (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)
        self.traditional_path = C3k2(c1, c2, n, c3k=False, e=e, g=g, shortcut=shortcut, **kwargs)
        
        # Gated enhancement path
        self.gated_path = GatedA2C2f(c1, c2, n, shortcut, g, e, area=area, **kwargs)
        
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